"""Logic tests for skills/apple_music_intel.py.

This skill tracks listening history, learns taste patterns, and remembers
session skips for Apple Music / iTunes. Tests cover the deterministic core:
  • time-of-day bucketing + track-key normalisation,
  • vibe-slot parsing ('friday night', 'now', aliases, bare),
  • the taste aggregation pass (by_slot / by_artist / skip_rate_by_genre /
    overall skip rate) over a synthesized history,
  • _top_artist_for_slot incl. session-skip suppression + MIN_LISTENS gate,
  • web-window-title parsing for Apple Music / Spotify,
  • session skip persistence + 6-hour expiry,
  • the music_history / music_taste / music_aggregate / play_vibe / skip_track
    actions with the data dir redirected to a temp dir.

All jsonl/json files are redirected to a temp dir (the module exposes its
paths as globals), so nothing under data/ is touched. iTunes COM / the
background poll loop are never invoked.
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class AppleMusicPureTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("apple_music_intel")

    # ── _time_of_day ─────────────────────────────────────────────────────
    def test_time_of_day_buckets(self):
        t = self.mod._time_of_day
        self.assertEqual(t(8), "morning")
        self.assertEqual(t(14), "afternoon")
        self.assertEqual(t(19), "evening")
        self.assertEqual(t(23), "night")
        self.assertEqual(t(3), "night")     # wraps past midnight

    def test_track_key_normalises(self):
        self.assertEqual(self.mod._track_key(" Michael Jackson ", "Beat It"),
                         "michael jackson|beat it")

    # ── _parse_vibe_slot ─────────────────────────────────────────────────
    def test_parse_vibe_slot_explicit(self):
        self.assertEqual(self.mod._parse_vibe_slot("friday night"), ("friday", "night"))

    def test_parse_vibe_slot_aliases(self):
        self.assertEqual(self.mod._parse_vibe_slot("fri evening"), ("friday", "evening"))
        # 'tonight' maps to night; 'sun' → sunday.
        self.assertEqual(self.mod._parse_vibe_slot("sun tonight"), ("sunday", "night"))

    def test_parse_vibe_slot_now_uses_clock(self):
        day, part = self.mod._parse_vibe_slot("now")
        self.assertIn(day, self.mod._DAYS)
        self.assertIn(part, {"morning", "afternoon", "evening", "night"})

    def test_parse_vibe_slot_bare_fills_both(self):
        day, part = self.mod._parse_vibe_slot("")
        self.assertIn(day, self.mod._DAYS)
        self.assertIn(part, {"morning", "afternoon", "evening", "night"})

    def test_parse_vibe_slot_partial_day_only(self):
        # Only a day given → part defaults from the clock.
        day, part = self.mod._parse_vibe_slot("monday")
        self.assertEqual(day, "monday")
        self.assertIn(part, {"morning", "afternoon", "evening", "night"})

    # ── web window-title parsing ─────────────────────────────────────────
    def test_apple_title_pattern_em_dash(self):
        m = self.mod._APPLE_TITLE_PATTERNS[0].match("Holocene — Bon Iver – Apple Music")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("title").strip(), "Holocene")
        self.assertEqual(m.group("artist").strip(), "Bon Iver")

    def test_spotify_title_pattern_middle_dot(self):
        m = self.mod._SPOTIFY_TITLE_PATTERNS[0].match("Beat It · Michael Jackson - Spotify")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("title").strip(), "Beat It")
        self.assertEqual(m.group("artist").strip(), "Michael Jackson")


class AppleMusicDataTests(unittest.TestCase):
    """Tests that read/write the data files — redirect every path global to a
    temp dir so nothing under data/ is touched."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("apple_music_intel")
        self.tmp = tempfile.mkdtemp(prefix="amintel_test_")
        self.addCleanup(self._cleanup)
        self.mod._DATA_DIR = self.tmp
        self.mod._HISTORY_FILE = os.path.join(self.tmp, "history.jsonl")
        self.mod._SKIPS_FILE = os.path.join(self.tmp, "skips.jsonl")
        self.mod._SESSION_FILE = os.path.join(self.tmp, "session.json")
        self.mod._TASTE_FILE = os.path.join(self.tmp, "taste.json")

    def _cleanup(self):
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def _seed_history(self):
        """Friday-evening Michael Jackson ×3, plus skipped Country tracks."""
        events = []
        for k in range(3):
            events.append({"ts": 1000 + k, "iso": f"2026-05-29T21:0{k}:00",
                           "date": "2026-05-29", "day": "friday", "hour": 21,
                           "part": "evening", "artist": "Michael Jackson",
                           "title": f"Track {k}", "album": "X", "genre": "Pop",
                           "source": "itunes", "secs": 200, "complete": True})
        for k in range(4):
            events.append({"ts": 2000 + k, "iso": f"2026-05-2{k}T15:00:00",
                           "date": "2026-05-20", "day": "tuesday", "hour": 15,
                           "part": "afternoon", "artist": "Some Country Act",
                           "title": f"Nope {k}", "album": "Y", "genre": "Country",
                           "source": "itunes", "secs": 8, "complete": False})
        for e in events:
            self.mod._append_jsonl(self.mod._HISTORY_FILE, e)

    # ── aggregate ────────────────────────────────────────────────────────
    def test_aggregate_builds_slot_and_artist_tables(self):
        self._seed_history()
        snap = self.mod.aggregate()
        self.assertEqual(snap["events"], 7)
        self.assertEqual(snap["by_slot"]["friday|evening"]["Michael Jackson"], 3)
        self.assertEqual(snap["by_artist"]["Michael Jackson"]["count"], 3)

    def test_aggregate_skip_rate_by_genre(self):
        self._seed_history()
        snap = self.mod.aggregate()
        # All 4 Country plays are skips → skip rate 1.0; Pop has 0 skips and
        # only 3 plays so it sits at 0.0 (>=3 sample threshold).
        self.assertEqual(snap["skip_rate_by_genre"]["Country"], 1.0)
        self.assertEqual(snap["skip_rate_by_genre"].get("Pop"), 0.0)

    def test_aggregate_overall_skip_rate(self):
        self._seed_history()
        snap = self.mod.aggregate()
        # 4 skips of 7 observations ≈ 0.571.
        self.assertAlmostEqual(snap["skip_rate_overall"], round(4 / 7, 3))

    def test_aggregate_empty_history(self):
        snap = self.mod.aggregate()
        self.assertEqual(snap["events"], 0)
        self.assertEqual(snap["by_artist"], {})

    # ── _top_artist_for_slot ─────────────────────────────────────────────
    def test_top_artist_for_slot(self):
        self._seed_history()
        self.mod.aggregate()
        top = self.mod._top_artist_for_slot("friday", "evening")
        self.assertEqual(top, ("Michael Jackson", 3))

    def test_top_artist_for_slot_below_min_listens(self):
        # One listen only → below MIN_LISTENS_FOR_VIBE (2) → None.
        self.mod._append_jsonl(self.mod._HISTORY_FILE, {
            "ts": 1, "iso": "2026-05-29T21:00:00", "date": "2026-05-29",
            "day": "saturday", "hour": 21, "part": "evening", "artist": "Solo Act",
            "title": "Once", "album": "Z", "genre": "Rock", "source": "itunes",
            "secs": 200, "complete": True})
        self.mod.aggregate()
        self.assertIsNone(self.mod._top_artist_for_slot("saturday", "evening"))

    def test_top_artist_skips_session_skipped_artist(self):
        self._seed_history()
        self.mod.aggregate()
        # Mark every Michael Jackson track as session-skipped.
        for k in range(3):
            self.mod._session_record_skip("Michael Jackson", f"Track {k}")
        self.assertIsNone(self.mod._top_artist_for_slot("friday", "evening"))

    # ── session skip persistence ─────────────────────────────────────────
    def test_session_skip_roundtrip(self):
        self.mod._session_record_skip("Artist A", "Song A")
        self.assertIn("artist a|song a", self.mod._session_skipped_keys())

    def test_session_expires_after_6_hours(self):
        # Write a session that started 7 hours ago → _load_session resets it.
        self.mod._save_session({"session_start": time.time() - 7 * 3600,
                                "skipped_keys": ["stale|key"]})
        self.assertNotIn("stale|key", self.mod._session_skipped_keys())

    # ── actions ──────────────────────────────────────────────────────────
    def test_music_history_empty(self):
        self.assertIn("No listening history", self.actions["music_history"](""))

    def test_music_history_lists_recent(self):
        self._seed_history()
        out = self.actions["music_history"]("")
        self.assertIn("Michael Jackson", out)
        self.assertIn("(skipped)", out)   # the Country tracks were skips

    def test_music_taste_no_data(self):
        self.assertIn("No taste data yet", self.actions["music_taste"](""))

    def test_music_taste_reports_top_artists(self):
        self._seed_history()
        self.mod.aggregate()
        out = self.actions["music_taste"]("")
        self.assertIn("Michael Jackson", out)
        self.assertIn("skip rate", out.lower())

    def test_music_aggregate_action(self):
        self._seed_history()
        out = self.actions["music_aggregate"]("")
        self.assertIn("7 listen events", out)

    def test_play_vibe_no_pattern(self):
        # No history → no slot pattern → polite refusal mentioning the slot.
        out = self.actions["play_vibe"]("friday night")
        self.assertIn("friday night", out)
        self.assertIn("don't have a strong", out)

    def test_play_vibe_reaches_player(self):
        self._seed_history()
        self.mod.aggregate()
        bc = mock.MagicMock()
        bc._play_music_core.return_value = (True, "now playing Michael Jackson")
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            out = self.actions["play_vibe"]("friday evening")
        self.assertIn("Vibing your usual friday evening", out)

    def test_skip_track_records_and_skips(self):
        # No current track; sampling returns one. media_next is the fallback.
        bc = mock.MagicMock()
        del bc._act_next_song  # force the media-key fallback branch
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_sample_now_playing",
                               return_value={"artist": "Foo", "title": "Bar",
                                             "source": "web_apple"}):
            out = self.actions["skip_track"]("")
        self.assertIn("set 'Bar' by Foo aside", out)
        self.assertIn("foo|bar", self.mod._session_skipped_keys())


if __name__ == "__main__":
    unittest.main()
