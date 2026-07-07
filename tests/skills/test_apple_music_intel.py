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
  • the jsonl/json IO helpers incl. rotation + every failure branch,
  • the iTunes-COM sampler and the window-title sampler (both driven by
    injected fakes — the real win32com/pygetwindow deps are NOT on CI),
  • the _listen_loop / _aggregator_loop background bodies (single-iteration,
    no real sleeping/threads),
  • the play_unheard / play_vibe / skip_track / music_history / music_taste /
    music_aggregate actions, including iTunes-absent, COM-error, empty-library
    and malformed-data edge paths,
  • register()'s action wiring + session reset (threads neutered).

All jsonl/json files are redirected to a temp dir (the module exposes its
paths as globals), so nothing under data/ is touched. iTunes COM, pygetwindow,
pythoncom and the monolith bobert_companion are NEVER imported for real — a
fake is injected wherever the skill reaches for them, installed in setUp and
removed in tearDown so other tests (and real numpy) stay isolated. No real
COM / network / subprocess / thread / sleep ever runs.
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── isolation helper ────────────────────────────────────────────────────
_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules and restore the prior
    state — including absence — on exit. Mirrors the save/restore contract used
    by tests/skills/test_self_diagnostic.py so injected fakes (pygetwindow,
    pythoncom, …) never persist process-wide and the real modules are restored
    for every other test. Pass dotted keys via ``**{"a.b": obj}``; the leaf is
    also set as an attribute on an already-imported parent package."""
    saved_mod: dict = {}
    missing: set = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


# ─── COM / window fakes ──────────────────────────────────────────────────

class _FakeTrack:
    """Minimal iTunes IITTrack stand-in for the now-playing sampler."""
    def __init__(self, name="Song", artist="Artist", album="Album",
                 genre="Genre"):
        self.Name = name
        self.Artist = artist
        self.Album = album
        self.Genre = genre


class _FakeLibTrack:
    """Library track for the play_unheard scan. PlayedDate is a tiny object
    that quacks like a pywintypes datetime (supports .timestamp())."""
    def __init__(self, name, artist, played_ts):
        self.Name = name
        self.Artist = artist
        self._played_ts = played_ts
        self.played = False

    @property
    def PlayedDate(self):
        if self._played_ts is None:
            return None
        return _FakeDateTime(self._played_ts)

    def Play(self):
        self.played = True


class _FakeDateTime:
    def __init__(self, ts):
        self._ts = ts

    def timestamp(self):
        return float(self._ts)

    def timetuple(self):
        return time.localtime(self._ts)


class _FakeTrackCollection:
    """1-based IITTrackCollection."""
    def __init__(self, tracks):
        self._tracks = tracks

    @property
    def Count(self):
        return len(self._tracks)

    def Item(self, i):
        return self._tracks[i - 1]   # iTunes COM is 1-based


class _FakeLibraryPlaylist:
    def __init__(self, tracks):
        self.Tracks = _FakeTrackCollection(tracks)


class _FakeITunesApp:
    """iTunes.Application stand-in covering CurrentTrack + LibraryPlaylist."""
    def __init__(self, current=None, player_state=1, lib_tracks=None):
        self.CurrentTrack = current
        self.PlayerState = player_state
        self.LibraryPlaylist = _FakeLibraryPlaylist(lib_tracks or [])


class _FakeWindow:
    def __init__(self, title):
        self.title = title


def _fake_pygetwindow(titles):
    """A pygetwindow module exposing getAllWindows() -> [_FakeWindow,...]."""
    mod = types.ModuleType("pygetwindow")
    mod.getAllWindows = lambda: [_FakeWindow(t) for t in titles]
    return mod


def _fake_pythoncom():
    """pythoncom stub — apple_music_intel only does `import pythoncom` then
    never calls into it (COM init already happened inside the bridge)."""
    return types.ModuleType("pythoncom")


# ─── pure helpers (no disk) ──────────────────────────────────────────────

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

    def test_time_of_day_negative_hour_clamped(self):
        # Defensive guard: a negative hour is clamped to 0 → morning bucket
        # boundary falls to 'night' (0 < 6).
        self.assertEqual(self.mod._time_of_day(-5), "night")

    def test_track_key_normalises(self):
        self.assertEqual(self.mod._track_key(" Michael Jackson ", "Beat It"),
                         "michael jackson|beat it")

    def test_norm_handles_none(self):
        self.assertEqual(self.mod._norm(None), "")
        self.assertEqual(self.mod._norm("  HeLLo  "), "hello")

    def test_slot_key(self):
        self.assertEqual(self.mod._slot_key("friday", "evening"), "friday|evening")

    def test_bobert_resolves_from_sys_modules(self):
        # _bobert() returns whatever __main__ / bobert_companion is registered.
        fake_main = types.ModuleType("__main__")
        with mock.patch.dict(sys.modules, {"__main__": fake_main}):
            self.assertIs(self.mod._bobert(), fake_main)

    def test_bobert_falls_back_to_bobert_companion(self):
        fake_bc = types.ModuleType("bobert_companion")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            sys.modules.pop("__main__", None)
            self.assertIs(self.mod._bobert(), fake_bc)

    # ── _parse_vibe_slot ─────────────────────────────────────────────────
    def test_parse_vibe_slot_explicit(self):
        self.assertEqual(self.mod._parse_vibe_slot("friday night"), ("friday", "night"))

    def test_parse_vibe_slot_aliases(self):
        self.assertEqual(self.mod._parse_vibe_slot("fri evening"), ("friday", "evening"))
        # 'tonight' maps to night; 'sun' → sunday.
        self.assertEqual(self.mod._parse_vibe_slot("sun tonight"), ("sunday", "night"))

    def test_parse_vibe_slot_more_aliases(self):
        # 'noon'/'lunch'/'midday' → afternoon; 'breakfast' → morning.
        self.assertEqual(self.mod._parse_vibe_slot("wed noon"), ("wednesday", "afternoon"))
        self.assertEqual(self.mod._parse_vibe_slot("tues breakfast"),
                         ("tuesday", "morning"))

    def test_parse_vibe_slot_now_uses_clock(self):
        day, part = self.mod._parse_vibe_slot("now")
        self.assertIn(day, self.mod._DAYS)
        self.assertIn(part, {"morning", "afternoon", "evening", "night"})

    def test_parse_vibe_slot_current_and_today(self):
        for word in ("current", "today"):
            day, part = self.mod._parse_vibe_slot(word)
            self.assertIn(day, self.mod._DAYS)
            self.assertIn(part, {"morning", "afternoon", "evening", "night"})

    def test_parse_vibe_slot_bare_fills_both(self):
        day, part = self.mod._parse_vibe_slot("")
        self.assertIn(day, self.mod._DAYS)
        self.assertIn(part, {"morning", "afternoon", "evening", "night"})

    def test_parse_vibe_slot_partial_day_only(self):
        # Only a day given → part defaults from the clock (599-600).
        day, part = self.mod._parse_vibe_slot("monday")
        self.assertEqual(day, "monday")
        self.assertIn(part, {"morning", "afternoon", "evening", "night"})

    def test_parse_vibe_slot_part_only_day_from_clock(self):
        # Only a part given → day defaults from the clock.
        day, part = self.mod._parse_vibe_slot("evening")
        self.assertIn(day, self.mod._DAYS)
        self.assertEqual(part, "evening")

    # ── web window-title parsing ─────────────────────────────────────────
    def test_apple_title_pattern_em_dash(self):
        m = self.mod._APPLE_TITLE_PATTERNS[0].match("Holocene — Bon Iver – Apple Music")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("title").strip(), "Holocene")
        self.assertEqual(m.group("artist").strip(), "Bon Iver")

    def test_apple_title_pattern_pipe(self):
        m = self.mod._APPLE_TITLE_PATTERNS[1].match("Holocene - Bon Iver | Apple Music")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("artist").strip(), "Bon Iver")

    def test_spotify_title_pattern_middle_dot(self):
        m = self.mod._SPOTIFY_TITLE_PATTERNS[0].match("Beat It · Michael Jackson - Spotify")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("title").strip(), "Beat It")
        self.assertEqual(m.group("artist").strip(), "Michael Jackson")


# ─── now-playing samplers (COM + window title) ───────────────────────────

class SamplerTests(unittest.TestCase):
    """_sample_itunes / _sample_window_title / _sample_now_playing — driven by
    injected fakes for itunes_bridge, pygetwindow and the monolith. The real
    win32com/pygetwindow are not assumed present (CI-faithful)."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("apple_music_intel")
        # Swap the bound itunes_bridge for a fake so no real COM/psutil is hit.
        self._real_bridge = self.mod.itunes_bridge
        self.bridge = mock.MagicMock(name="itunes_bridge")
        self.mod.itunes_bridge = self.bridge
        self.addCleanup(self._restore_bridge)

    def _restore_bridge(self):
        self.mod.itunes_bridge = self._real_bridge

    # ── _sample_itunes ───────────────────────────────────────────────────
    def test_sample_itunes_not_running(self):
        self.bridge.is_running.return_value = False
        self.assertIsNone(self.mod._sample_itunes())
        self.bridge.get_client.assert_not_called()

    def test_sample_itunes_client_none(self):
        self.bridge.is_running.return_value = True
        self.bridge.get_client.return_value = (None, "no client")
        self.assertIsNone(self.mod._sample_itunes())

    def test_sample_itunes_stopped_player(self):
        self.bridge.is_running.return_value = True
        app = _FakeITunesApp(current=_FakeTrack(), player_state=0)
        self.bridge.get_client.return_value = (app, None)
        self.assertIsNone(self.mod._sample_itunes())

    def test_sample_itunes_no_current_track(self):
        self.bridge.is_running.return_value = True
        app = _FakeITunesApp(current=None, player_state=1)
        self.bridge.get_client.return_value = (app, None)
        self.assertIsNone(self.mod._sample_itunes())

    def test_sample_itunes_playing(self):
        self.bridge.is_running.return_value = True
        track = _FakeTrack("Beat It", "Michael Jackson", "Thriller", "Pop")
        app = _FakeITunesApp(current=track, player_state=1)
        self.bridge.get_client.return_value = (app, None)
        r = self.mod._sample_itunes()
        self.assertEqual(r, {"artist": "Michael Jackson", "title": "Beat It",
                             "album": "Thriller", "genre": "Pop",
                             "source": "itunes"})

    def test_sample_itunes_paused_still_counts(self):
        # PlayerState 2 = paused — the skill deliberately still reports it.
        self.bridge.is_running.return_value = True
        track = _FakeTrack("Song", "Artist")
        app = _FakeITunesApp(current=track, player_state=2)
        self.bridge.get_client.return_value = (app, None)
        self.assertIsNotNone(self.mod._sample_itunes())

    def test_sample_itunes_com_error_returns_none(self):
        # Touching CurrentTrack raises (a COM error) → None, not an exception.
        self.bridge.is_running.return_value = True
        app = mock.MagicMock()
        app.PlayerState = 1
        type(app).CurrentTrack = mock.PropertyMock(side_effect=Exception("COM boom"))
        self.bridge.get_client.return_value = (app, None)
        self.assertIsNone(self.mod._sample_itunes())

    # ── _sample_window_title ─────────────────────────────────────────────
    def test_window_title_pygetwindow_absent(self):
        # Simulate the CI case where pygetwindow isn't installed: force the
        # in-function `import pygetwindow` to raise ImportError. Removing it
        # from sys.modules isn't enough on a dev box (the real one re-imports),
        # so we also intercept __import__.
        real_import = __import__

        def _blocked(name, *a, **k):
            if name == "pygetwindow":
                raise ImportError("blocked: pygetwindow")
            return real_import(name, *a, **k)
        with inject_modules(pygetwindow=None), \
             mock.patch("builtins.__import__", side_effect=_blocked):
            self.assertIsNone(self.mod._sample_window_title())

    def test_window_title_getall_raises(self):
        gw = types.ModuleType("pygetwindow")
        gw.getAllWindows = mock.MagicMock(side_effect=Exception("win32 boom"))
        with inject_modules(pygetwindow=gw):
            self.assertIsNone(self.mod._sample_window_title())

    def test_window_title_apple_match_and_note_seen(self):
        gw = _fake_pygetwindow(["", "Holocene — Bon Iver – Apple Music"])
        bc = types.ModuleType("__main__")
        bc._note_apple_music_seen = mock.MagicMock()
        with inject_modules(pygetwindow=gw), \
             mock.patch.dict(sys.modules, {"__main__": bc}):
            r = self.mod._sample_window_title()
        self.assertEqual(r["source"], "web_apple")
        self.assertEqual(r["artist"], "Bon Iver")
        self.assertEqual(r["title"], "Holocene")
        bc._note_apple_music_seen.assert_called_once()

    def test_window_title_apple_note_seen_crash_swallowed(self):
        # _note_apple_music_seen blowing up must not break sampling.
        gw = _fake_pygetwindow(["Song — Artist – Apple Music"])
        bc = types.ModuleType("__main__")
        bc._note_apple_music_seen = mock.MagicMock(side_effect=Exception("nope"))
        with inject_modules(pygetwindow=gw), \
             mock.patch.dict(sys.modules, {"__main__": bc}):
            r = self.mod._sample_window_title()
        self.assertEqual(r["source"], "web_apple")

    def test_window_title_apple_present_but_unparseable(self):
        # "apple music" in title but no pattern matches → falls through to None.
        gw = _fake_pygetwindow(["Apple Music"])
        bc = types.ModuleType("__main__")
        bc._note_apple_music_seen = mock.MagicMock()
        with inject_modules(pygetwindow=gw), \
             mock.patch.dict(sys.modules, {"__main__": bc}):
            self.assertIsNone(self.mod._sample_window_title())

    def test_window_title_spotify_match(self):
        gw = _fake_pygetwindow(["Beat It · Michael Jackson - Spotify"])
        with inject_modules(pygetwindow=gw):
            r = self.mod._sample_window_title()
        self.assertEqual(r["source"], "web_spotify")
        self.assertEqual(r["artist"], "Michael Jackson")

    def test_window_title_spotify_idle_view_no_match(self):
        # Idle library view "Spotify - Liked Songs" must NOT parse as a track.
        gw = _fake_pygetwindow(["Spotify - Liked Songs"])
        with inject_modules(pygetwindow=gw):
            self.assertIsNone(self.mod._sample_window_title())

    def test_window_title_none_match(self):
        gw = _fake_pygetwindow(["Inbox - Gmail", ""])
        with inject_modules(pygetwindow=gw):
            self.assertIsNone(self.mod._sample_window_title())

    # ── _sample_now_playing ──────────────────────────────────────────────
    def test_now_playing_prefers_itunes(self):
        with mock.patch.object(self.mod, "_sample_itunes",
                               return_value={"artist": "A", "title": "B"}), \
             mock.patch.object(self.mod, "_sample_window_title") as win:
            r = self.mod._sample_now_playing()
        self.assertEqual(r["artist"], "A")
        win.assert_not_called()   # short-circuits on first hit

    def test_now_playing_falls_back_to_window(self):
        with mock.patch.object(self.mod, "_sample_itunes", return_value=None), \
             mock.patch.object(self.mod, "_sample_window_title",
                               return_value={"artist": "C", "title": "D"}):
            r = self.mod._sample_now_playing()
        self.assertEqual(r["artist"], "C")

    def test_now_playing_sampler_raises_is_swallowed(self):
        with mock.patch.object(self.mod, "_sample_itunes",
                               side_effect=Exception("boom")), \
             mock.patch.object(self.mod, "_sample_window_title",
                               return_value=None):
            self.assertIsNone(self.mod._sample_now_playing())

    def test_now_playing_skips_partial_result(self):
        # A result missing 'title' is rejected; falls through to next source.
        with mock.patch.object(self.mod, "_sample_itunes",
                               return_value={"artist": "A", "title": ""}), \
             mock.patch.object(self.mod, "_sample_window_title",
                               return_value=None):
            self.assertIsNone(self.mod._sample_now_playing())


# ─── disk-backed: IO helpers, aggregation, sessions, actions ─────────────

class AppleMusicDataTests(unittest.TestCase):
    """Tests that read/write the data files — redirect every path global to a
    temp dir so nothing under data/ is touched. Module mutable state (_current,
    _aggregator_started) is reset in tearDown."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("apple_music_intel")
        self.tmp = tempfile.mkdtemp(prefix="amintel_test_")
        self.addCleanup(self._cleanup)
        self.mod._DATA_DIR = self.tmp
        self.mod._HISTORY_FILE = os.path.join(self.tmp, "history.jsonl")
        self.mod._SKIPS_FILE = os.path.join(self.tmp, "skips.jsonl")
        self.mod._SESSION_FILE = os.path.join(self.tmp, "session.json")
        self.mod._TASTE_FILE = os.path.join(self.tmp, "taste.json")
        # Reset module mutable state between tests.
        self.addCleanup(self._reset_state)

    def _reset_state(self):
        self.mod._current = None
        self.mod._aggregator_started[0] = False

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

    # ── IO helpers: jsonl read/write, rotation, failure paths ────────────
    def test_read_jsonl_missing_file(self):
        self.assertEqual(self.mod._read_jsonl(os.path.join(self.tmp, "nope.jsonl")), [])

    def test_read_jsonl_skips_blank_and_malformed(self):
        path = os.path.join(self.tmp, "mixed.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"a": 1}\n')
            f.write("\n")                 # blank — skipped
            f.write("not json at all\n")  # malformed — skipped
            f.write('{"b": 2}\n')
        out = self.mod._read_jsonl(path)
        self.assertEqual(out, [{"a": 1}, {"b": 2}])

    def test_read_jsonl_open_error_returns_empty(self):
        # File EXISTS but open() raises mid-read → defensive [] return (the
        # outer try/except, not the not-exists early return).
        self.mod._append_jsonl(self.mod._HISTORY_FILE, {"x": 1})
        with mock.patch("builtins.open", side_effect=OSError("denied")):
            self.assertEqual(self.mod._read_jsonl(self.mod._HISTORY_FILE), [])

    def test_append_jsonl_failure_is_swallowed(self):
        # A failing open() must not raise out of _append_jsonl.
        with mock.patch("builtins.open", side_effect=OSError("denied")):
            self.mod._append_jsonl(self.mod._HISTORY_FILE, {"x": 1})  # no raise

    def test_atomic_write_json_roundtrip(self):
        path = os.path.join(self.tmp, "obj.json")
        self.mod._atomic_write_json(path, {"hello": "world"})
        import json
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"hello": "world"})

    def test_atomic_write_json_cleans_tmp_on_error(self):
        # json.dump raising must unlink the temp file and re-raise.
        before = set(os.listdir(self.tmp))
        with mock.patch("json.dump", side_effect=ValueError("bad")):
            with self.assertRaises(ValueError):
                self.mod._atomic_write_json(os.path.join(self.tmp, "x.json"), object())
        after = set(os.listdir(self.tmp))
        self.assertEqual(before, after)   # no stray .tmp left behind

    def test_atomic_write_json_unlink_failure_still_raises(self):
        # os.replace fails AND the cleanup os.unlink also fails → the nested
        # except swallows the unlink error but the original error re-raises.
        with mock.patch("os.replace", side_effect=OSError("replace boom")), \
             mock.patch("os.unlink", side_effect=OSError("unlink boom")):
            with self.assertRaises(OSError):
                self.mod._atomic_write_json(os.path.join(self.tmp, "y.json"),
                                            {"a": 1})

    def test_ensure_data_dir_swallows_error(self):
        with mock.patch("os.makedirs", side_effect=OSError("ro")):
            self.mod._ensure_data_dir()   # no raise

    def test_maybe_rotate_under_cap_noop(self):
        path = self.mod._HISTORY_FILE
        for i in range(5):
            self.mod._append_jsonl(path, {"i": i})
        self.mod._maybe_rotate(path, cap=10)
        self.assertEqual(len(self.mod._read_jsonl(path)), 5)

    def test_maybe_rotate_trims_to_cap(self):
        path = self.mod._HISTORY_FILE
        for i in range(20):
            self.mod._append_jsonl(path, {"i": i})
        self.mod._maybe_rotate(path, cap=5)
        rows = self.mod._read_jsonl(path)
        self.assertEqual(len(rows), 5)
        # Keeps the LAST cap lines.
        self.assertEqual([r["i"] for r in rows], [15, 16, 17, 18, 19])

    def test_maybe_rotate_missing_file_swallowed(self):
        self.mod._maybe_rotate(os.path.join(self.tmp, "ghost.jsonl"), cap=5)  # no raise

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

    def test_aggregate_skips_events_missing_artist(self):
        # An event with a blank artist is skipped entirely (line ~500).
        self.mod._append_jsonl(self.mod._HISTORY_FILE, {
            "ts": 1, "iso": "2026-05-29T21:00:00", "day": "friday", "hour": 21,
            "part": "evening", "artist": "", "title": "Anon", "genre": "Pop",
            "complete": True})
        snap = self.mod.aggregate()
        self.assertEqual(snap["events"], 1)        # counted as an event line
        self.assertEqual(snap["by_artist"], {})    # but no artist tallied

    def test_aggregate_genre_below_threshold_excluded(self):
        # A genre with <3 plays is omitted from skip_rate_by_genre.
        for k in range(2):
            self.mod._append_jsonl(self.mod._HISTORY_FILE, {
                "ts": k, "iso": f"2026-05-29T10:0{k}:00", "day": "friday",
                "hour": 10, "part": "morning", "artist": "Rare", "title": f"R{k}",
                "genre": "Jazz", "complete": False})
        snap = self.mod.aggregate()
        self.assertNotIn("Jazz", snap["skip_rate_by_genre"])

    def test_aggregate_artist_last_played_iso_tracks_max(self):
        for iso in ("2026-05-01T10:00:00", "2026-05-09T10:00:00",
                    "2026-05-05T10:00:00"):
            self.mod._append_jsonl(self.mod._HISTORY_FILE, {
                "ts": 1, "iso": iso, "day": "friday", "hour": 10,
                "part": "morning", "artist": "Repeat", "title": "T",
                "genre": "Pop", "complete": True})
        snap = self.mod.aggregate()
        self.assertEqual(snap["by_artist"]["Repeat"]["last_played_iso"],
                         "2026-05-09T10:00:00")

    def test_load_taste_missing(self):
        self.assertEqual(self.mod._load_taste(), {})

    def test_load_taste_malformed_json(self):
        with open(self.mod._TASTE_FILE, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        self.assertEqual(self.mod._load_taste(), {})

    def test_load_taste_non_dict(self):
        self.mod._atomic_write_json(self.mod._TASTE_FILE, ["a", "list"])
        self.assertEqual(self.mod._load_taste(), {})

    # ── _top_artist_for_slot ─────────────────────────────────────────────
    def test_top_artist_for_slot(self):
        self._seed_history()
        self.mod.aggregate()
        top = self.mod._top_artist_for_slot("friday", "evening")
        self.assertEqual(top, ("Michael Jackson", 3))

    def test_top_artist_for_slot_no_taste_file(self):
        self.assertIsNone(self.mod._top_artist_for_slot("friday", "evening"))

    def test_top_artist_for_slot_empty_slot(self):
        self._seed_history()
        self.mod.aggregate()
        self.assertIsNone(self.mod._top_artist_for_slot("monday", "morning"))

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

    def test_session_record_skip_dedups(self):
        self.mod._session_record_skip("Artist A", "Song A")
        self.mod._session_record_skip("Artist A", "Song A")
        keys = [k for k in self.mod._load_session()["skipped_keys"]
                if k == "artist a|song a"]
        self.assertEqual(len(keys), 1)

    def test_session_record_skip_blank_artist_title(self):
        # NOTE (latent quirk, not fixed): _track_key("", "") returns "|", which
        # is truthy, so the `if key and ...` guard in _session_record_skip does
        # NOT treat a fully-blank track as empty — it records the bare "|" key.
        # Documented here so the behaviour is pinned rather than asserted away.
        self.mod._session_record_skip("", "")
        self.assertIn("|", self.mod._session_skipped_keys())

    def test_session_expires_after_6_hours(self):
        # Write a session that started 7 hours ago → _load_session resets it.
        self.mod._save_session({"session_start": time.time() - 7 * 3600,
                                "skipped_keys": ["stale|key"]})
        self.assertNotIn("stale|key", self.mod._session_skipped_keys())

    def test_session_missing_file_fresh(self):
        # No session file at all → fresh empty session.
        self.assertEqual(self.mod._session_skipped_keys(), set())

    def test_session_zero_start_resets(self):
        self.mod._atomic_write_json(self.mod._SESSION_FILE,
                                    {"session_start": 0, "skipped_keys": ["x|y"]})
        self.assertNotIn("x|y", self.mod._session_skipped_keys())

    def test_session_non_dict_resets(self):
        self.mod._atomic_write_json(self.mod._SESSION_FILE, ["not", "a", "dict"])
        self.assertEqual(self.mod._session_skipped_keys(), set())

    def test_session_malformed_json_resets(self):
        with open(self.mod._SESSION_FILE, "w", encoding="utf-8") as f:
            f.write("{broken")
        self.assertEqual(self.mod._session_skipped_keys(), set())

    def test_save_session_caps_skip_list(self):
        cap = self.mod.SESSION_SKIP_CAP
        keys = [f"a{i}|t{i}" for i in range(cap + 50)]
        self.mod._save_session({"session_start": time.time(), "skipped_keys": keys})
        stored = self.mod._load_session()["skipped_keys"]
        self.assertEqual(len(stored), cap)
        # Keeps the tail.
        self.assertEqual(stored[-1], f"a{cap + 49}|t{cap + 49}")

    def test_save_session_write_failure_swallowed(self):
        with mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            self.mod._save_session({"session_start": time.time(),
                                    "skipped_keys": []})   # no raise

    # ── _log_listen ──────────────────────────────────────────────────────
    def test_log_listen_complete_only_history(self):
        prev = {"artist": "A", "title": "T", "album": "Al", "genre": "G",
                "source": "itunes"}
        self.mod._log_listen(prev, listened_secs=120)
        hist = self.mod._read_jsonl(self.mod._HISTORY_FILE)
        skips = self.mod._read_jsonl(self.mod._SKIPS_FILE)
        self.assertEqual(len(hist), 1)
        self.assertTrue(hist[0]["complete"])
        self.assertEqual(hist[0]["secs"], 120)
        self.assertEqual(skips, [])   # complete → no skip line

    def test_log_listen_short_play_is_skip(self):
        prev = {"artist": "A", "title": "T", "source": "itunes"}
        self.mod._log_listen(prev, listened_secs=3)
        hist = self.mod._read_jsonl(self.mod._HISTORY_FILE)
        skips = self.mod._read_jsonl(self.mod._SKIPS_FILE)
        self.assertEqual(len(hist), 1)
        self.assertFalse(hist[0]["complete"])
        self.assertEqual(len(skips), 1)   # short play also logged as skip

    # ── _listen_loop body (single iteration, no real sleep/thread) ───────
    def _run_one_listen_iteration(self):
        """Drive exactly one pass of _listen_loop's while-body by making the
        first time.sleep raise to break out after one iteration. The initial
        INITIAL_DELAY_SECS sleep and the loop's trailing sleep are the same
        time.sleep; we let the first two calls pass then raise."""
        calls = {"n": 0}

        def fake_sleep(_):
            calls["n"] += 1
            if calls["n"] >= 2:   # 1=initial delay, 2=end of first iteration
                raise _StopLoop()
        # Silence the module's logging.exception so a deliberately-triggered
        # error in the body doesn't spray a traceback onto the test console.
        with mock.patch.object(self.mod.time, "sleep", side_effect=fake_sleep), \
             mock.patch.object(self.mod.logging, "exception"):
            with self.assertRaises(_StopLoop):
                self.mod._listen_loop()

    def test_listen_loop_records_new_track(self):
        with mock.patch.object(self.mod, "_sample_now_playing",
                               return_value={"artist": "A", "title": "T",
                                             "album": "", "genre": "",
                                             "source": "itunes"}):
            self._run_one_listen_iteration()
        # First observation of a track sets _current but logs nothing yet.
        self.assertIsNotNone(self.mod._current)
        self.assertEqual(self.mod._current["artist"], "A")
        self.assertEqual(self.mod._read_jsonl(self.mod._HISTORY_FILE), [])

    def test_listen_loop_logs_on_track_change(self):
        # Pre-seed a current track that started 100s ago, then sample a new one.
        self.mod._current = {"key": "old|song", "artist": "Old", "title": "Song",
                             "album": "", "genre": "", "source": "itunes",
                             "since": time.time() - 100, "logged": False}
        with mock.patch.object(self.mod, "_sample_now_playing",
                               return_value={"artist": "New", "title": "Tune",
                                             "album": "", "genre": "",
                                             "source": "itunes"}):
            self._run_one_listen_iteration()
        hist = self.mod._read_jsonl(self.mod._HISTORY_FILE)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["artist"], "Old")   # the track that ended
        self.assertEqual(self.mod._current["artist"], "New")

    def test_listen_loop_finalises_when_nothing_playing(self):
        self.mod._current = {"key": "a|b", "artist": "A", "title": "B",
                             "album": "", "genre": "", "source": "itunes",
                             "since": time.time() - 50, "logged": False}
        with mock.patch.object(self.mod, "_sample_now_playing", return_value=None):
            self._run_one_listen_iteration()
        hist = self.mod._read_jsonl(self.mod._HISTORY_FILE)
        self.assertEqual(len(hist), 1)
        self.assertIsNone(self.mod._current)   # finalised + cleared

    def test_listen_loop_iteration_exception_swallowed(self):
        # An exception inside the body hits the except → sleep(POLL) → our
        # fake_sleep raises _StopLoop on the 2nd call. Proves the except path
        # is exercised without the loop dying.
        with mock.patch.object(self.mod, "_sample_now_playing",
                               side_effect=Exception("sample boom")):
            self._run_one_listen_iteration()
        # Nothing got logged because the body blew up before touching state.
        self.assertEqual(self.mod._read_jsonl(self.mod._HISTORY_FILE), [])

    # ── _aggregator_loop body ────────────────────────────────────────────
    def test_aggregator_loop_runs_then_breaks(self):
        self._seed_history()
        calls = {"n": 0}

        def fake_sleep(_):
            calls["n"] += 1
            if calls["n"] >= 2:   # 1=initial delay, 2=after first aggregate
                raise _StopLoop()
        with mock.patch.object(self.mod.time, "sleep", side_effect=fake_sleep):
            with self.assertRaises(_StopLoop):
                self.mod._aggregator_loop()
        # aggregate() ran → taste file now exists with our 7 events.
        self.assertEqual(self.mod._load_taste()["events"], 7)

    def test_aggregator_loop_exception_swallowed(self):
        calls = {"n": 0}

        def fake_sleep(_):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise _StopLoop()
        with mock.patch.object(self.mod, "aggregate",
                               side_effect=Exception("agg boom")), \
             mock.patch.object(self.mod.time, "sleep", side_effect=fake_sleep), \
             mock.patch.object(self.mod.logging, "exception"):
            with self.assertRaises(_StopLoop):
                self.mod._aggregator_loop()   # except branch keeps it alive

    # ── music_history / music_taste / music_aggregate ────────────────────
    def test_music_history_empty(self):
        self.assertIn("No listening history", self.actions["music_history"](""))

    def test_music_history_lists_recent(self):
        self._seed_history()
        out = self.actions["music_history"]("")
        self.assertIn("Michael Jackson", out)
        self.assertIn("(skipped)", out)   # the Country tracks were skips

    def test_music_history_caps_at_five(self):
        for i in range(8):
            self.mod._append_jsonl(self.mod._HISTORY_FILE, {
                "iso": f"2026-05-29T10:0{i}:00", "title": f"S{i}",
                "artist": "A", "secs": 100, "complete": True})
        out = self.actions["music_history"]("")
        # Only the last 5 listens are read back.
        self.assertIn("S7", out)
        self.assertNotIn("S2", out)

    def test_music_taste_no_data(self):
        self.assertIn("No taste data yet", self.actions["music_taste"](""))

    def test_music_taste_reports_top_artists(self):
        self._seed_history()
        self.mod.aggregate()
        out = self.actions["music_taste"]("")
        self.assertIn("Michael Jackson", out)
        self.assertIn("skip rate", out.lower())

    def test_music_taste_overall_skip_rate_branch(self):
        # Taste with by_artist but NO per-genre skip rates (all genres <3
        # plays) → the 'overall skip rate' branch (861-862) is taken.
        snap = {
            "by_artist": {"A": {"count": 5, "last_played_iso": "", "skips": 1}},
            "by_slot": {},
            "skip_rate_by_genre": {},
            "skip_rate_overall": 0.25,
        }
        self.mod._atomic_write_json(self.mod._TASTE_FILE, snap)
        out = self.actions["music_taste"]("")
        # The skip clause is rendered via str.capitalize() → leading cap.
        self.assertIn("Overall skip rate 25%", out)

    def test_music_taste_no_slot_pattern_phrase(self):
        # by_artist present but no by_slot for the current slot → "no strong
        # <day> <part> pattern yet" branch.
        snap = {
            "by_artist": {"A": {"count": 5, "last_played_iso": "", "skips": 0}},
            "by_slot": {},
            "skip_rate_by_genre": {"Pop": 0.5},
            "skip_rate_overall": 0.5,
        }
        self.mod._atomic_write_json(self.mod._TASTE_FILE, snap)
        out = self.actions["music_taste"]("")
        self.assertIn("no strong", out.lower())

    def test_music_aggregate_action(self):
        self._seed_history()
        out = self.actions["music_aggregate"]("")
        self.assertIn("7 listen events", out)

    # ── play_vibe ────────────────────────────────────────────────────────
    def test_play_vibe_no_pattern(self):
        # No history → no slot pattern → polite refusal mentioning the slot.
        out = self.actions["play_vibe"]("friday night")
        self.assertIn("friday night", out)
        self.assertIn("don't have a strong", out)

    def test_play_vibe_reaggregate_raises_is_swallowed(self):
        # First lookup misses (no taste). The in-handler aggregate() raises,
        # the except swallows it, the retry still misses → polite refusal.
        with mock.patch.object(self.mod, "aggregate",
                               side_effect=Exception("agg boom")):
            out = self.actions["play_vibe"]("friday night")
        self.assertIn("don't have a strong", out)

    def test_play_vibe_reaggregates_then_finds(self):
        # Taste file stale/missing but history present: first lookup misses,
        # play_vibe re-runs aggregate(), second lookup hits. Player reached.
        self._seed_history()   # history on disk, but no taste.json yet
        bc = mock.MagicMock()
        bc._play_music_core.return_value = (True, "now playing Michael Jackson")
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            out = self.actions["play_vibe"]("friday evening")
        self.assertIn("Vibing your usual friday evening", out)
        bc._play_music_core.assert_called_once()

    def test_play_vibe_reaches_player_core(self):
        self._seed_history()
        self.mod.aggregate()
        bc = mock.MagicMock()
        bc._play_music_core.return_value = (True, "now playing Michael Jackson")
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            out = self.actions["play_vibe"]("friday evening")
        self.assertIn("Vibing your usual friday evening", out)

    def test_play_vibe_core_fails_falls_back_to_apple_music(self):
        # _play_music_core returns ok=False → fall through to _act_apple_music.
        self._seed_history()
        self.mod.aggregate()
        bc = mock.MagicMock()
        bc._play_music_core.return_value = (False, "no track")
        bc._act_apple_music.return_value = "queued on Apple Music web"
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            out = self.actions["play_vibe"]("friday evening")
        self.assertIn("Vibing your usual friday evening", out)
        bc._act_apple_music.assert_called_once_with("Michael Jackson")

    def test_play_vibe_core_raises_falls_back_to_apple_music(self):
        self._seed_history()
        self.mod.aggregate()
        bc = mock.MagicMock()
        bc._play_music_core.side_effect = Exception("core boom")
        bc._act_apple_music.return_value = "queued"
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            out = self.actions["play_vibe"]("friday evening")
        self.assertIn("Vibing your usual friday evening", out)

    def test_play_vibe_no_player_reachable(self):
        # bobert present but neither helper exists → final "can't reach a
        # player" message including the listen count.
        self._seed_history()
        self.mod.aggregate()
        bc = mock.MagicMock(spec=[])   # no _play_music_core / _act_apple_music
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            out = self.actions["play_vibe"]("friday evening")
        self.assertIn("Michael Jackson", out)
        self.assertIn("can't reach a player", out)

    def test_play_vibe_bobert_none(self):
        self._seed_history()
        self.mod.aggregate()
        with mock.patch.object(self.mod, "_bobert", return_value=None):
            out = self.actions["play_vibe"]("friday evening")
        self.assertIn("can't reach a player", out)

    def test_play_vibe_apple_music_raises_then_final_msg(self):
        # core missing, _act_apple_music present but raises → final fallback.
        self._seed_history()
        self.mod.aggregate()
        bc = mock.MagicMock(spec=["_act_apple_music"])
        bc._act_apple_music.side_effect = Exception("web boom")
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            out = self.actions["play_vibe"]("friday evening")
        self.assertIn("can't reach a player", out)

    # ── skip_track ───────────────────────────────────────────────────────
    def test_skip_track_records_and_skips_media_key(self):
        # No current track; sampling returns one. media_next is the fallback.
        bc = mock.MagicMock()
        del bc._act_next_song  # force the media-key fallback branch
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_sample_now_playing",
                               return_value={"artist": "Foo", "title": "Bar",
                                             "source": "web_apple"}):
            out = self.actions["skip_track"]("")
        self.assertIn("set 'Bar' by Foo aside", out)
        self.assertIn("via media key", out)
        self.assertIn("foo|bar", self.mod._session_skipped_keys())
        # A synthetic skip event was logged.
        skips = self.mod._read_jsonl(self.mod._SKIPS_FILE)
        self.assertTrue(any(s.get("user_skip") for s in skips))

    def test_skip_track_synthetic_event_reaches_history_and_taste_model(self):
        # Regression: the synthetic user-skip event must land in _HISTORY_FILE
        # (which aggregate() reads), not just the write-only _SKIPS_FILE.
        bc = mock.MagicMock()
        del bc._act_next_song
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_sample_now_playing",
                               return_value={"artist": "Foo", "title": "Bar",
                                             "source": "web_apple"}):
            self.actions["skip_track"]("")
        hist = self.mod._read_jsonl(self.mod._HISTORY_FILE)
        self.assertTrue(any(e.get("user_skip") for e in hist))
        snap = self.mod.aggregate()
        self.assertEqual(snap["by_artist"]["Foo"]["skips"], 1)

    def test_skip_track_uses_itunes_next_when_source_itunes(self):
        # _current is an iTunes track → _act_next_song path (via iTunes).
        self.mod._current = {"key": "mj|beat it", "artist": "MJ", "title": "Beat It",
                             "album": "", "genre": "Pop", "source": "itunes",
                             "since": time.time()}
        bc = mock.MagicMock()
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            out = self.actions["skip_track"]("")
        bc._act_next_song.assert_called_once()
        self.assertIn("via iTunes", out)
        self.assertIn("mj|beat it", self.mod._session_skipped_keys())

    def test_skip_track_itunes_next_raises_falls_back_to_media(self):
        self.mod._current = {"key": "mj|beat it", "artist": "MJ", "title": "Beat It",
                             "album": "", "genre": "Pop", "source": "itunes",
                             "since": time.time()}
        bc = mock.MagicMock()
        bc._act_next_song.side_effect = Exception("itunes boom")
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            out = self.actions["skip_track"]("")
        bc._act_media_next.assert_called_once()
        self.assertIn("via media key", out)

    def test_skip_track_unknown_track_still_skips(self):
        # No _current and sampling yields nothing → generic "current track".
        bc = mock.MagicMock()
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_sample_now_playing", return_value=None):
            out = self.actions["skip_track"]("")
        self.assertIn("the current track", out)
        # Nothing recorded (no artist/title known).
        self.assertEqual(self.mod._session_skipped_keys(), set())

    def test_skip_track_media_next_raises_no_where_clause(self):
        # No _act_next_song and _act_media_next raises → skipped_via stays None.
        bc = mock.MagicMock()
        del bc._act_next_song
        bc._act_media_next.side_effect = Exception("media boom")
        with mock.patch.object(self.mod, "_bobert", return_value=bc), \
             mock.patch.object(self.mod, "_sample_now_playing",
                               return_value={"artist": "Foo", "title": "Bar",
                                             "source": "web_apple"}):
            out = self.actions["skip_track"]("")
        self.assertIn("set 'Bar' by Foo aside", out)
        self.assertNotIn("via", out)   # no "(via ...)" suffix

    def test_skip_track_bobert_none(self):
        # _bobert() returns None → no skip issued, still records + replies.
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.object(self.mod, "_sample_now_playing",
                               return_value={"artist": "Foo", "title": "Bar",
                                             "source": "web_apple"}):
            out = self.actions["skip_track"]("")
        self.assertIn("set 'Bar' by Foo aside", out)


# ─── play_unheard (iTunes library scan over a fake COM library) ───────────

class PlayUnheardTests(unittest.TestCase):
    """_act_play_unheard walks the iTunes LibraryPlaylist via COM. We drive it
    with a fake itunes_bridge + fake library; pythoncom is injected as a stub
    because the skill does `import pythoncom` mid-function (it's not on CI)."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("apple_music_intel")
        self.tmp = tempfile.mkdtemp(prefix="amintel_unheard_")
        self.addCleanup(self._cleanup)
        self.mod._SESSION_FILE = os.path.join(self.tmp, "session.json")
        self.mod._DATA_DIR = self.tmp
        self._real_bridge = self.mod.itunes_bridge
        self.bridge = mock.MagicMock(name="itunes_bridge")
        self.mod.itunes_bridge = self.bridge
        self.addCleanup(self._restore_bridge)

    def _restore_bridge(self):
        self.mod.itunes_bridge = self._real_bridge

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

    @contextlib.contextmanager
    def _pythoncom(self):
        with inject_modules(pythoncom=_fake_pythoncom()):
            yield

    def test_play_unheard_itunes_unreachable(self):
        self.bridge.get_client.return_value = (None, "iTunes not reachable, sir.")
        out = self.actions["play_unheard"]("")
        self.assertEqual(out, "iTunes not reachable, sir.")

    def test_play_unheard_itunes_unreachable_default_msg(self):
        # Bridge returns (None, "") → default fallback string.
        self.bridge.get_client.return_value = (None, "")
        out = self.actions["play_unheard"]("")
        self.assertIn("iTunes not reachable", out)

    def test_play_unheard_empty_library(self):
        app = _FakeITunesApp(lib_tracks=[])
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("")
        self.assertIn("library is empty", out)

    def test_play_unheard_picks_never_played(self):
        # Two tracks: one played yesterday, one never played. With a 14-day
        # default cutoff the never-played one wins (lowest played_ts).
        recent = _FakeLibTrack("Recent", "ArtistR", time.time() - 86400)
        never = _FakeLibTrack("FreshCut", "ArtistN", None)
        app = _FakeITunesApp(lib_tracks=[recent, never])
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("")
        self.assertIn("FreshCut", out)
        self.assertIn("ArtistN", out)
        self.assertIn("never before", out)
        self.assertTrue(never.played)
        self.assertFalse(recent.played)

    def test_play_unheard_picks_oldest_played(self):
        # All played, but one long ago (>14 days) → eligible; report its date.
        old_ts = time.time() - 90 * 86400
        old = _FakeLibTrack("OldFave", "ArtistO", old_ts)
        app = _FakeITunesApp(lib_tracks=[old])
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("")
        self.assertIn("OldFave", out)
        self.assertIn("last on", out)
        self.assertTrue(old.played)

    def test_play_unheard_arg_override_days(self):
        # Track played 10 days ago. Default (14) would EXCLUDE it... no, 10<14
        # means within cutoff → excluded by default. With arg "5", 10>5 days
        # ago → eligible. Use the arg to flip eligibility.
        played_ts = time.time() - 10 * 86400
        t = _FakeLibTrack("TenDays", "ArtistT", played_ts)
        app = _FakeITunesApp(lib_tracks=[t])
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("5")
        self.assertIn("TenDays", out)

    def test_play_unheard_all_too_recent(self):
        # Everything played within the cutoff window → no candidates message.
        t = _FakeLibTrack("Yesterday", "ArtistY", time.time() - 86400)
        app = _FakeITunesApp(lib_tracks=[t])
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("30")
        self.assertIn("No tracks unheard", out)

    def test_play_unheard_invalid_arg_uses_default(self):
        # Non-numeric arg → falls back to UNHEARD_MIN_DAYS; never-played wins.
        never = _FakeLibTrack("Fresh", "A", None)
        app = _FakeITunesApp(lib_tracks=[never])
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("not a number")
        self.assertIn("Fresh", out)

    def test_play_unheard_skips_session_skipped(self):
        # The only eligible track is in the session skip set → excluded.
        never = _FakeLibTrack("Fresh", "SkipArtist", None)
        app = _FakeITunesApp(lib_tracks=[never])
        self.bridge.get_client.return_value = (app, None)
        self.mod._session_record_skip("SkipArtist", "Fresh")
        with self._pythoncom():
            out = self.actions["play_unheard"]("")
        self.assertIn("No tracks unheard", out)

    def test_play_unheard_skips_tracks_missing_name_or_artist(self):
        # A track with no Name/Artist is skipped; the valid one is chosen.
        blank = _FakeLibTrack("", "", None)
        good = _FakeLibTrack("Good", "GoodArtist", None)
        app = _FakeITunesApp(lib_tracks=[blank, good])
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("")
        self.assertIn("Good", out)

    def test_play_unheard_played_date_timestamp_fallback(self):
        # PlayedDate.timestamp() raises → falls back to time.mktime(timetuple()).
        class _PD:
            def timestamp(self):
                raise AttributeError("old pywin32")

            def timetuple(self):
                return time.localtime(time.time() - 90 * 86400)

        class _T(_FakeLibTrack):
            @property
            def PlayedDate(self):
                return _PD()
        t = _T("MktimePath", "ArtistM", 0)
        app = _FakeITunesApp(lib_tracks=[t])
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("")
        self.assertIn("MktimePath", out)

    def test_play_unheard_played_date_both_fallbacks_raise(self):
        # BOTH PlayedDate.timestamp() and mktime(timetuple()) raise → played_ts
        # forced to 0.0 (never-played) → track is eligible (696-697).
        class _PD:
            def timestamp(self):
                raise AttributeError("no ts")

            def timetuple(self):
                raise ValueError("no timetuple")

        class _T(_FakeLibTrack):
            @property
            def PlayedDate(self):
                return _PD()
        t = _T("BothRaise", "ArtistB", 0)
        app = _FakeITunesApp(lib_tracks=[t])
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("")
        self.assertIn("BothRaise", out)
        self.assertIn("never before", out)

    def test_play_unheard_per_track_getattr_raises_skipped(self):
        # A track whose attribute access blows up AFTER name/artist read (e.g.
        # PlayedDate raising) is skipped via the inner except (708-709); a
        # second, healthy track is chosen.
        class _Boom(_FakeLibTrack):
            @property
            def PlayedDate(self):
                raise Exception("attr boom")
        boom = _Boom("Boom", "ArtistBoom", 0)
        good = _FakeLibTrack("Healthy", "ArtistH", None)
        app = _FakeITunesApp(lib_tracks=[boom, good])
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("")
        self.assertIn("Healthy", out)

    def test_play_unheard_sentinel_date_treated_never(self):
        # A pre-1990 PlayedDate (iTunes' 1899 never-played sentinel) → treated
        # as never played (played_ts forced to 0) → eligible, "never before".
        sentinel_ts = -2208988800.0   # 1899-12-30
        t = _FakeLibTrack("SentinelTrack", "ArtistS", sentinel_ts)
        app = _FakeITunesApp(lib_tracks=[t])
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("")
        self.assertIn("SentinelTrack", out)
        self.assertIn("never before", out)

    def test_play_unheard_item_access_raises_skipped(self):
        # Tracks.Item(i) raising for one index is skipped; loop continues.
        good = _FakeLibTrack("Survivor", "ArtistG", None)

        class _RaisingCollection:
            Count = 2

            def Item(self, i):
                if i == 1:
                    raise Exception("COM item boom")
                return good
        app = _FakeITunesApp(lib_tracks=[])
        app.LibraryPlaylist.Tracks = _RaisingCollection()
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("")
        self.assertIn("Survivor", out)

    def test_play_unheard_outer_exception_returns_error(self):
        # LibraryPlaylist access blowing up → the broad except → error string.
        app = mock.MagicMock()
        type(app).LibraryPlaylist = mock.PropertyMock(side_effect=Exception("lib boom"))
        self.bridge.get_client.return_value = (app, None)
        with self._pythoncom():
            out = self.actions["play_unheard"]("")
        self.assertIn("unheard search failed", out)


# ─── register() wiring ───────────────────────────────────────────────────

class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="amintel_reg_")
        self.addCleanup(self._cleanup)

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

    def test_register_wires_all_actions(self):
        # The harness neuters Thread.start, so the loops never run.
        mod, actions = load_skill_isolated("apple_music_intel")
        for name in ("play_unheard", "play_vibe", "skip_track",
                     "music_history", "music_taste", "music_aggregate"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))

    def test_register_resets_session(self):
        # register() writes a fresh empty session on boot.
        mod, _ = load_skill_isolated("apple_music_intel", register=False)
        mod._SESSION_FILE = os.path.join(self.tmp, "session.json")
        mod._DATA_DIR = self.tmp
        # Pre-existing skips that should be wiped by register().
        mod._save_session({"session_start": time.time(),
                           "skipped_keys": ["old|skip"]})
        import threading
        with mock.patch.object(threading.Thread, "start", lambda self: None):
            mod.register({})
        self.assertEqual(mod._session_skipped_keys(), set())

    def test_register_session_reset_failure_swallowed(self):
        mod, _ = load_skill_isolated("apple_music_intel", register=False)
        import threading
        with mock.patch.object(mod, "_save_session",
                               side_effect=OSError("disk full")), \
             mock.patch.object(threading.Thread, "start", lambda self: None):
            mod.register({})   # must not raise

    def test_register_starts_named_threads_when_absent(self):
        # When no am-intel-* threads are alive, register() constructs both.
        mod, _ = load_skill_isolated("apple_music_intel", register=False)
        mod._DATA_DIR = self.tmp
        mod._SESSION_FILE = os.path.join(self.tmp, "session.json")
        import threading
        started = []
        real_init = threading.Thread.__init__

        def rec_init(self, *a, **k):
            real_init(self, *a, **k)
            if k.get("name"):
                started.append(k["name"])
        with mock.patch.object(threading.Thread, "__init__", rec_init), \
             mock.patch.object(threading.Thread, "start", lambda self: None):
            mod.register({})
        self.assertIn("am-intel-listen", started)
        self.assertIn("am-intel-aggregate", started)

    def test_register_skips_threads_already_alive(self):
        # Simulate both loops already running → register() starts neither.
        mod, _ = load_skill_isolated("apple_music_intel", register=False)
        mod._DATA_DIR = self.tmp
        mod._SESSION_FILE = os.path.join(self.tmp, "session.json")
        import threading

        class _FakeThread:
            def __init__(self, name):
                self.name = name

            def is_alive(self):
                return True
        fakes = [_FakeThread("am-intel-listen"), _FakeThread("am-intel-aggregate")]
        constructed = []

        def fake_enumerate():
            return fakes

        def rec_init(self, *a, **k):
            constructed.append(k.get("name"))
        # Patch enumerate to report the loops alive, and Thread() so that if
        # register() *did* try to start one we'd see its name.
        with mock.patch.object(threading, "enumerate", fake_enumerate), \
             mock.patch.object(threading.Thread, "__init__", rec_init), \
             mock.patch.object(threading.Thread, "start", lambda self: None):
            mod.register({})
        self.assertEqual([c for c in constructed if c and c.startswith("am-intel")],
                         [])


# Sentinel used to break background loops after one iteration.
class _StopLoop(Exception):
    pass


if __name__ == "__main__":
    unittest.main()
