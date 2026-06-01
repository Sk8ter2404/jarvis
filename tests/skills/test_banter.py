"""Logic tests for skills/banter.py.

Targets the behavioural-tell detectors that decide when JARVIS volunteers a
zinger: repeat_question (same utterance ≥2× in 10 min), repeat_open (same
target ≥5× today), tab_clutter (chrome/window count), and music_while_music.
Also covers text normalization, zinger selection/formatting, the visible-window
filter, in-call suppression, and the banter_status action.

The 90s scheduler thread is neutered by the harness. pattern_memory access,
psutil, and window enumeration are all mocked so detection is deterministic.
"""
from __future__ import annotations

import datetime
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _entry(text, ago_seconds, iso_date=None):
    ts = time.time() - ago_seconds
    iso = (iso_date or datetime.date.today().isoformat()) + "T12:00:00"
    return {"ts": ts, "iso": iso, "text": text}


class BanterNormalizeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("banter")

    def test_normalize_strips_punct_and_lowercases(self):
        self.assertEqual(self.mod._normalize_text("What's the WEATHER today!?"),
                         "whats the weather today")

    def test_normalize_collapses_whitespace(self):
        self.assertEqual(self.mod._normalize_text("  open    chrome  "),
                         "open chrome")


class BanterRepeatQuestionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("banter")

    def test_detects_near_duplicate_within_window(self):
        entries = [
            _entry("What's the weather today", 60),
            _entry("Whats the weather today!", 30),
        ]
        tell = self.mod._detect_repeat_question(entries)
        self.assertIsNotNone(tell)
        self.assertEqual(tell["tell"], "repeat_question")
        self.assertEqual(tell["n"], 2)
        self.assertEqual(tell["text"], "whats the weather today")

    def test_ignores_short_commands(self):
        # < 3 words are excluded so "stop"/"next" don't trigger.
        entries = [_entry("stop", 60), _entry("stop", 30)]
        self.assertIsNone(self.mod._detect_repeat_question(entries))

    def test_ignores_old_entries(self):
        # Both outside the 10-min window.
        entries = [
            _entry("what is the weather today", 60 * 60),
            _entry("what is the weather today", 50 * 60),
        ]
        self.assertIsNone(self.mod._detect_repeat_question(entries))

    def test_single_occurrence_no_tell(self):
        self.assertIsNone(self.mod._detect_repeat_question(
            [_entry("what is the weather today", 60)]))


class BanterRepeatOpenTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("banter")

    def test_detects_repeat_open_over_threshold(self):
        entries = [_entry("open chrome", 100 + i) for i in range(6)]
        with mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("open", "chrome")):
            tell = self.mod._detect_repeat_open(entries)
        self.assertIsNotNone(tell)
        self.assertEqual(tell["tell"], "repeat_open")
        self.assertEqual(tell["target"], "chrome")
        self.assertGreaterEqual(tell["n"], self.mod.REPEAT_OPEN_THRESHOLD)

    def test_under_threshold_no_tell(self):
        entries = [_entry("open chrome", 100 + i) for i in range(3)]
        with mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("open", "chrome")):
            self.assertIsNone(self.mod._detect_repeat_open(entries))

    def test_non_open_category_ignored(self):
        entries = [_entry("play jazz", 100 + i) for i in range(6)]
        with mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("play", "jazz")):
            self.assertIsNone(self.mod._detect_repeat_open(entries))


class BanterTabClutterTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("banter")

    def test_chrome_clutter_detected(self):
        with mock.patch.object(self.mod, "_chrome_process_count", return_value=47), \
             mock.patch.object(self.mod, "_visible_window_count", return_value=3):
            tell = self.mod._detect_tab_clutter()
        self.assertEqual(tell["tell"], "tab_clutter")
        self.assertEqual(tell["n"], 47)

    def test_window_clutter_fallback(self):
        with mock.patch.object(self.mod, "_chrome_process_count", return_value=2), \
             mock.patch.object(self.mod, "_visible_window_count", return_value=55):
            tell = self.mod._detect_tab_clutter()
        self.assertEqual(tell["tell"], "tab_clutter_windows")
        self.assertEqual(tell["n"], 55)

    def test_no_clutter(self):
        with mock.patch.object(self.mod, "_chrome_process_count", return_value=5), \
             mock.patch.object(self.mod, "_visible_window_count", return_value=10):
            self.assertIsNone(self.mod._detect_tab_clutter())

    def test_visible_window_count_filters_system_surfaces(self):
        titles = ["Program Manager", "jarvis_hud", "Real Work - VS Code",
                  "Inbox - Chrome"]
        with mock.patch.object(self.mod, "_all_window_titles", return_value=titles):
            # Two real windows survive the ignore filter.
            self.assertEqual(self.mod._visible_window_count(), 2)


class BanterMusicWhileMusicTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("banter")

    def test_no_bc_returns_none(self):
        import sys
        with mock.patch.dict(sys.modules, {"bobert_companion": None}):
            self.assertIsNone(self.mod._detect_music_while_music([]))

    def test_detects_play_after_jarvis_started_music(self):
        import sys
        fake_bc = mock.MagicMock()
        fake_bc._jarvis_played_music_at = [time.time() - 60]   # 1 min ago
        play_entry = _entry("play some jazz", 30)              # after that ts
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}), \
             mock.patch.object(self.mod, "_extract_target_safe",
                               return_value=("play", "jazz")):
            tell = self.mod._detect_music_while_music([play_entry])
        self.assertIsNotNone(tell)
        self.assertEqual(tell["tell"], "music_while_music")

    def test_no_recent_jarvis_music(self):
        import sys
        fake_bc = mock.MagicMock()
        fake_bc._jarvis_played_music_at = [time.time() - 99999]  # too old
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            self.assertIsNone(self.mod._detect_music_while_music([]))


class BanterZingerTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("banter")

    def test_pick_zinger_formats_placeholders(self):
        line = self.mod._pick_zinger({"tell": "tab_clutter", "n": 47})
        self.assertIn("47", line)

    def test_pick_zinger_repeat_question_interpolates_count(self):
        # Pin the variant that uses {n} so the assertion is deterministic.
        nth_variant = "That's the {n}th time you've asked me that today, sir."
        with mock.patch.object(self.mod.random, "choice", return_value=nth_variant):
            line = self.mod._pick_zinger(
                {"tell": "repeat_question", "n": 3, "minutes": 5})
        self.assertEqual(line, "That's the 3th time you've asked me that today, sir.")

    def test_pick_zinger_repeat_question_every_variant_renders(self):
        # No variant may crash on .format(**tell); each yields non-empty text.
        tell = {"tell": "repeat_question", "n": 3, "minutes": 5}
        for variant in self.mod._ZINGER_BANK["repeat_question"]:
            with mock.patch.object(self.mod.random, "choice", return_value=variant):
                line = self.mod._pick_zinger(tell)
            self.assertTrue(line)
            self.assertNotIn("{", line)   # all placeholders resolved

    def test_pick_zinger_unknown_tell_empty(self):
        self.assertEqual(self.mod._pick_zinger({"tell": "no_such_tell"}), "")

    def test_zinger_bank_each_tell_has_variants(self):
        for tell, variants in self.mod._ZINGER_BANK.items():
            self.assertGreaterEqual(len(variants), 2,
                                    f"{tell} should have ≥2 variants")


class BanterStatusTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("banter")

    def test_status_no_fires(self):
        with mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False):
            out = self.actions["banter_status"]("")
        self.assertIn("no zingers yet", out.lower())
        self.assertTrue(out.startswith("Banter engine"))

    def test_status_reports_last_zinger(self):
        state = {"last_fire_at": time.time() - 600, "last_tell": "tab_clutter"}
        with mock.patch.object(self.mod, "_load_state", return_value=state), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False):
            out = self.actions["banter_status"]("")
        self.assertIn("last zinger", out.lower())
        self.assertIn("tab_clutter", out)


if __name__ == "__main__":
    unittest.main()
