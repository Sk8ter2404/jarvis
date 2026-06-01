"""Logic tests for skills/suit_up.py.

Targets the cinematic boot logic without real TTS, HUD, or overlay:
  • _build_diagnostic_lines — the four-line readout, with/without a speaker.
  • _resolve_speaker_name — explicit name passthrough + fallback.
  • _is_warm_restart — the 18-hour window gate (time controlled).
  • play_suit_up_sequence — drives a fake speak_fn and asserts the lines are
    spoken in order and the welcome line is returned.
  • maybe_play_morning_suit_up — the (warm-restart AND not-fired-today) gate,
    with state persistence redirected to a temp file.

Overlay coordination (_ensure_holo_overlay_up / _dismiss_holo_overlay) and the
HUD animation writers are patched so nothing touches a real HUD; time.sleep in
_clear_animation is patched so tests stay fast.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class SuitUpBuilderTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")

    def test_diagnostic_lines_with_speaker(self):
        lines = self.mod._build_diagnostic_lines("Gaming Headset")
        self.assertEqual(lines[0], "Diagnostics: nominal.")
        self.assertEqual(lines[1], "Network: online.")
        self.assertIn("Gaming Headset", lines[2])
        self.assertEqual(lines[3], "Workshop: standing by.")

    def test_diagnostic_lines_without_speaker(self):
        lines = self.mod._build_diagnostic_lines("")
        self.assertEqual(lines[2], "Audio: connected.")

    def test_resolve_speaker_explicit(self):
        self.assertEqual(self.mod._resolve_speaker_name("My Headset"),
                         "My Headset")

    def test_resolve_speaker_blank_explicit_falls_back(self):
        # No explicit name and bobert_companion lookup fails → "system default".
        import sys
        with mock.patch.dict(sys.modules, {"bobert_companion": None}):
            self.assertEqual(self.mod._resolve_speaker_name(""), "system default")


class SuitUpWarmRestartTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")

    def test_warm_restart_within_window(self):
        # Previous session ended 2 hours ago → warm.
        with mock.patch.object(self.mod, "_last_session_end_ts",
                               return_value=time.time() - 2 * 3600):
            self.assertTrue(self.mod._is_warm_restart())

    def test_not_warm_restart_when_too_old(self):
        # 20 hours ago → outside the 18-hour window.
        with mock.patch.object(self.mod, "_last_session_end_ts",
                               return_value=time.time() - 20 * 3600):
            self.assertFalse(self.mod._is_warm_restart())

    def test_not_warm_restart_when_no_prior_session(self):
        with mock.patch.object(self.mod, "_last_session_end_ts",
                               return_value=0.0):
            self.assertFalse(self.mod._is_warm_restart())


class SuitUpSequenceTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")

    def test_sequence_speaks_all_lines_in_order(self):
        spoken = []
        speak_fn = lambda line: spoken.append(line)
        with mock.patch.object(self.mod, "_ensure_holo_overlay_up",
                               return_value=False), \
             mock.patch.object(self.mod, "_dismiss_holo_overlay"), \
             mock.patch.object(self.mod, "_start_animation", return_value=0.0), \
             mock.patch.object(self.mod, "_clear_animation"), \
             mock.patch.object(self.mod.time, "sleep"):
            welcome = self.mod.play_suit_up_sequence(speak_fn=speak_fn)
        # 4 diagnostics + 1 welcome.
        self.assertEqual(len(spoken), 5)
        self.assertEqual(spoken[0], "Diagnostics: nominal.")
        self.assertEqual(spoken[-1], "Welcome back, sir. Systems are yours.")
        self.assertEqual(welcome, "Welcome back, sir. Systems are yours.")

    def test_sequence_dismisses_overlay_only_if_it_launched_it(self):
        # When _ensure_holo_overlay_up returns True (we launched it), the
        # sequence must dismiss it afterwards.
        with mock.patch.object(self.mod, "_ensure_holo_overlay_up",
                               return_value=True), \
             mock.patch.object(self.mod, "_dismiss_holo_overlay") as dismiss, \
             mock.patch.object(self.mod, "_start_animation", return_value=0.0), \
             mock.patch.object(self.mod, "_clear_animation"), \
             mock.patch.object(self.mod.time, "sleep"):
            self.mod.play_suit_up_sequence(speak_fn=lambda _l: None)
        dismiss.assert_called_once()

    def test_sequence_survives_speak_failure(self):
        # A throwing speak_fn must not crash the sequence — it still returns.
        def boom(_line):
            raise RuntimeError("tts down")
        with mock.patch.object(self.mod, "_ensure_holo_overlay_up",
                               return_value=False), \
             mock.patch.object(self.mod, "_start_animation", return_value=0.0), \
             mock.patch.object(self.mod, "_clear_animation"), \
             mock.patch.object(self.mod.time, "sleep"):
            welcome = self.mod.play_suit_up_sequence(speak_fn=boom)
        self.assertEqual(welcome, "Welcome back, sir. Systems are yours.")


class SuitUpMorningGateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("suit_up")
        fd, self.statep = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.statep)   # start with no state file
        self._patch = mock.patch.object(self.mod, "_STATE_FILE", self.statep)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        try:
            os.unlink(self.statep)
        except OSError:
            pass

    def test_morning_fires_on_warm_restart_first_time(self):
        with mock.patch.object(self.mod, "_is_warm_restart", return_value=True), \
             mock.patch.object(self.mod, "play_suit_up_sequence",
                               return_value="Welcome back, sir. Systems are yours."
                               ) as seq:
            out = self.mod.maybe_play_morning_suit_up(speak_fn=lambda _l: None)
        self.assertIn("Welcome back", out)
        seq.assert_called_once()
        # State now records today's fire.
        with open(self.statep, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.assertEqual(state["last_fired_date"], time.strftime("%Y-%m-%d"))

    def test_morning_skips_when_not_warm_restart(self):
        with mock.patch.object(self.mod, "_is_warm_restart", return_value=False), \
             mock.patch.object(self.mod, "play_suit_up_sequence") as seq:
            out = self.mod.maybe_play_morning_suit_up(speak_fn=lambda _l: None)
        self.assertEqual(out, "")
        seq.assert_not_called()

    def test_morning_skips_when_already_fired_today(self):
        # Pre-seed state with today's date → gate blocks even on a warm restart.
        with open(self.statep, "w", encoding="utf-8") as f:
            json.dump({"last_fired_date": time.strftime("%Y-%m-%d")}, f)
        with mock.patch.object(self.mod, "_is_warm_restart", return_value=True), \
             mock.patch.object(self.mod, "play_suit_up_sequence") as seq:
            out = self.mod.maybe_play_morning_suit_up(speak_fn=lambda _l: None)
        self.assertEqual(out, "")
        seq.assert_not_called()


if __name__ == "__main__":
    unittest.main()
