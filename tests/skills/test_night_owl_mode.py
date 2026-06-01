"""Logic tests for skills/night_owl_mode.py.

Targets the pure pieces and the toggle logic the spec calls out, without
relying on the live monolith (which the skill monkeypatches at runtime):
  • _in_night_window — the wrap-around 23:00–06:00 gate (time controlled).
  • _adjust_rate_string — slow an edge-tts rate string by N percentage points.
  • the TTS-preset wrapper installed by _install_tts_modifier — gain ×0.85 and
    rate −5pp, applied to a stub _resolve_tts_preset (no real TTS).
  • enter/exit state machine + is_night_owl_active().
  • the good_morning action (releases the mode when active; plain greeting else)
    and night_owl_status.

All side effects (TTS install/restore, nudge suppression, overlay dim, prompt
addendum, announcement enqueue) are patched so only state transitions run.
"""
from __future__ import annotations

import contextlib
import unittest
from unittest import mock
from datetime import datetime

from tests._skill_harness import load_skill_isolated


@contextlib.contextmanager
def _neutered(mod):
    with mock.patch.object(mod, "_install_tts_modifier"), \
         mock.patch.object(mod, "_restore_tts_modifier"), \
         mock.patch.object(mod, "_install_nudge_suppressors"), \
         mock.patch.object(mod, "_restore_nudge_suppressors"), \
         mock.patch.object(mod, "_apply_prompt_addendum"), \
         mock.patch.object(mod, "_restore_prompt_addendum"), \
         mock.patch.object(mod, "_set_overlay_dim"), \
         mock.patch.object(mod, "_enqueue_speech"):
        yield


def _dt(hour):
    return datetime(2026, 6, 1, hour, 0, 0)


class NightOwlWindowTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("night_owl_mode")

    def test_in_window_late_night(self):
        self.assertTrue(self.mod._in_night_window(_dt(23)))
        self.assertTrue(self.mod._in_night_window(_dt(2)))
        self.assertTrue(self.mod._in_night_window(_dt(5)))

    def test_out_of_window_daytime(self):
        self.assertFalse(self.mod._in_night_window(_dt(6)))
        self.assertFalse(self.mod._in_night_window(_dt(12)))
        self.assertFalse(self.mod._in_night_window(_dt(22)))


class NightOwlRateTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("night_owl_mode")

    def test_adjust_rate_string_slows(self):
        self.assertEqual(self.mod._adjust_rate_string("+6%", 5), "+1%")
        self.assertEqual(self.mod._adjust_rate_string("-12%", 5), "-17%")
        self.assertEqual(self.mod._adjust_rate_string("+0%", 5), "-5%")

    def test_adjust_rate_string_handles_malformed(self):
        # Non-percent input → just the negative delta.
        self.assertEqual(self.mod._adjust_rate_string("fast", 5), "-5%")
        self.assertEqual(self.mod._adjust_rate_string("+x%", 5), "-5%")


class NightOwlTtsWrapperTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("night_owl_mode")
        self.mod._saved_resolve_preset[0] = None

    def test_wrapper_scales_gain_and_slows_rate(self):
        # Provide a fake bobert_companion with a stub _resolve_tts_preset, then
        # let _install_tts_modifier wrap it and inspect the wrapped output.
        import sys
        fake_bc = mock.MagicMock()
        fake_bc._resolve_tts_preset = lambda text, tone: (
            "base", {"rate": "+0%", "gain": 1.0})
        with mock.patch.dict(sys.modules, {"bobert_companion": fake_bc}):
            with mock.patch.object(self.mod.importlib, "import_module",
                                   return_value=fake_bc):
                self.mod._install_tts_modifier()
                name, preset = fake_bc._resolve_tts_preset("hello", None)
        self.assertTrue(name.endswith("_nightowl"))
        self.assertAlmostEqual(preset["gain"], 0.85, places=3)
        self.assertEqual(preset["rate"], "-5%")

    def test_wrapper_idempotent_install(self):
        fake_bc = mock.MagicMock()
        fake_bc._resolve_tts_preset = lambda text, tone: ("base", {})
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=fake_bc):
            self.mod._install_tts_modifier()
            saved = self.mod._saved_resolve_preset[0]
            # A second install must not re-wrap (saved original unchanged).
            self.mod._install_tts_modifier()
        self.assertIs(self.mod._saved_resolve_preset[0], saved)


class NightOwlStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("night_owl_mode")
        self.mod._night_owl_active[0] = False
        self.mod._trigger[0] = ""
        self.mod._started_at[0] = 0.0

    def test_enter_activates(self):
        with _neutered(self.mod):
            out = self.mod._enter_night_owl(trigger="manual")
        self.assertTrue(self.mod.is_night_owl_active())
        self.assertIn("engaged", out.lower())

    def test_enter_idempotent(self):
        with _neutered(self.mod):
            self.mod._enter_night_owl(trigger="manual")
            out = self.mod._enter_night_owl(trigger="auto")
        self.assertIn("already engaged", out.lower())

    def test_exit_deactivates(self):
        with _neutered(self.mod):
            self.mod._enter_night_owl(trigger="manual")
            out = self.mod._exit_night_owl(trigger="manual")
        self.assertFalse(self.mod.is_night_owl_active())
        self.assertIn("disengaged", out.lower())

    def test_exit_when_inactive(self):
        with _neutered(self.mod):
            out = self.mod._exit_night_owl(trigger="manual")
        self.assertIn("was not active", out.lower())

    def test_exit_good_morning_message(self):
        with _neutered(self.mod):
            self.mod._enter_night_owl(trigger="auto")
            out = self.mod._exit_night_owl(trigger="phrase_good_morning")
        self.assertIn("good morning", out.lower())


class NightOwlActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("night_owl_mode")
        self.mod._night_owl_active[0] = False
        self.mod._trigger[0] = ""
        self.mod._started_at[0] = 0.0

    def test_night_owl_on_off_actions(self):
        with _neutered(self.mod):
            on = self.actions["night_owl_on"]("")
            self.assertTrue(self.mod.is_night_owl_active())
            off = self.actions["night_owl_off"]("")
        self.assertIn("engaged", on.lower())
        self.assertIn("disengaged", off.lower())
        self.assertFalse(self.mod.is_night_owl_active())

    def test_good_morning_releases_active_mode(self):
        with _neutered(self.mod):
            self.actions["night_owl_on"]("")
            out = self.actions["good_morning"]("")
        self.assertIn("good morning", out.lower())
        self.assertFalse(self.mod.is_night_owl_active())

    def test_good_morning_plain_when_not_active(self):
        out = self.actions["good_morning"]("")
        self.assertEqual(out, "Good morning, sir.")

    def test_status_inactive(self):
        self.assertIn("not currently engaged",
                      self.actions["night_owl_status"]("").lower())

    def test_status_active_reports_trigger(self):
        with _neutered(self.mod):
            self.actions["night_owl_on"]("")
        out = self.actions["night_owl_status"]("")
        self.assertIn("engaged", out.lower())
        self.assertIn("by voice", out)   # manual trigger label


if __name__ == "__main__":
    unittest.main()
