"""Logic tests for skills/dnd_focus_mode.py.

Covers the duration parser, the minutes formatter, the rate/registry-free
parts of the focus-mode state machine (enter → extend → exit), the
is_focus_mode_active() helper other skills poll, and the registered actions.

All OS / network side effects are patched out: _set_focus_assist (reg.exe),
_set_teams_presence (Graph), nudge suppression, the prompt addendum, the
expiry thread, and the announcement enqueue. We assert on the returned
strings and the in-memory mode flags — never on real registry/Teams state.
"""
from __future__ import annotations

import contextlib
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


@contextlib.contextmanager
def _neutered_side_effects(mod):
    """Patch every external side effect of enter/exit so only the state
    machine runs."""
    with mock.patch.object(mod, "_set_focus_assist", return_value=True), \
         mock.patch.object(mod, "_set_teams_presence", return_value=True), \
         mock.patch.object(mod, "_install_nudge_suppressors"), \
         mock.patch.object(mod, "_restore_nudge_suppressors"), \
         mock.patch.object(mod, "_apply_prompt_addendum"), \
         mock.patch.object(mod, "_restore_prompt_addendum"), \
         mock.patch.object(mod, "_start_expiry_thread"), \
         mock.patch.object(mod, "_enqueue_speech"):
        yield


class FocusDurationParseTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dnd_focus_mode")

    def test_parse_units(self):
        p = self.mod._parse_duration_to_seconds
        self.assertEqual(p("90 minutes"), 5400)
        self.assertEqual(p("1 hour 30 min"), 5400)
        self.assertEqual(p("45m"), 2700)
        self.assertEqual(p("2 hours"), 7200)
        self.assertEqual(p("30 seconds"), 30)

    def test_parse_bare_number_is_minutes(self):
        self.assertEqual(self.mod._parse_duration_to_seconds("90"), 5400)

    def test_parse_invalid(self):
        self.assertIsNone(self.mod._parse_duration_to_seconds("soon"))
        self.assertIsNone(self.mod._parse_duration_to_seconds(""))

    def test_format_minutes(self):
        f = self.mod._format_minutes
        self.assertEqual(f(30), "30 seconds")
        self.assertEqual(f(90), "2 minutes")
        self.assertEqual(f(60), "1 minute")
        self.assertEqual(f(3600), "1 hour")
        self.assertEqual(f(5400), "1 hour 30 minutes")


class FocusStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dnd_focus_mode")
        # Ensure a clean inactive baseline (module-global lists).
        self.mod._focus_active[0] = False
        self.mod._focus_ends_at[0] = 0.0
        self.mod._focus_trigger[0] = ""

    def test_initially_inactive(self):
        self.assertFalse(self.mod.is_focus_mode_active())

    def test_enter_sets_active_and_returns_summary(self):
        with _neutered_side_effects(self.mod):
            already, msg = self.mod._enter_focus_mode(5400, trigger="voice")
        self.assertFalse(already)
        self.assertTrue(self.mod.is_focus_mode_active())
        self.assertIn("Holding all non-critical interruptions", msg)
        self.assertIn("1 hour 30 minutes", msg)

    def test_reenter_extends_not_restarts(self):
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(600, trigger="voice")
            already, msg = self.mod._enter_focus_mode(3600, trigger="voice")
        self.assertTrue(already)
        self.assertIn("Already in focus mode", msg)
        self.assertIn("extended", msg.lower())

    def test_duration_floor_enforced(self):
        # < 60s is clamped up to 60s minimum.
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(5, trigger="voice")
        remaining = self.mod._focus_ends_at[0] - self.mod._focus_started_at[0]
        self.assertGreaterEqual(remaining, 60)

    def test_duration_cap_enforced(self):
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(99 * 3600, trigger="voice")
        remaining = self.mod._focus_ends_at[0] - self.mod._focus_started_at[0]
        self.assertLessEqual(remaining, self.mod.MAX_DURATION_SECONDS)

    def test_exit_clears_active(self):
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(600, trigger="voice")
            msg = self.mod._exit_focus_mode(reason="manual")
        self.assertFalse(self.mod.is_focus_mode_active())
        self.assertIn("disengaged", msg.lower())

    def test_exit_when_inactive_is_graceful(self):
        msg = self.mod._exit_focus_mode(reason="manual")
        self.assertIn("was not active", msg.lower())

    def test_exit_expired_message(self):
        with _neutered_side_effects(self.mod):
            self.mod._enter_focus_mode(600, trigger="voice")
            msg = self.mod._exit_focus_mode(reason="expired")
        self.assertIn("complete", msg.lower())


class FocusActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("dnd_focus_mode")
        self.mod._focus_active[0] = False
        self.mod._focus_ends_at[0] = 0.0
        self.mod._focus_trigger[0] = ""

    def test_focus_mode_action_default_duration(self):
        with _neutered_side_effects(self.mod):
            out = self.actions["focus_mode"]("")   # blank → default 60 min
        self.assertIn("Holding all non-critical", out)
        self.assertTrue(self.mod.is_focus_mode_active())

    def test_focus_mode_action_custom_duration(self):
        with _neutered_side_effects(self.mod):
            out = self.actions["focus_mode"]("90 minutes")
        self.assertIn("1 hour 30 minutes", out)

    def test_end_focus_mode_action(self):
        with _neutered_side_effects(self.mod):
            self.actions["focus_mode"]("30 minutes")
            out = self.actions["end_focus_mode"]("")
        self.assertIn("disengaged", out.lower())
        self.assertFalse(self.mod.is_focus_mode_active())

    def test_status_inactive(self):
        self.assertIn("not currently engaged",
                      self.actions["focus_mode_status"]("").lower())

    def test_status_active_reports_remaining_and_trigger(self):
        with _neutered_side_effects(self.mod):
            self.actions["focus_mode"]("90 minutes")
        out = self.actions["focus_mode_status"]("")
        self.assertIn("engaged", out.lower())
        self.assertIn("by voice", out)
        self.assertIn("remaining", out.lower())


if __name__ == "__main__":
    unittest.main()
