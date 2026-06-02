"""Logic tests for core.voice_mood_selector — the context→mood router.

select_mood(context) returns one of four mood strings by a documented
first-match-wins priority: a HIGH-severity self-diagnostic problem outranks
everything, then chappie/banter conversational modes, then a VIP intercept
during work hours, else the calm_efficient default. The work-hours check is
driven by an explicit `now` epoch (or a `work_hours` override) so the suite is
deterministic regardless of wall-clock time.

stdlib unittest only.
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

from core import voice_mood_selector as vms


def _epoch_at(hour: int) -> float:
    """A local-time epoch on a fixed date at the given hour, so localtime()
    inside the module resolves to that hour regardless of the test host's TZ."""
    return time.mktime((2026, 5, 29, hour, 0, 0, 0, 0, -1))


class DefaultTests(unittest.TestCase):
    def test_empty_context_is_calm(self):
        self.assertEqual(vms.select_mood({}), vms.CALM_EFFICIENT)

    def test_none_context_is_calm(self):
        self.assertEqual(vms.select_mood(None), vms.CALM_EFFICIENT)

    def test_unknown_keys_ignored(self):
        self.assertEqual(vms.select_mood({"banana": True}), vms.CALM_EFFICIENT)


class PriorityTests(unittest.TestCase):
    def test_diagnostic_problem_outranks_all(self):
        ctx = {"self_diagnostic_problem": True, "vip_intercept": True,
               "work_hours": True, "chappie_mode": True}
        self.assertEqual(vms.select_mood(ctx), vms.CONCERNED_SOFT)

    def test_chappie_mode_is_dry_amused(self):
        self.assertEqual(vms.select_mood({"chappie_mode": True}), vms.DRY_AMUSED)

    def test_banter_flag_is_dry_amused(self):
        self.assertEqual(vms.select_mood({"banter": True}), vms.DRY_AMUSED)

    def test_conversation_mode_chappie(self):
        self.assertEqual(vms.select_mood({"conversation_mode": "chappie"}),
                         vms.DRY_AMUSED)

    def test_conversation_mode_banter_case_insensitive(self):
        self.assertEqual(vms.select_mood({"conversation_mode": "  BANTER "}),
                         vms.DRY_AMUSED)

    def test_conversation_mode_default_is_calm(self):
        self.assertEqual(vms.select_mood({"conversation_mode": "default"}),
                         vms.CALM_EFFICIENT)

    def test_chappie_beats_vip(self):
        # Conversational mode is checked before the VIP intercept branch.
        ctx = {"chappie_mode": True, "vip_intercept": True, "work_hours": True}
        self.assertEqual(vms.select_mood(ctx), vms.DRY_AMUSED)


class VipInterceptTests(unittest.TestCase):
    def test_vip_in_work_hours_override_true(self):
        self.assertEqual(
            vms.select_mood({"vip_intercept": True, "work_hours": True}),
            vms.URGENT_CLIPPED)

    def test_vip_outside_work_hours_override_false(self):
        self.assertEqual(
            vms.select_mood({"vip_intercept": True, "work_hours": False}),
            vms.CALM_EFFICIENT)

    def test_vip_uses_clock_when_no_override_daytime(self):
        self.assertEqual(
            vms.select_mood({"vip_intercept": True, "now": _epoch_at(10)}),
            vms.URGENT_CLIPPED)

    def test_vip_uses_clock_when_no_override_night(self):
        self.assertEqual(
            vms.select_mood({"vip_intercept": True, "now": _epoch_at(22)}),
            vms.CALM_EFFICIENT)

    def test_vip_without_work_context_falls_through(self):
        # No work_hours override and a night clock → not urgent.
        self.assertEqual(
            vms.select_mood({"vip_intercept": True, "now": _epoch_at(3)}),
            vms.CALM_EFFICIENT)


class WorkHoursHelperTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertTrue(vms._is_work_hours(_epoch_at(8)))     # inclusive start
        self.assertTrue(vms._is_work_hours(_epoch_at(17)))
        self.assertFalse(vms._is_work_hours(_epoch_at(18)))   # exclusive end
        self.assertFalse(vms._is_work_hours(_epoch_at(7)))
        self.assertFalse(vms._is_work_hours(_epoch_at(23)))

    def test_localtime_failure_is_not_work_hours(self):
        # A bad clock value (localtime raises, e.g. OverflowError on an
        # out-of-range epoch) degrades to "not work hours" rather than raising.
        with mock.patch.object(vms.time, "localtime",
                               side_effect=OverflowError("bad ts")):
            self.assertFalse(vms._is_work_hours(1.0))

    def test_returned_moods_are_all_valid(self):
        # Whatever select_mood emits is always a member of VALID_MOODS.
        for ctx in ({}, {"self_diagnostic_problem": True}, {"banter": True},
                    {"vip_intercept": True, "work_hours": True}):
            self.assertIn(vms.select_mood(ctx), vms.VALID_MOODS)


if __name__ == "__main__":
    unittest.main()
