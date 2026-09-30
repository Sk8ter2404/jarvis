"""NIGHT_QUIET_ENABLED in the skills (see tests/test_night_quiet.py for the
switch itself and the core gates). ON, the shipped default, keeps the old
behaviour; OFF, the clock alone never makes JARVIS quieter or chattier at
night:

  * skills/anticipation_engine.py: no late_hour nudge ("It's 11:02 PM, sir.
    We've been at this a while ...") and so no 'late_hour' trigger written to
    anticipation_state.json (core/tts.py reads that for the hushed_late voice);
    and no 23:00-06:59 hold on its other lines just because the owner has been
    silent for 30 minutes (the same silence does not hold them by day);
  * skills/night_owl_mode.py: the real watcher loop never switches night-owl
    mode on by itself (NIGHT_OWL_AUTO is the finer knob; see
    tests/skills/test_night_owl_auto.py). "night owl on" still works.

Every test patches the knob explicitly and pins the clock.

    python -m unittest tests.skills.test_night_quiet_skills
"""
from __future__ import annotations

import unittest
from unittest import mock

from core import config as cfg
from tests._skill_harness import load_skill_isolated
from tests.skills.test_anticipation_engine import (
    _EngineTestBase, _StopLoop, _sleep_after, _struct)


def _quiet(on):
    return mock.patch.object(cfg, "NIGHT_QUIET_ENABLED", on, create=True)


# ─── 4. the anticipation engine's late_hour nudge ────────────────────────
class LateHourNudgeTests(_EngineTestBase):

    def _nudge_at(self, hour, minute):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(hour, minute)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=60.0):
            return self.mod._try_late_hour_active()

    def test_on_nudges_at_23_02(self):
        with _quiet(True):
            line = self._nudge_at(23, 2)
        self.assertIn("11:02 PM", line)
        self.assertIn("at this a while", line)

    def test_off_never_nudges_at_night(self):
        with _quiet(False):
            for hour, minute in ((23, 2), (2, 15), (6, 45)):
                self.assertEqual(self._nudge_at(hour, minute), "", (hour, minute))

    def _one_scheduler_pass(self, *, hour=23, minute=2, age=60.0, offer=""):
        """Run the REAL _scheduler_loop for one poll with every other gate
        open, at hour:minute with the owner last heard `age` seconds ago and
        `offer` as the pattern offer (every other trigger empty). The default
        is 23:02 with the owner active and no offer.
        Returns (enqueue mock, the state dict _save_state received or None)."""
        enqueue = mock.MagicMock()
        saved = {}
        cfg_on = {"enabled": True, "cooldown": 20}
        # _StopLoop is an Exception, so the loop's own handler logs it once
        # before the recovery sleep ends the loop: keep that out of the output.
        with mock.patch.object(self.mod.logging, "exception"), \
             mock.patch.object(self.mod.time, "sleep",
                               side_effect=_sleep_after(2)), \
             mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(hour, minute)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=age), \
             mock.patch.object(self.mod, "_read_config", return_value=cfg_on), \
             mock.patch.object(self.mod, "_focused_window_title", return_value=""), \
             mock.patch.object(self.mod, "_is_sleep_or_standby", return_value=False), \
             mock.patch.object(self.mod, "_is_in_call", return_value=False), \
             mock.patch.object(self.mod, "_user_at_desk", return_value=None), \
             mock.patch.object(self.mod, "_load_state", return_value={}), \
             mock.patch.object(self.mod, "_try_pattern_offer", return_value=offer), \
             mock.patch.object(self.mod, "_try_long_dwell", return_value=("", "")), \
             mock.patch.object(self.mod.random, "random", return_value=0.0), \
             mock.patch.object(self.mod, "_enqueue_speech", enqueue), \
             mock.patch.object(self.mod, "_save_state",
                               side_effect=lambda s: saved.update(s)) as save:
            with self.assertRaises(_StopLoop):
                self.mod._scheduler_loop()
        return enqueue, (saved if save.called else None)

    def test_on_the_loop_speaks_it_and_records_late_hour(self):
        with _quiet(True):
            enqueue, saved = self._one_scheduler_pass()
        enqueue.assert_called_once()
        self.assertIn("11:02 PM", enqueue.call_args[0][0])
        self.assertEqual(saved["last_trigger"], "late_hour")

    def test_off_the_loop_says_nothing_and_writes_no_late_hour(self):
        with _quiet(False):
            enqueue, saved = self._one_scheduler_pass()
        enqueue.assert_not_called()
        self.assertIsNone(saved, "no nudge, so no state write at all")


# ─── 6. the 23:00-06:59 hold on the engine's OTHER lines ─────────────────
class LateNightHoldTests(_EngineTestBase):
    """_should_skip_late_night: from 23:00 to 06:59 every anticipation line
    (pattern offer, long dwell) is held once the owner has been silent for 30
    minutes. By day the same silence holds nothing, so with the knob off the
    night follows the daytime gates (the presence gate still applies)."""

    OFFER = "You usually put the kettle on about now, sir."
    SILENT = 45 * 60.0          # silent for 45 minutes

    # The same real-loop driver as the nudge tests above.
    _one_scheduler_pass = LateHourNudgeTests._one_scheduler_pass

    def _skip_at(self, hour, age):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(hour, 30)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=age):
            return self.mod._should_skip_late_night()

    def test_on_a_silent_owner_holds_lines_at_night_only(self):
        with _quiet(True):
            self.assertTrue(self._skip_at(23, self.SILENT))
            self.assertTrue(self._skip_at(3, None))
            self.assertFalse(self._skip_at(14, self.SILENT))

    def test_off_the_night_holds_nothing_the_day_would_not(self):
        with _quiet(False):
            for hour in (23, 0, 3, 6):
                for age in (self.SILENT, None):
                    self.assertFalse(self._skip_at(hour, age), (hour, age))

    def test_on_the_loop_holds_a_03_30_pattern_offer(self):
        with _quiet(True):
            enqueue, saved = self._one_scheduler_pass(
                hour=3, minute=30, age=self.SILENT, offer=self.OFFER)
        enqueue.assert_not_called()
        self.assertIsNone(saved)

    def test_off_the_loop_speaks_a_03_30_pattern_offer_like_a_daytime_one(self):
        with _quiet(False):
            night, night_saved = self._one_scheduler_pass(
                hour=3, minute=30, age=self.SILENT, offer=self.OFFER)
            day, _day_saved = self._one_scheduler_pass(
                hour=14, minute=30, age=self.SILENT, offer=self.OFFER)
        self.assertEqual(night.call_args_list, day.call_args_list)
        night.assert_called_once()
        self.assertEqual(night.call_args[0][0], self.OFFER)
        self.assertEqual(night_saved["last_trigger"], "pattern")


# ─── 1. night-owl mode's automatic 23:00 switch-on (real watcher loop) ───
class NightOwlWatcherMasterSwitchTests(unittest.TestCase):

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("night_owl_mode")
        m = self.mod
        m._opted_out_night[0] = ""
        m._night_owl_active[0] = False
        m._engaged_in_window[0] = False
        m._trigger[0] = ""

    def _one_watch_pass(self, *, quiet, auto=True):
        """One pass of the REAL _watch_loop inside the night window: the
        startup sleep no-ops, the in-loop sleep ends the loop."""
        calls = {"n": 0}

        def _sleep(_secs):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise KeyboardInterrupt

        with _quiet(quiet), \
             mock.patch.object(cfg, "NIGHT_OWL_AUTO", auto, create=True), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod, "_in_night_window", return_value=True), \
             mock.patch.object(self.mod, "_enter_night_owl") as ent, \
             mock.patch("builtins.print"):
            with self.assertRaises(KeyboardInterrupt):
                self.mod._watch_loop()
        return ent

    def test_on_engages_by_itself_at_night(self):
        self._one_watch_pass(quiet=True).assert_called_once_with(trigger="auto")

    def test_off_never_engages_by_itself(self):
        self._one_watch_pass(quiet=False).assert_not_called()

    def test_off_manual_night_owl_on_still_works(self):
        with _quiet(False), \
             mock.patch.object(self.mod, "_enter_night_owl") as ent:
            self.actions["night_owl_on"]("")
        ent.assert_called_once_with(trigger="manual")


if __name__ == "__main__":
    unittest.main()
