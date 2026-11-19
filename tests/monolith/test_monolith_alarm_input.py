"""A ringing alarm stops on a key or the mouse only when the input is his
(skills/alarm._input_is_his, core/config.py ALARM_INPUT_NEEDS_WITNESS).

Live 2026-11-16 every ring ended "[alarm] #N stopped (input)" a few seconds
in, and every boot logged "LL hook unavailable ... falling back to
GetLastInputInfo polling": the OS input timer was the only reading, and it
counts injected input too (AFK-Helper's clicks, an automation, the Kinect
air-mouse), so a wake-up alarm could be ended with nobody awake. Those three
live stops all had his voice under 30 s before them, so they must still
stop.

Driven through the REAL monolith presence helpers (_physical_input_age_s,
_polled_input_backed) and the REAL skills/_air_mouse_yield on its polling
fallback, with the OS input timer faked to "input just now". The alarm skill
is the real skills/alarm.py, loaded isolated, with every machine hook (tone,
beep, volume, speaker switch, speech queue, COM) replaced: nothing rings.

    python tools/run_tests.py monolith.test_monolith_alarm_input
"""
from __future__ import annotations

import contextlib
import io
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests._skill_harness import load_skill_isolated


@requires_monolith
class AlarmInputWitnessTests(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            self.alarm, _acts = load_skill_isolated("alarm", register=False)
        self.events = []
        self.said = []
        self.tick = [1000]
        al = self.alarm
        # Set for good on this private copy (as tests/skills/test_alarm.py
        # does): a ring thread on its way out must never find a real hook.
        for name, fn in (
                ("_sound_start", lambda path: self.events.append(("start",))),
                ("_sound_stop", lambda: self.events.append(("stop",))),
                ("_beep", lambda: None), ("_com_enter", lambda: None),
                ("_to_speakers", lambda: None),
                ("_volume_read", lambda: (0.2, False)),
                ("_volume_write", lambda *a: True),
                ("_input_tick", lambda: self.tick[0]),
                ("_say", lambda message, key=None: self.said.append(message)),
                ("_persist", lambda: None)):
            setattr(al, name, fn)
        self._p(al, "ALARM_INPUT_GRACE_S", 0.0)
        self.addCleanup(al._stop_ringing, "test over")
        # The real watcher on its polling fallback, the OS timer saying
        # "input just now" (the automation, or him).
        from skills import _air_mouse_yield as y
        self.y = y
        for name, val in (("_installed", True), ("_hook_ok", False),
                          ("_last_self_action_wall", float("-inf")),
                          ("_last_real_input", float("-inf")),
                          ("_last_hook_event", float("-inf"))):
            self._p(y, name, val)
        self._p(y, "_os_last_input_age_s", return_value=0.1)
        self._p(bc, "_yield_watch_mod", return_value=y)
        self._p(bc, "last_face_seen", 0.0)
        self._p(bc, "_last_owner_voice_at", [0.0])

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _owner_spoke(self, seconds_ago):
        self.bc._last_owner_voice_at[0] = self.bc._proactive_mono() - seconds_ago

    def _ring(self):
        self.buf = io.StringIO()

        def run():
            with contextlib.redirect_stdout(self.buf):
                self.alarm.ring(5, "")
        t = threading.Thread(target=run, daemon=True)
        self.addCleanup(t.join, 5.0)
        t.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not (
                self.alarm.is_ringing() and ("start",) in self.events):
            time.sleep(0.01)
        time.sleep(0.6)                     # the loop reads its baseline tick
        self.assertTrue(self.alarm.is_ringing())
        return t

    def _touch(self):
        self.tick[0] += 50
        time.sleep(1.0)

    def test_the_live_defaults(self):
        self.assertIs(self.alarm._monolith(), self.bc)
        self.assertIs(self.bc.ALARM_INPUT_NEEDS_WITNESS, True)
        self.assertEqual(self.alarm.ALARM_INPUT_WITNESS_S, 180.0)
        self.assertEqual(self.y.input_source(), "poll")

    def test_a_key_right_after_he_spoke_stops_it(self):
        # 11-16 11:13:00 ring, 11:12:59 his voice, 11:13:05 stopped (input).
        self._owner_spoke(6.0)
        t = self._ring()
        self._touch()
        t.join(5.0)
        self.assertFalse(self.alarm.is_ringing())
        self.assertIn("Alarm off, sir.", self.said)
        self.assertIn("#5 stopped (input)", self.buf.getvalue())

    def test_a_key_with_his_face_on_camera_stops_it(self):
        self._p(self.bc, "last_face_seen", time.time() - 20.0)
        t = self._ring()
        self._touch()
        t.join(5.0)
        self.assertFalse(self.alarm.is_ringing())

    def test_an_automation_in_an_empty_room_does_not_stop_it(self):
        self._owner_spoke(7 * 3600.0)       # he said "wake me at 7" last night
        t = self._ring()
        self._touch()
        self._touch()
        self.assertTrue(self.alarm.is_ringing(), "injected input ended it")
        self.assertIn("#5 input not counted", self.buf.getvalue())
        # "Stop" by voice always works.
        self.assertEqual(self.alarm.handle_ringing("stop"), "Alarm off, sir.")
        t.join(5.0)
        self.assertFalse(self.alarm.is_ringing())

    def test_the_alarm_window_is_its_own_not_the_presence_one(self):
        # 10 minutes ago is still presence (OWNER_PRESENT_POLLED_INPUT_BACKUP_S
        # is 15 minutes) but not a witness for a ring: a nap alarm set by
        # voice ten minutes before must not be ended by a macro.
        self._owner_spoke(600.0)
        self.assertTrue(self.bc._polled_input_backed())
        self.assertFalse(self.bc._polled_input_backed(
            self.alarm.ALARM_INPUT_WITNESS_S))
        t = self._ring()
        self._touch()
        self.assertTrue(self.alarm.is_ringing())
        self.alarm._stop_ringing("voice")
        t.join(5.0)

    def test_setting_off_is_any_input(self):
        self._p(self.bc, "ALARM_INPUT_NEEDS_WITNESS", False)
        t = self._ring()
        self._touch()
        t.join(5.0)
        self.assertFalse(self.alarm.is_ringing())


if __name__ == "__main__":
    unittest.main()
