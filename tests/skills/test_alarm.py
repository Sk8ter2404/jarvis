"""skills/alarm.py - a wake-up alarm that rings until someone answers it.

Live 2026-11-12 11:32-11:34 four asks for "an actual alarm that will wake me
up" each became set_timer: one spoken line, then silence.

Nothing real is reached: the tone, the volume, the input tick, the speaker
switch and the speech queue are the module's hooks, replaced here.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from datetime import datetime
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SAVED_DATA_DIR_ENV = None
_DATA_TMPDIR = None


def setUpModule() -> None:
    global _SAVED_DATA_DIR_ENV, _DATA_TMPDIR
    _SAVED_DATA_DIR_ENV = os.environ.get("JARVIS_DATA_DIR")
    _DATA_TMPDIR = tempfile.mkdtemp(prefix="jarvis_alarm_test_")
    os.environ["JARVIS_DATA_DIR"] = _DATA_TMPDIR


def tearDownModule() -> None:
    if _SAVED_DATA_DIR_ENV is None:
        os.environ.pop("JARVIS_DATA_DIR", None)
    else:
        os.environ["JARVIS_DATA_DIR"] = _SAVED_DATA_DIR_ENV
    if _DATA_TMPDIR:
        shutil.rmtree(_DATA_TMPDIR, ignore_errors=True)


class _Rig:
    """The machine, faked: what the alarm did to it, in order."""

    def __init__(self):
        self.events = []
        self.volume = (0.30, True)       # quiet and muted, as at night
        self.tick = 1000
        self.said = []

    def install(self, mod, case):
        def patch(name, fn):
            p = mock.patch.object(mod, name, fn)
            p.start()
            case.addCleanup(p.stop)

        patch("_sound_start", lambda path: self.events.append(("start", os.path.basename(path))))
        patch("_sound_stop", lambda: self.events.append(("stop",)))
        patch("_volume_read", lambda: self.volume)
        patch("_volume_write", self._write)
        patch("_input_tick", lambda: self.tick)
        patch("_to_speakers", lambda: self.events.append(("speakers",)))
        patch("_say", lambda message, key=None: self.said.append(message))

    def _write(self, level, muted):
        self.volume = (float(level), bool(muted))
        self.events.append(("volume", round(float(level), 2), bool(muted)))
        return True


class _AlarmCase(unittest.TestCase):
    def setUp(self):
        for name in ("alarms.json",):
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(_DATA_TMPDIR, name))
        with contextlib.redirect_stdout(io.StringIO()):
            self.mod, self.actions = load_skill_isolated("alarm")
        self.rig = _Rig()
        self.rig.install(self.mod, self)
        self.addCleanup(self._cancel_all)

    def _cancel_all(self):
        self.mod._stop_ringing("test over")
        with self.mod._lock:
            rows = list(self.mod._alarms.values())
            self.mod._alarms.clear()
        for a in rows:
            a["timer"].cancel()

    def _quiet(self, fn, *a, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **kw)

    def _ring_in_thread(self, label=""):
        t = threading.Thread(target=lambda: self._quiet(self.mod.ring, 7, label),
                             daemon=True)
        t.start()
        deadline = time.monotonic() + 5.0
        # Ringing AND the tone started (the first ring writes the tone file).
        while time.monotonic() < deadline and not (
                self.mod.is_ringing()
                and any(e[0] == "start" for e in self.rig.events)):
            time.sleep(0.01)
        self.assertTrue(self.mod.is_ringing())
        return t


class ParseTests(_AlarmCase):
    def test_clock_times_and_durations(self):
        p = self.mod._parse_when
        self.assertEqual(p("20 minutes"), 1200)
        self.assertEqual(p("in an hour"), 3600)
        for text in ("7:00 am", "at 6:30 pm", "07:15"):
            with self.subTest(text=text):
                secs = p(text)
                self.assertIsNotNone(secs)
                self.assertTrue(0 < secs <= 86400)

    def test_a_bare_hour_is_the_next_time_the_clock_reads_it(self):
        now = datetime(2026, 11, 12, 22, 30).timestamp()       # 10:30 PM
        with mock.patch.object(self.mod.time, "time", return_value=now):
            self.assertEqual(self.mod._parse_when("at 7"), int(8.5 * 3600))
            self.assertEqual(self.mod._parse_when("at 11"), 1800)

    def test_no_time_is_none_and_sets_nothing(self):
        self.assertIsNone(self.mod._parse_when("when I get tired"))
        out = self.actions["set_alarm"]("loud please")
        self.assertIn("couldn't tell when", out)
        self.assertEqual(self.mod._snapshot(), [])


class SetListCancelTests(_AlarmCase):
    def test_set_alarm_arms_saves_and_says_when(self):
        out = self.actions["set_alarm"]("20 minutes")
        self.assertIn("Alarm #1 set", out)
        self.assertIn("20 minutes from now", out)
        rows = self.mod._snapshot()
        self.assertEqual(len(rows), 1)
        with open(os.path.join(_DATA_TMPDIR, "alarms.json"), encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved[0]["id"], 1)
        self.assertAlmostEqual(saved[0]["at"], time.time() + 1200, delta=5)
        self.assertIn("One alarm", self.actions["list_alarms"](""))

    def test_cancel_alarm_drops_it_and_the_saved_copy(self):
        self.actions["set_alarm"]("20 minutes")
        self.assertIn("Cancelled", self.actions["cancel_alarm"](""))
        self.assertEqual(self.mod._snapshot(), [])
        with open(os.path.join(_DATA_TMPDIR, "alarms.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f), [])
        self.assertEqual(self.actions["cancel_alarm"](""), "No alarms are set, sir.")


class RingTests(_AlarmCase):
    def test_it_rings_loud_on_the_speakers_until_stopped_then_restores(self):
        t = self._ring_in_thread()
        self.assertEqual(self.rig.events[0], ("speakers",))
        self.assertIn(("volume", self.mod.ALARM_VOLUME, False), self.rig.events)
        self.assertIn(("start", "alarm_tone.wav"), self.rig.events)
        self.assertNotIn(("stop",), self.rig.events, "still ringing")
        self.assertEqual(self.actions["stop_alarm"](""), "Alarm off, sir.")
        t.join(5.0)
        self.assertFalse(self.mod.is_ringing())
        self.assertEqual(self.rig.events[-2:], [("stop",), ("volume", 0.3, True)],
                         "the tone stops and his own volume and mute come back")

    def test_a_louder_volume_is_never_lowered(self):
        self.rig.volume = (0.95, False)
        t = self._ring_in_thread()
        self.assertIn(("volume", 0.95, False), self.rig.events)
        self.mod._stop_ringing("voice")
        t.join(5.0)

    def test_a_key_or_the_mouse_stops_it_after_the_grace(self):
        with mock.patch.object(self.mod, "ALARM_INPUT_GRACE_S", 0.0):
            t = self._ring_in_thread()
            time.sleep(0.6)                 # the loop reads its baseline tick
            self.assertTrue(self.mod.is_ringing())
            self.rig.tick += 50
            t.join(5.0)
        self.assertFalse(self.mod.is_ringing())
        self.assertIn("Alarm off, sir.", self.rig.said)

    def test_it_gives_up_after_the_limit_and_says_so(self):
        with mock.patch.object(self.mod, "ALARM_MAX_RING_S", 0.3):
            t = self._ring_in_thread()
            t.join(5.0)
        self.assertFalse(self.mod.is_ringing())
        self.assertEqual(self.rig.events[-2:], [("stop",), ("volume", 0.3, True)])
        self.assertTrue(any("no answer" in s for s in self.rig.said), self.rig.said)

    def test_snooze_stops_it_and_arms_another(self):
        t = self._ring_in_thread("work")
        out = self.actions["snooze_alarm"]("")
        t.join(5.0)
        self.assertIn(f"{self.mod.ALARM_SNOOZE_MIN} minutes", out)
        rows = self.mod._snapshot()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "work")
        self.assertAlmostEqual(rows[0]["at"],
                               time.time() + self.mod.ALARM_SNOOZE_MIN * 60, delta=5)

    def test_a_failed_tone_still_restores_the_volume_and_tells_him(self):
        def boom(path):
            raise OSError("no audio device")
        with mock.patch.object(self.mod, "_sound_start", boom):
            self._quiet(self.mod.ring, 3, "")
        self.assertFalse(self.mod.is_ringing())
        self.assertEqual(self.rig.volume, (0.30, True))
        self.assertTrue(any("could not ring" in s for s in self.rig.said))


class HeardWhileRingingTests(_AlarmCase):
    def test_nothing_rings_nothing_is_a_command(self):
        for text in ("stop", "snooze", "I'm up"):
            self.assertEqual(self.mod.ringing_command(text), "")
        self.assertIsNone(self.mod.handle_ringing("stop"))

    def test_stop_and_snooze_phrases_while_ringing(self):
        t = self._ring_in_thread()
        cmd = self.mod.ringing_command
        for text in ("Stop.", "okay okay", "I'm up, I'm up", "turn it off",
                     "Jarvis, shut up", "I am awake", "that's enough"):
            with self.subTest(text=text):
                self.assertEqual(cmd(text), "stop")
        for text in ("snooze", "Jarvis snooze it", "five more minutes"):
            with self.subTest(text=text):
                self.assertEqual(cmd(text), "snooze")
        for text in ("what time is it", "is this for medical or dental"):
            with self.subTest(text=text):
                self.assertEqual(cmd(text), "")
        self.assertEqual(self.mod.handle_ringing("I'm up"), "Alarm off, sir.")
        t.join(5.0)
        self.assertFalse(self.mod.is_ringing())


class RestoreTests(_AlarmCase):
    def _save(self, rows):
        with open(os.path.join(_DATA_TMPDIR, "alarms.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f)

    def test_a_pending_alarm_is_re_armed_after_a_restart(self):
        at = time.time() + 3600
        self._save([{"id": 4, "at": at, "label": "class"}])
        self.assertEqual(self._quiet(self.mod.restore_persisted_alarms), 1)
        rows = self.mod._snapshot()
        self.assertEqual((rows[0]["label"], rows[0]["at"]), ("class", at))

    def test_one_missed_by_hours_is_reported_not_rung(self):
        self._save([{"id": 4, "at": time.time() - 5 * 3600, "label": ""}])
        self.assertEqual(self._quiet(self.mod.restore_persisted_alarms), 0)
        self.assertEqual(self.mod._snapshot(), [])
        self.assertTrue(any("never rang" in s for s in self.rig.said))

    def test_the_tone_file_is_a_loopable_wav(self):
        import wave
        path = self.mod._tone_path()
        with wave.open(path, "rb") as w:
            secs = w.getnframes() / w.getframerate()
            self.assertEqual((w.getnchannels(), w.getsampwidth()), (1, 2))
        self.assertTrue(1.5 < secs < 3.5, secs)


if __name__ == "__main__":
    unittest.main()
