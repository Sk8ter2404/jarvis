"""Logic tests for skills/anticipation_engine.py.

Targets the pure decision logic the proactive scheduler is built from:
duration/clock formatting, productivity-window detection + app-name shortening,
in-call suppression, the late-night gate, dwell tracking, and the individual
trigger pickers (_try_long_dwell / _try_late_hour_active). Also covers the
anticipation_status action and the speech-queue writer.

The background scheduler thread is neutered by the harness; we never let it run.
bobert_companion / face_tracker are absent (sys.modules lookups return None),
which the skill treats as the permissive default — exactly the path we assert.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _struct(hour, minute):
    return time.struct_time((2026, 6, 1, hour, minute, 0, 0, 152, -1))


class AnticipationHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("anticipation_engine")

    # ── formatting ───────────────────────────────────────────────────────
    def test_format_hours_minutes(self):
        f = self.mod._format_hours_minutes
        self.assertEqual(f(0), "0 minutes")
        self.assertEqual(f(600), "10 minutes")
        self.assertEqual(f(3600), "1 hour")
        self.assertEqual(f(7200), "2 hours")
        self.assertEqual(f(7800), "2 hours and 10 minutes")

    def test_format_clock(self):
        self.assertEqual(self.mod._format_clock(_struct(9, 5)), "9:05 AM")
        self.assertEqual(self.mod._format_clock(_struct(0, 0)), "12:00 AM")
        self.assertEqual(self.mod._format_clock(_struct(13, 30)), "1:30 PM")

    # ── window helpers ───────────────────────────────────────────────────
    def test_is_productivity_window(self):
        self.assertTrue(self.mod._is_productivity_window("untitled - Blender"))
        self.assertTrue(self.mod._is_productivity_window(
            "part.f3d - Autodesk Fusion 360"))
        self.assertFalse(self.mod._is_productivity_window("Solitaire"))
        self.assertFalse(self.mod._is_productivity_window(""))

    def test_shorten_app_name_known_hint(self):
        # The longest matching hint wins: "visual studio code" → title-cased.
        self.assertEqual(
            self.mod._shorten_app_name("project - Visual Studio Code"),
            "Visual Studio Code")
        # OpenSCAD is in the acronym set, so it stays styled.
        self.assertEqual(
            self.mod._shorten_app_name("model.scad - OpenSCAD"), "OpenSCAD")
        # The "vscode" hint maps to the "VS Code" acronym form.
        self.assertEqual(
            self.mod._shorten_app_name("foo - vscode"), "VS Code")

    def test_shorten_app_name_separator_fallback(self):
        # No known hint → take the tail after the last separator.
        self.assertEqual(
            self.mod._shorten_app_name("Inbox - SomeMailApp"), "SomeMailApp")

    def test_title_case_app_acronyms(self):
        self.assertEqual(self.mod._title_case_app("vscode"), "VS Code")
        self.assertEqual(self.mod._title_case_app("freecad"), "FreeCAD")
        self.assertEqual(self.mod._title_case_app("blender"), "Blender")

    def test_call_window_hints_present(self):
        # The shared call-hint list must include the major platforms.
        joined = " ".join(self.mod.CALL_WINDOW_HINTS)
        self.assertIn("zoom meeting", joined)
        self.assertIn("discord call", joined)

    # ── in-call detection ────────────────────────────────────────────────
    def test_is_in_call_matches_title(self):
        with mock.patch.object(self.mod, "_all_window_titles",
                               return_value=["Weekly sync | Microsoft Teams Meeting"]):
            self.assertTrue(self.mod._is_in_call())

    def test_is_in_call_false_when_no_meeting(self):
        with mock.patch.object(self.mod, "_all_window_titles",
                               return_value=["Inbox - Outlook", "Notepad"]):
            self.assertFalse(self.mod._is_in_call())

    def test_is_in_call_false_when_no_windows(self):
        with mock.patch.object(self.mod, "_all_window_titles", return_value=[]):
            self.assertFalse(self.mod._is_in_call())

    # ── absent-dependency defaults ───────────────────────────────────────
    def test_user_at_desk_none_without_tracker(self):
        # No face_tracker module loaded → None (permissive, not "away").
        self.assertIsNone(self.mod._user_at_desk())

    def test_sleep_or_standby_false_without_bc(self):
        # With bobert_companion absent, the gate reads False (permissive).
        import sys
        with mock.patch.dict(sys.modules, {"bobert_companion": None}):
            self.assertFalse(self.mod._is_sleep_or_standby())

    def test_last_speech_age_none_without_bc(self):
        # No bobert_companion → no last_speech_time → None.
        import sys
        with mock.patch.dict(sys.modules, {"bobert_companion": None}):
            self.assertIsNone(self.mod._last_speech_age_seconds())


class AnticipationDwellTriggerTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("anticipation_engine")
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = ""
            self.mod._dwell_state["started_at"] = 0.0
            self.mod._dwell_state["last_seen"] = 0.0

    def test_update_dwell_starts_and_continues(self):
        self.mod._update_dwell("model.scad - OpenSCAD")
        win, dwell = self.mod._current_dwell_seconds()
        self.assertEqual(win, "model.scad - OpenSCAD")
        self.assertGreaterEqual(dwell, 0.0)
        # Same window again keeps started_at (dwell keeps accruing).
        start_before = self.mod._dwell_state["started_at"]
        self.mod._update_dwell("model.scad - OpenSCAD")
        self.assertEqual(self.mod._dwell_state["started_at"], start_before)

    def test_update_dwell_resets_on_window_change(self):
        self.mod._update_dwell("A - Blender")
        first_start = self.mod._dwell_state["started_at"]
        self.mod._update_dwell("B - VS Code")
        self.assertEqual(self.mod._dwell_state["window"], "B - VS Code")
        # New window → started_at refreshed (>= previous).
        self.assertGreaterEqual(self.mod._dwell_state["started_at"], first_start)

    def test_try_long_dwell_requires_threshold(self):
        # Productivity window but only a few seconds of dwell → no line.
        self.mod._update_dwell("model.scad - OpenSCAD")
        line, key = self.mod._try_long_dwell({})
        self.assertEqual(line, "")
        self.assertEqual(key, "")

    def test_try_long_dwell_fires_after_long_session(self):
        # Force a 3-hour dwell on a productivity window.
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = "model.scad - OpenSCAD"
            self.mod._dwell_state["started_at"] = time.time() - 3 * 3600
            self.mod._dwell_state["last_seen"] = time.time()
        line, key = self.mod._try_long_dwell({})
        self.assertIn("OpenSCAD", line)
        self.assertTrue(key.startswith("dwell:"))

    def test_try_long_dwell_respects_repeat_gap(self):
        app = "OpenSCAD"
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = "model.scad - OpenSCAD"
            self.mod._dwell_state["started_at"] = time.time() - 3 * 3600
            self.mod._dwell_state["last_seen"] = time.time()
        # Remarked on this app moments ago → suppressed by LONG_DWELL_REPEAT_GAP.
        state = {"last_dwell_remark_at": {app: time.time() - 60}}
        line, key = self.mod._try_long_dwell(state)
        self.assertEqual(line, "")

    def test_try_long_dwell_ignores_non_productivity(self):
        with self.mod._dwell_lock:
            self.mod._dwell_state["window"] = "Solitaire"
            self.mod._dwell_state["started_at"] = time.time() - 3 * 3600
            self.mod._dwell_state["last_seen"] = time.time()
        line, key = self.mod._try_long_dwell({})
        self.assertEqual(line, "")


class AnticipationLateHourTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("anticipation_engine")

    def test_late_hour_active_fires_when_recently_spoke(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(23, 30)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=120.0):
            line = self.mod._try_late_hour_active()
        self.assertIn("stretch", line.lower())

    def test_late_hour_active_silent_when_idle(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(2, 0)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=99999.0):
            self.assertEqual(self.mod._try_late_hour_active(), "")

    def test_late_hour_active_silent_during_day(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(14, 0)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=60.0):
            self.assertEqual(self.mod._try_late_hour_active(), "")

    def test_should_skip_late_night_no_activity_skips(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(3, 0)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=None):
            self.assertTrue(self.mod._should_skip_late_night())

    def test_should_skip_late_night_false_during_day(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(12, 0)):
            self.assertFalse(self.mod._should_skip_late_night())

    def test_should_skip_late_night_recent_speech_does_not_skip(self):
        with mock.patch.object(self.mod.time, "localtime",
                               return_value=_struct(23, 30)), \
             mock.patch.object(self.mod, "_last_speech_age_seconds",
                               return_value=60.0):
            self.assertFalse(self.mod._should_skip_late_night())


class AnticipationStatusAndQueueTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("anticipation_engine")

    def test_status_no_fires_yet(self):
        with mock.patch.object(self.mod, "_load_state", return_value={}):
            out = self.actions["anticipation_status"]("")
        self.assertIn("no fires yet", out.lower())
        self.assertTrue(out.startswith("Anticipation engine"))

    def test_status_reports_last_fire(self):
        state = {"last_proactive_at": time.time() - 300}
        with mock.patch.object(self.mod, "_load_state", return_value=state):
            out = self.actions["anticipation_status"]("")
        self.assertIn("last fire", out.lower())

    def test_enqueue_speech_writes_to_temp_queue(self):
        fd, qp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with mock.patch.object(self.mod, "_SPEECH_QUEUE", qp):
                self.mod._enqueue_speech("Sir, a brief stretch would not go amiss.")
            with open(qp, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertIn("stretch", data[0]["message"])
        finally:
            os.unlink(qp)


if __name__ == "__main__":
    unittest.main()
