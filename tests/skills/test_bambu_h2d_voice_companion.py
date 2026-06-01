"""Logic tests for skills/bambu_h2d_voice_companion.py.

In-character H2D announcements layered on bambu_monitor's state-change hook.
We test:

  • formatter fallbacks (_format_minutes / _format_temp / _strip_filename) that
    work even when bambu_monitor isn't loaded
  • _scan_for_layer_shift / _scan_for_ams_issue keyword detectors
  • _process_snapshot — milestone (gated) vs layer-shift / AMS / FAILED
    (direct, rate-limit-bypassing) announcements + per-print dedup
  • the print_status action: offline / no-fresh-state / idle / finish / running
  • _register_bambu_hook graceful degradation

All speech routing is patched (_gated_announce / _direct_enqueue) so nothing
hits pending_speech.json, and bambu_monitor is represented by a fake module so
no real printer state is read.
"""
from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _fake_bambu(state=None):
    m = types.ModuleType("skill_bambu_monitor")
    m._state_lock = threading.Lock()
    m._state = state if state is not None else {"last_update": 0.0}
    m.register_state_change_hook = mock.MagicMock()
    return m


class VoiceCompanionMixin:
    def _load(self, bambu_state="__absent__"):
        patches = []
        if bambu_state != "__absent__":
            fake = _fake_bambu(bambu_state)
            p = mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake})
            p.start()
            patches.append(p)
            self._fake = fake
        else:
            # Ensure no stale bambu module is visible.
            p = mock.patch.dict(sys.modules, {"skill_bambu_monitor": None})
            p.start()
            patches.append(p)
            self._fake = None
        for p in patches:
            self.addCleanup(p.stop)
        mod, actions = load_skill_isolated("bambu_h2d_voice_companion")
        # Reset per-print bookkeeping.
        mod._current_filename[0] = None
        mod._announced_milestones.clear()
        mod._announced_error_codes.clear()
        mod._announced_layer_shift[0] = False
        mod._announced_ams_error[0] = False
        mod._announced_failed[0] = False
        return mod, actions


class VoiceCompanionFormatTests(VoiceCompanionMixin, unittest.TestCase):
    def test_format_minutes_fallback_without_bambu(self):
        mod, _a = self._load()  # bambu absent → local fallback path
        self.assertEqual(mod._format_minutes(5), "5 minutes")
        self.assertEqual(mod._format_minutes(125), "2 hours and 5 minutes")
        self.assertEqual(mod._format_minutes(0), "")
        self.assertEqual(mod._format_minutes(None), "")

    def test_format_temp_fallback(self):
        mod, _a = self._load()
        self.assertEqual(mod._format_temp(219.6), "220 degrees")
        self.assertEqual(mod._format_temp(0), "")

    def test_strip_filename_fallback(self):
        mod, _a = self._load()
        self.assertEqual(mod._strip_filename("My_Part.gcode"), "My Part")
        self.assertEqual(mod._strip_filename(""), "")


class VoiceCompanionScanTests(VoiceCompanionMixin, unittest.TestCase):
    def test_layer_shift_detected_in_error(self):
        mod, _a = self._load()
        self.assertTrue(mod._scan_for_layer_shift("layer shift on plate 1", None))
        self.assertTrue(mod._scan_for_layer_shift(None, "axis shifted"))

    def test_layer_shift_absent(self):
        mod, _a = self._load()
        self.assertFalse(mod._scan_for_layer_shift(0, "all nominal"))

    def test_ams_issue_requires_fault_signature(self):
        mod, _a = self._load()
        # The literal word "ams" in a healthy block must NOT trip.
        self.assertFalse(mod._scan_for_ams_issue(0, "ams tray 1 ready"))
        # A fault keyword alongside an AMS keyword does.
        self.assertTrue(mod._scan_for_ams_issue(0, "ams spool jam"))

    def test_ams_issue_none_payload(self):
        mod, _a = self._load()
        self.assertFalse(mod._scan_for_ams_issue("0", None))


class VoiceCompanionSnapshotTests(VoiceCompanionMixin, unittest.TestCase):
    def test_milestone_routed_through_gated_announce(self):
        mod, _a = self._load()
        mod._current_filename[0] = "cube"
        # 25 already spoken (one milestone per snapshot); crossing 50 announces
        # the 50 line and routes it through the gated path, not direct.
        mod._announced_milestones.add(25)
        snap = {"filename": "cube.3mf", "mc_percent": 50, "mc_remaining": 120}
        with mock.patch.object(mod, "_gated_announce") as gated, \
             mock.patch.object(mod, "_direct_enqueue") as direct:
            with mod._state_lock:
                mod._process_snapshot(snap, "RUNNING")
        msgs = [c.args[0] for c in gated.call_args_list]
        self.assertTrue(any("Print at 50%" in m for m in msgs))
        self.assertIn("2 hours", msgs[0])
        direct.assert_not_called()
        self.assertIn(50, mod._announced_milestones)

    def test_milestone_deduped(self):
        mod, _a = self._load()
        mod._current_filename[0] = "cube"
        snap = {"filename": "cube.3mf", "mc_percent": 25, "mc_remaining": 60}
        with mock.patch.object(mod, "_gated_announce") as gated:
            with mod._state_lock:
                mod._process_snapshot(snap, "RUNNING")
                mod._process_snapshot(snap, "RUNNING")
        self.assertEqual(sum("25%" in c.args[0] for c in gated.call_args_list), 1)

    def test_layer_shift_uses_direct_enqueue(self):
        mod, _a = self._load()
        mod._current_filename[0] = "cube"
        snap = {"filename": "cube.3mf", "print_error": "layer shift detected",
                "layer_num": 88}
        with mock.patch.object(mod, "_gated_announce"), \
             mock.patch.object(mod, "_direct_enqueue") as direct:
            with mod._state_lock:
                mod._process_snapshot(snap, "RUNNING")
        msgs = [c.args[0] for c in direct.call_args_list]
        self.assertTrue(any("Layer shift detected" in m for m in msgs))
        self.assertTrue(any("layer 88" in m for m in msgs))
        self.assertTrue(mod._announced_layer_shift[0])

    def test_ams_unwell_direct_enqueue(self):
        mod, _a = self._load()
        mod._current_filename[0] = "cube"
        snap = {"filename": "cube.3mf", "ams_status": "ams spool jam"}
        with mock.patch.object(mod, "_gated_announce"), \
             mock.patch.object(mod, "_direct_enqueue") as direct:
            with mod._state_lock:
                mod._process_snapshot(snap, "RUNNING")
        self.assertTrue(any("AMS appears to be unwell" in c.args[0]
                            for c in direct.call_args_list))

    def test_failed_state_direct_enqueue(self):
        mod, _a = self._load()
        mod._current_filename[0] = "cube"
        snap = {"filename": "cube.3mf", "layer_num": 142}
        with mock.patch.object(mod, "_gated_announce"), \
             mock.patch.object(mod, "_direct_enqueue") as direct:
            with mod._state_lock:
                mod._process_snapshot(snap, "FAILED")
        hit = [c.args[0] for c in direct.call_args_list
               if "failed" in c.args[0].lower()]
        self.assertTrue(hit)
        self.assertIn("layer 142", hit[0])
        self.assertTrue(mod._announced_failed[0])

    def test_new_filename_resets_bookkeeping(self):
        mod, _a = self._load()
        mod._current_filename[0] = "old"
        mod._announced_milestones.add(25)
        mod._announced_layer_shift[0] = True
        snap = {"filename": "new_part.3mf", "mc_percent": 5}
        with mock.patch.object(mod, "_gated_announce"), \
             mock.patch.object(mod, "_direct_enqueue"):
            with mod._state_lock:
                mod._process_snapshot(snap, "RUNNING")
        # Reset on the filename change.
        self.assertNotIn(25, mod._announced_milestones)
        self.assertFalse(mod._announced_layer_shift[0])


class VoiceCompanionStatusActionTests(VoiceCompanionMixin, unittest.TestCase):
    def test_status_offline_when_bambu_absent(self):
        mod, actions = self._load()  # no bambu module
        out = actions["print_status"]("")
        self.assertIn("monitor isn't running", out.lower())

    def test_status_no_fresh_state(self):
        mod, actions = self._load(bambu_state={"last_update": 0.0})
        out = actions["print_status"]("")
        self.assertIn("fresh status", out.lower())

    def test_status_idle(self):
        mod, actions = self._load(
            bambu_state={"last_update": time.time(), "gcode_state": "IDLE"})
        self.assertIn("No active print", actions["print_status"](""))

    def test_status_finish(self):
        mod, actions = self._load(bambu_state={
            "last_update": time.time(), "gcode_state": "FINISH",
            "filename": "cube.3mf"})
        self.assertIn("finished", actions["print_status"]("").lower())

    def test_status_running_with_temps(self):
        mod, actions = self._load(bambu_state={
            "last_update": time.time(), "gcode_state": "RUNNING",
            "filename": "widget.3mf", "layer_num": 10, "total_layer": 100,
            "mc_remaining": 30, "nozzle_temper": 220.0, "bed_temper": 60.0})
        out = actions["print_status"]("")
        self.assertIn("widget", out)
        self.assertIn("layer 10 of 100", out)
        self.assertIn("nozzle at 220 degrees", out)
        self.assertIn("bed at 60 degrees", out)


class VoiceCompanionHookTests(VoiceCompanionMixin, unittest.TestCase):
    def test_register_hook_when_bambu_present(self):
        mod, _a = self._load(bambu_state={"last_update": 0.0})
        # The fake exposes register_state_change_hook as a MagicMock.
        ok = mod._register_bambu_hook()
        self.assertTrue(ok)
        self.assertTrue(mod._hook_registered[0])
        self._fake.register_state_change_hook.assert_called()

    def test_register_hook_when_bambu_absent(self):
        mod, _a = self._load()  # no bambu
        self.assertFalse(mod._register_bambu_hook())

    def test_hook_callback_swallows_errors(self):
        mod, _a = self._load(bambu_state={"last_update": time.time()})
        # If _process_snapshot raises, the hook must not propagate it.
        with mock.patch.object(mod, "_process_snapshot",
                               side_effect=RuntimeError("boom")):
            # Should not raise.
            mod._on_bambu_state_change({"filename": "x"}, "IDLE", "RUNNING")


if __name__ == "__main__":
    unittest.main()
