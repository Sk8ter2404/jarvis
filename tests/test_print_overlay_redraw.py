"""The two print overlays redraw only when the printer state changes.

WHY THIS EXISTS
  ``hud/workshop_print_monitor.py`` (TICK_MS 250) and ``hud/bambu_h2d_overlay.py``
  (TICK_MS 200) deleted and redrew their whole canvas 4-5 times a second, while
  the bambu_overlay_state.json they draw changes about once a minute
  (GUI_REVIEW B22). Their accent pulse animates on ``self.frame``, so a plain
  "skip if unchanged" would freeze it mid-pulse; the fix is an idle mode: after
  a change the panel pulses for PULSE_AFTER_CHANGE_S, then draws one still
  frame and only polls (at IDLE_TICK_MS) until the state changes again. The
  bookkeeping stamps bambu_monitor rewrites on every report (written_at,
  last_update) are not drawn, so they alone do not count as a change.

ISOLATION
  ``tkinter.Tk`` / ``tkinter.Canvas`` are replaced with MagicMocks for the
  construction, so the REAL ``__init__`` and ``tick()`` run with no display
  (the headless CI runner has none). ``root.after`` is a mock, so nothing
  re-schedules; the test drives each tick by hand on a fake monotonic clock.
  STATE_FILE (and the print monitor's CONTROL_FILE) point into a temp dir, and
  parent_pid 0 means "no parent to watch", so no real process or project file
  is touched.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock


_HUD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hud",
)

_PRINTING = {
    "gcode_state": "RUNNING", "filename": "bracket.3mf",
    "layer_num": 40, "total_layer": 200, "mc_percent": 20,
    "mc_remaining": 95, "nozzle_temper": 220, "bed_temper": 60,
    "chamber_temper": 35, "risk_level": 0, "risk_note": "",
    "last_update": 1000.0, "written_at": 1000.0,
}


def _load(testcase, filename):
    mod_name = f"_{filename[:-3]}_redraw_under_test"
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(_HUD_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    testcase.addCleanup(lambda: sys.modules.pop(mod_name, None))
    spec.loader.exec_module(module)
    return module


class _RedrawBase:
    FILENAME = ""
    CLASS = ""

    def setUp(self):
        self.mod = _load(self, self.FILENAME)
        self.tmp = tempfile.mkdtemp(prefix="print_overlay_redraw_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state_file = os.path.join(self.tmp, "bambu_overlay_state.json")
        self.control_file = os.path.join(self.tmp, "control_state.json")
        self.clock = [0.0]
        for attr, value, create in (
                ("STATE_FILE", self.state_file, False),
                ("CONTROL_FILE", self.control_file, True),
                ("time", types.SimpleNamespace(
                    monotonic=lambda: self.clock[0]), True)):
            p = mock.patch.object(self.mod, attr, value, create=create)
            p.start()
            self.addCleanup(p.stop)
        self._write_state(_PRINTING)
        with mock.patch.object(self.mod.tk, "Tk") as tk_cls, \
                mock.patch.object(self.mod.tk, "Canvas") as canvas_cls:
            self.app = getattr(self.mod, self.CLASS)(0, 0, 400, 200, 0)
        self.root = tk_cls.return_value
        self.canvas = canvas_cls.return_value

    def _write_state(self, data):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _redraws(self):
        return sum(1 for c in self.canvas.delete.call_args_list
                   if c.args == ("all",))

    def _tick_at(self, t):
        self.clock[0] = t
        self.app.tick()

    def _settle(self):
        for t in range(1, 7):
            self._tick_at(float(t))

    def test_no_redraws_while_the_printer_state_is_unchanged(self):
        self._settle()
        before = self._redraws()
        for t in range(7, 30):
            self._tick_at(float(t))
        self.assertEqual(self._redraws(), before,
                         "redrew the canvas with no new printer data")

    def test_a_printer_update_redraws_at_once(self):
        self._settle()
        before = self._redraws()
        self._write_state(dict(_PRINTING, mc_percent=21, layer_num=42))
        self._tick_at(7.0)
        self.assertEqual(self._redraws(), before + 1)

    def test_a_rewrite_with_only_new_timestamps_is_not_a_change(self):
        self._settle()
        before = self._redraws()
        self._write_state(dict(_PRINTING, written_at=1060.0,
                               last_update=1060.0))
        for t in range(7, 12):
            self._tick_at(float(t))
        self.assertEqual(self._redraws(), before)

    def test_the_accent_still_pulses_right_after_a_change(self):
        # Idle mode must not kill the animation: right after new data the
        # panel redraws every tick at the fast cadence.
        self._settle()
        self._write_state(dict(_PRINTING, mc_percent=21))
        before = self._redraws()
        for t in (7.0, 7.25, 7.5, 7.75):
            self._tick_at(t)
            self.assertEqual(self.root.after.call_args.args[0],
                             self.mod.TICK_MS)
        self.assertEqual(self._redraws(), before + 4)

    def test_idle_panel_polls_more_slowly(self):
        self._settle()
        self.assertGreater(self.root.after.call_args.args[0],
                           self.mod.TICK_MS)

    def test_jitter_the_panel_does_not_show_is_not_a_change(self):
        """bambu_monitor rewrites the state file on EVERY MQTT report, with
        raw float temperatures, and the panel draws them rounded. Sub-degree
        jitter, and fields it never draws (stage, print_error, a risk_note at
        risk 0), must not restart the pulse, or the panel animates for the
        whole print (2026-10-02 review)."""
        base = dict(_PRINTING, nozzle_temper=219.84, bed_temper=60.1,
                    chamber_temper=35.2, stage=2, print_error=0)
        self._write_state(base)
        self._settle()
        before = self._redraws()
        for t, (noz, bed, ch) in ((7, (220.12, 59.9, 34.9)),
                                  (8, (219.6, 60.4, 35.4)),
                                  (9, (220.3, 59.7, 35.1))):
            self._write_state(dict(
                base, nozzle_temper=noz, bed_temper=bed, chamber_temper=ch,
                stage=t, print_error=None, risk_note=f"Chamber swing {t}.1",
                written_at=1000.0 + t, last_update=1000.0 + t))
            self._tick_at(float(t))
        self.assertEqual(self._redraws(), before,
                         "redrew for a change the panel does not show")
        self.assertGreater(self.root.after.call_args.args[0],
                           self.mod.TICK_MS, "the pulse restarted")

    def test_a_drawn_temperature_change_redraws(self):
        self._settle()
        before = self._redraws()
        self._write_state(dict(_PRINTING, nozzle_temper=231.2))
        self._tick_at(7.0)
        self.assertEqual(self._redraws(), before + 1)

    def test_a_shown_risk_note_change_redraws(self):
        warn = dict(_PRINTING, risk_level=1, risk_note="Chamber swing 6.0")
        self._write_state(warn)
        self._settle()
        before = self._redraws()
        self._write_state(dict(warn, risk_note="Chamber swing 7.5"))
        self._tick_at(7.0)
        self.assertEqual(self._redraws(), before + 1)


class WorkshopPrintMonitorRedrawTests(_RedrawBase, unittest.TestCase):
    FILENAME = "workshop_print_monitor.py"
    CLASS = "WorkshopPrintMonitor"

    def test_the_live_tick_still_blinks_while_idle(self):
        """The ●/○ beside the state chip is there so a glance tells the panel
        is alive. Idle mode froze it, so a hung panel looked like an idle one;
        each idle poll now flips just that item, with no full redraw."""
        self._settle()
        before = self._redraws()
        texts = []
        for t in range(7, 11):
            self.canvas.itemconfigure.reset_mock()
            self._tick_at(float(t))
            calls = [c for c in self.canvas.itemconfigure.call_args_list
                     if c.args and c.args[0] == "live_tick"]
            self.assertEqual(len(calls), 1, "the live tick did not move")
            texts.append(calls[0].kwargs["text"])
        self.assertEqual(self._redraws(), before)
        self.assertTrue(all(t.startswith("PRINTING") for t in texts), texts)
        dots = [t[-1] for t in texts]
        self.assertTrue(all(d in "●○" for d in dots), texts)
        self.assertTrue(all(a != b for a, b in zip(dots, dots[1:])), texts)

    def test_retire_signal_still_honoured_while_idle(self):
        self._settle()
        with open(self.control_file, "w", encoding="utf-8") as f:
            json.dump({"mode": "off"}, f)
        self._tick_at(30.0)
        self.root.destroy.assert_called_once()


class BambuH2DOverlayRedrawTests(_RedrawBase, unittest.TestCase):
    FILENAME = "bambu_h2d_overlay.py"
    CLASS = "BambuOverlay"


if __name__ == "__main__":
    unittest.main()
