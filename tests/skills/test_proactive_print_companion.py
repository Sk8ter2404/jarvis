"""Logic tests for skills/proactive_print_companion.py.

MCU-flavoured commentary layered on bambu_monitor. High-value targets:

  • pure helpers: _format_eta_clock, _infer_material, _bucket_key,
    _next_hedge_suffix, _milestone_commentary, _completion_offer_line
  • pattern persistence: _record_print_outcome / _historical_failure_rate /
    _maybe_warn_historical_failure (all pointed at a temp patterns file)
  • downstream-availability probes: _light_skill_available,
    _timer_skill_available, _vision_available (degrade to False on any doubt)
  • the print_companion_status / print_companion_history actions

The patterns JSON is redirected to a tempfile via patched _PATTERNS_FILE /
_DATA_DIR so the real data/ file is untouched, and _enqueue_speech is patched
so nothing reaches pending_speech.json.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _fake_bambu(state=None, announced=None):
    m = types.ModuleType("skill_bambu_monitor")
    m._state_lock = threading.Lock()
    m._state = state if state is not None else {"last_update": 0.0}
    m._announced_milestones = announced if announced is not None else set()
    m.register_state_change_hook = mock.MagicMock()
    return m


class PrintCompanionMixin:
    def _load(self, bambu_state="__absent__", announced=None):
        # Redirect the patterns file to a fresh temp dir before register() runs.
        self.tmpdir = tempfile.mkdtemp()
        self.patterns_path = os.path.join(self.tmpdir, "patterns.json")
        self.addCleanup(self._cleanup_tmp)

        ctx = []
        if bambu_state != "__absent__":
            fake = _fake_bambu(bambu_state, announced)
            p = mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake})
            p.start(); ctx.append(p)
            self._fake = fake
        else:
            p = mock.patch.dict(sys.modules, {"skill_bambu_monitor": None})
            p.start(); ctx.append(p)
            self._fake = None
        for p in ctx:
            self.addCleanup(p.stop)

        mod, actions = load_skill_isolated("proactive_print_companion")
        # Point persistence at the temp file (register() already ran, but every
        # _load_patterns/_save_patterns reads these module globals live).
        mod._PATTERNS_FILE = self.patterns_path
        mod._DATA_DIR = self.tmpdir
        return mod, actions

    def _cleanup_tmp(self):
        try:
            for f in os.listdir(self.tmpdir):
                os.remove(os.path.join(self.tmpdir, f))
            os.rmdir(self.tmpdir)
        except OSError:
            pass


class PrintCompanionHelperTests(PrintCompanionMixin, unittest.TestCase):
    def test_format_eta_clock(self):
        mod, _a = self._load()
        # 90 minutes from a fixed base → predictable HH:MM.
        base = time.mktime((2026, 6, 1, 10, 0, 0, 0, 0, -1))
        with mock.patch.object(mod.time, "time", return_value=base):
            self.assertEqual(mod._format_eta_clock(90), "11:30")
        self.assertEqual(mod._format_eta_clock(0), "")
        self.assertEqual(mod._format_eta_clock(None), "")

    def test_infer_material_plus_variant_first(self):
        mod, _a = self._load()
        # "pla+" must win over "pla".
        self.assertEqual(mod._infer_material("Bracket_PLA+_BLACK.3mf"), "plaplus")
        self.assertEqual(mod._infer_material("Gear_PETG_4h.3mf"), "petg")
        self.assertEqual(mod._infer_material("Vase_PLA.gcode"), "pla")
        self.assertEqual(mod._infer_material("noname.3mf"), "unknown")
        self.assertEqual(mod._infer_material(""), "unknown")

    def test_bucket_key_coarse_layers(self):
        mod, _a = self._load()
        # 215 and 250 land in the same 200 bucket.
        self.assertEqual(mod._bucket_key("pla", 215), "pla_200")
        self.assertEqual(mod._bucket_key("pla", 250), "pla_200")
        self.assertEqual(mod._bucket_key("petg", None), "petg_0")

    def test_next_hedge_suffix_cycles(self):
        mod, _a = self._load()
        mod._mcu_hedge_idx[0] = 0
        n = len(mod._MCU_HEDGE_SUFFIXES)
        seen = [mod._next_hedge_suffix() for _ in range(n)]
        self.assertEqual(len(set(seen)), n)  # one full unique cycle
        # Wraps back to the first.
        self.assertEqual(mod._next_hedge_suffix(), seen[0])

    def test_milestone_commentary_openers(self):
        mod, _a = self._load()
        base = time.mktime((2026, 6, 1, 10, 0, 0, 0, 0, -1))
        with mock.patch.object(mod.time, "time", return_value=base):
            self.assertIn("Quarter of the way along",
                          mod._milestone_commentary(25, 60))
            self.assertIn("Halfway through", mod._milestone_commentary(50, 60))
            self.assertIn("Three-quarters complete",
                          mod._milestone_commentary(75, 60))

    def test_milestone_commentary_unknown_eta(self):
        mod, _a = self._load()
        out = mod._milestone_commentary(25, None)
        self.assertIn("trajectory still settling", out)


class PrintCompanionCompletionOfferTests(PrintCompanionMixin, unittest.TestCase):
    def test_offer_with_both_lights_and_timer(self):
        mod, _a = self._load()
        with mock.patch.object(mod, "_light_skill_available", return_value=True), \
             mock.patch.object(mod, "_timer_skill_available", return_value=True):
            out = mod._completion_offer_line("cube")
        self.assertIn("cube", out)
        self.assertIn("dim the workshop lights", out)
        self.assertIn("queue a cooldown timer", out)

    def test_offer_lights_only(self):
        mod, _a = self._load()
        with mock.patch.object(mod, "_light_skill_available", return_value=True), \
             mock.patch.object(mod, "_timer_skill_available", return_value=False):
            out = mod._completion_offer_line("cube")
        self.assertIn("dim the workshop lights", out)
        self.assertNotIn("cooldown timer", out)

    def test_offer_neither_falls_back_to_plain(self):
        mod, _a = self._load()
        with mock.patch.object(mod, "_light_skill_available", return_value=False), \
             mock.patch.object(mod, "_timer_skill_available", return_value=False):
            out = mod._completion_offer_line("cube")
        self.assertIn("Print complete", out)
        self.assertNotIn("Shall I", out)


class PrintCompanionPatternsTests(PrintCompanionMixin, unittest.TestCase):
    def test_record_outcome_persists_and_failure_rate(self):
        mod, _a = self._load()
        mod._print_midflight[0] = False
        mod._print_start_ts[0] = time.time() - 600
        state = {"filename": "Gear_PLA.3mf", "total_layer": 150,
                 "print_error": 0}
        mod._record_print_outcome("success", state)
        mod._record_print_outcome("failed", dict(state))
        rate, count = mod._historical_failure_rate("pla", 150)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(rate, 0.5)

    def test_record_outcome_skips_midflight(self):
        mod, _a = self._load()
        mod._print_midflight[0] = True   # mid-flight discovery → no write
        mod._record_print_outcome("success", {"filename": "x_PLA.3mf",
                                              "total_layer": 100})
        rate, count = mod._historical_failure_rate("pla", 100)
        self.assertEqual(count, 0)

    def test_record_outcome_retention_cap(self):
        mod, _a = self._load()
        mod._PER_BUCKET_RETENTION = mod.PER_BUCKET_RETENTION  # use real cap
        mod._print_midflight[0] = False
        for _ in range(mod.PER_BUCKET_RETENTION + 10):
            mod._record_print_outcome("success",
                                      {"filename": "x_PLA.3mf", "total_layer": 100})
        data = mod._load_patterns()
        series = data["buckets"]["pla_100"]
        self.assertEqual(len(series), mod.PER_BUCKET_RETENTION)

    def test_historical_failure_rate_empty_bucket(self):
        mod, _a = self._load()
        self.assertEqual(mod._historical_failure_rate("abs", 999), (0.0, 0))

    def test_warn_historical_failure_fires_above_threshold(self):
        mod, _a = self._load()
        # Seed 4 records, 3 failed → 75% > 30% threshold, count >= 3.
        mod._print_midflight[0] = False
        for outcome in ("failed", "failed", "failed", "success"):
            mod._record_print_outcome(outcome,
                                      {"filename": "Gear_PLA.3mf",
                                       "total_layer": 150})
        mod._warned_historical_failure[0] = False
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._maybe_warn_historical_failure({"filename": "Gear_PLA.3mf",
                                               "total_layer": 150})
        enq.assert_called_once()
        self.assertIn("PLA", enq.call_args.args[0])
        self.assertIn("%", enq.call_args.args[0])

    def test_warn_historical_failure_silent_below_threshold(self):
        mod, _a = self._load()
        mod._print_midflight[0] = False
        for outcome in ("success", "success", "success", "failed"):  # 25% < 30%
            mod._record_print_outcome(outcome,
                                      {"filename": "Gear_PLA.3mf",
                                       "total_layer": 150})
        mod._warned_historical_failure[0] = False
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._maybe_warn_historical_failure({"filename": "Gear_PLA.3mf",
                                               "total_layer": 150})
        enq.assert_not_called()

    def test_warn_historical_failure_silent_insufficient_history(self):
        mod, _a = self._load()
        mod._print_midflight[0] = False
        # Only 1 record (< MIN_HISTORY_FOR_WARNING) even though it failed.
        mod._record_print_outcome("failed",
                                  {"filename": "Gear_PLA.3mf", "total_layer": 150})
        mod._warned_historical_failure[0] = False
        with mock.patch.object(mod, "_enqueue_speech") as enq:
            mod._maybe_warn_historical_failure({"filename": "Gear_PLA.3mf",
                                               "total_layer": 150})
        enq.assert_not_called()


class PrintCompanionAvailabilityTests(PrintCompanionMixin, unittest.TestCase):
    def test_light_available_requires_devices(self):
        mod, _a = self._load()
        hue = types.ModuleType("skill_sh_hue")
        hue.is_available = lambda: True
        hue.list_devices = lambda: [{"id": "bulb1"}]
        with mock.patch.dict(sys.modules, {"skill_sh_hue": hue}):
            self.assertTrue(mod._light_skill_available())
        # Available but zero devices → not available.
        hue.list_devices = lambda: []
        with mock.patch.dict(sys.modules, {"skill_sh_hue": hue,
                                           "skill_sh_govee": None}):
            self.assertFalse(mod._light_skill_available())

    def test_light_available_false_when_no_module(self):
        mod, _a = self._load()
        with mock.patch.dict(sys.modules,
                             {"skill_sh_hue": None, "skill_sh_govee": None}):
            self.assertFalse(mod._light_skill_available())

    def test_light_available_swallows_probe_errors(self):
        mod, _a = self._load()
        hue = types.ModuleType("skill_sh_hue")
        hue.is_available = lambda: True
        def _boom():
            raise RuntimeError("bridge offline")
        hue.list_devices = _boom
        with mock.patch.dict(sys.modules, {"skill_sh_hue": hue,
                                           "skill_sh_govee": None}):
            self.assertFalse(mod._light_skill_available())

    def test_timer_available_false_without_module(self):
        mod, _a = self._load()
        with mock.patch.dict(sys.modules, {"skill_timer": None}):
            self.assertFalse(mod._timer_skill_available())

    def test_vision_available_false_without_module(self):
        mod, _a = self._load()
        with mock.patch.dict(sys.modules, {"skill_local_vision": None}):
            self.assertFalse(mod._vision_available())


class PrintCompanionActionTests(PrintCompanionMixin, unittest.TestCase):
    def test_status_when_bambu_absent(self):
        mod, actions = self._load()  # no bambu
        out = actions["print_companion_status"]("")
        self.assertIn("monitor isn't running", out.lower())

    def test_status_armed_no_state(self):
        mod, actions = self._load(bambu_state={"last_update": 0.0})
        out = actions["print_companion_status"]("")
        self.assertIn("no fresh printer", out.lower())

    def test_status_tracking_running(self):
        mod, actions = self._load(bambu_state={
            "last_update": time.time(), "gcode_state": "RUNNING",
            "filename": "widget_PLA.3mf", "total_layer": 150})
        out = actions["print_companion_status"]("")
        self.assertIn("widget", out)
        self.assertIn("RUNNING", out)

    def test_history_empty(self):
        mod, actions = self._load()
        self.assertIn("No print history", actions["print_companion_history"](""))

    def test_history_renders_buckets(self):
        mod, actions = self._load()
        mod._print_midflight[0] = False
        mod._record_print_outcome("success",
                                  {"filename": "a_PLA.3mf", "total_layer": 100})
        mod._record_print_outcome("failed",
                                  {"filename": "b_PLA.3mf", "total_layer": 100})
        out = actions["print_companion_history"]("")
        self.assertIn("pla_100", out)
        self.assertIn("1 success / 1 failed", out)


if __name__ == "__main__":
    unittest.main()
