"""Logic tests for skills/system_pulse.py.

A fleet-wide status snapshot (CPU/RAM/GPU/disk/net/battery/uptime/apps + Bambu +
credits) rendered as a JARVIS sentence, plus a proactive abnormality detector
and a compact HUD strip. We test the pure renderers and detectors with fully
controlled pulse dicts, the cross-skill collectors (_read_bambu_status across
sys.modules, _read_credit_balance off a temp file), and the system_pulse action
serving the cache vs a live gather.

Both background threads (_hud_publish_loop, _proactive_loop) are never started
(harness neuters threads). No nvidia-smi / net sample / real file write occurs.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class PulseFormatTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    # ── _fmt_uptime ──────────────────────────────────────────────────────
    def test_fmt_uptime_days(self):
        self.assertEqual(self.mod._fmt_uptime(2 * 86400 + 3 * 3600),
                         "2 days 3 hours")

    def test_fmt_uptime_hours(self):
        self.assertEqual(self.mod._fmt_uptime(3 * 3600 + 15 * 60),
                         "3 hours 15 minutes")

    def test_fmt_uptime_minutes(self):
        self.assertEqual(self.mod._fmt_uptime(42 * 60), "42 minutes")

    def test_fmt_uptime_unknown(self):
        self.assertEqual(self.mod._fmt_uptime(0), "unknown")

    # ── _fmt_bambu ───────────────────────────────────────────────────────
    def test_fmt_bambu_running_hours_and_pct(self):
        out = self.mod._fmt_bambu({"gcode_state": "RUNNING",
                                   "hours_into": 3.0, "percent": 42})
        self.assertIn("3 hours into its print", out)
        self.assertIn("42 percent", out)

    def test_fmt_bambu_failed(self):
        self.assertIn("failed",
                      self.mod._fmt_bambu({"gcode_state": "FAILED"}).lower())

    def test_fmt_bambu_empty(self):
        self.assertEqual(self.mod._fmt_bambu({}), "")
        self.assertEqual(self.mod._fmt_bambu({"gcode_state": "IDLE"}), "")

    # ── _format_report ───────────────────────────────────────────────────
    def test_format_report_default_opener_and_tail(self):
        pulse = {"cpu_pct": 12.0, "ram_pct": 40.0, "gpu_temp_c": 55.0,
                 "disk_free_gb": 500.0, "active_apps": 8}
        out = self.mod._format_report(pulse)
        self.assertTrue(out.startswith("All systems nominal, sir."))
        self.assertIn("CPU 12 percent, memory 40 percent", out)
        self.assertIn("GPU idling at 55 degrees", out)
        self.assertIn("8 windows open", out)
        self.assertTrue(out.endswith("Anything further?"))

    def test_format_report_lead_replaces_opener(self):
        pulse = {"cpu_pct": 90.0, "ram_pct": 50.0}
        out = self.mod._format_report(pulse, lead="Slight problem, sir — CPU at 90 percent.")
        self.assertTrue(out.startswith("Slight problem, sir"))
        self.assertNotIn("All systems nominal", out)

    def test_format_report_gpu_hot_phrasing(self):
        pulse = {"cpu_pct": 10.0, "ram_pct": 30.0, "gpu_temp_c": 85.0}
        out = self.mod._format_report(pulse)
        self.assertIn("GPU running hot at 85 degrees", out)

    def test_format_report_includes_credits_and_bambu(self):
        pulse = {"cpu_pct": 10.0, "ram_pct": 30.0, "credits_dollars": 12.50,
                 "bambu": {"gcode_state": "RUNNING", "percent": 20}}
        out = self.mod._format_report(pulse)
        self.assertIn("$12.50", out)
        self.assertIn("20 percent into a print", out)

    def test_format_report_battery_only_on_battery_power(self):
        on_batt = {"cpu_pct": 10.0, "ram_pct": 30.0, "battery_pct": 55.0,
                   "battery_plugged": False}
        self.assertIn("battery at 55 percent on battery power",
                      self.mod._format_report(on_batt))
        plugged = dict(on_batt, battery_plugged=True)
        self.assertNotIn("battery at 55", self.mod._format_report(plugged))

    # ── _format_hud_strip ────────────────────────────────────────────────
    def test_hud_strip_compact_metrics(self):
        pulse = {"gpu_temp_c": 60.0, "battery_pct": 80.0,
                 "battery_plugged": True, "uptime_seconds": 3 * 3600 + 5 * 60,
                 "active_apps": 12, "net_down_kbps": 0.0, "net_up_kbps": 0.0}
        out = self.mod._format_hud_strip(pulse)
        self.assertIn("GPU 60C", out)
        self.assertIn("BAT 80%+", out)   # plugged → "+"
        self.assertIn("APPS 12", out)

    def test_hud_strip_network_only_when_busy(self):
        quiet = {"net_down_kbps": 10.0, "net_up_kbps": 10.0}
        self.assertNotIn("NET", self.mod._format_hud_strip(quiet))
        busy = {"net_down_kbps": 2048.0, "net_up_kbps": 100.0}
        self.assertIn("NET", self.mod._format_hud_strip(busy))
        self.assertIn("MB/s", self.mod._format_hud_strip(busy))


class PulseAbnormalReasonTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    def test_high_cpu_flagged(self):
        reasons = dict(self.mod._abnormal_reasons({"cpu_pct": 90.0}))
        self.assertIn("cpu", reasons)
        self.assertIn("90 percent", reasons["cpu"])

    def test_low_disk_flagged(self):
        reasons = dict(self.mod._abnormal_reasons(
            {"disk_free_gb": 5.0}))
        self.assertIn("disk", reasons)

    def test_disk_zero_not_flagged(self):
        # 0 GB means "reading unavailable" (the 0 < free guard), not low disk.
        reasons = dict(self.mod._abnormal_reasons({"disk_free_gb": 0.0}))
        self.assertNotIn("disk", reasons)

    def test_hot_gpu_flagged(self):
        reasons = dict(self.mod._abnormal_reasons({"gpu_temp_c": 85.0}))
        self.assertIn("gpu", reasons)

    def test_low_battery_only_when_unplugged(self):
        unplugged = {"battery_pct": 10.0, "battery_plugged": False}
        self.assertIn("battery", dict(self.mod._abnormal_reasons(unplugged)))
        plugged = {"battery_pct": 10.0, "battery_plugged": True}
        self.assertNotIn("battery", dict(self.mod._abnormal_reasons(plugged)))

    def test_failed_bambu_flagged(self):
        reasons = dict(self.mod._abnormal_reasons(
            {"bambu": {"gcode_state": "FAILED"}}))
        self.assertIn("bambu_fail", reasons)

    def test_low_credits_flagged(self):
        reasons = dict(self.mod._abnormal_reasons({"credits_dollars": 3.0}))
        self.assertIn("credits", reasons)

    def test_nothing_abnormal(self):
        ok = {"cpu_pct": 10.0, "ram_pct": 30.0, "disk_free_gb": 500.0,
              "gpu_temp_c": 50.0, "credits_dollars": 50.0}
        self.assertEqual(self.mod._abnormal_reasons(ok), [])


class PulseBambuStatusTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    def _fake_bambu(self, **state):
        m = types.ModuleType("skill_bambu_monitor")
        m._state_lock = threading.Lock()
        base = {"last_update": 0.0}
        base.update(state)
        m._state = base
        return m

    def test_empty_when_absent(self):
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": None}):
            self.assertEqual(self.mod._read_bambu_status(), {})

    def test_empty_when_no_fresh_state(self):
        fake = self._fake_bambu(last_update=0.0, gcode_state="RUNNING")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._read_bambu_status(), {})

    def test_running_computes_hours_into(self):
        # 25% done with 90 min remaining → total = 90*100/25 = 360 min,
        # elapsed = 360-90 = 270 min = 4.5h.
        fake = self._fake_bambu(last_update=time.time(), gcode_state="RUNNING",
                                mc_percent=25, mc_remaining=90)
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._read_bambu_status()
        self.assertEqual(out["gcode_state"], "RUNNING")
        self.assertEqual(out["percent"], 25)
        self.assertAlmostEqual(out["hours_into"], 4.5, places=2)


class PulseCreditBalanceTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    def _write_credits(self, balance, age_seconds):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"balance": balance,
                       "checked_at": time.time() - age_seconds}, f)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        return p

    def test_fresh_balance_read(self):
        p = self._write_credits(12.34, age_seconds=60)
        with mock.patch.object(self.mod, "_CREDITS_STATE", p):
            self.assertAlmostEqual(self.mod._read_credit_balance(), 12.34, places=2)

    def test_stale_balance_ignored(self):
        p = self._write_credits(12.34, age_seconds=25 * 3600)  # > 24h
        with mock.patch.object(self.mod, "_CREDITS_STATE", p):
            self.assertIsNone(self.mod._read_credit_balance())

    def test_missing_file(self):
        with mock.patch.object(self.mod, "_CREDITS_STATE",
                               os.path.join(tempfile.gettempdir(),
                                            "no_such_pulse_credits.json")):
            self.assertIsNone(self.mod._read_credit_balance())


class PulseActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")
        with self.mod._pulse_cache_lock:
            self.mod._pulse_cache = None

    def test_action_serves_fresh_cache(self):
        cached_pulse = {"cpu_pct": 5.0, "ram_pct": 20.0}
        with self.mod._pulse_cache_lock:
            self.mod._pulse_cache = {"pulse": cached_pulse, "ts": time.time()}
        # _gather_pulse must NOT be called when the cache is fresh.
        with mock.patch.object(self.mod, "_gather_pulse",
                               side_effect=AssertionError("should not gather")):
            out = self.actions["system_pulse"]("")
        self.assertIn("CPU 5 percent", out)

    def test_action_live_gather_when_cache_stale(self):
        with self.mod._pulse_cache_lock:
            self.mod._pulse_cache = {"pulse": {"cpu_pct": 1.0, "ram_pct": 1.0},
                                     "ts": time.time() - 999}  # stale
        live = {"cpu_pct": 77.0, "ram_pct": 60.0}
        with mock.patch.object(self.mod, "_gather_pulse", return_value=live):
            out = self.actions["system_pulse"]("")
        self.assertIn("CPU 77 percent", out)

    def test_action_wraps_exceptions(self):
        with self.mod._pulse_cache_lock:
            self.mod._pulse_cache = None
        with mock.patch.object(self.mod, "_gather_pulse",
                               side_effect=RuntimeError("boom")):
            out = self.actions["system_pulse"]("")
        self.assertIn("system pulse failed", out.lower())

    def test_status_report_is_alias(self):
        self.assertIs(self.actions["status_report"], self.actions["system_pulse"])


if __name__ == "__main__":
    unittest.main()
