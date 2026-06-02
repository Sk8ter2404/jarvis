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


# ─── psutil stand-ins ────────────────────────────────────────────────────
class _VM:
    def __init__(self, percent, used_gb):
        self.percent = percent
        self.used = int(used_gb * 1024 ** 3)


class _DiskFree:
    def __init__(self, free_gb):
        self.free = int(free_gb * 1024 ** 3)


class _Net:
    def __init__(self, recv, sent):
        self.bytes_recv = recv
        self.bytes_sent = sent


class _Batt:
    def __init__(self, percent, plugged):
        self.percent = percent
        self.power_plugged = plugged


# ─────────────────────────────────────────────────────────────────────────
# metric collectors — psutil mocked, no real readings.
# ─────────────────────────────────────────────────────────────────────────
class PulseCollectorTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    # ── _read_cpu_ram ────────────────────────────────────────────────────
    def test_cpu_ram_with_psutil(self):
        fake = mock.MagicMock()
        fake.cpu_percent.return_value = 33.0
        fake.virtual_memory.return_value = _VM(55.0, 9.0)
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            cpu, ram, used = self.mod._read_cpu_ram()
        self.assertEqual((cpu, ram), (33.0, 55.0))
        self.assertAlmostEqual(used, 9.0, places=3)

    def test_cpu_ram_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertEqual(self.mod._read_cpu_ram(), (0.0, 0.0, 0.0))

    # ── _read_disk_free_gb ───────────────────────────────────────────────
    def test_disk_free_with_psutil(self):
        fake = mock.MagicMock()
        fake.disk_usage.return_value = _DiskFree(250.0)
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertAlmostEqual(self.mod._read_disk_free_gb(), 250.0, places=2)

    def test_disk_free_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertEqual(self.mod._read_disk_free_gb(), 0.0)

    def test_disk_free_exception_returns_zero(self):
        fake = mock.MagicMock()
        fake.disk_usage.side_effect = OSError("no C drive")
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertEqual(self.mod._read_disk_free_gb(), 0.0)

    # ── _read_network_rates ──────────────────────────────────────────────
    def test_network_rates_computes_kbps(self):
        fake = mock.MagicMock()
        # +614400 bytes recv over 0.6s = 1024000 B/s = 1000 kB/s.
        fake.net_io_counters.side_effect = [_Net(0, 0), _Net(614400, 307200)]
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "sleep"):
            down, up = self.mod._read_network_rates()
        self.assertAlmostEqual(down, 1000.0, places=0)
        self.assertAlmostEqual(up, 500.0, places=0)

    def test_network_rates_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertEqual(self.mod._read_network_rates(), (0.0, 0.0))

    def test_network_rates_exception(self):
        fake = mock.MagicMock()
        fake.net_io_counters.side_effect = RuntimeError("nic gone")
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "sleep"):
            self.assertEqual(self.mod._read_network_rates(), (0.0, 0.0))

    # ── _read_battery ────────────────────────────────────────────────────
    def test_battery_present(self):
        fake = mock.MagicMock()
        fake.sensors_battery.return_value = _Batt(72.0, False)
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertEqual(self.mod._read_battery(), (72.0, False))

    def test_battery_desktop_none(self):
        fake = mock.MagicMock()
        fake.sensors_battery.return_value = None
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertIsNone(self.mod._read_battery())

    def test_battery_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertIsNone(self.mod._read_battery())

    def test_battery_exception(self):
        fake = mock.MagicMock()
        fake.sensors_battery.side_effect = RuntimeError("no acpi")
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertIsNone(self.mod._read_battery())

    # ── _read_uptime_seconds ─────────────────────────────────────────────
    def test_uptime_positive(self):
        fake = mock.MagicMock()
        fake.boot_time.return_value = 100.0
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake), \
             mock.patch.object(self.mod.time, "time", return_value=3700.0):
            self.assertAlmostEqual(self.mod._read_uptime_seconds(), 3600.0)

    def test_uptime_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertEqual(self.mod._read_uptime_seconds(), 0.0)

    def test_uptime_exception(self):
        fake = mock.MagicMock()
        fake.boot_time.side_effect = RuntimeError("x")
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertEqual(self.mod._read_uptime_seconds(), 0.0)


# ─────────────────────────────────────────────────────────────────────────
# _read_gpu_temp_c — nvidia-smi + psutil-sensors fallback.
# ─────────────────────────────────────────────────────────────────────────
class PulseGpuTempTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    def test_nvidia_smi_hottest_gpu(self):
        proc = types.SimpleNamespace(stdout="61\n74\n")
        with mock.patch.object(self.mod.shutil, "which", return_value="nvidia-smi"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertEqual(self.mod._read_gpu_temp_c(), 74.0)

    def test_nvidia_smi_nondigit_lines_skipped(self):
        proc = types.SimpleNamespace(stdout="N/A\n\n68\n")
        with mock.patch.object(self.mod.shutil, "which", return_value="nvidia-smi"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertEqual(self.mod._read_gpu_temp_c(), 68.0)

    def test_no_nvidia_smi_falls_back_to_psutil_sensors(self):
        entry = types.SimpleNamespace(current=70.0)
        fake = mock.MagicMock()
        fake.sensors_temperatures.return_value = {"nvidia_gpu": [entry]}
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertEqual(self.mod._read_gpu_temp_c(), 70.0)

    def test_no_source_returns_none(self):
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertIsNone(self.mod._read_gpu_temp_c())

    def test_nvidia_smi_exception_then_no_psutil(self):
        with mock.patch.object(self.mod.shutil, "which", return_value="nvidia-smi"), \
             mock.patch.object(self.mod.subprocess, "run",
                               side_effect=OSError("spawn failed")), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertIsNone(self.mod._read_gpu_temp_c())

    def test_sensors_no_gpu_label_returns_none(self):
        entry = types.SimpleNamespace(current=40.0)
        fake = mock.MagicMock()
        fake.sensors_temperatures.return_value = {"coretemp": [entry]}
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertIsNone(self.mod._read_gpu_temp_c())

    def test_sensors_gpu_label_but_no_current_returns_none(self):
        # GPU-labelled sensor group whose entries have no usable `current` →
        # the temps list is empty, so no reading is produced.
        entry = types.SimpleNamespace(current=None)
        fake = mock.MagicMock()
        fake.sensors_temperatures.return_value = {"amdgpu": [entry]}
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertIsNone(self.mod._read_gpu_temp_c())

    def test_sensors_exception_returns_none(self):
        fake = mock.MagicMock()
        fake.sensors_temperatures.side_effect = RuntimeError("no sensors")
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertIsNone(self.mod._read_gpu_temp_c())


# ─────────────────────────────────────────────────────────────────────────
# _read_active_app_count — pygetwindow gated.
# ─────────────────────────────────────────────────────────────────────────
class PulseAppCountTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    def _win(self, title, visible=True):
        return types.SimpleNamespace(title=title, visible=visible)

    def test_counts_distinct_titled_visible_windows(self):
        wins = [self._win("VS Code"), self._win("Chrome"),
                self._win("Chrome"),               # dup title → counted once
                self._win("", visible=True),        # blank title skipped
                self._win("Hidden", visible=False), # not visible skipped
                self._win("Program Manager")]       # system title skipped
        fake_gw = types.SimpleNamespace(getAllWindows=lambda: wins)
        with mock.patch.object(self.mod, "_HAS_GW", True), \
             mock.patch.object(self.mod, "gw", fake_gw, create=True):
            self.assertEqual(self.mod._read_active_app_count(), 2)

    def test_without_gw_returns_zero(self):
        with mock.patch.object(self.mod, "_HAS_GW", False):
            self.assertEqual(self.mod._read_active_app_count(), 0)

    def test_getallwindows_exception_returns_zero(self):
        fake_gw = types.SimpleNamespace(
            getAllWindows=mock.MagicMock(side_effect=RuntimeError("win32 boom")))
        with mock.patch.object(self.mod, "_HAS_GW", True), \
             mock.patch.object(self.mod, "gw", fake_gw, create=True):
            self.assertEqual(self.mod._read_active_app_count(), 0)

    def test_per_window_exception_skipped(self):
        class _BadWin:
            visible = True

            @property
            def title(self):
                raise RuntimeError("title boom")

        good = self._win("Good")
        fake_gw = types.SimpleNamespace(getAllWindows=lambda: [_BadWin(), good])
        with mock.patch.object(self.mod, "_HAS_GW", True), \
             mock.patch.object(self.mod, "gw", fake_gw, create=True):
            self.assertEqual(self.mod._read_active_app_count(), 1)


# ─────────────────────────────────────────────────────────────────────────
# _read_bambu_status — remaining branches (FINISH/idle, locked-state error).
# ─────────────────────────────────────────────────────────────────────────
class PulseBambuStatusEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    def _fake_bambu(self, **state):
        m = types.ModuleType("skill_bambu_monitor")
        m._state_lock = threading.Lock()
        base = {"last_update": time.time()}
        base.update(state)
        m._state = base
        return m

    def test_finish_state_has_no_hours(self):
        fake = self._fake_bambu(gcode_state="finish", mc_percent=100,
                                mc_remaining=0)
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._read_bambu_status()
        self.assertEqual(out["gcode_state"], "FINISH")
        self.assertNotIn("hours_into", out)

    def test_running_zero_percent_no_hours(self):
        # pct == 0 → the hours_into estimate is skipped.
        fake = self._fake_bambu(gcode_state="RUNNING", mc_percent=0,
                                mc_remaining=120)
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._read_bambu_status()
        self.assertEqual(out["gcode_state"], "RUNNING")
        self.assertNotIn("hours_into", out)

    def test_state_lock_access_raises_returns_empty(self):
        m = types.ModuleType("skill_bambu_monitor")
        # _state_lock missing → getattr raises inside the try → {}.
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": m}):
            self.assertEqual(self.mod._read_bambu_status(), {})

    def test_non_int_percent_swallowed(self):
        fake = self._fake_bambu(gcode_state="PAUSE", mc_percent="abc",
                                mc_remaining="xyz")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._read_bambu_status()
        # gcode captured; percent/minutes_remaining skipped on ValueError.
        self.assertEqual(out["gcode_state"], "PAUSE")
        self.assertNotIn("percent", out)

    def test_running_non_int_hours_estimate_swallowed(self):
        # RUNNING but mc_percent unparseable in the hours-estimate block →
        # the inner int() raises and the except just passes (no hours_into).
        # mc_percent is a valid int for the top-level read but the *second*
        # int() in the RUNNING block gets a bad mc_remaining.
        fake = self._fake_bambu(gcode_state="RUNNING", mc_percent=50,
                                mc_remaining=object())
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            out = self.mod._read_bambu_status()
        self.assertEqual(out["gcode_state"], "RUNNING")
        self.assertNotIn("hours_into", out)


# ─────────────────────────────────────────────────────────────────────────
# _read_credit_balance — remaining branches (bad JSON, no balance key).
# ─────────────────────────────────────────────────────────────────────────
class PulseCreditEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    def _write(self, text):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        return p

    def test_corrupt_json_returns_none(self):
        p = self._write("{not valid json")
        with mock.patch.object(self.mod, "_CREDITS_STATE", p):
            self.assertIsNone(self.mod._read_credit_balance())

    def test_no_balance_key_returns_none(self):
        p = self._write(json.dumps({"checked_at": time.time()}))
        with mock.patch.object(self.mod, "_CREDITS_STATE", p):
            self.assertIsNone(self.mod._read_credit_balance())

    def test_non_numeric_balance_returns_none(self):
        p = self._write(json.dumps({"balance": "lots", "checked_at": time.time()}))
        with mock.patch.object(self.mod, "_CREDITS_STATE", p):
            self.assertIsNone(self.mod._read_credit_balance())


# ─────────────────────────────────────────────────────────────────────────
# _abnormal_reasons — the network + ram branches not yet hit.
# ─────────────────────────────────────────────────────────────────────────
class PulseAbnormalExtraTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    def test_high_ram_flagged(self):
        reasons = dict(self.mod._abnormal_reasons({"ram_pct": 95.0}))
        self.assertIn("ram", reasons)

    def test_hot_network_flagged(self):
        reasons = dict(self.mod._abnormal_reasons(
            {"net_down_kbps": 60_000.0, "net_up_kbps": 0.0}))
        self.assertIn("network", reasons)
        self.assertIn("megabytes", reasons["network"])


# ─────────────────────────────────────────────────────────────────────────
# _format_report / _format_hud_strip — remaining conditional branches.
# ─────────────────────────────────────────────────────────────────────────
class PulseFormatExtraTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    def test_fmt_bambu_running_hours_only(self):
        out = self.mod._fmt_bambu({"gcode_state": "RUNNING", "hours_into": 2.0})
        self.assertIn("2 hours into its print", out)
        self.assertNotIn("percent", out)

    def test_fmt_bambu_running_bare(self):
        # RUNNING with neither hours nor percent → generic "is running".
        self.assertEqual(self.mod._fmt_bambu({"gcode_state": "RUNNING"}),
                         "Bambu printer is running")

    def test_fmt_bambu_finish(self):
        self.assertIn("finished",
                      self.mod._fmt_bambu({"gcode_state": "FINISH"}))

    def test_format_report_low_disk_mentioned(self):
        # disk < DISK_FREE_ABNORMAL_GB*2 (40) → C drive line appears.
        pulse = {"cpu_pct": 10.0, "ram_pct": 20.0, "disk_free_gb": 30.0}
        self.assertIn("30 gigs free", self.mod._format_report(pulse))

    def test_format_report_ample_disk_omitted(self):
        pulse = {"cpu_pct": 10.0, "ram_pct": 20.0, "disk_free_gb": 500.0}
        self.assertNotIn("gigs free", self.mod._format_report(pulse))

    def test_hud_strip_uptime_days_and_net_kb(self):
        pulse = {"uptime_seconds": 2 * 86400 + 5 * 3600,
                 "net_down_kbps": 100.0, "net_up_kbps": 60.0}
        out = self.mod._format_hud_strip(pulse)
        self.assertIn("UP 2d05h", out)
        self.assertIn("kB/s", out)

    def test_hud_strip_battery_unplugged_no_suffix(self):
        out = self.mod._format_hud_strip({"battery_pct": 44.0,
                                          "battery_plugged": False})
        self.assertIn("BAT 44%", out)
        self.assertNotIn("44%+", out)

    def test_hud_strip_empty_when_nothing(self):
        self.assertEqual(self.mod._format_hud_strip({}), "")


# ─────────────────────────────────────────────────────────────────────────
# _gather_pulse — the aggregator wiring (every collector stubbed).
# ─────────────────────────────────────────────────────────────────────────
class PulseGatherTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    def test_gather_assembles_all_fields_with_battery(self):
        patches = {
            "_read_cpu_ram": (15.0, 40.0, 8.0),
            "_read_disk_free_gb": 300.0,
            "_read_gpu_temp_c": 55.0,
            "_read_uptime_seconds": 7200.0,
            "_read_active_app_count": 6,
            "_read_bambu_status": {"gcode_state": "RUNNING"},
            "_read_credit_balance": 25.0,
            "_read_network_rates": (12.0, 3.0),
            "_read_battery": (88.0, True),
        }
        cms = [mock.patch.object(self.mod, name,
                                 return_value=val) for name, val in patches.items()]
        for c in cms:
            c.start()
        try:
            pulse = self.mod._gather_pulse()
        finally:
            for c in cms:
                c.stop()
        self.assertEqual(pulse["cpu_pct"], 15.0)
        self.assertEqual(pulse["ram_used_gb"], 8.0)
        self.assertEqual(pulse["net_down_kbps"], 12.0)
        self.assertEqual(pulse["battery_pct"], 88.0)
        self.assertTrue(pulse["battery_plugged"])

    def test_gather_omits_battery_on_desktop(self):
        cms = [
            mock.patch.object(self.mod, "_read_cpu_ram", return_value=(1.0, 2.0, 3.0)),
            mock.patch.object(self.mod, "_read_disk_free_gb", return_value=0.0),
            mock.patch.object(self.mod, "_read_gpu_temp_c", return_value=None),
            mock.patch.object(self.mod, "_read_uptime_seconds", return_value=0.0),
            mock.patch.object(self.mod, "_read_active_app_count", return_value=0),
            mock.patch.object(self.mod, "_read_bambu_status", return_value={}),
            mock.patch.object(self.mod, "_read_credit_balance", return_value=None),
            mock.patch.object(self.mod, "_read_network_rates", return_value=(0.0, 0.0)),
            mock.patch.object(self.mod, "_read_battery", return_value=None),
        ]
        for c in cms:
            c.start()
        try:
            pulse = self.mod._gather_pulse()
        finally:
            for c in cms:
                c.stop()
        self.assertNotIn("battery_pct", pulse)


# ─────────────────────────────────────────────────────────────────────────
# speech queue + HUD strip publishing.
# ─────────────────────────────────────────────────────────────────────────
class PulseSpeechQueueTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")

    def test_enqueue_routes_through_proactive_announce(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.mod._enqueue_speech("hello")
        bc.proactive_announce.assert_called_once_with("hello", source="pulse")

    def test_enqueue_falls_back_to_atomic_write(self):
        # No proactive_announce → direct atomic write to the queue file.
        bc = types.ModuleType("bobert_companion")  # no proactive_announce attr
        fd, qpath = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(qpath)  # start absent so the "create new" path runs
        self.addCleanup(lambda: os.path.exists(qpath) and os.remove(qpath))
        writes = {}

        def _fake_write(path, data, **k):
            writes["path"] = path
            writes["data"] = data

        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", qpath), \
             mock.patch.object(self.mod, "_atomic_write_json", _fake_write):
            self.mod._enqueue_speech("queued message")
        self.assertEqual(writes["path"], qpath)
        self.assertEqual(writes["data"][-1]["message"], "queued message")

    def test_enqueue_appends_to_existing_queue(self):
        bc = types.ModuleType("bobert_companion")
        fd, qpath = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump([{"ts": 1.0, "message": "old"}], f)
        self.addCleanup(lambda: os.path.exists(qpath) and os.remove(qpath))
        captured = {}
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", qpath), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               lambda p, d, **k: captured.update(data=d)):
            self.mod._enqueue_speech("new")
        msgs = [e["message"] for e in captured["data"]]
        self.assertEqual(msgs, ["old", "new"])

    def test_enqueue_corrupt_queue_file_reset(self):
        bc = types.ModuleType("bobert_companion")
        fd, qpath = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("{{corrupt")
        self.addCleanup(lambda: os.path.exists(qpath) and os.remove(qpath))
        captured = {}
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", qpath), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               lambda p, d, **k: captured.update(data=d)):
            self.mod._enqueue_speech("after corruption")
        # Corrupt file discarded → only the new message remains.
        self.assertEqual([e["message"] for e in captured["data"]],
                         ["after corruption"])

    def test_enqueue_write_failure_swallowed(self):
        bc = types.ModuleType("bobert_companion")
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE",
                               os.path.join(tempfile.gettempdir(), "nope_pulse.json")), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            # Must not raise.
            self.mod._enqueue_speech("doomed")

    def test_enqueue_announcer_raises_falls_back(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(side_effect=RuntimeError("boom"))
        captured = {}
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE",
                               os.path.join(tempfile.gettempdir(), "fb_pulse.json")), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               lambda p, d, **k: captured.update(data=d)):
            self.mod._enqueue_speech("fallback msg")
        self.assertEqual(captured["data"][-1]["message"], "fallback msg")


class PulseHudPublishTests(unittest.TestCase):
    def test_publish_uses_writer_when_present(self):
        writer = mock.MagicMock()
        utils = {"write_hud_state": writer}
        mod, _ = load_skill_isolated("system_pulse", utils=utils)
        mod._publish_hud_strip("GPU 60C")
        writer.assert_called_once()
        self.assertEqual(writer.call_args.kwargs["pulse_strip"], "GPU 60C")

    def test_publish_no_writer_is_noop(self):
        utils = {"write_hud_state": None}
        mod, _ = load_skill_isolated("system_pulse", utils=utils)
        # Should simply return without error.
        mod._publish_hud_strip("anything")

    def test_publish_writer_exception_swallowed(self):
        writer = mock.MagicMock(side_effect=RuntimeError("hud locked"))
        utils = {"write_hud_state": writer}
        mod, _ = load_skill_isolated("system_pulse", utils=utils)
        mod._publish_hud_strip("strip")  # must not raise


# ─────────────────────────────────────────────────────────────────────────
# background loops — one iteration via a sleep that breaks the while-loop.
# ─────────────────────────────────────────────────────────────────────────
class _StopLoop(Exception):
    pass


class PulseLoopTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("system_pulse")
        with self.mod._pulse_cache_lock:
            self.mod._pulse_cache = None
        self.mod._last_abnormal_alert.clear()

    def test_hud_loop_one_iteration_caches_and_publishes(self):
        pulse = {"gpu_temp_c": 60.0}
        published = {}
        with mock.patch.object(self.mod, "_gather_pulse", return_value=pulse), \
             mock.patch.object(self.mod, "_format_hud_strip", return_value="GPU 60C"), \
             mock.patch.object(self.mod, "_publish_hud_strip",
                               side_effect=lambda s: published.update(strip=s)), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_StopLoop):
            with self.assertRaises(_StopLoop):
                self.mod._hud_publish_loop()
        self.assertEqual(published["strip"], "GPU 60C")
        with self.mod._pulse_cache_lock:
            self.assertEqual(self.mod._pulse_cache["pulse"], pulse)

    def test_hud_loop_empty_strip_not_published(self):
        with mock.patch.object(self.mod, "_gather_pulse", return_value={}), \
             mock.patch.object(self.mod, "_format_hud_strip", return_value=""), \
             mock.patch.object(self.mod, "_publish_hud_strip") as pub, \
             mock.patch.object(self.mod.time, "sleep", side_effect=_StopLoop):
            with self.assertRaises(_StopLoop):
                self.mod._hud_publish_loop()
        pub.assert_not_called()

    def test_hud_loop_gather_exception_is_caught_then_sleeps(self):
        # Exception inside the loop body is caught; the loop then hits sleep,
        # which we use to break out.
        with mock.patch.object(self.mod, "_gather_pulse",
                               side_effect=RuntimeError("collect boom")), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_StopLoop):
            with self.assertRaises(_StopLoop):
                self.mod._hud_publish_loop()

    def test_proactive_loop_fires_on_abnormal(self):
        pulse = {"cpu_pct": 99.0}
        # First sleep = the INITIAL delay (no-op), second sleep breaks the loop.
        sleeps = {"n": 0}

        def _sleep(_):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:
                raise _StopLoop

        enq = {}
        with mock.patch.object(self.mod, "_gather_pulse", return_value=pulse), \
             mock.patch.object(self.mod, "_enqueue_speech",
                               side_effect=lambda m: enq.update(msg=m)), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod.time, "time", return_value=10_000.0):
            with self.assertRaises(_StopLoop):
                self.mod._proactive_loop()
        self.assertIn("CPU at 99 percent", enq["msg"])
        # Cooldown recorded for the cpu reason.
        self.assertIn("cpu", self.mod._last_abnormal_alert)

    def test_proactive_loop_cooldown_suppresses_repeat(self):
        pulse = {"cpu_pct": 99.0}
        sleeps = {"n": 0}

        def _sleep(_):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:
                raise _StopLoop

        # Pre-seed the cooldown so the fresh-reason filter drops it.
        self.mod._last_abnormal_alert["cpu"] = 10_000.0
        with mock.patch.object(self.mod, "_gather_pulse", return_value=pulse), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod.time, "time", return_value=10_001.0):
            with self.assertRaises(_StopLoop):
                self.mod._proactive_loop()
        enq.assert_not_called()

    def test_proactive_loop_no_abnormal_no_announce(self):
        pulse = {"cpu_pct": 5.0, "ram_pct": 10.0}
        sleeps = {"n": 0}

        def _sleep(_):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:
                raise _StopLoop

        with mock.patch.object(self.mod, "_gather_pulse", return_value=pulse), \
             mock.patch.object(self.mod, "_enqueue_speech") as enq, \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep), \
             mock.patch.object(self.mod.time, "time", return_value=10_000.0):
            with self.assertRaises(_StopLoop):
                self.mod._proactive_loop()
        enq.assert_not_called()

    def test_proactive_loop_body_exception_logged_then_sleeps(self):
        # _gather_pulse raising is caught by the loop's except; the subsequent
        # sleep breaks us out. INITIAL delay sleep is first.
        sleeps = {"n": 0}

        def _sleep(_):
            sleeps["n"] += 1
            if sleeps["n"] >= 2:
                raise _StopLoop

        with mock.patch.object(self.mod, "_gather_pulse",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_sleep):
            with self.assertRaises(_StopLoop):
                self.mod._proactive_loop()


# ─────────────────────────────────────────────────────────────────────────
# register — thread spawn gated on psutil.
# ─────────────────────────────────────────────────────────────────────────
class PulseRegisterTests(unittest.TestCase):
    def test_register_without_psutil_skips_threads(self):
        # _HAS_PSUTIL False at register time → actions present, no threads.
        mod, _ = load_skill_isolated("system_pulse")
        actions = {}
        with mock.patch.object(mod, "_HAS_PSUTIL", False), \
             mock.patch("threading.Thread.start") as start:
            mod.register(actions)
        self.assertIn("system_pulse", actions)
        start.assert_not_called()

    def test_register_with_psutil_starts_two_threads(self):
        mod, _ = load_skill_isolated("system_pulse")
        actions = {}
        with mock.patch.object(mod, "_HAS_PSUTIL", True), \
             mock.patch("threading.Thread.start") as start:
            mod.register(actions)
        self.assertIn("system_pulse", actions)
        self.assertEqual(start.call_count, 2)


if __name__ == "__main__":
    unittest.main()
