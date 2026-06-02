"""Logic tests for skills/status_panel.py.

Targets the "suit diagnostics" readout assembly without touching real hardware:
  • _shorten_foreground_title — pull the app name out of a noisy window title.
  • _gpu_phrase — telemetry-unavailable, hot, and holding wording.
  • _read_credit_balance — staleness + missing-file degradation (temp file).
  • _read_bambu_percent — reads sibling skill state across sys.modules.
  • _build_readout — opener (nominal vs "slight problem") + the conditional
    CPU/GPU/network/credits/Bambu/foreground/music lines, every collector mocked.
  • _build_hud_strip — the compact WIN/PING line.
  • the status_panel action serving a fresh cache vs falling back to a live
    build, with the card pop stubbed.

All collectors are patched, so the readout is deterministic and no nvidia-smi /
ping / COM call happens.
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


class _StopLoop(Exception):
    """Sentinel raised from a patched time.sleep to break a while-True loop
    after exactly one iteration."""


class StatusTitleTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def test_shorten_known_suffix(self):
        self.assertEqual(
            self.mod._shorten_foreground_title(
                "bobert_companion.py - jarvis - Visual Studio Code"),
            "Visual Studio Code")

    def test_shorten_chrome(self):
        self.assertEqual(
            self.mod._shorten_foreground_title("Gmail - Google Chrome"),
            "Google Chrome")

    def test_shorten_unknown_takes_tail(self):
        out = self.mod._shorten_foreground_title("Folder - SomeRandomApp")
        self.assertEqual(out, "SomeRandomApp")

    def test_shorten_empty(self):
        self.assertEqual(self.mod._shorten_foreground_title(""), "")


class StatusGpuPhraseTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def test_gpu_none_unavailable(self):
        self.assertIn("unavailable", self.mod._gpu_phrase(None).lower())

    def test_gpu_hot_wording(self):
        # >= GPU_TEMP_HOT_C must say "hot" in every variant.
        outs = [self.mod._gpu_phrase(85.0) for _ in range(30)]
        self.assertTrue(all("hot" in o for o in outs))
        self.assertTrue(all("85" in o for o in outs))

    def test_gpu_holding_wording(self):
        outs = [self.mod._gpu_phrase(50.0) for _ in range(30)]
        self.assertTrue(all("holding at" in o for o in outs))
        self.assertFalse(any("hot" in o for o in outs))


class StatusCollectorTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def test_credit_balance_fresh(self):
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"balance": 12.34, "checked_at": time.time()}, f)
            with mock.patch.object(self.mod, "_CREDITS_STATE", p):
                self.assertAlmostEqual(self.mod._read_credit_balance(), 12.34,
                                       places=2)
        finally:
            os.unlink(p)

    def test_credit_balance_stale_skipped(self):
        fd, p = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            old = time.time() - (self.mod.CREDITS_STATE_MAX_AGE_SECONDS + 100)
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"balance": 12.34, "checked_at": old}, f)
            with mock.patch.object(self.mod, "_CREDITS_STATE", p):
                self.assertIsNone(self.mod._read_credit_balance())
        finally:
            os.unlink(p)

    def test_credit_balance_missing_file(self):
        with mock.patch.object(self.mod, "_CREDITS_STATE",
                               os.path.join(tempfile.gettempdir(), "no_such_credits.json")):
            self.assertIsNone(self.mod._read_credit_balance())

    def _write_credits(self, text):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        self.addCleanup(lambda: os.path.exists(p) and os.remove(p))
        return p

    def test_credit_balance_corrupt_json(self):
        p = self._write_credits("{broken json")
        with mock.patch.object(self.mod, "_CREDITS_STATE", p):
            self.assertIsNone(self.mod._read_credit_balance())

    def test_credit_balance_no_balance_key(self):
        p = self._write_credits(json.dumps({"checked_at": time.time()}))
        with mock.patch.object(self.mod, "_CREDITS_STATE", p):
            self.assertIsNone(self.mod._read_credit_balance())

    def test_credit_balance_non_numeric(self):
        p = self._write_credits(json.dumps({"balance": "plenty",
                                            "checked_at": time.time()}))
        with mock.patch.object(self.mod, "_CREDITS_STATE", p):
            self.assertIsNone(self.mod._read_credit_balance())

    def test_read_bambu_percent_from_sibling(self):
        import sys
        fake = mock.MagicMock()

        class _NullLock:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        fake._state_lock = _NullLock()
        fake._state = {"last_update": time.time(), "mc_percent": 42,
                       "gcode_state": "RUNNING"}
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            result = self.mod._read_bambu_percent()
        self.assertEqual(result, (42, "RUNNING"))

    def test_read_bambu_percent_absent(self):
        import sys
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": None}):
            self.assertIsNone(self.mod._read_bambu_percent())


class StatusReadoutTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def _patch_all(self, *, cpu=20.0, ram=30.0, gpu=55.0, ping=12.0,
                   credits=None, bambu=None, foreground=None, music=None):
        return [
            mock.patch.object(self.mod, "_read_cpu_ram", return_value=(cpu, ram)),
            mock.patch.object(self.mod, "_read_gpu_temp_c", return_value=gpu),
            mock.patch.object(self.mod, "_read_ping_ms", return_value=ping),
            mock.patch.object(self.mod, "_read_credit_balance", return_value=credits),
            mock.patch.object(self.mod, "_read_bambu_percent", return_value=bambu),
            mock.patch.object(self.mod, "_read_foreground_app", return_value=foreground),
            mock.patch.object(self.mod, "_read_apple_music_track", return_value=music),
        ]

    def test_readout_nominal_opener(self):
        patches = self._patch_all()
        with mock.patch.object(self.mod.random, "random", return_value=0.99):
            for p in patches:
                p.start()
            try:
                out = self.mod._build_readout()
            finally:
                for p in patches:
                    p.stop()
        self.assertTrue(out.startswith("All systems nominal, sir."))
        self.assertIn("CPU at 20 percent", out)
        self.assertIn("RAM at 30 percent", out)
        self.assertIn("12 milliseconds", out)
        self.assertTrue(out.endswith("Shall I continue?"))

    def test_readout_concerning_opener_on_high_cpu(self):
        patches = self._patch_all(cpu=95.0)
        with mock.patch.object(self.mod.random, "random", return_value=0.99):
            for p in patches:
                p.start()
            try:
                out = self.mod._build_readout()
            finally:
                for p in patches:
                    p.stop()
        self.assertTrue(out.startswith("Slight problem, sir."))

    def test_readout_low_credits_warns(self):
        patches = self._patch_all(credits=2.50)
        with mock.patch.object(self.mod.random, "random", return_value=0.99):
            for p in patches:
                p.start()
            try:
                out = self.mod._build_readout()
            finally:
                for p in patches:
                    p.stop()
        self.assertTrue(out.startswith("Slight problem, sir."))
        self.assertIn("$2.50", out)
        self.assertIn("top-up", out.lower())

    def test_readout_bambu_running_line(self):
        patches = self._patch_all(bambu=(63, "RUNNING"))
        with mock.patch.object(self.mod.random, "random", return_value=0.99):
            for p in patches:
                p.start()
            try:
                out = self.mod._build_readout()
            finally:
                for p in patches:
                    p.stop()
        self.assertIn("Bambu printer at 63 percent", out)

    def test_readout_network_unavailable(self):
        patches = self._patch_all(ping=None)
        with mock.patch.object(self.mod.random, "random", return_value=0.99):
            for p in patches:
                p.start()
            try:
                out = self.mod._build_readout()
            finally:
                for p in patches:
                    p.stop()
        self.assertIn("Network response time unavailable", out)

    def test_readout_includes_foreground_and_music(self):
        patches = self._patch_all(foreground="Blender",
                                  music="'Earth Song' by Michael Jackson")
        with mock.patch.object(self.mod.random, "random", return_value=0.99):
            for p in patches:
                p.start()
            try:
                out = self.mod._build_readout()
            finally:
                for p in patches:
                    p.stop()
        self.assertIn("Foreground is Blender", out)
        self.assertIn("Apple Music playing 'Earth Song' by Michael Jackson", out)


class StatusHudStripAndActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def test_hud_strip_combines_win_and_ping(self):
        with mock.patch.object(self.mod, "_read_foreground_app", return_value="VS Code"), \
             mock.patch.object(self.mod, "_read_ping_ms", return_value=18.0):
            strip = self.mod._build_hud_strip()
        self.assertIn("WIN VS Code", strip)
        self.assertIn("PING 18ms", strip)

    def test_hud_strip_empty_when_nothing_available(self):
        with mock.patch.object(self.mod, "_read_foreground_app", return_value=None), \
             mock.patch.object(self.mod, "_read_ping_ms", return_value=None):
            self.assertEqual(self.mod._build_hud_strip(), "")

    def test_action_serves_fresh_cache(self):
        # Seed the cache with a recent readout; the action must return it
        # verbatim without re-building (no collectors run).
        with self.mod._readout_cache_lock:
            self.mod._readout_cache = {"text": "CACHED READOUT, sir.",
                                       "ts": time.time()}
        with mock.patch.object(self.mod, "_show_card_safe"), \
             mock.patch.object(self.mod, "_build_readout",
                               side_effect=AssertionError("should not rebuild")):
            out = self.actions["status_panel"]("")
        self.assertEqual(out, "CACHED READOUT, sir.")

    def test_action_falls_back_to_live_build_when_cache_stale(self):
        with self.mod._readout_cache_lock:
            self.mod._readout_cache = {"text": "OLD",
                                       "ts": time.time() - 1000}   # stale
        with mock.patch.object(self.mod, "_show_card_safe"), \
             mock.patch.object(self.mod, "_build_readout",
                               return_value="LIVE READOUT, sir."):
            out = self.actions["status_panel"]("")
        self.assertEqual(out, "LIVE READOUT, sir.")

    def test_action_aliases_share_handler(self):
        # system_status and suit_diagnostics map to the same readout action.
        with self.mod._readout_cache_lock:
            self.mod._readout_cache = {"text": "X, sir.", "ts": time.time()}
        with mock.patch.object(self.mod, "_show_card_safe"):
            self.assertEqual(self.actions["system_status"](""), "X, sir.")
            self.assertEqual(self.actions["suit_diagnostics"](""), "X, sir.")


# ─── psutil stand-ins ────────────────────────────────────────────────────
class _VM:
    def __init__(self, percent):
        self.percent = percent


# ─────────────────────────────────────────────────────────────────────────
# _read_cpu_ram — psutil mocked.
# ─────────────────────────────────────────────────────────────────────────
class StatusCpuRamTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def test_with_psutil(self):
        fake = mock.MagicMock()
        fake.cpu_percent.return_value = 22.0
        fake.virtual_memory.return_value = _VM(48.0)
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertEqual(self.mod._read_cpu_ram(), (22.0, 48.0))

    def test_without_psutil(self):
        with mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertEqual(self.mod._read_cpu_ram(), (0.0, 0.0))

    def test_exception_returns_zeroes(self):
        fake = mock.MagicMock()
        fake.cpu_percent.side_effect = RuntimeError("boom")
        with mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertEqual(self.mod._read_cpu_ram(), (0.0, 0.0))


# ─────────────────────────────────────────────────────────────────────────
# _read_gpu_temp_c — nvidia-smi + psutil-sensors fallback.
# ─────────────────────────────────────────────────────────────────────────
class StatusGpuTempTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def test_nvidia_smi_hottest(self):
        proc = types.SimpleNamespace(stdout="55\n81\n")
        with mock.patch.object(self.mod.shutil, "which", return_value="nvidia-smi"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertEqual(self.mod._read_gpu_temp_c(), 81.0)

    def test_no_smi_psutil_sensor_fallback(self):
        entry = types.SimpleNamespace(current=66.0)
        fake = mock.MagicMock()
        fake.sensors_temperatures.return_value = {"radeon": [entry]}
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertEqual(self.mod._read_gpu_temp_c(), 66.0)

    def test_none_when_no_source(self):
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertIsNone(self.mod._read_gpu_temp_c())

    def test_smi_exception_swallowed(self):
        with mock.patch.object(self.mod.shutil, "which", return_value="nvidia-smi"), \
             mock.patch.object(self.mod.subprocess, "run",
                               side_effect=OSError("spawn")), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertIsNone(self.mod._read_gpu_temp_c())

    def test_sensors_exception_swallowed(self):
        fake = mock.MagicMock()
        fake.sensors_temperatures.side_effect = RuntimeError("nope")
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertIsNone(self.mod._read_gpu_temp_c())

    def test_smi_no_digit_lines_falls_through_to_sensors(self):
        # nvidia-smi present but emits no parseable temps → the `if temps`
        # guard is False and execution falls through to the psutil sensors
        # path (which here also finds nothing → None).
        proc = types.SimpleNamespace(stdout="N/A\nERR\n")
        with mock.patch.object(self.mod.shutil, "which", return_value="nvidia-smi"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertIsNone(self.mod._read_gpu_temp_c())

    def test_sensors_skips_non_gpu_and_empty_current(self):
        # A non-GPU label (skipped by the substring guard) followed by a GPU
        # label whose entries have no usable `current` → loop exhausts → None.
        no_cur = types.SimpleNamespace(current=None)
        fake = mock.MagicMock()
        fake.sensors_temperatures.return_value = {
            "coretemp": [types.SimpleNamespace(current=45.0)],  # non-gpu, skip
            "nvidia": [no_cur],                                  # gpu, empty
        }
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertIsNone(self.mod._read_gpu_temp_c())


# ─────────────────────────────────────────────────────────────────────────
# _read_ping_ms — subprocess ping, both platform branches.
# ─────────────────────────────────────────────────────────────────────────
class StatusPingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def test_win_ping_parses_time(self):
        proc = types.SimpleNamespace(
            stdout="Reply from 1.1.1.1: bytes=32 time=14ms TTL=57")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertEqual(self.mod._read_ping_ms(), 14.0)

    def test_posix_ping_parses_decimal_time(self):
        proc = types.SimpleNamespace(
            stdout="64 bytes from 1.1.1.1: icmp_seq=1 ttl=57 time=12.3 ms")
        with mock.patch.object(self.mod.sys, "platform", "linux"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertAlmostEqual(self.mod._read_ping_ms(), 12.3, places=1)

    def test_no_time_match_returns_none(self):
        proc = types.SimpleNamespace(stdout="Request timed out.")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run", return_value=proc):
            self.assertIsNone(self.mod._read_ping_ms())

    def test_subprocess_exception_returns_none(self):
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "run",
                               side_effect=OSError("no ping")):
            self.assertIsNone(self.mod._read_ping_ms())


# ─────────────────────────────────────────────────────────────────────────
# _read_bambu_percent — remaining branches.
# ─────────────────────────────────────────────────────────────────────────
class StatusBambuPercentEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def _fake(self, **state):
        m = types.ModuleType("skill_bambu_monitor")
        m._state_lock = threading.Lock()
        base = {"last_update": time.time()}
        base.update(state)
        m._state = base
        return m

    def test_no_fresh_state_returns_none(self):
        fake = self._fake(last_update=0.0, gcode_state="RUNNING")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertIsNone(self.mod._read_bambu_percent())

    def test_non_int_percent_defaults_zero(self):
        fake = self._fake(gcode_state="RUNNING", mc_percent="bad")
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": fake}):
            self.assertEqual(self.mod._read_bambu_percent(), (0, "RUNNING"))

    def test_state_lock_missing_returns_none(self):
        m = types.ModuleType("skill_bambu_monitor")  # no _state_lock
        with mock.patch.dict(sys.modules, {"skill_bambu_monitor": m}):
            self.assertIsNone(self.mod._read_bambu_percent())


# ─────────────────────────────────────────────────────────────────────────
# _read_foreground_app — pygetwindow gated.
# ─────────────────────────────────────────────────────────────────────────
class StatusForegroundTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def test_shortens_active_window(self):
        win = types.SimpleNamespace(title="notes.txt - Notepad")
        fake_gw = types.SimpleNamespace(getActiveWindow=lambda: win)
        with mock.patch.object(self.mod, "_HAS_GW", True), \
             mock.patch.object(self.mod, "gw", fake_gw, create=True):
            self.assertEqual(self.mod._read_foreground_app(), "Notepad")

    def test_without_gw_returns_none(self):
        with mock.patch.object(self.mod, "_HAS_GW", False):
            self.assertIsNone(self.mod._read_foreground_app())

    def test_no_active_window_returns_none(self):
        fake_gw = types.SimpleNamespace(getActiveWindow=lambda: None)
        with mock.patch.object(self.mod, "_HAS_GW", True), \
             mock.patch.object(self.mod, "gw", fake_gw, create=True):
            self.assertIsNone(self.mod._read_foreground_app())

    def test_ignored_system_title_returns_none(self):
        win = types.SimpleNamespace(title="Program Manager")
        fake_gw = types.SimpleNamespace(getActiveWindow=lambda: win)
        with mock.patch.object(self.mod, "_HAS_GW", True), \
             mock.patch.object(self.mod, "gw", fake_gw, create=True):
            self.assertIsNone(self.mod._read_foreground_app())

    def test_exception_returns_none(self):
        fake_gw = types.SimpleNamespace(
            getActiveWindow=mock.MagicMock(side_effect=RuntimeError("boom")))
        with mock.patch.object(self.mod, "_HAS_GW", True), \
             mock.patch.object(self.mod, "gw", fake_gw, create=True):
            self.assertIsNone(self.mod._read_foreground_app())


# ─────────────────────────────────────────────────────────────────────────
# _read_apple_music_track — passive iTunes COM read (never launches).
# ─────────────────────────────────────────────────────────────────────────
class StatusAppleMusicTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def _proc(self, name):
        return types.SimpleNamespace(info={"name": name})

    def test_non_windows_returns_none(self):
        with mock.patch.object(self.mod.sys, "platform", "linux"):
            self.assertIsNone(self.mod._read_apple_music_track())

    def test_without_psutil_returns_none(self):
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", False):
            self.assertIsNone(self.mod._read_apple_music_track())

    def test_itunes_not_running_returns_none(self):
        fake = mock.MagicMock()
        fake.process_iter.return_value = [self._proc("chrome.exe")]
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertIsNone(self.mod._read_apple_music_track())

    def test_process_iter_exception_returns_none(self):
        fake = mock.MagicMock()
        fake.process_iter.side_effect = RuntimeError("denied")
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", fake):
            self.assertIsNone(self.mod._read_apple_music_track())

    def _running_psutil(self):
        fake = mock.MagicMock()
        fake.process_iter.return_value = [self._proc("iTunes.exe")]
        return fake

    def _patch_win32com(self, track):
        """Inject a fake win32com.client whose GetActiveObject yields an app
        with the given CurrentTrack. Returns the cleanup-registering cm."""
        client = types.ModuleType("win32com.client")
        app = types.SimpleNamespace(CurrentTrack=track)
        client.GetActiveObject = lambda _progid: app
        win32com = types.ModuleType("win32com")
        win32com.client = client
        return win32com, client

    def test_playing_track_with_artist(self):
        track = types.SimpleNamespace(Name="Earth Song",
                                      Artist="Michael Jackson")
        win32com, client = self._patch_win32com(track)
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", self._running_psutil()), \
             mock.patch.dict(sys.modules, {"win32com": win32com,
                                           "win32com.client": client}):
            out = self.mod._read_apple_music_track()
        self.assertEqual(out, "'Earth Song' by Michael Jackson")

    def test_playing_track_no_artist(self):
        track = types.SimpleNamespace(Name="Untitled", Artist="")
        win32com, client = self._patch_win32com(track)
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", self._running_psutil()), \
             mock.patch.dict(sys.modules, {"win32com": win32com,
                                           "win32com.client": client}):
            self.assertEqual(self.mod._read_apple_music_track(), "'Untitled'")

    def test_no_current_track_returns_none(self):
        win32com, client = self._patch_win32com(None)
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", self._running_psutil()), \
             mock.patch.dict(sys.modules, {"win32com": win32com,
                                           "win32com.client": client}):
            self.assertIsNone(self.mod._read_apple_music_track())

    def test_blank_track_name_returns_none(self):
        track = types.SimpleNamespace(Name="", Artist="Someone")
        win32com, client = self._patch_win32com(track)
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", self._running_psutil()), \
             mock.patch.dict(sys.modules, {"win32com": win32com,
                                           "win32com.client": client}):
            self.assertIsNone(self.mod._read_apple_music_track())

    def test_com_exception_returns_none(self):
        # GetActiveObject raising (COM server gone) → None, no crash.
        client = types.ModuleType("win32com.client")
        client.GetActiveObject = mock.MagicMock(side_effect=RuntimeError("no COM"))
        win32com = types.ModuleType("win32com")
        win32com.client = client
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod, "_HAS_PSUTIL", True), \
             mock.patch.object(self.mod, "psutil", self._running_psutil()), \
             mock.patch.dict(sys.modules, {"win32com": win32com,
                                           "win32com.client": client}):
            self.assertIsNone(self.mod._read_apple_music_track())


# ─────────────────────────────────────────────────────────────────────────
# _shorten_foreground_title — remaining branches (em-dash separator, cap).
# ─────────────────────────────────────────────────────────────────────────
class StatusTitleExtraTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def test_emdash_hint_match(self):
        # " — Spotify" (em-dash) anywhere in the title matches the hint.
        self.assertEqual(
            self.mod._shorten_foreground_title("Some Song — Spotify"),
            "Spotify")

    def test_long_unknown_capped_to_40(self):
        long_name = "Z" * 60
        out = self.mod._shorten_foreground_title(long_name)
        self.assertEqual(len(out), 40)

    def test_hyphen_hint_in_middle_not_endswith(self):
        # Hint appears as " - Discord" mid-title (endswith is "huddle"), so the
        # hyphen-substring branch decides.
        self.assertEqual(
            self.mod._shorten_foreground_title("Reply - Discord - huddle"),
            "Discord")


# ─────────────────────────────────────────────────────────────────────────
# _build_readout — the FINISH and FAILED Bambu lines + concerning-by-bambu.
# ─────────────────────────────────────────────────────────────────────────
class StatusReadoutBambuTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def _build(self, **over):
        defaults = dict(cpu=10.0, ram=20.0, gpu=50.0, ping=10.0, credits=None,
                        bambu=None, foreground=None, music=None)
        defaults.update(over)
        patches = [
            mock.patch.object(self.mod, "_read_cpu_ram",
                              return_value=(defaults["cpu"], defaults["ram"])),
            mock.patch.object(self.mod, "_read_gpu_temp_c",
                              return_value=defaults["gpu"]),
            mock.patch.object(self.mod, "_read_ping_ms",
                              return_value=defaults["ping"]),
            mock.patch.object(self.mod, "_read_credit_balance",
                              return_value=defaults["credits"]),
            mock.patch.object(self.mod, "_read_bambu_percent",
                              return_value=defaults["bambu"]),
            mock.patch.object(self.mod, "_read_foreground_app",
                              return_value=defaults["foreground"]),
            mock.patch.object(self.mod, "_read_apple_music_track",
                              return_value=defaults["music"]),
            mock.patch.object(self.mod.random, "random", return_value=0.99),
        ]
        for p in patches:
            p.start()
        try:
            return self.mod._build_readout()
        finally:
            for p in patches:
                p.stop()

    def test_bambu_finish_line(self):
        out = self._build(bambu=(100, "FINISH"))
        self.assertIn("finished its last print", out)

    def test_bambu_failed_line_and_concerning(self):
        out = self._build(bambu=(50, "FAILED"))
        self.assertTrue(out.startswith("Slight problem, sir."))
        self.assertIn("Bambu print has failed", out)

    def test_bambu_idle_no_line(self):
        # gcode not in {RUNNING(0<pct<100), FINISH, FAILED} → no Bambu line.
        out = self._build(bambu=(0, "IDLE"))
        self.assertNotIn("Bambu", out)

    def test_concerning_on_hot_gpu(self):
        out = self._build(gpu=85.0)
        self.assertTrue(out.startswith("Slight problem, sir."))

    def test_concerning_on_high_ram(self):
        out = self._build(ram=95.0)
        self.assertTrue(out.startswith("Slight problem, sir."))

    def test_ample_credits_line(self):
        out = self._build(credits=42.0)
        self.assertIn("$42.00 in Claude credits", out)
        self.assertNotIn("top-up", out)

    def test_gpu_easter_egg_phrasing(self):
        # Force the ~20% reactor easter-egg branch by pinning random < 0.20.
        with mock.patch.object(self.mod.random, "random", return_value=0.10):
            self.assertIn("reactor", self.mod._gpu_phrase(55.0))


# ─────────────────────────────────────────────────────────────────────────
# _build_hud_strip — remaining branch (ping-only, no foreground).
# ─────────────────────────────────────────────────────────────────────────
class StatusHudStripExtraTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def test_ping_only_strip(self):
        with mock.patch.object(self.mod, "_read_foreground_app", return_value=None), \
             mock.patch.object(self.mod, "_read_ping_ms", return_value=21.0):
            self.assertEqual(self.mod._build_hud_strip(), "PING 21ms")

    def test_foreground_truncated_to_22(self):
        long_fg = "A" * 40
        with mock.patch.object(self.mod, "_read_foreground_app", return_value=long_fg), \
             mock.patch.object(self.mod, "_read_ping_ms", return_value=None):
            out = self.mod._build_hud_strip()
        self.assertEqual(out, "WIN " + "A" * 22)


# ─────────────────────────────────────────────────────────────────────────
# _show_card_safe — lazy hud_card import.
# ─────────────────────────────────────────────────────────────────────────
class StatusShowCardTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def test_show_card_invokes_hud_card(self):
        hud_card = types.ModuleType("hud_card")
        hud_card.show_card = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"hud_card": hud_card}):
            self.mod._show_card_safe()
        hud_card.show_card.assert_called_once_with("status")

    def test_show_card_import_failure_swallowed(self):
        with mock.patch.object(self.mod.importlib, "import_module",
                               side_effect=ImportError("no hud_card")):
            self.mod._show_card_safe()  # must not raise


# ─────────────────────────────────────────────────────────────────────────
# speech queue + HUD strip publishing.
# ─────────────────────────────────────────────────────────────────────────
class StatusSpeechQueueTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")

    def test_routes_through_proactive_announce(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}):
            self.mod._enqueue_speech("hi")
        bc.proactive_announce.assert_called_once_with("hi", source="status_panel")

    def test_fallback_atomic_write_new_file(self):
        bc = types.ModuleType("bobert_companion")
        captured = {}
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE",
                               os.path.join(tempfile.gettempdir(), "sp_new.json")), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               lambda p, d, **k: captured.update(data=d)):
            self.mod._enqueue_speech("msg")
        self.assertEqual(captured["data"][-1]["message"], "msg")

    def test_fallback_appends_to_existing(self):
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
        self.assertEqual([e["message"] for e in captured["data"]], ["old", "new"])

    def test_fallback_corrupt_file_reset(self):
        bc = types.ModuleType("bobert_companion")
        fd, qpath = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("not json{")
        self.addCleanup(lambda: os.path.exists(qpath) and os.remove(qpath))
        captured = {}
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE", qpath), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               lambda p, d, **k: captured.update(data=d)):
            self.mod._enqueue_speech("fresh")
        self.assertEqual([e["message"] for e in captured["data"]], ["fresh"])

    def test_fallback_write_failure_swallowed(self):
        bc = types.ModuleType("bobert_companion")
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE",
                               os.path.join(tempfile.gettempdir(), "sp_doom.json")), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               side_effect=OSError("disk full")):
            self.mod._enqueue_speech("doomed")  # must not raise

    def test_announcer_exception_falls_back(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(side_effect=RuntimeError("x"))
        captured = {}
        with mock.patch.dict(sys.modules, {"bobert_companion": bc}), \
             mock.patch.object(self.mod, "_SPEECH_QUEUE",
                               os.path.join(tempfile.gettempdir(), "sp_fb.json")), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False), \
             mock.patch.object(self.mod, "_atomic_write_json",
                               lambda p, d, **k: captured.update(data=d)):
            self.mod._enqueue_speech("fb")
        self.assertEqual(captured["data"][-1]["message"], "fb")


class StatusHudPublishTests(unittest.TestCase):
    def test_publish_uses_writer(self):
        writer = mock.MagicMock()
        mod, _ = load_skill_isolated("status_panel",
                                     utils={"write_hud_state": writer})
        mod._publish_hud_strip("WIN VS Code")
        writer.assert_called_once()
        self.assertEqual(writer.call_args.kwargs["status_panel_strip"],
                         "WIN VS Code")

    def test_publish_no_writer_noop(self):
        mod, _ = load_skill_isolated("status_panel",
                                     utils={"write_hud_state": None})
        mod._publish_hud_strip("x")

    def test_publish_writer_exception_swallowed(self):
        writer = mock.MagicMock(side_effect=RuntimeError("locked"))
        mod, _ = load_skill_isolated("status_panel",
                                     utils={"write_hud_state": writer})
        mod._publish_hud_strip("x")


# ─────────────────────────────────────────────────────────────────────────
# _hud_publish_loop + register — one iteration / thread spawn.
# ─────────────────────────────────────────────────────────────────────────
class StatusLoopAndRegisterTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("status_panel")
        with self.mod._readout_cache_lock:
            self.mod._readout_cache = None

    def test_loop_one_iteration_publishes_and_caches(self):
        published = {}
        with mock.patch.object(self.mod, "_build_hud_strip",
                               return_value="WIN X  ·  PING 9ms"), \
             mock.patch.object(self.mod, "_publish_hud_strip",
                               side_effect=lambda s: published.update(s=s)), \
             mock.patch.object(self.mod, "_build_readout",
                               return_value="FULL READOUT, sir."), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_StopLoop):
            with self.assertRaises(_StopLoop):
                self.mod._hud_publish_loop()
        self.assertEqual(published["s"], "WIN X  ·  PING 9ms")
        with self.mod._readout_cache_lock:
            self.assertEqual(self.mod._readout_cache["text"], "FULL READOUT, sir.")

    def test_loop_empty_strip_not_published(self):
        with mock.patch.object(self.mod, "_build_hud_strip", return_value=""), \
             mock.patch.object(self.mod, "_publish_hud_strip") as pub, \
             mock.patch.object(self.mod, "_build_readout", return_value="R"), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_StopLoop):
            with self.assertRaises(_StopLoop):
                self.mod._hud_publish_loop()
        pub.assert_not_called()

    def test_loop_strip_build_exception_caught(self):
        # _build_hud_strip raising is caught; the readout build still runs and
        # then sleep breaks the loop.
        with mock.patch.object(self.mod, "_build_hud_strip",
                               side_effect=RuntimeError("strip boom")), \
             mock.patch.object(self.mod, "_build_readout", return_value="R"), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_StopLoop):
            with self.assertRaises(_StopLoop):
                self.mod._hud_publish_loop()

    def test_loop_readout_build_exception_caught(self):
        with mock.patch.object(self.mod, "_build_hud_strip", return_value=""), \
             mock.patch.object(self.mod, "_build_readout",
                               side_effect=RuntimeError("readout boom")), \
             mock.patch.object(self.mod.time, "sleep", side_effect=_StopLoop):
            with self.assertRaises(_StopLoop):
                self.mod._hud_publish_loop()

    def test_action_wraps_exceptions(self):
        with self.mod._readout_cache_lock:
            self.mod._readout_cache = None
        with mock.patch.object(self.mod, "_build_readout",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(self.mod, "_show_card_safe"):
            out = self.actions["status_panel"]("")
        self.assertIn("status panel failed", out.lower())

    def test_register_starts_hud_thread(self):
        mod, _ = load_skill_isolated("status_panel")
        actions = {}
        with mock.patch("threading.Thread.start") as start:
            mod.register(actions)
        for name in ("status_panel", "system_status", "suit_diagnostics"):
            self.assertIn(name, actions)
        start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
