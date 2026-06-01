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
import tempfile
import time
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


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


if __name__ == "__main__":
    unittest.main()
