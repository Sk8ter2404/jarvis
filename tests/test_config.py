"""Light sanity tests for core.config.

config.py is almost entirely dumb constants, so this suite stays minimal: it
pins the env-driven knobs (USER_NAME / BAMBU_*) actually read os.getenv with a
blank default, the safety CONFIRM_KEYWORDS list is non-empty, and a couple of
structural invariants other modules rely on (CONSOLE_MONITOR ∈ MONITORS, the
RAG paths expand ~). The env tests reload the module under a patched
environment so they assert the read semantics, not a committed personal value.

stdlib unittest + importlib only.
"""
from __future__ import annotations

import importlib
import os
import unittest
from unittest import mock

from core import config


class EnvDrivenTests(unittest.TestCase):
    """USER_NAME and the BAMBU_* secrets must come from the environment with a
    blank default — no personal value is ever committed to the repo."""

    def _reload_with_env(self, **env):
        with mock.patch.dict(os.environ, env, clear=False):
            return importlib.reload(config)

    def tearDown(self):
        # Restore the module to the ambient (unpatched) environment so a
        # patched value can't leak into later tests in the process.
        importlib.reload(config)

    def test_user_name_reads_env(self):
        mod = self._reload_with_env(JARVIS_USER_NAME="Tony")
        self.assertEqual(mod.USER_NAME, "Tony")

    def test_user_name_blank_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JARVIS_USER_NAME", None)
            mod = importlib.reload(config)
        self.assertEqual(mod.USER_NAME, "")

    def test_bambu_creds_read_env(self):
        mod = self._reload_with_env(
            BAMBU_PRINTER_IP="192.168.1.50",
            BAMBU_ACCESS_CODE="12345678",
            BAMBU_SERIAL="SN-XYZ",
        )
        self.assertEqual(mod.BAMBU_PRINTER_IP, "192.168.1.50")
        self.assertEqual(mod.BAMBU_ACCESS_CODE, "12345678")
        self.assertEqual(mod.BAMBU_SERIAL, "SN-XYZ")

    def test_bambu_creds_blank_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("BAMBU_PRINTER_IP", "BAMBU_ACCESS_CODE", "BAMBU_SERIAL"):
                os.environ.pop(k, None)
            mod = importlib.reload(config)
        self.assertEqual(mod.BAMBU_PRINTER_IP, "")
        self.assertEqual(mod.BAMBU_ACCESS_CODE, "")
        self.assertEqual(mod.BAMBU_SERIAL, "")


class SafetyConstantTests(unittest.TestCase):
    def test_confirm_keywords_non_empty_list(self):
        self.assertIsInstance(config.CONFIRM_KEYWORDS, list)
        self.assertTrue(config.CONFIRM_KEYWORDS)
        # The destructive verbs the safety layer keys on must be present.
        for kw in ("delete", "format", "transfer"):
            self.assertIn(kw, config.CONFIRM_KEYWORDS)

    def test_confirm_keywords_all_strings(self):
        self.assertTrue(all(isinstance(k, str) for k in config.CONFIRM_KEYWORDS))


class StructuralInvariantTests(unittest.TestCase):
    def test_console_monitor_is_a_known_monitor(self):
        # CONSOLE_MONITOR must name a key in MONITORS (or be blank).
        if config.CONSOLE_MONITOR:
            self.assertIn(config.CONSOLE_MONITOR, config.MONITORS)

    def test_hud_monitor_is_a_known_monitor(self):
        if config.HUD_MONITOR:
            self.assertIn(config.HUD_MONITOR, config.MONITORS)

    def test_monitor_tuples_are_four_ints(self):
        for name, geom in config.MONITORS.items():
            self.assertEqual(len(geom), 4, name)
            self.assertTrue(all(isinstance(v, int) for v in geom), name)

    def test_rag_paths_are_absolute_and_expanded(self):
        # RAG_INDEX_PATHS is the one import-time computed value (expanduser).
        self.assertTrue(config.RAG_INDEX_PATHS)
        for p in config.RAG_INDEX_PATHS:
            self.assertNotIn("~", p)
            self.assertTrue(os.path.isabs(p), p)

    def test_cameras_have_one_primary(self):
        primaries = [c for c in config.CAMERAS if c.get("primary")]
        self.assertEqual(len(primaries), 1)


if __name__ == "__main__":
    unittest.main()
