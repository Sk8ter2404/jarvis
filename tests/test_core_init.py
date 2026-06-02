"""Tests for the core package __init__ — the import-light blue/green role flag.

`BLUE_GREEN_ROLE` ("prod"/"staging") is derived ONCE at import from the
JARVIS_STAGING env var or a --staging argv flag, and `is_staging()` reports it.
Skills gate staging-only behaviour on this without importing the monolith. The
derivation is re-exercised here by reloading the module under a patched
environment so the prod and staging branches are both pinned; the live module
state is restored afterwards.

stdlib unittest only.
"""
from __future__ import annotations

import importlib
import os
import sys
import unittest
from unittest import mock

import core


class RoleFlagTests(unittest.TestCase):
    def test_is_staging_matches_role_constant(self):
        # The accessor is a pure read of the module-level constant.
        self.assertEqual(core.is_staging(), core.BLUE_GREEN_ROLE == "staging")

    def test_role_is_one_of_two_values(self):
        self.assertIn(core.BLUE_GREEN_ROLE, ("prod", "staging"))


class RoleDerivationTests(unittest.TestCase):
    """Reload the package under controlled env/argv to pin both branches, then
    restore the real module so other tests see the original role."""

    def _reload_with(self, env_staging, argv):
        environ = dict(os.environ)
        environ.pop("JARVIS_STAGING", None)
        if env_staging is not None:
            environ["JARVIS_STAGING"] = env_staging
        with mock.patch.dict(os.environ, environ, clear=True), \
             mock.patch.object(sys, "argv", argv):
            return importlib.reload(core)

    def setUp(self):
        self.addCleanup(importlib.reload, core)   # restore real role after

    def test_env_flag_selects_staging(self):
        mod = self._reload_with("1", ["prog"])
        self.assertEqual(mod.BLUE_GREEN_ROLE, "staging")
        self.assertTrue(mod.is_staging())

    def test_argv_flag_selects_staging(self):
        mod = self._reload_with(None, ["prog", "--staging"])
        self.assertEqual(mod.BLUE_GREEN_ROLE, "staging")

    def test_default_is_prod(self):
        mod = self._reload_with(None, ["prog"])
        self.assertEqual(mod.BLUE_GREEN_ROLE, "prod")
        self.assertFalse(mod.is_staging())

    def test_blank_env_is_prod(self):
        # An empty / whitespace JARVIS_STAGING is treated as unset (prod).
        mod = self._reload_with("  ", ["prog"])
        self.assertEqual(mod.BLUE_GREEN_ROLE, "prod")


if __name__ == "__main__":
    unittest.main()
