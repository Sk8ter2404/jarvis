"""Regression guard: JARVIS_STAGING must isolate ALL runtime state, not just
the settings file.

THE INCIDENT (2026-07-21, observed live)
========================================
A full `tools/action_smoke.py` sweep runs with JARVIS_STAGING=1, a redirected
JARVIS_SETTINGS_PATH, and an md5 tripwire on data/user_settings.json. The
tripwire stayed GREEN — and the sweep still rewrote the LIVE
`data/smart_home_devices.json`, replacing the catalog with `device_count: 0`
after its discovery actions ran and `[sh-discover] get_devices failed`.

Root cause: `JARVIS_STAGING` was honoured by the monolith's own paths and (after
an earlier incident, see tools/settings_window.settings_path) by the settings
file — but ~20 other modules resolved their own runtime-state directory with a
private `_DATA_DIR = os.path.join(_PROJECT_DIR, "data")` bound at import. The
smart-home catalog, every per-brand smart-home credential/state file (ecobee,
hue, govee, nest, ring, tuya, kasa), the scheduler, the diagnostic daemons,
ambient capture, pattern learning, the browser agent, network_deco and RAG all
wrote straight to the live box. Same rule fixed in one copy while the rest
rotted — this codebase's signature bug class.

These tests fail if a module starts resolving its own `data/` again.
"""
from __future__ import annotations

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import paths  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataDirResolutionTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k)
                       for k in ("JARVIS_STAGING", paths.DATA_DIR_ENV)}
        self._argv = list(sys.argv)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.argv[:] = self._argv

    def test_live_process_uses_data(self):
        os.environ.pop("JARVIS_STAGING", None)
        os.environ.pop(paths.DATA_DIR_ENV, None)
        sys.argv[:] = ["prog"]
        self.assertEqual(os.path.basename(paths.data_dir(create=False)), "data")

    def test_staging_process_uses_data_staging(self):
        os.environ["JARVIS_STAGING"] = "1"
        os.environ.pop(paths.DATA_DIR_ENV, None)
        self.assertEqual(os.path.basename(paths.data_dir(create=False)),
                         "data_staging")

    def test_staging_argv_flag_also_counts(self):
        os.environ.pop("JARVIS_STAGING", None)
        os.environ.pop(paths.DATA_DIR_ENV, None)
        sys.argv[:] = ["prog", "--staging"]
        self.assertEqual(os.path.basename(paths.data_dir(create=False)),
                         "data_staging")

    def test_explicit_env_override_wins(self):
        os.environ["JARVIS_STAGING"] = "1"
        os.environ[paths.DATA_DIR_ENV] = os.path.join(_PROJECT_ROOT, "zzz_tmp")
        self.assertEqual(os.path.basename(paths.data_dir(create=False)),
                         "zzz_tmp")

    def test_data_file_joins_under_data_dir(self):
        os.environ["JARVIS_STAGING"] = "1"
        os.environ.pop(paths.DATA_DIR_ENV, None)
        p = paths.data_file("smart_home_devices.json", create_dir=False)
        self.assertTrue(p.endswith(os.path.join("data_staging",
                                                "smart_home_devices.json")), p)

    def test_agrees_with_settings_path_staging_signal(self):
        """core.paths and settings_window must never disagree about whether
        this process owns the live box's state."""
        try:
            from tools import settings_window
        except Exception:
            self.skipTest("settings_window not importable in this tier")
        os.environ["JARVIS_STAGING"] = "1"
        os.environ.pop("JARVIS_SETTINGS_PATH", None)
        os.environ.pop(paths.DATA_DIR_ENV, None)
        self.assertTrue(paths.is_staging())
        self.assertIn("data_staging", settings_window.settings_path())


class NoPrivateDataDirTests(unittest.TestCase):
    """Tree-wide invariant — the guard that stops this coming back."""

    # Modules that legitimately name the LIVE data/ regardless of role.
    # These are NOT runtime-state writers acting on behalf of the current
    # process; they either define the live default that the staging chooser
    # picks between, or they deliberately manage BOTH roots.
    _EXEMPT = {
        # Builds both branches — it IS the chooser.
        os.path.join("core", "paths.py"),
        # Seeds and tears down data_staging/ from the LIVE side; it must be
        # able to name both roots explicitly.
        "blue_green_manager.py",
        # DATA_DIR here is the live default that settings_path() returns for a
        # NON-staging process; settings_path() adds the staging branch above it.
        os.path.join("tools", "settings_window.py"),
        # Last-resort fallback used only when no settings path was resolved at
        # all; the staging redirect has already been applied upstream.
        os.path.join("tools", "web_interface.py"),
        # Child UI processes: they are only ever spawned BY the live instance
        # (a staging instance runs headless), and they read the live HUD state
        # they were launched to display.
        "tray.py",
        os.path.join("hud", "workshop_hud.py"),
    }

    _PATTERNS = (
        re.compile(r'os\.path\.join\(\s*_?PROJECT_DIR\s*,\s*["\']data["\']\s*\)'),
        re.compile(r'os\.path\.join\(\s*_?PROJ_DIR\s*,\s*["\']data["\']\s*\)'),
        re.compile(r'os\.path\.join\(\s*_?HERE\s*,\s*["\']data["\']\s*\)'),
    )

    def _sources(self):
        for base, dirs, files in os.walk(_PROJECT_ROOT):
            dirs[:] = [d for d in dirs
                       if d not in ("tests", "__pycache__", ".git", ".claude",
                                    "backups", "_backups", "dist", "models",
                                    "logs", "logs_staging", "data", "data_staging",
                                    "node_modules", "Robot Project")]
            for fn in files:
                if fn.endswith(".py"):
                    yield os.path.join(base, fn)

    def test_no_module_resolves_its_own_data_dir(self):
        offenders = []
        for path in self._sources():
            rel = os.path.relpath(path, _PROJECT_ROOT)
            if rel in self._EXEMPT:
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
            except OSError:
                continue
            for pat in self._PATTERNS:
                for m in pat.finditer(src):
                    line_no = src[:m.start()].count("\n") + 1
                    line = src.splitlines()[line_no - 1]
                    # The documented fallback inside a `except Exception:` arm
                    # of the core.paths shim is intentional — it only fires if
                    # core.paths is unimportable.
                    prev = "\n".join(src.splitlines()[max(0, line_no - 6):line_no])
                    if "_jarvis_data_dir" in prev:
                        continue
                    offenders.append(f"{rel}:{line_no}: {line.strip()}")
        self.assertEqual(
            offenders, [],
            "Module(s) resolving a private data directory — these bypass "
            "JARVIS_STAGING and let a sweep/test write the LIVE box's runtime "
            "state. Use core.paths.data_dir(). Offenders: " + "; ".join(offenders))


if __name__ == "__main__":
    unittest.main()
