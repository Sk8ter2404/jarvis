"""Headless tests for the blue/green ``--state-file`` / ``--role`` handling in
the unified HUD (``hud/jarvis_unified_hud.py``).

WHY THIS EXISTS
  blue_green_manager gives the staging (green) instance its own
  ``data_staging/hud_state.json``, and ``bobert_companion._launch_hud``
  dutifully passes ``--role staging --state-file <that path>`` to the HUD it
  spawns. The unified HUD used to parse those flags into ``_unknown`` and
  drop them: its HUD_STATE_FILE / CONTROL_FILE / GEOMETRY_FILE globals were
  hard-coded to the prod paths, so a staging HUD rendered PROD state, chased
  prod's control file, restored and OVERWROTE prod's saved geometry — and
  drew no STAGING badge to warn anyone (the prose in _launch_hud even called
  the discard "harmless": the stale-duplicate of a rule that only
  hud/jarvis_hud.py — which nothing launches — implemented correctly).

  The fix declares both flags in main() and, when --state-file is supplied,
  repoints all three per-role globals into the state file's directory BEFORE
  anything reads them (and before the PyQt6 gate, so the reassignment is
  testable headless: main() returns 2 without PyQt6, after the repoint).

  Guarded two ways:
    • behavioral: main() with staging argv leaves the globals inside the
      staging dir; default argv keeps the prod paths byte-for-byte;
    • a SOURCE-SCANNING INVARIANT: every ``--flag`` literal that
      bobert_companion._launch_hud appends to the HUD argv must be declared
      via add_argument in the unified HUD's main() — so a future launcher
      flag silently swallowed by parse_known_args fails the suite instead of
      shipping another ignored contract.

ISOLATION
  The module is loaded from its file path under a synthetic name with PyQt6
  imports blocked (mirrors tests/skills/test_hud_geometry_clamp.py), so
  main() exits headless at the PyQt6 gate after the arg handling under test.
  The staging path points into a tempfile dir; main() performs no file I/O
  before the gate, and nothing under the project tree is written.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
import unittest
from unittest import mock


_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_HUD_DIR = os.path.join(_PROJECT_DIR, "hud")
_HUD_FILENAME = "jarvis_unified_hud.py"


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _function_body(source: str, func_name: str) -> str:
    """Source region of top-level ``def <func_name>`` — def line up to the
    next non-indented line. Text-level on purpose: bobert_companion must not
    be imported (its singleton lock sys.exit()s in a foreign process)."""
    m = re.search(rf"^def {re.escape(func_name)}\(", source, re.M)
    if not m:
        raise AssertionError(f"def {func_name}( not found")
    start = m.start()
    nxt = re.compile(r"^\S", re.M).search(source, m.end())
    return source[start:nxt.start()] if nxt else source[start:]


def _load_hud_no_pyqt(testcase, filename, mod_name):
    """Load a HUD source from hud/<filename> with PyQt6 import blocked, so the
    module takes its graceful-degrade path (``_HAS_PYQT6`` False, Qt names
    stubbed). Restores sys.modules + the real importer on cleanup."""
    path = os.path.join(_HUD_DIR, filename)
    real_import = __import__

    def _imp(name, *a, **k):
        if name.split(".")[0] == "PyQt6":
            raise ImportError(f"[test] PyQt6 blocked: {name}")
        return real_import(name, *a, **k)

    hidden = {n: sys.modules.pop(n)
              for n in list(sys.modules) if n.split(".")[0] == "PyQt6"}
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module

    def restore():
        sys.modules.pop(mod_name, None)
        sys.modules.update(hidden)

    testcase.addCleanup(restore)
    with mock.patch("builtins.__import__", side_effect=_imp):
        spec.loader.exec_module(module)
    testcase.assertFalse(module._HAS_PYQT6,
                         "PyQt6 should be blocked → headless degrade path")
    return module


class StateFileArgRepointsGlobals(unittest.TestCase):
    MOD_NAME = "_ju_hud_statefile_under_test"

    def setUp(self):
        self.mod = _load_hud_no_pyqt(self, _HUD_FILENAME, self.MOD_NAME)
        self.prod_paths = {
            "HUD_STATE_FILE": self.mod.HUD_STATE_FILE,
            "CONTROL_FILE": self.mod.CONTROL_FILE,
            "GEOMETRY_FILE": self.mod.GEOMETRY_FILE,
        }

    def _run_main(self, argv_tail):
        with mock.patch.object(sys, "argv",
                               [_HUD_FILENAME] + list(argv_tail)):
            return self.mod.main()

    def test_state_file_repoints_all_three_globals_into_role_dir(self):
        tmp = tempfile.mkdtemp(prefix="jarvis_hud_staging_")
        self.addCleanup(lambda: os.path.isdir(tmp) and os.rmdir(tmp))
        staging_state = os.path.join(tmp, "hud_state.json")
        rc = self._run_main(["--state-file", staging_state,
                             "--role", "staging"])
        # Headless: main() must reach the PyQt6 gate (exit 2) AFTER the
        # repoint — that ordering is what makes staging safe even degraded.
        self.assertEqual(rc, 2)
        self.assertEqual(self.mod.HUD_STATE_FILE, staging_state)
        # Control + geometry siblings live in the state file's directory so a
        # staging HUD can't chase prod control or clobber prod geometry.
        norm_tmp = os.path.normcase(os.path.abspath(tmp))
        for name in ("CONTROL_FILE", "GEOMETRY_FILE"):
            val = getattr(self.mod, name)
            self.assertEqual(
                os.path.normcase(os.path.dirname(os.path.abspath(val))),
                norm_tmp,
                f"{name} = {val!r} escaped the staging dir")
            self.assertNotEqual(os.path.normcase(os.path.abspath(val)),
                                os.path.normcase(os.path.abspath(
                                    self.prod_paths[name])),
                                f"{name} still points at the prod path")
        # The two siblings must not collide with each other or the state file.
        vals = {os.path.normcase(os.path.abspath(getattr(self.mod, n)))
                for n in ("HUD_STATE_FILE", "CONTROL_FILE", "GEOMETRY_FILE")}
        self.assertEqual(len(vals), 3, "per-role files must be distinct")

    def test_default_argv_keeps_prod_paths(self):
        rc = self._run_main([])
        self.assertEqual(rc, 2)
        for name, prod in self.prod_paths.items():
            self.assertEqual(getattr(self.mod, name), prod,
                             f"{name} changed on a default (prod) launch")

    def test_role_flag_alone_keeps_prod_paths(self):
        # --role without --state-file (prod boot through the launcher) must
        # not repoint anything.
        rc = self._run_main(["--role", "prod"])
        self.assertEqual(rc, 2)
        for name, prod in self.prod_paths.items():
            self.assertEqual(getattr(self.mod, name), prod)

    def test_unknown_flags_still_tolerated(self):
        # parse_known_args() must keep tolerating a NEWER launcher's extra
        # flags (the forward-compat half of the old comment that stays true).
        rc = self._run_main(["--future-flag", "x", "--role", "staging"])
        self.assertEqual(rc, 2)


class LauncherFlagsAreDeclaredInvariant(unittest.TestCase):
    """Every --flag _launch_hud passes must be declared in the HUD's main()."""

    def test_every_launcher_flag_is_declared_by_the_hud(self):
        launch_body = _function_body(
            _read_text(os.path.join(_PROJECT_DIR, "bobert_companion.py")),
            "_launch_hud")
        launcher_flags = set(re.findall(r'"(--[a-z][a-z0-9-]*)"',
                                        launch_body))
        self.assertTrue(launcher_flags,
                        "_launch_hud no longer builds a --flag argv — "
                        "update this extractor alongside the launcher")
        # Sanity floor: the blue/green pair must be part of what we check.
        self.assertTrue({"--role", "--state-file"} <= launcher_flags,
                        f"launcher no longer passes the blue/green flags: "
                        f"{sorted(launcher_flags)}")
        hud_main = _function_body(
            _read_text(os.path.join(_HUD_DIR, _HUD_FILENAME)), "main")
        declared = set(re.findall(r'add_argument\(\s*"(--[^"]+)"', hud_main))
        undeclared = sorted(launcher_flags - declared)
        self.assertEqual(
            undeclared, [],
            f"hud/{_HUD_FILENAME} main() silently drops launcher flag(s) "
            f"{undeclared} via parse_known_args — declare them (and honour "
            f"them) so the launcher contract can't rot invisibly again")


if __name__ == "__main__":
    unittest.main()
