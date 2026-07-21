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

import ast
import os
import re
import shutil
import sys
import tempfile
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


# ─── ineffective builtins.open mocks around atomic writers ────────────────
#
# THE SECOND INCIDENT OF 2026-07-21: tests/skills/test_sh_ecobee.py's
# test_save_tokens_swallows_errors patched builtins.open with an OSError
# side_effect and called sh_ecobee._save_tokens — but _save_tokens writes via
# core.atomic_io._atomic_write_json (tempfile.mkstemp + os.fdopen +
# os.replace), which never touches builtins.open. The mock never fired, the
# write SUCCEEDED, and a targeted run of that test file on the live box
# overwrote the owner's real data/sh_ecobee_tokens.json with the fake fixture
# tokens. The rule these helpers encode: a builtins.open mock can NOT block an
# atomic mkstemp write — inject failures at core.atomic_io._atomic_write_json
# instead (and redirect the module's path globals to a tempdir).

def _terminal_call_name(func: ast.expr) -> str | None:
    """'x.y._save_tokens(...)' → '_save_tokens'; 'open(...)' → 'open'."""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _unblockable_writers_in(path: str, root: str) -> dict[str, str]:
    """Functions in ONE source file whose body reaches _atomic_write_json with
    no plain open()/os.makedirs call on an earlier line (either of those gives
    a patched open/makedirs a chance to fire first, like sh_ring._save_token's
    leading os.makedirs). Returns {func_name: "relpath:lineno"}."""
    out: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return out
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        atomic_line = None
        guard_lines = []
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            name = _terminal_call_name(sub.func)
            if name == "_atomic_write_json":
                if atomic_line is None or sub.lineno < atomic_line:
                    atomic_line = sub.lineno
            elif name in ("open", "makedirs"):
                guard_lines.append(sub.lineno)
        if atomic_line is None or any(g < atomic_line for g in guard_lines):
            continue
        out[node.name] = f"{os.path.relpath(path, root)}:{node.lineno}"
    return out


def _modules_under_test(root: str, test_path: str) -> list[str]:
    """Source files a test file exercises, by naming convention: tests/skills/
    test_X.py → skills/X; tests/test_X.py → the X module wherever it lives.
    tests/monolith/* targets bobert_companion.py. A trailing _sec<N> shard
    suffix is stripped (test_actions_sec3 → actions)."""
    modname = os.path.basename(test_path)[len("test_"):-len(".py")]
    modname = re.sub(r"_sec\d+$", "", modname)
    if os.path.basename(os.path.dirname(test_path)) == "monolith" \
            or modname == "monolith":
        return [os.path.join(root, "bobert_companion.py")]
    cands = []
    for sub in ("skills", "core", "tools", "audio", "hud", ""):
        cands.append(os.path.join(root, sub, f"{modname}.py"))
        cands.append(os.path.join(root, sub, modname, "__init__.py"))
    return [c for c in cands if os.path.isfile(c)]


def _is_builtins_open_side_effect_patch(call: ast.expr) -> bool:
    """Matches mock.patch("builtins.open", side_effect=...) / bare patch(...)."""
    if not isinstance(call, ast.Call) or _terminal_call_name(call.func) != "patch":
        return False
    if not call.args:
        return False
    a0 = call.args[0]
    if not (isinstance(a0, ast.Constant) and a0.value == "builtins.open"):
        return False
    return any(kw.arg == "side_effect" for kw in call.keywords)


def _scan_for_ineffective_open_mocks(root: str) -> list[str]:
    """Offender list: with-blocks that patch builtins.open with a side_effect
    and call a module-under-test function the patch cannot block (its write
    goes through core.atomic_io._atomic_write_json — mkstemp, not open)."""
    offenders: list[str] = []
    writer_cache: dict[str, dict[str, str]] = {}
    tests_dir = os.path.join(root, "tests")
    for base, dirs, files in os.walk(tests_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in sorted(files):
            if not (fn.startswith("test_") and fn.endswith(".py")):
                continue
            path = os.path.join(base, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
            except OSError:
                continue
            if "builtins.open" not in src:
                continue
            writers: dict[str, str] = {}
            for mod_path in _modules_under_test(root, path):
                if mod_path not in writer_cache:
                    writer_cache[mod_path] = _unblockable_writers_in(mod_path,
                                                                     root)
                writers.update(writer_cache[mod_path])
            if not writers:
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.With):
                    continue
                if not any(_is_builtins_open_side_effect_patch(i.context_expr)
                           for i in node.items):
                    continue
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        name = _terminal_call_name(sub.func)
                        if name in writers:
                            offenders.append(
                                f"{os.path.relpath(path, root)}:{sub.lineno} "
                                f"calls {name}() (defined {writers[name]}) "
                                "inside a builtins.open side_effect patch")
    return offenders


class NoIneffectiveOpenMockTests(unittest.TestCase):
    """Tree-wide invariant: no test may 'block' an atomic write by patching
    builtins.open — the seed offender clobbered the live Ecobee OAuth tokens."""

    def test_no_builtins_open_mock_around_atomic_writer(self):
        offenders = _scan_for_ineffective_open_mocks(_PROJECT_ROOT)
        self.assertEqual(
            offenders, [],
            "Test(s) patch builtins.open with a side_effect around a function "
            "that writes via core.atomic_io._atomic_write_json (mkstemp + "
            "os.replace — open() is never called). The mock cannot fire, the "
            "write REALLY happens against the module's configured path; this "
            "exact mistake overwrote the live data/sh_ecobee_tokens.json "
            "(2026-07-21). Patch core.atomic_io._atomic_write_json instead "
            "and redirect the module's path globals to a tempdir. Offenders: "
            + "; ".join(offenders))

    def test_scanner_catches_the_seed_pattern(self):
        """Self-test on a synthetic tree so the guard above can never rot into
        a scanner that silently matches nothing. Reproduces the pre-fix ecobee
        shape (flagged) and the sh_ring shape (makedirs first — exempt)."""
        root = tempfile.mkdtemp(prefix="openmock_scan_selftest_")
        self.addCleanup(shutil.rmtree, root, True)
        os.makedirs(os.path.join(root, "skills"))
        os.makedirs(os.path.join(root, "tests", "skills"))

        def w(rel, text):
            with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                fh.write(text)

        w(os.path.join("skills", "foo.py"),
          "def _save_tokens(service):\n"
          "    try:\n"
          "        from core.atomic_io import _atomic_write_json\n"
          "        _atomic_write_json('p.json', {'a': 1})\n"
          "    except Exception:\n"
          "        pass\n")
        w(os.path.join("skills", "bar.py"),
          "import os\n"
          "def _save_token(tok):\n"
          "    try:\n"
          "        os.makedirs('d', exist_ok=True)\n"
          "        from core.atomic_io import _atomic_write_json\n"
          "        _atomic_write_json('d/p.json', tok)\n"
          "    except Exception:\n"
          "        pass\n")
        for name in ("foo", "bar"):
            fn = "_save_tokens" if name == "foo" else "_save_token"
            w(os.path.join("tests", "skills", f"test_{name}.py"),
              "from unittest import mock\n"
              "def test_swallows():\n"
              "    with mock.patch('builtins.open',\n"
              "                    side_effect=OSError('ro fs')):\n"
              f"        {fn}(object())\n")

        offenders = _scan_for_ineffective_open_mocks(root)
        self.assertEqual(len(offenders), 1, offenders)
        self.assertIn("test_foo.py", offenders[0])
        self.assertIn("_save_tokens", offenders[0])


class RunnerDataDirRedirectTests(unittest.TestCase):
    """Every suite runner (run_tests, run_tests_ci_sim, run_coverage) must
    point JARVIS_DATA_DIR at a throwaway BEFORE any test is imported, so no
    suite-run process resolves data_dir() to the live data/ even when a future
    test forgets its own path redirect."""

    def setUp(self):
        self._saved = os.environ.get(paths.DATA_DIR_ENV)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(paths.DATA_DIR_ENV, None)
        else:
            os.environ[paths.DATA_DIR_ENV] = self._saved

    def _assert_redirects(self, redirect_fn):
        os.environ.pop(paths.DATA_DIR_ENV, None)
        redirect_fn()
        val = (os.environ.get(paths.DATA_DIR_ENV) or "").strip()
        self.assertTrue(val, "redirect did not set JARVIS_DATA_DIR")
        self.assertTrue(os.path.isdir(val), val)
        self.assertNotEqual(os.path.normcase(os.path.abspath(val)),
                            os.path.normcase(os.path.join(_PROJECT_ROOT,
                                                          "data")))

    def test_run_tests_redirects_data_dir(self):
        from tools import run_tests as rt
        self._assert_redirects(rt._redirect_data_dir_to_throwaway)

    def test_ci_sim_redirects_data_dir(self):
        from tools import run_tests_ci_sim as cs
        self._assert_redirects(cs._redirect_data_dir_to_throwaway)

    def test_redirect_respects_external_override(self):
        from tools import run_tests as rt
        os.environ[paths.DATA_DIR_ENV] = "preset_by_operator"
        rt._redirect_data_dir_to_throwaway()
        self.assertEqual(os.environ[paths.DATA_DIR_ENV], "preset_by_operator")

    def test_runners_call_redirect_before_discovery(self):
        # run_coverage.py discovers the same suite (inside _run_suite), so it
        # needs the identical guard — it imports run_tests' implementation.
        for fn in ("run_tests.py", "run_tests_ci_sim.py", "run_coverage.py"):
            path = os.path.join(_PROJECT_ROOT, "tools", fn)
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
            call = src.find("\n    _redirect_data_dir_to_throwaway()")
            disc = src.find(".discover(")
            self.assertGreater(call, -1, f"{fn}: redirect is never called")
            self.assertGreater(disc, -1, f"{fn}: no discovery call found")
            self.assertLess(call, disc,
                            f"{fn}: data-dir redirect must run before "
                            "test discovery/import")


if __name__ == "__main__":
    unittest.main()
