"""tools/bounce_jarvis.py restarts the LIVE JARVIS only, never a staging one
(audit A103, 2026-10-02).

The script killed every python process whose command line named
bobert_companion, so a bounce also took down a blue/green "green" candidate,
a staging sweep or a test instance. The repo marks a staging JARVIS with the
same two signals everywhere: ``--staging`` on the command line (the prod-only
killers in tools/multi_agent_pipeline.py, upgrade_jarvis.py and the run-jarvis
driver) or ``JARVIS_STAGING=1`` in its environment (core.paths.is_staging,
blue_green_manager.is_staging and the monolith's own singleton-lock choice).

SAFETY: the old script did all of its work AT IMPORT -- importing it killed the
running JARVIS and relaunched it. So every test first checks, by AST and before
anything imports the file, that it has no import-time side effects, and only
then loads it. psutil and winreg are fakes; nothing is killed or spawned.
"""
from __future__ import annotations

import ast
import importlib.util
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "tools", "bounce_jarvis.py")

# Calls that act on the box (kill, spawn, sleep, registry, exit).
_ACTING_CALLS = {"kill", "terminate", "process_iter", "Popen", "run", "call",
                 "sleep", "OpenKey", "QueryValueEx", "exit", "main"}


def _is_main_guard(node) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__")


def _import_time_actions(path: str) -> list[str]:
    """Top-level statements of ``path`` that would act when it is imported."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    hits = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) or _is_main_guard(node):
            continue
        if isinstance(node, (ast.Try, ast.For, ast.While, ast.With)):
            hits.append(f"line {node.lineno}: top-level {type(node).__name__}")
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name in _ACTING_CALLS:
                    hits.append(f"line {sub.lineno}: {name}()")
    return hits


class _FakeProc:
    def __init__(self, pid, name, cmdline, env=None, env_error=None,
                 kill_error=None):
        self.pid = pid
        self.info = {"pid": pid, "name": name, "cmdline": cmdline}
        self._env = env if env is not None else {}
        self._env_error = env_error
        self._kill_error = kill_error
        self.killed = False

    def environ(self):
        if self._env_error is not None:
            raise self._env_error
        return dict(self._env)

    def kill(self):
        if self._kill_error is not None:
            raise self._kill_error
        self.killed = True


def _fake_psutil(procs):
    m = types.ModuleType("psutil")
    m.process_iter = lambda attrs=None: iter(list(procs))
    return m


def _fake_winreg():
    m = types.ModuleType("winreg")
    m.HKEY_CURRENT_USER = object()

    def _open_key(*_a, **_k):
        raise OSError("no registry in this test")
    m.OpenKey = _open_key
    return m


_PY = "C:\\Python314\\python.exe"
_PYW = "C:\\Python314\\pythonw.exe"


class BounceJarvisTargetsTests(unittest.TestCase):

    def _load(self):
        hits = _import_time_actions(_PATH)
        if hits:
            self.fail("tools/bounce_jarvis.py acts at import time, so it cannot "
                      "be loaded safely (it would bounce the real JARVIS): "
                      + "; ".join(hits))
        spec = importlib.util.spec_from_file_location("_bounce_jarvis_under_test",
                                                      _PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _kill(self, mod, procs):
        with mock.patch.dict(sys.modules, {"psutil": _fake_psutil(procs)}):
            return mod.kill_live_jarvis()

    def test_the_script_has_no_import_time_side_effects(self):
        self.assertEqual(_import_time_actions(_PATH), [])

    def test_only_the_live_instance_is_killed(self):
        mod = self._load()
        live = _FakeProc(101, "python.exe", [_PY, "-u", "bobert_companion.py"])
        live_w = _FakeProc(102, "pythonw.exe", [_PYW, "bobert_companion.py"])
        flag_staging = _FakeProc(201, "python.exe",
                                 [_PY, "-u", "bobert_companion.py", "--staging"])
        env_staging = _FakeProc(202, "python.exe", [_PY, "-u", "bobert_companion.py"],
                                env={"JARVIS_STAGING": "1", "MUTE_TTS": "1"})
        unrelated = _FakeProc(301, "python.exe", [_PY, "tools/run_tests.py"])
        not_python = _FakeProc(302, "notepad.exe", ["notepad.exe",
                                                    "bobert_companion.py"])
        procs = [live, flag_staging, env_staging, unrelated, not_python, live_w]
        killed = self._kill(mod, procs)
        self.assertEqual(sorted(killed), [101, 102])
        self.assertTrue(live.killed and live_w.killed)
        for spared in (flag_staging, env_staging, unrelated, not_python):
            self.assertFalse(spared.killed, spared.info)

    def test_an_unreadable_environment_falls_back_to_the_command_line(self):
        mod = self._load()
        live = _FakeProc(101, "python.exe", [_PY, "bobert_companion.py"],
                         env_error=PermissionError("access denied"))
        staging = _FakeProc(201, "python.exe",
                            [_PY, "bobert_companion.py", "--staging"],
                            env_error=PermissionError("access denied"))
        self.assertEqual(self._kill(mod, [live, staging]), [101])
        self.assertFalse(staging.killed)

    def test_a_staging_flag_of_zero_is_not_staging(self):
        mod = self._load()
        live = _FakeProc(101, "python.exe", [_PY, "bobert_companion.py"],
                         env={"JARVIS_STAGING": "0"})
        self.assertEqual(self._kill(mod, [live]), [101])

    def test_a_process_that_vanishes_mid_kill_does_not_stop_the_sweep(self):
        mod = self._load()
        gone = _FakeProc(101, "python.exe", [_PY, "bobert_companion.py"],
                         kill_error=ProcessLookupError("no such process"))
        live = _FakeProc(102, "python.exe", [_PY, "bobert_companion.py"])
        self.assertEqual(self._kill(mod, [gone, live]), [102])

    def test_main_kills_through_the_live_only_filter_then_relaunches(self):
        mod = self._load()
        launched = mock.MagicMock(pid=4242)
        with mock.patch.object(mod, "kill_live_jarvis", return_value=[101]) as kill, \
             mock.patch.object(mod.time, "sleep"), \
             mock.patch.object(mod.subprocess, "Popen",
                               return_value=launched) as popen, \
             mock.patch.dict(sys.modules, {"winreg": _fake_winreg()}), \
             redirect_stdout(io.StringIO()):
            rc = mod.main()
        kill.assert_called_once_with()
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(popen.call_args.args[0][1:], ["bobert_companion.py"])
        self.assertNotIn("--staging", popen.call_args.args[0])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
