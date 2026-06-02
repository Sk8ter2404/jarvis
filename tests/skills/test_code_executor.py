"""Logic tests for skills/code_executor.py.

code_executor runs Python in a sandboxed subprocess (or a persistent jupyter
kernel). We exercise the parsing, formatting, config, the credential-stripping
security guard, the process-tree teardown, the subprocess timeout/error paths,
and the FULL jupyter backend — all WITHOUT running untrusted code or escaping
the sandbox:

  * _strip_fence / _truncate / _format_result are pure — tested directly.
  * _sandbox_env (the security mitigation that removes credential-shaped env
    vars before a snippet inherits the environment) is tested directly.
  * _config_timeout / _config_backend env + bobert-attr + default resolution.
  * _kill_process_tree — the psutil descendant-walk, the Windows taskkill
    fallback (psutil forced absent), and the POSIX os.kill last resort. No real
    process is ever signalled: psutil / subprocess.run / os.kill are faked.
  * _run_subprocess — Popen is MOCKED so nothing is spawned; we drive the
    normal-completion, TimeoutExpired (→ tree kill), Popen-construction-error,
    and generic-exception paths, plus the temp-file unlink cleanup.
  * run_python dispatch to subprocess vs jupyter (both runners mocked).
  * jupyter backend (jupyter_client is NOT on CI → a fake module is injected
    and removed in the same test): _ensure_kernel success / import-absent /
    start-failure, _run_jupyter stream+result+error+idle handling, timeout
    with a live kernel (interrupt) and with a dead kernel (subprocess
    fallback), kernel-absent fallback, and _shutdown_kernel.

Exactly one end-to-end execution runs a trivial, safe `print(2 + 2)` snippet to
prove the local-exec path the skill is designed around actually works. Nothing
unsafe is ever executed.
"""
from __future__ import annotations

import contextlib
import os
import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install/remove fake modules in ``sys.modules`` for the
    duration of a block, restoring prior state — including absence — afterwards.
    For dotted names (``jupyter_client.manager``) the leaf is ALSO set as an
    attribute on its already-imported parent package so ``from
    jupyter_client.manager import KernelManager`` resolves OUR fake even when a
    real parent package is present. ``obj=None`` removes the module so a lazy
    import misses."""
    saved_mod: dict[str, object] = {}
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
            else:
                setattr(parent, leaf, prev)
        for name, prev in saved_mod.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


# ─── pure parsing / formatting ───────────────────────────────────────────
class StripFenceTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")

    def test_strips_python_fence(self):
        self.assertEqual(self.mod._strip_fence("```python\nprint(1)\n```"), "print(1)")

    def test_strips_py_fence(self):
        self.assertEqual(self.mod._strip_fence("```py\nprint(1)\n```"), "print(1)")

    def test_strips_bare_fence(self):
        self.assertEqual(self.mod._strip_fence("```\nx = 5\n```"), "x = 5")

    def test_leaves_unfenced_code(self):
        self.assertEqual(self.mod._strip_fence("print(42)"), "print(42)")

    def test_multiline_fenced_body_preserved(self):
        out = self.mod._strip_fence("```python\na = 1\nb = 2\nprint(a + b)\n```")
        self.assertEqual(out, "a = 1\nb = 2\nprint(a + b)")


class TruncateTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")

    def test_short_text_unchanged(self):
        self.assertEqual(self.mod._truncate("hello"), "hello")

    def test_exactly_at_limit_unchanged(self):
        s = "x" * self.mod._MAX_OUTPUT_CHARS
        self.assertEqual(self.mod._truncate(s), s)

    def test_long_text_truncated_with_marker(self):
        big = "x" * (self.mod._MAX_OUTPUT_CHARS + 500)
        out = self.mod._truncate(big)
        self.assertLess(len(out), len(big))
        self.assertIn("chars truncated", out)


class FormatResultTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")

    def test_ok_header_and_stdout(self):
        out = self.mod._format_result("the answer is 4", "", 0, 0.5)
        self.assertIn("[ok in 0.5s]", out)
        self.assertIn("stdout:", out)
        self.assertIn("the answer is 4", out)

    def test_nonzero_exit_header(self):
        out = self.mod._format_result("", "Traceback...", 1, 0.2)
        self.assertIn("[exited with code 1", out)
        self.assertIn("stderr:", out)

    def test_timeout_header(self):
        out = self.mod._format_result("partial", "", None, 30.0)
        self.assertIn("timeout after 30.0s", out)
        self.assertIn("process killed", out)

    def test_no_output_success(self):
        out = self.mod._format_result("", "", 0, 0.1)
        self.assertIn("(no output)", out)

    def test_both_streams_present(self):
        out = self.mod._format_result("out-text", "err-text", 0, 0.3)
        self.assertIn("stdout:", out)
        self.assertIn("out-text", out)
        self.assertIn("stderr:", out)
        self.assertIn("err-text", out)

    def test_failure_with_no_output_omits_no_output_line(self):
        # rc != 0 with no streams → header only, never the "(no output)" line
        # (that line is reserved for the rc==0 case).
        out = self.mod._format_result("", "", 5, 0.1)
        self.assertIn("[exited with code 5", out)
        self.assertNotIn("(no output)", out)


# ─── credential-stripping guard ──────────────────────────────────────────
class SandboxEnvTests(unittest.TestCase):
    """The credential-stripping guard is the security-critical bit."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")

    def test_strips_credential_shaped_vars(self):
        secrets = {
            "ANTHROPIC_API_KEY": "sk-xxx",
            "TELEGRAM_BOT_TOKEN": "123:abc",
            "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_x",
            "MY_SECRET": "shh",
            "DB_PASSWORD": "pw",
            "SOME_PRIVATE_KEY": "----",
            "PORCUPINE_ACCESS_KEY": "ak",
            "PATH": "/usr/bin",          # benign — must survive
            "NORMAL_VAR": "keep me",
        }
        with mock.patch.dict(os.environ, secrets, clear=True):
            env = self.mod._sandbox_env()
        for k in ("ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN",
                  "GITHUB_PERSONAL_ACCESS_TOKEN", "MY_SECRET", "DB_PASSWORD",
                  "SOME_PRIVATE_KEY", "PORCUPINE_ACCESS_KEY"):
            self.assertNotIn(k, env)
        self.assertEqual(env.get("PATH"), "/usr/bin")
        self.assertEqual(env.get("NORMAL_VAR"), "keep me")

    def test_case_insensitive_match(self):
        with mock.patch.dict(os.environ, {"lowercase_api_key": "x", "benign": "y"},
                             clear=True):
            env = self.mod._sandbox_env()
        upper = {k.upper() for k in env}
        self.assertNotIn("LOWERCASE_API_KEY", upper)
        self.assertIn("BENIGN", upper)

    def test_passwd_and_credential_substrings_stripped(self):
        with mock.patch.dict(os.environ,
                             {"SUDO_PASSWD": "x", "AWS_CREDENTIAL_FILE": "y",
                              "AUTH_TOKEN": "z", "HOME": "/home/me"},
                             clear=True):
            env = self.mod._sandbox_env()
        self.assertNotIn("SUDO_PASSWD", env)
        self.assertNotIn("AWS_CREDENTIAL_FILE", env)
        self.assertNotIn("AUTH_TOKEN", env)
        self.assertEqual(env.get("HOME"), "/home/me")


# ─── config resolution (env + bobert attr + default) ─────────────────────
class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")

    def test_timeout_default(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.mod, "_bobert", return_value=None):
            self.assertEqual(self.mod._config_timeout(), self.mod._DEFAULT_TIMEOUT)

    def test_timeout_from_env(self):
        with mock.patch.dict(os.environ, {"CODE_EXECUTOR_TIMEOUT": "12"}, clear=True):
            self.assertEqual(self.mod._config_timeout(), 12)

    def test_timeout_bad_value_falls_back(self):
        with mock.patch.dict(os.environ, {"CODE_EXECUTOR_TIMEOUT": "soon"}, clear=True), \
             mock.patch.object(self.mod, "_bobert", return_value=None):
            self.assertEqual(self.mod._config_timeout(), self.mod._DEFAULT_TIMEOUT)

    def test_timeout_clamped_to_minimum_one(self):
        with mock.patch.dict(os.environ, {"CODE_EXECUTOR_TIMEOUT": "0"}, clear=True):
            self.assertEqual(self.mod._config_timeout(), 1)   # max(1, 0)

    def test_timeout_from_bobert_attr_when_env_absent(self):
        # No env var, but the resolved bobert module carries the attribute.
        bc = types.SimpleNamespace(CODE_EXECUTOR_TIMEOUT=45)
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(self.mod._config_timeout(), 45)

    def test_timeout_bobert_attr_empty_falls_to_default(self):
        # bobert carries the attribute but it's empty → ignored, default used.
        bc = types.SimpleNamespace(CODE_EXECUTOR_TIMEOUT="")
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(self.mod._config_timeout(), self.mod._DEFAULT_TIMEOUT)

    def test_backend_default_is_subprocess(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.mod, "_bobert", return_value=None):
            self.assertEqual(self.mod._config_backend(), "subprocess")

    def test_backend_invalid_falls_back(self):
        with mock.patch.dict(os.environ, {"CODE_EXECUTOR_BACKEND": "docker"}, clear=True):
            self.assertEqual(self.mod._config_backend(), "subprocess")

    def test_backend_jupyter_honoured(self):
        with mock.patch.dict(os.environ, {"CODE_EXECUTOR_BACKEND": "JUPYTER"}, clear=True):
            self.assertEqual(self.mod._config_backend(), "jupyter")

    def test_backend_from_bobert_attr(self):
        bc = types.SimpleNamespace(CODE_EXECUTOR_BACKEND="jupyter")
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(self.mod._config_backend(), "jupyter")

    def test_bobert_resolves_main_module(self):
        # _bobert prefers __main__; just assert it returns a module object or
        # None without raising (it reads sys.modules).
        out = self.mod._bobert()
        self.assertTrue(out is None or isinstance(out, types.ModuleType))


# ─── _kill_process_tree ──────────────────────────────────────────────────
class KillProcessTreeTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")

    def _make_psutil(self, *, children=None, parent_raises_nosuch=False,
                     wait_raises=False, parent_kill_raises=False):
        psutil = types.ModuleType("psutil")

        class NoSuchProcess(Exception):
            pass

        killed = {"children": [], "parent": False}

        class _Proc:
            def __init__(self, pid):
                self.pid = pid
                if parent_raises_nosuch:
                    raise NoSuchProcess(pid)

            def children(self, recursive=False):
                return children or []

            def kill(self):
                if parent_kill_raises:
                    raise RuntimeError("parent kill nope")
                killed["parent"] = True

        def _wait_procs(procs, timeout=None):
            if wait_raises:
                raise RuntimeError("wait boom")
            return ([], [])

        psutil.Process = _Proc
        psutil.NoSuchProcess = NoSuchProcess
        psutil.wait_procs = _wait_procs
        psutil._killed = killed
        return psutil

    def test_kills_parent_and_children_via_psutil(self):
        child_killed = []

        class _Child:
            def kill(self):
                child_killed.append(True)
        psutil = self._make_psutil(children=[_Child(), _Child()])
        with inject_modules(psutil=psutil):
            self.mod._kill_process_tree(1234)
        self.assertTrue(psutil._killed["parent"])
        self.assertEqual(len(child_killed), 2)

    def test_no_such_process_returns_quietly(self):
        psutil = self._make_psutil(parent_raises_nosuch=True)
        with inject_modules(psutil=psutil):
            self.mod._kill_process_tree(999)   # must not raise

    def test_child_kill_exception_swallowed(self):
        class _BadChild:
            def kill(self):
                raise RuntimeError("child nope")
        psutil = self._make_psutil(children=[_BadChild()])
        with inject_modules(psutil=psutil):
            self.mod._kill_process_tree(1)   # must not raise
        self.assertTrue(psutil._killed["parent"])

    def test_wait_procs_exception_swallowed(self):
        psutil = self._make_psutil(children=[], wait_raises=True)
        with inject_modules(psutil=psutil):
            self.mod._kill_process_tree(1)   # must not raise

    def test_parent_kill_exception_swallowed(self):
        # parent.kill() raising is caught by its own try/except; wait_procs
        # still runs and the call returns cleanly.
        psutil = self._make_psutil(children=[], parent_kill_raises=True)
        with inject_modules(psutil=psutil):
            self.mod._kill_process_tree(1)   # must not raise

    def test_windows_taskkill_fallback_when_no_psutil(self):
        # psutil import fails → on win32 the taskkill branch runs. subprocess.run
        # is mocked so no real taskkill is spawned.
        runs = []
        real_import = __import__

        def _blocked(name, *a, **k):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *a, **k)
        # The taskkill branch passes ``creationflags=subprocess.CREATE_NO_WINDOW``
        # — a Windows-only constant absent on the Linux CI runner. With
        # sys.platform forced to "win32" that lookup raises AttributeError, which
        # _kill_process_tree swallows, so taskkill never runs (len(runs)==0).
        # Materialise the flag (no-op on Windows) so the win32 branch runs and is
        # covered on any host.
        with inject_modules(psutil=None), \
             mock.patch("builtins.__import__", side_effect=_blocked), \
             mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "CREATE_NO_WINDOW",
                               0x08000000, create=True), \
             mock.patch.object(self.mod.subprocess, "run",
                               side_effect=lambda *a, **k: runs.append((a, k))):
            self.mod._kill_process_tree(4321)
        self.assertEqual(len(runs), 1)
        # The taskkill argv targets the pid with /F /T.
        argv = runs[0][0][0]
        self.assertIn("taskkill", argv)
        self.assertIn("4321", argv)

    def test_windows_taskkill_exception_swallowed(self):
        real_import = __import__

        def _blocked(name, *a, **k):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *a, **k)
        # Materialise CREATE_NO_WINDOW (Windows-only) so the win32 taskkill
        # branch is what runs on any host and the OSError from run() — not a
        # missing-constant AttributeError — is the exception being swallowed.
        with inject_modules(psutil=None), \
             mock.patch("builtins.__import__", side_effect=_blocked), \
             mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "CREATE_NO_WINDOW",
                               0x08000000, create=True), \
             mock.patch.object(self.mod.subprocess, "run",
                               side_effect=OSError("taskkill missing")):
            self.mod._kill_process_tree(1)   # must not raise

    def test_posix_oskill_fallback_when_no_psutil(self):
        killed = []
        real_import = __import__

        def _blocked(name, *a, **k):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *a, **k)
        with inject_modules(psutil=None), \
             mock.patch("builtins.__import__", side_effect=_blocked), \
             mock.patch.object(self.mod.sys, "platform", "linux"), \
             mock.patch.object(self.mod.os, "kill",
                               side_effect=lambda pid, sig: killed.append((pid, sig))):
            self.mod._kill_process_tree(77)
        self.assertEqual(killed, [(77, 9)])

    def test_posix_oskill_exception_swallowed(self):
        real_import = __import__

        def _blocked(name, *a, **k):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *a, **k)
        with inject_modules(psutil=None), \
             mock.patch("builtins.__import__", side_effect=_blocked), \
             mock.patch.object(self.mod.sys, "platform", "linux"), \
             mock.patch.object(self.mod.os, "kill",
                               side_effect=ProcessLookupError("gone")):
            self.mod._kill_process_tree(77)   # must not raise


# ─── _run_subprocess (Popen mocked — nothing is spawned) ─────────────────
class _FakeProc:
    """Stand-in for a subprocess.Popen handle. communicate() either returns
    (out, err) or raises the supplied exception (e.g. TimeoutExpired)."""
    def __init__(self, *, out="", err="", returncode=0, communicate_exc=None,
                 second_out="", second_err=""):
        self.pid = 4242
        self._out = out
        self._err = err
        self.returncode = returncode
        self._exc = communicate_exc
        self._second = (second_out, second_err)
        self._calls = 0

    def communicate(self, timeout=None):
        self._calls += 1
        if self._exc is not None and self._calls == 1:
            raise self._exc
        if self._calls >= 2:
            return self._second
        return (self._out, self._err)


class RunSubprocessTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")

    def test_normal_completion_formats_result(self):
        proc = _FakeProc(out="hello\n", err="", returncode=0)
        with mock.patch.object(self.mod.subprocess, "Popen", return_value=proc):
            out = self.mod._run_subprocess("print('hello')", timeout=5)
        self.assertIn("[ok", out)
        self.assertIn("hello", out)

    def test_nonzero_returncode_formats_failure(self):
        proc = _FakeProc(out="", err="boom", returncode=1)
        with mock.patch.object(self.mod.subprocess, "Popen", return_value=proc):
            out = self.mod._run_subprocess("raise SystemExit(1)", timeout=5)
        self.assertIn("exited with code 1", out)
        self.assertIn("boom", out)

    def test_timeout_kills_tree_and_reports(self):
        # First communicate() raises TimeoutExpired; the skill kills the tree,
        # then drains a second communicate() for any late output.
        timeout_exc = self.mod.subprocess.TimeoutExpired(cmd="py", timeout=5)
        proc = _FakeProc(communicate_exc=timeout_exc,
                         second_out="partial-out", second_err="")
        with mock.patch.object(self.mod.subprocess, "Popen", return_value=proc), \
             mock.patch.object(self.mod, "_kill_process_tree") as kill:
            out = self.mod._run_subprocess("while True: pass", timeout=5)
        kill.assert_called_once_with(proc.pid)
        self.assertIn("timeout after", out)
        self.assertIn("partial-out", out)

    def test_timeout_drain_failure_still_reports(self):
        # The post-kill drain communicate() also raising → out/err default to "".
        timeout_exc = self.mod.subprocess.TimeoutExpired(cmd="py", timeout=5)

        class _Proc(_FakeProc):
            def communicate(self, timeout=None):
                self._calls += 1
                if self._calls == 1:
                    raise timeout_exc
                raise RuntimeError("drain boom")
        proc = _Proc()
        with mock.patch.object(self.mod.subprocess, "Popen", return_value=proc), \
             mock.patch.object(self.mod, "_kill_process_tree"):
            out = self.mod._run_subprocess("while True: pass", timeout=5)
        self.assertIn("timeout after", out)

    def test_popen_construction_failure_returns_error(self):
        with mock.patch.object(self.mod.subprocess, "Popen",
                               side_effect=OSError("cannot spawn")):
            out = self.mod._run_subprocess("print(1)", timeout=5)
        self.assertIn("execution failed", out)
        self.assertIn("OSError", out)

    def test_generic_communicate_exception_kills_and_reports(self):
        # A non-timeout exception from communicate() → tree kill + error string.
        proc = _FakeProc(communicate_exc=ValueError("weird"))
        with mock.patch.object(self.mod.subprocess, "Popen", return_value=proc), \
             mock.patch.object(self.mod, "_kill_process_tree") as kill:
            out = self.mod._run_subprocess("print(1)", timeout=5)
        kill.assert_called_once()
        self.assertIn("execution failed", out)
        self.assertIn("ValueError", out)

    def test_generic_exception_kill_failure_still_reports(self):
        # communicate() raises a generic error AND the subsequent
        # _kill_process_tree ALSO raises — the inner except swallows the kill
        # failure and the original error string is still returned (258-259).
        proc = _FakeProc(communicate_exc=ValueError("weird"))
        with mock.patch.object(self.mod.subprocess, "Popen", return_value=proc), \
             mock.patch.object(self.mod, "_kill_process_tree",
                               side_effect=RuntimeError("kill nope")):
            out = self.mod._run_subprocess("print(1)", timeout=5)
        self.assertIn("execution failed", out)
        self.assertIn("ValueError", out)

    def test_temp_file_unlink_failure_is_swallowed(self):
        proc = _FakeProc(out="ok\n", returncode=0)
        with mock.patch.object(self.mod.subprocess, "Popen", return_value=proc), \
             mock.patch.object(self.mod.os, "unlink",
                               side_effect=OSError("file locked")):
            out = self.mod._run_subprocess("print('ok')", timeout=5)
        # The OSError from cleanup must not surface; the result still returns.
        self.assertIn("ok", out)


# ─── run_python dispatch ─────────────────────────────────────────────────
class RunPythonDispatchTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("code_executor")

    def test_empty_code_returns_format_hint(self):
        out = self.actions["run_python"]("")
        self.assertIn("format:", out)
        self.assertIn("run_python", out)

    def test_whitespace_only_code_returns_hint(self):
        self.assertIn("format:", self.actions["run_python"]("   \n  "))

    def test_dispatches_to_subprocess_backend(self):
        with mock.patch.object(self.mod, "_config_backend", return_value="subprocess"), \
             mock.patch.object(self.mod, "_run_subprocess",
                               return_value="[ok in 0.0s]\nstdout:\n4") as runner:
            out = self.actions["run_python"]("```python\nprint(2+2)\n```")
        self.assertIn("4", out)
        # Fence must be stripped before the snippet reaches the runner.
        self.assertEqual(runner.call_args[0][0], "print(2+2)")

    def test_jupyter_backend_dispatch(self):
        with mock.patch.object(self.mod, "_config_backend", return_value="jupyter"), \
             mock.patch.object(self.mod, "_run_jupyter", return_value="[ok in 0.0s]") as jrun:
            self.actions["run_python"]("x=1")
        jrun.assert_called_once()

    def test_aliases_all_route_to_run_python(self):
        for alias in ("python", "eval_python", "compute"):
            with mock.patch.object(self.mod, "_config_backend",
                                   return_value="subprocess"), \
                 mock.patch.object(self.mod, "_run_subprocess",
                                   return_value="[ok]") as runner:
                self.actions[alias]("print(1)")
            runner.assert_called_once()

    def test_reset_kernel_when_none_running(self):
        out = self.actions["reset_kernel"]("")
        self.assertIn("no kernel", out.lower())

    def test_register_wires_expected_actions(self):
        actions: dict = {}
        self.mod.register(actions)
        for name in ("run_python", "python", "eval_python", "compute",
                     "reset_kernel"):
            self.assertIn(name, actions)


# ─── jupyter backend (fake jupyter_client injected + removed) ────────────
class _FakeKernelManager:
    """Stand-in for jupyter_client.manager.KernelManager."""
    def __init__(self, *, client, start_raises=False, alive=True):
        self._client = client
        self._start_raises = start_raises
        self._alive = alive
        self.shutdown_called = False
        self.interrupted = False

    def start_kernel(self, env=None):
        if self._start_raises:
            raise RuntimeError("kernel won't start")

    def blocking_client(self):
        return self._client

    def is_alive(self):
        return self._alive

    def interrupt_kernel(self):
        self.interrupted = True

    def shutdown_kernel(self, now=False):
        self.shutdown_called = True


class _FakeKernelClient:
    """Stand-in for a BlockingKernelClient. ``messages`` is a list of iopub
    message dicts handed out one per get_iopub_msg() call; when exhausted it
    raises (simulating a poll timeout) so the loop's deadline logic engages."""
    def __init__(self, messages=None, ready_raises=False):
        self._messages = list(messages or [])
        self._ready_raises = ready_raises
        self.channels_started = False
        self.channels_stopped = False
        self._exec_msg_id = "msg-1"

    def start_channels(self):
        self.channels_started = True

    def stop_channels(self):
        self.channels_stopped = True

    def wait_for_ready(self, timeout=None):
        if self._ready_raises:
            raise RuntimeError("never ready")

    def execute(self, code):
        return self._exec_msg_id

    def get_iopub_msg(self, timeout=None):
        if self._messages:
            return self._messages.pop(0)
        raise RuntimeError("no more messages")


def _msg(msg_type, content, parent="msg-1"):
    return {"msg_type": msg_type, "content": content,
            "parent_header": {"msg_id": parent}}


def make_jupyter_client_pkg(km):
    """Build a fake ``jupyter_client`` package whose ``.manager`` submodule
    exposes a ``KernelManager`` callable returning ``km``. Returned as a dict so
    the caller injects BOTH the package and the dotted submodule."""
    pkg = types.ModuleType("jupyter_client")
    manager = types.ModuleType("jupyter_client.manager")
    manager.KernelManager = lambda *a, **k: km
    pkg.manager = manager
    return {"jupyter_client": pkg, "jupyter_client.manager": manager}


class _JupyterBase(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("code_executor")
        # Always reset the per-process kernel singletons so no handle leaks.
        self.addCleanup(self._reset_kernel_globals)

    def _reset_kernel_globals(self):
        self.mod._kernel = None
        self.mod._kernel_client = None


class EnsureKernelTests(_JupyterBase):
    def test_ensure_kernel_returns_cached_client(self):
        sentinel = object()
        self.mod._kernel_client = sentinel
        self.assertIs(self.mod._ensure_kernel(), sentinel)

    def test_ensure_kernel_none_when_jupyter_absent(self):
        # from jupyter_client.manager import KernelManager raises ImportError.
        real_import = __import__

        def _blocked(name, *a, **k):
            if name.startswith("jupyter_client"):
                raise ImportError("no jupyter_client")
            return real_import(name, *a, **k)
        with inject_modules(**{"jupyter_client": None,
                               "jupyter_client.manager": None}), \
             mock.patch("builtins.__import__", side_effect=_blocked):
            self.assertIsNone(self.mod._ensure_kernel())

    def test_ensure_kernel_success_starts_and_caches(self):
        client = _FakeKernelClient()
        km = _FakeKernelManager(client=client)
        with inject_modules(**make_jupyter_client_pkg(km)):
            kc = self.mod._ensure_kernel()
        self.assertIs(kc, client)
        self.assertTrue(client.channels_started)
        self.assertIs(self.mod._kernel_client, client)

    def test_ensure_kernel_start_failure_returns_none(self):
        client = _FakeKernelClient()
        km = _FakeKernelManager(client=client, start_raises=True)
        with inject_modules(**make_jupyter_client_pkg(km)):
            self.assertIsNone(self.mod._ensure_kernel())
        # Globals stay cleared after a failed start.
        self.assertIsNone(self.mod._kernel_client)


class RunJupyterTests(_JupyterBase):
    def test_falls_back_to_subprocess_when_no_kernel(self):
        with mock.patch.object(self.mod, "_ensure_kernel", return_value=None), \
             mock.patch.object(self.mod, "_run_subprocess",
                               return_value="[ok] sub") as sub:
            out = self.mod._run_jupyter("print(1)", timeout=5)
        sub.assert_called_once()
        self.assertIn("sub", out)

    def test_collects_stream_result_and_idle(self):
        msgs = [
            _msg("stream", {"name": "stdout", "text": "line1\n"}),
            _msg("stream", {"name": "stderr", "text": "warn\n"}),
            _msg("execute_result", {"data": {"text/plain": "42"}}),
            _msg("status", {"execution_state": "idle"}),
        ]
        client = _FakeKernelClient(messages=msgs)
        with mock.patch.object(self.mod, "_ensure_kernel", return_value=client):
            out = self.mod._run_jupyter("40 + 2", timeout=5)
        self.assertIn("[ok", out)
        self.assertIn("line1", out)
        self.assertIn("42", out)
        self.assertIn("warn", out)

    def test_display_data_text_collected(self):
        msgs = [
            _msg("display_data", {"data": {"text/plain": "<figure>"}}),
            _msg("status", {"execution_state": "idle"}),
        ]
        client = _FakeKernelClient(messages=msgs)
        with mock.patch.object(self.mod, "_ensure_kernel", return_value=client):
            out = self.mod._run_jupyter("plot()", timeout=5)
        self.assertIn("<figure>", out)

    def test_error_message_marks_failure(self):
        msgs = [
            _msg("error", {"traceback": ["Traceback", "ValueError: x"]}),
            _msg("status", {"execution_state": "idle"}),
        ]
        client = _FakeKernelClient(messages=msgs)
        with mock.patch.object(self.mod, "_ensure_kernel", return_value=client):
            out = self.mod._run_jupyter("raise ValueError('x')", timeout=5)
        self.assertIn("exited with code 1", out)
        self.assertIn("ValueError", out)

    def test_ignores_messages_from_other_executions(self):
        # A message whose parent msg_id differs is skipped; only the matching
        # idle ends the loop.
        msgs = [
            _msg("stream", {"name": "stdout", "text": "ours\n"}, parent="msg-1"),
            _msg("stream", {"name": "stdout", "text": "theirs\n"}, parent="other"),
            _msg("status", {"execution_state": "idle"}, parent="msg-1"),
        ]
        client = _FakeKernelClient(messages=msgs)
        with mock.patch.object(self.mod, "_ensure_kernel", return_value=client):
            out = self.mod._run_jupyter("print('ours')", timeout=5)
        self.assertIn("ours", out)
        self.assertNotIn("theirs", out)

    def test_get_iopub_exception_continues_until_idle(self):
        # A client whose first poll raises (caught + continue), then yields idle.
        class _FlakyClient(_FakeKernelClient):
            def __init__(self):
                super().__init__(messages=[
                    _msg("status", {"execution_state": "idle"})])
                self._first = True

            def get_iopub_msg(self, timeout=None):
                if self._first:
                    self._first = False
                    raise RuntimeError("poll hiccup")
                return super().get_iopub_msg(timeout=timeout)
        client = _FlakyClient()
        with mock.patch.object(self.mod, "_ensure_kernel", return_value=client):
            out = self.mod._run_jupyter("x=1", timeout=5)
        self.assertIn("[ok", out)

    def test_timeout_with_live_kernel_interrupts(self):
        # No idle ever arrives and the deadline passes → interrupt the (alive)
        # kernel and return a timeout result. Drive time so the deadline trips
        # immediately without real waiting.
        client = _FakeKernelClient(messages=[])   # get_iopub_msg always raises
        km = _FakeKernelManager(client=client, alive=True)
        self.mod._kernel = km
        times = iter([1000.0, 1000.0, 2000.0, 2000.0, 2000.0])
        with mock.patch.object(self.mod, "_ensure_kernel", return_value=client), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=lambda: next(times)):
            out = self.mod._run_jupyter("while True: pass", timeout=5)
        self.assertIn("timeout after", out)
        self.assertTrue(km.interrupted)

    def test_execute_result_without_text_and_busy_status_skipped(self):
        # An execute_result lacking text/plain (no append) and a non-idle
        # 'busy' status both loop back without ending; the matching idle ends
        # it. Exercises the 354->loop and 359->loop back-edges.
        msgs = [
            _msg("execute_result", {"data": {"image/png": "..."}}),  # no text
            _msg("status", {"execution_state": "busy"}),             # not idle
            _msg("stream", {"name": "stdout", "text": "done\n"}),
            _msg("status", {"execution_state": "idle"}),
        ]
        client = _FakeKernelClient(messages=msgs)
        with mock.patch.object(self.mod, "_ensure_kernel", return_value=client):
            out = self.mod._run_jupyter("x", timeout=5)
        self.assertIn("[ok", out)
        self.assertIn("done", out)

    def test_timeout_is_alive_raises_falls_back_to_subprocess(self):
        # _kernel.is_alive() raising at the deadline → kernel treated as dead →
        # subprocess fallback (323-324).
        client = _FakeKernelClient(messages=[])

        class _BoomAlive(_FakeKernelManager):
            def is_alive(self):
                raise RuntimeError("is_alive boom")
        km = _BoomAlive(client=client)
        self.mod._kernel = km
        times = iter([1000.0, 1000.0, 2000.0, 2000.0, 2000.0])
        with mock.patch.object(self.mod, "_ensure_kernel", return_value=client), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=lambda: next(times)), \
             mock.patch.object(self.mod, "_run_subprocess",
                               return_value="[ok] sub-after-boom") as sub:
            out = self.mod._run_jupyter("while True: pass", timeout=5)
        sub.assert_called_once()
        self.assertIn("sub-after-boom", out)

    def test_timeout_interrupt_raises_still_reports_timeout(self):
        # Kernel alive but interrupt_kernel() raises → swallowed, still returns a
        # timeout result (328-329).
        client = _FakeKernelClient(messages=[])

        class _BoomInterrupt(_FakeKernelManager):
            def interrupt_kernel(self):
                raise RuntimeError("interrupt boom")
        km = _BoomInterrupt(client=client, alive=True)
        self.mod._kernel = km
        times = iter([1000.0, 1000.0, 2000.0, 2000.0, 2000.0])
        with mock.patch.object(self.mod, "_ensure_kernel", return_value=client), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=lambda: next(times)):
            out = self.mod._run_jupyter("while True: pass", timeout=5)
        self.assertIn("timeout after", out)

    def test_timeout_with_dead_kernel_falls_back_to_subprocess(self):
        # Deadline passes but the kernel is NOT alive → drop stale handles and
        # fall back to a one-shot subprocess.
        client = _FakeKernelClient(messages=[])
        km = _FakeKernelManager(client=client, alive=False)
        self.mod._kernel = km
        times = iter([1000.0, 1000.0, 2000.0, 2000.0, 2000.0])
        with mock.patch.object(self.mod, "_ensure_kernel", return_value=client), \
             mock.patch.object(self.mod.time, "time",
                               side_effect=lambda: next(times)), \
             mock.patch.object(self.mod, "_run_subprocess",
                               return_value="[ok] fell back") as sub:
            out = self.mod._run_jupyter("while True: pass", timeout=5)
        sub.assert_called_once()
        self.assertIn("fell back", out)
        # Stale handles were cleared.
        self.assertIsNone(self.mod._kernel_client)


class ShutdownKernelTests(_JupyterBase):
    def test_reset_kernel_no_kernel_message(self):
        self.mod._kernel_client = None
        out = self.mod.reset_kernel("")
        self.assertIn("no kernel", out.lower())

    def test_reset_kernel_shuts_down_active_kernel(self):
        client = _FakeKernelClient()
        km = _FakeKernelManager(client=client)
        self.mod._kernel = km
        self.mod._kernel_client = client
        out = self.mod.reset_kernel("")
        self.assertIn("kernel reset", out.lower())
        self.assertTrue(client.channels_stopped)
        self.assertTrue(km.shutdown_called)
        self.assertIsNone(self.mod._kernel)
        self.assertIsNone(self.mod._kernel_client)

    def test_reset_kernel_swallows_shutdown_errors(self):
        class _BadClient(_FakeKernelClient):
            def stop_channels(self):
                raise RuntimeError("stop boom")

        class _BadKM(_FakeKernelManager):
            def shutdown_kernel(self, now=False):
                raise RuntimeError("shutdown boom")
        client = _BadClient()
        self.mod._kernel = _BadKM(client=client)
        self.mod._kernel_client = client
        out = self.mod.reset_kernel("")   # must not raise
        self.assertIn("kernel reset", out.lower())
        self.assertIsNone(self.mod._kernel_client)


# ─── one real, safe end-to-end execution ─────────────────────────────────
class RunPythonEndToEndTests(unittest.TestCase):
    """A single, trivial, SAFE real execution to prove the local-exec path the
    skill is designed around works end to end. No untrusted code."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("code_executor")

    def test_trivial_arithmetic_executes(self):
        with mock.patch.dict(os.environ, {"CODE_EXECUTOR_BACKEND": "subprocess",
                                          "CODE_EXECUTOR_TIMEOUT": "30"}):
            out = self.actions["run_python"]("print(2 + 2)")
        self.assertIn("[ok", out)
        self.assertIn("4", out)


if __name__ == "__main__":
    unittest.main()
