"""Unit tests for ``tools/staging_inject_smoke.py`` — the lighter inject smoke
harness: boot a staging JARVIS subprocess, inject ONE utterance, assert it is
answered (and optionally that an expected substring appears in the reply).

SAFETY CONTRACT (this suite NEVER boots JARVIS) — identical strategy to the
staging_integration suite:
  * ``subprocess.Popen`` -> ``FakePopen`` (records terminate/wait/kill, no spawn).
  * ``_wait`` / ``_pump`` / ``_inject`` unit-tested directly on controlled state.
  * ``main``'s orchestration driven by a *scripted* ``_wait`` keyed on the marker
    substring (boot 'Listening'/'skills loaded'; echo 'You:'/'[inject]'; reply
    'JARVIS:'; the optional --expect substring), so every branch runs
    deterministically with no threads, no polling, no real subprocess.
  * ``threading.Thread`` neutered, ``time.sleep`` patched out, PROJECT/INJECT/LOG
    redirected into a per-test tempdir.

CI-faithful: tool is stdlib-only -> RUNS on the bare Linux runner.
PII: only synthetic fixtures. stdlib ``unittest`` only.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tools.staging_inject_smoke as SS  # noqa: E402


# ─────────────────────────── fakes / harness ─────────────────────────────


class _FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)
        self._i = 0

    def readline(self):
        if self._i < len(self._lines):
            line = self._lines[self._i]
            self._i += 1
            return line
        return ""


class FakePopen:
    instances: list = []

    def __init__(self, cmd, *a, **k):
        self.cmd = cmd
        self.args = cmd
        self.kwargs = k
        self.env = k.get("env")
        self.pid = 9988
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.wait_calls = 0
        self._wait_raises_timeout = False
        self.stdout = _FakeStdout([])
        type(self).instances.append(self)

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self._wait_raises_timeout:
            raise subprocess.TimeoutExpired(self.cmd, timeout or 0)
        self.returncode = 0
        return 0


class _NoopThread:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def join(self, timeout=None):
        pass


class _ScriptedWait:
    """Returns the configured value for the first mapping key that is a
    case-insensitive substring of the requested marker; else None."""

    def __init__(self, mapping):
        self._rules = [(k.lower(), v) for k, v in mapping.items()]
        self.calls = []

    def __call__(self, substr, timeout, since=0):
        s = substr.lower()
        self.calls.append((s, since))
        for needle, value in self._rules:
            if needle in s:
                return value
        return None


class _SmokeBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

        self._orig = {k: getattr(SS, k) for k in ("PROJECT", "INJECT", "LOG")}
        SS.PROJECT = self.tmp
        SS.INJECT = os.path.join(self.tmp, "injected_commands_staging.json")
        SS.LOG = os.path.join(self.tmp, "_staging_smoke.log")

        SS._lines.clear()
        FakePopen.instances.clear()
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._orig.items():
            setattr(SS, k, v)
        SS._lines.clear()
        self._tmp.cleanup()

    def _run_main(self, argv, wait_map, *, seed_lines=None):
        if seed_lines:
            SS._lines.extend(seed_lines)
        scripted = _ScriptedWait(wait_map)
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["staging_inject_smoke.py", *argv]), \
             mock.patch.object(SS.subprocess, "Popen", FakePopen), \
             mock.patch.object(SS.threading, "Thread", _NoopThread), \
             mock.patch.object(SS.time, "sleep", lambda *_a, **_k: None), \
             mock.patch.object(SS, "_wait", scripted), \
             mock.patch.object(SS, "_inject", lambda *_a, **_k: None), \
             redirect_stdout(buf):
            rc = SS.main()
        inst = FakePopen.instances[-1] if FakePopen.instances else None
        return rc, buf.getvalue(), inst, scripted


# ───────────────────────────── _wait ─────────────────────────────────────


class WaitTests(_SmokeBase):
    def test_finds_substring(self):
        SS._lines.extend(["x", "JARVIS: hello there", "y"])
        self.assertEqual(SS._wait("jarvis:", timeout=1.0), 2)

    def test_timeout_returns_none(self):
        SS._lines.extend(["a"])
        with mock.patch.object(SS.time, "sleep", lambda *_a, **_k: None):
            self.assertIsNone(SS._wait("nope", timeout=0.05))

    def test_since_offset(self):
        SS._lines.extend(["JARVIS: one", "JARVIS: two"])
        self.assertEqual(SS._wait("jarvis:", timeout=1.0, since=1), 2)


# ───────────────────────────── _pump ─────────────────────────────────────


class PumpTests(_SmokeBase):
    def test_pump_collects_and_writes(self):
        stream = _FakeStdout(["one\n", "two\n"])
        fh = io.StringIO()
        SS._pump(stream, fh)
        self.assertEqual(SS._lines, ["one", "two"])
        self.assertEqual(fh.getvalue(), "one\ntwo\n")

    def test_pump_survives_write_error(self):
        stream = _FakeStdout(["z\n"])

        class _BoomFH:
            def write(self, *_a):
                raise OSError("nope")

            def flush(self):
                raise OSError("nope")

        SS._pump(stream, _BoomFH())
        self.assertEqual(SS._lines, ["z"])


# ──────────────────────────── _inject ────────────────────────────────────


class InjectTests(_SmokeBase):
    def test_inject_writes_json_and_renames(self):
        SS._inject("hello alice")
        self.assertFalse(os.path.exists(SS.INJECT + ".tmp"))
        with open(SS.INJECT, encoding="utf-8") as f:
            self.assertEqual(json.load(f), [{"text": "hello alice"}])


# ──────────────────────────── main() flow ────────────────────────────────


class MainTests(_SmokeBase):
    def _seed(self):
        return ["[boot] Listening now", "JARVIS: it is 5 pm"]

    def test_happy_path_no_expect(self):
        m = {"listening": 1, "you:": 2, "jarvis:": 2}
        rc, out, proc, _ = self._run_main(["what time is it"], m, seed_lines=self._seed())
        self.assertEqual(rc, 0)
        self.assertIn("VERDICT: PASS", out)
        self.assertIn("bobert_companion.py", proc.cmd)
        self.assertIn("--staging", proc.cmd)
        self.assertTrue(proc.terminated)

    def test_staging_env_flags(self):
        m = {"listening": 1, "you:": 2, "jarvis:": 2}
        rc, out, proc, _ = self._run_main(["hi"], m, seed_lines=self._seed())
        self.assertEqual(proc.env.get("JARVIS_STAGING"), "1")
        self.assertEqual(proc.env.get("MUTE_TTS"), "1")
        self.assertEqual(proc.env.get("JARVIS_TEST_MODE"), "1")
        self.assertEqual(proc.env.get("PYTHONUNBUFFERED"), "1")

    def test_boot_via_skills_loaded_fallback(self):
        # primary 'Listening' missing; 'skills loaded' fallback present
        m = {"skills loaded": 3, "you:": 4, "jarvis:": 4}
        rc, out, _, _ = self._run_main(["hi"], m, seed_lines=self._seed())
        self.assertEqual(rc, 0)
        self.assertIn("VERDICT: PASS", out)

    def test_boot_timeout_fails(self):
        # neither 'Listening' nor 'skills loaded' -> boot is None -> FAIL,
        # and main() must NOT inject or wait for a reply
        m = {}
        rc, out, proc, scripted = self._run_main(["hi"], m, seed_lines=["noise"])
        self.assertEqual(rc, 1)
        self.assertIn("VERDICT: FAIL", out)
        self.assertTrue(proc.terminated)
        # no reply wait should have happened (only the two boot probes)
        self.assertNotIn("jarvis:", [c[0] for c in scripted.calls])

    def test_reply_missing_fails(self):
        # boot ok, echo seen, but no 'JARVIS:' reply -> FAIL
        m = {"listening": 1, "you:": 2}
        rc, out, _, _ = self._run_main(["hi"], m, seed_lines=self._seed())
        self.assertEqual(rc, 1)
        self.assertIn("VERDICT: FAIL", out)

    def test_echo_via_inject_fallback(self):
        # 'You:' missing but '[inject]' fallback present; reply present -> PASS
        m = {"listening": 1, "[inject]": 2, "jarvis:": 2}
        rc, out, _, _ = self._run_main(["hi"], m, seed_lines=self._seed())
        self.assertEqual(rc, 0)
        self.assertIn("VERDICT: PASS", out)

    def test_expect_hit_passes(self):
        m = {"listening": 1, "you:": 2, "jarvis:": 2, "it is": 2}
        rc, out, _, _ = self._run_main(["what time is it", "--expect", "it is"],
                                       m, seed_lines=self._seed())
        self.assertEqual(rc, 0)
        self.assertIn("VERDICT: PASS", out)
        self.assertIn("expected 'it is'", out)

    def test_expect_miss_fails(self):
        # reply present, but the expected substring never appears -> FAIL
        m = {"listening": 1, "you:": 2, "jarvis:": 2}  # 'banana' -> None
        rc, out, _, _ = self._run_main(["what time is it", "--expect", "banana"],
                                       m, seed_lines=self._seed())
        self.assertEqual(rc, 1)
        self.assertIn("VERDICT: FAIL", out)

    def test_expect_skipped_when_reply_missing(self):
        # reply missing AND --expect given: the expect check is guarded behind
        # `if ok and args.expect`, so it must be skipped (ok already False)
        m = {"listening": 1, "you:": 2}
        rc, out, _, scripted = self._run_main(["hi", "--expect", "zzz"],
                                              m, seed_lines=self._seed())
        self.assertEqual(rc, 1)
        self.assertNotIn("zzz", [c[0] for c in scripted.calls])

    def test_stale_files_removed_before_boot(self):
        for name in (SS.INJECT, SS.INJECT + ".consuming", SS.LOG):
            with open(name, "w") as f:
                f.write("stale")
        m = {"listening": 1, "you:": 2, "jarvis:": 2}
        rc, out, _, _ = self._run_main(["hi"], m, seed_lines=self._seed())
        self.assertEqual(rc, 0)

    def test_terminate_exception_swallowed(self):
        # proc.terminate() raising must be caught by the broad finally guard
        m = {"listening": 1, "you:": 2, "jarvis:": 2}

        def factory(cmd, *a, **k):
            p = FakePopen(cmd, *a, **k)

            def _boom():
                raise RuntimeError("terminate exploded")

            p.terminate = _boom
            return p

        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["staging_inject_smoke.py", "hi"]), \
             mock.patch.object(SS.subprocess, "Popen", side_effect=factory), \
             mock.patch.object(SS.threading, "Thread", _NoopThread), \
             mock.patch.object(SS.time, "sleep", lambda *_a, **_k: None), \
             mock.patch.object(SS, "_wait", _ScriptedWait(m)), \
             mock.patch.object(SS, "_inject", lambda *_a, **_k: None), \
             redirect_stdout(buf):
            SS._lines.extend(self._seed())
            rc = SS.main()  # must NOT raise
        self.assertEqual(rc, 0)
        self.assertIn("VERDICT: PASS", buf.getvalue())

    def test_log_handle_close_error_swallowed(self):
        # the log file handle raising on flush/close must be swallowed by the
        # second finally guard (line 108-109 in the tool)
        m = {"listening": 1, "you:": 2, "jarvis:": 2}
        real_open = open

        class _BadFH:
            def write(self, *_a):
                return 0

            def flush(self):
                raise OSError("flush boom")

            def close(self):
                raise OSError("close boom")

        def fake_open(path, *a, **k):
            if str(path) == SS.LOG:
                return _BadFH()
            return real_open(path, *a, **k)

        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["staging_inject_smoke.py", "hi"]), \
             mock.patch("builtins.open", side_effect=fake_open), \
             mock.patch.object(SS.subprocess, "Popen", FakePopen), \
             mock.patch.object(SS.threading, "Thread", _NoopThread), \
             mock.patch.object(SS.time, "sleep", lambda *_a, **_k: None), \
             mock.patch.object(SS, "_wait", _ScriptedWait(m)), \
             mock.patch.object(SS, "_inject", lambda *_a, **_k: None), \
             redirect_stdout(buf):
            SS._lines.extend(self._seed())
            rc = SS.main()  # must NOT raise despite the bad log handle
        self.assertEqual(rc, 0)

    def test_kill_on_wait_timeout(self):
        # force the finally-block wait() to time out so kill() runs
        m = {"listening": 1, "you:": 2, "jarvis:": 2}

        def factory(cmd, *a, **k):
            p = FakePopen(cmd, *a, **k)
            p._wait_raises_timeout = True
            return p

        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["staging_inject_smoke.py", "hi"]), \
             mock.patch.object(SS.subprocess, "Popen", side_effect=factory), \
             mock.patch.object(SS.threading, "Thread", _NoopThread), \
             mock.patch.object(SS.time, "sleep", lambda *_a, **_k: None), \
             mock.patch.object(SS, "_wait", _ScriptedWait(m)), \
             mock.patch.object(SS, "_inject", lambda *_a, **_k: None), \
             redirect_stdout(buf):
            SS._lines.extend(self._seed())
            rc = SS.main()
        self.assertEqual(rc, 0)
        proc = FakePopen.instances[-1]
        self.assertTrue(proc.terminated)
        self.assertTrue(proc.killed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
