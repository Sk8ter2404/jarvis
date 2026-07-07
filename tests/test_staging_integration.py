"""Unit tests for ``tools/staging_integration.py`` — the heavy LOCAL gate that
boots a staging JARVIS subprocess, injects a battery of read-only utterances,
and asserts NON-FABRICATED replies + a clean shutdown.

SAFETY CONTRACT (this suite NEVER boots JARVIS):
  * ``subprocess.Popen`` is replaced by a ``FakePopen`` that NEVER spawns a
    process; it records terminate / wait / kill so we can assert the shutdown
    choreography, and exposes a no-op stdout.
  * The two low-level helpers (``_wait`` / ``_pump`` / ``_inject`` / ``_finish``)
    are unit-tested DIRECTLY against controlled module state.
  * ``main``'s end-to-end orchestration is driven by a *scripted* ``_wait``
    keyed on the marker substring (boot markers, ``jarvis:`` reply, expected
    markers), so every branch of the battery loop is exercised deterministically
    with NO real threads, NO real polling and NO real subprocess.
  * ``threading.Thread`` is neutered to a no-op (the real pump never runs),
    ``time.sleep`` is patched out, and the module's PROJECT / INJECT / LOG path
    constants are redirected into a per-test tempdir so the real repo runtime
    files are never touched.

CI-faithful: the tool is stdlib-only, so this RUNS on the bare Linux runner.
PII: only synthetic fixtures ("alice", "10.0.0.5"). stdlib ``unittest`` only.
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

import tools.staging_integration as SI  # noqa: E402


# ─────────────────────────── fakes / harness ─────────────────────────────


class _FakeStdout:
    """Iterable stdout stand-in for direct ``_pump`` tests."""

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
    """Records lifecycle calls; never spawns."""

    instances: list = []

    def __init__(self, cmd, *a, **k):
        self.cmd = cmd
        self.args = cmd
        self.kwargs = k
        self.env = k.get("env")
        self.pid = 4321
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
    """``threading.Thread`` replacement whose ``start`` does nothing — the real
    ``_pump`` is never run because ``main`` is driven via a scripted ``_wait``."""

    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def join(self, timeout=None):
        pass


class _ScriptedWait:
    """Callable stand-in for ``SI._wait``.  Looks up the requested marker
    substring in ``mapping`` and returns the configured 1-based line index (or
    ``None``).  ``mapping`` keys are matched case-insensitively as substrings of
    the requested ``substr`` so the boot fallbacks ('standby', '[vad]') and the
    battery markers can all be scripted independently."""

    def __init__(self, mapping):
        # list of (needle_lower, value)
        self._rules = [(k.lower(), v) for k, v in mapping.items()]
        self.calls = []

    def __call__(self, substr, timeout, since=0):
        s = substr.lower()
        self.calls.append((s, since))
        for needle, value in self._rules:
            if needle in s:
                return value
        return None


class _IntegrationBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

        self._orig = {k: getattr(SI, k) for k in ("PROJECT", "INJECT", "LOG")}
        SI.PROJECT = self.tmp
        SI.INJECT = os.path.join(self.tmp, "injected_commands_staging.json")
        SI.LOG = os.path.join(self.tmp, "_staging_integration.log")

        SI._lines.clear()
        FakePopen.instances.clear()
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._orig.items():
            setattr(SI, k, v)
        SI._lines.clear()
        self._tmp.cleanup()

    def _run_main(self, argv, wait_map, *, api_key="x" + "y" * 8, seed_lines=None):
        """Run ``SI.main`` with a scripted ``_wait`` (``wait_map``), a FakePopen,
        neutered threading and no sleep.  ``seed_lines`` pre-populates
        ``SI._lines`` so verbose reply read-back has something to show.
        Returns (exit_code, stdout_text, popen_instance)."""
        if seed_lines:
            SI._lines.extend(seed_lines)
        scripted = _ScriptedWait(wait_map)
        env = {"ANTHROPIC_API_KEY": api_key} if api_key else {}
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["staging_integration.py", *argv]), \
             mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(SI.subprocess, "Popen", FakePopen), \
             mock.patch.object(SI.threading, "Thread", _NoopThread), \
             mock.patch.object(SI.time, "sleep", lambda *_a, **_k: None), \
             mock.patch.object(SI, "_wait", scripted), \
             mock.patch.object(SI, "_inject", lambda *_a, **_k: None), \
             redirect_stdout(buf):
            if not api_key:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            rc = SI.main()
        inst = FakePopen.instances[-1] if FakePopen.instances else None
        return rc, buf.getvalue(), inst, scripted


# ───────────────────────────── _wait ─────────────────────────────────────


class WaitTests(_IntegrationBase):
    def test_finds_substring_case_insensitive(self):
        SI._lines.extend(["boot", "Now Listening for wake word", "idle"])
        self.assertEqual(SI._wait("listening", timeout=1.0), 2)

    def test_returns_none_on_timeout(self):
        SI._lines.extend(["nothing", "here"])
        with mock.patch.object(SI.time, "sleep", lambda *_a, **_k: None):
            self.assertIsNone(SI._wait("absent-marker", timeout=0.05))

    def test_respects_since_offset(self):
        SI._lines.extend(["cpu first", "cpu second"])
        self.assertEqual(SI._wait("cpu", timeout=1.0, since=1), 2)

    def test_since_past_end_then_timeout(self):
        SI._lines.extend(["only"])
        with mock.patch.object(SI.time, "sleep", lambda *_a, **_k: None):
            self.assertIsNone(SI._wait("only", timeout=0.05, since=5))


# ───────────────────────────── _pump ─────────────────────────────────────


class PumpTests(_IntegrationBase):
    def test_pump_collects_lines_and_writes_log(self):
        stream = _FakeStdout(["alpha\n", "beta\n"])
        fh = io.StringIO()
        SI._pump(stream, fh)
        self.assertEqual(SI._lines, ["alpha", "beta"])
        self.assertEqual(fh.getvalue(), "alpha\nbeta\n")

    def test_pump_survives_log_write_error(self):
        stream = _FakeStdout(["x\n"])

        class _BoomFH:
            def write(self, *_a):
                raise OSError("disk full")

            def flush(self):
                raise OSError("disk full")

        SI._pump(stream, _BoomFH())  # must not raise
        self.assertEqual(SI._lines, ["x"])


# ──────────────────────────── _inject ────────────────────────────────────


class InjectTests(_IntegrationBase):
    def test_inject_writes_valid_bomless_json(self):
        SI._inject("what time is it")
        self.assertFalse(os.path.exists(SI.INJECT + ".tmp"))
        with open(SI.INJECT, "rb") as f:
            raw = f.read()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "inject file must be BOM-less")
        self.assertEqual(json.loads(raw.decode("utf-8")), [{"text": "what time is it"}])


# ──────────────────────────── _finish ────────────────────────────────────


class FinishTests(_IntegrationBase):
    def _mk_proc(self, *, wait_timeout=False):
        proc = FakePopen(["python"])
        proc._wait_raises_timeout = wait_timeout
        return proc

    def test_finish_terminates_and_returns_zero_on_ok(self):
        proc = self._mk_proc()
        fh = io.StringIO()
        with open(SI.INJECT, "w") as f:
            f.write("[]")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = SI._finish(proc, fh, [("u", True, "ok")], ok=True)
        self.assertEqual(rc, 0)
        self.assertTrue(proc.terminated)
        self.assertFalse(proc.killed)
        self.assertFalse(os.path.exists(SI.INJECT))
        self.assertIn("VERDICT: PASS", buf.getvalue())
        self.assertIn("1/1 passed", buf.getvalue())

    def test_finish_kills_on_wait_timeout(self):
        proc = self._mk_proc(wait_timeout=True)
        fh = io.StringIO()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = SI._finish(proc, fh, [("u", False, "no reply")], ok=False)
        self.assertEqual(rc, 1)
        self.assertTrue(proc.terminated)
        self.assertTrue(proc.killed)
        self.assertIn("VERDICT: FAIL", buf.getvalue())

    def test_finish_swallows_terminate_error(self):
        class _BadProc:
            def terminate(self):
                raise RuntimeError("already dead")

        fh = io.StringIO()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = SI._finish(_BadProc(), fh, [], ok=True)
        self.assertEqual(rc, 0)
        self.assertIn("0/0 passed", buf.getvalue())

    def test_finish_missing_inject_file_is_ignored(self):
        proc = self._mk_proc()
        fh = io.StringIO()
        self.assertFalse(os.path.exists(SI.INJECT))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = SI._finish(proc, fh, [("u", True, "ok")], ok=True)
        self.assertEqual(rc, 0)

    def test_finish_swallows_fh_close_error(self):
        proc = self._mk_proc()

        class _BadFH:
            def flush(self):
                raise OSError("boom")

            def close(self):
                raise OSError("boom")

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = SI._finish(proc, _BadFH(), [("u", True, "ok")], ok=True)
        self.assertEqual(rc, 0)


# ──────────────────────────── main() flow ────────────────────────────────


class MainNoApiKeyTests(_IntegrationBase):
    def test_missing_api_key_returns_2(self):
        rc, out, inst, _ = self._run_main([], {}, api_key=None)
        self.assertEqual(rc, 2)
        self.assertIn("ANTHROPIC_API_KEY not set", out)
        # must bail out BEFORE constructing a subprocess
        self.assertIsNone(inst)


class MainBootTests(_IntegrationBase):
    # marker -> reply-line index used by both reply readback and battery checks
    def _all_pass_map(self):
        # boot 'listening' found @1; every 'jarvis:' reply found @2; every
        # expected marker (current time is / 2.0.4 / cpu) found.
        return {
            "listening": 1,
            "jarvis:": 2,
            "current time is": 2,
            "2.0.4": 2,
            "cpu": 2,
        }

    def _seed(self):
        # something for reply read-back: index 2 (1-based) -> _lines[1]
        return ["[boot] Now Listening", "JARVIS: the current time is 5pm"]

    def test_happy_path_all_pass(self):
        rc, out, proc, _ = self._run_main([], self._all_pass_map(), seed_lines=self._seed())
        self.assertEqual(rc, 0)
        self.assertIn("VERDICT: PASS", out)
        self.assertIn("3/3 passed", out)
        self.assertIn("booted (marker @", out)
        self.assertIn("[PASS]", out)
        # subprocess launched with the staging entrypoint + flag
        self.assertIn("bobert_companion.py", proc.cmd)
        self.assertIn("--staging", proc.cmd)
        self.assertTrue(proc.terminated)

    def test_staging_env_flags_set(self):
        rc, out, proc, _ = self._run_main([], self._all_pass_map(), seed_lines=self._seed())
        self.assertEqual(proc.env.get("JARVIS_STAGING"), "1")
        self.assertEqual(proc.env.get("MUTE_TTS"), "1")
        self.assertEqual(proc.env.get("JARVIS_TEST_MODE"), "1")
        self.assertEqual(proc.env.get("PYTHONUNBUFFERED"), "1")

    def test_verbose_echoes_reply(self):
        rc, out, _, _ = self._run_main(["-v"], self._all_pass_map(), seed_lines=self._seed())
        self.assertEqual(rc, 0)
        self.assertIn("reply:", out)
        self.assertIn("current time is", out)

    def test_boot_timeout_fails(self):
        # no boot marker at all (and the 'standby'/'[vad]' fallbacks absent too)
        rc, out, proc, _ = self._run_main([], {}, seed_lines=["noise"])
        self.assertEqual(rc, 1)
        self.assertIn("never reached standby/Listening", out)
        self.assertIn("VERDICT: FAIL", out)
        self.assertTrue(proc.terminated)

    def test_boot_via_standby_fallback(self):
        # primary 'listening' missing; 'standby' fallback present
        m = {"standby": 1, "jarvis:": 2, "current time is": 2,
             "2.0.4": 2, "cpu": 2}
        rc, out, _, _ = self._run_main([], m, seed_lines=self._seed())
        self.assertEqual(rc, 0)
        self.assertIn("3/3 passed", out)

    def test_boot_via_vad_fallback(self):
        # both 'listening' and 'standby' missing; '[vad]' present
        m = {"[vad]": 7, "jarvis:": 2, "current time is": 2,
             "2.0.4": 2, "cpu": 2}
        rc, out, _, _ = self._run_main([], m, seed_lines=self._seed())
        self.assertEqual(rc, 0)
        self.assertIn("3/3 passed", out)

    def test_reply_but_missing_marker_fails_item(self):
        # boot ok; a 'jarvis:' reply arrives for every item, but the expected
        # markers never appear -> each item fails with 'missing'
        m = {"listening": 1, "jarvis:": 2}  # markers (current time is/...) -> None
        rc, out, _, _ = self._run_main([], m, seed_lines=self._seed())
        self.assertEqual(rc, 1)
        self.assertIn("missing", out)
        self.assertIn("VERDICT: FAIL", out)
        self.assertIn("0/3 passed", out)

    def test_no_reply_for_item_fails(self):
        # boot ok, but NO 'jarvis:' reply ever -> every item is 'no reply'
        m = {"listening": 1}
        rc, out, _, _ = self._run_main([], m, seed_lines=self._seed())
        self.assertEqual(rc, 1)
        self.assertIn("no reply", out)
        self.assertIn("0/3 passed", out)

    def test_partial_pass_is_overall_fail(self):
        # item 1 fully passes; items 2 & 3 get a reply but miss their marker
        m = {"listening": 1, "jarvis:": 2, "current time is": 2}
        rc, out, _, _ = self._run_main([], m, seed_lines=self._seed())
        self.assertEqual(rc, 1)
        self.assertIn("1/3 passed", out)
        self.assertIn("VERDICT: FAIL", out)

    def test_stale_files_removed_before_boot(self):
        # pre-create the stale inject/log files; main() must os.remove them in
        # its startup cleanup loop (the OSError-guarded block)
        for name in (SI.INJECT, SI.INJECT + ".consuming", SI.LOG):
            with open(name, "w") as f:
                f.write("stale")
        rc, out, _, _ = self._run_main([], self._all_pass_map(), seed_lines=self._seed())
        self.assertEqual(rc, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
