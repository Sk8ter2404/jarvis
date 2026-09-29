"""Unit tests for ``tools/run_tests.py`` — the headless stdlib-unittest runner
used by the self-upgrade pipeline and CI to execute everything under ``tests/``.

What this exercises
-------------------
``main(argv)`` has two arms:

  * **discovery** (no positional args) — ``loader.discover`` over ``_TESTS_DIR``.
    We MUST NOT let that recurse into the real ~40-file suite from inside one of
    those very files, so every discovery test redirects the module's
    ``_PROJECT_ROOT`` / ``_TESTS_DIR`` at a throwaway temp tree containing a
    *uniquely named* package (NOT ``tests`` — that name is already bound to the
    real suite in ``sys.modules``) holding one trivial passing/failing/skipping
    ``test_*.py``.  The tool's own ``sys.path`` bootstrap makes the temp root
    importable so discovery resolves the fixture package.
  * **selector** (positional args) — ``loadTestsFromName('tests.<name>')`` with
    the ``test_`` prefix / ``.py`` suffix normalisation.  The ``tests.`` prefix
    is hard-coded in the tool, so we install fixture modules straight into
    ``sys.modules['tests.test_<x>']`` *and* bind them as attributes on the real
    ``tests`` package (what ``loadTestsFromName`` ultimately ``getattr``s),
    removing both in tearDown.  Nothing real is loaded or run.

Both arms call ``TextTestRunner.run``; we assert the exit code, the
``=== JARVIS TESTS: ... ===`` summary line, the verbosity wiring and the
``sys.path`` bootstrap — without spawning a process or running the live suite.

CI-faithful: ``tools/run_tests.py`` is stdlib-only, so this RUNS (not skips) on
the bare Linux runner.  stdlib ``unittest`` only.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import textwrap
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

# Bootstrap the project root so ``import tools.run_tests`` resolves regardless
# of how the suite is launched.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tools.run_tests as RT  # noqa: E402

# A package name for the discovery fixtures that will NOT collide with the real
# ``tests`` package already present in sys.modules.
_FIX_PKG = "_rt_fixture_pkg"


# ───────────────────────────── fixtures ──────────────────────────────────

_PASS_SRC = textwrap.dedent(
    """
    import unittest
    class _Pass(unittest.TestCase):
        def test_ok(self):
            self.assertTrue(True)
    """
)

_FAIL_SRC = textwrap.dedent(
    """
    import unittest
    class _Fail(unittest.TestCase):
        def test_bad(self):
            self.assertEqual(1, 2)
    """
)

_SKIP_SRC = textwrap.dedent(
    """
    import unittest
    class _Skip(unittest.TestCase):
        @unittest.skip("nope")
        def test_skipped(self):
            pass
    """
)


# ─────────────────────────── discovery arm ───────────────────────────────


class DiscoveryArmTests(unittest.TestCase):
    """Redirects the module path globals at a per-test temp tree whose fixture
    package has a unique name, so discovery can never see the real suite."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.pkg_dir = os.path.join(self.root, _FIX_PKG)
        os.makedirs(self.pkg_dir)
        open(os.path.join(self.pkg_dir, "__init__.py"), "w").close()

        self._orig_root = RT._PROJECT_ROOT
        self._orig_tests = RT._TESTS_DIR
        RT._PROJECT_ROOT = self.root
        RT._TESTS_DIR = self.pkg_dir

        self._orig_path = list(sys.path)
        self._orig_modules = set(sys.modules)
        self.addCleanup(self._restore)

    def _restore(self):
        RT._PROJECT_ROOT = self._orig_root
        RT._TESTS_DIR = self._orig_tests
        sys.path[:] = self._orig_path
        for name in set(sys.modules) - self._orig_modules:
            sys.modules.pop(name, None)
        self._tmp.cleanup()

    def _write(self, filename, src):
        with open(os.path.join(self.pkg_dir, filename), "w", encoding="utf-8") as f:
            f.write(src)

    def _run(self, argv):
        buf = io.StringIO()
        # TextTestRunner writes its own report to stderr; swallow it so the
        # nested fixture-suite chatter doesn't pollute THIS suite's output.
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            rc = RT.main(argv)
        return rc, buf.getvalue()

    def test_discovers_and_passes(self):
        self._write("test_alpha.py", _PASS_SRC)
        rc, out = self._run([])
        self.assertEqual(rc, 0)
        self.assertIn("=== JARVIS TESTS:", out)
        self.assertIn("1 run", out)
        self.assertIn("0 failed", out)
        self.assertIn("0 errored", out)

    def test_discovery_failure_yields_nonzero(self):
        self._write("test_bad.py", _FAIL_SRC)
        rc, out = self._run([])
        self.assertEqual(rc, 1)
        self.assertIn("1 failed", out)

    def test_discovery_counts_multiple_files(self):
        self._write("test_a.py", _PASS_SRC)
        self._write("test_b.py", _PASS_SRC)
        rc, out = self._run([])
        self.assertEqual(rc, 0)
        self.assertIn("2 run", out)

    def test_discovery_reports_skips(self):
        self._write("test_s.py", _SKIP_SRC)
        rc, out = self._run([])
        self.assertEqual(rc, 0)
        self.assertIn("1 skipped", out)

    def test_non_matching_pattern_files_ignored(self):
        self._write("helper_not_a_test.py", _FAIL_SRC)
        self._write("test_real.py", _PASS_SRC)
        rc, out = self._run([])
        self.assertEqual(rc, 0)
        self.assertIn("1 run", out)

    def test_empty_suite_is_success(self):
        rc, out = self._run([])
        self.assertEqual(rc, 0)
        self.assertIn("0 run", out)

    def test_summary_combines_fail_and_skip(self):
        self._write("test_pass.py", _PASS_SRC)
        self._write("test_fail.py", _FAIL_SRC)
        self._write("test_skip.py", _SKIP_SRC)
        rc, out = self._run([])
        self.assertEqual(rc, 1)
        self.assertIn("3 run", out)
        self.assertIn("1 failed", out)
        self.assertIn("1 skipped", out)


# ──────────────────────── verbosity + path wiring ─────────────────────────


class WiringTests(DiscoveryArmTests):
    """Reuses the discovery harness (temp fixture pkg) to assert the runner's
    verbosity wiring and the ``sys.path`` bootstrap branch."""

    def _runner_kwargs(self, argv):
        """The kwargs main() builds its TextTestRunner with."""
        self._write("test_v.py", _PASS_SRC)
        with mock.patch.object(RT.unittest, "TextTestRunner",
                               wraps=RT.unittest.TextTestRunner) as runner:
            self._run(argv)
        runner.assert_called_once()
        return runner.call_args.kwargs

    def test_verbose_flag_sets_verbosity_2(self):
        self.assertEqual(self._runner_kwargs(["-v"])["verbosity"], 2)

    def test_long_verbose_flag(self):
        self.assertEqual(self._runner_kwargs(["--verbose"])["verbosity"], 2)

    def test_default_verbosity_1(self):
        self.assertEqual(self._runner_kwargs([])["verbosity"], 1)

    def test_the_runner_reports_through_the_time_watchdog(self):
        """tools/test_watchdog.py bounds how long the run may STALL, but only
        the result class makes the report name a TEST — without it a hang is
        still just 'somewhere in the suite' (the 2026-09-04 exit-124)."""
        from tools.test_watchdog import WatchdogTextTestResult
        self.assertIs(self._runner_kwargs([])["resultclass"],
                      WatchdogTextTestResult)

    def test_the_runner_is_handed_its_own_stream(self):
        """Runtime, not just source: main() passes stream= (taken before
        discovery — see RunnerStreamTests) instead of letting TextTestRunner
        read sys.stderr once every test module has been imported."""
        self.assertIn("stream", self._runner_kwargs([]))

    def test_inserts_project_root_when_absent(self):
        self._write("test_p.py", _PASS_SRC)
        sys.path[:] = [p for p in sys.path if p != self.root]
        self.assertNotIn(self.root, sys.path)
        self._run([])
        self.assertIn(self.root, sys.path)
        self.assertEqual(sys.path[0], self.root)

    def test_does_not_double_insert_when_present(self):
        self._write("test_p.py", _PASS_SRC)
        sys.path.insert(0, self.root)
        before = sys.path.count(self.root)
        self._run([])
        self.assertEqual(sys.path.count(self.root), before)


# ─────────────────────────── selector arm ────────────────────────────────


class SelectorArmTests(unittest.TestCase):
    """The selector arm hard-codes the ``tests.`` package prefix.  We install
    fixture modules into ``sys.modules['tests.test_<x>']`` and bind them as
    attributes on the real ``tests`` package (the object ``loadTestsFromName``
    ultimately ``getattr``s), cleaning both up afterward."""

    def setUp(self):
        import tests as _tests_pkg  # the real package this file lives in
        self._pkg = _tests_pkg
        self._installed = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for dotted, attr in self._installed:
            sys.modules.pop(dotted, None)
            if hasattr(self._pkg, attr):
                delattr(self._pkg, attr)

    def _install(self, attr, src):
        dotted = f"tests.{attr}"
        mod = types.ModuleType(dotted)
        exec(compile(src, dotted, "exec"), mod.__dict__)
        sys.modules[dotted] = mod
        setattr(self._pkg, attr, mod)
        self._installed.append((dotted, attr))

    def _run(self, argv):
        buf = io.StringIO()
        # TextTestRunner writes its own report to stderr; swallow it so the
        # nested fixture-suite chatter doesn't pollute THIS suite's output.
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            rc = RT.main(argv)
        return rc, buf.getvalue()

    def test_selector_bare_name_gets_prefix(self):
        self._install("test_widget", _PASS_SRC)
        rc, out = self._run(["widget"])
        self.assertEqual(rc, 0)
        self.assertIn("1 run", out)

    def test_selector_full_name_passthrough(self):
        self._install("test_gadget", _PASS_SRC)
        rc, out = self._run(["test_gadget"])
        self.assertEqual(rc, 0)
        self.assertIn("1 run", out)

    def test_selector_strips_py_suffix(self):
        self._install("test_thing", _PASS_SRC)
        rc, out = self._run(["test_thing.py"])
        self.assertEqual(rc, 0)
        self.assertIn("1 run", out)

    def test_selector_bare_name_with_py_suffix(self):
        # exercises BOTH the prefix-add and the .py-strip on one selector
        self._install("test_combo", _PASS_SRC)
        rc, out = self._run(["combo.py"])
        self.assertEqual(rc, 0)
        self.assertIn("1 run", out)

    def test_multiple_selectors_aggregate(self):
        self._install("test_one", _PASS_SRC)
        self._install("test_two", _PASS_SRC)
        rc, out = self._run(["one", "two"])
        self.assertEqual(rc, 0)
        self.assertIn("2 run", out)

    def test_selector_failure_nonzero(self):
        self._install("test_boom", _FAIL_SRC)
        rc, out = self._run(["boom"])
        self.assertEqual(rc, 1)
        self.assertIn("1 failed", out)

    def test_dash_flags_are_not_selectors(self):
        # '-v' is consumed as a flag, leaving 'solo' as the only selector
        self._install("test_solo", _PASS_SRC)
        rc, out = self._run(["-v", "solo"])
        self.assertEqual(rc, 0)
        self.assertIn("1 run", out)


# ─────────────────────── the runner's own stream ─────────────────────────


class RunnerStreamTests(unittest.TestCase):
    """``_runner_stream()`` — the private stream all three runners report
    through (the v2.0.128 CI crash: a leaked poller's import-time
    ``sys.stderr.reconfigure()`` left the SHARED stderr without an encoder for
    an instant, and the runner's "." died with ``io.UnsupportedOperation: not
    writable`` in TextTestResult.addSuccess).

    Each test stands a temp file in for the process's real stderr — BOTH
    ``sys.stderr`` and ``sys.__stderr__`` — so what the private dup writes can
    be read back without touching this run's own fd 2."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(prefix="rt_stream_", suffix=".err")
        os.close(fd)
        self.err = open(self.path, "w", encoding="utf-8",
                        errors="backslashreplace")
        self.addCleanup(self._cleanup)
        for attr in ("stderr", "__stderr__"):
            p = mock.patch.object(sys, attr, self.err)
            p.start()
            self.addCleanup(p.stop)

    def _cleanup(self):
        self.err.close()
        os.unlink(self.path)

    def _written(self) -> str:
        with open(self.path, encoding="utf-8") as f:
            return f.read()

    def _private(self):
        stream, owned = RT._runner_stream()
        self.assertTrue(owned, "the real stderr should get a private dup")
        self.addCleanup(stream.close)
        return stream

    def test_a_private_stream_on_the_same_destination(self):
        stream = self._private()
        self.assertIsNot(stream, self.err)
        self.assertNotEqual(stream.fileno(), self.err.fileno())
        stream.write("dots.")
        stream.flush()
        self.assertIn("dots.", self._written())

    def test_it_mirrors_stderrs_encoding_and_error_handler(self):
        stream = self._private()
        self.assertEqual(stream.encoding, self.err.encoding)
        self.assertEqual(stream.errors, self.err.errors)
        self.assertTrue(stream.line_buffering)

    def test_nothing_done_to_sys_stderr_afterwards_reaches_it(self):
        """THE v2.0.128 SHAPE, made deterministic: the object the runner would
        otherwise share becomes a stream whose write() raises exactly CI's
        exception — and is then CLOSED outright. The runner's stream writes on."""
        stream = self._private()
        read_only = open(self.path, encoding="utf-8")
        self.addCleanup(read_only.close)
        with self.assertRaisesRegex(io.UnsupportedOperation, "not writable"):
            read_only.write(".")          # the CI traceback's exact exception
        sys.stderr = read_only             # restored by the patch's cleanup
        self.err.close()
        stream.write("still reporting")
        stream.flush()
        self.assertIn("still reporting", self._written())

    def test_a_deliberate_redirect_is_honoured(self):
        """redirect_stderr around main() (this file's own harness) must still
        capture the report, so a redirected stderr is used as-is, unowned."""
        buf = io.StringIO()
        with mock.patch.object(sys, "stderr", buf):
            self.assertEqual(RT._runner_stream(), (buf, False))

    def test_no_descriptor_falls_back_to_stderr_itself(self):
        buf = io.StringIO()
        with mock.patch.object(sys, "stderr", buf), \
                mock.patch.object(sys, "__stderr__", buf):
            self.assertEqual(RT._runner_stream(), (buf, False))

    def test_no_stderr_at_all(self):
        with mock.patch.object(sys, "stderr", None), \
                mock.patch.object(sys, "__stderr__", None):
            self.assertEqual(RT._runner_stream(), (None, False))


class EveryRunnerOwnsItsStreamTests(unittest.TestCase):
    """All three discovery runners take the private stream BEFORE discovery
    and hand it to TextTestRunner. Left to its default, TextTestRunner reads
    ``sys.stderr`` only after discovery has imported every test module."""

    RUNNERS = ("run_tests.py", "run_coverage.py", "run_tests_ci_sim.py")

    def test_every_runner_takes_its_stream_before_discovery(self):
        for name in self.RUNNERS:
            with self.subTest(runner=name):
                with open(os.path.join(_ROOT, "tools", name),
                          encoding="utf-8") as f:
                    src = f.read()
                self.assertIn("= _runner_stream()", src)
                self.assertIn("stream=runner_stream", src)
                self.assertLess(src.index("= _runner_stream()"),
                                src.index(".discover("),
                                f"{name} takes its stream after discovery")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
