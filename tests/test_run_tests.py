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

    def test_verbose_flag_sets_verbosity_2(self):
        self._write("test_v.py", _PASS_SRC)
        with mock.patch.object(RT.unittest, "TextTestRunner",
                               wraps=RT.unittest.TextTestRunner) as runner:
            self._run(["-v"])
        runner.assert_called_once_with(verbosity=2)

    def test_long_verbose_flag(self):
        self._write("test_v.py", _PASS_SRC)
        with mock.patch.object(RT.unittest, "TextTestRunner",
                               wraps=RT.unittest.TextTestRunner) as runner:
            self._run(["--verbose"])
        runner.assert_called_once_with(verbosity=2)

    def test_default_verbosity_1(self):
        self._write("test_v.py", _PASS_SRC)
        with mock.patch.object(RT.unittest, "TextTestRunner",
                               wraps=RT.unittest.TextTestRunner) as runner:
            self._run([])
        runner.assert_called_once_with(verbosity=1)

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
