"""tools/mem_guard — the hard memory ceiling every test run must sit inside.

THE INCIDENT (2026-08-20, 05:04)
================================
An uncapped test run committed ~144 GB of virtual memory on a 48 GB box and
bugchecked the machine (0x0000000A). The fix is structural: EVERY runner applies
a process-wide ceiling BEFORE importing a single test, so a leak dies at the
ceiling instead of taking the desktop with it.

These tests pin that contract WITHOUT allocating anything — the ctypes /
resource work is behind one seam (``mem_guard._apply_ceiling``) which is
mocked here. They are pure-stdlib and platform-agnostic, so they pass on
Windows, on the ubuntu-latest CI runner, and under tools/run_tests_ci_sim.py
(which rewrites sys.platform to "linux" and deletes ctypes.windll).

The last tests are the STALE-DUPLICATE guard, this codebase's signature bug
class: a rule fixed in one copy while the others rot. They read the runners'
source and fail if a runner stops calling the guard, calls it too late, if a
NEW runner appears that never calls it at all, or if an existing runner quietly
drops off the checked list.
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import mem_guard  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR = os.path.join(_PROJECT_ROOT, "tools")

# EVERY module in tools/ that discovers the unittest suite: each MUST apply the
# ceiling first thing in main(). run_coverage.py joined this list on 2026-08-20
# when it was wired up — it is the runner CI itself invokes
# (``python tools/run_coverage.py --xml --fail-under 80``), and until then it was
# the last path that could run the whole suite unbounded.
_GUARDED_RUNNERS = ("run_tests.py", "run_tests_ci_sim.py", "run_coverage.py")

# Modules that discover the unittest suite but do NOT apply the ceiling, with
# the reason. DELIBERATELY EMPTY: every discovery runner in tools/ is now
# guarded, and test_known_unguarded_list_stays_honest() fails the moment an
# entry here starts calling apply_memory_ceiling(), so an exemption cannot rot
# into a lie. This hatch exists so a FUTURE runner cannot be added silently —
# not to bless a gap; anything landing here needs a written reason.
_KNOWN_UNGUARDED: set[str] = set()

# Calls that (directly or transitively) import test modules or spawn a child.
# The guard must come before every one of them. ``_run_suite`` is
# run_coverage.py's discovery helper — the loader lives one frame down, so the
# call in main() is where tests first get touched there.
_TEST_TOUCHING_CALLS = {"discover", "loadTestsFromName", "loadTestsFromModule",
                        "_run_ci_gates", "_run_suite", "run", "Popen"}

# The scan that finds a discovery runner by source inspection (kept in one place
# so the "no new runner" and "the list is complete" tests cannot disagree).
_DISCOVERY_MARKERS = ("unittest.TestLoader", "unittest.main(")


def _source(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _main_func(tree: ast.AST) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("no module-level main() found")


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


class CapResolutionTests(unittest.TestCase):
    """JARVIS_TEST_MEM_CAP_GB parsing — the knob must fail CLOSED (still
    capped) for every bad input, and only turn off when told to explicitly."""

    def test_default_when_unset(self):
        self.assertEqual(mem_guard.resolve_cap_gb({}), mem_guard.DEFAULT_CAP_GB)

    def test_default_is_generous_but_far_below_the_box(self):
        # Rationale pinned: above the suite's real 1-3 GB peak, nowhere near
        # the 48 GB of RAM the 144 GB run drowned.
        self.assertGreaterEqual(mem_guard.DEFAULT_CAP_GB, 4.0)
        self.assertLessEqual(mem_guard.DEFAULT_CAP_GB, 16.0)

    def test_env_override(self):
        for raw, want in (("2", 2.0), ("0.5", 0.5), ("12.5", 12.5),
                          ("  3  ", 3.0)):
            with self.subTest(raw=raw):
                self.assertEqual(
                    mem_guard.resolve_cap_gb({mem_guard.CAP_ENV_VAR: raw}), want)

    def test_off_switch(self):
        for raw in ("0", "off", "OFF", "none", "None", "unlimited", "disabled",
                    "no", "false"):
            with self.subTest(raw=raw):
                self.assertEqual(
                    mem_guard.resolve_cap_gb({mem_guard.CAP_ENV_VAR: raw}), 0.0)

    def test_blank_is_treated_as_unset_not_as_off(self):
        # A wrapper script that exports the var empty must NOT silently disable
        # the ceiling.
        for raw in ("", "   ", "\t"):
            with self.subTest(raw=repr(raw)):
                self.assertEqual(
                    mem_guard.resolve_cap_gb({mem_guard.CAP_ENV_VAR: raw}),
                    mem_guard.DEFAULT_CAP_GB)

    def test_garbage_and_negative_fall_back_to_default(self):
        # "nan"/"inf" parse as floats but would blow up the byte conversion
        # (int(nan) raises) — the guard must never raise, so they are garbage.
        for raw in ("banana", "8GB", "-1", "-0.5", "1e", "nan?", "nan", "inf",
                    "-inf", "NaN", "Infinity"):
            with self.subTest(raw=raw):
                self.assertEqual(
                    mem_guard.resolve_cap_gb({mem_guard.CAP_ENV_VAR: raw}),
                    mem_guard.DEFAULT_CAP_GB)

    def test_reads_os_environ_when_no_env_passed(self):
        with mock.patch.dict(os.environ, {mem_guard.CAP_ENV_VAR: "6"}):
            self.assertEqual(mem_guard.resolve_cap_gb(), 6.0)


class ApplyTests(unittest.TestCase):
    """apply_memory_ceiling() — one line of output, never raises, idempotent."""

    def setUp(self):
        mem_guard._reset_for_tests()
        self.addCleanup(mem_guard._reset_for_tests)

    @staticmethod
    def _apply(**kw):
        """Call the guard, returning (result, printed-stdout)."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = mem_guard.apply_memory_ceiling(**kw)
        return res, buf.getvalue()

    def test_happy_path_applies_and_announces_once(self):
        seam = mock.Mock(return_value="Job Object")
        with mock.patch.object(mem_guard, "_apply_ceiling", seam):
            res, out = self._apply(env={mem_guard.CAP_ENV_VAR: "8"})
        self.assertTrue(res)
        self.assertEqual(out.strip().splitlines(),
                         ["[mem-guard] ceiling 8 GB (Job Object)"])
        seam.assert_called_once_with(8 * (1 << 30))

    def test_fractional_cap_formats_and_converts(self):
        seam = mock.Mock(return_value="RLIMIT_AS")
        with mock.patch.object(mem_guard, "_apply_ceiling", seam):
            res, out = self._apply(env={mem_guard.CAP_ENV_VAR: "0.5"})
        self.assertTrue(res)
        self.assertIn("ceiling 0.5 GB (RLIMIT_AS)", out)
        seam.assert_called_once_with(1 << 29)

    def test_explicit_cap_argument_beats_env(self):
        seam = mock.Mock(return_value="Job Object")
        with mock.patch.object(mem_guard, "_apply_ceiling", seam):
            res, _ = self._apply(cap_gb=2, env={mem_guard.CAP_ENV_VAR: "8"})
        self.assertTrue(res)
        seam.assert_called_once_with(2 * (1 << 30))

    def test_absurdly_large_cap_is_clamped_so_it_cannot_wrap(self):
        """MEASURED: ctypes stores an over-64-bit int into a c_size_t field
        MODULO 2**64 and raises nothing — int(1e12 * 2**30) came back as
        3830667724846006272. An unclamped huge cap could therefore wrap to a
        tiny one and kill the very run it was protecting."""
        seam = mock.Mock(return_value="Job Object")
        with mock.patch.object(mem_guard, "_apply_ceiling", seam):
            res, out = self._apply(env={mem_guard.CAP_ENV_VAR: "1e12"})
        self.assertTrue(res)
        sent = seam.call_args[0][0]
        self.assertEqual(sent, int(mem_guard._MAX_CAP_GB * (1 << 30)))
        self.assertLess(sent, 1 << 64, "cap would wrap a 64-bit size_t")
        self.assertIn(f"ceiling {mem_guard._fmt_gb(mem_guard._MAX_CAP_GB)} GB",
                      out)  # the banner reports the EFFECTIVE ceiling

    def test_absurdly_small_cap_is_floored(self):
        # A fat-fingered 1 MB ceiling would kill the interpreter mid-import —
        # that is "breaking a run it cannot protect", so floor it.
        seam = mock.Mock(return_value="Job Object")
        with mock.patch.object(mem_guard, "_apply_ceiling", seam):
            res, _ = self._apply(env={mem_guard.CAP_ENV_VAR: "0.001"})
        self.assertTrue(res)
        self.assertEqual(seam.call_args[0][0],
                         int(mem_guard._MIN_CAP_GB * (1 << 30)))

    def test_os_refusal_warns_and_returns_falsy_without_raising(self):
        """The whole point of the guard's contract: it must never break a run
        it cannot protect (no privilege / hostile outer job / odd platform)."""
        boom = mock.Mock(side_effect=OSError(5, "Access is denied"))
        with mock.patch.object(mem_guard, "_apply_ceiling", boom):
            res, out = self._apply(env={mem_guard.CAP_ENV_VAR: "8"})
        self.assertFalse(res)
        self.assertEqual(len(out.strip().splitlines()), 1)
        self.assertIn("WARNING", out)
        self.assertIn("UNCAPPED", out)
        self.assertIn("Access is denied", out)

    def test_any_exception_type_is_swallowed(self):
        for exc in (RuntimeError("nested job"), ValueError("nope"),
                    AttributeError("no windll"), MemoryError()):
            with self.subTest(exc=type(exc).__name__):
                mem_guard._reset_for_tests()
                with mock.patch.object(mem_guard, "_apply_ceiling",
                                       mock.Mock(side_effect=exc)):
                    res, out = self._apply(env={})
                self.assertFalse(res)
                self.assertIn("WARNING", out)

    def test_nonsense_cap_argument_does_not_raise(self):
        """Even a caller passing junk must not take the run down — fall back to
        the env/default ceiling instead."""
        seam = mock.Mock(return_value="Job Object")
        with mock.patch.object(mem_guard, "_apply_ceiling", seam):
            res, out = self._apply(cap_gb="banana", env={})
        self.assertTrue(res)
        seam.assert_called_once_with(int(mem_guard.DEFAULT_CAP_GB * (1 << 30)))
        self.assertIn("ceiling", out)

    def test_nan_cap_argument_takes_the_warn_path_not_a_traceback(self):
        seam = mock.Mock(return_value="Job Object")
        with mock.patch.object(mem_guard, "_apply_ceiling", seam):
            res, out = self._apply(cap_gb=float("nan"), env={})
        self.assertFalse(res)
        self.assertIn("WARNING", out)
        seam.assert_not_called()

    def test_off_switch_skips_the_syscall_and_says_so_loudly(self):
        seam = mock.Mock(return_value="Job Object")
        with mock.patch.object(mem_guard, "_apply_ceiling", seam):
            res, out = self._apply(env={mem_guard.CAP_ENV_VAR: "off"})
        self.assertFalse(res)
        seam.assert_not_called()
        self.assertIn("DISABLED", out)
        self.assertIn(mem_guard.CAP_ENV_VAR, out)

    def test_idempotent_second_call_is_a_no_op(self):
        seam = mock.Mock(return_value="Job Object")
        with mock.patch.object(mem_guard, "_apply_ceiling", seam):
            first, out1 = self._apply(env={})
            second, out2 = self._apply(env={})
            third, out3 = self._apply(cap_gb=1)
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertTrue(third)
        self.assertEqual(seam.call_count, 1, "ceiling re-applied on 2nd call")
        self.assertNotEqual(out1.strip(), "")
        self.assertEqual(out2, "", "banner printed twice")
        self.assertEqual(out3, "")

    def test_idempotent_after_a_failure_too(self):
        boom = mock.Mock(side_effect=OSError("denied"))
        with mock.patch.object(mem_guard, "_apply_ceiling", boom):
            first, out1 = self._apply(env={})
            second, out2 = self._apply(env={})
        self.assertFalse(first)
        self.assertFalse(second)
        self.assertEqual(boom.call_count, 1)
        self.assertIn("WARNING", out1)
        self.assertEqual(out2, "")

    def test_quiet_suppresses_output_but_not_the_effect(self):
        seam = mock.Mock(return_value="Job Object")
        with mock.patch.object(mem_guard, "_apply_ceiling", seam):
            res, out = self._apply(env={}, quiet=True)
        self.assertTrue(res)
        self.assertEqual(out, "")
        seam.assert_called_once()


class PlatformDispatchTests(unittest.TestCase):
    """_apply_ceiling picks by os.name and refuses politely elsewhere. No real
    job object / rlimit is created here."""

    def test_windows_branch(self):
        win = mock.Mock(return_value="Job Object")
        with mock.patch.object(os, "name", "nt"), \
             mock.patch.object(mem_guard, "_apply_windows_job_object", win):
            self.assertEqual(mem_guard._apply_ceiling(123), "Job Object")
        win.assert_called_once_with(123)

    def test_posix_branch(self):
        posix = mock.Mock(return_value="RLIMIT_AS")
        with mock.patch.object(os, "name", "posix"), \
             mock.patch.object(mem_guard, "_apply_posix_rlimit", posix):
            self.assertEqual(mem_guard._apply_ceiling(456), "RLIMIT_AS")
        posix.assert_called_once_with(456)

    def test_unknown_platform_raises_for_the_caller_to_swallow(self):
        with mock.patch.object(os, "name", "java"):
            with self.assertRaises(RuntimeError):
                mem_guard._apply_ceiling(1)

    def test_uses_os_name_not_the_spoofed_sys_platform(self):
        """run_tests_ci_sim.py rewrites sys.platform to "linux" on this Windows
        box; the guard must key off the honest kernel identity instead."""
        win = mock.Mock(return_value="Job Object")
        with mock.patch.object(os, "name", "nt"), \
             mock.patch.object(sys, "platform", "linux"), \
             mock.patch.object(mem_guard, "_apply_windows_job_object", win):
            mem_guard._apply_ceiling(1)
        win.assert_called_once()

    def _fake_resource(self, hard):
        """A stand-in for the POSIX ``resource`` module — lets the real
        _apply_posix_rlimit run on ANY platform (it does ``import resource``
        inside the function, so sys.modules is the seam). Nothing is limited:
        setrlimit is a recorder."""
        fake = types.SimpleNamespace(RLIMIT_AS=9, RLIM_INFINITY=-1, calls=[])
        fake.getrlimit = lambda which: (-1, hard)
        fake.setrlimit = lambda which, pair: fake.calls.append((which, pair))
        return fake

    def test_posix_rlimit_sets_the_ceiling(self):
        fake = self._fake_resource(hard=-1)  # RLIM_INFINITY
        with mock.patch.dict(sys.modules, {"resource": fake}):
            self.assertEqual(mem_guard._apply_posix_rlimit(1 << 33), "RLIMIT_AS")
        self.assertEqual(fake.calls, [(9, (1 << 33, -1))])

    def test_posix_rlimit_never_tries_to_raise_the_hard_limit(self):
        """An unprivileged process cannot raise its hard limit — asking would
        just throw and lose the ceiling entirely, so clamp under it instead."""
        fake = self._fake_resource(hard=1 << 31)  # 2 GB inherited hard cap
        with mock.patch.dict(sys.modules, {"resource": fake}):
            mem_guard._apply_posix_rlimit(1 << 33)  # asked for 8 GB
        self.assertEqual(fake.calls, [(9, (1 << 31, 1 << 31))])

    def test_no_platform_only_imports_at_module_scope(self):
        """ctypes.wintypes (Windows-only) and resource (POSIX-only) must be
        imported INSIDE their branch — a top-level import would make
        ``import tools.mem_guard`` explode on the other platform, and under the
        ci-sim (which blocks ctypes.wintypes) it would break every runner."""
        tree = ast.parse(_source(os.path.join(_TOOLS_DIR, "mem_guard.py")))
        banned = {"resource", "ctypes.wintypes", "ctypes"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name, banned,
                                     f"{alias.name} imported at module scope")
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module or "", banned,
                                 f"from {node.module} at module scope")


class RunnerWiringTests(unittest.TestCase):
    """STALE-DUPLICATE GUARD. The ceiling is worthless if a runner forgets it,
    applies it after the first test import, or if a new runner skips it."""

    def _guard_and_first_test_touch(self, filename):
        tree = ast.parse(_source(os.path.join(_TOOLS_DIR, filename)))
        main = _main_func(tree)
        guard_line = None
        touch_line = None
        for node in ast.walk(main):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            # ast.walk is not source-ordered, so take the minimum line, not
            # the first one seen.
            if name == "apply_memory_ceiling":
                if guard_line is None or node.lineno < guard_line:
                    guard_line = node.lineno
            elif name in _TEST_TOUCHING_CALLS:
                if touch_line is None or node.lineno < touch_line:
                    touch_line = node.lineno
        return guard_line, touch_line

    def test_every_runner_imports_the_guard(self):
        for filename in _GUARDED_RUNNERS:
            with self.subTest(runner=filename):
                src = _source(os.path.join(_TOOLS_DIR, filename))
                self.assertIn("from tools.mem_guard import apply_memory_ceiling",
                              src, f"{filename} no longer imports the guard")

    def test_every_runner_applies_the_ceiling_before_touching_tests(self):
        for filename in _GUARDED_RUNNERS:
            with self.subTest(runner=filename):
                guard_line, touch_line = self._guard_and_first_test_touch(filename)
                self.assertIsNotNone(
                    guard_line,
                    f"{filename}: main() never calls apply_memory_ceiling()")
                self.assertIsNotNone(
                    touch_line, f"{filename}: main() no longer runs any tests?")
                self.assertLess(
                    guard_line, touch_line,
                    f"{filename}: the memory ceiling is applied at line "
                    f"{guard_line}, AFTER tests/subprocesses start at line "
                    f"{touch_line} — it must be the first thing main() does")

    @staticmethod
    def _discovery_runners() -> list[str]:
        """Every tools/*.py that loads/discovers the unittest suite."""
        found = []
        for name in sorted(os.listdir(_TOOLS_DIR)):
            if not name.endswith(".py"):
                continue
            src = _source(os.path.join(_TOOLS_DIR, name))
            if any(marker in src for marker in _DISCOVERY_MARKERS):
                found.append(name)
        return found

    def test_no_new_unguarded_test_runner_appears(self):
        """Any tools/ module that discovers the unittest suite must apply the
        ceiling (or be listed, with a reason, in _KNOWN_UNGUARDED)."""
        offenders = []
        for name in self._discovery_runners():
            if name in _KNOWN_UNGUARDED:
                continue
            src = _source(os.path.join(_TOOLS_DIR, name))
            if "apply_memory_ceiling" not in src:
                offenders.append(name)
        self.assertEqual(
            offenders, [],
            "these tools/ modules run the unittest suite with NO memory "
            "ceiling — call tools.mem_guard.apply_memory_ceiling() first "
            f"thing in main(): {offenders}")

    def test_guarded_list_covers_every_discovery_runner(self):
        """The positive half of the scan: the ordering/import checks above only
        look at the files NAMED in _GUARDED_RUNNERS, so a runner silently
        dropped from that tuple would stop being checked for *when* it applies
        the ceiling. Every discovery runner must be accounted for — guarded, or
        explicitly exempted."""
        accounted = set(_GUARDED_RUNNERS) | _KNOWN_UNGUARDED
        found = set(self._discovery_runners())
        self.assertIn("run_coverage.py", accounted,
                      "run_coverage.py is a discovery runner CI invokes "
                      "directly — it must stay in _GUARDED_RUNNERS")
        self.assertEqual(
            sorted(found - accounted), [],
            "discovery runners in tools/ that no test checks for the "
            f"ceiling: {sorted(found - accounted)}")
        self.assertEqual(
            sorted(accounted - found), [],
            "listed runners that no longer discover the suite (stale "
            f"entries): {sorted(accounted - found)}")

    def test_exemption_list_and_guarded_list_never_overlap(self):
        """A file cannot be both wired up and exempted — that pairing is how an
        exemption survives the wiring it was supposed to be replaced by."""
        overlap = sorted(set(_GUARDED_RUNNERS) & _KNOWN_UNGUARDED)
        self.assertEqual(overlap, [],
                         f"listed as guarded AND exempted: {overlap}")

    def test_known_unguarded_list_stays_honest(self):
        """Entries in _KNOWN_UNGUARDED must still exist and still be runners;
        once one is wired up (or deleted) the exemption has to go, or it rots
        into a lie that hides the next gap."""
        for name in sorted(_KNOWN_UNGUARDED):
            path = os.path.join(_TOOLS_DIR, name)
            with self.subTest(module=name):
                self.assertTrue(os.path.exists(path),
                                f"{name} is gone — drop it from _KNOWN_UNGUARDED")
                src = _source(path)
                self.assertNotIn(
                    "apply_memory_ceiling", src,
                    f"{name} now applies the ceiling — remove it from "
                    "_KNOWN_UNGUARDED so the scan covers it")


def _is_apply_call(node: ast.Call) -> bool:
    """``apply_memory_ceiling(...)`` however it was imported or aliased —
    tests/__init__.py binds it as ``_apply_memory_ceiling`` so the guards cannot
    be shadowed by a test module's own name."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id.lstrip("_") == "apply_memory_ceiling"
    if isinstance(func, ast.Attribute):
        return func.attr.lstrip("_") == "apply_memory_ceiling"
    return False


class ChokepointWiringTests(unittest.TestCase):
    """THE CHOKEPOINT, and the one place this guard was missing from.

    Importing ANY test module imports the ``tests`` package first, so
    ``tests/__init__.py`` is what covers ``python -m unittest tests.foo``,
    ``unittest discover``, an IDE runner and the scratchpad harness — every path
    the three ``tools/`` runners never see. Both sibling guards are armed there;
    the memory ceiling was not, so a TARGETED run was uncapped while printing
    ``[browser-guard] armed …`` and saying nothing about RAM.

    That matters because the 144 GB bugcheck came from BISECTING a suspected
    leak — i.e. re-running one suite, which is exactly the
    ``python -m unittest tests.<suite>`` path with no ceiling. Absence of a line
    is not a signal anyone reads."""

    _TESTS_INIT = os.path.join(_PROJECT_ROOT, "tests", "__init__.py")

    def _src(self):
        with open(self._TESTS_INIT, "r", encoding="utf-8") as fh:
            return fh.read()

    def test_tests_package_applies_the_memory_ceiling(self):
        src = self._src()
        self.assertIn("mem_guard", src,
                      "tests/__init__.py no longer references the memory guard")
        tree = ast.parse(src)
        found = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and _is_apply_call(n)]
        self.assertTrue(
            found,
            "tests/__init__.py must call apply_memory_ceiling() beside the two "
            "other guards — without it `python -m unittest tests.<suite>` (the "
            "bisect path that caused the 144 GB bugcheck) runs with NO ceiling")

    def test_the_ceiling_is_applied_at_import_time(self):
        """A call parked inside a function that nothing invokes would satisfy
        the test above and cap nothing."""
        tree = ast.parse(self._src())
        for node in tree.body:
            self.assertNotIsInstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                "tests/__init__.py must stay a flat module body so the ceiling "
                "is applied at import time")

    def test_the_ceiling_cannot_break_collection(self):
        tree = ast.parse(self._src())
        wrapped = any(
            isinstance(node, ast.Try)
            and any(isinstance(inner, ast.Call) and _is_apply_call(inner)
                    for inner in ast.walk(node))
            for node in tree.body)
        self.assertTrue(wrapped,
                        "the apply_memory_ceiling() call in tests/__init__.py "
                        "must sit inside a try/except — an unbounded run still "
                        "beats no run at all")

    def test_the_ceiling_is_applied_before_the_other_guards(self):
        """It bounds the guards' own imports too, and it is the ordering the
        three runners already use — keeping them identical is what stops one
        copy of the rule rotting."""
        tree = ast.parse(self._src())
        lines = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ""
            if _is_apply_call(node):
                name = "apply_memory_ceiling"
            elif (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "install"
                    and isinstance(node.func.value, ast.Name)):
                name = node.func.value.id.lstrip("_")
            if name in ("apply_memory_ceiling", "live_data_guard",
                        "browser_guard"):
                lines.setdefault(name, node.lineno)
        self.assertIn("apply_memory_ceiling", lines)
        for other in ("live_data_guard", "browser_guard"):
            with self.subTest(guard=other):
                self.assertLess(lines["apply_memory_ceiling"], lines[other],
                                "the memory ceiling must be applied before "
                                f"{other}.install() in tests/__init__.py")


class JobObjectFlagTests(unittest.TestCase):
    """KILL_ON_JOB_CLOSE + JOB_MEMORY apply to EVERY descendant, and neither
    ``CREATE_NO_WINDOW`` nor ``DETACHED_PROCESS`` breaks a child out of a Job
    Object — only ``CREATE_BREAKAWAY_FROM_JOB`` does, and that is refused unless
    the job carries ``JOB_OBJECT_LIMIT_BREAKAWAY_OK``.

    So a process the suite deliberately starts to OUTLIVE the run — the
    monolith's ``_ensure_ollama_running`` spawns ``ollama serve`` and its own
    comment says the server 'still outlives JARVIS' — joins the job, counts its
    ~13.5 GB model load against the 8 GB job-wide cap (failing the RUNNER's
    allocations, nowhere near the offending test), and is TERMINATED when the
    runner's handle closes. Killing ollama is a standing 'never do this' rule
    for this box, so the job must at least make the escape possible."""

    @unittest.skipUnless(os.name == "nt", "Job Objects are Windows-only")
    def test_the_job_allows_a_deliberate_breakaway(self):
        import ctypes
        # tools/run_tests_ci_sim.py emulates the Linux runner IN THIS PROCESS:
        # it deletes ``ctypes.WinDLL`` and evicts ``ctypes.wintypes`` from
        # sys.modules behind a blocking import shim. ``os.name`` stays "nt" —
        # deliberately, it is the honest kernel identity the ceiling itself keys
        # off (see test_uses_os_name_not_the_spoofed_sys_platform) — so this
        # Windows-only test still RUNS under the sim and has to supply the
        # Windows ctypes surface itself instead of exploding on the removed
        # attribute. It is NOT skipped there: the flags asserted below are the
        # whole point, and a test that quietly stops running is how a guard
        # stops being checked.
        #
        # The real wintypes module is still reachable as an ATTRIBUTE of the
        # ctypes package (the sim clears sys.modules, not the attribute), and
        # every runner applies the memory ceiling — which imports it — long
        # before the platform flip. Putting it back into sys.modules for the
        # duration is the sim's own documented pattern for a win-only import.
        wintypes = (sys.modules.get("ctypes.wintypes")
                    or getattr(ctypes, "wintypes", None))
        if wintypes is None:  # pragma: no cover - unreachable on a real nt box
            self.skipTest("ctypes.wintypes is unavailable on this host")
        captured = {}
        # getattr, not attribute access: under the sim there is no WinDLL to
        # snapshot. It is only ever used by the unreachable non-kernel32 branch
        # below, so its absence must not fail the test before it starts.
        real_windll = getattr(ctypes, "WinDLL", None)

        # Plain functions, assigned as INSTANCE attributes so they stay plain
        # functions (a bound method has no __dict__, and mem_guard sets
        # .restype / .argtypes on every prototype it uses).
        def _create(*a):
            return 0x1234

        def _setinfo(job, klass, ptr, size):
            blob = ctypes.string_at(ptr, size)
            # LimitFlags is the 3rd field: two c_longlong then a DWORD.
            captured["flags"] = int.from_bytes(blob[16:20], "little")
            return 1

        def _assign(*a):
            return 1

        def _current(*a):
            return 7

        class _FakeK32:
            def __init__(self):
                self.CreateJobObjectW = _create
                self.SetInformationJobObject = _setinfo
                self.AssignProcessToJobObject = _assign
                self.GetCurrentProcess = _current

        def _fake_windll(name, *a, **k):
            if name == "kernel32":
                return _FakeK32()
            if real_windll is None:  # pragma: no cover - only kernel32 is asked
                raise OSError(f"WinDLL({name!r}) is unavailable under the CI sim")
            return real_windll(name, *a, **k)   # pragma: no cover

        # create=True so the patch also works where the sim removed WinDLL; the
        # attribute is deleted again on exit, leaving the sim's Linux face
        # exactly as it was.
        with mock.patch.dict(sys.modules, {"ctypes.wintypes": wintypes}),              mock.patch.object(ctypes, "WinDLL", _fake_windll, create=True):
            mem_guard._apply_windows_job_object(1 << 30)
        mem_guard._job_handle = None

        BREAKAWAY_OK = 0x00000800
        self.assertTrue(
            captured["flags"] & BREAKAWAY_OK,
            "the job does not carry JOB_OBJECT_LIMIT_BREAKAWAY_OK, so a child "
            "that is SUPPOSED to outlive the run (ollama serve) cannot escape "
            "KILL_ON_JOB_CLOSE even by asking for CREATE_BREAKAWAY_FROM_JOB")
        for name, bit in (("PROCESS_MEMORY", 0x00000100),
                          ("JOB_MEMORY", 0x00000200),
                          ("KILL_ON_JOB_CLOSE", 0x00002000)):
            with self.subTest(flag=name):
                self.assertTrue(captured["flags"] & bit,
                                f"{name} was dropped from the job")

    def test_the_docstring_documents_the_breakaway(self):
        """STALE-DUPLICATE GUARD: the contract lives in the docstring, and a
        flag added without documenting it is how the next reader learns the
        wrong rule."""
        src = mem_guard.__doc__ or ""
        self.assertIn("BREAKAWAY", src.upper(),
                      "tools/mem_guard.py's docstring still says 'no child "
                      "outlives the runner' with no mention of how a child that "
                      "is meant to (ollama serve) can opt out")



if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    unittest.main()
