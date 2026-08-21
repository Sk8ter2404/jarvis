#!/usr/bin/env python3
"""Invariants for the LIVE-DATA guard — the net that stops a test run from
destroying the owner's runtime state.

WHY THIS FILE EXISTS
====================
On 2026-08-20 a test run deleted the owner's hand-written
``data/clean_shutdown.flag`` four times in one day (JARVIS resurrected at 03:29,
10:24, 17:44 and 18:34 against his explicit instruction to stay down). The
chain, proven by interception rather than inference:

    tests/test_audit_2026_07_14.py  LlmIndependentControlPlaneTests._dispatch
      patched "core.actions._act_restart"   <-- WRONG NAME
      bobert_companion._dispatch_tray_command resolves the MODULE GLOBAL,
      so the REAL core.actions._act_restart ran, which starts a daemon thread
      that 1.5s later spawns a DETACHED `python bobert_companion.py` whose
      boot path unlinks the live flag, then TerminateProcess()es the runner.

Three separate things had to be true for that to reach live data, and this file
pins all three so none of them can quietly come back:

1. the wrong-target patch (``PatchTargetTests``),
2. the missing process-wide net (``ChokepointWiringTests`` + ``GuardIsArmedTests``),
3. a flag path built off ``__file__`` instead of ``core.paths``, which is why no
   env redirect could help (``LiveFlagPathTests``).

The scans here are AST-based ON PURPOSE. ``tests/test_staging_data_isolation.py``
uses line-oriented regexes, and BOTH live-flag sites in ``bobert_companion.py``
wrap across a newline — so the existing invariant cannot see the very lines that
caused this incident. A rule that is green because its scanner cannot reach the
code is this repo's #1 bug class wearing a lab coat.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import unittest

from tests import live_data_guard

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS_DIR = os.path.join(_PROJECT_ROOT, "tests")
_TESTS_INIT = os.path.join(_TESTS_DIR, "__init__.py")
_TOOLS_DIR = os.path.join(_PROJECT_ROOT, "tools")

# The tools/ runners, kept in lockstep with tests/test_browser_guard.py and
# tests/test_mem_guard.py (both call this same tuple _GUARDED_RUNNERS). Every
# one of them arms the browser guard, so every one of them has to arm THIS
# guard first — see test_every_runner_arms_this_guard_before_the_browser_guard.
_GUARDED_RUNNERS = ("run_tests.py", "run_tests_ci_sim.py", "run_coverage.py")

_ROOTISH = {
    "PROJECT_DIR", "_PROJECT_DIR", "PROJ_DIR", "_PROJ_DIR", "PROJ", "_PROJ",
    "HERE", "_HERE", "BASE_DIR", "_BASE_DIR", "ROOT", "_ROOT",
    "PROJECT_ROOT", "_PROJECT_ROOT",
}


def _source(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _py_files(root: str):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs
                   if d not in ("__pycache__", ".git", ".claude", "backups",
                                "_backups", "dist", "models", "logs",
                                "logs_staging", "data", "data_staging",
                                "node_modules", "Robot Project")]
        for fn in files:
            if fn.endswith(".py"):
                yield os.path.join(base, fn)


def _rootish_expr(node: ast.AST) -> bool:
    """True when ``node`` evaluates to the PROJECT ROOT (not a tempdir).

    Covers the bare name (``PROJECT_DIR``), an attribute (``mod.PROJECT_DIR``)
    and the ``os.path.dirname(os.path.abspath(__file__))`` idiom that both
    live-flag sites use."""
    if isinstance(node, ast.Name):
        return node.id in _ROOTISH
    if isinstance(node, ast.Attribute):
        return node.attr in _ROOTISH
    if isinstance(node, ast.Call):
        dumped = ast.dump(node)
        return "__file__" in dumped and "dirname" in dumped
    return False


def _data_joins(tree: ast.AST):
    """Yield linenos of ``os.path.join(<project root>, "data", ...)`` calls."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "join"
                and len(node.args) >= 2):
            continue
        if not _rootish_expr(node.args[0]):
            continue
        second = node.args[1]
        if isinstance(second, ast.Constant) and second.value == "data":
            yield node.lineno


class ChokepointWiringTests(unittest.TestCase):
    """STALE-DUPLICATE GUARD. The net is worthless if the wiring is dropped, so
    these read the SOURCE of the one place that installs it.

    tests/__init__.py is THE chokepoint: importing any test module imports the
    package first, so it covers `python -m unittest tests.foo`, `unittest
    discover`, the scratchpad capped harness and an IDE runner — every path the
    tools/ runners never see. The incident escaped precisely because the data
    redirect lived only in the runners."""

    def test_tests_package_installs_the_guard(self):
        src = _source(_TESTS_INIT)
        self.assertIn("live_data_guard", src,
                      "tests/__init__.py no longer references the live-data guard")
        tree = ast.parse(src)
        found = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "install"
                 and isinstance(n.func.value, ast.Name)
                 and n.func.value.id.lstrip("_").startswith("live_data_guard")]
        self.assertTrue(found,
                        "tests/__init__.py must call live_data_guard.install() — "
                        "without it a test run can delete the owner's live data/")

    def test_install_is_not_hidden_inside_a_function(self):
        """It has to run on IMPORT. A call parked in a helper that nothing
        invokes would pass the test above and guard nothing."""
        tree = ast.parse(_source(_TESTS_INIT))
        for node in tree.body:
            self.assertNotIsInstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                "tests/__init__.py must stay a flat module body so the guard "
                "install runs at import time")

    def test_install_cannot_break_collection(self):
        """An exception from the guard must never stop the suite being
        collected — an unguarded run still beats no run at all."""
        tree = ast.parse(_source(_TESTS_INIT))
        wrapped = any(
            isinstance(node, ast.Try)
            and any(isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "install"
                    and isinstance(inner.func.value, ast.Name)
                    and inner.func.value.id.lstrip("_").startswith("live_data_guard")
                    for inner in ast.walk(node))
            for node in tree.body)
        self.assertTrue(wrapped,
                        "the live_data_guard.install() call in tests/__init__.py "
                        "must sit inside a try/except")

    def test_this_guard_is_armed_before_the_browser_guard(self):
        """LOAD-BEARING ORDER, not style.

        Both guards wrap ``os.startfile``. ``tools/browser_guard.py``'s stub
        BLOCKS — it never calls the function it wrapped — so whichever guard is
        installed LAST is the only one that ever runs, and whichever is
        installed FIRST is the one the other's ``_reset_for_tests()`` restores
        instead of discarding. Arming the live-data guard first therefore gives
        both properties at once: while the browser guard is on, nothing
        launches; when it is disabled via JARVIS_ALLOW_REAL_BROWSER, the
        live-data hook is what is left holding the shell route to a live
        JARVIS."""
        tree = ast.parse(_source(_TESTS_INIT))
        order = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "install"
                    and isinstance(node.func.value, ast.Name)):
                name = node.func.value.id.lstrip("_")
                if name.startswith(("live_data_guard", "browser_guard")):
                    order.append((node.lineno, name))
        order.sort()
        names = [n for _, n in order]
        self.assertIn("live_data_guard", names)
        self.assertIn("browser_guard", names)
        self.assertLess(
            names.index("live_data_guard"), names.index("browser_guard"),
            "tests/__init__.py must call live_data_guard.install() BEFORE "
            "browser_guard.install() — otherwise the browser guard's "
            "os.startfile stub sits underneath ours and its _reset_for_tests() "
            "throws the live-data hook away")

    def test_every_runner_arms_this_guard_before_the_browser_guard(self):
        """THE SAME ORDER RULE, IN THE OTHER FOUR PLACES IT HAS TO HOLD.

        The rule above lived ONLY in tests/__init__.py, and that is exactly this
        repo's #1 bug class: a rule applied in one copy while the others rot.
        Each tools/ runner called ``browser_guard.install()`` in a process that
        had no live-data guard in it yet, so the browser guard's snapshot of
        ``os.startfile`` was the RAW function — and the first
        ``_reset_for_tests()`` in tests/test_browser_guard.py put that back,
        leaving the remaining ~13,000 tests of a full CI run with NO
        interception on the shell route to a live JARVIS while
        ``is_armed()`` still answered True. Measured 2026-08-20.

        A runner satisfies this either by calling ``live_data_guard.install()``
        itself or by importing the ``tests`` package (the chokepoint, which does
        it) — whichever it does, it must come BEFORE browser_guard.install()."""
        for filename in _GUARDED_RUNNERS:
            with self.subTest(runner=filename):
                path = os.path.join(_TOOLS_DIR, filename)
                self.assertTrue(os.path.exists(path),
                                f"{filename} is gone — update _GUARDED_RUNNERS")
                tree = ast.parse(_source(path))
                arm_line = browser_line = None
                for node in ast.walk(tree):
                    arms_here = False
                    if isinstance(node, ast.Import):
                        arms_here = any(a.name.split(".")[0] == "tests"
                                        for a in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        arms_here = (node.module or "").split(".")[0] == "tests"
                    if arms_here and (arm_line is None or node.lineno < arm_line):
                        arm_line = node.lineno
                    if (isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "install"
                            and isinstance(node.func.value, ast.Name)):
                        name = node.func.value.id.lstrip("_")
                        if name.startswith("live_data_guard"):
                            if arm_line is None or node.lineno < arm_line:
                                arm_line = node.lineno
                        elif name.startswith("browser_guard"):
                            if browser_line is None or node.lineno < browser_line:
                                browser_line = node.lineno
                self.assertIsNotNone(
                    browser_line,
                    f"{filename} no longer installs the browser guard")
                self.assertIsNotNone(
                    arm_line,
                    f"{filename} arms the browser guard but never arms the "
                    f"live-data guard — import the tests package (the "
                    f"chokepoint) or call live_data_guard.install() first")
                self.assertLess(
                    arm_line, browser_line,
                    f"{filename}: the live-data guard is armed at line "
                    f"{arm_line}, AFTER the browser guard at line "
                    f"{browser_line}. The browser guard then snapshots the RAW "
                    f"os.startfile and its _reset_for_tests() discards the "
                    f"live-data hook for the rest of the run")


class GuardIsArmedTests(unittest.TestCase):
    """Prove the guard is REALLY armed in THIS process — not merely importable.

    Every assertion below is shaped like the damage it prevents but aimed at a
    target that is harmless if the guard has regressed: a path that does not
    exist, or a script name in a tempdir. A regression therefore shows up as a
    plain OSError, never as a deleted flag or a real JARVIS."""

    def setUp(self):
        if not live_data_guard._INSTALLED:
            self.skipTest("live-data guard not installed (JARVIS_ALLOW_LIVE_DATA?)"
                          " — ChokepointWiringTests covers the wiring")

    def test_delete_under_live_data_is_blocked(self):
        victim = os.path.join(live_data_guard.LIVE_DATA_DIR,
                              "__guard_probe_does_not_exist__")
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            os.remove(victim)
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            os.unlink(victim)

    def test_rmtree_of_live_data_is_blocked(self):
        import shutil
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            shutil.rmtree(live_data_guard.LIVE_DATA_DIR)

    def test_pathlib_unlink_under_live_data_is_blocked(self):
        import pathlib
        victim = pathlib.Path(live_data_guard.LIVE_DATA_DIR) / "__guard_probe__"
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            victim.unlink()

    def test_writing_the_clean_shutdown_flag_is_blocked(self):
        """The owner's sentinel is PROSE ('owner asked JARVIS stay down'), not
        the str(time.time()) production writes. A stray write would silently
        replace his note with a timestamp."""
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            open(live_data_guard.CLEAN_SHUTDOWN_FLAG, "w").close()

    def test_spawning_a_live_jarvis_is_blocked(self):
        """THE incident. Aimed at a tempdir copy of the name so a regression
        launches nothing real."""
        fake = os.path.join(tempfile.gettempdir(), "bobert_companion.py")
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            subprocess.Popen([sys.executable, fake])
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            subprocess.run([sys.executable, fake])

    def test_boot_script_is_blocked(self):
        fake = os.path.join(tempfile.gettempdir(), "_boot_jarvis.ps1")
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            subprocess.Popen(["powershell", "-File", fake])

    # ── the guard must not be able to go quietly OFF mid-run ───────────────
    def test_every_interception_is_still_in_place(self):
        """``_INSTALLED`` only says install() ran once. This asks whether the
        run is actually protected RIGHT NOW, which is the question the two
        spawn tests above were silently answering wrong for ~12,000 tests."""
        self.assertEqual(
            live_data_guard.unarmed_hooks(), [],
            "these live-data interceptions have been displaced by something "
            "else in this run — the guard is reporting itself installed while "
            "not protecting the owner's data/")

    def test_the_interception_lives_on_the_base_popen_class(self):
        """Install-ORDER independence, in one assertion.

        Whatever ``subprocess.Popen`` currently names, the hook is on the class
        at the bottom of its MRO — the one object a rebind cannot replace. Any
        other guard (tools/browser_guard.py does exactly this) can only add a
        SUBCLASS on top, which inherits the hook, so it does not matter which of
        the two installs first or how often either re-installs."""
        base = live_data_guard._popen_base(subprocess.Popen)
        self.assertIsNotNone(base)
        self.assertTrue(
            live_data_guard._marked(base.__init__),
            "the Popen interception is not on the base class — it is parked on "
            "whatever subprocess.Popen named at install time, and the next "
            "rebind will throw it away")

    def test_guard_survives_a_popen_rebind_and_a_fresh_wrapper_subclass(self):
        """REGRESSION, 2026-08-20. The old guard patched
        ``subprocess.Popen.__init__`` on whatever the module attribute named.
        ``tools/browser_guard.py`` rebinds that attribute to a fresh
        ``GuardedPopen`` subclass, and tests/test_browser_guard.py's
        ``_unlatched()`` restores the stock class and then re-wraps — after
        which the live-data hook was gone and a detached JARVIS could spawn.
        Both halves of that are reproduced here."""
        fake = os.path.join(tempfile.gettempdir(), "bobert_companion.py")
        base = live_data_guard._popen_base(subprocess.Popen)
        saved = subprocess.Popen
        try:
            subprocess.Popen = base            # a snapshot/restore wipes a wrapper
            with self.assertRaises(live_data_guard.LiveDataGuardError):
                subprocess.Popen([sys.executable, fake])

            class ReWrapped(base):             # ...and a coexisting guard re-wraps
                pass

            subprocess.Popen = ReWrapped
            with self.assertRaises(live_data_guard.LiveDataGuardError):
                subprocess.Popen([sys.executable, fake])
        finally:
            subprocess.Popen = saved

    def test_guard_survives_the_browser_guards_reset_and_reinstall(self):
        """The literal sequence that disarmed it, run for real.

        Mirrors tests/test_browser_guard.py's ``_unlatched()``: drop the browser
        guard's latch, re-arm it, and re-arm again in a finally so this test can
        never leave the rest of the run able to open real windows."""
        from tools import browser_guard
        if not browser_guard.is_installed():
            self.skipTest("browser guard disabled via "
                          + browser_guard.ALLOW_ENV_VAR)
        fake = os.path.join(tempfile.gettempdir(), "bobert_companion.py")
        saved_ledger = browser_guard.blocked_attempts()
        try:
            browser_guard._reset_for_tests()
            browser_guard.install(quiet=True)
            with self.assertRaises(live_data_guard.LiveDataGuardError):
                subprocess.Popen([sys.executable, fake])
            self.assertEqual(live_data_guard.unarmed_hooks(), [])
        finally:
            browser_guard._reset_for_tests()
            browser_guard.install(quiet=True)
            browser_guard._blocked.extend(saved_ledger)
        self.assertTrue(browser_guard.is_armed(),
                        "this test left the browser guard disarmed")

    def test_install_repairs_a_displaced_hook(self):
        """install() is idempotent AND repairing: a second entry point calling
        it re-installs what was lost instead of returning a no-op because a
        latch says the guard is already on."""
        victim = os.path.join(live_data_guard.LIVE_DATA_DIR, "__guard_probe__")
        saved = os.rmdir
        try:
            os.rmdir = lambda *a, **k: None            # displaced by something
            self.assertIn("os.rmdir", live_data_guard.unarmed_hooks())
            self.assertFalse(live_data_guard.is_armed())
            live_data_guard.install()                  # ...and repaired
            self.assertNotIn("os.rmdir", live_data_guard.unarmed_hooks())
            with self.assertRaises(live_data_guard.LiveDataGuardError):
                os.rmdir(victim)
        finally:
            os.rmdir = saved

    # ── negative controls: the guard must not break ordinary test work ──────
    def test_unrelated_delete_still_works(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        os.remove(path)
        self.assertFalse(os.path.exists(path))

    def test_unrelated_spawn_still_works(self):
        out = subprocess.run([sys.executable, "-c", "print('ok')"],
                             capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "ok")

    def test_prompt_text_mentioning_a_script_is_not_blocked(self):
        """tests/test_multi_agent_pipeline.py passes multi-line PROMPTS to the
        claude CLI that talk about the upgrade pipeline. Matching on a substring
        of the joined command line would block those; the guard matches each
        argument's BASENAME instead."""
        prompt = ("You are the REVIEWER stage of the JARVIS multi-agent "
                  "upgrade pipeline. Do not edit upgrade_jarvis.py logic.")
        self.assertIsNone(live_data_guard._boot_script_in(
            ["claude", "--print", "--", prompt]))

    def test_real_boot_script_argument_is_detected(self):
        self.assertEqual(
            live_data_guard._boot_script_in(
                [sys.executable, r"C:\JARVIS\bobert_companion.py"]),
            "bobert_companion.py")


class PatchTargetTests(unittest.TestCase):
    """The exact defect that fired: patching ``core.actions._act_restart`` when
    the caller resolves the monolith's MODULE GLOBAL.

    bobert_companion._dispatch_tray_command deliberately calls the module-level
    name (see the comment at bobert_companion.py:3322-3340), so a string patch
    of ``core.actions.*`` rebinds a name nothing on that path reads — the test
    passes while the REAL action runs. Use
    ``mock.patch.object(bc, "_act_restart")`` instead."""

    _FORBIDDEN = ("core.actions._act_restart", "core.actions._act_shutdown_jarvis")

    def test_no_test_patches_the_wrong_restart_target(self):
        offenders = []
        for path in _py_files(_TESTS_DIR):
            if os.path.abspath(path) == os.path.abspath(__file__):
                continue
            src = _source(path)
            for lineno, line in enumerate(src.splitlines(), 1):
                if "mock.patch(" not in line and "patch(" not in line:
                    continue
                for target in self._FORBIDDEN:
                    if f'"{target}"' in line or f"'{target}'" in line:
                        offenders.append(
                            f"{os.path.relpath(path, _PROJECT_ROOT)}:{lineno}")
        self.assertEqual(
            offenders, [],
            "these patch core.actions.* by STRING, but the tray dispatch path "
            "resolves bobert_companion's module global — the real action will "
            "run (spawning a live JARVIS / speaking through the owner's "
            "speakers). Use mock.patch.object(bc, ...): " + ", ".join(offenders))


class LiveFlagPathTests(unittest.TestCase):
    """The watchdog handshake flag must never gain a new path construction.

    ``data/clean_shutdown.flag`` is the owner's kill switch for the resurrection
    watchdog. Both existing production sites build it from ``__file__`` rather
    than ``core.paths.data_file()``, which is exactly why JARVIS_DATA_DIR and
    JARVIS_STAGING cannot protect it. Those two are frozen below as known debt;
    a THIRD site — or any site in tests/ — fails this test."""

    # NO LINE NUMBERS HERE, DELIBERATELY.
    # The first version of this ratchet froze {23204, 24170}. Other agents edited
    # the 24k-line monolith the same afternoon, the same two sites slid to 23238
    # and 24204, and the test went red for a change that touched NEITHER of them
    # — a false alarm on a rule nobody can then trust. A line number is not the
    # invariant. "These two constructions, and only these two, build the live
    # flag path" is, so each site is identified by where it lives:
    #   (enclosing function or "<module>", the name it is assigned to).
    # Both parts have to be edited deliberately; neither moves when unrelated
    # code above shifts.
    _KNOWN_PRODUCTION_SITES = {
        os.path.join("bobert_companion.py"): {
            # main()'s boot-path unlink. THIS is the site that deleted the
            # owner's flag on 2026-08-20, reached from a test via a DETACHED
            # bobert_companion.py child.
            ("main", "_clean_flag"),
            # The module-level constant that _write_clean_shutdown_flag writes
            # back to on a clean exit.
            ("<module>", "_CLEAN_SHUTDOWN_FLAG"),
        },
    }

    @staticmethod
    def _identify(node, parents):
        """(enclosing def name or "<module>", assigned name or "<unassigned>")."""
        func = "<module>"
        walker = parents.get(node)
        while walker is not None:
            if isinstance(walker, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = walker.name
                break
            if isinstance(walker, ast.Module):
                break
            walker = parents.get(walker)
        name = "<unassigned>"
        walker = parents.get(node)
        while walker is not None:
            if isinstance(walker, ast.Assign) and walker.targets:
                target = walker.targets[0]
                name = target.id if isinstance(target, ast.Name) else "<complex>"
                break
            if isinstance(walker, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.Module)):
                break
            walker = parents.get(walker)
        return func, name

    def _flag_sites(self, path):
        """``{(enclosing, assigned): (lineno, builds_it_off___file__)}`` for
        every place a clean_shutdown.flag path is CONSTRUCTED (not merely named
        in a comment or a docstring)."""
        try:
            tree = ast.parse(_source(path))
        except SyntaxError:  # pragma: no cover - unparseable file
            return {}
        parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        found = {}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "join"):
                continue
            if not any(isinstance(a, ast.Constant)
                       and a.value == "clean_shutdown.flag" for a in node.args):
                continue
            rooted = bool(node.args) and _rootish_expr(node.args[0])
            found[self._identify(node, parents)] = (node.lineno, rooted)
        return found

    def _production_sites(self):
        actual = {}
        for path in _py_files(_PROJECT_ROOT):
            rel = os.path.relpath(path, _PROJECT_ROOT)
            if rel.split(os.sep)[0] == "tests":
                continue
            sites = self._flag_sites(path)
            if sites:
                actual[rel] = sites
        # tools/jarvis_watchdog.py only READS the flag via os.path.exists; it is
        # allowed and deliberately excluded from the frozen set.
        actual.pop(os.path.join("tools", "jarvis_watchdog.py"), None)
        return actual

    def test_production_flag_sites_are_frozen(self):
        actual = self._production_sites()
        where = {rel: {ident: lineno for ident, (lineno, _r) in sites.items()}
                 for rel, sites in actual.items()}
        self.assertEqual(
            {rel: set(sites) for rel, sites in actual.items()},
            self._KNOWN_PRODUCTION_SITES,
            "the set of places that build a live clean_shutdown.flag path "
            "changed. Every NEW one is another way to destroy the owner's "
            "watchdog kill switch — route it through core.paths.data_file() "
            "instead. If you REMOVED one by fixing it, drop it from "
            "_KNOWN_PRODUCTION_SITES in the same commit. Line numbers are not "
            "part of the expectation; current ones: " + repr(where))

    def test_production_flag_sites_still_bypass_the_data_dir_redirect(self):
        """The reason the whole incident was possible, kept visible.

        Both sites resolve the flag off ``__file__``, so JARVIS_DATA_DIR,
        JARVIS_STAGING and the runners' _redirect_data_dir_to_throwaway() are
        ALL powerless over them and only a process-wide interception
        (tests/live_data_guard.py) can protect the owner's kill switch. The day
        someone routes them through core.paths.data_file() this assertion is
        what tells the next agent the guard's rationale just changed."""
        for rel, sites in self._production_sites().items():
            for ident, (lineno, rooted) in sites.items():
                with self.subTest(site=f"{rel}:{lineno} {ident}"):
                    self.assertTrue(
                        rooted,
                        f"{rel}:{lineno} {ident} no longer builds the flag path "
                        f"from __file__. If it now goes through "
                        f"core.paths.data_file(), that is a FIX — remove it "
                        f"from _KNOWN_PRODUCTION_SITES and say so here.")

    def test_no_test_builds_a_live_flag_path(self):
        """Tests may build the flag inside a tempdir (test_monolith_sec4 and
        test_jarvis_watchdog both do, correctly) — never off the project root."""
        offenders = []
        for path in _py_files(_TESTS_DIR):
            try:
                tree = ast.parse(_source(path))
            except SyntaxError:  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "join"
                        and node.args):
                    continue
                names_flag = any(isinstance(a, ast.Constant)
                                 and a.value == "clean_shutdown.flag"
                                 for a in node.args)
                if names_flag and _rootish_expr(node.args[0]):
                    offenders.append(
                        f"{os.path.relpath(path, _PROJECT_ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [],
                         "a test builds the LIVE clean_shutdown.flag path; use "
                         "a tempdir: " + ", ".join(offenders))


class TestsMustNotResolveLiveDataTests(unittest.TestCase):
    """Source-scanning invariant: no test may resolve a ``data/`` path off the
    project root.

    A test that joins ``PROJECT_DIR`` / ``os.path.dirname(__file__)`` with
    ``"data"`` has walked around ``core.paths.data_dir()``, so neither
    ``JARVIS_DATA_DIR`` nor ``JARVIS_STAGING`` — nor the tools/ runners'
    ``_redirect_data_dir_to_throwaway()`` — can protect the owner's files from
    it. The four entries below are read-only assertions about shipped defaults,
    audited 2026-08-20; the list is a RATCHET, so anything new fails."""

    _ALLOWED = {
        # Defines the live dir on purpose — it is the guard that protects it.
        os.path.join("tests", "live_data_guard.py"),
        # Asserts the SHIPPED DEFAULT constant points into data/; never writes.
        os.path.join("tests", "skills", "test_hud_camera_preview.py"),
        # Existence check on the owner's real settings file to decide whether to
        # skip a default-value assertion; read-only.
        os.path.join("tests", "test_config.py"),
        # The staging-isolation invariant itself: asserts the redirect target is
        # NOT the live data dir, so it must be able to name it.
        os.path.join("tests", "test_staging_data_isolation.py"),
    }

    def test_no_new_test_resolves_a_live_data_path(self):
        offenders = {}
        for path in _py_files(_TESTS_DIR):
            rel = os.path.relpath(path, _PROJECT_ROOT)
            if rel in self._ALLOWED:
                continue
            try:
                tree = ast.parse(_source(path))
            except SyntaxError:  # pragma: no cover
                continue
            lines = sorted(set(_data_joins(tree)))
            if lines:
                offenders[rel] = lines
        self.assertEqual(
            offenders, {},
            "these tests build a data/ path from the PROJECT ROOT, bypassing "
            "core.paths.data_dir() — no env redirect can protect the owner's "
            "files from them. Use tempfile.mkdtemp() (or add a documented, "
            "read-only entry to _ALLOWED): " + repr(offenders))



class ArmednessDetectionTests(unittest.TestCase):
    """The ratchet that answers "is this run REALLY protected?".

    THE DEFECT (2026-08-20, found by adversarial review of the same day's fix)
    -------------------------------------------------------------------------
    ``unarmed_hooks()`` used to answer the Popen question STRUCTURALLY — "does
    whatever ``subprocess.Popen`` currently names carry our marker on its
    ``__init__``?". ``core/no_window_subprocess.install()`` writes its own
    ``__init__`` onto the class ``subprocess.Popen`` names at that moment, which
    during a test run is ``tools/browser_guard.py``'s ``GuardedPopen`` SUBCLASS.
    That shadow was unmarked, so from the first monolith import onward the check
    reported ``['subprocess.Popen.__init__']`` — even though the shadow CHAINS
    to the marked base hook, so the spawn was in fact still blocked.

    Reproduced before the fix:

        python capped_verbose.py "tests.monolith.test_monolith_sec7
            tests.test_live_data_guard" 8 order.log
        -> FAIL: test_every_interception_is_still_in_place
           AssertionError: Lists differ: ['subprocess.Popen.__init__'] != []

    The full CI was green only because ``tests/test_browser_guard.py`` sorts
    between those two modules and its ``_unlatched()`` helper rebinds
    ``subprocess.Popen`` to a fresh subclass, throwing the shadow away. The one
    mechanism built to catch "reports armed while disarmed" was itself green for
    the wrong reason, and one alphabetical accident away from red.

    The honest answer is BEHAVIOUR, not structure: attempt a sentinel spawn and
    see whether it is refused. These tests pin that."""

    def setUp(self):
        if not live_data_guard._INSTALLED:
            self.skipTest("live-data guard not installed (JARVIS_ALLOW_LIVE_DATA?)")
        self._saved_popen = subprocess.Popen
        self._base = live_data_guard._popen_base(subprocess.Popen)
        self._saved_base_init = self._base.__dict__.get("__init__")
        self.addCleanup(self._restore)

    def _restore(self):
        subprocess.Popen = self._saved_popen
        if self._saved_base_init is not None:
            self._base.__init__ = self._saved_base_init
        live_data_guard.install()          # repair-only; puts back anything lost

    def _unguarded_init(self):
        """The innermost ``Popen.__init__`` — the end of the wrapper chain.

        Not just ``__wrapped__`` once: by the time a real run reaches here the
        base carries no_window(live_data(stock)), so one hop lands back INSIDE
        the guard and a "bypass" built from it would still refuse."""
        fn = self._saved_base_init
        for _ in range(12):
            nxt = getattr(fn, "__wrapped__", None)
            if nxt is None or nxt is fn:
                break
            fn = nxt
        return None if live_data_guard._marked(fn) else fn

    def test_a_shadowing_popen_init_does_not_report_the_guard_disarmed(self):
        """THE REPRODUCTION, as a unit test. A subclass ``__init__`` that
        forwards to the marked base hook still BLOCKS the spawn, so reporting it
        unarmed is a false alarm — and a ratchet that cries wolf gets normalised
        as flaky, which is exactly how a REAL displacement goes unnoticed."""
        base = self._base
        chained = base.__init__          # already carries the guard

        def _no_window_init(self, *a, **k):   # what core/no_window_subprocess did
            k.setdefault("creationflags", 0x08000000)
            return chained(self, *a, **k)

        class GuardedPopen(base):
            pass

        GuardedPopen.__init__ = _no_window_init   # the UNMARKED shadow
        subprocess.Popen = GuardedPopen

        fake = os.path.join(tempfile.gettempdir(), "bobert_companion.py")
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            subprocess.Popen([sys.executable, fake])
        self.assertEqual(
            live_data_guard.unarmed_hooks(), [],
            "a shadowing __init__ that still reaches the guard was reported as "
            "a displaced hook — the ratchet is answering structurally instead "
            "of by behaviour")
        self.assertTrue(live_data_guard.is_armed())

    def test_a_shadowing_popen_init_that_bypasses_the_guard_is_reported(self):
        """The other direction, which matters more: a shadow that does NOT
        reach the guard has to be reported, or the ratchet is worthless."""
        base = self._base
        real = self._unguarded_init()
        if real is None:
            self.skipTest("cannot reach the pre-guard __init__ to build a bypass")

        class Bypass(base):
            pass

        Bypass.__init__ = lambda self, *a, **k: real(self, *a, **k)
        subprocess.Popen = Bypass
        self.assertIn("subprocess.Popen.__init__", live_data_guard.unarmed_hooks())
        self.assertFalse(live_data_guard.is_armed())

    def test_install_repairs_a_shadowing_popen_init(self):
        """install() has to REPAIR the shadow, not merely notice it."""
        base = self._base
        real = self._unguarded_init()
        if real is None:
            self.skipTest("cannot reach the pre-guard __init__ to build a bypass")

        class Bypass(base):
            pass

        Bypass.__init__ = lambda self, *a, **k: real(self, *a, **k)
        subprocess.Popen = Bypass
        self.assertFalse(live_data_guard.is_armed())
        live_data_guard.install()
        self.assertEqual(live_data_guard.unarmed_hooks(), [])
        fake = os.path.join(tempfile.gettempdir(), "bobert_companion.py")
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            subprocess.Popen([sys.executable, fake])

    def test_the_no_window_net_installing_after_us_leaves_us_armed(self):
        """The LITERAL suite-order reproduction, without needing two suites.

        Importing the monolith calls ``core.no_window_subprocess.install()``
        while the live-data guard is already armed — that one call is the whole
        of what made ``tests.monolith.test_monolith_sec7`` +
        ``tests.test_live_data_guard`` fail as a pair."""
        from core import no_window_subprocess as nw
        nw.install()
        self.assertEqual(
            live_data_guard.unarmed_hooks(), [],
            "core/no_window_subprocess.install() disarmed the live-data guard "
            "(or made it report itself disarmed) — this is the reproduced "
            "tests.monolith.test_monolith_sec7 + tests.test_live_data_guard "
            "ordering failure")
        fake = os.path.join(tempfile.gettempdir(), "bobert_companion.py")
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            subprocess.Popen([sys.executable, fake])

    def test_marked_is_not_fooled_by_a_magicmock(self):
        """``MagicMock`` fabricates every attribute, so a bare ``getattr``
        marker check answers True for a hook that has been replaced by a mock —
        the most common way a hook gets displaced in this suite.
        tools/browser_guard.py already defends against exactly this."""
        from unittest import mock
        self.assertFalse(live_data_guard._marked(mock.MagicMock()))
        self.assertFalse(live_data_guard._marked(mock.Mock()))

    def test_unarmed_hooks_sees_a_mocked_out_delete_hook(self):
        from unittest import mock
        saved = os.remove
        try:
            os.remove = mock.MagicMock()
            self.assertIn("os.remove", live_data_guard.unarmed_hooks())
        finally:
            os.remove = saved


class ShellCommandLineTests(unittest.TestCase):
    """A boot script hidden INSIDE one argv element.

    ``core/actions._act_upgrade`` spawns ``["powershell", "-Command", "...;
    python 'C:\\\\JARVIS\\\\upgrade_jarvis.py' --relaunch"]``. The old matcher
    took ``os.path.basename`` of each whole element, so the basename of that
    blob was ``upgrade_jarvis.py' --relaunch`` and the guard returned None —
    while its docstring advertised ``upgrade_jarvis.py`` as covered."""

    def test_powershell_command_string_is_tokenised(self):
        ps_cmd = ("$env:ANTHROPIC_API_KEY=''; cd 'C:\\JARVIS'; "
                  "Write-Host '=== JARVIS UPGRADE PIPELINE ===' "
                  "-ForegroundColor Cyan; "
                  "python 'C:\\JARVIS\\upgrade_jarvis.py' --relaunch")
        self.assertEqual(
            live_data_guard._boot_script_in(["powershell", "-Command", ps_cmd]),
            "upgrade_jarvis.py")

    def test_cmd_slash_c_command_line_is_tokenised(self):
        self.assertEqual(
            live_data_guard._boot_script_in(
                ["cmd", "/c", "python C:\\JARVIS\\bobert_companion.py --flag"]),
            "bobert_companion.py")

    def test_powershell_call_operator_with_a_quoted_path(self):
        self.assertEqual(
            live_data_guard._boot_script_in(
                ["powershell", "-Command", "& 'C:\\JARVIS\\_boot_jarvis.ps1'"]),
            "_boot_jarvis.ps1")

    def test_bash_dash_c_command_line_is_tokenised(self):
        self.assertEqual(
            live_data_guard._boot_script_in(
                ["bash", "-c", "python3 /opt/jarvis/bobert_companion.py &"]),
            "bobert_companion.py")

    def test_a_bare_string_command_line_is_tokenised(self):
        self.assertEqual(
            live_data_guard._boot_script_in(
                'powershell -File "C:\\JARVIS\\_boot_jarvis.ps1" -Headless'),
            "_boot_jarvis.ps1")

    def test_the_real_act_upgrade_spawn_shape_is_blocked(self):
        """End to end, through the installed hook — this is the spawn that
        opens a real PowerShell window, kills JARVIS and deletes the flag.

        AIMED AT A TEMPDIR COPY OF THE NAME, like every other spawn test in
        this file, so a regression launches NOTHING REAL. That is not a style
        rule: an earlier draft of this test used the literal C:\\JARVIS path and,
        run against the unfixed matcher it was written to expose, spawned a real
        `powershell -Command ... upgrade_jarvis.py --relaunch`. A test that
        proves a guard is missing must never be the thing that walks through the
        hole."""
        if not live_data_guard._INSTALLED:
            self.skipTest("live-data guard not installed")
        fake_root = tempfile.gettempdir()
        ps_cmd = (f"$env:ANTHROPIC_API_KEY=''; cd '{fake_root}'; "
                  f"python '{os.path.join(fake_root, 'upgrade_jarvis.py')}' "
                  f"--relaunch")
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            subprocess.Popen(["powershell", "-Command", ps_cmd])

    # ── the anti-false-positive half, which is what makes the above safe ─────
    def test_a_prompt_positional_argument_is_still_not_tokenised(self):
        """tests/test_multi_agent_pipeline.py hands the claude CLI multi-line
        PROMPTS that name the boot scripts. They arrive as plain positional
        arguments, never after a -c / -Command switch, so they have to stay
        invisible to the matcher."""
        prompt = ("You are the REVIEWER stage of the JARVIS multi-agent "
                  "upgrade pipeline. Do not edit upgrade_jarvis.py logic, and "
                  "never touch bobert_companion.py's boot path.")
        self.assertIsNone(live_data_guard._boot_script_in(
            ["claude", "--print", "--", prompt]))
        self.assertIsNone(live_data_guard._boot_script_in(
            ["claude", "--print", prompt]))

    def test_an_ordinary_python_dash_c_is_not_blocked(self):
        self.assertIsNone(live_data_guard._boot_script_in(
            [sys.executable, "-c", "print('ok')"]))


class SecondOrderLauncherTests(unittest.TestCase):
    """``_BOOT_SCRIPTS`` listed only the scripts that ARE a live JARVIS; the
    LAUNCHERS were missing. The guard cannot reach into a child process — which
    is the very argument the module docstring makes about why a runner-scoped
    redirect could never have stopped the incident.

    ``tools/multi_agent_pipeline._run_tester`` spawns
    ``[sys.executable, STABILITY_SCRIPT, "--wait", n]`` and
    ``tools/stability_smoke_test._launch_jarvis()`` then runs
    ``powershell -File _boot_jarvis.ps1 -Headless``: a real JARVIS two processes
    away, with no guard installed in either child."""

    _LAUNCHERS = ("stability_smoke_test.py", "multi_agent_pipeline.py",
                  "bounce_jarvis.py", "staging_integration.py",
                  "staging_inject_smoke.py", "blue_green_manager.py")

    def test_every_second_order_launcher_is_a_boot_script(self):
        missing = [n for n in self._LAUNCHERS
                   if n not in live_data_guard._BOOT_SCRIPTS]
        self.assertEqual(
            missing, [],
            "these scripts exist to LAUNCH JARVIS but the guard does not know "
            f"them, so spawning one boots a real JARVIS in a child: {missing}")

    def test_the_run_tester_spawn_shape_is_detected(self):
        script = os.path.join(_PROJECT_ROOT, "tools", "stability_smoke_test.py")
        self.assertEqual(
            live_data_guard._boot_script_in(
                [sys.executable, script, "--wait", "120"]),
            "stability_smoke_test.py")

    def test_every_listed_launcher_still_exists_on_disk(self):
        """Keeps the list honest: a name that no longer exists is dead weight
        that hides the fact the real launcher was renamed."""
        for name in self._LAUNCHERS:
            with self.subTest(script=name):
                self.assertTrue(
                    os.path.exists(os.path.join(_PROJECT_ROOT, name))
                    or os.path.exists(os.path.join(_PROJECT_ROOT, "tools", name)),
                    f"{name} is in _BOOT_SCRIPTS but is not in the tree")


class WriteInterceptionBreadthTests(unittest.TestCase):
    """``builtins.open`` was the ONLY write hook, while ``unarmed_hooks()``
    reported the flag-write path fully covered. ``pathlib.Path.write_text`` —
    the more idiomatic spelling, one refactor away — goes through ``io.open``,
    a separate module attribute, and ``os.open`` / ``os.truncate`` bypass
    both."""

    def setUp(self):
        if not live_data_guard._INSTALLED:
            self.skipTest("live-data guard not installed")

    def test_io_open_of_the_flag_is_blocked(self):
        import io as _io
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            _io.open(live_data_guard.CLEAN_SHUTDOWN_FLAG, "w").close()

    def test_pathlib_write_text_on_the_flag_is_blocked(self):
        import pathlib
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            pathlib.Path(live_data_guard.CLEAN_SHUTDOWN_FLAG).write_text("x")

    def test_pathlib_open_of_the_flag_is_blocked(self):
        import pathlib
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            pathlib.Path(live_data_guard.CLEAN_SHUTDOWN_FLAG).open("w").close()

    def test_os_open_for_writing_the_flag_is_blocked(self):
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            os.open(live_data_guard.CLEAN_SHUTDOWN_FLAG,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC)

    def test_os_truncate_of_the_flag_is_blocked(self):
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            os.truncate(live_data_guard.CLEAN_SHUTDOWN_FLAG, 0)

    def test_the_new_hooks_are_covered_by_the_ratchet(self):
        """Coverage, proved the only honest way: displace each hook and check
        that ``unarmed_hooks()`` NAMES it. Asserting a name is absent from the
        list is not evidence of coverage — a hook the ratchet has never heard of
        is absent too, which is precisely how ``io.open`` was reported as
        covered by ``builtins.open`` while being wide open."""
        import io as _io
        self.assertEqual(live_data_guard.unarmed_hooks(), [])
        for module, attr, name in ((_io, "open", "io.open"),
                                   (os, "open", "os.open"),
                                   (os, "truncate", "os.truncate")):
            saved = getattr(module, attr)
            try:
                setattr(module, attr, lambda *a, **k: None)
                with self.subTest(hook=name):
                    self.assertIn(name, live_data_guard.unarmed_hooks())
            finally:
                setattr(module, attr, saved)
        self.assertEqual(live_data_guard.unarmed_hooks(), [])

    # ── negative controls: ordinary test I/O must not notice the guard ───────
    def test_reading_is_never_blocked(self):
        import io as _io
        import pathlib
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            with _io.open(path, "r"):
                pass
            pathlib.Path(path).read_text()
            os.close(os.open(path, os.O_RDONLY))
        finally:
            os.remove(path)

    def test_ordinary_writes_outside_live_data_still_work(self):
        import pathlib
        p = pathlib.Path(tempfile.mkdtemp()) / "note.txt"
        p.write_text("hello")
        self.assertEqual(p.read_text(), "hello")
        os.close(os.open(str(p), os.O_WRONLY | os.O_TRUNC))
        os.truncate(str(p), 0)


class LiveStartfileTests(unittest.TestCase):
    """``os.startfile`` is the Windows shell-launch primitive: of
    ``_boot_jarvis.ps1`` it boots a REAL JARVIS with no ``subprocess.Popen``
    anywhere in the path. Only tools/browser_guard.py hooked it, and only as a
    side effect of being the BROWSER guard — so ``JARVIS_ALLOW_REAL_BROWSER=1``
    removed the last interception on that route while the live-data guard went
    on reporting ``is_armed() -> True``."""

    def setUp(self):
        if not live_data_guard._INSTALLED:
            self.skipTest("live-data guard not installed")
        if not hasattr(os, "startfile"):
            self.skipTest("os.startfile is Windows-only")

    # Targets that DO NOT EXIST, so a regression opens nothing on the owner's
    # desktop — the unguarded call fails at the shell with FileNotFoundError.
    _FAKE_BOOT = os.path.join(tempfile.gettempdir(), "_boot_jarvis.ps1")
    _FAKE_LIVE = os.path.join(live_data_guard.LIVE_DATA_DIR,
                              "__guard_probe_does_not_exist__")

    def test_the_wrapper_refuses_a_boot_script(self):
        """The hook itself, on a sentinel 'real' startfile that records instead
        of launching — so this is provable regardless of which guard is on top
        of ``os.startfile`` at the moment."""
        reached = []
        hook = live_data_guard._guard_startfile(lambda p, *a, **k: reached.append(p))
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            hook(self._FAKE_BOOT)
        with self.assertRaises(live_data_guard.LiveDataGuardError):
            hook(self._FAKE_LIVE)
        self.assertEqual(reached, [], "the shell was reached anyway")
        hook(os.path.join(tempfile.gettempdir(), "harmless.txt"))
        self.assertEqual(len(reached), 1, "an unrelated startfile was blocked")

    def test_startfile_is_covered_by_the_ratchet(self):
        """Displace it and check the ratchet NAMES it — the only honest proof
        that ``os.startfile`` is a hook the guard actually knows about."""
        self.assertNotIn("os.startfile", live_data_guard.unarmed_hooks())
        saved = os.startfile
        try:
            os.startfile = lambda *a, **k: None
            self.assertIn("os.startfile", live_data_guard.unarmed_hooks())
        finally:
            os.startfile = saved
        self.assertEqual(live_data_guard.unarmed_hooks(), [])

    def test_the_browser_escape_hatch_no_longer_opens_the_startfile_route(self):
        """THE FINDING, end to end. ``JARVIS_ALLOW_REAL_BROWSER=1`` installs no
        browser stub at all, and before this hook existed that removed the ONLY
        ``os.startfile`` interception in the run — a shell route to a live
        JARVIS that ``core/actions.py`` and ``tray.py`` both use, with
        ``live_data_guard.is_armed()`` still answering True."""
        from tools import browser_guard
        saved_ledger = browser_guard.blocked_attempts()
        try:
            browser_guard._reset_for_tests()
            browser_guard.install(
                quiet=True, env={browser_guard.ALLOW_ENV_VAR: "1"})
            self.assertFalse(browser_guard.is_installed(),
                             "the browser guard should be OFF for this probe")
            with self.assertRaises(live_data_guard.LiveDataGuardError):
                os.startfile(self._FAKE_BOOT)
            with self.assertRaises(live_data_guard.LiveDataGuardError):
                os.startfile(self._FAKE_LIVE)
            self.assertEqual(live_data_guard.unarmed_hooks(), [])
        finally:
            browser_guard._reset_for_tests()
            browser_guard.install(quiet=True)
            browser_guard._blocked.extend(saved_ledger)
        self.assertTrue(browser_guard.is_armed(),
                        "this test left the browser guard disarmed")

    def test_the_browser_guards_reset_does_not_discard_our_hook(self):
        """The live-data hook has to survive the browser guard's un-wrap /
        re-wrap cycle. It does because it is installed UNDERNEATH — which is
        why tests/__init__.py arms the live-data guard FIRST."""
        from tools import browser_guard
        if not browser_guard.is_installed():
            self.skipTest("browser guard disabled via " + browser_guard.ALLOW_ENV_VAR)
        saved_ledger = browser_guard.blocked_attempts()
        try:
            browser_guard._reset_for_tests()
            browser_guard.install(quiet=True)
            self.assertEqual(live_data_guard.unarmed_hooks(), [])
        finally:
            browser_guard._reset_for_tests()
            browser_guard.install(quiet=True)
            browser_guard._blocked.extend(saved_ledger)


class BannerTests(unittest.TestCase):
    """The two siblings each emit exactly ONE loud line and both call that
    contract out explicitly (tools/mem_guard.py:35-37, tools/browser_guard.py
    :89-91). The live-data guard — the one protecting the artefact the owner
    actually lost — printed NOTHING, so a run with ``JARVIS_ALLOW_LIVE_DATA=1``
    was byte-identical in the log to a fully protected one: two armed banners
    and silence about live data."""

    def test_banner_states_the_armed_directory(self):
        text = live_data_guard.banner()
        self.assertTrue(text.startswith("[live-data-guard]"), text)
        if live_data_guard._INSTALLED:
            self.assertIn(live_data_guard.LIVE_DATA_DIR, text)
            self.assertIn("armed", text)

    def test_the_escape_hatch_banner_is_loud(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {live_data_guard._ENV_ESCAPE: "1"}):
            text = live_data_guard.banner()
        self.assertIn("DISABLED", text)
        self.assertIn(live_data_guard._ENV_ESCAPE, text)
        self.assertIn("data/", text)

    def test_the_chokepoint_prints_the_banner(self):
        """STALE-DUPLICATE GUARD: the banner is worthless the moment
        tests/__init__.py stops printing it."""
        src = _source(_TESTS_INIT)
        tree = ast.parse(src)
        printed = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == "print"
                   and any(isinstance(sub, ast.Call)
                           and isinstance(sub.func, ast.Attribute)
                           and sub.func.attr == "banner"
                           and isinstance(sub.func.value, ast.Name)
                           and sub.func.value.id.lstrip("_").startswith(
                               "live_data_guard")
                           for sub in ast.walk(n))]
        self.assertTrue(
            printed,
            "tests/__init__.py must print(live_data_guard.banner()) — without "
            "it a run with JARVIS_ALLOW_LIVE_DATA=1 is indistinguishable in "
            "the log from a fully protected one")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
