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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
