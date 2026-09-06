#!/usr/bin/env python3
"""Headless test runner for the JARVIS unittest suite.

Zero external deps (stdlib unittest only) so it runs in the App-Control-locked
environment and inside the self-upgrade pipeline without installing anything.

    python tools/run_tests.py            # run everything in tests/
    python tools/run_tests.py -v         # verbose
    python tools/run_tests.py atomic_io  # run only tests/test_atomic_io.py

Exit code 0 = all passed, 1 = failures/errors — so the pipeline can gate on it.

Every run sits inside a HARD MEMORY CEILING (tools/mem_guard.py, applied as the
first thing main() does, so it covers every import and every child process).
That is the RAM sibling of the _redirect_settings_to_throwaway() /
_redirect_data_dir_to_throwaway() guards below: a test run must not be able to
damage the box it runs on. See tools/mem_guard.py for the 2026-08-20 bugcheck
that motivated it and the JARVIS_TEST_MEM_CAP_GB knob (0 = off).
"""
from __future__ import annotations
import os
import shutil
import sys
import tempfile
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS_DIR = os.path.join(_PROJECT_ROOT, "tests")


def _redirect_settings_to_throwaway() -> None:
    """Point the WHOLE suite's settings reads/writes at a throwaway file so a
    test that exercises the real ``tools.settings_window.save_settings`` can
    NEVER clobber the owner's live ``data/user_settings.json``.

    Sets ``JARVIS_SETTINGS_PATH`` (honoured by settings_window.settings_path())
    to a file in a fresh temp dir BEFORE any test is imported. Seeds it with a
    copy of the real file when present, so tests that ``load_settings`` still
    see realistic data; if absent, load_settings just returns defaults. Respects
    an externally-set override (does nothing if already set)."""
    if (os.environ.get("JARVIS_SETTINGS_PATH") or "").strip():
        return
    throwaway_dir = tempfile.mkdtemp(prefix="jarvis_test_settings_")
    throwaway = os.path.join(throwaway_dir, "test_user_settings.json")
    real = os.path.join(_PROJECT_ROOT, "data", "user_settings.json")
    if os.path.exists(real):
        try:
            shutil.copyfile(real, throwaway)
        except OSError:
            pass
    os.environ["JARVIS_SETTINGS_PATH"] = throwaway


def _redirect_data_dir_to_throwaway() -> None:
    """Point core.paths.data_dir() at a throwaway directory so ANY test that
    forgets to redirect a module's file-path globals still cannot write the
    owner's live ``data/``.

    Companion to _redirect_settings_to_throwaway — same incident class: a
    targeted run of tests/skills/test_sh_ecobee.py once overwrote the LIVE
    ``data/sh_ecobee_tokens.json`` with fake fixture tokens because an
    ineffective builtins.open mock let the real atomic write through
    (2026-07-21). Sets ``JARVIS_DATA_DIR`` (highest precedence in
    core.paths.data_dir()) BEFORE any test is imported; respects an
    externally-set override (does nothing if already set)."""
    if (os.environ.get("JARVIS_DATA_DIR") or "").strip():
        return
    os.environ["JARVIS_DATA_DIR"] = tempfile.mkdtemp(prefix="jarvis_test_data_")


def _redirect_lock_dir_to_throwaway() -> None:
    """Point the singleton lock files at a throwaway directory so no test — and
    no subprocess a test spawns — can write the owner's live ``jarvis.lock``.

    Third sibling of _redirect_settings_to_throwaway /
    _redirect_data_dir_to_throwaway, and the same incident class: jarvis.lock is
    live runtime state. It names the PID of the running JARVIS, and both
    tools/stability_smoke_test.py and the upgrade pipeline read it to decide
    which process that is. On 2026-09-05 a tools/run_tests_ci_sim.py run stamped
    its OWN pid over the live one — bobert_companion._early_boot_singleton_lock
    ran on a plain `import bobert_companion` and its OS-lock helper fail-opened
    under the runner's faked ``sys.platform`` — so tools/audit_codebase.py then
    refused to start against a PID that had already exited. Sets
    ``JARVIS_LOCK_DIR`` (honoured by _early_boot_singleton_lock) BEFORE any test
    is imported, so every subprocess inherits it too; respects an externally-set
    override (does nothing if already set)."""
    if (os.environ.get("JARVIS_LOCK_DIR") or "").strip():
        return
    os.environ["JARVIS_LOCK_DIR"] = tempfile.mkdtemp(prefix="jarvis_test_lock_")


def _stop_lingering_daemons() -> None:
    """Best-effort: stop any opt-in background daemon a test may have left alive
    (currently the apple-music keep-alive watchdog) so it can't outlive the
    suite. Never raises; a never-imported module is a no-op."""
    try:
        mod = sys.modules.get("audio.apple_music_keeper")
        if mod is not None and hasattr(mod, "stop_keeper"):
            mod.stop_keeper(timeout=5.0)
    except Exception:
        pass


def main(argv: list[str]) -> int:
    # HARD MEMORY CEILING FIRST — before any test is imported, before anything
    # allocates. Same family as the two redirect guards below (a test run must
    # not be able to damage the box); this one is the RAM half, added after an
    # uncapped run committed 144 GB and bugchecked the machine on 2026-08-20.
    # See tools/mem_guard.py. The sys.path insert has to precede it so
    # ``tools.mem_guard`` is importable when this file is run as a script.
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from tools.mem_guard import apply_memory_ceiling
    apply_memory_ceiling()
    # THEN THE LIVE-DATA GUARD, AND ONLY THEN THE BROWSER GUARD (no live tabs in
    # the owner's Chrome). The ORDER is a contract, stated in full in
    # tests/__init__.py: both wrap os.startfile, and the one armed FIRST is the
    # one the other's un-wrap snapshot restores instead of discards. That rule
    # lived ONLY there, so this runner armed the browser guard with no live-data
    # guard in the process yet, and the first _reset_for_tests() in
    # tests/test_browser_guard.py wiped the live-data hook off os.startfile for
    # the rest of the run (full CI, 2026-08-20). The tests package IS that
    import tests  # chokepoint: it arms all three, in order, idempotently
    from tools import browser_guard
    browser_guard.install()
    # THEN THE TIME CEILING — the sibling of the memory ceiling above. mem_guard
    # bounds what a run may ALLOCATE; this bounds how long it may STALL, which
    # a memory ceiling structurally cannot see (a hang is not an allocation).
    # Armed before discovery so an import-time hang is named too. See
    # tools/test_watchdog.py (JARVIS_TEST_TIMEOUT_S=0 turns it off).
    from tools import test_watchdog
    test_watchdog.arm()
    # Redirect settings I/O to a throwaway copy BEFORE any test is imported, so
    # a leaked real save_settings can't touch data/user_settings.json.
    _redirect_settings_to_throwaway()
    # Same guard for the whole data/ directory: a forgotten per-test path
    # redirect must land in a throwaway, never the live runtime state.
    _redirect_data_dir_to_throwaway()
    # Third guard, same family: jarvis.lock is live runtime state (it names the
    # running JARVIS's PID), so a test run must not be able to write it either.
    _redirect_lock_dir_to_throwaway()
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    verbose = "-v" in argv or "--verbose" in argv
    selectors = [a for a in argv if not a.startswith("-")]

    loader = unittest.TestLoader()
    if selectors:
        suite = unittest.TestSuite()
        for sel in selectors:
            name = sel if sel.startswith("test_") else f"test_{sel}"
            name = name[:-3] if name.endswith(".py") else name
            suite.addTests(loader.loadTestsFromName(f"tests.{name}"))
    else:
        suite = loader.discover(start_dir=_TESTS_DIR, pattern="test_*.py",
                                top_level_dir=_PROJECT_ROOT)

    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        resultclass=test_watchdog.WatchdogTextTestResult)
    result = runner.run(suite)
    # Suite over: stand the watchdog down before the reap/summary below.
    test_watchdog.disarm()

    # Belt-and-suspenders: reap any opt-in background daemon a test left running
    # (e.g. the apple-music keep-alive watchdog, a non-terminating loop) so it
    # can't outlive the suite. Per-test cleanup is the real guard; this never
    # fails the run and is a no-op if the module was never imported.
    _stop_lingering_daemons()

    total = result.testsRun
    fails = len(result.failures)
    errs = len(result.errors)
    skipped = len(result.skipped)
    print(f"\n=== JARVIS TESTS: {total} run, {fails} failed, "
          f"{errs} errored, {skipped} skipped ===")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main(sys.argv[1:]))
