#!/usr/bin/env python3
"""Run the JARVIS test suite under coverage.py and report.

    python tools/run_coverage.py                  # run + terminal report
    python tools/run_coverage.py --xml            # also write coverage.xml (CI)
    python tools/run_coverage.py --fail-under 30  # exit 1 if total < 30%
    python tools/run_coverage.py --missing        # show uncovered line ranges

Measures ``core/`` + ``skills/`` + ``tools/`` — the unit-testable surface. The
~14K-line monolith (``bobert_companion.py``, at the repo root) and the heavy
GPU/audio modules that can't import on a bare runner are out of the measured
set — they're covered behaviourally by the staging integration tier — so the
percentage reflects real unit coverage of testable code, and we ratchet it up.

coverage runs in-process via its API (App-Control-safe, no ``.exe``). Install
with ``python -m pip install --user coverage`` if missing.

Every run sits inside a HARD MEMORY CEILING (tools/mem_guard.py, applied as the
first thing main() does, so it covers coverage's own start-up, discovery, every
test import and every child process). This is the THIRD discovery runner — CI
itself invokes it (``python tools/run_coverage.py --xml --fail-under 80``) — so
it carries the identical guard to tools/run_tests.py and
tools/run_tests_ci_sim.py: a test run must not be able to damage the box it runs
on. See tools/mem_guard.py for the 2026-08-20 bugcheck that motivated it and the
JARVIS_TEST_MEM_CAP_GB knob (0 = off).
"""
from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS = os.path.join(_ROOT, "tests")


def _run_suite() -> bool:
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    # Same live-box guards as tools/run_tests.py (imported, not copied — one
    # implementation, no stale duplicate): point settings AND the whole data/
    # dir at throwaways BEFORE any test is imported, so a forgotten per-test
    # path redirect can't clobber the owner's live runtime state (the ecobee
    # token-file incident, 2026-07-21). Both respect external overrides.
    from tools.run_tests import (_redirect_data_dir_to_throwaway,
                                 _redirect_settings_to_throwaway)
    _redirect_settings_to_throwaway()
    _redirect_data_dir_to_throwaway()
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=_TESTS, pattern="test_*.py",
                            top_level_dir=_ROOT)
    from tools import test_watchdog
    result = unittest.TextTestRunner(
        verbosity=1,
        resultclass=test_watchdog.WatchdogTextTestResult).run(suite)
    # Suite over: stand the watchdog down before coverage reports (a slow
    # report is not a hung test).
    test_watchdog.disarm()
    return result.wasSuccessful()


def main(argv: list[str]) -> int:
    # HARD MEMORY CEILING FIRST — before coverage boots, before discovery, and
    # before any test is imported, so it covers everything this run touches and
    # every process it spawns. Same family as the _redirect_settings_to_throwaway()
    # / _redirect_data_dir_to_throwaway() guards in _run_suite() (a test run must
    # not be able to damage the box); this one is the RAM half, added after an
    # uncapped run committed 144 GB and bugchecked the machine on 2026-08-20.
    # See tools/mem_guard.py. The sys.path insert has to precede it so
    # ``tools.mem_guard`` is importable when this file is run as a script.
    #
    # KNOWN, MEASURED SIDE EFFECT (2026-08-20): importing the guard here runs
    # tools/mem_guard.py's module body BEFORE cov.start(), so coverage scores
    # THAT ONE FILE lower — 67.0% -> 48.5%, ~19 of its 103 statements — with its
    # tests unchanged. Against the 52,885-statement measured denominator that is
    # 0.04 pts on the TOTAL the --fail-under gate reads, so the gate is
    # unaffected and the trade (a bounded run) is obviously worth it. Do NOT
    # "fix" the number by popping tools.mem_guard out of sys.modules to force a
    # re-import under measurement: the applied ceiling would then live in one
    # module object while the suite imports a second, fresh one — this repo's #1
    # bug class (the stale duplicate) wired straight into the safety guard.
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
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
    # THEN THE TIME CEILING (tools/test_watchdog.py) — mem_guard bounds what a
    # run may ALLOCATE, this bounds how long it may STALL. Coverage instruments
    # every executed line, so this run is legitimately several times slower than
    # a bare one: the ceilings are raised 4x here rather than shared, which is
    # the whole reason arm() takes them as arguments.
    from tools import test_watchdog
    test_watchdog.arm(per_phase_s=test_watchdog.DEFAULT_PER_PHASE_S * 4,
                      total_s=test_watchdog.DEFAULT_TOTAL_S * 4)
    try:
        import coverage
    except ImportError:
        print("coverage not installed — run: python -m pip install --user coverage")
        return 2

    os.chdir(_ROOT)  # so the relative source dirs resolve regardless of CWD
    want_xml = "--xml" in argv
    want_missing = "--missing" in argv
    fail_under = None
    if "--fail-under" in argv:
        try:
            fail_under = float(argv[argv.index("--fail-under") + 1])
        except (IndexError, ValueError):
            print("--fail-under needs a number, e.g. --fail-under 30")
            return 2

    want_full = "--full" in argv
    _omit = ["*/tests/*", "*/__pycache__/*", "tools/run_coverage.py",
             "tools/run_tests_ci_sim.py",
             # gitignored personal skills — not shipped, absent on CI, so they
             # don't belong in the shipped-coverage denominator.
             "*/skills/vip_intercept.py", "*/skills/vip_boss_mode.py",
             "*/skills/trip_planner.py", "*/skills/teams_screener.py",
             # One-off operational / hardware-bench / scratch utilities — run by
             # hand, not part of the shipped runtime library surface, and several
             # need hardware (camera/LAN/PyQt) absent on CI. Excluded from the
             # measured denominator (the product code + the release/CI gates are
             # what we hold to coverage; these are dev conveniences):
             "tools/audit_local.py", "tools/pii_local.py",
             "tools/bounce_jarvis.py", "tools/face_detect_bench.py",
             "tools/generate_jarvis_icon.py", "tools/identify_vendors.py",
             "tools/render_unified_hud.py", "tools/say_to_jarvis.py",
             "tools/scan_full_network.py", "tools/scan_lan_devices.py",
             "tools/test_local_prompt.py",
             # Developer-facing TEMPLATE skill (copy-me example), not a real
             # shipped/registered skill — documentation by example, not logic.
             "*/skills/_example_skill.py"]
    if want_full:
        # LOCAL full tier: adds the ~14K-line monolith + the other root product
        # modules + the smaller product packages. Needs all deps present (these
        # can't import on the bare CI runner), so this tier is local-only — the
        # default source below is the CI light-tier gate.
        # NB: hud/ is intentionally NOT measured — it's the PyQt holographic-
        # overlay presentation layer (GUI paint/layout). It needs a live Qt
        # display, can't import on the bare runner, and GUI paint code is
        # conventionally excluded from UNIT coverage (it's exercised
        # behaviorally by the staging tier, like the monolith's boot path).
        # Everything with unit-testable logic IS measured.
        _source = ["core", "skills", "tools", "adapters", "audio",
                   "bobert_companion", "tray", "boot_sequence", "upgrade_jarvis"]
        _label = "FULL local tier: core/skills/tools + monolith + root modules"
    else:
        _source = ["core", "skills", "tools"]
        _label = "core/ + skills/ + tools/"
    cov = coverage.Coverage(source=_source, omit=_omit, branch=False)
    # Conventional never-unit-tested lines, excluded everywhere so the report
    # reflects *reachable* code. cov.exclude ADDS to coverage's default
    # "pragma: no cover" (it doesn't replace it). Substantive unreachable blocks
    # (the boot entrypoint, while-True daemon loops, live mic/camera capture)
    # carry their own inline ``# pragma: no cover - <reason>`` at the block head.
    for _pat in (r"if __name__ == ['\"]__main__['\"]:",
                 r"if (typing\.)?TYPE_CHECKING:",
                 r"raise NotImplementedError",
                 r"@(abc\.)?abstractmethod"):
        cov.exclude(_pat)
    cov.start()
    ok = _run_suite()
    cov.stop()
    cov.save()

    print("\n" + "=" * 72)
    total = cov.report(show_missing=want_missing, skip_covered=False,
                       file=sys.stdout)
    print("=" * 72)
    print(f"TOTAL coverage: {total:.1f}%  (measured: {_label})")
    if want_xml:
        cov.xml_report(outfile=os.path.join(_ROOT, "coverage.xml"))
        print("wrote coverage.xml")

    if not ok:
        print("RESULT: TESTS FAILED")
        return 1
    if fail_under is not None and total < fail_under:
        print(f"RESULT: FAIL — coverage {total:.1f}% < required {fail_under:.1f}%")
        return 1
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
