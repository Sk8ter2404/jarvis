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
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=_TESTS, pattern="test_*.py",
                            top_level_dir=_ROOT)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return result.wasSuccessful()


def main(argv: list[str]) -> int:
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
             "tools/test_local_prompt.py"]
    if want_full:
        # LOCAL full tier: adds the ~14K-line monolith + the other root product
        # modules + the smaller product packages. Needs all deps present (these
        # can't import on the bare CI runner), so this tier is local-only — the
        # default source below is the CI light-tier gate.
        _source = ["core", "skills", "tools", "adapters", "hud", "audio",
                   "bobert_companion", "tray", "boot_sequence", "upgrade_jarvis"]
        _label = "FULL local tier: core/skills/tools + monolith + root modules"
    else:
        _source = ["core", "skills", "tools"]
        _label = "core/ + skills/ + tools/"
    cov = coverage.Coverage(source=_source, omit=_omit, branch=False)
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
