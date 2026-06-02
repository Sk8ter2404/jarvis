#!/usr/bin/env python3
"""Headless test runner for the JARVIS unittest suite.

Zero external deps (stdlib unittest only) so it runs in the App-Control-locked
environment and inside the self-upgrade pipeline without installing anything.

    python tools/run_tests.py            # run everything in tests/
    python tools/run_tests.py -v         # verbose
    python tools/run_tests.py atomic_io  # run only tests/test_atomic_io.py

Exit code 0 = all passed, 1 = failures/errors — so the pipeline can gate on it.
"""
from __future__ import annotations
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS_DIR = os.path.join(_PROJECT_ROOT, "tests")


def main(argv: list[str]) -> int:
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

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    total = result.testsRun
    fails = len(result.failures)
    errs = len(result.errors)
    skipped = len(result.skipped)
    print(f"\n=== JARVIS TESTS: {total} run, {fails} failed, "
          f"{errs} errored, {skipped} skipped ===")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main(sys.argv[1:]))
