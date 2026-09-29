"""tools/run_tests_ci_sim.py: the --coverage floor reader and the leaked-thread
monolith-import tripwire (both 2026-09-29, after v2.0.128's CI coverage crash).

--coverage reads CI's coverage floor out of .github/workflows/ci.yml instead of
keeping a second copy of the number.

``--coverage`` makes the local simulation ALSO be CI's coverage step (the same
suite under coverage.py, tools/run_coverage.py's measured surface via
run_coverage._coverage_for, the floor from ci.yml) under the light-tier
dependency set: the one combination that shows the percentage CI actually
enforces. These tests pin the floor reader against the REAL workflow file, so a
reworded coverage step fails here instead of silently disabling the check.

_OffMainMonolithImportTripwire fails a ci-sim run when any thread but the main
one freshly imports bobert_companion — the deterministic precondition of the
v2.0.128 race (a leaked poller re-running the always-failing monolith import).
Driven here through find_spec() directly: nothing is really imported.
stdlib unittest only; nothing is run.
"""
from __future__ import annotations

import os
import contextlib
import io
import shutil
import sys
import tempfile
import threading
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools import run_tests_ci_sim as cs  # noqa: E402


def _root_with_workflow(case: unittest.TestCase, text: str | None) -> str:
    root = tempfile.mkdtemp(prefix="cisim_floor_")
    case.addCleanup(shutil.rmtree, root, True)
    if text is not None:
        wf = os.path.join(root, ".github", "workflows")
        os.makedirs(wf)
        with open(os.path.join(wf, "ci.yml"), "w", encoding="utf-8") as f:
            f.write(text)
    return root


class CiCoverageFloorTests(unittest.TestCase):

    def test_reads_the_floor_ci_actually_enforces(self):
        floor = cs._ci_coverage_floor(_ROOT)
        self.assertIsNotNone(floor, "ci.yml's coverage step no longer carries "
                                    "`run_coverage.py ... --fail-under N`")
        with open(os.path.join(_ROOT, ".github", "workflows", "ci.yml"),
                  encoding="utf-8") as f:
            steps = [ln for ln in f if "run_coverage.py" in ln
                     and "--fail-under" in ln]
        self.assertEqual(len(steps), 1, steps)
        self.assertIn(f"--fail-under {floor:g}", steps[0])

    def test_no_workflow_means_no_floor(self):
        self.assertIsNone(cs._ci_coverage_floor(_root_with_workflow(self, None)))

    def test_a_workflow_without_the_flag_means_no_floor(self):
        root = _root_with_workflow(
            self,
            "# run coverage locally: tools/run_coverage.py --full\n"
            "      - run: python tools/run_coverage.py --xml\n")
        self.assertIsNone(cs._ci_coverage_floor(root))

    def test_the_flag_must_be_on_the_run_coverage_line(self):
        """A --fail-under on some OTHER tool's line is not CI's floor."""
        root = _root_with_workflow(
            self,
            "      - run: python tools/run_coverage.py --xml\n"
            "      - run: python tools/other_gate.py --fail-under 12\n")
        self.assertIsNone(cs._ci_coverage_floor(root))

    def test_decimal_floor(self):
        root = _root_with_workflow(
            self,
            "        run: python tools/run_coverage.py --xml --fail-under 81.5\n")
        self.assertEqual(cs._ci_coverage_floor(root), 81.5)


class OffMainMonolithImportTripwireTests(unittest.TestCase):

    def _from_worker(self, fn, name="leaked-poller"):
        out = []
        th = threading.Thread(target=lambda: out.append(fn()), name=name)
        th.start()
        th.join(timeout=10)
        self.assertFalse(th.is_alive())
        return out[0]

    def test_a_main_thread_import_is_not_a_leak(self):
        tw = cs._OffMainMonolithImportTripwire()
        self.assertIsNone(tw.find_spec("bobert_companion"))
        self.assertEqual(tw.hits, {})
        self.assertTrue(tw.report())

    def test_a_worker_thread_import_is_caught_and_never_blocked(self):
        tw = cs._OffMainMonolithImportTripwire()
        spec = self._from_worker(lambda: tw.find_spec("bobert_companion"))
        self.assertIsNone(spec, "the tripwire must never answer an import")
        self.assertEqual([v[0] for v in tw.hits.values()], ["leaked-poller"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertFalse(tw.report())
        self.assertIn("leaked-poller", buf.getvalue())
        self.assertIn("test_run_tests_ci_sim.py", buf.getvalue(),
                      "the report must show the code the thread was running")

    def test_repeats_from_one_site_are_counted_not_listed(self):
        tw = cs._OffMainMonolithImportTripwire()

        def twice():
            for _ in range(2):
                tw.find_spec("bobert_companion")

        self._from_worker(twice)
        self.assertEqual([v[1] for v in tw.hits.values()], [2])

    def test_other_modules_are_ignored(self):
        tw = cs._OffMainMonolithImportTripwire()
        self._from_worker(lambda: tw.find_spec("core.config"))
        self._from_worker(lambda: tw.find_spec("bobert_companion_extra"))
        self.assertEqual(tw.hits, {})

    def test_the_runner_installs_it_around_discovery_and_the_run(self):
        with open(os.path.join(_ROOT, "tools", "run_tests_ci_sim.py"),
                  encoding="utf-8") as f:
            src = f.read()
        body = src[src.index("def main("):]
        self.assertLess(body.index("sys.meta_path.insert(0, tripwire)"),
                        body.index(".discover("))
        self.assertIn("tripwire.report()", body)
        self.assertIn("and trip_ok", body)


if __name__ == "__main__":
    unittest.main()
