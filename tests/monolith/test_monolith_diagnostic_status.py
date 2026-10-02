"""Monolith half of '"diagnostic status" reports a FRESH self-check'
(owner's decision, 2026-10-02).

bobert_companion._fresh_diagnostic_status is what the boot registers as
ACTIONS["diagnostic_status"] (pinned by AST in
tests/test_diagnostic_status_fresh.py). It must run the SAME sweep "are you
ok" runs — the first registered of _SELF_CHECK_ACTIONS — and hand it to
core.diagnostic_daemons.fresh_diagnostic_status, whose fallback reads the
daemon counters as possibly stale. Generic fixtures only; the daemon state
file is never read (the counters are patched).

    python -m unittest tests.monolith.test_monolith_diagnostic_status
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

SUMMARY = "All systems nominal, sir. (2.1s sweep, 20 probes.)"
COUNTERS = "Last self-diagnostic at 2026-06-01T00:00:00, 7 runs total."


@requires_monolith
class FreshDiagnosticStatusTests(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        from core import diagnostic_daemons as dd
        self.dd = dd
        for name, kwargs in (("diagnostic_daemon_status_spoken",
                              {"return_value": COUNTERS}),
                             ("_read_state", {"return_value": {"paused": False}})):
            patcher = mock.patch.object(dd, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Start from NO self-check registered; each test adds what it needs.
        patcher = mock.patch.dict(self.bc.ACTIONS, clear=False)
        acts = patcher.start()
        self.addCleanup(patcher.stop)
        for name in self.bc._SELF_CHECK_ACTIONS:
            acts.pop(name, None)
        self.acts = acts

    def test_runs_the_are_you_ok_sweep_and_returns_its_summary(self):
        sweep = mock.Mock(return_value=SUMMARY)
        self.acts["are_you_ok"] = sweep
        self.assertEqual(self.bc._fresh_diagnostic_status(""), SUMMARY)
        sweep.assert_called_once_with("")

    def test_prefers_the_names_in_the_self_check_order(self):
        first = mock.Mock(return_value="first")
        later = mock.Mock(return_value="later")
        self.acts["system_check"] = later
        self.acts["run_diagnostic"] = first
        self.assertEqual(self.bc._fresh_diagnostic_status(""), "first")
        later.assert_not_called()

    def test_no_self_check_loaded_reports_the_counters_as_possibly_stale(self):
        out = self.bc._fresh_diagnostic_status("")
        self.assertIn("may be stale", out)
        self.assertTrue(out.endswith(COUNTERS), out)

    def test_a_raising_sweep_reports_the_counters_as_possibly_stale(self):
        self.acts["are_you_ok"] = mock.Mock(side_effect=RuntimeError("boom"))
        out = self.bc._fresh_diagnostic_status("")
        self.assertIn("did not finish", out)
        self.assertIn("may be stale", out)
        self.assertTrue(out.endswith(COUNTERS), out)
        self.assertNotIn("nominal", out.lower())

    def test_never_the_old_counters_when_the_sweep_answers(self):
        self.acts["are_you_ok"] = mock.Mock(return_value=SUMMARY)
        self.dd.diagnostic_daemon_status_spoken.reset_mock()
        self.bc._fresh_diagnostic_status("")
        self.dd.diagnostic_daemon_status_spoken.assert_not_called()

    def test_the_reply_is_spoken_verbatim(self):
        # The fresh summary is a finished sentence for the owner, read as-is.
        self.assertIn("diagnostic_status", self.bc.SPEAK_RESULT_VERBATIM_ACTIONS)


if __name__ == "__main__":
    unittest.main()
