""""diagnostic status" reports a FRESH self-check (owner's decision, 2026-10-02).

Before: the boot re-registered ``diagnostic_status`` to the background
daemons' counters (core.diagnostic_daemons.act_diagnostic_status) — last-run
timestamps that are only as fresh as the last 5-minute sweep — and a comment
called that intentional. The owner chose: saying "diagnostic status" runs the
same real sweep "are you ok" runs since v2.0.159, with a FAST fallback to the
counters (said to be possibly stale) when the sweep does not finish.

Light tier (stdlib only, no monolith import):

  * core.diagnostic_daemons.fresh_diagnostic_status — the sweep's own summary
    is the answer; no sweep / a raising sweep / an empty result falls back to
    the counters with a "may be stale" lead; nothing raises; every reply is
    marker-free so the verbatim speak path reads it as written;
  * the monolith's boot wiring, by AST (the boot block cannot run here):
    ``ACTIONS["diagnostic_status"]`` is the fresh handler, the counters stay on
    ``diagnostic_daemon_status``;
  * routing: the SELF DIAGNOSTIC section teaches 'diagnostic status' ->
    diagnostic_status and the slim local prompt keeps it.

tests/monolith/test_monolith_diagnostic_status.py drives the monolith handler.

    python -m unittest tests.test_diagnostic_status_fresh
"""
from __future__ import annotations

import ast
import json
import os
import unittest
from unittest import mock

from core import diagnostic_daemons as dd
from core import prompt_router as pr
from core import prompts
from core.failure_markers import FAILURE_MARKERS
from tests import test_diagnostic_daemons as tdd

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MONOLITH = os.path.join(_PROJECT_DIR, "bobert_companion.py")

SUMMARY = ("Sir, nothing is reporting a failure, but I am not able to call the "
           "system nominal: 1 check did not run, so it is UNVERIFIED — "
           "microphone. (2.1s sweep, 20 probes.)")


def _marker_hits(text: str) -> list:
    low = text.lower()
    return [m for m in FAILURE_MARKERS if m in low]


class FreshDiagnosticStatusTests(tdd._Base):
    def _counters(self, **state):
        base = {"paused": False,
                "self_diag": {"last_run_iso": "2026-06-01T00:00:00", "runs": 7},
                "crash_watch": {"detections": 1, "last_poll_ts": 1000.0},
                "deep_audit": {"last_run_iso": "2026-06-01T01:00:00",
                               "runs": 1, "pending_findings": 3},
                "anomaly_watch": {"last_poll_iso": "2026-06-01T02:00:00",
                                  "detections": 5}}
        base.update(state)
        with open(dd.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(base, f)

    def test_the_fresh_sweep_is_the_answer(self):
        self._counters()
        sweep = mock.Mock(return_value="  " + SUMMARY + "\n")
        out = dd.fresh_diagnostic_status(sweep)
        self.assertEqual(out, SUMMARY)
        sweep.assert_called_once_with("")
        # Not the counters: they are the fallback only.
        self.assertNotIn("runs total", out)

    def test_a_raising_sweep_falls_back_to_the_counters_as_possibly_stale(self):
        self._counters()
        sweep = mock.Mock(side_effect=RuntimeError("probe table missing"))
        out = dd.fresh_diagnostic_status(sweep)
        sweep.assert_called_once_with("")
        self.assertIn("did not finish", out)
        self.assertIn("may be stale", out)
        self.assertIn("7 runs total", out)        # the real counters
        self.assertIn("3 pending findings", out)
        self.assertNotIn("nominal", out.lower())

    def test_an_empty_or_non_string_result_falls_back(self):
        self._counters()
        for result in ("", "   ", None, 5, {"ok": True}):
            with self.subTest(result=result):
                out = dd.fresh_diagnostic_status(mock.Mock(return_value=result))
                self.assertIn("may be stale", out)
                self.assertIn("7 runs total", out)

    def test_no_self_check_loaded_falls_back(self):
        self._counters()
        for sweep in (None, "run_diagnostic", 42):
            with self.subTest(sweep=sweep):
                out = dd.fresh_diagnostic_status(sweep)
                self.assertIn("not loaded", out)
                self.assertIn("may be stale", out)
                self.assertIn("7 runs total", out)

    def test_the_fallback_keeps_the_paused_notice(self):
        self._counters(paused=True)
        out = dd.fresh_diagnostic_status(None)
        self.assertIn("Diagnostics are paused", out)

    def test_a_fresh_answer_still_says_when_the_daemons_are_paused(self):
        # The counters reply led with "Diagnostics are paused"; the fresh one
        # must not lose that.
        self._counters(paused=True)
        out = dd.fresh_diagnostic_status(mock.Mock(return_value=SUMMARY))
        self.assertTrue(out.startswith(SUMMARY), out)
        self.assertIn("paused", out[len(SUMMARY):])
        self._counters(paused=False)
        self.assertEqual(
            dd.fresh_diagnostic_status(mock.Mock(return_value=SUMMARY)), SUMMARY)

    def test_unreadable_counters_still_answer_and_never_raise(self):
        with mock.patch.object(dd, "diagnostic_daemon_status_spoken",
                               side_effect=OSError("state file locked")):
            out = dd.fresh_diagnostic_status(
                mock.Mock(side_effect=RuntimeError("boom")))
        self.assertIn("did not finish", out)
        self.assertIn("no recorded counters", out)
        with mock.patch.object(dd, "_read_state",
                               side_effect=ValueError("garbage")):
            self.assertEqual(
                dd.fresh_diagnostic_status(mock.Mock(return_value=SUMMARY)),
                SUMMARY)

    def test_every_own_sentence_is_marker_free(self):
        # diagnostic_status is in SPEAK_RESULT_VERBATIM_ACTIONS: a failure
        # marker would drop the reply off the verbatim path into an LLM
        # re-wording. Only the counters are appended, and they are the
        # existing marker-free daemon summary.
        self._counters(paused=True)
        replies = [
            dd.fresh_diagnostic_status(None),
            dd.fresh_diagnostic_status(mock.Mock(side_effect=RuntimeError())),
            dd.fresh_diagnostic_status(mock.Mock(return_value="")),
            dd.fresh_diagnostic_status(mock.Mock(return_value=SUMMARY)),
        ]
        with mock.patch.object(dd, "diagnostic_daemon_status_spoken",
                               return_value=""):
            replies.append(dd.fresh_diagnostic_status(None))
            replies.append(dd.fresh_diagnostic_status(
                mock.Mock(side_effect=RuntimeError())))
        for out in replies:
            with self.subTest(out=out):
                self.assertTrue(out.strip())
                self.assertEqual(_marker_hits(out), [])

    def test_the_counters_action_is_unchanged(self):
        # "diagnostic daemon status" still reads the counters back, no sweep.
        self._counters()
        self.assertIn("7 runs total", dd.act_diagnostic_status(""))


def _monolith_tree() -> ast.Module:
    with open(_MONOLITH, "r", encoding="utf-8") as f:
        return ast.parse(f.read(), filename=_MONOLITH)


def _actions_assignments(tree: ast.Module, key: str) -> list:
    """The value nodes of every ``ACTIONS["<key>"] = <value>`` in the tree."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "ACTIONS"
                    and isinstance(tgt.slice, ast.Constant)
                    and tgt.slice.value == key):
                out.append(node.value)
    return out


class MonolithWiringTests(unittest.TestCase):
    """The boot block that registers the daemon actions runs only in a real
    boot, so its wiring is pinned from the AST (the shortcut and the handler
    themselves are exercised in tests/monolith)."""

    @classmethod
    def setUpClass(cls):
        cls.tree = _monolith_tree()

    @classmethod
    def tearDownClass(cls):
        cls.tree = None

    def test_diagnostic_status_is_the_fresh_handler(self):
        values = _actions_assignments(self.tree, "diagnostic_status")
        names = [ast.unparse(v) for v in values]
        self.assertEqual(names, ["_fresh_diagnostic_status"], names)

    def test_the_counters_stay_on_diagnostic_daemon_status(self):
        values = _actions_assignments(self.tree, "diagnostic_daemon_status")
        self.assertEqual([ast.unparse(v) for v in values],
                         ["_diag_daemons.act_diagnostic_status"])

    def test_the_fresh_handler_runs_the_self_check_actions(self):
        # The same sweep "are you ok" runs: the ONE list of the self-check's
        # action names, not a second copy of it.
        fn = next(n for n in self.tree.body
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_fresh_diagnostic_status")
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        self.assertIn("_SELF_CHECK_ACTIONS", used)
        attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        self.assertIn("fresh_diagnostic_status", attrs)

    def test_the_self_check_never_names_diagnostic_status(self):
        # The fresh handler looks the sweep up in _SELF_CHECK_ACTIONS; if that
        # list ever named diagnostic_status the handler would call itself.
        tup = next(n.value for n in self.tree.body
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name)
                           and t.id == "_SELF_CHECK_ACTIONS"
                           for t in n.targets))
        names = ast.literal_eval(tup)
        self.assertIn("are_you_ok", names)
        self.assertNotIn("diagnostic_status", names)

    def test_the_tray_wrapper_holds_a_self_check_name_until_the_skill_loads(self):
        # Premise of FreshDiagnosticStatusTests.test_the_tray_wrapper_is_not_a_sweep
        # (tests/monolith): the import-time ACTIONS literal registers the
        # tray's async wrapper under run_diagnostic, one of
        # _SELF_CHECK_ACTIONS. The wrapper never returns a sweep's result, so
        # until the skill overrides it the fresh handler must skip it.
        literal = next(n.value for n in self.tree.body
                       if isinstance(n, ast.Assign)
                       and isinstance(n.value, ast.Dict)
                       and any(isinstance(t, ast.Name) and t.id == "ACTIONS"
                               for t in n.targets))
        held = {k.value: ast.unparse(v)
                for k, v in zip(literal.keys, literal.values)
                if isinstance(k, ast.Constant)}
        self.assertEqual(held.get("run_diagnostic"), "_act_run_diagnostic_tray")


class RoutingTests(unittest.TestCase):
    FULL = prompts.PC_CONTROL_PROMPT

    def test_the_section_teaches_diagnostic_status_with_an_arrow_example(self):
        _core, sections = pr.split_pc_control(self.FULL)
        body = " ".join(dict(sections)["SELF DIAGNOSTIC"].split())
        self.assertRegex(
            body, r"'diagnostic status' (?:→|->) \[ACTION: diagnostic_status\]")
        # ...and says it is a fresh sweep, with the counters on their own name.
        self.assertRegex(body.lower(), r"fresh sweep")
        self.assertIn("diagnostic_daemon_status", body)

    def test_the_slim_local_prompt_keeps_it(self):
        for q in ("diagnostic status", "what's the diagnostic status",
                  "give me a diagnostic status"):
            with self.subTest(q=q):
                slim = pr.slim_pc_control(q, self.FULL)
                self.assertIn("[ACTION: diagnostic_status]", slim)


if __name__ == "__main__":
    unittest.main()
