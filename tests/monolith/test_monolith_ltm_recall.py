"""Long-term-memory recall: one log line per turn, a budget counted from the
recall's start, a bounded number of recall threads (2026-11-17, brain-11).

WHY
===
_ltm_context waited LTM's recall for 0.6 s and returned '' on a miss with no
log line and no turn-trace field, so whether the owner's memories ever
reached a prompt (PROMPT_SLIM_MODE relies on this recall) could not be told
from anything JARVIS wrote. Now:
  * every recall logs ONE line: "[ltm] recall n=K ms=T", "[ltm] recall
    timeout ms=T (budget B ms)" or "[ltm] recall skipped: ..."; a recall that
    misses and finishes later logs "[ltm] recall finished late ms=T" - the
    number LTM_RECALL_BUDGET_S is tuned by - and the turn trace gets
    ltm={hit, n, ms, outcome};
  * _call_llm starts the recall at the top of the turn, so the routing runs
    alongside the embedding, and the wait is what is LEFT of the budget: the
    voice thread never waits longer than it did;
  * at most _LTM_RECALL_MAX_INFLIGHT recall threads run at once;
  * boot runs one warm recall and logs its time.

Fakes only (a fake long_term_memory module); nothing loads an embedder.

    python -m unittest tests.monolith.test_monolith_ltm_recall
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests._monolith_harness import (  # noqa: E402
    MonolithGlobalsTestCase, requires_monolith,
)
# The real-_call_llm scaffold (fake /api/chat, local route, quiet turn).
from tests.monolith.test_monolith_prompt_budget import (  # noqa: E402
    _Base as _TurnBase,
)


def _fake_ltm(retrieve):
    fake = types.ModuleType("core.long_term_memory")
    fake.retrieve_facts = retrieve
    return fake


class _InlineThread:
    """threading.Thread stand-in that runs the target on .start()."""

    def __init__(self, target=None, args=(), kwargs=None, **_):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


@requires_monolith
class LtmRecallTests(MonolithGlobalsTestCase):

    def setUp(self):
        super().setUp()
        self.traced = []
        self._p(self.bc, "_ltm_enabled", return_value=True)
        self._p(self.bc, "_turn_trace_call",
                side_effect=lambda fn, *a, **k: self.traced.append((fn,) + a))
        self.gates = []
        self.addCleanup(self._release_all)

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _release_all(self):
        for g in self.gates:
            g.set()
        for t in list(self.bc._ltm_recall_threads):
            t.join(timeout=5)

    def _gated(self, facts, seconds=None):
        """A retrieve that returns ``facts`` after ``seconds`` or, without
        it, once the test releases its gate."""
        gate = threading.Event()
        self.gates.append(gate)

        def _retrieve(q, k=6):
            if seconds is None:
                gate.wait(10)
            else:
                time.sleep(seconds)
            return list(facts)
        return gate, _retrieve

    def _ltm_lines(self, out):
        return [ln for ln in out.splitlines() if "[ltm]" in ln]

    def test_the_live_budget(self):
        import core.config as cfg
        self.assertEqual(cfg.LTM_RECALL_BUDGET_S, 0.6)
        self.assertEqual(self.bc._ltm_recall_budget_s(), 0.6)
        for raw, want in ((99, 3.0), (0, 0.05), ("x", 0.6), (1.2, 1.2)):
            with mock.patch.object(self.bc, "_LTM_RETRIEVE_BUDGET_S", raw):
                self.assertEqual(self.bc._ltm_recall_budget_s(), want, raw)

    def test_a_hit_logs_one_line_and_rides_the_trace(self):
        bc = self.bc
        self._p(bc, "_ltm_module", return_value=_fake_ltm(
            lambda q, k=6: [{"text": "Sir keeps three ferrets"},
                            {"text": "Prefers metric units"}]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            txt = bc._ltm_context("what do I keep")
        self.assertIn("three ferrets", txt)
        lines = self._ltm_lines(out.getvalue())
        self.assertEqual(len(lines), 1, out.getvalue())
        self.assertRegex(lines[0], r"\[ltm\] recall n=2 ms=\d+$")
        # Numbers only: no fact text in the log.
        self.assertNotIn("ferrets", out.getvalue())
        (call,) = [c for c in self.traced if c[0] == "note_ltm"]
        self.assertEqual((call[1], call[3]), (2, "hit"))

    def test_nothing_found_still_logs_its_line(self):
        bc = self.bc
        self._p(bc, "_ltm_module", return_value=_fake_ltm(lambda q, k=6: []))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(bc._ltm_context("anything"), "")
        self.assertRegex(self._ltm_lines(out.getvalue())[0],
                         r"\[ltm\] recall n=0 ms=\d+$")
        self.assertEqual([c[3] for c in self.traced if c[0] == "note_ltm"],
                         ["none"])

    def test_a_timeout_is_logged_and_so_is_the_late_finish(self):
        # At the live 0.6 s budget: a miss used to return '' silently.
        bc = self.bc
        gate, retrieve = self._gated([{"text": "too late"}])
        self._p(bc, "_ltm_module", return_value=_fake_ltm(retrieve))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            t0 = time.monotonic()
            txt = bc._ltm_context("query")
            waited = time.monotonic() - t0
            gate.set()
            for t in list(bc._ltm_recall_threads):
                t.join(timeout=5)
        self.assertEqual(txt, "")
        self.assertGreater(waited, 0.55)
        self.assertLess(waited, 1.5)
        lines = self._ltm_lines(out.getvalue())
        self.assertEqual(len(lines), 2, out.getvalue())
        self.assertRegex(lines[0], r"\[ltm\] recall timeout ms=\d+ "
                                   r"\(budget 600 ms\)")
        self.assertRegex(lines[1], r"\[ltm\] recall finished late ms=\d+ n=1 "
                                   r"\(budget 600 ms\)")
        self.assertEqual([c[3] for c in self.traced if c[0] == "note_ltm"],
                         ["timeout"])

    def test_the_wait_is_what_is_left_of_the_budget(self):
        # Started at the top of the turn: the routing time counts against
        # the budget, so the voice thread waits less, never more.
        bc = self.bc
        _gate, retrieve = self._gated([{"text": "fact"}], seconds=0.45)
        self._p(bc, "_ltm_module", return_value=_fake_ltm(retrieve))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            h = bc._ltm_recall_start("what is my printer")
            time.sleep(0.3)                     # the turn's routing
            t0 = time.monotonic()
            txt = bc._ltm_context("what is my printer", started=h)
            waited = time.monotonic() - t0
        self.assertIn("- fact", txt)
        self.assertLess(waited, 0.35)
        # ...and a recall still running when the budget is spent is not
        # waited for at all past it.
        gate, retrieve = self._gated([{"text": "slow"}])
        self._p(bc, "_ltm_module", return_value=_fake_ltm(retrieve))
        with contextlib.redirect_stdout(io.StringIO()):
            h = bc._ltm_recall_start("again")
            time.sleep(0.62)
            t0 = time.monotonic()
            self.assertEqual(bc._ltm_context("again", started=h), "")
            self.assertLess(time.monotonic() - t0, 0.1)
            gate.set()
            for t in list(bc._ltm_recall_threads):
                t.join(timeout=5)

    def test_a_handle_for_other_text_is_not_used(self):
        bc = self.bc
        seen = []
        self._p(bc, "_ltm_module", return_value=_fake_ltm(
            lambda q, k=6: seen.append(q) or [{"text": "x"}]))
        with contextlib.redirect_stdout(io.StringIO()):
            h = bc._ltm_recall_start("first words")
            for t in list(bc._ltm_recall_threads):
                t.join(timeout=5)
            bc._ltm_context("second words", started=h)
        self.assertEqual(seen, ["first words", "second words"])

    def test_recall_threads_are_bounded(self):
        bc = self.bc
        self.assertEqual(bc._LTM_RECALL_MAX_INFLIGHT, 2)
        _gate, retrieve = self._gated([])
        self._p(bc, "_ltm_module", return_value=_fake_ltm(retrieve))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            a = bc._ltm_recall_start("one")
            b = bc._ltm_recall_start("two")
            c = bc._ltm_recall_start("three")
            self.assertEqual(bc._ltm_context("three", started=c), "")
        self.assertNotIn("busy", a)
        self.assertNotIn("busy", b)
        self.assertTrue(c.get("busy"))
        self.assertIn("[ltm] recall skipped: 2 earlier recall(s) still "
                      "running", out.getvalue())
        self.assertIn("busy", [c[3] for c in self.traced
                               if c[0] == "note_ltm"])
        self._release_all()
        with contextlib.redirect_stdout(io.StringIO()):
            d = bc._ltm_recall_start("four")
        self.assertNotIn("busy", d)

    def test_off_blank_or_no_store_starts_nothing(self):
        bc = self.bc
        self.assertIsNone(bc._ltm_recall_start("   "))
        with mock.patch.object(bc, "_ltm_module", return_value=None):
            self.assertIsNone(bc._ltm_recall_start("hello"))
        with mock.patch.object(bc, "_ltm_enabled", return_value=False):
            self.assertIsNone(bc._ltm_recall_start("hello"))
            self.assertEqual(bc._ltm_context("hello"), "")
        self.assertEqual(bc._ltm_recall_threads, [])

    def test_boot_runs_one_warm_recall_and_logs_its_time(self):
        bc = self.bc
        fake = mock.Mock()
        fake.list_facts.return_value = []
        out = io.StringIO()
        with mock.patch.object(bc, "_ltm_module", return_value=fake), \
                mock.patch.object(bc.threading, "Thread", _InlineThread), \
                contextlib.redirect_stdout(out):
            bc._ltm_boot_warm()
        fake.retrieve_facts.assert_called_once_with("hello", k=1)
        self.assertIn("[ltm] embedder ready", out.getvalue())
        self.assertRegex(out.getvalue(), r"\[ltm\] warm recall ms=\d+ "
                                         r"\(recall budget 600 ms\)")


class CallLlmStartsTheRecallFirstTests(_TurnBase):
    """_call_llm starts the recall before the mood classifier and the prompt
    routing, and hands the handle to _ltm_context."""

    def test_the_recall_starts_before_the_routing(self):
        bc = self.bc
        order = []
        handle = {"text": "play some music", "sentinel": True}
        real_mood = bc._classify_turn_mood
        seen = {}

        def _start(text):
            order.append("recall")
            return handle

        def _mood(text):
            order.append("mood")
            return real_mood(text)

        def _ctx(text, started=None):
            seen["started"] = started
            return ""
        self._p(bc, "_ltm_recall_start", _start)
        self._p(bc, "_classify_turn_mood", _mood)
        self._p(bc, "_ltm_context", _ctx)
        self._turn("play some music", [])
        self.assertEqual(order[:2], ["recall", "mood"])
        self.assertIs(seen["started"], handle)


if __name__ == "__main__":
    unittest.main()
