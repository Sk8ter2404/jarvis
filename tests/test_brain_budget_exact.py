"""Brain prompt budget + measurability (2026-10-04, brain-prefix track).

WHY
===
Ollama's server.log (10-01..10-04) showed the local brain re-reading whole
13.5-16k-token prompts 46 times for JARVIS (3.1-4.1 s each), and two prompts
cut to 8,195 tokens with most of their system prompt gone. What these pin:

  * the prompt budget is TOKEN-AWARE: a prefix whose size Ollama already
    reported (the idle re-prime's system + history, its stage-A head) is
    counted exactly, only what follows is estimated, and a count far from
    the estimate (truncated, or uncached-only) is never stored;
  * the budget keeps SAFETY_MARGIN_TOKENS free on top of the reply's room;
  * a prompt estimated OVER the configured window teaches the observed
    window nothing (10-04 13:07: a proactive remark's ~18.4k prompt made
    every local prompt for 15 minutes budget to 8,195);
  * [turn-timing] says how much of each brain prompt was really re-read
    (pe_new / pe_state) and the idle re-prime's verdict (reprime).

Every class fails on origin/main d5931da (the names do not exist there).
Stdlib only: the CI-light tier runs it.
"""
from __future__ import annotations

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import prompt_budget as pb   # noqa: E402
from core import turn_timing as tt     # noqa: E402


def _msg(role, n, ch="x"):
    return {"role": role, "content": ch * n}


SYSTEM = "S" * 40000          # ~10.8k estimated tokens
HEAD = SYSTEM[:30000]


class ExactCountsTests(unittest.TestCase):
    def setUp(self):
        self.ex = pb.ExactCounts()

    def test_a_known_prefix_is_counted_exactly_and_the_rest_estimated(self):
        hist = [_msg("user", 400), _msg("assistant", 400)]
        est = pb.estimate_chat_tokens(SYSTEM, hist)
        real = int(est * 0.96)                      # the estimate errs high
        self.assertTrue(self.ex.note("m", SYSTEM, hist, real))
        new = _msg("user", 370)
        got = pb.measure_chat_tokens(SYSTEM, hist + [new], model="m",
                                     exact=self.ex)
        self.assertEqual(got, real + pb.MESSAGE_OVERHEAD_TOKENS + 100)
        # Without the record it is the plain character estimate.
        self.assertEqual(
            pb.measure_chat_tokens(SYSTEM, hist + [new], model="m",
                                   exact=pb.ExactCounts()),
            pb.estimate_chat_tokens(SYSTEM, hist + [new]))

    def test_the_longest_known_prefix_wins(self):
        hist = [_msg("user", 400), _msg("assistant", 400)]
        e0 = pb.estimate_chat_tokens(SYSTEM, [])
        e2 = pb.estimate_chat_tokens(SYSTEM, hist)
        self.ex.note("m", SYSTEM, [], int(e0 * 0.95))
        self.ex.note("m", SYSTEM, hist, int(e2 * 0.95))
        self.assertEqual(self.ex.lookup("m", SYSTEM, hist + [_msg("user", 9)]),
                         (int(e2 * 0.95), 2))

    def test_another_model_or_another_system_prompt_is_not_used(self):
        est = pb.estimate_chat_tokens(SYSTEM, [])
        self.ex.note("m", SYSTEM, [], int(est * 0.95))
        self.assertIsNone(self.ex.lookup("other", SYSTEM, []))
        self.assertIsNone(self.ex.lookup("m", SYSTEM + "!", []))
        # ...nor a history that diverges from the recorded one.
        self.ex.note("m", SYSTEM, [_msg("user", 40, "a")],
                     int(pb.estimate_chat_tokens(
                         SYSTEM, [_msg("user", 40, "a")]) * 0.95))
        self.assertEqual(
            self.ex.lookup("m", SYSTEM, [_msg("user", 40, "b")])[1], 0)

    def test_a_count_far_from_the_estimate_is_never_stored(self):
        est = pb.estimate_chat_tokens(SYSTEM, [])
        # 10-01: Ollama cut prompts to 8,195 tokens; some servers report only
        # the uncached part (5 tokens on a warm prefix).
        self.assertGreater(est * pb.EXACT_SANITY[0], 8195)
        for bad in (8195, 5, 0, -3, None, "x", int(est * 1.5)):
            self.assertFalse(self.ex.note("m", SYSTEM, [], bad), bad)
        self.assertIsNone(self.ex.lookup("m", SYSTEM, []))

    def test_a_prompt_with_images_is_never_stored(self):
        msgs = [{"role": "user", "content": "x" * 400, "images": ["iVBOR"]}]
        est = pb.estimate_chat_tokens(SYSTEM, msgs)
        self.assertFalse(self.ex.note("m", SYSTEM, msgs, est))

    def test_a_known_head_counts_the_head_exactly(self):
        est_head = pb.estimate_chat_tokens(HEAD, [])
        self.assertTrue(self.ex.note_head("m", HEAD, int(est_head * 0.95)))
        hist = [_msg("user", 370)]
        got = pb.measure_chat_tokens(SYSTEM, hist, model="m", exact=self.ex)
        self.assertEqual(
            got, int(est_head * 0.95) + pb.estimate_tokens(SYSTEM[30000:])
            + pb.MESSAGE_OVERHEAD_TOKENS + 100)
        # A system prompt that does not start with the head: plain estimate.
        other = "T" + SYSTEM[1:]
        self.assertEqual(pb.measure_chat_tokens(other, hist, model="m",
                                                exact=self.ex),
                         pb.estimate_chat_tokens(other, hist))

    def test_bounded_and_clearable(self):
        for i in range(pb.ExactCounts.MAX_ENTRIES + 5):
            s = SYSTEM + str(i)
            self.ex.note("m", s, [], pb.estimate_chat_tokens(s, []))
        self.assertEqual(len(self.ex._entries), pb.ExactCounts.MAX_ENTRIES)
        self.ex.clear()
        self.assertIsNone(self.ex.lookup("m", SYSTEM + "20", []))

    def test_thread_safe_and_never_raises(self):
        errs = []

        def hammer(k):
            try:
                for i in range(200):
                    s = SYSTEM[:1000 + k] + str(i % 7)
                    self.ex.note("m", s, [], pb.estimate_chat_tokens(s, []))
                    pb.measure_chat_tokens(s, [_msg("user", 9)], model="m",
                                           exact=self.ex)
            except Exception as e:      # pragma: no cover - the failure
                errs.append(e)
        ts = [threading.Thread(target=hammer, args=(k,)) for k in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        self.assertEqual(errs, [])
        self.assertIsNone(self.ex.lookup(None, None, None))
        self.assertIsNone(self.ex.head_for("m", None))


class BudgetMarginTests(unittest.TestCase):
    def test_the_margin_is_kept_free_on_top_of_the_reply(self):
        self.assertGreater(pb.SAFETY_MARGIN_TOKENS, 0)
        self.assertEqual(pb.budget_for(16384, 500),
                         16384 - pb.REPLY_RESERVE_TOKENS
                         - pb.SAFETY_MARGIN_TOKENS)

    def test_the_live_worst_turn_still_fits_once_its_prefix_is_known(self):
        # 10-03 17:29: the estimate read 16,090 for a prompt Ollama counted
        # at 15,345. With its system + history known exactly (the re-prime),
        # only the new user turn is estimated - and it fits with the margin.
        ex = pb.ExactCounts()
        system = "S" * 50667
        hist = [_msg("user", 300), _msg("assistant", 600)] * 3
        est_prefix = pb.estimate_chat_tokens(system, hist)
        ex.note("m", system, hist, int(est_prefix * 0.955))
        turn = hist + [_msg("user", 3500)]
        known = pb.measure_chat_tokens(system, turn, model="m", exact=ex)
        self.assertLessEqual(known, pb.budget_for(16384, 500))
        self.assertLess(known, pb.estimate_chat_tokens(system, turn))


class ObservedWindowOverBudgetTests(unittest.TestCase):
    def test_a_prompt_estimated_over_the_window_teaches_nothing(self):
        w = pb.ObservedWindow(clock=lambda: 0.0)
        # The proactive remark: ~18.4k estimated, Ollama kept 8,195.
        self.assertFalse(w.note(18400, 8195, num_ctx=16384))
        self.assertEqual(w.limit, 0)
        self.assertEqual(w.effective(16384), 16384)

    def test_a_prompt_that_should_have_fit_still_teaches_the_window(self):
        w = pb.ObservedWindow(clock=lambda: 0.0)
        self.assertTrue(w.note(15800, 8195, num_ctx=16384))
        self.assertEqual(w.effective(16384), 8195)

    def test_callers_that_pass_no_window_keep_the_old_behaviour(self):
        w = pb.ObservedWindow(clock=lambda: 0.0)
        self.assertTrue(w.note(18400, 8195))
        self.assertEqual(w.limit, 8195)


class PrefillEstimateTests(unittest.TestCase):
    """Calibrated on Ollama's server.log: median ms by evaluated tokens
    15 -> 160, 176 -> 222, 998 -> 426, 1,831 -> 623, 13,604 -> 3,449."""

    def test_the_logged_cases_classify_as_they_happened(self):
        # 10-04 13:03:15: a midnight day-count change, 3,591 ms.
        self.assertEqual(tt.prefill_estimate(14521, 3591)[1], "full")
        # 10-04 13:25:34: a warm turn, 437 ms.
        self.assertEqual(tt.prefill_estimate(14237, 437)[1], "warm")
        # A resume from the stable head (~5k of 15k re-read, ~1.35 s).
        self.assertEqual(tt.prefill_estimate(15000, 1350)[1], "partial")
        # The 5-token re-read of an unchanged prime.
        self.assertEqual(tt.prefill_estimate(13611, 184), (142, "warm"))

    def test_the_estimate_is_clamped_to_the_prompt(self):
        self.assertEqual(tt.prefill_estimate(1000, 99999), (1000, "full"))
        self.assertEqual(tt.prefill_estimate(1000, 10), (0, "warm"))

    def test_missing_or_junk_numbers(self):
        for args in ((None, 100), (100, None), (0, 100), (True, 100),
                     (100, False), ("x", 1), (100, -1)):
            self.assertEqual(tt.prefill_estimate(*args), (None, None), args)


class TurnLineBrainFieldsTests(unittest.TestCase):
    def setUp(self):
        self.now = [100.0]
        self.lines = []
        self.t = tt.TurnTiming(print_fn=self.lines.append,
                               clock=lambda: self.now[0])

    def _turn(self, stats, reprime=None):
        self.t.begin("typed")
        self.t.mark("you")
        if reprime is not None:
            self.t.set_first("reprime", reprime)
            self.t.set_first("reprime", "stale")    # first value wins
        self.t.llm_response(stats, served=True)
        return tt.parse_line(self.t.emit())

    def test_a_full_re_read_and_its_reprime_verdict_are_on_the_line(self):
        d = self._turn({"prompt_eval_count": 14521, "prompt_eval_ms": 3591},
                       reprime="hit")
        self.assertEqual(d["prompt_eval_count"], "14521")
        self.assertEqual(d["pe_state"], "full")
        self.assertEqual(int(d["pe_new"]), 14338)
        self.assertEqual(d["reprime"], "hit")

    def test_no_brain_call_prints_dashes(self):
        d = self._turn({})
        self.assertEqual((d["pe_new"], d["pe_state"], d["reprime"]),
                         ("-", "-", "-"))

    def test_the_fields_are_last_and_unique(self):
        self.assertEqual(tt.STAT_FIELDS[-len(tt.BRAIN_FIELDS):],
                         tt.BRAIN_FIELDS)
        self.assertEqual(len(set(tt.STAT_FIELDS)), len(tt.STAT_FIELDS))


if __name__ == "__main__":
    unittest.main()
