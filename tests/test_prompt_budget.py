"""Tests for core/prompt_budget: the local prompt must fit the model's window.

WHY (2026-10-01)
================
Ollama silently truncates a prompt longer than num_ctx to its first few tokens
plus the tail ("truncating input prompt limit=8195 prompt=17958 keep=5" in its
server log). Two live local turns that day read prompt_eval_count=8195: the
model answered with most of its system prompt (identity, action grammar,
safety rules) cut away. These tests pin the estimator against the live
measurements and the trim order on synthetic oversized prompts: the cheap
low-rank tail parts first, then the oldest history, then the sections, never
the system prompt and never the current message (nor, on a follow-up round,
the owner's request).
"""
from __future__ import annotations

import copy
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import prompt_budget as pb   # noqa: E402

# (characters, the tokens Ollama actually counted) from the live session logs,
# read-only. SYSTEM: each session's first idle re-prime (no history yet)
# against that session's sys_chars plus the 1,685-char local-mode directive.
# TURN: a turn's prompt_eval_count minus the re-prime that warmed its prefix,
# against its turn context + wrapper + the user's words. These include the
# densest (3.69 chars/token) and the loosest (4.52) material measured.
_SYSTEM_POINTS = [(50293, 13064), (51564, 13386), (51311, 13346),
                  (51072, 13294), (50916, 13271), (51168, 13354)]
_TURN_POINTS = [(2144, 581), (3626, 959), (4475, 1136), (6547, 1655),
                (5929, 1471), (12233, 2857), (5089, 1170), (3566, 789)]


def _msg(role, n, fill="x"):
    return {"role": role, "content": fill * n}


def _history(pairs, n=4000):
    out = []
    for i in range(pairs):
        out.append({"role": "user", "content": f"u{i} " + "q" * n})
        out.append({"role": "assistant", "content": f"a{i} " + "r" * n})
    return out


def _measure(system):
    return lambda msgs: pb.estimate_chat_tokens(system, msgs)


def _ctx_attach(messages, turn_ctx):
    """The monolith's _with_turn_context shape: a copy, the context on the
    front of the last user message."""
    if not turn_ctx:
        return messages
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = dict(out[i], content="[CTX]" + turn_ctx + "[/CTX]"
                          + out[i]["content"])
            return out
    return messages


class EstimatorCalibrationTests(unittest.TestCase):
    def test_never_under_the_live_system_prompt_counts(self):
        for chars, real in _SYSTEM_POINTS:
            with self.subTest(chars=chars):
                est = pb.estimate_chat_tokens("s" * chars, [])
                self.assertGreaterEqual(est, real)
                # ...and not so pessimistic that ordinary turns trim.
                self.assertLessEqual(est, real * 1.06)

    def test_never_under_the_live_turn_context_counts(self):
        for chars, real in _TURN_POINTS:
            with self.subTest(chars=chars):
                est = (pb.estimate_tokens("t" * chars)
                       + pb.MESSAGE_OVERHEAD_TOKENS)
                self.assertGreaterEqual(est, real)
                self.assertLessEqual(est, real * 1.30)

    def test_the_two_live_overflows_are_over_budget(self):
        # 2026-10-01 16:55 and 17:32: sys_chars 49,482 (+ directive) and
        # 12,861 / 13,686 chars of turn context, no history counted at all.
        budget = pb.budget_for(16384, 500)
        for ctx in (12861, 13686):
            with self.subTest(ctx=ctx):
                est = pb.estimate_chat_tokens(
                    "s" * (49482 + 1685), [_msg("user", ctx + 200)])
                self.assertGreater(est, budget)

    def test_a_typical_live_turn_is_within_budget(self):
        # Median live turn: the same system prompt, ~4.4k chars of turn
        # context and a modest history. It must NOT trim.
        est = pb.estimate_chat_tokens(
            "s" * (49219 + 1685),
            _history(2, n=300) + [_msg("user", 4410 + 200)])
        self.assertLess(est, pb.budget_for(16384, 500))

    def test_content_blocks_and_junk_are_counted_safely(self):
        self.assertEqual(pb.estimate_tokens(""), 0)
        self.assertEqual(pb.estimate_tokens(None), 0)
        self.assertEqual(pb.estimate_tokens(12345), 0)
        blocks = [{"role": "user",
                   "content": [{"type": "text", "text": "a" * 370},
                               {"type": "image", "source": {}}]}]
        self.assertEqual(pb.estimate_chat_tokens("", blocks),
                         pb.PROMPT_OVERHEAD_TOKENS
                         + 2 * pb.MESSAGE_OVERHEAD_TOKENS + 100)
        self.assertEqual(pb.estimate_chat_tokens("", [None, "x"]),
                         pb.PROMPT_OVERHEAD_TOKENS
                         + 3 * pb.MESSAGE_OVERHEAD_TOKENS)


class BudgetForTests(unittest.TestCase):
    def test_reserves_the_reply(self):
        # A voice reply's room, not its whole cap (review 2026-10-02).
        # ...and SAFETY_MARGIN_TOKENS (128) kept free (2026-10-04).
        self.assertEqual(pb.budget_for(16384, 500), 16056)
        self.assertEqual(pb.budget_for(12288, 400), 11960)
        self.assertEqual(pb.budget_for(16384, 50), 16206)

    def test_junk_never_yields_a_tiny_budget(self):
        self.assertEqual(pb.budget_for(None, 500), pb.MIN_BUDGET_TOKENS)
        self.assertEqual(pb.budget_for(1100, 4000), pb.MIN_BUDGET_TOKENS)
        self.assertEqual(pb.budget_for(16384, "x"),
                         16384 - pb.SAFETY_MARGIN_TOKENS)


class FitWithinBudgetTests(unittest.TestCase):
    def test_a_prompt_that_fits_is_sent_exactly_as_before(self):
        system = "S" * 1000
        msgs = _history(2, n=50) + [_msg("user", 20)]
        parts = [pb.TurnPart("A", "alpha\n", pb.RANK_SECTION),
                 pb.TurnPart("tone", "be brief", pb.RANK_REGISTER)]
        fit = pb.fit_chat(msgs, parts, budget=10_000,
                          measure=_measure(system), attach=_ctx_attach)
        self.assertEqual(fit.messages,
                         _ctx_attach(msgs, "alpha\nbe brief"))
        self.assertFalse(fit.trimmed)
        self.assertTrue(fit.fits)
        self.assertEqual(fit.before, fit.after)

    def test_no_parts_no_attach(self):
        msgs = [_msg("user", 10)]
        fit = pb.fit_chat(msgs, (), budget=10_000, measure=_measure(""))
        self.assertEqual(fit.messages, msgs)

    def test_empty_messages(self):
        fit = pb.fit_chat([], (), budget=10_000, measure=_measure("S"))
        self.assertEqual(fit.messages, [])
        self.assertTrue(fit.fits)


class FitOversizedTests(unittest.TestCase):
    """Synthetic prompts shaped like the live overflow."""

    SYSTEM = "S" * 51000       # the cache-stable system prompt, ~13.8k est.
    BUDGET = pb.budget_for(16384, 500)

    def _parts(self):
        return [
            pb.TurnPart("KINECT DEPTH SENSOR", "K" * 2500, pb.RANK_SECTION),
            pb.TurnPart("STREAMING SERVICES", "\n" + "M" * 5000,
                        pb.RANK_SECTION),
            pb.TurnPart("BAMBU 3D PRINTER", "\n" + "B" * 4000,
                        pb.RANK_SECTION),
            pb.TurnPart("tone", "T" * 300, pb.RANK_REGISTER),
            pb.TurnPart("long-term memory", "L" * 1500, pb.RANK_MEMORY),
            pb.TurnPart("phrase rotation", "P" * 200, pb.RANK_STYLE_HINT),
        ]

    def _fit(self, msgs, parts=None, **kw):
        return pb.fit_chat(msgs, self._parts() if parts is None else parts,
                           budget=self.BUDGET, measure=_measure(self.SYSTEM),
                           attach=_ctx_attach, **kw)

    def test_history_goes_oldest_first_after_the_cheap_tail(self):
        # The turn context alone fits; the 10-exchange history does not. The
        # low-rank tone hint goes first (it breaks no cached prefix, review
        # 2026-10-02), then the oldest history; the section stays.
        msgs = _history(10, n=1200) + [_msg("user", 60, "w")]
        small = [pb.TurnPart("KINECT DEPTH SENSOR", "K" * 1500,
                             pb.RANK_SECTION),
                 pb.TurnPart("tone", "T" * 300, pb.RANK_REGISTER)]
        fit = self._fit(msgs, small)
        self.assertTrue(fit.fits, fit)
        self.assertTrue(fit.trimmed)
        self.assertEqual(fit.dropped_parts, ("tone",))
        kept = [m["content"].split(" ")[0] for m in fit.messages[:-1]]
        # The newest exchanges survive, in order; the oldest went.
        self.assertEqual(kept, [f"{r}{i}" for i in range(10 - len(kept) // 2, 10)
                                for r in ("u", "a")])
        self.assertNotIn("u0", kept)
        self.assertEqual(fit.dropped_history, 20 - len(kept))

    def test_then_lowest_rank_parts_then_largest_section(self):
        # Even with history down to the last exchange, the context is over.
        msgs = _history(10, n=1200) + [_msg("user", 60, "w")]
        parts = self._parts() + [
            pb.TurnPart("TASK QUEUE", "\n" + "Q" * 9000, pb.RANK_SECTION)]
        fit = self._fit(msgs, parts)
        self.assertTrue(fit.fits, fit)
        self.assertEqual(len(fit.messages), 3,
                         "the last exchange is kept before any part goes")
        # Order: style hint, register, memory, then sections largest first.
        self.assertEqual(fit.dropped_parts[:3],
                         ("phrase rotation", "tone", "long-term memory"))
        self.assertEqual(fit.dropped_parts[3], "TASK QUEUE")
        last = fit.messages[-1]["content"]
        for gone in ("P" * 50, "T" * 50, "L" * 50, "Q" * 50):
            self.assertNotIn(gone, last)

    def test_the_current_message_and_system_are_never_touched(self):
        user = "turn off the lights in here please"
        msgs = _history(10, n=3000) + [{"role": "user", "content": user}]
        before = copy.deepcopy(msgs)
        fit = self._fit(msgs)
        self.assertTrue(fit.messages[-1]["content"].endswith(user))
        self.assertEqual(msgs, before, "the caller's list was mutated")
        self.assertLessEqual(fit.after, self.BUDGET)

    def test_history_stays_user_first(self):
        # A follow-up loop can leave consecutive assistant messages; the cloud
        # fallback rejects a list that opens with one.
        msgs = ([_msg("user", 3000), _msg("assistant", 3000),
                 _msg("assistant", 3000), _msg("assistant", 3000)]
                + _history(3, n=3000) + [_msg("user", 10)])
        fit = self._fit(msgs, parts=[])
        self.assertTrue(fit.trimmed)
        self.assertEqual(fit.messages[0]["role"], "user")

    def test_rest_of_history_goes_last(self):
        # Sections that alone nearly fill the window: every part and every
        # history message has to go.
        msgs = _history(2, n=2500) + [_msg("user", 10)]
        parts = [pb.TurnPart("HUGE", "H" * 9000, pb.RANK_SECTION)]
        fit = self._fit(msgs, parts)
        self.assertTrue(fit.fits, fit)
        self.assertEqual(fit.dropped_parts, ("HUGE",))
        self.assertEqual(len(fit.messages), 3,
                         "the parts go before the last exchange does")
        big = pb.TurnPart("HUGE", "H" * 9000, pb.RANK_SECTION)
        fit2 = pb.fit_chat(_history(2, n=4000) + [_msg("user", 10)], [big],
                           budget=self.BUDGET,
                           measure=_measure("S" * 57000), attach=_ctx_attach)
        self.assertTrue(fit2.fits, fit2)
        self.assertEqual(len(fit2.messages), 1)

    def test_cannot_fit_trims_everything_and_reports(self):
        # The system prompt + the current message alone are over. Ollama
        # keeps the first numKeep tokens and cuts the next (length - num_ctx),
        # so every history / part token left in costs one from the START of
        # the system prompt: everything that can go, goes (review 2026-10-02;
        # this used to send the prompt unchanged).
        for system, user in (("S" * 60000, 20), ("S" * 40000, 30000)):
            with self.subTest(system=len(system), user=user):
                msgs = _history(3, n=500) + [_msg("user", user, "w")]
                parts = self._parts()
                fit = pb.fit_chat(msgs, parts, budget=self.BUDGET,
                                  measure=_measure(system),
                                  attach=_ctx_attach)
                self.assertFalse(fit.fits)
                self.assertTrue(fit.trimmed)
                self.assertEqual(fit.messages, msgs[-1:])
                self.assertEqual(fit.after, fit.floor)
                self.assertGreater(fit.floor, self.BUDGET)
                note = pb.describe(fit, "turn", num_ctx=16384)
                self.assertIn("CANNOT FIT", note)
                self.assertIn(f"~{fit.floor:,} tok", note)
                self.assertNotIn("\n", note)

    def test_a_trim_that_works_always_ends_within_budget(self):
        # Just under the floor limit: everything trimmable has to go, and it
        # then fits.
        system = "S" * 57000
        msgs = _history(4, n=3000) + [_msg("user", 1000, "w")]
        fit = self._fit_with(system, msgs)
        self.assertTrue(fit.fits, fit)
        self.assertLessEqual(fit.floor, self.BUDGET)
        self.assertEqual(fit.messages[-1]["content"][-1000:], "w" * 1000)

    def _fit_with(self, system, msgs):
        return pb.fit_chat(msgs, self._parts(), budget=self.BUDGET,
                           measure=_measure(system), attach=_ctx_attach)

    def test_describe_is_one_line_naming_what_went(self):
        msgs = _history(10, n=1200) + [_msg("user", 60, "w")]
        parts = self._parts() + [
            pb.TurnPart(f"SECTION {i}", "\n" + "Z" * 2000, pb.RANK_SECTION)
            for i in range(6)]
        fit = self._fit(msgs, parts)
        note = pb.describe(fit, "turn", num_ctx=16384)
        self.assertTrue(note.startswith("[prompt-budget] turn: ~"))
        self.assertIn("budget 16,056 (num_ctx 16384)", note)
        self.assertIn("oldest history msg(s)", note)
        self.assertIn("phrase rotation", note)
        self.assertIn("more]", note)
        self.assertNotIn("\n", note)
        self.assertNotIn("STILL OVER", note)


class ReviewTrimOrderTests(unittest.TestCase):
    """Review 2026-10-02 (medium): the first thing an overflowing turn shed
    was the OLDEST HISTORY - which moves the cached prefix's divergence point
    to just after the system prompt and costs a full re-evaluation - while
    the low-rank parts at the END of the prompt (phrase rotation, tone,
    long-term memory), which break no cache, were kept."""

    SYSTEM = "S" * 50904

    def _parts(self):
        return [pb.TurnPart("SMART HOME", "X" * 4410, pb.RANK_SECTION),
                pb.TurnPart("tone", "T" * 250, pb.RANK_REGISTER),
                pb.TurnPart("long-term memory", "L" * 700, pb.RANK_MEMORY),
                pb.TurnPart("phrase rotation", "P" * 300,
                            pb.RANK_STYLE_HINT)]

    def test_low_rank_parts_go_before_any_history(self):
        msgs = _history(10, n=60) + [_msg("user", 40, "w")]
        measure = _measure(self.SYSTEM)
        full = measure(_ctx_attach(msgs, "".join(p.text for p in self._parts())))
        # Over by less than the three low-rank parts weigh.
        fit = pb.fit_chat(msgs, self._parts(), budget=full - 200,
                          measure=measure, attach=_ctx_attach)
        self.assertTrue(fit.fits, fit)
        self.assertEqual(fit.dropped_history, 0)
        self.assertEqual(fit.messages[:-1], msgs[:-1],
                         "history must go out byte-for-byte unchanged")
        self.assertEqual(set(fit.dropped_parts),
                         {"phrase rotation", "tone", "long-term memory"})
        self.assertIn("X" * 4410, fit.messages[-1]["content"])

    def test_an_inherited_section_goes_before_the_turns_own_memory(self):
        # prompt_router.inherited_turn_sections: a section only the history
        # routed ranks below the turn's own long-term-memory recall.
        self.assertLess(pb.RANK_INHERITED, pb.RANK_MEMORY)
        self.assertGreater(pb.RANK_INHERITED, pb.RANK_REGISTER)
        msgs = [_msg("user", 40, "w")]
        parts = [pb.TurnPart("OWN", "O" * 2000, pb.RANK_SECTION),
                 pb.TurnPart("INHERITED", "I" * 2000, pb.RANK_INHERITED),
                 pb.TurnPart("long-term memory", "L" * 700, pb.RANK_MEMORY)]
        measure = _measure(self.SYSTEM)
        full = measure(_ctx_attach(msgs, "".join(p.text for p in parts)))
        fit = pb.fit_chat(msgs, parts, budget=full - 300, measure=measure,
                          attach=_ctx_attach)
        self.assertEqual(fit.dropped_parts, ("INHERITED",))

    def test_the_reply_reserve_is_a_voice_reply_not_max_tokens(self):
        # Voice replies run ~15-60 tokens; reserving the full 500 trimmed
        # turns Ollama would never have cut (15.2k-16.4k real tokens).
        self.assertEqual(pb.budget_for(16384, 500),
                         16384 - pb.REPLY_RESERVE_TOKENS
                         - pb.SAFETY_MARGIN_TOKENS)
        self.assertEqual(pb.budget_for(16384, 100), 16156)
        self.assertLessEqual(pb.REPLY_RESERVE_TOKENS, 256)


class ReviewCannotFitTests(unittest.TestCase):
    """Review 2026-10-02 (medium): when the system prompt + the current
    message alone were over, the prompt was sent UNCHANGED. Ollama keeps the
    first numKeep tokens and drops the next (len - num_ctx), so every
    history or part token left in costs one token from the START of the
    system prompt - the identity and safety rules. Trim everything that can
    go, then say it still cannot fit."""

    def test_everything_droppable_goes_and_it_says_so(self):
        system = "S" * 60000
        msgs = _history(3, n=500) + [_msg("user", 20, "w")]
        parts = [pb.TurnPart("A", "A" * 3000, pb.RANK_SECTION),
                 pb.TurnPart("tone", "T" * 200, pb.RANK_REGISTER)]
        budget = pb.budget_for(16384, 500)
        fit = pb.fit_chat(msgs, parts, budget=budget,
                          measure=_measure(system), attach=_ctx_attach)
        self.assertFalse(fit.fits)
        self.assertTrue(fit.trimmed)
        self.assertEqual(fit.messages, [msgs[-1]])
        self.assertEqual(set(fit.dropped_parts), {"A", "tone"})
        self.assertEqual(fit.after, fit.floor)
        note = pb.describe(fit, "turn", num_ctx=16384)
        self.assertIn("CANNOT FIT", note)
        self.assertIn("dropped", note)
        self.assertNotIn("\n", note)


class ReviewFollowupFloorTests(unittest.TestCase):
    """Review 2026-10-02 (medium): on a follow-up round the 'current message'
    is the machine-made action-results message, so the owner's actual
    request was dropped before anything else - the model saw "Continue
    working toward completing the original task" with no task, next to
    possibly untrusted results. pin_last_user keeps the owner's last turn
    (and the chain after it) in the floor."""

    def _followup_msgs(self, results_chars):
        return (_history(3, n=2500)
                + [{"role": "user", "content": "OWNER: read my newest email"},
                   {"role": "assistant", "content": "[ACTION: read_email]"},
                   {"role": "user", "content": "RESULTS " + "r" * results_chars}])

    def test_the_owners_request_is_never_dropped(self):
        system = "S" * 51500
        msgs = self._followup_msgs(5200)
        parts = [pb.TurnPart("EMAIL", "E" * 2700, pb.RANK_SECTION)]
        measure = _measure(system)
        unpinned_floor = measure(_ctx_attach(msgs[-1:], ""))
        # Just above the results message alone: the old order fitted by
        # throwing away the owner's request and the action that answered it.
        budget = unpinned_floor + 5
        old = pb.fit_chat(msgs, parts, budget=budget, measure=measure,
                          attach=_ctx_attach)
        self.assertTrue(old.fits)
        self.assertEqual(old.messages, msgs[-1:])
        fit = pb.fit_chat(msgs, parts, budget=budget, measure=measure,
                          attach=_ctx_attach, pin_last_user=True)
        # The request and the chain after it are part of the floor now: the
        # caller learns it cannot fit (and clips the results) instead.
        self.assertEqual(fit.messages, msgs[-3:])
        self.assertFalse(fit.fits)
        self.assertEqual(fit.after, fit.floor)
        self.assertEqual(fit.dropped_history, 6)
        self.assertEqual(fit.dropped_parts, ("EMAIL",))

    def test_with_room_only_the_older_history_goes(self):
        system = "S" * 51500
        msgs = self._followup_msgs(5200)
        parts = [pb.TurnPart("EMAIL", "E" * 2700, pb.RANK_SECTION)]
        measure = _measure(system)
        pinned_floor = measure(_ctx_attach(msgs[-3:], ""))
        fit = pb.fit_chat(msgs, parts, budget=pinned_floor + 800,
                          measure=measure, attach=_ctx_attach,
                          pin_last_user=True)
        self.assertTrue(fit.fits, fit)
        self.assertEqual(fit.messages[:2], msgs[-3:-1])
        self.assertEqual(fit.dropped_parts, ())
        self.assertIn("E" * 2700, fit.messages[-1]["content"])

    def test_clip_middle_keeps_head_and_tail(self):
        text = "HEAD" + "m" * 5000 + "TAIL"
        out = pb.clip_middle(text, 600)
        self.assertLessEqual(len(out), 600 + 60)
        self.assertTrue(out.startswith("HEAD"))
        self.assertTrue(out.endswith("TAIL"))
        self.assertIn("characters cut", out)
        self.assertEqual(pb.clip_middle("short", 600), "short")
        self.assertEqual(pb.clip_middle(None, 600), "")
        self.assertTrue(all(a > b for a, b in zip(pb.RESULT_CLIP_STEPS,
                                                  pb.RESULT_CLIP_STEPS[1:])))


class ReviewObservedWindowTests(unittest.TestCase):
    """Review 2026-10-02 (low): the 10-01 incident read limit=8195 - half the
    16k num_ctx the budget assumes - so the budget alone would not have
    saved that prompt, and nothing compared the estimate with what Ollama
    actually evaluated. A prompt_eval_count far under the estimate is a
    truncation: say so loudly, and budget the next prompts to it for a
    while."""

    def test_truncation_is_a_count_far_under_the_estimate(self):
        self.assertTrue(pb.looks_truncated(15800, 8195))
        self.assertFalse(pb.looks_truncated(15800, 15100))  # estimate errs high
        self.assertFalse(pb.looks_truncated(15800, 14200))  # 4.5 chars/token
        self.assertFalse(pb.looks_truncated(900, 300))      # too small to judge
        for junk in (None, "x", 0, -5):
            self.assertFalse(pb.looks_truncated(15800, junk))

    def test_the_observed_limit_budgets_the_next_prompts_for_a_while(self):
        now = [1000.0]
        w = pb.ObservedWindow(clock=lambda: now[0])
        self.assertEqual(w.effective(16384), 16384)
        self.assertFalse(w.note(13000, 12500))
        self.assertTrue(w.note(15800, 8195))
        self.assertEqual(w.limit, 8195)
        self.assertEqual(w.effective(16384), 8195)
        # never larger than the configured window
        self.assertEqual(w.effective(4096), 4096)
        now[0] += pb.OBSERVED_LIMIT_TTL_S + 1
        self.assertEqual(w.effective(16384), 16384)

    def test_a_bigger_untruncated_prompt_clears_it(self):
        w = pb.ObservedWindow(clock=lambda: 0.0)
        w.note(15800, 8195)
        self.assertFalse(w.note(13600, 13400))
        self.assertEqual(w.effective(16384), 16384)


if __name__ == "__main__":
    unittest.main()
