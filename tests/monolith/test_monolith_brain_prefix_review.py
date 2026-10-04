"""Review fixes to the brain-prefix branch (2026-10-04) - the monolith tier.

WHY
===
The adversarial review of claude/brain-prefix-stable (5e4d596) found:

1. A follow-up round's DENSE action result (JSON, a sensor table) was
   estimated at 3.7 characters per token and sent unclipped while its real
   size overflowed num_ctx; Ollama then cut the start of the system prompt.
2. A cut to 8,195 tokens (Ollama halving a prompt over the 16k window) with
   an estimate under num_ctx still taught "the window is 8,195", and a real
   over-window cut was no longer logged at all.
3. Test gaps: no test drove the reload detection through its two call
   sites, the empty-reply streak reset, or the two prefix checks of the
   stable head - each could be deleted with every test green.
4. Reload detection saw only loads reported by _call_local_llm or a prime;
   an owner call that re-read its whole prompt (the brain reloaded by
   vision, game mode, the orchestrator or another client) never re-armed
   stage-A.
5. Cosmetic: the call path printed "CANNOT FIT ... sent as is" about a
   prompt it then replaced with the compact layout; stage-A's re-check
   printed the "capture is music" note twice.

Every class fails (or a mutation of the code it pins turns it red) on
5e4d596.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests.monolith.test_monolith_brain_prefix import (  # noqa: E402
    _Base, _Resp,
)


def _dense_json(n_chars: int) -> str:
    rows = [{"id": 1000 + (i * 7919) % 98999, "name": f"item_{i}",
             "v": round(((i * 31) % 1000) / 10.0, 1),
             "ts": "2026-10-04T15:%02d:%02dZ" % (i % 60, (i * 7) % 60)}
            for i in range(400)]
    return json.dumps(rows)[:n_chars]


# ════════════════════════════════════════════════════════════════════════════
#  1. A dense follow-up result is clipped instead of overflowing
# ════════════════════════════════════════════════════════════════════════════
class DenseFollowupResultTests(_Base):
    def setUp(self):
        super().setUp()
        self._quiet()
        self._route("local")
        bc = self.bc
        self._p(bc, "_RESOLVED_LOCAL_LLM_MODEL", ["gemma-test"])
        bc.conversation_history[:] = [
            {"role": "user", "content": "read me the latest numbers"},
            {"role": "assistant", "content": "[ACTION: web_search, numbers]"}]

    def test_a_6000_character_json_result_is_clipped_to_fit(self):
        bc = self.bc
        pb = bc._prompt_budget
        system = "S" * 48000          # ~13k tokens, like the live prompt
        result = _dense_json(6000)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            fitted = bc._fit_followup_round(system, [("web_search", result)],
                                            [])
        self.assertIsNotNone(fitted, out.getvalue())
        final = fitted[-1]["content"]
        # It no longer fits whole (the real result is ~4.2k tokens, not the
        # 1,622 that 3.7 characters per token counted): it was clipped.
        self.assertIn("characters cut", final)
        self.assertIn("clipped", out.getvalue())
        s, m = bc._local_chat_prompt(system, list(fitted))
        budget = pb.budget_for(16384, 400)
        self.assertLessEqual(pb.estimate_chat_tokens(s, m), budget)
        # A floor the real prompt cannot go under: the system prompt's own
        # (all-letters) estimate plus one token per digit and line break of
        # the results message - still inside the window.
        floor = (pb.estimate_tokens(s)
                 + sum(c.isdigit() for c in final) + final.count("\n"))
        self.assertLess(floor, 16384)

    def test_a_prose_result_of_the_same_size_is_sent_whole(self):
        bc = self.bc
        prose = ("The page says the new model was released this week and "
                 "early reviews are positive about its battery life. ") * 60
        prose = prose[:6000]
        with contextlib.redirect_stdout(io.StringIO()):
            fitted = bc._fit_followup_round("S" * 48000,
                                            [("web_search", prose)], [])
        self.assertNotIn("characters cut", fitted[-1]["content"])


# ════════════════════════════════════════════════════════════════════════════
#  2. A halved prompt is logged and teaches no window
# ════════════════════════════════════════════════════════════════════════════
class HalvedPromptTests(_Base):
    def setUp(self):
        super().setUp()
        pb = self.bc._prompt_budget
        pb.OBSERVED_WINDOW.clear()
        self.addCleanup(pb.OBSERVED_WINDOW.clear)

    def _note(self, chars, pe):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            learned = self.bc._note_prompt_window(
                "x" * chars, [{"role": "user", "content": "hi"}],
                {"prompt_eval_count": pe}, "gemma-test")
        return learned, out.getvalue()

    def test_the_reviews_probe_under_the_window(self):
        # ~16.2k estimated (inside num_ctx 16,384) but Ollama kept 8,195.
        learned, log = self._note(16200 * 37 // 10, 8195)
        self.assertFalse(learned)
        self.assertEqual(self.bc._prompt_budget.OBSERVED_WINDOW.limit, 0)
        self.assertIn("TRUNCATED", log)
        self.assertIn("over the window", log)

    def test_an_over_window_cut_is_logged(self):
        learned, log = self._note(18363 * 37 // 10, 8195)
        self.assertFalse(learned)
        self.assertIn("[prompt-budget] TRUNCATED: Ollama evaluated 8,195", log)

    def test_an_honest_count_logs_nothing_and_is_recorded(self):
        pb = self.bc._prompt_budget
        learned, log = self._note(40000, 10500)
        self.assertFalse(learned)
        self.assertEqual(log, "")
        self.assertIsNotNone(pb.EXACT.lookup(
            "gemma-test", "x" * 40000, [{"role": "user", "content": "hi"}]))


# ════════════════════════════════════════════════════════════════════════════
#  3. The reload detection, the empty streak and the prefix checks are wired
# ════════════════════════════════════════════════════════════════════════════
class ReloadDetectionCallSiteTests(_Base):
    def setUp(self):
        super().setUp()
        self._quiet()
        self._live_prompt()
        self._reprime_ready()

    def test_a_reload_seen_by_an_owner_call_rearms_stage_a(self):
        bc = self.bc
        self._ollama(load_ms=9000)
        bc._stage_a_primed[0] = "primed-key"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            bc._call_local_llm("sys", [{"role": "user", "content": "hi"}])
        self.assertEqual(bc._stage_a_primed[0], "")
        self.assertIn("(re)loaded (9000 ms)", out.getvalue())

    def test_a_warm_call_leaves_it(self):
        bc = self.bc
        self._ollama(load_ms=7)
        bc._stage_a_primed[0] = "primed-key"
        with contextlib.redirect_stdout(io.StringIO()):
            bc._call_local_llm("sys", [{"role": "user", "content": "hi"}])
        self.assertEqual(bc._stage_a_primed[0], "primed-key")

    def test_a_reload_seen_by_the_full_prime_rearms_stage_a(self):
        bc = self.bc
        self._ollama()
        self._once()                       # stage-A + full prime
        self.assertTrue(bc._stage_a_primed[0])
        # The next prime finds the head primed (no stage-A) but its full
        # prime reports a reload: warm prefill time, a 9 s load.
        self._p(bc, "requests", mock.Mock())
        posted = self._ollama(load_ms=9000)
        self._once()
        self.assertEqual(len(posted), 1)
        self.assertEqual(bc._stage_a_primed[0], "")


class FullReReadRearmsStageATests(_Base):
    """Review 2026-10-04 (finding 6): a brain-layout call that re-read its
    whole prompt had no checkpoint at the head - stage-A again."""

    def setUp(self):
        super().setUp()
        self._quiet()
        self._live_prompt()
        self._reprime_ready()

    def _call(self, system):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.bc._call_local_llm(system,
                                    [{"role": "user", "content": "hi"}])
        return out.getvalue()

    def test_a_whole_re_read_of_the_brain_prompt_rearms_it(self):
        bc = self.bc
        self._ollama(pe_ms=3600)            # ~13.6k tokens re-read
        bc._stage_a_primed[0] = "primed-key"
        log = self._call(bc._local_stable_system_prompt())
        self.assertEqual(bc._stage_a_primed[0], "", log)
        self.assertIn("re-read its whole prompt", log)

    def test_a_partial_read_does_not(self):
        bc = self.bc
        self._ollama(pe_ms=1000)            # ~3.5k of ~13k: from the head
        bc._stage_a_primed[0] = "primed-key"
        self._call(bc._local_stable_system_prompt())
        self.assertEqual(bc._stage_a_primed[0], "primed-key")

    def test_another_callers_prompt_does_not(self):
        bc = self.bc
        self._ollama(pe_ms=3600)
        bc._stage_a_primed[0] = "primed-key"
        self._call("a background job's own short system prompt")
        self.assertEqual(bc._stage_a_primed[0], "primed-key")


class EmptyStreakThroughRealCallsTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.calls = []
        self.replies = ["", "Hello sir.", ""]

        def _post(url, json=None, timeout=None, **kw):
            self.calls.append(json["model"])
            if json["model"] == "good:model":
                return _Resp({"message": {"content": "Fallback here."}})
            return _Resp({"message": {"content": self.replies.pop(0)}})
        fake = mock.Mock()
        fake.post.side_effect = _post
        self._p(bc, "LOCAL_LLM_FALLBACK", True)
        self._p(bc, "_ollama_alive", return_value=True)
        self._p(bc, "_ollama_has_model", return_value=True)
        self._p(bc, "_get_local_llm_model", return_value="broken:model")
        self._p(bc, "_next_local_llm_fallback", return_value="good:model")
        self._p(bc, "requests", fake)
        saved = list(bc._RESOLVED_LOCAL_LLM_MODEL)
        self.addCleanup(lambda: bc._RESOLVED_LOCAL_LLM_MODEL.__setitem__(
            slice(None), saved))
        bc._local_empty_streak[0] = 0

    def test_a_good_reply_between_two_empties_means_no_failover(self):
        out = []
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(3):
                out.append(self.bc._call_local_llm(
                    "sys", [{"role": "user", "content": "hi"}]))
        self.assertEqual(out, [None, "Hello sir.", None])
        self.assertEqual(self.calls, ["broken:model"] * 3)
        self.assertEqual(self.bc._local_empty_streak[0], 1)


class StableHeadPrefixCheckTests(_Base):
    def setUp(self):
        super().setUp()
        self._quiet()
        self._live_prompt()
        self._reprime_ready()
        self._ollama()      # the model lookup must never reach a server

    def test_a_system_text_that_does_not_start_with_the_head_has_none(self):
        bc = self.bc
        head = bc._local_prompt_head()
        self.assertTrue(head)
        same_shape = "Z" * len(head) + "\n\nWhat you know about your owner:"
        self.assertIsNone(bc._local_prompt_head(same_shape))
        self.assertEqual(bc._local_prompt_head(head + "\n\nlearned"), head)

    def test_stage_a_needs_the_head_at_the_start_of_the_full_prime(self):
        bc = self.bc
        head = bc._local_prompt_head()
        payload = bc._build_reprime_payload()
        self.assertIsNotNone(bc._build_stage_a_payload(payload))
        bad = dict(payload)
        bad["messages"] = ([{"role": "system",
                             "content": "Z" * len(head) + "\n\nrest"}]
                           + list(payload["messages"][1:]))
        self.assertIsNone(bc._build_stage_a_payload(bad))


# ════════════════════════════════════════════════════════════════════════════
#  4. Log lines say what happened, once
# ════════════════════════════════════════════════════════════════════════════
class HonestLogLineTests(_Base):
    def test_the_compact_layout_is_not_announced_as_sent_as_is(self):
        bc = self.bc
        self._quiet()
        self._route("local")
        posted = self._ollama()
        self._p(bc, "_DYNAMIC_LOCAL_PROMPT", True)
        self._p(bc, "_STABLE_LOCAL_PREFIX", True)
        self._p(bc, "_local_cheatsheet",
                return_value="CHEATSHEET " + "c" * 70000)
        full = "BASE IDENTITY\n" + bc.PC_CONTROL_PROMPT + "\n\nfacts"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            bc._call_local_llm(full, [{"role": "user", "content": "hi"}])
        self.assertNotIn("CHEATSHEET", posted[-1]["messages"][0]["content"])
        self.assertIn("compact local layout (fits)", out.getvalue())
        self.assertNotIn("CANNOT FIT", out.getvalue())
        self.assertNotIn("sent as is", out.getvalue())

    def test_a_prompt_that_cannot_fit_any_layout_still_says_so(self):
        bc = self.bc
        self._quiet()
        self._route("local")
        self._ollama()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            bc._call_local_llm("S" * 70000,
                               [{"role": "user", "content": "hi"}])
        self.assertIn("CANNOT FIT", out.getvalue())

    def test_the_music_note_prints_once_per_prime(self):
        bc = self.bc
        self._quiet()
        self._live_prompt()
        self._reprime_ready()
        self._ollama()
        bc._utterance_in_progress[0] = True
        self.addCleanup(bc._utterance_in_progress.__setitem__, 0, False)
        self._p(bc, "_capture_is_sustained_music", return_value=True)
        res, log = self._once()
        self.assertEqual(res, "primed")
        self.assertEqual(log.count("capture is music"), 1, log)


if __name__ == "__main__":
    unittest.main()
