"""The local brain re-reads as little as possible (2026-10-04).

WHY
===
gemma4 uses sliding-window attention, so llama.cpp can resume a changed
prompt only from a checkpoint saved at or before its first changed token,
and it saves checkpoints only near a prompt's END. Ollama's server.log
(10-01 18:38 .. 10-04) showed 46 full re-reads of 13.5-16k-token JARVIS
prompts (3.1-4.1 s each): 15 because the day / conversation count - bumped
at every boot, rolled at midnight - sat at token ~10.8k ahead of ~2.8k
tokens of memory, 5 because the topics list moved, 15 after brain reloads.
Two proactive remarks were cut to 8,195 tokens (system prompt gone), and
each taught the budget a false 8,195-token window for 15 minutes.

WHAT THESE TESTS PIN
====================
1. build_system_prompt: every learned section comes after the stable head,
   in a fixed order (facts, projects, summaries, topics, then the
   day / conversation count LAST), and no memory change touches the head.
2. The idle re-prime's STAGE A posts the local head alone - a strict prefix
   of the full prime, same model / options / keep_alive / think - when the
   server may not hold a checkpoint there (first prime, new head, reload,
   a full re-read), and not otherwise. The full prime is unchanged.
3. The prompt budget counts what Ollama already reported exactly; a prompt
   whose system part cannot fit is sent in the compact local layout; the
   proactive remark uses that layout on the local route; an over-window
   prompt teaches the observed window nothing.
4. One empty reply never loads another model (it unloads the brain).
5. JARVIS persists the OLLAMA_MAX_LOADED_MODELS value it runs with.
6. [turn-timing] carries the re-prime verdict.

Every class fails on origin/main d5931da.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests._monolith_harness import (  # noqa: E402
    MonolithGlobalsTestCase, requires_monolith,
)


class _Resp:
    def __init__(self, body, ok=True, status_code=200):
        self._body = body
        self.ok = ok
        self.status_code = status_code
        self.text = "body"

    def json(self):
        return self._body


def _memory(bc, **over):
    mem = bc._empty_memory()
    mem["first_meeting"] = "2026-05-26"
    mem["conversation_count"] = 770
    mem["facts"] = [f"Fact number {i} about the owner" for i in range(12)]
    mem["projects"] = ["A project", "Another project"]
    mem["sessions"] = [{"date": "2026-10-03", "location": "office",
                        "summary": f"Session summary {i}"} for i in range(3)]
    mem["topics"] = [{"date": "2026-10-04", "location": "office",
                      "topic": f"topic {i}"} for i in range(5)]
    mem.update(over)
    return mem


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _route(self, route):
        import core.config as cfg
        self._p(cfg, "model_route", return_value=route)

    def _build(self, mem=None, rules="RULESBLOCK"):
        bc = self.bc
        with mock.patch.object(bc, "_load_chappie_standing_rules",
                               return_value=rules), \
                mock.patch.object(bc, "PC_CONTROL_ENABLED", True):
            return bc.build_system_prompt(mem or _memory(bc))

    def _live_prompt(self, mem=None, rules="RULESBLOCK"):
        """Build a real prompt and make it the live one (stable layout)."""
        bc = self.bc
        self._p(bc, "_DYNAMIC_LOCAL_PROMPT", True)
        self._p(bc, "_STABLE_LOCAL_PREFIX", True)
        prompt = self._build(mem, rules)
        bc._system_prompt = prompt
        return prompt

    def _ollama(self, pe_ms=200, load_ms=5, statuses=None):
        """Record every /api/chat body; reply with an honest
        prompt_eval_count (~4 % under the estimate, as measured live)."""
        bc = self.bc
        pb = bc._prompt_budget
        posted = []
        fake = mock.Mock()
        statuses = list(statuses or [])

        def _post(url, json=None, timeout=None, **_k):
            posted.append(json)
            status = statuses.pop(0) if statuses else 200
            if status != 200:
                return _Resp({"error": "boom"}, ok=False, status_code=status)
            msgs = json["messages"]
            pe = int(pb.estimate_chat_tokens(msgs[0]["content"], msgs[1:])
                     * 0.96)
            ms = pe_ms(json) if callable(pe_ms) else pe_ms
            return _Resp({"model": json["model"],
                          "message": {"role": "assistant",
                                      "content": "Very good, sir."},
                          "done": True, "prompt_eval_count": pe,
                          "prompt_eval_duration": int(ms * 1e6),
                          "load_duration": int(load_ms * 1e6),
                          "eval_count": 3})
        fake.post.side_effect = _post
        self._p(bc, "LOCAL_LLM_FALLBACK", True)
        self._p(bc, "_ollama_alive", return_value=True)
        self._p(bc, "_ollama_has_model", return_value=True)
        self._p(bc, "_get_local_llm_model", return_value="gemma-test")
        self._p(bc, "_next_local_llm_fallback", return_value=None)
        self._p(bc, "requests", fake)
        return posted

    def _quiet(self):
        bc = self.bc
        self._p(bc, "_ltm_context", return_value="")
        self._p(bc, "_ltm_enqueue")
        self._p(bc, "load_memory", return_value=bc._empty_memory())
        self._p(bc, "save_memory")
        self._p(bc, "_voice_mood_response", None)
        bc._phrase_rotation_last[0] = {}

    def _reprime_ready(self):
        bc = self.bc
        self._route("local")
        self._p(bc, "LOCAL_PREFIX_REPRIME", True)
        import core.ollama_opts as oo
        self._p(oo, "model_resident", return_value=True)
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = False
        bc.conversation_history[:] = [
            {"role": "user", "content": "what's the weather"},
            {"role": "assistant", "content": "Clear skies, sir."}]

    def _once(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            res = self.bc._reprime_once()
        return res, out.getvalue()


# ════════════════════════════════════════════════════════════════════════════
#  1. Prompt layout
# ════════════════════════════════════════════════════════════════════════════
class PromptLayoutTests(_Base):
    def test_learned_sections_follow_the_head_in_a_fixed_order(self):
        bc = self.bc
        prompt = self._build()
        head = bc._system_prompt_head[0]
        self.assertTrue(head)
        self.assertTrue(prompt.startswith(head))
        tail = prompt[len(head):]
        marks = ["What you know about your owner:", "Projects picked up",
                 "Recent conversation summaries:", "Topics picked up",
                 "you've known your owner for"]
        at = [tail.find(m) for m in marks]
        self.assertNotIn(-1, at, tail[:400])
        self.assertEqual(at, sorted(at), "learned sections out of order")
        # The day / conversation count is the very last thing.
        self.assertTrue(prompt.rstrip().endswith(
            "across 770 conversations."), prompt[-200:])
        # ...and nothing learned is in the head.
        for m in marks + ["Fact number 0", "topic 0", "Session summary 0",
                          "770"]:
            self.assertNotIn(m, head)
        # The location line stayed in the head.
        self.assertIn(f"you are currently the {bc.LOCATION} instance", head)

    def test_no_memory_change_touches_the_head(self):
        bc = self.bc
        self._build(_memory(bc))
        head0 = bc._system_prompt_head[0]
        for over in ({"conversation_count": 771},
                     {"first_meeting": "2026-05-25"},
                     {"facts": ["A brand new fact"]},
                     {"projects": []},
                     {"topics": [{"date": "2026-10-05", "topic": "new"}]},
                     {"sessions": []}):
            prompt = self._build(_memory(bc, **over))
            self.assertEqual(bc._system_prompt_head[0], head0, over)
            self.assertTrue(prompt.startswith(head0), over)

    def test_the_cut_is_token_safe(self):
        # The head ends on a non-newline character and the tail opens with a
        # blank line: the brain's tokenizer never merges a newline with
        # anything else, so the head alone tokenizes as inside the prompt.
        bc = self.bc
        for mem in (_memory(bc), bc._empty_memory()):
            prompt = self._build(mem)
            head = bc._system_prompt_head[0]
            self.assertNotIn(head[-1], "\r\n")
            self.assertTrue(prompt[len(head):].startswith("\n\n"))

    def test_every_learned_item_is_still_sent_once(self):
        bc = self.bc
        mem = _memory(bc)
        prompt = self._build(mem)
        for f in mem["facts"]:
            self.assertEqual(prompt.count(f"- {f}\n") +
                             prompt.count(f"- {f}\n\n") >= 1, True, f)
        for s in mem["sessions"]:
            self.assertEqual(prompt.count(s["summary"]), 1)
        for t in mem["topics"]:
            self.assertEqual(prompt.count(f": {t['topic']}"), 1)
        self.assertEqual(prompt.count("you've known your owner for"), 1)
        self.assertEqual(prompt.count(f"the {bc.LOCATION} instance"), 1)

    def test_the_claude_cache_boundary_is_unchanged(self):
        # _system_prompt_stable_len (the Anthropic breakpoint) still sits
        # before the phrasebook; the local head is a separate, later mark.
        bc = self.bc
        with mock.patch.object(bc._mcu_phrases, "render_phrasebook_block",
                               return_value="PHRASEBOOK"):
            prompt = self._build()
        n = bc._system_prompt_stable_len[0]
        self.assertNotIn("PHRASEBOOK", prompt[:n])
        self.assertIn("PHRASEBOOK", bc._system_prompt_head[0])
        self.assertLess(n, len(bc._system_prompt_head[0]))


class LocalHeadTests(_Base):
    def test_the_local_head_is_a_strict_prefix_of_what_a_turn_sends(self):
        bc = self.bc
        self._live_prompt()
        head = bc._local_prompt_head()
        self.assertTrue(head)
        full, _m = bc._local_chat_prompt(bc._local_stable_system_prompt(),
                                         [{"role": "user", "content": "hi"}])
        self.assertTrue(full.startswith(head))
        self.assertEqual(full[len(head)], "\n")
        self.assertNotIn(bc.PC_CONTROL_PROMPT, head)

    def test_no_head_when_the_cut_would_split_a_newline_run(self):
        # "\n\n" may be ONE token: a head ending inside that run would not
        # tokenize alone as it does inside the prompt.
        bc = self.bc
        self._live_prompt()
        head = bc._system_prompt_head[0]
        self.assertEqual(bc._system_prompt[len(head):len(head) + 2], "\n\n")
        bc._system_prompt_head[0] = head + "\n"
        self.assertIsNone(bc._local_prompt_head())

    def test_no_head_when_it_is_not_a_prefix_of_the_live_prompt(self):
        bc = self.bc
        self._live_prompt()
        bc._system_prompt = "something else " + bc._system_prompt
        self.assertIsNone(bc._local_prompt_head())
        bc._system_prompt_head[0] = ""
        self.assertIsNone(bc._local_prompt_head())


# ════════════════════════════════════════════════════════════════════════════
#  2. Stage-A re-prime
# ════════════════════════════════════════════════════════════════════════════
class StageAReprimeTests(_Base):
    def setUp(self):
        super().setUp()
        self._quiet()
        self._live_prompt()
        self._reprime_ready()
        self.posted = self._ollama()

    def test_the_first_prime_posts_the_head_then_the_full_prime(self):
        bc = self.bc
        res, log = self._once()
        self.assertEqual(res, "primed")
        self.assertEqual(len(self.posted), 2, log)
        a, full = self.posted
        self.assertEqual(len(a["messages"]), 1)
        self.assertEqual(a["messages"][0]["role"], "system")
        head = a["messages"][0]["content"]
        self.assertEqual(head, bc._local_prompt_head())
        full_sys = full["messages"][0]["content"]
        self.assertTrue(full_sys.startswith(head))
        self.assertGreater(len(full_sys), len(head))
        self.assertEqual(full_sys[len(head)], "\n")
        # Same runner key: model, options (num_ctx!), keep_alive, think.
        for k in ("model", "keep_alive", "think", "stream"):
            self.assertEqual(a.get(k), full.get(k), k)
        self.assertEqual(a["options"], full["options"])
        self.assertEqual(a["options"]["num_predict"], 1)
        # The full prime is still exactly the next turn minus its message.
        self.assertEqual(full, bc._build_reprime_payload())
        self.assertRegex(log, r"\[reprime\] stage-A \d+ pe=\d+")
        self.assertRegex(log, r"\[reprime\] \d+ pe=\d+")

    def test_a_primed_head_is_not_primed_again(self):
        self._once()
        self.posted.clear()
        res, log = self._once()
        self.assertEqual(res, "primed")
        self.assertEqual(len(self.posted), 1, log)
        self.assertEqual(len(self.posted[0]["messages"]), 3)

    def test_a_learned_change_needs_no_new_stage_a(self):
        bc = self.bc
        self._once()
        self.posted.clear()
        bc._system_prompt = self._build(
            _memory(bc, facts=["A freshly learned fact"],
                    conversation_count=771))
        self._once()
        self.assertEqual(len(self.posted), 1)
        self.assertIn("A freshly learned fact",
                      self.posted[0]["messages"][0]["content"])

    def test_a_new_head_is_primed_again(self):
        bc = self.bc
        self._once()
        self.posted.clear()
        bc._system_prompt = self._build(rules="NEW RULES")
        self._once()
        self.assertEqual(len(self.posted), 2)
        self.assertIn("NEW RULES", self.posted[0]["messages"][0]["content"])

    def test_a_brain_reload_means_stage_a_again(self):
        bc = self.bc
        self._once()
        self.assertTrue(bc._stage_a_primed[0])
        bc._note_brain_response({"load_ms": 9000})
        self.assertEqual(bc._stage_a_primed[0], "")
        bc._note_brain_response({"load_ms": 7})        # a warm reply: no-op
        self.posted.clear()
        self._once()
        self.assertEqual(len(self.posted), 2)

    def test_a_full_re_read_after_a_primed_head_means_stage_a_again(self):
        bc = self.bc
        self._once()
        key = bc._stage_a_primed[0]
        self.assertTrue(key)
        # The server lost the checkpoint (reloaded by someone else): the
        # next full prime re-reads everything (~3.4 s).
        self._p(bc, "requests", mock.Mock())
        posted = self._ollama(pe_ms=3600)
        self._once()
        self.assertEqual(len(posted), 1)
        self.assertEqual(bc._stage_a_primed[0], "")

    def test_a_failed_stage_a_still_primes_and_retries_next_time(self):
        bc = self.bc
        self._p(bc, "requests", mock.Mock())
        posted = self._ollama(statuses=[500])
        res, log = self._once()
        self.assertEqual(res, "primed")
        self.assertEqual(len(posted), 2)
        self.assertIn("[reprime] stage-A failed", log)
        self.assertEqual(bc._stage_a_primed[0], "")

    def test_the_owner_starting_a_turn_stops_the_full_prime(self):
        bc = self.bc
        real = bc.requests.post.side_effect

        def _post(url, json=None, timeout=None, **k):
            out = real(url, json=json, timeout=timeout)
            bc._turn_in_progress[0] = True       # he spoke during stage-A
            return out
        bc.requests.post.side_effect = _post
        res, _log = self._once()
        self.assertEqual(res, "turn")
        self.assertEqual(len(self.posted), 1)
        self.assertEqual(bc._reprime_prefix_hash[0], "")

    def test_the_primes_record_exact_sizes_for_the_budget(self):
        bc = self.bc
        pb = bc._prompt_budget
        self._once()
        a, full = self.posted
        head = a["messages"][0]["content"]
        self.assertIsNotNone(pb.EXACT.head_for("gemma-test", head + "\n\nX"))
        hit = pb.EXACT.lookup("gemma-test", full["messages"][0]["content"],
                              full["messages"][1:])
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], len(full["messages"]) - 1)

    def test_the_switch_turns_stage_a_off(self):
        self._p(self.bc, "_REPRIME_STAGE_A", False)
        self._once()
        self.assertEqual(len(self.posted), 1)
        self.assertEqual(len(self.posted[0]["messages"]), 3)

    def test_no_stage_a_without_a_head(self):
        bc = self.bc
        bc._system_prompt_head[0] = ""
        self._once()
        self.assertEqual(len(self.posted), 1)


# ════════════════════════════════════════════════════════════════════════════
#  3. Budget: exact sizes, compact layout, proactive remark, observed window
# ════════════════════════════════════════════════════════════════════════════
class TokenAwareBudgetTests(_Base):
    def setUp(self):
        super().setUp()
        self._quiet()
        self._route("local")
        self.posted = self._ollama()

    def test_a_known_prefix_is_measured_not_estimated(self):
        bc = self.bc
        pb = bc._prompt_budget
        system = "S" * 50000
        hist = [{"role": "user", "content": "u" * 1500},
                {"role": "assistant", "content": "a" * 1500}] * 3
        msgs = hist + [{"role": "user", "content": "now"}]
        s_sent, m_sent = bc._local_chat_prompt(system, hist)
        est = pb.estimate_chat_tokens(s_sent, m_sent)
        # Over budget by the character estimate...
        budget = pb.budget_for(16384, 500)
        self.assertGreater(pb.estimate_chat_tokens(
            *bc._local_chat_prompt(system, msgs)), budget)
        # ...but Ollama said the prefix is 15 % smaller (dense-free text).
        pb.EXACT.note("gemma-test", s_sent, m_sent, int(est * 0.85))
        out = bc._fit_local_messages(system, msgs, (), where="test",
                                     model_tag="gemma-test", log=False)
        self.assertTrue(out.fits)
        self.assertFalse(out.trimmed)
        self.assertEqual(list(out), msgs)

    def test_a_system_prompt_that_cannot_fit_goes_compact(self):
        bc = self.bc
        self._p(bc, "_DYNAMIC_LOCAL_PROMPT", True)
        self._p(bc, "_STABLE_LOCAL_PREFIX", True)
        self._p(bc, "_local_cheatsheet",
                return_value="CHEATSHEET " + "c" * 70000)
        full = "BASE IDENTITY\n" + bc.PC_CONTROL_PROMPT + "\n\nfacts"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            text = bc._call_local_llm(full, [{"role": "user", "content": "hi"}])
        self.assertEqual(text, "Very good, sir.")
        sent = self.posted[-1]["messages"][0]["content"]
        self.assertNotIn("CHEATSHEET", sent)
        self.assertTrue(sent.startswith(bc._local_stable_system_prompt(full)))
        self.assertIn("compact local layout", out.getvalue())

    def test_a_prompt_that_fits_is_sent_as_before(self):
        bc = self.bc
        self._p(bc, "_local_cheatsheet", return_value="CHEATSHEET small")
        full = "BASE IDENTITY\n" + bc.PC_CONTROL_PROMPT + "\n\nfacts"
        bc._call_local_llm(full, [{"role": "user", "content": "hi"}])
        self.assertIn("CHEATSHEET small",
                      self.posted[-1]["messages"][0]["content"])

    def test_the_proactive_remark_uses_the_compact_layout_locally(self):
        bc = self.bc
        self._live_prompt()
        bc._proactive_recent[:] = []
        bc.generate_proactive_comment(now=1_760_000_000.0)
        self.assertEqual(len(self.posted), 1)
        sent = self.posted[0]["messages"][0]["content"]
        stable = bc._local_stable_system_prompt()
        self.assertTrue(sent.startswith(stable))
        self.assertIn("PROACTIVE comment", sent)
        self.assertNotIn("=== CONTROLLING THE PC ===", sent)
        # It shares the turns' warm prefix through the head at least.
        self.assertTrue(sent.startswith(bc._local_prompt_head()))
        pb = bc._prompt_budget
        self.assertLessEqual(
            pb.estimate_chat_tokens(sent, self.posted[0]["messages"][1:]),
            pb.budget_for(16384, 120))

    def test_the_cloud_remark_keeps_the_full_prompt(self):
        bc = self.bc
        self._live_prompt()
        self._route("cloud")
        self._p(bc, "AMBIENT_LEARNING_FORCE_LOCAL", False, create=True)
        import core.config as cfg
        self._p(cfg, "AMBIENT_LEARNING_FORCE_LOCAL", False)
        self._p(bc, "AI_BACKEND", "claude")
        seen = {}

        def _quick(system, user, max_tokens=200):
            seen["system"] = system
            return ""
        self._p(bc, "_llm_quick", side_effect=_quick)
        bc.generate_proactive_comment(now=1_760_000_000.0)
        self.assertTrue(seen["system"].startswith(bc._system_prompt))

    def test_an_over_window_prompt_teaches_no_window(self):
        bc = self.bc
        pb = bc._prompt_budget
        pb.OBSERVED_WINDOW.clear()
        self.addCleanup(pb.OBSERVED_WINDOW.clear)
        big = "S" * 70000                     # ~18.9k estimated tokens
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            learned = bc._note_prompt_window(
                big, [{"role": "user", "content": "x"}],
                {"prompt_eval_count": 8195}, "gemma-test")
        self.assertFalse(learned)
        self.assertEqual(pb.OBSERVED_WINDOW.limit, 0)
        # Review 2026-10-04: still a cut prompt (its system prompt's start
        # is gone) - logged, just not learned.
        self.assertIn("TRUNCATED", out.getvalue())
        self.assertIn("over the window", out.getvalue())
        # A prompt that should have fit and was HALVED (8,195 of 16,384:
        # the estimate was low) teaches nothing either...
        with contextlib.redirect_stdout(out):
            learned = bc._note_prompt_window(
                "S" * 55000, [{"role": "user", "content": "x"}],
                {"prompt_eval_count": 8195}, "gemma-test")
        self.assertFalse(learned)
        self.assertEqual(pb.OBSERVED_WINDOW.limit, 0)
        # ...a cut at any other count is a runner with a smaller window.
        with contextlib.redirect_stdout(out):
            learned = bc._note_prompt_window(
                "S" * 55000, [{"role": "user", "content": "x"}],
                {"prompt_eval_count": 4098}, "gemma-test")
        self.assertTrue(learned)
        self.assertEqual(pb.OBSERVED_WINDOW.limit, 4098)

    def test_an_honest_reply_records_the_prompts_exact_size(self):
        bc = self.bc
        pb = bc._prompt_budget
        system = "S" * 40000
        msgs = [{"role": "user", "content": "hello"}]
        est = pb.estimate_chat_tokens(system, msgs)
        bc._note_prompt_window(system, msgs,
                               {"prompt_eval_count": int(est * 0.96)},
                               "gemma-test")
        self.assertEqual(pb.EXACT.lookup("gemma-test", system, msgs),
                         (int(est * 0.96), 1))


# ════════════════════════════════════════════════════════════════════════════
#  4. One empty reply never loads another model
# ════════════════════════════════════════════════════════════════════════════
class EmptyFailoverNeedsTwoInARowTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.calls = []

        def _post(url, json=None, timeout=None, **kw):
            self.calls.append(json["model"])
            content = "" if json["model"] == "broken:model" else "Hello sir."
            return _Resp({"message": {"content": content}})
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

    def _call(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.bc._call_local_llm(
                "sys", [{"role": "user", "content": "hi"}])

    def test_the_first_empty_reply_loads_nothing(self):
        self.assertIsNone(self._call())
        self.assertEqual(self.calls, ["broken:model"])
        self.assertIsNone(self.bc._RESOLVED_LOCAL_LLM_MODEL[0])

    def test_the_second_in_a_row_fails_over_as_before(self):
        self._call()
        self.assertEqual(self._call(), "Hello sir.")
        self.assertEqual(self.calls, ["broken:model", "broken:model",
                                      "good:model"])
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], "good:model")
        self.assertEqual(self.bc._local_empty_streak[0], 0)

    def test_a_good_reply_in_between_resets_the_count(self):
        bc = self.bc
        self._call()
        bc._local_empty_streak[0] = 0          # what an 'ok' reply does
        self.assertIsNone(self._call())
        self.assertNotIn("good:model", self.calls)

    def test_a_background_thread_never_fails_over(self):
        self.bc._local_empty_streak[0] = 5
        out = []
        t = threading.Thread(target=lambda: out.append(self._call()))
        t.start()
        t.join(10)
        self.assertEqual(out, [None])
        self.assertNotIn("good:model", self.calls)


# ════════════════════════════════════════════════════════════════════════════
#  5. The persisted OLLAMA_MAX_LOADED_MODELS is the one JARVIS runs with
# ════════════════════════════════════════════════════════════════════════════
class MaxLoadedEnvTests(_Base):
    def _run(self, env):
        bc = self.bc
        seen = []
        self._p(bc, "_persist_user_env",
                side_effect=lambda n, v: seen.append((n, v)) or "already")
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("JARVIS_STAGING", "JARVIS_TEST_MODE",
                              "OLLAMA_MAX_LOADED_MODELS")}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True), \
                contextlib.redirect_stdout(io.StringIO()):
            bc._ensure_ollama_single_model_env()
        return seen

    def test_the_default_is_one(self):
        self.assertEqual(self._run({}),
                         [("OLLAMA_MAX_LOADED_MODELS", "1")])

    def test_an_explicit_override_is_kept_not_written_back_to_one(self):
        self.assertEqual(self._run({"OLLAMA_MAX_LOADED_MODELS": "2"}),
                         [("OLLAMA_MAX_LOADED_MODELS", "2")])

    def test_junk_reads_as_one(self):
        self.assertEqual(self._run({"OLLAMA_MAX_LOADED_MODELS": "lots"}),
                         [("OLLAMA_MAX_LOADED_MODELS", "1")])


# ════════════════════════════════════════════════════════════════════════════
#  6. [turn-timing] carries the re-prime verdict
# ════════════════════════════════════════════════════════════════════════════
class TurnLineReprimeVerdictTests(_Base):
    def test_the_owner_calls_verdict_is_on_the_line(self):
        bc = self.bc
        lines = []
        tt = bc._tt_mod.TurnTiming(print_fn=lines.append)
        self._p(bc, "_turn_timing", tt)
        payload = {"model": "m", "messages": [{"role": "system",
                                               "content": "s"}],
                   "options": {"num_ctx": 16384}, "keep_alive": "24h"}
        bc._reprime_prefix_hash[0] = bc._local_prefix_hash(payload,
                                                           drop_last=False)
        bc._reprime_posts_mark[0] = bc._lt.TRACKER.posts
        turn = dict(payload, messages=payload["messages"]
                    + [{"role": "user", "content": "hi"}])
        tt.begin("typed")
        bc._owner_chat_call.active = True
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(bc._reprime_check_stale(turn))
        finally:
            bc._owner_chat_call.active = False
        d = bc._tt_mod.parse_line(tt.emit())
        self.assertEqual(d["reprime"], "hit")


if __name__ == "__main__":
    unittest.main()
