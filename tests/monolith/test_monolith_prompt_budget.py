"""The local prompt budget, driven through the REAL call paths (2026-10-01).

WHY
===
Ollama silently truncates a prompt longer than num_ctx to its first few tokens
plus the tail. Two live local turns on 2026-10-01 read prompt_eval_count=8195:
a ~51k-character system prompt, 12-14k characters of routed PC sections and a
full history added up to ~17.5k tokens, and the model answered with most of
its system prompt cut away. A 'web' brain-eval prompt hit the same wall.

WHAT THESE TESTS PIN
====================
Each test builds a synthetic oversized prompt and captures the /api/chat POST
(or the last hop before the model). The prompt that goes out must:
  * fit the window (estimated with core.prompt_budget, reply room reserved);
  * keep the system prompt byte-identical and the current message intact;
  * shed the OLDEST history first, then whole per-turn sections;
  * say so in one ``[prompt-budget]`` line;
and the live conversation_history must be left untouched. A prompt that fits
must go out exactly as it did before, and JARVIS_LOCAL_PROMPT_BUDGET=0 must
restore the old behaviour.

Before the budget existed the oversized POSTs below carried every byte, so the
"fits the window" assertions fail on that code.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests._monolith_harness import (  # noqa: E402
    MonolithGlobalsTestCase, requires_monolith,
)
from core import prompt_budget as pb  # noqa: E402
from core import prompt_router as pr  # noqa: E402


class _Resp:
    ok = True
    status_code = 200
    text = "body"

    def json(self):
        return {"model": "gemma-test",
                "message": {"role": "assistant", "content": "Done, sir."},
                "done": True, "prompt_eval_count": 1234, "eval_count": 3}


# A turn that routes many large PC sections, like the live overflow.
_MANY = "kinect depth, the 3d printer, netflix and hulu, browser agent, " \
        "queue a task"
# The stable system prompt the live overflow turns carried (sys_chars).
_LIVE_SYS_CHARS = 49482


def _long_history(pairs=10, n=1200):
    out = []
    for i in range(pairs):
        out.append({"role": "user",
                    "content": f"h-u{i} " + ("tell me more about it " * n)[:n]})
        out.append({"role": "assistant",
                    "content": f"h-a{i} " + ("certainly sir, here it is "
                                             * n)[:n]})
    return out


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        import core.config as cfg
        self._p(cfg, "model_route", return_value="local")
        self._p(bc, "_DYNAMIC_LOCAL_PROMPT", True)
        self._p(bc, "_STABLE_LOCAL_PREFIX", True)
        self._p(bc, "_RESOLVED_LOCAL_LLM_MODEL", ["gemma-test"])
        # Quiet the turn: no disk, no LTM, no lights.
        self._p(bc, "_ltm_context", return_value="")
        self._p(bc, "_ltm_enqueue")
        self._p(bc, "load_memory", return_value=bc._empty_memory())
        self._p(bc, "save_memory")
        self._p(bc, "_voice_mood_response", None)
        bc._phrase_rotation_last[0] = {}
        # Every /api/chat POST is recorded, never sent.
        self.posted = []
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None, **_k):
            self.posted.append(json)
            return _Resp()
        fake_req.post.side_effect = _post
        self._p(bc, "LOCAL_LLM_FALLBACK", True)
        self._p(bc, "_ollama_alive", return_value=True)
        self._p(bc, "_ollama_has_model", return_value=True)
        self._p(bc, "_get_local_llm_model", return_value="gemma-test")
        self._p(bc, "_next_local_llm_fallback", return_value=None)
        self._p(bc, "requests", fake_req)
        self._big_system_prompt()

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _big_system_prompt(self, stable_chars=_LIVE_SYS_CHARS):
        """A _system_prompt whose cache-stable local form is `stable_chars`
        long, the size the live overflow turns carried."""
        bc = self.bc
        pc = bc.PC_CONTROL_PROMPT
        self.assertTrue(pc)
        tail = "\n\nWhat you know about your owner:\n- likes tea"
        bare = bc._local_stable_system_prompt("BASE IDENTITY\n" + pc + tail)
        pad = max(0, stable_chars - len(bare))
        filler = ("You are calm, precise and brief. " * (pad // 30 + 1))[:pad]
        self._p(bc, "_system_prompt", "BASE IDENTITY\n" + filler + pc + tail)
        self.stable = bc._local_stable_system_prompt()
        self.assertEqual(len(self.stable), max(stable_chars, len(bare)))

    def _est(self, payload):
        msgs = payload["messages"]
        return pb.estimate_chat_tokens(msgs[0]["content"], msgs[1:])

    def _budget(self, max_tokens=500):
        return pb.budget_for(self.bc._local_num_ctx("gemma-test"), max_tokens)

    def _turn(self, text, history):
        bc = self.bc
        bc.conversation_history[:] = [dict(m) for m in history]
        bc._phrase_rotation_last[0] = {}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc._call_llm(text)
        self.assertEqual(len(self.posted), 1)
        return self.posted[0], out.getvalue()

    @staticmethod
    def _notes(stdout):
        return [ln for ln in stdout.splitlines() if "[prompt-budget]" in ln]


class OversizedTurnTests(_Base):
    def test_overflowing_turn_is_fitted_to_the_window(self):
        block = pr.turn_pc_block(_MANY, self.bc.PC_CONTROL_PROMPT)
        self.assertGreater(len(block), 12000, "precondition: a big turn")
        # 7 exchanges: under the 20-message cap, so the chunked history trim
        # leaves it alone and every drop below is the budget's.
        history = _long_history(7)
        payload, stdout = self._turn(_MANY, history)
        # 1. It fits (before the budget this POST was ~26.3k estimated tokens
        #    against a 15,884 budget).
        self.assertLessEqual(self._est(payload), self._budget())
        # 2. The system prompt is the cache-stable one, byte for byte.
        self.assertEqual(payload["messages"][0]["content"],
                         self.stable + self.bc._LOCAL_MODE_DIRECTIVE)
        # 3. The user's words are intact and last.
        last = payload["messages"][-1]
        self.assertEqual(last["role"], "user")
        self.assertTrue(last["content"].endswith(_MANY))
        # 4. Oldest history went first; the newest exchanges survived, in
        #    order.
        names = [m["content"].split(" ")[0] for m in history]
        sent = [m["content"].split(" ")[0] for m in payload["messages"][1:-1]]
        self.assertLess(len(sent), len(names))
        self.assertGreaterEqual(len(sent), 2)
        self.assertEqual(sent, names[len(names) - len(sent):])
        # 5. Whole sections went, the largest first; what is left is whole.
        sections = dict(pr.split_turn_block(block))
        self.assertNotIn(sections["TASK QUEUE"], last["content"])
        kept = [h for h, t in sections.items() if t in last["content"]]
        self.assertTrue(kept, "every section went; the smallest fit")
        self.assertIn(self.bc._TURN_CTX_OPEN, last["content"])
        # 6. One line says what happened.
        notes = self._notes(stdout)
        self.assertEqual(len(notes), 1, stdout)
        self.assertIn("[prompt-budget] turn:", notes[0])
        self.assertIn("oldest history msg(s)", notes[0])
        self.assertIn("TASK QUEUE", notes[0])
        # 7. The live history itself is untouched: the trim is per call.
        self.assertEqual(self.bc.conversation_history[:len(history)],
                         history)

    def test_history_alone_overflowing_keeps_every_section(self):
        # Moderate sections; a long history is what overflows.
        self._big_system_prompt(40000)
        text = "play some music"
        block = pr.turn_pc_block(text, self.bc.PC_CONTROL_PROMPT)
        payload, stdout = self._turn(text, _long_history(9, n=2400))
        self.assertLessEqual(self._est(payload), self._budget())
        last = payload["messages"][-1]["content"]
        self.assertIn(block, last, "a section was dropped while history "
                                   "could still be trimmed")
        notes = self._notes(stdout)
        self.assertEqual(len(notes), 1, stdout)
        self.assertNotIn("turn part(s)", notes[0])

    def test_a_turn_that_fits_is_sent_exactly_as_before(self):
        history = _long_history(3, n=200)
        payload_on, out_on = self._turn("play some music", history)
        self.posted.clear()
        self._p(self.bc, "_LOCAL_PROMPT_BUDGET", False)
        payload_off, _out = self._turn("play some music", history)
        self.assertEqual(payload_on, payload_off)
        self.assertEqual(self._notes(out_on), [])

    def test_parts_join_to_the_exact_turn_context(self):
        bc = self.bc
        addenda = [("tone", "\n\nTONE", pb.RANK_REGISTER),
                   ("empty", "", pb.RANK_MEMORY),
                   ("phrase rotation", "\n\nHINT", pb.RANK_STYLE_HINT)]
        for text in (_MANY, "play some music", "what time is it"):
            with self.subTest(text=text):
                block = pr.turn_pc_block(text, bc.PC_CONTROL_PROMPT)
                parts = bc._turn_budget_parts(block, addenda)
                self.assertEqual("".join(p.text for p in parts),
                                 block + "\n\nTONE\n\nHINT")
                self.assertNotIn("empty", [p.label for p in parts])

    def test_a_prompt_that_cannot_fit_is_sent_unchanged(self):
        # A 30B-class tag gets the 12k window, which the live-size system
        # prompt alone overflows. Trimming cannot fix that, and Ollama keeps
        # the TAIL (history + turn context), so nothing is thrown away.
        bc = self.bc
        self._p(bc, "_RESOLVED_LOCAL_LLM_MODEL", ["big-test:32b"])
        self._p(bc, "_get_local_llm_model", return_value="big-test:32b")
        self.assertEqual(bc._local_num_ctx("big-test:32b"), 12288)
        history = _long_history(3, n=300)
        payload_on, out_on = self._turn("play some music", history)
        self.posted.clear()
        self._p(bc, "_LOCAL_PROMPT_BUDGET", False)
        payload_off, _out = self._turn("play some music", history)
        self.assertEqual(payload_on, payload_off)
        notes = self._notes(out_on)
        self.assertEqual(len(notes), 1, out_on)
        self.assertIn("CANNOT FIT", notes[0])
        self.assertIn("(num_ctx 12288)", notes[0])

    def test_kill_switch_restores_the_unbudgeted_prompt(self):
        self._p(self.bc, "_LOCAL_PROMPT_BUDGET", False)
        payload, stdout = self._turn(_MANY, _long_history(7))
        self.assertGreater(self._est(payload), self._budget())
        self.assertIn("h-u0", payload["messages"][1]["content"])
        self.assertEqual(self._notes(stdout), [])


class OversizedFollowupTests(_Base):
    """get_followup_response reuses the primary turn's stable system prompt
    and re-sends its section bodies; its round must be fitted too."""

    def _followup(self, results):
        bc = self.bc
        seen = {}

        def _fake(sys_prompt, messages, **kw):
            seen["sys"], seen["msgs"], seen["kw"] = sys_prompt, messages, kw
            return "ok"
        self._p(bc, "_local_then_cloud_or_honest", _fake)
        self._p(bc, "_last_stable_sys_prompt", [self.stable])
        self._p(bc, "_last_turn_pc_block",
                [pr.turn_pc_block(_MANY, bc.PC_CONTROL_PROMPT)])
        bc.conversation_history[:] = _long_history() + [
            {"role": "user", "content": _MANY},
            {"role": "assistant", "content": "[ACTION: get_time] One moment."}]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc.get_followup_response(results)
        return seen, out.getvalue()

    def test_followup_round_is_fitted(self):
        seen, stdout = self._followup([("get_time", "3:14 PM")])
        sys_now, msgs = self.bc._local_chat_prompt(seen["sys"], seen["msgs"])
        self.assertLessEqual(pb.estimate_chat_tokens(sys_now, msgs),
                             self._budget(400))
        self.assertEqual(seen["sys"], self.stable)
        self.assertIn("[get_time] returned: 3:14 PM", msgs[-1]["content"])
        # The exchange the follow-up exists to finish is still there.
        self.assertEqual(msgs[-2]["content"], "[ACTION: get_time] One moment.")
        notes = self._notes(stdout)
        self.assertEqual(len(notes), 1, stdout)
        self.assertIn("[prompt-budget] follow-up:", notes[0])


class OtherLocalCallersTests(_Base):
    """Callers that hand _call_local_llm a raw history (the cloud-error
    fallback _local_fallback_or, background jobs) are fitted there."""

    def test_raw_history_is_fitted_in_call_local_llm(self):
        bc = self.bc
        system = "S" * 50000
        history = _long_history(10, n=2000) + [
            {"role": "user", "content": "what is the time"}]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc._call_local_llm(system, history)
        payload = self.posted[0]
        self.assertLessEqual(self._est(payload), self._budget())
        self.assertEqual(payload["messages"][0]["content"],
                         system + bc._LOCAL_MODE_DIRECTIVE)
        self.assertEqual(payload["messages"][-1]["content"],
                         "what is the time")
        self.assertEqual(len(history), 21, "the caller's list was mutated")
        notes = self._notes(out.getvalue())
        self.assertEqual(len(notes), 1, out.getvalue())
        self.assertIn("[prompt-budget] call:", notes[0])

    def test_a_budgeted_turn_is_not_measured_twice(self):
        # The turn path already fitted (and logged); _call_local_llm must
        # neither trim it again nor log a second line.
        payload, stdout = self._turn(_MANY, _long_history(7))
        self.assertEqual(len(self._notes(stdout)), 1, stdout)
        self.assertLessEqual(self._est(payload), self._budget())

    def test_a_small_background_prompt_is_untouched(self):
        bc = self.bc
        msgs = [{"role": "user", "content": "summarise: it is sunny"}]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc._call_local_llm("You summarise.", msgs, max_tokens=50)
        self.assertEqual(self.posted[0]["messages"][1:], msgs)
        self.assertEqual(self._notes(out.getvalue()), [])


if __name__ == "__main__":
    unittest.main()
