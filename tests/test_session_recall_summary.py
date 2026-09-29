"""session_memory_recall summarises THIS session's conversation on request.

Live, v2.0.115 (2026-09-29, typed turn): "summarize what we talked about
today" -> no action, and the reply "I'm afraid I've no way to check that, sir —
I can only recall specific past conversations if you ask me about them
directly." A false decline: the session's conversation_history was in memory
the whole time, and session_memory_recall existed — but it only ever queried
the PRIOR-session summary index, its prompt section described it that way, and
'recap our conversation' / 'what have we discussed' did not even load it on the
local (slimmed) prompt path.

Pinned here:
  * summary mode — decided from the argument OR the owner's own words for this
    turn — summarises conversation_history through the same bc._llm_quick path
    the recall already used, with this request itself left out;
  * an honest "we've only just started" when there is nothing before it;
  * questions about an EARLIER window stay on the summary index;
  * the three phrasings reach the local model with the action name.

CI-faithful: the monolith is a Mock handed back through core.actions._bc();
no LLM, network or file is touched.
"""
from __future__ import annotations

import types
import unittest
from unittest import mock

import core.actions as A
from core.owner_turn import current_owner_utterance

_REQ = "summarize what we talked about today"
_REPLY = "Certainly, sir. [ACTION: session_memory_recall, " + _REQ + "]"


def _bc(history=None, utterance=None, in_turn=True, today_summaries=None,
        llm="You set a tea timer and asked about the printer, sir."):
    bc = mock.Mock()
    bc.conversation_history = list(history or [])
    bc._turn_in_progress = [bool(in_turn)]
    bc._last_user_text = [utterance]
    bc.pattern_memory.get_session_summaries.return_value = list(
        today_summaries or [])
    bc.pattern_memory.describe_window.return_value = "from that period"
    bc._llm_quick.return_value = llm
    return bc


def _session(*pairs, request=_REQ, reply=_REPLY):
    """A conversation_history as it stands WHEN the action runs: prior turns,
    then this turn's request and the assistant reply carrying the token."""
    h = [{"role": "assistant", "content": "Good afternoon, sir."}]  # boot line
    for u, a in pairs:
        h += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return h + [{"role": "user", "content": request},
                {"role": "assistant", "content": reply}]


def _run(bc, args=""):
    with mock.patch.object(A, "_bc", return_value=bc):
        return A._act_session_memory_recall(args)


class SummaryModeTests(unittest.TestCase):
    def test_bare_token_on_the_live_phrase_summarises_this_session(self):
        """The exact 2026-09-29 question, emitted as a bare token."""
        bc = _bc(_session(("set a tea timer for five minutes",
                           "Timer set, sir. [ACTION: set_timer, 5 minutes | tea]"),
                          ("how's the print going", "Layer 40 of 212, sir.")),
                 utterance=_REQ)
        out = _run(bc, "")
        self.assertEqual(out, "You set a tea timer and asked about the printer, sir.")
        bc._llm_quick.assert_called_once()
        user = bc._llm_quick.call_args.kwargs["user"]
        self.assertIn("User: set a tea timer for five minutes", user)
        self.assertIn("Assistant: Layer 40 of 212, sir.", user)
        # Action tokens are noise to the summariser, and THIS request is not
        # part of what is being summarised.
        self.assertNotIn("[ACTION:", user)
        self.assertNotIn(_REQ, user)
        # The prior-session index was not asked to answer this question.
        for call in bc.pattern_memory.get_session_summaries.call_args_list:
            self.assertEqual(call.args[0], "today")

    def test_argument_alone_selects_summary_mode(self):
        bc = _bc(_session(("open chrome", "Opening Chrome, sir."),
                          request="recap our conversation"),
                 utterance=None, in_turn=False)
        _run(bc, "recap our conversation")
        user = bc._llm_quick.call_args.kwargs["user"]
        self.assertIn("User: open chrome", user)

    def test_what_have_we_discussed(self):
        bc = _bc(_session(("what's the weather", "Currently 70, sir."),
                          request="what have we discussed"),
                 utterance="what have we discussed")
        _run(bc, "")
        self.assertIn("User: what's the weather",
                      bc._llm_quick.call_args.kwargs["user"])

    def test_session_just_started_is_said_honestly(self):
        bc = _bc(_session(), utterance=_REQ)
        out = _run(bc, _REQ)
        self.assertEqual(out, "We've only just started this session, sir — "
                              "there's nothing to summarise yet.")
        bc._llm_quick.assert_not_called()

    def test_earlier_sessions_today_are_folded_in(self):
        """A restart empties conversation_history but not his day."""
        bc = _bc(_session(), utterance=_REQ, today_summaries=[
            {"date": "2026-09-29", "summary": "Tuned the printer's first layer."}])
        _run(bc, "")
        user = bc._llm_quick.call_args.kwargs["user"]
        self.assertIn("Tuned the printer's first layer.", user)
        bc.pattern_memory.get_session_summaries.assert_called_once_with(
            "today", limit=8)

    def test_llm_failure_and_empty_reply_are_honest(self):
        bc = _bc(_session(("hi", "Hello, sir.")), utterance=_REQ)
        bc._llm_quick.side_effect = RuntimeError("local model down")
        self.assertEqual(_run(bc, ""),
                         "conversation summary LLM call failed: local model down")
        bc = _bc(_session(("hi", "Hello, sir.")), utterance=_REQ, llm="  ")
        self.assertEqual(_run(bc, ""), "I couldn't produce a summary of our "
                                       "conversation just now, sir.")

    def test_malformed_history_is_tolerated(self):
        bc = _bc(utterance=_REQ)
        bc.conversation_history = [None, "junk", {"role": "user"},
                                   {"role": "user", "content": "open notepad"},
                                   {"role": "assistant", "content": "Done, sir."}]
        _run(bc, "")
        self.assertIn("User: open notepad", bc._llm_quick.call_args.kwargs["user"])


class IndexModeStaysTests(unittest.TestCase):
    def test_earlier_window_uses_the_session_index(self):
        bc = _bc(_session(), utterance="what did we talk about yesterday")
        bc.pattern_memory.get_session_summaries.return_value = [
            {"date": "2026-09-28", "summary": "Worked on the garden planner."}]
        _run(bc, "what did we talk about yesterday")
        bc.pattern_memory.get_session_summaries.assert_called_once_with(
            "what did we talk about yesterday", limit=8)
        self.assertIn("recalling what the user",
                      bc._llm_quick.call_args.kwargs["system"])

    def test_bare_token_takes_the_time_reference_from_his_words(self):
        bc = _bc(utterance="what was I working on last night")
        _run(bc, "")
        bc.pattern_memory.get_session_summaries.assert_called_once_with(
            "what was I working on last night", limit=8)

    def test_stale_utterance_outside_a_turn_is_ignored(self):
        bc = _bc(_session(("hi", "Hello, sir.")), utterance=_REQ, in_turn=False)
        _run(bc, "what did we do yesterday")
        bc.pattern_memory.get_session_summaries.assert_called_once_with(
            "what did we do yesterday", limit=8)


class DetectionTests(unittest.TestCase):
    def test_summary_phrasings(self):
        for t in ("summarize what we talked about today",
                  "summarise what we talked about", "recap our conversation",
                  "what have we discussed", "what did we talk about earlier",
                  "give me a summary of our conversation",
                  "can you sum up what we've covered"):
            self.assertTrue(A._conversation_summary_requested(t), t)

    def test_not_summary_requests(self):
        for t in ("", "what did we do yesterday", "recap my day",
                  "summarize what we talked about yesterday",
                  "what did we discuss last week", "open chrome"):
            self.assertFalse(A._conversation_summary_requested(t), t)


class RoutingTests(unittest.TestCase):
    """The three phrasings must ship session_memory_recall to the local model,
    with the prompt saying it covers THIS conversation."""

    def test_phrasings_reach_the_local_model(self):
        from core import prompt_router, prompts
        for utt in ("summarize what we talked about today",
                    "recap our conversation", "what have we discussed"):
            slim = prompt_router.slim_pc_control(utt, prompts.PC_CONTROL_PROMPT)
            self.assertIn(f"'{utt}' → [ACTION: session_memory_recall]", slim, utt)

    def test_prompt_says_it_covers_this_session(self):
        from core import prompts
        self.assertIn("summarises THIS session's conversation",
                      prompts.PC_CONTROL_PROMPT)


class OwnerTurnTests(unittest.TestCase):
    """core.owner_turn — the one rule for reading his words."""

    def _mod(self, text, turn):
        m = types.ModuleType("bobert_companion")
        m._last_user_text = text
        m._turn_in_progress = turn
        return m

    def test_returns_the_utterance_only_during_a_turn(self):
        self.assertEqual(current_owner_utterance(
            self._mod([" will it rain tomorrow "], [True])), "will it rain tomorrow")
        self.assertEqual(current_owner_utterance(
            self._mod(["will it rain tomorrow"], [False])), "")

    def test_strict_shapes(self):
        for text, turn in ((None, [True]), ([None], [True]), ([], [True]),
                           (["x"], None), (["x"], [1]), (["x"], True)):
            self.assertEqual(current_owner_utterance(self._mod(text, turn)), "",
                             (text, turn))
        self.assertEqual(current_owner_utterance(mock.Mock()), "")

    def test_no_monolith_loaded(self):
        with mock.patch.dict("sys.modules", {"bobert_companion": None}):
            self.assertEqual(current_owner_utterance(), "")


if __name__ == "__main__":
    unittest.main()
