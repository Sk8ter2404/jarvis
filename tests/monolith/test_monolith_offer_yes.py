"""A yes to JARVIS's own offer carries the offer out (2026-10-02 live).

Live 14:48:08 a turn ended on JARVIS offering to move an app's existing
window to the top monitor. At 14:48:26 the owner said "Jarvis, yes." and the
brain answered with a quip ("A bold choice, sir ...") and ran nothing: the
dispatcher's confirmation queue was empty (it only holds actions IT deferred),
so nothing knew the yes answered an offer, and the turn reached the model as
a bare "yes".

The fix records the offer a finished turn ENDS with (_note_open_offer); the
next owner turn takes it (_take_open_offer), and a clear, prompt yes makes
_call_llm route on the offer's words and tell the brain to emit the offered
action. These drive the REAL _run_llm_dispatch -> _call_llm (local, cache-
stable layout) -> parse_and_run_actions with stub actions, a recording
_speak and a fake brain that acts on an offer ONLY when the prompt it gets
says the owner accepted it - so a green test proves the note reaches the
model. Pinned: offer + yes runs it once; no, an unrelated turn, a late yes,
a hedged yes and a yes after another turn do nothing; the self-termination
and confirmation gates still hold the offered action. Paraphrased fixtures;
no LLM, no audio, no real windows.

    python -m unittest tests.monolith.test_monolith_offer_yes
"""
from __future__ import annotations

import contextlib
import io
import unittest

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

# Paraphrases of the live lines.
OFFER_REPLY = ("[intent:dry_wit] The app seems to be running already, sir. "
               "Should I try moving its window to the top monitor?")
OFFER = "Should I try moving its window to the top monitor?"
MOVE = "[intent:confirmation] Right away, sir. " \
       "[ACTION: move_window_to_monitor, Example Chat | top]"
QUIP = "[intent:amused] A daring choice, sir. Standing by."


@requires_monolith
class _OfferBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        import core.config as cfg
        self._p(cfg, "model_route", return_value="local")
        self._p(bc, "_DYNAMIC_LOCAL_PROMPT", True)
        self._p(bc, "_STABLE_LOCAL_PREFIX", True)
        self._p(bc, "_RESOLVED_LOCAL_LLM_MODEL", ["gemma-test"])
        self._p(bc, "_get_local_llm_model", return_value="gemma-test")
        self._p(bc, "_ltm_context", return_value="")
        self._p(bc, "_ltm_enqueue")
        self._p(bc, "load_memory", return_value=bc._empty_memory())
        self._p(bc, "save_memory")
        self._p(bc, "_voice_mood_response", None)
        self._p(bc, "_phrase_rotation_last", [{}])
        self._p(bc, "_system_prompt",
                "BASE IDENTITY\n" + bc.PC_CONTROL_PROMPT + "\n")
        self._p(bc, "_turn_check_after_chain", return_value=None)
        # The owner-turn stamps and the open offer on ONE test clock, which
        # moves only when the brain "thinks".
        self.now = [1000.0]
        self.prev, self.last = [0.0], [0.0]
        self._p(bc, "_prev_owner_turn_at", self.prev)
        self._p(bc, "_last_owner_turn_at", self.last)
        self._p(bc, "_open_offer", bc._offer_reply.OpenOffer(
            bc.OFFER_YES_TTL_S, clock=lambda: self.now[0]))
        self._p(bc, "_accepted_offer", [""])
        self._p(bc, "_offer_ledger", [])
        self._keep(bc.conversation_history)
        self._keep(bc._pending_confirmation)
        self._keep(bc._pending_autocorrect_choice)
        self._p(bc, "_pending_confirmation_at", [0.0])
        # The brain: the REAL _call_llm, a fake model behind it.
        self.prompts: list[str] = []
        self.brain = None
        self._p(bc, "_local_then_cloud_or_honest", side_effect=self._model)
        self._p(bc, "get_response_with_animation", side_effect=bc._call_llm)
        self.followups: list[str] = []
        self.gfr = self._p(bc, "get_followup_response",
                           side_effect=lambda _info: (self.followups.pop(0)
                                                      if self.followups
                                                      else None))
        self._stub("move_window_to_monitor",
                   "moved 'Example Chat' to top monitor")

    def _keep(self, cell):
        saved = list(cell)
        cell.clear()
        self.addCleanup(lambda: (cell.clear(), cell.extend(saved)))

    def _model(self, _system, messages):
        self.now[0] += 1.0                        # the model takes a moment
        last = messages[-1]["content"]
        self.prompts.append(last)
        return self.brain(last)

    @staticmethod
    def obedient(offer_reply=OFFER_REPLY, act=MOVE, offer=OFFER):
        """Opens with ``offer_reply``; afterwards carries the offer out only
        when its prompt says the owner accepted ``offer``, else quips (what
        the live brain did)."""
        state = {"n": 0}

        def brain(prompt):
            state["n"] += 1
            if state["n"] == 1:
                return offer_reply
            if "OFFER ACCEPTED" in prompt and offer in prompt:
                return act
            return QUIP
        return brain

    def _turn(self, text, after_s=6.0):
        """One owner turn in the main loop's order: the clock moves, the turn
        is stamped (_note_owner_turn), then dispatched."""
        self.now[0] += after_s
        self.prev[0], self.last[0] = self.last[0], self.now[0]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.bc._run_llm_dispatch(text)
        return out.getvalue()


class OfferYesTests(_OfferBase):
    def test_offer_then_yes_runs_the_offered_action_once(self):
        self.brain = self.obedient()
        printed = self._turn("Jarvis, open the chat app on the top monitor.")
        self.assertIn("[offer-yes] open offer:", printed)
        self.assertEqual(self.calls["move_window_to_monitor"], [])
        printed = self._turn("Jarvis, yes.")
        self.assertIn("[offer-yes] sir said yes to", printed)
        self.assertEqual(self.calls["move_window_to_monitor"],
                         ["Example Chat | top"])
        # The note rode the turn's context, offer quoted, owner words last.
        self.assertIn("OFFER ACCEPTED", self.prompts[1])
        self.assertIn(OFFER, self.prompts[1])
        self.assertTrue(self.prompts[1].endswith("Jarvis, yes."))
        # ... and only that turn's: the owner's words in the history are his.
        self.assertIn({"role": "user", "content": "Jarvis, yes."},
                      self.bc.conversation_history)
        self.assertFalse(any("OFFER ACCEPTED" in str(m.get("content"))
                             for m in self.bc.conversation_history))
        self.assertEqual(self.bc._accepted_offer, [""])
        # A second yes has no offer left to answer: nothing runs again.
        self._turn("Jarvis, yes.")
        self.assertNotIn("OFFER ACCEPTED", self.prompts[2])
        self.assertEqual(len(self.calls["move_window_to_monitor"]), 1)

    def test_the_live_shape_an_offer_from_the_follow_up_round(self):
        # The action failed, and the follow-up round made the offer.
        self._stub("open_on_monitor", "no window matching 'Example Chat'")
        self.followups = [OFFER_REPLY]
        self.brain = self.obedient(
            offer_reply="Very good, sir. "
                        "[ACTION: open_on_monitor, top | Example Chat]")
        self._turn("Jarvis, open the chat app on the top monitor.")
        self.assertEqual(self.gfr.call_count, 1)
        self._turn("Jarvis, yes.")
        self.assertEqual(self.calls["move_window_to_monitor"],
                         ["Example Chat | top"])

    def test_without_the_fix_the_same_brain_only_quips(self):
        # Mutation guard: with the note suppressed the fake brain does what
        # the live one did, so the test above is green for the right reason.
        self.brain = self.obedient()
        self._turn("Jarvis, open the chat app on the top monitor.")
        self.bc._open_offer.clear()
        printed = self._turn("Jarvis, yes.")
        self.assertEqual(self.calls["move_window_to_monitor"], [])
        self.assertIn("A daring choice, sir.", " ".join(self.spoken))
        self.assertNotIn("[offer-yes] sir said yes", printed)

    def test_a_brain_that_still_does_nothing_is_logged(self):
        self.brain = lambda p: (OFFER_REPLY if len(self.prompts) == 1
                                else QUIP)
        self._turn("Jarvis, open the chat app on the top monitor.")
        printed = self._turn("Jarvis, yes.")
        self.assertIn("[offer-yes] the reply to the accepted offer named no "
                      "action", printed)


class NotAYesTests(_OfferBase):
    def _offer_then(self, answer, after_s=6.0):
        self.brain = self.obedient()
        self._turn("Jarvis, open the chat app on the top monitor.")
        printed = self._turn(answer, after_s=after_s)
        self.assertEqual(self.calls["move_window_to_monitor"], [], answer)
        self.assertNotIn("OFFER ACCEPTED", self.prompts[-1])
        return printed

    def test_offer_then_no_does_nothing(self):
        printed = self._offer_then("Jarvis, no.")
        self.assertIn("[offer-yes] not a clear yes", printed)

    def test_a_hedged_yes_is_not_a_yes(self):
        for answer in ("Jarvis, yes, but later.", "Yes, but not now."):
            with self.subTest(answer=answer):
                printed = self._offer_then(answer)
                self.assertIn("not a clear yes", printed)

    def test_an_unrelated_sentence_is_a_normal_turn_and_closes_the_offer(self):
        self._offer_then("Jarvis, tell me something about octopuses.")
        self.assertEqual(len(self.prompts), 2)
        self.assertTrue(self.prompts[1].endswith(
            "Jarvis, tell me something about octopuses."))
        # The offer is gone: a yes now answers nothing.
        self._turn("Jarvis, yes.")
        self.assertNotIn("OFFER ACCEPTED", self.prompts[2])
        self.assertEqual(self.calls["move_window_to_monitor"], [])

    def test_a_yes_long_after_the_offer_is_not_a_confirmation(self):
        printed = self._offer_then("Jarvis, yes.",
                                   after_s=self.bc.OFFER_YES_TTL_S + 5.0)
        self.assertIn("[offer-yes] the offer is more than 120 s old", printed)

    def test_a_yes_after_another_owner_turn_does_not_answer_it(self):
        self.brain = self.obedient()
        self._turn("Jarvis, open the chat app on the top monitor.")
        # A turn the main loop handled before the LLM (a voice shortcut).
        self.now[0] += 3.0
        self.prev[0], self.last[0] = self.last[0], self.now[0]
        printed = self._turn("Jarvis, yes.")
        self.assertIn("another turn came after the offer", printed)
        self.assertEqual(self.calls["move_window_to_monitor"], [])

    def test_an_offer_of_nothing_in_particular_is_not_opened(self):
        self.brain = self.obedient(
            offer_reply="Would you like to hear more about it, sir?")
        printed = self._turn("Jarvis, tell me about the chat app.")
        self.assertNotIn("[offer-yes] open offer", printed)
        self._turn("Jarvis, yes.")
        self.assertNotIn("OFFER ACCEPTED", self.prompts[1])


class GatesStillApplyTests(_OfferBase):
    def test_an_offered_self_termination_still_needs_the_owners_words(self):
        bc = self.bc
        self._stub("restart", "restarting")
        self.brain = self.obedient(
            offer_reply="The update is staged, sir. Shall I restart myself "
                        "to apply it?",
            act="Restarting, sir. [ACTION: restart]",
            offer="Shall I restart myself to apply it?")
        self._turn("Jarvis, is the update ready?")
        printed = self._turn("Jarvis, yes.")
        self.assertIn("OFFER ACCEPTED", self.prompts[1])
        self.assertEqual(self.calls["restart"], [], "ran on a bare yes")
        self.assertEqual(bc._pending_confirmation, [("restart", "")])
        self.assertIn("[self-term]", printed)
        # The held action is the turn's question now, not the offer.
        self.assertEqual(bc._open_offer.peek(), "")

    def test_an_offered_confirm_gated_action_still_asks(self):
        bc = self.bc
        self._stub("delete_file", "deleted")
        self._p(bc, "_needs_confirmation", lambda n, a: n == "delete_file")
        self.brain = self.obedient(
            offer_reply="That folder holds old exports, sir. Shall I delete "
                        "the oldest one?",
            act="[ACTION: delete_file, oldest export]",
            offer="Shall I delete the oldest one?")
        self._turn("Jarvis, what's in the exports folder?")
        self._turn("Jarvis, yes.")
        self.assertEqual(self.calls["delete_file"], [])
        self.assertEqual(bc._pending_confirmation,
                         [("delete_file", "oldest export")])
        self.assertTrue(any("confirmation" in s for s in self.spoken),
                        self.spoken)


if __name__ == "__main__":
    unittest.main()
