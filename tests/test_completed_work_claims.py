"""Completed-work stock lines and the "calculate" claim family (NEW #9,
2026-10-01).

THE LIVE TURN (20:58:34-36): "Jarvis, is there any better models that you
can use that makes you smarter but doesn't require extra resources?" ->
"I've run the calculations, sir — it's a bit of a paradox. …" No action ran
and list_models was never called. The line came straight from the persona's
stock "Status / working" phrases ("I've run the calculations, sir." /
"Cross-referencing now."), which lived in THREE copies - core/prompts.py,
mcu_phrases.MCU_PHRASES["status"] and core/persona.JARVIS_SIGNATURE_PHRASES -
and claim_validator had no family for calculating, so nothing caught it.

Pinned here:
  * no copy of the phrase pools offers a completed-work claim any more, and
    the work-in-progress lines that remain are allowed only beside the
    [ACTION:] doing the work;
  * claim_validator flags "I've run the calculations" / "cross-referencing" /
    "ran the numbers" when nothing ran, grounds them once any action ran this
    turn, and leaves JARVIS's own arithmetic (a figure in the reply) and the
    noun uses ("computing power has doubled") alone;
  * the prompt maps "a better model you can run" to [ACTION: list_models],
    and the slim router keeps that example for the live sentence.

stdlib unittest only; light tier.
"""
from __future__ import annotations

import unittest

import mcu_phrases
from core import claim_validator as cv
from core import persona, prompt_router, prompts

_LIVE_USER = ("Jarvis, is there any better models that you can use that "
              "makes you smarter but doesn't require extra resources?")
_LIVE_REPLY = (
    "[intent:briefing] I've run the calculations, sir — it's a bit of a "
    "paradox. To get more intelligence, we usually have to trade off speed "
    "or memory, but I can certainly try a different model from your local "
    "library if you'd like to test the waters.")

_COMPLETED_WORK = ("I've run the calculations", "Cross-referencing")


class PhrasePoolTests(unittest.TestCase):
    def test_no_pool_offers_a_completed_work_claim(self):
        pools = {
            "BASE_SYSTEM_PROMPT": prompts.BASE_SYSTEM_PROMPT,
            "MCU_PHRASES": " / ".join(
                p for ps in mcu_phrases.MCU_PHRASES.values() for p in ps),
            "JARVIS_SIGNATURE_PHRASES": " / ".join(
                persona.JARVIS_SIGNATURE_PHRASES),
            "phrasebook block": mcu_phrases.render_phrasebook_block(),
        }
        for where, text in pools.items():
            for line in _COMPLETED_WORK:
                with self.subTest(where=where, line=line):
                    self.assertNotIn(line, text)

    def test_the_signature_pool_keeps_its_shape(self):
        # core/persona.py: "the count (20) and the order matter".
        self.assertEqual(len(persona.JARVIS_SIGNATURE_PHRASES), 20)

    def test_work_in_progress_lines_need_their_action(self):
        base = prompts.BASE_SYSTEM_PROMPT
        start = base.index("Status / working")
        status = base[start:base.index("\n", start)]
        self.assertIn("[ACTION:", status)
        self.assertIn("never as a stand-in for an answer", status)


class CalculateFamilyTests(unittest.TestCase):
    def test_the_live_reply_is_an_unverified_claim(self):
        got = cv.find_unverified_claim(_LIVE_REPLY, user_text=_LIVE_USER)
        self.assertIsNotNone(got)
        self.assertIn("run the calculations", got)

    def test_other_completed_work_phrasings(self):
        for text in ("Cross-referencing now.",
                     "Calculating, sir.",
                     "I've crunched the numbers, sir — the 32B is your best bet.",
                     "I ran the numbers and the fast model wins.",
                     "I'm running the numbers now, sir.",
                     "I've cross-referenced your library, sir."):
            with self.subTest(text=text):
                self.assertIsNotNone(cv.find_unverified_claim(
                    text, user_text=_LIVE_USER))

    def test_grounded_once_an_action_ran_this_turn(self):
        self.assertIsNone(cv.find_unverified_claim(
            _LIVE_REPLY, ran_actions={"list_models"}, user_text=_LIVE_USER))

    def test_own_arithmetic_with_the_figure_is_not_a_claim(self):
        for text in ("Let me calculate that: fifteen percent of eighty is "
                     "twelve, sir.",
                     "I've calculated it, sir: 391.",
                     "Calculating: 100 °F is 37.8 °C, sir."):
            with self.subTest(text=text):
                self.assertIsNone(cv.find_unverified_claim(
                    text, user_text="what's fifteen percent of eighty"))

    def test_noun_uses_are_not_claims(self):
        for text in ("Computing power has doubled every two years, sir.",
                     "Calculated risks are rather your speciality, sir.",
                     "Let me double-check my calculations, sir."):
            with self.subTest(text=text):
                self.assertIsNone(cv.find_unverified_claim(text))


class ListModelsExampleTests(unittest.TestCase):
    def test_the_prompt_maps_a_better_model_to_list_models(self):
        self.assertIn("'is there a better model you can run?'",
                      prompts.PC_CONTROL_PROMPT)
        i = prompts.PC_CONTROL_PROMPT.index(
            "'is there a better model you can run?'")
        line = prompts.PC_CONTROL_PROMPT[i:prompts.PC_CONTROL_PROMPT.index(
            "\n", i)]
        self.assertIn("[ACTION: list_models]", line)

    def test_the_slim_router_keeps_it_for_the_live_sentence(self):
        slim = prompt_router.slim_pc_control(_LIVE_USER,
                                             prompts.PC_CONTROL_PROMPT)
        self.assertIn("'is there a better model you can run?'", slim)


if __name__ == "__main__":
    unittest.main()
