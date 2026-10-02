"""core/offer_reply.py - a yes to JARVIS's own offer (2026-10-02 live).

Live 14:48:08 a turn ended on JARVIS offering to move an app's window to
another monitor; at 14:48:26 the owner answered with a bare yes (wake word
first) and JARVIS answered with a quip and ran nothing. These pin the pure
half of the fix: which replies end on an offer a yes can carry out, the one
open offer and how an owner turn takes it (clear yes only, never late, never
after another turn), the note handed to the brain, and the prompt router's
``also`` routing that puts the offered action's section in front of it.
Paraphrased fixtures; no LLM, no audio.

    python -m unittest tests.test_offer_reply
"""
from __future__ import annotations

import unittest

from core import offer_reply as O
from core import prompt_router as pr
from core.prompts import PC_CONTROL_PROMPT

# The live shape, reworded.
_LIVE_OFFER = ("[intent:dry_wit] It seems the app may already be running, "
               "sir. Should I try moving the existing window to the top "
               "monitor?")


class OfferTextTests(unittest.TestCase):
    def test_the_live_reply_ends_on_an_offer(self):
        self.assertEqual(O.offer_text(_LIVE_OFFER),
                         "Should I try moving the existing window to the top "
                         "monitor?")

    def test_offers_that_name_an_action(self):
        for reply, want in (
                ("Done, sir. Shall I put it on the left monitor?",
                 "Shall I put it on the left monitor?"),
                ("Would you like me to read them to you?",
                 "Would you like me to read them to you?"),
                ("Do you want me to play something else?",
                 "Do you want me to play something else?"),
                ("I can turn the desk lamp off if you'd like.",
                 "I can turn the desk lamp off if you'd like."),
                ("If you’d like, I’ll pull up the forecast.",
                 "If you'd like, I'll pull up the forecast."),
                ("Opening it now. [ACTION: launch_app, notes] Shall I pin it?",
                 "Shall I pin it?")):
            with self.subTest(reply=reply):
                self.assertEqual(O.offer_text(reply), want)

    def test_a_do_that_takes_the_sentence_it_points_at(self):
        self.assertEqual(
            O.offer_text("I could restart the print job. Shall I do that?"),
            "I could restart the print job. Shall I do that?")

    def test_not_an_offer_a_yes_can_carry_out(self):
        for reply in (
                "Very good, sir.",
                "Would you like to hear more about it?",     # no action
                "Is there anything else, sir?",
                "Which monitor would you like it on?",       # wh-question
                "Shall I put it on the left or the right?",  # a choice
                "Shall I?",                                  # names nothing
                "Shall I move it? Actually, it is already there.",  # not last
                "", None):
            with self.subTest(reply=reply):
                self.assertEqual(O.offer_text(reply), "")

    def test_never_raises(self):
        self.assertEqual(O.offer_text(object()), "")


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class OpenOfferTests(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.slot = O.OpenOffer(120.0, clock=self.clock)
        self.offer = self.slot.note(_LIVE_OFFER)
        self.assertTrue(self.offer)

    def test_a_clear_yes_accepts_it(self):
        for answer in ("Jarvis, yes.", "Yes.", "Yes please.", "Go ahead.",
                       "Sure.", "Yeah, do it."):
            with self.subTest(answer=answer):
                self.slot.note(_LIVE_OFFER)
                self.clock.t += 10
                self.assertEqual(self.slot.take(answer, since=900.0),
                                 ("yes", self.offer))

    def test_a_no_and_a_hedged_yes_are_no(self):
        for answer in ("No.", "Jarvis, no thanks.", "Yes, but later.",
                       "Do it tomorrow.", "Yes, but wait."):
            with self.subTest(answer=answer):
                self.slot.note(_LIVE_OFFER)
                self.assertEqual(self.slot.take(answer)[0], "no")

    def test_an_unrelated_turn_is_other(self):
        self.assertEqual(self.slot.take("What's the weather tomorrow?")[0],
                         "other")

    def test_a_late_yes_does_not_answer_it(self):
        self.clock.t += 121.0
        self.assertEqual(self.slot.take("Yes."), ("expired", self.offer))

    def test_a_yes_after_another_owner_turn_does_not_answer_it(self):
        # The previous owner turn started AFTER the offer was made.
        self.assertEqual(self.slot.take("Yes.", since=self.clock.t + 1.0)[0],
                         "stale")

    def test_every_take_empties_the_slot(self):
        self.assertEqual(self.slot.take("What time is it?")[0], "other")
        self.assertEqual(self.slot.take("Yes."), ("none", ""))
        self.slot.note(_LIVE_OFFER)
        self.assertEqual(self.slot.take("Yes.")[0], "yes")
        self.assertEqual(self.slot.take("Yes."), ("none", ""))

    def test_a_reply_without_an_offer_closes_the_open_one(self):
        self.assertEqual(self.slot.note("Moved it, sir."), "")
        self.assertEqual(self.slot.peek(), "")
        self.assertEqual(self.slot.take("Yes."), ("none", ""))

    def test_a_malformed_ttl_accepts_nothing(self):
        slot = O.OpenOffer("soon", clock=self.clock)
        slot.note(_LIVE_OFFER)
        self.clock.t += 0.5
        self.assertEqual(slot.take("Yes.")[0], "expired")


class DirectiveTests(unittest.TestCase):
    def test_it_names_the_offer_and_asks_for_the_token(self):
        d = O.directive("Shall I put it on the top monitor?")
        self.assertIn('"Shall I put it on the top monitor?"', d)
        self.assertIn("[ACTION: ...]", d)
        self.assertIn("answered yes", d)
        self.assertTrue(d.startswith("\n\n"))

    def test_a_long_offer_is_clipped(self):
        self.assertLess(len(O.directive("x" * 5000)), 1000)


class AlsoRoutingTests(unittest.TestCase):
    """prompt_router's ``also``: the yes-turn routes on the offer too."""

    OFFER = "Shall I set a timer for ten minutes?"

    def test_a_bare_yes_routes_nothing_but_the_offer_routes_its_action(self):
        self.assertEqual(pr.turn_pc_block("Jarvis, yes.", PC_CONTROL_PROMPT),
                         "")
        block = pr.turn_pc_block("Jarvis, yes.", PC_CONTROL_PROMPT,
                                 also=[self.OFFER])
        self.assertIn("set_timer", block)
        self.assertNotIn("set_timer", pr.slim_pc_control(
            "Jarvis, yes.", PC_CONTROL_PROMPT).split("ADDITIONAL CAPABILITIES")[0])
        slim = pr.slim_pc_control("Jarvis, yes.", PC_CONTROL_PROMPT,
                                  also=self.OFFER)
        self.assertIn("set_timer", slim.split("ADDITIONAL CAPABILITIES")[0])

    def test_the_live_offers_action_is_always_in_front_of_the_model(self):
        # move_window_to_monitor lives in the always-on section, in the
        # cache-stable block; the offer adds WINDOW MANAGEMENT around it.
        self.assertIn("move_window_to_monitor",
                      pr.stable_pc_block(PC_CONTROL_PROMPT))
        block = pr.turn_pc_block(
            "Jarvis, yes.", PC_CONTROL_PROMPT,
            also=["Should I try moving the existing window to the top "
                  "monitor?"])
        self.assertIn("WINDOW MANAGEMENT", block)

    def test_also_never_routes_a_self_terminating_section(self):
        _core, sections = pr.split_pc_control(PC_CONTROL_PROMPT)
        offer = "Shall I shut down for the night and power off?"
        own, _ = pr.select_sections(offer, sections)
        selfterm = [n for n in own if n.upper() not in pr._ALWAYS
                    and pr.documents_self_terminating_action(
                        dict((h.strip(), b) for h, b in sections)[n])]
        self.assertTrue(selfterm, "precondition: the words route one")
        via_also, _ = pr.select_sections("Jarvis, yes.", sections,
                                         also=[offer])
        for name in selfterm:
            self.assertNotIn(name, via_also)

    def test_no_also_is_the_old_routing(self):
        _core, sections = pr.split_pc_control(PC_CONTROL_PROMPT)
        for text in ("Jarvis, yes.", "turn off the desk lamp",
                     "what's the weather"):
            with self.subTest(text=text):
                self.assertEqual(pr.select_sections(text, sections),
                                 pr.select_sections(text, sections, also=None))
                self.assertEqual(pr.select_sections(text, sections),
                                 pr.select_sections(text, sections, also=[]))


if __name__ == "__main__":
    unittest.main()
