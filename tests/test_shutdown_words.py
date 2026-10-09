"""The owner's own shutdown words (live 2026-10-06 22:28-22:30).

Four tries, zero shutdowns:

  1. "Jarvis shut down."            -> the overnight prompt was armed (fine).
  2. "Jarvis shut down no overnight protocol."
                                    -> "unrelated reply": the "no" sat
     mid-sentence. The prompt was cancelled and the model SAID "I've
     cancelled the overnight protocol and am powering down now." with
     nothing run - the claim check knew "shutting down", not "powering
     down".
  3. "Jarvis shut down with no overnight protocol."
                                    -> the model emitted shutdown_jarvis, but
     core.action_risk.asked_for_self_termination() read "with" as an object
     and held it for a yes.
  4. "Jarvis, yes, that is what I want."
                                    -> core.yes_no called it "other" (too many
     words after the yes): the held shutdown was cancelled.

Pinned here on the pure modules (core/action_risk.py, core/yes_no.py,
core/claim_validator.py) with the owner's exact words plus variants - and
the speech that must still NOT shut JARVIS down. The routers themselves are
pinned in tests/monolith/test_monolith_shutdown_words.py.

Review 2026-10-09 (the Review* classes): the first cut counted "updates" /
"updating" as the overnight protocol and dropped punctuation before matching,
so "Jarvis, power off, no updates." / "Shut down? No updates?" shut JARVIS
down at once and "Jarvis, laptop shut down without updating." armed the
prompt; any "no ..." the hedge list did not know ("No, stay on.") was a full
shutdown; a bare "That's what I said." confirmed a queued delete; a quoted
object was stripped into a bare "shut down". It also missed the owner's
"Yes, shut down with no overnight protocol." (a "no"), "Jarvis, Jarvis,
yes.", "Um, no.", "Just do it." and the overnight protocol said by name.

stdlib unittest only; CI-safe (light tier).
    python -m unittest tests.test_shutdown_words
"""
from __future__ import annotations

import unittest

from core import action_risk as ar
from core import claim_validator as cv
from core import yes_no

# The owner's exact four utterances, as transcribed live.
SAID_1 = "Jarvis shut down."
SAID_2 = "Jarvis shut down no overnight protocol."
SAID_3 = "Jarvis shut down with no overnight protocol."
SAID_4 = "Jarvis, yes, that is what I want."
# The reply spoken after (2) with no action run.
REPLY_2 = ("Very good, sir. I've cancelled the overnight protocol and am "
           "powering down now.")

# Speech that is NOT the owner asking JARVIS to go.
NOT_ASKING = (
    "Jarvis, don't shut down.",
    "Don't shut down",
    "Jarvis, do not shut down with no overnight protocol.",
    "No, don't shut down.",
    "Never go offline, Jarvis.",
    "Don't turn yourself off.",
    "should I shut down my PC?",
    "Should I shut down?",
    "Jarvis, shut down the printer.",
    "shut down the robot",
    "Jarvis, shut down the browser.",
    "Jarvis, shut down the printer with no overnight protocol.",
    'He said "shut down now."',
    'The sign said "shut down with no overnight protocol."',
    "The factory had to shut down with no warning.",
    "They shut down the highway without the permit.",
    "Shut down? No.",
    "It shut down last night.",
)


class AskedForSelfTerminationTests(unittest.TestCase):
    def test_the_owners_words_ask_for_a_shutdown(self):
        for said in (SAID_1, SAID_2, SAID_3,
                     "Jarvis, shut down without the protocol.",
                     "shut down without the overnight protocol",
                     "Jarvis, shut down, no overnight.",
                     "power down with no overnight protocol",
                     "Jarvis, shut down now.",
                     "shut down please",
                     "Jarvis, shut down for the night.",
                     "shut down right now",
                     "Jarvis, go offline with no upgrade."):
            for name in ("shutdown_jarvis", "exit_jarvis", "shut_down"):
                with self.subTest(said=said, name=name):
                    self.assertTrue(ar.asked_for_self_termination(name, said))

    def test_speech_that_is_not_the_owner_asking_does_not(self):
        for said in NOT_ASKING:
            for name in ("shutdown_jarvis", "exit_jarvis", "shut_down"):
                with self.subTest(said=said, name=name):
                    self.assertFalse(
                        ar.asked_for_self_termination(name, said))

    def test_a_decline_of_something_else_is_not_a_qualifier(self):
        for said in ("shut down with no warning", "shut down no",
                     "shut down not now", "shut down with chrome",
                     "shut down without saving"):
            with self.subTest(said=said):
                self.assertFalse(
                    ar.asked_for_self_termination("shutdown_jarvis", said))

    def test_negation_and_other_subjects_reach_restart_and_overnight_too(self):
        for name, said in (("restart", "don't restart"),
                           ("restart", "Should I restart?"),
                           ("restart", "it restarted? no, did it restart"),
                           ("start_overnight_upgrade", "I can't sleep"),
                           ("start_overnight_upgrade", "don't go to bed yet")):
            with self.subTest(name=name, said=said):
                self.assertFalse(ar.asked_for_self_termination(name, said))
        for name, said in (("restart", "Jarvis, restart."),
                           ("start_overnight_upgrade", "Goodnight, Jarvis."),
                           ("shutdown_jarvis", "Jarvis, I said shut down."),
                           ("shutdown_jarvis", "can you shut down")):
            with self.subTest(name=name, said=said):
                self.assertTrue(ar.asked_for_self_termination(name, said))


class ShutdownDecliningOvernightTests(unittest.TestCase):
    def test_a_shutdown_that_says_no_to_the_overnight_protocol(self):
        for said in (SAID_2, SAID_3,
                     "Jarvis, shut down, no overnight protocol.",
                     "Hey Jarvis, shut down now with no overnight protocol, "
                     "please.",
                     "shut down without the protocol",
                     "Power down without the overnight protocol.",
                     "Jarvis, just shut down, no overnight.",
                     "No overnight protocol, just shut down.",
                     "Jarvis, without the overnight protocol, shut down."):
            with self.subTest(said=said):
                self.assertTrue(ar.shutdown_declining_overnight(said))
                self.assertTrue(
                    ar.shutdown_declining_overnight(said, alone_ok=True))

    def test_the_decline_alone_counts_only_while_the_question_is_open(self):
        for said in ("Without the overnight protocol.",
                     "Jarvis, with no overnight protocol.",
                     "without the protocol, please", "No overnight protocol."):
            with self.subTest(said=said):
                self.assertFalse(ar.shutdown_declining_overnight(said))
                self.assertTrue(
                    ar.shutdown_declining_overnight(said, alone_ok=True))

    def test_anything_else_is_not_that_shape(self):
        for said in NOT_ASKING + (
                SAID_1, SAID_4, "", None, "No.",
                "shut down with the overnight protocol",
                "Jarvis, shut down, yes, overnight protocol.",
                "no overnight protocol tomorrow, remind me",
                "without the overnight protocol I would be lost"):
            with self.subTest(said=said):
                self.assertFalse(
                    ar.shutdown_declining_overnight(said, alone_ok=True))


class OwnerYesTests(unittest.TestCase):
    def test_the_owners_yes_is_a_yes(self):
        for said in (SAID_4, "Jarvis, yes please do.", "Jarvis, yes, do it.",
                     "Yes, that's what I want.", "Yes, that is what I said.",
                     "Yeah, that's what I meant.",
                     "Jarvis, yes, that's what I asked for."):
            with self.subTest(said=said):
                self.assertEqual(yes_no.classify_reply(said), "yes")

    def test_lookalikes_stay_other_or_no(self):
        # A bare restatement is not a yes anywhere (review 2026-10-09): "That's
        # what I said." is a correction as often as an answer - it confirmed
        # a queued delete. The held-shutdown question alone takes it
        # (action_risk.confirms_held_shutdown).
        for said in ("Yep, that is what she said.",
                     "Yes, that is what I want to know about the weather",
                     "That's what I want to know.", "That's what I want.",
                     "That's what I said.", "Jarvis, that's what I meant."):
            with self.subTest(said=said):
                self.assertEqual(yes_no.classify_reply(said), "other")
        for said in ("No, that is what I want to avoid.",
                     "Yes, that is what I want, but later.",
                     "No, don't."):
            with self.subTest(said=said):
                self.assertEqual(yes_no.classify_reply(said), "no")


class PoweringDownClaimTests(unittest.TestCase):
    def test_the_live_reply_is_an_unverified_claim(self):
        self.assertTrue(cv.find_unverified_claim(REPLY_2, user_text=SAID_2))

    def test_other_ways_of_saying_it(self):
        for reply in ("Powering down now, sir.",
                      "Very good, sir. Powering down.",
                      "I'm powering down now, sir.",
                      "I'll power down now, sir.",
                      "Understood, sir. I'm shutting myself down.",
                      "Shutting myself down, sir.",
                      "Going offline now, sir.",
                      "I'm going offline, sir.",
                      "I'm turning myself off now.",
                      "Understood, sir. I'm shutting down now."):
            with self.subTest(reply=reply):
                self.assertTrue(cv.find_unverified_claim(reply,
                                                         user_text=SAID_3))

    def test_grounded_offers_and_facts_are_not_claims(self):
        self.assertIsNone(cv.find_unverified_claim(
            "Powering down now, sir.", ran_actions=["shutdown_jarvis"],
            user_text=SAID_3))
        self.assertIsNone(cv.find_unverified_claim(
            "I've powered off the lamp, sir.",
            ran_actions=["smart_home_control"],
            user_text="turn off the lamp"))
        for reply in ("Shall I power down, sir?",
                      "Would you like me to go offline for the night?",
                      "Powering down a PC properly takes a few seconds.",
                      "The power went down for an hour last night, sir.",
                      "That would shut me down completely, sir. Say yes if "
                      "that is what you want."):
            with self.subTest(reply=reply):
                self.assertIsNone(cv.find_unverified_claim(reply,
                                                           user_text=SAID_1))


# ── Review 2026-10-09: what the first cut let through, and what it missed ──
SHUTDOWN_NAMES = ("shutdown_jarvis", "exit_jarvis", "quit_jarvis")


class ReviewGateTests(unittest.TestCase):
    """asked_for_self_termination: the decline credit is whole-utterance
    only, and the guards take questions, other subjects, quoted objects,
    the noun "shutdown" and a declined overnight protocol out."""

    def test_questions_and_hypotheticals_do_not_ask(self):
        for said in (
                "Jarvis, what happens if you shut down with no overnight "
                "protocol?",
                "Jarvis, if I say shut down with no overnight protocol, what "
                "happens?",
                "Jarvis, is it safe to shut down without the overnight "
                "protocol?",
                "Jarvis, do you shut down with no overnight protocol by "
                "default?",
                "Jarvis, how do I make you shut down without the overnight "
                "protocol?",
                "Jarvis, did you shut down without the update last night?",
                "Jarvis, would you power off without the update if I asked?",
                "Jarvis, why did you go offline last night?",
                "Jarvis, can you shut down without updating?"):
            for name in SHUTDOWN_NAMES:
                with self.subTest(said=said, name=name):
                    self.assertFalse(
                        ar.asked_for_self_termination(name, said))

    def test_someone_else_shutting_down_is_not_asking(self):
        for said in ("Jarvis, laptop shut down without updating.",
                     "Jarvis, my laptop decided to shut down without "
                     "updates.",
                     "Jarvis, Windows will shut down without the update "
                     "tonight.",
                     "Jarvis, the TV will power off with no updates",
                     "My laptop shut down.", "The factory had to shut down.",
                     "Windows will shut down.", "I'll shut down.",
                     "It'll shut down.", "The TV will power off.",
                     "Jarvis exit without updates",
                     "Jarvis, quit without updating, Steam"):
            for name in SHUTDOWN_NAMES:
                with self.subTest(said=said, name=name):
                    self.assertFalse(
                        ar.asked_for_self_termination(name, said))

    def test_a_quoted_object_still_fills_the_slot(self):
        for said in ('Jarvis, shut down "Plex".', 'quit "Fortnite"',
                     'shut down "Steam"', '"Shut down."'):
            for name in SHUTDOWN_NAMES:
                with self.subTest(said=said, name=name):
                    self.assertFalse(
                        ar.asked_for_self_termination(name, said))

    def test_the_noun_and_an_undo_verb_are_not_asking(self):
        for said in ("Cancel the shutdown.", "Jarvis, no shutdown.",
                     "No, no shutdown.", "Abort shut down.",
                     "No, I don't want you to shut down."):
            with self.subTest(said=said):
                self.assertFalse(
                    ar.asked_for_self_termination("shutdown_jarvis", said))

    def test_declining_the_overnight_protocol_is_not_asking_for_it(self):
        for said in (SAID_3, "Jarvis, no overnight protocol.",
                     "Jarvis skip the overnight protocol",
                     "Jarvis, do not shut down with no overnight protocol.",
                     "Jarvis what is the overnight protocol"):
            with self.subTest(said=said):
                self.assertFalse(ar.asked_for_self_termination(
                    "start_overnight_upgrade", said))

    def test_owner_orders_still_ask(self):
        for name, said in (
                ("shutdown_jarvis", SAID_1), ("shutdown_jarvis", SAID_2),
                ("shutdown_jarvis", SAID_3),
                ("shutdown_jarvis", "Jarvis, we're done, shut down."),
                ("shutdown_jarvis", "That's it, shut down."),
                ("shutdown_jarvis", "Jarvis, stop everything and shut down."),
                ("shutdown_jarvis", "Jarvis, it's time to shut down."),
                ("shutdown_jarvis", "Jarvis, you can shut down now."),
                ("shutdown_jarvis", "Jarvis, I want you to shut down."),
                ("shutdown_jarvis", "Jarvis I'm done for today shut down"),
                ("shutdown_jarvis", "Can you shut down?"),
                ("start_overnight_upgrade", "Goodnight, Jarvis."),
                ("start_overnight_upgrade", "Yes, run the overnight "
                                            "protocol.")):
            with self.subTest(name=name, said=said):
                self.assertTrue(ar.asked_for_self_termination(name, said))


class ReviewDeclineShapeTests(unittest.TestCase):
    def test_more_ways_to_decline_the_overnight_protocol(self):
        for said in (
                "Jarvis, Jarvis, shut down with no overnight protocol.",
                "Jarvis shut down, no over night protocol.",
                "Jarvis shut down but no overnight protocol.",
                "Jarvis shut down, skip the overnight protocol.",
                "Jarvis shut down, don't do the overnight protocol.",
                "Jarvis shut down, I don't need the overnight protocol.",
                "Jarvis, shut down without running the overnight protocol.",
                "Jarvis, power off, don't do the overnight protocol.",
                "Jarvis, shut down, we don't need the overnight protocol.",
                "Jarvis, shut it down, no overnight protocol.",
                "Jarvis turn off, no overnight protocol.",
                "Jarvis, don't do overnight, shut down.",
                "No. Shut down without the overnight protocol."):
            with self.subTest(said=said):
                self.assertTrue(ar.shutdown_declining_overnight(said))

    def test_the_decline_alone_while_the_question_is_open(self):
        for said in ("Skip the overnight protocol.", "Skip overnight.",
                     "I don't want the overnight protocol.",
                     "No, don't bother with the overnight protocol.",
                     "No overnight protocol, don't need it."):
            with self.subTest(said=said):
                self.assertFalse(ar.shutdown_declining_overnight(said))
                self.assertTrue(
                    ar.shutdown_declining_overnight(said, alone_ok=True))

    def test_one_request_only_and_only_the_overnight_protocol(self):
        for said in ("Shut down? No updates?", "Shut down. Not updating.",
                     "Jarvis, power off, no updates.",
                     "Jarvis, shut down without updating?",
                     "Jarvis, shut down without updating.",
                     "Jarvis, laptop shut down without updating.",
                     "Shut down with no overnight protocol? No, wait.",
                     "Jarvis, shut down. No, overnight protocol.",
                     '"Shut down without the protocol."',
                     "What happens if you shut down without the overnight "
                     "protocol?",
                     "No, the overnight protocol.",
                     "Shut it down, skip the protocol!",
                     "Turn it off, no protocol.",
                     "No shutdown, no overnight protocol.",
                     "Skip it."):
            with self.subTest(said=said):
                self.assertFalse(
                    ar.shutdown_declining_overnight(said, alone_ok=True))


class ReviewPromptReplyTests(unittest.TestCase):
    """The overnight question's own answers (core.action_risk)."""

    def test_a_plain_no_is_a_no(self):
        for said in ("no", "nope", "no no", "no thanks", "no thank you",
                     "no overnight", "no overnight protocol", "negative",
                     "no just shut down", "just shut down", "full shutdown",
                     "shut down completely", "no im good", "no go ahead",
                     "no shut down", "no overnight protocol just shut down"):
            with self.subTest(said=said):
                self.assertTrue(ar.plain_no_to_overnight(said))

    def test_a_no_that_says_anything_else_is_not(self):
        for said in ("no stay on", "no keep running", "no i changed my mind",
                     "nope stay awake", "no no keep going", "no i need you",
                     "no keep listening", "nope scratch that", "no forget it",
                     "no my laptop", "no the printer did", "no plex",
                     "no i meant my pc", "no no shutdown", "no shutdown",
                     "no the overnight protocol", "just shut down the printer",
                     "no updates on the printer", "no wait", "no dont",
                     "no not tonight just shut down", ""):
            with self.subTest(said=said):
                self.assertFalse(ar.plain_no_to_overnight(said))

    def test_the_overnight_protocol_by_name_is_a_yes_to_it(self):
        for said in ("Yes, run the overnight protocol, then shut down.",
                     "Overnight protocol, then shut down.",
                     "Do the overnight protocol and then shut down.",
                     "Overnight first, then shut down.",
                     "Start the overnight protocol and shut down.",
                     "Yes, overnight, then power off.",
                     "Yes, overnight protocol.",
                     "Jarvis, yes, overnight protocol.",
                     "Yeah, do the overnight protocol.",
                     "Run the overnight protocol.", "Yes, overnight first.",
                     "Jarvis yes do the overnight",
                     "Yes, with the overnight protocol.",
                     "Jarvis, shut down with the overnight protocol."):
            with self.subTest(said=said):
                self.assertTrue(ar.accepts_overnight(said))

    def test_declining_or_asking_about_it_is_not(self):
        for said in ("No overnight protocol.", "Without the overnight "
                     "protocol.", "Skip overnight.", "Overnight? No.",
                     "Do the overnight protocol later.",
                     "Don't do the overnight protocol.", "Yes, shut down.",
                     '"Overnight protocol."', "what is the overnight protocol",
                     "Sure, run it."):
            with self.subTest(said=said):
                self.assertFalse(ar.accepts_overnight(said))


class ReviewHeldShutdownYesTests(unittest.TestCase):
    """confirms_held_shutdown: a yes to "That would shut me down completely,
    sir. Say yes if that is what you want." that restates it."""

    def test_a_restated_yes(self):
        for said in (SAID_4, "Yes, shut down with no overnight protocol.",
                     "Jarvis, yes, shut down with no overnight protocol.",
                     "Yes, shut it down.", "Yes, turn off.",
                     "Yes, I want you to shut down.",
                     "Yes, I really want you to shut down.",
                     "Yes, I asked you to shut down.",
                     "Yes, that's what I want, shut down.",
                     "Yes, shut yourself down.", "Yes, power down.",
                     "Yes, shut down.", "Yes, please shut down.",
                     "Yes, go ahead and shut down.", "Yes, I want to shut down.",
                     "Yeah, I'm sure, shut down.", "Yes, go offline.",
                     "Yes, shut down already.",
                     "Yes, shut down, no overnight protocol.",
                     "That is what I want.", "That's what I said."):
            with self.subTest(said=said):
                self.assertTrue(ar.confirms_held_shutdown(said))

    def test_anything_else_is_not(self):
        for said in ("Yes, shut down the printer.", "Yes, shut down later.",
                     "Shut down?", "Yes, shut down?", 'Yes, "shut down".',
                     "Sure, shut down the TV.", "Yes, turn off the lights.",
                     "No, shut down.", "That's what she said.", "Please.",
                     "Shut down.", "Yes, don't shut down.", "Yes. No, wait.",
                     "that is what I want to know", ""):
            with self.subTest(said=said):
                self.assertFalse(ar.confirms_held_shutdown(said))


class ReviewYesNoTests(unittest.TestCase):
    def test_a_restatement_after_a_yes(self):
        for said in ("Yes, that's exactly what I want.",
                     "Yes, that's what I'm asking for.",
                     "Yes, that's what I'm saying.",
                     "Yes, that's what I would like.",
                     "Yes, that is what I want to happen.",
                     "Yes, I do want that.", "Yes I want that",
                     "Yes, I want it.", "Yes, I meant that.", "I said yes.",
                     "Yes. I said yes.", "Yes, you heard me.",
                     "Yes, that's the plan."):
            with self.subTest(said=said):
                self.assertEqual(yes_no.classify_reply(said), "yes")

    def test_a_restatement_only_ends_a_yes(self):
        for said in ("Yeah, I want that movie.", "I said yes to her.",
                     "Yes, you heard me wrong.", "That is right.",
                     "Just a second.", "Just kidding."):
            with self.subTest(said=said):
                self.assertEqual(yes_no.classify_reply(said), "other")

    def test_just_a_hesitation_and_a_repeated_wake_word(self):
        for said, verdict in (("Just do it.", "yes"),
                              ("Please, just do it.", "yes"),
                              ("Um, no.", "no"), ("Uh, no.", "no"),
                              ("Um, yes.", "yes"), ("Uh yeah.", "yes"),
                              ("Jarvis, Jarvis, yes.", "yes"),
                              ("Jarvis, Jarvis, no.", "no"),
                              ("Jarvis.", "other"), ("Um.", "other")):
            with self.subTest(said=said):
                self.assertEqual(yes_no.classify_reply(said), verdict)
        self.assertEqual(yes_no.normalize("Jarvis, Jarvis, no."), "no")
        self.assertEqual(yes_no.normalize("Um, Jarvis, no."), "no")
        self.assertEqual(yes_no.normalize("Jarvis."), "jarvis")


class ReviewPowerClaimTests(unittest.TestCase):
    def test_a_stop_that_ran_grounds_powering_down(self):
        for reply, ran in (
                ("I've powered down the air mouse, sir.", "air_mouse_disarm"),
                ("Powering down ambient listening, sir.",
                 "ambient_listen_stop"),
                ("Going offline now, sir. Phone bridge paused.",
                 "pause_phone_bridge"),
                ("I've powered down the stream, sir.", "obs_stop_recording"),
                ("Powering down the workshop HUD, sir.", "hide_workshop_hud"),
                ("Powering down the pipeline, sir.", "stop_pipeline")):
            with self.subTest(reply=reply):
                self.assertIsNone(cv.find_unverified_claim(
                    reply, ran_actions=[ran], user_text="stop it"))

    def test_a_past_event_is_an_answer(self):
        for reply in ("I went offline at 2 a.m. for the overnight protocol, "
                      "sir.",
                      "I powered down at two last night for the overnight "
                      "protocol, sir."):
            with self.subTest(reply=reply):
                self.assertIsNone(cv.find_unverified_claim(
                    reply, user_text="why were you offline?"))

    def test_the_live_claim_is_still_caught(self):
        self.assertTrue(cv.find_unverified_claim(REPLY_2, user_text=SAID_2))
        self.assertTrue(cv.find_unverified_claim(
            "I've powered down, sir.", user_text=SAID_3))


if __name__ == "__main__":
    unittest.main()
