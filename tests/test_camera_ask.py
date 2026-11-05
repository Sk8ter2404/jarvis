"""v3.0.11 lane V: what is in front of the camera (core.camera_ask), the claims
and promises about it (core.claim_validator), the one line said once per turn
(core.chain_lines), the last line said back (core.repeat_line), plain retries
carrying no mood (core.tone_detector, core.emotion_tracker) and screen looks for
a physical ask (core.action_fit rule 8).

    python tools/run_tests.py test_camera_ask
"""
from __future__ import annotations

import unittest
from unittest import mock

import core.config as cfg
from core import action_fit, camera_ask as ca, chain_lines, claim_validator as cv
from core import emotion_tracker, recall_actions, repeat_line, tone_detector
from core.tone_detector import detect_tone, is_plain_retry


class AskKindTests(unittest.TestCase):
    def test_strict_forms(self):
        for t in ("Jarvis, what am I holding?", "what am I holding right now, sir",
                  "what am I wearing", "what am I wearing today?", "what am I doing",
                  "what am I doing right now?", "what's in my hand?",
                  "what is in my hands", "what do I have in my hand",
                  "what's this I've got", "what is this I'm holding",
                  "what's this in my hand", "how do I look", "how do I look today?",
                  "do I look okay", "can you see what I'm holding?",
                  "tell me what I'm holding", "what the hell am I holding?",
                  "what color shirt am I wearing", "what kind of drink am I holding",
                  "look at my shirt", "Hey Jarvis, what am I holding?",
                  "what do you think I'm holding", "what's in my other hand"):
            self.assertEqual(ca.ask_kind(t), ca.STRICT, t)

    def test_ambiguous_forms(self):
        for t in ("what is this", "Jarvis, what's this?", "look at this",
                  "take a look at this", "can you see this", "do you see this",
                  "what colour is this", "what color is this thing",
                  "can you tell me what this is", "do you know what this is"):
            self.assertEqual(ca.ask_kind(t), ca.AMBIGUOUS, t)

    def test_screen_music_and_ordinary_speech_never_match(self):
        for t in ("what's on my screen", "what is this page", "what is this error",
                  "what's this song", "what is this about", "what is that",
                  "what am I doing tomorrow", "what am I doing today",
                  "what am I doing wrong", "what am I doing here",
                  "what am I doing with my life", "how do I look up a number",
                  "what am I watching", "what am I listening to",
                  "what stock am I holding", "what position am I holding",
                  "what should I wear", "what am I wearing to the wedding",
                  "look at this page", "look at this website", "can you see this window",
                  "what am I eating for dinner", "what time is it", "", None):
            self.assertIsNone(ca.ask_kind(t), t)


class RouteTests(unittest.TestCase):
    NOW = 1000.0

    def _row(self, name, arg="", ago=5.0):
        return {"action": name, "arg": arg, "result": "x", "at": self.NOW - ago}

    def test_strict_routes_with_his_words(self):
        self.assertEqual(ca.route_token("Jarvis, what am I holding?", now=self.NOW),
                         "[ACTION: see_user, what am I holding?]")

    def test_strict_routes_whatever_just_ran(self):
        tok = ca.route_token("what am I holding?", [self._row("see_screen")],
                             now=self.NOW)
        self.assertEqual(tok, "[ACTION: see_user, what am I holding?]")

    def test_ambiguous_routes_when_nothing_else_ran(self):
        self.assertEqual(ca.route_token("what is this", now=self.NOW),
                         "[ACTION: see_user, what is this]")

    def test_ambiguous_yields_to_a_recent_action(self):
        for name in ("see_screen", "play_music", "open_url"):
            self.assertIsNone(ca.route_token("what's this?", [self._row(name)],
                                             now=self.NOW), name)
        # ... but not one long gone, nor a camera look.
        self.assertIsNotNone(ca.route_token(
            "what's this?", [self._row("play_music", ago=300)], now=self.NOW))
        self.assertIsNotNone(ca.route_token(
            "what's this?", [self._row("see_user")], now=self.NOW))

    def test_ambiguous_yields_when_the_front_window_changed(self):
        self.assertIsNone(ca.route_token("what is this", now=self.NOW,
                                         focus_changed=lambda: True))
        # A strict ask never asks.
        self.assertIsNotNone(ca.route_token(
            "what am I holding", now=self.NOW,
            focus_changed=mock.Mock(side_effect=AssertionError)))

    def test_flag_off_no_ask(self):
        self.assertIsNone(ca.route_token("what am I holding", now=self.NOW,
                                         ask_on=False))

    def test_is_camera_route_token(self):
        self.assertTrue(ca.is_camera_route_token("[ACTION: see_user, x]"))
        self.assertFalse(ca.is_camera_route_token("[ACTION: see_screen, x]"))
        self.assertFalse(ca.is_camera_route_token(None))


class RetryTests(unittest.TestCase):
    NOW = 1000.0

    def _row(self, name, arg="", ago=5.0):
        return {"action": name, "arg": arg, "result": "x", "at": self.NOW - ago}

    def test_retry_forms(self):
        for t in ("Jarvis, look again.", "look harder", "look closer",
                  "no, look again", "take another look", "have another look at it",
                  "one more look"):
            self.assertEqual(ca.retry_kind(t), "look", t)
        for t in ("try again", "try that again", "check again",
                  "please check it again"):
            self.assertEqual(ca.retry_kind(t), "generic", t)

    def test_not_retries(self):
        for t in ("look", "look again at the page", "check the weather again",
                  "try harder", "do it again", "look at me", "", None):
            self.assertIsNone(ca.retry_kind(t), t)

    def test_after_a_camera_look_reruns_the_earlier_question(self):
        hist = [self._row("see_user", "what am I holding?")]
        for t in ("Jarvis, look again.", "look harder", "try again", "check again"):
            self.assertEqual(ca.route_token(t, hist, now=self.NOW),
                             "[ACTION: see_user, what am I holding?]", t)

    def test_after_a_screen_look_stays_a_screen_look(self):
        hist = [self._row("see_screen", "read this page")]
        self.assertEqual(ca.route_token("look again", hist, now=self.NOW),
                         "[ACTION: see_screen, read this page]")
        self.assertEqual(ca.route_token("try again", [self._row("recall_screen")],
                                        now=self.NOW), "[ACTION: see_screen]")

    def test_a_task_with_a_screen_look_in_it_stays_with_the_brain(self):
        hist = [self._row("click", "login", ago=10), self._row("see_screen", "", ago=5)]
        self.assertIsNone(ca.route_token("try again", hist, now=self.NOW))
        self.assertEqual(ca.route_token("look again", hist, now=self.NOW),
                         "[ACTION: see_screen]")

    def test_after_a_face_look_describes_him(self):
        self.assertEqual(ca.route_token("look again", [self._row("recognize_face")],
                                        now=self.NOW), "[ACTION: see_user]")

    def test_the_newest_action_decides(self):
        hist = [self._row("see_user", "what am I holding?", ago=20),
                self._row("play_music", "x", ago=5)]
        self.assertIsNone(ca.route_token("try again", hist, now=self.NOW))
        self.assertIsNone(ca.route_token("look again", hist, now=self.NOW))

    def test_window(self):
        hist = [self._row("see_user", "q", ago=90)]
        self.assertIsNotNone(ca.look_retry_token("look again", hist, self.NOW))
        self.assertIsNone(ca.look_retry_token("try again", hist, self.NOW))
        hist = [self._row("see_user", "q", ago=500)]
        self.assertIsNone(ca.look_retry_token("look again", hist, self.NOW))

    def test_nothing_ran_leaves_it_alone(self):
        self.assertIsNone(ca.route_token("look again", [], now=self.NOW))

    def test_flag_off(self):
        hist = [self._row("see_user", "q")]
        self.assertIsNone(ca.route_token("look again", hist, now=self.NOW,
                                         retry_on=False))


class SpokenAnswerTests(unittest.TestCase):
    def test_tag_and_note_come_off(self):
        self.assertEqual(
            ca.spoken_answer("[local-vision] You are holding a can.\n\n"
                             "(note: camera 0 frame is 6.2s old)"),
            "You are holding a can.")

    def test_no_answer(self):
        for r in ("no cameras configured", "no webcam frames available yet",
                  "no frame cached for that camera", "failed to encode webcam frame",
                  "(vision failed: HTTP 500)", "(local vision unavailable)", "",
                  None, "⚠ nope"):
            self.assertIsNone(ca.spoken_answer(r), r)

    def test_long_answer_is_cut_to_sentences(self):
        out = ca.spoken_answer("One. Two. Three. Four. Five.")
        self.assertEqual(out, "One. Two. Three.")
        self.assertLessEqual(len(ca.spoken_answer("word " * 300)), 421)


class CameraClaimTests(unittest.TestCase):
    ASK = "Jarvis, what am I holding?"

    def _find(self, text, user="how is it going", ran=(), grounding=()):
        return cv.find_camera_claim(text, ran_actions=ran, user_text=user,
                                    grounding=grounding)

    def test_live_lines(self):
        self.assertTrue(self._find("You're holding a box of Tic Tacs, sir.", self.ASK))
        self.assertTrue(self._find("You are holding a smartphone, sir.", self.ASK))

    def test_variants(self):
        for t in ("You appear to be holding a red mug.", "You're wearing a blue shirt.",
                  "It looks like you're holding up a phone.",
                  "You're clearly holding a can, sir.", "You have a pen in your hand.",
                  "That's a stapler in your right hand.", "You're drinking a soda."):
            self.assertTrue(self._find(t), t)

    def test_narrow(self):
        for t in ("You're holding up well.", "You are holding a meeting at 3.",
                  "You're holding up remarkably well, sir.",
                  "You're holding a team meeting tomorrow.", "You are holding the lead.",
                  "If you're holding a can, shake it.",
                  "While you're holding the door, I'll wait.",
                  "I can't see what you're holding, sir.",
                  "Are you holding a can?", "You're wearing me out, sir.",
                  "You're wearing thin on patience.", "You hold a fine opinion, sir.",
                  "Your meeting is at 3."):
            self.assertIsNone(self._find(t), t)

    def test_only_an_object_counts_when_he_did_not_ask(self):
        for t in ("You're holding a conversation with a machine, sir.",
                  "You're holding a steady pace on the savings floor, sir.",
                  "You're holding an open ticket with the carrier, sir.",
                  "You're holding a total of 14 open tabs, sir.",
                  "You're holding a 3D print in the queue, sir.",
                  "You're holding a CCNA quiz due at midnight, sir.",
                  "You are holding a reservation at 7, sir.",
                  "You're holding a phone number for the office, sir.",
                  "You're holding firm control of the budget, sir.",
                  "You're holding something back, sir.",
                  "You're holding the bag on this one, sir.",
                  "You're eating the cost, sir.",
                  "You're wearing two hats today, sir.",
                  "It's all in your hands, sir.",
                  "That's a decision in your hands, sir."):
            self.assertIsNone(self._find(t), t)
        for t in ("You're holding a remote control, sir.",
                  "You're holding a pair of scissors.",
                  "You appear to be holding what appears to be a coffee mug.",
                  "You're wearing a pair of glasses."):
            self.assertTrue(self._find(t), t)

    def test_any_thing_counts_when_his_line_was_about_it(self):
        about = "what do you think of what I'm holding?"
        self.assertTrue(self._find("You're holding a lemon, sir.", about))
        self.assertTrue(self._find("You're holding a lemon, sir.", self.ASK))
        self.assertIsNone(self._find("You're holding a total of 14 tabs, sir.", about))
        self.assertIsNone(self._find("You're holding up well, sir.", about))

    def test_function_words_back_nothing(self):
        for line, claim in (
                ("Jarvis, can you see me?", "You're holding a can of Coke, sir."),
                ("I can see it from here", "You're holding a can, sir."),
                ("we can do that later", "You're holding a can, sir."),
                ("give me a hand", "You have a phone in your hand."),
                ("is that one right", "You're holding a phone, sir."),
                ("I feel blue today", "You're wearing a blue shirt, sir."),
                ("phone him for me", "You're holding a phone, sir."),
                ("I need to phone the office", "You're holding a phone, sir.")):
            self.assertTrue(self._find(claim, "are you sure", grounding=[line]),
                            line)

    def test_a_noun_in_his_words_still_backs_it(self):
        for line, claim in (
                ("this is a coke can, dude", "You're holding a can, sir."),
                ("this can is empty", "You're holding a can, sir."),
                ("that can of coke", "You're holding a can, sir."),
                ("I love coke, you know", "You're drinking some coke."),
                ("Cans are great", "You're holding a can, sir."),
                ("I'm wearing my new hat", "You're wearing a hat, sir.")):
            self.assertIsNone(self._find(claim, "are you sure", grounding=[line]),
                              line)

    def test_flat_description_is_not_backed_by_its_own_small_words(self):
        for t in ("It's a can, sir.", "You're a can, sir."):
            self.assertTrue(self._find(t, self.ASK, grounding=[
                "Jarvis, can you see me?", "it's fine, you're right"]), t)
        self.assertIsNone(self._find("It's a can, sir.", self.ASK, grounding=[
            "You are holding a can of Coke."]))

    def test_a_look_backs_it(self):
        for name in ("see_user", "look_around", "where_am_i"):
            self.assertIsNone(self._find("You're holding a can.", self.ASK, ran=[name]))
        # A face match says nothing about a can.
        self.assertTrue(self._find("You're holding a can.", self.ASK,
                                   ran=["recognize_face"]))

    def test_his_own_words_back_it(self):
        self.assertIsNone(self._find("You're holding a coke can, sir.",
                                     "this is a coke can, dude"))
        self.assertIsNone(self._find("You're wearing a hat.", "I got a new hat"))

    def test_a_recent_look_backs_it(self):
        self.assertIsNone(self._find("You're holding a coke can.", "are you sure",
                                     grounding=["You are holding a can of Coke."]))
        self.assertTrue(self._find("You're holding a phone.", "are you sure",
                                   grounding=["You are holding a can of Coke."]))

    def test_any_flat_description_after_a_strict_ask(self):
        for t in ("It's a can, sir.", "That's a phone.", "You look sharp, sir.",
                  "You're typing."):
            self.assertTrue(self._find(t, "what am I holding"), t)
        for t in ("I can't see that from here, sir.", "Let me take a look, sir.",
                  "It's hard to say, sir."):
            self.assertIsNone(self._find(t, "what am I holding"), t)
        # Not a camera ask: those are nothing.
        self.assertIsNone(self._find("It's a can, sir.", "what is a tin"))

    def test_drop_keeps_the_rest(self):
        kept, flagged, sub = cv.drop_camera_claims(
            "[intent:observation] You are holding a phone, sir. The kettle is done.",
            user_text="how is it going")
        self.assertEqual(flagged, ["you are holding a phone"])
        self.assertEqual(kept, "[intent:observation] The kettle is done.")
        self.assertTrue(sub)

    def test_drop_no_claim_is_unchanged(self):
        t = "You're holding up well, sir."
        self.assertEqual(cv.drop_camera_claims(t, user_text="hi"), (t, [], True))

    def test_mentions_physical(self):
        self.assertTrue(ca.mentions_physical("what do you think of what I'm holding?"))
        self.assertTrue(ca.mentions_physical("what am I holding"))
        self.assertFalse(ca.mentions_physical("I'm holding a meeting at 3"))
        self.assertFalse(ca.mentions_physical("what's the weather"))


class RecordsPromiseTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(cfg, "SELF_CHANGE_RECORDS_CHECK", True, create=True)
        p.start()
        self.addCleanup(p.stop)

    def _find(self, text, ran=()):
        return cv.find_self_change_promise(text, ran_actions=ran)

    def test_live_line(self):
        live = ("My apologies, sir; I'll update my visual database to include 'can' "
                "instead of 'smartphone'.")
        self.assertTrue(self._find(live))
        self.assertTrue(cv.is_records_promise(live))

    def test_variants(self):
        for t in ("I'll update my visual memory.", "I'll update my records, sir.",
                  "I'll retrain my recognition model.", "I will add that to my database.",
                  "I'll add 'can' to my visual lexicon.", "Let me update my database.",
                  "I've updated my records.", "I'll correct my object recognition.",
                  "I'll make sure to update my model.", "My records have been updated.",
                  "I'm updating my visual database now."):
            self.assertTrue(self._find(t), t)

    def test_not_a_promise(self):
        for t in ("My apologies, sir.", "I'll check my records.",
                  "I'll update your records.", "I'll update the calendar.",
                  "I can't update my database, sir.", "Shall I update my records?",
                  "I have no database to update, sir.",
                  "I'll remember that.", "I'll update you when it finishes.",
                  "I'll update you once my records sync."):
            self.assertIsNone(self._find(t), t)

    def test_an_action_keeps_it(self):
        self.assertIsNone(self._find("I'll update my records.", ran=["remember"]))

    def test_my_apologies_stays(self):
        kept, flagged, _s = cv.drop_self_change_promises(
            "[intent:amused] [wry] My apologies, sir; I'll update my visual database "
            "to include 'can' instead of 'smartphone'. One does admire the optimism.")
        self.assertTrue(flagged)
        self.assertEqual(kept, "[intent:amused] [wry] My apologies, sir. "
                               "One does admire the optimism.")

    def test_flag_off_memory_word_lets_it_pass(self):
        with mock.patch.object(cfg, "SELF_CHANGE_RECORDS_CHECK", False, create=True):
            self.assertIsNone(self._find("I'll update my visual database."))
            self.assertIsNone(self._find("I'll update my visual memory."))


class ChainLinesTests(unittest.TestCase):
    def test_live_lead_goes(self):
        seen = chain_lines.spoken_keys("Apologies, sir.")
        self.assertEqual(chain_lines.drop_repeated_lead("Apologies, sir.", seen), ("", 1))

    def test_case_tags_and_punctuation_do_not_matter(self):
        seen = chain_lines.spoken_keys("[intent:confirmation] Apologies, sir.")
        self.assertEqual(chain_lines.drop_repeated_lead("[wry] APOLOGIES, SIR!", seen),
                         ("", 1))

    def test_the_rest_is_spoken(self):
        seen = chain_lines.spoken_keys("Apologies, sir.")
        self.assertEqual(
            chain_lines.drop_repeated_lead("Apologies, sir. It's a can.", seen),
            ("It's a can.", 1))

    def test_new_lead_is_untouched(self):
        seen = chain_lines.spoken_keys("Apologies, sir.")
        t = "It's a can. Apologies, sir."
        self.assertEqual(chain_lines.drop_repeated_lead(t, seen), (t, 0))
        self.assertEqual(chain_lines.drop_repeated_lead("x", set()), ("x", 0))


class RepeatLineTests(unittest.TestCase):
    def test_asks(self):
        for t in ("Jarvis, what did you say?", "say that again", "repeat that",
                  "come again", "pardon", "what was that"):
            self.assertTrue(repeat_line.is_repeat_ask(t), t)
        for t in ("what is that", "what's that", "what did you say about the weather",
                  "what am I holding", "repeat the last command"):
            self.assertFalse(repeat_line.is_repeat_ask(t), t)

    def test_word_for_word_and_joined_by_turn(self):
        ll = repeat_line.LastLine()
        ll.note("I'd wager so, sir.", turn=1, now=100.0)
        ll.note("You're quite clearly here.", turn=1, now=103.0)
        self.assertEqual(ll.line_for("what did you say?", now=110.0),
                         "I'd wager so, sir. You're quite clearly here.")
        ll.note("New line.", turn=2, now=120.0)
        self.assertEqual(ll.line_for("what did you say?", now=121.0), "New line.")

    def test_same_turn_far_apart_is_a_new_line(self):
        ll = repeat_line.LastLine()
        ll.note("One.", turn=1, now=100.0)
        ll.note("Two.", turn=1, now=200.0)
        self.assertEqual(ll.line_for("pardon", now=201.0), "Two.")

    def test_windows_and_empty(self):
        ll = repeat_line.LastLine()
        self.assertIsNone(ll.line_for("pardon", now=1.0))
        ll.note("One.", turn=1, now=100.0)
        self.assertIsNone(ll.line_for("what is the time", now=101.0))
        self.assertEqual(ll.line_for("what was that", now=140.0), "One.")
        self.assertIsNone(ll.line_for("what was that", now=150.0))
        self.assertEqual(ll.line_for("what did you say", now=650.0), "One.")
        self.assertIsNone(ll.line_for("what did you say", now=800.0))

    def test_recall_grounding_never_fires_on_one(self):
        with mock.patch.object(cfg, "REPEAT_LAST_LINE", True, create=True):
            self.assertFalse(recall_actions.is_recall_question("Jarvis, what did you say?"))
            # A real recall question still does.
            self.assertTrue(recall_actions.is_recall_question("did you send it?"))
        with mock.patch.object(cfg, "REPEAT_LAST_LINE", False, create=True):
            self.assertTrue(recall_actions.is_recall_question("Jarvis, what did you say?"))


class PlainRetryMoodTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(cfg, "TONE_PLAIN_RETRY_EXEMPT", True, create=True)
        p.start()
        self.addCleanup(p.stop)
        for mod in (tone_detector, ):
            q = mock.patch.object(mod, "_late_night_tone_applies", lambda *a: False)
            q.start()
            self.addCleanup(q.stop)

    def test_forms(self):
        for t in ("Jarvis, look again.", "try again", "check again", "try that again",
                  "please check it again", "could you look at it again", "try again, sir"):
            self.assertTrue(is_plain_retry(t), t)
        for t in ("no, look again", "ugh, try again", "I said try again",
                  "try again, damn it", "open it again", "for the last time, try again",
                  "you did it again", "again"):
            self.assertFalse(is_plain_retry(t), t)

    def test_tone(self):
        self.assertIsNone(detect_tone("Jarvis, look again."))
        self.assertIsNone(detect_tone("try again"))

    def test_real_markers_still_count(self):
        self.assertEqual(detect_tone("I said try again"), "frustrated")
        self.assertEqual(detect_tone("open it again"), "frustrated")
        self.assertEqual(detect_tone("look again!!"), "frustrated")

    def test_the_same_retry_twice_running_counts(self):
        self.assertEqual(detect_tone("look again at it", prev_user_text="look again at it"),
                         "frustrated")
        self.assertEqual(detect_tone("try it again", prev_user_text="try it again"),
                         "frustrated")

    def test_flag_off(self):
        with mock.patch.object(cfg, "TONE_PLAIN_RETRY_EXEMPT", False, create=True):
            self.assertEqual(detect_tone("Jarvis, look again."), "frustrated")

    def test_emotion_tracker(self):
        out = emotion_tracker.classify_emotion("Jarvis, look again.")
        self.assertNotEqual(out.label if hasattr(out, "label") else out, "frustrated")
        out = emotion_tracker.classify_emotion("I said try again")
        self.assertEqual(out.label if hasattr(out, "label") else out, "frustrated")


class ActionFitCameraRuleTests(unittest.TestCase):
    def _rule(self, words, name, arg="", on=True):
        return action_fit.check_rule(words, name, arg, camera_ask=on)[0]

    def test_physical_ask_refuses_the_monitors(self):
        for name in ("see_screen", "recall_screen", "last_screen", "screen_history"):
            self.assertEqual(self._rule("Jarvis, what am I holding?", name),
                             action_fit.RULE_CAMERA_ASK, name)

    def test_the_question_handed_to_it_counts(self):
        self.assertEqual(self._rule("Jarvis, look again.", "see_screen",
                                    "what am I holding?"), action_fit.RULE_CAMERA_ASK)

    def test_screen_asks_fit(self):
        for words in ("what's on my screen", "what is this page", "what is this",
                      "look at this", "look again"):
            self.assertEqual(self._rule(words, "see_screen"), 0, words)

    def test_see_user_fits_and_flag_off(self):
        self.assertEqual(self._rule("what am I holding", "see_user"), 0)
        self.assertEqual(self._rule("what am I holding", "see_screen", on=False), 0)
        self.assertEqual(action_fit.check_rule("what am I holding", "see_screen")[0], 0)


if __name__ == "__main__":
    unittest.main()
