"""core/fast_paths.py: deterministic pre-LLM answers (2026-09-29).

Live evidence this pins (v2.0.115, typed turns):
  * "what did I just ask you" -> "You just asked me what you had previously
    asked me, sir." (the CURRENT utterance was recalled);
  * "what's my name" -> [ACTION: recognize_face] -> "I don't see a face right
    now, sir." (the configured owner name was ignored).

Generic fixtures only (a made-up owner "Alex"). Stdlib unittest, CI-safe: the
monolith is never imported (tests/monolith/test_monolith_fast_paths.py covers
the wiring).

    python -m unittest tests.test_fast_paths
"""
from __future__ import annotations

import ast
import datetime as dt
import json
import os
import unittest

from core import fast_paths as fp

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUE = dt.datetime(2026, 9, 29, 14, 56)


def _hist(*pairs):
    """[(user, assistant), ...] -> conversation_history shape."""
    out = []
    for user, assistant in pairs:
        out.append({"role": "user", "content": user})
        out.append({"role": "assistant", "content": assistant})
    return out


_PRIOR = _hist(("Jarvis, what's the capital of France?", "Paris, sir."))


class NameQuestionTests(unittest.TestCase):
    POSITIVE = ("what's my name", "What is my name?", "whats my name jarvis",
                "Jarvis, what's my name again?", "do you know my name",
                "do you remember my name", "do you know what my name is",
                "you know my name", "tell me my name", "say my name",
                "what am I called", "what's my first name")
    # "who am I" left this list in v2.0.148: it is now answered from the
    # configured name too (IdentityQuestionTests). The presence/recognition
    # questions below still never match: they stay a live camera look.
    NEGATIVE = ("who is this", "who am I looking at",
                "do you recognize me", "do you recognise me",
                "can you see me", "who's at the desk", "who's here",
                "what's my wife's name", "what's my username",
                "what's my wifi name", "my name is Alex",
                "what's my name on the account", "remember my name is Alex")

    def test_name_questions_match(self):
        for text in self.POSITIVE:
            with self.subTest(text=text):
                self.assertTrue(fp.is_name_question(text))
                self.assertEqual(
                    fp.match(text, now=TUE, owner_name="Alex"),
                    fp.FastAnswer("owner-name", "Your name is Alex, sir."))

    def test_identity_and_other_questions_never_match(self):
        for text in self.NEGATIVE:
            with self.subTest(text=text):
                self.assertFalse(fp.is_name_question(text))
                self.assertFalse(fp.is_identity_question(text))
                self.assertIsNone(fp.match(text, now=TUE, owner_name="Alex"))

    def test_no_configured_name_falls_through_and_invents_nothing(self):
        for name in ("", "   ", None):
            with self.subTest(name=name):
                self.assertIsNone(fp.name_reply(name))
                self.assertIsNone(fp.match("what's my name", now=TUE,
                                           owner_name=name))


class IdentityQuestionTests(unittest.TestCase):
    """v2.0.140 live: "who am I" -> recognize_face -> "I don't see a face
    right now, sir." with the owner's name configured. The v1.55 reason for
    the camera route (never CLAIM to see him when he is out of frame) is
    kept: the reply only names who JARVIS is set up for, and presence /
    recognition questions still go to the camera (NameQuestionTests.NEGATIVE)."""

    POSITIVE = (("who am I", "You're Alex, sir."),
                ("Who am I?", "You're Alex, sir."),
                ("Jarvis, who am I", "You're Alex, sir."),
                ("tell me who I am", "You're Alex, sir."),
                ("do you know who I am", "Of course, sir. You're Alex."),
                ("you know who I am", "Of course, sir. You're Alex."),
                ("do you remember who I am", "Of course, sir. You're Alex."))

    def test_identity_questions_answer_from_the_configured_name(self):
        for text, reply in self.POSITIVE:
            with self.subTest(text=text):
                self.assertTrue(fp.is_identity_question(text))
                self.assertFalse(fp.is_name_question(text))
                self.assertEqual(
                    fp.match(text, now=TUE, owner_name="Alex"),
                    fp.FastAnswer("owner-identity", reply))

    def test_the_reply_never_claims_a_sighting(self):
        for text, _reply in self.POSITIVE:
            got = fp.identity_reply(text, "Alex").lower()
            for word in ("see", "camera", "frame", "look", "recogni"):
                self.assertNotIn(word, got)

    def test_no_configured_name_falls_through_and_invents_nothing(self):
        for name in ("", "   ", None):
            with self.subTest(name=name):
                self.assertIsNone(fp.identity_reply("who am I", name))
                self.assertIsNone(fp.match("who am I", now=TUE,
                                           owner_name=name))

    def test_presence_questions_are_not_identity_questions(self):
        for text in ("who am I looking at", "who am I talking to",
                     "who am I speaking with", "who is this", "who's here",
                     "do you recognize me", "can you see me",
                     "do you know who this is", "who am I to you"):
            with self.subTest(text=text):
                self.assertFalse(fp.is_identity_question(text))
                self.assertIsNone(fp.match(text, now=TUE, owner_name="Alex"))


class RecallQuestionTests(unittest.TestCase):
    POSITIVE = (
        ("what did I just ask you", "ask"),
        ("What did I just ask?", "ask"),
        ("jarvis what did i just ask you again", "ask"),
        ("what did I ask you a moment ago", "ask"),
        ("do you remember what I just asked", "ask"),
        ("can you tell me what I just asked", "ask"),
        ("what did I just say", "say"),
        ("what was I just saying", "say"),
        ("what was the last thing I said", "say"),
        ("what was my last question", "question"),
        ("what was my previous question", "question"),
        ("repeat my last question", "question"),
        ("what was my last request", "request"),
        # "what'd" normalises to "whatd" (second review, F1 residual).
        ("what'd I just ask you", "ask"),
        ("what'd I just say", "say"),
    )
    NEGATIVE = ("what did I ask you yesterday", "what did we do yesterday",
                "what was I working on last night", "what did I miss",
                "what did you just say", "say that again", "do that again",
                "what's the date tomorrow", "ask me a question")

    def test_recall_questions_match_with_the_right_wording(self):
        for text, verb in self.POSITIVE:
            with self.subTest(text=text):
                self.assertEqual(fp.recall_verb(text), verb)
                got = fp.match(text, now=TUE, history=_PRIOR)
                self.assertEqual(got.kind, "last-utterance")
                self.assertIn('"what\'s the capital of France"', got.reply)
                self.assertTrue(got.reply.endswith(", sir."))

    def test_other_questions_are_not_recall(self):
        for text in self.NEGATIVE:
            with self.subTest(text=text):
                self.assertIsNone(fp.recall_verb(text))
                got = fp.match(text, now=TUE, history=_PRIOR)
                self.assertTrue(got is None or got.kind != "last-utterance")

    def test_reply_wording(self):
        self.assertEqual(
            fp.last_utterance_reply("what did I just ask you", _PRIOR),
            'You asked me: "what\'s the capital of France", sir.')
        self.assertEqual(
            fp.last_utterance_reply("what did I just say", _PRIOR),
            'You said: "what\'s the capital of France", sir.')
        self.assertEqual(
            fp.last_utterance_reply("what was my last question", _PRIOR),
            'Your last question was: "what\'s the capital of France", sir.')

    def test_nothing_earlier_is_said_honestly(self):
        boot_only = [{"role": "assistant", "content": "Good evening, sir."}]
        for hist in ([], boot_only):
            got = fp.match("what did I just ask you", now=TUE, history=hist)
            self.assertEqual(
                got.reply,
                "You haven't asked me anything else in this conversation "
                "yet, sir.")
        self.assertEqual(
            fp.last_utterance_reply("what was my last question", []),
            "There's no earlier question from you in this conversation, sir.")

    def test_loose_detection_for_the_action_path(self):
        for text in ("what did I just ask you", "user asked what they just asked",
                     "remind me what my last question was",
                     "what was the last thing I said to you"):
            with self.subTest(text=text):
                self.assertTrue(fp.is_last_utterance_question(text, loose=True))
        for text in ("what did we do yesterday", "what did I ask you last night",
                     "what was my last question on Monday",
                     "what did I work on this morning", "q", ""):
            with self.subTest(text=text):
                self.assertFalse(fp.is_last_utterance_question(text, loose=True))


class PriorUtteranceTests(unittest.TestCase):
    def test_the_current_utterance_is_never_recalled(self):
        # The fast path runs BEFORE the turn is appended: history holds only
        # earlier turns, so the newest user entry is the prior one.
        self.assertEqual(fp.prior_owner_utterance(_PRIOR),
                         "what's the capital of France")
        # The LLM path appends the current turn first: skip_newest drops it.
        recorded = _PRIOR + _hist(
            ("remind me what my last question was",
             "[ACTION: session_memory_recall, remind me what my last "
             "question was]"))
        self.assertTrue(fp.recall_turn_recorded(recorded))
        self.assertEqual(fp.prior_owner_utterance(recorded, skip_newest=True),
                         "what's the capital of France")

    def test_asking_twice_recalls_the_real_question(self):
        hist = _PRIOR + _hist(
            ("what did I just ask you",
             'You asked me: "what\'s the capital of France", sir.'))
        got = fp.match("what did I just ask you", now=TUE, history=hist)
        self.assertEqual(got.reply,
                         'You asked me: "what\'s the capital of France", sir.')

    def test_recall_turn_recorded_only_with_the_action_token_after_it(self):
        # Pre-LLM paths (chain resolver / controlled mode): the current turn
        # is NOT in the history, so nothing may be skipped.
        self.assertFalse(fp.recall_turn_recorded(_PRIOR))
        self.assertFalse(fp.recall_turn_recorded([]))
        self.assertFalse(fp.recall_turn_recorded(None))
        followup = _PRIOR + [
            {"role": "user", "content": "what did I just ask"},
            {"role": "assistant", "content": "One moment, sir."},
            {"role": "assistant",
             "content": "[ACTION: session_memory_recall, what did I just ask]"}]
        self.assertTrue(fp.recall_turn_recorded(followup))

    def test_non_user_and_malformed_entries_are_ignored(self):
        hist = [{"role": "user", "content": "open the project notes"},
                {"role": "assistant", "content": "Done, sir."},
                {"role": "user", "content": ["not", "a", "string"]},
                {"role": "user", "content": "   "},
                "garbage", None,
                {"role": "system", "content": "sys"}]
        self.assertEqual(fp.prior_owner_utterance(hist),
                         "open the project notes")

    def test_long_utterances_are_trimmed_for_speech(self):
        hist = _hist(("word " * 100, "ok"))
        got = fp.prior_owner_utterance(hist)
        self.assertLessEqual(len(got), 200)
        self.assertTrue(got.endswith("..."))

    def test_wake_phrases_and_first_recall_questions_are_skipped(self):
        hist = _PRIOR + _hist(
            ("Jarvis", "Yes, sir?"),
            ("what was the first thing I asked you",
             'The first thing you asked me this session was: "x", sir.'),
            ("hey Jarvis, wake up", "At your service, sir."))
        self.assertEqual(fp.prior_owner_utterance(hist),
                         "what's the capital of France")


# The live session (v2.0.140): the first owner turn was "what's 12 times 12".
_SESSION = ["Jarvis", "what did I just ask you", "what's 12 times 12",
            "open the project notes"]


class FirstUtteranceRecallTests(unittest.TestCase):
    """v2.0.140 live: "what was the first thing I asked you in this
    conversation" -> session_memory_recall -> "no access". It is answered
    from the session's opening owner utterances the monolith records."""

    POSITIVE = (
        ("what was the first thing I asked you in this conversation", "ask"),
        ("what was the first thing I asked you today", "ask"),
        ("What was the first thing I asked?", "ask"),
        ("what did I ask you first", "ask"),
        ("what did I ask first this session", "ask"),
        ("what did I first ask you", "ask"),
        ("do you remember what I asked you first", "ask"),
        ("can you tell me the first thing I asked", "ask"),
        ("what was the very first thing I said to you", "say"),
        ("what was the first thing I told you", "say"),
        ("what was my first question", "question"),
        ("what was my first request today", "request"),
        ("what'd I ask you first", "ask"),
    )
    NEGATIVE = ("what was the first thing I asked you yesterday",
                "what did I ask you first last night",
                "what's the first thing on my list",
                "what was the first thing on the agenda",
                "first things first", "ask me first",
                "what did I just ask you", "what was my last question")

    def test_first_recall_questions_match(self):
        for text, verb in self.POSITIVE:
            with self.subTest(text=text):
                self.assertEqual(fp.first_recall_verb(text), verb)
                self.assertTrue(fp.is_first_utterance_question(text))
                got = fp.match(text, now=TUE, history=_PRIOR,
                               session_turns=_SESSION)
                self.assertEqual(got.kind, "first-utterance")
                self.assertIn('"what\'s 12 times 12"', got.reply)
                self.assertTrue(got.reply.endswith(", sir."))

    def test_live_reply(self):
        self.assertEqual(
            fp.match("what was the first thing I asked you in this "
                     "conversation", now=TUE, session_turns=_SESSION),
            fp.FastAnswer(
                "first-utterance",
                'The first thing you asked me this session was: "what\'s 12 '
                'times 12", sir.'))

    def test_other_questions_are_not_first_recall(self):
        for text in self.NEGATIVE:
            with self.subTest(text=text):
                self.assertIsNone(fp.first_recall_verb(text))
                got = fp.match(text, now=TUE, history=_PRIOR,
                               session_turns=_SESSION)
                self.assertTrue(got is None or got.kind != "first-utterance")

    def test_skips_wake_phrases_and_recall_questions(self):
        self.assertEqual(fp.first_owner_utterance(_SESSION),
                         "what's 12 times 12")
        self.assertEqual(
            fp.first_owner_utterance(["hey Jarvis", "Jarvis, wake up",
                                      "what did I ask you first",
                                      "Jarvis, open the project notes."]),
            "open the project notes")

    def test_nothing_earlier_is_said_honestly(self):
        for turns in ([], ["Jarvis"], ["what was the first thing I asked"]):
            with self.subTest(turns=turns):
                self.assertEqual(
                    fp.match("what was the first thing I asked", now=TUE,
                             session_turns=turns).reply,
                    "You haven't asked me anything else this session yet, "
                    "sir.")

    def test_unknown_session_falls_through(self):
        # No session record supplied (None): unknown, so the LLM answers.
        self.assertIsNone(fp.match("what was the first thing I asked you",
                                   now=TUE, history=_PRIOR))
        self.assertIsNone(fp.first_owner_utterance(None))
        self.assertIsNone(fp.first_owner_utterance("not a list"))

    def test_history_is_never_used_for_the_first_thing(self):
        # conversation_history may hold a previous process's tail (blue-green
        # handoff) or be trimmed: the answer comes from session_turns only.
        got = fp.match("what did I ask you first", now=TUE,
                       history=_hist(("an older process's question", "ok")),
                       session_turns=["what's 12 times 12"])
        self.assertIn("what's 12 times 12", got.reply)
        self.assertNotIn("older", got.reply)

    def test_loose_detection_for_the_action_path(self):
        for text in ("user wants the first thing they asked",
                     "the first question I asked today",
                     "what I asked you first"):
            with self.subTest(text=text):
                self.assertTrue(fp.is_first_utterance_question(text,
                                                               loose=True))
        for text in ("the first thing I asked you yesterday",
                     "what did I ask you last night", "first class", ""):
            with self.subTest(text=text):
                self.assertFalse(fp.is_first_utterance_question(text,
                                                                loose=True))


class RecallSkipRuleTests(unittest.TestCase):
    """Review F1 / F2: recall skipped any STORED utterance the LOOSE action-
    argument detectors matched, i.e. anything mentioning "first thing",
    "first message", "last message" or "just say". So "what did I just ask
    you" skipped "what's the first thing on my calendar today" (a regression
    from v2.0.140) and "what was the first thing I asked you" named the
    SECOND utterance. Stored utterances are judged by a tight detector that
    needs a recall lead AND the owner as the one who asked."""

    # Real requests that merely mention first / last / just: recalled.
    REAL = ("what's the first thing on my calendar today",
            "read me the first message in my inbox",
            "remind me to call mom first thing in the morning",
            "what should I do first thing tomorrow",
            "tell me first how the weather looks",
            "what was the first question on the exam",
            "what is the first thing I should do today",
            "read me my last message", "what did the caller just say",
            "cancel my last command", "undo my last request")
    # Recall questions (strict or paraphrased): skipped. The last four were
    # skipped by the base's loose rule and recalled verbatim after the first
    # fix (second review, F1 residual): "what'd" normalises to "whatd", and
    # the yes/no and "go back to" forms have no "what" at all.
    RECALL = ("what did I just ask you", "what was the first thing I asked",
              "remind me what my last question was",
              "what was the last thing I said",
              "what did I ask you at the start of today",
              "what's the earliest thing I asked you today",
              "what did I ask you when you started up",
              "what was my opening question",
              "what'd I just ask you", "what'd I just say",
              "did I just ask you something", "go back to my last question")

    def test_real_requests_are_not_recall_questions(self):
        for text in self.REAL:
            with self.subTest(text=text):
                self.assertFalse(fp._skip_for_recall(text))
                self.assertFalse(fp.is_stored_recall_question(text))

    def test_recall_questions_are_skipped(self):
        for text in self.RECALL:
            with self.subTest(text=text):
                self.assertTrue(fp._skip_for_recall(text))

    def test_just_asked_recalls_a_real_first_thing_request(self):
        # F1 repro (base v2.0.140 answered these; the batch skipped them).
        hist = _hist(("set a timer for five minutes", "Timer set, sir."),
                     ("what's the first thing on my calendar today",
                      "A stand-up at nine, sir."))
        self.assertEqual(
            fp.match("what did I just ask you", now=TUE, history=hist).reply,
            'You asked me: "what\'s the first thing on my calendar today", '
            'sir.')
        for text in self.REAL:
            with self.subTest(text=text):
                got = fp.match("what did I just ask you", now=TUE,
                               history=_hist((text, "Done, sir.")))
                self.assertEqual(got.reply, f'You asked me: "{text}", sir.')
                # The LLM route shares the same rule.
                self.assertEqual(
                    fp.last_utterance_reply(
                        "the last thing the user asked",
                        _hist((text, "Done, sir."),
                              ("what did I just ask you",
                               "[ACTION: session_memory_recall]")),
                        skip_newest=True),
                    f'You asked me: "{text}", sir.')

    def test_just_asked_skips_contracted_and_yes_no_recall_questions(self):
        # Second-review repro: each of these was recalled as "the thing you
        # just asked" (base v2.0.140 skipped them and recalled the lamp).
        for q in ("what'd I just ask you", "what'd I just say",
                  "did I just ask you something",
                  "go back to my last question"):
            with self.subTest(q=q):
                hist = _hist(("turn on the desk lamp", "Done, sir."),
                             (q, "..."))
                self.assertEqual(
                    fp.match("what did I just ask you", now=TUE,
                             history=hist).reply,
                    'You asked me: "turn on the desk lamp", sir.')
                # The action path (skip_newest) shares the rule.
                self.assertEqual(
                    fp.last_utterance_reply(
                        "what did I just ask you",
                        hist + _hist(("what did I just ask you",
                                      "[ACTION: session_memory_recall]")),
                        skip_newest=True),
                    'You asked me: "turn on the desk lamp", sir.')
        # "what'd I just ask you" is now answered on the fast path itself.
        self.assertEqual(
            fp.match("what'd I just ask you", now=TUE,
                     history=_hist(("turn on the desk lamp", "Done, sir."))),
            fp.FastAnswer("last-utterance",
                          'You asked me: "turn on the desk lamp", sir.'))

    def test_first_thing_recalls_a_real_first_request(self):
        # F2 repro: the record used to skip these and name the SECOND
        # utterance (or say nothing was asked).
        self.assertEqual(
            fp.first_owner_utterance(
                ["what's the first thing on my calendar today",
                 "turn on the desk lamp"]),
            "what's the first thing on my calendar today")
        self.assertEqual(
            fp.first_owner_utterance(["read me the first message in my inbox"]),
            "read me the first message in my inbox")
        self.assertEqual(
            fp.first_owner_utterance(["read me my last message",
                                      "open spotify"]),
            "read me my last message")
        self.assertEqual(
            fp.match("what was the first thing I asked you", now=TUE,
                     session_turns=["read me the first message in my inbox"]
                     ).reply,
            'The first thing you asked me this session was: "read me the '
            'first message in my inbox", sir.')

    def test_loose_first_needs_the_asking(self):
        # F3: the action path's paraphrase detector took ANY "first thing",
        # so work-history questions lost the session index.
        for text in ("what was the first thing I worked on today",
                     "what was the first thing we did today",
                     "first thing I did today",
                     "what was the first thing we talked about today",
                     "what's the first thing on my calendar today"):
            with self.subTest(text=text):
                self.assertFalse(
                    fp.is_first_utterance_question(text, loose=True))
        # Second review (F3 residual): someone ELSE asking or saying first
        # is not the owner's first utterance either — JARVIS ("you"), a
        # third party, or a bare pronoun with no user named.
        for text in ("the first thing you said to me today",
                     "what did you tell me first", "what the doctor said first",
                     "first question you asked me today",
                     "what was the first thing you said",
                     "what did my mom say first", "first thing he told me",
                     "the first message they sent"):
            with self.subTest(text=text):
                self.assertFalse(
                    fp.is_first_utterance_question(text, loose=True))
        for text in ("the first thing the user asked today",
                     "first thing I asked", "the user's first question this "
                     "session", "his first request", "what I asked you first",
                     "the earliest thing I asked", "what did the user say first",
                     "user wants the first thing he asked"):
            with self.subTest(text=text):
                self.assertTrue(
                    fp.is_first_utterance_question(text, loose=True))

    def test_after_a_handoff_nothing_asked_is_never_claimed(self):
        # F6 repro: the history holds the previous process's tail, the record
        # only the current question. Never "you haven't asked me anything",
        # and never a tail line passed off as the first thing either.
        tail = _hist(("set a timer for ten minutes", "Timer set, sir."))
        q = "what was the first thing I asked you today"
        got = fp.match(q, now=TUE, history=tail, session_turns=[q])
        self.assertEqual(got, fp.FastAnswer(
            "first-utterance",
            "I no longer have the start of this session on record, sir."))
        # With a carried-over record the real first thing is named.
        got = fp.match(q, now=TUE, history=tail,
                       session_turns=["what's 12 times 12", q])
        self.assertIn('"what\'s 12 times 12"', got.reply)
        # With no history either, "nothing earlier" is still the answer.
        self.assertEqual(
            fp.match(q, now=TUE, history=[], session_turns=[q]).reply,
            "You haven't asked me anything else this session yet, sir.")

    def test_a_lost_start_never_names_a_later_turn(self):
        # Second review (F5/F6 residual): after a forget / reset purged the
        # first utterance (or a handoff came without it), the monolith
        # latches session_start_lost; the next ordinary utterance in the
        # record must never be recited as the first thing he asked.
        unknown = "I no longer have the start of this session on record, sir."
        q = "what was the first thing I asked you"
        self.assertEqual(
            fp.match(q, now=TUE, history=[],
                     session_turns=["turn on the desk lamp"],
                     session_start_lost=True),
            fp.FastAnswer("first-utterance", unknown))
        self.assertEqual(
            fp.first_utterance_reply(q, ["turn on the desk lamp"],
                                     start_lost=True), unknown)
        # Not latched: the record is the answer, as before.
        self.assertIn('"turn on the desk lamp"', fp.match(
            q, now=TUE, session_turns=["turn on the desk lamp"]).reply)
        # Only a real True latches (a stray truthy value never does).
        self.assertIn('"turn on the desk lamp"', fp.match(
            q, now=TUE, session_turns=["turn on the desk lamp"],
            session_start_lost="yes").reply)


class WakeOnlyTests(unittest.TestCase):
    def test_wake_phrases(self):
        for text in ("Jarvis", "Jarvis.", "hey Jarvis", "OK Jarvis",
                     "Jarvis, wake up", "wake up", "are you there",
                     "Jarvis, are you there?", "hey jarvis you there"):
            with self.subTest(text=text):
                self.assertTrue(fp.is_wake_only(text))
        for text in ("Jarvis, open the notes", "hello", "wake me up at 7",
                     "are you there yet", "", None, 42):
            with self.subTest(text=text):
                self.assertFalse(fp.is_wake_only(text))

    def test_a_wake_behind_lead_interjections(self):
        # 2026-10-01: the wake-word gate admits "Um, Jarvis" (core.wake_prefix,
        # word 1-3); recall must skip it like a bare "Jarvis", not recite it.
        for text in ("Um, Jarvis.", "So Jarvis", "what jarvis?",
                     "uh okay Jarvis, are you there?", "Alright Jarvis, wake up"):
            with self.subTest(text=text):
                self.assertTrue(fp.is_wake_only(text))
        for text in ("Um, Jarvis, open the notes", "I asked Jarvis",
                     "so um uh Jarvis"):
            with self.subTest(text=text):
                self.assertFalse(fp.is_wake_only(text))
        hist = [{"role": "user", "content": "open the project notes"},
                {"role": "assistant", "content": "Done, sir."},
                {"role": "user", "content": "Um, Jarvis."},
                {"role": "assistant", "content": "Yes, sir?"}]
        self.assertEqual(fp.prior_owner_utterance(hist),
                         "open the project notes")


class MatchTests(unittest.TestCase):
    def test_date_questions_route_through_date_math(self):
        self.assertEqual(
            fp.match("what's the date tomorrow", now=TUE),
            fp.FastAnswer("date",
                          "Tomorrow is Wednesday, September 30, 2026, sir."))
        self.assertEqual(fp.match("how many days until christmas",
                                  now=TUE).kind, "days-until")

    def test_no_clock_means_no_date_answers(self):
        self.assertIsNone(fp.match("what's the date tomorrow", now=None))
        self.assertIsNone(fp.match("what time is it in London", now=None))

    def test_next_weekday_routes_through_date_math(self):
        # v2.0.140 live: "what's the date next Monday" -> "September 29th"
        self.assertEqual(
            fp.match("what's the date next Monday", now=TUE),
            fp.FastAnswer("date-of", "Next Monday is October 5, 2026, sir."))

    def test_world_clock_questions_route_through_world_clock(self):
        try:
            from zoneinfo import ZoneInfo
            now = dt.datetime(2026, 9, 29, 22, 17,
                              tzinfo=ZoneInfo("America/Chicago"))
        except Exception:   # pragma: no cover - no zone data
            self.skipTest("no IANA time zone data")
        self.assertEqual(
            fp.match("what time is it in London", now=now),
            fp.FastAnswer("world-clock", "It's 4:17 AM in London, sir. "
                                         "That's Wednesday there."))
        # An unknown place is left to the normal turn.
        self.assertIsNone(fp.match("what time is it in Narnia", now=now))

    def test_commands_fall_through(self):
        for text in ("remind me tomorrow to call the office",
                     "what's the weather tomorrow", "play music until friday",
                     "open the garage notes", "", None, 42):
            with self.subTest(text=text):
                self.assertIsNone(fp.match(text, now=TUE, history=_PRIOR,
                                           owner_name="Alex"))

    def test_never_raises(self):
        # A broken history object still yields an answer or None, never a raise.
        got = fp.match("what did I just ask you", now=TUE, history=object())
        self.assertIsNone(got)


class ConfigAndSettingsWiringTests(unittest.TestCase):
    """FAST_PATHS_ENABLED is wired like PROCESSING_FILLER_ENABLED: a
    core/config.py literal, a Settings-GUI schema row and the example json."""

    def _config_literals(self):
        with open(os.path.join(_ROOT, "core", "config.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        lits = {}
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and isinstance(node.value,
                                                           ast.Constant):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        lits[tgt.id] = node.value.value
        return src, lits

    def test_config_default_is_on_and_says_when_it_applies(self):
        src, lits = self._config_literals()
        self.assertIs(lits["FAST_PATHS_ENABLED"], True)
        head = src[:src.index("\nFAST_PATHS_ENABLED = True")]
        block = head[head.rindex("\n\n"):].strip().splitlines()
        self.assertTrue(block and all(ln.startswith("#") for ln in block),
                        block)
        self.assertIn("on the next start",
                      " ".join(ln.lstrip("# ") for ln in block))

    def test_settings_schema_row_and_example_json(self):
        import importlib.util
        path = os.path.join(_ROOT, "tools", "settings_window.py")
        spec = importlib.util.spec_from_file_location("sw_fast_paths", path)
        sw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sw)
        row = sw.SCHEMA["FAST_PATHS_ENABLED"]
        self.assertEqual(row["type"], "bool")
        self.assertIs(row["default"], True)
        self.assertIn("next start", row["help"])
        self.assertIn("FAST_PATHS_ENABLED", sw.persisted_keys())
        with open(os.path.join(_ROOT, "tools", "user_settings.example.json"),
                  encoding="utf-8") as fh:
            self.assertIs(json.load(fh)["FAST_PATHS_ENABLED"], True)


class PromptGuardTests(unittest.TestCase):
    def test_cloud_prompt_keeps_whats_my_name_off_the_camera(self):
        from core import prompts
        text = prompts.PC_CONTROL_PROMPT
        face = text.index("FACE RECOGNITION")
        guard = text.index("\"What's my name\" is NOT a camera look", face)
        # It sits inside the face-recognition block, after the identity rule.
        self.assertLess(text.index("IDENTITY + PRESENCE ARE A LIVE CAMERA LOOK",
                                   face), guard)
        self.assertIn("Never guess a name.", text[guard:guard + 200])



class TimerListRequestTests(unittest.TestCase):
    """2026-10-01: "list_timers can make things up" — a whole-utterance
    timer question is answered from the timer store by the monolith's
    _run_timer_list_shortcut, never by the model. These pin the classifier."""

    def test_timer_list_questions(self):
        for q in ("what timers do I have", "what timers are running",
                  "list my timers", "show me my timers", "Jarvis, any timers?",
                  "any timers running", "do I have any timers set",
                  "are there any reminders", "what reminders do I have",
                  "check my timers please", "how much time is left on my timer",
                  "how long is left on the tea timer",
                  "what's left on my timer", "when does my timer go off"):
            with self.subTest(q=q):
                self.assertTrue(fp.is_timer_list_request(q))

    def test_not_timer_list_questions(self):
        for q in ("set a timer for 5 minutes", "cancel my timer",
                  "remind me in 10 minutes to stretch", "what time is it",
                  "what times does the store open", "list my schedules",
                  "kill all timers", "", None):
            with self.subTest(q=q):
                self.assertFalse(fp.is_timer_list_request(q))

    def test_match_never_answers_it_itself(self):
        # The answer is the live timer store's; match() is pure.
        self.assertIsNone(fp.match("what timers do I have"))


if __name__ == "__main__":
    unittest.main()
