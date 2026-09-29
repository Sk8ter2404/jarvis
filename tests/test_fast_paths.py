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
    NEGATIVE = ("who am I", "who is this", "who am I looking at",
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
                self.assertIsNone(fp.match(text, now=TUE, owner_name="Alex"))

    def test_no_configured_name_falls_through_and_invents_nothing(self):
        for name in ("", "   ", None):
            with self.subTest(name=name):
                self.assertIsNone(fp.name_reply(name))
                self.assertIsNone(fp.match("what's my name", now=TUE,
                                           owner_name=name))


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


if __name__ == "__main__":
    unittest.main()
