"""core/fast_paths.py: "what time is it" answers instantly (2026-10-02 live).

Live 10:27:33: "Jarvis, what time is it?" went through the LLM - "One
moment, sir. [ACTION: get_time]", a second round, ~4 s - while "what day is
it" and "what time is it in London" were answered with no LLM at all.

The wake word, the "?" and the wake-mode canonicalisation were NOT the
cause: date_math.normalize already peels a leading / trailing "Jarvis", the
punctuation and "right now", and the fast paths run last among the voice
shortcuts, before any LLM call. There was simply no grammar for the LOCAL
time: date_math answers dates only, and world_clock answers a time only for
a named place or zone (its own tests pin "what time is it" -> None, leaving
it to get_time). Fix: a whole-utterance local-time grammar in fast_paths,
answered from the same clock seam (``now``).

Clock frozen; stdlib only (light tier).

    python -m unittest tests.test_fast_paths_local_time
"""
from __future__ import annotations

import datetime as dt
import unittest

from core import fast_paths as fp

FRI = dt.datetime(2026, 10, 2, 10, 27)


class LocalTimeAnswerTests(unittest.TestCase):
    def test_the_four_live_shapes(self):
        for text in ("Jarvis, what time is it?", "jarvis what time is it",
                     "What time is it, Jarvis?", "Jarvis what's the time"):
            with self.subTest(text=text):
                self.assertEqual(fp.match(text, now=FRI),
                                 fp.FastAnswer("time", "It's 10:27 AM, sir."))

    def test_other_plain_time_questions(self):
        for text in ("what time is it", "what time is it now",
                     "what time is it right now", "what's the time now",
                     "what's the current time", "what is the time please",
                     "do you know what time it is",
                     "can you tell me what time it is",
                     "could you tell me the time", "tell me the time",
                     "what time have you got", "do you have the time",
                     "hey jarvis what time is it here",
                     "what's the local time", "time check",
                     "Jarvis, what time is it sir?"):
            with self.subTest(text=text):
                got = fp.match(text, now=FRI)
                self.assertIsNotNone(got)
                self.assertEqual(got.kind, "time")
                self.assertEqual(got.reply, "It's 10:27 AM, sir.")

    def test_date_and_time_together(self):
        for text in ("what's the date and time", "what is the time and date",
                     "Jarvis, what's the time and date?"):
            with self.subTest(text=text):
                self.assertEqual(
                    fp.match(text, now=FRI),
                    fp.FastAnswer("time", "It's 10:27 AM on Friday, "
                                          "October 2, 2026, sir."))

    def test_the_clock_formats(self):
        for when, said in ((dt.datetime(2026, 10, 2, 0, 5), "12:05 AM"),
                           (dt.datetime(2026, 10, 2, 12, 0), "12:00 PM"),
                           (dt.datetime(2026, 10, 2, 13, 7), "1:07 PM"),
                           (dt.datetime(2026, 10, 2, 23, 59), "11:59 PM")):
            with self.subTest(when=when):
                self.assertEqual(fp.match("what time is it", now=when).reply,
                                 f"It's {said}, sir.")

    def test_no_clock_means_no_answer(self):
        self.assertIsNone(fp.match("what time is it", now=None))


class NotALocalTimeQuestionTests(unittest.TestCase):
    def test_world_clock_questions_keep_their_answer(self):
        try:
            from zoneinfo import ZoneInfo
            now = dt.datetime(2026, 10, 2, 10, 27,
                              tzinfo=ZoneInfo("America/Chicago"))
        except Exception:   # pragma: no cover - no zone data
            self.skipTest("no IANA time zone data")
        self.assertEqual(fp.match("what time is it in London", now=now).kind,
                         "world-clock")

    def test_other_questions_and_commands_fall_through(self):
        for text in ("what time is it there", "what time is it in Narnia",
                     "what time is my meeting", "what time does the store open",
                     "what time should I leave", "what time is the game",
                     "what time zone am I in", "time to go",
                     "set a timer for ten minutes", "what times are you open",
                     "what time is it and what's the weather",
                     "remind me what time it is in an hour",
                     "what time did I go to bed", "the time machine",
                     "what time", "time", ""):
            with self.subTest(text=text):
                got = fp.match(text, now=FRI)
                self.assertTrue(got is None or got.kind != "time", got)


if __name__ == "__main__":
    unittest.main()
