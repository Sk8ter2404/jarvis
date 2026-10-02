"""core/spoken_math.py + its two consumers (2026-10-01).

The 09-05 live diagnostic: "what's 12 times 7" never reached the calculator.
The local prompt router loads PYTHON SANDBOX (run_python) only on
"calculate" / "compute" / "math" / "python", so an operator WORD never put
the calculator in front of the model, and the local brain answered from its
head. These pin that spoken operator words with a number on BOTH sides
(a) route the calculator section and (b) are answered exactly by the fast
path — while "times" in an ordinary sentence ("what times does the store
open") stays ordinary.

Stdlib unittest, CI-safe: the monolith is never imported.

    python -m unittest tests.test_spoken_math
"""
from __future__ import annotations

import unittest
from fractions import Fraction

from core import fast_paths as fp
from core import prompt_router as pr
from core import prompts
from core import spoken_math as sm

FULL = prompts.PC_CONTROL_PROMPT


class AnswerTests(unittest.TestCase):
    def _reply(self, q):
        got = sm.answer(q)
        self.assertIsNotNone(got, f"{q!r} was not answered")
        return got.reply

    def test_the_five_operator_words(self):
        for q, want in (("what's 12 times 7", "12 times 7 is 84, sir."),
                        ("what is 144 divided by 12",
                         "144 divided by 12 is 12, sir."),
                        ("what's 5 plus 3", "5 plus 3 is 8, sir."),
                        ("what is 10 minus 4", "10 minus 4 is 6, sir."),
                        ("what is 2 to the power of 10",
                         "2 to the power of 10 is 1024, sir.")):
            with self.subTest(q=q):
                self.assertEqual(self._reply(q), want)

    def test_other_spellings_of_the_operators(self):
        for q, value in (("what's 12 x 7", 84), ("12 * 7", 84),
                         ("what is 6 multiplied by 7", 42),
                         ("how much is 15 over 4", Fraction(15, 4)),
                         ("what's 2 raised to the power of 8", 256),
                         ("what's 2 to the 10th", 1024),
                         ("what is 2 raised to the 10th power", 1024),
                         ("what's 9 squared", 81), ("what's 3 cubed", 27),
                         ("what's 2 ^ 5", 32), ("what is 10 - 4", 6),
                         ("what's 144 / 12", 12),
                         ("calculate 1,000 times 2.5", 2500)):
            with self.subTest(q=q):
                got = sm.answer(q)
                self.assertIsNotNone(got, q)
                self.assertEqual(got.value, value)

    def test_number_words_from_speech_to_text(self):
        for q, value in (("twelve times seven", 84),
                         ("two hundred and fifty divided by five", 50),
                         ("what is one point five times four", 6),
                         ("a thousand minus one", 999),
                         ("what's twenty-one plus nine", 30),
                         ("what is negative five times three", -15)):
            with self.subTest(q=q):
                self.assertEqual(sm.answer(q).value, value)

    def test_precedence_and_associativity(self):
        self.assertEqual(sm.answer("2 plus 3 times 4").value, 14)
        self.assertEqual(sm.answer("20 minus 6 minus 4").value, 10)
        self.assertEqual(sm.answer("2 to the power of 3 to the power of 2").value,
                         512)        # right-associative: 2 ** 9
        self.assertEqual(sm.answer("100 divided by 10 divided by 2").value, 5)

    def test_exact_arithmetic_and_honest_rounding(self):
        self.assertEqual(self._reply("what is 0.1 plus 0.2"),
                         "0.1 plus 0.2 is 0.3, sir.")
        self.assertEqual(self._reply("what's 1 divided by 8"),
                         "1 divided by 8 is 0.125, sir.")
        self.assertEqual(self._reply("what is 10 divided by 3"),
                         "10 divided by 3 is about 3.3333, sir.")
        self.assertEqual(self._reply("what's 3 minus 10"),
                         "3 minus 10 is negative 7, sir.")

    def test_wake_word_politeness_and_tails(self):
        for q in ("Jarvis, what's 12 times 7?", "hey jarvis what is 12 times 7",
                  "what's 12 times 7 sir", "what does 12 times 7 equal",
                  "what would 12 times 7 be", "12 times 7 please"):
            with self.subTest(q=q):
                self.assertEqual(sm.answer(q).value, 84)

    def test_division_by_zero_is_said_not_raised(self):
        reply = self._reply("what's 10 divided by 0")
        self.assertIn("undefined", reply)
        self.assertIsNone(sm.answer("what's 10 divided by 0").value)

    def test_huge_answers_are_said_in_powers_of_ten(self):
        self.assertEqual(
            self._reply("what's 2 to the power of 100"),
            "2 to the power of 100 is about 1.27 times 10 to the power of 30, "
            "sir.")
        # An absurd exponent is not computed at all (falls to the LLM).
        self.assertIsNone(sm.answer("what's 10 to the power of 5000"))

    def test_an_unspaced_slash_or_hyphen_is_left_to_the_llm(self):
        # Dates, idioms and codes: "9/11", "12/25", "24/7", "10-4". The router
        # still shows the model run_python for them.
        for q in ("what is 9/11", "what is 12/25", "what's 24/7",
                  "what is 10-4", "what's 144/12"):
            with self.subTest(q=q):
                self.assertIsNone(sm.answer(q))
                self.assertTrue(sm.is_arithmetic_request(q))

    def test_not_arithmetic_is_not_answered(self):
        for q in ("what times does the store open", "three times a day",
                  "play it two times", "what's 12 times 7 plus the tip",
                  "set a timer for 5 minutes", "12 times", "times 7",
                  "what time is it", "add milk to the list", "", None, 7):
            with self.subTest(q=q):
                self.assertIsNone(sm.answer(q))


class IsArithmeticRequestTests(unittest.TestCase):
    def test_numbers_on_both_sides_of_an_operator(self):
        for q in ("what's 12 times 7", "twelve times seven",
                  "what's 12 times 7 plus the tip", "is 2 to the power of 10 big",
                  "what is 144 divided by 12 again", "9 squared",
                  "0 to the power of -1", "what is 2 raised to the 10th power"):
            with self.subTest(q=q):
                self.assertTrue(sm.is_arithmetic_request(q))

    def test_an_operator_word_alone_is_not_arithmetic(self):
        for q in ("what times does the store open", "three times a day",
                  "add the plus size jacket", "turn it over",
                  "minus the drama", "version 2.0.157", "", None):
            with self.subTest(q=q):
                self.assertFalse(sm.is_arithmetic_request(q))


class CalculatorRoutingTests(unittest.TestCase):
    """The router half: the turn must put run_python in front of the model."""

    def setUp(self):
        _core, self.sections = pr.split_pc_control(FULL)

    def test_operator_words_load_the_calculator_section(self):
        for q in ("what's 12 times 7", "what is 144 divided by 12",
                  "what's 5 plus 3", "what is 10 minus 4",
                  "what is 2 to the power of 10",
                  "what's 12 times 7 plus a 20 percent tip"):
            with self.subTest(q=q):
                inc, _ = pr.select_sections(q, self.sections)
                self.assertIn("PYTHON SANDBOX", inc)
                self.assertIn("run_python", pr.slim_pc_control(q, FULL))
                self.assertIn("run_python", pr.turn_pc_block(q, FULL))

    def test_times_in_an_ordinary_sentence_does_not(self):
        for q in ("what times does the store open", "three times a day"):
            with self.subTest(q=q):
                inc, _ = pr.select_sections(q, self.sections)
                self.assertNotIn("PYTHON SANDBOX", inc)


class FastPathTests(unittest.TestCase):
    def test_arithmetic_is_answered_before_the_llm(self):
        hit = fp.match("what's 12 times 7")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.kind, "arithmetic")
        self.assertEqual(hit.reply, "12 times 7 is 84, sir.")

    def test_answered_with_or_without_a_clock(self):
        import datetime as dt
        now = dt.datetime(2026, 9, 29, 14, 56)
        self.assertEqual(fp.match("what is 2 to the power of 10", now=now).kind,
                         "arithmetic")

    def test_ordinary_times_falls_through(self):
        self.assertIsNone(fp.match("what times does the store open"))


if __name__ == "__main__":
    unittest.main()
