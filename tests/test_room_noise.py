"""core.room_noise - "how bad is the background noise?" from a READING.

Live 2026-10-19 10:45 the answer was an invented "roughly 42% of the current
audio is non-essential interference". No mic or audio session is touched:
capture and the PC peak are fakes.

    python -m unittest tests.test_room_noise
"""
from __future__ import annotations

import unittest

import numpy as np

from core import room_noise as R


class QuestionTests(unittest.TestCase):
    def test_room_noise_questions(self):
        for t in ("Jarvis, how bad is the background noise?",
                  "is it noisy in here",
                  "how loud is it in here",
                  "can you hear me over the noise?",
                  "Jarvis is there much background noise",
                  "what's the noise level like"):
            with self.subTest(t=t):
                self.assertTrue(R.is_room_noise_question(t))

    def test_not_room_noise_questions(self):
        for t in ("play some white noise",
                  "turn on noise cancelling",
                  "Jarvis, reduce the background noise",
                  "what was that noise?",
                  "what made that noise",
                  "the background noise is annoying",      # a remark, not a question
                  "how bad is the traffic",
                  ""):
            with self.subTest(t=t):
                self.assertFalse(R.is_room_noise_question(t))


class DescribeTests(unittest.TestCase):
    def test_quiet_noisy_and_in_between_say_what_was_measured(self):
        self.assertEqual(R.describe(0.001, 0.0),
                         "It's quiet in here, sir, about -60 dB on the mic over the "
                         "last second, and nothing's playing on the PC.")
        self.assertIn("under the level I treat as speech", R.describe(0.004, None))
        noisy = R.describe(0.05, 0.3)
        self.assertIn("It's noisy, sir, about -26 dB", noisy)
        self.assertTrue(noisy.endswith("your PC is playing sound right now."))

    def test_nothing_read_is_never_guessed(self):
        self.assertEqual(R.describe(None, None),
                         "I couldn't take a reading of the room just now, sir.")
        self.assertEqual(R.describe(None, 0.0),
                         "I couldn't read the microphone just now, sir, and nothing's "
                         "playing on the PC.")
        for s in (R.describe(0.001, 0.0), R.describe(0.05, 0.5), R.describe(None, None)):
            self.assertNotIn("%", s)


class MeasureTests(unittest.TestCase):
    def test_measure_reads_rms_and_peak(self):
        tone = np.full(16000, 0.01, dtype=np.float32)
        rms, peak = R.measure(lambda s: tone, lambda: 0.2)
        self.assertAlmostEqual(rms, 0.01, places=5)
        self.assertEqual(peak, 0.2)

    def test_measure_never_raises(self):
        def boom(*a):
            raise RuntimeError("mic busy")
        self.assertEqual(R.measure(boom, boom), (None, None))
        self.assertEqual(R.measure(lambda s: None, lambda: None), (None, None))
        self.assertEqual(R.measure(lambda s: np.zeros(0), lambda: None), (None, None))

    def test_the_action_asks_for_one_second(self):
        asked = []
        R.room_noise(lambda s: asked.append(s) or np.zeros(10), lambda: 0.0)
        self.assertEqual(asked, [R.SAMPLE_S])


if __name__ == "__main__":
    unittest.main()
