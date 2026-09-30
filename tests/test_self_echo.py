"""core/self_echo.py — JARVIS must never answer his own voice (R9, 2026-09-29).

Pure-stdlib unit tests of the timing and content layers. Every test freezes
the module clock (``_clock``) and passes explicit timestamps, so nothing here
sleeps or depends on wall time. The monolith wiring (record_speech,
play_with_lipsync, _speak, the main-loop / standby gate) is pinned by
tests/monolith/test_monolith_self_echo.py.

    python -m unittest tests.test_self_echo
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import self_echo as se  # noqa: E402

_LINE = "At your service, sir."


class _Clocked(unittest.TestCase):
    def setUp(self):
        se._reset_for_tests()
        self.addCleanup(se._reset_for_tests)
        self.t = [1000.0]
        patcher = mock.patch.object(se, "_clock", lambda: self.t[0])
        patcher.start()
        self.addCleanup(patcher.stop)


class TimingLayerTests(_Clocked):
    """capture_overlap(open, vad, end): the live incident and its neighbours."""

    def _play(self, start, end, text=_LINE):
        tok = se.playback_begin(text, at=start)
        se.playback_end(tok, at=end)
        return tok

    def test_live_incident_capture_open_while_another_thread_speaks(self):
        # The loop was listening from 100; the tray line played 101.0-102.6;
        # the VAD tripped on it at 101.2.
        self._play(101.0, 102.6)
        hit = se.capture_overlap(100.0, 101.2, 104.0, tail_s=0.8, at=104.0)
        self.assertIsNotNone(hit)
        self.assertLess(hit["gap"], 0.0)          # began while still playing
        self.assertFalse(hit["says_jarvis"])
        self.assertEqual(hit["count"], 1)

    def test_still_playing_counts_as_overlap(self):
        se.playback_begin(_LINE, at=101.0)        # never ended yet
        self.assertTrue(se.playback_live())
        hit = se.capture_overlap(100.0, 101.5, 101.9, at=102.0)
        self.assertIsNotNone(hit)

    def test_utterance_starting_inside_the_tail_is_an_echo(self):
        self._play(101.0, 102.6)
        hit = se.capture_overlap(100.0, 103.2, 105.0, tail_s=0.8, at=105.0)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit["gap"], 0.6, places=6)

    def test_owner_after_the_tail_passes(self):
        self._play(101.0, 102.6)
        self.assertIsNone(
            se.capture_overlap(100.0, 103.5, 105.0, tail_s=0.8, at=105.0))

    def test_tail_is_tuneable(self):
        self._play(101.0, 102.6)
        self.assertIsNone(
            se.capture_overlap(100.0, 103.2, 105.0, tail_s=0.3, at=105.0))
        self.assertIsNotNone(
            se.capture_overlap(100.0, 103.2, 105.0, tail_s=1.0, at=105.0))

    def test_capture_opened_after_the_line_is_not_tail_gated(self):
        # Speak-then-listen on the main thread: a quick "yes" to his own
        # question must get through even inside the tail.
        self._play(101.0, 102.6)
        self.assertIsNone(
            se.capture_overlap(102.7, 102.9, 103.5, tail_s=0.8, at=103.5))

    def test_owner_finished_before_the_line_started_passes(self):
        self._play(110.0, 111.0)
        self.assertIsNone(se.capture_overlap(100.0, 101.0, 103.0, at=112.0))

    def test_playback_starting_mid_utterance_is_an_overlap(self):
        self._play(102.0, 103.0)
        self.assertIsNotNone(se.capture_overlap(100.0, 101.0, 104.0,
                                                at=104.0))

    def test_line_saying_jarvis_is_flagged(self):
        self._play(101.0, 102.0, text="Say JARVIS when you need me.")
        hit = se.capture_overlap(100.0, 101.2, 103.0, at=103.0)
        self.assertTrue(hit["says_jarvis"])

    def test_playback_text_is_not_stored(self):
        tok = se.playback_begin("a private line", at=1.0)
        self.assertNotIn("a private line", repr(se._playbacks[tok]))

    def test_old_playbacks_are_pruned(self):
        self._play(1.0, 2.0)
        se.playback_begin("", at=2.0 + se._PLAYBACK_KEEP_S + 1.0)
        self.assertEqual(len(se._playbacks), 1)

    def test_never_raises(self):
        self.assertIsNone(se.capture_overlap("x", None, object()))
        se.playback_end(987654)                    # unknown token
        self.assertEqual(se.playback_begin(None, at="bad"), 0)


class ContentLayerTests(_Clocked):
    def _say(self, text, at=None):
        tok = se.remember(text, at=self.t[0] if at is None else at)
        se.refresh(tok, at=self.t[0] if at is None else at)
        return tok

    def test_exact_echo_is_matched(self):
        self._say(_LINE)
        self.t[0] += 5.0
        score, age = se.match("At your service, sir.")
        self.assertEqual(score, 1.0)
        self.assertAlmostEqual(age, 5.0)

    def test_misheard_echo_is_matched(self):
        self._say(_LINE)
        hit = se.match("at your services sir")
        self.assertIsNotNone(hit)
        self.assertGreaterEqual(hit[0], se.FUZZY_MIN_RATIO)

    def test_whisper_tail_is_stripped(self):
        self._say(_LINE)
        self.assertIsNotNone(se.match("Um, at your service, sir. Thank you."))

    def test_unrelated_owner_command_passes(self):
        self._say(_LINE)
        self.assertIsNone(se.match("what's on my calendar today"))

    def test_owner_extending_the_line_passes(self):
        self._say("Turning off the desk lamp.")
        self.assertIsNone(
            se.match("turning off the desk lamp was the wrong call, undo it"))

    def test_short_lines_are_exact_only(self):
        self._say("Done, sir.")
        self.assertIsNotNone(se.match("done sir"))
        self.assertIsNone(se.match("dune sir"))
        self.assertIsNone(se.match("done sir thanks"))

    def test_owner_vocabulary_is_protected(self):
        self._say("Okay.")
        self.assertIsNone(se.match("okay"))

    def test_caller_protected_phrases(self):
        self._say("Very good.")
        self.assertIsNotNone(se.match("very good"))
        self.assertIsNone(se.match("Very good!", protected={"very good"}))

    def test_stop_word_always_passes(self):
        self._say("Stop the timer when it rings, sir.")
        self.assertIsNone(se.match("stop the timer when it rings sir"))
        self._say("Stop.")
        self.assertIsNone(se.match("stop"))

    def test_each_sentence_of_a_long_line_is_remembered(self):
        self._say("The build finished. Two tests failed in the parser module.")
        self.assertIsNotNone(se.match("two tests failed in the parser module"))

    def test_lines_expire_after_the_window(self):
        self._say(_LINE)
        self.t[0] += 19.9
        self.assertIsNotNone(se.match(_LINE, window_s=20.0))
        self.t[0] += 0.2
        self.assertIsNone(se.match(_LINE, window_s=20.0))
        self.assertEqual(len(se._lines), 0)        # pruned, not just skipped

    def test_window_counts_from_the_end_of_the_line(self):
        tok = se.remember(_LINE, at=self.t[0])
        self.t[0] += 30.0                          # a 30 s reply
        se.refresh(tok)
        self.t[0] += 15.0
        self.assertIsNotNone(se.match(_LINE, window_s=20.0))

    def test_line_finished_before_the_clip_is_not_compared(self):
        # The owner re-issuing a command JARVIS just acknowledged, in a
        # capture that opened after the line ended, is not an echo.
        tok = se.remember("Turning off the desk lamp, sir.", at=100.0)
        se.refresh(tok, at=101.6)
        self.t[0] = 103.0
        self.assertIsNotNone(se.match("turn off the desk lamp"))  # no bound
        self.assertIsNone(se.match("turn off the desk lamp",
                                   clip_start=101.7))
        # A clip that began while the line was still playing still matches.
        self.assertIsNotNone(se.match("turning off the desk lamp sir",
                                      clip_start=101.5))

    def test_line_still_playing_is_always_compared(self):
        se.remember(_LINE, at=100.0)               # never refreshed yet
        self.t[0] = 105.0
        self.assertIsNotNone(se.match(_LINE, clip_start=104.0))

    def test_empty_and_junk_input(self):
        self.assertEqual(se.remember("  ...  "), 0)
        self.assertIsNone(se.match(""))
        self.assertIsNone(se.match(None))
        self.assertIsNone(se.match(_LINE, window_s="bad"))
        self.assertIsNone(se.match(_LINE, clip_start="bad"))
        se.refresh(0)
        se.refresh(424242)

    def test_memory_is_bounded(self):
        for i in range(se._MAX_LINES + 10):
            se.remember(f"line number {i} of many", at=self.t[0])
        self.assertEqual(len(se._lines), se._MAX_LINES)


if __name__ == "__main__":
    unittest.main()
