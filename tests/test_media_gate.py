"""Tests for core/media_gate.py (light CI: pure, fakes only).

THE LIVE FAILURE (2026-10-01 22:28:50): an Instagram reel playing on this PC said
"Jarvis, find me a restaurant ... build a website" and JARVIS ran it. While the PC
plays audio, a mic turn in a voice that is confidently not the owner's is dropped.

    python -m unittest tests.test_media_gate
"""
from __future__ import annotations

import unittest

from core import learn_gate as lg
from core import media_gate as mg


class _Meter:
    def __init__(self, *values, boom=False):
        self.values = list(values)
        self.calls = 0
        self.boom = boom

    def GetPeakValue(self):
        self.calls += 1
        if self.boom:
            raise OSError("COM went away")
        if len(self.values) > 1:
            return self.values.pop(0)
        return self.values[0]


def _sessions(*pairs):
    cleaned = []

    def fn():
        return list(pairs), (lambda: cleaned.append(True))
    fn.cleaned = cleaned
    return fn


class PeakTests(unittest.TestCase):
    def test_another_apps_audio_is_measured(self):
        fn = _sessions((4242, _Meter(0.31)))
        self.assertAlmostEqual(mg.pc_audio_peak((99,), sessions_fn=fn,
                                                sleep=lambda s: None), 0.31)
        self.assertEqual(fn.cleaned, [True])

    def test_own_process_and_system_sounds_are_not_counted(self):
        fn = _sessions((99, _Meter(0.9)), (0, _Meter(0.8)))
        self.assertEqual(mg.pc_audio_peak((99,), sessions_fn=fn,
                                          sleep=lambda s: None), 0.0)

    def test_no_active_session_is_silence_without_sampling(self):
        naps = []
        self.assertEqual(mg.pc_audio_peak((), sessions_fn=_sessions(),
                                          sleep=naps.append), 0.0)
        self.assertEqual(naps, [])

    def test_speech_gaps_are_sampled_through(self):
        # One device period can land in a gap between words.
        m = _Meter(0.0, 0.0, 0.42)
        naps = []
        peak = mg.pc_audio_peak((), samples=4, interval_s=0.02,
                                sessions_fn=_sessions((7, m)), sleep=naps.append)
        self.assertAlmostEqual(peak, 0.42)
        self.assertEqual(naps, [0.02, 0.02, 0.02])

    def test_stops_early_once_loud(self):
        m = _Meter(0.5)
        naps = []
        mg.pc_audio_peak((), samples=5, stop_at=0.2,
                         sessions_fn=_sessions((7, m)), sleep=naps.append)
        self.assertEqual(m.calls, 1)
        self.assertEqual(naps, [])

    def test_unreadable_is_none_and_never_raises(self):
        def boom():
            raise ImportError("no pycaw")
        self.assertIsNone(mg.pc_audio_peak((), sessions_fn=boom))
        # a meter that fails is skipped, the others still count
        fn = _sessions((7, _Meter(boom=True)), (8, _Meter(0.2)))
        self.assertAlmostEqual(mg.pc_audio_peak((), sessions_fn=fn,
                                                sleep=lambda s: None), 0.2)
        self.assertEqual(fn.cleaned, [True])


class PlayingTests(unittest.TestCase):
    def test_meter_threshold(self):
        self.assertTrue(mg.audio_playing(0.02, threshold=0.01))
        self.assertTrue(mg.audio_playing(0.01, threshold=0.01))
        self.assertFalse(mg.audio_playing(0.004, threshold=0.01))

    def test_media_session_only_when_the_meter_cannot_be_read(self):
        self.assertTrue(mg.audio_playing(None, True))
        self.assertFalse(mg.audio_playing(None, False))
        # a readable silent meter beats SMTC's "playing" (paused / muted media)
        self.assertFalse(mg.audio_playing(0.0, True))

    def test_junk_is_not_playing(self):
        self.assertFalse(mg.audio_playing("loud", threshold=0.01))


class DecideTests(unittest.TestCase):
    def test_the_live_reel_is_dropped(self):
        drop, line = mg.decide(True, lg.NOT_OWNER)
        self.assertTrue(drop)
        self.assertEqual(line, "[media-gate] PC audio playing and not the owner's voice")

    def test_the_owner_runs_over_his_own_music(self):
        for v in (lg.OWNER, lg.UNSURE):
            with self.subTest(v=v):
                drop, line = mg.decide(True, v)
                self.assertFalse(drop)
                self.assertIn("allowed", line)

    def test_unavailable_voice_id_keeps_todays_behaviour_but_logs(self):
        drop, line = mg.decide(True, lg.UNAVAILABLE)
        self.assertFalse(drop)
        self.assertIn("voice-ID unavailable", line)

    def test_no_pc_audio_no_gate_no_line(self):
        for v in lg.VERDICTS:
            with self.subTest(v=v):
                self.assertEqual(mg.decide(False, v), (False, ""))


class MediaControlTests(unittest.TestCase):
    """Review repair (2026-10-02): what he says OVER his media is mostly
    about the media. Short media-control commands always pass the gate - a
    reel saying "Jarvis, pause" costs nothing."""

    def test_media_controls_pass(self):
        for text in ("Jarvis, pause.", "Jarvis pause it", "Jarvis, resume",
                     "Jarvis, next song", "Jarvis, skip this one",
                     "Jarvis, previous track", "Hey Jarvis, turn it down",
                     "Jarvis, turn the volume down a bit", "Jarvis, mute",
                     "Jarvis, quieter please", "Jarvis, louder",
                     "Jarvis, volume up", "Jarvis, stop the music",
                     "Jarvis, unpause"):
            with self.subTest(text=text):
                self.assertTrue(mg.is_media_control(text))

    def test_other_commands_do_not(self):
        for text in ("Jarvis, find me a restaurant in Brickell, Miami, that "
                     "doesn't already have a website",
                     "Jarvis, what time is it?",
                     "Jarvis, find the next restaurant on the list and "
                     "build a website for them now",
                     "Jarvis, play Skrillex on YouTube", "", None):
            with self.subTest(text=text):
                self.assertFalse(mg.is_media_control(text))


class LeadingSpeechTests(unittest.TestCase):
    """Review repair (2026-10-02): live 22:59:47 the owner's "Jarvis, what
    time is it?" led a 20.7 s capture that was mostly the video behind him,
    and the whole capture scored 0.50. The wake word and his command are at
    the START of the capture, so that part is scored too."""

    def test_the_window_starts_at_the_speech_onset(self):
        import numpy as np
        sr = 16000
        audio = np.zeros(sr * 20, dtype=np.float32)
        audio[sr * 2:sr * 4] = 0.3                    # his words at 2-4 s
        audio[sr * 6:] = 0.05                         # the video after
        win = mg.leading_speech_window(audio, sr, seconds=3.0)
        self.assertIsNotNone(win)
        self.assertEqual(len(win), sr * 3)
        self.assertGreater(float(abs(win).max()), 0.29)
        self.assertGreater(float((abs(win) > 0.29).mean()), 0.5)

    def test_a_short_capture_has_no_separate_window(self):
        import numpy as np
        self.assertIsNone(mg.leading_speech_window(
            np.ones(16000 * 3, dtype=np.float32), 16000, seconds=3.0))

    def test_silence_or_junk_has_no_window(self):
        import numpy as np
        self.assertIsNone(mg.leading_speech_window(
            np.zeros(16000 * 20, dtype=np.float32), 16000))
        self.assertIsNone(mg.leading_speech_window(None, 16000))
        self.assertIsNone(mg.leading_speech_window([1, 2], 0))


if __name__ == "__main__":
    unittest.main()
