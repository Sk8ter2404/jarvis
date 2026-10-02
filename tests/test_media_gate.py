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


if __name__ == "__main__":
    unittest.main()
