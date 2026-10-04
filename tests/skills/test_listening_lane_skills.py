"""The listening lane's skill halves (2026-10-04).

skills/ambient_listen.py — the MUSIC GATE on the ambient mic worker. Live
2026-10-04 the worker batched every 2.5 s of the main loop's frames through
Whisper large-v3-turbo on the 1650 while music played (its 0.003 RMS gate
always passes music): ~15 decodes a minute of lyrics, each ~0.9 GPU-s +
~0.9 CPU-s, and ambient_transcripts.jsonl filled with lyrics. The host
(bobert_companion._music_gate_ambient) now answers per batch: None (as
always), 'skip' (over music, 'on': no decode at all), 'shadow' (as always,
then told what became of the batch).

skills/standby_audio_detect.py — HEADSET FIRST. The lyric loop's only action
(auto-standby) needs the headset as the output; on the speakers it still read
the mic and ran whisper-tiny + librosa every 5 s (9.5 CPU-s a minute measured
over music).

Fakes only: no stream, no Whisper, no voiceprint.

    python -m unittest tests.skills.test_listening_lane_skills
"""
from __future__ import annotations

import re
import sys
import types
import unittest
from unittest import mock

import numpy as np

from tests._skill_harness import load_skill_isolated
from tests.skills import test_ambient_listen as _amb


class AmbientMusicGateTests(_amb._TmpDirMixin, unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("ambient_listen")
        self._redirect_paths()
        self.mod._buffer.clear()
        self.mod._last_error = None
        self.mod._wake_pattern = re.compile(r"\bjarvis\b", re.I)

    def _bc(self, gate, text="la la la love you"):
        bc = _amb._FakeBobert()
        bc.transcribe = mock.MagicMock(return_value=(text, _amb._good_conf()))
        bc.is_valid_speech = mock.MagicMock(return_value=(True, "ok"))
        bc.is_ambient_music = mock.MagicMock(return_value=False)
        bc._music_gate_ambient = mock.MagicMock(return_value=gate)
        bc._music_gate_ambient_done = mock.MagicMock()
        return bc

    def _run(self, bc, ident=(None, 0.0)):
        block = np.ones(16000 * 3, dtype=np.float32) * 0.2
        with mock.patch.object(self.mod, "_identify_speaker_safe",
                               return_value=ident) as idf:
            _amb.MicWorkerLoopTests._run_worker(self, bc, feed_block=block,
                                                wait_returns=[True])
        return idf

    def test_skip_never_transcribes_or_keeps(self):
        bc = self._bc({"verdict": "skip"})
        idf = self._run(bc)
        bc._music_gate_ambient.assert_called_once()
        bc.transcribe.assert_not_called()
        idf.assert_not_called()
        self.assertEqual(len(self.mod._buffer), 0)
        bc._music_gate_ambient_done.assert_not_called()

    def test_shadow_transcribes_as_always_and_reports(self):
        gate = {"verdict": "shadow"}
        bc = self._bc(gate, text="Jarvis is a great name")
        self._run(bc, ident=(None, 0.31))
        bc.transcribe.assert_called_once()
        self.assertEqual(len(self.mod._buffer), 1)
        bc._music_gate_ambient_done.assert_called_once_with(
            gate, kept=True, wake=True, speaker=(None, 0.31))

    def test_shadow_reports_a_dropped_batch_too(self):
        gate = {"verdict": "shadow"}
        bc = self._bc(gate)
        bc.is_valid_speech = mock.MagicMock(return_value=(False, "halluc"))
        self._run(bc)
        bc._music_gate_ambient_done.assert_called_once_with(
            gate, kept=False, wake=False, speaker=None)

    def test_no_gate_on_the_host_is_todays_behaviour(self):
        bc = self._bc(None)
        del bc._music_gate_ambient
        del bc._music_gate_ambient_done
        idf = self._run(bc, ident=(None, 0.0))
        bc.transcribe.assert_called_once()
        idf.assert_called_once()
        self.assertEqual(len(self.mod._buffer), 1)

    def test_a_raising_gate_is_todays_behaviour(self):
        bc = self._bc(None)
        bc._music_gate_ambient = mock.MagicMock(side_effect=RuntimeError("x"))
        self._run(bc)
        bc.transcribe.assert_called_once()
        self.assertEqual(len(self.mod._buffer), 1)


class LyricLoopHeadsetFirstTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("standby_audio_detect")
        self.mod._loop_consecutive[0] = 2
        self.addCleanup(lambda: sys.modules.pop("bobert_companion", None))

    def _one_pass(self, headset):
        bc = types.ModuleType("bobert_companion")
        bc.SAMPLE_RATE = 16000
        bc._standby_mode = [False]
        bc._sleep_mode = [False]
        bc._jarvis_played_music_at = [0.0]
        bc.is_using_headset = headset
        t = np.arange(16000 * 3, dtype=np.float32) / 16000.0
        bc.get_mic_buffer = mock.MagicMock(
            return_value=(np.sin(2 * np.pi * 220 * t) * 0.2).astype(np.float32))
        state = {"i": 0}

        def _wait(_interval):
            state["i"] += 1
            return state["i"] > 1
        with _amb.inject_modules(bobert_companion=bc), \
                mock.patch.object(self.mod._loop_stop, "wait",
                                  side_effect=_wait), \
                mock.patch.object(self.mod, "_transcribe_buffer",
                                  return_value="la la") as tr, \
                mock.patch.object(self.mod, "_onset_energy",
                                  return_value=0.9), \
                mock.patch.object(self.mod, "_looks_like_lyrics",
                                  return_value=True):
            self.mod._background_loop()
        return bc, tr

    def test_on_the_speakers_nothing_is_read_or_transcribed(self):
        bc, tr = self._one_pass(lambda: False)
        bc.get_mic_buffer.assert_not_called()
        tr.assert_not_called()
        self.assertEqual(self.mod._loop_consecutive[0], 0)

    def test_an_unreadable_output_counts_as_not_the_headset(self):
        def boom():
            raise RuntimeError("no audio api")
        bc, tr = self._one_pass(boom)
        bc.get_mic_buffer.assert_not_called()
        tr.assert_not_called()

    def test_on_the_headset_it_listens_as_before(self):
        self.mod._loop_consecutive[0] = 0
        bc, tr = self._one_pass(lambda: True)
        bc.get_mic_buffer.assert_called_once()
        tr.assert_called_once()
        self.assertEqual(self.mod._loop_consecutive[0], 1)


if __name__ == "__main__":
    unittest.main()
