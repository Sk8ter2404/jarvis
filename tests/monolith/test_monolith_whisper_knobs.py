"""Speed plan R11 - Whisper decode knobs (WHISPER_BEAM_SIZE, WHISPER_TEMPERATURES).

no_speech_prob is always 0.0 on this rig, so faster-whisper's silence exit
never fires and every low-logprob decode walks all six temperatures - the
4-10 s transcript tails. The knobs let the owner cap that. Defaults keep the
decode byte-identical to before (beam 5, faster-whisper's own temperature
ladder: no temperature kwarg at all). The bounded no-VAD retry keeps its own
fixed beam 1 / temperature 0.0.
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from tests._monolith_harness import MonolithGlobalsTestCase, load_monolith, requires_monolith


class _Seg:
    def __init__(self, text, nsp=0.1, lp=-0.2):
        self.text, self.no_speech_prob, self.avg_logprob = text, nsp, lp


@requires_monolith
class WhisperKnobTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = (self.bc._stt, self.bc._stt_engine)

    def tearDown(self):
        self.bc._stt, self.bc._stt_engine = self._saved

    def _calls(self, *, beam=None, temps=None, results=None):
        bc = self.bc
        fake = mock.Mock()
        info = mock.Mock(no_speech_prob=0.1)
        seq = results or [[_Seg("turn on the lights")]]
        fake.transcribe.side_effect = [(iter(s), info) for s in seq]
        patches = [mock.patch.object(bc, "_ensure_whisper"),
                   mock.patch.object(bc, "_stt", fake),
                   mock.patch.object(bc, "_stt_engine", "faster_whisper"),
                   mock.patch.object(bc, "STT_HOTWORDS", "", create=True)]
        if beam is not None:
            patches.append(mock.patch.object(bc, "WHISPER_BEAM_SIZE", beam, create=True))
        if temps is not None:
            patches.append(mock.patch.object(bc, "WHISPER_TEMPERATURES", temps, create=True))
        for p in patches:
            p.start()
        try:
            bc.transcribe(np.zeros(1600, dtype=np.float32))
        finally:
            for p in reversed(patches):
                p.stop()
        return [c.kwargs for c in fake.transcribe.call_args_list]

    def test_defaults_are_byte_identical(self):
        kw = self._calls()[0]
        self.assertEqual(kw["beam_size"], 5)
        self.assertNotIn("temperature", kw)
        self.assertTrue(kw["vad_filter"])

    def test_knobs_reach_the_vad_decode(self):
        kw = self._calls(beam=1, temps=(0.0, 0.2, 0.4))[0]
        self.assertEqual(kw["beam_size"], 1)
        self.assertEqual(kw["temperature"], (0.0, 0.2, 0.4))

    def test_a_single_temperature_is_accepted(self):
        kw = self._calls(temps=0.0)[0]
        self.assertEqual(kw["temperature"], 0.0)

    def test_the_no_vad_retry_keeps_its_own_bounds(self):
        calls = self._calls(beam=3, temps=(0.0, 0.2),
                            results=[[], [_Seg("hello")]])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["beam_size"], 3)
        self.assertFalse(calls[1]["vad_filter"])
        self.assertEqual(calls[1]["beam_size"], 1)
        self.assertEqual(calls[1]["temperature"], 0.0)

    def test_bad_values_fall_back_to_the_defaults(self):
        for beam, temps in ((0, ()), (-2, "hot"), ("five", [1.5]),
                            (99, [-0.1, 0.2]), (True, {"t": 0})):
            kw = self._calls(beam=beam, temps=temps)[0]
            self.assertEqual(kw["beam_size"], 5, (beam, temps))
            self.assertNotIn("temperature", kw, (beam, temps))


if __name__ == "__main__":
    unittest.main()
