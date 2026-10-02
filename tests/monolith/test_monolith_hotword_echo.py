"""transcribe(): a hotword echo gets one plain re-decode before the turn is dropped.

THE LIVE FAILURE (2026-10-01 22:06:03-12, and ~20 more drops from 21:50 to 22:02):
about 1.8 s of the owner's real speech, boosted x10 by auto-gain, came back from the
first decode (hotwords=STT_HOTWORDS) as the hint list read back. v2.0.158's guard
returned '' for the whole utterance and never decoded it again, so the turn was
silently thrown away; the ambient copy of the same audio, decoded WITHOUT the hint,
was real speech in his voice (0.87).

Pinned here:
  * an echo is re-decoded once, same VAD settings, hotwords=None, and that text is
    kept when it is real speech;
  * the turn is dropped only when the plain re-decode is empty or an echo too;
  * an echo from the no-VAD retry (already a plain decode) is dropped without a
    third decode;
  * the hotwords Whisper gets are read through core.stt_vocab.live_hotwords, so an
    edit to data/user_settings.json applies without a restart.

Generic stand-in names only.

    python -m unittest tests.monolith.test_monolith_hotword_echo
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._monolith_harness import (MonolithGlobalsTestCase, load_monolith,
                                     requires_monolith)

try:
    import numpy as np
except Exception:  # pragma: no cover - light CI
    np = None


class _Seg:
    def __init__(self, text, no_speech_prob=0.1, avg_logprob=-0.2):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


HOT = "Zorblat, Flemwick, Quonset, Brindle"
ECHO = "Zorblat, Flemwick, Quonset, Brindle,"


@requires_monolith
class HotwordEchoRedecodeTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _run(self, side_effect, hotwords=HOT, live=None):
        bc = self.bc
        fake = mock.Mock()
        fake.transcribe.side_effect = side_effect
        live_fn = (lambda fb: live) if live is not None else (lambda fb: fb)
        with mock.patch.object(bc, "_ensure_whisper"), \
                mock.patch.object(bc, "_stt", fake), \
                mock.patch.object(bc, "_stt_engine", "faster_whisper"), \
                mock.patch.object(bc, "STT_HOTWORDS", hotwords, create=True), \
                mock.patch.object(bc, "STT_REPLACEMENTS", {}, create=True), \
                mock.patch.object(bc._stt_vocab, "live_hotwords", side_effect=live_fn,
                                  create=True):
            text, conf = bc.transcribe(np.zeros(8, dtype=np.float32))
        return text, conf, fake.transcribe.call_args_list

    def test_echo_is_redecoded_without_hotwords_and_real_speech_kept(self):
        info = mock.Mock(no_speech_prob=0.1)
        said = "Jarvis, play the Brindle mix on YouTube"
        text, conf, calls = self._run([
            (iter([_Seg(ECHO)]), info),
            (iter([_Seg(said, 0.05, -0.3)]), info),
        ])
        self.assertEqual(text, said)
        self.assertEqual(len(calls), 2)
        first, plain = calls
        self.assertEqual(first.kwargs["hotwords"], HOT)
        self.assertIsNone(plain.kwargs["hotwords"])
        # same VAD settings as the first pass: this audio DID contain speech
        self.assertTrue(plain.kwargs["vad_filter"])
        self.assertEqual(plain.kwargs.get("vad_parameters"),
                         first.kwargs.get("vad_parameters"))
        # the confidence is the re-decode's, not the echo's
        self.assertAlmostEqual(conf["no_speech_prob"], 0.05, places=6)
        self.assertAlmostEqual(conf["avg_logprob"], -0.3, places=6)

    def test_dropped_when_the_plain_redecode_is_empty(self):
        info = mock.Mock(no_speech_prob=0.1)
        text, conf, calls = self._run([
            (iter([_Seg(ECHO)]), info),
            (iter([]), info),
        ])
        self.assertEqual(text, "")
        self.assertEqual(conf["no_speech_prob"], 1.0)
        self.assertEqual(len(calls), 2)

    def test_dropped_when_the_plain_redecode_echoes_too(self):
        info = mock.Mock(no_speech_prob=0.1)
        text, conf, calls = self._run([
            (iter([_Seg(ECHO)]), info),
            (iter([_Seg("Zorblat, Zorblat, Flemwick.")]), info),
        ])
        self.assertEqual(text, "")
        self.assertEqual(conf["no_speech_prob"], 1.0)

    def test_an_echo_from_the_no_vad_retry_is_not_decoded_a_third_time(self):
        info = mock.Mock(no_speech_prob=0.1)
        text, conf, calls = self._run([
            (iter([]), info),                       # VAD pass: nothing
            (iter([_Seg(ECHO)]), info),             # no-VAD retry (no hotwords)
        ])
        self.assertEqual(text, "")
        self.assertEqual(len(calls), 2)

    def test_redecode_failure_never_raises_and_drops(self):
        info = mock.Mock(no_speech_prob=0.1)
        text, conf, _calls = self._run([
            (iter([_Seg(ECHO)]), info),
            RuntimeError("decode blew up"),
        ])
        self.assertEqual(text, "")
        self.assertEqual(conf["no_speech_prob"], 1.0)

    def test_live_settings_value_is_what_whisper_gets(self):
        # The owner emptied STT_HOTWORDS in user_settings.json after start: the
        # import-time "Zorblat, ..." must no longer reach Whisper.
        info = mock.Mock(no_speech_prob=0.1)
        text, _conf, calls = self._run([(iter([_Seg("hello there")]), info)],
                                       hotwords=HOT, live="")
        self.assertEqual(text, "hello there")
        self.assertIsNone(calls[0].kwargs["hotwords"])


if __name__ == "__main__":
    unittest.main()
