"""Two known third-party log lines are silenced at the logger (2026-09-30).

Live session log, every session:
    WARNING  [phonemizer] words count mismatch on 100.0% of the lines (1/1)
        - on nearly every Kokoro TTS line (dozens per session);
    WARNING  [huggingface_hub.utils._http] Warning: You are sending
        unauthenticated requests to the HF Hub. Please set a HF_TOKEN ...
        - at every long-term-memory embedder load.

The filters live in core/log_filters.py and are installed where the TTS engine
(core/kokoro_tts._engine) and the embedder (core/long_term_memory.
_try_import_embedder) are set up. ONLY those exact messages go: other records
of the same loggers, and anything above WARNING, still get through.

These tests drive the REAL setup paths (no model is loaded: the Kokoro model
check fails and the embedder class is a stub), then log the exact library
messages and check what reaches a handler. Stdlib only - light tier.
"""
from __future__ import annotations

import contextlib
import io
import logging
import sys
import types
import unittest
from unittest import mock

_PHON = "phonemizer"
_HF = "huggingface_hub.utils._http"
_PHON_MSG = ("words count mismatch on %s%% of the lines (%s/%s)", 100.0, 1, 1)
_HF_MSG = ("Warning: You are sending unauthenticated requests to the HF Hub. "
           "Please set a HF_TOKEN to enable higher rate limits and faster "
           "downloads.")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records: list = []

    def emit(self, record):
        self.records.append(record.getMessage())


class _LoggerCase(unittest.TestCase):
    def _watch(self, name):
        """A capturing handler on logger ``name``; the logger's filters,
        level, propagation and handlers are restored afterwards."""
        lg = logging.getLogger(name)
        saved = (list(lg.filters), lg.level, lg.propagate, list(lg.handlers))

        def _restore():
            lg.filters[:] = saved[0]
            lg.setLevel(saved[1])
            lg.propagate = saved[2]
            lg.handlers[:] = saved[3]
        self.addCleanup(_restore)
        lg.filters[:] = [f for f in lg.filters
                         if not getattr(f, "_jarvis_log_filter_tag", None)]
        cap = _Capture()
        lg.handlers[:] = [cap]
        lg.propagate = False
        lg.setLevel(logging.DEBUG)
        return lg, cap

    def _kokoro_setup(self):
        from core import kokoro_tts
        with mock.patch.object(kokoro_tts, "_ENGINE", [None]), \
             mock.patch.object(kokoro_tts, "_FAILED", [False]), \
             mock.patch.object(kokoro_tts, "_models_present",
                               return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(kokoro_tts._engine())

    def _embedder_setup(self):
        from core import long_term_memory as ltm
        fake = types.ModuleType("sentence_transformers")
        fake.SentenceTransformer = lambda *a, **k: object()
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake}), \
             mock.patch.object(ltm, "_embedder", None), \
             mock.patch.object(ltm, "_embedder_failed_until", 0.0), \
             mock.patch("core.config.LTM_EMBED_DEVICE", "cpu", create=True), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNotNone(ltm._try_import_embedder())


class PhonemizerMismatchTests(_LoggerCase):
    def test_tts_engine_setup_silences_the_mismatch_line(self):
        lg, cap = self._watch(_PHON)
        self._kokoro_setup()
        lg.warning(*_PHON_MSG)
        lg.warning("words count mismatch on %s%% of the lines (%s/%s)",
                   round(4 / 7, 2) * 100, 4, 7)
        self.assertEqual(cap.records, [])

    def test_other_phonemizer_lines_and_errors_still_get_through(self):
        lg, cap = self._watch(_PHON)
        self._kokoro_setup()
        lg.warning("espeak-mbrola backend cannot preserve punctuation")
        lg.error(*_PHON_MSG)
        lg.warning("words count mismatch on line 3 (espeak dropped a word)")
        self.assertEqual(cap.records, [
            "espeak-mbrola backend cannot preserve punctuation",
            "words count mismatch on 100.0% of the lines (1/1)",
            "words count mismatch on line 3 (espeak dropped a word)"])

    def test_survives_phonemizers_own_logger_reset(self):
        # phonemizer.logger.get_logger() resets level + handlers of its
        # logger on every import that calls it - never its filters.
        lg, cap = self._watch(_PHON)
        self._kokoro_setup()
        lg.setLevel(logging.WARNING)
        lg.handlers[:] = [cap]
        lg.warning(*_PHON_MSG)
        self.assertEqual(cap.records, [])


class HfUnauthenticatedTests(_LoggerCase):
    def test_embedder_setup_silences_the_unauthenticated_line(self):
        lg, cap = self._watch(_HF)
        self._embedder_setup()
        lg.warning(_HF_MSG)
        lg.warning(_HF_MSG[len("Warning: "):])
        self.assertEqual(cap.records, [])

    def test_real_hf_http_problems_still_get_through(self):
        lg, cap = self._watch(_HF)
        self._embedder_setup()
        lines = ["HTTP Error 500 thrown while requesting GET https://example",
                 "Retrying in 1s [Retry 1/5]."]
        for ln in lines:
            lg.warning(ln)
        lg.error(_HF_MSG)
        self.assertEqual(cap.records, lines + [_HF_MSG])


class InstallContractTests(_LoggerCase):
    def test_install_is_idempotent(self):
        from core import log_filters
        lg, _cap = self._watch(_PHON)
        a = log_filters.install_phonemizer_filter()
        b = log_filters.install_phonemizer_filter()
        self.assertIs(a, b)
        ours = [f for f in lg.filters
                if getattr(f, "_jarvis_log_filter_tag", None)]
        self.assertEqual(len(ours), 1)

    def test_dropped_lines_are_counted(self):
        from core import log_filters
        lg, _cap = self._watch(_HF)
        f = log_filters.install_hf_unauthenticated_filter()
        lg.warning(_HF_MSG)
        lg.warning(_HF_MSG)
        self.assertEqual(f.dropped, 2)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
