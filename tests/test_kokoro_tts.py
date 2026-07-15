"""Tests for core/kokoro_tts — the CPU Kokoro TTS backend (2026-07-15, P2).

Pins the fail-closed contract that keeps JARVIS from ever going mute: synthesize()
NEVER raises and returns None on empty text / unavailable engine / render failure /
memoized prior failure, so the caller's edge → pyttsx3 → SAPI5 → silence ladder
takes over. Also pins CI-safety: importing the module must NOT drag in kokoro_onnx
or onnxruntime (the real import lives behind the lazy _engine() seam), so the bare
CI runner (tools/run_tests_ci_sim.py) never loads a heavy native dep.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from unittest import mock

from core import kokoro_tts as k


class KokoroTtsTests(unittest.TestCase):
    def setUp(self):
        # reset the module singleton/fail latch between tests
        k._ENGINE[0] = None
        k._FAILED[0] = False

    def tearDown(self):
        k._ENGINE[0] = None
        k._FAILED[0] = False

    def test_import_does_not_pull_in_kokoro_onnx(self):
        # the heavy native dep must load lazily, never at import — keeps CI light
        importlib.reload(k)
        self.assertNotIn("kokoro_onnx", sys.modules,
                         "kokoro_onnx must not import at module load (CI safety)")

    def test_empty_text_returns_none(self):
        self.assertIsNone(k.synthesize(""))
        self.assertIsNone(k.synthesize("   "))
        self.assertIsNone(k.synthesize(None))

    def test_unavailable_when_spec_missing(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(k.is_available())

    def test_unavailable_when_models_missing(self):
        with mock.patch.object(k, "_models_present", return_value=False):
            self.assertFalse(k.is_available())

    def test_unavailable_after_memoized_failure(self):
        k._FAILED[0] = True
        self.assertFalse(k.is_available())
        # and _engine short-circuits without retrying
        self.assertIsNone(k._engine())

    def test_synthesize_none_when_unavailable(self):
        with mock.patch.object(k, "is_available", return_value=False):
            self.assertIsNone(k.synthesize("hello sir"))

    def test_synthesize_is_fail_closed_on_engine_none(self):
        # available, but the engine fails to build → render yields nothing → None,
        # never an exception (the whole point of the contract)
        with mock.patch.object(k, "is_available", return_value=True), \
             mock.patch.object(k, "_engine", return_value=None):
            self.assertIsNone(k.synthesize("all systems online"))

    def test_synthesize_returns_audio_on_success(self):
        import numpy as np
        fake = mock.Mock()
        fake.create.return_value = (np.zeros(2400, dtype=np.float32), 24000)
        with mock.patch.object(k, "is_available", return_value=True), \
             mock.patch.object(k, "_engine", return_value=fake):
            res = k.synthesize("hello")
        self.assertIsNotNone(res)
        audio, sr = res
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.dtype, np.dtype("float32"))
        self.assertEqual(audio.ndim, 1)

    def test_render_exception_is_swallowed(self):
        boom = mock.Mock()
        boom.create.side_effect = RuntimeError("onnx boom")
        with mock.patch.object(k, "is_available", return_value=True), \
             mock.patch.object(k, "_engine", return_value=boom):
            self.assertIsNone(k.synthesize("this should not raise"))


if __name__ == "__main__":
    unittest.main()
