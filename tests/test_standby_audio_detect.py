"""Focused unit tests for ``skills.standby_audio_detect``.

These target the 2026-07-08 fix (findings #19/#36): the always-on lyric
detector must never load faster-whisper with bare ``device='cuda'`` /
``float16``. The opt-in GPU path is gated behind a free-VRAM preflight and,
when taken, pins ``device_index`` + ``int8``. Stdlib ``unittest`` only.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from skills import standby_audio_detect as s


class _FakeWhisperModel:
    """Records the kwargs faster-whisper's WhisperModel was built with."""
    last_kwargs = None

    def __init__(self, model_name, **kwargs):
        _FakeWhisperModel.last_kwargs = dict(kwargs, model_name=model_name)


def _fake_faster_whisper():
    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = _FakeWhisperModel
    return mod


class WhisperDeviceSafetyTests(unittest.TestCase):
    def setUp(self):
        s._whisper_model[0] = None
        _FakeWhisperModel.last_kwargs = None
        self._saved_cfg = dict(s._loop_cfg)

    def tearDown(self):
        s._whisper_model[0] = None
        s._loop_cfg.clear()
        s._loop_cfg.update(self._saved_cfg)

    def test_default_path_uses_cpu_int8(self):
        s._loop_cfg["prefer_gpu"] = False
        with mock.patch.dict(sys.modules,
                             {"faster_whisper": _fake_faster_whisper()}):
            model = s._ensure_whisper_tiny()
        self.assertIsNotNone(model)
        self.assertEqual(_FakeWhisperModel.last_kwargs["device"], "cpu")
        self.assertEqual(_FakeWhisperModel.last_kwargs["compute_type"], "int8")

    def test_gpu_optin_without_vram_falls_back_to_cpu(self):
        # Preflight returns None (pynvml unavailable / probe failed) -> stay CPU,
        # never touch the GPU. This is the crash-avoidance guarantee.
        s._loop_cfg["prefer_gpu"] = True
        with mock.patch.object(s, "_cuda_free_vram_mb", return_value=None), \
             mock.patch.dict(sys.modules,
                             {"faster_whisper": _fake_faster_whisper()}):
            s._ensure_whisper_tiny()
        self.assertEqual(_FakeWhisperModel.last_kwargs["device"], "cpu")

    def test_gpu_optin_with_vram_never_uses_float16(self):
        # Plenty of free VRAM -> GPU allowed, but pinned to int8 + device_index,
        # NEVER bare cuda/float16.
        s._loop_cfg["prefer_gpu"] = True
        with mock.patch.object(s, "_cuda_free_vram_mb",
                               return_value=s._GPU_MIN_FREE_VRAM_MB + 5000), \
             mock.patch.dict(sys.modules,
                             {"faster_whisper": _fake_faster_whisper()}):
            s._ensure_whisper_tiny()
        kw = _FakeWhisperModel.last_kwargs
        self.assertEqual(kw["device"], "cuda")
        self.assertEqual(kw["device_index"], 0)
        self.assertEqual(kw["compute_type"], "int8")
        self.assertNotEqual(kw["compute_type"], "float16")

    def test_vram_probe_is_failsafe_when_pynvml_missing(self):
        with mock.patch.dict(sys.modules, {"pynvml": None}):
            self.assertIsNone(s._cuda_free_vram_mb())


if __name__ == "__main__":
    unittest.main()
