"""Focused unit tests for ``skills.standby_audio_detect``.

These target the 2026-07-08 fix (findings #19/#36): the always-on lyric
detector must never load faster-whisper with bare ``device='cuda'`` /
``float16``. The opt-in GPU path is gated behind a free-VRAM preflight and,
when taken, pins ``device_index`` + ``int8``. Since 2026-10-04 that index is
the LISTEN card (LISTEN_GPU — the 1650 here), never cuda:0 = the RTX 3090 the
local brain fills: the path was dead only because its pynvml probe was not
installed, and the working NVML probe revived it there (review 2026-10-04).
Stdlib ``unittest`` only.
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
        # Preflight returns None (NVML unavailable / probe failed) -> stay CPU,
        # never touch the GPU. This is the crash-avoidance guarantee.
        s._loop_cfg["prefer_gpu"] = True
        with mock.patch.object(s, "_gpu_index", return_value=1), \
             mock.patch.object(s, "_cuda_free_vram_mb", return_value=None), \
             mock.patch.dict(sys.modules,
                             {"faster_whisper": _fake_faster_whisper()}):
            s._ensure_whisper_tiny()
        self.assertEqual(_FakeWhisperModel.last_kwargs["device"], "cpu")

    def test_gpu_optin_without_a_listen_card_is_the_cpu(self):
        s._loop_cfg["prefer_gpu"] = True
        probe = mock.Mock(return_value=9999.0)
        with mock.patch.object(s, "_gpu_index", return_value=None), \
             mock.patch.object(s, "_cuda_free_vram_mb", probe), \
             mock.patch.dict(sys.modules,
                             {"faster_whisper": _fake_faster_whisper()}):
            s._ensure_whisper_tiny()
        self.assertEqual(_FakeWhisperModel.last_kwargs["device"], "cpu")
        probe.assert_not_called()

    def test_gpu_optin_with_vram_never_uses_float16(self):
        # Plenty of free VRAM -> GPU allowed, but pinned to int8 + the listen
        # card's device_index, NEVER bare cuda/float16.
        s._loop_cfg["prefer_gpu"] = True
        with mock.patch.object(s, "_gpu_index", return_value=1), \
             mock.patch.object(s, "_cuda_free_vram_mb",
                               return_value=s._GPU_MIN_FREE_VRAM_MB + 5000), \
             mock.patch.dict(sys.modules,
                             {"faster_whisper": _fake_faster_whisper()}):
            s._ensure_whisper_tiny()
        kw = _FakeWhisperModel.last_kwargs
        self.assertEqual(kw["device"], "cuda")
        self.assertEqual(kw["device_index"], 1)
        self.assertEqual(kw["compute_type"], "int8")
        self.assertNotEqual(kw["compute_type"], "float16")

    def test_gpu_optin_never_loads_on_the_brains_card(self):
        # The REAL probe chain (core.gpu_probe -> core.listen_devices) over
        # this desk's two cards, the 3090 with plenty of room: the model
        # goes to the listen card, never cuda:0.
        from core import gpu_probe
        cards = [
            {"index": 0, "name": "NVIDIA GeForce RTX 3090",
             "uuid": "GPU-aaaa", "pci_bus": 1, "total_mb": 24576,
             "free_mb": 9000, "used_mb": 15576, "util_pct": 5},
            {"index": 1, "name": "NVIDIA GeForce GTX 1650 SUPER",
             "uuid": "GPU-bbbb", "pci_bus": 8, "total_mb": 4096,
             "free_mb": 2769, "used_mb": 1327, "util_pct": 0},
        ]
        s._loop_cfg["prefer_gpu"] = True
        with mock.patch.object(gpu_probe, "cuda_gpus",
                               return_value=cards), \
             mock.patch.dict(sys.modules,
                             {"faster_whisper": _fake_faster_whisper()}):
            s._ensure_whisper_tiny()
        kw = _FakeWhisperModel.last_kwargs
        self.assertEqual((kw["device"], kw.get("device_index")), ("cuda", 1))

    def test_vram_probe_is_failsafe_when_nvml_unreadable(self):
        # 2026-10-04: NVML through core/gpu_probe (the pynvml package this
        # used is not installed, so the opt-in always read None). Unreadable
        # = None = stay on the CPU.
        from core import gpu_probe
        with mock.patch.object(gpu_probe, "cuda_memory_mb",
                               return_value=None):
            self.assertIsNone(s._cuda_free_vram_mb(1))

    def test_vram_probe_reads_the_card_it_is_asked_from_nvml(self):
        from core import gpu_probe
        with mock.patch.object(gpu_probe, "cuda_memory_mb",
                               return_value=(1700, 4096)) as probe:
            self.assertEqual(s._cuda_free_vram_mb(1), 1700.0)
        probe.assert_called_once_with(1)

    def test_the_card_is_the_listen_card(self):
        from core import config, gpu_probe
        cards = [{"index": 0, "name": "RTX 3090", "uuid": "GPU-aaaa",
                  "free_mb": 9000, "total_mb": 24576},
                 {"index": 1, "name": "GTX 1650 SUPER", "uuid": "GPU-bbbb",
                  "free_mb": 2769, "total_mb": 4096}]
        with mock.patch.object(gpu_probe, "cuda_gpus", return_value=cards):
            for lg, want in (("cuda:1", 1), ("1650", 1), ("GPU-bbbb", 1),
                             ("", None), ("4090", None)):
                with mock.patch.object(config, "LISTEN_GPU", lg):
                    self.assertEqual(s._gpu_index(), want, lg)


if __name__ == "__main__":
    unittest.main()
