"""core/voice_id.py places Resemblyzer on an EXPLICIT device (2026-10-04).

THE LIVE FINDING: JARVIS (pid 95748) held a CUDA context on the RTX 3090 —
the brain's card, ~1.1 GB to spare — although nothing in JARVIS was meant to
run there. Thread birth times put its creation at 13:50:56.037, inside the
ambient listener's first voice-ID; py-spy caught identify_speaker ->
embed_utterance -> VoiceEncoder.forward on the ambient-listen and
room-talk-voice threads. _load_encoder called ``VoiceEncoder(verbose=False)``
with no device, and Resemblyzer's own default is ``torch.device("cuda" if
torch.cuda.is_available() else "cpu")`` = cuda:0. The torch cache then grew
with the longest clip it saw (135 MB process total for a 2.5 s clip, 441 MB
after a 30 s one, never released).

Pinned here (a fake resemblyzer that reproduces that default, a fake torch,
a fake card probe — no GPU):
  * the default (VOICE_ID_DEVICE 'cpu') builds the encoder on the CPU and
    never touches torch.cuda;
  * 'listen' puts it on the listen card when it is there with room, else the
    CPU with ONE log line;
  * on a card the torch cache is handed back after every embedding;
  * identify_speaker counts its embeddings (the music gate's minute line).

    python -m unittest tests.test_voice_id_device
"""
from __future__ import annotations

import contextlib
import io
import types
import unittest
from unittest import mock

import numpy as np

import core.config as cfg
import core.voice_id as vid
from tests.test_voice_id import _VoiceIdBase, emb_vec, inject_modules


class _FakeCuda:
    def __init__(self, available=True):
        self.available = available
        self.touched = []           # every device the encoder was put on
        self.emptied = []

    def is_available(self):
        return self.available

    def empty_cache(self):
        self.emptied.append(self.current)

    @contextlib.contextmanager
    def device(self, dev):
        self.current = str(dev)
        yield


def _fake_torch(cuda):
    t = types.ModuleType("torch")
    t.cuda = cuda
    t.device = lambda d: d
    return t


def _fake_resemblyzer(torch_mod):
    """Resemblyzer's VoiceEncoder with its REAL device default: None ->
    "cuda" when torch has CUDA (= cuda:0, the 3090), else "cpu"."""
    mod = types.ModuleType("resemblyzer")

    class VoiceEncoder:
        def __init__(self, device=None, verbose=True, weights_fpath=None):
            if device is None:
                device = "cuda" if torch_mod.cuda.is_available() else "cpu"
            self.device = str(device)
            torch_mod.cuda.touched.append(self.device)

        def embed_utterance(self, wav):
            return emb_vec(7)
    mod.VoiceEncoder = VoiceEncoder
    return mod


class _DeviceBase(_VoiceIdBase):
    def setUp(self):
        super().setUp()
        had = hasattr(vid, "_encoder_device")       # (absent on origin/main)
        saved = getattr(vid, "_encoder_device", "")

        def _restore():
            if had:
                vid._encoder_device = saved
            elif hasattr(vid, "_encoder_device"):
                del vid._encoder_device
        self.addCleanup(_restore)
        vid._encoder_device = ""
        self.cuda = _FakeCuda()
        self.torch = _fake_torch(self.cuda)
        self.rz = _fake_resemblyzer(self.torch)

    def _load(self, device_setting, probe_hit=None):
        """_load_encoder with VOICE_ID_DEVICE = device_setting; the card probe
        answers ``probe_hit`` ((cuda_index, gpu) or None)."""
        from core import gpu_probe
        out = io.StringIO()
        with inject_modules(resemblyzer=self.rz, torch=self.torch), \
                mock.patch.object(cfg, "VOICE_ID_DEVICE", device_setting), \
                mock.patch.object(cfg, "LISTEN_GPU", "cuda:1"), \
                mock.patch.object(cfg, "LISTEN_GPU_RESERVE_MB", 512), \
                mock.patch.object(gpu_probe, "find",
                                  return_value=probe_hit) as find, \
                contextlib.redirect_stdout(out):
            enc = vid._load_encoder()
        return enc, out.getvalue(), find


class PlacementTests(_DeviceBase):
    GPU_1650 = {"name": "NVIDIA GeForce GTX 1650 SUPER", "free_mb": 2769,
                "total_mb": 4096}

    def test_resemblyzers_cuda_default_is_never_used(self):
        # The live bug in one line: torch has CUDA, the setting is the
        # shipped one, and the encoder must still be on the CPU. (No new
        # module is touched here, so this runs as-is against origin/main,
        # where it fails: the encoder lands on "cuda" = cuda:0.)
        with inject_modules(resemblyzer=self.rz, torch=self.torch), \
                mock.patch.object(cfg, "VOICE_ID_DEVICE", "cpu", create=True):
            enc = vid._load_encoder()
        self.assertEqual(enc.device, "cpu")
        self.assertEqual(self.cuda.touched, ["cpu"])

    def test_default_is_the_cpu_never_cuda0(self):
        enc, out, find = self._load("cpu")
        self.assertEqual(enc.device, "cpu")
        self.assertEqual(self.cuda.touched, ["cpu"])   # never "cuda"
        self.assertEqual(vid._encoder_device, "cpu")
        find.assert_not_called()                       # no card probe at all
        self.assertEqual(out, "")

    def test_listen_card_with_room(self):
        enc, out, _find = self._load("listen", probe_hit=(1, self.GPU_1650))
        self.assertEqual(enc.device, "cuda:1")
        self.assertEqual(self.cuda.touched, ["cuda:1"])  # never bare "cuda"
        self.assertEqual(out, "")

    def test_listen_card_missing_is_the_cpu_with_one_line(self):
        enc, out, _find = self._load("listen", probe_hit=None)
        self.assertEqual(enc.device, "cpu")
        self.assertEqual(out.count("[listen] voice-id: asked listen"), 1)
        self.assertIn("running on the CPU", out)

    def test_listen_card_full_is_the_cpu(self):
        full = dict(self.GPU_1650, free_mb=600)          # < 400 + 512
        enc, out, _find = self._load("listen", probe_hit=(1, full))
        self.assertEqual(enc.device, "cpu")
        self.assertIn("600 MB free < 400 + 512 MB reserve", out)

    def test_torch_without_cuda_is_the_cpu(self):
        self.cuda.available = False
        enc, out, _find = self._load("cuda:1", probe_hit=(1, self.GPU_1650))
        self.assertEqual(enc.device, "cpu")
        self.assertIn("torch has no CUDA", out)


class GpuCacheTests(_DeviceBase):
    def _identify_once(self, device_setting, probe_hit=None):
        self._seed("owner", vec=emb_vec(7))
        enc, _out, _find = self._load(device_setting, probe_hit)
        with inject_modules(resemblyzer=self.rz, torch=self.torch):
            name, score = vid.identify_speaker(
                np.ones(16000, dtype=np.float32) * 0.1, 16000)
        return name, score

    def test_cache_handed_back_after_each_embedding_on_a_card(self):
        hit = (1, {"name": "1650", "free_mb": 2769, "total_mb": 4096})
        self._identify_once("listen", hit)
        self.assertEqual(self.cuda.emptied, ["cuda:1"])

    def test_cpu_never_touches_torch_cuda(self):
        self._identify_once("cpu")
        self.assertEqual(self.cuda.emptied, [])

    def test_identify_counts_its_embeddings(self):
        before = vid.identify_calls
        self._identify_once("cpu")
        self.assertEqual(vid.identify_calls, before + 1)


class StatusTests(_DeviceBase):
    def test_status_names_the_device(self):
        self._load("cpu")
        with inject_modules(resemblyzer=self.rz, torch=self.torch):
            self.assertEqual(vid.encoder_status()["encoder_device"], "cpu")


if __name__ == "__main__":
    unittest.main()
