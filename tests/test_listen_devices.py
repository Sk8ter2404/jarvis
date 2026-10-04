"""core/listen_devices.py — where each listening model runs (2026-10-04).

One setting per model (PARAKEET_DEVICE, SMART_TURN_DEVICE, VOICE_ID_DEVICE;
Whisper's WHISPER_DEVICE takes 'listen' too) and one resolver: 'cpu' as
itself, 'listen' through LISTEN_GPU, 'cuda:N' directly; a card that is
missing, lacks the model's GPU runtime or the room for it (need + reserve)
is the CPU with ONE log line saying why.

Fakes only (a fake probe and a fake onnxruntime): no driver, no GPU.

    python -m unittest tests.test_listen_devices
"""
from __future__ import annotations

import types
import unittest

from core import listen_devices as ld

GPUS = [
    {"index": 0, "name": "NVIDIA GeForce RTX 3090", "uuid": "GPU-aaaa",
     "pci_bus": 1, "total_mb": 24576, "free_mb": 1100},
    {"index": 1, "name": "NVIDIA GeForce GTX 1650 SUPER", "uuid": "GPU-bbbb",
     "pci_bus": 8, "total_mb": 4096, "free_mb": 2769},
]


class FakeProbe:
    """core.gpu_probe.find(spec, env) over GPUS (index = CUDA index)."""

    def __init__(self, gpus=GPUS):
        self.gpus = gpus
        self.calls = []

    def find(self, spec, env=None):
        self.calls.append(spec)
        s = str(spec).lower()
        for i, g in enumerate(self.gpus):
            if s in (f"cuda:{i}",) or (s == "cuda" and i == 0) \
                    or s in g["name"].lower() or g["uuid"].lower() == s:
                return i, g
        return None


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.probe = FakeProbe()

    def _r(self, spec, **kw):
        kw.setdefault("listen_gpu", "cuda:1")
        kw.setdefault("reserve_mb", 512)
        kw.setdefault("probe", self.probe)
        return ld.resolve("parakeet", spec, **kw)

    def test_cpu_is_the_cpu_and_never_asks_the_driver(self):
        for spec in ("cpu", "CPU", "", None, "  "):
            p = self._r(spec)
            self.assertEqual((p.device, p.index, p.reason), ("cpu", None, ""))
        self.assertEqual(self.probe.calls, [])
        self.assertEqual(ld.fallback_line("parakeet", self._r("cpu")), "")

    def test_listen_means_the_listen_card(self):
        p = self._r("listen", need_mb=1200)
        self.assertEqual((p.device, p.index, p.reason), ("cuda:1", 1, ""))
        self.assertEqual(self.probe.calls, ["cuda:1"])
        p = self._r("listen", listen_gpu="1650", need_mb=1200)
        self.assertEqual(p.device, "cuda:1")
        self.assertEqual(ld.label(p), "cuda:1 (NVIDIA GeForce GTX 1650 SUPER)")

    def test_a_named_card_works_directly(self):
        p = self._r("cuda:1", need_mb=100)
        self.assertEqual(p.device, "cuda:1")

    def test_missing_card_is_the_cpu_with_one_line(self):
        p = self._r("listen", listen_gpu="cuda:2")
        self.assertEqual(p.device, "cpu")
        self.assertIn("listen card 'cuda:2' not found", p.reason)
        line = ld.fallback_line("parakeet", p)
        self.assertTrue(line.startswith("  [listen] parakeet: asked listen"))
        self.assertTrue(line.endswith("running on the CPU"))
        p = self._r("cuda:7")
        self.assertIn("cuda:7 not found", p.reason)

    def test_no_listen_card_set(self):
        p = self._r("listen", listen_gpu="")
        self.assertEqual(p.device, "cpu")
        self.assertIn("LISTEN_GPU is empty", p.reason)
        self.assertEqual(self.probe.calls, [])

    def test_unknown_spec(self):
        p = self._r("tpu")
        self.assertEqual(p.device, "cpu")
        self.assertIn("unknown device", p.reason)

    def test_room_is_need_plus_reserve(self):
        # The 1650 has 2,769 MB free.
        self.assertEqual(self._r("listen", need_mb=2257).device, "cuda:1")
        p = self._r("listen", need_mb=2258)
        self.assertEqual(p.device, "cpu")
        self.assertIn("2769 MB free < 2258 + 512 MB reserve", p.reason)
        self.assertEqual(self._r("listen", need_mb=2769,
                                 reserve_mb=0).device, "cuda:1")

    def test_default_need_is_the_models(self):
        p = ld.resolve("voice_id", "listen", listen_gpu="cuda:0",
                       reserve_mb=512, probe=self.probe)
        # 3090: 1,100 MB free < 400 + 512? no: 912 fits.
        self.assertEqual(p.device, "cuda:0")
        p = ld.resolve("parakeet", "listen", listen_gpu="cuda:0",
                       reserve_mb=512, probe=self.probe)
        self.assertEqual(p.device, "cpu")           # 1,100 < 1,200 + 512

    def test_runtime_refusal(self):
        p = self._r("listen", need_mb=1,
                    runtime=lambda i: (False, "no GPU runtime here"))
        self.assertEqual(p.device, "cpu")
        self.assertEqual(p.reason, "no GPU runtime here")
        p = self._r("listen", need_mb=1, runtime=lambda i: 1 / 0)
        self.assertEqual(p.device, "cpu")
        self.assertIn("runtime check failed", p.reason)

    def test_a_raising_probe_never_raises(self):
        class Boom:
            def find(self, spec, env=None):
                raise OSError("driver")
        p = ld.resolve("parakeet", "listen", probe=Boom())
        self.assertEqual(p.device, "cpu")
        self.assertIn("device check failed", p.reason)


class OrtTests(unittest.TestCase):
    def test_cuda_provider_required(self):
        cpu_build = types.SimpleNamespace(
            get_available_providers=lambda: ["AzureExecutionProvider",
                                             "CPUExecutionProvider"])
        ok, why = ld.ort_cuda_runtime(cpu_build)(1)
        self.assertFalse(ok)
        self.assertIn("onnxruntime has no CUDA provider", why)
        gpu_build = types.SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider",
                                             "CPUExecutionProvider"])
        self.assertEqual(ld.ort_cuda_runtime(gpu_build)(1), (True, ""))

    def test_providers(self):
        on_card = ld.Placement("cuda:1", 1, "listen", "", GPUS[1])
        self.assertEqual(ld.ort_providers(on_card),
                         [("CUDAExecutionProvider", {"device_id": 1}),
                          "CPUExecutionProvider"])
        cpu = ld.Placement("cpu", None, "cpu", "", None)
        self.assertEqual(ld.ort_providers(cpu), ["CPUExecutionProvider"])
        self.assertEqual(ld.ort_providers(None), ["CPUExecutionProvider"])


class DefaultsTests(unittest.TestCase):
    """The shipped defaults are the measured best: every listening model on
    the CPU except Whisper (WHISPER_DEVICE, unchanged), the listen card the
    second card by PCI bus, a 512 MB reserve."""

    def test_config_defaults(self):
        import ast
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "core", "config.py")
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        lits = {}
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                for t in targets:
                    if isinstance(t, ast.Name):
                        try:
                            lits[t.id] = ast.literal_eval(node.value)
                        except Exception:
                            pass
        self.assertEqual(lits["PARAKEET_DEVICE"], "cpu")
        self.assertEqual(lits["SMART_TURN_DEVICE"], "cpu")
        self.assertEqual(lits["VOICE_ID_DEVICE"], "cpu")
        self.assertEqual(lits["LISTEN_GPU"], ld.DEFAULT_LISTEN_GPU)
        self.assertEqual(lits["LISTEN_GPU_RESERVE_MB"], ld.DEFAULT_RESERVE_MB)
        self.assertEqual(lits["WHISPER_DEVICE"], "auto")


if __name__ == "__main__":
    unittest.main()
