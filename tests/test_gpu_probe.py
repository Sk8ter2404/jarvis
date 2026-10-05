"""core/gpu_probe.py — free VRAM from NVML without a CUDA context (2026-10-04).

THE LIVE FINDING: every free-VRAM check in JARVIS asked
``torch.cuda.mem_get_info(i)``, which CREATES a CUDA context on GPU ``i`` the
first time it runs (+59 MB measured on this box's torch 2.10, kept until the
process exits). Four copies asked GPU 0 — the RTX 3090 that holds the local
brain with ~1.1 GB to spare. NVML reads the same counter from the driver with
no context.

Pinned here (stdlib only, a fake NVML library — no driver, no GPU):
  * the ctypes reader fills name / UUID / PCI bus / MB from the library;
  * CUDA's numbering is mapped onto NVML's ONLY when it can be known
    (CUDA_DEVICE_ORDER=PCI_BUS_ID, or one GPU), CUDA_VISIBLE_DEVICES honoured
    (empty = no GPU, indices, UUID prefixes, stop at a bad entry);
  * find() for 'cuda', 'cuda:N', a UUID prefix and a piece of the name;
  * NVML missing / nvmlInit failing = unknown (None), loaded and initialised
    once, never raising;
  * a GPU NVML cannot read KEEPS ITS PLACE (review 2026-10-04: skipping it
    moved every later card down one CUDA index, so the 1650's free VRAM read
    as the 3090's and 'listen' = '1650' resolved to cuda:0, the brain's card);
  * the module imports nothing that could open a CUDA context.

    python -m unittest tests.test_gpu_probe
"""
from __future__ import annotations

import ast
import ctypes
import os
import unittest

from core import gpu_probe as gp

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_MB = 1024 * 1024
# Synthetic cards (never a real UUID).
CARDS = [
    {"name": b"NVIDIA GeForce RTX 3090", "uuid": b"GPU-aaaa1111-0000",
     "bus": 1, "total": 24576, "free": 1100, "util": 17},
    {"name": b"NVIDIA GeForce GTX 1650 SUPER", "uuid": b"GPU-bbbb2222-0000",
     "bus": 8, "total": 4096, "free": 2769, "util": 0},
]


class FakeNvml:
    """Just enough of nvml.dll for core/gpu_probe: writes through ctypes
    byref()/buffers the way the real library does."""

    def __init__(self, cards=CARDS, init_rc=0):
        self.cards = cards
        self.init_rc = init_rc
        self.inits = 0

    def nvmlInit_v2(self):
        self.inits += 1
        return self.init_rc

    def nvmlDeviceGetCount_v2(self, ref):
        ref._obj.value = len(self.cards)
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, i, ref):
        rc = self.cards[int(i.value)].get("handle_rc", 0)
        if rc:
            return rc                              # e.g. GPU lost (15)
        ref._obj.value = int(i.value) + 1          # handle = index + 1
        return 0

    def _card(self, h):
        return self.cards[int(h.value) - 1]

    def nvmlDeviceGetMemoryInfo(self, h, ref):
        c = self._card(h)
        if c.get("mem_rc"):
            return c["mem_rc"]
        ref._obj.total = c["total"] * _MB
        ref._obj.free = c["free"] * _MB
        ref._obj.used = (c["total"] - c["free"]) * _MB
        return 0

    def nvmlDeviceGetName(self, h, buf, _n):
        buf.value = self._card(h)["name"]
        return 0

    def nvmlDeviceGetUUID(self, h, buf, _n):
        buf.value = self._card(h)["uuid"]
        return 0

    def nvmlDeviceGetPciInfo_v3(self, h, ref):
        ref._obj.bus = self._card(h)["bus"]
        return 0

    def nvmlDeviceGetUtilizationRates(self, h, ref):
        ref._obj.gpu = self._card(h)["util"]
        return 0


class _Base(unittest.TestCase):
    def setUp(self):
        self.lib = FakeNvml()
        gp.set_loader(lambda: self.lib)
        self.addCleanup(gp.set_loader, None)


class ReaderTests(_Base):
    def test_reads_every_card_in_nvml_order(self):
        got = gp.gpus()
        self.assertEqual([g["index"] for g in got], [0, 1])
        g = got[1]
        self.assertEqual(g["name"], "NVIDIA GeForce GTX 1650 SUPER")
        self.assertEqual(g["uuid"], "GPU-bbbb2222-0000")
        self.assertEqual((g["pci_bus"], g["total_mb"], g["free_mb"],
                          g["used_mb"], g["util_pct"]),
                         (8, 4096, 2769, 4096 - 2769, 0))

    def test_initialises_once(self):
        gp.gpus()
        gp.gpus()
        gp.cuda_gpus({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
        self.assertEqual(self.lib.inits, 1)

    def test_no_library_is_unknown(self):
        gp.set_loader(lambda: None)
        self.assertFalse(gp.available())
        self.assertEqual(gp.gpus(), [])
        self.assertIsNone(gp.cuda_gpus({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}))
        self.assertIsNone(gp.cuda_memory_mb(0, {}))

    def test_failed_init_is_unknown_and_not_retried(self):
        lib = FakeNvml(init_rc=9)
        gp.set_loader(lambda: lib)
        self.assertFalse(gp.available())
        self.assertFalse(gp.available())
        self.assertEqual(lib.inits, 1)
        self.assertIsNone(gp.cuda_memory_mb(0, {}))

    def test_a_raising_library_never_raises(self):
        lib = FakeNvml()
        lib.nvmlDeviceGetCount_v2 = lambda ref: 1 / 0
        gp.set_loader(lambda: lib)
        self.assertEqual(gp.gpus(), [])

        def boom():
            raise OSError("no driver")
        gp.set_loader(boom)
        self.assertFalse(gp.available())


class CudaNumberingTests(_Base):
    PCI = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

    def test_pci_order_is_nvml_order(self):
        self.assertEqual([g["pci_bus"] for g in gp.cuda_gpus(self.PCI)],
                         [1, 8])
        self.assertEqual(gp.cuda_memory_mb(1, self.PCI), (2769, 4096))
        self.assertEqual(gp.cuda_memory_mb(0, self.PCI), (1100, 24576))
        self.assertIsNone(gp.cuda_memory_mb(2, self.PCI))

    def test_fastest_first_with_two_cards_is_unknown(self):
        # CUDA's default order cannot be derived from NVML: never a guess.
        self.assertIsNone(gp.cuda_gpus({}))
        self.assertIsNone(gp.cuda_memory_mb(0, {}))
        self.assertIsNone(gp.cuda_gpus({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}))

    def test_one_card_needs_no_order(self):
        self.lib.cards = CARDS[:1]
        self.assertEqual(gp.cuda_memory_mb(0, {}), (1100, 24576))

    def test_visible_devices(self):
        env = dict(self.PCI)
        env["CUDA_VISIBLE_DEVICES"] = ""
        self.assertEqual(gp.cuda_gpus(env), [])            # the test runners
        self.assertIsNone(gp.cuda_memory_mb(0, env))
        env["CUDA_VISIBLE_DEVICES"] = "1"
        self.assertEqual(gp.cuda_memory_mb(0, env), (2769, 4096))
        env["CUDA_VISIBLE_DEVICES"] = "1,0"
        self.assertEqual([g["pci_bus"] for g in gp.cuda_gpus(env)], [8, 1])
        env["CUDA_VISIBLE_DEVICES"] = "0,-1,1"                 # stops at -1
        self.assertEqual([g["pci_bus"] for g in gp.cuda_gpus(env)], [1])
        env["CUDA_VISIBLE_DEVICES"] = "0,7"                    # stops at 7
        self.assertEqual(len(gp.cuda_gpus(env)), 1)

    def test_visible_devices_by_uuid(self):
        env = {"CUDA_VISIBLE_DEVICES": "GPU-bbbb"}          # any order is fine
        self.assertEqual(gp.cuda_memory_mb(0, env), (2769, 4096))
        env["CUDA_VISIBLE_DEVICES"] = "GPU-"                  # ambiguous
        self.assertIsNone(gp.cuda_gpus(env))

    def test_visible_indices_under_fastest_first_are_unknown(self):
        self.assertIsNone(gp.cuda_gpus({"CUDA_VISIBLE_DEVICES": "1"}))


class FindTests(_Base):
    PCI = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

    def _idx(self, spec):
        hit = gp.find(spec, self.PCI)
        return None if hit is None else hit[0]

    def test_specs(self):
        self.assertEqual(self._idx("cuda"), 0)
        self.assertEqual(self._idx("cuda:1"), 1)
        self.assertEqual(self._idx("CUDA:1"), 1)
        self.assertIsNone(self._idx("cuda:2"))
        self.assertIsNone(self._idx("cuda:x"))
        self.assertEqual(self._idx("GPU-bbbb"), 1)            # UUID prefix
        self.assertIsNone(self._idx("GPU-"))                  # ambiguous
        self.assertEqual(self._idx("1650"), 1)                # piece of name
        self.assertEqual(self._idx("rtx 3090"), 0)
        self.assertIsNone(self._idx("4090"))
        self.assertIsNone(self._idx(""))
        self.assertIsNone(self._idx(None))

    def test_unknown_numbering_finds_nothing(self):
        self.assertIsNone(gp.find("cuda:1", {}))

    def test_describe(self):
        idx, g = gp.find("1650", self.PCI)
        s = gp.describe(g, idx)
        self.assertTrue(s.startswith("cuda:1 NVIDIA GeForce GTX 1650 SUPER"))
        self.assertIn("bus 8", s)
        self.assertIn("2769/4096 MB free", s)
        self.assertEqual(gp.describe(None), "no GPU")


# The brain's card unreadable (a TDR / GPU-lost state), the 1650 fine. A
# whole-length synthetic UUID for the 1650 (never a real one).
UUID_1650 = "GPU-bbbb2222-0000-0000-0000-000000000000"


class UnreadableCardTests(_Base):
    PCI = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

    def setUp(self):
        super().setUp()
        self.lib.cards = [dict(CARDS[0], mem_rc=15),
                          dict(CARDS[1], uuid=UUID_1650.encode())]

    def test_the_cards_after_it_keep_their_index(self):
        lst = gp.cuda_gpus(self.PCI)
        self.assertEqual(len(lst), 2)
        self.assertTrue(lst[0].get("unreadable"))
        self.assertEqual(lst[1]["pci_bus"], 8)
        # cuda:0 is unknown — never the 1650's numbers read as the 3090's.
        self.assertIsNone(gp.cuda_memory_mb(0, self.PCI))
        self.assertEqual(gp.cuda_memory_mb(1, self.PCI), (2769, 4096))
        self.assertIsNone(gp.cuda_memory_mb(2, self.PCI))

    def test_a_failed_handle_keeps_its_place_too(self):
        self.lib.cards = [dict(CARDS[0], handle_rc=15), CARDS[1]]
        self.assertEqual(gp.cuda_memory_mb(1, self.PCI), (2769, 4096))
        self.assertIsNone(gp.cuda_memory_mb(0, self.PCI))

    def test_display_lists_only_readable_cards(self):
        self.assertEqual([g["pci_bus"] for g in gp.gpus()], [8])

    def test_find_names_the_right_index(self):
        self.assertEqual(gp.find("1650", self.PCI)[0], 1)
        self.assertEqual(gp.find(UUID_1650, self.PCI)[0], 1)
        self.assertEqual(gp.find(UUID_1650.lower(), self.PCI)[0], 1)
        # A short prefix might be the unreadable card's own: unknown.
        self.assertIsNone(gp.find("GPU-bbbb", self.PCI))
        self.assertIsNone(gp.find("3090", self.PCI))     # cannot be named
        idx, g = gp.find("cuda:0", self.PCI)
        self.assertEqual(idx, 0)
        self.assertTrue(g.get("unreadable"))
        self.assertEqual(gp.describe(g, 0), "cuda:0 GPU unreadable")

    def test_listen_never_lands_on_the_brains_card(self):
        from core import listen_devices as ld
        for spec in ("1650", UUID_1650, "cuda:1"):
            p = ld.resolve("voice_id", "listen", listen_gpu=spec,
                           reserve_mb=512, env=self.PCI)
            self.assertEqual(p.device, "cuda:1", spec)
        # Asked for the unreadable card itself: no room can be proved.
        p = ld.resolve("voice_id", "cuda:0", reserve_mb=512, env=self.PCI)
        self.assertEqual(p.device, "cpu")
        self.assertIn("unreadable", p.reason)


class SerialisedReadTests(unittest.TestCase):
    """v2.0.180 defence in depth (after the 2026-10-04 native abort, which
    was not NVML's doing): NVML is initialised once and its device reads run
    one at a time, so no two driver reads from this process ever overlap."""

    def test_device_reads_never_overlap(self):
        import threading
        lib = FakeNvml()
        lock = threading.Lock()
        active = [0]
        peak = [0]
        overlapped = threading.Event()
        real = lib.nvmlDeviceGetMemoryInfo

        def _slow(h, ref):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
                if active[0] > 1:
                    overlapped.set()
            overlapped.wait(0.1)      # every chance for a second reader to enter
            with lock:
                active[0] -= 1
            return real(h, ref)
        lib.nvmlDeviceGetMemoryInfo = _slow
        gp.set_loader(lambda: lib)
        self.addCleanup(gp.set_loader, None)
        ts = [threading.Thread(target=gp.gpus, daemon=True) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10.0)
        self.assertEqual(peak[0], 1)
        self.assertEqual(lib.inits, 1)


class NoContextTests(unittest.TestCase):
    """The probe must not be able to open a CUDA context: it imports nothing
    beyond the stdlib (no torch, ctranslate2, onnxruntime, pynvml)."""

    def test_imports_are_stdlib_only(self):
        with open(os.path.join(_PROJECT, "core", "gpu_probe.py"),
                  encoding="utf-8") as f:
            tree = ast.parse(f.read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
        self.assertLessEqual(mods, {"__future__", "ctypes", "os", "sys",
                                    "threading"})

    def test_real_structures_match_nvml_layout(self):
        # nvmlMemory_t is three unsigned long longs; nvmlPciInfo_t is 16 +
        # 5 * 4 + 32 bytes — a wrong layout would read garbage, not raise.
        self.assertEqual(ctypes.sizeof(gp._Memory), 24)
        self.assertEqual(ctypes.sizeof(gp._PciInfo), 16 + 5 * 4 + 32)


if __name__ == "__main__":
    unittest.main()
