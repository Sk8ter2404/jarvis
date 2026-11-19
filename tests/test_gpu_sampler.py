"""core/gpu_sampler.py - one cached in-process GPU reading (v3.0.26).

THE FINDING (overhaul reliability-05): the unified HUD started
``nvidia-smi --query-gpu=utilization.gpu`` once a second for as long as it
ran (~86,000 process starts a day), on top of the HUD's VRAM panel (6 s), the
pulse strip (15 s, two starts) and the diagnostics panel (20 s), each with its
own nvidia-smi. Each start loads and initialises the NVIDIA management
library (NVML) on both cards. core/gpu_probe already reads NVML in-process.

Pinned here (a fake NVML library - no driver, no GPU, no process):
  * gpu_probe reads temperature, power and the allocated memory (the figure
    nvidia-smi prints as memory.used), each optional;
  * the sampler aggregates the cards (busiest util, hottest temp, summed
    memory and power), caches the reading for max_age_s, re-reads after it,
    drops a reading taken through a replaced library, answers None without
    NVML, and never holds a caller on a stuck driver read (at most one read
    in flight, a bounded wait);
  * every all-day poller - core/gpu_usage (HUD VRAM panel), skills/
    system_pulse (strip + pinned-GPU check), skills/status_panel - answers
    from NVML WITHOUT starting nvidia-smi, and still falls back to it where
    NVML is unavailable;
  * the HUD overlays use the sampler first, start nvidia-smi at most every
    GPU_SMI_FALLBACK_S without NVML (was every second), and the unified HUD
    samples nothing while hidden (tests/skills/test_hud_gpu_util.py);
  * a test run never loads the real NVML (JARVIS_NO_NVML=1 from
    tests/__init__.py; tools/hermetic_guard refuses the library too).

    python tools/run_tests.py test_gpu_sampler
"""
from __future__ import annotations

import contextlib
import ctypes
import io
import os
import threading
import time
import types
import unittest
from unittest import mock

from core import gpu_probe as gp
from core import gpu_sampler as gs
from tests._skill_harness import load_skill_isolated

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MB = 1024 * 1024

# Synthetic cards (never a real name or UUID).
CARDS = [
    {"name": b"Synthetic GPU A", "uuid": b"GPU-synth-aaaa", "bus": 1,
     "total": 24576, "free": 4000, "reserved": 300, "util": 37, "temp": 61,
     "mw": 112500},
    {"name": b"Synthetic GPU B", "uuid": b"GPU-synth-bbbb", "bus": 8,
     "total": 4096, "free": 2000, "reserved": 100, "util": 92, "temp": 48,
     "mw": 15250},
]


class FakeNvml:
    """Just enough of nvml.dll for core/gpu_probe (v1 + the v3.0.26 calls),
    writing through ctypes byref()/buffers the way the real library does.
    ``reads`` counts device-count calls (one per full read)."""

    def __init__(self, cards=CARDS, extras=True):
        self.cards = [dict(c) for c in cards]
        self.reads = 0
        self.inits = 0
        self.gate = None                  # threading.Event: block reads
        if not extras:
            for name in ("nvmlDeviceGetTemperature",
                         "nvmlDeviceGetPowerUsage",
                         "nvmlDeviceGetMemoryInfo_v2"):
                setattr(self, name, None)  # "missing" -> raises when called

    def nvmlInit_v2(self):
        self.inits += 1
        return 0

    def nvmlDeviceGetCount_v2(self, ref):
        self.reads += 1
        if self.gate is not None:
            self.gate.wait(10.0)
        ref._obj.value = len(self.cards)
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, i, ref):
        ref._obj.value = int(i.value) + 1
        return 0

    def _card(self, h):
        return self.cards[int(h.value) - 1]

    def nvmlDeviceGetMemoryInfo(self, h, ref):
        c = self._card(h)
        ref._obj.total = c["total"] * _MB
        ref._obj.free = c["free"] * _MB
        ref._obj.used = (c["total"] - c["free"]) * _MB
        return 0

    def nvmlDeviceGetMemoryInfo_v2(self, h, ref):
        c = self._card(h)
        assert ref._obj.version == gp._MEMORY_V2_VERSION, "unversioned v2 call"
        ref._obj.total = c["total"] * _MB
        ref._obj.reserved = c["reserved"] * _MB
        ref._obj.free = c["free"] * _MB
        ref._obj.used = (c["total"] - c["free"] - c["reserved"]) * _MB
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

    def nvmlDeviceGetTemperature(self, h, sensor, ref):
        assert int(sensor.value) == 0, "NVML_TEMPERATURE_GPU"
        ref._obj.value = self._card(h)["temp"]
        return 0

    def nvmlDeviceGetPowerUsage(self, h, ref):
        ref._obj.value = self._card(h)["mw"]
        return 0


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class _Base(unittest.TestCase):
    def setUp(self):
        self.lib = FakeNvml()
        gp.set_loader(lambda: self.lib)
        self.addCleanup(gp.set_loader, None)
        self.clock = _Clock()
        gs.set_clock(self.clock)
        self.addCleanup(gs.set_clock, None)

    def no_nvml(self):
        gp.set_loader(lambda: None)


class ProbeExtrasTests(_Base):
    def test_reads_temperature_power_and_allocated_memory(self):
        a, b = gp.gpus()
        self.assertEqual((a["temp_c"], a["power_w"]), (61, 112.5))
        self.assertEqual((b["temp_c"], b["power_w"]), (48, 15.2))
        # nvidia-smi's memory.used = total - free - reserved (v2); v1's used
        # (unchanged) still counts the reservation.
        self.assertEqual(a["alloc_mb"], 24576 - 4000 - 300)
        self.assertEqual(a["used_mb"], 24576 - 4000)

    def test_each_extra_is_optional(self):
        self.lib = FakeNvml(extras=False)
        gp.set_loader(lambda: self.lib)
        a = gp.gpus()[0]
        self.assertEqual((a["temp_c"], a["power_w"], a["alloc_mb"]),
                         (None, None, None))
        self.assertEqual(a["free_mb"], 4000)       # the v1 reading stands

    def test_memory_v2_layout(self):
        # version (uint) + 4 bytes padding + 4 unsigned long longs; the
        # version word is NVML_STRUCT_VERSION(Memory, 2).
        self.assertEqual(ctypes.sizeof(gp._MemoryV2), 40)
        self.assertEqual(gp._MEMORY_V2_VERSION, 40 | (2 << 24))


class SamplerTests(_Base):
    def test_aggregates_the_cards(self):
        s = gs.sample()
        self.assertEqual(s["source"], "nvml")
        self.assertEqual(s["util_pct"], 92)            # busiest card
        self.assertEqual(s["temp_c"], 61)              # hottest card
        self.assertEqual(s["used_mb"],
                         (24576 - 4000 - 300) + (4096 - 2000 - 100))
        self.assertEqual(s["free_mb"], 6000)
        self.assertEqual(s["total_mb"], 24576 + 4096)
        self.assertEqual(s["power_w"], 127.7)
        self.assertEqual([g["index"] for g in s["gpus"]], [0, 1])
        self.assertEqual(gs.util_pct(), 92.0)
        self.assertEqual(gs.temp_c(), 61.0)

    def test_one_read_per_max_age(self):
        gs.sample(2.0)
        self.clock.t += 1.9
        gs.sample(2.0)
        gs.util_pct(2.0)
        self.assertEqual(self.lib.reads, 1)
        self.clock.t += 0.2                             # 2.1 s after the read
        self.lib.cards[1]["util"] = 10
        self.assertEqual(gs.util_pct(2.0), 37.0)
        self.assertEqual(self.lib.reads, 2)

    def test_a_shorter_max_age_reads_again(self):
        gs.sample(5.0)
        self.clock.t += 1.5
        gs.sample(1.0)
        self.assertEqual(self.lib.reads, 2)

    def test_a_replaced_library_drops_the_cached_reading(self):
        self.assertEqual(gs.util_pct(60.0), 92.0)
        other = FakeNvml(cards=[dict(CARDS[0], util=5)])
        gp.set_loader(lambda: other)
        self.assertEqual(gs.util_pct(60.0), 5.0)

    def test_no_nvml_is_none(self):
        self.no_nvml()
        self.assertIsNone(gs.sample())
        self.assertIsNone(gs.util_pct())
        self.assertIsNone(gs.temp_c())

    def test_a_stuck_driver_read_never_holds_the_caller(self):
        self.assertEqual(gs.util_pct(1.0), 92.0)        # a reading to fall back on
        self.clock.t += 5.0
        self.lib.gate = threading.Event()
        self.addCleanup(self.lib.gate.set)
        t0 = time.monotonic()
        out = io.StringIO()
        with mock.patch.object(gs, "_READ_WAIT_S", 0.05), \
                mock.patch.object(gs, "_slow_logged", [False]), \
                contextlib.redirect_stdout(out):
            first = gs.util_pct(1.0)
            second = gs.util_pct(1.0)
        took = time.monotonic() - t0
        self.assertEqual((first, second), (92.0, 92.0))  # the last reading
        self.assertLess(took, 2.0)
        self.assertEqual(self.lib.reads, 2)    # ONE stuck read, not one per call
        self.assertEqual(out.getvalue().count("[gpu-sampler] an NVML read is "
                                              "still running"), 1)
        self.lib.cards[1]["util"] = 11
        self.lib.gate.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and gs._inflight["thread"].is_alive():
            time.sleep(0.01)
        self.assertEqual(gs.util_pct(1.0), 37.0)  # the stuck read landed

    def test_never_raises(self):
        with mock.patch.object(gs, "_read", side_effect=RuntimeError("x")):
            self.assertIsNone(gs.sample(0.0))
        gs.reset()          # a failed read is cached (not retried at once)
        self.assertEqual(gs.sample("not a number")["util_pct"], 92)

    def test_a_fake_thread_left_behind_never_wedges_it(self):
        # A test elsewhere that patches threading.Thread with a stand-in
        # (no is_alive) must not leave the sampler answering None for the
        # rest of the run.
        gs._inflight["thread"] = types.SimpleNamespace()
        gs._inflight["done"] = threading.Event()
        self.assertEqual(gs.util_pct(0.0), 92.0)


class NoRealNvmlInTestsTests(unittest.TestCase):
    def test_the_test_package_sets_the_switch(self):
        self.assertEqual(os.environ.get("JARVIS_NO_NVML"), "1")
        with open(os.path.join(_PROJECT, "tests", "__init__.py"),
                  encoding="utf-8") as fh:
            self.assertIn('setdefault("JARVIS_NO_NVML", "1")', fh.read())

    def test_the_default_loader_loads_nothing_under_it(self):
        with mock.patch.object(gp.ctypes, "CDLL") as cdll:
            self.assertIsNone(gp._default_loader())
        cdll.assert_not_called()


# ─── the pollers ─────────────────────────────────────────────────────────

def _no_spawn(testcase):
    """subprocess.run / shutil.which that fail the test when touched."""
    def _boom(*a, **k):
        testcase.fail(f"nvidia-smi was started: {a!r}")
    return _boom


class GpuUsagePanelTests(_Base):
    def setUp(self):
        super().setUp()
        from core import gpu_usage
        self.gu = gpu_usage
        p = mock.patch.object(gpu_usage, "_resident_models", return_value=[])
        p.start()
        self.addCleanup(p.stop)

    def test_nvml_answers_without_nvidia_smi(self):
        with mock.patch.object(self.gu, "_run_nvidia_smi_query",
                               side_effect=_no_spawn(self)), \
                mock.patch.object(self.gu.subprocess, "run",
                                  side_effect=_no_spawn(self)):
            snap = self.gu.gpu_snapshot(use_cache=False)
        self.assertEqual(snap["util_pct"], 92)
        self.assertEqual(snap["temp_c"], 61)
        self.assertEqual(snap["total_mb"], 24576 + 4096)
        self.assertEqual(snap["free_mb"], 6000)
        self.assertEqual(snap["used_mb"],
                         (24576 - 4000 - 300) + (4096 - 2000 - 100))

    def test_without_nvml_nvidia_smi_still_answers(self):
        self.no_nvml()
        with mock.patch.object(self.gu, "_run_nvidia_smi_query",
                               return_value="100, 900, 1000, 7, 40\n") as q:
            snap = self.gu.gpu_snapshot(use_cache=False)
        q.assert_called_once()
        self.assertEqual((snap["used_mb"], snap["util_pct"]), (100, 7))


class PulseStripTests(_Base):
    def setUp(self):
        super().setUp()
        self.mod, _ = load_skill_isolated("system_pulse")

    def test_util_temp_and_pinned_from_nvml(self):
        with mock.patch.object(self.mod.shutil, "which",
                               side_effect=_no_spawn(self)), \
                mock.patch.object(self.mod.subprocess, "run",
                                  side_effect=_no_spawn(self)):
            self.assertEqual(self.mod._read_gpu_util_pct(), 92.0)
            self.assertEqual(self.mod._read_gpu_temp_c(), 61.0)
            self.lib.cards[0]["util"] = 99
            self.clock.t += 10.0
            self.assertEqual(self.mod._read_pinned_gpus(), {0})
            self.lib.cards[0]["util"] = 3
            self.clock.t += 10.0
            self.assertEqual(self.mod._read_pinned_gpus(), set())

    def test_the_resample_gap_always_gets_a_new_reading(self):
        # _qualify_gpu_reason re-reads GPU_RESAMPLE_GAP_S after the first
        # sample to see whether the load is sustained: the shared reading
        # must be younger than that gap, or the "second" sample is the first.
        self.assertLess(self.mod.GPU_SHARED_MAX_AGE_S,
                        self.mod.GPU_RESAMPLE_GAP_S)
        self.assertEqual(self.mod._read_gpu_util_pct(), 92.0)
        self.lib.cards[1]["util"] = 20
        self.clock.t += self.mod.GPU_RESAMPLE_GAP_S
        self.assertEqual(self.mod._read_gpu_util_pct(), 37.0)

    def test_without_nvml_nvidia_smi_still_answers(self):
        self.no_nvml()
        proc = types.SimpleNamespace(stdout="44\n")
        with mock.patch.object(self.mod.shutil, "which",
                               return_value="nvidia-smi"), \
                mock.patch.object(self.mod.subprocess, "run",
                                  return_value=proc) as run:
            self.assertEqual(self.mod._read_gpu_util_pct(), 44.0)
        run.assert_called_once()


class DiagnosticsPanelTests(_Base):
    def setUp(self):
        super().setUp()
        self.mod, _ = load_skill_isolated("status_panel")

    def test_nvml_answers_without_nvidia_smi(self):
        with mock.patch.object(self.mod.shutil, "which",
                               side_effect=_no_spawn(self)), \
                mock.patch.object(self.mod.subprocess, "run",
                                  side_effect=_no_spawn(self)):
            self.assertEqual(self.mod._read_gpu_util_pct(), 92.0)

    def test_without_nvml_nvidia_smi_still_answers(self):
        self.no_nvml()
        proc = types.SimpleNamespace(stdout="12\n")
        with mock.patch.object(self.mod.shutil, "which",
                               return_value="nvidia-smi"), \
                mock.patch.object(self.mod.subprocess, "run",
                                  return_value=proc) as run:
            self.assertEqual(self.mod._read_gpu_util_pct(), 12.0)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
