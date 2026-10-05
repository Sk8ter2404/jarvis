"""The monolith's Whisper paths: no thread that exits owns CTranslate2 CUDA
state, and the CUDA driver starts before ctranslate2 is imported (v2.0.180).

THE CRASH (v2.0.179, 2026-10-04 20:40): a short-lived thread decoded with the
cuda:1 faster-whisper model and exited; CTranslate2's per-thread CUDA
destructors threw during the loader's thread teardown -> std::terminate ->
abort (crash dumps). 179 had also lost 178's DLL order (driver before
ctranslate2), which is what had made those exits survivable.

Pinned here against the REAL monolith functions, with a fake model (no GPU,
no ctranslate2):
  * transcribe() called from a short-lived thread decodes - transcribe() AND
    the generator drain - on the ct2-host thread for a CUDA model, inline for
    a CPU one;
  * _ensure_whisper() builds a CUDA model on the host, and starts the driver
    BEFORE faster_whisper is imported (an import hook records the order);
  * _resolve_whisper_device() / _ctranslate2_sees_cuda() start the driver
    before importing ctranslate2 and run the device count on the host;
  * a CUDA model dropped after a CUDA error is released on the host.

    python -m unittest tests.monolith.test_monolith_ct2_thread_exit
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import types
import unittest
import weakref
from unittest import mock

import numpy as np

from tests._monolith_harness import (MonolithGlobalsTestCase, load_monolith,
                                     requires_monolith)


class FakeCT2Whisper:
    """faster-whisper's WhisperModel: records the thread of every call that
    would leave CTranslate2 CUDA state on its caller."""

    def __init__(self, *a, **kw):
        self.kw = kw
        self.built_on = threading.current_thread()
        self.state_threads: list = []

    def _touch(self):
        self.state_threads.append(threading.current_thread())

    def transcribe(self, audio, **kw):
        self._touch()

        def _gen():
            self._touch()
            yield types.SimpleNamespace(text="turn on the lights",
                                        no_speech_prob=0.1, avg_logprob=-0.2)
        return _gen(), types.SimpleNamespace(no_speech_prob=0.1)


def _on_short_lived_thread(fn, name="probe-stt"):
    box = {}

    def _run():
        try:
            box["result"] = fn()
        except BaseException as e:
            box["exc"] = e
        box["thread"] = threading.current_thread()
    t = threading.Thread(target=_run, name=name, daemon=True)
    t.start()
    t.join(15.0)
    assert not t.is_alive(), "short-lived caller did not finish"
    if "exc" in box:
        raise box["exc"]
    return box


class _RecordingFinder:
    """A meta-path finder that serves fake modules and records WHEN each is
    imported (the moment ctranslate2.dll would load)."""

    def __init__(self, mods, events):
        self.mods = mods
        self.events = events

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.mods:
            self.events.append(f"import {fullname}")
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return self.mods[spec.name]

    def exec_module(self, module):
        pass


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = (self.bc._stt, self.bc._stt_device,
                       self.bc._stt_model_name, self.bc._stt_engine)

    def tearDown(self):
        (self.bc._stt, self.bc._stt_device,
         self.bc._stt_model_name, self.bc._stt_engine) = self._saved

    def _imports(self, mods, events):
        """Serve `mods` through a recording finder; drop any real copies."""
        finder = _RecordingFinder(mods, events)
        missing = object()
        for name in mods:
            saved = sys.modules.pop(name, missing)

            def _restore(name=name, saved=saved):
                if saved is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = saved
            self.addCleanup(_restore)
        sys.meta_path.insert(0, finder)
        self.addCleanup(sys.meta_path.remove, finder)

    def _driver_recorder(self, events):
        def _rec(*a, **k):
            events.append("driver")
            return {"done": True, "ran": True, "ok": True, "late": False,
                    "rc": 0, "detail": "cuInit(0) = 0"}
        p = mock.patch.object(self.bc._cuda_preinit, "before_ctranslate2",
                              side_effect=_rec)
        p.start()
        self.addCleanup(p.stop)


class TranscribeThreadTests(_Base):
    def _vocab(self):
        return (mock.patch.object(self.bc, "_ensure_whisper"),
                mock.patch.object(self.bc, "STT_HOTWORDS", "", create=True),
                mock.patch.object(self.bc, "STT_REPLACEMENTS", {}, create=True))

    def test_cuda_decode_from_a_short_lived_thread_runs_on_the_host(self):
        model = FakeCT2Whisper()
        self.bc._stt, self.bc._stt_engine = model, "faster_whisper"
        self.bc._stt_device = "cuda:1"
        a, b, c = self._vocab()
        with a, b, c:
            box = _on_short_lived_thread(
                lambda: self.bc.transcribe(np.zeros(1600, dtype=np.float32)))
        text, _conf = box["result"]
        self.assertEqual(text, "turn on the lights")
        self.assertEqual(len(model.state_threads), 2)
        for t in model.state_threads:
            self.assertIsNot(t, box["thread"])
            self.assertEqual(t.name, "ct2-host")

    def test_cpu_decode_stays_on_the_caller(self):
        model = FakeCT2Whisper()
        self.bc._stt, self.bc._stt_engine = model, "faster_whisper"
        self.bc._stt_device = "cpu"
        a, b, c = self._vocab()
        with a, b, c:
            box = _on_short_lived_thread(
                lambda: self.bc.transcribe(np.zeros(1600, dtype=np.float32)))
        self.assertEqual({t.name for t in model.state_threads}, {"probe-stt"})
        self.assertIs(model.state_threads[0], box["thread"])

    def test_a_dropped_cuda_model_is_released_on_the_host(self):
        freed_on = []

        class _Failing(FakeCT2Whisper):
            def transcribe(self, audio, **kw):
                raise RuntimeError("CUDA failed with error out of memory")

        def _install():
            m = _Failing()
            weakref.finalize(m, lambda: freed_on.append(threading.current_thread()))
            self.bc._stt = m            # the global is the only reference
        _install()
        self.bc._stt_engine, self.bc._stt_device = "faster_whisper", "cuda:1"
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = False
        a, b, c = self._vocab()
        with a, b, c, mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                mock.patch.object(self.bc._ct2_host, "_RETIRE_GRACE_S", 0.2):
            box = _on_short_lived_thread(
                lambda: self.bc.transcribe(np.zeros(1600, dtype=np.float32)))
            self.assertEqual(box["result"][0], "")
            self.assertIsNone(self.bc._stt)
            for _ in range(80):
                if freed_on:
                    break
                self.bc._ct2_host.run(lambda: None)
                threading.Event().wait(0.05)
        self.assertEqual(len(freed_on), 1)
        self.assertEqual(freed_on[0].name, "ct2-host")


class LoadOrderAndHostTests(_Base):
    def test_cuda_model_is_built_on_the_host_after_the_driver_started(self):
        events = []
        fw = types.ModuleType("faster_whisper")
        fw.WhisperModel = FakeCT2Whisper
        self._imports({"faster_whisper": fw}, events)
        self._driver_recorder(events)
        self.bc._stt = None
        with mock.patch.object(self.bc, "_register_cuda_dll_dirs"), \
                mock.patch.object(self.bc, "_resolve_whisper_device",
                                  return_value="cuda:1"), \
                mock.patch.object(self.bc, "_force_whisper_cpu_int8", False), \
                mock.patch.object(self.bc, "_whisper_cuda_plan",
                                  return_value=("int8", False, "fits")), \
                mock.patch.object(self.bc, "_log_cuda_load_order"):
            box = _on_short_lived_thread(self.bc._ensure_whisper)
        model = self.bc._stt
        self.assertIsInstance(model, FakeCT2Whisper)
        self.assertEqual(model.kw.get("device"), "cuda")
        self.assertEqual(model.kw.get("device_index"), 1)
        self.assertEqual(model.built_on.name, "ct2-host")
        self.assertIsNot(model.built_on, box["thread"])
        self.assertEqual(self.bc._stt_device, "cuda:1")
        self.assertIn("driver", events)
        self.assertIn("import faster_whisper", events)
        self.assertLess(events.index("driver"),
                        events.index("import faster_whisper"))

    def test_auto_device_check_starts_the_driver_before_ctranslate2(self):
        events = []
        counted_on = []
        ct2 = types.ModuleType("ctranslate2")

        def _count():
            counted_on.append(threading.current_thread())
            return 1
        ct2.get_cuda_device_count = _count
        self._imports({"ctranslate2": ct2}, events)
        self._driver_recorder(events)
        with mock.patch.object(self.bc, "WHISPER_DEVICE", "auto"):
            box = _on_short_lived_thread(self.bc._resolve_whisper_device)
        self.assertEqual(box["result"], "cuda")
        self.assertLess(events.index("driver"), events.index("import ctranslate2"))
        self.assertEqual([t.name for t in counted_on], ["ct2-host"])

    def test_preflight_cuda_check_starts_the_driver_before_ctranslate2(self):
        events = []
        counted_on = []
        ct2 = types.ModuleType("ctranslate2")

        def _count():
            counted_on.append(threading.current_thread())
            return 0
        ct2.get_cuda_device_count = _count
        self._imports({"ctranslate2": ct2}, events)
        self._driver_recorder(events)
        self.assertFalse(self.bc._ctranslate2_sees_cuda())
        self.assertLess(events.index("driver"), events.index("import ctranslate2"))
        self.assertEqual([t.name for t in counted_on], ["ct2-host"])


if __name__ == "__main__":
    unittest.main()
