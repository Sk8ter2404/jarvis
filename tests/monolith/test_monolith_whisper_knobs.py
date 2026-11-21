"""Speed plan R11 - Whisper decode knobs (WHISPER_BEAM_SIZE, WHISPER_TEMPERATURES).

no_speech_prob is always 0.0 on this rig, so faster-whisper's silence exit
never fires and every low-logprob decode walks all six temperatures - the
4-10 s transcript tails. The knobs let the owner cap that. Defaults keep the
decode byte-identical to before (beam 5, faster-whisper's own temperature
ladder: no temperature kwarg at all). The bounded no-VAD retry keeps its own
fixed beam 1 / temperature 0.0.
"""
from __future__ import annotations

import contextlib
import importlib
import sys
import types
import unittest
import weakref
from unittest import mock

import numpy as np

from tests._monolith_harness import MonolithGlobalsTestCase, load_monolith, requires_monolith


class _Seg:
    def __init__(self, text, nsp=0.1, lp=-0.2):
        self.text, self.no_speech_prob, self.avg_logprob = text, nsp, lp


@requires_monolith
class WhisperKnobTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = (self.bc._stt, self.bc._stt_engine)

    def tearDown(self):
        self.bc._stt, self.bc._stt_engine = self._saved

    def _calls(self, *, beam=None, temps=None, results=None):
        bc = self.bc
        fake = mock.Mock()
        info = mock.Mock(no_speech_prob=0.1)
        seq = results or [[_Seg("turn on the lights")]]
        fake.transcribe.side_effect = [(iter(s), info) for s in seq]
        patches = [mock.patch.object(bc, "_ensure_whisper"),
                   mock.patch.object(bc, "_stt", fake),
                   mock.patch.object(bc, "_stt_engine", "faster_whisper"),
                   mock.patch.object(bc, "STT_HOTWORDS", "", create=True)]
        if beam is not None:
            patches.append(mock.patch.object(bc, "WHISPER_BEAM_SIZE", beam, create=True))
        if temps is not None:
            patches.append(mock.patch.object(bc, "WHISPER_TEMPERATURES", temps, create=True))
        for p in patches:
            p.start()
        try:
            bc.transcribe(np.zeros(1600, dtype=np.float32))
        finally:
            for p in reversed(patches):
                p.stop()
        return [c.kwargs for c in fake.transcribe.call_args_list]

    def test_defaults_are_byte_identical(self):
        kw = self._calls()[0]
        self.assertEqual(kw["beam_size"], 5)
        self.assertNotIn("temperature", kw)
        self.assertTrue(kw["vad_filter"])

    def test_knobs_reach_the_vad_decode(self):
        kw = self._calls(beam=1, temps=(0.0, 0.2, 0.4))[0]
        self.assertEqual(kw["beam_size"], 1)
        self.assertEqual(kw["temperature"], (0.0, 0.2, 0.4))

    def test_a_single_temperature_is_accepted(self):
        kw = self._calls(temps=0.0)[0]
        self.assertEqual(kw["temperature"], 0.0)

    def test_the_no_vad_retry_keeps_its_own_bounds(self):
        calls = self._calls(beam=3, temps=(0.0, 0.2),
                            results=[[], [_Seg("hello")]])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["beam_size"], 3)
        self.assertFalse(calls[1]["vad_filter"])
        self.assertEqual(calls[1]["beam_size"], 1)
        self.assertEqual(calls[1]["temperature"], 0.0)

    def test_bad_values_fall_back_to_the_defaults(self):
        for beam, temps in ((0, ()), (-2, "hot"), ("five", [1.5]),
                            (99, [-0.1, 0.2]), (True, {"t": 0})):
            kw = self._calls(beam=beam, temps=temps)[0]
            self.assertEqual(kw["beam_size"], 5, (beam, temps))
            self.assertNotIn("temperature", kw, (beam, temps))


class _FakeWhisperLib:
    """openai-whisper: records (model, device); a CUDA load raises the
    11-20 OOM after building a 'partial model' that only the exception's
    traceback keeps alive."""

    def __init__(self):
        self.calls = []
        self.partial_freed = []
        self.freed_before_cpu = None

    def load_model(self, model, device="cpu"):
        self.calls.append((model, device))
        if str(device).startswith("cuda"):
            partial = type("_Partial", (), {})()
            weakref.finalize(partial, self.partial_freed.append, True)
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 "
                               "MiB. GPU 1 has a total capacity of 4.00 GiB "
                               "of which 0 bytes is free")
        self.freed_before_cpu = bool(self.partial_freed)
        return ("openai-model", model, device)


@requires_monolith
class WhisperEngineFallbackTests(MonolithGlobalsTestCase):
    """The 11-20 Whisper load (n20-F1, DL-1, DL-2): an import deadlock is
    retried on faster-whisper; any other faster-whisper failure goes to
    openai-whisper on the CPU with the CPU model; a CUDA OOM on 'cuda:N'
    falls back to the CPU model after releasing the VRAM."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        bc = self.bc
        self._saved = (bc._stt, bc._stt_device, bc._stt_model_name,
                       bc._stt_engine)
        bc._stt = None
        self.wlib = _FakeWhisperLib()
        self.torch = mock.Mock()
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.mem_get_info.return_value = (3 << 30, 4 << 30)
        self.plan = ("int8", False, "cuda:1: 3930 MB free / 4096 MB total")
        self.out = []

    def tearDown(self):
        bc = self.bc
        (bc._stt, bc._stt_device, bc._stt_model_name,
         bc._stt_engine) = self._saved

    def _load(self, fw_module, device="cuda:1"):
        bc = self.bc
        mods = {"faster_whisper": fw_module, "whisper": self.wlib,
                "torch": self.torch}
        patches = [
            mock.patch.dict(sys.modules, mods),
            mock.patch.object(bc, "_cuda_driver_first"),
            mock.patch.object(bc._cuda_preinit, "before_ctranslate2"),
            mock.patch.object(bc, "_register_cuda_dll_dirs"),
            mock.patch.object(bc, "_resolve_whisper_device",
                              return_value=device),
            mock.patch.object(bc, "_force_whisper_cpu_int8", False),
            mock.patch.object(bc, "_whisper_cuda_plan", return_value=self.plan),
            mock.patch.object(bc, "_log_cuda_load_order"),
            mock.patch.object(bc.time, "sleep"),
            mock.patch("builtins.print", side_effect=lambda *a, **k:
                       self.out.append(" ".join(str(x) for x in a))),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            bc._ensure_whisper()          # must not raise
        return bc

    @staticmethod
    def _fw(model_cls):
        fw = types.ModuleType("faster_whisper")
        fw.WhisperModel = model_cls
        return fw

    def test_non_import_error_goes_to_the_cpu_model_never_the_cuda_one(self):
        class _Broken:
            def __init__(self, *a, **k):
                raise RuntimeError("corrupt CTranslate2 cache")
        bc = self._load(self._fw(_Broken))
        self.assertEqual(self.wlib.calls, [(bc.WHISPER_MODEL_CPU, "cpu")])
        self.assertNotIn(bc.WHISPER_MODEL_CUDA,
                         [m for m, _d in self.wlib.calls])
        self.assertEqual(bc._stt_engine, "openai_whisper")
        self.assertEqual(bc._stt_device, "cpu")
        self.assertEqual(bc._stt_model_name, bc.WHISPER_MODEL_CPU)

    def test_cuda_n_oom_falls_back_to_cpu_after_releasing_vram(self):
        # faster-whisper not installed: Engine 2 tries the CUDA model on
        # 'cuda:1' (the plan cleared it), OOMs, and must degrade, not raise.
        bc = self._load(None)
        self.assertEqual(self.wlib.calls,
                         [(bc.WHISPER_MODEL_CUDA, "cuda:1"),
                          (bc.WHISPER_MODEL_CPU, "cpu")])
        self.assertEqual(bc._stt_device, "cpu")
        self.assertEqual(bc._stt_engine, "openai_whisper")
        self.assertEqual(bc._stt_model_name, bc.WHISPER_MODEL_CPU)
        self.torch.cuda.empty_cache.assert_called_once_with()
        # The half-built model was gone before the CPU model loaded.
        self.assertTrue(self.wlib.freed_before_cpu)
        said = " | ".join(self.out)
        self.assertIn("falling back to CPU model", said)
        self.assertIn("released the failed CUDA load", said)

    def test_engine_2_honours_a_cpu_vram_plan(self):
        self.plan = ("int8", True, "cuda:1 has 332 MB free; using CPU int8")
        bc = self._load(None)
        self.assertEqual(self.wlib.calls, [(bc.WHISPER_MODEL_CPU, "cpu")])
        self.torch.cuda.empty_cache.assert_not_called()

    def test_an_import_deadlock_retries_faster_whisper_not_openai(self):
        built = []

        class _Racy:
            def __init__(self, *a, **k):
                if not built:
                    built.append("deadlock")
                    raise importlib._bootstrap._DeadlockError(
                        "deadlock detected by _ModuleLock("
                        "'faster_whisper.feature_extractor')")
                built.append(k.get("device"))
                self.kw = k
        bc = self._load(self._fw(_Racy), device="cpu")
        self.assertEqual(self.wlib.calls, [])
        self.assertEqual(bc._stt_engine, "faster_whisper")
        self.assertIsInstance(bc._stt, _Racy)
        self.assertEqual(built, ["deadlock", "cpu"])
        self.assertIn("retrying faster-whisper once", " | ".join(self.out))

    def test_a_cuda_build_deadlock_is_retried_not_sent_to_the_cpu_model(self):
        built = []

        class _RacyCuda:
            def __init__(self, *a, **k):
                built.append(k.get("device"))
                if len(built) == 1:
                    raise importlib._bootstrap._DeadlockError("deadlock")
                self.kw = k
        bc = self._load(self._fw(_RacyCuda))
        self.assertEqual(built, ["cuda", "cuda"])
        self.assertEqual(bc._stt_device, "cuda:1")
        self.assertEqual(bc._stt_model_name, bc.WHISPER_MODEL_CUDA)
        self.assertEqual(self.wlib.calls, [])

    def test_two_deadlocks_end_on_the_cpu_model(self):
        class _Stuck:
            def __init__(self, *a, **k):
                raise importlib._bootstrap._DeadlockError("deadlock")
        bc = self._load(self._fw(_Stuck))
        self.assertEqual(self.wlib.calls, [(bc.WHISPER_MODEL_CPU, "cpu")])
        self.assertEqual(bc._stt_device, "cpu")


if __name__ == "__main__":
    unittest.main()
