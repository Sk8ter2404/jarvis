"""No thread that exits may own CTranslate2 CUDA state; the CUDA driver starts
before ctranslate2 loads (v2.0.180).

THE CRASH (v2.0.179, 2026-10-04 20:40:22-55; crash dumps
pythonw.exe.158452.dmp / pythonw.exe(1).158452.dmp): the self-diagnostic boot
sweep ran _probe_stt on its throwaway ``probe-stt`` thread; the probe decoded
1 s of tone with the cached faster-whisper model on cuda:1 and returned; as
the thread EXITED the loader ran ctranslate2.dll's thread-local destructors,
``ctranslate2::set_device_index`` threw std::runtime_error ("No CUDA context
is current to the calling thread"), std::terminate -> abort -> "Fatal Python
error: Aborted". The 31 s "stalled camera preview" logged just before was
Windows Error Reporting suspending every thread while it wrote the 833 MB dump.

Pinned here (light tier - no GPU, no ctranslate2, no monolith import):
  * THE PATH ITSELF: _probe_stt run by the real _run_with_timeout (its real
    ``probe-stt`` thread) on a cached CUDA model leaves no CTranslate2 state
    on that thread - the decode and the generator drain run on ct2-host;
  * the standby lyric detector's opt-in CUDA model does the same;
  * every function that imports faster_whisper / ctranslate2 starts the CUDA
    driver first (core/cuda_preinit), and none imports them at module level;
  * the monolith's faster-whisper decodes all go through _ct2_host.decode,
    a CUDA model is only ever built through _ct2_host.run_for, and the boot
    starts the driver before the preflight.

    python -m unittest tests.test_ct2_thread_exit
"""
from __future__ import annotations

import ast
import os
import sys
import threading
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeCT2Whisper:
    """faster-whisper's WhisperModel surface. Records the thread of every
    call that would leave CTranslate2 CUDA thread state behind on its caller
    (transcribe() and the drain of its lazy generator)."""

    def __init__(self):
        self.state_threads: list = []

    def _touch(self):
        self.state_threads.append(threading.current_thread())

    def transcribe(self, audio, **kw):
        self._touch()

        def _gen():
            self._touch()
            yield types.SimpleNamespace(text="", no_speech_prob=0.9,
                                        avg_logprob=-1.0)
        return _gen(), types.SimpleNamespace(no_speech_prob=0.9)


FakeCT2Whisper.__name__ = "WhisperModel"     # the probe keys on the class name


class SelfDiagnosticProbeTests(unittest.TestCase):
    """The exact v2.0.179 crash path."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("self_diagnostic")

    def _bc(self, model, device):
        return types.SimpleNamespace(_stt=model, _stt_model_name="large-v3-turbo",
                                     _stt_device=device,
                                     _stt_lock=threading.RLock())

    def test_probe_stt_thread_owns_no_ctranslate2_state(self):
        model = FakeCT2Whisper()
        probe_threads = []
        real_probe = self.mod._probe_stt

        def _probe():
            probe_threads.append(threading.current_thread())
            return real_probe()
        with mock.patch.object(self.mod, "_bc",
                               return_value=self._bc(model, "cuda:1")):
            # The real runner: its real throwaway "probe-stt" thread.
            r = self.mod._run_with_timeout(_probe, 10.0, name="stt")
        self.assertTrue(r["ok"], r)
        self.assertEqual([t.name for t in probe_threads], ["probe-stt"])
        probe = probe_threads[0]
        probe.join(5.0)
        self.assertFalse(probe.is_alive())          # it EXITED, as at 20:40:22
        self.assertEqual(len(model.state_threads), 2)
        for t in model.state_threads:
            self.assertIsNot(t, probe)
            self.assertEqual(t.name, "ct2-host")
            self.assertTrue(t.is_alive())           # the owner never exits

    def test_probe_on_a_cpu_model_stays_on_its_own_thread(self):
        # CTranslate2 on the CPU has no CUDA thread state: unchanged.
        model = FakeCT2Whisper()
        with mock.patch.object(self.mod, "_bc",
                               return_value=self._bc(model, "cpu")):
            r = self.mod._run_with_timeout(self.mod._probe_stt, 10.0,
                                           name="stt")
        self.assertTrue(r["ok"], r)
        self.assertEqual({t.name for t in model.state_threads}, {"probe-stt"})


class StandbyDetectorTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("standby_audio_detect")
        self.addCleanup(lambda: self.mod._whisper_model.__setitem__(0, None))
        self.addCleanup(lambda: self.mod._whisper_cuda_model.__setitem__(0, None))
        self._saved_cfg = dict(self.mod._loop_cfg)
        self.addCleanup(lambda: (self.mod._loop_cfg.clear(),
                                 self.mod._loop_cfg.update(self._saved_cfg)))

    def test_opt_in_cuda_model_is_built_and_decoded_on_the_host(self):
        built_on = []

        class _FWM(FakeCT2Whisper):
            def __init__(self, name, **kw):
                super().__init__()
                self.kw = kw
                built_on.append(threading.current_thread())
        fw = types.ModuleType("faster_whisper")
        fw.WhisperModel = _FWM
        self.mod._whisper_model[0] = None
        self.mod._loop_cfg["prefer_gpu"] = True
        with mock.patch.dict(sys.modules, {"faster_whisper": fw}), \
             mock.patch.object(self.mod, "_gpu_index", return_value=1), \
             mock.patch.object(self.mod, "_cuda_free_vram_mb", return_value=3000):
            box = {}

            def _loop_thread():
                box["text"] = self.mod._transcribe_buffer(
                    __import__("numpy").zeros(1600, dtype="float32"), 16000)
            t = threading.Thread(target=_loop_thread, name="standby-loop")
            t.start()
            t.join(10.0)
        model = self.mod._whisper_model[0]
        self.assertIsInstance(model, _FWM)
        self.assertEqual(model.kw.get("device"), "cuda")
        self.assertEqual([x.name for x in built_on], ["ct2-host"])
        self.assertEqual(len(model.state_threads), 2)
        self.assertEqual({x.name for x in model.state_threads}, {"ct2-host"})


# ─── structural ratchets ──────────────────────────────────────────────────
_CT2_MODULES = ("faster_whisper", "ctranslate2")
_DRIVER_FIRST = ("before_ctranslate2", "_cuda_driver_first")
_SCAN_DIRS = ("core", "skills", "audio", "tools", "hud", "adapters")


def _production_files():
    out = [os.path.join(_ROOT, "bobert_companion.py")]
    for d in _SCAN_DIRS:
        base = os.path.join(_ROOT, d)
        for dirpath, dirnames, files in os.walk(base):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for f in files:
                if f.endswith(".py"):
                    out.append(os.path.join(dirpath, f))
    return out


def _imports_ct2(node) -> bool:
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] in _CT2_MODULES for a in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] in _CT2_MODULES and not node.level
    return False


def _call_name(call) -> str:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _parse(path):
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def _func(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    raise AssertionError(f"{name} not found")


class DriverFirstRatchetTests(unittest.TestCase):
    def test_every_ctranslate2_import_starts_the_driver_first(self):
        offenders, seen = [], 0
        for path in _production_files():
            tree = _parse(path)
            rel = os.path.relpath(path, _ROOT)
            for node in tree.body:
                if _imports_ct2(node):
                    offenders.append(f"{rel}:{node.lineno} module-level import")
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # Only this function's own statements (nested defs are
                # checked on their own).
                own = []
                stack = list(fn.body)
                while stack:
                    n = stack.pop()
                    own.append(n)
                    for c in ast.iter_child_nodes(n):
                        if not isinstance(c, (ast.FunctionDef,
                                              ast.AsyncFunctionDef,
                                              ast.Lambda)):
                            stack.append(c)
                firsts = [n.lineno for n in own if isinstance(n, ast.Call)
                          and _call_name(n) in _DRIVER_FIRST]
                for n in own:
                    if _imports_ct2(n):
                        seen += 1
                        if not any(ln < n.lineno for ln in firsts):
                            offenders.append(f"{rel}:{n.lineno} in {fn.name}()")
        # Blindness floor: the scan must have seen the known import sites
        # (monolith x3, self-diagnostic, standby detector, endpointing).
        self.assertGreaterEqual(seen, 6)
        self.assertEqual(offenders, [], "ctranslate2 imported before the CUDA "
                         "driver was started (core/cuda_preinit): " +
                         "; ".join(offenders))


class MonolithShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = _parse(os.path.join(_ROOT, "bobert_companion.py"))

    def test_faster_whisper_decodes_all_go_through_the_host(self):
        fn = _func(self.tree, "_transcribe_impl")
        fw_branch = None
        for n in ast.walk(fn):
            if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                    and isinstance(n.test.left, ast.Name)
                    and n.test.left.id == "_stt_engine"
                    and any(isinstance(c, ast.Constant) and c.value == "faster_whisper"
                            for c in n.test.comparators)):
                fw_branch = n
                break
        self.assertIsNotNone(fw_branch)
        direct, hosted = [], 0
        for stmt in fw_branch.body:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Call) and _call_name(n) == "transcribe":
                    direct.append(n.lineno)
                if (isinstance(n, ast.Call) and _call_name(n) == "decode"
                        and isinstance(n.func, ast.Attribute)
                        and isinstance(n.func.value, ast.Name)
                        and n.func.value.id == "_ct2_host"):
                    hosted += 1
        self.assertEqual(direct, [], "a faster-whisper decode bypasses _ct2_host")
        self.assertEqual(hosted, 3)     # VAD decode, no-VAD retry, echo re-decode

    def test_a_cuda_model_is_only_built_through_run_for(self):
        fn = _func(self.tree, "_ensure_whisper_locked")
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and _call_name(n) == "_FWM":
                dev = [k.value for k in n.keywords if k.arg == "device"]
                self.assertEqual(len(dev), 1)
                self.assertIsInstance(dev[0], ast.Constant,
                                      f"line {n.lineno}: a non-CPU _FWM build "
                                      "outside _ct2_host.run_for")
                self.assertEqual(dev[0].value, "cpu")
        runs = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and _call_name(n) == "run_for"]
        self.assertEqual(len(runs), 1)

    def test_boot_starts_the_driver_before_the_preflight(self):
        hit = 0
        for fn in ast.walk(self.tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
            pre = [n.lineno for n in calls if _call_name(n) == "_startup_preflight"
                   and isinstance(n.func, ast.Name)]
            if not pre:
                continue
            hit += 1
            first = [n.lineno for n in calls if _call_name(n) == "_cuda_driver_first"]
            self.assertTrue(first and min(first) < min(pre),
                            f"{fn.name}: _cuda_driver_first() must run before "
                            "_startup_preflight()")
        self.assertGreaterEqual(hit, 1)

    def test_a_dropped_cuda_model_is_retired_to_the_host(self):
        fn = _func(self.tree, "_transcribe_impl")
        retires = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                   and _call_name(n) == "retire"]
        self.assertEqual(len(retires), 1)


if __name__ == "__main__":
    unittest.main()
