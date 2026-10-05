"""core/cuda_preinit.py - the CUDA driver starts BEFORE ctranslate2 loads,
with no CUDA context (v2.0.180).

THE CRASH (v2.0.179, 2026-10-04 20:40): Windows runs DLL thread-detach code
in reverse initialisation order. 178 initialised the driver (via torch) long
before ctranslate2.dll (#156 vs #252) and survived 94 decode-thread exits;
179 read VRAM through NVML, so ctranslate2.dll came first (#200, driver #263)
and the first exiting decode thread aborted the process.

Pinned here with a fake driver library (no GPU, no NVIDIA driver):
  * before_ctranslate2() calls cuInit(0) once - and NOTHING that makes a
    context (cuCtx*, cuDevicePrimaryCtx*) - so the 3090 stays untouched;
  * it reports "late" when ctranslate2 was already loaded (order unfixable);
  * it skips (loads nothing) when CUDA_VISIBLE_DEVICES hides every GPU or
    under JARVIS_TEST_MODE=1, and never raises without a driver;
  * load_order() reads the module list like the investigation did: 179's
    order is flagged, 178's passes.

    python -m unittest tests.test_cuda_preinit
"""
from __future__ import annotations

import unittest

from core import cuda_preinit as cp


class FakeDriver:
    """Records every attribute the module touches on the driver library."""

    def __init__(self, rc=0):
        self.rc = rc
        self.touched: list = []
        self.cuinit_args: list = []

    def __getattr__(self, name):
        self.touched.append(name)
        if name == "cuInit":
            def _cuinit(flags):
                self.cuinit_args.append(flags)
                return self.rc
            return _cuinit
        raise AssertionError(f"the pre-init touched {name} - only cuInit "
                             f"is allowed (no context, ever)")


class _Base(unittest.TestCase):
    def setUp(self):
        self.drv = FakeDriver()
        self.loads = []
        self.ct2_loaded = [False]

        def _load():
            self.loads.append(1)
            return self.drv
        cp.set_hooks(loader=_load, ct2_loaded=lambda: self.ct2_loaded[0])
        self.addCleanup(cp.set_hooks, None, None)


class BeforeCtranslate2Tests(_Base):
    def test_cuinit_only_once_and_nothing_that_makes_a_context(self):
        st = cp.before_ctranslate2(env={})
        cp.before_ctranslate2(env={})
        self.assertTrue(st["ran"])
        self.assertTrue(st["ok"])
        self.assertFalse(st["late"])
        self.assertEqual(self.drv.cuinit_args, [0])
        self.assertEqual(self.drv.touched, ["cuInit"])
        self.assertEqual(len(self.loads), 1)
        self.assertIn("before ctranslate2", cp.status_line())
        self.assertIn("no context", cp.status_line())

    def test_late_when_ctranslate2_already_loaded(self):
        self.ct2_loaded[0] = True
        st = cp.before_ctranslate2(env={})
        self.assertTrue(st["late"])
        self.assertIn("WARNING", cp.status_line(st))
        self.assertIn("ct2-host", cp.status_line(st))

    def test_nonzero_cuinit_is_reported_not_raised(self):
        self.drv.rc = 100                     # CUDA_ERROR_NO_DEVICE
        st = cp.before_ctranslate2(env={})
        self.assertFalse(st["ok"])
        self.assertEqual(st["rc"], 100)
        self.assertIn("100", cp.status_line(st))

    def test_no_driver_is_reported_not_raised(self):
        def _missing():
            raise OSError("nvcuda.dll not found")
        cp.set_hooks(loader=_missing, ct2_loaded=lambda: False)
        st = cp.before_ctranslate2(env={})
        self.assertFalse(st["ok"])
        self.assertFalse(st["ran"])
        self.assertIn("no CUDA driver", st["detail"])

    def test_hidden_gpus_load_nothing(self):
        for vis in ("", "-1", "-1,0", " -1 "):
            cp.set_hooks(loader=lambda: self.loads.append(1) or self.drv,
                         ct2_loaded=lambda: False)
            st = cp.before_ctranslate2(env={"CUDA_VISIBLE_DEVICES": vis})
            self.assertFalse(st["ran"], vis)
            self.assertIn("hides every GPU", st["detail"])
        self.assertEqual(self.loads, [])

    def test_visible_gpu_list_still_starts_the_driver(self):
        st = cp.before_ctranslate2(env={"CUDA_VISIBLE_DEVICES": "1"})
        self.assertTrue(st["ran"])

    def test_test_mode_loads_nothing(self):
        st = cp.before_ctranslate2(env={"JARVIS_TEST_MODE": "1"})
        self.assertFalse(st["ran"])
        self.assertEqual(self.loads, [])

    def test_no_cuda_driver_flag_loads_nothing(self):
        st = cp.before_ctranslate2(env={"JARVIS_NO_CUDA_DRIVER": "1"})
        self.assertFalse(st["ran"])
        self.assertEqual(self.loads, [])

    def test_the_test_run_itself_never_loads_the_real_driver(self):
        # tests/__init__.py arms it for every entry path (unittest, runners).
        import os
        self.assertEqual(os.environ.get("JARVIS_NO_CUDA_DRIVER"), "1")
        init = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "__init__.py")
        with open(init, encoding="utf-8") as fh:
            self.assertIn('setdefault("JARVIS_NO_CUDA_DRIVER", "1")', fh.read())

    def test_never_raises_on_a_broken_library(self):
        class _Broken:
            def __getattr__(self, name):
                raise RuntimeError("weird binding")
        cp.set_hooks(loader=lambda: _Broken(), ct2_loaded=lambda: False)
        st = cp.before_ctranslate2(env={})
        self.assertFalse(st["ok"])


class StdlibOnlyTests(unittest.TestCase):
    """Neither module may pull in ctranslate2, torch or any CUDA library -
    importing one would load a DLL ahead of the driver and undo the fix."""

    def _imports(self, rel):
        import ast
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, *rel.split("/")), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
        return mods

    def test_cuda_preinit_is_stdlib_only(self):
        self.assertLessEqual(self._imports("core/cuda_preinit.py"),
                             {"__future__", "ctypes", "os", "sys", "threading"})

    def test_ct2_host_is_stdlib_only(self):
        self.assertLessEqual(self._imports("core/ct2_host.py"),
                             {"__future__", "queue", "sys", "threading",
                              "time", "traceback"})


class LoadOrderTests(unittest.TestCase):
    @staticmethod
    def _names(order):
        out = [f"m{i}.dll" for i in range(1, 300)]
        for name, pos in order.items():
            out[pos - 1] = name
        return out

    def test_the_179_order_is_flagged(self):
        # From the 179 crash dump: ctranslate2 #200, nvcuda #214, nvcuda64 #263.
        o = cp.load_order(self._names({"ctranslate2.dll": 200,
                                       "nvcuda.dll": 214,
                                       "nvcuda64.dll": 263}))
        self.assertEqual((o["ctranslate2"], o["nvcuda"], o["nvcuda64"]),
                         (200, 214, 263))
        self.assertFalse(o["driver_first"])
        self.assertIn("WARNING", cp.load_order_line(o))

    def test_the_178_order_passes(self):
        # From the live 178 process: nvcuda64 #156, ctranslate2 #252.
        o = cp.load_order(self._names({"nvcuda.dll": 150, "nvcuda64.dll": 156,
                                       "ctranslate2.dll": 252}))
        self.assertTrue(o["driver_first"])
        self.assertIn("OK", cp.load_order_line(o))

    def test_ctranslate2_absent(self):
        o = cp.load_order(["python314.dll", "nvcuda.dll"])
        self.assertIsNone(o["driver_first"])
        self.assertIn("not loaded", cp.load_order_line(o))

    def test_unreadable_list(self):
        self.assertIsNone(cp.load_order([])["driver_first"])


if __name__ == "__main__":
    unittest.main()
