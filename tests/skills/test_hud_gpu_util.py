"""GPU utilization parse + threshold tests for the two PyQt6 reactor HUDs that
render a GPU metric: ``hud/jarvis_unified_hud.py`` (the auto-launched HUD) and
``hud/arc_reactor_status_hud.py`` (the secondary arc-reactor ring).

WHY THIS EXISTS
  The GPU surface flipped from *temperature* to *utilization %* (the owner's
  rig runs cool; load is the interesting signal). That change couples ONE
  producer token written by ``skills/system_pulse`` into ``hud_state.json``
  (``pulse_strip``) with the TWO HUD fallback parsers here: the producer now
  emits ``GPU 54%`` (was ``GPU 54C``), and both HUDs must read that ``%`` token
  into a utilization field, colour it on a flat 0-100 scale, and use the new
  90% crit threshold — never feed a 0-100 load value into a 30-95 °C band.

  These two HUDs have no other test harness (they import PyQt6 and subclass
  QWidget/QGraphicsScene). PyQt6 is NOT on the CI runner, and even where it is
  a renderer test must never spin a real Qt loop. The GPU *reader* under test
  (``_read_gpu_util``) touches no Qt at all — only ``shutil`` / ``subprocess`` /
  ``time`` / ``_read_json`` module globals — so we load each source with PyQt6
  genuinely ABSENT (the module's own ``except ImportError`` stub path makes the
  Qt names harmless placeholders and ``_HAS_PYQT6`` False) and call the reader
  as an unbound method against a tiny stand-in ``self``. No widget is built, no
  display is touched, and the suite runs identically on the headless runner.

ISOLATION
  • PyQt6 imports are blocked for the duration of each test (a fake ``__import__``
    raises ImportError for the ``PyQt6`` package), restored on cleanup. Any
    previously-cached PyQt6 submodules are hidden during the load and restored
    after, so a dev box that *has* PyQt6 still exercises the headless path.
  • Each source is loaded from its file path under a synthetic module name and
    dropped on cleanup → pristine module globals (thresholds, file paths).
  • ``HUD_STATE_FILE`` is redirected to a per-test temp file, so no real project
    file is read. ``nvidia-smi`` is forced absent (``shutil.which`` → None) so the
    deterministic pulse_strip *fallback* path is what we assert — the live
    nvidia-smi branch would otherwise read this dev box's real GPU.

stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


_HUD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hud",
)


class _InlineThread:
    """A ``threading.Thread`` stand-in whose ``start()`` runs the target
    synchronously on the calling thread, so a test can drive the async GPU
    sampler to completion deterministically (no real worker thread to join)."""

    def __init__(self, *, target=None, daemon=None, **_kw):
        self._target = target

    def start(self):
        if self._target is not None:
            self._target()


class _NoStartThread:
    """A ``threading.Thread`` stand-in whose ``start()`` is a no-op — proves the
    blocking sampler is *offloaded* (never run inline): the worker body, and
    thus subprocess.run, is not reached on the calling/paint thread."""

    def __init__(self, *, target=None, daemon=None, **_kw):
        self._target = target

    def start(self):
        pass


def _load_hud_no_pyqt(testcase, filename, mod_name):
    """Load a HUD source from hud/<filename> with PyQt6 import blocked, so the
    module takes its graceful-degrade path (``_HAS_PYQT6`` False, Qt names
    stubbed). Restores sys.modules + the real importer on cleanup."""
    path = os.path.join(_HUD_DIR, filename)
    real_import = __import__

    def _imp(name, *a, **k):
        if name.split(".")[0] == "PyQt6":
            raise ImportError(f"[test] PyQt6 blocked: {name}")
        return real_import(name, *a, **k)

    hidden = {n: sys.modules.pop(n)
              for n in list(sys.modules) if n.split(".")[0] == "PyQt6"}
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module

    def restore():
        sys.modules.pop(mod_name, None)
        sys.modules.update(hidden)

    testcase.addCleanup(restore)
    with mock.patch("builtins.__import__", side_effect=_imp):
        spec.loader.exec_module(module)
    testcase.assertFalse(module._HAS_PYQT6,
                         "PyQt6 should be blocked → headless degrade path")
    return module


class _HudParseBase(unittest.TestCase):
    # Subclasses set these.
    FILENAME = ""
    MOD_NAME = ""
    SCENE_CLS = ""
    UTIL_FIELD = ""        # name of the cached utilization attr on the scene

    def setUp(self):
        self.mod = _load_hud_no_pyqt(self, self.FILENAME, self.MOD_NAME)
        self.cls = getattr(self.mod, self.SCENE_CLS)
        self.tmp = tempfile.mkdtemp(prefix="hud_gpu_util_test_")
        self.addCleanup(self._cleanup_tmp)
        self.hud_state = os.path.join(self.tmp, "hud_state.json")
        self.mod.HUD_STATE_FILE = self.hud_state

    def _cleanup_tmp(self):
        for fn in os.listdir(self.tmp):
            try:
                os.unlink(os.path.join(self.tmp, fn))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def _write_strip(self, strip):
        with open(self.hud_state, "w", encoding="utf-8") as f:
            json.dump({"pulse_strip": strip}, f)

    def _fake_self(self):
        """A minimal stand-in for the scene. ``_read_gpu_util`` is now async —
        on a cache miss it offloads the blocking sample to a worker thread — so
        the stand-in must carry the ``_gpu_sampling`` guard plus the worker /
        sampler methods (bound to this object) the read path delegates to."""
        s = types.SimpleNamespace(**{self.UTIL_FIELD: None,
                                     "_gpu_cached_at": 0.0,
                                     "_gpu_sampling": False})
        s._gpu_sample_worker = types.MethodType(self.cls._gpu_sample_worker, s)
        s._sample_gpu_util = types.MethodType(self.cls._sample_gpu_util, s)
        return s

    def _read(self, fresh_self=None):
        """Drive _read_gpu_util with nvidia-smi forced absent (so the pulse_strip
        fallback is exercised deterministically, not this box's real GPU) and the
        spawned sampler thread run INLINE, then return the value the worker
        publishes into the cached utilization field — the value the paint loop
        reads. Exercises the real read -> spawn -> sample chain end to end."""
        s = fresh_self if fresh_self is not None else self._fake_self()
        with mock.patch.object(self.mod.shutil, "which", return_value=None), \
                mock.patch.object(self.mod.threading, "Thread", _InlineThread), \
                mock.patch.object(self.mod.time, "time", return_value=10_000.0):
            self.cls._read_gpu_util(s)
        return getattr(s, self.UTIL_FIELD)

    def test_paint_path_never_calls_nvidia_smi_directly(self):
        # The core anti-freeze guarantee: a cache-miss read must offload the
        # blocking nvidia-smi spawn to a background thread, NOT run it inline on
        # the calling (GUI/paint) thread. With Thread.start a no-op the worker
        # never runs, so subprocess.run must not be touched by _read_gpu_util.
        s = self._fake_self()
        with mock.patch.object(self.mod.time, "time", return_value=10_000.0), \
                mock.patch.object(self.mod.threading, "Thread",
                                  side_effect=_NoStartThread) as thread, \
                mock.patch.object(self.mod.shutil, "which",
                                  return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(self.mod.subprocess, "run") as run:
            ret = self.cls._read_gpu_util(s)
        run.assert_not_called()          # never blocks the caller
        thread.assert_called_once()      # sampling was offloaded to a thread
        self.assertIsNone(ret)           # returns the (empty) cached value

    # ── the producer/consumer token contract ────────────────────────────────
    def test_parses_percent_token_from_pulse_strip(self):
        self._write_strip("CPU 12% GPU 49% RAM 30%")
        self.assertEqual(self._read(), 49.0)

    def test_percent_sign_terminates_the_number(self):
        # The "%" must stop the digit-scan exactly where "C" used to — no stray
        # trailing characters folded into the value.
        self._write_strip("GPU 7%")
        self.assertEqual(self._read(), 7.0)

    def test_full_load_token(self):
        self._write_strip("GPU 100%")
        self.assertEqual(self._read(), 100.0)

    def test_legacy_temp_token_still_parses_as_number(self):
        # If a temp-only host emits the fallback "GPU 54C" token, the digit-scan
        # still yields the number (the value is displayed either way); the point
        # of the lockstep change is that the PRODUCER prefers "%" so this is rare.
        self._write_strip("GPU 54C")
        self.assertEqual(self._read(), 54.0)

    def test_no_gpu_token_returns_none(self):
        self._write_strip("CPU 12% RAM 30%")
        self.assertIsNone(self._read())

    def test_missing_state_file_returns_none(self):
        # No hud_state.json at all → None (and no exception).
        self.assertIsNone(self._read())

    def test_cache_returns_prev_within_window(self):
        # Inside GPU_CACHE_SECONDS the cached value is returned without re-reading
        # (which() must not even be consulted).
        s = types.SimpleNamespace(**{self.UTIL_FIELD: 63.0,
                                     "_gpu_cached_at": 9_999.5})
        with mock.patch.object(self.mod.time, "time", return_value=10_000.0), \
                mock.patch.object(self.mod.shutil, "which") as which:
            self.assertEqual(self.cls._read_gpu_util(s), 63.0)
        which.assert_not_called()

    # ── colour-scale + threshold contract ───────────────────────────────────
    def test_gpu_cache_is_short_for_fast_utilization(self):
        # Utilization swings fast; the cache must be ≤ ~1 s (was 4 s for temp).
        self.assertLessEqual(self.mod.GPU_CACHE_SECONDS, 1.0)


class UnifiedHudGpuUtilTests(_HudParseBase):
    FILENAME = "jarvis_unified_hud.py"
    MOD_NAME = "_ju_hud_under_test"
    SCENE_CLS = "UnifiedHud"
    UTIL_FIELD = "gpu_util"

    def test_thresholds_are_utilization_percent(self):
        # WARN 70 / CRIT 90 on a 0-100 load scale (was 70/82 °C). A 0-100 value
        # must never hit a 30-95 temp band — crit at 90 is the guard.
        self.assertEqual((self.mod.GPU_WARN, self.mod.GPU_CRIT), (70.0, 90.0))


class ArcReactorGpuUtilTests(_HudParseBase):
    FILENAME = "arc_reactor_status_hud.py"
    MOD_NAME = "_arc_hud_under_test"
    SCENE_CLS = "ArcReactorStatusScene"
    UTIL_FIELD = "gpu_util_pct"

    def test_thresholds_are_utilization_percent(self):
        self.assertEqual((self.mod.GPU_WARN_PCT, self.mod.GPU_CRIT_PCT),
                         (70.0, 90.0))


def load_tests(loader, standard_tests, pattern):  # noqa: D401 - unittest hook
    """Exclude the abstract ``_HudParseBase`` from collection; keep both concrete
    subclasses. The base defines the shared ``test_*`` methods but carries empty
    FILENAME/SCENE_CLS config, so running it directly would fail in setUp — drop
    any test whose class is exactly the base."""
    suite = unittest.TestSuite()
    for tests in standard_tests:
        for t in tests:
            if type(t) is _HudParseBase:
                continue
            suite.addTest(t)
    return suite


if __name__ == "__main__":
    unittest.main()
