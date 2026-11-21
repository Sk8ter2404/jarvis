"""core/fw_import.py - one faster_whisper import at a time (the 11-20 boot).

On 11-20 the boot Whisper thread (parent package first) and the Smart Turn
warmer (core/endpointing.whisper_features: the feature_extractor SUBMODULE
first) imported faster_whisper at once; the two per-module import locks were
taken in opposite orders, Python raised _DeadlockError in the Whisper thread
and Whisper ran on the CPU all session. These tests use a real stub package
on disk whose __init__ blocks on an Event, so the race is the real one.
"""
from __future__ import annotations

import builtins
import importlib
import importlib._bootstrap as _bootstrap
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from core import fw_import

_INIT = '''
import threading
import _fw_race_gate as _g
_g.inits.append(threading.current_thread().name)
_g.entered.set()
_g.gate.wait(10)
from .feature_extractor import FeatureExtractor


class WhisperModel:
    def __init__(self, *a, **k):
        self.a, self.kw = a, k
'''

_FE = '''
class FeatureExtractor:
    def __init__(self, **kw):
        self.kw = kw

    def __call__(self, x, padding=160):
        return ("features", padding)
'''


class StubFasterWhisper:
    """A real faster_whisper package on disk (sys.path[0]) whose __init__
    blocks on ``gate`` after setting ``entered``. Restores sys.modules and
    sys.path on close."""

    def __init__(self, test: unittest.TestCase):
        self.dir = tempfile.mkdtemp(prefix="fw_stub_")
        pkg = os.path.join(self.dir, "faster_whisper")
        os.makedirs(pkg)
        with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as fh:
            fh.write(_INIT)
        with open(os.path.join(pkg, "feature_extractor.py"), "w",
                  encoding="utf-8") as fh:
            fh.write(_FE)
        self.ctl = types.ModuleType("_fw_race_gate")
        self.ctl.gate = threading.Event()
        self.ctl.entered = threading.Event()
        self.ctl.inits = []
        saved = {k: v for k, v in sys.modules.items()
                 if k == "faster_whisper" or k.startswith("faster_whisper.")}
        for k in saved:
            del sys.modules[k]
        sys.modules["_fw_race_gate"] = self.ctl
        sys.path.insert(0, self.dir)
        importlib.invalidate_caches()

        def _restore():
            for k in [k for k in sys.modules if k == "faster_whisper"
                      or k.startswith("faster_whisper.")]:
                del sys.modules[k]
            sys.modules.update(saved)
            sys.modules.pop("_fw_race_gate", None)
            try:
                sys.path.remove(self.dir)
            except ValueError:
                pass
            importlib.invalidate_caches()
            shutil.rmtree(self.dir, ignore_errors=True)
        test.addCleanup(_restore)


def _run(target, name, box):
    def body():
        try:
            box[name] = ("ok", target())
        except BaseException as e:      # noqa: BLE001 - recorded for the test
            box[name] = ("err", e)
    t = threading.Thread(target=body, name=name, daemon=True)
    t.start()
    return t


def _child_first():
    from faster_whisper.feature_extractor import FeatureExtractor
    return FeatureExtractor


class RaceTests(unittest.TestCase):
    def setUp(self):
        self.stub = StubFasterWhisper(self)

    def _race(self, a_target, b_target):
        box = {}
        ta = _run(a_target, "whisper", box)
        self.assertTrue(self.stub.ctl.entered.wait(5))
        tb = _run(b_target, "warmer", box)
        time.sleep(0.4)                   # the warmer is now blocked
        self.stub.ctl.gate.set()
        ta.join(15)
        tb.join(15)
        self.assertFalse(ta.is_alive() or tb.is_alive(), "import hung")
        return box

    def test_two_helper_importers_never_deadlock(self):
        # Parent-first (Whisper load) against submodule-wanted (Smart Turn):
        # both through the helper, one at a time, one package init.
        with mock.patch.object(fw_import, "_DEADLOCK_RETRY_WAIT_S", 0.0), \
                mock.patch("builtins.print") as pr:
            box = self._race(
                lambda: fw_import.import_faster_whisper().WhisperModel,
                lambda: fw_import.import_faster_whisper(
                    "feature_extractor").FeatureExtractor)
        self.assertEqual(box["whisper"][0], "ok", box["whisper"])
        self.assertEqual(box["warmer"][0], "ok", box["warmer"])
        self.assertEqual(box["whisper"][1].__name__, "WhisperModel")
        self.assertEqual(box["warmer"][1].__name__, "FeatureExtractor")
        self.assertEqual(self.stub.ctl.inits, ["whisper"])
        self.assertNotIn("deadlock", " ".join(str(c) for c in pr.call_args_list))

    def test_a_child_first_importer_outside_the_lock_costs_one_retry(self):
        # The 11-20 race itself: a submodule-first importer that does not
        # take the lock. Python raises _DeadlockError in the helper's thread;
        # the helper retries once and gets the package.
        with mock.patch.object(fw_import, "_DEADLOCK_RETRY_WAIT_S", 0.3), \
                mock.patch("builtins.print") as pr:
            box = self._race(
                lambda: fw_import.import_faster_whisper().WhisperModel,
                _child_first)
        self.assertEqual(box["whisper"][0], "ok", box["whisper"])
        self.assertEqual(box["warmer"][0], "ok", box["warmer"])
        said = " ".join(str(c) for c in pr.call_args_list)
        self.assertIn("import deadlock", said)
        self.assertIn("retrying once", said)


class RetryTests(unittest.TestCase):
    def _flaky_import(self, errors):
        real = builtins.__import__
        calls = []

        def imp(name, *a, **k):
            if name == "faster_whisper":
                calls.append(name)
                if errors:
                    raise errors.pop(0)
                return fake
            return real(name, *a, **k)
        fake = types.ModuleType("faster_whisper")
        return imp, calls, fake

    def test_one_deadlock_is_retried(self):
        imp, calls, fake = self._flaky_import(
            [_bootstrap._DeadlockError("deadlock detected by _ModuleLock")])
        with mock.patch.object(builtins, "__import__", imp), \
                mock.patch.object(fw_import, "_DEADLOCK_RETRY_WAIT_S", 0.0), \
                mock.patch("builtins.print"):
            self.assertIs(fw_import.import_faster_whisper(), fake)
        self.assertEqual(len(calls), 2)

    def test_a_second_deadlock_propagates(self):
        imp, calls, _fake = self._flaky_import(
            [_bootstrap._DeadlockError("a"), _bootstrap._DeadlockError("b")])
        with mock.patch.object(builtins, "__import__", imp), \
                mock.patch.object(fw_import, "_DEADLOCK_RETRY_WAIT_S", 0.0), \
                mock.patch("builtins.print"):
            with self.assertRaises(RuntimeError) as cm:
                fw_import.import_faster_whisper()
        self.assertTrue(fw_import.is_import_deadlock(cm.exception))
        self.assertEqual(len(calls), 2)

    def test_import_error_and_other_errors_are_not_retried(self):
        for err in (ModuleNotFoundError("No module named 'faster_whisper'"),
                    OSError("ctranslate2.dll failed")):
            imp, calls, _fake = self._flaky_import([err])
            with mock.patch.object(builtins, "__import__", imp):
                with self.assertRaises(type(err)):
                    fw_import.import_faster_whisper()
            self.assertEqual(len(calls), 1)

    def test_missing_package_is_an_import_error(self):
        with mock.patch.dict(sys.modules, {"faster_whisper": None}):
            with self.assertRaises(ImportError):
                fw_import.import_faster_whisper()

    def test_deadlock_is_matched_by_class_name(self):
        self.assertTrue(fw_import.is_import_deadlock(
            _bootstrap._DeadlockError("x")))
        self.assertFalse(fw_import.is_import_deadlock(RuntimeError("deadlock")))
        self.assertFalse(fw_import.is_import_deadlock(None))

    def test_the_driver_starts_before_the_import(self):
        events = []
        real = builtins.__import__

        def imp(name, *a, **k):
            if name == "faster_whisper":
                events.append("import")
                return types.ModuleType("faster_whisper")
            return real(name, *a, **k)
        from core import cuda_preinit
        with mock.patch.object(cuda_preinit, "before_ctranslate2",
                               side_effect=lambda: events.append("driver")), \
                mock.patch.object(builtins, "__import__", imp):
            fw_import.import_faster_whisper()
        self.assertEqual(events, ["driver", "import"])


if __name__ == "__main__":
    unittest.main()
