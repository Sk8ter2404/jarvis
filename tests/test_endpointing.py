"""core/endpointing.py — the Silero speech tail behind [turn-timing] tail_ms
(speed plan R1, 2026-10-01).

Light tier: a FAKE onnxruntime session (CI never installs onnxruntime) whose
per-window probability is "speech" when the window is loud, so the clip's real
end of speech is known to the sample. The real-model test at the bottom runs
only where onnxruntime and faster-whisper's bundled model exist.

Run: python tools/run_tests.py test_endpointing
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

from core import endpointing as ep

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = ep.SAMPLE_RATE
W = ep.WINDOW


class _FakeSession:
    """Speech probability 0.9 for a window whose |mean| of the LAST 512
    samples (the window itself, after the 64-sample context) is above 0.05,
    else 0.02. Records every batch it was asked to score."""

    def __init__(self, fail=None, bad_len=False, nan=False, gate=None):
        self.batches = []
        self.fail = fail
        self.bad_len = bad_len
        self.nan = nan
        self.gate = gate
        self.active = 0
        self.max_active = 0
        self._mu = threading.Lock()

    def run(self, _names, feeds):
        with self._mu:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.gate is not None:
                self.gate.wait(2.0)
            if self.fail is not None:
                raise self.fail
            x = feeds["input"]
            assert x.shape[1] == ep.CONTEXT + W, x.shape
            assert feeds["h"].shape == (1, 1, 128)
            self.batches.append(x.copy())
            loud = np.abs(x[:, ep.CONTEXT:]).mean(axis=1) > 0.05
            probs = np.where(loud, 0.9, 0.02).astype(np.float32)
            if self.bad_len:
                probs = probs[:-1]
            if self.nan:
                probs[0] = np.nan
            return [probs.reshape(-1, 1), feeds["h"], feeds["c"]]
        finally:
            with self._mu:
                self.active -= 1


def _vad(session=None, path="fake.onnx"):
    sess = session if session is not None else _FakeSession()
    calls = []

    def factory(p):
        calls.append(p)
        return sess

    v = ep.SileroVad(model_path=path, session_factory=factory)
    return v, sess, calls


def _clip(speech_s, tail_s, lead_s=0.5):
    """lead silence, `speech_s` of loud signal, `tail_s` of silence."""
    lead = np.zeros(int(lead_s * SR), dtype=np.float32)
    speech = np.full(int(speech_s * SR), 0.3, dtype=np.float32)
    tail = np.zeros(int(tail_s * SR), dtype=np.float32)
    return np.concatenate([lead, speech, tail])


class SpeechTailTests(unittest.TestCase):
    def test_tail_is_clip_end_minus_the_last_speech_window(self):
        v, _, _ = _vad()
        # 1.344 s of trailing silence: whole windows from the END are
        # silent up to the speech, so the tail is exact to one window.
        got = v.speech_tail_ms(_clip(1.5, 1.344))
        self.assertIsNotNone(got)
        self.assertLessEqual(abs(got - 1344), 32, got)

    def test_window_aligned_to_the_clip_end(self):
        # Speech ends exactly on a window boundary counted from the end:
        # the result is exact.
        v, _, _ = _vad()
        tail = np.zeros(10 * W, dtype=np.float32)
        speech = np.full(7 * W + 100, 0.3, dtype=np.float32)
        got = v.speech_tail_ms(np.concatenate([speech, tail]))
        self.assertEqual(got, int(round(10 * W * 1000 / SR)))

    def test_no_trailing_silence_is_zero(self):
        v, _, _ = _vad()
        self.assertEqual(v.speech_tail_ms(_clip(2.0, 0.0)), 0)

    def test_scans_only_the_last_four_seconds(self):
        v, sess, _ = _vad()
        self.assertIsNotNone(v.speech_tail_ms(_clip(8.0, 1.0)))
        self.assertEqual(len(sess.batches), 1)
        self.assertEqual(sess.batches[0].shape[0],
                         int(ep.TAIL_SCAN_S * SR) // W)

    def test_context_carries_the_samples_before_the_scan(self):
        v, sess, _ = _vad()
        a = np.arange(int(6 * SR), dtype=np.float32) / 1e6
        v.speech_tail_ms(a)
        b = sess.batches[0]
        start = len(a) - b.shape[0] * W
        np.testing.assert_array_equal(b[0, :ep.CONTEXT],
                                      a[start - ep.CONTEXT:start])
        np.testing.assert_array_equal(b[1, :ep.CONTEXT],
                                      a[start + W - ep.CONTEXT:start + W])

    def test_speech_older_than_the_scan_falls_back_to_the_whole_clip(self):
        # Noise held the RMS gate open: the owner stopped 5 s before the
        # break. The tail is measured, not capped at the 4 s scan.
        v, sess, _ = _vad()
        got = v.speech_tail_ms(_clip(1.0, 5.0))
        self.assertEqual(len(sess.batches), 2)
        self.assertLessEqual(abs(got - 5000), 32, got)

    def test_no_speech_anywhere_is_none(self):
        v, _, _ = _vad()
        self.assertIsNone(v.speech_tail_ms(np.zeros(3 * SR, np.float32)))
        self.assertIsNone(v.speech_tail_ms(np.zeros(6 * SR, np.float32)))
        self.assertEqual(v.failed, "")       # not a failure: nothing said

    def test_unmeasurable_input_is_none_and_never_latches(self):
        v, sess, calls = _vad()
        for bad in (None, object(), "text", np.zeros(W - 1, np.float32),
                    np.zeros((2, SR), np.float32)):
            self.assertIsNone(v.speech_tail_ms(bad))
        self.assertIsNone(v.speech_tail_ms(_clip(1, 1), sample_rate=48000))
        self.assertEqual(v.failed, "")
        self.assertEqual(calls, [])          # nothing even loaded
        # (n, 1) mono is accepted.
        self.assertIsNotNone(v.speech_tail_ms(_clip(1, 1).reshape(-1, 1)))

    def test_int16_like_input_is_converted(self):
        v, _, _ = _vad()
        a = (_clip(1.0, 1.0) * 30000).astype(np.int16)
        self.assertIsNotNone(v.speech_tail_ms(a))


class LatchTests(unittest.TestCase):
    def test_load_failure_latches_and_is_never_retried(self):
        calls = []

        def factory(p):
            calls.append(p)
            raise OSError("no such file")

        v = ep.SileroVad(model_path="x.onnx", session_factory=factory)
        self.assertIsNone(v.speech_tail_ms(_clip(1, 1)))
        self.assertIsNone(v.speech_tail_ms(_clip(1, 1)))
        self.assertEqual(len(calls), 1)
        self.assertIn("load failed: OSError", v.failed)

    def test_run_failure_latches(self):
        v, sess, _ = _vad(_FakeSession(fail=RuntimeError("ort")))
        self.assertIsNone(v.speech_tail_ms(_clip(1, 1)))
        self.assertIn("run failed", v.failed)
        sess.fail = None
        self.assertIsNone(v.speech_tail_ms(_clip(1, 1)))   # stays off

    def test_bad_output_latches(self):
        for sess in (_FakeSession(bad_len=True), _FakeSession(nan=True)):
            v, _, _ = _vad(sess)
            self.assertIsNone(v.speech_tail_ms(_clip(1, 1)))
            self.assertIn("bad output", v.failed)

    def test_missing_model_latches(self):
        with mock.patch.object(ep, "bundled_model_path", return_value=None):
            v = ep.SileroVad(session_factory=lambda p: _FakeSession())
            self.assertIsNone(v.speech_tail_ms(_clip(1, 1)))
        self.assertIn("model not found", v.failed)

    def test_warm(self):
        v, sess, calls = _vad()
        self.assertTrue(v.warm())
        self.assertEqual(len(calls), 1)
        self.assertEqual(sess.batches[0].shape, (1, ep.CONTEXT + W))
        bad, _, _ = _vad(_FakeSession(fail=RuntimeError("x")))
        with self.assertRaises(RuntimeError):
            bad.warm()
        self.assertTrue(bad.failed)

    def test_concurrent_calls_are_serialised(self):
        gate = threading.Event()
        sess = _FakeSession(gate=gate)
        v, _, calls = _vad(sess)
        out = []
        ths = [threading.Thread(target=lambda: out.append(
            v.speech_tail_ms(_clip(1, 1)))) for _ in range(4)]
        for th in ths:
            th.start()
        time.sleep(0.05)
        gate.set()
        for th in ths:
            th.join(5)
        self.assertEqual(len(out), 4)
        self.assertEqual(sess.max_active, 1)
        self.assertEqual(len(calls), 1)      # one load, shared


class ModelPathTests(unittest.TestCase):
    def test_not_installed_is_none(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertIsNone(ep.bundled_model_path())
        with mock.patch("importlib.util.find_spec",
                        side_effect=ValueError("bad")):
            self.assertIsNone(ep.bundled_model_path())

    def test_finds_the_bundled_asset_without_importing(self):
        d = tempfile.mkdtemp(prefix="jarvis_ep_")
        self.addCleanup(lambda: __import__("shutil").rmtree(d, True))
        os.makedirs(os.path.join(d, "assets"))
        p = os.path.join(d, "assets", ep.MODEL_FILE)
        with open(p, "wb") as fh:
            fh.write(b"onnx")
        spec = mock.Mock(submodule_search_locations=[d])
        with mock.patch("importlib.util.find_spec", return_value=spec):
            self.assertEqual(ep.bundled_model_path(), p)
        spec = mock.Mock(submodule_search_locations=[os.path.join(d, "x")])
        with mock.patch("importlib.util.find_spec", return_value=spec):
            self.assertIsNone(ep.bundled_model_path())


class ModuleHygieneTests(unittest.TestCase):
    def test_module_level_imports_are_stdlib_only(self):
        with open(ep.__file__, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        mods = set()
        for node in tree.body:     # module level only; methods import lazily
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        self.assertLessEqual(mods, {"__future__", "importlib", "os",
                                    "threading"})

    def test_import_pulls_in_no_native_dependency(self):
        # In a FRESH interpreter (this one may already hold onnxruntime from
        # another test): importing the module must not import onnxruntime,
        # numpy or faster_whisper. A module-level `import onnxruntime` fails
        # here on any box that has it installed.
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "import core.endpointing; "
                "print(sorted(m for m in ('onnxruntime', 'numpy', "
                "'faster_whisper') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code, _ROOT],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]", out.stdout)


def _real_model_available():
    try:
        import importlib.util
        return (importlib.util.find_spec("onnxruntime") is not None
                and ep.bundled_model_path() is not None)
    except Exception:
        return False


@unittest.skipUnless(_real_model_available(),
                     "onnxruntime + faster-whisper's Silero model: local tier")
class RealModelTests(unittest.TestCase):
    def test_real_session_loads_and_scores_silence(self):
        v = ep.SileroVad()
        self.assertTrue(v.warm())
        self.assertIsNone(v.speech_tail_ms(np.zeros(3 * SR, np.float32)))
        self.assertEqual(v.failed, "")


if __name__ == "__main__":
    unittest.main()
