"""The wake pre-gate (2026-10-05): core/wake_pregate.py.

openWakeWord is replaced by a fake model (scores by rule) for the worker,
the gain and the timing; one test loads the REAL hey_jarvis ONNX graph on
the CPU when openwakeword is installed (skipped on the CI runner) and checks
it is CPU-only and single-threaded. No device is opened. numpy + stdlib.

    python -m unittest tests.test_wake_pregate
"""
from __future__ import annotations

import importlib.util
import time
import unittest

import numpy as np

from core import listen_media as lm
from core import wake_pregate as wpg

SR = 16000


class FakeModel:
    """Scores 0.9 for an 80 ms frame whose int16 peak exceeds 20000 (a
    "Jarvis"), else 0.0; remembers what it saw."""

    def __init__(self):
        self.frames = []

    def predict(self, pcm):
        self.frames.append(pcm.copy())
        return {"hey_jarvis_v0.1": 0.9 if int(np.abs(pcm).max()) > 20000
                else 0.0}

    def reset(self):
        pass


class DetectorTests(unittest.TestCase):

    def test_80_ms_frames_timed_from_the_chunk_arrival(self):
        m = FakeModel()
        d = wpg.Detector(model_factory=lambda: m, gain=False)
        out = d.feed(np.zeros(1024, np.float32), t_end=10.0)
        self.assertEqual(out, [])                  # under one frame
        out = d.feed(np.zeros(2048, np.float32), t_end=10.128)
        self.assertEqual(len(out), 2)              # 3072 samples = 2 frames
        # The 2nd frame ended at sample 2560 of 3072: 512 samples before
        # the newest, which arrived at 10.128.
        self.assertAlmostEqual(out[1][0], 10.128 - 512 / SR, places=6)
        self.assertEqual(len(m.frames[0]), wpg.FRAME)
        self.assertEqual(m.frames[0].dtype, np.int16)

    def test_a_quiet_stream_gets_the_capture_auto_gain(self):
        m = FakeModel()
        d = wpg.Detector(model_factory=lambda: m, gain=True)
        speech = 0.05 * np.sin(np.linspace(0, 400, 1280)).astype(np.float32)
        d.feed(speech, t_end=1.0)
        peak = int(np.abs(m.frames[-1]).max())
        # 0.05 RMS * (0.25 / 0.05 target) ~ x5 -> peak ~0.35 of full scale
        self.assertGreater(peak, 8000)
        g = wpg.StreamGain()
        self.assertEqual(g.gain(np.zeros(1280, np.float32), 0.0), 1.0)
        self.assertEqual(g.gain(np.full(1280, 0.4, np.float32), 0.1), 1.0)

    def test_a_load_failure_latches_off(self):
        calls = []

        def boom():
            calls.append(1)
            raise RuntimeError("no model")

        d = wpg.Detector(model_factory=boom)
        for i in range(20):                  # 20 frames: ONE load attempt
            self.assertEqual(d.feed(np.zeros(2560, np.float32), 1.0 + i), [])
        self.assertIn("no model", d.failed)
        self.assertFalse(d.ready())
        self.assertEqual(len(calls), 1, "latched: never re-loaded per frame")


class WorkerTests(unittest.TestCase):

    def _worker(self, **kw):
        m = FakeModel()
        w = wpg.PregateWorker("mic", track=lm.ScoreTrack(),
                              detector=wpg.Detector(model_factory=lambda: m,
                                                    gain=False),
                              log=lambda line: None, **kw)
        self.addCleanup(w.shutdown)
        self.assertTrue(w.start())
        return w

    def _wait(self, cond, timeout=3.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.01)
        return False

    def test_scores_reach_the_track_and_the_trigger(self):
        hits = []
        w = self._worker(trigger=lm.PregateTrigger(0.15),
                         on_trigger=lambda t, s: hits.append((t, s)))
        loud = np.zeros(1280, np.float32)
        loud[100] = 0.9
        w.feed(np.zeros(1280, np.float32), 5.0)
        w.feed(loud, 5.08)
        self.assertTrue(self._wait(lambda: hits))
        self.assertAlmostEqual(hits[0][0], 5.08, places=6)
        self.assertAlmostEqual(w.track.max_between(5.0, 5.1), 0.9)

    def test_the_record_tap_stamps_arrival_time(self):
        w = self._worker()
        t0 = time.monotonic()
        w.tap.put_nowait(np.zeros(1280, np.float32))
        self.assertTrue(self._wait(lambda: w.frames == 1))
        t = w.track.last_time()
        self.assertGreaterEqual(t, t0)
        self.assertLess(t, t0 + 2.0)

    def test_off_drops_frames_unscored(self):
        w = self._worker(active=lambda: False)
        w.feed(np.zeros(2560, np.float32), 1.0)
        time.sleep(0.1)
        self.assertEqual(w.frames, 0)

    def test_the_queue_is_bounded(self):
        w = wpg.PregateWorker("mic", track=lm.ScoreTrack(),
                              detector=wpg.Detector(
                                  model_factory=FakeModel, gain=False))
        for i in range(wpg.QUEUE_MAX + 10):
            w.feed(np.zeros(1024, np.float32), float(i))
        self.assertEqual(w.dropped, 10)            # never started: queued


@unittest.skipIf(importlib.util.find_spec("openwakeword") is None
                 or importlib.util.find_spec("onnxruntime") is None,
                 "openwakeword / onnxruntime not installed")
class RealModelTests(unittest.TestCase):
    """Installed is not importable: under tools/run_tests_ci_sim.py (the
    Linux runner faked on Windows) onnxruntime's import fails with "DLL
    initialization routine failed" (2026-10-09 review: an ERROR in the
    pre-push gate). The import is tried once, here, and a failure skips."""

    @classmethod
    def setUpClass(cls):
        try:
            importlib.import_module("onnxruntime")
            importlib.import_module("openwakeword.model")
        except (Exception, SystemExit) as e:   # ImportError, OSError ...
            raise unittest.SkipTest(f"openwakeword / onnxruntime do not "
                                    f"import here ({type(e).__name__})")

    def test_the_real_graph_loads_on_the_cpu_single_threaded(self):
        m = wpg.load_model()
        sess = m.models["hey_jarvis_v0.1"]
        self.assertEqual(sess.get_providers(), ["CPUExecutionProvider"])
        opts = sess.get_session_options()
        self.assertEqual(opts.intra_op_num_threads, 1)
        self.assertEqual(opts.inter_op_num_threads, 1)
        self.assertEqual(m.preprocessor.melspec_model.get_providers(),
                         ["CPUExecutionProvider"])
        d = wpg.Detector(model_factory=lambda: m)
        rng = np.random.default_rng(5)
        out = d.feed((0.001 * rng.standard_normal(SR)).astype(np.float32),
                     t_end=1.0)
        self.assertEqual(len(out), SR // wpg.FRAME)
        self.assertTrue(all(0.0 <= s < 0.1 for _t, s in out))


if __name__ == "__main__":
    unittest.main()
