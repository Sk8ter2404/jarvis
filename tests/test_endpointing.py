"""core/endpointing.py — the Silero speech tail behind [turn-timing] tail_ms
(speed plan R1, 2026-10-01) and the Smart Turn end of turn (R7, 2026-10-02).

Light tier: a FAKE onnxruntime session (CI never installs onnxruntime) whose
per-window probability is "speech" when the window is loud, so the clip's real
end of speech is known to the sample. The real-model test at the bottom runs
only where onnxruntime and faster-whisper's bundled model exist.

R7: EotDecider is pure, so its tests script the Silero probabilities and the
Smart Turn p directly (the fake vad is the identity: each "chunk" IS that
chunk's probability list). SileroStream and SmartTurn run on fake sessions,
fake features and a fake clock; no real model, no onnxruntime import.

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
        self.assertLessEqual(mods, {"__future__", "importlib", "math", "os",
                                    "threading", "time"})

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


# ─── R7: Smart Turn end of turn ──────────────────────────────────────────

SPEECH = [0.9, 0.9]     # one 64 ms chunk = two Silero windows
QUIET = [0.02, 0.02]


class _Predict:
    """Smart Turn p from a list (the last value repeats); records, at every
    call, how many chunks the fake vad had been fed (`fed`, when given)."""

    def __init__(self, ps, fed=None):
        self.ps = list(ps) if isinstance(ps, (list, tuple)) else [ps]
        self.calls = []
        self.fed = fed

    def __call__(self):
        self.calls.append(None if self.fed is None else len(self.fed))
        return self.ps[min(len(self.calls), len(self.ps)) - 1]


def _decider(mode="on", ps=0.99, **kw):
    fed = []
    pred = _Predict(ps, fed)

    def vad(probs):
        fed.append(probs)
        return probs

    return ep.EotDecider(mode, vad=vad, predict=pred, **kw), pred


def _speak(dec, chunks=20):
    """`chunks` voiced chunks (silence_n 0): 2 x `chunks` Silero speech
    windows. Returns the update() results."""
    return [dec.update(SPEECH, 0) for _ in range(chunks)]


def _pause(dec, chunks, probs=QUIET, start=1):
    """`chunks` RMS-silent chunks (silence_n start, start+1, ...). The
    1-based pause chunk on which update() returned True, or None."""
    for i in range(chunks):
        if dec.update(probs, start + i):
            return i + 1
    return None


class EotDeciderGateTests(unittest.TestCase):
    def test_ends_on_the_first_due_check_when_p_clears_the_threshold(self):
        dec, pred = _decider("on", 0.99)
        self.assertEqual(_speak(dec), [False] * 20)
        self.assertEqual(_pause(dec, 21), 4)          # 4 x 64 = 256 ms
        self.assertTrue(dec.should_end())
        self.assertEqual(len(pred.calls), 1)
        self.assertEqual(dec.record().stats(),
                         {"eot": "st", "st_p": 0.99, "st_n": 1})

    def test_never_ends_early_while_silero_says_speech(self):
        # RMS-silent but Silero hears speech (a quiet talker): no check runs,
        # whatever Smart Turn would say. Mutation-proven: deleting the
        # Silero-silence guard in EotDecider._due turns this red.
        dec, pred = _decider("on", 0.99)
        _speak(dec)
        self.assertIsNone(_pause(dec, 40, probs=SPEECH))
        self.assertEqual(pred.calls, [])
        self.assertEqual(dec.record().eot, "rms")

    def test_not_before_256_ms_of_silence(self):
        dec, pred = _decider("on", 0.99)
        _speak(dec)
        # Silero has been silent for 8 windows (256 ms) by RMS chunk 3, but
        # the capture's own silence count is only 3 chunks (192 ms).
        self.assertFalse(dec.update([0.02] * 8, 1))
        self.assertFalse(dec.update(QUIET, 2))
        self.assertFalse(dec.update(QUIET, 3))
        self.assertEqual(pred.calls, [])
        self.assertTrue(dec.update(QUIET, 4))
        # The knob is honoured, rounded UP to whole chunks: 0.27 s -> 5.
        dec, _ = _decider("on", 0.99, min_silence_s=0.27)
        _speak(dec)
        self.assertEqual(_pause(dec, 21), 5)

    def test_needs_a_silero_silence_run_of_200_ms(self):
        dec, pred = _decider("on", 0.99)
        _speak(dec)
        # RMS silent for 4 chunks while Silero still heard speech...
        for n in range(1, 5):
            self.assertFalse(dec.update(SPEECH, n))
        # ... then 6 silent windows (192 ms): not yet.
        self.assertFalse(dec.update([0.02] * 6, 5))
        self.assertEqual(pred.calls, [])
        # The 7th (224 ms) makes it a run.
        self.assertTrue(dec.update([0.02], 6))

    def test_not_with_less_than_one_second_of_speech(self):
        dec, pred = _decider("on", 0.99)
        dec.update([0.9] * 31, 0)                     # 992 ms voiced
        self.assertIsNone(_pause(dec, 21))
        self.assertEqual(pred.calls, [])
        dec, pred = _decider("on", 0.99)
        dec.update([0.9] * 32, 0)                     # 1,024 ms voiced
        self.assertEqual(_pause(dec, 21), 4)

    def test_keeps_waiting_below_the_threshold_with_checks_5_chunks_apart(self):
        dec, pred = _decider("on", [0.5, 0.69, 0.7])
        _speak(dec)
        self.assertEqual(_pause(dec, 21), 14)         # checks at 4, 9, 14
        self.assertEqual(pred.calls, [24, 29, 34])
        rec = dec.record()
        self.assertEqual((rec.eot, rec.st_p, rec.st_n), ("st", 0.7, 3))

    def test_never_after_the_check_cap(self):
        dec, pred = _decider("on", [0.5] * ep.ST_MAX_CHECKS + [0.99])
        _speak(dec)
        self.assertIsNone(_pause(dec, 200))
        self.assertEqual(len(pred.calls), ep.ST_MAX_CHECKS)
        self.assertEqual(dec.record().stats(),
                         {"eot": "rms", "st_p": 0.5, "st_n": ep.ST_MAX_CHECKS})

    def test_a_new_turn_starts_clean(self):
        dec, pred = _decider("on", 0.99)
        _speak(dec)
        _pause(dec, 21)
        dec.reset()
        self.assertFalse(dec.should_end())
        self.assertIsNone(_pause(dec, 21))            # no speech this turn
        self.assertEqual(dec.record().stats(),
                         {"eot": "rms", "st_p": None, "st_n": 0})


class EotDeciderHysteresisTests(unittest.TestCase):
    def test_speech_holds_until_p_drops_below_0_35(self):
        dec, pred = _decider("on", 0.99)
        _speak(dec)
        # 0.4 is under the 0.5 entry but over the 0.35 exit: still speech.
        self.assertIsNone(_pause(dec, 10, probs=[0.4, 0.4]))
        self.assertEqual(pred.calls, [])
        # 0.34 leaves speech: a silence run starts.
        self.assertTrue(dec.update([0.34] * 7, 11))

    def test_held_windows_count_as_speech(self):
        dec, _ = _decider("on", 0.99)
        dec.update([0.9] * 31 + [0.4], 0)             # 32 windows of speech
        self.assertEqual(_pause(dec, 21), 4)
        dec, pred = _decider("on", 0.99)
        dec.update([0.9] * 31 + [0.34], 0)            # 31: under 1 s
        self.assertIsNone(_pause(dec, 21))
        self.assertEqual(pred.calls, [])

    def test_speech_starts_only_at_0_5(self):
        dec, pred = _decider("on", 0.99)
        dec.update([0.49] * 40, 0)                    # never enters speech
        self.assertIsNone(_pause(dec, 21))
        self.assertEqual(pred.calls, [])
        dec, pred = _decider("on", 0.99)
        dec.update([0.5] * 32, 0)
        self.assertEqual(_pause(dec, 21), 4)

    def test_a_blip_inside_a_pause_restarts_the_silero_run(self):
        dec, pred = _decider("on", 0.99)
        _speak(dec)
        self.assertFalse(dec.update([0.02] * 6, 4))
        self.assertFalse(dec.update([0.6], 5))        # speech again
        self.assertFalse(dec.update([0.02] * 6, 6))   # 192 ms: not yet
        self.assertTrue(dec.update([0.02], 7))
        self.assertEqual(len(pred.calls), 1)


class EotDeciderFallbackTests(unittest.TestCase):
    def test_no_silero_never_ends_early(self):
        pred = _Predict(0.99)
        dec = ep.EotDecider("on", vad=None, predict=pred)
        _speak(dec)
        self.assertIsNone(_pause(dec, 21))
        self.assertEqual(pred.calls, [])

    def test_silero_lost_mid_turn_never_ends_early(self):
        feeds = []

        def vad(probs):
            feeds.append(probs)
            return None if len(feeds) == 21 else probs

        pred = _Predict(0.99)
        dec = ep.EotDecider("on", vad=vad, predict=pred)
        _speak(dec)
        self.assertIsNone(_pause(dec, 21))
        self.assertEqual(pred.calls, [])
        self.assertEqual(len(feeds), 21)              # not asked again

    def test_bad_silero_output_or_a_raising_vad_never_ends_early(self):
        def boom(_):
            raise RuntimeError("x")

        for vad in (lambda p: [float("nan")], lambda p: [1.5], boom):
            pred = _Predict(0.99)
            dec = ep.EotDecider("on", vad=vad, predict=pred)
            dec.update(SPEECH, 0)
            self.assertIsNone(_pause(dec, 21))
            self.assertEqual(pred.calls, [])

    def test_smart_turn_unavailable_never_ends_early(self):
        def boom():
            raise RuntimeError("x")

        for predict in (None, lambda: None, lambda: float("nan"),
                        lambda: 1.2, boom):
            dec = ep.EotDecider("on", vad=lambda p: p, predict=predict)
            _speak(dec)
            self.assertIsNone(_pause(dec, 21), predict)
            self.assertEqual(dec.record().eot, "rms")

    def test_unavailable_smart_turn_is_asked_once_per_turn(self):
        calls = []
        dec = ep.EotDecider("on", vad=lambda p: p,
                            predict=lambda: calls.append(1))
        _speak(dec)
        _pause(dec, 21)
        self.assertEqual(len(calls), 1)
        self.assertEqual(dec.record().stats(),
                         {"eot": "rms", "st_p": None, "st_n": 1})

    def test_unknown_mode_is_off_and_mode_is_normalised(self):
        self.assertEqual(ep.EotDecider(" On ").mode, "on")
        self.assertEqual(ep.EotDecider("SHADOW").mode, "shadow")
        for bad in ("", "true", "1", None):
            self.assertEqual(ep.EotDecider(bad).mode, "off")


class EotDeciderModeTests(unittest.TestCase):
    def test_off_calls_neither_model(self):
        vad_calls = []
        pred = _Predict(0.99)
        dec = ep.EotDecider("off", vad=lambda p: vad_calls.append(p) or p,
                            predict=pred)
        _speak(dec)
        self.assertIsNone(_pause(dec, 21))
        self.assertEqual((vad_calls, pred.calls), ([], []))
        rec = dec.record()
        self.assertEqual(rec.stats(), {"eot": "rms", "st_p": None, "st_n": 0})
        self.assertIsNone(rec.shadow_line())

    def test_shadow_never_ends_but_records_what_it_would_have_done(self):
        dec, pred = _decider("shadow", 0.91)
        self.assertEqual(_speak(dec), [False] * 20)
        self.assertIsNone(_pause(dec, 21))
        self.assertFalse(dec.should_end())
        self.assertEqual(len(pred.calls), 1)          # stops at the fire
        rec = dec.record()
        self.assertEqual(rec.stats(), {"eot": "rms", "st_p": 0.91, "st_n": 1})
        self.assertEqual(rec.fire_ms, 24 * 64)
        self.assertEqual(rec.actual_ms, 41 * 64)
        self.assertEqual(rec.shadow_line(),
                         "[eot-shadow] fire_ms=1536 p=0.910 resumed=0 "
                         "actual_ms=2624")

    def test_shadow_marks_a_turn_that_went_on_after_the_fire(self):
        dec, pred = _decider("shadow", 0.91)
        _speak(dec)
        _pause(dec, 6)                                # fired at pause 4
        self.assertEqual(dec.record().resumed, 0)
        _speak(dec, 3)                                # the owner went on
        _pause(dec, 21)
        rec = dec.record()
        self.assertEqual((rec.resumed, rec.fire_ms, rec.st_n),
                         (1, 24 * 64, 1))
        self.assertIn("resumed=1", rec.shadow_line())

    def test_shadow_line_when_it_never_fired(self):
        dec, _ = _decider("shadow", 0.2)
        _speak(dec)
        _pause(dec, 21)
        self.assertEqual(dec.record().shadow_line(),
                         "[eot-shadow] fire_ms=- p=0.200 resumed=0 "
                         "actual_ms=2624")
        dec, _ = _decider("shadow", 0.2)
        _pause(dec, 21)                               # nothing said
        self.assertEqual(dec.record().shadow_line(),
                         "[eot-shadow] fire_ms=- p=- resumed=0 "
                         "actual_ms=1344")

    def test_on_mode_record_has_no_shadow_line(self):
        dec, _ = _decider("on", 0.99)
        _speak(dec)
        _pause(dec, 21)
        self.assertIsNone(dec.record().shadow_line())
        self.assertEqual(dec.record("max").eot, "st")  # it did end the turn
        dec, _ = _decider("on", 0.1)
        self.assertEqual(dec.record("max").eot, "max")

    def test_stats_are_turn_timing_note_fields(self):
        from core import turn_timing
        dec, _ = _decider("on", 0.99)
        self.assertLessEqual(set(dec.record().stats()),
                             set(turn_timing.NOTE_FIELDS))


class _StreamSession:
    """Silero-shaped fake that carries state: h_out = h_in + windows,
    c_out = c_in - windows; p 0.9 for a loud window. Records every feed."""

    def __init__(self, fail=None, bad_state=False):
        self.feeds = []
        self.fail = fail
        self.bad_state = bad_state

    def run(self, _names, feeds):
        if self.fail is not None:
            raise self.fail
        self.feeds.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        x = feeds["input"]
        n = x.shape[0]
        loud = np.abs(x[:, ep.CONTEXT:]).mean(axis=1) > 0.05
        h = feeds["h"] + n
        if self.bad_state:
            h = np.zeros((2, 128), np.float32)
        return [np.where(loud, 0.9, 0.02).astype(np.float32).reshape(-1, 1),
                h, feeds["c"] - n]


def _stream(sess=None):
    sess = sess if sess is not None else _StreamSession()
    calls = []

    def factory(p):
        calls.append(p)
        return sess

    return ep.SileroStream(model_path="fake.onnx",
                           session_factory=factory), sess, calls


class SileroStreamTests(unittest.TestCase):
    def test_carries_state_and_context_across_chunks(self):
        s, sess, calls = _stream()
        a = (np.arange(3 * 1024, dtype=np.float32) + 1) / 1e4
        out = [s.feed(a[i:i + 1024]) for i in range(0, len(a), 1024)]
        self.assertEqual([len(o) for o in out], [2, 2, 2])
        self.assertEqual(len(calls), 1)
        f = sess.feeds
        self.assertTrue(np.all(f[0]["h"] == 0) and np.all(f[0]["c"] == 0))
        self.assertTrue(np.all(f[1]["h"] == 2) and np.all(f[1]["c"] == -2))
        self.assertTrue(np.all(f[2]["h"] == 4))
        np.testing.assert_array_equal(f[0]["input"][0, :ep.CONTEXT],
                                      np.zeros(ep.CONTEXT, np.float32))
        np.testing.assert_array_equal(f[1]["input"][0, :ep.CONTEXT],
                                      a[1024 - ep.CONTEXT:1024])

    def test_streamed_input_equals_the_one_shot_batch(self):
        # Odd chunk sizes: samples short of a window wait for the next chunk,
        # and the windows, with their contexts, are exactly the ones a single
        # SileroVad run over the whole clip builds.
        a = np.sin(np.arange(20 * W, dtype=np.float32) / 7.0) * 0.3
        s, sess, _ = _stream()
        sizes = [len(s.feed(a[i:i + 700])) for i in range(0, len(a), 700)]
        self.assertEqual(sum(sizes), 20)
        self.assertEqual(sizes[:3], [1, 1, 2])        # 700, 888, 1076 samples
        streamed = np.concatenate([f["input"] for f in sess.feeds])
        v, one, _ = _vad()
        v._probs(a, None)
        np.testing.assert_array_equal(streamed, one.batches[0])

    def test_short_chunks_wait_for_a_whole_window(self):
        s, sess, _ = _stream()
        self.assertEqual(s.feed(np.zeros(300, np.float32)), [])
        self.assertEqual(sess.feeds, [])
        self.assertEqual(len(s.feed(np.zeros(300, np.float32))), 1)

    def test_probabilities_follow_the_session(self):
        s, _, _ = _stream()
        np.testing.assert_allclose(s.feed(np.full(1024, 0.3, np.float32)),
                                   [0.9, 0.9], rtol=1e-6)
        np.testing.assert_allclose(s.feed(np.zeros(1024, np.float32)),
                                   [0.02, 0.02], rtol=1e-6)

    def test_reset_starts_a_new_turn(self):
        s, sess, calls = _stream()
        a = np.full(1024 + 100, 0.3, np.float32)
        s.feed(a)
        s.reset()
        self.assertEqual(len(s.feed(np.zeros(1024, np.float32))), 2)
        f = sess.feeds[-1]
        self.assertTrue(np.all(f["h"] == 0))
        np.testing.assert_array_equal(f["input"][0, :ep.CONTEXT],
                                      np.zeros(ep.CONTEXT, np.float32))
        self.assertEqual(len(calls), 1)               # same session

    def test_unusable_chunks_are_none_and_never_latch(self):
        s, sess, calls = _stream()
        for bad in (None, "x", np.zeros((2, 1024), np.float32)):
            self.assertIsNone(s.feed(bad))
        self.assertIsNone(s.feed(np.zeros(1024, np.float32), sample_rate=48000))
        self.assertEqual(s.failed, "")
        self.assertEqual(len(s.feed(np.zeros((1024, 1), np.float32))), 2)

    def test_failures_latch(self):
        s, sess, _ = _stream(_StreamSession(fail=RuntimeError("ort")))
        self.assertIsNone(s.feed(np.zeros(1024, np.float32)))
        self.assertIn("run failed", s.failed)
        sess.fail = None
        self.assertIsNone(s.feed(np.zeros(1024, np.float32)))
        s, _, _ = _stream(_StreamSession(bad_state=True))
        self.assertIsNone(s.feed(np.zeros(1024, np.float32)))
        self.assertIn("bad output", s.failed)
        calls = []

        def factory(p):
            calls.append(p)
            raise OSError("gone")

        s = ep.SileroStream(model_path="x.onnx", session_factory=factory)
        self.assertIsNone(s.feed(np.zeros(1024, np.float32)))
        self.assertIsNone(s.feed(np.zeros(1024, np.float32)))
        self.assertEqual(len(calls), 1)

    def test_warm_does_not_touch_the_stream_state(self):
        s, sess, _ = _stream()
        s.feed(np.full(1024, 0.3, np.float32))
        self.assertTrue(s.warm())
        s.feed(np.zeros(1024, np.float32))
        self.assertTrue(np.all(sess.feeds[-1]["h"] == 2))


class _Meta:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _TurnSession:
    """Smart Turn-shaped fake: p and per-run cost (s of fake clock) from
    lists (the last value repeats). Its clock is the fake clock."""

    def __init__(self, ps=0.8, costs=0.03, inputs=None, outputs=None,
                 fail=None):
        self.ps = ps if isinstance(ps, list) else [ps]
        self.costs = costs if isinstance(costs, list) else [costs]
        self.inputs = inputs if inputs is not None else [
            _Meta("input_features", ["batch", 80, 800])]
        self.outputs = outputs if outputs is not None else [
            _Meta("logits", ["batch", 1])]
        self.fail = fail
        self.now = 0.0
        self.feeds = []

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def clock(self):
        return self.now

    def run(self, _names, feeds):
        i = len(self.feeds)
        self.feeds.append(feeds)
        self.now += self.costs[min(i, len(self.costs) - 1)]
        if self.fail is not None:
            raise self.fail
        return [np.array([[self.ps[min(i, len(self.ps) - 1)]]], np.float32)]


def _turn(sess=None, path="fake.onnx", features=None):
    sess = sess if sess is not None else _TurnSession()
    seen = []
    calls = []

    def feats(x):
        seen.append(np.array(x, copy=True))
        return np.zeros((80, 800), np.float32)

    def factory(p):
        calls.append(p)
        return sess

    st = ep.SmartTurn(path, session_factory=factory,
                      features=features if features is not None else feats,
                      clock=sess.clock)
    return st, sess, seen, calls


class SmartTurnTests(unittest.TestCase):
    def test_predict_feeds_the_last_8_s_left_padded_and_normalised(self):
        st, sess, seen, _ = _turn()
        clip = _clip(1.0, 0.5)
        self.assertAlmostEqual(st.predict(clip), 0.8, places=6)
        x = seen[0]
        self.assertEqual(x.shape, (ep.ST_SAMPLES,))
        self.assertAlmostEqual(float(x.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(x.std()), 1.0, places=3)
        # The clip sits at the END; the left is padding (one value).
        pad = ep.ST_SAMPLES - len(clip)
        self.assertEqual(len(np.unique(x[:pad])), 1)
        raw = np.concatenate([np.zeros(pad, np.float32), clip])
        np.testing.assert_allclose(
            x, (raw - raw.mean()) / np.sqrt(raw.var() + 1e-7), atol=1e-5)
        feed = sess.feeds[0]
        self.assertEqual(list(feed), ["input_features"])
        self.assertEqual(feed["input_features"].shape, (1, 80, 800))
        self.assertEqual(feed["input_features"].dtype, np.float32)

    def test_a_long_clip_keeps_only_its_last_8_s(self):
        st, _, seen, _ = _turn()
        clip = np.concatenate([np.full(5 * SR, 0.9, np.float32),
                               np.linspace(-0.2, 0.2, 8 * SR,
                                           dtype=np.float32)])
        st.predict(clip)
        tail = clip[-ep.ST_SAMPLES:]
        np.testing.assert_allclose(
            seen[0], (tail - tail.mean()) / np.sqrt(tail.var() + 1e-7),
            atol=1e-5)

    def test_warm_checks_the_io_and_runs_zeros_twice(self):
        st, sess, seen, calls = _turn(_TurnSession(costs=[1.0, 0.05]))
        self.assertTrue(st.warm())        # a slow FIRST run is not counted
        self.assertEqual(len(sess.feeds), 2)
        self.assertTrue(all(np.all(x == 0) for x in seen))
        self.assertEqual((len(calls), st.failed), (1, ""))

    def test_wrong_io_latches_at_load(self):
        cases = (
            _TurnSession(inputs=[_Meta("input", ["batch", 80, 800])]),
            _TurnSession(inputs=[_Meta("input_features", ["batch", 80, 3000])]),
            _TurnSession(inputs=[_Meta("input_features", [80, 800])]),
            _TurnSession(outputs=[_Meta("probs", ["batch", 1])]),
            _TurnSession(outputs=[_Meta("logits", ["batch", 2])]),
        )
        for sess in cases:
            st, _, _, _ = _turn(sess)
            with self.assertRaises(RuntimeError):
                st.warm()
            self.assertIn("not a Smart Turn v3 model", st.failed)
            self.assertEqual(sess.feeds, [])
            self.assertIsNone(st.predict(_clip(1, 1)))

    def test_a_warm_call_over_150_ms_latches(self):
        st, sess, _, _ = _turn(_TurnSession(costs=[0.05, 0.151]))
        with self.assertRaises(RuntimeError):
            st.warm()
        self.assertIn("too slow: warm call 151 ms", st.failed)
        self.assertIsNone(st.predict(_clip(1, 1)))
        self.assertEqual(len(sess.feeds), 2)

    def test_three_slow_runtime_calls_in_a_row_latch(self):
        st, sess, _, _ = _turn(_TurnSession(
            costs=[0.2, 0.2, 0.01, 0.2, 0.2, 0.2, 0.01]))
        got = [st.predict(_clip(1, 1)) for _ in range(7)]
        # Two slow, one fast (streak reset), then three slow: the third
        # latches and returns None, and nothing runs after it.
        self.assertEqual([g is not None for g in got],
                         [True, True, True, True, True, False, False])
        self.assertIn("too slow: 3 calls in a row", st.failed)
        self.assertEqual(len(sess.feeds), 6)

    def test_non_finite_or_out_of_range_p_latches(self):
        for p in (float("nan"), float("inf"), 1.01, -0.01):
            st, sess, _, _ = _turn(_TurnSession(ps=[p, 0.5]))
            self.assertIsNone(st.predict(_clip(1, 1)))
            self.assertIn("bad output", st.failed)
            self.assertIsNone(st.predict(_clip(1, 1)))
            self.assertEqual(len(sess.feeds), 1)
        for p in (0.0, 1.0):
            st, _, _, _ = _turn(_TurnSession(ps=p))
            self.assertEqual(st.predict(_clip(1, 1)), p)

    def test_load_failures_latch_and_are_never_retried(self):
        calls = []

        def factory(p):
            calls.append(p)
            raise OSError("no such file")

        st = ep.SmartTurn("x.onnx", session_factory=factory,
                          features=lambda x: np.zeros((80, 800), np.float32))
        self.assertIsNone(st.predict(_clip(1, 1)))
        self.assertIsNone(st.predict(_clip(1, 1)))
        self.assertEqual(len(calls), 1)
        self.assertIn("load failed: OSError", st.failed)
        st = ep.SmartTurn("", session_factory=factory)
        self.assertIsNone(st.predict(_clip(1, 1)))
        self.assertIn("no model path", st.failed)
        with self.assertRaises(RuntimeError):
            st.warm()

    def test_run_failure_and_bad_features_latch(self):
        st, _, _, _ = _turn(_TurnSession(fail=RuntimeError("ort")))
        self.assertIsNone(st.predict(_clip(1, 1)))
        self.assertIn("run failed", st.failed)
        st, sess, _, _ = _turn(
            features=lambda x: np.zeros((80, 801), np.float32))
        self.assertIsNone(st.predict(_clip(1, 1)))
        self.assertIn("bad features", st.failed)
        self.assertEqual(sess.feeds, [])

    def test_unusable_input_is_none_and_never_latches(self):
        st, sess, _, calls = _turn()
        for bad in (None, "text", np.zeros(0, np.float32),
                    np.zeros((2, SR), np.float32)):
            self.assertIsNone(st.predict(bad))
        self.assertIsNone(st.predict(_clip(1, 1), sample_rate=48000))
        self.assertEqual((st.failed, calls), ("", []))
        self.assertIsNotNone(st.predict(_clip(1, 1).reshape(-1, 1)))


class EotWithModelsTests(unittest.TestCase):
    """The decider over the real SileroStream / SmartTurn classes (fake
    sessions): a model that cannot load, or has latched off, is 'rms'."""

    def _turn_with(self, stream, st, mode="on"):
        stream.reset()
        clip = []
        dec = ep.EotDecider(mode, vad=stream.feed,
                            predict=lambda: st.predict(np.concatenate(clip)))
        speech = np.full(1024, 0.3, np.float32)
        quiet = np.zeros(1024, np.float32)
        ended = None
        for i in range(20 + 21):
            loud = i < 20
            clip.append(speech if loud else quiet)
            if dec.update(clip[-1], 0 if loud else i - 19):
                ended = i - 19
                break
        return ended, dec.record()

    def test_both_models_working_end_the_turn_early(self):
        stream, _, _ = _stream()
        st, _, _, _ = _turn(_TurnSession(ps=0.9))
        ended, rec = self._turn_with(stream, st)
        self.assertEqual((ended, rec.eot, rec.st_n), (4, "st", 1))

    def test_a_load_failure_behaves_exactly_like_rms(self):
        def gone(p):
            raise OSError("gone")

        stream, _, _ = _stream()
        bad_st = ep.SmartTurn("x.onnx", session_factory=gone,
                              features=lambda x: x)
        bad_stream = ep.SileroStream(model_path="x.onnx",
                                     session_factory=gone)
        good_st, sess, _, _ = _turn(_TurnSession(ps=0.99))
        for s, st in ((stream, bad_st), (bad_stream, good_st)):
            ended, rec = self._turn_with(s, st)
            self.assertIsNone(ended)
            self.assertEqual(rec.stats()["eot"], "rms")
            self.assertIsNone(rec.st_p)
            self.assertIsNone(rec.fire_ms)
        self.assertEqual(sess.feeds, [])

    def test_never_once_latched_off(self):
        stream, _, _ = _stream()
        st, sess, _, _ = _turn(_TurnSession(ps=[1.5, 0.99]))
        ended, rec = self._turn_with(stream, st)    # this turn latches it
        self.assertIsNone(ended)
        self.assertIn("bad output", st.failed)
        for _ in range(3):                          # and every later turn
            ended, rec = self._turn_with(stream, st)
            self.assertIsNone(ended)
            self.assertEqual(rec.eot, "rms")
        self.assertEqual(len(sess.feeds), 1)
        # Too slow: two slow checks under the threshold, then a slow 0.99
        # that latches, so it does not end the turn; nor does any later.
        st, sess, _, _ = _turn(_TurnSession(costs=0.2, ps=[0.1, 0.1, 0.99]))
        ended, rec = self._turn_with(stream, st)
        self.assertIsNone(ended)
        self.assertEqual(rec.st_n, 3)
        self.assertIn("too slow", st.failed)
        self.assertIsNone(self._turn_with(stream, st)[0])
        self.assertEqual(len(sess.feeds), 3)

    def test_mode_off_never_imports_a_model_runtime(self):
        # A fresh interpreter whose import system records every attempt to
        # import onnxruntime / faster_whisper (installed or not). Mode 'off'
        # builds the real objects, resets the stream and runs a whole turn:
        # no attempt. The same objects in 'shadow' DO attempt it, so the spy
        # is live.
        code = r'''
import sys
sys.path.insert(0, sys.argv[1])
tried = []
class Spy:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("onnxruntime", "faster_whisper"):
            tried.append(name.split(".")[0])
        return None
sys.meta_path.insert(0, Spy())
import numpy as np
from core import endpointing as ep
def turn(mode):
    stream = ep.SileroStream(model_path="x.onnx")
    st = ep.SmartTurn("x.onnx")
    stream.reset()
    clip = []
    dec = ep.EotDecider(mode, vad=stream.feed,
                        predict=lambda: st.predict(np.concatenate(clip)))
    for i in range(41):
        clip.append(np.full(1024, 0.3 if i < 20 else 0.0, np.float32))
        dec.update(clip[-1], 0 if i < 20 else i - 19)
    return dec.record().eot
off = turn("off")
print(off, sorted(set(tried)))
shadow = turn("shadow")
print(shadow, sorted(set(tried)))
'''
        out = subprocess.run([sys.executable, "-c", code, _ROOT],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.strip().splitlines()
        self.assertEqual(lines[0], "rms []", out.stdout)
        self.assertTrue(lines[1].startswith("rms ["), out.stdout)
        self.assertIn("onnxruntime", lines[1], out.stdout)


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
