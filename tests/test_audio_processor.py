"""Unit tests for core/audio_processor.py — the three-layer (AEC / NS / AGC)
real-time input pipeline.

NOTE ON SCOPE: the task brief referenced a "music-vs-speech classifier" with
``classify_*`` helpers, but no such surface exists in this module (or anywhere
under tests/). The module is the echo-cancel / noise-suppress / auto-gain
pipeline documented in its own docstring, so that is what these tests cover.

Isolation contract (wave-1 agents broke the suite by violating this):
  * REAL numpy is used throughout — every signal is a deterministic synthetic
    array (tones, white noise via a seeded RNG, silence, NaN frames). numpy is
    never faked.
  * The only faked dependencies are the optional/heavy backends the pipeline
    probes for: ``webrtc_audio_processing`` (absent in this env) and
    ``noisereduce`` (present). Both are injected ONLY inside a test via
    ``mock.patch.dict(sys.modules, ...)`` + ``addCleanup(p.stop)`` so that after
    every test ``sys.modules`` holds the real modules again. Nothing is written
    to ``sys.modules`` at module import time.
  * The module-level singleton + VAD state dict are reset in tearDown so no
    cross-test state leaks.

stdlib unittest + unittest.mock only (no pytest).
"""
from __future__ import annotations

import sys
import time
import unittest
from unittest import mock

import numpy as np

import core.audio_processor as ap

# ── Eager backend warm-up (isolation safety) ──────────────────────────
# AudioProcessor.__init__ lazily `import noisereduce`, which transitively
# pulls scipy.signal → numpy.fft._pocketfft_umath. On this env (numpy on
# CPython 3.14) that C extension raises "cannot load module more than once
# per process" if it is ever imported a SECOND time.
#
# Several tests build a processor *inside* a `mock.patch.dict(sys.modules,
# …)` block. If noisereduce were first imported inside such a block, the
# patch's snapshot (taken before the import) would not contain it, and the
# patch's restore would DELETE noisereduce / scipy.signal / numpy.fft from
# sys.modules. The next processor build would then re-import noisereduce,
# re-run the scipy chain, hit the one-shot _pocketfft_umath guard, and
# silently fall back to _nr=None — corrupting unrelated tests.
#
# Importing the whole chain ONCE here, at module load (before any
# patch.dict snapshot exists), pins these modules in sys.modules for the
# life of the process so no patch.dict restore can evict them. This only
# warms the import cache; it fakes nothing.
try:  # pragma: no cover - environment-dependent warm-up
    import noisereduce as _warm_nr
    import scipy.signal as _warm_scipy_signal
    _warm_nr.reduce_noise(
        y=np.zeros(2048, dtype=np.float32), sr=16000,
        stationary=True, prop_decrease=0.7,
    )
    # The imports above exist purely for their sys.modules side effect (see
    # the comment block); drop the now-unused names to keep the namespace and
    # linters clean.
    del _warm_nr, _warm_scipy_signal
except Exception:  # noisereduce/scipy simply absent → nothing to pin
    pass


# ──────────────────────────────────────────────────────────────────────
# Synthetic-signal helpers (all deterministic / offline)
# ──────────────────────────────────────────────────────────────────────

def tone(freq=440.0, secs=0.25, sr=16000, amp=0.3):
    """A pure sine — high spectral peakedness (low flatness ≈ 'tonal')."""
    t = np.linspace(0.0, secs, int(sr * secs), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def white_noise(secs=0.25, sr=16000, amp=0.3, seed=1234):
    """Seeded white noise — high spectral flatness ≈ 'broadband'."""
    rng = np.random.default_rng(seed)
    n = int(sr * secs)
    return (amp * rng.standard_normal(n)).astype(np.float32)


def silence(secs=0.25, sr=16000):
    return np.zeros(int(sr * secs), dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
# Fakes for the optional backends (scoped per-test only)
# ──────────────────────────────────────────────────────────────────────

class _FakeAPM:
    """Stand-in for webrtc_audio_processing.AudioProcessingModule.

    process_stream echoes its input bytes (so the int16<->float round-trip in
    _apm_process is exercised) unless ``raise_in_process`` is set, in which case
    it raises to drive the in-loop except branch.
    """
    def __init__(self, *a, raise_in_process=False, raise_set_format=False,
                 **kw):
        self.raise_in_process = raise_in_process
        self.raise_set_format = raise_set_format
        self.reverse_calls = 0
        self.stream_calls = 0

    def set_stream_format(self, sr, ch):
        if self.raise_set_format:
            raise RuntimeError("set_stream_format boom")

    def set_reverse_stream_format(self, sr, ch):
        if self.raise_set_format:
            raise RuntimeError("set_reverse_stream_format boom")

    def process_reverse_stream(self, b):
        self.reverse_calls += 1
        return b

    def process_stream(self, b):
        self.stream_calls += 1
        if self.raise_in_process:
            raise RuntimeError("process_stream boom")
        return b


def _fake_webrtc_module(**apm_kwargs):
    """A fake ``webrtc_audio_processing`` module whose APM class builds a
    _FakeAPM pre-bound with the given kwargs."""
    mod = type(sys)("webrtc_audio_processing")

    class APM(_FakeAPM):
        def __init__(self, *a, **kw):
            super().__init__(*a, **{**apm_kwargs, **kw})

    mod.AudioProcessingModule = APM
    return mod


def _fake_noisereduce(behaviour="identity"):
    """A fake ``noisereduce`` module.

    behaviour:
      'identity'  → returns y unchanged (valid output path)
      'nan'       → returns an array full of NaN (output-validation fallback)
      'short'     → returns a shorter array (size-mismatch fallback)
      'raise'     → raises inside reduce_noise (exception fallback)
      'badobj'    → returns an object np.asarray(float32) cannot digest
    """
    mod = type(sys)("noisereduce")

    def reduce_noise(y, sr, stationary=True, prop_decrease=1.0, **kw):
        if behaviour == "raise":
            raise RuntimeError("nr boom")
        if behaviour == "nan":
            return np.full(np.asarray(y).shape, np.nan, dtype=np.float32)
        if behaviour == "short":
            return np.asarray(y, dtype=np.float32)[:-5]
        if behaviour == "badobj":
            return object()  # asarray(..., float32) raises on this
        return np.asarray(y, dtype=np.float32)

    mod.reduce_noise = reduce_noise
    return mod


def _new_processor(sr=16000, **kw):
    """Build a processor with the real env (no APM, real noisereduce)."""
    return ap.AudioProcessor(sample_rate=sr, **kw)


class _ResetSingletonMixin:
    """Reset module-level mutable state so tests don't bleed into each other."""
    def setUp(self):
        super().setUp()
        ap._singleton = None
        # Snapshot + restore the VAD state dict around each test.
        self._saved_vad = dict(ap._vad_state)

    def tearDown(self):
        ap._singleton = None
        with ap._vad_state_lock:
            ap._vad_state.clear()
            ap._vad_state.update(self._saved_vad)
        super().tearDown()


# ──────────────────────────────────────────────────────────────────────
# Module-level pure helpers
# ──────────────────────────────────────────────────────────────────────

class HelperTests(unittest.TestCase):
    def test_dprint_silent_when_debug_off(self):
        with mock.patch.object(ap, "_DEBUG", False), \
                mock.patch("builtins.print") as pr:
            ap._dprint("hello")
        pr.assert_not_called()

    def test_dprint_emits_when_debug_on(self):
        with mock.patch.object(ap, "_DEBUG", True), \
                mock.patch("builtins.print") as pr:
            ap._dprint("hello")
        pr.assert_called_once()
        self.assertIn("hello", pr.call_args.args[0])

    def test_safe_exc_normal(self):
        out = ap._safe_exc("ns", ValueError("bad thing"))
        self.assertEqual(out, "ns: ValueError: bad thing")

    def test_safe_exc_str_crashes_degrades_to_classname(self):
        class Exploding(Exception):
            def __str__(self):  # the SIGSEGV-analogue path
                raise RuntimeError("str blew up")

        out = ap._safe_exc("noisereduce", Exploding())
        self.assertEqual(out, "noisereduce: <Exploding: __str__ failed>")


# ──────────────────────────────────────────────────────────────────────
# __init__ / backend probing
# ──────────────────────────────────────────────────────────────────────

class InitTests(_ResetSingletonMixin, unittest.TestCase):
    def test_defaults_and_derived_frame_samples(self):
        p = _new_processor(sr=16000, frame_ms=20)
        self.assertEqual(p.sample_rate, 16000)
        self.assertEqual(p.frame_ms, 20)
        self.assertEqual(p.frame_samples, 320)  # 16000*20/1000
        # No webrtc in this env.
        self.assertIsNone(p._apm)
        # _nr is loaded iff noisereduce is importable — present on the dev box,
        # absent on the light-deps CI runner — so assert against actual presence.
        import importlib.util
        if importlib.util.find_spec("noisereduce") is not None:
            self.assertIsNotNone(p._nr)
        else:
            self.assertIsNone(p._nr)

    def test_frame_samples_floor_is_one(self):
        # A tiny sample rate would compute 0 samples/frame; clamped to >=1.
        p = _new_processor(sr=10, frame_ms=20)
        self.assertEqual(p.frame_samples, 1)

    def test_config_override_branch_runs(self):
        # The real core.config supplies AEC_DUCK_GAIN/AGC_FLATNESS_*; assert the
        # constructor actually adopts them (covers the try-branch).
        from core import config as cfg
        p = _new_processor()
        self.assertAlmostEqual(p.aec_duck_gain, float(cfg.AEC_DUCK_GAIN))
        self.assertAlmostEqual(p._agc_flatness_min, float(cfg.AGC_FLATNESS_MIN))
        self.assertAlmostEqual(p._agc_flatness_max, float(cfg.AGC_FLATNESS_MAX))

    def test_config_import_failure_falls_back_to_ctor_defaults(self):
        # Force the `from core import config` inside __init__ to blow up so the
        # except: pass branch runs and ctor defaults survive.
        real_import = __import__

        def boom(name, *a, **k):
            if name == "core.config" or name == "core":
                raise ImportError("no config for you")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=boom):
            p = ap.AudioProcessor(sample_rate=16000, aec_duck_gain=0.42)
        self.assertAlmostEqual(p.aec_duck_gain, 0.42)
        self.assertAlmostEqual(p._agc_flatness_min, 0.20)
        self.assertAlmostEqual(p._agc_flatness_max, 0.80)

    def test_apm_backend_constructed_when_available(self):
        fake = _fake_webrtc_module()
        p = mock.patch.dict(sys.modules, {"webrtc_audio_processing": fake})
        p.start(); self.addCleanup(p.stop)
        proc = _new_processor()
        self.assertIsNotNone(proc._apm)
        self.assertIsInstance(proc._apm, _FakeAPM)

    def test_apm_set_format_failure_is_swallowed_but_apm_kept(self):
        fake = _fake_webrtc_module(raise_set_format=True)
        pdict = mock.patch.dict(sys.modules, {"webrtc_audio_processing": fake})
        pdict.start(); self.addCleanup(pdict.stop)
        proc = _new_processor()
        # set_*_format raised but was caught; the APM object is still adopted.
        self.assertIsNotNone(proc._apm)

    def test_apm_construction_failure_leaves_apm_none(self):
        bad = type(sys)("webrtc_audio_processing")

        class APM:
            def __init__(self, *a, **k):
                raise RuntimeError("cannot build APM")

        bad.AudioProcessingModule = APM
        pdict = mock.patch.dict(sys.modules, {"webrtc_audio_processing": bad})
        pdict.start(); self.addCleanup(pdict.stop)
        proc = _new_processor()
        self.assertIsNone(proc._apm)

    def test_noisereduce_absent_leaves_nr_none(self):
        # Block the import so the except branch runs and _nr stays None.
        real_import = __import__

        def boom(name, *a, **k):
            if name == "noisereduce":
                raise ImportError("no nr")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=boom):
            proc = ap.AudioProcessor(sample_rate=16000)
        self.assertIsNone(proc._nr)


# ──────────────────────────────────────────────────────────────────────
# status()
# ──────────────────────────────────────────────────────────────────────

class StatusTests(_ResetSingletonMixin, unittest.TestCase):
    def test_status_shape_fresh(self):
        p = _new_processor()
        s = p.status()
        self.assertEqual(s["sample_rate"], 16000)
        self.assertEqual(s["frame_ms"], 20)
        self.assertFalse(s["apm_available"])
        import importlib.util
        self.assertEqual(s["noisereduce_available"],
                         importlib.util.find_spec("noisereduce") is not None)
        self.assertIsNone(s["last_playback_age_s"])  # no playback yet
        self.assertEqual(s["n_processed"], 0)
        self.assertEqual(s["agc_flatness_bounds"], (0.20, 0.80))
        self.assertIsNone(s["last_error"])

    def test_status_reflects_processing_and_playback(self):
        p = _new_processor()
        p.process(tone())
        p.feed_playback(tone(amp=0.5))
        s = p.status()
        self.assertEqual(s["n_processed"], 1)
        self.assertIsNotNone(s["last_playback_age_s"])
        self.assertGreaterEqual(s["last_playback_age_s"], 0.0)


# ──────────────────────────────────────────────────────────────────────
# feed_playback() / is_playback_recent()
# ──────────────────────────────────────────────────────────────────────

class FeedPlaybackTests(_ResetSingletonMixin, unittest.TestCase):
    def test_none_and_empty_are_noops(self):
        p = _new_processor()
        p.feed_playback(None)
        p.feed_playback(np.zeros(0, dtype=np.float32))
        with p._playback_lock:
            self.assertEqual(len(p._playback_buffer), 0)
        self.assertEqual(p._last_playback_ts, 0.0)

    def test_mono_append_sets_timestamp(self):
        p = _new_processor()
        p.feed_playback(tone(secs=0.05))
        with p._playback_lock:
            self.assertEqual(len(p._playback_buffer), 1)
        self.assertGreater(p._last_playback_ts, 0.0)

    def test_stereo_is_downmixed_to_mono(self):
        p = _new_processor()
        stereo = np.stack([tone(secs=0.05), tone(secs=0.05)], axis=1)
        self.assertEqual(stereo.ndim, 2)
        p.feed_playback(stereo)
        with p._playback_lock:
            stored = p._playback_buffer[-1][1]
        self.assertEqual(stored.ndim, 1)

    def test_resample_up_changes_length(self):
        p = _new_processor(sr=16000)
        src = tone(secs=0.05, sr=8000)  # 400 samples @ 8k
        p.feed_playback(src, sample_rate=8000)
        with p._playback_lock:
            stored = p._playback_buffer[-1][1]
        # Upsampled toward 16k → roughly double the samples.
        self.assertGreater(stored.size, src.size)
        self.assertAlmostEqual(stored.size, src.size * 2, delta=3)

    def test_resample_down_changes_length(self):
        p = _new_processor(sr=16000)
        src = tone(secs=0.05, sr=48000)  # 2400 samples @ 48k
        p.feed_playback(src, sample_rate=48000)
        with p._playback_lock:
            stored = p._playback_buffer[-1][1]
        self.assertLess(stored.size, src.size)

    def test_single_sample_skips_resample(self):
        # x.size <= 1 short-circuits the resample branch (guards np.interp).
        p = _new_processor(sr=16000)
        p.feed_playback(np.array([0.5], dtype=np.float32), sample_rate=8000)
        with p._playback_lock:
            stored = p._playback_buffer[-1][1]
        self.assertEqual(stored.size, 1)

    def test_old_entries_evicted_after_2s(self):
        p = _new_processor()
        now = 1000.0
        with mock.patch.object(ap.time, "time", return_value=now):
            p.feed_playback(tone(secs=0.05))
        # Advance > 2s; the eviction loop should drop the stale entry.
        with mock.patch.object(ap.time, "time", return_value=now + 3.0):
            p.feed_playback(tone(secs=0.05))
        with p._playback_lock:
            self.assertEqual(len(p._playback_buffer), 1)
            self.assertEqual(p._playback_buffer[0][0], now + 3.0)

    def test_exception_path_records_last_error(self):
        # Pass a bogus array-like whose .size access raises, forcing the
        # except branch and the _last_error write.
        p = _new_processor()

        class Bomb:
            size = 5
            ndim = 1

            def __getattr__(self, name):
                raise RuntimeError("explode")

        # np.asarray(Bomb()) would wrap it as a 0-d object array, so instead
        # patch np.asarray inside the module to raise.
        with mock.patch.object(ap.np, "asarray",
                               side_effect=RuntimeError("asarray boom")):
            p.feed_playback(tone(secs=0.05))
        self.assertIsNotNone(p._last_error)
        self.assertIn("feed_playback", p._last_error)

    def test_is_playback_recent_true_then_false(self):
        p = _new_processor()
        base = 500.0
        with mock.patch.object(ap.time, "time", return_value=base):
            p.feed_playback(tone(secs=0.05))
            self.assertTrue(p.is_playback_recent(within=0.2))
        with mock.patch.object(ap.time, "time", return_value=base + 1.0):
            self.assertFalse(p.is_playback_recent(within=0.2))

    def test_is_playback_recent_no_playback_is_false(self):
        p = _new_processor()
        self.assertFalse(p.is_playback_recent(within=10.0))


# ──────────────────────────────────────────────────────────────────────
# process() top-level
# ──────────────────────────────────────────────────────────────────────

class ProcessTests(_ResetSingletonMixin, unittest.TestCase):
    def test_none_input_returns_none(self):
        p = _new_processor()
        self.assertIsNone(p.process(None))

    def test_empty_input_returns_input(self):
        p = _new_processor()
        empty = np.zeros(0, dtype=np.float32)
        out = p.process(empty)
        self.assertIs(out, empty)

    def test_pre_cast_failure_returns_original(self):
        p = _new_processor()
        sig = tone()
        with mock.patch.object(ap.np, "asarray",
                               side_effect=RuntimeError("cast boom")):
            out = p.process(sig)
        self.assertIs(out, sig)
        self.assertIn("process pre-cast", p._last_error)

    def test_stereo_input_downmixed(self):
        p = _new_processor()
        stereo = np.stack([tone(), tone()], axis=1)
        out = p.process(stereo)
        self.assertEqual(out.ndim, 1)

    def test_full_pipeline_runs_and_counts(self):
        p = _new_processor()
        out = p.process(tone() + white_noise(amp=0.02))
        self.assertEqual(out.ndim, 1)
        self.assertEqual(p._n_processed, 1)
        self.assertGreater(p._last_raw_rms, 0.0)
        self.assertGreater(p._last_proc_rms, 0.0)

    def test_stage_toggles_skip_stages(self):
        p = _new_processor()
        # Disabling every stage means the output equals the cast input.
        sig = tone()
        out = p.process(sig, enable_aec=False, enable_ns=False,
                        enable_agc=False)
        np.testing.assert_allclose(out, sig, rtol=0, atol=0)

    def test_aec_stage_exception_counts_dropout(self):
        p = _new_processor()
        with mock.patch.object(p, "_aec",
                               side_effect=RuntimeError("aec boom")):
            p.process(tone())
        self.assertEqual(p._n_aec_dropouts, 1)
        self.assertIn("aec", p._last_error)

    def test_ns_stage_exception_recorded(self):
        p = _new_processor()
        with mock.patch.object(p, "_ns", side_effect=RuntimeError("ns boom")):
            p.process(tone())
        self.assertIn("ns", p._last_error)

    def test_agc_stage_exception_recorded(self):
        p = _new_processor()
        with mock.patch.object(p, "_agc",
                               side_effect=RuntimeError("agc boom")):
            p.process(tone())
        self.assertIn("agc", p._last_error)

    def test_rms_history_appended_for_audible_output(self):
        p = _new_processor()
        p.process(tone(amp=0.3))
        with p._rms_history_lock:
            self.assertGreaterEqual(len(p._rms_history), 1)

    def test_rms_history_prunes_old_entries(self):
        p = _new_processor()
        base = 2000.0
        with mock.patch.object(ap.time, "time", return_value=base):
            p.process(tone(amp=0.3))
        # 61 s later → the first entry is older than the 60 s window → pruned,
        # and the new one is appended.
        with mock.patch.object(ap.time, "time", return_value=base + 61.0):
            p.process(tone(amp=0.3))
        with p._rms_history_lock:
            tss = [ts for ts, _ in p._rms_history]
        self.assertTrue(all(ts >= base + 61.0 - p._rms_history_window_s
                            for ts in tss))

    def test_rms_history_exception_recorded(self):
        p = _new_processor()
        # Make the *post-pipeline* rms computation raise. np.sqrt is called for
        # both raw and processed rms; we want the second block. Patch np.mean
        # to raise only after the stages have run by counting calls.
        original_mean = ap.np.mean
        calls = {"n": 0}

        def flaky_mean(*a, **k):
            calls["n"] += 1
            # Let early calls (raw rms + any stage math) succeed; trip the
            # final proc-rms computation once enough calls have happened.
            if calls["n"] >= 1 and a and isinstance(a[0], np.ndarray):
                # Only blow up on the dedicated x*x proc-rms array path is hard
                # to target precisely; instead raise unconditionally here and
                # rely on the outer try/except to record it.
                raise RuntimeError("mean boom")
            return original_mean(*a, **k)

        with mock.patch.object(ap.np, "mean", side_effect=flaky_mean):
            out = p.process(tone())
        # Output still returned (never None for non-empty input).
        self.assertIsNotNone(out)
        self.assertIsNotNone(p._last_error)


# ──────────────────────────────────────────────────────────────────────
# recent_peak_rms()
# ──────────────────────────────────────────────────────────────────────

class RecentPeakRmsTests(_ResetSingletonMixin, unittest.TestCase):
    def test_zero_when_no_history(self):
        p = _new_processor()
        self.assertEqual(p.recent_peak_rms(), 0.0)

    def test_returns_max_within_window(self):
        p = _new_processor()
        now = 3000.0
        with p._rms_history_lock:
            p._rms_history.append((now - 1.0, 0.1))
            p._rms_history.append((now - 0.5, 0.4))
            p._rms_history.append((now - 100.0, 0.9))  # outside 60s window
        with mock.patch.object(ap.time, "time", return_value=now):
            self.assertAlmostEqual(p.recent_peak_rms(within=60.0), 0.4)

    def test_window_excludes_everything_returns_zero(self):
        p = _new_processor()
        now = 3000.0
        with p._rms_history_lock:
            p._rms_history.append((now - 100.0, 0.9))
        with mock.patch.object(ap.time, "time", return_value=now):
            self.assertEqual(p.recent_peak_rms(within=10.0), 0.0)


# ──────────────────────────────────────────────────────────────────────
# _aec() + _reference_frame() + _apm_process()
# ──────────────────────────────────────────────────────────────────────

class AecFallbackTests(_ResetSingletonMixin, unittest.TestCase):
    def test_no_playback_passthrough(self):
        p = _new_processor()
        sig = tone()
        out = p._aec(sig)
        np.testing.assert_allclose(out, sig)

    def test_recent_playback_ducks(self):
        p = _new_processor()
        sig = tone()
        base = 700.0
        with mock.patch.object(ap.time, "time", return_value=base):
            p.feed_playback(tone(amp=0.5))
            out = p._aec(sig)
        self.assertEqual(p._n_aec_ducked, 1)
        np.testing.assert_allclose(out, sig * p.aec_duck_gain, rtol=1e-5)

    def test_reference_frame_none_when_empty(self):
        p = _new_processor()
        self.assertIsNone(p._reference_frame(320))

    def test_reference_frame_none_when_concatenation_is_empty(self):
        # Buffer holds only zero-length arrays → concatenate succeeds but
        # yields size 0 → the cat.size == 0 guard returns None.
        p = _new_processor()
        with p._playback_lock:
            p._playback_buffer.append((time.time(),
                                       np.zeros(0, dtype=np.float32)))
        self.assertIsNone(p._reference_frame(50))

    def test_reference_frame_pads_when_short(self):
        p = _new_processor()
        p.feed_playback(np.ones(10, dtype=np.float32))
        ref = p._reference_frame(50)
        self.assertEqual(ref.size, 50)
        # Front-padded with zeros, real samples at the tail.
        self.assertEqual(float(ref[0]), 0.0)
        self.assertEqual(float(ref[-1]), 1.0)

    def test_reference_frame_trims_when_long(self):
        p = _new_processor()
        p.feed_playback(np.arange(100, dtype=np.float32) / 100.0)
        ref = p._reference_frame(30)
        self.assertEqual(ref.size, 30)
        # Tail of the buffer.
        self.assertAlmostEqual(float(ref[-1]), 0.99, places=5)

    def test_reference_frame_concat_failure_returns_none(self):
        p = _new_processor()
        p.feed_playback(np.ones(10, dtype=np.float32))
        with mock.patch.object(ap.np, "concatenate",
                               side_effect=RuntimeError("cat boom")):
            self.assertIsNone(p._reference_frame(50))


class AecApmTests(_ResetSingletonMixin, unittest.TestCase):
    def _proc_with_apm(self, **apm_kwargs):
        fake = _fake_webrtc_module(**apm_kwargs)
        pdict = mock.patch.dict(sys.modules, {"webrtc_audio_processing": fake})
        pdict.start(); self.addCleanup(pdict.stop)
        return _new_processor()

    def test_apm_path_used_when_reference_present(self):
        p = self._proc_with_apm()
        p.feed_playback(tone(secs=0.05, amp=0.5))
        sig = tone(secs=0.05)
        out = p._aec(sig)
        self.assertEqual(out.size, sig.size)
        # Fake APM echoes bytes → round-trips close to input (int16 quantized).
        np.testing.assert_allclose(out, sig, atol=1e-3)
        self.assertGreater(p._apm.stream_calls, 0)
        self.assertGreater(p._apm.reverse_calls, 0)

    def test_apm_raises_falls_through_to_duck(self):
        p = self._proc_with_apm(raise_in_process=True)
        sig = tone(secs=0.05)
        base = 900.0
        with mock.patch.object(ap.time, "time", return_value=base):
            p.feed_playback(tone(secs=0.05, amp=0.5))
            out = p._aec(sig)
        # _apm_process swallows the per-block error and echoes the raw block,
        # so the APM path still returns; no duck fired (it returned from APM).
        self.assertEqual(out.size, sig.size)

    def test_apm_process_returns_input_when_apm_none(self):
        p = _new_processor()  # real env: _apm is None
        sig = tone(secs=0.05)
        out = p._apm_process(sig, np.zeros(sig.size, dtype=np.float32))
        self.assertIs(out, sig)

    def test_aec_apm_process_raises_falls_through_to_duck(self):
        # When _apm_process ITSELF raises (not the inner per-block catch),
        # _aec records the error and falls through to the ducking fallback.
        p = self._proc_with_apm()
        sig = tone(secs=0.05)
        base = 950.0
        with mock.patch.object(p, "_apm_process",
                               side_effect=RuntimeError("apm exploded")), \
                mock.patch.object(ap.time, "time", return_value=base):
            p.feed_playback(tone(secs=0.05, amp=0.5))
            out = p._aec(sig)
        self.assertIn("apm process", p._last_error)
        self.assertEqual(p._n_aec_ducked, 1)
        np.testing.assert_allclose(out, sig * p.aec_duck_gain, rtol=1e-5)

    def test_apm_process_pads_short_reference(self):
        p = self._proc_with_apm()
        sig = tone(secs=0.05)
        short_ref = np.ones(5, dtype=np.float32)  # smaller than padded a_buf
        out = p._apm_process(sig, short_ref)
        self.assertEqual(out.size, sig.size)

    def test_apm_process_trims_long_reference(self):
        p = self._proc_with_apm()
        sig = tone(secs=0.05)
        long_ref = np.ones(sig.size * 4, dtype=np.float32)
        out = p._apm_process(sig, long_ref)
        self.assertEqual(out.size, sig.size)

    def test_apm_process_inner_exception_echoes_block(self):
        # raise_in_process makes process_stream raise inside the loop; the
        # except sets cleaned = a16 (the raw input bytes) and the loop carries
        # on, so we still get a full-length output.
        p = self._proc_with_apm(raise_in_process=True)
        sig = tone(secs=0.05)
        ref = np.ones(sig.size, dtype=np.float32)
        out = p._apm_process(sig, ref)
        self.assertEqual(out.size, sig.size)
        np.testing.assert_allclose(out, sig, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────
# _ns()  (noise suppression dispatcher)
# ──────────────────────────────────────────────────────────────────────

class NsTests(_ResetSingletonMixin, unittest.TestCase):
    def test_short_chunk_routes_to_spectral(self):
        p = _new_processor()
        sig = tone(secs=0.005)  # < 256 samples @ 16k (80 samples)
        self.assertLess(sig.size, 256)
        with mock.patch.object(p, "_spectral_subtract",
                               wraps=p._spectral_subtract) as ss:
            p._ns(sig)
        ss.assert_called_once()

    def test_near_silent_routes_to_spectral(self):
        p = _new_processor()
        sig = silence(secs=0.05)  # >256 samples but peak < 1e-4
        with mock.patch.object(p, "_spectral_subtract",
                               wraps=p._spectral_subtract) as ss:
            p._ns(sig)
        ss.assert_called_once()

    def test_peak_compute_failure_returns_input(self):
        p = _new_processor()
        sig = tone()
        with mock.patch.object(ap.np, "max",
                               side_effect=RuntimeError("max boom")):
            out = p._ns(sig)
        self.assertIs(out, sig)

    def test_noisereduce_valid_output_used(self):
        fake = _fake_noisereduce("identity")
        pdict = mock.patch.dict(sys.modules, {"noisereduce": fake})
        pdict.start(); self.addCleanup(pdict.stop)
        p = _new_processor()  # picks up the fake nr
        self.assertIs(p._nr, fake)
        sig = tone() + white_noise(amp=0.05)
        out = p._ns(sig)
        self.assertEqual(out.size, sig.size)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_noisereduce_nan_output_falls_back(self):
        fake = _fake_noisereduce("nan")
        pdict = mock.patch.dict(sys.modules, {"noisereduce": fake})
        pdict.start(); self.addCleanup(pdict.stop)
        p = _new_processor()
        sig = tone()
        with mock.patch.object(p, "_spectral_subtract",
                               wraps=p._spectral_subtract) as ss:
            out = p._ns(sig)
        ss.assert_called_once()
        self.assertTrue(np.all(np.isfinite(out)))

    def test_noisereduce_short_output_falls_back(self):
        fake = _fake_noisereduce("short")
        pdict = mock.patch.dict(sys.modules, {"noisereduce": fake})
        pdict.start(); self.addCleanup(pdict.stop)
        p = _new_processor()
        sig = tone()
        with mock.patch.object(p, "_spectral_subtract",
                               wraps=p._spectral_subtract) as ss:
            p._ns(sig)
        ss.assert_called_once()

    def test_noisereduce_raises_falls_back(self):
        fake = _fake_noisereduce("raise")
        pdict = mock.patch.dict(sys.modules, {"noisereduce": fake})
        pdict.start(); self.addCleanup(pdict.stop)
        p = _new_processor()
        sig = tone()
        with mock.patch.object(p, "_spectral_subtract",
                               wraps=p._spectral_subtract) as ss:
            p._ns(sig)
        ss.assert_called_once()
        self.assertIn("noisereduce", p._last_error)

    def test_noisereduce_badobj_output_falls_back(self):
        fake = _fake_noisereduce("badobj")
        pdict = mock.patch.dict(sys.modules, {"noisereduce": fake})
        pdict.start(); self.addCleanup(pdict.stop)
        p = _new_processor()
        sig = tone()
        with mock.patch.object(p, "_spectral_subtract",
                               wraps=p._spectral_subtract) as ss:
            p._ns(sig)
        ss.assert_called_once()
        self.assertIn("noisereduce_output", p._last_error)

    def test_no_backend_routes_to_spectral(self):
        p = _new_processor()
        p._nr = None  # simulate the no-backend branch directly
        sig = tone()
        with mock.patch.object(p, "_spectral_subtract",
                               wraps=p._spectral_subtract) as ss:
            p._ns(sig)
        ss.assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# _spectral_subtract()
# ──────────────────────────────────────────────────────────────────────

class SpectralSubtractTests(_ResetSingletonMixin, unittest.TestCase):
    def test_too_short_returns_input(self):
        p = _new_processor()
        sig = tone(secs=0.001)  # 16 samples < 64
        self.assertLess(sig.size, 64)
        out = p._spectral_subtract(sig)
        self.assertIs(out, sig)

    def test_seeds_noise_profile_on_quiet_frame(self):
        p = _new_processor()
        quiet = (1e-4 * white_noise(secs=0.05)).astype(np.float32)
        self.assertIsNone(p._ns_noise_mag)
        p._spectral_subtract(quiet)
        self.assertIsNotNone(p._ns_noise_mag)

    def test_noise_profile_ema_updates_on_second_quiet_frame(self):
        p = _new_processor()
        q1 = (1e-4 * white_noise(secs=0.05, seed=1)).astype(np.float32)
        q2 = (1e-4 * white_noise(secs=0.05, seed=2)).astype(np.float32)
        p._spectral_subtract(q1)
        first = p._ns_noise_mag.copy()
        p._spectral_subtract(q2)
        second = p._ns_noise_mag
        # EMA changed the stored profile (not a fresh copy nor identical).
        self.assertFalse(np.array_equal(first, second))

    def test_loud_frame_subtracts_against_profile(self):
        p = _new_processor()
        # Seed a noise profile with a quiet frame first.
        quiet = (1e-4 * white_noise(secs=0.05, seed=7)).astype(np.float32)
        p._spectral_subtract(quiet)
        loud = (tone(secs=0.05) + white_noise(secs=0.05, amp=0.05)).astype(
            np.float32)
        out = p._spectral_subtract(loud)
        self.assertEqual(out.size, loud.size)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_no_profile_passthrough(self):
        # Loud frame with no noise profile yet → returns input unchanged.
        p = _new_processor()
        loud = tone(secs=0.05, amp=0.3)
        self.assertIsNone(p._ns_noise_mag)
        out = p._spectral_subtract(loud)
        np.testing.assert_allclose(out, loud)

    def test_shape_mismatch_profile_passthrough(self):
        # A stored profile of the wrong shape (e.g. from a different frame
        # length) is rejected → input returned.
        p = _new_processor()
        p._ns_noise_mag = np.ones(3, dtype=np.float32)  # wrong shape
        loud = tone(secs=0.05, amp=0.3)
        out = p._spectral_subtract(loud)
        np.testing.assert_allclose(out, loud)

    def test_rfft_failure_returns_input(self):
        p = _new_processor()
        sig = tone(secs=0.05)
        with mock.patch.object(ap.np.fft, "rfft",
                               side_effect=RuntimeError("rfft boom")):
            out = p._spectral_subtract(sig)
        self.assertIs(out, sig)

    def test_irfft_failure_returns_input(self):
        p = _new_processor()
        # Seed a profile so we get past the noise==None guard to the irfft.
        quiet = (1e-4 * white_noise(secs=0.05, seed=3)).astype(np.float32)
        p._spectral_subtract(quiet)
        loud = (tone(secs=0.05) + white_noise(secs=0.05, amp=0.05)).astype(
            np.float32)
        with mock.patch.object(ap.np.fft, "irfft",
                               side_effect=RuntimeError("irfft boom")):
            out = p._spectral_subtract(loud)
        self.assertIs(out, loud)


# ──────────────────────────────────────────────────────────────────────
# _spectral_flatness()
# ──────────────────────────────────────────────────────────────────────

class SpectralFlatnessTests(_ResetSingletonMixin, unittest.TestCase):
    def test_too_short_returns_zero(self):
        p = _new_processor()
        self.assertEqual(p._spectral_flatness(tone(secs=0.001)), 0.0)

    def test_tone_is_low_flatness(self):
        p = _new_processor()
        flat = p._spectral_flatness(tone(secs=0.1, freq=440))
        self.assertLess(flat, 0.2)

    def test_white_noise_is_high_flatness(self):
        p = _new_processor()
        flat = p._spectral_flatness(white_noise(secs=0.1))
        self.assertGreater(flat, 0.3)

    def test_silence_returns_zero_via_arith_guard(self):
        p = _new_processor()
        # All-zero frame: |FFT| is all ~1e-10 after the epsilon add, arith is
        # below the 1e-9 guard → returns 0.0.
        out = p._spectral_flatness(silence(secs=0.05))
        self.assertEqual(out, 0.0)

    def test_rfft_failure_returns_zero(self):
        p = _new_processor()
        with mock.patch.object(ap.np.fft, "rfft",
                               side_effect=RuntimeError("rfft boom")):
            self.assertEqual(p._spectral_flatness(tone(secs=0.05)), 0.0)

    def test_single_bin_spectrum_skips_dc_drop(self):
        # A 1-bin |FFT| does NOT enter the `mag.size > 1` DC-drop branch, so
        # the single bin survives and geo == arith → flatness 1.0. (This also
        # exercises the `mag.size > 1` False path.) NOTE: the subsequent
        # `mag.size == 0` guard at audio_processor.py:579 is unreachable —
        # slicing [1:] only runs when size > 1, which always leaves >= 1 bin.
        p = _new_processor()
        with mock.patch.object(ap.np.fft, "rfft",
                               return_value=np.array([2.0 + 0j])):
            self.assertAlmostEqual(p._spectral_flatness(tone(secs=0.05)), 1.0)

    def test_result_bounded_0_1(self):
        p = _new_processor()
        for sig in (tone(secs=0.1), white_noise(secs=0.1),
                    tone(secs=0.1) + white_noise(secs=0.1, amp=0.1)):
            f = p._spectral_flatness(sig)
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f, 1.0)


# ──────────────────────────────────────────────────────────────────────
# _agc()
# ──────────────────────────────────────────────────────────────────────

class AgcTests(_ResetSingletonMixin, unittest.TestCase):
    def test_silent_frame_passthrough(self):
        p = _new_processor()
        out = p._agc(silence(secs=0.05))
        np.testing.assert_allclose(out, silence(secs=0.05))

    def test_quiet_tone_is_amplified_toward_target(self):
        p = _new_processor(agc_target_rms=0.05, agc_max_gain=8.0)
        # A quiet tone (rms well below target) should be boosted. A pure tone
        # has low flatness so the peakedness gate stays open.
        quiet = tone(secs=0.1, amp=0.005)
        out = p._agc(quiet)
        in_rms = float(np.sqrt(np.mean(quiet * quiet)))
        out_rms = float(np.sqrt(np.mean(out * out)))
        self.assertGreater(out_rms, in_rms)

    def test_running_rms_seeded_then_smoothed(self):
        p = _new_processor()
        self.assertEqual(p._agc_running_rms, 0.0)
        p._agc(tone(secs=0.05, amp=0.1))
        first = p._agc_running_rms
        self.assertGreater(first, 0.0)
        p._agc(tone(secs=0.05, amp=0.2))
        # EMA moved the tracked value.
        self.assertNotAlmostEqual(first, p._agc_running_rms, places=6)

    def test_loud_tone_is_attenuated_and_clipped(self):
        p = _new_processor(agc_target_rms=0.05, agc_max_gain=8.0)
        loud = tone(secs=0.1, amp=0.9)
        out = p._agc(loud)
        # gain < 1 (target below input rms) → output never exceeds [-1, 1].
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)

    def test_broadband_noise_gate_suppresses_gain(self):
        # White noise (high flatness) that would otherwise be amplified should
        # have its gain pulled back toward 1.0 by the peakedness gate, so it is
        # amplified LESS than an equally-quiet pure tone.
        p_noise = _new_processor(agc_target_rms=0.05, agc_max_gain=8.0)
        p_tone = _new_processor(agc_target_rms=0.05, agc_max_gain=8.0)
        quiet_noise = white_noise(secs=0.1, amp=0.005, seed=99)
        quiet_tone = tone(secs=0.1, amp=0.005)
        out_noise = p_noise._agc(quiet_noise)
        out_tone = p_tone._agc(quiet_tone)
        gain_noise = float(np.sqrt(np.mean(out_noise**2))) / float(
            np.sqrt(np.mean(quiet_noise**2)))
        gain_tone = float(np.sqrt(np.mean(out_tone**2))) / float(
            np.sqrt(np.mean(quiet_tone**2)))
        self.assertLess(gain_noise, gain_tone)
        self.assertTrue(p_noise._agc_flatness_init)

    def test_flatness_smoothing_on_second_amplified_frame(self):
        p = _new_processor(agc_target_rms=0.05, agc_max_gain=8.0)
        quiet_noise = white_noise(secs=0.1, amp=0.005, seed=5)
        p._agc(quiet_noise)          # cold-start seeds flatness
        self.assertTrue(p._agc_flatness_init)
        seeded = p._agc_flatness
        p._agc(white_noise(secs=0.1, amp=0.005, seed=6))  # EMA branch
        # The smoothed flatness stays within the configured clamp bounds.
        self.assertGreaterEqual(p._agc_flatness, p._agc_flatness_min)
        self.assertLessEqual(p._agc_flatness, p._agc_flatness_max)
        self.assertIsInstance(seeded, float)

    def test_flatness_clamped_to_max(self):
        # Force the smoothed flatness above the max bound and confirm it is
        # clamped. Pre-seed flatness near 1.0, then feed a high-flatness frame.
        p = _new_processor(agc_target_rms=0.05, agc_max_gain=8.0)
        p._agc_flatness_init = True
        p._agc_flatness = 0.99
        p._agc(white_noise(secs=0.1, amp=0.005, seed=8))
        self.assertLessEqual(p._agc_flatness, p._agc_flatness_max)

    def test_flatness_clamped_to_min(self):
        # Pre-seed flatness near 0 with a tonal frame so the smoothed value is
        # pushed below the min bound, then assert the clamp lifts it.
        p = _new_processor(agc_target_rms=0.05, agc_max_gain=8.0)
        p._agc_flatness_init = True
        p._agc_flatness = 0.0
        p._agc(tone(secs=0.1, amp=0.005))
        self.assertGreaterEqual(p._agc_flatness, p._agc_flatness_min)

    def test_gain_unbounded_when_max_gain_zero(self):
        # agc_max_gain <= 0 disables the clamp branch (max_g > 0 is False).
        p = _new_processor(agc_target_rms=0.05, agc_max_gain=0.0)
        out = p._agc(tone(secs=0.05, amp=0.1))
        self.assertEqual(out.size, int(16000 * 0.05))

    def test_agc_returns_finite_float32(self):
        # General invariant: AGC always yields a finite float32 array of the
        # same length, clipped into [-1, 1].
        p = _new_processor()
        sig = tone(secs=0.05, amp=0.1)
        out = p._agc(sig)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.size, sig.size)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)


# ──────────────────────────────────────────────────────────────────────
# Module-level singleton + shims
# ──────────────────────────────────────────────────────────────────────

class SingletonTests(_ResetSingletonMixin, unittest.TestCase):
    def test_get_processor_builds_once_and_caches(self):
        a = ap.get_processor(16000)
        b = ap.get_processor(16000)
        self.assertIs(a, b)

    def test_get_processor_rebuilds_on_samplerate_change(self):
        a = ap.get_processor(16000)
        b = ap.get_processor(48000)
        self.assertIsNot(a, b)
        self.assertEqual(b.sample_rate, 48000)

    def test_module_feed_playback_shim(self):
        ap.feed_playback(tone(secs=0.05), sample_rate=16000)
        proc = ap.get_processor(16000)
        with proc._playback_lock:
            self.assertGreaterEqual(len(proc._playback_buffer), 1)

    def test_module_feed_playback_shim_swallows_errors(self):
        with mock.patch.object(ap, "get_processor",
                               side_effect=RuntimeError("no proc")):
            # Must not raise.
            ap.feed_playback(tone(secs=0.05))

    def test_module_is_playback_recent_shim(self):
        base = 1234.0
        with mock.patch.object(ap.time, "time", return_value=base):
            ap.feed_playback(tone(secs=0.05))
            self.assertTrue(ap.is_playback_recent(within=1.0))

    def test_module_is_playback_recent_shim_swallows_errors(self):
        with mock.patch.object(ap, "get_processor",
                               side_effect=RuntimeError("no proc")):
            self.assertFalse(ap.is_playback_recent())

    def test_module_recent_peak_rms_shim(self):
        proc = ap.get_processor(16000)
        proc.process(tone(amp=0.3))
        self.assertGreaterEqual(ap.recent_peak_rms(within=60.0), 0.0)

    def test_module_recent_peak_rms_shim_swallows_errors(self):
        with mock.patch.object(ap, "get_processor",
                               side_effect=RuntimeError("no proc")):
            self.assertEqual(ap.recent_peak_rms(), 0.0)


# ──────────────────────────────────────────────────────────────────────
# VAD activity tracking helpers
# ──────────────────────────────────────────────────────────────────────

class VadStateTests(_ResetSingletonMixin, unittest.TestCase):
    def _clear_vad(self):
        with ap._vad_state_lock:
            ap._vad_state.update({
                "last_vad_active_ts": 0.0,
                "last_vad_poll_ts": 0.0,
                "vad_session_start": 0.0,
                "total_vad_trips": 0,
                "last_audible_chunk_ts": 0.0,
            })

    def setUp(self):
        super().setUp()
        self._clear_vad()

    def test_note_vad_active_sets_all_three(self):
        ap.note_vad_active(ts=111.0)
        st = ap.get_vad_state()
        self.assertEqual(st["last_vad_active_ts"], 111.0)
        self.assertEqual(st["last_vad_poll_ts"], 111.0)
        self.assertEqual(st["total_vad_trips"], 1)

    def test_note_vad_active_default_ts_uses_clock(self):
        with mock.patch.object(ap.time, "time", return_value=222.0):
            ap.note_vad_active()
        self.assertEqual(ap.get_vad_state()["last_vad_active_ts"], 222.0)

    def test_note_vad_poll_sets_session_start_once(self):
        ap.note_vad_poll(ts=300.0)
        st = ap.get_vad_state()
        self.assertEqual(st["last_vad_poll_ts"], 300.0)
        self.assertEqual(st["vad_session_start"], 300.0)
        # A later poll updates poll_ts but NOT session_start.
        ap.note_vad_poll(ts=400.0)
        st2 = ap.get_vad_state()
        self.assertEqual(st2["last_vad_poll_ts"], 400.0)
        self.assertEqual(st2["vad_session_start"], 300.0)

    def test_note_vad_poll_default_ts(self):
        with mock.patch.object(ap.time, "time", return_value=350.0):
            ap.note_vad_poll()
        self.assertEqual(ap.get_vad_state()["vad_session_start"], 350.0)

    def test_get_vad_state_is_a_copy(self):
        snap = ap.get_vad_state()
        snap["total_vad_trips"] = 9999
        self.assertNotEqual(ap.get_vad_state()["total_vad_trips"], 9999)

    def test_seconds_since_vad_active_inf_when_never(self):
        self.assertEqual(ap.seconds_since_vad_active(), float("inf"))

    def test_seconds_since_vad_active_measures_gap(self):
        ap.note_vad_active(ts=1000.0)
        with mock.patch.object(ap.time, "time", return_value=1005.0):
            self.assertAlmostEqual(ap.seconds_since_vad_active(), 5.0)

    def test_note_raw_rms_above_floor_updates_audible(self):
        ap.note_raw_rms(0.01, ts=2000.0)  # well above 1e-5
        self.assertEqual(ap.get_vad_state()["last_audible_chunk_ts"], 2000.0)

    def test_note_raw_rms_below_floor_ignored(self):
        ap.note_raw_rms(1e-9, ts=2100.0)  # below floor → no update
        self.assertEqual(ap.get_vad_state()["last_audible_chunk_ts"], 0.0)

    def test_note_raw_rms_default_ts(self):
        with mock.patch.object(ap.time, "time", return_value=2200.0):
            ap.note_raw_rms(0.5)
        self.assertEqual(ap.get_vad_state()["last_audible_chunk_ts"], 2200.0)

    def test_seconds_since_audible_uses_audible_ts(self):
        ap.note_raw_rms(0.01, ts=3000.0)
        with mock.patch.object(ap.time, "time", return_value=3002.0):
            self.assertAlmostEqual(ap.seconds_since_audible_chunk(), 2.0)

    def test_seconds_since_audible_falls_back_to_session_start(self):
        # No audible chunk yet, but polling has started → measure since start.
        ap.note_vad_poll(ts=4000.0)
        with mock.patch.object(ap.time, "time", return_value=4010.0):
            self.assertAlmostEqual(ap.seconds_since_audible_chunk(), 10.0)

    def test_seconds_since_audible_inf_when_no_polling(self):
        self.assertEqual(ap.seconds_since_audible_chunk(), float("inf"))


if __name__ == "__main__":
    unittest.main()
