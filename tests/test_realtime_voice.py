"""Unit tests for core/realtime_voice.py — the streaming STT+TTS pipeline.

Isolation contract (matches tests/test_audio_processor.py):
  * REAL numpy throughout — the only signals used are deterministic synthetic
    int16/float32 arrays built here. numpy is never faked.
  * RealtimeSTT / RealtimeTTS (and the SystemEngine / Azure / … engines they
    expose) are the heavy optional backends. In this env RealtimeTTS doesn't
    even import (it pulls pyaudio), and on the CI runner BOTH packages are
    absent. So every test that needs the pipeline "available" injects a FAKE
    RealtimeSTT/RealtimeTTS into sys.modules via ``mock.patch.dict`` +
    ``addCleanup(p.stop)`` — scoped per-test, auto-restored. Nothing is written
    to sys.modules at import time.
  * No real audio stream is ever opened, no real thread is relied upon: the
    STT pump loop and the playback drain are driven synchronously by calling
    the methods directly with the stop flag / playing flag pre-set.
  * The module-level singleton (_singleton) is reset in tearDown so no
    cross-test state leaks.

stdlib unittest + unittest.mock only (no pytest).
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np

import core.realtime_voice as rtv


# ──────────────────────────────────────────────────────────────────────
# Fakes for the RealtimeSTT / RealtimeTTS backends (scoped per-test only)
# ──────────────────────────────────────────────────────────────────────

class FakeTTSStream:
    """Stand-in for RealtimeTTS.TextToAudioStream.

    Records feed() text, tracks a playing flag, and lets each method be told
    to raise so the pipeline's except branches are exercised. ``accept_hook``
    controls whether play_async tolerates the on_audio_chunk kwarg (older
    RealtimeTTS builds raise TypeError on it → the no-kwarg fallback path).
    """

    def __init__(self, engine=None, accept_hook=True):
        self.engine = engine
        self.accept_hook = accept_hook
        self.fed = []
        self._playing = False
        self.play_async_calls = 0
        self.play_async_hooked = 0
        self.stop_calls = 0
        self.feed_raises = False
        self.is_playing_raises = False
        self.stop_raises = False
        # finalize/flush hook controls
        self.finalize_calls = 0
        self.finalize_raises = False

    def feed(self, text):
        if self.feed_raises:
            raise RuntimeError("feed boom")
        self.fed.append(text)

    def is_playing(self):
        if self.is_playing_raises:
            raise RuntimeError("is_playing boom")
        return self._playing

    def play_async(self, on_audio_chunk=None):
        if on_audio_chunk is not None:
            if not self.accept_hook:
                raise TypeError("play_async() got an unexpected keyword 'on_audio_chunk'")
            self.play_async_hooked += 1
        self.play_async_calls += 1
        self._playing = True

    def stop(self):
        self.stop_calls += 1
        if self.stop_raises:
            raise RuntimeError("stop boom")
        self._playing = False

    def finalize(self):
        self.finalize_calls += 1
        if self.finalize_raises:
            raise RuntimeError("finalize boom")


class FakeRecorder:
    """Stand-in for RealtimeSTT.AudioToTextRecorder.

    text() pops from a scripted queue of return values; each entry may be a
    string or an Exception instance (raised when reached). stop()/shutdown()
    are recorded so teardown can be asserted.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.script = []          # queue of str | Exception for text()
        self.text_calls = 0
        self.stop_calls = 0
        self.shutdown_calls = 0
        self.stop_raises = False

    def text(self):
        self.text_calls += 1
        if self.script:
            val = self.script.pop(0)
            if isinstance(val, BaseException):
                raise val
            return val
        return ""

    def stop(self):
        self.stop_calls += 1
        if self.stop_raises:
            raise RuntimeError("recorder stop boom")

    def shutdown(self):
        self.shutdown_calls += 1


class FakeEngine:
    """Stand-in for a RealtimeTTS engine (SystemEngine/AzureEngine/…)."""

    def __init__(self, *a, **kw):
        self.args = a
        self.kwargs = kw
        self.shutdown_calls = 0
        self.shutdown_raises = False

    def shutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_raises:
            raise RuntimeError("engine shutdown boom")


def _make_fake_realtimetts(
    *,
    engine_classes=None,
    stream_factory=None,
    stream_raises=False,
):
    """Build a fake ``RealtimeTTS`` module.

    engine_classes: dict mapping the engine attr name (SystemEngine, AzureEngine,
        ElevenlabsEngine, CoquiEngine) → a class/callable. Omitted names simply
        aren't present on the module (so ``from RealtimeTTS import X`` raises
        ImportError, which is what an uninstalled engine looks like).
    stream_factory: callable(engine) → stream object for TextToAudioStream.
    stream_raises: if True, TextToAudioStream(engine) raises (drives the
        _init_tts outer except).
    """
    mod = type(sys)("RealtimeTTS")

    def TextToAudioStream(engine):
        if stream_raises:
            raise RuntimeError("TextToAudioStream boom")
        if stream_factory is not None:
            return stream_factory(engine)
        return FakeTTSStream(engine)

    mod.TextToAudioStream = TextToAudioStream
    for name, cls in (engine_classes or {}).items():
        setattr(mod, name, cls)
    return mod


def _make_fake_realtimestt(*, recorder_factory=None, import_error=False):
    """Build a fake ``RealtimeSTT`` module exposing AudioToTextRecorder.

    recorder_factory: callable(**kwargs) → recorder. Default builds a FakeRecorder.
    import_error: if True the module lacks AudioToTextRecorder so
        ``from RealtimeSTT import AudioToTextRecorder`` raises ImportError.
    """
    mod = type(sys)("RealtimeSTT")
    if not import_error:
        def AudioToTextRecorder(**kwargs):
            if recorder_factory is not None:
                return recorder_factory(**kwargs)
            return FakeRecorder(**kwargs)
        mod.AudioToTextRecorder = AudioToTextRecorder
    return mod


def _patch_backends(testcase, *, stt=None, tts=None):
    """Inject fake RealtimeSTT/RealtimeTTS into sys.modules for one test.

    Pass a prebuilt fake module, or None to use a default-behaviour fake.
    Auto-restored via addCleanup.
    """
    mods = {
        "RealtimeSTT": stt if stt is not None else _make_fake_realtimestt(),
        "RealtimeTTS": tts if tts is not None else _make_fake_realtimetts(
            engine_classes={"SystemEngine": FakeEngine}),
    }
    p = mock.patch.dict(sys.modules, mods)
    p.start()
    testcase.addCleanup(p.stop)
    return mods


class _ResetSingletonMixin:
    """Reset the module-level singleton so tests don't bleed into each other."""

    def setUp(self):
        super().setUp()
        rtv._singleton = None

    def tearDown(self):
        rtv._singleton = None
        super().tearDown()


# ──────────────────────────────────────────────────────────────────────
# is_available()
# ──────────────────────────────────────────────────────────────────────

class IsAvailableTests(unittest.TestCase):
    def test_both_present(self):
        _patch_backends(self)
        ok, why = rtv.is_available()
        self.assertTrue(ok)
        self.assertEqual(why, "")

    def test_stt_missing(self):
        # Force `import RealtimeSTT` to raise.
        def fake_import(name, *a, **k):
            if name == "RealtimeSTT":
                raise ImportError("no RealtimeSTT")
            return _orig_import(name, *a, **k)

        _orig_import = __import__
        with mock.patch("builtins.__import__", side_effect=fake_import):
            ok, why = rtv.is_available()
        self.assertFalse(ok)
        self.assertIn("RealtimeSTT missing", why)

    def test_tts_missing(self):
        # RealtimeSTT imports, RealtimeTTS raises.
        _patch_backends(self, stt=_make_fake_realtimestt())

        def fake_import(name, *a, **k):
            if name == "RealtimeTTS":
                raise ImportError("no RealtimeTTS")
            return _orig_import(name, *a, **k)

        _orig_import = __import__
        with mock.patch("builtins.__import__", side_effect=fake_import):
            ok, why = rtv.is_available()
        self.assertFalse(ok)
        self.assertIn("RealtimeTTS missing", why)


# ──────────────────────────────────────────────────────────────────────
# __init__ / status / simple state
# ──────────────────────────────────────────────────────────────────────

class InitTests(unittest.TestCase):
    def test_defaults(self):
        p = rtv.RealtimeVoicePipeline()
        self.assertEqual(p.stt_model, rtv.DEFAULT_STT_MODEL)
        self.assertEqual(p.stt_language, rtv.DEFAULT_STT_LANGUAGE)
        self.assertEqual(p.tts_engine_name, rtv.DEFAULT_TTS_ENGINE)
        self.assertEqual(p.tts_voice, rtv.DEFAULT_TTS_VOICE)
        self.assertEqual(p.sample_rate, rtv.DEFAULT_SAMPLE_RATE)
        self.assertIsNone(p.input_device)
        self.assertFalse(p.is_running())
        self.assertFalse(p.is_playing())

    def test_engine_name_normalised(self):
        p = rtv.RealtimeVoicePipeline(tts_engine="  AZURE ")
        self.assertEqual(p.tts_engine_name, "azure")

    def test_engine_name_empty_falls_back_to_default(self):
        p = rtv.RealtimeVoicePipeline(tts_engine="")
        self.assertEqual(p.tts_engine_name, rtv.DEFAULT_TTS_ENGINE)

    def test_numeric_casts(self):
        p = rtv.RealtimeVoicePipeline(
            sample_rate="22050", silero_sensitivity="0.7", webrtc_sensitivity="3",
            input_device="2",
        )
        self.assertEqual(p.sample_rate, 22050)
        self.assertIsInstance(p.sample_rate, int)
        self.assertAlmostEqual(p.silero_sensitivity, 0.7)
        self.assertIsInstance(p.silero_sensitivity, float)
        self.assertEqual(p.webrtc_sensitivity, 3)
        # input_device is stored raw (cast happens at _init_stt time)
        self.assertEqual(p.input_device, "2")

    def test_status_shape(self):
        p = rtv.RealtimeVoicePipeline(stt_model="small", stt_language="fr",
                                      tts_engine="azure", tts_voice="V")
        st = p.status()
        self.assertEqual(st["running"], False)
        self.assertEqual(st["playing"], False)
        self.assertEqual(st["stt_model"], "small")
        self.assertEqual(st["stt_language"], "fr")
        self.assertEqual(st["tts_engine"], "azure")
        self.assertEqual(st["tts_voice"], "V")
        self.assertEqual(st["last_partial"], "")
        self.assertEqual(st["last_utterance"], "")
        self.assertEqual(st["last_utterance_ts"], 0.0)
        self.assertEqual(st["last_barge_in_ts"], 0.0)


# ──────────────────────────────────────────────────────────────────────
# start() / stop()  (lifecycle, mocked backends, no real loop)
# ──────────────────────────────────────────────────────────────────────

class StartStopTests(unittest.TestCase):
    def test_start_unavailable_returns_false(self):
        p = rtv.RealtimeVoicePipeline()
        with mock.patch.object(rtv, "is_available", return_value=(False, "nope")), \
                mock.patch("builtins.print"):
            self.assertFalse(p.start())
        self.assertFalse(p.is_running())

    def test_start_already_running_returns_true(self):
        p = rtv.RealtimeVoicePipeline()
        p._running = True
        with mock.patch.object(rtv, "is_available", return_value=(True, "")):
            self.assertTrue(p.start())

    def test_start_tts_init_fail_returns_false(self):
        p = rtv.RealtimeVoicePipeline()
        with mock.patch.object(rtv, "is_available", return_value=(True, "")), \
                mock.patch.object(p, "_init_tts", return_value=False):
            self.assertFalse(p.start())
        self.assertFalse(p.is_running())

    def test_start_stt_init_fail_tears_down_tts(self):
        p = rtv.RealtimeVoicePipeline()
        with mock.patch.object(rtv, "is_available", return_value=(True, "")), \
                mock.patch.object(p, "_init_tts", return_value=True), \
                mock.patch.object(p, "_init_stt", return_value=False), \
                mock.patch.object(p, "_teardown_tts") as td:
            self.assertFalse(p.start())
        td.assert_called_once()
        self.assertFalse(p.is_running())

    def test_start_success_spawns_thread_and_runs(self):
        """Full happy path. The STT thread really starts; we pre-set the stop
        flag so the loop exits on its first check, then join it — no real
        audio, no busy spin."""
        _patch_backends(self)
        p = rtv.RealtimeVoicePipeline()
        with mock.patch("builtins.print"):
            # Pre-set the stop flag so _stt_loop exits immediately. start()
            # calls _stop_flag.clear() first, so instead force the recorder's
            # text() to raise-then-stop is overkill: simplest is to let the
            # loop run one iteration against an empty recorder then stop.
            self.assertTrue(p.start())
            self.assertTrue(p.is_running())
            self.assertIsNotNone(p._stt_thread)
            # Now stop cleanly and join.
            p.stop()
        self.assertFalse(p.is_running())
        if p._stt_thread is not None:
            p._stt_thread.join(timeout=2.0)
            self.assertFalse(p._stt_thread.is_alive())

    def test_stop_calls_recorder_stop_and_teardown(self):
        p = rtv.RealtimeVoicePipeline()
        rec = FakeRecorder()
        p._recorder = rec
        with mock.patch.object(p, "_teardown_tts") as td:
            p.stop()
        self.assertEqual(rec.stop_calls, 1)
        self.assertEqual(rec.shutdown_calls, 1)
        td.assert_called_once()
        self.assertIsNone(p._recorder)
        self.assertFalse(p.is_running())
        self.assertTrue(p._stop_flag.is_set())

    def test_stop_swallows_recorder_exception(self):
        p = rtv.RealtimeVoicePipeline()
        rec = FakeRecorder()
        rec.stop_raises = True
        p._recorder = rec
        with mock.patch.object(p, "_teardown_tts"):
            p.stop()  # must not raise
        # shutdown still attempted after stop() raised
        self.assertEqual(rec.shutdown_calls, 1)

    def test_stop_idempotent_no_recorder(self):
        p = rtv.RealtimeVoicePipeline()
        with mock.patch.object(p, "_teardown_tts") as td:
            p.stop()
            p.stop()
        self.assertEqual(td.call_count, 2)
        self.assertFalse(p.is_running())


# ──────────────────────────────────────────────────────────────────────
# feed_response_chunk()
# ──────────────────────────────────────────────────────────────────────

class FeedResponseTests(unittest.TestCase):
    def _pipe_with_stream(self, **stream_kw):
        p = rtv.RealtimeVoicePipeline()
        s = FakeTTSStream(**stream_kw)
        p._tts_stream = s
        return p, s

    def test_empty_text_noop(self):
        p, s = self._pipe_with_stream()
        p.feed_response_chunk("")
        self.assertEqual(s.fed, [])
        self.assertEqual(s.play_async_calls, 0)

    def test_no_stream_noop(self):
        p = rtv.RealtimeVoicePipeline()
        p._tts_stream = None
        p.feed_response_chunk("hello")  # must not raise

    def test_feed_and_play_with_hook(self):
        p, s = self._pipe_with_stream(accept_hook=True)
        p.feed_response_chunk("Hello, sir. ")
        self.assertEqual(s.fed, ["Hello, sir. "])
        self.assertEqual(s.play_async_calls, 1)
        self.assertEqual(s.play_async_hooked, 1)
        self.assertTrue(p.is_playing())

    def test_play_async_typeerror_falls_back_to_no_kwargs(self):
        p, s = self._pipe_with_stream(accept_hook=False)
        p.feed_response_chunk("Hi.")
        self.assertEqual(s.play_async_calls, 1)   # the fallback call counts
        self.assertEqual(s.play_async_hooked, 0)  # hook attempt rejected
        self.assertTrue(p.is_playing())

    def test_already_playing_does_not_replay(self):
        p, s = self._pipe_with_stream()
        s._playing = True
        p.feed_response_chunk("more text")
        self.assertEqual(s.fed, ["more text"])     # still fed
        self.assertEqual(s.play_async_calls, 0)    # but no new play_async

    def test_feed_exception_clears_playing(self):
        p, s = self._pipe_with_stream()
        s.feed_raises = True
        p._playing.set()
        with mock.patch("builtins.print"):
            p.feed_response_chunk("boom")
        self.assertFalse(p.is_playing())


# ──────────────────────────────────────────────────────────────────────
# flush_response()
# ──────────────────────────────────────────────────────────────────────

class FlushResponseTests(unittest.TestCase):
    def test_no_stream_noop(self):
        p = rtv.RealtimeVoicePipeline()
        p._tts_stream = None
        p.flush_response()  # must not raise

    def test_calls_first_available_hook(self):
        p = rtv.RealtimeVoicePipeline()
        s = FakeTTSStream()
        p._tts_stream = s
        p.flush_response()
        self.assertEqual(s.finalize_calls, 1)

    def test_hook_exception_swallowed(self):
        p = rtv.RealtimeVoicePipeline()
        s = FakeTTSStream()
        s.finalize_raises = True
        p._tts_stream = s
        p.flush_response()  # must not raise
        self.assertEqual(s.finalize_calls, 1)

    def test_no_hook_present_is_noop(self):
        # A stream object exposing none of finalize/finalise/end_of_stream/flush.
        class Bare:
            pass
        p = rtv.RealtimeVoicePipeline()
        p._tts_stream = Bare()
        p.flush_response()  # must not raise

    def test_prefers_finalize_over_flush(self):
        # Object with both finalize and flush — finalize wins (first in list),
        # flush is never called.
        calls = []

        class Both:
            def finalize(self):
                calls.append("finalize")

            def flush(self):
                calls.append("flush")

        p = rtv.RealtimeVoicePipeline()
        p._tts_stream = Both()
        p.flush_response()
        self.assertEqual(calls, ["finalize"])


# ──────────────────────────────────────────────────────────────────────
# wait_for_playback()
# ──────────────────────────────────────────────────────────────────────

class WaitForPlaybackTests(unittest.TestCase):
    def test_no_stream_returns_true(self):
        p = rtv.RealtimeVoicePipeline()
        p._tts_stream = None
        self.assertTrue(p.wait_for_playback())

    def test_not_playing_returns_true_and_clears(self):
        p = rtv.RealtimeVoicePipeline()
        s = FakeTTSStream()
        s._playing = False
        p._tts_stream = s
        p._playing.set()
        self.assertTrue(p.wait_for_playback())
        self.assertFalse(p.is_playing())   # _on_playback_end cleared it

    def test_is_playing_exception_treated_as_done(self):
        p = rtv.RealtimeVoicePipeline()
        s = FakeTTSStream()
        s.is_playing_raises = True
        p._tts_stream = s
        self.assertTrue(p.wait_for_playback())

    def test_timeout_returns_false(self):
        p = rtv.RealtimeVoicePipeline()
        s = FakeTTSStream()
        s._playing = True   # stays playing forever
        p._tts_stream = s
        # timeout=0 → deadline is now; the loop checks playing then the
        # deadline and returns False without a real sleep stalling the suite.
        with mock.patch.object(rtv.time, "sleep"):
            self.assertFalse(p.wait_for_playback(timeout=0.0))

    def test_drains_after_a_couple_polls(self):
        """Stays playing for two polls, then drains → returns True."""
        p = rtv.RealtimeVoicePipeline()
        s = FakeTTSStream()
        seq = [True, True, False]

        def fake_is_playing():
            return seq.pop(0)

        s.is_playing = fake_is_playing
        p._tts_stream = s
        with mock.patch.object(rtv.time, "sleep"):
            self.assertTrue(p.wait_for_playback(timeout=10.0))
        self.assertFalse(p.is_playing())


# ──────────────────────────────────────────────────────────────────────
# barge_in()
# ──────────────────────────────────────────────────────────────────────

class BargeInTests(unittest.TestCase):
    def test_stops_stream_clears_playing_fires_callback(self):
        fired = []
        p = rtv.RealtimeVoicePipeline(on_barge_in=lambda: fired.append(1))
        s = FakeTTSStream()
        p._tts_stream = s
        p._playing.set()
        p.barge_in()
        self.assertEqual(s.stop_calls, 1)
        self.assertFalse(p.is_playing())
        self.assertEqual(fired, [1])
        self.assertGreater(p._last_barge_in_ts, 0.0)

    def test_no_stream_still_fires_callback(self):
        fired = []
        p = rtv.RealtimeVoicePipeline(on_barge_in=lambda: fired.append(1))
        p._tts_stream = None
        p.barge_in()
        self.assertEqual(fired, [1])

    def test_no_callback_is_fine(self):
        p = rtv.RealtimeVoicePipeline(on_barge_in=None)
        s = FakeTTSStream()
        p._tts_stream = s
        p.barge_in()  # must not raise
        self.assertEqual(s.stop_calls, 1)

    def test_stream_stop_exception_swallowed_callback_still_fires(self):
        fired = []
        p = rtv.RealtimeVoicePipeline(on_barge_in=lambda: fired.append(1))
        s = FakeTTSStream()
        s.stop_raises = True
        p._tts_stream = s
        p.barge_in()
        self.assertEqual(fired, [1])
        self.assertFalse(p.is_playing())

    def test_callback_exception_swallowed(self):
        def boom():
            raise RuntimeError("cb boom")

        p = rtv.RealtimeVoicePipeline(on_barge_in=boom)
        p._tts_stream = FakeTTSStream()
        with mock.patch("builtins.print"):
            p.barge_in()  # must not raise


# ──────────────────────────────────────────────────────────────────────
# _on_audio_chunk()  (uses REAL numpy; audio_processor is mocked)
# ──────────────────────────────────────────────────────────────────────

class OnAudioChunkTests(unittest.TestCase):
    def _install_fake_ap(self, mod):
        """Bind a fake ``core.audio_processor`` so ``from core import
        audio_processor`` inside _on_audio_chunk resolves to it.

        ``from core import audio_processor`` binds the *attribute* on the
        already-imported ``core`` package. If some earlier test imported the
        real submodule, that attribute is set and patching only sys.modules
        wouldn't shadow it (this is exactly what bit us under the CI-sim, where
        thousands of prior tests import the real module first). So we patch both
        the package attribute (create=True covers the not-yet-imported case) and
        the sys.modules entry, auto-restored via addCleanup.
        """
        import core as core_pkg
        p1 = mock.patch.object(core_pkg, "audio_processor", mod, create=True)
        p2 = mock.patch.dict(sys.modules, {"core.audio_processor": mod})
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def _fake_ap(self):
        """A fake ``core.audio_processor`` module recording feed_playback args."""
        mod = type(sys)("core.audio_processor")
        captured = {}

        def feed_playback(arr, sample_rate=None):
            captured["arr"] = arr
            captured["sr"] = sample_rate

        mod.feed_playback = feed_playback
        return mod, captured

    def test_bytes_chunk_converted_to_float32(self):
        mod, captured = self._fake_ap()
        p = rtv.RealtimeVoicePipeline(sample_rate=16000)
        pcm = np.array([0, 16384, -16384, 32767], dtype=np.int16).tobytes()
        self._install_fake_ap(mod)
        p._on_audio_chunk(pcm)
        arr = captured["arr"]
        self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(captured["sr"], 16000)
        # int16 32767 → ~1.0 after /32767
        self.assertAlmostEqual(float(arr[-1]), 1.0, places=4)
        self.assertAlmostEqual(float(arr[0]), 0.0, places=6)

    def test_ndarray_chunk_passed_through(self):
        mod, captured = self._fake_ap()
        p = rtv.RealtimeVoicePipeline(sample_rate=22050)
        chunk = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        self._install_fake_ap(mod)
        p._on_audio_chunk(chunk)
        arr = captured["arr"]
        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_allclose(arr, chunk, rtol=1e-6)
        self.assertEqual(captured["sr"], 22050)

    def test_audio_processor_import_failure_is_silent(self):
        p = rtv.RealtimeVoicePipeline()

        # Make `from core import audio_processor` fail.
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "core" and fromlist and "audio_processor" in fromlist:
                raise ImportError("no audio_processor")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            p._on_audio_chunk(b"\x00\x00")  # must not raise

    def test_inner_exception_is_silent(self):
        """feed_playback raising must be swallowed (HUD reference is best-effort)."""
        mod = type(sys)("core.audio_processor")

        def feed_playback(arr, sample_rate=None):
            raise RuntimeError("feed_playback boom")

        mod.feed_playback = feed_playback
        p = rtv.RealtimeVoicePipeline()
        self._install_fake_ap(mod)
        p._on_audio_chunk(np.zeros(4, dtype=np.float32))  # must not raise


# ──────────────────────────────────────────────────────────────────────
# _init_tts() / _build_tts_engine() / _teardown_tts()
# ──────────────────────────────────────────────────────────────────────

class InitTtsTests(unittest.TestCase):
    def test_init_tts_success(self):
        _patch_backends(self)
        p = rtv.RealtimeVoicePipeline(tts_engine="system")
        self.assertTrue(p._init_tts())
        self.assertIsNotNone(p._tts_stream)
        self.assertIsInstance(p._tts_engine, FakeEngine)

    def test_init_tts_unknown_engine_returns_false(self):
        _patch_backends(self)
        p = rtv.RealtimeVoicePipeline(tts_engine="bogus")
        with mock.patch("builtins.print"):
            self.assertFalse(p._init_tts())
        self.assertIsNone(p._tts_stream)

    def test_init_tts_stream_ctor_raises_returns_false(self):
        tts = _make_fake_realtimetts(
            engine_classes={"SystemEngine": FakeEngine}, stream_raises=True)
        _patch_backends(self, tts=tts)
        p = rtv.RealtimeVoicePipeline(tts_engine="system")
        with mock.patch("builtins.print"):
            self.assertFalse(p._init_tts())

    def test_build_system_engine_voice_kwarg(self):
        _patch_backends(self)
        p = rtv.RealtimeVoicePipeline(tts_engine="system", tts_voice="Ryan")
        eng = p._build_tts_engine()
        self.assertIsInstance(eng, FakeEngine)
        self.assertEqual(eng.kwargs.get("voice"), "Ryan")

    def test_build_system_engine_typeerror_fallback(self):
        # SystemEngine that rejects the voice kwarg → no-arg fallback.
        class PickyEngine(FakeEngine):
            def __init__(self, *a, **kw):
                if "voice" in kw:
                    raise TypeError("no voice kwarg")
                super().__init__(*a, **kw)

        tts = _make_fake_realtimetts(engine_classes={"SystemEngine": PickyEngine})
        _patch_backends(self, tts=tts)
        p = rtv.RealtimeVoicePipeline(tts_engine="system", tts_voice="Ryan")
        eng = p._build_tts_engine()
        self.assertIsInstance(eng, PickyEngine)
        self.assertNotIn("voice", eng.kwargs)

    def test_build_azure_missing_key_returns_none(self):
        tts = _make_fake_realtimetts(engine_classes={"AzureEngine": FakeEngine})
        _patch_backends(self, tts=tts)
        p = rtv.RealtimeVoicePipeline(tts_engine="azure")
        with mock.patch.dict(os.environ, {"AZURE_TTS_KEY": ""}, clear=False), \
                mock.patch("builtins.print"):
            self.assertIsNone(p._build_tts_engine())

    def test_build_azure_with_key(self):
        tts = _make_fake_realtimetts(engine_classes={"AzureEngine": FakeEngine})
        _patch_backends(self, tts=tts)
        p = rtv.RealtimeVoicePipeline(tts_engine="azure", tts_voice="en-US-Jenny")
        with mock.patch.dict(os.environ,
                             {"AZURE_TTS_KEY": "k", "AZURE_TTS_REGION": "westus"},
                             clear=False):
            eng = p._build_tts_engine()
        self.assertIsInstance(eng, FakeEngine)
        self.assertEqual(eng.kwargs.get("speech_key"), "k")
        self.assertEqual(eng.kwargs.get("speech_region"), "westus")
        self.assertEqual(eng.kwargs.get("voice"), "en-US-Jenny")

    def test_build_azure_default_region(self):
        tts = _make_fake_realtimetts(engine_classes={"AzureEngine": FakeEngine})
        _patch_backends(self, tts=tts)
        p = rtv.RealtimeVoicePipeline(tts_engine="azure")
        with mock.patch.dict(os.environ,
                             {"AZURE_TTS_KEY": "k", "AZURE_TTS_REGION": ""},
                             clear=False):
            eng = p._build_tts_engine()
        self.assertEqual(eng.kwargs.get("speech_region"), "eastus")

    def test_build_elevenlabs_missing_key_returns_none(self):
        tts = _make_fake_realtimetts(engine_classes={"ElevenlabsEngine": FakeEngine})
        _patch_backends(self, tts=tts)
        p = rtv.RealtimeVoicePipeline(tts_engine="elevenlabs")
        with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}, clear=False), \
                mock.patch("builtins.print"):
            self.assertIsNone(p._build_tts_engine())

    def test_build_elevenlabs_with_key_kwarg(self):
        tts = _make_fake_realtimetts(engine_classes={"ElevenlabsEngine": FakeEngine})
        _patch_backends(self, tts=tts)
        p = rtv.RealtimeVoicePipeline(tts_engine="elevenlabs", tts_voice="Bella")
        with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}, clear=False):
            eng = p._build_tts_engine()
        self.assertEqual(eng.kwargs.get("api_key"), "k")
        self.assertEqual(eng.kwargs.get("voice"), "Bella")

    def test_build_elevenlabs_positional_fallback(self):
        # Engine that rejects api_key= kwarg → positional fallback path.
        class PosEngine(FakeEngine):
            def __init__(self, *a, **kw):
                if "api_key" in kw:
                    raise TypeError("api_key not accepted")
                super().__init__(*a, **kw)

        tts = _make_fake_realtimetts(engine_classes={"ElevenlabsEngine": PosEngine})
        _patch_backends(self, tts=tts)
        p = rtv.RealtimeVoicePipeline(tts_engine="elevenlabs", tts_voice="Bella")
        with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": "k"}, clear=False):
            eng = p._build_tts_engine()
        self.assertEqual(eng.args, ("k",))           # key passed positionally
        self.assertEqual(eng.kwargs.get("voice"), "Bella")

    def test_build_coqui_voice_kwarg(self):
        tts = _make_fake_realtimetts(engine_classes={"CoquiEngine": FakeEngine})
        _patch_backends(self, tts=tts)
        p = rtv.RealtimeVoicePipeline(tts_engine="coqui", tts_voice="speaker.wav")
        eng = p._build_tts_engine()
        self.assertEqual(eng.kwargs.get("voice"), "speaker.wav")

    def test_build_coqui_typeerror_fallback(self):
        class PickyCoqui(FakeEngine):
            def __init__(self, *a, **kw):
                if "voice" in kw:
                    raise TypeError("no voice")
                super().__init__(*a, **kw)

        tts = _make_fake_realtimetts(engine_classes={"CoquiEngine": PickyCoqui})
        _patch_backends(self, tts=tts)
        p = rtv.RealtimeVoicePipeline(tts_engine="coqui", tts_voice="x")
        eng = p._build_tts_engine()
        self.assertIsInstance(eng, PickyCoqui)

    def test_build_engine_import_error_returns_none(self):
        # RealtimeTTS present but SystemEngine attr absent → ImportError on
        # `from RealtimeTTS import SystemEngine`.
        tts = _make_fake_realtimetts(engine_classes={})   # no engines
        _patch_backends(self, tts=tts)
        p = rtv.RealtimeVoicePipeline(tts_engine="system")
        with mock.patch("builtins.print"):
            self.assertIsNone(p._build_tts_engine())

    def test_build_engine_generic_exception_returns_none(self):
        class Exploding(FakeEngine):
            def __init__(self, *a, **kw):
                raise RuntimeError("engine load boom")

        tts = _make_fake_realtimetts(engine_classes={"SystemEngine": Exploding})
        _patch_backends(self, tts=tts)
        p = rtv.RealtimeVoicePipeline(tts_engine="system")
        with mock.patch("builtins.print"):
            self.assertIsNone(p._build_tts_engine())

    def test_init_tts_engine_none_returns_false(self):
        # _build_tts_engine returns None → _init_tts returns False without a stream.
        p = rtv.RealtimeVoicePipeline(tts_engine="system")
        _patch_backends(self)
        with mock.patch.object(p, "_build_tts_engine", return_value=None):
            self.assertFalse(p._init_tts())
        self.assertIsNone(p._tts_stream)


class TeardownTtsTests(unittest.TestCase):
    def test_teardown_stops_stream_and_shuts_engine(self):
        p = rtv.RealtimeVoicePipeline()
        s = FakeTTSStream()
        eng = FakeEngine()
        p._tts_stream = s
        p._tts_engine = eng
        p._teardown_tts()
        self.assertEqual(s.stop_calls, 1)
        self.assertEqual(eng.shutdown_calls, 1)
        self.assertIsNone(p._tts_stream)
        self.assertIsNone(p._tts_engine)

    def test_teardown_swallows_stream_and_engine_exceptions(self):
        p = rtv.RealtimeVoicePipeline()
        s = FakeTTSStream()
        s.stop_raises = True
        eng = FakeEngine()
        eng.shutdown_raises = True
        p._tts_stream = s
        p._tts_engine = eng
        p._teardown_tts()  # must not raise
        self.assertIsNone(p._tts_stream)
        self.assertIsNone(p._tts_engine)

    def test_teardown_no_stream_no_engine(self):
        p = rtv.RealtimeVoicePipeline()
        p._tts_stream = None
        p._tts_engine = None
        p._teardown_tts()  # must not raise

    def test_teardown_engine_without_shutdown_attr(self):
        class NoShutdown:
            pass
        p = rtv.RealtimeVoicePipeline()
        p._tts_stream = None
        p._tts_engine = NoShutdown()
        p._teardown_tts()  # must not raise
        self.assertIsNone(p._tts_engine)


# ──────────────────────────────────────────────────────────────────────
# _init_stt()
# ──────────────────────────────────────────────────────────────────────

class InitSttTests(unittest.TestCase):
    def test_import_failure_returns_false(self):
        stt = _make_fake_realtimestt(import_error=True)
        _patch_backends(self, stt=stt)
        p = rtv.RealtimeVoicePipeline()
        with mock.patch("builtins.print"):
            self.assertFalse(p._init_stt())
        self.assertIsNone(p._recorder)

    def test_success_full_kwargs(self):
        captured = {}

        def factory(**kwargs):
            captured.update(kwargs)
            return FakeRecorder(**kwargs)

        stt = _make_fake_realtimestt(recorder_factory=factory)
        _patch_backends(self, stt=stt)
        p = rtv.RealtimeVoicePipeline(stt_model="small", stt_language="es")
        self.assertTrue(p._init_stt())
        self.assertIsInstance(p._recorder, FakeRecorder)
        self.assertEqual(captured["model"], "small")
        self.assertEqual(captured["language"], "es")
        self.assertTrue(captured["enable_realtime_transcription"])
        # wired callbacks point at the pipeline's bound methods
        self.assertEqual(captured["on_recording_start"], p._on_recording_start)
        self.assertEqual(captured["on_recording_stop"], p._on_recording_stop)
        self.assertNotIn("input_device_index", captured)

    def test_input_device_index_passed(self):
        captured = {}

        def factory(**kwargs):
            captured.update(kwargs)
            return FakeRecorder(**kwargs)

        stt = _make_fake_realtimestt(recorder_factory=factory)
        _patch_backends(self, stt=stt)
        p = rtv.RealtimeVoicePipeline(input_device=3)
        self.assertTrue(p._init_stt())
        self.assertEqual(captured["input_device_index"], 3)

    def test_typeerror_falls_back_to_minimal_kwargs(self):
        """First ctor call (full kwargs) raises TypeError → minimal retry."""
        calls = []

        def factory(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                # full kwargs set includes silero_sensitivity → reject it once
                raise TypeError("unexpected kwarg silero_sensitivity")
            return FakeRecorder(**kwargs)

        stt = _make_fake_realtimestt(recorder_factory=factory)
        _patch_backends(self, stt=stt)
        p = rtv.RealtimeVoicePipeline()
        self.assertTrue(p._init_stt())
        self.assertEqual(len(calls), 2)
        # the minimal retry must NOT carry the sensitivity kwargs
        self.assertNotIn("silero_sensitivity", calls[1])
        self.assertNotIn("webrtc_sensitivity", calls[1])
        self.assertIn("on_recording_start", calls[1])

    def test_generic_exception_returns_false(self):
        def factory(**kwargs):
            raise RuntimeError("recorder boom")

        stt = _make_fake_realtimestt(recorder_factory=factory)
        _patch_backends(self, stt=stt)
        p = rtv.RealtimeVoicePipeline()
        with mock.patch("builtins.print"):
            self.assertFalse(p._init_stt())
        self.assertIsNone(p._recorder)


# ──────────────────────────────────────────────────────────────────────
# _stt_loop()  (driven synchronously — no real thread)
# ──────────────────────────────────────────────────────────────────────

class SttLoopTests(unittest.TestCase):
    def test_no_recorder_returns_immediately(self):
        p = rtv.RealtimeVoicePipeline()
        p._recorder = None
        p._stt_loop()  # returns without touching anything

    def test_utterance_dispatched_to_callback(self):
        seen = []
        p = rtv.RealtimeVoicePipeline(on_user_utterance=seen.append)
        rec = FakeRecorder()
        # one good utterance, then the stop flag trips on the next loop top.
        rec.script = ["Turn on the lights."]
        p._recorder = rec

        # Drive exactly one iteration: after the first text() returns, set the
        # stop flag from inside the callback so the while-loop exits.
        def cb(text):
            seen.append(text)
            p._stop_flag.set()

        p.on_user_utterance = cb
        p._stt_loop()
        self.assertEqual(seen, ["Turn on the lights."])
        self.assertEqual(p._last_utterance, "Turn on the lights.")
        self.assertGreater(p._last_utterance_ts, 0.0)

    def test_empty_text_skipped_then_stop(self):
        p = rtv.RealtimeVoicePipeline(on_user_utterance=lambda t: None)
        rec = FakeRecorder()
        rec.script = ["", "   "]    # both blank → skipped
        p._recorder = rec
        # Stop after the loop has consumed both blanks: patch sleep to set the
        # flag once the script is exhausted.
        def fake_sleep(_):
            if not rec.script:
                p._stop_flag.set()

        with mock.patch.object(rtv.time, "sleep", fake_sleep):
            p._stt_loop()
        self.assertEqual(p._last_utterance, "")   # nothing dispatched

    def test_callback_none_still_records_last_utterance(self):
        p = rtv.RealtimeVoicePipeline(on_user_utterance=None)
        rec = FakeRecorder()
        rec.script = ["hello"]
        p._recorder = rec

        # No callback → the `continue` path runs; set stop via sleep patch so
        # the loop ends after one real iteration.
        def fake_sleep(_):
            p._stop_flag.set()

        # The success branch with cb=None doesn't sleep, so trip the flag right
        # after the first text() by wrapping it.
        orig_text = rec.text

        def wrapped_text():
            out = orig_text()
            p._stop_flag.set()
            return out

        rec.text = wrapped_text
        p._stt_loop()
        self.assertEqual(p._last_utterance, "hello")

    def test_callback_exception_swallowed(self):
        def boom(text):
            p._stop_flag.set()
            raise RuntimeError("utterance cb boom")

        p = rtv.RealtimeVoicePipeline(on_user_utterance=boom)
        rec = FakeRecorder()
        rec.script = ["x"]
        p._recorder = rec
        with mock.patch("builtins.print"):
            p._stt_loop()  # must not raise
        self.assertEqual(p._last_utterance, "x")

    def test_text_exception_logs_and_continues(self):
        """rec.text() raising (stop flag NOT set) → logs, sleeps, continues;
        next iteration the flag is set so it exits."""
        p = rtv.RealtimeVoicePipeline()
        rec = FakeRecorder()
        rec.script = [RuntimeError("text boom")]
        p._recorder = rec

        def fake_sleep(_):
            p._stop_flag.set()   # exit on the post-error sleep

        with mock.patch.object(rtv.time, "sleep", fake_sleep), \
                mock.patch("builtins.print") as pr:
            p._stt_loop()
        self.assertTrue(any("STT loop error" in str(c.args[0])
                            for c in pr.call_args_list))

    def test_text_exception_during_stop_returns_quietly(self):
        """If text() raises *because* we're shutting down (stop flag set), the
        loop returns without logging."""
        p = rtv.RealtimeVoicePipeline()
        rec = FakeRecorder()

        def raising_text():
            p._stop_flag.set()     # simulate stop() racing the blocking text()
            raise RuntimeError("interrupted")

        rec.text = raising_text
        p._recorder = rec
        with mock.patch("builtins.print") as pr:
            p._stt_loop()
        # no "STT loop error" line — it returned on the stop-flag check
        self.assertFalse(any("STT loop error" in str(c.args[0])
                             for c in pr.call_args_list))


# ──────────────────────────────────────────────────────────────────────
# _on_partial() / _on_recording_start() / _on_recording_stop() / _on_playback_end()
# ──────────────────────────────────────────────────────────────────────

class CallbackTests(unittest.TestCase):
    def test_on_partial_records_and_forwards(self):
        seen = []
        p = rtv.RealtimeVoicePipeline(on_partial_transcript=seen.append)
        p._on_partial("hel")
        self.assertEqual(p._last_partial, "hel")
        self.assertEqual(seen, ["hel"])

    def test_on_partial_none_text_becomes_empty(self):
        p = rtv.RealtimeVoicePipeline()
        p._on_partial(None)
        self.assertEqual(p._last_partial, "")

    def test_on_partial_callback_exception_swallowed(self):
        def boom(t):
            raise RuntimeError("partial cb boom")

        p = rtv.RealtimeVoicePipeline(on_partial_transcript=boom)
        p._on_partial("text here")   # must not raise
        self.assertEqual(p._last_partial, "text here")

    def test_on_partial_triggers_barge_in_when_playing(self):
        p = rtv.RealtimeVoicePipeline()
        p._playing.set()
        with mock.patch.object(p, "barge_in") as bi:
            p._on_partial("abc")    # len 3 >= PARTIAL_BARGE_IN_MIN_CHARS
        bi.assert_called_once()

    def test_on_partial_no_barge_when_short(self):
        p = rtv.RealtimeVoicePipeline()
        p._playing.set()
        with mock.patch.object(p, "barge_in") as bi:
            p._on_partial("ab")     # below threshold
        bi.assert_not_called()

    def test_on_partial_no_barge_when_not_playing(self):
        p = rtv.RealtimeVoicePipeline()
        # _playing not set
        with mock.patch.object(p, "barge_in") as bi:
            p._on_partial("abcdef")
        bi.assert_not_called()

    def test_on_partial_whitespace_padding_counts_stripped_chars(self):
        # "  ab  " strips to 2 chars → below threshold, no barge.
        p = rtv.RealtimeVoicePipeline()
        p._playing.set()
        with mock.patch.object(p, "barge_in") as bi:
            p._on_partial("  ab  ")
        bi.assert_not_called()

    def test_on_recording_start_fires_vad_and_barge(self):
        vad = []
        p = rtv.RealtimeVoicePipeline(on_vad_start=lambda: vad.append(1))
        p._playing.set()
        with mock.patch.object(p, "barge_in") as bi:
            p._on_recording_start()
        self.assertEqual(vad, [1])
        bi.assert_called_once()

    def test_on_recording_start_no_barge_when_idle(self):
        p = rtv.RealtimeVoicePipeline()
        with mock.patch.object(p, "barge_in") as bi:
            p._on_recording_start()
        bi.assert_not_called()

    def test_on_recording_start_vad_callback_exception_swallowed(self):
        def boom():
            raise RuntimeError("vad start boom")

        p = rtv.RealtimeVoicePipeline(on_vad_start=boom)
        p._on_recording_start()  # must not raise

    def test_on_recording_stop_fires_vad(self):
        vad = []
        p = rtv.RealtimeVoicePipeline(on_vad_stop=lambda: vad.append(1))
        p._on_recording_stop()
        self.assertEqual(vad, [1])

    def test_on_recording_stop_no_callback(self):
        p = rtv.RealtimeVoicePipeline(on_vad_stop=None)
        p._on_recording_stop()  # must not raise

    def test_on_recording_stop_callback_exception_swallowed(self):
        def boom():
            raise RuntimeError("vad stop boom")

        p = rtv.RealtimeVoicePipeline(on_vad_stop=boom)
        p._on_recording_stop()  # must not raise

    def test_on_playback_end_clears_playing(self):
        p = rtv.RealtimeVoicePipeline()
        p._playing.set()
        p._on_playback_end()
        self.assertFalse(p.is_playing())


# ──────────────────────────────────────────────────────────────────────
# Module-level: get_pipeline / start_pipeline / stop_pipeline
# ──────────────────────────────────────────────────────────────────────

class ModuleLevelTests(_ResetSingletonMixin, unittest.TestCase):
    def test_get_pipeline_none_initially(self):
        self.assertIsNone(rtv.get_pipeline())

    def test_start_pipeline_turn_based_returns_none(self):
        self.assertIsNone(rtv.start_pipeline(voice_mode="turn_based"))
        self.assertIsNone(rtv.get_pipeline())

    def test_start_pipeline_default_mode_is_turn_based(self):
        # No voice_mode arg → defaults to turn_based → None.
        self.assertIsNone(rtv.start_pipeline())

    def test_start_pipeline_realtime_unavailable_returns_none(self):
        # start() returns False (unavailable) → singleton stays None.
        with mock.patch.object(rtv.RealtimeVoicePipeline, "start",
                               return_value=False):
            self.assertIsNone(rtv.start_pipeline(voice_mode="realtime"))
        self.assertIsNone(rtv.get_pipeline())

    def test_start_pipeline_realtime_success_sets_singleton(self):
        with mock.patch.object(rtv.RealtimeVoicePipeline, "start",
                               return_value=True):
            pipe = rtv.start_pipeline(
                voice_mode="realtime",
                on_user_utterance=lambda t: None,
                stt_model="tiny", tts_engine="system",
            )
        self.assertIsNotNone(pipe)
        self.assertIs(rtv.get_pipeline(), pipe)
        self.assertEqual(pipe.stt_model, "tiny")

    def test_start_pipeline_reuses_running_singleton(self):
        with mock.patch.object(rtv.RealtimeVoicePipeline, "start",
                               return_value=True):
            first = rtv.start_pipeline(voice_mode="realtime")
            # Mark it running so the second call returns the same object
            # without building/starting a new one.
            with mock.patch.object(first, "is_running", return_value=True):
                second = rtv.start_pipeline(voice_mode="realtime")
        self.assertIs(first, second)

    def test_start_pipeline_case_insensitive_mode(self):
        with mock.patch.object(rtv.RealtimeVoicePipeline, "start",
                               return_value=True):
            pipe = rtv.start_pipeline(voice_mode="REALTIME")
        self.assertIsNotNone(pipe)

    def test_stop_pipeline_noop_when_none(self):
        rtv.stop_pipeline()  # must not raise
        self.assertIsNone(rtv.get_pipeline())

    def test_stop_pipeline_stops_and_clears(self):
        with mock.patch.object(rtv.RealtimeVoicePipeline, "start",
                               return_value=True):
            pipe = rtv.start_pipeline(voice_mode="realtime")
        with mock.patch.object(pipe, "stop") as st:
            rtv.stop_pipeline()
        st.assert_called_once()
        self.assertIsNone(rtv.get_pipeline())

    def test_stop_pipeline_swallows_stop_exception(self):
        with mock.patch.object(rtv.RealtimeVoicePipeline, "start",
                               return_value=True):
            pipe = rtv.start_pipeline(voice_mode="realtime")

        def boom():
            raise RuntimeError("stop boom")

        with mock.patch.object(pipe, "stop", side_effect=boom):
            rtv.stop_pipeline()  # must not raise
        self.assertIsNone(rtv.get_pipeline())


# ──────────────────────────────────────────────────────────────────────
# Integration-ish: a full feed → wait → barge cycle on the fakes
# ──────────────────────────────────────────────────────────────────────

class PipelineFlowTests(unittest.TestCase):
    def test_feed_then_drain_then_partial_barge(self):
        """Exercise the realtime contract end-to-end on the fakes:
        feed two chunks (playback starts), drain via wait_for_playback, then a
        long partial while a new playback is in flight fires barge_in once."""
        bargein = []
        p = rtv.RealtimeVoicePipeline(on_barge_in=lambda: bargein.append(1))
        s = FakeTTSStream()
        p._tts_stream = s

        p.feed_response_chunk("Hello, sir. ")
        p.feed_response_chunk("Right away.")
        self.assertEqual(s.fed, ["Hello, sir. ", "Right away."])
        self.assertEqual(s.play_async_calls, 1)   # second feed sees is_playing
        self.assertTrue(p.is_playing())

        # Drain.
        s._playing = False
        self.assertTrue(p.wait_for_playback())
        self.assertFalse(p.is_playing())

        # New playback in flight, user starts talking (long partial) → barge.
        s._playing = True
        p._playing.set()
        p._on_partial("stop please")
        self.assertEqual(bargein, [1])
        self.assertFalse(p.is_playing())


if __name__ == "__main__":
    unittest.main()
