"""First-ever unit tests for a SECTION of the bobert_companion.py monolith.

SECTION 3 — the top-level functions/classes defined between lines 4443 and
6881 of bobert_companion.py. That band is the audio-capture → STT → local-LLM
fallback → TTS-synthesis → playback spine:

  * mic plumbing            record_speech / get_mic_buffer / tap registry
  * audio-processor gating  _process_capture_chunk / _feed_playback_reference
  * stream teardown         _safe_close_stream
  * Whisper STT             _resolve_whisper_device / _ensure_whisper /
                            transcribe / CUDA-DLL remediation helpers
  * dependency check        _parse_requirements / check_dependencies
  * tone/emotion wrappers   detect_tone / route_voice_emotion
  * local-LLM fallback      _ollama_* / _get_local_llm_model / _call_local_llm
                            / _local_fallback_or / _call_local_vision
  * main LLM dispatch       _call_llm
  * TTS tag parsing         _parse_mood_tag / _parse_intent_tag
  * TTS prosody + render    _resolve_tts_preset / _render_edge_tts / synthesise
                            / _pyttsx3_tts / _silent_clip / _try_sapi5_then_silence
  * playback + ducking      is_using_headset / _start_barge_in_listener /
                            _AudioDucker / play_with_lipsync

Everything external is mocked: no real mic/cv2/sounddevice, no network, no
LLM, no threads doing real work, no filesystem outside tempfiles, no
time.sleep stalls. The monolith is imported ONCE via the harness cache; we
only ever patch ``bc`` attributes per-test (and restore any directly-mutated
module globals in tearDown). Real numpy stays intact.

Runs in the LOCAL full tier only (heavy deps present); skips on the
light-deps CI runner via ``@requires_monolith``.
"""
from __future__ import annotations

import io
import os
import queue
import sys
import threading
import unittest
from unittest import mock

import numpy as np

from tests._monolith_harness import (
    MonolithGlobalsTestCase, load_monolith, requires_monolith)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
class _FakeResp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, ok=True, status_code=200, json_data=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text

    def json(self):
        return self._json

    # support `with requests.post(...) as r:` (streamed pulls)
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self):
        return iter(())


class _ImmediateThread:
    """Drop-in for threading.Thread that runs target() synchronously on
    start() — lets us exercise the body of code that the monolith hands to a
    daemon thread without leaving a real thread running."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon
        self.name = name

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False


# ===========================================================================
# Audio-processor gating: _process_capture_chunk / _feed_playback_reference
# ===========================================================================
@requires_monolith
class ProcessCaptureChunkTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self.chunk = np.ones(16, dtype=np.float32)

    def test_passthrough_when_master_disabled(self):
        with mock.patch.object(self.bc, "_audio_master_enabled", [False]):
            out = self.bc._process_capture_chunk(self.chunk)
        self.assertIs(out, self.chunk)

    def test_passthrough_when_processor_none(self):
        with mock.patch.object(self.bc, "_audio_master_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_processor", None):
            out = self.bc._process_capture_chunk(self.chunk)
        self.assertIs(out, self.chunk)

    def test_delegates_to_processor_with_stage_flags(self):
        proc = mock.Mock()
        processed = np.zeros(16, dtype=np.float32)
        proc.process.return_value = processed
        ap = mock.Mock()
        ap.get_processor.return_value = proc
        with mock.patch.object(self.bc, "_audio_master_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_aec_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_ns_enabled", [False]), \
                mock.patch.object(self.bc, "_audio_agc_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_processor", ap):
            out = self.bc._process_capture_chunk(self.chunk, 16000)
        ap.get_processor.assert_called_once_with(16000)
        _, kwargs = proc.process.call_args
        self.assertEqual(kwargs, {"enable_aec": True, "enable_ns": False,
                                  "enable_agc": True})
        self.assertIs(out, processed)

    def test_processor_exception_falls_through_to_raw(self):
        ap = mock.Mock()
        ap.get_processor.side_effect = RuntimeError("boom")
        with mock.patch.object(self.bc, "_audio_master_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_processor", ap), \
                mock.patch.object(self.bc, "_debug_mode", [False]):
            out = self.bc._process_capture_chunk(self.chunk)
        self.assertIs(out, self.chunk)

    def test_processor_exception_with_debug_prints_and_falls_through(self):
        # 4461-4462: same failure but with debug mode on -> the pass-through is
        # logged before returning the raw chunk.
        ap = mock.Mock()
        ap.get_processor.side_effect = RuntimeError("proc boom")
        with mock.patch.object(self.bc, "_audio_master_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_processor", ap), \
                mock.patch.object(self.bc, "_debug_mode", [True]):
            out = self.bc._process_capture_chunk(self.chunk)
        self.assertIs(out, self.chunk)


@requires_monolith
class FeedPlaybackReferenceTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_noop_when_disabled(self):
        ap = mock.Mock()
        with mock.patch.object(self.bc, "_audio_master_enabled", [False]), \
                mock.patch.object(self.bc, "_audio_processor", ap):
            self.bc._feed_playback_reference(np.zeros(4, dtype=np.float32), 24000)
        ap.feed_playback.assert_not_called()

    def test_forwards_to_processor(self):
        ap = mock.Mock()
        buf = np.zeros(4, dtype=np.float32)
        with mock.patch.object(self.bc, "_audio_master_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_processor", ap):
            self.bc._feed_playback_reference(buf, 24000)
        ap.feed_playback.assert_called_once_with(buf, sample_rate=24000)

    def test_processor_error_swallowed(self):
        ap = mock.Mock()
        ap.feed_playback.side_effect = ValueError("nope")
        with mock.patch.object(self.bc, "_audio_master_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_processor", ap):
            # must not raise
            self.bc._feed_playback_reference(np.zeros(4, dtype=np.float32), 24000)


# ===========================================================================
# _safe_close_stream
# ===========================================================================
@requires_monolith
class SafeCloseStreamTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_none_stream_is_noop(self):
        self.bc._safe_close_stream(None)  # no raise

    def test_stop_then_close_called(self):
        stream = mock.Mock()
        # real thread used here, but close() returns instantly so wait() passes
        self.bc._safe_close_stream(stream, timeout_sec=1.0)
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    def test_stop_raises_still_closes(self):
        stream = mock.Mock()
        stream.stop.side_effect = RuntimeError("stop boom")
        self.bc._safe_close_stream(stream, timeout_sec=1.0)
        stream.close.assert_called_once()

    def test_hung_close_forces_sd_stop(self):
        # close() blocks past the timeout → the helper must call sd.stop().
        release = threading.Event()
        stream = mock.Mock()

        def _slow_close():
            release.wait(2.0)
        stream.close.side_effect = _slow_close
        fake_sd = mock.Mock()
        try:
            with mock.patch.object(self.bc, "sd", fake_sd):
                self.bc._safe_close_stream(stream, timeout_sec=0.05)
            fake_sd.stop.assert_called_once()
        finally:
            release.set()  # let the daemon close-thread finish


# ===========================================================================
# Record-tap registry: _fanout_record_frame / add_record_tap / remove_record_tap
# ===========================================================================
@requires_monolith
class RecordTapRegistryTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        # snapshot + clear the shared tap list and active flag
        self._saved_taps = list(self.bc._record_speech_taps)
        self._saved_active = list(self.bc._record_speech_active)
        self.bc._record_speech_taps.clear()

    def tearDown(self):
        self.bc._record_speech_taps[:] = self._saved_taps
        self.bc._record_speech_active[:] = self._saved_active

    def test_add_returns_active_state_and_registers(self):
        q = queue.Queue()
        self.bc._record_speech_active[0] = True
        self.assertTrue(self.bc.add_record_tap(q))
        self.assertIn(q, self.bc._record_speech_taps)

    def test_add_when_idle_returns_false(self):
        q = queue.Queue()
        self.bc._record_speech_active[0] = False
        self.assertFalse(self.bc.add_record_tap(q))

    def test_add_is_idempotent(self):
        q = queue.Queue()
        self.bc.add_record_tap(q)
        self.bc.add_record_tap(q)
        self.assertEqual(self.bc._record_speech_taps.count(q), 1)

    def test_remove_unregistered_is_safe(self):
        self.bc.remove_record_tap(queue.Queue())  # no raise

    def test_fanout_delivers_to_all_taps(self):
        q1, q2 = queue.Queue(), queue.Queue()
        self.bc.add_record_tap(q1)
        self.bc.add_record_tap(q2)
        frame = np.ones(8, dtype=np.float32)
        self.bc._fanout_record_frame(frame)
        self.assertIs(q1.get_nowait(), frame)
        self.assertIs(q2.get_nowait(), frame)

    def test_fanout_no_taps_is_noop(self):
        self.bc._fanout_record_frame(np.ones(2, dtype=np.float32))  # no raise

    def test_fanout_survives_bad_consumer(self):
        bad = mock.Mock()
        bad.put_nowait.side_effect = RuntimeError("full")
        good = queue.Queue()
        self.bc._record_speech_taps.append(bad)
        self.bc._record_speech_taps.append(good)
        frame = np.zeros(2, dtype=np.float32)
        self.bc._fanout_record_frame(frame)  # bad tap must not abort the loop
        self.assertIs(good.get_nowait(), frame)


# ===========================================================================
# record_speech / get_mic_buffer — mic-disabled short-circuit paths
# ===========================================================================
@requires_monolith
class RecordSpeechDisabledTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_returns_none_when_mic_disabled(self):
        with mock.patch.object(self.bc, "_mic_input_disabled", return_value=True), \
                mock.patch.object(self.bc.time, "sleep") as slp:
            self.assertIsNone(self.bc.record_speech(timeout=5.0))
        slp.assert_called_once()

    def test_mic_disabled_sleep_clamped_for_short_timeout(self):
        with mock.patch.object(self.bc, "_mic_input_disabled", return_value=True), \
                mock.patch.object(self.bc.time, "sleep") as slp:
            self.bc.record_speech(timeout=0.2)
        # min(0.5, max(0.1, 0.2)) == 0.2
        self.assertAlmostEqual(slp.call_args[0][0], 0.2, places=6)

    def test_mic_disabled_no_timeout_sleeps_half_second(self):
        with mock.patch.object(self.bc, "_mic_input_disabled", return_value=True), \
                mock.patch.object(self.bc.time, "sleep") as slp:
            self.bc.record_speech(timeout=None)
        self.assertAlmostEqual(slp.call_args[0][0], 0.5, places=6)


@requires_monolith
class GetMicBufferDisabledTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_returns_none_when_mic_disabled(self):
        with mock.patch.object(self.bc, "_mic_input_disabled", return_value=True):
            self.assertIsNone(self.bc.get_mic_buffer(1.0))

    def test_pathB_open_failure_returns_none(self):
        # No wake listener, record_speech not active → Path B opens a stream;
        # force the open to fail and assert a graceful None.
        fake_sd = mock.Mock()
        fake_sd.InputStream.side_effect = RuntimeError("device gone")
        with mock.patch.object(self.bc, "_mic_input_disabled", return_value=False), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.dict(self.bc.sys.modules, {}, clear=False), \
                mock.patch.object(self.bc, "sd", fake_sd), \
                mock.patch.object(self.bc, "get_input_device", return_value=None):
            # ensure no wake listener module is present
            self.bc.sys.modules.pop("skill_wake_listener", None)
            out = self.bc.get_mic_buffer(0.1)
        self.assertIsNone(out)


# ===========================================================================
# CUDA-DLL helpers + whisper device resolution
# ===========================================================================
@requires_monolith
class CudaDllHelperTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_is_cuda_dll_error_matches_known_patterns(self):
        self.assertTrue(self.bc._is_cuda_dll_error(
            RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")))
        self.assertTrue(self.bc._is_cuda_dll_error(
            OSError("could not load library cudnn64_9.dll")))

    def test_is_cuda_dll_error_rejects_unrelated(self):
        self.assertFalse(self.bc._is_cuda_dll_error(ValueError("out of memory")))
        self.assertFalse(self.bc._is_cuda_dll_error(KeyError("bad model name")))

    def test_remediation_note_includes_reason(self):
        fn = self.bc._register_cuda_dll_dirs
        with mock.patch.object(fn, "_reason", "nvidia pip namespace not importable",
                               create=True), \
                mock.patch.object(fn, "_registered", [], create=True), \
                mock.patch.object(fn, "_missing", [], create=True):
            note = self.bc._cuda_dll_remediation_note()
        self.assertIn("cublas64_12.dll", note)
        self.assertIn("nvidia pip namespace not importable", note)

    def test_remediation_note_includes_counts_when_no_reason(self):
        fn = self.bc._register_cuda_dll_dirs
        with mock.patch.object(fn, "_reason", None, create=True), \
                mock.patch.object(fn, "_registered", ["a", "b"], create=True), \
                mock.patch.object(fn, "_missing", ["c"], create=True):
            note = self.bc._cuda_dll_remediation_note()
        self.assertIn("registered 2", note)
        self.assertIn("missing 1", note)


@requires_monolith
class ResolveWhisperDeviceTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_explicit_cpu(self):
        with mock.patch.object(self.bc, "WHISPER_DEVICE", "cpu"):
            self.assertEqual(self.bc._resolve_whisper_device(), "cpu")

    def test_explicit_cuda_honoured_verbatim(self):
        with mock.patch.object(self.bc, "WHISPER_DEVICE", "cuda"):
            self.assertEqual(self.bc._resolve_whisper_device(), "cuda")

    def test_none_defaults_to_auto_and_falls_back_cpu(self):
        # auto with both ctranslate2 + torch reporting no GPU → cpu.
        fake_ct2 = mock.Mock()
        fake_ct2.get_cuda_device_count.return_value = 0
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = False
        with mock.patch.object(self.bc, "WHISPER_DEVICE", None), \
                mock.patch.dict(sys.modules, {"ctranslate2": fake_ct2,
                                              "torch": fake_torch}):
            self.assertEqual(self.bc._resolve_whisper_device(), "cpu")

    def test_auto_picks_cuda_when_ctranslate2_sees_gpu(self):
        fake_ct2 = mock.Mock()
        fake_ct2.get_cuda_device_count.return_value = 1
        with mock.patch.object(self.bc, "WHISPER_DEVICE", "auto"), \
                mock.patch.dict(sys.modules, {"ctranslate2": fake_ct2}):
            self.assertEqual(self.bc._resolve_whisper_device(), "cuda")


# ===========================================================================
# _register_cuda_dll_dirs — the early-return / nvidia-missing branches
# ===========================================================================
@requires_monolith
class RegisterCudaDllDirsTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        # snapshot the function-object cache attributes so we don't leak the
        # "_done" sentinel between tests / to the rest of the suite.
        fn = self.bc._register_cuda_dll_dirs
        self._saved = {k: getattr(fn, k) for k in
                       ("_done", "_registered", "_missing", "_reason")
                       if hasattr(fn, k)}

    def tearDown(self):
        fn = self.bc._register_cuda_dll_dirs
        for k in ("_done", "_registered", "_missing", "_reason"):
            if k in self._saved:
                setattr(fn, k, self._saved[k])
            elif hasattr(fn, k):
                delattr(fn, k)

    def test_already_done_short_circuits(self):
        fn = self.bc._register_cuda_dll_dirs
        fn._done = True
        # If it short-circuits it won't touch sys.modules / import nvidia.
        with mock.patch.dict(sys.modules, {}, clear=False):
            sentinel = object()
            sys.modules["nvidia"] = sentinel  # would be used if it proceeded
            fn()
            # _registered should be untouched (still whatever it was)
        # nothing to assert beyond "did not raise"; reaching here is success.

    def test_nvidia_import_missing_sets_reason(self):
        fn = self.bc._register_cuda_dll_dirs
        fn._done = False

        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "nvidia":
                raise ImportError("No module named 'nvidia'")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            fn()
        self.assertTrue(fn._done)
        self.assertIsNotNone(fn._reason)
        self.assertIn("nvidia", fn._reason)


# ===========================================================================
# _ensure_whisper — engine selection via mocked faster_whisper
# ===========================================================================
@requires_monolith
class EnsureWhisperTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = (self.bc._stt, self.bc._stt_device,
                       self.bc._stt_model_name, self.bc._stt_engine)
        self.bc._stt = None

    def tearDown(self):
        (self.bc._stt, self.bc._stt_device,
         self.bc._stt_model_name, self.bc._stt_engine) = self._saved

    def test_noop_when_already_loaded(self):
        self.bc._stt = object()
        with mock.patch.object(self.bc, "_register_cuda_dll_dirs") as reg:
            self.bc._ensure_whisper()
        reg.assert_not_called()

    def test_loads_faster_whisper_on_cpu(self):
        fake_model_obj = object()
        fake_wm = mock.Mock(return_value=fake_model_obj)
        fake_fw_module = mock.Mock()
        fake_fw_module.WhisperModel = fake_wm
        with mock.patch.object(self.bc, "_register_cuda_dll_dirs"), \
                mock.patch.object(self.bc, "_resolve_whisper_device",
                                  return_value="cpu"), \
                mock.patch.object(self.bc, "WHISPER_MODEL_CPU", "base"), \
                mock.patch.dict(sys.modules, {"faster_whisper": fake_fw_module}):
            self.bc._ensure_whisper()
        self.assertIs(self.bc._stt, fake_model_obj)
        self.assertEqual(self.bc._stt_engine, "faster_whisper")
        self.assertEqual(self.bc._stt_device, "cpu")
        # compute_type int8 on cpu
        _, kwargs = fake_wm.call_args
        self.assertEqual(kwargs.get("compute_type"), "int8")
        self.assertEqual(kwargs.get("device"), "cpu")

    def test_force_cpu_int8_overrides_cuda(self):
        fake_wm = mock.Mock(return_value=object())
        fake_fw_module = mock.Mock()
        fake_fw_module.WhisperModel = fake_wm
        with mock.patch.object(self.bc, "_register_cuda_dll_dirs"), \
                mock.patch.object(self.bc, "_resolve_whisper_device",
                                  return_value="cuda"), \
                mock.patch.object(self.bc, "_force_whisper_cpu_int8", True), \
                mock.patch.object(self.bc, "WHISPER_MODEL_CPU", "base"), \
                mock.patch.dict(sys.modules, {"faster_whisper": fake_fw_module}):
            self.bc._ensure_whisper()
        _, kwargs = fake_wm.call_args
        self.assertEqual(kwargs.get("device"), "cpu")
        self.assertEqual(self.bc._stt_device, "cpu")


# ===========================================================================
# transcribe — abstraction over faster-whisper / openai-whisper
# ===========================================================================
class _Seg:
    def __init__(self, text, no_speech_prob=0.1, avg_logprob=-0.2):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


@requires_monolith
class TranscribeTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = (self.bc._stt, self.bc._stt_engine)

    def tearDown(self):
        self.bc._stt, self.bc._stt_engine = self._saved

    def test_faster_whisper_aggregates_segments(self):
        fake_stt = mock.Mock()
        info = mock.Mock(no_speech_prob=0.3)
        segs = [_Seg(" hello ", 0.1, -0.2), _Seg("world", 0.3, -0.4)]
        fake_stt.transcribe.return_value = (iter(segs), info)
        with mock.patch.object(self.bc, "_ensure_whisper"), \
                mock.patch.object(self.bc, "_stt", fake_stt), \
                mock.patch.object(self.bc, "_stt_engine", "faster_whisper"):
            text, conf = self.bc.transcribe(np.zeros(8, dtype=np.float32))
        self.assertEqual(text, "hello world")
        self.assertAlmostEqual(conf["no_speech_prob"], 0.2, places=6)
        self.assertAlmostEqual(conf["avg_logprob"], -0.3, places=6)

    def test_faster_whisper_empty_segments(self):
        fake_stt = mock.Mock()
        info = mock.Mock(no_speech_prob=0.9)
        fake_stt.transcribe.return_value = (iter(()), info)
        with mock.patch.object(self.bc, "_ensure_whisper"), \
                mock.patch.object(self.bc, "_stt", fake_stt), \
                mock.patch.object(self.bc, "_stt_engine", "faster_whisper"):
            text, conf = self.bc.transcribe(np.zeros(8, dtype=np.float32))
        self.assertEqual(text, "")
        self.assertAlmostEqual(conf["no_speech_prob"], 0.9, places=6)
        self.assertEqual(conf["avg_logprob"], -10.0)

    def test_openai_whisper_path(self):
        fake_stt = mock.Mock()
        fake_stt.transcribe.return_value = {
            "text": "  greetings ",
            "segments": [{"no_speech_prob": 0.2, "avg_logprob": -0.5},
                         {"no_speech_prob": 0.4, "avg_logprob": -0.5}],
        }
        with mock.patch.object(self.bc, "_ensure_whisper"), \
                mock.patch.object(self.bc, "_stt", fake_stt), \
                mock.patch.object(self.bc, "_stt_engine", "openai_whisper"):
            text, conf = self.bc.transcribe(np.zeros(8, dtype=np.float32))
        self.assertEqual(text, "greetings")
        self.assertAlmostEqual(conf["no_speech_prob"], 0.3, places=6)
        self.assertAlmostEqual(conf["avg_logprob"], -0.5, places=6)

    def test_exception_returns_empty_and_drops_model_on_cuda_oom(self):
        with mock.patch.object(self.bc, "_ensure_whisper",
                               side_effect=RuntimeError("CUDA out of memory")), \
                mock.patch.object(self.bc, "_stt", object()), \
                mock.patch.dict(sys.modules, {"torch": mock.Mock(
                    cuda=mock.Mock(is_available=mock.Mock(return_value=False)))}):
            text, conf = self.bc.transcribe(np.zeros(4, dtype=np.float32))
            # model dropped (inside the patch context) so the next utterance
            # reloads cleanly from a fresh GPU state.
            self.assertIsNone(self.bc._stt)
        self.assertEqual(text, "")
        self.assertEqual(conf["no_speech_prob"], 1.0)

    def test_generic_exception_keeps_model(self):
        sentinel = mock.Mock()
        sentinel.transcribe.side_effect = ValueError("bad audio")
        with mock.patch.object(self.bc, "_ensure_whisper"), \
                mock.patch.object(self.bc, "_stt", sentinel), \
                mock.patch.object(self.bc, "_stt_engine", "faster_whisper"):
            text, conf = self.bc.transcribe(np.zeros(4, dtype=np.float32))
            # Assert INSIDE the patch context: a non-CUDA error must NOT drop
            # the model (the CUDA path sets _stt=None; this path leaves it).
            self.assertIs(self.bc._stt, sentinel)
        self.assertEqual(text, "")
        self.assertEqual(conf["no_speech_prob"], 1.0)


# ===========================================================================
# _parse_requirements / check_dependencies
# ===========================================================================
@requires_monolith
class ParseRequirementsTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _write(self, text):
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_strips_versions_comments_and_options(self):
        path = self._write(
            "# a comment\n"
            "\n"
            "paho-mqtt>=2.0  # inline comment\n"
            "numpy==1.26.0\n"
            "requests\n"
            "--extra-index-url https://example.test/simple\n"
            "-r other.txt\n"
            "pkg[extra]>=1.0\n"
            "marked; python_version>='3.10'\n"
        )
        pkgs = self.bc._parse_requirements(path)
        self.assertEqual(
            pkgs,
            ["paho-mqtt", "numpy", "requests", "pkg", "marked"],
        )

    def test_missing_file_returns_empty(self):
        self.assertEqual(
            self.bc._parse_requirements(r"C:\nope\does-not-exist.txt"), [])


@requires_monolith
class CheckDependenciesTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_all_present_returns_empty(self):
        with mock.patch.object(self.bc, "_parse_requirements",
                               return_value=["numpy", "requests"]):
            missing = self.bc.check_dependencies()
        self.assertEqual(missing, [])

    def test_missing_package_triggers_spoken_alert(self):
        # psutil "missing" → has a feature note → _speak should fire once.
        # check_dependencies() resolves imports via importlib.import_module,
        # so that's the seam to fail (NOT builtins.__import__).
        import importlib
        real_import_module = importlib.import_module

        def fake_import_module(name, *a, **k):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import_module(name, *a, **k)

        speak = mock.Mock()
        hist = []
        with mock.patch.object(self.bc, "_parse_requirements",
                               return_value=["psutil"]), \
                mock.patch.object(importlib, "import_module",
                                  side_effect=fake_import_module), \
                mock.patch.object(self.bc, "_speak", speak), \
                mock.patch.object(self.bc, "conversation_history", hist):
            missing = self.bc.check_dependencies()
        self.assertEqual(missing, ["psutil"])
        speak.assert_called_once()
        self.assertIn("system monitor", speak.call_args[0][0])
        # the alert is also appended to history
        self.assertTrue(hist and hist[-1]["role"] == "assistant")

    def test_missing_without_feature_note_no_speak(self):
        import importlib
        real_import_module = importlib.import_module

        def fake_import_module(name, *a, **k):
            if name == "somerandompkg":
                raise ImportError("nope")
            return real_import_module(name, *a, **k)

        speak = mock.Mock()
        with mock.patch.object(self.bc, "_parse_requirements",
                               return_value=["somerandompkg"]), \
                mock.patch.object(importlib, "import_module",
                                  side_effect=fake_import_module), \
                mock.patch.object(self.bc, "_speak", speak):
            missing = self.bc.check_dependencies()
        self.assertEqual(missing, ["somerandompkg"])
        speak.assert_not_called()

    def test_dashed_pip_names_resolve_to_correct_import_modules(self):
        # These three pip distributions have import paths that the default
        # "-"→"_" fallback gets WRONG, so without explicit map entries they were
        # falsely reported MISSING (+ a bogus pip-install line) on every boot of
        # a working install. Assert the map resolves each to its real module.
        cases = {
            "winrt-Windows.Media.Control": "winrt.windows.media.control",
            "nvidia-cublas-cu12":          "nvidia.cublas",
            "nvidia-cudnn-cu12":           "nvidia.cudnn",
        }
        for pkg, expected_mod in cases.items():
            with self.subTest(pkg=pkg):
                # This mirrors the exact resolution check_dependencies() does.
                resolved = self.bc._DEP_IMPORT_NAME.get(
                    pkg, pkg.replace("-", "_"))
                self.assertEqual(resolved, expected_mod)
                # And the buggy fallback would NOT have produced the right name.
                self.assertNotEqual(pkg.replace("-", "_"), expected_mod)

    def test_mapped_dashed_names_not_reported_missing_when_importable(self):
        # End-to-end: when the mapped import modules resolve (mocked present),
        # check_dependencies() must NOT list these packages as missing.
        import importlib
        present = {
            "winrt.windows.media.control": object(),
            "nvidia.cublas":               object(),
            "nvidia.cudnn":                object(),
        }

        def fake_import_module(name, *a, **k):
            if name in present:
                return present[name]
            raise ImportError(f"unexpected import {name!r}")

        with mock.patch.object(
                self.bc, "_parse_requirements",
                return_value=list(
                    ["winrt-Windows.Media.Control",
                     "nvidia-cublas-cu12", "nvidia-cudnn-cu12"])), \
                mock.patch.object(importlib, "import_module",
                                  side_effect=fake_import_module):
            missing = self.bc.check_dependencies()
        self.assertEqual(missing, [])


# ===========================================================================
# detect_tone / route_voice_emotion thin wrappers
# ===========================================================================
@requires_monolith
class ToneWrapperTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_detect_tone_passes_prev_user_text(self):
        hist = [
            {"role": "user", "content": "first thing"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second thing"},
        ]
        fake_td = mock.Mock()
        fake_td.detect_tone.return_value = "rushed"
        with mock.patch.object(self.bc, "conversation_history", hist), \
                mock.patch.object(self.bc, "_tone_detector", fake_td):
            out = self.bc.detect_tone("second thing")
        self.assertEqual(out, "rushed")
        _, kwargs = fake_td.detect_tone.call_args
        # prev_user_text is the most recent *earlier* user line
        self.assertEqual(kwargs.get("prev_user_text"), "first thing")

    def test_route_voice_emotion_forwards_now_and_prev(self):
        hist = [{"role": "user", "content": "earlier"},
                {"role": "user", "content": "now"}]
        fake_ve = mock.Mock()
        fake_ve.route_voice_emotion.return_value = {"mood": "casual",
                                                    "addendum": ""}
        with mock.patch.object(self.bc, "conversation_history", hist), \
                mock.patch.object(self.bc, "_voice_emotion", fake_ve):
            out = self.bc.route_voice_emotion("now", now=123.0)
        self.assertEqual(out["mood"], "casual")
        _, kwargs = fake_ve.route_voice_emotion.call_args
        self.assertEqual(kwargs.get("now"), 123.0)
        self.assertEqual(kwargs.get("prev_user_text"), "earlier")


# ===========================================================================
# Ollama reachability / model presence helpers
# ===========================================================================
@requires_monolith
class OllamaProbeTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_alive_true_on_ok(self):
        fake_req = mock.Mock()
        fake_req.get.return_value = _FakeResp(ok=True)
        with mock.patch.object(self.bc, "requests", fake_req):
            self.assertTrue(self.bc._ollama_alive())

    def test_alive_false_on_exception(self):
        fake_req = mock.Mock()
        fake_req.get.side_effect = OSError("conn refused")
        with mock.patch.object(self.bc, "requests", fake_req):
            self.assertFalse(self.bc._ollama_alive())

    def test_has_model_exact_match(self):
        fake_req = mock.Mock()
        fake_req.get.return_value = _FakeResp(
            ok=True, json_data={"models": [{"name": "llama3.1:8b-instruct-q5_K_M"}]})
        with mock.patch.object(self.bc, "requests", fake_req):
            self.assertTrue(self.bc._ollama_has_model("llama3.1:8b-instruct-q5_K_M"))

    def test_has_model_base_name_match(self):
        fake_req = mock.Mock()
        fake_req.get.return_value = _FakeResp(
            ok=True, json_data={"models": [{"name": "qwen2.5:latest"}]})
        with mock.patch.object(self.bc, "requests", fake_req):
            self.assertTrue(self.bc._ollama_has_model("qwen2.5:14b-instruct"))

    def test_has_model_absent(self):
        fake_req = mock.Mock()
        fake_req.get.return_value = _FakeResp(
            ok=True, json_data={"models": [{"name": "phi3:mini"}]})
        with mock.patch.object(self.bc, "requests", fake_req):
            self.assertFalse(self.bc._ollama_has_model("llama3.1"))

    def test_has_model_http_not_ok(self):
        fake_req = mock.Mock()
        fake_req.get.return_value = _FakeResp(ok=False, status_code=500)
        with mock.patch.object(self.bc, "requests", fake_req):
            self.assertFalse(self.bc._ollama_has_model("anything"))


# ===========================================================================
# _get_local_llm_model — resolution priority + caching
# ===========================================================================
@requires_monolith
class GetLocalLlmModelTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_cache = list(self.bc._RESOLVED_LOCAL_LLM_MODEL)
        self.bc._RESOLVED_LOCAL_LLM_MODEL[0] = None
        self._saved_env = os.environ.get("JARVIS_LOCAL_LLM_MODEL")
        os.environ.pop("JARVIS_LOCAL_LLM_MODEL", None)

    def tearDown(self):
        self.bc._RESOLVED_LOCAL_LLM_MODEL[:] = self._saved_cache
        if self._saved_env is None:
            os.environ.pop("JARVIS_LOCAL_LLM_MODEL", None)
        else:
            os.environ["JARVIS_LOCAL_LLM_MODEL"] = self._saved_env

    def test_returns_cached_value_first(self):
        self.bc._RESOLVED_LOCAL_LLM_MODEL[0] = "cached:model"
        with mock.patch.object(self.bc, "requests") as req:
            self.assertEqual(self.bc._get_local_llm_model(), "cached:model")
        req.get.assert_not_called()

    def test_env_override_wins(self):
        os.environ["JARVIS_LOCAL_LLM_MODEL"] = "  custom:tag  "
        with mock.patch.object(self.bc, "_log_gpu_state"):
            out = self.bc._get_local_llm_model()
        self.assertEqual(out, "custom:tag")
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], "custom:tag")

    def test_picks_preference_when_installed(self):
        pref = self.bc._LOCAL_LLM_PREFERENCE[0]
        fake_req = mock.Mock()
        fake_req.get.return_value = _FakeResp(
            ok=True, json_data={"models": [{"name": pref}, {"name": "phi3:mini"}]})
        with mock.patch.object(self.bc, "requests", fake_req), \
                mock.patch.object(self.bc, "_log_gpu_state"):
            out = self.bc._get_local_llm_model()
        self.assertEqual(out, pref)

    def test_falls_back_to_first_installed_offlist(self):
        fake_req = mock.Mock()
        fake_req.get.return_value = _FakeResp(
            ok=True, json_data={"models": [{"name": "mistral:7b"}]})
        with mock.patch.object(self.bc, "requests", fake_req), \
                mock.patch.object(self.bc, "_log_gpu_state"):
            out = self.bc._get_local_llm_model()
        self.assertEqual(out, "mistral:7b")

    def test_no_models_returns_default_without_caching(self):
        fake_req = mock.Mock()
        fake_req.get.return_value = _FakeResp(ok=True, json_data={"models": []})
        with mock.patch.object(self.bc, "requests", fake_req), \
                mock.patch.object(self.bc, "LOCAL_LLM_MODEL", "default:model"):
            out = self.bc._get_local_llm_model()
        self.assertEqual(out, "default:model")
        # NOT cached — a later finished pull should still be picked up
        self.assertIsNone(self.bc._RESOLVED_LOCAL_LLM_MODEL[0])


# ===========================================================================
# Async install/pull helpers — trigger-once latch + synchronous body run
# ===========================================================================
@requires_monolith
class OllamaAsyncHelperTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = (list(self.bc._OLLAMA_INSTALL_TRIGGERED),
                       list(self.bc._OLLAMA_PULL_TRIGGERED),
                       list(self.bc._LOCAL_VISION_PULL_TRIGGERED))
        self.bc._OLLAMA_INSTALL_TRIGGERED[0] = False
        self.bc._OLLAMA_PULL_TRIGGERED[0] = False
        self.bc._LOCAL_VISION_PULL_TRIGGERED[0] = False

    def tearDown(self):
        (self.bc._OLLAMA_INSTALL_TRIGGERED[:],
         self.bc._OLLAMA_PULL_TRIGGERED[:],
         self.bc._LOCAL_VISION_PULL_TRIGGERED[:]) = self._saved

    def test_install_async_runs_winget_once(self):
        fake_sp = mock.Mock()
        with mock.patch.object(self.bc.threading, "Thread", _ImmediateThread), \
                mock.patch.dict(sys.modules, {"subprocess": fake_sp}):
            self.bc._ollama_install_async()
            self.bc._ollama_install_async()  # latched — second is a no-op
        fake_sp.run.assert_called_once()
        self.assertTrue(self.bc._OLLAMA_INSTALL_TRIGGERED[0])

    def test_pull_async_streams_once(self):
        fake_req = mock.Mock()
        fake_req.post.return_value = _FakeResp(ok=True)
        with mock.patch.object(self.bc.threading, "Thread", _ImmediateThread), \
                mock.patch.object(self.bc, "requests", fake_req):
            self.bc._ollama_pull_async("some:model")
            self.bc._ollama_pull_async("some:model")
        fake_req.post.assert_called_once()
        self.assertTrue(self.bc._OLLAMA_PULL_TRIGGERED[0])

    def test_vision_pull_resets_latch_on_network_error(self):
        fake_req = mock.Mock()
        # The monolith references requests.RequestException for the reset path.
        fake_req.RequestException = self.bc.requests.RequestException
        fake_req.post.side_effect = self.bc.requests.RequestException("down")
        with mock.patch.object(self.bc.threading, "Thread", _ImmediateThread), \
                mock.patch.object(self.bc, "requests", fake_req):
            self.bc._ollama_pull_vision_async("vlm:7b")
        # latch reset so a transient failure doesn't block vision forever
        self.assertFalse(self.bc._LOCAL_VISION_PULL_TRIGGERED[0])

    def test_vision_pull_noop_when_already_triggered(self):
        # 5789-5790: the latch is already set -> early return, no thread/post.
        self.bc._LOCAL_VISION_PULL_TRIGGERED[0] = True
        fake_req = mock.Mock()
        with mock.patch.object(self.bc.threading, "Thread", _ImmediateThread), \
                mock.patch.object(self.bc, "requests", fake_req):
            self.bc._ollama_pull_vision_async("vlm:7b")
        fake_req.post.assert_not_called()

    def test_text_pull_drains_stream_lines(self):
        # 5556-5557: the streamed pull response yields lines -> the drain loop
        # body executes (we just consume them to keep the connection moving).
        class _LineResp(_FakeResp):
            def iter_lines(self):
                return iter([b'{"status":"pulling"}', b'{"status":"done"}'])

        fake_req = mock.Mock()
        fake_req.post.return_value = _LineResp(ok=True)
        with mock.patch.object(self.bc.threading, "Thread", _ImmediateThread), \
                mock.patch.object(self.bc, "requests", fake_req):
            self.bc._ollama_pull_async("some:model")
        fake_req.post.assert_called_once()
        self.assertTrue(self.bc._OLLAMA_PULL_TRIGGERED[0])


# ===========================================================================
# _local_cheatsheet — compact action reference
# ===========================================================================
@requires_monolith
class LocalCheatsheetTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = list(self.bc._LOCAL_CHEATSHEET_CACHE)
        self.bc._LOCAL_CHEATSHEET_CACHE[0] = None

    def tearDown(self):
        self.bc._LOCAL_CHEATSHEET_CACHE[:] = self._saved

    def test_builds_and_lists_registered_actions(self):
        with mock.patch.object(self.bc, "ACTIONS",
                               {"play_music": 1, "set_timer": 2}):
            out = self.bc._local_cheatsheet()
        self.assertIn("[ACTION: name, argument]", out)
        self.assertIn("play_music", out)
        self.assertIn("set_timer", out)
        self.assertIn("END PC CONTROL", out)

    def test_result_is_cached(self):
        with mock.patch.object(self.bc, "ACTIONS", {"a": 1}):
            first = self.bc._local_cheatsheet()
        # second call should return the cached object even if ACTIONS changes
        with mock.patch.object(self.bc, "ACTIONS", {"different": 1}):
            second = self.bc._local_cheatsheet()
        self.assertEqual(first, second)


# ===========================================================================
# _call_local_llm — the gating ladder + payload
# ===========================================================================
@requires_monolith
class CallLocalLlmTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_returns_none_when_fallback_disabled(self):
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", False):
            self.assertIsNone(self.bc._call_local_llm("sys", []))

    def test_kicks_install_when_ollama_dead(self):
        install = mock.Mock()
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=False), \
                mock.patch.object(self.bc, "_ollama_install_async", install):
            self.assertIsNone(self.bc._call_local_llm("sys", []))
        install.assert_called_once()

    def test_kicks_pull_when_model_absent(self):
        pull = mock.Mock()
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_get_local_llm_model",
                                  return_value="m:tag"), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=False), \
                mock.patch.object(self.bc, "_ollama_pull_async", pull):
            self.assertIsNone(self.bc._call_local_llm("sys", []))
        pull.assert_called_once_with("m:tag")

    def test_successful_chat_returns_text_and_appends_directive(self):
        captured = {}
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None):
            captured["payload"] = json
            return _FakeResp(ok=True, json_data={"message": {"content": "  hi sir  "}})
        fake_req.post.side_effect = _post
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_get_local_llm_model",
                                  return_value="m:tag"), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=True), \
                mock.patch.object(self.bc, "requests", fake_req):
            out = self.bc._call_local_llm("BASE PROMPT", [{"role": "user",
                                                           "content": "hello"}])
        self.assertEqual(out, "hi sir")
        sys_msg = captured["payload"]["messages"][0]
        self.assertEqual(sys_msg["role"], "system")
        # the LOCAL_MODE_DIRECTIVE is appended at the most-salient tail
        self.assertIn("YOU ARE RUNNING ON THE LOCAL MODEL", sys_msg["content"])
        self.assertEqual(captured["payload"]["options"]["num_ctx"], 16384)

    def test_http_error_returns_none(self):
        fake_req = mock.Mock()
        fake_req.post.return_value = _FakeResp(ok=False, status_code=503,
                                               text="busy")
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_get_local_llm_model",
                                  return_value="m:tag"), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=True), \
                mock.patch.object(self.bc, "requests", fake_req):
            self.assertIsNone(self.bc._call_local_llm("sys", []))

    def test_swaps_pc_control_prompt_for_cheatsheet(self):
        captured = {}
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None):
            captured["sys"] = json["messages"][0]["content"]
            return _FakeResp(ok=True, json_data={"message": {"content": "ok"}})
        fake_req.post.side_effect = _post
        big_prompt = "PCPROMPT_SENTINEL_BLOCK"
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_get_local_llm_model",
                                  return_value="m:tag"), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=True), \
                mock.patch.object(self.bc, "PC_CONTROL_PROMPT", big_prompt), \
                mock.patch.object(self.bc, "_local_cheatsheet",
                                  return_value="CHEATSHEET_MARK"), \
                mock.patch.object(self.bc, "requests", fake_req):
            self.bc._call_local_llm("intro " + big_prompt + " outro", [])
        self.assertNotIn("PCPROMPT_SENTINEL_BLOCK", captured["sys"])
        self.assertIn("CHEATSHEET_MARK", captured["sys"])


# ===========================================================================
# _local_fallback_or
# ===========================================================================
@requires_monolith
class LocalFallbackOrTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_returns_default_when_local_empty(self):
        with mock.patch.object(self.bc, "_call_local_llm", return_value=None):
            out = self.bc._local_fallback_or("sys", "CLOUD ERR")
        self.assertEqual(out, "CLOUD ERR")

    def test_returns_local_text_without_prefix(self):
        with mock.patch.object(self.bc, "_call_local_llm",
                               return_value="all done sir"):
            out = self.bc._local_fallback_or("sys", "CLOUD ERR")
        self.assertEqual(out, "all done sir")

    def test_strips_stale_local_tag(self):
        with mock.patch.object(self.bc, "_call_local_llm",
                               return_value="[local] hi there"):
            out = self.bc._local_fallback_or("sys", "default")
        self.assertEqual(out, "hi there")


# ===========================================================================
# _call_local_vision
# ===========================================================================
@requires_monolith
class CallLocalVisionTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_disabled_returns_none(self):
        with mock.patch.object(self.bc, "LOCAL_VISION_FALLBACK", False):
            self.assertIsNone(self.bc._call_local_vision("q", [b"png"]))

    def test_no_images_returns_none(self):
        with mock.patch.object(self.bc, "LOCAL_VISION_FALLBACK", True), \
                mock.patch.object(self.bc, "LOCAL_VISION_MODEL", "vlm:7b"):
            self.assertIsNone(self.bc._call_local_vision("q", []))

    def test_model_absent_pulls_and_returns_none(self):
        pull = mock.Mock()
        with mock.patch.object(self.bc, "LOCAL_VISION_FALLBACK", True), \
                mock.patch.object(self.bc, "LOCAL_VISION_MODEL", "vlm:7b"), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=False), \
                mock.patch.object(self.bc, "_ollama_pull_vision_async", pull):
            self.assertIsNone(self.bc._call_local_vision("q", [b"img"]))
        pull.assert_called_once_with("vlm:7b")

    def test_success_returns_text_and_b64_encodes_images(self):
        captured = {}
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None):
            captured["payload"] = json
            return _FakeResp(ok=True, json_data={"message": {"content": " a cat "}})
        fake_req.post.side_effect = _post
        with mock.patch.object(self.bc, "LOCAL_VISION_FALLBACK", True), \
                mock.patch.object(self.bc, "LOCAL_VISION_MODEL", "vlm:7b"), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=True), \
                mock.patch.object(self.bc, "_log_gpu_state"), \
                mock.patch.object(self.bc, "requests", fake_req):
            out = self.bc._call_local_vision("describe", [b"\x89PNG-bytes"])
        self.assertEqual(out, "a cat")
        import base64
        expected_b64 = base64.standard_b64encode(b"\x89PNG-bytes").decode()
        self.assertEqual(captured["payload"]["messages"][0]["images"],
                         [expected_b64])


# ===========================================================================
# _call_llm — the main dispatcher
# ===========================================================================
@requires_monolith
class CallLlmTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        # Isolate the shared conversation buffer for each test.
        self._saved_hist = list(self.bc.conversation_history)
        self.bc.conversation_history.clear()
        # Silence the per-turn classifier side-trips by default.
        self._patches = [
            mock.patch.object(self.bc, "detect_tone", return_value=None),
            mock.patch.object(self.bc, "route_voice_emotion",
                              return_value={"mood": "casual", "addendum": ""}),
            mock.patch.object(self.bc, "_emotion_tracker", None),
            mock.patch.object(self.bc, "_voice_mood_response", None),
            mock.patch.object(self.bc, "_trim_conversation_history"),
            mock.patch.object(self.bc, "_system_prompt", "SYS"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.bc.conversation_history[:] = self._saved_hist

    def _make_phrases(self):
        m = mock.Mock()
        m.detect_phrases_in_reply.return_value = {}
        return m

    def test_claude_happy_path_via_llm_client(self):
        client = mock.Mock()
        client.complete.return_value = "At your service, sir."
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
                mock.patch.object(self.bc, "_llm_client", client), \
                mock.patch.dict(sys.modules, {"anthropic": mock.Mock()}), \
                mock.patch.object(self.bc, "_mcu_phrases", self._make_phrases()):
            reply = self.bc._call_llm("hello")
        self.assertEqual(reply, "At your service, sir.")
        # both the user turn and the assistant reply land in history
        self.assertEqual(self.bc.conversation_history[0]["content"], "hello")
        self.assertEqual(self.bc.conversation_history[-1]["content"],
                         "At your service, sir.")

    def test_claude_credit_balance_error_falls_back_local(self):
        fake_anthropic = mock.Mock()

        class _BadReq(Exception):
            pass
        fake_anthropic.BadRequestError = _BadReq
        fake_anthropic.RateLimitError = type("RL", (Exception,), {})
        fake_anthropic.APIStatusError = type("AS", (Exception,), {})
        client = mock.Mock()
        client.complete.side_effect = _BadReq("Your credit balance is too low")
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
                mock.patch.object(self.bc, "_llm_client", client), \
                mock.patch.dict(sys.modules, {"anthropic": fake_anthropic}), \
                mock.patch.object(self.bc, "_local_fallback_or",
                                  return_value="LOCAL REPLY") as lf, \
                mock.patch.object(self.bc, "_mcu_phrases", self._make_phrases()):
            reply = self.bc._call_llm("do a thing")
        self.assertEqual(reply, "LOCAL REPLY")
        # the *default* passed to the fallback mentions credits
        self.assertIn("credits", lf.call_args[0][1].lower())

    def test_unconfigured_backend_message(self):
        with mock.patch.object(self.bc, "AI_BACKEND", "frobnicate"), \
                mock.patch.object(self.bc, "_mcu_phrases", self._make_phrases()):
            reply = self.bc._call_llm("hi")
        self.assertIn("AI backend not configured", reply)

    def test_ollama_backend_failure_is_caught(self):
        # The ollama path now goes through the bounded helper; a wedged-runner
        # timeout (or any error) raised by it must be caught, not propagated.
        with mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
                mock.patch.object(self.bc, "OLLAMA_MODEL", "m"), \
                mock.patch.object(self.bc, "_ollama_chat_bounded",
                                  side_effect=RuntimeError("ollama down")), \
                mock.patch.object(self.bc, "_mcu_phrases", self._make_phrases()):
            reply = self.bc._call_llm("hi")
        self.assertIn("local model isn't responding", reply)


# ===========================================================================
# TTS tag parsing: _parse_mood_tag / _parse_intent_tag
# ===========================================================================
@requires_monolith
class ParseTagTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_mood_tag_known(self):
        mood, stripped = self.bc._parse_mood_tag("[mood:dry_amused] Indeed, sir.")
        self.assertEqual(mood, "dry_amused")
        self.assertEqual(stripped, "Indeed, sir.")

    def test_mood_tag_unknown_strips_but_returns_none(self):
        mood, stripped = self.bc._parse_mood_tag("[mood:bogus] hello")
        self.assertIsNone(mood)
        self.assertEqual(stripped, "hello")

    def test_mood_tag_absent(self):
        mood, stripped = self.bc._parse_mood_tag("plain text")
        self.assertIsNone(mood)
        self.assertEqual(stripped, "plain text")

    def test_mood_tag_empty_string(self):
        self.assertEqual(self.bc._parse_mood_tag(""), (None, ""))

    def test_intent_tag_known(self):
        # use whatever the live _INTENT_PRESETS map actually contains
        keys = [k for k in self.bc._INTENT_PRESETS.keys()
                if isinstance(k, str)]
        if not keys:
            self.skipTest("no _INTENT_PRESETS available to test against")
        name = keys[0]
        intent, stripped = self.bc._parse_intent_tag(f"[intent:{name}] hi there")
        self.assertEqual(intent, name)
        self.assertEqual(stripped, "hi there")

    def test_intent_tag_unknown_strips_but_returns_none(self):
        intent, stripped = self.bc._parse_intent_tag(
            "[intent:zzz_not_a_real_intent] hello")
        self.assertIsNone(intent)
        self.assertEqual(stripped, "hello")

    def test_intent_tag_absent(self):
        self.assertEqual(self.bc._parse_intent_tag("nope"), (None, "nope"))


# ===========================================================================
# _resolve_tts_preset shim
# ===========================================================================
@requires_monolith
class ResolveTtsPresetTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_returns_neutral_when_tts_layer_missing(self):
        with mock.patch.object(self.bc, "_tts_layer", None):
            preset, params = self.bc._resolve_tts_preset("hi", None)
        self.assertEqual(preset, "neutral")
        self.assertEqual(params["gain"], 1.0)

    def test_forwards_live_state_to_core_tts(self):
        fake_layer = mock.Mock()
        fake_layer.resolve_tts_preset.return_value = ("amused",
                                                      {"rate": "+5%"})
        er = mock.Mock(tts_preset="calm")
        with mock.patch.object(self.bc, "_tts_layer", fake_layer), \
                mock.patch.object(self.bc, "_last_wry", [True]), \
                mock.patch.object(self.bc, "_last_intent_override", ["greet"]), \
                mock.patch.object(self.bc, "_last_mood", ["dry_amused"]), \
                mock.patch.object(self.bc, "_last_user_text", ["yo"]), \
                mock.patch.object(self.bc, "_last_emotion", [er]), \
                mock.patch.dict(sys.modules,
                                {"core.audio_processor": mock.Mock(
                                    recent_peak_rms=lambda: 0.42)}):
            preset, params = self.bc._resolve_tts_preset("hello", "rushed")
        self.assertEqual(preset, "amused")
        _, kwargs = fake_layer.resolve_tts_preset.call_args
        self.assertEqual(kwargs["wry"], True)
        self.assertEqual(kwargs["intent_override"], "greet")
        self.assertEqual(kwargs["mood"], "dry_amused")
        self.assertEqual(kwargs["emotion_preset"], "calm")
        self.assertAlmostEqual(kwargs["peak_rms"], 0.42, places=6)


# ===========================================================================
# _silent_clip / _try_sapi5_then_silence / _pyttsx3_tts
# ===========================================================================
@requires_monolith
class SilentClipTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_shape_and_dtype(self):
        audio, sr = self.bc._silent_clip(sr=24000, ms=80)
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape[0], int(24000 * 80 / 1000))
        self.assertFalse(np.any(audio))

    def test_minimum_one_sample(self):
        audio, _ = self.bc._silent_clip(sr=1, ms=0)
        self.assertEqual(audio.shape[0], 1)


@requires_monolith
class TrySapi5ThenSilenceTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_module_missing_returns_silence(self):
        # Force `from tts.render import render_sapi5` to ImportError.
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "tts.render" or name.startswith("tts.render"):
                raise ImportError("no tts.render")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            audio, sr = self.bc._try_sapi5_then_silence("hi")
        self.assertEqual(audio.dtype, np.float32)
        self.assertFalse(np.any(audio))

    def test_render_success_returned(self):
        fake_mod = mock.Mock()
        good = (np.ones(4, dtype=np.float32), 22050)
        fake_mod.render_sapi5.return_value = good
        with mock.patch.dict(sys.modules, {"tts.render": fake_mod}):
            audio, sr = self.bc._try_sapi5_then_silence("hi", rate="+0%",
                                                        pitch="+0Hz")
        self.assertEqual(sr, 22050)
        self.assertEqual(audio.shape[0], 4)

    def test_render_failure_returns_silence(self):
        fake_mod = mock.Mock()
        fake_mod.render_sapi5.side_effect = RuntimeError("sapi boom")
        with mock.patch.dict(sys.modules, {"tts.render": fake_mod}):
            audio, sr = self.bc._try_sapi5_then_silence("hi")
        self.assertFalse(np.any(audio))


@requires_monolith
class Pyttsx3TtsTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_missing_pyttsx3_chains_to_sapi5(self):
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "pyttsx3":
                raise ImportError("no pyttsx3")
            return real_import(name, *a, **k)

        sentinel = (np.ones(2, dtype=np.float32), 16000)
        with mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch.object(self.bc, "_try_sapi5_then_silence",
                                  return_value=sentinel) as chain:
            out = self.bc._pyttsx3_tts("hello")
        chain.assert_called_once()
        self.assertEqual(out, sentinel)

    def test_engine_failure_chains_to_sapi5(self):
        fake_pyttsx3 = mock.Mock()
        fake_pyttsx3.init.side_effect = RuntimeError("driver init failed")
        sentinel = (np.zeros(2, dtype=np.float32), 16000)
        with mock.patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}), \
                mock.patch.object(self.bc, "_try_sapi5_then_silence",
                                  return_value=sentinel) as chain:
            out = self.bc._pyttsx3_tts("hello")
        chain.assert_called_once()
        self.assertEqual(out, sentinel)


# ===========================================================================
# _render_xtts_or_raise / _render_edge_tts
# ===========================================================================
@requires_monolith
class RenderXttsTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_raises_when_skill_not_loaded(self):
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_custom_voice", None)
            with self.assertRaises(RuntimeError):
                self.bc._render_xtts_or_raise("hi", "+0%", "+0Hz")

    def test_delegates_to_skill_render(self):
        fake_mod = mock.Mock()
        fake_mod.render.return_value = (np.ones(3, dtype=np.float32), 24000)
        with mock.patch.dict(sys.modules, {"skill_custom_voice": fake_mod}):
            audio, sr = self.bc._render_xtts_or_raise("hi", "+1%", "+2Hz")
        fake_mod.render.assert_called_once_with("hi", "+1%", "+2Hz")
        self.assertEqual(sr, 24000)


@requires_monolith
class RenderEdgeTtsTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_cache_hit_skips_network(self):
        fake_layer = mock.Mock()
        cached = (np.ones(5, dtype=np.float32), 24000)
        fake_layer.tts_cache_get.return_value = cached
        with mock.patch.object(self.bc, "_tts_layer", fake_layer), \
                mock.patch.object(self.bc, "_ensure_tts_loop") as ens:
            audio, sr = self.bc._render_edge_tts("hi", "+0%", "+0Hz")
        self.assertEqual(sr, 24000)
        ens.assert_not_called()  # never touched the asyncio loop

    def test_renders_and_caches_on_miss(self):
        fake_layer = mock.Mock()
        fake_layer.tts_cache_get.return_value = None
        # Build a tiny WAV in memory that sf.read can decode.
        raw_wav = _make_wav_bytes()
        fake_future = mock.Mock()
        fake_future.result.return_value = raw_wav
        with mock.patch.object(self.bc, "_tts_layer", fake_layer), \
                mock.patch.object(self.bc, "_ensure_tts_loop"), \
                mock.patch.object(self.bc, "_tts_loop", mock.Mock()), \
                mock.patch.object(self.bc.asyncio, "run_coroutine_threadsafe",
                                  return_value=fake_future):
            audio, sr = self.bc._render_edge_tts("hi there", "+0%", "+0Hz")
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(sr, 24000)
        fake_layer.tts_cache_put.assert_called_once()

    def test_import_error_is_not_retried(self):
        fake_layer = mock.Mock()
        fake_layer.tts_cache_get.return_value = None
        fake_future = mock.Mock()
        fake_future.result.side_effect = ImportError("edge_tts missing")
        with mock.patch.object(self.bc, "_tts_layer", fake_layer), \
                mock.patch.object(self.bc, "_ensure_tts_loop"), \
                mock.patch.object(self.bc, "_tts_loop", mock.Mock()), \
                mock.patch.object(self.bc.asyncio, "run_coroutine_threadsafe",
                                  return_value=fake_future), \
                mock.patch.object(self.bc.time, "sleep") as slp:
            with self.assertRaises(ImportError):
                self.bc._render_edge_tts("hi", "+0%", "+0Hz")
        slp.assert_not_called()  # ImportError → no backoff retry


def _make_wav_bytes():
    """Create a minimal 24 kHz mono WAV that soundfile can read back."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(240, dtype=np.float32), 24000, format="WAV")
    buf.seek(0)
    return buf.read()


# ===========================================================================
# synthesise — backend selection + gain application
# ===========================================================================
@requires_monolith
class SynthesiseTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_edge_path_applies_gain(self):
        base = np.full(8, 0.5, dtype=np.float32)
        with mock.patch.object(self.bc, "_last_voice_route", [None]), \
                mock.patch.object(self.bc, "_last_user_tone", [None]), \
                mock.patch.object(self.bc, "_last_mood", [None]), \
                mock.patch.object(self.bc, "_resolve_tts_preset",
                                  return_value=("amused",
                                                {"rate": "+0%", "pitch": "+0Hz",
                                                 "gain": 2.0})), \
                mock.patch.object(self.bc, "_render_edge_tts",
                                  return_value=(base, 24000)):
            audio, sr = self.bc.synthesise("hello")
        self.assertEqual(sr, 24000)
        # gain 2.0 then clipped to 1.0
        self.assertTrue(np.allclose(audio, 1.0))

    def test_xtts_backend_failure_falls_back_to_edge(self):
        edge_audio = np.ones(4, dtype=np.float32)
        with mock.patch.object(self.bc, "_last_voice_route", [None]), \
                mock.patch.object(self.bc, "_last_user_tone", [None]), \
                mock.patch.object(self.bc, "_last_mood", [None]), \
                mock.patch.object(self.bc, "_resolve_tts_preset",
                                  return_value=("neutral",
                                                {"rate": "+0%", "pitch": "+0Hz",
                                                 "gain": 1.0})), \
                mock.patch.dict(self.bc.__dict__, {"TTS_BACKEND": "xtts"}), \
                mock.patch.object(self.bc, "_render_xtts_or_raise",
                                  side_effect=RuntimeError("xtts dead")), \
                mock.patch.object(self.bc, "_render_edge_tts",
                                  return_value=(edge_audio, 24000)):
            audio, sr = self.bc.synthesise("hi")
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.shape[0], 4)

    def test_total_failure_returns_silence(self):
        # Both edge and pyttsx3 raise → synthesise returns _silent_clip().
        with mock.patch.object(self.bc, "_last_voice_route", [None]), \
                mock.patch.object(self.bc, "_last_user_tone", [None]), \
                mock.patch.object(self.bc, "_last_mood", [None]), \
                mock.patch.object(self.bc, "_resolve_tts_preset",
                                  return_value=("neutral",
                                                {"rate": "+0%", "pitch": "+0Hz",
                                                 "gain": 1.0})), \
                mock.patch.dict(self.bc.__dict__, {"TTS_BACKEND": "edge"}), \
                mock.patch.object(self.bc, "_render_edge_tts",
                                  side_effect=RuntimeError("edge dead")), \
                mock.patch.object(self.bc, "_pyttsx3_tts",
                                  side_effect=RuntimeError("pyttsx3 dead")):
            audio, sr = self.bc.synthesise("hi")
        self.assertFalse(np.any(audio))  # silence keeps JARVIS online


# ===========================================================================
# is_using_headset / _start_barge_in_listener
# ===========================================================================
@requires_monolith
class IsUsingHeadsetTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_true_when_name_matches_hint(self):
        fake_sd = mock.Mock()
        fake_sd.query_devices.return_value = {"name": "Gaming Headset Pro"}
        with mock.patch.object(self.bc, "get_output_device", return_value=3), \
                mock.patch.object(self.bc, "sd", fake_sd), \
                mock.patch.object(self.bc, "HEADSET_NAME_HINTS",
                                  ["headset", "headphone"]):
            self.assertTrue(self.bc.is_using_headset())

    def test_false_when_no_device(self):
        with mock.patch.object(self.bc, "get_output_device", return_value=None):
            self.assertFalse(self.bc.is_using_headset())

    def test_false_on_exception(self):
        fake_sd = mock.Mock()
        fake_sd.query_devices.side_effect = RuntimeError("no such dev")
        with mock.patch.object(self.bc, "get_output_device", return_value=1), \
                mock.patch.object(self.bc, "sd", fake_sd):
            self.assertFalse(self.bc.is_using_headset())

    def test_false_for_speaker_name(self):
        fake_sd = mock.Mock()
        fake_sd.query_devices.return_value = {"name": "Desktop Speakers"}
        with mock.patch.object(self.bc, "get_output_device", return_value=2), \
                mock.patch.object(self.bc, "sd", fake_sd), \
                mock.patch.object(self.bc, "HEADSET_NAME_HINTS", ["headset"]):
            self.assertFalse(self.bc.is_using_headset())


@requires_monolith
class StartBargeInListenerTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_returns_none_when_mic_disabled(self):
        with mock.patch.object(self.bc, "_mic_input_disabled", return_value=True):
            self.assertIsNone(self.bc._start_barge_in_listener())

    def test_opens_stream_and_resets_flag(self):
        fake_stream = mock.Mock()
        fake_sd = mock.Mock()
        fake_sd.InputStream.return_value = fake_stream
        with mock.patch.object(self.bc, "_mic_input_disabled", return_value=False), \
                mock.patch.object(self.bc, "get_input_device", return_value=0), \
                mock.patch.object(self.bc, "sd", fake_sd):
            self.bc._barge_in_interrupted = True  # should be reset to False
            out = self.bc._start_barge_in_listener()
        self.assertIs(out, fake_stream)
        fake_stream.start.assert_called_once()
        self.assertFalse(self.bc._barge_in_interrupted)

    def test_returns_none_on_open_failure(self):
        fake_sd = mock.Mock()
        fake_sd.InputStream.side_effect = RuntimeError("device busy")
        with mock.patch.object(self.bc, "_mic_input_disabled", return_value=False), \
                mock.patch.object(self.bc, "get_input_device", return_value=0), \
                mock.patch.object(self.bc, "sd", fake_sd):
            self.assertIsNone(self.bc._start_barge_in_listener())

    def test_callback_flags_interrupt_after_sustained_loud(self):
        # Capture the callback the monolith hands to sd.InputStream, then feed
        # it loud frames and assert the sustain logic flips the global flag.
        holder = {}
        fake_sd = mock.Mock()

        def _make_stream(**kwargs):
            holder["cb"] = kwargs["callback"]
            return mock.Mock()
        fake_sd.InputStream.side_effect = _make_stream
        with mock.patch.object(self.bc, "_mic_input_disabled", return_value=False), \
                mock.patch.object(self.bc, "get_input_device", return_value=0), \
                mock.patch.object(self.bc, "sd", fake_sd), \
                mock.patch.object(self.bc, "BARGE_IN_THRESHOLD", 0.01), \
                mock.patch.object(self.bc, "BARGE_IN_SUSTAIN_CHUNKS", 2):
            self.bc._start_barge_in_listener()
            cb = holder["cb"]
            loud = np.full(64, 0.5, dtype=np.float32).reshape(-1, 1)
            self.assertFalse(self.bc._barge_in_interrupted)
            cb(loud, 64, None, None)   # 1st loud chunk
            self.assertFalse(self.bc._barge_in_interrupted)
            cb(loud, 64, None, None)   # 2nd → sustain reached
            self.assertTrue(self.bc._barge_in_interrupted)
        # leave the global clean for other tests
        self.bc._barge_in_interrupted = False

    def test_callback_resets_on_quiet_and_returns_when_interrupted(self):
        # 6471: a sub-threshold chunk resets the sustain counter; 6463: once the
        # interrupt flag is already set, the callback returns immediately
        # without re-evaluating RMS.
        holder = {}
        fake_sd = mock.Mock()

        def _make_stream(**kwargs):
            holder["cb"] = kwargs["callback"]
            return mock.Mock()
        fake_sd.InputStream.side_effect = _make_stream
        with mock.patch.object(self.bc, "_mic_input_disabled", return_value=False), \
                mock.patch.object(self.bc, "get_input_device", return_value=0), \
                mock.patch.object(self.bc, "sd", fake_sd), \
                mock.patch.object(self.bc, "BARGE_IN_THRESHOLD", 0.01), \
                mock.patch.object(self.bc, "BARGE_IN_SUSTAIN_CHUNKS", 2):
            self.bc._start_barge_in_listener()
            cb = holder["cb"]
            loud = np.full(64, 0.5, dtype=np.float32).reshape(-1, 1)
            quiet = np.zeros((64, 1), dtype=np.float32)
            cb(loud, 64, None, None)    # sustain == 1
            cb(quiet, 64, None, None)   # 6471: sub-threshold → sustain reset to 0
            self.assertFalse(self.bc._barge_in_interrupted)
            # one loud chunk now only gets sustain back to 1, not the 2 needed
            cb(loud, 64, None, None)
            self.assertFalse(self.bc._barge_in_interrupted)
            # Force the interrupted state, then a further loud chunk must early-
            # return at 6463 (no exception, flag stays True).
            self.bc._barge_in_interrupted = True
            cb(loud, 64, None, None)
            self.assertTrue(self.bc._barge_in_interrupted)
        self.bc._barge_in_interrupted = False


# ===========================================================================
# _AudioDucker
# ===========================================================================
@requires_monolith
class AudioDuckerTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_duck_noop_when_disabled(self):
        d = self.bc._AudioDucker()
        with mock.patch.object(self.bc, "AUDIO_DUCKING_ENABLED", False):
            d.duck()  # must not raise / must not enumerate
        self.assertEqual(d._saved, [])

    def test_check_available_false_off_windows(self):
        d = self.bc._AudioDucker()
        cls = type(d)
        saved = cls._AVAILABLE
        try:
            cls._AVAILABLE = None
            with mock.patch.object(self.bc.sys, "platform", "linux"):
                self.assertFalse(d._check_available())
        finally:
            cls._AVAILABLE = saved

    def test_check_available_caches(self):
        d = self.bc._AudioDucker()
        cls = type(d)
        saved = cls._AVAILABLE
        try:
            cls._AVAILABLE = True
            self.assertTrue(d._check_available())  # short-circuits on cache
        finally:
            cls._AVAILABLE = saved

    def test_duck_skips_when_no_targets_matched(self):
        d = self.bc._AudioDucker()
        with mock.patch.object(self.bc, "AUDIO_DUCKING_ENABLED", True), \
                mock.patch.object(d, "_check_available", return_value=True), \
                mock.patch.object(d, "_enumerate_targets", return_value=[]):
            d.duck()
        self.assertEqual(d._saved, [])

    def test_duck_enqueues_fade_when_targets_present(self):
        d = self.bc._AudioDucker()
        iface = mock.Mock()
        with mock.patch.object(self.bc, "AUDIO_DUCKING_ENABLED", True), \
                mock.patch.object(self.bc, "AUDIO_DUCKING_LEVEL", 0.2), \
                mock.patch.object(d, "_check_available", return_value=True), \
                mock.patch.object(d, "_enumerate_targets",
                                  return_value=[(iface, 0.9)]), \
                mock.patch.object(d, "_ensure_worker"):
            d.duck()
        self.assertEqual(d._saved, [(iface, 0.9)])
        job = d._work_queue.get_nowait()
        plans, level, cancellable, done = job
        self.assertEqual(plans, [(iface, 0.9)])
        self.assertEqual(level, 0.2)
        self.assertTrue(cancellable)

    def test_duck_idempotent_while_saved(self):
        d = self.bc._AudioDucker()
        d._saved = [(mock.Mock(), 0.8)]  # pretend already ducked
        with mock.patch.object(self.bc, "AUDIO_DUCKING_ENABLED", True), \
                mock.patch.object(d, "_check_available", return_value=True), \
                mock.patch.object(d, "_enumerate_targets") as enum:
            d.duck()
        enum.assert_not_called()

    def test_restore_noop_when_nothing_saved(self):
        d = self.bc._AudioDucker()
        with mock.patch.object(d, "_ensure_worker") as ew:
            d.restore()
        ew.assert_not_called()

    def test_restore_enqueues_fade_up(self):
        d = self.bc._AudioDucker()
        iface = mock.Mock()
        d._saved = [(iface, 0.85)]
        with mock.patch.object(self.bc, "AUDIO_DUCKING_FADE_MS", 1), \
                mock.patch.object(d, "_ensure_worker"):
            d.restore()
        self.assertEqual(d._saved, [])  # cleared
        # the fade-up job carries target_level=None (= restore to original)
        job = d._work_queue.get_nowait()
        plans, level, cancellable, done = job
        self.assertEqual(plans, [(iface, 0.85)])
        self.assertIsNone(level)
        self.assertFalse(cancellable)

    def test_fade_run_steps_volume_to_target(self):
        d = self.bc._AudioDucker()
        iface = mock.Mock()
        iface.GetMasterVolume.return_value = 1.0
        with mock.patch.object(self.bc, "AUDIO_DUCKING_FADE_MS", 0), \
                mock.patch.object(self.bc.time, "sleep"):
            d._fade_run([(iface, 1.0)], 0.2, cancellable=False)
        # final SetMasterVolume call should land at the target level (0.2)
        last_level = iface.SetMasterVolume.call_args_list[-1][0][0]
        self.assertAlmostEqual(last_level, 0.2, places=6)

    def test_fade_run_empty_plans_is_noop(self):
        d = self.bc._AudioDucker()
        d._fade_run([], 0.2, cancellable=False)  # no raise


# ===========================================================================
# play_with_lipsync — muted path (no real device I/O)
# ===========================================================================
@requires_monolith
class PlayWithLipsyncTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_flag = list(self.bc._tts_playback_active)

    def tearDown(self):
        self.bc._tts_playback_active[:] = self._saved_flag

    def test_muted_path_skips_device_and_resets_flag(self):
        fake_sd = mock.Mock()
        fake_layer = mock.Mock()
        fake_layer.is_muted.return_value = True
        ducker = mock.Mock()
        audio = np.zeros(240, dtype=np.float32)
        with mock.patch.object(self.bc, "sd", fake_sd), \
                mock.patch.object(self.bc, "_tts_layer", fake_layer), \
                mock.patch.object(self.bc, "_audio_ducker", ducker), \
                mock.patch.object(self.bc, "BARGE_IN_ENABLED", False), \
                mock.patch.object(self.bc, "ROBOT_ENABLED", False), \
                mock.patch.object(self.bc, "get_output_device", return_value=1), \
                mock.patch.object(self.bc, "_write_hud_state"), \
                mock.patch.object(self.bc, "_feed_playback_reference"), \
                mock.patch.object(self.bc.threading, "Thread", _ImmediateThread), \
                mock.patch.object(self.bc.time, "sleep"):
            self.bc.play_with_lipsync(audio, 24000)
        # MUTE path must NOT play audio to the device
        fake_sd.play.assert_not_called()
        # ducking still engaged + restored around the (silent) playback
        ducker.duck.assert_called_once()
        ducker.restore.assert_called_once()
        # guard flag reset in finally
        self.assertFalse(self.bc._tts_playback_active[0])

    def test_unmuted_no_robot_plays_audio(self):
        fake_sd = mock.Mock()
        fake_layer = mock.Mock()
        fake_layer.is_muted.return_value = False
        ducker = mock.Mock()
        audio = np.zeros(48, dtype=np.float32)
        with mock.patch.object(self.bc, "sd", fake_sd), \
                mock.patch.object(self.bc, "_tts_layer", fake_layer), \
                mock.patch.object(self.bc, "_audio_ducker", ducker), \
                mock.patch.object(self.bc, "BARGE_IN_ENABLED", False), \
                mock.patch.object(self.bc, "ROBOT_ENABLED", False), \
                mock.patch.object(self.bc, "get_output_device", return_value=1), \
                mock.patch.object(self.bc, "_write_hud_state"), \
                mock.patch.object(self.bc, "_feed_playback_reference"), \
                mock.patch.object(self.bc.threading, "Thread", _ImmediateThread), \
                mock.patch.object(self.bc.time, "sleep"):
            self.bc.play_with_lipsync(audio, 24000)
        fake_sd.play.assert_called_once()
        self.assertFalse(self.bc._tts_playback_active[0])


# ===========================================================================
# _ensure_tts_loop — loop lifecycle (no real event loop thread left running)
# ===========================================================================
@requires_monolith
class EnsureTtsLoopTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = (self.bc._tts_loop, self.bc._tts_loop_thread)

    def tearDown(self):
        self.bc._tts_loop, self.bc._tts_loop_thread = self._saved

    def test_reuses_running_loop(self):
        # Set the module globals directly (the SUT rebinds _tts_loop via
        # `global`, which interacts badly with mock.patch.object on this
        # interpreter — assert on observable side effects instead).
        live_loop = mock.Mock()
        live_loop.is_running.return_value = True
        live_thread = mock.Mock()
        live_thread.is_alive.return_value = True
        self.bc._tts_loop = live_loop
        self.bc._tts_loop_thread = live_thread
        with mock.patch.object(self.bc.asyncio, "new_event_loop") as nel:
            self.bc._ensure_tts_loop()
        nel.assert_not_called()  # healthy loop reused, no new loop created
        self.assertIs(self.bc._tts_loop, live_loop)  # unchanged

    def test_recreates_when_thread_dead(self):
        dead_loop = mock.Mock()
        dead_loop.is_running.return_value = True
        dead_thread = mock.Mock()
        dead_thread.is_alive.return_value = False  # thread died
        new_loop = mock.Mock()
        self.bc._tts_loop = dead_loop
        self.bc._tts_loop_thread = dead_thread
        with mock.patch.object(self.bc.asyncio, "new_event_loop",
                               return_value=new_loop) as nel, \
                mock.patch.object(self.bc.threading, "Thread",
                                  _ImmediateThread):
            self.bc._ensure_tts_loop()
            # assert inside the patch context, before tearDown restores globals
            self.assertIs(self.bc._tts_loop, new_loop)
        # a fresh loop was created to replace the half-dead one
        nel.assert_called_once()

    def test_stop_of_old_loop_exception_swallowed(self):
        # 6216-6219: the half-dead loop's call_soon_threadsafe(stop) raises ->
        # the except swallows it and a fresh loop is still created.
        dead_loop = mock.Mock()
        dead_loop.is_running.return_value = False   # not healthy -> replace
        dead_loop.call_soon_threadsafe.side_effect = RuntimeError("loop wedged")
        new_loop = mock.Mock()
        self.bc._tts_loop = dead_loop
        self.bc._tts_loop_thread = None
        with mock.patch.object(self.bc.asyncio, "new_event_loop",
                               return_value=new_loop) as nel, \
                mock.patch.object(self.bc.threading, "Thread",
                                  _ImmediateThread):
            self.bc._ensure_tts_loop()   # must not raise
            self.assertIs(self.bc._tts_loop, new_loop)
        dead_loop.call_soon_threadsafe.assert_called_once()
        nel.assert_called_once()


# ===========================================================================
# SECOND BATCH — deeper branch coverage on safe (non-hardware) paths
# ===========================================================================

@requires_monolith
class RegisterCudaDllDirsScanTests(MonolithGlobalsTestCase):
    """Exercise the dir-scan body of _register_cuda_dll_dirs with a fake
    `nvidia` namespace package so the PATH-prepend + add_dll_directory
    bookkeeping runs without touching a real CUDA install."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        fn = self.bc._register_cuda_dll_dirs
        self._saved = {k: getattr(fn, k) for k in
                       ("_done", "_registered", "_missing", "_reason")
                       if hasattr(fn, k)}
        self._saved_path = os.environ.get("PATH", "")

    def tearDown(self):
        fn = self.bc._register_cuda_dll_dirs
        for k in ("_done", "_registered", "_missing", "_reason"):
            if k in self._saved:
                setattr(fn, k, self._saved[k])
            elif hasattr(fn, k):
                delattr(fn, k)
        os.environ["PATH"] = self._saved_path

    def test_scan_registers_existing_dirs(self):
        fn = self.bc._register_cuda_dll_dirs
        fn._done = False
        fake_nvidia = mock.Mock()
        fake_nvidia.__path__ = [r"C:\fake\nvidia"]
        add_dll = mock.Mock()
        with mock.patch.dict(sys.modules, {"nvidia": fake_nvidia}), \
                mock.patch.object(self.bc.os.path, "isdir", return_value=True), \
                mock.patch.object(self.bc.os, "add_dll_directory", add_dll,
                                  create=True):
            fn()
        self.assertTrue(fn._done)
        # both cublas + cudnn bin dirs registered
        self.assertEqual(len(fn._registered), 2)
        self.assertEqual(add_dll.call_count, 2)

    def test_scan_records_missing_dirs(self):
        fn = self.bc._register_cuda_dll_dirs
        fn._done = False
        fake_nvidia = mock.Mock()
        fake_nvidia.__path__ = [r"C:\fake\nvidia"]
        with mock.patch.dict(sys.modules, {"nvidia": fake_nvidia}), \
                mock.patch.object(self.bc.os.path, "isdir", return_value=False):
            fn()
        self.assertTrue(fn._done)
        self.assertEqual(len(fn._missing), 2)
        self.assertEqual(fn._registered, [])


@requires_monolith
class EnsureWhisperFallbackTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = (self.bc._stt, self.bc._stt_device,
                       self.bc._stt_model_name, self.bc._stt_engine)
        self.bc._stt = None

    def tearDown(self):
        (self.bc._stt, self.bc._stt_device,
         self.bc._stt_model_name, self.bc._stt_engine) = self._saved

    def test_cuda_dll_failure_retries_on_cpu(self):
        # First WhisperModel(... device="cuda") raises a CUDA-DLL error; the
        # loader must retry with device="cpu"/int8 and succeed.
        good = object()
        calls = []

        def _wm(model, device=None, compute_type=None):
            calls.append((device, compute_type))
            if device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found")
            return good
        fake_fw = mock.Mock()
        fake_fw.WhisperModel = mock.Mock(side_effect=_wm)
        with mock.patch.object(self.bc, "_register_cuda_dll_dirs"), \
                mock.patch.object(self.bc, "_resolve_whisper_device",
                                  return_value="cuda"), \
                mock.patch.object(self.bc, "_force_whisper_cpu_int8", False), \
                mock.patch.object(self.bc, "WHISPER_MODEL_CUDA", "large-v3"), \
                mock.patch.dict(sys.modules, {"faster_whisper": fake_fw}):
            self.bc._ensure_whisper()
        self.assertIs(self.bc._stt, good)
        self.assertEqual(self.bc._stt_device, "cpu")
        # retried on cpu after the cuda attempt
        self.assertIn(("cuda", "float16"), calls)
        self.assertIn(("cpu", "int8"), calls)

    def test_openai_whisper_used_when_faster_absent(self):
        # faster_whisper import fails → fall through to openai-whisper.
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "faster_whisper":
                raise ImportError("not installed")
            return real_import(name, *a, **k)

        good = object()
        fake_whisper = mock.Mock()
        fake_whisper.load_model.return_value = good
        with mock.patch.object(self.bc, "_register_cuda_dll_dirs"), \
                mock.patch.object(self.bc, "_resolve_whisper_device",
                                  return_value="cpu"), \
                mock.patch.object(self.bc, "WHISPER_MODEL_CPU", "base"), \
                mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch.dict(sys.modules, {"whisper": fake_whisper}):
            self.bc._ensure_whisper()
        self.assertIs(self.bc._stt, good)
        self.assertEqual(self.bc._stt_engine, "openai_whisper")
        fake_whisper.load_model.assert_called_with("base", device="cpu")


@requires_monolith
class CallLocalLlmWebSearchGuardTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_unread_web_search_prepends_no_fabricate_guard(self):
        captured = {}
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None):
            captured["sys"] = json["messages"][0]["content"]
            return _FakeResp(ok=True, json_data={"message": {"content": "ok sir"}})
        fake_req.post.side_effect = _post
        # A web_search was fired but never followed by a see_screen read.
        msgs = [{"role": "assistant", "content": "[ACTION: web_search, census]"}]
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_get_local_llm_model",
                                  return_value="m:tag"), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=True), \
                mock.patch.object(self.bc, "requests", fake_req):
            self.bc._call_local_llm("BASE", msgs)
        self.assertIn("Do NOT fabricate", captured["sys"])

    def test_web_search_scan_exception_is_swallowed(self):
        # 5723-5724: the messages object passes isinstance(list) but raises when
        # sliced for the recent-window scan -> the except swallows it and the
        # call proceeds normally (no guard prepended, no crash).
        class _BadSliceList(list):
            def __getitem__(self, key):
                if isinstance(key, slice):
                    raise RuntimeError("slice exploded")
                return super().__getitem__(key)

        captured = {}
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None):
            captured["sys"] = json["messages"][0]["content"]
            return _FakeResp(ok=True, json_data={"message": {"content": "ok sir"}})
        fake_req.post.side_effect = _post
        msgs = _BadSliceList([{"role": "user", "content": "hi"}])
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_get_local_llm_model",
                                  return_value="m:tag"), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=True), \
                mock.patch.object(self.bc, "requests", fake_req):
            out = self.bc._call_local_llm("BASE", msgs)
        self.assertEqual(out, "ok sir")
        self.assertNotIn("Do NOT fabricate", captured["sys"])


@requires_monolith
class SynthesiseExtraPathsTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _common_patches(self, preset_name, preset):
        return [
            mock.patch.object(self.bc, "_last_voice_route", [None]),
            mock.patch.object(self.bc, "_last_user_tone", [None]),
            mock.patch.object(self.bc, "_last_mood", [None]),
            mock.patch.object(self.bc, "_resolve_tts_preset",
                              return_value=(preset_name, preset)),
        ]

    def test_wry_split_splices_silence_between_clauses(self):
        fake_layer = mock.Mock()
        fake_layer.split_for_wry_pause.return_value = ("Setup,", "punchline.")
        fake_layer.WRY_PAUSE_MS = 200
        head = np.ones(100, dtype=np.float32)
        tail = np.ones(50, dtype=np.float32)
        preset = {"rate": "+0%", "pitch": "+0Hz", "gain": 1.0}
        patches = self._common_patches("wry", preset)
        patches.append(mock.patch.object(self.bc, "_tts_layer", fake_layer))
        patches.append(mock.patch.dict(self.bc.__dict__, {"TTS_BACKEND": "edge"}))
        patches.append(mock.patch.object(
            self.bc, "_render_edge_tts",
            side_effect=[(head, 24000), (tail, 24000)]))
        for p in patches:
            p.start()
        try:
            audio, sr = self.bc.synthesise("Setup, punchline.")
        finally:
            for p in patches:
                p.stop()
        # head + (200ms silence @24k = 4800) + tail
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.shape[0], 100 + int(24000 * 200 / 1000) + 50)

    def test_pyttsx3_backend_selected(self):
        base = np.full(8, 0.25, dtype=np.float32)
        preset = {"rate": "+0%", "pitch": "+0Hz", "gain": 1.0}
        patches = self._common_patches("neutral", preset)
        patches.append(mock.patch.dict(self.bc.__dict__,
                                       {"TTS_BACKEND": "pyttsx3"}))
        patches.append(mock.patch.object(self.bc, "_pyttsx3_tts",
                                         return_value=(base, 22050)))
        for p in patches:
            p.start()
        try:
            audio, sr = self.bc.synthesise("hello")
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(sr, 22050)


@requires_monolith
class RenderEdgeTtsRetryTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_transient_error_retries_then_succeeds(self):
        fake_layer = mock.Mock()
        fake_layer.tts_cache_get.return_value = None
        raw_wav = _make_wav_bytes()
        # 1st attempt: transient 503 → backoff+retry; 2nd attempt: success.
        f_fail = mock.Mock()
        f_fail.result.side_effect = RuntimeError("503 Service Unavailable")
        f_ok = mock.Mock()
        f_ok.result.return_value = raw_wav
        with mock.patch.object(self.bc, "_tts_layer", fake_layer), \
                mock.patch.object(self.bc, "_ensure_tts_loop"), \
                mock.patch.object(self.bc, "_tts_loop", mock.Mock()), \
                mock.patch.object(self.bc.asyncio, "run_coroutine_threadsafe",
                                  side_effect=[f_fail, f_ok]), \
                mock.patch.object(self.bc.time, "sleep") as slp:
            audio, sr = self.bc._render_edge_tts("retry me", "+0%", "+0Hz")
        self.assertEqual(sr, 24000)
        slp.assert_called_once()  # exactly one backoff between the two tries


@requires_monolith
class AudioDuckerEnumerateTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_enumerate_matches_target_above_duck_level(self):
        d = self.bc._AudioDucker()
        # Build a fake pycaw session whose process name matches a target and
        # whose current volume is above the duck level.
        vol = mock.Mock()
        vol.GetMasterVolume.return_value = 0.9
        sess = mock.Mock()
        sess.Process = mock.Mock()
        sess.Process.pid = self.bc._AudioDucker._SELF_PID + 12345
        sess.Process.name.return_value = "spotify.exe"
        sess.SimpleAudioVolume = vol
        fake_pycaw_mod = mock.Mock()
        fake_pycaw_mod.AudioUtilities.GetAllSessions.return_value = [sess]
        with mock.patch.dict(sys.modules,
                             {"pycaw": mock.Mock(),
                              "pycaw.pycaw": mock.Mock(
                                  AudioUtilities=fake_pycaw_mod.AudioUtilities),
                              "comtypes": mock.Mock()}), \
                mock.patch.object(self.bc, "AUDIO_DUCKING_TARGETS", ["spotify"]), \
                mock.patch.object(self.bc, "AUDIO_DUCKING_LEVEL", 0.2):
            matched = d._enumerate_targets()
        self.assertEqual(len(matched), 1)
        self.assertIs(matched[0][0], vol)
        self.assertAlmostEqual(matched[0][1], 0.9, places=6)

    def test_enumerate_skips_self_and_quiet_sessions(self):
        d = self.bc._AudioDucker()
        # session 1: our own PID → skip; session 2: already below duck level → skip
        own = mock.Mock()
        own.Process = mock.Mock(pid=self.bc._AudioDucker._SELF_PID)
        own.Process.name.return_value = "spotify.exe"
        quiet_vol = mock.Mock()
        quiet_vol.GetMasterVolume.return_value = 0.1
        quiet = mock.Mock()
        quiet.Process = mock.Mock(pid=self.bc._AudioDucker._SELF_PID + 7)
        quiet.Process.name.return_value = "spotify.exe"
        quiet.SimpleAudioVolume = quiet_vol
        fake_au = mock.Mock()
        fake_au.GetAllSessions.return_value = [own, quiet]
        with mock.patch.dict(sys.modules,
                             {"pycaw": mock.Mock(),
                              "pycaw.pycaw": mock.Mock(AudioUtilities=fake_au),
                              "comtypes": mock.Mock()}), \
                mock.patch.object(self.bc, "AUDIO_DUCKING_TARGETS", ["spotify"]), \
                mock.patch.object(self.bc, "AUDIO_DUCKING_LEVEL", 0.2):
            matched = d._enumerate_targets()
        self.assertEqual(matched, [])

    def test_fade_run_cancel_aborts_early(self):
        d = self.bc._AudioDucker()
        iface = mock.Mock()
        iface.GetMasterVolume.return_value = 1.0
        d._fade_cancel.set()  # pre-set cancel → cancellable fade returns at once
        with mock.patch.object(self.bc, "AUDIO_DUCKING_FADE_MS", 100), \
                mock.patch.object(self.bc.time, "sleep") as slp:
            d._fade_run([(iface, 1.0)], 0.2, cancellable=True)
        # aborted before any sleep / volume step
        slp.assert_not_called()
        iface.SetMasterVolume.assert_not_called()
        d._fade_cancel.clear()


@requires_monolith
class CallLlmMoreBranchesTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_hist = list(self.bc.conversation_history)
        self.bc.conversation_history.clear()
        self._patches = [
            mock.patch.object(self.bc, "detect_tone", return_value=None),
            mock.patch.object(self.bc, "route_voice_emotion",
                              return_value={"mood": "casual", "addendum": ""}),
            mock.patch.object(self.bc, "_emotion_tracker", None),
            mock.patch.object(self.bc, "_voice_mood_response", None),
            mock.patch.object(self.bc, "_trim_conversation_history"),
            mock.patch.object(self.bc, "_system_prompt", "SYS"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.bc.conversation_history[:] = self._saved_hist

    def _anthropic_stub(self):
        a = mock.Mock()
        a.BadRequestError = type("BadReq", (Exception,), {})

        class _RL(Exception):
            pass
        a.RateLimitError = _RL

        class _AS(Exception):
            def __init__(self, status_code=500, message="boom"):
                super().__init__(message)
                self.status_code = status_code
                self.message = message
        a.APIStatusError = _AS
        return a

    def test_rate_limit_error_falls_back_local(self):
        a = self._anthropic_stub()
        client = mock.Mock()
        client.complete.side_effect = a.RateLimitError("slow down")
        phrases = mock.Mock()
        phrases.detect_phrases_in_reply.return_value = {}
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
                mock.patch.object(self.bc, "_llm_client", client), \
                mock.patch.dict(sys.modules, {"anthropic": a}), \
                mock.patch.object(self.bc, "_local_fallback_or",
                                  return_value="LOCAL") as lf, \
                mock.patch.object(self.bc, "_mcu_phrases", phrases):
            reply = self.bc._call_llm("hi")
        self.assertEqual(reply, "LOCAL")
        self.assertIn("rate-limited", lf.call_args[0][1].lower())

    def test_api_status_error_falls_back_local(self):
        a = self._anthropic_stub()
        client = mock.Mock()
        client.complete.side_effect = a.APIStatusError(status_code=529,
                                                       message="overloaded")
        phrases = mock.Mock()
        phrases.detect_phrases_in_reply.return_value = {}
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
                mock.patch.object(self.bc, "_llm_client", client), \
                mock.patch.dict(sys.modules, {"anthropic": a}), \
                mock.patch.object(self.bc, "_local_fallback_or",
                                  return_value="LOCAL2") as lf, \
                mock.patch.object(self.bc, "_mcu_phrases", phrases):
            reply = self.bc._call_llm("hi")
        self.assertEqual(reply, "LOCAL2")
        self.assertIn("529", lf.call_args[0][1])

    def test_phrase_rotation_persists_hits_to_memory(self):
        a = self._anthropic_stub()
        client = mock.Mock()
        client.complete.return_value = "As you wish, sir."
        phrases = mock.Mock()
        phrases.detect_phrases_in_reply.return_value = {"greeting": "As you wish"}
        mem = {}
        save = mock.Mock()
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
                mock.patch.object(self.bc, "_llm_client", client), \
                mock.patch.dict(sys.modules, {"anthropic": a}), \
                mock.patch.object(self.bc, "_mcu_phrases", phrases), \
                mock.patch.object(self.bc, "load_memory", return_value=mem), \
                mock.patch.object(self.bc, "save_memory", save):
            reply = self.bc._call_llm("hello")
        self.assertEqual(reply, "As you wish, sir.")
        save.assert_called_once()
        # the hit was recorded in the rotation bucket
        self.assertEqual(mem["last_used_phrase_by_intent"]["greeting"],
                         "As you wish")

    def test_claude_usage_limit_message(self):
        a = self._anthropic_stub()
        client = mock.Mock()
        client.complete.side_effect = a.BadRequestError(
            "You have reached your monthly usage limit")
        phrases = mock.Mock()
        phrases.detect_phrases_in_reply.return_value = {}
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
                mock.patch.object(self.bc, "_llm_client", client), \
                mock.patch.dict(sys.modules, {"anthropic": a}), \
                mock.patch.object(self.bc, "_local_fallback_or",
                                  return_value="LOCAL3") as lf, \
                mock.patch.object(self.bc, "_mcu_phrases", phrases):
            reply = self.bc._call_llm("hi")
        self.assertEqual(reply, "LOCAL3")
        self.assertIn("monthly", lf.call_args[0][1].lower())


# ===========================================================================
# THIRD BATCH — coverage-extension pass over the audio/STT/LLM/TTS spine.
# Targets the Missing line ranges that the first two batches left in
# 4443-6881: the per-turn classifier side-trips in _call_llm, the ollama
# backend + phrase-rotation tails, _call_local_llm / _call_local_vision
# failure branches, the openai-whisper + CUDA-recovery paths in transcribe,
# the _register_cuda_dll_dirs error arms, synthesise's xtts/pyttsx3 gain +
# wry-split-failure paths, _pyttsx3_tts's temp-file render, the _AudioDucker
# worker/restore/duck machinery, get_mic_buffer's tap paths, and the
# barge-in + robot arms of play_with_lipsync. Every hardware/network/thread
# boundary is mocked; the noted genuinely-unrunnable capture/worker LOOPS
# (record_speech VAD stream, _amp_pump/_sync/_barge_watch live bodies) are
# driven only where _ImmediateThread lets their closure run once
# deterministically, never as real spinning threads.
# ===========================================================================
@requires_monolith
class CallLlmClassifierSideTripsTests(MonolithGlobalsTestCase):
    """Drive the per-turn classifier branches in _call_llm that the happy-path
    suite skips: a non-empty tone print, the voice-mood-response adapter (both
    success and failure), and the emotion-tracker addendum (label differs /
    matches / raises)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_hist = list(self.bc.conversation_history)
        self.bc.conversation_history.clear()
        self._patches = [
            mock.patch.object(self.bc, "_trim_conversation_history"),
            mock.patch.object(self.bc, "_system_prompt", "SYS"),
            mock.patch.object(self.bc, "AI_BACKEND", "claude"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.bc.conversation_history[:] = self._saved_hist

    def _phrases(self):
        m = mock.Mock()
        m.detect_phrases_in_reply.return_value = {}
        return m

    def _client(self):
        c = mock.Mock()
        c.complete.return_value = "Indeed, sir."
        return c

    def test_nondefault_tone_and_mood_and_emotion_addenda(self):
        # tone non-empty (5880), route mood non-casual (5890), voice-mood
        # adapter applied (5899-5906), emotion label differs from tone so its
        # addendum stacks (5917-5926).
        vm = mock.Mock()
        vm.apply_voice_mood_response.return_value = " [be brief]"
        et = mock.Mock()
        er = mock.Mock()
        er.label = "focused"
        er.reason = "imperatives"
        er.addendum = " [focus mode]"
        et.classify_emotion.return_value = er
        captured = {}

        def _complete(**kw):
            captured["system"] = kw["system"]
            return "On it, sir."
        client = mock.Mock()
        client.complete.side_effect = _complete
        with mock.patch.object(self.bc, "detect_tone", return_value="rushed"), \
                mock.patch.object(self.bc, "route_voice_emotion",
                                  return_value={"mood": "stressed",
                                                "addendum": " [stress]"}), \
                mock.patch.object(self.bc, "_voice_mood_response", vm), \
                mock.patch.object(self.bc, "_emotion_tracker", et), \
                mock.patch.object(self.bc, "_llm_client", client), \
                mock.patch.dict(sys.modules, {"anthropic": mock.Mock()}), \
                mock.patch.object(self.bc, "_mcu_phrases", self._phrases()):
            reply = self.bc._call_llm("hurry up and fix it")
        self.assertEqual(reply, "On it, sir.")
        vm.apply_voice_mood_response.assert_called_once()
        # both addenda flowed into the per-turn system prompt
        self.assertIn("[be brief]", captured["system"])
        self.assertIn("[focus mode]", captured["system"])
        # cached classifier state was stamped
        self.assertEqual(self.bc._last_user_tone[0], "rushed")
        self.assertIs(self.bc._last_emotion[0], er)

    def test_voice_mood_adapter_exception_is_swallowed(self):
        # 5907-5908: adapter raises → caught, addendum stays "" , call proceeds.
        vm = mock.Mock()
        vm.apply_voice_mood_response.side_effect = RuntimeError("vm boom")
        with mock.patch.object(self.bc, "detect_tone", return_value=None), \
                mock.patch.object(self.bc, "route_voice_emotion",
                                  return_value={"mood": "casual", "addendum": ""}), \
                mock.patch.object(self.bc, "_voice_mood_response", vm), \
                mock.patch.object(self.bc, "_emotion_tracker", None), \
                mock.patch.object(self.bc, "_llm_client", self._client()), \
                mock.patch.dict(sys.modules, {"anthropic": mock.Mock()}), \
                mock.patch.object(self.bc, "_mcu_phrases", self._phrases()):
            reply = self.bc._call_llm("hi")
        self.assertEqual(reply, "Indeed, sir.")

    def test_emotion_tracker_exception_clears_last_emotion(self):
        # 5927-5929: classifier raises → _last_emotion reset to None.
        et = mock.Mock()
        et.classify_emotion.side_effect = ValueError("bad emotion")
        self.bc._last_emotion[0] = "stale"
        with mock.patch.object(self.bc, "detect_tone", return_value=None), \
                mock.patch.object(self.bc, "route_voice_emotion",
                                  return_value={"mood": "casual", "addendum": ""}), \
                mock.patch.object(self.bc, "_voice_mood_response", None), \
                mock.patch.object(self.bc, "_emotion_tracker", et), \
                mock.patch.object(self.bc, "_llm_client", self._client()), \
                mock.patch.dict(sys.modules, {"anthropic": mock.Mock()}), \
                mock.patch.object(self.bc, "_mcu_phrases", self._phrases()):
            self.bc._call_llm("hi")
        self.assertIsNone(self.bc._last_emotion[0])

    def test_emotion_label_equals_tone_skips_addendum(self):
        # 5925 false branch: er.label == tone → emotion_addendum stays "".
        et = mock.Mock()
        er = mock.Mock()
        er.label = "rushed"   # identical to detect_tone's label
        er.reason = "r"
        er.addendum = " [DUP]"
        et.classify_emotion.return_value = er
        captured = {}

        def _complete(**kw):
            captured["system"] = kw["system"]
            return "Mm."
        client = mock.Mock()
        client.complete.side_effect = _complete
        with mock.patch.object(self.bc, "detect_tone", return_value="rushed"), \
                mock.patch.object(self.bc, "route_voice_emotion",
                                  return_value={"mood": "casual", "addendum": ""}), \
                mock.patch.object(self.bc, "_voice_mood_response", None), \
                mock.patch.object(self.bc, "_emotion_tracker", et), \
                mock.patch.object(self.bc, "_llm_client", client), \
                mock.patch.dict(sys.modules, {"anthropic": mock.Mock()}), \
                mock.patch.object(self.bc, "_mcu_phrases", self._phrases()):
            self.bc._call_llm("go go go")
        self.assertNotIn("[DUP]", captured["system"])

    def test_no_llm_client_uses_anthropic_messages_create(self):
        # 5964-5969: _llm_client is None → direct anthropic.Anthropic path.
        fake_anthropic = mock.Mock()
        msg = mock.Mock()
        msg.content = [mock.Mock(text="Direct path, sir.")]
        fake_anthropic.Anthropic.return_value.messages.create.return_value = msg
        with mock.patch.object(self.bc, "detect_tone", return_value=None), \
                mock.patch.object(self.bc, "route_voice_emotion",
                                  return_value={"mood": "casual", "addendum": ""}), \
                mock.patch.object(self.bc, "_voice_mood_response", None), \
                mock.patch.object(self.bc, "_emotion_tracker", None), \
                mock.patch.object(self.bc, "_llm_client", None), \
                mock.patch.dict(sys.modules, {"anthropic": fake_anthropic}), \
                mock.patch.object(self.bc, "_mcu_phrases", self._phrases()):
            reply = self.bc._call_llm("hello there")
        self.assertEqual(reply, "Direct path, sir.")
        fake_anthropic.Anthropic.return_value.messages.create.assert_called_once()

    def test_unrecognised_4xx_falls_back_local(self):
        # 5992: generic BadRequestError (not a known phrase) → _local_fallback_or.
        a = mock.Mock()

        class _BadReq(Exception):
            pass
        a.BadRequestError = _BadReq
        a.RateLimitError = type("RL", (Exception,), {})
        a.APIStatusError = type("AS", (Exception,), {})
        client = mock.Mock()
        client.complete.side_effect = _BadReq("model not found for this account")
        with mock.patch.object(self.bc, "detect_tone", return_value=None), \
                mock.patch.object(self.bc, "route_voice_emotion",
                                  return_value={"mood": "casual", "addendum": ""}), \
                mock.patch.object(self.bc, "_voice_mood_response", None), \
                mock.patch.object(self.bc, "_emotion_tracker", None), \
                mock.patch.object(self.bc, "_llm_client", client), \
                mock.patch.dict(sys.modules, {"anthropic": a}), \
                mock.patch.object(self.bc, "_local_fallback_or",
                                  return_value="LOCAL4xx") as lf, \
                mock.patch.object(self.bc, "_mcu_phrases", self._phrases()):
            reply = self.bc._call_llm("hi")
        self.assertEqual(reply, "LOCAL4xx")
        self.assertIn("400", lf.call_args[0][1])

    def test_unexpected_exception_falls_back_local(self):
        # 6007-6008: a non-anthropic Exception type → generic fallback arm.
        a = mock.Mock()
        a.BadRequestError = type("BR", (Exception,), {})
        a.RateLimitError = type("RL", (Exception,), {})
        a.APIStatusError = type("AS", (Exception,), {})
        client = mock.Mock()
        client.complete.side_effect = KeyError("weird")
        with mock.patch.object(self.bc, "detect_tone", return_value=None), \
                mock.patch.object(self.bc, "route_voice_emotion",
                                  return_value={"mood": "casual", "addendum": ""}), \
                mock.patch.object(self.bc, "_voice_mood_response", None), \
                mock.patch.object(self.bc, "_emotion_tracker", None), \
                mock.patch.object(self.bc, "_llm_client", client), \
                mock.patch.dict(sys.modules, {"anthropic": a}), \
                mock.patch.object(self.bc, "_local_fallback_or",
                                  return_value="LOCALerr") as lf, \
                mock.patch.object(self.bc, "_mcu_phrases", self._phrases()):
            reply = self.bc._call_llm("hi")
        self.assertEqual(reply, "LOCALerr")
        self.assertIn("Unexpected LLM error", lf.call_args[0][1])

    def test_ollama_backend_success(self):
        # AI_BACKEND == "ollama" happy path now flows through the bounded helper.
        with mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
                mock.patch.object(self.bc, "detect_tone", return_value=None), \
                mock.patch.object(self.bc, "route_voice_emotion",
                                  return_value={"mood": "casual", "addendum": ""}), \
                mock.patch.object(self.bc, "_voice_mood_response", None), \
                mock.patch.object(self.bc, "_emotion_tracker", None), \
                mock.patch.object(self.bc, "_ollama_chat_bounded",
                                  return_value={"message": {"content": "Local says hi."}}), \
                mock.patch.object(self.bc, "_mcu_phrases", self._phrases()):
            reply = self.bc._call_llm("hi")
        self.assertEqual(reply, "Local says hi.")

    def test_phrase_rotation_persist_failure_is_swallowed(self):
        # 6046-6047: detect_phrases_in_reply raises → caught, reply unchanged.
        phrases = mock.Mock()
        phrases.detect_phrases_in_reply.side_effect = RuntimeError("phr boom")
        with mock.patch.object(self.bc, "detect_tone", return_value=None), \
                mock.patch.object(self.bc, "route_voice_emotion",
                                  return_value={"mood": "casual", "addendum": ""}), \
                mock.patch.object(self.bc, "_voice_mood_response", None), \
                mock.patch.object(self.bc, "_emotion_tracker", None), \
                mock.patch.object(self.bc, "_llm_client", self._client()), \
                mock.patch.dict(sys.modules, {"anthropic": mock.Mock()}), \
                mock.patch.object(self.bc, "_mcu_phrases", phrases):
            reply = self.bc._call_llm("hi")
        self.assertEqual(reply, "Indeed, sir.")


@requires_monolith
class CallLocalLlmBranchTests(MonolithGlobalsTestCase):
    """The _call_local_llm guard/return arms the first batch left uncovered:
    the install/pull kicks, the cheatsheet swap, the HTTP-not-ok and empty-
    text returns, and the outer-exception arm."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_disabled_returns_none(self):
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", False):
            self.assertIsNone(self.bc._call_local_llm("sys", []))

    def test_ollama_dead_kicks_install(self):
        # 5681-5683 (already partly covered) + ensures install fired here.
        install = mock.Mock()
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=False), \
                mock.patch.object(self.bc, "_ollama_install_async", install):
            self.assertIsNone(self.bc._call_local_llm("sys", []))
        install.assert_called_once()

    def test_model_absent_kicks_pull(self):
        pull = mock.Mock()
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_get_local_llm_model",
                                  return_value="m:tag"), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=False), \
                mock.patch.object(self.bc, "_ollama_pull_async", pull):
            self.assertIsNone(self.bc._call_local_llm("sys", []))
        pull.assert_called_once_with("m:tag")

    def test_swaps_pc_control_prompt_for_cheatsheet(self):
        # 5698-5699: PC_CONTROL_PROMPT present in system → replaced by the
        # compact cheatsheet before the POST.
        captured = {}
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None):
            captured["sys"] = json["messages"][0]["content"]
            return _FakeResp(ok=True, json_data={"message": {"content": "ok"}})
        fake_req.post.side_effect = _post
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_get_local_llm_model",
                                  return_value="m:tag"), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=True), \
                mock.patch.object(self.bc, "PC_CONTROL_PROMPT", "FULL-PC-PROMPT"), \
                mock.patch.object(self.bc, "_local_cheatsheet",
                                  return_value="CHEAT"), \
                mock.patch.object(self.bc, "requests", fake_req):
            out = self.bc._call_local_llm("persona FULL-PC-PROMPT tail", [])
        self.assertEqual(out, "ok")
        self.assertIn("CHEAT", captured["sys"])
        self.assertNotIn("FULL-PC-PROMPT", captured["sys"])

    def test_http_not_ok_returns_none(self):
        # 5751-5753.
        fake_req = mock.Mock()
        fake_req.post.return_value = _FakeResp(ok=False, status_code=500,
                                               text="boom")
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_get_local_llm_model",
                                  return_value="m:tag"), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=True), \
                mock.patch.object(self.bc, "requests", fake_req):
            self.assertIsNone(self.bc._call_local_llm("sys", []))

    def test_empty_text_returns_none(self):
        # 5755-5756: HTTP ok but the model produced only whitespace.
        fake_req = mock.Mock()
        fake_req.post.return_value = _FakeResp(
            ok=True, json_data={"message": {"content": "   "}})
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_get_local_llm_model",
                                  return_value="m:tag"), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=True), \
                mock.patch.object(self.bc, "requests", fake_req):
            self.assertIsNone(self.bc._call_local_llm("sys", []))

    def test_request_exception_returns_none(self):
        # 5759-5761: requests.post raises → caught, None returned.
        fake_req = mock.Mock()
        fake_req.post.side_effect = RuntimeError("network down")
        with mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(self.bc, "_ollama_alive", return_value=True), \
                mock.patch.object(self.bc, "_get_local_llm_model",
                                  return_value="m:tag"), \
                mock.patch.object(self.bc, "_ollama_has_model", return_value=True), \
                mock.patch.object(self.bc, "requests", fake_req):
            self.assertIsNone(self.bc._call_local_llm("sys", []))


@requires_monolith
class CallLocalVisionBranchTests(MonolithGlobalsTestCase):
    """The _call_local_vision failure arms: ollama-dead install kick, HTTP-
    not-ok, non-JSON body, wrong response shape, empty content, and the
    RequestException catch."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _enable(self):
        return [
            mock.patch.object(self.bc, "LOCAL_VISION_FALLBACK", True),
            mock.patch.object(self.bc, "LOCAL_VISION_MODEL", "vlm:7b"),
        ]

    def test_ollama_dead_kicks_install(self):
        install = mock.Mock()
        patches = self._enable() + [
            mock.patch.object(self.bc, "_ollama_alive", return_value=False),
            mock.patch.object(self.bc, "_ollama_install_async", install),
        ]
        for p in patches:
            p.start()
        try:
            self.assertIsNone(self.bc._call_local_vision("q", [b"img"]))
        finally:
            for p in patches:
                p.stop()
        install.assert_called_once()

    def _live_patches(self, fake_req):
        return self._enable() + [
            mock.patch.object(self.bc, "_ollama_alive", return_value=True),
            mock.patch.object(self.bc, "_ollama_has_model", return_value=True),
            mock.patch.object(self.bc, "_log_gpu_state"),
            mock.patch.object(self.bc, "requests", fake_req),
        ]

    def _run(self, fake_req):
        patches = self._live_patches(fake_req)
        for p in patches:
            p.start()
        try:
            return self.bc._call_local_vision("describe", [b"png"])
        finally:
            for p in patches:
                p.stop()

    def test_http_not_ok_returns_none(self):
        fake_req = mock.Mock()
        fake_req.post.return_value = _FakeResp(ok=False, status_code=503,
                                               text="down")
        fake_req.RequestException = self.bc.requests.RequestException
        self.assertIsNone(self._run(fake_req))

    def test_non_json_body_returns_none(self):
        bad = _FakeResp(ok=True)
        bad.json = mock.Mock(side_effect=ValueError("not json"))
        bad.text = "<html>"
        fake_req = mock.Mock()
        fake_req.post.return_value = bad
        fake_req.RequestException = self.bc.requests.RequestException
        self.assertIsNone(self._run(fake_req))

    def test_message_not_dict_returns_none(self):
        fake_req = mock.Mock()
        fake_req.post.return_value = _FakeResp(
            ok=True, json_data={"message": "oops a string"})
        fake_req.RequestException = self.bc.requests.RequestException
        self.assertIsNone(self._run(fake_req))

    def test_empty_content_returns_none(self):
        fake_req = mock.Mock()
        fake_req.post.return_value = _FakeResp(
            ok=True, json_data={"message": {"content": "  "},
                                "done_reason": "stop"})
        fake_req.RequestException = self.bc.requests.RequestException
        self.assertIsNone(self._run(fake_req))

    def test_request_exception_returns_none(self):
        fake_req = mock.Mock()
        fake_req.RequestException = self.bc.requests.RequestException
        fake_req.post.side_effect = self.bc.requests.RequestException("boom")
        self.assertIsNone(self._run(fake_req))


@requires_monolith
class TranscribeMoreBranchTests(MonolithGlobalsTestCase):
    """The openai-whisper empty-segments arm + the CUDA-error VRAM-recovery
    path (torch.cuda.empty_cache + model drop)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = (self.bc._stt, self.bc._stt_engine)

    def tearDown(self):
        self.bc._stt, self.bc._stt_engine = self._saved

    def test_openai_whisper_empty_segments(self):
        # 5298: openai path, no segments → fixed (1.0, -10.0) confidence.
        fake_model = mock.Mock()
        fake_model.transcribe.return_value = {"text": " hi ", "segments": []}
        with mock.patch.object(self.bc, "_ensure_whisper"), \
                mock.patch.object(self.bc, "_stt", fake_model), \
                mock.patch.object(self.bc, "_stt_engine", "openai_whisper"):
            text, conf = self.bc.transcribe(np.zeros(16, dtype=np.float32))
        self.assertEqual(text, "hi")
        self.assertEqual(conf["no_speech_prob"], 1.0)
        self.assertEqual(conf["avg_logprob"], -10.0)

    def test_openai_whisper_aggregates_segments(self):
        # 5299-5302: averages no_speech_prob / avg_logprob across segments.
        fake_model = mock.Mock()
        fake_model.transcribe.return_value = {
            "text": "a b",
            "segments": [
                {"no_speech_prob": 0.2, "avg_logprob": -1.0},
                {"no_speech_prob": 0.4, "avg_logprob": -3.0},
            ],
        }
        with mock.patch.object(self.bc, "_ensure_whisper"), \
                mock.patch.object(self.bc, "_stt", fake_model), \
                mock.patch.object(self.bc, "_stt_engine", "openai_whisper"):
            text, conf = self.bc.transcribe(np.zeros(16, dtype=np.float32))
        self.assertEqual(text, "a b")
        self.assertAlmostEqual(conf["no_speech_prob"], 0.3, places=6)
        self.assertAlmostEqual(conf["avg_logprob"], -2.0, places=6)

    def test_cuda_error_empties_cache_and_drops_model(self):
        # 5312-5322: a CUDA error string triggers torch.cuda.empty_cache()
        # and nulls _stt so the next utterance reloads clean.
        fake_model = mock.Mock()
        fake_model.transcribe.side_effect = RuntimeError("CUBLAS_STATUS failure")
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True
        with mock.patch.object(self.bc, "_ensure_whisper"), \
                mock.patch.object(self.bc, "_stt", fake_model), \
                mock.patch.object(self.bc, "_stt_engine", "faster_whisper"), \
                mock.patch.dict(sys.modules, {"torch": fake_torch}):
            text, conf = self.bc.transcribe(np.zeros(16, dtype=np.float32))
        self.assertEqual(text, "")
        fake_torch.cuda.empty_cache.assert_called_once()
        self.assertIsNone(self.bc._stt)

    def test_cuda_error_empty_cache_failure_still_drops_model(self):
        # 5318-5319: torch.cuda.empty_cache() itself raises -> the except
        # swallows it and the model is still dropped (_stt nulled) so recovery
        # proceeds on the next utterance.
        fake_model = mock.Mock()
        fake_model.transcribe.side_effect = RuntimeError("CUDA out of memory")
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.empty_cache.side_effect = RuntimeError("driver wedged")
        with mock.patch.object(self.bc, "_ensure_whisper"), \
                mock.patch.object(self.bc, "_stt", fake_model), \
                mock.patch.object(self.bc, "_stt_engine", "faster_whisper"), \
                mock.patch.dict(sys.modules, {"torch": fake_torch}):
            text, conf = self.bc.transcribe(np.zeros(16, dtype=np.float32))
        self.assertEqual(text, "")
        self.assertIsNone(self.bc._stt)


@requires_monolith
class RegisterCudaDllDirsErrorArmsTests(MonolithGlobalsTestCase):
    """The error arms of _register_cuda_dll_dirs the scan-success tests skip:
    nvidia import raising a non-ImportError, __path__ resolution failing, and
    add_dll_directory raising OSError (PATH prepend still counts as success)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        fn = self.bc._register_cuda_dll_dirs
        self._saved = {k: getattr(fn, k) for k in
                       ("_done", "_registered", "_missing", "_reason")
                       if hasattr(fn, k)}
        self._saved_path = os.environ.get("PATH", "")

    def tearDown(self):
        fn = self.bc._register_cuda_dll_dirs
        for k in ("_done", "_registered", "_missing", "_reason"):
            if k in self._saved:
                setattr(fn, k, self._saved[k])
            elif hasattr(fn, k):
                delattr(fn, k)
        os.environ["PATH"] = self._saved_path

    def test_nvidia_import_raises_nonimport_error(self):
        # 5015-5019: a non-ImportError from `import nvidia` records the reason.
        fn = self.bc._register_cuda_dll_dirs
        fn._done = False
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "nvidia":
                raise RuntimeError("namespace exploded")
            return real_import(name, *a, **k)
        with mock.patch("builtins.__import__", side_effect=fake_import):
            fn()
        self.assertTrue(fn._done)
        self.assertIn("nvidia import raised", fn._reason)

    def test_nvidia_path_resolution_fails(self):
        # 5023-5027: list(nvidia.__path__)[0] raises → reason recorded.
        fn = self.bc._register_cuda_dll_dirs
        fn._done = False
        fake_nvidia = mock.Mock()
        # __path__ that raises on iteration/index
        type(fake_nvidia).__path__ = mock.PropertyMock(
            side_effect=RuntimeError("no path"))
        with mock.patch.dict(sys.modules, {"nvidia": fake_nvidia}):
            fn()
        self.assertTrue(fn._done)
        self.assertIn("could not resolve nvidia.__path__", fn._reason)

    def test_add_dll_directory_oserror_still_registers_via_path(self):
        # 5055-5060: add_dll_directory raises OSError, but the PATH prepend
        # means the dir is still recorded as registered.
        fn = self.bc._register_cuda_dll_dirs
        fn._done = False
        fake_nvidia = mock.Mock()
        fake_nvidia.__path__ = [r"C:\fake\nvidia"]
        with mock.patch.dict(sys.modules, {"nvidia": fake_nvidia}), \
                mock.patch.object(self.bc.os.path, "isdir", return_value=True), \
                mock.patch.object(self.bc.os, "add_dll_directory",
                                  side_effect=OSError("denied"), create=True):
            fn()
        self.assertTrue(fn._done)
        # both dirs still counted as registered (PATH prepend worked)
        self.assertEqual(len(fn._registered), 2)
        # and both recorded as add_dll_directory-missing
        self.assertTrue(any("add_dll_directory" in m for m in fn._missing))


@requires_monolith
class EnsureWhisperOpenaiCudaFallbackTests(MonolithGlobalsTestCase):
    """The openai-whisper CUDA-load-failure → CPU-fallback arm (5142-5155)
    that the faster-whisper fallback test doesn't reach."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = (self.bc._stt, self.bc._stt_device,
                       self.bc._stt_model_name, self.bc._stt_engine)
        self.bc._stt = None

    def tearDown(self):
        (self.bc._stt, self.bc._stt_device,
         self.bc._stt_model_name, self.bc._stt_engine) = self._saved

    def test_openai_cuda_dll_failure_retries_cpu(self):
        good = object()
        calls = []

        def _load(model, device=None):
            calls.append((model, device))
            if device == "cuda":
                raise RuntimeError("Could not load library cudnn64_9.dll")
            return good
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "faster_whisper":
                raise ImportError("absent")
            return real_import(name, *a, **k)
        fake_whisper = mock.Mock()
        fake_whisper.load_model.side_effect = _load
        with mock.patch.object(self.bc, "_register_cuda_dll_dirs"), \
                mock.patch.object(self.bc, "_resolve_whisper_device",
                                  return_value="cuda"), \
                mock.patch.object(self.bc, "_force_whisper_cpu_int8", False), \
                mock.patch.object(self.bc, "WHISPER_MODEL_CUDA", "large-v3"), \
                mock.patch.object(self.bc, "WHISPER_MODEL_CPU", "base"), \
                mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch.dict(sys.modules, {"whisper": fake_whisper}):
            self.bc._ensure_whisper()
        self.assertIs(self.bc._stt, good)
        self.assertEqual(self.bc._stt_device, "cpu")
        self.assertEqual(self.bc._stt_model_name, "base")
        self.assertIn(("large-v3", "cuda"), calls)
        self.assertIn(("base", "cpu"), calls)

    def test_openai_non_dll_cuda_failure_retries_cpu(self):
        # 5149 (the else branch: generic CUDA error, not a DLL pattern).
        good = object()

        def _load(model, device=None):
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return good
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "faster_whisper":
                raise ImportError("absent")
            return real_import(name, *a, **k)
        fake_whisper = mock.Mock()
        fake_whisper.load_model.side_effect = _load
        with mock.patch.object(self.bc, "_register_cuda_dll_dirs"), \
                mock.patch.object(self.bc, "_resolve_whisper_device",
                                  return_value="cuda"), \
                mock.patch.object(self.bc, "_force_whisper_cpu_int8", False), \
                mock.patch.object(self.bc, "WHISPER_MODEL_CUDA", "large-v3"), \
                mock.patch.object(self.bc, "WHISPER_MODEL_CPU", "base"), \
                mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch.dict(sys.modules, {"whisper": fake_whisper}):
            self.bc._ensure_whisper()
        self.assertIs(self.bc._stt, good)
        self.assertEqual(self.bc._stt_device, "cpu")


@requires_monolith
class ResolveWhisperDeviceTorchTests(MonolithGlobalsTestCase):
    """The torch-backed auto branch (4976-4981) that the ctranslate2 test
    skips."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_auto_picks_cuda_via_torch_when_ct2_missing(self):
        # ctranslate2 import fails (printed warning, 4974-4975), torch sees a
        # GPU → "cuda".
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "ctranslate2":
                raise ImportError("no ct2")
            return real_import(name, *a, **k)
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True
        with mock.patch.object(self.bc, "WHISPER_DEVICE", "auto"), \
                mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(self.bc._resolve_whisper_device(), "cuda")

    def test_auto_falls_back_cpu_when_no_backend(self):
        # ct2 reports 0 devices, torch import fails → "cpu" (4981).
        fake_ct2 = mock.Mock()
        fake_ct2.get_cuda_device_count.return_value = 0
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *a, **k)
        with mock.patch.object(self.bc, "WHISPER_DEVICE", "auto"), \
                mock.patch.dict(sys.modules, {"ctranslate2": fake_ct2}), \
                mock.patch("builtins.__import__", side_effect=fake_import):
            self.assertEqual(self.bc._resolve_whisper_device(), "cpu")


@requires_monolith
class CheckDependenciesMultiFeatureTests(MonolithGlobalsTestCase):
    """The two-feature alert string (5250-5252) and the _speak-failure swallow
    (5256-5257)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_hist = list(self.bc.conversation_history)

    def tearDown(self):
        self.bc.conversation_history[:] = self._saved_hist

    def test_two_missing_features_builds_combined_alert(self):
        spoken = {}

        def _speak(msg):
            spoken["msg"] = msg
        notes = {"pkgA": "feature A is offline", "pkgB": "feature B is offline"}
        with mock.patch.object(self.bc, "_parse_requirements",
                               return_value=["pkgA", "pkgB"]), \
                mock.patch.object(self.bc, "_DEP_IMPORT_NAME", {}), \
                mock.patch.object(self.bc, "_DEP_FEATURE_NOTE", notes), \
                mock.patch.object(self.bc, "_speak", _speak):
            missing = self.bc.check_dependencies()
        self.assertEqual(missing, ["pkgA", "pkgB"])
        self.assertIn("a few things are offline", spoken["msg"])
        self.assertIn("feature A is offline", spoken["msg"])

    def test_three_missing_features_appends_and_more(self):
        spoken = {}
        notes = {"pkgA": "alpha", "pkgB": "beta", "pkgC": "gamma"}
        with mock.patch.object(self.bc, "_parse_requirements",
                               return_value=["pkgA", "pkgB", "pkgC"]), \
                mock.patch.object(self.bc, "_DEP_IMPORT_NAME", {}), \
                mock.patch.object(self.bc, "_DEP_FEATURE_NOTE", notes), \
                mock.patch.object(self.bc, "_speak",
                                  side_effect=lambda m: spoken.update(msg=m)):
            self.bc.check_dependencies()
        self.assertIn("and 1 more", spoken["msg"])

    def test_speak_failure_is_swallowed(self):
        # 5256-5257: _speak raises → caught; check_dependencies still returns.
        with mock.patch.object(self.bc, "_parse_requirements",
                               return_value=["pkgA"]), \
                mock.patch.object(self.bc, "_DEP_IMPORT_NAME", {}), \
                mock.patch.object(self.bc, "_DEP_FEATURE_NOTE",
                                  {"pkgA": "alpha"}), \
                mock.patch.object(self.bc, "_speak",
                                  side_effect=RuntimeError("tts down")):
            missing = self.bc.check_dependencies()
        self.assertEqual(missing, ["pkgA"])


@requires_monolith
class SmallHelperBranchTests(MonolithGlobalsTestCase):
    """One-off uncovered branches in small helpers: _log_gpu_state body,
    _get_local_llm_model's /api/tags failure, _ollama_has_model's exception,
    _local_cheatsheet's registry-failure swallow, _parse_intent_tag's empty
    input, and _resolve_tts_preset's recent_peak_rms failure."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_log_gpu_state_delegates_to_core(self):
        import core
        fake_gpu = mock.Mock()
        # `from core import gpu_state` resolves via the core package attribute
        # once core.gpu_state has been imported by any sibling test, so patch
        # BOTH sys.modules and the parent-package attr — otherwise this passes
        # alone but the real module is used in the full suite.
        with mock.patch.dict(sys.modules, {"core.gpu_state": fake_gpu}), \
             mock.patch.object(core, "gpu_state", fake_gpu, create=True):
            self.bc._log_gpu_state("m:tag")
        fake_gpu.log_gpu_state.assert_called_once_with("m:tag")

    def test_log_gpu_state_swallows_import_error(self):
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "core.gpu_state" or (
                    name == "core" and "gpu_state" in (a[2] or ())):
                raise ImportError("no gpu_state")
            return real_import(name, *a, **k)
        with mock.patch("builtins.__import__", side_effect=fake_import):
            # must not raise
            self.bc._log_gpu_state("m:tag")

    def test_get_local_llm_model_api_failure_returns_default(self):
        # 5484-5485: /api/tags raises → installed=[] → LOCAL_LLM_MODEL, no cache.
        self.bc._RESOLVED_LOCAL_LLM_MODEL[0] = None
        saved_env = os.environ.pop("JARVIS_LOCAL_LLM_MODEL", None)
        fake_req = mock.Mock()
        fake_req.get.side_effect = RuntimeError("connection refused")
        try:
            with mock.patch.object(self.bc, "requests", fake_req), \
                    mock.patch.object(self.bc, "LOCAL_LLM_MODEL", "def:model"):
                out = self.bc._get_local_llm_model()
            self.assertEqual(out, "def:model")
            self.assertIsNone(self.bc._RESOLVED_LOCAL_LLM_MODEL[0])
        finally:
            if saved_env is not None:
                os.environ["JARVIS_LOCAL_LLM_MODEL"] = saved_env

    def test_ollama_has_model_exception_returns_false(self):
        # 5519-5520.
        fake_req = mock.Mock()
        fake_req.get.side_effect = RuntimeError("down")
        with mock.patch.object(self.bc, "requests", fake_req):
            self.assertFalse(self.bc._ollama_has_model("m:tag"))

    def test_ollama_has_model_http_not_ok_returns_false(self):
        # 5513-5514.
        fake_req = mock.Mock()
        fake_req.get.return_value = _FakeResp(ok=False, status_code=500)
        with mock.patch.object(self.bc, "requests", fake_req):
            self.assertFalse(self.bc._ollama_has_model("m:tag"))

    def test_local_cheatsheet_registry_failure_swallowed(self):
        # 5660-5661: ACTIONS.keys() raises → allnames stays "", still returns.
        self.bc._LOCAL_CHEATSHEET_CACHE[0] = None
        boom = mock.Mock()
        boom.keys.side_effect = RuntimeError("registry exploded")
        try:
            with mock.patch.object(self.bc, "ACTIONS", boom):
                out = self.bc._local_cheatsheet()
            self.assertIn("CONTROLLING THE PC", out)
            self.assertIn("END PC CONTROL", out)
        finally:
            self.bc._LOCAL_CHEATSHEET_CACHE[0] = None

    def test_parse_intent_tag_empty_input(self):
        # 6132: empty string → (None, "").
        self.assertEqual(self.bc._parse_intent_tag(""), (None, ""))

    def test_resolve_tts_preset_peak_rms_failure_defaults_zero(self):
        # 6164-6165: recent_peak_rms raises → peak_rms=0.0, still resolves.
        fake_layer = mock.Mock()
        fake_layer.resolve_tts_preset.return_value = (
            "neutral", {"rate": "+0%", "pitch": "+0Hz", "gain": 1.0})
        fake_ap = mock.Mock()
        fake_ap.recent_peak_rms.side_effect = RuntimeError("no rms")
        with mock.patch.object(self.bc, "_tts_layer", fake_layer), \
                mock.patch.object(self.bc, "_last_emotion", [None]), \
                mock.patch.dict(sys.modules,
                                {"core.audio_processor": fake_ap}):
            chosen, preset = self.bc._resolve_tts_preset("hi", None)
        self.assertEqual(chosen, "neutral")
        # peak_rms defaulted to 0.0 on the exception
        self.assertEqual(fake_layer.resolve_tts_preset.call_args.kwargs["peak_rms"],
                         0.0)


@requires_monolith
class SynthesiseGainPathTests(MonolithGlobalsTestCase):
    """The gain-application + fall-through arms of synthesise the first batch
    left: xtts success with gain, pyttsx3-backend with gain + its failure
    fall-through, the wry-split exception, and the final edge→pyttsx3 fallback
    with gain."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _common(self, name, preset):
        return [
            mock.patch.object(self.bc, "_last_voice_route", [None]),
            mock.patch.object(self.bc, "_last_user_tone", [None]),
            mock.patch.object(self.bc, "_last_mood", [None]),
            mock.patch.object(self.bc, "_resolve_tts_preset",
                              return_value=(name, preset)),
        ]

    def test_xtts_success_applies_gain(self):
        # 6317-6319: xtts returns audio, gain != 1.0 → clipped + returned.
        base = np.full(6, 0.8, dtype=np.float32)
        preset = {"rate": "+0%", "pitch": "+0Hz", "gain": 2.0}
        patches = self._common("amused", preset) + [
            mock.patch.dict(self.bc.__dict__, {"TTS_BACKEND": "xtts"}),
            mock.patch.object(self.bc, "_render_xtts_or_raise",
                              return_value=(base, 24000)),
        ]
        for p in patches:
            p.start()
        try:
            audio, sr = self.bc.synthesise("hi")
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(sr, 24000)
        self.assertTrue(np.allclose(audio, 1.0))  # 0.8*2 clipped to 1.0

    def test_pyttsx3_backend_applies_gain(self):
        # 6326-6328: pyttsx3 backend selected, gain applied.
        base = np.full(6, 0.3, dtype=np.float32)
        preset = {"rate": "+0%", "pitch": "+0Hz", "gain": 2.0}
        patches = self._common("neutral", preset) + [
            mock.patch.dict(self.bc.__dict__, {"TTS_BACKEND": "pyttsx3"}),
            mock.patch.object(self.bc, "_pyttsx3_tts",
                              return_value=(base, 22050)),
        ]
        for p in patches:
            p.start()
        try:
            audio, sr = self.bc.synthesise("hi")
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(sr, 22050)
        self.assertTrue(np.allclose(audio, 0.6))

    def test_pyttsx3_backend_failure_falls_through_to_edge(self):
        # 6329-6330: pyttsx3 backend raises → falls through to the edge render.
        edge_audio = np.ones(4, dtype=np.float32)
        preset = {"rate": "+0%", "pitch": "+0Hz", "gain": 1.0}
        patches = self._common("neutral", preset) + [
            mock.patch.dict(self.bc.__dict__, {"TTS_BACKEND": "pyttsx3"}),
            mock.patch.object(self.bc, "_pyttsx3_tts",
                              side_effect=RuntimeError("pyttsx3 dead")),
            mock.patch.object(self.bc, "_render_edge_tts",
                              return_value=(edge_audio, 24000)),
        ]
        for p in patches:
            p.start()
        try:
            audio, sr = self.bc.synthesise("hi")
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.shape[0], 4)

    def test_wry_split_exception_falls_back_single_pass(self):
        # 6343-6344: split_for_wry_pause raises → wry_split stays None →
        # single-pass edge render still succeeds.
        fake_layer = mock.Mock()
        fake_layer.split_for_wry_pause.side_effect = RuntimeError("split boom")
        edge_audio = np.ones(7, dtype=np.float32)
        preset = {"rate": "+0%", "pitch": "+0Hz", "gain": 1.0}
        patches = self._common("wry", preset) + [
            mock.patch.object(self.bc, "_tts_layer", fake_layer),
            mock.patch.dict(self.bc.__dict__, {"TTS_BACKEND": "edge"}),
            mock.patch.object(self.bc, "_render_edge_tts",
                              return_value=(edge_audio, 24000)),
        ]
        for p in patches:
            p.start()
        try:
            audio, sr = self.bc.synthesise("Setup punchline")
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(audio.shape[0], 7)  # single-pass, no silence splice

    def test_edge_failure_falls_to_pyttsx3_with_gain(self):
        # 6360-6364: edge raises → pyttsx3 fallback, gain applied to its output.
        py_audio = np.full(5, 0.4, dtype=np.float32)
        preset = {"rate": "+0%", "pitch": "+0Hz", "gain": 2.0}
        patches = self._common("amused", preset) + [
            mock.patch.dict(self.bc.__dict__, {"TTS_BACKEND": "edge"}),
            mock.patch.object(self.bc, "_render_edge_tts",
                              side_effect=RuntimeError("edge 503")),
            mock.patch.object(self.bc, "_pyttsx3_tts",
                              return_value=(py_audio, 22050)),
        ]
        for p in patches:
            p.start()
        try:
            audio, sr = self.bc.synthesise("hi")
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(sr, 22050)
        self.assertTrue(np.allclose(audio, 0.8))  # 0.4*2


@requires_monolith
class Pyttsx3RenderTests(MonolithGlobalsTestCase):
    """The successful pyttsx3 temp-file render path (6400-6410) — engine
    saves a WAV, soundfile reads it back, temp file is cleaned up."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_engine_renders_wav_and_cleans_up(self):
        import soundfile as sf
        import tempfile
        # Real temp dir; the fake engine writes a real WAV there so sf.read
        # decodes it. Stub pyttsx3.init() to produce that file on save.
        tmpdir = tempfile.mkdtemp()
        produced = {}

        def _save_to_file(text, path):
            produced["path"] = path
            sf.write(path, np.full(120, 0.1, dtype=np.float32), 22050,
                     format="WAV")
        engine = mock.Mock()
        engine.save_to_file.side_effect = _save_to_file
        fake_pyttsx3 = mock.Mock()
        fake_pyttsx3.init.return_value = engine

        # Force NamedTemporaryFile into our temp dir so cleanup is observable.
        real_ntf = tempfile.NamedTemporaryFile

        def _ntf(*a, **k):
            k.setdefault("dir", tmpdir)
            return real_ntf(*a, **k)
        with mock.patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}), \
                mock.patch.object(self.bc.tempfile, "NamedTemporaryFile",
                                  side_effect=_ntf):
            audio, sr = self.bc._pyttsx3_tts("hello world")
        self.assertEqual(sr, 22050)
        self.assertEqual(audio.dtype, np.float32)
        engine.runAndWait.assert_called_once()
        engine.stop.assert_called_once()
        # temp .wav removed in the finally
        self.assertFalse(os.path.exists(produced["path"]))
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


@requires_monolith
class AudioDuckerWorkerTests(MonolithGlobalsTestCase):
    """_AudioDucker plumbing the enumerate/fade tests don't reach:
    _check_available, the worker loop drain (single job + sentinel),
    _ensure_worker reuse, _enumerate_targets error arms, and duck/restore
    enqueueing onto the (synchronous) worker."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_check_available_true_when_pycaw_imports(self):
        # 6514-6520: reset the class latch, force win32 + successful imports.
        saved = self.bc._AudioDucker._AVAILABLE
        self.bc._AudioDucker._AVAILABLE = None
        try:
            with mock.patch.object(self.bc.sys, "platform", "win32"), \
                    mock.patch.dict(sys.modules,
                                    {"pycaw": mock.Mock(),
                                     "pycaw.pycaw": mock.Mock(),
                                     "comtypes": mock.Mock()}):
                self.assertTrue(self.bc._AudioDucker._check_available())
            # cached now
            self.assertTrue(self.bc._AudioDucker._AVAILABLE)
        finally:
            self.bc._AudioDucker._AVAILABLE = saved

    def test_check_available_false_when_pycaw_missing(self):
        saved = self.bc._AudioDucker._AVAILABLE
        self.bc._AudioDucker._AVAILABLE = None
        real_import = __import__

        def fake_import(name, *a, **k):
            if name.startswith("pycaw"):
                raise ImportError("no pycaw")
            return real_import(name, *a, **k)
        try:
            with mock.patch.object(self.bc.sys, "platform", "win32"), \
                    mock.patch("builtins.__import__", side_effect=fake_import):
                self.assertFalse(self.bc._AudioDucker._check_available())
        finally:
            self.bc._AudioDucker._AVAILABLE = saved

    def test_worker_loop_runs_one_job_then_sentinel_stops(self):
        # 6540-6566: drain one (plans, level, cancellable, done_event) job —
        # _fade_run is stubbed — then a None sentinel returns out of the loop.
        d = self.bc._AudioDucker()
        done = threading.Event()
        d._work_queue.put(([("iface", 1.0)], 0.2, True, done))
        d._work_queue.put(None)  # sentinel → loop returns
        with mock.patch.dict(sys.modules, {"comtypes": mock.Mock()}), \
                mock.patch.object(d, "_fade_run") as fr:
            d._worker_loop()   # runs synchronously to the sentinel
        fr.assert_called_once_with([("iface", 1.0)], 0.2, True)
        self.assertTrue(done.is_set())   # done_event set in the finally

    def test_worker_loop_fade_exception_still_sets_done(self):
        # 6555-6559: _fade_run raises → caught, done_event still set.
        d = self.bc._AudioDucker()
        done = threading.Event()
        d._work_queue.put(([("iface", 1.0)], None, False, done))
        d._work_queue.put(None)
        with mock.patch.dict(sys.modules, {"comtypes": mock.Mock()}), \
                mock.patch.object(d, "_fade_run",
                                  side_effect=RuntimeError("fade boom")):
            d._worker_loop()
        self.assertTrue(done.is_set())

    def test_ensure_worker_reuses_live_thread(self):
        # 6528-6530: a live worker thread short-circuits _ensure_worker.
        d = self.bc._AudioDucker()
        live = mock.Mock()
        live.is_alive.return_value = True
        d._worker_thread = live
        with mock.patch.object(self.bc.threading, "Thread") as T:
            d._ensure_worker()
        T.assert_not_called()
        self.assertIs(d._worker_thread, live)

    def test_ensure_worker_spawns_when_none(self):
        # 6531-6537: no thread → a daemon worker is created + started.
        d = self.bc._AudioDucker()
        d._worker_thread = None
        with mock.patch.object(self.bc.threading, "Thread", _ImmediateThread):
            # _ImmediateThread.start() runs _worker_loop synchronously; feed a
            # sentinel first so it returns immediately instead of blocking.
            d._work_queue.put(None)
            with mock.patch.dict(sys.modules, {"comtypes": mock.Mock()}):
                d._ensure_worker()
        self.assertIsInstance(d._worker_thread, _ImmediateThread)

    def test_enumerate_get_all_sessions_failure_returns_empty(self):
        # 6580-6582: GetAllSessions raises → [] returned.
        d = self.bc._AudioDucker()
        fake_au = mock.Mock()
        fake_au.GetAllSessions.side_effect = RuntimeError("WASAPI boom")
        with mock.patch.dict(sys.modules,
                             {"pycaw": mock.Mock(),
                              "pycaw.pycaw": mock.Mock(AudioUtilities=fake_au),
                              "comtypes": mock.Mock()}):
            self.assertEqual(d._enumerate_targets(), [])

    def test_enumerate_skips_unmatched_and_null_volume(self):
        # 6591-6592 (name doesn't match) + 6594-6595 (vol_iface None) +
        # 6602-6603 (a session that raises mid-inspect is skipped).
        d = self.bc._AudioDucker()
        SELF = self.bc._AudioDucker._SELF_PID

        unmatched = mock.Mock()
        unmatched.Process = mock.Mock(pid=SELF + 1)
        unmatched.Process.name.return_value = "discord.exe"  # not a target

        null_vol = mock.Mock()
        null_vol.Process = mock.Mock(pid=SELF + 2)
        null_vol.Process.name.return_value = "spotify.exe"
        null_vol.SimpleAudioVolume = None

        boom = mock.Mock()
        boom.Process = mock.Mock(pid=SELF + 3)
        boom.Process.name.side_effect = RuntimeError("dead session")

        fake_au = mock.Mock()
        fake_au.GetAllSessions.return_value = [unmatched, null_vol, boom]
        with mock.patch.dict(sys.modules,
                             {"pycaw": mock.Mock(),
                              "pycaw.pycaw": mock.Mock(AudioUtilities=fake_au),
                              "comtypes": mock.Mock()}), \
                mock.patch.object(self.bc, "AUDIO_DUCKING_TARGETS", ["spotify"]), \
                mock.patch.object(self.bc, "AUDIO_DUCKING_LEVEL", 0.2):
            self.assertEqual(d._enumerate_targets(), [])

    def test_enumerate_comtypes_import_failure_path(self):
        # 6574-6576: comtypes import fails → com_inited stays False, the
        # CoUninitialize finally is skipped (6610-6611 not taken), still works.
        d = self.bc._AudioDucker()
        vol = mock.Mock()
        vol.GetMasterVolume.return_value = 0.9
        sess = mock.Mock()
        sess.Process = mock.Mock(pid=self.bc._AudioDucker._SELF_PID + 9)
        sess.Process.name.return_value = "spotify.exe"
        sess.SimpleAudioVolume = vol
        fake_au = mock.Mock()
        fake_au.GetAllSessions.return_value = [sess]
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "comtypes":
                raise ImportError("no comtypes")
            return real_import(name, *a, **k)
        with mock.patch.dict(sys.modules,
                             {"pycaw": mock.Mock(),
                              "pycaw.pycaw": mock.Mock(AudioUtilities=fake_au)}), \
                mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch.object(self.bc, "AUDIO_DUCKING_TARGETS", ["spotify"]), \
                mock.patch.object(self.bc, "AUDIO_DUCKING_LEVEL", 0.2):
            matched = d._enumerate_targets()
        self.assertEqual(len(matched), 1)

    def test_worker_loop_coinitialize_failure_swallowed(self):
        # 6545-6546: comtypes.CoInitialize() raises -> com_inited stays False,
        # the loop still drains its sentinel and the CoUninitialize finally is
        # skipped (no crash).
        d = self.bc._AudioDucker()
        done = threading.Event()
        d._work_queue.put(([("iface", 1.0)], 0.2, True, done))
        d._work_queue.put(None)
        bad_com = mock.Mock()
        bad_com.CoInitialize.side_effect = OSError("CoInitialize failed")
        with mock.patch.dict(sys.modules, {"comtypes": bad_com}), \
                mock.patch.object(d, "_fade_run"):
            d._worker_loop()
        self.assertTrue(done.is_set())
        bad_com.CoUninitialize.assert_not_called()   # never inited -> never uninit

    def test_worker_loop_couninitialize_failure_swallowed(self):
        # 6561-6566: CoInitialize succeeds but the finally's CoUninitialize
        # raises -> swallowed, the loop still returns cleanly.
        d = self.bc._AudioDucker()
        done = threading.Event()
        d._work_queue.put(([("iface", 1.0)], 0.2, True, done))
        d._work_queue.put(None)
        com = mock.Mock()
        com.CoUninitialize.side_effect = OSError("CoUninitialize failed")
        with mock.patch.dict(sys.modules, {"comtypes": com}), \
                mock.patch.object(d, "_fade_run"):
            d._worker_loop()   # must not raise
        self.assertTrue(done.is_set())
        com.CoUninitialize.assert_called_once()

    def test_enumerate_couninitialize_failure_swallowed(self):
        # 6606-6611: _enumerate_targets' finally CoUninitialize raises ->
        # swallowed; the matched list is still returned.
        d = self.bc._AudioDucker()
        vol = mock.Mock()
        vol.GetMasterVolume.return_value = 0.9
        sess = mock.Mock()
        sess.Process = mock.Mock(pid=self.bc._AudioDucker._SELF_PID + 11)
        sess.Process.name.return_value = "spotify.exe"
        sess.SimpleAudioVolume = vol
        fake_au = mock.Mock()
        fake_au.GetAllSessions.return_value = [sess]
        com = mock.Mock()
        com.CoUninitialize.side_effect = OSError("CoUninitialize failed")
        with mock.patch.dict(sys.modules,
                             {"pycaw": mock.Mock(),
                              "pycaw.pycaw": mock.Mock(AudioUtilities=fake_au),
                              "comtypes": com}), \
                mock.patch.object(self.bc, "AUDIO_DUCKING_TARGETS", ["spotify"]), \
                mock.patch.object(self.bc, "AUDIO_DUCKING_LEVEL", 0.2):
            matched = d._enumerate_targets()   # must not raise
        self.assertEqual(len(matched), 1)
        com.CoUninitialize.assert_called_once()

    def test_duck_enqueues_and_restore_drains(self):
        # duck(): 6646-6659 enqueue onto worker; restore(): 6661-6682 cancel +
        # fade-up enqueue. _ensure_worker + the worker are stubbed so nothing
        # spins; we assert the queue traffic + saved-state bookkeeping.
        d = self.bc._AudioDucker()
        iface = mock.Mock()
        with mock.patch.object(self.bc, "AUDIO_DUCKING_ENABLED", True), \
                mock.patch.object(d, "_check_available", return_value=True), \
                mock.patch.object(d, "_enumerate_targets",
                                  return_value=[(iface, 0.9)]), \
                mock.patch.object(d, "_ensure_worker"), \
                mock.patch.object(d._work_queue, "put") as put:
            d.duck()
            self.assertEqual(d._saved, [(iface, 0.9)])
            duck_job = put.call_args_list[0][0][0]
            self.assertEqual(duck_job[1], self.bc.AUDIO_DUCKING_LEVEL)
            self.assertTrue(duck_job[2])   # cancellable
        # restore: cancel set, saved cleared, fade-up (target None) enqueued
        with mock.patch.object(d, "_ensure_worker"), \
                mock.patch.object(d._work_queue, "put") as put2, \
                mock.patch.object(self.bc, "AUDIO_DUCKING_FADE_MS", 50):
            d.restore()
        self.assertEqual(d._saved, [])
        restore_job = put2.call_args_list[0][0][0]
        self.assertIsNone(restore_job[1])   # fade UP to original
        self.assertFalse(restore_job[2])    # not cancellable

    def test_duck_noop_when_no_targets(self):
        # 6651-6652: enumerate finds nothing → no save, no enqueue.
        d = self.bc._AudioDucker()
        with mock.patch.object(self.bc, "AUDIO_DUCKING_ENABLED", True), \
                mock.patch.object(d, "_check_available", return_value=True), \
                mock.patch.object(d, "_enumerate_targets", return_value=[]), \
                mock.patch.object(d._work_queue, "put") as put:
            d.duck()
        put.assert_not_called()
        self.assertEqual(d._saved, [])

    def test_duck_enumerate_exception_swallowed(self):
        # 6648-6650: _enumerate_targets raises → caught, no save.
        d = self.bc._AudioDucker()
        with mock.patch.object(self.bc, "AUDIO_DUCKING_ENABLED", True), \
                mock.patch.object(d, "_check_available", return_value=True), \
                mock.patch.object(d, "_enumerate_targets",
                                  side_effect=RuntimeError("enum boom")):
            d.duck()   # must not raise
        self.assertEqual(d._saved, [])

    def test_fade_run_steps_to_target(self):
        # 6618-6638 happy path (not cancelled): 10 volume steps toward target.
        d = self.bc._AudioDucker()
        iface = mock.Mock()
        iface.GetMasterVolume.return_value = 1.0
        with mock.patch.object(self.bc, "AUDIO_DUCKING_FADE_MS", 100), \
                mock.patch.object(self.bc.time, "sleep"):
            d._fade_run([(iface, 1.0)], 0.2, cancellable=False)
        self.assertEqual(iface.SetMasterVolume.call_count, 10)
        # final step lands near the 0.2 target
        last = iface.SetMasterVolume.call_args_list[-1][0][0]
        self.assertAlmostEqual(last, 0.2, places=6)

    def test_fade_run_skips_iface_that_raises_on_read(self):
        # 6622-6625: GetMasterVolume raises → that iface is dropped from the plan.
        d = self.bc._AudioDucker()
        bad = mock.Mock()
        bad.GetMasterVolume.side_effect = RuntimeError("read fail")
        with mock.patch.object(self.bc, "AUDIO_DUCKING_FADE_MS", 100), \
                mock.patch.object(self.bc.time, "sleep"):
            d._fade_run([(bad, 1.0)], 0.2, cancellable=False)
        bad.SetMasterVolume.assert_not_called()

    def test_fade_run_set_volume_exception_swallowed(self):
        # 6636-6637: the iface reads fine but SetMasterVolume raises on every
        # step → each failure is swallowed and the fade loop runs to completion.
        d = self.bc._AudioDucker()
        iface = mock.Mock()
        iface.GetMasterVolume.return_value = 1.0
        iface.SetMasterVolume.side_effect = RuntimeError("set fail")
        with mock.patch.object(self.bc, "AUDIO_DUCKING_FADE_MS", 100), \
                mock.patch.object(self.bc.time, "sleep"):
            d._fade_run([(iface, 1.0)], 0.2, cancellable=False)   # must not raise
        # all 10 steps attempted despite every SetMasterVolume raising
        self.assertEqual(iface.SetMasterVolume.call_count, 10)

    def test_restore_noop_when_nothing_saved(self):
        d = self.bc._AudioDucker()
        d._saved = []
        with mock.patch.object(d._work_queue, "put") as put:
            d.restore()
        put.assert_not_called()


@requires_monolith
class GetMicBufferTapPathTests(MonolithGlobalsTestCase):
    """The wake-listener tap (Path A) and record_speech tap (Path A2) return
    arms of get_mic_buffer — exercised by pre-seeding the tap queue and using
    a deterministic time source so the capture while-loop runs a bounded
    number of iterations without a real stream."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_active = list(self.bc._record_speech_active)
        self._saved_sr = list(self.bc._record_speech_sr)

    def tearDown(self):
        self.bc._record_speech_active[:] = self._saved_active
        self.bc._record_speech_sr[:] = self._saved_sr
        sys.modules.pop("skill_wake_listener", None)

    def test_path_a_taps_wake_listener(self):
        # 4812-4831: a running detector at the right rate → tap delivers frames.
        frame = np.ones(8000, dtype=np.float32)
        taps = []

        det = mock.Mock()
        det.is_running.return_value = True
        det.sample_rate = 16000

        def _add_tap(q):
            taps.append(q)
            q.put(frame)   # pre-seed exactly one full second of audio
        det.add_tap.side_effect = _add_tap
        det.remove_tap = mock.Mock()
        wl = mock.Mock()
        wl._detector = det
        with mock.patch.dict(sys.modules, {"skill_wake_listener": wl}), \
                mock.patch.object(self.bc, "_mic_input_disabled",
                                  return_value=False):
            out = self.bc.get_mic_buffer(0.5, sample_rate=16000)
        self.assertIsNotNone(out)
        self.assertEqual(out.dtype, np.float32)
        det.remove_tap.assert_called_once()
        # need = 16000*0.5 = 8000 samples; got exactly that, sliced to need
        self.assertEqual(out.size, 8000)

    def test_path_a_no_frames_returns_none(self):
        # 4828-4829: deadline passes with an empty tap queue → None.
        det = mock.Mock()
        det.is_running.return_value = True
        det.sample_rate = 16000
        det.add_tap = mock.Mock()       # never seeds the queue
        det.remove_tap = mock.Mock()
        wl = mock.Mock()
        wl._detector = det
        # Deterministic clock: first read of time() is the deadline base,
        # subsequent reads jump past it so the while-loop exits immediately.
        times = iter([1000.0, 1000.0, 9999.0, 9999.0, 9999.0])
        with mock.patch.dict(sys.modules, {"skill_wake_listener": wl}), \
                mock.patch.object(self.bc, "_mic_input_disabled",
                                  return_value=False), \
                mock.patch.object(self.bc.time, "time",
                                  side_effect=lambda: next(times)):
            out = self.bc.get_mic_buffer(0.1, sample_rate=16000)
        self.assertIsNone(out)
        det.remove_tap.assert_called_once()

    def test_path_a2_taps_record_speech_stream(self):
        # 4841-4862: no wake listener, but record_speech owns the mic at the
        # right rate → tap its frames.
        frame = np.full(8000, 0.2, dtype=np.float32)
        self.bc._record_speech_active[0] = True
        self.bc._record_speech_sr[0] = 16000

        def _add(q):
            q.put(frame)
        with mock.patch.object(self.bc, "_mic_input_disabled",
                               return_value=False), \
                mock.patch.object(self.bc, "add_record_tap",
                                  side_effect=_add) as art, \
                mock.patch.object(self.bc, "remove_record_tap") as rrt:
            out = self.bc.get_mic_buffer(0.5, sample_rate=16000)
        self.assertIsNotNone(out)
        self.assertEqual(out.size, 8000)
        art.assert_called_once()
        rrt.assert_called_once()

    def test_path_a2_no_frames_returns_none(self):
        # 4859-4860: record_speech live but never delivers → None (does NOT
        # fall through to Path B's competing open).
        self.bc._record_speech_active[0] = True
        self.bc._record_speech_sr[0] = 16000
        times = iter([1000.0, 1000.0, 9999.0, 9999.0, 9999.0])
        with mock.patch.object(self.bc, "_mic_input_disabled",
                               return_value=False), \
                mock.patch.object(self.bc, "add_record_tap"), \
                mock.patch.object(self.bc, "remove_record_tap") as rrt, \
                mock.patch.object(self.bc.time, "time",
                                  side_effect=lambda: next(times)):
            out = self.bc.get_mic_buffer(0.1, sample_rate=16000)
        self.assertIsNone(out)
        rrt.assert_called_once()

    def test_path_a2_breaks_when_record_speech_closes_mid_tap(self):
        # 4849-4850: we enter Path A2 with record_speech holding the mic, but
        # it releases ownership before any frame arrives — the loop sees the
        # flag flip False and breaks (returning None, no Path-B open).
        self.bc._record_speech_active[0] = True
        self.bc._record_speech_sr[0] = 16000

        def _add(_q):
            # record_speech "closes the stream" right after we registered.
            self.bc._record_speech_active[0] = False
        with mock.patch.object(self.bc, "_mic_input_disabled",
                               return_value=False), \
                mock.patch.object(self.bc, "add_record_tap", side_effect=_add), \
                mock.patch.object(self.bc, "remove_record_tap") as rrt:
            out = self.bc.get_mic_buffer(0.5, sample_rate=16000)
        self.assertIsNone(out)        # broke with nothing tapped
        rrt.assert_called_once()


@requires_monolith
class PlayWithLipsyncExtraPathsTests(MonolithGlobalsTestCase):
    """The barge-in arm (headset + listener + watch thread) and the robot
    lip-sync arm of play_with_lipsync. Threads run synchronously via
    _ImmediateThread so each closure body (_amp_pump, _barge_watch, _sync,
    _safe_wait*) executes once and the done-event is set before the bounded
    wait, exercising the post-play teardown."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_flag = list(self.bc._tts_playback_active)
        self._saved_barge = self.bc._barge_in_interrupted

    def tearDown(self):
        self.bc._tts_playback_active[:] = self._saved_flag
        self.bc._barge_in_interrupted = self._saved_barge

    def _base_patches(self, fake_sd, fake_layer, ducker):
        return [
            mock.patch.object(self.bc, "sd", fake_sd),
            mock.patch.object(self.bc, "_tts_layer", fake_layer),
            mock.patch.object(self.bc, "_audio_ducker", ducker),
            mock.patch.object(self.bc, "get_output_device", return_value=1),
            mock.patch.object(self.bc, "_write_hud_state"),
            mock.patch.object(self.bc, "_feed_playback_reference"),
            mock.patch.object(self.bc.threading, "Thread", _ImmediateThread),
            mock.patch.object(self.bc.time, "sleep"),
        ]

    def test_barge_in_headset_path_opens_and_closes_listener(self):
        # 6706-6726 + 6867-6868: BARGE_IN_ENABLED + headset → listener opened,
        # watch closure runs (flag set → sd.stop), stream closed in finally.
        fake_sd = mock.Mock()
        fake_layer = mock.Mock()
        fake_layer.is_muted.return_value = False
        ducker = mock.Mock()
        barge_stream = mock.Mock()
        # The watch closure observes the interrupt flag and calls sd.stop().
        self.bc._barge_in_interrupted = True
        audio = np.zeros(48, dtype=np.float32)
        scs_patch = mock.patch.object(self.bc, "_safe_close_stream")
        patches = self._base_patches(fake_sd, fake_layer, ducker) + [
            mock.patch.object(self.bc, "BARGE_IN_ENABLED", True),
            mock.patch.object(self.bc, "ROBOT_ENABLED", False),
            mock.patch.object(self.bc, "is_using_headset", return_value=True),
            mock.patch.object(self.bc, "_start_barge_in_listener",
                              return_value=barge_stream),
            scs_patch,
        ]
        started = [p.start() for p in patches]
        scs = started[patches.index(scs_patch)]
        try:
            self.bc.play_with_lipsync(audio, 24000)
        finally:
            for p in patches:
                p.stop()
        fake_sd.play.assert_called_once()
        # watch thread saw the interrupt and stopped playback
        fake_sd.stop.assert_called()
        scs.assert_called_once_with(barge_stream)
        self.assertFalse(self.bc._tts_playback_active[0])

    def test_robot_branch_runs_sync_loop(self):
        # 6804-6846: ROBOT_ENABLED → the _sync lip-sync closure streams mouth
        # values via send(); sd.play + bounded wait still run.
        fake_sd = mock.Mock()
        fake_layer = mock.Mock()
        fake_layer.is_muted.return_value = False
        ducker = mock.Mock()
        audio = np.full(96, 0.5, dtype=np.float32)
        send_patch = mock.patch.object(self.bc, "send")
        patches = self._base_patches(fake_sd, fake_layer, ducker) + [
            mock.patch.object(self.bc, "BARGE_IN_ENABLED", False),
            mock.patch.object(self.bc, "ROBOT_ENABLED", True),
            mock.patch.object(self.bc, "is_using_headset", return_value=False),
            send_patch,
            mock.patch.object(self.bc, "MOUTH_SCALE", 9.0),
        ]
        started = [p.start() for p in patches]
        send = started[patches.index(send_patch)]
        try:
            self.bc.play_with_lipsync(audio, 24000)
        finally:
            for p in patches:
                p.stop()
        fake_sd.play.assert_called_once()
        # _sync streamed at least one mouth value and a final mouth=0.0
        self.assertTrue(send.called)
        self.assertFalse(self.bc._tts_playback_active[0])

    def test_robot_sync_send_exception_is_logged_not_raised(self):
        # 6817-6818 + 6823-6824: send() raises inside _sync → caught per-
        # iteration; play_with_lipsync still completes.
        fake_sd = mock.Mock()
        fake_layer = mock.Mock()
        fake_layer.is_muted.return_value = False
        ducker = mock.Mock()
        audio = np.full(96, 0.5, dtype=np.float32)
        patches = self._base_patches(fake_sd, fake_layer, ducker) + [
            mock.patch.object(self.bc, "BARGE_IN_ENABLED", False),
            mock.patch.object(self.bc, "ROBOT_ENABLED", True),
            mock.patch.object(self.bc, "is_using_headset", return_value=False),
            mock.patch.object(self.bc, "send",
                              side_effect=RuntimeError("robot offline")),
        ]
        for p in patches:
            p.start()
        try:
            self.bc.play_with_lipsync(audio, 24000)   # must not raise
        finally:
            for p in patches:
                p.stop()
        self.assertFalse(self.bc._tts_playback_active[0])


# ===========================================================================
# FOURTH BATCH — remaining cheap, fully-mockable branches: get_mic_buffer's
# Path-B capture (driven via the captured callback), _tts_bytes' async edge
# stream, the _render_edge_tts cache/ndim arms, _call_local_llm's guard
# exception swallows, _pyttsx3_tts' stereo + unlink arms, _call_llm's
# mode-router exception, transcribe's faster-whisper non-CUDA error, and the
# _AudioDucker.restore wait arms.
# ===========================================================================
@requires_monolith
class GetMicBufferPathBTests(MonolithGlobalsTestCase):
    """Drive get_mic_buffer's fallback InputStream (Path B): capture the
    callback handed to the (mocked) stream, push frames through it to fill the
    buffer, and confirm the slice + teardown. No wake listener, no live
    record_speech."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_active = list(self.bc._record_speech_active)

    def tearDown(self):
        self.bc._record_speech_active[:] = self._saved_active

    def test_path_b_captures_via_callback(self):
        # 4865-4901: open a mock stream, feed one full frame through its
        # callback so the while-loop reaches `need` and returns the slice.
        self.bc._record_speech_active[0] = False
        holder = {}

        class _FakeStream:
            def __init__(self, *a, **k):
                holder["cb"] = k["callback"]

            def start(self):
                # Deliver a full second of audio the moment capture starts.
                frame = np.ones(16000, dtype=np.float32)
                # indata is (frames, channels); the cb flattens column 0.
                holder["cb"](frame.reshape(-1, 1), 16000, None, None)

        fake_sd = mock.Mock()
        fake_sd.InputStream.side_effect = _FakeStream
        with mock.patch.object(self.bc, "_mic_input_disabled",
                               return_value=False), \
                mock.patch.object(self.bc, "sd", fake_sd), \
                mock.patch.object(self.bc, "get_input_device", return_value=3), \
                mock.patch.object(self.bc, "_safe_close_stream") as scs, \
                mock.patch.object(self.bc.time, "sleep"):
            out = self.bc.get_mic_buffer(0.5, sample_rate=16000)
        self.assertIsNotNone(out)
        self.assertEqual(out.size, 8000)   # need = 16000*0.5, sliced down
        self.assertEqual(out.dtype, np.float32)
        scs.assert_called_once()

    def test_path_b_no_frames_returns_none(self):
        # 4898-4899: stream starts but never delivers → deadline → None.
        self.bc._record_speech_active[0] = False

        class _SilentStream:
            def __init__(self, *a, **k):
                pass

            def start(self):
                pass

        fake_sd = mock.Mock()
        fake_sd.InputStream.side_effect = _SilentStream
        times = iter([1000.0, 1000.0, 9999.0, 9999.0, 9999.0])
        with mock.patch.object(self.bc, "_mic_input_disabled",
                               return_value=False), \
                mock.patch.object(self.bc, "sd", fake_sd), \
                mock.patch.object(self.bc, "get_input_device", return_value=3), \
                mock.patch.object(self.bc, "_safe_close_stream"), \
                mock.patch.object(self.bc.time, "time",
                                  side_effect=lambda: next(times)):
            out = self.bc.get_mic_buffer(0.1, sample_rate=16000)
        self.assertIsNone(out)

    def test_path_b_start_failure_returns_none(self):
        # 4893-4895: the Path-B stream constructs fine but .start() raises ->
        # the except prints and returns None (stream still closed in finally).
        self.bc._record_speech_active[0] = False

        class _BadStartStream:
            def __init__(self, *a, **k):
                pass

            def start(self):
                raise RuntimeError("stream start failed")

        fake_sd = mock.Mock()
        fake_sd.InputStream.side_effect = _BadStartStream
        with mock.patch.object(self.bc, "_mic_input_disabled",
                               return_value=False), \
                mock.patch.object(self.bc, "sd", fake_sd), \
                mock.patch.object(self.bc, "get_input_device", return_value=3), \
                mock.patch.object(self.bc, "_safe_close_stream") as scs:
            out = self.bc.get_mic_buffer(0.1, sample_rate=16000)
        self.assertIsNone(out)
        scs.assert_called_once()


@requires_monolith
class TtsBytesAsyncTests(MonolithGlobalsTestCase):
    """Run the _tts_bytes coroutine against a fake edge_tts.Communicate whose
    .stream() is an async generator, asserting only audio chunks are kept."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_streams_audio_chunks_only(self):
        import asyncio

        class _FakeCommunicate:
            def __init__(self, text, voice, rate=None, pitch=None):
                self.text = text

            async def stream(self):
                yield {"type": "audio", "data": b"AB"}
                yield {"type": "WordBoundary", "data": b"ignored"}
                yield {"type": "audio", "data": b"CD"}

        fake_edge = mock.Mock()
        fake_edge.Communicate = _FakeCommunicate
        with mock.patch.dict(sys.modules, {"edge_tts": fake_edge}):
            out = asyncio.run(self.bc._tts_bytes("hi", rate="+0%", pitch="+0Hz"))
        self.assertEqual(out, b"ABCD")   # word-boundary chunk dropped


@requires_monolith
class RenderEdgeTtsCacheArmsTests(MonolithGlobalsTestCase):
    """The cache-get exception swallow (6240-6241), the stereo→mono mean
    (6251), and the cache-put exception swallow (6255-6256)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_cache_get_raises_then_renders_stereo_and_put_raises(self):
        import soundfile as sf
        # Build a STEREO wav so the ndim>1 mean path (6251) runs.
        buf = io.BytesIO()
        sf.write(buf, np.zeros((240, 2), dtype=np.float32), 24000, format="WAV")
        buf.seek(0)
        raw_wav = buf.read()

        fake_layer = mock.Mock()
        fake_layer.tts_cache_get.side_effect = RuntimeError("cache read boom")
        fake_layer.tts_cache_put.side_effect = RuntimeError("cache write boom")
        fake_future = mock.Mock()
        fake_future.result.return_value = raw_wav
        with mock.patch.object(self.bc, "_tts_layer", fake_layer), \
                mock.patch.object(self.bc, "_ensure_tts_loop"), \
                mock.patch.object(self.bc, "_tts_loop", mock.Mock()), \
                mock.patch.object(self.bc.asyncio, "run_coroutine_threadsafe",
                                  return_value=fake_future):
            audio, sr = self.bc._render_edge_tts("hi", "+0%", "+0Hz")
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.ndim, 1)   # collapsed to mono
        fake_layer.tts_cache_put.assert_called_once()


@requires_monolith
class CallLocalLlmGuardArmsTests(MonolithGlobalsTestCase):
    """The narrow swallow/branch arms inside _call_local_llm: the cheatsheet
    replace raising (5700-5701) and the web-search scan visiting a non-str
    content / a see_screen-after-search (5711, 5715, 5723-5724)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _live(self, fake_req):
        return [
            mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True),
            mock.patch.object(self.bc, "_ollama_alive", return_value=True),
            mock.patch.object(self.bc, "_get_local_llm_model",
                              return_value="m:tag"),
            mock.patch.object(self.bc, "_ollama_has_model", return_value=True),
            mock.patch.object(self.bc, "requests", fake_req),
        ]

    def test_cheatsheet_replace_exception_is_swallowed(self):
        # 5700-5701: `_local_cheatsheet()` raises during the prompt swap →
        # caught, the original system prompt is used, call still completes.
        fake_req = mock.Mock()
        fake_req.post.return_value = _FakeResp(
            ok=True, json_data={"message": {"content": "ok"}})
        patches = self._live(fake_req) + [
            mock.patch.object(self.bc, "PC_CONTROL_PROMPT", "FULL"),
            mock.patch.object(self.bc, "_local_cheatsheet",
                              side_effect=RuntimeError("cheat boom")),
        ]
        for p in patches:
            p.start()
        try:
            out = self.bc._call_local_llm("persona FULL tail", [])
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(out, "ok")

    def test_web_search_scan_handles_nonstr_and_see_after_search(self):
        # 5711 (content not str → continue) + 5712-5715 (search then a later
        # see_screen → guard NOT prepended).
        captured = {}
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None):
            captured["sys"] = json["messages"][0]["content"]
            return _FakeResp(ok=True, json_data={"message": {"content": "ok"}})
        fake_req.post.side_effect = _post
        msgs = [
            {"role": "assistant", "content": ["not", "a", "string"]},
            {"role": "assistant", "content": "[ACTION: web_search, x]"},
            {"role": "assistant", "content": "[ACTION: see_screen]"},
        ]
        patches = self._live(fake_req)
        for p in patches:
            p.start()
        try:
            out = self.bc._call_local_llm("BASE", msgs)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(out, "ok")
        # see_screen came AFTER the search → no fabrication guard prepended
        self.assertNotIn("Do NOT fabricate", captured["sys"])


@requires_monolith
class Pyttsx3RenderArmTests(MonolithGlobalsTestCase):
    """_pyttsx3_tts' stereo-mean (6409) and the temp-file unlink OSError
    swallow (6417-6418)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_stereo_output_collapsed_and_unlink_oserror_swallowed(self):
        import soundfile as sf
        import tempfile
        tmpdir = tempfile.mkdtemp()
        produced = {}

        def _save_to_file(text, path):
            produced["path"] = path
            sf.write(path, np.zeros((100, 2), dtype=np.float32), 22050,
                     format="WAV")
        engine = mock.Mock()
        engine.save_to_file.side_effect = _save_to_file
        fake_pyttsx3 = mock.Mock()
        fake_pyttsx3.init.return_value = engine
        real_ntf = tempfile.NamedTemporaryFile

        def _ntf(*a, **k):
            k.setdefault("dir", tmpdir)
            return real_ntf(*a, **k)
        with mock.patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}), \
                mock.patch.object(self.bc.tempfile, "NamedTemporaryFile",
                                  side_effect=_ntf), \
                mock.patch.object(self.bc.os, "unlink",
                                  side_effect=OSError("file busy")):
            audio, sr = self.bc._pyttsx3_tts("hi")
        self.assertEqual(sr, 22050)
        self.assertEqual(audio.ndim, 1)   # stereo collapsed to mono
        # clean up the real temp artifacts ourselves (unlink was stubbed out)
        try:
            if produced.get("path") and os.path.exists(produced["path"]):
                os.remove(produced["path"])
            os.rmdir(tmpdir)
        except OSError:
            pass


@requires_monolith
class CallLlmModeRouterArmTests(MonolithGlobalsTestCase):
    """_call_llm's mode-router addendum arm — both a real addendum and the
    import-failure swallow (5942-5943)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_hist = list(self.bc.conversation_history)
        self.bc.conversation_history.clear()

    def tearDown(self):
        self.bc.conversation_history[:] = self._saved_hist

    def _phrases(self):
        m = mock.Mock()
        m.detect_phrases_in_reply.return_value = {}
        return m

    def test_mode_router_import_failure_swallowed(self):
        # core.mode_router import raises → mode_addendum stays "", call ok.
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "core.mode_router" or (
                    name == "core" and "mode_router" in (a[2] or ())):
                raise ImportError("no mode_router")
            return real_import(name, *a, **k)
        client = mock.Mock()
        client.complete.return_value = "Very good, sir."
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
                mock.patch.object(self.bc, "detect_tone", return_value=None), \
                mock.patch.object(self.bc, "route_voice_emotion",
                                  return_value={"mood": "casual", "addendum": ""}), \
                mock.patch.object(self.bc, "_voice_mood_response", None), \
                mock.patch.object(self.bc, "_emotion_tracker", None), \
                mock.patch.object(self.bc, "_trim_conversation_history"), \
                mock.patch.object(self.bc, "_system_prompt", "SYS"), \
                mock.patch.object(self.bc, "_llm_client", client), \
                mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch.object(self.bc, "_mcu_phrases", self._phrases()):
            reply = self.bc._call_llm("hi")
        self.assertEqual(reply, "Very good, sir.")


@requires_monolith
class AudioDuckerRestoreWaitTests(MonolithGlobalsTestCase):
    """_AudioDucker.restore's fade_done.wait + restore_done.wait timeout
    swallows (6672-6673, 6681-6682)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_restore_swallows_wait_exceptions(self):
        d = self.bc._AudioDucker()
        d._saved = [(mock.Mock(), 0.9)]
        # Make BOTH the fade_done wait and the freshly-created restore_done
        # wait raise, so both try/except arms (6672-6673, 6681-6682) run.
        boom_done = mock.Mock()
        boom_done.wait.side_effect = RuntimeError("wait boom")
        with mock.patch.object(d, "_ensure_worker"), \
                mock.patch.object(d._work_queue, "put"), \
                mock.patch.object(self.bc.threading, "Event",
                                  return_value=boom_done):
            # also force the in-flight fade_done to the booming event
            d._fade_done = boom_done
            d.restore()   # must not raise
        self.assertEqual(d._saved, [])


@requires_monolith
class OllamaAsyncFailureArmTests(MonolithGlobalsTestCase):
    """The failure / success arms inside the background `_do` closures that the
    happy-path async tests don't reach: winget install failure (5538-5539),
    text-pull failure (5559-5560), and the vision-pull success print
    (5800-5802). All run synchronously via _ImmediateThread."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = (list(self.bc._OLLAMA_INSTALL_TRIGGERED),
                       list(self.bc._OLLAMA_PULL_TRIGGERED),
                       list(self.bc._LOCAL_VISION_PULL_TRIGGERED))
        self.bc._OLLAMA_INSTALL_TRIGGERED[0] = False
        self.bc._OLLAMA_PULL_TRIGGERED[0] = False
        self.bc._LOCAL_VISION_PULL_TRIGGERED[0] = False

    def tearDown(self):
        (self.bc._OLLAMA_INSTALL_TRIGGERED[:],
         self.bc._OLLAMA_PULL_TRIGGERED[:],
         self.bc._LOCAL_VISION_PULL_TRIGGERED[:]) = self._saved

    def test_install_winget_failure_swallowed(self):
        # 5538-5539: subprocess.run raises → caught inside _do.
        fake_sp = mock.Mock()
        fake_sp.run.side_effect = RuntimeError("winget exploded")
        with mock.patch.object(self.bc.threading, "Thread", _ImmediateThread), \
                mock.patch.dict(sys.modules, {"subprocess": fake_sp}):
            self.bc._ollama_install_async()   # must not raise
        fake_sp.run.assert_called_once()
        self.assertTrue(self.bc._OLLAMA_INSTALL_TRIGGERED[0])

    def test_text_pull_failure_swallowed(self):
        # 5559-5560: requests.post raises → caught inside _do.
        fake_req = mock.Mock()
        fake_req.post.side_effect = RuntimeError("pull down")
        with mock.patch.object(self.bc.threading, "Thread", _ImmediateThread), \
                mock.patch.object(self.bc, "requests", fake_req):
            self.bc._ollama_pull_async("some:model")
        fake_req.post.assert_called_once()
        self.assertTrue(self.bc._OLLAMA_PULL_TRIGGERED[0])

    def test_vision_pull_success_drains_stream(self):
        # 5800-5802: a streamed pull that yields lines → the for-loop drains
        # them and the success print fires (latch stays set).
        resp = _FakeResp(ok=True)
        resp.iter_lines = lambda: iter([b'{"status":"pulling"}',
                                        b'{"status":"success"}'])
        fake_req = mock.Mock()
        fake_req.RequestException = self.bc.requests.RequestException
        fake_req.post.return_value = resp
        with mock.patch.object(self.bc.threading, "Thread", _ImmediateThread), \
                mock.patch.object(self.bc, "requests", fake_req):
            self.bc._ollama_pull_vision_async("vlm:7b")
        fake_req.post.assert_called_once()
        # success path → latch NOT reset
        self.assertTrue(self.bc._LOCAL_VISION_PULL_TRIGGERED[0])


@requires_monolith
class EnsureWhisperRaiseArmTests(MonolithGlobalsTestCase):
    """The `raise` arms of _ensure_whisper that fire when a NON-cuda device
    load fails (faster-whisper 5132, openai-whisper 5155) and the non-DLL
    CUDA-error print (5124)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = (self.bc._stt, self.bc._stt_device,
                       self.bc._stt_model_name, self.bc._stt_engine)
        self.bc._stt = None

    def tearDown(self):
        (self.bc._stt, self.bc._stt_device,
         self.bc._stt_model_name, self.bc._stt_engine) = self._saved

    def test_faster_whisper_cpu_load_failure_propagates(self):
        # device resolves to cpu; the faster-whisper load raises → 5132 `raise`
        # (no CPU retry since we're already on CPU). The outer caller sees it.
        fake_fw = mock.Mock()
        fake_fw.WhisperModel = mock.Mock(side_effect=RuntimeError("cpu load boom"))
        with mock.patch.object(self.bc, "_register_cuda_dll_dirs"), \
                mock.patch.object(self.bc, "_resolve_whisper_device",
                                  return_value="cpu"), \
                mock.patch.object(self.bc, "_force_whisper_cpu_int8", False), \
                mock.patch.object(self.bc, "WHISPER_MODEL_CPU", "base"), \
                mock.patch.dict(sys.modules, {"faster_whisper": fake_fw}):
            with self.assertRaises(RuntimeError):
                self.bc._ensure_whisper()
        self.assertIsNone(self.bc._stt)

    def test_openai_whisper_cpu_load_failure_propagates(self):
        # faster-whisper absent → openai path; cpu load_model raises → 5155.
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "faster_whisper":
                raise ImportError("absent")
            return real_import(name, *a, **k)
        fake_whisper = mock.Mock()
        fake_whisper.load_model.side_effect = RuntimeError("cpu whisper boom")
        with mock.patch.object(self.bc, "_register_cuda_dll_dirs"), \
                mock.patch.object(self.bc, "_resolve_whisper_device",
                                  return_value="cpu"), \
                mock.patch.object(self.bc, "WHISPER_MODEL_CPU", "base"), \
                mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch.dict(sys.modules, {"whisper": fake_whisper}):
            with self.assertRaises(RuntimeError):
                self.bc._ensure_whisper()

    def test_faster_whisper_non_dll_cuda_error_retries_cpu(self):
        # 5124 (else: a non-DLL CUDA error prints the plain message) then the
        # cuda branch retries on CPU and succeeds.
        good = object()

        def _wm(model, device=None, compute_type=None):
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")  # not a DLL pattern
            return good
        fake_fw = mock.Mock()
        fake_fw.WhisperModel = mock.Mock(side_effect=_wm)
        with mock.patch.object(self.bc, "_register_cuda_dll_dirs"), \
                mock.patch.object(self.bc, "_resolve_whisper_device",
                                  return_value="cuda"), \
                mock.patch.object(self.bc, "_force_whisper_cpu_int8", False), \
                mock.patch.object(self.bc, "WHISPER_MODEL_CUDA", "large-v3"), \
                mock.patch.dict(sys.modules, {"faster_whisper": fake_fw}):
            self.bc._ensure_whisper()
        self.assertIs(self.bc._stt, good)
        self.assertEqual(self.bc._stt_device, "cpu")


# ===========================================================================
# COVERAGE-COMPLETION BATCH — record_speech live-stream setup/teardown driven
# through a fake InputStream, plus play_with_lipsync's defensive exception
# handlers. The capture LOOP body itself is pragma'd in the monolith as live-
# mic-only; here we feed frames through the captured callback so the setup
# (4587-4598), post-start publish (4638-4640), the finally release (4771-4772)
# and the post-loop concatenate/return (4774-4781) all execute, and we drive
# the InputStream.start() failure arm (4632-4635) with a stream whose start
# raises.
# ===========================================================================
class _FakeRecordStream:
    """Fake sd.InputStream for record_speech: captures the callback handed to
    it and, on start(), feeds a 1 loud + N silent frame burst so the VAD loop
    trips into recording then breaks on sustained silence."""

    def __init__(self, *a, **kw):
        self.callback = kw.get("callback")
        self.started = False
        self.closed = False
        self._frames = kw.pop("_frames", None)

    def start(self):
        self.started = True
        for f in (self._frames or []):
            self.callback(f, len(f), None, None)

    def stop(self):
        pass

    def close(self):
        self.closed = True


@requires_monolith
class RecordSpeechLiveStreamTests(MonolithGlobalsTestCase):
    """Drive record_speech past the mic-disabled short-circuit with a fake
    InputStream so the (otherwise live-mic-only) setup/teardown statements run
    deterministically. The pragma'd capture-loop body executes incidentally as
    the fed frames are drained; we assert the function's observable result."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _common_patches(self, fake_sd):
        return [
            mock.patch.object(self.bc, "_mic_input_disabled", return_value=False),
            mock.patch.object(self.bc, "sd", fake_sd),
            mock.patch.object(self.bc, "get_input_device", return_value=1),
            mock.patch.object(self.bc, "_process_capture_chunk",
                              side_effect=lambda c, sr: c),
            mock.patch.object(self.bc, "pause_face_tracking"),
            mock.patch.object(self.bc, "set_state"),
            mock.patch.object(self.bc, "_heartbeat"),
            mock.patch.object(self.bc, "_write_hud_state"),
            mock.patch.object(self.bc, "HUD_ENABLED", False),
            mock.patch.object(self.bc, "_record_speech_active", [False]),
            mock.patch.object(self.bc, "_record_speech_sr", [0]),
        ]

    def test_happy_path_records_and_returns_audio(self):
        # 4587-4598 setup, 4638-4640 publish-ownership, the capture loop break
        # on sustained silence, 4771-4772 finally release, 4774-4781 post-loop
        # concat + return. silence_lim == int(1.4*16000/1024) == 21, so 1 loud
        # frame trips recording and 21 silent frames end the utterance.
        loud = np.full(1024, 0.5, dtype=np.float32).reshape(-1, 1)
        silent = np.zeros((1024, 1), dtype=np.float32)
        frames = [loud] + [silent] * 21
        fake_sd = mock.Mock()
        fake_sd.InputStream.side_effect = (
            lambda *a, **kw: _FakeRecordStream(*a, _frames=frames, **kw))
        scs_patch = mock.patch.object(self.bc, "_safe_close_stream")
        patches = self._common_patches(fake_sd) + [scs_patch]
        started = [p.start() for p in patches]
        scs = started[patches.index(scs_patch)]
        try:
            out = self.bc.record_speech(timeout=None)
        finally:
            for p in patches:
                p.stop()
        # 22 frames of 1024 samples each were concatenated and flattened.
        self.assertIsNotNone(out)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.size, 22 * 1024)
        # ownership flag was flipped True (4639) then released in finally (4771)
        self.assertFalse(self.bc._record_speech_active[0])
        scs.assert_called_once()

    def test_no_chunks_returns_none(self):
        # 4779-4780: stream opens + starts but no frame ever crosses the VAD
        # floor before the timeout, so chunks stays empty and we return None
        # (distinct from the mic-disabled short-circuit). One sub-threshold
        # frame, timeout already elapsed -> 4763 timeout-return path drains to
        # the finally; assert the graceful None.
        silent = np.zeros((1024, 1), dtype=np.float32)
        fake_sd = mock.Mock()
        fake_sd.InputStream.side_effect = (
            lambda *a, **kw: _FakeRecordStream(*a, _frames=[silent], **kw))
        patches = self._common_patches(fake_sd) + [
            mock.patch.object(self.bc, "_safe_close_stream"),
        ]
        for p in patches:
            p.start()
        try:
            # timeout=0 so the first sub-threshold frame's elapsed>=timeout
            # check returns None without recording.
            out = self.bc.record_speech(timeout=0.0)
        finally:
            for p in patches:
                p.stop()
        self.assertIsNone(out)

    def test_inputstream_start_failure_returns_none(self):
        # 4632-4635: InputStream opens but .start() raises -> logged,
        # _safe_close_stream invoked, None returned (before the capture loop).
        class _BadStartStream(_FakeRecordStream):
            def start(self):
                raise RuntimeError("start boom")

        fake_sd = mock.Mock()
        fake_sd.InputStream.side_effect = lambda *a, **kw: _BadStartStream(*a, **kw)
        scs_patch = mock.patch.object(self.bc, "_safe_close_stream")
        patches = self._common_patches(fake_sd) + [scs_patch]
        started = [p.start() for p in patches]
        scs = started[patches.index(scs_patch)]
        try:
            out = self.bc.record_speech(timeout=1.0)
        finally:
            for p in patches:
                p.stop()
        self.assertIsNone(out)
        scs.assert_called_once()


@requires_monolith
class PlayWithLipsyncDefensiveHandlerTests(MonolithGlobalsTestCase):
    """The single-line except handlers inside play_with_lipsync's closures and
    teardown. Threads run synchronously via _ImmediateThread; the barge-watch
    closure is given an already-set interrupt flag so it returns at once."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_flag = list(self.bc._tts_playback_active)
        self._saved_barge = self.bc._barge_in_interrupted

    def tearDown(self):
        self.bc._tts_playback_active[:] = self._saved_flag
        self.bc._barge_in_interrupted = self._saved_barge

    def _base(self, fake_sd, fake_layer, ducker, thread_cls=_ImmediateThread):
        return [
            mock.patch.object(self.bc, "sd", fake_sd),
            mock.patch.object(self.bc, "_tts_layer", fake_layer),
            mock.patch.object(self.bc, "_audio_ducker", ducker),
            mock.patch.object(self.bc, "get_output_device", return_value=1),
            mock.patch.object(self.bc, "_write_hud_state"),
            mock.patch.object(self.bc, "_feed_playback_reference"),
            mock.patch.object(self.bc.threading, "Thread", thread_cls),
            mock.patch.object(self.bc.time, "sleep"),
            mock.patch.object(self.bc, "_safe_close_stream"),
        ]

    def test_is_muted_probe_exception_defaults_unmuted(self):
        # 6772-6773: _tts_layer.is_muted() raises -> _muted stays False and
        # playback proceeds down the no-robot arm. No barge listener.
        fake_sd = mock.Mock()
        fake_layer = mock.Mock()
        fake_layer.is_muted.side_effect = RuntimeError("mute probe boom")
        ducker = mock.Mock()
        audio = np.zeros(48, dtype=np.float32)
        patches = self._base(fake_sd, fake_layer, ducker) + [
            mock.patch.object(self.bc, "BARGE_IN_ENABLED", False),
            mock.patch.object(self.bc, "ROBOT_ENABLED", False),
            mock.patch.object(self.bc, "is_using_headset", return_value=False),
        ]
        for p in patches:
            p.start()
        try:
            self.bc.play_with_lipsync(audio, 24000)   # must not raise
        finally:
            for p in patches:
                p.stop()
        fake_sd.play.assert_called_once()
        self.assertFalse(self.bc._tts_playback_active[0])

    def test_sd_wait_exception_is_logged_not_raised(self):
        # 6792-6793: sd.wait() inside _safe_wait raises -> logged, done-event
        # still set in finally, playback completes. No barge listener.
        fake_sd = mock.Mock()
        fake_sd.wait.side_effect = RuntimeError("wait boom")
        fake_layer = mock.Mock()
        fake_layer.is_muted.return_value = False
        ducker = mock.Mock()
        audio = np.zeros(48, dtype=np.float32)
        patches = self._base(fake_sd, fake_layer, ducker) + [
            mock.patch.object(self.bc, "BARGE_IN_ENABLED", False),
            mock.patch.object(self.bc, "ROBOT_ENABLED", False),
            mock.patch.object(self.bc, "is_using_headset", return_value=False),
        ]
        for p in patches:
            p.start()
        try:
            self.bc.play_with_lipsync(audio, 24000)
        finally:
            for p in patches:
                p.stop()
        fake_sd.wait.assert_called()
        self.assertFalse(self.bc._tts_playback_active[0])

    def test_barge_watch_sd_stop_exception_swallowed(self):
        # 6722-6723: the barge-watch closure sees the interrupt flag and calls
        # sd.stop(), which raises -> swallowed. Flag pre-set so the closure
        # returns immediately under _ImmediateThread.
        fake_sd = mock.Mock()
        fake_sd.stop.side_effect = RuntimeError("stop boom")
        fake_layer = mock.Mock()
        fake_layer.is_muted.return_value = False
        ducker = mock.Mock()
        barge_stream = mock.Mock()
        self.bc._barge_in_interrupted = True
        audio = np.zeros(48, dtype=np.float32)
        patches = self._base(fake_sd, fake_layer, ducker) + [
            mock.patch.object(self.bc, "BARGE_IN_ENABLED", True),
            mock.patch.object(self.bc, "ROBOT_ENABLED", False),
            mock.patch.object(self.bc, "is_using_headset", return_value=True),
            mock.patch.object(self.bc, "_start_barge_in_listener",
                              return_value=barge_stream),
        ]
        for p in patches:
            p.start()
        try:
            self.bc.play_with_lipsync(audio, 24000)   # must not raise
        finally:
            for p in patches:
                p.stop()
        fake_sd.stop.assert_called()
        self.assertFalse(self.bc._tts_playback_active[0])

    def test_ducker_restore_exception_swallowed(self):
        # 6872-6873: _audio_ducker.restore() in the finally raises -> swallowed.
        fake_sd = mock.Mock()
        fake_layer = mock.Mock()
        fake_layer.is_muted.return_value = False
        ducker = mock.Mock()
        ducker.restore.side_effect = RuntimeError("restore boom")
        audio = np.zeros(48, dtype=np.float32)
        patches = self._base(fake_sd, fake_layer, ducker) + [
            mock.patch.object(self.bc, "BARGE_IN_ENABLED", False),
            mock.patch.object(self.bc, "ROBOT_ENABLED", False),
            mock.patch.object(self.bc, "is_using_headset", return_value=False),
        ]
        for p in patches:
            p.start()
        try:
            self.bc.play_with_lipsync(audio, 24000)   # must not raise
        finally:
            for p in patches:
                p.stop()
        ducker.restore.assert_called_once()
        self.assertFalse(self.bc._tts_playback_active[0])

    def test_robot_sd_wait_exception_is_logged_not_raised(self):
        # 6836-6837: in the robot arm, sd.wait() inside _safe_wait_robot raises
        # -> logged, done-event still set, playback completes. No barge listener.
        fake_sd = mock.Mock()
        fake_sd.wait.side_effect = RuntimeError("robot wait boom")
        fake_layer = mock.Mock()
        fake_layer.is_muted.return_value = False
        ducker = mock.Mock()
        audio = np.full(96, 0.4, dtype=np.float32)
        patches = self._base(fake_sd, fake_layer, ducker) + [
            mock.patch.object(self.bc, "BARGE_IN_ENABLED", False),
            mock.patch.object(self.bc, "ROBOT_ENABLED", True),
            mock.patch.object(self.bc, "is_using_headset", return_value=False),
            mock.patch.object(self.bc, "send"),
        ]
        for p in patches:
            p.start()
        try:
            self.bc.play_with_lipsync(audio, 24000)   # must not raise
        finally:
            for p in patches:
                p.stop()
        fake_sd.wait.assert_called()
        self.assertFalse(self.bc._tts_playback_active[0])

    def test_thread_join_exceptions_swallowed(self):
        # 6855-6856 (amp_thread.join) + 6863-6864 (barge_watch_thread.join):
        # both joins raise in the finally -> swallowed. Barge path so the watch
        # thread exists; interrupt flag pre-set so the closure exits at once.
        class _JoinRaisingThread(_ImmediateThread):
            def join(self, timeout=None):
                raise RuntimeError("join boom")

        fake_sd = mock.Mock()
        fake_layer = mock.Mock()
        fake_layer.is_muted.return_value = False
        ducker = mock.Mock()
        barge_stream = mock.Mock()
        self.bc._barge_in_interrupted = True
        audio = np.zeros(48, dtype=np.float32)
        patches = self._base(fake_sd, fake_layer, ducker,
                             thread_cls=_JoinRaisingThread) + [
            mock.patch.object(self.bc, "BARGE_IN_ENABLED", True),
            mock.patch.object(self.bc, "ROBOT_ENABLED", False),
            mock.patch.object(self.bc, "is_using_headset", return_value=True),
            mock.patch.object(self.bc, "_start_barge_in_listener",
                              return_value=barge_stream),
        ]
        for p in patches:
            p.start()
        try:
            self.bc.play_with_lipsync(audio, 24000)   # must not raise
        finally:
            for p in patches:
                p.stop()
        self.assertFalse(self.bc._tts_playback_active[0])


# ===========================================================================
# LOCAL-32B UPGRADE + RESILIENCE (feat/local-32b-upgrade)
# ===========================================================================
# Covers: the 32B-first preference chain, model-aware num_ctx, the honest
# local→cloud→honest-message fallback for a LOCAL-routed turn, and the
# best-effort Smart-App-Control event-log probe (parse / negatives / rate-
# limit). Everything is mocked — no real Ollama, no network, no PowerShell.
# ===========================================================================
@requires_monolith
class LocalNumCtxTests(MonolithGlobalsTestCase):
    """_local_num_ctx — the 32B needs the tighter 12k window to stay 100 % on
    the 3090; 14B/8B keep 16k. (See the measured 12k=49tps vs 16k=spill fact.)"""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_32b_tag_gets_12288(self):
        self.assertEqual(
            self.bc._local_num_ctx("qwen2.5:32b-instruct-q4_K_M"), 12288)

    def test_32b_case_insensitive(self):
        self.assertEqual(self.bc._local_num_ctx("Qwen2.5:32B-Instruct"), 12288)

    def test_other_large_tags_also_12288(self):
        for tag in ("llama3.1:70b", "qwen:72b-chat", "yi:34b", "foo:65b"):
            self.assertEqual(self.bc._local_num_ctx(tag), 12288, tag)

    def test_production_qwen3_30b_moe_gets_12288(self):
        # P0-1: the PRODUCTION tag is qwen3:30b-a3b… — a 30B MoE that was
        # falling through to 16384 (the literal list only had 32b/34b/…),
        # running ~40 % slower with a CPU spill every turn. It must now get
        # the tight 12k window like the rest of the 30B-class.
        for tag in ("qwen3:30b-a3b-q4_K_M", "qwen3:30b", "Qwen3:30B-A3B"):
            self.assertEqual(self.bc._local_num_ctx(tag), 12288, tag)

    def test_moe_active_param_suffix_not_misread_as_small(self):
        # The `a3b` active-param suffix must NOT trick the param-parse into
        # reading 3B and keeping 16k — the architecturally-relevant size is the
        # 30B total. (Regression guard for the negative-lookbehind in the parse.)
        self.assertEqual(self.bc._local_num_ctx("qwen3:30b-a3b"), 12288)

    def test_future_large_tag_param_parse_12288(self):
        # General ≥30B param-parse: a future tag with no literal in the list
        # (e.g. a 40B / 110B / 235B) still gets the tight window.
        for tag in ("foo:40b-instruct", "bar:110b", "baz:235b-a22b"):
            self.assertEqual(self.bc._local_num_ctx(tag), 12288, tag)

    def test_14b_keeps_16384(self):
        self.assertEqual(
            self.bc._local_num_ctx("qwen2.5:14b-instruct-q5_K_M"), 16384)

    def test_8b_keeps_16384(self):
        self.assertEqual(
            self.bc._local_num_ctx("llama3.1:8b-instruct-q5_K_M"), 16384)

    def test_empty_or_none_defaults_16384(self):
        self.assertEqual(self.bc._local_num_ctx(""), 16384)
        self.assertEqual(self.bc._local_num_ctx(None), 16384)


@requires_monolith
class GetLocalLlmModel32bTests(MonolithGlobalsTestCase):
    """The 32B is now first in the preference chain; it must be picked when
    installed, fall cleanly to the 14B when the 32B is absent, and still honour
    JARVIS_LOCAL_LLM_MODEL."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_cache = list(self.bc._RESOLVED_LOCAL_LLM_MODEL)
        self.bc._RESOLVED_LOCAL_LLM_MODEL[0] = None
        self._saved_env = os.environ.get("JARVIS_LOCAL_LLM_MODEL")
        os.environ.pop("JARVIS_LOCAL_LLM_MODEL", None)

    def tearDown(self):
        self.bc._RESOLVED_LOCAL_LLM_MODEL[:] = self._saved_cache
        if self._saved_env is None:
            os.environ.pop("JARVIS_LOCAL_LLM_MODEL", None)
        else:
            os.environ["JARVIS_LOCAL_LLM_MODEL"] = self._saved_env

    def test_chain_lists_32b_first(self):
        self.assertEqual(self.bc._LOCAL_LLM_PREFERENCE[0],
                         "qwen2.5:32b-instruct-q4_K_M")

    def test_picks_32b_when_tags_list_it(self):
        fake_req = mock.Mock()
        fake_req.get.return_value = _FakeResp(ok=True, json_data={"models": [
            {"name": "qwen2.5:32b-instruct-q4_K_M"},
            {"name": "qwen2.5:14b-instruct-q5_K_M"},
            {"name": "llama3.1:8b-instruct-q5_K_M"},
        ]})
        with mock.patch.object(self.bc, "requests", fake_req), \
                mock.patch.object(self.bc, "_log_gpu_state"):
            out = self.bc._get_local_llm_model()
        self.assertEqual(out, "qwen2.5:32b-instruct-q4_K_M")

    def test_falls_to_14b_when_only_14b_present(self):
        # 32B not installed → cleanly drops to the next chain entry (14B).
        fake_req = mock.Mock()
        fake_req.get.return_value = _FakeResp(ok=True, json_data={"models": [
            {"name": "qwen2.5:14b-instruct-q5_K_M"},
            {"name": "llama3.1:8b-instruct-q5_K_M"},
        ]})
        with mock.patch.object(self.bc, "requests", fake_req), \
                mock.patch.object(self.bc, "_log_gpu_state"):
            out = self.bc._get_local_llm_model()
        self.assertEqual(out, "qwen2.5:14b-instruct-q5_K_M")

    def test_env_override_beats_installed_32b(self):
        os.environ["JARVIS_LOCAL_LLM_MODEL"] = "  my:custom  "
        fake_req = mock.Mock()
        fake_req.get.return_value = _FakeResp(ok=True, json_data={"models": [
            {"name": "qwen2.5:32b-instruct-q4_K_M"}]})
        with mock.patch.object(self.bc, "requests", fake_req), \
                mock.patch.object(self.bc, "_log_gpu_state"):
            out = self.bc._get_local_llm_model()
        self.assertEqual(out, "my:custom")


@requires_monolith
class CallLocalLlmNumCtxAndTuningTests(MonolithGlobalsTestCase):
    """_call_local_llm wires the model-aware num_ctx + the light tuning knobs
    (repeat_penalty / top_k) into the /api/chat options, and a generate read-
    timeout returns None (so the caller can fall back) rather than raising."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _arm(self, fake_req, model):
        return [
            mock.patch.object(self.bc, "LOCAL_LLM_FALLBACK", True),
            mock.patch.object(self.bc, "_ollama_alive", return_value=True),
            mock.patch.object(self.bc, "_get_local_llm_model", return_value=model),
            mock.patch.object(self.bc, "_ollama_has_model", return_value=True),
            mock.patch.object(self.bc, "requests", fake_req),
        ]

    def test_32b_model_sends_12288_and_tuning(self):
        captured = {}
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None):
            captured["opts"] = json["options"]
            captured["timeout"] = timeout
            return _FakeResp(ok=True, json_data={"message": {"content": "ok"}})
        fake_req.post.side_effect = _post
        patches = self._arm(fake_req, "qwen2.5:32b-instruct-q4_K_M")
        for p in patches:
            p.start()
        try:
            self.bc._call_local_llm("SYS", [{"role": "user", "content": "hi"}])
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(captured["opts"]["num_ctx"], 12288)
        self.assertEqual(captured["opts"]["repeat_penalty"], 1.05)
        self.assertEqual(captured["opts"]["top_k"], 40)
        # still the conservative sampling we kept
        self.assertEqual(captured["opts"]["temperature"], 0.4)
        self.assertEqual(captured["opts"]["top_p"], 0.9)
        # (connect, read) tuple so a blocked runner fails fast, not at 120 s
        self.assertEqual(captured["timeout"], self.bc._LOCAL_GENERATE_TIMEOUT)
        self.assertIsInstance(self.bc._LOCAL_GENERATE_TIMEOUT, tuple)

    def test_14b_model_sends_16384(self):
        captured = {}
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None):
            captured["opts"] = json["options"]
            return _FakeResp(ok=True, json_data={"message": {"content": "ok"}})
        fake_req.post.side_effect = _post
        patches = self._arm(fake_req, "qwen2.5:14b-instruct-q5_K_M")
        for p in patches:
            p.start()
        try:
            self.bc._call_local_llm("SYS", [])
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(captured["opts"]["num_ctx"], 16384)

    def test_generate_timeout_returns_none(self):
        # The SAC-blocked-runner signature: /api/tags is fine, but the generate
        # hangs and trips the read timeout. _call_local_llm must return None
        # (NOT raise) so the caller treats local as unavailable this turn.
        fake_req = mock.Mock()
        fake_req.Timeout = self.bc.requests.Timeout
        fake_req.post.side_effect = self.bc.requests.Timeout("read timed out")
        patches = self._arm(fake_req, "qwen2.5:32b-instruct-q4_K_M")
        for p in patches:
            p.start()
        try:
            out = self.bc._call_local_llm("SYS", [])
        finally:
            for p in patches:
                p.stop()
        self.assertIsNone(out)


@requires_monolith
class OllamaChatBoundedTests(MonolithGlobalsTestCase):
    """P1-2: the two hot-path ollama.chat calls (_call_llm + get_followup_response)
    must be wall-clock bounded so a wedged runner can't deaf-loop JARVIS forever.
    _ollama_chat_bounded routes through ollama.Client(timeout=…) and raises on a
    hang so each caller's existing `except` degrades (cloud/local fallback)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_builds_client_with_timeout_and_forwards_call(self):
        captured = {}
        fake_client = mock.Mock()
        fake_client.chat.return_value = {"message": {"content": "ok"}}

        def _ctor(*args, **kwargs):
            captured["ctor_kwargs"] = kwargs
            return fake_client
        fake_ollama = mock.Mock()
        fake_ollama.Client.side_effect = _ctor
        with mock.patch.dict(sys.modules, {"ollama": fake_ollama}):
            out = self.bc._ollama_chat_bounded(
                "m:tag", [{"role": "user", "content": "hi"}])
        # A real wall-clock timeout was passed to the client (NOT None).
        self.assertEqual(captured["ctor_kwargs"].get("timeout"),
                         self.bc._OLLAMA_CHAT_TIMEOUT_S)
        self.assertIsInstance(self.bc._OLLAMA_CHAT_TIMEOUT_S, (int, float))
        self.assertGreater(self.bc._OLLAMA_CHAT_TIMEOUT_S, 0)
        # model + messages forwarded through to the client's .chat().
        _, ckw = fake_client.chat.call_args
        self.assertEqual(ckw["model"], "m:tag")
        self.assertEqual(ckw["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(out, {"message": {"content": "ok"}})

    def test_timeout_propagates_so_caller_can_fall_back(self):
        # A hung runner -> the client raises -> _ollama_chat_bounded must RAISE
        # (not swallow), so the caller's except fires its fallback. We use a
        # stand-in exception that mimics httpx.TimeoutException.
        class _Timeout(Exception):
            pass
        fake_client = mock.Mock()
        fake_client.chat.side_effect = _Timeout("read timed out")
        fake_ollama = mock.Mock()
        fake_ollama.Client.return_value = fake_client
        with mock.patch.dict(sys.modules, {"ollama": fake_ollama}):
            with self.assertRaises(_Timeout):
                self.bc._ollama_chat_bounded("m:tag", [])

    def test_call_llm_ollama_timeout_degrades_not_hangs(self):
        # End-to-end: a timeout in the bounded helper must surface the honest
        # "local model isn't responding" line, never propagate or hang.
        with mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
                mock.patch.object(self.bc, "OLLAMA_MODEL", "m"), \
                mock.patch.object(self.bc, "detect_tone", return_value=None), \
                mock.patch.object(self.bc, "route_voice_emotion",
                                  return_value={"mood": "casual", "addendum": ""}), \
                mock.patch.object(self.bc, "_voice_mood_response", None), \
                mock.patch.object(self.bc, "_emotion_tracker", None), \
                mock.patch.object(self.bc, "_trim_conversation_history"), \
                mock.patch.object(self.bc, "_system_prompt", "SYS"), \
                mock.patch.object(self.bc, "_ollama_chat_bounded",
                                  side_effect=TimeoutError("wedged")), \
                mock.patch.object(self.bc, "_mcu_phrases", mock.Mock(
                    **{"detect_phrases_in_reply.return_value": {}})):
            saved = list(self.bc.conversation_history)
            self.bc.conversation_history.clear()
            try:
                reply = self.bc._call_llm("hi")
            finally:
                self.bc.conversation_history[:] = saved
        self.assertIn("local model isn't responding", reply)

    def test_followup_ollama_timeout_falls_back_to_local(self):
        # get_followup_response: a bounded-ollama timeout must drop into the
        # _call_local_llm fallback (the existing except arm), not hang/raise.
        with mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
                mock.patch.object(self.bc, "OLLAMA_MODEL", "m"), \
                mock.patch.object(self.bc, "_last_voice_route",
                                  [{"addendum": ""}]), \
                mock.patch.object(self.bc, "_last_user_tone", [None]), \
                mock.patch.object(self.bc, "_system_prompt", "SYS"), \
                mock.patch.object(self.bc, "_ollama_chat_bounded",
                                  side_effect=TimeoutError("wedged")), \
                mock.patch.object(self.bc, "_call_local_llm",
                                  return_value="local saved the chain") as loc:
            out = self.bc.get_followup_response([("get_time", "noon")])
        self.assertEqual(out, "local saved the chain")
        loc.assert_called_once()


@requires_monolith
class LocalThenCloudOrHonestTests(MonolithGlobalsTestCase):
    """The LOCAL-routed resilience path: local primary → Claude fallback when
    reachable → honest message when BOTH are down (never a fabricated answer)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_local_success_returned_and_tag_stripped(self):
        with mock.patch.object(self.bc, "_call_local_llm",
                               return_value="[local] done sir"), \
                mock.patch.object(self.bc, "_claude_oneshot") as cloud:
            out = self.bc._local_then_cloud_or_honest("SYS", [])
        self.assertEqual(out, "done sir")
        cloud.assert_not_called()   # local answered → cloud never consulted

    def test_local_down_falls_back_to_cloud(self):
        with mock.patch.object(self.bc, "_call_local_llm", return_value=None), \
                mock.patch.object(self.bc, "_claude_oneshot",
                                  return_value="cloud reply sir") as cloud, \
                mock.patch.object(self.bc, "_sac_blocked_local_recently",
                                  return_value=False):
            out = self.bc._local_then_cloud_or_honest("SYS", [{"role": "user",
                                                               "content": "q"}])
        self.assertEqual(out, "cloud reply sir")
        cloud.assert_called_once()

    def test_both_down_returns_honest_generic_message(self):
        with mock.patch.object(self.bc, "_call_local_llm", return_value=None), \
                mock.patch.object(self.bc, "_claude_oneshot", return_value=None), \
                mock.patch.object(self.bc, "_sac_blocked_local_recently",
                                  return_value=False):
            out = self.bc._local_then_cloud_or_honest("SYS", [])
        low = out.lower()
        # honest: names BOTH the local model and the cloud being unreachable
        self.assertIn("local model", low)
        self.assertIn("cloud", low)
        # NOT a fabricated answer to the user's question
        self.assertNotIn("[local]", out)

    def test_both_down_sac_blocked_message_is_specific(self):
        with mock.patch.object(self.bc, "_call_local_llm", return_value=None), \
                mock.patch.object(self.bc, "_claude_oneshot", return_value=None), \
                mock.patch.object(self.bc, "_sac_blocked_local_recently",
                                  return_value=True):
            out = self.bc._local_then_cloud_or_honest("SYS", [])
        self.assertIn("Smart App Control", out)


@requires_monolith
class ClaudeReachableTests(MonolithGlobalsTestCase):
    """_claude_reachable gates whether to ATTEMPT a Claude fallback: Claude
    backend + a key present → True; Ollama backend or keyless → False."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_key = os.environ.get("ANTHROPIC_API_KEY")

    def tearDown(self):
        if self._saved_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = self._saved_key

    def test_true_when_claude_backend_and_key(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"):
            self.assertTrue(self.bc._claude_reachable())

    def test_false_when_no_key(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"):
            self.assertFalse(self.bc._claude_reachable())

    def test_false_when_ollama_backend(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        with mock.patch.object(self.bc, "AI_BACKEND", "ollama"):
            self.assertFalse(self.bc._claude_reachable())


@requires_monolith
class CallLlmLocalRouteFallbackTests(MonolithGlobalsTestCase):
    """_call_llm under MODEL_ROUTING['chat']=='local': a local failure must
    fall back to Claude when reachable, and surface the honest both-down line
    when neither works — instead of the old static '(local unavailable)'."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_hist = list(self.bc.conversation_history)
        self.bc.conversation_history.clear()
        self._patches = [
            mock.patch.object(self.bc, "detect_tone", return_value=None),
            mock.patch.object(self.bc, "route_voice_emotion",
                              return_value={"mood": "casual", "addendum": ""}),
            mock.patch.object(self.bc, "_emotion_tracker", None),
            mock.patch.object(self.bc, "_voice_mood_response", None),
            mock.patch.object(self.bc, "_trim_conversation_history"),
            mock.patch.object(self.bc, "_system_prompt", "SYS"),
        ]
        for p in self._patches:
            p.start()
        import core.config as _cfg
        self._cfg = _cfg
        self._saved_route = dict(_cfg.MODEL_ROUTING)
        _cfg.MODEL_ROUTING = dict(_cfg.MODEL_ROUTING, chat="local")

    def tearDown(self):
        self._cfg.MODEL_ROUTING = self._saved_route
        for p in self._patches:
            p.stop()
        self.bc.conversation_history[:] = self._saved_hist

    def _phrases(self):
        m = mock.Mock()
        m.detect_phrases_in_reply.return_value = {}
        return m

    def test_local_route_uses_local_when_it_answers(self):
        with mock.patch.object(self.bc, "_call_local_llm",
                               return_value="local answer sir"), \
                mock.patch.object(self.bc, "_claude_oneshot") as cloud, \
                mock.patch.object(self.bc, "_mcu_phrases", self._phrases()):
            reply = self.bc._call_llm("hello")
        self.assertEqual(reply, "local answer sir")
        cloud.assert_not_called()
        self.assertEqual(self.bc.conversation_history[-1]["content"],
                         "local answer sir")

    def test_local_route_falls_back_to_cloud_on_local_failure(self):
        # local down (returns None) → Claude reachable → cloud answers.
        with mock.patch.object(self.bc, "_call_local_llm", return_value=None), \
                mock.patch.object(self.bc, "_claude_oneshot",
                                  return_value="from the cloud sir"), \
                mock.patch.object(self.bc, "_sac_blocked_local_recently",
                                  return_value=False), \
                mock.patch.object(self.bc, "_mcu_phrases", self._phrases()):
            reply = self.bc._call_llm("hello")
        self.assertEqual(reply, "from the cloud sir")

    def test_local_route_honest_when_both_down(self):
        with mock.patch.object(self.bc, "_call_local_llm", return_value=None), \
                mock.patch.object(self.bc, "_claude_oneshot", return_value=None), \
                mock.patch.object(self.bc, "_sac_blocked_local_recently",
                                  return_value=False), \
                mock.patch.object(self.bc, "_mcu_phrases", self._phrases()):
            reply = self.bc._call_llm("hello")
        low = reply.lower()
        self.assertIn("local model", low)
        self.assertIn("cloud", low)


@requires_monolith
class SacBlockedLocalRecentlyTests(MonolithGlobalsTestCase):
    """_sac_blocked_local_recently — best-effort Windows event-log probe:
    parses a blocked 3077 event as True, returns False on empty / error /
    non-Windows, and is rate-limited (second call inside the window doesn't
    re-fork PowerShell)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        # Reset the module-level rate-limit cache before each test.
        self._saved_cache = list(self.bc._SAC_CHECK_CACHE)
        self.bc._SAC_CHECK_CACHE[0] = 0.0
        self.bc._SAC_CHECK_CACHE[1] = False

    def tearDown(self):
        self.bc._SAC_CHECK_CACHE[:] = self._saved_cache

    def test_parses_blocked_event_as_true(self):
        fake_sp = mock.Mock()
        fake_sp.run.return_value = mock.Mock(
            stdout="Code Integrity blocked C:\\Users\\x\\AppData\\...\\ollama "
                   "runner loading ggml.dll", stderr="")
        with mock.patch.object(self.bc, "subprocess", fake_sp), \
                mock.patch("platform.system", return_value="Windows"):
            self.assertTrue(self.bc._sac_blocked_local_recently())

    def test_empty_output_is_false(self):
        fake_sp = mock.Mock()
        fake_sp.run.return_value = mock.Mock(stdout="", stderr="")
        with mock.patch.object(self.bc, "subprocess", fake_sp), \
                mock.patch("platform.system", return_value="Windows"):
            self.assertFalse(self.bc._sac_blocked_local_recently())

    def test_subprocess_error_is_false(self):
        fake_sp = mock.Mock()
        fake_sp.run.side_effect = OSError("powershell missing")
        with mock.patch.object(self.bc, "subprocess", fake_sp), \
                mock.patch("platform.system", return_value="Windows"):
            self.assertFalse(self.bc._sac_blocked_local_recently())

    def test_non_windows_is_false_without_subprocess(self):
        fake_sp = mock.Mock()
        with mock.patch.object(self.bc, "subprocess", fake_sp), \
                mock.patch("platform.system", return_value="Linux"):
            self.assertFalse(self.bc._sac_blocked_local_recently())
        fake_sp.run.assert_not_called()   # never forks off-Windows

    def test_rate_limited_second_call_skips_subprocess(self):
        fake_sp = mock.Mock()
        fake_sp.run.return_value = mock.Mock(
            stdout="ollama ggml.dll blocked", stderr="")
        with mock.patch.object(self.bc, "subprocess", fake_sp), \
                mock.patch("platform.system", return_value="Windows"):
            first = self.bc._sac_blocked_local_recently()
            second = self.bc._sac_blocked_local_recently()
        self.assertTrue(first)
        self.assertTrue(second)            # cached result reused
        fake_sp.run.assert_called_once()   # NOT re-forked within the TTL window


if __name__ == "__main__":
    unittest.main()
