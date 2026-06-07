"""Logic tests for skills/standby_audio_detect.py.

This skill adds music/lyric detection that finishes the auto-standby feature.
Tests cover the deterministic, numpy-backed core:
  • the pure-FFT spectral classifier (_classify_chunk) on synthesized tone vs
    noise vs near-silence,
  • the rhyme-density heuristic,
  • _looks_like_lyrics combining onset + rhyme + whisper markers,
  • should_refuse_wake gating (clear 'jarvis' vs lyric near-miss while music),
  • the feed_audio → sustained-music state machine and SUSTAINED_HOLD gate,
  • staleness reset,
  • music_state_summary phrasing,
  • the audio_music_status action.

All audio is synthesized in-memory; whisper / librosa / the background loop
are never invoked (the loop thread is neutered by the harness).
"""
from __future__ import annotations

import contextlib
import sys
import types
import unittest
from unittest import mock

import numpy as np

from tests._skill_harness import load_skill_isolated

_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules (restoring prior
    state — including absence — on exit). Dotted-name leaves are ALSO set as an
    attribute on the already-imported parent package. Mirrors the helper in
    test_self_diagnostic.py so this file stays self-contained and isolated.

    Per the CI contract: sounddevice / librosa / whisper / faster_whisper /
    torch are NOT installed on CI, so they are injected as fakes here; numpy IS
    on CI and stays real (used for the synthetic signals)."""
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


def _tone(freq=120.0, secs=1.0, sr=16000, amp=0.5):
    t = np.arange(int(secs * sr)) / sr
    # Bass tone + harmonic so flatness is low and bass-band energy is high.
    sig = amp * (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2 * freq * t))
    return sig.astype(np.float32)


def _noise(secs=1.0, sr=16000, amp=0.5):
    rng = np.random.default_rng(42)
    return (amp * rng.standard_normal(int(secs * sr))).astype(np.float32)


class SpectralClassifierTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("standby_audio_detect")

    def test_classify_tone_as_musical(self):
        self.assertTrue(self.mod._classify_chunk(_tone(), 16000))

    def test_classify_white_noise_not_musical(self):
        # White noise is spectrally flat → not flagged as tonal/musical.
        self.assertFalse(self.mod._classify_chunk(_noise(), 16000))

    def test_classify_near_silence_not_musical(self):
        quiet = (np.zeros(16000) + 1e-4).astype(np.float32)
        self.assertFalse(self.mod._classify_chunk(quiet, 16000))

    def test_classify_too_short_not_musical(self):
        # < 0.25s of audio → bail out False.
        self.assertFalse(self.mod._classify_chunk(_tone(secs=0.1), 16000))

    def test_classify_handles_stereo(self):
        mono = _tone()
        stereo = np.stack([mono, mono], axis=1)
        self.assertTrue(self.mod._classify_chunk(stereo, 16000))

    def test_classify_none_audio_false(self):
        self.assertFalse(self.mod._classify_chunk(None, 16000))


class LyricHeuristicTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("standby_audio_detect")

    # ── _rhyme_density ───────────────────────────────────────────────────
    def test_rhyme_density_high_for_rhyming_lines(self):
        # Each adjacent pair shares the last-2 suffix → density 1.0.
        text = "the cat sat splat that"
        self.assertGreater(self.mod._rhyme_density(text), 0.5)

    def test_rhyme_density_low_for_prose(self):
        text = "please schedule the quarterly budget review meeting"
        self.assertLess(self.mod._rhyme_density(text), 0.5)

    def test_rhyme_density_zero_for_short(self):
        self.assertEqual(self.mod._rhyme_density("hi there"), 0.0)
        self.assertEqual(self.mod._rhyme_density(""), 0.0)

    # ── _looks_like_lyrics ───────────────────────────────────────────────
    def test_looks_like_lyrics_whisper_marker(self):
        # An explicit [music] marker short-circuits to True.
        self.assertTrue(self.mod._looks_like_lyrics("[music] la la la", onset=0.0))

    def test_looks_like_lyrics_requires_onset_and_rhyme(self):
        # Rhyme-dense AND onset above the min → lyric.
        self.assertTrue(self.mod._looks_like_lyrics("cat sat splat that mat", onset=0.9))
        # Rhyme-dense but onset too low → not a lyric (could be a poem you typed).
        self.assertFalse(self.mod._looks_like_lyrics("cat sat splat that mat", onset=0.0))

    def test_looks_like_lyrics_empty_and_silent(self):
        self.assertFalse(self.mod._looks_like_lyrics("", onset=0.0))

    def test_looks_like_lyrics_records_last_score(self):
        # The non-marker path updates _loop_last_score for diagnostics.
        self.mod._looks_like_lyrics("cat sat splat that mat", onset=0.9)
        self.assertEqual(self.mod._loop_last_score["text"],
                         "cat sat splat that mat")
        self.assertGreater(self.mod._loop_last_score["onset"], 0.0)

    def test_looks_like_lyrics_high_onset_no_rhyme(self):
        # Loud but non-rhyming prose → not a lyric (musical_audio AND
        # rhyme_dense both required).
        self.assertFalse(self.mod._looks_like_lyrics(
            "please bring the quarterly numbers tomorrow", onset=0.95))

    def test_rhyme_density_strips_punctuation_and_short_words(self):
        # Words < 3 chars and non-alpha are filtered before the suffix check.
        self.assertEqual(self.mod._rhyme_density("a an the of"), 0.0)


class WakeGatingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("standby_audio_detect")

    def test_no_refuse_when_no_music(self):
        with mock.patch.object(self.mod, "is_music_currently_playing", return_value=False):
            self.assertFalse(self.mod.should_refuse_wake("some lyric words here"))

    def test_refuse_lyric_near_miss_during_music(self):
        with mock.patch.object(self.mod, "is_music_currently_playing", return_value=True):
            # Long mid-sentence phrase while music plays → treated as lyric.
            self.assertTrue(self.mod.should_refuse_wake("and the radio kept playing on"))

    def test_allow_clear_wake_during_music(self):
        with mock.patch.object(self.mod, "is_music_currently_playing", return_value=True):
            self.assertFalse(self.mod.should_refuse_wake("jarvis"))
            self.assertFalse(self.mod.should_refuse_wake("hey jarvis"))

    def test_refuse_empty_text_during_music(self):
        with mock.patch.object(self.mod, "is_music_currently_playing", return_value=True):
            self.assertTrue(self.mod.should_refuse_wake(""))

    def test_refuse_whitespace_only_text_during_music(self):
        # Strips to empty word list → refuse (lyric near-miss assumption).
        with mock.patch.object(self.mod, "is_music_currently_playing",
                               return_value=True):
            self.assertTrue(self.mod.should_refuse_wake("   "))

    def test_allow_okay_jarvis_prefix(self):
        with mock.patch.object(self.mod, "is_music_currently_playing",
                               return_value=True):
            self.assertFalse(self.mod.should_refuse_wake("okay jarvis"))

    def test_refuse_short_phrase_without_wake_word(self):
        # <= NEAR_MISS_MAX_WORDS but first word isn't a wake word → refuse.
        with mock.patch.object(self.mod, "is_music_currently_playing",
                               return_value=True):
            self.assertTrue(self.mod.should_refuse_wake("baby tonight"))


class FeedAudioStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("standby_audio_detect")
        # Reset rolling state between tests.
        self.mod._classifications.clear()
        self.mod._music_active[0] = False
        self.mod._music_since[0] = 0.0
        self.mod._last_feed_at[0] = 0.0
        self.mod._total_chunks_seen[0] = 0
        self.mod._total_music_chunks[0] = 0

    def test_sustained_music_activates_after_coverage(self):
        # Feed several musical chunks (each classifies True) past the 5s
        # coverage floor → _music_active flips on.
        tone = _tone(secs=1.0)
        for _ in range(7):
            self.mod.feed_audio(tone, 16000)
        self.assertTrue(self.mod._music_active[0])

    def test_is_music_playing_requires_sustained_hold(self):
        tone = _tone(secs=1.0)
        for _ in range(7):
            self.mod.feed_audio(tone, 16000)
        # Active, but _music_since is "now" → hold (15s) not yet satisfied.
        self.assertFalse(self.mod.is_music_currently_playing())
        # Backdate the activation past the hold window.
        self.mod._music_since[0] = self.mod._music_since[0] - (self.mod.SUSTAINED_HOLD_SECONDS + 1)
        self.assertTrue(self.mod.is_music_currently_playing())

    def test_noise_does_not_activate(self):
        for _ in range(7):
            self.mod.feed_audio(_noise(secs=1.0), 16000)
        self.assertFalse(self.mod._music_active[0])

    def test_stale_state_resets(self):
        self.mod._music_active[0] = True
        self.mod._last_feed_at[0] = 0.0  # ancient → > MUSIC_TIMEOUT_SECONDS ago
        self.assertFalse(self.mod.is_music_currently_playing())
        self.assertFalse(self.mod._music_active[0])

    def test_feed_none_is_safe(self):
        # None audio must not raise and must not change counters.
        before = self.mod._total_chunks_seen[0]
        self.mod.feed_audio(None, 16000)
        self.assertEqual(self.mod._total_chunks_seen[0], before)

    def test_feed_classify_exception_is_swallowed(self):
        # If the classifier raises, feed_audio returns early without recording.
        before = self.mod._total_chunks_seen[0]
        with mock.patch.object(self.mod, "_classify_chunk",
                               side_effect=RuntimeError("fft boom")):
            self.mod.feed_audio(_tone(secs=1.0), 16000)
        self.assertEqual(self.mod._total_chunks_seen[0], before)

    def test_music_ends_when_coverage_drops(self):
        # Build sustained music, then feed enough non-musical chunks to push the
        # musical fraction below threshold → _music_active flips back off.
        tone = _tone(secs=1.0)
        for _ in range(7):
            self.mod.feed_audio(tone, 16000)
        self.assertTrue(self.mod._music_active[0])
        noise = _noise(secs=1.0)
        for _ in range(10):
            self.mod.feed_audio(noise, 16000)
        self.assertFalse(self.mod._music_active[0])

    def test_old_classifications_evicted_from_window(self):
        # An ancient classification is dropped once newer audio advances the
        # WINDOW_SECONDS cutoff past it.
        self.mod._classifications.append(
            (self.mod.time.time() - 999.0, True, 1.0))
        self.mod.feed_audio(_tone(secs=1.0), 16000)
        for ts, _is_m, _d in self.mod._classifications:
            self.assertGreater(ts, self.mod.time.time() - self.mod.WINDOW_SECONDS)


class StatusActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("standby_audio_detect")
        self.mod._classifications.clear()
        self.mod._music_active[0] = False
        self.mod._total_chunks_seen[0] = 0
        self.mod._total_music_chunks[0] = 0

    def test_status_no_audio_yet(self):
        out = self.actions["audio_music_status"]("")
        self.assertIn("No audio analysed yet", out)

    def test_status_reports_chunk_ratio(self):
        self.mod._total_chunks_seen[0] = 10
        self.mod._total_music_chunks[0] = 3
        out = self.actions["audio_music_status"]("")
        self.assertIn("3 of 10", out)
        self.assertIn("30%", out)

    def test_status_active_music(self):
        self.mod._music_active[0] = True
        self.mod._last_feed_at[0] = self.mod.time.time()
        self.mod._music_since[0] = self.mod.time.time() - (self.mod.SUSTAINED_HOLD_SECONDS + 5)
        out = self.actions["audio_music_status"]("")
        self.assertIn("Wake-word filtering is active", out)

    def test_status_tonal_but_still_confirming(self):
        # Active but the sustained-hold window not yet satisfied → "confirming".
        self.mod._music_active[0] = True
        self.mod._last_feed_at[0] = self.mod.time.time()
        self.mod._music_since[0] = self.mod.time.time() - 2   # < hold seconds
        out = self.actions["audio_music_status"]("")
        self.assertIn("Still confirming", out)

    def test_status_action_swallows_exception(self):
        # The action wraps music_state_summary in try/except.
        with mock.patch.object(self.mod, "music_state_summary",
                               side_effect=RuntimeError("boom")):
            out = self.actions["audio_music_status"]("")
        self.assertIn("audio music status failed", out)


class ResetIfStaleTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("standby_audio_detect")
        self.mod._music_active[0] = False
        self.mod._classifications.clear()

    def test_reset_noop_when_inactive(self):
        self.mod._music_active[0] = False
        self.mod._reset_if_stale()   # early return, nothing happens
        self.assertFalse(self.mod._music_active[0])

    def test_reset_clears_when_stale(self):
        self.mod._music_active[0] = True
        self.mod._last_feed_at[0] = self.mod.time.time() - (
            self.mod.MUSIC_TIMEOUT_SECONDS + 1)
        self.mod._classifications.append((self.mod.time.time(), True, 1.0))
        self.mod._reset_if_stale()
        self.assertFalse(self.mod._music_active[0])
        self.assertEqual(len(self.mod._classifications), 0)

    def test_reset_keeps_fresh_state(self):
        self.mod._music_active[0] = True
        self.mod._last_feed_at[0] = self.mod.time.time()   # just fed
        self.mod._reset_if_stale()
        self.assertTrue(self.mod._music_active[0])


# ─── librosa onset energy ────────────────────────────────────────────────
class OnsetEnergyTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("standby_audio_detect")
        # Reset the librosa cache so each test controls availability.
        self.mod._librosa_mod[0] = None
        self.addCleanup(lambda: self.mod._librosa_mod.__setitem__(0, None))

    def _librosa(self, env=None, raises=False):
        lib = types.ModuleType("librosa")
        onset_ns = types.SimpleNamespace()

        def _onset_strength(y=None, sr=None):
            if raises:
                raise RuntimeError("librosa boom")
            return np.array([0.4, 0.6, 0.8], dtype=np.float32) if env is None else env
        onset_ns.onset_strength = _onset_strength
        lib.onset = onset_ns
        return lib

    def test_onset_zero_when_librosa_absent(self):
        # _try_import_librosa returns None → 0.0.
        with mock.patch.object(self.mod, "_try_import_librosa", return_value=None):
            self.assertEqual(self.mod._onset_energy(_tone(), 16000), 0.0)

    def test_onset_mean_when_available(self):
        with mock.patch.object(self.mod, "_try_import_librosa",
                               return_value=self._librosa()):
            val = self.mod._onset_energy(_tone(secs=0.5), 16000)
        self.assertAlmostEqual(val, (0.4 + 0.6 + 0.8) / 3, places=4)

    def test_onset_zero_for_empty_audio(self):
        with mock.patch.object(self.mod, "_try_import_librosa",
                               return_value=self._librosa()):
            self.assertEqual(
                self.mod._onset_energy(np.array([], dtype=np.float32), 16000),
                0.0)

    def test_onset_stereo_is_downmixed(self):
        mono = _tone(secs=0.5)
        stereo = np.stack([mono, mono], axis=1)
        with mock.patch.object(self.mod, "_try_import_librosa",
                               return_value=self._librosa()):
            val = self.mod._onset_energy(stereo, 16000)
        self.assertGreater(val, 0.0)

    def test_onset_empty_env_returns_zero(self):
        lib = self._librosa(env=np.array([], dtype=np.float32))
        with mock.patch.object(self.mod, "_try_import_librosa", return_value=lib):
            self.assertEqual(self.mod._onset_energy(_tone(secs=0.5), 16000), 0.0)

    def test_onset_computation_exception_returns_zero(self):
        with mock.patch.object(self.mod, "_try_import_librosa",
                               return_value=self._librosa(raises=True)):
            self.assertEqual(self.mod._onset_energy(_tone(secs=0.5), 16000), 0.0)


# ─── _try_import_librosa caching ─────────────────────────────────────────
class TryImportLibrosaTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("standby_audio_detect")
        self.mod._librosa_mod[0] = None
        self.addCleanup(lambda: self.mod._librosa_mod.__setitem__(0, None))

    def test_returns_cached_when_present(self):
        sentinel = types.ModuleType("librosa")
        self.mod._librosa_mod[0] = sentinel
        self.assertIs(self.mod._try_import_librosa(), sentinel)

    def test_imports_and_caches(self):
        fake = types.ModuleType("librosa")
        with inject_modules(librosa=fake):
            got = self.mod._try_import_librosa()
        self.assertIs(got, fake)
        self.assertIs(self.mod._librosa_mod[0], fake)

    def test_returns_none_when_unimportable(self):
        # Force the import to fail even though librosa may be installed locally.
        with mock.patch.dict(sys.modules, {"librosa": None}):
            self.assertIsNone(self.mod._try_import_librosa())


# ─── whisper model loader ────────────────────────────────────────────────
class EnsureWhisperTinyTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("standby_audio_detect")
        self.mod._whisper_model[0] = None
        self.addCleanup(lambda: self.mod._whisper_model.__setitem__(0, None))

    def test_returns_cached_model(self):
        sentinel = object()
        self.mod._whisper_model[0] = sentinel
        self.assertIs(self.mod._ensure_whisper_tiny(), sentinel)

    def test_faster_whisper_cuda_path(self):
        # CUDA-first only happens when the GPU gate is opted in.
        self.mod._loop_cfg["prefer_gpu"] = True
        fw = types.ModuleType("faster_whisper")
        model = object()
        fw.WhisperModel = mock.MagicMock(return_value=model)
        with inject_modules(faster_whisper=fw):
            got = self.mod._ensure_whisper_tiny()
        self.assertIs(got, model)
        # First attempt is the cuda/float16 path.
        _args, kwargs = fw.WhisperModel.call_args
        self.assertEqual(kwargs.get("device"), "cuda")

    def test_faster_whisper_cpu_fallback(self):
        # With the GPU gate on, a cuda failure falls back to cpu/int8.
        self.mod._loop_cfg["prefer_gpu"] = True
        fw = types.ModuleType("faster_whisper")
        cpu_model = object()
        calls = {"n": 0}

        def _ctor(name, device=None, compute_type=None):
            calls["n"] += 1
            if device == "cuda":
                raise RuntimeError("no cuda")
            return cpu_model
        fw.WhisperModel = _ctor
        with inject_modules(faster_whisper=fw):
            got = self.mod._ensure_whisper_tiny()
        self.assertIs(got, cpu_model)
        self.assertEqual(calls["n"], 2)   # cuda failed → cpu succeeded

    def test_faster_whisper_cpu_by_default_skips_cuda(self):
        # The VRAM-stability default (STANDBY_WHISPER_PREFER_GPU False): the
        # loop must NOT touch CUDA — it loads straight on cpu/int8 in ONE call
        # so it never competes with the resident local LLM for the 24GB.
        self.assertFalse(self.mod._loop_cfg.get("prefer_gpu", False))
        fw = types.ModuleType("faster_whisper")
        cpu_model = object()
        calls = []

        def _ctor(name, device=None, compute_type=None):
            calls.append((device, compute_type))
            return cpu_model
        fw.WhisperModel = _ctor
        with inject_modules(faster_whisper=fw):
            got = self.mod._ensure_whisper_tiny()
        self.assertIs(got, cpu_model)
        self.assertEqual(len(calls), 1)            # no cuda probe at all
        self.assertEqual(calls[0], ("cpu", "int8"))

    def test_prefer_gpu_flag_restores_cuda_first(self):
        # Flipping the gate True puts the cuda/float16 attempt back first.
        self.mod._loop_cfg["prefer_gpu"] = True
        fw = types.ModuleType("faster_whisper")
        fw.WhisperModel = mock.MagicMock(return_value=object())
        with inject_modules(faster_whisper=fw):
            self.mod._ensure_whisper_tiny()
        self.assertEqual(fw.WhisperModel.call_args_list[0].kwargs.get("device"),
                         "cuda")

    def test_faster_whisper_both_fail_then_openai(self):
        fw = types.ModuleType("faster_whisper")
        fw.WhisperModel = mock.MagicMock(side_effect=RuntimeError("dead"))
        wlib = types.ModuleType("whisper")
        wmodel = object()
        wlib.load_model = mock.MagicMock(return_value=wmodel)
        # torch absent → openai-whisper stays on cpu.
        with inject_modules(faster_whisper=fw, whisper=wlib), \
             mock.patch.dict(sys.modules, {"torch": None}):
            got = self.mod._ensure_whisper_tiny()
        self.assertIs(got, wmodel)
        self.assertEqual(wlib.load_model.call_args.kwargs.get("device"), "cpu")

    def test_openai_whisper_cuda_when_torch_available(self):
        # No faster_whisper, openai-whisper present, torch reports cuda.
        wlib = types.ModuleType("whisper")
        wmodel = object()
        wlib.load_model = mock.MagicMock(return_value=wmodel)
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        with inject_modules(whisper=wlib, torch=torch), \
             mock.patch.dict(sys.modules, {"faster_whisper": None}):
            self.mod._ensure_whisper_tiny()
        self.assertEqual(wlib.load_model.call_args.kwargs.get("device"), "cuda")

    def test_all_engines_unavailable_returns_none(self):
        with mock.patch.dict(sys.modules,
                             {"faster_whisper": None, "whisper": None}):
            self.assertIsNone(self.mod._ensure_whisper_tiny())

    def test_openai_whisper_load_failure_returns_none(self):
        wlib = types.ModuleType("whisper")
        wlib.load_model = mock.MagicMock(side_effect=RuntimeError("no weights"))
        with inject_modules(whisper=wlib), \
             mock.patch.dict(sys.modules, {"faster_whisper": None, "torch": None}):
            self.assertIsNone(self.mod._ensure_whisper_tiny())


# ─── transcribe buffer ───────────────────────────────────────────────────
class TranscribeBufferTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("standby_audio_detect")
        self.addCleanup(lambda: self.mod._whisper_model.__setitem__(0, None))

    def test_empty_when_no_model(self):
        with mock.patch.object(self.mod, "_ensure_whisper_tiny", return_value=None):
            self.assertEqual(self.mod._transcribe_buffer(_tone(), 16000), "")

    def test_empty_for_empty_audio(self):
        with mock.patch.object(self.mod, "_ensure_whisper_tiny",
                               return_value=object()):
            self.assertEqual(
                self.mod._transcribe_buffer(np.array([], dtype=np.float32), 16000),
                "")

    def test_openai_dict_return_shape(self):
        model = types.SimpleNamespace(
            transcribe=lambda a, language=None: {"text": "Hello World"})
        with mock.patch.object(self.mod, "_ensure_whisper_tiny", return_value=model):
            out = self.mod._transcribe_buffer(_tone(secs=0.5), 16000)
        self.assertEqual(out, "hello world")   # lowercased + stripped

    def test_faster_whisper_tuple_return_shape(self):
        seg = types.SimpleNamespace(text="La La ")
        model = types.SimpleNamespace(
            transcribe=lambda a, language=None: (iter([seg]), {"lang": "en"}))
        with mock.patch.object(self.mod, "_ensure_whisper_tiny", return_value=model):
            out = self.mod._transcribe_buffer(_tone(secs=0.5), 16000)
        self.assertEqual(out, "la la")

    def test_unexpected_return_is_stringified(self):
        model = types.SimpleNamespace(transcribe=lambda a, language=None: 12345)
        with mock.patch.object(self.mod, "_ensure_whisper_tiny", return_value=model):
            out = self.mod._transcribe_buffer(_tone(secs=0.5), 16000)
        self.assertEqual(out, "12345")

    def test_stereo_downmixed_before_transcribe(self):
        captured = {}

        def _tx(a, language=None):
            captured["ndim"] = a.ndim
            return {"text": "x"}
        model = types.SimpleNamespace(transcribe=_tx)
        mono = _tone(secs=0.5)
        stereo = np.stack([mono, mono], axis=1)
        with mock.patch.object(self.mod, "_ensure_whisper_tiny", return_value=model):
            self.mod._transcribe_buffer(stereo, 16000)
        self.assertEqual(captured["ndim"], 1)

    def test_transcribe_exception_returns_empty(self):
        model = types.SimpleNamespace(
            transcribe=mock.MagicMock(side_effect=RuntimeError("bad audio")))
        with mock.patch.object(self.mod, "_ensure_whisper_tiny", return_value=model):
            self.assertEqual(self.mod._transcribe_buffer(_tone(secs=0.5), 16000), "")

    def test_model_without_transcribe_returns_empty(self):
        model = object()   # no .transcribe attr
        with mock.patch.object(self.mod, "_ensure_whisper_tiny", return_value=model):
            self.assertEqual(self.mod._transcribe_buffer(_tone(secs=0.5), 16000), "")


# ─── _suppress_due_to_state ──────────────────────────────────────────────
class SuppressDueToStateTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("standby_audio_detect")

    def test_suppress_when_standby(self):
        bc = types.SimpleNamespace(_standby_mode=[True], _sleep_mode=[False],
                                   _jarvis_played_music_at=[0.0])
        self.assertTrue(self.mod._suppress_due_to_state(bc))

    def test_suppress_when_sleep(self):
        bc = types.SimpleNamespace(_standby_mode=[False], _sleep_mode=[True],
                                   _jarvis_played_music_at=[0.0])
        self.assertTrue(self.mod._suppress_due_to_state(bc))

    def test_suppress_when_jarvis_recently_played_music(self):
        bc = types.SimpleNamespace(
            _standby_mode=[False], _sleep_mode=[False],
            _jarvis_played_music_at=[self.mod.time.time() - 5])   # < 60s ago
        self.assertTrue(self.mod._suppress_due_to_state(bc))

    def test_no_suppress_when_idle(self):
        bc = types.SimpleNamespace(
            _standby_mode=[False], _sleep_mode=[False],
            _jarvis_played_music_at=[self.mod.time.time() - 600])  # long ago
        self.assertFalse(self.mod._suppress_due_to_state(bc))

    def test_missing_attrs_do_not_raise(self):
        bc = types.SimpleNamespace()   # none of the probed attrs exist
        self.assertFalse(self.mod._suppress_due_to_state(bc))


# ─── _load_loop_cfg ──────────────────────────────────────────────────────
class LoadLoopCfgTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("standby_audio_detect")
        self._orig_cfg = dict(self.mod._loop_cfg)
        self.addCleanup(lambda: self.mod._loop_cfg.update(self._orig_cfg))

    def test_pulls_values_from_core_config(self):
        cfg = types.ModuleType("core.config")
        cfg.STANDBY_LOOP_ENABLED = False
        cfg.STANDBY_LOOP_BUFFER_SECONDS = 4.0
        cfg.STANDBY_LOOP_CHECK_INTERVAL_SEC = 7.0
        cfg.STANDBY_LOOP_MATCH_WINDOWS = 5
        cfg.STANDBY_LOOP_ONSET_ENERGY_MIN = 0.5
        cfg.STANDBY_LOOP_RHYME_RATIO_MIN = 0.6
        cfg.STANDBY_LOOP_WHISPER_MODEL = "base"
        with inject_modules(**{"core.config": cfg}):
            self.mod._load_loop_cfg()
        self.assertFalse(self.mod._loop_cfg["enabled"])
        self.assertEqual(self.mod._loop_cfg["buffer_seconds"], 4.0)
        self.assertEqual(self.mod._loop_cfg["check_interval"], 7.0)
        self.assertEqual(self.mod._loop_cfg["match_windows"], 5)
        self.assertEqual(self.mod._loop_cfg["whisper_model"], "base")

    def test_missing_attrs_keep_defaults(self):
        cfg = types.ModuleType("core.config")   # defines none of the names
        with inject_modules(**{"core.config": cfg}):
            self.mod._load_loop_cfg()
        # Untouched defaults survive.
        self.assertEqual(self.mod._loop_cfg["buffer_seconds"],
                         self.mod._LOOP_DEFAULTS["buffer_seconds"])

    def test_import_failure_is_silent(self):
        with mock.patch.dict(sys.modules, {"core.config": None}), \
             mock.patch("builtins.__import__",
                        side_effect=ImportError("no config")):
            self.mod._load_loop_cfg()   # no raise


# ─── start/stop background loop ──────────────────────────────────────────
class BackgroundLoopLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("standby_audio_detect")
        # Make sure no real thread is considered running.
        self.mod._loop_thread = None
        self.mod._loop_stop.clear()
        self.addCleanup(self._teardown)

    def _teardown(self):
        self.mod._loop_stop.set()
        self.mod._loop_thread = None

    def test_start_skipped_when_disabled(self):
        self.mod._loop_cfg["enabled"] = False
        self.addCleanup(lambda: self.mod._loop_cfg.__setitem__("enabled", True))
        with mock.patch.object(self.mod.threading, "Thread") as T:
            self.mod._start_background_loop()
        T.assert_not_called()
        self.assertIsNone(self.mod._loop_thread)

    def test_start_skipped_when_librosa_absent(self):
        self.mod._loop_cfg["enabled"] = True
        with mock.patch.object(self.mod, "_try_import_librosa", return_value=None), \
             mock.patch.object(self.mod.threading, "Thread") as T:
            self.mod._start_background_loop()
        T.assert_not_called()

    def test_start_launches_thread_when_ready(self):
        self.mod._loop_cfg["enabled"] = True
        fake_thread = mock.MagicMock()
        with mock.patch.object(self.mod, "_try_import_librosa",
                               return_value=types.ModuleType("librosa")), \
             mock.patch.object(self.mod.threading, "Thread",
                               return_value=fake_thread) as T:
            self.mod._start_background_loop()
        T.assert_called_once()
        fake_thread.start.assert_called_once()
        self.assertFalse(self.mod._loop_stop.is_set())   # cleared before start

    def test_start_noop_when_already_running(self):
        running = mock.MagicMock()
        running.is_alive.return_value = True
        self.mod._loop_thread = running
        with mock.patch.object(self.mod.threading, "Thread") as T:
            self.mod._start_background_loop()
        T.assert_not_called()

    def test_stop_sets_event(self):
        self.mod._loop_stop.clear()
        self.mod.stop_background_loop()
        self.assertTrue(self.mod._loop_stop.is_set())


# ─── _background_loop (single-iteration driven) ──────────────────────────
class BackgroundLoopBodyTests(unittest.TestCase):
    """Drive exactly one loop pass by making _loop_stop.wait return False once
    (proceed) then True (exit). No real audio, whisper, or threads."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("standby_audio_detect")
        self.mod._loop_consecutive[0] = 0
        self._orig_cfg = dict(self.mod._loop_cfg)
        self.addCleanup(lambda: self.mod._loop_cfg.update(self._orig_cfg))
        self.addCleanup(lambda: sys.modules.pop("bobert_companion", None))

    @contextlib.contextmanager
    def _run_iterations(self, n):
        state = {"i": 0}

        def _wait(_interval):
            state["i"] += 1
            return state["i"] > n   # False for n calls, then True → stop
        with mock.patch.object(self.mod._loop_stop, "wait", side_effect=_wait):
            yield

    def _bc(self, **attrs):
        bc = types.ModuleType("bobert_companion")
        bc.SAMPLE_RATE = 16000
        bc.get_mic_buffer = attrs.pop(
            "get_mic_buffer", lambda secs, sample_rate=16000: _tone(secs=0.5))
        bc.is_using_headset = attrs.pop("is_using_headset", lambda: True)
        bc._standby_auto_engage = attrs.pop(
            "_standby_auto_engage", mock.MagicMock(return_value=True))
        for k, v in attrs.items():
            setattr(bc, k, v)
        return bc

    def test_no_bc_continues(self):
        with inject_modules(bobert_companion=None), self._run_iterations(1):
            self.mod._background_loop()   # bc is None → continue, then stop

    def test_suppressed_state_resets_consecutive(self):
        bc = self._bc()
        self.mod._loop_consecutive[0] = 2
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_suppress_due_to_state",
                               return_value=True), \
             self._run_iterations(1):
            self.mod._background_loop()
        self.assertEqual(self.mod._loop_consecutive[0], 0)

    def test_mic_buffer_failure_continues(self):
        def _boom(secs, sample_rate=16000):
            raise RuntimeError("mic gone")
        bc = self._bc(get_mic_buffer=_boom)
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_suppress_due_to_state",
                               return_value=False), \
             self._run_iterations(1):
            self.mod._background_loop()   # exception caught → continue

    def test_empty_buffer_resets_consecutive(self):
        bc = self._bc(get_mic_buffer=lambda secs, sample_rate=16000:
                      np.array([], dtype=np.float32))
        self.mod._loop_consecutive[0] = 3
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_suppress_due_to_state",
                               return_value=False), \
             self._run_iterations(1):
            self.mod._background_loop()
        self.assertEqual(self.mod._loop_consecutive[0], 0)

    def test_scoring_exception_releases_buffer_and_continues(self):
        bc = self._bc()
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_suppress_due_to_state",
                               return_value=False), \
             mock.patch.object(self.mod, "_transcribe_buffer",
                               side_effect=RuntimeError("score boom")), \
             self._run_iterations(1):
            self.mod._background_loop()

    def test_non_lyric_resets_consecutive(self):
        bc = self._bc()
        self.mod._loop_consecutive[0] = 2
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_suppress_due_to_state",
                               return_value=False), \
             mock.patch.object(self.mod, "_transcribe_buffer", return_value=""), \
             mock.patch.object(self.mod, "_onset_energy", return_value=0.0), \
             mock.patch.object(self.mod, "_looks_like_lyrics", return_value=False), \
             self._run_iterations(1):
            self.mod._background_loop()
        self.assertEqual(self.mod._loop_consecutive[0], 0)

    def test_lyric_below_threshold_increments_only(self):
        bc = self._bc()
        self.mod._loop_cfg["match_windows"] = 3
        self.mod._loop_consecutive[0] = 0
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_suppress_due_to_state",
                               return_value=False), \
             mock.patch.object(self.mod, "_transcribe_buffer", return_value="la"), \
             mock.patch.object(self.mod, "_onset_energy", return_value=0.9), \
             mock.patch.object(self.mod, "_looks_like_lyrics", return_value=True), \
             self._run_iterations(1):
            self.mod._background_loop()
        # One positive window, threshold 3 → just incremented, no engage.
        self.assertEqual(self.mod._loop_consecutive[0], 1)
        bc._standby_auto_engage.assert_not_called()

    def test_threshold_reached_not_headset_skips_engage(self):
        bc = self._bc(is_using_headset=lambda: False)
        self.mod._loop_cfg["match_windows"] = 1
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_suppress_due_to_state",
                               return_value=False), \
             mock.patch.object(self.mod, "_transcribe_buffer", return_value="la la"), \
             mock.patch.object(self.mod, "_onset_energy", return_value=0.9), \
             mock.patch.object(self.mod, "_looks_like_lyrics", return_value=True), \
             self._run_iterations(1):
            self.mod._background_loop()
        bc._standby_auto_engage.assert_not_called()

    def test_threshold_reached_headset_engages_standby(self):
        engage = mock.MagicMock(return_value=True)
        bc = self._bc(is_using_headset=lambda: True, _standby_auto_engage=engage)
        self.mod._loop_cfg["match_windows"] = 1
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_suppress_due_to_state",
                               return_value=False), \
             mock.patch.object(self.mod, "_transcribe_buffer", return_value="la la"), \
             mock.patch.object(self.mod, "_onset_energy", return_value=0.9), \
             mock.patch.object(self.mod, "_looks_like_lyrics", return_value=True), \
             self._run_iterations(1):
            self.mod._background_loop()
        engage.assert_called_once_with("vocal-music")
        self.assertEqual(self.mod._loop_consecutive[0], 0)   # reset after firing

    def test_headset_check_raises_treated_as_not_headset(self):
        def _boom():
            raise RuntimeError("no audio api")
        engage = mock.MagicMock(return_value=True)
        bc = self._bc(is_using_headset=_boom, _standby_auto_engage=engage)
        self.mod._loop_cfg["match_windows"] = 1
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_suppress_due_to_state",
                               return_value=False), \
             mock.patch.object(self.mod, "_transcribe_buffer", return_value="la la"), \
             mock.patch.object(self.mod, "_onset_energy", return_value=0.9), \
             mock.patch.object(self.mod, "_looks_like_lyrics", return_value=True), \
             self._run_iterations(1):
            self.mod._background_loop()
        engage.assert_not_called()

    def test_bridge_function_missing_resets_and_continues(self):
        bc = self._bc(is_using_headset=lambda: True)
        bc._standby_auto_engage = None   # bridge not wired
        self.mod._loop_cfg["match_windows"] = 1
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_suppress_due_to_state",
                               return_value=False), \
             mock.patch.object(self.mod, "_transcribe_buffer", return_value="la la"), \
             mock.patch.object(self.mod, "_onset_energy", return_value=0.9), \
             mock.patch.object(self.mod, "_looks_like_lyrics", return_value=True), \
             self._run_iterations(1):
            self.mod._background_loop()
        self.assertEqual(self.mod._loop_consecutive[0], 0)

    def test_engage_raises_is_caught(self):
        def _boom(_reason):
            raise RuntimeError("engage boom")
        bc = self._bc(is_using_headset=lambda: True, _standby_auto_engage=_boom)
        self.mod._loop_cfg["match_windows"] = 1
        with inject_modules(bobert_companion=bc), \
             mock.patch.object(self.mod, "_suppress_due_to_state",
                               return_value=False), \
             mock.patch.object(self.mod, "_transcribe_buffer", return_value="la la"), \
             mock.patch.object(self.mod, "_onset_energy", return_value=0.9), \
             mock.patch.object(self.mod, "_looks_like_lyrics", return_value=True), \
             self._run_iterations(1):
            self.mod._background_loop()   # engage exception caught → no raise
        self.assertEqual(self.mod._loop_consecutive[0], 0)


# ─── register() wiring ───────────────────────────────────────────────────
class RegisterTests(unittest.TestCase):
    def test_register_exposes_status_action_and_starts_loop(self):
        # The harness neuters Thread.start, so _start_background_loop is safe;
        # just confirm registration wired the action.
        mod, actions = load_skill_isolated("standby_audio_detect")
        self.assertIn("audio_music_status", actions)
        self.assertTrue(callable(actions["audio_music_status"]))


if __name__ == "__main__":
    unittest.main()
