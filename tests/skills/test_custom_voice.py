"""Logic tests for skills/custom_voice.py (Coqui XTTS-v2 voice cloning).

XTTS needs a GPU + the `TTS` package + a sample WAV, none of which exist in the
test env — so graceful degradation is the dominant path and is easy to assert.
We also test the pure prosody parsers, config resolution off a faked
bobert_companion, the backend-toggle validation/refusal, and the voice
pre-router regex. No model is ever loaded; the only render tests exercise the
early-return and the availability-failure raise.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np

from tests._skill_harness import load_skill_isolated


class _FakeBobert:
    """A stand-in bobert_companion with the config attrs custom_voice reads."""
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class RateParseTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_positive_percent(self):
        self.assertAlmostEqual(self.mod._parse_rate("+5%"), 1.05)

    def test_negative_percent(self):
        self.assertAlmostEqual(self.mod._parse_rate("-10%"), 0.90)

    def test_clamped_to_bounds(self):
        self.assertEqual(self.mod._parse_rate("+500%"), 2.0)   # upper clamp
        self.assertEqual(self.mod._parse_rate("-90%"), 0.5)    # lower clamp

    def test_unparseable_is_unity(self):
        self.assertEqual(self.mod._parse_rate("fast"), 1.0)
        self.assertEqual(self.mod._parse_rate(""), 1.0)


class PitchParseTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_zero_hz_is_zero_semitones(self):
        self.assertEqual(self.mod._parse_pitch_semitones("+0Hz"), 0.0)

    def test_positive_hz_positive_semitones(self):
        st = self.mod._parse_pitch_semitones("+4Hz")
        self.assertGreater(st, 0.0)
        self.assertLess(st, 1.0)   # ~0.34 semitones around a 200 Hz base

    def test_unparseable_is_zero(self):
        self.assertEqual(self.mod._parse_pitch_semitones("higher"), 0.0)


class ConfigResolutionTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_backend_from_bobert(self):
        fake = _FakeBobert(TTS_BACKEND="xtts")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}):
            self.assertEqual(self.mod.get_backend(), "xtts")

    def test_backend_from_env_when_bobert_absent(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {"TTS_BACKEND": "pyttsx3"}, clear=True):
            self.assertEqual(self.mod.get_backend(), "pyttsx3")

    def test_backend_defaults_to_edge(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.mod.get_backend(), "edge")

    def test_invalid_backend_value_ignored(self):
        fake = _FakeBobert(TTS_BACKEND="banana")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.mod.get_backend(), "edge")

    def test_language_default(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.mod.get_language(), "en")


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_unavailable_when_tts_lib_missing(self):
        with mock.patch.object(self.mod, "_probe_tts_lib", return_value=False):
            self.assertFalse(self.mod.is_available())
            reason = self.mod.availability_reason()
        self.assertIn("Coqui TTS isn't installed", reason)

    def test_unavailable_when_no_sample(self):
        with mock.patch.object(self.mod, "_probe_tts_lib", return_value=True), \
             mock.patch.object(self.mod, "get_sample_path", return_value=""):
            self.assertFalse(self.mod.is_available())
            self.assertIn("No voice sample", self.mod.availability_reason())

    def test_unavailable_when_sample_path_missing_file(self):
        with mock.patch.object(self.mod, "_probe_tts_lib", return_value=True), \
             mock.patch.object(self.mod, "get_sample_path",
                               return_value=r"C:\nope\missing.wav"):
            self.assertFalse(self.mod.is_available())
            self.assertIn("can't find the voice sample", self.mod.availability_reason())

    def test_available_when_lib_and_sample_present(self):
        with mock.patch.object(self.mod, "_probe_tts_lib", return_value=True), \
             mock.patch.object(self.mod, "get_sample_path",
                               return_value=r"C:\voice\sample.wav"), \
             mock.patch.object(self.mod.os.path, "isfile", return_value=True):
            self.assertTrue(self.mod.is_available())
            self.assertEqual(self.mod.availability_reason(), "")


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_empty_text_returns_silence(self):
        audio, sr = self.mod.render("   ")
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(len(audio), 1)

    def test_render_raises_when_unavailable(self):
        # availability_reason non-empty → render must raise so the caller
        # (bobert synthesise) can fall back to edge-tts.
        with mock.patch.object(self.mod, "availability_reason",
                               return_value="Coqui TTS isn't installed, sir"):
            with self.assertRaises(RuntimeError) as cm:
                self.mod.render("hello there")
        self.assertIn("Coqui TTS isn't installed", str(cm.exception))


class SetBackendTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("custom_voice")

    def test_invalid_backend_rejected(self):
        out = self.mod.set_backend("banana")
        self.assertIn("isn't a TTS backend", out)
        self.assertIn("edge", out)

    def test_switch_to_edge_confirms_and_sets(self):
        fake = _FakeBobert(TTS_BACKEND="xtts")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}):
            out = self.mod.set_backend("edge")
        self.assertIn("edge-tts", out)
        self.assertEqual(fake.TTS_BACKEND, "edge")

    def test_switch_to_xtts_refused_when_unavailable(self):
        fake = _FakeBobert(TTS_BACKEND="edge")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}), \
             mock.patch.object(self.mod, "availability_reason",
                               return_value="No voice sample, sir"):
            out = self.mod.set_backend("xtts")
        self.assertIn("No voice sample", out)
        # Backend must NOT have been flipped to xtts on refusal.
        self.assertEqual(fake.TTS_BACKEND, "edge")

    def test_switch_to_xtts_when_available_warms_and_sets(self):
        fake = _FakeBobert(TTS_BACKEND="edge")
        with mock.patch.dict(sys.modules, {"bobert_companion": fake}), \
             mock.patch.object(self.mod, "availability_reason", return_value=""), \
             mock.patch.object(self.mod.threading, "Thread") as Thread:
            out = self.mod.set_backend("xtts")
        self.assertIn("cloned voice", out)
        self.assertEqual(fake.TTS_BACKEND, "xtts")
        Thread.assert_called_once()   # background warm-up spawned (not started for real)


class MaybeSwitchBackendTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("custom_voice")

    def test_use_my_voice_routes_to_xtts(self):
        with mock.patch.object(self.mod, "set_backend",
                               return_value="ok-xtts") as sb:
            out = self.mod.maybe_switch_backend("use my voice")
        self.assertEqual(out, "ok-xtts")
        sb.assert_called_once_with("xtts")

    def test_switch_to_edge_voice_routes_to_edge(self):
        with mock.patch.object(self.mod, "set_backend", return_value="ok") as sb:
            self.mod.maybe_switch_backend("switch to the edge voice")
        sb.assert_called_once_with("edge")

    def test_offline_voice_routes_to_pyttsx3(self):
        with mock.patch.object(self.mod, "set_backend", return_value="ok") as sb:
            self.mod.maybe_switch_backend("use the offline voice")
        sb.assert_called_once_with("pyttsx3")

    def test_non_matching_utterance_returns_none(self):
        self.assertIsNone(self.mod.maybe_switch_backend("what time is it"))

    def test_empty_returns_none(self):
        self.assertIsNone(self.mod.maybe_switch_backend(""))


class EnrollSampleActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("custom_voice")

    def test_requires_path(self):
        self.assertIn("format:", self.actions["enroll_xtts_sample"](""))

    def test_missing_file(self):
        out = self.actions["enroll_xtts_sample"](r"C:\nope\missing.wav")
        self.assertIn("no such file", out.lower())

    def test_sets_path_and_invalidates_cache(self):
        fake = _FakeBobert()
        with mock.patch.object(self.mod.os.path, "isfile", return_value=True), \
             mock.patch.dict(sys.modules, {"bobert_companion": fake}), \
             mock.patch.object(self.mod, "_invalidate_model_cache") as inval:
            out = self.actions["enroll_xtts_sample"](r"C:\voice\new.wav")
        self.assertIn("Voice sample updated", out)
        self.assertTrue(getattr(fake, "XTTS_VOICE_SAMPLE", "").endswith("new.wav"))
        inval.assert_called_once()

    def test_list_backends_reports_current(self):
        with mock.patch.object(self.mod, "get_backend", return_value="edge"), \
             mock.patch.object(self.mod, "is_available", return_value=False), \
             mock.patch.object(self.mod, "availability_reason",
                               return_value="Coqui TTS isn't installed"), \
             mock.patch.object(self.mod, "get_sample_path", return_value=""):
            out = self.actions["list_tts_backends"]("")
        self.assertIn("current backend: edge", out)
        self.assertIn("xtts status", out)


if __name__ == "__main__":
    unittest.main()
