"""Logic tests for skills/enroll_voice.py.

enroll_voice is a thin voice layer over core.voice_id (Resemblyzer). Its logic
is almost entirely graceful-degradation + delegation, so tests focus on:
  • the missing-core / Resemblyzer-unavailable / no-one-enrolled messages,
  • _default_user resolution from bobert / env / fallback,
  • the enroll / identify / list / forget / set-active happy + error paths
    with core.voice_id and the mic-capture mocked (no sounddevice, no model).

mod._voice_id is patched per-test to return a controllable stub, and
mod._record_seconds is patched so no real audio device is opened.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from tests._skill_harness import load_skill_isolated


def _fake_vid(available=True, enrolled=None, **over):
    vid = mock.MagicMock()
    vid.is_available.return_value = available
    vid.list_enrolled.return_value = enrolled if enrolled is not None else []
    vid.CONFIDENCE_THRESHOLD = over.get("threshold", 0.75)
    return vid


_FAKE_AUDIO = np.zeros(16000, dtype=np.float32)


class EnrollVoiceDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("enroll_voice")

    def test_enroll_missing_core(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=None):
            self.assertIn("Voice ID core is missing", self.actions["enroll_voice"](""))

    def test_enroll_resemblyzer_unavailable(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=_fake_vid(available=False)):
            out = self.actions["enroll_voice"]("")
        self.assertIn("Resemblyzer isn't installed", out)

    def test_whos_talking_no_enrollments(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=_fake_vid(enrolled=[])):
            out = self.actions["whos_talking"]("")
        self.assertIn("single-user mode", out.lower())

    def test_list_enrolled_none(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=_fake_vid(enrolled=[])):
            out = self.actions["list_enrolled_voices"]("")
        self.assertIn("No voiceprints enrolled", out)

    def test_forget_requires_name(self):
        with mock.patch.object(self.mod, "_voice_id", return_value=_fake_vid()):
            self.assertIn("whose voiceprint to forget", self.actions["forget_voice"](""))

    # ── _default_user resolution ─────────────────────────────────────────
    def test_default_user_from_bobert(self):
        bc = mock.MagicMock()
        bc.VOICE_ID_DEFAULT_USER = "Alice"
        with mock.patch.object(self.mod, "_bobert", return_value=bc):
            self.assertEqual(self.mod._default_user(), "Alice")

    def test_default_user_from_env_when_no_bobert(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {"VOICE_ID_DEFAULT_USER": "Bob"}, clear=False):
            self.assertEqual(self.mod._default_user(), "Bob")

    def test_default_user_fallback(self):
        with mock.patch.object(self.mod, "_bobert", return_value=None), \
             mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.mod._default_user(), "user")


class EnrollVoiceActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("enroll_voice")

    # ── enroll_voice ─────────────────────────────────────────────────────
    def test_enroll_capture_failure(self):
        vid = _fake_vid()
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=None), \
             mock.patch.object(self.mod, "_say"):
            out = self.actions["enroll_voice"]("Alice")
        self.assertIn("couldn't capture the mic", out)

    def test_enroll_happy_first_sample(self):
        vid = _fake_vid()
        vid.enroll_from_audio.return_value = {"ok": True, "name": "Alice", "sample_count": 1}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=_FAKE_AUDIO), \
             mock.patch.object(self.mod, "_say"):
            out = self.actions["enroll_voice"]("Alice")
        self.assertIn("Voiceprint saved for Alice", out)
        # First sample → no "sample N" suffix.
        self.assertNotIn("sample 1", out)

    def test_enroll_reports_additional_samples(self):
        vid = _fake_vid()
        vid.enroll_from_audio.return_value = {"ok": True, "name": "Alice", "sample_count": 3}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=_FAKE_AUDIO), \
             mock.patch.object(self.mod, "_say"):
            out = self.actions["enroll_voice"]("Alice")
        self.assertIn("sample 3", out)

    def test_enroll_failure_surfaces_error(self):
        vid = _fake_vid()
        vid.enroll_from_audio.return_value = {"ok": False, "error": "embedding too short"}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=_FAKE_AUDIO), \
             mock.patch.object(self.mod, "_say"):
            out = self.actions["enroll_voice"]("Alice")
        self.assertIn("Enrollment failed", out)
        self.assertIn("embedding too short", out)

    def test_enroll_defaults_name_when_blank(self):
        vid = _fake_vid()
        vid.enroll_from_audio.return_value = {"ok": True, "name": "user", "sample_count": 1}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_default_user", return_value="user"), \
             mock.patch.object(self.mod, "_record_seconds", return_value=_FAKE_AUDIO), \
             mock.patch.object(self.mod, "_say"):
            self.actions["enroll_voice"]("")
        # The resolved default name is passed to enroll_from_audio.
        self.assertEqual(vid.enroll_from_audio.call_args[0][0], "user")

    # ── whos_talking ─────────────────────────────────────────────────────
    def test_whos_talking_match(self):
        vid = _fake_vid(enrolled=["Alice"])
        vid.identify_speaker.return_value = ("Alice", 0.91)
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=_FAKE_AUDIO):
            out = self.actions["whos_talking"]("")
        self.assertIn("sounds like Alice", out)
        self.assertIn("0.91", out)

    def test_whos_talking_no_match(self):
        vid = _fake_vid(enrolled=["Alice"], threshold=0.75)
        vid.identify_speaker.return_value = (None, 0.40)
        with mock.patch.object(self.mod, "_voice_id", return_value=vid), \
             mock.patch.object(self.mod, "_record_seconds", return_value=_FAKE_AUDIO):
            out = self.actions["whos_talking"]("")
        self.assertIn("doesn't match anyone", out)
        self.assertIn("0.40", out)

    # ── list / forget / set-active ───────────────────────────────────────
    def test_list_enrolled_with_active(self):
        vid = _fake_vid(enrolled=["Alice", "Bob"])
        vid.get_active_speaker.return_value = "Alice"
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["list_enrolled_voices"]("")
        self.assertIn("Alice, Bob", out)
        self.assertIn("Active speaker: Alice", out)

    def test_forget_known(self):
        vid = _fake_vid()
        vid.forget_speaker.return_value = True
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["forget_voice"]("Bob")
        self.assertIn("Forgotten Bob's voiceprint", out)

    def test_forget_unknown(self):
        vid = _fake_vid()
        vid.forget_speaker.return_value = False
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["forget_voice"]("Nobody")
        self.assertIn("don't have a voiceprint enrolled for Nobody", out)

    def test_set_active_clears_on_empty(self):
        vid = _fake_vid()
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["set_active_speaker"]("")
        self.assertIn("Active speaker cleared", out)
        vid.set_active_speaker.assert_called_once_with(None)

    def test_set_active_known(self):
        vid = _fake_vid()
        vid.set_active_speaker.return_value = True
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["set_active_speaker"]("Alice")
        self.assertIn("Active speaker set to Alice", out)

    # ── voice_id_status ──────────────────────────────────────────────────
    def test_status_offline(self):
        vid = _fake_vid()
        vid.encoder_status.return_value = {"encoder_loaded": False,
                                           "encoder_error": "model file missing"}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["voice_id_status"]("")
        self.assertIn("Voice ID is offline", out)
        self.assertIn("model file missing", out)

    def test_status_online(self):
        vid = _fake_vid()
        vid.encoder_status.return_value = {
            "encoder_loaded": True, "enrolled": ["Alice"],
            "active_speaker": "Alice", "threshold": 0.75}
        with mock.patch.object(self.mod, "_voice_id", return_value=vid):
            out = self.actions["voice_id_status"]("")
        self.assertIn("Voice ID is online", out)
        self.assertIn("Alice", out)
        self.assertIn("0.75", out)


if __name__ == "__main__":
    unittest.main()
