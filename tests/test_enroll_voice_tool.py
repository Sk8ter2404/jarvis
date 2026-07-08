"""Tests for tools/enroll_voice.py — the out-of-band voice-clone enrollment CLI.

Focus: the 2026-07-08 fix that a FAILED reference-wav copy/record must not leave
an empty profile directory behind (which would show up in --list as an enrolled
profile with no meta.json/reference.wav), while a failed RE-enroll must NOT wipe
a pre-existing profile.

Pure stdlib unittest; PROFILES_DIR is redirected to a per-test temp dir and the
heavy _copy_reference_wav is mocked, so no real audio / soundfile is touched.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from tools import enroll_voice


class EnrollCleanupTests(unittest.TestCase):
    def test_failed_copy_removes_freshly_created_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(enroll_voice, "PROFILES_DIR", tmp), \
                 mock.patch.object(enroll_voice, "_copy_reference_wav",
                                   side_effect=FileNotFoundError("bad wav")):
                with self.assertRaises(FileNotFoundError):
                    enroll_voice.enroll("me", "owner", consent=True,
                                        from_wav="does_not_exist.wav")
                # No half-written profile dir left behind → --list stays clean.
                self.assertFalse(os.path.isdir(os.path.join(tmp, "me")))
                self.assertEqual(enroll_voice.list_profiles(), [])

    def test_preexisting_profile_survives_failed_reenroll(self):
        with tempfile.TemporaryDirectory() as tmp:
            prof = os.path.join(tmp, "me")
            os.makedirs(prof)
            with open(os.path.join(prof, "reference.wav"), "wb") as f:
                f.write(b"RIFF-existing")
            with mock.patch.object(enroll_voice, "PROFILES_DIR", tmp), \
                 mock.patch.object(enroll_voice, "_copy_reference_wav",
                                   side_effect=FileNotFoundError("bad wav")):
                with self.assertRaises(FileNotFoundError):
                    enroll_voice.enroll("me", "owner", consent=True,
                                        from_wav="does_not_exist.wav")
            # A failed re-enroll must NOT wipe the pre-existing profile.
            self.assertTrue(os.path.isdir(prof))
            self.assertTrue(os.path.isfile(os.path.join(prof, "reference.wav")))

    def test_successful_enroll_writes_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(enroll_voice, "PROFILES_DIR", tmp), \
                 mock.patch.object(enroll_voice, "_copy_reference_wav",
                                   return_value=os.path.join(tmp, "me",
                                                             "reference.wav")):
                enroll_voice.enroll("me", "owner", consent=True,
                                    from_wav="ok.wav")
            self.assertIn("me", enroll_voice_list(tmp))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "me", "meta.json")))


def enroll_voice_list(tmp):
    with mock.patch.object(enroll_voice, "PROFILES_DIR", tmp):
        return enroll_voice.list_profiles()


if __name__ == "__main__":
    unittest.main()
