"""Unit tests for bobert_companion.py lines ~2705-4442.

Section 2 of the monolith: HUD tray-state mirror/restore, the tray command
drainer + state publisher, face-tracking helpers (cv2 cascade detection,
camera probing), monitor/mic/speaker enumeration, audio-device auto-switching
(_pick_device / _refresh_devices / get_input_device ...), the proactive-speech
queue (proactive_announce), the proactive-idle gate, the late-night-remark
state machine, the thinking-eye animation, and the main-loop watchdog.

The monolith is imported ONCE via the cached harness (load_monolith()). Every
test patches the exact bc.* attributes the function under test touches with
mock.patch.object (auto-restores). External I/O (cv2 / sounddevice / requests /
psutil / ctypes / subprocess / threads / time.sleep / the LLM / the filesystem)
is mocked — nothing here opens a real camera, microphone, network socket, LLM
session, or spawns a long-lived thread.

These are decorated @requires_monolith so they SKIP on the light-deps CI runner
and RUN in the local full tier (intended).
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class _MonolithSec2Base(MonolithGlobalsTestCase):
    """Shared setup: load the cached monolith once for the whole class and
    deep-restore the mutated bobert_companion globals after each test
    (inherited from ``MonolithGlobalsTestCase``)."""


# ───────────────────────────────────────────────────────────────────────────
#  _friendly_device_name  (pure string transform)
# ───────────────────────────────────────────────────────────────────────────
class FriendlyDeviceNameTests(_MonolithSec2Base):
    def test_parenthetical_extracted(self):
        self.assertEqual(
            self.bc._friendly_device_name("Microphone (USB Mic), MME"),
            "USB Mic")

    def test_headset_microphone_prefix_with_parens(self):
        self.assertEqual(
            self.bc._friendly_device_name(
                "Headset Microphone (Gaming Headset), Windows DirectSound"),
            "Gaming Headset")

    def test_speakers_parenthetical(self):
        self.assertEqual(
            self.bc._friendly_device_name("Speakers (Realtek)"), "Realtek")

    def test_prefix_strip_without_parens(self):
        # No "(...)" group → falls through to the prefix-strip branch.
        self.assertEqual(
            self.bc._friendly_device_name("Microphone Blue Yeti"), "Blue Yeti")

    def test_no_match_returns_first_segment(self):
        self.assertEqual(
            self.bc._friendly_device_name("Realtek Audio, MME"), "Realtek Audio")

    def test_empty_returns_empty(self):
        self.assertEqual(self.bc._friendly_device_name(""), "")


# ───────────────────────────────────────────────────────────────────────────
#  Speech dedupe window  (_speech_was_recently_spoken / _mark_speech_spoken)
# ───────────────────────────────────────────────────────────────────────────
class SpeechDedupeTests(_MonolithSec2Base):
    def setUp(self):
        # Start each test from an empty dedupe table; restore afterwards.
        with self.bc._recent_spoken_lock:
            self._saved = dict(self.bc._recent_spoken_messages)
            self.bc._recent_spoken_messages.clear()

    def tearDown(self):
        with self.bc._recent_spoken_lock:
            self.bc._recent_spoken_messages.clear()
            self.bc._recent_spoken_messages.update(self._saved)

    def test_unseen_message_not_recent(self):
        self.assertFalse(self.bc._speech_was_recently_spoken("hello sam"))

    def test_marked_message_is_recent(self):
        self.bc._mark_speech_spoken("hello sam")
        self.assertTrue(self.bc._speech_was_recently_spoken("hello sam"))

    def test_distinct_messages_independent(self):
        self.bc._mark_speech_spoken("alpha")
        self.assertTrue(self.bc._speech_was_recently_spoken("alpha"))
        self.assertFalse(self.bc._speech_was_recently_spoken("beta"))

    def test_expired_entry_pruned(self):
        # Insert a stale timestamp directly, then a check should prune it.
        with self.bc._recent_spoken_lock:
            self.bc._recent_spoken_messages["old"] = (
                time.time() - self.bc._RECENT_SPEECH_DEDUPE_WINDOW - 5)
        self.assertFalse(self.bc._speech_was_recently_spoken("trigger-prune"))
        with self.bc._recent_spoken_lock:
            self.assertNotIn("old", self.bc._recent_spoken_messages)


# ───────────────────────────────────────────────────────────────────────────
#  _detect_face  (synthetic frames; the cascade is mocked so detection is
#  deterministic and CPU-free)
# ───────────────────────────────────────────────────────────────────────────
class DetectFaceTests(_MonolithSec2Base):
    def _frame(self):
        import numpy as np
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_no_cascade_returns_none(self):
        with mock.patch.object(self.bc, "_face_cascade", None):
            self.assertIsNone(self.bc._detect_face(self._frame()))

    def test_largest_face_centre_normalised(self):
        import numpy as np
        # Frontal cascade returns one box on the FIRST call; pick its centre.
        # box = (x=160, y=120, w=320, h=240) → centre (320, 240) in 640x480
        #   → fx=0.5, fy=0.5
        fake = mock.Mock()
        fake.detectMultiScale.return_value = np.array([[160, 120, 320, 240]])
        with mock.patch.object(self.bc, "_face_cascade", fake), \
                mock.patch.object(self.bc, "_profile_cascade", None), \
                mock.patch.object(self.bc, "MIRROR_EYES_X", False), \
                mock.patch.object(self.bc, "MIRROR_EYES_Y", False):
            out = self.bc._detect_face(self._frame())
        self.assertIsNotNone(out)
        fx, fy = out
        self.assertAlmostEqual(fx, 0.5, places=3)
        self.assertAlmostEqual(fy, 0.5, places=3)

    def test_picks_biggest_of_several(self):
        import numpy as np
        fake = mock.Mock()
        # Two boxes; the second (area 200*200) is larger than the first
        # (40*40). Its centre is at (100+100, 100+100)=(200,200) → (0.3125,
        # 0.4166...).
        fake.detectMultiScale.return_value = np.array(
            [[0, 0, 40, 40], [100, 100, 200, 200]])
        with mock.patch.object(self.bc, "_face_cascade", fake), \
                mock.patch.object(self.bc, "_profile_cascade", None), \
                mock.patch.object(self.bc, "MIRROR_EYES_X", False), \
                mock.patch.object(self.bc, "MIRROR_EYES_Y", False):
            fx, fy = self.bc._detect_face(self._frame())
        self.assertAlmostEqual(fx, 200 / 640, places=3)
        self.assertAlmostEqual(fy, 200 / 480, places=3)

    def test_mirror_x_flips(self):
        import numpy as np
        fake = mock.Mock()
        fake.detectMultiScale.return_value = np.array([[0, 0, 64, 64]])
        # centre fx = 32/640 = 0.05 → mirrored → 0.95
        with mock.patch.object(self.bc, "_face_cascade", fake), \
                mock.patch.object(self.bc, "_profile_cascade", None), \
                mock.patch.object(self.bc, "MIRROR_EYES_X", True), \
                mock.patch.object(self.bc, "MIRROR_EYES_Y", False):
            fx, fy = self.bc._detect_face(self._frame())
        self.assertAlmostEqual(fx, 1.0 - (32 / 640), places=3)

    def test_no_detection_returns_none(self):
        import numpy as np
        fake = mock.Mock()
        fake.detectMultiScale.return_value = np.empty((0, 4))
        with mock.patch.object(self.bc, "_face_cascade", fake), \
                mock.patch.object(self.bc, "_profile_cascade", None):
            self.assertIsNone(self.bc._detect_face(self._frame()))

    def test_profile_fallback_used_when_frontal_empty(self):
        import numpy as np
        frontal = mock.Mock()
        frontal.detectMultiScale.return_value = np.empty((0, 4))
        profile = mock.Mock()
        # profile cascade finds one face on its (first, non-mirrored) call
        profile.detectMultiScale.return_value = np.array([[200, 200, 80, 80]])
        with mock.patch.object(self.bc, "_face_cascade", frontal), \
                mock.patch.object(self.bc, "_profile_cascade", profile), \
                mock.patch.object(self.bc, "MIRROR_EYES_X", False), \
                mock.patch.object(self.bc, "MIRROR_EYES_Y", False):
            out = self.bc._detect_face(self._frame())
        self.assertIsNotNone(out)
        profile.detectMultiScale.assert_called()


# ───────────────────────────────────────────────────────────────────────────
#  _devices_signature / _input_openable / _pick_device  (sounddevice mocked)
# ───────────────────────────────────────────────────────────────────────────
class DeviceSelectionTests(_MonolithSec2Base):
    DEVICES = [
        {"name": "Microphone (USB Mic), MME",
         "max_input_channels": 1, "max_output_channels": 0,
         "default_samplerate": 16000},
        {"name": "Speakers (Realtek), MME",
         "max_input_channels": 0, "max_output_channels": 2,
         "default_samplerate": 48000},
    ]

    def _sd(self, **over):
        sd = mock.Mock()
        sd.query_devices.return_value = list(self.DEVICES)
        sd.check_input_settings.return_value = None
        for k, v in over.items():
            setattr(sd, k, v)
        return sd

    def test_devices_signature_tuple_shape(self):
        with mock.patch.object(self.bc, "sd", self._sd()):
            sig = self.bc._devices_signature()
        self.assertEqual(
            sig,
            ((0, "Microphone (USB Mic), MME", 1, 0),
             (1, "Speakers (Realtek), MME", 0, 2)))

    def test_devices_signature_none_on_error(self):
        sd = mock.Mock()
        sd.query_devices.side_effect = RuntimeError("portaudio down")
        with mock.patch.object(self.bc, "sd", sd):
            self.assertIsNone(self.bc._devices_signature())

    def test_input_openable_true(self):
        with mock.patch.object(self.bc, "sd", self._sd()):
            self.assertTrue(self.bc._input_openable(0))

    def test_input_openable_false_on_raise(self):
        sd = self._sd()
        sd.check_input_settings.side_effect = Exception("format not supported")
        with mock.patch.object(self.bc, "sd", sd):
            self.assertFalse(self.bc._input_openable(0))

    def test_pick_input_device_match(self):
        with mock.patch.object(self.bc, "sd", self._sd()):
            idx, name = self.bc._pick_device(["USB Mic"], want_input=True)
        self.assertEqual(idx, 0)
        self.assertIn("USB Mic", name)

    def test_pick_output_device_match(self):
        with mock.patch.object(self.bc, "sd", self._sd()):
            idx, name = self.bc._pick_device(["Realtek"], want_input=False)
        self.assertEqual(idx, 1)

    def test_pick_device_no_match_returns_none(self):
        with mock.patch.object(self.bc, "sd", self._sd()):
            idx, name = self.bc._pick_device(["Nonexistent"], want_input=True)
        self.assertIsNone(idx)
        self.assertEqual(name, "")

    def test_pick_input_skips_unopenable_match(self):
        # First preferred matches but is NOT openable → keep scanning. Add a
        # second openable device that matches a later preference.
        devices = [
            {"name": "Microphone (WDM-KS Dud)",
             "max_input_channels": 1, "max_output_channels": 0,
             "default_samplerate": 16000},
            {"name": "Microphone (Good USB)",
             "max_input_channels": 1, "max_output_channels": 0,
             "default_samplerate": 16000},
        ]
        sd = mock.Mock()
        sd.query_devices.return_value = devices
        # Dud (idx 0) raises; Good (idx 1) is fine.
        sd.check_input_settings.side_effect = (
            lambda device, **kw: (_ for _ in ()).throw(Exception("nope"))
            if device == 0 else None)
        with mock.patch.object(self.bc, "sd", sd):
            idx, name = self.bc._pick_device(
                ["WDM-KS Dud", "Good USB"], want_input=True)
        self.assertEqual(idx, 1)
        self.assertIn("Good USB", name)

    def test_pick_device_query_failure_returns_none(self):
        sd = mock.Mock()
        sd.query_devices.side_effect = Exception("boom")
        with mock.patch.object(self.bc, "sd", sd):
            idx, name = self.bc._pick_device(["x"], want_input=True)
        self.assertIsNone(idx)
        self.assertEqual(name, "")


# ───────────────────────────────────────────────────────────────────────────
#  _mic_input_disabled  (staging gate + negative-index gate)
# ───────────────────────────────────────────────────────────────────────────
class MicInputDisabledTests(_MonolithSec2Base):
    def test_disabled_when_staging(self):
        with mock.patch.object(self.bc, "_is_staging", return_value=True):
            self.assertTrue(self.bc._mic_input_disabled())

    def test_negative_index_disables(self):
        with mock.patch.object(self.bc, "_is_staging", return_value=False), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", -1):
            self.assertTrue(self.bc._mic_input_disabled())

    def test_normal_index_enabled(self):
        with mock.patch.object(self.bc, "_is_staging", return_value=False), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", 3):
            self.assertFalse(self.bc._mic_input_disabled())

    def test_none_index_enabled(self):
        with mock.patch.object(self.bc, "_is_staging", return_value=False), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None):
            self.assertFalse(self.bc._mic_input_disabled())


# ───────────────────────────────────────────────────────────────────────────
#  get_input_device / get_output_device / get_current_*_name
# ───────────────────────────────────────────────────────────────────────────
class DeviceAccessorTests(_MonolithSec2Base):
    def setUp(self):
        # Snapshot the device cache so each test starts clean.
        self._saved_cache = dict(self.bc._device_cache)

    def tearDown(self):
        self.bc._device_cache.clear()
        self.bc._device_cache.update(self._saved_cache)

    def test_get_input_device_disabled_returns_none(self):
        with mock.patch.object(self.bc, "_mic_input_disabled",
                               return_value=True):
            self.assertIsNone(self.bc.get_input_device())

    def test_get_input_device_returns_cached_index(self):
        sd = mock.Mock()
        sd.query_devices.return_value = {"name": "USB Mic"}
        self.bc._device_cache["in"] = 2
        with mock.patch.object(self.bc, "_mic_input_disabled",
                               return_value=False), \
                mock.patch.object(self.bc, "_refresh_devices"), \
                mock.patch.object(self.bc, "sd", sd):
            self.assertEqual(self.bc.get_input_device(), 2)

    def test_get_input_device_none_when_cache_none(self):
        self.bc._device_cache["in"] = None
        with mock.patch.object(self.bc, "_mic_input_disabled",
                               return_value=False), \
                mock.patch.object(self.bc, "_refresh_devices"):
            self.assertIsNone(self.bc.get_input_device())

    def test_get_input_device_stale_index_clears_cache(self):
        sd = mock.Mock()
        sd.query_devices.side_effect = Exception("Error querying device 5")
        self.bc._device_cache["in"] = 5
        with mock.patch.object(self.bc, "_mic_input_disabled",
                               return_value=False), \
                mock.patch.object(self.bc, "_refresh_devices"), \
                mock.patch.object(self.bc, "sd", sd):
            self.assertIsNone(self.bc.get_input_device())
        self.assertIsNone(self.bc._device_cache["in"])
        self.assertEqual(self.bc._device_cache["checked_at"], 0.0)

    def test_get_output_device_returns_cached_index(self):
        sd = mock.Mock()
        sd.query_devices.return_value = {"name": "Realtek"}
        self.bc._device_cache["out"] = 7
        with mock.patch.object(self.bc, "_refresh_devices"), \
                mock.patch.object(self.bc, "sd", sd):
            self.assertEqual(self.bc.get_output_device(), 7)

    def test_get_output_device_stale_index_clears_cache(self):
        sd = mock.Mock()
        sd.query_devices.side_effect = Exception("gone")
        self.bc._device_cache["out"] = 9
        with mock.patch.object(self.bc, "_refresh_devices"), \
                mock.patch.object(self.bc, "sd", sd):
            self.assertIsNone(self.bc.get_output_device())
        self.assertIsNone(self.bc._device_cache["out"])

    def test_get_current_mic_name_system_default(self):
        self.bc._device_cache["in"] = None
        with mock.patch.object(self.bc, "_refresh_devices"):
            self.assertEqual(self.bc.get_current_mic_name(), "(system default)")

    def test_get_current_mic_name_with_index(self):
        sd = mock.Mock()
        sd.query_devices.return_value = {"name": "Blue Yeti"}
        self.bc._device_cache["in"] = 4
        with mock.patch.object(self.bc, "_refresh_devices"), \
                mock.patch.object(self.bc, "sd", sd):
            self.assertEqual(self.bc.get_current_mic_name(), "[4] Blue Yeti")

    def test_get_current_mic_name_unknown_on_error(self):
        sd = mock.Mock()
        sd.query_devices.side_effect = Exception("x")
        self.bc._device_cache["in"] = 4
        with mock.patch.object(self.bc, "_refresh_devices"), \
                mock.patch.object(self.bc, "sd", sd):
            self.assertEqual(self.bc.get_current_mic_name(), "[4] (unknown)")

    def test_get_current_speaker_name_system_default(self):
        self.bc._device_cache["out"] = None
        self.assertEqual(self.bc.get_current_speaker_name(), "(system default)")

    def test_get_current_speaker_name_with_index(self):
        sd = mock.Mock()
        sd.query_devices.return_value = {"name": "Realtek"}
        self.bc._device_cache["out"] = 1
        with mock.patch.object(self.bc, "sd", sd):
            self.assertEqual(self.bc.get_current_speaker_name(), "[1] Realtek")


# ───────────────────────────────────────────────────────────────────────────
#  _refresh_devices  (the destructive-reinit guards + change announcement)
# ───────────────────────────────────────────────────────────────────────────
class RefreshDevicesTests(_MonolithSec2Base):
    def setUp(self):
        self._saved_cache = dict(self.bc._device_cache)

    def tearDown(self):
        self.bc._device_cache.clear()
        self.bc._device_cache.update(self._saved_cache)

    def test_time_gate_skips_when_recent(self):
        # checked_at = now → within DEVICE_CHECK_INTERVAL → early return, no
        # query at all.
        self.bc._device_cache["checked_at"] = time.time()
        sd = mock.Mock()
        with mock.patch.object(self.bc, "sd", sd):
            self.bc._refresh_devices(force=False)
        sd._terminate.assert_not_called()

    def test_unchanged_signature_skips_reinit(self):
        # force=False, signature identical to last, cache POPULATED, periodic
        # re-enum not due → bump checked_at only; no _terminate/_initialize
        # and no re-pick. (The pristine cache has last_reenum_at=0.0, which
        # makes the periodic re-enum due and forces a fall-through — pin it
        # fresh so this test exercises the short-circuit itself. A CLEARED
        # cache deliberately no longer short-circuits: see the
        # cleared-cache re-pick tests below. 2026-07-21 audit.)
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["in"] = 2
        self.bc._device_cache["out"] = 4
        self.bc._device_cache["last_reenum_at"] = time.time()
        sig = ((0, "Mic", 1, 0),)
        self.bc._device_cache["last_devices_signature"] = sig
        with mock.patch.object(self.bc, "_devices_signature", return_value=sig):
            sd = mock.Mock()
            with mock.patch.object(self.bc, "sd", sd), \
                    mock.patch.object(self.bc, "_pick_device") as pick:
                self.bc._refresh_devices(force=False)
            sd._terminate.assert_not_called()
            pick.assert_not_called()
        self.assertGreater(self.bc._device_cache["checked_at"], 0.0)
        self.assertEqual(self.bc._device_cache["in"], 2)
        self.assertEqual(self.bc._device_cache["out"], 4)

    def test_cleared_cache_repicks_without_reinit(self):
        # THE 2026-07-21 regression ("Mic-device cache invalidation is a
        # no-op"): an invalidating call site cleared the cache (in/out=None,
        # checked_at=0.0) but the device signature is unchanged → the old
        # gate returned early and the re-pick never ran, so
        # get_input_device() kept returning None (= the system-default mic)
        # for up to DEVICE_REENUM_INTERVAL. The cleared cache must force the
        # cheap re-pick — WITHOUT the destructive reinit and WITHOUT pausing
        # the wake-word detector.
        sig = ((0, "Mic", 1, 0),)
        self.bc._device_cache.update({
            "in": None, "out": None, "checked_at": 0.0,
            "last_devices_signature": sig,
            "last_reenum_at": time.time(),   # periodic re-enum NOT due
        })
        sd = mock.Mock()
        det = mock.Mock()
        det.is_running.return_value = True
        wl = mock.Mock(_detector=det)
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=sig), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_pathb_mic_active", [False]), \
                mock.patch.object(self.bc, "_ambient_stream_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_wake_listener": wl}), \
                mock.patch.object(self.bc, "_pick_device",
                                  side_effect=[(3, "Mic X"), (5, "Spk Y")]):
            self.bc._refresh_devices(force=False)
        # The re-pick RAN (this is the assertion that fails at HEAD)…
        self.assertEqual(self.bc._device_cache["in"], 3)
        self.assertEqual(self.bc._device_cache["out"], 5)
        # …via the repick-only path: no destructive reinit, no wake-word pause.
        sd._terminate.assert_not_called()
        det.pause.assert_not_called()

    def test_cleared_cache_no_preferred_device_never_reinit_churns(self):
        # Churn guard: a permanently unpickable preferred device (headset
        # unplugged, no fallback match) leaves the cache legitimately None on
        # every pass. That steady state must NOT convert each
        # DEVICE_CHECK_INTERVAL pass into a destructive sd._terminate()
        # cycle — the 0xc0000374 churn the signature gate exists to prevent.
        sig = ((0, "Mic", 1, 0),)
        self.bc._device_cache.update({
            "in": None, "out": None, "checked_at": 0.0,
            "last_devices_signature": sig,
            "last_reenum_at": time.time(),
        })
        sd = mock.Mock()
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=sig), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_pathb_mic_active", [False]), \
                mock.patch.object(self.bc, "_ambient_stream_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "_pick_device",
                                  return_value=(None, "")):
            self.bc._refresh_devices(force=False)
            self.bc._device_cache["checked_at"] = 0.0   # re-arm the time gate
            self.bc._refresh_devices(force=False)
        sd._terminate.assert_not_called()

    def test_invalidated_cache_recovers_on_next_get_input_device(self):
        # End-to-end: get_input_device()'s stale-index probe invalidates the
        # cache; the NEXT get_input_device() must return the re-picked index
        # instead of None (the pre-fix behaviour silently captured on the
        # system-default mic until the 300s forced re-enumeration).
        sig = ((0, "Mic", 1, 0),)
        self.bc._device_cache.update({
            "in": 5, "out": 6, "checked_at": time.time(),
            "last_devices_signature": sig,
            "last_reenum_at": time.time(),
        })
        sd = mock.Mock()

        def _query(idx=None):
            if idx == 5:
                raise Exception("Error querying device 5")
            return {"name": "Mic X"}
        sd.query_devices.side_effect = _query
        with mock.patch.object(self.bc, "_mic_input_disabled",
                               return_value=False), \
                mock.patch.object(self.bc, "sd", sd), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=sig), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_pathb_mic_active", [False]), \
                mock.patch.object(self.bc, "_ambient_stream_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "_pick_device",
                                  side_effect=[(3, "Mic X"), (6, "Spk Y")]):
            # Stale probe → invalidate → fall back to system default once…
            self.assertIsNone(self.bc.get_input_device())
            # …but the NEXT call re-picks instead of staying on None.
            self.assertEqual(self.bc.get_input_device(), 3)
        sd._terminate.assert_not_called()

    def test_inlock_recheck_returns_when_peer_just_refreshed(self):
        # 3872-3873: the pre-lock time gate passes (checked_at stale), but a
        # peer thread refreshes the cache during the _devices_signature()
        # snapshot — modelled by stamping checked_at fresh from that mock — so
        # the in-lock re-check short-circuits before any query/_terminate.
        self.bc._device_cache["checked_at"] = 0.0
        sd = mock.Mock()

        def _sig_then_peer_refresh():
            # Simulate the race: another caller refreshed while we snapshotted.
            self.bc._device_cache["checked_at"] = time.time()
            return ((0, "Mic", 1, 0),)
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.object(self.bc, "_devices_signature",
                                  side_effect=_sig_then_peer_refresh):
            self.bc._refresh_devices(force=False)
        sd.query_devices.assert_not_called()
        sd._terminate.assert_not_called()

    def test_record_speech_active_defers_reinit(self):
        # Drift present (force=True bypasses sig short-circuit) but record_speech
        # owns the mic → the destructive sd._terminate() must be skipped.
        self.bc._device_cache["checked_at"] = 0.0
        sd = mock.Mock()
        sd.query_devices.return_value = {"name": "X"}
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.object(self.bc, "_record_speech_active", [True]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", 0), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", 0), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None):
            self.bc._refresh_devices(force=True)
        sd._terminate.assert_not_called()

    def test_tts_playback_active_defers_reinit(self):
        self.bc._device_cache["checked_at"] = 0.0
        sd = mock.Mock()
        sd.query_devices.return_value = {"name": "X"}
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [True]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", 0), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", 0), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None):
            self.bc._refresh_devices(force=True)
        sd._terminate.assert_not_called()

    def test_idle_path_performs_reinit_and_picks(self):
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = None
        self.bc._device_cache["last_out_name"] = None
        sd = mock.Mock()
        sd.query_devices.return_value = {"name": "ignored"}
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None), \
                mock.patch.object(self.bc, "_pick_device",
                                  side_effect=[(0, "USB Mic"), (1, "Realtek")]):
            self.bc._refresh_devices(force=True)
        sd._terminate.assert_called_once()
        sd._initialize.assert_called_once()
        self.assertEqual(self.bc._device_cache["in"], 0)
        self.assertEqual(self.bc._device_cache["out"], 1)

    def test_mic_switch_enqueues_announcement(self):
        # A genuine mid-session mic change (prev non-None → new name) should
        # call _enqueue_device_announcement. (The device list is unreadable
        # through this bare Mock, so loss is UNKNOWN and the neutral wording
        # is used — see FollowTheDefaultDeviceTests for both wordings.)
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = "Gaming Headset"
        self.bc._device_cache["last_out_name"] = None
        announced = []
        with mock.patch.object(self.bc, "sd", mock.Mock()), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None), \
                mock.patch.object(self.bc, "_pick_device",
                                  side_effect=[(0, "Fallback Laptop Mic"),
                                               (1, "Speakers")]), \
                mock.patch.object(self.bc, "_enqueue_device_announcement",
                                  side_effect=announced.append):
            self.bc._refresh_devices(force=True)
        self.assertEqual(len(announced), 1)
        self.assertIn("Switched to", announced[0])


# ───────────────────────────────────────────────────────────────────────────
#  proactive_announce / _enqueue_device_announcement  (real temp queue file)
# ───────────────────────────────────────────────────────────────────────────
class ProactiveAnnounceTests(_MonolithSec2Base):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jarvis_sec2_")
        # proactive_announce derives its queue path from
        # os.path.dirname(os.path.abspath(__file__)) inside bobert_companion.
        # Patch the module's __file__ so the queue lands in our temp dir.
        self._file_patch = mock.patch.object(
            self.bc, "__file__", os.path.join(self.tmp, "bobert_companion.py"))
        self._file_patch.start()
        self.queue = os.path.join(self.tmp, "pending_speech.json")

    def tearDown(self):
        self._file_patch.stop()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read_queue(self):
        with open(self.queue, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_enqueue_writes_entry(self):
        self.assertTrue(self.bc.proactive_announce("print is done"))
        data = self._read_queue()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["message"], "print is done")
        self.assertIn("ts", data[0])

    def test_mood_recorded(self):
        self.bc.proactive_announce("alert", mood="urgent_clipped")
        self.assertEqual(self._read_queue()[0]["mood"], "urgent_clipped")

    def test_volume_scale_recorded_only_when_nondefault(self):
        self.bc.proactive_announce("whisper", volume_scale=0.4)
        self.bc.proactive_announce("normal")
        data = self._read_queue()
        self.assertEqual(data[0]["volume_scale"], 0.4)
        self.assertNotIn("volume_scale", data[1])

    def test_appends_to_existing_queue(self):
        self.bc.proactive_announce("one")
        self.bc.proactive_announce("two")
        data = self._read_queue()
        self.assertEqual([d["message"] for d in data], ["one", "two"])

    def test_queue_capped_at_50(self):
        # Seed 60 entries directly, then one more enqueue trims to 50.
        seed = [{"ts": 0.0, "message": f"m{i}"} for i in range(60)]
        with open(self.queue, "w", encoding="utf-8") as f:
            json.dump(seed, f)
        self.bc.proactive_announce("newest")
        data = self._read_queue()
        self.assertEqual(len(data), 50)
        self.assertEqual(data[-1]["message"], "newest")

    def test_corrupt_existing_file_treated_as_empty(self):
        with open(self.queue, "w", encoding="utf-8") as f:
            f.write("{not json at all")
        self.assertTrue(self.bc.proactive_announce("recovered"))
        data = self._read_queue()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["message"], "recovered")

    def test_returns_false_and_prints_on_write_failure(self):
        # Make os.replace raise so the write path fails; should return False,
        # not propagate.
        with mock.patch.object(self.bc.os, "replace",
                               side_effect=OSError("read-only share")):
            self.assertFalse(self.bc.proactive_announce("doomed", source="x"))

    def test_enqueue_device_announcement_routes_through(self):
        with mock.patch.object(self.bc, "proactive_announce",
                               return_value=True) as pa:
            self.bc._enqueue_device_announcement("mic swapped")
        pa.assert_called_once()
        # source tag is the dedicated [audio] one
        self.assertEqual(pa.call_args.kwargs.get("source"), "audio")


# ───────────────────────────────────────────────────────────────────────────
#  find_camera_locking_processes / probe helpers  (psutil + cv2 mocked)
# ───────────────────────────────────────────────────────────────────────────
class CameraLockProcessTests(_MonolithSec2Base):
    def test_no_psutil_returns_empty(self):
        # Force `import psutil` inside the function to raise ImportError.
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            self.assertEqual(self.bc.find_camera_locking_processes(), [])

    def test_detects_known_lock_holder(self):
        fake_psutil = mock.Mock()

        class _Proc:
            def __init__(self, nm):
                self.info = {"name": nm}

        fake_psutil.process_iter.return_value = [
            _Proc("teams.exe"), _Proc("notepad.exe")]
        fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "psutil":
                return fake_psutil
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch.object(self.bc, "CAMERA_LOCK_PROCESSES",
                                  {"teams.exe", "zoom.exe"}):
            out = self.bc.find_camera_locking_processes()
        self.assertEqual(out, ["teams.exe"])


class ProbeCameraIndexTests(_MonolithSec2Base):
    def test_returns_true_when_frame_read(self):
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, object())
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        with mock.patch.object(self.bc, "cv2", cv2):
            self.assertTrue(self.bc._probe_camera_index(0, timeout_sec=2.0))
        cap.release.assert_called()

    def test_returns_false_when_not_opened(self):
        cap = mock.Mock()
        cap.isOpened.return_value = False
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        with mock.patch.object(self.bc, "cv2", cv2):
            self.assertFalse(self.bc._probe_camera_index(3, timeout_sec=2.0))

    def test_returns_false_when_read_fails(self):
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        with mock.patch.object(self.bc, "cv2", cv2):
            self.assertFalse(self.bc._probe_camera_index(0, timeout_sec=2.0))


class ProbeCamerasAndUpdateConfigTests(_MonolithSec2Base):
    def setUp(self):
        self._saved_cams = [dict(c) for c in self.bc.CAMERAS]

    def tearDown(self):
        self.bc.CAMERAS[:] = self._saved_cams

    def test_disabled_returns_empty_pair(self):
        with mock.patch.object(self.bc, "CAMERA_PROBE_ENABLED", False):
            self.assertEqual(
                self.bc.probe_cameras_and_update_config(), ([], []))

    def test_configured_cameras_work_keeps_config(self):
        cams = [{"index": 1, "label": "L", "primary": False,
                 "look_x": 0.15, "look_y": 0.5},
                {"index": 0, "label": "R", "primary": True,
                 "look_x": 0.85, "look_y": 0.5}]
        with mock.patch.object(self.bc, "CAMERA_PROBE_ENABLED", True), \
                mock.patch.object(self.bc, "CAMERAS", cams), \
                mock.patch.object(self.bc, "_probe_camera_index",
                                  return_value=True):
            working, failed = self.bc.probe_cameras_and_update_config()
        self.assertCountEqual(working, [1, 0])
        self.assertEqual(failed, [])

    def test_lock_holder_short_circuits_sweep(self):
        cams = [{"index": 1, "label": "L", "primary": False,
                 "look_x": 0.15, "look_y": 0.5}]
        with mock.patch.object(self.bc, "CAMERA_PROBE_ENABLED", True), \
                mock.patch.object(self.bc, "CAMERAS", cams), \
                mock.patch.object(self.bc, "_probe_camera_index",
                                  return_value=False), \
                mock.patch.object(self.bc, "find_camera_locking_processes",
                                  return_value=["teams.exe"]):
            working, failed = self.bc.probe_cameras_and_update_config()
        self.assertEqual(working, [])
        self.assertEqual(failed, [1])

    def test_fallback_sweep_finds_camera_rewrites_config(self):
        cams = [{"index": 5, "label": "L", "primary": True,
                 "look_x": 0.15, "look_y": 0.5}]

        # Configured idx 5 fails; sweep finds idx 2 only.
        def probe(i, *a, **k):
            return i == 2

        with mock.patch.object(self.bc, "CAMERA_PROBE_ENABLED", True), \
                mock.patch.object(self.bc, "CAMERA_PROBE_MAX", 4), \
                mock.patch.object(self.bc, "CAMERAS", cams), \
                mock.patch.object(self.bc, "_probe_camera_index",
                                  side_effect=probe), \
                mock.patch.object(self.bc, "find_camera_locking_processes",
                                  return_value=[]):
            working, failed = self.bc.probe_cameras_and_update_config()
            # The function rewrites CAMERAS in-place (CAMERAS[:] = ...). Capture
            # the rewritten list WHILE the patch is active — patch.object
            # restores the original bc.CAMERAS once the `with` block exits.
            rewritten = list(self.bc.CAMERAS)
        self.assertIn(2, working)
        # CAMERAS rewritten with the found index, marked primary.
        self.assertEqual(rewritten[0]["index"], 2)
        self.assertTrue(rewritten[0]["primary"])


# ───────────────────────────────────────────────────────────────────────────
#  get_monitors / list_monitors_cli  (ctypes / Win32 mocked)
# ───────────────────────────────────────────────────────────────────────────
class MonitorTests(_MonolithSec2Base):
    def test_get_monitors_non_windows_returns_empty(self):
        with mock.patch.object(self.bc.sys, "platform", "linux"):
            self.assertEqual(self.bc.get_monitors(), [])

    def test_list_monitors_cli_no_monitors(self):
        with mock.patch.object(self.bc, "get_monitors", return_value=[]):
            # Should print the "no monitors" line and return cleanly.
            self.bc.list_monitors_cli()

    def test_list_monitors_cli_with_monitors(self):
        mons = [(0, 0, 1920, 1080), (1920, 0, 2560, 1440), (-1920, 0, 1920, 1080)]
        with mock.patch.object(self.bc, "get_monitors", return_value=mons):
            # Exercises the position-guess branches without raising.
            self.bc.list_monitors_cli()

    def test_get_monitors_real_win32_enum(self):
        # Cover the ctypes EnumDisplayMonitors callback body on Windows. This
        # is a read-only Win32 enumeration (no device mutation). Skip off-win32.
        if self.bc.sys.platform != "win32":
            self.skipTest("Win32-only monitor enumeration")
        mons = self.bc.get_monitors()
        self.assertIsInstance(mons, list)
        for m in mons:
            self.assertEqual(len(m), 4)
            self.assertTrue(all(isinstance(v, int) for v in m))


# ───────────────────────────────────────────────────────────────────────────
#  list_microphones / list_speakers  (sounddevice mocked)
# ───────────────────────────────────────────────────────────────────────────
class ListAudioDevicesTests(_MonolithSec2Base):
    DEVICES = [
        {"name": "USB Mic", "max_input_channels": 1,
         "max_output_channels": 0, "default_samplerate": 16000},
        {"name": "Realtek Speakers", "max_input_channels": 0,
         "max_output_channels": 2, "default_samplerate": 48000},
    ]

    def _sd(self):
        sd = mock.Mock()
        sd.query_devices.return_value = list(self.DEVICES)
        sd.default.device = [0, 1]
        return sd

    def test_list_microphones_runs(self):
        with mock.patch.object(self.bc, "sd", self._sd()):
            self.bc.list_microphones()

    def test_list_speakers_runs(self):
        with mock.patch.object(self.bc, "sd", self._sd()):
            self.bc.list_speakers()

    def test_list_microphones_no_default(self):
        sd = self._sd()
        sd.default.device = None
        with mock.patch.object(self.bc, "sd", sd):
            self.bc.list_microphones()


# ───────────────────────────────────────────────────────────────────────────
#  list_cameras  (cv2 + threads mocked; no real device opened)
# ───────────────────────────────────────────────────────────────────────────
class ListCamerasTests(_MonolithSec2Base):
    def test_list_cameras_writes_previews(self):
        import numpy as np
        frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, frame)
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        tmp = tempfile.mkdtemp(prefix="jarvis_cam_")
        try:
            with mock.patch.object(self.bc, "cv2", cv2), \
                    mock.patch.object(self.bc, "find_camera_locking_processes",
                                      return_value=[]), \
                    mock.patch.object(self.bc.time, "sleep"), \
                    mock.patch.object(self.bc.os.path, "dirname",
                                      return_value=tmp):
                # Only check index 0 to keep it fast.
                self.bc.list_cameras(max_check=1)
            cv2.imwrite.assert_called()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_list_cameras_handles_no_camera(self):
        cap = mock.Mock()
        cap.isOpened.return_value = False
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        tmp = tempfile.mkdtemp(prefix="jarvis_cam_")
        try:
            with mock.patch.object(self.bc, "cv2", cv2), \
                    mock.patch.object(self.bc, "find_camera_locking_processes",
                                      return_value=[]), \
                    mock.patch.object(self.bc.time, "sleep"), \
                    mock.patch.object(self.bc.os.path, "dirname",
                                      return_value=tmp):
                self.bc.list_cameras(max_check=1)
            cv2.imwrite.assert_not_called()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


# ───────────────────────────────────────────────────────────────────────────
#  pause_face_tracking / resume_face_tracking
# ───────────────────────────────────────────────────────────────────────────
class FaceTrackingToggleTests(_MonolithSec2Base):
    def tearDown(self):
        # Leave both events clear (their boot defaults).
        self.bc._face_track_pause.clear()
        self.bc._face_track_camera_off.clear()

    def test_pause_sets_event(self):
        self.bc._face_track_pause.clear()
        self.bc.pause_face_tracking()
        self.assertTrue(self.bc._face_track_pause.is_set())

    def test_resume_clears_event(self):
        self.bc._face_track_pause.set()
        self.bc.resume_face_tracking()
        self.assertFalse(self.bc._face_track_pause.is_set())

    def test_pause_does_not_set_camera_off(self):
        # Pausing for voice must NOT flip the genuine camera-off flag — that is
        # what keeps the camera visibly ON while JARVIS listens/speaks.
        self.bc._face_track_camera_off.clear()
        self.bc.pause_face_tracking()
        self.assertFalse(self.bc._face_track_camera_off.is_set())

    def test_set_clear_camera_off(self):
        self.bc._face_track_camera_off.clear()
        self.bc.set_face_tracking_camera_off()
        self.assertTrue(self.bc._face_track_camera_off.is_set())
        self.bc.clear_face_tracking_camera_off()
        self.assertFalse(self.bc._face_track_camera_off.is_set())


# ───────────────────────────────────────────────────────────────────────────
#  _face_tracking_thread  (no cameras → fast clean exit)
# ───────────────────────────────────────────────────────────────────────────
#
#  HARDWARE FIREWALL + WATCHDOG  (2026-08-20 isolation fix)
#  ────────────────────────────────────────────────────────
#  These tests used to open the REAL Kinect v2 and then spin the capture loop
#  forever: the process reached 130-144 GB of commit and BUGCHECKED the machine
#  (0x0000000A). Three facts combined:
#
#    1. data/user_settings.json sets KINECT_ENABLED=true, so merely IMPORTING
#       the monolith runs ``_kinect_bridge.set_enabled(True)``
#       (bobert_companion.py:557) -> ``start_body_pump()`` -> a daemon thread
#       that opens the real sensor and publishes it into the PROCESS-GLOBAL
#       ``audio.kinect_bridge._runtime[0]``.
#    2. data/user_settings.json also sets KINECT_AS_CAMERA=true, and
#       ``_open_capture`` computes (bobert_companion.py:5493)
#           want_kinect = (cam.get("type") == "kinect"
#                          or (KINECT_AS_CAMERA and not cam_name))
#       - so a CAMERAS entry with NO "name" key HIJACKS the mocked webcam and is
#       handed a real ``kinect_bridge.KinectCapture`` instead.
#    3. Whether that KinectCapture reports ``isOpened()`` depends purely on
#       WALL-CLOCK TIME: while the import-time pump is still inside its ~16 s
#       open gauntlet, ``_open_attempt_lock.acquire(timeout=0.5)`` fails and the
#       capture reports "Kinect open already in progress" -> closed -> caps
#       empty -> the "No cameras available" early return fires and every test
#       passes. Once the pump HAS published the runtime, the same code gets a
#       LIVE sensor -> caps non-empty -> the loop runs until the process dies,
#       while the bridge's stale-stream watchdog re-opens the runtime again and
#       again, leaking a full set of frame buffers each time.
#
#  So the failure was ORDER- and TIME-dependent, not a leaked global from an
#  earlier class. PROOF (capped runs, 4 GB job-object cap):
#    * DeviceAccessorExtraTests + DeviceAccessorTests + DeviceSelectionTests
#      then this class ............................. peak 968 MB, OK
#    * THE SAME THREE CLASSES REPEATED 3x then this class
#                                    ............... *** HIT CAP *** 4096 MB
#  Same classes, more elapsed time -> balloon. The contaminated global is
#  ``audio.kinect_bridge._runtime[0]``, filled in asynchronously by the
#  import-time body pump - nothing this module could "restore in a tearDown".
#
#  The fix is therefore defence in depth, so ANY execution order is safe:
#    * every CAMERAS entry here carries a "name" -> the KINECT_AS_CAMERA hijack
#      cannot fire (the rule _open_capture documents at bobert_companion.py:5485);
#    * KINECT_AS_CAMERA is patched False anyway;
#    * ``bc._kinect_bridge`` is swapped for ``_StubKinectBridge``, whose
#      KinectCapture NEVER opens - so no test can reach the real sensor even if
#      both of the above regress;
#    * ``_dshow_name_to_index`` is stubbed so a "name" never triggers a LIVE
#      pygrabber DirectShow enumeration;
#    * every call goes through ``_run_face_tracking_bounded()``, a watchdog that
#      sets ``_face_track_stop`` and FAILS the test if the thread has not
#      returned - an unbounded loop now asserts in seconds instead of allocating;
#    * the three face-track Events are cleared before AND after every test.
# ───────────────────────────────────────────────────────────────────────────
class _StubKinectCapture:
    """A ``kinect_bridge.KinectCapture`` work-alike that is ALWAYS closed.

    Mirrors the real contract (``isOpened``/``read``/``get``/``set``/``release``
    plus ``_open_error``, which ``_open_capture`` prints) so the opener takes its
    "Kinect requested but unavailable -> fall back to the webcam" branch WITHOUT
    touching the sensor. Deliberately a plain class and not a ``mock.Mock``: a
    Mock retains every call and every argument forever, which is itself an
    unbounded leak when the thing calling it is a per-frame capture loop."""

    def __init__(self):
        self._open_error = "kinect stubbed out for tests"

    def isOpened(self):             # noqa: N802 - mirrors cv2.VideoCapture
        return False

    def read(self):
        return False, None

    def get(self, _prop):
        return 0.0

    def set(self, *_a, **_k):
        return False

    def release(self):
        pass


class _StubKinectBridge:
    """Stand-in for ``bobert_companion._kinect_bridge`` in the face-track tests.

    Counts how many times ``_open_capture`` asked for a ``KinectCapture`` so a
    test can PROVE the KINECT_AS_CAMERA hijack did not fire."""

    def __init__(self):
        self.capture_requests = 0

    def KinectCapture(self):        # noqa: N802 - mirrors the bridge's API
        self.capture_requests += 1
        return _StubKinectCapture()


class FaceTrackingThreadTests(_MonolithSec2Base):
    # The single configured camera every test in this class uses. The "name"
    # key is LOAD-BEARING: without it, KINECT_AS_CAMERA (true on this box via
    # data/user_settings.json) makes _open_capture bypass the mocked webcam for
    # a real Kinect. Never drop it. See the module note above.
    CAM = {"index": 0, "label": "X", "name": "Test Webcam 0",
           "primary": True, "look_x": 0.5, "look_y": 0.5}

    def _cams(self):
        """A fresh one-entry CAMERAS list (copied, so an in-place rewrite by the
        code under test cannot mutate the class constant)."""
        return [dict(self.CAM)]

    def setUp(self):
        bc = self.bc
        # -- hardware firewall ------------------------------------------------
        # Started here rather than as per-test `with` blocks so EVERY test in
        # the class is covered, including any added later; addCleanup unwinds
        # them even when a test raises.
        for patcher in (
                # Belt: the global opt-in is off for the whole class.
                mock.patch.object(bc, "KINECT_AS_CAMERA", False),
                # Braces: even with it ON, the bridge can never open a sensor.
                mock.patch.object(bc, "_kinect_bridge", _StubKinectBridge()),
                # A "name" must not trigger a LIVE DirectShow enumeration.
                # Resolving to the same static index keeps the opener quiet.
                mock.patch.object(bc, "_dshow_name_to_index",
                                  side_effect=lambda _name: self.CAM["index"]),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        # -- deterministic event baseline -------------------------------------
        # No test may inherit a set/cleared face-track Event from a sibling, and
        # none may leak one into a later class (the harness restores globals but
        # NOT threading.Events).
        for ev in (bc._face_track_stop, bc._face_track_pause,
                   bc._face_track_camera_off):
            ev.clear()
            self.addCleanup(ev.clear)

    def _run_face_tracking_bounded(self, timeout=8.0):
        """Call ``bc._face_tracking_thread()`` under a WATCHDOG.

        REGRESSION GUARD. If the function has not returned within ``timeout``
        the capture loop is unbounded - a camera the test meant to be closed
        actually opened - so we SET ``_face_track_stop`` to break the loop, wait
        for the thread, and FAIL LOUDLY. Before this guard existed that exact
        condition hung the runner and allocated until the box bugchecked, which
        is unreadable as a test result; now it is a one-line assertion.

        Runs the function on its own thread (which is how production runs it),
        so the watchdog can regain control; any exception is re-raised on the
        calling thread so unittest still reports it normally."""
        box: dict = {}

        def _runner():
            try:
                self.bc._face_tracking_thread()
            except BaseException as exc:      # noqa: BLE001 - re-raised below
                box["exc"] = exc

        t = threading.Thread(target=_runner, name="sec2-face-track-watchdog",
                             daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            self.bc._face_track_stop.set()    # break the loop ...
            t.join(10.0)                      # ... and let it unwind cleanly
            self.fail(
                f"_face_tracking_thread did not return within {timeout}s - the "
                "capture loop is UNBOUNDED (a real camera/Kinect opened where "
                "the test expected none). Stopped via _face_track_stop; see the "
                "KINECT_AS_CAMERA note above this class.")
        exc = box.get("exc")
        if exc is not None:
            raise exc

    def test_no_cameras_returns_immediately(self):
        # _open_capture returns None for every configured cam -> caps empty ->
        # the thread prints "No cameras available" and returns without looping.
        cv2 = mock.Mock()
        bad_cap = mock.Mock()
        bad_cap.isOpened.return_value = False
        cv2.VideoCapture.return_value = bad_cap
        cv2.CAP_DSHOW = 700
        # Belt: even if a capture DID open, the loop must not run - the stop
        # event is armed before the call and cleared again by setUp/addCleanup.
        self.bc._face_track_stop.set()
        buf = io.StringIO()
        with mock.patch.object(self.bc, "cv2", cv2), \
                mock.patch.object(self.bc, "CAMERAS", self._cams()), \
                contextlib.redirect_stdout(buf):
            # Should return quickly with no surviving thread/loop.
            self._run_face_tracking_bounded()
        # PROVE the early-return branch actually ran. Asserting only "it
        # returned" was what let the balloon hide: with the stop event armed the
        # thread ALSO returns quickly after opening a real camera, so the branch
        # itself has to be observed.
        self.assertIn("No cameras available", buf.getvalue())
        # ...and the mocked webcam is what was tried, not the Kinect.
        cv2.VideoCapture.assert_called_once_with(0, 700)
        self.assertEqual(self.bc._kinect_bridge.capture_requests, 0)

    def test_kinect_as_camera_cannot_hijack_a_named_camera(self):
        # REGRESSION GUARD for the 144 GB balloon. With KINECT_AS_CAMERA ON, a
        # NAMED CAMERAS entry must still open the WEBCAM, never the Kinect -
        # the rule _open_capture documents at bobert_companion.py:5485. Every
        # CAMERAS dict in this class relies on it.
        bridge = _StubKinectBridge()
        cv2 = mock.Mock()
        bad_cap = mock.Mock()
        bad_cap.isOpened.return_value = False
        cv2.VideoCapture.return_value = bad_cap
        cv2.CAP_DSHOW = 700
        self.bc._face_track_stop.set()
        buf = io.StringIO()
        with mock.patch.object(self.bc, "KINECT_AS_CAMERA", True), \
                mock.patch.object(self.bc, "_kinect_bridge", bridge), \
                mock.patch.object(self.bc, "cv2", cv2), \
                mock.patch.object(self.bc, "CAMERAS", self._cams()), \
                contextlib.redirect_stdout(buf):
            self._run_face_tracking_bounded()
        self.assertEqual(
            bridge.capture_requests, 0,
            "KINECT_AS_CAMERA hijacked a NAMED camera entry - the mocked webcam "
            "was bypassed for a Kinect capture (this is the exact defect that "
            "made these tests open the real sensor and allocate 130+ GB)")
        cv2.VideoCapture.assert_called_once_with(0, 700)

    def test_unnamed_camera_entry_opts_into_kinect(self):
        # The mechanism, pinned. With KINECT_AS_CAMERA on, an entry with NO
        # "name" IS handed a KinectCapture - that is deliberate (an unnamed slot
        # opts the sensor in), and it is precisely why every CAMERAS dict in
        # this class must carry a name. Uses the STUB bridge, so this documents
        # the hijack without ever touching the sensor.
        bridge = _StubKinectBridge()
        cv2 = mock.Mock()
        bad_cap = mock.Mock()
        bad_cap.isOpened.return_value = False
        cv2.VideoCapture.return_value = bad_cap
        cv2.CAP_DSHOW = 700
        unnamed = {"index": 0, "label": "X", "primary": True,
                   "look_x": 0.5, "look_y": 0.5}
        self.bc._face_track_stop.set()
        buf = io.StringIO()
        with mock.patch.object(self.bc, "KINECT_AS_CAMERA", True), \
                mock.patch.object(self.bc, "_kinect_bridge", bridge), \
                mock.patch.object(self.bc, "cv2", cv2), \
                mock.patch.object(self.bc, "CAMERAS", [unnamed]), \
                contextlib.redirect_stdout(buf):
            self._run_face_tracking_bounded()
        self.assertEqual(bridge.capture_requests, 1)
        self.assertIn("Kinect requested but unavailable", buf.getvalue())

    def test_watchdog_fails_loudly_on_unbounded_loop(self):
        # REGRESSION GUARD for the guard: a capture that keeps opening and never
        # honours the stop event must make _run_face_tracking_bounded FAIL, not
        # hang. Everything here is a plain object (no Mock retains frames) and
        # the window is one second, so this cannot itself balloon.
        class _NeverEndingCap:
            def isOpened(self):     # noqa: N802 - mirrors cv2.VideoCapture
                return True

            def read(self):
                time.sleep(0.01)
                return False, None      # never a frame, never a stop

            def get(self, _prop):
                return 0

            def set(self, *_a, **_k):
                return True

            def release(self):
                pass

        class _Cv2Stub:
            CAP_DSHOW = 700
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            CAP_PROP_BUFFERSIZE = 38
            error = self.bc.cv2.error

            def VideoCapture(self, *_a, **_k):   # noqa: N802 - cv2 API name
                return _NeverEndingCap()

        buf = io.StringIO()
        logging.disable(logging.CRITICAL)
        try:
            with mock.patch.object(self.bc, "cv2", _Cv2Stub()), \
                    mock.patch.object(self.bc, "CAMERAS", self._cams()), \
                    mock.patch.object(self.bc, "_note_camera_read_attempt"), \
                    mock.patch.object(self.bc, "find_camera_locking_processes",
                                      return_value=[]), \
                    mock.patch.object(self.bc, "_hud_camera_preview_enabled",
                                      return_value=False), \
                    mock.patch.object(self.bc, "send"), \
                    contextlib.redirect_stdout(buf):
                with self.assertRaises(self.failureException) as caught:
                    self._run_face_tracking_bounded(timeout=1.0)
        finally:
            logging.disable(logging.NOTSET)
        self.assertIn("UNBOUNDED", str(caught.exception))
        # The watchdog must also have actually STOPPED the thread it caught.
        self.assertTrue(self.bc._face_track_stop.is_set())

    def test_one_good_frame_iteration_then_stop(self):
        # Drive exactly one healthy loop iteration: the camera opens, yields a
        # good frame, a face is detected on the primary cam -> the eye-control
        # math + send() path runs, then _face_track_stop ends the loop. Covers
        # the frame-cache / detection / tracking-math body (not just the empty
        # early-return). No real device - the thread is watchdog-bounded.
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 1280

        stop = self.bc._face_track_stop

        # STOP FROM THE LOOP, NOT FROM read(). This used to arm the stop event
        # inside cap.read(), on the assumption that the FIRST read the code
        # makes is the loop's. That stopped being true on 2026-09-05:
        # _camera_open() proves the device delivers a frame before handing the
        # handle back (under Media Foundation a camera another process holds
        # reports isOpened() True and then produces nothing, so "opened" is no
        # longer evidence), which consumes one read per camera BEFORE the loop
        # starts — so the old fake stopped the loop before it ran at all.
        # _note_camera_read_attempt is called only by the loop, once per read
        # result, so arming from there restores "exactly one iteration"
        # whatever the open costs.
        cap.read.return_value = (True, frame)
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700

        sends = []
        with mock.patch.object(self.bc, "cv2", cv2), \
                mock.patch.object(self.bc, "CAMERAS", self._cams()), \
                mock.patch.object(self.bc, "_detect_face",
                                  return_value=(0.5, 0.5)), \
                mock.patch.object(self.bc, "_note_camera_read_attempt",
                                  side_effect=lambda *a, **k: stop.set()), \
                mock.patch.object(self.bc, "send",
                                  side_effect=lambda **k: sends.append(k)), \
                mock.patch.object(self.bc.time, "sleep"):
            self._run_face_tracking_bounded()
        # The good frame was cached for see_user.
        with self.bc._camera_state_lock:
            self.assertIn(0, self.bc._camera_latest_frame)

    # -- camera STAYS ON during listening/speaking (fix/camera-stays-on) ------
    def _drive_one_preview_iteration(self, *, paused, camera_off, standby):
        """Run exactly one face-track loop iteration with the HUD-preview
        writer/remover mocked, returning (write_mock, remove_mock, detect_mock)
        so callers can assert whether the preview was written/removed and
        whether the HEAVY detection ran. One good primary frame, then stop."""
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 1280

        stop = self.bc._face_track_stop
        pause = self.bc._face_track_pause
        cam_off = self.bc._face_track_camera_off
        if paused:
            pause.set()
        if camera_off:
            cam_off.set()

        # Armed from _note_camera_read_attempt below, not from read() — see
        # test_one_good_frame_iteration_then_stop for why read() is no longer a
        # reliable "the loop is running" signal.
        cap.read.return_value = (True, frame)

        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700

        write_mock  = mock.Mock(return_value=True)
        remove_mock = mock.Mock()
        detect_mock = mock.Mock(return_value=(0.5, 0.5))
        # _standby_mode is a list-of-one cell; set [0] for the standby case.
        saved_standby = self.bc._standby_mode[0]
        self.bc._standby_mode[0] = bool(standby)
        try:
            with mock.patch.object(self.bc, "cv2", cv2), \
                    mock.patch.object(self.bc, "CAMERAS", self._cams()), \
                    mock.patch.object(self.bc, "_detect_face", detect_mock), \
                    mock.patch.object(self.bc, "_hud_camera_preview_enabled",
                                      return_value=True), \
                    mock.patch.object(self.bc, "_hud_camera_preview_write",
                                      write_mock), \
                    mock.patch.object(self.bc, "_hud_camera_preview_remove",
                                      remove_mock), \
                    mock.patch.object(self.bc, "_note_camera_read_attempt",
                                      side_effect=lambda *a, **k: stop.set()), \
                    mock.patch.object(self.bc, "send"), \
                    mock.patch.object(self.bc.time, "sleep"):
                self._run_face_tracking_bounded()
        finally:
            self.bc._standby_mode[0] = saved_standby
        return write_mock, remove_mock, detect_mock

    def test_paused_for_voice_keeps_preview_live(self):
        # The core fix: while paused for voice (listening/thinking/speaking) the
        # camera must STAY visibly ON - the primary frame is still mirrored to
        # the HUD preview - while the HEAVY cv2 face detection is still skipped
        # (the recognition cost the pause exists to save).
        write_mock, remove_mock, detect_mock = self._drive_one_preview_iteration(
            paused=True, camera_off=False, standby=False)
        # Preview kept live -> HUD never shows "CAMERA OFF" just for listening.
        # (The thread's on-shutdown cleanup removes the file once after the loop
        # ends - the single test iteration triggers that teardown - so the live
        # signal is "write happened" + "no top-of-loop blank", i.e. remove was
        # NOT called BEFORE the write. That ordering is asserted below.)
        write_mock.assert_called_once()
        # Mock records calls in order across distinct mocks via a shared parent
        # only if attached; here we assert the in-loop blank never fired by
        # checking remove was called at most once (the post-loop teardown) - a
        # top-of-loop blank would make it 2 (blank + teardown), as the
        # camera_off / standby tests confirm.
        self.assertLessEqual(remove_mock.call_count, 1)
        # Frame still cached so the air-mouse / gaze keep getting frames.
        with self.bc._camera_state_lock:
            self.assertIn(0, self.bc._camera_latest_frame)
        # ...but the expensive recognition pass was NOT run while paused.
        detect_mock.assert_not_called()

    def test_low_memory_camera_off_blanks_preview(self):
        # The KINECT_REVIEW P0 low-memory guard sets _face_track_camera_off ->
        # the preview must blank (file removed, never written) so the HUD shows
        # "CAMERA OFF" and we stop mirroring HD frames under memory pressure,
        # even though the capture loop keeps running. Removed twice here: the
        # top-of-loop blank AND the post-loop teardown.
        write_mock, remove_mock, _ = self._drive_one_preview_iteration(
            paused=False, camera_off=True, standby=False)
        write_mock.assert_not_called()
        self.assertGreaterEqual(remove_mock.call_count, 2)

    def test_standby_blanks_preview(self):
        # Standby / empty room (_standby_mode) must STILL blank the preview to
        # the "CAMERA OFF" placeholder - only the listening/speaking case was
        # changed to keep the camera on. Removed twice (top-of-loop + teardown).
        write_mock, remove_mock, _ = self._drive_one_preview_iteration(
            paused=False, camera_off=False, standby=True)
        write_mock.assert_not_called()
        self.assertGreaterEqual(remove_mock.call_count, 2)

    def test_detect_face_cv2_error_degrades_gracefully(self):
        # A cv2.error out of _detect_face (e.g. an OpenCL/GPU hiccup) must NOT
        # unwind the tracking thread. The loop should swallow it, cache the
        # frame for see_user, back off with time.sleep(0.5), and continue -
        # then exit cleanly when _face_track_stop fires.
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 1280

        stop = self.bc._face_track_stop

        # Armed from _note_camera_read_attempt below, not from read() — see
        # test_one_good_frame_iteration_then_stop for why read() is no longer a
        # reliable "the loop is running" signal.
        cap.read.return_value = (True, frame)

        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        # The source does `except cv2.error:` against the module-level cv2, so
        # the mock's .error MUST be the real OpenCV exception class for the
        # handler to resolve and catch.
        cv2.error = self.bc.cv2.error

        def _boom(_frame):
            raise self.bc.cv2.error("CL_OUT_OF_RESOURCES")

        sleeps = []
        with mock.patch.object(self.bc, "cv2", cv2), \
                mock.patch.object(self.bc, "CAMERAS", self._cams()), \
                mock.patch.object(self.bc, "_detect_face",
                                  side_effect=_boom), \
                mock.patch.object(self.bc, "_note_camera_read_attempt",
                                  side_effect=lambda *a, **k: stop.set()), \
                mock.patch.object(self.bc, "send"), \
                mock.patch.object(self.bc.time, "sleep",
                                  side_effect=lambda s: sleeps.append(s)):
            # Must return normally - the cv2.error is swallowed, not raised.
            # Silence the expected logging.exception traceback so the test
            # log stays clean (the handler under test logs the swallow).
            logging.disable(logging.CRITICAL)
            try:
                self._run_face_tracking_bounded()
            finally:
                logging.disable(logging.NOTSET)
        # Frame was still cached despite the detection failure.
        with self.bc._camera_state_lock:
            self.assertIn(0, self.bc._camera_latest_frame)
        # And we backed off by 0.5s after the failed detect.
        self.assertIn(0.5, sleeps)


# ───────────────────────────────────────────────────────────────────────────
#  should_be_proactive / generate_proactive_comment
# ───────────────────────────────────────────────────────────────────────────
class ProactiveDecisionTests(_MonolithSec2Base):
    def setUp(self):
        self._saved_last_speech = self.bc.last_speech_time
        self._saved_last_face = self.bc.last_face_seen

    def tearDown(self):
        self.bc.last_speech_time = self._saved_last_speech
        self.bc.last_face_seen = self._saved_last_face

    def test_disabled_returns_false(self):
        with mock.patch.object(self.bc, "PROACTIVE_ENABLED", False):
            self.assertFalse(self.bc.should_be_proactive())

    def test_insufficient_silence_returns_false(self):
        with mock.patch.object(self.bc, "PROACTIVE_ENABLED", True), \
                mock.patch.object(self.bc, "PROACTIVE_MIN_SILENCE", 180):
            self.bc.last_speech_time = time.time()  # zero silence
            self.assertFalse(self.bc.should_be_proactive())

    def test_no_recent_face_returns_false_when_required(self):
        with mock.patch.object(self.bc, "PROACTIVE_ENABLED", True), \
                mock.patch.object(self.bc, "PROACTIVE_MIN_SILENCE", 1), \
                mock.patch.object(self.bc, "PROACTIVE_REQUIRE_FACE", True), \
                mock.patch.object(self.bc, "_voice_mood_response", None):
            self.bc.last_speech_time = time.time() - 1000
            self.bc.last_face_seen = 0.0  # never seen
            self.assertFalse(self.bc.should_be_proactive())

    def test_high_silence_with_face_fires_when_rng_low(self):
        with mock.patch.object(self.bc, "PROACTIVE_ENABLED", True), \
                mock.patch.object(self.bc, "PROACTIVE_MIN_SILENCE", 1), \
                mock.patch.object(self.bc, "PROACTIVE_MAX_SILENCE", 10), \
                mock.patch.object(self.bc, "PROACTIVE_REQUIRE_FACE", True), \
                mock.patch.object(self.bc, "_voice_mood_response", None), \
                mock.patch.object(self.bc.random, "random", return_value=0.0):
            self.bc.last_speech_time = time.time() - 1000
            self.bc.last_face_seen = time.time()
            self.assertTrue(self.bc.should_be_proactive())

    def test_high_silence_with_face_skips_when_rng_high(self):
        with mock.patch.object(self.bc, "PROACTIVE_ENABLED", True), \
                mock.patch.object(self.bc, "PROACTIVE_MIN_SILENCE", 1), \
                mock.patch.object(self.bc, "PROACTIVE_MAX_SILENCE", 10), \
                mock.patch.object(self.bc, "PROACTIVE_REQUIRE_FACE", True), \
                mock.patch.object(self.bc, "_voice_mood_response", None), \
                mock.patch.object(self.bc.random, "random", return_value=0.99):
            self.bc.last_speech_time = time.time() - 1000
            self.bc.last_face_seen = time.time()
            self.assertFalse(self.bc.should_be_proactive())

    def test_generate_proactive_comment_uses_llm_first_line(self):
        with mock.patch.object(self.bc, "_llm_quick",
                               return_value="  Nice work on the build.\nextra"):
            out = self.bc.generate_proactive_comment()
        self.assertEqual(out, "Nice work on the build.")

    def test_generate_proactive_comment_empty_on_llm_error(self):
        with mock.patch.object(self.bc, "_llm_quick",
                               side_effect=Exception("cap hit")):
            self.assertEqual(self.bc.generate_proactive_comment(), "")


# ───────────────────────────────────────────────────────────────────────────
#  Late-night remark state machine
# ───────────────────────────────────────────────────────────────────────────
class LateNightTests(_MonolithSec2Base):
    # A fixed epoch known to fall at 03:00 local time is awkward across TZs,
    # so we instead pass explicit `now=` where the API allows, and patch
    # _in_late_night_window for the orchestrator (maybe_late_night_remark).

    def _at_hour(self, hour):
        """Return an epoch whose LOCAL hour == `hour` today."""
        lt = list(time.localtime())
        lt[3] = hour
        lt[4] = 0
        lt[5] = 0
        return time.mktime(time.struct_time(tuple(lt)))

    def test_in_window_true_at_3am(self):
        self.assertTrue(self.bc._in_late_night_window(self._at_hour(3)))

    def test_in_window_false_at_noon(self):
        self.assertFalse(self.bc._in_late_night_window(self._at_hour(12)))

    def test_in_window_boundary_5am_exclusive(self):
        self.assertFalse(self.bc._in_late_night_window(self._at_hour(5)))

    def test_in_window_boundary_1am_inclusive(self):
        self.assertTrue(self.bc._in_late_night_window(self._at_hour(1)))

    def test_hour_word_is_str_digit(self):
        self.assertEqual(self.bc._late_night_hour_word(self._at_hour(3)), "3")

    def test_session_key_is_date(self):
        key = self.bc._late_night_session_key(self._at_hour(3))
        self.assertRegex(key, r"^\d{4}-\d{2}-\d{2}$")

    def test_suppression_roundtrip(self):
        now = self._at_hour(2)
        key = self.bc._late_night_session_key(now)
        mem = {"late_night_no_comments_until": key}
        self.assertTrue(self.bc._is_late_night_suppressed(mem, now))

    def test_not_suppressed_when_absent(self):
        self.assertFalse(self.bc._is_late_night_suppressed({}, self._at_hour(2)))

    def test_not_suppressed_when_stale_key(self):
        mem = {"late_night_no_comments_until": "1999-01-01"}
        self.assertFalse(
            self.bc._is_late_night_suppressed(mem, self._at_hour(2)))

    def test_set_suppression_persists(self):
        # _set_late_night_suppression persists against a FRESH locked
        # load_memory() (the shutdown-path clobber fix), NOT the passed-in
        # snapshot — so mock the load too, or the assertion depends on this
        # box's real bobert_memory.json. (Stale pin of the pre-fix behaviour
        # repaired 2026-07-21.) The passed snapshot must still be updated
        # in-session.
        mem = {}
        with mock.patch.object(self.bc, "save_memory") as save, \
                mock.patch.object(self.bc, "load_memory", return_value={}):
            self.bc._set_late_night_suppression(mem)
        self.assertIn("late_night_no_comments_until", mem)
        key = mem["late_night_no_comments_until"]
        save.assert_called_once_with({"late_night_no_comments_until": key})

    def test_matches_suppress_phrase_true(self):
        self.assertTrue(self.bc._matches_suppress_phrase("no comments tonight"))
        self.assertTrue(self.bc._matches_suppress_phrase("Please skip the remarks"))

    def test_matches_suppress_phrase_false_when_long(self):
        long_text = ("kindly refrain from any commentary or remarks for the "
                     "duration of tonight please thanks")
        self.assertFalse(self.bc._matches_suppress_phrase(long_text))

    def test_matches_suppress_phrase_false_when_absent(self):
        self.assertFalse(self.bc._matches_suppress_phrase("turn on the lights"))

    def test_maybe_remark_outside_window_empty(self):
        with mock.patch.object(self.bc, "_in_late_night_window",
                               return_value=False):
            self.assertEqual(self.bc.maybe_late_night_remark("hi", {}), "")

    def test_maybe_remark_suppress_phrase_acknowledges(self):
        mem = {}
        with mock.patch.object(self.bc, "_in_late_night_window",
                               return_value=True), \
                mock.patch.object(self.bc, "save_memory"):
            out = self.bc.maybe_late_night_remark("no comments tonight", mem)
        self.assertEqual(out, "As you wish, sir. Silent until morning.")
        self.assertIn("late_night_no_comments_until", mem)

    def test_maybe_remark_returns_empty_when_suppressed(self):
        with mock.patch.object(self.bc, "_in_late_night_window",
                               return_value=True), \
                mock.patch.object(self.bc, "_matches_suppress_phrase",
                                  return_value=False), \
                mock.patch.object(self.bc, "_is_late_night_suppressed",
                                  return_value=True):
            self.assertEqual(self.bc.maybe_late_night_remark("do x", {}), "")

    def test_maybe_remark_cooldown_blocks_repeat(self):
        with mock.patch.object(self.bc, "_in_late_night_window",
                               return_value=True), \
                mock.patch.object(self.bc, "_matches_suppress_phrase",
                                  return_value=False), \
                mock.patch.object(self.bc, "_is_late_night_suppressed",
                                  return_value=False), \
                mock.patch.object(self.bc, "_late_night_last_remark",
                                  [time.time()]), \
                mock.patch.object(self.bc, "LATE_NIGHT_COOLDOWN", 600):
            self.assertEqual(self.bc.maybe_late_night_remark("do x", {}), "")

    def test_maybe_remark_emits_and_advances_cursor(self):
        idx_cell = [0]
        last_cell = [0.0]
        with mock.patch.object(self.bc, "_in_late_night_window",
                               return_value=True), \
                mock.patch.object(self.bc, "_matches_suppress_phrase",
                                  return_value=False), \
                mock.patch.object(self.bc, "_is_late_night_suppressed",
                                  return_value=False), \
                mock.patch.object(self.bc, "_late_night_phrase_idx", idx_cell), \
                mock.patch.object(self.bc, "_late_night_last_remark",
                                  last_cell), \
                mock.patch.object(self.bc, "_late_night_hour_word",
                                  return_value="3"):
            out = self.bc.maybe_late_night_remark("do x", {})
        self.assertTrue(out)               # non-empty remark
        self.assertEqual(idx_cell[0], 1)   # cursor advanced
        self.assertGreater(last_cell[0], 0.0)  # cooldown stamp set


# ───────────────────────────────────────────────────────────────────────────
#  _thinking_loop / get_response_with_animation  (send + LLM mocked)
# ───────────────────────────────────────────────────────────────────────────
class ThinkingAnimationTests(_MonolithSec2Base):
    def test_thinking_loop_exits_on_event_and_ticks(self):
        stop = threading.Event()
        sends = []
        beats = []
        with mock.patch.object(self.bc, "send",
                               side_effect=lambda **k: sends.append(k)), \
                mock.patch.object(self.bc, "_heartbeat",
                                  side_effect=lambda: beats.append(1)), \
                mock.patch.object(self.bc.time, "sleep") as slp:
            # Stop the loop on the 3rd sleep so it runs a few iterations.
            calls = {"n": 0}

            def _sleep(_):
                calls["n"] += 1
                if calls["n"] >= 3:
                    stop.set()

            slp.side_effect = _sleep
            self.bc._thinking_loop(stop)
        # Each iteration sends eye coordinates.
        self.assertTrue(sends)
        self.assertIn("eyes_x", sends[0])

    def test_thinking_loop_survives_send_exception(self):
        stop = threading.Event()
        with mock.patch.object(self.bc, "send",
                               side_effect=RuntimeError("robot down")), \
                mock.patch.object(self.bc, "_heartbeat"), \
                mock.patch.object(self.bc.time, "sleep",
                                  side_effect=lambda _: stop.set()):
            # The except-branch logs and sleeps; loop must not raise.
            self.bc._thinking_loop(stop)

    def test_get_response_with_animation_returns_reply(self):
        with mock.patch.object(self.bc, "pause_face_tracking"), \
                mock.patch.object(self.bc, "set_state"), \
                mock.patch.object(self.bc, "_call_llm",
                                  return_value="Right away, sir."), \
                mock.patch.object(self.bc, "_thinking_loop"):
            # _thinking_loop is stubbed so the spawned daemon thread is a no-op
            # that returns immediately; anim.join() then returns at once.
            out = self.bc.get_response_with_animation("status?")
        self.assertEqual(out, "Right away, sir.")


# ───────────────────────────────────────────────────────────────────────────
#  Main-loop watchdog  (_heartbeat / _main_loop_watchdog_check / thread)
# ───────────────────────────────────────────────────────────────────────────
class WatchdogTests(_MonolithSec2Base):
    def setUp(self):
        self._saved_hb = self.bc._main_loop_heartbeat[0]
        self.bc._watchdog_reset_signal.clear()

    def tearDown(self):
        self.bc._main_loop_heartbeat[0] = self._saved_hb
        self.bc._watchdog_reset_signal.clear()
        self.bc._watchdog_stop_event.clear()

    def test_heartbeat_updates_and_clears_signal(self):
        self.bc._watchdog_reset_signal.set()
        self.bc._main_loop_heartbeat[0] = 0.0
        self.bc._heartbeat()
        self.assertGreater(self.bc._main_loop_heartbeat[0], 0.0)
        self.assertFalse(self.bc._watchdog_reset_signal.is_set())

    # ── the ON-DISK half of the heartbeat (2026-08-20) ──────────────────
    # core/diagnostic_daemons._check_stuck_loop used hud_state.json's mtime as
    # its main-loop liveness signal. _write_hud_state returns immediately when
    # HUD_ENABLED is False, and nothing ever deletes hud_state.json, so
    # unticking "On-screen HUD" in Settings froze the mtime and the anomaly
    # watcher queued "[anomaly] main loop appears stuck" every 30 minutes for
    # the life of the session — a false alarm that never stops, into a todo
    # file the overnight upgrade pipeline consumes.

    def _hb_sandbox(self):
        """(tmpdir, path) with the heartbeat file redirected into it."""
        import tempfile
        tmp = tempfile.mkdtemp(prefix="jv_hb_")
        self.addCleanup(__import__("shutil").rmtree, tmp, True)
        return tmp, os.path.join(tmp, "main_loop_heartbeat")

    def _patch_hb_path(self, path):
        return mock.patch.object(self.bc, "_main_loop_heartbeat_path_cache",
                                 [path])

    def test_publish_writes_the_file(self):
        _tmp, path = self._hb_sandbox()
        with self._patch_hb_path(path), \
             mock.patch.object(self.bc, "_last_heartbeat_publish", [0.0]):
            self.bc._publish_main_loop_heartbeat(force=True, now=1234.5)
        self.assertTrue(os.path.exists(path))

    def test_publish_is_not_gated_by_hud_enabled(self):
        """THE regression. The whole defect was a liveness signal sitting
        behind a feature toggle."""
        _tmp, path = self._hb_sandbox()
        with self._patch_hb_path(path), \
             mock.patch.object(self.bc, "HUD_ENABLED", False), \
             mock.patch.object(self.bc, "_last_heartbeat_publish", [0.0]):
            self.bc._heartbeat()
        self.assertTrue(
            os.path.exists(path),
            "the main-loop heartbeat must be published with the HUD OFF — "
            "gating it is exactly the defect this file replaced")

    def test_publish_is_throttled_but_force_bypasses(self):
        _tmp, path = self._hb_sandbox()
        with self._patch_hb_path(path), \
             mock.patch.object(self.bc, "_last_heartbeat_publish", [0.0]):
            self.bc._publish_main_loop_heartbeat(now=1000.0)   # writes
            first = os.path.getmtime(path)
            os.utime(path, (first - 500, first - 500))
            # 0.1 s later: inside the throttle window, must NOT rewrite.
            self.bc._publish_main_loop_heartbeat(now=1000.1)
            self.assertEqual(os.path.getmtime(path), first - 500)
            # force ignores the throttle (used once at loop entry so the
            # daemon never judges this run against a previous run's mtime).
            self.bc._publish_main_loop_heartbeat(force=True, now=1000.2)
            self.assertNotEqual(os.path.getmtime(path), first - 500)

    def test_publish_never_raises_into_the_main_loop(self):
        # An unwritable path must not be able to kill the loop it measures.
        with self._patch_hb_path(os.path.join(os.sep, "no", "such", "dir",
                                              "main_loop_heartbeat")), \
             mock.patch.object(self.bc, "_last_heartbeat_publish", [0.0]):
            self.bc._publish_main_loop_heartbeat(force=True, now=1.0)
            self.bc._heartbeat()          # must not raise

    def test_writer_and_watcher_agree_on_one_path(self):
        """No second private join. Writer/watcher disagreeing about the path
        would reproduce the permanently-firing false alarm from the other
        side, and this repo's #1 bug class is one rule in two copies."""
        from core import diagnostic_daemons as _dd
        with mock.patch.object(self.bc, "_main_loop_heartbeat_path_cache", [""]):
            self.assertEqual(self.bc._main_loop_heartbeat_path(),
                             _dd.MAIN_LOOP_HEARTBEAT_FILE)

    def test_watchdog_check_detects_stall(self):
        self.bc._main_loop_heartbeat[0] = 100.0
        # now far ahead of heartbeat, threshold small → stall
        fired = self.bc._main_loop_watchdog_check(now=1000.0, threshold=10.0)
        self.assertTrue(fired)
        self.assertTrue(self.bc._watchdog_reset_signal.is_set())

    def test_watchdog_check_no_stall_when_fresh(self):
        self.bc._main_loop_heartbeat[0] = 995.0
        fired = self.bc._main_loop_watchdog_check(now=1000.0, threshold=60.0)
        self.assertFalse(fired)
        self.assertFalse(self.bc._watchdog_reset_signal.is_set())

    def test_watchdog_check_no_double_fire(self):
        self.bc._main_loop_heartbeat[0] = 0.0
        self.bc._watchdog_reset_signal.set()  # already raised
        fired = self.bc._main_loop_watchdog_check(now=1000.0, threshold=10.0)
        self.assertFalse(fired)  # signal already set → returns False

    def test_watchdog_thread_exits_on_stop_event(self):
        # The thread loops on _watchdog_stop_event.wait(interval); set it so
        # the first wait returns True and the thread exits immediately.
        self.bc._watchdog_stop_event.set()
        with mock.patch.object(self.bc, "_main_loop_watchdog_check",
                               return_value=False):
            t = threading.Thread(target=self.bc._main_loop_watchdog_thread)
            t.start()
            t.join(timeout=5.0)
            self.assertFalse(t.is_alive())


# ───────────────────────────────────────────────────────────────────────────
#  Main-loop per-turn exception net  (_recover_from_main_loop_error + the
#  `except Exception` wrap around the `while True:` body in main())
#
#  Regression guard for the P1 where the whole turn loop ran in one try whose
#  ONLY handler was `except KeyboardInterrupt`: any exception escaping a callee
#  (LLM dispatch, learn_from_turn, the prompt-rebuild Timer, …) propagated out
#  of main() and killed the detached pythonw process with no auto-restart — one
#  malformed turn took the assistant permanently offline. The watchdog only
#  recovers a STALLED loop; it cannot relaunch a loop that threw and exited.
# ───────────────────────────────────────────────────────────────────────────
class MainLoopRecoveryTests(_MonolithSec2Base):
    def test_recover_returns_to_idle(self):
        # The recovery path must drop the HUD/avatar back to idle so a failed
        # turn doesn't strand it in "thinking"/"listening".
        with mock.patch.object(self.bc, "set_state") as set_state, \
                mock.patch.object(self.bc, "logging") as log:
            self.bc._recover_from_main_loop_error(RuntimeError("boom"))
        set_state.assert_called_once_with("idle")
        # And it logs the full traceback (logging.exception) for diagnosis.
        self.assertTrue(log.exception.called)

    def test_recover_never_raises_even_if_set_state_fails(self):
        # The net must be bullet-proof: a failure inside recovery must NOT
        # re-break the loop it is protecting (it runs in that loop's except arm).
        with mock.patch.object(self.bc, "set_state",
                               side_effect=RuntimeError("hud down")), \
                mock.patch.object(self.bc, "logging") as log:
            # No exception should escape.
            self.bc._recover_from_main_loop_error(ValueError("turn blew up"))
        # The turn failure is still logged even though set_state then failed.
        self.assertTrue(log.exception.called)

    def test_recover_does_not_swallow_keyboardinterrupt_semantics(self):
        # The helper takes BaseException so it can log a KeyboardInterrupt if
        # ever handed one, but it must return cleanly (it never re-raises).
        with mock.patch.object(self.bc, "set_state"), \
                mock.patch.object(self.bc, "logging"):
            self.assertIsNone(
                self.bc._recover_from_main_loop_error(KeyboardInterrupt()))

    def test_main_loop_body_wrapped_in_exception_net(self):
        # Wiring guard: the `while True:` turn body must sit inside an
        # `except Exception` arm that recovers and continues — otherwise the
        # per-turn net is gone and one bad turn kills JARVIS again. Asserting on
        # main()'s source lets us guard this without running the boot sequence.
        import inspect
        src = inspect.getsource(self.bc.main)
        # The loop body opens a try right under `while True:`.
        self.assertRegex(src, r"while True:\s*\n\s*try:")
        # …closed by an `except Exception` that dispatches to the recovery
        # helper and then `continue`s back into the loop.
        self.assertIn("except Exception as _loop_exc:", src)
        after = src[src.index("except Exception as _loop_exc:"):]
        recover_at = after.find("_recover_from_main_loop_error(_loop_exc)")
        self.assertNotEqual(recover_at, -1,
                            "per-turn except arm never calls the recovery helper")
        # The actual `continue` statement (not the word in a comment) must come
        # AFTER the recovery call so the loop resumes only once recovered.
        cont_at = after.find("\n                continue", recover_at)
        self.assertNotEqual(cont_at, -1,
                            "per-turn except arm never continues the loop after "
                            "recovery")
        # The clean-shutdown handler for Ctrl-C must still be present and OUTSIDE
        # the per-turn net (KeyboardInterrupt is not an Exception), so the inner
        # net can never swallow a deliberate shutdown.
        self.assertIn("except KeyboardInterrupt:", src)


# ───────────────────────────────────────────────────────────────────────────
#  HUD tray-state mirror + restore  (_publish_audio_state /
#  _restore_tray_toggle_state) — real temp HUD state file
# ───────────────────────────────────────────────────────────────────────────
class HudTrayStateTests(_MonolithSec2Base):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jarvis_hud_")
        self.hud_file = os.path.join(self.tmp, "hud_state.json")
        # Patch the file path + cache so writes go to our temp file.
        self._patches = [
            mock.patch.object(self.bc, "HUD_STATE_FILE", self.hud_file),
            mock.patch.object(self.bc, "HUD_ENABLED", True),
        ]
        for p in self._patches:
            p.start()
        with self.bc._hud_state_lock:
            self._saved_cache = dict(self.bc._hud_state_cache)
            self.bc._hud_state_cache.clear()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        with self.bc._hud_state_lock:
            self.bc._hud_state_cache.clear()
            self.bc._hud_state_cache.update(self._saved_cache)
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_publish_audio_state_writes_flags(self):
        with mock.patch.object(self.bc, "_audio_master_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_aec_enabled", [False]), \
                mock.patch.object(self.bc, "_audio_ns_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_agc_enabled", [False]):
            self.bc._publish_audio_state()
        with open(self.hud_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(data["audio_processing_enabled"])
        self.assertFalse(data["echo_cancel_enabled"])
        self.assertTrue(data["noise_suppress_enabled"])
        self.assertFalse(data["agc_enabled"])

    def test_restore_missing_file_is_noop(self):
        # No file present → returns without raising and without mutating flags.
        self.assertFalse(os.path.exists(self.hud_file))
        with mock.patch.object(self.bc, "ACTIONS", {}):
            self.bc._restore_tray_toggle_state()

    def test_restore_reads_persisted_flags(self):
        persisted = {
            "tts_muted": True,
            "ambient_mode_active": False,
            "daemons_paused": False,
            "debug_mode": True,
            "sleep_mode": True,
            "standby_mode": False,
            "audio_processing_enabled": False,
            "echo_cancel_enabled": True,
            "noise_suppress_enabled": False,
            "agc_enabled": True,
        }
        with open(self.hud_file, "w", encoding="utf-8") as f:
            json.dump(persisted, f)
        tts = [False]
        amb = [False]
        paused = [False]
        dbg = [False]
        slp = [False]
        stby = [False]
        am = [True]
        aec = [False]
        ns = [True]
        agc = [False]
        with mock.patch.object(self.bc, "_tts_muted", tts), \
                mock.patch.object(self.bc, "_ambient_mode_active", amb), \
                mock.patch.object(self.bc, "_daemons_paused", paused), \
                mock.patch.object(self.bc, "_debug_mode", dbg), \
                mock.patch.object(self.bc, "_sleep_mode", slp), \
                mock.patch.object(self.bc, "_standby_mode", stby), \
                mock.patch.object(self.bc, "_audio_master_enabled", am), \
                mock.patch.object(self.bc, "_audio_aec_enabled", aec), \
                mock.patch.object(self.bc, "_audio_ns_enabled", ns), \
                mock.patch.object(self.bc, "_audio_agc_enabled", agc), \
                mock.patch.object(self.bc, "ACTIONS", {}):
            self.bc._restore_tray_toggle_state()
        self.assertTrue(tts[0])
        self.assertTrue(dbg[0])
        self.assertTrue(slp[0])
        self.assertFalse(am[0])
        self.assertTrue(aec[0])

    def test_restore_corrupt_file_is_noop(self):
        with open(self.hud_file, "w", encoding="utf-8") as f:
            f.write("{ this is : not json")
        with mock.patch.object(self.bc, "ACTIONS", {}):
            # JSON decode error is caught → returns cleanly.
            self.bc._restore_tray_toggle_state()

    def test_restore_resumes_ambient_when_active(self):
        persisted = {"ambient_mode_active": True}
        with open(self.hud_file, "w", encoding="utf-8") as f:
            json.dump(persisted, f)
        called = []
        amb = [False]
        with mock.patch.object(self.bc, "_ambient_mode_active", amb), \
                mock.patch.object(self.bc, "ACTIONS",
                                  {"ambient_listen_start":
                                   lambda _: called.append(1)}):
            self.bc._restore_tray_toggle_state()
        self.assertTrue(amb[0])
        self.assertEqual(called, [1])  # resume hook fired


# ───────────────────────────────────────────────────────────────────────────
#  Tray drainer + state publisher (single-iteration; loops broken via stop
#  events / patched poll so no long-lived thread runs)
# ───────────────────────────────────────────────────────────────────────────
class TrayDrainerTests(_MonolithSec2Base):
    def test_drainer_runs_one_iteration_then_stops(self):
        # Patch the per-iteration drain to flip the stop event so the while
        # loop body executes exactly once and exits.
        stop = self.bc._tray_drain_stop
        stop.clear()
        calls = []

        def _drain_once():
            calls.append(1)
            stop.set()
            return 0

        with mock.patch.object(self.bc, "_drain_tray_commands_once",
                               side_effect=_drain_once):
            try:
                self.bc._tray_command_drainer()
            finally:
                stop.clear()
        self.assertEqual(calls, [1])

    def test_drainer_survives_iteration_exception(self):
        stop = self.bc._tray_drain_stop
        stop.clear()
        state = {"n": 0}

        def _boom():
            state["n"] += 1
            stop.set()
            raise RuntimeError("inbox parse blew up")

        with mock.patch.object(self.bc, "_drain_tray_commands_once",
                               side_effect=_boom), \
                mock.patch.object(self.bc.logging, "exception"):
            try:
                self.bc._tray_command_drainer()
            finally:
                stop.clear()
        self.assertEqual(state["n"], 1)


class TrayStatePublisherTests(_MonolithSec2Base):
    def test_publisher_one_iteration_writes_when_changed(self):
        stop = self.bc._tray_publisher_stop
        stop.clear()

        # Break the loop after the first .wait() by setting stop in wait().
        def _wait(_):
            stop.set()
            return True

        writes = []
        with mock.patch.object(self.bc, "_write_hud_state",
                               side_effect=lambda **k: writes.append(k)), \
                mock.patch.dict(self.bc.sys.modules, {}, clear=False), \
                mock.patch.object(stop, "wait", side_effect=_wait), \
                mock.patch.object(self.bc, "_hud_cal_last", [time.time()]):
            # No system_monitor / bambu_monitor modules present → alert/bambu
            # both default False. Seed the cache so the change-detector fires.
            with self.bc._hud_state_lock:
                self.bc._hud_state_cache["alert_active"] = True
                self.bc._hud_state_cache["bambu_active"] = True
            try:
                self.bc._tray_state_publisher()
            finally:
                stop.clear()
        # alert/bambu computed False, cache had True → a write happened.
        self.assertTrue(
            any(w.get("alert_active") is False for w in writes))

    def test_publisher_detects_system_monitor_alert(self):
        stop = self.bc._tray_publisher_stop
        stop.clear()

        def _wait(_):
            stop.set()
            return True

        # Fake skill_system_monitor with a very recent CPU-alert timestamp →
        # alert should compute True.
        sm = mock.Mock()
        sm._last_cpu_alert_at = [time.time()]
        sm._last_ram_alert_at = [0]

        writes = []
        with mock.patch.object(self.bc, "_write_hud_state",
                               side_effect=lambda **k: writes.append(k)), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_system_monitor": sm}, clear=False), \
                mock.patch.object(stop, "wait", side_effect=_wait), \
                mock.patch.object(self.bc, "_hud_cal_last", [time.time()]):
            with self.bc._hud_state_lock:
                self.bc._hud_state_cache["alert_active"] = False
                self.bc._hud_state_cache["bambu_active"] = False
            try:
                self.bc._tray_state_publisher()
            finally:
                stop.clear()
        self.assertTrue(any(w.get("alert_active") is True for w in writes))

    def test_publisher_detects_bambu_running(self):
        stop = self.bc._tray_publisher_stop
        stop.clear()

        def _wait(_):
            stop.set()
            return True

        # Fake skill_bambu_monitor reporting a RUNNING print → bambu True.
        bm = mock.Mock()
        bm._state = {"gcode_state": "RUNNING"}
        bm._state_lock = threading.Lock()

        writes = []
        with mock.patch.object(self.bc, "_write_hud_state",
                               side_effect=lambda **k: writes.append(k)), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_bambu_monitor": bm}, clear=False), \
                mock.patch.object(stop, "wait", side_effect=_wait), \
                mock.patch.object(self.bc, "_hud_cal_last", [time.time()]):
            with self.bc._hud_state_lock:
                self.bc._hud_state_cache["alert_active"] = False
                self.bc._hud_state_cache["bambu_active"] = False
            try:
                self.bc._tray_state_publisher()
            finally:
                stop.clear()
        self.assertTrue(any(w.get("bambu_active") is True for w in writes))


# ═══════════════════════════════════════════════════════════════════════════
#  COVERAGE-EXTENSION TESTS (2026-06 push toward ~88%+ on lines 2705-4442)
#  Each class mirrors the harness rules above: inherits _MonolithSec2Base,
#  mocks all I/O, restores mutated globals, deterministic + offline.
# ═══════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────
#  _publish_audio_state — exception-swallow path (2716-2717)
# ───────────────────────────────────────────────────────────────────────────
class PublishAudioStateErrorTests(_MonolithSec2Base):
    def test_write_failure_is_swallowed(self):
        # _write_hud_state raising must be caught (the bare `except: pass`).
        with mock.patch.object(self.bc, "_write_hud_state",
                               side_effect=OSError("disk full")), \
                mock.patch.object(self.bc, "_audio_master_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_aec_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_ns_enabled", [True]), \
                mock.patch.object(self.bc, "_audio_agc_enabled", [True]):
            # Returns None, does not raise.
            self.assertIsNone(self.bc._publish_audio_state())


# ───────────────────────────────────────────────────────────────────────────
#  _restore_tray_toggle_state — read-failure + resume-hook failure branches
#  (2734, 2785-2786, 2790-2801)
# ───────────────────────────────────────────────────────────────────────────
class RestoreTrayToggleErrorTests(_MonolithSec2Base):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jarvis_hud_err_")
        self.hud_file = os.path.join(self.tmp, "hud_state.json")
        self._patches = [
            mock.patch.object(self.bc, "HUD_STATE_FILE", self.hud_file),
            mock.patch.object(self.bc, "HUD_ENABLED", True),
        ]
        for p in self._patches:
            p.start()
        with self.bc._hud_state_lock:
            self._saved_cache = dict(self.bc._hud_state_cache)
            self.bc._hud_state_cache.clear()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        with self.bc._hud_state_lock:
            self.bc._hud_state_cache.clear()
            self.bc._hud_state_cache.update(self._saved_cache)
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_open_raises_after_exists_prints_and_returns(self):
        # os.path.exists True but open() raises → the `except Exception as e`
        # read-failure branch (2734-2737) prints and returns.
        with open(self.hud_file, "w", encoding="utf-8") as f:
            f.write("{}")
        with mock.patch.object(self.bc.os.path, "exists", return_value=True), \
                mock.patch.object(self.bc, "open",
                                  side_effect=OSError("share gone"),
                                  create=True), \
                mock.patch.object(self.bc, "ACTIONS", {}):
            # Returns cleanly (None) — error is caught and logged.
            self.assertIsNone(self.bc._restore_tray_toggle_state())

    def test_non_dict_payload_returns_early(self):
        # A JSON list (not a dict) → `if not isinstance(persisted, dict)` early
        # return path.
        with open(self.hud_file, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        with mock.patch.object(self.bc, "ACTIONS", {}):
            self.assertIsNone(self.bc._restore_tray_toggle_state())

    def test_ambient_resume_hook_failure_is_caught(self):
        # ambient_mode_active True but the ACTIONS hook raises → the
        # `except Exception as e` around fn("") (2785-2786) prints, doesn't
        # propagate.
        with open(self.hud_file, "w", encoding="utf-8") as f:
            json.dump({"ambient_mode_active": True}, f)
        amb = [False]

        def _boom(_):
            raise RuntimeError("ambient start blew up")

        with mock.patch.object(self.bc, "_ambient_mode_active", amb), \
                mock.patch.object(self.bc, "ACTIONS",
                                  {"ambient_listen_start": _boom}):
            self.assertIsNone(self.bc._restore_tray_toggle_state())
        self.assertTrue(amb[0])

    def test_daemons_paused_pushes_into_owners(self):
        # daemons_paused True → exercises the diagnostic_daemons.pause +
        # skill_ambient_listen.set_paused push (2790-2801).
        with open(self.hud_file, "w", encoding="utf-8") as f:
            json.dump({"daemons_paused": True}, f)
        paused = [False]
        fake_diag = mock.Mock()
        fake_al = mock.Mock()
        # Make set_paused exist so the hasattr() guard passes.
        fake_al.set_paused = mock.Mock()
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "core.diagnostic_daemons" or name == "core":
                # `from core import diagnostic_daemons as _diag_daemons`
                if name == "core":
                    mod = mock.Mock()
                    mod.diagnostic_daemons = fake_diag
                    return mod
            return real_import(name, *a, **k)

        with mock.patch.object(self.bc, "_daemons_paused", paused), \
                mock.patch.object(self.bc, "ACTIONS", {}), \
                mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_ambient_listen": fake_al},
                                clear=False):
            self.assertIsNone(self.bc._restore_tray_toggle_state())
        self.assertTrue(paused[0])
        # The restore now RECONCILES the on-disk mirror instead of only ever
        # pushing the True case - see reconcile_paused's docstring.
        fake_diag.reconcile_paused.assert_called_once_with(True)
        fake_al.set_paused.assert_called_once_with(True)

    def test_daemons_paused_owner_failures_are_caught(self):
        # Both owner pushes raise → the two `except Exception as e` branches
        # (2793-2794, 2799-2800) print and continue.
        with open(self.hud_file, "w", encoding="utf-8") as f:
            json.dump({"daemons_paused": True}, f)
        paused = [False]
        fake_al = mock.Mock()
        fake_al.set_paused = mock.Mock(side_effect=RuntimeError("set fail"))
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "core":
                mod = mock.Mock()
                # accessing .diagnostic_daemons.pause_diagnostics raises
                dd = mock.Mock()
                dd.reconcile_paused.side_effect = RuntimeError("pause fail")
                mod.diagnostic_daemons = dd
                return mod
            return real_import(name, *a, **k)

        with mock.patch.object(self.bc, "_daemons_paused", paused), \
                mock.patch.object(self.bc, "ACTIONS", {}), \
                mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_ambient_listen": fake_al},
                                clear=False):
            # Both failures caught → returns cleanly.
            self.assertIsNone(self.bc._restore_tray_toggle_state())
        self.assertTrue(paused[0])


    def test_unpaused_boot_clears_a_stale_paused_mirror(self):
        """REGRESSION (found live on this box 2026-09-06).

        A stale ``"paused": true`` in data/diagnostic_daemons.json used to
        survive every restart. This block only ever pushed the True case, so a
        mirror left paused by a path the visible HUD flag never followed - a
        spoken pause_diagnostics, or a lever whose restore never ran because
        the box was power-cycled - could not be cleared by restarting JARVIS.
        It failed silently in the worst direction: all four loops stamp their
        alive_ts BEFORE testing paused, so every liveness check saw four happy
        heartbeats while the daemons did no work at all. Measured that day:
        last real work 94-101 days stale under a boot log printing
        ``paused=False``, with crash-watch among the daemons quietly disarmed.

        Drives the REAL core.diagnostic_daemons against a temp state file, so
        this proves the whole chain writes the cleared flag to disk rather
        than just asserting that a mock was called.
        """
        from core import diagnostic_daemons as real_dd
        state_path = os.path.join(self.tmp, "diagnostic_daemons.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"paused": True,
                       "crash_watch": {"last_seen_record_id": 4242,
                                       "detections": 7}}, f)
        with open(self.hud_file, "w", encoding="utf-8") as f:
            json.dump({"daemons_paused": False}, f)
        # Seeded True so a no-op restore is caught by the assert below.
        paused = [True]
        with mock.patch.object(real_dd, "STATE_FILE", state_path), \
                mock.patch.object(real_dd, "DATA_DIR", self.tmp), \
                mock.patch.object(self.bc, "_daemons_paused", paused), \
                mock.patch.object(self.bc, "ACTIONS", {}):
            self.assertIsNone(self.bc._restore_tray_toggle_state())
        self.assertFalse(paused[0], "hud daemons_paused=False not restored")
        with open(state_path, encoding="utf-8") as f:
            after = json.load(f)
        self.assertFalse(
            after["paused"],
            "stale paused mirror survived the restart - the daemons would "
            "keep heartbeating while doing no work")
        # Reconciling must touch ONLY `paused`: the crash-watch cursor is what
        # stops a re-seed from replaying every historical APPCRASH.
        self.assertEqual(after["crash_watch"]["last_seen_record_id"], 4242)
        self.assertEqual(after["crash_watch"]["detections"], 7)

    def test_paused_boot_still_writes_the_mirror_through(self):
        """The other direction still works: a real tray pause reaches disk."""
        from core import diagnostic_daemons as real_dd
        state_path = os.path.join(self.tmp, "diagnostic_daemons.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"paused": False}, f)
        with open(self.hud_file, "w", encoding="utf-8") as f:
            json.dump({"daemons_paused": True}, f)
        paused = [False]
        with mock.patch.object(real_dd, "STATE_FILE", state_path), \
                mock.patch.object(real_dd, "DATA_DIR", self.tmp), \
                mock.patch.object(self.bc, "_daemons_paused", paused), \
                mock.patch.object(self.bc, "ACTIONS", {}):
            self.assertIsNone(self.bc._restore_tray_toggle_state())
        self.assertTrue(paused[0])
        with open(state_path, encoding="utf-8") as f:
            self.assertTrue(json.load(f)["paused"])


# ───────────────────────────────────────────────────────────────────────────
#  _tray_state_publisher — exception handlers + calendar publish branch
#  (2849-2850, 2862, 2865-2866, 2875-2883, 2891-2892)
# ───────────────────────────────────────────────────────────────────────────
class TrayStatePublisherErrorTests(_MonolithSec2Base):
    def _one_shot_wait(self, stop):
        def _wait(_):
            stop.set()
            return True
        return _wait

    def test_publisher_swallows_iteration_exception(self):
        # Force the try-body to raise by making the `with _hud_state_lock:`
        # (inside the try) blow up on __enter__ → the outer
        # `except: logging.exception(...)` (2891-2892) runs.
        stop = self.bc._tray_publisher_stop
        stop.clear()
        boom_lock = mock.MagicMock()
        boom_lock.__enter__.side_effect = RuntimeError("lock blew up")
        with mock.patch.object(self.bc, "_hud_state_lock", boom_lock), \
                mock.patch.object(self.bc, "_write_hud_state"), \
                mock.patch.object(stop, "wait",
                                  side_effect=self._one_shot_wait(stop)), \
                mock.patch.object(self.bc, "_hud_cal_last", [time.time()]), \
                mock.patch.object(self.bc.logging, "exception") as logx:
            try:
                self.bc._tray_state_publisher()
            finally:
                stop.clear()
        logx.assert_called()

    def test_publisher_calendar_publish_branch(self):
        # _hud_cal_last far in the past → the calendar/mail publish block
        # (2872-2883) runs: imports hud_card, gathers events + unread mail,
        # writes them. hud_card is faked in sys.modules.
        stop = self.bc._tray_publisher_stop
        stop.clear()
        fake_hc = mock.Mock()
        fake_hc._gather_calendar.return_value = [{"title": "Standup"}]
        fake_hc._gather_unread_mail.return_value = 3
        writes = []
        with mock.patch.object(self.bc, "_write_hud_state",
                               side_effect=lambda **k: writes.append(k)), \
                mock.patch.dict(self.bc.sys.modules,
                                {"hud_card": fake_hc}, clear=False), \
                mock.patch.object(stop, "wait",
                                  side_effect=self._one_shot_wait(stop)), \
                mock.patch.object(self.bc, "_hud_cal_last", [0.0]):
            with self.bc._hud_state_lock:
                self.bc._hud_state_cache["alert_active"] = False
                self.bc._hud_state_cache["bambu_active"] = False
            try:
                self.bc._tray_state_publisher()
            finally:
                stop.clear()
        # The calendar publish wrote next_event + unread_mail.
        self.assertTrue(any("next_event" in w for w in writes))
        self.assertTrue(
            any(w.get("unread_mail") == 3 for w in writes))

    def test_publisher_calendar_failure_is_caught(self):
        # hud_card._gather_calendar raises → the calendar block's
        # `except: pass` (2882-2883) swallows it; no crash.
        stop = self.bc._tray_publisher_stop
        stop.clear()
        fake_hc = mock.Mock()
        fake_hc._gather_calendar.side_effect = RuntimeError("graph down")
        with mock.patch.object(self.bc, "_write_hud_state"), \
                mock.patch.dict(self.bc.sys.modules,
                                {"hud_card": fake_hc}, clear=False), \
                mock.patch.object(stop, "wait",
                                  side_effect=self._one_shot_wait(stop)), \
                mock.patch.object(self.bc, "_hud_cal_last", [0.0]):
            try:
                self.bc._tray_state_publisher()
            finally:
                stop.clear()

    def test_publisher_system_monitor_access_failure_is_caught(self):
        # skill_system_monitor present but attribute access raises inside the
        # try → the inner `except: pass` (2849-2850) swallows it.
        stop = self.bc._tray_publisher_stop
        stop.clear()
        sm = mock.Mock()
        # Reading _last_cpu_alert_at[0] raises (not subscriptable Mock attr).
        type(sm)._last_cpu_alert_at = mock.PropertyMock(
            side_effect=RuntimeError("boom"))
        with mock.patch.object(self.bc, "_write_hud_state"), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_system_monitor": sm}, clear=False), \
                mock.patch.object(stop, "wait",
                                  side_effect=self._one_shot_wait(stop)), \
                mock.patch.object(self.bc, "_hud_cal_last", [time.time()]):
            try:
                self.bc._tray_state_publisher()
            finally:
                stop.clear()

    def test_publisher_bambu_no_lock_path(self):
        # bambu monitor _state present but _state_lock is None → the
        # else-branch (read without lock, 2861-2862) runs.
        stop = self.bc._tray_publisher_stop
        stop.clear()
        bm = mock.Mock()
        bm._state = {"gcode_state": "RUNNING"}
        bm._state_lock = None
        writes = []
        with mock.patch.object(self.bc, "_write_hud_state",
                               side_effect=lambda **k: writes.append(k)), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_bambu_monitor": bm}, clear=False), \
                mock.patch.object(stop, "wait",
                                  side_effect=self._one_shot_wait(stop)), \
                mock.patch.object(self.bc, "_hud_cal_last", [time.time()]):
            with self.bc._hud_state_lock:
                self.bc._hud_state_cache["alert_active"] = False
                self.bc._hud_state_cache["bambu_active"] = False
            try:
                self.bc._tray_state_publisher()
            finally:
                stop.clear()
        self.assertTrue(any(w.get("bambu_active") is True for w in writes))


# ───────────────────────────────────────────────────────────────────────────
#  _detect_face — profile-mirror fallback pass (3019-3024) + escalation pass
# ───────────────────────────────────────────────────────────────────────────
class DetectFaceFallbackTests(_MonolithSec2Base):
    def _frame(self):
        import numpy as np
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_escalation_pass_finds_face(self):
        # Strict frontal returns empty on the FIRST call, then the escalation
        # pass (minNeighbors=3, larger minSize) returns a box on the 2nd call.
        import numpy as np
        frontal = mock.Mock()
        frontal.detectMultiScale.side_effect = [
            np.empty((0, 4)),                       # strict pass: nothing
            np.array([[100, 100, 80, 80]]),         # escalation: a face
        ]
        with mock.patch.object(self.bc, "_face_cascade", frontal), \
                mock.patch.object(self.bc, "_profile_cascade", None), \
                mock.patch.object(self.bc, "MIRROR_EYES_X", False), \
                mock.patch.object(self.bc, "MIRROR_EYES_Y", False):
            out = self.bc._detect_face(self._frame())
        self.assertIsNotNone(out)
        self.assertEqual(frontal.detectMultiScale.call_count, 2)

    def test_profile_mirror_fallback_used(self):
        # Frontal (both passes) empty; profile non-mirrored empty; profile on
        # the MIRRORED frame finds a box → exercises 3019-3024 (cv2.flip +
        # mirror coordinate remap).
        import numpy as np
        frontal = mock.Mock()
        frontal.detectMultiScale.return_value = np.empty((0, 4))
        profile = mock.Mock()
        profile.detectMultiScale.side_effect = [
            np.empty((0, 4)),                       # non-mirrored: nothing
            np.array([[10, 50, 60, 60]]),           # mirrored: a face
        ]
        cv2 = mock.Mock()
        cv2.COLOR_BGR2GRAY = 6
        # cvtColor / equalizeHist / flip just return a usable ndarray.
        gray = np.zeros((480, 640), dtype=np.uint8)
        cv2.cvtColor.return_value = gray
        cv2.equalizeHist.return_value = gray
        cv2.flip.return_value = gray
        with mock.patch.object(self.bc, "cv2", cv2), \
                mock.patch.object(self.bc, "_face_cascade", frontal), \
                mock.patch.object(self.bc, "_profile_cascade", profile), \
                mock.patch.object(self.bc, "MIRROR_EYES_X", False), \
                mock.patch.object(self.bc, "MIRROR_EYES_Y", False):
            out = self.bc._detect_face(self._frame())
        self.assertIsNotNone(out)
        cv2.flip.assert_called_once()
        # Mirror remap: original mirrored x=10,w=60 → (640 - 10 - 60)=570 centre
        fx, _ = out
        self.assertAlmostEqual(fx, (570 + 60 / 2) / 640, places=3)

    def test_profile_mirror_also_empty_returns_none(self):
        # Every cascade pass empty (incl. the mirrored profile) → None.
        import numpy as np
        frontal = mock.Mock()
        frontal.detectMultiScale.return_value = np.empty((0, 4))
        profile = mock.Mock()
        profile.detectMultiScale.return_value = np.empty((0, 4))
        cv2 = mock.Mock()
        cv2.COLOR_BGR2GRAY = 6
        gray = np.zeros((480, 640), dtype=np.uint8)
        cv2.cvtColor.return_value = gray
        cv2.equalizeHist.return_value = gray
        cv2.flip.return_value = gray
        with mock.patch.object(self.bc, "cv2", cv2), \
                mock.patch.object(self.bc, "_face_cascade", frontal), \
                mock.patch.object(self.bc, "_profile_cascade", profile):
            self.assertIsNone(self.bc._detect_face(self._frame()))

    def test_mirror_y_flips(self):
        # Exercise the MIRROR_EYES_Y branch (fy = 1.0 - fy).
        import numpy as np
        fake = mock.Mock()
        fake.detectMultiScale.return_value = np.array([[0, 0, 64, 64]])
        with mock.patch.object(self.bc, "_face_cascade", fake), \
                mock.patch.object(self.bc, "_profile_cascade", None), \
                mock.patch.object(self.bc, "MIRROR_EYES_X", False), \
                mock.patch.object(self.bc, "MIRROR_EYES_Y", True):
            _, fy = self.bc._detect_face(self._frame())
        self.assertAlmostEqual(fy, 1.0 - (32 / 480), places=3)


# ───────────────────────────────────────────────────────────────────────────
#  _probe_camera_index — worker body internals (3060-3076)
# ───────────────────────────────────────────────────────────────────────────
class ProbeCameraIndexWorkerTests(_MonolithSec2Base):
    def test_worker_exception_during_open_reports_false(self):
        # cv2.VideoCapture raising inside the worker → the worker's
        # `except: pass` (3060-3061) leaves result['ok']=False, release in
        # the finally still runs cleanly.
        cv2 = mock.Mock()
        cv2.VideoCapture.side_effect = RuntimeError("dshow exploded")
        cv2.CAP_DSHOW = 700
        with mock.patch.object(self.bc, "cv2", cv2):
            self.assertFalse(self.bc._probe_camera_index(0, timeout_sec=2.0))

    def test_worker_release_exception_swallowed(self):
        # Open succeeds + frame read True, but release() raises → the finally's
        # `except: pass` (3066-3067) swallows it; still returns True.
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, object())
        cap.release.side_effect = RuntimeError("release boom")
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        with mock.patch.object(self.bc, "cv2", cv2):
            self.assertTrue(self.bc._probe_camera_index(0, timeout_sec=2.0))

    def test_wedged_worker_reports_false(self):
        # Worker thread never returns within timeout → t.is_alive() True →
        # the wedge branch (3072-3076) returns False without waiting forever.
        # Simulate a wedge by making VideoCapture block until we release it.
        release = threading.Event()
        cv2 = mock.Mock()

        def _hang(*a, **k):
            release.wait(5.0)   # blocks well past the 0.05s join timeout
            cap = mock.Mock()
            cap.isOpened.return_value = False
            return cap

        cv2.VideoCapture.side_effect = _hang
        cv2.CAP_DSHOW = 700
        try:
            with mock.patch.object(self.bc, "cv2", cv2):
                # Tiny timeout → join() returns while the worker is still in _hang.
                self.assertFalse(
                    self.bc._probe_camera_index(9, timeout_sec=0.05))
        finally:
            release.set()   # let the daemon worker unwind


# ───────────────────────────────────────────────────────────────────────────
#  probe_cameras_and_update_config — _runner failure + no-camera suspect
#  branches (3125-3127, 3178-3186)
# ───────────────────────────────────────────────────────────────────────────
class ProbeCamerasExtraBranchTests(_MonolithSec2Base):
    def setUp(self):
        self._saved_cams = [dict(c) for c in self.bc.CAMERAS]

    def tearDown(self):
        self.bc.CAMERAS[:] = self._saved_cams

    def test_runner_exception_marks_index_failed(self):
        # _probe_camera_index raises for the configured index → the inner
        # _runner `except: ... results[i] = False` (3125-3127) path; the
        # index is then reported failed.
        cams = [{"index": 1, "label": "L", "primary": True,
                 "look_x": 0.15, "look_y": 0.5}]
        with mock.patch.object(self.bc, "CAMERA_PROBE_ENABLED", True), \
                mock.patch.object(self.bc, "CAMERA_PROBE_MAX", 1), \
                mock.patch.object(self.bc, "CAMERAS", cams), \
                mock.patch.object(self.bc, "_probe_camera_index",
                                  side_effect=RuntimeError("probe raised")), \
                mock.patch.object(self.bc, "find_camera_locking_processes",
                                  return_value=[]), \
                mock.patch.object(self.bc.logging, "exception"):
            working, failed = self.bc.probe_cameras_and_update_config()
        self.assertEqual(working, [])
        self.assertIn(1, failed)

    def test_sweep_empty_with_suspects_prints_hint(self):
        # Configured fails, no lock-holder at step-3, sweep finds nothing, and
        # THEN a suspect appears at step-4's recheck (3178-3182).
        cams = [{"index": 5, "label": "L", "primary": True,
                 "look_x": 0.15, "look_y": 0.5}]
        # First find_camera_locking_processes() (step 3) → none; second call
        # (after empty sweep) → a suspect.
        suspect_calls = [[], ["teams.exe"]]
        with mock.patch.object(self.bc, "CAMERA_PROBE_ENABLED", True), \
                mock.patch.object(self.bc, "CAMERA_PROBE_MAX", 3), \
                mock.patch.object(self.bc, "CAMERAS", cams), \
                mock.patch.object(self.bc, "_probe_camera_index",
                                  return_value=False), \
                mock.patch.object(self.bc, "find_camera_locking_processes",
                                  side_effect=suspect_calls):
            working, failed = self.bc.probe_cameras_and_update_config()
        self.assertEqual(working, [])
        self.assertIn(5, failed)

    def test_sweep_empty_no_suspects_prints_unplugged_hint(self):
        # Configured fails, sweep empty, and NO suspects either time → the
        # else "webcams may be unplugged" branch (3184-3186).
        cams = [{"index": 5, "label": "L", "primary": True,
                 "look_x": 0.15, "look_y": 0.5}]
        with mock.patch.object(self.bc, "CAMERA_PROBE_ENABLED", True), \
                mock.patch.object(self.bc, "CAMERA_PROBE_MAX", 3), \
                mock.patch.object(self.bc, "CAMERAS", cams), \
                mock.patch.object(self.bc, "_probe_camera_index",
                                  return_value=False), \
                mock.patch.object(self.bc, "find_camera_locking_processes",
                                  return_value=[]):
            working, failed = self.bc.probe_cameras_and_update_config()
        self.assertEqual(working, [])
        self.assertIn(5, failed)

    def test_sweep_finds_two_cameras_marks_left_right(self):
        # Sweep finds indices 2 AND 3 → both new_cams entries built (the
        # i==0 primary and the else secondary branch, 3191-3201).
        cams = [{"index": 5, "label": "L", "primary": True,
                 "look_x": 0.15, "look_y": 0.5}]

        def probe(i, *a, **k):
            return i in (2, 3)

        with mock.patch.object(self.bc, "CAMERA_PROBE_ENABLED", True), \
                mock.patch.object(self.bc, "CAMERA_PROBE_MAX", 4), \
                mock.patch.object(self.bc, "CAMERAS", cams), \
                mock.patch.object(self.bc, "_probe_camera_index",
                                  side_effect=probe), \
                mock.patch.object(self.bc, "find_camera_locking_processes",
                                  return_value=[]):
            working, failed = self.bc.probe_cameras_and_update_config()
            rewritten = list(self.bc.CAMERAS)
        self.assertCountEqual(working, [2, 3])
        self.assertEqual(rewritten[0]["index"], 2)
        self.assertTrue(rewritten[0]["primary"])
        self.assertEqual(rewritten[1]["index"], 3)
        self.assertFalse(rewritten[1]["primary"])


# ───────────────────────────────────────────────────────────────────────────
#  find_camera_locking_processes — process_iter raising (3097-3100)
# ───────────────────────────────────────────────────────────────────────────
class CameraLockProcessErrorTests(_MonolithSec2Base):
    def test_process_iter_exception_returns_empty(self):
        fake_psutil = mock.Mock()
        fake_psutil.process_iter.side_effect = RuntimeError("psutil broke")
        fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "psutil":
                return fake_psutil
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            self.assertEqual(self.bc.find_camera_locking_processes(), [])

    def test_per_proc_nosuchprocess_skipped(self):
        # One proc raises NoSuchProcess on attribute access → the per-proc
        # `except (NoSuchProcess, AccessDenied): continue` (3097-3098) skips it,
        # the next good proc is still collected.
        NoSuch = type("NoSuchProcess", (Exception,), {})
        Denied = type("AccessDenied", (Exception,), {})
        fake_psutil = mock.Mock()
        fake_psutil.NoSuchProcess = NoSuch
        fake_psutil.AccessDenied = Denied

        class _BadProc:
            @property
            def info(self):
                raise NoSuch("gone")

        class _GoodProc:
            info = {"name": "zoom.exe"}

        fake_psutil.process_iter.return_value = [_BadProc(), _GoodProc()]
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "psutil":
                return fake_psutil
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch.object(self.bc, "CAMERA_LOCK_PROCESSES",
                                  {"zoom.exe"}):
            out = self.bc.find_camera_locking_processes()
        self.assertEqual(out, ["zoom.exe"])


# ───────────────────────────────────────────────────────────────────────────
#  list_monitors_cli — ABOVE / BELOW position-guess branches (3563, 3565)
# ───────────────────────────────────────────────────────────────────────────
class ListMonitorsGuessTests(_MonolithSec2Base):
    def test_above_and_below_guesses(self):
        # y<0 → "ABOVE"; y>0 → "BELOW". Both lines (3563 / 3565) exercised.
        mons = [(0, -1080, 1920, 1080), (0, 1080, 1920, 1080)]
        with mock.patch.object(self.bc, "get_monitors", return_value=mons):
            self.bc.list_monitors_cli()   # prints both guesses, no raise

    def test_left_of_primary_guess(self):
        # x<0, y==0 → the final "to the LEFT of primary" branch (3571).
        mons = [(-1920, 0, 1920, 1080)]
        with mock.patch.object(self.bc, "get_monitors", return_value=mons):
            self.bc.list_monitors_cli()


# ───────────────────────────────────────────────────────────────────────────
#  _refresh_devices — wake-word pause/resume + destructive reinit + its error
#  branches (3895-3900, 3927-3928, 3979-3982)
# ───────────────────────────────────────────────────────────────────────────
class SilentMicReportingTests(_MonolithSec2Base):
    """_report_silent_mic — the escalation the [self-heal] silent-mic detector
    never had.

    2026-08-20 LOW finding. The detector printed ONE line per process and did
    nothing else. Two things made that worse than it looks: the reset arm is
    `elif rms > 1e-5` and core.audio_processor._AUDIBLE_RMS_FLOOR is that same
    1e-5, so a device returning true digital silence could never clear the
    latch; and JARVIS normally runs under pythonw, where nobody reads the
    console. The one fault the owner cannot ask about — the microphone — was
    the one reported only in writing.
    """

    def setUp(self):
        # MonolithGlobalsTestCase deep-restores these after each test (they are
        # listed in tests/_monolith_harness.py), so plain assignment is safe.
        self.bc._silent_mic_warned[0] = False
        self.bc._silent_mic_warned_at[0] = 0.0
        self.bc._silent_mic_warned_device[0] = ""

    def test_first_report_speaks_and_prints(self):
        with mock.patch.object(self.bc, "proactive_announce") as msay, \
                mock.patch.object(self.bc, "_device_cache",
                                  {"last_in_name": "Yeti X"}), \
                mock.patch("builtins.print") as mprint:
            self.assertTrue(self.bc._report_silent_mic(45.0, 1000.0))
        msay.assert_called_once()
        said = msay.call_args[0][0]
        self.assertIn("silent", said.lower())
        self.assertIn("Yeti", said)
        self.assertTrue(
            any("silent-mic" in " ".join(str(a) for a in c.args)
                for c in mprint.call_args_list),
            "the console line must survive — speech can be muted or focused")

    def test_repeat_inside_the_window_is_throttled(self):
        with mock.patch.object(self.bc, "proactive_announce") as msay, \
                mock.patch.object(self.bc, "_device_cache",
                                  {"last_in_name": "Yeti X"}), \
                mock.patch("builtins.print"):
            self.assertTrue(self.bc._report_silent_mic(45.0, 1000.0))
            self.assertFalse(self.bc._report_silent_mic(60.0, 1000.0 + 10))
            self.assertFalse(self.bc._report_silent_mic(
                600.0, 1000.0 + self.bc.MIC_SILENT_REWARN_SECONDS - 1))
        msay.assert_called_once()

    def test_it_re_warns_after_the_window(self):
        """THE regression: the old latch was one-shot for the life of the
        process, so a fault that started at 09:00 was never mentioned again."""
        with mock.patch.object(self.bc, "proactive_announce") as msay, \
                mock.patch.object(self.bc, "_device_cache",
                                  {"last_in_name": "Yeti X"}), \
                mock.patch("builtins.print"):
            self.assertTrue(self.bc._report_silent_mic(45.0, 1000.0))
            self.assertTrue(self.bc._report_silent_mic(
                9000.0, 1000.0 + self.bc.MIC_SILENT_REWARN_SECONDS + 1))
        self.assertEqual(msay.call_count, 2)

    def test_a_different_silent_device_re_arms_immediately(self):
        # Following the Windows default to a NEW endpoint that is also silent
        # is new information, not a repeat.
        cache = {"last_in_name": "Yeti X"}
        with mock.patch.object(self.bc, "proactive_announce") as msay, \
                mock.patch.object(self.bc, "_device_cache", cache), \
                mock.patch("builtins.print"):
            self.assertTrue(self.bc._report_silent_mic(45.0, 1000.0))
            cache["last_in_name"] = "Realtek USB Audio"
            self.assertTrue(self.bc._report_silent_mic(45.0, 1001.0))
        self.assertEqual(msay.call_count, 2)

    def test_it_does_not_pin_the_input_device(self):
        """It must not 'demote' a silent input by writing _device_cache["in"].

        Since the follow-the-default contract, that key is deliberately None so
        every stream open re-resolves the Windows default; pinning a name is
        exactly what left this box deaf for 90 minutes behind a powered-off
        headset that still enumerated and still passed check_input_settings."""
        cache = {"last_in_name": "Yeti X", "in": None, "checked_at": 123.0}
        with mock.patch.object(self.bc, "proactive_announce"), \
                mock.patch.object(self.bc, "_device_cache", cache), \
                mock.patch("builtins.print"):
            self.bc._report_silent_mic(45.0, 1000.0)
        self.assertIsNone(cache["in"])
        self.assertEqual(cache["checked_at"], 123.0)

    def test_a_failing_announcement_never_escapes_the_capture_loop(self):
        with mock.patch.object(self.bc, "proactive_announce",
                               side_effect=RuntimeError("queue full")), \
                mock.patch.object(self.bc, "_device_cache",
                                  {"last_in_name": "Yeti X"}), \
                mock.patch("builtins.print"):
            self.assertTrue(self.bc._report_silent_mic(45.0, 1000.0))

    def test_an_unreadable_device_cache_still_reports(self):
        broken = mock.Mock()
        broken.get.side_effect = RuntimeError("cache boom")
        with mock.patch.object(self.bc, "proactive_announce") as msay, \
                mock.patch.object(self.bc, "_device_cache", broken), \
                mock.patch("builtins.print"):
            self.assertTrue(self.bc._report_silent_mic(45.0, 1000.0))
        msay.assert_called_once()


class WakeWordResumeOrReportTests(_MonolithSec2Base):
    """_wake_word_resume_or_report — the honest replacement for a bare
    `try: paused_det.resume() except Exception: print(...)`.

    Why the two obvious fixes are wrong, pinned here so they are not
    reintroduced:
      * "retry resume()" is a no-op. Detector.resume() opens with
        `if not self._paused: return False`, and the failing path already
        cleared _paused before it failed. start() is the only real retry.
      * "clear skill_wake_listener._detector so the next refresh rebuilds it"
        does nothing — _refresh_devices never builds a detector, it only pauses
        the one it finds; clearing the global guarantees it is never restarted.
    """

    def test_clean_resume_needs_no_restart_and_no_announcement(self):
        det = mock.Mock()
        det.resume.return_value = True
        with mock.patch.object(self.bc,
                               "_enqueue_device_announcement") as msay:
            self.assertTrue(self.bc._wake_word_resume_or_report(det))
        det.start.assert_not_called()
        msay.assert_not_called()

    def test_false_resume_escalates_to_start(self):
        det = mock.Mock()
        det.resume.return_value = False
        det.start.return_value = True
        with mock.patch.object(self.bc,
                               "_enqueue_device_announcement") as msay, \
                mock.patch("builtins.print"):
            self.assertTrue(self.bc._wake_word_resume_or_report(det))
        det.start.assert_called_once()
        msay.assert_not_called()

    def test_raising_resume_also_escalates_to_start(self):
        det = mock.Mock()
        det.resume.side_effect = RuntimeError("stream gone")
        det.start.return_value = True
        with mock.patch("builtins.print"):
            self.assertTrue(self.bc._wake_word_resume_or_report(det))
        det.start.assert_called_once()

    def test_total_failure_is_reported_out_loud_and_returns_false(self):
        det = mock.Mock()
        det.resume.return_value = False
        det.start.return_value = False
        with mock.patch.object(self.bc,
                               "_enqueue_device_announcement") as msay, \
                mock.patch("builtins.print") as mprint:
            self.assertFalse(self.bc._wake_word_resume_or_report(det))
        msay.assert_called_once()
        self.assertTrue(
            any("WAKE-WORD DETECTOR IS DOWN" in " ".join(str(a) for a in c.args)
                for c in mprint.call_args_list))

    def test_raising_start_is_contained(self):
        det = mock.Mock()
        det.resume.return_value = False
        det.start.side_effect = RuntimeError("engine gone")
        with mock.patch.object(self.bc,
                               "_enqueue_device_announcement") as msay, \
                mock.patch("builtins.print"):
            self.assertFalse(self.bc._wake_word_resume_or_report(det))
        msay.assert_called_once()

    def test_a_failing_announcement_does_not_escape(self):
        # This runs inside _refresh_devices' finally, with the device-refresh
        # lock held; an exception here would leave the lock in an odd state and
        # break every get_input_device() caller.
        det = mock.Mock()
        det.resume.return_value = False
        det.start.return_value = False
        with mock.patch.object(self.bc, "_enqueue_device_announcement",
                               side_effect=RuntimeError("queue full")), \
                mock.patch("builtins.print"):
            self.assertFalse(self.bc._wake_word_resume_or_report(det))


class RefreshDevicesReinitTests(_MonolithSec2Base):
    def setUp(self):
        self._saved_cache = dict(self.bc._device_cache)

    def tearDown(self):
        self.bc._device_cache.clear()
        self.bc._device_cache.update(self._saved_cache)

    def _detector(self, running=True, pause_raises=False, resume_raises=False):
        det = mock.Mock()
        det.is_running.return_value = running
        if pause_raises:
            det.pause.side_effect = RuntimeError("pause fail")
        if resume_raises:
            det.resume.side_effect = RuntimeError("resume fail")
        return det

    def test_wake_word_paused_and_resumed_around_reinit(self):
        # A running wake-word detector is paused before sd._terminate() and
        # resumed in the finally (3894-3900 + 3978-3982).
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = None
        self.bc._device_cache["last_out_name"] = None
        det = self._detector(running=True)
        wl = mock.Mock()
        wl._detector = det
        sd = mock.Mock()
        sd.query_devices.return_value = {"name": "ignored"}
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_wake_listener": wl}, clear=False), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None), \
                mock.patch.object(self.bc, "_pick_device",
                                  side_effect=[(0, "USB Mic"), (1, "Realtek")]):
            self.bc._refresh_devices(force=True)
        det.pause.assert_called_once()
        det.resume.assert_called_once()
        sd._terminate.assert_called_once()
        sd._initialize.assert_called_once()

    def test_wake_word_pause_failure_is_caught(self):
        # det.pause() raises → the `except Exception as e` (3899-3900) prints;
        # paused_det stays None so no resume is attempted, refresh completes.
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = None
        self.bc._device_cache["last_out_name"] = None
        det = self._detector(running=True, pause_raises=True)
        wl = mock.Mock()
        wl._detector = det
        sd = mock.Mock()
        sd.query_devices.return_value = {"name": "ignored"}
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_wake_listener": wl}, clear=False), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None), \
                mock.patch.object(self.bc, "_pick_device",
                                  side_effect=[(0, "USB Mic"), (1, "Realtek")]):
            self.bc._refresh_devices(force=True)
        det.pause.assert_called_once()
        det.resume.assert_not_called()   # never paused → never resumed

    def test_wake_word_resume_failure_is_caught(self):
        # Detector paused OK but resume() raises → the finally's
        # `except Exception as e` (3981-3982) swallows it.
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = None
        self.bc._device_cache["last_out_name"] = None
        det = self._detector(running=True, resume_raises=True)
        wl = mock.Mock()
        wl._detector = det
        sd = mock.Mock()
        sd.query_devices.return_value = {"name": "ignored"}
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_wake_listener": wl}, clear=False), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None), \
                mock.patch.object(self.bc, "_pick_device",
                                  side_effect=[(0, "USB Mic"), (1, "Realtek")]):
            self.bc._refresh_devices(force=True)
        det.resume.assert_called_once()

    def test_resume_returning_false_restarts_and_announces(self):
        """THE 2026-08-20 finding. core/wake_word.Detector.resume() does NOT
        raise on failure — it clears _running and returns False — so the old
        `try: resume() except Exception:` caught nothing that ever happens and
        the boolean was discarded. With _running clear, the pause site's
        `if det.is_running()` guard skips the detector on every later refresh,
        so a single transient endpoint blip disarmed acoustic barge-in for the
        whole session, and the only record was one line in a log nobody reads
        under pythonw."""
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = None
        self.bc._device_cache["last_out_name"] = None
        det = self._detector(running=True)
        det.resume.return_value = False
        det.start.return_value = True          # the restart takes
        wl = mock.Mock()
        wl._detector = det
        sd = mock.Mock()
        sd.query_devices.return_value = {"name": "ignored"}
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_wake_listener": wl}, clear=False), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None), \
                mock.patch.object(self.bc, "_pick_device",
                                  side_effect=[(0, "USB Mic"), (1, "Realtek")]), \
                mock.patch.object(self.bc,
                                  "_enqueue_device_announcement") as msay:
            self.bc._refresh_devices(force=True)
        det.resume.assert_called_once()
        det.start.assert_called_once()
        # It came back, so the owner must NOT be told barge-in is down.
        msay.assert_not_called()

    def test_permanent_resume_failure_is_announced_to_the_owner(self):
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = None
        self.bc._device_cache["last_out_name"] = None
        det = self._detector(running=True)
        det.resume.return_value = False
        det.start.return_value = False         # and the restart fails too
        wl = mock.Mock()
        wl._detector = det
        sd = mock.Mock()
        sd.query_devices.return_value = {"name": "ignored"}
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_wake_listener": wl}, clear=False), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None), \
                mock.patch.object(self.bc, "_pick_device",
                                  side_effect=[(0, "USB Mic"), (1, "Realtek")]), \
                mock.patch.object(self.bc,
                                  "_enqueue_device_announcement") as msay:
            self.bc._refresh_devices(force=True)
        det.start.assert_called_once()
        msay.assert_called_once()
        said = msay.call_args[0][0].lower()
        self.assertIn("wake-word", said)
        self.assertIn("barge-in", said)

    def test_portaudio_reinit_failure_is_caught(self):
        # sd._terminate() raises → the `except Exception as e` (3927-3928)
        # prints "PortAudio re-init failed" and the refresh still finishes
        # picking devices.
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = None
        self.bc._device_cache["last_out_name"] = None
        sd = mock.Mock()
        sd._terminate.side_effect = RuntimeError("portaudio terminate boom")
        sd.query_devices.return_value = {"name": "ignored"}
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.dict(self.bc.sys.modules, {}, clear=False), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None), \
                mock.patch.object(self.bc, "_pick_device",
                                  side_effect=[(0, "USB Mic"), (1, "Realtek")]):
            # No skill_wake_listener → det is None, pause path skipped.
            self.bc.sys.modules.pop("skill_wake_listener", None)
            self.bc._refresh_devices(force=True)
        self.assertEqual(self.bc._device_cache["in"], 0)
        self.assertEqual(self.bc._device_cache["out"], 1)

    def test_explicit_indices_query_name_failure_falls_back_empty(self):
        # MICROPHONE_INDEX / SPEAKER_INDEX set but sd.query_devices(idx) raises
        # → the `except: in_name = ""` / `out_name = ""` branches (3934, 3942).
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = None
        self.bc._device_cache["last_out_name"] = None
        sd = mock.Mock()
        sd.query_devices.side_effect = RuntimeError("no such device")
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.dict(self.bc.sys.modules, {}, clear=False), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", 3), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", 4), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None):
            self.bc.sys.modules.pop("skill_wake_listener", None)
            self.bc._refresh_devices(force=True)
        # Indices still cached even though the names couldn't be resolved.
        self.assertEqual(self.bc._device_cache["in"], 3)
        self.assertEqual(self.bc._device_cache["out"], 4)

    def test_headset_drop_message_phrasing(self):
        # prev mic name contains "headset" AND the headset is GONE from the
        # device list → the headset-specific loss message fires.
        #
        # 2026-08-20 (follow-the-default): the loss wording is now gated on the
        # previous endpoint having actually VANISHED from the enumeration, not
        # on its name containing "headset" — a deliberate Stream Deck switch
        # away from a still-present headset gets the neutral line instead (see
        # FollowTheDefaultDeviceTests). So this test must arrange a real
        # disappearance: the mocked device list contains only the new mic.
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = "Gaming Headset"
        self.bc._device_cache["last_in_index"] = 9
        self.bc._device_cache["last_out_name"] = None
        announced = []
        sd = mock.Mock()

        def _qd(idx=None, **kw):
            if idx is None:
                return [{"name": "Fallback Laptop Mic",
                         "max_input_channels": 1, "max_output_channels": 0},
                        {"name": "Speakers",
                         "max_input_channels": 0, "max_output_channels": 2}]
            return {"name": "ignored"}
        sd.query_devices.side_effect = _qd
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.dict(self.bc.sys.modules, {}, clear=False), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None), \
                mock.patch.object(self.bc, "_pick_device",
                                  side_effect=[(0, "Fallback Laptop Mic"),
                                               (1, "Speakers")]), \
                mock.patch.object(self.bc, "_enqueue_device_announcement",
                                  side_effect=announced.append):
            self.bc.sys.modules.pop("skill_wake_listener", None)
            self.bc._refresh_devices(force=True)
        self.assertEqual(len(announced), 1)
        self.assertIn("headset", announced[0].lower())


# ───────────────────────────────────────────────────────────────────────────
#  get_output_device / get_current_speaker_name — None + error branches
#  (4020, 4062-4063)
# ───────────────────────────────────────────────────────────────────────────
class DeviceAccessorExtraTests(_MonolithSec2Base):
    def setUp(self):
        self._saved_cache = dict(self.bc._device_cache)

    def tearDown(self):
        self.bc._device_cache.clear()
        self.bc._device_cache.update(self._saved_cache)

    def test_get_output_device_none_when_cache_none(self):
        self.bc._device_cache["out"] = None
        with mock.patch.object(self.bc, "_refresh_devices"):
            self.assertIsNone(self.bc.get_output_device())

    def test_get_current_speaker_name_unknown_on_error(self):
        sd = mock.Mock()
        sd.query_devices.side_effect = RuntimeError("gone")
        self.bc._device_cache["out"] = 2
        with mock.patch.object(self.bc, "sd", sd):
            self.assertEqual(self.bc.get_current_speaker_name(), "[2] (unknown)")


# ───────────────────────────────────────────────────────────────────────────
#  list_cameras — lock-holder hint + opened-no-frame + dark-frame quality
#  (4080-4081, 4111-4115, 4132)
# ───────────────────────────────────────────────────────────────────────────
class ListCamerasExtraTests(_MonolithSec2Base):
    def test_lock_holder_hint_printed(self):
        # find_camera_locking_processes returns a suspect → the warning hint
        # (4080-4081) prints before scanning.
        cap = mock.Mock()
        cap.isOpened.return_value = False
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        tmp = tempfile.mkdtemp(prefix="jarvis_cam_lock_")
        try:
            with mock.patch.object(self.bc, "cv2", cv2), \
                    mock.patch.object(self.bc, "find_camera_locking_processes",
                                      return_value=["teams.exe"]), \
                    mock.patch.object(self.bc.time, "sleep"), \
                    mock.patch.object(self.bc.os.path, "dirname",
                                      return_value=tmp):
                self.bc.list_cameras(max_check=1)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_opened_but_no_frame_status(self):
        # Camera opens but read() never returns a frame → "opened but no frame"
        # status branch (4131-4132).
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)   # warm-up + final read both fail
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        tmp = tempfile.mkdtemp(prefix="jarvis_cam_noframe_")
        try:
            with mock.patch.object(self.bc, "cv2", cv2), \
                    mock.patch.object(self.bc, "find_camera_locking_processes",
                                      return_value=[]), \
                    mock.patch.object(self.bc.time, "sleep"), \
                    mock.patch.object(self.bc.os.path, "dirname",
                                      return_value=tmp):
                self.bc.list_cameras(max_check=1)
            cv2.imwrite.assert_not_called()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dark_frame_marked_black(self):
        # A frame with mean brightness <= 10 → the "⚠ BLACK / blocked" quality
        # label (4129). Still writes a preview.
        import numpy as np
        dark = np.zeros((1080, 1920, 3), dtype=np.uint8)   # mean 0
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, dark)
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        tmp = tempfile.mkdtemp(prefix="jarvis_cam_dark_")
        try:
            with mock.patch.object(self.bc, "cv2", cv2), \
                    mock.patch.object(self.bc, "find_camera_locking_processes",
                                      return_value=[]), \
                    mock.patch.object(self.bc.time, "sleep"), \
                    mock.patch.object(self.bc.os.path, "dirname",
                                      return_value=tmp):
                self.bc.list_cameras(max_check=1)
            cv2.imwrite.assert_called()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_open_worker_read_exception_swallowed(self):
        # cap opens (isOpened True) but cap.read() raises → the worker's
        # try/except (4111-4112) swallows it and the finally still releases
        # (4114-4115). opened True but ret False → "opened but no frame".
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.read.side_effect = RuntimeError("read exploded")
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700
        tmp = tempfile.mkdtemp(prefix="jarvis_cam_exc_")
        try:
            with mock.patch.object(self.bc, "cv2", cv2), \
                    mock.patch.object(self.bc, "find_camera_locking_processes",
                                      return_value=[]), \
                    mock.patch.object(self.bc.time, "sleep"), \
                    mock.patch.object(self.bc.os.path, "dirname",
                                      return_value=tmp):
                self.bc.list_cameras(max_check=1)
            cv2.imwrite.assert_not_called()
            cap.release.assert_called()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


# ───────────────────────────────────────────────────────────────────────────
#  should_be_proactive — voice-mood stress-suppression gate (4165-4170)
# ───────────────────────────────────────────────────────────────────────────
class ProactiveVoiceMoodTests(_MonolithSec2Base):
    def setUp(self):
        self._saved_last_speech = self.bc.last_speech_time
        self._saved_last_face = self.bc.last_face_seen

    def tearDown(self):
        self.bc.last_speech_time = self._saved_last_speech
        self.bc.last_face_seen = self._saved_last_face

    def test_stress_suppression_blocks(self):
        # _voice_mood_response.is_stress_suppression_active True → returns False
        # before the face/probability checks (4164-4170).
        vmr = mock.Mock()
        vmr.is_stress_suppression_active.return_value = True
        with mock.patch.object(self.bc, "PROACTIVE_ENABLED", True), \
                mock.patch.object(self.bc, "PROACTIVE_MIN_SILENCE", 1), \
                mock.patch.object(self.bc, "_voice_mood_response", vmr), \
                mock.patch.object(self.bc, "load_memory", return_value={}):
            self.bc.last_speech_time = time.time() - 1000
            self.assertFalse(self.bc.should_be_proactive())
        vmr.is_stress_suppression_active.assert_called_once()

    def test_stress_suppression_check_error_is_caught(self):
        # The suppression check raising → the `except: pass` (4169-4170) is
        # swallowed and evaluation continues (face required, none → False).
        vmr = mock.Mock()
        vmr.is_stress_suppression_active.side_effect = RuntimeError("mem boom")
        with mock.patch.object(self.bc, "PROACTIVE_ENABLED", True), \
                mock.patch.object(self.bc, "PROACTIVE_MIN_SILENCE", 1), \
                mock.patch.object(self.bc, "PROACTIVE_REQUIRE_FACE", True), \
                mock.patch.object(self.bc, "_voice_mood_response", vmr), \
                mock.patch.object(self.bc, "load_memory", return_value={}):
            self.bc.last_speech_time = time.time() - 1000
            self.bc.last_face_seen = 0.0   # never seen → fails face gate
            self.assertFalse(self.bc.should_be_proactive())

    def test_face_required_but_stale_returns_false(self):
        # last_face_seen older than 60s with REQUIRE_FACE → the stale-face gate
        # (4174-4175) returns False.
        with mock.patch.object(self.bc, "PROACTIVE_ENABLED", True), \
                mock.patch.object(self.bc, "PROACTIVE_MIN_SILENCE", 1), \
                mock.patch.object(self.bc, "PROACTIVE_REQUIRE_FACE", True), \
                mock.patch.object(self.bc, "_voice_mood_response", None):
            self.bc.last_speech_time = time.time() - 1000
            self.bc.last_face_seen = time.time() - 120   # stale
            self.assertFalse(self.bc.should_be_proactive())


# ───────────────────────────────────────────────────────────────────────────
#  _set_late_night_suppression — save_memory failure path (4275-4276)
# ───────────────────────────────────────────────────────────────────────────
class LateNightSuppressionErrorTests(_MonolithSec2Base):
    def test_save_failure_is_caught(self):
        # save_memory raising → the `except Exception as e` (4275-4276) prints
        # but the in-memory flag is still set.
        mem = {}
        with mock.patch.object(self.bc, "save_memory",
                               side_effect=RuntimeError("disk gone")):
            self.bc._set_late_night_suppression(mem)
        self.assertIn("late_night_no_comments_until", mem)


# ───────────────────────────────────────────────────────────────────────────
#  _thinking_loop — heartbeat tick cadence (4351) + watchdog thread iteration
#  exception (4434-4440)
# ───────────────────────────────────────────────────────────────────────────
class ThinkingLoopHeartbeatTests(_MonolithSec2Base):
    def test_heartbeat_fires_after_20_iterations(self):
        # The loop ticks _heartbeat() every 20th iteration (`_i % 20 == 0`,
        # line 4350-4351). Drive 20+ iterations so the heartbeat branch runs.
        stop = threading.Event()
        beats = []
        calls = {"n": 0}

        def _sleep(_):
            calls["n"] += 1
            if calls["n"] >= 21:   # past the first _i%20==0 boundary
                stop.set()

        with mock.patch.object(self.bc, "send"), \
                mock.patch.object(self.bc, "_heartbeat",
                                  side_effect=lambda: beats.append(1)), \
                mock.patch.object(self.bc.time, "sleep", side_effect=_sleep):
            self.bc._thinking_loop(stop)
        # At least one heartbeat tick happened at the 20-iteration mark.
        self.assertGreaterEqual(len(beats), 1)


class WatchdogThreadErrorTests(_MonolithSec2Base):
    def setUp(self):
        self._saved_hb = self.bc._main_loop_heartbeat[0]
        self.bc._watchdog_reset_signal.clear()
        self.bc._watchdog_stop_event.clear()

    def tearDown(self):
        self.bc._main_loop_heartbeat[0] = self._saved_hb
        self.bc._watchdog_reset_signal.clear()
        self.bc._watchdog_stop_event.clear()

    def test_thread_survives_check_exception(self):
        # _main_loop_watchdog_check raises → the thread's `except Exception as
        # e` (4436-4437) prints, then wait() returns True (stop set) → exits.
        stop = self.bc._watchdog_stop_event
        stop.clear()

        def _check(*a, **k):
            raise RuntimeError("watchdog check blew up")

        # stop.wait returns True on the first call so the thread exits after
        # the single (raising) check.
        def _wait(_):
            stop.set()
            return True

        with mock.patch.object(self.bc, "_main_loop_watchdog_check",
                               side_effect=_check), \
                mock.patch.object(stop, "wait", side_effect=_wait):
            t = threading.Thread(target=self.bc._main_loop_watchdog_thread)
            t.start()
            t.join(timeout=5.0)
            self.assertFalse(t.is_alive())


# ───────────────────────────────────────────────────────────────────────────
#  Residual targeted branches (small, high-value single-line gaps)
# ───────────────────────────────────────────────────────────────────────────
class ResidualBranchTests(_MonolithSec2Base):
    # ── _devices_signature: tuple-comprehension raising (3642-3643) ──────────
    def test_devices_signature_inner_exception_returns_none(self):
        # query_devices returns objects WITHOUT a .get() method → the tuple
        # comprehension raises AttributeError → the inner `except: return None`.
        class _Dev:  # no .get
            name = "X"
        sd = mock.Mock()
        sd.query_devices.return_value = [_Dev(), _Dev()]
        with mock.patch.object(self.bc, "sd", sd):
            self.assertIsNone(self.bc._devices_signature())

    # ── _main_loop_watchdog_check: now=None default branch (4418) ────────────
    def test_watchdog_check_default_now_no_stall(self):
        # Fresh heartbeat + default now (None → time.time()) → no stall, but the
        # `if now is None: now = time.time()` line (4417-4418) executes.
        saved = self.bc._main_loop_heartbeat[0]
        self.bc._watchdog_reset_signal.clear()
        try:
            self.bc._main_loop_heartbeat[0] = time.time()
            fired = self.bc._main_loop_watchdog_check()   # no args
            self.assertFalse(fired)
        finally:
            self.bc._main_loop_heartbeat[0] = saved
            self.bc._watchdog_reset_signal.clear()


class TrayPublisherBambuEdgeTests(_MonolithSec2Base):
    def test_bambu_state_get_exception_is_caught(self):
        # bambu _state is a dict-like whose .get() raises inside the try → the
        # bambu block's `except: pass` (2865-2866) swallows it.
        stop = self.bc._tray_publisher_stop
        stop.clear()

        class _BadState(dict):
            def get(self, *a, **k):
                raise RuntimeError("state get boom")

        bm = mock.Mock()
        bm._state = _BadState({"gcode_state": "RUNNING"})
        bm._state_lock = None

        def _wait(_):
            stop.set()
            return True

        with mock.patch.object(self.bc, "_write_hud_state"), \
                mock.patch.dict(self.bc.sys.modules,
                                {"skill_bambu_monitor": bm}, clear=False), \
                mock.patch.object(stop, "wait", side_effect=_wait), \
                mock.patch.object(self.bc, "_hud_cal_last", [time.time()]):
            with self.bc._hud_state_lock:
                self.bc._hud_state_cache["alert_active"] = False
                self.bc._hud_state_cache["bambu_active"] = False
            try:
                self.bc._tray_state_publisher()
            finally:
                stop.clear()


class ProactiveAnnounceFdCleanupTests(_MonolithSec2Base):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jarvis_sec2_fd_")
        self._file_patch = mock.patch.object(
            self.bc, "__file__", os.path.join(self.tmp, "bobert_companion.py"))
        self._file_patch.start()

    def tearDown(self):
        self._file_patch.stop()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fdopen_failure_closes_descriptor(self):
        # os.fdopen raising (while fd >= 0, before fdopen takes ownership) →
        # the except branch's `if fd >= 0: os.close(fd)` (3767-3769) runs, then
        # the error propagates to the outer handler → returns False.
        with mock.patch.object(self.bc.os, "fdopen",
                               side_effect=OSError("fdopen blew up")), \
                mock.patch.object(self.bc.os, "close") as oclose:
            self.assertFalse(
                self.bc.proactive_announce("doomed", source="x"))
        oclose.assert_called()   # the dangling descriptor was closed

    def test_existing_queue_read_failure_treated_as_empty(self):
        # Queue file exists but open() raises (perm/lock) → the OUTER
        # `except Exception: data = []` (3740-3741) recovers, and the enqueue
        # still succeeds with a single fresh entry.
        queue = os.path.join(self.tmp, "pending_speech.json")
        with open(queue, "w", encoding="utf-8") as f:
            f.write("[]")
        real_open = open
        calls = {"n": 0}

        def _flaky_open(path, *a, **k):
            # Fail only the first read of the queue file; let temp writes work.
            if str(path) == queue and calls["n"] == 0:
                calls["n"] += 1
                raise OSError("queue locked")
            return real_open(path, *a, **k)

        with mock.patch.object(self.bc, "open", _flaky_open, create=True):
            self.assertTrue(self.bc.proactive_announce("recovered"))
        with real_open(queue, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["message"], "recovered")


class RefreshDevicesNonHeadsetSwitchTests(_MonolithSec2Base):
    def setUp(self):
        self._saved_cache = dict(self.bc._device_cache)

    def tearDown(self):
        self.bc._device_cache.clear()
        self.bc._device_cache.update(self._saved_cache)

    def test_non_headset_previous_mic_uses_disconnected_phrasing(self):
        # prev mic GONE from the device list and its name has NO
        # "headset"/"headphone" → the "appears to have disconnected" message.
        # (Loss is decided from the device LIST since 2026-08-20 — the mocked
        # enumeration below deliberately no longer contains "Blue Yeti".)
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = "Blue Yeti"
        self.bc._device_cache["last_in_index"] = 9
        self.bc._device_cache["last_out_name"] = None
        announced = []
        sd = mock.Mock()

        def _qd(idx=None, **kw):
            if idx is None:
                return [{"name": "Fallback Laptop Mic",
                         "max_input_channels": 1, "max_output_channels": 0},
                        {"name": "Speakers",
                         "max_input_channels": 0, "max_output_channels": 2}]
            return {"name": "ignored"}
        sd.query_devices.side_effect = _qd
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.dict(self.bc.sys.modules, {}, clear=False), \
                mock.patch.object(self.bc, "_record_speech_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active", [False]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=None), \
                mock.patch.object(self.bc, "_pick_device",
                                  side_effect=[(0, "Fallback Laptop Mic"),
                                               (1, "Speakers")]), \
                mock.patch.object(self.bc, "_enqueue_device_announcement",
                                  side_effect=announced.append):
            self.bc.sys.modules.pop("skill_wake_listener", None)
            self.bc._refresh_devices(force=True)
        self.assertEqual(len(announced), 1)
        self.assertIn("disconnected", announced[0].lower())
        self.assertNotIn("headset", announced[0].lower())


# ───────────────────────────────────────────────────────────────────────────
#  FOLLOW-THE-DEFAULT audio + spoken switch announcements (2026-08-20)
#
#  The owner selects his mic and speakers from a Stream Deck, which moves the
#  WINDOWS DEFAULT device. JARVIS must FOLLOW that default (never pin an index
#  of its own) and SAY which device it moved to, in both directions.
#
#  These tests pin the feature and cover the two defects it fixes:
#    D1 — the change guard compared the NAME ONLY and stored no index, so a
#         device moving 4 → 3 under an identical (MME-truncated) name was
#         invisible.
#    D2 — in the follow-the-default configuration _pick_device returns
#         (None, ""), and the `if in_name and …` guard therefore suppressed the
#         log AND the announcement outright; the output branch never announced
#         at all.
#  …and the outage that motivated it: PREFERRED_INPUT_DEVICES pinned "CORSAIR",
#  the powered-off headset still enumerated and still passed
#  check_input_settings, so it won every re-pick while the live Blue Snowball
#  sat at the Windows default — JARVIS was deaf for 90 minutes (2026-08-20).
#
#  Deterministic: no real device is opened or enumerated anywhere below.
# ───────────────────────────────────────────────────────────────────────────
class _FollowTheDefaultFixtures:
    """Stub fixtures shared by the follow-the-default test classes.

    A MIXIN, not a base test class, deliberately: subclassing one TestCase from
    another makes unittest re-collect and re-run every inherited test method
    under the child's name, which is the same duplicate-coverage trap the
    stale-duplicate rule warns about — the copies drift and one of them starts
    passing for a reason nobody checks."""

    CORSAIR = "Headset Microphone (CORSAIR VOID ELITE Wireless Gaming Dongle)"
    SNOWBALL = "Microphone (Blue Snowball)"
    SPEAKERS = "Speakers (Realtek(R) Audio)"
    HEADPHONES = "Headphones (CORSAIR VOID ELITE Wireless Gaming Dongle)"

    def setUp(self):
        self._saved_cache = dict(self.bc._device_cache)

    def tearDown(self):
        self.bc._device_cache.clear()
        self.bc._device_cache.update(self._saved_cache)

    # ── fixtures ──────────────────────────────────────────────────────────
    def _devices(self, names):
        """Device list where any name containing 'microphone'/'mic' is an input
        and everything else is an output."""
        out = []
        for n in names:
            is_in = "microphone" in n.lower() or "mic" in n.lower()
            out.append({"name": n,
                        "max_input_channels": 1 if is_in else 0,
                        "max_output_channels": 0 if is_in else 2,
                        "default_samplerate": 16000 if is_in else 48000})
        return out

    def _sd(self, names, default_in, default_out):
        """sounddevice stub: a readable device list plus a Windows default
        pair, which is the whole surface _default_device_identity touches.

        Every row carries hostapi 0 and sd.default.hostapi is 0, because
        _endpoint_device_identity (2026-08-21) restricts its endpoint→index
        match to the DEFAULT HOST API — the same restriction that makes the
        resolved index equal to what a re-enumeration would have produced."""
        devices = self._devices(names)
        sd = mock.Mock()

        def _query(idx=None, **kw):
            if idx is None:
                return list(devices)
            if isinstance(idx, int) and 0 <= idx < len(devices):
                return devices[idx]
            raise RuntimeError(f"Error querying device {idx}")
        sd.query_devices.side_effect = _query
        sd.check_input_settings.return_value = None
        sd.default = mock.Mock()
        sd.default.device = (default_in, default_out)
        sd.default.hostapi = 0
        return sd

    def _refresh(self, sd, *, prefs_in=(), prefs_out=(), announced=None,
                 endpoints=(None, None), endpoint_names=None, printed=None,
                 mic_live=False, tts_live=False, signature=None, force=True):
        """Drive one forced _refresh_devices pass against the stub.

        ``endpoints`` is the (render id, capture id) pair
        _win_default_endpoints() would report. It is PATCHED, never live: the
        real helper queries this machine's MMDevice API, which would make every
        assertion here depend on which speakers the owner happens to be using.
        The default (None, None) models a host with no MMDevice API at all —
        the conservative branch, where an index-only shift has no corroborating
        evidence.

        ``endpoint_names`` maps endpoint id → the Windows FRIENDLY NAME, i.e.
        what _win_endpoint_friendly_name() reads out of the endpoint's property
        store. It is patched for the same reason and defaults to EMPTY, which
        models "the id cannot be resolved to a name" and therefore keeps every
        pre-2026-08-21 test on the fallback path it was written against.

        ``mic_live`` / ``tts_live`` raise the corresponding PortAudio owner
        flag, which is how the deferral branches are reached. ``signature``, if
        given, is what _devices_signature() returns (a stable value means "the
        device list did not change", so no teardown is wanted); the default
        None reproduces the old fixture, where a null signature forces the
        reinit chain on every pass. ``printed`` collects the [audio] log lines.
        """
        sink = announced.append if announced is not None else (lambda _m: None)
        names = dict(endpoint_names or {})
        log = printed if printed is not None else []
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.object(self.bc, "_win_default_endpoints",
                                  return_value=tuple(endpoints)), \
                mock.patch.object(self.bc, "_win_endpoint_friendly_name",
                                  side_effect=lambda eid: names.get(eid)), \
                mock.patch.dict(self.bc.sys.modules, {}, clear=False), \
                mock.patch.object(self.bc, "_record_speech_active",
                                  [bool(mic_live)]), \
                mock.patch.object(self.bc, "_pathb_mic_active", [False]), \
                mock.patch.object(self.bc, "_ambient_stream_active", [False]), \
                mock.patch.object(self.bc, "_diag_capture_active", [False]), \
                mock.patch.object(self.bc, "_enroll_capture_active", [False]), \
                mock.patch.object(self.bc, "_tts_playback_active",
                                  [bool(tts_live)]), \
                mock.patch.object(self.bc, "MICROPHONE_INDEX", None), \
                mock.patch.object(self.bc, "SPEAKER_INDEX", None), \
                mock.patch.object(self.bc, "PREFERRED_INPUT_DEVICES",
                                  list(prefs_in)), \
                mock.patch.object(self.bc, "PREFERRED_OUTPUT_DEVICES",
                                  list(prefs_out)), \
                mock.patch.object(self.bc, "_devices_signature",
                                  return_value=signature), \
                mock.patch.object(self.bc, "_enqueue_device_announcement",
                                  side_effect=sink), \
                mock.patch("builtins.print",
                           side_effect=lambda *a, **k: log.append(
                               " ".join(str(x) for x in a))):
            self.bc.sys.modules.pop("skill_wake_listener", None)
            self.bc._device_cache["checked_at"] = 0.0
            self.bc._refresh_devices(force=force)
        return log

    def _boot(self, *, signature=None, reenum_at=0.0):
        """Neutral starting state: nothing tracked yet (boot).

        ``signature``/``reenum_at`` let a test start from a SETTLED session
        instead (enumeration already current, hotplug sweep freshly armed) —
        the state JARVIS is actually in when the owner presses his Stream
        Deck, and the only state in which "was a teardown avoided?" is a
        meaningful question."""
        self.bc._pa_defer_logged[0] = None
        self.bc._device_cache.update({
            "in": None, "out": None, "checked_at": 0.0,
            "last_in_name": None, "last_out_name": None,
            "last_in_index": None, "last_out_index": None,
            "last_in_endpoint": None, "last_out_endpoint": None,
            "last_default_endpoints": None,
            "last_devices_signature": signature, "last_reenum_at": reenum_at,
        })


class FollowTheDefaultDeviceTests(_FollowTheDefaultFixtures, _MonolithSec2Base):
    # ── _default_device_identity: total, and never pins anything ──────────
    def test_default_identity_reads_current_default(self):
        sd = self._sd([self.CORSAIR, self.SNOWBALL, self.SPEAKERS],
                      default_in=1, default_out=2)
        with mock.patch.object(self.bc, "sd", sd):
            self.assertEqual(self.bc._default_device_identity(want_input=True),
                             (1, self.SNOWBALL))
            self.assertEqual(self.bc._default_device_identity(want_input=False),
                             (2, self.SPEAKERS))

    def test_default_identity_handles_no_default(self):
        # PortAudio reports "no default endpoint" as -1.
        sd = self._sd([self.SNOWBALL], default_in=-1, default_out=-1)
        with mock.patch.object(self.bc, "sd", sd):
            self.assertEqual(self.bc._default_device_identity(want_input=True),
                             (None, ""))
            self.assertEqual(self.bc._default_device_identity(want_input=False),
                             (None, ""))

    def test_default_identity_swallows_errors(self):
        # sd.default.device raising, and a default index that can't be queried,
        # must BOTH degrade quietly — this runs inside the device loop.
        sd = mock.Mock()
        type(sd.default).device = mock.PropertyMock(
            side_effect=RuntimeError("portaudio down"))
        with mock.patch.object(self.bc, "sd", sd):
            self.assertEqual(self.bc._default_device_identity(want_input=True),
                             (None, ""))
        sd2 = self._sd([self.SNOWBALL], default_in=7, default_out=0)
        with mock.patch.object(self.bc, "sd", sd2):
            # index out of range → name unknown, but the index still tracks
            self.assertEqual(self.bc._default_device_identity(want_input=True),
                             (7, ""))

    # ── the outage: an empty list follows, a pinned list does not ─────────
    def test_empty_preferences_follow_the_default_without_mmdevice(self):
        # NO endpoint names available (no MMDevice API, or an id that resolves
        # to nothing): the index falls back to None so the open is left to
        # sounddevice's own default. This was the WHOLE contract until
        # 2026-08-21; it is now the FALLBACK, kept for hosts without pycaw.
        self._boot()
        sd = self._sd([self.CORSAIR, self.SNOWBALL, self.SPEAKERS],
                      default_in=1, default_out=2)
        self._refresh(sd)
        self.assertIsNone(self.bc._device_cache["in"])
        self.assertIsNone(self.bc._device_cache["out"])
        # …but the default's identity IS tracked, so a later switch is visible.
        self.assertEqual(self.bc._device_cache["last_in_index"], 1)
        self.assertEqual(self.bc._device_cache["last_in_name"], self.SNOWBALL)
        self.assertEqual(self.bc._device_cache["last_out_index"], 2)
        self.assertEqual(self.bc._device_cache["last_out_name"], self.SPEAKERS)

    def test_empty_preferences_resolve_the_live_default_to_an_index(self):
        # WITH the MMDevice API the live default endpoint is resolved to a row
        # in the CURRENT enumeration and that index is what gets opened.
        #
        # WHY THIS REPLACED "the cache must stay None" (2026-08-21). That rule
        # rested on "None means every open re-resolves the current default".
        # It does not: sounddevice maps device=None to
        # Pa_GetDefaultInput/OutputDevice, a struct field each host API fills
        # in ONCE inside Pa_Initialize — frozen until sd._terminate(). So None
        # followed the default JARVIS BOOTED ON. Proved live in
        # logs/session_2026-08-21_00-00-51.log: the owner moved his default to
        # the CORSAIR headset and JARVIS stayed on the Realtek for the whole
        # session while printing 34 deferral lines.
        self._boot()
        sd = self._sd([self.CORSAIR, self.SNOWBALL, self.SPEAKERS],
                      default_in=1, default_out=2)
        self._refresh(sd, endpoints=("{ep-spk}", "{ep-snow}"),
                      endpoint_names={"{ep-snow}": self.SNOWBALL,
                                      "{ep-spk}": self.SPEAKERS})
        self.assertEqual(self.bc._device_cache["in"], 1)
        self.assertEqual(self.bc._device_cache["out"], 2)

    def test_resolved_index_is_not_a_pin_it_tracks_the_live_default(self):
        # The index is a RESULT of the live endpoint read, recomputed every
        # pass — so moving the default moves the index, and moving it back
        # moves it back. That is the property "never pin" was protecting.
        names = [self.CORSAIR, self.SNOWBALL, self.SPEAKERS, self.HEADPHONES]
        eps = {"{ep-corsair}": self.CORSAIR, "{ep-snow}": self.SNOWBALL,
               "{ep-spk}": self.SPEAKERS, "{ep-phones}": self.HEADPHONES}
        self._boot()
        sd = self._sd(names, default_in=1, default_out=2)
        self._refresh(sd, endpoints=("{ep-spk}", "{ep-snow}"),
                      endpoint_names=eps)
        self.assertEqual((self.bc._device_cache["in"],
                          self.bc._device_cache["out"]), (1, 2))
        self._refresh(sd, endpoints=("{ep-phones}", "{ep-corsair}"),
                      endpoint_names=eps)
        self.assertEqual((self.bc._device_cache["in"],
                          self.bc._device_cache["out"]), (0, 3))
        self._refresh(sd, endpoints=("{ep-spk}", "{ep-snow}"),
                      endpoint_names=eps)
        self.assertEqual((self.bc._device_cache["in"],
                          self.bc._device_cache["out"]), (1, 2))

    def test_preferred_name_list_pins_the_dead_headset(self):
        # Regression for the 90-minute deafness: the powered-off CORSAIR still
        # enumerates and still passes check_input_settings, so a name list wins
        # over the live default. This is the behaviour the owner's config no
        # longer opts into (PREFERRED_INPUT_DEVICES = []).
        self._boot()
        sd = self._sd([self.CORSAIR, self.SNOWBALL, self.SPEAKERS],
                      default_in=1, default_out=2)
        self._refresh(sd, prefs_in=["CORSAIR", "Blue Snowball"])
        self.assertEqual(self.bc._device_cache["in"], 0,
                         "a name preference pins index 0 — the dead headset")
        # Same enumeration, no preferences: the live default wins instead.
        self._boot()
        self._refresh(sd)
        self.assertIsNone(self.bc._device_cache["in"])
        self.assertEqual(self.bc._device_cache["last_in_index"], 1)

    # ── D2: a default change is DETECTED and ANNOUNCED, both directions ───
    def test_boot_detection_is_silent(self):
        self._boot()
        announced = []
        sd = self._sd([self.CORSAIR, self.SNOWBALL, self.SPEAKERS],
                      default_in=1, default_out=2)
        self._refresh(sd, announced=announced)
        self.assertEqual(announced, [],
                         "None → first device is boot, not a switch")

    def test_input_default_change_is_announced_neutrally(self):
        self._boot()
        names = [self.CORSAIR, self.SNOWBALL, self.SPEAKERS]
        announced = []
        self._refresh(self._sd(names, 1, 2), announced=announced)   # boot
        self.assertEqual(announced, [])
        # Stream Deck moves the default mic to the headset.
        self._refresh(self._sd(names, 0, 2), announced=announced)
        self.assertEqual(len(announced), 1, announced)
        self.assertEqual(announced[0], "Switched to your headset, sir.")
        self.assertEqual(self.bc._device_cache["last_in_index"], 0)
        # …and with no resolvable endpoint NAME here, this direction stays on
        # the device=None fallback (see
        # test_empty_preferences_follow_the_default_without_mmdevice).
        self.assertIsNone(self.bc._device_cache["in"])

    def test_output_default_change_is_announced(self):
        # D2 (output half): the old code only PRINTED for the speaker branch.
        self._boot()
        names = [self.SNOWBALL, self.SPEAKERS, self.HEADPHONES]
        announced = []
        self._refresh(self._sd(names, 0, 1), announced=announced)   # boot
        self.assertEqual(announced, [])
        self._refresh(self._sd(names, 0, 2), announced=announced)
        self.assertEqual(len(announced), 1, announced)
        self.assertEqual(announced[0], "Switched to your headset, sir.")
        self.assertEqual(self.bc._device_cache["last_out_index"], 2)
        # Same: no endpoint name is supplied, so the fallback applies.
        self.assertIsNone(self.bc._device_cache["out"])

    def test_switch_to_snowball_names_the_device(self):
        self._boot()
        names = [self.CORSAIR, self.SNOWBALL, self.SPEAKERS]
        announced = []
        self._refresh(self._sd(names, 0, 2), announced=announced)   # boot
        self._refresh(self._sd(names, 1, 2), announced=announced)
        self.assertEqual(announced, ["Switched to the Blue Snowball, sir."])

    # ── D1: an index-only change under an identical name is detected ──────
    def test_index_only_change_same_name_is_detected(self):
        # Two endpoints can share an MME-truncated name; the old name-only
        # compare saw nothing at all when the default moved between them. The
        # pair-compare still DETECTS it (tracking follows the new index) —
        # what it may no longer do is SPEAK on the index alone. See
        # test_index_only_change_is_not_announced_without_endpoint_evidence.
        self._boot()
        dup = "Microphone (USB Audio Device)"
        announced = []
        self._refresh(self._sd([self.SPEAKERS, dup, dup], 2, 0),
                      announced=announced)          # boot, tracked at index 2
        self.assertEqual(self.bc._device_cache["last_in_index"], 2)
        self.assertEqual(self.bc._device_cache["last_in_name"], dup)
        self._refresh(self._sd([self.SPEAKERS, dup, dup], 1, 0),
                      announced=announced)          # same NAME, index 2 → 1
        self.assertEqual(self.bc._device_cache["last_in_index"], 1,
                         "index-only change must be detected (D1)")
        self.assertEqual(self.bc._device_cache["last_in_name"], dup,
                         "the name is identical — which is the whole point")

    def test_index_only_change_speaks_when_the_endpoint_really_moved(self):
        # D1 done honestly: the same MME-truncated name at a new index IS a
        # real switch when Windows says the default endpoint moved. The
        # endpoint id neither renumbers nor truncates, so it is the evidence
        # the index alone cannot supply.
        self._boot()
        dup = "Microphone (USB Audio Device)"
        announced = []
        self._refresh(self._sd([self.SPEAKERS, dup, dup], 2, 0),
                      announced=announced, endpoints=("{render}", "{mic-A}"))
        self._refresh(self._sd([self.SPEAKERS, dup, dup], 1, 0),
                      announced=announced, endpoints=("{render}", "{mic-B}"))
        self.assertEqual(self.bc._device_cache["last_in_index"], 1)
        self.assertEqual(len(announced), 1, announced)
        self.assertEqual(announced[0], "Switched to USB Audio Device, sir.")

    def test_index_only_change_is_not_announced_without_endpoint_evidence(self):
        # THE REGRESSION (2026-08-20 review). Inputs enumerate before outputs,
        # so ANY input appearing or disappearing slides every output index by
        # one while the endpoint itself never changes — and the (index, name)
        # announce trigger spoke "Switched to Realtek(R) Audio, sir." about the
        # speakers JARVIS was already using. With no MMDevice evidence the
        # shift is tracked and logged, never spoken.
        self._boot()
        announced = []
        self._refresh(self._sd([self.SNOWBALL, self.CORSAIR, self.SPEAKERS],
                               0, 2), announced=announced)
        self.assertEqual(self.bc._device_cache["last_out_index"], 2)
        # The CORSAIR mic goes away: the speakers slide 2 → 1, same name.
        self._refresh(self._sd([self.SNOWBALL, self.SPEAKERS], 0, 1),
                      announced=announced)
        self.assertEqual(self.bc._device_cache["last_out_index"], 1,
                         "the re-key must still happen — tracking follows")
        self.assertEqual(self.bc._device_cache["last_out_name"], self.SPEAKERS)
        self.assertEqual(announced, [],
                         "an index-only shift with no endpoint evidence must "
                         "NOT claim a switch that did not happen")

    def test_index_only_shift_is_silent_when_the_endpoint_is_unchanged(self):
        # Negative control for the evidence path: MMDevice IS available and
        # reports the SAME endpoint id across the renumbering. Evidence that
        # says "unchanged" must not be read as evidence of a change.
        self._boot()
        announced = []
        self._refresh(self._sd([self.SNOWBALL, self.CORSAIR, self.SPEAKERS],
                               0, 2), announced=announced,
                      endpoints=("{spk-1}", "{mic-1}"))
        self._refresh(self._sd([self.SNOWBALL, self.SPEAKERS], 0, 1),
                      announced=announced, endpoints=("{spk-1}", "{mic-1}"))
        self.assertEqual(self.bc._device_cache["last_out_index"], 1)
        self.assertEqual(announced, [])

    def test_name_change_is_announced_with_no_endpoint_api_at_all(self):
        # Negative control for the guard as a whole: removing the announce on
        # index-only shifts must not remove the announce on a REAL switch. No
        # MMDevice API here — the name is enough on its own.
        self._boot()
        names = [self.CORSAIR, self.SNOWBALL, self.SPEAKERS]
        announced = []
        self._refresh(self._sd(names, 0, 2), announced=announced)
        self._refresh(self._sd(names, 1, 2), announced=announced)
        self.assertEqual(announced, ["Switched to the Blue Snowball, sir."])

    # ── wording: neutral for a switch, loss only for a real disappearance ─
    def test_still_present_previous_device_gets_neutral_wording(self):
        self._boot()
        names = [self.CORSAIR, self.SNOWBALL, self.SPEAKERS]
        announced = []
        self._refresh(self._sd(names, 0, 2), announced=announced)   # boot
        self._refresh(self._sd(names, 1, 2), announced=announced)
        self.assertEqual(len(announced), 1)
        self.assertNotIn("dropped off", announced[0].lower())
        self.assertNotIn("disconnected", announced[0].lower())

    def test_vanished_previous_device_keeps_loss_wording(self):
        # The headset sits LAST so unplugging it leaves the speaker index
        # alone — this test is about the input announcement only.
        self._boot()
        announced = []
        # Headset is the default mic…
        self._refresh(self._sd([self.SNOWBALL, self.SPEAKERS, self.CORSAIR],
                               2, 1), announced=announced)
        # …then it powers off: gone from the enumeration entirely, and Windows
        # falls back to the Snowball.
        self._refresh(self._sd([self.SNOWBALL, self.SPEAKERS], 0, 1),
                      announced=announced)
        self.assertEqual(len(announced), 1, announced)
        self.assertIn("dropped off", announced[0].lower())
        self.assertIn("the Blue Snowball", announced[0])

    def test_vanished_non_headset_device_uses_disconnected_wording(self):
        self._boot()
        announced = []
        self._refresh(self._sd([self.SNOWBALL, self.SPEAKERS], 0, 1),
                      announced=announced)
        # The Snowball is unplugged; the headset takes over as default mic.
        self._refresh(self._sd([self.CORSAIR, self.SPEAKERS], 0, 1),
                      announced=announced)
        self.assertEqual(len(announced), 1, announced)
        self.assertIn("disconnected", announced[0].lower())
        self.assertIn("the Blue Snowball", announced[0])

    def test_unreadable_device_list_is_not_treated_as_a_loss(self):
        # _device_name_present returns None ("unknown") — that must fall to the
        # neutral line, never to a fabricated disconnect report.
        sd = mock.Mock()
        sd.query_devices.side_effect = RuntimeError("portaudio down")
        announced = []
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.object(self.bc, "_enqueue_device_announcement",
                                  side_effect=announced.append):
            self.assertIsNone(self.bc._device_name_present("Gaming Headset"))
            self.bc._announce_device_change("Gaming Headset", self.SNOWBALL,
                                            want_input=True)
        self.assertEqual(announced, ["Switched to the Blue Snowball, sir."])

    def test_one_press_moving_both_defaults_enqueues_identical_text(self):
        # A single Stream Deck press moves the mic AND the speakers to the
        # headset. Both branches announce (by design), and because the text is
        # IDENTICAL _speak_pending's dedupe collapses it to one spoken line.
        self._boot()
        names = [self.SNOWBALL, self.SPEAKERS, self.CORSAIR, self.HEADPHONES]
        announced = []
        self._refresh(self._sd(names, 0, 1), announced=announced)   # boot
        self._refresh(self._sd(names, 2, 3), announced=announced)
        self.assertEqual(announced,
                         ["Switched to your headset, sir."] * 2)
        self.assertEqual(len(set(announced)), 1,
                         "identical text is what the speech dedupe collapses")

    def test_unchanged_default_never_repeats_the_announcement(self):
        # Anti-spam: the refresh runs every DEVICE_CHECK_INTERVAL (4 s); a
        # stable default must stay silent forever.
        self._boot()
        names = [self.CORSAIR, self.SNOWBALL, self.SPEAKERS]
        announced = []
        self._refresh(self._sd(names, 0, 2), announced=announced)   # boot
        self._refresh(self._sd(names, 1, 2), announced=announced)   # switch
        for _ in range(3):
            self._refresh(self._sd(names, 1, 2), announced=announced)
        self.assertEqual(len(announced), 1, announced)


# ───────────────────────────────────────────────────────────────────────────
#  FOLLOW-THE-DEFAULT, SECOND PASS — the two defects the FIRST LIVE BOOT of
#  the 2026-08-20 work exposed (logs/session_2026-08-21_00-00-51.log).
#
#  DEFECT 1 — THE FEATURE NEVER RAN. The trigger suppressed itself whenever a
#    PortAudio stream owner was live (`and not _owners_busy`), on the theory
#    that firing it into a guaranteed deferral was pure churn. But
#    record_speech re-opens the mic continuously, so an owner is live
#    essentially always on an awake session: in 15 minutes the trigger never
#    asserted once, the endpoint baseline was never rebased (it rebased only
#    after a successful re-enumeration), and a Stream Deck press after boot was
#    never followed. The root error was equating "follow the default" with
#    "re-enumerate": a press moves the default between endpoints PortAudio
#    ALREADY has rows for, so the new device's INDEX — not a teardown — is what
#    is actually needed.
#
#  DEFECT 2 — THE DEFERRAL LOGGED PER CYCLE. 34 "device drift detected …"
#    lines in 15 minutes, none of which described a real deferral: the owner
#    checks were evaluated BEFORE the "is a teardown even wanted?" check, so
#    every cache-cleared re-pick pass that happened to coincide with a live mic
#    printed one. A repeating condition is a STATE and must log its
#    transitions, not its cycles.
#
#  Deterministic: no real device, no real MMDevice API, no real PortAudio.
# ───────────────────────────────────────────────────────────────────────────
class FollowTheDefaultWhileStreamsAreLiveTests(_FollowTheDefaultFixtures,
                                               _MonolithSec2Base):
    EPS = {"{ep-snow}": _FollowTheDefaultFixtures.SNOWBALL,
           "{ep-spk}": _FollowTheDefaultFixtures.SPEAKERS,
           "{ep-corsair}": _FollowTheDefaultFixtures.CORSAIR,
           "{ep-phones}": _FollowTheDefaultFixtures.HEADPHONES}
    NAMES = [_FollowTheDefaultFixtures.CORSAIR,
             _FollowTheDefaultFixtures.SNOWBALL,
             _FollowTheDefaultFixtures.SPEAKERS,
             _FollowTheDefaultFixtures.HEADPHONES]
    SIG = ("stable-device-list",)

    def _settled(self, sd, eps=("{ep-spk}", "{ep-snow}"), **kw):
        """Settle into a session already tracking `eps`, with the enumeration
        current and the 300 s hotplug sweep freshly armed — so any teardown
        that happens afterwards can only have been caused by the default move
        under test."""
        self._boot(signature=self.SIG, reenum_at=time.time())
        self._refresh(sd, endpoints=eps, endpoint_names=self.EPS,
                      signature=self.SIG, force=False, **kw)

    # ── DEFECT 1 ─────────────────────────────────────────────────────────
    def test_default_move_is_adopted_while_the_mic_is_live(self):
        sd = self._sd(self.NAMES, default_in=1, default_out=2)
        self._settled(sd)
        self.assertEqual((self.bc._device_cache["in"],
                          self.bc._device_cache["out"]), (1, 2))
        sd._terminate.reset_mock()
        sd._initialize.reset_mock()
        announced = []
        # THE PRESS, with record_speech holding the mic — the exact condition
        # that made the shipped version do nothing at all.
        self._refresh(sd, endpoints=("{ep-phones}", "{ep-corsair}"),
                      endpoint_names=self.EPS, signature=self.SIG,
                      force=False, mic_live=True, announced=announced)
        self.assertEqual(self.bc._device_cache["in"], 0,
                         "the new default MIC must be adopted")
        self.assertEqual(self.bc._device_cache["out"], 3,
                         "the new default SPEAKERS must be adopted")
        sd._terminate.assert_not_called()
        sd._initialize.assert_not_called()
        self.assertEqual(announced, ["Switched to your headset, sir."] * 2)

    def test_adopted_move_rebases_the_baseline_so_it_cannot_re_fire(self):
        # The shipped version rebased ONLY after a successful re-enumeration,
        # which is why an un-runnable teardown turned into an endless retry.
        sd = self._sd(self.NAMES, default_in=1, default_out=2)
        self._settled(sd)
        moved = ("{ep-phones}", "{ep-corsair}")
        log = []
        for _ in range(12):
            self._refresh(sd, endpoints=moved, endpoint_names=self.EPS,
                          signature=self.SIG, force=False, mic_live=True,
                          printed=log)
        self.assertEqual(self.bc._device_cache["last_default_endpoints"],
                         moved)
        hits = [ln for ln in log if "default audio endpoint moved" in ln]
        self.assertEqual(len(hits), 1,
                         f"one move, one line — got {len(hits)}: {hits}")

    def test_move_to_an_unenumerated_device_still_escalates_to_a_reinit(self):
        # The teardown is still the right answer for TRUE hotplug: a default
        # that points at a device PortAudio has no row for cannot be opened
        # from the frozen list, so the destructive path must still be taken.
        sd = self._sd(self.NAMES, default_in=1, default_out=2)
        self._settled(sd)
        sd._terminate.reset_mock()
        log = []
        self._refresh(sd, endpoints=("{ep-spk}", "{ep-ghost}"),
                      endpoint_names=dict(self.EPS,
                                          **{"{ep-ghost}": "Microphone (Just Plugged In)"}),
                      signature=self.SIG, force=False, printed=log)
        sd._terminate.assert_called_once()
        self.assertTrue(any("not in PortAudio's current enumeration" in ln
                            for ln in log), log)

    def test_ambiguous_truncated_name_refuses_to_guess(self):
        # Two endpoints sharing an MME-truncated 31-character name is a real
        # configuration here (defect D1). Opening the WRONG mic is strictly
        # worse than staying on the stale-but-known one, so ambiguity resolves
        # to "unresolved" and the fallback applies.
        dup31 = "Microphone (USB Audio Device XY"      # exactly 31 chars
        self.assertEqual(len(dup31), 31)
        full = "Microphone (USB Audio Device XYZ Rev B)"
        sd = self._sd([dup31, dup31, self.SPEAKERS], default_in=0, default_out=2)
        self._boot()
        self._refresh(sd, endpoints=("{ep-spk}", "{ep-dup}"),
                      endpoint_names={"{ep-spk}": self.SPEAKERS,
                                      "{ep-dup}": full},
                      signature=self.SIG, force=False)
        self.assertIsNone(self.bc._device_cache["in"],
                          "an ambiguous match must NOT pick a device")
        self.assertEqual(self.bc._device_cache["out"], 2,
                         "the unambiguous direction still resolves")

    def test_truncated_name_matches_its_untruncated_endpoint(self):
        # The complement: ONE 31-character row that prefixes the friendly name
        # is the MME truncation of it, and must match.
        trunc = "Headset Earphone (CORSAIR VOID "     # exactly 31 chars
        self.assertEqual(len(trunc), 31)
        full = "Headset Earphone (CORSAIR VOID ELITE Wireless Gaming Headset)"
        sd = self._sd([self.SNOWBALL, trunc], default_in=0, default_out=1)
        self._boot()
        self._refresh(sd, endpoints=("{ep-void}", "{ep-snow}"),
                      endpoint_names={"{ep-void}": full,
                                      "{ep-snow}": self.SNOWBALL},
                      signature=self.SIG, force=False)
        self.assertEqual(self.bc._device_cache["out"], 1)

    def test_resolution_is_restricted_to_the_default_host_api(self):
        # Every endpoint enumerates once per host API (MME / DirectSound /
        # WASAPI / WDM-KS). Only the DEFAULT host API's row is the one
        # device=None would have resolved to after a reinit, so only that row
        # may be picked — otherwise the duplicates read as ambiguity and the
        # feature would never resolve anything on a real machine.
        sd = self._sd([self.SNOWBALL, self.SPEAKERS, self.SPEAKERS],
                      default_in=0, default_out=1)
        devices = sd.query_devices(None)
        devices[2]["hostapi"] = 2           # the WASAPI twin of index 1
        sd.query_devices.side_effect = lambda idx=None, **kw: (
            list(devices) if idx is None else devices[idx])
        self._boot()
        self._refresh(sd, endpoints=("{ep-spk}", "{ep-snow}"),
                      endpoint_names={"{ep-spk}": self.SPEAKERS,
                                      "{ep-snow}": self.SNOWBALL},
                      signature=self.SIG, force=False)
        self.assertEqual(self.bc._device_cache["out"], 1,
                         "the default-host-API row wins; the twin is ignored")

    # ── DEFECT 2 ─────────────────────────────────────────────────────────
    def test_a_persistent_deferral_logs_once_not_every_cycle(self):
        sd = self._sd(self.NAMES, default_in=1, default_out=2)
        self._settled(sd)
        log = []
        # A teardown that IS wanted (the 300 s hotplug sweep is due) and CANNOT
        # run (the mic is live) — held for 40 consecutive passes, which is
        # ~5 minutes of the live cadence.
        self.bc._device_cache["last_reenum_at"] = 0.0
        for _ in range(40):
            self._refresh(sd, endpoints=("{ep-spk}", "{ep-snow}"),
                          endpoint_names=self.EPS, signature=self.SIG,
                          force=False, mic_live=True, printed=log)
        deny = [ln for ln in log if "mid-capture" in ln]
        self.assertEqual(len(deny), 1,
                         f"one state, one line — got {len(deny)}")
        sd._terminate.assert_not_called()
        # The owner releases the mic: exactly one closing line, and the
        # teardown that was being held off finally runs.
        log2 = []
        self._refresh(sd, endpoints=("{ep-spk}", "{ep-snow}"),
                      endpoint_names=self.EPS, signature=self.SIG,
                      force=False, printed=log2)
        self.assertEqual(len([ln for ln in log2 if "no longer deferred" in ln]),
                         1, log2)
        sd._terminate.assert_called_once()

    def test_a_repick_pass_with_a_live_mic_prints_no_deny_line(self):
        # THE 34 LINES. None of them was a real deferral: the pass wanted no
        # teardown at all (unchanged signature, cache-cleared re-pick), it just
        # happened to coincide with a live mic. Asking "is a teardown wanted?"
        # first is what removes them.
        sd = self._sd(self.NAMES, default_in=1, default_out=2)
        self._settled(sd)
        log = []
        for _ in range(25):
            # An invalidating call site cleared the cache — a re-pick REQUEST,
            # not drift. record_speech is live throughout, as it is live.
            self.bc._device_cache["in"] = None
            self._refresh(sd, endpoints=("{ep-spk}", "{ep-snow}"),
                          endpoint_names=self.EPS, signature=self.SIG,
                          force=False, mic_live=True, printed=log)
        self.assertEqual([ln for ln in log if "device drift detected" in ln],
                         [], "no drift happened, so no drift may be reported")
        self.assertEqual(self.bc._device_cache["in"], 1,
                         "…and the re-pick it asked for still happened")
        sd._terminate.assert_not_called()

    def test_deferral_reason_change_is_a_transition_and_does_log(self):
        # mic → TTS is different information about why JARVIS is not following
        # the device yet, so it prints; a repeat of the same reason does not.
        sd = self._sd(self.NAMES, default_in=1, default_out=2)
        self._settled(sd)
        self.bc._device_cache["last_reenum_at"] = 0.0
        log = []
        kw = dict(endpoints=("{ep-spk}", "{ep-snow}"), endpoint_names=self.EPS,
                  signature=self.SIG, force=False, printed=log)
        self._refresh(sd, mic_live=True, **kw)
        self._refresh(sd, mic_live=True, **kw)
        self._refresh(sd, tts_live=True, **kw)
        self._refresh(sd, tts_live=True, **kw)
        self.assertEqual(len([ln for ln in log if "mid-capture" in ln]), 1, log)
        self.assertEqual(len([ln for ln in log if "TTS playback is" in ln]), 1,
                         log)

    def test_log_reinit_deferral_is_total_and_transition_only(self):
        # The helper itself: same reason is silent, a change prints, clearing
        # prints once, clearing twice is silent, and nothing it does can raise
        # into the device loop.
        self.bc._pa_defer_logged[0] = None
        with mock.patch("builtins.print") as p:
            self.bc._log_reinit_deferral("mic", "  [audio] first")
            self.bc._log_reinit_deferral("mic", "  [audio] first")
            self.bc._log_reinit_deferral("mic", "  [audio] first")
        self.assertEqual(p.call_count, 1)
        with mock.patch("builtins.print") as p:
            self.bc._log_reinit_deferral(None)
            self.bc._log_reinit_deferral(None)
        self.assertEqual(p.call_count, 1)
        self.assertIn("no longer deferred", p.call_args[0][0])
        with mock.patch("builtins.print", side_effect=RuntimeError("boom")):
            self.bc._log_reinit_deferral("tts", "  [audio] x")   # must not raise

    # ── the helper, in isolation ─────────────────────────────────────────
    def test_endpoint_device_identity_is_total(self):
        sd = self._sd([self.SNOWBALL, self.SPEAKERS], 0, 1)
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.object(self.bc, "_win_endpoint_friendly_name",
                                  return_value=None):
            self.assertEqual(
                self.bc._endpoint_device_identity("{ep}", want_input=True),
                (None, ""), "an unreadable friendly name is not a crash")
            self.assertEqual(
                self.bc._endpoint_device_identity(None, want_input=True),
                (None, ""))
        # sd.query_devices exploding, and a non-integer host api, both degrade.
        broken = mock.Mock()
        broken.query_devices.side_effect = RuntimeError("portaudio down")
        broken.default.hostapi = 0
        with mock.patch.object(self.bc, "sd", broken), \
                mock.patch.object(self.bc, "_win_endpoint_friendly_name",
                                  return_value=self.SNOWBALL):
            self.assertEqual(
                self.bc._endpoint_device_identity("{ep}", want_input=True),
                (None, ""))
        nohost = self._sd([self.SNOWBALL], 0, 0)
        nohost.default.hostapi = None
        with mock.patch.object(self.bc, "sd", nohost), \
                mock.patch.object(self.bc, "_win_endpoint_friendly_name",
                                  return_value=self.SNOWBALL):
            self.assertEqual(
                self.bc._endpoint_device_identity("{ep}", want_input=True),
                (None, ""))

    def test_unresolvable_name_is_forgotten_so_a_rename_self_heals(self):
        # A cached friendly name that no longer matches anything must be
        # dropped, or a renamed device would need a restart — the stale-
        # duplicate rule applied to a cache.
        self.bc._win_endpoint_names.clear()
        self.bc._win_endpoint_names["{ep}"] = "Microphone (Old Name)"
        sd = self._sd([self.SNOWBALL, self.SPEAKERS], 0, 1)
        with mock.patch.object(self.bc, "sd", sd), \
                mock.patch.object(self.bc, "_win_endpoint_enumerator",
                                  return_value=None):
            self.assertEqual(
                self.bc._endpoint_device_identity("{ep}", want_input=True),
                (None, ""))
        self.assertNotIn("{ep}", self.bc._win_endpoint_names)


class FriendlyDeviceNameSpokenAliasTests(_MonolithSec2Base):
    """The announcement reads _friendly_device_name OUT LOUD, so the owner's
    three real endpoints must come out speakable."""

    def test_corsair_headset_alias(self):
        self.assertEqual(
            self.bc._friendly_device_name(
                "Headset Microphone (CORSAIR VOID ELITE Wireless Gaming Dongle)"),
            "your headset")

    def test_corsair_output_alias(self):
        self.assertEqual(
            self.bc._friendly_device_name(
                "Headphones (CORSAIR VOID ELITE Wireless Gaming Dongle), MME"),
            "your headset")

    def test_snowball_alias(self):
        self.assertEqual(
            self.bc._friendly_device_name("Microphone (Blue Snowball), MME"),
            "the Blue Snowball")

    def test_generic_speakers_possessed(self):
        self.assertEqual(self.bc._friendly_device_name("Speakers, MME"),
                         "your speakers")

    def test_mme_duplicate_index_prefix_stripped(self):
        self.assertEqual(
            self.bc._friendly_device_name("Speakers (2- USB Audio Device)"),
            "USB Audio Device")

    def test_branded_speakers_unchanged(self):
        # The generic parse still owns everything outside the alias table.
        self.assertEqual(self.bc._friendly_device_name("Speakers (Realtek)"),
                         "Realtek")

    def test_nested_trademark_marker_does_not_truncate_the_brand(self):
        # The real default-speaker name on this box. A non-greedy parenthetical
        # match stopped at the nested ')' and spoke "Realtek(R".
        self.assertEqual(
            self.bc._friendly_device_name("Speakers (Realtek(R) Audio), MME"),
            "Realtek Audio")

    def test_sibling_parentheticals_fall_back_to_the_first_group(self):
        # Greedy would spill across both groups ("2- USB) (Realtek"); the
        # balance check rejects that and takes the first group, whose MME
        # duplicate-index prefix is then stripped.
        self.assertEqual(
            self.bc._friendly_device_name("Microphone (2- USB) (Realtek)"),
            "USB")

    def test_mme_truncated_description_keeps_the_brand(self):
        # MME cuts descriptions at 31 chars — closing paren and all. Real names
        # off this box: 'Speakers (DualSense Wireless Co', 'Headset Earphone
        # (CORSAIR VOID '.
        self.assertEqual(
            self.bc._friendly_device_name("Speakers (DualSense Wireless Co"),
            "DualSense Wireless Co")
        self.assertEqual(
            self.bc._friendly_device_name("Headset Earphone (CORSAIR VOID "),
            "your headset")

    def test_empty_parenthetical_falls_back_to_the_endpoint_word(self):
        # Real name off this box: 'Microphone ()' — must not speak "( )".
        self.assertEqual(self.bc._friendly_device_name("Microphone ()"),
                         "your microphone")

    def test_parens_balanced_helper(self):
        self.assertTrue(self.bc._parens_balanced("Realtek(R) Audio"))
        self.assertFalse(self.bc._parens_balanced("2- USB) (Realtek"))
        self.assertFalse(self.bc._parens_balanced("open ("))


if __name__ == "__main__":
    unittest.main()
