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

import json
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
        # force=False, signature identical to last → bump checked_at, no
        # _terminate/_initialize.
        self.bc._device_cache["checked_at"] = 0.0
        sig = ((0, "Mic", 1, 0),)
        self.bc._device_cache["last_devices_signature"] = sig
        with mock.patch.object(self.bc, "_devices_signature", return_value=sig):
            sd = mock.Mock()
            with mock.patch.object(self.bc, "sd", sd):
                self.bc._refresh_devices(force=False)
            sd._terminate.assert_not_called()
        self.assertGreater(self.bc._device_cache["checked_at"], 0.0)

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
        # call _enqueue_device_announcement.
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
        # Leave the pause event clear (its boot default).
        self.bc._face_track_pause.clear()

    def test_pause_sets_event(self):
        self.bc._face_track_pause.clear()
        self.bc.pause_face_tracking()
        self.assertTrue(self.bc._face_track_pause.is_set())

    def test_resume_clears_event(self):
        self.bc._face_track_pause.set()
        self.bc.resume_face_tracking()
        self.assertFalse(self.bc._face_track_pause.is_set())


# ───────────────────────────────────────────────────────────────────────────
#  _face_tracking_thread  (no cameras → fast clean exit)
# ───────────────────────────────────────────────────────────────────────────
class FaceTrackingThreadTests(_MonolithSec2Base):
    def test_no_cameras_returns_immediately(self):
        # _open_capture returns None for every configured cam → caps empty →
        # the thread prints "No cameras available" and returns without looping.
        cv2 = mock.Mock()
        bad_cap = mock.Mock()
        bad_cap.isOpened.return_value = False
        cv2.VideoCapture.return_value = bad_cap
        cv2.CAP_DSHOW = 700
        with mock.patch.object(self.bc, "cv2", cv2), \
                mock.patch.object(self.bc, "CAMERAS",
                                  [{"index": 0, "label": "X", "primary": True,
                                    "look_x": 0.5, "look_y": 0.5}]):
            # Should return quickly with no surviving thread/loop.
            self.bc._face_tracking_thread()

    def test_one_good_frame_iteration_then_stop(self):
        # Drive exactly one healthy loop iteration: the camera opens, yields a
        # good frame, a face is detected on the primary cam → the eye-control
        # math + send() path runs, then _face_track_stop ends the loop. Covers
        # the frame-cache / detection / tracking-math body (not just the empty
        # early-return). No real device, no real thread — runs inline.
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 1280

        stop = self.bc._face_track_stop
        pause = self.bc._face_track_pause
        stop.clear()
        pause.clear()

        # First read() yields a good frame; immediately arm the stop event so
        # the while-loop condition is False at the top of the next iteration.
        def _read():
            stop.set()
            return True, frame

        cap.read.side_effect = _read
        cv2 = mock.Mock()
        cv2.VideoCapture.return_value = cap
        cv2.CAP_DSHOW = 700

        sends = []
        try:
            with mock.patch.object(self.bc, "cv2", cv2), \
                    mock.patch.object(self.bc, "CAMERAS",
                                      [{"index": 0, "label": "X",
                                        "primary": True,
                                        "look_x": 0.5, "look_y": 0.5}]), \
                    mock.patch.object(self.bc, "_detect_face",
                                      return_value=(0.5, 0.5)), \
                    mock.patch.object(self.bc, "_note_camera_read_attempt"), \
                    mock.patch.object(self.bc, "send",
                                      side_effect=lambda **k: sends.append(k)), \
                    mock.patch.object(self.bc.time, "sleep"):
                self.bc._face_tracking_thread()
        finally:
            stop.clear()
            pause.clear()
        # The good frame was cached for see_user.
        with self.bc._camera_state_lock:
            self.assertIn(0, self.bc._camera_latest_frame)

    def test_detect_face_cv2_error_degrades_gracefully(self):
        # A cv2.error out of _detect_face (e.g. an OpenCL/GPU hiccup) must NOT
        # unwind the tracking thread. The loop should swallow it, cache the
        # frame for see_user, back off with time.sleep(0.5), and continue —
        # then exit cleanly when _face_track_stop fires.
        import logging
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 1280

        stop = self.bc._face_track_stop
        pause = self.bc._face_track_pause
        stop.clear()
        pause.clear()

        def _read():
            stop.set()          # end the loop after this single iteration
            return True, frame
        cap.read.side_effect = _read

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
        try:
            with mock.patch.object(self.bc, "cv2", cv2), \
                    mock.patch.object(self.bc, "CAMERAS",
                                      [{"index": 0, "label": "X",
                                        "primary": True,
                                        "look_x": 0.5, "look_y": 0.5}]), \
                    mock.patch.object(self.bc, "_detect_face",
                                      side_effect=_boom), \
                    mock.patch.object(self.bc, "_note_camera_read_attempt"), \
                    mock.patch.object(self.bc, "send"), \
                    mock.patch.object(self.bc.time, "sleep",
                                      side_effect=lambda s: sleeps.append(s)):
                # Must return normally — the cv2.error is swallowed, not raised.
                # Silence the expected logging.exception traceback so the test
                # log stays clean (the handler under test logs the swallow).
                logging.disable(logging.CRITICAL)
                try:
                    self.bc._face_tracking_thread()
                finally:
                    logging.disable(logging.NOTSET)
        finally:
            stop.clear()
            pause.clear()
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
        mem = {}
        with mock.patch.object(self.bc, "save_memory") as save:
            self.bc._set_late_night_suppression(mem)
        self.assertIn("late_night_no_comments_until", mem)
        save.assert_called_once_with(mem)

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
        fake_diag.pause_diagnostics.assert_called_once()
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
                dd.pause_diagnostics.side_effect = RuntimeError("pause fail")
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
        # prev mic name contains "headset" → the headset-specific switch
        # message branch (3956-3958) is used.
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = "Gaming Headset"
        self.bc._device_cache["last_out_name"] = None
        announced = []
        sd = mock.Mock()
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
        # prev mic name has NO "headset"/"headphone" → the else branch
        # (3960-3962) builds the "appears to have disconnected" message.
        self.bc._device_cache["checked_at"] = 0.0
        self.bc._device_cache["last_in_name"] = "Blue Yeti"
        self.bc._device_cache["last_out_name"] = None
        announced = []
        sd = mock.Mock()
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
                                  side_effect=[(0, "Fallback Laptop Mic"),
                                               (1, "Speakers")]), \
                mock.patch.object(self.bc, "_enqueue_device_announcement",
                                  side_effect=announced.append):
            self.bc.sys.modules.pop("skill_wake_listener", None)
            self.bc._refresh_devices(force=True)
        self.assertEqual(len(announced), 1)
        self.assertIn("disconnected", announced[0].lower())
        self.assertNotIn("headset", announced[0].lower())


if __name__ == "__main__":
    unittest.main()
