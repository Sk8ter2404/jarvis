"""The silent-mic warning, driven through the REAL record_speech (2026-11-17,
overhaul findings speech-mic-silent-rearm + proactive-04).

THE LIVE LOGS. 33 "[vad] WARNING: raw mic RMS has been ~0" lines across six
sessions, every one on a working desk mic, every one followed seconds later
by "mic is audible again ... re-armed" at RMS 0.00005 - 0.00054: a quiet room
sits right at core.audio_processor._AUDIBLE_RMS_FLOOR (5e-5), and the recovery
arm cleared the latch AND the 900 s re-warn throttle on ONE such chunk. 11-17:
warnings at 14:49:10, 14:50:52, 14:54:50, 14:57:14, 14:58:41; five copies
folded into a recap at 16:02:41, and at 16:02:53 "silent for 40 seconds" was
spoken 12 s into listening, right after a 28 s turn and a 12 s recap.

What is faked: the PortAudio stream (a per-capture stand-in that feeds its
frames into record_speech's REAL callback), the device resolution and the UI
edges (tests/monolith/test_monolith_listen_media._Base). The silence clock is
core.audio_processor's real one, seeded by hand. Device names are SYNTHETIC.

Mutation-checked by hand (each load-bearing line restored to the old code ->
the named test goes red): the recovery arm's `_silent_mic_warned_at[0] = 0.0`
-> test_noise_floor_does_not_re_arm_or_reset_the_throttle; the audio_processor
gap booking (`unpolled_s += t - prev`) -> test_time_nobody_was_listening_is_
not_silence; the throttle check's dependence on the latch ->
test_a_sustained_recovery_clears_the_latch_but_not_the_throttle.

    python tools/run_tests.py  (selector monolith.test_monolith_silent_mic)
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_listen_media import (
    _Base as _ListenBase, _frame, _loud, _quiet)

DEV = "Microphone (Synthetic Desk Mic)"
OTHER = "Headset Microphone (Synthetic Headset)"


@requires_monolith
class _SilentBase(_ListenBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        from core import audio_processor as ap
        self.ap = ap
        with ap._vad_state_lock:
            saved = dict(ap._vad_state)

        def _restore():
            with ap._vad_state_lock:
                ap._vad_state.clear()
                ap._vad_state.update(saved)
        self.addCleanup(_restore)
        real_report = bc._report_silent_mic
        self._edges()
        # _edges stubs the reporter; these tests are ABOUT it.
        self._p(bc, "_report_silent_mic", real_report)
        self._p(bc, "MIC_BUS_MODE", "off", create=True)
        # These tests pin the v3.0.26 path (warn at once); the one automatic
        # reopen first (2026-11-21) is test_monolith_silent_mic_reopen's.
        self._p(bc, "MIC_SILENT_REOPEN_FIRST", False, create=True)
        self._p(bc, "_safe_close_stream", lambda *a, **k: None)
        self._p(bc, "_tts_playback_active", [False])
        self.cache = dict(bc._device_cache)
        self.cache["last_in_name"] = DEV
        self._p(bc, "_device_cache", self.cache)
        self.said: list = []
        self._p(bc, "proactive_announce",
                side_effect=lambda m, *a, **k: self.said.append((m, k)) or True)
        self.printed: list = []
        self.now = time.time()

    def _seed(self, *, audible_ago: float, poll_ago: float) -> None:
        """core.audio_processor's real clock: the last audible chunk and the
        last inspected chunk were this long ago, nothing unpolled booked."""
        with self.ap._vad_state_lock:
            self.ap._vad_state.update({
                "last_audible_chunk_ts": self.now - audible_ago,
                "last_vad_poll_ts": self.now - poll_ago,
                "vad_session_start": self.now - 3600.0,
                "unpolled_s": 0.0,
            })

    def _latched(self, ago: float = 100.0, device: str = DEV) -> None:
        """A warning was issued ``ago`` seconds ago on ``device``."""
        bc = self.bc
        bc._silent_mic_warned[0] = True
        bc._silent_mic_warned_at[0] = self.now - ago
        bc._silent_mic_warned_device[0] = device
        bc._silent_mic_recover_s[0] = 0.0

    def _listen(self, frames, timeout: float = 0.3):
        """One REAL record_speech over a stream that delivers ``frames``."""
        bc = self.bc

        class Feed:
            def __init__(s, *a, callback=None, **k):
                s.cb = callback

            def start(s):
                for f in frames:
                    s.cb(f.reshape(-1, 1), len(f), None, None)

        self._p(bc.sd, "InputStream", Feed)
        with mock.patch("builtins.print",
                        side_effect=lambda *a, **k: self.printed.append(
                            " ".join(str(x) for x in a))):
            return bc.record_speech(timeout=timeout)

    def _report(self, age: float, at: float) -> bool:
        with mock.patch("builtins.print"):
            return self.bc._report_silent_mic(age, at)


class RearmAndThrottleTests(_SilentBase):

    def test_noise_floor_does_not_re_arm_or_reset_the_throttle(self):
        """THE DEFECT: one chunk just over the floor cleared the latch and the
        throttle, so the next 30 s of quiet room warned again."""
        bc = self.bc
        self._latched(ago=100.0)
        self._seed(audible_ago=100.0, poll_ago=0.05)
        # A quiet room: every chunk just over the 5e-5 floor, far under 3e-4.
        self.assertIsNone(self._listen([_frame(6e-5)] * 40))
        self.assertTrue(bc._silent_mic_warned[0],
                        "one chunk at the noise floor is not a recovery")
        self.assertEqual(bc._silent_mic_warned_at[0], self.now - 100.0,
                         "the re-warn throttle must survive a recovery")
        self.assertFalse(any("audible again" in ln for ln in self.printed),
                         self.printed)
        # ...so the next silent stretch on the same device is NOT re-warned.
        self.assertFalse(self._report(45.0, self.now + 5.0))
        self.assertEqual(self.said, [])

    def test_a_sustained_recovery_clears_the_latch_but_not_the_throttle(self):
        bc = self.bc
        self._latched(ago=100.0)
        self._seed(audible_ago=100.0, poll_ago=0.05)
        clip = self._listen([_loud(0)] * 3 + [_quiet(0)] * 40)
        self.assertIsNotNone(clip, "he spoke: a capture")
        self.assertFalse(bc._silent_mic_warned[0],
                         "speech is a sustained recovery")
        self.assertTrue(any("audible again" in ln and "sustained" in ln
                            for ln in self.printed), self.printed)
        self.assertEqual(bc._silent_mic_warned_at[0], self.now - 100.0)
        self.assertFalse(self._report(45.0, self.now + 5.0),
                         "a recovery must not shorten the 900 s limit")
        self.assertTrue(self._report(
            45.0, self.now - 100.0 + bc.MIC_SILENT_REWARN_SECONDS + 1.0))
        self.assertEqual(len(self.said), 1)

    def test_three_seconds_of_clear_sound_is_a_recovery_two_is_not(self):
        bc = self.bc
        self._latched(ago=100.0)
        self._seed(audible_ago=100.0, poll_ago=0.05)
        chunk_s = 1024 / 16000.0
        two_s = int(2.0 / chunk_s)                 # 31 chunks
        self._listen([_frame(5e-4)] * two_s)       # clear, under VAD 0.008
        self.assertTrue(bc._silent_mic_warned[0])
        self._listen([_frame(5e-4)] * (int(1.2 / chunk_s)))
        self.assertFalse(bc._silent_mic_warned[0],
                         "3 s of chunks over 3e-4 is a sustained recovery")

    def test_a_different_device_still_warns_at_once(self):
        self._latched(ago=10.0, device=OTHER)
        self.assertTrue(self._report(45.0, self.now))
        self.assertEqual(len(self.said), 1)


class SilenceClockTests(_SilentBase):

    def _spy(self):
        calls = []
        self._p(self.bc, "_report_silent_mic",
                lambda age, now: calls.append(age) or True)
        return calls

    def test_time_nobody_was_listening_is_not_silence(self):
        """16:02:53: 40 s since the last audible chunk, but JARVIS was
        speaking for all of it - the first inspected chunk must not warn."""
        calls = self._spy()
        self._seed(audible_ago=40.0, poll_ago=40.0)
        self._listen([_frame(1e-5)] * 20)
        self.assertEqual(calls, [], "warned about time nobody was listening")
        with self.ap._vad_state_lock:
            self.assertGreater(self.ap._vad_state["last_vad_poll_ts"],
                               self.now - 1.0, "the capture loop really ran")

    def test_heard_silence_still_warns(self):
        """NEGATIVE CONTROL: 40 s of LISTENED-TO silence (a muted headset at
        rms 1e-5) still warns - the fix must not make a dead mic quiet."""
        calls = self._spy()
        self._seed(audible_ago=40.0, poll_ago=0.05)
        self._listen([_frame(1e-5)] * 20)
        self.assertTrue(calls, "a dead mic must still be reported")
        self.assertGreater(calls[0], 30.0)

    def test_not_while_jarvis_is_speaking(self):
        calls = self._spy()
        self.bc._tts_playback_active[0] = True
        self._seed(audible_ago=40.0, poll_ago=0.05)
        self._listen([_frame(1e-5)] * 20)
        self.assertEqual(calls, [])


class QueuedWarningTests(_SilentBase):

    def test_the_queued_line_is_tagged_once_and_checkable(self):
        self.assertTrue(self._report(45.0, self.now))
        (msg, kw), = self.said
        self.assertIn("completely silent", msg)
        self.assertEqual(kw.get("supersede"), self.bc._MIC_SILENT_SUPERSEDE)
        self.assertEqual(kw.get("dedupe_key"), "audio:mic_silent")
        self.assertEqual(kw.get("still_valid"), f"mic_silent:{DEV}")

    def test_still_valid_follows_the_latch_and_the_device(self):
        bc = self.bc
        self._report(45.0, self.now)
        token = self.said[0][1]["still_valid"]
        self.assertTrue(bc._silent_mic_still_valid(token))
        self.cache["last_in_name"] = OTHER
        self.assertFalse(bc._silent_mic_still_valid(token),
                         "JARVIS moved off that mic")
        self.cache["last_in_name"] = DEV
        bc._silent_mic_warned[0] = False           # a sustained recovery
        self.assertFalse(bc._silent_mic_still_valid(token))

    def test_the_queue_age_limit_is_the_re_warn_limit(self):
        from core import pending_policy as pp
        self.assertEqual(pp.QUEUE_RULES["mic_silent"][0],
                         self.bc.MIC_SILENT_REWARN_SECONDS)
        self.assertFalse(pp.QUEUE_RULES["mic_silent"][1], "never folded")


if __name__ == "__main__":
    unittest.main()
