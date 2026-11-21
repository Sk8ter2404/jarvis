"""The silent-mic warning asks for ONE automatic stream reopen first
(2026-11-21, overhaul speech-mic-silent-rearm point d, MIC_SILENT_REOPEN_FIRST).

Before v3.0.27 the warning was spoken the moment the listened silence passed
MIC_SILENT_WARN_SECONDS, even when a fresh stream open would have cured the
stream (live 11-20 20:57:36: a warning on a mic that was heard again 38 s
later). Now the detector first asks for one reopen and warns only when the
mic stays silent for MIC_SILENT_REOPEN_GRACE_S of listened time after it.

What is faked: the PortAudio stream (tests/monolith/test_monolith_silent_mic
._SilentBase, real record_speech), the mic bus (a stand-in with
request_reopen), the clock (passed in). Device names are SYNTHETIC.

    python tools/run_tests.py monolith.test_monolith_silent_mic_reopen
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests.monolith.test_monolith_listen_media import _frame
from tests.monolith.test_monolith_silent_mic import _SilentBase


class _FakeBus:
    def __init__(self, ok=True):
        self.ok = ok
        self.asked = 0

    def request_reopen(self):
        self.asked += 1
        return self.ok


class _ReopenBase(_SilentBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "MIC_SILENT_REOPEN_FIRST", True, create=True)
        self._p(bc, "MIC_SILENT_REOPEN_GRACE_S", 20.0, create=True)
        self._p(bc, "_silent_mic_reopen", [0.0, 0.0, "", 0.0])
        self._p(bc, "_silent_mic_warned", [False])
        self._p(bc, "_silent_mic_warned_at", [0.0])
        self._p(bc, "_silent_mic_warned_device", [""])
        self.bus = _FakeBus()
        self.cap = {"bus": self.bus}

    def gate(self, age, at, *, recording=False, tts=False, cap="bus"):
        with mock.patch("builtins.print"):
            return self.bc._silent_mic_gate(
                age, at, recording=recording, tts_playing=tts,
                bus_cap=self.cap if cap == "bus" else cap)

    def heard_now(self, ts):
        with self.ap._vad_state_lock:
            self.ap._vad_state["last_audible_chunk_ts"] = ts


class GateTests(_ReopenBase):

    def test_reopen_fixes_it_no_warning(self):
        """The reopen brings sound back: the silence clock restarts and the
        gate is never asked again - nothing is reported or said."""
        self._seed(audible_ago=31.0, poll_ago=0.05)
        self.assertEqual(self.gate(31.0, self.now), "hold")
        self.assertEqual(self.bus.asked, 1)
        # Before the grace runs out the warning still holds...
        self.assertEqual(self.gate(45.0, self.now + 14.0), "hold")
        # ...and then the mic delivers sound (the reopen worked).
        self.heard_now(self.now + 15.0)
        self.assertEqual(self.said, [])
        self.assertFalse(self.bc._silent_mic_warned[0])
        self.assertEqual(self.bus.asked, 1)

    def test_reopen_does_not_fix_it_one_warning(self):
        """Still silent MIC_SILENT_REOPEN_GRACE_S after the reopen: exactly
        one warning, and no second reopen."""
        self._seed(audible_ago=31.0, poll_ago=0.05)
        results = []
        for k in range(0, 40):
            age, at = 31.0 + k, self.now + k
            if self.gate(age, at) == "warn":
                results.append(self._report(age, at))
        self.assertEqual(self.bus.asked, 1, "one automatic reopen only")
        self.assertEqual(results.count(True), 1, results)
        self.assertEqual(len(self.said), 1)
        self.assertIn("completely silent", self.said[0][0])
        # The first warn came at the grace, not before.
        self.assertTrue(self.bc._silent_mic_warned[0])
        self.assertGreaterEqual(self.bc._silent_mic_warned_at[0],
                                self.now + 20.0)

    def test_tts_playing_or_a_capture_never_reopens(self):
        self._seed(audible_ago=31.0, poll_ago=0.05)
        self.assertEqual(self.gate(31.0, self.now, tts=True), "hold")
        self.assertEqual(self.gate(31.0, self.now, recording=True), "hold")
        self.assertEqual(self.bus.asked, 0)
        self.assertEqual(self.said, [])
        # The next idle chunk asks.
        self.assertEqual(self.gate(32.0, self.now + 1.0), "hold")
        self.assertEqual(self.bus.asked, 1)

    def test_flag_off_is_the_old_path(self):
        self._p(self.bc, "MIC_SILENT_REOPEN_FIRST", False, create=True)
        self.assertEqual(self.gate(31.0, self.now), "warn")
        self.assertEqual(self.bus.asked, 0)

    def test_a_reopen_that_cannot_be_asked_warns_at_once(self):
        self.bus.ok = False                  # bus suspended / not open
        self._seed(audible_ago=31.0, poll_ago=0.05)
        self.assertEqual(self.gate(31.0, self.now), "warn")

    def test_an_error_in_the_gate_is_the_old_path(self):
        self._p(self.bc, "_silent_mic_reopen", None)   # unpackable -> raises
        self.assertEqual(self.gate(31.0, self.now), "warn")

    def test_sound_then_silence_again_inside_the_window_warns(self):
        """A dead mic that clicks on every reopen must not be reopened
        forever and never reported."""
        self._seed(audible_ago=31.0, poll_ago=0.05)
        self.assertEqual(self.gate(31.0, self.now), "hold")
        self.heard_now(self.now + 1.0)       # one click after the reopen
        self.assertEqual(self.gate(31.0, self.now + 32.0), "warn")
        self.assertEqual(self.bus.asked, 1)

    def test_the_re_warn_throttle_never_reopens(self):
        self._latched(ago=100.0)
        self._seed(audible_ago=131.0, poll_ago=0.05)
        self.assertEqual(self.gate(131.0, self.now), "warn")
        self.assertEqual(self.bus.asked, 0)
        self.assertFalse(self._report(131.0, self.now), "throttled no-op")

    def test_a_per_capture_stream_ends_the_idle_listen(self):
        self._seed(audible_ago=31.0, poll_ago=0.05)
        self.assertEqual(self.gate(31.0, self.now, cap=None), "exit")
        self.assertEqual(self.gate(35.0, self.now + 4.0, cap=None), "hold")


class RecordSpeechTests(_ReopenBase):
    """Through the REAL record_speech (per-capture stream, MIC_BUS_MODE off)."""

    def test_record_speech_reopens_instead_of_warning(self):
        self._seed(audible_ago=40.0, poll_ago=0.05)
        self.assertIsNone(self._listen([_frame(1e-5)] * 20, timeout=5.0))
        self.assertEqual(self.said, [], "no warning before the reopen")
        self.assertGreater(self.bc._silent_mic_reopen[0], 0.0)
        self.assertTrue(any("reopening the capture stream once" in ln
                            for ln in self.printed), self.printed)

    def test_record_speech_flag_off_warns_at_once(self):
        self._p(self.bc, "MIC_SILENT_REOPEN_FIRST", False, create=True)
        self._seed(audible_ago=40.0, poll_ago=0.05)
        self._listen([_frame(1e-5)] * 20)
        self.assertEqual(len(self.said), 1)
        self.assertEqual(self.bc._silent_mic_reopen[0], 0.0)

    def test_record_speech_tts_playing_no_reopen(self):
        self.bc._tts_playback_active[0] = True
        self._seed(audible_ago=40.0, poll_ago=0.05)
        self._listen([_frame(1e-5)] * 20)
        self.assertEqual(self.bc._silent_mic_reopen[0], 0.0)
        self.assertEqual(self.said, [])

    def test_the_flags_are_in_the_turn_flag_line(self):
        self.assertIn("MIC_SILENT_REOPEN_FIRST", self.bc._TURN_FLAG_KEYS)
        self.assertIn("MIC_SILENT_REOPEN_GRACE_S", self.bc._TURN_FLAG_KEYS)
        from core import config as cfg
        self.assertIs(cfg.MIC_SILENT_REOPEN_FIRST, True)
        self.assertEqual(cfg.MIC_SILENT_REOPEN_GRACE_S, 20.0)


if __name__ == "__main__":
    unittest.main()
