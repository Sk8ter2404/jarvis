"""Monolith wiring tests for the experimental low-latency voice paths.

These assert that the THIN branch points added to bobert_companion for F1
(realtime streaming capture) and F2 (neural wake detector in standby) behave
correctly — most importantly that with the DEFAULT (off) flags the historical
path is taken and the realtime/wake selector is never engaged, and that when a
flag is on but the selector returns None (deps absent / init failed) the code
falls back to the existing path.

The SELECTION logic itself is unit-tested CI-side in tests/test_voice_pipeline.py
with the deps mocked absent; here we only verify the monolith's call+branch+
fallback glue. These tests import the real ~13K-line monolith and so run in the
LOCAL full-deps tier only — @requires_monolith (via MonolithGlobalsTestCase)
skips them cleanly on the light-deps CI runner.

No real audio, mic, model, or thread: record_speech / transcribe / the realtime
session / the wake detector are all replaced with fakes via mock.patch.object on
the harness-cached module. The new single-element-list latches the wiring uses
(_realtime_session, _realtime_disabled_for_session, _standby_wake_detector,
_standby_wake_disabled_for_session) are reset around every test so no state
leaks between them.

stdlib unittest + unittest.mock only; no personal data.
"""
from __future__ import annotations

import contextlib
import queue
import sys
import unittest
from unittest import mock

import numpy as np

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class _VoiceWiringBase(MonolithGlobalsTestCase):
    """Shared setup: reset the wiring's module-level latches before AND after
    each test so a session/detector built (or a disable-latch tripped) in one
    test can't bleed into the next. MonolithGlobalsTestCase doesn't track these
    new names, so we manage them explicitly here."""

    _LATCH_RESETS = {
        "_realtime_session": [None],
        "_realtime_disabled_for_session": [False],
        "_standby_wake_detector": [None],
        "_standby_wake_disabled_for_session": [False],
    }

    def setUp(self):
        super().setUp()
        # Reset latches + drain the realtime utterance queue to a clean slate.
        self._reset_latches()

    def tearDown(self):
        self._reset_latches()
        super().tearDown()

    def _reset_latches(self):
        bc = self.bc
        for name, val in self._LATCH_RESETS.items():
            if hasattr(bc, name):
                getattr(bc, name)[:] = val
        # Empty the module-level realtime utterance queue.
        q = getattr(bc, "_realtime_utterances", None)
        if isinstance(q, queue.Queue):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    @contextlib.contextmanager
    def _patched_vp(self, fake_vp):
        """Make `from core import voice_pipeline as _vp` (inside the monolith's
        wiring helpers) resolve to `fake_vp` for the duration of the block.

        Patches BOTH sys.modules['core.voice_pipeline'] AND the `core` package
        attribute, because once the real submodule has been imported earlier in
        the process (e.g. by tests/test_voice_pipeline.py), ``from core import
        voice_pipeline`` resolves via the parent's attribute, not sys.modules —
        so patching sys.modules alone wouldn't shadow it. Both are restored on
        exit (including the prior attribute, or its absence)."""
        import core as core_pkg
        sentinel = object()
        prev_attr = getattr(core_pkg, "voice_pipeline", sentinel)
        with mock.patch.dict(sys.modules, {"core.voice_pipeline": fake_vp}):
            setattr(core_pkg, "voice_pipeline", fake_vp)
            try:
                yield fake_vp
            finally:
                if prev_attr is sentinel:
                    try:
                        delattr(core_pkg, "voice_pipeline")
                    except AttributeError:
                        pass
                else:
                    setattr(core_pkg, "voice_pipeline", prev_attr)


# ──────────────────────────────────────────────────────────────────────
# F1: realtime capture branch in _capture_utterance
# ──────────────────────────────────────────────────────────────────────

class RealtimeCaptureWiringTests(_VoiceWiringBase):
    def test_default_flags_take_turn_based_path_selector_not_engaged(self):
        """Default (VOICE_MODE off): _get_realtime_session must return None and
        the capture path must call the historical record_speech()."""
        bc = self.bc
        rec_called = {"n": 0}

        def fake_record(timeout=None):
            rec_called["n"] += 1
            return None  # simulate silence → capture returns None cleanly

        with mock.patch.object(bc, "_get_realtime_session",
                               return_value=None) as get_sess, \
                mock.patch.object(bc, "record_speech", side_effect=fake_record), \
                mock.patch.object(bc, "_speak_pending", return_value=False), \
                mock.patch.object(bc, "should_be_proactive", return_value=False), \
                mock.patch.object(bc, "_do_proactive_turn"), \
                mock.patch.object(bc, "set_state"), \
                mock.patch.object(bc, "resume_face_tracking"), \
                mock.patch.object(bc, "_heartbeat"), \
                mock.patch("builtins.print"):
            out = bc._capture_utterance(None, memory=mock.Mock())
        self.assertIsNone(out)
        get_sess.assert_called()          # selector consulted
        self.assertEqual(rec_called["n"], 1)  # historical path ran

    def test_realtime_session_present_uses_streaming_capture(self):
        """When a session exists, capture pulls from the realtime queue and does
        NOT call record_speech()."""
        bc = self.bc
        sentinel_session = object()
        with mock.patch.object(bc, "_get_realtime_session",
                               return_value=sentinel_session), \
                mock.patch.object(bc, "_realtime_capture",
                                  return_value=("turn on the lights",
                                                {"no_speech_prob": 0.0,
                                                 "avg_logprob": -0.1})) as cap, \
                mock.patch.object(bc, "record_speech") as rec, \
                mock.patch.object(bc, "set_state"), \
                mock.patch.object(bc, "resume_face_tracking"), \
                mock.patch.object(bc, "_heartbeat"), \
                mock.patch.object(bc, "_speak_pending", return_value=False), \
                mock.patch("builtins.print"):
            out = bc._capture_utterance(None, memory=mock.Mock())
        self.assertIsNotNone(out)
        text, conf = out
        self.assertEqual(text, "turn on the lights")
        self.assertIn("no_speech_prob", conf)
        cap.assert_called_once()
        rec.assert_not_called()           # streaming path replaced the mic

    def test_realtime_capture_none_mirrors_timeout_branch(self):
        """Realtime capture returning None (no utterance in window) must behave
        like record_speech() timing out: check reminders, maybe proactive, then
        return None — WITHOUT falling through to record_speech()."""
        bc = self.bc
        with mock.patch.object(bc, "_get_realtime_session",
                               return_value=object()), \
                mock.patch.object(bc, "_realtime_capture", return_value=None), \
                mock.patch.object(bc, "record_speech") as rec, \
                mock.patch.object(bc, "_speak_pending", return_value=False), \
                mock.patch.object(bc, "should_be_proactive",
                                  return_value=True) as prox, \
                mock.patch.object(bc, "_do_proactive_turn") as do_prox, \
                mock.patch.object(bc, "set_state"), \
                mock.patch.object(bc, "resume_face_tracking"), \
                mock.patch.object(bc, "_heartbeat"), \
                mock.patch("builtins.print"):
            out = bc._capture_utterance(None, memory=mock.Mock())
        self.assertIsNone(out)
        rec.assert_not_called()
        prox.assert_called_once()
        do_prox.assert_called_once()

    def test_realtime_capture_raises_falls_back_to_turn_based(self):
        """If the realtime capture raises, the wiring latches a disable and
        falls through to record_speech() so the turn still happens."""
        bc = self.bc

        def boom(timeout=20.0):
            raise RuntimeError("queue boom")

        with mock.patch.object(bc, "_get_realtime_session",
                               return_value=object()), \
                mock.patch.object(bc, "_realtime_capture", side_effect=boom), \
                mock.patch.object(bc, "record_speech",
                                  return_value=None) as rec, \
                mock.patch.object(bc, "_speak_pending", return_value=False), \
                mock.patch.object(bc, "should_be_proactive", return_value=False), \
                mock.patch.object(bc, "_do_proactive_turn"), \
                mock.patch.object(bc, "set_state"), \
                mock.patch.object(bc, "resume_face_tracking"), \
                mock.patch.object(bc, "_heartbeat"), \
                mock.patch("builtins.print"):
            out = bc._capture_utterance(None, memory=mock.Mock())
        self.assertIsNone(out)
        rec.assert_called_once()          # fell back to mic capture
        self.assertTrue(bc._realtime_disabled_for_session[0])

    def test_inject_path_unaffected_by_realtime(self):
        """An injected command must still bypass mic + the realtime branch and
        return its pass-through tuple unchanged (default-behaviour guard)."""
        bc = self.bc
        with mock.patch.object(bc, "_get_realtime_session") as get_sess, \
                mock.patch.object(bc, "record_speech") as rec, \
                mock.patch.object(bc, "_speak_pending", return_value=False), \
                mock.patch.object(bc, "set_state"), \
                mock.patch.object(bc, "_heartbeat"), \
                mock.patch("builtins.print"):
            out = bc._capture_utterance("hello jarvis", memory=mock.Mock())
        self.assertIsNotNone(out)
        text, conf = out
        self.assertEqual(text, "hello jarvis")
        # Inject short-circuits BEFORE the realtime branch and the mic.
        get_sess.assert_not_called()
        rec.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# F1: _get_realtime_session / _realtime_capture units
# ──────────────────────────────────────────────────────────────────────

class GetRealtimeSessionTests(_VoiceWiringBase):
    def test_returns_none_and_latches_when_selector_off(self):
        bc = self.bc
        fake_vp = mock.Mock()
        fake_vp.realtime_enabled.return_value = False
        with self._patched_vp(fake_vp):
            self.assertIsNone(bc._get_realtime_session())
        # Latched so we don't re-probe each loop iteration.
        self.assertTrue(bc._realtime_disabled_for_session[0])

    def test_returns_none_when_make_session_returns_none(self):
        bc = self.bc
        fake_vp = mock.Mock()
        fake_vp.realtime_enabled.return_value = True
        fake_vp.make_realtime_session.return_value = None
        with self._patched_vp(fake_vp):
            self.assertIsNone(bc._get_realtime_session())
        self.assertTrue(bc._realtime_disabled_for_session[0])

    def test_caches_built_session(self):
        bc = self.bc
        sentinel = object()
        fake_vp = mock.Mock()
        fake_vp.realtime_enabled.return_value = True
        fake_vp.make_realtime_session.return_value = sentinel
        with self._patched_vp(fake_vp):
            first = bc._get_realtime_session()
            second = bc._get_realtime_session()
        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        # make_realtime_session built it once; the cache served the 2nd call.
        self.assertEqual(fake_vp.make_realtime_session.call_count, 1)

    def test_disabled_latch_short_circuits(self):
        bc = self.bc
        bc._realtime_disabled_for_session[0] = True
        with mock.patch.dict("sys.modules", {}):
            # Even with no voice_pipeline import attempted, returns None fast.
            self.assertIsNone(bc._get_realtime_session())

    def test_on_utterance_callback_feeds_queue(self):
        """The session's on_user_utterance hook must enqueue onto the module's
        realtime queue so _realtime_capture can drain it."""
        bc = self.bc
        captured = {}
        fake_vp = mock.Mock()
        fake_vp.realtime_enabled.return_value = True

        def fake_make(**kwargs):
            captured.update(kwargs)
            return object()

        fake_vp.make_realtime_session.side_effect = fake_make
        with self._patched_vp(fake_vp):
            bc._get_realtime_session()
        cb = captured.get("on_user_utterance")
        self.assertIsNotNone(cb)
        cb("lights on")
        # The capture helper should now return that utterance.
        out = bc._realtime_capture(timeout=0.1)
        self.assertEqual(out[0], "lights on")


class RealtimeCaptureHelperTests(_VoiceWiringBase):
    def test_empty_queue_returns_none(self):
        bc = self.bc
        self.assertIsNone(bc._realtime_capture(timeout=0.05))

    def test_blank_utterance_returns_none(self):
        bc = self.bc
        bc._realtime_utterances.put("   ")
        self.assertIsNone(bc._realtime_capture(timeout=0.1))

    def test_real_utterance_returns_tuple(self):
        bc = self.bc
        bc._realtime_utterances.put("hello sir")
        out = bc._realtime_capture(timeout=0.1)
        self.assertEqual(out[0], "hello sir")
        self.assertEqual(out[1]["no_speech_prob"], 0.0)


# ──────────────────────────────────────────────────────────────────────
# F2: neural wake detector branch in _handle_sleep_standby
# ──────────────────────────────────────────────────────────────────────

class StandbyWakeWiringTests(_VoiceWiringBase):
    def _common_patches(self, bc):
        """Patches shared by the standby tests: silence the music feed, state,
        heartbeat, prints; a real 1-second audio buffer so the len() gate (>=
        0.4s) passes."""
        sr = int(getattr(bc, "SAMPLE_RATE", 16000))
        audio = np.zeros(sr, dtype=np.float32)
        return audio, [
            mock.patch.object(bc, "record_speech", return_value=audio),
            mock.patch.object(bc, "_audio_music_feed"),
            mock.patch.object(bc, "set_state"),
            mock.patch.object(bc, "_heartbeat"),
            mock.patch("builtins.print"),
        ]

    def test_default_flags_use_whisper_path_selector_not_engaged(self):
        """Default (WAKE_WORD_AUTOSTART off): _standby_wake_detected returns None
        and the standby loop runs the full transcribe() exactly as before."""
        bc = self.bc
        audio, patches = self._common_patches(bc)
        with mock.patch.object(bc, "_standby_wake_detected",
                               return_value=None) as wake, \
                mock.patch.object(bc, "transcribe",
                                  return_value=("nothing here",
                                                {})) as tx, \
                mock.patch.object(bc, "_ambient_learning", [False]), \
                mock.patch.object(bc, "_sleep_mode", [True]), \
                mock.patch.object(bc, "_standby_mode", [False]):
            for p in patches:
                p.start()
                self.addCleanup(p.stop)
            bc._handle_sleep_standby(None)
        wake.assert_called_once_with(audio)
        tx.assert_called_once()           # Whisper path taken (default)

    def test_neural_true_wakes_without_transcribe(self):
        """Detector reports a wake → JARVIS wakes and transcribe() is NEVER
        called (the whole latency win)."""
        bc = self.bc
        audio, patches = self._common_patches(bc)
        with mock.patch.object(bc, "_standby_wake_detected",
                               return_value=True), \
                mock.patch.object(bc, "transcribe") as tx, \
                mock.patch.object(bc, "_audio_music_should_refuse_wake",
                                  return_value=False), \
                mock.patch.object(bc, "context_aware_greeting",
                                  return_value=("Yes, sir?", 1.0)), \
                mock.patch.object(bc, "_speak"), \
                mock.patch.object(bc, "_write_hud_state"), \
                mock.patch.object(bc, "_sleep_mode", [True]), \
                mock.patch.object(bc, "_standby_mode", [True]), \
                mock.patch.object(bc, "_ambient_music_hits", [0]), \
                mock.patch.object(bc, "_ambient_learning", [False]), \
                mock.patch.object(bc, "_resume_to_ambient", [False]):
            for p in patches:
                p.start()
                self.addCleanup(p.stop)
            bc._handle_sleep_standby(None)
            # Capture flag state WHILE the patches are live — the with-exit
            # restores _sleep_mode/_standby_mode to their original objects, so
            # reading them after the block would test the restored value.
            tx.assert_not_called()
            slept_after = bc._sleep_mode[0]
            standby_after = bc._standby_mode[0]
        # Woke up: both sleep + standby flags cleared.
        self.assertFalse(slept_after)
        self.assertFalse(standby_after)

    def test_neural_false_stays_asleep_without_transcribe(self):
        """Detector reports NO wake → JARVIS stays asleep and transcribe() is
        not called; the sleep flag remains set."""
        bc = self.bc
        audio, patches = self._common_patches(bc)
        with mock.patch.object(bc, "_standby_wake_detected",
                               return_value=False), \
                mock.patch.object(bc, "transcribe") as tx, \
                mock.patch.object(bc, "_ambient_learning", [False]), \
                mock.patch.object(bc, "_sleep_mode", [True]), \
                mock.patch.object(bc, "_standby_mode", [False]):
            for p in patches:
                p.start()
                self.addCleanup(p.stop)
            bc._handle_sleep_standby(None)
            # Read the flag WHILE patched (with-exit restores the original).
            tx.assert_not_called()
            slept_after = bc._sleep_mode[0]
        self.assertTrue(slept_after)   # still asleep

    def test_inject_standby_path_unaffected(self):
        """An injected wake phrase must still wake JARVIS via the existing
        substring check, never touching the neural detector or the mic."""
        bc = self.bc
        with mock.patch.object(bc, "_standby_wake_detected") as wake, \
                mock.patch.object(bc, "record_speech") as rec, \
                mock.patch.object(bc, "transcribe") as tx, \
                mock.patch.object(bc, "_audio_music_should_refuse_wake",
                                  return_value=False), \
                mock.patch.object(bc, "context_aware_greeting",
                                  return_value=("Yes, sir?", 1.0)), \
                mock.patch.object(bc, "_speak"), \
                mock.patch.object(bc, "_write_hud_state"), \
                mock.patch.object(bc, "_heartbeat"), \
                mock.patch.object(bc, "set_state"), \
                mock.patch.object(bc, "_sleep_mode", [True]), \
                mock.patch.object(bc, "_standby_mode", [False]), \
                mock.patch.object(bc, "_ambient_music_hits", [0]), \
                mock.patch.object(bc, "_ambient_learning", [False]), \
                mock.patch.object(bc, "_resume_to_ambient", [False]), \
                mock.patch("builtins.print"):
            bc._handle_sleep_standby("jarvis wake up")
            slept_after = bc._sleep_mode[0]   # read while patched
        wake.assert_not_called()   # neural detector untouched on inject path
        rec.assert_not_called()
        tx.assert_not_called()
        self.assertFalse(slept_after)  # woke via the inject substring


# ──────────────────────────────────────────────────────────────────────
# F2: _get_standby_wake_detector / _standby_wake_detected units
# ──────────────────────────────────────────────────────────────────────

class GetStandbyWakeDetectorTests(_VoiceWiringBase):
    def test_returns_none_and_latches_when_flag_off(self):
        bc = self.bc
        fake_vp = mock.Mock()
        fake_vp.wake_word_autostart_enabled.return_value = False
        with self._patched_vp(fake_vp):
            self.assertIsNone(bc._get_standby_wake_detector())
        self.assertTrue(bc._standby_wake_disabled_for_session[0])

    def test_builds_idle_detector_when_enabled(self):
        bc = self.bc
        sentinel = object()
        captured = {}
        fake_vp = mock.Mock()
        fake_vp.wake_word_autostart_enabled.return_value = True

        def fake_make(**kwargs):
            captured.update(kwargs)
            return sentinel

        fake_vp.make_wake_detector.side_effect = fake_make
        with self._patched_vp(fake_vp):
            got = bc._get_standby_wake_detector()
        self.assertIs(got, sentinel)
        # autostart=False so it never opens a 2nd mic stream.
        self.assertFalse(captured.get("autostart", True))

    def test_disabled_latch_short_circuits(self):
        bc = self.bc
        bc._standby_wake_disabled_for_session[0] = True
        self.assertIsNone(bc._get_standby_wake_detector())


class StandbyWakeDetectedTests(_VoiceWiringBase):
    def test_none_when_no_detector(self):
        bc = self.bc
        with mock.patch.object(bc, "_get_standby_wake_detector",
                               return_value=None):
            self.assertIsNone(bc._standby_wake_detected(
                np.zeros(16000, dtype=np.float32)))

    def test_true_when_event_fires(self):
        bc = self.bc
        fake_det = mock.Mock()
        fake_det.events = queue.Queue()

        def fake_on_frame(frame):
            # Simulate a wake: enqueue an event the first time we see a frame.
            if fake_det.events.empty():
                fake_det.events.put({"phrase": "jarvis", "score": 0.95})

        fake_det._on_frame.side_effect = fake_on_frame
        with mock.patch.object(bc, "_get_standby_wake_detector",
                               return_value=fake_det):
            out = bc._standby_wake_detected(np.zeros(16000, dtype=np.float32))
        self.assertIs(out, True)

    def test_false_when_no_event(self):
        bc = self.bc
        fake_det = mock.Mock()
        fake_det.events = queue.Queue()
        fake_det._on_frame.side_effect = lambda frame: None  # never fires
        with mock.patch.object(bc, "_get_standby_wake_detector",
                               return_value=fake_det):
            out = bc._standby_wake_detected(np.zeros(16000, dtype=np.float32))
        self.assertIs(out, False)

    def test_detector_error_returns_none_and_latches(self):
        bc = self.bc
        fake_det = mock.Mock()
        fake_det.events = queue.Queue()
        fake_det._on_frame.side_effect = RuntimeError("predict boom")
        with mock.patch.object(bc, "_get_standby_wake_detector",
                               return_value=fake_det), \
                mock.patch("builtins.print"):
            out = bc._standby_wake_detected(np.zeros(16000, dtype=np.float32))
        self.assertIsNone(out)
        self.assertTrue(bc._standby_wake_disabled_for_session[0])


if __name__ == "__main__":
    unittest.main()
