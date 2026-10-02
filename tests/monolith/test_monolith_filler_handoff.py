"""Speed plan R3 — the filler handoff, monolith glue (bobert_companion).

Phase 0 (scheduler hygiene; every flag ships at today's behaviour):
  * PROCESSING_FILLER_LATE_START_S reaches the shared ProcessingFiller through
    its late_fn; the default 3.0 is today's window.
  * PROCESSING_FILLER_SKIP_PLEASANTRIES: _filler_should_arm refuses a bare
    "thank you" / "hello" only when the flag is on.
  * FILLER_DUCK_HOLD: _filler_play takes ONE _audio_ducker hold after an "ok"
    claim; _filler_end_turn / _filler_teardown give it back and the next voice
    turn drops a stale one, so the count stays balanced on success, on a
    playback exception, on a raising turn and on teardown. While a hold is up
    with the flag on, a second duck() is a no-op even when the first scan
    matched nothing.

The scheduler itself: tests/test_processing_filler_r3.py (light tier).
No real audio; fakes only.

    python -m unittest tests.monolith.test_monolith_filler_handoff
"""
from __future__ import annotations

import contextlib
import io
import threading
import unittest
from unittest import mock

from tests.monolith.test_monolith_processing_filler import _Base


class _FakeDucker:
    """Records hold / release / duck / restore and keeps the real count."""

    def __init__(self):
        self.calls: list = []
        self.holds = 0

    def hold(self):
        self.calls.append("hold")
        self.holds += 1

    def release(self):
        self.calls.append("release")
        self.holds = max(0, self.holds - 1)

    def duck(self):
        self.calls.append("duck")

    def restore(self):
        self.calls.append("restore")


class _DuckBase(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.ducker = _FakeDucker()
        self._p(bc, "_audio_ducker", self.ducker)
        bc._filler_duck_held[0] = False
        self.addCleanup(bc._filler_duck_held.__setitem__, 0, False)


# ════════════════════════════════════════════════════════════════════════════
#  PROCESSING_FILLER_LATE_START_S
# ════════════════════════════════════════════════════════════════════════════
class LateStartWiringTests(_Base):
    def test_shared_filler_reads_the_knob_at_every_arm(self):
        bc = self.bc
        f = bc._processing_filler
        self.assertIsNotNone(f._late_fn)
        self._p(bc, "PROCESSING_FILLER_LATE_START_S", 0.6)
        self.assertEqual(f._read_late(), 0.6)
        self._p(bc, "PROCESSING_FILLER_LATE_START_S", 2.0)
        self.assertEqual(f._read_late(), 2.0)

    def test_shipped_default_is_todays_window(self):
        bc = self.bc
        from core import config
        from core import processing_filler as pf
        self.assertEqual(config.PROCESSING_FILLER_LATE_START_S, 3.0)
        self.assertIsInstance(config.PROCESSING_FILLER_LATE_START_S, float)
        f = bc._processing_filler
        self.assertEqual(f._first_retry_s, 3.0)
        self._p(bc, "PROCESSING_FILLER_LATE_START_S", 3.0)
        turn = pf.FillerTurn(0.0, 0.5, None, first_retry=f._read_late())
        # min(3.0, the fixed 1.0 cap) — exactly the window without the knob.
        self.assertEqual(f._first_window(turn), f._first_window())
        self.assertEqual(f._first_window(turn), bc._FILLER_FIRST_LATE_S)


# ════════════════════════════════════════════════════════════════════════════
#  PROCESSING_FILLER_SKIP_PLEASANTRIES
# ════════════════════════════════════════════════════════════════════════════
class PleasantryGateTests(_Base):
    PLEASANT = ("Thank you.", "Thanks, JARVIS.", "Hello.", "Good night, sir.",
                "Okay.")

    def test_flag_off_keeps_todays_gate(self):
        self._enable()
        self._p(self.bc, "PROCESSING_FILLER_SKIP_PLEASANTRIES", False)
        for text in self.PLEASANT:
            self.assertTrue(self.bc._filler_should_arm(text), text)

    def test_flag_on_skips_bare_pleasantries_only(self):
        bc = self.bc
        self._enable()
        self._p(bc, "PROCESSING_FILLER_SKIP_PLEASANTRIES", True)
        for text in self.PLEASANT:
            self.assertFalse(bc._filler_should_arm(text), text)
        for text in ("Thank you, what's the weather?", "hello, play music",
                     "what time is it"):
            self.assertTrue(bc._filler_should_arm(text), text)
        self.assertFalse(bc._filler_should_arm("stop"))   # quiet still wins

    def test_shipped_default_is_off(self):
        from core import config
        self.assertIs(config.PROCESSING_FILLER_SKIP_PLEASANTRIES, False)

    def test_dispatch_never_arms_for_a_pleasantry_with_the_flag_on(self):
        bc = self.bc
        self._enable()
        self._p(bc, "PROCESSING_FILLER_SKIP_PLEASANTRIES", True)
        self._p(bc, "_run_llm_dispatch_body", return_value="You're welcome.")
        self._p(bc, "_filler_warm_if_needed")
        fake = mock.Mock()
        fake.playing.return_value = False
        self._p(bc, "_processing_filler", fake)
        bc._run_llm_dispatch("Thank you, JARVIS.", voice=True)
        fake.arm.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
#  FILLER_DUCK_HOLD — the turn's hold
# ════════════════════════════════════════════════════════════════════════════
class DuckHoldTests(_DuckBase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._enable()
        self.lock = threading.Lock()
        self._p(bc, "_SPEAK_LOCK", self.lock)
        self.f = self._fresh_filler()
        self._fresh_clips()
        self.play = self._p(bc, "play_with_lipsync")
        self._p(bc, "set_state")
        self._p(bc, "_filler_warm_if_needed")

    def _play(self, turn, stage=1):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.bc._filler_play(turn, stage)

    def test_flag_off_takes_no_hold(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", False)
        turn = self.f.arm()
        self.assertEqual(self._play(turn), "played")
        bc._filler_end_turn(turn)
        self.assertEqual(self.ducker.calls, [])
        self.assertFalse(bc._filler_duck_held[0])

    def test_hold_taken_after_the_claim_and_given_back_at_turn_end(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        order = []
        self.ducker.hold = lambda: (order.append("hold"),
                                    setattr(self.ducker, "holds", 1))
        real_claim = self.f.claim

        def claim(turn, stage):
            order.append("claim")
            return real_claim(turn, stage)
        self._p(self.f, "claim", side_effect=claim)
        self.play.side_effect = lambda a, sr: order.append("play")
        turn = self.f.arm()
        self.assertEqual(self._play(turn), "played")
        self.assertEqual(order, ["claim", "hold", "play"])
        self.assertTrue(bc._filler_duck_held[0])
        bc._filler_end_turn(turn)
        self.assertFalse(bc._filler_duck_held[0])
        self.assertEqual(self.ducker.calls, ["release"])
        self.assertEqual(self.ducker.holds, 0)

    def test_stage_two_reuses_the_turns_hold(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        turn = self.f.arm()
        self.assertEqual(self._play(turn, 1), "played")
        turn.still = 0.0                       # stage 2 due at once
        self.assertEqual(self._play(turn, 2), "played")
        self.assertEqual(self.ducker.calls.count("hold"), 1)
        bc._filler_end_turn(turn)
        self.assertEqual(self.ducker.holds, 0)

    def test_no_hold_without_an_ok_claim(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        turn = self.f.arm()
        self.f.note_speech()                   # stage 1 is gone
        self.assertEqual(self._play(turn), "skipped")
        self.assertEqual(self.ducker.calls, [])

    def test_balanced_when_the_clip_fails_to_play(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        self.play.side_effect = RuntimeError("PortAudio reinit hung")
        turn = self.f.arm()
        self.assertEqual(self._play(turn), "played")
        bc._filler_end_turn(turn)
        self.assertEqual(self.ducker.holds, 0)
        self.assertFalse(bc._filler_duck_held[0])

    def test_balanced_when_the_turn_raises(self):
        # The clip plays (hold taken) inside the turn, then the turn raises:
        # the dispatch wrapper's finally (_filler_end_turn) gives it back.
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        seen = []

        def body(text):
            turn = self.f._current            # armed by the wrapper
            self.assertEqual(self._play(turn), "played")
            seen.append(self.ducker.holds)
            raise RuntimeError("x")
        self._p(bc, "_run_llm_dispatch_body", side_effect=body)
        with self.assertRaises(RuntimeError), \
                contextlib.redirect_stdout(io.StringIO()):
            bc._run_llm_dispatch("what's the weather", voice=True)
        self.assertEqual(seen, [1])
        self.assertEqual(self.ducker.holds, 0)
        self.assertEqual(self.ducker.calls, ["hold", "release"])

    def test_balanced_on_teardown(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        turn = self.f.arm()
        self._play(turn)
        self.assertEqual(self.ducker.holds, 1)
        bc._filler_teardown("tray:restart")
        self.assertEqual(self.ducker.holds, 0)
        self.assertFalse(bc._filler_duck_held[0])
        bc._filler_teardown("again")           # nothing left to give back
        self.assertEqual(self.ducker.calls.count("release"), 1)

    def test_next_voice_turn_drops_a_stale_hold(self):
        bc = self.bc
        self._p(bc, "FILLER_DUCK_HOLD", True)
        turn = self.f.arm()
        self._play(turn)
        self.f.disarm(turn)                    # its end_turn never ran
        self.assertEqual(self.ducker.holds, 1)
        bc._filler_should_arm("what's the weather")
        self.assertEqual(self.ducker.holds, 0)
        self.assertFalse(bc._filler_duck_held[0])

    def test_release_without_a_hold_is_a_no_op(self):
        bc = self.bc
        bc._filler_duck_release()
        bc._filler_end_turn(None)
        bc._filler_should_arm("x")
        self.assertEqual(self.ducker.calls, [])

    def test_hold_helpers_never_raise(self):
        bc = self.bc
        broken = mock.Mock()
        broken.hold.side_effect = RuntimeError("x")
        broken.release.side_effect = RuntimeError("y")
        self._p(bc, "_audio_ducker", broken)
        bc._filler_duck_hold()
        self.assertFalse(bc._filler_duck_held[0])
        bc._filler_duck_held[0] = True
        bc._filler_duck_release()
        self.assertFalse(bc._filler_duck_held[0])

    def test_shipped_default_is_off(self):
        from core import config
        self.assertIs(config.FILLER_DUCK_HOLD, False)


# ════════════════════════════════════════════════════════════════════════════
#  FILLER_DUCK_HOLD — _AudioDucker.duck: one scan per hold
# ════════════════════════════════════════════════════════════════════════════
class _DuckQueue:
    def __init__(self, log):
        self.log = log

    def put(self, job):
        _plans, target, _cancellable, done = job
        self.log.append(("fade", target))
        if done is not None:
            done.set()


class DuckerOneScanTests(_Base):
    def _ducker(self, matched):
        bc = self.bc
        d = bc._AudioDucker()
        self.scans = []
        self.fades = []

        def enum():
            self.scans.append(1)
            return list(matched)
        d._check_available = lambda: True
        d._enumerate_targets = enum
        d._ensure_worker = lambda: None
        d._work_queue = _DuckQueue(self.fades)
        self._p(bc, "AUDIO_DUCKING_ENABLED", True)
        return d

    def test_flag_off_a_held_unmatched_duck_scans_every_time(self):
        self._p(self.bc, "FILLER_DUCK_HOLD", False)
        d = self._ducker([])
        d.hold()
        d.duck()
        d.duck()
        self.assertEqual(len(self.scans), 2)       # today's behaviour
        d.release()

    def test_flag_on_a_held_duck_scans_once_even_when_nothing_matched(self):
        self._p(self.bc, "FILLER_DUCK_HOLD", True)
        d = self._ducker([])
        d.hold()
        d.duck()
        d.duck()
        d.duck()
        self.assertEqual(len(self.scans), 1)
        d.release()
        self.assertFalse(d._held_ducked)
        d.duck()                                   # not held: scans again
        self.assertEqual(len(self.scans), 2)

    def test_flag_on_without_a_hold_scans_every_time(self):
        self._p(self.bc, "FILLER_DUCK_HOLD", True)
        d = self._ducker([])
        d.duck()
        d.duck()
        self.assertEqual(len(self.scans), 2)
        self.assertFalse(d._held_ducked)

    def test_flag_on_matched_duck_restores_once_at_the_last_release(self):
        self._p(self.bc, "FILLER_DUCK_HOLD", True)
        d = self._ducker([("session", 0.8)])
        self._p(self.bc, "AUDIO_DUCKING_FADE_MS", 1)
        d.hold()
        d.duck()                  # the filler clip
        d.restore()               # its end: held, no swell
        d.duck()                  # the answer: no second scan
        d.restore()
        self.assertEqual(len(self.scans), 1)
        self.assertEqual(self.fades, [("fade", self.bc.AUDIO_DUCKING_LEVEL)])
        d.release()               # the end of the turn
        self.assertEqual(self.fades[-1], ("fade", None))
        self.assertEqual(d._holds, 0)

    def test_a_failed_scan_is_retried_by_the_next_duck(self):
        self._p(self.bc, "FILLER_DUCK_HOLD", True)
        d = self._ducker([])
        boom = [True]

        def enum():
            self.scans.append(1)
            if boom[0]:
                boom[0] = False
                raise OSError("COM")
            return []
        d._enumerate_targets = enum
        d.hold()
        with contextlib.redirect_stdout(io.StringIO()):
            d.duck()
        d.duck()
        d.duck()
        self.assertEqual(len(self.scans), 2)
        d.release()


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
