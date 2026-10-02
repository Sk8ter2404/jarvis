"""A cold first turn after a restart (NEW #8, 2026-10-02).

THE LIVE EVIDENCE (session_2026-10-01_21-48-36.log): the first voice turn
after the 21:48 restart (22:04:11) took 8.4 s to its first audio for a
one-sentence answer - the prompt was cold (pe=15934 / 3856 ms) and the 0.5 s
filler slipped to +3.6 s, then delayed the answer by about 0.9 s.

  * the boot prime at 21:49:59 was skipped ("utterance": music on the PC kept
    the desk mic tripping, so _utterance_in_progress was set) and
    _start_boot_reprime never tried again;
  * _reprime_after_background refused while _last_owner_turn_at == 0, so
    nothing re-warmed the prefix before his first turn either.

Now:
  * the boot prime retries every _BOOT_REPRIME_RETRY_S until it is primed
    (stops on a permanent reason, after the owner's first turn, or after
    _BOOT_REPRIME_MAX_TRIES);
  * a capture during sustained PC audio (two or more captures in a row that
    ended without a turn while the playback meter read audio) is music, not
    the owner mid-sentence: it no longer holds the re-prime off;
  * before the first owner turn the re-prime after background work is
    allowed (reference = main() reaching the turn loop, window at least an
    hour; an import or a test never starts it, so nothing changes there);
  * the monolith's filler skips a stage-1 line more than
    _FILLER_FIRST_LATE_S late (the scheduler half: tests/test_processing_
    filler_late.py).

    python -m unittest tests.monolith.test_monolith_boot_prime_retry
"""
from __future__ import annotations

import contextlib
import io
import threading
import time
import unittest

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_prompt_freeze import _Base, _RunNow


@requires_monolith
class BootPrimeRetryTests(_Base):

    def setUp(self):
        super().setUp()
        bc = self.bc
        self.slept: list = []
        self._p(bc.time, "sleep", side_effect=self.slept.append)
        self._p(bc.threading, "Thread", _RunNow)
        self._p(bc, "LOCAL_PREFIX_REPRIME", True)
        self._p(bc, "LOCAL_REPRIME_AT_BOOT_S", 20.0)
        self._p(bc, "_last_owner_turn_at", [0.0])
        bc._reprime_running[0] = False
        bc._reprime_again[0] = False

    def _boot(self, outcomes):
        once = self._p(self.bc, "_reprime_once", side_effect=list(outcomes))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertTrue(self.bc._start_boot_reprime())
        return once, buf.getvalue()

    def test_a_skipped_boot_prime_retries_until_primed(self):
        once, log = self._boot(["utterance", "utterance", "primed"])
        self.assertEqual(
            once.call_count, 3,
            "the boot prime was skipped once and never tried again - the "
            "first turn after the restart re-read the whole prompt")
        self.assertEqual(self.slept[0], 20.0)
        self.assertEqual(self.slept.count(self.bc._BOOT_REPRIME_RETRY_S), 2)
        self.assertEqual(self.bc._BOOT_REPRIME_RETRY_S, 30.0)
        self.assertIn("retry", log)

    def test_a_permanent_reason_stops_the_retries(self):
        for reason in ("disabled", "local-off", "route", "layout"):
            with self.subTest(reason=reason):
                self.slept.clear()
                once, _ = self._boot([reason, "primed"])
                self.assertEqual(once.call_count, 1)

    def test_the_owner_speaking_first_stops_the_retries(self):
        bc = self.bc

        def _once():
            # He spoke while the boot prime was being held off.
            bc._last_owner_turn_at[0] = time.monotonic()
            return "utterance"
        once = self._p(bc, "_reprime_once", side_effect=_once)
        with contextlib.redirect_stdout(io.StringIO()):
            bc._start_boot_reprime()
        self.assertEqual(once.call_count, 1)

    def test_retries_are_bounded(self):
        once, _ = self._boot(["utterance"] * 500)
        self.assertEqual(once.call_count, self.bc._BOOT_REPRIME_MAX_TRIES)


@requires_monolith
class MusicCaptureTests(_Base):
    """A capture while PC audio plays, after captures that never became a
    turn, is music: the re-prime must not wait for it."""

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "LOCAL_PREFIX_REPRIME", True)
        self._p(bc, "LOCAL_LLM_FALLBACK", True)
        self._p(bc, "_chat_takes_local_branch", return_value=True)
        self._p(bc, "_realtime_session", [None])
        self._p(bc, "MEDIA_VOICE_GATE_PEAK", 0.01)
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = False
        streak = getattr(bc, "_music_capture_streak", None)
        if streak is not None:
            self._p(bc, "_music_capture_streak", [0])
            self._p(bc, "_music_capture_last_at", [0.0])

    def _dropped_capture(self, peak):
        """One capture that tripped, was measured, and was dropped."""
        bc = self.bc
        done = threading.Event()
        done.set()
        bc._utterance_in_progress[0] = True
        bc._media_probe[0] = {"audio_id": 1, "peak": peak, "done": done,
                              "thread": None}
        bc._note_turn_boundary()

    def test_sustained_music_captures_do_not_hold_the_prime(self):
        bc = self.bc
        self._dropped_capture(0.30)
        self._dropped_capture(0.28)
        bc._utterance_in_progress[0] = True      # the next music capture
        with contextlib.redirect_stdout(io.StringIO()):
            reason = bc._reprime_skip_reason()
        self.assertNotEqual(
            reason, "utterance",
            "music on the PC kept the desk mic tripping and the boot prime "
            "was skipped as if the owner were mid-sentence")

    def test_a_capture_with_no_pc_audio_still_holds_it(self):
        bc = self.bc
        self._dropped_capture(0.0)
        self._dropped_capture(0.0)
        bc._utterance_in_progress[0] = True
        self.assertEqual(bc._reprime_skip_reason(), "utterance")

    def test_one_capture_over_audio_is_not_yet_sustained(self):
        bc = self.bc
        self._dropped_capture(0.30)
        bc._utterance_in_progress[0] = True
        self.assertEqual(bc._reprime_skip_reason(), "utterance")

    def test_an_owner_turn_resets_the_streak(self):
        bc = self.bc
        self._dropped_capture(0.30)
        self._dropped_capture(0.30)
        bc._note_owner_turn()
        bc._note_turn_boundary()
        bc._utterance_in_progress[0] = True
        self.assertEqual(bc._reprime_skip_reason(), "utterance")

    def test_an_old_streak_expires(self):
        bc = self.bc
        self._dropped_capture(0.30)
        self._dropped_capture(0.30)
        bc._music_capture_last_at[0] -= 3600.0
        bc._utterance_in_progress[0] = True
        self.assertEqual(bc._reprime_skip_reason(), "utterance")


@requires_monolith
class ReprimeBeforeFirstTurnTests(_Base):

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "LOCAL_PREFIX_REPRIME", True)
        self._p(bc, "LOCAL_LLM_FALLBACK", True)
        self._p(bc, "LOCAL_REPRIME_AFTER_BACKGROUND_WINDOW_S", 600.0)
        self._p(bc, "_chat_takes_local_branch", return_value=True)
        self._p(bc, "_last_owner_turn_at", [0.0])
        self.sched = self._p(bc, "_schedule_local_reprime", return_value=True)

    def _after(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.bc._reprime_after_background("ambient_extract")

    def test_background_eviction_before_the_first_turn_reprimes(self):
        # 16 minutes after a restart, no owner turn yet (the 22:04 case).
        self._p(self.bc, "_main_loop_started_at",
                [time.monotonic() - 16 * 60], create=True)
        self.assertTrue(self._after(),
                        "no re-prime before the owner's first turn: his "
                        "first turn after the restart started cold")
        self.sched.assert_called_once_with()

    def test_long_idle_after_boot_with_no_turn_stops(self):
        self._p(self.bc, "_main_loop_started_at",
                [time.monotonic() - 5 * 3600], create=True)
        self.assertFalse(self._after())
        self.sched.assert_not_called()

    def test_after_a_turn_the_normal_window_applies(self):
        bc = self.bc
        self._p(bc, "_main_loop_started_at", [time.monotonic() - 60],
                create=True)
        bc._last_owner_turn_at[0] = time.monotonic() - 1200.0
        self.assertFalse(self._after())
        bc._last_owner_turn_at[0] = time.monotonic() - 30.0
        self.assertTrue(self._after())


@requires_monolith
class FillerLateWiringTests(_Base):

    def test_the_live_filler_skips_a_late_first_line(self):
        bc = self.bc
        late = getattr(bc._processing_filler, "_first_late_s", None)
        self.assertIsNotNone(
            late, "the live filler has no lateness cap: stage 1 retried for "
                  "3 s and played over the answer (+3.6 s at 22:04:11)")
        self.assertLessEqual(late, 1.0)
        self.assertEqual(late, bc._FILLER_FIRST_LATE_S)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
