"""The DEFERRED BRAIN RESTORE — skills/game_mode.py's one un-payable promise.

WHY THIS FILE EXISTS (regression, 2026-09-04)

`_exit()` deliberately refuses to re-warm the 15 GB brain while a game is still
running: repointing mid-match means his next utterance cold-loads ~15 GB into a
card measured at 23269/24576 MiB, which IS the crash the mode prevents. That
refusal is correct. What was not correct is that the refusal was permanent:

  * the restore was parked in `_st.applied["__brain__"]` and `_st.active` was
    set False — and `decide()` only ever returns "exit" while `st.active` is
    True, so `_exit()` could never run again and that closure was never called;
  * when the game finally exited the watcher simply idled — nothing walks
    `_st.applied` while the mode is off, so the brain stayed on the small tag;
  * the next `_enter()` then OVERWROTE `_st.applied["__brain__"]` with None,
    because `_pick_game_brain()` answers "already on <small>" so `downshifted`
    is False — destroying the restore outright;
  * and the owner was told "Back to normal power, sir. I'm on gemma4:12b
    again" — a sentence that reports a restoration that had not happened, over
    a brain that was still downshifted. That is this repo's defining defect
    (claiming a result that was never established) said out loud, and it is
    what stopped anyone noticing the debt was never paid.

Net effect on the live box: he says "full power", is told he has it, and stays
on the 12B for the rest of the JARVIS process's life — the one thing the module
header promises he will never have to undo by hand is exactly what he is left
holding.

These tests drive the real `_enter` / `_exit` / `_tick` and assert on the
monolith cells the skill actually repoints. Nothing here touches a real
process, a real GPU, a real model, or the disk: `_find_game_pid` is stubbed in
EVERY test because the real one would find the owner's actually-running
Fortnite and answer with live state.

stdlib unittest + unittest.mock only (pytest is not installed).
"""
from __future__ import annotations

import unittest

from tests.skills.test_game_mode import BIG, GAME, SMALL, _Base

PID = 47036


class _Debt(_Base):
    """Loads the skill with the shared fake monolith, then closes off every
    path to the live machine that these tests exercise."""

    def _loaded(self, **kw):
        mod, actions = self._load(**kw)
        # The watcher must never be armed for real here: game_mode_off arms it
        # when it defers, and a live poll thread would enumerate his processes.
        self.started = []
        mod.start_watcher = lambda: (self.started.append(True), False)[1]
        mod._foreground_pid = lambda: None
        self._set_game(PID)
        return mod, actions

    def _set_game(self, pid):
        """Point the skill's game detector at a fake result."""
        self.mod._find_game_pid = (lambda: (pid, GAME)) if pid else (lambda: (None, None))

    def _brain(self):
        return (self.bc._RESOLVED_LOCAL_LLM_MODEL[0],
                self.bc.LOCAL_LLM_MODEL,
                self.bc.LOCAL_VISION_MODEL)


class TestDeferredRestoreIsActuallyPaid(_Debt):
    def test_full_power_mid_game_holds_the_brain_but_keeps_the_debt(self):
        """The refusal to re-warm is correct. Losing the restore is not."""
        mod, actions = self._loaded()
        mod._enter(PID, GAME)
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], SMALL)

        actions["game_mode_off"]("")            # he says "full power", still in the match

        self.assertEqual(self._brain(), (SMALL, SMALL, SMALL),
                         "re-warmed the 15 GB brain while he was still playing")
        self.assertIsNotNone(
            mod._st.pending_brain,
            "the brain restore was dropped: nothing walks _st.applied once "
            "_st.active is False, so it could never run again")
        self.assertEqual(mod._st.pending_brain.get("resolved"), BIG)

    def test_the_debt_is_paid_the_moment_the_game_closes(self):
        """This is the assertion the defect fails. After 'full power' mid-game
        the watcher just idled forever and he stayed on the 12B."""
        mod, actions = self._loaded()
        mod._enter(PID, GAME)
        actions["game_mode_off"]("")

        self._set_game(None)                    # Fortnite closes
        mod._tick()

        self.assertEqual(self._brain(), (BIG, BIG, BIG),
                         "the game closed and the big brain never came back — "
                         "he has to restart JARVIS by hand, which is the one "
                         "thing this module promises he never has to do")
        self.assertIsNone(mod._st.pending_brain)

    def test_full_power_must_not_claim_a_restore_that_did_not_happen(self):
        """'Back to normal power, sir. I'm on gemma4:12b again' over a brain
        that is still downshifted is the report-success-you-never-verified
        defect in one sentence."""
        mod, actions = self._loaded()
        mod._enter(PID, GAME)
        out = actions["game_mode_off"]("")

        self.assertNotIn("I'm on %s again" % SMALL, out)
        self.assertIn(SMALL, out)
        self.assertIn("still running", out)
        self.assertTrue(
            "put it back" in out or "back the moment" in out,
            f"the held restore was never promised back to him: {out!r}")

    def test_a_deferred_off_arms_something_that_can_pay_the_debt(self):
        """Only _tick() can pay it, so a deferral with no ticker behind it is a
        promise with no mechanism — the failure this whole file is written
        against. game_mode_on already arms the watcher for the same reason."""
        mod, actions = self._loaded()
        mod._enter(PID, GAME)
        actions["game_mode_off"]("")
        self.assertTrue(self.started,
                        "deferred the restore without arming the watcher that "
                        "is the only thing able to honour it")

    def test_re_entering_after_a_deferred_off_does_not_destroy_the_debt(self):
        """The nastiest half: the next _enter() resolved prev_brain to the
        SMALL tag it was already on and overwrote the payload, so even a later
        clean exit restored nothing."""
        mod, actions = self._loaded()
        mod._enter(PID, GAME)
        actions["game_mode_off"]("")

        mod._st.inhibit_pid = None               # a new session of the game
        mod._enter(PID, GAME)                    # already on SMALL -> no downshift
        self.assertEqual(mod._st.pending_brain.get("resolved"), BIG,
                         "re-entry overwrote the outstanding restore with the "
                         "small tag it was already sitting on")

        self._set_game(None)
        mod._exit("game process exited")
        self.assertEqual(self._brain(), (BIG, BIG, BIG))

    def test_deadman_exit_while_gaming_still_ends_up_restored(self):
        """Same hole, reached without the owner saying anything: an L3 deadman
        exit with the game still up also defers."""
        mod, _ = self._loaded()
        mod._enter(PID, GAME)
        mod._exit("deadman ceiling reached")
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], SMALL)

        self._set_game(None)
        mod._tick()
        self.assertEqual(self._brain(), (BIG, BIG, BIG))

    def test_the_debt_is_paid_once_and_then_left_alone(self):
        """A payer that re-fires would fight a NEW session of game mode and
        drag the big brain back mid-match."""
        mod, actions = self._loaded()
        mod._enter(PID, GAME)
        actions["game_mode_off"]("")
        self._set_game(None)
        mod._tick()
        self.assertIsNone(mod._st.pending_brain)

        mod._tick()                               # idle poll, nothing owed
        self.assertEqual(self._brain(), (BIG, BIG, BIG))

        self._set_game(PID)                       # he starts playing again
        mod._enter(PID, GAME)
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], SMALL)
        mod._tick()                               # game still running
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], SMALL,
                         "the debt payer fired against a live game session")

    def test_full_power_after_the_game_closed_pays_the_debt_by_hand(self):
        """His hand-operated way out if the watcher never ran: say it again
        once the game is gone."""
        mod, actions = self._loaded()
        mod._enter(PID, GAME)
        actions["game_mode_off"]("")              # deferred, mode now inactive

        self._set_game(None)
        out = actions["game_mode_off"]("")        # "full power" once more
        self.assertEqual(self._brain(), (BIG, BIG, BIG))
        self.assertIn(BIG, out)
        self.assertIsNone(mod._st.pending_brain)

    def test_no_debt_is_recorded_when_nothing_was_downshifted(self):
        """Entry that never repointed the brain owes nothing, and a later exit
        must not invent a restore to a tag it never left."""
        mod, _ = self._loaded(settings={"GAME_MODE_BRAIN": "nonexistent:70b"})
        mod._enter(PID, GAME)
        self.assertIsNone(mod._st.pending_brain)
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], BIG)
        mod._exit("game process exited")
        self.assertIsNone(mod._st.pending_brain)
        self.assertEqual(self._brain(), (BIG, BIG, BIG))


if __name__ == "__main__":       # pragma: no cover
    unittest.main(verbosity=2)
