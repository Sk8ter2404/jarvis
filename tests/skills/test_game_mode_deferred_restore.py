"""Regression tests for game mode's DEFERRED brain restore — the debt that
nothing used to pay.

THE DEFECT THESE PIN (reproduced 2026-09-04 against the then-live
skills/game_mode.py with this suite's own _Base harness):

    _exit() may not re-warm the big brain while a game is still running — that
    is a ~15 GB cold load into a card the game already owns, and it is the
    crash the whole feature exists to prevent. So it deferred the restore. But
    the deferred restore was parked in ``_st.applied["__brain__"]`` and

      * every _exit() early-returns on ``if not _st.active``, so no later exit
        could reach it, and decide() only ever returns "exit" while active;
      * _restore_luxuries() ends with ``applied.clear()``;
      * the next _enter() wrote ``_st.applied["__brain__"] = ... if downshifted
        else None`` UNCONDITIONALLY — and on a re-entry the brain is ALREADY
        small, so _pick_game_brain() returns None ("already on gemma4:12b"),
        `downshifted` is False, and that line overwrote the only route back to
        the big brain with None.

    Measured end state, three separate ways in: _RESOLVED_LOCAL_LLM_MODEL[0],
    LOCAL_LLM_MODEL and LOCAL_VISION_MODEL all stuck on the small tag for the
    rest of the process, while game_mode_off answered "Game mode isn't engaged,
    sir." and game_mode_status answered "Game mode is not engaged, sir ... I'm
    on the gemma4:12b brain." There was no game-mode way back — only an
    explicit model_picker set_model (which the skill's own docstring notes
    ALWAYS persists) or a JARVIS restart. The build report's "he must not have
    to undo it manually afterwards" was exactly inverted: the one action he
    would use to undo it, "full power", is what made it permanent.

THE THREE WAYS IN, one test each:
  1. OWNER-DRIVEN, the most likely: he says "full power" mid-match.
  2. FULLY AUTOMATIC, no owner error: quit to desktop and relaunch inside the
     45 s GAME_MODE_EXIT_GRACE_SECONDS.
  3. The 24 h absolute ceiling firing while the game is still up.

Plus the counter-tests that keep the fix from over-correcting: the debt must
STILL not be paid while a game is running, or we have simply re-armed the
15 GB spike on a different trigger.

EVERY TEST ASSERTS THE OUTCOME — where the three brain cells actually point —
BEFORE it looks at any bookkeeping field, so a rename cannot make one of these
go red for a reason that is not the defect.

VERIFIED ABLE TO FAIL, which is the only thing that makes a green run mean
anything here. Run against a mutant copy of the skill with the fix reverted to
the pre-fix code (_enter parking the closure in `applied` unconditionally,
_exit re-parking it there, no drain in _tick, a bare _exit in game_mode_off):
6 of these 8 fail — 4 on ``'gemma4:12b' != 'gemma4:26b-a4b-it-qat'`` for the
resolver cache, 1 on nothing being armed to pay the debt, 1 on the reply
"Back to normal power, sir. I'm on gemma4:12b again." The 2 real counter-tests
pass on BOTH copies, which is what a counter-test is for.

Nothing here touches a real process, GPU, model or the disk — it reuses
test_game_mode._Base, which pins a fake monolith in sys.modules and stubs every
outward-facing primitive. Separate file because skills/game_mode.py and its
main suite are under concurrent edit; stdlib unittest only (no pytest).
"""
from __future__ import annotations

import unittest

from tests.skills.test_game_mode import BIG, GAME, SMALL, _Base

PID = 47036
PID2 = 47037        # the relaunched client gets a NEW pid


class _Deferred(_Base):
    """Shared setup: engage on a running game, then hand the test control of
    what _find_game_pid()/_foreground_pid() report from that point on."""

    def _engaged(self, **load_kw):
        mod, actions = self._load(**load_kw)
        self._gaming(mod, PID)
        mod._enter(PID, GAME)
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], SMALL,
                         "precondition: the downshift must have taken")
        return mod, actions

    def _gaming(self, mod, pid):
        mod._find_game_pid = lambda: (pid, GAME)
        mod._foreground_pid = lambda: pid
        self.alive_pids = {pid}

    def _game_closed(self, mod):
        mod._find_game_pid = lambda: (None, None)
        mod._foreground_pid = lambda: None
        self.alive_pids = set()

    def _assert_big_brain_back(self, where):
        """THE outcome. The resolver cache goes first because it is the cell
        that decides every real call: _call_local_llm() re-reads
        _get_local_llm_model() -> _RESOLVED_LOCAL_LLM_MODEL[0] on EVERY turn."""
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], BIG,
                         f"{where}: the resolver cache — which _call_local_llm "
                         f"re-reads on EVERY turn — is still on {SMALL}, so he "
                         f"is permanently downshifted")
        for attr in ("LOCAL_LLM_MODEL", "LOCAL_VISION_MODEL"):
            self.assertEqual(getattr(self.bc, attr), BIG,
                             f"{where}: {attr} left downshifted")

    def _assert_still_small(self, where):
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], SMALL,
                         f"{where}: re-warmed the big brain with a game still "
                         f"running — that IS the ~15 GB spike")


class TestDeferredBrainRestoreIsAlwaysPaid(_Deferred):
    def test_full_power_mid_match_is_not_a_permanent_downshift(self):
        """PATH 1, the one he is most likely to take. 'Full power' while the
        game is up must not be the thing that makes the downshift permanent."""
        mod, actions = self._engaged()

        out = actions["game_mode_off"]("")
        self._assert_still_small("right after 'full power' mid-match")
        self.assertFalse(mod._st.active)
        self.assertNotIn(BIG.lower(), out.lower(),
                         f"claimed the big brain was back while still on "
                         f"{SMALL}: {out!r}")

        # He keeps playing; nothing changes. Then Fortnite closes.
        self._game_closed(mod)
        self.assertEqual(mod._tick(), "none")
        self._assert_big_brain_back("after the game closed")
        self.assertIsNone(mod._st.pending_brain, "a debt is still recorded")

    def test_a_relaunch_inside_the_grace_does_not_destroy_the_debt(self):
        """PATH 2, fully automatic — no owner error at all. He quits to the
        desktop and relaunches inside GAME_MODE_EXIT_GRACE_SECONDS. The re-entry
        resolves its own 'before' to the SMALL tag it is already sitting on, so
        an unconditional write of the restore payload replaces the one pointing
        at the big brain with one pointing at the small one — destroying the
        restore outright while looking like bookkeeping."""
        mod, _ = self._engaged()

        # Exit with the game still up -> the restore is deferred, not run.
        mod._exit("absolute ceiling reached")
        self._assert_still_small("after a deferred exit")

        # The relaunched client re-enters. _pick_game_brain refuses ("already
        # on <small>"), so downshifted is False on this pass — the case that
        # used to erase the debt.
        self._gaming(mod, PID2)
        mod._enter(PID2, GAME)

        # That second session ends normally.
        self._game_closed(mod)
        out = mod._exit("game process exited")
        self._assert_big_brain_back("after the relaunched session ended")
        self.assertIn(BIG, out, f"exit reported the wrong brain: {out!r}")

        # Defence in depth: the debt must not live in the dict that
        # _restore_luxuries() clears and _enter() rebuilds from scratch.
        self.assertEqual(mod._st.applied, {},
                         "the lever dict is cleared on exit — anything the "
                         "restore needs cannot be kept in it")

    def test_the_ceiling_exit_leaves_a_debt_the_watcher_pays(self):
        """PATH 3: the absolute ceiling fires while he is still playing, and he
        closes the game before the watcher re-enters. _tick() itself has to pay
        — decide() can only return 'exit' while the mode is ACTIVE, and it is
        not."""
        mod, _ = self._engaged()
        mod._exit("absolute ceiling reached (24.0 h)")
        self.assertFalse(mod._st.active)
        self._assert_still_small("after the ceiling exit, game still up")

        self._game_closed(mod)
        self.assertEqual(mod._tick(), "none")
        self._assert_big_brain_back("after the ceiling exit")

    def test_off_is_not_a_dead_end_once_the_game_is_closed(self):
        """The stuck state used to be unreachable through the mode's own
        vocabulary: 'game mode off' answered "isn't engaged" and did nothing,
        while status cheerfully reported the small brain as normal. Saying
        'full power' with the game closed must put the big brain back even if
        no tick ever runs."""
        mod, actions = self._engaged()
        actions["game_mode_off"]("")          # deferred, game still up
        self._game_closed(mod)

        out = actions["game_mode_off"]("")    # he asks again, game now closed
        self._assert_big_brain_back("after a second 'full power'")
        self.assertIn(BIG, out, f"did not report the restored brain: {out!r}")
        self.assertIn(BIG, actions["game_mode_status"](""))

    def test_off_arms_the_ticker_that_owes_the_promise(self):
        """Deferring is only defensible if something will honour it. On the
        manual path the watcher may never have been started, and _tick() is the
        only thing that pays the debt — a promise with no mechanism behind it is
        this project's defining defect."""
        mod, actions = self._engaged()
        started = []
        mod.start_watcher = lambda: (started.append(True), True)[1]
        actions["game_mode_off"]("")
        self._assert_still_small("after 'full power' mid-match")
        self.assertTrue(started,
                        "deferred the restore without arming anything that "
                        "could ever run it")


class TestTheDebtIsNotPaidTooEarly(_Deferred):
    """The counter-tests. A fix that pays the debt while the game is still
    running has only moved the ~15 GB cold load onto a different trigger, so
    these must hold both before and after the fix."""

    def test_ticking_while_the_game_runs_never_rewarms_the_big_brain(self):
        mod, _ = self._engaged()
        mod._exit("absolute ceiling reached")
        self._gaming(mod, PID)                # still playing
        for _ in range(5):
            mod._tick()
        self._assert_still_small("after five ticks with the game still up")

    def test_off_while_the_game_runs_holds_the_brain_and_promises_it_back(self):
        """Holding the brain back is correct. Saying so is the other half: the
        old reply, "Back to normal power, sir. I'm on gemma4:12b again", read as
        a completed restore and is why nobody noticed the debt was never paid.
        The reply must name a restore that has NOT happened yet."""
        mod, actions = self._engaged()
        out = actions["game_mode_off"]("")
        self._assert_still_small("after 'full power' with the game up")
        low = out.lower()
        self.assertNotIn(BIG.lower(), low,
                         f"named the big brain in a reply that did not "
                         f"restore it: {out!r}")
        self.assertIn(SMALL.lower(), low,
                      f"held the brain back without saying what it is on: "
                      f"{out!r}")
        self.assertTrue(
            any(p in low for p in ("put it back", "when the game closes",
                                   "once the game closes", "still on",
                                   "stayed on", "staying on")),
            f"the reply does not tell him a restore is still outstanding, so "
            f"it reads as a finished job: {out!r}")

    def test_a_clean_exit_owes_nothing_and_invents_nothing(self):
        """The ordinary path must be untouched: game closes, brain restored
        inline, no debt left behind to be paid twice."""
        mod, _ = self._engaged()
        self._game_closed(mod)
        mod._exit("game process exited")
        self._assert_big_brain_back("on the ordinary exit path")
        # A second tick must find nothing to do rather than re-running a
        # restore against stale state.
        self.assertEqual(mod._tick(), "none")
        self._assert_big_brain_back("after an idle tick")


if __name__ == "__main__":
    unittest.main()
