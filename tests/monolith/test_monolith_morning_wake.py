"""Monolith-tier regressions for B096 (2026-10-01): the morning chain never
auto-fired under the owner's settings.

skills/morning_chain.py dispatches the day's morning briefing when
bobert_companion._last_wake_date[0] turns to today inside 06:00-12:00. That
cell was stamped only by the standby wake (context_aware_greeting) and the
tray force_wake. The owner runs with START_IN_STANDBY off and never sleeps
JARVIS, so his "Jarvis ..." mornings never stamped it: 47 session logs, 47
"[morning-chain] active" lines, zero dispatches.

Pinned here:
  * the day's first accepted owner turn (06:00 on, voice or typed, never a
    test inject, never a staging instance's smoke prompt) stamps the wake
    through the one helper, _note_wake_event,
    which also takes the pre-wake silence snapshot -- measured from his
    PREVIOUS turn, so the overnight gap reaches morning_arrival's 6-hour gate;
  * later turns the same day do not re-stamp;
  * the tray force_wake takes the same snapshot (it used to stamp the date
    alone, leaving the gate to read a stale snapshot);
  * main() stamps between "You:" and the turn mark, before the reply;
  * end to end: that stamp makes the real chain watcher dispatch, and the
    real morning_arrival silence gate passes on it.
"""
from __future__ import annotations

import datetime as _dtmod
import inspect
import time
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from tests._skill_harness import load_skill_isolated


class _LoopBreak(BaseException):
    """Ends the chain watcher's `while True` from a stubbed sleep."""


def _fake_now(hour, day=1):
    class _FakeDateTime(_dtmod.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 6, day, hour, 30, 0)
    return mock.patch.object(_dtmod, "datetime", _FakeDateTime)


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        bc = self.bc
        bc._last_wake_date[0] = None
        bc._pre_wake_silence_seconds[0] = 0.0
        # JARVIS's own overnight line a minute ago (a reminder): the silence
        # must still be measured from the owner's last turn, not from this.
        self._p(bc, "last_speech_time", time.time() - 60.0)
        # The harness imports the monolith as the STAGING instance
        # (JARVIS_STAGING=1); the owner talks to prod.
        self._p(bc, "BLUE_GREEN_ROLE", "prod")

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _owner_last_spoke(self, seconds_ago):
        self.bc._last_owner_turn_at[0] = time.monotonic() - seconds_ago


class FirstOwnerTurnWakeTests(_Base):
    def test_first_turn_of_the_day_stamps_the_wake_and_the_overnight_silence(self):
        bc = self.bc
        self._owner_last_spoke(8 * 3600.0)
        with _fake_now(7):
            self.assertTrue(bc._note_first_owner_turn_of_day())
        self.assertEqual(bc._last_wake_date[0], "2026-06-01")
        self.assertAlmostEqual(bc._pre_wake_silence_seconds[0], 8 * 3600.0,
                               delta=60.0)

    def test_later_turns_the_same_day_do_not_restamp(self):
        bc = self.bc
        self._owner_last_spoke(8 * 3600.0)
        with _fake_now(7):
            bc._note_first_owner_turn_of_day()
            bc._note_owner_turn()                  # that turn, accepted
            self._owner_last_spoke(30.0)
            self.assertFalse(bc._note_first_owner_turn_of_day())
        # The overnight snapshot survives the day's second turn.
        self.assertAlmostEqual(bc._pre_wake_silence_seconds[0], 8 * 3600.0,
                               delta=60.0)

    def test_the_next_day_stamps_again(self):
        bc = self.bc
        bc._last_wake_date[0] = "2026-05-31"
        self._owner_last_spoke(9 * 3600.0)
        with _fake_now(9):
            self.assertTrue(bc._note_first_owner_turn_of_day())
        self.assertEqual(bc._last_wake_date[0], "2026-06-01")

    def test_a_turn_before_six_is_last_nights_tail(self):
        # Stamping a 02:30 turn would have the chain brief an empty room at
        # 06:00 while he sleeps (arrival declines on the short gap, then the
        # handoff fallback speaks).
        bc = self.bc
        self._owner_last_spoke(600.0)
        with _fake_now(2):
            self.assertFalse(bc._note_first_owner_turn_of_day())
        self.assertIsNone(bc._last_wake_date[0])
        self.assertEqual(bc._pre_wake_silence_seconds[0], 0.0)

    def test_a_test_inject_never_wakes_the_day(self):
        # driver.py / say_to_jarvis at 07:00 is a Claude Code live check, not
        # the owner getting up.
        bc = self.bc
        self._owner_last_spoke(8 * 3600.0)
        with _fake_now(7):
            self.assertFalse(bc._note_first_owner_turn_of_day(test_inject=True))
        self.assertIsNone(bc._last_wake_date[0])

    def test_a_staging_instance_turn_never_wakes_the_day(self):
        # The green candidate's turns are the upgrade gate's untagged smoke
        # prompts ("are you the new one" at 07:00), and it shares the morning
        # state files: stamping one would let green's chain run the handoff's
        # predictive setup and mark the day briefed for prod.
        bc = self.bc
        self._owner_last_spoke(8 * 3600.0)
        self._p(bc, "BLUE_GREEN_ROLE", "staging")
        with _fake_now(7):
            self.assertFalse(bc._note_first_owner_turn_of_day())
        self.assertIsNone(bc._last_wake_date[0])
        self.assertEqual(bc._pre_wake_silence_seconds[0], 0.0)


class WakeEventBookkeepingTests(_Base):
    def test_force_wake_snapshots_the_pre_wake_silence(self):
        # The tray wake stamped the date alone, so arrival's gate read whatever
        # snapshot an older wake had left (0.0 on a fresh process).
        bc = self.bc
        self._owner_last_spoke(7 * 3600.0)
        bc._sleep_mode[0] = True
        bc._standby_mode[0] = True
        with mock.patch.object(bc, "_write_hud_state"), \
             mock.patch.object(bc, "_speak"), \
             mock.patch.object(bc.os.path, "exists", return_value=False), \
             _fake_now(7):
            bc._dispatch_tray_command("force_wake", {})
        self.assertEqual(bc._last_wake_date[0], "2026-06-01")
        self.assertAlmostEqual(bc._pre_wake_silence_seconds[0], 7 * 3600.0,
                               delta=60.0)

    def test_greeting_reads_first_of_day_before_the_stamp(self):
        # context_aware_greeting now stamps through _note_wake_event; it must
        # still read "first wake of the day" BEFORE that stamp.
        bc = self.bc
        self._owner_last_spoke(8 * 3600.0)
        with mock.patch.object(bc, "_bambu_print_progress", return_value=None), \
             mock.patch.object(bc, "_user_looking_away", return_value=False), \
             mock.patch.object(bc, "_pick_wake_variety",
                               return_value=("Sir?", 1.0)), \
             _fake_now(7):
            first, _ = bc.context_aware_greeting(from_standby=True)
            second, _ = bc.context_aware_greeting(from_standby=True)
        self.assertEqual(first, "Good morning, sir.")
        self.assertEqual(second, "Sir?")
        self.assertEqual(bc._last_wake_date[0], "2026-06-01")

    def test_a_morning_turn_before_the_standby_wake_owns_the_good_morning(self):
        # He spoke at 07:00 (the day's wake); a standby wake at 07:30 is not
        # the first of the day, so no second "Good morning, sir."
        bc = self.bc
        self._owner_last_spoke(8 * 3600.0)
        with mock.patch.object(bc, "_bambu_print_progress", return_value=None), \
             mock.patch.object(bc, "_user_looking_away", return_value=False), \
             mock.patch.object(bc, "_pick_wake_variety",
                               return_value=("Sir?", 1.0)), \
             _fake_now(7):
            bc._note_first_owner_turn_of_day()
            text, _ = bc.context_aware_greeting(from_standby=True)
        self.assertEqual(text, "Sir?")


class MainLoopWiringTests(_Base):
    """main() cannot run in a test; pin the call site at source level."""

    def test_first_turn_stamp_sits_between_you_and_the_turn_mark(self):
        src = inspect.getsource(self.bc.main)
        you = src.index('_tt("mark", "you")')
        stamp = src.index("_note_first_owner_turn_of_day(", you)
        turn = src.index("_note_owner_turn()", you)
        reply = src.index("reply = _run_llm_dispatch(text", you)
        # After "You:" (an accepted turn), before the turn mark (the silence
        # snapshot reads his PREVIOUS turn) and before JARVIS replies.
        self.assertLess(you, stamp)
        self.assertLess(stamp, turn)
        self.assertLess(turn, reply)
        call = src[stamp:turn]
        self.assertIn('_last_inject_source[0] == "test"', call)
        self.assertIn("except Exception", call)   # never kills the turn


class ChainEndToEndTests(_Base):
    """The real stamp, the real chain watcher, the real arrival gate."""

    def test_first_turn_makes_the_chain_dispatch_and_arrival_pass_its_gate(self):
        bc = self.bc
        self._owner_last_spoke(8 * 3600.0)
        with _fake_now(7):
            bc._note_first_owner_turn_of_day()
        bc._turn_in_progress[0] = False        # that turn has been answered
        bc._sleep_mode[0] = False              # ...and JARVIS is awake

        chain, _ = load_skill_isolated("morning_chain")
        invoked = []
        with mock.patch.object(chain.importlib, "import_module", return_value=bc), \
             mock.patch.object(chain.time, "strftime", return_value="2026-06-01"), \
             mock.patch.object(chain.time, "localtime",
                               return_value=types.SimpleNamespace(tm_hour=7)), \
             mock.patch.object(chain, "_choose_skill_for_today",
                               return_value="arrival"), \
             mock.patch.object(chain, "_morning_already_covered_today",
                               return_value=False), \
             mock.patch.object(chain, "_invoke_skill",
                               side_effect=lambda n, r: invoked.append(n) or True), \
             mock.patch.object(chain.time, "sleep", side_effect=_LoopBreak):
            with self.assertRaises(_LoopBreak):
                chain._watch_for_first_wake()
        self.assertEqual(invoked, ["arrival"])

        # ...and arrival's 6-hour gate reads the overnight gap from the
        # snapshot, not the ~0 s since JARVIS last spoke.
        arrival, _ = load_skill_isolated("morning_arrival")
        with mock.patch.object(arrival.importlib, "import_module",
                               return_value=bc):
            hours = arrival._silence_hours_since_last_speech()
        self.assertGreaterEqual(hours, arrival.MIN_SILENCE_HOURS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
