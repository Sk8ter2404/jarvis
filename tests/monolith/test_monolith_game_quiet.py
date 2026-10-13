"""Monolith wiring for game mode's QUIET hold (2026-10-12).

While game mode is engaged JARVIS makes no unsolicited remarks and holds the
speech queue, except for the lines the owner needs even mid-match:

  * his own timers / schedules / promises and the guard-mode alert (the
    EXEMPT_SOURCES), and any line flagged urgent (a print failure);
  * everything else waits, in order, untouched on disk, and is spoken after the
    game as ONE short "Earlier, sir" recap (not "While you were away" - he
    never left), with stale moment-lines (a break nudge) simply expiring.

Every test drives the REAL _speak_pending / _presence_hold_reason /
should_be_proactive with the voice and the clock faked; the queue lives in a
temp dir. The game skill is a stand-in module exposing only the `_st.active`
flag the monolith reads. Synthetic lines only.

    python -m unittest tests.monolith.test_monolith_game_quiet
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from tests.monolith.test_monolith_presence_hold import (
    _Base, WELLNESS, PULSE, TIMER, GUARD)
from tests._monolith_harness import requires_monolith


def _game_skill(active: bool):
    mod = types.ModuleType("skill_game_mode")
    mod._st = types.SimpleNamespace(active=active)
    return mod


@requires_monolith
class GameQuietBase(_Base):
    def setUp(self):
        super().setUp()
        self.game = _game_skill(True)
        p = mock.patch.dict(sys.modules, {"skill_game_mode": self.game})
        p.start()
        self.addCleanup(p.stop)
        import core.config as cfg
        self._cfg = cfg
        if not hasattr(cfg, "GAME_MODE_QUIET"):          # pre-fix tree
            cfg.GAME_MODE_QUIET = True
            self.addCleanup(lambda: delattr(cfg, "GAME_MODE_QUIET"))
        self._p(cfg, "GAME_MODE_QUIET", True)
        self.bc._game_hold_seen[0] = False

    def end_game(self):
        self.game._st.active = False


class QuietPredicateTests(GameQuietBase):
    def test_quiet_while_engaged(self):
        self.assertTrue(self.bc._game_quiet_active())

    def test_not_quiet_when_game_mode_is_off(self):
        self.end_game()
        self.assertFalse(self.bc._game_quiet_active())

    def test_the_setting_turns_quiet_off_without_leaving_game_mode(self):
        self._p(self._cfg, "GAME_MODE_QUIET", False)
        self.assertFalse(self.bc._game_quiet_active())
        self.assertTrue(self.bc._game_mode_active())

    def test_a_missing_skill_is_not_quiet(self):
        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("skill_game_mode", None)
            self.assertFalse(self.bc._game_quiet_active())

    def test_a_broken_state_object_fails_open(self):
        self.game._st = None
        self.assertFalse(self.bc._game_quiet_active())

    def test_no_spontaneous_remarks_while_gaming(self):
        self._p(self.bc, "PROACTIVE_ENABLED", True)
        self._p(self.bc, "focus_mode_active",
                side_effect=AssertionError("the game gate must come first"))
        self.assertFalse(self.bc.should_be_proactive())


class QuietDrainTests(GameQuietBase):
    def setUp(self):
        super().setUp()
        # He is right here (so presence alone would never hold anything).
        self.input_age[0] = 3.0

    def test_status_lines_wait_and_are_not_even_claimed(self):
        self._write([self._e(WELLNESS, "wellness"), self._e(PULSE, "pulse")])
        spoke, log = self._drain()
        self.assertFalse(spoke)
        self.assertEqual(self.spoke, [])
        self.assertEqual([e["message"] for e in self._read()],
                         [WELLNESS, PULSE])
        self.assertIn("holding queued speech: gaming", log)

    def test_the_hold_works_with_the_presence_gate_unarmed(self):
        self.bc._presence_gate_armed[0] = False
        self._write([self._e(PULSE, "pulse")])
        self._drain()
        self.assertEqual(self.spoke, [])

    def test_his_timer_the_guard_alert_and_urgent_lines_still_speak(self):
        urgent = dict(self._e("Slight problem, sir - the printer has stopped.",
                              "bambu"), urgent=True)
        self._write([self._e(WELLNESS, "wellness"),
                     dict(self._e(TIMER, "timer"), dedupe_key="timer#1"),
                     self._e(GUARD, "guard"), urgent,
                     self._e(PULSE, "pulse")])
        self._drain()
        self.assertEqual(self.spoke, [
            TIMER, GUARD, "Slight problem, sir - the printer has stopped."])
        self.assertEqual([e["message"] for e in self._read()],
                         [WELLNESS, PULSE])

    def test_the_game_ending_releases_the_queue_as_one_earlier_recap(self):
        old = 20 * 60
        self._write([self._e(PULSE, "pulse", age_s=old),
                     self._e("Your H2D is at layer 40, sir.", "bambu",
                             age_s=old),
                     self._e(WELLNESS, "wellness", age_s=old)])
        self._drain()                       # held while the game runs
        self.assertEqual(self.spoke, [])
        self.end_game()
        self._drain()
        self.assertEqual(len(self.spoke), 2, self.spoke)
        recap = [m for m in self.spoke if m.startswith("Earlier, sir")]
        self.assertTrue(recap, self.spoke)
        self.assertNotIn("While you were away", " ".join(self.spoke))
        # the break nudge was about a moment that is over: not spoken at all
        self.assertNotIn(WELLNESS, self.spoke)
        self.assertEqual(self._read(), [])

    def test_fresh_lines_are_spoken_as_they_are_after_the_game(self):
        self._write([self._e(PULSE, "pulse", age_s=5.0)])
        self._drain()
        self.end_game()
        self._drain()
        self.assertEqual(self.spoke, [PULSE])

    def test_no_hold_and_no_recap_when_game_mode_never_engaged(self):
        self.end_game()
        self._write([self._e(PULSE, "pulse")])
        _, log = self._drain()
        self.assertEqual(self.spoke, [PULSE])
        self.assertNotIn("holding queued speech", log)

    def test_the_quiet_setting_off_lets_everything_through(self):
        self._p(self._cfg, "GAME_MODE_QUIET", False)
        self._write([self._e(PULSE, "pulse")])
        self._drain()
        self.assertEqual(self.spoke, [PULSE])


if __name__ == "__main__":
    unittest.main()
