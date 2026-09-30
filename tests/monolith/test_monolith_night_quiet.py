"""NIGHT_QUIET_ENABLED in the monolith (see tests/test_night_quiet.py for the
switch itself). ON, the shipped default, keeps the old behaviour; OFF, the
clock alone never changes how JARVIS sounds at night:

  * 5. the wake-word greeting: 22:00-04:59 prefers the 'soft' phrases at
       volume 0.85 (context_aware_greeting -> _pick_wake_variety). Off, a
       23:30 wake sounds like a 14:00 one; a tired owner still gets the soft
       greeting.
       The 01:00-04:59 "Still up, sir?" (a third wake inside 10 minutes) is a
       remark about the hour: off, that wake is greeted like a daytime one.
  * 6. the late-night remark (maybe_late_night_remark, 01:00-04:59): an extra
       spoken line before complying, purely because of the hour. Off, none.
  * 6. the system prompt's clock-only LATE HOUR rule (build_system_prompt ->
       core.prompts.base_system_prompt). Off, it is not in the prompt.

The knob is patched explicitly in every test and the clock is pinned (a fake
datetime.now(), or the window helper), so neither the gitignored
data/user_settings.json nor the real time of day can decide a result.

    python -m unittest tests.monolith.test_monolith_night_quiet
"""
from __future__ import annotations

import datetime as _dtmod
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


def _quiet(on):
    from core import config as cfg
    return mock.patch.object(cfg, "NIGHT_QUIET_ENABLED", on, create=True)


def _clock_at(hour, minute=30):
    """context_aware_greeting / _pick_wake_variety do `from datetime import
    datetime` inside the function, so pin the stdlib attribute itself."""
    class _FakeDateTime(_dtmod.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 29, hour, minute, 0)
    return mock.patch.object(_dtmod, "datetime", _FakeDateTime)


def _prefer_soft(seq):
    """random.choice stand-in: the first soft-tagged candidate, else the
    first candidate (so the test sees whether 'soft' was on offer)."""
    for text, tags in seq:
        if "soft" in tags:
            return (text, tags)
    return seq[0]


@requires_monolith
class WakeGreetingNightVolumeTests(MonolithGlobalsTestCase):

    def setUp(self):
        super().setUp()
        bc = self.bc
        saved = (list(bc._wake_history), list(bc._last_wake_date),
                 list(bc._pre_wake_silence_seconds), list(bc._last_wake_phrase))

        def _restore():
            bc._wake_history[:] = saved[0]
            bc._last_wake_date[:] = saved[1]
            bc._pre_wake_silence_seconds[:] = saved[2]
            bc._last_wake_phrase[:] = saved[3]
        self.addCleanup(_restore)
        bc._wake_history[:] = []
        bc._last_wake_date[0] = None
        bc._last_wake_phrase[0] = None

    def _greet(self, *, quiet, hour, tone=None):
        bc = self.bc
        with _quiet(quiet), _clock_at(hour), \
             mock.patch.object(bc, "detect_tone", return_value=tone), \
             mock.patch.object(bc, "_bambu_print_progress", return_value=None), \
             mock.patch.object(bc, "_user_looking_away", return_value=False), \
             mock.patch.object(bc.random, "choice", side_effect=_prefer_soft):
            return bc.context_aware_greeting(from_standby=False,
                                             wake_text="jarvis")

    def test_on_a_23_30_wake_is_soft_and_quieter(self):
        text, vol = self._greet(quiet=True, hour=23)
        soft = {t for (t, tags) in self.bc._WAKE_PHRASE_BANK if "soft" in tags}
        self.assertIn(text, soft)
        self.assertEqual(vol, 0.85)

    def test_off_a_23_30_wake_is_full_volume(self):
        _text, vol = self._greet(quiet=False, hour=23)
        self.assertEqual(vol, 1.0)

    def test_off_a_23_30_wake_matches_a_14_00_wake(self):
        night = self._greet(quiet=False, hour=23)
        # Same starting state for the daytime wake (an extra recent wake would
        # add the 'terse' preference and change the pool).
        self.bc._wake_history[:] = []
        self.bc._last_wake_phrase[0] = None
        self.bc._last_wake_date[0] = None
        day = self._greet(quiet=False, hour=14)
        self.assertEqual(night, day)

    def test_off_a_tired_owner_still_gets_the_soft_greeting(self):
        _text, vol = self._greet(quiet=False, hour=14, tone="tired")
        self.assertEqual(vol, 0.85)

    # ── "Still up, sir?" (01:00-04:59, a third wake inside 10 minutes) ────
    def _third_wake(self, *, quiet, hour):
        now = self.bc.time.time()
        self.bc._wake_history[:] = [now - 120.0, now - 60.0]
        self.bc._last_wake_phrase[0] = None
        self.bc._last_wake_date[0] = None
        return self._greet(quiet=quiet, hour=hour)

    def test_on_a_third_wake_at_02_30_asks_if_he_is_still_up(self):
        self.assertEqual(self._third_wake(quiet=True, hour=2),
                         ("Still up, sir?", 1.0))

    def test_off_a_third_wake_at_02_30_is_greeted_like_one_at_14_30(self):
        night = self._third_wake(quiet=False, hour=2)
        day = self._third_wake(quiet=False, hour=14)
        self.assertNotEqual(night[0], "Still up, sir?")
        self.assertEqual(night, day)


@requires_monolith
class SystemPromptLateHourRuleTests(MonolithGlobalsTestCase):
    """build_system_prompt() starts from core.prompts.base_system_prompt():
    the prompt's clock-only LATE HOUR rule (22:00-05:59 = quietly stressed,
    one sentence) and its 'daylight hours' return condition are there with
    night quieting on and gone with it off."""

    def _build(self, *, quiet):
        bc = self.bc
        with _quiet(quiet), \
             mock.patch.object(bc, "_load_chappie_standing_rules",
                               return_value=""), \
             mock.patch.object(bc._mcu_phrases, "render_phrasebook_block",
                               return_value="PB"):
            return bc.build_system_prompt(bc._empty_memory())

    def test_on_the_built_prompt_carries_the_late_hour_rule(self):
        prompt = self._build(quiet=True)
        self.assertIn("LATE HOUR — local time after 22:00", prompt)
        self.assertTrue(prompt.startswith(self.bc.BASE_SYSTEM_PROMPT))

    def test_off_the_built_prompt_has_no_clock_only_rule(self):
        prompt = self._build(quiet=False)
        self.assertNotIn("LATE HOUR", prompt)
        self.assertNotIn("daylight hours", prompt)
        self.assertIn("VENTING KEYWORDS", prompt)


@requires_monolith
class LateNightRemarkTests(MonolithGlobalsTestCase):

    def _remark(self, text, *, quiet, memory=None):
        bc = self.bc
        idx_cell, last_cell = [0], [0.0]
        with _quiet(quiet), \
             mock.patch.object(bc, "_in_late_night_window", return_value=True), \
             mock.patch.object(bc, "_is_late_night_suppressed", return_value=False), \
             mock.patch.object(bc, "_late_night_phrase_idx", idx_cell), \
             mock.patch.object(bc, "_late_night_last_remark", last_cell), \
             mock.patch.object(bc, "_late_night_hour_word", return_value="3"), \
             mock.patch.object(bc, "load_memory", return_value={}), \
             mock.patch.object(bc, "save_memory") as save:
            out = bc.maybe_late_night_remark(text, {} if memory is None else memory)
        return out, idx_cell[0], save

    def test_on_a_03_00_command_gets_a_remark_first(self):
        out, cursor, _save = self._remark("open the notes", quiet=True)
        self.assertTrue(out)
        self.assertEqual(cursor, 1)

    def test_off_a_03_00_command_gets_no_remark(self):
        out, cursor, _save = self._remark("open the notes", quiet=False)
        self.assertEqual(out, "")
        self.assertEqual(cursor, 0, "the phrase rotation must not advance")

    def test_on_the_suppress_phrase_is_acknowledged(self):
        out, _cursor, save = self._remark("no comments tonight", quiet=True)
        self.assertEqual(out, "As you wish, sir. Silent until morning.")
        save.assert_called_once()

    def test_off_the_suppress_phrase_is_left_to_the_normal_turn(self):
        mem = {}
        out, _cursor, save = self._remark("no comments tonight", quiet=False,
                                          memory=mem)
        self.assertEqual(out, "")
        save.assert_not_called()
        self.assertNotIn("late_night_no_comments_until", mem)


if __name__ == "__main__":
    unittest.main()
