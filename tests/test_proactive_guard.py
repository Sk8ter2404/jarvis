"""core/proactive_guard — the proactive-remark text and pacing gates.

Live evidence (session_2026-09-29_22-06-02.log, session_2026-09-30_09-48-41.log,
grep "[proactive]"): the spoken remark was the phrasebook line "You seem rather
determined this evening, sir." at 22:43:17, 22:49:21, 23:17:02, 23:36:11,
00:02:42, 07:54:17, 08:07:45, 08:14:14 and — after a restart — at 09:59:54,
ten in the morning. The remarks came every 3-6 minutes whether or not anyone
answered (09:56:22 then 09:59:54).

Light tier: stdlib only (the persona pool imports mcu_phrases and core.prompts,
both pure). Synthetic remarks apart from the quoted live line.

    python -m unittest tests.test_proactive_guard
"""
from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import proactive_guard as pg  # noqa: E402

LIVE_LINE = "You seem rather determined this evening, sir."


class TimeOfDayTests(unittest.TestCase):
    """A1: a remark may not state a time of day the local clock contradicts."""

    def test_the_live_line_at_ten_in_the_morning_is_dropped(self):
        # 09:59:54 local on 2026-09-30.
        ok, why = pg.check_remark(LIVE_LINE, hour=9)
        self.assertFalse(ok)
        self.assertIn("this evening", why)
        self.assertEqual(pg.time_of_day_conflict(LIVE_LINE, 9), "this evening")
        self.assertEqual(pg.time_of_day_conflict(LIVE_LINE, 10), "this evening")

    def test_each_phrase_against_the_clock(self):
        cases = [
            ("Busy this morning, sir?", 9, ""),
            ("Busy this morning, sir?", 20, "this morning"),
            ("A quiet afternoon so far, sir.", 14, ""),
            ("A quiet afternoon so far, this afternoon.", 23, "this afternoon"),
            ("Working late tonight, sir?", 22, ""),
            ("Working late tonight, sir?", 2, ""),
            ("Working late tonight, sir?", 10, "tonight"),
            ("Another late night, sir?", 1, ""),
            ("Another late-night session, sir?", 15, "late night"),
            ("Still at it at this hour, sir?", 23, ""),
            ("Still at it at this hour, sir?", 11, "at this hour"),
            ("Good morning, sir.", 21, "good morning"),
            ("You are up rather late, sir.", 14, ""),   # no phrase named
        ]
        for text, hour, want in cases:
            with self.subTest(text=text, hour=hour):
                self.assertEqual(pg.time_of_day_conflict(text, hour), want)

    def test_tags_and_curly_apostrophes_do_not_hide_a_phrase(self):
        text = "[intent:observation] [wry] You’re determined this evening, sir."
        self.assertEqual(pg.time_of_day_conflict(text, 10), "this evening")

    def test_day_part_labels(self):
        self.assertEqual(pg.day_part(9), "morning")
        self.assertEqual(pg.day_part(13), "afternoon")
        self.assertEqual(pg.day_part(19), "evening")
        self.assertEqual(pg.day_part(23), "night")
        self.assertEqual(pg.day_part(3), "night")


class PersonaCopyTests(unittest.TestCase):
    """A1: a remark may not be a verbatim / near-verbatim persona example."""

    def test_the_live_line_is_a_copy_even_when_the_hour_fits(self):
        # 22:43:17 — evening, so the clock check passes; the copy check must
        # still drop it (this is how the SAME line played eight times).
        ok, why = pg.check_remark(LIVE_LINE, hour=22)
        self.assertFalse(ok)
        self.assertIn("copies the persona example", why)

    def test_near_copies_are_caught(self):
        for text in ("You seem rather determined, sir.",
                     "[intent:observation] You seem rather determined this morning, sir.",
                     "You seem rather determined tonight.",
                     "I couldn't help but notice, sir..."):
            with self.subTest(text=text):
                self.assertTrue(pg.persona_copy(text), text)

    def test_an_original_remark_passes(self):
        for text in ("I've noticed you've been quite focused for a while, sir.",
                     "The first webcam, incidentally, watched a coffee pot.",
                     "Shall I queue something calmer to work to, sir?"):
            with self.subTest(text=text):
                self.assertEqual(pg.persona_copy(text), "")
                self.assertEqual(pg.check_remark(text, hour=10), (True, ""))

    def test_the_pool_is_built_from_every_source(self):
        pool = pg.persona_pool()
        import mcu_phrases
        from core import persona
        for bucket in mcu_phrases.MCU_PHRASES.values():
            for line in bucket:
                self.assertIn(line, pool)
        for line in persona.JARVIS_SIGNATURE_PHRASES:
            self.assertIn(line, pool)
        # a quoted example that lives ONLY in the base prompt
        self.assertIn("Will that be all, sir? Or shall I add it to the running "
                      "list?", pool)
        # the retired live offender stays caught
        self.assertIn(LIVE_LINE, pool)


class PersonaPoolIsTimeNeutralTests(unittest.TestCase):
    """A1: the example lines the model is SHOWN must not state a time of day,
    or it copies the time along with the line. Scans the live sources (not
    RETIRED_EXAMPLES). The one allowed example is explicitly conditioned in
    the prompt ("only ever after midnight")."""

    CONDITIONED = {"It is well past midnight, sir."}

    def _sources(self):
        import mcu_phrases
        from core import persona, prompts
        lines = [line for bucket in mcu_phrases.MCU_PHRASES.values()
                 for line in bucket]
        lines += list(persona.JARVIS_SIGNATURE_PHRASES)
        lines += pg.quoted_examples(prompts.BASE_SYSTEM_PROMPT)
        return lines

    def test_no_example_names_a_time_of_day(self):
        offenders = [(line, pg.time_phrases(line)) for line in self._sources()
                     if pg.time_phrases(line) and line not in self.CONDITIONED]
        self.assertEqual(offenders, [])

    def test_the_conditioned_example_is_conditioned_in_the_prompt(self):
        from core import prompts
        text = prompts.BASE_SYSTEM_PROMPT
        for line in self.CONDITIONED:
            at = text.index(line)
            self.assertIn("only ever after midnight", text[at:at + 120])

    def test_the_extractor_sees_the_persona_examples(self):
        # Blindness floor: an extractor that finds nothing would pass the
        # neutrality test vacuously.
        from core import prompts
        quoted = pg.quoted_examples(prompts.BASE_SYSTEM_PROMPT)
        self.assertGreater(len(quoted), 40)
        self.assertIn("You seem rather determined, sir.", quoted)
        self.assertIn("I'm afraid that's inadvisable, sir.", quoted)


class RepeatRingTests(unittest.TestCase):
    """A2: a remark may not repeat or near-repeat a recent one."""

    FIRST = "I've noticed you've been quite focused for a while, sir."

    def test_an_exact_repeat_is_dropped(self):
        ok, why = pg.check_remark(self.FIRST, hour=22, recent=[self.FIRST])
        self.assertFalse(ok)
        self.assertEqual(why, "repeats a recent remark")

    def test_near_repeats_are_dropped(self):
        for text in ("You've been quite focused for a while, sir.",
                     "I've noticed you've been rather focused for a while, sir.",
                     "[intent:observation] I've noticed you've been quite "
                     "focused for a while this evening, sir."):
            with self.subTest(text=text):
                self.assertTrue(pg.repeats_recent(text, [self.FIRST]))

    def test_a_different_remark_with_shared_words_passes(self):
        recent = ["Shall I put some music on, sir?"]
        self.assertEqual(pg.repeats_recent("Shall I put the kettle on, sir?",
                                           recent), "")
        self.assertEqual(pg.repeats_recent(
            "The kettle, sir, has been cold since noon.", [self.FIRST]), "")

    def test_empty_remark_is_not_spoken(self):
        self.assertEqual(pg.check_remark("  ", hour=10), (False, "empty"))
        self.assertEqual(pg.check_remark("[intent:dry_wit]", hour=10),
                         (False, "empty"))


class RateVerdictTests(unittest.TestCase):
    """A4: an unanswered remark backs off; two buy silence."""

    KW = dict(cooldown_s=600.0, factor=2.0, max_unanswered=2,
              attempt_gap_s=300.0)

    def _v(self, now, remarks, owner_at, attempt_at=0.0, **over):
        kw = dict(self.KW)
        kw.update(over)
        return pg.rate_verdict(now, remark_times=remarks,
                               owner_turn_at=owner_at,
                               last_attempt_at=attempt_at, **kw)

    def test_the_live_follow_up_is_held(self):
        # 09:56:22 remark, nobody answered; 09:59:54 = 212 s later.
        t0 = 10_000.0
        ok, why = self._v(t0 + 212, [t0], owner_at=0.0, attempt_at=t0)
        self.assertFalse(ok)
        ok, why = self._v(t0 + 312, [t0], owner_at=0.0, attempt_at=t0)
        self.assertFalse(ok)
        self.assertIn("1 unanswered", why)
        self.assertTrue(self._v(t0 + 601, [t0], owner_at=0.0,
                                attempt_at=t0)[0])

    def test_two_unanswered_remarks_mean_quiet_until_the_owner_speaks(self):
        t0 = 10_000.0
        remarks = [t0, t0 + 700]
        ok, why = self._v(t0 + 99_999, remarks, owner_at=0.0,
                          attempt_at=t0 + 700)
        self.assertFalse(ok)
        self.assertIn("quiet until the owner speaks", why)
        # He answers after the second remark: the streak resets.
        self.assertTrue(self._v(t0 + 99_999, remarks, owner_at=t0 + 800,
                                attempt_at=t0 + 700)[0])

    def test_the_wait_grows_with_each_unanswered_remark(self):
        t0 = 10_000.0
        # max 3 so the second step is visible: 600 s, then 1200 s.
        ok, _ = self._v(t0 + 1100, [t0, t0 + 601], owner_at=0.0,
                        attempt_at=t0 + 601, max_unanswered=3)
        self.assertFalse(ok)          # 499 s after the 2nd < 1200 s
        ok, _ = self._v(t0 + 1802, [t0, t0 + 601], owner_at=0.0,
                        attempt_at=t0 + 601, max_unanswered=3)
        self.assertTrue(ok)

    def test_an_answered_remark_does_not_back_off(self):
        t0 = 10_000.0
        self.assertTrue(self._v(t0 + 400, [t0], owner_at=t0 + 30,
                                attempt_at=t0)[0])

    def test_a_dropped_attempt_waits_the_attempt_gap(self):
        t0 = 10_000.0
        ok, why = self._v(t0 + 100, [], owner_at=0.0, attempt_at=t0)
        self.assertFalse(ok)
        self.assertIn("last attempt", why)
        self.assertTrue(self._v(t0 + 301, [], owner_at=0.0, attempt_at=t0)[0])

    def test_unanswered_count(self):
        self.assertEqual(pg.unanswered([1.0, 5.0, 9.0], 4.0), 2)
        self.assertEqual(pg.unanswered([], 4.0), 0)


if __name__ == "__main__":
    unittest.main()
