"""core/pending_policy: per-source age and fold rules of the speech queue, and
core/brain_warmup. Fixtures are the lines of the live morning of 2026-11-05: a
week-old evening briefing, recap, clap acknowledgement and mic change spoken at
10:54, and RAM recaps at 12:55 and 14:44."""
from __future__ import annotations

import unittest

from core import brain_warmup, owner_presence, pending_policy as pp

NOW = 1_800_000_000.0
DAY = 86400.0
WEEK = 7 * DAY
MORNING = 10   # 10:54
TAG = "[intent:briefing] "


def _e(message, source, age=0.0, **kw):
    return dict({"ts": NOW - age, "message": message, "source": source}, **kw)


class ClassTests(unittest.TestCase):
    def test_the_lines_of_the_live_morning(self):
        for msg, src, cls in (
            ("You rang, sir?", "clap", "clap_ack"),
            ("Switched to the Blue Snowball, sir.", "audio", "mic_change"),
            ("Switched to your headset, sir — the mic appears to have "
             "disconnected.", "audio", "mic_change"),
            ("headset off - audio back to the speakers", "audio",
             "mic_change"),
            ("Sir, memory usage is at 92 percent. The biggest memory users "
             "are ...", "system_monitor", "ram"),
            (TAG + "Good evening, sir. A quiet day.", "evening", "briefing"),
            (TAG + "Sir, today played 1 tracks.", "recap", "briefing"),
            (TAG + "Good morning, sir.", "arrival_v2", "briefing"),
            ("Your week so far, sir.", "weekly_digest_briefing", "briefing"),
        ):
            self.assertEqual(pp.line_class(_e(msg, src)), cls, msg)

    def test_a_similar_line_from_another_source_keeps_its_own_class(self):
        # The voice skill also says "Switched to ..."; a skill may say "you
        # rang"; only the source that means it gets the short limit.
        self.assertEqual(pp.line_class(_e("Switched to the 'x' voice, sir.",
                                          "voice_clone")), "other")
        self.assertEqual(pp.line_class(_e("You rang the bell.", "doorbell")),
                         "other")
        self.assertEqual(pp.line_class(_e("Your routine is done, sir.",
                                          "clap")), "other")

    def test_other_audio_and_system_lines_stay_ordinary(self):
        # A hearing alert is not a mic change: it may still be recapped.
        self.assertEqual(pp.line_class(_e(
            "Sir, I may not be able to hear you.", "audio")), "other")
        self.assertEqual(pp.line_class(_e(
            "Sir, the disk is nearly full.", "system_monitor")), "other")

    def test_his_own_and_urgent_lines_have_no_class(self):
        for src in ("timer", "schedule", "promise:7", "guard"):
            self.assertEqual(pp.line_class(_e("Reminder, sir.", src)), "")
        self.assertEqual(pp.line_class(
            _e("Print failed.", "bambu", urgent=True)), "")
        self.assertEqual(pp.line_class("not an entry"), "")


class AgeTests(unittest.TestCase):
    def _kept(self, entry, hour=MORNING):
        return pp.plan_ages([entry], now=NOW, local_hour=hour)[0] == [entry]

    def test_the_week_old_batch_is_dropped(self):
        items = [
            _e(TAG + "Sir, today played 1 tracks. Shall I queue?", "recap",
               WEEK),
            _e(TAG + "Good evening, sir. A quiet day.", "evening", WEEK),
            _e("You rang, sir?", "clap", WEEK),
            _e("Switched to the Blue Snowball, sir.", "audio", WEEK),
            _e("A status line.", "whatever", WEEK),
        ]
        kept, dropped = pp.plan_ages(items, now=NOW, local_hour=MORNING)
        self.assertEqual(kept, [])
        self.assertEqual(len(dropped), 5)
        self.assertTrue(all(why for _e2, why in dropped))

    def test_clap_ack_limit_is_about_ten_seconds(self):
        self.assertTrue(self._kept(_e("You rang, sir?", "clap", 2)))
        self.assertTrue(self._kept(_e("You rang, sir?", "clap", 9.5)))
        self.assertFalse(self._kept(_e("You rang, sir?", "clap", 67)))
        self.assertFalse(self._kept(_e("You rang, sir?", "clap", 96)))

    def test_ram_and_mic_limits(self):
        ram = "Sir, memory usage is at 92 percent."
        self.assertTrue(self._kept(_e(ram, "system_monitor", 60)))
        self.assertFalse(self._kept(_e(ram, "system_monitor", 600)))
        mic = "Switched to the Blue Snowball, sir."
        self.assertTrue(self._kept(_e(mic, "audio", 60)))
        self.assertFalse(self._kept(_e(mic, "audio", 900)))

    def test_a_morning_briefing_may_wait_a_few_hours(self):
        self.assertTrue(self._kept(
            _e(TAG + "Good morning, sir. Rain later.", "morning", 3 * 3600)))
        self.assertFalse(self._kept(
            _e(TAG + "Good morning, sir. Rain later.", "morning", 7 * 3600)))

    def test_his_own_reminders_and_urgent_lines_never_expire(self):
        for e in (_e("Reminder, sir: your timer.", "timer", WEEK),
                  _e("As promised, sir.", "promise:3", WEEK),
                  _e("Someone is in the room.", "guard", WEEK),
                  _e("Print failed, sir.", "bambu", WEEK, urgent=True)):
            self.assertTrue(self._kept(e), e["message"])

    def test_no_timestamp_counts_as_fresh(self):
        self.assertTrue(self._kept({"message": "You rang, sir?",
                                    "source": "clap"}))
        self.assertTrue(self._kept({"message": "You rang, sir?",
                                    "source": "clap", "ts": "soon"}))

    def test_a_clock_that_moved_back_drops_nothing(self):
        self.assertTrue(self._kept(_e("You rang, sir?", "clap", -500)))

    def test_order_and_non_entries_survive(self):
        items = ["odd", _e("A", "x"), _e("You rang, sir?", "clap", 99),
                 _e("B", "y")]
        kept, _d = pp.plan_ages(items, now=NOW, local_hour=MORNING)
        self.assertEqual(kept, [items[0], items[1], items[3]])

    def test_a_bad_queue_comes_back_unchanged(self):
        kept, dropped = pp.plan_ages(["x"], now="not a number")
        self.assertEqual((kept, dropped), (["x"], []))


class GoodEveningTests(unittest.TestCase):
    LINE = TAG + "Good evening, sir. A quiet day."

    def _kept(self, age, hour):
        e = _e(self.LINE, "evening", age)
        return pp.plan_ages([e], now=NOW, local_hour=hour)[0] == [e]

    def test_not_in_the_morning_or_afternoon_once_it_has_waited(self):
        for hour in (4, 8, 10, 12, 16):
            self.assertFalse(self._kept(900, hour), hour)

    def test_said_in_the_evening_and_the_small_hours(self):
        for hour in (17, 18, 22, 23, 0, 1, 3):
            self.assertTrue(self._kept(900, hour), hour)

    def test_a_briefing_he_just_asked_for_still_speaks(self):
        # The action queues it too; 120 s is what tells that from a waited one.
        self.assertTrue(self._kept(5, 11))

    def test_only_a_line_that_opens_with_it(self):
        e = _e("Sir, your good evening playlist is ready.", "morning", 900)
        self.assertEqual(pp.plan_ages([e], now=NOW, local_hour=11)[0], [e])


class FoldTests(unittest.TestCase):
    def test_never_fold_lines(self):
        self.assertTrue(pp.never_fold(_e("You rang, sir?", "clap")))
        self.assertTrue(pp.never_fold(_e("Switched to the mic, sir.", "audio")))
        self.assertTrue(pp.never_fold(
            _e("Sir, memory usage is at 90 percent.", "system_monitor")))
        self.assertTrue(pp.never_fold(_e(TAG + "Good morning, sir.",
                                         "arrival_v2")))
        self.assertFalse(pp.never_fold(_e("Sir, the disk is full.",
                                          "system_monitor")))
        self.assertFalse(pp.never_fold(_e("Reminder.", "timer")))

    def test_plan_return_drain_keeps_a_never_fold_line_whole(self):
        ram = _e("Sir, memory usage is at 90 percent.", "system_monitor", 700)
        disk = _e("Sir, the disk is nearly full.", "system_monitor2", 700)
        out, dropped, folded = owner_presence.plan_return_drain(
            [ram, disk], now=NOW, stale_s=600, never_fold=pp.never_fold)
        self.assertEqual(folded, [disk])
        self.assertEqual(dropped, [])
        self.assertIn(ram, out)

    def test_plan_return_drain_without_it_folds_as_before(self):
        ram = _e("Sir, memory usage is at 90 percent.", "system_monitor", 700)
        _out, _d, folded = owner_presence.plan_return_drain(
            [ram], now=NOW, stale_s=600)
        self.assertEqual(folded, [ram])


class GreetingFirstTests(unittest.TestCase):
    GREET = _e(TAG + "Good morning, sir. Nothing of note overnight.",
               "arrival_v2")
    RECAP = _e("While you were away, sir: the disk.", "recap:presence")

    def test_the_greeting_goes_ahead_of_the_recap(self):
        self.assertEqual(pp.greeting_first([self.RECAP, self.GREET]),
                         [self.GREET, self.RECAP])

    def test_his_timer_and_an_urgent_line_stay_ahead(self):
        timer = _e("Reminder, sir: your timer.", "timer")
        urgent = _e("Print failed, sir.", "bambu", urgent=True)
        self.assertEqual(
            pp.greeting_first([timer, urgent, self.RECAP, self.GREET]),
            [timer, urgent, self.GREET, self.RECAP])

    def test_nothing_moves_without_a_greeting(self):
        a, b = _e("A", "x"), _e("B", "y")
        self.assertEqual(pp.greeting_first([a, b]), [a, b])

    def test_a_good_morning_from_elsewhere_is_not_a_greeting_line(self):
        # A skill quoting the words, not the arrival or a briefing.
        chat = _e("Good morning was the title, sir.", "media_recall")
        a = _e("A", "x")
        self.assertEqual(pp.greeting_first([a, chat]), [a, chat])

    def test_several_greetings_keep_their_order(self):
        g1 = _e("Good morning, sir.", "morning")
        g2 = _e("Good afternoon, sir.", "handoff")
        self.assertEqual(pp.greeting_first([self.RECAP, g1, g2]),
                         [g1, g2, self.RECAP])


class BrainWarmupTests(unittest.TestCase):
    def setUp(self):
        brain_warmup.note_done()
        self.addCleanup(brain_warmup.note_done)

    def test_warming_between_start_and_done(self):
        self.assertFalse(brain_warmup.warming())
        brain_warmup.note_start()
        self.assertTrue(brain_warmup.warming())
        brain_warmup.note_done()
        self.assertFalse(brain_warmup.warming())

    def test_a_warmup_that_never_ended_stops_counting(self):
        brain_warmup.note_start(now=100.0)
        self.assertTrue(brain_warmup.warming(now=100.0 + 60))
        self.assertFalse(brain_warmup.warming(
            now=100.0 + brain_warmup.WARMUP_MAX_S + 1))


if __name__ == "__main__":
    unittest.main()
