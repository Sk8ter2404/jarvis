"""set_timer never invents a duration (2026-10-12).

"Remind me to buy two eggs" used to read the number word inside the text as
minutes (a silent 2-minute timer), and a model that emitted set_timer for an
untimed reminder got a timer or a refusal. Now a number is a duration only when
it is all there is, and an untimed reminder reaching set_timer goes to the list
instead - never to a timer.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from tests._skill_harness import load_skill_isolated


class _Base(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("JARVIS_DATA_DIR")
        self._dir = tempfile.mkdtemp(prefix="jarvis_timer_untimed_")
        os.environ["JARVIS_DATA_DIR"] = self._dir
        self.added = []

        def add_reminder(arg):
            self.added.append(arg)
            return f"Added to your to-do list, sir: {arg}."

        self.mod, self.actions = load_skill_isolated(
            "timer", actions={"add_reminder": add_reminder})
        self.mod._timers.clear()
        self.mod._next_id[0] = 1

    def tearDown(self):
        # test_a_real_timer_is_untouched arms a REAL 5-minute threading.Timer;
        # clearing the dict alone left the thread alive, and 300 s later, mid
        # suite, its _fire re-imported the monolith from a background thread
        # (the ci-sim "LEAKED THREAD IMPORTED THE MONOLITH" note, every run
        # since 2026-10-12). Cancel before forgetting.
        for timer, _msg, _fire_at in list(self.mod._timers.values()):
            timer.cancel()
            # cancel() sets the event the thread waits on, so it exits at once
            # without firing; join proves it is gone rather than hoping.
            if timer.is_alive():
                timer.join(5)
                self.assertFalse(timer.is_alive(), "timer thread outlived tearDown")
        self.mod._timers.clear()
        if self._saved is None:
            os.environ.pop("JARVIS_DATA_DIR", None)
        else:
            os.environ["JARVIS_DATA_DIR"] = self._saved
        shutil.rmtree(self._dir, ignore_errors=True)


class BareNumberTests(_Base):
    def test_a_bare_number_is_still_minutes(self):
        for raw, secs in (("5", 300), ("five", 300), ("for 5", 300),
                          ("a timer of ten", 600), ("5 minutes", 300),
                          ("an hour", 3600), ("5 minutes for tea", 300)):
            with self.subTest(raw=raw):
                self.assertEqual(self.mod._split_timer_args(raw)[0], secs)

    def test_a_number_word_inside_other_text_is_not_a_duration(self):
        for raw in ("buy two eggs", "call one of the guys", "get one gallon",
                    "pick up three things", "buy milk"):
            with self.subTest(raw=raw):
                self.assertEqual(self.mod._split_timer_args(raw), (None, ""))

    def test_a_clock_time_or_unit_inside_a_label_is_unchanged(self):
        secs, label = self.mod._split_timer_args("5 minutes | buy two eggs")
        self.assertEqual((secs, label), (300, "buy two eggs"))


class SetTimerRedirectTests(_Base):
    def test_no_duration_with_a_task_goes_to_the_list_not_a_timer(self):
        out = self.actions["set_timer"]("later | call the dentist")
        self.assertEqual(self.added, ["call the dentist"])
        self.assertEqual(self.mod._timers, {})
        self.assertTrue(out.startswith("No time given, so no timer, sir."))
        self.assertIn("to-do list", out)

    def test_remind_me_form_without_a_time(self):
        self.actions["set_timer"]("remind me to call the dentist")
        self.assertEqual(self.added, ["call the dentist"])
        self.assertEqual(self.mod._timers, {})

    def test_a_purchase_without_a_time(self):
        self.actions["set_timer"]("buy two eggs")
        self.assertEqual(self.added, ["buy two eggs"])
        self.assertEqual(self.mod._timers, {})

    def test_no_default_24_hour_timer_is_ever_made(self):
        for arg in ("buy milk", "later | buy milk", "someday | call mom",
                    "tomorrow", "remind me to buy milk"):
            self.actions["set_timer"](arg)
        self.assertEqual(self.mod._timers, {})

    def test_a_real_timer_is_untouched(self):
        out = self.actions["set_timer"]("5 minutes | tea")
        self.assertIn("timer #1 set", out)
        self.assertEqual(self.added, [])
        self.assertEqual(len(self.mod._timers), 1)

    def test_bare_gibberish_is_still_an_honest_refusal(self):
        out = self.actions["set_timer"]("for a bit")
        self.assertIn("couldn't tell how long", out)
        self.assertEqual(self.added, [])
        self.assertEqual(self.mod._timers, {})

    def test_without_the_list_action_the_refusal_stays_honest(self):
        del self.actions["add_reminder"]
        out = self.actions["set_timer"]("later | call the dentist")
        self.assertIn("couldn't tell how long", out)
        self.assertEqual(self.mod._timers, {})


if __name__ == "__main__":
    unittest.main()
