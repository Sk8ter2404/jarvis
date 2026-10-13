"""Untimed reminders + the shopping list (core/untimed_reminders.py).

"Remind me to buy X" with no time -> shopping list; any other untimed reminder
-> to-do list; a reminder that names a time is NOT matched here (it stays a
timer / schedule).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from core import untimed_reminders as ur


def _m(text):
    got = ur.match(text)
    return None if got is None else (got.action, got.arg)


class RemindRoutingTests(unittest.TestCase):
    def test_buying_goes_to_the_shopping_list(self):
        self.assertEqual(_m("remind me to buy milk"),
                         (ur.ADD_ITEM, "milk"))
        self.assertEqual(_m("Jarvis, remind me to buy milk."),
                         (ur.ADD_ITEM, "milk"))
        self.assertEqual(_m("can you please remind me to pick up eggs"),
                         (ur.ADD_ITEM, "eggs"))
        self.assertEqual(_m("don't let me forget to buy batteries"),
                         (ur.ADD_ITEM, "batteries"))
        self.assertEqual(_m("set a reminder to order more filament"),
                         (ur.ADD_ITEM, "more filament"))

    def test_a_list_of_things_becomes_several_items(self):
        self.assertEqual(_m("remind me to buy milk, eggs and bread"),
                         (ur.ADD_ITEM, "milk, eggs, bread"))
        self.assertEqual(_m("remind me to get some eggs and butter"),
                         (ur.ADD_ITEM, "eggs, butter"))

    def test_other_untimed_reminders_go_to_the_todo_list(self):
        self.assertEqual(_m("remind me to call the dentist"),
                         (ur.ADD_REMINDER, "call the dentist"))
        self.assertEqual(_m("remind me about my registration"),
                         (ur.ADD_REMINDER, "my registration"))
        self.assertEqual(_m("remind me that I need to renew my passport"),
                         (ur.ADD_REMINDER, "renew my passport"))

    def test_errands_that_use_a_shopping_verb_are_not_shopping(self):
        for task in ("pick up the kids", "get the oil changed",
                     "get a haircut", "pick up my prescription",
                     "grab lunch with Sam", "get back to Alex"):
            with self.subTest(task=task):
                self.assertEqual(_m(f"remind me to {task}"),
                                 (ur.ADD_REMINDER, task))

    def test_a_chore_in_the_same_breath_keeps_the_whole_thing_a_todo(self):
        self.assertEqual(
            _m("remind me to buy milk and call mom"),
            (ur.ADD_REMINDER, "buy milk and call mom"))

    def test_vague_when_words_still_mean_no_time(self):
        self.assertEqual(_m("remind me to buy milk later"),
                         (ur.ADD_ITEM, "milk"))
        self.assertEqual(_m("remind me to call Sam sometime"),
                         (ur.ADD_REMINDER, "call Sam"))

    def test_conditions_are_kept_in_the_text(self):
        self.assertEqual(
            _m("remind me to check the print when it finishes"),
            (ur.ADD_REMINDER, "check the print when it finishes"))

    def test_items_with_numbers_are_not_times(self):
        self.assertEqual(_m("remind me to buy 2 percent milk"),
                         (ur.ADD_ITEM, "2 percent milk"))
        self.assertEqual(_m("remind me to buy sun cream"),
                         (ur.ADD_ITEM, "sun cream"))
        self.assertEqual(_m("remind me to buy water"), (ur.ADD_ITEM, "water"))

    def test_a_reminder_with_a_time_is_left_alone(self):
        for t in ("remind me to buy milk in 20 minutes",
                  "remind me in an hour to stretch",
                  "remind me to call mom tomorrow",
                  "remind me to buy milk at 5",
                  "remind me to call Sam at 3:30 pm",
                  "remind me tonight to lock up",
                  "remind me every monday to buy milk",
                  "remind me to pay rent on friday",
                  "remind me next week to renew",
                  "remind me to leave at noon",
                  "remind me to take the pill every morning",
                  "remind me this evening to call",
                  "remind me to stretch in half an hour"):
            with self.subTest(t=t):
                self.assertIsNone(_m(t))

    def test_timers_and_other_requests_never_match(self):
        for t in ("set a timer for five minutes", "5 minutes", "play music",
                  "what time is it", "remind me", "remind me to", "",
                  "what do I remind you of"):
            with self.subTest(t=t):
                self.assertIsNone(_m(t))

    def test_non_strings_never_raise(self):
        for t in (None, 5, [], b"remind me to buy milk"):
            self.assertIsNone(ur.match(t))


class ListCommandTests(unittest.TestCase):
    def test_add_by_name(self):
        self.assertEqual(_m("add milk to my shopping list"),
                         (ur.ADD_ITEM, "milk"))
        self.assertEqual(_m("add eggs and bread to the grocery list"),
                         (ur.ADD_ITEM, "eggs, bread"))
        self.assertEqual(_m("put paper towels on the shopping list"),
                         (ur.ADD_ITEM, "paper towels"))

    def test_read(self):
        for t in ("what's on my shopping list", "what is on the grocery list",
                  "whats on my shopping list", "read me my shopping list",
                  "show me the shopping list", "read my grocery list",
                  "do I have anything on my shopping list",
                  "what do I need to buy"):
            with self.subTest(t=t):
                self.assertEqual(_m(t), (ur.SHOW_LIST, ""))

    def test_remove_one(self):
        self.assertEqual(_m("remove milk from the shopping list"),
                         (ur.REMOVE_ITEM, "milk"))
        self.assertEqual(_m("take eggs off my shopping list"),
                         (ur.REMOVE_ITEM, "eggs"))
        self.assertEqual(_m("cross bread off the grocery list"),
                         (ur.REMOVE_ITEM, "bread"))

    def test_clear(self):
        for t in ("clear my shopping list", "clear the shopping list",
                  "empty the grocery list", "wipe my shopping list",
                  "delete the shopping list",
                  "clear everything off the shopping list",
                  "remove everything from my shopping list",
                  "start over on my shopping list"):
            with self.subTest(t=t):
                self.assertEqual(_m(t), (ur.CLEAR_LIST, ""))

    def test_other_lists_are_not_ours(self):
        for t in ("what's on my to do list", "clear the task list",
                  "add milk to my calendar", "show me the list"):
            with self.subTest(t=t):
                self.assertIsNone(_m(t))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("JARVIS_DATA_DIR")
        self._dir = tempfile.mkdtemp(prefix="jarvis_shop_test_")
        os.environ["JARVIS_DATA_DIR"] = self._dir

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("JARVIS_DATA_DIR", None)
        else:
            os.environ["JARVIS_DATA_DIR"] = self._saved
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_missing_file_is_an_empty_list(self):
        self.assertEqual(ur.load_items(), [])

    def test_add_persists_and_dedupes(self):
        added, dup = ur.add_items(["Milk", "eggs"])
        self.assertEqual((added, dup), (["Milk", "eggs"], []))
        added, dup = ur.add_items(["milk", "bread", "Eggs"])
        self.assertEqual((added, dup), (["bread"], ["milk", "Eggs"]))
        self.assertEqual([i["text"] for i in ur.load_items()],
                         ["Milk", "eggs", "bread"])

    def test_plurals_and_articles_count_as_the_same_item(self):
        ur.add_items(["a banana"])
        added, dup = ur.add_items(["bananas", "the banana"])
        self.assertEqual(added, [])
        self.assertEqual(len(dup), 2)

    def test_remove_exact_and_partial(self):
        ur.add_items(["whole milk", "eggs", "bread"])
        self.assertEqual(ur.remove_matching("eggs"), ("eggs", []))
        self.assertEqual(ur.remove_matching("milk"), ("whole milk", []))
        self.assertEqual([i["text"] for i in ur.load_items()], ["bread"])

    def test_remove_ambiguous_asks_and_removes_nothing(self):
        ur.add_items(["whole milk", "oat milk"])
        removed, cands = ur.remove_matching("milk")
        self.assertIsNone(removed)
        self.assertEqual(sorted(cands), ["oat milk", "whole milk"])
        self.assertEqual(len(ur.load_items()), 2)

    def test_exact_match_beats_partial(self):
        ur.add_items(["milk", "oat milk"])
        self.assertEqual(ur.remove_matching("milk"), ("milk", []))

    def test_remove_missing(self):
        ur.add_items(["eggs"])
        self.assertEqual(ur.remove_matching("caviar"), (None, []))

    def test_clear_keeps_a_backup(self):
        ur.add_items(["eggs", "milk"])
        self.assertEqual(ur.clear_all(), 2)
        self.assertEqual(ur.load_items(), [])
        self.assertTrue(os.path.exists(
            os.path.join(self._dir, "shopping_list.json.bak")))
        self.assertEqual(ur.clear_all(), 0)

    def test_corrupt_file_reads_as_empty_not_an_error(self):
        with open(os.path.join(self._dir, "shopping_list.json"), "w") as f:
            f.write("{not json")
        self.assertEqual(ur.load_items(), [])

    def test_spoken_join(self):
        self.assertEqual(ur.spoken_join(["a"]), "a")
        self.assertEqual(ur.spoken_join(["a", "b"]), "a and b")
        self.assertEqual(ur.spoken_join(["a", "b", "c"]), "a, b and c")


if __name__ == "__main__":
    unittest.main()
