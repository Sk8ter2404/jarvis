"""Logic tests for skills/shopping_list.py: the shopping list and the
reminders that name no time. Every reply must say where the item went."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _Base(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("JARVIS_DATA_DIR")
        self._dir = tempfile.mkdtemp(prefix="jarvis_shoplist_test_")
        os.environ["JARVIS_DATA_DIR"] = self._dir
        self.queued = []
        self.mod, self.actions = load_skill_isolated(
            "shopping_list", actions={"queue_task": self._queue})

    def _queue(self, arg):
        self.queued.append(arg)
        return f"queued: {arg}"

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("JARVIS_DATA_DIR", None)
        else:
            os.environ["JARVIS_DATA_DIR"] = self._saved
        shutil.rmtree(self._dir, ignore_errors=True)


class RegistrationTests(_Base):
    def test_actions_and_speak_set(self):
        for name in ("add_reminder", "add_shopping_item", "show_shopping_list",
                     "remove_shopping_item", "clear_shopping_list"):
            self.assertIn(name, self.actions)
            self.assertIn(name, self.mod.SPEAK_VERBATIM_ACTIONS)


class AddAndReadTests(_Base):
    def test_add_says_where_it_went(self):
        out = self.actions["add_shopping_item"]("milk, eggs")
        self.assertEqual(out, "Added milk and eggs to your shopping list, sir.")
        self.assertEqual(
            self.actions["show_shopping_list"](""),
            "Your shopping list has 2 items, sir: milk and eggs.")

    def test_empty_list(self):
        self.assertEqual(self.actions["show_shopping_list"](""),
                         "Your shopping list is empty, sir.")

    def test_one_item_grammar(self):
        self.actions["add_shopping_item"]("milk")
        self.assertEqual(self.actions["show_shopping_list"](""),
                         "Your shopping list has 1 item, sir: milk.")

    def test_duplicate_is_reported_not_added_twice(self):
        self.actions["add_shopping_item"]("milk")
        out = self.actions["add_shopping_item"]("Milk")
        self.assertEqual(out, "Milk is already on your shopping list, sir.")
        out = self.actions["add_shopping_item"]("milk, bread")
        self.assertEqual(
            out, "Added bread to your shopping list, sir. Milk was already "
                 "there.")

    def test_blank_add_asks(self):
        self.assertIn("What should I add", self.actions["add_shopping_item"](""))

    def test_failed_save_is_not_reported_as_success(self):
        with mock.patch("core.untimed_reminders._save",
                        side_effect=OSError("disk full")):
            out = self.actions["add_shopping_item"]("milk")
        self.assertIn("nothing was added", out)
        self.assertNotIn("Added", out)


class RemoveAndClearTests(_Base):
    def setUp(self):
        super().setUp()
        self.actions["add_shopping_item"]("whole milk, oat milk, eggs")

    def test_remove_one(self):
        self.assertEqual(self.actions["remove_shopping_item"]("eggs"),
                         "Took eggs off your shopping list, sir.")
        self.assertNotIn("eggs", self.actions["show_shopping_list"](""))

    def test_ambiguous_remove_asks_and_keeps_both(self):
        out = self.actions["remove_shopping_item"]("milk")
        self.assertIn("More than one item matches", out)
        self.assertIn("whole milk", out)
        self.assertIn("3 items", self.actions["show_shopping_list"](""))

    def test_remove_missing(self):
        self.assertIn("I don't see 'caviar'",
                      self.actions["remove_shopping_item"]("caviar"))

    def test_remove_everything_clears(self):
        out = self.actions["remove_shopping_item"]("everything")
        self.assertIn("Cleared your shopping list", out)
        self.assertEqual(self.actions["show_shopping_list"](""),
                         "Your shopping list is empty, sir.")

    def test_clear(self):
        self.assertEqual(self.actions["clear_shopping_list"](""),
                         "Cleared your shopping list, sir - 3 items removed.")
        self.assertEqual(self.actions["clear_shopping_list"](""),
                         "Your shopping list was already empty, sir.")


class AddReminderTests(_Base):
    """No-time reminders go to their OWN reminders list - never to the
    developer queue (queue_task / jarvis_todo.md), which the overnight
    upgrade pipeline reads as coding tasks."""

    def _reminders(self):
        from core import untimed_reminders as ur
        return [i["text"] for i in ur.load_items(ur.REMINDERS_LIST)]

    def test_purchase_goes_to_the_shopping_list(self):
        out = self.actions["add_reminder"]("buy milk")
        self.assertEqual(out, "Added milk to your shopping list, sir.")
        self.assertEqual(self.queued, [])
        self.assertEqual(self._reminders(), [])

    def test_other_reminder_goes_to_the_reminders_list(self):
        out = self.actions["add_reminder"]("call the dentist")
        self.assertEqual(self._reminders(), ["call the dentist"])
        self.assertEqual(self.queued, [], "must never reach the dev queue")
        self.assertIn("reminders list", out)
        self.assertIn("call the dentist", out)
        self.assertIn("no time", out)

    def test_duplicate_reminder_says_so(self):
        self.actions["add_reminder"]("call the dentist")
        out = self.actions["add_reminder"]("call the dentist")
        self.assertIn("already on your reminders list", out)
        self.assertEqual(self._reminders(), ["call the dentist"])

    def test_save_failure_is_an_honest_failure(self):
        with mock.patch.object(self.mod._ur, "add_items",
                               side_effect=OSError("disk")):
            out = self.actions["add_reminder"]("call the dentist")
        self.assertIn("nothing was saved", out)

    def test_blank_asks(self):
        self.assertIn("What should I remind you",
                      self.actions["add_reminder"](""))

    def test_never_schedules_a_timer_or_queues_a_task(self):
        self.assertNotIn("set_timer", self.actions)
        self.actions["add_reminder"]("buy milk")
        self.actions["add_reminder"]("call mom")
        self.assertEqual(self.queued, [])
        self.assertEqual(self._reminders(), ["call mom"])


class RemindersListTests(_Base):
    def test_registered_and_spoken_verbatim(self):
        for name in ("show_reminders_list", "remove_reminder_item",
                     "clear_reminders_list"):
            self.assertIn(name, self.actions)
            self.assertIn(name, self.mod.SPEAK_VERBATIM_ACTIONS)

    def test_show_empty_then_full(self):
        self.assertEqual(self.actions["show_reminders_list"](),
                         "Your reminders list is empty, sir.")
        self.actions["add_reminder"]("call the dentist")
        self.actions["add_reminder"]("renew the registration")
        out = self.actions["show_reminders_list"]()
        self.assertIn("2 items", out)
        self.assertIn("call the dentist and renew the registration", out)

    def test_remove_one(self):
        self.actions["add_reminder"]("call the dentist")
        out = self.actions["remove_reminder_item"]("dentist")
        self.assertIn("off your reminders list", out)
        self.assertEqual(self.actions["show_reminders_list"](),
                         "Your reminders list is empty, sir.")

    def test_remove_missing_says_so(self):
        out = self.actions["remove_reminder_item"]("the dentist")
        self.assertIn("I don't see", out)

    def test_clear_keeps_a_backup_and_leaves_shopping_alone(self):
        self.actions["add_reminder"]("call the dentist")
        self.actions["add_shopping_item"]("eggs")
        out = self.actions["clear_reminders_list"]()
        self.assertIn("1 item removed", out)
        self.assertIn("eggs", self.actions["show_shopping_list"]())
        self.assertTrue(os.path.exists(
            os.path.join(self._dir, "reminders_list.json.bak")))


if __name__ == "__main__":
    unittest.main()
