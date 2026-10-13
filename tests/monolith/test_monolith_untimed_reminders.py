"""Monolith wiring for untimed reminders, the shopping list and "is it hours
week" (2026-10-12), with the REAL monolith and every action and speech stubbed:

  * "remind me to buy milk" reaches add_shopping_item (never a timer);
  * "remind me to call the dentist" reaches add_reminder (the to-do list);
  * a reminder that names a time falls through to the model/timers;
  * "what's on my shopping list" reads the list, with no LLM;
  * "is it hours week" is answered by hours_status.

    python -m unittest tests.monolith.test_monolith_untimed_reminders
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class UntimedReminderRouteTests(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self.spoken: list[str] = []
        self.calls: dict[str, list[str]] = {}
        self._p(bc, "_speak",
                side_effect=lambda t, *a, **k: self.spoken.append(t))
        self._p(bc, "set_state")
        self._p(bc, "_append_turn", lambda *a, **k: None)
        self._p(bc, "FAST_PATHS_ENABLED", True)
        self._actions = dict(bc.ACTIONS)
        self._p(bc, "ACTIONS", self._actions)
        for name, reply in (
                ("add_shopping_item", "Added milk to your shopping list, sir."),
                ("add_reminder", "Added to your to-do list, sir: x."),
                ("show_shopping_list", "Your shopping list is empty, sir."),
                ("remove_shopping_item", "Took milk off your shopping list."),
                ("clear_shopping_list", "Cleared your shopping list, sir."),
                ("hours_status", "Yes, sir, it's hours week.")):
            self._stub(name, reply)
        self._stub("set_timer", "timer #1 set")

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _stub(self, name, result):
        self.calls[name] = []

        def fn(arg=""):
            self.calls[name].append(arg)
            return result

        self._actions[name] = fn

    def _run(self, text):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.bc._run_voice_shortcuts(text)

    def test_purchase_reaches_the_shopping_list_not_a_timer(self):
        self.assertTrue(self._run("Jarvis, remind me to buy milk"))
        self.assertEqual(self.calls["add_shopping_item"], ["milk"])
        self.assertEqual(self.calls["set_timer"], [])
        self.assertEqual(self.spoken, ["Added milk to your shopping list, sir."])

    def test_other_reminder_reaches_the_todo_list(self):
        self.assertTrue(self._run("remind me to call the dentist"))
        self.assertEqual(self.calls["add_reminder"], ["call the dentist"])
        self.assertEqual(self.calls["add_shopping_item"], [])
        self.assertEqual(self.calls["set_timer"], [])

    def test_a_reminder_with_a_time_is_not_taken(self):
        self.assertFalse(self.bc._run_untimed_reminder_shortcut(
            "remind me to buy milk in 20 minutes"))
        self.assertEqual(self.calls["add_shopping_item"], [])
        self.assertEqual(self.calls["add_reminder"], [])

    def test_read_remove_and_clear_the_list(self):
        self.assertTrue(self._run("what's on my shopping list"))
        self.assertEqual(self.calls["show_shopping_list"], [""])
        self.assertTrue(self._run("take milk off the shopping list"))
        self.assertEqual(self.calls["remove_shopping_item"], ["milk"])
        self.assertTrue(self._run("clear the shopping list"))
        self.assertEqual(self.calls["clear_shopping_list"], [""])

    def test_hours_week_question(self):
        self.assertTrue(self._run("is it hours week?"))
        self.assertEqual(self.calls["hours_status"], [""])
        self.assertEqual(self.spoken, ["Yes, sir, it's hours week."])

    def test_ordinary_hours_talk_is_not_taken(self):
        self.assertFalse(self.bc._run_hours_shortcut(
            "how many hours of battery are left"))
        self.assertEqual(self.calls["hours_status"], [])

    def test_off_when_fast_paths_are_off(self):
        self._p(self.bc, "FAST_PATHS_ENABLED", False)
        self.assertFalse(self.bc._run_untimed_reminder_shortcut(
            "remind me to buy milk"))
        self.assertEqual(self.calls["add_shopping_item"], [])

    def test_missing_action_falls_through_to_the_model(self):
        del self._actions["add_shopping_item"]
        self.assertFalse(self.bc._run_untimed_reminder_shortcut(
            "remind me to buy milk"))


    def test_a_to_do_reminder_is_not_what_you_were_working_on(self):
        # A reminder line sits in the same to-do file as the build tasks; the
        # warm-restart greeting must not announce it as work in progress.
        import os
        import tempfile
        import time
        stamp = time.strftime("%Y-%m-%d %H:%M",
                              time.localtime(time.time() - 3600))
        td = tempfile.mkdtemp()
        todo = os.path.join(td, "todo.md")
        with open(todo, "w", encoding="utf-8") as f:
            f.write(f"- [ ] **{stamp}** - [reminder] call the dentist" + chr(10))
        with mock.patch.object(self.bc, "TODO_FILE", todo):
            self.assertEqual(self.bc._last_queued_task_line(), "")


if __name__ == "__main__":
    unittest.main()
