"""Logic tests for skills/teams_nudge.py.

Vision-based Teams unread nudger. We never capture a screen or call a real VLM:
bobert_companion is faked via _import_companion. Coverage:

  • _build_message — singular / plural / sender / no-sender phrasing
  • _ask_vision_for_teams_state — UNREAD:N | sender parsing, NONE, and
    capture/vision failure degradation
  • _check_once — snooze: identical alert suppressed within SNOOZE_SECONDS
  • _enqueue_speech — the draft_confirm gate is fail-closed: a denied/timed-out
    confirmation drops the nudge and never writes to pending_speech.json
  • the check_teams action — unread vs clear

The background monitor thread never starts (harness neuters threads).
"""
from __future__ import annotations

import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


def _fake_companion(*, images=("img1",), answer="UNREAD: 2 | Alex Morgan"):
    bc = types.ModuleType("bobert_companion")
    bc.take_all_monitor_screenshots = mock.MagicMock(return_value=list(images))
    bc.ask_vision_multi = mock.MagicMock(return_value=answer)
    bc.proactive_announce = mock.MagicMock(return_value=True)
    return bc


class TeamsBuildMessageTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("teams_nudge")

    def test_single_with_sender(self):
        out = self.mod._build_message(1, "Alex Morgan")
        self.assertIn("an unread message", out)
        self.assertIn("Alex Morgan", out)

    def test_single_without_sender(self):
        out = self.mod._build_message(1, "")
        self.assertIn("an unread message on Teams", out)
        self.assertNotIn("from", out)

    def test_plural_with_sender(self):
        out = self.mod._build_message(3, "Alex Morgan")
        self.assertIn("3 unread messages", out)
        self.assertIn("including one from Alex Morgan", out)

    def test_plural_without_sender(self):
        out = self.mod._build_message(5, "")
        self.assertIn("5 unread messages on Teams", out)


class TeamsVisionParseTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("teams_nudge")

    def _ask(self, *, images=("img1",), answer="UNREAD: 2 | Alex Morgan"):
        bc = _fake_companion(images=images, answer=answer)
        with mock.patch.object(self.mod, "_import_companion", return_value=bc):
            return self.mod._ask_vision_for_teams_state()

    def test_parses_count_and_sender(self):
        has, count, sender = self._ask(answer="UNREAD: 2 | Alex Morgan")
        self.assertTrue(has)
        self.assertEqual(count, 2)
        self.assertEqual(sender, "Alex Morgan")

    def test_parses_count_with_none_sender(self):
        has, count, sender = self._ask(answer="UNREAD: 1 | NONE")
        self.assertTrue(has)
        self.assertEqual(count, 1)
        self.assertEqual(sender, "")

    def test_none_answer_is_no_unread(self):
        has, count, sender = self._ask(answer="NONE")
        self.assertFalse(has)
        self.assertEqual(count, 0)

    def test_no_images_degrades(self):
        has, count, raw = self._ask(images=())
        self.assertFalse(has)
        self.assertEqual(raw, "no_images")

    def test_vision_failure_degrades(self):
        bc = _fake_companion()
        bc.ask_vision_multi = mock.MagicMock(side_effect=RuntimeError("boom"))
        with mock.patch.object(self.mod, "_import_companion", return_value=bc):
            has, count, raw = self.mod._ask_vision_for_teams_state()
        self.assertFalse(has)
        self.assertIn("vision_failed", raw)

    def test_capture_failure_degrades(self):
        bc = _fake_companion()
        bc.take_all_monitor_screenshots = mock.MagicMock(
            side_effect=RuntimeError("no displays"))
        with mock.patch.object(self.mod, "_import_companion", return_value=bc):
            has, count, raw = self.mod._ask_vision_for_teams_state()
        self.assertFalse(has)
        self.assertIn("capture_failed", raw)


class TeamsCheckOnceSnoozeTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("teams_nudge")
        self.mod._last_alert_at[0] = 0.0
        self.mod._last_alert_text[0] = ""

    def test_returns_message_on_first_unread(self):
        with mock.patch.object(self.mod, "_ask_vision_for_teams_state",
                               return_value=(True, 2, "Alex")):
            has, payload = self.mod._check_once()
        self.assertTrue(has)
        self.assertIn("2 unread messages", payload)

    def test_identical_alert_snoozed(self):
        with mock.patch.object(self.mod, "_ask_vision_for_teams_state",
                               return_value=(True, 2, "Alex")):
            self.mod._check_once()                  # first → arms snooze
            has, payload = self.mod._check_once()    # immediate repeat
        self.assertTrue(has)
        self.assertEqual(payload, "snoozed")

    def test_no_unread_returns_clear(self):
        with mock.patch.object(self.mod, "_ask_vision_for_teams_state",
                               return_value=(False, 0, "")):
            has, payload = self.mod._check_once()
        self.assertFalse(has)
        self.assertEqual(payload, "no_unread")

    def test_snooze_expires_after_window(self):
        with mock.patch.object(self.mod, "_ask_vision_for_teams_state",
                               return_value=(True, 2, "Alex")):
            self.mod._check_once()
            # Push the last-alert timestamp past the snooze window.
            self.mod._last_alert_at[0] = time.time() - self.mod.SNOOZE_SECONDS - 1
            has, payload = self.mod._check_once()
        self.assertTrue(has)
        self.assertNotEqual(payload, "snoozed")


class TeamsEnqueueGateTests(unittest.TestCase):
    """The draft_confirm gate must be fail-closed."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("teams_nudge")

    def test_denied_confirmation_drops_nudge(self):
        # draft_confirm returns False → nothing should be announced/written.
        with mock.patch.object(self.mod, "draft_confirm", return_value=False), \
             mock.patch("importlib.import_module") as imp:
            self.mod._enqueue_speech("You have an unread message, sir.")
        # The companion proactive_announce path must never be reached.
        imp.assert_not_called()

    def test_approved_confirmation_routes_to_announce(self):
        bc = types.ModuleType("bobert_companion")
        bc.proactive_announce = mock.MagicMock(return_value=True)
        with mock.patch.object(self.mod, "draft_confirm", return_value=True), \
             mock.patch("importlib.import_module", return_value=bc):
            self.mod._enqueue_speech("You have an unread message, sir.")
        bc.proactive_announce.assert_called_once()
        self.assertIn("unread message", bc.proactive_announce.call_args.args[0])


class TeamsActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("teams_nudge")

    def test_check_teams_unread(self):
        with mock.patch.object(self.mod, "_ask_vision_for_teams_state",
                               return_value=(True, 3, "Alex Morgan")):
            out = self.actions["check_teams"]("")
        self.assertIn("3 unread messages", out)
        self.assertIn("Alex Morgan", out)

    def test_check_teams_clear(self):
        with mock.patch.object(self.mod, "_ask_vision_for_teams_state",
                               return_value=(False, 0, "")):
            out = self.actions["check_teams"]("")
        self.assertIn("No unread Teams messages", out)


if __name__ == "__main__":
    unittest.main()
