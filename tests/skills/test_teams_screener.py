"""Logic tests for skills/teams_screener.py.

Regression coverage for the priority-VIP auto-decline grace window:

  • PRIORITY_AUTO_DECLINE_SECONDS must be long enough for the queued
    announcement to drain and be spoken (~3 s) PLUS the wake/STT/LLM routing
    of the user's spoken "answer"/"decline" (several more seconds). The old
    4 s window expired before the announcement even finished, so every
    off-hours priority call was auto-declined regardless of user intent.
  • The auto-decline thread sleeps the FULL configured grace window before
    checking _active_call, and does nothing when the user consumed the call
    (answered/declined) inside the window.

The background monitor thread never starts (harness neuters threads).
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

# skills/teams_screener.py is a GITIGNORED personal skill (it carries the
# owner's VIP names) — it exists on the owner's box but not in the public
# repo, so on GitHub CI there is nothing to load. Skip cleanly there.
_SKILL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "skills", "teams_screener.py")


@unittest.skipUnless(os.path.exists(_SKILL_PATH),
                     "teams_screener is a gitignored personal skill "
                     "(absent on CI)")
class PriorityGraceWindowTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("teams_screener")

    def test_grace_window_covers_announcement_and_reply(self):
        # Announcement drain (~3 s) + wake/STT/LLM routing means anything
        # under ~20 s auto-declines before the user can possibly respond.
        # Must also decline before Teams' ~30 s ring hits voicemail.
        self.assertGreaterEqual(self.mod.PRIORITY_AUTO_DECLINE_SECONDS, 20)
        self.assertLess(self.mod.PRIORITY_AUTO_DECLINE_SECONDS, 30)

    def test_auto_decline_sleeps_full_grace_window(self):
        vip = self.mod.VIPS[0]
        self.mod._arm_call(vip)
        sleeps = []
        # 2026-07-14 audit: the grace thread now re-checks live window state
        # before declining, so the "still unanswered" case this test pins
        # requires _detect to report the call is STILL ringing.
        with mock.patch.object(self.mod, "_pause_music_via_main",
                               return_value=True), \
             mock.patch.object(self.mod, "_enqueue_speech"), \
             mock.patch.object(self.mod, "_detect",
                               return_value=("call", vip, "caller | Microsoft Teams")), \
             mock.patch.object(self.mod, "_act_decline_call",
                               return_value="") as decline, \
             mock.patch.object(self.mod.time, "sleep",
                               side_effect=sleeps.append), \
             mock.patch.object(self.mod.threading, "Thread") as thread_cls:
            self.mod._vip_priority_handler(vip)
            # Run the countdown body synchronously.
            target = thread_cls.call_args.kwargs["target"]
            target()
        self.assertIn(self.mod.PRIORITY_AUTO_DECLINE_SECONDS, sleeps)
        decline.assert_called_once()

    def test_no_auto_decline_when_user_already_answered(self):
        vip = self.mod.VIPS[0]
        self.mod._arm_call(vip)
        with mock.patch.object(self.mod, "_pause_music_via_main",
                               return_value=True), \
             mock.patch.object(self.mod, "_enqueue_speech"), \
             mock.patch.object(self.mod, "_act_decline_call",
                               return_value="") as decline, \
             mock.patch.object(self.mod.time, "sleep"), \
             mock.patch.object(self.mod.threading, "Thread") as thread_cls:
            self.mod._vip_priority_handler(vip)
            # User says "answer" inside the grace window: the call is consumed
            # before the countdown wakes up.
            with mock.patch.object(self.mod, "_send_teams_hotkey",
                                   return_value=True):
                self.mod._act_answer_call("")
            target = thread_cls.call_args.kwargs["target"]
            target()
        decline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
