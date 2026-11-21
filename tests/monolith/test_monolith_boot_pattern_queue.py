"""n20-F6 (live 11-17 19:51:29) - BOOT_PATTERN_VIA_QUEUE: the boot-time day/hour
pattern remark and the habit offer were spoken straight to the speaker, before
listening was ready and past every hold (room talk, presence, a game). On (the
default) they go through the proactive queue as source "pattern_offer"; an
offer whose "yes" would run nothing (the app-usage line) is not made at all.
Off: spoken at once, as before.

Everything external is mocked: no audio, no queue file."""
from __future__ import annotations

import inspect
import unittest
from unittest import mock

from tests._monolith_harness import (
    MonolithGlobalsTestCase, load_monolith, requires_monolith)

APP_USAGE = ("You typically check Claude around now, sir - "
             "anything you'd like me to peek at?")
PLAY = ("Around this time on Fridays, sir, you usually put on Synth Mix. "
        "Shall I queue the essentials?")
REMARK = "A quiet Friday evening, sir; you usually wind down about now."


@requires_monolith
class BootPatternQueueTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _p(self, *a, **k):
        p = mock.patch.object(*a, **k)
        m = p.start()
        self.addCleanup(p.stop)
        return m

    def _setup(self, flag):
        bc = self.bc
        self._p(bc, "BOOT_PATTERN_VIA_QUEUE", flag, create=True)
        spoke = self._p(bc, "_speak")
        announce = self._p(bc, "proactive_announce", return_value=True)
        hist = self._p(bc, "conversation_history", [])
        return spoke, announce, hist

    def test_default_is_on_and_logged(self):
        import core.config as cfg
        self.assertIs(cfg.BOOT_PATTERN_VIA_QUEUE, True)
        self.assertIn("BOOT_PATTERN_VIA_QUEUE", self.bc._TURN_FLAG_KEYS)
        self.assertTrue(self.bc._boot_pattern_via_queue_on())

    def test_main_goes_through_the_helper_not_a_direct_speak(self):
        src = inspect.getsource(self.bc.main)
        self.assertIn("_boot_pattern_say(_pattern_remark, offer=False)", src)
        self.assertIn("_boot_pattern_say(_cmd_offer, offer=True)", src)
        self.assertNotIn("_speak(_pattern_remark)", src)
        self.assertNotIn("_speak(_cmd_offer)", src)

    def test_on_queues_the_offer_and_does_not_speak_it(self):
        spoke, announce, hist = self._setup(True)
        with mock.patch("builtins.print"):
            self.assertTrue(self.bc._boot_pattern_say(PLAY, offer=True))
        spoke.assert_not_called()
        announce.assert_called_once_with(PLAY, source="pattern_offer",
                                         offer=True)
        self.assertEqual(hist, [], "recorded before it was spoken")

    def test_on_queues_the_remark_as_a_plain_line(self):
        spoke, announce, _h = self._setup(True)
        with mock.patch("builtins.print"):
            self.bc._boot_pattern_say(REMARK, offer=False)
        spoke.assert_not_called()
        announce.assert_called_once_with(REMARK, source="pattern_offer",
                                         offer=False)

    def test_an_offer_with_no_action_behind_a_yes_is_dropped(self):
        spoke, announce, _h = self._setup(True)
        with mock.patch("builtins.print"):
            self.assertFalse(self.bc._boot_pattern_say(APP_USAGE, offer=True))
        spoke.assert_not_called()
        announce.assert_not_called()

    def test_off_speaks_here_and_records_history(self):
        spoke, announce, hist = self._setup(False)
        self.assertTrue(self.bc._boot_pattern_say(APP_USAGE, offer=True))
        spoke.assert_called_once_with(APP_USAGE)
        announce.assert_not_called()
        self.assertEqual(hist, [{"role": "assistant", "content": APP_USAGE}])

    def test_off_speaks_a_remark_without_history(self):
        spoke, announce, hist = self._setup(False)
        self.bc._boot_pattern_say(REMARK, offer=False)
        spoke.assert_called_once_with(REMARK)
        self.assertEqual(hist, [])

    def test_a_failing_queue_never_raises(self):
        spoke, announce, _h = self._setup(True)
        announce.side_effect = RuntimeError("disk full")
        with mock.patch("builtins.print"):
            self.assertFalse(self.bc._boot_pattern_say(PLAY, offer=True))
        spoke.assert_not_called()

    def test_a_stale_pattern_offer_is_dropped_not_recapped(self):
        from core import owner_presence as op
        self.assertIn("pattern_offer", op.EXPIRE_QUIETLY_SOURCES)
        self.assertFalse(op.KEEP_WHOLE_SOURCES & op.EXPIRE_QUIETLY_SOURCES)
        self.assertFalse(op.EXEMPT_SOURCES & op.EXPIRE_QUIETLY_SOURCES)

    def test_the_app_usage_phrase_is_the_one_with_no_action(self):
        import memory
        self.assertFalse(memory.offer_has_action(APP_USAGE))
        self.assertTrue(memory.offer_has_action(PLAY))
        self.assertTrue(memory.offer_has_action(""))
        pat = {"category": "app_usage", "target": "Claude", "dow": "Friday",
               "bucket": "evening", "hour": 20, "ratio": 0.5}
        self.assertFalse(memory.offer_has_action(memory._phrase_for(pat)),
                         "the real app-usage phrase must be the one dropped")


if __name__ == "__main__":
    unittest.main()
