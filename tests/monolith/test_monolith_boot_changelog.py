"""3.0.26 (reliability-11) - BOOT_CHANGELOG_BACKGROUND: the release-notes
summary after a version bump is an LLM call; on the boot thread it kept
JARVIS deaf until the brain answered. On (the default) it runs on a daemon
thread and the line goes through the proactive queue; off, as before.

Everything external is mocked: no model, no audio, no queue file."""
from __future__ import annotations

import inspect
import threading
import unittest
from unittest import mock

from tests._monolith_harness import (
    MonolithGlobalsTestCase, load_monolith, requires_monolith)


@requires_monolith
class ChangelogBootTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _p(self, *a, **k):
        p = mock.patch.object(*a, **k)
        m = p.start()
        self.addCleanup(p.stop)
        return m

    def test_default_is_on_and_logged(self):
        import core.config as cfg
        self.assertIs(cfg.BOOT_CHANGELOG_BACKGROUND, True)
        self.assertIn("BOOT_CHANGELOG_BACKGROUND", self.bc._TURN_FLAG_KEYS)
        self.assertTrue(self.bc._changelog_background_on())

    def test_main_goes_through_the_helper_not_the_inline_llm_call(self):
        src = inspect.getsource(self.bc.main)
        self.assertIn("_changelog_announce(_last_ver, _cur_ver, "
                      "_latest_entry,", src)
        self.assertNotIn("boot_changelog_summary", src)

    def test_on_the_boot_thread_does_not_wait_for_the_llm(self):
        bc = self.bc
        release = threading.Event()
        done = threading.Event()

        def slow_llm(*a, **k):
            release.wait(10)
            return "Sir, since we last spoke, things changed."

        self._p(bc, "_llm_quick", slow_llm)
        self._p(bc, "BOOT_CHANGELOG_BACKGROUND", True, create=True)
        spoke = self._p(bc, "_speak")
        announce = self._p(bc, "proactive_announce",
                           side_effect=lambda *a, **k: done.set() or True)
        hist: list = []
        with mock.patch("builtins.print") as pr:
            bc._changelog_announce("1.0", "1.1", "- a change", hist)
            # returned although the LLM has not answered
            self.assertFalse(done.is_set())
            self.assertTrue(any("runs in the background" in str(c.args[0])
                                for c in pr.call_args_list))
            release.set()
            self.assertTrue(done.wait(5))
        spoke.assert_not_called()
        self.assertEqual(announce.call_count, 1)
        self.assertEqual(announce.call_args.kwargs["source"], "changelog")
        self.assertIn("since we last spoke",
                      announce.call_args.kwargs["history_text"])
        self.assertEqual(hist, [])      # the drainer records it once spoken

    def test_off_speaks_here_and_records_history(self):
        bc = self.bc
        self._p(bc, "_llm_quick", lambda *a, **k: "Sir, notes.")
        self._p(bc, "BOOT_CHANGELOG_BACKGROUND", False, create=True)
        spoke = self._p(bc, "_speak")
        announce = self._p(bc, "proactive_announce")
        hist: list = []
        bc._changelog_announce("1.0", "1.1", "- a change", hist)
        spoke.assert_called_once_with("Sir, notes.")
        announce.assert_not_called()
        self.assertEqual(hist, [{"role": "assistant",
                                 "content": "Sir, notes."}])

    def test_nothing_to_say_says_nothing(self):
        bc = self.bc
        self._p(bc, "BOOT_CHANGELOG_BACKGROUND", False, create=True)
        llm = self._p(bc, "_llm_quick", lambda *a, **k: "")
        spoke = self._p(bc, "_speak")
        hist: list = []
        bc._changelog_announce("1.0", "1.1", "", hist)     # empty entry
        bc._changelog_announce("1.0", "1.1", "- x", hist)  # LLM gave ''
        spoke.assert_not_called()
        self.assertEqual(hist, [])

    def test_a_failing_llm_never_raises(self):
        bc = self.bc
        self._p(bc, "BOOT_CHANGELOG_BACKGROUND", False, create=True)

        def boom(*a, **k):
            raise RuntimeError("brain down")

        self._p(bc, "_llm_quick", boom)
        spoke = self._p(bc, "_speak")
        bc._changelog_announce("1.0", "1.1", "- x", [])
        spoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
