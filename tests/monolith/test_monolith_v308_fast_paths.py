"""v3.0.8 H2: "what can you do" is short, and a yes gets the rest.

Live 2026-10-28 19:47 (3.0.6 canary) the fixed overview was right but long:
every backed skill, about 20 items, spoken in one breath. It now names the
CAPABILITY_SHORT_MAX most useful backed skills and ends on "Want the full
list, sir?"; a yes to that, or "what else can you do" right after it, says the
rest. Through the REAL entry point a live turn uses
(bobert_companion._run_voice_shortcuts -> core.fast_paths.match, with the
real conversation_history), the LLM booby-trapped. Neutral fixtures.

    python -m unittest tests.monolith.test_monolith_v308_fast_paths
"""
from __future__ import annotations

import re
import unittest
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_diag_fixes import _ShortcutBase

# Every action that backs a phrase of the overview (core.fast_paths).
_ALL_BACKING = (
    "set_timer", "weather_briefing", "calendar_today", "draft_reply",
    "play_music", "control_smart_home", "check_print", "system_pulse",
    "server_status", "see_screen", "look_around", "web_search",
    "note_for_claude",
)
_OFFER = "Want the full list, sir?"


def _items(line: str) -> int:
    """How many skills a spoken overview line lists."""
    body = re.sub(r"^(?:Quite a bit, sir: I can|I can also) ", "", line)
    body = body.split(". ")[0]
    return len([p for p in re.split(r",\s*(?:and\s+)?", body) if p.strip()])


@requires_monolith
class ShortCapabilityOverview(_ShortcutBase):
    def setUp(self):
        super().setUp()
        import core.config as cfg
        acts = {n: mock.Mock(return_value="") for n in _ALL_BACKING}
        self._p(self.bc, "ACTIONS", acts)
        self._p(cfg, "SERVER_WATCH_HOST", "nas.example", create=True)
        self._p(cfg, "CAPABILITIES_SHORT_OVERVIEW", True, create=True)

    def test_live_question_is_short_and_ends_on_the_offer(self):
        handled, log = self._turn("What can you do for me?")
        self.assertTrue(handled)
        self.assertIn("[fast-path] capabilities", log)
        said = self.spoken[0]
        self.assertTrue(said.endswith(_OFFER), said)
        self.assertGreaterEqual(_items(said), 6)
        self.assertLessEqual(_items(said), 8)
        for phrase in ("set timers and reminders", "tell you the weather",
                       "search the web and your own files"):
            self.assertIn(phrase, said)
        self.assertNotIn("pass a note to Claude", said)
        self.llm.assert_not_called()

    def test_a_yes_gets_the_rest(self):
        self._turn("What can you do for me?")
        first = self.spoken[0]
        for yes in ("Yes.", "Jarvis, yes please.", "the full list", "go on"):
            with self.subTest(yes=yes):
                del self.spoken[1:]
                # Each answer comes straight after the overview.
                del self.bc.conversation_history[2:]
                handled, log = self._turn(yes)
                self.assertTrue(handled, f"{yes!r} was not answered")
                self.assertIn("[fast-path] capabilities-more", log)
                rest = self.spoken[1]
                self.assertTrue(rest.startswith("I can also "), rest)
                self.assertIn("pass a note to Claude", rest)
                self.assertIn("keep an eye on the 3D printer", rest)
                # Nothing said twice.
                self.assertNotIn("tell you the weather", rest)
                self.assertNotIn(_OFFER, rest)
                self.assertNotIn("pass a note to Claude", first)
        self.llm.assert_not_called()

    def test_what_else_right_after_gives_the_rest(self):
        self._turn("what can you do")
        handled, log = self._turn("what else can you do")
        self.assertTrue(handled)
        self.assertIn("[fast-path] capabilities-more", log)
        self.assertTrue(self.spoken[1].startswith("I can also "))

    def test_a_yes_without_the_offer_is_not_claimed(self):
        handled, log = self._turn("yes")
        self.assertFalse(handled)
        self.assertNotIn("[fast-path]", log)

    def test_a_no_to_the_offer_is_not_claimed(self):
        self._turn("what can you do")
        handled, _ = self._turn("no thanks")
        self.assertFalse(handled)

    def test_setting_off_is_the_whole_list(self):
        import core.config as cfg
        self._p(cfg, "CAPABILITIES_SHORT_OVERVIEW", False, create=True)
        handled, _ = self._turn("What can you do for me?")
        self.assertTrue(handled)
        said = self.spoken[0]
        self.assertNotIn(_OFFER, said)
        self.assertIn("pass a note to Claude", said)
        self.assertEqual(_items(said), 13)   # every phrase of the overview
        handled, _ = self._turn("yes")
        self.assertFalse(handled)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
