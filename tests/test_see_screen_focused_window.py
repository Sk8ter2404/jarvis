"""see_screen reads "this page / this window" from the focused window (NEW #11, 2026-10-02).

THE LIVE EVIDENCE (session_2026-10-01_19-43-10.log, 20:45-21:00): "read this
page for me and see if there's any issues" reached see_screen, which logged
"Capturing all 4 monitors" and sent four 1024-px monitor shots in one call to
local vision - each page shrunk to a corner of a composite, a poor way to
read text. v2.0.159 (460380f) fixed only the question text.

  * the owner's words name a page / window / tab / article / document ->
    capture ONLY the focused window, through the existing focused-window
    capture (privacy-gated), at full resolution for a 2560-px monitor, and
    ask single-image vision. The cache entry says "the focused window".
  * anything else (a general "what's on my screen", a named monitor) keeps
    the old capture.
  * the focused capture fails, or the focused window is JARVIS's own (a
    typed request from the dashboard) -> the all-monitor capture, as before.

Light tier: the monolith is a Mock (tests/test_actions_sec3.py's harness).

    python -m unittest tests.test_see_screen_focused_window
"""
from __future__ import annotations

import types
import unittest
from unittest import mock

from core import actions as A
from tests.test_actions_sec3 import _base_bc, _patch_bc

PAGE_PNG = b"\x89PNG-focused-window"
SAID = "Jarvis read this page for me and see if there's any issues"


class SeeScreenFocusedWindowTests(unittest.TestCase):

    def _bc(self, user_text, focused_png=PAGE_PNG, title="Some Article - Browser"):
        bc = _base_bc()
        bc._see_screen_budget_state = types.SimpleNamespace(used=0)
        bc.SEE_SCREEN_BUDGET_PER_INTENT = 3
        bc._parse_monitor_prefix.side_effect = lambda q: (None, q)
        bc._turn_user_text.return_value = user_text
        bc._capture_focused_window_png.return_value = focused_png
        bc._focused_window_state = {"title": title}
        bc._read_focused_window.return_value = (1, title, (0, 0, 2576, 1408))
        bc.take_all_monitor_screenshots.return_value = {"m": b"PNG"}
        bc.ask_vision_multi.return_value = "composite answer"
        bc.ask_vision.return_value = "page answer"
        return bc

    def _run(self, bc, raw="read the page and list any issues"):
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", {"m": (0, 0, 1, 1)}):
            return A._act_see_screen(raw)

    def test_this_page_reads_only_the_focused_window(self):
        bc = self._bc(SAID)
        out = self._run(bc)
        self.assertEqual(out, "page answer")
        bc._capture_focused_window_png.assert_called_once()
        bc.take_all_monitor_screenshots.assert_not_called()
        bc.ask_vision_multi.assert_not_called()
        self.assertIs(bc.ask_vision.call_args[0][1], PAGE_PNG)

    def test_the_capture_is_full_resolution_for_a_2560_monitor(self):
        bc = self._bc(SAID)
        self._run(bc)
        call = bc._capture_focused_window_png.call_args
        self.assertIsNotNone(call, "the focused window was never captured")
        kw = call.kwargs
        self.assertGreaterEqual(
            kw.get("max_dim", 1568), 2576,
            "a maximised 2560-px window would be downscaled to the 1568 glance "
            "size - not full resolution")

    def test_the_cache_entry_names_the_focused_window(self):
        bc = self._bc(SAID)
        self._run(bc)
        args = bc._push_screen_context.call_args[0]
        self.assertEqual(args[0], "the focused window")
        self.assertEqual(args[3], {"the focused window": PAGE_PNG})

    def test_window_tab_article_and_document_phrasings(self):
        for said in ("what does this window say", "summarize this article",
                     "is there a typo on this tab", "check the document for me",
                     "proofread the email I'm writing",
                     "what's on this web page"):
            with self.subTest(said=said):
                bc = self._bc(said)
                self._run(bc, "check it")
                bc._capture_focused_window_png.assert_called_once()
                bc.take_all_monitor_screenshots.assert_not_called()

    def test_a_general_screen_question_keeps_every_monitor(self):
        for said in ("what's on my screen", "look at my screens",
                     "what am I looking at", "describe the desktop"):
            with self.subTest(said=said):
                bc = self._bc(said)
                out = self._run(bc, "describe")
                self.assertEqual(out, "composite answer")
                bc._capture_focused_window_png.assert_not_called()
                bc.take_all_monitor_screenshots.assert_called_once()

    def test_a_failed_focused_capture_falls_back_to_every_monitor(self):
        bc = self._bc(SAID, focused_png=None)
        out = self._run(bc)
        self.assertEqual(out, "composite answer")
        bc.take_all_monitor_screenshots.assert_called_once()

    def test_jarvis_own_window_in_focus_falls_back(self):
        # Typed from the dashboard: the focused window is JARVIS itself, not
        # the page he means.
        bc = self._bc(SAID, title="JARVIS Dashboard - Browser")
        out = self._run(bc)
        self.assertEqual(out, "composite answer")
        bc._capture_focused_window_png.assert_not_called()

    def test_no_owner_words_uses_the_question(self):
        bc = self._bc("")
        self._run(bc, "read the text on this page")
        bc._capture_focused_window_png.assert_called_once()

    def test_a_named_monitor_is_untouched(self):
        bc = self._bc(SAID)
        bc._parse_monitor_prefix.side_effect = lambda q: ("left", "what is here")
        bc.take_screenshot.return_value = b"PNG"
        with _patch_bc(bc), \
                mock.patch("core.config.MONITORS", {"left": (0, 0, 1, 1)}):
            A._act_see_screen("monitor:left | what is here")
        bc._capture_focused_window_png.assert_not_called()
        bc.take_screenshot.assert_called_once_with(monitor="left")

    def test_budget_still_counts_the_focused_capture(self):
        bc = self._bc(SAID)
        self._run(bc)
        self.assertEqual(bc._see_screen_budget_state.used, 1)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
