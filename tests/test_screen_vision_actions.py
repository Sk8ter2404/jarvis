"""The 2026-10-05 screen-vision actions through core.actions (the monolith
faked): note_for_claude, screen_memory, forget_screen, undo_click,
click_on_screen; see_screen's text-first read (core.screen_digest) and
open_url's page wait. Nothing real is opened, read or clicked.

    python -m unittest tests.test_screen_vision_actions
"""
from __future__ import annotations

import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

import core.actions as A
from core import config as cfg
from core import dev_notes as DN
from core import screen_digest as SD
from core.failure_markers import FAILURE_MARKERS
from tests import _screen_fakes as F

LIVE_NOTE_1 = ("Jarvis, also tell Claude to go ahead and do some research on "
               "the screen vision. Your screen vision isn't working properly.")
LIVE_NOTE_2 = ("Jarvis, tell Claude that I want it to be able to watch what "
               "you see, so it can learn.")


def _failure(text) -> bool:
    low = str(text).lower()
    return any(m in low for m in FAILURE_MARKERS)


class _Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="svact_")
        self.addCleanup(shutil.rmtree, self.td, True)
        env = mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.td})
        env.start()
        self.addCleanup(env.stop)
        self.bc = mock.Mock()
        self.bc._turn_user_text.return_value = ""
        self.bc._parse_monitor_prefix.side_effect = lambda q: (None, q)
        self.bc.screenshot_privacy_block_reason.return_value = None
        self.bc._see_screen_budget_state = types.SimpleNamespace(used=0)
        self.bc.SEE_SCREEN_BUDGET_PER_INTENT = 3
        self.bc._is_self_close_attempt.return_value = False
        self.bc._STREAMING_SERVICES = {}
        for p in (mock.patch.object(A, "_bc", return_value=self.bc),
                  mock.patch.object(A, "_loaded_bc", return_value=self.bc),
                  mock.patch.object(cfg, "MONITORS", F.MONITORS),
                  mock.patch.object(cfg, "VISION_TRACE", "off", create=True),
                  mock.patch("builtins.print")):
            p.start()
            self.addCleanup(p.stop)

    def said(self, text):
        self.bc._turn_user_text.return_value = text


class NoteForClaudeTests(_Base):
    def test_the_note_is_saved_and_the_line_never_claims_a_relay(self):
        self.said(LIVE_NOTE_1)
        out = A._act_note_for_claude("research the screen vision")
        self.assertEqual(out, DN.SPOKEN_LINE)
        self.assertNotIn("relay", out.lower())
        self.assertFalse(_failure(out), "the line must be spoken verbatim")
        rec = DN.read_notes()[-1]
        self.assertEqual(rec["utterance"], LIVE_NOTE_1)
        self.assertEqual(rec["note"], "research the screen vision")
        self.assertTrue(os.path.exists(os.path.join(self.td,
                                                    "notes_for_claude.jsonl")))

    def test_the_note_comes_from_his_words_when_the_arg_is_empty(self):
        self.said(LIVE_NOTE_2)
        A._act_note_for_claude("")
        self.assertIn("watch", DN.read_notes()[-1]["note"])

    def test_append_only_with_context(self):
        DN.set_context_provider(lambda: {"session_log": "s.log", "log_offset": 9,
                                         "last_results": ["a", "b"],
                                         "trace_ids": ["vt-1"]})
        self.addCleanup(DN.set_context_provider, None)
        DN.add_note("one", "u1")
        DN.add_note("two", "u2")
        notes = DN.read_notes()
        self.assertEqual([n["note"] for n in notes], ["one", "two"])
        self.assertEqual(notes[-1]["session_log"], "s.log")
        self.assertEqual(notes[-1]["last_5_trace_ids"], ["vt-1"])

    def test_no_network(self):
        src = open(DN.__file__, encoding="utf-8").read()
        for word in ("requests", "urllib.request", "socket", "http.client"):
            self.assertNotIn(f"import {word}", src)


class ScreenMemoryActionTests(_Base):
    def test_controls_route_to_the_watcher(self):
        from core import screen_memory as SW
        with mock.patch.object(SW, "pause", return_value="paused") as pa, \
                mock.patch.object(SW, "unpause", return_value="resumed") as re_, \
                mock.patch.object(SW, "status_line", return_value="st") as st, \
                mock.patch.object(SW, "exclude_foreground",
                                  return_value="ex") as ex, \
                mock.patch.object(SW, "exclude_app", return_value="app") as ea:
            self.assertEqual(A._act_screen_memory("pause 10"), "paused")
            pa.assert_called_with(10.0)
            self.assertEqual(A._act_screen_memory("pause"), "paused")
            pa.assert_called_with(None)
            self.assertEqual(A._act_screen_memory("unpause"), "resumed")
            self.assertEqual(A._act_screen_memory("status"), "st")
            self.assertEqual(A._act_screen_memory("exclude_this"), "ex")
            self.assertEqual(A._act_screen_memory("exclude Discord"), "app")
            ea.assert_called_with("Discord")
        self.assertTrue(re_.called and st.called and ex.called)

    def test_forget_spans(self):
        from core import screen_memory as SW
        with mock.patch.object(SW, "forget", return_value="done") as fg:
            A._act_forget_screen("60m")
            self.assertEqual(fg.call_args.args[0], {"seconds": 3600.0})
            A._act_forget_screen("today")
            self.assertEqual(fg.call_args.args[0], {"today": True})
            A._act_forget_screen("all")
            self.assertEqual(fg.call_args.args[0], {"all": True})


class ClickActionTests(_Base):
    def test_click_on_screen_passes_his_words_and_returns_the_line(self):
        from core import grounded_click as G
        self.said("click that MrBeast video")
        with mock.patch.object(G, "run_bounded", return_value=G.Result(
                "Playing 'X' on the middle monitor, sir.", G.VERIFIED)) as run:
            out = A._act_click_on_screen("that MrBeast video")
        self.assertEqual(out, "Playing 'X' on the middle monitor, sir.")
        run.assert_called_once_with("that MrBeast video",
                                    said="click that MrBeast video",
                                    mode="click")

    def test_undo_click(self):
        from core import grounded_click as G
        with mock.patch.object(G, "undo", return_value=G.Result(
                "Back on 'Home', sir.", G.VERIFIED)) as u:
            self.assertEqual(A._act_undo_click("other"), "Back on 'Home', sir.")
            self.assertTrue(u.call_args.kwargs["other"])
            A._act_undo_click("")
            self.assertFalse(u.call_args.kwargs["other"])


class SeeScreenTextFirstTests(_Base):
    def setUp(self):
        super().setUp()
        self.home = F.FakeWindow(101, "home_dark", "middle")
        self.fake = F.FakeBackend([self.home], fg=101)
        p = mock.patch.object(A, "_screen_digest_backend", [self.fake])
        p.start()
        self.addCleanup(p.stop)

    def test_a_reading_question_is_answered_from_the_page_text(self):
        # Live 00:28:01: four monitor shots -> "several video thumbnails and
        # categories", not one title.
        self.said("what videos are on my screen")
        out = A._act_see_screen("what videos are on screen?")
        self.assertIn("I Survived 7 Days In An Abandoned City", out)
        self.assertIn("MrBeast", out)
        self.assertIn("[middle]", out)
        self.bc.ask_vision.assert_not_called()
        self.bc.ask_vision_multi.assert_not_called()
        self.bc.take_all_monitor_screenshots.assert_not_called()
        self.assertEqual(self.bc._see_screen_budget_state.used, 0)

    def test_a_bare_url_question_reads_the_page(self):
        self.said("open YouTube")
        out = A._act_see_screen("https://www.youtube.com")
        self.assertIn("videos:", out)
        self.bc.ask_vision_multi.assert_not_called()

    def test_a_visual_question_gets_one_look_with_the_text_beside_it(self):
        self.said("what colour is the first thumbnail")
        self.bc.take_all_monitor_screenshots.return_value = {"middle": b"P"}
        self.bc.ask_vision_multi.return_value = "red"
        A._act_see_screen("what colour is the first thumbnail")
        q = self.bc.ask_vision_multi.call_args.args[0]
        self.assertIn("quote titles exactly", q)
        self.assertIn("I Survived 7 Days", q)

    def test_nothing_readable_falls_back_to_vision(self):
        p = mock.patch.object(A, "_screen_digest_backend",
                              [F.FakeBackend([])])
        p.start()
        self.addCleanup(p.stop)
        self.bc.take_all_monitor_screenshots.return_value = {"middle": b"P"}
        self.bc.ask_vision_multi.return_value = "a desktop"
        out = A._act_see_screen("what's on screen")
        self.assertEqual(out, "a desktop")

    def test_a_private_window_is_listed_never_read(self):
        bank = F.FakeWindow(102, "home_dark", "top",
                            title="Accounts - BankingSite - Google Chrome")
        fake = F.FakeBackend([bank, self.home], fg=101)
        with mock.patch.object(cfg, "SCREENSHOT_PRIVACY_BLOCKLIST",
                               ["bankingsite"]):
            d = SD.digest("overview", backend=fake)
        self.assertIn("(a private window)", d["text"])
        self.assertNotIn("BankingSite", d["text"])


class OpenUrlWaitTests(_Base):
    def test_the_wait_returns_as_soon_as_the_page_is_named(self):
        self.bc._strip_bidi_and_nbsp = lambda s: s or ""
        self.bc._BROWSER_CHROME_SUFFIXES = (" - google chrome",)
        seq = [(1, "Inbox - Mail - Google Chrome", (0, 0, 2560, 1400)),
               (2, "youtube.com - Google Chrome", (0, 0, 2560, 1400)),
               (2, "YouTube - Google Chrome", (0, 0, 2560, 1400))]
        self.bc._read_focused_window.side_effect = seq + [seq[-1]] * 50
        with mock.patch.object(A.webbrowser, "open"), \
                mock.patch.object(A.time, "sleep") as sl:
            out = A._act_open_url("https://www.youtube.com")
        self.assertEqual(sl.call_count, 2)       # the loading title is skipped
        self.assertIn("on the middle monitor", out)
        self.assertIn("page title 'YouTube'", out)
        self.assertNotIn("see_screen", out)



class RegistrationTests(unittest.TestCase):
    """The five 2026-10-05 actions are registered in the monolith's ACTIONS
    dict, spoken verbatim, and resolvable by name in core.actions (the
    action index's "tested" column reads these literals)."""

    NAMES = ("click_on_screen", "undo_click", "note_for_claude",
             "screen_memory", "forget_screen")

    def test_registered_verbatim_and_defined(self):
        import re
        mono = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "bobert_companion.py")
        with open(mono, encoding="utf-8") as f:
            src = f.read()
        for name in self.NAMES:
            self.assertRegex(src, rf'\n\s+"{name}":\s+_act_{name},', name)
            self.assertTrue(callable(getattr(A, f"_act_{name}", None)), name)
        start = re.search(r"^SPEAK_RESULT_VERBATIM_ACTIONS\b[^\n]*=\s*\{",
                          src, re.M).start()
        block = src[start:src.index("\n}\n", start)]
        for name in self.NAMES:
            self.assertTrue(re.search(rf'"{name}"', block), name)

    def test_the_new_memory_action_does_not_reuse_the_wellness_name(self):
        # skills/screen_watch.py (the stare nudge) owns screen_watch_status;
        # screen memory is a different feature with its own name.
        self.assertFalse(hasattr(A, "_act_screen_watch"))


if __name__ == "__main__":
    unittest.main()
