"""core/opened_ledger.py - what JARVIS itself opened last, and the "close that
and open X instead" compound (S1, 2026-10-02).

Live (paraphrased): right after JARVIS put a search page on the main monitor
the owner said "close that and open <a streaming service> instead"; the brain
answered with ONE token, the open, and the close was dropped. Pure functions
and a thread-safe list; nothing real is opened or closed.

    python -m unittest tests.test_opened_ledger
"""
from __future__ import annotations

import unittest

from core import opened_ledger as L

# The live reply shape (the target is a placeholder).
LIVE_REPLY = ("[intent:confirmation] Very good, sir. "
              "[ACTION: open_on_monitor, main | example-tv.com]")


class LedgerTests(unittest.TestCase):
    def setUp(self):
        L.reset()
        self.addCleanup(L.reset)

    def test_newest_entry_is_that(self):
        L.note_opened("open_url", "https://a.example", hwnd=1, kind="tab", now=100.0)
        L.note_opened("open_on_monitor", "https://b.example", hwnd=2,
                      monitor="middle", now=110.0)
        e = L.last_opened(now=120.0)
        self.assertEqual((e.via, e.hwnd, e.kind, e.monitor),
                         ("open_on_monitor", 2, "window", "middle"))

    def test_an_old_entry_is_not_that(self):
        L.note_opened("open_url", "https://a.example", hwnd=1, now=100.0)
        self.assertIsNone(L.last_opened(now=100.0 + L.CLOSE_MAX_AGE_S + 1))
        self.assertIsNotNone(L.last_opened(now=100.0 + L.CLOSE_MAX_AGE_S - 1))

    def test_forget_reveals_the_one_before(self):
        a = L.note_opened("open_url", "https://a.example", hwnd=1, now=100.0)
        b = L.note_opened("open_url", "https://b.example", hwnd=2, now=101.0)
        L.forget(b)
        self.assertEqual(L.last_opened(now=102.0), a)
        L.forget(1)
        self.assertIsNone(L.last_opened(now=102.0))

    def test_the_same_window_recorded_twice_is_one_entry(self):
        L.note_opened("play_streaming", "https://a.example", hwnd=7, now=100.0)
        L.note_opened("play_streaming", "https://b.example", hwnd=7, now=101.0)
        L.forget(L.last_opened(now=102.0))
        self.assertIsNone(L.last_opened(now=102.0))

    def test_a_non_int_handle_is_not_kept(self):
        e = L.note_opened("open_on_monitor", "x", hwnd=object(), now=1.0)
        self.assertIsNone(e.hwnd)

    def test_bounded(self):
        for i in range(30):
            L.note_opened("open_url", f"https://{i}.example", hwnd=i, now=float(i))
        self.assertLessEqual(len(L._entries), L._MAX_ENTRIES)

    def test_describe(self):
        for target, want in (
                ("https://www.youtube.com/results?search_query=x", "YouTube page"),
                ("https://play.hbomax.com/search?q=x", "HBO Max page"),
                ("https://www.google.com/search?q=x", "Google page"),
                ("notepad", "notepad window"),
                ("notepad.exe", "notepad.exe window")):
            with self.subTest(target=target):
                e = L.Opened("x", target, None, "window", None, "", 0.0)
                self.assertEqual(L.describe(e), want)

    def test_is_web_target(self):
        for target, want in (("https://example.com/x", True), ("max.com", True),
                             ("www.netflix.com/search?q=x", True),
                             ("notepad", False), ("notepad.exe", False),
                             ("Spotify", False), ("", False), (None, False)):
            with self.subTest(target=target):
                self.assertIs(L.is_web_target(target), want)


class CloseThenOpenTests(unittest.TestCase):
    def test_compounds(self):
        for text in (
                "Jarvis, close that and open up the other service instead.",
                "close it, then open Netflix",
                "close that window and open the calculator",
                "close that. Open Hulu instead.",
                "can you close that out and play the show on Hulu",
                "Jarvis closed that and open Netflix",       # Parakeet's "closed"
                "close the tab and pull up my email"):
            with self.subTest(text=text):
                self.assertTrue(L.is_close_then_open(text))

    def test_not_compounds(self):
        for text in (
                "close that", "close that please",
                "close the YouTube window and open Netflix",   # named: his call
                "close Spotify and open Netflix",
                "I closed that and opened Netflix",            # narration
                "open Netflix", "", None):
            with self.subTest(text=text):
                self.assertFalse(L.is_close_then_open(text))

    def test_the_live_reply_gets_the_close_in_front_of_the_open(self):
        new, dropped = L.rewrite_close_then_open(LIVE_REPLY)
        self.assertEqual(new, "[intent:confirmation] Very good, sir. "
                              "[ACTION: close_last_opened] "
                              "[ACTION: open_on_monitor, main | example-tv.com]")
        self.assertEqual(dropped, [])

    def test_a_guessed_close_is_replaced(self):
        reply = ("Right away, sir. [ACTION: open_url, https://b.example] "
                 "[ACTION: close_window, Some Page]")
        new, dropped = L.rewrite_close_then_open(reply)
        self.assertEqual(new, "Right away, sir. [ACTION: close_last_opened] "
                              "[ACTION: open_url, https://b.example]")
        self.assertEqual(dropped, ["[ACTION: close_window, Some Page]"])

    def test_a_guessed_bulk_close_is_replaced(self):
        # Review 2026-10-03: the guessed-close list named a placeholder
        # "close_all_windows"; the real bulk close is close_all_windows_except.
        # "Close that and open X" never asks for every other window to go.
        reply = ("Right away, sir. [ACTION: close_all_windows_except, Claude] "
                 "[ACTION: open_url, https://b.example]")
        new, dropped = L.rewrite_close_then_open(reply)
        self.assertEqual(new, "Right away, sir. [ACTION: close_last_opened] "
                              "[ACTION: open_url, https://b.example]")
        self.assertEqual(dropped,
                         ["[ACTION: close_all_windows_except, Claude]"])

    def test_a_close_only_reply_closes_what_jarvis_opened(self):
        new, dropped = L.rewrite_close_then_open(
            "Done, sir. [ACTION: close_window, chrome]")
        self.assertEqual(new, "Done, sir. [ACTION: close_last_opened]")
        self.assertEqual(dropped, ["[ACTION: close_window, chrome]"])

    def test_unchanged_replies(self):
        for reply in ("Of course, sir.",
                      "[ACTION: close_last_opened] [ACTION: open_url, x.com]",
                      "[ACTION: get_time]", "", None):
            with self.subTest(reply=reply):
                self.assertEqual(L.rewrite_close_then_open(reply),
                                 (str(reply or ""), []))


if __name__ == "__main__":
    unittest.main()
