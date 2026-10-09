"""core/screen_timeline.py - the text record of what was on screen
(2026-10-05): FTS search, retention, forget, redaction, query stripping.

    python -m unittest tests.test_screen_timeline
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import time
import unittest

from core import screen_timeline as T


class _Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="tl_")
        self.addCleanup(shutil.rmtree, self.td, True)
        self.tl = T.Timeline(os.path.join(self.td, "screen_timeline.db"))
        self.addCleanup(self.tl.close)


class CleanUrlTests(unittest.TestCase):
    def test_query_is_stripped_except_what_names_the_page(self):
        self.assertEqual(T.clean_url("https://www.youtube.com/watch?v=abc&"
                                     "t=42&si=TRACKER&pp=x#frag"),
                         "https://www.youtube.com/watch?v=abc&t=42")
        self.assertEqual(T.clean_url("https://www.google.com/search?q=cats&"
                                     "sca_esv=1&ei=2"),
                         "https://www.google.com/search?q=cats")
        self.assertEqual(T.clean_url("https://mail.example.com/?token=SECRET"),
                         "https://mail.example.com/")
        self.assertEqual(T.clean_url(""), "")


class StoreTests(_Base):
    def test_fts_query_with_times_and_monitor(self):
        now = time.time()
        self.tl.add_now(ts=now - 600, monitor="middle", hwnd=1,
                        process="chrome.exe", title="Home - YouTube",
                        url="https://www.youtube.com/", source="uia",
                        text="I Survived 7 Days In An Abandoned City\nMrBeast")
        self.tl.add_now(ts=now - 30, monitor="top", hwnd=2,
                        process="chrome.exe", title="News", source="title",
                        text="opened")
        rows = self.tl.query(text="abandoned city")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["monitor"], "middle")
        self.assertEqual(self.tl.query(text="abandoned", monitor="top"), [])
        self.assertEqual(len(self.tl.query(since=now - 60)), 1)
        self.assertEqual(len(self.tl.query(sources=("uia",))), 1)
        self.assertEqual(self.tl.count(), 2)

    def test_prefix_words(self):
        self.tl.add_now(title="x", text="Veritasium bridges", source="uia")
        self.assertTrue(self.tl.query(text="veritas"))

    def test_secret_lines_are_dropped(self):
        self.tl.add_now(title="Settings", source="uia",
                        text="Profile\npassword: hunter2\nTheme: dark")
        row = self.tl.query(text="theme")[0]
        self.assertNotIn("hunter2", row["text"])
        self.assertIn("Theme: dark", row["text"])

    def test_an_empty_row_is_not_stored(self):
        self.assertFalse(self.tl.add_now(source="uia", text="   "))

    def test_pragmas(self):
        self.tl.add_now(title="x", text="y")
        con = sqlite3.connect(self.tl.path)
        try:
            self.assertEqual(con.execute("PRAGMA auto_vacuum").fetchone()[0], 2)
            self.assertEqual(con.execute("PRAGMA journal_mode").fetchone()[0],
                             "wal")
        finally:
            con.close()

    def test_the_writer_thread(self):
        for i in range(20):
            self.assertTrue(self.tl.add(title=f"t{i}", text=f"line {i}",
                                        source="uia"))
        self.assertTrue(self.tl.flush(5))
        self.assertEqual(self.tl.count(), 20)

    def test_close_stops_the_writer_after_its_queued_rows(self):
        # rel-182 (2026-10-09): every test on a fresh data dir left one
        # writer thread waiting forever (32 at once in one suite).
        for i in range(5):
            self.tl.add(title=f"t{i}", text=f"line {i}", source="uia")
        th = self.tl._thread
        self.assertTrue(th is not None and th.is_alive())
        self.tl.close()
        self.assertFalse(th.is_alive())
        self.assertEqual(self.tl.count(), 5)          # they landed first
        self.assertTrue(self.tl.add(title="again", text="later", source="uia"))
        self.assertTrue(self.tl.flush(5))
        self.assertEqual(self.tl.count(), 6)          # a new writer took it
        self.tl.close()
        self.tl.close()                               # idempotent

    def test_a_new_data_dir_closes_the_replaced_timeline(self):
        from unittest import mock
        other = tempfile.mkdtemp(prefix="tl2_")
        self.addCleanup(shutil.rmtree, other, True)
        self.addCleanup(T._singleton.__setitem__, "tl", T._singleton["tl"])
        with mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.td}):
            first = T.get()
            self.addCleanup(first.close)
            first.add(title="a", text="one", source="uia")
            th = first._thread
        with mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": other}):
            second = T.get()
            self.addCleanup(second.close)
        self.assertIsNot(first, second)
        th.join(5)
        self.assertFalse(th.is_alive())
        self.assertEqual(first.count(), 1)            # its row still landed


class ForgetAndRetentionTests(_Base):
    def test_forget_a_span(self):
        now = time.time()
        for i in range(5):
            self.tl.add_now(ts=now - 7200 + i, title="old", text=f"a{i}")
        for i in range(3):
            self.tl.add_now(ts=now - 60 + i, title="new", text=f"b{i}")
        self.assertEqual(self.tl.forget(since=now - 3600, until=now), 3)
        self.assertEqual(self.tl.count(), 5)

    def test_forget_one_window(self):
        now = time.time()
        self.tl.add_now(ts=now - 10, hwnd=7, title="secret chat", text="x")
        self.tl.add_now(ts=now - 10, hwnd=8, title="other", text="y")
        self.assertEqual(self.tl.forget(since=now - 300, hwnd=7), 1)
        self.assertEqual([r["hwnd"] for r in self.tl.query()], [8])

    def test_prune_by_days(self):
        now = time.time()
        self.tl.add_now(ts=now - 9 * 86400, title="old", text="x")
        self.tl.add_now(ts=now - 60, title="new", text="y")
        out = self.tl.prune(now=now, days=7, max_mb=200)
        self.assertEqual(out["age"], 1)
        self.assertEqual([r["title"] for r in self.tl.query()], ["new"])

    def test_prune_by_size(self):
        now = time.time()
        for i in range(400):
            self.tl.add_now(ts=now - 1000 + i, title=f"t{i}",
                            text="lorem ipsum " * 40)
        before = self.tl.count()
        cap = self.tl.size_mb() * 0.6
        out = self.tl.prune(now=now, days=7, max_mb=cap)
        self.assertGreater(out["size"], 0)
        self.assertLess(self.tl.count(), before)
        self.assertGreater(self.tl.count(), 0)
        self.assertLessEqual(self.tl.size_mb(), cap + 0.01)
        # oldest first: the newest row survives
        self.assertTrue(self.tl.query(text="t399"))

    def test_prune_all_covers_the_trace_too(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"JARVIS_DATA_DIR": self.td}):
            res = T.prune_all()
        self.assertIn("timeline", res)
        self.assertIn("trace", res)


if __name__ == "__main__":
    unittest.main()
