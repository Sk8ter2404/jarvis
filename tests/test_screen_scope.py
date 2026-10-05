"""core.screen_scope - which windows a spoken "click that X" may look in
(2026-10-05). A fake enumerator only: no real window is ever listed.

  * JARVIS's own windows (its process tree, terminals / python hosts, its
    HUD / console titles) are never in scope, and visible_windows() drops
    them unless asked;
  * a monitor named in the OWNER's words is a hard filter; the model's
    monitor, the ledger window and the foreground are soft priors;
  * the top TWO windows per monitor (a small window over a maximised
    browser must not hide the page);
  * a browser showing JARVIS's loopback dashboard is dropped.

    python -m unittest tests.test_screen_scope
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from core import config as cfg
from core import screen_scope as S
from tests import _screen_fakes as F


def _w(hwnd, title, process="chrome.exe", monitor="middle", pid=None, z=0):
    return S.Win(hwnd=hwnd, title=title, process=process,
                 pid=pid if pid is not None else 5000 + hwnd,
                 rect=F.MONITORS[monitor], monitor=monitor, z=z)


class _Base(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(cfg, "MONITORS", F.MONITORS, create=True)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(S.set_enumerator, None)

    def enum(self, wins):
        S.set_enumerator(lambda: list(wins))


class VisibleWindowsTests(_Base):
    def test_jarvis_windows_are_dropped_unless_asked(self):
        self.enum([
            _w(1, "Home - YouTube - Google Chrome"),
            _w(2, "JARVIS HUD overlay", process="pythonw.exe"),
            _w(3, "Windows PowerShell", process="WindowsTerminal.exe"),
            _w(4, "Settings", process="python.exe", pid=os.getpid()),
            _w(5, "Untitled - Notepad", process="notepad.exe"),
        ])
        self.assertEqual([w.hwnd for w in S.visible_windows()], [1, 5])
        everything = S.visible_windows(include_jarvis=True)
        self.assertEqual([w.hwnd for w in everything], [1, 2, 3, 4, 5])
        self.assertEqual([w.hwnd for w in everything if w.jarvis], [2, 3, 4])

    def test_a_test_process_never_enumerates_real_windows(self):
        # tests/__init__.py sets JARVIS_NO_SCREEN_READ=1: with no injected
        # enumerator the real EnumWindows is never reached.
        S.set_enumerator(None)
        with mock.patch.object(S, "_win32_windows",
                               side_effect=AssertionError("real enum")):
            self.assertEqual(S.visible_windows(), [])


class ScopeTests(_Base):
    def test_owner_named_monitor_is_a_hard_filter(self):
        wins = [_w(1, "A - Chrome", monitor="top", z=0),
                _w(2, "B - Chrome", monitor="middle", z=1)]
        sc = S.scope_for("click the video on the middle monitor",
                         windows=wins)
        self.assertEqual([w.hwnd for w in sc.windows], [2])
        self.assertEqual(sc.hard_monitor, "middle")

    def test_model_monitor_is_only_a_prior(self):
        wins = [_w(1, "A - Chrome", monitor="top", z=0),
                _w(2, "B - Chrome", monitor="middle", z=1)]
        sc = S.scope_for("click that video", model_monitor="top",
                         windows=wins)
        self.assertEqual({w.hwnd for w in sc.windows}, {1, 2})
        self.assertGreater(sc.priors.get(1, 0), 0)

    def test_two_windows_per_monitor(self):
        wins = [_w(1, "Untitled - Notepad", process="notepad.exe", z=0),
                _w(2, "Home - YouTube - Google Chrome", z=1),
                _w(3, "Third - Chrome", z=2)]
        sc = S.scope_for("click that video", windows=wins)
        self.assertEqual([w.hwnd for w in sc.windows], [1, 2])

    def test_jarvis_and_dashboard_windows_are_never_in_scope(self):
        wins = [_w(1, "JARVIS console", process="python.exe", z=0)._replace(
                    jarvis=True),
                _w(2, "JARVIS - Google Chrome", z=1),
                _w(3, "Home - YouTube - Google Chrome", monitor="top", z=2)]
        urls = {2: "http://127.0.0.1:8766/", 3: "https://www.youtube.com/"}
        sc = S.scope_for("click that video", windows=wins,
                         url_of=lambda h: urls.get(h, ""))
        self.assertEqual([w.hwnd for w in sc.windows], [3])

    def test_ledger_and_foreground_come_first(self):
        wins = [_w(1, "A - Chrome", monitor="top", z=0),
                _w(2, "B - Chrome", monitor="middle", z=1),
                _w(3, "C - Chrome", monitor="left", z=2)]
        sc = S.scope_for("click that video", ledger_hwnd=3,
                         foreground_hwnd=2, windows=wins)
        self.assertEqual([w.hwnd for w in sc.windows][:2], [3, 2])


if __name__ == "__main__":
    unittest.main()
