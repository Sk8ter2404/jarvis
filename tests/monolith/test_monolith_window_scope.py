"""The monolith's window-name lookup goes through core.window_scope (2026-10-03).

Live 17:22 the owner asked JARVIS to close every window but the Claude app.
The brain answered with minimize_window on one of his folders, the JARVIS
HUD, the JARVIS Reticle, a media player, "Program Manager" and "Windows Input
Experience". _find_windows_by_title - the lookup behind close_window,
minimize_window, focus_window, move_window_to_monitor and close_window's
pushback count - matched every one of them.

Now it never returns a shell window, and returns one of JARVIS's own windows
only when the owner's own words this turn name it ("minimize the HUD").
Every window and Win32 read is faked; nothing real is minimized.

    python -m unittest tests.monolith.test_monolith_window_scope
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from unittest import mock

from core import window_scope as ws
from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

# What the brain answered live, titles generic.
IMPROVISED = ("[intent:confirmation] Certainly, sir. "
              "[ACTION: minimize_window, Downloads - File Explorer] "
              "[ACTION: minimize_window, JARVIS HUD] "
              "[ACTION: minimize_window, JARVIS Reticle] "
              "[ACTION: minimize_window, Program Manager]")


class _Win:
    def __init__(self, title, hwnd):
        self.title = title
        self._hWnd = hwnd
        self.width, self.height = 900, 700
        self.minimized = self.closed = False

    def minimize(self):
        self.minimized = True

    def close(self):
        self.closed = True


@requires_monolith
class FindWindowsScopeTests(_Base):
    def setUp(self):
        super().setUp()
        self.hud = _Win("JARVIS HUD", 0x50)
        self.files = _Win("Downloads - File Explorer", 0x51)
        self.desk = _Win("Program Manager", 0x52)
        self.reticle = _Win("JARVIS Reticle", 0x53)
        wins = [self.hud, self.files, self.desk, self.reticle]
        pids = {0x50: 8001, 0x51: 700, 0x52: 701, 0x53: 8002}
        classes = {0x52: "progman"}
        for name, fn in (
                ("probe", lambda w: ws.WindowFacts(
                    w._hWnd, pids.get(w._hWnd), classes.get(w._hWnd, ""),
                    False)),
                ("own_pids", lambda: frozenset({8000, 8001, 8002})),
                ("own_window_handles", lambda: frozenset())):
            self._p(ws, name, side_effect=fn)
        fake = types.SimpleNamespace(getAllWindows=lambda: list(wins))
        p = mock.patch.dict(sys.modules, {"pygetwindow": fake})
        p.start()
        self.addCleanup(p.stop)
        self.owner = self._p(self.bc, "_turn_user_text",
                             return_value="Jarvis, close every window except "
                                          "for Claude")

    def test_the_brain_naming_the_hud_does_not_reach_it(self):
        self.assertEqual(self.bc._find_windows_by_title("JARVIS HUD"), [])
        out = self._quiet(self.bc._act_minimize_window, "JARVIS HUD")
        self.assertEqual(out, "no window matching 'JARVIS HUD'")
        self.assertFalse(self.hud.minimized)

    def test_the_owner_naming_the_hud_reaches_it(self):
        self.owner.return_value = "Jarvis, minimize the HUD"
        self.assertEqual(self.bc._find_windows_by_title("hud"), [self.hud])
        out = self._quiet(self.bc._act_minimize_window, "hud")
        self.assertEqual(out, "minimized: JARVIS HUD")
        self.assertTrue(self.hud.minimized)

    def test_shell_windows_are_never_found(self):
        self.owner.return_value = "close program manager"
        self.assertEqual(self.bc._find_windows_by_title("program manager"), [])

    def test_the_owners_windows_are_still_found(self):
        self.assertEqual(self.bc._find_windows_by_title("file explorer"),
                         [self.files])

    def test_list_windows_shows_only_the_owners_window(self):
        out = self._quiet(self.bc._act_list_windows, "")
        self.assertEqual(out, "Open windows:\n  - Downloads - File Explorer")

    def test_the_live_improvised_reply_minimizes_only_the_owners_window(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.bc.parse_and_run_actions(IMPROVISED)
        self.assertTrue(self.files.minimized)
        for w in (self.hud, self.reticle, self.desk):
            self.assertFalse(w.minimized, w.title)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
