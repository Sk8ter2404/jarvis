"""Replays of the 2026-10-02 close-window turns through the REAL dispatch and
the REAL close_window / list_windows (review of claude/live-fixes-1202).

The fix commits each test their own layer with the neighbour stubbed:
the claim detector with a stub close_window, the terminal line with a stub
that already returns it. These drive the live SEQUENCES end to end - the
model's replies are canned, but parse_and_run_actions, the follow-up loop,
the verbatim speech, core.actions._act_close_window and its target
resolution all run for real against a fake desktop:

  * 12:00:09-12:01:04 - a misheard command, a first-person claim with no
    token (never spoken now), then close_window on the elevated Task Manager
    by its PROCESS name, then by "<title> | <monitor>". Each now ends on the
    one honest administrator line, with no see_screen circle.
  * 11:57:29 - "close every window but one" answered with a claim and no
    token; the follow-up rounds list the windows and close them for real.
    Since 2026-10-03 that request itself is routed to close_all_windows_except
    without the model (core.dispatcher.window_keep_route), so the brain-led
    chain is replayed with a phrasing the route leaves to the model, and the
    routed turn is pinned on its own.

Made-up fixtures of the live shapes (paraphrased; none of the owner's words).
pygetwindow is a fake module, process names and elevation come from a
table, every window is a stub object: nothing real is listed, queried or
closed, and no LLM, audio or network is touched.

    python -m unittest tests.monolith.test_monolith_close_chain_replay
"""
from __future__ import annotations

import sys
import types
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

_ADMIN_LINE = ("Task Manager runs as administrator, sir; Windows won't let "
               "me close it from here.")


class _Win:
    """A pygetwindow-like window. ``denied``: close() raises the error
    pygetwindow raises when UIPI refuses WM_CLOSE to an elevated window."""

    def __init__(self, title, hwnd, *, denied=False, box=(-2400, 100, 800,
                                                          600)):
        self.title = title
        self._hWnd = hwnd
        self._denied = denied
        self.left, self.top, self.width, self.height = box
        self.closed = False

    def close(self):
        if self._denied:
            raise Exception("Error code from Windows: 5 - Access is denied.")
        self.closed = True


@requires_monolith
class CloseChainReplayTests(_Base):
    def setUp(self):
        super().setUp()
        import core.actions as A
        self.windows: list = []
        fake = types.SimpleNamespace(getAllWindows=lambda: [
            w for w in self.windows if not w.closed])
        p = mock.patch.dict(sys.modules, {"pygetwindow": fake})
        p.start()
        self.addCleanup(p.stop)
        self.procs: dict = {}
        self.elevated: set = set()
        self._p(A, "_window_process_name",
                side_effect=lambda w: self.procs.get(w._hWnd))
        self._p(A, "_window_is_elevated",
                side_effect=lambda w: w._hWnd in self.elevated)
        # The window lookups go through core.window_scope (2026-10-03): its
        # Win32 reads must not resolve these small fake handles to real
        # windows.
        from core import window_scope as ws
        self._p(ws, "probe", side_effect=lambda w: ws.WindowFacts(
            w._hWnd, None, "", False))
        self._p(ws, "own_pids", return_value=frozenset())
        self._p(ws, "own_window_handles", return_value=frozenset())
        self._stub("see_screen", "Task Manager is on the LEFT monitor.")

    def _task_manager(self):
        tm = _Win("Task Manager", 0x10, denied=True)
        pad = _Win("Untitled - Notepad", 0x11, box=(100, 100, 800, 600))
        self.procs.update({0x10: "Taskmgr.exe", 0x11: "notepad.exe"})
        self.elevated.add(0x10)
        self.windows[:] = [tm, pad]
        return tm, pad

    def test_the_misheard_close_replay_ends_on_one_honest_line(self):
        tm, pad = self._task_manager()
        self._dispatch(
            "Jarvis had been at close the task manager.",
            "[intent:confirmation] As you wish, sir. I've closed that "
            "program for you.",
            ["[intent:dry_wit] Certainly, sir. "
             "[ACTION: close_window, taskmgr.exe]",
             "[intent:bad_news] Windows seems stubborn, sir. "
             "[ACTION: see_screen, task manager]",
             "[intent:confirmation] Right away, sir. "
             "[ACTION: close_window, Task Manager | left]"])
        self.assertEqual(self.spoken, [_ADMIN_LINE])
        self.assertEqual(self.gfr.call_count, 1)
        self.assertEqual(self._followup_names(), ["_unverified_claim"])
        self.assertEqual(self.calls["see_screen"], [])
        self.assertFalse(tm.closed)
        self.assertFalse(pad.closed)

    def test_the_monitor_suffix_shape_is_the_same_honest_line(self):
        tm, pad = self._task_manager()
        self._dispatch(
            "close the task manager on the left",
            "[intent:confirmation] Right away, sir. "
            "[ACTION: close_window, Task Manager | left]",
            ["[ACTION: see_screen, task manager]"])
        self.assertEqual(self.spoken, [_ADMIN_LINE])
        self.gfr.assert_not_called()
        self.assertFalse(tm.closed)
        self.assertFalse(pad.closed)

    def test_close_everything_but_one_replay_closes_for_real(self):
        editor = _Win("notes.txt - Code Editor", 0x20)
        music = _Win("Music Player", 0x21)
        mail = _Win("Inbox - Mail", 0x22)
        self.windows[:] = [editor, music, mail]
        self._dispatch(
            # Not a route phrasing ("get rid of"): the brain answers it.
            "Jarvis, get rid of every window except the editor.",
            "[intent:confirmation] Very good, sir. I've taken the liberty of "
            "closing everything else so you have some room.",
            ["[intent:confirmation] One moment, sir. [ACTION: list_windows]",
             "[intent:confirmation] Certainly, sir. "
             "[ACTION: close_window, Music Player] "
             "[ACTION: close_window, Inbox - Mail]"])
        self.assertEqual(self._followup_names(0), ["_unverified_claim"])
        self.assertEqual(self._followup_names(1), ["list_windows"])
        # Two clean closes are not informative: the chain ends on them.
        self.assertEqual(self.gfr.call_count, 2)
        self.assertTrue(music.closed)
        self.assertTrue(mail.closed)
        self.assertFalse(editor.closed)
        for s in self.spoken:
            self.assertNotIn("taken the liberty", s)
            self.assertNotIn("everything else", s)
        self.assertEqual(self.spoken[-1], "[intent:confirmation] Certainly, "
                                          "sir.")

    def test_close_everything_but_one_now_runs_one_action(self):
        # The live shape itself (2026-10-03): routed to
        # close_all_windows_except, so the model is never asked, the real
        # action closes the two windows and keeps the editor, and its one
        # summary is the whole reply.
        editor = _Win("notes.txt - Code Editor", 0x20)
        music = _Win("Music Player", 0x21)
        mail = _Win("Inbox - Mail", 0x22)
        self.windows[:] = [editor, music, mail]
        self._dispatch(
            "Jarvis closed every window except the editor.",
            "[intent:confirmation] Very good, sir. I've taken the liberty of "
            "closing everything else so you have some room.",
            ["[intent:confirmation] One moment, sir. [ACTION: list_windows]"])
        self.bc.get_response_with_animation.assert_not_called()
        self.gfr.assert_not_called()
        self.assertTrue(music.closed)
        self.assertTrue(mail.closed)
        self.assertFalse(editor.closed)
        self.assertEqual(self.spoken,
                         ["Closed 2 windows, sir; kept the editor."])
