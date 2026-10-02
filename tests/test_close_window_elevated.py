"""close_window on an ELEVATED window (2026-10-02 live).

Task Manager runs as administrator. JARVIS runs at normal integrity, so
Windows' UIPI refuses its WM_CLOSE ("Access is denied", error 5). Live
12:00:18-12:01:04 close_window answered "no window matching 'taskmgr.exe'",
then "could not close", then "no window matching 'Task Manager | left'", and
the chain went round close_window / see_screen until the depth cap - never
telling the owner why.

Now:
  * a refused close (access denied, or a target whose process is elevated)
    returns ONE honest, TERMINAL line - "Task Manager runs as administrator,
    sir; Windows won't let me close it from here." - behind
    core.failure_markers.TERMINAL_FAILURE_PREFIX, which the dispatcher speaks
    verbatim and which ends the follow-up chain;
  * a process name ("taskmgr.exe") resolves to that process's windows;
  * a title with a monitor suffix ("Task Manager | left", the
    move_window_to_monitor format the model copied) resolves to the title,
    narrowed to that monitor when several windows match.

Every window, process and Win32 call is faked; nothing real is closed or
queried. Light tier (core.actions with a mocked monolith).

    python -m unittest tests.test_close_window_elevated
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import core.actions as A
from core import failure_markers as fm


class _Win:
    """A pygetwindow-like window: title, handle, geometry, close()."""

    def __init__(self, title, hwnd=None, *, raises=None, box=(0, 0, 800, 600)):
        self.title = title
        if hwnd is not None:
            self._hWnd = hwnd
        self._raises = raises
        self.left, self.top, self.width, self.height = box
        self.closed = False

    def close(self):
        if self._raises is not None:
            raise self._raises
        self.closed = True


def _denied():
    # pygetwindow's own wording for a refused PostMessage(WM_CLOSE).
    return Exception("Error code from Windows: 5 - Access is denied.")


class _Base(unittest.TestCase):
    def setUp(self):
        self.bc = mock.Mock()
        self.bc.FORBIDDEN_TARGETS = ["bobert_companion", "jarvis terminal"]
        self.bc._BROWSER_CHROME_SUFFIXES = (" - google chrome",)
        self.bc._strip_bidi_and_nbsp = lambda s: s
        self.bc._find_windows_by_title.return_value = []
        p = mock.patch.object(A, "_bc", return_value=self.bc)
        p.start()
        self.addCleanup(p.stop)
        # Nothing here may query a real process: elevation and process names
        # come from these fakes only.
        self.elevated: set = set()
        self.procs: dict = {}
        for name, fn in (
                ("_window_is_elevated",
                 lambda w: getattr(w, "_hWnd", None) in self.elevated),
                ("_window_process_name",
                 lambda w: self.procs.get(getattr(w, "_hWnd", None)))):
            q = mock.patch.object(A, name, side_effect=fn)
            q.start()
            self.addCleanup(q.stop)

    def _all_windows(self, windows):
        """Install a fake pygetwindow whose getAllWindows() is ``windows``."""
        fake = types.SimpleNamespace(getAllWindows=lambda: list(windows))
        p = mock.patch.dict(sys.modules, {"pygetwindow": fake})
        p.start()
        self.addCleanup(p.stop)


class ElevatedCloseTests(_Base):
    def _assert_terminal(self, out, name="Task Manager"):
        self.assertTrue(out.startswith(fm.TERMINAL_FAILURE_PREFIX), out)
        line = fm.terminal_failure_text(out)
        self.assertIn(f"{name} runs as administrator, sir", line)
        self.assertIn("Windows won't let me close it from here", line)
        # Still a failure to every consumer of FAILURE_MARKERS (the chain
        # resolver must never read it as a success).
        self.assertTrue(any(m in out.lower() for m in fm.FAILURE_MARKERS))

    def test_access_denied_is_an_honest_terminal_line(self):
        w = _Win("Task Manager", 0x10, raises=_denied())
        self.bc._find_windows_by_title.return_value = [w]
        self._assert_terminal(A._act_close_window("Task Manager"))

    def test_a_permission_error_is_the_same(self):
        w = _Win("Task Manager", 0x10, raises=PermissionError(13, "denied"))
        self.bc._find_windows_by_title.return_value = [w]
        self._assert_terminal(A._act_close_window("task manager"))

    def test_a_failed_close_of_an_elevated_process_is_terminal(self):
        # Whatever the error text, a failed close of an elevated window is
        # the UIPI refusal.
        w = _Win("Task Manager", 0x10, raises=Exception("Error code from "
                                                        "Windows: 1400"))
        self.elevated.add(0x10)
        self.bc._find_windows_by_title.return_value = [w]
        self._assert_terminal(A._act_close_window("task manager"))

    def test_a_close_that_works_never_asks_about_the_process(self):
        w = _Win("Notepad", 0x11)
        self.bc._find_windows_by_title.return_value = [w]
        self.assertIn("closed: Notepad", A._act_close_window("notepad"))
        A._window_is_elevated.assert_not_called()

    def test_an_ordinary_failure_is_unchanged(self):
        w = _Win("Notepad", 0x11, raises=Exception("locked"))
        self.bc._find_windows_by_title.return_value = [w]
        self.assertEqual(A._act_close_window("notepad"), "could not close")

    def test_the_administrator_prefix_is_not_the_name(self):
        w = _Win("Administrator: Command Prompt", 0x12, raises=_denied())
        self.bc._find_windows_by_title.return_value = [w]
        out = A._act_close_window("command prompt")
        self._assert_terminal(out, name="Command Prompt")

    def test_closed_windows_are_reported_with_the_refused_one(self):
        ok = _Win("Notepad - Task list", 0x13)
        tm = _Win("Task Manager", 0x10, raises=_denied())
        self.bc._find_windows_by_title.return_value = [ok, tm]
        out = A._act_close_window("task")
        self.assertTrue(ok.closed)
        line = fm.terminal_failure_text(out)
        self.assertIn("I closed the other window, sir", line)
        self.assertIn("Task Manager runs as administrator", line)


class TargetResolutionTests(_Base):
    def test_a_process_name_resolves_to_its_windows(self):
        tm = _Win("Task Manager", 0x10)
        pad = _Win("Notepad", 0x11)
        self.procs.update({0x10: "Taskmgr.exe", 0x11: "notepad.exe"})
        self._all_windows([tm, pad])
        out = A._act_close_window("taskmgr.exe")
        self.assertTrue(tm.closed)
        self.assertFalse(pad.closed)
        self.assertIn("closed: Task Manager", out)

    def test_a_process_name_with_no_window_is_still_no_match(self):
        self.procs.update({0x11: "notepad.exe"})
        self._all_windows([_Win("Notepad", 0x11)])
        self.assertEqual(A._act_close_window("taskmgr.exe"),
                         "no window matching 'taskmgr.exe'")

    def test_an_elevated_process_name_is_terminal(self):
        tm = _Win("Task Manager", 0x10, raises=_denied())
        self.procs[0x10] = "Taskmgr.exe"
        self._all_windows([tm])
        out = A._act_close_window("taskmgr.exe")
        self.assertIn("Task Manager runs as administrator",
                      fm.terminal_failure_text(out))

    def test_a_process_name_never_reaches_the_desktop_window(self):
        # Review 2026-10-02: explorer.exe also owns the desktop ("Program
        # Manager"); WM_CLOSE on it opens Windows' shut-down dialog.
        desk = _Win("Program Manager", 0x30)
        files = _Win("Downloads - File Explorer", 0x31)
        self.procs.update({0x30: "explorer.exe", 0x31: "explorer.exe"})
        self._all_windows([desk, files])
        out = A._act_close_window("explorer.exe")
        self.assertTrue(files.closed)
        self.assertFalse(desk.closed)
        self.assertNotIn("Program Manager", out)
        self._all_windows([desk])
        self.assertEqual(A._act_close_window("explorer.exe"),
                         "no window matching 'explorer.exe'")
        self.assertFalse(desk.closed)

    def test_a_monitor_suffix_resolves_to_the_title(self):
        w = _Win("Task Manager", 0x10)
        self.bc._find_windows_by_title.side_effect = (
            lambda q: [w] if q.lower() == "task manager" else [])
        out = A._act_close_window("Task Manager | left")
        self.assertTrue(w.closed)
        self.assertIn("closed: Task Manager", out)

    def test_the_monitor_narrows_several_matches(self):
        from core.config import MONITORS
        lx, ly, lw, lh = MONITORS["left"]
        rx, ry, rw, rh = MONITORS["right"]
        left = _Win("Notes", 0x20, box=(lx + 100, ly + 100, 600, 400))
        right = _Win("Notes", 0x21, box=(rx + 100, ry + 100, 600, 400))
        self.bc._find_windows_by_title.side_effect = (
            lambda q: [left, right] if q.lower() == "notes" else [])
        A._act_close_window("notes | left")
        self.assertTrue(left.closed)
        self.assertFalse(right.closed)

    def test_a_pipe_that_is_not_a_monitor_is_left_alone(self):
        w = _Win("Report | Draft", 0x22)
        self.bc._find_windows_by_title.side_effect = (
            lambda q: [w] if q.lower() == "report | draft" else [])
        A._act_close_window("Report | Draft")
        self.assertTrue(w.closed)


class TerminalPrefixTests(unittest.TestCase):
    def test_terminal_text_round_trip(self):
        line = "Task Manager runs as administrator, sir."
        self.assertEqual(
            fm.terminal_failure_text(fm.TERMINAL_FAILURE_PREFIX + line), line)
        for other in ("could not close", "", None, 42, "closed: Notepad"):
            with self.subTest(other=other):
                self.assertEqual(fm.terminal_failure_text(other), "")


if __name__ == "__main__":
    unittest.main()
