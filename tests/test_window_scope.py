"""core/window_scope.py - the ONE rule for which windows a window command may
touch (2026-10-03).

Live 17:22 the owner asked JARVIS to close every window but the Claude app.
list_windows handed the brain JARVIS's own HUD and Reticle plus two shell
windows ("Program Manager", "Windows Input Experience") next to the owner's
apps, and the brain minimized all of them. Now:

  * list_windows, every bulk window command and the name lookups of
    close_window / minimize_window / focus_window / move_window_to_monitor
    share core.window_scope;
  * the shell's windows (by title, by window class, cloaked or zero-size
    frames) are never a target;
  * JARVIS's own windows (this process and its python / console children,
    its console window, the known JARVIS titles) are never a bulk target,
    and a single-window command reaches one only when the query AND the
    owner's own words name it ("close the HUD").

Every window, Win32 read and process lookup is faked: ``probe``,
``own_pids`` and ``own_window_handles`` are patched, so nothing real is
queried, closed or minimized. Light tier.

    python -m unittest tests.test_window_scope
"""
from __future__ import annotations

import ast
import os
import sys
import types
import unittest
from unittest import mock

from core import window_scope as ws

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ME = 4242          # this (fake) JARVIS process
_HUD_CHILD = 4343   # a python child: the HUD / an overlay
_OTHER = 777        # some app of the owner's


class _Win:
    """A pygetwindow-like window."""

    def __init__(self, title, hwnd=None, *, width=800, height=600):
        self.title = title
        if hwnd is not None:
            self._hWnd = hwnd
        self.width, self.height = width, height


def _live_listing():
    """The windows of the live turn (titles generic), with what Win32 would
    have said about each: (window, pid, class, cloaked)."""
    return [
        (_Win("Claude", 0x101), _OTHER, "chrome_widgetwin_1", False),
        (_Win("Downloads - File Explorer", 0x102), _OTHER + 1,
         "cabinetwclass", False),
        (_Win("JARVIS HUD", 0x103), _HUD_CHILD, "qt6qwindowicon", False),
        (_Win("JARVIS Reticle", 0x104), _HUD_CHILD + 1, "tkchild", False),
        (_Win("Media Player", 0x105), _OTHER + 2,
         "applicationframewindow", False),
        (_Win("Program Manager", 0x106), _OTHER + 3, "progman", False),
        (_Win("Windows Input Experience", 0x107), _OTHER + 4,
         "windows.ui.core.corewindow", True),
    ]


class _ScopeBase(unittest.TestCase):
    """Patches every Win32 / process read core.window_scope makes."""

    def setUp(self):
        self.facts: dict = {}
        self.own = {_ME, _HUD_CHILD, _HUD_CHILD + 1}
        self.console = set()
        for name, fn in (
                ("probe", lambda w: self.facts.get(
                    id(w), ws.WindowFacts(getattr(w, "_hWnd", None), None,
                                          "", False))),
                ("own_pids", lambda: frozenset(self.own)),
                ("own_window_handles", lambda: frozenset(self.console))):
            p = mock.patch.object(ws, name, side_effect=fn)
            p.start()
            self.addCleanup(p.stop)

    def _install(self, rows):
        wins = []
        for w, pid, cls, cloaked in rows:
            self.facts[id(w)] = ws.WindowFacts(getattr(w, "_hWnd", None), pid,
                                               cls, cloaked)
            wins.append(w)
        return wins


class LiveListingTests(_ScopeBase):
    def test_user_windows_are_only_the_owners(self):
        wins = self._install(_live_listing())
        got = [w.title for w in ws.user_windows(wins)]
        self.assertEqual(got, ["Claude", "Downloads - File Explorer",
                               "Media Player"])

    def test_list_windows_never_shows_jarvis_or_shell_windows(self):
        import core.actions as A
        wins = self._install(_live_listing())
        fake = types.SimpleNamespace(getAllWindows=lambda: list(wins))
        with mock.patch.dict(sys.modules, {"pygetwindow": fake}):
            out = A._act_list_windows("")
        for gone in ("JARVIS HUD", "JARVIS Reticle", "Program Manager",
                     "Windows Input Experience"):
            self.assertNotIn(gone, out)
        for kept in ("Claude", "Downloads - File Explorer", "Media Player"):
            self.assertIn(f"  - {kept}", out)


class SystemWindowTests(_ScopeBase):
    def test_shell_titles_without_any_win32_facts(self):
        for title in ("Program Manager", "Windows Input Experience",
                      "Microsoft Text Input Application",
                      "  program   MANAGER "):
            with self.subTest(title=title):
                self.assertTrue(ws.is_system_window(_Win(title)))

    def test_taskbar_and_tray_hosts_by_class(self):
        for cls in ("shell_traywnd", "shell_secondarytraywnd",
                    "notifyiconoverflowwindow", "progman", "workerw"):
            with self.subTest(cls=cls):
                w = _Win("Some Title", 0x200)
                self.facts[id(w)] = ws.WindowFacts(0x200, _OTHER, cls, False)
                self.assertTrue(ws.is_system_window(w))
                self.assertEqual(ws.user_windows([w]), [])

    def test_cloaked_and_zero_size_frames(self):
        cloaked = _Win("Settings", 0x201)
        self.facts[id(cloaked)] = ws.WindowFacts(
            0x201, _OTHER, "applicationframewindow", True)
        flat = _Win("Calculator", 0x202, width=0, height=0)
        self.assertEqual(ws.user_windows([cloaked, flat]), [])

    def test_an_ordinary_app_window_stays(self):
        w = _Win("Report - Notepad", 0x203)
        self.facts[id(w)] = ws.WindowFacts(0x203, _OTHER, "notepad", False)
        self.assertFalse(ws.is_system_window(w))
        self.assertEqual(ws.user_windows([w]), [w])

    def test_untitled_windows_are_never_listed(self):
        self.assertEqual(ws.user_windows([_Win(""), _Win("   ")]), [])


class JarvisWindowTests(_ScopeBase):
    def test_own_and_child_process_windows(self):
        for pid in (_ME, _HUD_CHILD):
            with self.subTest(pid=pid):
                w = _Win("Some Overlay", 0x300)
                self.facts[id(w)] = ws.WindowFacts(0x300, pid, "tk", False)
                self.assertTrue(ws.is_jarvis_window(w))
                self.assertEqual(ws.user_windows([w]), [])

    def test_this_process_console_window(self):
        self.console.add(0x301)
        w = _Win("C:\\Python\\python.exe", 0x301)
        self.facts[id(w)] = ws.WindowFacts(0x301, _OTHER, "consolewindowclass",
                                           False)
        self.assertTrue(ws.is_jarvis_window(w))

    def test_known_titles_in_other_processes(self):
        for title in ("JARVIS HUD", "About JARVIS", "JARVIS Settings",
                      "JARVIS Settings \u2014 C:\\tmp\\settings.json",
                      "JARVIS \u2014 Live - Google Chrome",
                      "Today's Summary \u2014 JARVIS"):
            with self.subTest(title=title):
                self.assertTrue(ws.is_jarvis_title(title))
                self.assertEqual(ws.user_windows([_Win(title)]), [])

    def test_the_owners_windows_that_mention_jarvis_stay(self):
        for title in ("JARVIS - File Explorer", "notes about jarvis.txt - "
                      "Notepad", "bobert_companion.py - JARVIS - Visual "
                      "Studio Code", "Jarvis (film) - Wikipedia"):
            with self.subTest(title=title):
                self.assertFalse(ws.is_jarvis_title(title))
                w = _Win(title)
                self.assertEqual(ws.user_windows([w]), [w])


class OwnPidsTests(unittest.TestCase):
    def test_python_and_console_children_only(self):
        kids = [types.SimpleNamespace(pid=11, name=lambda: "pythonw.exe"),
                types.SimpleNamespace(pid=12, name=lambda: "conhost.exe"),
                types.SimpleNamespace(pid=13, name=lambda: "notepad.exe"),
                types.SimpleNamespace(pid=14, name=lambda: "chrome.exe")]
        proc = mock.Mock()
        proc.children.return_value = kids
        fake_psutil = types.SimpleNamespace(Process=lambda pid: proc)
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}), \
                mock.patch.object(ws, "_own_cache", [None, -1, frozenset()]):
            got = ws.own_pids()
        self.assertEqual(got, frozenset({os.getpid(), 11, 12}))
        proc.children.assert_called_once_with(recursive=True)

    def test_a_psutil_fault_is_just_this_process(self):
        def boom(pid):
            raise RuntimeError("no process table")
        fake_psutil = types.SimpleNamespace(Process=boom)
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}), \
                mock.patch.object(ws, "_own_cache", [None, -1, frozenset()]):
            self.assertEqual(ws.own_pids(), frozenset({os.getpid()}))


class NamedJarvisWindowTests(_ScopeBase):
    """A SINGLE-window command may reach a JARVIS window only by name."""

    def setUp(self):
        super().setUp()
        self.hud = _Win("JARVIS HUD", 0x400)
        self.facts[id(self.hud)] = ws.WindowFacts(0x400, _HUD_CHILD, "qt",
                                                  False)
        self.note = _Win("HUD sketch.txt - Notepad", 0x401)
        self.facts[id(self.note)] = ws.WindowFacts(0x401, _OTHER, "notepad",
                                                   False)

    def test_names_jarvis_window(self):
        self.assertTrue(ws.names_jarvis_window("close the HUD", "JARVIS HUD"))
        self.assertTrue(ws.names_jarvis_window("hide the reticle",
                                               "JARVIS Reticle"))
        self.assertTrue(ws.names_jarvis_window("close the overlay",
                                               "JARVIS Air Cursor"))
        self.assertTrue(ws.names_jarvis_window(
            "close the dashboard", "JARVIS \u2014 Live - Google Chrome"))
        self.assertFalse(ws.names_jarvis_window("close jarvis", "JARVIS HUD"))
        self.assertFalse(ws.names_jarvis_window(
            "close every window but the Claude app", "JARVIS HUD"))

    def test_the_brain_copying_a_title_is_not_the_owner_naming_it(self):
        # The live failure: the query names the HUD, the owner never did.
        got = ws.matching_windows([self.hud], "JARVIS HUD",
                                  owner_text="Jarvis, close every window "
                                             "but the Claude app")
        self.assertEqual(got, [])

    def test_the_owner_naming_it_reaches_it(self):
        got = ws.matching_windows([self.hud, self.note], "hud",
                                  owner_text="Jarvis, close the HUD")
        self.assertEqual(got, [self.hud, self.note])

    def test_no_owner_turn_lets_the_query_decide(self):
        self.assertEqual(ws.matching_windows([self.hud], "HUD"), [self.hud])

    def test_jarvis_alone_names_no_window(self):
        self.assertEqual(ws.matching_windows([self.hud], "jarvis",
                                             owner_text="close jarvis"), [])

    def test_shell_windows_are_never_matched_by_name(self):
        desk = _Win("Program Manager", 0x402)
        self.assertEqual(ws.matching_windows([desk], "program manager",
                                             owner_text="close Program "
                                                        "Manager"), [])

    def test_plain_substring_match_is_unchanged(self):
        a, b = _Win("Spotify Premium"), _Win("My SPOTIFY tab")
        self.assertEqual(ws.matching_windows([a, _Win("Notepad"), _Win(""), b],
                                             "spotify"), [a, b])


class ProbeTests(unittest.TestCase):
    def test_a_window_without_a_handle_is_unknown(self):
        self.assertEqual(ws.probe(_Win("x")), ws.WindowFacts(None, None, "",
                                                             False))
        self.assertEqual(ws.probe(_Win("x", True)).hwnd, None)

    def test_no_win32_is_unknown_not_an_error(self):
        fake_ctypes = types.SimpleNamespace()   # no windll
        with mock.patch.dict(sys.modules, {"ctypes": fake_ctypes}):
            facts = ws.probe(_Win("x", 0x500))
        self.assertEqual(facts, ws.WindowFacts(0x500, None, "", False))


def _window_title_literals():
    """(file, literal) for every JARVIS window title literal in the tree:
    ``x.title("...")`` / ``x.setWindowTitle("...")`` calls and ``title =
    "..."`` assignments that mention JARVIS, in the GUI code."""
    roots = [os.path.join(_ROOT, "hud"), os.path.join(_ROOT, "skills")]
    files = [os.path.join(_ROOT, "tray.py"),
             os.path.join(_ROOT, "tools", "settings_window.py")]
    for r in roots:
        for dp, _dn, fns in os.walk(r):
            files += [os.path.join(dp, f) for f in fns if f.endswith(".py")]
    out = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            lit = None
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("title", "setWindowTitle")
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                lit = node.args[0].value
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "title"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and path.endswith("settings_window.py")):
                lit = node.value.value
            if lit and "JARVIS" in lit:
                out.append((os.path.relpath(path, _ROOT), lit))
    return out


class TitleRatchetTests(unittest.TestCase):
    """A JARVIS window added later must not reach list_windows: every JARVIS
    window title literal in the tree is recognised (or the HUD's title
    drifted and the brain gets to minimize it again)."""

    def test_every_jarvis_window_title_is_recognised(self):
        found = _window_title_literals()
        # Blindness floor: the scan must still see the HUD and the reticle.
        titles = {lit for _f, lit in found}
        self.assertTrue({"JARVIS HUD", "JARVIS Reticle"} <= titles, titles)
        missed = [(f, lit) for f, lit in found if not ws.is_jarvis_title(lit)]
        self.assertEqual(missed, [], "JARVIS window titles core.window_scope "
                                     "does not recognise - add them to "
                                     "JARVIS_WINDOW_TITLES")

    def test_the_dashboard_page_title_is_recognised(self):
        with open(os.path.join(_ROOT, "tools", "web_interface.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        start = src.index("<title>") + len("<title>")
        page = src[start:src.index("</title>", start)]
        self.assertTrue(page.startswith("JARVIS"), page)
        self.assertTrue(ws.is_jarvis_title(page + " - Google Chrome"))
        self.assertTrue(ws.names_jarvis_window("the dashboard",
                                               page + " - Google Chrome"))


if __name__ == "__main__":
    unittest.main()
