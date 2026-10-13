"""Generic fullscreen-game detection for skills/game_mode.py (2026-10-12).

Before this the mode only knew the one allowlisted title, so it never engaged
for any other game. A foreground window that covers its whole monitor and whose
executable lives in a game-library folder is now a game too. What these pin:

  * a game-library path makes a fullscreen app a game; the same fullscreen
    window from a browser / media player / meeting / office app never is, even
    if that program happens to sit under a "games" folder;
  * an unrecognised fullscreen app is NEVER entered, and is logged once;
  * GAME_MODE_DETECT_FULLSCREEN=False switches the whole thing off;
  * a recognised game is remembered by pid, so alt-tab does not make it
    "disappear" (exit is by process lifetime, exactly as for the allowlist),
    and a dead or recycled pid is pruned;
  * the pure decide() core enters on it after the foreground dwell and exits
    when the process dies.

Nothing here touches the real desktop, a real process or the GPU.
"""
from __future__ import annotations

import io
import contextlib
import sys
import types
import unittest
from unittest import mock

from tests.skills.test_game_mode import _Base, GAME

STEAM_EXE = r"D:\SteamLibrary\steamapps\common\Some Game\SomeGame-Win64-Shipping.exe"
EPIC_EXE = r"C:\Program Files\Epic Games\Fortnite\FortniteClient.exe"
CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
SLIDES_EXE = r"C:\Users\someone\Desktop\deck.exe"


def _fake_psutil(alive: dict):
    """alive: pid -> process name. Mirrors the two calls the skill makes."""
    ps = types.ModuleType("psutil")
    ps.pid_exists = lambda pid: int(pid) in alive

    class _P:
        def __init__(self, pid):
            if int(pid) not in alive:
                raise ps.NoSuchProcess()
            self._name = alive[int(pid)]

        def name(self):
            return self._name

    class NoSuchProcess(Exception):
        pass
    ps.NoSuchProcess = NoSuchProcess
    ps.Process = _P
    ps.process_iter = lambda attrs=None: iter(())
    return ps


class _FullscreenBase(_Base):
    SETTINGS = dict(_Base.SETTINGS,
                    GAME_MODE_DETECT_FULLSCREEN=True,
                    GAME_MODE_PROCESS_HINTS=[GAME],
                    GAME_MODE_PATH_MARKERS=[
                        "\\steamapps\\common\\", "\\epic games\\",
                        "\\games\\"])

    def _fg(self, pid, path):
        """Pretend the foreground window is a fullscreen one from `path`."""
        exe = path.rsplit("\\", 1)[-1].lower()
        self.mod._fullscreen_foreground = lambda: (pid, exe, path)

    def _no_fullscreen(self):
        self.mod._fullscreen_foreground = lambda: (None, None, None)

    def _load_fs(self, alive=None, **kw):
        mod, actions = self._load(**kw)
        mod._seen_games.clear()
        mod._unrecognised_logged.clear()
        ps = _fake_psutil(alive if alive is not None else {})
        p = mock.patch.dict(sys.modules, {"psutil": ps})
        p.start()
        self.addCleanup(p.stop)
        self.psutil = ps
        return mod, actions


class TestLooksLikeAGame(_FullscreenBase):
    def test_game_library_path_is_a_game(self):
        mod, _ = self._load_fs()
        self.assertTrue(mod._looks_like_game("somegame-win64-shipping.exe", STEAM_EXE))
        self.assertTrue(mod._looks_like_game("fortniteclient.exe", EPIC_EXE))

    def test_marker_match_is_case_and_slash_insensitive(self):
        mod, _ = self._load_fs()
        self.assertTrue(mod._looks_like_game(
            "x.exe", "d:/steamlibrary/STEAMAPPS/Common/x/x.exe"))

    def test_a_program_outside_every_library_is_not(self):
        mod, _ = self._load_fs()
        self.assertFalse(mod._looks_like_game("deck.exe", SLIDES_EXE))

    def test_browsers_and_players_never_are(self):
        mod, _ = self._load_fs()
        self.assertFalse(mod._looks_like_game("chrome.exe", CHROME_EXE))
        # even from a folder called Games
        self.assertFalse(mod._looks_like_game(
            "vlc.exe", r"D:\Games\VideoLAN\vlc.exe"))
        self.assertFalse(mod._looks_like_game(
            "powerpnt.exe", r"C:\Games\Office\powerpnt.exe"))

    def test_game_launchers_are_not_games(self):
        mod, _ = self._load_fs()
        self.assertFalse(mod._looks_like_game(
            "epicgameslauncher.exe",
            r"C:\Program Files (x86)\Epic Games\Launcher\Portal\EpicGamesLauncher.exe"))
        self.assertFalse(mod._looks_like_game(
            "steam.exe", r"C:\Games\Steam\steam.exe"))

    def test_the_allowlist_always_wins_and_so_does_a_learned_name(self):
        mod, _ = self._load_fs()
        self.assertTrue(mod._looks_like_game(GAME, None))
        self.assertFalse(mod._looks_like_game("vlc.exe", r"D:\Games\vlc.exe"))
        self.bc.GAME_MODE_PROCESS_HINTS = [GAME, "vlc.exe"]
        self.assertTrue(mod._looks_like_game("vlc.exe", None))

    def test_no_name_or_no_path_is_not_a_game(self):
        mod, _ = self._load_fs()
        self.assertFalse(mod._looks_like_game("", STEAM_EXE))
        self.assertFalse(mod._looks_like_game("a.exe", None))

    def test_empty_marker_list_recognises_nothing_by_path(self):
        mod, _ = self._load_fs(settings={"GAME_MODE_PATH_MARKERS": []})
        self.assertFalse(mod._looks_like_game("a.exe", STEAM_EXE))


class TestFindFullscreenGame(_FullscreenBase):
    def test_a_fullscreen_library_game_is_found_and_remembered(self):
        mod, _ = self._load_fs(alive={4242: "somegame-win64-shipping.exe"})
        self._fg(4242, STEAM_EXE)
        self.assertEqual(mod._find_game_pid(),
                         (4242, "somegame-win64-shipping.exe"))
        self.assertIn(4242, mod._seen_games)

    def test_alt_tab_keeps_it_running_while_the_process_lives(self):
        mod, _ = self._load_fs(alive={4242: "somegame-win64-shipping.exe"})
        self._fg(4242, STEAM_EXE)
        mod._find_game_pid()
        self._no_fullscreen()                   # he tabbed out
        self.assertEqual(mod._find_game_pid()[0], 4242)

    def test_a_dead_process_is_pruned(self):
        mod, _ = self._load_fs(alive={4242: "somegame-win64-shipping.exe"})
        self._fg(4242, STEAM_EXE)
        mod._find_game_pid()
        self._no_fullscreen()
        self.psutil.pid_exists = lambda pid: False
        self.assertEqual(mod._find_game_pid(), (None, None))
        self.assertEqual(mod._seen_games, {})

    def test_a_recycled_pid_is_not_the_game(self):
        mod, _ = self._load_fs(alive={4242: "somegame-win64-shipping.exe"})
        self._fg(4242, STEAM_EXE)
        mod._find_game_pid()
        self._no_fullscreen()
        # same pid, now a different program
        self.psutil.pid_exists = lambda pid: True
        self.psutil.Process = lambda pid: types.SimpleNamespace(
            name=lambda: "notepad.exe")
        self.assertEqual(mod._find_game_pid(), (None, None))
        self.assertEqual(mod._seen_games, {})

    def test_an_unrecognised_fullscreen_app_is_not_a_game_and_logs_once(self):
        mod, _ = self._load_fs()
        self._fg(777, SLIDES_EXE)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(mod._find_game_pid(), (None, None))
            self.assertEqual(mod._find_game_pid(), (None, None))
        self.assertEqual(buf.getvalue().count("not in a game library"), 1)
        self.assertEqual(mod._seen_games, {})

    def test_a_fullscreen_browser_is_silently_ignored(self):
        mod, _ = self._load_fs()
        self._fg(778, CHROME_EXE)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(mod._find_game_pid(), (None, None))
        self.assertNotIn("not in a game library", buf.getvalue())

    def test_the_switch_turns_the_whole_thing_off(self):
        mod, _ = self._load_fs(settings={"GAME_MODE_DETECT_FULLSCREEN": False})
        self._fg(4242, STEAM_EXE)
        self.assertEqual(mod._find_game_pid(), (None, None))

    def test_the_allowlist_still_works_without_fullscreen(self):
        mod, _ = self._load_fs()
        ps = self.psutil
        ps.process_iter = lambda attrs=None: iter([types.SimpleNamespace(
            info={"pid": 99, "name": GAME})])
        self._no_fullscreen()
        self.assertEqual(mod._find_game_pid(), (99, GAME))

    def test_no_allowlist_still_detects_by_fullscreen(self):
        mod, _ = self._load_fs(settings={"GAME_MODE_PROCESS_HINTS": []},
                               alive={4242: "somegame-win64-shipping.exe"})
        self._fg(4242, STEAM_EXE)
        self.assertEqual(mod._find_game_pid()[0], 4242)


class TestEntryAndExitOnADetectedGame(_FullscreenBase):
    CFG = {"dwell": 20.0, "grace": 45.0, "max_seconds": 43200.0,
           "require_foreground": True}

    def test_enters_after_the_dwell_and_leaves_with_the_process(self):
        mod, _ = self._load_fs(alive={4242: "somegame-win64-shipping.exe"})
        self._fg(4242, STEAM_EXE)
        st = mod._State()
        pid, _exe = mod._find_game_pid()
        # first sighting starts the dwell, 20 s later it enters
        self.assertEqual(mod.decide(100.0, pid, 4242, st, self.CFG)[0], "none")
        self.assertEqual(mod.decide(121.0, pid, 4242, st, self.CFG)[0], "enter")
        st.active, st.game_pid, st.entered_at = True, pid, 121.0
        st.last_seen_at = 121.0
        # alt-tab: not fullscreen any more, process still alive -> stays on
        self._no_fullscreen()
        pid2, _ = mod._find_game_pid()
        self.assertEqual(pid2, 4242)
        self.assertEqual(mod.decide(200.0, pid2, 9999, st, self.CFG)[0], "none")
        # process dies -> after the relaunch grace, exit
        self.psutil.pid_exists = lambda p: False
        mod._pid_alive = lambda p: False
        pid3, _ = mod._find_game_pid()
        self.assertIsNone(pid3)
        self.assertEqual(mod.decide(210.0, pid3, 9999, st, self.CFG)[0], "none")
        self.assertEqual(mod.decide(300.0, pid3, 9999, st, self.CFG)[0], "exit")

    def test_an_unrecognised_fullscreen_app_never_enters(self):
        mod, _ = self._load_fs()
        self._fg(777, SLIDES_EXE)
        st = mod._State()
        for t in (0.0, 30.0, 60.0):
            pid, _ = mod._find_game_pid()
            self.assertEqual(mod.decide(t, pid, 777, st, self.CFG)[0], "none")

    def test_the_watcher_starts_with_fullscreen_detection_even_with_no_list(self):
        # register() used to need a non-empty allowlist.
        self.bc = None
        with mock.patch.dict("os.environ", {"JARVIS_STAGING": ""}):
            mod, _ = self._load(settings={"GAME_MODE_ENABLED": True,
                                          "GAME_MODE_PROCESS_HINTS": [],
                                          "GAME_MODE_DETECT_FULLSCREEN": True})
        # load_skill_isolated neuters Thread.start; a thread object exists
        self.assertIsNotNone(mod._thread[0])
        mod._thread[0] = None


if __name__ == "__main__":
    unittest.main()
