"""Unit tests for the ``_act_*`` handlers in core/actions.py, lines 378-997.

Scope (Phase-4 handler block, "Section 2"):
    _act_switch_llm_picker, _act_show_llm_stats, _act_press, _act_scroll,
    _act_list_skills, _act_apple_music, _act_launch_app, _act_pause_music,
    _act_resume_music, _act_now_playing, _act_queue_task, _act_list_windows,
    _act_focus_window, _act_minimize_window, _act_close_window, _act_type,
    _act_next_song, _act_previous_song, _act_show_tasks, _act_ambient_mode_set,
    _act_reload_skills, _act_show_recent_facts, _act_export_memory,
    _act_run_diagnostic_tray, _act_show_last_diagnostic, _act_play_streaming,
    _act_click, _act_hotkey, _act_stop_pipeline.

Design / CI-safety contract (mirrors the section brief):
  * Every handler reaches the ~14K-line monolith only via ``A._bc()``. We
    NEVER let the real module load: each test patches ``A._bc`` to return a
    ``mock.Mock`` configured with exactly the attributes/methods the handler
    under test touches. That keeps the suite import-light and CI-faithful
    (the run_tests_ci_sim blocker for heavy deps is never tripped).
  * All other I/O is mocked: subprocess.Popen, shutil.which, os.startfile,
    webbrowser, threads (handlers spawn via ``bc._tray_async`` which is a Mock,
    so nothing actually runs), and the filesystem (redirected to a per-test
    tempdir). No real hardware, network, or clock dependence.
  * ``pygetwindow`` is absent on CI, so handlers that ``import pygetwindow``
    are exercised by injecting a fake module into ``sys.modules`` for the
    duration of a single test (auto-removed in tearDown). This is per-test,
    never a module-level write.
  * ``time`` is frozen where the output embeds a timestamp.

These tests assert the section's source behaviour as-found; any line noted as
a BUG in the final report is asserted at its *current* (buggy) behaviour so the
suite stays green and documents the defect rather than masking it.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import core.actions as A


# ── shared helpers ───────────────────────────────────────────────────────────
def _fresh_bc(**attrs):
    """Build a Mock standing in for the bobert_companion module.

    ``UIFailsafeError`` defaults to a real Exception subclass so handlers that
    do ``except bc.UIFailsafeError`` have a valid type to catch. Any keyword
    overrides the attribute on the returned mock.
    """
    bc = mock.Mock()
    bc.UIFailsafeError = _UIFailsafe
    for k, v in attrs.items():
        setattr(bc, k, v)
    return bc


class _UIFailsafe(Exception):
    """Stand-in for bobert_companion.UIFailsafeError."""


class _FakeWindow:
    """Minimal pygetwindow Window-like object."""
    def __init__(self, title="", *, on_activate=None, on_minimize=None,
                 on_restore=None, on_close=None):
        self.title = title
        self._on_activate = on_activate
        self._on_minimize = on_minimize
        self._on_restore = on_restore
        self._on_close = on_close
        self.activated = self.minimized = self.restored = self.closed = False

    def activate(self):
        self.activated = True
        if self._on_activate:
            self._on_activate()

    def minimize(self):
        self.minimized = True
        if self._on_minimize:
            self._on_minimize()

    def restore(self):
        self.restored = True
        if self._on_restore:
            self._on_restore()

    def close(self):
        self.closed = True
        if self._on_close:
            self._on_close()


class _BaseActTest(unittest.TestCase):
    """Common scaffolding: patch A._bc per test and offer install_fake_module."""

    def setUp(self):
        self.bc = _fresh_bc()
        self._bc_patcher = mock.patch.object(A, "_bc", return_value=self.bc)
        self._bc_patcher.start()
        self.addCleanup(self._bc_patcher.stop)
        self._injected_modules = []

    def install_fake_module(self, name, module):
        """Put a fake module in sys.modules for the duration of this test only.

        Records prior state and restores it in cleanup so isolation holds even
        if the real module happened to be importable on the dev box.
        """
        had = name in sys.modules
        prev = sys.modules.get(name)
        sys.modules[name] = module
        self._injected_modules.append((name, had, prev))

        def _restore():
            if had:
                sys.modules[name] = prev
            else:
                sys.modules.pop(name, None)
        self.addCleanup(_restore)

    def make_pygetwindow(self, *, all_windows=None, active=None):
        m = mock.Mock(name="pygetwindow")
        m.getAllWindows = mock.Mock(return_value=list(all_windows or []))
        m.getActiveWindow = mock.Mock(return_value=active)
        return m

    def patch_apple_music_app(self, *, is_active=False, running=False,
                              installed=False, now_playing=None):
        """Patch A._apple_music_app() to return a configured Mock bridge for
        the duration of the test. Returns the mock so callers can assert on it.
        Pass None as a kwarg value's source to simulate the bridge being
        unimportable (use patch_apple_music_app_none for that)."""
        amapp = mock.Mock(name="apple_music_app")
        amapp.is_active_media_app.return_value = is_active
        amapp.is_running.return_value = running
        amapp.is_installed.return_value = installed
        amapp.now_playing.return_value = now_playing
        amapp.launch.return_value = (True, None)
        p = mock.patch.object(A, "_apple_music_app", return_value=amapp)
        p.start()
        self.addCleanup(p.stop)
        return amapp

    def patch_apple_music_app_none(self):
        """Simulate the apple_music_app bridge being unimportable (returns None)."""
        p = mock.patch.object(A, "_apple_music_app", return_value=None)
        p.start()
        self.addCleanup(p.stop)


# ── _act_switch_llm_picker / _act_show_llm_stats ─────────────────────────────
class LLMMenuTests(_BaseActTest):
    def test_switch_llm_picker_lists_models_with_costs(self):
        out = A._act_switch_llm_picker("")
        # now lists the priced model options + a switch hint (was a stub)
        for label in ("Claude Haiku", "Claude Sonnet", "Claude Opus", "Qwen"):
            self.assertIn(label, out)
        self.assertIn("/conv", out)        # cost-per-conversation shown
        self.assertIn("switch to", out)

    def test_show_llm_stats_claude_backend(self):
        with mock.patch("core.config.AI_BACKEND", "claude"), \
             mock.patch("core.config.CLAUDE_MODEL", "claude-test-model"), \
             mock.patch("core.config.OLLAMA_MODEL", "llama-x"):
            out = A._act_show_llm_stats("")
        self.assertIn("backend=claude", out)
        self.assertIn("model=claude-test-model", out)

    def test_show_llm_stats_ollama_backend_uses_ollama_model(self):
        with mock.patch("core.config.AI_BACKEND", "ollama"), \
             mock.patch("core.config.CLAUDE_MODEL", "claude-x"), \
             mock.patch("core.config.OLLAMA_MODEL", "llama-test-model"):
            out = A._act_show_llm_stats("")
        self.assertIn("backend=ollama", out)
        self.assertIn("model=llama-test-model", out)
        self.assertNotIn("claude-x", out)


# ── _act_press / _act_scroll ─────────────────────────────────────────────────
class PressScrollTests(_BaseActTest):
    def test_press_normalizes_and_reports(self):
        out = A._act_press("  ENTER ")
        self.bc.ui_press.assert_called_once_with("enter")
        self.assertEqual(out, "pressed   ENTER ")  # report echoes raw arg

    def test_press_failsafe_returns_message(self):
        self.bc.ui_press.side_effect = _UIFailsafe("failsafe tripped")
        self.assertEqual(A._act_press("a"), "failsafe tripped")

    def test_scroll_positive(self):
        out = A._act_scroll(" 5 ")
        self.bc.ui_scroll.assert_called_once_with(5)
        self.assertEqual(out, "scrolled 5")

    def test_scroll_negative(self):
        self.assertEqual(A._act_scroll("-3"), "scrolled -3")
        self.bc.ui_scroll.assert_called_once_with(-3)

    def test_scroll_non_integer(self):
        out = A._act_scroll("down")
        self.assertIn("must be an integer", out)
        self.bc.ui_scroll.assert_not_called()

    def test_scroll_failsafe(self):
        self.bc.ui_scroll.side_effect = _UIFailsafe("no can do")
        self.assertEqual(A._act_scroll("2"), "no can do")


# ── _act_list_skills ─────────────────────────────────────────────────────────
class ListSkillsTests(_BaseActTest):
    def test_no_directory(self):
        self.bc.SKILLS_DIR = os.path.join(tempfile.gettempdir(), "definitely_absent_skills_xyz")
        with mock.patch.object(A.os.path, "isdir", return_value=False):
            self.assertEqual(A._act_list_skills(""), "no skills directory yet")

    def test_lists_py_files_excluding_dunder(self):
        self.bc.SKILLS_DIR = "/fake/skills"
        listing = ["alpha.py", "beta.py", "_private.py", "notes.txt", "__init__.py"]
        with mock.patch.object(A.os.path, "isdir", return_value=True), \
             mock.patch.object(A.os, "listdir", return_value=listing):
            out = A._act_list_skills("")
        self.assertIn("installed skills:", out)
        self.assertIn("alpha", out)
        self.assertIn("beta", out)
        self.assertNotIn("_private", out)
        self.assertNotIn("__init__", out)
        self.assertNotIn("notes", out)

    def test_empty_after_filter(self):
        self.bc.SKILLS_DIR = "/fake/skills"
        with mock.patch.object(A.os.path, "isdir", return_value=True), \
             mock.patch.object(A.os, "listdir", return_value=["_x.py", "readme.md"]):
            self.assertEqual(A._act_list_skills(""), "no skills installed yet")


# ── _act_apple_music ─────────────────────────────────────────────────────────
class AppleMusicTests(_BaseActTest):
    def test_playlist_route(self):
        self.bc._looks_like_playlist_request.return_value = (True, "Chill Mix")
        self.bc._apple_music_play_playlist.return_value = "playing playlist Chill Mix"
        out = A._act_apple_music("my chill mix playlist")
        self.bc._apple_music_play_playlist.assert_called_once_with("Chill Mix")
        self.bc._streaming_auto_play.assert_not_called()
        self.assertEqual(out, "playing playlist Chill Mix")

    def test_search_route_when_not_playlist(self):
        self.bc._looks_like_playlist_request.return_value = (False, None)
        self.bc._streaming_auto_play.return_value = "playing track"
        out = A._act_apple_music("some song")
        self.bc._streaming_auto_play.assert_called_once_with("apple_music", "some song")
        self.assertEqual(out, "playing track")

    def test_playlist_flag_but_empty_name_falls_through_to_search(self):
        self.bc._looks_like_playlist_request.return_value = (True, "")
        self.bc._streaming_auto_play.return_value = "searched"
        A._act_apple_music("playlist:")
        self.bc._apple_music_play_playlist.assert_not_called()
        self.bc._streaming_auto_play.assert_called_once_with("apple_music", "playlist:")


# ── _act_launch_app ──────────────────────────────────────────────────────────
class LaunchAppTests(_BaseActTest):
    def test_apple_music_launches_via_bridge(self):
        # "open apple music" routes to the UWP bridge's launch(), NOT a doomed
        # exe / startfile lookup.
        amapp = self.patch_apple_music_app()
        amapp.launch.return_value = (True, None)
        with mock.patch.object(A.subprocess, "Popen") as popen, \
                mock.patch.object(A.os, "startfile", create=True) as startfile:
            out = A._act_launch_app("Apple Music")
        amapp.launch.assert_called_once_with()
        self.assertEqual(out, "launched Apple Music")
        popen.assert_not_called()
        startfile.assert_not_called()
        self.bc._resolve_known_app.assert_not_called()

    def test_music_app_alias_launches_via_bridge(self):
        amapp = self.patch_apple_music_app()
        out = A._act_launch_app("music app")
        amapp.launch.assert_called_once_with()
        self.assertEqual(out, "launched Apple Music")

    def test_apple_music_launch_failure_reported(self):
        amapp = self.patch_apple_music_app()
        amapp.launch.return_value = (False, "explorer denied")
        out = A._act_launch_app("apple music")
        self.assertIn("could not launch Apple Music", out)
        self.assertIn("explorer denied", out)

    def test_known_app_launches(self):
        self.bc._resolve_known_app.return_value = r"C:\Apps\bambu.exe"
        with mock.patch.object(A.subprocess, "Popen") as popen:
            out = A._act_launch_app("bambu studio")
        popen.assert_called_once_with([r"C:\Apps\bambu.exe"], close_fds=True)
        self.assertEqual(out, "launched bambu studio")

    def test_known_app_popen_error(self):
        self.bc._resolve_known_app.return_value = r"C:\Apps\bambu.exe"
        with mock.patch.object(A.subprocess, "Popen", side_effect=OSError("boom")):
            out = A._act_launch_app("bambu studio")
        self.assertIn("could not launch bambu studio", out)
        self.assertIn("boom", out)

    def test_path_executable_found(self):
        self.bc._resolve_known_app.return_value = None
        with mock.patch.object(A.shutil, "which", return_value=r"C:\bin\code.exe"), \
             mock.patch.object(A.subprocess, "Popen") as popen:
            out = A._act_launch_app("code")
        popen.assert_called_once_with([r"C:\bin\code.exe"], close_fds=True)
        self.assertEqual(out, "launched code")

    def test_win32_startfile_when_not_on_path(self):
        self.bc._resolve_known_app.return_value = None
        with mock.patch.object(A.shutil, "which", return_value=None), \
             mock.patch.object(A.sys, "platform", "win32"), \
             mock.patch.object(A.os, "startfile", create=True) as startfile:
            out = A._act_launch_app("notepad")
        startfile.assert_called_once_with("notepad")
        self.assertEqual(out, "launched notepad")

    def test_non_win_fallback_popen_name(self):
        self.bc._resolve_known_app.return_value = None
        with mock.patch.object(A.shutil, "which", return_value=None), \
             mock.patch.object(A.sys, "platform", "linux"), \
             mock.patch.object(A.subprocess, "Popen") as popen:
            out = A._act_launch_app("gedit")
        popen.assert_called_once_with(["gedit"], close_fds=True)
        self.assertEqual(out, "launched gedit")

    def test_path_branch_error_is_caught(self):
        self.bc._resolve_known_app.return_value = None
        with mock.patch.object(A.shutil, "which", return_value=r"C:\bin\x.exe"), \
             mock.patch.object(A.subprocess, "Popen", side_effect=OSError("nope")):
            out = A._act_launch_app("x")
        self.assertIn("could not launch x", out)


# ── _act_pause_music / _act_resume_music ─────────────────────────────────────
# Classic iTunes COM is DEAD: pause/resume now press the OS playpause media key
# when the browser Apple Music tab OR the new UWP Apple Music app is live, and
# return an honest "nothing's playing" line otherwise. Neither path touches
# _get_itunes anymore.
class PauseResumeMusicTests(_BaseActTest):
    def test_pause_routes_to_media_when_chrome_active(self):
        self.bc._apple_music_chrome_active.return_value = True
        self.bc._media_key_with_focus.return_value = "media play/pause pressed"
        out = A._act_pause_music("")
        # _act_pause_music -> _act_media_playpause -> bc._media_key_with_focus
        self.bc._media_key_with_focus.assert_called_once()
        self.assertEqual(out, "media play/pause pressed")

    def test_pause_routes_to_media_when_uwp_app_active(self):
        self.bc._apple_music_chrome_active.return_value = False
        self.bc._media_key_with_focus.return_value = "media play/pause pressed"
        self.patch_apple_music_app(is_active=True)
        out = A._act_pause_music("")
        self.bc._media_key_with_focus.assert_called_once()
        self.assertEqual(out, "media play/pause pressed")

    def test_pause_nothing_playing_honest_message(self):
        self.bc._apple_music_chrome_active.return_value = False
        self.patch_apple_music_app(is_active=False)
        out = A._act_pause_music("")
        self.assertIn("Nothing seems to be playing", out)
        # Must NOT call the dead iTunes COM.
        self.bc._get_itunes.assert_not_called()

    def test_pause_never_calls_dead_com(self):
        self.bc._apple_music_chrome_active.return_value = False
        self.patch_apple_music_app(is_active=True)
        A._act_pause_music("")
        self.bc._get_itunes.assert_not_called()

    def test_pause_bridge_unimportable_honest_message(self):
        self.bc._apple_music_chrome_active.return_value = False
        self.patch_apple_music_app_none()
        out = A._act_pause_music("")
        self.assertIn("Nothing seems to be playing", out)

    def test_resume_routes_to_media_when_uwp_app_active(self):
        self.bc._apple_music_chrome_active.return_value = False
        self.bc._media_key_with_focus.return_value = "media play/pause pressed"
        self.patch_apple_music_app(is_active=True)
        out = A._act_resume_music("")
        self.bc._media_key_with_focus.assert_called_once()
        self.assertEqual(out, "media play/pause pressed")

    def test_resume_routes_to_media_when_chrome_active(self):
        self.bc._apple_music_chrome_active.return_value = True
        self.bc._media_key_with_focus.return_value = "media play/pause pressed"
        self.assertEqual(A._act_resume_music(""), "media play/pause pressed")

    def test_resume_nothing_playing_honest_message(self):
        self.bc._apple_music_chrome_active.return_value = False
        self.patch_apple_music_app(is_active=False)
        out = A._act_resume_music("")
        self.assertIn("Nothing seems to be playing", out)
        self.bc._get_itunes.assert_not_called()


# ── _act_now_playing ─────────────────────────────────────────────────────────
# COM is dead. now_playing tries the UWP app's window title first, then the
# browser tab title — parsing the REAL "<Song> — <Artist>" track out of it via
# the monolith's _apple_music_title_now_playing (NOT echoing the raw window
# title, which gave the useless "Apple Music: Apple Music") — then an honest
# fallback. Never touches _get_itunes.
class NowPlayingTests(_BaseActTest):
    def setUp(self):
        super().setUp()
        # _act_now_playing consults the OS media session (SMTC) first; pin it
        # off here so these cases exercise the Apple Music fallback they cover.
        _p = mock.patch("core.media_now_playing.get_now_playing", return_value=None)
        _p.start()
        self.addCleanup(_p.stop)

    def test_smtc_session_wins(self):
        # When the OS media session reports a track, it is named first —
        # source-agnostic, no window-title scraping.
        with mock.patch("core.media_now_playing.get_now_playing", return_value={
                "app": "Chrome", "title": "The Lady in My Life",
                "artist": "Michael Jackson", "status": "playing", "playing": True}):
            out = A._act_now_playing("")
        self.assertEqual(
            out, "The Lady in My Life by Michael Jackson — playing in Chrome, sir.")

    def test_uwp_app_now_playing_wins(self):
        self.patch_apple_music_app(now_playing="Smooth Criminal")
        # Even if chrome is also "active", the app's reading is preferred.
        self.bc._apple_music_chrome_active.return_value = True
        out = A._act_now_playing("")
        self.assertEqual(out, "Apple Music: Smooth Criminal")

    def test_chrome_active_returns_parsed_track(self):
        # The browser tab is playing — the monolith helper parses the real
        # song/artist out of the tab title and now_playing reports THAT, not
        # the raw window title.
        self.patch_apple_music_app(now_playing=None, is_active=False)
        self.bc._apple_music_chrome_active.return_value = True
        self.bc._apple_music_title_now_playing.return_value = "Billie Jean — Michael Jackson"
        out = A._act_now_playing("")
        self.assertEqual(out, "Apple Music: Billie Jean — Michael Jackson")

    def test_chrome_active_idle_title_is_honest_not_echoed(self):
        # Regression for "Apple Music: Apple Music": when nothing is playing
        # the helper returns None, so we must NOT echo a bare service title.
        self.patch_apple_music_app(now_playing=None, is_active=False)
        self.bc._apple_music_chrome_active.return_value = True
        self.bc._apple_music_title_now_playing.return_value = None
        # No track loaded in the web player either (page-title reporter).
        self.bc._apple_music_loaded_track_from_title.return_value = None
        out = A._act_now_playing("")
        self.assertNotIn("Apple Music: Apple Music", out)
        self.assertIn("nothing seems to be playing", out.lower())

    def test_chrome_active_helper_raises_falls_back(self):
        # If the title helper blows up, degrade to the honest line rather
        # than crashing the action.
        self.patch_apple_music_app(now_playing=None, is_active=False)
        self.bc._apple_music_chrome_active.return_value = True
        self.bc._apple_music_title_now_playing.side_effect = RuntimeError("boom")
        self.bc._apple_music_loaded_track_from_title.return_value = None
        out = A._act_now_playing("")
        self.assertIn("nothing seems to be playing", out.lower())

    def test_chrome_active_reports_loaded_track_from_page_title(self):
        # The web player keeps a page title even while playing, so the strict
        # now-playing helper returns None; the page-title reporter still names
        # the loaded track instead of claiming nothing is playing.
        self.patch_apple_music_app(now_playing=None, is_active=False)
        self.bc._apple_music_chrome_active.return_value = True
        self.bc._apple_music_title_now_playing.return_value = None
        self.bc._apple_music_loaded_track_from_title.return_value = (
            "Billie Jean by Michael Jackson")
        out = A._act_now_playing("")
        self.assertIn("Billie Jean by Michael Jackson", out)

    def test_app_running_but_no_title_honest(self):
        # App is the live media app but its title gave no track → in-between line.
        self.patch_apple_music_app(now_playing=None, is_active=True)
        self.bc._apple_music_chrome_active.return_value = False
        out = A._act_now_playing("")
        self.assertIn("isn't telling me the track name", out)
        self.bc._get_itunes.assert_not_called()

    def test_nothing_playing_honest(self):
        self.patch_apple_music_app(now_playing=None, is_active=False)
        self.bc._apple_music_chrome_active.return_value = False
        out = A._act_now_playing("")
        self.assertIn("Nothing seems to be playing", out)
        self.bc._get_itunes.assert_not_called()

    def test_bridge_unimportable_falls_to_browser_then_honest(self):
        self.patch_apple_music_app_none()
        self.bc._apple_music_chrome_active.return_value = False
        out = A._act_now_playing("")
        self.assertIn("Nothing seems to be playing", out)


# ── _act_open_apple_music / _act_music_status ────────────────────────────────
class OpenAppleMusicTests(_BaseActTest):
    def test_already_running(self):
        self.patch_apple_music_app(running=True)
        self.assertIn("already open", A._act_open_apple_music(""))

    def test_launches_when_not_running(self):
        amapp = self.patch_apple_music_app(running=False)
        amapp.launch.return_value = (True, None)
        out = A._act_open_apple_music("")
        amapp.launch.assert_called_once_with()
        self.assertEqual(out, "launched Apple Music")

    def test_launch_failure_reported(self):
        amapp = self.patch_apple_music_app(running=False)
        amapp.launch.return_value = (False, "no explorer")
        out = A._act_open_apple_music("")
        self.assertIn("could not launch Apple Music", out)
        self.assertIn("no explorer", out)

    def test_bridge_unavailable(self):
        self.patch_apple_music_app_none()
        self.assertIn("isn't available", A._act_open_apple_music(""))


class MusicStatusTests(_BaseActTest):
    def test_running_with_now_playing(self):
        self.patch_apple_music_app(running=True, now_playing="Thriller")
        out = A._act_music_status("")
        self.assertIn("running", out)
        self.assertIn("Thriller", out)

    def test_running_nothing_playing(self):
        self.patch_apple_music_app(running=True, now_playing=None)
        out = A._act_music_status("")
        self.assertIn("running", out)
        self.assertIn("nothing is playing", out)

    def test_installed_not_running(self):
        self.patch_apple_music_app(running=False, installed=True)
        out = A._act_music_status("")
        self.assertIn("installed but not running", out)

    def test_not_running_unknown_install(self):
        self.patch_apple_music_app(running=False, installed=False)
        out = A._act_music_status("")
        self.assertIn("doesn't appear to be running", out)

    def test_bridge_unavailable(self):
        self.patch_apple_music_app_none()
        self.assertIn("isn't available", A._act_music_status(""))


# ── _act_queue_task ──────────────────────────────────────────────────────────
class QueueTaskTests(_BaseActTest):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.bc.TODO_FILE = os.path.join(self.tmp, "jarvis_todo.md")

    def test_empty_args(self):
        self.assertIn("format: queue_task", A._act_queue_task("   "))
        self.assertFalse(os.path.exists(self.bc.TODO_FILE))

    def test_creates_file_with_header_and_entry(self):
        with mock.patch.object(A.time, "strftime", return_value="2026-06-01 12:00"):
            out = A._act_queue_task("fix the thing")
        self.assertEqual(out, "queued: fix the thing")
        with open(self.bc.TODO_FILE, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("# JARVIS Task Queue", content)
        self.assertIn("- [ ] **2026-06-01 12:00** — fix the thing", content)

    def test_appends_without_duplicating_header(self):
        with mock.patch.object(A.time, "strftime", return_value="2026-06-01 12:00"):
            A._act_queue_task("first")
            A._act_queue_task("second")
        with open(self.bc.TODO_FILE, encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content.count("# JARVIS Task Queue"), 1)
        self.assertIn("first", content)
        self.assertIn("second", content)

    def test_long_description_truncated_in_reply(self):
        long = "x" * 200
        with mock.patch.object(A.time, "strftime", return_value="t"):
            out = A._act_queue_task(long)
        self.assertEqual(out, "queued: " + "x" * 80)


# ── _act_list_windows ────────────────────────────────────────────────────────
class ListWindowsTests(_BaseActTest):
    def test_pygetwindow_missing(self):
        # Make ``import pygetwindow`` raise ImportError inside the handler even
        # though the real package is installed on this dev box. Patch the
        # builtin __import__ so only the pygetwindow name fails.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "pygetwindow":
                raise ImportError("no pgw")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            out = A._act_list_windows("")
        self.assertIn("pygetwindow not available", out)

    def test_lists_sorted_unique_titles(self):
        gw = self.make_pygetwindow(all_windows=[
            _FakeWindow(title="Zebra"),
            _FakeWindow(title="Apple"),
            _FakeWindow(title="Apple"),      # duplicate collapses
            _FakeWindow(title="   "),         # blank dropped
            _FakeWindow(title=""),            # empty dropped
        ])
        self.install_fake_module("pygetwindow", gw)
        out = A._act_list_windows("")
        self.assertTrue(out.startswith("Open windows:"))
        self.assertIn("  - Apple", out)
        self.assertIn("  - Zebra", out)
        # sorted: Apple before Zebra
        self.assertLess(out.index("Apple"), out.index("Zebra"))
        self.assertEqual(out.count("Apple"), 1)

    def test_no_windows_visible(self):
        gw = self.make_pygetwindow(all_windows=[_FakeWindow(title="")])
        self.install_fake_module("pygetwindow", gw)
        self.assertEqual(A._act_list_windows(""), "no windows visible")


# ── _act_focus_window ────────────────────────────────────────────────────────
class FocusWindowTests(_BaseActTest):
    def test_empty_query(self):
        self.assertIn("format: focus_window", A._act_focus_window("  "))

    def test_no_match(self):
        self.bc._find_windows_by_title.return_value = []
        self.assertEqual(A._act_focus_window("ghost"), "no window matching 'ghost'")

    def test_activate_success(self):
        win = _FakeWindow(title="Editor")
        self.bc._find_windows_by_title.return_value = [win]
        out = A._act_focus_window("edit")
        self.assertTrue(win.activated)
        self.bc._flash_window_reticle.assert_called_once_with(win, "focus")
        self.assertEqual(out, "focused 'Editor'")

    def test_benign_windows_error_treated_as_success(self):
        win = _FakeWindow(
            title="Editor",
            on_activate=lambda: (_ for _ in ()).throw(
                Exception("Operation completed successfully.")),
        )
        self.bc._find_windows_by_title.return_value = [win]
        out = A._act_focus_window("edit")
        self.assertEqual(out, "focused 'Editor'")
        self.bc._flash_window_reticle.assert_called_once_with(win, "focus")

    def test_fallback_restore_trick(self):
        win = _FakeWindow(
            title="Editor",
            on_activate=lambda: (_ for _ in ()).throw(Exception("real failure")),
        )
        self.bc._find_windows_by_title.return_value = [win]
        out = A._act_focus_window("edit")
        self.assertTrue(win.minimized and win.restored)
        self.assertEqual(out, "focused 'Editor' (via restore)")

    def test_fallback_restore_also_fails(self):
        def boom():
            raise Exception("hard fail")
        win = _FakeWindow(title="Editor", on_activate=boom, on_minimize=boom)
        self.bc._find_windows_by_title.return_value = [win]
        out = A._act_focus_window("edit")
        self.assertIn("could not focus 'Editor'", out)


# ── _act_minimize_window ─────────────────────────────────────────────────────
class MinimizeWindowTests(_BaseActTest):
    def test_empty_query(self):
        self.assertIn("format: minimize_window", A._act_minimize_window(""))

    def test_no_match(self):
        self.bc._find_windows_by_title.return_value = []
        self.assertEqual(A._act_minimize_window("x"), "no window matching 'x'")

    def test_minimizes_all_matches(self):
        w1, w2 = _FakeWindow(title="A"), _FakeWindow(title="B")
        self.bc._find_windows_by_title.return_value = [w1, w2]
        out = A._act_minimize_window("ab")
        self.assertTrue(w1.minimized and w2.minimized)
        self.assertIn("minimized:", out)
        self.assertIn("A", out)
        self.assertIn("B", out)

    def test_all_minimize_calls_fail(self):
        def boom():
            raise Exception("fail")
        w = _FakeWindow(title="A", on_minimize=boom)
        self.bc._find_windows_by_title.return_value = [w]
        self.assertEqual(A._act_minimize_window("a"), "could not minimize")

    def test_partial_failure_reports_succeeded_only(self):
        def boom():
            raise Exception("fail")
        good = _FakeWindow(title="Good")
        bad = _FakeWindow(title="Bad", on_minimize=boom)
        self.bc._find_windows_by_title.return_value = [good, bad]
        out = A._act_minimize_window("x")
        self.assertIn("Good", out)
        self.assertNotIn("Bad", out)


# ── _act_close_window ────────────────────────────────────────────────────────
class CloseWindowTests(_BaseActTest):
    def setUp(self):
        super().setUp()
        self.bc.FORBIDDEN_TARGETS = ["bobert_companion", "jarvis terminal"]

    def test_empty_query(self):
        self.assertIn("format: close_window", A._act_close_window(""))

    def test_refuses_forbidden_query(self):
        out = A._act_close_window("the JARVIS Terminal window")
        self.assertIn("REFUSED", out)
        self.bc._find_windows_by_title.assert_not_called()

    def test_no_match(self):
        self.bc._find_windows_by_title.return_value = []
        self.assertEqual(A._act_close_window("ghost"), "no window matching 'ghost'")

    def test_closes_allowed_windows(self):
        w = _FakeWindow(title="Notepad")
        self.bc._find_windows_by_title.return_value = [w]
        out = A._act_close_window("notepad")
        self.assertTrue(w.closed)
        self.assertIn("closed:", out)
        self.assertIn("Notepad", out)

    def test_skips_forbidden_window_found_in_results(self):
        # Query is innocuous but a matched window's title is forbidden ->
        # defence-in-depth skips it.
        safe = _FakeWindow(title="Notepad")
        host = _FakeWindow(title="bobert_companion console")
        self.bc._find_windows_by_title.return_value = [safe, host]
        out = A._act_close_window("window")
        self.assertTrue(safe.closed)
        self.assertFalse(host.closed)
        self.assertIn("Notepad", out)
        self.assertNotIn("bobert_companion", out)

    def test_close_exception_swallowed(self):
        def boom():
            raise Exception("locked")
        w = _FakeWindow(title="Notepad", on_close=boom)
        self.bc._find_windows_by_title.return_value = [w]
        self.assertEqual(A._act_close_window("notepad"), "could not close")


# ── _act_type ────────────────────────────────────────────────────────────────
class TypeTests(_BaseActTest):
    def test_refuses_shell_command_without_terminal(self):
        self.bc._looks_like_shell_command.return_value = True
        self.bc._active_window_is_terminal.return_value = False
        out = A._act_type("rm -rf /tmp/x\nsecond line")
        self.assertIn("REFUSED", out)
        self.assertIn("run_shell", out)
        self.bc.ui_type.assert_not_called()

    def test_types_when_terminal_focused(self):
        self.bc._looks_like_shell_command.return_value = True
        self.bc._active_window_is_terminal.return_value = True
        out = A._act_type("ls -la")
        self.bc.ui_type.assert_called_once_with("ls -la")
        self.assertEqual(out, "typed: ls -la")

    def test_types_plain_text(self):
        self.bc._looks_like_shell_command.return_value = False
        out = A._act_type("hello world")
        self.bc.ui_type.assert_called_once_with("hello world")
        self.assertEqual(out, "typed: hello world")

    def test_long_text_truncated_with_ellipsis(self):
        self.bc._looks_like_shell_command.return_value = False
        text = "a" * 70
        out = A._act_type(text)
        self.assertEqual(out, "typed: " + "a" * 60 + "...")

    def test_failsafe_message(self):
        self.bc._looks_like_shell_command.return_value = False
        self.bc.ui_type.side_effect = _UIFailsafe("locked out")
        self.assertEqual(A._act_type("x"), "locked out")


# ── _act_next_song / _act_previous_song ──────────────────────────────────────
class NextPrevSongTests(_BaseActTest):
    # COM is dead → next/previous press the OS media key when the browser tab
    # OR the UWP app is live, else an honest line. Never touches _get_itunes /
    # _run_itunes_com_timeout.
    def test_next_chrome_active_routes_media(self):
        self.bc._apple_music_chrome_active.return_value = True
        self.bc._media_key_with_focus.return_value = "media next pressed"
        self.assertEqual(A._act_next_song(""), "media next pressed")

    def test_next_uwp_app_active_routes_media(self):
        self.bc._apple_music_chrome_active.return_value = False
        self.bc._media_key_with_focus.return_value = "media next pressed"
        self.patch_apple_music_app(is_active=True)
        out = A._act_next_song("")
        self.assertEqual(out, "media next pressed")
        self.bc._get_itunes.assert_not_called()

    def test_next_nothing_playing_honest(self):
        self.bc._apple_music_chrome_active.return_value = False
        self.patch_apple_music_app(is_active=False)
        out = A._act_next_song("")
        self.assertIn("Nothing seems to be playing", out)
        self.bc._get_itunes.assert_not_called()

    def test_next_bridge_unimportable_honest(self):
        self.bc._apple_music_chrome_active.return_value = False
        self.patch_apple_music_app_none()
        self.assertIn("Nothing seems to be playing", A._act_next_song(""))

    def test_prev_chrome_active_routes_media(self):
        self.bc._apple_music_chrome_active.return_value = True
        self.bc._media_key_with_focus.return_value = "media previous pressed"
        self.assertEqual(A._act_previous_song(""), "media previous pressed")

    def test_prev_uwp_app_active_routes_media(self):
        self.bc._apple_music_chrome_active.return_value = False
        self.bc._media_key_with_focus.return_value = "media previous pressed"
        self.patch_apple_music_app(is_active=True)
        out = A._act_previous_song("")
        self.assertEqual(out, "media previous pressed")
        self.bc._get_itunes.assert_not_called()

    def test_prev_nothing_playing_honest(self):
        self.bc._apple_music_chrome_active.return_value = False
        self.patch_apple_music_app(is_active=False)
        out = A._act_previous_song("")
        self.assertIn("Nothing seems to be playing", out)
        self.bc._get_itunes.assert_not_called()


# ── _act_show_tasks ──────────────────────────────────────────────────────────
class ShowTasksTests(_BaseActTest):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.bc.TODO_FILE = os.path.join(self.tmp, "todo.md")

    def _write(self, text):
        with open(self.bc.TODO_FILE, "w", encoding="utf-8") as f:
            f.write(text)

    def test_no_file(self):
        self.assertEqual(A._act_show_tasks(""), "no tasks queued yet")

    def test_file_with_no_tasks(self):
        self._write("# Header\n\nsome prose\n")
        self.assertEqual(A._act_show_tasks(""), "the file exists but no tasks are in it")

    def test_all_done(self):
        self._write("- [x] done one\n- [x] done two\n")
        self.assertEqual(A._act_show_tasks(""), "all 2 task(s) are done — nothing left to do")

    def test_pending_only(self):
        self._write("- [ ] alpha\n- [ ] beta\n")
        out = A._act_show_tasks("")
        self.assertIn("2 pending task(s)", out)
        self.assertIn("- [ ] alpha", out)
        self.assertIn("- [ ] beta", out)
        self.assertNotIn("already done", out)

    def test_mixed_pending_and_done(self):
        self._write("- [ ] alpha\n- [x] gamma\n")
        out = A._act_show_tasks("")
        self.assertIn("1 pending task(s)", out)
        self.assertIn("(1 already done)", out)
        self.assertIn("- [ ] alpha", out)


# ── _act_ambient_mode_set ────────────────────────────────────────────────────
class AmbientModeSetTests(_BaseActTest):
    def setUp(self):
        super().setUp()
        self.bc._ambient_mode_active = [False]
        self.bc.ACTIONS = {}

    def test_turn_on_invokes_start_action(self):
        start = mock.Mock(return_value="")
        self.bc.ACTIONS = {"ambient_listen_start": start}
        out = A._act_ambient_mode_set(True)
        self.assertTrue(self.bc._ambient_mode_active[0])
        self.bc._write_hud_state.assert_called_once_with(ambient_mode_active=True)
        start.assert_called_once_with("")
        self.assertIn("active", out)
        self.assertIn("listening quietly", out)

    def test_turn_off_invokes_stop_action(self):
        self.bc._ambient_mode_active = [True]
        stop = mock.Mock(return_value="")
        self.bc.ACTIONS = {"ambient_listen_stop": stop}
        out = A._act_ambient_mode_set(False)
        self.assertFalse(self.bc._ambient_mode_active[0])
        stop.assert_called_once_with("")
        self.assertIn("off", out)
        self.assertIn("standing down", out)

    def test_missing_action_is_tolerated(self):
        self.bc.ACTIONS = {}  # no ambient_listen_* registered
        out = A._act_ambient_mode_set(True)
        self.assertIn("active", out)

    def test_daemon_refusal_surfaced(self):
        start = mock.Mock(side_effect=RuntimeError("busy"))
        self.bc.ACTIONS = {"ambient_listen_start": start}
        out = A._act_ambient_mode_set(True)
        self.assertIn("ambient daemon refused", out)
        self.assertIn("busy", out)

    def test_truthy_coerced_to_bool(self):
        self.bc.ACTIONS = {"ambient_listen_start": mock.Mock(return_value="")}
        A._act_ambient_mode_set("yes-truthy")
        # stored as real bool True, not the raw string
        self.assertIs(self.bc._ambient_mode_active[0], True)
        self.bc._write_hud_state.assert_called_once_with(ambient_mode_active=True)


# ── _act_reload_skills ───────────────────────────────────────────────────────
class ReloadSkillsTests(_BaseActTest):
    def test_skills_disabled(self):
        with mock.patch("core.config.SKILLS_ENABLED", False):
            out = A._act_reload_skills("")
        self.assertEqual(out, "skills disabled")
        self.bc._tray_async.assert_not_called()

    def test_schedules_async_reload(self):
        with mock.patch("core.config.SKILLS_ENABLED", True):
            out = A._act_reload_skills("")
        self.assertEqual(out, "reloading skills")
        self.bc._tray_async.assert_called_once()
        name, fn = self.bc._tray_async.call_args[0]
        self.assertEqual(name, "reload_skills")
        # Drive the deferred closure to cover its body (load_skills is a Mock).
        self.bc.ACTIONS = {"a": 1}
        self.bc.load_skills.return_value = None
        msg = fn()
        self.assertIn("skills reloaded", msg)

    def test_deferred_closure_handles_error(self):
        with mock.patch("core.config.SKILLS_ENABLED", True):
            A._act_reload_skills("")
        _, fn = self.bc._tray_async.call_args[0]
        self.bc.ACTIONS = {}
        self.bc.load_skills.side_effect = RuntimeError("import blew up")
        self.assertIn("reload_skills failed", fn())


# ── _act_show_recent_facts ───────────────────────────────────────────────────
class ShowRecentFactsTests(_BaseActTest):
    def setUp(self):
        super().setUp()
        # _memory_lock is used as a context manager: configure __enter__/__exit__.
        self.bc._memory_lock = mock.MagicMock()

    def test_no_facts(self):
        self.bc.load_memory.return_value = {"facts": []}
        self.assertEqual(A._act_show_recent_facts(""), "no facts in memory yet, sir")

    def test_missing_facts_key(self):
        self.bc.load_memory.return_value = {}
        self.assertEqual(A._act_show_recent_facts(""), "no facts in memory yet, sir")

    def test_tails_last_ten(self):
        facts = [f"fact {i}" for i in range(25)]
        self.bc.load_memory.return_value = {"facts": facts}
        out = A._act_show_recent_facts("")
        self.assertIn("showed 10 recent fact(s) of 25 total", out)

    def test_fewer_than_ten(self):
        self.bc.load_memory.return_value = {"facts": ["a", "b", "c"]}
        out = A._act_show_recent_facts("")
        self.assertIn("showed 3 recent fact(s) of 3 total", out)

    def test_exception_path(self):
        self.bc.load_memory.side_effect = RuntimeError("disk error")
        self.assertIn("show_recent_facts failed", A._act_show_recent_facts(""))


# ── _act_export_memory ───────────────────────────────────────────────────────
class ExportMemoryTests(_BaseActTest):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.bc._memory_lock = mock.MagicMock()
        self.bc.MEMORY_FILE = os.path.join(self.tmp, "bobert_memory.json")

    def test_no_memory_file(self):
        self.assertEqual(A._act_export_memory(""), "no memory file to export")

    def test_exports_copy_to_backups(self):
        with open(self.bc.MEMORY_FILE, "w", encoding="utf-8") as f:
            f.write('{"facts": []}')
        with mock.patch.object(A.time, "strftime", return_value="20260601_120000"):
            out = A._act_export_memory("")
        self.assertIn("memory exported -> backups/memory_export_20260601_120000.json", out)
        export_path = os.path.join(self.tmp, "backups", "memory_export_20260601_120000.json")
        self.assertTrue(os.path.exists(export_path))

    def test_exception_path(self):
        with open(self.bc.MEMORY_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
        with mock.patch.object(A.shutil, "copy2", side_effect=OSError("perm denied")):
            out = A._act_export_memory("")
        self.assertIn("export_memory failed", out)


# ── _act_run_diagnostic_tray / _act_show_last_diagnostic ─────────────────────
class DiagnosticTrayTests(_BaseActTest):
    def test_run_no_module(self):
        self.bc._selfdiag_module.return_value = None
        self.assertEqual(A._act_run_diagnostic_tray(""), "self_diagnostic skill not loaded")

    def test_run_module_without_attr(self):
        sd = mock.Mock(spec=[])  # no run_diagnostic attribute
        self.bc._selfdiag_module.return_value = sd
        self.assertEqual(A._act_run_diagnostic_tray(""), "self_diagnostic skill not loaded")

    def test_run_schedules_async(self):
        sd = mock.Mock()
        sd.run_diagnostic = mock.Mock(return_value="done")
        self.bc._selfdiag_module.return_value = sd
        out = A._act_run_diagnostic_tray("")
        self.assertEqual(out, "diagnostic sweep started")
        self.bc._tray_async.assert_called_once()
        name, fn = self.bc._tray_async.call_args[0]
        self.assertEqual(name, "run_diagnostic")
        fn()  # drive closure
        sd.run_diagnostic.assert_called_once_with("")

    def test_show_last_no_module(self):
        self.bc._selfdiag_module.return_value = None
        self.assertEqual(A._act_show_last_diagnostic(""), "self_diagnostic skill not loaded")

    def test_show_last_module_without_attr(self):
        sd = mock.Mock(spec=[])
        self.bc._selfdiag_module.return_value = sd
        self.assertEqual(A._act_show_last_diagnostic(""), "self_diagnostic skill not loaded")

    def test_show_last_prints_summary(self):
        sd = mock.Mock()
        sd.last_diagnostic_run = mock.Mock(return_value="line one\nline two\nline three")
        self.bc._selfdiag_module.return_value = sd
        out = A._act_show_last_diagnostic("")
        self.assertIn("printed last run", out)
        self.assertIn("chars total", out)

    def test_show_last_long_first_line_truncated(self):
        sd = mock.Mock()
        sd.last_diagnostic_run = mock.Mock(return_value="z" * 500)
        self.bc._selfdiag_module.return_value = sd
        # Should not raise; truncation happens internally.
        out = A._act_show_last_diagnostic("")
        self.assertIn("printed last run (500 chars total)", out)

    def test_show_last_none_return(self):
        sd = mock.Mock()
        sd.last_diagnostic_run = mock.Mock(return_value=None)
        self.bc._selfdiag_module.return_value = sd
        out = A._act_show_last_diagnostic("")
        self.assertIn("printed last run (0 chars total)", out)

    def test_show_last_exception(self):
        sd = mock.Mock()
        sd.last_diagnostic_run = mock.Mock(side_effect=RuntimeError("x"))
        self.bc._selfdiag_module.return_value = sd
        self.assertIn("show_last_diagnostic failed", A._act_show_last_diagnostic(""))


# ── _act_play_streaming ──────────────────────────────────────────────────────
class PlayStreamingTests(_BaseActTest):
    def setUp(self):
        super().setUp()
        self.bc._STREAMING_SERVICES = {"netflix": "url1", "youtube": "url2"}
        self.bc._normalize_service.side_effect = lambda s: s.strip().lower()
        self.bc._streaming_auto_play.return_value = "playing"

    def test_service_pipe_known(self):
        out = A._act_play_streaming("netflix | Stranger Things")
        self.bc._streaming_auto_play.assert_called_once_with("netflix", "Stranger Things")
        self.assertEqual(out, "playing")

    def test_service_pipe_unknown(self):
        out = A._act_play_streaming("hbo | something")
        self.assertIn("unknown service 'hbo'", out)
        self.assertIn("netflix", out)
        self.assertIn("youtube", out)
        self.bc._streaming_auto_play.assert_not_called()

    def test_no_pipe_defaults_to_youtube(self):
        out = A._act_play_streaming("  funny cats  ")
        self.bc._streaming_auto_play.assert_called_once_with("youtube", "funny cats")
        self.assertEqual(out, "playing")


# ── _act_click ───────────────────────────────────────────────────────────────
class ClickTests(_BaseActTest):
    def setUp(self):
        super().setUp()
        # default: no monitor prefix, pass args through unchanged
        self.bc._parse_monitor_prefix.side_effect = lambda a: (None, a)

    def test_coordinate_left_click(self):
        out = A._act_click("100, 200")
        self.bc.ui_click.assert_called_once_with(100, 200, "left")
        self.assertEqual(out, "clicked left at (100,200)")

    def test_coordinate_right_click(self):
        out = A._act_click("10,20,right")
        self.bc.ui_click.assert_called_once_with(10, 20, "right")
        self.assertEqual(out, "clicked right at (10,20)")

    def test_negative_coordinates(self):
        out = A._act_click("-2215, 249")
        self.bc.ui_click.assert_called_once_with(-2215, 249, "left")
        self.assertEqual(out, "clicked left at (-2215,249)")

    def test_coordinate_failsafe(self):
        self.bc.ui_click.side_effect = _UIFailsafe("blocked")
        self.assertEqual(A._act_click("1,2"), "blocked")

    def test_description_self_close_refused(self):
        self.bc._is_self_close_attempt.return_value = True
        out = A._act_click("close the terminal")
        self.assertIn("REFUSED", out)
        self.bc.find_click_target.assert_not_called()

    def test_description_found_and_clicked(self):
        self.bc._is_self_close_attempt.return_value = False
        self.bc.find_click_target.return_value = (300, 400)
        out = A._act_click("the play button")
        self.bc.find_click_target.assert_called_once_with("the play button", monitor=None)
        self.bc.ui_click.assert_called_once_with(300, 400)
        self.assertEqual(out, "clicked 'the play button' at (300, 400)")

    def test_description_not_found(self):
        self.bc._is_self_close_attempt.return_value = False
        self.bc.find_click_target.return_value = None
        out = A._act_click("nonexistent thing")
        self.assertIn("could not locate", out)

    def test_description_not_found_with_monitor(self):
        self.bc._parse_monitor_prefix.side_effect = lambda a: ("left", "the play button")
        self.bc._is_self_close_attempt.return_value = False
        self.bc.find_click_target.return_value = None
        out = A._act_click("monitor:left|the play button")
        self.bc.find_click_target.assert_called_once_with("the play button", monitor="left")
        self.assertIn("on left monitor", out)

    def test_description_click_failsafe(self):
        self.bc._is_self_close_attempt.return_value = False
        self.bc.find_click_target.return_value = (5, 6)
        self.bc.ui_click.side_effect = _UIFailsafe("nope")
        self.assertEqual(A._act_click("thing"), "nope")


# ── _act_hotkey ──────────────────────────────────────────────────────────────
class HotkeyTests(_BaseActTest):
    def setUp(self):
        super().setUp()
        self.bc._normalize_key.side_effect = lambda k: k.strip().lower()
        self.bc.FORBIDDEN_TARGETS = ["bobert_companion", "jarvis terminal"]

    def test_basic_hotkey(self):
        out = A._act_hotkey("ctrl+c")
        self.bc.ui_hotkey.assert_called_once_with("ctrl", "c")
        self.assertEqual(out, "pressed ctrl+c")

    def test_altf4_refused_when_host_focused(self):
        active = _FakeWindow(title="JARVIS Terminal")
        gw = self.make_pygetwindow(active=active)
        self.install_fake_module("pygetwindow", gw)
        out = A._act_hotkey("alt+f4")
        self.assertIn("REFUSED", out)
        self.bc.ui_hotkey.assert_not_called()

    def test_altf4_allowed_when_other_window_focused(self):
        active = _FakeWindow(title="Notepad")
        gw = self.make_pygetwindow(active=active)
        self.install_fake_module("pygetwindow", gw)
        out = A._act_hotkey("alt+f4")
        self.bc.ui_hotkey.assert_called_once_with("alt", "f4")
        self.assertEqual(out, "pressed alt+f4")

    def test_altf4_import_failure_proceeds(self):
        # No pygetwindow injected; inner import raises -> guarded, proceeds.
        self.install_fake_module(
            "pygetwindow",
            mock.Mock(getActiveWindow=mock.Mock(side_effect=RuntimeError("x"))),
        )
        out = A._act_hotkey("alt+f4")
        self.bc.ui_hotkey.assert_called_once_with("alt", "f4")
        self.assertEqual(out, "pressed alt+f4")

    def test_hotkey_failsafe(self):
        self.bc.ui_hotkey.side_effect = _UIFailsafe("locked")
        self.assertEqual(A._act_hotkey("ctrl+a"), "locked")

    def test_hotkey_generic_exception(self):
        self.bc.ui_hotkey.side_effect = RuntimeError("weird")
        out = A._act_hotkey("ctrl+a")
        self.assertIn("hotkey failed", out)
        self.assertIn("weird", out)


# ── _act_stop_pipeline ───────────────────────────────────────────────────────
class StopPipelineTests(_BaseActTest):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.bc.OVERNIGHT_FLAG_FILE = os.path.join(self.tmp, "overnight.flag")
        self.bc._sleep_mode = [False]
        # _overnight_run_now: event-like with is_set/clear
        self.bc._overnight_run_now = mock.Mock()
        self.bc._overnight_run_now.is_set.return_value = False

    def test_nothing_pending(self):
        out = A._act_stop_pipeline("")
        self.assertEqual(out, "nothing pending to halt")

    def test_clears_run_now_flag(self):
        self.bc._overnight_run_now.is_set.return_value = True
        out = A._act_stop_pipeline("")
        self.bc._overnight_run_now.clear.assert_called_once_with()
        self.assertEqual(out, "overnight engine quieted")

    def test_clears_sleep_mode(self):
        self.bc._sleep_mode = [True]
        out = A._act_stop_pipeline("")
        self.assertFalse(self.bc._sleep_mode[0])
        self.assertEqual(out, "overnight engine quieted")

    def test_removes_flag_file_and_writes_hud(self):
        with open(self.bc.OVERNIGHT_FLAG_FILE, "w") as f:
            f.write("x")
        out = A._act_stop_pipeline("")
        self.assertFalse(os.path.exists(self.bc.OVERNIGHT_FLAG_FILE))
        self.bc._write_hud_state.assert_called_once_with(overnight_expiry=0.0)
        self.assertEqual(out, "overnight engine quieted")

    def test_run_now_is_set_raises_is_swallowed(self):
        self.bc._overnight_run_now.is_set.side_effect = RuntimeError("boom")
        # other branches yield nothing -> still "nothing pending"
        out = A._act_stop_pipeline("")
        self.assertEqual(out, "nothing pending to halt")

    def test_flag_removal_error_swallowed(self):
        with open(self.bc.OVERNIGHT_FLAG_FILE, "w") as f:
            f.write("x")
        with mock.patch.object(A.os, "remove", side_effect=OSError("locked")):
            out = A._act_stop_pipeline("")
        # remove failed -> nothing counted as cleared
        self.assertEqual(out, "nothing pending to halt")

    def test_sleep_mode_access_raises_is_swallowed(self):
        # _sleep_mode[0] read raises -> caught by the surrounding except (line 985).
        class _Boom:
            def __getitem__(self, i):
                raise RuntimeError("boom")
        self.bc._sleep_mode = _Boom()
        out = A._act_stop_pipeline("")
        # run_now not set, flag absent -> nothing cleared
        self.assertEqual(out, "nothing pending to halt")

    def test_hud_write_error_during_flag_removal_swallowed(self):
        # Flag removed (cleared=True) but _write_hud_state raises -> inner
        # except (line 992) swallows it; outer result still "quieted".
        with open(self.bc.OVERNIGHT_FLAG_FILE, "w") as f:
            f.write("x")
        self.bc._write_hud_state.side_effect = RuntimeError("hud down")
        out = A._act_stop_pipeline("")
        self.assertFalse(os.path.exists(self.bc.OVERNIGHT_FLAG_FILE))
        self.assertEqual(out, "overnight engine quieted")


if __name__ == "__main__":
    unittest.main()
