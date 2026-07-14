"""Unit tests for core/actions.py — section 1: the ``_act_*`` handlers
defined between lines 46 and 377 (``_act_open_url`` .. ``_act_restart``),
plus the in-range helper ``_set_unified_hud_hidden``.

Design (CI-safe, no heavy deps, no @requires_monolith):

  * Every handler reaches the ~14K-line monolith via the module-level
    ``_bc()`` helper. We patch ``core.actions._bc`` to return a configured
    ``mock.Mock()`` so the real bobert_companion is NEVER imported — the
    tests run identically on CI (Linux + light deps) and locally.

  * All other I/O is mocked at the boundary: ``webbrowser.open``,
    ``time.sleep``, ``time.strftime`` (frozen), ``subprocess``,
    ``threading.Thread`` (no-op / inline), ``os._exit``, and the
    filesystem (redirected to per-test ``tempfile`` dirs).

  * Patches are per-test (``with`` blocks / addCleanup) so they auto-restore
    and the suite stays isolated. No module-level ``sys.modules`` writes.
    ``core.config`` is imported by ``_act_show_llm_stats`` but is a light
    stdlib-only module, so it is exercised for real.

Only ``tests/test_actions_sec1.py`` is created; the source is untouched.
Bugs found are documented in NOTE comments, not fixed.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import core.actions as A


# ─────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────
def _patch_bc(fake):
    """Context manager: make A._bc() return ``fake`` for the duration."""
    return mock.patch.object(A, "_bc", return_value=fake)


def _no_sleep():
    """Patch time.sleep in core.actions to a no-op (handlers call time.sleep
    for page-load waits we don't want in tests)."""
    return mock.patch.object(A.time, "sleep", return_value=None)


# ─────────────────────────────────────────────────────────────────────────
# Browser + search basics
# ─────────────────────────────────────────────────────────────────────────
class OpenUrlTests(unittest.TestCase):
    def test_bare_host_gets_https_prefix(self):
        with mock.patch.object(A.webbrowser, "open") as mopen, _no_sleep():
            out = A._act_open_url("example.com")
        mopen.assert_called_once_with("https://example.com")
        self.assertIn("opened https://example.com", out)
        self.assertIn("see_screen", out)

    def test_http_url_passed_through_unchanged(self):
        with mock.patch.object(A.webbrowser, "open") as mopen, _no_sleep():
            out = A._act_open_url("http://example.com/page")
        mopen.assert_called_once_with("http://example.com/page")
        self.assertIn("http://example.com/page", out)

    def test_https_url_passed_through_unchanged(self):
        with mock.patch.object(A.webbrowser, "open") as mopen, _no_sleep():
            A._act_open_url("https://secure.example.com")
        mopen.assert_called_once_with("https://secure.example.com")

    def test_waits_for_page_load(self):
        with mock.patch.object(A.webbrowser, "open"), \
                mock.patch.object(A.time, "sleep") as msleep:
            A._act_open_url("example.com")
        msleep.assert_called_once_with(3.0)


class WebSearchTests(unittest.TestCase):
    def _bc(self, hints=("video", "youtube", "watch"), yt_url=None):
        fake = mock.Mock()
        fake._VIDEO_QUERY_HINTS = hints
        fake._extract_youtube_url_from_search.return_value = yt_url
        return fake

    def test_plain_query_opens_google_serp(self):
        fake = self._bc()
        with _patch_bc(fake), mock.patch.object(A.webbrowser, "open") as mopen, \
                _no_sleep():
            out = A._act_web_search("weather today")
        url = mopen.call_args[0][0]
        self.assertTrue(url.startswith("https://www.google.com/search?q="))
        # query is URL-quoted (space -> %20)
        self.assertIn("weather%20today", url)
        self.assertIn("Google search", out)
        self.assertIn("see_screen", out)
        # no video intent -> extraction never attempted
        fake._extract_youtube_url_from_search.assert_not_called()

    def test_video_intent_opens_extracted_youtube_url(self):
        fake = self._bc(yt_url="https://www.youtube.com/watch?v=abc123")
        with _patch_bc(fake), mock.patch.object(A.webbrowser, "open") as mopen, \
                _no_sleep():
            out = A._act_web_search("funny cat video")
        fake._extract_youtube_url_from_search.assert_called_once_with(
            "funny cat video")
        mopen.assert_called_once_with("https://www.youtube.com/watch?v=abc123")
        self.assertIn("video is now playing", out)
        self.assertIn("no further action needed", out)

    def test_video_intent_extraction_miss_falls_through_to_google(self):
        # video hint present but extractor returns None -> Google SERP fallback
        fake = self._bc(yt_url=None)
        with _patch_bc(fake), mock.patch.object(A.webbrowser, "open") as mopen, \
                _no_sleep():
            out = A._act_web_search("watch the game")
        fake._extract_youtube_url_from_search.assert_called_once()
        url = mopen.call_args[0][0]
        self.assertTrue(url.startswith("https://www.google.com/search?q="))
        self.assertIn("Google search", out)

    def test_video_intent_is_case_insensitive(self):
        fake = self._bc(yt_url="https://www.youtube.com/watch?v=z")
        with _patch_bc(fake), mock.patch.object(A.webbrowser, "open"), \
                _no_sleep():
            A._act_web_search("Funny VIDEO of dogs")
        fake._extract_youtube_url_from_search.assert_called_once()


class YoutubeSearchTests(unittest.TestCase):
    def test_opens_results_page_no_bc_needed(self):
        with mock.patch.object(A.webbrowser, "open") as mopen:
            out = A._act_youtube("lofi beats")
        url = mopen.call_args[0][0]
        self.assertTrue(
            url.startswith("https://www.youtube.com/results?search_query="))
        self.assertIn("lofi%20beats", url)
        self.assertEqual(out, "searching YouTube for lofi beats")


class GetTimeTests(unittest.TestCase):
    def test_formats_current_time(self):
        # strftime is deterministic given a fixed struct_time. Capture the
        # REAL strftime before patching so the side_effect doesn't recurse
        # into the mock (A.time is the same module object we patch).
        import time as _t
        real_strftime = _t.strftime
        fixed = _t.struct_time((2026, 6, 1, 14, 30, 0, 0, 152, 0))
        with mock.patch.object(A.time, "strftime",
                               side_effect=lambda fmt: real_strftime(fmt, fixed)):
            out = A._act_get_time("")
        self.assertEqual(
            out, "current time is 02:30 PM on Monday, June 01, 2026")

    def test_includes_real_calendar_date(self):
        # Regression: "what's today's date" routes to get_time, so the handler
        # must surface the actual month/day/year from the system clock — not a
        # bare weekday that forces the LLM to fabricate (off-by-one) the date.
        import time as _t
        real_strftime = _t.strftime
        fixed = _t.struct_time((2026, 6, 7, 14, 30, 0, 6, 158, 0))  # Sun Jun 7
        with mock.patch.object(A.time, "strftime",
                               side_effect=lambda fmt: real_strftime(fmt, fixed)):
            out = A._act_get_time("")
        # The real date components are present and grounded in the clock.
        self.assertIn("June 07, 2026", out)
        self.assertIn("Sunday", out)

    def test_accepts_ignored_arg(self):
        with mock.patch.object(A.time, "strftime", return_value="frozen"):
            self.assertEqual(A._act_get_time("anything"), "frozen")


# ─────────────────────────────────────────────────────────────────────────
# Screenshot
# ─────────────────────────────────────────────────────────────────────────
class ScreenshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jv_shot_")
        self.addCleanup(self._cleanup)
        # bc.__file__ drives the output dir; point it inside our temp dir.
        self.fake = mock.Mock()
        self.fake.__file__ = os.path.join(self.tmp, "bobert_companion.py")
        # _act_screenshot checks bc.screenshot_privacy_block_reason() first;
        # default to "not blocked" so a bare Mock doesn't trip the refusal.
        self.fake.screenshot_privacy_block_reason.return_value = None

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_saves_png_from_take_screenshot(self):
        self.fake.take_screenshot.return_value = b"\x89PNGfakebytes"
        with _patch_bc(self.fake), \
                mock.patch.object(A.time, "strftime",
                                  return_value="screenshot_20260601_120000.png"):
            out = A._act_screenshot("")
        expected = os.path.join(self.tmp, "screenshots",
                                "screenshot_20260601_120000.png")
        self.assertIn("screenshot saved to", out)
        self.assertTrue(os.path.exists(expected))
        with open(expected, "rb") as f:
            self.assertEqual(f.read(), b"\x89PNGfakebytes")

    def test_save_failure_reports_error(self):
        self.fake.take_screenshot.return_value = b"data"
        # Force open() to raise so the except branch is taken.
        with _patch_bc(self.fake), \
                mock.patch.object(A.time, "strftime",
                                  return_value="s.png"), \
                mock.patch("builtins.open",
                           side_effect=OSError("disk full")):
            out = A._act_screenshot("")
        self.assertIn("save failed", out)
        self.assertIn("disk full", out)

    def test_powershell_fallback_on_win32(self):
        # take_screenshot returns None -> fallback path. Force win32.
        self.fake.take_screenshot.return_value = None
        with _patch_bc(self.fake), \
                mock.patch.object(A.sys, "platform", "win32"), \
                mock.patch.object(A.time, "strftime", return_value="s.png"), \
                mock.patch.object(A.subprocess, "run") as mrun:
            out = A._act_screenshot("")
        mrun.assert_called_once()
        # invoked powershell with a -Command
        argv = mrun.call_args[0][0]
        self.assertEqual(argv[0], "powershell")
        self.assertIn("-Command", argv)
        self.assertIn("screenshot saved to", out)

    def test_unsupported_when_no_png_and_not_win32(self):
        self.fake.take_screenshot.return_value = None
        with _patch_bc(self.fake), \
                mock.patch.object(A.sys, "platform", "linux"), \
                mock.patch.object(A.time, "strftime", return_value="s.png"):
            out = A._act_screenshot("")
        self.assertIn("not supported", out)

    def test_privacy_blocklist_refuses_and_skips_powershell_fallback(self):
        # When a SCREENSHOT_PRIVACY_BLOCKLIST window is focused, _act_screenshot
        # must refuse BEFORE the PowerShell fallback — otherwise that fallback
        # would capture the private screen directly, bypassing the gate that
        # take_screenshot() enforces.
        self.fake.screenshot_privacy_block_reason.return_value = "banking"
        self.fake.SCREENSHOT_PRIVACY_REFUSAL = "REFUSED-PRIVATE"
        with _patch_bc(self.fake), \
                mock.patch.object(A.sys, "platform", "win32"), \
                mock.patch.object(A.subprocess, "run") as mrun:
            out = A._act_screenshot("")
        self.assertEqual(out, "REFUSED-PRIVATE")
        mrun.assert_not_called()                       # PowerShell never ran
        self.fake.take_screenshot.assert_not_called()  # no capture attempted


# ─────────────────────────────────────────────────────────────────────────
# Media keys (delegate to bc._media_key_with_focus)
# ─────────────────────────────────────────────────────────────────────────
class MediaKeyTests(unittest.TestCase):
    def _run(self, handler):
        fake = mock.Mock()
        fake._media_key_with_focus.return_value = "RESULT"
        with _patch_bc(fake):
            out = handler("")
        return out, fake._media_key_with_focus.call_args

    def test_media_next_delegates(self):
        out, call = self._run(A._act_media_next)
        self.assertEqual(out, "RESULT")
        args = call[0]
        self.assertEqual(args[0], "nexttrack")
        self.assertEqual(args[2], "media next pressed")

    def test_media_prev_delegates(self):
        out, call = self._run(A._act_media_prev)
        self.assertEqual(out, "RESULT")
        self.assertEqual(call[0][0], "prevtrack")
        self.assertEqual(call[0][2], "media previous pressed")

    def test_media_playpause_delegates(self):
        out, call = self._run(A._act_media_playpause)
        self.assertEqual(out, "RESULT")
        self.assertEqual(call[0][0], "playpause")
        self.assertEqual(call[0][2], "media play/pause pressed")


# ─────────────────────────────────────────────────────────────────────────
# Volume keys (delegate to bc._get_pyautogui)
# ─────────────────────────────────────────────────────────────────────────
class VolumeTests(unittest.TestCase):
    def _bc_with_pag(self):
        fake = mock.Mock()
        pag = mock.Mock()
        fake._get_pyautogui.return_value = pag
        return fake, pag

    def _bc_no_pag(self):
        fake = mock.Mock()
        fake._get_pyautogui.return_value = None
        return fake

    def test_volume_up_presses_key(self):
        fake, pag = self._bc_with_pag()
        with _patch_bc(fake):
            out = A._act_volume_up("")
        pag.press.assert_called_once_with("volumeup")
        self.assertEqual(out, "volume up")

    def test_volume_down_presses_key(self):
        fake, pag = self._bc_with_pag()
        with _patch_bc(fake):
            out = A._act_volume_down("")
        pag.press.assert_called_once_with("volumedown")
        self.assertEqual(out, "volume down")

    def test_volume_mute_presses_key(self):
        fake, pag = self._bc_with_pag()
        with _patch_bc(fake):
            out = A._act_volume_mute("")
        pag.press.assert_called_once_with("volumemute")
        self.assertEqual(out, "mute toggled")

    def test_set_volume_rejects_unparseable_and_out_of_range(self):
        fake = mock.Mock()
        fake._parse_spoken_number.return_value = None
        with _patch_bc(fake):
            self.assertIn("couldn't parse", A._act_set_volume("loudish"))
            self.assertIn("couldn't parse", A._act_set_volume("150"))
            self.assertIn("couldn't parse", A._act_set_volume(""))

    def test_set_volume_spoken_number_falls_back_to_parser(self):
        # "thirty" → the monolith's _parse_spoken_number → 30; pycaw mocked
        # via sys.modules so no real COM endpoint is touched. Uses the modern
        # AudioDevice.EndpointVolume shape (verified on-box 2026-07-10).
        fake = mock.Mock()
        fake._parse_spoken_number.return_value = 30
        fake_vol = mock.MagicMock()
        fake_dev = mock.Mock()
        fake_dev.EndpointVolume = fake_vol
        fake_pycaw = mock.MagicMock()
        fake_pycaw.AudioUtilities.GetSpeakers.return_value = fake_dev
        with _patch_bc(fake), \
                mock.patch.dict(sys.modules, {
                    "pycaw": mock.MagicMock(),
                    "pycaw.pycaw": fake_pycaw,
                }):
            out = A._act_set_volume("thirty")
        self.assertEqual(out, "volume set to 30 percent, sir")
        fake_vol.SetMasterVolumeLevelScalar.assert_called_once_with(0.3, None)

    def test_set_volume_digits_direct(self):
        fake_vol = mock.MagicMock()
        fake_dev = mock.Mock()
        fake_dev.EndpointVolume = fake_vol
        fake_pycaw = mock.MagicMock()
        fake_pycaw.AudioUtilities.GetSpeakers.return_value = fake_dev
        with _patch_bc(mock.Mock()), \
                mock.patch.dict(sys.modules, {
                    "pycaw": mock.MagicMock(),
                    "pycaw.pycaw": fake_pycaw,
                }):
            out = A._act_set_volume("30%")
        self.assertEqual(out, "volume set to 30 percent, sir")
        fake_vol.SetMasterVolumeLevelScalar.assert_called_once_with(0.3, None)

    def test_volume_up_unavailable(self):
        with _patch_bc(self._bc_no_pag()):
            self.assertEqual(A._act_volume_up(""), "pyautogui unavailable")

    def test_volume_down_unavailable(self):
        with _patch_bc(self._bc_no_pag()):
            self.assertEqual(A._act_volume_down(""), "pyautogui unavailable")

    def test_volume_mute_unavailable(self):
        with _patch_bc(self._bc_no_pag()):
            self.assertEqual(A._act_volume_mute(""), "pyautogui unavailable")


# ─────────────────────────────────────────────────────────────────────────
# Streaming auto-play one-liners (delegate to bc._streaming_auto_play)
# ─────────────────────────────────────────────────────────────────────────
class StreamingTests(unittest.TestCase):
    CASES = [
        (lambda: A._act_netflix("q"), "netflix"),
        (lambda: A._act_prime_video("q"), "prime_video"),
        (lambda: A._act_disney_plus("q"), "disney_plus"),
        (lambda: A._act_hulu("q"), "hulu"),
        (lambda: A._act_max("q"), "max"),
        (lambda: A._act_spotify("q"), "spotify"),
        (lambda: A._act_youtube_play("q"), "youtube"),
    ]

    def test_each_service_delegates_with_service_and_query(self):
        for call, service in self.CASES:
            with self.subTest(service=service):
                fake = mock.Mock()
                fake._streaming_auto_play.return_value = f"playing on {service}"
                with _patch_bc(fake):
                    out = call()
                fake._streaming_auto_play.assert_called_once_with(service, "q")
                self.assertEqual(out, f"playing on {service}")


# ─────────────────────────────────────────────────────────────────────────
# HUD visibility toggles
# ─────────────────────────────────────────────────────────────────────────
class HideHudTests(unittest.TestCase):
    def test_hide_writes_invisible_state(self):
        fake = mock.Mock()
        with _patch_bc(fake):
            out = A._act_hide_hud("")
        fake._write_hud_state.assert_called_once_with(visible=False)
        self.assertIn("HUD hidden", out)


class SetUnifiedHudHiddenTests(unittest.TestCase):
    """``_set_unified_hud_hidden`` writes the ✕-button 'hidden' flag into
    <repo>/unified_hud_state.json via a mkstemp+replace. We redirect the
    control-file location by patching the os/json/tempfile modules the
    function imports locally (it does ``import os as _os`` etc.)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jv_uhud_")
        self.ctrl = os.path.join(self.tmp, "unified_hud_state.json")
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_with_ctrl_path(self, hidden):
        """Run _set_unified_hud_hidden with os.path.join forced to return our
        temp control path (regardless of __file__ dirname math)."""
        import os as real_os
        orig_join = real_os.path.join

        def fake_join(*parts):
            # The function builds <root>/unified_hud_state.json; intercept that.
            if parts and parts[-1] == "unified_hud_state.json":
                return self.ctrl
            return orig_join(*parts)

        with mock.patch("os.path.join", side_effect=fake_join):
            A._set_unified_hud_hidden(hidden)

    def test_creates_control_file_with_hidden_flag(self):
        self._run_with_ctrl_path(True)
        self.assertTrue(os.path.exists(self.ctrl))
        import json
        with open(self.ctrl, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["hidden"], True)

    def test_merges_into_existing_control_file(self):
        import json
        with open(self.ctrl, "w", encoding="utf-8") as f:
            json.dump({"visible": True, "other": 7}, f)
        self._run_with_ctrl_path(False)
        with open(self.ctrl, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["hidden"], False)
        # pre-existing keys are preserved
        self.assertEqual(data["other"], 7)

    def test_corrupt_existing_file_is_tolerated(self):
        with open(self.ctrl, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        # Should not raise; falls back to empty dict then writes hidden flag.
        self._run_with_ctrl_path(True)
        import json
        with open(self.ctrl, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["hidden"], True)

    def test_swallows_all_exceptions(self):
        # Outer try/except swallows everything: force mkstemp to blow up.
        with mock.patch("tempfile.mkstemp", side_effect=OSError("nope")):
            # must not raise
            A._set_unified_hud_hidden(True)

    def test_replace_failure_cleans_temp_and_is_swallowed(self):
        # Drives the inner except (os.replace fails -> remove temp -> re-raise,
        # caught by the outer try). Must not raise; control file not committed.
        seen = {}
        orig_mkstemp = tempfile.mkstemp

        def spy_mkstemp(*a, **k):
            fd, path = orig_mkstemp(*a, **k)
            seen["tmp"] = path
            return fd, path

        with mock.patch("tempfile.mkstemp", side_effect=spy_mkstemp), \
                mock.patch("os.replace", side_effect=OSError("replace boom")):
            self._run_with_ctrl_path(True)
        # temp file was cleaned up by the inner except branch
        self.assertFalse(os.path.exists(seen["tmp"]))
        # nothing committed to the real control path
        self.assertFalse(os.path.exists(self.ctrl))

    def test_replace_and_cleanup_both_fail_still_swallowed(self):
        # replace fails AND the temp-removal in the inner except ALSO fails;
        # the innermost ``except Exception: pass`` swallows it, then the
        # re-raise bubbles to the outer try and is swallowed too. No raise.
        with mock.patch("os.replace", side_effect=OSError("replace boom")), \
                mock.patch("os.remove", side_effect=OSError("remove boom")):
            # must not raise despite both failures
            self._run_with_ctrl_path(True)
        self.assertFalse(os.path.exists(self.ctrl))


class ShowHudTests(unittest.TestCase):
    def test_show_writes_visible_and_clears_x_hide(self):
        fake = mock.Mock()
        with _patch_bc(fake), \
                mock.patch.object(A, "_set_unified_hud_hidden") as mclear:
            out = A._act_show_hud("")
        fake._write_hud_state.assert_called_once_with(visible=True)
        mclear.assert_called_once_with(False)
        self.assertIn("HUD restored", out)


class ToggleHudTests(unittest.TestCase):
    def _fake_bc(self, currently_visible):
        fake = mock.Mock()
        # _hud_state_lock is used as a context manager.
        fake._hud_state_lock = mock.MagicMock()
        fake._hud_state_cache = {"visible": currently_visible}
        return fake

    def test_toggle_from_visible_hides(self):
        fake = self._fake_bc(True)
        with _patch_bc(fake), \
                mock.patch.object(A, "_set_unified_hud_hidden") as mclear:
            out = A._act_toggle_hud("")
        fake._write_hud_state.assert_called_once_with(visible=False)
        # Hiding must NOT clear the ✕-button latch.
        mclear.assert_not_called()
        self.assertIn("HUD hidden", out)

    def test_toggle_from_hidden_shows(self):
        fake = self._fake_bc(False)
        with _patch_bc(fake), \
                mock.patch.object(A, "_set_unified_hud_hidden") as mclear:
            out = A._act_toggle_hud("")
        fake._write_hud_state.assert_called_once_with(visible=True)
        # Toggling back to visible must clear the ✕-button latch too, else the
        # persisted 'hidden' flag keeps the window down (P1-hud reopen bug).
        mclear.assert_called_once_with(False)
        self.assertIn("HUD restored", out)

    def test_toggle_defaults_visible_when_cache_read_raises(self):
        # If reading the cache raises, code defaults currently_visible=True,
        # so it hides.
        fake = mock.Mock()
        fake._hud_state_lock = mock.MagicMock()
        # .get on a Mock (not a dict) -> make the lock context raise instead.
        fake._hud_state_lock.__enter__ = mock.Mock(
            side_effect=RuntimeError("lock boom"))
        with _patch_bc(fake):
            out = A._act_toggle_hud("")
        fake._write_hud_state.assert_called_once_with(visible=False)
        self.assertIn("HUD hidden", out)

    def test_toggle_missing_visible_key_defaults_true(self):
        # cache has no "visible" key -> .get(...,True) => True => hide
        fake = mock.Mock()
        fake._hud_state_lock = mock.MagicMock()
        fake._hud_state_cache = {}
        with _patch_bc(fake):
            out = A._act_toggle_hud("")
        fake._write_hud_state.assert_called_once_with(visible=False)
        self.assertIn("HUD hidden", out)


# ─────────────────────────────────────────────────────────────────────────
# Self-diagnostic probes (delegate to bc._probe_via_selfdiag)
# ─────────────────────────────────────────────────────────────────────────
class SelfDiagProbeTests(unittest.TestCase):
    CASES = [
        (lambda: A._act_test_mic(""), "mic", "_probe_microphone"),
        (lambda: A._act_test_tts(""), "tts", "_probe_tts"),
        (lambda: A._act_test_vision(""), "vision", "_probe_webcam"),
    ]

    def test_each_probe_delegates(self):
        for call, label, probe in self.CASES:
            with self.subTest(label=label):
                fake = mock.Mock()
                fake._probe_via_selfdiag.return_value = f"{label} ok"
                with _patch_bc(fake):
                    out = call()
                fake._probe_via_selfdiag.assert_called_once_with(label, probe)
                self.assertEqual(out, f"{label} ok")


# ─────────────────────────────────────────────────────────────────────────
# clear_tasks
# ─────────────────────────────────────────────────────────────────────────
class ClearTasksTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jv_tasks_")
        self.todo = os.path.join(self.tmp, "jarvis_todo.md")
        self.addCleanup(self._cleanup)
        self.fake = mock.Mock()
        self.fake.TODO_FILE = self.todo

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_task_file(self):
        # TODO_FILE does not exist
        with _patch_bc(self.fake):
            out = A._act_clear_tasks("")
        self.assertEqual(out, "no task file to clear")

    def test_backs_up_then_removes(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("- [ ] buy milk\n")
        with _patch_bc(self.fake):
            out = A._act_clear_tasks("")
        # original gone
        self.assertFalse(os.path.exists(self.todo))
        # backup created under backups/
        backup_dir = os.path.join(self.tmp, "backups")
        backups = os.listdir(backup_dir)
        self.assertEqual(len(backups), 1)
        self.assertTrue(backups[0].startswith("jarvis_todo_"))
        self.assertIn("task queue cleared", out)
        self.assertIn("backups/", out)
        # backup content preserved
        with open(os.path.join(backup_dir, backups[0]), encoding="utf-8") as f:
            self.assertEqual(f.read(), "- [ ] buy milk\n")

    def test_backup_collision_appends_suffix(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("data\n")
        # Pre-create the backups dir AND the exact-mtime-named backup so the
        # collision branch (append _1) is exercised. Use a frozen strftime.
        backup_dir = os.path.join(self.tmp, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        # Pre-create BOTH the base and the _1 snapshot so the while-loop body
        # (n += 1) runs at least once, landing the new backup at _2.
        for name in ("jarvis_todo_FROZEN.md", "jarvis_todo_FROZEN_1.md"):
            with open(os.path.join(backup_dir, name), "w") as f:
                f.write("old")
        with _patch_bc(self.fake), \
                mock.patch.object(A.time, "strftime", return_value="FROZEN"):
            out = A._act_clear_tasks("")
        names = sorted(os.listdir(backup_dir))
        self.assertIn("jarvis_todo_FROZEN.md", names)
        self.assertIn("jarvis_todo_FROZEN_1.md", names)
        self.assertIn("jarvis_todo_FROZEN_2.md", names)
        self.assertIn("jarvis_todo_FROZEN_2.md", out)

    def test_backup_failure_refuses_to_wipe(self):
        with open(self.todo, "w", encoding="utf-8") as f:
            f.write("keep me\n")
        with _patch_bc(self.fake), \
                mock.patch.object(A.shutil, "copy2",
                                  side_effect=OSError("copy boom")):
            out = A._act_clear_tasks("")
        self.assertIn("backup failed, refused to clear", out)
        self.assertIn("copy boom", out)
        # file must NOT be removed when backup fails
        self.assertTrue(os.path.exists(self.todo))


# ─────────────────────────────────────────────────────────────────────────
# session_resume
# ─────────────────────────────────────────────────────────────────────────
class SessionResumeTests(unittest.TestCase):
    def test_returns_resume_text_when_available(self):
        fake = mock.Mock()
        fake._build_session_resume.return_value = ("Picking up the report.",
                                                   {"k": "v"})
        with _patch_bc(fake):
            out = A._act_session_resume("")
        fake._build_session_resume.assert_called_once_with(force=True)
        self.assertEqual(out, "Picking up the report.")

    def test_falls_back_when_no_text(self):
        fake = mock.Mock()
        fake._build_session_resume.return_value = ("", None)
        with _patch_bc(fake):
            out = A._act_session_resume("")
        self.assertIn("no clear recollection", out)


# ─────────────────────────────────────────────────────────────────────────
# restart
# ─────────────────────────────────────────────────────────────────────────
class RestartTests(unittest.TestCase):
    def test_spawns_thread_and_returns_message(self):
        fake = mock.Mock()
        fake.__file__ = os.path.join(tempfile.gettempdir(),
                                     "bobert_companion.py")
        # Patch threading.Thread (imported inside the function from the real
        # threading module) so no background thread runs.
        import threading
        with _patch_bc(fake), \
                mock.patch.object(threading, "Thread") as mthread:
            out = A._act_restart("")
        mthread.assert_called_once()
        # daemon thread requested + started
        _, kwargs = mthread.call_args
        self.assertTrue(kwargs.get("daemon"))
        mthread.return_value.start.assert_called_once()
        self.assertIn("Restarting now", out)

    def test_inner_restart_relaunches_and_exits(self):
        """Drive the inner _do_restart closure by capturing the target and
        invoking it with subprocess/time.sleep/Timer mocked. 2026-07-12: the
        closure now (a) arms a failsafe Timer, (b) releases the singleton
        BEFORE Popen (the replacement must not singleton-suicide against a
        lingering old process), and (c) exits via bc._hard_exit(0,
        clean=False) — the un-deadlockable TerminateProcess path — instead
        of a raw os._exit that can hang in ExitProcess forever."""
        fake = mock.Mock()
        script_path = os.path.join(tempfile.gettempdir(), "bc_restart.py")
        fake.__file__ = script_path
        fake._hard_exit.side_effect = SystemExit
        import threading
        captured = {}
        order = []
        fake._release_singleton.side_effect = \
            lambda *a, **k: order.append("singleton")

        def capture_thread(target=None, daemon=None):
            captured["target"] = target
            return mock.Mock()

        with _patch_bc(fake), \
                mock.patch.object(threading, "Thread",
                                  side_effect=capture_thread):
            A._act_restart("")

        # Now run the captured target with all side effects mocked.
        with _no_sleep(), \
                mock.patch("threading.Timer",
                           side_effect=lambda *a, **k: (order.append("timer"),
                                                        mock.Mock())[1]) as mtimer, \
                mock.patch.object(
                    A, "_release_native_resources",
                    side_effect=lambda *a, **k: order.append("release")), \
                mock.patch.object(
                    A.subprocess, "Popen",
                    side_effect=lambda *a, **k: order.append("popen")
                ) as mpopen:
            with self.assertRaises(SystemExit):
                captured["target"]()
        mpopen.assert_called_once()
        # LOAD-BEARING ORDER (2026-07-14): singleton freed → successor SPAWNED
        # → failsafe armed → natives released. Releasing the natives first
        # blocked in torch.cuda.synchronize() on the in-flight farewell TTS,
        # the 20s failsafe fired, and the Popen never ran: JARVIS vanished.
        self.assertEqual(order, ["singleton", "popen", "timer", "release"])
        # failsafe armed AFTER the spawn, pointing at the un-deadlockable helper
        mtimer.assert_called_once()
        self.assertIs(mtimer.call_args[0][1], A._hard_exit_via_bc)
        # launched with the python executable + script path
        argv = mpopen.call_args[0][0]
        self.assertEqual(argv[0], A.sys.executable)
        self.assertEqual(argv[1], os.path.abspath(script_path))
        # exited via the helper; a restart is NOT a clean stop (watchdog
        # must resurrect if the relaunch also failed)
        fake._hard_exit.assert_called_once_with(0, clean=False)

    def test_inner_restart_handles_popen_failure(self):
        """If Popen raises, the closure prints and still calls os._exit."""
        fake = mock.Mock()
        fake.__file__ = os.path.join(tempfile.gettempdir(), "bc.py")
        import threading
        captured = {}

        def capture_thread(target=None, daemon=None):
            captured["target"] = target
            return mock.Mock()

        with _patch_bc(fake), \
                mock.patch.object(threading, "Thread",
                                  side_effect=capture_thread):
            A._act_restart("")

        fake._hard_exit.side_effect = SystemExit
        with _no_sleep(), \
                mock.patch("threading.Timer"), \
                mock.patch.object(A.subprocess, "Popen",
                                  side_effect=OSError("spawn fail")), \
                mock.patch("builtins.print") as mprint:
            with self.assertRaises(SystemExit):
                captured["target"]()
        # error was logged, and the closure still hard-exited
        self.assertTrue(
            any("relaunch failed" in str(c) for c in mprint.call_args_list))
        fake._hard_exit.assert_called_once_with(0, clean=False)


if __name__ == "__main__":
    unittest.main()
