"""Unit tests for bobert_companion.py monolith — SECTION 4 (lines ~6882-8881).

Covers the vision / screenshot helpers, the pyautogui UI-automation wrappers,
the streaming-service auto-play pipeline, iTunes COM helpers, and the
session-resume file readers.

Everything external is mocked: no real screenshots, no pyautogui, no Chrome,
no anthropic / requests network, no iTunes COM, no filesystem outside temp +
mock_open. Tests run only in the LOCAL full tier (heavy deps present) and skip
on the light-deps CI runner via ``@requires_monolith``.

ISOLATION CONTRACT: the monolith is imported ONCE via the cached harness. We
NEVER re-import or swap sys.modules for the monolith itself. Per-test patches
use ``mock.patch.object(self.bc, ...)`` which auto-restore. The few directly
mutated globals (``_apple_music_last_seen``) are saved in setUp and restored in
tearDown.
"""
from __future__ import annotations

import io
import itertools
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

from tests._monolith_harness import (
    MonolithGlobalsTestCase, load_monolith, requires_monolith)


# A fake anthropic module whose error classes are real Exception subclasses so
# the monolith's ``except anthropic.APIStatusError`` clauses are well-formed.
def _make_fake_anthropic():
    m = mock.MagicMock(name="anthropic")
    m.APIStatusError = type("APIStatusError", (Exception,), {})
    m.APIConnectionError = type("APIConnectionError", (Exception,), {})
    m.APITimeoutError = type("APITimeoutError", (Exception,), {})
    return m


# The monolith's _ui_safe identifies a pyautogui failsafe trip by the EXACT
# class name "FailSafeException" (type(e).__name__), so our stand-in must carry
# that exact __name__ — not the underscore-prefixed module-local name.
class FailSafeException(Exception):
    """Stand-in for pyautogui.FailSafeException — matched by class *name* in
    the monolith's _ui_safe, so the actual type doesn't matter (only __name__)."""


class _FakePag:
    """Minimal pyautogui stand-in recording the calls made against it."""

    FAILSAFE = True
    PAUSE = 0.15
    FAILSAFE_POINTS = [(0, 0)]

    def __init__(self, pos=(500, 500), size=(1920, 1080)):
        self._pos = pos
        self._size = size
        self.moved: list = []
        self.clicks: list = []
        self.double_clicks: list = []
        self.written: list = []
        self.pressed: list = []
        self.hotkeys: list = []
        self.scrolled: list = []

    def position(self):
        return self._pos

    def size(self):
        return self._size

    def moveTo(self, x, y, duration=0):
        self.moved.append((x, y))
        self._pos = (x, y)

    def click(self, x, y, button="left"):
        self.clicks.append((x, y, button))

    def doubleClick(self, x, y):
        self.double_clicks.append((x, y))

    def write(self, text, interval=0.0):
        self.written.append(text)

    def press(self, key):
        self.pressed.append(key)

    def hotkey(self, *keys):
        self.hotkeys.append(tuple(keys))

    def scroll(self, amount):
        self.scrolled.append(amount)


@requires_monolith
class VisionAnswerParseTests(MonolithGlobalsTestCase):
    """_vision_answer_is_yes — the strict YES/NO gate for verify-playback."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_plain_yes(self):
        self.assertTrue(self.bc._vision_answer_is_yes("YES it is playing"))

    def test_yes_with_trailing_punct(self):
        self.assertTrue(self.bc._vision_answer_is_yes("yes. music is on"))
        self.assertTrue(self.bc._vision_answer_is_yes("Yes, definitely"))

    def test_no(self):
        self.assertFalse(self.bc._vision_answer_is_yes("NO not playing"))

    def test_empty_and_whitespace(self):
        self.assertFalse(self.bc._vision_answer_is_yes(""))
        self.assertFalse(self.bc._vision_answer_is_yes("   "))

    def test_vision_failure_stub_counts_as_no(self):
        self.assertFalse(self.bc._vision_answer_is_yes("(vision failed: HTTP 500)"))

    def test_yes_not_first_word_is_no(self):
        # "Maybe yes" — only a leading YES counts.
        self.assertFalse(self.bc._vision_answer_is_yes("Maybe yes"))

    def test_local_vision_tag_is_stripped(self):
        # 2026-07-13: ask_vision prefixes local answers with
        # '[local-vision] ' — the tag became the "first word" and EVERY
        # local YES parsed as NO (three straight ✗ while the video
        # demonstrably played). Bracketed lead tags must be transparent.
        self.assertTrue(self.bc._vision_answer_is_yes("[local-vision] YES"))
        self.assertTrue(self.bc._vision_answer_is_yes(
            "[local-vision] Yes — one large player is visible."))
        self.assertFalse(self.bc._vision_answer_is_yes("[local-vision] NO"))
        self.assertFalse(self.bc._vision_answer_is_yes("[local-vision]"))


@requires_monolith
class NormalizeServiceTests(MonolithGlobalsTestCase):
    """_normalize_service — alias folding for streaming service keys."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_canonical_passthrough(self):
        self.assertEqual(self.bc._normalize_service("Netflix"), "netflix")
        self.assertEqual(self.bc._normalize_service("spotify"), "spotify")

    def test_aliases(self):
        self.assertEqual(self.bc._normalize_service("PRIME"), "prime_video")
        self.assertEqual(self.bc._normalize_service("amazon prime"), "prime_video")
        self.assertEqual(self.bc._normalize_service("HBO Max"), "max")
        self.assertEqual(self.bc._normalize_service("apple music"), "apple_music")
        self.assertEqual(self.bc._normalize_service("yt"), "youtube")

    def test_plus_sign_normalized(self):
        self.assertEqual(self.bc._normalize_service("disney+"), "disney_plus")

    def test_unknown_returns_normalized_key(self):
        # Unknown services still get whitespace/dash → underscore folding.
        self.assertEqual(self.bc._normalize_service("Foo Bar"), "foo_bar")

    def test_aliases_resolve_to_real_services(self):
        # Every alias target must be a real service key (catches typo'd aliases).
        for target in self.bc._STREAMING_ALIASES.values():
            self.assertIn(target, self.bc._STREAMING_SERVICES)


@requires_monolith
class PlaylistRequestTests(MonolithGlobalsTestCase):
    """_looks_like_playlist_request — prefix/suffix playlist intent detection."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_prefix_form(self):
        self.assertEqual(
            self.bc._looks_like_playlist_request("playlist chill vibes"),
            (True, "chill vibes"),
        )

    def test_prefix_with_colon_and_called(self):
        self.assertEqual(
            self.bc._looks_like_playlist_request("play the playlist: deep focus"),
            (True, "deep focus"),
        )
        self.assertEqual(
            self.bc._looks_like_playlist_request("playlist called morning jams"),
            (True, "morning jams"),
        )

    def test_suffix_form(self):
        self.assertEqual(
            self.bc._looks_like_playlist_request("my workout playlist"),
            (True, "workout"),
        )
        self.assertEqual(
            self.bc._looks_like_playlist_request("chill playlist"),
            (True, "chill"),
        )

    def test_generic_suffix_name_rejected(self):
        # "music playlist" — generic name, not a real named playlist.
        self.assertEqual(
            self.bc._looks_like_playlist_request("music playlist"),
            (False, "music playlist"),
        )

    def test_not_a_playlist(self):
        self.assertEqual(
            self.bc._looks_like_playlist_request("bohemian rhapsody"),
            (False, "bohemian rhapsody"),
        )

    def test_empty(self):
        self.assertEqual(self.bc._looks_like_playlist_request(""), (False, ""))


@requires_monolith
class SummariseTaskLineTests(MonolithGlobalsTestCase):
    """_summarise_task_line — strip the `- [ ] **date** [tag] —` prefix."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_full_prefix_stripped_and_first_sentence(self):
        line = "- [ ] **2026-05-30 14:22** [feature] — Add dark mode to the tray. Then test."
        self.assertEqual(self.bc._summarise_task_line(line), "Add dark mode to the tray")

    def test_plain_line(self):
        self.assertEqual(
            self.bc._summarise_task_line("- [ ] just a plain task line"),
            "just a plain task line",
        )

    def test_long_line_truncated_with_ellipsis(self):
        line = "- [ ] **2026-05-30 14:22** — " + ("x" * 120)
        out = self.bc._summarise_task_line(line)
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(len(out), 90)

    def test_filepath_not_split_early(self):
        # period followed by NON-space (tray.py) must not truncate.
        line = "- [ ] **2026-05-30 14:22** [bug] — fix tray.py crash on boot"
        self.assertEqual(self.bc._summarise_task_line(line), "fix tray.py crash on boot")


@requires_monolith
class YoutubeRegexTests(MonolithGlobalsTestCase):
    """_YT_VIDEO_RE — extracts an 11-char video id from SERP HTML variants."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _vid(self, text):
        m = self.bc._YT_VIDEO_RE.search(text)
        if not m:
            return None
        return m.group(1) or m.group(2) or m.group(3)

    def test_full_watch_url(self):
        self.assertEqual(
            self._vid("x https://www.youtube.com/watch?v=dQw4w9WgXcQ y"),
            "dQw4w9WgXcQ",
        )

    def test_short_youtu_be(self):
        self.assertEqual(self._vid("http://youtu.be/abcdefghijk"), "abcdefghijk")

    def test_percent_encoded_wrapper(self):
        text = "/url?q=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D12345678901"
        self.assertEqual(self._vid(text), "12345678901")

    def test_no_match(self):
        self.assertIsNone(self._vid("nothing to see here"))


@requires_monolith
class ActionRegexTests(MonolithGlobalsTestCase):
    """_ACTION_RE — parses [ACTION: name, arg] tokens (digits allowed in name)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_action_with_arg(self):
        m = self.bc._ACTION_RE.search("blah [ACTION: open_url, https://example.com] blah")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "open_url")
        self.assertEqual(m.group(2), "https://example.com")

    def test_action_no_arg(self):
        m = self.bc._ACTION_RE.search("[ACTION: screenshot]")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "screenshot")
        self.assertIsNone(m.group(2))

    def test_action_name_with_digits(self):
        # Regression: names like switch_to_gpt4 must not truncate at the digit.
        m = self.bc._ACTION_RE.search("[ACTION: switch_to_gpt4]")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "switch_to_gpt4")


@requires_monolith
class ResolveKnownAppTests(MonolithGlobalsTestCase):
    """_resolve_known_app / _find_chrome — path resolution with mocked exists."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_alias_resolves_to_existing_path(self):
        target = self.bc._KNOWN_APP_PATHS["bambu studio"][0]
        with mock.patch.object(self.bc.os.path, "exists", side_effect=lambda p: p == target):
            self.assertEqual(self.bc._resolve_known_app("Bamboo Studio"), target)
            # Whitespace/case tolerance.
            self.assertEqual(self.bc._resolve_known_app("  BAMBU   studio "), target)

    def test_unknown_app_returns_none(self):
        with mock.patch.object(self.bc.os.path, "exists", return_value=False):
            self.assertIsNone(self.bc._resolve_known_app("totally fake app"))

    def test_known_alias_but_no_file_returns_none(self):
        with mock.patch.object(self.bc.os.path, "exists", return_value=False):
            self.assertIsNone(self.bc._resolve_known_app("bambu"))

    def test_find_chrome_first_existing_wins(self):
        first = self.bc._CHROME_PATHS[0]
        with mock.patch.object(self.bc.os.path, "exists", side_effect=lambda p: p == first):
            self.assertEqual(self.bc._find_chrome(), first)

    def test_find_chrome_none(self):
        with mock.patch.object(self.bc.os.path, "exists", return_value=False):
            self.assertIsNone(self.bc._find_chrome())


@requires_monolith
class OpenUrlNewWindowTests(MonolithGlobalsTestCase):
    """_open_url_new_window — spawns a separate Chrome window via subprocess."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_no_chrome_returns_false(self):
        with mock.patch.object(self.bc, "_find_chrome", return_value=None):
            self.assertFalse(self.bc._open_url_new_window("example.com"))

    def test_spawns_new_window_and_prefixes_scheme(self):
        with mock.patch.object(self.bc, "_find_chrome", return_value=r"C:\chrome.exe"), \
             mock.patch.object(self.bc.subprocess, "Popen") as popen:
            self.assertTrue(self.bc._open_url_new_window("example.com"))
            args = popen.call_args[0][0]
            self.assertIn("--new-window", args)
            # bare host got https:// prepended.
            self.assertIn("https://example.com", args)

    def test_popen_failure_returns_false(self):
        with mock.patch.object(self.bc, "_find_chrome", return_value=r"C:\chrome.exe"), \
             mock.patch.object(self.bc.subprocess, "Popen", side_effect=OSError("nope")):
            self.assertFalse(self.bc._open_url_new_window("https://example.com"))


@requires_monolith
class NativeCaptureSizeTests(MonolithGlobalsTestCase):
    """_native_capture_size — native px of the captured region."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_known_monitor_uses_monitor_dims(self):
        # MONITORS entries are (x, y, w, h); 'middle' resolves without mss.
        name = "middle"
        w, h = self.bc.MONITORS[name][2], self.bc.MONITORS[name][3]
        self.assertEqual(self.bc._native_capture_size(name), (int(w), int(h)))

    def test_unknown_monitor_falls_through_to_mss(self):
        mssmod = mock.MagicMock()
        sct = mock.MagicMock()
        sct.monitors = [{"left": 0, "top": 0, "width": 3000, "height": 2000}]
        mssmod.mss.return_value.__enter__ = lambda s: sct
        mssmod.mss.return_value.__exit__ = lambda *a: False
        del mssmod.MSS  # force getattr(mss,'MSS',mss.mss) → mss.mss
        with mock.patch.dict(sys.modules, {"mss": mssmod}):
            self.assertEqual(self.bc._native_capture_size(None), (3000, 2000))

    def test_mss_failure_returns_zeroes(self):
        with mock.patch.dict(sys.modules, {"mss": None}):
            # monitor=None and import mss → None is not a module → AttributeError → (0,0)
            self.assertEqual(self.bc._native_capture_size("does_not_exist"), (0, 0))


@requires_monolith
class TakeScreenshotTests(MonolithGlobalsTestCase):
    """take_screenshot — mss-preferred capture with downscale, fully mocked."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _fake_pil(self, size=(800, 600)):
        img = mock.MagicMock()
        img.size = size
        img.resize.return_value = img

        def _save(buf, format=None, optimize=None):
            buf.write(b"PNGBYTES")

        img.save.side_effect = _save
        pil = mock.MagicMock()
        pil.Image.frombytes.return_value = img
        pil.Image.LANCZOS = 1
        return pil, img

    def test_missing_pillow_returns_none(self):
        with mock.patch.dict(sys.modules, {"PIL": None}):
            self.assertIsNone(self.bc.take_screenshot())

    def test_mss_path_returns_png_bytes(self):
        pil, _img = self._fake_pil()
        mssmod = mock.MagicMock()
        sct = mock.MagicMock()
        raw = mock.MagicMock()
        raw.size = (800, 600)
        raw.bgra = b""
        sct.grab.return_value = raw
        sct.monitors = [{"left": 0, "top": 0, "width": 800, "height": 600}]
        mssmod.mss.return_value.__enter__ = lambda s: sct
        mssmod.mss.return_value.__exit__ = lambda *a: False
        del mssmod.MSS
        with mock.patch.dict(sys.modules, {"mss": mssmod, "PIL": pil}):
            self.assertEqual(self.bc.take_screenshot(), b"PNGBYTES")

    def test_downscale_when_too_large(self):
        pil, img = self._fake_pil(size=(4000, 3000))
        mssmod = mock.MagicMock()
        sct = mock.MagicMock()
        raw = mock.MagicMock()
        raw.size = (4000, 3000)
        raw.bgra = b""
        sct.grab.return_value = raw
        sct.monitors = [{"left": 0, "top": 0, "width": 4000, "height": 3000}]
        mssmod.mss.return_value.__enter__ = lambda s: sct
        mssmod.mss.return_value.__exit__ = lambda *a: False
        del mssmod.MSS
        with mock.patch.dict(sys.modules, {"mss": mssmod, "PIL": pil}):
            out = self.bc.take_screenshot(max_dim=1568)
            self.assertEqual(out, b"PNGBYTES")
            img.resize.assert_called_once()  # downscale happened

    def test_privacy_blocklist_hard_blocks_capture(self):
        # Focused window title matches SCREENSHOT_PRIVACY_BLOCKLIST -> hard
        # gate returns None WITHOUT ever invoking mss/PIL, so even a caller
        # that forgot the high-level check can't leak a private screen.
        from core import config as cfg
        mssmod = mock.MagicMock()
        with mock.patch.object(cfg, "SCREENSHOT_PRIVACY_BLOCKLIST",
                               ["1password", "banking"]), \
                mock.patch.object(self.bc, "_read_focused_window",
                                  return_value=(1, "Chase Banking — Home", None)), \
                mock.patch.dict(sys.modules, {"mss": mssmod}):
            self.assertIsNone(self.bc.take_screenshot())
        mssmod.mss.assert_not_called()       # never reached the capture backend

    def test_privacy_blocklist_empty_is_noop(self):
        # Empty blocklist (the default) must NOT change behaviour: a private-
        # looking title still captures normally.
        from core import config as cfg
        pil, _img = self._fake_pil()
        mssmod = mock.MagicMock()
        sct = mock.MagicMock()
        raw = mock.MagicMock()
        raw.size = (800, 600)
        raw.bgra = b""
        sct.grab.return_value = raw
        sct.monitors = [{"left": 0, "top": 0, "width": 800, "height": 600}]
        mssmod.mss.return_value.__enter__ = lambda s: sct
        mssmod.mss.return_value.__exit__ = lambda *a: False
        del mssmod.MSS
        with mock.patch.object(cfg, "SCREENSHOT_PRIVACY_BLOCKLIST", []), \
                mock.patch.object(self.bc, "_read_focused_window",
                                  return_value=(1, "1Password", None)), \
                mock.patch.dict(sys.modules, {"mss": mssmod, "PIL": pil}):
            self.assertEqual(self.bc.take_screenshot(), b"PNGBYTES")


@requires_monolith
class AskVisionTests(MonolithGlobalsTestCase):
    """ask_vision — disabled gate, local fallback, and catch-all fallback."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_disabled_returns_stub(self):
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", False):
            self.assertIn("disabled", self.bc.ask_vision("what is this?"))

    def test_no_screen_capture(self):
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "take_screenshot", return_value=None):
            self.assertIn("could not capture", self.bc.ask_vision("q"))

    def test_non_claude_backend_uses_local_vlm(self):
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value="a cat"):
            self.assertEqual(self.bc.ask_vision("what?", b"PNG"), "[local-vision] a cat")

    def test_non_claude_backend_no_local_returns_stub(self):
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value=None):
            self.assertIn("requires Claude backend", self.bc.ask_vision("q", b"PNG"))

    def test_claude_success(self):
        anth = _make_fake_anthropic()
        block = mock.MagicMock()
        block.text = "  it is a login screen  "
        anth.Anthropic.return_value.messages.create.return_value.content = [block]
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            self.assertEqual(self.bc.ask_vision("q", b"PNG"), "it is a login screen")

    def test_claude_status_error_falls_back_to_local(self):
        anth = _make_fake_anthropic()
        err = anth.APIStatusError("rate limited")
        err.status_code = 429
        anth.Anthropic.side_effect = err
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value="local answer"), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            self.assertEqual(self.bc.ask_vision("q", b"PNG"), "[local-vision] local answer")

    def test_claude_generic_error_catchall_falls_back(self):
        anth = _make_fake_anthropic()
        anth.Anthropic.side_effect = RuntimeError("explode")
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value="fb"), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            self.assertEqual(self.bc.ask_vision("q", b"PNG"), "[local-vision] fb")

    def test_claude_error_and_no_local_returns_failure_text(self):
        anth = _make_fake_anthropic()
        err = anth.APIStatusError("server")
        err.status_code = 500
        anth.Anthropic.side_effect = err
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value=None), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            self.assertIn("HTTP 500", self.bc.ask_vision("q", b"PNG"))


@requires_monolith
class AskVisionMultiTests(MonolithGlobalsTestCase):
    """ask_vision_multi — labelled multi-monitor vision call."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_disabled(self):
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", False):
            self.assertIn("disabled", self.bc.ask_vision_multi("q", {"left": b"x"}))

    def test_no_images(self):
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True):
            self.assertIn("no screens", self.bc.ask_vision_multi("q", {}))

    def test_non_claude_local_fallback(self):
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value="multi ans"):
            out = self.bc.ask_vision_multi("q", {"left": b"a", "right": b"b"})
            self.assertEqual(out, "[local-vision] multi ans")

    def test_claude_success_builds_interleaved_content(self):
        anth = _make_fake_anthropic()
        block = mock.MagicMock()
        block.text = "answer"
        anth.Anthropic.return_value.messages.create.return_value.content = [block]
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            out = self.bc.ask_vision_multi("q", {"left": b"a", "right": b"b"})
            self.assertEqual(out, "answer")
            sent = anth.Anthropic.return_value.messages.create.call_args
            content = sent.kwargs["messages"][0]["content"]
            # intro text + (label+image)*2 + question = 6 blocks
            self.assertEqual(len(content), 6)


@requires_monolith
class TakeAllMonitorScreenshotsTests(MonolithGlobalsTestCase):
    """take_all_monitor_screenshots — gathers per-monitor PNGs, skipping fails."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_collects_successes_skips_none(self):
        names = list(self.bc.MONITORS.keys())

        def fake_shot(monitor=None, max_dim=1024):
            # Fail the first monitor, succeed the rest.
            return None if monitor == names[0] else b"PNG-" + monitor.encode()

        with mock.patch.object(self.bc, "take_screenshot", side_effect=fake_shot):
            out = self.bc.take_all_monitor_screenshots()
        self.assertNotIn(names[0], out)
        for n in names[1:]:
            self.assertEqual(out[n], b"PNG-" + n.encode())


@requires_monolith
class QueryVisionForCoordsTests(MonolithGlobalsTestCase):
    """_query_vision_for_coords — strict coordinate parsing from vision reply."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_clean_coords(self):
        with mock.patch.object(self.bc, "ask_vision", return_value="432,718"):
            self.assertEqual(
                self.bc._query_vision_for_coords("btn", b"x", 1000, 1000), (432, 718)
            )

    def test_parenthesised_coords(self):
        with mock.patch.object(self.bc, "ask_vision", return_value="(12, 34)."):
            self.assertEqual(
                self.bc._query_vision_for_coords("btn", b"x", 1000, 1000), (12, 34)
            )

    def test_not_found(self):
        with mock.patch.object(self.bc, "ask_vision", return_value="NOT_FOUND"):
            self.assertIsNone(self.bc._query_vision_for_coords("btn", b"x", 1000, 1000))

    def test_long_prose_refused(self):
        # A chatty reply must NOT be mined for a coordinate (safety).
        reply = "I can see 2 buttons, 3 tabs, and several more controls here ok"
        with mock.patch.object(self.bc, "ask_vision", return_value=reply):
            self.assertIsNone(self.bc._query_vision_for_coords("btn", b"x", 1000, 1000))

    def test_short_tail_coordinate_accepted(self):
        with mock.patch.object(self.bc, "ask_vision", return_value="the point is 50,60"):
            self.assertEqual(
                self.bc._query_vision_for_coords("btn", b"x", 1000, 1000), (50, 60)
            )

    def test_out_of_bounds_rejected(self):
        with mock.patch.object(self.bc, "ask_vision", return_value="5000,6000"):
            self.assertIsNone(self.bc._query_vision_for_coords("btn", b"x", 100, 100))


@requires_monolith
class FindClickTargetTests(MonolithGlobalsTestCase):
    """find_click_target — two-pass vision locate, with PIL + vision mocked."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _fake_pil_image(self, size):
        img = mock.MagicMock()
        img.size = size
        return img

    def test_no_pillow_returns_none(self):
        with mock.patch.dict(sys.modules, {"PIL": None}):
            self.assertIsNone(self.bc.find_click_target("the button"))

    def test_pass1_capture_fail_returns_none(self):
        pil = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"PIL": pil}), \
             mock.patch.object(self.bc, "take_screenshot", return_value=None):
            self.assertIsNone(self.bc.find_click_target("the button"))

    def test_pass1_not_found_returns_none(self):
        pil = mock.MagicMock()
        pil.Image.open.return_value = self._fake_pil_image((1568, 900))
        with mock.patch.dict(sys.modules, {"PIL": pil}), \
             mock.patch.object(self.bc, "take_screenshot", return_value=b"png"), \
             mock.patch.object(self.bc, "_query_vision_for_coords", return_value=None):
            self.assertIsNone(self.bc.find_click_target("the button"))

    # A deterministic fake monitor layout used by every translate test below so
    # the assertions DO NOT depend on this dev box's real displays (which made
    # the prior tests leak host state and fail on any other geometry / on CI).
    # It mirrors the OWNER's reported NEGATIVE-ORIGIN four-monitor rig: virtual
    # desktop 7680x2880 with origin (-2560,-1440) — a monitor above-left of the
    # primary. All dims are LOGICAL (what pyautogui clicks in).
    FAKE_MONITORS = {
        "left":   (-2560, 0,     2560, 1440),
        "middle": (0,     0,     2560, 1440),
        "right":  (2560,  0,     2560, 1440),
        "top":    (0,     -1440, 2560, 1440),
    }

    def _fake_mss(self, left, top, width, height):
        """An mss stand-in whose monitors[0] (virtual screen) reports the given
        NATIVE rect — used so _captured_region / _native_capture_size are
        deterministic instead of querying the real machine."""
        mssmod = mock.MagicMock()
        sct = mock.MagicMock()
        sct.monitors = [{"left": left, "top": top,
                         "width": width, "height": height}]
        mssmod.mss.return_value.__enter__ = lambda s: sct
        mssmod.mss.return_value.__exit__ = lambda *a: False
        del mssmod.MSS  # force the getattr(mss,"MSS",mss.mss) path to mss.mss
        return mssmod

    def test_known_monitor_translates_to_absolute(self):
        # BASELINE (single-monitor / 100% DPI, no downscale). The named monitor's
        # LOGICAL size, the Pass-1 image size, AND the captured NATIVE size are
        # ALL equal (1568x900), so every scale factor is exactly 1.0 and the
        # Pass-1 coords (100,100) translate straight to (origin+100, origin+100).
        # Pass-2 capture fails → native size comes from _native_capture_size.
        # Geometry is injected so this is host-independent (was: read the real
        # box's MONITORS, which leaked display state).
        mons = {"main": (0, 0, 1568, 900)}
        mx, my = mons["main"][0], mons["main"][1]
        pil = mock.MagicMock()
        pil.Image.open.return_value = self._fake_pil_image((1568, 900))

        shots = [b"png1", None]   # pass1 ok, full(=pass2) capture fails

        with mock.patch.object(self.bc, "MONITORS", mons), \
             mock.patch.dict(sys.modules, {"PIL": pil}), \
             mock.patch.object(self.bc, "take_screenshot",
                               side_effect=lambda monitor=None, max_dim=1568: shots.pop(0)), \
             mock.patch.object(self.bc, "_query_vision_for_coords", return_value=(100, 100)), \
             mock.patch.object(self.bc, "_native_capture_size", return_value=(1568, 900)):
            result = self.bc.find_click_target("the button", monitor="main")
        # image==native==logical → scale 1.0 → (100,100) offset by origin.
        self.assertEqual(result, (mx + 100, my + 100))

    def test_negative_origin_monitor_offsets_by_negative_top(self):
        # NEGATIVE-ORIGIN named monitor (the owner's "top" display at y=-1440).
        # No downscale, 100% DPI (image==native==logical=2560x1440). A Pass-1
        # point (1280,720) — the monitor centre — must land at the monitor's
        # absolute centre (0+1280, -1440+720) = (1280, -720). Before the fix a
        # naive translate that ignored the NEGATIVE top would have clicked at
        # +720 on the primary instead.
        mons = {"top": (0, -1440, 2560, 1440)}
        pil = mock.MagicMock()
        pil.Image.open.return_value = self._fake_pil_image((2560, 1440))
        shots = [b"png1", None]
        with mock.patch.object(self.bc, "MONITORS", mons), \
             mock.patch.dict(sys.modules, {"PIL": pil}), \
             mock.patch.object(self.bc, "take_screenshot",
                               side_effect=lambda monitor=None, max_dim=1568: shots.pop(0)), \
             mock.patch.object(self.bc, "_query_vision_for_coords", return_value=(1280, 720)), \
             mock.patch.object(self.bc, "_native_capture_size", return_value=(2560, 1440)):
            result = self.bc.find_click_target("the button", monitor="top")
        self.assertEqual(result, (1280, -720))

    def test_downscaled_pass1_scales_up_to_logical(self):
        # DOWNSCALED Pass-1 image, 100% DPI, Pass-2 fails. The monitor is
        # 3136x1800 LOGICAL == NATIVE, but Pass-1 was downscaled to 1568x900
        # (half size). A Pass-1 coord (100,100) therefore represents logical
        # (200,200), NOT (100,100): it must be scaled UP by native/pass1 = 2.0.
        # This is the off-by-2x miss on any monitor wider than 1568px.
        mons = {"main": (0, 0, 3136, 1800)}
        pil = mock.MagicMock()
        pil.Image.open.return_value = self._fake_pil_image((1568, 900))
        shots = [b"png1", None]
        with mock.patch.object(self.bc, "MONITORS", mons), \
             mock.patch.dict(sys.modules, {"PIL": pil}), \
             mock.patch.object(self.bc, "take_screenshot",
                               side_effect=lambda monitor=None, max_dim=1568: shots.pop(0)), \
             mock.patch.object(self.bc, "_query_vision_for_coords", return_value=(100, 100)), \
             mock.patch.object(self.bc, "_native_capture_size", return_value=(3136, 1800)):
            result = self.bc.find_click_target("the button", monitor="main")
        # cx_full=int(100*3136/1568)=200; native==logical → *1.0; +origin(0,0).
        self.assertEqual(result, (200, 200))

    def test_known_monitor_dpi_scaled_translates_native_to_logical(self):
        # DPI-SCALED named monitor. The monitor is 2560x1440 LOGICAL but renders
        # at 200% so the un-downscaled Pass-2 capture is 5120x2880 NATIVE. A
        # refined point at native (1000,1000) must scale DOWN to logical
        # (500,500) before the logical origin is added — otherwise the click
        # overshoots far past the target (the overshoot the owner saw).
        name = "middle"
        mx, my = self.FAKE_MONITORS[name][0], self.FAKE_MONITORS[name][1]

        class _Img:
            def __init__(self, size):
                self.size = size

            def crop(self, box):
                l, t, r, b = box
                return _Img((r - l, b - t))

            def save(self, buf, format=None):
                buf.write(b"x")

        pil = mock.MagicMock()
        # Pass-1 image then full-res (native) image.
        imgs = [_Img((1568, 882)), _Img((5120, 2880))]
        pil.Image.open.side_effect = lambda b: imgs.pop(0)
        pil.Image.LANCZOS = 1
        shots = [b"png1", b"fullpng"]
        # Pass-1 returns a point; scaled to full it becomes the crop centre.
        # Pass-2 returns a point inside the crop that resolves to native
        # (1000,1000) absolute-in-image. We compute pass-1 so cx_full lands so
        # the crop's top-left is (750,750) and pass-2 (250,250) → 750+250=1000.
        # Pass-1 (306,306) → *5120/1568=999 ~ within a px; simpler: drive the
        # refined point directly via pass-2 offset math below.
        coords = [(306, 306), (250, 250)]

        with mock.patch.object(self.bc, "MONITORS", self.FAKE_MONITORS), \
             mock.patch.dict(sys.modules, {"PIL": pil}), \
             mock.patch.object(self.bc, "take_screenshot",
                               side_effect=lambda monitor=None, max_dim=1568: shots.pop(0)), \
             mock.patch.object(self.bc, "_query_vision_for_coords",
                               side_effect=lambda *a: coords.pop(0)):
            result = self.bc.find_click_target("the button", monitor=name)
        # cx_full = int(306*5120/1568)=999, crop left=int(999-250)=749,
        # refined = 749 + 250 = 999 native; *logical/native = *2560/5120 = 0.5
        # → 499; + origin. Same for Y (306*2880/882=999, top=749, +250=999,
        # *1440/2880=0.5 → 499).
        self.assertEqual(result, (mx + 499, my + 499))

    def test_two_pass_refine_no_monitor_translates_by_virtual_origin(self):
        # NEGATIVE-ORIGIN virtual desktop, monitor=None, 100% DPI.
        # Pass1 1568x882 → full 7680x2880 (bigger ⇒ pass-2 crop runs). The mss
        # virtual screen and the (injected) MONITORS agree on origin (-2560,
        # -1440) and size 7680x2880 ⇒ native==logical ⇒ scale 1.0. A refined
        # point at native (2560,720) — the centre of the TOP monitor within the
        # grab — must translate to logical (-2560+2560, -1440+720) = (0,-720),
        # i.e. the top monitor's centre. This is the exact case the owner hits:
        # clicking something on the above-left monitor.
        class _CropImg:
            def __init__(self, size):
                self.size = size

            def crop(self, box):
                left, top, right, bot = box
                return _CropImg((right - left, bot - top))

            def save(self, buf, format=None):
                buf.write(b"crop")

        pil = mock.MagicMock()
        imgs = [_CropImg((1568, 882)), _CropImg((7680, 2880))]
        pil.Image.open.side_effect = lambda b: imgs.pop(0)
        pil.Image.LANCZOS = 1
        shots = [b"png1", b"fullpng"]
        # Pass-1 (523,220) → *7680/1568=2561, *2880/882=718 ≈ centre of top mon.
        # crop top-left = (2311, 468); pass-2 (250,252) → refined (2561,720).
        coords = [(523, 220), (250, 252)]

        mssmod = self._fake_mss(-2560, -1440, 7680, 2880)

        with mock.patch.object(self.bc, "MONITORS", self.FAKE_MONITORS), \
             mock.patch.dict(sys.modules, {"PIL": pil, "mss": mssmod}), \
             mock.patch.object(self.bc, "take_screenshot",
                               side_effect=lambda monitor=None, max_dim=1568: shots.pop(0)), \
             mock.patch.object(self.bc, "_query_vision_for_coords",
                               side_effect=lambda *a: coords.pop(0)):
            result = self.bc.find_click_target("the button")
        # cx_full=int(523*7680/1568)=2561; crop left=max(0,2561-250)=2311;
        # refined_x=2311+250=2561; scale 7680/7680=1.0 → vx(-2560)+2561=1.
        # cy_full=int(220*2880/882)=718; crop top=max(0,718-250)=468;
        # refined_y=468+252=720; scale 1.0 → vy(-1440)+720=-720.
        self.assertEqual(result, (1, -720))

    def test_no_monitor_dpi_scaled_virtual_scales_native_to_logical(self):
        # monitor=None on a uniformly-200% virtual desktop: logical 7680x2880 @
        # (-2560,-1440) but the live mss capture is 15360x5760 NATIVE. The
        # Pass-2 full image IS that native size, so a refined native point must
        # be scaled by logical/native (0.5) before adding the LOGICAL origin.
        class _CropImg:
            def __init__(self, size):
                self.size = size

            def crop(self, box):
                left, top, right, bot = box
                return _CropImg((right - left, bot - top))

            def save(self, buf, format=None):
                buf.write(b"crop")

        pil = mock.MagicMock()
        imgs = [_CropImg((1568, 588)), _CropImg((15360, 5760))]
        pil.Image.open.side_effect = lambda b: imgs.pop(0)
        pil.Image.LANCZOS = 1
        shots = [b"png1", b"fullpng"]
        # Pass-1 (784,294) → *15360/1568=7680, *5760/588=2880 → native centre.
        # crop left=7430, top=2630; pass-2 (250,250) → refined (7680,2880).
        coords = [(784, 294), (250, 250)]

        mssmod = self._fake_mss(-2560, -1440, 15360, 5760)

        with mock.patch.object(self.bc, "MONITORS", self.FAKE_MONITORS), \
             mock.patch.dict(sys.modules, {"PIL": pil, "mss": mssmod}), \
             mock.patch.object(self.bc, "take_screenshot",
                               side_effect=lambda monitor=None, max_dim=1568: shots.pop(0)), \
             mock.patch.object(self.bc, "_query_vision_for_coords",
                               side_effect=lambda *a: coords.pop(0)):
            result = self.bc.find_click_target("the button")
        # refined native (7680,2880); scale logical/native = 7680/15360 = 0.5,
        # 2880/5760 = 0.5 → (3840,1440); + logical origin (-2560,-1440)
        # → (1280, 0) = dead centre of the logical virtual desktop.
        self.assertEqual(result, (1280, 0))


@requires_monolith
class OpenUrlOffscreenCaptureTests(MonolithGlobalsTestCase):
    """_open_url_offscreen_capture — early-exit paths (win32 GDI happy-path is
    not unit-testable without real window handles; we cover the guards)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_missing_win32_dep(self):
        with mock.patch.dict(sys.modules, {"win32gui": None}):
            png, reason = self.bc._open_url_offscreen_capture("https://example.com")
        self.assertIsNone(png)
        self.assertIn("missing dep", reason)

    def test_chrome_not_found(self):
        w32 = mock.MagicMock()
        with mock.patch.dict(sys.modules,
                             {"win32gui": w32, "win32ui": w32, "win32con": w32}), \
             mock.patch.object(self.bc, "_find_chrome", return_value=None):
            self.assertEqual(
                self.bc._open_url_offscreen_capture("https://example.com"),
                (None, "chrome_not_found"),
            )

    def test_chrome_spawn_failed(self):
        w32 = mock.MagicMock()
        with mock.patch.dict(sys.modules,
                             {"win32gui": w32, "win32ui": w32, "win32con": w32}), \
             mock.patch.object(self.bc, "_find_chrome", return_value=r"C:\chrome.exe"), \
             mock.patch.object(self.bc.subprocess, "Popen", side_effect=OSError("nope")):
            png, reason = self.bc._open_url_offscreen_capture("https://example.com")
        self.assertIsNone(png)
        self.assertIn("chrome_spawn_failed", reason)

    def test_new_window_never_appears(self):
        # EnumWindows finds no NEW chrome window before the 5s deadline; we make
        # the deadline already-elapsed via a monotonic time.time so the poll
        # loop exits immediately.
        w32 = mock.MagicMock()
        w32.GetClassName.return_value = "other"
        ticks = iter([1000.0] + [1100.0] * 50)  # start, then way past deadline
        with mock.patch.dict(sys.modules,
                             {"win32gui": w32, "win32ui": w32, "win32con": w32}), \
             mock.patch.object(self.bc, "_find_chrome", return_value=r"C:\chrome.exe"), \
             mock.patch.object(self.bc.subprocess, "Popen"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc.time, "time", side_effect=lambda: next(ticks)):
            self.assertEqual(
                self.bc._open_url_offscreen_capture("https://example.com"),
                (None, "chrome_window_not_found"),
            )


@requires_monolith
class UISafeAndNudgeTests(MonolithGlobalsTestCase):
    """_ui_safe / _nudge_from_corner — failsafe recovery semantics."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_op_success_passthrough(self):
        pag = _FakePag()
        self.assertEqual(self.bc._ui_safe(pag, lambda: "ok"), "ok")

    def test_non_failsafe_error_reraised(self):
        pag = _FakePag()
        with self.assertRaises(ValueError):
            self.bc._ui_safe(pag, lambda: (_ for _ in ()).throw(ValueError("x")))

    def test_failsafe_retry_after_nudge_then_succeeds(self):
        pag = _FakePag(pos=(0, 0))  # cursor in corner → nudge will fire
        calls = {"n": 0}

        def op():
            calls["n"] += 1
            if calls["n"] == 1:
                raise FailSafeException()
            return "recovered"

        self.assertEqual(self.bc._ui_safe(pag, op), "recovered")
        self.assertEqual(calls["n"], 2)
        self.assertTrue(pag.moved)  # a nudge happened

    def test_failsafe_persists_raises_ui_failsafe_error(self):
        pag = _FakePag(pos=(0, 0))

        def op():
            raise FailSafeException()

        with self.assertRaises(self.bc.UIFailsafeError):
            self.bc._ui_safe(pag, op)

    def test_nudge_at_corner_moves_inward(self):
        pag = _FakePag(pos=(0, 0))
        self.assertTrue(self.bc._nudge_from_corner(pag))
        self.assertEqual(pag.moved[-1], (50, 50))

    def test_nudge_off_corner_noop(self):
        pag = _FakePag(pos=(500, 500))
        self.assertFalse(self.bc._nudge_from_corner(pag))
        self.assertEqual(pag.moved, [])


@requires_monolith
class UIWrapperTests(MonolithGlobalsTestCase):
    """ui_click / ui_type / ui_press / ui_hotkey / ui_scroll / ui_double_click."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_ui_click_noop_without_pyautogui(self):
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=None):
            self.assertIsNone(self.bc.ui_click(1, 2))

    def test_ui_click_clamps_to_virtual_bounds(self):
        pag = _FakePag()
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(self.bc, "_publish_reticle"), \
             mock.patch.object(self.bc, "_virtual_screen_bounds", return_value=(0, 0, 1920, 1080)):
            self.bc.ui_click(99999, -99999)
        # clamped into [0,1919] x [0,1079]
        self.assertEqual(pag.clicks, [(1919, 0, "left")])

    def test_ui_click_right_button_label(self):
        pag = _FakePag()
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(self.bc, "_publish_reticle") as ret, \
             mock.patch.object(self.bc, "_virtual_screen_bounds", return_value=(0, 0, 1920, 1080)):
            self.bc.ui_click(10, 20, button="right")
        self.assertEqual(pag.clicks, [(10, 20, "right")])
        self.assertEqual(ret.call_args[0][2], "right-click")

    def test_ui_double_click(self):
        pag = _FakePag()
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(self.bc, "_publish_reticle"):
            self.bc.ui_double_click(7, 8)
        self.assertEqual(pag.double_clicks, [(7, 8)])

    def test_ui_type(self):
        pag = _FakePag()
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(self.bc, "_publish_reticle"), \
             mock.patch.object(self.bc, "_active_window_center", return_value=(5, 5)):
            self.bc.ui_type("hello world")
        self.assertEqual(pag.written, ["hello world"])

    def test_ui_press(self):
        pag = _FakePag()
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(self.bc, "_publish_reticle"), \
             mock.patch.object(self.bc, "_active_window_center", return_value=None):
            self.bc.ui_press("enter")
        self.assertEqual(pag.pressed, ["enter"])

    def test_ui_hotkey(self):
        pag = _FakePag()
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(self.bc, "_publish_reticle"), \
             mock.patch.object(self.bc, "_active_window_center", return_value=(1, 1)):
            self.bc.ui_hotkey("ctrl", "c")
        self.assertEqual(pag.hotkeys, [("ctrl", "c")])

    def test_ui_scroll(self):
        pag = _FakePag()
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(self.bc, "_publish_reticle"):
            self.bc.ui_scroll(-3)
        self.assertEqual(pag.scrolled, [-3])

    def test_ui_scroll_position_error_still_scrolls(self):
        # If pag.position() raises, the reticle step is swallowed and the
        # scroll still happens.
        pag = _FakePag()
        pag.position = mock.Mock(side_effect=RuntimeError("no cursor"))
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(self.bc, "_publish_reticle"):
            self.bc.ui_scroll(5)
        self.assertEqual(pag.scrolled, [5])

    def test_ui_type_noop_without_pyautogui(self):
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=None):
            self.assertIsNone(self.bc.ui_type("hi"))

    def test_ui_press_noop_without_pyautogui(self):
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=None):
            self.assertIsNone(self.bc.ui_press("enter"))


@requires_monolith
class GetPyautoguiTests(MonolithGlobalsTestCase):
    """_get_pyautogui — lazy import + module-level cache."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = self.bc._pyautogui

    def tearDown(self):
        self.bc._pyautogui = self._saved

    def test_returns_cached_instance_without_reimport(self):
        sentinel = object()
        self.bc._pyautogui = sentinel
        self.assertIs(self.bc._get_pyautogui(), sentinel)

    def test_import_failure_returns_none(self):
        self.bc._pyautogui = None
        with mock.patch.dict(sys.modules, {"pyautogui": None}):
            # `import pyautogui` with None in sys.modules → ImportError path.
            self.assertIsNone(self.bc._get_pyautogui())
        self.assertIsNone(self.bc._pyautogui)


@requires_monolith
class ExtractYoutubeUrlTests(MonolithGlobalsTestCase):
    """_extract_youtube_url_from_search — yt-dlp skill then SERP regex."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_skill_stage_hit(self):
        skill = mock.MagicMock()
        skill.find_direct_url.return_value = "https://www.youtube.com/watch?v=AAAAAAAAAAA"
        with mock.patch.dict(sys.modules, {"skill_youtube_search": skill}):
            self.assertEqual(
                self.bc._extract_youtube_url_from_search("lofi"),
                "https://www.youtube.com/watch?v=AAAAAAAAAAA",
            )

    def test_serp_fallback_extracts_video(self):
        resp = mock.Mock(status_code=200,
                         text="z https://www.youtube.com/watch?v=dQw4w9WgXcQ z")
        with mock.patch.dict(sys.modules, {"skill_youtube_search": None}), \
             mock.patch.object(self.bc.requests, "get", return_value=resp):
            self.assertEqual(
                self.bc._extract_youtube_url_from_search("never gonna"),
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            )

    def test_serp_non_200_returns_none(self):
        resp = mock.Mock(status_code=503, text="")
        with mock.patch.dict(sys.modules, {"skill_youtube_search": None}), \
             mock.patch.object(self.bc.requests, "get", return_value=resp):
            self.assertIsNone(self.bc._extract_youtube_url_from_search("q"))

    def test_serp_request_exception_returns_none(self):
        with mock.patch.dict(sys.modules, {"skill_youtube_search": None}), \
             mock.patch.object(self.bc.requests, "get", side_effect=Exception("net")):
            self.assertIsNone(self.bc._extract_youtube_url_from_search("q"))

    def test_serp_no_video_in_html_returns_none(self):
        resp = mock.Mock(status_code=200, text="<html>no videos</html>")
        with mock.patch.dict(sys.modules, {"skill_youtube_search": None}), \
             mock.patch.object(self.bc.requests, "get", return_value=resp):
            self.assertIsNone(self.bc._extract_youtube_url_from_search("q"))


@requires_monolith
class StreamingFindWithRetryTests(MonolithGlobalsTestCase):
    """_streaming_find_with_retry — retry wrapper around find_click_target."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_first_attempt_hit(self):
        with mock.patch.object(self.bc, "find_click_target", return_value=(1, 2)) as f, \
             mock.patch.object(self.bc.time, "sleep"):
            self.assertEqual(self.bc._streaming_find_with_retry("h"), (1, 2))
            self.assertEqual(f.call_count, 1)

    def test_retries_then_succeeds(self):
        with mock.patch.object(self.bc, "find_click_target", side_effect=[None, None, (3, 4)]) as f, \
             mock.patch.object(self.bc.time, "sleep") as slept:
            self.assertEqual(self.bc._streaming_find_with_retry("h", attempts=3), (3, 4))
            self.assertEqual(f.call_count, 3)
            self.assertEqual(slept.call_count, 2)

    def test_all_fail_returns_none(self):
        with mock.patch.object(self.bc, "find_click_target", return_value=None), \
             mock.patch.object(self.bc.time, "sleep"):
            self.assertIsNone(self.bc._streaming_find_with_retry("h", attempts=2))


@requires_monolith
class StreamingVerifyPlaybackTests(MonolithGlobalsTestCase):
    """_streaming_verify_playback — screenshot + vision YES/NO."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_no_screen(self):
        with mock.patch.object(self.bc, "take_screenshot", return_value=None):
            ok, ans = self.bc._streaming_verify_playback("playing?")
        self.assertFalse(ok)
        self.assertIn("could not capture", ans)

    def test_yes(self):
        with mock.patch.object(self.bc, "take_screenshot", return_value=b"png"), \
             mock.patch.object(self.bc, "ask_vision", return_value="YES it's playing"):
            ok, ans = self.bc._streaming_verify_playback("playing?")
        self.assertTrue(ok)
        self.assertTrue(ans.startswith("YES"))

    def test_no(self):
        with mock.patch.object(self.bc, "take_screenshot", return_value=b"png"), \
             mock.patch.object(self.bc, "ask_vision", return_value="NO still paused"):
            ok, _ = self.bc._streaming_verify_playback("playing?")
        self.assertFalse(ok)


@requires_monolith
class StreamingApplyPlayStrategyTests(MonolithGlobalsTestCase):
    """_streaming_apply_play_strategy — single play-strategy dispatch."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_play_button_found(self):
        with mock.patch.object(self.bc, "_streaming_find_with_retry", return_value=(10, 20)), \
             mock.patch.object(self.bc, "ui_click") as click:
            attempted, desc = self.bc._streaming_apply_play_strategy(
                "play_button", {"play_hint": "h"}, None)
        self.assertTrue(attempted)
        self.assertIn("(10, 20)", desc)
        click.assert_called_once_with(10, 20)

    def test_play_button_not_found(self):
        with mock.patch.object(self.bc, "_streaming_find_with_retry", return_value=None), \
             mock.patch.object(self.bc, "ui_click") as click:
            attempted, desc = self.bc._streaming_apply_play_strategy(
                "play_button", {"play_hint": "h"}, None)
        self.assertFalse(attempted)
        click.assert_not_called()

    def test_double_click_result_with_coords(self):
        with mock.patch.object(self.bc, "ui_double_click") as dc:
            attempted, desc = self.bc._streaming_apply_play_strategy(
                "double_click_result", {}, (5, 6))
        self.assertTrue(attempted)
        dc.assert_called_once_with(5, 6)

    def test_double_click_result_without_coords_is_noop(self):
        with mock.patch.object(self.bc, "ui_double_click") as dc:
            attempted, desc = self.bc._streaming_apply_play_strategy(
                "double_click_result", {}, None)
        self.assertFalse(attempted)
        dc.assert_not_called()

    def test_space_strategy(self):
        with mock.patch.object(self.bc, "ui_press") as press:
            attempted, _ = self.bc._streaming_apply_play_strategy("space", {}, None)
        self.assertTrue(attempted)
        press.assert_called_once_with("space")

    def test_playpause_strategy(self):
        with mock.patch.object(self.bc, "ui_press") as press:
            attempted, _ = self.bc._streaming_apply_play_strategy("playpause", {}, None)
        self.assertTrue(attempted)
        press.assert_called_once_with("playpause")

    def test_unknown_strategy(self):
        attempted, desc = self.bc._streaming_apply_play_strategy("zzz", {}, None)
        self.assertFalse(attempted)
        self.assertIn("unknown play strategy", desc)

    def test_highlighted_row_double_clicks(self):
        with mock.patch.object(self.bc, "_streaming_find_with_retry", return_value=(7, 8)), \
             mock.patch.object(self.bc, "ui_double_click") as dc:
            attempted, desc = self.bc._streaming_apply_play_strategy(
                "highlighted_row", {"track_play_hint": "h"}, None)
        self.assertTrue(attempted)
        dc.assert_called_once_with(7, 8)
        self.assertIn("(7, 8)", desc)

    def test_highlighted_row_not_found_is_noop(self):
        with mock.patch.object(self.bc, "_streaming_find_with_retry", return_value=None), \
             mock.patch.object(self.bc, "ui_double_click") as dc:
            attempted, _ = self.bc._streaming_apply_play_strategy(
                "highlighted_row", {"track_play_hint": "h"}, None)
        self.assertFalse(attempted)
        dc.assert_not_called()

    def test_recheck_is_noop_action_but_attempted(self):
        # recheck does NO UI action but returns attempted=True so the caller
        # waits + re-verifies without a re-click that would restart the track.
        with mock.patch.object(self.bc, "ui_click") as click, \
             mock.patch.object(self.bc, "ui_double_click") as dc, \
             mock.patch.object(self.bc, "ui_press") as press:
            attempted, desc = self.bc._streaming_apply_play_strategy("recheck", {}, None)
        self.assertTrue(attempted)
        click.assert_not_called()
        dc.assert_not_called()
        press.assert_not_called()
        self.assertIn("re-check", desc.lower())


@requires_monolith
class StreamingGoFullscreenTests(MonolithGlobalsTestCase):
    """_streaming_go_fullscreen — best-effort full-screen keypress."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_no_key_is_noop(self):
        with mock.patch.object(self.bc, "ui_press") as press, \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True):
            self.bc._streaming_go_fullscreen({"fullscreen_key": None}, "Svc")
        press.assert_not_called()

    def test_ui_disabled_is_noop(self):
        with mock.patch.object(self.bc, "ui_press") as press, \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", False):
            self.bc._streaming_go_fullscreen({"fullscreen_key": "f"}, "Svc")
        press.assert_not_called()

    def test_single_key_pressed(self):
        with mock.patch.object(self.bc, "ui_press") as press, \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc.time, "sleep"):
            self.bc._streaming_go_fullscreen(
                {"fullscreen_key": "f", "fullscreen_wait": 0}, "Svc")
        press.assert_called_once_with("f")

    def test_hotkey_tuple(self):
        with mock.patch.object(self.bc, "ui_hotkey") as hk, \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc.time, "sleep"):
            self.bc._streaming_go_fullscreen(
                {"fullscreen_key": ("ctrl", "f"), "fullscreen_wait": 0}, "Svc")
        hk.assert_called_once_with("ctrl", "f")

    def test_master_switch_off_is_noop(self):
        # STREAMING_AUTO_FULLSCREEN off ⇒ no key sent even with a valid key +
        # UI automation on. The master switch is read LIVE from core.config.
        import core.config as _cfg
        with mock.patch.object(self.bc, "ui_press") as press, \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(_cfg, "STREAMING_AUTO_FULLSCREEN", False), \
             mock.patch.object(self.bc.time, "sleep"):
            self.bc._streaming_go_fullscreen(
                {"fullscreen_key": "f", "fullscreen_wait": 0}, "Svc")
        press.assert_not_called()

    def test_master_switch_on_sends_key(self):
        import core.config as _cfg
        with mock.patch.object(self.bc, "ui_press") as press, \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(_cfg, "STREAMING_AUTO_FULLSCREEN", True), \
             mock.patch.object(self.bc.time, "sleep"):
            self.bc._streaming_go_fullscreen(
                {"fullscreen_key": "f", "fullscreen_wait": 0}, "Svc")
        press.assert_called_once_with("f")

    def test_focuses_recorded_hwnd_before_pressing(self):
        # When the service has a recorded media-window handle, the fullscreen
        # step focuses EXACTLY that window before the keypress so 'f' lands on
        # the player, never on a random foreground window.
        import core.config as _cfg
        cfg = {"fullscreen_key": "f", "fullscreen_wait": 0,
               "service_key": "netflix"}
        with mock.patch.object(self.bc, "ui_press") as press, \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(_cfg, "STREAMING_AUTO_FULLSCREEN", True), \
             mock.patch.object(self.bc, "_focus_window_hwnd",
                               return_value=True) as focus, \
             mock.patch.dict(self.bc._JARVIS_MEDIA_WINDOW_HWND,
                             {"netflix": 4242}, clear=True), \
             mock.patch.object(self.bc.time, "sleep"):
            self.bc._streaming_go_fullscreen(cfg, "Netflix")
        focus.assert_called_once_with(4242)
        press.assert_called_once_with("f")

    def test_no_recorded_hwnd_still_presses_on_foreground(self):
        # No recorded handle ⇒ don't try to focus; fall back to pressing on the
        # current foreground (the just-activated player).
        import core.config as _cfg
        cfg = {"fullscreen_key": "f", "fullscreen_wait": 0,
               "service_key": "netflix"}
        with mock.patch.object(self.bc, "ui_press") as press, \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(_cfg, "STREAMING_AUTO_FULLSCREEN", True), \
             mock.patch.object(self.bc, "_focus_window_hwnd") as focus, \
             mock.patch.dict(self.bc._JARVIS_MEDIA_WINDOW_HWND, {}, clear=True), \
             mock.patch.object(self.bc.time, "sleep"):
            self.bc._streaming_go_fullscreen(cfg, "Netflix")
        focus.assert_not_called()
        press.assert_called_once_with("f")


@requires_monolith
class EnsureWindowVisibleMaximizedTests(MonolithGlobalsTestCase):
    """_ensure_window_visible_maximized — pull a media window fully on-screen and
    maximize it ("windowed full screen"). All Win32 is mocked."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _win32(self, rect, is_window=True, work=(0, 0, 2560, 1440)):
        """Build a mock win32gui + win32con + win32api trio. `rect` is the
        (left, top, right, bot) GetWindowRect returns; `work` is the monitor
        work area GetMonitorInfo returns."""
        w32 = mock.MagicMock(name="win32gui")
        w32.IsWindow.return_value = is_window
        w32.GetWindowRect.return_value = rect
        con = mock.MagicMock(name="win32con")
        con.SW_RESTORE = 9
        con.SW_MAXIMIZE = 3
        api = mock.MagicMock(name="win32api")
        api.MonitorFromWindow.return_value = 111
        api.GetMonitorInfo.return_value = {"Work": work}
        return w32, con, api

    def test_zero_hwnd_is_noop(self):
        # 0 handle ⇒ False, and Win32 is never touched.
        w32, con, api = self._win32((0, 0, 100, 100))
        with mock.patch.dict(sys.modules,
                             {"win32gui": w32, "win32con": con, "win32api": api}):
            self.assertFalse(self.bc._ensure_window_visible_maximized(0))
        w32.SetWindowPos.assert_not_called()
        w32.ShowWindow.assert_not_called()

    def test_bad_hwnd_is_noop(self):
        # A non-int / None handle ⇒ False without raising.
        self.assertFalse(self.bc._ensure_window_visible_maximized(None))
        self.assertFalse(self.bc._ensure_window_visible_maximized("nope"))

    def test_dead_handle_is_noop(self):
        # IsWindow False ⇒ no move/maximize (recycled handle guard).
        w32, con, api = self._win32((0, 0, 100, 100), is_window=False)
        with mock.patch.dict(sys.modules,
                             {"win32gui": w32, "win32con": con, "win32api": api}):
            self.assertFalse(self.bc._ensure_window_visible_maximized(555))
        w32.SetWindowPos.assert_not_called()

    def test_off_top_edge_is_clamped_then_maximized(self):
        # THE reported bug: title bar above the monitor top (top = -80, work
        # area starts at y=0). Must SetWindowPos to bring it back onto the work
        # area, then SW_MAXIMIZE.
        w32, con, api = self._win32((100, -80, 1300, 720),
                                    work=(0, 0, 2560, 1440))
        with mock.patch.dict(sys.modules,
                             {"win32gui": w32, "win32con": con, "win32api": api}):
            self.assertTrue(self.bc._ensure_window_visible_maximized(4242))
        # Moved fully back on-screen: new top must be >= work-area top (0).
        w32.SetWindowPos.assert_called_once()
        pos_args = w32.SetWindowPos.call_args.args
        # SetWindowPos(hwnd, insertAfter, x, y, cx, cy, flags)
        new_top = pos_args[3]
        self.assertGreaterEqual(new_top, 0)
        # And it was maximized (SW_MAXIMIZE == 3) as the final step.
        show_modes = [c.args[1] for c in w32.ShowWindow.call_args_list]
        self.assertIn(con.SW_MAXIMIZE, show_modes)
        self.assertEqual(show_modes[-1], con.SW_MAXIMIZE)

    def test_on_screen_window_only_maximized_not_moved(self):
        # A window already fully inside the work area ⇒ no SetWindowPos, just
        # restore + maximize.
        w32, con, api = self._win32((100, 100, 1300, 900),
                                    work=(0, 0, 2560, 1440))
        with mock.patch.dict(sys.modules,
                             {"win32gui": w32, "win32con": con, "win32api": api}):
            self.assertTrue(self.bc._ensure_window_visible_maximized(4242))
        w32.SetWindowPos.assert_not_called()
        show_modes = [c.args[1] for c in w32.ShowWindow.call_args_list]
        self.assertEqual(show_modes[-1], con.SW_MAXIMIZE)

    def test_negative_origin_monitor_top_edge(self):
        # Rig's real 'top' monitor sits at y=-1440..0. A window whose title bar
        # is above THAT monitor's top (top=-1500, work area top=-1440) is off-
        # screen and must be clamped down to >= -1440.
        w32, con, api = self._win32((10, -1500, 1210, -780),
                                    work=(0, -1440, 2560, 0))
        with mock.patch.dict(sys.modules,
                             {"win32gui": w32, "win32con": con, "win32api": api}):
            self.assertTrue(self.bc._ensure_window_visible_maximized(4242))
        pos_args = w32.SetWindowPos.call_args.args
        self.assertGreaterEqual(pos_args[3], -1440)  # new top >= work-area top

    def test_no_pywin32_uses_pgw_fallback(self):
        # win32gui import fails ⇒ route through the pygetwindow fallback.
        with mock.patch.dict(sys.modules, {"win32gui": None}), \
             mock.patch.object(self.bc, "_ensure_window_visible_maximized_pgw",
                               return_value=True) as pgw:
            self.assertTrue(self.bc._ensure_window_visible_maximized(4242))
        pgw.assert_called_once_with(4242)

    def test_getrect_failure_falls_back_to_bare_maximize(self):
        # GetWindowRect raising ⇒ still attempt a bare SW_MAXIMIZE (better than
        # nothing) and report success.
        w32, con, api = self._win32((0, 0, 0, 0))
        w32.GetWindowRect.side_effect = RuntimeError("no rect")
        with mock.patch.dict(sys.modules,
                             {"win32gui": w32, "win32con": con, "win32api": api}):
            self.assertTrue(self.bc._ensure_window_visible_maximized(4242))
        show_modes = [c.args[1] for c in w32.ShowWindow.call_args_list]
        self.assertIn(con.SW_MAXIMIZE, show_modes)

    def test_never_raises_on_setwindowpos_error(self):
        # A SetWindowPos blow-up must be swallowed (never abort the media flow).
        w32, con, api = self._win32((100, -80, 1300, 720))
        w32.SetWindowPos.side_effect = RuntimeError("win32 boom")
        with mock.patch.dict(sys.modules,
                             {"win32gui": w32, "win32con": con, "win32api": api}):
            self.assertFalse(self.bc._ensure_window_visible_maximized(4242))


@requires_monolith
class FocusWindowHwndTests(MonolithGlobalsTestCase):
    """_focus_window_hwnd — bring a specific hwnd to the foreground."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_zero_hwnd_is_false(self):
        self.assertFalse(self.bc._focus_window_hwnd(0))

    def test_bad_hwnd_is_false(self):
        self.assertFalse(self.bc._focus_window_hwnd(None))
        self.assertFalse(self.bc._focus_window_hwnd("x"))

    def test_activates_matching_window(self):
        win = mock.MagicMock(_hWnd=4242)
        gw = mock.MagicMock()
        gw.getAllWindows.return_value = [mock.MagicMock(_hWnd=1), win]
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertTrue(self.bc._focus_window_hwnd(4242))
        win.activate.assert_called_once()

    def test_handle_not_found_is_false(self):
        gw = mock.MagicMock()
        gw.getAllWindows.return_value = [mock.MagicMock(_hWnd=1)]
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertFalse(self.bc._focus_window_hwnd(4242))

    def test_benign_win32_success_error_treated_as_ok(self):
        # activate() sometimes raises the "operation completed successfully"
        # pseudo-error — treated as success (mirrors _focus_music_window).
        win = mock.MagicMock(_hWnd=4242)
        win.activate.side_effect = Exception("Error code from Windows: 0")
        gw = mock.MagicMock()
        gw.getAllWindows.return_value = [win]
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertTrue(self.bc._focus_window_hwnd(4242))


@requires_monolith
class StreamingKeyboardSelectTests(MonolithGlobalsTestCase):
    """_streaming_keyboard_select_first_result — Tab*n + Enter navigation."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_disabled_returns_false(self):
        with mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", False):
            self.assertFalse(
                self.bc._streaming_keyboard_select_first_result({}, "Svc"))

    def test_sends_tabs_then_enter(self):
        cfg = {"keyboard_pre_wait": 0, "keyboard_tab_count": 3,
               "keyboard_tab_interval": 0, "keyboard_post_wait": 0}
        with mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "ui_press") as press, \
             mock.patch.object(self.bc.time, "sleep"):
            self.assertTrue(
                self.bc._streaming_keyboard_select_first_result(cfg, "Svc"))
        keys = [c.args[0] for c in press.call_args_list]
        self.assertEqual(keys, ["tab", "tab", "tab", "enter"])

    def test_failsafe_propagates(self):
        cfg = {"keyboard_pre_wait": 0, "keyboard_tab_count": 1,
               "keyboard_tab_interval": 0, "keyboard_post_wait": 0}
        with mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "ui_press",
                               side_effect=self.bc.UIFailsafeError("corner")), \
             mock.patch.object(self.bc.time, "sleep"):
            with self.assertRaises(self.bc.UIFailsafeError):
                self.bc._streaming_keyboard_select_first_result(cfg, "Svc")

    def test_other_exception_returns_false(self):
        cfg = {"keyboard_pre_wait": 0, "keyboard_tab_count": 1,
               "keyboard_tab_interval": 0, "keyboard_post_wait": 0}
        with mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "ui_press", side_effect=RuntimeError("x")), \
             mock.patch.object(self.bc.time, "sleep"):
            self.assertFalse(
                self.bc._streaming_keyboard_select_first_result(cfg, "Svc"))


@requires_monolith
class StreamingPlayAndVerifyTests(MonolithGlobalsTestCase):
    """_streaming_play_and_verify — the verify-and-retry play loop."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_verify_first_already_playing_skips_play(self):
        cfg = {"verify_first": True, "verify_question": "playing?",
               "fullscreen_key": None}
        with mock.patch.object(self.bc, "_streaming_verify_playback",
                               return_value=(True, "YES playing")), \
             mock.patch.object(self.bc, "_streaming_apply_play_strategy") as strat, \
             mock.patch.object(self.bc, "_streaming_go_fullscreen") as fs, \
             mock.patch.object(self.bc.time, "sleep"):
            out = self.bc._streaming_play_and_verify(cfg, "Svc", "song")
        self.assertIn("playing 'song' on Svc", out)
        strat.assert_not_called()
        fs.assert_called_once()

    def test_play_then_verify_confirms(self):
        cfg = {"play_strategies": ["play_button"], "verify_attempts": 2,
               "verify_wait": 0, "fullscreen_key": None}
        with mock.patch.object(self.bc, "_streaming_apply_play_strategy",
                               return_value=(True, "clicked play")), \
             mock.patch.object(self.bc, "_streaming_verify_playback",
                               return_value=(True, "YES")), \
             mock.patch.object(self.bc, "_streaming_go_fullscreen"), \
             mock.patch.object(self.bc.time, "sleep"):
            out = self.bc._streaming_play_and_verify(cfg, "Svc", "song")
        self.assertEqual(out, "playing 'song' on Svc")

    def test_all_attempts_fail_returns_could_not_confirm(self):
        cfg = {"play_strategies": ["play_button"], "verify_attempts": 2,
               "verify_wait": 0, "fullscreen_key": None}
        with mock.patch.object(self.bc, "_streaming_apply_play_strategy",
                               return_value=(True, "clicked play")), \
             mock.patch.object(self.bc, "_streaming_verify_playback",
                               return_value=(False, "NO paused")), \
             mock.patch.object(self.bc, "_streaming_go_fullscreen"), \
             mock.patch.object(self.bc.time, "sleep"):
            out = self.bc._streaming_play_and_verify(cfg, "Svc", "song")
        self.assertIn("couldn't confirm", out)

    def test_failsafe_during_play_aborts(self):
        cfg = {"play_strategies": ["play_button"], "verify_attempts": 1,
               "verify_wait": 0}
        with mock.patch.object(self.bc, "_streaming_apply_play_strategy",
                               side_effect=self.bc.UIFailsafeError("corner")), \
             mock.patch.object(self.bc.time, "sleep"):
            out = self.bc._streaming_play_and_verify(cfg, "Svc", "song")
        self.assertIn("aborted", out)

    def test_skipped_strategy_advances_without_verifying(self):
        # First strategy is a no-op (attempted=False) ⇒ loop should advance to
        # the next attempt without calling verify; second attempt confirms.
        cfg = {"play_strategies": ["double_click_result", "play_button"],
               "verify_attempts": 2, "verify_wait": 0, "fullscreen_key": None}
        apply_results = [(False, "no remembered result coords"),
                         (True, "clicked play")]
        with mock.patch.object(self.bc, "_streaming_apply_play_strategy",
                               side_effect=apply_results), \
             mock.patch.object(self.bc, "_streaming_verify_playback",
                               return_value=(True, "YES")) as verify, \
             mock.patch.object(self.bc, "_streaming_go_fullscreen"), \
             mock.patch.object(self.bc.time, "sleep"):
            out = self.bc._streaming_play_and_verify(cfg, "Svc", "song")
        self.assertEqual(out, "playing 'song' on Svc")
        # verify ran exactly once (only for the attempted strategy).
        self.assertEqual(verify.call_count, 1)


@requires_monolith
class StreamingAutoPlayTests(MonolithGlobalsTestCase):
    """_streaming_auto_play — top-level open/select/play orchestration."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_unknown_service(self):
        out = self.bc._streaming_auto_play("nope_service", "x")
        self.assertIn("unknown streaming service", out)

    def test_empty_query_opens_home(self):
        # Empty query opens the homepage via the force-a-real-browser helper.
        # 2026-07-06 audit fix: with NO recorded media-window handle it must
        # pass close_matching=None (legacy title-substring close could destroy
        # the user's own browser window) and close_hwnd=None.
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome") as opn, \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "_find_browser_window_matching",
                               return_value=None), \
             mock.patch.dict(self.bc._JARVIS_MEDIA_WINDOW_HWND, {}, clear=True):
            out = self.bc._streaming_auto_play("netflix", "   ")
        self.assertEqual(out, "opened Netflix")
        opn.assert_called_once_with(
            self.bc._STREAMING_SERVICES["netflix"]["home"],
            close_matching=None, close_hwnd=None)

    def test_empty_query_reuses_recorded_handle(self):
        # With a recorded handle, the homepage open closes ONLY that window
        # (safe hwnd mode) — and records the NEW homepage window's handle so
        # the next request can also engage safe mode.
        new_win = mock.MagicMock(_hWnd=777)
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome") as opn, \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "_find_browser_window_matching",
                               return_value=new_win), \
             mock.patch.dict(self.bc._JARVIS_MEDIA_WINDOW_HWND,
                             {"netflix": 555}, clear=True):
            out = self.bc._streaming_auto_play("netflix", "")
            self.assertEqual(self.bc._JARVIS_MEDIA_WINDOW_HWND["netflix"], 777)
        self.assertEqual(out, "opened Netflix")
        opn.assert_called_once_with(
            self.bc._STREAMING_SERVICES["netflix"]["home"],
            close_matching=["netflix"], close_hwnd=555)

    def test_all_vision_services_have_tab_match(self):
        # 2026-07-06 audit: spotify/prime_video/disney_plus/hulu/max had no
        # tab_match → no window reuse AND no vision-monitor pinning, so
        # find_click_target photographed the whole virtual screen and missed
        # the play controls. Every vision-select service must define one.
        for key, cfg in self.bc._STREAMING_SERVICES.items():
            if cfg.get("select_method", "vision") == "vision":
                self.assertTrue(cfg.get("tab_match"),
                                msg=f"{key} is a vision service with no tab_match")

    def test_capabilities_missing_returns_open_only(self):
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", False):
            out = self.bc._streaming_auto_play("netflix", "the matrix")
        self.assertIn("auto-click needs", out)

    def test_youtube_no_play_hint_path(self):
        # YouTube has play_hint=None → after clicking the result it just
        # full-screens and reports playing.
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "find_click_target", return_value=(100, 200)), \
             mock.patch.object(self.bc, "ui_click"), \
             mock.patch.object(self.bc, "_streaming_go_fullscreen") as fs:
            out = self.bc._streaming_auto_play("youtube", "lofi beats")
        self.assertEqual(out, "playing 'lofi beats' on YouTube")
        fs.assert_called_once()

    def test_result_not_seen_returns_hint(self):
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "find_click_target", return_value=None):
            out = self.bc._streaming_auto_play("netflix", "the matrix")
        self.assertIn("couldn't see the", out)

    def test_default_play_click_path(self):
        # Netflix: vision select + separate play button, no verify.
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "find_click_target", return_value=(50, 60)), \
             mock.patch.object(self.bc, "ui_click"), \
             mock.patch.object(self.bc, "_streaming_go_fullscreen"):
            out = self.bc._streaming_auto_play("netflix", "the matrix")
        self.assertEqual(out, "playing 'the matrix' on Netflix")

    def test_new_media_window_is_placed_on_screen(self):
        # ISSUE A: after JARVIS opens the media window (recorded hwnd), it must
        # call _ensure_window_visible_maximized on EXACTLY that handle so the
        # window can't land with its title bar off the top of the monitor.
        new_win = mock.MagicMock(_hWnd=909)
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_find_browser_window_matching",
                               return_value=new_win), \
             mock.patch.object(self.bc, "_monitor_name_for_window",
                               return_value="middle"), \
             mock.patch.object(self.bc, "find_click_target", return_value=(50, 60)), \
             mock.patch.object(self.bc, "ui_click"), \
             mock.patch.object(self.bc, "_streaming_go_fullscreen"), \
             mock.patch.object(self.bc, "_ensure_window_visible_maximized") as place, \
             mock.patch.dict(self.bc._JARVIS_MEDIA_WINDOW_HWND, {}, clear=True):
            self.bc._streaming_auto_play("netflix", "the matrix")
            # Capture inside the patch.dict scope (it restores on exit).
            recorded = self.bc._JARVIS_MEDIA_WINDOW_HWND.get("netflix")
        place.assert_called_once_with(909)
        # And the handle was recorded for safe reuse/close next time.
        self.assertEqual(recorded, 909)

    def test_fullscreen_key_only_after_confirmed_playback(self):
        # ISSUE B end-to-end (strict service, verify_play): the fullscreen key
        # is sent ONLY once playback is confirmed, and only to the recorded
        # hwnd. Drive a strict service through _streaming_play_and_verify with a
        # confirmed play and assert ui_press('f') fired on the recorded window.
        cfg = {
            "name": "Netflix", "fullscreen_key": "f", "fullscreen_wait": 0,
            "service_key": "netflix",
            "play_strategies": ["play_button"], "verify_attempts": 1,
            "verify_wait": 0,
        }
        import core.config as _cfg
        with mock.patch.object(self.bc, "_streaming_apply_play_strategy",
                               return_value=(True, "clicked play")), \
             mock.patch.object(self.bc, "_streaming_verify_playback",
                               return_value=(True, "YES")), \
             mock.patch.object(_cfg, "STREAMING_AUTO_FULLSCREEN", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "_focus_window_hwnd",
                               return_value=True) as focus, \
             mock.patch.object(self.bc, "ui_press") as press, \
             mock.patch.dict(self.bc._JARVIS_MEDIA_WINDOW_HWND,
                             {"netflix": 909}, clear=True), \
             mock.patch.object(self.bc.time, "sleep"):
            out = self.bc._streaming_play_and_verify(cfg, "Netflix", "the matrix")
        self.assertEqual(out, "playing 'the matrix' on Netflix")
        focus.assert_called_once_with(909)
        press.assert_called_once_with("f")

    def test_no_fullscreen_key_when_playback_never_confirms(self):
        # If playback never confirms, the flow returns the could-not-confirm
        # message and the fullscreen key is NEVER sent (nothing is playing).
        cfg = {
            "name": "Netflix", "fullscreen_key": "f", "fullscreen_wait": 0,
            "service_key": "netflix",
            "play_strategies": ["play_button"], "verify_attempts": 1,
            "verify_wait": 0,
        }
        import core.config as _cfg
        with mock.patch.object(self.bc, "_streaming_apply_play_strategy",
                               return_value=(True, "clicked play")), \
             mock.patch.object(self.bc, "_streaming_verify_playback",
                               return_value=(False, "NO paused")), \
             mock.patch.object(_cfg, "STREAMING_AUTO_FULLSCREEN", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "ui_press") as press, \
             mock.patch.dict(self.bc._JARVIS_MEDIA_WINDOW_HWND,
                             {"netflix": 909}, clear=True), \
             mock.patch.object(self.bc.time, "sleep"):
            out = self.bc._streaming_play_and_verify(cfg, "Netflix", "the matrix")
        self.assertIn("couldn't confirm", out)
        press.assert_not_called()

    def test_knob_off_no_fullscreen_key_even_after_confirm(self):
        # ISSUE B master switch: STREAMING_AUTO_FULLSCREEN off ⇒ playback still
        # confirmed + reported, but NO fullscreen key is sent.
        cfg = {
            "name": "Netflix", "fullscreen_key": "f", "fullscreen_wait": 0,
            "service_key": "netflix",
            "play_strategies": ["play_button"], "verify_attempts": 1,
            "verify_wait": 0,
        }
        import core.config as _cfg
        with mock.patch.object(self.bc, "_streaming_apply_play_strategy",
                               return_value=(True, "clicked play")), \
             mock.patch.object(self.bc, "_streaming_verify_playback",
                               return_value=(True, "YES")), \
             mock.patch.object(_cfg, "STREAMING_AUTO_FULLSCREEN", False), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "_focus_window_hwnd") as focus, \
             mock.patch.object(self.bc, "ui_press") as press, \
             mock.patch.dict(self.bc._JARVIS_MEDIA_WINDOW_HWND,
                             {"netflix": 909}, clear=True), \
             mock.patch.object(self.bc.time, "sleep"):
            out = self.bc._streaming_play_and_verify(cfg, "Netflix", "the matrix")
        self.assertEqual(out, "playing 'the matrix' on Netflix")
        focus.assert_not_called()
        press.assert_not_called()


@requires_monolith
class AppleMusicPlayPlaylistTests(MonolithGlobalsTestCase):
    """_apple_music_play_playlist — Library>Playlists direct navigation."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_capabilities_missing(self):
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", False):
            out = self.bc._apple_music_play_playlist("chill")
        self.assertIn("auto-click needs", out)

    def test_playlist_not_found_after_sidebar_fallback(self):
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_streaming_find_with_retry", return_value=None), \
             mock.patch.object(self.bc, "find_click_target", return_value=None):
            out = self.bc._apple_music_play_playlist("chill")
        self.assertIn("couldn't find a playlist named 'chill'", out)

    def test_playlist_found_delegates_to_play_and_verify(self):
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_streaming_find_with_retry", return_value=(11, 22)), \
             mock.patch.object(self.bc, "ui_click"), \
             mock.patch.object(self.bc, "_streaming_play_and_verify",
                               return_value="playing 'chill' on Apple Music") as pv:
            out = self.bc._apple_music_play_playlist("chill")
        self.assertEqual(out, "playing 'chill' on Apple Music")
        pv.assert_called_once()
        # v1.84.0 [10]: the playlist flow passes a cfg COPY with verify_first
        # disabled (a freshly-opened Library>Playlists view has nothing "already
        # playing"), so _streaming_play_and_verify never skips the real play step;
        # and it must not mutate the shared _STREAMING_SERVICES template.
        cfg_arg = pv.call_args[0][0]
        self.assertFalse(cfg_arg.get("verify_first"))
        self.assertIsNot(cfg_arg, self.bc._STREAMING_SERVICES["apple_music"])


@requires_monolith
class CameraHealthTests(MonolithGlobalsTestCase):
    """get_camera_health — snapshot of webcam I/O health."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_returns_entry_per_configured_camera(self):
        health = self.bc.get_camera_health()
        self.assertIsInstance(health, dict)
        for cam in self.bc.CAMERAS:
            self.assertIn(cam["index"], health)
        # Each entry has the documented keys.
        if health:
            entry = next(iter(health.values()))
            for key in ("last_frame_at", "last_read_error",
                        "last_read_error_at", "wake_attempts", "recoveries"):
                self.assertIn(key, entry)


@requires_monolith
class AppleMusicChromeActiveTests(MonolithGlobalsTestCase):
    """_apple_music_chrome_active / _note_apple_music_seen — browser-routing cache."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        # _apple_music_last_seen is a directly-mutated module global.
        self._saved_seen = self.bc._apple_music_last_seen[0]

    def tearDown(self):
        self.bc._apple_music_last_seen[0] = self._saved_seen

    def test_note_sets_timestamp(self):
        self.bc._apple_music_last_seen[0] = 0.0
        with mock.patch.object(self.bc.time, "time", return_value=12345.0):
            self.bc._note_apple_music_seen()
        self.assertEqual(self.bc._apple_music_last_seen[0], 12345.0)

    def test_live_window_title_match(self):
        win = mock.MagicMock()
        win.title = "Apple Music — Now Playing"
        gw = mock.MagicMock()
        gw.getAllWindows.return_value = [win]
        self.bc._apple_music_last_seen[0] = 0.0
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertTrue(self.bc._apple_music_chrome_active())
        # Sighting warmed the cache.
        self.assertGreater(self.bc._apple_music_last_seen[0], 0.0)

    def test_no_window_but_warm_cache(self):
        gw = mock.MagicMock()
        gw.getAllWindows.return_value = []
        self.bc._apple_music_last_seen[0] = time.time()
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertTrue(self.bc._apple_music_chrome_active())

    def test_no_window_stale_cache(self):
        gw = mock.MagicMock()
        gw.getAllWindows.return_value = []
        self.bc._apple_music_last_seen[0] = 0.0
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertFalse(self.bc._apple_music_chrome_active())

    def test_pygetwindow_absent_uses_cache_only(self):
        self.bc._apple_music_last_seen[0] = 0.0
        with mock.patch.dict(sys.modules, {"pygetwindow": None}):
            self.assertFalse(self.bc._apple_music_chrome_active())


@requires_monolith
class ItunesHelperTests(MonolithGlobalsTestCase):
    """_itunes_is_running / _get_itunes — thin bridge wrappers."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_is_running_delegates_to_bridge(self):
        with mock.patch.object(self.bc._itunes_bridge, "is_running", return_value=True):
            self.assertTrue(self.bc._itunes_is_running())

    def test_get_itunes_delegates_with_kwargs(self):
        sentinel = (object(), None)
        with mock.patch.object(self.bc._itunes_bridge, "get_client",
                               return_value=sentinel) as gc:
            self.assertIs(self.bc._get_itunes(force=True), sentinel)
            self.assertTrue(gc.call_args.kwargs["force"])


@requires_monolith
class RunItunesComTimeoutTests(MonolithGlobalsTestCase):
    """_run_itunes_com_timeout — runs COM work on a join-timeout daemon."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_returns_work_result(self):
        self.assertEqual(
            self.bc._run_itunes_com_timeout(lambda: (True, "ok"), timeout=5),
            (True, "ok"),
        )

    def test_timeout_returns_timeout_msg(self):
        def slow():
            time.sleep(2.0)
            return (True, "late")

        out = self.bc._run_itunes_com_timeout(
            slow, timeout=0.15, timeout_msg=(False, "iTunes not responding"))
        self.assertEqual(out, (False, "iTunes not responding"))

    def test_worker_exception_propagates(self):
        def boom():
            raise ValueError("com blew up")

        with self.assertRaises(ValueError):
            self.bc._run_itunes_com_timeout(boom, timeout=5)


@requires_monolith
class PlayMusicCoreTests(MonolithGlobalsTestCase):
    """_play_music_core — iTunes COM search/play, fully mocked."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_empty_args(self):
        ok, msg = self.bc._play_music_core("   ")
        self.assertFalse(ok)
        self.assertIn("format:", msg)

    def test_itunes_unavailable(self):
        with mock.patch.object(self.bc, "_get_itunes", return_value=(None, "iTunes off")):
            ok, msg = self.bc._play_music_core("the beatles")
        self.assertFalse(ok)
        self.assertEqual(msg, "iTunes off")

    def test_no_matches(self):
        app = mock.MagicMock()
        tracks = mock.MagicMock()
        tracks.Count = 0
        app.LibraryPlaylist.Search.return_value = tracks
        with mock.patch.object(self.bc, "_get_itunes", return_value=(app, None)):
            ok, msg = self.bc._play_music_core("nonexistent song")
        self.assertFalse(ok)
        self.assertIn("no tracks found", msg)

    def test_successful_play(self):
        app = mock.MagicMock()
        tracks = mock.MagicMock()
        tracks.Count = 1
        first = mock.MagicMock()
        first.Name = "Yesterday"
        first.Artist = "The Beatles"
        tracks.Item.return_value = first
        app.LibraryPlaylist.Search.return_value = tracks
        with mock.patch.object(self.bc, "_get_itunes", return_value=(app, None)):
            ok, msg = self.bc._play_music_core("yesterday")
        self.assertTrue(ok)
        self.assertIn("Yesterday", msg)
        self.assertIn("The Beatles", msg)
        first.Play.assert_called_once()

    def test_field_prefix_parsing(self):
        app = mock.MagicMock()
        tracks = mock.MagicMock()
        tracks.Count = 2
        first = mock.MagicMock()
        first.Name = "Song"
        first.Artist = "Artist"
        tracks.Item.return_value = first
        app.LibraryPlaylist.Search.return_value = tracks
        with mock.patch.object(self.bc, "_get_itunes", return_value=(app, None)):
            ok, msg = self.bc._play_music_core("artist: The Beatles")
        self.assertTrue(ok)
        # Searched the ARTISTS field with the stripped query.
        called_args = app.LibraryPlaylist.Search.call_args[0]
        self.assertEqual(called_args[0], "The Beatles")
        self.assertEqual(called_args[1], self.bc._ITUNES_SEARCH_ARTISTS)
        self.assertIn("2 matches", msg)

    def test_com_exception_during_play(self):
        app = mock.MagicMock()
        app.LibraryPlaylist.Search.side_effect = RuntimeError("COM error")
        with mock.patch.object(self.bc, "_get_itunes", return_value=(app, None)):
            ok, msg = self.bc._play_music_core("song")
        self.assertFalse(ok)
        self.assertIn("iTunes playback failed", msg)


@requires_monolith
class SessionResumeReaderTests(MonolithGlobalsTestCase):
    """_last_session_end_ts / _last_n_user_commands / _last_queued_task_line /
    _summarise_task_line — the session-resume data sources."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    # ---- _last_session_end_ts ----
    def test_last_ts_from_epoch(self):
        pm = mock.MagicMock()
        pm.get_session_summaries.return_value = [{"ts": 1700000000.0}]
        with mock.patch.object(self.bc, "pattern_memory", pm):
            self.assertEqual(self.bc._last_session_end_ts(), 1700000000.0)

    def test_last_ts_from_iso(self):
        pm = mock.MagicMock()
        pm.get_session_summaries.return_value = [{"iso_end": "2026-05-30T20:00:00"}]
        with mock.patch.object(self.bc, "pattern_memory", pm):
            ts = self.bc._last_session_end_ts()
        self.assertEqual(ts, time.mktime(time.strptime("2026-05-30T20:00:00",
                                                        "%Y-%m-%dT%H:%M:%S")))

    def test_last_ts_empty_returns_zero(self):
        pm = mock.MagicMock()
        pm.get_session_summaries.return_value = []
        with mock.patch.object(self.bc, "pattern_memory", pm):
            self.assertEqual(self.bc._last_session_end_ts(), 0.0)

    def test_last_ts_exception_returns_zero(self):
        pm = mock.MagicMock()
        pm.get_session_summaries.side_effect = RuntimeError("db down")
        with mock.patch.object(self.bc, "pattern_memory", pm):
            self.assertEqual(self.bc._last_session_end_ts(), 0.0)

    # ---- _last_n_user_commands ----
    def test_last_n_commands_newest_first(self):
        data = "\n".join([
            json.dumps({"text": "first"}),
            json.dumps({"text": "second"}),
            "not json",
            json.dumps({"text": "third"}),
        ])
        with mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=data)):
            self.assertEqual(self.bc._last_n_user_commands(2), ["third", "second"])

    def test_last_n_commands_missing_file(self):
        with mock.patch.object(self.bc.os.path, "exists", return_value=False):
            self.assertEqual(self.bc._last_n_user_commands(3), [])

    # ---- _last_queued_task_line ----
    def test_last_queued_skips_internal_tasks(self):
        td = tempfile.mkdtemp()
        todo = os.path.join(td, "todo.md")
        with open(todo, "w", encoding="utf-8") as f:
            f.write("- [ ] **2026-05-30 10:00** [anomaly] — internal burst\n")
            f.write("- [ ] **2026-05-30 11:00** [feature] — Build the thing.\n")
        with mock.patch.object(self.bc, "TODO_FILE", todo):
            line = self.bc._last_queued_task_line()
        self.assertIn("Build the thing", line)
        self.assertNotIn("anomaly", line)

    def test_last_queued_missing_file(self):
        with mock.patch.object(self.bc, "TODO_FILE",
                               os.path.join(tempfile.mkdtemp(), "nope.md")):
            self.assertEqual(self.bc._last_queued_task_line(), "")

    def test_last_queued_only_internal_returns_empty(self):
        td = tempfile.mkdtemp()
        todo = os.path.join(td, "todo.md")
        with open(todo, "w", encoding="utf-8") as f:
            f.write("- [ ] **2026-05-30 10:00** [self-heal] — fix it\n")
            f.write("- [ ] **2026-05-30 11:00** [deep-audit] — scan\n")
        with mock.patch.object(self.bc, "TODO_FILE", todo):
            self.assertEqual(self.bc._last_queued_task_line(), "")


# ──────────────────────────────────────────────────────────────────────────
#  COVERAGE EXTENSION (2026-06) — additional branches in the sec4 range
#  (take_screenshot monitor/PIL/encode paths, ask_vision connection-error and
#  no-local tails, ask_vision_multi error fallbacks, the pyautogui-import and
#  ui_* reticle branches, offscreen-capture poll/SetWindowPos/close paths, the
#  streaming keyboard+strict orchestration, Apple Music sidebar fallback,
#  iTunes COM pythoncom-absent paths, and session-resume reader edge cases).
# ──────────────────────────────────────────────────────────────────────────


def _fake_mss_module(sct):
    """A MagicMock standing in for the ``mss`` module whose ``mss()`` context
    manager yields *sct*. ``MSS`` is deleted so the monolith's
    ``getattr(mss, 'MSS', mss.mss)`` resolves to ``mss.mss`` (mirrors the
    existing TakeScreenshotTests idiom)."""
    mssmod = mock.MagicMock()
    mssmod.mss.return_value.__enter__ = lambda s: sct
    mssmod.mss.return_value.__exit__ = lambda *a: False
    del mssmod.MSS
    return mssmod


@requires_monolith
class TakeScreenshotBranchTests(MonolithGlobalsTestCase):
    """take_screenshot — monitor-region, PIL-fallback, and encode-failure
    branches not exercised by the original TakeScreenshotTests."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _fake_pil(self, size=(800, 600)):
        img = mock.MagicMock()
        img.size = size
        img.resize.return_value = img

        def _save(buf, format=None, optimize=None):
            buf.write(b"PNGBYTES")

        img.save.side_effect = _save
        pil = mock.MagicMock()
        pil.Image.frombytes.return_value = img
        pil.Image.LANCZOS = 1
        return pil, img

    def test_named_monitor_uses_monitor_region(self):
        # monitor in MONITORS → builds an explicit {left,top,width,height}
        # region from MONITORS[name] rather than sct.monitors[0]. (line 6901-2)
        name = next(iter(self.bc.MONITORS))
        pil, _img = self._fake_pil()
        sct = mock.MagicMock()
        raw = mock.MagicMock()
        raw.size = (800, 600)
        raw.bgra = b""
        sct.grab.return_value = raw
        # sct.monitors[0] is deliberately NOT used on this path.
        sct.monitors = [{"left": 999, "top": 999, "width": 1, "height": 1}]
        with mock.patch.dict(sys.modules,
                             {"mss": _fake_mss_module(sct), "PIL": pil}):
            self.assertEqual(self.bc.take_screenshot(monitor=name), b"PNGBYTES")
        x, y, w, h = self.bc.MONITORS[name]
        sct.grab.assert_called_once_with(
            {"left": x, "top": y, "width": w, "height": h})

    def test_pil_fallback_when_mss_raises(self):
        # mss import/grab explodes → ImageGrab fallback. With monitor=None the
        # fallback must grab the WHOLE virtual desktop (bbox = the virtual
        # bounds, all_screens=True) to match the mss path — a bare grab() would
        # return only the primary monitor and mistranslate clicks on secondary /
        # negative-origin displays. Geometry is injected so the asserted bbox is
        # host-independent (was: asserted grab() and leaked the real box layout).
        mons = {"left": (-2560, 0, 2560, 1440), "main": (0, 0, 2560, 1440)}
        pil, _img = self._fake_pil()
        grabbed = mock.MagicMock()
        grabbed.size = (800, 600)
        grabbed.resize.return_value = grabbed
        grabbed.save.side_effect = lambda buf, **k: buf.write(b"PILBYTES")
        pil.ImageGrab.grab.return_value = grabbed
        mssmod = mock.MagicMock()
        mssmod.mss.side_effect = RuntimeError("no mss here")
        del mssmod.MSS
        with mock.patch.object(self.bc, "MONITORS", mons), \
             mock.patch.dict(sys.modules, {"mss": mssmod, "PIL": pil}):
            self.assertEqual(self.bc.take_screenshot(), b"PILBYTES")
        # virtual bounds of `mons` = (-2560,0)..(2560,1440) → bbox below.
        pil.ImageGrab.grab.assert_called_once_with(
            bbox=(-2560, 0, 2560, 1440), all_screens=True)

    def test_pil_fallback_named_monitor_uses_bbox(self):
        # mss fails AND a known monitor is requested → ImageGrab.grab(bbox=…,
        # all_screens=True). (lines 6913-6915)
        name = next(iter(self.bc.MONITORS))
        x, y, w, h = self.bc.MONITORS[name]
        pil, _img = self._fake_pil()
        grabbed = mock.MagicMock()
        grabbed.size = (w, h)
        grabbed.resize.return_value = grabbed
        grabbed.save.side_effect = lambda buf, **k: buf.write(b"BBOXBYTES")
        pil.ImageGrab.grab.return_value = grabbed
        mssmod = mock.MagicMock()
        mssmod.mss.side_effect = RuntimeError("nope")
        del mssmod.MSS
        with mock.patch.dict(sys.modules, {"mss": mssmod, "PIL": pil}):
            self.assertEqual(self.bc.take_screenshot(monitor=name), b"BBOXBYTES")
        pil.ImageGrab.grab.assert_called_once_with(
            bbox=(x, y, x + w, y + h), all_screens=True)

    def test_both_capture_paths_fail_returns_none(self):
        # mss raises and ImageGrab also raises → None. (lines 6918-6920)
        pil, _img = self._fake_pil()
        pil.ImageGrab.grab.side_effect = RuntimeError("no display")
        mssmod = mock.MagicMock()
        mssmod.mss.side_effect = RuntimeError("no mss")
        del mssmod.MSS
        with mock.patch.dict(sys.modules, {"mss": mssmod, "PIL": pil}):
            self.assertIsNone(self.bc.take_screenshot())

    def test_encode_failure_returns_none(self):
        # img.save() raises during PNG encode → None. (lines 6931-6933)
        pil, img = self._fake_pil()
        img.save.side_effect = RuntimeError("encode boom")
        sct = mock.MagicMock()
        raw = mock.MagicMock()
        raw.size = (800, 600)
        raw.bgra = b""
        sct.grab.return_value = raw
        sct.monitors = [{"left": 0, "top": 0, "width": 800, "height": 600}]
        with mock.patch.dict(sys.modules,
                             {"mss": _fake_mss_module(sct), "PIL": pil}):
            self.assertIsNone(self.bc.take_screenshot())


@requires_monolith
class AskVisionTailTests(MonolithGlobalsTestCase):
    """ask_vision — the connection/timeout-error and no-local tail branches
    (lines 6982-7000) the original AskVisionTests didn't reach."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_connection_error_falls_back_to_local(self):
        # APIConnectionError → local VLM fallback returns text. (lines 6982-6986)
        anth = _make_fake_anthropic()
        anth.Anthropic.side_effect = anth.APIConnectionError("unreachable")
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value="offline eye"), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            self.assertEqual(self.bc.ask_vision("q", b"PNG"),
                             "[local-vision] offline eye")

    def test_connection_error_no_local_returns_failure(self):
        # APITimeoutError + no local → "(vision failed: …)". (line 6987)
        anth = _make_fake_anthropic()
        anth.Anthropic.side_effect = anth.APITimeoutError("timed out")
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value=None), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            out = self.bc.ask_vision("q", b"PNG")
        self.assertIn("vision failed", out)

    def test_catchall_no_local_returns_typed_failure(self):
        # Generic (non-anthropic) error + no local → "(vision failed:
        # <ExcType>)" via the catch-all. (line 7000)
        anth = _make_fake_anthropic()
        anth.Anthropic.side_effect = RuntimeError("weird sdk")
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value=None), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            out = self.bc.ask_vision("q", b"PNG")
        self.assertIn("vision failed", out)
        self.assertIn("RuntimeError", out)

    def test_screenshot_taken_when_png_none(self):
        # png_bytes omitted → take_screenshot() is invoked. (line 6947)
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(self.bc, "take_screenshot",
                               return_value=b"CAP") as cap, \
             mock.patch.object(self.bc, "_call_local_vision", return_value="x"):
            self.bc.ask_vision("q")
        cap.assert_called_once()


@requires_monolith
class AskVisionMultiTailTests(MonolithGlobalsTestCase):
    """ask_vision_multi — non-claude no-local stub and the Claude-error→local
    fallback tails (lines 7056, 7084-7107)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_non_claude_no_local_returns_stub(self):
        # AI_BACKEND != claude and local fallback empty → stub. (line 7056)
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value=None):
            out = self.bc.ask_vision_multi("q", {"left": b"a"})
        self.assertIn("requires Claude backend", out)

    def test_status_error_falls_back_to_local(self):
        # Claude APIStatusError → _local_multi_fallback returns text. (7084-7090)
        anth = _make_fake_anthropic()
        err = anth.APIStatusError("rate")
        err.status_code = 429
        anth.Anthropic.side_effect = err
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value="m"), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            out = self.bc.ask_vision_multi("q", {"left": b"a"})
        self.assertEqual(out, "[local-vision] m")

    def test_status_error_no_local_returns_http_failure(self):
        # Claude APIStatusError + no local → "(vision failed: HTTP …)". (7091)
        anth = _make_fake_anthropic()
        err = anth.APIStatusError("server")
        err.status_code = 503
        anth.Anthropic.side_effect = err
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value=None), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            out = self.bc.ask_vision_multi("q", {"left": b"a"})
        self.assertIn("HTTP 503", out)

    def test_connection_error_no_local_returns_failure(self):
        # APIConnectionError + no local → "(vision failed: …)". (7092-7097)
        anth = _make_fake_anthropic()
        anth.Anthropic.side_effect = anth.APIConnectionError("down")
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value=None), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            out = self.bc.ask_vision_multi("q", {"left": b"a"})
        self.assertIn("vision failed", out)

    def test_connection_error_falls_back_to_local(self):
        # APITimeoutError → local fallback returns text. (7092-7096)
        anth = _make_fake_anthropic()
        anth.Anthropic.side_effect = anth.APITimeoutError("slow")
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value="ans"), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            out = self.bc.ask_vision_multi("q", {"left": b"a"})
        self.assertEqual(out, "[local-vision] ans")

    def test_catchall_falls_back_to_local(self):
        # Generic error in the Claude path → catch-all routes to local. (7098-7106)
        anth = _make_fake_anthropic()
        anth.Anthropic.side_effect = RuntimeError("kaboom")
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value="cf"), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            out = self.bc.ask_vision_multi("q", {"left": b"a"})
        self.assertEqual(out, "[local-vision] cf")

    def test_catchall_no_local_returns_typed_failure(self):
        # Generic error + no local → "(vision failed: <ExcType>)". (7107)
        anth = _make_fake_anthropic()
        anth.Anthropic.side_effect = RuntimeError("kaboom")
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_call_local_vision", return_value=None), \
             mock.patch.dict(sys.modules, {"anthropic": anth}):
            out = self.bc.ask_vision_multi("q", {"left": b"a"})
        self.assertIn("vision failed", out)
        self.assertIn("RuntimeError", out)


@requires_monolith
class FindClickTargetTranslateTests(MonolithGlobalsTestCase):
    """find_click_target — the no-monitor virtual-origin translate failure
    branch (lines 7255-7256) where the mss re-open raises and the raw refined
    coords are returned unchanged."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    @staticmethod
    def _real_png(w, h):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (w, h)).save(buf, format="PNG")
        return buf.getvalue()

    # Single-monitor layout anchored at the ORIGIN (0,0) with logical == the
    # mocked native size, injected so the translate math is deterministic on any
    # host. With origin (0,0) and scale 1.0 the absolute result equals the raw
    # refined coords — the original intent of this test, now made host-safe.
    _ORIGIN_MONITOR = {"main": (0, 0, 100, 80)}

    def test_virtual_origin_lookup_failure_returns_raw_coords(self):
        # Pass-1 succeeds on a real (small) PNG; the Pass-2 full-res capture
        # returns None so refine is skipped; with no monitor specified the
        # translate uses _virtual_screen_bounds() for the LOGICAL origin and the
        # native size from _native_capture_size. mss is unavailable here, so
        # _captured_region returns None (no origin-mismatch warning) and the math
        # reduces to origin(0,0) + refined(40,50) → (40,50) unchanged.
        png1 = self._real_png(100, 80)
        with mock.patch.object(self.bc, "MONITORS", self._ORIGIN_MONITOR), \
             mock.patch.object(self.bc, "take_screenshot",
                               side_effect=[png1, None]), \
             mock.patch.object(self.bc, "_native_capture_size",
                               return_value=(100, 80)), \
             mock.patch.object(self.bc, "_query_vision_for_coords",
                               return_value=(40, 50)), \
             mock.patch.dict(sys.modules, {"mss": None}):
            # virtual origin (0,0), native==logical (100x80) → scale 1.0 →
            # refined (40,50) returned unchanged.
            self.assertEqual(self.bc.find_click_target("a button"), (40, 50))


@requires_monolith
class GetPyautoguiImportSuccessTests(MonolithGlobalsTestCase):
    """_get_pyautogui — the successful-import branch (lines 7282-7284) that
    sets FAILSAFE/PAUSE and caches the module."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved = self.bc._pyautogui

    def tearDown(self):
        self.bc._pyautogui = self._saved

    def test_import_success_configures_and_caches(self):
        self.bc._pyautogui = None
        fake = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"pyautogui": fake}):
            got = self.bc._get_pyautogui()
        self.assertIs(got, fake)
        self.assertIs(self.bc._pyautogui, fake)
        self.assertTrue(fake.FAILSAFE)
        self.assertEqual(fake.PAUSE, 0.15)


@requires_monolith
class UISafeRetryReraiseTests(MonolithGlobalsTestCase):
    """_ui_safe / _nudge_from_corner — the retry-raises-a-non-failsafe path
    (lines 7331-7332) and the nudge position-error swallow (7310-7311)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_retry_raises_other_exception_propagates(self):
        # First call FailSafe → nudge → retry raises a *different* error,
        # which must propagate (not be wrapped as UIFailsafeError). (7331)
        pag = _FakePag(pos=(0, 0))
        calls = {"n": 0}

        def op():
            calls["n"] += 1
            if calls["n"] == 1:
                raise FailSafeException()
            raise ValueError("second-call boom")

        with self.assertRaises(ValueError):
            self.bc._ui_safe(pag, op)
        self.assertEqual(calls["n"], 2)

    def test_nudge_position_error_returns_false(self):
        # pag.position() raising → _nudge_from_corner swallows and returns
        # False. (lines 7310-7311)
        pag = _FakePag()
        pag.position = mock.Mock(side_effect=RuntimeError("no cursor"))
        self.assertFalse(self.bc._nudge_from_corner(pag))

    def test_failsafe_with_no_nudge_raises_ui_error(self):
        # FailSafe fires but the cursor is NOT in a corner, so
        # _nudge_from_corner returns False → no retry, raise UIFailsafeError
        # directly. (line 7332)
        pag = _FakePag(pos=(500, 500))  # mid-screen → no nudge

        def op():
            raise FailSafeException()

        with self.assertRaises(self.bc.UIFailsafeError):
            self.bc._ui_safe(pag, op)
        self.assertEqual(pag.moved, [])  # no nudge attempted


@requires_monolith
class UIWrapperBranchTests(MonolithGlobalsTestCase):
    """ui_* wrappers — the no-pyautogui early returns for double_click/hotkey/
    scroll (lines 7361, 7386, 7395) and the ui_press reticle branch when a
    window centre is available (line 7380)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_ui_double_click_noop_without_pyautogui(self):
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=None):
            self.assertIsNone(self.bc.ui_double_click(1, 2))

    def test_ui_hotkey_noop_without_pyautogui(self):
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=None):
            self.assertIsNone(self.bc.ui_hotkey("ctrl", "c"))

    def test_ui_scroll_noop_without_pyautogui(self):
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=None):
            self.assertIsNone(self.bc.ui_scroll(3))

    def test_ui_press_publishes_reticle_at_window_center(self):
        # center is not None → the reticle is published at it. (line 7380)
        pag = _FakePag()
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(self.bc, "_active_window_center", return_value=(9, 12)), \
             mock.patch.object(self.bc, "_publish_reticle") as ret:
            self.bc.ui_press("enter")
        self.assertEqual(pag.pressed, ["enter"])
        self.assertEqual(ret.call_args[0][:2], (9, 12))

    def test_ui_click_virtual_bounds_error_still_clicks(self):
        # _virtual_screen_bounds raising is swallowed; click proceeds with the
        # original (unclamped) coords. (lines 7348-7349)
        pag = _FakePag()
        with mock.patch.object(self.bc, "_get_pyautogui", return_value=pag), \
             mock.patch.object(self.bc, "_publish_reticle"), \
             mock.patch.object(self.bc, "_virtual_screen_bounds",
                               side_effect=RuntimeError("no bounds")):
            self.bc.ui_click(33, 44)
        self.assertEqual(pag.clicks, [(33, 44, "left")])


@requires_monolith
class OffscreenCaptureFlowTests(MonolithGlobalsTestCase):
    """_open_url_offscreen_capture — the scheme-prefix, window-found poll,
    SetWindowPos-failure, capture-error, and printwindow_failed paths past the
    early guards already covered by OpenUrlOffscreenCaptureTests.

    NOTE: the win32 GDI PrintWindow/bitmap-blit *happy* path (≈ lines
    7581-7610) needs real HWND/DC handles and is not unit-testable here — it is
    left for a later ``# pragma: no cover`` pass."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _win32(self, found_hwnd=4242):
        """A win32gui mock whose EnumWindows reports exactly one NEW chrome
        window (so the spawn-poll finds a target). The GDI capture is forced
        to fail so we exercise the post-capture cleanup/close without needing
        real handles."""
        w32 = mock.MagicMock()
        # EnumWindows(cb, None): the callback is driven by the monolith. We
        # bypass the real callback path by having _enum return our hwnd via a
        # side effect on EnumWindows that invokes the callback for our hwnd.
        def _enum(cb, _):
            cb(found_hwnd, None)
            return True
        w32.EnumWindows.side_effect = _enum
        w32.GetClassName.return_value = "Chrome_WidgetWin_1"
        w32.IsWindowVisible.return_value = True
        w32.GetWindowRect.return_value = (0, 0, 800, 600)
        return w32

    def test_prefixes_scheme_and_capture_fails_returns_printwindow_failed(self):
        # url lacks scheme (→ https:// prefixed, line 7489); a NEW window is
        # found; the GDI capture raises → png stays None → "printwindow_failed"
        # and WM_CLOSE is posted. (lines 7489, 7549-7554, 7635-7647)
        w32 = self._win32()
        # First _enum_chrome_hwnds() (pre) must be EMPTY, the second (post-
        # spawn) must contain our hwnd. Toggle GetWindowDC to raise so the
        # capture body fails fast.
        w32.GetWindowDC.side_effect = RuntimeError("no DC in tests")
        states = {"spawned": False}

        def _enum(cb, _):
            if states["spawned"]:
                cb(4242, None)
            return True
        w32.EnumWindows.side_effect = _enum

        def _popen(*a, **k):
            states["spawned"] = True
            return mock.MagicMock()

        # Monotonic clock that never exhausts: deadline = first_tick + 5.0, and
        # each subsequent tick is +0.1 so the poll loop stays inside the window
        # and exits by *finding* the new hwnd, not by timing out.
        clock = iter(100.0 + 0.1 * i for i in itertools.count())
        w32ui = mock.MagicMock()
        ctypes_mod = mock.MagicMock()
        pil = mock.MagicMock()
        with mock.patch.dict(sys.modules, {
                "win32gui": w32, "win32ui": w32ui, "win32con": mock.MagicMock(),
                "ctypes": ctypes_mod, "PIL": pil}), \
             mock.patch.object(self.bc, "_find_chrome", return_value=r"C:\chrome.exe"), \
             mock.patch.object(self.bc.subprocess, "Popen", side_effect=_popen), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc.time, "time", side_effect=lambda: next(clock)):
            png, reason = self.bc._open_url_offscreen_capture("example.com")
        self.assertIsNone(png)
        self.assertEqual(reason, "printwindow_failed")
        # WM_CLOSE (0x0010) posted to the found window.
        w32.PostMessage.assert_called_once_with(4242, 0x0010, 0, 0)

    def test_setwindowpos_error_swallowed_then_capture_fails(self):
        # The window is found (large rect passes the >200 enum filter), then
        # SetWindowPos raises and is swallowed (lines 7567-7568); the capture
        # also fails (GetWindowDC raises) → "printwindow_failed".
        w32 = self._win32()  # GetWindowRect → (0,0,800,600), passes the filter
        w32.SetWindowPos.side_effect = RuntimeError("park failed")
        w32.GetWindowDC.side_effect = RuntimeError("no DC")
        states = {"spawned": False}

        def _enum(cb, _):
            if states["spawned"]:
                cb(4242, None)
            return True
        w32.EnumWindows.side_effect = _enum

        def _popen(*a, **k):
            states["spawned"] = True
            return mock.MagicMock()

        clock = iter(100.0 + 0.1 * i for i in itertools.count())
        with mock.patch.dict(sys.modules, {
                "win32gui": w32, "win32ui": mock.MagicMock(),
                "win32con": mock.MagicMock(), "ctypes": mock.MagicMock(),
                "PIL": mock.MagicMock()}), \
             mock.patch.object(self.bc, "_find_chrome", return_value=r"C:\chrome.exe"), \
             mock.patch.object(self.bc.subprocess, "Popen", side_effect=_popen), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc.time, "time", side_effect=lambda: next(clock)):
            png, reason = self.bc._open_url_offscreen_capture("https://example.com")
        self.assertIsNone(png)
        self.assertEqual(reason, "printwindow_failed")
        w32.SetWindowPos.assert_called_once()

    def test_enum_callback_and_enumwindows_errors_swallowed(self):
        # Inside _enum_chrome_hwnds: the per-window callback swallows
        # GetClassName errors (lines 7516-7517) and EnumWindows itself is
        # guarded (7521-7522). With EnumWindows always raising, no window is
        # ever found → "chrome_window_not_found".
        w32 = mock.MagicMock()
        w32.GetClassName.side_effect = RuntimeError("class lookup failed")

        def _enum(cb, _):
            # Drive the callback once (exercising its except), then blow up to
            # exercise the outer guard.
            cb(1, None)
            raise RuntimeError("EnumWindows failed")
        w32.EnumWindows.side_effect = _enum

        clock = iter(100.0 + 6.0 * i for i in itertools.count())  # 2nd tick past deadline
        with mock.patch.dict(sys.modules, {
                "win32gui": w32, "win32ui": mock.MagicMock(),
                "win32con": mock.MagicMock(), "ctypes": mock.MagicMock(),
                "PIL": mock.MagicMock()}), \
             mock.patch.object(self.bc, "_find_chrome", return_value=r"C:\chrome.exe"), \
             mock.patch.object(self.bc.subprocess, "Popen", return_value=mock.MagicMock()), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc.time, "time", side_effect=lambda: next(clock)):
            png, reason = self.bc._open_url_offscreen_capture("https://example.com")
        self.assertIsNone(png)
        self.assertEqual(reason, "chrome_window_not_found")


@requires_monolith
class ExtractYoutubeUrlBranchTests(MonolithGlobalsTestCase):
    """_extract_youtube_url_from_search — the skill-raises swallow (lines
    7727-7728)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_skill_raises_then_serp_fallback(self):
        # find_direct_url raises → swallowed → SERP path runs and matches.
        skill = mock.MagicMock()
        skill.find_direct_url.side_effect = RuntimeError("yt-dlp died")
        resp = mock.Mock(status_code=200,
                         text="x https://www.youtube.com/watch?v=dQw4w9WgXcQ x")
        with mock.patch.dict(sys.modules, {"skill_youtube_search": skill}), \
             mock.patch.object(self.bc.requests, "get", return_value=resp):
            self.assertEqual(
                self.bc._extract_youtube_url_from_search("never gonna"),
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            )

    def test_skill_returns_empty_then_serp_fallback(self):
        # find_direct_url returns falsy → no early return → SERP path. (7725)
        skill = mock.MagicMock()
        skill.find_direct_url.return_value = ""
        resp = mock.Mock(status_code=200,
                         text="https://youtu.be/abcdefghijk here")
        with mock.patch.dict(sys.modules, {"skill_youtube_search": skill}), \
             mock.patch.object(self.bc.requests, "get", return_value=resp):
            self.assertEqual(
                self.bc._extract_youtube_url_from_search("clip"),
                "https://www.youtube.com/watch?v=abcdefghijk",
            )


@requires_monolith
class StreamingKeyboardSelectBranchTests(MonolithGlobalsTestCase):
    """_streaming_keyboard_select_first_result — disabled, happy Tab+Enter
    sequence, failsafe re-raise, and generic-error swallow."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _cfg(self, **over):
        cfg = {"keyboard_pre_wait": 0.0, "keyboard_tab_count": 3,
               "keyboard_tab_interval": 0.0, "keyboard_post_wait": 0.0}
        cfg.update(over)
        return cfg

    def test_disabled_returns_false(self):
        with mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", False):
            self.assertFalse(
                self.bc._streaming_keyboard_select_first_result(self._cfg(), "Svc"))

    def test_sends_tabs_then_enter(self):
        presses = []
        with mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "ui_press", side_effect=presses.append):
            ok = self.bc._streaming_keyboard_select_first_result(self._cfg(), "Svc")
        self.assertTrue(ok)
        # 3 tabs + 1 enter
        self.assertEqual(presses, ["tab", "tab", "tab", "enter"])

    def test_failsafe_propagates(self):
        with mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "ui_press",
                               side_effect=self.bc.UIFailsafeError("corner")):
            with self.assertRaises(self.bc.UIFailsafeError):
                self.bc._streaming_keyboard_select_first_result(self._cfg(), "Svc")

    def test_other_exception_returns_false(self):
        with mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "ui_press",
                               side_effect=RuntimeError("kbd lost")):
            self.assertFalse(
                self.bc._streaming_keyboard_select_first_result(self._cfg(), "Svc"))


@requires_monolith
class StreamingAutoPlayBranchTests(MonolithGlobalsTestCase):
    """_streaming_auto_play — keyboard-select path, strict-find failure,
    ui_click failsafe surfacing, strict delegation, and the default-path
    play-button-missing / failsafe / fullscreen branches."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _caps(self):
        """Enter the with-block having all auto-click capabilities enabled."""
        return [
            mock.patch.object(self.bc.webbrowser, "open"),
            mock.patch.object(self.bc.time, "sleep"),
            mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True),
            mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True),
            mock.patch.object(self.bc, "AI_BACKEND", "claude"),
        ]

    def test_keyboard_select_failsafe_returns_message(self):
        # Apple Music uses select_method=keyboard; the keyboard select raising
        # UIFailsafeError surfaces a friendly message. (lines 8233-8236)
        # resolve->None forces the SEARCH-PAGE (select_method="keyboard") path:
        # v1.35.0 (#56) added resolve_itunes, whose live iTunes call otherwise
        # switches to the resolved-track-page ("none") path and bypasses the
        # keyboard branch this test exercises.
        with self._caps()[0], self._caps()[1], \
             mock.patch.object(self.bc, "_apple_music_resolve_track",
                               return_value=None), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_streaming_keyboard_select_first_result",
                               side_effect=self.bc.UIFailsafeError("in a corner, sir")):
            out = self.bc._streaming_auto_play("apple_music", "some song")
        self.assertIn("in a corner", out)

    def test_keyboard_select_not_sent_returns_hint(self):
        # keyboard select returns False (UI automation unavailable) → hint.
        # (lines 8237-8242)
        # resolve->None forces the search-page keyboard path (see above /
        # v1.35.0 resolve_itunes note).
        with self._caps()[0], self._caps()[1], \
             mock.patch.object(self.bc, "_apple_music_resolve_track",
                               return_value=None), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_streaming_keyboard_select_first_result",
                               return_value=False):
            out = self.bc._streaming_auto_play("apple_music", "some song")
        self.assertIn("UI", out)
        self.assertIn("click the first result yourself", out)

    def test_keyboard_path_delegates_to_play_and_verify(self):
        # keyboard select succeeds, play_hint present, strict → delegates to
        # _streaming_play_and_verify. (lines 8249-context, 8279)
        with self._caps()[0], self._caps()[1], \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_streaming_keyboard_select_first_result",
                               return_value=True), \
             mock.patch.object(self.bc, "_streaming_play_and_verify",
                               return_value="playing 'x' on Apple Music") as pv:
            out = self.bc._streaming_auto_play("apple_music", "x")
        self.assertEqual(out, "playing 'x' on Apple Music")
        pv.assert_called_once()

    def test_strict_vision_find_miss_returns_hint(self):
        # Spotify is vision-select; force it strict via a patched cfg so the
        # strict find (via _streaming_find_with_retry) returning None yields
        # the "couldn't see the first result" hint. (lines 8248-8258)
        cfg = dict(self.bc._STREAMING_SERVICES["spotify"])
        cfg["verify_play"] = True
        with self._caps()[0], self._caps()[1], \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.dict(self.bc._STREAMING_SERVICES, {"spotify": cfg}), \
             mock.patch.object(self.bc, "_streaming_find_with_retry", return_value=None):
            out = self.bc._streaming_auto_play("spotify", "some album")
        self.assertIn("couldn't see the", out)

    def test_vision_click_failsafe_returns_message(self):
        # Vision select path: ui_click raises UIFailsafeError → message.
        # (lines 8259-8262)
        with self._caps()[0], self._caps()[1], \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "find_click_target", return_value=(5, 6)), \
             mock.patch.object(self.bc, "ui_click",
                               side_effect=self.bc.UIFailsafeError("corner")):
            out = self.bc._streaming_auto_play("netflix", "the matrix")
        self.assertIn("corner", out)

    def test_default_play_button_missing_returns_hint(self):
        # Netflix default path, play button not located → hint. (lines 8284-8288)
        with self._caps()[0], self._caps()[1], \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "find_click_target",
                               side_effect=[(10, 20), None]), \
             mock.patch.object(self.bc, "ui_click"):
            out = self.bc._streaming_auto_play("netflix", "the matrix")
        self.assertIn("couldn't locate the play", out)

    def test_default_play_click_failsafe_returns_message(self):
        # Netflix default path, play-button click raises UIFailsafeError.
        # (lines 8289-8292)
        with self._caps()[0], self._caps()[1], \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "find_click_target", return_value=(10, 20)), \
             mock.patch.object(self.bc, "ui_click",
                               side_effect=[None, self.bc.UIFailsafeError("corner")]):
            out = self.bc._streaming_auto_play("netflix", "the matrix")
        self.assertIn("corner", out)


@requires_monolith
class StreamingGoFullscreenErrorTests(MonolithGlobalsTestCase):
    """_streaming_go_fullscreen — the exception-swallow branch (8321-8322)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_fullscreen_press_error_swallowed(self):
        cfg = {"fullscreen_key": "f", "fullscreen_wait": 0.0}
        with mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "ui_press", side_effect=RuntimeError("boom")):
            # Must not raise.
            self.assertIsNone(self.bc._streaming_go_fullscreen(cfg, "Svc"))


@requires_monolith
class AppleMusicPlaylistSidebarTests(MonolithGlobalsTestCase):
    """_apple_music_play_playlist — the sidebar-fallback navigation branch
    (lines 8398-8438) when the direct URL doesn't surface the playlist."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _caps(self):
        return mock.patch.multiple(
            self.bc, SCREEN_VISION_ENABLED=True, UI_AUTOMATION_ENABLED=True,
            AI_BACKEND="claude")

    def test_sidebar_fallback_then_found_delegates(self):
        # find_with_retry: miss, then hit after sidebar clicks. Library and
        # Playlists links both found and clicked. (lines 8398-8425, 8434-8438)
        with mock.patch.object(self.bc.webbrowser, "open"), \
             mock.patch.object(self.bc.time, "sleep"), \
             self._caps(), \
             mock.patch.object(self.bc, "_streaming_find_with_retry",
                               side_effect=[None, (12, 13)]), \
             mock.patch.object(self.bc, "find_click_target",
                               side_effect=[(1, 1), (2, 2)]), \
             mock.patch.object(self.bc, "ui_click") as click, \
             mock.patch.object(self.bc, "_streaming_play_and_verify",
                               return_value="playing 'chill' on Apple Music") as pv:
            out = self.bc._apple_music_play_playlist("chill")
        self.assertEqual(out, "playing 'chill' on Apple Music")
        # Library click, Playlists click, then the playlist tile click.
        self.assertEqual(click.call_count, 3)
        pv.assert_called_once()

    def test_sidebar_library_click_failsafe(self):
        # Library sidebar click raises UIFailsafeError → friendly message.
        # (lines 8408-8411)
        with mock.patch.object(self.bc.webbrowser, "open"), \
             mock.patch.object(self.bc.time, "sleep"), \
             self._caps(), \
             mock.patch.object(self.bc, "_streaming_find_with_retry", return_value=None), \
             mock.patch.object(self.bc, "find_click_target", return_value=(1, 1)), \
             mock.patch.object(self.bc, "ui_click",
                               side_effect=self.bc.UIFailsafeError("corner")):
            out = self.bc._apple_music_play_playlist("chill")
        self.assertIn("couldn't navigate to Apple Music Library", out)

    def test_sidebar_playlists_click_failsafe(self):
        # Library click OK, Playlists sub-link click raises. (lines 8417-8421)
        with mock.patch.object(self.bc.webbrowser, "open"), \
             mock.patch.object(self.bc.time, "sleep"), \
             self._caps(), \
             mock.patch.object(self.bc, "_streaming_find_with_retry", return_value=None), \
             mock.patch.object(self.bc, "find_click_target",
                               side_effect=[(1, 1), (2, 2)]), \
             mock.patch.object(self.bc, "ui_click",
                               side_effect=[None, self.bc.UIFailsafeError("corner")]):
            out = self.bc._apple_music_play_playlist("chill")
        self.assertIn("couldn't open Apple Music Playlists", out)

    def test_playlist_tile_click_failsafe(self):
        # Playlist found on first try, but the tile click raises. (8434-8438)
        with mock.patch.object(self.bc.webbrowser, "open"), \
             mock.patch.object(self.bc.time, "sleep"), \
             self._caps(), \
             mock.patch.object(self.bc, "_streaming_find_with_retry", return_value=(7, 8)), \
             mock.patch.object(self.bc, "ui_click",
                               side_effect=self.bc.UIFailsafeError("corner")):
            out = self.bc._apple_music_play_playlist("chill")
        self.assertIn("found playlist 'chill'", out)
        self.assertIn("corner", out)


@requires_monolith
class AppleMusicChromeActiveErrorTests(MonolithGlobalsTestCase):
    """_apple_music_chrome_active — the generic-exception branch (8604-8605)
    where pygetwindow imports but enumerating windows raises."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        self._saved_seen = self.bc._apple_music_last_seen[0]

    def tearDown(self):
        self.bc._apple_music_last_seen[0] = self._saved_seen

    def test_getallwindows_raises_falls_through_to_cache(self):
        gw = mock.MagicMock()
        gw.getAllWindows.side_effect = RuntimeError("win enum blew up")
        self.bc._apple_music_last_seen[0] = 0.0   # cold cache → False
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertFalse(self.bc._apple_music_chrome_active())

    def test_getallwindows_raises_but_warm_cache_true(self):
        gw = mock.MagicMock()
        gw.getAllWindows.side_effect = RuntimeError("boom")
        self.bc._apple_music_last_seen[0] = time.time()  # warm → True
        with mock.patch.dict(sys.modules, {"pygetwindow": gw}):
            self.assertTrue(self.bc._apple_music_chrome_active())


@requires_monolith
class RunItunesComTimeoutPythoncomTests(MonolithGlobalsTestCase):
    """_run_itunes_com_timeout — the pythoncom-absent CoInitialize/Uninitialize
    swallow paths (lines 8647-8648, 8658-8659) run on the worker thread."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_pythoncom_absent_still_returns_result(self):
        # pythoncom import fails on the worker → CoInitialize swallowed (8647-8)
        # and the CoUninitialize block is skipped (com_inited stays False).
        with mock.patch.dict(sys.modules, {"pythoncom": None}):
            out = self.bc._run_itunes_com_timeout(lambda: (True, "ok"), timeout=5)
        self.assertEqual(out, (True, "ok"))

    def test_pythoncom_uninit_error_swallowed(self):
        # CoInitialize succeeds (com_inited=True) so the finally runs, but
        # CoUninitialize raises → swallowed; result still returned. (8658-8659)
        fake = mock.MagicMock()
        fake.CoUninitialize.side_effect = RuntimeError("uninit failed")
        with mock.patch.dict(sys.modules, {"pythoncom": fake}):
            out = self.bc._run_itunes_com_timeout(lambda: (True, "done"), timeout=5)
        self.assertEqual(out, (True, "done"))
        fake.CoInitialize.assert_called_once()


@requires_monolith
class PlayMusicCoreFieldTests(MonolithGlobalsTestCase):
    """_play_music_core — additional field-prefix mappings (song/album/track)
    and the single-match (no '(N matches)' suffix) message form."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _itunes_with_one_track(self, name="Track", artist="Artist", count=1):
        app = mock.MagicMock()
        tracks = mock.MagicMock()
        tracks.Count = count
        first = mock.MagicMock()
        first.Name = name
        first.Artist = artist
        tracks.Item.return_value = first
        app.LibraryPlaylist.Search.return_value = tracks
        return app

    def test_song_prefix_searches_songs_field(self):
        app = self._itunes_with_one_track()
        with mock.patch.object(self.bc, "_get_itunes", return_value=(app, None)):
            ok, _msg = self.bc._play_music_core("song: yesterday")
        self.assertTrue(ok)
        args = app.LibraryPlaylist.Search.call_args[0]
        self.assertEqual(args[0], "yesterday")
        self.assertEqual(args[1], self.bc._ITUNES_SEARCH_SONGS)

    def test_album_prefix_searches_albums_field(self):
        app = self._itunes_with_one_track()
        with mock.patch.object(self.bc, "_get_itunes", return_value=(app, None)):
            ok, _msg = self.bc._play_music_core("album: abbey road")
        self.assertTrue(ok)
        self.assertEqual(app.LibraryPlaylist.Search.call_args[0][1],
                         self.bc._ITUNES_SEARCH_ALBUMS)

    def test_track_prefix_maps_to_songs_field(self):
        app = self._itunes_with_one_track()
        with mock.patch.object(self.bc, "_get_itunes", return_value=(app, None)):
            ok, _msg = self.bc._play_music_core("track: come together")
        self.assertTrue(ok)
        self.assertEqual(app.LibraryPlaylist.Search.call_args[0][1],
                         self.bc._ITUNES_SEARCH_SONGS)

    def test_single_match_has_no_queue_suffix(self):
        app = self._itunes_with_one_track(name="Solo", artist="One", count=1)
        with mock.patch.object(self.bc, "_get_itunes", return_value=(app, None)):
            ok, msg = self.bc._play_music_core("solo")
        self.assertTrue(ok)
        self.assertNotIn("matches", msg)
        self.assertIn("Solo", msg)

    def test_itunes_handle_none_returns_error_message(self):
        # _get_itunes returns (None, err) inside the worker → (False, err).
        with mock.patch.object(self.bc, "_get_itunes",
                               return_value=(None, "iTunes asleep")):
            ok, msg = self.bc._play_music_core("anything")
        self.assertFalse(ok)
        self.assertEqual(msg, "iTunes asleep")


@requires_monolith
class SessionResumeReaderBranchTests(MonolithGlobalsTestCase):
    """_last_session_end_ts / _last_n_user_commands — the fall-through ladder
    (non-numeric ts → iso parse-fail → legacy date) and the file-open-error /
    blank-line paths not covered by SessionResumeReaderTests."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    # ---- _last_session_end_ts fall-through ladder ----
    def test_non_numeric_ts_falls_to_iso(self):
        # ts present but non-numeric → float() raises (8788-8789) → iso_end used.
        pm = mock.MagicMock()
        pm.get_session_summaries.return_value = [
            {"ts": "not-a-number", "iso_end": "2026-05-30T20:00:00"}]
        with mock.patch.object(self.bc, "pattern_memory", pm):
            ts = self.bc._last_session_end_ts()
        self.assertEqual(ts, time.mktime(time.strptime("2026-05-30T20:00:00",
                                                       "%Y-%m-%dT%H:%M:%S")))

    def test_bad_iso_falls_to_legacy_date(self):
        # iso_end unparseable (8794-8795) → legacy `date` at 20:00. (8796-8802)
        pm = mock.MagicMock()
        pm.get_session_summaries.return_value = [
            {"iso_end": "garbage-timestamp", "date": "2026-05-29"}]
        with mock.patch.object(self.bc, "pattern_memory", pm):
            ts = self.bc._last_session_end_ts()
        self.assertEqual(ts, time.mktime(time.strptime("2026-05-29T20:00:00",
                                                       "%Y-%m-%dT%H:%M:%S")))

    def test_bad_date_returns_zero(self):
        # date present but unparseable (8803-8804) → falls off → 0.0.
        pm = mock.MagicMock()
        pm.get_session_summaries.return_value = [{"date": "not-a-date"}]
        with mock.patch.object(self.bc, "pattern_memory", pm):
            self.assertEqual(self.bc._last_session_end_ts(), 0.0)

    def test_empty_ts_string_skips_to_iso(self):
        # ts falsy ("") → the `if ts:` guard skips the numeric block entirely,
        # iso_end then drives the result.
        pm = mock.MagicMock()
        pm.get_session_summaries.return_value = [
            {"ts": "", "iso_start": "2026-05-28T20:00:00"}]
        with mock.patch.object(self.bc, "pattern_memory", pm):
            ts = self.bc._last_session_end_ts()
        self.assertEqual(ts, time.mktime(time.strptime("2026-05-28T20:00:00",
                                                       "%Y-%m-%dT%H:%M:%S")))

    # ---- _last_n_user_commands edge cases ----
    def test_open_error_returns_empty(self):
        # File exists but open() raises → []. (lines 8820-8821)
        with mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("locked")):
            self.assertEqual(self.bc._last_n_user_commands(3), [])

    def test_blank_lines_are_skipped(self):
        # Blank/whitespace-only lines are skipped (line 8826); only the two
        # real entries are returned newest-first.
        data = "\n".join(["", "   ",
                          json.dumps({"text": "alpha"}),
                          json.dumps({"text": "beta"})])
        with mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=data)):
            self.assertEqual(self.bc._last_n_user_commands(5), ["beta", "alpha"])

    def test_entry_without_text_is_skipped(self):
        # JSON line with no 'text' field contributes nothing. (8831-8832 false)
        data = "\n".join([json.dumps({"foo": "bar"}),
                          json.dumps({"text": "kept"})])
        with mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", mock.mock_open(read_data=data)):
            self.assertEqual(self.bc._last_n_user_commands(5), ["kept"])

    # ---- _last_queued_task_line file-open error ----
    def test_last_queued_open_error_returns_empty(self):
        # TODO_FILE exists but read raises → "". (lines 8855-8856)
        with mock.patch.object(self.bc.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", side_effect=OSError("io")):
            self.assertEqual(self.bc._last_queued_task_line(), "")


# ─────────────────────────────────────────────────────────────────────────
#  Apple Music web-player: vision-free play + title-based confirmation
# ─────────────────────────────────────────────────────────────────────────
#
# The streaming auto-play pipeline opens the user's REAL browser via
# webbrowser.open (no selenium/playwright DOM), so playback is triggered by
# keyboard SPACE (music.apple.com's play/pause shortcut) and confirmed by the
# browser TAB TITLE ("<Song> — <Artist>" while playing). Vision is demoted to a
# last-resort fallback. These tests pin that behaviour deterministically,
# patching the capability globals explicitly so they hold on any box.
class _FakeWin:
    """Minimal pygetwindow Window stand-in exposing only .title."""
    def __init__(self, title=""):
        self.title = title


@requires_monolith
class CloseBrowserByHandleTests(MonolithGlobalsTestCase):
    """v1.85.0 finding #9: _close_browser_windows_matching(only_hwnd=...) closes
    ONLY the window JARVIS opened (by native handle), never a title-substring
    match that could be the user's own main browser window."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    class _HWin:
        def __init__(self, hwnd, title):
            self._hWnd = hwnd
            self.title = title
            self.closed = False

        def close(self):
            self.closed = True

    def _install(self, wins):
        gw = mock.Mock(name="pygetwindow")
        gw.getAllWindows.return_value = list(wins)
        p = mock.patch.dict(sys.modules, {"pygetwindow": gw})
        p.start()
        self.addCleanup(p.stop)

    def test_only_hwnd_closes_exact_window_not_title_match(self):
        am = self._HWin(111, "Apple Music - Web Player - Google Chrome")
        spot = self._HWin(222, "Spotify - Web Player: Music - Google Chrome")
        self._install([am, spot])
        # Handle beats title: close hWnd 222 even though the terms say "apple
        # music"; the hWnd-111 Apple Music window is left alone.
        n = self.bc._close_browser_windows_matching(["apple music"], only_hwnd=222)
        self.assertEqual(n, 1)
        self.assertTrue(spot.closed)
        self.assertFalse(am.closed)

    def test_only_hwnd_absent_closes_nothing(self):
        # Stale / never-opened handle → inert (does NOT fall back to title match).
        am = self._HWin(111, "Apple Music - Google Chrome")
        self._install([am])
        n = self.bc._close_browser_windows_matching(["apple music"], only_hwnd=999)
        self.assertEqual(n, 0)
        self.assertFalse(am.closed)

    def test_legacy_substring_still_works_without_hwnd(self):
        am = self._HWin(111, "Apple Music - Google Chrome")
        other = self._HWin(222, "Some Doc - Google Chrome")
        self._install([am, other])
        n = self.bc._close_browser_windows_matching(["apple music"])
        self.assertEqual(n, 1)
        self.assertTrue(am.closed)
        self.assertFalse(other.closed)


@requires_monolith
class AppleMusicTitleParseTests(MonolithGlobalsTestCase):
    """_parse_apple_music_track_title / _clean_browser_title — pure parsing."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_plain_track_title(self):
        self.assertEqual(
            self.bc._parse_apple_music_track_title("Billie Jean — Michael Jackson"),
            "Billie Jean — Michael Jackson")

    def test_strips_browser_chrome_suffix(self):
        self.assertEqual(
            self.bc._parse_apple_music_track_title("Africa — Toto - Google Chrome"),
            "Africa — Toto")

    def test_hyphen_separator_accepted(self):
        self.assertEqual(
            self.bc._parse_apple_music_track_title("Thriller - Michael Jackson"),
            "Thriller - Michael Jackson")

    def test_bare_service_title_is_not_a_track(self):
        # Regression for "Apple Music: Apple Music".
        self.assertIsNone(self.bc._parse_apple_music_track_title("Apple Music"))
        self.assertIsNone(
            self.bc._parse_apple_music_track_title("Apple Music - Google Chrome"))

    def test_search_page_title_rejected(self):
        self.assertIsNone(
            self.bc._parse_apple_music_track_title("Michael Jackson on Apple Music"))
        self.assertIsNone(
            self.bc._parse_apple_music_track_title("Search - Apple Music"))

    def test_non_music_browser_tabs_rejected(self):
        # Regression (2026-07-06, live): now_playing said "Apple Music: tallest
        # building in the world - Google Search" — a Google-search Chrome tab
        # reached the parser (browser marker present) and its bare " - " title
        # parsed as a bogus track. Any hyphenated tab ending in a known site/app
        # name is not an Apple Music now-playing title.
        for bogus in (
            "tallest building in the world - Google Search",
            "how to center a div - Stack Overflow",
            "anthropics/anthropic-sdk-python - GitHub",
            "Inbox (5) - Gmail",
            "some video - YouTube",
            "bobert_companion.py - Visual Studio Code",
        ):
            self.assertIsNone(
                self.bc._parse_apple_music_track_title(bogus),
                msg=f"{bogus!r} should not parse as a track")
        # A genuine em-dash track still parses even with a browser suffix.
        self.assertEqual(
            self.bc._parse_apple_music_track_title("Billie Jean — Michael Jackson - Google Chrome"),
            "Billie Jean — Michael Jackson")

    def test_web_player_landing_title_rejected(self):
        # Regression (2026-06-04, live screenshot): the web-player idle tab
        # title "Apple Music - Web Player" contains " - " and so was wrongly
        # read as a "<Song> - <Artist>" track, making verify_first skip the
        # real play step -> nothing played. Both the bare form and the real
        # title (which carries a leading U+200E bidi mark + an NBSP) must be
        # rejected, with and without the browser-chrome suffix.
        for raw in (
            "Apple Music - Web Player",
            "‎Apple Music - Web Player",          # as the OS reports it
            "‎Apple Music - Web Player - Google Chrome",
            "Apple Music — Web Player",                # em-dash variant
        ):
            self.assertIsNone(
                self.bc._parse_apple_music_track_title(raw),
                f"should reject landing title: {raw!r}")

    def test_bidi_mark_stripped_from_real_track(self):
        # The same leading bidi mark / NBSP must NOT block a genuine track.
        self.assertEqual(
            self.bc._parse_apple_music_track_title(
                "‎Billie Jean — Michael Jackson"),
            "Billie Jean — Michael Jackson")

    def test_track_and_album_detail_page_titles_rejected(self):
        # Regression (2026-06-04, live): the iTunes-resolved play path lands on
        # a track/album DETAIL page whose title is "<Song> - Song by <Artist> -
        # Apple Music" — a PAGE title, not a now-playing track. verify_first
        # must not treat it as one and skip the real play step.
        for raw in (
            "Billie Jean - Song by Michael Jackson - Apple Music",
            "Billie Jean - Song by Michael Jackson - Apple Music - Google Chrome",
            "Thriller by Michael Jackson - Apple Music",
        ):
            self.assertIsNone(
                self.bc._parse_apple_music_track_title(raw),
                f"should reject detail-page title: {raw!r}")

    def test_empty_and_too_short(self):
        self.assertIsNone(self.bc._parse_apple_music_track_title(""))
        self.assertIsNone(self.bc._parse_apple_music_track_title("a"))


@requires_monolith
class AppleMusicTitleNowPlayingTests(MonolithGlobalsTestCase):
    """_apple_music_title_now_playing — scan window titles for a live track."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _install_pgw(self, windows):
        gw = mock.Mock(name="pygetwindow")
        gw.getAllWindows.return_value = list(windows)
        p = mock.patch.dict(sys.modules, {"pygetwindow": gw})
        p.start()
        self.addCleanup(p.stop)
        return gw

    def test_returns_first_playing_track(self):
        self._install_pgw([
            _FakeWin("Some Editor"),
            _FakeWin("Billie Jean — Michael Jackson - Google Chrome"),
        ])
        self.assertEqual(self.bc._apple_music_title_now_playing(),
                         "Billie Jean — Michael Jackson")

    def test_non_browser_hyphenated_titles_rejected(self):
        # v1.84.0: a hyphenated NON-browser window title must not be mistaken for
        # a now-playing track. Previously any "X - Y" top-level title passed the
        # loose parser and false-confirmed playback, so verify_first skipped the
        # real play step.
        self._install_pgw([
            _FakeWin("bobert_companion.py - Visual Studio Code"),
            _FakeWin("Inbox (5) - user@gmail.com"),
            _FakeWin("Documents - File Explorer"),
            _FakeWin("Slack - general - Acme"),
        ])
        self.assertIsNone(self.bc._apple_music_title_now_playing())

    def test_playing_title_without_browser_marker_not_trusted(self):
        # A bare "<Song> — <Artist>" with no browser suffix isn't a browser tab,
        # so it is no longer trusted as playback proof.
        self._install_pgw([_FakeWin("Billie Jean — Michael Jackson")])
        self.assertIsNone(self.bc._apple_music_title_now_playing())

    def test_none_when_only_idle_titles(self):
        self._install_pgw([_FakeWin("Apple Music"), _FakeWin("New Tab - Google Chrome")])
        self.assertIsNone(self.bc._apple_music_title_now_playing())

    def test_loaded_track_from_page_title(self):
        # now_playing reporter: the web player's PAGE title names the loaded
        # track even though the strict confirm parser rejects it (2026-06-04).
        self._install_pgw([
            _FakeWin("Some Editor"),
            _FakeWin("Billie Jean - Song by Michael Jackson - Apple Music - Google Chrome"),
        ])
        self.assertEqual(self.bc._apple_music_loaded_track_from_title(),
                         "Billie Jean by Michael Jackson")

    def test_loaded_track_none_without_apple_music_tab(self):
        self._install_pgw([_FakeWin("Some Editor"), _FakeWin("New Tab - Google Chrome")])
        self.assertIsNone(self.bc._apple_music_loaded_track_from_title())

    def test_pygetwindow_absent_returns_none(self):
        # Simulate import failure.
        p = mock.patch.dict(sys.modules, {"pygetwindow": None})
        p.start()
        self.addCleanup(p.stop)
        self.assertIsNone(self.bc._apple_music_title_now_playing())

    def test_getallwindows_raises_returns_none(self):
        gw = mock.Mock(name="pygetwindow")
        gw.getAllWindows.side_effect = RuntimeError("boom")
        p = mock.patch.dict(sys.modules, {"pygetwindow": gw})
        p.start()
        self.addCleanup(p.stop)
        self.assertIsNone(self.bc._apple_music_title_now_playing())


@requires_monolith
class FindMusicWindowTests(MonolithGlobalsTestCase):
    """_find_music_window — must prefer the scriptable browser web-player tab
    over the UWP tray app of the same name. Regression (2026-06-04, live):
    APPLE_MUSIC_KEEP_OPEN parks an "Apple Music" UWP window in the tray, and
    the `space` play strategy focused IT instead of the Chrome web player, so
    Space went nowhere and nothing played."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    # The real Chrome tab title as Windows reports it: a leading U+200E bidi
    # mark and an NBSP between "Apple" and "Music".
    WEB_TITLE = "‎Apple\xa0Music - Web Player - Google Chrome"

    def _install_pgw(self, windows):
        gw = mock.Mock(name="pygetwindow")
        gw.getAllWindows.return_value = list(windows)
        p = mock.patch.dict(sys.modules, {"pygetwindow": gw})
        p.start()
        self.addCleanup(p.stop)

    def test_prefers_browser_web_player_over_tray_app(self):
        tray = _FakeWin("Apple Music")          # UWP keeper app
        web = _FakeWin(self.WEB_TITLE)
        # Tray listed FIRST so a naive first-match would wrongly return it.
        self._install_pgw([tray, web])
        self.assertIs(self.bc._find_music_window(), web)

    def test_nbsp_titled_web_player_is_matched(self):
        # The NBSP between Apple and Music must not dodge the "apple music"
        # hint once the title is normalized.
        web = _FakeWin(self.WEB_TITLE)
        self._install_pgw([_FakeWin("Some Editor"), web])
        self.assertIs(self.bc._find_music_window(), web)

    def test_falls_back_to_tray_app_when_no_browser(self):
        tray = _FakeWin("Apple Music")
        self._install_pgw([_FakeWin("Notepad"), tray])
        self.assertIs(self.bc._find_music_window(), tray)

    def test_none_when_no_music_window(self):
        self._install_pgw([_FakeWin("Notepad"), _FakeWin("Some Editor")])
        self.assertIsNone(self.bc._find_music_window())


@requires_monolith
class StreamingTitleConfirmTests(MonolithGlobalsTestCase):
    """_streaming_title_confirms_playback — deterministic title gate."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_unsupported_service_returns_false(self):
        ok, detail = self.bc._streaming_title_confirms_playback({})  # no title_confirm
        self.assertFalse(ok)
        self.assertIn("not supported", detail)

    def test_track_present_confirms(self):
        cfg = {"title_confirm": True}
        with mock.patch.object(self.bc, "_apple_music_title_now_playing",
                               return_value="Africa — Toto"):
            ok, detail = self.bc._streaming_title_confirms_playback(cfg)
        self.assertTrue(ok)
        self.assertIn("Africa — Toto", detail)

    def test_no_track_does_not_confirm(self):
        cfg = {"title_confirm": True}
        with mock.patch.object(self.bc, "_apple_music_title_now_playing",
                               return_value=None):
            ok, _ = self.bc._streaming_title_confirms_playback(cfg)
        self.assertFalse(ok)


@requires_monolith
class StreamingConfirmPlaybackTests(MonolithGlobalsTestCase):
    """_streaming_confirm_playback — title first, vision only as fallback."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_title_hit_skips_vision(self):
        cfg = {"title_confirm": True}
        with mock.patch.object(self.bc, "_apple_music_title_now_playing",
                               return_value="Billie Jean — Michael Jackson"), \
             mock.patch.object(self.bc, "_streaming_verify_playback") as vis:
            ok, detail = self.bc._streaming_confirm_playback(cfg, "q?")
        self.assertTrue(ok)
        self.assertIn("now-playing track", detail)
        vis.assert_not_called()          # vision NOT consulted when title confirms

    def test_title_miss_uses_vision_when_available(self):
        cfg = {"title_confirm": True}
        with mock.patch.object(self.bc, "_apple_music_title_now_playing",
                               return_value=None), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_streaming_verify_playback",
                               return_value=(True, "YES playing")) as vis:
            ok, detail = self.bc._streaming_confirm_playback(cfg, "q?")
        self.assertTrue(ok)
        self.assertIn("vision", detail)
        vis.assert_called_once()

    def test_title_miss_vision_off_degrades_gracefully(self):
        # The key fix: no silent vision-timeout when vision is unavailable —
        # an honest negative with a clear reason instead.
        cfg = {"title_confirm": True}
        with mock.patch.object(self.bc, "_apple_music_title_now_playing",
                               return_value=None), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", False), \
             mock.patch.object(self.bc, "_streaming_verify_playback") as vis:
            ok, detail = self.bc._streaming_confirm_playback(cfg, "q?")
        self.assertFalse(ok)
        self.assertIn("vision fallback unavailable", detail)
        vis.assert_not_called()


@requires_monolith
class SpaceStrategyFocusTests(MonolithGlobalsTestCase):
    """_streaming_apply_play_strategy('space') focuses the music window first."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_space_focuses_then_presses(self):
        with mock.patch.object(self.bc, "_focus_music_window",
                               return_value="Africa — Toto - Google Chrome") as fmw, \
             mock.patch.object(self.bc, "ui_press") as up, \
             mock.patch.object(self.bc.time, "sleep"):
            attempted, desc = self.bc._streaming_apply_play_strategy("space", {}, None)
        self.assertTrue(attempted)
        fmw.assert_called_once()
        up.assert_called_once_with("space")
        self.assertIn("space", desc)

    def test_space_still_presses_when_no_window_found(self):
        with mock.patch.object(self.bc, "_focus_music_window", return_value=None), \
             mock.patch.object(self.bc, "ui_press") as up, \
             mock.patch.object(self.bc.time, "sleep"):
            attempted, _ = self.bc._streaming_apply_play_strategy("space", {}, None)
        self.assertTrue(attempted)
        up.assert_called_once_with("space")


@requires_monolith
class AppleMusicStrategyOrderTests(MonolithGlobalsTestCase):
    """The shipped Apple Music config puts the vision-located detail-page
    play_button FIRST, SPACE as the deterministic backup, and enables
    title-based confirmation.

    NOTE (expectation updated): v1.31.0 briefly ordered these SPACE-first, but
    v1.35.0 (#56) reverted to play_button-first after live testing showed SPACE
    is a play/PAUSE toggle that pauses a track an earlier action already
    started ("never use SPACE as a retry"). The resolved iTunes track page
    reliably carries the one large Play button, so play_button leads; on the
    search-page fallback that button is absent (a no-op) and the strategy
    advances to SPACE. Current shipped order: ["play_button","space","play_button"]."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_play_button_is_primary_strategy(self):
        cfg = self.bc._STREAMING_SERVICES["apple_music"]
        self.assertEqual(cfg["play_strategies"][0], "play_button")
        self.assertIn("space", cfg["play_strategies"])  # kept as the backup
        # play_button must come before the first SPACE (vision-confirmed button
        # leads; SPACE is the deterministic fallback if the button isn't there).
        self.assertLess(cfg["play_strategies"].index("play_button"),
                        cfg["play_strategies"].index("space"))

    def test_title_confirm_enabled(self):
        self.assertTrue(self.bc._STREAMING_SERVICES["apple_music"].get("title_confirm"))


@requires_monolith
class AppleMusicAutoPlayNoVisionTests(MonolithGlobalsTestCase):
    """_streaming_auto_play for Apple Music end-to-end with vision OFF —
    keyboard-select + SPACE + title-confirm must still report playback, and
    must NOT call vision (take_screenshot / find_click_target).

    NOTE (expectation updated): v1.35.0 (#56) added resolve_itunes, which makes
    _apple_music_resolve_track hit the live iTunes Search API and, on success,
    switches to the resolved-track-page path (select_method='none',
    highlighted_row double-click, query rewritten to '<track> by <artist>').
    That live call made these tests non-deterministic and bypassed the
    keyboard/SPACE path they were written to exercise. We pin the resolve to
    None so the tests deterministically drive the SEARCH-PAGE fallback — the
    exact vision-free keyboard-select + SPACE + tab-title contract under test."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_plays_without_vision_via_title(self):
        bc = self.bc
        with mock.patch.object(bc, "_apple_music_resolve_track",
                               return_value=None), \
             mock.patch.object(bc, "_open_url_in_browser",
                               return_value="chrome") as oub, \
             mock.patch.object(bc.webbrowser, "open") as wbo, \
             mock.patch.object(bc.time, "sleep"), \
             mock.patch.object(bc, "SCREEN_VISION_ENABLED", False), \
             mock.patch.object(bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(bc, "_streaming_keyboard_select_first_result",
                               return_value=True), \
             mock.patch.object(bc, "_focus_music_window", return_value="win"), \
             mock.patch.object(bc, "ui_press"), \
             mock.patch.object(bc, "_streaming_go_fullscreen"), \
             mock.patch.object(bc, "take_screenshot") as ts, \
             mock.patch.object(bc, "find_click_target") as fct, \
             mock.patch.object(bc, "_apple_music_title_now_playing",
                               return_value="Billie Jean — Michael Jackson"):
            out = bc._streaming_auto_play("apple_music", "Michael Jackson")
        self.assertEqual(out, "playing 'Michael Jackson' on Apple Music")
        # The search URL must be opened through the force-a-real-browser
        # helper, NOT the default handler (which is the UWP app on this box).
        oub.assert_called_once()
        self.assertIn("music.apple.com", oub.call_args.args[0])
        wbo.assert_not_called()
        ts.assert_not_called()           # no vision screenshot
        fct.assert_not_called()          # no vision click-target search

    def test_space_fires_when_enter_did_not_start(self):
        # verify_first reports idle (title None), so the SPACE strategy runs;
        # after SPACE the title shows a track and playback is confirmed.
        bc = self.bc
        titles = iter([None, "Africa — Toto"])

        def _title(*_a, **_k):
            try:
                return next(titles)
            except StopIteration:
                return "Africa — Toto"

        # resolve->None forces the search-page fallback so the play_strategies
        # loop runs; play_button (attempt 1) is a vision no-op with vision OFF,
        # so it advances to SPACE (attempt 2). See class note (v1.35.0).
        with mock.patch.object(bc, "_apple_music_resolve_track",
                               return_value=None), \
             mock.patch.object(bc, "_open_url_in_browser",
                               return_value="chrome"), \
             mock.patch.object(bc.time, "sleep"), \
             mock.patch.object(bc, "SCREEN_VISION_ENABLED", False), \
             mock.patch.object(bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(bc, "_streaming_keyboard_select_first_result",
                               return_value=True), \
             mock.patch.object(bc, "_focus_music_window", return_value="win"), \
             mock.patch.object(bc, "ui_press") as up, \
             mock.patch.object(bc, "_streaming_go_fullscreen"), \
             mock.patch.object(bc, "_apple_music_title_now_playing",
                               side_effect=_title):
            out = bc._streaming_auto_play("apple_music", "Africa by Toto")
        self.assertEqual(out, "playing 'Africa by Toto' on Apple Music")
        up.assert_any_call("space")      # SPACE was the trigger

    def test_no_ui_automation_degrades_clearly(self):
        # The honest message replaces the old silent vision-timeout.
        bc = self.bc
        with mock.patch.object(bc, "_open_url_in_browser",
                               return_value="chrome"), \
             mock.patch.object(bc.time, "sleep"), \
             mock.patch.object(bc, "UI_AUTOMATION_ENABLED", False):
            out = bc._streaming_auto_play("apple_music", "Thriller")
        self.assertIn("UI automation", out)
        self.assertIn("Thriller", out)


# ─────────────────────────────────────────────────────────────────────────
#  Force-a-real-browser opener (fix/apple-music-force-chrome)
# ─────────────────────────────────────────────────────────────────────────
#
# The UWP Apple Music app registered itself as the handler for
# music.apple.com, so a bare webbrowser.open launches the APP, not a browser.
# _open_url_in_browser forces Chrome (then Edge) so the real WEB PLAYER loads.
# Every external boundary (webbrowser, subprocess.Popen, the App-Paths
# registry, os.path.exists, shutil.which) is mocked — no real browser ever
# launches.
@requires_monolith
class OpenUrlInBrowserTests(MonolithGlobalsTestCase):
    """_open_url_in_browser prefers Chrome via webbrowser.get, then chrome.exe,
    then Edge, then the default handler — and NEVER raises."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_prefers_webbrowser_get_chrome(self):
        bc = self.bc
        fake_browser = mock.Mock()
        fake_browser.open.return_value = True
        with mock.patch.object(bc.webbrowser, "get",
                               return_value=fake_browser) as wbget, \
             mock.patch.object(bc.subprocess, "Popen") as popen, \
             mock.patch.object(bc.webbrowser, "open") as wbopen:
            via = bc._open_url_in_browser("https://music.apple.com/us/search?term=x")
        self.assertEqual(via, "chrome:webbrowser")
        wbget.assert_called_once_with("chrome")
        fake_browser.open.assert_called_once()
        popen.assert_not_called()        # didn't need to shell out
        wbopen.assert_not_called()       # didn't fall back to default handler

    def test_media_mode_closes_prior_and_uses_new_window(self):
        # close_matching => media mode: close prior matching windows, SKIP the
        # webbrowser controller, and open a dedicated --new-window so repeats
        # reuse one tab instead of piling up.
        bc = self.bc
        with mock.patch.object(bc, "_close_browser_windows_matching",
                               return_value=2) as closer, \
             mock.patch.object(bc.webbrowser, "get") as wbget, \
             mock.patch.object(bc, "_find_chrome", return_value=r"C:\chrome.exe"), \
             mock.patch.object(bc.subprocess, "Popen") as popen, \
             mock.patch.object(bc.time, "sleep"):
            via = bc._open_url_in_browser("https://music.apple.com/x",
                                          close_matching=["apple music", "web player"])
        self.assertEqual(via, "chrome")
        # Expectation updated for v1.85.0 (#120): _open_url_in_browser now threads
        # a close_hwnd through to _close_browser_windows_matching(..., only_hwnd=...)
        # so it closes ONLY the exact window JARVIS opened, never a title match
        # that could hit the user's own browser. With no close_hwnd passed here,
        # only_hwnd defaults to None (legacy title-substring mode).
        closer.assert_called_once_with(["apple music", "web player"], only_hwnd=None)
        wbget.assert_not_called()         # webbrowser controller skipped in media mode
        popen.assert_called_once()
        self.assertIn("--new-window", popen.call_args.args[0])

    def test_no_close_matching_keeps_webbrowser_path(self):
        # Without close_matching, behaviour is unchanged (webbrowser first).
        bc = self.bc
        fake_browser = mock.Mock(); fake_browser.open.return_value = True
        with mock.patch.object(bc, "_close_browser_windows_matching") as closer, \
             mock.patch.object(bc.webbrowser, "get", return_value=fake_browser):
            via = bc._open_url_in_browser("https://music.apple.com/x")
        self.assertEqual(via, "chrome:webbrowser")
        closer.assert_not_called()

    def test_falls_back_to_chrome_exe_path(self):
        bc = self.bc
        # webbrowser.get("chrome") raises (no registered controller) → locate
        # chrome.exe and Popen it.
        with mock.patch.object(bc.webbrowser, "get",
                               side_effect=KeyError("chrome")), \
             mock.patch.object(bc, "_find_chrome",
                               return_value=r"C:\chrome.exe"), \
             mock.patch.object(bc.subprocess, "Popen") as popen, \
             mock.patch.object(bc.webbrowser, "open") as wbopen:
            via = bc._open_url_in_browser("https://music.apple.com/x")
        self.assertEqual(via, "chrome")
        popen.assert_called_once()
        # --new-window + the URL are passed to chrome.exe.
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], r"C:\chrome.exe")
        self.assertIn("--new-window", argv)
        self.assertIn("https://music.apple.com/x", argv)
        wbopen.assert_not_called()

    def test_falls_back_to_edge_when_no_chrome(self):
        bc = self.bc
        with mock.patch.object(bc.webbrowser, "get",
                               side_effect=KeyError("chrome")), \
             mock.patch.object(bc, "_find_chrome", return_value=None), \
             mock.patch.object(bc, "_find_edge",
                               return_value=r"C:\msedge.exe"), \
             mock.patch.object(bc.subprocess, "Popen") as popen, \
             mock.patch.object(bc.webbrowser, "open") as wbopen:
            via = bc._open_url_in_browser("https://music.apple.com/x")
        self.assertEqual(via, "edge")
        self.assertEqual(popen.call_args.args[0][0], r"C:\msedge.exe")
        wbopen.assert_not_called()

    def test_last_resort_default_handler(self):
        bc = self.bc
        with mock.patch.object(bc.webbrowser, "get",
                               side_effect=KeyError("chrome")), \
             mock.patch.object(bc, "_find_chrome", return_value=None), \
             mock.patch.object(bc, "_find_edge", return_value=None), \
             mock.patch.object(bc.webbrowser, "open",
                               return_value=True) as wbopen:
            via = bc._open_url_in_browser("https://music.apple.com/x")
        self.assertEqual(via, "default")
        wbopen.assert_called_once_with("https://music.apple.com/x")

    def test_prepends_https_scheme(self):
        bc = self.bc
        with mock.patch.object(bc.webbrowser, "get",
                               side_effect=KeyError("chrome")), \
             mock.patch.object(bc, "_find_chrome",
                               return_value=r"C:\chrome.exe"), \
             mock.patch.object(bc.subprocess, "Popen") as popen:
            bc._open_url_in_browser("music.apple.com/x")
        self.assertIn("https://music.apple.com/x", popen.call_args.args[0])

    def test_never_raises_when_everything_fails(self):
        bc = self.bc
        # Popen blows up AND the default handler blows up — still returns,
        # never propagates.
        with mock.patch.object(bc.webbrowser, "get",
                               side_effect=KeyError("chrome")), \
             mock.patch.object(bc, "_find_chrome",
                               return_value=r"C:\chrome.exe"), \
             mock.patch.object(bc.subprocess, "Popen",
                               side_effect=OSError("spawn failed")), \
             mock.patch.object(bc, "_find_edge", return_value=None), \
             mock.patch.object(bc.webbrowser, "open",
                               side_effect=RuntimeError("no handler")):
            via = bc._open_url_in_browser("https://music.apple.com/x")
        self.assertEqual(via, "default")   # reached the last branch, no raise


@requires_monolith
class BrowserExeLocatorTests(MonolithGlobalsTestCase):
    """_exe_from_app_paths / _find_chrome / _find_edge — registry-first
    executable resolution with on-disk + PATH fallbacks."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_app_paths_registry_hit(self):
        bc = self.bc
        fake_winreg = mock.MagicMock()
        fake_winreg.HKEY_CURRENT_USER = 1
        fake_winreg.HKEY_LOCAL_MACHINE = 2
        ctx = mock.MagicMock()
        fake_winreg.OpenKey.return_value.__enter__.return_value = ctx
        fake_winreg.QueryValueEx.return_value = (r"C:\reg\chrome.exe", 1)
        with mock.patch.dict(sys.modules, {"winreg": fake_winreg}), \
             mock.patch.object(bc.os.path, "exists", return_value=True):
            got = bc._exe_from_app_paths("chrome.exe")
        self.assertEqual(got, r"C:\reg\chrome.exe")

    def test_app_paths_absent_returns_none(self):
        bc = self.bc
        fake_winreg = mock.MagicMock()
        fake_winreg.HKEY_CURRENT_USER = 1
        fake_winreg.HKEY_LOCAL_MACHINE = 2
        fake_winreg.OpenKey.side_effect = FileNotFoundError("no key")
        with mock.patch.dict(sys.modules, {"winreg": fake_winreg}):
            self.assertIsNone(bc._exe_from_app_paths("chrome.exe"))

    def test_find_chrome_uses_registry_first(self):
        bc = self.bc
        with mock.patch.object(bc, "_exe_from_app_paths",
                               return_value=r"C:\reg\chrome.exe") as eap:
            self.assertEqual(bc._find_chrome(), r"C:\reg\chrome.exe")
        eap.assert_called_once_with("chrome.exe")

    def test_find_chrome_falls_back_to_known_path(self):
        bc = self.bc
        with mock.patch.object(bc, "_exe_from_app_paths", return_value=None), \
             mock.patch.object(bc.os.path, "exists",
                               side_effect=lambda p: p == bc._CHROME_PATHS[0]), \
             mock.patch.object(bc.shutil, "which", return_value=None):
            self.assertEqual(bc._find_chrome(), bc._CHROME_PATHS[0])

    def test_find_chrome_falls_back_to_path_which(self):
        bc = self.bc
        with mock.patch.object(bc, "_exe_from_app_paths", return_value=None), \
             mock.patch.object(bc.os.path, "exists", return_value=False), \
             mock.patch.object(bc.shutil, "which",
                               return_value=r"C:\path\chrome.exe"):
            self.assertEqual(bc._find_chrome(), r"C:\path\chrome.exe")

    def test_find_edge_resolves_msedge(self):
        bc = self.bc
        with mock.patch.object(bc, "_exe_from_app_paths",
                               return_value=r"C:\reg\msedge.exe") as eap:
            self.assertEqual(bc._find_edge(), r"C:\reg\msedge.exe")
        eap.assert_called_once_with("msedge.exe")


@requires_monolith
class MusicWindowRecognisesChromeWebPlayerTests(MonolithGlobalsTestCase):
    """_find_music_window / _focus_music_window must recognise the Chrome
    window whose tab title carries 'Apple Music' (the forced web player) so the
    SPACE play keypress lands on the right window."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _install_pgw(self, windows):
        gw = mock.Mock(name="pygetwindow")
        gw.getAllWindows.return_value = list(windows)
        p = mock.patch.dict(sys.modules, {"pygetwindow": gw})
        p.start()
        self.addCleanup(p.stop)
        return gw

    def test_finds_chrome_apple_music_tab(self):
        bc = self.bc
        self._install_pgw([
            _FakeWin("Visual Studio Code"),
            _FakeWin("Michael Jackson - Apple Music - Google Chrome"),
        ])
        win = bc._find_music_window()
        self.assertIsNotNone(win)
        self.assertIn("Apple Music", win.title)

    def test_focus_returns_chrome_apple_music_title(self):
        bc = self.bc
        target = _FakeWin("Search - Apple Music - Google Chrome")
        target.activate = mock.Mock()
        self._install_pgw([_FakeWin("Some Other Window"), target])
        title = bc._focus_music_window()
        self.assertEqual(title, "Search - Apple Music - Google Chrome")
        target.activate.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────
#  VRAM-brick hardening (REVIEW_FINDINGS_2 P0-2 / P0-3)
# ──────────────────────────────────────────────────────────────────────────
GB = 1024 * 1024 * 1024


def _fake_requests_with_ps(ps_models, ok=True):
    """A stand-in `requests` whose .get('<base>/api/ps') returns a JSON body of
    {'models': ps_models}. Any other GET returns an empty-models 200. Used to
    drive _ollama_loaded_models / _ollama_big_model_resident deterministically
    without a live Ollama."""
    fake = mock.MagicMock(name="requests")

    def _get(url, *a, **k):
        resp = mock.MagicMock()
        resp.ok = ok
        if url.endswith("/api/ps"):
            resp.json.return_value = {"models": ps_models}
        else:
            resp.json.return_value = {"models": []}
        return resp

    fake.get.side_effect = _get
    return fake


@requires_monolith
class OllamaLoadedModelsTests(MonolithGlobalsTestCase):
    """_ollama_loaded_models + _ollama_big_model_resident — the /api/ps
    residency reads that back the VRAM co-load guard."""

    def test_loaded_models_parses_ps(self):
        bc = self.bc
        fake = _fake_requests_with_ps([{"name": "qwen3:30b", "size_vram": 20 * GB}])
        with mock.patch.object(bc, "requests", fake):
            out = bc._ollama_loaded_models()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "qwen3:30b")

    def test_loaded_models_empty_on_http_error(self):
        bc = self.bc
        fake = _fake_requests_with_ps([], ok=False)
        with mock.patch.object(bc, "requests", fake):
            self.assertEqual(bc._ollama_loaded_models(), [])

    def test_loaded_models_empty_on_exception(self):
        bc = self.bc
        fake = mock.MagicMock()
        fake.get.side_effect = RuntimeError("server down")
        with mock.patch.object(bc, "requests", fake):
            self.assertEqual(bc._ollama_loaded_models(), [])

    def test_big_model_resident_detects_30b(self):
        bc = self.bc
        fake = _fake_requests_with_ps([{"name": "qwen3:30b-a3b", "size_vram": 20 * GB}])
        with mock.patch.object(bc, "requests", fake):
            self.assertEqual(bc._ollama_big_model_resident(), "qwen3:30b-a3b")

    def test_big_model_resident_ignores_small_vlm(self):
        bc = self.bc
        # A 7B VLM at ~7 GB is below the 12 GB 'big' threshold → not flagged.
        fake = _fake_requests_with_ps([{"name": "qwen2.5vl:7b", "size_vram": 7 * GB}])
        with mock.patch.object(bc, "requests", fake):
            self.assertIsNone(bc._ollama_big_model_resident())

    def test_big_model_resident_excludes_named_model(self):
        bc = self.bc
        # Even a 'big' tag is not counted against itself (re-use, not co-load).
        fake = _fake_requests_with_ps([{"name": "qwen3:30b", "size_vram": 20 * GB}])
        with mock.patch.object(bc, "requests", fake):
            self.assertIsNone(bc._ollama_big_model_resident(exclude_model="qwen3:30b"))

    def test_big_model_resident_handles_missing_vram_field(self):
        bc = self.bc
        fake = _fake_requests_with_ps([{"name": "mystery"}])  # no size_vram
        with mock.patch.object(bc, "requests", fake):
            self.assertIsNone(bc._ollama_big_model_resident())


@requires_monolith
class LocalVisionColoadGuardTests(MonolithGlobalsTestCase):
    """_call_local_vision refuses to load the VLM while the 30B brain is
    resident (the actual brick path), and proceeds otherwise / when overridden."""

    def _common_patches(self, bc):
        # Enable local vision and make Ollama look alive + VLM installed so the
        # function reaches the co-load guard / POST.
        return (
            mock.patch.object(bc, "LOCAL_VISION_FALLBACK", True),
            mock.patch.object(bc, "LOCAL_VISION_MODEL", "qwen2.5vl:7b"),
            mock.patch.object(bc, "_ollama_alive", return_value=True),
            mock.patch.object(bc, "_ollama_has_model", return_value=True),
        )

    def test_refuses_when_big_model_resident(self):
        bc = self.bc
        post = mock.MagicMock()
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(bc, "_ollama_big_model_resident",
                               return_value="qwen3:30b"), \
             mock.patch.object(bc, "requests") as req:
            os.environ.pop("JARVIS_ALLOW_VLM_COLOAD", None)
            req.post = post
            ps = self._common_patches(bc)
            with ps[0], ps[1], ps[2], ps[3]:
                out = bc._call_local_vision("what is this?", [b"PNG"])
        # Refused → None, and crucially NO POST was made (no 2nd model load).
        self.assertIsNone(out)
        post.assert_not_called()

    def test_proceeds_when_no_big_model_resident(self):
        bc = self.bc
        # Build a fake POST that returns a normal vision reply.
        resp = mock.MagicMock()
        resp.ok = True
        resp.json.return_value = {"message": {"content": "a login screen"}}
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(bc, "_ollama_big_model_resident", return_value=None), \
             mock.patch.object(bc, "_log_gpu_state"), \
             mock.patch.object(bc, "requests") as req:
            os.environ.pop("JARVIS_ALLOW_VLM_COLOAD", None)
            req.post.return_value = resp
            ps = self._common_patches(bc)
            with ps[0], ps[1], ps[2], ps[3]:
                out = bc._call_local_vision("what is this?", [b"PNG"])
        self.assertEqual(out, "a login screen")
        req.post.assert_called_once()

    def test_override_env_allows_coload(self):
        bc = self.bc
        resp = mock.MagicMock()
        resp.ok = True
        resp.json.return_value = {"message": {"content": "ok"}}
        # With JARVIS_ALLOW_VLM_COLOAD=1 the guard is bypassed: even though a big
        # model is resident, the POST fires. _ollama_big_model_resident must NOT
        # be consulted at all in that case.
        with mock.patch.dict(os.environ, {"JARVIS_ALLOW_VLM_COLOAD": "1"}), \
             mock.patch.object(bc, "_ollama_big_model_resident",
                               side_effect=AssertionError("guard must be skipped")), \
             mock.patch.object(bc, "_log_gpu_state"), \
             mock.patch.object(bc, "requests") as req:
            req.post.return_value = resp
            ps = self._common_patches(bc)
            with ps[0], ps[1], ps[2], ps[3]:
                out = bc._call_local_vision("q", [b"PNG"])
        self.assertEqual(out, "ok")
        req.post.assert_called_once()


@requires_monolith
class EnsureOllamaRunningTests(MonolithGlobalsTestCase):
    """_ensure_ollama_running — self-heal that starts the Ollama server when it's
    down (the local brain). 2026-07-09."""

    def test_noop_when_already_alive(self):
        bc = self.bc
        with mock.patch.object(bc, "_ollama_alive", return_value=True), \
                mock.patch.object(bc.subprocess, "Popen") as popen:
            self.assertTrue(bc._ensure_ollama_running(timeout_sec=1.0))
        popen.assert_not_called()   # never launch when it's already up

    def test_starts_server_when_down_then_comes_up(self):
        bc = self.bc
        # down on the first check, up after we "start" it.
        alive = iter([False, True])
        with mock.patch.object(bc, "_ollama_alive", side_effect=lambda: next(alive)), \
                mock.patch("shutil.which", return_value=r"C:\ollama.exe"), \
                mock.patch.object(bc, "_reap_wedged_ollama") as reap, \
                mock.patch.object(bc.subprocess, "Popen") as popen:
            ok = bc._ensure_ollama_running(timeout_sec=5.0)
        self.assertTrue(ok)
        popen.assert_called_once()
        # launched the serve subcommand
        args = popen.call_args.args[0]
        self.assertEqual(args[-1], "serve")
        # cleared any wedged stack before relaunching
        reap.assert_called_once()

    def test_returns_false_when_exe_missing(self):
        bc = self.bc
        with mock.patch.object(bc, "_ollama_alive", return_value=False), \
                mock.patch("shutil.which", return_value=None), \
                mock.patch("os.path.isfile", return_value=False), \
                mock.patch.object(bc.subprocess, "Popen") as popen:
            self.assertFalse(bc._ensure_ollama_running(timeout_sec=1.0))
        popen.assert_not_called()


@requires_monolith
class OllamaRuntimeSelfHealTests(MonolithGlobalsTestCase):
    """_ollama_selfheal_async — MID-SESSION restart of a dead Ollama (live
    outage 2026-07-10 09:13: llama-server wedged during a model swap and the
    local brain stayed dead until reboot). 2026-07-10."""

    def setUp(self):
        self._saved = dict(self.bc._OLLAMA_HEAL_STATE)
        self.bc._OLLAMA_HEAL_STATE.update({"last": 0.0, "active": False})

    def tearDown(self):
        self.bc._OLLAMA_HEAL_STATE.update(self._saved)

    def _drain(self):
        # the heal runs on a daemon thread; wait for it to finish
        import time as _t
        deadline = _t.time() + 3.0
        while self.bc._OLLAMA_HEAL_STATE["active"] and _t.time() < deadline:
            _t.sleep(0.02)

    def test_fires_ensure_on_daemon_thread(self):
        bc = self.bc
        with mock.patch.object(bc, "_ensure_ollama_running") as ens:
            bc._ollama_selfheal_async()
            self._drain()
        ens.assert_called_once()
        self.assertFalse(bc._OLLAMA_HEAL_STATE["active"])

    def test_throttled_within_cooldown(self):
        bc = self.bc
        with mock.patch.object(bc, "_ensure_ollama_running") as ens:
            bc._ollama_selfheal_async()
            self._drain()
            bc._ollama_selfheal_async()   # inside cooldown — must be a no-op
            self._drain()
        self.assertEqual(ens.call_count, 1)

    def test_never_concurrent(self):
        bc = self.bc
        import threading as _th
        release = _th.Event()
        with mock.patch.object(bc, "_ensure_ollama_running",
                               side_effect=lambda: release.wait(2.0)) as ens:
            bc._OLLAMA_HEAL_STATE["last"] = 0.0
            bc._ollama_selfheal_async()          # starts, blocks in ensure
            bc._OLLAMA_HEAL_STATE["last"] = 0.0  # defeat the time throttle
            bc._ollama_selfheal_async()          # active=True — must not start
            release.set()
            self._drain()
        self.assertEqual(ens.call_count, 1)

    def test_reap_is_windows_only_and_never_raises(self):
        bc = self.bc
        with mock.patch.object(bc.subprocess, "run",
                               side_effect=OSError("boom")), \
                mock.patch.object(bc.time, "sleep"):
            bc._reap_wedged_ollama()   # must swallow the error

    def test_call_local_llm_down_branch_fires_selfheal(self):
        bc = self.bc
        with mock.patch.object(bc, "LOCAL_LLM_FALLBACK", True), \
                mock.patch.object(bc, "_ollama_alive", return_value=False), \
                mock.patch.object(bc, "_ollama_selfheal_async") as heal, \
                mock.patch.object(bc, "_ollama_install_async"):
            out = bc._call_local_llm("sys", [{"role": "user", "content": "hi"}])
        self.assertIsNone(out)
        heal.assert_called_once()


@requires_monolith
class EnsureOllamaSingleModelEnvTests(MonolithGlobalsTestCase):
    """_persist_user_env + _ensure_ollama_single_model_env — the OLLAMA_MAX_
    LOADED_MODELS=1 cap that makes Ollama evict instead of co-load."""

    def _fake_winreg(self, existing=None):
        """A minimal in-memory winreg whose Environment 'key' holds `existing`
        values and records SetValueEx writes into `.writes`."""
        wr = mock.MagicMock(name="winreg")
        wr.HKEY_CURRENT_USER = "HKCU"
        wr.KEY_READ = 1
        wr.KEY_SET_VALUE = 2
        wr.REG_SZ = 1
        wr.FileNotFoundError = FileNotFoundError
        store = dict(existing or {})
        wr.writes = []

        class _Key:
            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False

        wr.OpenKey.return_value = _Key()

        def _query(_key, name):
            if name in store:
                return (store[name], wr.REG_SZ)
            raise FileNotFoundError(name)

        def _set(_key, name, _res, _typ, val):
            store[name] = val
            wr.writes.append((name, val))

        wr.QueryValueEx.side_effect = _query
        wr.SetValueEx.side_effect = _set
        return wr

    def test_persist_writes_when_absent(self):
        bc = self.bc
        wr = self._fake_winreg(existing={})
        with mock.patch.dict(sys.modules, {"winreg": wr}):
            status = bc._persist_user_env("OLLAMA_MAX_LOADED_MODELS", "1")
        self.assertEqual(status, "set")
        self.assertIn(("OLLAMA_MAX_LOADED_MODELS", "1"), wr.writes)

    def test_persist_idempotent_when_already_set(self):
        bc = self.bc
        wr = self._fake_winreg(existing={"OLLAMA_MAX_LOADED_MODELS": "1"})
        with mock.patch.dict(sys.modules, {"winreg": wr}):
            status = bc._persist_user_env("OLLAMA_MAX_LOADED_MODELS", "1")
        self.assertEqual(status, "already")
        self.assertEqual(wr.writes, [])  # no write performed

    def test_persist_overwrites_different_value(self):
        bc = self.bc
        wr = self._fake_winreg(existing={"OLLAMA_MAX_LOADED_MODELS": "3"})
        with mock.patch.dict(sys.modules, {"winreg": wr}):
            status = bc._persist_user_env("OLLAMA_MAX_LOADED_MODELS", "1")
        self.assertEqual(status, "set")
        self.assertIn(("OLLAMA_MAX_LOADED_MODELS", "1"), wr.writes)

    def test_persist_noop_without_winreg(self):
        bc = self.bc
        # Simulate non-Windows: importing winreg raises.
        real_import = __import__

        def _imp(name, *a, **k):
            if name == "winreg":
                raise ModuleNotFoundError("no winreg")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=_imp):
            status = bc._persist_user_env("OLLAMA_MAX_LOADED_MODELS", "1")
        self.assertEqual(status, "noop")

    def test_ensure_sets_process_env_and_persists(self):
        bc = self.bc
        wr = self._fake_winreg(existing={})
        # Clear the staging/test gate so the ensure actually runs, and start
        # from an env without the var so setdefault takes effect.
        with mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(bc, "_ollama_loaded_models", return_value=[]), \
             mock.patch.dict(sys.modules, {"winreg": wr}):
            for k in ("JARVIS_STAGING", "JARVIS_TEST_MODE",
                      "OLLAMA_MAX_LOADED_MODELS"):
                os.environ.pop(k, None)
            bc._ensure_ollama_single_model_env()
            self.assertEqual(os.environ.get("OLLAMA_MAX_LOADED_MODELS"), "1")
        self.assertIn(("OLLAMA_MAX_LOADED_MODELS", "1"), wr.writes)

    def test_ensure_gated_off_in_test_mode(self):
        bc = self.bc
        wr = self._fake_winreg(existing={})
        # With JARVIS_TEST_MODE=1 the ensure must short-circuit: no registry
        # write, no os.environ mutation.
        with mock.patch.dict(os.environ, {"JARVIS_TEST_MODE": "1"}), \
             mock.patch.dict(sys.modules, {"winreg": wr}):
            os.environ.pop("OLLAMA_MAX_LOADED_MODELS", None)
            bc._ensure_ollama_single_model_env()
            self.assertIsNone(os.environ.get("OLLAMA_MAX_LOADED_MODELS"))
        self.assertEqual(wr.writes, [])

    def test_ensure_does_not_clobber_operator_override(self):
        bc = self.bc
        wr = self._fake_winreg(existing={})
        # An operator who deliberately set =2 in the process env keeps it
        # (setdefault never overwrites an existing value).
        with mock.patch.dict(os.environ, {"OLLAMA_MAX_LOADED_MODELS": "2"}), \
             mock.patch.object(bc, "_ollama_loaded_models", return_value=[]), \
             mock.patch.dict(sys.modules, {"winreg": wr}):
            os.environ.pop("JARVIS_STAGING", None)
            os.environ.pop("JARVIS_TEST_MODE", None)
            bc._ensure_ollama_single_model_env()
            self.assertEqual(os.environ.get("OLLAMA_MAX_LOADED_MODELS"), "2")


@requires_monolith
class VisionClickBackendAvailableTests(MonolithGlobalsTestCase):
    """_vision_click_backend_available — the auto-play capability gate must
    accept the LOCAL vision route, not just Claude (the stale Claude-only
    check stranded local-route playback on the search page, 2026-07-10)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_claude_backend_is_enough(self):
        with mock.patch.object(self.bc, "AI_BACKEND", "claude"):
            self.assertTrue(self.bc._vision_click_backend_available())

    def test_local_backend_without_vision_model(self):
        with mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(self.bc, "LOCAL_VISION_MODEL", ""):
            self.assertFalse(self.bc._vision_click_backend_available())

    def test_local_backend_vision_model_off(self):
        with mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(self.bc, "LOCAL_VISION_MODEL", "off"):
            self.assertFalse(self.bc._vision_click_backend_available())

    def test_local_backend_with_fallback_enabled(self):
        with mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(self.bc, "LOCAL_VISION_MODEL", "gemma4:12b"), \
             mock.patch.object(self.bc, "LOCAL_VISION_FALLBACK", True):
            self.assertTrue(self.bc._vision_click_backend_available())

    def test_local_backend_via_model_route(self):
        import core.config as _cc
        with mock.patch.object(self.bc, "AI_BACKEND", "ollama"), \
             mock.patch.object(self.bc, "LOCAL_VISION_MODEL", "gemma4:12b"), \
             mock.patch.object(self.bc, "LOCAL_VISION_FALLBACK", False), \
             mock.patch.object(_cc, "model_route", return_value="local"):
            self.assertTrue(self.bc._vision_click_backend_available())


@requires_monolith
class YoutubeResolveVideoTests(MonolithGlobalsTestCase):
    """_youtube_resolve_video — first ORGANIC videoId from results HTML."""

    # Ad slot first (no videoRenderer inside), then the organic row: the
    # first "videoRenderer" match must be the organic one.
    _HTML = (
        '{"adSlotRenderer":{"promoted":true}}'
        '{"videoRenderer":{"videoId":"dQw4w9WgXcQ","thumbnail":{},'
        '"title":{"runs":[{"text":"Darude - Sandstorm"}]}}}'
        '{"videoRenderer":{"videoId":"aaaaaaaaaaa"}}'
    )

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _resp(self, ok=True, text=""):
        r = mock.Mock()
        r.ok = ok
        r.text = text
        return r

    def test_resolves_first_organic_video(self):
        with mock.patch.object(self.bc.requests, "get",
                               return_value=self._resp(text=self._HTML)):
            out = self.bc._youtube_resolve_video("sandstorm")
        self.assertEqual(out["video_id"], "dQw4w9WgXcQ")
        self.assertEqual(out["url"],
                         "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(out["title"], "Darude - Sandstorm")

    def test_no_video_renderer_returns_none(self):
        with mock.patch.object(self.bc.requests, "get",
                               return_value=self._resp(text="<html>nope</html>")):
            self.assertIsNone(self.bc._youtube_resolve_video("q"))

    def test_http_error_returns_none(self):
        with mock.patch.object(self.bc.requests, "get",
                               return_value=self._resp(ok=False)):
            self.assertIsNone(self.bc._youtube_resolve_video("q"))

    def test_network_exception_returns_none(self):
        with mock.patch.object(self.bc.requests, "get",
                               side_effect=OSError("offline")):
            self.assertIsNone(self.bc._youtube_resolve_video("q"))


@requires_monolith
class AutoPlayYoutubeResolvedPathTests(MonolithGlobalsTestCase):
    """_streaming_auto_play('youtube', …) wiring (review 2026-07-11):
    1. the resolved watch-URL path (select_method 'none') must pass the
       capability gate WITHOUT a vision backend — no click is needed, and
    2. verify_play services with play_hint=None must reach the strict
       verify loop instead of returning an unverified "playing" (the old
       early-return made every youtube verify_* key dead config)."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _drive(self, resolved):
        bc = self.bc
        verify = mock.Mock(return_value="VERIFY-PATH")
        with mock.patch.object(bc, "_youtube_resolve_video",
                               return_value=resolved), \
             mock.patch.object(bc, "_open_url_in_browser",
                               return_value="chrome") as opener, \
             mock.patch.object(bc, "_find_browser_window_matching",
                               return_value=None), \
             mock.patch.object(bc, "_streaming_play_and_verify", verify), \
             mock.patch.object(bc, "_streaming_go_fullscreen"), \
             mock.patch.object(bc.time, "sleep"), \
             mock.patch.object(bc, "SCREEN_VISION_ENABLED", False), \
             mock.patch.object(bc, "UI_AUTOMATION_ENABLED", True):
            out = bc._streaming_auto_play("youtube", "sandstorm")
        return out, verify, opener

    def test_resolved_path_verifies_even_without_vision(self):
        resolved = {"url": "https://www.youtube.com/watch?v=abc123def45",
                    "video_id": "abc123def45", "title": "Sandstorm"}
        out, verify, opener = self._drive(resolved)
        # Gate must NOT bail (no click needed), and the result must come from
        # the strict verify loop — never an assumed "playing".
        self.assertEqual(out, "VERIFY-PATH")
        verify.assert_called_once()
        # The watch URL itself was opened, not the search page.
        opened_url = opener.call_args[0][0]
        self.assertIn("/watch?v=abc123def45", opened_url)

    def test_unresolved_without_vision_reports_gate_honestly(self):
        # Resolver fails → vision-click path → with no vision backend the
        # gate must return the honest capability message, not "playing".
        out, verify, opener = self._drive(None)
        self.assertIn("auto-click needs", out)
        verify.assert_not_called()
        opened_url = opener.call_args[0][0]
        self.assertIn("results?search_query=", opened_url)


@requires_monolith
class HardExitTests(MonolithGlobalsTestCase):
    """_hard_exit / _write_clean_shutdown_flag (2026-07-12): the
    un-deadlockable exit. os._exit → ExitProcess hangs on the loader lock
    when a thread is wedged in a CUDA/driver DLL (the 22h zombie), and it
    skips atexit so the watchdog clean-flag handshake never fired on voice
    shutdowns — the flag must be written EXPLICITLY before terminating."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_clean_writes_flag_then_terminates(self):
        bc = self.bc
        calls = []
        with tempfile.TemporaryDirectory() as d:
            flag = os.path.join(d, "clean_shutdown.flag")
            with mock.patch.object(bc, "_CLEAN_SHUTDOWN_FLAG", flag), \
                 mock.patch.object(bc, "_is_staging", return_value=False), \
                 mock.patch.object(bc, "_terminate_process_now",
                                   side_effect=lambda c: calls.append(c)):
                bc._hard_exit(0, clean=True)
            self.assertTrue(os.path.isfile(flag),
                            "clean exit must leave the watchdog flag")
        self.assertEqual(calls, [0])

    def test_unclean_exit_skips_flag(self):
        bc = self.bc
        calls = []
        with tempfile.TemporaryDirectory() as d:
            flag = os.path.join(d, "clean_shutdown.flag")
            with mock.patch.object(bc, "_CLEAN_SHUTDOWN_FLAG", flag), \
                 mock.patch.object(bc, "_terminate_process_now",
                                   side_effect=lambda c: calls.append(c)):
                bc._hard_exit(3, clean=False)
            self.assertFalse(os.path.exists(flag),
                             "restart/crash paths must NOT leave the flag "
                             "(the watchdog is their backstop)")
        self.assertEqual(calls, [3])

    def test_staging_process_never_touches_prod_flag(self):
        bc = self.bc
        with tempfile.TemporaryDirectory() as d:
            flag = os.path.join(d, "clean_shutdown.flag")
            with mock.patch.object(bc, "_CLEAN_SHUTDOWN_FLAG", flag), \
                 mock.patch.object(bc, "_is_staging", return_value=True), \
                 mock.patch.object(bc, "_terminate_process_now"):
                bc._hard_exit(0, clean=True)
            self.assertFalse(os.path.exists(flag))

    # ── INTENT GATE (2026-07-21) ──────────────────────────────────────────
    # _write_clean_shutdown_flag was registered with atexit and took no
    # argument, so it fired on EVERY normal interpreter exit — including an
    # unhandled exception out of main() and every boot-path sys.exit(1),
    # because Python runs atexit handlers after printing a traceback. A CRASH
    # therefore left the "I meant to stop" flag and tools/jarvis_watchdog.ps1
    # declined to resurrect. Live evidence: JARVIS was down 2026-07-15 →
    # 2026-07-21 with the watchdog installed and enabled.

    def _flagged(self, bc, d, *, mark, force=False):
        flag = os.path.join(d, "clean_shutdown.flag")
        with mock.patch.object(bc, "_CLEAN_SHUTDOWN_FLAG", flag), \
             mock.patch.object(bc, "_is_staging", return_value=False), \
             mock.patch.object(bc, "_intentional_exit", [bool(mark)]):
            bc._write_clean_shutdown_flag(force=force)
        return os.path.isfile(flag)

    def test_atexit_on_a_crash_leaves_no_flag(self):
        """The regression that matters: no declared intent → no flag → the
        watchdog still sees an unintended death and resurrects."""
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(self._flagged(self.bc, d, mark=False))

    def test_atexit_after_declared_intent_writes_flag(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(self._flagged(self.bc, d, mark=True))

    def test_force_bypasses_the_gate(self):
        """_hard_exit(clean=True) IS the declaration of intent."""
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(self._flagged(self.bc, d, mark=False, force=True))

    def test_mark_intentional_exit_sets_the_flag_variable(self):
        bc = self.bc
        with mock.patch.object(bc, "_intentional_exit", [False]) as box:
            bc.mark_intentional_exit()
            self.assertTrue(box[0])

    def test_hard_exit_clean_marks_intent_for_any_later_atexit(self):
        bc = self.bc
        box = [False]
        with tempfile.TemporaryDirectory() as d:
            flag = os.path.join(d, "clean_shutdown.flag")
            with mock.patch.object(bc, "_CLEAN_SHUTDOWN_FLAG", flag), \
                 mock.patch.object(bc, "_is_staging", return_value=False), \
                 mock.patch.object(bc, "_intentional_exit", box), \
                 mock.patch.object(bc, "_terminate_process_now"):
                bc._hard_exit(0, clean=True)
            self.assertTrue(box[0], "clean=True must declare intent")
            self.assertTrue(os.path.isfile(flag))


@requires_monolith
class ProcessCaptureChunkSkipNsTests(MonolithGlobalsTestCase):
    """_process_capture_chunk(skip_ns=) — the idle-silence CPU guard
    (2026-07-12): noisereduce spectral gating costs 1-2s CPU per 2.6s chunk,
    and running it on chunks the raw-RMS VAD was about to discard pinned the
    main loop for minutes while room noise hovered under the threshold.
    skip_ns must drop ONLY the NS stage; AEC/AGC flags pass through."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _run(self, skip_ns):
        bc = self.bc
        import numpy as np
        calls = {}

        class _Proc:
            def process(self, chunk, enable_aec, enable_ns, enable_agc):
                calls.update(aec=enable_aec, ns=enable_ns, agc=enable_agc)
                return chunk

        fake = mock.Mock()
        fake.get_processor.return_value = _Proc()
        with mock.patch.object(bc, "_audio_processor", fake), \
             mock.patch.object(bc, "_audio_master_enabled", [True]), \
             mock.patch.object(bc, "_audio_aec_enabled", [True]), \
             mock.patch.object(bc, "_audio_ns_enabled", [True]), \
             mock.patch.object(bc, "_audio_agc_enabled", [True]):
            bc._process_capture_chunk(np.zeros(160, dtype=np.float32),
                                      16000, skip_ns=skip_ns)
        return calls

    def test_skip_ns_drops_only_ns(self):
        calls = self._run(skip_ns=True)
        self.assertFalse(calls["ns"])
        self.assertTrue(calls["aec"], "AEC must stay on (adaptive state)")
        self.assertTrue(calls["agc"], "AGC must stay on")

    def test_default_keeps_ns(self):
        calls = self._run(skip_ns=False)
        self.assertTrue(calls["ns"])


@requires_monolith
class VisionWedgeDetectorTests(MonolithGlobalsTestCase):
    """The vision-wedge detector (2026-07-14). A wedged MULTIMODAL runner is
    INVISIBLE to _ollama_alive() — the server answers /api/tags and text chat
    in ~1s while every image request hangs. The old self-heal called
    _ensure_ollama_running(), a no-op in that state, so the wedge survived
    until a human killed ollama by hand (twice: 2026-07-12, 2026-07-13).
    Consecutive vision TIMEOUTS now escalate to a full stack reap."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        for st in self.bc._WEDGE_STATE.values():
            st["timeouts"] = 0
            st["last_reap"] = 0.0

    def _thread_runner(self):
        """Run the daemon thread body inline so the reap is observable."""
        def _fake_thread(target=None, name=None, daemon=None):
            t = mock.Mock()
            t.start = target          # calling .start() runs the body inline
            return t
        return mock.patch.object(self.bc.threading, "Thread",
                                 side_effect=_fake_thread)

    def test_single_timeout_does_not_reap(self):
        bc = self.bc
        with mock.patch.object(bc, "_reap_wedged_ollama") as reap, \
             mock.patch.object(bc, "_ensure_ollama_running") as ensure, \
             mock.patch("builtins.print"):
            bc._vision_wedge_note_timeout()
        reap.assert_not_called()
        ensure.assert_not_called()
        self.assertEqual(bc._WEDGE_STATE["vision"]["timeouts"], 1)

    def test_threshold_timeouts_reap_and_restart(self):
        bc = self.bc
        with mock.patch.object(bc, "_reap_wedged_ollama") as reap, \
             mock.patch.object(bc, "_ensure_ollama_running") as ensure, \
             self._thread_runner(), mock.patch("builtins.print"):
            bc._vision_wedge_note_timeout()
            bc._vision_wedge_note_timeout()
        reap.assert_called_once()
        ensure.assert_called_once()
        # counter reset so the next wedge starts clean
        self.assertEqual(bc._WEDGE_STATE["vision"]["timeouts"], 0)

    def test_success_clears_the_counter(self):
        bc = self.bc
        with mock.patch.object(bc, "_reap_wedged_ollama") as reap, \
             mock.patch("builtins.print"):
            bc._vision_wedge_note_timeout()
            bc._vision_wedge_note_ok()          # a good call in between
            bc._vision_wedge_note_timeout()     # this is now only #1 again
        reap.assert_not_called()

    def test_text_path_also_reaps(self):
        # 2026-07-14: the TEXT path — which answers EVERY turn — had NO wedge
        # detector at all. A wedged text runner answers /api/tags in ms while
        # every generate hangs, so the brain stayed "not responding" until a
        # human killed ollama.exe.
        bc = self.bc
        with mock.patch.object(bc, "_reap_wedged_ollama") as reap,              mock.patch.object(bc, "_ensure_ollama_running") as ensure,              self._thread_runner(), mock.patch("builtins.print"):
            bc._text_wedge_note_timeout()
            bc._text_wedge_note_timeout()
        reap.assert_called_once()
        ensure.assert_called_once()

    def test_text_success_clears_text_counter(self):
        bc = self.bc
        with mock.patch.object(bc, "_reap_wedged_ollama") as reap,              mock.patch("builtins.print"):
            bc._text_wedge_note_timeout()
            bc._text_wedge_note_ok()
            bc._text_wedge_note_timeout()
        reap.assert_not_called()

    def test_paths_are_independent(self):
        # One vision timeout + one text timeout is NOT a wedge on either path.
        bc = self.bc
        with mock.patch.object(bc, "_reap_wedged_ollama") as reap,              mock.patch("builtins.print"):
            bc._vision_wedge_note_timeout()
            bc._text_wedge_note_timeout()
        reap.assert_not_called()

    def test_cooldown_prevents_reap_thrash(self):
        bc = self.bc
        with mock.patch.object(bc, "_reap_wedged_ollama") as reap, \
             mock.patch.object(bc, "_ensure_ollama_running"), \
             self._thread_runner(), mock.patch("builtins.print"):
            bc._vision_wedge_note_timeout()
            bc._vision_wedge_note_timeout()     # reap #1
            bc._vision_wedge_note_timeout()
            bc._vision_wedge_note_timeout()     # inside cooldown → no reap
        reap.assert_called_once()


if __name__ == "__main__":
    unittest.main()
