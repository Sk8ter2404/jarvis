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

    def test_known_monitor_translates_to_absolute(self):
        # Pass1 image 1568x900, full capture fails → uses native size; with a
        # known monitor the result is offset by the monitor origin.
        name = "middle"
        mx, my = self.bc.MONITORS[name][0], self.bc.MONITORS[name][1]
        pil = mock.MagicMock()
        pil.Image.open.return_value = self._fake_pil_image((1568, 900))

        # take_screenshot: first call (pass1) returns bytes, second (full) None.
        shots = [b"png1", None]

        def fake_shot(monitor=None, max_dim=1568):
            return shots.pop(0)

        with mock.patch.dict(sys.modules, {"PIL": pil}), \
             mock.patch.object(self.bc, "take_screenshot", side_effect=fake_shot), \
             mock.patch.object(self.bc, "_query_vision_for_coords", return_value=(100, 100)), \
             mock.patch.object(self.bc, "_native_capture_size", return_value=(1568, 900)):
            result = self.bc.find_click_target("the button", monitor=name)
        # native==pass1 size → scale 1.0 → (100,100) offset by monitor origin.
        self.assertEqual(result, (mx + 100, my + 100))

    def test_two_pass_refine_no_monitor_translates_by_virtual_origin(self):
        # Pass1 1568x900 → full 3000x1800 (bigger ⇒ pass-2 crop runs). Pass1
        # coords (100,100) scale to full (191,200); the 500px crop clamps to
        # (0,0); pass-2 returns (50,50) inside it ⇒ refined (50,50); monitor=None
        # ⇒ translated by the virtual-screen origin (-100,-50) ⇒ (-50, 0).
        class _CropImg:
            def __init__(self, size):
                self.size = size

            def crop(self, box):
                left, top, right, bot = box
                return _CropImg((right - left, bot - top))

            def save(self, buf, format=None):
                buf.write(b"crop")

        pil = mock.MagicMock()
        imgs = [_CropImg((1568, 900)), _CropImg((3000, 1800))]
        pil.Image.open.side_effect = lambda b: imgs.pop(0)
        pil.Image.LANCZOS = 1
        shots = [b"png1", b"fullpng"]
        coords = [(100, 100), (50, 50)]  # pass1, then pass2

        mssmod = mock.MagicMock()
        sct = mock.MagicMock()
        sct.monitors = [{"left": -100, "top": -50, "width": 3000, "height": 1800}]
        mssmod.mss.return_value.__enter__ = lambda s: sct
        mssmod.mss.return_value.__exit__ = lambda *a: False
        del mssmod.MSS

        with mock.patch.dict(sys.modules, {"PIL": pil, "mss": mssmod}), \
             mock.patch.object(self.bc, "take_screenshot",
                               side_effect=lambda monitor=None, max_dim=1568: shots.pop(0)), \
             mock.patch.object(self.bc, "_query_vision_for_coords",
                               side_effect=lambda *a: coords.pop(0)):
            result = self.bc.find_click_target("the button")
        self.assertEqual(result, (-50, 0))


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
        with mock.patch.object(self.bc.webbrowser, "open") as opn:
            out = self.bc._streaming_auto_play("netflix", "   ")
        self.assertEqual(out, "opened Netflix")
        opn.assert_called_once_with(self.bc._STREAMING_SERVICES["netflix"]["home"])

    def test_capabilities_missing_returns_open_only(self):
        with mock.patch.object(self.bc.webbrowser, "open"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", False):
            out = self.bc._streaming_auto_play("netflix", "the matrix")
        self.assertIn("auto-click needs", out)

    def test_youtube_no_play_hint_path(self):
        # YouTube has play_hint=None → after clicking the result it just
        # full-screens and reports playing.
        with mock.patch.object(self.bc.webbrowser, "open"), \
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
        with mock.patch.object(self.bc.webbrowser, "open"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "find_click_target", return_value=None):
            out = self.bc._streaming_auto_play("netflix", "the matrix")
        self.assertIn("couldn't see the", out)

    def test_default_play_click_path(self):
        # Netflix: vision select + separate play button, no verify.
        with mock.patch.object(self.bc.webbrowser, "open"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "find_click_target", return_value=(50, 60)), \
             mock.patch.object(self.bc, "ui_click"), \
             mock.patch.object(self.bc, "_streaming_go_fullscreen"):
            out = self.bc._streaming_auto_play("netflix", "the matrix")
        self.assertEqual(out, "playing 'the matrix' on Netflix")


@requires_monolith
class AppleMusicPlayPlaylistTests(MonolithGlobalsTestCase):
    """_apple_music_play_playlist — Library>Playlists direct navigation."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_capabilities_missing(self):
        with mock.patch.object(self.bc.webbrowser, "open"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", False):
            out = self.bc._apple_music_play_playlist("chill")
        self.assertIn("auto-click needs", out)

    def test_playlist_not_found_after_sidebar_fallback(self):
        with mock.patch.object(self.bc.webbrowser, "open"), \
             mock.patch.object(self.bc.time, "sleep"), \
             mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
             mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
             mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
             mock.patch.object(self.bc, "_streaming_find_with_retry", return_value=None), \
             mock.patch.object(self.bc, "find_click_target", return_value=None):
            out = self.bc._apple_music_play_playlist("chill")
        self.assertIn("couldn't find a playlist named 'chill'", out)

    def test_playlist_found_delegates_to_play_and_verify(self):
        with mock.patch.object(self.bc.webbrowser, "open"), \
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


if __name__ == "__main__":
    unittest.main()
