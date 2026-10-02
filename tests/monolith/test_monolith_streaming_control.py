"""Replays of the 2026-10-02 16:12-16:14 streaming turns through the REAL
dispatch (parse_and_run_actions, the utterance routes, the follow-up loop,
the verbatim / terminal speech) and the REAL streaming, vision and click
plumbing - with every model reply canned and every real-world edge faked:
pygetwindow is a fake module, the browser open, the screenshot, the vision
answer and the mouse are mocks. Nothing real is opened, closed, captured or
clicked; no LLM, audio or network is touched. Fixtures are paraphrased (no
owner words).

  S1  "close that and open <service> instead" ran only the open;
  S2  the brain opened a guessed hbomax.com/search URL (a 404);
  S4  local vision named the page's monitor two ways, and a click meant for
      the MIDDLE monitor landed on the LEFT one;
  S5  "Sign In" / "Oops ... isn't working" on the page, and the turn kept
      clicking.

    python -m unittest tests.monolith.test_monolith_streaming_control
"""
from __future__ import annotations

import io
import sys
import types
from unittest import mock

from tests._monolith_harness import requires_monolith
from tests.monolith.test_monolith_claim_validation import _Base

SUFFIX = " - Google Chrome"
SIGN_IN_LINE = ("HBO Max isn't signed in on this browser, sir - sign in once "
                "and I can take it from there.")
# The owner's rig shape, scaled down 10x so the fake screenshots stay tiny:
# left (negative x), middle at the origin, right, and top (negative y).
SMALL_MONS = {
    "left":   (-256, 0, 256, 144),
    "middle": (0, 0, 256, 144),
    "right":  (256, 0, 256, 144),
    "top":    (0, -144, 256, 144),
}


class _Win:
    def __init__(self, title, hwnd, box=(0, 0, 2560, 1400)):
        self.title = title
        self._hWnd = hwnd
        self.left, self.top, self.width, self.height = box
        self.closed = False

    def close(self):
        self.closed = True


def _png(w, h) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


class _StreamingBase(_Base):
    def setUp(self):
        super().setUp()
        from core import opened_ledger
        self.ledger = opened_ledger
        opened_ledger.reset()
        self.addCleanup(opened_ledger.reset)
        self.windows: list = []
        fake = types.SimpleNamespace(getAllWindows=lambda: [
            w for w in self.windows if not w.closed])
        p = mock.patch.dict(sys.modules, {"pygetwindow": fake})
        p.start()
        self.addCleanup(p.stop)
        # Nothing may reach a real screen, model or mouse.
        self.shot = self._p(self.bc, "take_screenshot", return_value=b"PNG")
        self.vision = self._p(self.bc, "ask_vision", return_value="OK")
        self.click = self._p(self.bc, "ui_click")
        self.find = self._p(self.bc, "find_click_target", return_value=None)
        self.fg = self._p(self.bc, "_read_focused_window",
                          return_value=(None, "", None))


# ════════════════════════════════════════════════════════════════════════════
#  S1 - "close that and open X instead"
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class CloseThenOpenReplayTests(_StreamingBase):
    def setUp(self):
        super().setUp()
        self.order: list = []
        self.his = _Win("His Stream" + SUFFIX, 0x100)
        self.mine = _Win("some show - YouTube" + SUFFIX, 0x200)
        self.windows[:] = [self.his, self.mine]
        # What JARVIS itself opened a moment ago (open_on_monitor records it).
        self.ledger.note_opened(
            "open_on_monitor",
            "https://www.youtube.com/results?search_query=some+show",
            hwnd=0x200, kind="window", monitor="middle")
        real_close = self._actions["close_last_opened"]

        def _close(arg=""):
            self.order.append("close")
            return real_close(arg)
        self._actions["close_last_opened"] = _close
        for name in ("open_on_monitor", "open_url"):
            self._actions[name] = (lambda n: lambda arg="": (
                self.order.append(f"{n}:{arg}") or f"opened {arg}"))(name)
        self.close_window = self._stub("close_window", "closed: Spotify")

    def test_the_live_reply_closes_what_jarvis_opened_then_opens(self):
        self._dispatch(
            "Jarvis, close that and open Netflix instead. The show is on there.",
            "[intent:confirmation] Very good, sir. "
            "[ACTION: open_on_monitor, main | netflix.com]")
        self.assertEqual(self.order, ["close", "open_on_monitor:main | netflix.com"])
        self.assertTrue(self.mine.closed)
        self.assertFalse(self.his.closed)
        self.assertIsNone(self.ledger.last_opened())

    def test_a_guessed_close_is_replaced_by_jarvis_own_window(self):
        self._dispatch(
            "close that and open Netflix",
            "Right away, sir. [ACTION: close_window, His Stream] "
            "[ACTION: open_url, netflix.com]")
        self.assertEqual(self.order, ["close", "open_url:netflix.com"])
        self.assertEqual(self.calls["close_window"], [])
        self.assertFalse(self.his.closed)
        self.assertTrue(self.mine.closed)

    def test_a_named_close_is_his_call_and_untouched(self):
        self._dispatch(
            "close Spotify and open Netflix",
            "Right away, sir. [ACTION: close_window, Spotify] "
            "[ACTION: open_url, netflix.com]")
        self.assertEqual(self.calls["close_window"], ["Spotify"])
        self.assertEqual(self.order, ["open_url:netflix.com"])
        self.assertFalse(self.mine.closed)

    def test_a_plain_open_turn_closes_nothing(self):
        self._dispatch("open Netflix",
                       "Of course, sir. [ACTION: open_url, netflix.com]")
        self.assertEqual(self.order, ["open_url:netflix.com"])
        self.assertFalse(self.mine.closed)

    def test_a_close_only_reply_gets_the_open_as_the_dropped_step(self):
        self._dispatch(
            "close that and open Netflix instead",
            "Done, sir. [ACTION: close_last_opened]",
            ["Of course, sir. [ACTION: open_url, netflix.com]"])
        self.assertIn("_dropped_step", self._followup_names(0))
        self.assertEqual(self.order, ["close", "open_url:netflix.com"])


# ════════════════════════════════════════════════════════════════════════════
#  S2 - the verified links, before the brain and inside the streaming flow
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class StreamingRouteReplayTests(_StreamingBase):
    def test_find_and_play_on_a_service_never_reaches_the_brain(self):
        self._stub("play_streaming", "playing 'Some Show' on HBO Max")
        self._dispatch("Jarvis, try again. Find Some Show on HBO Max and "
                       "start playing.", "SHOULD NOT BE USED")
        self.assertEqual(self.calls["play_streaming"], ["max|Some Show"])
        self.bc.get_response_with_animation.assert_not_called()

    def test_find_alone_opens_the_verified_search(self):
        self._stub("streaming_search", "Apple TV's search for Severance is open, sir.")
        self._dispatch("find Severance on Apple TV", "SHOULD NOT BE USED")
        self.assertEqual(self.calls["streaming_search"], ["apple_tv|Severance"])
        self.assertEqual(self.spoken, ["Apple TV's search for Severance is open, sir."])

    def test_the_service_table_is_the_verified_one(self):
        from core import streaming_search as S
        for key in ("max", "netflix", "hulu", "prime_video", "youtube",
                    "apple_tv", "disney_plus"):
            with self.subTest(key=key):
                cfg = self.bc._STREAMING_SERVICES[key]
                self.assertEqual(cfg["home"], S.SERVICES[key].home)
                self.assertEqual(cfg["search_url"], S.SERVICES[key].search)
        self.assertEqual(self.bc._normalize_service("Apple TV+"), "apple_tv")
        self.assertEqual(self.bc._normalize_service("hbo max"), "max")

    def _open(self, fn, *args):
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome") as opn, \
                mock.patch.object(self.bc.time, "sleep"), \
                mock.patch.object(self.bc, "_window_handles_snapshot",
                                  return_value=set()), \
                mock.patch.object(self.bc, "_find_browser_window_matching",
                                  return_value=None), \
                mock.patch.dict(self.bc._JARVIS_MEDIA_WINDOW_HWND, {}, clear=True):
            out = fn(*args)
        return out, opn

    def test_play_on_a_service_without_a_search_link_opens_home_and_says_so(self):
        from core.failure_markers import terminal_failure_text
        out, opn = self._open(self.bc._streaming_auto_play, "disney_plus", "Bluey")
        self.assertEqual(opn.call_args[0][0], "https://www.disneyplus.com")
        self.assertEqual(terminal_failure_text(out),
                         "I don't have a verified search link for Disney+, sir, "
                         "so I've opened its home page - search for Bluey there.")
        self.find.assert_not_called()

    def test_streaming_search_opens_the_verified_url(self):
        out, opn = self._open(self.bc._streaming_open_search, "max", "Some Show")
        self.assertEqual(opn.call_args[0][0],
                         "https://play.hbomax.com/search?q=Some%20Show")
        self.assertEqual(out, "HBO Max's search for Some Show is open, sir.")

    def test_play_on_hbo_max_opens_the_verified_url(self):
        with mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", False):
            _out, opn = self._open(self.bc._streaming_auto_play, "max", "Some Show")
        self.assertEqual(opn.call_args[0][0],
                         "https://play.hbomax.com/search?q=Some%20Show")


# ════════════════════════════════════════════════════════════════════════════
#  S5 - a sign-in wall ends the turn, with no clicks
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class SignInWallReplayTests(_StreamingBase):
    def _play(self, verdict):
        page = _Win("HBO Max | Stream Series and Movies" + SUFFIX, 0x500,
                    (-8, -8, 2576, 1456))
        self.windows[:] = [page]
        self.vision.return_value = verdict
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome"), \
                mock.patch.object(self.bc.time, "sleep"), \
                mock.patch.object(self.bc, "_window_handles_snapshot",
                                  return_value=set()), \
                mock.patch.object(self.bc, "_find_browser_window_matching",
                                  return_value=page), \
                mock.patch.object(self.bc, "_ensure_window_visible_maximized"), \
                mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
                mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
                mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
                mock.patch.object(self.bc, "_vision_click_backend_available",
                                  return_value=True), \
                mock.patch.dict(self.bc._JARVIS_MEDIA_WINDOW_HWND, {}, clear=True):
            return self.bc._streaming_auto_play("max", "Some Show")

    def test_sign_in_wall_is_one_plain_line_and_no_click(self):
        from core.failure_markers import terminal_failure_text
        out = self._play("[local-vision] SIGNIN - a Sign In button, no profile")
        self.assertEqual(terminal_failure_text(out), SIGN_IN_LINE)
        self.find.assert_not_called()
        self.click.assert_not_called()
        # The capture was the pinned player monitor, never the whole desktop.
        self.shot.assert_called_once_with(monitor="middle")
        # ... and "close that" now means this window.
        self.assertEqual(self.ledger.last_opened().hwnd, 0x500)

    def test_an_ok_page_goes_on_to_the_result_click(self):
        out = self._play("OK - search results are shown")
        self.find.assert_called()
        self.assertNotIn("isn't signed in", out)

    def test_a_tab_in_his_window_is_pinned_but_never_the_media_window(self):
        from core.failure_markers import terminal_failure_text
        # The page became a TAB in his existing browser window (no new
        # window): the foreground changed to it, on the LEFT monitor.
        self.fg.side_effect = [
            (0x10, "His Mail" + SUFFIX, (0, 0, 2560, 1400)),
            (0x10, "HBO Max | Stream Series and Movies" + SUFFIX,
             (-2568, -8, 2576, 1456))]
        self.vision.return_value = "SIGNIN"
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome:webbrowser"), \
                mock.patch.object(self.bc.time, "sleep"), \
                mock.patch.object(self.bc, "_window_handles_snapshot",
                                  return_value={0x10}), \
                mock.patch.object(self.bc, "_find_browser_window_matching",
                                  return_value=None), \
                mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
                mock.patch.dict(self.bc._JARVIS_MEDIA_WINDOW_HWND, {}, clear=True):
            out = self.bc._streaming_auto_play("max", "Some Show")
            media = dict(self.bc._JARVIS_MEDIA_WINDOW_HWND)
        self.assertEqual(terminal_failure_text(out), SIGN_IN_LINE)
        self.shot.assert_called_once_with(monitor="left")
        self.assertEqual(media, {})          # his window is never "ours"
        e = self.ledger.last_opened()
        self.assertEqual((e.hwnd, e.kind, e.monitor), (0x10, "tab", "left"))

    def test_an_unchanged_foreground_pins_nothing(self):
        same = (0x10, "HBO Max | Home" + SUFFIX, (0, 0, 2560, 1400))
        self.fg.side_effect = [same, same]
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome:webbrowser"), \
                mock.patch.object(self.bc.time, "sleep"), \
                mock.patch.object(self.bc, "_window_handles_snapshot",
                                  return_value={0x10}), \
                mock.patch.object(self.bc, "_find_browser_window_matching",
                                  return_value=None), \
                mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
                mock.patch.object(self.bc, "UI_AUTOMATION_ENABLED", True), \
                mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
                mock.patch.object(self.bc, "_vision_click_backend_available",
                                  return_value=True):
            self.bc._streaming_auto_play("max", "Some Show")
        self.shot.assert_not_called()          # no pinned page, no wall look
        self.assertIsNone(self.ledger.last_opened())

    def test_the_dispatched_turn_says_the_line_once_and_stops(self):
        page = _Win("HBO Max" + SUFFIX, 0x600, (-8, -8, 2576, 1456))
        self.windows[:] = [page]
        self.vision.return_value = "SIGNIN"
        with mock.patch.object(self.bc, "_open_url_in_browser",
                               return_value="chrome"), \
                mock.patch.object(self.bc.time, "sleep"), \
                mock.patch.object(self.bc, "_window_handles_snapshot",
                                  return_value=set()), \
                mock.patch.object(self.bc, "_find_browser_window_matching",
                                  return_value=page), \
                mock.patch.object(self.bc, "_ensure_window_visible_maximized"), \
                mock.patch.object(self.bc, "SCREEN_VISION_ENABLED", True), \
                mock.patch.dict(self.bc._JARVIS_MEDIA_WINDOW_HWND, {}, clear=True):
            self._dispatch("find Some Show on HBO Max", "SHOULD NOT BE USED",
                           ["[ACTION: click, the first result]"])
        self.assertEqual(self.spoken, [SIGN_IN_LINE])
        self.gfr.assert_not_called()
        self.click.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
#  S4 - click mapping on four monitors, and one label scheme for vision
# ════════════════════════════════════════════════════════════════════════════
@requires_monolith
class ClickMappingTests(_Base):
    """The REAL find_click_target on a 4-monitor layout (left at negative x,
    top at negative y): the screenshots and the vision coordinates are
    faked; the two-pass scaling and the image -> desktop mapping run."""

    VB = {"left": -256, "top": -144, "width": 768, "height": 288}

    def _map(self, monitor, pass1, sizes, pass2=None):
        shots = iter([_png(*sizes[0]), _png(*sizes[1])])
        with mock.patch.object(self.bc, "MONITORS", SMALL_MONS), \
                mock.patch.object(self.bc, "take_screenshot",
                                  side_effect=lambda **k: next(shots)), \
                mock.patch.object(self.bc, "_query_vision_for_coords",
                                  side_effect=[pass1, pass2]), \
                mock.patch.object(self.bc, "_captured_region",
                                  return_value=self.VB if monitor is None else None):
            return self._quiet(self.bc.find_click_target, "the target",
                               monitor=monitor)

    def test_each_monitor_maps_onto_itself(self):
        from core.monitor_geometry import monitor_at
        for name, (x, y, w, h) in SMALL_MONS.items():
            with self.subTest(monitor=name):
                got = self._map(name, (100, 50), [(w, h), (w, h)])
                self.assertEqual(got, (x + 100, y + 50))
                self.assertEqual(monitor_at(*got, SMALL_MONS), name)

    def test_the_left_monitor_gives_negative_x_through_both_passes(self):
        # Pass 1 on a half-size shot -> (128, 72) at full size; the whole
        # 256x144 shot is the crop, and pass 2 refines to (132, 70) in it.
        got = self._map("left", (64, 36), [(128, 72), (256, 144)],
                        pass2=(132, 70))
        self.assertEqual(got, (-256 + 132, 70))

    def test_the_whole_desktop_maps_back_onto_the_right_monitor(self):
        from core.monitor_geometry import monitor_at
        for name, point in (("left", (60, 200)), ("middle", (380, 200)),
                            ("right", (700, 200)), ("top", (380, 60))):
            with self.subTest(monitor=name):
                got = self._map(None, point, [(768, 288), (768, 288)])
                self.assertEqual(monitor_at(*got, SMALL_MONS), name)


@requires_monolith
class VisionLabelTests(_StreamingBase):
    NAMES = ("left", "middle", "right", "top")

    def _images(self):
        return {n: b"PNG" for n in self.NAMES}

    def test_local_route_labels_names_and_the_chat_note(self):
        seen = {}

        def _local(prompt, pngs, max_tokens=600):
            seen["prompt"], seen["n"] = prompt, len(pngs)
            return "The page is on Image #2 (TOP)."
        with mock.patch("core.config.model_route", return_value="local"), \
                mock.patch.object(self.bc, "_call_local_vision", side_effect=_local):
            out = self.bc.ask_vision_multi("What is on the page?", self._images())
        for i, n in enumerate(self.NAMES):
            self.assertIn(f"Image {i + 1} = {n.upper()} monitor", seen["prompt"])
        self.assertIn("Ignore chat and assistant windows", seen["prompt"])
        self.assertEqual(seen["n"], 4)
        # One naming in the answer: image 2 IS the middle monitor.
        self.assertEqual(out, "[local-vision] The page is on the MIDDLE monitor.")

    def test_cloud_route_uses_the_same_labels(self):
        seen = {}

        def _create(*a, **k):
            seen["content"] = k["messages"][0]["content"]
            return object()
        with mock.patch("core.config.model_route", return_value="cloud"), \
                mock.patch.object(self.bc, "AI_BACKEND", "claude"), \
                mock.patch.object(self.bc, "_claude_create", side_effect=_create), \
                mock.patch.object(self.bc, "_claude_reply_text",
                                  return_value="It is on Image #1."):
            out = self.bc.ask_vision_multi("What is on the page?", self._images())
        texts = [b["text"] for b in seen["content"] if b.get("type") == "text"]
        for i, n in enumerate(self.NAMES):
            self.assertIn(f"Image {i + 1} = {n.upper()} monitor", texts)
        kinds = [b["type"] for b in seen["content"]]
        # Each label sits right before its image.
        for i, b in enumerate(seen["content"]):
            if b["type"] == "image":
                self.assertTrue(seen["content"][i - 1]["text"].startswith("Image "))
        self.assertEqual(kinds.count("image"), 4)
        self.assertEqual(out, "It is on the LEFT monitor.")

    def test_a_question_about_the_chat_keeps_the_note_out(self):
        seen = {}
        with mock.patch("core.config.model_route", return_value="local"), \
                mock.patch.object(self.bc, "_call_local_vision",
                                  side_effect=lambda p, i, max_tokens=600:
                                  seen.setdefault("p", p) and "ok"):
            self.bc.ask_vision_multi("What does the Teams chat say?", self._images())
        self.assertNotIn("Ignore chat", seen["p"])
