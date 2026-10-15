"""Holo mode + Kinect depth (Phase 2, 2026-10-15): the shared state, the voice
("open up the hologram with the Kinect"), the status line, the dispatcher route,
the prompt, and the page's static contract for the depth code (pinned URLs, depth
subscription, no camera, no OS input, the wire constants in step with
core/handtrack_depth.py). The page's own math is covered by hud/holo/selftest.js
(/holo?selftest=1). Hermetic: no browser, no server, no Kinect, no camera.
"""
from __future__ import annotations

import os
import re
import unittest
from unittest import mock

from core import config, holo_web, prompt_router as pr, prompts
from core import handtrack_depth as hd
from skills import holo_mode

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_HOLO_DIR = os.path.join(_ROOT, "hud", "holo")


def _read(name):
    with open(os.path.join(_HOLO_DIR, name), encoding="utf-8") as f:
        return f.read()


class _Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


# ── shared state ──────────────────────────────────────────────────────────
class StateTests(unittest.TestCase):
    def setUp(self):
        self.clk = _Clock()
        self.st = holo_web.HoloState(clock=self.clk)

    def test_depth_is_off_by_default_and_travels_in_the_payload(self):
        self.assertFalse(self.st.payload(0)["depth"])
        self.st.set_on(True, mode="camera", depth=True)
        self.assertTrue(self.st.payload(0)["depth"])
        self.st.set_on(True, mode="camera")                   # depth=None leaves it alone
        self.assertTrue(self.st.depth)

    def test_depth_command_and_close(self):
        self.st.set_on(True, mode="camera")
        self.st.push_command("depth", "1")
        self.assertTrue(self.st.depth)
        self.st.push_command("depth", "0")
        self.assertFalse(self.st.depth)
        self.st.push_command("depth", "on")
        self.st.push_command("close")
        self.assertFalse(self.st.depth)
        self.st.set_on(True, depth=True)
        self.st.set_on(False)
        self.assertFalse(self.st.depth)

    def test_kinect_session_lives_exactly_as_long_as_the_hologram(self):
        self.assertFalse(self.st.kinect_session())            # nothing asked
        self.st.set_on(True, mode="camera", depth=False)
        self.assertFalse(self.st.kinect_session())            # webcam hologram
        self.st.set_on(True, mode="camera", depth=True)
        self.assertTrue(self.st.kinect_session())             # opening grace
        self.clk.t += 11
        self.assertFalse(self.st.kinect_session())            # never polled: the window did not open
        self.st.heartbeat()
        self.assertTrue(self.st.kinect_session())             # the page is polling
        self.clk.t += holo_web.ALIVE_S + 1
        self.assertFalse(self.st.kinect_session())            # window closed by hand
        self.st.heartbeat()
        self.assertTrue(self.st.kinect_session())
        self.st.push_command("close")
        self.assertFalse(self.st.kinect_session())            # closed by voice

    def test_the_depth_command_is_known(self):
        self.assertIn("depth", holo_web.COMMANDS)
        self.assertIn("/holo/depth.js", holo_web._ASSETS)


# ── the voice ─────────────────────────────────────────────────────────────
class VoiceTests(unittest.TestCase):
    def setUp(self):
        self.clk = _Clock()
        self.state = holo_web.HoloState(clock=self.clk)
        self.opened = []
        self.kinect_on = [True]
        self._patches = [
            mock.patch.object(holo_web, "STATE", self.state),
            mock.patch.object(holo_web, "list_models", return_value=[]),
            mock.patch.object(holo_mode, "_ensure_server",
                              return_value=(True, "127.0.0.1", 8766, "")),
            mock.patch.object(holo_mode, "_launch_window",
                              side_effect=lambda u: self.opened.append(u) or True),
            mock.patch.object(holo_mode, "_enabled", return_value=True),
            mock.patch.object(holo_mode, "_tracker", side_effect=lambda: self.trk),
            mock.patch.object(holo_mode, "_kinect_enabled",
                              side_effect=lambda: self.kinect_on[0]),
            mock.patch.multiple(config, HAND_TRACK_CAMERA="primary",
                                HOLO_DEPTH_ENABLED=False),
        ]
        self.trk = mock.Mock()
        self.trk.ensure_for_holo.return_value = (True, "started")
        for p in self._patches:
            p.start()
        holo_mode._last_open[0] = 0.0

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _page_up(self):
        self.state.set_on(True)
        self.state.heartbeat()

    def test_with_the_kinect_opens_camera_mode_with_depth(self):
        out = holo_mode.holo_mode_on("kinect")
        self.assertEqual(self.opened, ["http://127.0.0.1:8766/holo?mode=camera&depth=1"])
        self.assertTrue(self.state.depth)
        self.assertEqual(self.state.mode, "camera")
        self.assertTrue(self.state.kinect_session())
        self.assertIn("on the Kinect", out)
        self.assertIn("once it is streaming", out)             # does not promise depth is already on
        self.trk.ensure_for_holo.assert_called_once()

    def test_it_rewrites_no_setting(self):
        holo_mode.holo_mode_on("camera with the kinect")
        self.assertEqual(config.HAND_TRACK_CAMERA, "primary")
        self.assertFalse(config.HOLO_DEPTH_ENABLED)
        holo_mode.holo_mode_off()
        self.assertEqual(config.HAND_TRACK_CAMERA, "primary")
        self.assertFalse(self.state.kinect_session())          # closing ends the Kinect session

    def test_a_switched_off_kinect_is_said_and_the_webcam_is_used(self):
        self.kinect_on[0] = False
        out = holo_mode.holo_mode_on("kinect")
        self.assertEqual(self.opened, ["http://127.0.0.1:8766/holo?mode=camera"])
        self.assertFalse(self.state.depth)
        self.assertIn("Kinect is switched off in my settings", out)
        self.assertIn("webcam", out)
        self.assertTrue(out.startswith("Holo mode up, sir."))

    def test_a_plain_request_stays_the_webcam_one(self):
        holo_mode.holo_mode_on()
        self.assertEqual(self.opened, ["http://127.0.0.1:8766/holo?mode=camera"])
        self.assertFalse(self.state.depth)

    def test_the_hand_menu_never_asks_for_depth(self):
        holo_mode.hand_menu()
        self.assertEqual(self.opened, ["http://127.0.0.1:8766/holo?mode=desktop"])
        self.assertFalse(self.state.depth)

    def test_kinect_while_the_hologram_is_already_up_switches_it(self):
        self._page_up()
        self.state.mode = "camera"
        out = holo_mode.holo_mode_on("the hologram with the kinect")
        self.assertIn("Switched the hologram to the Kinect", out)
        self.assertTrue(self.state.depth)
        self.assertEqual(self.opened, [])
        cmds = [(c["cmd"], c["arg"]) for c in self.state.payload(0)["commands"]]
        self.assertEqual(cmds, [("depth", "1")])
        # ... and when the Kinect is off it says so instead
        self.state.set_on(False)
        self._page_up()
        self.state.mode = "camera"
        self.kinect_on[0] = False
        out = holo_mode.holo_mode_on("kinect")
        self.assertIn("switched off", out)
        self.assertFalse(self.state.depth)

    def test_permanent_settings_make_a_plain_request_the_kinect_one(self):
        with mock.patch.multiple(config, HAND_TRACK_CAMERA="kinect", HOLO_DEPTH_ENABLED=True):
            holo_mode.holo_mode_on()
        self.assertEqual(self.opened, ["http://127.0.0.1:8766/holo?mode=camera&depth=1"])
        self.assertTrue(self.state.depth)

    def test_permanent_settings_do_nothing_when_the_kinect_is_off_or_depth_is_off(self):
        self.kinect_on[0] = False
        with mock.patch.multiple(config, HAND_TRACK_CAMERA="kinect", HOLO_DEPTH_ENABLED=True):
            holo_mode.holo_mode_on()
        self.assertEqual(self.opened, ["http://127.0.0.1:8766/holo?mode=camera"])
        self.state.set_on(False)
        self.opened.clear()
        holo_mode._last_open[0] = 0.0
        self.kinect_on[0] = True
        with mock.patch.multiple(config, HAND_TRACK_CAMERA="kinect", HOLO_DEPTH_ENABLED=False):
            holo_mode.holo_mode_on()
        self.assertEqual(self.opened, ["http://127.0.0.1:8766/holo?mode=camera"])

    def test_mode_words_are_unchanged(self):
        for arg, mode in (("desktop", "desktop"), ("hand menu", "desktop"),
                          ("camera", "camera"), ("", "camera"), ("kinect", "camera")):
            self.assertEqual(holo_mode._norm_mode(arg, "camera"), mode, arg)
        self.assertTrue(holo_mode._wants_kinect("with the Kinect"))
        self.assertFalse(holo_mode._wants_kinect("camera"))

    def test_the_kinect_master_switch_is_never_flipped_here(self):
        with open(os.path.join(_ROOT, "skills", "holo_mode.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("set_enabled", src)
        self.assertNotIn("KINECT_ENABLED =", src)
        self.assertNotIn("setattr", src)


class VoiceRouteTests(unittest.TestCase):
    def setUp(self):
        from core.dispatcher import holo_route
        self.r = holo_route

    def test_with_the_kinect_routes_to_holo_mode_on_kinect(self):
        for t in ("open up the hologram with the Kinect", "Jarvis, open the hologram with the kinect.",
                  "open the hologram using the kinect camera", "turn on the hologram on the kinect",
                  "open up the kinect hologram", "show me the hologram with the kinect and depth",
                  "bring up the hologram with kinect depth"):
            self.assertEqual(self.r(t), "[ACTION: holo_mode_on, kinect]", t)

    def test_plain_and_unrelated_requests_are_unchanged(self):
        self.assertEqual(self.r("open up the hologram"), "[ACTION: holo_mode_on]")
        self.assertEqual(self.r("close the hologram"), "[ACTION: holo_mode_off]")
        self.assertEqual(self.r("hand menu"), "[ACTION: hand_menu]")
        for t in ("open my files with the kinect", "what does the kinect see",
                  "turn on the kinect", "close the hologram with the kinect",
                  "open the hologram with the kinect and play music"):
            self.assertIsNone(self.r(t, holo_up=True), t)


class PromptTests(unittest.TestCase):
    FULL = prompts.PC_CONTROL_PROMPT

    def test_the_brain_is_taught_the_kinect_argument(self):
        _core, sections = pr.split_pc_control(self.FULL)
        body = dict(sections)
        holo = body["HOLO MODE"]
        self.assertIn("holo_mode_on [, desktop|camera|kinect]", holo)
        self.assertIn("[ACTION: holo_mode_on, kinect]", holo)
        self.assertIn("ONLY when he names the Kinect", holo)
        self.assertIn("hand_tracking_status", holo)
        hand = body["HAND TRACKING"]
        self.assertIn("whether its DEPTH is really flowing", hand)

    def test_the_sections_load_for_the_new_phrases(self):
        _core, sections = pr.split_pc_control(self.FULL)
        for q, section, action in (
                ("open up the hologram with the Kinect", "HOLO MODE", "holo_mode_on"),
                ("use the kinect for the hologram", "HOLO MODE", "holo_mode_on"),
                ("is the depth working", "HAND TRACKING", "hand_tracking_status"),
                ("is the kinect depth on", "HAND TRACKING", "hand_tracking_status")):
            with self.subTest(q=q):
                self.assertIn(section, pr.select_sections(q, sections)[0])
                self.assertIn(action, pr.turn_pc_block(q, self.FULL))


# ── the page ──────────────────────────────────────────────────────────────
class PageContractTests(unittest.TestCase):
    PAGE_FILES = ("holo.html", "holo.js", "gestures.js", "depth.js", "reactor.js",
                  "models.js", "fakehands.js", "selftest.js")

    def test_depth_module_is_served_like_the_others(self):
        self.assertEqual(holo_web._ASSETS["/holo/depth.js"][0], "depth.js")
        body, ctype = holo_web.asset_for("/holo/depth.js")
        self.assertTrue(ctype.startswith("text/javascript"))
        self.assertGreater(len(body), 500)

    def test_still_only_the_pinned_cdn_and_no_new_host(self):
        for f in self.PAGE_FILES:
            for u in re.findall(r"https?://[^\s\"'`)]+", _read(f)):
                if u.startswith("http://www.w3.org"):
                    continue
                self.assertTrue(u.startswith("https://cdn.jsdelivr.net/npm/three@0.160.0/"), (f, u))
        self.assertEqual(set(re.findall(r"three@[\d.]+", _read("holo.js"))), {"three@0.160.0"})
        csp = holo_web.HOLO_CSP
        self.assertEqual(sorted(set(re.findall(r"(?:https?|wss?)://[\w.\[\]:-]+", csp))),
                         ["https://cdn.jsdelivr.net", "ws://127.0.0.1:8793", "ws://localhost:8793"])

    def test_the_page_subscribes_to_depth_on_the_same_socket(self):
        js = _read("holo.js")
        self.assertIn('type: "subscribe"', js)
        self.assertIn('depth: wantDepth && mode === "camera"', js)
        self.assertIn('from "/holo/depth.js"', js)
        self.assertIn('ws.binaryType = "arraybuffer"', js)
        self.assertIn("isDepthMessage(ev.data)", js)
        self.assertIn('qs.get("depth") === "1"', js)
        self.assertIn('case "depth":', js)                   # the voice command

    def test_the_page_never_opens_a_camera_and_sends_no_os_input(self):
        for f in self.PAGE_FILES:
            text = _read(f)
            for banned in ("getUserMedia", "mediaDevices", "tasks-vision", "HandLandmarker",
                           "pyautogui", "SetCursorPos", "SendInput", "mouse_event",
                           "keybd_event", "requestPointerLock", "navigator.hid",
                           "navigator.serial", "navigator.usb"):
                self.assertNotIn(banned, text, (f, banned))

    def test_depth_module_is_pure(self):
        code = re.sub(r"//[^\n]*", "", _read("depth.js"))
        for banned in ("document.", "window.", "fetch(", "setTimeout", "setInterval",
                       "WebSocket", "THREE", "import ", "performance.", "Date.now",
                       "Math.random", "requestAnimationFrame"):
            self.assertNotIn(banned, code, banned)

    def test_gesture_module_is_still_pure_with_the_push_pull_hooks(self):
        code = re.sub(r"//[^\n]*", "", _read("gestures.js"))
        for banned in ("document.", "window.", "fetch(", "setTimeout", "WebSocket",
                       "THREE", "import "):
            self.assertNotIn(banned, code, banned)
        self.assertIn("sc.beginPushPull", code)
        self.assertIn("scene.endPushPull", code)

    def test_wire_constants_match_the_python_side(self):
        js = _read("depth.js")
        magic = [int(x, 16) for x in re.search(r"MAGIC: \[([^\]]+)\]", js).group(1).split(",")]
        self.assertEqual(bytes(magic), hd.WIRE_MAGIC)
        self.assertEqual(int(re.search(r"HEADER: (\d+)", js).group(1)), hd.WIRE_HEADER)
        self.assertEqual(int(re.search(r"VERSION: (\d+)", js).group(1)), hd.WIRE_VERSION)
        # the DataView offsets used by parseDepthMessage (struct "<4sBBHHHQHH")
        for needle in ("getUint8(4)", "getUint8(5)", "getUint16(6, true)", "getUint16(8, true)",
                       "getBigUint64(12, true)", "getUint16(20, true)", "getUint16(22, true)"):
            self.assertIn(needle, js, needle)
        import struct
        self.assertEqual(struct.calcsize(hd._WIRE_FMT), hd.WIRE_HEADER)
        self.assertEqual(struct.calcsize("<4sBBHHH"), 12)      # seq starts at 12
        self.assertEqual(struct.calcsize("<4sBBHHHQ"), 20)     # near_mm at 20, far_mm at 22
        # the near / far the tracker sends are the ones the page assumes
        self.assertEqual((hd.NEAR_MM, hd.FAR_MM), (500, 4500))

    def test_occlusion_and_push_pull_are_wired(self):
        js = _read("holo.js")
        self.assertIn("occlusionMask(mm, w, h, holoZ, maskBuf)", js)
        self.assertIn("CustomBlending", js)
        self.assertIn("renderer.autoClear = false", js)
        self.assertIn("drawOcclusion();", js)
        self.assertIn("sizeAtDistance(holoZRef, holoZ)", js)
        self.assertIn("beginPushPull", js)
        self.assertIn("pp.update(z, dt)", js)
        # honest fallback: nothing depth-related runs unless depthOn()
        self.assertIn("if (!depthOn() || !scene.reactorVisible) return;", js)
        self.assertIn("DEPTH: WAITING FOR THE KINECT", js)
        self.assertIn('id="chip-depth"', _read("holo.html"))

    def test_selftest_covers_occlusion_and_push_pull(self):
        st = _read("selftest.js")
        self.assertIn('from "./depth.js"', st)
        for name in ("occlusion: only what is nearer than the hologram hides it",
                     "occlusion: the margin and single-pixel noise",
                     "push/pull: a fist moving toward the sensor",
                     "push/pull: jitter inside the dead band",
                     "router: a fist with depth begins, drives and ends push/pull",
                     "depth wire message parses"):
            self.assertIn(name, st, name)

    def test_fake_depth_exists_so_the_page_can_be_shown_without_a_kinect(self):
        fk = _read("fakehands.js")
        self.assertIn("depthPlane(now)", fk)
        self.assertIn("this.depth", fk)
        self.assertIn("fake=1&depth=1", _read("holo.js"))

    # Phase 3 (the Quest page) has its own files and its own contract: tests/test_holo_xr.py.
    XR_FILES = ("xr.html", "xr.js", "xrgestures.js", "xrfake.js")

    def test_every_served_file_is_still_a_listed_page_file(self):
        served = {f for f, _ in holo_web._ASSETS.values()}
        self.assertEqual(served, set(self.PAGE_FILES) | set(self.XR_FILES))


class StatusTests(unittest.TestCase):
    """hand_tracking_status says whether the Kinect's depth is really flowing."""

    def setUp(self):
        from tests.test_hand_tracking_skill import FakeLauncher, ht
        self.ht = ht
        self.la = FakeLauncher()
        self.la._running = self.la._wanted = True
        self.feed = {"selected": True, "depth_wanted": True, "reason": ""}
        self.hb = {"age_s": 0.5, "attached": True, "hands": 1, "fps": 14.0}
        self.la.hb = self.hb
        for p in (mock.patch.object(ht, "_launcher", lambda: self.la),
                  mock.patch.object(ht, "_kinect_feed", lambda: self.feed),
                  mock.patch.object(hd, "holo_session", return_value=False),
                  mock.patch.multiple(config, HAND_TRACKING_ENABLED=False,
                                      HAND_MENU_ENABLED=False, HAND_TRACK_CAMERA="kinect",
                                      HOLO_DEPTH_ENABLED=True)):
            p.start()
            self.addCleanup(p.stop)

    def test_kinect_with_depth_flowing(self):
        self.hb["depth"] = True
        out = self.ht._hand_tracking_status()
        self.assertIn("from the Kinect camera, with depth", out)

    def test_kinect_but_depth_not_flowing_is_said_plainly(self):
        self.hb["depth"] = False
        out = self.ht._hand_tracking_status()
        self.assertIn("from the Kinect camera, but depth isn't flowing yet", out)
        self.assertNotIn("with depth", out)

    def test_kinect_colour_only_when_depth_is_off(self):
        self.feed["depth_wanted"] = False
        self.hb["depth"] = False
        self.assertIn("from the Kinect camera, colour only (depth is off)",
                      self.ht._hand_tracking_status())

    def test_kinect_not_open_is_named(self):
        self.hb["attached"] = False
        self.feed["reason"] = "kinect not open"
        out = self.ht._hand_tracking_status()
        self.assertIn("isn't getting frames from the Kinect camera", out)
        self.assertIn("the Kinect isn't open", out)

    def test_a_webcam_says_nothing_about_depth(self):
        with mock.patch.multiple(config, HAND_TRACK_CAMERA="primary"):
            out = self.ht._hand_tracking_status()
        self.assertIn("from the primary camera", out)
        self.assertNotIn("depth", out)

    def test_a_kinect_hologram_is_spoken_as_the_kinect_even_with_a_webcam_setting(self):
        self.hb["depth"] = True
        with mock.patch.multiple(config, HAND_TRACK_CAMERA="primary"), \
                mock.patch.object(hd, "holo_session", return_value=True):
            out = self.ht._hand_tracking_status()
        self.assertIn("from the Kinect camera, with depth", out)


if __name__ == "__main__":
    unittest.main()
