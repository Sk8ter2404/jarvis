"""Holo XR (2026-10-15, Phase 3): the Quest 3 / WebXR page at /holo/xr - its
routes, CSP and token rules, the page's static contract (pinned libraries, no
camera, no OS input, the secure-address fallback text, the XR session options),
the "open the hologram on the Quest" voice route and reply, and prompt
reachability. The gesture math itself is covered by hud/holo/selftest.js
(/holo?selftest=1 or /holo/xr?selftest=1).

Hermetic: servers bind 127.0.0.1:0, no browser and no headset, no network.
"""
from __future__ import annotations

import json
import os
import re
import unittest
import urllib.request
from unittest import mock

from core import config, holo_web, prompt_router as pr, prompts
from skills import holo_mode
from tests.test_holo_mode import _HoloServer, _get_raw

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_HOLO_DIR = os.path.join(_ROOT, "hud", "holo")

XR_FILES = ("xr.html", "xr.js", "xrgestures.js", "xrfake.js")
THREE_BASE = "https://cdn.jsdelivr.net/npm/three@0.160.0/"
PROFILES_BASE = "https://cdn.jsdelivr.net/npm/@webxr-input-profiles/assets@1.0.20/"


def _read(name):
    with open(os.path.join(_HOLO_DIR, name), encoding="utf-8") as fh:
        return fh.read()


# --- the pure helpers in core/holo_web.py -----------------------------------------
class XrWebHelperTests(unittest.TestCase):
    def test_the_xr_files_are_published_and_exist(self):
        for path, fname in (("/holo/xr", "xr.html"), ("/holo/xr.js", "xr.js"),
                            ("/holo/xrgestures.js", "xrgestures.js"),
                            ("/holo/xrfake.js", "xrfake.js")):
            self.assertEqual(holo_web._ASSETS[path][0], fname)
            body, ctype = holo_web.asset_for(path)
            self.assertGreater(len(body), 500, path)
            self.assertTrue(ctype.startswith(
                "text/html" if path == "/holo/xr" else "text/javascript"), path)

    def test_the_page_gets_the_token_and_the_host_hint_baked_in(self):
        with mock.patch.object(config, "HOLO_XR_HINT_HOST", "jarvis.example.test"):
            body, _ = holo_web.asset_for("/holo/xr", token="tok</script>")
        text = body.decode("utf-8")
        self.assertNotIn("__JARVIS_TOKEN__", text)
        self.assertNotIn("__JARVIS_XR_HOST__", text)
        self.assertIn('window.JARVIS_XR_HOST = "jarvis.example.test"', text)
        self.assertNotIn("tok</script>", text)          # </ is escaped inside the script
        self.assertIn("tok<\\/script>", text)

    def test_no_host_hint_by_default_is_an_empty_string(self):
        self.assertEqual(config.HOLO_XR_HINT_HOST, "")
        body, _ = holo_web.asset_for("/holo/xr")
        self.assertIn('window.JARVIS_XR_HOST = ""', body.decode("utf-8"))

    def test_the_old_page_is_unchanged_by_the_host_slot(self):
        body, _ = holo_web.asset_for("/holo", token="t")
        self.assertNotIn("JARVIS_XR_HOST", body.decode("utf-8"))

    def test_xr_hint_host_accepts_only_a_plain_host_name(self):
        for ok in ("jarvis.example.test", "a-b.c1.example.org:8443", "nas", " nas.lan "):
            self.assertEqual(holo_web.xr_hint_host(ok), ok.strip(), ok)
        for bad in ("", None, "http://x.example", "x.example/path", "a b", "x\"y",
                    "<script>", "x.example:99999999", "-x.example", "x..example",
                    "x.example;rm", "a" * 300):
            self.assertEqual(holo_web.xr_hint_host(bad), "", repr(bad))

    def test_the_csp_is_picked_per_page(self):
        self.assertEqual(holo_web.csp_for("/holo/xr"), holo_web.HOLO_XR_CSP)
        for p in ("/holo", "/holo/holo.js", "/holo/xr.js", "/holo/xrgestures.js"):
            self.assertEqual(holo_web.csp_for(p), holo_web.HOLO_CSP, p)

    def test_the_xr_csp_allows_only_self_and_the_pinned_cdn_host(self):
        csp = holo_web.HOLO_XR_CSP
        self.assertIn("https://cdn.jsdelivr.net", csp)
        for banned in ("ws:", "wss:", "8793", "unsafe-eval", "wasm", "googleapis",
                       "mediastream", "media-src", "*"):
            self.assertNotIn(banned, csp, banned)
        self.assertIn("default-src 'self'", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        # no host other than the one CDN
        hosts = set(re.findall(r"https?://[^\s;]+", csp))
        self.assertEqual(hosts, {"https://cdn.jsdelivr.net"})
        # the tracker page keeps its own (with the tracker socket)
        self.assertIn("ws://127.0.0.1:8793", holo_web.HOLO_CSP)


# --- the routes ---------------------------------------------------------------
class XrRouteTests(_HoloServer):
    def test_the_page_is_served_with_the_xr_csp(self):
        with urllib.request.urlopen(self.base + "/holo/xr", timeout=5) as r:
            body = r.read().decode("utf-8")
            csp = r.headers["Content-Security-Policy"]
            ctype = r.headers["Content-Type"]
        self.assertTrue(ctype.startswith("text/html"))
        self.assertIn("JARVIS Holo XR", body)
        self.assertNotIn("__JARVIS_TOKEN__", body)
        self.assertEqual(csp, holo_web.HOLO_XR_CSP)
        self.assertNotIn("ws://", csp)

    def test_the_tracker_page_still_has_its_own_csp(self):
        with urllib.request.urlopen(self.base + "/holo", timeout=5) as r:
            self.assertEqual(r.headers["Content-Security-Policy"], holo_web.HOLO_CSP)

    def test_a_trailing_slash_and_the_modules_are_served(self):
        for path in ("/holo/xr/", "/holo/xr.js", "/holo/xrgestures.js", "/holo/xrfake.js"):
            code, text = _get_raw(self.base + path)
            self.assertEqual(code, 200, path)
            self.assertGreater(len(text), 500, path)
        with urllib.request.urlopen(self.base + "/holo/xr.js", timeout=5) as r:
            self.assertTrue(r.headers["Content-Type"].startswith("text/javascript"))

    def test_only_published_files_are_served(self):
        for bad in ("/holo/xr.html", "/holo/xr/../core/config.py", "/holo/xr.js.map",
                    "/holo/xrx.js", "/holo/xr/xr.js"):
            code, _ = _get_raw(self.base + bad)
            self.assertEqual(code, 404, bad)

    def test_the_models_still_come_from_the_same_routes(self):
        code, text = _get_raw(self.base + "/api/holo/models")
        self.assertEqual(code, 200)
        self.assertIn("models", json.loads(text))

    def test_the_host_hint_reaches_the_served_page(self):
        with mock.patch.object(config, "HOLO_XR_HINT_HOST", "jarvis.example.test"):
            code, html = _get_raw(self.base + "/holo/xr")
        self.assertEqual(code, 200)
        self.assertIn('"jarvis.example.test"', html)


class XrRouteTokenTests(_HoloServer):
    token = "s3cret-xr-token"

    def test_a_local_bind_serves_the_page_token_free_and_bakes_the_token_in(self):
        code, html = _get_raw(self.base + "/holo/xr")
        self.assertEqual(code, 200)
        self.assertIn('"s3cret-xr-token"', html)

    def test_an_exposed_bind_needs_the_token_for_the_page_and_the_modules(self):
        self.httpd.config["local_bind"] = False          # what a LAN bind sets
        for path in ("/holo/xr", "/holo/xr.js", "/holo/xrgestures.js", "/holo/xrfake.js"):
            code, _ = _get_raw(self.base + path)
            self.assertEqual(code, 401, path)
            code, _ = _get_raw(self.base + path + "?token=" + self.token)
            self.assertEqual(code, 200, path)
        code, _ = _get_raw(self.base + "/holo/xr?token=wrong")
        self.assertEqual(code, 401)
        code, _ = _get_raw(self.base + "/api/holo/models")
        self.assertEqual(code, 401)
        code, _ = _get_raw(self.base + "/api/holo/models",
                           headers={"X-Auth-Token": self.token})
        self.assertEqual(code, 200)


# --- the page's static contract -------------------------------------------------
class XrPageContractTests(unittest.TestCase):
    def test_only_the_pinned_jsdelivr_urls_and_exact_versions(self):
        for f in XR_FILES:
            for u in re.findall(r"https?://[^\s\"'`)]+", _read(f)):
                if u.startswith("http://www.w3.org") or u.startswith("http://localhost"):
                    continue
                self.assertTrue(u.startswith(THREE_BASE) or u.startswith(PROFILES_BASE), (f, u))
                self.assertRegex(u, r"/npm/(@[\w.-]+/)?[\w.-]+@\d+\.\d+\.\d+/", u)
                self.assertNotIn("@latest", u)
        for f in XR_FILES:
            self.assertNotIn("@latest", _read(f), f)
        self.assertEqual(set(re.findall(r"three@[\d.]+", _read("xr.js"))), {"three@0.160.0"})

    def test_it_loads_the_same_three_build_as_the_tracker_page(self):
        build = THREE_BASE + "build/three.module.js"
        self.assertIn(build, _read("xr.js"))
        self.assertIn(build, _read("holo.js"))
        self.assertIn(build, _read("xr.html"))            # the import map

    def test_the_webxr_helpers_come_from_the_same_pinned_path(self):
        js = _read("xr.js")
        for name in ("XRHandModelFactory", "XRControllerModelFactory"):
            self.assertIn(THREE_BASE + "examples/jsm/webxr/" + name + ".js", js)
        # the controller factory's own default is the floating @1.0: it is pinned
        self.assertIn("cf.path = CONTROLLER_PROFILES_PATH", js)
        self.assertIn(PROFILES_BASE + "dist/profiles", js)
        self.assertNotIn("assets@1.0/", js)

    def test_the_session_asks_for_ar_then_vr_with_hand_tracking_as_optional(self):
        js = _read("xr.js")
        self.assertIn('const AR_MODE = "immersive-ar"', js)
        self.assertIn('const VR_MODE = "immersive-vr"', js)
        self.assertIn('optionalFeatures: ["hand-tracking", "local-floor"]', js)
        self.assertIn("navigator.xr.requestSession(mode, SESSION_INIT)", js)
        self.assertIn("isSessionSupported", js)
        self.assertLess(js.index("[AR_MODE, VR_MODE]"), js.index("ENTER VR"))
        self.assertIn("ENTER AR", js + _read("xr.html"))
        self.assertIn("renderer.xr.enabled = true", js)

    def test_passthrough_is_a_transparent_clear_colour(self):
        js = _read("xr.js")
        self.assertIn("alpha: true", js)
        self.assertIn("setClearColor(0x000000, 0)", js)
        # an opaque (VR) headset gets a dark background instead
        self.assertIn('environmentBlendMode === "opaque"', js)

    def test_the_hologram_is_the_same_content_and_the_same_part_logic(self):
        js = _read("xr.js")
        self.assertIn('from "/holo/reactor.js"', js)
        self.assertIn('from "/holo/models.js"', js)
        self.assertIn("/holo/xrgestures.js", js)
        for k in ("loadOwnModel", "fetchModelList", "setExploded", "pullPart", "partByName"):
            self.assertIn(k, js)
        self.assertNotIn("/api/holo/state", js)           # no heartbeat: the PC window's state stays its own

    def test_plain_http_gets_instructions_not_a_dead_button(self):
        js = _read("xr.js")
        self.assertIn("window.isSecureContext", js)
        for text in ("Option A: a Cloudflare-issued certificate for a LAN name",
                     "Option B: Quest developer mode + adb reverse tcp:",
                     "then open http://localhost:", "/holo/xr on the Quest",
                     " - your call."):
            self.assertIn(text, js)
        html = _read("xr.html")
        self.assertIn("The headset needs a secure address.", html)
        self.assertIn('id="optA"', html)
        self.assertIn('id="optB"', html)
        # the secure check comes before the headset is even asked about
        self.assertLess(js.index("const secure ="), js.index("isSessionSupported"))
        self.assertIn("HINT_HOST", js)

    def test_the_lan_name_is_never_in_the_code(self):
        files = [os.path.join(_HOLO_DIR, f) for f in XR_FILES]
        files += [os.path.join(_ROOT, "core", "holo_web.py"),
                  os.path.join(_ROOT, "skills", "holo_mode.py"),
                  os.path.join(_ROOT, "core", "config.py"), os.path.abspath(__file__)]
        needle = "brad" + "enscloud"                     # (built here so this file does not contain it)
        for p in files:
            with open(p, encoding="utf-8") as fh:
                self.assertNotIn(needle, fh.read().lower(), os.path.basename(p))

    def test_page_never_opens_the_camera_a_socket_or_a_hand_model_of_its_own(self):
        for f in XR_FILES:
            text = _read(f)
            for banned in ("getUserMedia", "mediaDevices", "tasks-vision", "HandLandmarker",
                           "FilesetResolver", "new WebSocket", "8793"):
                self.assertNotIn(banned, text, (f, banned))

    def test_the_gesture_and_fake_modules_are_pure(self):
        for f in ("xrgestures.js", "xrfake.js"):
            code = re.sub(r"//[^\n]*", "", _read(f))
            for banned in ("document.", "window.", "fetch(", "setTimeout", "setInterval",
                           "WebSocket", "THREE", "navigator", "https://"):
                self.assertNotIn(banned, code, (f, banned))
            for imp in re.findall(r'from\s+"([^"]+)"', code):
                self.assertTrue(imp.startswith("./"), (f, imp))

    def test_nothing_in_the_xr_page_can_move_the_os_cursor_or_send_input(self):
        paths = [os.path.join(_HOLO_DIR, f) for f in XR_FILES]
        paths += [os.path.join(_ROOT, "skills", "holo_mode.py"),
                  os.path.join(_ROOT, "core", "holo_web.py")]
        for p in paths:
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
            for banned in ("pyautogui", "SetCursorPos", "SendInput", "mouse_event",
                           "keybd_event", "ui_click", "ui_hotkey", "air_control",
                           "kinect_air_mouse"):
                self.assertNotIn(banned, text, (os.path.basename(p), banned))

    def test_the_thresholds_live_in_one_table_and_the_selftest_covers_them(self):
        g = _read("xrgestures.js")
        for k in ("PINCH_ON_M", "PINCH_OFF_M", "APART_SPEED_MPS", "CLAP_SPEED_MPS", "INFO_HOLD_S",
                  "GRAB_MARGIN_M", "PLACE_DIST_M", "PLACE_DROP_M"):
            self.assertIn(k + ":", g)
        st = _read("selftest.js")
        self.assertIn('from "./xrgestures.js"', st)
        self.assertGreaterEqual(len(re.findall(r'\["xr: ', st)), 15)
        for what in ("FAST explodes", "clap", "poke", "hysteresis", "scripted fake hands"):
            self.assertIn(what, st)
        self.assertIn('qs.get("selftest")', _read("xr.js"))

    def test_the_hologram_is_placed_half_a_metre_ahead_at_chest_height(self):
        g = _read("xrgestures.js")
        self.assertIn("PLACE_DIST_M: 0.5", g)
        self.assertIn("PLACE_DROP_M: 0.30", g)
        self.assertIn("placeInFront(head.pos, head.quat)", _read("xr.js"))


# --- the voice ---------------------------------------------------------------
class XrVoiceActionTests(unittest.TestCase):
    def setUp(self):
        self.state = holo_web.HoloState()
        self._patches = [
            mock.patch.object(holo_web, "STATE", self.state),
            mock.patch.object(holo_mode, "_enabled", return_value=True),
            mock.patch.object(holo_mode, "_launch_window",
                              side_effect=AssertionError("no window may open on the PC")),
            mock.patch.object(holo_mode, "_server_up", return_value=(True, "127.0.0.1", 8766)),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_it_is_registered_and_spoken_verbatim(self):
        acts = {}
        holo_mode.register(acts)
        self.assertIn("holo_quest", acts)
        self.assertIn("holo_quest", holo_mode.SPEAK_VERBATIM_ACTIONS)

    def test_without_a_chosen_secure_address_it_says_what_is_needed(self):
        with mock.patch.object(holo_web, "xr_hint_host", return_value=""), \
                mock.patch.object(holo_mode, "_ensure_server",
                                  side_effect=AssertionError("must not start anything")):
            out = holo_mode.holo_quest()
        self.assertIn("secure address", out)
        self.assertIn("certificate", out)
        self.assertIn("adb tunnel", out)
        self.assertIn("http://localhost:8766/holo/xr", out)
        self.assertIn("your call", out.lower())
        self.assertNotIn("https://", out)                  # no address that cannot work yet
        self.assertLess(len(out), 330)                     # short enough to say

    def test_the_port_is_the_running_servers_else_the_setting(self):
        with mock.patch.object(holo_web, "xr_hint_host", return_value=""), \
                mock.patch.object(holo_mode, "_server_up", return_value=(False, "", 0)), \
                mock.patch.object(config, "WEB_INTERFACE_PORT", 9123):
            self.assertIn("http://localhost:9123/holo/xr", holo_mode.holo_quest())

    def test_with_a_lan_name_set_it_gives_the_address(self):
        with mock.patch.object(holo_web, "xr_hint_host", return_value="jarvis.example.test"), \
                mock.patch.object(holo_mode, "_ensure_server",
                                  return_value=(True, "127.0.0.1", 8766, "")) as ens:
            out = holo_mode.holo_quest()
        ens.assert_called_once()
        self.assertEqual(out, "On the Quest, open https://jarvis.example.test/holo/xr, sir.")

    def test_a_web_interface_that_cannot_start_is_said(self):
        with mock.patch.object(holo_web, "xr_hint_host", return_value="jarvis.example.test"), \
                mock.patch.object(holo_mode, "_ensure_server",
                                  return_value=(False, "", 0, "Not while I'm in staging, sir.")):
            self.assertEqual(holo_mode.holo_quest(), "Not while I'm in staging, sir.")

    def test_it_is_off_when_holo_mode_is_off_and_changes_no_state(self):
        with mock.patch.object(holo_mode, "_enabled", return_value=False):
            self.assertEqual(holo_mode.holo_quest(), holo_mode._off_line())
        with mock.patch.object(holo_web, "xr_hint_host", return_value=""):
            holo_mode.holo_quest()
        self.assertFalse(self.state.on)
        self.assertEqual(self.state.payload(0)["commands"], [])


class XrVoiceRouteTests(unittest.TestCase):
    def setUp(self):
        from core.dispatcher import holo_route
        self.r = holo_route

    def test_only_specific_quest_phrasings_route(self):
        for t in ("open the hologram on the Quest", "Jarvis, open up the hologram on my Quest.",
                  "open the hologram on the quest 3", "open hologram on quest",
                  "show me the hologram on the headset", "put up the hologram on my headset",
                  "open the hologram in the headset", "open the hologram in the VR headset",
                  "start the hologram on the Meta Quest", "open the Quest hologram",
                  "turn on the hologram on the quest", "open the hologram on the quest please"):
            self.assertEqual(self.r(t), "[ACTION: holo_quest]", t)

    def test_everything_else_keeps_its_own_route(self):
        self.assertEqual(self.r("open the hologram"), "[ACTION: holo_mode_on]")
        self.assertEqual(self.r("open the hologram with the kinect"),
                         "[ACTION: holo_mode_on, kinect]")
        self.assertEqual(self.r("close the hologram"), "[ACTION: holo_mode_off]")
        self.assertEqual(self.r("hand menu"), "[ACTION: hand_menu]")
        for t in ("open the hologram on the quest and play music", "what is the quest",
                  "start a quest in the game", "show me the headset", "open the quest",
                  "close the hologram on the quest", "open my quest settings",
                  "is the hologram on the quest", "turn the headset on"):
            self.assertIsNone(self.r(t, holo_up=True), t)


class XrPromptTests(unittest.TestCase):
    FULL = prompts.PC_CONTROL_PROMPT

    def setUp(self):
        _core, self.sections = pr.split_pc_control(self.FULL)

    def test_the_action_is_taught_in_the_holo_section_in_a_few_lines(self):
        body = dict(self.sections)["HOLO MODE"]
        self.assertIn("holo_quest", body)
        self.assertIn("[ACTION: holo_quest]", body)
        self.assertIn("ONLY 'on the Quest/headset'", body)
        lines = [ln for ln in body.splitlines() if "holo_quest" in ln or "Quest" in ln
                 or "headset" in ln]
        self.assertLessEqual(len(lines), 4)

    def test_the_phrase_loads_the_section_and_the_action(self):
        for q in ("open the hologram on the Quest", "open up the hologram on my headset"):
            with self.subTest(q=q):
                self.assertIn("HOLO MODE", pr.select_sections(q, self.sections)[0])
                self.assertIn("holo_quest", pr.turn_pc_block(q, self.FULL))


if __name__ == "__main__":
    unittest.main()
