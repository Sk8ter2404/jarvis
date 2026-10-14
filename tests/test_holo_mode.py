"""Holo mode (2026-10-14): the /holo page + /api/holo/state routes, the shared
state, the voice actions, prompt reachability and the page's static contract
(pinned libraries, no camera, no OS input).

Hermetic: servers bind 127.0.0.1:0, no browser is launched (the skill's window
launcher is replaced), no network, no camera, no tracker service.
"""
from __future__ import annotations

import os
import re
import sys
import unittest
import urllib.error
import urllib.request
from unittest import mock

from core import holo_web, prompt_router as pr, prompts
from skills import holo_mode
from tests.test_web_interface import _ServerBase, _get, _get_raw

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_HOLO_DIR = os.path.join(_ROOT, "hud", "holo")

ACTIONS = ("hand_menu", "holo_mode_on", "holo_mode_off", "holo_show",
           "holo_explode", "holo_assemble")


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _read(name):
    with open(os.path.join(_HOLO_DIR, name), encoding="utf-8") as fh:
        return fh.read()


# --- the shared state ---------------------------------------------------------
class HoloStateTests(unittest.TestCase):
    def setUp(self):
        self.clk = _Clock()
        self.st = holo_web.HoloState(clock=self.clk)

    def test_commands_get_increasing_seq_and_seen_filters(self):
        a = self.st.push_command("explode")
        b = self.st.push_command("assemble")
        self.assertEqual((a, b), (1, 2))
        self.assertEqual([c["cmd"] for c in self.st.payload(0)["commands"]],
                         ["explode", "assemble"])
        self.assertEqual([c["cmd"] for c in self.st.payload(1)["commands"]],
                         ["assemble"])
        self.assertEqual(self.st.payload(2)["commands"], [])

    def test_unknown_command_is_refused(self):
        with self.assertRaises(ValueError):
            self.st.push_command("self_destruct")

    def test_queue_is_bounded(self):
        for _ in range(holo_web.MAX_COMMANDS + 5):
            self.st.push_command("explode")
        self.assertEqual(len(self.st.payload(0)["commands"]),
                         holo_web.MAX_COMMANDS)

    def test_page_alive_follows_the_heartbeat(self):
        self.assertFalse(self.st.page_alive)
        self.st.heartbeat(shown="arc reactor", exploded=True)
        self.assertTrue(self.st.page_alive)
        self.assertEqual((self.st.shown, self.st.exploded),
                         ("arc reactor", True))
        self.clk.t += holo_web.ALIVE_S + 0.1
        self.assertFalse(self.st.page_alive)

    def test_close_and_mode_commands_update_state(self):
        self.st.set_on(True, model="arc reactor", mode="desktop")
        self.assertEqual(self.st.mode, "desktop")
        self.st.push_command("mode", "camera")
        self.assertEqual(self.st.mode, "camera")
        self.st.push_command("close")
        self.assertFalse(self.st.on)
        self.assertEqual(self.st.model, "")

    def test_state_payload_is_the_pages_heartbeat(self):
        out = holo_web.state_payload(
            {"seen": ["0"], "shown": ["arc reactor"], "exploded": ["1"]},
            state=self.st)
        self.assertTrue(self.st.page_alive)
        self.assertTrue(out["exploded"])
        self.assertEqual(out["shown"], "arc reactor")
        # junk values never raise
        holo_web.state_payload({"seen": ["x"]}, state=self.st)


# --- the routes ---------------------------------------------------------------
class _HoloServer(_ServerBase):
    def setUp(self):
        self.state = holo_web.HoloState()
        self._p = mock.patch.object(holo_web, "STATE", self.state)
        self._p.start()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        self._p.stop()


class HoloRouteTests(_HoloServer):
    def test_page_is_served_with_its_own_csp(self):
        req = urllib.request.Request(self.base + "/holo?mode=desktop")
        with urllib.request.urlopen(req, timeout=5) as r:
            body = r.read().decode("utf-8")
            csp = r.headers["Content-Security-Policy"]
        self.assertIn("JARVIS Holo", body)
        self.assertNotIn("__JARVIS_TOKEN__", body)
        self.assertIn("https://cdn.jsdelivr.net", csp)
        self.assertIn("ws://127.0.0.1:8793", csp)
        self.assertNotIn("unsafe-eval", csp)
        self.assertNotIn("wasm", csp)
        self.assertNotIn("googleapis", csp)
        self.assertIn("frame-ancestors 'none'", csp)

    def test_static_files_served_and_nothing_else(self):
        for name in ("holo.js", "gestures.js", "reactor.js", "fakehands.js",
                     "selftest.js"):
            code, text = _get_raw(self.base + "/holo/" + name)
            self.assertEqual(code, 200, name)
            self.assertGreater(len(text), 500, name)
        for bad in ("/holo/nope.js", "/holo/holo.html", "/holo/../core/config.py",
                    "/holo/%2e%2e/core/config.py"):
            code, _ = _get_raw(self.base + bad)
            self.assertEqual(code, 404, bad)

    def test_js_content_type(self):
        with urllib.request.urlopen(self.base + "/holo/holo.js", timeout=5) as r:
            self.assertTrue(r.headers["Content-Type"].startswith("text/javascript"))

    def test_state_endpoint_shape_and_heartbeat(self):
        self.state.push_command("explode")
        code, data = _get(self.base + "/api/holo/state?seen=0&shown=arc%20reactor&exploded=0")
        self.assertEqual(code, 200)
        for k in ("on", "mode", "model", "seq", "commands", "shown", "exploded"):
            self.assertIn(k, data)
        self.assertEqual(data["commands"][0]["cmd"], "explode")
        self.assertTrue(self.state.page_alive)
        self.assertEqual(self.state.shown, "arc reactor")


class HoloRouteTokenTests(_HoloServer):
    token = "s3cret-token"

    def test_state_needs_the_token_and_the_page_carries_it(self):
        code, _ = _get_raw(self.base + "/api/holo/state")
        self.assertEqual(code, 401)
        code, data = _get(self.base + "/api/holo/state",
                          headers={"X-Auth-Token": self.token})
        self.assertEqual(code, 200)
        code, html = _get_raw(self.base + "/holo")
        self.assertEqual(code, 200)             # local bind: page is token-free
        self.assertIn('"s3cret-token"', html)   # ...and bakes the token in for its polls


# --- the actions --------------------------------------------------------------
class HoloActionTests(unittest.TestCase):
    def setUp(self):
        self.clk = _Clock()
        self.state = holo_web.HoloState(clock=self.clk)
        self.opened = []
        self._patches = [
            mock.patch.object(holo_web, "STATE", self.state),
            mock.patch.object(holo_mode, "_ensure_server",
                              return_value=(True, "127.0.0.1", 8766, "")),
            mock.patch.object(holo_mode, "_launch_window",
                              side_effect=lambda u: self.opened.append(u) or True),
            mock.patch.object(holo_mode, "_enabled", return_value=True),
            mock.patch.object(holo_mode, "_tracker", side_effect=lambda: self.trk),
        ]
        self.trk = mock.Mock()
        self.trk.ensure_for_holo.return_value = (True, "started")
        self.trk.release_for_holo.return_value = True
        for p in self._patches:
            p.start()
        holo_mode._last_open[0] = 0.0

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _page_up(self, shown="", exploded=False):
        self.state.set_on(True)
        self.state.heartbeat(shown=shown, exploded=exploded)

    def test_holo_mode_on_opens_the_camera_page(self):
        self.assertEqual(holo_mode.holo_mode_on(), "Holo mode up, sir.")
        self.assertEqual(self.opened, ["http://127.0.0.1:8766/holo?mode=camera"])
        self.assertTrue(self.state.on)
        self.assertEqual(self.state.mode, "camera")

    def test_hand_menu_opens_the_desktop_page(self):
        self.assertEqual(holo_mode.hand_menu(), "Hand menu up, sir.")
        self.assertEqual(self.opened, ["http://127.0.0.1:8766/holo?mode=desktop"])
        self.assertEqual(self.state.mode, "desktop")

    def test_the_hologram_starts_the_hand_tracker(self):
        holo_mode.holo_mode_on()
        self.trk.ensure_for_holo.assert_called_once()
        holo_mode.hand_menu()                     # already opening: no second start
        self.assertEqual(self.trk.ensure_for_holo.call_count, 1)

    def test_a_tracker_that_cannot_start_is_said(self):
        self.trk.ensure_for_holo.return_value = (False, "the hand model file isn't downloaded")
        out = holo_mode.holo_mode_on()
        self.assertTrue(out.startswith("Holo mode up, sir."))
        self.assertIn("couldn't start (the hand model file isn't downloaded)", out)
        self.assertIn("won't see your hands", out)

    def test_no_hand_tracking_skill_is_said(self):
        self.trk = None
        self.assertIn("isn't loaded", holo_mode.holo_mode_on())

    def test_closing_the_hologram_releases_the_tracker(self):
        self._page_up()
        self.assertEqual(holo_mode.holo_mode_off(), "Holo mode closed, sir.")
        self.trk.release_for_holo.assert_called_once()

    def test_mode_words(self):
        for arg, mode in (("desktop", "desktop"), ("hand menu", "desktop"),
                          ("camera", "camera"), ("hologram", "camera"),
                          ("", "camera"), ("nonsense", "camera")):
            self.assertEqual(holo_mode._norm_mode(arg, "camera"), mode, arg)

    def test_a_second_request_while_opening_does_not_open_a_second_window(self):
        holo_mode.holo_mode_on()
        self.assertIn("coming up", holo_mode.holo_mode_on())
        self.assertEqual(len(self.opened), 1)

    def test_already_up_and_mode_switch(self):
        self._page_up()
        self.state.mode = "camera"
        self.assertIn("already up", holo_mode.holo_mode_on("camera"))
        self.assertIn("Switched", holo_mode.holo_mode_on("desktop"))
        self.assertEqual(self.state.payload(0)["commands"][-1],
                         {"seq": 1, "cmd": "mode", "arg": "desktop"})
        self.assertEqual(self.opened, [])

    def test_off_is_honest_and_closes(self):
        self.assertIn("isn't up", holo_mode.holo_mode_off())
        self._page_up()
        self.assertIn("closed", holo_mode.holo_mode_off())
        self.assertEqual(self.state.payload(0)["commands"][-1]["cmd"], "close")
        self.assertFalse(self.state.on)

    def test_explode_and_assemble_state_transitions(self):
        self.assertIn("isn't up", holo_mode.holo_explode())
        self._page_up(shown="")
        self.assertIn("nothing on the holo display", holo_mode.holo_explode())
        self._page_up(shown="arc reactor", exploded=False)
        self.assertIn("It's already in one piece", holo_mode.holo_assemble())
        self.assertEqual(holo_mode.holo_explode(), "Exploding it, sir.")
        self.assertEqual(self.state.payload(0)["commands"][-1]["cmd"], "explode")
        self._page_up(shown="arc reactor", exploded=True)
        self.assertIn("already apart", holo_mode.holo_explode())
        self.assertEqual(holo_mode.holo_assemble(),
                         "Putting it back together, sir.")
        self.assertEqual(self.state.payload(0)["commands"][-1]["cmd"], "assemble")

    def test_show_unknown_model_is_honest(self):
        out = holo_mode.holo_show("tiger pendant")
        self.assertIn("only have the arc reactor", out)
        self.assertEqual(self.opened, [])
        self.assertEqual(self.state.payload(0)["commands"], [])

    def test_show_opens_the_page_with_the_model_when_down(self):
        out = holo_mode.holo_show("the arc reactor")
        self.assertIn("arc reactor is up", out)
        self.assertEqual(self.opened, ["http://127.0.0.1:8766/holo?mode=desktop"])
        self.assertEqual(self.state.model, "arc reactor")

    def test_show_pushes_a_command_when_the_page_is_up(self):
        self._page_up()
        holo_mode.holo_show("arc reactor")
        self.assertEqual(self.opened, [])
        self.assertEqual(self.state.payload(0)["commands"][-1],
                         {"seq": 1, "cmd": "show", "arg": "arc reactor"})

    def test_show_default_model_when_no_argument(self):
        self._page_up()
        holo_mode.holo_show("")
        self.assertEqual(self.state.payload(0)["commands"][-1]["arg"], "arc reactor")

    def test_window_that_cannot_open_is_reported_and_state_rolled_back(self):
        with mock.patch.object(holo_mode, "_launch_window", return_value=False):
            self.assertIn("couldn't open", holo_mode.holo_mode_on())
        self.assertFalse(self.state.on)

    def test_server_refusal_is_spoken(self):
        with mock.patch.object(holo_mode, "_ensure_server",
                               return_value=(False, "", 0, "Not while I'm in staging, sir.")):
            self.assertEqual(holo_mode.holo_mode_on(), "Not while I'm in staging, sir.")
        self.assertEqual(self.opened, [])

    def test_disabled_setting_answers_off_for_every_action(self):
        with mock.patch.object(holo_mode, "_enabled", return_value=False):
            for fn in (holo_mode.holo_mode_on, holo_mode.holo_mode_off,
                       holo_mode.holo_show, holo_mode.holo_explode,
                       holo_mode.holo_assemble):
                self.assertIn("switched off", fn(""), fn.__name__)
        self.assertEqual(self.opened, [])

    def test_register_and_speak_set(self):
        acts = {}
        holo_mode.register(acts)
        self.assertEqual(set(acts), set(ACTIONS))
        self.assertEqual(set(holo_mode.SPEAK_VERBATIM_ACTIONS), set(ACTIONS))
        for name, fn in acts.items():
            self.assertTrue(callable(fn), name)


class HoloLaunchTests(unittest.TestCase):
    def test_launcher_uses_an_app_window_not_a_tab(self):
        with mock.patch.object(holo_mode, "_chrome_paths", return_value=["C:/x/chrome.exe"]), \
             mock.patch.object(holo_mode.os.path, "isfile", return_value=True), \
             mock.patch.object(holo_mode.subprocess, "Popen") as popen:
            self.assertTrue(holo_mode._launch_window("http://127.0.0.1:8766/holo?mode=camera"))
        argv = popen.call_args[0][0]
        self.assertIn("--app=http://127.0.0.1:8766/holo?mode=camera", argv)

    def test_launcher_falls_back_to_open_url(self):
        seen = []
        with mock.patch.object(holo_mode, "_chrome_paths", return_value=[]), \
             mock.patch.object(holo_mode, "skill_utils", {"open_url": seen.append}):
            self.assertTrue(holo_mode._launch_window("http://127.0.0.1:1/holo"))
        self.assertEqual(seen, ["http://127.0.0.1:1/holo"])

    def test_launcher_reports_failure_with_no_browser(self):
        with mock.patch.object(holo_mode, "_chrome_paths", return_value=[]), \
             mock.patch.object(holo_mode, "skill_utils", {}):
            self.assertFalse(holo_mode._launch_window("http://127.0.0.1:1/holo"))

    def test_ensure_server_without_the_web_skill_loaded(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop("skill_web_interface", None)
            ok, _h, _p, why = holo_mode._ensure_server()
        self.assertFalse(ok)
        self.assertIn("web interface", why)

    def test_ensure_server_starts_a_down_server_and_reports_its_port(self):
        fake = mock.Mock()
        fake._httpd = None
        fake._bound = ("", 0)

        def start():
            fake._httpd, fake._bound = object(), ("127.0.0.1", 8766)
            return True, "up"
        fake._start.side_effect = start
        with mock.patch.dict(sys.modules, {"skill_web_interface": fake}):
            self.assertEqual(holo_mode._ensure_server(),
                             (True, "127.0.0.1", 8766, ""))
        fake._start.assert_called_once()

    def test_ensure_server_passes_up_the_refusal(self):
        fake = mock.Mock()
        fake._httpd = None
        fake._start.return_value = (False, "Not while I'm in staging, sir.")
        with mock.patch.dict(sys.modules, {"skill_web_interface": fake}):
            ok, _h, _p, why = holo_mode._ensure_server()
        self.assertFalse(ok)
        self.assertEqual(why, "Not while I'm in staging, sir.")


# --- prompt reachability ------------------------------------------------------
class HoloPromptTests(unittest.TestCase):
    FULL = prompts.PC_CONTROL_PROMPT

    def setUp(self):
        _core, self.sections = pr.split_pc_control(self.FULL)

    def test_every_action_is_in_the_prompt(self):
        for name in ACTIONS:
            self.assertIn(name, self.FULL, name)

    def test_holo_mode_is_a_parsed_section(self):
        self.assertIn("HOLO MODE", [h for h, _b in self.sections])

    def test_voice_phrases_load_the_section_and_their_action(self):
        for q, action in (("open up the hologram", "holo_mode_on"),
                          ("hand menu", "hand_menu"),
                          ("show me the arc reactor", "holo_show"),
                          ("explode it", "holo_explode"),
                          ("put it back together", "holo_assemble"),
                          ("close the hologram", "holo_mode_off")):
            with self.subTest(q=q):
                inc = pr.select_sections(q, self.sections)[0]
                self.assertIn("HOLO MODE", inc)
                self.assertIn(action, pr.turn_pc_block(q, self.FULL))


# --- the page's static contract ----------------------------------------------
class HoloPageContractTests(unittest.TestCase):
    PAGE_FILES = ("holo.html", "holo.js", "gestures.js", "reactor.js",
                  "fakehands.js", "selftest.js")

    def test_every_served_file_exists_and_every_file_is_served(self):
        for path, (fname, _ct) in holo_web._ASSETS.items():
            self.assertTrue(os.path.isfile(os.path.join(_HOLO_DIR, fname)), path)
        on_disk = set(os.listdir(_HOLO_DIR))
        served = {f for f, _ in holo_web._ASSETS.values()}
        self.assertEqual(on_disk - {"__pycache__"}, served)

    def test_libraries_are_pinned_to_exact_versions(self):
        urls = []
        for f in self.PAGE_FILES:
            urls += re.findall(r"https://[^\s\"'`)]+", _read(f))
        cdn = [u for u in urls if "jsdelivr" in u]
        self.assertTrue(cdn, "no CDN library found")
        for u in cdn:
            self.assertRegex(u, r"/npm/(@[\w.-]+/)?[\w.-]+@\d+\.\d+\.\d+/", u)
            self.assertNotIn("@latest", u)
        self.assertIn(
            "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            _read("holo.js"))
        for f in self.PAGE_FILES:
            self.assertNotIn("@latest", _read(f), f)

    def test_no_third_party_host_except_the_pinned_cdn(self):
        for f in self.PAGE_FILES:
            for u in re.findall(r"https?://[^\s\"'`)]+", _read(f)):
                if u.startswith("http://www.w3.org"):
                    continue
                self.assertTrue(u.startswith("https://cdn.jsdelivr.net/npm/three@"), (f, u))

    def test_page_never_opens_the_camera_or_runs_a_hand_model(self):
        for f in self.PAGE_FILES:
            text = _read(f)
            for banned in ("getUserMedia", "mediaDevices", "tasks-vision",
                           "HandLandmarker", "FilesetResolver"):
                self.assertNotIn(banned, text, (f, banned))

    def test_page_talks_to_the_tracker_and_has_both_modes_and_fake(self):
        js = _read("holo.js")
        self.assertIn("ws://127.0.0.1:8793", js)
        self.assertIn('"camera"', js)
        self.assertIn('"desktop"', js)
        self.assertIn('qs.get("fake")', js)
        self.assertIn('type: "subscribe"', js)

    def test_every_legend_key_is_bound(self):
        js = _read("holo.js")
        for k in ("e", "c", "b", "f", "l", "d"):
            self.assertIn(f'case "{k}":', js, k)

    def test_reactor_has_the_named_parts(self):
        js = _read("reactor.js")
        for name in ("Core", "Inner ring", "Outer ring", "Housing", "Back plate",
                     "Copper coil"):
            self.assertIn(name, js)
        self.assertIn("const COILS = 10", js)

    def test_gesture_module_is_pure(self):
        js = _read("gestures.js")
        code = re.sub(r"//[^\n]*", "", js)
        for banned in ("document.", "window.", "fetch(", "setTimeout", "WebSocket",
                       "THREE", "import "):
            self.assertNotIn(banned, code, banned)

    def test_nothing_in_holo_can_move_the_os_cursor_or_send_input(self):
        paths = [os.path.join(_ROOT, "skills", "holo_mode.py"),
                 os.path.join(_ROOT, "core", "holo_web.py")]
        paths += [os.path.join(_HOLO_DIR, f) for f in self.PAGE_FILES]
        for p in paths:
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
            for banned in ("pyautogui", "SetCursorPos", "SendInput", "mouse_event",
                           "keybd_event", "ui_click", "ui_hotkey", "air_control",
                           "kinect_air_mouse"):
                self.assertNotIn(banned, text, (os.path.basename(p), banned))


if __name__ == "__main__":
    unittest.main()
