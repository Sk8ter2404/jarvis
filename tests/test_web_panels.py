"""Skill-declared web panels: core/web_panels.py + the tools/web_interface routes.

Every fixture is a SYNTHETIC "desk device" - the public tree must never name a
private device, so neither may its tests.

Covers: spec validation (and the log line a rejected spec leaves), the loader
hook, the time-limited cached state(), the action gate (declared actions only,
confirm, rate limit, stops never throttled), every route (auth, 404s, the
cross-origin guard, JSON-only), the MJPEG stream + snapshot, the hold/stop JS
contract, and a structural check that no private panel id is in the tree.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import subprocess
import time
import types
import unittest
import urllib.request

from core import web_panels as wp
from tools import web_interface as wi
from tests.test_web_interface import (_ServerBase, _get, _get_raw, _js_fn,
                                      _post, _raw_post_status, _urlopen_retry)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
JPEG = b"\xff\xd8\xff\xe0FAKEJPEG\xff\xd9"


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class DeskDevice:
    """A pretend device: state, a hold-to-drive action, a stop, a confirm-gated
    reset, and a camera stream."""

    def __init__(self):
        self.calls = []
        self.frame = JPEG
        self.state_delay = 0.0
        self.state_raises = False
        self.state_calls = 0

    def state(self):
        self.state_calls += 1
        if self.state_delay:
            time.sleep(self.state_delay)
        if self.state_raises:
            raise RuntimeError("sensor bus busy")
        return {"battery": 87, "mode": "idle", "events": ["booted"], "lights": True}

    def act(self, name):
        def fn(args):
            self.calls.append((name, dict(args)))
            return {"did": name}
        return fn

    def spec(self, **over):
        s = {
            "id": "desk_device", "title": "Desk device", "order": 50,
            "poll_ms": 500, "state": self.state, "state_timeout_s": 0.3,
            "stop_action": "stop",
            "layout": [
                {"type": "stat", "key": "battery", "label": "Battery", "unit": "%"},
                {"type": "badge", "key": "mode", "map": {"idle": "ok", "fault": "bad"}},
                {"type": "gauge", "key": "battery", "min": 0, "max": 100},
                {"type": "text", "key": "mode"},
                {"type": "events", "key": "events", "max": 5},
                {"type": "image", "stream": "cam"},
                {"type": "buttons", "buttons": [{"label": "Wave", "action": "wave",
                                                 "args": {"times": 2}}]},
                {"type": "input", "action": "say", "arg": "text"},
                {"type": "toggle", "key": "lights", "action": "lights"},
                {"type": "slider", "key": "battery", "action": "speed", "min": 0, "max": 10},
                {"type": "hold", "label": "Forward", "action": "drive", "args": {"dir": "f"}},
                {"type": "estop", "label": "STOP", "action": "estop"},
            ],
            "actions": {
                "stop": self.act("stop"), "estop": self.act("estop"),
                "wave": self.act("wave"), "say": self.act("say"),
                "lights": self.act("lights"), "speed": self.act("speed"),
                "drive": {"fn": self.act("drive"), "hold": True, "rate_hz": 5},
                "reset": {"fn": self.act("reset"), "confirm": True, "danger": True,
                          "label": "Factory reset"},
            },
            "streams": {"cam": lambda: self.frame},
        }
        s.update(over)
        return s


class RegistryValidationTests(unittest.TestCase):

    def setUp(self):
        self.lines = []
        self.clock = _Clock()
        self.reg = wp.PanelRegistry(clock=self.clock, log=self.lines.append)
        self.dev = DeskDevice()

    def test_a_good_spec_registers_and_its_metadata_is_json_only(self):
        self.assertTrue(self.reg.register(self.dev.spec(), owner="desk_skill"))
        meta = self.reg.list_meta()
        self.assertEqual([m["id"] for m in meta], ["desk_device"])
        json.dumps(meta)                          # no callables leak out
        m = meta[0]
        self.assertTrue(m["actions"]["reset"]["confirm"])
        self.assertTrue(m["actions"]["drive"]["hold"])
        self.assertTrue(m["actions"]["stop"]["stop"])
        self.assertTrue(m["actions"]["estop"]["stop"])
        self.assertEqual(m["streams"], ["cam"])
        self.assertIn("registered", self.lines[-1])

    def _rejects(self, why, **over):
        self.lines.clear()
        self.assertFalse(self.reg.register(self.dev.spec(**over), owner="desk_skill"),
                         why)
        self.assertTrue(self.lines and "REJECTED" in self.lines[-1], why)
        self.assertEqual(self.reg.ids(), [], why)

    def test_bad_specs_are_rejected_with_a_log_line(self):
        d = self.dev
        self._rejects("bad id", id="Desk Device!")
        self._rejects("no title", title="")
        self._rejects("unknown key", colour="red")
        self._rejects("poll too fast", poll_ms=10)
        self._rejects("state not callable", state="nope")
        self._rejects("unknown widget", layout=[{"type": "joystick"}])
        self._rejects("undeclared action",
                      layout=[{"type": "buttons", "buttons": [{"label": "x", "action": "ghost"}]}])
        self._rejects("undeclared stream", layout=[{"type": "image", "stream": "nope"}])
        self._rejects("stat without key", layout=[{"type": "stat"}])
        self._rejects("hold without stop_action", stop_action=None)
        self._rejects("hold on a non-hold action",
                      layout=[{"type": "hold", "action": "wave"}])
        spec = d.spec()
        spec["actions"]["estop"] = {"fn": d.act("estop"), "confirm": True}
        self.lines.clear()
        self.assertFalse(self.reg.register(spec, owner="desk_skill"))
        self.assertIn("must not require confirmation", self.lines[-1])
        self._rejects("rate too high",
                      actions=dict(d.spec()["actions"], drive={"fn": d.act("drive"), "hold": True, "rate_hz": 500}))
        self._rejects("bad badge tone",
                      layout=[{"type": "badge", "key": "mode", "map": {"idle": "purple"}}])
        self.assertFalse(self.reg.register("not a dict"))

    def test_another_skill_cannot_take_an_id(self):
        self.assertTrue(self.reg.register(self.dev.spec(), owner="a"))
        self.assertFalse(self.reg.register(self.dev.spec(title="Other"), owner="b"))
        self.assertEqual(self.reg.list_meta()[0]["title"], "Desk device")
        self.assertTrue(self.reg.register(self.dev.spec(title="Renamed"), owner="a"))
        self.assertEqual(self.reg.list_meta()[0]["title"], "Renamed")

    def test_register_from_module_replaces_on_reload(self):
        mod = types.ModuleType("skill_desk")
        mod.WEB_PANELS = [self.dev.spec(), self.dev.spec(id="desk_two", title="Two")]
        self.assertEqual(self.reg.register_from_module(mod, "desk"), 2)
        mod.WEB_PANELS = [self.dev.spec(id="desk_two", title="Two")]
        self.assertEqual(self.reg.register_from_module(mod, "desk"), 1)
        self.assertEqual(self.reg.ids(), ["desk_two"])
        del mod.WEB_PANELS
        self.assertEqual(self.reg.register_from_module(mod, "desk"), 0)
        self.assertEqual(self.reg.ids(), [])
        mod.WEB_PANELS = "junk"
        self.assertEqual(self.reg.register_from_module(mod, "desk"), 0)
        self.assertIn("must be a list", self.lines[-1])

    def test_panels_sort_by_order_then_title(self):
        self.reg.register(self.dev.spec(id="b", title="B", order=5), owner="x")
        self.reg.register(self.dev.spec(id="a", title="A", order=9), owner="x")
        self.assertEqual([m["id"] for m in self.reg.list_meta()], ["b", "a"])


class RegistryStateTests(unittest.TestCase):

    def setUp(self):
        self.clock = _Clock()
        self.reg = wp.PanelRegistry(clock=self.clock, log=lambda _l: None)
        self.dev = DeskDevice()
        self.reg.register(self.dev.spec(), owner="desk")

    def test_state_is_fetched_then_cached(self):
        code, p = self.reg.state("desk_device")
        self.assertEqual(code, 200)
        self.assertFalse(p["stale"])
        self.assertEqual(p["state"]["battery"], 87)
        self.dev.state_raises = True           # a cached read never calls it
        code, p = self.reg.state("desk_device")
        self.assertFalse(p["stale"])

    def test_a_slow_state_never_hangs_the_request(self):
        self.reg.state("desk_device")          # one good value
        self.clock.t += 5                      # cache expired
        self.dev.state_delay = 2.0
        t0 = time.monotonic()
        code, p = self.reg.state("desk_device")
        took = time.monotonic() - t0
        self.assertLess(took, 1.5, "a slow state() held the request")
        self.assertTrue(p["stale"])
        self.assertEqual(p["state"]["battery"], 87, "the last good value is served")
        self.assertIn("still running", p["error"])
        # ...and only ONE fetch runs at a time: a second request does not pile on
        code, p2 = self.reg.state("desk_device")
        self.assertTrue(p2["stale"])
        self.assertEqual(self.dev.state_calls, 2,
                         "a second request started a second state() while the "
                         "first was still running")

    def test_a_failing_state_is_stale_with_the_error(self):
        self.reg.state("desk_device")
        self.clock.t += 5
        self.dev.state_raises = True
        code, p = self.reg.state("desk_device")
        self.assertTrue(p["stale"])
        self.assertIn("sensor bus busy", p["error"])
        self.assertEqual(p["state"]["battery"], 87)

    def test_unknown_panel(self):
        self.assertEqual(self.reg.state("nope")[0], 404)

    def test_non_json_state_is_coerced(self):
        self.reg.register(self.dev.spec(id="odd", title="Odd",
                                        state=lambda: {"when": object()}), owner="desk")
        code, p = self.reg.state("odd")
        self.assertIsInstance(p["state"]["when"], str)


class RegistryActionTests(unittest.TestCase):

    def setUp(self):
        self.clock = _Clock()
        self.reg = wp.PanelRegistry(clock=self.clock, log=lambda _l: None)
        self.dev = DeskDevice()
        self.reg.register(self.dev.spec(), owner="desk")

    def test_declared_actions_only(self):
        self.assertEqual(self.reg.call_action("desk_device", "ghost", {})[0], 404)
        self.assertEqual(self.reg.call_action("nope", "wave", {})[0], 404)
        self.assertEqual(self.reg.call_action("desk_device", "wave", [1])[0], 400)
        code, p = self.reg.call_action("desk_device", "wave", {"times": 2})
        self.assertEqual((code, p["result"]), (200, {"did": "wave"}))
        self.assertEqual(self.dev.calls, [("wave", {"times": 2})])

    def test_confirm_gate(self):
        code, p = self.reg.call_action("desk_device", "reset", {})
        self.assertEqual(code, 409)
        self.assertTrue(p["confirm_required"])
        self.assertEqual(self.dev.calls, [])
        self.assertEqual(self.reg.call_action("desk_device", "reset", {}, confirm=True)[0], 200)

    def test_rate_limit_but_stops_always_pass(self):
        self.assertEqual(self.reg.call_action("desk_device", "drive", {})[0], 200)
        self.clock.t += 0.01
        self.assertEqual(self.reg.call_action("desk_device", "drive", {})[0], 429)
        self.clock.t += 0.2                    # > 1 / (2 * 5 Hz)
        self.assertEqual(self.reg.call_action("desk_device", "drive", {})[0], 200)
        for _ in range(5):                     # never throttled
            self.assertEqual(self.reg.call_action("desk_device", "stop", {})[0], 200)
            self.assertEqual(self.reg.call_action("desk_device", "estop", {})[0], 200)

    def test_a_raising_action_is_a_500_not_an_exception(self):
        self.reg.register(self.dev.spec(id="bad", title="Bad", layout=[],
                                        stop_action=None,
                                        actions={"x": lambda a: 1 / 0}), owner="desk")
        code, p = self.reg.call_action("bad", "x", {})
        self.assertEqual(code, 500)
        self.assertIn("ZeroDivisionError", p["error"])


class _PanelServer(_ServerBase):
    def server_extra(self):
        self.clock = _Clock()
        self.reg = wp.PanelRegistry(clock=self.clock, log=lambda _l: None)
        self.dev = DeskDevice()
        self.reg.register(self.dev.spec(), owner="desk")
        return {"panels": self.reg, "runtime": wi.NoRuntime()}


class PanelRouteTests(_PanelServer):

    def test_panels_lists_metadata(self):
        code, d = _get(self.base + "/api/panels")
        self.assertEqual(code, 200)
        self.assertEqual(d["panels"][0]["id"], "desk_device")
        self.assertNotIn("state", d["panels"][0])

    def test_state_route(self):
        code, d = _get(self.base + "/api/panel/desk_device/state")
        self.assertEqual(code, 200)
        self.assertEqual(d["state"]["mode"], "idle")
        self.assertEqual(_get_raw(self.base + "/api/panel/nope/state")[0], 404)
        self.assertEqual(_get_raw(self.base + "/api/panel/desk_device/bogus")[0], 404)

    def test_state_route_times_out_to_stale(self):
        _get(self.base + "/api/panel/desk_device/state")
        self.clock.t += 5
        self.dev.state_delay = 3.0
        t0 = time.monotonic()
        code, d = _get(self.base + "/api/panel/desk_device/state")
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertTrue(d["stale"])

    def _action(self, body, ctype="application/json"):
        req = urllib.request.Request(
            self.base + "/api/panel/desk_device/action",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": ctype}, method="POST")
        try:
            with _urlopen_retry(req, timeout=5) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def test_action_route(self):
        code, d = self._action({"name": "wave", "args": {"times": 1}})
        self.assertEqual(code, 200)
        self.assertEqual(self.dev.calls, [("wave", {"times": 1})])
        self.assertEqual(self._action({"name": "ghost"})[0], 404)
        self.assertEqual(self._action({"name": "reset"})[0], 409)
        self.assertEqual(self._action({"name": "reset", "confirm": True})[0], 200)
        self.assertEqual(self._action({"name": "wave"}, ctype="text/plain")[0], 415)
        # declared actions run DIRECTLY - nothing reaches the command channel
        self.assertFalse(os.path.exists(self.inject_path))

    def test_action_route_rate_limit(self):
        self.assertEqual(self._action({"name": "drive"})[0], 200)
        self.assertEqual(self._action({"name": "drive"})[0], 429)
        self.assertEqual(self._action({"name": "stop"})[0], 200)

    def test_cross_origin_action_is_refused(self):
        code = _raw_post_status(self.host, self.port, "/api/panel/desk_device/action",
                                {"name": "wave"},
                                {"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        self.assertEqual(self.dev.calls, [])

    def test_snapshot_and_stream(self):
        req = urllib.request.Request(self.base + "/api/panel/desk_device/stream/cam?still=1")
        with _urlopen_retry(req, timeout=5) as r:
            self.assertEqual(r.headers.get("Content-Type"), "image/jpeg")
            self.assertEqual(r.read(), JPEG)
        req = urllib.request.Request(self.base + "/api/panel/desk_device/stream/cam")
        r = _urlopen_retry(req, timeout=5)
        try:
            self.assertIn("multipart/x-mixed-replace", r.headers.get("Content-Type"))
            head = b""
            deadline = time.monotonic() + 3.0
            while JPEG not in head and time.monotonic() < deadline:
                chunk = r.read1(256)          # one frame is sent, then nothing new
                if not chunk:
                    break
                head += chunk
            self.assertIn(b"Content-Type: image/jpeg", head)
            self.assertIn(JPEG, head)
        finally:
            r.close()
            self.dev.frame = None              # let the stream close on its own
        self.assertEqual(_get_raw(self.base + "/api/panel/desk_device/stream/nope")[0], 404)

    def test_no_frame_is_404(self):
        self.dev.frame = None
        self.assertEqual(_get_raw(self.base + "/api/panel/desk_device/stream/cam")[0], 404)


class PanelTokenTests(_PanelServer):
    token = "s3cr3t"

    def test_every_panel_route_needs_the_token(self):
        for path in ("/api/panels", "/api/panel/desk_device/state",
                     "/api/panel/desk_device/stream/cam?still=1"):
            self.assertEqual(_get_raw(self.base + path)[0], 401, path)
        self.assertEqual(_post(self.base + "/api/panel/desk_device/action",
                               {"name": "wave"})[0], 401)
        self.assertEqual(self.dev.calls, [])
        code, _ = _get(self.base + "/api/panels", headers={"X-Auth-Token": self.token})
        self.assertEqual(code, 200)


class PanelPageContractTests(_ServerBase):
    """The hold/stop contract is enforced in the page, so it is pinned here."""

    def test_hold_resends_while_pressed_and_stops_on_every_release(self):
        html = _get_raw(self.base + "/")[1]
        hold = _js_fn(html, "bindHold")
        for ev in ("'pointerdown', start", "'pointerup', stop", "'pointercancel', stop",
                   "'pointerleave', stop"):
            self.assertIn(ev, hold)
        # A captured pointer never fires pointerleave: sliding off the button
        # would keep the device moving. The implicit (touch) capture is dropped.
        self.assertNotIn("setPointerCapture", hold)
        self.assertIn("releasePointerCapture", hold)
        self.assertIn("window.addEventListener('blur', stop)", hold)
        self.assertIn("document.addEventListener('visibilitychange'", hold)
        self.assertIn("if (document.hidden) stop();", hold)
        self.assertIn("setInterval(send, Math.max(33, Math.round(1000 / rate)))", hold)
        self.assertIn("clearInterval(timer)", hold)
        self.assertIn("panelAction(p, p.stop_action", hold)
        self.assertIn("HOLD_STOPPERS.push(stop)", hold)
        # leaving the view releases every hold too
        self.assertIn("stopAllHolds()", _js_fn(html, "stopPanelMedia"))
        self.assertIn("stopPanelMedia()", _js_fn(html, "stopViewTimers"))

    def test_estop_is_pinned_and_unconditional(self):
        html = _get_raw(self.base + "/")[1]
        self.assertIn('<div id="estopDock" hidden', html)
        self.assertIn("#estopDock { position:fixed;", html)
        widget = _js_fn(html, "renderWidget")
        self.assertIn("estopDock.appendChild(b)", widget)
        self.assertIn("stopAllHolds(); panelAction(p, w.action, {}, {noConfirm: true})", widget)

    def test_state_polls_only_while_the_panel_is_visible(self):
        html = _get_raw(self.base + "/")[1]
        start = _js_fn(html, "startPanelView")
        self.assertIn("if (pollsWanted()) loadPanelState(p);", start)
        self.assertIn("if (panelTimer) { clearInterval(panelTimer); panelTimer = null; }",
                      _js_fn(html, "stopViewTimers"))
        self.assertIn("startPanelView(VIEWS[which].panel)", _js_fn(html, "showView"))

    def test_every_widget_type_has_a_renderer(self):
        widget = _js_fn(_get_raw(self.base + "/")[1], "renderWidget")
        for t in wp.WIDGET_TYPES:
            self.assertIn("'%s'" % t, widget, t)


class PanelImagePlaceholderTests(_ServerBase):
    """The generic image widget BEFORE its first frame (2026-09-30).

    THE DEFECT: renderWidget pointed the <img> at the still URL the moment the
    view opened and startPanelView re-requested it every refresh_ms. With no
    frame yet the box showed the browser's broken-image icon and the page
    fetched a 404 every 2 s, forever. Now the box starts as a neutral "No
    picture yet" placeholder and nothing is requested until the panel's STATE
    (the widget's `key`) or a USER ACTION says a frame exists (a keyless
    widget also takes one look when the view opens). A 404 goes back to the
    placeholder and stops asking; with a frame, snapshot mode refreshes every
    refresh_ms as before. These pin that contract in the page source (the
    page has no JS runtime in this suite)."""

    def setUp(self):
        super().setUp()
        self.html = _get_raw(self.base + "/")[1]

    def fn(self, name):
        return _js_fn(self.html, name)

    def image_branch(self):
        widget = self.fn("renderWidget")
        start = widget.index("if (t === 'image')")
        return widget[start:widget.index("if (t === 'buttons')", start)]

    def test_the_box_starts_as_a_neutral_placeholder(self):
        branch = self.image_branch()
        self.assertIn("empty.textContent = 'No picture yet'", branch)
        self.assertIn("img.hidden = true", branch)
        self.assertNotIn(".src", branch.replace("m.src", "").replace("src:", ""),
                         "the widget must not request anything while it is built")
        self.assertIn(".pw .pimg-empty {", self.html)
        self.assertIn(".pw img[hidden] { display:none; }", self.html)

    def test_opening_the_view_never_starts_a_blind_poll(self):
        start = self.fn("startPanelView")
        self.assertNotIn("?still=1", start)
        self.assertNotIn(".src =", start)
        self.assertIn("startPanelImage(m)", start)
        img = self.fn("startPanelImage")
        self.assertIn("setInterval(() => panelImageTick(m), m.refresh)", img)
        self.assertIn("if (!m.key || !m.p.has_state) panelImageLoad(m);", img)

    def test_the_poll_asks_only_with_a_frame_or_an_armed_action(self):
        tick = self.fn("panelImageTick")
        self.assertIn("if (m.has || Date.now() < m.armedUntil) panelImageFetch(m);", tick)
        self.assertIn("pollsWanted()", tick)

    def test_a_404_puts_the_placeholder_back_and_stops_asking(self):
        fetch = self.fn("panelImageFetch")
        self.assertIn("r.status === 404", fetch)
        self.assertIn("panelImageEmpty(m)", fetch)
        self.assertIn("{headers: hdr()}", fetch)
        empty = self.fn("panelImageEmpty")
        self.assertIn("m.has = false;", empty)
        self.assertIn("m.empty.hidden = false", empty)
        # ...which stops the poll: the tick asks only with a frame or an arm
        self.assertIn("if (m.has || Date.now() < m.armedUntil) panelImageFetch(m);",
                      self.fn("panelImageTick"))
        self.assertIn("removeAttribute('src')", empty)
        # a broken frame never shows the icon either
        self.assertIn("img.addEventListener('error', () => { if (m.img.getAttribute('src')) "
                      "panelImageEmpty(m); });", self.image_branch())

    def test_snapshot_keeps_refreshing_once_a_frame_exists(self):
        fetch = self.fn("panelImageFetch")
        self.assertIn("m.url = url; m.has = true; m.armedUntil = 0;", fetch)
        self.assertIn("q(m.src + '?still=1&t=' + Date.now())", fetch)
        self.assertIn("m.has = true; m.img.hidden = false; m.empty.hidden = true;",
                      self.fn("panelImageShown"))

    def test_the_state_key_says_when_a_frame_exists(self):
        self.assertIn("if (m.key) ups.push((s) => panelImageState(m, s[m.key]));",
                      self.image_branch())
        st = self.fn("panelImageState")
        self.assertIn("if (!on) { if (m.has || m.img.getAttribute('src')) panelImageEmpty(m); return; }",
                      st)
        self.assertIn("if (changed || !m.has) panelImageLoad(m);", st)

    def test_a_user_action_arms_the_panel_images(self):
        act = self.fn("panelAction")
        self.assertIn("if (res.ok && !opts.quiet) {", act)
        self.assertIn("panelImagesArm(p);", act)
        arm = self.fn("panelImagesArm")
        self.assertIn("m.armedUntil = Date.now() + PANEL_IMAGE_ARM_MS;", arm)
        self.assertIn("if (!m.active || (m.key && p.has_state)) continue;", arm)
        # a hold's resends are quiet and never arm anything
        self.assertIn("{quiet: true, noConfirm: true}", self.fn("bindHold"))

    def test_leaving_the_view_stops_the_image_timers(self):
        stop = self.fn("stopPanelMedia")
        self.assertIn("m.active = false; m.armedUntil = 0;", stop)
        self.assertIn("clearInterval(m.timer)", stop)

    def test_a_keyed_image_widget_keeps_its_key_in_the_metadata(self):
        reg = wp.PanelRegistry(clock=_Clock(), log=lambda _l: None)
        dev = DeskDevice()
        spec = dev.spec(layout=[{"type": "image", "stream": "cam", "mode": "snapshot",
                                 "refresh_ms": 2000, "key": "picture"}])
        self.assertTrue(reg.register(spec, owner="desk"))
        widget = reg.list_meta()[0]["layout"][0]
        self.assertEqual(widget, {"type": "image", "key": "picture", "stream": "cam",
                                  "mode": "snapshot", "refresh_ms": 2000})


class PanelStillRouteTests(_PanelServer):
    """The server half the placeholder relies on: a still with no frame is a
    JSON 404 (never an empty 200 image), and one with a frame is the JPEG."""

    def test_still_without_a_frame_is_a_404(self):
        self.dev.frame = None
        code, body = _get_raw(self.base + "/api/panel/desk_device/stream/cam?still=1")
        self.assertEqual(code, 404)
        self.assertIn("no frame", body)
        self.dev.frame = JPEG
        req = urllib.request.Request(self.base + "/api/panel/desk_device/stream/cam?still=1")
        with _urlopen_retry(req, timeout=5) as r:
            self.assertEqual(r.read(), JPEG)


class LoaderHookTests(unittest.TestCase):

    def test_load_skills_collects_web_panels(self):
        with open(os.path.join(_PROJECT, "bobert_companion.py"),
                  encoding="utf-8", errors="replace") as f:
            src = f.read()
        body = src[src.index("def load_skills("):]
        body = body[:body.index("\ndef ", 10)]
        self.assertIn("_collect_skill_web_panels(mod, name)", body)
        coll = src[src.index("def _collect_skill_web_panels("):]
        coll = coll[:coll.index("\ndef ", 10)]
        self.assertIn("REGISTRY.register_from_module(mod, name)", coll)


def _declared_panel_ids(path):
    """Literal "id" values of WEB_PANELS specs in one source file (AST only -
    the file is never imported or executed)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return set()
    ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if (isinstance(k, ast.Constant) and k.value == "id"
                    and isinstance(v, ast.Constant) and isinstance(v.value, str)):
                if any(isinstance(k2, ast.Constant) and k2.value == "layout"
                       for k2 in node.keys):
                    ids.add(v.value)
    return ids


class NoPrivatePanelIdsInTheTreeTests(unittest.TestCase):
    """A panel declared by an UNTRACKED / GITIGNORED (private) skill must never
    have its id spelled anywhere in the tracked tree. On a clean clone there
    are no private skills and this passes trivially; on the owner's machine it
    checks every one he has installed."""

    def test_private_panel_ids_stay_private(self):
        try:
            r = subprocess.run(["git", "-C", _PROJECT, "ls-files", "-z"],
                               capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            self.skipTest("git unavailable")
        if r.returncode != 0:
            self.skipTest("not a git checkout")
        tracked = {p.decode("utf-8", "replace") for p in r.stdout.split(b"\0") if p}
        private_ids = set()
        for p in glob.glob(os.path.join(_PROJECT, "skills", "*.py")):
            rel = os.path.relpath(p, _PROJECT).replace("\\", "/")
            if rel not in tracked:
                private_ids |= _declared_panel_ids(p)
        public_ids = set()
        for rel in tracked:
            if rel.startswith("skills/") and rel.endswith(".py"):
                public_ids |= _declared_panel_ids(os.path.join(_PROJECT, rel))
        self.assertEqual(sorted(private_ids & public_ids), [])
        leaks = []
        for rel in sorted(tracked):
            if os.path.splitext(rel)[1].lower() not in (".py", ".md", ".json", ".txt",
                                                        ".yml", ".yaml", ".html", ".js"):
                continue
            try:
                with open(os.path.join(_PROJECT, rel), encoding="utf-8",
                          errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            for pid in private_ids:
                if len(pid) >= 4 and pid in text:
                    leaks.append(rel)
                    break
        self.assertEqual(leaks, [], "a PRIVATE panel id is spelled in tracked "
                                    "file(s) (paths only): %s" % leaks)

    def test_the_scanner_sees_a_declared_id(self):
        """Guard against a scan that matches nothing and passes blind."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.py")
            with open(p, "w", encoding="utf-8") as f:
                f.write('WEB_PANELS = [{"id": "desk_device", "title": "T", "layout": []}]\n')
            self.assertEqual(_declared_panel_ids(p), {"desk_device"})


if __name__ == "__main__":
    unittest.main()
