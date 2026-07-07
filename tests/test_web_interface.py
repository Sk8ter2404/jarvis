"""Tests for tools/web_interface.py — the stdlib HTTP dashboard + inject channel.

Everything here runs on HEADLESS LINUX CI: the server binds 127.0.0.1:0 (an
ephemeral port the OS picks), the inject file / log dir / hud_state file are all
pointed at a per-test temp dir, and the reply-wait is stubbed so no live JARVIS
log is tailed. No win32, no real GPU, no real JARVIS — every source degrades
gracefully when its file is absent.

Coverage:
  • create_server binds an ephemeral port and /api/status returns JSON.
  • /api/say writes the injected command to the (temp) inject file in the exact
    shape the monolith's _drain_injected_command consumes (a JSON list of
    {"text": ...} dicts), and returns the stubbed reply.
  • inject_command appends (doesn't clobber) and stays valid JSON.
  • /api/log/tail returns the tail of the newest session log, and an empty tail
    (with running=False) when no log exists.
  • Token required: with a token set, an API call without it is 401 and with it
    is 200 (header AND query-param forms).
  • SECURITY: create_server REFUSES a non-local bind with an empty token
    (InsecureBindError) and ALLOWS a local bind with no token.
  • build_status is graceful when hud_state / log / gpu are all absent.
  • DASHBOARD ENHANCEMENTS (v2 web controls): the page carries the quick-action
    button markup (QUICK_ACTIONS array + preset labels/commands) and the
    auto-refresh toggle — with AND without a token set; POSTing a preset command
    ("mouse control on") to /api/say still round-trips a stubbed reply.
  • STATUS ENHANCEMENTS: /api/status always carries a JSON-valid ``uptime`` field
    (None with no log, a float derived from the log's first timestamp otherwise);
    ``air_mouse`` is present ONLY when the skill module is loaded in-process
    (simulated via sys.modules) and OMITTED otherwise.

stdlib unittest + urllib only; no pytest, no third-party HTTP client.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from tools import web_interface as wi


def _urlopen_retry(req, timeout=5, attempts=6):
    """urlopen that RETRIES a connection-level failure but NOT an HTTPError.

    A freshly-started 127.0.0.1:0 ThreadingHTTPServer can reset the very first
    connect if its serve_forever thread hasn't reached accept() yet — on Windows
    this surfaces as WinError 10053/10054 (ConnectionAborted/Reset) wrapped in a
    URLError, and flaked test_unknown_path_404 ~1 run in 4. An HTTPError, by
    contrast, IS a real HTTP response (e.g. a 404) and must propagate unchanged.
    2026-07-07 flaky-test fix."""
    last = None
    for i in range(attempts):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError:
            raise                                  # real status → caller handles
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last = e
            time.sleep(0.05 * (i + 1))
    raise last


def _wait_server_ready(host, port, timeout=3.0):
    """Block until the server ACCEPTS a TCP connect (or timeout) so no request
    fires before serve_forever is live. Belt-and-braces with _urlopen_retry."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.02)
    return False


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with _urlopen_retry(req, timeout=5) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _get_raw(url, headers=None):
    """GET returning (status, body_text) even on a 4xx (urllib raises on those)."""
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with _urlopen_retry(req, timeout=5) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _post(url, obj, headers=None):
    body = json.dumps(obj).encode("utf-8")
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    try:
        with _urlopen_retry(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


class _ServerBase(unittest.TestCase):
    """Spin up a real server on 127.0.0.1:0 in a temp dir; tear it down cleanly."""

    token = ""
    reply_reader = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.inject_path = os.path.join(self.d, "injected_commands.json")
        self.log_dir = os.path.join(self.d, "logs")
        self.hud_path = os.path.join(self.d, "hud_state.json")
        os.makedirs(self.log_dir, exist_ok=True)
        self.httpd = wi.create_server(
            bind="127.0.0.1", port=0, token=self.token,
            inject_path=self.inject_path, log_dir=self.log_dir,
            hud_state_path=self.hud_path, reply_reader=self.reply_reader,
        )
        self.host, self.port = self.httpd.server_address[:2]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = wi.serve_in_thread(self.httpd)
        _wait_server_ready(self.host, self.port)   # no request before accept() is live

    def tearDown(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass
        try:
            self.thread.join(timeout=3)
        except Exception:
            pass
        self.tmp.cleanup()


class StatusEndpointTests(_ServerBase):
    def test_status_returns_json_with_expected_keys(self):
        code, data = _get(self.base + "/api/status")
        self.assertEqual(code, 200)
        for key in ("version", "state", "running", "gpu_lines", "ts"):
            self.assertIn(key, data)
        # No live JARVIS/log/hud in the temp dir → graceful defaults.
        self.assertFalse(data["running"])
        self.assertIsInstance(data["gpu_lines"], list)

    def test_root_serves_dashboard_html(self):
        code, body = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        self.assertIn("J.A.R.V.I.S", body)
        self.assertIn("/api/status", body)   # the page polls it

    def test_dashboard_has_quick_action_buttons(self):
        # The quick-action row is data-driven from a QUICK_ACTIONS JS array; assert
        # both the array and a representative preset (label + the exact phrase it
        # POSTs) are present so a broken f-string / renamed preset is caught.
        code, body = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        self.assertIn("QUICK_ACTIONS", body)
        self.assertIn('id="actions"', body)           # the container the buttons render into
        self.assertIn("Arm mouse control", body)       # a preset label
        self.assertIn("mouse control on", body)        # the phrase that preset POSTs
        self.assertIn("system status", body)           # the "what's my status" preset phrase

    def test_dashboard_has_autorefresh_toggle(self):
        # The auto-refresh checkbox lets the user freeze polling; assert its element
        # and the gating helper are in the page.
        code, body = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        self.assertIn('id="autorefresh"', body)
        self.assertIn("auto-refresh", body)

    def test_unknown_path_404(self):
        code, data = _post(self.base + "/api/nope", {})
        self.assertEqual(code, 404)

    def test_status_carries_uptime_field(self):
        # uptime is a first-class /api/status field — None with no log here, but the
        # KEY must always be present + JSON-valid so the client can rely on it.
        code, data = _get(self.base + "/api/status")
        self.assertEqual(code, 200)
        self.assertIn("uptime", data)
        self.assertIsNone(data["uptime"])              # no session log in the temp dir
        # air_mouse is OMITTED when the skill isn't loaded in-process (the default
        # in headless CI) — its ABSENCE is meaningful, so assert it's not there.
        self.assertNotIn("air_mouse", data)


class LogTailTests(_ServerBase):
    def test_tail_empty_when_no_log(self):
        code, data = _get(self.base + "/api/log/tail?lines=10")
        self.assertEqual(code, 200)
        self.assertEqual(data["lines"], [])
        self.assertFalse(data["running"])

    def test_tail_returns_recent_lines(self):
        lg = os.path.join(self.log_dir, "session_2026-07-07_00-00-00.log")
        with open(lg, "w", encoding="utf-8") as f:
            f.write("\n".join(f"line {i}" for i in range(100)) + "\n")
        code, data = _get(self.base + "/api/log/tail?lines=5")
        self.assertEqual(code, 200)
        self.assertEqual(data["lines"], [f"line {i}" for i in range(95, 100)])
        # Freshly written -> running heuristic is True.
        self.assertTrue(data["running"])


class SayInjectTests(_ServerBase):
    # Stub the reply-wait so no real log is tailed; assert the inject file write.
    reply_reader = staticmethod(
        lambda text, log_dir, timeout: {"status": "ok", "lines": [f"JARVIS: echo {text}"]}
    )

    def test_say_writes_inject_file_in_drain_shape(self):
        code, data = _post(self.base + "/api/say", {"text": "what time is it"})
        self.assertEqual(code, 200)
        self.assertTrue(data["accepted"])
        self.assertIn("echo what time is it", data["reply"])
        # The inject file must be a JSON LIST of dicts with a "text" key — exactly
        # what bobert_companion._drain_injected_command pops.
        with open(self.inject_path, encoding="utf-8") as f:
            items = json.load(f)
        self.assertIsInstance(items, list)
        self.assertEqual(items[-1]["text"], "what time is it")

    def test_say_empty_text_400(self):
        code, data = _post(self.base + "/api/say", {"text": "   "})
        self.assertEqual(code, 400)

    def test_say_appends_not_clobbers(self):
        _post(self.base + "/api/say", {"text": "first"})
        _post(self.base + "/api/say", {"text": "second"})
        with open(self.inject_path, encoding="utf-8") as f:
            items = json.load(f)
        self.assertEqual([i["text"] for i in items], ["first", "second"])

    def test_quick_action_preset_command_round_trips(self):
        # A quick-action button POSTs a PRESET phrase to /api/say exactly as the
        # typed form does. Drive the same endpoint with a preset ("mouse control
        # on") and assert it injects in drain-shape and returns the stubbed reply —
        # proving the button path (which shares sendCommand) still works.
        code, data = _post(self.base + "/api/say", {"text": "mouse control on"})
        self.assertEqual(code, 200)
        self.assertTrue(data["accepted"])
        self.assertIn("echo mouse control on", data["reply"])
        with open(self.inject_path, encoding="utf-8") as f:
            items = json.load(f)
        self.assertEqual(items[-1]["text"], "mouse control on")


class InjectHelperTests(unittest.TestCase):
    """Unit-level: inject_command mirrors the driver's atomic append."""

    def test_inject_appends_and_stays_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "injected_commands.json")
            wi.inject_command("a", p)
            wi.inject_command("b", p)
            with open(p, encoding="utf-8") as f:
                items = json.load(f)
            self.assertEqual([i["text"] for i in items], ["a", "b"])

    def test_inject_starts_fresh_when_file_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "injected_commands.json")
            wi.inject_command("solo", p)
            with open(p, encoding="utf-8") as f:
                items = json.load(f)
            self.assertEqual(items[-1]["text"], "solo")

    def test_inject_recovers_from_corrupt_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "injected_commands.json")
            with open(p, "w", encoding="utf-8") as f:
                f.write("}{ not json")
            # Should not raise — corrupt content is discarded, ours is written.
            wi.inject_command("ok", p)
            with open(p, encoding="utf-8") as f:
                items = json.load(f)
            self.assertEqual(items, [{"text": items[0]["text"], "ts": items[0]["ts"]}])
            self.assertEqual(items[0]["text"], "ok")


class SecurityBindTests(unittest.TestCase):
    """The whole point: a non-local bind with no token must be refused."""

    def test_non_local_bind_empty_token_refused(self):
        with self.assertRaises(wi.InsecureBindError):
            wi.create_server(bind="0.0.0.0", port=0, token="")

    def test_non_local_bind_lan_ip_empty_token_refused(self):
        with self.assertRaises(wi.InsecureBindError):
            wi.create_server(bind="192.168.1.50", port=0, token="")

    def test_local_bind_no_token_allowed(self):
        httpd = wi.create_server(bind="127.0.0.1", port=0, token="")
        try:
            self.assertTrue(httpd.config["local_bind"])
        finally:
            httpd.server_close()

    def test_non_local_bind_with_token_allowed(self):
        # We don't actually bind 0.0.0.0 in CI (may be restricted); a token means
        # create_server won't raise — bind to loopback to prove construction path.
        httpd = wi.create_server(bind="127.0.0.1", port=0, token="secret")
        try:
            self.assertEqual(httpd.config["token"], "secret")
        finally:
            httpd.server_close()

    def test_refuses_to_cobind_an_actively_served_port(self):
        # The Windows SO_REUSEADDR footgun guard: if a server is already LISTENing
        # on the port, create_server must refuse (OSError) rather than silently
        # co-bind and split connections into a hang.
        first = wi.create_server(bind="127.0.0.1", port=0, token="")
        thread = wi.serve_in_thread(first)
        try:
            port = first.server_address[1]        # the real ephemeral port
            with self.assertRaises(OSError):
                wi.create_server(bind="127.0.0.1", port=port, token="")
        finally:
            first.shutdown()
            first.server_close()
            thread.join(timeout=2)

    def test_free_port_probe_is_false(self):
        # A concrete port with no listener probes False → bind proceeds. Uses a
        # port we bind+immediately release so it's almost certainly free.
        tmp = wi.create_server(bind="127.0.0.1", port=0, token="")
        port = tmp.server_address[1]
        tmp.server_close()                        # release it (no listener now)
        self.assertFalse(wi._port_actively_served("127.0.0.1", port))

    def test_ephemeral_port_zero_skips_the_probe(self):
        # port 0 must never be probed (it's "pick any free port") — two ephemeral
        # servers coexist fine.
        a = wi.create_server(bind="127.0.0.1", port=0, token="")
        b = wi.create_server(bind="127.0.0.1", port=0, token="")
        try:
            self.assertNotEqual(a.server_address[1], b.server_address[1])
        finally:
            a.server_close()
            b.server_close()

    def test_is_local_bind_classification(self):
        self.assertTrue(wi.is_local_bind("127.0.0.1"))
        self.assertTrue(wi.is_local_bind("localhost"))
        self.assertTrue(wi.is_local_bind("::1"))
        self.assertFalse(wi.is_local_bind("0.0.0.0"))
        self.assertFalse(wi.is_local_bind("192.168.1.10"))
        self.assertFalse(wi.is_local_bind(""))


class TokenAuthTests(_ServerBase):
    token = "s3cr3t"
    reply_reader = staticmethod(lambda text, log_dir, timeout: {"status": "ok", "lines": []})

    def test_api_without_token_is_401(self):
        code, _ = _get_raw(self.base + "/api/status")
        self.assertEqual(code, 401)

    def test_api_with_header_token_ok(self):
        code, data = _get(self.base + "/api/status",
                          headers={"X-Auth-Token": self.token})
        self.assertEqual(code, 200)
        self.assertIn("version", data)

    def test_api_with_bearer_token_ok(self):
        code, data = _get(self.base + "/api/status",
                          headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(code, 200)

    def test_api_with_query_token_ok(self):
        code, data = _get(self.base + f"/api/status?token={self.token}")
        self.assertEqual(code, 200)

    def test_api_with_wrong_token_401(self):
        code, _ = _get_raw(self.base + "/api/status?token=nope")
        self.assertEqual(code, 401)

    def test_say_without_token_401(self):
        code, _ = _post(self.base + "/api/say", {"text": "hi"})
        self.assertEqual(code, 401)

    def test_page_allowed_token_free_on_local_bind(self):
        # Convenience: on a LOCAL bind the dashboard PAGE loads without a token
        # (the JS then supplies it on API calls). Only API routes are gated here.
        code, body = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        self.assertIn("J.A.R.V.I.S", body)

    def test_page_renders_enhancements_with_token_set(self):
        # With a token configured the page still renders fully (and bakes the token
        # into the JS). Assert the enhancements survive the token path: quick-action
        # markup + the auto-refresh toggle are present, and the token is embedded.
        code, body = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        self.assertIn("QUICK_ACTIONS", body)
        self.assertIn("Arm mouse control", body)
        self.assertIn('id="autorefresh"', body)
        self.assertIn(self.token, body)                # token baked into the page JS


class BuildStatusGracefulTests(unittest.TestCase):
    """build_status must never raise when every source is missing."""

    def test_status_with_all_sources_absent(self):
        with tempfile.TemporaryDirectory() as d:
            status = wi.build_status(os.path.join(d, "nope.json"),
                                     os.path.join(d, "no_logs"))
            self.assertEqual(status["state"], "Unknown")
            self.assertFalse(status["running"])
            self.assertIsInstance(status["gpu_lines"], list)

    def test_status_reads_hud_state_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            hud = os.path.join(d, "hud_state.json")
            with open(hud, "w", encoding="utf-8") as f:
                json.dump({"state": "Standby", "now_playing": "jazz"}, f)
            status = wi.build_status(hud, os.path.join(d, "logs"))
            self.assertEqual(status["state"], "Standby")
            self.assertEqual(status["now_playing"], "jazz")


class WaitForReplyTests(unittest.TestCase):
    """The default reply reader tails a real (temp) log; assert its verdicts."""

    def test_no_log_returns_no_log_status(self):
        with tempfile.TemporaryDirectory() as d:
            res = wi.wait_for_reply("hello", os.path.join(d, "logs"), timeout=1.0)
            self.assertEqual(res["status"], "no_log")

    def test_captures_reply_lines_after_inject_anchor(self):
        with tempfile.TemporaryDirectory() as d:
            log_dir = os.path.join(d, "logs")
            os.makedirs(log_dir)
            lg = os.path.join(log_dir, "session_2026-07-07_00-00-00.log")
            # Seed a pre-existing line so wait_for_reply starts at EOF.
            with open(lg, "w", encoding="utf-8") as f:
                f.write("[00:00:00] boot\n")

            # Append the inject anchor + a reply on a background timer so the
            # poll loop sees them appear.
            def _append():
                time.sleep(0.3)
                with open(lg, "a", encoding="utf-8") as f:
                    f.write("[00:00:01]   [inject] what time is it\n")
                    f.write("[00:00:02]   JARVIS: it is noon, sir\n")
            import threading
            threading.Thread(target=_append, daemon=True).start()
            res = wi.wait_for_reply("what time is it", log_dir, timeout=5.0)
            self.assertEqual(res["status"], "ok")
            self.assertTrue(any("noon" in ln for ln in res["lines"]))


class UptimeTests(unittest.TestCase):
    """_uptime_seconds derives a same-day delta from the log's first timestamp."""

    def test_uptime_none_when_no_log(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(wi._uptime_seconds(os.path.join(d, "logs")))

    def test_uptime_none_when_no_timestamp_in_head(self):
        with tempfile.TemporaryDirectory() as d:
            ld = os.path.join(d, "logs")
            os.makedirs(ld)
            lg = os.path.join(ld, "session_2026-07-07_00-00-00.log")
            with open(lg, "w", encoding="utf-8") as f:
                f.write("no timestamps here\njust plain lines\n")
            self.assertIsNone(wi._uptime_seconds(ld))

    def test_uptime_derived_from_first_timestamp(self):
        # Seed a first line stamped ~2 minutes ago; uptime should be ~120s. We use
        # the LOCAL clock (matching _uptime_seconds) so this holds regardless of TZ.
        with tempfile.TemporaryDirectory() as d:
            ld = os.path.join(d, "logs")
            os.makedirs(ld)
            t = time.localtime(time.time() - 120)
            stamp = "[%02d:%02d:%02d]" % (t.tm_hour, t.tm_min, t.tm_sec)
            lg = os.path.join(ld, "session_2026-07-07_00-00-00.log")
            with open(lg, "w", encoding="utf-8") as f:
                f.write("boot banner (no ts)\n")
                f.write(stamp + " loop starting\n")
            up = wi._uptime_seconds(ld)
            self.assertIsNotNone(up)
            # Allow a wide window for test-runner slowness / a midnight-rollover
            # clamp (which would read 0.0); the point is it's a sane float, not None.
            self.assertTrue(up == 0.0 or (100 <= up <= 140), f"unexpected uptime {up}")


class AirMouseStatusTests(unittest.TestCase):
    """_air_mouse_status reads the skill via sys.modules (no import) and OMITS the
    field when the skill isn't loaded — mirroring bobert's preview reader."""

    def tearDown(self):
        import sys as _sys
        _sys.modules.pop("skill_kinect_air_mouse", None)   # never leak the fake

    def test_none_when_skill_not_loaded(self):
        import sys as _sys
        _sys.modules.pop("skill_kinect_air_mouse", None)
        self.assertIsNone(wi._air_mouse_status())

    def test_reads_armed_engaged_when_skill_loaded(self):
        import sys as _sys
        import types
        fake = types.ModuleType("skill_kinect_air_mouse")
        fake.get_air_mouse_state = lambda: {  # type: ignore[attr-defined]
            "engaged": True, "armed": True, "grip": "open", "ts": 0.0}
        _sys.modules["skill_kinect_air_mouse"] = fake
        self.assertEqual(wi._air_mouse_status(), {"armed": True, "engaged": True})

    def test_build_status_includes_air_mouse_when_loaded(self):
        import sys as _sys
        import types
        fake = types.ModuleType("skill_kinect_air_mouse")
        fake.get_air_mouse_state = lambda: {  # type: ignore[attr-defined]
            "engaged": False, "armed": True}
        _sys.modules["skill_kinect_air_mouse"] = fake
        with tempfile.TemporaryDirectory() as d:
            s = wi.build_status(os.path.join(d, "nope.json"),
                                os.path.join(d, "logs"))
            # JSON-valid and carries the trimmed air_mouse dict.
            json.dumps(s)
            self.assertEqual(s["air_mouse"], {"armed": True, "engaged": False})

    def test_build_status_omits_air_mouse_when_not_loaded(self):
        import sys as _sys
        _sys.modules.pop("skill_kinect_air_mouse", None)
        with tempfile.TemporaryDirectory() as d:
            s = wi.build_status(os.path.join(d, "nope.json"),
                                os.path.join(d, "logs"))
            self.assertNotIn("air_mouse", s)


if __name__ == "__main__":
    unittest.main()
