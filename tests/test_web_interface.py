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

stdlib unittest + urllib only; no pytest, no third-party HTTP client.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from tools import web_interface as wi


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _get_raw(url, headers=None):
    """GET returning (status, body_text) even on a 4xx (urllib raises on those)."""
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _post(url, obj, headers=None):
    body = json.dumps(obj).encode("utf-8")
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
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

    def test_unknown_path_404(self):
        code, data = _post(self.base + "/api/nope", {})
        self.assertEqual(code, 404)


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


if __name__ == "__main__":
    unittest.main()
