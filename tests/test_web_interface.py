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
  • SETTINGS CONTROL PANEL (web-settings-panel): GET /api/settings returns the
    schema + current values (WAKE_WORD_AUTOSTART present with a value); POST a
    bool + an enum persists to the TEMP user_settings.json (preserving other keys)
    and round-trips; an unknown key or a bad enum value → 400; a settings write
    requires the token when one is set (401 without); and the dashboard HTML
    carries the Settings section + the prominent wake-word control.

stdlib unittest + urllib only; no pytest, no third-party HTTP client.
"""
from __future__ import annotations

import ast
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
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


def _bounded(call, timeout, what):
    """Run ``call`` on a throwaway daemon thread and give up after ``timeout``.

    Returns True if it finished. Never raises: the whole point is to survive a
    call that cannot be interrupted."""
    box = {"done": False}

    def _run():
        try:
            call()
        except Exception:
            pass
        box["done"] = True

    t = threading.Thread(target=_run, daemon=True, name=f"test-{what}")
    t.start()
    t.join(timeout)
    if not box["done"]:
        print(f"  [test-web-interface] {what}() wedged after {timeout}s "
              f"— abandoning it so the suite keeps moving")
    return box["done"]


def _stop_server(httpd, thread=None, *, shutdown_timeout=5.0,
                 close_timeout=5.0, join_timeout=3.0):
    """Tear a REAL served ThreadingHTTPServer down with NO unbounded wait.

    THE POINT. Both stdlib calls a fixture reaches for here block forever in a
    reachable state, and one of them has already frozen a JARVIS ci_sim run:

      * ``BaseServer.shutdown()`` sets a flag and then waits on an Event that
        ONLY ``serve_forever()`` sets as it exits. A serve thread that never
        started (spawn race) or that already died leaves that Event unset and
        the wait is FOREVER. Live 2026-07-12: py-spy caught exactly that —
        tearDown -> shutdown -> Event.wait — mid-suite, with no test id and no
        traceback, because a hang produces no unittest output at all.
      * ``ThreadingMixIn.server_close()`` joins EVERY handler thread with no
        timeout. ``ThreadingHTTPServer`` sets ``daemon_threads = True`` but
        leaves ``block_on_close`` True, so handler threads are still tracked
        and still joined (verified on this box, CPython 3.14.4). And
        tools/web_interface.py deliberately parks a worker for the entire life
        of an ``/api/camera-stream`` MJPEG connection, and for up to
        ``_REPLY_TIMEOUT_MAX`` (120s) on a reply-wait — so this join can
        outlive by minutes the test that opened the connection.

    ``skills/web_interface.py::_stop()`` time-boxes the FIRST of those. That fix
    landed only there: this module — which starts a real server for every one of
    its ~200 tests and is the largest single module in the suite — kept calling
    the raw pair inside a ``try/except Exception`` that catches nothing, because
    a block is not an exception. That is the stale-duplicate shape the house
    rule warns about, so the time-box now lives in ONE helper that every fixture
    in this file goes through, and
    ``NoUnboundedServerTeardownTests`` fails if a new fixture skips it."""
    if httpd is not None:
        _bounded(httpd.shutdown, shutdown_timeout, "shutdown")
        # server_close() runs even when shutdown() wedged — it is what frees the
        # listening socket, and an abandoned ephemeral port costs nothing.
        _bounded(httpd.server_close, close_timeout, "server_close")
    if thread is not None:
        try:
            thread.join(timeout=join_timeout)
        except Exception:
            pass


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
        # POST /api/settings writes here — a THROWAWAY file so a settings write in
        # a test can NEVER touch the real data/user_settings.json (the same safety
        # contract as inject_path/log_dir/hud_state_path). It doesn't exist yet;
        # _write_settings creates it on first write.
        self.user_settings_path = os.path.join(self.d, "user_settings.json")
        # The control-panel sources are ALSO pointed at throwaway temp paths so a
        # test can never read the real camera frame / action index and the 404
        # camera-preview case is deterministic (the file simply doesn't exist).
        self.camera_preview_path = os.path.join(self.d, ".hud_camera_preview.jpg")
        self.action_index_path = os.path.join(self.d, "ACTION_INDEX.md")
        os.makedirs(self.log_dir, exist_ok=True)
        self.httpd = wi.create_server(
            bind="127.0.0.1", port=0, token=self.token,
            inject_path=self.inject_path, log_dir=self.log_dir,
            hud_state_path=self.hud_path,
            user_settings_path=self.user_settings_path,
            camera_preview_path=self.camera_preview_path,
            action_index_path=self.action_index_path,
            reply_reader=self.reply_reader,
        )
        self.host, self.port = self.httpd.server_address[:2]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = wi.serve_in_thread(self.httpd)
        _wait_server_ready(self.host, self.port)   # no request before accept() is live

    def tearDown(self):
        # Time-boxed: see _stop_server. The raw pair that used to be here is a
        # pair of UNBOUNDED waits, and one of them froze a whole ci_sim run.
        _stop_server(self.httpd, self.thread)
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


class SettingsEndpointTests(_ServerBase):
    """The FULL settings control panel: GET the schema+values, POST changes that
    persist to the temp user_settings.json and round-trip on the next GET, and the
    validation 400s (unknown key / bad enum).

    2026-07-21 fix: GET /api/settings now overlays the saved user_settings.json
    (the exact file POST writes) on top of the boot-time core.config snapshot for
    keys PRESENT in the file, with an honest ``pending_restart`` flag when the
    file diverges from the live constant — the panel used to echo the stale
    snapshot, so every save visibly reverted the moment the panel re-fetched."""

    def test_get_returns_schema_with_wake_word_value(self):
        # GET /api/settings serves every persisted knob with its current value.
        # Assert a KNOWN key (WAKE_WORD_AUTOSTART — the wake-word toggle) is present
        # with a value + type, and that the payload is grouped-able by tab.
        code, data = _get(self.base + "/api/settings")
        self.assertEqual(code, 200)
        self.assertIn("settings", data)
        self.assertIn("tabs", data)
        self.assertIn("note", data)
        by_name = {it["name"]: it for it in data["settings"]}
        self.assertIn("WAKE_WORD_AUTOSTART", by_name)
        wake = by_name["WAKE_WORD_AUTOSTART"]
        self.assertEqual(wake["type"], "bool")
        self.assertIn("value", wake)          # current effective value present
        self.assertIn("default", wake)
        self.assertEqual(wake["tab"], "voice")
        # The Alexa-style wake-word-mode knob the banner switch drives is also here.
        self.assertIn("START_IN_STANDBY", by_name)
        # Read-only integration STATUS rows must NOT leak into a write panel.
        self.assertFalse(any(n.startswith("_") for n in by_name))

    def test_post_bool_persists_and_reflects_on_next_get(self):
        # POST a bool (the wake-word toggle) → it lands in the temp file AND the
        # next GET reports the SAVED value (2026-07-21 fix: the payload used to
        # echo the boot-time core.config snapshot, so the banner toggle visibly
        # flipped back the moment the panel re-fetched after a save).
        code, data = _post(self.base + "/api/settings",
                           {"name": "WAKE_WORD_AUTOSTART", "value": True})
        self.assertEqual(code, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["applied"]["WAKE_WORD_AUTOSTART"], True)
        self.assertIn("note", data)           # the honest restart caveat
        # Persisted to the (temp) file in the exact key the config reader expects.
        with open(self.user_settings_path, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertIs(saved["WAKE_WORD_AUTOSTART"], True)
        # The next GET reports the FILE value — the round-trip the panel renders.
        # Pin the live constant to False so the divergence (and its honest
        # pending_restart flag) is deterministic regardless of core.config state.
        orig = wi._config_value
        wi._config_value = (lambda key, default:
                            False if key == "WAKE_WORD_AUTOSTART"
                            else orig(key, default))
        try:
            code, data = _get(self.base + "/api/settings")
        finally:
            wi._config_value = orig
        self.assertEqual(code, 200)
        by_name = {it["name"]: it for it in data["settings"]}
        wake = by_name["WAKE_WORD_AUTOSTART"]
        self.assertIs(wake["value"], True)            # the SAVED value, not the snapshot
        self.assertIs(wake["pending_restart"], True)  # honest: the loop still runs False
        # And the round-trip holds for a SECOND write too (toggle back off).
        _post(self.base + "/api/settings",
              {"name": "WAKE_WORD_AUTOSTART", "value": False})
        code, data = _get(self.base + "/api/settings")
        self.assertEqual(code, 200)
        by_name = {it["name"]: it for it in data["settings"]}
        self.assertIs(by_name["WAKE_WORD_AUTOSTART"]["value"], False)
        with open(self.user_settings_path, encoding="utf-8") as f:
            self.assertIs(json.load(f)["WAKE_WORD_AUTOSTART"], False)

    def test_only_file_present_keys_are_overlaid(self):
        # The overlay must NOT route through settings_window.load_settings()
        # (which backfills every missing key with the SCHEMA default): a knob
        # ABSENT from the file must keep reporting the LIVE core.config constant
        # even when that constant differs from the schema default.
        _post(self.base + "/api/settings",
              {"name": "WAKE_WORD_AUTOSTART", "value": True})  # file has ONLY this key
        sentinel = "SENTINEL-LIVE-VALUE-31"
        orig = wi._config_value
        wi._config_value = (lambda key, default:
                            sentinel if key == "TTS_VOICE"
                            else orig(key, default))
        try:
            code, data = _get(self.base + "/api/settings")
        finally:
            wi._config_value = orig
        self.assertEqual(code, 200)
        by_name = {it["name"]: it for it in data["settings"]}
        # Unsaved key → the live constant wins (NOT the schema default).
        self.assertEqual(by_name["TTS_VOICE"]["value"], sentinel)
        # And no divergence flag on a row nothing was saved for.
        self.assertNotIn("pending_restart", by_name["TTS_VOICE"])

    def test_saved_secret_reflects_is_set_without_leaking(self):
        # Saving a token via the panel must flip is_set on the next GET (the
        # file overlay feeds the redaction block) while the value stays "".
        secret = "panel-saved-token-9000"
        code, data = _post(self.base + "/api/settings",
                           {"name": "WEB_INTERFACE_TOKEN", "value": secret})
        self.assertEqual(code, 200)
        self.assertTrue(data["ok"])
        code, data = _get(self.base + "/api/settings")
        self.assertEqual(code, 200)
        by_name = {it["name"]: it for it in data["settings"]}
        row = by_name["WEB_INTERFACE_TOKEN"]
        self.assertEqual(row["value"], "")            # still redacted
        self.assertTrue(row.get("secret"))
        self.assertTrue(row.get("is_set"))            # the file-saved token counts
        self.assertNotIn(secret, json.dumps(data))    # never leaks anywhere

    def test_post_enum_persists(self):
        # An enum value in-choices persists coerced.
        code, data = _post(self.base + "/api/settings",
                           {"name": "TTS_BACKEND", "value": "pyttsx3"})
        self.assertEqual(code, 200)
        self.assertTrue(data["ok"])
        with open(self.user_settings_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["TTS_BACKEND"], "pyttsx3")

    def test_post_batch_settings_form(self):
        # The {settings: {name: value, ...}} batch form applies all at once.
        code, data = _post(self.base + "/api/settings",
                           {"settings": {"WAKE_WORD_AUTOSTART": True,
                                         "START_IN_STANDBY": True}})
        self.assertEqual(code, 200)
        self.assertEqual(set(data["applied"]),
                         {"WAKE_WORD_AUTOSTART", "START_IN_STANDBY"})
        with open(self.user_settings_path, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertIs(saved["WAKE_WORD_AUTOSTART"], True)
        self.assertIs(saved["START_IN_STANDBY"], True)

    def test_post_preserves_other_keys(self):
        # A pre-existing unrelated key in the file survives a targeted merge.
        with open(self.user_settings_path, "w", encoding="utf-8") as f:
            json.dump({"SOME_FUTURE_KEY": "keepme"}, f)
        _post(self.base + "/api/settings",
              {"name": "WAKE_WORD_AUTOSTART", "value": True})
        with open(self.user_settings_path, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["SOME_FUTURE_KEY"], "keepme")   # untouched
        self.assertIs(saved["WAKE_WORD_AUTOSTART"], True)

    def test_post_unknown_key_400(self):
        code, data = _post(self.base + "/api/settings",
                           {"name": "NOT_A_REAL_KEY", "value": 1})
        self.assertEqual(code, 400)
        self.assertIn("unknown", data["error"].lower())

    def test_post_bad_enum_value_400(self):
        code, data = _post(self.base + "/api/settings",
                           {"name": "TTS_BACKEND", "value": "definitely-not-a-backend"})
        self.assertEqual(code, 400)
        self.assertIn("not one of", data["error"].lower())
        # A rejected write must NOT create/alter the file.
        self.assertFalse(os.path.exists(self.user_settings_path))

    def test_post_empty_body_400(self):
        code, data = _post(self.base + "/api/settings", {})
        self.assertEqual(code, 400)

    def test_dashboard_has_settings_section_and_wake_control(self):
        # The page must carry the Settings view markup + the prominent wake-word
        # control so a broken f-string / renamed element is caught.
        code, body = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        self.assertIn("viewSettings", body)          # the settings section
        self.assertIn("navSettings", body)           # the nav toggle
        self.assertIn('id="wakeToggle"', body)       # the prominent wake-word switch
        self.assertIn("Wake-word mode", body)        # its label
        self.assertIn("/api/settings", body)         # the page calls the endpoint
        self.assertIn("START_IN_STANDBY", body)      # the knob the banner drives


class SettingsTokenTests(_ServerBase):
    """A settings write is POWERFUL, so it MUST require the token when one is set
    (401 without it) — same auth contract as /api/say."""

    token = "s3cr3t"

    def test_get_settings_without_token_401(self):
        code, _ = _get_raw(self.base + "/api/settings")
        self.assertEqual(code, 401)

    def test_post_settings_without_token_401(self):
        code, _ = _post(self.base + "/api/settings",
                        {"name": "WAKE_WORD_AUTOSTART", "value": True})
        self.assertEqual(code, 401)
        # And the write never happened.
        self.assertFalse(os.path.exists(self.user_settings_path))

    def test_post_settings_with_token_ok(self):
        code, data = _post(self.base + "/api/settings",
                           {"name": "WAKE_WORD_AUTOSTART", "value": True},
                           headers={"X-Auth-Token": self.token})
        self.assertEqual(code, 200)
        self.assertTrue(data["ok"])
        with open(self.user_settings_path, encoding="utf-8") as f:
            self.assertIs(json.load(f)["WAKE_WORD_AUTOSTART"], True)

    def test_cross_origin_refused_even_with_token_on_local_bind(self):
        # 2026-07-08 security fix: on a LOCAL bind the token is NOT the boundary —
        # it's served token-free in the dashboard page and baked into its JS, so a
        # DNS-rebinding page could read it. Therefore a foreign Origin is refused
        # even WITH a valid token; the loopback-Host allowlist is the real boundary
        # on a local bind. (On an exposed bind, _authorized requires the token and
        # the guard steps aside — covered by TokenAuthTests.)
        code, _ = _post(self.base + "/api/settings",
                        {"name": "WAKE_WORD_AUTOSTART", "value": True},
                        headers={"X-Auth-Token": self.token,
                                 "Origin": "http://some-app.example"})
        self.assertEqual(code, 403)

    def test_same_origin_with_token_still_ok(self):
        # A same-origin request carrying the token still works (the real dashboard).
        code, data = _post(self.base + "/api/settings",
                           {"name": "WAKE_WORD_AUTOSTART", "value": True},
                           headers={"X-Auth-Token": self.token, "Origin": self.base})
        self.assertEqual(code, 200)
        self.assertTrue(data["ok"])


def _raw_post_status(host, port, path, body_obj, extra_headers=None):
    """Send a raw HTTP/1.1 POST so a test can set an arbitrary Host header (which
    urllib fixes to the URL host), and return just the numeric status code. Used to
    exercise the anti-DNS-rebinding Host check deterministically."""
    extra_headers = extra_headers or {}
    body = json.dumps(body_obj).encode("utf-8")
    host_hdr = extra_headers.get("Host", f"{host}:{port}")
    lines = [f"POST {path} HTTP/1.1", f"Host: {host_hdr}",
             "Content-Type: application/json", f"Content-Length: {len(body)}",
             "Connection: close"]
    for k, v in extra_headers.items():
        if k.lower() != "host":
            lines.append(f"{k}: {v}")
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8") + body
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(raw)
        buf = b""
        while b"\r\n" not in buf:
            chunk = sock.recv(256)
            if not chunk:
                break
            buf += chunk
    head = buf.split(b"\r\n", 1)[0].decode("latin-1").split()
    return int(head[1]) if len(head) > 1 and head[1].isdigit() else 0


def _raw_get_status(host, port, path, extra_headers=None):
    """Send a raw HTTP/1.1 GET with an arbitrary Host header; return the status
    code. Exercises the anti-rebinding Host check on GET routes deterministically."""
    extra_headers = extra_headers or {}
    host_hdr = extra_headers.get("Host", f"{host}:{port}")
    lines = [f"GET {path} HTTP/1.1", f"Host: {host_hdr}", "Connection: close"]
    for k, v in extra_headers.items():
        if k.lower() != "host":
            lines.append(f"{k}: {v}")
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(raw)
        buf = b""
        while b"\r\n" not in buf:
            chunk = sock.recv(256)
            if not chunk:
                break
            buf += chunk
    head = buf.split(b"\r\n", 1)[0].decode("latin-1").split()
    return int(head[1]) if len(head) > 1 and head[1].isdigit() else 0


class CrossOriginGuardTests(_ServerBase):
    """On a token-FREE local bind, state-changing POSTs must refuse a browser-driven
    cross-origin (CSRF) or foreign-Host (DNS-rebinding) request, while leaving
    same-origin and non-browser (no-Origin, loopback-Host) callers untouched."""

    token = ""

    def test_cross_origin_settings_403(self):
        code, data = _post(self.base + "/api/settings",
                           {"name": "WAKE_WORD_AUTOSTART", "value": True},
                           headers={"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        # The write was refused BEFORE touching disk.
        self.assertFalse(os.path.exists(self.user_settings_path))

    def test_cross_origin_say_403(self):
        code, _ = _post(self.base + "/api/say", {"text": "hello"},
                        headers={"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        # Refused before injecting the command.
        self.assertFalse(os.path.exists(self.inject_path))

    def test_same_origin_settings_ok(self):
        # Origin == the server's own origin (the real dashboard) is allowed.
        code, data = _post(self.base + "/api/settings",
                           {"name": "WAKE_WORD_AUTOSTART", "value": True},
                           headers={"Origin": self.base})
        self.assertEqual(code, 200)
        self.assertTrue(data["ok"])

    def test_no_origin_still_ok(self):
        # A non-browser client (curl / PowerShell / the driver) sends no Origin and
        # a loopback Host — unaffected by the guard.
        code, data = _post(self.base + "/api/settings",
                           {"name": "WAKE_WORD_AUTOSTART", "value": True})
        self.assertEqual(code, 200)
        self.assertTrue(data["ok"])

    def test_foreign_host_403_rebinding(self):
        # A rebound request (Host resolves to us but names an attacker host) is
        # refused even without an Origin header.
        code = _raw_post_status(self.host, self.port, "/api/settings",
                                {"name": "WAKE_WORD_AUTOSTART", "value": True},
                                extra_headers={"Host": "evil.example"})
        self.assertEqual(code, 403)

    def test_loopback_host_ok(self):
        # Same raw path, but a legitimate loopback Host → allowed.
        code = _raw_post_status(self.host, self.port, "/api/settings",
                                {"name": "WAKE_WORD_AUTOSTART", "value": True},
                                extra_headers={"Host": f"127.0.0.1:{self.port}"})
        self.assertEqual(code, 200)

    # ── GET routes are host-guarded too (2026-07-08 fix for the read-leak) ──
    def test_get_status_cross_origin_403(self):
        # A rebinding page must not be able to READ /api/status (or the log /
        # settings / system / memory snapshots) via a foreign Origin.
        code, _ = _get_raw(self.base + "/api/status",
                           headers={"Origin": "http://evil.example"})
        self.assertEqual(code, 403)

    def test_get_foreign_host_403_rebinding(self):
        code = _raw_get_status(self.host, self.port, "/api/status",
                               extra_headers={"Host": "evil.example"})
        self.assertEqual(code, 403)

    def test_get_loopback_host_ok(self):
        code = _raw_get_status(self.host, self.port, "/api/status",
                               extra_headers={"Host": f"127.0.0.1:{self.port}"})
        self.assertEqual(code, 200)


class WebSettingsSaveFixTests(_ServerBase):
    """2026-07-08 fixes to the settings write path + /api/say body parsing."""

    # Stub the reply-wait so /api/say returns immediately (no 30s log-tail).
    reply_reader = staticmethod(
        lambda text, log_dir, timeout: {"status": "ok", "lines": []})

    def test_list_setting_saved_as_real_list_not_json_blob(self):
        # The panel POSTs a list setting as a JSON-array STRING; it must round-trip
        # to a real multi-element list, not a 1-element list holding the raw JSON.
        val = ["1password", "bitwarden", "banking"]
        code, data = _post(self.base + "/api/settings",
                           {"name": "SCREENSHOT_PRIVACY_BLOCKLIST",
                            "value": json.dumps(val)})
        self.assertEqual(code, 200)
        self.assertTrue(data["ok"])
        with open(self.user_settings_path, encoding="utf-8") as f:
            saved = json.load(f)["SCREENSHOT_PRIVACY_BLOCKLIST"]
        self.assertEqual(saved, val)                 # NOT ['["1password",...]']

    def test_routing_setting_saved_as_dict_not_reset(self):
        code, data = _post(self.base + "/api/settings",
                           {"name": "MODEL_ROUTING",
                            "value": json.dumps({"chat": "local"})})
        self.assertEqual(code, 200)
        with open(self.user_settings_path, encoding="utf-8") as f:
            saved = json.load(f)["MODEL_ROUTING"]
        self.assertEqual(saved.get("chat"), "local")  # merged, not reset to default

    def test_say_keeps_command_when_timeout_unparseable(self):
        # A non-numeric timeout must NOT drop the valid command as 'empty text'.
        code, data = _post(self.base + "/api/say",
                           {"text": "hello there", "timeout": "soon"})
        self.assertNotEqual(code, 400)               # command was NOT discarded
        self.assertTrue(os.path.exists(self.inject_path))  # it was injected


class HostOfHelperTests(unittest.TestCase):
    """Unit-level: _host_of normalises Host/Origin/Referer header values."""

    def test_host_extraction(self):
        f = wi._Handler._host_of
        self.assertEqual(f("http://localhost:8766/x"), "localhost")
        self.assertEqual(f("127.0.0.1:8766"), "127.0.0.1")
        self.assertEqual(f("https://Evil.Example"), "evil.example")
        self.assertEqual(f("[::1]:8766"), "[::1]")
        self.assertEqual(f(""), "")


class SettingsWriteHelperTests(unittest.TestCase):
    """Unit-level: _write_settings merges + validates without a live server."""

    def test_merge_preserves_and_validates(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "user_settings.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"KEEP": 1}, f)
            applied = wi._write_settings({"WAKE_WORD_AUTOSTART": "yes"}, p)
            # coerce_value maps "yes" → True for a bool knob.
            self.assertIs(applied["WAKE_WORD_AUTOSTART"], True)
            with open(p, encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["KEEP"], 1)
            self.assertIs(saved["WAKE_WORD_AUTOSTART"], True)

    def test_unknown_key_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "user_settings.json")
            with self.assertRaises(wi.SettingsWriteError):
                wi._write_settings({"NOPE": 1}, p)
            # No file created on a rejected write.
            self.assertFalse(os.path.exists(p))

    def test_secret_knob_value_is_redacted_in_get_payload(self):
        # A configured web token must NEVER be echoed back in the settings
        # snapshot — the row is present (so the owner can SET one) but its value
        # is redacted to "" with secret/is_set flags. Patch _config_value so the
        # "live" token is a known secret, then prove it does not appear anywhere.
        orig = wi._config_value
        wi._config_value = (lambda key, default:
                            "REDACT-ME-TOKEN-123"
                            if key == "WEB_INTERFACE_TOKEN" else orig(key, default))
        try:
            payload = wi.build_settings_schema()
        finally:
            wi._config_value = orig
        by_name = {it["name"]: it for it in payload["settings"]}
        self.assertIn("WEB_INTERFACE_TOKEN", by_name)
        row = by_name["WEB_INTERFACE_TOKEN"]
        self.assertEqual(row["value"], "")          # never the real token
        self.assertTrue(row.get("secret"))
        self.assertTrue(row.get("is_set"))          # but "a value is set" is known
        # The secret does not leak into ANY field of ANY row.
        self.assertNotIn("REDACT-ME-TOKEN-123", json.dumps(payload))


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
            _stop_server(first, thread, join_timeout=2)

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
    """_uptime_seconds prefers the FULL boot timestamp in the log's FILENAME
    (session_%Y-%m-%d_%H-%M-%S.log — survives midnight; 2026-07-21 fix) and falls
    back to the date-less same-day H:M:S heuristic only for a log whose name
    doesn't parse (the fallback tests use ``session_x.log`` for exactly that)."""

    def test_uptime_none_when_no_log(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(wi._uptime_seconds(os.path.join(d, "logs")))

    def test_uptime_crosses_midnight_via_filename(self):
        # The 2026-07-21 regression case: a boot ~26h ago (guaranteed to cross
        # midnight) must report ~26h — not the clamped 0.0 / sub-24h value the
        # old same-day H:M:S heuristic produced. The file's first stamped line
        # carries the boot's H:M:S (as a real log would), which would mislead
        # the old code; the filename's full date must win.
        with tempfile.TemporaryDirectory() as d:
            ld = os.path.join(d, "logs")
            os.makedirs(ld)
            boot = time.time() - 26 * 3600
            name = time.strftime("session_%Y-%m-%d_%H-%M-%S.log",
                                 time.localtime(boot))
            t = time.localtime(boot)
            stamp = "[%02d:%02d:%02d]" % (t.tm_hour, t.tm_min, t.tm_sec)
            with open(os.path.join(ld, name), "w", encoding="utf-8") as f:
                f.write(stamp + " loop starting\n")
            up = wi._uptime_seconds(ld)
            self.assertIsNotNone(up)
            self.assertTrue(abs(up - 26 * 3600) <= 120, f"unexpected uptime {up}")

    def test_uptime_filename_wins_without_in_file_timestamps(self):
        # A parseable filename alone is enough — no "[HH:MM:SS]" line needed
        # (the old head-scan-only code returned None here).
        with tempfile.TemporaryDirectory() as d:
            ld = os.path.join(d, "logs")
            os.makedirs(ld)
            boot = time.time() - 2 * 3600
            name = time.strftime("session_%Y-%m-%d_%H-%M-%S.log",
                                 time.localtime(boot))
            with open(os.path.join(ld, name), "w", encoding="utf-8") as f:
                f.write("boot banner only, no timestamps\n")
            up = wi._uptime_seconds(ld)
            self.assertIsNotNone(up)
            self.assertTrue(abs(up - 2 * 3600) <= 120, f"unexpected uptime {up}")

    def test_uptime_none_when_no_timestamp_in_head(self):
        # Unparseable filename AND no stamped line → None (field omitted).
        with tempfile.TemporaryDirectory() as d:
            ld = os.path.join(d, "logs")
            os.makedirs(ld)
            lg = os.path.join(ld, "session_x.log")
            with open(lg, "w", encoding="utf-8") as f:
                f.write("no timestamps here\njust plain lines\n")
            self.assertIsNone(wi._uptime_seconds(ld))

    def test_uptime_derived_from_first_timestamp(self):
        # FALLBACK path: an unparseable (hand-named) log still yields a same-day
        # delta from its first "[HH:MM:SS]" line — ~120s here. We use the LOCAL
        # clock (matching _uptime_seconds) so this holds regardless of TZ.
        with tempfile.TemporaryDirectory() as d:
            ld = os.path.join(d, "logs")
            os.makedirs(ld)
            t = time.localtime(time.time() - 120)
            stamp = "[%02d:%02d:%02d]" % (t.tm_hour, t.tm_min, t.tm_sec)
            lg = os.path.join(ld, "session_x.log")
            with open(lg, "w", encoding="utf-8") as f:
                f.write("boot banner (no ts)\n")
                f.write(stamp + " loop starting\n")
            up = wi._uptime_seconds(ld)
            self.assertIsNotNone(up)
            # Allow a wide window for test-runner slowness / a midnight-rollover
            # clamp (which would read 0.0); the point is it's a sane float, not None.
            self.assertTrue(up == 0.0 or (100 <= up <= 140), f"unexpected uptime {up}")


class DashboardTokenSerializationTests(unittest.TestCase):
    """2026-07-21 fix: the dashboard used html.escape on the auth token before
    baking it into the inline <script>. <script> is an HTML raw-text element —
    character references are NOT decoded there — so a token containing &"'<>
    reached the JS as &amp;/&quot;/… and every API call 401'd, and a backslash
    (html.escape leaves it untouched) produced an unterminated JS string that
    killed the whole inline script. The token is now serialized with json.dumps
    (+ "</" → "<\\/"): these tests prove the page-baked literal decodes to
    EXACTLY the token the server accepts."""

    _TOKEN_RE = re.compile(r"const TOKEN = (.+);")

    def _server(self, token):
        """A real 127.0.0.1:0 server with ``token``, torn down via addCleanup.
        All file sources point at a throwaway temp dir (same safety contract as
        _ServerBase)."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = tmp.name
        log_dir = os.path.join(d, "logs")
        os.makedirs(log_dir, exist_ok=True)
        httpd = wi.create_server(
            bind="127.0.0.1", port=0, token=token,
            inject_path=os.path.join(d, "injected_commands.json"),
            log_dir=log_dir,
            hud_state_path=os.path.join(d, "hud_state.json"),
            user_settings_path=os.path.join(d, "user_settings.json"),
            camera_preview_path=os.path.join(d, ".hud_camera_preview.jpg"),
            action_index_path=os.path.join(d, "ACTION_INDEX.md"),
        )
        thread = wi.serve_in_thread(httpd)

        self.addCleanup(_stop_server, httpd, thread)
        host, port = httpd.server_address[:2]
        _wait_server_ready(host, port)
        return f"http://127.0.0.1:{port}"

    def _token_literal(self, base):
        code, body = _get_raw(base + "/")     # local bind serves the page token-free
        self.assertEqual(code, 200)
        m = self._TOKEN_RE.search(body)
        self.assertIsNotNone(m, "const TOKEN line missing from the page")
        return m.group(1)

    def test_hostile_token_round_trips_and_authenticates(self):
        # Every escaper class at once: &, double quote, single quote, <, >, \.
        token = 'a&b"c\'d<e>f\\g'
        base = self._server(token)
        recovered = json.loads(self._token_literal(base))
        # Under html.escape this held a&amp;b&quot;c… and could never match.
        self.assertEqual(recovered, token)
        # The token the page bakes in IS the token the server accepts.
        code, data = _get(base + "/api/status",
                          headers={"X-Auth-Token": recovered})
        self.assertEqual(code, 200)
        self.assertIn("version", data)

    def test_trailing_backslash_token_stays_valid_js(self):
        # html.escape left "\" untouched, so '…\\' + the closing quote became an
        # escaped quote → unterminated string → SyntaxError killing the script.
        token = "trailing\\"
        base = self._server(token)
        literal = self._token_literal(base)
        self.assertEqual(json.loads(literal), token)   # parses cleanly, round-trips

    def test_script_terminator_token_cannot_break_out(self):
        # "</script>" inside the literal would TERMINATE the raw-text <script>
        # element early; the "</" → "<\\/" replacement must keep the raw
        # terminator out while still decoding to the exact token.
        token = "x</script><script>y"
        base = self._server(token)
        literal = self._token_literal(base)
        self.assertNotIn("</", literal)                # no raw script terminator
        self.assertEqual(json.loads(literal), token)   # yet decodes exactly

    def test_source_never_html_escapes_the_token(self):
        # Stale-duplicate guard for this bug class: the WRONG escaper must not
        # silently return here (or in a copy-pasted dashboard builder inside
        # this module).
        with open(wi.__file__, encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("html.escape(token", src)


class RenderCapDisclosureTests(_ServerBase):
    """2026-07-21 fix: renderActions capped the DOM at 400 rows but only
    admitted it ('· N shown') when a filter was typed — an empty search claimed
    the full total while silently hiding the rest — and renderFacts (the Memory
    tab's copy of the same loop) had the cap with NO qualifier at all. These
    source-scanning invariants on the served page pin the fix for BOTH copies
    AND any future copy-pasted render loop (the stale-duplicate bug class)."""

    _CAP_RE = re.compile(r"shown\s*>=\s*\d+")
    _OVERFLOW_MARKER = "more — refine your search"

    def test_every_dom_cap_advertises_its_truncation(self):
        code, body = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        caps = self._CAP_RE.findall(body)
        # Both known cap sites exist (renderActions + renderFacts) …
        self.assertGreaterEqual(len(caps), 2)
        # … and EVERY cap site (present or future) must emit the visible
        # '…N more — refine your search' overflow row, or this fails.
        self.assertEqual(
            len(caps), body.count(self._OVERFLOW_MARKER),
            "a capped render loop does not advertise its truncation")

    def test_shown_qualifier_is_unconditional(self):
        code, body = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        # The filter-conditional qualifier pattern is gone for good …
        self.assertIsNone(
            re.search(r"\(f\s*\?[^;]*shown", body),
            "the '· N shown' qualifier is still filter-conditional")
        # … and the suffix itself still renders in the count line.
        self.assertIn("' shown'", body)


class GuiOnlyKeyRoundTripTests(unittest.TestCase):
    """2026-07-21 fix: keys with NO core.config constant (the GUI-only
    connection hints OBS_HOST_HINT / OBS_PORT_HINT / HUE_BRIDGE_IP_HINT) always
    rendered blank in the web panel — build_settings_schema sourced every value
    from core.config, so a successful save could never redisplay. Saved values
    must now round-trip from the file."""

    def test_hint_keys_round_trip_from_file(self):
        # The card's exact failure: save a hint, re-build the payload, see it.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "user_settings.json")
            wanted = {"OBS_HOST_HINT": "localhost",
                      "OBS_PORT_HINT": "4455",
                      "HUE_BRIDGE_IP_HINT": "192.168.1.50"}
            wi._write_settings(wanted, p)
            by_name = {it["name"]: it
                       for it in wi.build_settings_schema(p)["settings"]}
            for key, val in wanted.items():
                self.assertEqual(by_name[key]["value"], val, key)

    def test_every_gui_only_schema_key_round_trips(self):
        # Class-level invariant (the stale-duplicate rule, applied to keys):
        # derive EVERY persisted, non-secret schema key with no core.config
        # constant and prove each round-trips through the REAL write path into
        # the GET payload — so a FUTURE GUI-only key added to the schema is
        # covered the day it lands, not when someone notices it renders blank.
        try:
            from core import config as core_config
        except Exception:
            self.skipTest("core.config unavailable — cannot derive the GUI-only set")
        from tools import settings_window as sw
        gui_only = [k for k in sw.SCHEMA
                    if not k.startswith("_")
                    and sw.SCHEMA[k].get("type") != "status"
                    and k not in wi._SECRET_SETTING_KEYS
                    and not hasattr(core_config, k)]
        # The three hint keys are the known members; an empty derivation would
        # mean this invariant tests nothing.
        for known in ("OBS_HOST_HINT", "OBS_PORT_HINT", "HUE_BRIDGE_IP_HINT"):
            self.assertIn(known, gui_only)

        def _distinct(spec, i):
            # A type-appropriate, non-default value for any schema type, so the
            # strict write-path validation accepts it.
            typ = spec.get("type")
            if typ == "bool":
                return not spec.get("default")
            if typ in ("int", "device"):
                return 4200 + i
            if typ == "float":
                return 42.5 + i
            if typ == "enum":
                choices = spec.get("choices") or []
                dflt = spec.get("default")
                others = [c for c in choices if c != dflt]
                return (others or choices or [dflt])[0]
            if typ == "text":
                return [f"distinct-{i}"]
            if typ == "routing":
                return dict(spec.get("default") or {})
            return f"distinct-{i}"

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "user_settings.json")
            updates = {k: _distinct(sw.SCHEMA[k], i)
                       for i, k in enumerate(gui_only)}
            # Through the REAL write path, so what we compare against is the
            # coerced value that actually persisted.
            applied = wi._write_settings(updates, p)
            by_name = {it["name"]: it
                       for it in wi.build_settings_schema(p)["settings"]}
            for k in gui_only:
                self.assertEqual(by_name[k]["value"], applied[k], k)


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


class ControlPanelEndpointTests(_ServerBase):
    """The five new control-panel tabs + their GET endpoints (System / Actions /
    Voice / Camera / Memory). Each is auth-gated like /api/status, read-only, and
    returns a stable JSON shape (or the camera JPEG / 404) even when its live
    source is absent in headless CI."""

    def test_system_returns_expected_keys(self):
        code, data = _get(self.base + "/api/system")
        self.assertEqual(code, 200)
        for key in ("gpus", "cpu_pct", "ram_used_gb", "ram_total_gb",
                    "disks", "version", "uptime", "routing"):
            self.assertIn(key, data)
        self.assertIsInstance(data["gpus"], list)
        self.assertIsInstance(data["disks"], list)

    def test_actions_returns_expected_keys(self):
        # No action index file in the temp dir → graceful empty inventory.
        code, data = _get(self.base + "/api/actions")
        self.assertEqual(code, 200)
        self.assertIn("actions", data)
        self.assertIn("count", data)
        self.assertEqual(data["actions"], [])
        self.assertEqual(data["count"], 0)

    def test_actions_parses_index_table(self):
        # Drop a small ACTION_INDEX.md fixture and prove the parser expands
        # aliases + reads the speak class (VERBATIM / INFORMATIVE / neither).
        with open(self.action_index_path, "w", encoding="utf-8") as f:
            f.write(
                "## Full index\n\n"
                "| action(s) | handler | speak | ex? | tests |\n"
                "|---|---|:--:|:--:|\n"
                "| `get_time` | `core/actions.py:117` | *INFORMATIVE* | | 4 |\n"
                "| `version_info`, `what_version` | `core/actions.py:1206` | **VERBATIM** | | 2 |\n"
                "| `click` | `core/actions.py:1058` | neither | yes | 4 |\n"
            )
        code, data = _get(self.base + "/api/actions")
        self.assertEqual(code, 200)
        self.assertEqual(data["count"], 4)               # aliases expanded
        by_name = {a["name"]: a["spoken"] for a in data["actions"]}
        self.assertEqual(by_name["get_time"], "INFORMATIVE")
        self.assertEqual(by_name["version_info"], "VERBATIM")
        self.assertEqual(by_name["what_version"], "VERBATIM")
        self.assertEqual(by_name["click"], "neither")

    def test_voices_returns_expected_keys(self):
        code, data = _get(self.base + "/api/voices")
        self.assertEqual(code, 200)
        for key in ("profiles", "active", "enabled", "tts_backend", "tts_voice"):
            self.assertIn(key, data)
        self.assertIsInstance(data["profiles"], list)

    def test_memory_returns_expected_keys(self):
        code, data = _get(self.base + "/api/memory")
        self.assertEqual(code, 200)
        for key in ("facts", "episodes", "counts"):
            self.assertIn(key, data)
        self.assertIsInstance(data["facts"], list)
        self.assertIsInstance(data["episodes"], list)
        self.assertIn("facts", data["counts"])
        self.assertIn("episodes", data["counts"])

    def test_camera_preview_404_when_missing(self):
        # No preview file in the temp dir → 404 JSON, not a 500.
        code, body = _get_raw(self.base + "/api/camera-preview")
        self.assertEqual(code, 404)
        self.assertIn("no preview", body)

    def test_camera_preview_serves_fresh_jpeg(self):
        # A freshly-written file is served as image/jpeg with its exact bytes.
        with open(self.camera_preview_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0JPEGDATA")
        req = urllib.request.Request(self.base + "/api/camera-preview")
        with _urlopen_retry(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.headers.get("Content-Type"), "image/jpeg")
            self.assertEqual(r.read(), b"\xff\xd8\xff\xe0JPEGDATA")

    def test_camera_preview_404_when_stale(self):
        # A file older than the stale window is treated as "camera off" → 404.
        with open(self.camera_preview_path, "wb") as f:
            f.write(b"old-frame")
        old = time.time() - (wi._CAMERA_PREVIEW_STALE_S + 5)
        os.utime(self.camera_preview_path, (old, old))
        code, _ = _get_raw(self.base + "/api/camera-preview")
        self.assertEqual(code, 404)

    def test_camera_preview_per_cam_param(self):
        # ?cam=left|right|kinect serves the per-camera tile file next to the
        # main preview; unknown cams 404; a missing per-cam file 404s while
        # OTHER cams still serve (independent tiles). 2026-07-10.
        left = os.path.join(os.path.dirname(self.camera_preview_path),
                            ".hud_camera_preview_left.jpg")
        with open(left, "wb") as f:
            f.write(b"\xff\xd8LEFT")
        req = urllib.request.Request(self.base + "/api/camera-preview?cam=left")
        with _urlopen_retry(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.read(), b"\xff\xd8LEFT")
        code, _ = _get_raw(self.base + "/api/camera-preview?cam=kinect")
        self.assertEqual(code, 404)     # no kinect file written
        code, body = _get_raw(self.base + "/api/camera-preview?cam=bogus")
        self.assertEqual(code, 404)
        self.assertIn("unknown cam", body)

    def test_dashboard_has_percam_tiles(self):
        code, body = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        for tid in ("camLeft", "camRight", "camKinect", "camgrid"):
            self.assertIn(tid, body)

    def test_dashboard_has_new_nav_ids(self):
        # The five new nav buttons + their view sections + endpoint wiring.
        code, body = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        for nid in ("navSystem", "navActions", "navVoice", "navCamera", "navMemory"):
            self.assertIn('id="' + nid + '"', body)
        for vid in ("viewSystem", "viewActions", "viewVoice", "viewCamera", "viewMemory"):
            self.assertIn('id="' + vid + '"', body)
        for ep in ("/api/system", "/api/actions", "/api/voices",
                   "/api/camera-preview", "/api/memory"):
            self.assertIn(ep, body)


class ControlPanelTokenTests(_ServerBase):
    """The new GET endpoints honour the token gate exactly like /api/status."""

    token = "s3cr3t"

    def test_new_endpoints_without_token_401(self):
        for ep in ("/api/system", "/api/actions", "/api/voices",
                   "/api/memory", "/api/camera-preview"):
            code, _ = _get_raw(self.base + ep)
            self.assertEqual(code, 401, ep)

    def test_new_endpoints_with_token_ok(self):
        h = {"X-Auth-Token": self.token}
        for ep in ("/api/system", "/api/actions", "/api/voices", "/api/memory"):
            code, _ = _get(self.base + ep, headers=h)
            self.assertEqual(code, 200, ep)
        # camera-preview: authorized but no file → 404 (NOT 401).
        code, _ = _get_raw(self.base + "/api/camera-preview", headers=h)
        self.assertEqual(code, 404)


class ActionIndexParseTests(unittest.TestCase):
    """Unit-level: _parse_action_index expands aliases + skips junk rows."""

    def test_parse_and_alias_expansion(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ACTION_INDEX.md")
            with open(p, "w", encoding="utf-8") as f:
                # Includes a Summary-style row (no backticks) that must be skipped.
                f.write("| Total registered actions | 508 |\n"
                        "| `a`, `b` | `h:1` | **VERBATIM** | | 1 |\n"
                        "| `c` | `h:2` | neither | | 0 |\n")
            out = wi._parse_action_index(p)
            self.assertEqual(out["count"], 3)
            self.assertEqual({a["name"] for a in out["actions"]}, {"a", "b", "c"})

    def test_missing_file_is_empty(self):
        out = wi._parse_action_index(os.path.join("nope", "ACTION_INDEX.md"))
        self.assertEqual(out, {"actions": [], "count": 0})


class SystemInfoHelperTests(unittest.TestCase):
    """Unit-level: _system_info is always JSON-valid with the full key set even
    when every hardware source is unavailable (the CI degrade path)."""

    def test_shape_is_stable(self):
        with tempfile.TemporaryDirectory() as d:
            s = wi._system_info(os.path.join(d, "nope.json"),
                                os.path.join(d, "logs"))
            json.dumps(s)                     # must be JSON-serialisable
            for key in ("gpus", "cpu_pct", "ram_used_gb", "ram_total_gb",
                        "disks", "version", "uptime", "routing"):
                self.assertIn(key, s)
            self.assertIsInstance(s["gpus"], list)
            self.assertIsInstance(s["disks"], list)


class WebInterfaceBugfix20260708Tests(unittest.TestCase):
    """Unit-level regression tests for the 2026-07-08 bug-fix batch (findings
    #21/#22/#23/#37/#38/#39). Each is standalone — no live server needed."""

    # #38 — _read_hud_state must return a DICT even for valid-but-non-object JSON.
    def test_hud_state_non_object_json_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "hud_state.json")
            with open(p, "w", encoding="utf-8") as f:
                f.write("[1, 2, 3]")          # valid JSON, but a list not a dict
            out = wi._read_hud_state(p)
            self.assertEqual(out, {})
            # And a real object still comes back intact.
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"state": "Idle"}, f)
            self.assertEqual(wi._read_hud_state(p)["state"], "Idle")

    # #23 — tail_log seeks a bounded window instead of reading the whole file.
    def test_tail_log_bounded_window_returns_last_lines(self):
        with tempfile.TemporaryDirectory() as d:
            lg = os.path.join(d, "session_2026-07-08_00-00-00.log")
            # Write far more than the 256KB window so the head is guaranteed to
            # fall outside it; the tail must still be exact.
            with open(lg, "w", encoding="utf-8") as f:
                f.write("".join(f"padding line {i} ................\n"
                                 for i in range(40000)))
            out = wi.tail_log(d, 5)
            self.assertEqual(out["lines"],
                             [f"padding line {i} ................"
                              for i in range(39995, 40000)])

    # #37 — a non-numeric MICROPHONE_INDEX (device type) must 400, not coerce to None.
    def test_device_setting_bad_value_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "user_settings.json")
            with self.assertRaises(wi.SettingsWriteError):
                wi._write_settings({"MICROPHONE_INDEX": "not-a-number"}, p)
            self.assertFalse(os.path.exists(p))     # rejected before any write
            # A real index and the 'auto' sentinels remain accepted.
            self.assertEqual(
                wi._write_settings({"MICROPHONE_INDEX": "3"}, p)["MICROPHONE_INDEX"], 3)
            self.assertIsNone(
                wi._write_settings({"MICROPHONE_INDEX": ""}, p)["MICROPHONE_INDEX"])

    # #21 — concurrent _write_settings calls must not drop each other's update.
    def test_concurrent_settings_writes_do_not_lose_updates(self):
        import threading as _t
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "user_settings.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"KEEP": 1}, f)
            barrier = _t.Barrier(2)

            def _writer(val):
                barrier.wait()                       # maximise write overlap
                wi._write_settings({"WAKE_WORD_AUTOSTART": val}, p)

            a = _t.Thread(target=_writer, args=("yes",))
            b = _t.Thread(target=_writer, args=("no",))
            a.start(); b.start(); a.join(); b.join()
            with open(p, encoding="utf-8") as f:
                saved = json.load(f)
            # The pre-existing key must survive both writers (no lost merge).
            self.assertEqual(saved["KEEP"], 1)
            self.assertIn("WAKE_WORD_AUTOSTART", saved)

    # #22 — the handler carries a per-request socket timeout so an under-delivered
    # Content-Length can't hang a worker thread forever.
    def test_handler_has_request_timeout(self):
        self.assertIsNotNone(wi._Handler.timeout)
        self.assertLessEqual(wi._Handler.timeout, 30)

    # #39 — nvidia-smi is TTL-cached: within the TTL the probe is not re-run.
    def test_nvidia_smi_is_ttl_cached(self):
        orig_uncached = wi._nvidia_smi_gpus_uncached
        orig_cache = dict(wi._nvidia_smi_cache)
        calls = {"n": 0}

        def _counting():
            calls["n"] += 1
            return [{"index": 0, "name": "stub"}]

        wi._nvidia_smi_gpus_uncached = _counting
        wi._nvidia_smi_cache["ts"] = 0.0             # force a cold first read
        wi._nvidia_smi_cache["gpus"] = []
        try:
            first = wi._nvidia_smi_gpus()
            second = wi._nvidia_smi_gpus()           # served from cache
            self.assertEqual(first, second)
            self.assertEqual(calls["n"], 1)          # probe ran exactly once
        finally:
            wi._nvidia_smi_gpus_uncached = orig_uncached
            wi._nvidia_smi_cache.update(orig_cache)


# ════════════════════════════════════════════════════════════════════════════
#  THE HANG GUARD — this module must never be able to freeze the suite again
#
#  2026-07-12: a wedged socketserver.shutdown() in a tearDown froze an entire
#  ci_sim run mid-suite. The time-box that fixed it went into
#  skills/web_interface.py::_stop() and into tests/skills/test_web_interface.py
#  — but NOT here, in the module that actually starts a real ThreadingHTTPServer
#  for every one of its tests. A fix that lands in one copy while the others rot
#  is the house's #1 bug class, so the two tests below make the omission
#  impossible to repeat: the time-box is the ONLY way out of this file.
# ════════════════════════════════════════════════════════════════════════════
class NoUnboundedServerTeardownTests(unittest.TestCase):
    @staticmethod
    def _source() -> str:
        with open(__file__, "r", encoding="utf-8") as fh:
            return fh.read()

    # This class quotes the very patterns it bans, and _stop_server is the one
    # place the raw calls are supposed to live, so both are exempt from the scan.
    _EXEMPT = ("_stop_server", "NoUnboundedServerTeardownTests")

    def _units(self, *, top_level: bool):
        """(name, source) pairs to scan, with the exempt units removed.

        Two granularities, because the two rules genuinely need different ones:

          * TOP LEVEL (class or module function) for the "must use the
            time-box" pairing — _ServerBase starts the serve thread in setUp
            and stops it in tearDown, so a per-function rule could never see
            the pair.
          * PER FUNCTION for the raw-pair ban — SecurityBindTests holds one
            method that serves a real server and two that only bind and
            immediately close one. An unserved server has no handler threads
            and no serve_forever Event, so ITS server_close() is genuinely
            instant and must not be flagged."""
        src = self._source()
        tree = ast.parse(src)
        skip = []
        for node in tree.body:
            if (isinstance(node, (ast.ClassDef, ast.FunctionDef))
                    and node.name in self._EXEMPT):
                skip.append((node.lineno, node.end_lineno or node.lineno))
        nodes = tree.body if top_level else list(ast.walk(tree))
        out = []
        for node in nodes:
            want = (ast.ClassDef, ast.FunctionDef) if top_level else ast.FunctionDef
            if not isinstance(node, want):
                continue
            if any(lo <= node.lineno <= hi for lo, hi in skip):
                continue
            out.append((node.name, ast.get_source_segment(src, node) or ""))
        return out

    def _served(self, *, top_level: bool):
        return [(n, s) for n, s in self._units(top_level=top_level)
                if "serve_in_thread(" in s]

    def test_the_scan_still_sees_the_real_fixtures(self):
        """A source scan that matches nothing passes for the wrong reason."""
        names = [n for n, _ in self._served(top_level=True)]
        self.assertIn("_ServerBase", names)
        self.assertGreaterEqual(len(names), 2, names)

    def test_every_served_fixture_tears_down_through_the_timebox(self):
        offenders = [name for name, seg in self._served(top_level=True)
                     if "_stop_server" not in seg]
        self.assertEqual(
            offenders, [],
            "these units start a REAL serve thread but do not stop it through "
            "_stop_server(): %s. BaseServer.shutdown() and "
            "ThreadingMixIn.server_close() are both UNBOUNDED waits — a serve "
            "thread that never started, or a handler parked in the MJPEG "
            "camera stream, makes them block forever and the whole suite hangs "
            "with no test id and no traceback." % offenders)

    def test_no_served_function_calls_the_raw_pair(self):
        offenders = [name for name, seg in self._served(top_level=False)
                     if ".shutdown()" in seg or ".server_close()" in seg]
        self.assertEqual(
            offenders, [],
            "these functions start a serve thread and then call the raw "
            "unbounded teardown pair directly instead of going through "
            "_stop_server(): %s" % offenders)

    # ── the helper actually survives both wedges ─────────────────────────────
    def test_stop_server_survives_a_wedged_shutdown(self):
        never = threading.Event()
        self.addCleanup(never.set)
        closed = threading.Event()
        httpd = type("_Wedged", (), {})()
        httpd.shutdown = never.wait                  # blocks forever
        httpd.server_close = closed.set
        t0 = time.monotonic()
        _stop_server(httpd, None, shutdown_timeout=0.2, close_timeout=1.0)
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertTrue(closed.is_set(),
                        "server_close must still run — it is what frees the "
                        "listening socket")

    def test_stop_server_survives_a_wedged_server_close(self):
        # The half the 2026-07-12 fix never covered: ThreadingMixIn.
        # server_close() joins every handler thread, and a handler parked in
        # /api/camera-stream or on a 120s reply-wait makes that join outlive
        # the test.
        never = threading.Event()
        self.addCleanup(never.set)
        httpd = type("_Wedged", (), {})()
        httpd.shutdown = lambda: None
        httpd.server_close = never.wait              # blocks forever
        t0 = time.monotonic()
        _stop_server(httpd, None, shutdown_timeout=1.0, close_timeout=0.2)
        self.assertLess(time.monotonic() - t0, 5.0)

    def test_stop_server_is_a_no_op_cost_on_the_healthy_path(self):
        calls = []
        httpd = type("_Clean", (), {})()
        httpd.shutdown = lambda: calls.append("shutdown")
        httpd.server_close = lambda: calls.append("close")
        t0 = time.monotonic()
        _stop_server(httpd, None)
        self.assertEqual(calls, ["shutdown", "close"])
        self.assertLess(time.monotonic() - t0, 1.0,
                        "the time-box must cost nothing when nothing wedges")


# ═══════════════════════════════════════════════════════════════════════════
# WHY IS THIS TILE BLANK  (/api/camera-reason)
# ═══════════════════════════════════════════════════════════════════════════
def _health(**over):
    """A get_stream_health() payload with every key present, so a test overrides
    only the ONE symbol it is exercising and can never pass by accident because
    a key it forgot happened to be missing."""
    h = {"open": False, "color_pending": False, "body_pending": False,
         "depth_pending": False, "color_age_s": None, "body_age_s": None,
         "depth_age_s": None, "infrared": "unsupported", "ts": 0.0,
         "enabled": True, "open_error": None, "cooldown_s": 0.0,
         "pump_alive": True}
    h.update(over)
    return h


# The bridge's REAL failure strings, copied from audio/kinect_bridge.py so a
# reworded message there shows up here as a failing test rather than as a tile
# that silently drops to the generic rung.
ERR_NO_FRAMES = ("Kinect opened but streamed no frames after 4 attempts (opened "
                 "but no frames streaming); sensor may be held by another process")
ERR_NO_PYKINECT = "pykinect2 not installed \u2014 pip install pykinect2"
# The bridge's SECOND pykinect2 string, from the non-ImportError branch of
# import_pykinect2(): the package was FOUND and something else blew up - a
# broken comtypes, a venv rebuilt against a new Python, a corrupted
# site-packages. It had no constant here before 2026-09-05, which is why the
# suite could not see it collapsing into the same message as ERR_NO_PYKINECT.
ERR_PYKINECT_BROKEN = ("pykinect2 failed to load: ImportError: DLL load failed "
                       "while importing _ctypes: The specified module could "
                       "not be found.")
ERR_CTOR = "could not open Kinect sensor: OSError: [WinError 5] Access is denied"
# NOT a failure: what _open_runtime_locked hands the loser of its 0.5 s
# open-lock acquire while the WINNER is still inside the open gauntlet. It is
# latched into the same open_error cell as the three above, which is how the
# ladder used to mistake it for a completed failure.
ERR_OPEN_IN_PROGRESS = "Kinect open already in progress"


class _ReasonBase(_ServerBase):
    """Serves a real server and stubs the two live inputs of the ladder.

    NOTHING here may spawn powershell: _probe_kinect_devices is replaced for
    every test, so the suite stays headless-CI safe AND a Windows dev box does
    not fire ~20 subprocesses per run."""

    def setUp(self):
        super().setUp()
        self._saved = (wi._kinect_health, wi._probe_kinect_devices,
                       dict(wi._kinect_enum_cache))
        self.probe_calls = []
        self.set_enum(True, ("Xbox NUI Sensor",))
        self.set_health(None)
        self.addCleanup(self._restore)

    def _restore(self):
        wi._kinect_health, wi._probe_kinect_devices, cache = self._saved
        wi._kinect_enum_cache.clear()
        wi._kinect_enum_cache.update(cache)

    def set_enum(self, present, names=(), how="stub"):
        """Replace the OS device probe and clear its TTL cache."""
        def _probe():
            self.probe_calls.append(present)
            return present, tuple(names), how
        wi._probe_kinect_devices = _probe
        wi._kinect_enum_cache.update({"ts": 0.0, "present": None, "names": (),
                                      "how": "never run"})

    def set_health(self, health):
        wi._kinect_health = lambda: health

    def dark(self, cam):
        """Make sure ``cam`` has no preview file at all (the outage path)."""
        path = wi._preview_path_for(
            {"camera_preview_path": self.camera_preview_path}, cam)
        if os.path.exists(path):
            os.remove(path)

    def lit(self, cam):
        """Write a fresh preview file for ``cam`` (the working path)."""
        path = wi._preview_path_for(
            {"camera_preview_path": self.camera_preview_path}, cam)
        with open(path, "wb") as f:
            f.write(b"\xff\xd8\xff\xd9")             # a minimal JPEG-ish blob
        return path

    def reason(self, cam):
        code, data = _get(self.base + "/api/camera-reason?cam=" + cam)
        self.assertEqual(code, 200, data)
        return data


class CameraReasonStateTests(_ReasonBase):
    """Every rung of the ladder produces its OWN message, and every rung is
    backed by a symbol that actually establishes it."""

    def test_fresh_preview_is_live_not_an_outage(self):
        # The tile can fire `error` for reasons that are not the camera (a
        # stream refused at the client cap). Claiming "off" then would be a lie
        # of exactly the kind this endpoint exists to stop.
        self.lit("kinect")
        self.set_health(_health(open=False, open_error=ERR_NO_FRAMES))
        self.assertEqual(self.reason("kinect")["state"], "live")

    def test_switched_off_reports_the_switch_not_the_hardware(self):
        self.dark("kinect")
        self.set_health(_health(enabled=False))
        r = self.reason("kinect")
        self.assertEqual(r["state"], "disabled")
        self.assertIn("switched off", r["message"])

    def test_enumeration_empty_gives_the_owner_his_literal_string(self):
        self.dark("kinect")
        self.set_enum(False, ())                      # sweep ran, matched nothing
        self.set_health(_health(open=False, open_error=ERR_NO_FRAMES))
        r = self.reason("kinect")
        self.assertEqual(r["state"], "not_detected")
        self.assertIn("kinect not detected", r["message"].lower())

    def test_no_frames_with_the_device_present(self):
        self.dark("kinect")
        self.set_enum(True, ("Xbox NUI Sensor",))
        self.set_health(_health(open=False, open_error=ERR_NO_FRAMES))
        r = self.reason("kinect")
        self.assertEqual(r["state"], "no_frames")
        self.assertIn("plugged in", r["message"])
        self.assertIn("no pictures", r["message"])

    def test_missing_python_package(self):
        self.dark("kinect")
        self.set_health(_health(open=False, open_error=ERR_NO_PYKINECT))
        self.assertEqual(self.reason("kinect")["state"], "pykinect2_missing")

    def test_unloadable_python_package_is_its_own_rung(self):
        self.dark("kinect")
        self.set_health(_health(open=False, open_error=ERR_PYKINECT_BROKEN))
        self.assertEqual(self.reason("kinect")["state"], "pykinect2_unusable")

    def test_open_failure_reports_verbatim_in_detail(self):
        self.dark("kinect")
        self.set_health(_health(open=False, open_error=ERR_CTOR))
        r = self.reason("kinect")
        self.assertEqual(r["state"], "open_failed")
        self.assertEqual(r["detail"], ERR_CTOR)       # the raw error, not prose

    def test_no_recorded_error_is_not_reported_as_a_failure(self):
        # open False AND open_error None must blame no hardware. It is ALSO not
        # a licence to claim the sensor was never tried - that half is nailed
        # down in CameraReasonNotOpenNoErrorTests below.
        self.dark("kinect")
        self.set_health(_health(open=False, open_error=None))
        r = self.reason("kinect")
        self.assertEqual(r["state"], "not_open_no_error")
        for blame in ("not detected", "power", "plugged", "missing"):
            self.assertNotIn(blame, r["message"].lower(), r["message"])

    def test_dead_worker_is_named_as_software_not_hardware(self):
        self.dark("kinect")
        self.set_health(_health(open=True, pump_alive=False))
        r = self.reason("kinect")
        self.assertEqual(r["state"], "worker_stopped")
        self.assertIn("worker", r["message"])
        self.assertNotIn("not detected", r["message"].lower())

    def test_frames_arriving_but_not_reaching_the_page(self):
        # color_pending True is poll-independent PROOF a new COLOR frame is
        # arriving, and color is the only stream this tile renders, so the gap
        # is downstream of the sensor and must not be blamed on it.
        self.dark("kinect")
        self.set_health(_health(open=True, color_pending=True))
        r = self.reason("kinect")
        self.assertEqual(r["state"], "frames_not_shown")
        self.assertNotIn("power", r["message"].lower())

    def test_a_non_color_stream_may_not_claim_the_picture_is_being_produced(self):
        """THE DEPTH-PINNED LIE (2026-09-05).

        This rung used to be `any(color_pending, body_pending, depth_pending)`,
        so a stream the tile does NOT display could assert that the picture was
        being produced and the PAGE was the fault.

        DEPTH is the worst of the two because it is pinned pending forever:
        audio.kinect_bridge advances _depth_time_seen only inside get_depth(),
        and nothing polls depth on a schedule (the bridge's own docstring:
        "nothing polls depth on a schedule; body has the 30 Hz pump"). So while
        the sensor emits depth at all, depth_pending never goes False. BODY is
        the documented BODY-but-no-COLOR reopen the bridge's require_color=True
        flag exists to reject.

        With the color camera DEAD, either one rendered "Kinect is running, but
        its picture is not reaching this page." - sending the owner to debug his
        browser while the camera was what died."""
        self.dark("kinect")
        for label, h in (
                ("depth only", _health(open=True, depth_pending=True)),
                ("body only", _health(open=True, body_pending=True)),
                ("body+depth", _health(open=True, body_pending=True,
                                       depth_pending=True))):
            self.set_health(h)
            r = self.reason("kinect")
            msg = r["message"].lower()
            self.assertEqual(
                r["state"], "color_unconfirmed",
                "%s (color_pending FALSE) reached %s: a stream this tile does "
                "not display asserted the picture was being produced"
                % (label, r["state"]))
            # The specific sentence that sent him to the browser.
            self.assertNotIn("not reaching this page", msg, label)
            # ...and it must not swing the other way into an unestablished
            # hardware verdict either: color_pending False is a failure to
            # confirm, not proof the camera died.
            for blame in ("not detected", "power", "adapter", "unplug",
                          "plugged in"):
                self.assertNotIn(blame, msg, "%s / %r" % (label, r["message"]))
            # Both candidates named, neither asserted.
            self.assertIn("color camera", msg, label)
            self.assertIn("this page", msg, label)
            # The evidence rides along in the tooltip.
            self.assertIn("color_pending false", r.get("detail", ""), label)

    def test_only_color_pending_can_reach_frames_not_shown(self):
        """The inverse guard, stated over the WHOLE truth table so a future
        `any()` cannot creep back in on some other key."""
        self.dark("kinect")
        for body in (False, True):
            for depth in (False, True):
                for color in (False, True):
                    self.set_health(_health(open=True, color_pending=color,
                                            body_pending=body,
                                            depth_pending=depth))
                    r = self.reason("kinect")
                    combo = "color=%s body=%s depth=%s" % (color, body, depth)
                    if r["state"] == "frames_not_shown":
                        self.assertTrue(
                            color,
                            "%s claimed the picture is produced and the page is "
                            "at fault WITHOUT a pending color frame" % combo)
                    if color:
                        self.assertEqual(r["state"], "frames_not_shown", combo)

    def test_open_but_quiet_is_hedged(self):
        # pending False is NOT proof of death (it is the normal reading right
        # after the 30 Hz pump consumed the frame), so no cause may be named.
        self.dark("kinect")
        self.set_health(_health(open=True))
        r = self.reason("kinect")
        self.assertEqual(r["state"], "open_quiet")
        for blame in ("not detected", "power", "adapter", "unplug"):
            self.assertNotIn(blame, r["message"].lower(), r["message"])

    def test_bridge_absent_is_unknown_not_a_verdict(self):
        self.dark("kinect")
        self.set_health(None)                         # module not in sys.modules
        r = self.reason("kinect")
        self.assertEqual(r["state"], "unknown")
        self.assertIn("unknown", r["message"].lower())

    def test_every_state_has_its_own_distinct_message(self):
        seen = {}
        for state, msg in wi._CAMERA_REASONS.items():
            self.assertNotIn(msg, seen,
                             "%s and %s share a message, so the tile cannot "
                             "tell them apart" % (state, seen.get(msg)))
            seen[msg] = state
            self.assertTrue(msg.strip().endswith("."), state)

    def test_the_states_the_ladder_can_reach_are_all_in_the_table(self):
        """A rung returning a slug with no table entry would silently render the
        generic 'unknown' text - the stale-duplicate shape, in message form."""
        with open(wi.__file__, encoding="utf-8") as fh:
            src = fh.read()
        body = src[src.index("def _camera_off_reason("):]
        used = set(re.findall(r'_reason\(cam, "([a-z_]+)"', body))
        self.assertTrue(used, "the scan found no rungs - it would pass blind")
        self.assertEqual(used - set(wi._CAMERA_REASONS), set())


class NotDetectedIsNeverGuessedTests(_ReasonBase):
    """THE POINT OF THE WHOLE TASK.

    Measured on the owner's machine 2026-09-04 23:09-23:28: enumeration reported
    Xbox NUI Sensor / WDF KinectSensor Interface 0 / Microphone Array all Status
    OK for the entire six minutes the Kinect sent no frames. A tile that had said
    "Kinect not detected" would have been factually wrong at every instant, and
    would have sent him to check a cable that was fine. So: the words may appear
    ONLY when a completed device sweep matched nothing."""

    def test_never_says_not_detected_while_enumeration_succeeds(self):
        self.dark("kinect")
        self.set_enum(True, ("Xbox NUI Sensor", "WDF KinectSensor Interface 0"))
        bad = []
        for label, h in (
                ("no frames", _health(open=False, open_error=ERR_NO_FRAMES)),
                ("pykinect2 missing", _health(open=False, open_error=ERR_NO_PYKINECT)),
                ("pykinect2 broken", _health(open=False, open_error=ERR_PYKINECT_BROKEN)),
                ("ctor failed", _health(open=False, open_error=ERR_CTOR)),
                ("never probed", _health(open=False, open_error=None)),
                ("disabled", _health(enabled=False)),
                ("worker dead", _health(open=True, pump_alive=False)),
                ("open, quiet", _health(open=True)),
                ("streaming", _health(open=True, color_pending=True)),
                ("body/depth only", _health(open=True, body_pending=True,
                                            depth_pending=True)),
                ("bridge absent", None)):
            self.set_health(h)
            r = self.reason("kinect")
            if r["state"] == "not_detected" or "not detected" in r["message"].lower():
                bad.append("%s -> %s / %r" % (label, r["state"], r["message"]))
        self.assertEqual(
            bad, [],
            "the tile claimed the Kinect was NOT DETECTED while device "
            "enumeration was reporting it present: %s. That is the exact lie "
            "this endpoint exists to prevent - the OS saw the sensor the whole "
            "time the owner was being told to go hunt for it." % bad)

    def test_a_probe_that_could_not_run_is_unknown_never_absent(self):
        # not Windows / no powershell / timeout / non-zero exit -> present None.
        # None must NEVER be allowed to decay into "not detected".
        self.dark("kinect")
        for how in ("not windows", "probe failed: TimeoutExpired", "probe exit 1"):
            self.set_enum(None, (), how)
            self.set_health(_health(open=False, open_error=ERR_NO_FRAMES))
            r = self.reason("kinect")
            self.assertNotEqual(r["state"], "not_detected", how)
            self.assertEqual(r["state"], "no_frames_unverified", how)

    def test_unverified_no_frames_does_not_claim_the_device_is_plugged_in(self):
        # With enumeration unknown we have NOT established that anything is
        # attached, so the message may not say so - it lists the candidates.
        self.dark("kinect")
        self.set_enum(None, (), "not windows")
        self.set_health(_health(open=False, open_error=ERR_NO_FRAMES))
        m = self.reason("kinect")["message"]
        self.assertNotIn("is plugged in but", m)
        self.assertIn("check it is plugged in", m)

    def test_power_is_offered_as_a_check_never_asserted(self):
        # The OS cannot see mains power. Every message that mentions power must
        # phrase it as something to CHECK, and none may state it as fact.
        for state, msg in wi._CAMERA_REASONS.items():
            low = msg.lower()
            for claimed in ("is not powered", "has no power", "is unpowered",
                            "is not turned on", "power is off"):
                self.assertNotIn(claimed, low, "%s asserts power state" % state)
            if "power adapter" in low:
                self.assertIn("check", low, "%s must hedge" % state)

    def test_not_detected_says_only_what_enumeration_established(self):
        self.dark("kinect")
        self.set_enum(False, ())
        self.set_health(_health(open=False, open_error=ERR_NO_FRAMES))
        r = self.reason("kinect")
        self.assertEqual(r["state"], "not_detected")
        # It may report ONLY what the sweep established - that Windows currently
        # enumerates no Kinect device. It must not go on to diagnose power or
        # other apps, which it cannot see, NOR to assert that nothing is plugged
        # in, which an empty sweep does not establish either (next test).
        self.assertNotIn("power", r["message"].lower())
        self.assertIn("enumeration", r.get("detail", ""))

    def test_not_detected_never_claims_nothing_is_plugged_in(self):
        """Enumeration is ASYMMETRIC and the message must respect that.

        A match proves attachment (-PresentOnly means attached right now), so
        the no_frames rung may say "plugged in". An EMPTY sweep proves only that
        Windows enumerates no Kinect: the v2's USB3 adapter carries the data
        link AND the mains brick, so a brick switched off or dead - and equally
        a removed, blocked or disabled driver stack - stops the whole adapter
        enumerating with the cable still fully seated. That is the state the
        owner reported ("isn't showing or even powered on"), and a tile that
        answered "no Kinect is plugged in" sent him to re-seat a cable that was
        fine while the power switch was the actual fault."""
        self.dark("kinect")
        self.set_enum(False, ())                      # sweep ran, matched nothing
        self.set_health(_health(open=False, open_error=ERR_NO_FRAMES))
        r = self.reason("kinect")
        self.assertEqual(r["state"], "not_detected")
        low = r["message"].lower()
        for claimed in ("no kinect is plugged in", "nothing is plugged in",
                        "not plugged in", "is unplugged", "no kinect is connected",
                        "nothing is connected", "is disconnected"):
            self.assertNotIn(
                claimed, low,
                "the tile asserted a cable state that device enumeration cannot "
                "establish (%r): an empty sweep is equally the reading for a "
                "Kinect whose power brick is off, or whose driver was removed, "
                "with the cable still seated." % r["message"])
        # ...and it still hands the owner the words he asked for, plus the one
        # fact that WAS established.
        self.assertIn("kinect not detected", low)
        self.assertIn("windows", low)

    def test_no_message_in_the_table_asserts_a_cable_state(self):
        """The table-wide twin of the test above: no rung, present or future,
        may state as fact something only a human at the desk can see."""
        for state, msg in wi._CAMERA_REASONS.items():
            low = msg.lower()
            for claimed in ("no kinect is plugged in", "nothing is plugged in",
                            "is not plugged in", "is unplugged",
                            "nothing is connected", "is disconnected"):
                self.assertNotIn(claimed, low,
                                 "%s asserts a cable state: %r" % (state, msg))


class Pykinect2IsNotTheKinectDriverTests(_ReasonBase):
    """A pip-level import failure must never be reported as a missing DRIVER.

    kinect_bridge._open_runtime_locked has exactly two pykinect2 failure
    strings, both produced by its ``import_pykinect2()`` call:

        except ImportError:  return None, "pykinect2 not installed - pip install pykinect2"
        except Exception:    return None, f"pykinect2 failed to load: {type(e).__name__}: {e}"

    Both are raised BEFORE PyKinectRuntime is ever constructed, so neither
    establishes one single thing about the Windows Kinect driver - and the
    enumeration rung two steps above can be returning present True in the very
    same request, i.e. the driver is demonstrably fine.

    Until 2026-09-05 the ladder mapped BOTH to state "driver_missing" and the
    sentence "Kinect driver software is missing, so JARVIS cannot use it." That
    sends the owner off to reinstall the Kinect SDK - an hour and a reboot - for
    a fault whose fix is one pip command, and the true text survived only in the
    hover title attribute he has no reason to hover. It also called a package
    that was FOUND-but-broken "missing", which is false on its own terms."""

    def _ask(self, err, present=True):
        self.dark("kinect")
        self.set_enum(present, ("Xbox NUI Sensor", "WDF KinectSensor Interface 0")
                      if present else ())
        self.set_health(_health(open=False, open_error=err))
        return self.reason("kinect")

    def test_the_absent_package_is_named_and_pip_is_the_instruction(self):
        r = self._ask(ERR_NO_PYKINECT)
        self.assertEqual(r["state"], "pykinect2_missing")
        low = r["message"].lower()
        self.assertIn("pykinect2", low, r["message"])
        self.assertIn("pip install", low, r["message"])

    def test_a_broken_package_is_not_called_missing(self):
        # This string is reachable ONLY when the import got past absent, so
        # "not installed"/"is missing" would be a false statement about it.
        r = self._ask(ERR_PYKINECT_BROKEN)
        self.assertEqual(r["state"], "pykinect2_unusable")
        self.assertIn("pykinect2", r["message"].lower())
        for lie in ("not installed", "is missing", "missing,"):
            self.assertNotIn(lie, r["message"].lower(), r["message"])

    def test_the_two_bridge_strings_do_not_collapse_to_one_verdict(self):
        # THE REGRESSION. Both used to return the identical state AND sentence,
        # so the tile could not tell "run pip install" from "read the error".
        a, b = self._ask(ERR_NO_PYKINECT), self._ask(ERR_PYKINECT_BROKEN)
        self.assertNotEqual(a["state"], b["state"])
        self.assertNotEqual(a["message"], b["message"])

    def test_no_pykinect2_failure_blames_the_driver_software(self):
        """THE DEFECT, stated as the property it violated.

        A pykinect2 import failure may name the pip package and may say JARVIS
        never got as far as the driver. It may NOT assert that the driver stack
        itself is absent, broken or in need of reinstalling - nothing in this
        request established that, and enumeration is concurrently saying the
        opposite."""
        for err in (ERR_NO_PYKINECT, ERR_PYKINECT_BROKEN,
                    "pykinect2 went sideways in some brand new way"):
            r = self._ask(err)
            low = r["message"].lower()
            for lie in ("driver software is missing", "driver software",
                        "driver is missing", "driver is not installed",
                        "reinstall the driver", "kinect sdk"):
                self.assertNotIn(
                    lie, low,
                    "a pip-level import failure (%r) blamed the Kinect DRIVER: "
                    "%r. Device enumeration returned present True in this same "
                    "request, so the driver is the one thing here that is known "
                    "to be fine." % (err, r["message"]))
            # ...and it must positively name the thing that DID fail, or the
            # owner has nowhere to go but the driver anyway.
            self.assertIn("pykinect2", low, "%r -> %r" % (err, r["message"]))

    def test_an_unrecognised_pykinect2_string_gets_the_hedged_rung(self):
        # A reworded/new bridge string must fall to the vaguer message, never be
        # guessed into the specific "is not installed" one.
        r = self._ask("pykinect2 went sideways in some brand new way")
        self.assertEqual(r["state"], "pykinect2_unusable")

    def test_the_raw_bridge_error_still_reaches_the_detail(self):
        for err in (ERR_NO_PYKINECT, ERR_PYKINECT_BROKEN):
            self.assertEqual(self._ask(err)["detail"], err)

    def test_both_strings_still_come_from_the_bridge_verbatim(self):
        """The stale-duplicate guard: these markers are matched in
        web_interface and PRODUCED in kinect_bridge. If either string is
        reworded there, this fails loudly instead of the tile silently sliding
        down to the generic open_failed rung."""
        with open(os.path.join(os.path.dirname(wi.__file__), "..",
                               "audio", "kinect_bridge.py"), encoding="utf-8") as fh:
            bridge = fh.read()
        self.assertIn("pykinect2 not installed", bridge)
        self.assertIn("pykinect2 failed to load:", bridge)
        # ...and the prefix the ladder actually keys on holds for both.
        for err in (ERR_NO_PYKINECT, ERR_PYKINECT_BROKEN):
            self.assertTrue(err.startswith("pykinect2 "), err)


class FreshFileMeansTheStreamFailed_NotTheCameraTests(_ReasonBase):
    """A FRESH preview file with a BLANK tile is not "Live." — it is a broken
    video connection, and the tile must say so.

    /api/camera-reason is fetched from exactly one place: a tile's `error`
    handler. Its message is written into .camoff — the dashed placeholder that is
    only visible BECAUSE there is no picture. So an answer of "Live." rendered
    the word Live inside an empty box: the same class of lie as "not detected" on
    a sensor Windows can see, and it points the owner at nothing.

    REPRODUCED 2026-09-05 on a private instance: eight concurrent
    /api/camera-stream?cam=kinect clients filled _CAMERA_STREAM_MAX_CLIENTS, the
    ninth got `503 {"error": "too many camera streams"}`, and camera-reason
    answered `{"state": "live", "message": "Live."}` at that same instant. Three
    dashboard tabs (3 tiles x 3 = 9 streams) is enough to reach it in normal use.
    """

    def setUp(self):
        super().setUp()
        self.addCleanup(self._restore_stream_count,
                        wi._camera_stream_clients[0])
        wi._camera_stream_clients[0] = 0              # a known-free baseline

    def _restore_stream_count(self, n):
        wi._camera_stream_clients[0] = n

    def saturate(self):
        """Fill every MJPEG slot, exactly as 8 live tiles would."""
        wi._camera_stream_clients[0] = wi._CAMERA_STREAM_MAX_CLIENTS

    def test_a_blank_tile_is_never_answered_with_bare_liveness(self):
        # THE BUG: fresh file -> "Live." -> rendered in the empty placeholder.
        self.lit("kinect")
        r = self.reason("kinect")
        low = r["message"].lower().strip().rstrip(".")
        self.assertNotIn(low, ("live", "ok", "working", "streaming", "fine"),
                         "the placeholder is only ever visible when the tile has "
                         "NO picture, so %r asserts the opposite of what the "
                         "owner is looking at" % r["message"])
        # It must instead explain the empty box: the camera is fine, this page
        # is not getting the picture.
        self.assertIn("this page", r["message"].lower())

    def test_no_rung_claims_a_camera_is_fine_without_explaining_the_blank_box(self):
        """Table-wide twin: every message is read inside an empty placeholder, so
        a rung that says the camera IS sending pictures must go on to say the
        page is not getting them. Otherwise it contradicts the box it sits in."""
        for state, msg in wi._CAMERA_REASONS.items():
            low = msg.lower()
            if "sending pictures" not in low:
                continue
            self.assertTrue("this page" in low or "video connection" in low,
                            "%s says the camera is sending pictures but never "
                            "says why the tile is blank: %r" % (state, msg))

    def test_a_full_stream_cap_is_named_instead_of_guessed(self):
        # The one sub-case the server can actually establish - it is the thing
        # the server itself is doing - so it is the only one with an action.
        self.lit("kinect")
        self.saturate()
        r = self.reason("kinect")
        self.assertEqual(r["state"], "stream_busy")
        low = r["message"].lower()
        self.assertIn("close other", low)             # something he can DO
        for blame in ("not detected", "power", "adapter", "unplug", "switched off"):
            self.assertNotIn(blame, low, "%r blames the hardware for a refused "
                                         "stream" % r["message"])

    def test_the_rung_and_the_503_read_the_same_counter(self):
        """The refusal and the explanation must flip together, or the tile is
        told 'live' at the exact moment its stream is being refused - which is
        how this defect was reproduced."""
        self.lit("kinect")
        self.assertEqual(self.reason("kinect")["state"], "live")   # slots free
        self.saturate()
        code, body = _get_raw(self.base + "/api/camera-stream?cam=kinect")
        self.assertEqual(code, 503, body)
        self.assertIn("too many camera streams", body)
        self.assertEqual(self.reason("kinect")["state"], "stream_busy")

    def test_a_webcam_at_the_cap_gets_the_same_answer_as_the_kinect(self):
        # The cap is global, so a refused webcam stream is the same event; and
        # this must still cost NO device probe (a webcam never touches it).
        self.lit("left")
        self.saturate()
        r = self.reason("left")
        self.assertEqual(r["state"], "stream_busy")
        self.assertNotIn("kinect", r["message"].lower())
        self.assertEqual(self.probe_calls, [])

    def test_the_saturation_check_never_blocks_the_reason_path(self):
        # It is one int read under the counter's own lock. If it ever grew into
        # something that waits (or that opens a socket), a dead tile asking 4x a
        # second would drag /api/status down with it.
        self.assertFalse(wi._camera_streams_saturated())
        self.saturate()
        self.assertTrue(wi._camera_streams_saturated())
        t0 = time.monotonic()
        for _ in range(200):
            wi._camera_streams_saturated()
        self.assertLess(time.monotonic() - t0, 0.5)

    def test_the_page_stops_re_arming_a_stream_the_server_is_refusing(self):
        """The other half of the defect: an <img> `error` clears
        dataset.streaming, so the 250 ms supervisor re-arms the tile and is
        refused again ~4x a second, forever - camMode's fallback cannot help
        because camEverStreamed is already true from the two working tiles. The
        server's answer has to be able to end it."""
        code, html = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        self.assertIn("stream_busy", html)                # the client keys on it
        self.assertIn("img.dataset.mode = 'poll'", html)  # ...and demotes itself
        # and the supervisor must HONOUR the demotion, both ways round, or the
        # demotion is a dead letter: no new stream for that tile, and it still
        # gets a picture from the still endpoint.
        self.assertIn("if (img.dataset.mode === 'poll') continue;", html)
        self.assertIn("if (img.dataset.mode === 'poll') pollTile(img);", html)


class WebcamsAreUnaffectedByTheKinectTests(_ReasonBase):
    """A dead Kinect must never blank, or re-label, a working webcam."""

    def test_working_webcam_stays_live_while_the_kinect_is_dead(self):
        self.lit("left")
        self.lit("right")
        self.dark("kinect")
        self.set_enum(False, ())                      # the worst Kinect verdict
        self.set_health(_health(open=False, open_error=ERR_NO_FRAMES))
        self.assertEqual(self.reason("left")["state"], "live")
        self.assertEqual(self.reason("right")["state"], "live")
        self.assertEqual(self.reason("kinect")["state"], "not_detected")
        # and the picture itself still serves (raw bytes: _get_raw decodes as
        # utf-8, and a JPEG is not text)
        req = urllib.request.Request(self.base + "/api/camera-preview?cam=left")
        with _urlopen_retry(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertTrue(resp.read())

    def test_a_dark_webcam_never_borrows_a_kinect_message(self):
        self.dark("left")
        self.set_enum(False, ())
        self.set_health(_health(open=False, open_error=ERR_NO_FRAMES))
        r = self.reason("left")
        self.assertEqual(r["state"], "webcam_off")
        self.assertNotIn("kinect", r["message"].lower())

    def test_a_webcam_reason_never_runs_the_device_probe(self):
        # The probe is a ~0.7 s powershell spawn; a webcam has nothing to do with
        # it, and paying for it on every dark webcam tile would be a real cost on
        # the error path.
        self.dark("left")
        self.dark("right")
        self.reason("left")
        self.reason("right")
        self.assertEqual(self.probe_calls, [])


class CameraReasonPlumbingTests(_ReasonBase):
    """Cost, caching, auth and the wiring into the page."""

    def test_enumeration_is_cached_for_the_ttl(self):
        self.dark("kinect")
        self.set_health(_health(open=False, open_error=ERR_NO_FRAMES))
        for _ in range(5):
            self.reason("kinect")
        self.assertEqual(len(self.probe_calls), 1,
                         "the ~0.7 s device sweep must run once per TTL, not "
                         "once per tile error: %s" % self.probe_calls)

    def test_the_ttl_is_bounded_so_a_replug_is_noticed(self):
        self.assertGreater(wi._KINECT_ENUM_TTL_S, 0)
        self.assertLessEqual(wi._KINECT_ENUM_TTL_S, 60.0)

    def test_health_is_read_from_sys_modules_and_never_opens_the_sensor(self):
        """available()/get_runtime() run an open gauntlet of up to ~16 s inline
        and have tripped the main-loop watchdog. The status read must not go
        anywhere near them."""
        wi._kinect_health = self._saved[0]            # the real implementation

        class _Boom:
            @staticmethod
            def get_stream_health():
                return _health(open=True, color_pending=True)

            @staticmethod
            def available():
                raise AssertionError("the tile opened the sensor")

            @staticmethod
            def get_runtime():
                raise AssertionError("the tile opened the sensor")

        prev = sys.modules.get("audio.kinect_bridge")
        sys.modules["audio.kinect_bridge"] = _Boom()

        def _put_back():
            if prev is None:
                sys.modules.pop("audio.kinect_bridge", None)
            else:
                sys.modules["audio.kinect_bridge"] = prev
        self.addCleanup(_put_back)

        self.dark("kinect")
        self.assertEqual(self.reason("kinect")["state"], "frames_not_shown")

    def test_a_broken_health_getter_degrades_to_unknown(self):
        def _raise():
            raise RuntimeError("bridge exploded")

        wi._kinect_health = self._saved[0]
        prev = sys.modules.get("audio.kinect_bridge")
        sys.modules["audio.kinect_bridge"] = type(
            "_M", (), {"get_stream_health": staticmethod(_raise)})()

        def _put_back():
            if prev is None:
                sys.modules.pop("audio.kinect_bridge", None)
            else:
                sys.modules["audio.kinect_bridge"] = prev
        self.addCleanup(_put_back)

        self.dark("kinect")
        self.assertEqual(self.reason("kinect")["state"], "unknown")

    def test_unknown_cam_is_404(self):
        code, _ = _get_raw(self.base + "/api/camera-reason?cam=ceiling")
        self.assertEqual(code, 404)

    def test_dashboard_wires_the_placeholder_to_the_endpoint(self):
        code, html = _get_raw(self.base + "/")
        self.assertEqual(code, 200)
        self.assertIn("/api/camera-reason?cam=", html)
        self.assertIn("explainTile", html)
        # still exactly the three tiles, and each has its own placeholder
        for tile in ("camLeftOff", "camRightOff", "camKinectOff"):
            self.assertIn(tile, html)
        self.assertEqual(html.count('class="camtile"'), 3)
        # the reason fetch must live on the ERROR path only - never in the tick
        self.assertNotIn("explainTile(",
                         html.split("function refreshCamera")[1])


class CameraReasonTokenTests(_ReasonBase):
    token = "s3cr3t"

    def test_reason_requires_the_token(self):
        self.dark("kinect")
        code, _ = _get_raw(self.base + "/api/camera-reason?cam=kinect")
        self.assertEqual(code, 401)
        code, _ = _get_raw(self.base + "/api/camera-reason?cam=kinect",
                           headers={"X-Auth-Token": self.token})
        self.assertEqual(code, 200)


class ProbeHonestyTests(unittest.TestCase):
    """The device probe itself: a sweep that did not COMPLETE must report
    unknown, because 'unknown' is what stops the owner's literal string."""

    def _run_with(self, fake_run):
        """Swap wi.subprocess and wi.sys for shims so this runs identically on
        headless Linux CI and on the Windows box - patching the REAL subprocess
        module (or sys.platform) would leak into every other test."""
        saved_sub, saved_sys = wi.subprocess, wi.sys
        wi.subprocess = type("_Sub", (), {"run": staticmethod(fake_run),
                                          "CREATE_NO_WINDOW": 0})
        wi.sys = type("_Sys", (), {"platform": "win32",
                                   "modules": sys.modules})

        def _put_back():
            wi.subprocess, wi.sys = saved_sub, saved_sys
        self.addCleanup(_put_back)
        return wi._probe_kinect_devices()

    @staticmethod
    def _proc(returncode, stdout=""):
        return lambda *a, **k: type("_P", (), {"returncode": returncode,
                                               "stdout": stdout})()

    def test_non_zero_exit_is_unknown(self):
        self.assertIsNone(self._run_with(self._proc(1))[0])

    def test_an_exception_is_unknown(self):
        def _boom(*a, **k):
            raise OSError("powershell missing")
        self.assertIsNone(self._run_with(_boom)[0])

    def test_a_clean_empty_sweep_is_a_definitive_no(self):
        present, names, _how = self._run_with(self._proc(0, "\n  \n"))
        self.assertIs(present, False)
        self.assertEqual(names, ())

    def test_a_clean_sweep_with_matches_is_a_definitive_yes(self):
        present, names, _how = self._run_with(self._proc(
            0, "Xbox NUI Sensor\nWDF KinectSensor Interface 0\n"))
        self.assertIs(present, True)
        self.assertEqual(len(names), 2)

    def test_off_windows_the_probe_is_unknown_not_a_verdict(self):
        saved_sys = wi.sys
        wi.sys = type("_Sys", (), {"platform": "linux", "modules": sys.modules})
        self.addCleanup(lambda: setattr(wi, "sys", saved_sys))
        self.assertIsNone(wi._probe_kinect_devices()[0])

    def test_the_query_only_matches_present_devices(self):
        # Without -PresentOnly the PnP Enum registry answers for every device the
        # machine has EVER seen, so a Kinect unplugged months ago would read as
        # present and the owner's string could never fire.
        self.assertIn("-PresentOnly", wi._KINECT_PNP_PS)

    def test_the_query_also_matches_raw_kinect_usb_ids(self):
        # A sensor whose driver failed to install has no recognisable
        # FriendlyName but still enumerates its USB node; calling that "not
        # detected" would be the same lie in a new costume.
        self.assertIn("VID_045E&PID_02C4", wi._KINECT_PNP_PS)


class KinectProbeProjectionTests(unittest.TestCase):
    """The Where-Object filter and the ForEach-Object projection must AGREE.

    Every -like clause in _KINECT_PNP_PS exists to make some device COUNT as
    found. A clause that can match a device the projection then throws away is
    worse than having no clause at all, because the failure is silent and
    inverted: powershell exits 0, stdout holds nothing printable,
    _probe_kinect_devices' `if ln.strip()` drops it, names==() and present
    becomes False - the single reading that unlocks the owner's literal
    "Kinect not detected". The tile then sends him to check a cable on a sensor
    Windows is enumerating perfectly well.

    The five InstanceId clauses are exactly that hazard: they were added for a
    sensor whose driver failed to bind, and such a node has a NULL FriendlyName.
    Measured on the owner's box 2026-09-04, Get-PnpDevice -PresentOnly really
    does return present nodes with a null FriendlyName (the root PnP node is one),
    so this is a reachable state and not a thought experiment."""

    # A Kinect v2 hub whose driver did not bind: matches ONLY on InstanceId,
    # and FriendlyName is $null exactly as Windows reports it.
    _SYNTH = (r"@([pscustomobject]@{FriendlyName=$null;"
              r"InstanceId='USB\VID_045E&PID_02C4&2d0d3e5f&0&4'})")

    def test_the_projection_can_emit_an_instance_id(self):
        """Platform-independent guard: InstanceId must appear in the PROJECTION
        half, not merely in the filter half. This is the half of the assertion
        test_the_query_also_matches_raw_kinect_usb_ids was missing - it proved
        the clause was written, never that a match on it could survive."""
        head, sep, projection = wi._KINECT_PNP_PS.partition("ForEach-Object")
        self.assertTrue(sep, "the probe no longer has a projection stage")
        self.assertIn("VID_045E", head,
                      "the InstanceId clauses moved out of the filter")
        self.assertIn(
            "InstanceId", projection,
            "the projection discards InstanceId-only matches, so a driverless "
            "but ENUMERATED Kinect reads as present=False -> 'not detected'")

    @unittest.skipUnless(sys.platform == "win32", "needs a real powershell.exe")
    def test_a_driverless_kinect_survives_the_real_pipeline(self):
        """Run the REAL filter+projection text - only the device SOURCE is
        swapped for a synthetic one, so no hardware is touched and
        Get-PnpDevice is never called. Fails on the pre-fix projection with
        present False; passes only when a nameless match still prints."""
        cmd = wi._KINECT_PNP_PS.replace("Get-PnpDevice -PresentOnly", self._SYNTH)
        self.assertIn(self._SYNTH, cmd,
                      "the device source string changed; this test would have "
                      "silently enumerated real hardware instead")
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # Parsed with the exact expression _probe_kinect_devices uses, because
        # the blank-line drop is half of what made the old bug invisible.
        names = tuple(ln.strip() for ln in (proc.stdout or "").splitlines()
                      if ln.strip())
        self.assertTrue(
            names,
            "a matched, PRESENT device vanished between filter and stdout; "
            "present would be False and the tile would say 'not detected' "
            "about a Kinect Windows is enumerating")
        self.assertIn("VID_045E&PID_02C4", names[0])


class OpenStillRunningIsNotAFailureTests(_ReasonBase):
    """An open that has not FINISHED must never be rendered as one that FAILED.

    kinect_bridge._open_runtime_locked hands "Kinect open already in progress"
    to whoever LOSES the 0.5 s acquire on its open-attempt lock while the winner
    is inside the ~16 s verify/retry gauntlet, and get_runtime() feeds that
    string straight to _publish_open_failure - which latches it into the SAME
    open_error cell a real failure uses. The ladder's generic `if err:` rung then
    called it open_failed / "JARVIS could not open the Kinect."

    REPRODUCED END TO END 2026-09-05, no sensor touched: a thread holding
    _open_attempt_lock made get_stream_health() return open=False,
    open_error='Kinect open already in progress', cooldown_s=5.0, and feeding
    that dict to _camera_off_reason gave state=open_failed. At boot that is the
    NORMAL reading - the always-on body pump is in the gauntlet, the preview
    compositor's get_color_bgr() loses the race, and the preview file is still
    last session's, so the tile asks precisely then. The open it was calling
    dead then SUCCEEDED ("[kinect] sensor live after 2 open attempts",
    2026-09-04 23:16): a first-frame delay rendered as a hardware fault."""

    def test_an_open_still_in_flight_is_not_reported_as_a_failed_open(self):
        self.dark("kinect")
        self.set_enum(True, ("Xbox NUI Sensor",))
        self.set_health(_health(open=False, open_error=ERR_OPEN_IN_PROGRESS,
                                cooldown_s=5.0))
        r = self.reason("kinect")
        self.assertEqual(
            r["state"], "open_in_progress",
            "an open that is STILL RUNNING was reported as %r / %r - the tile "
            "asserted a failure that has not happened" % (r["state"], r["message"]))
        self.assertEqual(r["detail"], ERR_OPEN_IN_PROGRESS)   # raw, not prose

    def test_the_in_flight_message_asserts_no_fault_at_all(self):
        # It may say the open has not finished. It may not say anything broke,
        # and it may not send the owner to a cable, a plug or a power brick.
        self.dark("kinect")
        self.set_enum(True, ("Xbox NUI Sensor",))
        self.set_health(_health(open=False, open_error=ERR_OPEN_IN_PROGRESS,
                                cooldown_s=5.0))
        low = self.reason("kinect")["message"].lower()
        for lie in ("could not", "failed", "not detected", "power", "adapter",
                    "plugged", "missing", "stopped"):
            self.assertNotIn(lie, low, "in-flight message claims %r: %r" % (lie, low))

    def test_a_genuinely_completed_failure_still_reports_open_failed(self):
        # The new rung must not swallow the real thing it was carved out of: a
        # constructor that actually returned an error IS a finished failure.
        self.dark("kinect")
        self.set_enum(True, ("Xbox NUI Sensor",))
        self.set_health(_health(open=False, open_error=ERR_CTOR))
        self.assertEqual(self.reason("kinect")["state"], "open_failed")
        self.assertNotIn(wi._KINECT_OPEN_IN_PROGRESS, ERR_CTOR)

    def test_a_latched_completed_verdict_still_wins_over_the_marker(self):
        # The bridge returns `_open_error[0] or "<in progress>"`, so a caller that
        # loses the acquire while a REAL verdict is already latched gets that
        # verdict's text back. Those keep their own rungs - the in-flight rung is
        # checked after them, and must not steal them.
        self.dark("kinect")
        self.set_enum(True, ("Xbox NUI Sensor",))
        for err, want in ((ERR_NO_FRAMES, "no_frames"),
                          (ERR_NO_PYKINECT, "pykinect2_missing")):
            self.set_health(_health(open=False, open_error=err))
            self.assertEqual(self.reason("kinect")["state"], want, err)

    def test_the_bridge_really_emits_the_marker_this_rung_matches(self):
        """STALE-DUPLICATE GUARD. The rung matches a substring of a string that
        lives in ANOTHER file; if kinect_bridge rewords it, the tile would drop
        silently back to "JARVIS could not open the Kinect." with nothing
        failing. Fail here instead, loudly, in the copy that can be fixed."""
        src = os.path.join(wi.PROJECT_DIR, "audio", "kinect_bridge.py")
        with open(src, encoding="utf-8") as fh:
            bridge = fh.read()
        self.assertIn(
            ERR_OPEN_IN_PROGRESS, bridge,
            "audio/kinect_bridge.py no longer emits %r, so the ladder's "
            "in-flight rung matches nothing and an open that is merely still "
            "running renders as a completed failure again" % ERR_OPEN_IN_PROGRESS)
        self.assertIn(wi._KINECT_OPEN_IN_PROGRESS, ERR_OPEN_IN_PROGRESS)


class CameraReasonNotOpenNoErrorTests(_ReasonBase):
    """(open False, open_error None) is TWO situations, and the tile may not
    pick one of them.

    THE BUG THIS PINS (2026-09-05, reproduced against the real bridge).
    kinect_bridge._publish_runtime clears ``_open_error[0]`` on every SUCCESSFUL
    open, and reset_if_body_stale then drops ``_runtime[0]`` on the
    both-planes-stale path without ever writing an error. So a Kinect that
    opened, streamed for an hour and then went quiet on body AND color - the
    owner's "skeleton rendered for a while then stopped" intermittent - leaves
    get_stream_health() reading open False / open_error None: byte for byte the
    cold-start reading. The tile answered "Kinect state unknown - JARVIS has not
    tried the sensor yet." for the whole ~16 s reopen gauntlet, which sent him
    hunting a boot problem while the sensor was mid-death."""

    # What get_stream_health() returns in the instant AFTER reset_if_body_stale
    # tears a LIVE runtime down: no runtime, no error, the always-on pump still
    # alive, and BOTH frame clocks re-seeded by the reset itself - so pump_alive
    # and the *_age_s cells look exactly like a freshly enabled cold start too.
    TORN_DOWN = dict(open=False, open_error=None, pump_alive=True,
                     body_age_s=0.4, color_age_s=0.4)

    def setUp(self):
        super().setUp()
        self.dark("kinect")
        self.set_enum(True, ("Xbox NUI Sensor",))     # enumeration SUCCEEDS

    def test_a_sensor_that_died_mid_session_is_not_called_untried(self):
        self.set_health(_health(**self.TORN_DOWN))
        r = self.reason("kinect")
        msg = r["message"].lower()
        # The LIE first, so a regression fails on the sentence the owner reads
        # rather than on the state name, which is only its label.
        for lie in ("has not tried", "not tried the sensor", "never tried",
                    "has not probed", "state unknown"):
            self.assertNotIn(
                lie, msg,
                "the tile told the owner %r about a sensor that had just been "
                "streaming: %s" % (lie, r["message"]))
        self.assertEqual(r["state"], "not_open_no_error")

    def test_the_message_names_both_candidates_and_asserts_neither(self):
        self.set_health(_health(**self.TORN_DOWN))
        msg = self.reason("kinect")["message"].lower()
        self.assertIn("probed", msg)                  # candidate 1: never asked
        self.assertIn("dropped", msg)                 # candidate 2: died on us
        self.assertIn(" or ", msg)                    # ...offered as a choice
        # And it still blames no hardware it has not established.
        for blame in ("not detected", "power", "plugged", "missing"):
            self.assertNotIn(blame, msg, msg)

    def test_cold_start_and_torn_down_get_the_SAME_answer(self):
        """Neither reading may out-rank the other: the ladder cannot tell them
        apart, so producing two different sentences would mean it had guessed."""
        self.set_health(_health(open=False, open_error=None))
        cold = self.reason("kinect")
        self.set_health(_health(**self.TORN_DOWN))
        dead = self.reason("kinect")
        self.assertEqual(cold["state"], dead["state"])
        self.assertEqual(cold["message"], dead["message"])

    def test_detail_carries_observed_evidence_not_a_verdict(self):
        # detail lands in the tile's hover title. It may report what was seen -
        # the preview file's age - and must not name a cause.
        self.set_health(_health(**self.TORN_DOWN))
        d = self.reason("kinect").get("detail", "")
        self.assertIn("no runtime open", d)
        self.assertIn("no recorded open error", d)
        self.assertIn("none on disk", d)              # nothing was ever written
        path = self.lit("kinect")                     # now a frame DOES exist...
        old = time.time() - (wi._CAMERA_PREVIEW_STALE_S + 30)
        os.utime(path, (old, old))                    # ...but a stale one
        d2 = self.reason("kinect").get("detail", "")
        self.assertIn("last kinect preview frame:", d2)
        self.assertIn("s ago", d2)
        self.assertNotIn("none on disk", d2)

    def test_the_bridge_still_cannot_tell_these_two_apart(self):
        """STALE-DUPLICATE GUARD, and the trigger to SHARPEN this rung.

        The hedge is only correct while the bridge records nothing on teardown.
        The day reset_if_body_stale starts marking the runtime as torn down (or
        _publish_runtime starts latching a has-ever-opened flag), the tile can
        stop hedging - and this test fails to say so, in the copy that has to
        change, instead of the hedge quietly outliving its reason."""
        src = os.path.join(wi.PROJECT_DIR, "audio", "kinect_bridge.py")
        with open(src, encoding="utf-8") as fh:
            bridge = fh.read()
        tree = ast.parse(bridge)
        bodies = {n.name: ast.get_source_segment(bridge, n)
                  for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)}
        self.assertIn("_publish_runtime", bodies)
        self.assertIn("reset_if_body_stale", bodies)
        self.assertIn("_open_error[0] = None", bodies["_publish_runtime"],
                      "_publish_runtime no longer clears the error on a good "
                      "open - re-check whether the ladder still has to hedge")
        self.assertNotIn(
            "_open_error", bodies["reset_if_body_stale"],
            "reset_if_body_stale now writes _open_error, so a torn-down runtime "
            "is finally distinguishable from a never-probed one: the "
            "not_open_no_error rung can name which one it is")


def _js_fn(html, name):
    """The rendered source of one page function, so a test can assert about the
    body it cares about instead of about the whole 180 KB document."""
    start = html.index("function %s(" % name)
    return html[start:html.index("\n}", start)]


class MidStreamDeathTests(_ReasonBase):
    """A camera that dies WHILE STREAMING.

    THE DEFECT (2026-09-05): the tile could only ever be taken down by the
    <img> `error` event, and a server that closes a multipart/x-mixed-replace
    response after a COMPLETE part fires no such event. Measured against a real
    Chrome on this box, three different endings - a clean close after a
    complete part, a truncated final part, and boundary+headers then close -
    each produced exactly ONE event, `load` at 413 ms (just after the FIRST
    frame), and nothing at all at the close ~1.6 s later; img.complete stayed
    true and naturalWidth kept the last frame's size the whole time. So `load`
    is neither per-frame nor an end-of-stream signal, and NO DOM signal reports
    the ending at all.

    The live consequence, reproduced on this machine: with all three tiles
    streaming, stopping ONLY the kinect writer left the server 404ing
    /api/camera-preview?cam=kinect 61 s later while the browser tile still read
    streaming='1', errs='0', naturalWidth=160 and 'checking...' - a minutes-old
    frozen picture the dashboard was presenting as live, with the entire reason
    ladder unreachable behind it. That is the exact failure this whole feature
    exists to prevent: a tile asserting something it has not established.

    So the page cannot wait for an event, it has to ASK. These tests cover the
    fact it asks for (/api/camera-live) and the wiring that lets the tick take a
    streaming tile down with no DOM event involved.
    """

    def live(self):
        code, j = _get(self.base + "/api/camera-live")
        self.assertEqual(code, 200)
        return j["cams"]

    def test_camera_live_answers_for_every_tile_independently(self):
        self.lit("left")
        self.dark("right")
        self.dark("kinect")
        self.assertEqual(self.live(),
                         {"left": True, "right": False, "kinect": False})

    def test_camera_live_flips_at_the_same_instant_the_stream_closes(self):
        """The supervisor and the thing it supervises must share ONE rule.

        If /api/camera-live were even slightly more patient than
        _stream_camera's staleness test, the tick would keep re-arming a stream
        the server has already decided to close, and the tile would flap."""
        self.lit("kinect")
        self.assertTrue(self.live()["kinect"])
        path = wi._preview_path_for(
            {"camera_preview_path": self.camera_preview_path}, "kinect")
        old = time.time() - (wi._CAMERA_PREVIEW_STALE_S + 1.0)
        os.utime(path, (old, old))
        self.assertFalse(self.live()["kinect"],
                         "the supervisor still calls this tile live after the "
                         "stream route would have closed it")
        # ...and the stream route agrees, right now, about the same file. Safe
        # to request only because it is STALE: a fresh one would stream forever.
        code, _ = _get_raw(self.base + "/api/camera-stream?cam=kinect")
        self.assertEqual(code, 404)

    def test_camera_live_never_runs_the_device_probe(self):
        """It sits on the POLL path (about 1 Hz while the Camera tab is open),
        so it must never reach the ~0.7 s powershell sweep behind the ladder."""
        self.dark("kinect")
        self.set_health(_health(open=False, open_error=ERR_NO_FRAMES))
        for _ in range(5):
            self.live()
        self.assertEqual(self.probe_calls, [],
                         "the liveness poll spawned the device probe: %s"
                         % self.probe_calls)

    def test_the_tick_can_take_a_streaming_tile_down_with_no_dom_event(self):
        """THE REGRESSION TEST.

        Before the fix nothing in the tick could clear a tile's
        dataset.streaming - only the `error` listener did, and that listener
        never runs for a mid-stream close. So startCameraStreams() skipped the
        tile forever and the frozen frame stayed up."""
        code, html = _get_raw(self.base + "/")
        self.assertEqual(code, 200)

        tick = html.split("function refreshCamera")[1]
        self.assertIn("checkCameraLiveness()", tick,
                      "the supervisor tick does not consult the server, so a "
                      "camera that dies mid-stream is again invisible to it")

        live = _js_fn(html, "checkCameraLiveness")
        self.assertIn("/api/camera-live", live)
        self.assertIn("tileDown(", live,
                      "the liveness answer is fetched but cannot take the "
                      "tile down")

        down = _js_fn(html, "tileDown")
        self.assertIn("dataset.streaming = ''", down,
                      "a tile taken down still looks connected, so the tick "
                      "will keep skipping it")
        self.assertIn("explainTile(", down,
                      "the tile goes dark with no sentence - a blank tile and "
                      "no reason is the state this feature exists to remove")

    def test_a_tile_can_never_be_hidden_without_being_explained(self):
        """Hiding lives in exactly ONE function. Two copies of 'hide the tile'
        is how one of them ends up without the explanation."""
        _code, html = _get_raw(self.base + "/")
        self.assertEqual(html.count("off.style.display='flex'"), 1,
                         "more than one place hides a tile behind its "
                         "placeholder; they will drift")
        self.assertIn("off.style.display='flex'", _js_fn(html, "tileDown"))

    def test_liveness_may_only_ever_take_a_tile_down(self):
        """A fresh FILE is not proof that THIS browser is receiving frames, so
        the supervisor must not be able to declare a tile live - that stays the
        stream's job, on evidence the stream itself delivered."""
        live = _js_fn(html=_get_raw(self.base + "/")[1],
                      name="checkCameraLiveness")
        for up in ("display='block'", 'display="block"', "streaming = '1'"):
            self.assertNotIn(up, live,
                             "the liveness poll brings a tile UP on file "
                             "freshness alone: %r" % up)

    def test_the_stream_docstring_no_longer_claims_the_close_fires_error(self):
        """The claim was measured false in Chrome. It sat in the code for a day
        and is exactly why the client was built to wait for an event that never
        comes, so it must not come back."""
        with open(wi.__file__, encoding="utf-8") as fh:
            src = fh.read()
        body = src[src.index("def _stream_camera("):]
        body = body[:body.index('"""', body.index('"""') + 3)]
        self.assertNotIn("fires the same `error` on the client", body)
        self.assertIn("/api/camera-live", body,
                      "the docstring does not say what actually notices the "
                      "close, so the next reader re-learns it the hard way")


class CameraLiveTokenTests(_ReasonBase):
    token = "s3cr3t"

    def test_liveness_requires_the_token(self):
        code, _ = _get_raw(self.base + "/api/camera-live")
        self.assertEqual(code, 401)
        code, _ = _get_raw(self.base + "/api/camera-live",
                           headers={"X-Auth-Token": self.token})
        self.assertEqual(code, 200)


if __name__ == "__main__":
    unittest.main()
