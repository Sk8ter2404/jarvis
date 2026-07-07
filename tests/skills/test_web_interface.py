"""Tests for skills/web_interface — the web-interface lifecycle wiring.

Loads the skill through tests/_skill_harness.load_skill_isolated (the same
injection contract as the real loader). No sockets are actually bound in most
tests: the tools.web_interface engine is patched with a fake so the skill's
start/stop/status logic is exercised without opening a real port — every test
passes on headless Linux CI with no real JARVIS and no network.

Asserts the safety + behaviour contract the feature ships on:
  • web_interface_on / _off / _status are registered.
  • OFF BY DEFAULT: with WEB_INTERFACE_ENABLED False (the shipped default),
    register() starts NO server — no LAN socket opens uninvited at boot.
  • Knob True → register() DOES auto-start (owner opt-in).
  • The security refusal (non-local bind + empty token → InsecureBindError)
    surfaces as an honest spoken sentence, and NO server is left running.
  • Voice on starts it even with the knob False (explicit command = consent);
    voice off stops it and reports off; status reflects running/where.
  • Staging refuses to start.
  • Graceful when a running JARVIS / log is absent (the engine is file-based, so
    a temp/fake works) and when the engine import failed entirely.

stdlib unittest + mock only; no pytest; no real sockets left listening.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _FakeInsecureBindError(RuntimeError):
    pass


def _fake_engine(*, insecure=False, oserror=False):
    """A stand-in tools.web_interface exposing exactly what the skill touches:
    create_server / serve_in_thread / is_local_bind / InsecureBindError /
    FLASK_AVAILABLE. create_server returns a fake httpd (records shutdown), or
    raises the configured error to exercise the refusal / port-clash paths."""
    m = types.ModuleType("tools.web_interface")
    m.InsecureBindError = _FakeInsecureBindError
    m.FLASK_AVAILABLE = False
    m.is_local_bind = lambda b: (b or "").strip().lower() in (
        "127.0.0.1", "localhost", "::1")

    class _FakeHTTPD:
        def __init__(self, bind, port, token):
            self.config = {"token": token, "local_bind": m.is_local_bind(bind)}
            self.shutdown_called = False
            self.closed = False

        def shutdown(self):
            self.shutdown_called = True

        def server_close(self):
            self.closed = True

    def create_server(*, bind, port, token="", **_kw):
        if insecure:
            raise _FakeInsecureBindError("refusing non-local bind, empty token")
        if oserror:
            raise OSError(98, "Address already in use")
        return _FakeHTTPD(bind, port, token)

    m.create_server = create_server
    m._last_thread = None

    def serve_in_thread(httpd):
        t = mock.MagicMock(name="serve_thread")
        m._last_thread = t
        return t

    m.serve_in_thread = serve_in_thread
    return m


class WebInterfaceSkillTest(unittest.TestCase):
    def setUp(self):
        self._saved_engine = sys.modules.get("tools.web_interface")
        self._saved_staging = os.environ.pop("JARVIS_STAGING", None)
        self.mod = None

    def tearDown(self):
        # Stop any server the test started so nothing lingers.
        if self.mod is not None:
            try:
                self.mod._stop()
            except Exception:
                pass
        if self._saved_engine is not None:
            sys.modules["tools.web_interface"] = self._saved_engine
        else:
            sys.modules.pop("tools.web_interface", None)
        if self._saved_staging is not None:
            os.environ["JARVIS_STAGING"] = self._saved_staging
        else:
            os.environ.pop("JARVIS_STAGING", None)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _load(self, *, engine=None, cfg=None):
        """Load the skill with a fake engine + fake config pinned. Returns
        (mod, actions)."""
        if engine is not None:
            sys.modules["tools.web_interface"] = engine
        mod, actions = load_skill_isolated("web_interface")
        # The skill reads config fresh via `from core import config`. Patch the
        # engine + config lookups on the loaded module for deterministic tests.
        if engine is not None:
            mod._engine = engine
            mod._HAS_ENGINE = True
        if cfg is not None:
            mod._cfg = lambda name, default, _c=cfg: _c.get(name, default)
        self.mod = mod
        return mod, actions

    # ── registration + off-by-default ────────────────────────────────────────
    def test_actions_registered(self):
        eng = _fake_engine()
        _mod, actions = self._load(engine=eng,
                                   cfg={"WEB_INTERFACE_ENABLED": False})
        for name in ("web_interface_on", "web_interface_off",
                     "web_interface_status"):
            self.assertIn(name, actions)

    def test_off_by_default_starts_nothing(self):
        # With the fake engine pinned AFTER load, register() has already run with
        # the real (default-False) config, so no server should have auto-started.
        # Re-load with the fake + knob False and assert _httpd is None.
        eng = _fake_engine()
        sys.modules["tools.web_interface"] = eng
        # Patch config so the module's fresh read sees the knob False.
        with mock.patch.dict("os.environ", {}, clear=False):
            mod, _actions = load_skill_isolated("web_interface")
            self.mod = mod
            mod._engine = eng
            mod._HAS_ENGINE = True
            # register already ran during load with real config; ensure nothing
            # is running regardless.
            self.assertIsNone(mod._httpd)

    def test_knob_true_auto_starts(self):
        # Auto-start is the ENABLED-True branch of register(). We exercise it
        # deterministically by loading the skill (register runs with the shipped
        # default False → nothing starts), then pinning the FAKE engine + a
        # True-knob config onto the module and re-running register(). This avoids
        # depending on load-time `from tools import web_interface` binding (the
        # `tools` package may already hold the real submodule from a prior test).
        eng = _fake_engine()
        mod, actions = self._load(
            engine=eng,
            cfg={"WEB_INTERFACE_ENABLED": True, "WEB_INTERFACE_BIND": "127.0.0.1",
                 "WEB_INTERFACE_PORT": 8766, "WEB_INTERFACE_TOKEN": ""})
        # After _load, register already ran once at load time; nothing is running
        # yet (that load saw the real default-False knob). Re-run register with
        # the fake engine + True knob now pinned.
        fresh_actions: dict = {}
        mod.register(fresh_actions)
        self.assertIsNotNone(mod._httpd)
        self.assertEqual(mod._bound, ("127.0.0.1", 8766))

    # ── voice on / off / status ──────────────────────────────────────────────
    def test_voice_on_starts_even_when_knob_false(self):
        eng = _fake_engine()
        mod, actions = self._load(
            engine=eng,
            cfg={"WEB_INTERFACE_ENABLED": False, "WEB_INTERFACE_BIND": "127.0.0.1",
                 "WEB_INTERFACE_PORT": 8766, "WEB_INTERFACE_TOKEN": ""})
        msg = actions["web_interface_on"]("")
        self.assertIn("online", msg.lower())
        self.assertIsNotNone(mod._httpd)

    def test_voice_off_stops_and_reports(self):
        eng = _fake_engine()
        mod, actions = self._load(
            engine=eng,
            cfg={"WEB_INTERFACE_BIND": "127.0.0.1", "WEB_INTERFACE_PORT": 8766,
                 "WEB_INTERFACE_TOKEN": ""})
        actions["web_interface_on"]("")
        httpd = mod._httpd
        msg = actions["web_interface_off"]("")
        self.assertIn("off", msg.lower())
        self.assertIsNone(mod._httpd)
        self.assertTrue(httpd.shutdown_called)
        self.assertTrue(httpd.closed)

    def test_status_reflects_running(self):
        eng = _fake_engine()
        mod, actions = self._load(
            engine=eng,
            cfg={"WEB_INTERFACE_BIND": "127.0.0.1", "WEB_INTERFACE_PORT": 8766,
                 "WEB_INTERFACE_TOKEN": "", "WEB_INTERFACE_ENABLED": False})
        self.assertIn("off", actions["web_interface_status"]("").lower())
        actions["web_interface_on"]("")
        running_msg = actions["web_interface_status"]("").lower()
        self.assertIn("running", running_msg)
        self.assertIn("8766", running_msg)

    def test_double_on_is_idempotent(self):
        eng = _fake_engine()
        mod, actions = self._load(
            engine=eng,
            cfg={"WEB_INTERFACE_BIND": "127.0.0.1", "WEB_INTERFACE_PORT": 8766,
                 "WEB_INTERFACE_TOKEN": ""})
        actions["web_interface_on"]("")
        msg2 = actions["web_interface_on"]("")
        self.assertIn("already running", msg2.lower())

    # ── security refusal + port clash ────────────────────────────────────────
    def test_insecure_bind_refused_gracefully(self):
        eng = _fake_engine(insecure=True)
        mod, actions = self._load(
            engine=eng,
            cfg={"WEB_INTERFACE_BIND": "0.0.0.0", "WEB_INTERFACE_PORT": 8766,
                 "WEB_INTERFACE_TOKEN": ""})
        msg = actions["web_interface_on"]("")
        self.assertIn("token", msg.lower())
        self.assertIsNone(mod._httpd)   # NOT left half-started

    def test_port_in_use_reported(self):
        eng = _fake_engine(oserror=True)
        mod, actions = self._load(
            engine=eng,
            cfg={"WEB_INTERFACE_BIND": "127.0.0.1", "WEB_INTERFACE_PORT": 8766,
                 "WEB_INTERFACE_TOKEN": ""})
        msg = actions["web_interface_on"]("")
        self.assertIn("couldn't start", msg.lower())
        self.assertIsNone(mod._httpd)

    # ── staging refusal ──────────────────────────────────────────────────────
    def test_staging_refuses_to_start(self):
        os.environ["JARVIS_STAGING"] = "1"
        eng = _fake_engine()
        mod, actions = self._load(
            engine=eng,
            cfg={"WEB_INTERFACE_BIND": "127.0.0.1", "WEB_INTERFACE_PORT": 8766,
                 "WEB_INTERFACE_TOKEN": ""})
        msg = actions["web_interface_on"]("")
        self.assertIn("staging", msg.lower())
        self.assertIsNone(mod._httpd)

    # ── engine-absent graceful path ──────────────────────────────────────────
    def test_engine_absent_actions_reply_gracefully(self):
        mod, actions = load_skill_isolated("web_interface")
        self.mod = mod
        mod._HAS_ENGINE = False
        mod._engine = None
        on_msg = actions["web_interface_on"]("")
        status_msg = actions["web_interface_status"]("")
        self.assertIn("didn't load", on_msg.lower())
        self.assertIn("isn't loaded", status_msg.lower())


class WebInterfaceVerbatimContractTest(unittest.TestCase):
    """The three actions must each return ONE non-empty string (they're spoken
    verbatim). A None/empty result would be dropped by _speak_verbatim_results."""

    def test_actions_return_nonempty_strings(self):
        eng = _fake_engine()
        sys.modules["tools.web_interface"] = eng
        try:
            mod, actions = load_skill_isolated("web_interface")
            mod._engine = eng
            mod._HAS_ENGINE = True
            mod._cfg = lambda name, default: {
                "WEB_INTERFACE_BIND": "127.0.0.1", "WEB_INTERFACE_PORT": 8766,
                "WEB_INTERFACE_TOKEN": "", "WEB_INTERFACE_ENABLED": False,
            }.get(name, default)
            for name in ("web_interface_status", "web_interface_on",
                         "web_interface_off"):
                out = actions[name]("")
                self.assertIsInstance(out, str)
                self.assertTrue(out.strip())
            mod._stop()
        finally:
            sys.modules.pop("tools.web_interface", None)


if __name__ == "__main__":
    unittest.main()
