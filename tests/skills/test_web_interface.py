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
        """Load the skill with a fake engine + fake config pinned, THEN run
        register(). Returns (mod, actions).

        register() is deferred (register=False) until the engine + config are
        pinned: letting it run at load time made the whole suite depend on
        the OWNER'S LIVE STATE — with WEB_INTERFACE_ENABLED true in the real
        settings and port 8766 momentarily free, every load auto-started a
        REAL ThreadingHTTPServer whose harness-neutered serve thread then
        wedged shutdown() FOREVER (the 2026-07-12 ci_sim freeze; the suite
        had only ever passed because the live JARVIS normally keeps 8766
        occupied and the autostart failed cleanly)."""
        if engine is not None:
            sys.modules["tools.web_interface"] = engine
        mod, _ = load_skill_isolated("web_interface", register=False)
        if engine is not None:
            mod._engine = engine
            mod._HAS_ENGINE = True
        pinned = dict(cfg) if cfg is not None else {}
        # Never let a test's register() consult the live config knob.
        pinned.setdefault("WEB_INTERFACE_ENABLED", False)
        mod._cfg = lambda name, default, _c=pinned: _c.get(name, default)
        actions: dict = {}
        mod.register(actions)
        self.mod = mod
        return mod, actions

    # ── _stop timebox (2026-07-12) ───────────────────────────────────────────
    def test_stop_survives_wedged_shutdown(self):
        # socketserver.shutdown() waits on an event only serve_forever() sets
        # on exit — a serve thread that never started (spawn race) or already
        # died made shutdown() block FOREVER and froze an entire ci_sim run
        # mid-suite. A wedged shutdown() must not hang _stop, and
        # server_close() must still free the port.
        import threading as _t
        import time as _time
        eng = _fake_engine()
        mod, _actions = self._load(engine=eng,
                                   cfg={"WEB_INTERFACE_ENABLED": False})

        never = _t.Event()                    # never set → blocks forever
        httpd = mock.Mock()
        httpd.shutdown.side_effect = lambda: never.wait()
        closed = _t.Event()
        httpd.server_close.side_effect = closed.set
        mod._httpd = httpd
        mod._serve_thread = None
        mod._bound = ("127.0.0.1", 1)

        t0 = _time.monotonic()
        with mock.patch("builtins.print"):
            ok, msg = mod._stop()
        dt = _time.monotonic() - t0
        self.assertTrue(ok)
        self.assertIn("off", msg.lower())
        # returned promptly (5s shutdown box + slack), socket still closed
        self.assertLess(dt, 12.0)
        self.assertTrue(closed.is_set(),
                        "server_close must run even when shutdown() wedges")
        never.set()                           # release the daemon worker

    # ── registration + off-by-default ────────────────────────────────────────
    def test_actions_registered(self):
        eng = _fake_engine()
        _mod, actions = self._load(engine=eng,
                                   cfg={"WEB_INTERFACE_ENABLED": False})
        for name in ("web_interface_on", "web_interface_off",
                     "web_interface_status"):
            self.assertIn(name, actions)

    def test_off_by_default_starts_nothing(self):
        # register() with the knob False must start NO server — no LAN socket
        # opens uninvited at boot. (_load pins the knob False and defers
        # register until the fake engine is in place, so this is deterministic
        # regardless of the owner's live settings or port state.)
        eng = _fake_engine()
        mod, _actions = self._load(engine=eng,
                                   cfg={"WEB_INTERFACE_ENABLED": False})
        self.assertIsNone(mod._httpd)

    def test_knob_true_auto_starts(self):
        # Auto-start is the ENABLED-True branch of register(). _load defers
        # register() until the FAKE engine + True-knob config are pinned, so
        # the auto-start exercises the fake — never a real socket.
        eng = _fake_engine()
        mod, actions = self._load(
            engine=eng,
            cfg={"WEB_INTERFACE_ENABLED": True, "WEB_INTERFACE_BIND": "127.0.0.1",
                 "WEB_INTERFACE_PORT": 8766, "WEB_INTERFACE_TOKEN": ""})
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
        # Deferred register (no live-config autostart), THEN break the engine.
        mod, _ = load_skill_isolated("web_interface", register=False)
        self.mod = mod
        mod._HAS_ENGINE = False
        mod._engine = None
        mod._cfg = lambda name, default: {"WEB_INTERFACE_ENABLED": False
                                          }.get(name, default)
        actions: dict = {}
        mod.register(actions)
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
            # Deferred register: pin the fake engine + config FIRST so the
            # load can never auto-start a real server off the live settings.
            mod, _ = load_skill_isolated("web_interface", register=False)
            mod._engine = eng
            mod._HAS_ENGINE = True
            mod._cfg = lambda name, default: {
                "WEB_INTERFACE_BIND": "127.0.0.1", "WEB_INTERFACE_PORT": 8766,
                "WEB_INTERFACE_TOKEN": "", "WEB_INTERFACE_ENABLED": False,
            }.get(name, default)
            actions: dict = {}
            mod.register(actions)
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
