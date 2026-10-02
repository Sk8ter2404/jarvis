"""The Bambu monitor says WHY it has no printer status (NEW #3, 2026-10-02).

THE LIVE EVIDENCE (session_2026-10-01_19-43-10.log / _21-48-36.log): print
status, progress callouts and finish alerts were dead all session and not one
line said so. "How's my print doing?" (21:36) got an invented excuse, then
"isn't responding". A probe at 22:06 found the printer IP answering ping with
TCP 8883 closed. _start_mqtt used connect_async + loop_start and logged only
inside on_connect - there was no on_connect_fail, so a refused or timed-out
connect retried silently forever - and how_is_the_print could only say
"Either it isn't reachable or the monitor hasn't connected."

  * on_connect_fail is set; a failed attempt is classified with one bounded
    TCP probe (refused / no answer / unreachable) and logged with its reason.
  * every connection state CHANGE is logged once with its reason (a repeat of
    the same failure only counts - no log spam every reconnect).
  * "never connected since boot" is tracked (connection_status()).
  * how_is_the_print / check_print say why there is no status and give the
    hint that fits: power, LAN / developer mode, the configured IP, or the
    access code.

Light tier: paho and the socket are faked; nothing touches the network.
Addresses are documentation-range (RFC 5737).

    python -m unittest tests.skills.test_bambu_connection_health
"""
from __future__ import annotations

import contextlib
import io
import socket
import time
import types
import unittest
from unittest import mock

from tests.skills.test_bambu_monitor import _load_bambu

IP = "192.0.2.10"


class _Base(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0
        self.mod._mqtt_connected_ok[0] = False
        reset = getattr(self.mod, "_reset_connection_health", None)
        if reset:
            reset()

    def _fake_mqtt(self):
        fake = types.SimpleNamespace()
        fake.MQTTv311 = 4
        fake.ssl = types.SimpleNamespace(CERT_NONE=0)
        client = mock.MagicMock()
        fake.Client = mock.MagicMock(return_value=client)
        fake.CallbackAPIVersion = types.SimpleNamespace(VERSION2="v2")
        return fake, client

    def _start(self):
        """Run the real _start_mqtt against a fake paho; return the client
        with the callbacks it installed, and the captured stdout."""
        fake, client = self._fake_mqtt()
        buf = io.StringIO()
        with mock.patch.object(self.mod, "mqtt", fake), \
             mock.patch.object(self.mod, "_HAS_MQTT", True), \
             contextlib.redirect_stdout(buf):
            self.mod._start_mqtt(IP, "code", "serial")
        return client, buf

    def _status(self):
        fn = getattr(self.mod, "connection_status", None)
        self.assertIsNotNone(fn, "no connection_status() - nothing records "
                                 "why the printer is unreachable")
        return fn()

    def _fail(self, client, exc):
        """Fire paho's on_connect_fail with the TCP probe raising ``exc``."""
        cb = client.on_connect_fail
        # On the fake client an attribute nobody set is an auto-Mock.
        self.assertNotIsInstance(
            cb, mock.Mock,
            "no on_connect_fail: a refused connect retries silently")
        buf = io.StringIO()
        with mock.patch.object(self.mod.socket, "create_connection",
                               side_effect=exc), \
             contextlib.redirect_stdout(buf):
            cb(client, None)
        return buf.getvalue()


class ConnectFailureTests(_Base):

    def test_refused_connect_is_logged_with_reason(self):
        client, _ = self._start()
        out = self._fail(client, ConnectionRefusedError(10061, "refused"))
        self.assertIn("[bambu]", out)
        self.assertIn("refused", out.lower())
        self.assertIn(IP, out)
        st = self._status()
        self.assertEqual(st["state"], "failed")
        self.assertEqual(st["kind"], "refused")
        self.assertFalse(st["ever_connected"])
        self.assertTrue(st["never_connected_since_boot"])
        self.assertEqual(st["failures"], 1)

    def test_a_repeated_failure_counts_but_logs_once(self):
        client, _ = self._start()
        self._fail(client, ConnectionRefusedError(10061, "refused"))
        out2 = self._fail(client, ConnectionRefusedError(10061, "refused"))
        self.assertEqual(out2.strip(), "",
                         "the same failure re-logged on every reconnect")
        self.assertEqual(self._status()["failures"], 2)

    def test_no_answer_and_unreachable_are_told_apart(self):
        client, _ = self._start()
        self._fail(client, socket.timeout("timed out"))
        self.assertEqual(self._status()["kind"], "timeout")
        err = OSError(10065, "A socket operation was attempted to an "
                             "unreachable host")
        err.winerror = 10065
        self._fail(client, err)
        self.assertEqual(self._status()["kind"], "unreachable")

    def test_rejected_access_code_is_auth(self):
        client, _ = self._start()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            client.on_connect(client, None, None, 5)
        st = self._status()
        self.assertEqual(st["kind"], "auth")
        self.assertIn("access code", st["hint"].lower())
        self.assertIn("[bambu]", buf.getvalue())

    def test_connect_then_drop_is_logged_and_remembered(self):
        client, _ = self._start()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            client.on_connect(client, None, None, 0)
        st = self._status()
        self.assertEqual(st["state"], "connected")
        self.assertTrue(st["ever_connected"])
        self.assertFalse(st["never_connected_since_boot"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            client.on_disconnect(client, None, None, 7, None)
        self.assertIn("disconnect", buf.getvalue().lower())
        st = self._status()
        self.assertEqual(st["state"], "disconnected")
        self.assertTrue(st["ever_connected"], "a drop is not 'never connected'")

    def test_a_crashing_probe_never_escapes_into_paho(self):
        client, _ = self._start()
        with mock.patch.object(self.mod, "_classify_connect_failure",
                               side_effect=RuntimeError("boom"),
                               create=True), \
             contextlib.redirect_stdout(io.StringIO()):
            client.on_connect_fail(client, None)     # must not raise
        self.assertEqual(self._status()["state"], "failed")


class NoStatusReplyTests(_Base):

    def test_how_is_the_print_says_why_refused(self):
        client, _ = self._start()
        self._fail(client, ConnectionRefusedError(10061, "refused"))
        out = self.actions["how_is_the_print"]("")
        low = out.lower()
        self.assertNotIn("either it isn't reachable", low,
                         "still the generic can't-say-why line")
        self.assertIn("since i started", low)
        self.assertIn(IP, out)
        self.assertTrue("lan" in low or "developer mode" in low, out)

    def test_how_is_the_print_hints_power_when_nothing_answers(self):
        client, _ = self._start()
        self._fail(client, socket.timeout("timed out"))
        low = self.actions["how_is_the_print"]("").lower()
        self.assertIn("powered", low)
        self.assertIn(IP, low)

    def test_check_print_gives_the_same_reason(self):
        client, _ = self._start()
        self._fail(client, ConnectionRefusedError(10061, "refused"))
        low = self.actions["check_print"]("").lower()
        self.assertIn("since i started", low)

    def test_connected_but_silent_says_so(self):
        client, _ = self._start()
        with contextlib.redirect_stdout(io.StringIO()):
            client.on_connect(client, None, None, 0)
        low = self.actions["how_is_the_print"]("").lower()
        self.assertIn("connected", low)
        self.assertNotIn("since i started", low)

    def test_a_stale_snapshot_after_a_drop_is_not_reported_as_live(self):
        client, _ = self._start()
        with contextlib.redirect_stdout(io.StringIO()):
            client.on_connect(client, None, None, 0)
            client.on_disconnect(client, None, None, 7, None)
        with self.mod._state_lock:
            self.mod._state.update(
                gcode_state="RUNNING", layer_num=10, total_layer=100,
                mc_remaining=30, filename="part.3mf",
                last_update=time.time() - 3600)
        low = self.actions["how_is_the_print"]("").lower()
        self.assertIn("connection", low)
        self.assertIn("ago", low)
        self.assertIn("layer 10 of 100", low)

    def test_unconfigured_stays_the_old_line(self):
        # No monitor ever started: nothing to explain beyond the old text.
        out = self.actions["how_is_the_print"]("")
        self.assertIn("fresh status", out.lower())


class CompanionPrintStatusTests(_Base):
    """skills/bambu_h2d_voice_companion's `print_status` is a second copy of
    how_is_the_print. Its no-status line was the same can't-say-why text, so
    it must ask the live monitor for the reason too."""

    def test_print_status_gives_the_monitor_reason(self):
        from tests.skills.test_bambu_h2d_voice_companion import (
            VoiceCompanionMixin)
        client, _ = self._start()
        self._fail(client, ConnectionRefusedError(10061, "refused"))

        class _T(VoiceCompanionMixin, unittest.TestCase):
            def runTest(self):   # pragma: no cover - never run as a test
                pass
        t = _T()
        try:
            _cmod, actions = t._load(bambu_module=self.mod)
            out = actions["print_status"]("")
        finally:
            t.doCleanups()
        self.assertIn("since i started", out.lower())
        self.assertIn(IP, out)


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
