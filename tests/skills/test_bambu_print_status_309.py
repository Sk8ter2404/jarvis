"""v3.0.9 lane B: a full printer status on connect and honest print-status
replies (skills/bambu_monitor.py).

Live 2026-10-28 21:18 "how much longer [on my] print?" ran check_print and
got "Printing '<name>', sir." - no layer, percent or time - and the brain
then said it could not see the time remaining. The printer names the time
left mc_remaining_time and the layer count total_layer_num, and the monitor
only read mc_remaining / total_layer (the overlay state on file after a whole
print: layer 334, percent 100, both null). It also asked for a full status
("pushall") only after 5 silent minutes, so a restart left the figures
unknown until they changed.

Pinned, with each flag on and off (off = the old path exactly):
  * BAMBU_PUSHALL_ON_CONNECT - one read-only pushall on every (re)connect,
    to device/<serial>/request, and nothing else is published;
  * PRINT_STATUS_HONEST - the printer's own field names are read; check_print
    / how_is_the_print give percent, layer and time left, say FINISH /
    FAILED / PAUSE / PREPARE plainly, and say why the time is missing.

Paraphrased fixtures; no network, no printer.

    python -m unittest tests.skills.test_bambu_print_status_309
"""
from __future__ import annotations

import contextlib
import io
import json
import time
import types
import unittest
from unittest import mock

from tests.skills.test_bambu_monitor import _load_bambu


class _Base(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = _load_bambu()
        with self.mod._state_lock:
            for k in list(self.mod._state):
                self.mod._state[k] = None if k != "last_update" else 0.0
        import core.config as cfg
        self.cfg = cfg

    def _flag(self, name, value):
        p = mock.patch.object(self.cfg, name, value, create=True)
        p.start()
        self.addCleanup(p.stop)

    def _set(self, **kw):
        kw.setdefault("last_update", time.time())
        with self.mod._state_lock:
            self.mod._state.update(kw)

    def _msg(self, report):
        m = mock.MagicMock()
        m.payload = json.dumps({"print": report}).encode("utf-8")
        with mock.patch.object(self.mod, "_handle_state_change"), \
             mock.patch.object(self.mod, "_write_overlay_state"):
            self.mod._on_message(None, None, m)


class PrinterFieldNameTests(_Base):
    LIVE = {"gcode_state": "RUNNING", "layer_num": 12,
            "total_layer_num": 334, "mc_percent": 4,
            "mc_remaining_time": 151, "subtask_name": "bracket.3mf"}

    def test_the_printers_own_names_are_read(self):
        self._flag("PRINT_STATUS_HONEST", True)
        self._msg(dict(self.LIVE))
        with self.mod._state_lock:
            self.assertEqual(self.mod._state["mc_remaining"], 151)
            self.assertEqual(self.mod._state["total_layer"], 334)

    def test_flag_off_reads_only_the_old_names(self):
        self._flag("PRINT_STATUS_HONEST", False)
        self._msg(dict(self.LIVE))
        with self.mod._state_lock:
            self.assertIsNone(self.mod._state["mc_remaining"])
            self.assertIsNone(self.mod._state["total_layer"])
            self.assertEqual(self.mod._state["layer_num"], 12)

    def test_the_live_question_now_gets_the_time_left(self):
        self._flag("PRINT_STATUS_HONEST", True)
        self._msg(dict(self.LIVE))
        out = self.actions["check_print"]("")
        self.assertIn("4 percent done", out)
        self.assertIn("layer 12 of 334", out)
        self.assertIn("about 2 hours and 31 minutes remaining", out)


class HonestReplyTests(_Base):
    def setUp(self):
        super().setUp()
        self._flag("PRINT_STATUS_HONEST", True)

    def test_running_without_an_estimate_says_why(self):
        self._set(gcode_state="RUNNING", mc_percent=0, layer_num=1,
                  filename="bracket.3mf")
        out = self.actions["check_print"]("")
        self.assertIn("Printing 'bracket'", out)
        self.assertIn("just starting", out)
        self.assertIn("hasn't sent its time estimate yet, sir", out)
        self.assertNotIn("can't see", out)

    def test_finish_failed_pause_prepare_are_said_plainly(self):
        self._set(gcode_state="FINISH", filename="bracket.3mf")
        self.assertIn("has finished", self.actions["check_print"](""))
        self._set(gcode_state="FAILED", layer_num=47)
        out = self.actions["check_print"]("")
        self.assertIn("failed at layer 47", out)
        self.assertNotIn("Printing", out)
        self._set(gcode_state="PAUSE", mc_percent=40, total_layer=100,
                  layer_num=40)
        out = self.actions["check_print"]("")
        self.assertIn("is paused, 40 percent done, layer 40 of 100", out)
        self._set(gcode_state="PREPARE")
        out = self.actions["check_print"]("")
        self.assertIn("getting ready to print 'bracket'", out)
        self.assertIn("hasn't sent its time estimate yet", out)
        self._set(gcode_state="IDLE")
        self.assertIn("No active print", self.actions["check_print"](""))

    def test_no_print_state_yet_is_not_called_idle(self):
        self._set(gcode_state=None, nozzle_temper=25.0)
        out = self.actions["check_print"]("")
        self.assertIn("hasn't sent its print state yet", out)
        self.assertNotIn("idle", out.lower())

    def test_how_is_the_print_adds_the_temperatures(self):
        self._set(gcode_state="RUNNING", mc_percent=50, layer_num=10,
                  total_layer=100, mc_remaining=30, filename="part.3mf",
                  nozzle_temper=220.0, bed_temper=60.0)
        out = self.actions["how_is_the_print"]("")
        self.assertEqual(out, "Printing 'part', 50 percent done, layer 10 of "
                              "100, about 30 minutes remaining — nozzle at "
                              "220 degrees and bed at 60 degrees, sir.")
        self.assertIs(self.actions["print_details"],
                      self.actions["how_is_the_print"])

    def test_no_status_at_all_still_says_why(self):
        self.assertIn("fresh status", self.actions["check_print"]("").lower())

    def test_a_bad_snapshot_falls_back_to_the_no_status_reply(self):
        self._set(gcode_state="RUNNING")
        with mock.patch.object(self.mod, "_stale_prefix",
                               side_effect=RuntimeError("boom")):
            out = self.actions["check_print"]("")
        self.assertIn("printer", out.lower())


class OldRepliesTests(_Base):
    def test_flag_off_is_the_old_reply_exactly(self):
        self._flag("PRINT_STATUS_HONEST", False)
        self._set(gcode_state="FAILED", layer_num=47, filename="bracket.3mf")
        self.assertEqual(self.actions["check_print"](""),
                         "Printing 'bracket', sir.")
        self._set(gcode_state="RUNNING", layer_num=1, mc_percent=0)
        self.assertEqual(self.actions["check_print"](""),
                         "Printing 'bracket', sir.")


class PushallOnConnectTests(_Base):
    def _connect(self):
        fake = types.SimpleNamespace(MQTTv311=4,
                                     ssl=types.SimpleNamespace(CERT_NONE=0))
        client = mock.MagicMock()
        fake.Client = mock.MagicMock(return_value=client)
        with mock.patch.object(self.mod, "mqtt", fake), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.mod._start_mqtt("printer.test", "code", "SERIAL1")
            client.on_connect(client, None, None, 0)
        return client, out.getvalue()

    def test_one_full_status_request_on_connect(self):
        self._flag("BAMBU_PUSHALL_ON_CONNECT", True)
        client, out = self._connect()
        client.publish.assert_called_once()
        topic, payload = client.publish.call_args.args
        self.assertEqual(topic, "device/SERIAL1/request")
        self.assertEqual(json.loads(payload),
                         {"pushing": {"sequence_id": "0",
                                      "command": "pushall"}})
        self.assertIn("asked the printer for a full status", out)
        # A reconnect asks again - once per connect, nothing else is sent.
        client.on_connect(client, None, None, 0)
        self.assertEqual(client.publish.call_count, 2)
        for call in client.publish.call_args_list:
            self.assertIn('"pushall"', call.args[1])

    def test_a_failed_connect_sends_nothing(self):
        self._flag("BAMBU_PUSHALL_ON_CONNECT", True)
        client, _out = self._connect()
        client.publish.reset_mock()
        with contextlib.redirect_stdout(io.StringIO()):
            client.on_connect(client, None, None, 5)
        client.publish.assert_not_called()

    def test_flag_off_sends_nothing_on_connect(self):
        self._flag("BAMBU_PUSHALL_ON_CONNECT", False)
        client, _out = self._connect()
        client.publish.assert_not_called()
        client.subscribe.assert_called_once_with("device/SERIAL1/report")

    def test_a_publish_error_does_not_break_the_connect(self):
        self._flag("BAMBU_PUSHALL_ON_CONNECT", True)
        client = mock.MagicMock()
        client.publish.side_effect = RuntimeError("not connected")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertFalse(self.mod._request_pushall(client, "SERIAL1"))
        self.assertIn("pushall on connect failed", out.getvalue())
        self.assertFalse(self.mod._request_pushall(None, "SERIAL1"))
        self.assertFalse(self.mod._request_pushall(client, ""))


if __name__ == "__main__":
    unittest.main()
