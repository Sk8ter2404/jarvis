"""Logic tests for skills/sh_govee.py (Govee LAN + cloud controller).

Govee is unusual: is_available() is always True (LAN sockets are stdlib), so
graceful degradation here means set_state returns a clear error when a device
resolves via NEITHER the LAN scan NOR the cloud API. We also verify:
  * the LAN command payloads (turn / brightness / colorwc) are well-formed,
  * the colour-before-brightness ordering on the LAN path,
  * _api_key resolution from env / config,
  * govee_list messaging.
All sockets / requests are mocked — no UDP, no HTTP.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class GoveeListAndKeyTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_govee")

    def test_is_available_always_true(self):
        self.assertTrue(self.mod.is_available())

    def test_api_key_from_env(self):
        with mock.patch.dict(os.environ, {"GOVEE_API_KEY": "  abc123  "}):
            self.assertEqual(self.mod._api_key(), "abc123")

    def test_api_key_none_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertIsNone(self.mod._api_key())

    def test_govee_list_informative_when_empty(self):
        with mock.patch.object(self.mod, "list_devices", return_value=[]):
            out = self.actions["govee_list"]("")
        self.assertIn("No Govee devices", out)
        self.assertIn("GOVEE_API_KEY", out)

    def test_govee_list_names_devices(self):
        devs = [{"name": "Strip", "model": "H6159"},
                {"name": "Bulb", "model": "H6001"}]
        with mock.patch.object(self.mod, "list_devices", return_value=devs):
            out = self.actions["govee_list"]("")
        self.assertIn("2 Govee device", out)
        self.assertIn("Strip", out)


class GoveeSetStateDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_govee")

    def test_set_state_no_lan_no_cloud_match_errors(self):
        # No lan_ip on the record, and _cloud_match returns None → clean error.
        with mock.patch.object(self.mod, "_cloud_match", return_value=None):
            res = self.mod.set_state({"name": "Lamp"}, on=True)
        self.assertIn("not found", res["error"])

    def test_cloud_control_errors_without_key(self):
        with mock.patch.object(self.mod, "_api_key", return_value=None):
            res = self.mod._cloud_control({"device": "x", "model": "y"},
                                          "turn", "on")
        self.assertIn("no Govee API key", res["error"])

    def test_cloud_devices_empty_without_key(self):
        with mock.patch.object(self.mod, "_api_key", return_value=None):
            self.assertEqual(self.mod._cloud_devices(), [])


class GoveeLanPathTests(unittest.TestCase):
    """Drive set_state down the LAN branch by supplying a lan_ip, capturing the
    payloads handed to _send_lan_cmd (so nothing actually opens a socket)."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_govee")

    def test_turn_on_sends_turn_value_1(self):
        sender = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "_send_lan_cmd", sender):
            res = self.mod.set_state({"name": "Strip", "lan_ip": "10.0.0.7"},
                                     on=True)
        self.assertEqual(res["path"], "lan")
        self.assertTrue(res["ok"])
        ip, cmd = sender.call_args[0]
        self.assertEqual(ip, "10.0.0.7")
        self.assertEqual(cmd["cmd"], "turn")
        self.assertEqual(cmd["data"]["value"], 1)

    def test_brightness_clamped_and_sent(self):
        sender = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "_send_lan_cmd", sender):
            res = self.mod.set_state({"name": "Strip", "lan_ip": "10.0.0.7"},
                                     brightness=150)  # over-range
        self.assertEqual(res["applied"]["brightness"], 100)  # clamped
        # The last LAN cmd is the brightness command, value 100.
        _ip, cmd = sender.call_args[0]
        self.assertEqual(cmd["cmd"], "brightness")
        self.assertEqual(cmd["data"]["value"], 100)

    def test_color_sent_before_brightness(self):
        """On Govee, colorwc resets brightness, so set_state must send color
        FIRST and brightness LAST. Assert call ordering."""
        calls = []
        def _rec(ip, cmd):
            calls.append(cmd["cmd"])
            return {"ok": True}
        with mock.patch.object(self.mod, "_send_lan_cmd", side_effect=_rec):
            self.mod.set_state({"name": "Strip", "lan_ip": "10.0.0.7"},
                               color=(255, 0, 0), brightness=40)
        self.assertIn("colorwc", calls)
        self.assertIn("brightness", calls)
        self.assertLess(calls.index("colorwc"), calls.index("brightness"))

    def test_color_payload_carries_rgb(self):
        sender = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "_send_lan_cmd", sender):
            self.mod.set_state({"name": "Strip", "lan_ip": "10.0.0.7"},
                               color=(10, 20, 30))
        _ip, cmd = sender.call_args[0]
        self.assertEqual(cmd["cmd"], "colorwc")
        self.assertEqual(cmd["data"]["color"], {"r": 10, "g": 20, "b": 30})

    def test_lan_send_failure_propagates_error(self):
        with mock.patch.object(self.mod, "_send_lan_cmd",
                               return_value={"error": "lan send failed: boom"}):
            res = self.mod.set_state({"name": "Strip", "lan_ip": "10.0.0.7"},
                                     on=True)
        self.assertIn("lan send failed", res["error"])


class GoveeListDevicesTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_govee")

    def test_lan_device_listed_with_ip(self):
        lan = {"10.0.0.7": {"ip": "10.0.0.7", "sku": "H6159", "device": "Strip"}}
        with mock.patch.object(self.mod, "_refresh_lan", return_value=lan), \
             mock.patch.object(self.mod, "_cloud_devices", return_value=[]):
            devs = self.mod.list_devices()
        self.assertEqual(len(devs), 1)
        self.assertEqual(devs[0]["lan_ip"], "10.0.0.7")
        self.assertEqual(devs[0]["brand"], "Govee")


if __name__ == "__main__":
    unittest.main()
