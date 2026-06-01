"""Logic tests for skills/sh_kasa.py (TP-Link Kasa/Tapo controller).

Thin wrapper over the optional `python-kasa` library plus a freeform voice
control entry point (`smart_home_control`). Coverage:
  * graceful degradation when python-kasa absent / nothing discovered,
  * the pure _rgb_to_hsv helper,
  * smart_home_control intent parsing + device matching, fully mocked so no
    LAN broadcast happens.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class KasaDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_kasa")

    def test_is_available_false_without_kasa(self):
        with mock.patch.object(self.mod, "_kasa", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_kasa_list_informative_when_no_devices(self):
        with mock.patch.object(self.mod, "list_devices", return_value=[]):
            out = self.actions["kasa_list"]("")
        self.assertIn("No Kasa", out)
        self.assertIn("9999", out)  # mentions the UDP discovery port hint

    def test_get_state_device_not_found(self):
        with mock.patch.object(self.mod, "_device_for", return_value=None):
            res = self.mod.get_state({"name": "Lamp"})
        self.assertIn("not found", res["error"])

    def test_set_state_device_not_found(self):
        with mock.patch.object(self.mod, "_device_for", return_value=None):
            res = self.mod.set_state({"name": "Lamp"}, on=True)
        self.assertIn("not found", res["error"])

    def test_list_devices_empty_on_empty_discovery(self):
        with mock.patch.object(self.mod, "_refresh_discovery", return_value={}):
            self.assertEqual(self.mod.list_devices(), [])


class KasaRgbToHsvTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_kasa")

    def test_pure_red(self):
        h, s, v = self.mod._rgb_to_hsv((255, 0, 0))
        self.assertEqual(h, 0)
        self.assertEqual(s, 100)
        self.assertEqual(v, 100)

    def test_pure_green_hue_120(self):
        h, s, v = self.mod._rgb_to_hsv((0, 255, 0))
        self.assertEqual(h, 120)
        self.assertEqual(s, 100)

    def test_pure_blue_hue_240(self):
        h, _s, _v = self.mod._rgb_to_hsv((0, 0, 255))
        self.assertEqual(h, 240)

    def test_black_is_zero_saturation_value(self):
        h, s, v = self.mod._rgb_to_hsv((0, 0, 0))
        self.assertEqual((h, s, v), (0, 0, 0))


class KasaSmartHomeControlTests(unittest.TestCase):
    """smart_home_control parses intent + matches a device by name, then routes
    to set_state/get_state. We stub list_devices + set/get_state so nothing
    touches the LAN, and assert on the spoken result + the kwargs dispatched."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_kasa")
        self._devs = [
            {"name": "Entry Light", "lan_ip": "10.0.0.5"},
            {"name": "Dining Room", "lan_ip": "10.0.0.6"},
        ]

    def test_empty_request_prompts(self):
        out = self.actions["smart_home_control"]("")
        self.assertIn("control", out.lower())

    def test_no_devices_message(self):
        with mock.patch.object(self.mod, "list_devices", return_value=[]), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None):
            out = self.actions["smart_home_control"]("turn on entry light")
        self.assertIn("don't see any", out.lower())

    def test_turn_on_named_device_dispatches_on_true(self):
        set_state = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "list_devices", return_value=self._devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None), \
             mock.patch.object(self.mod, "set_state", set_state):
            out = self.actions["smart_home_control"]("turn on the entry light")
        # Matched 'Entry Light', dispatched on=True.
        self.assertIn("entry light on", out.lower())
        self.assertIn("done", out.lower())
        _args, kwargs = set_state.call_args
        self.assertEqual(kwargs.get("on"), True)

    def test_off_uses_word_boundary_not_substring(self):
        # 'office' must NOT be read as 'off'. We add an 'Office' device and ask
        # to turn it ON; intent must resolve to 'on', not 'off'.
        devs = [{"name": "Office", "lan_ip": "10.0.0.9"}]
        set_state = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "list_devices", return_value=devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None), \
             mock.patch.object(self.mod, "set_state", set_state):
            self.actions["smart_home_control"]("turn on the office")
        _args, kwargs = set_state.call_args
        self.assertEqual(kwargs.get("on"), True)

    def test_toggle_reads_state_then_flips(self):
        get_state = mock.Mock(return_value={"on": False})
        set_state = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "list_devices", return_value=self._devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None), \
             mock.patch.object(self.mod, "get_state", get_state), \
             mock.patch.object(self.mod, "set_state", set_state):
            out = self.actions["smart_home_control"]("toggle dining room")
        # Was off → toggled on.
        _args, kwargs = set_state.call_args
        self.assertEqual(kwargs.get("on"), True)
        self.assertIn("dining room on", out.lower())

    def test_status_query_reports_on_off(self):
        # A phrase with NO on/off/toggle/enable word → intent is None → status.
        # 'dining' avoids the 'on'/'off' word-boundary triggers entirely.
        get_state = mock.Mock(return_value={"on": True})
        set_state = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "list_devices", return_value=self._devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None), \
             mock.patch.object(self.mod, "get_state", get_state), \
             mock.patch.object(self.mod, "set_state", set_state):
            out = self.actions["smart_home_control"]("status of the dining room")
        self.assertIn("status", out.lower())
        self.assertIn("dining room is on", out.lower())
        set_state.assert_not_called()  # a status query must not write state

    def test_all_keyword_targets_every_device(self):
        set_state = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "list_devices", return_value=self._devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None), \
             mock.patch.object(self.mod, "set_state", set_state):
            out = self.actions["smart_home_control"]("turn off all the lights")
        # Both devices addressed.
        self.assertEqual(set_state.call_count, 2)
        self.assertIn("entry light off", out.lower())
        self.assertIn("dining room off", out.lower())


if __name__ == "__main__":
    unittest.main()
