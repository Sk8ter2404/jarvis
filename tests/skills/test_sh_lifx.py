"""Logic tests for skills/sh_lifx.py (LIFX LAN controller).

Thin wrapper over the optional `lifxlan` library. Coverage:
  * graceful degradation when lifxlan is absent / nothing discovered,
  * the pure _rgb_to_hsbk colour conversion (0..65535 channel scaling),
  * list_devices / get_state / set_state against a fake bulb, with the
    network-touching _refresh stubbed so no UDP broadcast happens.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _FakeBulb:
    def __init__(self, label="Kitchen", mac="d0:73:d5:00:00:01", power=0,
                 color=(0, 0, 32768, 3500), supports_color=True):
        self._label = label
        self._mac = mac
        self._power = power
        self._color = color
        self._supports = supports_color
        self.set_color_calls = []
        self.set_power_calls = []

    def get_label(self): return self._label
    def get_mac_addr(self): return self._mac
    def get_power(self): return self._power
    def get_color(self): return self._color
    def supports_color(self): return self._supports

    def set_power(self, v): self.set_power_calls.append(v)
    def set_color(self, hsbk): self.set_color_calls.append(hsbk)


class LifxDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_lifx")

    def test_is_available_false_without_lifxlan(self):
        with mock.patch.object(self.mod, "_lifxlan", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_refresh_returns_empty_without_lib(self):
        with mock.patch.object(self.mod, "_lifxlan", return_value=None):
            self.assertEqual(self.mod._refresh(), {})

    def test_list_devices_empty_when_none_discovered(self):
        with mock.patch.object(self.mod, "_refresh", return_value={}):
            self.assertEqual(self.mod.list_devices(), [])

    def test_lifx_list_informative_when_empty(self):
        with mock.patch.object(self.mod, "list_devices", return_value=[]):
            out = self.actions["lifx_list"]("")
        self.assertIn("No LIFX bulbs", out)
        self.assertIn("56700", out)  # mentions the UDP port hint

    def test_get_state_bulb_not_found(self):
        with mock.patch.object(self.mod, "_bulb_for", return_value=None):
            res = self.mod.get_state({"name": "Ghost"})
        self.assertIn("not found", res["error"])

    def test_set_state_bulb_not_found(self):
        with mock.patch.object(self.mod, "_bulb_for", return_value=None):
            res = self.mod.set_state({"name": "Ghost"}, on=True)
        self.assertIn("not found", res["error"])


class LifxRgbToHsbkTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_lifx")

    def test_red_full_saturation_and_brightness(self):
        h, s, b, k = self.mod._rgb_to_hsbk((255, 0, 0), kelvin=3500)
        self.assertEqual(h, 0)          # red hue 0°
        self.assertEqual(s, 65535)      # fully saturated
        self.assertEqual(b, 65535)      # full brightness
        self.assertEqual(k, 3500)       # kelvin passed through

    def test_green_hue_is_third_of_scale(self):
        h, _s, _b, _k = self.mod._rgb_to_hsbk((0, 255, 0))
        # 120° / 360° * 65535 ≈ 21845.
        self.assertAlmostEqual(h, 21845, delta=2)

    def test_white_zero_saturation(self):
        h, s, b, _k = self.mod._rgb_to_hsbk((255, 255, 255))
        self.assertEqual(s, 0)
        self.assertEqual(b, 65535)


class LifxListAndStateTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_lifx")

    def test_list_devices_shapes_record(self):
        bulb = _FakeBulb(label="Kitchen")
        with mock.patch.object(self.mod, "_refresh",
                               return_value={"kitchen": bulb}):
            devs = self.mod.list_devices()
        self.assertEqual(len(devs), 1)
        d = devs[0]
        self.assertEqual(d["name"], "Kitchen")
        self.assertEqual(d["brand"], "LIFX")
        self.assertIn("color", d["capabilities"])  # supports_color True
        self.assertEqual(d["lan_mac"], "d0:73:d5:00:00:01")

    def test_get_state_translates_power_and_brightness(self):
        # power on, brightness mid (32768/65535 ≈ 50%), kelvin 3500.
        bulb = _FakeBulb(power=65535, color=(0, 0, 32768, 3500))
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            st = self.mod.get_state({"name": "Kitchen"})
        self.assertTrue(st["on"])
        self.assertEqual(st["brightness"], 50)
        self.assertEqual(st["color_temperature_k"], 3500)

    def test_set_power_off(self):
        bulb = _FakeBulb()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, on=False)
        self.assertEqual(res["applied"]["on"], False)
        self.assertEqual(bulb.set_power_calls, ["off"])

    def test_set_brightness_scales_and_powers_on(self):
        bulb = _FakeBulb()
        with mock.patch.object(self.mod, "_bulb_for", return_value=bulb):
            res = self.mod.set_state({"name": "Kitchen"}, brightness=50)
        self.assertEqual(res["applied"]["brightness"], 50)
        # set_color called with brightness ≈ 32767 (50% of 65535).
        self.assertTrue(bulb.set_color_calls)
        _h, _s, bri, _k = bulb.set_color_calls[-1]
        self.assertAlmostEqual(bri, 32767, delta=2)
        # >0% brightness also powers the bulb on.
        self.assertIn("on", bulb.set_power_calls)


if __name__ == "__main__":
    unittest.main()
