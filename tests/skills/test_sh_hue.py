"""Logic tests for skills/sh_hue.py (Philips Hue controller).

The skill is a thin wrapper over the optional `phue` library. The highest-
value coverage is GRACEFUL DEGRADATION — every public path must return an
informative dict/string (never raise) when phue isn't installed or the bridge
isn't reachable. We also exercise the pure colour-maths helpers and the
set_state apply logic against a fake phue.Light.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class _FakeLight:
    """Minimal stand-in for a phue.Light — attribute writes are recorded."""
    def __init__(self, name="Office", ltype="Extended color light", colormode="xy"):
        self.name = name
        self.type = ltype
        self.colormode = colormode
        self.on = False
        self.brightness = 0
        self.reachable = True
        self.light_id = 7
        self.xy = None
        self.colortemp = None


class _FakeBridge:
    def __init__(self, lights):
        self._lights = lights  # dict[name -> _FakeLight]

    def get_light_objects(self, mode="name"):
        return self._lights


class HueDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_hue")

    def test_is_available_false_without_phue(self):
        with mock.patch.object(self.mod, "_phue", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_list_devices_empty_without_bridge(self):
        # _get_bridge returns None (no phue / no connect) → empty list, no raise.
        with mock.patch.object(self.mod, "_get_bridge", return_value=None):
            self.assertEqual(self.mod.list_devices(), [])

    def test_hue_list_action_informative_when_unreachable(self):
        with mock.patch.object(self.mod, "_get_bridge", return_value=None):
            out = self.actions["hue_list"]("")
        # Not awaiting a button press → the bridge-unreachable hint.
        self.assertIn("No Hue bulbs", out)
        self.assertIn("bridge", out.lower())

    def test_hue_list_action_button_hint_when_awaiting(self):
        self.mod._pending["awaiting_button_until"] = self.mod.time.time() + 30
        with mock.patch.object(self.mod, "_get_bridge", return_value=None):
            out = self.actions["hue_list"]("")
        self.assertIn("press the button", out.lower())

    def test_set_state_reports_bridge_not_connected(self):
        with mock.patch.object(self.mod, "_get_bridge", return_value=None):
            res = self.mod.set_state({"name": "Office"}, on=True)
        self.assertIn("not connected", res["error"])

    def test_get_state_reports_bulb_not_found(self):
        bridge = _FakeBridge({})  # bridge connected but no such bulb
        with mock.patch.object(self.mod, "_get_bridge", return_value=bridge):
            res = self.mod.get_state({"name": "Nonexistent"})
        self.assertIn("not found", res["error"])

    def test_retry_connect_surfaces_last_error(self):
        self.mod._pending["awaiting_button_until"] = 0.0
        self.mod._pending["last_error"] = "no bridge IP"
        with mock.patch.object(self.mod, "_get_bridge", return_value=None):
            out = self.actions["hue_retry_connect"]("")
        self.assertIn("no bridge IP", out)


class HueColorMathTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_hue")

    def test_rgb_to_xy_white_is_d65ish(self):
        x, y = self.mod._rgb_to_xy((255, 255, 255))
        # Pure white sRGB sits near the D65 white point (~0.31, 0.33).
        self.assertAlmostEqual(x, 0.3127, delta=0.02)
        self.assertAlmostEqual(y, 0.3290, delta=0.02)

    def test_rgb_to_xy_black_is_origin(self):
        self.assertEqual(self.mod._rgb_to_xy((0, 0, 0)), (0.0, 0.0))

    def test_rgb_to_xy_red_in_gamut(self):
        x, y = self.mod._rgb_to_xy((255, 0, 0))
        # Red chromaticity has large x, small y; both valid [0,1].
        self.assertGreater(x, 0.6)
        self.assertLess(y, 0.4)

    def test_kelvin_to_mired_clamped(self):
        # 6500K -> 153 mired (the bright/cool clamp), 2000K -> 500 (warm clamp).
        self.assertEqual(self.mod._kelvin_to_mired(10_000), 153)
        self.assertEqual(self.mod._kelvin_to_mired(1000), 500)
        self.assertEqual(self.mod._kelvin_to_mired(0), 366)  # default 2700K
        # 2857K ≈ 350 mired, within range.
        self.assertEqual(self.mod._kelvin_to_mired(2857), 350)


class HueSetStateApplyTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_hue")

    def _bridge_with(self, light):
        return _FakeBridge({light.name: light})

    def test_brightness_percent_maps_to_254_and_turns_on(self):
        light = _FakeLight()
        with mock.patch.object(self.mod, "_get_bridge",
                               return_value=self._bridge_with(light)):
            res = self.mod.set_state({"name": "Office"}, brightness=50)
        self.assertTrue(res["ok"])
        self.assertEqual(res["applied"]["brightness"], 50)
        self.assertEqual(light.brightness, 127)  # round(50/100*254)
        self.assertTrue(light.on)               # >0% forces power on

    def test_on_off_applied(self):
        light = _FakeLight()
        with mock.patch.object(self.mod, "_get_bridge",
                               return_value=self._bridge_with(light)):
            res = self.mod.set_state({"name": "Office"}, on=True)
        self.assertEqual(res["applied"]["on"], True)
        self.assertTrue(light.on)

    def test_color_temperature_skipped_for_color_only_bulb(self):
        # A 'Color light' lacks a Kelvin range → skill records a skip, no raise.
        light = _FakeLight(ltype="Color light", colormode="hs")
        with mock.patch.object(self.mod, "_get_bridge",
                               return_value=self._bridge_with(light)):
            res = self.mod.set_state({"name": "Office"}, color_temperature=2700)
        self.assertTrue(res["ok"])
        self.assertEqual(res["applied"]["color_temperature_skipped"],
                         "bulb_lacks_ct_range")
        self.assertIsNone(light.colortemp)

    def test_name_match_is_case_insensitive(self):
        light = _FakeLight(name="Living Room")
        bridge = self._bridge_with(light)
        found = self.mod._light_by_name(bridge, "living room")
        self.assertIs(found, light)


if __name__ == "__main__":
    unittest.main()
