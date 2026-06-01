"""Logic tests for skills/smart_home_discover.py (Alexa discovery wizard).

The wizard's sign-in path is interactive and network-bound, so we don't drive
it. Instead we cover the rich PURE helpers it exposes (which the catalog build
depends on) plus the safe, non-interactive actions:
  * graceful degradation when alexapy is absent,
  * brand → controller-skill mapping,
  * Alexa capability-namespace → short-tag flattening (dict/list/str shapes),
  * coarse device-type classification,
  * ARP-table brand cross-reference + brand normalisation,
  * cookie staleness maths,
  * smart_home_catalog / smart_home_purge_cookie actions.
No Amazon sign-in, no Playwright, no disk writes (all I/O mocked).
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


class DiscoverDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("smart_home_discover")

    def test_is_available_false_without_alexapy(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_discover_action_offline_without_alexapy(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=None):
            out = self.actions["smart_home_discover"]("")
        self.assertIn("offline", out)
        self.assertIn("alexapy", out)

    def test_catalog_action_when_no_catalog(self):
        with mock.patch.object(self.mod, "_load_catalog", return_value=None):
            out = self.actions["smart_home_catalog"]("")
        self.assertIn("No smart-home catalog", out)


class DiscoverBrandMappingTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_controller_skill_substring_match(self):
        cs = self.mod._controller_skill
        self.assertEqual(cs("Philips Hue"), "sh_hue")
        self.assertEqual(cs("Signify Netherlands B.V."), "sh_hue")
        self.assertEqual(cs("tp-link Tapo"), "sh_kasa")
        self.assertEqual(cs("LIFX"), "sh_lifx")
        self.assertEqual(cs("Govee"), "sh_govee")
        self.assertEqual(cs("Google Nest"), "sh_nest")

    def test_controller_skill_unknown_brand(self):
        self.assertIsNone(self.mod._controller_skill("Wyze"))
        self.assertIsNone(self.mod._controller_skill(""))

    def test_normalise_brand_collapses_whitespace(self):
        self.assertEqual(self.mod._normalise_brand("  Philips   Hue  "),
                         "Philips Hue")
        self.assertEqual(self.mod._normalise_brand(None), "")


class DiscoverCapabilityTagTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_list_of_interface_dicts(self):
        raw = [{"interface": "Alexa.PowerController"},
               {"interface": "Alexa.BrightnessController"}]
        self.assertEqual(self.mod._capability_tags(raw), ["dim", "on_off"])

    def test_unknown_alexa_namespace_passed_through(self):
        raw = [{"interface": "Alexa.WeirdNewController"}]
        self.assertEqual(self.mod._capability_tags(raw), ["weirdnewcontroller"])

    def test_string_shape(self):
        self.assertEqual(self.mod._capability_tags("Alexa.LockController"),
                         ["lock"])

    def test_nested_dict_shape_flattens(self):
        raw = {"x": [{"interface": "Alexa.ThermostatController"}],
               "y": "Alexa.TemperatureSensor"}
        self.assertEqual(self.mod._capability_tags(raw),
                         ["temperature", "thermostat"])

    def test_empty_caps(self):
        self.assertEqual(self.mod._capability_tags(None), [])


class DiscoverEntityTypeTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_lock_wins(self):
        self.assertEqual(self.mod._entity_type(["lock", "on_off"], "Yale", {}),
                         "lock")

    def test_color_capable_is_light(self):
        self.assertEqual(self.mod._entity_type(["color", "on_off"], "Hue", {}),
                         "light")

    def test_on_off_known_light_brand_is_light(self):
        self.assertEqual(self.mod._entity_type(["on_off"], "LIFX", {}), "light")

    def test_on_off_unknown_brand_is_plug(self):
        self.assertEqual(self.mod._entity_type(["on_off"], "Generic", {}), "plug")

    def test_falls_back_to_display_category(self):
        ent = {"displayCategories": ["SWITCH"]}
        self.assertEqual(self.mod._entity_type([], "", ent), "switch")

    def test_thermostat_and_camera_and_scene(self):
        self.assertEqual(self.mod._entity_type(["thermostat"], "", {}), "thermostat")
        self.assertEqual(self.mod._entity_type(["camera"], "", {}), "camera")
        self.assertEqual(self.mod._entity_type(["scene"], "", {}), "scene")


class DiscoverArpAndCookieTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_match_arp_entry_by_brand_hint(self):
        arp = [{"ip": "10.0.0.2", "mac": "00:17:88:11:22:33",
                "brand_oui_hint": "Philips Hue"}]
        got = self.mod._match_arp_entry("Philips Hue", arp)
        self.assertEqual(got, ("10.0.0.2", "00:17:88:11:22:33"))

    def test_match_arp_entry_no_hit(self):
        arp = [{"ip": "10.0.0.2", "mac": "aa:bb:cc:dd:ee:ff",
                "brand_oui_hint": None}]
        self.assertIsNone(self.mod._match_arp_entry("LIFX", arp))
        self.assertIsNone(self.mod._match_arp_entry("", arp))

    def test_cookie_is_stale_when_old(self):
        import time
        old = {"saved_at": time.time() - 400 * 86400}  # 400 days old
        self.assertTrue(self.mod._cookie_is_stale(old))

    def test_cookie_fresh_when_recent(self):
        import time
        fresh = {"saved_at": time.time() - 5 * 86400}  # 5 days old
        self.assertFalse(self.mod._cookie_is_stale(fresh))

    def test_cookie_stale_when_missing_timestamp(self):
        self.assertTrue(self.mod._cookie_is_stale({}))


class DiscoverCatalogActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("smart_home_discover")

    def test_catalog_summary_groups_by_room(self):
        cat = {
            "device_count": 3,
            "devices": [
                {"name": "L1", "alexa_room": "Kitchen"},
                {"name": "L2", "alexa_room": "Kitchen"},
                {"name": "T1", "alexa_room": "Hall"},
            ],
        }
        with mock.patch.object(self.mod, "_load_catalog", return_value=cat):
            out = self.actions["smart_home_catalog"]("")
        self.assertIn("3 smart-home devices", out)
        self.assertIn("2 in Kitchen", out)
        self.assertIn("1 in Hall", out)

    def test_purge_cookie_reports_removed_count(self):
        # Pretend both cookie files exist and unlink succeeds.
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod.os, "unlink") as unlink:
            out = self.actions["smart_home_purge_cookie"]("")
        self.assertEqual(unlink.call_count, 2)
        self.assertIn("cleared", out)

    def test_purge_cookie_when_nothing_cached(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            out = self.actions["smart_home_purge_cookie"]("")
        self.assertIn("No cached Alexa cookie", out)


if __name__ == "__main__":
    unittest.main()
