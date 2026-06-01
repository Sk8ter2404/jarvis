"""Logic tests for skills/sh_tuya.py (Tuya / Smart-Life LAN controller).

Wraps the optional `tinytuya` library; devices are catalogued in
data/tuya_devices.json and only those with a per-device local key are
controllable. Coverage:
  * graceful degradation when tinytuya absent / a record has no key,
  * the ready/all/pending split (_ready_devices vs _all_devices),
  * list_devices shaping (only keyed devices, router-compatible record),
  * _find_record matching by carried record / ip / name,
  * set_state on/off + the tuya_list summary (ready + pending wording).
The catalog loader and tinytuya are mocked — no disk read, no LAN I/O.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


_CATALOG = [
    {"name": "Entry Plug", "id": "dev1", "ip": "10.0.0.3", "key": "KEY1",
     "version": "3.3", "product": "plug"},
    {"name": "Lamp", "id": "dev2", "ip": "10.0.0.4", "key": "KEY2"},
    {"name": "Pending Plug", "id": "dev3", "ip": "10.0.0.5", "key": ""},  # no key
]


class TuyaCatalogSplitTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_tuya")

    def test_ready_devices_only_keyed(self):
        with mock.patch.object(self.mod, "_load_catalog", return_value=list(_CATALOG)):
            ready = self.mod._ready_devices()
        names = {d["name"] for d in ready}
        self.assertEqual(names, {"Entry Plug", "Lamp"})  # Pending excluded

    def test_all_devices_includes_pending(self):
        with mock.patch.object(self.mod, "_load_catalog", return_value=list(_CATALOG)):
            self.assertEqual(len(self.mod._all_devices()), 3)

    def test_list_devices_shapes_records(self):
        with mock.patch.object(self.mod, "_load_catalog", return_value=list(_CATALOG)):
            devs = self.mod.list_devices()
        self.assertEqual(len(devs), 2)  # only keyed
        d = devs[0]
        self.assertEqual(d["brand"], "Tuya")
        self.assertEqual(d["capabilities"], ["on_off"])
        self.assertEqual(d["lan_ip"], "10.0.0.3")
        self.assertIn("_tuya", d)  # carries full record for control


class TuyaListMessagingTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_tuya")

    def test_empty_catalog_message(self):
        with mock.patch.object(self.mod, "_load_catalog", return_value=[]):
            out = self.actions["tuya_list"]("")
        self.assertIn("No Tuya devices catalogued", out)

    def test_lists_ready_and_pending_counts(self):
        with mock.patch.object(self.mod, "_load_catalog", return_value=list(_CATALOG)):
            out = self.actions["tuya_list"]("")
        self.assertIn("2 Tuya device(s) ready", out)
        self.assertIn("Entry Plug", out)
        self.assertIn("1 discovered but awaiting", out)  # the keyless one


class TuyaDegradationTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_tuya")

    def test_device_handle_none_without_lib(self):
        with mock.patch.object(self.mod, "_tt", return_value=None):
            self.assertIsNone(self.mod._device(_CATALOG[0]))

    def test_device_handle_none_without_key(self):
        # Even with the lib, a keyless record yields no handle.
        with mock.patch.object(self.mod, "_tt", return_value=object()):
            self.assertIsNone(self.mod._device({"id": "x", "ip": "y", "key": ""}))

    def test_get_state_not_ready_when_no_handle(self):
        with mock.patch.object(self.mod, "_find_record", return_value=None):
            res = self.mod.get_state({"name": "Lamp"})
        self.assertIn("not ready", res["error"])

    def test_set_state_not_ready_when_no_handle(self):
        with mock.patch.object(self.mod, "_find_record", return_value=None):
            res = self.mod.set_state({"name": "Lamp"}, on=True)
        self.assertIn("not ready", res["error"])


class TuyaFindRecordTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_tuya")

    def test_carried_record_wins(self):
        rec = {"id": "carried"}
        self.assertEqual(self.mod._find_record({"_tuya": rec}), rec)

    def test_match_by_ip(self):
        with mock.patch.object(self.mod, "_load_catalog", return_value=list(_CATALOG)):
            found = self.mod._find_record({"lan_ip": "10.0.0.4"})
        self.assertEqual(found["name"], "Lamp")

    def test_match_by_name_case_insensitive(self):
        with mock.patch.object(self.mod, "_load_catalog", return_value=list(_CATALOG)):
            found = self.mod._find_record({"name": "entry plug"})
        self.assertEqual(found["id"], "dev1")

    def test_no_match_returns_none(self):
        with mock.patch.object(self.mod, "_load_catalog", return_value=list(_CATALOG)):
            self.assertIsNone(self.mod._find_record({"name": "nope"}))


class TuyaSetStateApplyTests(unittest.TestCase):
    """set_state drives a tinytuya OutletDevice. We stub _device so a fake
    handle records turn_on/turn_off, avoiding any real LAN socket."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_tuya")

    def test_turn_on(self):
        handle = mock.Mock()
        with mock.patch.object(self.mod, "_find_record", return_value=_CATALOG[0]), \
             mock.patch.object(self.mod, "_device", return_value=handle):
            res = self.mod.set_state({"name": "Entry Plug"}, on=True)
        self.assertEqual(res["applied"]["on"], True)
        handle.turn_on.assert_called_once()

    def test_turn_off(self):
        handle = mock.Mock()
        with mock.patch.object(self.mod, "_find_record", return_value=_CATALOG[0]), \
             mock.patch.object(self.mod, "_device", return_value=handle):
            res = self.mod.set_state({"name": "Entry Plug"}, on=False)
        self.assertEqual(res["applied"]["on"], False)
        handle.turn_off.assert_called_once()

    def test_get_state_reads_dps_switch(self):
        handle = mock.Mock()
        handle.status.return_value = {"dps": {"1": True}}
        with mock.patch.object(self.mod, "_find_record", return_value=_CATALOG[0]), \
             mock.patch.object(self.mod, "_device", return_value=handle):
            st = self.mod.get_state({"name": "Entry Plug"})
        self.assertTrue(st["on"])
        self.assertEqual(st["raw"], {"1": True})


if __name__ == "__main__":
    unittest.main()
