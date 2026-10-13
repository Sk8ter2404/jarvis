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


class _FakeTuya:
    """A Tuya device that keeps its data points: set_value really changes
    them (unless `stuck`), status() reads them back, `dead` = unreachable."""

    def __init__(self, dps, stuck=False, dead=False, raise_on_set=False):
        self.dps = dict(dps)
        self.stuck, self.dead, self.raise_on_set = stuck, dead, raise_on_set
        self.sets = []

    def status(self):
        if self.dead:
            return {"Error": "Network Error: Device Unreachable", "Err": "905"}
        return {"dps": dict(self.dps)}

    def set_value(self, index, value):
        if self.raise_on_set:
            raise OSError("offline")
        self.sets.append((str(index), value))
        if not self.stuck:
            self.dps[str(index)] = value
        return {"dps": {str(index): value}}


BULB = {"20": False, "21": "white", "22": 1000, "23": 400}
PLUG = {"1": False, "9": 0}


class TuyaSetStateApplyTests(unittest.TestCase):
    """set_state drives the device's REAL data points (read first) and
    confirms by reading back (2026-10-13: the old OutletDevice.turn_on drove
    dp 1, which Tuya bulbs don't have, and still said OK)."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_tuya")

    def _run(self, dev, **kw):
        with mock.patch.object(self.mod, "_find_record", return_value=_CATALOG[0]),              mock.patch.object(self.mod, "_device", return_value=dev):
            return self.mod.set_state({"name": "Entry Plug"}, **kw)

    def test_bulb_turns_on_with_dp_20(self):
        dev = _FakeTuya(BULB)
        res = self._run(dev, on=True)
        self.assertTrue(res["ok"])
        self.assertEqual(dev.sets, [("20", True)])
        self.assertIs(dev.dps["20"], True)

    def test_plug_turns_off_with_dp_1(self):
        dev = _FakeTuya(dict(PLUG, **{"1": True}))
        res = self._run(dev, on=False)
        self.assertTrue(res["ok"])
        self.assertEqual(dev.sets, [("1", False)])

    def test_brightness_turns_the_bulb_on_and_sets_dp_22(self):
        dev = _FakeTuya(BULB)
        res = self._run(dev, brightness=60)
        self.assertEqual(res["applied"], {"on": True, "brightness": 60})
        self.assertEqual(dev.sets, [("20", True), ("22", 600)])

    def test_brightness_is_clamped_and_never_below_the_floor(self):
        dev = _FakeTuya(BULB)
        self.assertEqual(self._run(dev, brightness=250)["applied"]["brightness"], 100)
        self.assertEqual(dev.sets[-1], ("22", 1000))
        dev2 = _FakeTuya(BULB)
        self._run(dev2, brightness=0.4)
        self.assertEqual(dev2.sets, [("20", False)])     # 0 % means off

    def test_a_device_that_does_not_change_is_not_reported_ok(self):
        res = self._run(_FakeTuya(BULB, stuck=True), on=True)
        self.assertIn("did not confirm", res["error"])

    def test_unreachable_device_errors_so_the_router_falls_back(self):
        res = self._run(_FakeTuya(BULB, dead=True), on=True)
        self.assertIn("didn't answer", res["error"])

    def test_set_failure_returns_partial(self):
        res = self._run(_FakeTuya(BULB, raise_on_set=True), on=True)
        self.assertIn("tuya set_state failed", res["error"])
        self.assertIn("partial", res)

    def test_colour_only_is_left_to_alexa(self):
        res = self._run(_FakeTuya(BULB), color_name="red")
        self.assertIn("isn't supported locally", res["error"])

    def test_no_brightness_point_is_an_error_not_a_fake_ok(self):
        res = self._run(_FakeTuya(PLUG), brightness=50)
        self.assertIn("no brightness control", res["error"])

    def test_get_state_bulb(self):
        with mock.patch.object(self.mod, "_find_record", return_value=_CATALOG[0]),              mock.patch.object(self.mod, "_device",
                               return_value=_FakeTuya({"20": True, "22": 500})):
            st = self.mod.get_state({"name": "Entry Plug"})
        self.assertTrue(st["on"])
        self.assertEqual(st["brightness"], 50)

    def test_get_state_plug_and_read_failure(self):
        with mock.patch.object(self.mod, "_find_record", return_value=_CATALOG[0]),              mock.patch.object(self.mod, "_device", return_value=_FakeTuya({"1": True})):
            self.assertTrue(self.mod.get_state({"name": "x"})["on"])
        with mock.patch.object(self.mod, "_find_record", return_value=_CATALOG[0]),              mock.patch.object(self.mod, "_device", return_value=_FakeTuya(BULB, dead=True)):
            self.assertIn("didn't answer", self.mod.get_state({"name": "x"})["error"])


class TuyaTtImportTests(unittest.TestCase):
    """_tt(): cache + import-success and import-failure branches."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_tuya")

    def test_tt_imports_and_caches(self):
        import sys
        sentinel = mock.MagicMock(name="fake-tinytuya")
        self.mod._tinytuya = None  # force a fresh import attempt
        with mock.patch.dict(sys.modules, {"tinytuya": sentinel}):
            got = self.mod._tt()
        self.assertIs(got, sentinel)
        # Second call returns the cached object without re-importing.
        self.assertIs(self.mod._tt(), sentinel)

    def test_tt_import_failure_caches_false(self):
        import sys
        self.mod._tinytuya = None
        # Block the import → the except sets the cache to False → returns None.
        with mock.patch.dict(sys.modules, {"tinytuya": None}):
            self.assertIsNone(self.mod._tt())
        self.assertIs(self.mod._tinytuya, False)
        # And a subsequent call still returns None (cached False).
        self.assertIsNone(self.mod._tt())


class TuyaDeviceHandleTests(unittest.TestCase):
    """_device(): OutletDevice construction, version parsing, failure."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_tuya")

    def _fake_tt(self):
        tt = mock.MagicMock(name="tinytuya")
        return tt

    def test_builds_handle_with_parsed_version(self):
        tt = self._fake_tt()
        with mock.patch.object(self.mod, "_tt", return_value=tt):
            handle = self.mod._device(_CATALOG[0])  # version "3.3"
        self.assertIs(handle, tt.OutletDevice.return_value)
        _args, kwargs = tt.OutletDevice.call_args
        self.assertEqual(kwargs["version"], 3.3)
        self.assertEqual(kwargs["dev_id"], "dev1")
        handle.set_socketTimeout.assert_called_once_with(3)

    def test_bad_version_defaults_to_33(self):
        tt = self._fake_tt()
        rec = {"id": "x", "ip": "1.2.3.4", "key": "K", "version": "garbage"}
        with mock.patch.object(self.mod, "_tt", return_value=tt):
            self.mod._device(rec)
        _args, kwargs = tt.OutletDevice.call_args
        self.assertEqual(kwargs["version"], 3.3)

    def test_construction_failure_returns_none(self):
        tt = self._fake_tt()
        tt.OutletDevice.side_effect = RuntimeError("bad address")
        with mock.patch.object(self.mod, "_tt", return_value=tt):
            self.assertIsNone(self.mod._device(_CATALOG[0]))


class TuyaLoadCatalogTests(unittest.TestCase):
    """_load_catalog(): real file parse + malformed/missing degradation."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("sh_tuya")

    def test_loads_devices_from_file(self):
        import json
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "tuya_devices.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"devices": [{"name": "A"}, "not-a-dict", {"name": "B"}]}, f)
            with mock.patch.object(self.mod, "_CATALOG", p):
                devs = self.mod._load_catalog()
        # Non-dict entries are filtered out.
        self.assertEqual([d["name"] for d in devs], ["A", "B"])

    def test_missing_file_returns_empty(self):
        import os
        import tempfile
        missing = os.path.join(tempfile.gettempdir(), "no_such_tuya_catalog.json")
        with mock.patch.object(self.mod, "_CATALOG", missing):
            self.assertEqual(self.mod._load_catalog(), [])

    def test_malformed_json_returns_empty(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "tuya_devices.json")
            with open(p, "w", encoding="utf-8") as f:
                f.write("{not valid json")
            with mock.patch.object(self.mod, "_CATALOG", p):
                self.assertEqual(self.mod._load_catalog(), [])


if __name__ == "__main__":
    unittest.main()
