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

The second half of this file (everything below the ORIGINAL block) drives the
deeper, previously-uncovered machinery — ARP parsing, cookie persistence to a
throwaway temp dir, catalog assembly/merge, the async login/fetch/playwright
coroutines (driven for real via asyncio.run against fakes), the todo queue, and
both wizard entry points — with EVERY network/socket/subprocess/sleep/thread
mocked so the suite is deterministic and fully offline.
"""
from __future__ import annotations

import asyncio
import os
import pickle
import shutil
import sys
import tempfile
import types
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


# ════════════════════════════════════════════════════════════════════════════
# EXTENDED COVERAGE — everything below targets the deep machinery that the
# original block above left at ~20%. Helpers:
#
#   _FakeAlexapy / _FakeLogin   — duck-typed stand-ins so we never touch the
#                                 real alexapy install or the network.
#   _tmpdir_paths()             — redirect every module path constant at a fresh
#                                 throwaway temp dir so real file/pickle writes
#                                 stay isolated + gitignored, and the realpath
#                                 path-safety checks pass for real.
#   run_coro()                  — drive a coroutine with asyncio.run (there is
#                                 never a running loop inside a unittest call,
#                                 so this mirrors the module's own _run_async).
# ════════════════════════════════════════════════════════════════════════════


def run_coro(coro):
    return asyncio.run(coro)


class _FakeLogin:
    """Duck-typed AlexaLogin. ``login()`` walks a scripted list of status
    dicts (one per call), recording the kwargs it was handed so tests can
    assert the wizard fed back the right captcha/otp answer."""

    def __init__(self, statuses=None, email="", session=None, raise_on_login=None):
        self.email = email
        self._statuses = list(statuses or [])
        self.status = {}
        self.calls = []
        self._session = session
        self.reset_called = False
        self._raise_on_login = raise_on_login
        # attributes the restore path may setattr / read
        self._cookies = None
        self.cookies = None

    async def login(self, data=None, cookies=None):
        self.calls.append({"data": data, "cookies": cookies})
        if self._raise_on_login is not None:
            raise self._raise_on_login
        if self._statuses:
            self.status = self._statuses.pop(0)
        return None

    async def reset(self):
        self.reset_called = True


class _FakeAlexaAPI:
    """Class-method API surface alexapy exposes; each returns a canned list."""
    _devices = []
    _smarthome = []
    _groups = []

    @classmethod
    async def get_devices(cls, login):
        return list(cls._devices)

    @classmethod
    async def get_smarthome_devices(cls, login):
        return list(cls._smarthome)

    @classmethod
    async def get_smarthome_groups(cls, login):
        return list(cls._groups)


def _make_fake_alexapy(login_ctor=None, api=None):
    """Build a fake `alexapy` module object with AlexaLogin + AlexaAPI."""
    fake = types.ModuleType("alexapy")
    fake.AlexaLogin = login_ctor if login_ctor is not None else _FakeLogin
    fake.AlexaAPI = api if api is not None else _FakeAlexaAPI
    return fake


class _TmpPaths:
    """Context manager: point every module path constant at a fresh temp dir
    and create the data dir so writes succeed. Cleans up on exit."""

    def __init__(self, mod):
        self.mod = mod
        self.root = None
        self._saved = {}

    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="shdisc_test_")
        data_dir = os.path.join(self.root, "data")
        os.makedirs(data_dir, exist_ok=True)
        self.paths = {
            "_PROJECT_DIR": self.root,
            "_DATA_DIR": data_dir,
            "_COOKIE_JSON_PATH": os.path.join(data_dir, "alexa_cookie.json"),
            "_COOKIE_PICKLE_PATH": os.path.join(data_dir, "alexa_cookie.pickle"),
            "_CATALOG_PATH": os.path.join(data_dir, "smart_home_devices.json"),
            "_TODO_PATH": os.path.join(self.root, "jarvis_todo.md"),
        }
        for k, v in self.paths.items():
            self._saved[k] = getattr(self.mod, k)
            setattr(self.mod, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            setattr(self.mod, k, v)
        if self.root and os.path.isdir(self.root):
            shutil.rmtree(self.root, ignore_errors=True)
        return False


# ── _scan_lan_arp / ARP parsing ─────────────────────────────────────────────
class ScanLanArpTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def _run_with_arp_output(self, raw_bytes):
        with mock.patch.object(self.mod.subprocess, "check_output",
                               return_value=raw_bytes) as co:
            rows = self.mod._scan_lan_arp()
        return rows, co

    def test_parses_windows_arp_table_and_labels_oui(self):
        raw = (
            "\r\nInterface: 192.168.1.5 --- 0x4\r\n"
            "  Internet Address      Physical Address      Type\r\n"
            "  192.168.1.1           aa-bb-cc-dd-ee-ff     dynamic\r\n"
            "  192.168.1.20          00-17-88-11-22-33     dynamic\r\n"   # Philips Hue OUI
            "  192.168.1.30          d0-73-d5-aa-bb-cc     dynamic\r\n"   # LIFX OUI
        ).encode("utf-8")
        rows, _ = self._run_with_arp_output(raw)
        ips = {r["ip"] for r in rows}
        self.assertEqual(ips, {"192.168.1.1", "192.168.1.20", "192.168.1.30"})
        by_ip = {r["ip"]: r for r in rows}
        self.assertEqual(by_ip["192.168.1.20"]["mac"], "00:17:88:11:22:33")
        self.assertEqual(by_ip["192.168.1.20"]["oui"], "001788")
        self.assertEqual(by_ip["192.168.1.20"]["brand_oui_hint"], "Philips Hue")
        self.assertEqual(by_ip["192.168.1.30"]["brand_oui_hint"], "LIFX")
        # Unknown OUI → no hint.
        self.assertIsNone(by_ip["192.168.1.1"]["brand_oui_hint"])

    def test_colon_separated_macs_also_parse(self):
        # The regex matches an IP immediately followed (whitespace only) by a
        # MAC. Colon separators are accepted just like the hyphen form.
        raw = (b"Interface: 10.0.0.1\r\n"
               b"  10.0.0.9      a4:c1:38:00:11:22   dynamic\r\n")  # Govee OUI
        rows, _ = self._run_with_arp_output(raw)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ip"], "10.0.0.9")
        self.assertEqual(rows[0]["mac"], "A4:C1:38:00:11:22")
        self.assertEqual(rows[0]["brand_oui_hint"], "Govee")

    def test_duplicate_ip_deduped_keeps_first(self):
        raw = (
            "  192.168.1.50          00-17-88-00-00-01     dynamic\r\n"
            "  192.168.1.50          00-17-88-00-00-02     dynamic\r\n"
        ).encode("utf-8")
        rows, _ = self._run_with_arp_output(raw)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["mac"].endswith("00:01"))

    def test_malformed_lines_ignored(self):
        raw = b"garbage line\nno mac here 1.2.3.4\nInterface: x\n"
        rows, _ = self._run_with_arp_output(raw)
        self.assertEqual(rows, [])

    def test_cp1252_fallback_decode_path(self):
        # No 'Interface:'/'Physical' marker after utf-8 decode → triggers the
        # cp1252 re-decode branch. Bytes still hold a valid ARP row.
        raw = b"  192.168.1.7           50-c7-bf-00-00-00     dynamic\r\n"
        rows, _ = self._run_with_arp_output(raw)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["brand_oui_hint"], "TP-Link")

    def test_subprocess_failure_returns_empty(self):
        with mock.patch.object(self.mod.subprocess, "check_output",
                               side_effect=OSError("arp missing")):
            self.assertEqual(self.mod._scan_lan_arp(), [])

    def test_subprocess_timeout_returns_empty(self):
        err = self.mod.subprocess.TimeoutExpired(cmd="arp", timeout=10)
        with mock.patch.object(self.mod.subprocess, "check_output",
                               side_effect=err):
            self.assertEqual(self.mod._scan_lan_arp(), [])

    def test_win32_creationflags_passed(self):
        raw = b"  192.168.1.1   aa-bb-cc-dd-ee-ff   dynamic\r\n"
        with mock.patch.object(self.mod.sys, "platform", "win32"), \
             mock.patch.object(self.mod.subprocess, "check_output",
                               return_value=raw) as co:
            self.mod._scan_lan_arp()
        # creationflags kwarg should be present and non-zero on win32.
        self.assertIn("creationflags", co.call_args.kwargs)


# ── _atomic_write_json / _save_catalog / _load_catalog ──────────────────────
class CatalogIoTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_save_then_load_roundtrip(self):
        with _TmpPaths(self.mod):
            cat = {"version": 1, "device_count": 1, "devices": [{"name": "X"}]}
            self.mod._save_catalog(cat)
            self.assertTrue(os.path.exists(self.mod._CATALOG_PATH))
            got = self.mod._load_catalog()
        self.assertEqual(got, cat)

    def test_load_catalog_missing_returns_none(self):
        with _TmpPaths(self.mod):
            self.assertIsNone(self.mod._load_catalog())

    def test_load_catalog_corrupt_returns_none(self):
        with _TmpPaths(self.mod):
            with open(self.mod._CATALOG_PATH, "w", encoding="utf-8") as f:
                f.write("{ this is not json")
            self.assertIsNone(self.mod._load_catalog())

    def test_atomic_write_cleans_tmp_on_dump_failure(self):
        with _TmpPaths(self.mod):
            data_dir = self.mod._DATA_DIR
            # default=str is used, but a set is unserialisable even with that,
            # so json.dump raises → the except branch must unlink the tmp file.
            class _Boom:
                def __repr__(self):  # default=str calls str(), keep it raising
                    raise ValueError("nope")
            with mock.patch.object(self.mod.json, "dump",
                                   side_effect=ValueError("boom")):
                with self.assertRaises(ValueError):
                    self.mod._atomic_write_json(
                        os.path.join(data_dir, "x.json"), {"a": 1})
            # No leftover .tmp files.
            leftovers = [n for n in os.listdir(data_dir) if n.endswith(".tmp")]
            self.assertEqual(leftovers, [])


# ── _merge_with_existing_catalog ────────────────────────────────────────────
class MergeCatalogTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def _fresh(self):
        return {"devices": [
            {"alexa_entity_id": "e1", "name": "Fresh Name",
             "controller_skill": None, "lan_ip": "", "lan_mac": ""},
            {"alexa_entity_id": "e2", "name": "Only In Fresh",
             "controller_skill": "sh_hue", "lan_ip": "", "lan_mac": ""},
        ]}

    def test_no_existing_returns_fresh_unchanged(self):
        with _TmpPaths(self.mod):
            fresh = self._fresh()
            self.assertIs(self.mod._merge_with_existing_catalog(fresh), fresh)

    def test_existing_user_overrides_preserved(self):
        with _TmpPaths(self.mod):
            existing = {"devices": [
                {"alexa_entity_id": "e1", "name": "User Renamed",
                 "controller_skill": "sh_custom", "lan_ip": "10.0.0.9",
                 "lan_mac": "AA:BB:CC:DD:EE:FF"},
            ]}
            self.mod._save_catalog(existing)
            merged = self.mod._merge_with_existing_catalog(self._fresh())
        d1 = next(d for d in merged["devices"] if d["alexa_entity_id"] == "e1")
        self.assertEqual(d1["name"], "User Renamed")          # preserved
        self.assertEqual(d1["controller_skill"], "sh_custom")  # preserved
        self.assertEqual(d1["lan_ip"], "10.0.0.9")            # preserved
        # e2 has no existing match → untouched.
        d2 = next(d for d in merged["devices"] if d["alexa_entity_id"] == "e2")
        self.assertEqual(d2["name"], "Only In Fresh")

    def test_existing_without_devices_list_returns_fresh(self):
        with _TmpPaths(self.mod):
            self.mod._save_catalog({"version": 1})  # no 'devices' key
            fresh = self._fresh()
            self.assertIs(self.mod._merge_with_existing_catalog(fresh), fresh)

    def test_existing_with_no_entity_ids_returns_fresh(self):
        with _TmpPaths(self.mod):
            self.mod._save_catalog({"devices": [{"name": "no id here"}]})
            fresh = self._fresh()
            self.assertIs(self.mod._merge_with_existing_catalog(fresh), fresh)

    def test_fresh_device_missing_entity_id_skipped(self):
        with _TmpPaths(self.mod):
            self.mod._save_catalog({"devices": [
                {"alexa_entity_id": "e1", "name": "Keep"}]})
            fresh = {"devices": [
                {"name": "no id"},  # skipped (no eid)
                {"alexa_entity_id": "e1", "name": "Fresh"},
            ]}
            merged = self.mod._merge_with_existing_catalog(fresh)
        d1 = next(d for d in merged["devices"]
                  if d.get("alexa_entity_id") == "e1")
        self.assertEqual(d1["name"], "Keep")


# ── cookie persistence: _save_cookie_meta / _load_cookie_meta ───────────────
class CookieMetaTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_save_writes_pickle_and_meta(self):
        with _TmpPaths(self.mod):
            jar = {"sess": "abc"}  # any picklable object
            self.mod._save_cookie_meta("me@example.com",
                                       {"login_successful": True, "x": 1}, jar)
            self.assertTrue(os.path.exists(self.mod._COOKIE_PICKLE_PATH))
            meta = self.mod._load_cookie_meta()
        self.assertEqual(meta["email"], "me@example.com")
        self.assertEqual(meta["version"], 1)
        self.assertIn("login_successful", meta["status_keys"])
        self.assertIn("x", meta["status_keys"])
        # round-trip the pickle to be sure it's the same object.
        with _TmpPaths(self.mod):
            pass

    def test_save_handles_pickle_failure_but_writes_meta(self):
        with _TmpPaths(self.mod):
            with mock.patch.object(self.mod.pickle, "dump",
                                   side_effect=OSError("disk full")):
                self.mod._save_cookie_meta("e@e.com", {"a": 1}, object())
            # meta JSON still written despite pickle failure.
            meta = self.mod._load_cookie_meta()
        self.assertEqual(meta["email"], "e@e.com")

    def test_save_cookie_meta_none_status(self):
        with _TmpPaths(self.mod):
            self.mod._save_cookie_meta("e@e.com", None, {"j": 1})
            meta = self.mod._load_cookie_meta()
        self.assertEqual(meta["status_keys"], [])

    def test_load_cookie_meta_missing(self):
        with _TmpPaths(self.mod):
            self.assertIsNone(self.mod._load_cookie_meta())

    def test_load_cookie_meta_corrupt(self):
        with _TmpPaths(self.mod):
            with open(self.mod._COOKIE_JSON_PATH, "w", encoding="utf-8") as f:
                f.write("not json {{{")
            self.assertIsNone(self.mod._load_cookie_meta())


# ── _entity_room / _entity_groups / _entity_to_record / _build_catalog ──────
class EntityRoomGroupTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_room_direct_field(self):
        self.assertEqual(self.mod._entity_room({"roomName": "Den"}, []), "Den")
        self.assertEqual(self.mod._entity_room({"room": "Loft"}, []), "Loft")

    def test_room_via_linked_echo(self):
        ent = {"applianceDetails": {"alexaDeviceId": "SN-123"}}
        echos = [{"serialNumber": "SN-123", "accountName": "Office Echo"}]
        self.assertEqual(self.mod._entity_room(ent, echos), "Office Echo")

    def test_room_via_linked_echo_alt_serial_and_name(self):
        ent = {"applianceDetails": {"alexaDeviceId": "SN-9"}}
        echos = [{"deviceSerialNumber": "SN-9", "deviceName": "Bedroom Dot"}]
        self.assertEqual(self.mod._entity_room(ent, echos), "Bedroom Dot")

    def test_room_empty_when_no_match(self):
        self.assertEqual(self.mod._entity_room({}, []), "")
        ent = {"applianceDetails": {"alexaDeviceId": "X"}}
        self.assertEqual(self.mod._entity_room(ent, [{"serialNumber": "Y"}]), "")

    def test_groups_membership_variants(self):
        groups = [
            {"name": "Downstairs", "applianceIds": ["e1", "e2"]},
            {"groupName": "Lights", "entityIds": ["e1"]},
            {"name": "Other", "members": ["e9"]},
        ]
        out = self.mod._entity_groups("e1", groups)
        self.assertIn("Downstairs", out)
        self.assertIn("Lights", out)
        self.assertNotIn("Other", out)

    def test_groups_empty_inputs(self):
        self.assertEqual(self.mod._entity_groups("", [{"name": "x"}]), [])
        self.assertEqual(self.mod._entity_groups("e1", []), [])


class EntityToRecordTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_full_record_assembly_with_lan_match(self):
        arp = [{"ip": "10.0.0.5", "mac": "00:17:88:AA:BB:CC",
                "brand_oui_hint": "Philips Hue"}]
        ent = {
            "friendlyName": "Living Room Lamp",
            "manufacturerName": "Philips Hue",
            "modelName": "LCT015",
            "capabilities": [{"interface": "Alexa.PowerController"},
                             {"interface": "Alexa.ColorController"}],
            "entityId": "e-100",
            "roomName": "Living Room",
        }
        groups = [{"name": "Mood", "applianceIds": ["e-100"]}]
        rec = self.mod._entity_to_record(ent, [], groups, arp)
        self.assertEqual(rec["name"], "Living Room Lamp")
        self.assertEqual(rec["brand"], "Philips Hue")
        self.assertEqual(rec["model"], "LCT015")
        self.assertEqual(rec["type"], "light")
        self.assertIn("color", rec["capabilities"])
        self.assertIn("on_off", rec["capabilities"])
        self.assertEqual(rec["alexa_entity_id"], "e-100")
        self.assertEqual(rec["alexa_room"], "Living Room")
        self.assertEqual(rec["alexa_groups"], ["Mood"])
        self.assertEqual(rec["lan_ip"], "10.0.0.5")
        self.assertEqual(rec["lan_mac"], "00:17:88:AA:BB:CC")
        self.assertEqual(rec["controller_skill"], "sh_hue")

    def test_record_falls_back_to_appliance_details(self):
        ent = {
            "applianceDetails": {
                "friendlyName": "Garage Plug",
                "manufacturerName": "Generic Co",
                "modelName": "PLG1",
                "capabilities": [{"interface": "Alexa.PowerController"}],
                "applianceId": "appl-7",
            },
        }
        rec = self.mod._entity_to_record(ent, [], [], [])
        self.assertEqual(rec["name"], "Garage Plug")
        self.assertEqual(rec["brand"], "Generic Co")
        self.assertEqual(rec["model"], "PLG1")
        self.assertEqual(rec["type"], "plug")
        self.assertEqual(rec["alexa_entity_id"], "appl-7")
        self.assertIsNone(rec["controller_skill"])
        self.assertEqual(rec["lan_ip"], "")

    def test_record_unnamed_default(self):
        rec = self.mod._entity_to_record({}, [], [], [])
        self.assertEqual(rec["name"], "(unnamed)")
        self.assertEqual(rec["brand"], "")
        self.assertEqual(rec["type"], "unknown")


class BuildCatalogTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_build_catalog_counts_and_sorts(self):
        devices = {
            "echo": [{"serialNumber": "SN1"}, {"serialNumber": "SN2"}],
            "smarthome": [
                {"friendlyName": "Z Lamp", "manufacturerName": "Hue",
                 "entityId": "e1", "roomName": "Kitchen",
                 "capabilities": [{"interface": "Alexa.ColorController"}]},
                {"friendlyName": "A Lamp", "manufacturerName": "Hue",
                 "entityId": "e2", "roomName": "Kitchen",
                 "capabilities": [{"interface": "Alexa.ColorController"}]},
                "not-a-dict",  # ignored by isinstance filter
            ],
            "groups": [{"name": "G1"}],
        }
        arp = [{"ip": "1.1.1.1", "mac": "00:00:00:00:00:00",
                "brand_oui_hint": None}]
        cat = self.mod._build_catalog(devices, arp)
        self.assertEqual(cat["device_count"], 2)
        self.assertEqual(cat["echo_count"], 2)
        self.assertEqual(cat["group_count"], 1)
        self.assertEqual(cat["arp_seen"], 1)
        self.assertEqual(cat["version"], 1)
        # Sorted by (room, name) → 'A Lamp' before 'Z Lamp'.
        self.assertEqual([d["name"] for d in cat["devices"]],
                         ["A Lamp", "Z Lamp"])
        self.assertTrue(cat["generated_at"].endswith("Z"))

    def test_build_catalog_empty_inputs(self):
        cat = self.mod._build_catalog({}, [])
        self.assertEqual(cat["device_count"], 0)
        self.assertEqual(cat["echo_count"], 0)
        self.assertEqual(cat["devices"], [])


# ── _queue_missing_skill_tasks ──────────────────────────────────────────────
class QueueMissingSkillTaskTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def _catalog(self, *brands_with_skill):
        return {"devices": [
            {"brand": b, "controller_skill": s} for b, s in brands_with_skill]}

    def test_no_todo_file_returns_zero(self):
        with _TmpPaths(self.mod):
            # _TODO_PATH does not exist yet.
            n = self.mod._queue_missing_skill_tasks(
                self._catalog(("Wyze", None)))
        self.assertEqual(n, 0)

    def test_appends_one_task_per_unknown_brand(self):
        with _TmpPaths(self.mod):
            with open(self.mod._TODO_PATH, "w", encoding="utf-8") as f:
                f.write("# todo\n")
            cat = self._catalog(("Wyze", None), ("Hue", "sh_hue"),
                                ("Eufy Security", None))
            n = self.mod._queue_missing_skill_tasks(cat)
            with open(self.mod._TODO_PATH, "r", encoding="utf-8") as f:
                body = f.read()
        self.assertEqual(n, 2)
        self.assertIn("brand 'Wyze'", body)
        self.assertIn("brand 'Eufy Security'", body)
        self.assertIn("skills/sh_wyze.py", body)
        self.assertIn("skills/sh_eufy_security.py", body)  # slugified
        self.assertNotIn("brand 'Hue'", body)  # has controller skill

    def test_idempotent_skips_existing_marker(self):
        with _TmpPaths(self.mod):
            cat = self._catalog(("Wyze", None))
            with open(self.mod._TODO_PATH, "w", encoding="utf-8") as f:
                f.write("# todo\n")
            first = self.mod._queue_missing_skill_tasks(cat)
            second = self.mod._queue_missing_skill_tasks(cat)
        self.assertEqual(first, 1)
        self.assertEqual(second, 0)  # marker already present → nothing added

    def test_no_missing_brands_returns_zero(self):
        with _TmpPaths(self.mod):
            with open(self.mod._TODO_PATH, "w", encoding="utf-8") as f:
                f.write("# todo\n")
            n = self.mod._queue_missing_skill_tasks(
                self._catalog(("Hue", "sh_hue")))
        self.assertEqual(n, 0)

    def test_unknown_brand_blank_string_ignored(self):
        with _TmpPaths(self.mod):
            with open(self.mod._TODO_PATH, "w", encoding="utf-8") as f:
                f.write("# todo\n")
            n = self.mod._queue_missing_skill_tasks(
                self._catalog(("   ", None)))
        self.assertEqual(n, 0)

    def test_read_failure_returns_zero(self):
        with _TmpPaths(self.mod):
            with open(self.mod._TODO_PATH, "w", encoding="utf-8") as f:
                f.write("# todo\n")
            with mock.patch.object(self.mod, "open",
                                   side_effect=OSError("locked"), create=True):
                n = self.mod._queue_missing_skill_tasks(
                    self._catalog(("Wyze", None)))
        self.assertEqual(n, 0)

    def test_append_failure_returns_zero(self):
        with _TmpPaths(self.mod):
            with open(self.mod._TODO_PATH, "w", encoding="utf-8") as f:
                f.write("# todo\n")
            real_open = open

            def flaky_open(*a, **k):
                # first call (read) ok; second call (append) explodes.
                mode = a[1] if len(a) > 1 else k.get("mode", "r")
                if "a" in mode:
                    raise OSError("append fail")
                return real_open(*a, **k)

            with mock.patch.object(self.mod, "open", side_effect=flaky_open,
                                   create=True):
                n = self.mod._queue_missing_skill_tasks(
                    self._catalog(("Wyze", None)))
        self.assertEqual(n, 0)


# ── _construct_login ────────────────────────────────────────────────────────
class ConstructLoginTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_none_when_alexapy_absent(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=None):
            self.assertIsNone(self.mod._construct_login("e", "p"))

    def test_first_candidate_succeeds(self):
        seen = {}

        class _Login:
            def __init__(self, **kw):
                seen.update(kw)

        fake = _make_fake_alexapy(login_ctor=_Login)
        with mock.patch.object(self.mod, "_alexapy", return_value=fake):
            obj = self.mod._construct_login("me@x.com", "pw")
        self.assertIsInstance(obj, _Login)
        # First candidate passes outputpath as a callable.
        self.assertTrue(callable(seen.get("outputpath")))
        self.assertEqual(seen.get("email"), "me@x.com")

    def test_falls_through_typeerror_to_later_candidate(self):
        attempts = []

        class _Login:
            def __init__(self, **kw):
                attempts.append(set(kw.keys()))
                # Reject the first two shapes (callable + dir outputpath),
                # accept only the minimal 3-arg kwargs.
                if "outputpath" in kw or "debug" in kw:
                    raise TypeError("unexpected kwarg")

        fake = _make_fake_alexapy(login_ctor=_Login)
        with mock.patch.object(self.mod, "_alexapy", return_value=fake):
            obj = self.mod._construct_login("e", "p")
        self.assertIsInstance(obj, _Login)
        self.assertEqual(len(attempts), 3)  # tried all three

    def test_non_typeerror_breaks_and_returns_none(self):
        class _Login:
            def __init__(self, **kw):
                raise RuntimeError("ctor blew up")

        fake = _make_fake_alexapy(login_ctor=_Login)
        with mock.patch.object(self.mod, "_alexapy", return_value=fake):
            self.assertIsNone(self.mod._construct_login("e", "p"))


# ── _login_async (the CAPTCHA / OTP / claimspicker state machine) ───────────
class LoginAsyncTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def _drive(self, login, inputs=None):
        """Run _login_async with _construct_login stubbed to return `login`
        and input() feeding scripted answers."""
        feed = iter(inputs or [])
        with mock.patch.object(self.mod, "_construct_login", return_value=login), \
             mock.patch.object(self.mod, "input",
                               side_effect=lambda *_a: next(feed), create=True):
            return run_coro(self.mod._login_async("e@e.com", "pw"))

    def test_immediate_success(self):
        login = _FakeLogin(statuses=[{"login_successful": True}])
        out = self._drive(login)
        self.assertIs(out, login)

    def test_construct_failure_returns_none(self):
        with mock.patch.object(self.mod, "_construct_login", return_value=None):
            out = run_coro(self.mod._login_async("e", "p"))
        self.assertIsNone(out)

    def test_captcha_then_success_feeds_answer(self):
        login = _FakeLogin(statuses=[
            {"captcha_required": True, "captcha_image_url": "http://img"},
            {"login_successful": True},
        ])
        out = self._drive(login, inputs=["ABCD"])
        self.assertIs(out, login)
        # The captcha answer was fed back to login(data={"captcha": ...}).
        self.assertEqual(login.calls[-1]["data"], {"captcha": "ABCD"})

    def test_claimspicker_then_success(self):
        login = _FakeLogin(statuses=[
            {"claimspicker_required": True,
             "claimspicker_options": {"0": "email", "1": "sms"}},
            {"login_successful": True},
        ])
        out = self._drive(login, inputs=["1"])
        self.assertIs(out, login)
        self.assertEqual(login.calls[-1]["data"], {"claimsoption": "1"})

    def test_authselect_then_success(self):
        login = _FakeLogin(statuses=[
            {"authselect_required": True, "authselect_options": {"0": "app"}},
            {"login_successful": True},
        ])
        out = self._drive(login, inputs=["0"])
        self.assertIs(out, login)
        self.assertEqual(login.calls[-1]["data"], {"authselectoption": "0"})

    def test_verificationcode_then_success(self):
        login = _FakeLogin(statuses=[
            {"verificationcode_required": True},
            {"login_successful": True},
        ])
        out = self._drive(login, inputs=["123456"])
        self.assertIs(out, login)
        self.assertEqual(login.calls[-1]["data"], {"verificationcode": "123456"})

    def test_securitycode_then_success(self):
        login = _FakeLogin(statuses=[
            {"securitycode_required": True},
            {"login_successful": True},
        ])
        out = self._drive(login, inputs=["999"])
        self.assertIs(out, login)
        self.assertEqual(login.calls[-1]["data"], {"securitycode": "999"})

    def test_login_failed_returns_none(self):
        login = _FakeLogin(statuses=[{"login_failed": True, "error": "bad pw"}])
        out = self._drive(login)
        self.assertIsNone(out)

    def test_unknown_state_raises_needs_playwright(self):
        login = _FakeLogin(statuses=[{"some_unknown_key": True}])
        with self.assertRaises(self.mod._LoginNeedsPlaywright):
            self._drive(login)

    def test_eof_during_captcha_returns_none(self):
        login = _FakeLogin(statuses=[
            {"captcha_required": True, "captcha_url": "u"}])

        def _raise_eof(*_a):
            raise EOFError()

        with mock.patch.object(self.mod, "_construct_login", return_value=login), \
             mock.patch.object(self.mod, "input", side_effect=_raise_eof,
                               create=True):
            out = run_coro(self.mod._login_async("e", "p"))
        self.assertIsNone(out)

    def test_step_limit_exceeded_raises_needs_playwright(self):
        # 8 consecutive captcha prompts → never succeeds → loop limit hit.
        login = _FakeLogin(statuses=[
            {"captcha_required": True, "captcha_url": "u"} for _ in range(20)])
        feed = iter(["x"] * 20)
        with mock.patch.object(self.mod, "_construct_login", return_value=login), \
             mock.patch.object(self.mod, "input",
                               side_effect=lambda *_a: next(feed), create=True):
            with self.assertRaises(self.mod._LoginNeedsPlaywright):
                run_coro(self.mod._login_async("e", "p"))


# ── _convert_playwright_cookies ─────────────────────────────────────────────
class ConvertPlaywrightCookiesTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_builds_jar_with_attributes(self):
        cookies = [
            {"name": "session-id", "value": "abc", "domain": ".amazon.com",
             "path": "/", "secure": True, "httpOnly": True,
             "expires": 1893456000},
            {"name": "", "value": "skip-me"},        # no name → skipped
            {"value": "noname"},                      # no name → skipped
        ]
        jar = self.mod._convert_playwright_cookies(cookies)
        self.assertEqual(jar.get("session-id"), "abc")
        names = [c.name for c in jar]
        self.assertEqual(names, ["session-id"])

    def test_negative_expiry_ignored(self):
        cookies = [{"name": "x", "value": "1", "expires": -1}]
        jar = self.mod._convert_playwright_cookies(cookies)
        self.assertEqual(jar.get("x"), "1")

    def test_empty_list_returns_empty_jar(self):
        jar = self.mod._convert_playwright_cookies([])
        self.assertEqual(list(jar), [])

    def test_none_returns_empty_jar(self):
        jar = self.mod._convert_playwright_cookies(None)
        self.assertEqual(list(jar), [])

    def test_per_cookie_exception_is_skipped(self):
        # A cookie whose .get raises must be skipped without aborting the rest.
        class _Bad(dict):
            def get(self, *a, **k):
                raise RuntimeError("boom")
        cookies = [_Bad(), {"name": "ok", "value": "v"}]
        jar = self.mod._convert_playwright_cookies(cookies)
        self.assertEqual(jar.get("ok"), "v")


# ── _login_via_playwright ───────────────────────────────────────────────────
class _FakePwPage:
    def __init__(self, url_sequence, closed_after=None):
        self._urls = list(url_sequence)
        self._closed_after = closed_after
        self._reads = 0
        self.goto_calls = []

    @property
    def url(self):
        if self._urls:
            return self._urls[0] if len(self._urls) == 1 else self._urls.pop(0)
        return ""

    def is_closed(self):
        if self._closed_after is None:
            return False
        self._reads += 1
        return self._reads > self._closed_after

    async def goto(self, url):
        self.goto_calls.append(url)


class _FakePwContext:
    def __init__(self, cookies):
        self._cookies = cookies

    async def cookies(self):
        return list(self._cookies)

    async def new_page(self):
        return self._page

    def attach_page(self, page):
        self._page = page


class _FakePwBrowser:
    def __init__(self, context, fail_new_context=False):
        self._context = context
        self.closed = False
        self._fail_new_context = fail_new_context

    async def new_context(self):
        if self._fail_new_context:
            raise RuntimeError("ctx fail")
        return self._context

    async def close(self):
        self.closed = True


class _FakeChromium:
    def __init__(self, browser, launch_errors=None):
        self._browser = browser
        # launch_errors: dict channel->Exception to raise for that channel.
        self._launch_errors = launch_errors or {}
        self.launch_calls = []

    async def launch(self, headless=False, channel=None):
        self.launch_calls.append(channel)
        if channel in self._launch_errors:
            raise self._launch_errors[channel]
        if None in self._launch_errors and channel is None:
            raise self._launch_errors[None]
        return self._browser


class _FakePwManager:
    """async context manager returned by async_playwright()."""
    def __init__(self, chromium):
        self.chromium = chromium

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _install_fake_playwright(mod, manager):
    """Patch the `from playwright.async_api import async_playwright` import the
    coroutine does at call time by injecting a fake module into sys.modules."""
    fake_async_api = types.ModuleType("playwright.async_api")
    fake_async_api.async_playwright = lambda: manager
    fake_root = types.ModuleType("playwright")
    fake_root.async_api = fake_async_api
    return mock.patch.dict(sys.modules, {
        "playwright": fake_root,
        "playwright.async_api": fake_async_api,
    })


class LoginViaPlaywrightTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")
        # neutralise the 2s poll sleep so the loop spins instantly.
        self._sleep_patch = mock.patch.object(
            self.mod.asyncio, "sleep",
            new=mock.AsyncMock(return_value=None))
        self._sleep_patch.start()
        self.addCleanup(self._sleep_patch.stop)

    def _run(self, manager, timeout=5.0):
        with _install_fake_playwright(self.mod, manager):
            return run_coro(self.mod._login_via_playwright("e@e.com", timeout))

    def test_playwright_import_missing_returns_none(self):
        # Inject a playwright package whose async_api import raises.
        broken = types.ModuleType("playwright")
        with mock.patch.dict(sys.modules, {"playwright": broken}):
            # ensure the submodule import fails
            sys.modules.pop("playwright.async_api", None)
            out = run_coro(self.mod._login_via_playwright("e", 1.0))
        self.assertIsNone(out)

    def test_success_captures_cookies(self):
        cookies = [{"name": "session-id", "value": "v", "domain": ".amazon.com"}]
        ctx = _FakePwContext(cookies)
        # First poll URL is still on /ap/signin, second has landed on amazon home.
        page = _FakePwPage(["https://www.amazon.com/ap/signin?x",
                            "https://www.amazon.com/gp/homepage"])
        ctx.attach_page(page)
        browser = _FakePwBrowser(ctx)
        chromium = _FakeChromium(browser)
        result = self._run(_FakePwManager(chromium))
        self.assertIsNotNone(result)
        jar, raw = result
        self.assertEqual(raw, cookies)
        self.assertEqual(jar.get("session-id"), "v")
        self.assertTrue(browser.closed)
        # Chrome channel tried first and succeeded.
        self.assertEqual(chromium.launch_calls, ["chrome"])

    def test_channel_fallback_to_bundled_then_edge(self):
        cookies = [{"name": "at-main", "value": "v"}]
        ctx = _FakePwContext(cookies)
        page = _FakePwPage(["https://www.amazon.com/gp/x"])
        ctx.attach_page(page)
        browser = _FakePwBrowser(ctx)
        # chrome + bundled(None) fail, msedge succeeds.
        chromium = _FakeChromium(browser, launch_errors={
            "chrome": RuntimeError("no chrome"),
            None: RuntimeError("spawn UNKNOWN"),
        })
        result = self._run(_FakePwManager(chromium))
        self.assertIsNotNone(result)
        self.assertEqual(chromium.launch_calls, ["chrome", None, "msedge"])

    def test_all_browsers_fail_returns_none(self):
        # Every channel raises at launch → new_context() is never reached.
        chromium = _FakeChromium(None, launch_errors={
            "chrome": RuntimeError("a"), None: RuntimeError("b"),
            "msedge": RuntimeError("c")})
        out = self._run(_FakePwManager(chromium))
        self.assertIsNone(out)

    def test_window_closed_grabs_cookies_anyway(self):
        cookies = [{"name": "sess", "value": "v"}]
        ctx = _FakePwContext(cookies)
        # URL never leaves /ap/; page reports closed on the 1st is_closed check.
        page = _FakePwPage(["https://www.amazon.com/ap/signin"],
                           closed_after=0)
        ctx.attach_page(page)
        browser = _FakePwBrowser(ctx)
        chromium = _FakeChromium(browser)
        result = self._run(_FakePwManager(chromium))
        self.assertIsNotNone(result)
        _jar, raw = result
        self.assertEqual(raw, cookies)

    def test_timeout_returns_none(self):
        ctx = _FakePwContext([{"name": "x", "value": "y"}])
        # URL stays on /ap/signin forever; loop.time() must exceed deadline.
        page = _FakePwPage(["https://www.amazon.com/ap/signin"])
        ctx.attach_page(page)
        browser = _FakePwBrowser(ctx)
        chromium = _FakeChromium(browser)
        # Make the event loop clock jump past the deadline on the 2nd read.
        result = self._run(_FakePwManager(chromium), timeout=0.0)
        self.assertIsNone(result)

    def test_goto_failure_returns_none(self):
        class _BadPage(_FakePwPage):
            async def goto(self, url):
                raise RuntimeError("net::ERR")
        ctx = _FakePwContext([])
        page = _BadPage(["https://www.amazon.com/ap/signin"])
        ctx.attach_page(page)
        browser = _FakePwBrowser(ctx)
        chromium = _FakeChromium(browser)
        out = self._run(_FakePwManager(chromium))
        self.assertIsNone(out)

    def test_no_cookies_captured_returns_none(self):
        ctx = _FakePwContext([])  # empty cookie capture
        page = _FakePwPage(["https://www.amazon.com/gp/home"])
        ctx.attach_page(page)
        browser = _FakePwBrowser(ctx)
        chromium = _FakeChromium(browser)
        out = self._run(_FakePwManager(chromium))
        self.assertIsNone(out)

    def test_outer_exception_returns_none(self):
        # async_playwright() context manager __aenter__ raises.
        class _BoomManager:
            async def __aenter__(self):
                raise RuntimeError("pw boom")

            async def __aexit__(self, *exc):
                return False
        out = self._run(_BoomManager())
        self.assertIsNone(out)


# ── _PlaywrightLoginShim / _build_login_from_playwright_cookies ─────────────
class PlaywrightShimTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_shim_attributes(self):
        shim = self.mod._PlaywrightLoginShim("e@e.com", {"jar": 1})
        self.assertEqual(shim.email, "e@e.com")
        self.assertEqual(shim._cookies, {"jar": 1})
        self.assertTrue(shim.status["login_successful"])
        self.assertTrue(shim.status["playwright"])

    def test_build_login_alexapy_absent(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=None):
            self.assertIsNone(
                self.mod._build_login_from_playwright_cookies("e", []))

    def test_build_login_construct_none(self):
        fake = _make_fake_alexapy()
        with mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_construct_login", return_value=None):
            self.assertIsNone(
                self.mod._build_login_from_playwright_cookies("e", []))

    def test_build_login_injects_cookies_into_session_jar(self):
        updates = []

        class _Jar:
            def update_cookies(self, sc, url):
                updates.append((sc, url))

        class _Session:
            cookie_jar = _Jar()

        login = _FakeLogin(session=_Session())
        fake = _make_fake_alexapy()
        cookies = [
            {"name": "a", "value": "1", "domain": ".amazon.com",
             "path": "/", "secure": True, "httpOnly": True},
            {"name": "", "value": "skip"},      # no name → skipped
            {"name": "b", "value": "2"},        # default domain branch
        ]
        with mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_construct_login", return_value=login):
            out = self.mod._build_login_from_playwright_cookies("e", cookies)
        self.assertIs(out, login)
        self.assertEqual(len(updates), 2)  # 'a' and 'b' injected, blank skipped

    def test_build_login_no_session_returns_none(self):
        login = _FakeLogin(session=None)
        # both _session and session attrs are None → returns None
        login.session = None
        fake = _make_fake_alexapy()
        with mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_construct_login", return_value=login):
            out = self.mod._build_login_from_playwright_cookies(
                "e", [{"name": "a", "value": "1"}])
        self.assertIsNone(out)

    def test_build_login_session_without_cookie_jar_returns_none(self):
        class _Session:
            cookie_jar = None

        login = _FakeLogin(session=_Session())
        fake = _make_fake_alexapy()
        with mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_construct_login", return_value=login):
            out = self.mod._build_login_from_playwright_cookies(
                "e", [{"name": "a", "value": "1"}])
        self.assertIsNone(out)


# ── _fetch_devices_async ────────────────────────────────────────────────────
class FetchDevicesAsyncTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_alexapy_absent_returns_empty(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=None):
            out = run_coro(self.mod._fetch_devices_async(_FakeLogin()))
        self.assertEqual(out, {})

    def test_login_none_returns_empty(self):
        fake = _make_fake_alexapy()
        with mock.patch.object(self.mod, "_alexapy", return_value=fake):
            out = run_coro(self.mod._fetch_devices_async(None))
        self.assertEqual(out, {})

    def test_collects_all_three_lists(self):
        class _API:
            @classmethod
            async def get_devices(cls, login):
                return [{"serialNumber": "SN1"}]

            @classmethod
            async def get_smarthome_devices(cls, login):
                return [{"entityId": "e1"}]

            @classmethod
            async def get_smarthome_groups(cls, login):
                return [{"name": "G"}]

        fake = _make_fake_alexapy(api=_API)
        with mock.patch.object(self.mod, "_alexapy", return_value=fake):
            out = run_coro(self.mod._fetch_devices_async(_FakeLogin()))
        self.assertEqual(out["echo"], [{"serialNumber": "SN1"}])
        self.assertEqual(out["smarthome"], [{"entityId": "e1"}])
        self.assertEqual(out["groups"], [{"name": "G"}])

    def test_appliances_and_groups_fallback_methods(self):
        # No get_smarthome_devices/get_smarthome_groups; only the legacy
        # get_appliances / get_groups variants exist.
        class _API:
            @classmethod
            async def get_devices(cls, login):
                return []

            @classmethod
            async def get_appliances(cls, login):
                return [{"applianceId": "a1"}]

            @classmethod
            async def get_groups(cls, login):
                return [{"groupName": "g1"}]

        fake = _make_fake_alexapy(api=_API)
        with mock.patch.object(self.mod, "_alexapy", return_value=fake):
            out = run_coro(self.mod._fetch_devices_async(_FakeLogin()))
        self.assertEqual(out["smarthome"], [{"applianceId": "a1"}])
        self.assertEqual(out["groups"], [{"groupName": "g1"}])

    def test_individual_call_failures_are_swallowed(self):
        class _API:
            @classmethod
            async def get_devices(cls, login):
                raise RuntimeError("echo boom")

            @classmethod
            async def get_smarthome_devices(cls, login):
                raise RuntimeError("sh boom")

            @classmethod
            async def get_smarthome_groups(cls, login):
                raise RuntimeError("grp boom")

        fake = _make_fake_alexapy(api=_API)
        with mock.patch.object(self.mod, "_alexapy", return_value=fake):
            out = run_coro(self.mod._fetch_devices_async(_FakeLogin()))
        # All failed but the structure is intact with empty lists.
        self.assertEqual(out, {"echo": [], "smarthome": [], "groups": []})

    def test_missing_methods_leave_empty_lists(self):
        class _API:  # no methods at all
            pass
        fake = _make_fake_alexapy(api=_API)
        with mock.patch.object(self.mod, "_alexapy", return_value=fake):
            out = run_coro(self.mod._fetch_devices_async(_FakeLogin()))
        self.assertEqual(out, {"echo": [], "smarthome": [], "groups": []})


# ── _extract_cookie_jar ─────────────────────────────────────────────────────
class ExtractCookieJarTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_returns_cookie_jar_when_present(self):
        class _Sess:
            cookie_jar = {"jar": 1}

        class _Login:
            _cookies = _Sess()

        self.assertEqual(self.mod._extract_cookie_jar(_Login()), {"jar": 1})

    def test_returns_value_itself_when_no_cookie_jar(self):
        class _Login:
            _cookies = None
            _session = "raw-session-object"

        self.assertEqual(self.mod._extract_cookie_jar(_Login()),
                         "raw-session-object")

    def test_returns_none_when_nothing(self):
        class _Login:
            pass
        self.assertIsNone(self.mod._extract_cookie_jar(_Login()))


# ── _load_cookie_pickle_safe / _restore_login_from_cookie ───────────────────
class CookiePickleSafeTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_missing_pickle_returns_none(self):
        with _TmpPaths(self.mod):
            self.assertIsNone(self.mod._load_cookie_pickle_safe())

    def test_loads_valid_pickle(self):
        with _TmpPaths(self.mod):
            with open(self.mod._COOKIE_PICKLE_PATH, "wb") as f:
                pickle.dump({"cookie": "data"}, f)
            got = self.mod._load_cookie_pickle_safe()
        self.assertEqual(got, {"cookie": "data"})

    def test_corrupt_pickle_returns_none(self):
        with _TmpPaths(self.mod):
            with open(self.mod._COOKIE_PICKLE_PATH, "wb") as f:
                f.write(b"\x00\x01 not a pickle")
            self.assertIsNone(self.mod._load_cookie_pickle_safe())

    def test_path_safety_rejects_when_realpath_mismatch(self):
        with _TmpPaths(self.mod):
            with open(self.mod._COOKIE_PICKLE_PATH, "wb") as f:
                pickle.dump({"x": 1}, f)
            # Force the realpath safety check to fail by making commonpath
            # report a different dir.
            with mock.patch.object(self.mod.os.path, "commonpath",
                                   return_value="C:\\elsewhere"):
                self.assertIsNone(self.mod._load_cookie_pickle_safe())


class RestoreLoginFromCookieTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_alexapy_absent_returns_none(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=None):
            self.assertIsNone(self.mod._restore_login_from_cookie())

    def test_no_meta_returns_none(self):
        fake = _make_fake_alexapy()
        with _TmpPaths(self.mod), \
             mock.patch.object(self.mod, "_alexapy", return_value=fake):
            # meta file absent
            self.assertIsNone(self.mod._restore_login_from_cookie())

    def test_meta_present_but_no_pickle_returns_none(self):
        fake = _make_fake_alexapy()
        with _TmpPaths(self.mod), \
             mock.patch.object(self.mod, "_alexapy", return_value=fake):
            self.mod._save_cookie_meta("e@e.com", {"a": 1}, {"jar": 1})
            os.unlink(self.mod._COOKIE_PICKLE_PATH)  # remove just the pickle
            self.assertIsNone(self.mod._restore_login_from_cookie())

    def test_successful_restore_sets_cookies_and_resets(self):
        fake = _make_fake_alexapy()
        login = _FakeLogin(email="e@e.com")
        with _TmpPaths(self.mod), \
             mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_construct_login", return_value=login), \
             mock.patch.object(self.mod, "_run_async",
                               side_effect=lambda c: run_coro(c)):
            self.mod._save_cookie_meta("e@e.com", {"a": 1}, {"jar": "data"})
            out = self.mod._restore_login_from_cookie()
        self.assertIs(out, login)
        self.assertEqual(login._cookies, {"jar": "data"})
        self.assertTrue(login.reset_called)

    def test_restore_construct_login_none(self):
        fake = _make_fake_alexapy()
        with _TmpPaths(self.mod), \
             mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_construct_login", return_value=None):
            self.mod._save_cookie_meta("e@e.com", {"a": 1}, {"jar": 1})
            self.assertIsNone(self.mod._restore_login_from_cookie())

    def test_restore_path_safety_failure(self):
        fake = _make_fake_alexapy()
        with _TmpPaths(self.mod), \
             mock.patch.object(self.mod, "_alexapy", return_value=fake):
            self.mod._save_cookie_meta("e@e.com", {"a": 1}, {"jar": 1})
            with mock.patch.object(self.mod.os.path, "commonpath",
                                   return_value="C:\\nope"):
                self.assertIsNone(self.mod._restore_login_from_cookie())


# ── _restore_and_fetch_async ────────────────────────────────────────────────
class RestoreAndFetchAsyncTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_alexapy_absent_returns_none(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=None):
            self.assertIsNone(run_coro(self.mod._restore_and_fetch_async()))

    def test_no_cached_cookie_returns_none(self):
        fake = _make_fake_alexapy()
        with mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_load_cookie_pickle_safe",
                               return_value=None):
            self.assertIsNone(run_coro(self.mod._restore_and_fetch_async()))

    def test_construct_none_returns_none(self):
        fake = _make_fake_alexapy()
        with mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_load_cookie_pickle_safe",
                               return_value={"c": 1}), \
             mock.patch.object(self.mod, "_load_cookie_meta",
                               return_value={"email": "e@e.com"}), \
             mock.patch.object(self.mod, "_construct_login", return_value=None):
            self.assertIsNone(run_coro(self.mod._restore_and_fetch_async()))

    def test_login_cookies_kwarg_success_then_fetch(self):
        login = _FakeLogin(statuses=[{"login_successful": True}])
        fake = _make_fake_alexapy()
        with mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_load_cookie_pickle_safe",
                               return_value={"c": 1}), \
             mock.patch.object(self.mod, "_load_cookie_meta",
                               return_value={"email": "e@e.com"}), \
             mock.patch.object(self.mod, "_construct_login", return_value=login), \
             mock.patch.object(self.mod, "_fetch_devices_async",
                               new=mock.AsyncMock(return_value={"echo": ["ok"]})):
            out = run_coro(self.mod._restore_and_fetch_async())
        self.assertEqual(out, {"echo": ["ok"]})
        # login() was called with cookies kwarg.
        self.assertEqual(login.calls[0]["cookies"], {"c": 1})

    def test_login_typeerror_falls_back_to_attr_injection(self):
        # First login(cookies=...) raises TypeError (old alexapy), then the
        # fallback path sets attrs + reset() + retries login.
        class _OldLogin(_FakeLogin):
            async def login(self, data=None, cookies=None):
                self.calls.append({"data": data, "cookies": cookies})
                if cookies is not None and len(self.calls) == 1:
                    raise TypeError("no cookies kwarg")
                self.status = {}
                return None

        login = _OldLogin(email="e")
        fake = _make_fake_alexapy()
        with mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_load_cookie_pickle_safe",
                               return_value={"c": 9}), \
             mock.patch.object(self.mod, "_load_cookie_meta",
                               return_value={"email": "e"}), \
             mock.patch.object(self.mod, "_construct_login", return_value=login), \
             mock.patch.object(self.mod, "_fetch_devices_async",
                               new=mock.AsyncMock(return_value={"smarthome": []})):
            out = run_coro(self.mod._restore_and_fetch_async())
        self.assertEqual(out, {"smarthome": []})
        self.assertTrue(login.reset_called)
        self.assertEqual(login._cookies, {"c": 9})

    def test_login_generic_exception_then_attr_injection(self):
        class _ErrLogin(_FakeLogin):
            async def login(self, data=None, cookies=None):
                self.calls.append({"data": data, "cookies": cookies})
                if len(self.calls) == 1:
                    raise RuntimeError("transient")
                return None

        login = _ErrLogin(email="e")
        fake = _make_fake_alexapy()
        with mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_load_cookie_pickle_safe",
                               return_value={"c": 1}), \
             mock.patch.object(self.mod, "_load_cookie_meta",
                               return_value={"email": "e"}), \
             mock.patch.object(self.mod, "_construct_login", return_value=login), \
             mock.patch.object(self.mod, "_fetch_devices_async",
                               new=mock.AsyncMock(return_value={})):
            out = run_coro(self.mod._restore_and_fetch_async())
        self.assertEqual(out, {})
        self.assertTrue(login.reset_called)


# ── _run_async ──────────────────────────────────────────────────────────────
class RunAsyncTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_runs_coroutine_no_loop(self):
        async def _coro():
            return 42
        self.assertEqual(self.mod._run_async(_coro()), 42)

    def test_runs_in_worker_thread_when_loop_running(self):
        async def _outer():
            async def _inner():
                return "threaded"
            # Inside a running loop, _run_async must delegate to a worker thread.
            return self.mod._run_async(_inner())
        self.assertEqual(run_coro(_outer()), "threaded")

    def test_worker_thread_propagates_exception(self):
        async def _outer():
            async def _inner():
                raise ValueError("boom-in-thread")
            return self.mod._run_async(_inner())
        with self.assertRaises(ValueError):
            run_coro(_outer())


# ── _say ─────────────────────────────────────────────────────────────────────
class SayTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_say_uses_bobert_companion_speak(self):
        bc = mock.MagicMock()
        with mock.patch.object(self.mod, "_bc", return_value=bc):
            self.mod._say("hello sir")
        bc._speak.assert_called_once_with("hello sir")

    def test_say_falls_back_to_print_on_failure(self):
        with mock.patch.object(self.mod, "_bc",
                               side_effect=RuntimeError("no audio")):
            # Must not raise.
            self.mod._say("fallback text")


# ── _alexapy import helper ──────────────────────────────────────────────────
class AlexapyImportTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_alexapy_import_failure_returns_none(self):
        # Force the inner `import alexapy` to raise.
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "alexapy":
                raise ImportError("not installed")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            self.assertIsNone(self.mod._alexapy())

    def test_alexapy_import_success_returns_module(self):
        sentinel = types.ModuleType("alexapy")
        with mock.patch.dict(sys.modules, {"alexapy": sentinel}):
            self.assertIs(self.mod._alexapy(), sentinel)


# ── smart_home_discover (voice action) ──────────────────────────────────────
class SmartHomeDiscoverActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("smart_home_discover")
        self.fake = _make_fake_alexapy()

    def test_no_cookie_returns_cli_hint(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=None), \
             mock.patch.object(self.mod, "_say") as say:
            out = self.actions["smart_home_discover"]("")
        self.assertIn("interactive terminal", out)
        say.assert_called_once()

    def test_force_refresh_returns_cli_hint(self):
        # Even with a cookie, 'force' forces the CLI hint.
        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta",
                               return_value={"email": "e"}), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod, "_say"):
            out = self.actions["smart_home_discover"]("please force reauth")
        self.assertIn("interactive terminal", out)

    def test_cached_refresh_success_full_path(self):
        devices = {"echo": [{"serialNumber": "SN"}],
                   "smarthome": [{"friendlyName": "Lamp",
                                  "manufacturerName": "Hue", "entityId": "e1",
                                  "capabilities": [
                                      {"interface": "Alexa.ColorController"}]}],
                   "groups": []}
        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta",
                               return_value={"email": "e"}), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod, "_run_async", return_value=devices), \
             mock.patch.object(self.mod, "_scan_lan_arp", return_value=[]), \
             mock.patch.object(self.mod, "_merge_with_existing_catalog",
                               side_effect=lambda c: c), \
             mock.patch.object(self.mod, "_save_catalog") as save, \
             mock.patch.object(self.mod, "_queue_missing_skill_tasks",
                               return_value=1), \
             mock.patch.object(self.mod, "_say"):
            out = self.actions["smart_home_discover"]("")
        self.assertIn("Catalog complete", out)
        save.assert_called_once()

    def test_cached_refresh_fetch_returns_none_gives_cli_hint(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta",
                               return_value={"email": "e"}), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod, "_run_async", return_value=None), \
             mock.patch.object(self.mod, "_say"):
            out = self.actions["smart_home_discover"]("")
        self.assertIn("interactive terminal", out)

    def test_cached_refresh_fetch_raises_returns_error(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta",
                               return_value={"email": "e"}), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch.object(self.mod, "_run_async",
                               side_effect=RuntimeError("net down")), \
             mock.patch.object(self.mod, "_say"):
            out = self.actions["smart_home_discover"]("")
        self.assertIn("Catalog refresh failed", out)
        self.assertIn("net down", out)

    def test_wizard_lock_already_held(self):
        # Acquire the lock so the action reports "already running".
        self.assertTrue(self.mod._wizard_lock.acquire(blocking=False))
        try:
            with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
                 mock.patch.object(self.mod, "_load_cookie_meta",
                                   return_value={"email": "e"}), \
                 mock.patch.object(self.mod.os.path, "exists",
                                   return_value=True):
                out = self.actions["smart_home_discover"]("")
        finally:
            self.mod._wizard_lock.release()
        self.assertIn("already running", out)


# ── _summarise_catalog ──────────────────────────────────────────────────────
class SummariseCatalogTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def _cat(self, devices):
        return {
            "device_count": len(devices),
            "echo_count": 2,
            "group_count": 1,
            "devices": devices,
        }

    def test_unknown_brands_and_lan_cross_ref(self):
        devices = [
            {"controller_skill": None, "lan_ip": "10.0.0.1"},
            {"controller_skill": "sh_hue", "lan_ip": ""},
        ]
        out = self.mod._summarise_catalog(self._cat(devices),
                                          arp_table=[{"ip": "10.0.0.1"}],
                                          queued=3, speak=False)
        self.assertIn("1 brand(s) have no controller skill", out)
        self.assertIn("queued 3 build task(s)", out)
        self.assertIn("1 device(s) cross-referenced", out)

    def test_all_mapped_no_arp(self):
        devices = [{"controller_skill": "sh_hue", "lan_ip": ""}]
        out = self.mod._summarise_catalog(self._cat(devices), arp_table=[],
                                          queued=0, speak=False)
        self.assertIn("Every brand has a controller skill mapped", out)
        self.assertNotIn("cross-referenced", out)

    def test_speak_true_invokes_say(self):
        devices = [{"controller_skill": "sh_hue", "lan_ip": ""}]
        with mock.patch.object(self.mod, "_say") as say:
            self.mod._summarise_catalog(self._cat(devices), arp_table=[],
                                        queued=0, speak=True)
        say.assert_called_once()


# ── _run_wizard_interactive ─────────────────────────────────────────────────
class RunWizardInteractiveTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")
        self.fake = _make_fake_alexapy()

    def test_lock_already_held(self):
        self.assertTrue(self.mod._wizard_lock.acquire(blocking=False))
        try:
            out = self.mod._run_wizard_interactive("")
        finally:
            self.mod._wizard_lock.release()
        self.assertIn("already running", out)

    def test_alexapy_absent_returns_offline(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=None):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("offline", out)

    def test_cached_cookie_reuse_success(self):
        devices = {"echo": [], "smarthome": [], "groups": []}
        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta",
                               return_value={"email": "e",
                                             "saved_at": 1e9}), \
             mock.patch.object(self.mod, "_cookie_is_stale", return_value=False), \
             mock.patch.object(self.mod, "_run_async", return_value=devices), \
             mock.patch.object(self.mod, "_scan_lan_arp", return_value=[]), \
             mock.patch.object(self.mod, "_merge_with_existing_catalog",
                               side_effect=lambda c: c), \
             mock.patch.object(self.mod, "_save_catalog"), \
             mock.patch.object(self.mod, "_queue_missing_skill_tasks",
                               return_value=0):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("Catalog complete", out)

    def test_stale_cookie_warns_then_reuses(self):
        devices = {"echo": [], "smarthome": [], "groups": []}
        import time as _t
        old_meta = {"email": "e", "saved_at": _t.time() - 400 * 86400}
        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta",
                               return_value=old_meta), \
             mock.patch.object(self.mod, "_run_async", return_value=devices), \
             mock.patch.object(self.mod, "_scan_lan_arp", return_value=[]), \
             mock.patch.object(self.mod, "_merge_with_existing_catalog",
                               side_effect=lambda c: c), \
             mock.patch.object(self.mod, "_save_catalog"), \
             mock.patch.object(self.mod, "_queue_missing_skill_tasks",
                               return_value=0):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("Catalog complete", out)

    def test_cached_fetch_raises_then_prompts_and_cancels(self):
        # cached fetch raises → devices None → prompt creds → user cancels.
        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta",
                               return_value={"email": "e", "saved_at": 1e9}), \
             mock.patch.object(self.mod, "_cookie_is_stale", return_value=False), \
             mock.patch.object(self.mod, "_run_async",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(self.mod, "_prompt_credentials",
                               return_value=None):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("cancelled", out)

    def test_fresh_login_success_full_path(self):
        login = _FakeLogin(statuses=[{"login_successful": True}])
        devices = {"echo": [], "smarthome": [], "groups": []}
        run_returns = iter([login, devices])  # _login_async, then fetch
        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=None), \
             mock.patch.object(self.mod, "_prompt_credentials",
                               return_value=("e@e.com", "pw")), \
             mock.patch.object(self.mod, "_run_async",
                               side_effect=lambda c: next(run_returns)), \
             mock.patch.object(self.mod, "_extract_cookie_jar",
                               return_value={"jar": 1}), \
             mock.patch.object(self.mod, "_save_cookie_meta"), \
             mock.patch.object(self.mod, "_scan_lan_arp", return_value=[]), \
             mock.patch.object(self.mod, "_merge_with_existing_catalog",
                               side_effect=lambda c: c), \
             mock.patch.object(self.mod, "_save_catalog"), \
             mock.patch.object(self.mod, "_queue_missing_skill_tasks",
                               return_value=0):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("Catalog complete", out)

    def test_fresh_login_returns_none_fails(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=None), \
             mock.patch.object(self.mod, "_prompt_credentials",
                               return_value=("e@e.com", "pw")), \
             mock.patch.object(self.mod, "_run_async", return_value=None):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("did not complete", out)

    def test_fresh_login_raises_returns_error(self):
        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=None), \
             mock.patch.object(self.mod, "_prompt_credentials",
                               return_value=("e@e.com", "pw")), \
             mock.patch.object(self.mod, "_run_async",
                               side_effect=RuntimeError("sign-in boom")):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("failed during sign-in", out)

    def test_playwright_fallback_success(self):
        # _login_async raises _LoginNeedsPlaywright; playwright path returns a
        # jar + cookies; subsequent restore-fetch yields devices.
        devices = {"echo": [], "smarthome": [], "groups": []}
        jar = object()
        pw_cookies = [{"name": "a", "value": "1"}]

        def _run(coro):
            # distinguish coroutines by their qualified name.
            name = getattr(coro, "__qualname__", "") or getattr(
                getattr(coro, "cr_code", None), "co_name", "")
            try:
                coro.close()
            except Exception:
                pass
            if "_login_async" in name:
                raise self.mod._LoginNeedsPlaywright()
            if "_login_via_playwright" in name:
                return (jar, pw_cookies)
            if "_restore_and_fetch_async" in name:
                return devices
            return None

        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=None), \
             mock.patch.object(self.mod, "_prompt_credentials",
                               return_value=("e@e.com", "pw")), \
             mock.patch.object(self.mod, "_run_async", side_effect=_run), \
             mock.patch.object(self.mod, "_save_cookie_meta"), \
             mock.patch.object(self.mod, "_scan_lan_arp", return_value=[]), \
             mock.patch.object(self.mod, "_merge_with_existing_catalog",
                               side_effect=lambda c: c), \
             mock.patch.object(self.mod, "_save_catalog"), \
             mock.patch.object(self.mod, "_queue_missing_skill_tasks",
                               return_value=0):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("Catalog complete", out)

    def test_playwright_fallback_unavailable(self):
        def _run(coro):
            name = getattr(coro, "__qualname__", "")
            try:
                coro.close()
            except Exception:
                pass
            if "_login_async" in name:
                raise self.mod._LoginNeedsPlaywright()
            if "_login_via_playwright" in name:
                return None
            return None

        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=None), \
             mock.patch.object(self.mod, "_prompt_credentials",
                               return_value=("e@e.com", "pw")), \
             mock.patch.object(self.mod, "_run_async", side_effect=_run):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("did not complete", out)

    def test_playwright_fallback_empty_device_fetch(self):
        jar = object()
        pw_cookies = [{"name": "a", "value": "1"}]

        def _run(coro):
            name = getattr(coro, "__qualname__", "")
            try:
                coro.close()
            except Exception:
                pass
            if "_login_async" in name:
                raise self.mod._LoginNeedsPlaywright()
            if "_login_via_playwright" in name:
                return (jar, pw_cookies)
            if "_restore_and_fetch_async" in name:
                return None  # device fetch empty
            return None

        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=None), \
             mock.patch.object(self.mod, "_prompt_credentials",
                               return_value=("e@e.com", "pw")), \
             mock.patch.object(self.mod, "_run_async", side_effect=_run), \
             mock.patch.object(self.mod, "_save_cookie_meta"):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("device fetch", out)

    def test_fresh_fetch_raises_returns_error(self):
        # login succeeds (non-playwright), but the SUBSEQUENT device fetch
        # (second _run_async) raises.
        login = _FakeLogin(statuses=[{"login_successful": True}])
        seq = iter([login])

        def _run(coro):
            try:
                return next(seq)
            except StopIteration:
                raise RuntimeError("fetch boom")
            finally:
                try:
                    coro.close()
                except Exception:
                    pass

        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=None), \
             mock.patch.object(self.mod, "_prompt_credentials",
                               return_value=("e@e.com", "pw")), \
             mock.patch.object(self.mod, "_run_async", side_effect=_run), \
             mock.patch.object(self.mod, "_extract_cookie_jar",
                               return_value={}), \
             mock.patch.object(self.mod, "_save_cookie_meta"):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("failed during device fetch", out)

    def test_cookie_save_failure_is_swallowed(self):
        login = _FakeLogin(statuses=[{"login_successful": True}])
        devices = {"echo": [], "smarthome": [], "groups": []}
        seq = iter([login, devices])
        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=None), \
             mock.patch.object(self.mod, "_prompt_credentials",
                               return_value=("e@e.com", "pw")), \
             mock.patch.object(self.mod, "_run_async",
                               side_effect=lambda c: next(seq)), \
             mock.patch.object(self.mod, "_extract_cookie_jar",
                               side_effect=RuntimeError("extract boom")), \
             mock.patch.object(self.mod, "_save_catalog"), \
             mock.patch.object(self.mod, "_scan_lan_arp", return_value=[]), \
             mock.patch.object(self.mod, "_merge_with_existing_catalog",
                               side_effect=lambda c: c), \
             mock.patch.object(self.mod, "_queue_missing_skill_tasks",
                               return_value=0):
            out = self.mod._run_wizard_interactive("")
        # Still completes despite the cookie-save error.
        self.assertIn("Catalog complete", out)


# ── _prompt_credentials ─────────────────────────────────────────────────────
class PromptCredentialsTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_returns_email_password(self):
        with mock.patch.object(self.mod, "input", return_value="me@x.com",
                               create=True), \
             mock.patch.object(self.mod.getpass, "getpass",
                               return_value="secret"):
            self.assertEqual(self.mod._prompt_credentials(), ("me@x.com", "secret"))

    def test_blank_email_returns_none(self):
        with mock.patch.object(self.mod, "input", return_value="  ",
                               create=True):
            self.assertIsNone(self.mod._prompt_credentials())

    def test_blank_password_returns_none(self):
        with mock.patch.object(self.mod, "input", return_value="me@x.com",
                               create=True), \
             mock.patch.object(self.mod.getpass, "getpass", return_value=""):
            self.assertIsNone(self.mod._prompt_credentials())

    def test_keyboard_interrupt_returns_none(self):
        with mock.patch.object(self.mod, "input",
                               side_effect=KeyboardInterrupt, create=True):
            self.assertIsNone(self.mod._prompt_credentials())

    def test_eof_returns_none(self):
        with mock.patch.object(self.mod, "input", side_effect=EOFError,
                               create=True):
            self.assertIsNone(self.mod._prompt_credentials())


# ── smart_home_purge_cookie (extra branches) ────────────────────────────────
class PurgeCookieExtraTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("smart_home_discover")

    def test_purge_handles_unlink_error(self):
        with _TmpPaths(self.mod):
            # create both files so exists() is True
            for p in (self.mod._COOKIE_JSON_PATH, self.mod._COOKIE_PICKLE_PATH):
                with open(p, "w", encoding="utf-8") as f:
                    f.write("x")
            with mock.patch.object(self.mod.os, "unlink",
                                   side_effect=OSError("perm denied")):
                out = self.actions["smart_home_purge_cookie"]("")
        # Both unlinks failed → removed count 0 → "No cached" message.
        self.assertIn("No cached Alexa cookie", out)

    def test_purge_removes_real_files(self):
        with _TmpPaths(self.mod):
            for p in (self.mod._COOKIE_JSON_PATH, self.mod._COOKIE_PICKLE_PATH):
                with open(p, "w", encoding="utf-8") as f:
                    f.write("x")
            out = self.actions["smart_home_purge_cookie"]("")
            still_there = [p for p in (self.mod._COOKIE_JSON_PATH,
                                       self.mod._COOKIE_PICKLE_PATH)
                           if os.path.exists(p)]
        self.assertIn("cleared", out)
        self.assertEqual(still_there, [])


# ── register() wires every documented action/alias ──────────────────────────
class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("smart_home_discover")

    def test_all_actions_registered(self):
        for name in ("smart_home_discover", "discover_smart_home",
                     "smart_home_setup", "refresh_smart_home",
                     "smart_home_catalog", "list_smart_home_devices",
                     "smart_home_purge_cookie", "forget_alexa_login"):
            self.assertIn(name, self.actions)

    def test_discover_aliases_point_to_same_callable(self):
        self.assertIs(self.actions["smart_home_discover"],
                      self.actions["discover_smart_home"])
        self.assertIs(self.actions["smart_home_discover"],
                      self.actions["smart_home_setup"])
        self.assertIs(self.actions["smart_home_discover"],
                      self.actions["refresh_smart_home"])

    def test_catalog_aliases_match(self):
        self.assertIs(self.actions["smart_home_catalog"],
                      self.actions["list_smart_home_devices"])

    def test_purge_aliases_match(self):
        self.assertIs(self.actions["smart_home_purge_cookie"],
                      self.actions["forget_alexa_login"])


# ════════════════════════════════════════════════════════════════════════════
# Final branch-completion pass — picks off the remaining error/EOF/fallback
# lines so the module clears 90% comfortably.
# ════════════════════════════════════════════════════════════════════════════


class BcAndAtomicWriteBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_bc_imports_bobert_companion(self):
        sentinel = types.ModuleType("bobert_companion")
        with mock.patch.object(self.mod.importlib, "import_module",
                               return_value=sentinel) as imp:
            self.assertIs(self.mod._bc(), sentinel)
        imp.assert_called_once_with("bobert_companion")

    def test_atomic_write_unlink_runs_when_replace_fails(self):
        # os.replace raises AFTER the tmp file is written → except branch must
        # unlink the tmp file and then re-raise. unlink itself succeeds here so
        # line 178-179's happy unlink executes.
        with _TmpPaths(self.mod):
            target = os.path.join(self.mod._DATA_DIR, "out.json")
            with mock.patch.object(self.mod.os, "replace",
                                   side_effect=OSError("rename fail")):
                with self.assertRaises(OSError):
                    self.mod._atomic_write_json(target, {"k": "v"})
            leftovers = [n for n in os.listdir(self.mod._DATA_DIR)
                         if n.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_atomic_write_unlink_failure_still_reraises(self):
        # Both replace AND the cleanup unlink fail → original error still
        # propagates (the inner unlink except: pass swallows the unlink error).
        with _TmpPaths(self.mod):
            target = os.path.join(self.mod._DATA_DIR, "out2.json")
            with mock.patch.object(self.mod.os, "replace",
                                   side_effect=OSError("rename fail")), \
                 mock.patch.object(self.mod.os, "unlink",
                                   side_effect=OSError("unlink fail")):
                with self.assertRaises(OSError):
                    self.mod._atomic_write_json(target, {"k": "v"})


class ConstructLoginOutputCallbackTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_outputpath_callback_is_invoked(self):
        captured = {}

        class _Login:
            def __init__(self, **kw):
                captured["outputpath"] = kw.get("outputpath")

        fake = _make_fake_alexapy(login_ctor=_Login)
        with mock.patch.object(self.mod, "_alexapy", return_value=fake):
            self.mod._construct_login("e@x.com", "pw")
        cb = captured["outputpath"]
        self.assertTrue(callable(cb))
        # Invoking it exercises the inner _out() print line.
        cb("a status line")  # must not raise


class LoginAsyncEofMatrixTests(unittest.TestCase):
    """EOF during each interactive OTP/claims step returns None — covers the
    EOFError branches the single captcha-EOF test didn't reach."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def _eof_for(self, status):
        login = _FakeLogin(statuses=[status])

        def _raise_eof(*_a):
            raise EOFError()

        with mock.patch.object(self.mod, "_construct_login", return_value=login), \
             mock.patch.object(self.mod, "input", side_effect=_raise_eof,
                               create=True):
            return run_coro(self.mod._login_async("e", "p"))

    def test_eof_claimspicker(self):
        self.assertIsNone(self._eof_for(
            {"claimspicker_required": True, "claimspicker_options": {"0": "x"}}))

    def test_eof_authselect(self):
        self.assertIsNone(self._eof_for(
            {"authselect_required": True, "authselect_options": {"0": "x"}}))

    def test_eof_verificationcode(self):
        self.assertIsNone(self._eof_for({"verificationcode_required": True}))

    def test_eof_securitycode(self):
        self.assertIsNone(self._eof_for({"securitycode_required": True}))


class BuildLoginOuterExceptionTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_outer_exception_returns_none(self):
        # Make the import of SimpleCookie/yarl machinery blow up by giving the
        # session a cookie_jar whose update_cookies is fine but forcing the
        # OUTER try to fail: patch SimpleCookie import target via a jar that
        # raises on attribute access used before the per-cookie try.
        class _Jar:
            def update_cookies(self, sc, url):
                raise RuntimeError("ignored per-cookie")  # caught inside loop

        class _Session:
            cookie_jar = _Jar()

        login = _FakeLogin(session=_Session())
        fake = _make_fake_alexapy()
        # Force the OUTER try/except (line 746) by making YURL construction
        # raise — patch the yarl import the function performs.
        bad_yarl = types.ModuleType("yarl")

        class _BoomURL:
            def __init__(self, *a, **k):
                raise RuntimeError("yarl boom")
        bad_yarl.URL = _BoomURL
        with mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_construct_login", return_value=login), \
             mock.patch.dict(sys.modules, {"yarl": bad_yarl}):
            out = self.mod._build_login_from_playwright_cookies(
                "e", [{"name": "a", "value": "1"}])
        # Per-cookie errors are swallowed; YURL boom is also inside the loop's
        # try, so the function still returns the login (not None) here.
        self.assertIs(out, login)

    def test_outer_exception_on_session_access_returns_none(self):
        # The session lookup (`getattr(login, "session", ...)`) raising a
        # non-AttributeError propagates to the OUTER try/except (746-748),
        # which prints + returns None.
        fake = _make_fake_alexapy()

        class _Login:
            email = "e"
            _session = None  # falls through to .session

            @property
            def session(self):
                raise RuntimeError("session explode")

        login = _Login()
        with mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_construct_login", return_value=login):
            out = self.mod._build_login_from_playwright_cookies(
                "e", [{"name": "a", "value": "1"}])
        self.assertIsNone(out)


class RestoreLoginResetBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_setattr_and_reset_failures_swallowed(self):
        # A login whose _cookies setattr raises and whose reset() raises — both
        # swallowed; login still returned (covers 1030-1031, 1035-1036).
        fake = _make_fake_alexapy()

        class _Login:
            email = "e"

            @property
            def _cookies(self):
                return None

            @_cookies.setter
            def _cookies(self, v):
                raise RuntimeError("cannot set")

            def reset(self):
                raise RuntimeError("reset boom")

        login = _Login()
        with _TmpPaths(self.mod), \
             mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_construct_login", return_value=login), \
             mock.patch.object(self.mod, "_run_async",
                               side_effect=lambda c: run_coro(c)
                               if asyncio.iscoroutine(c) else c):
            self.mod._save_cookie_meta("e", {"a": 1}, {"jar": 1})
            out = self.mod._restore_login_from_cookie()
        self.assertIs(out, login)

    def test_outer_restore_exception_returns_none(self):
        # _construct_login itself raises inside the second try → outer except
        # (1038-1040) returns None.
        fake = _make_fake_alexapy()
        with _TmpPaths(self.mod), \
             mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_construct_login",
                               side_effect=RuntimeError("ctor boom")):
            self.mod._save_cookie_meta("e", {"a": 1}, {"jar": 1})
            self.assertIsNone(self.mod._restore_login_from_cookie())

    def test_pickle_load_failure_returns_none(self):
        # A valid path but pickle.load raises → 1017-1019.
        fake = _make_fake_alexapy()
        with _TmpPaths(self.mod), \
             mock.patch.object(self.mod, "_alexapy", return_value=fake):
            self.mod._save_cookie_meta("e", {"a": 1}, {"jar": 1})
            with mock.patch.object(self.mod.pickle, "load",
                                   side_effect=RuntimeError("bad pickle")):
                self.assertIsNone(self.mod._restore_login_from_cookie())


class RestoreAndFetchAttrInjectionTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_attr_setattr_reset_and_retry_failures_swallowed(self):
        # authed stays False; then setattr raises, reset() raises, and the
        # retry login() raises — all swallowed (1208-1218) before fetch.
        fake = _make_fake_alexapy()

        class _Login:
            email = "e"
            status = {}  # never login_successful

            async def login(self, data=None, cookies=None):
                # raise on the cookies-kwarg retry to hit 1217-1218.
                raise RuntimeError("login retry boom")

            @property
            def _cookies(self):
                return None

            @_cookies.setter
            def _cookies(self, v):
                raise RuntimeError("setattr boom")

            async def reset(self):
                raise RuntimeError("reset boom")

        login = _Login()
        with mock.patch.object(self.mod, "_alexapy", return_value=fake), \
             mock.patch.object(self.mod, "_load_cookie_pickle_safe",
                               return_value={"c": 1}), \
             mock.patch.object(self.mod, "_load_cookie_meta",
                               return_value={"email": "e"}), \
             mock.patch.object(self.mod, "_construct_login", return_value=login), \
             mock.patch.object(self.mod, "_fetch_devices_async",
                               new=mock.AsyncMock(return_value={"ok": 1})):
            out = run_coro(self.mod._restore_and_fetch_async())
        self.assertEqual(out, {"ok": 1})


class PlaywrightCookieCaptureExceptionTests(unittest.TestCase):
    """context.cookies() raising during capture must degrade to an empty list
    (covers 662-663 / 677-678 exception arms)."""

    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")
        self._sleep_patch = mock.patch.object(
            self.mod.asyncio, "sleep", new=mock.AsyncMock(return_value=None))
        self._sleep_patch.start()
        self.addCleanup(self._sleep_patch.stop)

    def _run(self, manager, timeout=5.0):
        with _install_fake_playwright(self.mod, manager):
            return run_coro(self.mod._login_via_playwright("e@e.com", timeout))

    def test_cookies_raise_on_url_left_branch(self):
        class _Ctx(_FakePwContext):
            async def cookies(self):
                raise RuntimeError("cookie read boom")
        ctx = _Ctx([])
        page = _FakePwPage(["https://www.amazon.com/gp/home"])
        ctx.attach_page(page)
        browser = _FakePwBrowser(ctx)
        chromium = _FakeChromium(browser)
        # cookies() raises → pw_cookies stays [] → returns None.
        out = self._run(_FakePwManager(chromium))
        self.assertIsNone(out)

    def test_cookies_raise_on_window_closed_branch(self):
        class _Ctx(_FakePwContext):
            async def cookies(self):
                raise RuntimeError("cookie read boom")
        ctx = _Ctx([])
        page = _FakePwPage(["https://www.amazon.com/ap/signin"], closed_after=0)
        ctx.attach_page(page)
        browser = _FakePwBrowser(ctx)
        chromium = _FakeChromium(browser)
        out = self._run(_FakePwManager(chromium))
        self.assertIsNone(out)

    def test_page_url_access_raises_then_times_out(self):
        # page.url raising is swallowed (current_url=""), loop continues until
        # the deadline (covers 667-668).
        class _UrlBoomPage(_FakePwPage):
            @property
            def url(self):
                raise RuntimeError("url boom")
        ctx = _FakePwContext([{"name": "x", "value": "y"}])
        page = _UrlBoomPage(["x"])
        ctx.attach_page(page)
        browser = _FakePwBrowser(ctx)
        chromium = _FakeChromium(browser)
        out = self._run(_FakePwManager(chromium), timeout=0.0)
        self.assertIsNone(out)


class WizardPlaywrightCookieSaveFailureTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")
        self.fake = _make_fake_alexapy()

    def test_cookie_save_failure_then_fetch_success(self):
        # Playwright path: _save_cookie_meta raises (1277-1278) but the wizard
        # presses on and the restore-fetch yields devices.
        devices = {"echo": [], "smarthome": [], "groups": []}
        jar = object()
        pw_cookies = [{"name": "a", "value": "1"}]

        def _run(coro):
            name = getattr(coro, "__qualname__", "")
            try:
                coro.close()
            except Exception:
                pass
            if "_login_async" in name:
                raise self.mod._LoginNeedsPlaywright()
            if "_login_via_playwright" in name:
                return (jar, pw_cookies)
            if "_restore_and_fetch_async" in name:
                return devices
            return None

        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=None), \
             mock.patch.object(self.mod, "_prompt_credentials",
                               return_value=("e@e.com", "pw")), \
             mock.patch.object(self.mod, "_run_async", side_effect=_run), \
             mock.patch.object(self.mod, "_save_cookie_meta",
                               side_effect=RuntimeError("save boom")), \
             mock.patch.object(self.mod, "_scan_lan_arp", return_value=[]), \
             mock.patch.object(self.mod, "_merge_with_existing_catalog",
                               side_effect=lambda c: c), \
             mock.patch.object(self.mod, "_save_catalog"), \
             mock.patch.object(self.mod, "_queue_missing_skill_tasks",
                               return_value=0):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("Catalog complete", out)

    def test_playwright_post_signin_fetch_raises_sets_devices_none(self):
        # _restore_and_fetch_async raises after playwright sign-in (1288-1290),
        # devices→None, used_playwright True → "device fetch" empty message.
        jar = object()
        pw_cookies = [{"name": "a", "value": "1"}]

        def _run(coro):
            name = getattr(coro, "__qualname__", "")
            try:
                coro.close()
            except Exception:
                pass
            if "_login_async" in name:
                raise self.mod._LoginNeedsPlaywright()
            if "_login_via_playwright" in name:
                return (jar, pw_cookies)
            if "_restore_and_fetch_async" in name:
                raise RuntimeError("post-signin fetch boom")
            return None

        with mock.patch.object(self.mod, "_alexapy", return_value=self.fake), \
             mock.patch.object(self.mod, "_load_cookie_meta", return_value=None), \
             mock.patch.object(self.mod, "_prompt_credentials",
                               return_value=("e@e.com", "pw")), \
             mock.patch.object(self.mod, "_run_async", side_effect=_run), \
             mock.patch.object(self.mod, "_save_cookie_meta"):
            out = self.mod._run_wizard_interactive("")
        self.assertIn("device fetch", out)


class CapabilityListOfStringsTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")

    def test_list_of_string_items(self):
        # A list whose ITEMS are strings (not dicts) hits the str-in-list arm.
        raw = ["Alexa.PowerController", "Alexa.WeirdThing"]
        self.assertEqual(self.mod._capability_tags(raw),
                         ["on_off", "weirdthing"])


class PlaywrightBrowserCloseAndUrlBranchTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("smart_home_discover")
        self._sleep_patch = mock.patch.object(
            self.mod.asyncio, "sleep", new=mock.AsyncMock(return_value=None))
        self._sleep_patch.start()
        self.addCleanup(self._sleep_patch.stop)

    def _run(self, manager, timeout=5.0):
        with _install_fake_playwright(self.mod, manager):
            return run_coro(self.mod._login_via_playwright("e@e.com", timeout))

    def test_browser_close_failure_swallowed(self):
        # Successful capture, but browser.close() raises in the finally
        # (684-685); the function still returns the captured cookies.
        cookies = [{"name": "session-id", "value": "v"}]
        ctx = _FakePwContext(cookies)
        page = _FakePwPage(["https://www.amazon.com/gp/home"])
        ctx.attach_page(page)

        class _Browser(_FakePwBrowser):
            async def close(self):
                raise RuntimeError("close boom")

        browser = _Browser(ctx)
        chromium = _FakeChromium(browser)
        result = self._run(_FakePwManager(chromium))
        self.assertIsNotNone(result)
        _jar, raw = result
        self.assertEqual(raw, cookies)

    def test_page_url_access_raises_then_window_closes(self):
        # First loop: page.url raises → current_url="" (667-668), sleep (no-op);
        # second loop: is_closed() True → grabs cookies → returns them.
        cookies = [{"name": "sess", "value": "v"}]

        class _UrlBoomPage(_FakePwPage):
            @property
            def url(self):
                raise RuntimeError("url boom")

        ctx = _FakePwContext(cookies)
        page = _UrlBoomPage(["x"], closed_after=1)  # closed on 2nd is_closed
        ctx.attach_page(page)
        browser = _FakePwBrowser(ctx)
        chromium = _FakeChromium(browser)
        result = self._run(_FakePwManager(chromium))
        self.assertIsNotNone(result)
        _jar, raw = result
        self.assertEqual(raw, cookies)


if __name__ == "__main__":
    unittest.main()
