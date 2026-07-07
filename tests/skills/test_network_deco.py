"""Logic tests for skills/network_deco.py (the genericised TP-Link Deco skill).

No real router / ARP / network / crypto: the dependency module, ``win32crypt``,
``subprocess``, threads and ``time.sleep`` are all faked so every branch runs
deterministically and offline. The skill reads env vars (DECO_HOST,
DECO_SUBNET_PREFIX, DECO_PASSWORD, BAMBU_PRINTER_IP) — those are patched via
``mock.patch.dict`` so nothing leaks between tests, and the module's global
``_state`` is reset in ``tearDown``.

Stdlib ``unittest`` + ``unittest.mock`` only (no pytest); App-Control-safe.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import (
    SKILLS_DIR,
    load_skill_isolated,
    make_fake_skill_utils,
)


# ──────────────────────────────────────────────────────────────────────────
# Fakes for the optional `tplinkrouterc6u` dependency.
# ──────────────────────────────────────────────────────────────────────────
class _FakeDevice:
    """Stand-in for a tplinkrouterc6u device object exposing assorted
    version-dependent attribute names that ``_serialise_device`` copies."""

    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class _FakeRouter:
    """Configurable fake router handle. Each capability is opt-in so individual
    tests can exercise the method-name fallthrough in ``_collect_via`` /
    ``_authorize`` / ``_set_guest_network``."""

    def __init__(self, devices=None, guest=None, parental=None,
                 raise_on=None, authorize=True):
        self._devices = devices
        self._guest = guest
        self._parental = parental
        self._raise_on = set(raise_on or ())
        self._authorize_ok = authorize
        self.logged_out = False
        self.guest_calls = []

    # auth
    def authorize(self):
        if "authorize" in self._raise_on:
            raise RuntimeError("auth boom")
        if not self._authorize_ok:
            raise RuntimeError("bad password")

    def logout(self):
        if "logout" in self._raise_on:
            raise RuntimeError("logout boom")
        self.logged_out = True

    # clients
    def get_devices(self):
        if "get_devices" in self._raise_on:
            raise RuntimeError("devices boom")
        return self._devices

    # guest
    def get_guest_wifi(self):
        if "get_guest_wifi" in self._raise_on:
            raise RuntimeError("guest boom")
        return self._guest

    # parental
    def get_parental_control(self):
        if "get_parental_control" in self._raise_on:
            raise RuntimeError("parental boom")
        return self._parental

    # guest toggle
    def set_guest_wifi(self, enable):
        self.guest_calls.append(("set_guest_wifi", enable))
        if "set_guest_wifi" in self._raise_on:
            raise RuntimeError("set boom")


def _fake_tplink_module(*, factory=None, classes=None):
    """Build a fake ``tplinkrouterc6u`` module. ``factory`` (if given) is set as
    ``TplinkRouterProvider`` with a ``get_client`` classmethod; ``classes`` maps
    extra attribute names to callables/classes."""
    mod = types.ModuleType("tplinkrouterc6u")
    if factory is not None:
        mod.TplinkRouterProvider = factory
    for name, obj in (classes or {}).items():
        setattr(mod, name, obj)
    return mod


# ──────────────────────────────────────────────────────────────────────────
# Loading the module with controlled env / dependency presence.
# ──────────────────────────────────────────────────────────────────────────
def _load_with_env(env):
    """Re-exec skills/network_deco.py with ``os.environ`` patched to ``env``
    (cleared first) so the import-time constants (DECO_HOST_DEFAULT,
    DECO_SUBNET_PREFIX, _PRINTER_IPS) are recomputed. Threads are neutered and
    stdout captured. Returns (module, actions)."""
    path = os.path.join(SKILLS_DIR, "network_deco.py")
    spec = importlib.util.spec_from_file_location("skill_network_deco", path)
    mod = importlib.util.module_from_spec(spec)
    mod.skill_utils = make_fake_skill_utils()
    actions: dict = {}
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(threading.Thread, "start", lambda self: None), \
            contextlib.redirect_stdout(io.StringIO()):
        sys.modules["skill_network_deco"] = mod
        spec.loader.exec_module(mod)
        if hasattr(mod, "register"):
            mod.register(actions)
    return mod, actions


# ──────────────────────────────────────────────────────────────────────────
# Base test case: loads the skill in isolation and guarantees state reset.
# ──────────────────────────────────────────────────────────────────────────
class _DecoBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("network_deco")
        self._reset_state()
        self.addCleanup(self._reset_state)

    def _reset_state(self):
        m = getattr(self, "mod", None)
        if m is None:
            return
        m._state.update({
            "snapshot": None,
            "snapshot_at": 0.0,
            "router_handle": None,
            "router_host": None,
            "router_class": None,
            "auth_ok": False,
            "last_error": None,
            "missing_dep": False,
        })
        try:
            m._stop_evt.set()
            m._poll_thread.clear()
        except Exception:
            pass

    # convenience: inject a snapshot into _state
    def _set_snapshot(self, snap):
        with self.mod._lock:
            self.mod._state["snapshot"] = snap
            self.mod._state["snapshot_at"] = time.time()


# ──────────────────────────────────────────────────────────────────────────
# 1. Existing contract tests (kept, must still pass).
# ──────────────────────────────────────────────────────────────────────────
class NetworkDecoTests(_DecoBase):
    def test_subnet_prefix_is_derived_from_host(self):
        m = self.mod
        self.assertEqual(m.DECO_SUBNET_PREFIX,
                         m.DECO_HOST_DEFAULT.rsplit(".", 1)[0] + ".")
        self.assertTrue(m.DECO_SUBNET_PREFIX.endswith("."))

    def test_printer_ips_is_a_set(self):
        self.assertIsInstance(self.mod._PRINTER_IPS, set)

    def test_registers_core_actions(self):
        for a in ("who_is_on_wifi", "is_printer_online", "deco_status"):
            self.assertIn(a, self.actions)
        self.assertGreater(len(self.actions), 8)

    def test_actions_degrade_gracefully_without_router(self):
        with mock.patch.object(self.mod, "is_available", return_value=False), \
                mock.patch.object(self.mod, "_current_snapshot", return_value=None), \
                mock.patch.object(self.mod, "_refresh_snapshot", return_value=None), \
                mock.patch.object(self.mod, "_host", return_value="192.168.1.1"), \
                mock.patch.object(self.mod, "_arp_table", return_value=[]):
            for a in ("who_is_on_wifi", "is_printer_online", "deco_status",
                      "network_usage", "deco_topology"):
                out = self.actions[a]("")
                self.assertIsInstance(out, str, a)
                self.assertTrue(out.strip(), f"{a} returned empty")


# ──────────────────────────────────────────────────────────────────────────
# 2. Import-time env-driven constants.
# ──────────────────────────────────────────────────────────────────────────
class EnvResolutionTests(unittest.TestCase):
    def test_host_default_from_env(self):
        mod, _ = _load_with_env({"DECO_HOST": "10.0.0.5"})
        self.assertEqual(mod.DECO_HOST_DEFAULT, "10.0.0.5")
        self.assertEqual(mod.DECO_SUBNET_PREFIX, "10.0.0.")

    def test_host_default_falls_back_when_env_absent(self):
        mod, _ = _load_with_env({})
        self.assertEqual(mod.DECO_HOST_DEFAULT, "192.168.1.1")
        self.assertEqual(mod.DECO_SUBNET_PREFIX, "192.168.1.")

    def test_explicit_subnet_prefix_overrides(self):
        mod, _ = _load_with_env({"DECO_HOST": "10.0.0.5",
                                 "DECO_SUBNET_PREFIX": "172.16.4."})
        self.assertEqual(mod.DECO_SUBNET_PREFIX, "172.16.4.")

    def test_printer_ips_populated_from_env(self):
        mod, _ = _load_with_env({"BAMBU_PRINTER_IP": "192.168.1.65"})
        self.assertEqual(mod._PRINTER_IPS, {"192.168.1.65"})

    def test_printer_ips_empty_when_blank(self):
        mod, _ = _load_with_env({"BAMBU_PRINTER_IP": "   "})
        self.assertEqual(mod._PRINTER_IPS, set())


# ──────────────────────────────────────────────────────────────────────────
# 3. Pure helpers: _json_sanitize.
# ──────────────────────────────────────────────────────────────────────────
class JsonSanitizeTests(_DecoBase):
    def test_scalars_passthrough(self):
        js = self.mod._json_sanitize
        self.assertEqual(js("x"), "x")
        self.assertEqual(js(3), 3)
        self.assertEqual(js(1.5), 1.5)
        self.assertEqual(js(True), True)
        self.assertIsNone(js(None))

    def test_nested_dict_and_list(self):
        js = self.mod._json_sanitize
        out = js({"a": [1, 2, {"b": "c"}], 7: "intkey"})
        self.assertEqual(out["a"], [1, 2, {"b": "c"}])
        self.assertIn("7", out)  # non-str key coerced

    def test_set_and_tuple_become_list(self):
        out = self.mod._json_sanitize((1, 2))
        self.assertEqual(out, [1, 2])
        self.assertIsInstance(self.mod._json_sanitize({1, 2}), list)

    def test_non_scalar_object_coerced_to_str(self):
        class Weird:
            def __str__(self):
                return "WEIRD"
        self.assertEqual(self.mod._json_sanitize(Weird()), "WEIRD")

    def test_circular_reference_handled(self):
        d: dict = {}
        d["self"] = d
        out = self.mod._json_sanitize(d)
        self.assertIn("circular", str(out["self"]))

    def test_unserialisable_str_falls_back(self):
        class Boom:
            def __str__(self):
                raise ValueError("nope")
        out = self.mod._json_sanitize(Boom())
        self.assertIn("unserialisable", out)


# ──────────────────────────────────────────────────────────────────────────
# 4. Pure helpers: device serialisation / topology / formatting.
# ──────────────────────────────────────────────────────────────────────────
class SerialiseDeviceTests(_DecoBase):
    def test_attrs_mapped_and_mac_normalised(self):
        d = _FakeDevice(hostname="Phone", ipaddr="192.168.1.20",
                        macaddr="00-11-22-aa-bb-cc", signal=-50,
                        connection="wireless", online=True)
        out = self.mod._serialise_device(d)
        self.assertEqual(out["name"], "Phone")
        self.assertEqual(out["ip"], "192.168.1.20")
        self.assertEqual(out["mac"], "00:11:22:AA:BB:CC")
        self.assertEqual(out["signal"], -50)
        self.assertEqual(out["connection"], "wireless")
        self.assertTrue(out["online"])

    def test_dict_input_merged(self):
        out = self.mod._serialise_device({"custom": "kept", "mac": "aa-bb"})
        self.assertEqual(out["custom"], "kept")
        # dict-provided mac is normalised too
        self.assertEqual(out["mac"], "AA:BB")

    def test_first_present_attr_wins(self):
        # hostname precedes name/alias in the mapping; once 'name' is set it's
        # not overwritten by a later attribute.
        d = _FakeDevice(hostname="H", name="N", alias="A")
        out = self.mod._serialise_device(d)
        self.assertEqual(out["name"], "H")

    def test_non_str_mac_left_alone(self):
        out = self.mod._serialise_device(_FakeDevice(mac=12345))
        self.assertEqual(out["mac"], 12345)


class SummariseTopologyTests(_DecoBase):
    def test_counts_and_node_discovery(self):
        devices = [
            {"mac": "50:C7:BF:00:00:01", "ip": "192.168.1.1",
             "name": "deco-main", "online": True, "connection": "wired"},
            {"mac": "AA:BB:CC:DD:EE:FF", "online": True, "connection": "wireless"},
            {"mac": "11:22:33:44:55:66", "online": False, "connection": "2.4ghz"},
        ]
        topo = self.mod._summarise_topology(devices)
        self.assertEqual(topo["clients_total"], 3)
        self.assertEqual(topo["online"], 2)
        self.assertEqual(topo["offline"], 1)
        self.assertEqual(topo["wireless"], 2)
        self.assertEqual(topo["wired"], 1)
        self.assertEqual(len(topo["deco_nodes"]), 1)
        self.assertEqual(topo["deco_nodes"][0]["name"], "deco-main")

    def test_node_without_name_gets_default(self):
        topo = self.mod._summarise_topology(
            [{"mac": "50:C7:BF:11:22:33"}])
        self.assertEqual(topo["deco_nodes"][0]["name"], "deco-node")

    def test_empty_list(self):
        topo = self.mod._summarise_topology([])
        self.assertEqual(topo["clients_total"], 0)
        self.assertEqual(topo["deco_nodes"], [])


class FmtBytesTests(_DecoBase):
    def test_units(self):
        f = self.mod._fmt_bytes
        self.assertEqual(f(512), "512 B")
        self.assertEqual(f(1536), "1.5 KB")
        self.assertEqual(f(1024 * 1024 * 3), "3.0 MB")
        self.assertTrue(f(1024 ** 5 * 2).endswith("PB"))

    def test_unparseable(self):
        self.assertEqual(self.mod._fmt_bytes("nan-ish"), "?")


class DeviceNameAndOnlineTests(_DecoBase):
    def test_device_name_priority(self):
        n = self.mod._device_name
        self.assertEqual(n({"name": "A", "ip": "1.2.3.4"}), "A")
        self.assertEqual(n({"hostname": "H"}), "H")
        self.assertEqual(n({"ip": "1.2.3.4"}), "1.2.3.4")
        self.assertEqual(n({"mac": "AA:BB"}), "AA:BB")
        # 2026-07-07 bug-hunt: the fully-unlabelled fallback must be "an unnamed
        # device", NOT the literal "unknown" — the old value collided with the
        # "unknown " FAILURE_MARKER, so a legit "<name> is online, sir." success
        # was read as a failure and swallowed.
        self.assertEqual(n({}), "an unnamed device")

    def test_unlabelled_device_name_carries_no_failure_marker(self):
        # Guard the fix directly: the fallback + an "is online" success line must
        # NOT trip the shared failure classifier.
        from core.dispatcher import _is_failure_result
        name = self.mod._device_name({})
        self.assertFalse(_is_failure_result(name),
                         f"unlabelled device name must not read as a failure: {name!r}")
        self.assertFalse(
            _is_failure_result(f"{name} is online, sir."),
            "an 'online' success for an unlabelled device must not read as a failure")

    def test_is_online_variants(self):
        io_ = self.mod._is_online
        self.assertTrue(io_({}))            # missing -> assumed online
        self.assertTrue(io_({"online": True}))
        self.assertFalse(io_({"online": False}))
        self.assertFalse(io_({"online": 0}))
        self.assertFalse(io_({"online": "0"}))
        self.assertFalse(io_({"online": "false"}))
        self.assertFalse(io_({"online": "False"}))
        self.assertTrue(io_({"online": "yes"}))


# ──────────────────────────────────────────────────────────────────────────
# 5. ARP parsing & host verification.
# ──────────────────────────────────────────────────────────────────────────
class ArpTableTests(_DecoBase):
    def test_parses_arp_output(self):
        raw = (b"Interface: 192.168.1.10 --- 0x5\n"
               b"  Internet Address      Physical Address      Type\n"
               b"  192.168.1.1           50-c7-bf-00-00-01     dynamic\n"
               b"  192.168.1.20          aa-bb-cc-dd-ee-ff     dynamic\n")
        with mock.patch.object(self.mod.subprocess, "check_output",
                               return_value=raw):
            arp = self.mod._arp_table()
        ips = {e["ip"] for e in arp}
        self.assertIn("192.168.1.1", ips)
        self.assertIn("192.168.1.20", ips)
        first = [e for e in arp if e["ip"] == "192.168.1.1"][0]
        self.assertEqual(first["oui"], "50C7BF")
        self.assertEqual(first["mac"], "50-C7-BF-00-00-01")

    def test_line_with_empty_oui_skipped(self):
        # A MAC field of only separators matches _ARP_LINE (>=11 chars of the
        # [0-9A-Fa-f-:] class) but normalises to an empty OUI -> that row is
        # skipped while a valid row on the same output is still returned.
        raw = (b"  192.168.1.9             :::::::::::::::::     dynamic\n"
               b"  192.168.1.1           50-c7-bf-00-00-01     dynamic\n")
        with mock.patch.object(self.mod.subprocess, "check_output",
                               return_value=raw):
            arp = self.mod._arp_table()
        ips = {e["ip"] for e in arp}
        self.assertIn("192.168.1.1", ips)
        self.assertNotIn("192.168.1.9", ips)

    def test_subprocess_failure_returns_empty(self):
        with mock.patch.object(self.mod.subprocess, "check_output",
                               side_effect=OSError("no arp")):
            self.assertEqual(self.mod._arp_table(), [])

    def test_cp1252_fallback_decode(self):
        # Bytes invalid as utf-8 but valid cp1252 (0xE9 = é in an iface name).
        raw = (b"Interface: \xe9th0\n"
               b"  192.168.1.1           50-c7-bf-00-00-01     dynamic\n")
        with mock.patch.object(self.mod.subprocess, "check_output",
                               return_value=raw):
            arp = self.mod._arp_table()
        self.assertTrue(any(e["ip"] == "192.168.1.1" for e in arp))

    def test_undecodable_returns_empty(self):
        # 0x81 is a lone continuation byte (invalid UTF-8) and is one of the
        # few bytes UNDEFINED in cp1252, so BOTH decodes raise and the helper
        # returns []. No mocking of the (immutable) bytes.decode required.
        raw = b"\x81\x81 garbage"
        with self.assertRaises(UnicodeDecodeError):
            raw.decode("utf-8")
        with self.assertRaises(UnicodeDecodeError):
            raw.decode("cp1252")
        with mock.patch.object(self.mod.subprocess, "check_output",
                               return_value=raw):
            self.assertEqual(self.mod._arp_table(), [])


class VerifyDecoHostTests(_DecoBase):
    def test_host_matches_deco_oui(self):
        arp = [{"ip": "192.168.1.1", "mac": "X", "oui": "50C7BF", "kind": "d"}]
        with mock.patch.object(self.mod, "_arp_table", return_value=arp):
            self.assertEqual(self.mod._verify_deco_host("192.168.1.1"),
                             "192.168.1.1")

    def test_falls_back_to_other_deco_on_subnet(self):
        arp = [
            {"ip": "192.168.1.50", "mac": "X", "oui": "B0BE76", "kind": "d"},
        ]
        with mock.patch.object(self.mod, "_arp_table", return_value=arp), \
                mock.patch.object(self.mod, "DECO_SUBNET_PREFIX", "192.168.1."):
            self.assertEqual(self.mod._verify_deco_host("192.168.1.99"),
                             "192.168.1.50")

    def test_no_match_returns_original(self):
        arp = [{"ip": "10.0.0.5", "mac": "X", "oui": "DEADBE", "kind": "d"}]
        with mock.patch.object(self.mod, "_arp_table", return_value=arp):
            self.assertEqual(self.mod._verify_deco_host("192.168.1.1"),
                             "192.168.1.1")


# ──────────────────────────────────────────────────────────────────────────
# 6. Config file + password + host resolution (+ DPAPI).
# ──────────────────────────────────────────────────────────────────────────
class ConfigIOTests(_DecoBase):
    def test_read_config_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._read_config(), {})

    def test_read_config_parses_json(self):
        data = '{"host": "192.168.1.7", "password": "pw"}'
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
                mock.patch("builtins.open",
                           mock.mock_open(read_data=data)):
            cfg = self.mod._read_config()
        self.assertEqual(cfg["host"], "192.168.1.7")

    def test_read_config_bad_json_returns_empty(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
                mock.patch("builtins.open",
                           mock.mock_open(read_data="{not json")):
            self.assertEqual(self.mod._read_config(), {})

    def test_write_config_calls_atomic_write(self):
        with mock.patch.object(self.mod.os, "makedirs") as mk, \
                mock.patch.object(self.mod, "_atomic_write_json") as aw:
            self.mod._write_config({"host": "h"})
        mk.assert_called_once()
        aw.assert_called_once()


class HostResolutionTests(_DecoBase):
    def test_env_wins(self):
        with mock.patch.dict(os.environ, {"DECO_HOST": "10.1.1.1"},
                             clear=False), \
                mock.patch.object(self.mod, "_read_config",
                                  return_value={"host": "5.5.5.5"}):
            self.assertEqual(self.mod._host(), "10.1.1.1")

    def test_config_used_when_env_absent(self):
        env = {k: v for k, v in os.environ.items() if k != "DECO_HOST"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(self.mod, "_read_config",
                                  return_value={"host": "5.5.5.5"}):
            self.assertEqual(self.mod._host(), "5.5.5.5")

    def test_default_when_nothing_set(self):
        env = {k: v for k, v in os.environ.items() if k != "DECO_HOST"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(self.mod, "_read_config", return_value={}):
            self.assertEqual(self.mod._host(), self.mod.DECO_HOST_DEFAULT)


class PasswordTests(_DecoBase):
    def test_env_password_wins(self):
        with mock.patch.dict(os.environ, {"DECO_PASSWORD": "envpw"},
                             clear=False):
            self.assertEqual(self.mod._password(), "envpw")

    def test_dpapi_encrypted_password(self):
        env = {k: v for k, v in os.environ.items() if k != "DECO_PASSWORD"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(self.mod, "_read_config",
                                  return_value={"password_dpapi": "CIPHER"}), \
                mock.patch.object(self.mod, "_dpapi_decrypt",
                                  return_value="secret") as dec:
            self.assertEqual(self.mod._password(), "secret")
        dec.assert_called_once_with("CIPHER")

    def test_dpapi_undecryptable_falls_through_to_plain(self):
        env = {k: v for k, v in os.environ.items() if k != "DECO_PASSWORD"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(
                    self.mod, "_read_config",
                    return_value={"password_dpapi": "BAD", "password": "plain"}), \
                mock.patch.object(self.mod, "_dpapi_decrypt", return_value=None), \
                mock.patch.object(self.mod, "_dpapi_encrypt", return_value=None):
            # migration encrypt also fails -> plaintext returned unchanged
            self.assertEqual(self.mod._password(), "plain")

    def test_plaintext_migrates_to_dpapi(self):
        env = {k: v for k, v in os.environ.items() if k != "DECO_PASSWORD"}
        cfg = {"password": "plain"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(self.mod, "_read_config", return_value=cfg), \
                mock.patch.object(self.mod, "_dpapi_encrypt",
                                  return_value="ENC"), \
                mock.patch.object(self.mod, "_write_config") as wc:
            self.assertEqual(self.mod._password(), "plain")
        # migration persisted the encrypted form and dropped plaintext
        wc.assert_called_once()
        written = wc.call_args[0][0]
        self.assertEqual(written["password_dpapi"], "ENC")
        self.assertNotIn("password", written)

    def test_plaintext_migration_exception_swallowed(self):
        env = {k: v for k, v in os.environ.items() if k != "DECO_PASSWORD"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(self.mod, "_read_config",
                                  return_value={"password": "plain"}), \
                mock.patch.object(self.mod, "_dpapi_encrypt",
                                  side_effect=RuntimeError("dpapi down")):
            self.assertEqual(self.mod._password(), "plain")

    def test_no_password_anywhere(self):
        env = {k: v for k, v in os.environ.items() if k != "DECO_PASSWORD"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(self.mod, "_read_config", return_value={}):
            self.assertIsNone(self.mod._password())


class DpapiTests(_DecoBase):
    def test_encrypt_empty_returns_none(self):
        self.assertIsNone(self.mod._dpapi_encrypt(""))

    def test_decrypt_empty_returns_none(self):
        self.assertIsNone(self.mod._dpapi_decrypt(""))

    def test_encrypt_roundtrip_with_fake_win32(self):
        fake = types.ModuleType("win32crypt")
        fake.CryptProtectData = lambda data, *a, **k: b"BLOB:" + data
        fake.CryptUnprotectData = lambda blob, *a, **k: ("desc",
                                                         blob[len(b"BLOB:"):])
        with mock.patch.dict(sys.modules, {"win32crypt": fake}):
            enc = self.mod._dpapi_encrypt("hello")
            self.assertIsInstance(enc, str)
            dec = self.mod._dpapi_decrypt(enc)
        self.assertEqual(dec, "hello")

    def test_encrypt_failure_returns_none(self):
        fake = types.ModuleType("win32crypt")

        def boom(*a, **k):
            raise RuntimeError("crypt fail")
        fake.CryptProtectData = boom
        with mock.patch.dict(sys.modules, {"win32crypt": fake}):
            self.assertIsNone(self.mod._dpapi_encrypt("x"))

    def test_decrypt_failure_returns_none(self):
        fake = types.ModuleType("win32crypt")

        def boom(*a, **k):
            raise RuntimeError("crypt fail")
        fake.CryptUnprotectData = boom
        with mock.patch.dict(sys.modules, {"win32crypt": fake}):
            self.assertIsNone(self.mod._dpapi_decrypt("QUJD"))


# ──────────────────────────────────────────────────────────────────────────
# 7. Dependency import + availability.
# ──────────────────────────────────────────────────────────────────────────
class DependencyTests(_DecoBase):
    def test_tplink_returns_none_when_missing(self):
        # Force the import inside _tplink to fail.
        with mock.patch.dict(sys.modules, {"tplinkrouterc6u": None}):
            self.assertIsNone(self.mod._tplink())

    def test_tplink_returns_module_when_present(self):
        fake = _fake_tplink_module()
        with mock.patch.dict(sys.modules, {"tplinkrouterc6u": fake}):
            self.assertIs(self.mod._tplink(), fake)

    def test_is_available_false_without_dep(self):
        with mock.patch.object(self.mod, "_tplink", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_is_available_false_without_password(self):
        with mock.patch.object(self.mod, "_tplink",
                               return_value=_fake_tplink_module()), \
                mock.patch.object(self.mod, "_password", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_is_available_true_with_dep_and_password(self):
        with mock.patch.object(self.mod, "_tplink",
                               return_value=_fake_tplink_module()), \
                mock.patch.object(self.mod, "_password", return_value="pw"):
            self.assertTrue(self.mod.is_available())


# ──────────────────────────────────────────────────────────────────────────
# 8. Router construction + authorize + logout.
# ──────────────────────────────────────────────────────────────────────────
class MakeRouterTests(_DecoBase):
    def test_none_when_no_dep(self):
        with mock.patch.object(self.mod, "_tplink", return_value=None):
            self.assertEqual(self.mod._make_router("h", "p"), (None, None))

    def test_factory_provider_path(self):
        created = {}

        class Provider:
            @staticmethod
            def get_client(url, password):
                created["url"] = url
                created["pw"] = password
                return _FakeRouter()
        fake = _fake_tplink_module(factory=Provider)
        with mock.patch.object(self.mod, "_tplink", return_value=fake):
            router, name = self.mod._make_router("192.168.1.1", "secret")
        self.assertIsInstance(router, _FakeRouter)
        self.assertEqual(name, "Provider")
        self.assertEqual(created["url"], "http://192.168.1.1")
        self.assertEqual(created["pw"], "secret")

    def test_factory_failure_then_class_fallback(self):
        class Provider:
            @staticmethod
            def get_client(url, password):
                raise RuntimeError("provider down")

        class TPLinkDecoRouter:
            def __init__(self, url, password):
                self.url = url
        fake = _fake_tplink_module(
            factory=Provider, classes={"TPLinkDecoRouter": TPLinkDecoRouter})
        with mock.patch.object(self.mod, "_tplink", return_value=fake):
            router, name = self.mod._make_router("h", "p")
        self.assertIsInstance(router, TPLinkDecoRouter)
        self.assertEqual(name, "TPLinkDecoRouter")
        self.assertIn("provider down", self.mod._state["last_error"])

    def test_class_constructor_raises_tries_next(self):
        class TPLinkDecoRouter:
            def __init__(self, url, password):
                raise RuntimeError("deco ctor boom")

        class TplinkRouter:
            def __init__(self, url, password):
                self.ok = True
        fake = _fake_tplink_module(classes={
            "TPLinkDecoRouter": TPLinkDecoRouter,
            "TplinkRouter": TplinkRouter,
        })
        with mock.patch.object(self.mod, "_tplink", return_value=fake):
            router, name = self.mod._make_router("h", "p")
        self.assertEqual(name, "TplinkRouter")
        self.assertIn("deco ctor boom", self.mod._state["last_error"])

    def test_all_paths_fail_returns_none(self):
        fake = _fake_tplink_module()  # no factory, no classes
        with mock.patch.object(self.mod, "_tplink", return_value=fake):
            self.assertEqual(self.mod._make_router("h", "p"), (None, None))


class AuthorizeLogoutTests(_DecoBase):
    def test_authorize_success(self):
        self.assertTrue(self.mod._authorize(_FakeRouter()))

    def test_authorize_failure_records_error(self):
        self.assertFalse(self.mod._authorize(_FakeRouter(authorize=False)))
        self.assertIn("authorize()", self.mod._state["last_error"])

    def test_authorize_uses_login_fallback(self):
        class R:
            def login(self):
                self.logged = True
        r = R()
        self.assertTrue(self.mod._authorize(r))
        self.assertTrue(r.logged)

    def test_authorize_no_method_returns_false(self):
        self.assertFalse(self.mod._authorize(object()))

    def test_logout_calls_method(self):
        r = _FakeRouter()
        self.mod._logout(r)
        self.assertTrue(r.logged_out)

    def test_logout_swallows_errors(self):
        r = _FakeRouter(raise_on={"logout"})
        self.mod._logout(r)  # must not raise

    def test_logout_no_method_is_noop(self):
        self.mod._logout(object())  # must not raise


# ──────────────────────────────────────────────────────────────────────────
# 9. Snapshot collection (_collect_via).
# ──────────────────────────────────────────────────────────────────────────
class CollectViaTests(_DecoBase):
    def test_full_collection(self):
        devices = [
            _FakeDevice(hostname="Phone", ipaddr="192.168.1.20",
                        macaddr="aa:bb:cc:dd:ee:ff", online=True,
                        connection="wireless"),
            _FakeDevice(hostname="deco", macaddr="50:C7:BF:00:00:01",
                        online=True, connection="wired"),
        ]
        router = _FakeRouter(devices=devices,
                             guest={"enabled": True, "ssid": "Guest"},
                             parental=[{"name": "kid"}])
        self.mod._state["router_host"] = "192.168.1.1"
        self.mod._state["router_class"] = "TPLinkDecoRouter"
        snap = self.mod._collect_via(router)
        self.assertEqual(len(snap["devices"]), 2)
        self.assertEqual(snap["host"], "192.168.1.1")
        self.assertEqual(snap["class"], "TPLinkDecoRouter")
        self.assertEqual(snap["guest_network"], {"enabled": True, "ssid": "Guest"})
        self.assertEqual(snap["parental_profiles"], [{"name": "kid"}])
        self.assertEqual(snap["topology"]["clients_total"], 2)

    def test_guest_object_coerced(self):
        guest_obj = _FakeDevice(enable=False, ssid="G")
        router = _FakeRouter(devices=[], guest=guest_obj)
        snap = self.mod._collect_via(router)
        self.assertEqual(snap["guest_network"]["enabled"], False)
        self.assertEqual(snap["guest_network"]["ssid"], "G")

    def test_parental_single_wrapped_in_list(self):
        router = _FakeRouter(devices=[], parental={"name": "solo"})
        snap = self.mod._collect_via(router)
        self.assertEqual(snap["parental_profiles"], [{"name": "solo"}])

    def test_errors_accumulated_per_endpoint(self):
        router = _FakeRouter(
            devices=[_FakeDevice(name="x")],
            raise_on={"get_guest_wifi", "get_parental_control"})
        snap = self.mod._collect_via(router)
        self.assertIn("errors", snap)
        joined = " ".join(snap["errors"])
        self.assertIn("guest boom", joined)
        self.assertIn("parental boom", joined)

    def test_devices_via_status_fallback(self):
        # No get_devices/get_clients; only get_status carrying .devices.
        class StatusObj:
            devices = [_FakeDevice(name="fromstatus", online=True)]

        class R:
            def get_status(self):
                return StatusObj()
        snap = self.mod._collect_via(R())
        self.assertEqual(len(snap["devices"]), 1)
        self.assertEqual(snap["devices"][0]["name"], "fromstatus")

    def test_get_devices_exception_then_status(self):
        class StatusObj:
            clients = [_FakeDevice(name="viaclients")]

        class R:
            def get_devices(self):
                raise RuntimeError("dev fail")

            def get_status(self):
                return StatusObj()
        snap = self.mod._collect_via(R())
        self.assertEqual(snap["devices"][0]["name"], "viaclients")
        self.assertIn("get_devices: dev fail", " ".join(snap["errors"]))

    def test_status_fallback_exception_recorded(self):
        class R:
            def get_status(self):
                raise RuntimeError("status fail")
        snap = self.mod._collect_via(R())
        self.assertEqual(snap["devices"], [])
        self.assertIn("get_status: status fail", " ".join(snap["errors"]))

    def test_non_list_devices_coerced(self):
        # get_devices returns a tuple -> list(r)
        class R:
            def get_devices(self):
                return (_FakeDevice(name="t1"), _FakeDevice(name="t2"))
        snap = self.mod._collect_via(R())
        self.assertEqual(len(snap["devices"]), 2)


# ──────────────────────────────────────────────────────────────────────────
# 10. _refresh_snapshot branches.
# ──────────────────────────────────────────────────────────────────────────
class RefreshSnapshotTests(_DecoBase):
    def test_no_password(self):
        with mock.patch.object(self.mod, "_password", return_value=None):
            self.assertIsNone(self.mod._refresh_snapshot())
        self.assertEqual(self.mod._state["last_error"], "no DECO_PASSWORD set")

    def test_missing_dep(self):
        with mock.patch.object(self.mod, "_password", return_value="pw"), \
                mock.patch.object(self.mod, "_tplink", return_value=None):
            self.assertIsNone(self.mod._refresh_snapshot())
        self.assertTrue(self.mod._state["missing_dep"])
        self.assertIn("tplinkrouterc6u not installed",
                      self.mod._state["last_error"])

    def test_make_router_fails(self):
        with mock.patch.object(self.mod, "_password", return_value="pw"), \
                mock.patch.object(self.mod, "_tplink",
                                  return_value=_fake_tplink_module()), \
                mock.patch.object(self.mod, "_verify_deco_host",
                                  return_value="192.168.1.1"), \
                mock.patch.object(self.mod, "_make_router",
                                  return_value=(None, None)):
            self.assertIsNone(self.mod._refresh_snapshot())
        self.assertIn("could not construct router handle",
                      self.mod._state["last_error"])

    def test_authorize_fails(self):
        router = _FakeRouter(authorize=False)
        with mock.patch.object(self.mod, "_password", return_value="pw"), \
                mock.patch.object(self.mod, "_tplink",
                                  return_value=_fake_tplink_module()), \
                mock.patch.object(self.mod, "_verify_deco_host",
                                  return_value="192.168.1.1"), \
                mock.patch.object(self.mod, "_make_router",
                                  return_value=(router, "Cls")), \
                mock.patch.object(self.mod, "_authorize", return_value=False):
            self.assertIsNone(self.mod._refresh_snapshot())

    def test_success_persists_and_caches(self):
        router = _FakeRouter(devices=[_FakeDevice(name="x", online=True)])
        with mock.patch.object(self.mod, "_password", return_value="pw"), \
                mock.patch.object(self.mod, "_tplink",
                                  return_value=_fake_tplink_module()), \
                mock.patch.object(self.mod, "_verify_deco_host",
                                  return_value="192.168.1.1"), \
                mock.patch.object(self.mod, "_make_router",
                                  return_value=(router, "Cls")), \
                mock.patch.object(self.mod, "_atomic_write_json") as aw:
            snap = self.mod._refresh_snapshot()
        self.assertIsNotNone(snap)
        self.assertTrue(self.mod._state["auth_ok"])
        self.assertIs(self.mod._state["snapshot"], snap)
        aw.assert_called_once()

    def test_reuses_existing_handle_for_same_host(self):
        router = _FakeRouter(devices=[])
        self.mod._state["router_handle"] = router
        self.mod._state["router_host"] = "192.168.1.1"
        with mock.patch.object(self.mod, "_password", return_value="pw"), \
                mock.patch.object(self.mod, "_tplink",
                                  return_value=_fake_tplink_module()), \
                mock.patch.object(self.mod, "_make_router") as mk, \
                mock.patch.object(self.mod, "_atomic_write_json"):
            snap = self.mod._refresh_snapshot()
        self.assertIsNotNone(snap)
        mk.assert_not_called()  # existing handle reused

    def test_collect_exception_resets_handle(self):
        router = _FakeRouter(devices=[])
        self.mod._state["router_handle"] = router
        self.mod._state["router_host"] = "192.168.1.1"
        with mock.patch.object(self.mod, "_password", return_value="pw"), \
                mock.patch.object(self.mod, "_tplink",
                                  return_value=_fake_tplink_module()), \
                mock.patch.object(self.mod, "_collect_via",
                                  side_effect=RuntimeError("collect boom")):
            self.assertIsNone(self.mod._refresh_snapshot())
        self.assertFalse(self.mod._state["auth_ok"])
        self.assertIsNone(self.mod._state["router_handle"])
        self.assertIn("collect boom", self.mod._state["last_error"])

    def test_snapshot_write_failure_swallowed(self):
        router = _FakeRouter(devices=[])
        with mock.patch.object(self.mod, "_password", return_value="pw"), \
                mock.patch.object(self.mod, "_tplink",
                                  return_value=_fake_tplink_module()), \
                mock.patch.object(self.mod, "_verify_deco_host",
                                  return_value="192.168.1.1"), \
                mock.patch.object(self.mod, "_make_router",
                                  return_value=(router, "Cls")), \
                mock.patch.object(self.mod, "_atomic_write_json",
                                  side_effect=OSError("disk full")):
            snap = self.mod._refresh_snapshot()
        # snapshot still returned despite the write failing
        self.assertIsNotNone(snap)


# ──────────────────────────────────────────────────────────────────────────
# 11. Cached snapshot loaders.
# ──────────────────────────────────────────────────────────────────────────
class CachedSnapshotTests(_DecoBase):
    def test_load_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertIsNone(self.mod._load_cached_snapshot())

    def test_load_parses_json(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
                mock.patch("builtins.open",
                           mock.mock_open(read_data='{"devices": []}')):
            self.assertEqual(self.mod._load_cached_snapshot(), {"devices": []})

    def test_load_bad_json_returns_none(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
                mock.patch("builtins.open",
                           mock.mock_open(read_data="not json")):
            self.assertIsNone(self.mod._load_cached_snapshot())

    def test_current_prefers_in_memory(self):
        self._set_snapshot({"devices": [{"name": "live"}]})
        with mock.patch.object(self.mod, "_load_cached_snapshot") as lc:
            snap = self.mod._current_snapshot()
        self.assertEqual(snap["devices"][0]["name"], "live")
        lc.assert_not_called()

    def test_current_falls_back_to_disk(self):
        with mock.patch.object(self.mod, "_load_cached_snapshot",
                               return_value={"devices": [{"name": "disk"}]}):
            snap = self.mod._current_snapshot()
        self.assertEqual(snap["devices"][0]["name"], "disk")


# ──────────────────────────────────────────────────────────────────────────
# 12. Background poller / monitor.
# ──────────────────────────────────────────────────────────────────────────
class PollLoopTests(_DecoBase):
    def test_poll_loop_exits_on_initial_stop(self):
        # _stop_evt.wait returns True -> early return, no refresh.
        with mock.patch.object(self.mod._stop_evt, "wait", return_value=True), \
                mock.patch.object(self.mod, "_refresh_snapshot") as rs:
            self.mod._poll_loop()
        rs.assert_not_called()

    def test_poll_loop_runs_once_then_stops(self):
        # First wait (initial delay) False -> enter loop; is_set False once then
        # the post-refresh wait returns True to exit.
        waits = iter([False, True])
        is_sets = iter([False, True])
        with mock.patch.object(self.mod._stop_evt, "wait",
                               side_effect=lambda *_: next(waits)), \
                mock.patch.object(self.mod._stop_evt, "is_set",
                                  side_effect=lambda: next(is_sets)), \
                mock.patch.object(self.mod, "_refresh_snapshot") as rs:
            self.mod._poll_loop()
        rs.assert_called_once()

    def test_poll_loop_swallows_refresh_error(self):
        waits = iter([False, True])
        is_sets = iter([False, True])
        with mock.patch.object(self.mod._stop_evt, "wait",
                               side_effect=lambda *_: next(waits)), \
                mock.patch.object(self.mod._stop_evt, "is_set",
                                  side_effect=lambda: next(is_sets)), \
                mock.patch.object(self.mod, "_refresh_snapshot",
                                  side_effect=RuntimeError("poll boom")):
            self.mod._poll_loop()  # must not raise


class StartMonitorTests(_DecoBase):
    def test_already_running_returns_true(self):
        class AliveThread:
            def is_alive(self):
                return True
        self.mod._poll_thread.append(AliveThread())
        self.assertTrue(self.mod._start_monitor())

    def test_no_dep_disables(self):
        with mock.patch.object(self.mod, "_tplink", return_value=None):
            self.assertFalse(self.mod._start_monitor())

    def test_no_password_disables(self):
        with mock.patch.object(self.mod, "_tplink",
                               return_value=_fake_tplink_module()), \
                mock.patch.object(self.mod, "_password", return_value=None):
            self.assertFalse(self.mod._start_monitor())

    def test_starts_thread_when_ready(self):
        with mock.patch.object(self.mod, "_tplink",
                               return_value=_fake_tplink_module()), \
                mock.patch.object(self.mod, "_password", return_value="pw"), \
                mock.patch.object(self.mod, "_host", return_value="192.168.1.1"), \
                mock.patch.object(threading.Thread, "start",
                                  lambda self: None):
            self.assertTrue(self.mod._start_monitor())
        self.assertTrue(self.mod._poll_thread)


# ──────────────────────────────────────────────────────────────────────────
# 13. Registered actions — happy + degradation paths.
# ──────────────────────────────────────────────────────────────────────────
class WhoIsOnWifiActionTests(_DecoBase):
    def test_no_snapshot_degrades(self):
        with mock.patch.object(self.mod, "_current_snapshot", return_value=None), \
                mock.patch.object(self.mod, "_refresh_snapshot", return_value=None):
            self.mod._state["last_error"] = "boom"
            out = self.actions["who_is_on_wifi"]("")
        self.assertIn("can't reach the Deco", out)
        self.assertIn("boom", out)

    def test_nobody_online(self):
        self._set_snapshot({"devices": [{"name": "x", "online": False}]})
        out = self.actions["who_is_on_wifi"]("")
        self.assertIn("Nobody", out)

    def test_lists_clients(self):
        self._set_snapshot({"devices": [
            {"name": "Phone", "online": True},
            {"name": "Laptop", "online": True},
        ]})
        out = self.actions["who_is_on_wifi"]("")
        self.assertIn("Phone", out)
        self.assertIn("Laptop", out)
        self.assertIn("2 clients online", out)

    def test_many_clients_truncated(self):
        devs = [{"name": f"dev{i:02d}", "online": True} for i in range(15)]
        self._set_snapshot({"devices": devs})
        out = self.actions["who_is_on_wifi"]("")
        self.assertIn("more", out)
        self.assertIn("15 clients online", out)


class IsPrinterOnlineActionTests(_DecoBase):
    def test_no_snapshot(self):
        with mock.patch.object(self.mod, "_current_snapshot", return_value=None), \
                mock.patch.object(self.mod, "_refresh_snapshot", return_value=None):
            out = self.actions["is_printer_online"]("")
        self.assertIn("can't reach the Deco", out)

    def test_printer_found_online_by_name(self):
        self._set_snapshot({"devices": [
            {"name": "Bambu-X1C", "ip": "192.168.1.65", "online": True}]})
        out = self.actions["is_printer_online"]("")
        self.assertIn("online", out)
        self.assertIn("Bambu-X1C", out)

    def test_printer_found_offline(self):
        self._set_snapshot({"devices": [
            {"name": "epson-home", "ip": "192.168.1.30", "online": False}]})
        out = self.actions["is_printer_online"]("")
        self.assertIn("offline", out)

    def test_printer_found_by_ip(self):
        with mock.patch.object(self.mod, "_PRINTER_IPS", {"192.168.1.65"}):
            self._set_snapshot({"devices": [
                {"name": "mystery", "ip": "192.168.1.65", "online": True}]})
            out = self.actions["is_printer_online"]("")
        self.assertIn("online", out)

    def test_no_printer_in_list(self):
        self._set_snapshot({"devices": [{"name": "laptop", "online": True}]})
        out = self.actions["is_printer_online"]("")
        self.assertIn("don't see a printer", out)


class IsDeviceOnlineActionTests(_DecoBase):
    def test_empty_name_prompts(self):
        self.assertIn("Which device", self.actions["is_device_online"](""))

    def test_no_snapshot(self):
        with mock.patch.object(self.mod, "_current_snapshot", return_value=None), \
                mock.patch.object(self.mod, "_refresh_snapshot", return_value=None):
            out = self.actions["is_device_online"]("phone")
        self.assertIn("can't reach the Deco", out)

    def test_no_match(self):
        self._set_snapshot({"devices": [{"name": "laptop", "online": True}]})
        out = self.actions["is_device_online"]("toaster")
        self.assertIn("don't see anything matching", out)

    def test_match_online_by_name(self):
        self._set_snapshot({"devices": [
            {"name": "Phone", "ip": "192.168.1.20", "online": True}]})
        out = self.actions["is_device_online"]("phone")
        self.assertIn("is online", out)
        self.assertIn("192.168.1.20", out)

    def test_match_offline(self):
        self._set_snapshot({"devices": [
            {"name": "Phone", "ip": "192.168.1.20", "online": False}]})
        out = self.actions["is_device_online"]("phone")
        self.assertIn("offline", out)

    def test_match_by_ip_and_mac(self):
        self._set_snapshot({"devices": [
            {"name": "x", "ip": "192.168.1.50", "mac": "aa:bb:cc:dd:ee:ff",
             "online": True}]})
        self.assertIn("online", self.actions["is_device_online"]("192.168.1.50"))
        self.assertIn("online", self.actions["is_device_online"]("aa:bb"))


class NetworkUsageActionTests(_DecoBase):
    def test_no_snapshot(self):
        with mock.patch.object(self.mod, "_current_snapshot", return_value=None), \
                mock.patch.object(self.mod, "_refresh_snapshot", return_value=None):
            out = self.actions["network_usage"]("")
        self.assertIn("can't reach the Deco", out)

    def test_no_byte_totals(self):
        self._set_snapshot({"devices": [{"name": "x"}]})
        out = self.actions["network_usage"]("")
        self.assertIn("isn't reporting", out)

    def test_ranks_top_users(self):
        self._set_snapshot({"devices": [
            {"name": "Hog", "up_total": 1000, "down_total": 9000},
            {"name": "Light", "up_total": 10, "down_total": 10},
            {"name": "Idle", "up_total": 0, "down_total": 0},
        ]})
        out = self.actions["network_usage"]("")
        self.assertIn("Top bandwidth users", out)
        self.assertIn("Hog", out)
        self.assertLess(out.index("Hog"), out.index("Light"))
        self.assertNotIn("Idle", out)

    def test_falls_back_to_speeds_when_no_totals(self):
        # Real Deco firmware never sends up_total/down_total — only
        # up_speed/down_speed. The action must rank by speed instead of
        # giving the no-totals excuse.
        self._set_snapshot({"devices": [
            {"name": "Streamer", "up_speed": 100, "down_speed": 9000},
            {"name": "Browser", "up_speed": 5, "down_speed": 50},
            {"name": "Idle", "up_speed": 0, "down_speed": 0},
        ]})
        out = self.actions["network_usage"]("")
        self.assertIn("Top bandwidth users", out)
        self.assertIn("Streamer", out)
        self.assertLess(out.index("Streamer"), out.index("Browser"))
        self.assertNotIn("Idle", out)
        self.assertIn("/s", out)
        self.assertNotIn("isn't reporting", out)

    def test_totals_preferred_over_speeds(self):
        self._set_snapshot({"devices": [
            {"name": "Historic", "up_total": 100, "down_total": 900,
             "up_speed": 0, "down_speed": 0},
            {"name": "Fast", "up_speed": 9999, "down_speed": 9999},
        ]})
        out = self.actions["network_usage"]("")
        self.assertIn("Historic", out)
        self.assertNotIn("Fast", out)

    def test_unparseable_totals_treated_as_zero(self):
        self._set_snapshot({"devices": [
            {"name": "Bad", "up_total": "xx", "down_total": None}]})
        out = self.actions["network_usage"]("")
        # totals collapse to 0 -> "isn't reporting"
        self.assertIn("isn't reporting", out)


class GuestNetworkActionTests(_DecoBase):
    def test_no_password(self):
        with mock.patch.object(self.mod, "_password", return_value=None):
            out = self.actions["kick_guest_network"]("")
        self.assertIn("need the Deco password", out)

    def test_router_unreachable(self):
        with mock.patch.object(self.mod, "_password", return_value="pw"), \
                mock.patch.object(self.mod, "_refresh_snapshot",
                                  return_value=None):
            self.mod._state["router_handle"] = None
            out = self.actions["kick_guest_network"]("")
        self.assertIn("can't reach the Deco", out)

    def test_disable_success_kwargs(self):
        router = _FakeRouter()
        self.mod._state["router_handle"] = router
        with mock.patch.object(self.mod, "_password", return_value="pw"):
            out = self.actions["kick_guest_network"]("")
        self.assertIn("disabled", out)
        self.assertIn(("set_guest_wifi", False), router.guest_calls)

    def test_enable_success(self):
        router = _FakeRouter()
        self.mod._state["router_handle"] = router
        with mock.patch.object(self.mod, "_password", return_value="pw"):
            out = self.actions["enable_guest_network"]("")
        self.assertIn("enabled", out)

    def test_refresh_obtains_handle(self):
        router = _FakeRouter()

        def fake_refresh(*a, **k):
            self.mod._state["router_handle"] = router
            return {"devices": []}
        with mock.patch.object(self.mod, "_password", return_value="pw"), \
                mock.patch.object(self.mod, "_refresh_snapshot",
                                  side_effect=fake_refresh):
            self.mod._state["router_handle"] = None
            out = self.actions["enable_guest_network"]("")
        self.assertIn("enabled", out)

    def test_toggle_typeerror_tries_next_arg_form(self):
        # set_guest_wifi only accepts a positional bool -> kwargs raise
        # TypeError, positional tuple form succeeds.
        class R:
            def __init__(self):
                self.called = None

            def set_guest_wifi(self, value):
                self.called = value
        r = R()
        self.mod._state["router_handle"] = r
        with mock.patch.object(self.mod, "_password", return_value="pw"):
            out = self.actions["kick_guest_network"]("")
        self.assertIn("disabled", out)
        self.assertEqual(r.called, False)

    def test_toggle_unexposed(self):
        # Router with no set_guest_* method at all.
        self.mod._state["router_handle"] = object()
        with mock.patch.object(self.mod, "_password", return_value="pw"):
            out = self.actions["kick_guest_network"]("")
        self.assertIn("isn't exposed by this firmware", out)

    def test_toggle_non_typeerror_records_and_continues(self):
        # set call raises a non-TypeError -> recorded in last_error, ultimately
        # falls through to the "not exposed" message.
        class R:
            def set_guest_wifi(self, **kw):
                raise RuntimeError("api 500")
        self.mod._state["router_handle"] = R()
        with mock.patch.object(self.mod, "_password", return_value="pw"):
            out = self.actions["kick_guest_network"]("")
        self.assertIn("isn't exposed", out)
        self.assertIn("api 500", self.mod._state["last_error"])


class TopologyStatusRefreshActionTests(_DecoBase):
    def test_topology_no_snapshot(self):
        with mock.patch.object(self.mod, "_current_snapshot", return_value=None), \
                mock.patch.object(self.mod, "_refresh_snapshot", return_value=None):
            out = self.actions["deco_topology"]("")
        self.assertIn("can't reach the Deco", out)

    def test_topology_no_nodes(self):
        self._set_snapshot({"topology": {"clients_total": 5, "deco_nodes": []}})
        out = self.actions["deco_topology"]("")
        self.assertIn("didn't expose individual nodes", out)
        self.assertIn("5 clients", out)

    def test_topology_with_nodes(self):
        self._set_snapshot({"topology": {
            "clients_total": 10, "online": 8,
            "deco_nodes": [{"name": "main"}, {"name": "office"}]}})
        out = self.actions["deco_topology"]("")
        self.assertIn("2 Deco node(s)", out)
        self.assertIn("main", out)
        self.assertIn("office", out)

    def test_status_no_snapshot(self):
        with mock.patch.object(self.mod, "_current_snapshot", return_value=None), \
                mock.patch.object(self.mod, "_refresh_snapshot", return_value=None):
            self.mod._state["last_error"] = "down"
            out = self.actions["deco_status"]("")
        self.assertIn("Deco link is down", out)

    def test_status_reports_age(self):
        self._set_snapshot({
            "fetched_at": time.time() - 30,
            "topology": {"online": 3, "clients_total": 5}})
        out = self.actions["deco_status"]("")
        self.assertIn("Deco mesh nominal", out)
        self.assertIn("3 of 5", out)
        self.assertIn("old", out)

    def test_status_refreshes_when_no_cache(self):
        with mock.patch.object(self.mod, "_current_snapshot", return_value=None), \
                mock.patch.object(
                    self.mod, "_refresh_snapshot",
                    return_value={"fetched_at": time.time(),
                                  "topology": {"online": 1, "clients_total": 1}}):
            out = self.actions["deco_status"]("")
        self.assertIn("nominal", out)

    def test_refresh_failure(self):
        with mock.patch.object(self.mod, "_refresh_snapshot", return_value=None):
            self.mod._state["last_error"] = "no route"
            out = self.actions["deco_refresh"]("")
        self.assertIn("Refresh failed", out)
        self.assertIn("no route", out)

    def test_refresh_success(self):
        with mock.patch.object(
                self.mod, "_refresh_snapshot",
                return_value={"topology": {"online": 4, "clients_total": 6}}):
            out = self.actions["deco_refresh"]("")
        self.assertIn("refreshed", out)
        self.assertIn("4 of 6", out)


# ──────────────────────────────────────────────────────────────────────────
# 14. register() wiring + monitor kickoff.
# ──────────────────────────────────────────────────────────────────────────
class RegisterTests(_DecoBase):
    def test_all_aliases_present(self):
        expected = {
            "who_is_on_wifi", "who_is_on_the_wifi", "network_clients",
            "list_wifi_clients", "is_printer_online", "printer_online",
            "is_device_online", "device_online", "network_usage",
            "bandwidth_hogs", "whats_using_bandwidth", "kick_guest_network",
            "disable_guest_network", "enable_guest_network", "deco_topology",
            "network_topology", "deco_status", "deco_refresh", "refresh_network",
        }
        self.assertTrue(expected.issubset(set(self.actions)))

    def test_register_invokes_start_monitor(self):
        actions: dict = {}
        with mock.patch.object(self.mod, "_start_monitor") as sm:
            self.mod.register(actions)
        sm.assert_called_once()
        self.assertIn("who_is_on_wifi", actions)

    def test_aliases_share_callable(self):
        self.assertIs(self.actions["who_is_on_wifi"],
                      self.actions["network_clients"])
        self.assertIs(self.actions["kick_guest_network"],
                      self.actions["disable_guest_network"])


# ──────────────────────────────────────────────────────────────────────────
# 15. Import-time _atomic_write_json fallback (when core.atomic_io is absent).
# ──────────────────────────────────────────────────────────────────────────
class AtomicWriteFallbackTests(unittest.TestCase):
    """On a host where ``core.atomic_io`` can't be imported at module load, the
    skill defines its own local ``_atomic_write_json``. We re-exec the module
    with that import blocked to bind the fallback, then exercise it (the
    happy write AND the cleanup-on-error path)."""

    def _load_module_without_core_atomic_io(self):
        path = os.path.join(SKILLS_DIR, "network_deco.py")
        spec = importlib.util.spec_from_file_location(
            "skill_network_deco_nofallback", path)
        mod = importlib.util.module_from_spec(spec)
        mod.skill_utils = make_fake_skill_utils()
        # Block the import so the `except` branch defines the local fallback.
        blocked = {"core.atomic_io": None}
        with mock.patch.dict(sys.modules, blocked), \
                mock.patch.object(threading.Thread, "start", lambda self: None), \
                contextlib.redirect_stdout(io.StringIO()):
            sys.modules["skill_network_deco_nofallback"] = mod
            spec.loader.exec_module(mod)
        self.addCleanup(sys.modules.pop, "skill_network_deco_nofallback", None)
        return mod

    def test_fallback_is_defined_when_core_atomic_io_missing(self):
        mod = self._load_module_without_core_atomic_io()
        # The bound _atomic_write_json is the module-local fallback, not the
        # core helper (its __module__ is this re-exec'd module).
        self.assertEqual(mod._atomic_write_json.__module__,
                         "skill_network_deco_nofallback")

    def test_fallback_writes_json_atomically(self):
        import json as _json
        import tempfile
        mod = self._load_module_without_core_atomic_io()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "snap.json")
            mod._atomic_write_json(p, {"hello": "world", "n": 3})
            with open(p, encoding="utf-8") as f:
                data = _json.load(f)
        self.assertEqual(data, {"hello": "world", "n": 3})

    def test_fallback_tolerates_fsync_oserror(self):
        # On some filesystems fsync raises OSError; the fallback swallows it and
        # still completes the write (covers the inner except-pass at 89-90).
        import json as _json
        import tempfile
        mod = self._load_module_without_core_atomic_io()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "snap.json")
            with mock.patch.object(mod.os, "fsync",
                                   side_effect=OSError("fsync unsupported")):
                mod._atomic_write_json(p, {"k": "v"})
            with open(p, encoding="utf-8") as f:
                data = _json.load(f)
        self.assertEqual(data, {"k": "v"})

    def test_fallback_cleans_up_tmp_on_write_error(self):
        import tempfile
        mod = self._load_module_without_core_atomic_io()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "snap.json")
            # Make os.replace fail AFTER the tmp file is written so the except
            # branch runs its unlink + re-raise (covers 92-97).
            with mock.patch.object(mod.os, "replace",
                                   side_effect=OSError("replace denied")):
                with self.assertRaises(OSError):
                    mod._atomic_write_json(p, {"x": 1})
            # No stray *.tmp left behind in the directory.
            leftovers = [f for f in os.listdir(d) if f.endswith(".tmp")]
            self.assertEqual(leftovers, [])

    def test_fallback_unlink_failure_is_swallowed_but_reraises(self):
        import tempfile
        mod = self._load_module_without_core_atomic_io()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "snap.json")
            # Both the replace and the cleanup unlink fail → the inner
            # except-pass runs and the original error still propagates.
            with mock.patch.object(mod.os, "replace",
                                   side_effect=OSError("replace denied")), \
                 mock.patch.object(mod.os, "unlink",
                                   side_effect=OSError("unlink denied")):
                with self.assertRaises(OSError):
                    mod._atomic_write_json(p, {"x": 1})


if __name__ == "__main__":
    unittest.main()
