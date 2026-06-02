"""Logic tests for skills/sh_govee.py (Govee LAN + cloud controller).

Govee is unusual: is_available() is always True (LAN sockets are stdlib), so
graceful degradation here means set_state returns a clear error when a device
resolves via NEITHER the LAN scan NOR the cloud API. We also verify:
  * the LAN command payloads (turn / brightness / colorwc) are well-formed,
  * the colour-before-brightness ordering on the LAN path,
  * _api_key resolution from env / config,
  * govee_list messaging,
  * the real _lan_scan / _send_lan_cmd socket dance (sockets fully faked — no
    UDP leaves the box),
  * the cloud REST layer (devices list / state / control) with fake `requests`,
  * list_devices LAN+cloud merge, get_state cloud read, set_state cloud path.

No real network: the skill resolves `requests` lazily via `_requests()` (patched
to a fake) and uses `socket.socket` (we patch the module's `socket` attribute).
Module global `_state` is reset in tearDown so discovery caches never leak.
"""
from __future__ import annotations

import json
import os
import socket as _real_socket
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── fake UDP socket for the LAN scan / command paths ────────────────────
class FakeUDPSocket:
    """Stands in for socket.socket(AF_INET, SOCK_DGRAM). Records sends and
    replays a scripted list of recvfrom outcomes. Each ``recv_script`` entry is
    either ('data', bytes, addr) or ('raise', exc_type)."""

    def __init__(self, recv_script=None, bind_raises=False, send_raises=False,
                 close_raises=False):
        self._recv = list(recv_script or [])
        self._bind_raises = bind_raises
        self._send_raises = send_raises
        self._close_raises = close_raises
        self.sent: list = []
        self.closed = False
        self.opts: list = []
        self.timeout = None

    def setsockopt(self, *a):
        self.opts.append(a)

    def bind(self, addr):
        if self._bind_raises:
            raise OSError("bind boom")

    def settimeout(self, t):
        self.timeout = t

    def sendto(self, payload, addr):
        if self._send_raises:
            raise OSError("send boom")
        self.sent.append((payload, addr))

    def recvfrom(self, _bufsize):
        if not self._recv:
            raise _real_socket.timeout()
        kind, *rest = self._recv.pop(0)
        if kind == "raise":
            raise rest[0]()
        data, addr = rest
        return data, addr

    def close(self):
        self.closed = True
        if self._close_raises:
            raise OSError("close boom")


def make_fake_socket_module(scan_sock=None, cmd_sock=None):
    """A fake `socket` module exposing the constants the skill reads plus a
    socket() factory that hands back the scan socket first, then the command
    socket (so a test can drive a scan and a command in sequence)."""
    mod = types.ModuleType("socket")
    mod.AF_INET = _real_socket.AF_INET
    mod.SOCK_DGRAM = _real_socket.SOCK_DGRAM
    mod.SOL_SOCKET = _real_socket.SOL_SOCKET
    mod.SO_BROADCAST = _real_socket.SO_BROADCAST
    mod.IPPROTO_IP = _real_socket.IPPROTO_IP
    mod.IP_MULTICAST_TTL = _real_socket.IP_MULTICAST_TTL
    mod.timeout = _real_socket.timeout
    queue = [s for s in (scan_sock, cmd_sock) if s is not None]

    def _factory(*a, **k):
        if queue:
            return queue.pop(0)
        return FakeUDPSocket()

    mod.socket = _factory
    return mod


def make_fake_requests(*, get_resp=None, put_resp=None, get_raises=False,
                       put_raises=False):
    """Fake `requests` module. ``get_resp`` / ``put_resp`` are dicts the
    response .json() returns; set *_raises to simulate transport errors."""
    req = types.ModuleType("requests")

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    req.get = (mock.MagicMock(side_effect=RuntimeError("net down"))
               if get_raises else
               mock.MagicMock(return_value=_Resp(get_resp or {})))
    req.put = (mock.MagicMock(side_effect=RuntimeError("net down"))
               if put_raises else
               mock.MagicMock(return_value=_Resp(put_resp or {})))
    return req


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


# ─── shared base: reset _state discovery caches between tests ─────────────
class _GoveeBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_govee")
        self._saved_state = {k: (dict(v) if isinstance(v, dict)
                                 else list(v) if isinstance(v, list) else v)
                             for k, v in self.mod._state.items()}
        self.addCleanup(self._restore_state)

    def _restore_state(self):
        self.mod._state.clear()
        self.mod._state.update(self._saved_state)

    def _stale_caches(self):
        self.mod._state["lan_fetched"] = 0.0
        self.mod._state["cloud_fetched"] = 0.0
        self.mod._state["cloud_devices"] = []


class GoveeApiKeyTests(_GoveeBase):
    def test_api_key_from_config_file(self):
        m = mock.mock_open(read_data='{"api_key": "  cfgkey  "}')
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._api_key(), "cfgkey")

    def test_api_key_config_bad_json_returns_none(self):
        m = mock.mock_open(read_data="{not json")
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertIsNone(self.mod._api_key())

    def test_api_key_config_blank_returns_none(self):
        m = mock.mock_open(read_data='{"api_key": "   "}')
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertIsNone(self.mod._api_key())


class GoveeLanScanTests(_GoveeBase):
    def _scan_with(self, sock):
        fake_mod = make_fake_socket_module(scan_sock=sock)
        return mock.patch.object(self.mod, "socket", fake_mod)

    def test_lan_scan_collects_one_device(self):
        reply = json.dumps({"msg": {"data": {
            "ip": "10.0.0.7", "sku": "H6159", "device": "Strip",
            "bleVersionHard": "1.0"}}}).encode("utf-8")
        sock = FakeUDPSocket(recv_script=[
            ("data", reply, ("10.0.0.7", 4002)),
            ("raise", OSError),   # breaks the recv loop immediately after
        ])
        with self._scan_with(sock):
            found = self.mod._lan_scan(timeout=5.0)
        self.assertIn("10.0.0.7", found)
        self.assertEqual(found["10.0.0.7"]["sku"], "H6159")
        self.assertEqual(found["10.0.0.7"]["device"], "Strip")
        self.assertTrue(sock.closed)        # socket always closed
        self.assertTrue(sock.sent)          # scan datagram was sent

    def test_lan_scan_uses_addr_ip_when_payload_missing_ip(self):
        reply = json.dumps({"msg": {"data": {"sku": "H6159"}}}).encode("utf-8")
        sock = FakeUDPSocket(recv_script=[
            ("data", reply, ("10.0.0.9", 4002)),
            ("raise", OSError),
        ])
        with self._scan_with(sock):
            found = self.mod._lan_scan(timeout=5.0)
        self.assertIn("10.0.0.9", found)    # fell back to addr[0]

    def test_lan_scan_skips_bad_json_then_times_out(self):
        sock = FakeUDPSocket(recv_script=[
            ("data", b"{not json", ("10.0.0.7", 4002)),  # skipped
            ("raise", _real_socket.timeout),              # continue
            ("raise", OSError),                           # break
        ])
        with self._scan_with(sock):
            found = self.mod._lan_scan(timeout=5.0)
        self.assertEqual(found, {})

    def test_lan_scan_bind_failure_returns_empty(self):
        sock = FakeUDPSocket(bind_raises=True)
        with self._scan_with(sock):
            self.assertEqual(self.mod._lan_scan(), {})
        self.assertTrue(sock.closed)

    def test_lan_scan_send_failure_returns_empty(self):
        sock = FakeUDPSocket(send_raises=True)
        with self._scan_with(sock):
            self.assertEqual(self.mod._lan_scan(), {})
        self.assertTrue(sock.closed)

    def test_lan_scan_close_error_is_swallowed(self):
        # sock.close() raising in the finally block must not propagate.
        reply = json.dumps({"msg": {"data": {"ip": "10.0.0.7"}}}).encode("utf-8")
        sock = FakeUDPSocket(recv_script=[("data", reply, ("10.0.0.7", 4002)),
                                          ("raise", OSError)],
                             close_raises=True)
        with self._scan_with(sock):
            found = self.mod._lan_scan(timeout=5.0)
        self.assertIn("10.0.0.7", found)   # returned despite close() blowing up


class GoveeRefreshLanTests(_GoveeBase):
    def test_refresh_lan_caches(self):
        self._stale_caches()
        found = {"10.0.0.7": {"ip": "10.0.0.7", "sku": "H6159"}}
        with mock.patch.object(self.mod, "_lan_scan", return_value=found) as scan:
            out1 = self.mod._refresh_lan()
            # Second call within TTL → cached, no rescan.
            out2 = self.mod._refresh_lan()
        self.assertEqual(out1, found)
        self.assertEqual(out2, found)
        scan.assert_called_once()


class GoveeSendLanCmdTests(_GoveeBase):
    def test_send_lan_cmd_success(self):
        sock = FakeUDPSocket()
        fake_mod = make_fake_socket_module(scan_sock=sock)
        with mock.patch.object(self.mod, "socket", fake_mod):
            res = self.mod._send_lan_cmd("10.0.0.7", {"cmd": "turn",
                                                       "data": {"value": 1}})
        self.assertTrue(res["ok"])
        self.assertTrue(sock.closed)
        payload, addr = sock.sent[0]
        self.assertEqual(addr, ("10.0.0.7", self.mod._LAN_CMD_PORT))
        self.assertEqual(json.loads(payload.decode())["msg"]["cmd"], "turn")

    def test_send_lan_cmd_socket_error(self):
        sock = FakeUDPSocket(send_raises=True)
        fake_mod = make_fake_socket_module(scan_sock=sock)
        with mock.patch.object(self.mod, "socket", fake_mod):
            res = self.mod._send_lan_cmd("10.0.0.7", {"cmd": "turn"})
        self.assertIn("lan send failed", res["error"])


class GoveeRequestsImportTests(_GoveeBase):
    def test_requests_none_when_absent(self):
        real_import = __import__

        def _blocked(name, *a, **k):
            if name == "requests":
                raise ImportError("no requests")
            return real_import(name, *a, **k)

        import sys as _sys
        with mock.patch.dict(_sys.modules, {"requests": None}), \
             mock.patch("builtins.__import__", side_effect=_blocked):
            self.assertIsNone(self.mod._requests())

    def test_requests_present(self):
        import importlib.util as _u
        if _u.find_spec("requests") is None:
            self.skipTest("requests not installed")
        self.assertIsNotNone(self.mod._requests())


class GoveeCloudDevicesTests(_GoveeBase):
    def test_cloud_devices_fetches_and_caches(self):
        self._stale_caches()
        payload = {"data": {"devices": [
            {"device": "AA:BB", "model": "H6159", "deviceName": "Strip",
             "supportCmds": ["turn", "brightness", "color", "colorTem"]}]}}
        req = make_fake_requests(get_resp=payload)
        with mock.patch.object(self.mod, "_api_key", return_value="k"), \
             mock.patch.object(self.mod, "_requests", return_value=req):
            out1 = self.mod._cloud_devices()
            out2 = self.mod._cloud_devices()   # cached
        self.assertEqual(len(out1), 1)
        self.assertEqual(out1[0]["deviceName"], "Strip")
        self.assertEqual(out2, out1)           # cache returns the same data
        req.get.assert_called_once()           # second call served from cache
        # Header carried the API key.
        _a, kw = req.get.call_args
        self.assertEqual(kw["headers"]["Govee-API-Key"], "k")

    def test_cloud_devices_no_requests_module(self):
        self._stale_caches()
        with mock.patch.object(self.mod, "_api_key", return_value="k"), \
             mock.patch.object(self.mod, "_requests", return_value=None):
            self.assertEqual(self.mod._cloud_devices(), [])

    def test_cloud_devices_http_error_returns_empty(self):
        self._stale_caches()
        req = make_fake_requests(get_raises=True)
        with mock.patch.object(self.mod, "_api_key", return_value="k"), \
             mock.patch.object(self.mod, "_requests", return_value=req):
            self.assertEqual(self.mod._cloud_devices(), [])


class GoveeCloudMatchTests(_GoveeBase):
    def test_cloud_match_by_name(self):
        devs = [{"deviceName": "Strip", "device": "AA", "model": "H6159"}]
        with mock.patch.object(self.mod, "_cloud_devices", return_value=devs):
            m = self.mod._cloud_match({"name": "strip"})   # case-insensitive
        self.assertEqual(m["device"], "AA")

    def test_cloud_match_blank_name_none(self):
        self.assertIsNone(self.mod._cloud_match({"name": "  "}))

    def test_cloud_match_no_hit(self):
        with mock.patch.object(self.mod, "_cloud_devices", return_value=[]):
            self.assertIsNone(self.mod._cloud_match({"name": "Ghost"}))


class GoveeCloudControlTests(_GoveeBase):
    def test_cloud_control_put_success(self):
        req = make_fake_requests(put_resp={"code": 200})
        with mock.patch.object(self.mod, "_api_key", return_value="k"), \
             mock.patch.object(self.mod, "_requests", return_value=req):
            res = self.mod._cloud_control({"device": "AA", "model": "H6159"},
                                          "turn", "on")
        self.assertTrue(res["ok"])
        _a, kw = req.put.call_args
        body = kw["json"]
        self.assertEqual(body["device"], "AA")
        self.assertEqual(body["cmd"], {"name": "turn", "value": "on"})

    def test_cloud_control_no_requests(self):
        with mock.patch.object(self.mod, "_api_key", return_value="k"), \
             mock.patch.object(self.mod, "_requests", return_value=None):
            res = self.mod._cloud_control({"device": "AA", "model": "y"},
                                          "turn", "on")
        self.assertIn("requests not installed", res["error"])

    def test_cloud_control_http_error(self):
        req = make_fake_requests(put_raises=True)
        with mock.patch.object(self.mod, "_api_key", return_value="k"), \
             mock.patch.object(self.mod, "_requests", return_value=req):
            res = self.mod._cloud_control({"device": "AA", "model": "y"},
                                          "turn", "on")
        self.assertIn("cloud control failed", res["error"])


class GoveeListDevicesCloudTests(_GoveeBase):
    def test_list_devices_merges_cloud_caps(self):
        cloud = [{"deviceName": "Cloud Bulb", "model": "H6001",
                  "supportCmds": ["turn", "brightness", "color", "colorTem"]}]
        with mock.patch.object(self.mod, "_refresh_lan", return_value={}), \
             mock.patch.object(self.mod, "_cloud_devices", return_value=cloud):
            devs = self.mod.list_devices()
        self.assertEqual(len(devs), 1)
        d = devs[0]
        self.assertEqual(d["name"], "Cloud Bulb")
        for cap in ("on_off", "dim", "color", "color_temperature"):
            self.assertIn(cap, d["capabilities"])

    def test_list_devices_dedupes_lan_and_cloud_same_device(self):
        lan = {"10.0.0.7": {"ip": "10.0.0.7", "sku": "H6159", "device": "Strip"}}
        cloud = [{"deviceName": "Strip", "model": "H6159",
                  "supportCmds": ["turn"]}]
        with mock.patch.object(self.mod, "_refresh_lan", return_value=lan), \
             mock.patch.object(self.mod, "_cloud_devices", return_value=cloud):
            devs = self.mod.list_devices()
        # LAN + cloud refer to the same Strip/H6159 → not duplicated.
        self.assertEqual(len(devs), 1)
        self.assertEqual(devs[0]["lan_ip"], "10.0.0.7")

    def test_list_devices_cloud_no_supportcmds(self):
        cloud = [{"deviceName": "Basic", "model": "H1"}]  # no supportCmds
        with mock.patch.object(self.mod, "_refresh_lan", return_value={}), \
             mock.patch.object(self.mod, "_cloud_devices", return_value=cloud):
            devs = self.mod.list_devices()
        self.assertEqual(devs[0]["capabilities"], ["on_off"])


class GoveeGetStateTests(_GoveeBase):
    def test_get_state_unknown_when_no_cloud_match(self):
        with mock.patch.object(self.mod, "_cloud_match", return_value=None):
            st = self.mod.get_state({"name": "Strip"})
        self.assertEqual(st["on"], "unknown")
        self.assertIn("write-only", st["note"])

    def test_get_state_unknown_when_no_key(self):
        with mock.patch.object(self.mod, "_cloud_match",
                               return_value={"device": "AA", "model": "y"}), \
             mock.patch.object(self.mod, "_api_key", return_value=None):
            st = self.mod.get_state({"name": "Strip"})
        self.assertEqual(st["on"], "unknown")

    def test_get_state_reads_cloud_properties(self):
        payload = {"data": {"properties": [
            {"powerState": "on"}, {"brightness": 80}]}}
        req = make_fake_requests(get_resp=payload)
        with mock.patch.object(self.mod, "_cloud_match",
                               return_value={"device": "AA", "model": "H6159"}), \
             mock.patch.object(self.mod, "_api_key", return_value="k"), \
             mock.patch.object(self.mod, "_requests", return_value=req):
            st = self.mod.get_state({"name": "Strip"})
        self.assertEqual(st["powerState"], "on")
        self.assertEqual(st["brightness"], 80)

    def test_get_state_cloud_error(self):
        req = make_fake_requests(get_raises=True)
        with mock.patch.object(self.mod, "_cloud_match",
                               return_value={"device": "AA", "model": "H6159"}), \
             mock.patch.object(self.mod, "_api_key", return_value="k"), \
             mock.patch.object(self.mod, "_requests", return_value=req):
            st = self.mod.get_state({"name": "Strip"})
        self.assertIn("cloud state read failed", st["error"])


class GoveeLanColorTempTests(_GoveeBase):
    def test_lan_color_temperature_sent(self):
        sender = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "_send_lan_cmd", sender):
            res = self.mod.set_state({"name": "Strip", "lan_ip": "10.0.0.7"},
                                     color_temperature=4000)
        self.assertEqual(res["applied"]["color_temperature_k"], 4000)
        _ip, cmd = sender.call_args[0]
        self.assertEqual(cmd["cmd"], "colorwc")
        self.assertEqual(cmd["data"]["colorTemInKelvin"], 4000)

    def test_lan_color_temperature_error_propagates(self):
        with mock.patch.object(self.mod, "_send_lan_cmd",
                               return_value={"error": "lan send failed: x"}):
            res = self.mod.set_state({"name": "Strip", "lan_ip": "10.0.0.7"},
                                     color_temperature=4000)
        self.assertIn("lan send failed", res["error"])

    def test_lan_color_error_propagates(self):
        with mock.patch.object(self.mod, "_send_lan_cmd",
                               return_value={"error": "lan send failed: y"}):
            res = self.mod.set_state({"name": "Strip", "lan_ip": "10.0.0.7"},
                                     color=(1, 2, 3))
        self.assertIn("lan send failed", res["error"])

    def test_lan_brightness_error_propagates(self):
        # Brightness is the LAST LAN command; its error path returns early too.
        with mock.patch.object(self.mod, "_send_lan_cmd",
                               return_value={"error": "lan send failed: z"}):
            res = self.mod.set_state({"name": "Strip", "lan_ip": "10.0.0.7"},
                                     brightness=50)
        self.assertIn("lan send failed", res["error"])


class GoveeCloudSetStateTests(_GoveeBase):
    """Drive set_state down the CLOUD branch (no lan_ip) — each command routes
    through _cloud_control, which we stub to capture the calls."""

    def _cloud(self, ctl):
        match = {"device": "AA", "model": "H6159"}
        return (mock.patch.object(self.mod, "_cloud_match", return_value=match),
                mock.patch.object(self.mod, "_cloud_control", ctl))

    def test_cloud_turn_on(self):
        ctl = mock.Mock(return_value={"ok": True})
        m, c = self._cloud(ctl)
        with m, c:
            res = self.mod.set_state({"name": "Strip"}, on=True)
        self.assertEqual(res["path"], "cloud")
        self.assertTrue(res["applied"]["on"])
        _d, name, value = ctl.call_args[0]
        self.assertEqual((name, value), ("turn", "on"))

    def test_cloud_off(self):
        ctl = mock.Mock(return_value={"ok": True})
        m, c = self._cloud(ctl)
        with m, c:
            self.mod.set_state({"name": "Strip"}, on=False)
        _d, name, value = ctl.call_args[0]
        self.assertEqual(value, "off")

    def test_cloud_brightness_clamped(self):
        ctl = mock.Mock(return_value={"ok": True})
        m, c = self._cloud(ctl)
        with m, c:
            res = self.mod.set_state({"name": "Strip"}, brightness=250)
        self.assertEqual(res["applied"]["brightness"], 100)
        _d, name, value = ctl.call_args[0]
        self.assertEqual((name, value), ("brightness", 100))

    def test_cloud_color_payload(self):
        ctl = mock.Mock(return_value={"ok": True})
        m, c = self._cloud(ctl)
        with m, c:
            res = self.mod.set_state({"name": "Strip"}, color=(10, 20, 30))
        self.assertEqual(res["applied"]["color"], [10, 20, 30])
        _d, name, value = ctl.call_args[0]
        self.assertEqual(name, "color")
        self.assertEqual(value, {"r": 10, "g": 20, "b": 30})

    def test_cloud_color_temperature(self):
        ctl = mock.Mock(return_value={"ok": True})
        m, c = self._cloud(ctl)
        with m, c:
            res = self.mod.set_state({"name": "Strip"}, color_temperature=4000)
        self.assertEqual(res["applied"]["color_temperature_k"], 4000)
        _d, name, value = ctl.call_args[0]
        self.assertEqual((name, value), ("colorTem", 4000))

    def test_cloud_turn_error_propagates(self):
        ctl = mock.Mock(return_value={"error": "cloud control failed: 401"})
        m, c = self._cloud(ctl)
        with m, c:
            res = self.mod.set_state({"name": "Strip"}, on=True)
        self.assertIn("cloud control failed", res["error"])

    def test_cloud_brightness_error_propagates(self):
        ctl = mock.Mock(return_value={"error": "cloud control failed: 429"})
        m, c = self._cloud(ctl)
        with m, c:
            res = self.mod.set_state({"name": "Strip"}, brightness=50)
        self.assertIn("cloud control failed", res["error"])

    def test_cloud_color_error_propagates(self):
        ctl = mock.Mock(return_value={"error": "cloud control failed: 500"})
        m, c = self._cloud(ctl)
        with m, c:
            res = self.mod.set_state({"name": "Strip"}, color=(1, 2, 3))
        self.assertIn("cloud control failed", res["error"])

    def test_cloud_color_temperature_error_propagates(self):
        ctl = mock.Mock(return_value={"error": "cloud control failed: 503"})
        m, c = self._cloud(ctl)
        with m, c:
            res = self.mod.set_state({"name": "Strip"}, color_temperature=4000)
        self.assertIn("cloud control failed", res["error"])


if __name__ == "__main__":
    unittest.main()
