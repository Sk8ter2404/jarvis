"""Logic tests for skills/sh_kasa.py (TP-Link Kasa/Tapo controller).

Thin wrapper over the optional `python-kasa` library plus a freeform voice
control entry point (`smart_home_control`). Coverage:
  * graceful degradation when python-kasa absent / nothing discovered,
  * the pure _rgb_to_hsv helper,
  * smart_home_control intent parsing + device matching, fully mocked so no
    LAN broadcast happens,
  * discovery (broadcast + cached + Tapo Credentials kwarg + fallbacks),
  * device resolution by IP (discover_single / SmartPlug fallback) and by alias,
  * list_devices capability mapping, get_state / set_state apply branches.

The real `python-kasa` is NOT on the CI runner, so we never import it: the
skill resolves it lazily through `_kasa()`, which we patch to hand back a
hand-rolled fake module (FakeKasaModule). The async helpers are exercised for
real through the skill's own `_run_async` (asyncio.run) — the fakes' coroutine
methods complete instantly and touch no network. Module globals (`_state`) are
reset in tearDown so nothing leaks between tests.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── async-capable fakes for the `kasa` library ──────────────────────────
class FakeKasaDevice:
    """Stand-in for a python-kasa SmartDevice/SmartBulb. Records calls so the
    set_state branch coverage can be asserted. All control methods are async."""

    def __init__(self, alias="Lamp", model="HS100", *, is_on=False,
                 is_dimmable=False, is_color=False, is_variable_color_temp=False,
                 is_bulb=False, is_strip=False, is_dimmer=False, brightness=0,
                 update_raises=False):
        self.alias = alias
        self.model = model
        self.is_on = is_on
        self.is_dimmable = is_dimmable
        self.is_color = is_color
        self.is_variable_color_temp = is_variable_color_temp
        self.is_bulb = is_bulb
        self.is_strip = is_strip
        self.is_dimmer = is_dimmer
        self.brightness = brightness
        self._update_raises = update_raises
        self.calls: list = []

    async def update(self):
        self.calls.append(("update",))
        if self._update_raises:
            raise RuntimeError("update boom")

    async def turn_on(self):
        self.calls.append(("turn_on",))
        self.is_on = True

    async def turn_off(self):
        self.calls.append(("turn_off",))
        self.is_on = False

    async def set_brightness(self, pct):
        self.calls.append(("set_brightness", pct))
        self.brightness = pct

    async def set_color_temp(self, k):
        self.calls.append(("set_color_temp", k))

    async def set_hsv(self, h, s, v):
        self.calls.append(("set_hsv", h, s, v))


def make_fake_kasa(*, discover_result=None, discover_raises=None,
                   discover_typeerror=False, discover_noarg_raises=False,
                   credentials_raises=False, discover_single=None,
                   discover_single_raises=False, has_discover_single=True,
                   smartplug_device=None, smartplug_raises=False,
                   has_smartplug=True, has_credentials=True):
    """Build a fake `kasa` module. ``discover_result`` is the {ip -> device}
    dict Discover.discover() returns. Flags toggle the experimental-API
    fallbacks the skill walks (discover_single vs SmartPlug, Credentials)."""
    mod = types.ModuleType("kasa")
    cred_seen: dict = {}

    if has_credentials:
        class Credentials:
            def __init__(self, username, password):
                if credentials_raises:
                    raise RuntimeError("bad creds")
                cred_seen["username"] = username
                cred_seen["password"] = password
        mod.Credentials = Credentials
    mod._cred_seen = cred_seen

    class Discover:
        last_kwargs: dict = {}

        @staticmethod
        async def discover(**kwargs):
            Discover.last_kwargs = kwargs
            if not kwargs and discover_noarg_raises:
                # The no-argument retry (after a TypeError) also fails.
                raise RuntimeError("no-arg discover boom")
            if discover_typeerror and "timeout" in kwargs:
                # Old python-kasa lacked the timeout/credentials kwargs.
                raise TypeError("unexpected kwarg")
            if discover_raises is not None:
                raise discover_raises
            return discover_result if discover_result is not None else {}

        @staticmethod
        async def discover_single(ip):
            if discover_single_raises:
                raise RuntimeError("single boom")
            return discover_single

    if has_discover_single:
        # discover_single is an attribute of the class above already.
        pass
    else:
        delattr(Discover, "discover_single")
    mod.Discover = Discover

    if has_smartplug:
        class SmartPlug:
            def __init__(self, ip):
                self.ip = ip
                self._dev = smartplug_device

            async def update(self):
                if smartplug_raises:
                    raise RuntimeError("plug boom")
        mod.SmartPlug = SmartPlug
    return mod


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


# ─── shared base: reset module globals + patch _kasa with a fake ──────────
class _KasaBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_kasa")
        # Snapshot _state so each test starts from a clean discovery cache and
        # nothing leaks across tests.
        self._saved_state = dict(self.mod._state)
        self.addCleanup(self._restore_state)

    def _restore_state(self):
        self.mod._state.clear()
        self.mod._state.update(self._saved_state)

    def _use_kasa(self, fake):
        """Patch _kasa() to return our fake module for the duration of a test
        and force the discovery cache stale so a refresh actually runs."""
        self.mod._state["fetched_at"] = 0.0
        p = mock.patch.object(self.mod, "_kasa", return_value=fake)
        p.start()
        self.addCleanup(p.stop)
        return fake


class KasaConfigTests(_KasaBase):
    def test_read_config_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._read_config(), {})

    def test_read_config_parses_json(self):
        m = mock.mock_open(read_data='{"username": "u", "password": "p"}')
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            cfg = self.mod._read_config()
        self.assertEqual(cfg["username"], "u")

    def test_read_config_bad_json_returns_empty(self):
        m = mock.mock_open(read_data="{not json")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._read_config(), {})

    def test_is_available_true_with_kasa(self):
        with mock.patch.object(self.mod, "_kasa", return_value=object()):
            self.assertTrue(self.mod.is_available())

    def test_kasa_import_returns_none_when_absent(self):
        # Force `import kasa` to raise so the except-branch returns None. Robust
        # even on the dev box where python-kasa IS installed.
        real_import = __import__

        def _blocked(name, *a, **k):
            if name == "kasa" or name.split(".")[0] == "kasa":
                raise ImportError("blocked kasa")
            return real_import(name, *a, **k)

        import sys as _sys
        with mock.patch.dict(_sys.modules, {"kasa": None}), \
             mock.patch("builtins.__import__", side_effect=_blocked):
            self.assertIsNone(self.mod._kasa())

    def test_kasa_import_succeeds_when_present(self):
        # On the dev box python-kasa is installed → real import path (42-44)
        # runs and returns a module. On CI (no kasa) this skips cleanly.
        import importlib.util as _u
        if _u.find_spec("kasa") is None:
            self.skipTest("python-kasa not installed on this runner")
        self.assertIsNotNone(self.mod._kasa())


class KasaRunAsyncTests(_KasaBase):
    def test_run_async_simple(self):
        async def _coro():
            return 21 * 2
        self.assertEqual(self.mod._run_async(_coro()), 42)

    def test_run_async_propagates_error(self):
        async def _coro():
            raise ValueError("nope")
        with self.assertRaises(ValueError):
            self.mod._run_async(_coro())

    def test_run_async_from_within_running_loop(self):
        # When called from inside an event loop, _run_async delegates to a
        # worker thread. Drive that nested branch.
        async def _inner():
            return 7

        async def _outer():
            # We're now inside a running loop; _run_async must still work.
            return self.mod._run_async(_inner())

        self.assertEqual(asyncio.run(_outer()), 7)

    def test_run_async_worker_thread_error_reraised(self):
        async def _inner():
            raise RuntimeError("worker boom")

        async def _outer():
            return self.mod._run_async(_inner())

        with self.assertRaises(RuntimeError):
            asyncio.run(_outer())


class KasaDiscoveryTests(_KasaBase):
    def test_discover_returns_devices_and_indexes_by_name(self):
        dev = FakeKasaDevice(alias="Entry Light")
        fake = self._use_kasa(make_fake_kasa(discover_result={"10.0.0.5": dev}))
        found = self.mod._refresh_discovery(force=True)
        self.assertIn("10.0.0.5", found)
        # by_name index is lower-cased.
        self.assertIs(self.mod._state["by_name"]["entry light"], dev)
        # discover() was called with the default timeout kwarg.
        self.assertEqual(fake.Discover.last_kwargs.get("timeout"), 5)

    def test_discover_passes_tapo_credentials_when_configured(self):
        dev = FakeKasaDevice(alias="Tapo Plug")
        fake = make_fake_kasa(discover_result={"10.0.0.8": dev})
        self._use_kasa(fake)
        with mock.patch.object(self.mod, "_read_config",
                               return_value={"username": "u@example.com",
                                             "password": "secret"}):
            self.mod._refresh_discovery(force=True)
        self.assertIn("credentials", fake.Discover.last_kwargs)
        self.assertEqual(fake._cred_seen["username"], "u@example.com")

    def test_discover_typeerror_falls_back_to_no_kwargs(self):
        # Old python-kasa: discover(**kwargs) raises TypeError, skill retries
        # discover() with no args.
        dev = FakeKasaDevice(alias="Old Plug")
        fake = make_fake_kasa(discover_result={"10.0.0.9": dev},
                              discover_typeerror=True)
        self._use_kasa(fake)
        found = self.mod._refresh_discovery(force=True)
        self.assertIn("10.0.0.9", found)

    def test_discover_returns_empty_when_kasa_absent(self):
        self.mod._state["fetched_at"] = 0.0
        with mock.patch.object(self.mod, "_kasa", return_value=None):
            self.assertEqual(self.mod._refresh_discovery(force=True), {})

    def test_discover_handles_exception(self):
        fake = make_fake_kasa(discover_raises=RuntimeError("net down"))
        self._use_kasa(fake)
        # discover() raises (not TypeError) → inner handler returns {}.
        self.assertEqual(self.mod._refresh_discovery(force=True), {})

    def test_discover_no_discover_attr(self):
        mod = types.ModuleType("kasa")  # no Discover at all
        self._use_kasa(mod)
        self.assertEqual(self.mod._refresh_discovery(force=True), {})

    def test_discovery_cache_hit_skips_rescan(self):
        dev = FakeKasaDevice(alias="Cached")
        fake = self._use_kasa(make_fake_kasa(discover_result={"10.0.0.5": dev}))
        self.mod._refresh_discovery(force=True)
        first_kwargs = dict(fake.Discover.last_kwargs)
        # Mark cache fresh and call WITHOUT force → no rescan.
        self.mod._state["fetched_at"] = self.mod.time.monotonic()
        fake.Discover.last_kwargs = {"sentinel": True}
        out = self.mod._refresh_discovery(force=False)
        self.assertEqual(out, {"10.0.0.5": dev})
        self.assertEqual(fake.Discover.last_kwargs, {"sentinel": True})  # untouched
        self.assertEqual(first_kwargs.get("timeout"), 5)

    def test_alias_indexing_skips_blank_alias(self):
        dev = FakeKasaDevice(alias="   ")  # whitespace alias → not indexed
        self._use_kasa(make_fake_kasa(discover_result={"10.0.0.5": dev}))
        self.mod._refresh_discovery(force=True)
        self.assertEqual(self.mod._state["by_name"], {})

    def test_credentials_construction_error_is_swallowed(self):
        # Credentials() raising must not abort discovery — the skill proceeds
        # without the credentials kwarg.
        dev = FakeKasaDevice(alias="Plug")
        fake = make_fake_kasa(discover_result={"10.0.0.5": dev},
                              credentials_raises=True)
        self._use_kasa(fake)
        with mock.patch.object(self.mod, "_read_config",
                               return_value={"username": "u", "password": "p"}):
            found = self.mod._refresh_discovery(force=True)
        self.assertIn("10.0.0.5", found)
        self.assertNotIn("credentials", fake.Discover.last_kwargs)

    def test_discover_typeerror_then_noarg_also_raises(self):
        # discover(**kwargs) → TypeError, retried discover() → also raises →
        # inner handler returns {}.
        fake = make_fake_kasa(discover_typeerror=True, discover_noarg_raises=True)
        self._use_kasa(fake)
        self.assertEqual(self.mod._refresh_discovery(force=True), {})

    def test_refresh_discovery_run_async_error_swallowed(self):
        # _run_async itself blowing up is caught at the _refresh_discovery level.
        self._use_kasa(make_fake_kasa())
        with mock.patch.object(self.mod, "_run_async",
                               side_effect=RuntimeError("loop boom")):
            self.assertEqual(self.mod._refresh_discovery(force=True), {})

    def test_alias_access_exception_skipped(self):
        # A device whose .alias access raises is skipped during the by_name
        # indexing loop (inner try/except).
        class _BadAlias:
            @property
            def alias(self):
                raise RuntimeError("alias boom")
        good = FakeKasaDevice(alias="Good")
        self._use_kasa(make_fake_kasa(discover_result={
            "10.0.0.5": good, "10.0.0.6": _BadAlias()}))
        self.mod._refresh_discovery(force=True)
        self.assertIn("good", self.mod._state["by_name"])
        self.assertEqual(len(self.mod._state["by_name"]), 1)


class KasaDeviceResolutionTests(_KasaBase):
    def test_device_from_ip_uses_discover_single(self):
        target = FakeKasaDevice(alias="Direct")
        fake = make_fake_kasa(discover_single=target)
        self._use_kasa(fake)
        dev = self.mod._device_for({"name": "x", "lan_ip": "10.0.0.5"})
        self.assertIs(dev, target)

    def test_device_from_ip_falls_back_to_smartplug(self):
        plug = FakeKasaDevice(alias="Plug")
        fake = make_fake_kasa(discover_single_raises=True, smartplug_device=plug)
        self._use_kasa(fake)
        # discover_single raises → SmartPlug(ip).update() path returns the plug.
        dev = self.mod._device_for({"name": "x", "lan_ip": "10.0.0.5"})
        self.assertIsNotNone(dev)
        self.assertTrue(hasattr(dev, "ip"))

    def test_device_from_ip_smartplug_update_raises_returns_none_then_scans(self):
        # No discover_single, SmartPlug.update raises → IP path yields None, so
        # _device_for falls through to a broadcast scan + alias lookup.
        named = FakeKasaDevice(alias="Hall Lamp")
        fake = make_fake_kasa(has_discover_single=False,
                              smartplug_raises=True,
                              discover_result={"10.0.0.6": named})
        self._use_kasa(fake)
        dev = self.mod._device_for({"name": "Hall Lamp", "lan_ip": "10.0.0.5"})
        self.assertIs(dev, named)

    def test_device_for_no_ip_resolves_by_alias(self):
        named = FakeKasaDevice(alias="Kitchen")
        self._use_kasa(make_fake_kasa(discover_result={"10.0.0.7": named}))
        dev = self.mod._device_for({"name": "Kitchen"})
        self.assertIs(dev, named)

    def test_device_for_unknown_name_returns_none(self):
        self._use_kasa(make_fake_kasa(discover_result={}))
        self.assertIsNone(self.mod._device_for({"name": "Ghost"}))

    def test_device_from_ip_async_no_kasa(self):
        with mock.patch.object(self.mod, "_kasa", return_value=None):
            res = self.mod._run_async(self.mod._device_from_ip_async("10.0.0.5"))
        self.assertIsNone(res)

    def test_device_from_ip_async_blank_ip(self):
        self._use_kasa(make_fake_kasa())
        res = self.mod._run_async(self.mod._device_from_ip_async(""))
        self.assertIsNone(res)

    def test_device_from_ip_async_no_single_no_smartplug(self):
        fake = make_fake_kasa(has_discover_single=False, has_smartplug=False)
        self._use_kasa(fake)
        res = self.mod._run_async(self.mod._device_from_ip_async("10.0.0.5"))
        self.assertIsNone(res)

    def test_device_for_ip_path_exception_falls_through_to_scan(self):
        # _run_async raising on the direct-IP attempt is caught; the function
        # then scans and resolves by alias.
        named = FakeKasaDevice(alias="Den")
        self._use_kasa(make_fake_kasa(discover_result={"10.0.0.6": named}))
        real_run = self.mod._run_async
        calls = {"n": 0}

        def _flaky(coro, *_a, **_kw):   # _run_async now takes timeout/what
            calls["n"] += 1
            if calls["n"] == 1:
                coro.close()  # avoid 'never awaited' warning
                raise RuntimeError("ip attempt boom")
            return real_run(coro)

        with mock.patch.object(self.mod, "_run_async", side_effect=_flaky):
            dev = self.mod._device_for({"name": "Den", "lan_ip": "10.0.0.5"})
        self.assertIs(dev, named)


class KasaListDevicesTests(_KasaBase):
    def test_list_devices_maps_capabilities_and_type(self):
        bulb = FakeKasaDevice(alias="Color Bulb", model="KL130",
                              is_dimmable=True, is_color=True,
                              is_variable_color_temp=True, is_bulb=True)
        self._use_kasa(make_fake_kasa(discover_result={"10.0.0.5": bulb}))
        devs = self.mod.list_devices()
        self.assertEqual(len(devs), 1)
        d = devs[0]
        self.assertEqual(d["name"], "Color Bulb")
        self.assertEqual(d["type"], "light")
        self.assertEqual(d["brand"], "TP-Link")
        for cap in ("on_off", "dim", "color", "color_temperature"):
            self.assertIn(cap, d["capabilities"])
        self.assertEqual(d["lan_ip"], "10.0.0.5")

    def test_list_devices_strip_and_dimmer_types(self):
        strip = FakeKasaDevice(alias="Strip", is_strip=True)
        dimmer = FakeKasaDevice(alias="Dimmer", is_dimmer=True, is_dimmable=True)
        self._use_kasa(make_fake_kasa(discover_result={
            "10.0.0.5": strip, "10.0.0.6": dimmer}))
        by_name = {d["name"]: d for d in self.mod.list_devices()}
        self.assertEqual(by_name["Strip"]["type"], "strip")
        self.assertEqual(by_name["Dimmer"]["type"], "dimmer")
        self.assertIn("dim", by_name["Dimmer"]["capabilities"])

    def test_list_devices_plug_default_type(self):
        plug = FakeKasaDevice(alias="Plug")
        self._use_kasa(make_fake_kasa(discover_result={"10.0.0.5": plug}))
        self.assertEqual(self.mod.list_devices()[0]["type"], "plug")

    def test_kasa_list_action_lists_names(self):
        plug = FakeKasaDevice(alias="Entry Plug")
        self._use_kasa(make_fake_kasa(discover_result={"10.0.0.5": plug}))
        out = self.actions["kasa_list"]("")
        self.assertIn("Entry Plug", out)
        self.assertIn("1 Kasa device", out)


class KasaGetStateTests(_KasaBase):
    def test_get_state_reports_on_and_brightness(self):
        dev = FakeKasaDevice(alias="Lamp", model="HS220", is_on=True,
                             is_dimmable=True, brightness=66)
        with mock.patch.object(self.mod, "_device_for", return_value=dev):
            st = self.mod.get_state({"name": "Lamp"})
        self.assertTrue(st["on"])
        self.assertEqual(st["brightness"], 66)
        self.assertEqual(st["alias"], "Lamp")
        self.assertEqual(st["model"], "HS220")

    def test_get_state_non_dimmable_brightness_none(self):
        dev = FakeKasaDevice(alias="Plug", is_on=False, is_dimmable=False)
        with mock.patch.object(self.mod, "_device_for", return_value=dev):
            st = self.mod.get_state({"name": "Plug"})
        self.assertIsNone(st["brightness"])
        self.assertFalse(st["on"])

    def test_get_state_update_failure_returns_error(self):
        dev = FakeKasaDevice(alias="Lamp", update_raises=True)
        with mock.patch.object(self.mod, "_device_for", return_value=dev):
            st = self.mod.get_state({"name": "Lamp"})
        self.assertIn("state read failed", st["error"])


class KasaSetStateTests(_KasaBase):
    def _dev_patch(self, dev):
        p = mock.patch.object(self.mod, "_device_for", return_value=dev)
        p.start()
        self.addCleanup(p.stop)
        return dev

    def test_set_on_true(self):
        dev = self._dev_patch(FakeKasaDevice(alias="Lamp"))
        res = self.mod.set_state({"name": "Lamp"}, on=True)
        self.assertTrue(res["ok"])
        self.assertTrue(res["applied"]["on"])
        self.assertIn(("turn_on",), dev.calls)

    def test_set_off(self):
        dev = self._dev_patch(FakeKasaDevice(alias="Lamp", is_on=True))
        res = self.mod.set_state({"name": "Lamp"}, on=False)
        self.assertFalse(res["applied"]["on"])
        self.assertIn(("turn_off",), dev.calls)

    def test_set_brightness_on_dimmable_turns_on_when_off(self):
        dev = self._dev_patch(FakeKasaDevice(alias="Lamp", is_dimmable=True))
        res = self.mod.set_state({"name": "Lamp"}, brightness=150)  # clamps to 100
        self.assertEqual(res["applied"]["brightness"], 100)
        self.assertTrue(res["applied"]["on"])          # >0% forced power-on
        self.assertIn(("set_brightness", 100), dev.calls)

    def test_set_brightness_zero_does_not_force_on(self):
        dev = self._dev_patch(FakeKasaDevice(alias="Lamp", is_dimmable=True))
        res = self.mod.set_state({"name": "Lamp"}, brightness=0)
        self.assertEqual(res["applied"]["brightness"], 0)
        self.assertNotIn("on", res["applied"])
        self.assertNotIn(("turn_on",), dev.calls)

    def test_set_brightness_on_non_dimmable_reports_the_skip(self):
        # REGRESSION (C1): the plug is switched fully ON and the brightness is
        # never applied. The old code returned a flat ok:True, so the router
        # spoke "Set to 50%, sir" about a level the hardware cannot reach.
        self._dev_patch(FakeKasaDevice(alias="Plug", is_dimmable=False))
        res = self.mod.set_state({"name": "Plug"}, brightness=50)
        self.assertNotIn("brightness", res["applied"])
        self.assertTrue(res["applied"].get("on"))   # >0 still forces power on
        self.assertFalse(res.get("ok"))             # NOT a success
        self.assertIn("not dimmable", res["skipped"]["brightness"])
        self.assertEqual(res["failed"], {})         # nothing was attempted
        self.assertIn("brightness not applied", res["error"])
        self.assertIn("Plug", res["error"])

    def test_set_color_temperature_on_capable_bulb(self):
        dev = self._dev_patch(FakeKasaDevice(alias="Bulb",
                                             is_variable_color_temp=True))
        res = self.mod.set_state({"name": "Bulb"}, color_temperature=3000)
        self.assertEqual(res["applied"]["color_temperature_k"], 3000)
        self.assertIn(("set_color_temp", 3000), dev.calls)

    def test_set_color_temperature_skipped_when_unsupported(self):
        dev = self._dev_patch(FakeKasaDevice(alias="Plug",
                                             is_variable_color_temp=False))
        res = self.mod.set_state({"name": "Plug"}, color_temperature=3000)
        self.assertNotIn("color_temperature_k", res["applied"])
        self.assertNotIn(("set_color_temp", 3000), dev.calls)
        self.assertFalse(res.get("ok"))
        self.assertIn("colour temperature", res["skipped"]["color_temperature"])

    def test_set_color_on_color_bulb(self):
        dev = self._dev_patch(FakeKasaDevice(alias="Bulb", is_color=True))
        res = self.mod.set_state({"name": "Bulb"}, color=(255, 0, 0))
        self.assertEqual(res["applied"]["color"], [255, 0, 0])
        # set_hsv called with red → hue 0, sat 100, val 100.
        self.assertIn(("set_hsv", 0, 100, 100), dev.calls)

    def test_set_color_skipped_on_non_color_device(self):
        self._dev_patch(FakeKasaDevice(alias="Plug", is_color=False))
        res = self.mod.set_state({"name": "Plug"}, color=(255, 0, 0))
        self.assertNotIn("color", res["applied"])
        self.assertFalse(res.get("ok"))
        self.assertIn("not colour-capable", res["skipped"]["color"])

    def test_set_state_apply_exception_returns_partial(self):
        dev = FakeKasaDevice(alias="Lamp", update_raises=True)
        with mock.patch.object(self.mod, "_device_for", return_value=dev):
            res = self.mod.set_state({"name": "Lamp"}, on=True)
        self.assertIn("set_state failed", res["error"])
        self.assertIn("partial", res)

    def test_set_brightness_set_call_failure_is_not_ok(self):
        # REGRESSION (C1): a sub-command that was ATTEMPTED and RAISED must
        # never yield ok:True. It used to be `except Exception: pass`.
        dev = FakeKasaDevice(alias="Lamp", is_dimmable=True)
        async def _boom(_pct):
            raise RuntimeError("dim boom")
        dev.set_brightness = _boom
        with mock.patch.object(self.mod, "_device_for", return_value=dev):
            res = self.mod.set_state({"name": "Lamp"}, brightness=40)
        self.assertFalse(res.get("ok"))
        self.assertNotIn("brightness", res["applied"])
        self.assertIn("dim boom", res["failed"]["brightness"])
        self.assertIn("brightness failed", res["error"])
        # ...and what DID land is still reported, not thrown away.
        self.assertTrue(res["applied"].get("on"))
        self.assertIn("applied: on=True", res["error"])

    def test_set_color_temp_call_failure_is_not_ok(self):
        dev = FakeKasaDevice(alias="Bulb", is_variable_color_temp=True)
        async def _boom(_k):
            raise RuntimeError("ct boom")
        dev.set_color_temp = _boom
        with mock.patch.object(self.mod, "_device_for", return_value=dev):
            res = self.mod.set_state({"name": "Bulb"}, color_temperature=3000)
        self.assertFalse(res.get("ok"))
        self.assertNotIn("color_temperature_k", res["applied"])
        self.assertIn("ct boom", res["failed"]["color_temperature"])

    def test_set_color_call_failure_is_not_ok(self):
        dev = FakeKasaDevice(alias="Bulb", is_color=True)
        async def _boom(_h, _s, _v):
            raise RuntimeError("hsv boom")
        dev.set_hsv = _boom
        with mock.patch.object(self.mod, "_device_for", return_value=dev):
            res = self.mod.set_state({"name": "Bulb"}, color=(0, 255, 0))
        self.assertFalse(res.get("ok"))
        self.assertNotIn("color", res["applied"])
        self.assertIn("hsv boom", res["failed"]["color"])

    def test_set_brightness_forced_on_failure_is_reported(self):
        # brightness>0 on a non-dimmable device tries turn_on, which raises.
        # Nothing at all reached the hardware — the result must say so twice
        # over (skipped brightness AND failed power) and never claim success.
        dev = FakeKasaDevice(alias="Lamp", is_dimmable=False)
        async def _boom():
            raise RuntimeError("on boom")
        dev.turn_on = _boom
        with mock.patch.object(self.mod, "_device_for", return_value=dev):
            res = self.mod.set_state({"name": "Lamp"}, brightness=40)
        self.assertFalse(res.get("ok"))
        self.assertNotIn("on", res["applied"])
        self.assertIn("on boom", res["failed"]["on"])
        self.assertIn("not dimmable", res["skipped"]["brightness"])
        self.assertIn("applied: nothing", res["error"])

    def test_fully_applied_request_is_still_plain_ok(self):
        # The honest-failure work must not make healthy calls look degraded.
        dev = self._dev_patch(FakeKasaDevice(alias="Bulb", is_dimmable=True))
        res = self.mod.set_state({"name": "Bulb"}, on=True, brightness=40)
        self.assertTrue(res["ok"])
        self.assertNotIn("error", res)
        self.assertNotIn("failed", res)
        self.assertNotIn("skipped", res)
        self.assertEqual(res["applied"]["brightness"], 40)
        self.assertEqual(res["device"], "Bulb")
        self.assertIn(("set_brightness", 40), dev.calls)


class KasaListDevicesEdgeTests(_KasaBase):
    def test_list_devices_skips_device_that_raises(self):
        good = FakeKasaDevice(alias="Good")

        class _Boom:
            # Attribute access blows up so the per-device try/except `continue`
            # branch in list_devices is exercised.
            alias = "Bad"
            def __getattr__(self, _n):
                raise RuntimeError("attr boom")
        self._use_kasa(make_fake_kasa(discover_result={
            "10.0.0.5": good, "10.0.0.6": _Boom()}))
        devs = self.mod.list_devices()
        names = [d["name"] for d in devs]
        self.assertIn("Good", names)
        self.assertNotIn("Bad", names)


class KasaTuyaModTests(_KasaBase):
    def test_tuya_mod_found_in_sys_modules(self):
        import sys as _sys
        sentinel = types.ModuleType("skill_sh_tuya")
        with mock.patch.dict(_sys.modules, {"skill_sh_tuya": sentinel}):
            self.assertIs(self.mod._tuya_mod(), sentinel)

    def test_tuya_mod_none_when_absent(self):
        import importlib as _il
        import sys as _sys
        names = ("skill_sh_tuya", "sh_tuya", "skills.sh_tuya")
        # Remove any cached candidates AND force importlib.import_module to fail
        # so the function walks every branch to its final `return None`. The
        # function does `import importlib` locally, which re-binds the same
        # cached module object, so patching its import_module attr is seen.
        cleared = {n: None for n in names}
        with mock.patch.dict(_sys.modules, cleared, clear=False), \
             mock.patch.object(_il, "import_module",
                               side_effect=ImportError("no sh_tuya")):
            self.assertIsNone(self.mod._tuya_mod())


class KasaSmartHomeControlExtraTests(_KasaBase):
    def test_list_devices_exception_is_swallowed(self):
        # list_devices raising must not crash smart_home_control; with no tuya
        # and no devices it returns the 'no controllable devices' line.
        with mock.patch.object(self.mod, "list_devices",
                               side_effect=RuntimeError("scan boom")), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None):
            out = self.actions["smart_home_control"]("turn on lamp")
        self.assertIn("don't see any", out.lower())

    def test_tuya_devices_merged_and_routed(self):
        # A Tuya module supplies a device; smart_home_control must route control
        # to the tuya module's set_state, not kasa's.
        tuya = types.SimpleNamespace()
        tuya.list_devices = lambda: [{"name": "Tuya Lamp", "_tuya": {"id": "abc"}}]
        tuya.set_state = mock.Mock(return_value={"ok": True})
        tuya.get_state = mock.Mock(return_value={"on": False})
        kasa_set = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "list_devices", return_value=[]), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=tuya), \
             mock.patch.object(self.mod, "set_state", kasa_set):
            out = self.actions["smart_home_control"]("turn on tuya lamp")
        self.assertIn("tuya lamp on", out.lower())
        tuya.set_state.assert_called_once()
        kasa_set.assert_not_called()

    def test_tuya_list_devices_exception_swallowed(self):
        tuya = types.SimpleNamespace()
        tuya.list_devices = mock.Mock(side_effect=RuntimeError("tuya boom"))
        kasa_devs = [{"name": "Entry", "lan_ip": "10.0.0.5"}]
        kasa_set = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "list_devices", return_value=kasa_devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=tuya), \
             mock.patch.object(self.mod, "set_state", kasa_set):
            out = self.actions["smart_home_control"]("turn on entry")
        self.assertIn("entry on", out.lower())

    def test_word_overlap_fallback_matches_best(self):
        # No exact substring match, but 'bedroom' overlaps 'Master Bedroom'.
        devs = [{"name": "Master Bedroom", "lan_ip": "10.0.0.5"},
                {"name": "Garage", "lan_ip": "10.0.0.6"}]
        kasa_set = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "list_devices", return_value=devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None), \
             mock.patch.object(self.mod, "set_state", kasa_set):
            out = self.actions["smart_home_control"]("turn on the bedroom please")
        self.assertIn("master bedroom on", out.lower())

    def test_no_match_lists_controllable_devices(self):
        # A request that matches no device name and shares no words → the
        # 'not sure which one' help line listing devices.
        devs = [{"name": "Garage", "lan_ip": "10.0.0.5"}]
        with mock.patch.object(self.mod, "list_devices", return_value=devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None):
            out = self.actions["smart_home_control"]("toggle xyzzy")
        self.assertIn("not sure which", out.lower())
        self.assertIn("Garage", out)

    def test_control_failure_reports_failed(self):
        # REGRESSION (C1): "turn on the entry light", the LAN call fails —
        # JARVIS must NOT say "Done", and must name the device and the reason.
        devs = [{"name": "Entry", "lan_ip": "10.0.0.5"}]
        with mock.patch.object(self.mod, "list_devices", return_value=devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None), \
             mock.patch.object(self.mod, "set_state",
                               return_value={"error": "device offline"}):
            out = self.actions["smart_home_control"]("turn on entry")
        low = out.lower()
        self.assertFalse(low.startswith("done, sir"))
        self.assertIn("that didn", low)
        self.assertIn("entry (failed: device offline)", low)

    def test_control_partial_failure_is_not_reported_as_done(self):
        # One of two devices fails → "Partly done", both outcomes named.
        devs = [{"name": "Entry Light", "lan_ip": "10.0.0.5"},
                {"name": "Dining Room", "lan_ip": "10.0.0.6"}]

        def _set(rec, **_kw):
            if rec["name"] == "Entry Light":
                return {"ok": True}
            return {"error": "no route to host"}

        with mock.patch.object(self.mod, "list_devices", return_value=devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None), \
             mock.patch.object(self.mod, "set_state", side_effect=_set):
            out = self.actions["smart_home_control"]("turn off all the lights")
        low = out.lower()
        self.assertFalse(low.startswith("done, sir"))
        self.assertIn("partly done", low)
        self.assertIn("entry light off", low)
        self.assertIn("dining room (failed: no route to host)", low)

    def test_control_skipped_subcommand_is_not_reported_as_done(self):
        # set_state's honest ok:False (capability skip) must survive into the
        # spoken line rather than being read as success.
        devs = [{"name": "Entry", "lan_ip": "10.0.0.5"}]
        res = {"ok": False, "device": "Entry", "applied": {"on": True},
               "skipped": {"brightness": "device is not dimmable"},
               "error": "kasa 'Entry': brightness not applied "
                        "(device is not dimmable) (applied: on=True)"}
        with mock.patch.object(self.mod, "list_devices", return_value=devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None), \
             mock.patch.object(self.mod, "set_state", return_value=res):
            out = self.actions["smart_home_control"]("turn on entry")
        self.assertFalse(out.lower().startswith("done, sir"))
        self.assertIn("not dimmable", out)

    def test_toggle_refuses_to_guess_when_state_unreadable(self):
        # NEVER GUESS: a failed get_state used to fall through to
        # `not bool(st.get("on"))` == True and silently switch the device ON.
        devs = [{"name": "Entry", "lan_ip": "10.0.0.5"}]
        set_state = mock.Mock(return_value={"ok": True})
        with mock.patch.object(self.mod, "list_devices", return_value=devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None), \
             mock.patch.object(self.mod, "get_state",
                               return_value={"error": "kasa state read failed"}), \
             mock.patch.object(self.mod, "set_state", set_state):
            out = self.actions["smart_home_control"]("toggle entry")
        set_state.assert_not_called()
        low = out.lower()
        self.assertIn("read state", low)
        self.assertFalse(low.startswith("done, sir"))

    def test_status_query_says_unknown_instead_of_off(self):
        devs = [{"name": "Dining Room", "lan_ip": "10.0.0.6"}]
        with mock.patch.object(self.mod, "list_devices", return_value=devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None), \
             mock.patch.object(self.mod, "get_state",
                               return_value={"error": "kasa state read failed: boom"}):
            out = self.actions["smart_home_control"]("status of the dining room")
        low = out.lower()
        self.assertIn("status unknown", low)
        self.assertNotIn("is off", low)

    def test_control_turn_budget_reports_devices_not_attempted(self):
        # REGRESSION (C2): "turn off everything" fans out sequentially, so a
        # per-call bound alone still multiplies by device count. Once the turn
        # budget is spent the remaining devices must be NAMED as not attempted,
        # never silently dropped under a "Done".
        devs = [{"name": "Entry Light", "lan_ip": "10.0.0.5"},
                {"name": "Dining Room", "lan_ip": "10.0.0.6"}]

        def _slow(_rec, **_kw):
            time.sleep(0.12)
            return {"ok": True}

        with mock.patch.object(self.mod, "list_devices", return_value=devs), \
             mock.patch.object(self.mod, "_tuya_mod", return_value=None), \
             mock.patch.object(self.mod, "_CONTROL_BUDGET_SECS", 0.05), \
             mock.patch.object(self.mod, "set_state", side_effect=_slow) as ss:
            out = self.actions["smart_home_control"]("turn off all the lights")
        self.assertEqual(ss.call_count, 1)          # second never attempted
        low = out.lower()
        self.assertFalse(low.startswith("done, sir"))
        self.assertIn("partly done", low)
        self.assertIn("1 not attempted", low)
        self.assertIn("dining room", low)
        self.assertIn("couldn", low)   # matches FAILURE_MARKERS ("couldn't")


# ─── C2: every await is BOUNDED, and expiry is reported, never swallowed ──
class KasaBoundedCallTests(_KasaBase):
    """skills/sh_kasa runs on the voice dispatch thread and nothing above it
    applies a timeout (bobert_companion's mid-task status line is gated on
    LONG_RUNNING_ACTIONS, which has no smart-home entry). `_run_async` used to
    be `asyncio.run(coro)` / bare `t.join()` with no bound at all, so a
    half-open plug could stall JARVIS past the 60 s main-loop heartbeat in
    total silence. These pin the bound AND the honesty of its expiry."""

    def test_run_async_times_out_on_the_production_path(self):
        # Not nested → `asyncio.run(_bounded())`, the branch that actually runs
        # in production. The old code had no future to time out here at all.
        t0 = time.monotonic()
        with self.assertRaises(TimeoutError) as ctx:
            self.mod._run_async(asyncio.sleep(30), timeout=0.05,
                                what="probe call")
        self.assertLess(time.monotonic() - t0, 5.0)
        msg = str(ctx.exception)
        self.assertIn("probe call", msg)        # says WHAT timed out
        self.assertIn("timed out", msg)
        self.assertIn("0.05", msg)              # …and after how long

    def test_run_async_cancels_rather_than_orphaning_the_coroutine(self):
        # asyncio.wait_for is the load-bearing part: a plain join(timeout)
        # would return control but leave the python-kasa query running on an
        # open socket forever.
        seen = []

        async def _long():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                seen.append("cancelled")
                raise

        with self.assertRaises(TimeoutError):
            self.mod._run_async(_long(), timeout=0.05)
        self.assertEqual(seen, ["cancelled"])

    def test_run_async_times_out_inside_a_running_loop(self):
        async def _outer():
            return self.mod._run_async(asyncio.sleep(30), timeout=0.05,
                                       what="nested probe")
        with self.assertRaises(TimeoutError) as ctx:
            asyncio.run(_outer())
        self.assertIn("nested probe", str(ctx.exception))

    def test_run_async_abandons_a_worker_that_ignores_cancellation(self):
        # A coroutine blocked in a native call never sees the cancel, so the
        # join carries its own grace margin and then gives up out loud.
        async def _blocking():
            time.sleep(0.4)        # blocks the worker's whole event loop
            return "late"

        async def _outer():
            return self.mod._run_async(_blocking(), timeout=0.01,
                                       what="wedged call")

        with mock.patch.object(self.mod, "_JOIN_GRACE_SECS", 0.02):
            with self.assertRaises(TimeoutError) as ctx:
                asyncio.run(_outer())
        self.assertIn("worker thread abandoned", str(ctx.exception))

    def test_run_async_default_timeout_is_read_at_call_time(self):
        # The budgets must stay tunable/patchable, not frozen into a default
        # argument at import.
        with mock.patch.object(self.mod, "_CALL_TIMEOUT_SECS", 0.05):
            with self.assertRaises(TimeoutError):
                self.mod._run_async(asyncio.sleep(30))

    def test_run_async_unbounded_only_when_asked(self):
        self.assertEqual(self.mod._run_async(asyncio.sleep(0, result=5),
                                             timeout=None), 5)

    def test_set_state_timeout_is_reported_not_swallowed(self):
        # REGRESSION (C2 + honest-failure): a hung device must time out AND
        # say so — never a silent hang, never a claimed success.
        dev = FakeKasaDevice(alias="Entry Light")

        async def _hang():
            await asyncio.sleep(30)
        dev.update = _hang

        with mock.patch.object(self.mod, "_device_for", return_value=dev), \
             mock.patch.object(self.mod, "_CALL_TIMEOUT_SECS", 0.05):
            t0 = time.monotonic()
            res = self.mod.set_state({"name": "Entry Light"}, on=True)
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertNotIn("ok", res)
        self.assertIn("timed out", res["error"])
        self.assertIn("Entry Light", res["error"])
        self.assertEqual(res["applied"], {})     # nothing reached the hardware
        self.assertNotIn(("turn_on",), dev.calls)

    def test_get_state_timeout_is_reported_not_swallowed(self):
        dev = FakeKasaDevice(alias="Entry Light")

        async def _hang():
            await asyncio.sleep(30)
        dev.update = _hang

        with mock.patch.object(self.mod, "_device_for", return_value=dev), \
             mock.patch.object(self.mod, "_CALL_TIMEOUT_SECS", 0.05):
            st = self.mod.get_state({"name": "Entry Light"})
        self.assertIn("timed out", st["error"])
        self.assertNotIn("on", st)

    def test_discovery_timeout_degrades_to_empty_and_logs(self):
        async def _hang():
            await asyncio.sleep(30)

        buf = io.StringIO()
        self.mod._state["fetched_at"] = 0.0
        with mock.patch.object(self.mod, "_discover_async",
                               side_effect=_hang), \
             mock.patch.object(self.mod, "_DISCOVERY_TIMEOUT_SECS", 0.05), \
             contextlib.redirect_stdout(buf):
            found = self.mod._refresh_discovery(force=True)
        self.assertEqual(found, {})
        out = buf.getvalue()
        self.assertIn("discovery error", out)
        self.assertIn("TimeoutError", out)       # the degraded path LOGS

    def test_direct_ip_failure_logs_before_falling_back_to_broadcast(self):
        # Was `except Exception: pass` — a timeout here was invisible and we
        # silently paid for a full LAN broadcast on top of it.
        async def _hang(_ip):
            await asyncio.sleep(30)

        buf = io.StringIO()
        with mock.patch.object(self.mod, "_device_from_ip_async",
                               side_effect=_hang), \
             mock.patch.object(self.mod, "_refresh_discovery",
                               return_value={}), \
             mock.patch.object(self.mod, "_CALL_TIMEOUT_SECS", 0.05), \
             contextlib.redirect_stdout(buf):
            dev = self.mod._device_for({"name": "Entry", "lan_ip": "10.0.0.5"})
        self.assertIsNone(dev)
        out = buf.getvalue()
        self.assertIn("direct connect to 10.0.0.5 failed", out)
        self.assertIn("falling back to broadcast", out)


if __name__ == "__main__":
    unittest.main()
