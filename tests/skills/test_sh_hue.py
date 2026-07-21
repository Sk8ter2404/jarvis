"""Logic tests for skills/sh_hue.py (Philips Hue controller).

The skill is a thin wrapper over the optional `phue` library. The highest-
value coverage is GRACEFUL DEGRADATION — every public path must return an
informative dict/string (never raise) when phue isn't installed or the bridge
isn't reachable. We also exercise the pure colour-maths helpers and the
set_state apply logic against a fake phue.Light, plus the bridge
connect/discovery/retry machinery and config IO.

`phue` is NOT on the CI runner; the skill resolves it lazily via `_phue()`,
which we patch to a hand-rolled fake module (make_fake_phue). The connect path
uses a real daemon thread with a join-timeout — for the fast cases the fake
Bridge returns instantly so no timeout fires; the timeout branch is driven by
patching `_threaded_connect` to return a canned ``timed_out`` dict (no real
30 s wait). Module globals (`_bridge_cache`, `_pending`) are reset in tearDown.
"""
from __future__ import annotations

import contextlib
import sys
import threading
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_SENTINEL = object()


@contextlib.contextmanager
def inject_core_submodule(leaf: str, obj):
    """Install/remove a fake ``core.<leaf>`` for the duration of a block.

    ``from core import <leaf>`` resolves the leaf via ``getattr(core, leaf)``
    once the real ``core`` package is imported, so patching ``sys.modules``
    ALONE is bypassed when another test already imported ``core.<leaf>``. We
    therefore ALSO set/clear the attribute on the live ``core`` package object,
    saving and restoring prior state (including absence). ``obj=None`` forces
    the import to fail (module removed + attr removed). Mirrors the proven
    ``inject_modules`` helper in test_self_diagnostic.py.
    """
    dotted = f"core.{leaf}"
    saved_mod = sys.modules.get(dotted, _SENTINEL)
    core_pkg = sys.modules.get("core")
    saved_attr = getattr(core_pkg, leaf, _SENTINEL) if core_pkg is not None else _SENTINEL
    if obj is None:
        sys.modules.pop(dotted, None)
        if core_pkg is not None and hasattr(core_pkg, leaf):
            try:
                delattr(core_pkg, leaf)
            except AttributeError:
                pass
    else:
        sys.modules[dotted] = obj
        if core_pkg is not None:
            setattr(core_pkg, leaf, obj)
    try:
        yield
    finally:
        if saved_mod is _SENTINEL:
            sys.modules.pop(dotted, None)
        else:
            sys.modules[dotted] = saved_mod
        if core_pkg is not None:
            if saved_attr is _SENTINEL:
                if hasattr(core_pkg, leaf):
                    try:
                        delattr(core_pkg, leaf)
                    except AttributeError:
                        pass
            else:
                setattr(core_pkg, leaf, saved_attr)


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
    def __init__(self, lights, raise_on_enum=False):
        self._lights = lights  # dict[name -> _FakeLight]
        self._raise_on_enum = raise_on_enum

    def get_light_objects(self, mode="name"):
        if self._raise_on_enum:
            raise RuntimeError("enum boom")
        return self._lights

    def connect(self):
        pass


class _PhueRegistrationException(Exception):
    """Mirrors phue.PhueRegistrationException by class NAME — the skill checks
    ``'PhueRegistrationException' in error.__class__.__name__``."""


def make_fake_phue(*, bridge=None, ctor_raises=None):
    """Fake `phue` module. ``bridge`` is returned by Bridge(ip); ``ctor_raises``
    (an exception instance) makes the Bridge ctor raise — e.g. a registration
    exception (button not pressed)."""
    mod = types.ModuleType("phue")
    seen: dict = {}

    def _Bridge(ip):
        seen["ip"] = ip
        if ctor_raises is not None:
            raise ctor_raises
        return bridge if bridge is not None else _FakeBridge({})

    mod.Bridge = _Bridge
    mod.PhueRegistrationException = _PhueRegistrationException
    mod._seen = seen
    return mod


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


# ─── shared base: reset bridge cache + pending between tests ──────────────
class _HueBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_hue")
        self._saved_cache = dict(self.mod._bridge_cache)
        self._saved_pending = dict(self.mod._pending)
        self.addCleanup(self._restore)

    def _restore(self):
        self.mod._bridge_cache.clear()
        self.mod._bridge_cache.update(self._saved_cache)
        self.mod._pending.clear()
        self.mod._pending.update(self._saved_pending)

    def _use_phue(self, fake):
        p = mock.patch.object(self.mod, "_phue", return_value=fake)
        p.start()
        self.addCleanup(p.stop)
        return fake


class HueDependencyTests(_HueBase):
    def test_phue_import_none_when_absent(self):
        real_import = __import__

        def _blocked(name, *a, **k):
            if name == "phue" or name.split(".")[0] == "phue":
                raise ImportError("blocked phue")
            return real_import(name, *a, **k)

        import sys as _sys
        with mock.patch.dict(_sys.modules, {"phue": None}), \
             mock.patch("builtins.__import__", side_effect=_blocked):
            self.assertIsNone(self.mod._phue())

    def test_phue_import_succeeds_when_present(self):
        import importlib.util as _u
        if _u.find_spec("phue") is None:
            self.skipTest("phue not installed on this runner")
        self.assertIsNotNone(self.mod._phue())

    def test_is_available_true_with_saved_bridge_ip(self):
        self._use_phue(make_fake_phue())
        with mock.patch.object(self.mod, "_read_config",
                               return_value={"bridge_ip": "192.168.1.10"}):
            self.assertTrue(self.mod.is_available())

    def test_is_available_true_without_bridge_ip(self):
        # phue installed but no saved IP → still "available" (discovery deferred).
        self._use_phue(make_fake_phue())
        with mock.patch.object(self.mod, "_read_config", return_value={}):
            self.assertTrue(self.mod.is_available())


class HueConfigTests(_HueBase):
    def test_read_config_missing(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._read_config(), {})

    def test_read_config_parses(self):
        m = mock.mock_open(read_data='{"bridge_ip": "192.168.1.10"}')
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._read_config()["bridge_ip"], "192.168.1.10")

    def test_read_config_bad_json(self):
        m = mock.mock_open(read_data="{bad")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._read_config(), {})

    def test_write_config_uses_atomic_writer(self):
        writer = mock.MagicMock()
        fake_atomic = types.ModuleType("core.atomic_io")
        fake_atomic._atomic_write_json = writer
        with mock.patch.object(self.mod.os, "makedirs"), \
             inject_core_submodule("atomic_io", fake_atomic):
            self.mod._write_config({"bridge_ip": "192.168.1.10"})
        writer.assert_called_once()
        # The path + payload are forwarded to the atomic writer.
        args, _ = writer.call_args
        self.assertEqual(args[1], {"bridge_ip": "192.168.1.10"})

    def test_write_config_swallows_writer_error(self):
        fake_atomic = types.ModuleType("core.atomic_io")
        fake_atomic._atomic_write_json = mock.MagicMock(
            side_effect=RuntimeError("disk full"))
        with mock.patch.object(self.mod.os, "makedirs"), \
             inject_core_submodule("atomic_io", fake_atomic):
            # Must not raise.
            self.mod._write_config({"bridge_ip": "x"})


class HueAutodiscoverTests(_HueBase):
    def _fake_requests(self, *, json_data=None, raises=False, status_raises=False):
        req = types.ModuleType("requests")

        class _Resp:
            def __init__(self):
                self._json = json_data

            def raise_for_status(self):
                if status_raises:
                    raise RuntimeError("HTTP 500")

            def json(self):
                return self._json

        if raises:
            req.get = mock.MagicMock(side_effect=RuntimeError("offline"))
        else:
            req.get = mock.MagicMock(return_value=_Resp())
        return req

    def test_autodiscover_returns_internal_ip(self):
        req = self._fake_requests(
            json_data=[{"internalipaddress": "192.168.1.42"}])
        import sys as _sys
        with mock.patch.dict(_sys.modules, {"requests": req}):
            ip = self.mod._autodiscover_bridge_ip()
        self.assertEqual(ip, "192.168.1.42")

    def test_autodiscover_no_requests(self):
        real_import = __import__

        def _blocked(name, *a, **k):
            if name == "requests":
                raise ImportError("no requests")
            return real_import(name, *a, **k)

        import sys as _sys
        with mock.patch.dict(_sys.modules, {"requests": None}), \
             mock.patch("builtins.__import__", side_effect=_blocked):
            self.assertIsNone(self.mod._autodiscover_bridge_ip())

    def test_autodiscover_request_error(self):
        req = self._fake_requests(raises=True)
        import sys as _sys
        with mock.patch.dict(_sys.modules, {"requests": req}):
            self.assertIsNone(self.mod._autodiscover_bridge_ip())

    def test_autodiscover_empty_list(self):
        req = self._fake_requests(json_data=[])
        import sys as _sys
        with mock.patch.dict(_sys.modules, {"requests": req}):
            self.assertIsNone(self.mod._autodiscover_bridge_ip())

    def test_autodiscover_missing_ip_key(self):
        req = self._fake_requests(json_data=[{"id": "abc"}])  # no internalipaddress
        import sys as _sys
        with mock.patch.dict(_sys.modules, {"requests": req}):
            self.assertIsNone(self.mod._autodiscover_bridge_ip())


class HueThreadedConnectTests(_HueBase):
    def test_threaded_connect_success(self):
        bridge = _FakeBridge({})
        fake = make_fake_phue(bridge=bridge)
        res = self.mod._threaded_connect(fake, "192.168.1.10", timeout=2.0)
        self.assertIs(res["bridge"], bridge)
        self.assertIsNone(res["error"])
        self.assertFalse(res["timed_out"])
        self.assertEqual(fake._seen["ip"], "192.168.1.10")

    def test_threaded_connect_ctor_raises_records_error(self):
        fake = make_fake_phue(ctor_raises=_PhueRegistrationException("press button"))
        res = self.mod._threaded_connect(fake, "192.168.1.10", timeout=2.0)
        self.assertIsNone(res["bridge"])
        self.assertIsInstance(res["error"], _PhueRegistrationException)
        self.assertFalse(res["timed_out"])

    def test_threaded_connect_inner_connect_raises_is_swallowed(self):
        # Bridge() ctor succeeds but the secondary b.connect() raises — the
        # inner try/except swallows it and still returns the bridge.
        class _Bridge:
            def __init__(self, ip):
                pass
            def connect(self):
                raise RuntimeError("already connected")
        fake = types.ModuleType("phue")
        fake.Bridge = _Bridge
        fake.PhueRegistrationException = _PhueRegistrationException
        res = self.mod._threaded_connect(fake, "192.168.1.10", timeout=2.0)
        self.assertIsInstance(res["bridge"], _Bridge)
        self.assertIsNone(res["error"])

    def test_threaded_connect_timeout(self):
        # A Bridge ctor that blocks past the timeout → timed_out True. Use a
        # short timeout and an event we never set; the worker is a daemon so the
        # test won't hang.
        block = threading.Event()
        mod_phue = types.ModuleType("phue")

        def _Bridge(ip):
            block.wait(5.0)   # longer than the join timeout below
            return _FakeBridge({})

        mod_phue.Bridge = _Bridge
        mod_phue.PhueRegistrationException = _PhueRegistrationException
        try:
            res = self.mod._threaded_connect(mod_phue, "192.168.1.10", timeout=0.05)
            self.assertTrue(res["timed_out"])
            self.assertIsNone(res["bridge"])
        finally:
            block.set()   # release the worker so it can exit


class HueGetBridgeTests(_HueBase):
    def test_get_bridge_none_without_phue(self):
        with mock.patch.object(self.mod, "_phue", return_value=None):
            self.assertIsNone(self.mod._get_bridge())

    def test_get_bridge_returns_cached(self):
        self._use_phue(make_fake_phue())
        sentinel = _FakeBridge({})
        self.mod._bridge_cache["bridge"] = sentinel
        self.mod._bridge_cache["fetched_at"] = self.mod.time.monotonic()
        self.assertIs(self.mod._get_bridge(), sentinel)

    def test_get_bridge_no_ip_and_discovery_fails(self):
        self._use_phue(make_fake_phue())
        self.mod._bridge_cache["bridge"] = None
        with mock.patch.object(self.mod, "_read_config", return_value={}), \
             mock.patch.object(self.mod, "_autodiscover_bridge_ip",
                               return_value=None):
            self.assertIsNone(self.mod._get_bridge())
        self.assertEqual(self.mod._pending["last_error"], "no bridge IP")

    def test_get_bridge_autodiscovers_and_persists_ip(self):
        bridge = _FakeBridge({})
        self._use_phue(make_fake_phue(bridge=bridge))
        self.mod._bridge_cache["bridge"] = None
        wrote = {}
        with mock.patch.object(self.mod, "_read_config", return_value={}), \
             mock.patch.object(self.mod, "_autodiscover_bridge_ip",
                               return_value="192.168.1.55"), \
             mock.patch.object(self.mod, "_write_config",
                               side_effect=lambda cfg: wrote.update(cfg)), \
             mock.patch.object(self.mod, "_threaded_connect",
                               return_value={"bridge": bridge, "error": None,
                                             "timed_out": False}):
            out = self.mod._get_bridge()
        self.assertIs(out, bridge)
        self.assertEqual(wrote.get("bridge_ip"), "192.168.1.55")

    def test_get_bridge_timeout_sets_awaiting_and_schedules_retry(self):
        self._use_phue(make_fake_phue())
        self.mod._bridge_cache["bridge"] = None
        sched = mock.MagicMock()
        with mock.patch.object(self.mod, "_read_config",
                               return_value={"bridge_ip": "192.168.1.10"}), \
             mock.patch.object(self.mod, "_threaded_connect",
                               return_value={"bridge": None, "error": None,
                                             "timed_out": True}), \
             mock.patch.object(self.mod, "_autodiscover_bridge_ip") as disco, \
             mock.patch.object(self.mod, "_write_config") as writer, \
             mock.patch.object(self.mod, "_schedule_retry", sched):
            self.assertIsNone(self.mod._get_bridge())
        self.assertTrue(self.mod._awaiting_button())
        self.assertEqual(self.mod._pending["last_error"], "connect timed out")
        sched.assert_called_once()
        # Timeout means "press the button" — the stored IP is likely good, so
        # the self-heal path must NOT rediscover or touch the config.
        disco.assert_not_called()
        writer.assert_not_called()

    def test_get_bridge_registration_error_sets_awaiting(self):
        self._use_phue(make_fake_phue())
        self.mod._bridge_cache["bridge"] = None
        sched = mock.MagicMock()
        err = _PhueRegistrationException("press the button")
        with mock.patch.object(self.mod, "_read_config",
                               return_value={"bridge_ip": "192.168.1.10"}), \
             mock.patch.object(self.mod, "_threaded_connect",
                               return_value={"bridge": None, "error": err,
                                             "timed_out": False}), \
             mock.patch.object(self.mod, "_autodiscover_bridge_ip") as disco, \
             mock.patch.object(self.mod, "_write_config") as writer, \
             mock.patch.object(self.mod, "_schedule_retry", sched):
            self.assertIsNone(self.mod._get_bridge())
        self.assertTrue(self.mod._awaiting_button())
        sched.assert_called_once()
        # Registration error = bridge reached, button not pressed — the stored
        # IP is correct, so no rediscovery and no config write.
        disco.assert_not_called()
        writer.assert_not_called()

    def test_get_bridge_generic_error_no_awaiting(self):
        self._use_phue(make_fake_phue())
        self.mod._bridge_cache["bridge"] = None
        self.mod._pending["awaiting_button_until"] = 0.0
        with mock.patch.object(self.mod, "_read_config",
                               return_value={"bridge_ip": "192.168.1.10"}), \
             mock.patch.object(self.mod, "_threaded_connect",
                               return_value={"bridge": None,
                                             "error": RuntimeError("conn refused"),
                                             "timed_out": False}), \
             mock.patch.object(self.mod, "_autodiscover_bridge_ip",
                               return_value=None), \
             mock.patch.object(self.mod, "_write_config") as writer, \
             mock.patch.object(self.mod, "_schedule_retry") as sched:
            self.assertIsNone(self.mod._get_bridge())
        self.assertFalse(self.mod._awaiting_button())
        self.assertIn("conn refused", self.mod._pending["last_error"])
        sched.assert_not_called()
        # Transient-outage safety: the stored IP is syntactically valid and
        # discovery found nothing better — the config must survive untouched.
        writer.assert_not_called()

    def test_get_bridge_self_heals_poisoned_config_via_discovery(self):
        # The stored bridge_ip is the audit's literal poison "test": the first
        # connect fails generically, discovery answers with the real bridge →
        # the config is repaired in place and the connect retried once.
        bridge = _FakeBridge({})
        self._use_phue(make_fake_phue())
        self.mod._bridge_cache["bridge"] = None
        wrote = {}

        def _connect(phue_mod, ip, timeout=None):
            if ip == "test":
                return {"bridge": None, "error": RuntimeError("conn refused"),
                        "timed_out": False}
            return {"bridge": bridge, "error": None, "timed_out": False}

        with mock.patch.object(self.mod, "_read_config",
                               return_value={"bridge_ip": "test"}), \
             mock.patch.object(self.mod, "_autodiscover_bridge_ip",
                               return_value="192.168.1.55"), \
             mock.patch.object(self.mod, "_write_config",
                               side_effect=lambda cfg: wrote.update(cfg)), \
             mock.patch.object(self.mod, "_threaded_connect",
                               side_effect=_connect):
            out = self.mod._get_bridge()
        self.assertIs(out, bridge)
        self.assertEqual(wrote.get("bridge_ip"), "192.168.1.55")
        self.assertIs(self.mod._bridge_cache["bridge"], bridge)
        self.assertIsNone(self.mod._pending["last_error"])

    def test_get_bridge_persists_discovered_ip_even_if_retry_fails(self):
        # Discovery found a different bridge but it isn't connecting yet —
        # the discovered address still replaces the failing stored one.
        self._use_phue(make_fake_phue())
        self.mod._bridge_cache["bridge"] = None
        wrote = {}
        with mock.patch.object(self.mod, "_read_config",
                               return_value={"bridge_ip": "test"}), \
             mock.patch.object(self.mod, "_autodiscover_bridge_ip",
                               return_value="192.168.1.55"), \
             mock.patch.object(self.mod, "_write_config",
                               side_effect=lambda cfg: wrote.update(cfg)), \
             mock.patch.object(self.mod, "_threaded_connect",
                               return_value={"bridge": None,
                                             "error": RuntimeError("conn refused"),
                                             "timed_out": False}):
            self.assertIsNone(self.mod._get_bridge())
        self.assertEqual(wrote.get("bridge_ip"), "192.168.1.55")
        self.assertIn("not connecting yet", self.mod._pending["last_error"])

    def test_get_bridge_drops_garbage_ip_when_discovery_fails(self):
        # No discovery answer AND the stored value is syntactic garbage →
        # the key is dropped so the next call rediscovers from scratch.
        self._use_phue(make_fake_phue())
        self.mod._bridge_cache["bridge"] = None
        writes = []
        with mock.patch.object(self.mod, "_read_config",
                               return_value={"bridge_ip": "test"}), \
             mock.patch.object(self.mod, "_autodiscover_bridge_ip",
                               return_value=None), \
             mock.patch.object(self.mod, "_write_config",
                               side_effect=lambda cfg: writes.append(dict(cfg))), \
             mock.patch.object(self.mod.socket, "getaddrinfo",
                               side_effect=OSError("no resolution")), \
             mock.patch.object(self.mod, "_threaded_connect",
                               return_value={"bridge": None,
                                             "error": RuntimeError("conn refused"),
                                             "timed_out": False}):
            self.assertIsNone(self.mod._get_bridge())
        self.assertEqual(writes, [{}])

    def test_get_bridge_success_caches_and_clears_pending(self):
        bridge = _FakeBridge({})
        self._use_phue(make_fake_phue(bridge=bridge))
        self.mod._bridge_cache["bridge"] = None
        self.mod._pending["awaiting_button_until"] = self.mod.time.time() + 99
        with mock.patch.object(self.mod, "_read_config",
                               return_value={"bridge_ip": "192.168.1.10"}), \
             mock.patch.object(self.mod, "_threaded_connect",
                               return_value={"bridge": bridge, "error": None,
                                             "timed_out": False}):
            out = self.mod._get_bridge()
        self.assertIs(out, bridge)
        self.assertIs(self.mod._bridge_cache["bridge"], bridge)
        self.assertEqual(self.mod._pending["awaiting_button_until"], 0.0)
        self.assertIsNone(self.mod._pending["last_error"])


class HueScheduleRetryTests(_HueBase):
    def test_schedule_retry_calls_scheduler(self):
        sched = types.ModuleType("core.scheduler")
        sched.schedule_once = mock.MagicMock()
        # inject_core_submodule sets the attr on the live `core` package too, so
        # `from core import scheduler` resolves OUR fake even when another test
        # already imported the real core.scheduler.
        with inject_core_submodule("scheduler", sched):
            self.mod._schedule_retry()
        sched.schedule_once.assert_called_once()
        _a, kw = sched.schedule_once.call_args
        self.assertEqual(kw.get("action"), "hue_retry_connect")

    def test_schedule_retry_no_scheduler_module(self):
        real_import = __import__

        def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
            # `from core import scheduler` calls __import__("core", ...,
            # fromlist=["scheduler"]); the submodule fallback calls
            # __import__("core.scheduler"). Block BOTH resolution paths so the
            # import fails regardless of whether core.scheduler is cached.
            if name == "core.scheduler" or (name == "core"
                                            and "scheduler" in (fromlist or ())):
                raise ImportError("no scheduler")
            return real_import(name, globals, locals, fromlist, level)

        # obj=None removes the module AND the core.scheduler attribute so the
        # blocked __import__ is actually consulted and the import fails.
        with inject_core_submodule("scheduler", None), \
             mock.patch("builtins.__import__", side_effect=_blocked):
            self.mod._schedule_retry()   # must not raise

    def test_schedule_retry_swallows_scheduler_error(self):
        sched = types.ModuleType("core.scheduler")
        sched.schedule_once = mock.MagicMock(side_effect=RuntimeError("no jobstore"))
        with inject_core_submodule("scheduler", sched):
            self.mod._schedule_retry()   # must not raise


class HueRetryConnectActionTests(_HueBase):
    def test_retry_connect_success_message(self):
        with mock.patch.object(self.mod, "_get_bridge",
                               return_value=_FakeBridge({})):
            out = self.actions["hue_retry_connect"]("")
        self.assertIn("connected", out.lower())

    def test_retry_connect_still_awaiting(self):
        self.mod._pending["awaiting_button_until"] = self.mod.time.time() + 30
        with mock.patch.object(self.mod, "_get_bridge", return_value=None):
            out = self.actions["hue_retry_connect"]("")
        self.assertIn("still waiting", out.lower())


class HueLightByNameTests(_HueBase):
    def test_light_by_name_exact(self):
        light = _FakeLight(name="Desk")
        bridge = _FakeBridge({"Desk": light})
        self.assertIs(self.mod._light_by_name(bridge, "Desk"), light)

    def test_light_by_name_none_bridge_or_name(self):
        self.assertIsNone(self.mod._light_by_name(None, "Desk"))
        self.assertIsNone(self.mod._light_by_name(_FakeBridge({}), ""))

    def test_light_by_name_enum_raises_returns_none(self):
        bridge = _FakeBridge({}, raise_on_enum=True)
        self.assertIsNone(self.mod._light_by_name(bridge, "Desk"))

    def test_light_by_name_unknown(self):
        bridge = _FakeBridge({"Desk": _FakeLight(name="Desk")})
        self.assertIsNone(self.mod._light_by_name(bridge, "Ghost"))


class HueListDevicesTests(_HueBase):
    def test_list_devices_maps_caps(self):
        color_light = _FakeLight(name="Color", colormode="xy")
        ct_light = _FakeLight(name="White", colormode="ct")
        bridge = _FakeBridge({"Color": color_light, "White": ct_light})
        with mock.patch.object(self.mod, "_get_bridge", return_value=bridge):
            devs = self.mod.list_devices()
        by_name = {d["name"]: d for d in devs}
        self.assertIn("color", by_name["Color"]["capabilities"])
        self.assertIn("color_temperature", by_name["White"]["capabilities"])
        self.assertIn("dim", by_name["Color"]["capabilities"])
        self.assertEqual(by_name["Color"]["brand"], "Philips Hue")
        self.assertEqual(by_name["Color"]["native_id"], 7)

    def test_list_devices_enum_failure_returns_empty(self):
        bridge = _FakeBridge({}, raise_on_enum=True)
        with mock.patch.object(self.mod, "_get_bridge", return_value=bridge):
            self.assertEqual(self.mod.list_devices(), [])

    def test_list_devices_colormode_access_swallowed(self):
        # A light whose colormode attribute raises still yields on_off + dim.
        class _BadLight:
            name = "Weird"
            light_id = 3
            @property
            def colormode(self):
                raise RuntimeError("colormode boom")
        bridge = _FakeBridge({"Weird": _BadLight()})
        with mock.patch.object(self.mod, "_get_bridge", return_value=bridge):
            devs = self.mod.list_devices()
        self.assertEqual(devs[0]["capabilities"], ["on_off"])

    def test_hue_list_action_lists_names(self):
        bridge = _FakeBridge({"Desk": _FakeLight(name="Desk")})
        with mock.patch.object(self.mod, "_get_bridge", return_value=bridge):
            out = self.actions["hue_list"]("")
        self.assertIn("Desk", out)
        self.assertIn("1 Hue bulb", out)


class HueGetStateTests(_HueBase):
    def test_get_state_reports_on_brightness_reachable(self):
        light = _FakeLight(name="Desk")
        light.on = True
        light.brightness = 254
        bridge = _FakeBridge({"Desk": light})
        with mock.patch.object(self.mod, "_get_bridge", return_value=bridge):
            st = self.mod.get_state({"name": "Desk"})
        self.assertTrue(st["on"])
        self.assertEqual(st["brightness"], 100)   # 254 → 100%
        self.assertTrue(st["reachable"])

    def test_get_state_read_failure_returns_error(self):
        class _BadLight:
            name = "Desk"
            @property
            def on(self):
                raise RuntimeError("read boom")
        bridge = _FakeBridge({"Desk": _BadLight()})
        with mock.patch.object(self.mod, "_get_bridge", return_value=bridge):
            st = self.mod.get_state({"name": "Desk"})
        self.assertIn("state read failed", st["error"])

    def test_get_state_bridge_none(self):
        with mock.patch.object(self.mod, "_get_bridge", return_value=None):
            st = self.mod.get_state({"name": "Desk"})
        self.assertIn("not connected", st["error"])

    def test_get_state_bulb_not_found_on_connected_bridge(self):
        bridge = _FakeBridge({})   # connected, but no such bulb
        with mock.patch.object(self.mod, "_get_bridge", return_value=bridge):
            st = self.mod.get_state({"name": "Ghost"})
        self.assertIn("not found on bridge", st["error"])


class HueSetStateBranchTests(_HueBase):
    def _bridge_with(self, light):
        return _FakeBridge({light.name: light})

    def test_set_state_bulb_not_found_on_connected_bridge(self):
        bridge = _FakeBridge({})   # connected, but no such bulb
        with mock.patch.object(self.mod, "_get_bridge", return_value=bridge):
            res = self.mod.set_state({"name": "Ghost"}, on=True)
        self.assertIn("not found on bridge", res["error"])

    def test_set_color_applies_xy(self):
        light = _FakeLight(name="Desk")
        with mock.patch.object(self.mod, "_get_bridge",
                               return_value=self._bridge_with(light)):
            res = self.mod.set_state({"name": "Desk"}, color=(255, 0, 0))
        self.assertIn("color", res["applied"])
        self.assertIsNotNone(light.xy)

    def test_set_color_white_only_bulb_skips_silently(self):
        # Assigning .xy raises on a white-only bulb → skill swallows it. Use a
        # standalone fake (NOT subclassing _FakeLight, whose __init__ would
        # assign .xy and trip the raising setter before the test runs).
        class _WhiteOnly:
            name = "Desk"
            type = "Dimmable light"
            colormode = "none"
            @property
            def xy(self):
                return None
            @xy.setter
            def xy(self, _v):
                raise RuntimeError("no color support")
        light = _WhiteOnly()
        with mock.patch.object(self.mod, "_get_bridge",
                               return_value=self._bridge_with(light)):
            res = self.mod.set_state({"name": "Desk"}, color=(255, 0, 0))
        self.assertTrue(res["ok"])
        self.assertNotIn("color", res["applied"])

    def test_set_color_temperature_applied_on_extended_bulb(self):
        light = _FakeLight(name="Desk", ltype="Extended color light",
                           colormode="ct")
        with mock.patch.object(self.mod, "_get_bridge",
                               return_value=self._bridge_with(light)):
            res = self.mod.set_state({"name": "Desk"}, color_temperature=4000)
        self.assertEqual(res["applied"]["color_temperature_k"], 4000)
        self.assertIsNotNone(light.colortemp)

    def test_set_color_temperature_skipped_for_dimmable_only(self):
        light = _FakeLight(name="Desk", ltype="Dimmable light", colormode="none")
        with mock.patch.object(self.mod, "_get_bridge",
                               return_value=self._bridge_with(light)):
            res = self.mod.set_state({"name": "Desk"}, color_temperature=4000)
        self.assertEqual(res["applied"]["color_temperature_skipped"],
                         "bulb_lacks_ct_range")

    def test_set_color_temperature_skipped_for_hs_color_named_light(self):
        # colormode 'hs' + a type without 'color' in it → ct unsupported.
        light = _FakeLight(name="Desk", ltype="Lightstrip", colormode="hs")
        with mock.patch.object(self.mod, "_get_bridge",
                               return_value=self._bridge_with(light)):
            res = self.mod.set_state({"name": "Desk"}, color_temperature=4000)
        self.assertEqual(res["applied"]["color_temperature_skipped"],
                         "bulb_lacks_ct_range")

    def test_set_color_temperature_set_raises_records_unsupported(self):
        # ct is "supported" by type/colormode, but the actual setter raises →
        # recorded as 'unsupported: ...'. Standalone fake so __init__ doesn't
        # pre-trip the raising setter.
        class _CtBoom:
            name = "Desk"
            type = "Extended color light"
            colormode = "ct"
            @property
            def colortemp(self):
                return None
            @colortemp.setter
            def colortemp(self, _v):
                raise RuntimeError("ct nope")
        light = _CtBoom()
        with mock.patch.object(self.mod, "_get_bridge",
                               return_value=self._bridge_with(light)):
            res = self.mod.set_state({"name": "Desk"}, color_temperature=4000)
        self.assertIn("unsupported", res["applied"]["color_temperature_skipped"])

    def test_set_state_midway_exception_returns_partial(self):
        # Setting .on raises → mid-way failure path with partial dict. Standalone
        # fake whose `on` setter always raises.
        class _OnBoom:
            name = "Desk"
            type = "Extended color light"
            colormode = "xy"
            @property
            def on(self):
                return False
            @on.setter
            def on(self, _v):
                raise RuntimeError("on nope")
        light = _OnBoom()
        with mock.patch.object(self.mod, "_get_bridge",
                               return_value=self._bridge_with(light)):
            res = self.mod.set_state({"name": "Desk"}, on=True)
        self.assertIn("set_state failed mid-way", res["error"])
        self.assertIn("partial", res)


class HueSetBridgeIpTests(_HueBase):
    def test_set_bridge_ip_empty_arg_help(self):
        self.assertIn("Format", self.actions["hue_set_bridge_ip"](""))

    def test_set_bridge_ip_persists_and_clears_cache(self):
        wrote = {}
        self.mod._bridge_cache["bridge"] = _FakeBridge({})
        with mock.patch.object(self.mod, "_read_config", return_value={}), \
             mock.patch.object(self.mod, "_write_config",
                               side_effect=lambda cfg: wrote.update(cfg)):
            out = self.actions["hue_set_bridge_ip"]("192.168.1.77")
        self.assertIn("192.168.1.77", out)
        self.assertEqual(wrote.get("bridge_ip"), "192.168.1.77")
        # Cache invalidated so the next call reconnects.
        self.assertIsNone(self.mod._bridge_cache["bridge"])

    def test_set_bridge_ip_rejects_garbage_without_persisting(self):
        # "test" (the audit's live poison), an out-of-range quad and a
        # truncated quad must all be rejected BEFORE any config write. DNS is
        # stubbed to fail so a wildcard-resolving LAN can't skew the result.
        saved_cache = dict(self.mod._bridge_cache)
        with mock.patch.object(self.mod, "_write_config") as writer, \
             mock.patch.object(self.mod.socket, "getaddrinfo",
                               side_effect=OSError("no resolution")):
            for bad in ("test", "999.999.1.1", "192.168.1"):
                out = self.actions["hue_set_bridge_ip"](bad)
                self.assertIn("not saved", out)
                self.assertIn(bad, out)
        writer.assert_not_called()
        self.assertEqual(self.mod._bridge_cache, saved_cache)

    def test_set_bridge_ip_resolvable_hostname_persists(self):
        wrote = {}
        with mock.patch.object(self.mod, "_read_config", return_value={}), \
             mock.patch.object(self.mod, "_write_config",
                               side_effect=lambda cfg: wrote.update(cfg)), \
             mock.patch.object(self.mod.socket, "getaddrinfo",
                               return_value=[("stub",)]):
            out = self.actions["hue_set_bridge_ip"]("hue-bridge.lan")
        self.assertIn("hue-bridge.lan", out)
        self.assertEqual(wrote.get("bridge_ip"), "hue-bridge.lan")

    def test_set_bridge_ip_unresolvable_hostname_rejected(self):
        with mock.patch.object(self.mod, "_write_config") as writer, \
             mock.patch.object(self.mod.socket, "getaddrinfo",
                               side_effect=OSError("NXDOMAIN")):
            out = self.actions["hue_set_bridge_ip"]("hue-bridge.lan")
        self.assertIn("not saved", out)
        writer.assert_not_called()


class HueBridgeHostValidatorTests(_HueBase):
    def test_literal_ip_accepted_without_dns(self):
        # A dotted-quad short-circuits via ipaddress — DNS must not be hit.
        with mock.patch.object(self.mod.socket, "getaddrinfo",
                               side_effect=OSError("DNS must not be hit")):
            self.assertTrue(self.mod._valid_bridge_host("192.168.1.10"))
            self.assertTrue(self.mod._valid_bridge_host("::1"))

    def test_garbage_rejected_when_unresolvable(self):
        with mock.patch.object(self.mod.socket, "getaddrinfo",
                               side_effect=OSError("no resolution")):
            for bad in ("test", "999.999.1.1", "192.168.1"):
                self.assertFalse(self.mod._valid_bridge_host(bad))

    def test_resolvable_hostname_accepted(self):
        with mock.patch.object(self.mod.socket, "getaddrinfo",
                               return_value=[("ok",)]):
            self.assertTrue(self.mod._valid_bridge_host("hue-bridge.lan"))


class ActionSmokeDenylistTests(unittest.TestCase):
    """Source-scan invariant: the action sweep must never invoke
    hue_set_bridge_ip — it would persist the benign sweep arg ("test") as the
    bridge IP and report a false OK. tools/action_smoke.py mutates os.environ
    at import time, so we scan its SOURCE via ast instead of importing it."""

    def test_hue_set_bridge_ip_in_action_smoke_denylist(self):
        import ast
        import os
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        path = os.path.join(root, "tools", "action_smoke.py")
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        names = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Name)
                            and target.id == "_DENYLIST_NAMES"):
                        names = ast.literal_eval(node.value)
        self.assertIsNotNone(
            names, "_DENYLIST_NAMES literal not found in tools/action_smoke.py")
        self.assertIn("hue_set_bridge_ip", names)


if __name__ == "__main__":
    unittest.main()
