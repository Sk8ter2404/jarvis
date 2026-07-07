"""Logic tests for skills/sh_ecobee.py (Ecobee thermostat controller).

Wraps the optional `pyecobee` library behind an interactive PIN-based OAuth
flow. Coverage:
  * is_available gating on lib presence AND a configured api_key,
  * _pyecobee import (present via injected fake / absent),
  * _read_config / _load_tokens (present / absent / malformed) and _save_tokens
    (writes the three token fields; swallows IO errors),
  * _run_with_timeout ok / error / timed-out branches,
  * _get_service construction, the no-api-key + missing-EcobeeService guards,
    the refresh-tokens success/error/timeout caching policy, and cache hit,
  * _fetch_thermostats success + failure, list_devices mapping,
  * _match_thermostat by identifier and by name (+ miss),
  * get_state's tenths→degrees translation (+ not-found / not-initialised /
    read exception),
  * set_state's on/off→auto/off mapping (via update_thermostats), whole-degree
    keyword-only set_hold call shape, and the mode / hold failure partials,
  * the two-step PIN flow: _do_request_pin guards + happy path, request_pin
    action, complete_setup (no lib / no key / no pending token / success /
    failure), the authorize alias, and the interactive wizard,
  * register wiring.

`pyecobee` is NOT a CI dependency, so it is ALWAYS injected as a fake — never
imported for real — keeping the suite deterministic and offline. Module
globals (_state) are reset in tearDown.
"""
from __future__ import annotations

import contextlib
import io
import sys
import threading
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules, restoring prior state
    (including absence) on exit so tests stay isolated."""
    saved: dict[str, object] = {}
    for name, obj in mods.items():
        saved[name] = sys.modules.get(name, _SENTINEL)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
    try:
        yield
    finally:
        for name, prev in saved.items():
            if prev is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


@contextlib.contextmanager
def block_import(*names):
    """Force ``import <name>`` to raise ImportError inside the block."""
    real_import = __import__
    blocked = set(names)

    def _fake_import(name, *args, **kwargs):
        if name in blocked or name.split(".")[0] in blocked:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    saved: dict[str, object] = {}
    for name in blocked:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    try:
        with mock.patch("builtins.__import__", side_effect=_fake_import):
            yield
    finally:
        for name, mod in saved.items():
            sys.modules[name] = mod


class _FakeThermostat:
    def __init__(self, name="Main", identifier="abc123", model="nikeSmart",
                 actual=712, cool=750, heat=680, hvac="heat"):
        self.name = name
        self.identifier = identifier
        self.model_number = model
        self.runtime = types.SimpleNamespace(
            actual_temperature=actual, desired_cool=cool, desired_heat=heat)
        self.settings = types.SimpleNamespace(hvac_mode=hvac)


def _fake_pyecobee(service=None, omit_service=False):
    """A fake pyecobee module exposing EcobeeService + Selection/SelectionType.
    ``service`` injects a pre-built service the constructor returns."""
    mod = types.ModuleType("pyecobee")

    captured = {}

    class _EcobeeService:
        def __new__(cls, *a, **k):
            if service is not None:
                captured["kwargs"] = k
                return service
            return super().__new__(cls)

        def __init__(self, *a, **k):
            captured["kwargs"] = k
            self.access_token = k.get("access_token") or ""
            self.refresh_token = k.get("refresh_token") or ""
            self.authorization_token = k.get("authorization_token") or ""

    if not omit_service:
        mod.EcobeeService = _EcobeeService

    class _Selection:
        def __init__(self, **k):
            self.kw = k

    class _SelType:
        REGISTERED = types.SimpleNamespace(value="registered")
        THERMOSTATS = types.SimpleNamespace(value="thermostats")

    class _Settings:
        def __init__(self, **k):
            self.kw = k

    class _Thermostat:
        def __init__(self, **k):
            self.kw = k

    mod.Selection = _Selection
    mod.SelectionType = _SelType
    mod.Settings = _Settings
    mod.Thermostat = _Thermostat
    mod._captured = captured
    return mod


class _EcobeeBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_ecobee")
        self.addCleanup(self._reset_state)

    def _reset_state(self):
        self.mod._state["service"] = None
        self.mod._state["fetched_at"] = 0.0
        self.mod._state["thermostats"] = []


# ─── availability + lib import ───────────────────────────────────────────
class EcobeeAvailabilityTests(_EcobeeBase):
    def test_unavailable_without_lib(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_unavailable_without_api_key(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=object()), \
             mock.patch.object(self.mod, "_read_config", return_value={}):
            self.assertFalse(self.mod.is_available())

    def test_available_with_lib_and_key(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=object()), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}):
            self.assertTrue(self.mod.is_available())

    def test_pyecobee_none_when_absent(self):
        with block_import("pyecobee"):
            self.assertIsNone(self.mod._pyecobee())

    def test_pyecobee_returns_module_when_present(self):
        with inject_modules(pyecobee=_fake_pyecobee()):
            self.assertIsNotNone(self.mod._pyecobee())


# ─── config / token IO ───────────────────────────────────────────────────
class EcobeeIOTests(_EcobeeBase):
    def test_read_config_missing_and_valid_and_bad(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._read_config(), {})
        m = mock.mock_open(read_data='{"api_key": "k"}')
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._read_config(), {"api_key": "k"})
        bad = mock.mock_open(read_data="{nope")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", bad):
            self.assertEqual(self.mod._read_config(), {})

    def test_load_tokens_missing_and_valid_and_bad(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._load_tokens(), {})
        m = mock.mock_open(read_data='{"refresh_token": "rt"}')
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._load_tokens(), {"refresh_token": "rt"})
        bad = mock.mock_open(read_data="{nope")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", bad):
            self.assertEqual(self.mod._load_tokens(), {})

    def test_save_tokens_writes_three_fields(self):
        svc = types.SimpleNamespace(access_token="at", refresh_token="rt",
                                    authorization_token="auth")
        captured = {}
        def _cap(path, data, **kw):
            captured["data"] = data
        with mock.patch("core.atomic_io._atomic_write_json", side_effect=_cap):
            self.mod._save_tokens(svc)
        self.assertEqual(captured["data"]["access_token"], "at")
        self.assertEqual(captured["data"]["refresh_token"], "rt")
        self.assertEqual(captured["data"]["authorization_token"], "auth")

    def test_save_tokens_swallows_errors(self):
        svc = types.SimpleNamespace(access_token="at", refresh_token="rt",
                                    authorization_token="auth")
        with mock.patch("builtins.open", side_effect=OSError("ro fs")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.mod._save_tokens(svc)   # no exception


# ─── _run_with_timeout ───────────────────────────────────────────────────
class EcobeeRunWithTimeoutTests(_EcobeeBase):
    def test_ok_branch(self):
        res = self.mod._run_with_timeout(lambda: None, timeout=2.0)
        self.assertTrue(res["ok"])
        self.assertFalse(res["timed_out"])

    def test_error_branch(self):
        def _boom():
            raise RuntimeError("refresh failed")
        res = self.mod._run_with_timeout(_boom, timeout=2.0)
        self.assertFalse(res["ok"])
        self.assertIsInstance(res["error"], RuntimeError)

    def test_timed_out_branch(self):
        gate = threading.Event()

        def _block():
            gate.wait(5.0)
        try:
            res = self.mod._run_with_timeout(_block, timeout=0.1)
            self.assertTrue(res["timed_out"])
        finally:
            gate.set()


# ─── _get_service ────────────────────────────────────────────────────────
class EcobeeGetServiceTests(_EcobeeBase):
    def test_none_when_lib_absent(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=None):
            self.assertIsNone(self.mod._get_service())

    def test_cache_hit_within_ttl(self):
        sentinel = object()
        self.mod._state["service"] = sentinel
        self.mod._state["fetched_at"] = self.mod.time.monotonic()
        with mock.patch.object(self.mod, "_pyecobee",
                               return_value=_fake_pyecobee()):
            self.assertIs(self.mod._get_service(), sentinel)

    def test_none_without_api_key(self):
        with mock.patch.object(self.mod, "_pyecobee",
                               return_value=_fake_pyecobee()), \
             mock.patch.object(self.mod, "_read_config", return_value={}):
            self.assertIsNone(self.mod._get_service())

    def test_none_when_ecobeeservice_missing(self):
        with mock.patch.object(self.mod, "_pyecobee",
                               return_value=_fake_pyecobee(omit_service=True)), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_load_tokens", return_value={}):
            self.assertIsNone(self.mod._get_service())

    def test_no_refresh_token_caches_service_without_refresh(self):
        # No refresh token → skip the exchange, cache the service as-is.
        svc = mock.Mock()
        svc.refresh_token = ""
        fake = _fake_pyecobee(service=svc)
        with mock.patch.object(self.mod, "_pyecobee", return_value=fake), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_load_tokens", return_value={}):
            got = self.mod._get_service()
        self.assertIs(got, svc)
        svc.refresh_tokens.assert_not_called()

    def test_refresh_success_saves_and_caches(self):
        svc = mock.Mock()
        svc.refresh_token = "rt"
        fake = _fake_pyecobee(service=svc)
        with mock.patch.object(self.mod, "_pyecobee", return_value=fake), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_load_tokens",
                               return_value={"refresh_token": "rt"}), \
             mock.patch.object(self.mod, "_run_with_timeout",
                               return_value={"ok": True, "error": None,
                                             "timed_out": False}), \
             mock.patch.object(self.mod, "_save_tokens") as save:
            got = self.mod._get_service()
        self.assertIs(got, svc)
        save.assert_called_once_with(svc)
        self.assertIs(self.mod._state["service"], svc)

    def test_refresh_timeout_not_cached(self):
        svc = mock.Mock()
        svc.refresh_token = "rt"
        fake = _fake_pyecobee(service=svc)
        with mock.patch.object(self.mod, "_pyecobee", return_value=fake), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_load_tokens",
                               return_value={"refresh_token": "rt"}), \
             mock.patch.object(self.mod, "_run_with_timeout",
                               return_value={"ok": False, "error": None,
                                             "timed_out": True}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.mod._get_service())
        self.assertIsNone(self.mod._state["service"])

    def test_refresh_error_not_cached(self):
        svc = mock.Mock()
        svc.refresh_token = "rt"
        fake = _fake_pyecobee(service=svc)
        with mock.patch.object(self.mod, "_pyecobee", return_value=fake), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_load_tokens",
                               return_value={"refresh_token": "rt"}), \
             mock.patch.object(self.mod, "_run_with_timeout",
                               return_value={"ok": False,
                                             "error": RuntimeError("bad"),
                                             "timed_out": False}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.mod._get_service())

    def test_service_construction_raises_returns_none(self):
        # EcobeeService(...) raising inside the try → caught, None returned.
        fake = _fake_pyecobee()

        def _raise(*a, **k):
            raise RuntimeError("ctor boom")
        fake.EcobeeService = _raise
        with mock.patch.object(self.mod, "_pyecobee", return_value=fake), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_load_tokens", return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.mod._get_service())


# ─── _fetch_thermostats + list_devices ───────────────────────────────────
class EcobeeFetchListTests(_EcobeeBase):
    def test_fetch_thermostats_none_when_no_lib_or_svc(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=None):
            self.assertEqual(self.mod._fetch_thermostats(object()), [])
        with mock.patch.object(self.mod, "_pyecobee",
                               return_value=_fake_pyecobee()):
            self.assertEqual(self.mod._fetch_thermostats(None), [])

    def test_fetch_thermostats_success(self):
        t = _FakeThermostat()
        svc = mock.Mock()
        svc.request_thermostats.return_value = types.SimpleNamespace(
            thermostat_list=[t])
        with mock.patch.object(self.mod, "_pyecobee",
                               return_value=_fake_pyecobee()):
            out = self.mod._fetch_thermostats(svc)
        self.assertEqual(out, [t])

    def test_fetch_thermostats_failure_returns_empty(self):
        svc = mock.Mock()
        svc.request_thermostats.side_effect = RuntimeError("api down")
        with mock.patch.object(self.mod, "_pyecobee",
                               return_value=_fake_pyecobee()), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.mod._fetch_thermostats(svc), [])

    def test_list_devices_empty_without_service(self):
        with mock.patch.object(self.mod, "_get_service", return_value=None):
            self.assertEqual(self.mod.list_devices(), [])

    def test_list_devices_maps_fields(self):
        t = _FakeThermostat(name="Upstairs", identifier="xyz", model="ecobee4")
        with mock.patch.object(self.mod, "_get_service", return_value=object()), \
             mock.patch.object(self.mod, "_fetch_thermostats", return_value=[t]):
            out = self.mod.list_devices()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "Upstairs")
        self.assertEqual(out[0]["native_id"], "xyz")
        self.assertEqual(out[0]["model"], "ecobee4")
        self.assertEqual(out[0]["brand"], "ecobee")
        self.assertIn("thermostat", out[0]["capabilities"])


# ─── _match_thermostat ───────────────────────────────────────────────────
class EcobeeMatchTests(_EcobeeBase):
    def test_match_by_identifier(self):
        t = _FakeThermostat(identifier="abc123")
        with mock.patch.object(self.mod, "_fetch_thermostats", return_value=[t]):
            found = self.mod._match_thermostat(object(), {"native_id": "abc123"})
        self.assertIs(found, t)

    def test_match_by_name_case_insensitive(self):
        t = _FakeThermostat(name="Main Floor")
        with mock.patch.object(self.mod, "_fetch_thermostats", return_value=[t]):
            found = self.mod._match_thermostat(object(), {"name": "main floor"})
        self.assertIs(found, t)

    def test_match_miss_returns_none(self):
        t = _FakeThermostat(name="Main", identifier="abc123")
        with mock.patch.object(self.mod, "_fetch_thermostats", return_value=[t]):
            self.assertIsNone(
                self.mod._match_thermostat(object(), {"name": "Garage"}))


# ─── get_state ───────────────────────────────────────────────────────────
class EcobeeGetStateTests(_EcobeeBase):
    def test_errors_without_service(self):
        with mock.patch.object(self.mod, "_get_service", return_value=None):
            res = self.mod.get_state({"name": "Main"})
        self.assertIn("not initialized", res["error"])

    def test_not_found(self):
        with mock.patch.object(self.mod, "_get_service", return_value=object()), \
             mock.patch.object(self.mod, "_match_thermostat", return_value=None):
            res = self.mod.get_state({"name": "Ghost"})
        self.assertIn("not found", res["error"])

    def test_divides_tenths(self):
        thermo = _FakeThermostat(actual=712, cool=750, heat=680, hvac="heat")
        with mock.patch.object(self.mod, "_get_service", return_value=object()), \
             mock.patch.object(self.mod, "_match_thermostat", return_value=thermo):
            st = self.mod.get_state({"name": "Main"})
        self.assertEqual(st["actual_f"], 71.2)
        self.assertEqual(st["cool_set"], 75.0)
        self.assertEqual(st["heat_set"], 68.0)
        self.assertEqual(st["mode"], "heat")

    def test_missing_runtime_yields_none_fields(self):
        thermo = types.SimpleNamespace(
            runtime=types.SimpleNamespace(actual_temperature=None,
                                          desired_cool=None, desired_heat=None),
            settings=types.SimpleNamespace(hvac_mode=""))
        with mock.patch.object(self.mod, "_get_service", return_value=object()), \
             mock.patch.object(self.mod, "_match_thermostat", return_value=thermo):
            st = self.mod.get_state({"name": "Main"})
        self.assertIsNone(st["actual_f"])
        self.assertIsNone(st["cool_set"])

    def test_read_exception_wrapped(self):
        # getattr(t, "runtime") fine, but accessing a runtime field raises.
        class _BadRuntime:
            @property
            def actual_temperature(self):
                raise RuntimeError("read boom")
        thermo = types.SimpleNamespace(runtime=_BadRuntime(),
                                       settings=types.SimpleNamespace(hvac_mode=""))
        with mock.patch.object(self.mod, "_get_service", return_value=object()), \
             mock.patch.object(self.mod, "_match_thermostat", return_value=thermo):
            res = self.mod.get_state({"name": "Main"})
        self.assertIn("read failed", res["error"])


# ─── set_state ───────────────────────────────────────────────────────────
class EcobeeSetStateTests(_EcobeeBase):
    def _ctx(self, svc, thermo):
        return (mock.patch.object(self.mod, "_get_service", return_value=svc),
                mock.patch.object(self.mod, "_pyecobee",
                                  return_value=_fake_pyecobee()),
                mock.patch.object(self.mod, "_match_thermostat",
                                  return_value=thermo))

    def test_errors_without_service_mentions_authorize(self):
        with mock.patch.object(self.mod, "_get_service", return_value=None):
            res = self.mod.set_state({"name": "Main"}, temperature=70)
        self.assertIn("not initialized", res["error"])
        self.assertIn("ecobee_authorize", res["error"])

    def test_not_found(self):
        svc = mock.Mock()
        with mock.patch.object(self.mod, "_get_service", return_value=svc), \
             mock.patch.object(self.mod, "_pyecobee",
                               return_value=_fake_pyecobee()), \
             mock.patch.object(self.mod, "_match_thermostat", return_value=None):
            res = self.mod.set_state({"name": "Ghost"}, temperature=70)
        self.assertIn("not found", res["error"])

    def test_off_maps_mode_off(self):
        svc = mock.Mock()
        thermo = _FakeThermostat(identifier="abc123")
        c1, c2, c3 = self._ctx(svc, thermo)
        with c1, c2, c3:
            res = self.mod.set_state({"name": "Main"}, on=False)
        self.assertEqual(res["applied"]["mode"], "off")
        svc.update_thermostats.assert_called_once()
        _args, kwargs = svc.update_thermostats.call_args
        self.assertEqual(kwargs["thermostat"].kw["settings"].kw["hvac_mode"],
                         "off")

    def test_on_maps_mode_auto(self):
        svc = mock.Mock()
        thermo = _FakeThermostat(identifier="abc123")
        c1, c2, c3 = self._ctx(svc, thermo)
        with c1, c2, c3:
            res = self.mod.set_state({"name": "Main"}, on=True)
        self.assertEqual(res["applied"]["mode"], "auto")

    def test_explicit_mode_passthrough(self):
        svc = mock.Mock()
        thermo = _FakeThermostat(identifier="abc123")
        c1, c2, c3 = self._ctx(svc, thermo)
        with c1, c2, c3:
            res = self.mod.set_state({"name": "Main"}, mode="cool")
        self.assertEqual(res["applied"]["mode"], "cool")

    def test_temperature_uses_whole_degrees_keyword_selection(self):
        svc = mock.Mock()
        thermo = _FakeThermostat(identifier="abc123")
        c1, c2, c3 = self._ctx(svc, thermo)
        with c1, c2, c3:
            res = self.mod.set_state({"name": "Main"}, temperature=72)
        self.assertEqual(res["applied"]["temperature"], 72)
        args, kwargs = svc.set_hold.call_args
        # pyecobee set_hold: temps are whole degrees F (NOT tenths) and
        # selection MUST be a keyword — positional lands in cool_hold_temp.
        self.assertEqual(args, ())
        self.assertEqual(kwargs.get("cool_hold_temp"), 72)
        self.assertEqual(kwargs.get("heat_hold_temp"), 72)
        self.assertIn("selection", kwargs)

    def test_set_hvac_mode_failure_returns_partial(self):
        svc = mock.Mock()
        svc.update_thermostats.side_effect = RuntimeError("mode rejected")
        thermo = _FakeThermostat(identifier="abc123")
        c1, c2, c3 = self._ctx(svc, thermo)
        with c1, c2, c3:
            res = self.mod.set_state({"name": "Main"}, on=False)
        self.assertIn("set_hvac_mode failed", res["error"])
        self.assertIn("partial", res)

    def test_set_hold_failure_returns_partial(self):
        svc = mock.Mock()
        svc.set_hold.side_effect = RuntimeError("hold rejected")
        thermo = _FakeThermostat(identifier="abc123")
        c1, c2, c3 = self._ctx(svc, thermo)
        with c1, c2, c3:
            res = self.mod.set_state({"name": "Main"}, mode="heat", temperature=70)
        self.assertIn("set_hold failed", res["error"])
        # mode landed before the hold failed.
        self.assertEqual(res["partial"]["mode"], "heat")


# ─── degradation + list action ───────────────────────────────────────────
class EcobeeDegradationTests(_EcobeeBase):
    def test_list_action_counts(self):
        with mock.patch.object(self.mod, "list_devices",
                               return_value=[{"name": "Main"}]):
            out = self.actions["ecobee_list_devices"]("")
        self.assertIn("1 Ecobee thermostat", out)


# ─── PIN flow: _do_request_pin + request_pin action ──────────────────────
class EcobeePinFlowGuardTests(_EcobeeBase):
    def test_request_pin_without_lib(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=None):
            out = self.actions["ecobee_request_pin"]("")
        self.assertIn("not installed", out)

    def test_request_pin_without_api_key(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=object()), \
             mock.patch.object(self.mod, "_read_config", return_value={}):
            out = self.actions["ecobee_request_pin"]("")
        self.assertIn("API key", out)
        self.assertIn("ecobee.com/developers", out)

    def test_request_pin_missing_ecobeeservice(self):
        with mock.patch.object(self.mod, "_pyecobee",
                               return_value=_fake_pyecobee(omit_service=True)), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}):
            out = self.actions["ecobee_request_pin"]("")
        self.assertIn("EcobeeService missing", out)

    def test_request_pin_service_construction_failure(self):
        fake = _fake_pyecobee()

        def _raise(*a, **k):
            raise RuntimeError("ctor boom")
        fake.EcobeeService = _raise
        with mock.patch.object(self.mod, "_pyecobee", return_value=fake), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}):
            out = self.actions["ecobee_request_pin"]("")
        self.assertIn("construction failed", out)

    def test_request_pin_authorize_failure(self):
        svc = mock.Mock()
        svc.authorize.side_effect = RuntimeError("authorize boom")
        fake = _fake_pyecobee(service=svc)
        with mock.patch.object(self.mod, "_pyecobee", return_value=fake), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}):
            out = self.actions["ecobee_request_pin"]("")
        self.assertIn("authorize call failed", out)

    def test_request_pin_happy_returns_pin(self):
        svc = mock.Mock()
        svc.authorize.return_value = mock.Mock(ecobee_pin="ABCD")
        fake = _fake_pyecobee(service=svc)
        with mock.patch.object(self.mod, "_pyecobee", return_value=fake), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_save_tokens") as save, \
             contextlib.redirect_stdout(io.StringIO()):
            out = self.actions["ecobee_request_pin"]("")
        self.assertIn("ABCD", out)
        save.assert_called_once()  # authorization_token persisted


# ─── complete_setup + authorize alias ────────────────────────────────────
class EcobeeCompleteSetupTests(_EcobeeBase):
    def test_complete_setup_without_lib(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=None):
            out = self.actions["ecobee_complete_setup"]("")
        self.assertIn("not installed", out)

    def test_complete_setup_without_api_key(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=object()), \
             mock.patch.object(self.mod, "_read_config", return_value={}):
            out = self.actions["ecobee_complete_setup"]("")
        self.assertIn("API key", out)

    def test_complete_setup_without_pending_token(self):
        with mock.patch.object(self.mod, "_pyecobee", return_value=object()), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_load_tokens", return_value={}):
            out = self.actions["ecobee_complete_setup"]("")
        self.assertIn("No pending authorization", out)

    def test_complete_setup_missing_ecobeeservice(self):
        with mock.patch.object(self.mod, "_pyecobee",
                               return_value=_fake_pyecobee(omit_service=True)), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_load_tokens",
                               return_value={"authorization_token": "auth"}):
            out = self.actions["ecobee_complete_setup"]("")
        self.assertIn("EcobeeService missing", out)

    def test_complete_setup_request_tokens_failure(self):
        svc = mock.Mock()
        svc.request_tokens.side_effect = RuntimeError("pin expired")
        fake = _fake_pyecobee(service=svc)
        with mock.patch.object(self.mod, "_pyecobee", return_value=fake), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_load_tokens",
                               return_value={"authorization_token": "auth"}):
            out = self.actions["ecobee_complete_setup"]("")
        self.assertIn("token request failed", out)

    def test_complete_setup_success_caches_service(self):
        svc = mock.Mock()
        fake = _fake_pyecobee(service=svc)
        with mock.patch.object(self.mod, "_pyecobee", return_value=fake), \
             mock.patch.object(self.mod, "_read_config",
                               return_value={"api_key": "k"}), \
             mock.patch.object(self.mod, "_load_tokens",
                               return_value={"authorization_token": "auth"}), \
             mock.patch.object(self.mod, "_save_tokens") as save:
            out = self.actions["ecobee_complete_setup"]("")
        self.assertIn("authorized", out)
        save.assert_called_once_with(svc)
        self.assertIs(self.mod._state["service"], svc)

    def test_authorize_alias_explains_two_steps(self):
        out = self.actions["ecobee_authorize"]("")
        self.assertIn("request", out.lower())
        self.assertIn("complete setup", out.lower())


# ─── interactive wizard ──────────────────────────────────────────────────
class EcobeeWizardTests(_EcobeeBase):
    def test_wizard_pin_error_returns_early(self):
        # _do_request_pin returns an error → wizard surfaces it without prompting.
        with mock.patch.object(self.mod, "_do_request_pin",
                               return_value=("", "pyecobee not installed, sir.")):
            out = self.mod._run_ecobee_wizard_interactive()
        self.assertIn("not installed", out)

    def test_wizard_happy_completes_after_enter(self):
        with mock.patch.object(self.mod, "_do_request_pin",
                               return_value=("WXYZ", "")), \
             mock.patch("builtins.input", return_value=""), \
             mock.patch.object(self.mod, "ecobee_complete_setup",
                               return_value="Ecobee authorized, sir."), \
             contextlib.redirect_stdout(io.StringIO()):
            out = self.mod._run_ecobee_wizard_interactive()
        self.assertIn("authorized", out)

    def test_wizard_cancelled_at_enter(self):
        with mock.patch.object(self.mod, "_do_request_pin",
                               return_value=("WXYZ", "")), \
             mock.patch("builtins.input", side_effect=KeyboardInterrupt), \
             contextlib.redirect_stdout(io.StringIO()):
            out = self.mod._run_ecobee_wizard_interactive()
        self.assertIn("cancelled", out)


# ─── register ────────────────────────────────────────────────────────────
class EcobeeRegisterTests(_EcobeeBase):
    def test_register_wires_all_actions(self):
        acts = {}
        self.mod.register(acts)
        for name in ("ecobee_list_devices", "ecobee_request_pin",
                     "ecobee_complete_setup", "ecobee_authorize"):
            self.assertIn(name, acts)


if __name__ == "__main__":
    unittest.main()
