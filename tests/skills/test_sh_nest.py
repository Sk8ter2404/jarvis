"""Logic tests for skills/sh_nest.py (Nest SDM thermostat/camera controller).

Wraps the optional `google-nest-sdm` library behind a non-trivial OAuth setup.
The dominant risk is graceful degradation: with the lib absent OR the OAuth
config incomplete, every public path returns a clean error/empty result and
never raises. We also test:
  * is_available gating on both lib presence AND all four config keys,
  * _read_config file present / absent / malformed,
  * the asyncio loop runner (_start_loop_thread / _run_async) on a real,
    in-process loop with fully-faked coroutines (no network, no sleep),
  * _build_client_async's OAuth token-exchange flow (fake aiohttp + requests)
    plus every degradation branch (lib absent, config incomplete, aiohttp
    absent, token POST raises, no AccessTokenAuth),
  * _get_client caching, rebuild, and stale-session close,
  * list_devices' SDM payload parse (thermostat / camera / doorbell / unknown
    + parentRelations displayName resolution),
  * _device_id resolution (native_id, name match, miss),
  * get_state's Celsius->Fahrenheit translation and trait extraction,
  * set_state's on/off -> HEATCOOL/OFF mode mapping + per-command failures,
  * the authorize action's missing-config guard AND its happy URL path.

`google-nest-sdm` is NOT a CI dependency, so the SDM/aiohttp/requests modules
are ALWAYS injected as fakes — never imported for real — keeping the suite
deterministic and offline on any host. Module globals (_state) are reset and
any spawned loop is shut down in tearDown so tests stay isolated.
"""
from __future__ import annotations

import contextlib
import sys
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── fake-module injection helpers (self-contained; mirror test_self_diagnostic) ─
_SENTINEL = object()


@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules, restoring the prior
    state (including absence) on exit so deferred imports see exactly the fake
    we provide and tests stay isolated."""
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
    """Force ``import <name>`` to raise ImportError inside the block, so a
    deferred-import miss branch is exercised even when the real dep is present
    on the dev box."""
    real_import = __import__
    blocked = set(names)

    def _fake_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if name in blocked or top in blocked:
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


def _fake_sdm_modules():
    """A minimal google_nest_sdm package tree: auth (with AccessTokenAuth),
    device_manager (with DeviceManager), google_nest_subscriber. Injected so
    _sdm() resolves without the real (uninstalled) library."""
    pkg = types.ModuleType("google_nest_sdm")
    auth = types.ModuleType("google_nest_sdm.auth")
    dm = types.ModuleType("google_nest_sdm.device_manager")
    gns = types.ModuleType("google_nest_sdm.google_nest_subscriber")

    class AccessTokenAuth:
        def __init__(self, session, token, api_url):
            self.session = session
            self.token = token
            self.api_url = api_url

        async def request(self, method, path, **kw):  # pragma: no cover - overridden
            return {}

    class DeviceManager:
        pass

    auth.AccessTokenAuth = AccessTokenAuth
    auth.AbstractAuth = None  # AccessTokenAuth branch is the one we exercise
    dm.DeviceManager = DeviceManager
    pkg.auth = auth
    pkg.device_manager = dm
    pkg.google_nest_subscriber = gns
    return {
        "google_nest_sdm": pkg,
        "google_nest_sdm.auth": auth,
        "google_nest_sdm.device_manager": dm,
        "google_nest_sdm.google_nest_subscriber": gns,
    }


def _fake_aiohttp(close_record=None):
    """A fake aiohttp module whose ClientSession.close() is an awaitable that
    records it ran. No socket is ever opened."""
    aiohttp = types.ModuleType("aiohttp")

    class _Session:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True
            if close_record is not None:
                close_record.append(self)

    aiohttp.ClientSession = _Session
    return aiohttp


def _fake_requests(token="access-123", status_raises=False, json_obj=None):
    """A fake requests module exposing only .post(...).{raise_for_status,json}."""
    req = types.ModuleType("requests")

    class _Resp:
        def raise_for_status(self):
            if status_raises:
                raise RuntimeError("401 Unauthorized")

        def json(self):
            return json_obj if json_obj is not None else {"access_token": token}

    req.post = mock.MagicMock(return_value=_Resp())
    return req


_FULL_CFG = {"project_id": "proj", "client_id": "cid",
             "client_secret": "secret", "refresh_token": "rtok"}


class _NestBase(unittest.TestCase):
    """Loads sh_nest in isolation and guarantees module globals are reset and
    any spawned asyncio loop is shut down after each test."""

    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_nest")
        self.addCleanup(self._reset_state)

    def _reset_state(self):
        st = self.mod._state
        loop = st.get("loop")
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        st["client"] = None
        st["devices"] = []
        st["fetched_at"] = 0.0
        st["loop"] = None
        st["thread"] = None


# ─── is_available ────────────────────────────────────────────────────────
class NestAvailabilityTests(_NestBase):
    def test_unavailable_without_lib(self):
        with mock.patch.object(self.mod, "_sdm", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_unavailable_with_lib_but_no_config(self):
        with mock.patch.object(self.mod, "_sdm", return_value=("a", "b", "c")), \
             mock.patch.object(self.mod, "_read_config", return_value={}):
            self.assertFalse(self.mod.is_available())

    def test_unavailable_with_partial_config(self):
        partial = {"project_id": "p", "client_id": "c"}  # missing secret+token
        with mock.patch.object(self.mod, "_sdm", return_value=("a", "b", "c")), \
             mock.patch.object(self.mod, "_read_config", return_value=partial):
            self.assertFalse(self.mod.is_available())

    def test_available_with_lib_and_full_config(self):
        with mock.patch.object(self.mod, "_sdm", return_value=("a", "b", "c")), \
             mock.patch.object(self.mod, "_read_config", return_value=dict(_FULL_CFG)):
            self.assertTrue(self.mod.is_available())


# ─── _sdm import ─────────────────────────────────────────────────────────
class NestSdmImportTests(_NestBase):
    def test_sdm_returns_none_when_lib_absent(self):
        with block_import("google_nest_sdm"):
            self.assertIsNone(self.mod._sdm())

    def test_sdm_returns_triple_when_present(self):
        with inject_modules(**_fake_sdm_modules()):
            triple = self.mod._sdm()
        self.assertIsNotNone(triple)
        self.assertEqual(len(triple), 3)


# ─── _read_config ────────────────────────────────────────────────────────
class NestReadConfigTests(_NestBase):
    def test_missing_file_returns_empty(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._read_config(), {})

    def test_valid_json_parsed(self):
        m = mock.mock_open(read_data='{"project_id": "p"}')
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._read_config(), {"project_id": "p"})

    def test_malformed_json_returns_empty(self):
        m = mock.mock_open(read_data="{not json")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._read_config(), {})

    def test_null_json_coerced_to_empty(self):
        m = mock.mock_open(read_data="null")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._read_config(), {})


# ─── asyncio loop runner ─────────────────────────────────────────────────
class NestAsyncRunnerTests(_NestBase):
    def test_run_async_executes_coroutine_on_loop(self):
        async def _co():
            return 41 + 1
        self.assertEqual(self.mod._run_async(_co()), 42)
        # The loop is cached on _state.
        self.assertIsNotNone(self.mod._state["loop"])

    def test_start_loop_thread_is_idempotent(self):
        loop1 = self.mod._start_loop_thread()
        loop2 = self.mod._start_loop_thread()
        self.assertIs(loop1, loop2)

    def test_run_async_propagates_exception(self):
        async def _boom():
            raise ValueError("kaboom")
        with self.assertRaises(ValueError):
            self.mod._run_async(_boom())

    def test_loop_start_timeout_raises(self):
        # If the worker thread never signals `ready` within 5s, _start_loop_thread
        # raises RuntimeError. Rather than wait 5s, neuter Thread.start so the
        # worker never runs (ready stays clear) and stub the ready Event's wait()
        # to report timeout immediately. A real threading.Event subclass keeps
        # the Thread bookkeeping (is_set/set) intact.
        class _ImmediateTimeoutEvent(self.mod.threading.Event):
            def wait(self, timeout=None):
                return False

        with mock.patch.object(self.mod.threading, "Event",
                               _ImmediateTimeoutEvent), \
             mock.patch.object(self.mod.threading.Thread, "start",
                               lambda self: None):
            with self.assertRaises(RuntimeError):
                self.mod._start_loop_thread()
        # clean up the orphan loop we created (never ran, so just close it)
        with self.mod._lock:
            self.mod._state["loop"] = None
            self.mod._state["thread"] = None


# ─── _build_client_async ─────────────────────────────────────────────────
class NestBuildClientTests(_NestBase):
    def _build(self):
        """Run the real _build_client_async coroutine on the module loop."""
        return self.mod._run_async(self.mod._build_client_async())

    def test_returns_none_when_sdm_absent(self):
        with mock.patch.object(self.mod, "_sdm", return_value=None):
            self.assertIsNone(self._build())

    def test_returns_none_when_config_incomplete(self):
        with inject_modules(**_fake_sdm_modules()), \
             mock.patch.object(self.mod, "_read_config", return_value={}):
            self.assertIsNone(self._build())

    def test_returns_none_when_aiohttp_absent(self):
        with inject_modules(**_fake_sdm_modules()), \
             mock.patch.object(self.mod, "_read_config", return_value=dict(_FULL_CFG)), \
             block_import("aiohttp"):
            self.assertIsNone(self._build())

    def test_happy_path_returns_client_triple(self):
        req = _fake_requests(token="tok-xyz")
        closed = []
        with inject_modules(**_fake_sdm_modules(),
                            aiohttp=_fake_aiohttp(closed), requests=req), \
             mock.patch.object(self.mod, "_read_config", return_value=dict(_FULL_CFG)):
            result = self._build()
        self.assertIsNotNone(result)
        client, project_id, session = result
        self.assertEqual(project_id, "proj")
        self.assertEqual(client.token, "tok-xyz")   # access_token threaded in
        self.assertFalse(session.closed)            # session kept open on success
        # The token POST hit Google's OAuth endpoint with the refresh grant.
        _args, kwargs = req.post.call_args
        self.assertEqual(kwargs["data"]["grant_type"], "refresh_token")
        self.assertEqual(kwargs["data"]["refresh_token"], "rtok")

    def test_token_exchange_failure_closes_session_returns_none(self):
        req = _fake_requests(status_raises=True)
        closed = []
        with inject_modules(**_fake_sdm_modules(),
                            aiohttp=_fake_aiohttp(closed), requests=req), \
             mock.patch.object(self.mod, "_read_config", return_value=dict(_FULL_CFG)):
            result = self._build()
        self.assertIsNone(result)
        self.assertEqual(len(closed), 1)            # session was closed on failure

    def test_no_access_token_auth_closes_session(self):
        mods = _fake_sdm_modules()
        mods["google_nest_sdm.auth"].AccessTokenAuth = None  # neither auth class
        mods["google_nest_sdm"].auth.AccessTokenAuth = None
        closed = []
        with inject_modules(**mods, aiohttp=_fake_aiohttp(closed),
                            requests=_fake_requests()), \
             mock.patch.object(self.mod, "_read_config", return_value=dict(_FULL_CFG)):
            result = self._build()
        self.assertIsNone(result)
        self.assertEqual(len(closed), 1)

    def test_outer_exception_returns_none(self):
        # ClientSession() itself raises (after the aiohttp import succeeds) ->
        # the broad outer `except Exception` swallows it and returns None.
        aiohttp = types.ModuleType("aiohttp")

        def _boom(*a, **k):
            raise RuntimeError("session ctor exploded")

        aiohttp.ClientSession = _boom
        with inject_modules(**_fake_sdm_modules(), aiohttp=aiohttp,
                            requests=_fake_requests()), \
             mock.patch.object(self.mod, "_read_config", return_value=dict(_FULL_CFG)):
            self.assertIsNone(self._build())


# ─── _get_client caching ─────────────────────────────────────────────────
class NestGetClientTests(_NestBase):
    def test_returns_cached_client_within_ttl(self):
        sentinel = ("client", "proj", "sess")
        self.mod._state["client"] = sentinel
        self.mod._state["fetched_at"] = self.mod.time.monotonic()
        # No build should happen — _build_client_async must not be touched.
        with mock.patch.object(self.mod, "_run_async",
                               side_effect=AssertionError("should not build")):
            self.assertIs(self.mod._get_client(), sentinel)

    def test_builds_and_caches_on_miss(self):
        built = ("newclient", "proj", "newsess")
        # Patch with a plain (non-async) callable so calling it doesn't mint an
        # un-awaited coroutine; _run_async is mocked, so its arg is irrelevant.
        with mock.patch.object(self.mod, "_build_client_async", lambda: None), \
             mock.patch.object(self.mod, "_run_async", return_value=built):
            got = self.mod._get_client()
        self.assertIs(got, built)
        self.assertIs(self.mod._state["client"], built)

    def test_build_exception_returns_none(self):
        with mock.patch.object(self.mod, "_build_client_async", lambda: None), \
             mock.patch.object(self.mod, "_run_async",
                               side_effect=RuntimeError("loop dead")):
            self.assertIsNone(self.mod._get_client())

    def test_stale_session_closed_on_rebuild(self):
        # An expired cached triple whose session.close() must be awaited.
        class _OldSess:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True

        old_sess = _OldSess()
        self.mod._state["client"] = ("old", "proj", old_sess)
        self.mod._state["fetched_at"] = self.mod.time.monotonic() - 999  # expired
        new = ("new", "proj", "newsess")
        real_run = self.mod._run_async
        calls = {"n": 0}

        def _fake_run(coro):
            calls["n"] += 1
            # 1st call is the (faked) build → return the new triple directly.
            # 2nd call is old_sess.close() → run that real coroutine on the loop.
            if calls["n"] == 1:
                return new
            return real_run(coro)

        # _build_client_async is patched with a plain callable so the build call
        # passes `None` to _fake_run (never awaited); the close() coroutine is
        # real and IS awaited via real_run.
        with mock.patch.object(self.mod, "_build_client_async", lambda: None), \
             mock.patch.object(self.mod, "_run_async", side_effect=_fake_run):
            got = self.mod._get_client()
        self.assertIs(got, new)
        self.assertTrue(old_sess.closed)   # stale session was closed

    def test_inner_recheck_returns_fresh_client_built_by_racer(self):
        # The outer cache check misses (expired), but by the time we take
        # _build_lock another caller has populated a FRESH client. The inner
        # double-checked re-read returns it without building. We simulate the
        # racer by populating _state when _build_lock is entered.
        fresh = ("fresh-by-racer", "proj", "sess")
        mod = self.mod

        class _RacingLock:
            """Stand-in for _build_lock: on entry, simulate another caller
            having just populated a fresh client (the race the inner
            double-check guards against)."""
            def __enter__(self):
                with mod._lock:
                    mod._state["client"] = fresh
                    mod._state["fetched_at"] = mod.time.monotonic()
                return self

            def __exit__(self, *exc):
                return False

        with mock.patch.object(self.mod, "_build_lock", _RacingLock()), \
             mock.patch.object(self.mod, "_run_async",
                               side_effect=AssertionError("must not build")):
            got = self.mod._get_client()
        self.assertIs(got, fresh)

    def test_stale_session_close_failure_swallowed(self):
        # The expired session's close() raises during rebuild -> the broad
        # except swallows it (lines logging "stale session close failed") and
        # the new client is still installed.
        class _BadSess:
            async def close(self):
                raise RuntimeError("close blew up")

        self.mod._state["client"] = ("old", "proj", _BadSess())
        self.mod._state["fetched_at"] = self.mod.time.monotonic() - 999
        new = ("new", "proj", "newsess")
        real_run = self.mod._run_async
        calls = {"n": 0}

        def _fake_run(coro):
            calls["n"] += 1
            if calls["n"] == 1:
                return new            # the (faked) build
            return real_run(coro)     # the close() coroutine -> raises -> swallowed

        with mock.patch.object(self.mod, "_build_client_async", lambda: None), \
             mock.patch.object(self.mod, "_run_async", side_effect=_fake_run):
            got = self.mod._get_client()
        self.assertIs(got, new)
        self.assertIs(self.mod._state["client"], new)


# ─── list_devices payload parse ──────────────────────────────────────────
class NestListDevicesTests(_NestBase):
    def _list_with_payload(self, payload):
        class _Client:
            async def request(self, method, path, **kw):
                return payload

        fake = (_Client(), "proj", "sess")
        with inject_modules(**_fake_sdm_modules()), \
             mock.patch.object(self.mod, "_get_client", return_value=fake):
            return self.mod.list_devices()

    def test_empty_without_lib_or_client(self):
        with mock.patch.object(self.mod, "_sdm", return_value=None), \
             mock.patch.object(self.mod, "_get_client", return_value=None):
            self.assertEqual(self.mod.list_devices(), [])

    def test_thermostat_parsed_with_displayname(self):
        payload = {"devices": [{
            "name": "enterprises/proj/devices/T1",
            "type": "sdm.devices.types.THERMOSTAT",
            "traits": {},
            "parentRelations": [{"displayName": "Hallway"}],
        }]}
        out = self._list_with_payload(payload)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "Hallway")
        self.assertEqual(out[0]["type"], "thermostat")
        self.assertIn("thermostat", out[0]["capabilities"])
        self.assertEqual(out[0]["native_id"], "enterprises/proj/devices/T1")

    def test_camera_and_doorbell_typed_as_camera(self):
        payload = {"devices": [
            {"name": "enterprises/proj/devices/C1",
             "type": "sdm.devices.types.CAMERA", "traits": {},
             "parentRelations": []},
            {"name": "enterprises/proj/devices/D1",
             "type": "sdm.devices.types.DOORBELL", "traits": {},
             "parentRelations": []},
        ]}
        out = self._list_with_payload(payload)
        kinds = {d["native_id"]: d for d in out}
        self.assertEqual(kinds["enterprises/proj/devices/C1"]["type"], "camera")
        self.assertEqual(kinds["enterprises/proj/devices/D1"]["type"], "camera")
        # Name falls back to the last path segment when no displayName.
        self.assertEqual(kinds["enterprises/proj/devices/C1"]["name"], "C1")

    def test_unknown_type_when_neither(self):
        payload = {"devices": [{
            "name": "enterprises/proj/devices/X1",
            "type": "sdm.devices.types.SOMETHING", "traits": {},
            "parentRelations": [],
        }]}
        out = self._list_with_payload(payload)
        self.assertEqual(out[0]["type"], "unknown")
        self.assertEqual(out[0]["capabilities"], [])

    def test_non_dict_payload_yields_empty(self):
        self.assertEqual(self._list_with_payload(["not", "a", "dict"]), [])

    def test_request_exception_returns_empty(self):
        class _Client:
            async def request(self, method, path, **kw):
                raise RuntimeError("api boom")

        fake = (_Client(), "proj", "sess")
        with inject_modules(**_fake_sdm_modules()), \
             mock.patch.object(self.mod, "_get_client", return_value=fake):
            self.assertEqual(self.mod.list_devices(), [])


# ─── _device_id resolution ───────────────────────────────────────────────
class NestDeviceIdTests(_NestBase):
    def test_native_id_passthrough(self):
        self.assertEqual(
            self.mod._device_id({"native_id": "enterprises/x/devices/y"}),
            "enterprises/x/devices/y")

    def test_resolves_by_name(self):
        with mock.patch.object(self.mod, "list_devices", return_value=[
                {"name": "Hallway", "native_id": "nid-1"}]):
            self.assertEqual(self.mod._device_id({"name": "hallway"}), "nid-1")

    def test_returns_none_when_no_name_and_no_id(self):
        self.assertIsNone(self.mod._device_id({}))

    def test_returns_none_when_name_unmatched(self):
        with mock.patch.object(self.mod, "list_devices", return_value=[
                {"name": "Bedroom", "native_id": "nid-2"}]):
            self.assertIsNone(self.mod._device_id({"name": "Garage"}))


# ─── degradation paths for public API ────────────────────────────────────
class NestDegradationTests(_NestBase):
    def test_list_devices_empty_without_client(self):
        with mock.patch.object(self.mod, "_sdm", return_value=None), \
             mock.patch.object(self.mod, "_get_client", return_value=None):
            self.assertEqual(self.mod.list_devices(), [])

    def test_get_state_errors_without_client(self):
        with mock.patch.object(self.mod, "_get_client", return_value=None):
            res = self.mod.get_state({"name": "Hallway"})
        self.assertIn("not initialized", res["error"])

    def test_get_state_device_not_found(self):
        fake_client = ("client", "project", "session")
        with mock.patch.object(self.mod, "_get_client", return_value=fake_client), \
             mock.patch.object(self.mod, "_device_id", return_value=None):
            res = self.mod.get_state({"name": "Ghost"})
        self.assertIn("not found", res["error"])

    def test_get_state_request_exception(self):
        class _Client:
            async def request(self, method, path, **kw):
                raise RuntimeError("net down")

        fake = (_Client(), "proj", "sess")
        with mock.patch.object(self.mod, "_get_client", return_value=fake), \
             mock.patch.object(self.mod, "_device_id", return_value="nid"):
            res = self.mod.get_state({"name": "Hallway"})
        self.assertIn("get_state failed", res["error"])

    def test_set_state_errors_without_client_mentions_authorize(self):
        with mock.patch.object(self.mod, "_get_client", return_value=None):
            res = self.mod.set_state({"name": "Hallway"}, temperature=70)
        self.assertIn("not initialized", res["error"])
        self.assertIn("sh_nest_authorize", res["error"])

    def test_set_state_device_not_found(self):
        fake_client = ("client", "project", "session")
        with mock.patch.object(self.mod, "_get_client", return_value=fake_client), \
             mock.patch.object(self.mod, "_device_id", return_value=None):
            res = self.mod.set_state({"name": "Ghost"}, temperature=70)
        self.assertIn("not found", res["error"])

    def test_nest_list_action_counts_devices(self):
        with mock.patch.object(self.mod, "list_devices",
                               return_value=[{"name": "A"}, {"name": "B"}]):
            out = self.actions["nest_list_devices"]("")
        self.assertIn("2 Nest device", out)

    def test_authorize_requires_config(self):
        with mock.patch.object(self.mod, "_read_config", return_value={}):
            out = self.actions["nest_authorize"]("")
        self.assertIn("project_id", out)
        self.assertIn("client_id", out)

    def test_authorize_happy_prints_url(self):
        cfg = {"client_id": "CID", "project_id": "PID"}
        with mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             contextlib.redirect_stdout(__import__("io").StringIO()):
            out = self.actions["nest_authorize"]("")
        self.assertIn("authorization URL", out)


class NestGetStateTranslationTests(_NestBase):
    """get_state pulls SDM traits and converts ambient °C → °F. We stub the
    client + _run_async so a canned trait payload flows through."""

    def test_celsius_to_fahrenheit_and_mode(self):
        traits = {
            "sdm.devices.traits.Temperature": {"ambientTemperatureCelsius": 20.0},
            "sdm.devices.traits.ThermostatMode": {"mode": "HEAT"},
            "sdm.devices.traits.ThermostatTemperatureSetpoint":
                {"heatCelsius": 21.0, "coolCelsius": 25.0},
        }

        class _Client:
            async def request(self, method, path, **kw):
                return {"traits": traits}

        fake_client = (_Client(), "project", "session")
        with mock.patch.object(self.mod, "_get_client", return_value=fake_client), \
             mock.patch.object(self.mod, "_device_id",
                               return_value="enterprises/x/devices/y"):
            st = self.mod.get_state({"name": "Hallway"})
        self.assertEqual(st["actual_c"], 20.0)
        self.assertEqual(st["actual_f"], 68.0)          # 20°C == 68°F
        self.assertEqual(st["mode"], "HEAT")
        self.assertEqual(st["heat_set_c"], 21.0)
        self.assertEqual(st["cool_set_c"], 25.0)

    def test_get_state_missing_temperature_trait_is_none(self):
        class _Client:
            async def request(self, method, path, **kw):
                return {"traits": {}}

        fake_client = (_Client(), "project", "session")
        with mock.patch.object(self.mod, "_get_client", return_value=fake_client), \
             mock.patch.object(self.mod, "_device_id", return_value="nid"):
            st = self.mod.get_state({"name": "Hallway"})
        self.assertIsNone(st["actual_c"])
        self.assertIsNone(st["actual_f"])
        self.assertIsNone(st["mode"])


class NestSetStateModeMappingTests(_NestBase):
    """set_state maps on/off → HEATCOOL/OFF and issues SetMode/SetHeat commands.
    We capture the coroutines via a _run_async stub that just runs them with a
    fake client whose .request records calls."""

    def _run_with_recorder(self, request_impl=None, **set_kwargs):
        recorded = []

        async def _default(method, path, **kw):
            recorded.append((method, path, kw))
            return {}

        impl = request_impl or _default

        class _Client:
            async def request(self, method, path, **kw):
                return await impl(method, path, **kw)

        fake = (_Client(), "project", "session")
        with mock.patch.object(self.mod, "_get_client", return_value=fake), \
             mock.patch.object(self.mod, "_device_id",
                               return_value="enterprises/x/devices/y"):
            res = self.mod.set_state({"name": "Hallway"}, **set_kwargs)
        return res, recorded

    def test_on_maps_to_heatcool(self):
        res, recorded = self._run_with_recorder(on=True)
        self.assertEqual(res["applied"]["mode"], "HEATCOOL")
        bodies = [kw.get("json", {}) for _m, _p, kw in recorded]
        self.assertTrue(any(b.get("params", {}).get("mode") == "HEATCOOL"
                            for b in bodies))

    def test_off_maps_to_off(self):
        res, _recorded = self._run_with_recorder(on=False)
        self.assertEqual(res["applied"]["mode"], "OFF")

    def test_explicit_mode_passthrough(self):
        res, recorded = self._run_with_recorder(mode="COOL")
        self.assertEqual(res["applied"]["mode"], "COOL")

    def test_temperature_recorded_in_fahrenheit_field(self):
        res, recorded = self._run_with_recorder(temperature=70)
        self.assertEqual(res["applied"]["temperature"], 70)
        heat_bodies = [kw.get("json", {}) for _m, _p, kw in recorded
                       if "SetHeat" in (kw.get("json", {}) or {}).get("command", "")]
        self.assertTrue(heat_bodies)
        c = heat_bodies[0]["params"]["heatCelsius"]
        self.assertAlmostEqual(c, (70 - 32) * 5 / 9, places=4)

    def test_mode_and_temperature_together(self):
        res, recorded = self._run_with_recorder(on=True, temperature=68)
        self.assertEqual(res["applied"]["mode"], "HEATCOOL")
        self.assertEqual(res["applied"]["temperature"], 68)

    def test_set_mode_failure_returns_error(self):
        async def _raise(method, path, **kw):
            raise RuntimeError("mode rejected")
        res, _ = self._run_with_recorder(request_impl=_raise, mode="HEAT")
        self.assertIn("set_mode failed", res["error"])

    def test_set_setpoint_failure_returns_error(self):
        # Mode succeeds (no mode given), setpoint command raises.
        async def _raise(method, path, **kw):
            raise RuntimeError("setpoint rejected")
        res, _ = self._run_with_recorder(request_impl=_raise, temperature=70)
        self.assertIn("set_setpoint failed", res["error"])

    def test_outer_exception_returns_partial(self):
        # _run_async itself blows up → outer except returns partial dict. Close
        # the un-awaited _go() coroutine it receives to avoid a RuntimeWarning.
        def _boom(coro):
            try:
                coro.close()
            except AttributeError:
                pass
            raise RuntimeError("loop gone")

        with mock.patch.object(self.mod, "_get_client",
                               return_value=("c", "p", "s")), \
             mock.patch.object(self.mod, "_device_id", return_value="nid"), \
             mock.patch.object(self.mod, "_run_async", side_effect=_boom):
            res = self.mod.set_state({"name": "Hallway"}, temperature=70)
        self.assertIn("set_state failed", res["error"])
        self.assertIn("partial", res)


# ─── register ────────────────────────────────────────────────────────────
class NestRegisterTests(_NestBase):
    def test_register_wires_both_actions(self):
        acts = {}
        self.mod.register(acts)
        self.assertIn("nest_list_devices", acts)
        self.assertIn("nest_authorize", acts)


if __name__ == "__main__":
    unittest.main()
