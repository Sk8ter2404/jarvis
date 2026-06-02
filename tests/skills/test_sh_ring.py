"""Logic tests for skills/sh_ring.py (Ring camera/doorbell controller).

Wraps the optional `ring_doorbell` library. Coverage:
  * is_available gating on lib presence AND a cached token,
  * _ring_doorbell import (present via injected fake / absent),
  * _read_token / _read_config (present / absent / malformed) and _save_token
    (delegates to core.atomic_io, swallows errors),
  * _run_with_timeout ok / error / timed-out branches,
  * _get_ring client construction + cache, the update_data timeout + error
    paths, and the missing-Auth/Ring guard,
  * _enumerate flattening + enum failure, list_devices' capability derivation
    (camera / chime / on_off / siren),
  * _match by native_id and by case-insensitive name,
  * get_state attribute read (+ not-found / not-authorized),
  * set_state's capability-gated apply (lights / siren / chime), the
    siren-failure partial path, and the "nothing landed" path,
  * _do_fetch_token (no lib / no Auth / 2FA-required / success / failure),
  * ring_authorize inline-creds, cached-config, CLI-hint, and 2FA messages,
  * _run_ring_wizard_interactive (inline + prompted + 2FA + cancel),
  * register wiring.

`ring_doorbell` is NOT a CI dependency, so it is ALWAYS injected as a fake —
never imported for real — keeping the suite deterministic and offline. Module
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


class _FakeRingDevice:
    """Fake stickup-cam / floodlight exposing `lights` and `siren`."""
    def __init__(self, name="Front Door", did="dev1", has_lights=True,
                 has_siren=True, family="stickup_cams", battery=88,
                 has_chime=False):
        self.name = name
        self.id = did
        self.family = family
        self.battery_life = battery
        self.connection_status = "online"
        self.last_motion = "2026-06-01T10:00:00"
        if has_lights:
            self.lights = "off"
        if has_siren:
            self.siren = "off"
        if has_chime:
            self.existing_doorbell_type_enabled = False


def _fake_ring_doorbell(auth_factory=None, ring_factory=None,
                        omit_auth=False, omit_ring=False):
    """A fake ring_doorbell module exposing Auth + Ring. Factories let a test
    customise construction behaviour (e.g. raise)."""
    mod = types.ModuleType("ring_doorbell")

    class _Auth:
        def __init__(self, ua, token=None, save_cb=None):
            self.ua = ua
            self.token = token
            self.save_cb = save_cb
            if auth_factory:
                auth_factory(self)

        def fetch_token(self, email, password, code=None):
            return {"refresh_token": "rt"}

    class _Ring:
        def __init__(self, auth):
            self.auth = auth
            self._devices = {}
            if ring_factory:
                ring_factory(self)

        def update_data(self):
            return None

        def devices(self):
            return self._devices

    if not omit_auth:
        mod.Auth = _Auth
    if not omit_ring:
        mod.Ring = _Ring
    return mod


class _RingBase(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("sh_ring")
        self.addCleanup(self._reset_state)

    def _reset_state(self):
        self.mod._state["ring"] = None
        self.mod._state["fetched_at"] = 0.0
        self.mod._state["devices_cache"] = {}


# ─── is_available + lib import ───────────────────────────────────────────
class RingAvailabilityTests(_RingBase):
    def test_unavailable_without_lib(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=None):
            self.assertFalse(self.mod.is_available())

    def test_unavailable_without_token(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_read_token", return_value={}):
            self.assertFalse(self.mod.is_available())

    def test_available_with_lib_and_token(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_read_token",
                               return_value={"refresh_token": "x"}):
            self.assertTrue(self.mod.is_available())

    def test_ring_doorbell_none_when_absent(self):
        with block_import("ring_doorbell"):
            self.assertIsNone(self.mod._ring_doorbell())

    def test_ring_doorbell_returns_module_when_present(self):
        with inject_modules(ring_doorbell=_fake_ring_doorbell()):
            self.assertIsNotNone(self.mod._ring_doorbell())


# ─── token / config IO ───────────────────────────────────────────────────
class RingTokenConfigIOTests(_RingBase):
    def test_read_token_missing_file(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._read_token(), {})

    def test_read_token_valid(self):
        m = mock.mock_open(read_data='{"refresh_token": "rt"}')
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._read_token(), {"refresh_token": "rt"})

    def test_read_token_malformed(self):
        m = mock.mock_open(read_data="{bad")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._read_token(), {})

    def test_read_config_missing_and_valid(self):
        with mock.patch.object(self.mod.os.path, "exists", return_value=False):
            self.assertEqual(self.mod._read_config(), {})
        m = mock.mock_open(read_data='{"email": "a@b.com"}')
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._read_config(), {"email": "a@b.com"})

    def test_read_config_malformed(self):
        m = mock.mock_open(read_data="{bad")
        with mock.patch.object(self.mod.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", m):
            self.assertEqual(self.mod._read_config(), {})

    def test_save_token_delegates_to_atomic_io(self):
        fake_atomic = types.ModuleType("core.atomic_io")
        recorded = {}
        fake_atomic._atomic_write_json = lambda path, data: recorded.update(
            {"path": path, "data": data})
        with inject_modules(**{"core.atomic_io": fake_atomic}), \
             mock.patch.object(self.mod.os, "makedirs"):
            self.mod._save_token({"refresh_token": "rt"})
        self.assertEqual(recorded["data"], {"refresh_token": "rt"})

    def test_save_token_swallows_errors(self):
        # makedirs raising must not propagate (best-effort persistence).
        with mock.patch.object(self.mod.os, "makedirs",
                               side_effect=OSError("ro fs")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.mod._save_token({"x": 1})   # no exception


# ─── _run_with_timeout ───────────────────────────────────────────────────
class RingRunWithTimeoutTests(_RingBase):
    def test_ok_branch(self):
        res = self.mod._run_with_timeout(lambda: None, timeout=2.0)
        self.assertTrue(res["ok"])
        self.assertIsNone(res["error"])
        self.assertFalse(res["timed_out"])

    def test_error_branch(self):
        def _boom():
            raise RuntimeError("update failed")
        res = self.mod._run_with_timeout(_boom, timeout=2.0)
        self.assertFalse(res["ok"])
        self.assertIsInstance(res["error"], RuntimeError)

    def test_timed_out_branch(self):
        # Worker blocks on an event we never set within the (tiny) timeout, so
        # join() returns with the thread still alive → timed_out. We release it
        # afterward so the daemon exits cleanly.
        gate = threading.Event()

        def _block():
            gate.wait(5.0)
        try:
            res = self.mod._run_with_timeout(_block, timeout=0.1)
            self.assertTrue(res["timed_out"])
            self.assertFalse(res["ok"])
        finally:
            gate.set()


# ─── _get_ring ───────────────────────────────────────────────────────────
class RingGetRingTests(_RingBase):
    def test_none_when_lib_absent(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=None):
            self.assertIsNone(self.mod._get_ring())

    def test_returns_cached_within_ttl(self):
        sentinel = object()
        self.mod._state["ring"] = sentinel
        self.mod._state["fetched_at"] = self.mod.time.monotonic()
        with mock.patch.object(self.mod, "_ring_doorbell",
                               return_value=_fake_ring_doorbell()):
            self.assertIs(self.mod._get_ring(), sentinel)

    def test_none_when_no_token(self):
        with mock.patch.object(self.mod, "_ring_doorbell",
                               return_value=_fake_ring_doorbell()), \
             mock.patch.object(self.mod, "_read_token", return_value={}):
            self.assertIsNone(self.mod._get_ring())

    def test_none_when_auth_or_ring_missing(self):
        with mock.patch.object(self.mod, "_ring_doorbell",
                               return_value=_fake_ring_doorbell(omit_ring=True)), \
             mock.patch.object(self.mod, "_read_token",
                               return_value={"refresh_token": "rt"}):
            self.assertIsNone(self.mod._get_ring())

    def test_happy_path_constructs_and_caches(self):
        fake = _fake_ring_doorbell()
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=fake), \
             mock.patch.object(self.mod, "_read_token",
                               return_value={"refresh_token": "rt"}):
            ring = self.mod._get_ring()
        self.assertIsNotNone(ring)
        self.assertIs(self.mod._state["ring"], ring)

    def test_update_data_timeout_not_cached(self):
        fake = _fake_ring_doorbell()
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=fake), \
             mock.patch.object(self.mod, "_read_token",
                               return_value={"refresh_token": "rt"}), \
             mock.patch.object(self.mod, "_run_with_timeout",
                               return_value={"ok": False, "error": None,
                                             "timed_out": True}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.mod._get_ring())
        self.assertIsNone(self.mod._state["ring"])   # not cached

    def test_update_data_error_not_cached(self):
        fake = _fake_ring_doorbell()
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=fake), \
             mock.patch.object(self.mod, "_read_token",
                               return_value={"refresh_token": "rt"}), \
             mock.patch.object(self.mod, "_run_with_timeout",
                               return_value={"ok": False,
                                             "error": RuntimeError("boom"),
                                             "timed_out": False}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.mod._get_ring())

    def test_auth_construction_raises_returns_none(self):
        def _raise(_self):
            raise RuntimeError("bad token blob")
        fake = _fake_ring_doorbell(auth_factory=_raise)
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=fake), \
             mock.patch.object(self.mod, "_read_token",
                               return_value={"refresh_token": "rt"}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.mod._get_ring())


# ─── enumeration + list_devices ──────────────────────────────────────────
class RingEnumerationTests(_RingBase):
    def test_enumerate_none_ring(self):
        self.assertEqual(self.mod._enumerate(None), {})

    def test_enumerate_flattens_device_dict(self):
        d1 = _FakeRingDevice(name="Front Door", did="a")
        d2 = _FakeRingDevice(name="Backyard", did="b")
        fake_ring = mock.Mock()
        fake_ring.devices.return_value = {"doorbots": [d1], "stickup_cams": [d2]}
        out = self.mod._enumerate(fake_ring)
        self.assertEqual(set(out.keys()), {"a", "b"})

    def test_enumerate_handles_enum_exception(self):
        fake_ring = mock.Mock()
        fake_ring.devices.side_effect = RuntimeError("enum failed")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.mod._enumerate(fake_ring), {})

    def test_enumerate_non_dict_returns_empty(self):
        fake_ring = mock.Mock()
        fake_ring.devices.return_value = ["not", "a", "dict"]
        self.assertEqual(self.mod._enumerate(fake_ring), {})

    def test_list_devices_empty_without_client(self):
        with mock.patch.object(self.mod, "_get_ring", return_value=None):
            self.assertEqual(self.mod.list_devices(), [])

    def test_list_devices_camera_caps(self):
        # Family must contain "cam"/"doorbell" for the camera capability to be
        # derived (list_devices keys off the family string, not the dict key).
        d = _FakeRingDevice(name="Front Door", did="a", family="stickup_cams",
                            has_lights=True, has_siren=True)
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_enumerate", return_value={"a": d}):
            out = self.mod.list_devices()
        self.assertEqual(len(out), 1)
        rec = out[0]
        self.assertEqual(rec["type"], "camera")
        self.assertIn("camera", rec["capabilities"])
        self.assertIn("on_off", rec["capabilities"])
        self.assertIn("siren", rec["capabilities"])
        self.assertEqual(rec["native_id"], "a")
        self.assertEqual(rec["brand"], "Ring")

    def test_list_devices_chime_type(self):
        d = _FakeRingDevice(name="Indoor Chime", did="c", family="chimes",
                            has_lights=False, has_siren=False)
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_enumerate", return_value={"c": d}):
            out = self.mod.list_devices()
        self.assertEqual(out[0]["type"], "chime")
        self.assertIn("chime", out[0]["capabilities"])

    def test_list_devices_skips_device_that_raises(self):
        # A device whose attribute access raises mid-build is skipped, not fatal.
        good = _FakeRingDevice(name="Indoor Chime", did="c", family="chimes",
                               has_lights=False, has_siren=False)

        class _Bad:
            @property
            def name(self):
                raise RuntimeError("name boom")
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_enumerate",
                               return_value={"bad": _Bad(), "c": good}):
            out = self.mod.list_devices()
        # Only the good device survives.
        self.assertEqual([d["native_id"] for d in out], ["c"])


# ─── _match ──────────────────────────────────────────────────────────────
class RingMatchTests(_RingBase):
    def test_match_by_native_id(self):
        d1 = _FakeRingDevice(name="Front Door", did="a")
        with mock.patch.object(self.mod, "_enumerate", return_value={"a": d1}):
            found = self.mod._match(object(), {"native_id": "a"})
        self.assertIs(found, d1)

    def test_match_by_name_case_insensitive(self):
        d1 = _FakeRingDevice(name="Front Door", did="a")
        fake_ring = mock.Mock()
        fake_ring.devices.return_value = {"doorbots": [d1]}
        found = self.mod._match(fake_ring, {"name": "front door"})
        self.assertIs(found, d1)

    def test_match_returns_none_when_unmatched(self):
        d1 = _FakeRingDevice(name="Front Door", did="a")
        with mock.patch.object(self.mod, "_enumerate", return_value={"a": d1}):
            self.assertIsNone(self.mod._match(object(), {"name": "Garage"}))


# ─── get_state ───────────────────────────────────────────────────────────
class RingGetStateTests(_RingBase):
    def test_errors_without_client(self):
        with mock.patch.object(self.mod, "_get_ring", return_value=None):
            res = self.mod.get_state({"name": "Front Door"})
        self.assertIn("not authorized", res["error"])

    def test_device_not_found(self):
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=None):
            res = self.mod.get_state({"name": "Ghost"})
        self.assertIn("not found", res["error"])

    def test_reads_attributes(self):
        d = _FakeRingDevice(name="Front Door", battery=73)
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=d):
            res = self.mod.get_state({"name": "Front Door"})
        self.assertEqual(res["battery"], 73)
        self.assertEqual(res["online"], "online")
        self.assertEqual(res["last_motion"], "2026-06-01T10:00:00")
        self.assertEqual(res["lights"], "off")

    def test_state_read_exception_wrapped(self):
        # A device whose attribute access raises → outer except returns error.
        class _Dev:
            @property
            def battery_life(self):
                raise RuntimeError("read boom")
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=_Dev()):
            res = self.mod.get_state({"name": "Front Door"})
        self.assertIn("state read failed", res["error"])


# ─── degradation + list action ───────────────────────────────────────────
class RingDegradationTests(_RingBase):
    def test_set_state_errors_without_client_mentions_authorize(self):
        with mock.patch.object(self.mod, "_get_ring", return_value=None):
            res = self.mod.set_state({"name": "Front Door"}, on=True)
        self.assertIn("not authorized", res["error"])
        self.assertIn("ring_authorize", res["error"])

    def test_list_action_counts(self):
        with mock.patch.object(self.mod, "list_devices",
                               return_value=[{"name": "A"}, {"name": "B"}]):
            out = self.actions["ring_list_devices"]("")
        self.assertIn("2 Ring device", out)


# ─── set_state apply ─────────────────────────────────────────────────────
class RingSetStateApplyTests(_RingBase):
    def test_lights_on_applied(self):
        dev = _FakeRingDevice()
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=dev):
            res = self.mod.set_state({"name": "Front Door"}, on=True)
        self.assertEqual(res["applied"]["lights"], True)
        self.assertEqual(dev.lights, "on")

    def test_lights_off_applied(self):
        dev = _FakeRingDevice()
        dev.lights = "on"
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=dev):
            res = self.mod.set_state({"name": "Front Door"}, on=False)
        self.assertEqual(res["applied"]["lights"], False)
        self.assertEqual(dev.lights, "off")

    def test_siren_on_applied(self):
        dev = _FakeRingDevice()
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=dev):
            res = self.mod.set_state({"name": "Front Door"}, siren=True)
        self.assertEqual(res["applied"]["siren"], True)
        self.assertEqual(dev.siren, "on")

    def test_chime_applied(self):
        dev = _FakeRingDevice(has_lights=False, has_siren=False, has_chime=True)
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=dev):
            res = self.mod.set_state({"name": "Chime"}, chime=True)
        self.assertEqual(res["applied"]["chime"], True)
        self.assertTrue(dev.existing_doorbell_type_enabled)

    def test_lights_toggle_failure_returns_error(self):
        # Make assigning .lights raise via a property setter.
        class _Boom:
            @property
            def lights(self):
                return "off"

            @lights.setter
            def lights(self, _v):
                raise RuntimeError("light api down")
        boom = _Boom()
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=boom):
            res = self.mod.set_state({"name": "Front Door"}, on=True)
        self.assertIn("light toggle failed", res["error"])

    def test_siren_failure_returns_partial(self):
        # lights succeed, siren raises → error + partial carrying the light.
        class _Dev:
            def __init__(self):
                self.lights = "off"

            @property
            def siren(self):
                return "off"

            @siren.setter
            def siren(self, _v):
                raise RuntimeError("siren stuck")
        dev = _Dev()
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=dev):
            res = self.mod.set_state({"name": "Front Door"}, on=True, siren=True)
        self.assertIn("siren failed", res["error"])
        self.assertEqual(res["partial"]["lights"], True)

    def test_chime_failure_silently_ignored(self):
        # chime set raises but is swallowed; lights still land.
        class _Dev:
            def __init__(self):
                self.lights = "off"

            @property
            def existing_doorbell_type_enabled(self):
                return False

            @existing_doorbell_type_enabled.setter
            def existing_doorbell_type_enabled(self, _v):
                raise RuntimeError("chime api down")
        dev = _Dev()
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=dev):
            res = self.mod.set_state({"name": "Front Door"}, on=True, chime=True)
        self.assertTrue(res["ok"])
        self.assertEqual(res["applied"]["lights"], True)
        self.assertNotIn("chime", res["applied"])   # swallowed

    def test_no_supported_controls_returns_informative_error(self):
        dev = _FakeRingDevice(has_lights=False, has_siren=False)
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=dev):
            res = self.mod.set_state({"name": "Chime"}, on=True, siren=True)
        self.assertIn("doesn't expose", res["error"])
        self.assertIn("on", res["requested"])

    def test_device_not_found(self):
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=None):
            res = self.mod.set_state({"name": "Ghost"}, on=True)
        self.assertIn("not found", res["error"])

    def test_outer_exception_wrapped(self):
        # hasattr() only swallows AttributeError; a property raising a non-Attr
        # error propagates out of the inner gate into the outer except wrapper.
        class _Dev:
            @property
            def lights(self):
                raise RuntimeError("hardware fault")
        with mock.patch.object(self.mod, "_get_ring", return_value=object()), \
             mock.patch.object(self.mod, "_match", return_value=_Dev()):
            res = self.mod.set_state({"name": "Front Door"}, on=True)
        self.assertIn("set_state failed", res["error"])
        self.assertIn("partial", res)


# ─── _do_fetch_token ─────────────────────────────────────────────────────
class RingDoFetchTokenTests(_RingBase):
    def test_no_lib(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=None):
            out = self.mod._do_fetch_token("a@b.com", "pw")
        self.assertIn("not installed", out)

    def test_no_auth_class(self):
        with mock.patch.object(self.mod, "_ring_doorbell",
                               return_value=_fake_ring_doorbell(omit_auth=True)):
            out = self.mod._do_fetch_token("a@b.com", "pw")
        self.assertIn("Auth missing", out)

    def test_auth_construction_failure(self):
        def _raise(_self):
            raise RuntimeError("construct boom")
        with mock.patch.object(self.mod, "_ring_doorbell",
                               return_value=_fake_ring_doorbell(auth_factory=_raise)):
            out = self.mod._do_fetch_token("a@b.com", "pw")
        self.assertIn("construction failed", out)

    def test_success_no_code(self):
        with mock.patch.object(self.mod, "_ring_doorbell",
                               return_value=_fake_ring_doorbell()):
            out = self.mod._do_fetch_token("a@b.com", "pw")
        self.assertIn("authorized", out)

    def test_2fa_required_when_fetch_raises_without_code(self):
        fake = _fake_ring_doorbell()
        fake.Auth.fetch_token = lambda self, *a, **k: (_ for _ in ()).throw(
            RuntimeError("Need 2FA"))
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=fake):
            out = self.mod._do_fetch_token("a@b.com", "pw")
        self.assertTrue(out.startswith("2FA_REQUIRED:"))

    def test_failure_with_code(self):
        fake = _fake_ring_doorbell()
        fake.Auth.fetch_token = lambda self, *a, **k: (_ for _ in ()).throw(
            RuntimeError("bad code"))
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=fake):
            out = self.mod._do_fetch_token("a@b.com", "pw", "000000")
        self.assertIn("sign-in failed", out)


# ─── ring_authorize ──────────────────────────────────────────────────────
class RingAuthorizeTests(_RingBase):
    def test_authorize_without_lib(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=None):
            out = self.actions["ring_authorize"]("")
        self.assertIn("not installed", out)

    def test_authorize_no_creds_returns_cli_hint(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_read_config", return_value={}):
            out = self.actions["ring_authorize"]("")
        self.assertIn("interactive terminal", out)

    def test_authorize_inline_creds_calls_fetch(self):
        fetch = mock.Mock(return_value="Ring authorized, sir — token persisted.")
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_do_fetch_token", fetch):
            out = self.actions["ring_authorize"]("me@x.com|pw")
        fetch.assert_called_once_with("me@x.com", "pw", "")
        self.assertIn("authorized", out)

    def test_authorize_inline_creds_with_code(self):
        fetch = mock.Mock(return_value="Ring authorized, sir.")
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_do_fetch_token", fetch):
            self.actions["ring_authorize"]("me@x.com|pw|123456")
        fetch.assert_called_once_with("me@x.com", "pw", "123456")

    def test_authorize_uses_cached_config(self):
        fetch = mock.Mock(return_value="Ring authorized, sir.")
        cfg = {"email": "cfg@x.com", "password": "cpw", "code": ""}
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_read_config", return_value=cfg), \
             mock.patch.object(self.mod, "_do_fetch_token", fetch):
            self.actions["ring_authorize"]("")
        fetch.assert_called_once_with("cfg@x.com", "cpw", "")

    def test_authorize_2fa_required_message(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_do_fetch_token",
                               return_value="2FA_REQUIRED: need code"):
            out = self.actions["ring_authorize"]("me@x.com|pw")
        self.assertIn("2FA code", out)


# ─── interactive wizard ──────────────────────────────────────────────────
class RingWizardTests(_RingBase):
    def test_wizard_without_lib(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=None):
            out = self.mod._run_ring_wizard_interactive("")
        self.assertIn("not installed", out)

    def test_wizard_inline_creds_success(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_do_fetch_token",
                               return_value="Ring authorized, sir."):
            out = self.mod._run_ring_wizard_interactive("me@x.com|pw")
        self.assertIn("authorized", out)

    def test_wizard_prompts_when_no_creds(self):
        # No inline args, empty config → prompts via input()/getpass().
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_read_config", return_value={}), \
             mock.patch("builtins.input", return_value="typed@x.com"), \
             mock.patch("getpass.getpass", return_value="typedpw"), \
             mock.patch.object(self.mod, "_do_fetch_token",
                               return_value="Ring authorized, sir.") as fetch, \
             contextlib.redirect_stdout(io.StringIO()):
            out = self.mod._run_ring_wizard_interactive("")
        self.assertIn("authorized", out)
        fetch.assert_called_with("typed@x.com", "typedpw")

    def test_wizard_prompt_cancelled(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_read_config", return_value={}), \
             mock.patch("builtins.input", side_effect=EOFError), \
             contextlib.redirect_stdout(io.StringIO()):
            out = self.mod._run_ring_wizard_interactive("")
        self.assertIn("No credentials", out)

    def test_wizard_prompt_empty_email(self):
        # input() returns blank email → the post-prompt empty check returns.
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_read_config", return_value={}), \
             mock.patch("builtins.input", return_value="   "), \
             mock.patch("getpass.getpass", return_value=""), \
             contextlib.redirect_stdout(io.StringIO()):
            out = self.mod._run_ring_wizard_interactive("")
        self.assertIn("No credentials", out)

    def test_wizard_2fa_empty_code(self):
        # 2FA required, but the code prompt returns blank → "No 2FA code".
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_do_fetch_token",
                               return_value="2FA_REQUIRED: need code"), \
             mock.patch("builtins.input", return_value="  "), \
             contextlib.redirect_stdout(io.StringIO()):
            out = self.mod._run_ring_wizard_interactive("me@x.com|pw")
        self.assertIn("No 2FA code", out)

    def test_wizard_2fa_then_complete(self):
        # First fetch returns 2FA_REQUIRED, then a code prompt + retry succeeds.
        results = ["2FA_REQUIRED: need code", "Ring authorized, sir."]
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_do_fetch_token",
                               side_effect=results), \
             mock.patch("builtins.input", return_value="654321"), \
             contextlib.redirect_stdout(io.StringIO()):
            out = self.mod._run_ring_wizard_interactive("me@x.com|pw")
        self.assertIn("authorized", out)

    def test_wizard_2fa_code_cancelled(self):
        with mock.patch.object(self.mod, "_ring_doorbell", return_value=object()), \
             mock.patch.object(self.mod, "_do_fetch_token",
                               return_value="2FA_REQUIRED: need code"), \
             mock.patch("builtins.input", side_effect=KeyboardInterrupt), \
             contextlib.redirect_stdout(io.StringIO()):
            out = self.mod._run_ring_wizard_interactive("me@x.com|pw")
        self.assertIn("cancelled", out)


# ─── register ────────────────────────────────────────────────────────────
class RingRegisterTests(_RingBase):
    def test_register_wires_both_actions(self):
        acts = {}
        self.mod.register(acts)
        self.assertIn("ring_list_devices", acts)
        self.assertIn("ring_authorize", acts)


if __name__ == "__main__":
    unittest.main()
