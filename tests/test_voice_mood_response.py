"""Tests for adapters.voice_mood_response — the stressed-user reflex adapter.

When route_voice_emotion() flags the user 'stressed', this adapter (1) persists
a 15-minute proactive-suppression timestamp into memory, (2) fires a
non-blocking calming-lights smart-home dim, and (3) returns a deferential
system-prompt addendum. All three are driven by apply_voice_mood_response();
the smart-home call is gated on a configured catalog and de-duplicated via a
single-shot in-flight Event.

core.smart_home_router is imported lazily inside the adapter, so every test
injects a fake module into sys.modules (or patches the adapter's helpers) —
no real Hue/Govee devices, no real memory file, no network. The daemon thread
is joined deterministically before assertions. stdlib unittest + mock only.

CI-safe: the adapter imports cleanly on bare Linux (smart_home_router is a
deferred import), and these tests never touch it for real.
"""
from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from unittest import mock

from adapters import voice_mood_response as vmr


def _make_fake_router(*, devices=True, control_result="dimmed 3 lights",
                      ensure_raises=False, control_raises=False):
    """Build a stand-in core.smart_home_router module.

    devices=True  → _ensure_catalog() returns a catalog WITH a device.
    devices=False → returns an empty catalog (catalog_ready() False).
    """
    mod = types.ModuleType("core.smart_home_router")

    def _ensure_catalog():
        if ensure_raises:
            raise RuntimeError("router exploded")
        return {"devices": [{"id": "light1"}]} if devices else {"devices": []}

    def smart_home_control(cmd):
        if control_raises:
            raise RuntimeError("device offline")
        return control_result

    mod._ensure_catalog = _ensure_catalog
    mod.smart_home_control = smart_home_control
    return mod


class _RouterInjectingBase(unittest.TestCase):
    """Inject a fake core.smart_home_router and always clear the in-flight
    Event so a leaked set() from one test can't suppress the next."""

    def _inject_router(self, router):
        # The adapter does `from core import smart_home_router`, which resolves
        # against BOTH sys.modules["core.smart_home_router"] AND the
        # `smart_home_router` attribute on the already-imported `core` package.
        # We set/restore both so the injection is honoured whether or not the
        # real submodule was ever imported — and so it survives the CI-sim
        # runner's module eviction + patched import_module. (router=None →
        # remove the entry so `from core import ...` raises, exercising the
        # import-failure guard.)
        import core as core_pkg
        old_mod = sys.modules.get("core.smart_home_router")
        had_attr = hasattr(core_pkg, "smart_home_router")
        old_attr = getattr(core_pkg, "smart_home_router", None)

        if router is None:
            sys.modules.pop("core.smart_home_router", None)
            if had_attr:
                delattr(core_pkg, "smart_home_router")
        else:
            sys.modules["core.smart_home_router"] = router
            setattr(core_pkg, "smart_home_router", router)

        def restore():
            if old_mod is not None:
                sys.modules["core.smart_home_router"] = old_mod
            else:
                sys.modules.pop("core.smart_home_router", None)
            if had_attr:
                setattr(core_pkg, "smart_home_router", old_attr)
            elif hasattr(core_pkg, "smart_home_router"):
                delattr(core_pkg, "smart_home_router")
        self.addCleanup(restore)

    def setUp(self):
        vmr._smart_home_in_flight.clear()
        self.addCleanup(vmr._smart_home_in_flight.clear)

    def _join_lights_thread(self, timeout=2.0):
        """Wait for the daemon dim thread (named 'voice-mood-lights') to finish
        so its print/clear side effects are observable before we assert."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            alive = [t for t in threading.enumerate()
                     if t.name == "voice-mood-lights"]
            if not alive:
                return
            for t in alive:
                t.join(timeout=0.1)
        # Last-ditch join.
        for t in threading.enumerate():
            if t.name == "voice-mood-lights":
                t.join(timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────
# is_stress_suppression_active
# ─────────────────────────────────────────────────────────────────────────
class StressSuppressionActiveTests(unittest.TestCase):
    def test_empty_memory_false(self):
        self.assertFalse(vmr.is_stress_suppression_active({}))
        self.assertFalse(vmr.is_stress_suppression_active(None))

    def test_missing_key_false(self):
        self.assertFalse(vmr.is_stress_suppression_active({"other": 1}))

    def test_non_numeric_value_false(self):
        self.assertFalse(
            vmr.is_stress_suppression_active({vmr.MEMORY_KEY: "soon"}))

    def test_future_timestamp_active(self):
        future = time.time() + 500
        self.assertTrue(
            vmr.is_stress_suppression_active({vmr.MEMORY_KEY: future}))

    def test_past_timestamp_inactive(self):
        past = time.time() - 10
        self.assertFalse(
            vmr.is_stress_suppression_active({vmr.MEMORY_KEY: past}))

    def test_explicit_now_arg_used(self):
        # until=100; now=50 → active; now=150 → inactive.
        self.assertTrue(
            vmr.is_stress_suppression_active({vmr.MEMORY_KEY: 100.0}, now=50.0))
        self.assertFalse(
            vmr.is_stress_suppression_active({vmr.MEMORY_KEY: 100.0}, now=150.0))

    def test_int_value_accepted(self):
        self.assertTrue(
            vmr.is_stress_suppression_active(
                {vmr.MEMORY_KEY: int(time.time()) + 500}))


# ─────────────────────────────────────────────────────────────────────────
# _smart_home_catalog_ready
# ─────────────────────────────────────────────────────────────────────────
class CatalogReadyTests(_RouterInjectingBase):
    def test_ready_with_devices(self):
        self._inject_router(_make_fake_router(devices=True))
        self.assertTrue(vmr._smart_home_catalog_ready())

    def test_not_ready_when_no_devices(self):
        self._inject_router(_make_fake_router(devices=False))
        self.assertFalse(vmr._smart_home_catalog_ready())

    def test_ensure_catalog_raise_is_not_ready(self):
        self._inject_router(_make_fake_router(ensure_raises=True))
        self.assertFalse(vmr._smart_home_catalog_ready())

    def test_import_failure_is_not_ready(self):
        # smart_home_router = None → `from core import smart_home_router`
        # still binds the None module, but _ensure_catalog lookup raises →
        # caught → not ready.
        self._inject_router(None)
        self.assertFalse(vmr._smart_home_catalog_ready())


# ─────────────────────────────────────────────────────────────────────────
# _dispatch_calming_lights (the daemon-thread target, run inline here)
# ─────────────────────────────────────────────────────────────────────────
class DispatchCalmingLightsTests(_RouterInjectingBase):
    def test_calls_control_and_clears_flag(self):
        self._inject_router(_make_fake_router(control_result="dimmed ok"))
        vmr._smart_home_in_flight.set()
        with mock.patch("builtins.print") as mprint:
            vmr._dispatch_calming_lights()
        # flag cleared in finally
        self.assertFalse(vmr._smart_home_in_flight.is_set())
        self.assertTrue(any("dimmed ok" in str(c) for c in mprint.call_args_list))

    def test_control_exception_still_clears_flag(self):
        self._inject_router(_make_fake_router(control_raises=True))
        vmr._smart_home_in_flight.set()
        with mock.patch("builtins.print") as mprint:
            vmr._dispatch_calming_lights()
        self.assertFalse(vmr._smart_home_in_flight.is_set())
        self.assertTrue(
            any("dispatch failed" in str(c) for c in mprint.call_args_list))

    def test_passes_expected_command(self):
        seen = {}
        router = _make_fake_router()
        orig = router.smart_home_control
        router.smart_home_control = lambda cmd: seen.setdefault("cmd", cmd) or orig(cmd)
        self._inject_router(router)
        vmr._smart_home_in_flight.set()
        with mock.patch("builtins.print"):
            vmr._dispatch_calming_lights()
        self.assertIn("dim all lights", seen["cmd"])
        self.assertIn("2700K", seen["cmd"])


# ─────────────────────────────────────────────────────────────────────────
# _kick_smart_home
# ─────────────────────────────────────────────────────────────────────────
class KickSmartHomeTests(_RouterInjectingBase):
    def test_skips_when_no_catalog(self):
        self._inject_router(_make_fake_router(devices=False))
        self.assertFalse(vmr._kick_smart_home())
        self.assertFalse(vmr._smart_home_in_flight.is_set())

    def test_returns_true_when_already_in_flight(self):
        self._inject_router(_make_fake_router(devices=True))
        vmr._smart_home_in_flight.set()        # pretend a dim is already running
        # Catalog is ready, but in-flight → returns True without a new thread.
        before = threading.active_count()
        self.assertTrue(vmr._kick_smart_home())
        self.assertEqual(threading.active_count(), before)

    def test_dispatches_thread_and_returns_true(self):
        self._inject_router(_make_fake_router(devices=True))
        self.assertTrue(vmr._kick_smart_home())
        self._join_lights_thread()
        # After the daemon finishes it clears the in-flight flag.
        self.assertFalse(vmr._smart_home_in_flight.is_set())


# ─────────────────────────────────────────────────────────────────────────
# apply_voice_mood_response — the top-level entry point
# ─────────────────────────────────────────────────────────────────────────
class ApplyVoiceMoodResponseTests(_RouterInjectingBase):
    def test_no_route_is_noop(self):
        self.assertEqual(vmr.apply_voice_mood_response(None, "hi"), "")

    def test_non_stressed_mood_is_noop(self):
        self.assertEqual(
            vmr.apply_voice_mood_response({"mood": "calm"}, "hi"), "")

    def test_stressed_returns_addendum_and_persists(self):
        # Catalog has no devices so no thread is spawned (keeps the test fast);
        # we focus on the memory-persist + addendum contract.
        self._inject_router(_make_fake_router(devices=False))
        store = {}
        lock = threading.RLock()

        def load_memory():
            return dict(store)

        def save_memory(m):
            store.clear()
            store.update(m)

        with mock.patch("builtins.print"):
            out = vmr.apply_voice_mood_response(
                {"mood": "stressed"}, "everything is on fire",
                memory_lock=lock, load_memory=load_memory,
                save_memory=save_memory)
        self.assertEqual(out, vmr.STRESS_DEFERENTIAL_ADDENDUM)
        # A future suppression deadline was written.
        self.assertIn(vmr.MEMORY_KEY, store)
        self.assertGreater(store[vmr.MEMORY_KEY], time.time())

    def test_stressed_persists_without_lock(self):
        # memory_lock=None path: still loads + saves.
        self._inject_router(_make_fake_router(devices=False))
        store = {}
        with mock.patch("builtins.print"):
            out = vmr.apply_voice_mood_response(
                {"mood": "stressed"}, "panic",
                memory_lock=None,
                load_memory=lambda: dict(store),
                save_memory=lambda m: (store.clear(), store.update(m)))
        self.assertEqual(out, vmr.STRESS_DEFERENTIAL_ADDENDUM)
        self.assertIn(vmr.MEMORY_KEY, store)

    def test_stressed_skips_persist_when_helpers_missing(self):
        # No load/save callables → persist block skipped, addendum still returned.
        self._inject_router(_make_fake_router(devices=False))
        with mock.patch("builtins.print"):
            out = vmr.apply_voice_mood_response({"mood": "stressed"}, "x")
        self.assertEqual(out, vmr.STRESS_DEFERENTIAL_ADDENDUM)

    def test_memory_persist_exception_swallowed(self):
        # load_memory raises → except prints, addendum still returned.
        self._inject_router(_make_fake_router(devices=False))

        def boom():
            raise RuntimeError("memory disk gone")

        with mock.patch("builtins.print") as mprint:
            out = vmr.apply_voice_mood_response(
                {"mood": "stressed"}, "x",
                load_memory=boom, save_memory=lambda m: None)
        self.assertEqual(out, vmr.STRESS_DEFERENTIAL_ADDENDUM)
        self.assertTrue(
            any("memory persist failed" in str(c) for c in mprint.call_args_list))

    def test_stressed_with_catalog_logs_dimming(self):
        # Catalog ready → _kick_smart_home returns True → "; dimming lights"
        # appears in the suppression log line.
        self._inject_router(_make_fake_router(devices=True))
        with mock.patch("builtins.print") as mprint:
            out = vmr.apply_voice_mood_response({"mood": "stressed"}, "x")
        self._join_lights_thread()
        self.assertEqual(out, vmr.STRESS_DEFERENTIAL_ADDENDUM)
        joined = " ".join(str(c) for c in mprint.call_args_list)
        self.assertIn("dimming lights", joined)

    def test_stressed_no_catalog_omits_dimming_note(self):
        self._inject_router(_make_fake_router(devices=False))
        with mock.patch("builtins.print") as mprint:
            vmr.apply_voice_mood_response({"mood": "stressed"}, "x")
        joined = " ".join(str(c) for c in mprint.call_args_list)
        # The suppression line is present but WITHOUT the dimming suffix.
        self.assertIn("proactive suppression", joined)
        self.assertNotIn("dimming lights", joined)


if __name__ == "__main__":
    unittest.main()
