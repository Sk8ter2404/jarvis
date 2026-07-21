"""Tests for skills/kinect_pointing — the point-to-control wiring: calibration,
point_control dispatch through the EXISTING smart-home action, the honest
no-ops, the toggle/persistence, and the "turn that on" pointing integration.

Loads the skill in isolation (no monolith boot) via the shared harness, with a
fake kinect_bridge injected into sys.modules and the smart-home action MOCKED
(no real device is ever controlled). The calibration store is redirected to a
TMP path via JARVIS_POINTING_PATH, and the settings writer is redirected via
JARVIS_SETTINGS_PATH + a patched tools.settings_window — so neither the real
data/kinect_pointing.json nor data/user_settings.json is touched.

stdlib unittest + mock.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

from audio import kinect_pointing as kp


# ─── settings-file + pointing-file safety nets (module-wide) ────────────────
_SAVED_SETTINGS_ENV: "str | None" = None
_SAVED_POINTING_ENV: "str | None" = None
_TMPDIR: "str | None" = None


def setUpModule() -> None:
    global _SAVED_SETTINGS_ENV, _SAVED_POINTING_ENV, _TMPDIR
    _SAVED_SETTINGS_ENV = os.environ.get("JARVIS_SETTINGS_PATH")
    _SAVED_POINTING_ENV = os.environ.get("JARVIS_POINTING_PATH")
    _TMPDIR = tempfile.mkdtemp(prefix="jarvis_pointskill_test_")
    os.environ["JARVIS_SETTINGS_PATH"] = os.path.join(
        _TMPDIR, "test_user_settings.json")
    os.environ["JARVIS_POINTING_PATH"] = os.path.join(
        _TMPDIR, "test_kinect_pointing.json")


def tearDownModule() -> None:
    for key, saved in (("JARVIS_SETTINGS_PATH", _SAVED_SETTINGS_ENV),
                       ("JARVIS_POINTING_PATH", _SAVED_POINTING_ENV)):
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved


# ─── fakes ──────────────────────────────────────────────────────────────────
def _arm_body(direction=(1.0, 0.0, 0.0), state=2, distance_m=2.0):
    """A get_bodies()-shaped body whose right arm points along `direction`
    (shoulder at origin, hand one unit along the direction)."""
    sx, sy, sz = 0.0, 0.0, 2.0
    hx, hy, hz = sx + direction[0], sy + direction[1], sz + direction[2]
    mx, my, mz = (sx + hx) / 2, (sy + hy) / 2, (sz + hz) / 2
    j = {
        "shoulder_right": (sx, sy, sz, state),
        "elbow_right": (mx, my, mz, state),
        "hand_right": (hx, hy, hz, state),
        "hand_tip_right": (hx + 0.05, hy, hz, state),
    }
    return {"id": 0, "joints": j, "distance_m": distance_m,
            "head": None, "facing": None}


def _fake_bridge(*, enabled=True, available=(True, ""), bodies=None):
    m = types.ModuleType("audio.kinect_bridge")
    m.get_enabled = lambda: enabled
    m.available = lambda: available
    m.get_bodies = lambda: (bodies if bodies is not None else [])
    m.get_presence = lambda: {"present": bool(bodies), "count": len(bodies or []),
                              "nearest_m": None, "facing": None, "ts": 0.0}
    return m


class _Base(unittest.TestCase):
    def _inject(self, name, module):
        old = sys.modules.get(name)
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
        self.addCleanup(
            lambda: sys.modules.__setitem__(name, old) if old is not None
            else sys.modules.pop(name, None))

    def _load(self):
        mod, _actions = load_skill_isolated("kinect_pointing", register=False)
        return mod

    def _patch_flag(self, value):
        from core import config as cfg
        p = mock.patch.object(cfg, "KINECT_POINT_CONTROL_ENABLED", value,
                              create=True)
        p.start()
        self.addCleanup(p.stop)

    def _not_staging(self, mod):
        p = mock.patch.object(mod, "_is_staging", lambda: False)
        p.start()
        self.addCleanup(p.stop)

    def _mock_smart_home(self, mod, reply="On, sir — Desk Lamp."):
        """Replace the skill's smart-home seam with a recording mock so no real
        device is touched and we can assert the exact utterance it fired."""
        calls = []

        def _fake(utterance):
            calls.append(utterance)
            return reply

        p = mock.patch.object(mod, "_smart_home_control", _fake)
        p.start()
        self.addCleanup(p.stop)
        return calls

    def _fresh_store_path(self):
        """A throwaway store path for this test, set via the env the store
        reads (so both the skill and any direct kp.PointingStore() agree)."""
        d = tempfile.mkdtemp(prefix="jarvis_pt_")
        path = os.path.join(d, "pointing.json")
        p = mock.patch.dict(os.environ, {"JARVIS_POINTING_PATH": path})
        p.start()
        self.addCleanup(p.stop)
        # rmtree the whole dir (handles the atomic-writer's .tmp stragglers and
        # ignores a Windows lock); LIFO cleanup means this runs after p.stop.
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        return path

    def _calibrate(self, path, name, direction, device=None):
        kp.PointingStore(path=path).put(name, direction, device=device)


# ─── point_control: resolves + fires the smart-home action ──────────────────
class PointControlTests(_Base):
    def test_resolves_and_fires_on(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._calibrate(path, "desk lamp", (1.0, 0.0, 0.0), device="Desk Lamp")
        # User points along +x (at the desk lamp).
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        calls = self._mock_smart_home(mod, reply="On, sir — Desk Lamp.")
        out = mod.point_control("on")
        # Fired the existing smart-home path with the bound device + state.
        self.assertEqual(calls, ["turn on Desk Lamp"])
        self.assertIn("desk lamp", out.lower())

    def test_resolves_and_fires_off(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._calibrate(path, "fan", (0.0, 0.0, 1.0))   # device defaults to name
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((0.0, 0.0, 1.0))]))
        calls = self._mock_smart_home(mod, reply="Off, sir — fan.")
        mod.point_control("off")
        self.assertEqual(calls, ["turn off fan"])

    def test_toggle_uses_toggle_phrasing(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._calibrate(path, "lamp", (1.0, 0.0, 0.0), device="Lamp")
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        calls = self._mock_smart_home(mod, reply="Toggled, sir — Lamp.")
        mod.point_control("toggle")
        self.assertEqual(calls, ["toggle Lamp"])

    def test_noop_when_flag_off(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(False)
        path = self._fresh_store_path()
        self._calibrate(path, "lamp", (1.0, 0.0, 0.0))
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        calls = self._mock_smart_home(mod)
        out = mod.point_control("on")
        self.assertEqual(calls, [])                  # never dispatched
        self.assertIn("off", out.lower())            # honest: point-control off

    def test_noop_when_sensor_absent(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._fresh_store_path()
        self._inject("audio.kinect_bridge", None)    # bridge not loaded
        calls = self._mock_smart_home(mod)
        out = mod.point_control("on")
        self.assertEqual(calls, [])
        self.assertIn("can't tell where you're pointing", out.lower())

    def test_noop_when_sensor_disabled(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._fresh_store_path()
        self._inject("audio.kinect_bridge", _fake_bridge(enabled=False))
        calls = self._mock_smart_home(mod)
        out = mod.point_control("on")
        self.assertEqual(calls, [])
        self.assertIn("switched off", out.lower())

    def test_noop_when_not_pointing(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._calibrate(path, "lamp", (1.0, 0.0, 0.0))
        # No body in view → no pointing direction.
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[]))
        calls = self._mock_smart_home(mod)
        out = mod.point_control("on")
        self.assertEqual(calls, [])
        self.assertIn("don't see you pointing", out.lower())

    def test_noop_when_point_matches_nothing(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._calibrate(path, "lamp", (1.0, 0.0, 0.0))   # calibrated +x
        # User points straight UP (+y) — 90° off the only target → no match.
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((0.0, 1.0, 0.0))]))
        calls = self._mock_smart_home(mod)
        out = mod.point_control("on")
        self.assertEqual(calls, [])
        self.assertIn("not pointing at anything", out.lower())

    def test_noop_when_no_state(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._fresh_store_path()
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        calls = self._mock_smart_home(mod)
        out = mod.point_control("")          # no on/off
        self.assertEqual(calls, [])
        self.assertIn("on or off", out.lower())

    def test_staging_never_controls(self):
        mod = self._load()
        # Force staging True.
        p = mock.patch.object(mod, "_is_staging", lambda: True)
        p.start(); self.addCleanup(p.stop)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._calibrate(path, "lamp", (1.0, 0.0, 0.0))
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        calls = self._mock_smart_home(mod)
        out = mod.point_control("on")
        self.assertEqual(calls, [])
        self.assertIn("staging", out.lower())

    def test_surfaces_smart_home_failure(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._calibrate(path, "lamp", (1.0, 0.0, 0.0), device="Lamp")
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        # smart_home_control reports a miss → point_control must NOT claim success.
        self._mock_smart_home(
            mod, reply="I don't see anything in the catalog matching 'Lamp', sir.")
        out = mod.point_control("on")
        self.assertIn("didn't go through", out.lower())


# ─── "turn that on" natural-phrase integration ──────────────────────────────
class PronounIntegrationTests(_Base):
    def test_is_pronoun_device_command(self):
        mod = self._load()
        for yes in ("turn that on", "turn that off", "that one off",
                    "switch this on", "that on", "this one on", "Turn That On.",
                    # 2026-07-21 audit (stale lead-filler rule): Whisper's
                    # comma wake form and STACKED fillers must match too.
                    "JARVIS, turn that on", "hey jarvis, that one off",
                    "could you please turn that off"):
            self.assertTrue(mod.is_pronoun_device_command(yes), yes)
        for no in ("turn off the office light", "turn on the desk lamp",
                   "set the bedroom to 65", "play music", "", "lights on",
                   "JARVIS, turn off the office light"):
            self.assertFalse(mod.is_pronoun_device_command(no), no)

    def test_resolve_pointing_command_fires_when_active(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._calibrate(path, "desk lamp", (1.0, 0.0, 0.0), device="Desk Lamp")
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        calls = self._mock_smart_home(mod, reply="On, sir — Desk Lamp.")
        out = mod.resolve_pointing_command("turn that on")
        self.assertIsNotNone(out)
        self.assertEqual(calls, ["turn on Desk Lamp"])

    def test_resolve_returns_none_when_flag_off(self):
        # Flag off → returns None so the caller's existing behaviour is unchanged.
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(False)
        path = self._fresh_store_path()
        self._calibrate(path, "lamp", (1.0, 0.0, 0.0))
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        calls = self._mock_smart_home(mod)
        self.assertIsNone(mod.resolve_pointing_command("turn that on"))
        self.assertEqual(calls, [])

    def test_resolve_returns_none_for_named_device(self):
        # A NAMED device command is not a pronoun command → None (the router
        # handles it normally). Active flag + pointing must not hijack it.
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._calibrate(path, "lamp", (1.0, 0.0, 0.0))
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        calls = self._mock_smart_home(mod)
        self.assertIsNone(mod.resolve_pointing_command("turn off the office light"))
        self.assertEqual(calls, [])

    def test_resolve_returns_none_when_not_pointing(self):
        # Pronoun command + active, but the user isn't pointing → None, so the
        # existing ask-which-device flow proceeds unchanged.
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._calibrate(path, "lamp", (1.0, 0.0, 0.0))
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[]))
        calls = self._mock_smart_home(mod)
        self.assertIsNone(mod.resolve_pointing_command("turn that on"))
        self.assertEqual(calls, [])

    def test_resolve_returns_none_when_point_unmatched(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._calibrate(path, "lamp", (1.0, 0.0, 0.0))   # +x
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((0.0, 1.0, 0.0))]))  # +y
        calls = self._mock_smart_home(mod)
        self.assertIsNone(mod.resolve_pointing_command("turn that on"))
        self.assertEqual(calls, [])


# ─── router-level integration: smart_home_control routes via pointing ───────
class RouterHookTests(_Base):
    def test_router_routes_pronoun_through_pointing(self):
        """core.smart_home_router.smart_home_control('turn that on') must call
        the pointing resolver first and return its result when it resolves."""
        from core import smart_home_router as shr
        # Stub the pointing seam the router imports lazily.
        fake = types.ModuleType("skills.kinect_pointing")
        fake.is_pronoun_device_command = lambda u: "that" in (u or "").lower()
        fake.resolve_pointing_command = lambda u: "On, sir — Desk Lamp (pointed)."
        self._inject("skills.kinect_pointing", fake)
        out = shr.smart_home_control("turn that on")
        self.assertEqual(out, "On, sir — Desk Lamp (pointed).")

    def test_router_unchanged_when_pointing_returns_none(self):
        """When pointing resolves nothing, the router falls through to its
        existing behaviour (here: the empty-catalog message) — non-breaking."""
        from core import smart_home_router as shr
        fake = types.ModuleType("skills.kinect_pointing")
        fake.is_pronoun_device_command = lambda u: True
        fake.resolve_pointing_command = lambda u: None    # nothing resolved
        self._inject("skills.kinect_pointing", fake)
        # Force the catalog empty so we hit the deterministic no-catalog branch
        # (live-LAN fallback stubbed to a miss so no real discovery runs).
        with mock.patch.object(shr, "_ensure_catalog", lambda *a, **k: None), \
             mock.patch.object(shr, "_live_lan_fallback", lambda u: None):
            out = shr.smart_home_control("turn that on")
        self.assertIn("no smart-home catalog", out.lower())

    def test_router_named_command_skips_pointing(self):
        """A normal named command must NOT invoke the pointing resolver."""
        from core import smart_home_router as shr
        called = []
        fake = types.ModuleType("skills.kinect_pointing")
        fake.is_pronoun_device_command = lambda u: False   # not a pronoun
        fake.resolve_pointing_command = lambda u: called.append(u) or "X"
        self._inject("skills.kinect_pointing", fake)
        with mock.patch.object(shr, "_ensure_catalog", lambda *a, **k: None), \
             mock.patch.object(shr, "_live_lan_fallback", lambda u: None):
            out = shr.smart_home_control("turn off the office light")
        self.assertEqual(called, [])
        self.assertIn("no smart-home catalog", out.lower())


# ─── calibration writes the store (to the TMP path) ─────────────────────────
class CalibrateTests(_Base):
    def test_calibrate_writes_store(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        # Sample instantly + deterministically (no real sleeping / wall-clock).
        self._patch_fast_sample(mod, direction=(1.0, 0.0, 0.0), frames=18,
                                pointing=18)
        out = mod.point_calibrate("desk lamp")
        self.assertIn("remember", out.lower())
        # The store on the TMP path now has the target.
        store = kp.PointingStore(path=path)
        targets = store.list_targets()
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["name"], "desk lamp")
        # Real data file untouched (we wrote only the tmp path).
        self.assertTrue(path.endswith("pointing.json"))

    def test_calibrate_honest_when_not_pointing(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._fresh_store_path()
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[]))
        self._patch_fast_sample(mod, direction=None, frames=18, pointing=0)
        out = mod.point_calibrate("desk lamp")
        self.assertIn("couldn't see you pointing", out.lower())

    def test_calibrate_honest_when_unsteady(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._fresh_store_path()
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        # Frames seen, but averaging rejected them (unsteady) → direction None.
        self._patch_fast_sample(mod, direction=None, frames=18, pointing=12)
        out = mod.point_calibrate("desk lamp")
        self.assertIn("unsteady", out.lower())

    def test_calibrate_requires_name(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        out = mod.point_calibrate("")
        self.assertIn("which device", out.lower())

    def test_calibrate_honest_when_flag_off(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(False)
        out = mod.point_calibrate("desk lamp")
        self.assertIn("off", out.lower())

    def test_calibrate_honest_when_sensor_off(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._fresh_store_path()
        self._inject("audio.kinect_bridge", _fake_bridge(enabled=False))
        out = mod.point_calibrate("desk lamp")
        self.assertIn("can't calibrate", out.lower())

    def _patch_fast_sample(self, mod, *, direction, frames, pointing):
        p = mock.patch.object(mod, "_sample_direction",
                              lambda *a, **k: (direction, frames, pointing))
        p.start()
        self.addCleanup(p.stop)


# ─── list / forget ──────────────────────────────────────────────────────────
class ListForgetTests(_Base):
    def test_list_empty(self):
        mod = self._load()
        self._fresh_store_path()
        self.assertIn("nothing", mod.list_point_targets("").lower())

    def test_list_targets(self):
        mod = self._load()
        path = self._fresh_store_path()
        self._calibrate(path, "desk lamp", (1.0, 0.0, 0.0), device="Office Lamp")
        self._calibrate(path, "fan", (0.0, 0.0, 1.0))
        out = mod.list_point_targets("")
        self.assertIn("desk lamp", out.lower())
        self.assertIn("fan", out.lower())
        self.assertIn("office lamp", out.lower())   # shows the bound device

    def test_forget(self):
        mod = self._load()
        path = self._fresh_store_path()
        self._calibrate(path, "lamp", (1.0, 0.0, 0.0))
        out = mod.forget_point_target("lamp")
        self.assertIn("forgotten", out.lower())
        self.assertEqual(kp.PointingStore(path=path).list_targets(), [])

    def test_forget_unknown(self):
        mod = self._load()
        self._fresh_store_path()
        out = mod.forget_point_target("nope")
        self.assertIn("no pointing calibration", out.lower())


# ─── toggle + persistence ───────────────────────────────────────────────────
class ToggleTests(_Base):
    def _patch_settings_writer(self, initial=None):
        from tools import settings_window as sw
        saved = dict(initial or {})
        p1 = mock.patch.object(sw, "load_settings", lambda *a, **k: dict(saved))
        p2 = mock.patch.object(sw, "save_settings",
                               lambda d, *a, **k: saved.update(d))
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)
        return saved

    def test_on_persists_flag(self):
        mod = self._load()
        self._patch_flag(False)
        saved = self._patch_settings_writer()
        self._inject("audio.kinect_bridge", _fake_bridge(enabled=True))
        out = mod.point_control_on("")
        self.assertIn("on", out.lower())
        self.assertTrue(saved.get("KINECT_POINT_CONTROL_ENABLED"))
        from core import config as cfg
        self.assertTrue(cfg.KINECT_POINT_CONTROL_ENABLED)

    def test_off_persists_flag(self):
        mod = self._load()
        self._patch_flag(True)
        saved = self._patch_settings_writer(
            {"KINECT_POINT_CONTROL_ENABLED": True})
        out = mod.point_control_off("")
        self.assertIn("off", out.lower())
        self.assertFalse(saved.get("KINECT_POINT_CONTROL_ENABLED"))

    def test_on_warns_when_sensor_off(self):
        mod = self._load()
        self._patch_flag(False)
        self._patch_settings_writer()
        self._inject("audio.kinect_bridge", _fake_bridge(enabled=False))
        out = mod.point_control_on("")
        self.assertIn("off", out.lower())   # mentions the sensor is still off


# ─── status ─────────────────────────────────────────────────────────────────
class StatusTests(_Base):
    def test_status_off(self):
        mod = self._load()
        self._patch_flag(False)
        self._fresh_store_path()
        self.assertIn("off", mod.point_status("").lower())

    def test_status_on_nothing_calibrated(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._fresh_store_path()
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        out = mod.point_status("")
        self.assertIn("nothing", out.lower())

    def test_status_on_with_targets_and_pointing(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        path = self._fresh_store_path()
        self._calibrate(path, "lamp", (1.0, 0.0, 0.0))
        self._inject("audio.kinect_bridge",
                     _fake_bridge(bodies=[_arm_body((1.0, 0.0, 0.0))]))
        out = mod.point_status("")
        self.assertIn("on", out.lower())
        self.assertIn("calibrated", out.lower())

    def test_status_on_sensor_off(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._fresh_store_path()
        self._inject("audio.kinect_bridge", _fake_bridge(enabled=False))
        out = mod.point_status("")
        self.assertIn("on", out.lower())
        self.assertIn("switched off", out.lower())


# ─── registration ───────────────────────────────────────────────────────────
class RegisterTests(_Base):
    def test_register_exposes_actions(self):
        mod, actions = load_skill_isolated("kinect_pointing", register=True)
        for name in ("point_calibrate", "list_point_targets",
                     "forget_point_target", "point_control",
                     "point_control_on", "point_control_off", "point_status"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))


if __name__ == "__main__":
    unittest.main()
