"""Tests for skills/air_control — the AIR CONTROL live wiring.

Loads the skill through tests/_skill_harness.load_skill_isolated (the same
injection contract as the real loader) with a FAKE audio.kinect_bridge module
and a FAKE pyautogui pinned into sys.modules — so NO sensor, NO real mouse,
NO display. Everything here must pass on headless Linux CI: pyautogui is never
truly imported (the fake shadows it before the skill's lazy _pyautogui() call),
and the Kinect bridge is a hand-rolled types.ModuleType.

Asserts the safety contract the feature ships on:
  * air_control_on / air_control_off / air_control_status are registered.
  * OFF BY DEFAULT: with AIR_CONTROL_ENABLED False (the shipped default),
    register() starts NO loop — the mouse cannot move uninvited at boot.
  * Knob True → register() DOES auto-start the loop (owner opt-in).
  * No Kinect (bridge disabled / unavailable / missing) → air_control_on
    replies gracefully ("Kinect isn't available, sir — …") and starts nothing.
  * Voice on → loop runs even with the knob False (explicit command = consent);
    voice off → loop stops AND pyautogui.mouseUp fires (never a stranded grab).
  * FAILSAFE: an exception inside the loop (get_bodies raising) releases the
    mouse (mouseUp), stops the loop, and records the failsafe reason.
  * Staging refuses to start ("Not while I'm in staging, sir.").

stdlib unittest + mock only; no pytest; no real hardware/network/threads left
running (every test that starts the loop stops and joins it).
"""
from __future__ import annotations

import os
import sys
import time
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── fakes ──────────────────────────────────────────────────────────────────
def _fake_pyautogui():
    """A pyautogui stand-in: the exact attrs the skill touches, all recorded.
    Module-typed so `import pyautogui` resolves to it via sys.modules."""
    m = types.ModuleType("pyautogui")
    m.FAILSAFE = True
    m.PAUSE = 0.1
    m.moveTo = mock.MagicMock(name="moveTo")
    m.mouseDown = mock.MagicMock(name="mouseDown")
    m.mouseUp = mock.MagicMock(name="mouseUp")
    m.click = mock.MagicMock(name="click")
    m.scroll = mock.MagicMock(name="scroll")
    return m


def _fake_bridge(*, enabled=True, available=(True, ""), bodies=None,
                 get_bodies=None):
    """A stand-in audio.kinect_bridge exposing only what the skill reads.
    Pass `get_bodies` to override wholesale (e.g. a raiser for the failsafe
    test); else it returns `bodies` (default [])."""
    m = types.ModuleType("audio.kinect_bridge")
    m.get_enabled = lambda: enabled
    m.available = lambda: available
    if get_bodies is not None:
        m.get_bodies = get_bodies
    else:
        m.get_bodies = lambda: (bodies if bodies is not None else [])
    return m


class AirControlSkillTest(unittest.TestCase):
    """Common plumbing: pin fakes into sys.modules, load the skill fresh,
    and guarantee the loop is stopped + modules restored on teardown."""

    def setUp(self):
        self._saved = {k: sys.modules.get(k)
                       for k in ("pyautogui", "audio.kinect_bridge")}
        self._saved_staging = os.environ.pop("JARVIS_STAGING", None)
        self.pg = _fake_pyautogui()
        sys.modules["pyautogui"] = self.pg
        self.mod = None

    def tearDown(self):
        if self.mod is not None:
            try:
                self.mod._stop_loop()
                t = self.mod._loop_thread
                if t is not None:
                    t.join(timeout=2.0)
            except Exception:
                pass
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        if self._saved_staging is not None:
            os.environ["JARVIS_STAGING"] = self._saved_staging

    def _load(self, bridge, **kw):
        if bridge is None:
            sys.modules.pop("audio.kinect_bridge", None)
        else:
            sys.modules["audio.kinect_bridge"] = bridge
        self.mod, actions = load_skill_isolated("air_control", **kw)
        return self.mod, actions

    def _wait(self, predicate, timeout=2.0):
        """Poll `predicate` up to `timeout` s (the loop ticks at ~30 Hz)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()


class TestRegistration(AirControlSkillTest):
    def test_actions_registered(self):
        _, actions = self._load(_fake_bridge())
        for name in ("air_control_on", "air_control_off", "air_control_status"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))

    def test_off_by_default_no_loop_at_register(self):
        # The shipped default AIR_CONTROL_ENABLED=False: register() must start
        # NOTHING — the safety knob's whole point.
        from core import config
        self.assertFalse(config.AIR_CONTROL_ENABLED)   # pin the shipped default
        mod, actions = self._load(_fake_bridge())
        self.assertFalse(mod._loop_running())
        self.assertIn("off", actions["air_control_status"]())

    def test_knob_true_auto_starts_loop(self):
        from core import config
        with mock.patch.object(config, "AIR_CONTROL_ENABLED", True):
            # neuter_threads=False: the auto-start IS the thing under test.
            mod, _ = self._load(_fake_bridge(), neuter_threads=False)
            self.assertTrue(self._wait(mod._loop_running))


class TestNoKinect(AirControlSkillTest):
    def test_bridge_disabled_graceful_reply(self):
        mod, actions = self._load(_fake_bridge(enabled=False))
        out = actions["air_control_on"]()
        self.assertIn("Kinect isn't available, sir", out)
        self.assertFalse(mod._loop_running())

    def test_sensor_unavailable_graceful_reply(self):
        mod, actions = self._load(
            _fake_bridge(available=(False, "no Kinect sensor detected")))
        out = actions["air_control_on"]()
        self.assertIn("Kinect isn't available, sir", out)
        self.assertIn("no Kinect sensor detected", out)
        self.assertFalse(mod._loop_running())

    def test_bridge_missing_entirely(self):
        # No audio.kinect_bridge in sys.modules AND the real import blocked.
        with mock.patch.dict(sys.modules, {"audio.kinect_bridge": None}):
            mod, actions = self._load(None)
            # sys.modules None entry makes `from audio import kinect_bridge`
            # raise → _bridge() → None → graceful reply.
            mod._bridge = lambda: None
            out = actions["air_control_on"]()
            self.assertIn("Kinect isn't available, sir", out)


class TestOnOff(AirControlSkillTest):
    def test_on_starts_off_stops_and_releases(self):
        mod, actions = self._load(_fake_bridge(bodies=[]))
        out = actions["air_control_on"]()
        self.assertIn("Air control on", out)
        self.assertTrue(self._wait(mod._loop_running))
        # Status while running.
        self.assertIn("Air control is on", actions["air_control_status"]())
        # Off: stops + mouseUp fired (the always-release contract).
        out = actions["air_control_off"]()
        self.assertIn("off", out)
        self.assertTrue(self._wait(lambda: not mod._loop_running()))
        self.pg.mouseUp.assert_called()

    def test_on_works_even_with_knob_false(self):
        # The knob only gates AUTO-start; the explicit voice command is consent.
        from core import config
        self.assertFalse(config.AIR_CONTROL_ENABLED)
        mod, actions = self._load(_fake_bridge())
        actions["air_control_on"]()
        self.assertTrue(self._wait(mod._loop_running))

    def test_off_when_already_off(self):
        _, actions = self._load(_fake_bridge())
        self.assertIn("already off", actions["air_control_off"]())

    def test_staging_refuses_to_start(self):
        os.environ["JARVIS_STAGING"] = "1"
        try:
            mod, actions = self._load(_fake_bridge())
            self.assertIn("staging", actions["air_control_on"]())
            self.assertFalse(mod._loop_running())
        finally:
            os.environ.pop("JARVIS_STAGING", None)


class TestFailsafe(AirControlSkillTest):
    def test_loop_exception_releases_mouse_and_stops(self):
        def _boom():
            raise RuntimeError("sensor exploded")
        mod, actions = self._load(_fake_bridge(get_bodies=_boom))
        actions["air_control_on"]()
        # The very first tick raises → the failsafe must stop the loop and
        # release the button, never leaving a zombie driving the mouse.
        self.assertTrue(self._wait(lambda: not mod._loop_running()))
        self.pg.mouseUp.assert_called()
        self.assertIn("failsafe", mod._last_stop_reason)
        self.assertIn("sensor exploded", mod._last_stop_reason)

    def test_pyautogui_missing_is_survivable(self):
        # Headless CI with no pyautogui at all: _release_mouse/_apply_op must
        # degrade silently rather than raise out of the actions.
        sys.modules.pop("pyautogui", None)
        with mock.patch.dict(sys.modules, {"pyautogui": None}):
            mod, actions = self._load(_fake_bridge())
            self.assertIsNone(mod._pyautogui())
            mod._release_mouse()                      # must not raise
            out = actions["air_control_off"]()        # must not raise either
            self.assertIn("already off", out)


class TestAutoYield(AirControlSkillTest):
    def test_real_input_suppresses_ops_and_releases_drag(self):
        # Owner touches the real mouse mid-run: the loop must stop applying
        # ops (no moveTo while suppressed) and release a held drag.
        mod, actions = self._load(_fake_bridge())
        fake_yield = mock.Mock()
        fake_yield.install.return_value = True
        fake_yield.real_input_recent.return_value = True
        fake_yield.mark_self_action.return_value = None
        with mock.patch.object(mod, "_yield_mod", return_value=fake_yield):
            actions["air_control_on"]()
            self.assertTrue(self._wait(
                lambda: fake_yield.real_input_recent.called))
            # Simulate a drag in progress so release() returns an OP_UP.
            eng = mod._engine
            if eng is not None:
                eng._button_down = True
            self.assertTrue(self._wait(lambda: self.pg.mouseUp.called))
            self.pg.moveTo.assert_not_called()   # never drove the cursor
            actions["air_control_off"]()

    def test_missing_yield_helper_is_old_behavior(self):
        # No _air_mouse_yield module → accessors degrade: no suppression, no
        # raise; the loop runs exactly as before.
        mod, actions = self._load(_fake_bridge())
        with mock.patch.object(mod, "_yield_mod", return_value=None):
            self.assertFalse(mod._real_input_recent())
            mod._install_yield_watcher()          # must not raise
            mod._mark_self_action()               # must not raise
            actions["air_control_on"]()
            self.assertTrue(self._wait(lambda: mod._loop_running()))
            actions["air_control_off"]()

    def test_self_actions_marked_after_real_op(self):
        # When the engine emits a real op, the loop must tell the watcher it
        # was self-injected (so the polling fallback can't mistake it for the
        # owner's hand on the mouse).
        from core.air_control import AirOp, OP_MOVE
        mod, actions = self._load(_fake_bridge())
        fake_yield = mock.Mock()
        fake_yield.install.return_value = True
        fake_yield.real_input_recent.return_value = False
        with mock.patch.object(mod, "_yield_mod", return_value=fake_yield):
            actions["air_control_on"]()
            self.assertTrue(self._wait(lambda: mod._loop_running()))
            with mock.patch.object(
                    mod._engine, "update",
                    return_value=AirOp(OP_MOVE, x=10, y=10, engaged=True)):
                self.assertTrue(self._wait(
                    lambda: fake_yield.mark_self_action.called))
            actions["air_control_off"]()


if __name__ == "__main__":
    unittest.main()
