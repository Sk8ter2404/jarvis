"""Tests for skills/kinect_gestures — gesture→action mapping, the live gate,
and the toggle/persistence actions.

Loads the skill in isolation (no monolith boot) via the shared harness, with a
fake kinect_bridge + a fake bobert_companion ('bc') injected into sys.modules.
No real sensor, no real recognizer poll thread. Asserts:

  * WAVE wakes ONLY when dormant; RAISE_HAND confirms ONLY when a pending
    confirmation exists; SWIPE interrupts speech + clears the pending queue,
  * _poll_once no-ops when KINECT_GESTURES_ENABLED is False, when staging, and
    when the bridge is absent/disabled,
  * gestures_on persists KINECT_GESTURES_ENABLED via the reused settings writer
    (mocked), and gesture_status reflects enabled state + body-in-view.

stdlib unittest + mock.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

from audio import kinect_gestures as kg


# ─── settings-file safety net ───────────────────────────────────────────────
# The toggle tests below patch tools.settings_window.load_settings/save_settings
# at the source module so persistence is captured in-memory and the real
# data/user_settings.json is never written. This module-level redirect is a
# belt-and-suspenders second layer: it points JARVIS_SETTINGS_PATH at a
# throwaway file for the whole module, so even if a future test forgot to mock
# the writer, the real settings file STILL can't be clobbered. Restored on exit.
_SAVED_SETTINGS_ENV: "str | None" = None
_SETTINGS_TMPDIR: "str | None" = None


def setUpModule() -> None:
    global _SAVED_SETTINGS_ENV, _SETTINGS_TMPDIR
    _SAVED_SETTINGS_ENV = os.environ.get("JARVIS_SETTINGS_PATH")
    _SETTINGS_TMPDIR = tempfile.mkdtemp(prefix="jarvis_kinect_test_")
    os.environ["JARVIS_SETTINGS_PATH"] = os.path.join(
        _SETTINGS_TMPDIR, "test_user_settings.json")


def tearDownModule() -> None:
    if _SAVED_SETTINGS_ENV is None:
        os.environ.pop("JARVIS_SETTINGS_PATH", None)
    else:
        os.environ["JARVIS_SETTINGS_PATH"] = _SAVED_SETTINGS_ENV


# ─── fakes ──────────────────────────────────────────────────────────────────
def _fake_bridge(*, enabled=True, available=(True, ""), bodies=None,
                 presence=None):
    m = types.ModuleType("audio.kinect_bridge")
    m.get_enabled = lambda: enabled
    m.available = lambda: available
    m.get_bodies = lambda: (bodies if bodies is not None else [])
    m.get_presence = lambda: (presence if presence is not None
                              else {"present": False, "count": 0,
                                    "nearest_m": None, "facing": None, "ts": 0.0})
    return m


def _fake_bc(*, standby=False, sleep=False, pending=None, speaking=False):
    bc = types.ModuleType("bobert_companion")
    bc._standby_mode = [bool(standby)]
    bc._sleep_mode = [bool(sleep)]
    bc._standby_auto_engage_lock = threading.Lock()
    bc._pending_confirmation = list(pending or [])
    bc._tts_playback_active = [bool(speaking)]
    bc._barge_in_interrupted = False

    # Mirrors the real request_tts_interrupt: refuses when nothing is playing;
    # a non-acoustic source (a hand in front of the Kinect) skips the mic-facing
    # echo / wake-word gates, which exist only because the mic hears the
    # speakers. Records what it accepted. 2026-07-14 audit.
    bc._tts_interrupts = []

    def _request_tts_interrupt(source="wake-word", acoustic=True):
        if not bc._tts_playback_active[0]:
            return False
        bc._tts_interrupts.append((source, acoustic))
        return True

    bc.request_tts_interrupt = _request_tts_interrupt
    bc._hud_writes = []
    bc._write_hud_state = lambda **kw: bc._hud_writes.append(kw)
    bc._spoken = []
    bc._speak = lambda *a, **k: bc._spoken.append(a[0] if a else "")
    bc._announced = []
    bc.proactive_announce = lambda msg, source="skill", **k: (
        bc._announced.append((source, msg)) or True)

    # A handle_confirmation_response that mimics the real one: drains the queue,
    # runs each action's fn, returns True. Records what it executed.
    bc._executed = []

    def _handle(text):
        if not bc._pending_confirmation:
            return False
        t = (text or "").strip().lower()
        if any(t.startswith(w) for w in ("yes", "confirm", "do it")):
            while bc._pending_confirmation:
                name, arg = bc._pending_confirmation.pop(0)
                bc._executed.append((name, arg))
        else:
            bc._pending_confirmation.clear()
        return True
    bc.handle_confirmation_response = _handle
    return bc


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
        # register=False so the background poll thread isn't even constructed.
        mod, _actions = load_skill_isolated("kinect_gestures", register=False)
        return mod

    def _patch_flag(self, value):
        """Set KINECT_GESTURES_ENABLED on the live core.config (the skill reads
        it via `from core import config`)."""
        from core import config as cfg
        p = mock.patch.object(cfg, "KINECT_GESTURES_ENABLED", value, create=True)
        p.start()
        self.addCleanup(p.stop)

    def _not_staging(self, mod):
        # Force the skill's staging gate to False regardless of env.
        p = mock.patch.object(mod, "_is_staging", lambda: False)
        p.start()
        self.addCleanup(p.stop)


# ─── WAVE → wake only when dormant ──────────────────────────────────────────
class WaveMappingTests(_Base):
    def test_wave_wakes_when_dormant(self):
        mod = self._load()
        bc = _fake_bc(standby=True, sleep=True)
        mod._do_wave(bc)
        self.assertFalse(bc._standby_mode[0])
        self.assertFalse(bc._sleep_mode[0])
        self.assertTrue(any("waved" in s.lower() for s in bc._spoken))

    def test_wave_noop_when_awake(self):
        mod = self._load()
        bc = _fake_bc(standby=False, sleep=False)
        mod._do_wave(bc)
        # No speech, flags untouched.
        self.assertEqual(bc._spoken, [])
        self.assertFalse(bc._standby_mode[0])


# ─── RAISE_HAND → confirm only when pending ────────────────────────────────
class RaiseHandMappingTests(_Base):
    def test_confirms_pending(self):
        mod = self._load()
        bc = _fake_bc(pending=[("reset_memory", "")])
        mod._do_raise_hand(bc)
        self.assertEqual(bc._executed, [("reset_memory", "")])
        self.assertEqual(bc._pending_confirmation, [])

    def test_noop_when_nothing_pending(self):
        mod = self._load()
        bc = _fake_bc(pending=[])
        mod._do_raise_hand(bc)
        self.assertEqual(bc._executed, [])


# ─── SWIPE → stop speech + clear pending ───────────────────────────────────
class SwipeMappingTests(_Base):
    def test_swipe_interrupts_speech(self):
        """Asserted `_barge_in_interrupted` until the 2026-07-14 audit — which
        was green-by-mock: that flag's only speaker-stopping reader is the
        `_barge_watch` thread, and play_with_lipsync only starts that thread
        when a barge-in mic stream exists (BARGE_IN_ENABLED *and* a headset).
        On speakers nobody read it, so the gesture did nothing while this test
        stayed green. Pin the mechanism that actually stops the audio."""
        mod = self._load()
        bc = _fake_bc(speaking=True)
        mod._do_swipe(bc)
        self.assertEqual(bc._tts_interrupts, [("kinect-swipe", False)])

    def test_swipe_clears_pending(self):
        mod = self._load()
        bc = _fake_bc(pending=[("shutdown_pc", "")])
        mod._do_swipe(bc)
        self.assertEqual(bc._pending_confirmation, [])
        self.assertTrue(any("never mind" in s.lower() for s in bc._spoken))

    def test_swipe_noop_when_idle(self):
        mod = self._load()
        bc = _fake_bc(speaking=False, pending=[])
        # Should not raise, should not interrupt (nothing is playing).
        mod._do_swipe(bc)
        self.assertEqual(bc._tts_interrupts, [])


# ─── _dispatch routes each gesture to its handler ──────────────────────────
class DispatchTests(_Base):
    def test_dispatch_wave(self):
        mod = self._load()
        bc = _fake_bc(standby=True, sleep=True)
        mod._dispatch(bc, kg.WAVE)
        self.assertFalse(bc._standby_mode[0])

    def test_dispatch_raise_hand(self):
        mod = self._load()
        bc = _fake_bc(pending=[("x", "y")])
        mod._dispatch(bc, kg.RAISE_HAND)
        self.assertEqual(bc._executed, [("x", "y")])

    def test_dispatch_swipe_left_and_right_both_cancel(self):
        for g in (kg.SWIPE_LEFT, kg.SWIPE_RIGHT):
            mod = self._load()
            bc = _fake_bc(pending=[("x", "y")])
            mod._dispatch(bc, g)
            self.assertEqual(bc._pending_confirmation, [], g)


# ─── _poll_once gating ──────────────────────────────────────────────────────
class _StubRec:
    """Recognizer stub: returns the queued gesture once per update call, and
    records update()/reset() calls so the air-mouse-suppression gate (which
    skips update() and calls reset() while engaged) can be asserted."""
    def __init__(self, gestures):
        self._g = list(gestures)
        self.updates = 0
        self.resets = 0

    def update(self, bodies):
        self.updates += 1
        return self._g.pop(0) if self._g else None

    def reset(self):
        self.resets += 1


class _FakeAirMouseSkill(types.ModuleType):
    """A stand-in skills/kinect_air_mouse exposing get_air_mouse_state (the engaged
    flag) AND two_hand_active() (FILTER 5 — two-hand pinch-resize heartbeat), so the
    gesture skill's _air_mouse_engaged() / _two_hand_active() can read them. Injected
    as sys.modules['skill_kinect_air_mouse'] (the loader's module name)."""
    def __init__(self, engaged=False, two_hand=False):
        super().__init__("skill_kinect_air_mouse")
        self._engaged = bool(engaged)
        self._two_hand = bool(two_hand)
        self.get_air_mouse_state = lambda: {
            "engaged": self._engaged, "hand": "right", "grip": "open", "ts": 0.0}
        self.two_hand_active = lambda: self._two_hand


# ══════════════════════════════════════════════════════════════════════════
#  AIR-MOUSE SUPPRESSION: gestures fire ONLY in the RELAXED state (no arm
#  extended). While the air-mouse is ENGAGED (an arm reaching out) detection is
#  SUPPRESSED — wave/swipe must not register, and the recognizer is RESET so the
#  reach motion can't fire a gesture the instant the arm relaxes.
# ══════════════════════════════════════════════════════════════════════════
class AirMouseSuppressionTests(_Base):
    def _set_air_mouse(self, engaged):
        self._inject("skill_kinect_air_mouse", _FakeAirMouseSkill(engaged=engaged))

    def test_engaged_helper_reads_air_mouse_state(self):
        mod = self._load()
        self._set_air_mouse(True)
        self.assertTrue(mod._air_mouse_engaged())
        self._set_air_mouse(False)
        self.assertFalse(mod._air_mouse_engaged())

    def test_engaged_helper_false_when_skill_absent(self):
        mod = self._load()
        # No air-mouse skill loaded → gestures still work (helper returns False).
        self._inject("skill_kinect_air_mouse", None)
        self.assertFalse(mod._air_mouse_engaged())

    def test_gesture_suppressed_while_air_mouse_engaged(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._set_air_mouse(True)               # arm extended → air-mouse engaged
        bc = _fake_bc(standby=True, sleep=True)
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[{"x": 1}]))
        rec = _StubRec([kg.WAVE])
        got = mod._poll_once(rec, bc)
        self.assertIsNone(got)                  # gesture SUPPRESSED
        self.assertEqual(rec.updates, 0)        # detection skipped entirely
        self.assertGreaterEqual(rec.resets, 1)  # recognizer reset while engaged
        self.assertTrue(bc._standby_mode[0])    # NOT woken

    def test_gesture_fires_when_relaxed(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._set_air_mouse(False)              # no arm extended → relaxed
        bc = _fake_bc(standby=True, sleep=True)
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[{"x": 1}]))
        rec = _StubRec([kg.WAVE])
        got = mod._poll_once(rec, bc)
        self.assertEqual(got, kg.WAVE)          # fires only when relaxed
        self.assertEqual(rec.updates, 1)
        self.assertFalse(bc._standby_mode[0])   # woke

    def test_engage_then_relax_resumes_detection(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        bc = _fake_bc(standby=True, sleep=True)
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[{"x": 1}]))
        # While engaged: suppressed + reset, queued gesture untouched.
        self._set_air_mouse(True)
        rec = _StubRec([kg.WAVE])
        self.assertIsNone(mod._poll_once(rec, bc))
        self.assertEqual(rec.updates, 0)
        # Relax → the same queued gesture now fires (detection resumed).
        self._set_air_mouse(False)
        self.assertEqual(mod._poll_once(rec, bc), kg.WAVE)
        self.assertTrue(rec.resets >= 1 and rec.updates == 1)


# ══════════════════════════════════════════════════════════════════════════
#  FILTER 5 — TWO-HAND SUPPRESSION: gestures are ALSO suppressed while two-hand
#  pinch-to-resize mode is active (it leaves the air-mouse `engaged` False, so a
#  two-hand SPREAD must not be mis-read as a SWIPE mid-resize).
# ══════════════════════════════════════════════════════════════════════════
class TwoHandSuppressionTests(_Base):
    def _set_air_mouse(self, *, engaged=False, two_hand=False):
        self._inject("skill_kinect_air_mouse",
                     _FakeAirMouseSkill(engaged=engaged, two_hand=two_hand))

    def test_two_hand_active_helper_reads_skill(self):
        mod = self._load()
        self._set_air_mouse(two_hand=True)
        self.assertTrue(mod._two_hand_active())
        self._set_air_mouse(two_hand=False)
        self.assertFalse(mod._two_hand_active())

    def test_two_hand_active_false_when_skill_absent(self):
        mod = self._load()
        self._inject("skill_kinect_air_mouse", None)
        self.assertFalse(mod._two_hand_active())

    def test_gesture_suppressed_while_two_hand_active(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        # air-mouse NOT engaged, but TWO-HAND mode IS active.
        self._set_air_mouse(engaged=False, two_hand=True)
        bc = _fake_bc(standby=True, sleep=True)
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[{"x": 1}]))
        rec = _StubRec([kg.SWIPE_LEFT])
        got = mod._poll_once(rec, bc)
        self.assertIsNone(got)                  # gesture SUPPRESSED
        self.assertEqual(rec.updates, 0)        # detection skipped entirely
        self.assertGreaterEqual(rec.resets, 1)  # recognizer reset while resizing
        self.assertTrue(bc._standby_mode[0])    # NOT woken / no side effect

    def test_gesture_fires_once_two_hand_releases(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        bc = _fake_bc(standby=True, sleep=True)
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[{"x": 1}]))
        # While two-hand active: suppressed.
        self._set_air_mouse(two_hand=True)
        rec = _StubRec([kg.WAVE])
        self.assertIsNone(mod._poll_once(rec, bc))
        self.assertEqual(rec.updates, 0)
        # Two-hand releases → the same queued gesture fires.
        self._set_air_mouse(two_hand=False)
        self.assertEqual(mod._poll_once(rec, bc), kg.WAVE)
        self.assertTrue(rec.resets >= 1 and rec.updates == 1)


class PollGateTests(_Base):
    def test_poll_dispatches_when_enabled(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        bc = _fake_bc(standby=True, sleep=True)
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[{"x": 1}]))
        rec = _StubRec([kg.WAVE])
        got = mod._poll_once(rec, bc)
        self.assertEqual(got, kg.WAVE)
        self.assertFalse(bc._standby_mode[0])   # woke

    def test_poll_recognizes_but_skips_dispatch_when_flag_off(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(False)
        bc = _fake_bc(standby=True, sleep=True)
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[{"x": 1}]))
        rec = _StubRec([kg.WAVE])
        got = mod._poll_once(rec, bc)
        self.assertIsNone(got)                  # gated → no gesture returned
        self.assertEqual(rec.updates, 1)        # recognizer STILL fed
        self.assertTrue(bc._standby_mode[0])    # NOT woken

    def test_poll_noop_when_staging(self):
        mod = self._load()
        # Force staging True.
        p = mock.patch.object(mod, "_is_staging", lambda: True)
        p.start()
        self.addCleanup(p.stop)
        self._patch_flag(True)
        bc = _fake_bc(standby=True, sleep=True)
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[{"x": 1}]))
        rec = _StubRec([kg.WAVE])
        got = mod._poll_once(rec, bc)
        self.assertIsNone(got)
        self.assertTrue(bc._standby_mode[0])    # NOT woken in staging

    def test_poll_noop_when_bridge_absent(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._inject("audio.kinect_bridge", None)
        bc = _fake_bc()
        rec = _StubRec([kg.WAVE])
        self.assertIsNone(mod._poll_once(rec, bc))

    def test_poll_noop_when_sensor_disabled(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._inject("audio.kinect_bridge", _fake_bridge(enabled=False))
        bc = _fake_bc(standby=True, sleep=True)
        rec = _StubRec([kg.WAVE])
        self.assertIsNone(mod._poll_once(rec, bc))
        self.assertTrue(bc._standby_mode[0])

    def test_poll_noop_when_sensor_unavailable(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        self._inject("audio.kinect_bridge",
                     _fake_bridge(available=(False, "no sensor")))
        bc = _fake_bc(standby=True, sleep=True)
        rec = _StubRec([kg.WAVE])
        self.assertIsNone(mod._poll_once(rec, bc))


# ─── toggle + persistence ───────────────────────────────────────────────────
class ToggleTests(_Base):
    def _patch_settings_writer(self, initial=None):
        """Patch the REAL tools.settings_window.load_settings/save_settings (the
        exact functions the skill's `from tools import settings_window` resolves
        to) so persistence is captured in-memory. Patching the live module —
        rather than injecting a fake into sys.modules — is robust to import
        ordering: under the CI sim the real module may already be bound as the
        `tools` package attribute, which `from tools import settings_window`
        would prefer over a sys.modules swap. Returns the captured dict.

        settings_window is stdlib-only (its docstring guarantees no display /
        network on import), so importing it here is safe on the reduced runner.
        """
        from tools import settings_window as sw
        saved = dict(initial or {})
        p1 = mock.patch.object(sw, "load_settings", lambda *a, **k: dict(saved))
        p2 = mock.patch.object(sw, "save_settings",
                               lambda d, *a, **k: saved.update(d))
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)
        return saved

    def test_gestures_on_persists_flag(self):
        mod = self._load()
        self._patch_flag(False)
        saved = self._patch_settings_writer()
        self._inject("audio.kinect_bridge", _fake_bridge(enabled=True))
        out = mod.gestures_on("")
        self.assertIn("on", out.lower())
        self.assertTrue(saved.get("KINECT_GESTURES_ENABLED"))
        # Live config flag flipped too.
        from core import config as cfg
        self.assertTrue(cfg.KINECT_GESTURES_ENABLED)

    def test_gestures_off_persists_flag(self):
        mod = self._load()
        self._patch_flag(True)
        saved = self._patch_settings_writer({"KINECT_GESTURES_ENABLED": True})
        out = mod.gestures_off("")
        self.assertIn("off", out.lower())
        self.assertFalse(saved.get("KINECT_GESTURES_ENABLED"))

    def test_gestures_on_warns_when_sensor_off(self):
        mod = self._load()
        self._patch_flag(False)
        self._patch_settings_writer()
        self._inject("audio.kinect_bridge", _fake_bridge(enabled=False))
        out = mod.gestures_on("")
        self.assertIn("kinect", out.lower())   # mentions the sensor is still off


# ─── gesture_status ─────────────────────────────────────────────────────────
class StatusTests(_Base):
    def test_status_off(self):
        mod = self._load()
        self._patch_flag(False)
        self.assertIn("off", mod.gesture_status("").lower())

    def test_status_on_with_body_in_view(self):
        mod = self._load()
        self._patch_flag(True)
        self._inject("audio.kinect_bridge",
                     _fake_bridge(presence={"present": True, "count": 1}))
        out = mod.gesture_status("")
        self.assertIn("on", out.lower())
        self.assertIn("see you", out.lower())

    def test_status_on_but_sensor_off(self):
        mod = self._load()
        self._patch_flag(True)
        self._inject("audio.kinect_bridge", _fake_bridge(enabled=False))
        out = mod.gesture_status("")
        self.assertIn("on", out.lower())
        self.assertIn("unavailable", out.lower())


# ══════════════════════════════════════════════════════════════════════════
#  B3 — the gesture-pop badge: a fired gesture is published with a label + a
#  timestamp, and the badge opacity FADES from 1.0 → 0.0 over ~1 s, then is
#  gone. The pure label + fade helpers are what the HUD preview renders.
# ══════════════════════════════════════════════════════════════════════════
class GestureLabelTests(_Base):
    def test_known_gestures_get_friendly_labels(self):
        mod = self._load()
        self.assertEqual(mod.gesture_label_for(kg.WAVE), "WAVE")
        self.assertEqual(mod.gesture_label_for(kg.RAISE_HAND), "RAISE HAND")
        # Both swipe directions read "SWIPE" (direction is irrelevant).
        self.assertEqual(mod.gesture_label_for(kg.SWIPE_LEFT), "SWIPE")
        self.assertEqual(mod.gesture_label_for(kg.SWIPE_RIGHT), "SWIPE")

    def test_unknown_gesture_falls_back_to_upper(self):
        mod = self._load()
        self.assertEqual(mod.gesture_label_for("double_tap"), "DOUBLE TAP")
        self.assertEqual(mod.gesture_label_for(""), "")


class GesturePopAlphaTests(_Base):
    def test_alpha_is_full_at_fire_then_fades_to_zero(self):
        mod = self._load()
        ttl = mod.GESTURE_POP_TTL_SECONDS
        fired = 100.0
        # Full opacity the instant it fires …
        self.assertEqual(mod.gesture_pop_alpha(fired, fired, ttl), 1.0)
        # … about half-faded at the midpoint …
        self.assertAlmostEqual(
            mod.gesture_pop_alpha(fired + ttl / 2.0, fired, ttl), 0.5, delta=0.05)
        # … fully gone once the TTL has elapsed …
        self.assertEqual(mod.gesture_pop_alpha(fired + ttl, fired, ttl), 0.0)
        self.assertEqual(mod.gesture_pop_alpha(fired + ttl + 1.0, fired, ttl), 0.0)

    def test_alpha_monotonically_decreases(self):
        mod = self._load()
        fired = 50.0           # positive (ts<=0 is the 'nothing fired' sentinel)
        ttl = mod.GESTURE_POP_TTL_SECONDS
        seq = [mod.gesture_pop_alpha(fired + t, fired, ttl)
               for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        self.assertEqual(seq, sorted(seq, reverse=True))   # never increases
        self.assertEqual(seq[0], 1.0)
        self.assertEqual(seq[-1], 0.0)

    def test_no_fire_yet_is_zero(self):
        mod = self._load()
        # ts <= 0 (nothing fired) → draw nothing.
        self.assertEqual(mod.gesture_pop_alpha(123.0, 0.0), 0.0)
        self.assertEqual(mod.gesture_pop_alpha(123.0, -5.0), 0.0)


class GesturePopStateTests(_Base):
    def test_default_state_is_empty(self):
        mod = self._load()
        st = mod.get_last_gesture()
        self.assertIsNone(st["gesture"])
        self.assertEqual(st["ts"], 0.0)

    def test_set_last_gesture_publishes_label_and_ts(self):
        mod = self._load()
        clock = [42.0]
        mod._set_last_gesture(kg.WAVE, now_fn=lambda: clock[0])
        st = mod.get_last_gesture()
        self.assertEqual(st["gesture"], kg.WAVE)
        self.assertEqual(st["label"], "WAVE")
        self.assertEqual(st["ts"], 42.0)
        # Copy semantics: mutating the snapshot can't corrupt shared state.
        st["label"] = "TAMPERED"
        self.assertEqual(mod.get_last_gesture()["label"], "WAVE")

    def test_poll_publishes_fired_gesture_for_pop(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        bc = _fake_bc(standby=True, sleep=True)
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[{"x": 1}]))
        rec = _StubRec([kg.SWIPE_LEFT])
        got = mod._poll_once(rec, bc)
        self.assertEqual(got, kg.SWIPE_LEFT)
        # The pop badge state now reflects the fired gesture, ready for the HUD.
        st = mod.get_last_gesture()
        self.assertEqual(st["gesture"], kg.SWIPE_LEFT)
        self.assertEqual(st["label"], "SWIPE")
        self.assertGreater(st["ts"], 0.0)
        # And it's freshly visible (alpha > 0 right after firing).
        self.assertGreater(mod.gesture_pop_alpha(st["ts"], st["ts"]), 0.0)

    def test_poll_does_not_publish_when_gated_off(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(False)              # gestures OFF
        bc = _fake_bc(standby=True, sleep=True)
        self._inject("audio.kinect_bridge", _fake_bridge(bodies=[{"x": 1}]))
        rec = _StubRec([kg.WAVE])
        mod._poll_once(rec, bc)
        # No badge should pop for a gesture that didn't dispatch.
        self.assertIsNone(mod.get_last_gesture()["gesture"])


if __name__ == "__main__":
    unittest.main()
