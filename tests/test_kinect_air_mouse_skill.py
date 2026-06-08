"""Tests for skills/kinect_air_mouse — the Kinect air-mouse.

Drives the PURE CORE (reach-box mapping, EMA smoothing, grip debounce, the
raise-to-engage dead-man gate, the open→move / closed→LEFT-button / re-open→
release state machine, the overlay colour mapping) directly, and the LIVE
_poll_once path with a fake kinect_bridge + the real mouse actuation mocked out —
so NO sensor, NO real cursor, NO Qt is touched. Asserts the owner-facing
contract:

  * the reach-box maps the hand NON-mirrored (hand RIGHT → larger cursor_x) and
    across the ENTIRE virtual desktop — every monitor, including one arranged
    LEFT of the primary (negative virtual-screen origin) — not just primary,
  * ENGAGE GATE (the headline dead-man fix): the cursor is driven ONLY while the
    hand is RAISED above the body reference AND tracked. A LOWERED hand, a hand
    with no reference, or a body/hand untracked beyond the ~0.3 s grace
    DISENGAGES — cursor=None (so the live loop calls NO SetCursorPos and the
    PHYSICAL mouse is free), any held button RELEASED, overlay hidden. Raise/drop
    hysteresis stops threshold flicker; raising the hand re-engages.
  * raised OPEN hand     → the cursor MOVES (no button), overlay "track"/cyan,
  * OPEN → CLOSED        → the LEFT button goes DOWN once, overlay "grab"/gold,
  * CLOSED held + move   → cursor still moves (a left-DRAG), button stays down,
  * CLOSED → OPEN        → the LEFT button goes UP once (close→open = a click),
  * the button is ALWAYS the LEFT/primary one, NEVER the right,
  * a single flickered CLOSED frame is DEBOUNCED (no stray button),
  * dead-man: hand untracked while held → button RELEASED, overlay hidden,
  * _poll_once no-ops the SIDE EFFECTS when KINECT_AIR_MOUSE_ENABLED is False,
    when staging, and when the bridge is absent/disabled,
  * air_mouse_on persists KINECT_AIR_MOUSE_ENABLED via the reused settings
    writer (mocked), and air_mouse_status reflects enabled + hand-in-view.

stdlib unittest + mock; App-Control-safe; CI-sim clean (no win32/pyautogui/Qt).
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated


# ─── settings-file safety net (belt-and-suspenders; see kinect_gestures test) ─
_SAVED_SETTINGS_ENV: "str | None" = None
_SETTINGS_TMPDIR: "str | None" = None


def setUpModule() -> None:
    global _SAVED_SETTINGS_ENV, _SETTINGS_TMPDIR
    _SAVED_SETTINGS_ENV = os.environ.get("JARVIS_SETTINGS_PATH")
    _SETTINGS_TMPDIR = tempfile.mkdtemp(prefix="jarvis_airmouse_test_")
    os.environ["JARVIS_SETTINGS_PATH"] = os.path.join(
        _SETTINGS_TMPDIR, "test_user_settings.json")


def tearDownModule() -> None:
    if _SAVED_SETTINGS_ENV is None:
        os.environ.pop("JARVIS_SETTINGS_PATH", None)
    else:
        os.environ["JARVIS_SETTINGS_PATH"] = _SAVED_SETTINGS_ENV


# ─── fakes ──────────────────────────────────────────────────────────────────
def _fake_bridge(*, enabled=True, available=(True, ""), bodies=None,
                 hand_states=None):
    """A stand-in audio.kinect_bridge exposing only what the skill reads."""
    m = types.ModuleType("audio.kinect_bridge")
    m.get_enabled = lambda: enabled
    m.available = lambda: available
    m.get_bodies = lambda: (bodies if bodies is not None else [])
    m.get_hand_states = lambda: (hand_states if hand_states is not None else
                                 {"right": "unknown", "left": "unknown",
                                  "tracked": False, "ts": 0.0})
    return m


def _body(*, hand_x=0.0, hand_y=0.30, grip_right="open", grip_left="unknown",
          distance=1.8, side="right", ref_y=0.0, with_ref=True):
    """A get_bodies()-shaped body dict with a hand joint + grip on `side`, plus
    the ENGAGE-reference joint (spine_mid) at `ref_y`. The default hand_y=0.30 is
    well above the default ref_y=0.0 + the ~0.08 m engage threshold, so the hand
    reads as RAISED (engaged) unless a test lowers hand_y toward/under ref_y.
    with_ref=False omits ALL reference joints (spine + shoulders + elbow) so the
    sample carries ref_y=None — the 'no reference → disengage' case."""
    joints = {
        f"hand_{side}": (hand_x, hand_y, distance, 2),
    }
    if with_ref:
        # spine_mid is the primary engage reference (_ENGAGE_REF_JOINTS[0]).
        joints["spine_mid"] = (0.0, ref_y, distance, 2)
        joints["spine_shoulder"] = (0.0, ref_y + 0.15, distance, 2)
        joints["head"] = (0.0, ref_y + 0.6, distance, 2)
        # An elbow below the hand, so the elbow fall-back also reads "raised".
        joints[f"elbow_{side}"] = (hand_x, ref_y - 0.1, distance, 2)
    return {
        "id": 0, "joints": joints,
        "head": (0.0, ref_y + 0.6, distance), "distance_m": distance,
        "facing": True,
        "hand_right": grip_right, "hand_left": grip_left,
    }


class _Base(unittest.TestCase):
    def _load(self):
        # register=False so the background poll thread isn't constructed.
        mod, _actions = load_skill_isolated("kinect_air_mouse", register=False)
        return mod

    def _patch_flag(self, value):
        from core import config as cfg
        p = mock.patch.object(cfg, "KINECT_AIR_MOUSE_ENABLED", value, create=True)
        p.start()
        self.addCleanup(p.stop)

    def _not_staging(self, mod):
        p = mock.patch.object(mod, "_is_staging", lambda: False)
        p.start()
        self.addCleanup(p.stop)

    def _capture_mouse(self, mod):
        """Replace the real mouse actuation with recorders. Returns (moves,
        buttons) lists that fill as the skill acts."""
        moves: list = []
        buttons: list = []
        p1 = mock.patch.object(mod, "_set_cursor_pos",
                               lambda x, y: (moves.append((x, y)) or True))
        p2 = mock.patch.object(mod, "_mouse_button",
                               lambda action: (buttons.append(action) or True))
        # Silence overlay file I/O + spawning entirely.
        p3 = mock.patch.object(mod, "_publish_overlay_state", lambda *a, **k: None)
        p4 = mock.patch.object(mod, "_clear_overlay_state", lambda *a, **k: None)
        p5 = mock.patch.object(mod, "_spawn_overlay", lambda *a, **k: None)
        p6 = mock.patch.object(mod, "_overlay_alive", lambda: True)
        for p in (p1, p2, p3, p4, p5, p6):
            p.start()
            self.addCleanup(p.stop)
        return moves, buttons


# ══════════════════════════════════════════════════════════════════════════
#  PURE CORE
# ══════════════════════════════════════════════════════════════════════════
class ReachBoxTests(_Base):
    def test_center_maps_to_screen_center(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        px, py = rb.map(mod.REACH_CENTER_X, mod.REACH_CENTER_Y)
        # Centre of the box → centre of the screen (±1 px rounding).
        self.assertAlmostEqual(px, (2560 - 1) // 2, delta=2)
        self.assertAlmostEqual(py, (1440 - 1) // 2, delta=2)

    def test_x_not_mirrored_hand_right_is_cursor_right(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        # FIX 1: the Kinect image is mirror-flipped relative to the user, so we
        # do NOT negate x — hand to the sensor's RIGHT (+x) must map to the RIGHT
        # of the screen (larger px), and hand LEFT to the LEFT (px 0). Natural,
        # un-mirrored.
        right_px, _ = rb.map(mod.REACH_CENTER_X + mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        left_px, _ = rb.map(mod.REACH_CENTER_X - mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        self.assertGreater(right_px, left_px)       # hand-right → larger cursor_x
        self.assertEqual(left_px, 0)
        self.assertEqual(right_px, 2560 - 1)

    def test_x_monotonic_left_to_right(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        # Sweeping the hand left→right sweeps the cursor left→right (monotone up).
        xs = [rb.map(mod.REACH_CENTER_X + f * mod.REACH_HALF_W,
                     mod.REACH_CENTER_Y)[0]
              for f in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        self.assertEqual(xs, sorted(xs))
        self.assertLess(xs[0], xs[-1])

    def test_y_inverted_camera_up_is_screen_top(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        # Y axis UNCHANGED by the fix: hand UP (+y camera) → TOP of screen (small).
        _, up_py = rb.map(mod.REACH_CENTER_X, mod.REACH_CENTER_Y + mod.REACH_HALF_H)
        _, down_py = rb.map(mod.REACH_CENTER_X, mod.REACH_CENTER_Y - mod.REACH_HALF_H)
        self.assertEqual(up_py, 0)
        self.assertEqual(down_py, 1440 - 1)

    def test_overshoot_is_clamped(self):
        mod = self._load()
        rb = mod.ReachBox(2560, 1440)
        px, py = rb.map(mod.REACH_CENTER_X + 99.0, mod.REACH_CENTER_Y - 99.0)
        self.assertTrue(0 <= px <= 2559)
        self.assertTrue(0 <= py <= 1439)


class VirtualDesktopMappingTests(_Base):
    """FIX 2: the hand maps across the ENTIRE virtual desktop (all monitors), not
    just the primary — including a monitor LEFT of the primary, which has a
    NEGATIVE virtual-screen origin. Modelled on the owner's 4-monitor rig: a
    2560-wide primary at x=0 with another 2560-wide monitor to its LEFT
    (origin_x = -2560) and one above (origin_y = -1440), giving a virtual desktop
    of 7680×2880 anchored at (-2560, -1440)."""

    # (origin_x, origin_y, width, height) — left + above the primary, so the
    # origin is negative on both axes (the case a primary-only map can't reach).
    VX, VY, VW, VH = -2560, -1440, 7680, 2880

    def _rb(self, mod):
        return mod.ReachBox(self.VW, self.VH, origin_x=self.VX, origin_y=self.VY)

    def test_center_maps_to_virtual_desktop_center(self):
        mod = self._load()
        rb = self._rb(mod)
        px, py = rb.map(mod.REACH_CENTER_X, mod.REACH_CENTER_Y)
        self.assertAlmostEqual(px, self.VX + (self.VW - 1) // 2, delta=2)
        self.assertAlmostEqual(py, self.VY + (self.VH - 1) // 2, delta=2)

    def test_left_edge_reaches_negative_origin_monitor(self):
        mod = self._load()
        rb = self._rb(mod)
        # Hand fully LEFT → the LEFT edge of the desktop, which is NEGATIVE
        # (the monitor left of primary) — unreachable with a primary-only map.
        left_px, _ = rb.map(mod.REACH_CENTER_X - mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        self.assertEqual(left_px, self.VX)          # == -2560
        self.assertLess(left_px, 0)

    def test_right_edge_reaches_far_monitor(self):
        mod = self._load()
        rb = self._rb(mod)
        # Hand fully RIGHT → the RIGHT edge of the whole desktop (far monitor).
        right_px, _ = rb.map(mod.REACH_CENTER_X + mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        self.assertEqual(right_px, self.VX + self.VW - 1)   # == 5119

    def test_top_edge_reaches_negative_y_origin(self):
        mod = self._load()
        rb = self._rb(mod)
        # Hand fully UP → the TOP edge of the desktop (monitor above primary),
        # which is NEGATIVE on y.
        _, top_py = rb.map(mod.REACH_CENTER_X, mod.REACH_CENTER_Y + mod.REACH_HALF_H)
        self.assertEqual(top_py, self.VY)           # == -1440
        self.assertLess(top_py, 0)

    def test_un_mirrored_across_virtual_desktop(self):
        mod = self._load()
        rb = self._rb(mod)
        # The headline contract: hand RIGHT → larger cursor_x, hand LEFT →
        # smaller cursor_x, ACROSS the full multi-monitor desktop.
        left_px, _ = rb.map(mod.REACH_CENTER_X - mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        right_px, _ = rb.map(mod.REACH_CENTER_X + mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        self.assertGreater(right_px, left_px)
        self.assertEqual(left_px, self.VX)
        self.assertEqual(right_px, self.VX + self.VW - 1)

    def test_overshoot_clamps_to_virtual_bounds(self):
        mod = self._load()
        rb = self._rb(mod)
        # A wild overshoot parks at the desktop edges, never outside them.
        px_lo, py_lo = rb.map(mod.REACH_CENTER_X - 99.0, mod.REACH_CENTER_Y + 99.0)
        px_hi, py_hi = rb.map(mod.REACH_CENTER_X + 99.0, mod.REACH_CENTER_Y - 99.0)
        self.assertEqual((px_lo, py_lo), (self.VX, self.VY))
        self.assertEqual((px_hi, py_hi),
                         (self.VX + self.VW - 1, self.VY + self.VH - 1))

    def test_reach_box_builder_uses_cached_virtual_bounds(self):
        mod = self._load()
        # _reach_box_for_virtual_desktop() must build a ReachBox spanning the
        # bounds _virtual_screen_bounds() reports (here a negative-origin desk).
        with mock.patch.object(mod, "_virtual_screen_bounds",
                               lambda: (self.VX, self.VY, self.VW, self.VH)):
            rb = mod._reach_box_for_virtual_desktop(refresh=True)
        self.assertEqual((rb.origin_x, rb.origin_y, rb.screen_w, rb.screen_h),
                         (self.VX, self.VY, self.VW, self.VH))
        # And a hand-right maps to a larger cursor_x than hand-left on it.
        left_px, _ = rb.map(mod.REACH_CENTER_X - mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        right_px, _ = rb.map(mod.REACH_CENTER_X + mod.REACH_HALF_W, mod.REACH_CENTER_Y)
        self.assertGreater(right_px, left_px)

    def test_cached_bounds_refresh_picks_up_layout_change(self):
        mod = self._load()
        # First read caches bounds; a later refresh=True picks up a new layout
        # (monitor hot-plugged / rearranged) without a restart.
        seq = [(0, 0, 2560, 1440), (self.VX, self.VY, self.VW, self.VH)]
        calls = {"n": 0}

        def _fake_bounds():
            i = min(calls["n"], len(seq) - 1)
            calls["n"] += 1
            return seq[i]

        # Reset the module-level cache so this test is order-independent.
        mod._VBOUNDS_CACHE[0] = None
        mod._VBOUNDS_CACHE[1] = 0.0
        with mock.patch.object(mod, "_virtual_screen_bounds", _fake_bounds):
            first = mod._cached_virtual_bounds(refresh=True)
            self.assertEqual(first, (0, 0, 2560, 1440))
            # Without refresh + within the interval, the cache holds (no re-read).
            self.assertEqual(mod._cached_virtual_bounds(refresh=False),
                             (0, 0, 2560, 1440))
            # Forced refresh re-reads → the new layout.
            self.assertEqual(mod._cached_virtual_bounds(refresh=True),
                             (self.VX, self.VY, self.VW, self.VH))


class EMATests(_Base):
    def test_first_value_seeds(self):
        mod = self._load()
        e = mod.EMA(0.5)
        self.assertEqual(e.update(10.0), 10.0)   # first sample is the seed

    def test_smooths_toward_target(self):
        mod = self._load()
        e = mod.EMA(0.5)
        e.update(0.0)
        v = e.update(10.0)
        self.assertAlmostEqual(v, 5.0)           # halfway with alpha 0.5

    def test_reset_reseeds(self):
        mod = self._load()
        e = mod.EMA(0.5)
        e.update(0.0); e.update(10.0)
        e.reset()
        self.assertEqual(e.update(100.0), 100.0)


class GripDebouncerTests(_Base):
    def test_requires_n_consecutive_frames(self):
        mod = self._load()
        d = mod.GripDebouncer(frames=3, initial="open")
        self.assertEqual(d.update("closed"), "open")   # 1
        self.assertEqual(d.update("closed"), "open")   # 2
        self.assertEqual(d.update("closed"), "closed")  # 3 → flips

    def test_single_flicker_is_ignored(self):
        mod = self._load()
        d = mod.GripDebouncer(frames=3, initial="open")
        self.assertEqual(d.update("closed"), "open")   # a lone flicker
        self.assertEqual(d.update("open"), "open")     # back to open
        # Streak was broken → stable never moved off "open".
        self.assertEqual(d.stable, "open")

    def test_unknown_holds_current_stable(self):
        mod = self._load()
        d = mod.GripDebouncer(frames=2, initial="closed")
        # An ambiguous frame must not flip a held grip (dead-man releases, not
        # a single 'unknown').
        self.assertEqual(d.update("unknown"), "closed")
        self.assertEqual(d.update("lasso"), "closed")


class OverlayColorTests(_Base):
    def test_track_is_cyan_grab_is_gold(self):
        mod = self._load()
        self.assertEqual(mod.overlay_color_for("track"), "cyan")
        self.assertEqual(mod.overlay_color_for("grab"), "gold")
        self.assertEqual(mod.overlay_color_for("hidden"), "cyan")


class ControllerStateMachineTests(_Base):
    """The raised-open→move / closed→LMB-down / re-open→LMB-up contract, asserted
    on AirMouseController decisions (the pure brain the live loop applies). Every
    sample passes a reference (REF) well BELOW the hand y so the hand is RAISED
    (engaged); the dead-man engage gate itself is exercised separately below."""

    # An engage reference well below the hands used here (0.10 / 0.30), so all
    # these samples count as RAISED. The hysteresis dead-band is ~2-8 cm, so a
    # REF this far down keeps every frame comfortably engaged.
    REF = -0.20

    def _ctrl(self, mod, debounce=1):
        # debounce=1 makes each grip change take effect immediately so the state
        # machine is easy to assert; the debounce itself is tested separately.
        # grace_sec=0 so an untracked frame releases IMMEDIATELY here (the
        # tracking-loss GRACE window is exercised in its own test class).
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=debounce, grace_sec=0.0)

    def test_open_hand_moves_no_button_cyan(self):
        mod = self._load()
        c = self._ctrl(mod)
        d = c.update((0.1, 0.30), "open", True, self.REF)
        self.assertIsNotNone(d.cursor)
        self.assertIsNone(d.button)
        self.assertEqual(d.overlay, "track")
        self.assertEqual(mod.overlay_color_for(d.overlay), "cyan")

    def test_close_presses_left_down_once_gold(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update((0.0, 0.30), "open", True, self.REF)
        d = c.update((0.0, 0.30), "closed", True, self.REF)
        self.assertEqual(d.button, "down")
        self.assertEqual(d.overlay, "grab")
        self.assertEqual(mod.overlay_color_for(d.overlay), "gold")
        # Held: a second closed frame does NOT re-press.
        d2 = c.update((0.0, 0.30), "closed", True, self.REF)
        self.assertIsNone(d2.button)
        self.assertTrue(c.button_is_down)

    def test_closed_hand_still_moves_drag(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update((0.0, 0.30), "open", True, self.REF)
        c.update((0.0, 0.30), "closed", True, self.REF)
        d = c.update((0.2, 0.10), "closed", True, self.REF)   # move while closed
        self.assertIsNotNone(d.cursor)              # cursor still tracks → drag
        self.assertIsNone(d.button)                 # button stays down (no edge)
        self.assertEqual(d.overlay, "grab")

    def test_reopen_releases_left_up_once(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update((0.0, 0.30), "open", True, self.REF)
        c.update((0.0, 0.30), "closed", True, self.REF)       # down
        d = c.update((0.0, 0.30), "open", True, self.REF)     # re-open → up
        self.assertEqual(d.button, "up")
        self.assertEqual(d.overlay, "track")
        self.assertFalse(c.button_is_down)

    def test_deadman_release_when_untracked_while_held(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update((0.0, 0.30), "open", True, self.REF)
        c.update((0.0, 0.30), "closed", True, self.REF)       # button down
        # Hand lost mid-grip. This controller has grace_sec=0, so the loss
        # releases IMMEDIATELY (the grace-window hold is a separate test).
        d = c.update(None, "unknown", False, None)            # hand lost mid-grip
        self.assertEqual(d.button, "up")            # dead-man releases
        self.assertEqual(d.overlay, "hidden")
        self.assertIsNone(d.cursor)
        self.assertFalse(c.button_is_down)
        # Idempotent: a second lost frame keeps it hidden with no new edge.
        d2 = c.update(None, "unknown", False, None)
        self.assertIsNone(d2.button)
        self.assertEqual(d2.overlay, "hidden")

    def test_flicker_does_not_fire_button_with_real_debounce(self):
        mod = self._load()
        # Real debounce (3 frames): a single closed flicker must NOT press.
        c = mod.AirMouseController(mod.ReachBox(2560, 1440), debounce_frames=3)
        c.update((0.0, 0.30), "open", True, self.REF)
        d1 = c.update((0.0, 0.30), "closed", True, self.REF)  # flicker frame 1
        d2 = c.update((0.0, 0.30), "open", True, self.REF)    # back to open
        self.assertIsNone(d1.button)
        self.assertIsNone(d2.button)
        self.assertFalse(c.button_is_down)


# ══════════════════════════════════════════════════════════════════════════
#  DEAD-MAN ENGAGE GATE (FIX 1) — pure core
# ══════════════════════════════════════════════════════════════════════════
class _FakeClock:
    """A controllable monotonic clock for the grace-window tests."""
    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)


class EngageGateTests(_Base):
    """FIX 1 — the cursor is driven ONLY while the hand is RAISED above the body
    reference AND tracked. Lowering the hand, losing tracking (beyond the grace),
    or having no reference DISENGAGES: cursor=None (→ NO SetCursorPos, the
    physical mouse is free), any held button RELEASED, overlay hidden. With
    hysteresis so it doesn't flicker at the line."""

    REF = 0.0   # reference at y=0; engage threshold ~+0.08, disengage ~+0.02.

    def _ctrl(self, mod, **kw):
        kw.setdefault("debounce_frames", 1)
        return mod.AirMouseController(mod.ReachBox(2560, 1440), **kw)

    # ── the headline guarantee: hand DOWN ⇒ disengaged ──────────────────────
    def test_hand_below_reference_disengages_no_cursor(self):
        mod = self._load()
        c = self._ctrl(mod)
        # Hand well BELOW the reference (resting at the side / in the lap).
        d = c.update((0.1, self.REF - 0.20), "open", True, self.REF)
        self.assertIsNone(d.cursor)          # NO SetCursorPos
        self.assertEqual(d.overlay, "hidden")
        self.assertIsNone(d.button)
        self.assertFalse(c.engaged)

    def test_hand_raised_engages_and_moves(self):
        mod = self._load()
        c = self._ctrl(mod)
        d = c.update((0.1, self.REF + 0.20), "open", True, self.REF)  # clearly up
        self.assertIsNotNone(d.cursor)       # cursor driven
        self.assertEqual(d.overlay, "track")
        self.assertTrue(c.engaged)

    def test_lowering_hand_while_held_releases_button(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update((0.0, self.REF + 0.20), "open", True, self.REF)     # engaged
        c.update((0.0, self.REF + 0.20), "closed", True, self.REF)   # LMB down
        self.assertTrue(c.button_is_down)
        # Now DROP the hand below the reference mid-grip → disengage + release.
        d = c.update((0.0, self.REF - 0.20), "closed", True, self.REF)
        self.assertEqual(d.button, "up")     # held button RELEASED
        self.assertIsNone(d.cursor)          # NO SetCursorPos
        self.assertEqual(d.overlay, "hidden")
        self.assertFalse(c.button_is_down)
        self.assertFalse(c.engaged)

    def test_no_reference_disengages(self):
        mod = self._load()
        c = self._ctrl(mod)
        # ref_y=None (couldn't read a spine/elbow joint) → fail safe to released.
        d = c.update((0.1, 0.50), "open", True, None)
        self.assertIsNone(d.cursor)
        self.assertEqual(d.overlay, "hidden")
        self.assertFalse(c.engaged)

    def test_hysteresis_no_flicker_at_threshold(self):
        mod = self._load()
        c = self._ctrl(mod, engage_raise_m=0.08, disengage_drop_m=0.02)
        # Below the ENGAGE bar (ref+0.08) but above the DISENGAGE bar (ref+0.02):
        # while DISENGAGED this stays disengaged (must clear the higher bar).
        d0 = c.update((0.0, self.REF + 0.05), "open", True, self.REF)
        self.assertFalse(c.engaged)
        self.assertIsNone(d0.cursor)
        # Rise above the engage bar → engage.
        d1 = c.update((0.0, self.REF + 0.10), "open", True, self.REF)
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d1.cursor)
        # Sag back into the dead-band (ref+0.05): still above the DISENGAGE bar,
        # so it STAYS engaged — no flicker.
        d2 = c.update((0.0, self.REF + 0.05), "open", True, self.REF)
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d2.cursor)
        # Drop below the disengage bar → disengage.
        d3 = c.update((0.0, self.REF + 0.01), "open", True, self.REF)
        self.assertFalse(c.engaged)
        self.assertIsNone(d3.cursor)

    def test_reengage_after_lowering(self):
        mod = self._load()
        c = self._ctrl(mod)
        c.update((0.0, self.REF + 0.20), "open", True, self.REF)     # engaged
        c.update((0.0, self.REF - 0.20), "open", True, self.REF)     # lowered
        self.assertFalse(c.engaged)
        d = c.update((0.0, self.REF + 0.20), "open", True, self.REF)  # raised again
        self.assertTrue(c.engaged)
        self.assertIsNotNone(d.cursor)

    # ── tracking-loss grace window (injected clock) ─────────────────────────
    def test_brief_dropout_holds_then_releases_after_grace(self):
        mod = self._load()
        clk = _FakeClock(100.0)
        c = self._ctrl(mod, grace_sec=0.30, clock=clk)
        c.update((0.0, self.REF + 0.20), "open", True, self.REF)     # engaged
        c.update((0.0, self.REF + 0.20), "closed", True, self.REF)   # LMB down
        self.assertTrue(c.button_is_down)
        # A brief untracked dropout WITHIN the grace: hold — button stays down,
        # no cursor motion (no sample), overlay keeps "grab".
        clk.advance(0.10)
        d_hold = c.update(None, "unknown", False, None)
        self.assertIsNone(d_hold.button)         # NOT released yet
        self.assertIsNone(d_hold.cursor)         # no SetCursorPos
        self.assertEqual(d_hold.overlay, "grab")
        self.assertTrue(c.button_is_down)
        # Dropout persists PAST the grace → full dead-man release.
        clk.advance(0.40)
        d_rel = c.update(None, "unknown", False, None)
        self.assertEqual(d_rel.button, "up")
        self.assertEqual(d_rel.overlay, "hidden")
        self.assertFalse(c.button_is_down)

    def test_hand_down_means_zero_cursor_over_many_frames(self):
        mod = self._load()
        c = self._ctrl(mod)
        # The owner's lockout scenario: hand resting low for a long run of frames
        # must NEVER yield a cursor (so the live loop calls SetCursorPos zero
        # times) and must never hold a button.
        cursors = []
        buttons = []
        for _ in range(60):
            d = c.update((0.2, self.REF - 0.25), "open", True, self.REF)
            cursors.append(d.cursor)
            buttons.append(d.button)
        self.assertTrue(all(c0 is None for c0 in cursors))   # zero cursor moves
        self.assertTrue(all(b is None for b in buttons))
        self.assertFalse(c.engaged)


# ══════════════════════════════════════════════════════════════════════════
#  LIVE _poll_once — sensor + mouse mocked
# ══════════════════════════════════════════════════════════════════════════
class PollActsTests(_Base):
    def _ctrl(self, mod):
        # grace_sec=0 so a hand-loss frame releases at once (deterministic);
        # the grace window is covered in EngageGateTests with an injected clock.
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0)

    def test_open_hand_moves_cursor(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        bridge = _fake_bridge(bodies=[_body(grip_right="open")])
        ctrl = self._ctrl(mod)
        d = mod._poll_once(ctrl, bridge)
        self.assertEqual(len(moves), 1)             # cursor moved
        self.assertEqual(buttons, [])               # no button
        self.assertEqual(d.overlay, "track")

    def test_close_then_open_clicks_left(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        # open → closed → open : a LEFT-click (down then up).
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="closed")]))
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        self.assertEqual(buttons, ["down", "up"])
        # And it actuates the LEFT (primary) button, never the right.
        self.assertEqual(mod.AIR_MOUSE_BUTTON, "left")

    def test_deadman_releases_when_hand_lost(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="closed")]))
        self.assertEqual(buttons, ["down"])
        # Hand leaves the frame (no bodies) → dead-man release.
        mod._poll_once(ctrl, _fake_bridge(bodies=[]))
        self.assertEqual(buttons, ["down", "up"])


class PollGateTests(_Base):
    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0)

    def test_noop_side_effects_when_flag_off(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(False)                     # air-mouse OFF
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        d = mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        self.assertEqual(moves, [])                 # cursor NOT moved
        self.assertEqual(buttons, [])               # no button
        self.assertIsNotNone(d)                     # controller still advanced

    def test_flag_off_during_drag_still_releases(self):
        mod = self._load()
        self._not_staging(mod)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        # Grab while enabled.
        self._patch_flag(True)
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="closed")]))
        self.assertEqual(buttons, ["down"])
        # Now flip OFF mid-drag, then re-open: the pending 'up' must still fire
        # so the button isn't stranded down.
        from core import config as cfg
        with mock.patch.object(cfg, "KINECT_AIR_MOUSE_ENABLED", False, create=True):
            mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        self.assertEqual(buttons, ["down", "up"])

    def test_noop_when_staging(self):
        mod = self._load()
        p = mock.patch.object(mod, "_is_staging", lambda: True)
        p.start(); self.addCleanup(p.stop)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(bodies=[_body(grip_right="open")]))
        self.assertEqual(moves, [])
        self.assertEqual(buttons, [])

    def test_noop_when_bridge_absent(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        ctrl = self._ctrl(mod)
        self.assertIsNone(mod._poll_once(ctrl, None))

    def test_noop_when_sensor_disabled(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(enabled=False,
                                          bodies=[_body(grip_right="open")]))
        self.assertEqual(moves, [])                 # dead-man: not tracked

    def test_noop_when_sensor_unavailable(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(available=(False, "no sensor"),
                                          bodies=[_body(grip_right="open")]))
        self.assertEqual(moves, [])


# ══════════════════════════════════════════════════════════════════════════
#  LIVE dead-man ENGAGE gate end-to-end (FIX 1) — _poll_once + mocked mouse
# ══════════════════════════════════════════════════════════════════════════
class PollEngageGateTests(_Base):
    """The owner-facing guarantee through the LIVE path: a LOWERED (or
    no-reference) hand drives ZERO _set_cursor_pos calls — releasing the real
    mouse — while a RAISED hand moves it. Mouse actuation mocked; no real cursor
    is touched."""

    def _ctrl(self, mod):
        return mod.AirMouseController(mod.ReachBox(2560, 1440),
                                      debounce_frames=1, grace_sec=0.0)

    def test_hand_down_makes_zero_setcursorpos_calls(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        # Hand resting BELOW the spine_mid reference (ref_y default 0.0) for a
        # long run of frames — the owner's hand at rest. Confirm: ZERO cursor
        # moves (the physical mouse is never fought) and never a button.
        for _ in range(30):
            mod._poll_once(ctrl, _fake_bridge(
                bodies=[_body(grip_right="open", hand_y=-0.25, ref_y=0.0)]))
        self.assertEqual(moves, [])                 # ZERO SetCursorPos
        self.assertEqual(buttons, [])

    def test_lowering_hand_mid_drag_releases_and_stops_cursor(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        # Raise + grab (engaged, LMB down), then DROP the hand → must release the
        # button and stop moving the cursor.
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(grip_right="open", hand_y=0.30, ref_y=0.0)]))
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(grip_right="closed", hand_y=0.30, ref_y=0.0)]))
        self.assertEqual(buttons, ["down"])
        moves_before = len(moves)
        # Hand drops low for several frames.
        for _ in range(10):
            mod._poll_once(ctrl, _fake_bridge(
                bodies=[_body(grip_right="closed", hand_y=-0.25, ref_y=0.0)]))
        self.assertEqual(buttons, ["down", "up"])   # released exactly once
        self.assertEqual(len(moves), moves_before)  # no further cursor motion

    def test_raised_hand_moves_cursor(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        mod._poll_once(ctrl, _fake_bridge(
            bodies=[_body(grip_right="open", hand_y=0.30, ref_y=0.0)]))
        self.assertEqual(len(moves), 1)             # raised → cursor driven

    def test_no_reference_joint_makes_zero_setcursorpos(self):
        mod = self._load()
        self._not_staging(mod)
        self._patch_flag(True)
        moves, buttons = self._capture_mouse(mod)
        ctrl = self._ctrl(mod)
        # Body tracked but NO spine/shoulder/elbow joints → ref_y=None → fail
        # safe to disengaged → zero cursor moves.
        for _ in range(10):
            mod._poll_once(ctrl, _fake_bridge(
                bodies=[_body(grip_right="open", hand_y=0.9, with_ref=False)]))
        self.assertEqual(moves, [])


# ══════════════════════════════════════════════════════════════════════════
#  LEFT-button actuation (FIX 2) — the win32 flag mapping
# ══════════════════════════════════════════════════════════════════════════
class LeftButtonActuationTests(_Base):
    """FIX 2 — the closed hand actuates the LEFT (primary) button. Asserts both
    the config default and that _mouse_button() emits the LEFT win32 events,
    NEVER the right ones."""

    def test_button_is_left_by_default(self):
        mod = self._load()
        self.assertEqual(mod.AIR_MOUSE_BUTTON, "left")

    def test_mouse_button_emits_left_win32_events_not_right(self):
        mod = self._load()
        # Fake win32api/win32con so _mouse_button takes the win32 path and we can
        # capture which event flag it fires (no real mouse touched).
        fired = []
        fake_win32api = types.ModuleType("win32api")
        fake_win32api.mouse_event = lambda flag, *a, **k: fired.append(flag)
        fake_win32con = types.ModuleType("win32con")
        fake_win32con.MOUSEEVENTF_LEFTDOWN = 0x0002
        fake_win32con.MOUSEEVENTF_LEFTUP = 0x0004
        fake_win32con.MOUSEEVENTF_RIGHTDOWN = 0x0008
        fake_win32con.MOUSEEVENTF_RIGHTUP = 0x0010
        with mock.patch.dict(sys.modules, {"win32api": fake_win32api,
                                           "win32con": fake_win32con}):
            self.assertTrue(mod._mouse_button("down"))
            self.assertTrue(mod._mouse_button("up"))
        # LEFT down then LEFT up — and NEITHER right flag ever fired.
        self.assertEqual(fired, [0x0002, 0x0004])
        self.assertNotIn(0x0008, fired)
        self.assertNotIn(0x0010, fired)


# ══════════════════════════════════════════════════════════════════════════
#  toggle + persistence + status
# ══════════════════════════════════════════════════════════════════════════
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

    def _inject_bridge(self, mod, bridge):
        old = sys.modules.get("audio.kinect_bridge")
        sys.modules["audio.kinect_bridge"] = bridge
        self.addCleanup(
            lambda: sys.modules.__setitem__("audio.kinect_bridge", old)
            if old is not None else sys.modules.pop("audio.kinect_bridge", None))

    def test_on_persists_flag(self):
        mod = self._load()
        self._patch_flag(False)
        saved = self._patch_settings_writer()
        self._inject_bridge(mod, _fake_bridge(enabled=True))
        out = mod.air_mouse_on("")
        self.assertIn("on", out.lower())
        self.assertTrue(saved.get("KINECT_AIR_MOUSE_ENABLED"))
        from core import config as cfg
        self.assertTrue(cfg.KINECT_AIR_MOUSE_ENABLED)

    def test_off_persists_flag_and_releases(self):
        mod = self._load()
        self._patch_flag(True)
        saved = self._patch_settings_writer({"KINECT_AIR_MOUSE_ENABLED": True})
        # off() releases the button + clears the overlay — make those inert.
        released = []
        with mock.patch.object(mod, "_mouse_button",
                               lambda a: released.append(a)), \
                mock.patch.object(mod, "_shutdown_overlay", lambda: None), \
                mock.patch.object(mod, "_clear_overlay_state", lambda: None):
            out = mod.air_mouse_off("")
        self.assertIn("off", out.lower())
        self.assertFalse(saved.get("KINECT_AIR_MOUSE_ENABLED"))
        self.assertIn("up", released)               # safety release on disable

    def test_on_warns_when_sensor_off(self):
        mod = self._load()
        self._patch_flag(False)
        self._patch_settings_writer()
        self._inject_bridge(mod, _fake_bridge(enabled=False))
        out = mod.air_mouse_on("")
        self.assertIn("kinect", out.lower())        # mentions the sensor is off


class StatusTests(_Base):
    def _inject_bridge(self, bridge):
        old = sys.modules.get("audio.kinect_bridge")
        sys.modules["audio.kinect_bridge"] = bridge
        self.addCleanup(
            lambda: sys.modules.__setitem__("audio.kinect_bridge", old)
            if old is not None else sys.modules.pop("audio.kinect_bridge", None))

    def test_status_off(self):
        mod = self._load()
        self._patch_flag(False)
        self.assertIn("off", mod.air_mouse_status("").lower())

    def test_status_on_with_hand_in_view(self):
        mod = self._load()
        self._patch_flag(True)
        self._inject_bridge(_fake_bridge(
            hand_states={"right": "open", "left": "unknown",
                         "tracked": True, "ts": 0.0}))
        out = mod.air_mouse_status("")
        self.assertIn("on", out.lower())
        self.assertIn("see your hand", out.lower())

    def test_status_on_but_sensor_off(self):
        mod = self._load()
        self._patch_flag(True)
        self._inject_bridge(_fake_bridge(enabled=False))
        out = mod.air_mouse_status("")
        self.assertIn("on", out.lower())
        self.assertIn("unavailable", out.lower())


# ─── registration wires the three actions ──────────────────────────────────
class RegisterTests(_Base):
    def test_register_adds_actions_without_starting_real_thread(self):
        # neuter_threads=True (harness default) makes Thread.start a no-op, so
        # register() wires the actions but never spins the live poller.
        mod, actions = load_skill_isolated("kinect_air_mouse", register=True)
        for name in ("air_mouse_on", "air_mouse_off", "air_mouse_status"):
            self.assertIn(name, actions)
            self.assertTrue(callable(actions[name]))


if __name__ == "__main__":
    unittest.main()
