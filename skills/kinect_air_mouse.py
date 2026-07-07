"""
kinect_air_mouse skill — a Kinect v2 "air-mouse" for JARVIS.

THE FEATURE  (RAISE-TO-ENGAGE model + AUTO-YIELD)
=================================================
RAISE a hand clearly ABOVE THE SHOULDER to take the cursor; the cursor follows
that raised hand. CLOSE a hand to click, and which hand clicks which button is
HAND-SPECIFIC. When you LOWER the hand back down (below the shoulder) the
air-mouse DISENGAGES and gestures take over. A hand resting on the desk sits far
below the shoulder, so it does NOTHING — only a raised hand engages. And the
instant you touch your REAL mouse or keyboard the air-mouse YIELDS — it releases
and stays out of the way until ~1.5 s after your last real input.

  • hand RAISED above the shoulder (camera-up Y delta clears the engage margin)
                          → ENGAGE: the cursor follows that hand.
  • LEFT hand closes      → LEFT mouse button (down on close, up on open; hold-
                            closed + move = a LEFT-drag).
  • RIGHT hand closes     → RIGHT mouse button (right-click; hold = right-drag).
                            Either hand can click regardless of which one is
                            driving the cursor.
  • hand LOWERED / lost   → DISENGAGE — the cursor is released so the PHYSICAL
                            mouse works again, any held button is let go, the
                            reticle hides, and GESTURES re-arm. Raise again to
                            re-engage.
  • REAL mouse/keyboard   → AUTO-YIELD: force-disengage + stay suppressed until
                            ~1.5 s after the most recent real hardware input.

So: raise a hand = mouse (left/right hand = left/right click), lower = gestures,
hand on the desk = nothing, touch the real mouse = instant yield.

A glowing JARVIS targeting reticle (hud/jarvis_air_cursor.py, a separate
click-THROUGH overlay process) follows the cursor — cyan + gently pulsing while
TRACKING the extended hand, snapping inward to a GOLD lock on a grab/drag, and
HIDDEN while disengaged.

This module is the LIVE WIRING; the testable core is pure and lives right here
alongside it (no sensor, no real mouse, no Qt needed to exercise it):

  • ReachBox + map_hand_to_cursor() — turn a hand position (camera-space metres)
    into an absolute VIRTUAL-DESKTOP pixel (spanning ALL monitors), clamped to
    the desktop bounds.
  • EMA — exponential smoothing to fight the Kinect's hand-joint jitter.
  • GripDebouncer — the per-hand open/closed state machine: requires N
    consecutive frames of a new grip before it flips, tolerates 1-frame Unknown
    dropouts (carries the last confident grip), and treats Lasso as closed — so a
    single flickered frame never fires a stray click and a fist reliably clicks /
    an open hand reliably releases.
  • ArmExtension lift gate — the HEIGHT gate: hand_y minus a shoulder-line Y
    reference (lift_m), with up/down engage HYSTERESIS so a hand hovering at the
    line can't flap, plus a short tracking-loss GRACE so a 1-frame dropout doesn't
    strand a held button. forward-reach/straightness are demoted to non-gating.
  • _air_mouse_yield — the AUTO-YIELD watcher: a low-level WH_MOUSE_LL /
    WH_KEYBOARD_LL hook (dedicated thread + message pump) that timestamps the last
    REAL (non-injected) hardware input, so the air-mouse force-disengages + stays
    suppressed the instant the owner touches their real mouse/keyboard.
  • AirMouseController — ties those together into a per-frame decision:
    cursor_xy, per-hand button edges (left/right down|up), overlay state, the
    controlling hand, and the per-hand grips — for the HUD preview hand-circle.

V1 MAPPING (deliberately simple + robust)
=========================================
The EXTENDED hand's (x, y) is mapped from a CALIBRATED comfortable reach-box in
front of the user onto the ENTIRE virtual desktop (every monitor, including any
left of / above the primary, which have a negative virtual-screen origin). This
is robust and needs no calibration ritual — it just maps "hand left↔right /
up↔down within arm's reach" to "cursor left↔right / up↔down across all screens",
NON-mirrored (hand right → cursor right). It is NOT ray-projection.

  v2 (deferred, noted here so it isn't lost): project the actual arm RAY
  (shoulder→hand, via audio/kinect_pointing.arm_direction) onto each monitor's
  screen plane for true "point AT the pixel" aiming across the whole multi-
  monitor virtual desktop, plus per-user reach-box calibration and fine-tuning
  of the smoothing / dead-zone. v1 ships the simple mapping so it's usable today.

EVERYTHING is opt-in + safe (mirrors skills/kinect_gestures.py):
  • Gated by core.config.KINECT_AIR_MOUSE_ENABLED (default False), RE-READ each
    tick so a Settings toggle takes effect with no restart.
  • A staging / test instance NEVER moves the mouse (JARVIS_STAGING /
    bobert_companion._is_staging()) — the poll loop self-gates every tick.
  • All sensor contact is via audio/kinect_bridge (accessors never raise); a
    missing / disabled sensor degrades to a quiet no-op.
  • DEAD-MAN / ENGAGE GATE: the cursor is driven ONLY while a hand is RAISED above
    the shoulder (lift_m clears the engage margin — see ArmExtension below) AND the
    body+hand are tracked AND no real input is recent. The moment the owner LOWERS
    the hand below the shoulder — or the hand/body goes untracked for more than
    DISENGAGE_GRACE_SEC (~0.3 s), or the owner touches their REAL mouse/keyboard —
    the air-mouse DISENGAGES: it stops calling SetCursorPos entirely (releasing
    control so the PHYSICAL mouse works again), releases any held button, and hides
    the reticle. Up/down engage hysteresis keeps it from flickering at the line, and
    a closed hand that drops or leaves the frame can never strand the button down.

Voice actions:
  air_mouse_on / air_mouse_off — toggle KINECT_AIR_MOUSE_ENABLED live, persisted
                                 via the same Settings writer kinect_gestures /
                                 model_picker use.
  air_mouse_status             — is the air-mouse on + can I see your hand.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional


# ─── AUTO-YIELD bridge (skills/_air_mouse_yield) ─────────────────────────────
# Thin, import-light wrappers around the low-level input watcher so the rest of
# the module (and the tests) can ask "did the owner just touch their real
# mouse/keyboard?" without importing ctypes here. The watcher is lazy: install()
# is a no-op until first called from the live loop, and every accessor degrades to
# a safe default if the helper can't load. NEVER raise.
def _yield_mod():
    try:
        from skills import _air_mouse_yield as _y
        return _y
    except Exception:
        try:
            import _air_mouse_yield as _y   # isolated-skill import fallback
            return _y
        except Exception:
            return None


def _install_yield_watcher() -> bool:
    """Lazily install the real-input hook (or its polling fallback). Idempotent;
    safe to call every tick. Returns True iff the LL hook is active."""
    y = _yield_mod()
    if y is None:
        return False
    try:
        return bool(y.install())
    except Exception:
        return False


def real_input_recent(window: "Optional[float]" = None,
                      now: "Optional[float]" = None) -> bool:
    """True when REAL (non-injected) hardware input happened within `window`
    seconds — the signal that the air-mouse must YIELD and stay SUPPRESSED.
    Defaults to AIR_MOUSE_YIELD_WINDOW_SEC. Safe (False) when the watcher is
    unavailable. NEVER raises."""
    if window is None:
        window = AIR_MOUSE_YIELD_WINDOW_SEC
    y = _yield_mod()
    if y is None:
        return False
    try:
        return bool(y.real_input_recent(window, now))
    except Exception:
        return False


def _mark_self_action() -> None:
    """Tell the watcher the air-mouse just moved/clicked the cursor (so its polling
    fallback won't mistake our own activity for the owner's). NEVER raises."""
    y = _yield_mod()
    if y is None:
        return
    try:
        y.mark_self_action()
    except Exception:
        pass


# ─── tunables ────────────────────────────────────────────────────────────
AIR_MOUSE_POLL_HZ = 30.0                      # cursor update rate (~30 Hz)
AIR_MOUSE_POLL_INTERVAL = 1.0 / AIR_MOUSE_POLL_HZ
INITIAL_DELAY_SECONDS = 6.0                   # let the monolith + bridge come up
_THREAD_NAME = "kinect-air-mouse-skill"

# EMA smoothing factor for the cursor (0..1). LOWER = smoother but laggier;
# HIGHER = snappier but jitterier. Tunable; v2 may auto-adapt it to hand speed.
#
# TUNED 2026-06-08 (owner: "works but laggy, make it snappier"): 0.35 → 0.55.
#   At the 30 Hz poll rate the EMA's response time-constant is ~1/alpha frames.
#   0.35 → ≈2.9-frame constant (~95 ms to reach 63 % of a step) which read as a
#   cursor "dragging behind" the hand. 0.55 → ≈1.8-frame constant (~60 ms): the
#   cursor catches the hand markedly faster (≈35 ms less lag) while still
#   averaging roughly two frames, so the Kinect hand-joint jitter is still tamed
#   (a lone jittered sample only moves the cursor ~55 % of the way, not 100 %).
#   Nudge back toward 0.4-0.45 if it feels twitchy; toward 0.6-0.7 for even less
#   lag at the cost of more jitter.
AIR_MOUSE_EMA_ALPHA = 0.55

# How many CONSECUTIVE frames a new grip (open↔closed) must persist before the
# per-hand state machine accepts it. This is the anti-stray-click guard: a single
# flickered Kinect hand-state frame must never fire a click.
#
# TUNED 2026-06-08 (owner: "click should fire promptly"): 3 → 2.
#   At 30 Hz, 3 frames ≈ 100 ms of latency before a close registers as a click;
#   2 frames ≈ 67 ms — the click fires ~33 ms sooner so it feels prompt/instant.
#   TWO consecutive frames still rejects a lone 1-frame flicker (the actual
#   failure mode the Kinect produces), so accidental clicks are still prevented;
#   we only gave up the third confirmation frame, which was belt-and-suspenders.
#   Raise back to 3 if any stray clicks appear; 2 is the snappy-but-safe floor.
AIR_MOUSE_GRIP_DEBOUNCE_FRAMES = 2

# ─── HEIGHT-TO-ENGAGE gate (RAISE a hand ABOVE THE SHOULDER to drive the cursor) ─
# THE PRIMARY GATE (the 2026-06-09 "forward-reach is broken for a desk user" fix):
#   The forward-reach model was PROVEN unusable at a desk — the live log showed
#   hands RESTING on the desk reading forward_reach ~0.38 m / ratio ~1.0-1.25,
#   INDISTINGUISHABLE from a deliberate reach (~0.60 m / ratio ~1.8), so the
#   air-mouse stayed engaged 574/585 frames and the owner could not use their real
#   mouse. The owner chose RAISE-HIGH to engage.
#
#   The air-mouse now engages ONLY while the controlling hand is raised clearly
#   ABOVE THE SHOULDER. In Kinect camera space joint Y increases UPWARD, so we
#   compare the hand-joint Y to a shoulder-line reference Y (spine_shoulder, or the
#   same-side shoulder as a fallback) and gate on the HEIGHT delta lift_m =
#   hand_y - shoulder_ref_y, with HYSTERESIS:
#       • to ENGAGE       lift_m must clear the HIGHER bar  (hand at/above shoulder)
#       • to STAY engaged lift_m may relax only to the LOWER bar; the instant it
#                         drops below that → DISENGAGE.
#   This is BODY-RELATIVE — invariant to rotation, chair position, and distance: a
#   hand resting on the desk sits at WAIST level, far below the shoulder (lift_m
#   strongly negative), so it NEVER engages; a hand raised to point at the screen
#   sits at/above shoulder level (lift_m ≳ 0), so it does. The CONTROLLING hand is
#   whichever hand is raised HIGHEST above the shoulder (sticky-hand hysteresis
#   keeps it from thrashing); if neither hand is above the engage line, NOT engaged.
#
# FORWARD-REACH IS DEMOTED to an optional weak secondary cue that can never engage
# or hold the gate (its engage/disengage bars are made permissive below so they
# cannot keep the air-mouse engaged); HEIGHT is the necessary + primary gate.
#
# DEFAULTS: a hand AT the shoulder (lift_m ≈ 0) sits just under the engage line;
# raising it a touch (≳ +7 cm) engages; dropping it ~10 cm below the shoulder
# releases.
#
# TUNED 2026-06-09 (owner: "better about not triggering unless I reach but not
#   perfect"): up-margin 0.05 → 0.07. A hand merely AT shoulder height (lift ≈ 0,
#   e.g. resting an elbow / gesturing) sat only 5 cm under the old bar and
#   occasionally tripped a false engage; +7 cm requires a clearer, deliberate LIFT
#   to take the cursor while still being an easy reach. PAIRED with the engage
#   DEBOUNCE below (the lift must HOLD above the bar for a few frames) so a 1-frame
#   Kinect Y-spike can't engage. DISENGAGE stays responsive (immediate drop below
#   the down-margin / on yield) — only ENGAGE is hardened.
AIR_MOUSE_ENGAGE_UP_MARGIN_M = 0.07      # lift_m must exceed this to ENGAGE (hand
#                                          clearly above the shoulder)
AIR_MOUSE_ENGAGE_DOWN_MARGIN_M = -0.10   # drop below this (~10 cm under the
#                                          shoulder) to DISENGAGE (hysteresis:
#                                          DOWN < UP so the gate can't flap)

# ENGAGE DEBOUNCE (FIX 3): how many CONSECUTIVE frames the lift must hold above the
# engage (up) margin before the air-mouse actually ENGAGES. This rejects a 1-frame
# Kinect height spike (a momentary hand-joint jump above the shoulder) so a brief
# blip never grabs the cursor — only a SUSTAINED raise does. DISENGAGE is NOT
# debounced (it stays instant on a drop below the down-margin or on auto-yield), so
# the gate is hard to engage by accident but quick to release.
#
# At 30 Hz, 3 frames ≈ 100 ms — long enough to swallow a lone spike, short enough
# that a real raise still feels immediate. Drop to 2 if engaging feels sluggish;
# raise toward 4-5 if spikes still slip through.
AIR_MOUSE_ENGAGE_DEBOUNCE_FRAMES = 3

# ─── LEGACY forward-REACH cues (DEMOTED to permissive secondary) ─────────────
# Forward reach is no longer the gate. These bars are kept ONLY so the old
# calibration keys / value-object plumbing keep working; they're set PERMISSIVE
# (≈0) so a forward arm can never independently engage or HOLD the height gate.
# PERMISSIVE (≈0) so forward reach can neither engage nor HOLD the gate — the
# height gate above is the only thing that engages / holds. A relaxed hand on the
# desk reads forward_reach ~0.38 m, INDISTINGUISHABLE from a real reach, which is
# exactly why forward reach was abandoned as the gate; setting these to 0 makes the
# forward cue always "satisfied" so it can't VETO a valid raise, while the height
# delta does the real gating. (Kept as named constants for the calibration keys.)
AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE = 0.0     # permissive: forward never gates
AIR_MOUSE_EXTEND_REACH_RATIO_DISENGAGE = 0.0  # permissive: forward never gates
#
# LEGACY ABSOLUTE forward-depth bars (metres) — also permissive now (the height
# gate replaced them). Retained as named constants for back-compat plumbing.
AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M = 0.0     # permissive (height gate is primary)
AIR_MOUSE_EXTEND_FORWARD_DISENGAGE_M = 0.0  # permissive (height gate is primary)
#
# ARM-STRAIGHTNESS — no longer a gate or a veto under the height model. Set to 0
# (permissive) so a raised hand is never rejected for a bent elbow; kept as a named
# constant only for the persisted KINECT_STRAIGHT_* keys / calibration plumbing.
AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE = 0.0      # permissive (height gate is primary)
AIR_MOUSE_EXTEND_STRAIGHT_DISENGAGE = 0.0   # permissive (height gate is primary)

# ─── persisted per-body CALIBRATION (data/user_settings.json) ────────────────
# 'calibrate air mouse' / 'calibrate reach' captures the owner's RELAXED +
# EXTENDED reach RATIO (body-relative) and writes the fitted thresholds under
# these keys; the live gate reads them every tick (via _reach_thresholds()),
# falling back to the (position-independent) defaults above when unset. The
# engage bar is placed ~60 % of the way relaxed→extended and the disengage bar
# ~40 %, so the gate fits the owner's actual RATIO range (the calibrated cue).
# Because the stored value is the RELATIVE ratio (not absolute metres), a
# calibration done at one seating distance holds at any other — and the defaults
# are good enough that calibration is OPTIONAL. Straightness is NOT calibrated as
# an engage span any more (it is only the modest ENGAGE-time veto floor above), but
# the KINECT_STRAIGHT_* keys are still read/written for back-compat.
#
# NB the key names are kept (KINECT_REACH_ENGAGE / _DISENGAGE) for back-compat with
# any persisted settings, but they now hold the dimensionless REACH RATIO, not
# metres. KINECT_STRAIGHT_* are unchanged (straightness is already body-relative).
SETTING_REACH_ENGAGE = "KINECT_REACH_ENGAGE"              # reach-ratio engage
SETTING_REACH_DISENGAGE = "KINECT_REACH_DISENGAGE"        # reach-ratio disengage
SETTING_STRAIGHT_ENGAGE = "KINECT_STRAIGHT_ENGAGE"        # straightness engage
SETTING_STRAIGHT_DISENGAGE = "KINECT_STRAIGHT_DISENGAGE"  # straightness disengage
# HEIGHT-gate margins (the live PRIMARY gate). Optional persisted overrides so the
# owner can tune the raise-to-engage line live; default to the module margins.
SETTING_UP_MARGIN = "KINECT_LIFT_UP_MARGIN"              # lift to ENGAGE (m)
SETTING_DOWN_MARGIN = "KINECT_LIFT_DOWN_MARGIN"          # lift to DISENGAGE (m)
CALIB_ENGAGE_FRACTION = 0.60     # engage bar this far relaxed→extended
CALIB_DISENGAGE_FRACTION = 0.40  # disengage bar this far relaxed→extended

# ─── HAND MIRROR (selfie-view correction) ────────────────────────────────────
# The Kinect color/skeleton stream the owner sees is MIRRORED (selfie view), so
# the owner's REAL left hand appears on the RIGHT of the image and vice-versa.
# The owner reported clicks + the controlling-hand circle landing on the WRONG
# side. With KINECT_HAND_MIRROR True (the owner's default) the air-mouse SWAPS the
# bridge's left↔right hands — BOTH the grip strings AND the per-arm extension/
# joints — so the owner's REAL left hand → LEFT button + left-side circle and
# their REAL right hand → RIGHT button + right-side circle. Flip this False
# (Settings GUI / user_settings.json) if a future build un-mirrors the stream.
KINECT_HAND_MIRROR_DEFAULT = True

# Grace window for a TRACKING dropout. A single lost/ambiguous frame (the Kinect
# briefly drops the body or hand joint) must NOT instantly disengage and re-snap
# — that would make the cursor jump and a drag stutter. While engaged, a dropout
# is tolerated for up to this long (button stays held, no cursor motion since
# there's no sample); past it the dead-man fully releases. ~0.3 s per the spec.
AIR_MOUSE_DISENGAGE_GRACE_SEC = 0.30

# ABSOLUTE cumulative untracked-time CEILING per engagement (FILTER 7). The 0.30 s
# grace above is renewed on every single Tracked frame, so a body FLICKERING
# tracked/untracked (one good frame, then lost again) could renew the grace
# forever and hold a button-down DRAG far longer than 0.30 s of real tracking. To
# cap that, the controller also sums the time spent UNTRACKED across a single
# engagement; once it exceeds this ceiling the dead-man force-releases regardless
# of how recently a lone Tracked frame renewed the per-dropout grace. A run of
# CONSECUTIVE Tracked frames (a genuinely re-acquired body) RESETS the accumulator
# so a long, healthy session never trips it — only sustained flicker does. ~0.5 s
# absolute, comfortably above the 0.30 s single-dropout grace.
AIR_MOUSE_UNTRACKED_CEILING_SEC = 0.50

# How many CONSECUTIVE fully-Tracked frames re-arm the untracked-time accumulator
# (FILTER 7): a real re-acquisition (the body is solidly back) clears the summed
# untracked time, while a lone 1-frame blip between dropouts does NOT. At 30 Hz, 3
# frames ≈ 100 ms of solid tracking to declare the body re-acquired.
AIR_MOUSE_RETRACK_FRAMES = 3

# ─── SMART-ENGAGE (2026-07, feat/smart-engage) ───────────────────────────────
# The owner: "hand tracking triggers when it shouldn't; I need a foolproof way to
# make it trigger every time I want it but with FEWER false triggers." The old
# gate rode on ONE signal — hand HEIGHT above the shoulder — so ANY raised hand
# (reaching, stretching, gesturing) engaged, and it wasn't reliable when wanted.
#
# The fix is a HYBRID engage model with TWO modes (all decided by the PURE
# engage_decision() below, so it is unit-testable with no sensor):
#
#   • PASSIVE (default, the strict smart-pose gate): the cursor is taken only when
#     ALL of these hold and are SUSTAINED for a brief DWELL —
#       – lift_m clears the up-margin (hand raised above the shoulder),
#       – grip is OPEN (an open palm) when AIR_MOUSE_REQUIRE_OPEN_PALM,
#       – the body FACES the sensor within AIR_MOUSE_FACING_MAX_DEG (missing facing
#         is treated as OK so it never becomes an un-passable gate), and
#       – the hand is held STILL (total travel < AIR_MOUSE_ENGAGE_STILL_M over the
#         dwell window).
#     A natural FAST reach passes through the zone quicker than the dwell and never
#     engages; a deliberate OPEN-PALM + BRIEF HOLD does. This is the false-trigger
#     fix — gesturing / reaching / a closed or pointing hand no longer grabs it.
#
#   • ARMED (opt-in by voice — the owner explicitly asked for control, so be
#     responsive): with AIR_MOUSE_ARM_RELAXES_GATE the gate RELAXES to height-only
#     held for a SHORT debounce (grip / facing / stillness / long dwell NOT
#     required). air_mouse_arm() / air_mouse_disarm() flip the module-level armed
#     flag; disarm returns to PASSIVE (it does NOT disable the feature — the
#     KINECT_AIR_MOUSE_ENABLED master flag is the real off).
#
# DISENGAGE stays snappy in BOTH modes (drop below the down-margin, a sustained
# closed fist while engaged behind AIR_MOUSE_FIST_RELEASES, tracking-loss grace,
# real-input auto-yield, per-app disable, or voice disarm) — never harder than
# before. The passive DWELL PROGRESS (0→1) is published for the HUD priming ring.

def _cfg(name: str, default):
    """Live read of a core.config value (float/bool/list), fresh each call so a
    Settings tweak takes effect with no restart. Returns `default` on any failure.
    NEVER raises. (Booleans use _cfg_flag; this is the general typed reader.)"""
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def _cfg_float(name: str, default: float) -> float:
    try:
        return float(_cfg(name, default))
    except (TypeError, ValueError):
        return float(default)


# PASSIVE smart-pose defaults (mirrors of the core.config knobs — read live at run
# time via _cfg_float, but named here so the pure decision + the tests share one
# source of truth and the module still works if core.config is absent).
AIR_MOUSE_REQUIRE_OPEN_PALM = True       # PASSIVE: require an OPEN palm to engage
AIR_MOUSE_ENGAGE_DWELL_SEC = 0.30        # PASSIVE: hold the full pose this long
AIR_MOUSE_ENGAGE_STILL_M = 0.06          # PASSIVE: max hand travel over the dwell
AIR_MOUSE_FACING_MAX_DEG = 40.0          # PASSIVE: face the sensor within this
AIR_MOUSE_ARM_RELAXES_GATE = True        # ARMED: relax to height-only + short hold
AIR_MOUSE_ARM_ENGAGE_DEBOUNCE_SEC = 0.15  # ARMED: short engage hold (relaxed gate)
AIR_MOUSE_FIST_RELEASES = True           # a sustained fist while engaged releases
AIR_MOUSE_FIST_RELEASE_SEC = 0.60        # …held closed this long counts as release
AIR_MOUSE_PER_APP_DISABLE = True         # stand down over disabled-app windows
AIR_MOUSE_DISABLED_APP_HINTS = [         # lower-case title/class substrings
    "full screen", "fullscreen",
    "netflix", "youtube - ", "prime video",
    "vlc media player", "mpc-hc", "kodi",
    "steam big picture", "moonlight",
    "unrealwindow", "unitywndclass",
]


# ── module-level ARMED flag (single-element-list idiom, per house style) ──────
# _air_mouse_armed[0] True → the owner explicitly asked for the cursor ("mouse
# control on"), so the PASSIVE strict gate relaxes to the responsive armed gate.
# Default False (PASSIVE). Flipped by air_mouse_arm() / air_mouse_disarm(); read
# by the live loop each tick. NB this is orthogonal to KINECT_AIR_MOUSE_ENABLED
# (the master on/off) — disarming returns to passive, it does NOT disable.
_air_mouse_armed = [False]


def air_mouse_arm() -> None:
    """ARM the air-mouse: relax the PASSIVE gate to the responsive armed gate (the
    owner explicitly asked for the cursor). Idempotent. NEVER raises."""
    _air_mouse_armed[0] = True


def air_mouse_disarm() -> None:
    """DISARM the air-mouse: return to the PASSIVE strict smart-pose gate. Does NOT
    disable the feature (KINECT_AIR_MOUSE_ENABLED is the master off). Idempotent."""
    _air_mouse_armed[0] = False


def air_mouse_is_armed() -> bool:
    """True while the air-mouse is ARMED (relaxed gate). NEVER raises."""
    return bool(_air_mouse_armed[0])


# What the pure engage gate decides each frame.
class EngageVerdict:
    """The PURE smart-engage decision for one frame.

      engaged:  bool  — the gate says the cursor should be (or stay) taken.
      priming:  bool  — a valid PASSIVE pose is being HELD toward engage but the
                        dwell hasn't completed yet (drives the HUD priming ring).
      prime:    float — dwell PROGRESS in [0.0, 1.0]: 0 at pose-start, 1 at engage,
                        0.0 when idle or already engaged (nothing to prime).
    """
    __slots__ = ("engaged", "priming", "prime")

    def __init__(self, engaged: bool, priming: bool, prime: float):
        self.engaged = bool(engaged)
        self.priming = bool(priming)
        self.prime = max(0.0, min(1.0, float(prime)))

    def __repr__(self):   # pragma: no cover - debug aid
        return (f"EngageVerdict(engaged={self.engaged}, priming={self.priming}, "
                f"prime={self.prime:.2f})")


def _pose_ok_passive(*, lift_ok: bool, grip: "Optional[str]", facing_deg,
                     hand_still: bool, require_open_palm: bool,
                     facing_max_deg: float) -> bool:
    """Does the PASSIVE smart pose hold THIS frame (before the dwell test)? All of:
    the hand raised (lift_ok), an OPEN palm (when required — the DEBOUNCED stable
    grip so a 1-frame flicker doesn't matter), FACING the sensor within
    facing_max_deg (missing facing → treated as OK, never an un-passable gate), and
    the hand held STILL. PURE; NEVER raises."""
    if not lift_ok:
        return False
    if require_open_palm and (grip or "").lower() != "open":
        return False
    # FACING — skip gracefully when the bridge didn't provide it (facing_deg None):
    # a missing signal must never block engagement (treat as facing OK).
    if facing_deg is not None:
        try:
            if abs(float(facing_deg)) > float(facing_max_deg):
                return False
        except (TypeError, ValueError):
            pass   # unparseable facing → treat as OK, don't gate on garbage
    if not hand_still:
        return False
    return True


def engage_decision(*, lift_ok: bool, currently_engaged: bool, armed: bool,
                    grip: "Optional[str]" = None, facing_deg=None,
                    hand_still: bool = True, dwell_elapsed: float = 0.0,
                    arm_debounce_elapsed: float = 0.0,
                    require_open_palm: "Optional[bool]" = None,
                    facing_max_deg: "Optional[float]" = None,
                    dwell_sec: "Optional[float]" = None,
                    arm_debounce_sec: "Optional[float]" = None,
                    arm_relaxes_gate: "Optional[bool]" = None) -> "EngageVerdict":
    """THE PURE SMART-ENGAGE GATE — decide engage / priming / prime-progress from
    the per-frame signals, with NO sensor, mouse, clock, or config I/O (every knob
    is an argument, defaulting to the live core.config value). Unit-testable
    directly; the controller feeds it the measured signals + elapsed timers.

    Inputs:
      lift_ok            — the HEIGHT gate already passed (hand raised above the
                           shoulder past the engage/stay-engage margin with the
                           existing hysteresis). This wraps the existing lift line.
      currently_engaged  — was the air-mouse engaged last frame (staying engaged
                           only needs lift_ok — the pose/dwell are for ACQUIRING).
      armed              — the module ARMED flag (owner explicitly asked for the
                           cursor).
      grip               — the DEBOUNCED stable grip of the controlling hand
                           ("open"/"closed"/…); the open-palm test uses it.
      facing_deg         — |body yaw| from square, or None when unavailable (None →
                           facing treated as OK, never blocks).
      hand_still         — True when total hand travel over the dwell window is
                           under the stillness bar.
      dwell_elapsed      — seconds the PASSIVE pose has been continuously held.
      arm_debounce_elapsed — seconds the ARMED height-only pose has been held.

    Returns an EngageVerdict(engaged, priming, prime):
      • ARMED + arm_relaxes_gate: RELAXED gate — engage on lift_ok held for
        arm_debounce_sec (grip/facing/stillness/long-dwell NOT required). No
        priming ring (engagement is near-instant); prime stays 0.
      • PASSIVE (or armed with arm_relaxes_gate False): the STRICT smart pose must
        hold for dwell_sec. While the pose holds but the dwell isn't met →
        priming=True, prime = dwell_elapsed / dwell_sec (0→1). The instant it
        completes → engaged=True.
      • STAYING engaged: once engaged, lift_ok alone keeps it (the pose/dwell gate
        only guards ACQUISITION); prime 0, priming False. DISENGAGE (lift lost,
        fist, yield, per-app, disarm) is handled by the controller, not here.

    A fast reach never engages passively: it clears lift for fewer frames than the
    dwell, so the dwell never completes and the pose resets on the way out."""
    # Resolve knobs (arg overrides win; else live config; else module default).
    if require_open_palm is None:
        require_open_palm = bool(_cfg_flag("AIR_MOUSE_REQUIRE_OPEN_PALM",
                                           AIR_MOUSE_REQUIRE_OPEN_PALM))
    if facing_max_deg is None:
        facing_max_deg = _cfg_float("AIR_MOUSE_FACING_MAX_DEG",
                                    AIR_MOUSE_FACING_MAX_DEG)
    if dwell_sec is None:
        dwell_sec = _cfg_float("AIR_MOUSE_ENGAGE_DWELL_SEC",
                               AIR_MOUSE_ENGAGE_DWELL_SEC)
    if arm_debounce_sec is None:
        arm_debounce_sec = _cfg_float("AIR_MOUSE_ARM_ENGAGE_DEBOUNCE_SEC",
                                      AIR_MOUSE_ARM_ENGAGE_DEBOUNCE_SEC)
    if arm_relaxes_gate is None:
        arm_relaxes_gate = bool(_cfg_flag("AIR_MOUSE_ARM_RELAXES_GATE",
                                          AIR_MOUSE_ARM_RELAXES_GATE))

    # ── STAY-ENGAGED: once engaged, height alone holds it (the pose/dwell gate is
    #    only for ACQUIRING). Snappy disengage lives in the controller. ──────────
    if currently_engaged:
        return EngageVerdict(engaged=lift_ok, priming=False, prime=0.0)

    # ── ARMED (relaxed): height-only, held for a short debounce. Responsive by
    #    design — the owner explicitly asked for the cursor. No priming ring. ────
    if armed and arm_relaxes_gate:
        if not lift_ok:
            return EngageVerdict(engaged=False, priming=False, prime=0.0)
        engaged = arm_debounce_elapsed >= max(0.0, float(arm_debounce_sec))
        return EngageVerdict(engaged=engaged, priming=False, prime=0.0)

    # ── PASSIVE strict smart pose (also armed-but-not-relaxed): the full pose must
    #    hold for the dwell. Publish the dwell PROGRESS while priming. ───────────
    pose_ok = _pose_ok_passive(
        lift_ok=lift_ok, grip=grip, facing_deg=facing_deg,
        hand_still=hand_still, require_open_palm=require_open_palm,
        facing_max_deg=facing_max_deg)
    if not pose_ok:
        return EngageVerdict(engaged=False, priming=False, prime=0.0)
    d = max(1e-6, float(dwell_sec))
    prog = max(0.0, min(1.0, float(dwell_elapsed) / d))
    if dwell_elapsed >= float(dwell_sec):
        return EngageVerdict(engaged=True, priming=False, prime=0.0)
    return EngageVerdict(engaged=False, priming=True, prime=prog)


# ─── AUTO-YIELD to real input (the air-mouse never fights the real mouse) ─────
# A low-level input hook (skills/_air_mouse_yield) timestamps the last REAL
# (non-injected) hardware mouse/keyboard event. While real input happened within
# this window the air-mouse YIELDS: it force-disengages (releasing any held
# button + stopping SetCursorPos) and stays SUPPRESSED — it cannot re-engage —
# until this long after the MOST RECENT real input. So the instant the owner
# touches their real mouse or keyboard, the air-mouse releases and stays out of
# the way. The air-mouse's OWN clicks are injected (LLMHF_INJECTED) and ignored by
# the hook, so it never self-suppresses.
AIR_MOUSE_YIELD_WINDOW_SEC = 1.5

# ─── controlling-hand HYSTERESIS (ISSUE 3: both-hands stability) ─────────────
# With BOTH hands raised the cursor must NOT thrash between them frame-to-frame.
# Once a hand is driving the cursor it STAYS the controlling hand until the OTHER
# hand is BOTH clearly more extended (its reach_score leads by at least
# HAND_SWITCH_MARGIN) AND has been so for HAND_SWITCH_FRAMES consecutive frames.
# A brief wobble where the idle hand momentarily out-reaches by a hair can never
# flip control. (The L/R clicks are tracked for BOTH hands regardless — only the
# CURSOR-driving hand is sticky.)
#
# REVERTED 2026-06-09 (v1.73.0 tablet-rework rollback): the v1.73.0 "both-hands
#   lock" tightened these to 0.08 m / 8 frames to pin two raised hands to ONE
#   cursor. The owner did NOT want a single locked cursor when both hands are up —
#   both hands raised now enters TWO-HAND pinch-to-resize mode (skills/
#   kinect_two_hand.py), which suppresses this single-hand cursor entirely. So the
#   single-hand hysteresis is restored to the v1.72.0 values (this only ever
#   matters now for a brief overlap before two-hand mode takes over).
HAND_SWITCH_MARGIN = 0.25     # challenger must lead the holder's score by this
HAND_SWITCH_FRAMES = 6        # …for this many consecutive frames before it wins

# The comfortable reach-box in front of the user, in camera-space METRES, that
# maps onto the whole virtual desktop. Centred roughly on where a seated user's
# hand naturally sits when reaching at the screen. x: sensor-RIGHT is +; the box
# is wider than tall to match a 16:9 screen. y: sensor-UP is +; centred near
# shoulder height. These are the v1 defaults; v2 makes them per-user calibrated.
#   half-width  → ±X metres from centre maps to the desktop's left/right edges
#   half-height → ±Y metres from centre maps to the desktop's top/bottom edges
#
# TUNED 2026-06-08 (owner: "shouldn't need huge arm swings to cross the screen").
#   The smaller the box, the LESS hand travel maps to the full desktop, i.e.
#   higher sensitivity. The old ±0.35 m / ±0.22 m box demanded a ~70 cm-wide
#   sweep edge-to-edge — a whole-arm shoulder swing. A natural pointing arc with
#   the elbow tucked (forearm pivoting at the elbow/wrist) is only ~±25 cm
#   horizontal / ~±15 cm vertical, so:
#     REACH_HALF_W 0.35 → 0.26  (full desktop width  in a ~52 cm hand sweep)
#     REACH_HALF_H 0.22 → 0.16  (full desktop height in a ~32 cm hand sweep)
#   That maps a comfortable forearm arc to the whole virtual desktop — small
#   hand moves now cover the screen. The ~1.6 W:H ratio is kept ≈16:9 so x and y
#   sensitivity stay proportionate (no axis feels twitchier than the other).
#   The EMA + debounce above keep this from feeling jittery despite the higher
#   gain. Widen these (toward the old values) if it feels too sensitive; shrink
#   further for even less travel.
#
# REVERTED 2026-06-09: v1.73.0's "tablet-feel" rework made this plane ABSOLUTE +
#   body-relative + aspect-matched (the cursor pinned to the hand's absolute spot
#   on a body-centred plane). The owner found it JITTERY and unwanted — "it was
#   fine before" — so the mapping is back to the v1.72.0 fixed-centre reach-box
#   below (relative to the sensor axis, fixed half-width/height, NO body centring,
#   NO aspect derivation). The ReachBox.map() is the plain 2-arg form again.
REACH_CENTER_X = 0.0      # metres (centred on the sensor's optical axis)
REACH_CENTER_Y = 0.30     # metres above the sensor (≈ seated shoulder height)
REACH_HALF_W = 0.26       # ±0.26 m horizontal reach → full desktop width
REACH_HALF_H = 0.16       # ±0.16 m vertical reach → full desktop height

# Default geometry used only as a fallback when the real virtual-desktop bounds
# can't be read (headless / win32 absent). The live bounds are resolved at
# runtime by _virtual_screen_bounds().
_DEFAULT_SCREEN_W = 2560
_DEFAULT_SCREEN_H = 1440

# How often the live poll loop re-reads the virtual-desktop bounds, so that
# hot-plugging a monitor / changing the display layout is picked up without a
# restart (the metrics are otherwise cached so we don't hit win32 every tick).
VIRTUAL_BOUNDS_REFRESH_SECONDS = 5.0

# Overlay state-file (sibling to bobert_companion.py — same convention the
# reticle / holo-HUD use). The poller writes the live cursor + grip; the overlay
# process reads it each tick to draw the reticle.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIR_CURSOR_STATE_FILE = os.path.join(PROJECT_DIR, "air_cursor_state.json")


# ══════════════════════════════════════════════════════════════════════════
#  PURE CORE (no sensor, no mouse, no Qt — unit-tested directly)
# ══════════════════════════════════════════════════════════════════════════

# Kinect v2 TrackingState slot (the 4th element of a joint tuple): 0=NotTracked,
# 1=Inferred (the SDK GUESSED the position — noisy + unreliable), 2=Tracked. The
# interaction skills must act ONLY on joints the sensor actually SEES (state >= 2),
# never on an inferred/guessed joint, so a phantom 2nd-hand / occluded joint can't
# drive the cursor, grab a window, or fire a click. (FILTERS 1/2/4 — mirrors the
# bridge arm_extension TrackingState fix on another branch.)
JOINT_TRACKED_STATE = 2


def joint_well_tracked(joint, *, min_state: int = JOINT_TRACKED_STATE) -> bool:
    """True iff `joint` is a usable, sensor-TRACKED joint: it carries (x, y, z) +
    a TrackingState slot [3] >= min_state, its coords are FINITE, and it is NOT
    the exact-zero (0, 0, 0) origin sentinel the SDK emits for an unseen joint.
    An INFERRED (state 1) or NOT-tracked (state 0) joint, a degenerate all-zero
    joint, or a NaN/±inf coordinate all read FALSE so the gate / click / grab
    code leaves them alone. PURE; NEVER raises (a malformed joint reads False)."""
    try:
        if not joint or len(joint) < 4:
            return False
        if int(joint[3]) < int(min_state):
            return False
        x, y, z = float(joint[0]), float(joint[1]), float(joint[2])
        # Reject non-finite coords (NaN / ±inf) — they'd poison the lift/dist math.
        for v in (x, y, z):
            if v != v or v == float("inf") or v == float("-inf"):
                return False
        # Reject the exact-zero origin sentinel (an unseen joint reads (0, 0, 0)).
        if x == 0.0 and y == 0.0 and z == 0.0:
            return False
        return True
    except (TypeError, ValueError, IndexError):
        return False


class ReachBox:
    """The comfortable reach-box → VIRTUAL-DESKTOP mapping.

    Maps a hand (x, y) in camera-space metres onto an absolute virtual-desktop
    pixel that spans EVERY monitor. The desktop is described by its top-left
    origin (origin_x, origin_y) and its (width, height); the origin is NEGATIVE
    for monitors arranged left-of / above the primary, so a fully left monitor is
    reachable too (SetCursorPos accepts these virtual coordinates directly). A
    hand at the box centre lands at the desktop centre, the box edges land at the
    desktop edges, and anything beyond is CLAMPED to the desktop bounds (so a hand
    that overshoots the box parks the cursor at the edge rather than flying off).

    X is NON-mirrored: the Kinect color/body image is itself mirror-flipped
    relative to the user, so the user's hand moving to THEIR right reads as +x and
    we map +x straight to a larger cursor x — hand right → cursor right, hand left
    → cursor left, natural and un-mirrored. y increases UP in camera space but
    screen y increases DOWN, so y is inverted.

    Back-compat: the 2-positional form ``ReachBox(width, height)`` keeps the old
    primary-only behaviour with a (0, 0) origin; pass origin_x / origin_y to span
    the whole virtual desktop."""

    def __init__(self, width: int, height: int,
                 origin_x: int = 0, origin_y: int = 0,
                 center_x: float = REACH_CENTER_X,
                 center_y: float = REACH_CENTER_Y,
                 half_w: float = REACH_HALF_W,
                 half_h: float = REACH_HALF_H):
        # Kept named screen_w / screen_h for back-compat with existing callers;
        # these are the virtual-desktop extents (all monitors), not just primary.
        self.screen_w = int(width)
        self.screen_h = int(height)
        self.origin_x = int(origin_x)
        self.origin_y = int(origin_y)
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        # Guard against a zero/negative half-extent (divide-by-zero); floor it.
        self.half_w = max(1e-3, float(half_w))
        self.half_h = max(1e-3, float(half_h))

    def map(self, hand_x: float, hand_y: float) -> tuple[int, int]:
        """(hand_x, hand_y) metres → (px, py) absolute VIRTUAL-DESKTOP pixel,
        clamped to the desktop bounds (origin .. origin+extent-1)."""
        # Normalise to -1..+1 within the box.
        nx = (float(hand_x) - self.center_x) / self.half_w
        ny = (float(hand_y) - self.center_y) / self.half_h
        # X is NON-mirrored (the camera image is already mirror-flipped, so +x =
        # hand-right = cursor-right). Invert y (camera-up → screen-down).
        ny = -ny
        # -1..+1 → 0..1 → absolute virtual-desktop pixel (origin + offset).
        fx = (nx + 1.0) * 0.5
        fy = (ny + 1.0) * 0.5
        px = self.origin_x + int(round(fx * (self.screen_w - 1)))
        py = self.origin_y + int(round(fy * (self.screen_h - 1)))
        # Clamp to the desktop so an overshoot parks at the edge.
        px = max(self.origin_x, min(self.origin_x + self.screen_w - 1, px))
        py = max(self.origin_y, min(self.origin_y + self.screen_h - 1, py))
        return px, py


class EMA:
    """Exponential moving average for a single channel. Heavily smooths the
    jittery Kinect hand position. seed() / reset() so a fresh hand (after the
    hand left the frame) snaps to the new position instead of sweeping the
    cursor across the screen from the stale last value."""

    def __init__(self, alpha: float = AIR_MOUSE_EMA_ALPHA):
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self._value: Optional[float] = None

    def reset(self) -> None:
        self._value = None

    def update(self, x: float) -> float:
        x = float(x)
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value


class GripDebouncer:
    """Debounce the OPEN↔CLOSED hand transition for ONE hand.

    Feed the raw grip string each frame ("open" / "closed" / "lasso" /
    "unknown"); the *stable* grip only changes after the new grip has been seen
    for `frames` consecutive ticks. ROBUST-CLOSE rules:
      • "lasso" (the two-finger pointer) is treated as CLOSED — a half-curled
        fist often reads as Lasso, and the owner means it as a click.
      • "unknown" / "nottracked" never flip the stable state (they're "no new
        evidence"): a 1-frame grip dropout HOLDS the last confident grip rather
        than spuriously releasing a drag. The dead-man (hand UNTRACKED) is what
        releases a held button, not a single ambiguous frame.
    Hysteresis falls out of the consecutive-frame requirement: a fist must be
    seen `frames` times to latch CLOSED, and an open hand `frames` times to
    latch OPEN, so neither flickers.

    `stable` starts at "open" so the first real close is a clean down-edge."""

    def __init__(self, frames: int = AIR_MOUSE_GRIP_DEBOUNCE_FRAMES,
                 initial: str = "open"):
        self.frames = max(1, int(frames))
        self._stable = initial
        self._candidate: Optional[str] = None
        self._count = 0

    @property
    def stable(self) -> str:
        return self._stable

    @property
    def is_closed(self) -> bool:
        return self._stable == "closed"

    def reset(self, initial: str = "open") -> None:
        self._stable = initial
        self._candidate = None
        self._count = 0

    @staticmethod
    def _canon(raw: str) -> Optional[str]:
        """Canonicalise a raw bridge grip to "open"/"closed", or None for an
        ambiguous frame that carries no vote. Lasso → closed (a half-fist the
        owner means as a click); unknown / nottracked → None (hold)."""
        g = (raw or "unknown").lower()
        if g == "closed" or g == "lasso":
            return "closed"
        if g == "open":
            return "open"
        return None   # unknown / nottracked → no evidence this frame

    def update(self, raw: str) -> str:
        """Feed a raw grip; return the (possibly unchanged) stable grip."""
        vote = self._canon(raw)
        # Ambiguous frames hold the current stable grip and reset any in-flight
        # candidate streak (so a flicker mid-streak doesn't count toward a flip).
        if vote is None:
            self._candidate = None
            self._count = 0
            return self._stable
        if vote == self._stable:
            # Already stable here — clear any partial streak toward the other.
            self._candidate = None
            self._count = 0
            return self._stable
        # vote differs from stable: build/extend the candidate streak.
        if vote == self._candidate:
            self._count += 1
        else:
            self._candidate = vote
            self._count = 1
        if self._count >= self.frames:
            self._stable = vote
            self._candidate = None
            self._count = 0
        return self._stable


# Per-frame decision returned by AirMouseController.update().
class AirMouseDecision:
    """What the live loop should DO this frame.

      cursor:  (px, py) | None   — where to put the cursor (None = don't move)
      left:    "down" | "up" | None  — actuate the LEFT button (edge only; None
               means no change). Fired by the LEFT hand closing/opening.
      right:   "down" | "up" | None  — actuate the RIGHT button (edge only).
               Fired by the RIGHT hand closing/opening.
      overlay: "track" | "grab" | "hidden" — the reticle state to publish
               (cyan-track / gold-grab / hidden). "grab" while EITHER button held.
      hand:    "left" | "right" | None — which hand is driving the cursor (for
               the preview hand-circle), None while disengaged.
      grip:    the controlling hand's debounced stable grip ("open"/"closed") —
               drives the preview circle colour + diagnostics.
      prime:   float in [0.0, 1.0] — the PASSIVE engage-dwell PROGRESS while the
               smart pose is being HELD toward engage (0 at pose-start → 1 at
               engage); 0.0 when idle or already engaged. Published in the overlay
               state so the HUD can draw a filling priming ring.
    """
    __slots__ = ("cursor", "left", "right", "overlay", "hand", "grip", "prime")

    def __init__(self, cursor, left, right, overlay, hand, grip, prime=0.0):
        self.cursor = cursor
        self.left = left
        self.right = right
        self.overlay = overlay
        self.hand = hand
        self.grip = grip
        self.prime = max(0.0, min(1.0, float(prime)))

    @property
    def button_edges(self):
        """The (button, action) edges this frame, e.g. [("left","down")]. Order:
        left then right. Used by the live loop to actuate the real mouse."""
        out = []
        if self.left in ("down", "up"):
            out.append(("left", self.left))
        if self.right in ("down", "up"):
            out.append(("right", self.right))
        return out

    def __repr__(self):   # pragma: no cover - debug aid
        return (f"AirMouseDecision(cursor={self.cursor}, left={self.left!r}, "
                f"right={self.right!r}, overlay={self.overlay!r}, "
                f"hand={self.hand!r}, grip={self.grip!r}, prime={self.prime:.2f})")


class ArmExtension:
    """One arm's engage cues + the HEIGHT engage-hysteresis test. A thin value
    object fed the bridge's arm_extension() dict so the controller can ask "is this
    hand raised above the shoulder?" with the right (engage vs stay-engaged) bar.
    PURE — no sensor.

    The HEIGHT delta lift_m (hand_y - shoulder_ref_y; camera-up) is the PRIMARY and
    NECESSARY signal: a hand is "extended" (engaged) only while it is raised above
    the shoulder by the margin. forward_m / reach_ratio / straightness are retained
    for back-compat + the debug log but are DEMOTED to permissive secondary cues —
    with their bars at ≈0 they can never engage, hold, or veto the gate (the height
    delta does all the gating). This is the "forward-reach is broken for a desk
    user" fix: a hand RESTING on the desk reads a big forward reach yet sits far
    BELOW the shoulder (lift_m ≪ 0), so the height gate keeps it disengaged.

    BODY-RELATIVE: lift_m is hand height ABOVE the shoulder line, so engage/disengage
    is invariant to rotation, chair position, and distance from the sensor."""

    __slots__ = ("side", "forward_m", "reach_ratio", "straightness", "hand",
                 "lift_m", "shoulder_ref_y")

    def __init__(self, side: str, forward_m: Optional[float],
                 straightness: Optional[float], hand=None,
                 reach_ratio: Optional[float] = None,
                 lift_m: Optional[float] = None,
                 shoulder_ref_y: Optional[float] = None):
        self.side = side
        self.forward_m = forward_m
        self.reach_ratio = reach_ratio   # forward reach / body scale (secondary)
        self.straightness = straightness
        self.hand = hand   # (x, y, z, state) of the controlling hand, or None
        self.lift_m = lift_m            # hand_y - shoulder_ref_y (PRIMARY gate)
        self.shoulder_ref_y = shoulder_ref_y

    @classmethod
    def from_bridge(cls, ext: dict) -> "ArmExtension":
        ext = ext or {}
        return cls(ext.get("side", ""), ext.get("forward_reach_m"),
                   ext.get("straightness"), ext.get("hand"),
                   reach_ratio=ext.get("reach_ratio"),
                   lift_m=ext.get("lift_m"),
                   shoulder_ref_y=ext.get("shoulder_ref_y"))

    def is_extended(self, *, engaged: bool, thresholds: "Optional[dict]" = None,
                    up_margin: float = AIR_MOUSE_ENGAGE_UP_MARGIN_M,
                    down_margin: float = AIR_MOUSE_ENGAGE_DOWN_MARGIN_M,
                    ratio_engage: float = AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE,
                    ratio_disengage: float = AIR_MOUSE_EXTEND_REACH_RATIO_DISENGAGE,
                    fwd_engage: float = AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M,
                    fwd_disengage: float = AIR_MOUSE_EXTEND_FORWARD_DISENGAGE_M,
                    straight_engage: float = AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE,
                    straight_disengage: float = AIR_MOUSE_EXTEND_STRAIGHT_DISENGAGE
                    ) -> bool:
        """Is this hand raised enough ABOVE THE SHOULDER to (stay) engaged? The
        HEIGHT delta lift_m is the PRIMARY and NECESSARY signal, with HYSTERESIS:
        when currently DISENGAGED lift_m must clear the HIGHER up_margin (hand
        at/above the shoulder); once ENGAGED it only has to stay above the LOWER
        down_margin (~10 cm below the shoulder), and the INSTANT it drops below that
        the hand is no longer extended → DISENGAGE. down_margin < up_margin gives
        the hysteresis so a hand hovering at the line can't flap.

        A hand resting on the desk sits at WAIST level — lift_m strongly negative,
        below down_margin — so it NEVER engages, which is the whole point. The
        forward-reach / straightness cues are DEMOTED: with their bars permissive
        (≈0) they never gate, so the only thing that engages/holds is the height.

        `thresholds` (live CALIBRATED bars from _reach_thresholds(); may also carry
        up_margin / down_margin) wins over the keyword defaults when given, so the
        owner can tune the margins live without re-plumbing every caller."""
        if thresholds:
            up_margin = thresholds.get("up_margin", up_margin)
            down_margin = thresholds.get("down_margin", down_margin)
            ratio_engage = thresholds.get("ratio_engage", ratio_engage)
            ratio_disengage = thresholds.get("ratio_disengage", ratio_disengage)
            fwd_engage = thresholds.get("fwd_engage", fwd_engage)
            fwd_disengage = thresholds.get("fwd_disengage", fwd_disengage)
            straight_engage = thresholds.get("straight_engage", straight_engage)
            straight_disengage = thresholds.get("straight_disengage",
                                                straight_disengage)
        # PRIMARY + NECESSARY HEIGHT gate: the hand must be raised above the
        # shoulder by the margin. Hysteresis: a higher bar to engage, a lower bar to
        # stay engaged. A missing lift reading (shoulder/hand not measurable this
        # frame) means we CANNOT confirm a raise → NOT extended (fail safe: the
        # owner's real mouse is left alone unless we positively see a raised hand).
        if self.lift_m is None:
            return False
        lift_bar = down_margin if engaged else up_margin
        if self.lift_m < lift_bar:
            return False
        # SECONDARY (DEMOTED) forward-reach cue: permissive by default (bars ≈0) so
        # it can neither hold nor veto the gate; only blocks if a non-trivial bar is
        # configured (kept for the calibration plumbing). A relaxed-but-raised hand
        # must still engage on height alone, so a None/zero forward cue never vetoes.
        ratio_bar = ratio_disengage if engaged else ratio_engage
        if ratio_bar > 0.0 and self.reach_ratio is not None:
            if self.reach_ratio < ratio_bar:
                return False
        elif ratio_bar <= 0.0:
            fwd_bar = fwd_disengage if engaged else fwd_engage
            if fwd_bar > 0.0 and self.forward_m is not None:
                if self.forward_m < fwd_bar:
                    return False
        # SECONDARY (DEMOTED) straightness veto: permissive by default (bar 0). Only
        # rejects on the rising edge if a non-trivial bar is configured.
        if not engaged and straight_engage > 0.0 and self.straightness is not None:
            if self.straightness < straight_engage:
                return False
        return True

    def reach_score(self) -> float:
        """A scalar "how raised" used to pick the CONTROLLING hand — whichever hand
        is raised HIGHEST above the shoulder. Ranks by lift_m (height above the
        shoulder line); a missing lift sorts to the bottom. Body-relative so the
        comparison between the two hands is fair regardless of distance/rotation."""
        if self.lift_m is None:
            return float("-inf")
        return float(self.lift_m)


def extended_arms(left: "ArmExtension", right: "ArmExtension", *, engaged: bool,
                  thresholds: "Optional[dict]" = None) -> "list[ArmExtension]":
    """The arms currently EXTENDED enough to (stay) engaged — each with a usable
    hand joint and clearing its reach bar (engage hysteresis). PURE."""
    return [a for a in (left, right)
            if a is not None and a.hand is not None
            and a.is_extended(engaged=engaged, thresholds=thresholds)]


def choose_controlling_arm(left: "ArmExtension", right: "ArmExtension",
                           *, engaged: bool, thresholds: "Optional[dict]" = None,
                           current_side: "Optional[str]" = None,
                           margin: float = HAND_SWITCH_MARGIN
                           ) -> "Optional[ArmExtension]":
    """Pick which arm drives the cursor: among the EXTENDED arms (engage / stay-
    engage hysteresis), the MORE-extended (higher reach_score). Returns None when
    neither is extended (→ disengage). PURE.

    HAND-HYSTERESIS (ISSUE 3): when `current_side` (the hand already driving) is
    still among the extended candidates, it is KEPT unless the other arm leads it
    by at least `margin` — i.e. a tie / marginal lead never flips control. The
    controller adds the multi-FRAME requirement on top; this gives the per-frame
    stickiness (the holder wins ties)."""
    candidates = extended_arms(left, right, engaged=engaged, thresholds=thresholds)
    if not candidates:
        return None
    best = max(candidates, key=lambda a: a.reach_score())
    if current_side is not None:
        holder = next((a for a in candidates if a.side == current_side), None)
        if holder is not None and holder is not best:
            # The holder is still extended but isn't the top score: keep it unless
            # the challenger leads by the margin (sticky tie-break).
            if best.reach_score() - holder.reach_score() < margin:
                return holder
    return best


def _median(values: "list[float]") -> "Optional[float]":
    """Median of a list of floats, or None when empty. Pure; robust to the odd
    outlier sample the Kinect throws (better than a mean for calibration)."""
    xs = sorted(v for v in values if v is not None)
    n = len(xs)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return float(xs[mid])
    return (float(xs[mid - 1]) + float(xs[mid])) / 2.0


def compute_reach_thresholds(
        lowered_lift: "Optional[float]", raised_lift: "Optional[float]",
        lowered_straight: "Optional[float]", raised_straight: "Optional[float]",
        *, engage_fraction: float = CALIB_ENGAGE_FRACTION,
        disengage_fraction: float = CALIB_DISENGAGE_FRACTION) -> dict:
    """Fit the HEIGHT-gate margins from a captured LOWERED + RAISED hand pose.

    The cue is the BODY-RELATIVE hand HEIGHT above the shoulder (lift_m), so the
    fitted margins are POSITION-INDEPENDENT — a calibration done at one distance
    holds at any other. Each margin is placed a FRACTION of the way lowered→raised:
    the up (engage) margin ~60 % (so a raise a little short of full still engages)
    and the down (disengage) margin ~40 % (so a small sag doesn't drop it) — up
    strictly above down, giving the hysteresis. A pose pair that's
    missing/degenerate (None, or raised not clearly above lowered) FALLS BACK to
    the module margins, so a partial capture still yields a safe, usable gate. PURE
    — no sensor; the live action feeds it the captured medians.

    Returns {up_margin, down_margin, ...} (the forward/straightness bars are left
    at their permissive module defaults — they no longer gate; straightness is
    still fitted + returned for back-compat but does not affect the height gate)."""
    ef = min(max(float(engage_fraction), 0.0), 1.0)
    df = min(max(float(disengage_fraction), 0.0), 1.0)

    def _bars(lo, hi, default_engage, default_disengage, min_span):
        # Need both ends AND a real span (raised clearly above lowered) to fit;
        # otherwise keep the defaults rather than emit a nonsense (inverted/tiny)
        # margin from a flubbed capture.
        if lo is None or hi is None or (hi - lo) < min_span:
            return default_engage, default_disengage
        span = hi - lo
        return lo + ef * span, lo + df * span

    # HEIGHT margins need a clear vertical span (~15 cm lower→raise) to fit.
    up_m, down_m = _bars(lowered_lift, raised_lift,
                         AIR_MOUSE_ENGAGE_UP_MARGIN_M,
                         AIR_MOUSE_ENGAGE_DOWN_MARGIN_M, 0.15)
    str_e, str_d = _bars(lowered_straight, raised_straight,
                         AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE,
                         AIR_MOUSE_EXTEND_STRAIGHT_DISENGAGE, 0.05)
    return {"up_margin": up_m, "down_margin": down_m,
            "straight_engage": str_e, "straight_disengage": str_d,
            # Forward/ratio bars stay permissive (height gate is primary).
            "ratio_engage": AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE,
            "ratio_disengage": AIR_MOUSE_EXTEND_REACH_RATIO_DISENGAGE,
            "fwd_engage": AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M,
            "fwd_disengage": AIR_MOUSE_EXTEND_FORWARD_DISENGAGE_M}


class AirMouseController:
    """The pure per-frame brain. Holds the smoothing, the PER-HAND grip
    debouncers + button state, and the HEIGHT engage state, turning each
    (left_ext, right_ext, left_grip, right_grip, tracked, real_input_recent) sample
    into an AirMouseDecision. NO I/O — the live loop applies the decision (move
    cursor, press buttons, publish overlay state). Re-buildable cheaply; reset() on
    disable / hand-loss.

    HEIGHT ENGAGE GATE (RAISE a hand ABOVE THE SHOULDER to drive the cursor):
      The controller is ENGAGED only while at least one hand is raised above the
      shoulder line (its lift_m — the PRIMARY, NECESSARY cue — clears the engage
      margin; forward-reach/straightness are demoted and never gate, see
      ArmExtension) AND a body+hand are tracked AND the owner has NOT just touched
      their real mouse/keyboard. Engage/stay-engage hysteresis (a higher up-margin
      to engage, a lower down-margin to stay) stops it flapping at the line. The
      cursor follows the HIGHEST-raised hand (sticky hand-hysteresis). While
      DISENGAGED — both hands below the line, untracked beyond the ~0.3 s grace, OR
      AUTO-YIELDING to recent real input — the decision carries cursor=None (so the
      live loop calls NO SetCursorPos and the PHYSICAL mouse is free), releases any
      held button, and hides the overlay. Re-raising a hand re-engages, EMA re-snaps.

    PER-HAND clicks (HAND-SPECIFIC), evaluated EVERY engaged frame for BOTH hands
    regardless of which one drives the cursor:
      • LEFT hand  OPEN→CLOSED → emit LEFT  "down"; CLOSED→OPEN → LEFT  "up".
      • RIGHT hand OPEN→CLOSED → emit RIGHT "down"; CLOSED→OPEN → RIGHT "up".
      A held-closed hand keeps its button down while the cursor moves (a drag);
      a quick close→open with no move is a click. The overlay shows "grab" while
      EITHER button is held."""

    def __init__(self, reach: ReachBox,
                 alpha: float = AIR_MOUSE_EMA_ALPHA,
                 debounce_frames: int = AIR_MOUSE_GRIP_DEBOUNCE_FRAMES,
                 grace_sec: float = AIR_MOUSE_DISENGAGE_GRACE_SEC,
                 clock=time.monotonic,
                 switch_margin: float = HAND_SWITCH_MARGIN,
                 switch_frames: int = HAND_SWITCH_FRAMES,
                 engage_debounce_frames: int = AIR_MOUSE_ENGAGE_DEBOUNCE_FRAMES,
                 untracked_ceiling_sec: float = AIR_MOUSE_UNTRACKED_CEILING_SEC,
                 retrack_frames: int = AIR_MOUSE_RETRACK_FRAMES,
                 dwell_sec: "Optional[float]" = None,
                 still_m: "Optional[float]" = None,
                 arm_debounce_sec: "Optional[float]" = None,
                 fist_release_sec: "Optional[float]" = None,
                 require_open_palm: "Optional[bool]" = None,
                 facing_max_deg: "Optional[float]" = None,
                 arm_relaxes_gate: "Optional[bool]" = None,
                 fist_releases: "Optional[bool]" = None):
        self.reach = reach
        self._ema_x = EMA(alpha)
        self._ema_y = EMA(alpha)
        # One debouncer + one button-down flag PER HAND so each hand drives its
        # own (left/right) mouse button independently.
        self._grip_left = GripDebouncer(debounce_frames, initial="open")
        self._grip_right = GripDebouncer(debounce_frames, initial="open")
        self._left_down = False
        self._right_down = False
        # Engage gate state.
        self._grace_sec = max(0.0, float(grace_sec))
        self._clock = clock          # injectable monotonic clock (tests)
        self._engaged = False        # is the air-mouse currently driving the cursor
        self._hand: Optional[str] = None   # which hand is driving ("left"/"right")
        self._last_engaged_at = 0.0  # clock() of the last EXTENDED+tracked frame
        # ENGAGE DEBOUNCE (FIX 3): a raised hand must persist above the engage line
        # for this many CONSECUTIVE frames before the cursor is actually taken, so a
        # 1-frame Kinect height spike can't grab it. DISENGAGE stays instant.
        self._engage_debounce_frames = max(1, int(engage_debounce_frames))
        self._engage_streak = 0      # consecutive frames a hand has cleared the line
        # Controlling-hand HYSTERESIS (ISSUE 3): the challenger must out-reach the
        # holder by `switch_margin` for `switch_frames` consecutive frames before
        # the cursor switches hands, so two raised hands can't thrash the cursor.
        self._switch_margin = max(0.0, float(switch_margin))
        self._switch_frames = max(1, int(switch_frames))
        self._challenge_side: Optional[str] = None   # the side currently challenging
        self._challenge_count = 0                    # its consecutive-lead streak
        # BODY-ID PIN (FILTER 6): the id of the body that took control on the
        # engage edge. While engaged the controller drives ONLY this body; if the
        # nearest body's id CHANGES (a closer 2nd person) it is treated as a
        # tracking-loss (dead-man release + EMA reset) rather than a seamless
        # retarget, so a passer-by can't steal the cursor mid-drag.
        self._locked_body_id = None
        # GRACE CAP (FILTER 7): cumulative UNTRACKED time this engagement + the
        # clock of the last frame (to integrate the gap), and a consecutive-Tracked
        # streak that re-arms (zeroes) the accumulator on a solid re-acquisition.
        self._untracked_ceiling_sec = max(0.0, float(untracked_ceiling_sec))
        self._retrack_frames = max(1, int(retrack_frames))
        self._untracked_accum = 0.0      # summed untracked seconds this engagement
        self._retrack_streak = 0         # consecutive fully-Tracked frames
        self._last_frame_at: Optional[float] = None   # clock() of the prior frame

        # ── SMART-ENGAGE (2026-07): the PASSIVE strict smart-pose gate + the ARMED
        #    relaxed gate + the fist-release timer. Knobs default to the live
        #    core.config values (resolved once at construction; None → default), so
        #    a plainly-constructed controller uses the shipped policy and a test can
        #    pin any of them. The pure engage_decision() does the actual gate math;
        #    the controller only measures the timers + stillness it needs. ─────────
        self._dwell_sec = (float(dwell_sec) if dwell_sec is not None
                           else _cfg_float("AIR_MOUSE_ENGAGE_DWELL_SEC",
                                           AIR_MOUSE_ENGAGE_DWELL_SEC))
        self._still_m = (float(still_m) if still_m is not None
                         else _cfg_float("AIR_MOUSE_ENGAGE_STILL_M",
                                         AIR_MOUSE_ENGAGE_STILL_M))
        self._arm_debounce_sec = (
            float(arm_debounce_sec) if arm_debounce_sec is not None
            else _cfg_float("AIR_MOUSE_ARM_ENGAGE_DEBOUNCE_SEC",
                            AIR_MOUSE_ARM_ENGAGE_DEBOUNCE_SEC))
        self._fist_release_sec = (
            float(fist_release_sec) if fist_release_sec is not None
            else _cfg_float("AIR_MOUSE_FIST_RELEASE_SEC",
                            AIR_MOUSE_FIST_RELEASE_SEC))
        self._require_open_palm = (
            bool(require_open_palm) if require_open_palm is not None
            else bool(_cfg_flag("AIR_MOUSE_REQUIRE_OPEN_PALM",
                                AIR_MOUSE_REQUIRE_OPEN_PALM)))
        self._facing_max_deg = (
            float(facing_max_deg) if facing_max_deg is not None
            else _cfg_float("AIR_MOUSE_FACING_MAX_DEG", AIR_MOUSE_FACING_MAX_DEG))
        self._arm_relaxes_gate = (
            bool(arm_relaxes_gate) if arm_relaxes_gate is not None
            else bool(_cfg_flag("AIR_MOUSE_ARM_RELAXES_GATE",
                                AIR_MOUSE_ARM_RELAXES_GATE)))
        self._fist_releases = (
            bool(fist_releases) if fist_releases is not None
            else bool(_cfg_flag("AIR_MOUSE_FIST_RELEASES", AIR_MOUSE_FIST_RELEASES)))
        # Priming state (the smart-pose dwell). _pose_started_at is the clock() when
        # the CURRENT continuous valid pose began (None when no pose held); the dwell
        # elapsed = now - that. _prime is the last progress published (0..1).
        self._pose_started_at: Optional[float] = None
        self._arm_pose_started_at: Optional[float] = None  # ARMED height-hold start
        self._prime = 0.0
        # Stillness: the hand position at pose-start + the max travel seen since, so
        # a hand that drifts past the still bar during the dwell breaks priming.
        self._pose_anchor_xy: Optional[tuple] = None
        self._pose_travel = 0.0
        # FIST-RELEASE timer: clock() when the controlling hand's stable grip first
        # went CLOSED while engaged (None when open); a sustained close past
        # _fist_release_sec force-disengages when _fist_releases is on.
        self._fist_closed_at: Optional[float] = None

    def reset(self) -> None:
        """Drop all smoothing + grip + engage state. Used by the dead-man and on
        disable so the next reach starts clean (no cursor sweep from a stale
        value, no phantom button edge, freshly DISENGAGED)."""
        self._ema_x.reset()
        self._ema_y.reset()
        self._grip_left.reset(initial="open")
        self._grip_right.reset(initial="open")
        # NB: this does NOT itself emit a button-up — the caller (dead-man) is
        # responsible for releasing held buttons. We only clear our own view.
        self._left_down = False
        self._right_down = False
        self._engaged = False
        self._hand = None
        self._last_engaged_at = 0.0
        self._engage_streak = 0
        self._challenge_side = None
        self._challenge_count = 0
        self._locked_body_id = None
        self._untracked_accum = 0.0
        self._retrack_streak = 0
        self._last_frame_at = None
        # SMART-ENGAGE: drop the priming dwell + stillness + fist-release timers so
        # the next acquisition starts a fresh pose/dwell.
        self._pose_started_at = None
        self._arm_pose_started_at = None
        self._prime = 0.0
        self._pose_anchor_xy = None
        self._pose_travel = 0.0
        self._fist_closed_at = None

    @property
    def button_is_down(self) -> bool:
        """True while EITHER mouse button is held (left or right)."""
        return self._left_down or self._right_down

    @property
    def left_is_down(self) -> bool:
        return self._left_down

    @property
    def right_is_down(self) -> bool:
        return self._right_down

    @property
    def engaged(self) -> bool:
        """True while the air-mouse is actively driving the cursor (an arm
        extended + tracked). False while disengaged (arm relaxed / lost) — in
        which state update() returns cursor=None so the physical mouse is free."""
        return self._engaged

    @property
    def hand(self) -> Optional[str]:
        """Which hand is driving the cursor ("left"/"right"), or None while
        disengaged. Read by the preview to draw the circle on the right hand."""
        return self._hand

    def release_decision(self) -> AirMouseDecision:
        """The DEAD-MAN / disengaged decision: if a button was held, command it
        UP (per hand); hide the overlay; clear smoothing + grips + engage so the
        next acquisition snaps. cursor=None so the live loop issues NO
        SetCursorPos and the physical mouse is free. Idempotent — once released,
        repeated calls just keep the overlay hidden with no button edge."""
        left = "up" if self._left_down else None
        right = "up" if self._right_down else None
        self._left_down = False
        self._right_down = False
        self._engaged = False
        self._hand = None
        self._engage_streak = 0
        self._challenge_side = None
        self._challenge_count = 0
        # FILTER 6/7: drop the body-id pin + the grace-cap accumulators so the next
        # engagement re-locks a body and starts a fresh untracked budget.
        self._locked_body_id = None
        self._untracked_accum = 0.0
        self._retrack_streak = 0
        self._last_frame_at = None
        # SMART-ENGAGE: a full release also drops the priming dwell + fist timer so
        # the next acquisition re-primes from scratch (no stale progress ring).
        self._pose_started_at = None
        self._arm_pose_started_at = None
        self._prime = 0.0
        self._pose_anchor_xy = None
        self._pose_travel = 0.0
        self._fist_closed_at = None
        self._ema_x.reset()
        self._ema_y.reset()
        self._grip_left.reset(initial="open")
        self._grip_right.reset(initial="open")
        return AirMouseDecision(cursor=None, left=left, right=right,
                                overlay="hidden", hand=None, grip="open", prime=0.0)

    def _hold_off_decision(self, prime: float = 0.0) -> AirMouseDecision:
        """The ENGAGE hold-off 'not yet' decision: a hand is in a valid engage pose
        but hasn't held long enough (the ARMED short debounce or the PASSIVE dwell)
        to take the cursor. Like a disengaged frame — cursor=None (NO SetCursorPos,
        the physical mouse is free) and the overlay hidden — but WITHOUT touching the
        grip debouncers / EMA (we haven't engaged, so there's nothing to release and
        no edge to emit). `prime` (0..1) is the PASSIVE dwell progress so the HUD can
        draw a filling priming ring while the pose is being held."""
        return AirMouseDecision(cursor=None, left=None, right=None,
                                overlay="hidden", hand=None, grip="open",
                                prime=max(0.0, min(1.0, float(prime))))

    @staticmethod
    def _grip_if_tracked(ext, raw_grip: str) -> str:
        """The grip to feed a hand's debouncer, gated on the hand's JOINT being
        sensor-TRACKED (FILTER 4). Returns the raw grip ONLY when `ext` has a
        well-tracked hand joint (TrackingState >= 2, finite, non-zero); otherwise
        "unknown" — which GripDebouncer._canon treats as 'no evidence', so an
        untracked / inferred / missing hand can never press a button (its
        debouncer just holds). PURE."""
        if ext is None or ext.hand is None or not joint_well_tracked(ext.hand):
            return "unknown"
        return raw_grip

    def _hand_button_edge(self, debouncer: GripDebouncer, raw_grip: str,
                          down_attr: str) -> Optional[str]:
        """Advance ONE hand's debouncer and emit its button edge. `down_attr` is
        the instance attribute name holding that hand's button-down flag
        ("_left_down"/"_right_down"). Returns "down"/"up"/None."""
        stable = debouncer.update(raw_grip)
        return self._button_edge_from_stable(stable, down_attr)

    def _button_edge_from_stable(self, stable: str,
                                 down_attr: str) -> Optional[str]:
        """Emit a hand's button edge from an ALREADY-ADVANCED stable grip (no
        debouncer mutation). Used when the debouncer was advanced earlier this frame
        (during priming, to read the pose grip) so the engaged frame doesn't
        double-advance it. Returns "down"/"up"/None."""
        held = getattr(self, down_attr)
        want_down = (stable == "closed")
        if want_down and not held:
            setattr(self, down_attr, True)
            return "down"
        if not want_down and held:
            setattr(self, down_attr, False)
            return "up"
        return None

    def _select_controlling_arm(self, left_ext, right_ext,
                                thresholds: "Optional[dict]"):
        """Pick the cursor-driving arm with HAND-HYSTERESIS (ISSUE 3). Keeps the
        current controlling hand unless the OTHER arm out-reaches it by the margin
        for `switch_frames` consecutive frames; a brief or marginal lead never
        flips control. Updates the challenger streak as a side effect and returns
        the chosen ArmExtension (or None when neither arm is extended)."""
        candidates = extended_arms(left_ext, right_ext, engaged=self._engaged,
                                   thresholds=thresholds)
        if not candidates:
            self._challenge_side = None
            self._challenge_count = 0
            return None
        best = max(candidates, key=lambda a: a.reach_score())
        # Is the hand currently driving still a live candidate?
        holder = next((a for a in candidates if a.side == self._hand), None)
        if holder is None:
            # The driving hand relaxed/left (or we weren't engaged): take the most-
            # extended arm outright and reset any challenge.
            self._challenge_side = None
            self._challenge_count = 0
            return best
        if best is holder:
            # The holder is still the most extended — no challenge in progress.
            self._challenge_side = None
            self._challenge_count = 0
            return holder
        # A DIFFERENT arm out-scores the holder. Require a sustained, clear lead.
        lead = best.reach_score() - holder.reach_score()
        if lead >= self._switch_margin:
            if self._challenge_side == best.side:
                self._challenge_count += 1
            else:
                self._challenge_side = best.side
                self._challenge_count = 1
            if self._challenge_count >= self._switch_frames:
                # Sustained clear lead → switch hands.
                self._challenge_side = None
                self._challenge_count = 0
                return best
        else:
            # Lead too small this frame → challenge resets (no thrash).
            self._challenge_side = None
            self._challenge_count = 0
        return holder

    @staticmethod
    def _both_hands_raised(left_ext, right_ext,
                           thresholds: "Optional[dict]") -> bool:
        """True when the controller LOCALLY sees BOTH hands raised above the strict
        ENGAGE line (FILTER 8 pre-empt). Mirrors kinect_two_hand._both_hands_engaged
        exactly — both arms present, both with a sensor-tracked hand joint, both
        clearing the engage (not the looser stay-engaged) bar — so the single-hand
        cursor stands down the instant the owner raises a second hand, without
        waiting for the cross-process two-hand heartbeat. PURE; NEVER raises."""
        try:
            if left_ext is None or right_ext is None:
                return False
            if left_ext.hand is None or right_ext.hand is None:
                return False
            if not (joint_well_tracked(left_ext.hand)
                    and joint_well_tracked(right_ext.hand)):
                return False
            return bool(
                left_ext.is_extended(engaged=False, thresholds=thresholds)
                and right_ext.is_extended(engaged=False, thresholds=thresholds))
        except Exception:
            return False

    def update(self, left_ext, right_ext, left_grip: str, right_grip: str,
               tracked: bool, thresholds: "Optional[dict]" = None,
               real_input_recent: bool = False, body_id=None,
               facing_deg=None, armed: "Optional[bool]" = None,
               per_app_disabled: bool = False
               ) -> AirMouseDecision:
        """Advance one frame.

        left_ext / right_ext: the per-arm ArmExtension (or None when that arm's
            joints couldn't be read) describing the hand HEIGHT above the shoulder
            (lift_m, the primary gate) + the demoted forward/straightness cues.
        left_grip / right_grip: the raw bridge grips for each hand
            ("open"/"closed"/"lasso"/"unknown").
        tracked: True when the bridge reported a tracked body this frame.
        thresholds: the live engage-gate bars from _reach_thresholds() (the owner's
            margins / calibration, or the defaults). None → the module defaults.
        real_input_recent: True when the owner touched their REAL mouse/keyboard
            within the yield window — the air-mouse YIELDS: force-disengage, release
            any held button, and stay SUPPRESSED (cannot re-engage) this frame.
        body_id: the id of the NEAREST body this frame (or None). The controller
            PINS the body that took control (FILTER 6): once engaged it drives ONLY
            that id; if the nearest-body id CHANGES (a closer 2nd person) it is a
            tracking-loss (release + EMA reset), never a seamless retarget.
        facing_deg: the body's yaw from square (degrees; None when unavailable) —
            the PASSIVE gate requires |facing_deg| within the facing bar to engage;
            None (bridge didn't provide it) is treated as facing OK.
        armed: the ARMED flag (owner explicitly asked for the cursor → the relaxed
            gate). None → read the module _air_mouse_armed flag live.
        per_app_disabled: True when the FOREGROUND app matches a disabled-app hint —
            the air-mouse STANDS DOWN (force-disengage + no engage) this frame.

        DISENGAGES (returns cursor=None — no SetCursorPos — and releases any held
        button) when real input is recent (AUTO-YIELD), when the foreground app is
        on the disabled list (per_app_disabled), when the body/hand is NOT tracked,
        when the controlling BODY-ID changed under it, when BOTH hands are raised
        (two-hand mode pre-empt), when a SUSTAINED closed fist while engaged fires
        the fist-release, or when the smart-engage gate says not-engaged. A brief
        tracking dropout while ENGAGED is tolerated for up to the grace window
        (button held, cursor parked) before the full release — but cumulative
        untracked time is capped per engagement so a FLICKERING body can't renew the
        grace forever (FILTER 7).

        ENGAGE is HYBRID (the pure engage_decision() decides): ARMED relaxes to a
        height-only short-hold gate (responsive — the owner asked for it); PASSIVE
        requires the full smart pose (raised + OPEN palm + FACING + STILL) HELD for
        the dwell, publishing the dwell PROGRESS as `prime` for the HUD ring. The
        cursor follows the highest-raised hand with HAND-HYSTERESIS; per-hand
        close→click is evaluated for BOTH hands every engaged frame."""
        if armed is None:
            armed = air_mouse_is_armed()
        # Integrate the inter-frame gap for the FILTER 7 untracked-time cap, then
        # remember this frame's time for the next call. Done first so every return
        # path below has an up-to-date clock baseline.
        now = self._clock()
        dt = 0.0
        if self._last_frame_at is not None:
            dt = max(0.0, now - self._last_frame_at)
        self._last_frame_at = now
        # ── AUTO-YIELD: the owner just touched their real mouse/keyboard. Release
        #    everything and stay SUPPRESSED this frame — the real input always wins,
        #    immediately, and the air-mouse cannot re-engage until the window
        #    elapses. This is checked FIRST so it overrides a raised hand. ────────
        if real_input_recent:
            return self.release_decision()
        # ── PER-APP DISABLE: the foreground app is on the disabled-app list (a
        #    fullscreen game/video, say). STAND DOWN — release + no engage, exactly
        #    like a yield — so a stray cursor grab can't disrupt it. Defensive: the
        #    caller already treats any win32 failure as "not disabled", so this only
        #    fires on a positive match. Checked before the pose gate so it overrides
        #    a valid pose. ─────────────────────────────────────────────────────────
        if per_app_disabled:
            return self.release_decision()
        # ── tracking-loss path, with a short grace so a 1-frame dropout doesn't
        #    disengage + re-snap (a held drag must survive a flicker). ──────────
        if not tracked:
            # FILTER 7: sum the untracked gap. A run of CONSECUTIVE Tracked frames
            # (below) re-arms (zeroes) this; an untracked frame breaks the streak.
            self._retrack_streak = 0
            if self._engaged:
                self._untracked_accum += dt
            within_grace = (self._engaged and self._grace_sec > 0.0
                            and (now - self._last_engaged_at) <= self._grace_sec)
            within_ceiling = (self._untracked_ceiling_sec <= 0.0
                              or self._untracked_accum <= self._untracked_ceiling_sec)
            if within_grace and within_ceiling:
                # Brief dropout: hold. No sample → no cursor motion; keep any held
                # button and the current overlay. Do NOT refresh the engage clock,
                # so a sustained dropout still ages out into a full release. The
                # cumulative cap above force-releases a body that keeps flickering
                # back for a single frame to renew the per-dropout grace.
                overlay = "grab" if self.button_is_down else "track"
                return AirMouseDecision(cursor=None, left=None, right=None,
                                        overlay=overlay, hand=self._hand,
                                        grip=self._controlling_grip())
            return self.release_decision()

        # ── BODY-ID PIN (FILTER 6): while engaged we drive ONLY the body that took
        #    control. If the nearest-body id changed under us (a closer 2nd person
        #    stole the "nearest" slot), treat it as tracking-loss — full dead-man
        #    release (which clears the EMA + pin) so the cursor does NOT seamlessly
        #    jump onto the interloper mid-drag. Re-engage will re-lock the new body
        #    cleanly. body_id None (caller didn't supply one) disables the pin. ───
        if (self._engaged and self._locked_body_id is not None
                and body_id is not None and body_id != self._locked_body_id):
            return self.release_decision()

        # FILTER 7: a fully-Tracked frame — advance the re-acquisition streak and,
        # once solidly back (≥ retrack_frames in a row), zero the untracked budget
        # so a long healthy session never trips the cap.
        self._retrack_streak += 1
        if self._retrack_streak >= self._retrack_frames:
            self._untracked_accum = 0.0

        # ── TWO-HAND PRE-EMPT (FILTER 8): if the controller LOCALLY observes BOTH
        #    hands raised above the engage line, STAND DOWN immediately rather than
        #    waiting for the cross-process two-hand heartbeat (which lags a frame).
        #    This closes the 1-frame entry twitch where the single-hand cursor would
        #    grab one hand on the very frame the owner raised the second. Computed
        #    with the strict ENGAGE bar so it matches the two-hand poller's gate. ─
        if self._both_hands_raised(left_ext, right_ext, thresholds):
            return self.release_decision()

        # ── HEIGHT gate: pick the controlling hand (highest above the shoulder,
        #    engage hysteresis + sticky hand-hysteresis). No hand raised above the
        #    line → disengage (fail SAFE, the real mouse is left alone). ─────────
        arm = self._select_controlling_arm(left_ext, right_ext, thresholds)
        if arm is None:
            return self.release_decision()

        # ── ADVANCE the grip debouncers ONCE per tracked frame (FILTER 4: an
        #    untracked/inferred hand is fed "unknown" so it can never latch a click).
        #    Done here — before the gate — so the PASSIVE open-palm test can read the
        #    controlling hand's DEBOUNCED stable grip (a 1-frame flicker doesn't
        #    matter), and so we never double-advance on the engage frame. Button
        #    EDGES are emitted only when ENGAGED (below), from these stable grips. ──
        left_stable = self._grip_left.update(
            self._grip_if_tracked(left_ext, left_grip))
        right_stable = self._grip_right.update(
            self._grip_if_tracked(right_ext, right_grip))
        ctrl_stable = left_stable if arm.side == "left" else right_stable

        # ── HYBRID SMART-ENGAGE GATE (the pure engage_decision decides). While
        #    DISENGAGED we measure the PASSIVE dwell + stillness (or the ARMED short
        #    hold) that engage_decision needs; once engaged, height alone holds it.
        relaxed = bool(armed and self._arm_relaxes_gate)
        hand = arm.hand
        hand_xy = (float(hand[0]), float(hand[1]), float(hand[2])
                   if len(hand) > 2 else 0.0)
        if not self._engaged:
            if relaxed:
                # ARMED (relaxed): a short height-only hold. Track the hold start.
                if self._arm_pose_started_at is None:
                    self._arm_pose_started_at = now
                arm_elapsed = now - self._arm_pose_started_at
                dwell_elapsed = 0.0
                hand_still = True
            else:
                # PASSIVE: the smart pose must be HELD. Anchor the pose on its first
                # frame; each frame accumulate hand TRAVEL from the anchor and hold
                # STILL only while the summed travel is under the still bar.
                if self._pose_started_at is None:
                    self._pose_started_at = now
                    self._pose_anchor_xy = hand_xy
                    self._pose_travel = 0.0
                else:
                    prev = self._pose_anchor_xy or hand_xy
                    step = ((hand_xy[0] - prev[0]) ** 2
                            + (hand_xy[1] - prev[1]) ** 2
                            + (hand_xy[2] - prev[2]) ** 2) ** 0.5
                    self._pose_travel += step
                    self._pose_anchor_xy = hand_xy
                arm_elapsed = 0.0
                dwell_elapsed = now - self._pose_started_at
                hand_still = self._pose_travel <= self._still_m
        else:
            arm_elapsed = 0.0
            dwell_elapsed = 0.0
            hand_still = True

        # The lift already passed _select_controlling_arm (that IS the height gate
        # with hysteresis), so lift_ok is True here; the pure gate adds pose/dwell.
        # LEGACY FRAME DEBOUNCE bridge: a controller built the old way (with
        # engage_debounce_frames) — or a test that never advances a clock — engages
        # after that many valid-pose frames even if the wall-clock dwell hasn't
        # elapsed. Credit each held frame a slice of the dwell so both the
        # frame-count policy and the time policy agree; the larger wins.
        if not self._engaged and not relaxed:
            self._engage_streak += 1
            frame_credit = (self._dwell_sec
                            * (self._engage_streak / self._engage_debounce_frames)
                            if self._engage_debounce_frames > 0 else self._dwell_sec)
            dwell_elapsed = max(dwell_elapsed, frame_credit)

        verdict = engage_decision(
            lift_ok=True, currently_engaged=self._engaged, armed=bool(armed),
            grip=ctrl_stable, facing_deg=facing_deg, hand_still=hand_still,
            dwell_elapsed=dwell_elapsed, arm_debounce_elapsed=arm_elapsed,
            require_open_palm=self._require_open_palm,
            facing_max_deg=self._facing_max_deg, dwell_sec=self._dwell_sec,
            arm_debounce_sec=self._arm_debounce_sec,
            arm_relaxes_gate=self._arm_relaxes_gate)

        if not verdict.engaged:
            # Not (yet) engaged this frame. Three cases:
            #   1) PASSIVE priming — a valid pose filling toward the dwell: hold off
            #      and publish the dwell PROGRESS for the HUD ring.
            #   2) ARMED valid-hold — lift is up (arm selected) and the relaxed gate
            #      is just waiting out its SHORT debounce: hold off WITHOUT resetting
            #      the arm timer, so the next frame can cross it.
            #   3) INVALID pose (wrong grip / not facing / moving, or passive with no
            #      hold started): reset the dwell + arm timers so a fresh hold must
            #      start, and stand down (releasing if we had been engaged).
            if verdict.priming:
                self._prime = verdict.prime
                return self._hold_off_decision(prime=verdict.prime)
            if relaxed and not self._engaged:
                # ARMED, lift up, waiting for the short debounce — keep the timer.
                self._prime = 0.0
                self._pose_started_at = None
                self._pose_anchor_xy = None
                self._pose_travel = 0.0
                return self._hold_off_decision(prime=0.0)
            # Invalid pose → drop all priming state (a new hold restarts the dwell).
            self._pose_started_at = None
            self._arm_pose_started_at = None
            self._pose_anchor_xy = None
            self._pose_travel = 0.0
            self._prime = 0.0
            if self._engaged:
                # Was engaged but the (armed) gate dropped lift → full release.
                return self.release_decision()
            return self._hold_off_decision(prime=0.0)

        # ── ENGAGED. On the rising edge (was disengaged) snap the smoothing to
        #    the new hand position so the cursor doesn't sweep from a stale value,
        #    and LOCK the controlling body's id (FILTER 6) so a later id change is
        #    treated as tracking-loss rather than a silent retarget.
        if not self._engaged:
            self._ema_x.reset()
            self._ema_y.reset()
            self._locked_body_id = body_id
            self._untracked_accum = 0.0   # fresh untracked budget for this grab
            self._fist_closed_at = None   # fresh fist-release timer this grab
        self._engaged = True
        self._engage_streak = 0       # met the bar; clear so a later re-engage re-debounces
        self._prime = 0.0             # engaged → nothing to prime
        self._pose_started_at = None
        self._arm_pose_started_at = None
        self._hand = arm.side
        self._last_engaged_at = now

        # ── FIST-RELEASE (AIR_MOUSE_FIST_RELEASES): a SUSTAINED closed fist on the
        #    controlling hand while engaged is an explicit "let go" — force-disengage
        #    without lowering the hand. Timed (AIR_MOUSE_FIST_RELEASE_SEC) so a normal
        #    click / short drag never trips it. Uses the DEBOUNCED stable grip. ──────
        if self._fist_releases:
            if ctrl_stable == "closed":
                if self._fist_closed_at is None:
                    self._fist_closed_at = now
                elif (now - self._fist_closed_at) >= self._fist_release_sec:
                    # Held a fist long enough → release everything, stand down.
                    return self.release_decision()
            else:
                self._fist_closed_at = None

        # Smooth the controlling hand's position, then map to a pixel.
        # (REVERTED 2026-06-09: the v1.73.0 body-relative/absolute "tablet" remap was
        # jittery + unwanted — back to the plain fixed-reach-box map(sx, sy).)
        sx = self._ema_x.update(float(hand[0]))
        sy = self._ema_y.update(float(hand[1]))
        cursor = self.reach.map(sx, sy)

        # Per-hand clicks: evaluate BOTH hands every engaged frame so either hand
        # can click regardless of which drives the cursor. LEFT hand → LEFT
        # button, RIGHT hand → RIGHT button. The debouncers were ALREADY advanced
        # this frame (above), so emit edges from their stable grips WITHOUT
        # re-advancing (no double-advance).
        left_edge = self._button_edge_from_stable(left_stable, "_left_down")
        right_edge = self._button_edge_from_stable(right_stable, "_right_down")

        overlay = "grab" if self.button_is_down else "track"
        return AirMouseDecision(cursor=cursor, left=left_edge, right=right_edge,
                                overlay=overlay, hand=arm.side,
                                grip=self._controlling_grip(), prime=0.0)

    def _controlling_grip(self) -> str:
        """The stable grip of whichever hand is driving the cursor (for the
        preview circle colour). Falls back to "open" when disengaged."""
        if self._hand == "left":
            return self._grip_left.stable
        if self._hand == "right":
            return self._grip_right.stable
        return "open"


def overlay_color_for(overlay_state: str) -> str:
    """Map an overlay state to the reticle's accent colour name. "grab" →
    "gold" (the locked state), everything else → "cyan" (the tracking state).
    Pure helper shared by the live publisher and the unit test so the colour
    contract is asserted against the same source the overlay reads."""
    return "gold" if overlay_state == "grab" else "cyan"


# ══════════════════════════════════════════════════════════════════════════
#  PREVIEW FEEDBACK: hand-circle colour + thread-safe air-mouse state (B2)
# ══════════════════════════════════════════════════════════════════════════
# The HUD's Kinect skeleton preview (bobert_companion._compose_kinect_preview)
# draws a translucent circle around the controlling (extended) hand joint,
# coloured by the LIVE air-mouse state so the owner SEES when the cursor is active
# / clicking:
#   • ENGAGED + open hand   → BLUE   (cursor active, tracking)
#   • ENGAGED + closed hand → ORANGE (click / drag held)
#   • disengaged / off      → faint GREY idle (or no circle)
# The colour logic is a PURE helper (no cv2 / sensor) so the geometry module and
# the unit test assert the exact same contract; the preview just calls it.

# BGR triples (OpenCV order) for the three hand-circle states. Stark, saturated
# colours that read on the small downscaled preview tile.
HAND_CIRCLE_COLOR_ENGAGED = (255, 160, 32)   # BGR ≈ bright BLUE  (#20a0ff)
HAND_CIRCLE_COLOR_CLOSED  = (32, 170, 255)   # BGR ≈ ORANGE/amber (#ffaa20)
HAND_CIRCLE_COLOR_IDLE    = (150, 150, 150)  # dim GREY (disengaged idle hint)


def hand_circle_color_for(engaged: bool, grip: str) -> "tuple[int, int, int] | None":
    """The hand-circle BGR colour for the preview, by air-mouse state. PURE +
    hardware-free so the preview and the test share one source of truth.

      • engaged + grip "closed"        → ORANGE  (click / drag active)
      • engaged + any other grip       → BLUE    (cursor engaged, tracking)
      • not engaged                    → GREY    (faint idle hint)

    Returns a (B, G, R) tuple. The caller decides idle→draw-faint-or-skip; we
    return the idle grey so the colour mapping itself stays total + testable."""
    if engaged:
        if (grip or "").lower() == "closed":
            return HAND_CIRCLE_COLOR_CLOSED
        return HAND_CIRCLE_COLOR_ENGAGED
    return HAND_CIRCLE_COLOR_IDLE


# Thread-safe snapshot of the LIVE air-mouse engage state + which hand + grip,
# written by _poll_once each tick and read by the HUD preview compositor (a
# DIFFERENT thread — the face-tracking loop). A tiny lock keeps the
# (engaged, hand, grip, ts) tuple consistent. The preview reads it to decide the
# hand-circle colour + which hand to draw it on; stale reads paint the last state.
_air_mouse_state_lock = threading.Lock()
_air_mouse_state: dict = {"engaged": False, "hand": None, "grip": "open",
                          "prime": 0.0, "armed": False, "ts": 0.0}


def _set_air_mouse_state(engaged: bool, grip: str,
                         hand: "str | None" = None,
                         prime: float = 0.0) -> None:
    """Publish the live engage state + which hand + grip + PRIME (the passive
    engage-dwell progress, 0..1) + armed flag for the preview (thread-safe)."""
    with _air_mouse_state_lock:
        _air_mouse_state["engaged"] = bool(engaged)
        _air_mouse_state["hand"] = hand
        _air_mouse_state["grip"] = (grip or "open")
        _air_mouse_state["prime"] = max(0.0, min(1.0, float(prime)))
        _air_mouse_state["armed"] = air_mouse_is_armed()
        _air_mouse_state["ts"] = time.time()


def get_air_mouse_state() -> dict:
    """Thread-safe snapshot {'engaged': bool, 'hand': str|None, 'grip': str,
    'prime': float, 'armed': bool, 'ts': float} of the air-mouse. Read by the HUD
    skeleton preview to colour the hand circle (engaged→blue, closed→orange,
    off→grey), place it on the controlling hand, and (via `prime`) draw the passive
    engage-dwell priming ring (0 at pose-start → 1 at engage; 0 when idle/engaged).
    Returns a COPY so the caller can't mutate the shared dict. Never raises."""
    with _air_mouse_state_lock:
        return dict(_air_mouse_state)


def get_air_mouse_prime() -> float:
    """The current PASSIVE engage-dwell PROGRESS (0.0..1.0) — 0 when idle or already
    engaged, filling toward 1 while a valid smart pose is being held. Cheap
    thread-safe read for the HUD priming ring. NEVER raises."""
    try:
        with _air_mouse_state_lock:
            return float(_air_mouse_state.get("prime", 0.0) or 0.0)
    except Exception:
        return 0.0


# ── TWO-HAND mode hand-off (skills/kinect_two_hand.py) ──────────────────────
# When BOTH hands are engaged the two-hand pinch-to-resize poller takes over and
# the single-hand air-mouse must STAND DOWN (no cursor move, no overlay, any held
# button released) so the two don't fight over the cursor. The two-hand poller
# publishes a heartbeat here every tick it is active; the air-mouse reads it and
# suppresses itself while it is FRESH. The freshness TTL means a crashed/paused
# two-hand poller can NEVER permanently strand the single-hand cursor — if the
# heartbeat goes stale the air-mouse resumes on its own. Same thread-safe pattern
# as the engage-state snapshot above (a different thread writes it).
_TWO_HAND_ACTIVE_TTL_SEC = 0.5     # heartbeat older than this → treat as inactive
_two_hand_state_lock = threading.Lock()
_two_hand_state: dict = {"active": False, "ts": 0.0}


def set_two_hand_active(active: bool) -> None:
    """Publish whether TWO-HAND mode is currently driving (thread-safe). Called by
    skills/kinect_two_hand.py each tick. Stamps the time so the air-mouse can age
    out a stale heartbeat. NEVER raises."""
    try:
        with _two_hand_state_lock:
            _two_hand_state["active"] = bool(active)
            _two_hand_state["ts"] = time.time()
    except Exception:
        pass


def two_hand_active() -> bool:
    """True when TWO-HAND mode is engaged AND its heartbeat is FRESH (within the
    TTL). The single-hand air-mouse reads this and suppresses itself so the two
    don't fight the cursor; a stale heartbeat (two-hand poller gone) reads False so
    the cursor is never permanently stranded. NEVER raises."""
    try:
        with _two_hand_state_lock:
            if not _two_hand_state["active"]:
                return False
            age = time.time() - float(_two_hand_state["ts"] or 0.0)
        return 0.0 <= age <= _TWO_HAND_ACTIVE_TTL_SEC
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════
#  LIVE WIRING (sensor, mouse, overlay, staging gate, config flag)
# ══════════════════════════════════════════════════════════════════════════

def _bridge():
    """Live kinect_bridge module, or None. Prefer the instance the monolith
    already imported; fall back to a direct import (mirrors kinect_gestures)."""
    mod = sys.modules.get("audio.kinect_bridge")
    if mod is not None:
        return mod
    try:
        from audio import kinect_bridge as _kb
        return _kb
    except Exception:
        return None


def _bc():
    """Live monolith module (main or by-name), or None."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _speak(text: str) -> None:
    """Speak a line mid-action (used by the calibration walk-through) via the
    monolith's TTS. Best-effort + silent — mirrors kinect_gestures._speak. NEVER
    raises; a headless/test instance just no-ops."""
    bc = _bc()
    if bc is None:
        return
    try:
        fn = getattr(bc, "_speak", None) or getattr(bc, "speak", None)
        if callable(fn):
            fn(text)
    except Exception:
        pass


def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence. Read fresh
    each call so a Settings toggle takes effect without a restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _saved_settings() -> dict:
    """The owner's persisted settings dict (data/user_settings.json) via the same
    reader model_picker / kinect_gestures use — honours JARVIS_SETTINGS_PATH so a
    test never touches the real file. Returns {} on any failure. NEVER raises."""
    try:
        from tools import settings_window as sw
        cur = sw.load_settings()
        return cur if isinstance(cur, dict) else {}
    except Exception:
        return {}


def _saved_float(settings: dict, key: str) -> "Optional[float]":
    """A persisted float by key, or None when absent / unparseable."""
    try:
        v = settings.get(key)
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _reach_thresholds() -> dict:
    """The LIVE reach-gate thresholds the engage test uses, read fresh each call:
    persisted CALIBRATION values (KINECT_REACH_* / KINECT_STRAIGHT_* in
    user_settings.json) when the owner has calibrated, else the position-
    independent module defaults. Returns {ratio_engage, ratio_disengage,
    fwd_engage, fwd_disengage, straight_engage, straight_disengage}.

    The KINECT_REACH_ENGAGE / _DISENGAGE keys now hold the dimensionless body-
    relative REACH RATIO (the gate's primary forward cue); the absolute fwd_* bars
    are always the module defaults (a fallback used only when no body scale is
    measurable at runtime). A partially written calibration (only some keys) falls
    back per-field to the default, so a half-finished calibration can never strand
    the gate. Defaults are good enough to skip calibration entirely. NEVER raises."""
    s = _saved_settings()
    up_m = _saved_float(s, SETTING_UP_MARGIN)
    down_m = _saved_float(s, SETTING_DOWN_MARGIN)
    ratio_e = _saved_float(s, SETTING_REACH_ENGAGE)
    ratio_d = _saved_float(s, SETTING_REACH_DISENGAGE)
    str_e = _saved_float(s, SETTING_STRAIGHT_ENGAGE)
    str_d = _saved_float(s, SETTING_STRAIGHT_DISENGAGE)
    return {
        # The PRIMARY height-gate margins (persisted override or module default).
        "up_margin": up_m if up_m is not None else AIR_MOUSE_ENGAGE_UP_MARGIN_M,
        "down_margin": down_m if down_m is not None
        else AIR_MOUSE_ENGAGE_DOWN_MARGIN_M,
        # DEMOTED forward / straightness cues — permissive (≈0) by default so they
        # never gate; only the persisted KINECT_REACH_* / KINECT_STRAIGHT_* keys (if
        # the owner sets them) re-enable a secondary bar.
        "ratio_engage": ratio_e if ratio_e is not None
        else AIR_MOUSE_EXTEND_REACH_RATIO_ENGAGE,
        "ratio_disengage": ratio_d if ratio_d is not None
        else AIR_MOUSE_EXTEND_REACH_RATIO_DISENGAGE,
        "fwd_engage": AIR_MOUSE_EXTEND_FORWARD_ENGAGE_M,
        "fwd_disengage": AIR_MOUSE_EXTEND_FORWARD_DISENGAGE_M,
        "straight_engage": str_e if str_e is not None
        else AIR_MOUSE_EXTEND_STRAIGHT_ENGAGE,
        "straight_disengage": str_d if str_d is not None
        else AIR_MOUSE_EXTEND_STRAIGHT_DISENGAGE,
    }


def _hand_mirror_enabled() -> bool:
    """Whether to SWAP the bridge's left↔right hands (selfie-view correction, see
    KINECT_HAND_MIRROR_DEFAULT). Read fresh each call from core.config (Settings
    GUI override) so the owner can flip it live; defaults True."""
    return _cfg_flag("KINECT_HAND_MIRROR", KINECT_HAND_MIRROR_DEFAULT)


def _is_staging() -> bool:
    """True on the staging/test instance — the air-mouse must NEVER move the
    real cursor there. Matches the monolith's own gate plus the raw env var so
    the check holds even before the monolith is importable."""
    if os.environ.get("JARVIS_STAGING", "").strip() == "1":
        return True
    bc = _bc()
    if bc is not None:
        fn = getattr(bc, "_is_staging", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
    return False


def _air_mouse_enabled() -> bool:
    """The master gate for the live loop: opt-in flag ON and not staging."""
    return _cfg_flag("KINECT_AIR_MOUSE_ENABLED") and not _is_staging()


# ─── primary-monitor geometry ──────────────────────────────────────────────
def _primary_screen_size() -> tuple[int, int]:
    """The PRIMARY monitor's (width, height) in pixels. Tries win32 first, then
    a configured MONITORS entry, then a safe default. Never raises."""
    # 1) win32 — the real primary-monitor metrics (SM_CXSCREEN / SM_CYSCREEN).
    try:
        import win32api
        import win32con
        w = int(win32api.GetSystemMetrics(win32con.SM_CXSCREEN))
        h = int(win32api.GetSystemMetrics(win32con.SM_CYSCREEN))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    # 2) A configured MONITORS dict (core.config) — use the "middle"/primary-ish
    #    entry's (w, h) when present.
    try:
        from core import config as _cfg
        monitors = getattr(_cfg, "MONITORS", None)
        if isinstance(monitors, dict):
            for key in ("middle", "primary", "main"):
                ent = monitors.get(key)
                if isinstance(ent, (list, tuple)) and len(ent) >= 4:
                    return int(ent[2]), int(ent[3])
    except Exception:
        pass
    # 3) Fallback.
    return _DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H


# ─── mouse actuation (win32api primary, pyautogui fallback) ────────────────
def _set_cursor_pos(px: int, py: int) -> bool:
    """Move the OS cursor to an absolute primary-monitor pixel. win32api first
    (lowest latency), pyautogui as a fallback. Returns True on success. Never
    raises — a failed move is a silent no-op (the next frame retries)."""
    try:
        import win32api
        win32api.SetCursorPos((int(px), int(py)))
        return True
    except Exception:
        pass
    try:
        import pyautogui
        # FAILSAFE off: the air-mouse legitimately parks the cursor in a screen
        # corner (reach-box clamp), which pyautogui's default failsafe treats as
        # an abort. We do our own clamping, so disable it.
        pyautogui.FAILSAFE = False
        pyautogui.moveTo(int(px), int(py))
        return True
    except Exception:
        return False


def _mouse_button(action: str, button: str = "left") -> bool:
    """Press ('down') or release ('up') the LEFT or RIGHT mouse button at the
    current cursor position. win32api mouse_event flags first, pyautogui fallback.
    `button` ∈ {"left","right"} — the LEFT hand closing maps to "left", the RIGHT
    hand to "right". Returns True on success; never raises."""
    button = (button or "left").lower()
    # win32api path: event flags per button + up/down.
    try:
        import win32api
        import win32con
        if button == "right":
            flag = (win32con.MOUSEEVENTF_RIGHTDOWN if action == "down"
                    else win32con.MOUSEEVENTF_RIGHTUP)
        else:  # left (primary)
            flag = (win32con.MOUSEEVENTF_LEFTDOWN if action == "down"
                    else win32con.MOUSEEVENTF_LEFTUP)
        win32api.mouse_event(flag, 0, 0, 0, 0)
        return True
    except Exception:
        pass
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        btn = "right" if button == "right" else "left"
        if action == "down":
            pyautogui.mouseDown(button=btn)
        else:
            pyautogui.mouseUp(button=btn)
        return True
    except Exception:
        return False


# ─── overlay state publishing + spawn ──────────────────────────────────────
def _publish_overlay_state(decision: AirMouseDecision, visible: bool,
                           prime_xy: "Optional[tuple]" = None) -> None:
    """Write the live cursor + reticle state to AIR_CURSOR_STATE_FILE for the
    overlay process. Atomic-ish (write then it's a tiny file); best-effort and
    silent on failure — the overlay just renders the last good frame.

    Shape: {"x": int, "y": int, "state": "track"|"grab"|"hidden"|"prime",
            "color": "cyan"|"gold", "ts": <epoch>, "visible": bool,
            "prime": float in [0,1]}

    SHARED CONTRACT (2026-07): `prime` is the PASSIVE engage-dwell PROGRESS — 0 at
    pose-start → 1 at engage while priming, 0.0 when idle or already engaged — so
    the overlay can draw a FILLING priming ring. While priming the cursor isn't
    moved (decision.cursor is None), but `prime_xy` (the projected hand pixel) lets
    the overlay position the ring at the hand; the state is "prime" and visible so
    the ring shows without a solid reticle."""
    try:
        import json
        prime = getattr(decision, "prime", 0.0) or 0.0
        priming = prime > 0.0 and decision.cursor is None
        if decision.cursor is not None:
            x, y = decision.cursor
        elif priming and prime_xy is not None:
            x, y = prime_xy   # position the priming ring at the hand
        else:
            x, y = -10000, -10000   # off-screen sentinel; overlay hides anyway
        if priming:
            # PRIMING: show the filling ring at the hand, but no solid reticle.
            state = "prime"
            vis = bool(visible)
        else:
            state = decision.overlay if visible else "hidden"
            vis = bool(visible and state != "hidden")
        data = {
            "x": int(x), "y": int(y),
            "state": state,
            "color": overlay_color_for(state),
            "visible": vis,
            "prime": max(0.0, min(1.0, float(prime))),
            "ts": time.time(),
        }
        with open(AIR_CURSOR_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _clear_overlay_state() -> None:
    """Publish a hidden/blank overlay state (used when the air-mouse turns off or
    the hand is lost) so the reticle disappears promptly. prime 0 — nothing to
    prime while cleared."""
    try:
        import json
        with open(AIR_CURSOR_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"x": -10000, "y": -10000, "state": "hidden",
                       "color": "cyan", "visible": False, "prime": 0.0,
                       "ts": time.time()}, f)
    except Exception:
        pass


_overlay_process = [None]   # module-list so the loop can (re)assign without global

# Edge flag for the DISABLED poll path: once the hidden overlay state + the
# DISENGAGED preview state have been published, don't rewrite them every ~33 ms
# tick — that's pointless file churn and it clobbers the two-hand poller's
# frames. Reset when the enabled path runs (mirrors kinect_two_hand.py's
# _two_hand_overlay_was_active write-once-on-edge flag).
_disabled_state_published = [False]


def _overlay_alive() -> bool:
    proc = _overlay_process[0]
    if proc is None:
        return False
    try:
        return proc.poll() is None
    except Exception:
        return False


def _spawn_overlay() -> None:
    """Spawn hud/jarvis_air_cursor.py as a click-through overlay subprocess sized
    to the virtual desktop, mirroring _launch_reticle_overlay() in the monolith
    (same --x/--y/--width/--height/--parent-pid contract + CREATE_NO_WINDOW).
    Silent on failure so a missing tkinter / odd geometry never breaks the loop.
    Only ever called from the live loop, never in staging/test."""
    if _overlay_alive():
        return
    try:
        import subprocess
        overlay_path = os.path.join(PROJECT_DIR, "hud", "jarvis_air_cursor.py")
        if not os.path.exists(overlay_path):
            return
        vx, vy, vw, vh = _virtual_screen_bounds()
        parent_pid = os.getpid()
        # Prefer the monolith's PID so the overlay dies with JARVIS, not with a
        # transient skill thread (which shares this process anyway, but be
        # explicit/robust if a future reload changes that).
        bc = _bc()
        if bc is not None:
            try:
                parent_pid = int(getattr(bc, "_MAIN_PID", parent_pid) or parent_pid)
            except Exception:
                parent_pid = os.getpid()
        flags = 0
        try:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        except Exception:
            flags = 0
        _overlay_process[0] = subprocess.Popen(
            [sys.executable, overlay_path,
             "--x", str(vx), "--y", str(vy),
             "--width", str(vw), "--height", str(vh),
             "--parent-pid", str(parent_pid)],
            creationflags=flags, close_fds=True,
        )
        print(f"  [air-mouse] cursor overlay launched "
              f"({vw}x{vh} @ {vx},{vy}, pid {_overlay_process[0].pid})")
    except Exception as e:
        print(f"  [air-mouse] overlay launch failed: {e}")
        _overlay_process[0] = None


def _shutdown_overlay() -> None:
    """Terminate the overlay subprocess (best-effort)."""
    proc = _overlay_process[0]
    _overlay_process[0] = None
    if proc is None:
        return
    try:
        proc.terminate()
    except Exception:
        pass


# Win32 GetSystemMetrics indices for the VIRTUAL desktop (all monitors). Used
# both via win32con and, as a no-pywin32 fallback, via ctypes user32 directly.
_SM_XVIRTUALSCREEN = 76     # left edge of the virtual desktop (NEGATIVE if a
_SM_YVIRTUALSCREEN = 77     #   monitor sits left of / above the primary)
_SM_CXVIRTUALSCREEN = 78    # full virtual-desktop width  (sum across monitors)
_SM_CYVIRTUALSCREEN = 79    # full virtual-desktop height


def _virtual_screen_bounds() -> tuple[int, int, int, int]:
    """(x, y, w, h) of the whole virtual desktop spanning ALL monitors. Prefer
    the monolith's helper (single source of truth); fall back to win32, then to
    ctypes user32.GetSystemMetrics, then to the primary size. x/y are NEGATIVE
    when a monitor is arranged left-of / above the primary — SetCursorPos accepts
    these directly, so the whole desktop is reachable."""
    bc = _bc()
    if bc is not None:
        fn = getattr(bc, "_virtual_screen_bounds", None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    try:
        import win32api
        import win32con
        vx = int(win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN))
        vy = int(win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN))
        vw = int(win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN))
        vh = int(win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN))
        if vw > 0 and vh > 0:
            return vx, vy, vw, vh
    except Exception:
        pass
    # pywin32 absent but we may still be on real Windows: ask user32 directly.
    try:
        import ctypes
        gsm = ctypes.windll.user32.GetSystemMetrics
        vx = int(gsm(_SM_XVIRTUALSCREEN))
        vy = int(gsm(_SM_YVIRTUALSCREEN))
        vw = int(gsm(_SM_CXVIRTUALSCREEN))
        vh = int(gsm(_SM_CYVIRTUALSCREEN))
        if vw > 0 and vh > 0:
            return vx, vy, vw, vh
    except Exception:
        pass
    w, h = _primary_screen_size()
    return 0, 0, w, h


# Cached virtual-desktop bounds + the time we last refreshed them, so the live
# loop doesn't hit win32 every tick but still notices a display-layout change.
_VBOUNDS_CACHE: list = [None, 0.0]   # [(x, y, w, h) | None, last_refresh_ts]


def _cached_virtual_bounds(refresh: bool = False) -> tuple[int, int, int, int]:
    """The virtual-desktop bounds, cached. Re-reads when `refresh` is True, when
    nothing is cached yet, or when VIRTUAL_BOUNDS_REFRESH_SECONDS have elapsed —
    so hot-plugging a monitor is picked up without a restart."""
    cached, last = _VBOUNDS_CACHE
    now = time.time()
    if (cached is None or refresh
            or (now - last) >= VIRTUAL_BOUNDS_REFRESH_SECONDS):
        cached = _virtual_screen_bounds()
        _VBOUNDS_CACHE[0] = cached
        _VBOUNDS_CACHE[1] = now
    return cached


def _reach_box_for_virtual_desktop(refresh: bool = False) -> "ReachBox":
    """Build a ReachBox mapped across the WHOLE virtual desktop (all monitors),
    using the cached virtual-screen bounds."""
    vx, vy, vw, vh = _cached_virtual_bounds(refresh=refresh)
    return ReachBox(vw, vh, origin_x=vx, origin_y=vy)


# ─── the per-tick read → decide → act path (unit-tested via _poll_once) ────
def _dist3(a, b) -> Optional[float]:
    """Euclidean 3D distance between two (x, y, z, ...) joints, or None. Local
    mirror of the bridge helper, used only by the no-bridge-helper fallback."""
    try:
        if not a or not b or len(a) < 3 or len(b) < 3:
            return None
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        return (dx * dx + dy * dy + dz * dz) ** 0.5
    except (TypeError, ValueError):
        return None


def _local_arm_extension(joints: dict, side: str) -> dict:
    """Local fallback for the bridge's arm_extension() — same forward-depth +
    straightness math — used only if the loaded bridge lacks the helper (older
    build). NEVER raises. (audio.kinect_bridge.arm_extension is the canonical
    one; this keeps the air-mouse working against any bridge.)"""
    out = {"side": side, "hand": None, "forward_reach_m": None,
           "reach_ratio": None, "body_scale_m": None,
           "straightness": None, "shoulder_hand_m": None, "arm_len_m": None,
           "shoulder_ref_y": None, "lift_m": None}
    try:
        shoulder = joints.get(f"shoulder_{side}")
        elbow = joints.get(f"elbow_{side}")
        hand = joints.get(f"hand_{side}") or joints.get(f"wrist_{side}")
        out["hand"] = hand
        # HEIGHT / LIFT (PRIMARY gate): hand Y above the shoulder line. Prefer
        # spine_shoulder, fall back to the same-side shoulder. Camera y is UP, so
        # lift_m > 0 = hand at/above the shoulder; a desk-resting hand is well below.
        #
        # TRACKING-STATE FLOOR (FILTER 1, mirrors the bridge arm_extension fix on
        # another branch): compute lift_m ONLY when BOTH the hand joint AND the
        # shoulder-ref joint are sensor-TRACKED (TrackingState slot [3] >= 2),
        # finite, and not the exact-zero origin sentinel. An INFERRED (state 1) or
        # not-tracked hand/ref — the noisy guess the SDK emits for an occluded or
        # phantom 2nd-hand joint — leaves lift_m None, so the height gate cannot
        # engage on a hand the Kinect doesn't actually see (fail safe: the real
        # mouse is left alone). shoulder_ref_y is likewise only published when its
        # joint is well-tracked, so a half-seen body never feeds the gate.
        shoulder_ref = joints.get("spine_shoulder")
        if not joint_well_tracked(shoulder_ref):
            shoulder_ref = shoulder
        if joint_well_tracked(shoulder_ref) and joint_well_tracked(hand):
            out["shoulder_ref_y"] = float(shoulder_ref[1])
            out["lift_m"] = float(hand[1]) - float(shoulder_ref[1])
        body_ref = None
        for name in ("spine_mid", "spine_shoulder", "spine_base"):
            j = joints.get(name)
            if j and len(j) >= 3 and float(j[2]) > 0:
                body_ref = j
                break
        if body_ref is None and shoulder and len(shoulder) >= 3:
            body_ref = shoulder
        if (body_ref is not None and hand and len(hand) >= 3
                and float(hand[2]) > 0 and float(body_ref[2]) > 0):
            out["forward_reach_m"] = float(body_ref[2]) - float(hand[2])
        # BODY-RELATIVE reach ratio (POSITION-INDEPENDENT): forward reach / body
        # scale (shoulder width preferred, torso height fallback). Mirrors the
        # bridge's arm_extension so the local fallback gates identically.
        scale = _local_body_scale(joints)
        out["body_scale_m"] = scale
        if (out["forward_reach_m"] is not None and scale is not None
                and scale > 1e-3):
            out["reach_ratio"] = out["forward_reach_m"] / scale
        chord = _dist3(shoulder, hand)
        upper = _dist3(shoulder, elbow)
        fore = _dist3(elbow, hand)
        out["shoulder_hand_m"] = chord
        if upper is not None and fore is not None:
            arm_len = upper + fore
            out["arm_len_m"] = arm_len
            if chord is not None and arm_len > 1e-3:
                out["straightness"] = min(1.0, chord / arm_len)
    except (TypeError, ValueError, KeyError):
        pass
    return out


def _local_body_scale(joints: dict) -> Optional[float]:
    """Local fallback mirroring audio.kinect_bridge._body_scale_m: a body-size
    span (shoulder width, then torso height) in metres for normalising the forward
    reach into a position-independent ratio. NEVER raises."""
    try:
        width = _dist3(joints.get("shoulder_left"), joints.get("shoulder_right"))
        if width is not None and width > 0.10:
            return width
        torso = _dist3(joints.get("spine_base"),
                       joints.get("spine_shoulder") or joints.get("spine_mid"))
        if torso is not None and torso > 0.10:
            return torso
    except (TypeError, ValueError, KeyError):
        return None
    return None


def _arm_extension(bridge, joints: dict, side: str) -> "ArmExtension":
    """Build the ArmExtension for one side. Prefers the bridge's arm_extension()
    geometry helper (single source of truth for forward-reach + straightness);
    falls back to computing it locally so the air-mouse still works against an
    older bridge build that lacks the helper. NEVER raises."""
    try:
        fn = getattr(bridge, "arm_extension", None)
        if callable(fn):
            return ArmExtension.from_bridge(fn(joints, side))
    except Exception:
        pass
    try:
        return ArmExtension.from_bridge(_local_arm_extension(joints, side))
    except Exception:
        return ArmExtension(side, None, None, None)


# The NEAREST body's id from the most recent _hand_sample(), stashed here so
# _poll_once can pass it to AirMouseController.update(body_id=...) WITHOUT changing
# _hand_sample's 5-tuple return arity (FILTER 6 body-id pin). Module-list so it
# survives across calls; None when no body was in view this tick.
_last_body_id: list = [None]

# The NEAREST body's FACING yaw (|degrees from square|) from the most recent
# _hand_sample(), stashed like _last_body_id so _poll_once can feed the PASSIVE
# smart-engage facing gate WITHOUT changing _hand_sample's return arity. None when
# the bridge didn't provide facing (older build / not measurable) — the gate treats
# a None facing as "facing OK" so it never becomes an un-passable gate.
_last_facing_deg: list = [None]


def _body_facing_deg(body: dict) -> "Optional[float]":
    """|facing yaw| in degrees from square for a bridge body dict, or None when
    unavailable. Prefers the numeric "facing_yaw_deg" (0=square, ±=left/right);
    falls back to the boolean "facing" (True → 0° i.e. squarely facing, so it
    passes the gate; False → a large angle so it fails). None when neither is
    present. PURE; NEVER raises."""
    try:
        yaw = body.get("facing_yaw_deg")
        if isinstance(yaw, (int, float)):
            return abs(float(yaw))
        facing = body.get("facing")
        if facing is True:
            return 0.0            # squarely facing → passes the facing gate
        if facing is False:
            return 180.0          # turned away → fails any sane facing bar
    except Exception:
        return None
    return None


def _hand_sample(bridge) -> tuple["Optional[ArmExtension]", "Optional[ArmExtension]",
                                  str, str, bool]:
    """Read the per-arm extension + per-hand grips from the bridge:
    (left_ext, right_ext, left_grip, right_grip, tracked).

    left_ext / right_ext are the ArmExtension (forward-reach + straightness +
    controlling-hand joint) for each arm of the NEAREST body, or None when that
    arm's joints aren't usable. left_grip / right_grip are that body's raw grips.
    tracked is whether a body was in view. As a side effect, stashes the nearest
    body's id in _last_body_id[0] (FILTER 6) — None when no body. NEVER raises —
    any failure degrades to (None, None, "unknown", "unknown", False) which the
    controller treats as a dead-man release (no arm extended / not tracked →
    disengaged)."""
    none_result = (None, None, "unknown", "unknown", False)
    _last_body_id[0] = None
    _last_facing_deg[0] = None
    try:
        if not bridge.get_enabled():
            return none_result
        ok, _reason = bridge.available()
        if not ok:
            return none_result
        bodies = bridge.get_bodies()
    except Exception:
        return none_result
    if not bodies:
        return none_result

    # Nearest body (same ranking the rest of the stack uses).
    def _key(b):
        d = b.get("distance_m") if isinstance(b, dict) else None
        return d if isinstance(d, (int, float)) and d > 0 else float("inf")
    try:
        body = min((b for b in bodies if isinstance(b, dict)), key=_key)
    except (TypeError, ValueError):
        return none_result

    # Stash the controlling body's id for the FILTER 6 pin (best-effort).
    try:
        _last_body_id[0] = body.get("id")
    except Exception:
        _last_body_id[0] = None
    # Stash the body's FACING yaw for the PASSIVE smart-engage facing gate (None
    # when unavailable → treated as facing OK downstream).
    try:
        _last_facing_deg[0] = _body_facing_deg(body)
    except Exception:
        _last_facing_deg[0] = None

    joints = body.get("joints") or {}
    left_grip = (body.get("hand_left") or "unknown").lower()
    right_grip = (body.get("hand_right") or "unknown").lower()
    left_ext = _arm_extension(bridge, joints, "left")
    right_ext = _arm_extension(bridge, joints, "right")
    # ISSUE 1 — selfie-view correction: the Kinect stream is MIRRORED, so the
    # owner's REAL left hand is what the SDK labels "right" (and vice-versa). When
    # KINECT_HAND_MIRROR is on, SWAP the two hands here — BOTH the grips AND the
    # per-arm extensions (relabelling each .side) — so everything downstream (the
    # per-hand L/R clicks, choose_controlling_arm, the published which-hand, the
    # preview circle's prefer_side) treats the owner's REAL left hand as LEFT.
    if _hand_mirror_enabled():
        left_ext, right_ext = (_relabel_arm_side(right_ext, "left"),
                               _relabel_arm_side(left_ext, "right"))
        left_grip, right_grip = right_grip, left_grip
    return left_ext, right_ext, left_grip, right_grip, True


def _relabel_arm_side(ext: "Optional[ArmExtension]",
                      side: str) -> "Optional[ArmExtension]":
    """Return `ext` with its .side relabelled (used by the mirror swap so a
    swapped arm reports the side it now drives). None passes through. The geometry
    (forward-reach / straightness / hand joint) is unchanged — only the label, so
    the published which-hand + preview circle land on the correct side."""
    if ext is None:
        return None
    return ArmExtension(side, ext.forward_m, ext.straightness, ext.hand,
                        reach_ratio=ext.reach_ratio, lift_m=ext.lift_m,
                        shoulder_ref_y=ext.shoulder_ref_y)


# ─── ISSUE 2a: CALIBRATION capture ──────────────────────────────────────────
CALIBRATE_CAPTURE_SECONDS = 3.0           # hold each pose this long
CALIBRATE_POLL_HZ = 15.0                  # sample cadence while capturing
CALIBRATE_POLL_INTERVAL = 1.0 / CALIBRATE_POLL_HZ
CALIBRATE_MAX_SECONDS = 4.0               # hard wall-time cap per pose (slack)


def _capture_reach(bridge, seconds: float = CALIBRATE_CAPTURE_SECONDS,
                   sleep_fn=time.sleep, now_fn=time.monotonic
                   ) -> "tuple[Optional[float], Optional[float], int]":
    """Sample the HIGHEST-raised hand's HEIGHT (lift_m, the primary cue) for
    ~`seconds` and return its MEDIAN plus the usable-frame count:
    (median_lift_m, median_straightness, n_samples).

    Captures the HEIGHT of the hand above the shoulder line (body-relative, so the
    fit is POSITION-INDEPENDENT — the owner can calibrate at any distance). Reads
    the same _hand_sample() the live gate uses (so the mirror swap + geometry are
    identical), taking, per frame, the highest-raised hand. Median (not mean)
    shrugs off the odd Kinect outlier. A wedged sensor can't hang the voice loop —
    capped at CALIBRATE_MAX_SECONDS. NEVER raises."""
    lift_samples: list = []
    straight_samples: list = []
    n = 0
    try:
        deadline = now_fn() + min(float(seconds), CALIBRATE_MAX_SECONDS)
        while now_fn() < deadline:
            left_ext, right_ext, _lg, _rg, tracked = _hand_sample(bridge)
            if tracked:
                arms = [a for a in (left_ext, right_ext) if a is not None]
                arm = (max(arms, key=lambda a: a.reach_score())
                       if arms else None)
                if arm is not None:
                    n += 1
                    if arm.lift_m is not None:
                        lift_samples.append(float(arm.lift_m))
                    if arm.straightness is not None:
                        straight_samples.append(float(arm.straightness))
            sleep_fn(CALIBRATE_POLL_INTERVAL)
    except Exception:
        pass
    return _median(lift_samples), _median(straight_samples), n


def _persist_reach_thresholds(th: dict) -> bool:
    """Write the fitted HEIGHT-gate margins to user_settings.json (KINECT_LIFT_*),
    plus the straightness keys for back-compat, via the hardened settings writer.
    The KINECT_LIFT_* keys persist the body-relative raise/lower margins
    (position-independent), so the calibration holds at any seating distance.
    All-or-nothing-ish: each key is written; returns True iff every write reported
    success. NEVER raises."""
    ok = True
    ok = _persist_setting(SETTING_UP_MARGIN, float(th["up_margin"])) and ok
    ok = _persist_setting(SETTING_DOWN_MARGIN, float(th["down_margin"])) and ok
    ok = _persist_setting(SETTING_STRAIGHT_ENGAGE,
                          float(th["straight_engage"])) and ok
    ok = _persist_setting(SETTING_STRAIGHT_DISENGAGE,
                          float(th["straight_disengage"])) and ok
    return ok


# ─── PER-APP AUTO-DISABLE (stand down over fullscreen games / video) ─────────
# The air-mouse STANDS DOWN whenever the FOREGROUND window's title/class matches a
# disabled-app hint (AIR_MOUSE_DISABLED_APP_HINTS) — e.g. a fullscreen game or
# video where a stray cursor grab would be disruptive. The MATCH is a pure,
# unit-testable function; the live read reuses kinect_two_hand's foreground
# helpers (single source of truth for the win32 GetForegroundWindow / class /
# title wrappers). Defensive: any win32 failure is treated as NOT disabled, so a
# glitch never accidentally kills the mouse.

def app_is_disabled(title: str, class_name: str,
                    hints: "Optional[list]" = None) -> bool:
    """PURE: does the foreground window's TITLE or CLASS contain any disabled-app
    hint (case-insensitive substring)? `hints` defaults to the live
    AIR_MOUSE_DISABLED_APP_HINTS. Empty/whitespace title+class → False (nothing to
    match). NEVER raises (a non-string degrades to empty)."""
    try:
        if hints is None:
            hints = _cfg("AIR_MOUSE_DISABLED_APP_HINTS",
                         AIR_MOUSE_DISABLED_APP_HINTS)
        hay = ((str(title or "") + " " + str(class_name or "")).lower())
        if not hay.strip():
            return False
        for h in (hints or []):
            h = (str(h or "")).strip().lower()
            if h and h in hay:
                return True
    except Exception:
        return False
    return False


def _foreground_title_class() -> "tuple[str, str]":
    """(title, class_name) of the current FOREGROUND window, reusing
    kinect_two_hand's win32 helpers (the codebase's foreground helper). ("", "") on
    any failure / off Windows. NEVER raises."""
    try:
        from skills import kinect_two_hand as _kt
    except Exception:
        try:
            import kinect_two_hand as _kt   # isolated-skill import fallback
        except Exception:
            return "", ""
    try:
        hwnd = _kt._get_foreground_hwnd()
        if not hwnd:
            return "", ""
        title = _kt._window_title(hwnd) or ""
        cls = _kt._window_class_name(hwnd) or ""
        return title, cls
    except Exception:
        return "", ""


def _per_app_disabled() -> bool:
    """True when the air-mouse should STAND DOWN because the foreground app matches
    a disabled-app hint. Gated on AIR_MOUSE_PER_APP_DISABLE (read live). DEFENSIVE:
    any win32/read failure → False (never accidentally kills the mouse). NEVER
    raises."""
    try:
        if not _cfg_flag("AIR_MOUSE_PER_APP_DISABLE", AIR_MOUSE_PER_APP_DISABLE):
            return False
        title, cls = _foreground_title_class()
        return app_is_disabled(title, cls)
    except Exception:
        return False


def _priming_cursor(ctrl: "AirMouseController", left_ext, right_ext
                    ) -> "Optional[tuple]":
    """The projected cursor pixel for the PRIMING ring — the highest-raised arm's
    hand mapped through the reach box (WITHOUT moving the real cursor). Best-effort;
    None when no usable arm. The overlay draws the filling ring here so it sits on
    the hand the owner is priming with. NEVER raises."""
    try:
        arms = [a for a in (left_ext, right_ext)
                if a is not None and a.hand is not None]
        if not arms:
            return None
        arm = max(arms, key=lambda a: a.reach_score())
        h = arm.hand
        return ctrl.reach.map(float(h[0]), float(h[1]))
    except Exception:
        return None


def _apply_decision(decision: AirMouseDecision) -> None:
    """Perform the side effects of a decision: move the cursor and actuate the
    per-hand buttons. Pure-core stays I/O-free; THIS is where the real mouse is
    touched. Best-effort; never raises out to the loop."""
    acted = False
    if decision.cursor is not None:
        _set_cursor_pos(decision.cursor[0], decision.cursor[1])
        acted = True
    for button, action in decision.button_edges:
        _mouse_button(action, button)
        acted = True
    # Stamp our own injected activity so the auto-yield polling fallback doesn't
    # mistake the air-mouse's OWN cursor move / click for the owner's real input.
    # (The LL-hook path already ignores injected events directly.)
    if acted:
        _mark_self_action()


# ─── ISSUE 2b: live HEIGHT-gate DEBUG LOG (~2 Hz) ────────────────────────────
# While the air-mouse is enabled, print the live numbers at ~2 Hz so the owner can
# SEE what the gate sees and tune the margins, e.g.
#   [air-mouse] lift=+0.07 hand=right engaged=True yield=False reach=0.18 straight=0.91
# lift is the HEIGHT of the controlling hand above the shoulder (the PRIMARY gate);
# yield is True while the air-mouse is suppressed by recent REAL input. The
# highest-raised hand is logged (the one the gate is judging). reach/straight are
# kept as the demoted secondary cues for context.
_AIR_MOUSE_DEBUG_INTERVAL = 0.5             # seconds between debug lines (~2 Hz)
_air_mouse_debug_last = [0.0]               # module-list so the throttle persists


def _format_reach_debug(left_ext, right_ext, tracked: bool, ctrl,
                        yielding: "Optional[bool]" = None) -> str:
    """The ~2 Hz debug line. Leads with the HEIGHT delta (lift = hand_y minus the
    shoulder Y, the PRIMARY gate) for the highest-raised hand, the controlling
    side, engaged, and the auto-YIELD state; then the demoted reach/straight cues
    for context. PURE-ish (reads ctrl state); NEVER raises."""
    try:
        if yielding is None:
            yielding = real_input_recent()
        arms = [a for a in (left_ext, right_ext) if a is not None]
        arm = max(arms, key=lambda a: a.reach_score()) if arms else None
        if arm is None:
            lift_s = "n/a"
            reach_s, straight_s, hand_s = "n/a", "n/a", "none"
        else:
            lift_s = ("%+.2f" % arm.lift_m) if arm.lift_m is not None else "n/a"
            reach_s = ("%.2f" % arm.forward_m) if arm.forward_m is not None else "n/a"
            straight_s = ("%.2f" % arm.straightness
                          if arm.straightness is not None else "n/a")
            hand_s = arm.side or "?"
        return ("  [air-mouse] lift=%s hand=%s engaged=%s yield=%s reach=%s "
                "straight=%s tracked=%s"
                % (lift_s, hand_s, bool(ctrl.engaged), bool(yielding),
                   reach_s, straight_s, bool(tracked)))
    except Exception:
        return "  [air-mouse] lift=? hand=? engaged=? yield=? reach=? straight=?"


def _maybe_debug_log(left_ext, right_ext, tracked: bool, ctrl,
                     now: "Optional[float]" = None,
                     yielding: "Optional[bool]" = None) -> bool:
    """Emit the height-gate debug line if the throttle window has elapsed. Returns
    True iff a line was printed (for the test). NEVER raises."""
    try:
        t = time.monotonic() if now is None else float(now)
        if (t - _air_mouse_debug_last[0]) < _AIR_MOUSE_DEBUG_INTERVAL:
            return False
        _air_mouse_debug_last[0] = t
        print(_format_reach_debug(left_ext, right_ext, tracked, ctrl,
                                  yielding=yielding))
        return True
    except Exception:
        return False


def _poll_once(ctrl: AirMouseController, bridge) -> Optional[AirMouseDecision]:
    """One air-mouse tick: read the arms, decide, and (only when enabled +
    not staging) ACT — move the cursor, actuate the per-hand buttons, publish the
    overlay state. Returns the decision (for tests) or None when the bridge is
    absent. NEVER raises.

    GATING: the controller is ALWAYS advanced (so its smoothing/grip state stays
    current and a re-enable doesn't see a huge gap), but the SIDE EFFECTS (mouse
    move, buttons, visible overlay) only happen when KINECT_AIR_MOUSE_ENABLED is
    on AND not staging. Flipping the flag off therefore stops the cursor moving
    instantly and releases any held button via the dead-man path."""
    if bridge is None:
        return None
    left_ext, right_ext, left_grip, right_grip, tracked = _hand_sample(bridge)
    # Live engage bars (owner's persisted height margins / calibration or the
    # defaults), read fresh each tick so a tweak takes effect with no restart. The
    # controller applies them in its height engage hysteresis.
    thresholds = _reach_thresholds()
    # AUTO-YIELD: has the owner touched their REAL mouse/keyboard recently? If so
    # the controller force-disengages + stays suppressed this frame so the real
    # input always wins. Lazily ensure the watcher is installed (the LL hook, or
    # its polling fallback). Best-effort; never blocks the gate.
    _install_yield_watcher()
    # STAND DOWN for TWO-HAND mode: when both hands are engaged the two-hand pinch-
    # to-resize poller (skills/kinect_two_hand.py) drives instead, so the single-
    # hand air-mouse must not also grab the cursor. Fold it into the same yield
    # signal as recent real input — the controller force-disengages, releases any
    # held button, and stays suppressed this frame. (Fresh-heartbeat-gated, so a
    # dead two-hand poller can't strand the cursor.)
    two_hand = two_hand_active()
    yielding = real_input_recent() or two_hand
    # The nearest body's id this tick (stashed by _hand_sample) so the controller
    # PINS the controlling body and releases — rather than silently retargets — if
    # a closer 2nd person steals the "nearest" slot mid-drag (FILTER 6).
    body_id = _last_body_id[0]
    # The body's FACING yaw (|deg from square|, or None) for the PASSIVE smart-pose
    # facing gate; the module ARMED flag (relaxed gate when the owner asked for the
    # cursor); and the PER-APP disable (stand down over a fullscreen game/video).
    facing_deg = _last_facing_deg[0]
    armed = air_mouse_is_armed()
    per_app_disabled = _per_app_disabled()
    try:
        decision = ctrl.update(left_ext, right_ext, left_grip, right_grip,
                               tracked, thresholds=thresholds,
                               real_input_recent=yielding, body_id=body_id,
                               facing_deg=facing_deg, armed=armed,
                               per_app_disabled=per_app_disabled)
    except Exception:
        # A controller error must not strand a held button — force a release.
        try:
            decision = ctrl.release_decision()
        except Exception:
            return None

    enabled = _air_mouse_enabled()
    # ISSUE 2b: while enabled, surface the live height/lift numbers + yield state
    # at ~2 Hz for tuning. Throttled + best-effort; only when enabled so a disabled
    # poller stays quiet.
    if enabled:
        _maybe_debug_log(left_ext, right_ext, tracked, ctrl, yielding=yielding)
    if not enabled:
        # Gated OFF mid-session: make sure no button is left held and the
        # overlay is hidden. ctrl.update already returned a (possibly
        # button-up) decision if it had been holding; honour pending 'up's
        # (per hand) so a flag flip during a drag still releases, but never a
        # 'down'.
        for button, action in decision.button_edges:
            if action == "up":
                _mouse_button("up", button)
        # Write-once-on-edge: publish the hidden overlay + DISENGAGED preview
        # state (grey hand-circle, never live blue/orange) exactly once per
        # disable, not on every ~33 ms tick — the poller runs even when the
        # feature is off, and a per-tick rewrite is SSD churn that also fights
        # the two-hand poller for AIR_CURSOR_STATE_FILE.
        if not _disabled_state_published[0]:
            _clear_overlay_state()
            _set_air_mouse_state(False, "open", None)
            _disabled_state_published[0] = True
        return decision

    # Enabled + not staging: act.
    _disabled_state_published[0] = False
    _apply_decision(decision)
    priming = getattr(decision, "prime", 0.0) > 0.0 and decision.cursor is None
    # While PRIMING the reticle isn't shown (no cursor move), but the overlay draws
    # a filling ring at the hand — so it's "visible" for the ring even though the
    # decision overlay is hidden. Project the priming hand to a pixel for the ring.
    visible = (tracked and (decision.overlay != "hidden" or priming))
    prime_xy = _priming_cursor(ctrl, left_ext, right_ext) if priming else None
    # STAND DOWN on the overlay file too: while two-hand mode is active its
    # poller owns AIR_CURSOR_STATE_FILE (dual reticles). Publishing our hidden
    # frames at ~30 Hz alongside it makes the overlay strobe between dual-
    # reticle and hidden depending on which writer landed last — so stay quiet;
    # the two-hand skill edge-clears the file itself when its mode ends.
    if not two_hand:
        _publish_overlay_state(decision, visible, prime_xy=prime_xy)
    # Publish the LIVE engage state + which hand + grip + PRIME for the HUD skeleton
    # preview's hand circle (B2): engaged→blue, closed→orange, disengaged→grey,
    # drawn on the controlling hand; the passive dwell PROGRESS drives the priming
    # ring. Read off the controller (engaged + hand) + the decision (grip + prime)
    # so the preview colour + ring match the cursor/reticle.
    try:
        _set_air_mouse_state(bool(ctrl.engaged), decision.grip, ctrl.hand,
                             prime=getattr(decision, "prime", 0.0))
    except Exception:
        pass
    # Keep the reticle overlay process alive while enabled.
    if visible and not _overlay_alive():
        _spawn_overlay()
    return decision


def _poll_loop() -> None:  # pragma: no cover - non-terminating daemon; each tick delegates to _poll_once, which is unit-tested directly
    time.sleep(INITIAL_DELAY_SECONDS)
    bridge = _bridge()
    if bridge is None:
        print("  [air-mouse] kinect_bridge unavailable — poller exiting")
        return
    # Map across the WHOLE virtual desktop (all monitors), not just primary.
    ctrl = AirMouseController(_reach_box_for_virtual_desktop(refresh=True))
    was_enabled = False
    last_bounds_refresh = time.time()
    while True:
        try:
            bridge = _bridge() or bridge
            enabled = _air_mouse_enabled()
            now = time.time()
            if enabled and not was_enabled:
                # Just turned on — re-read the virtual-desktop bounds (the user
                # may have changed displays) and start fresh so the first reach
                # snaps to where it's pointing.
                ctrl.reach = _reach_box_for_virtual_desktop(refresh=True)
                last_bounds_refresh = now
                ctrl.reset()
            elif enabled and (now - last_bounds_refresh) >= VIRTUAL_BOUNDS_REFRESH_SECONDS:
                # Periodic refresh while running so a hot-plugged / rearranged
                # monitor is picked up live. _cached_virtual_bounds() only hits
                # win32 when the interval has actually elapsed, so this is cheap;
                # the ReachBox is rebuilt only when the bounds changed.
                vb = _cached_virtual_bounds(refresh=True)
                last_bounds_refresh = now
                cur = ctrl.reach
                if (vb[0], vb[1], vb[2], vb[3]) != (
                        cur.origin_x, cur.origin_y, cur.screen_w, cur.screen_h):
                    ctrl.reach = ReachBox(vb[2], vb[3], origin_x=vb[0], origin_y=vb[1])
            if was_enabled and not enabled:
                # Just turned off — tear the overlay down + clear its state.
                _shutdown_overlay()
                _clear_overlay_state()
            was_enabled = enabled
            _poll_once(ctrl, bridge)
        except Exception as e:
            print(f"  [air-mouse] poll error: {e}")
        time.sleep(AIR_MOUSE_POLL_INTERVAL)


# ─── persistence (reuse the hardened Settings writer) ──────────────────────
def _persist_setting(key: str, value) -> bool:
    """Write {key: value} into the settings file WITHOUT clobbering the owner's
    other settings — the EXACT path model_picker / kinect_gestures use
    (settings_window.load_settings + save_settings, which honour
    JARVIS_SETTINGS_PATH so tests can't touch the real file). Best-effort."""
    try:
        from tools import settings_window as sw
    except Exception:
        return False
    try:
        current = sw.load_settings()
        if not isinstance(current, dict):
            current = {}
        current[key] = value
        sw.save_settings(current)
        return True
    except Exception:
        return False


def _set_enabled(on: bool) -> bool:
    """Flip KINECT_AIR_MOUSE_ENABLED live (core.config) and persist it."""
    try:
        import core.config as _cfg
        _cfg.KINECT_AIR_MOUSE_ENABLED = bool(on)
    except Exception:
        pass
    return _persist_setting("KINECT_AIR_MOUSE_ENABLED", bool(on))


# ─── sensor-readiness (honest spoken reasons) ──────────────────────────────
def _sensor_ready() -> tuple[bool, str]:
    """(True, "") when the Kinect is enabled AND available; else (False, why)."""
    kb = _bridge()
    if kb is None:
        return False, "the Kinect bridge isn't loaded"
    try:
        if not kb.get_enabled():
            return False, "the Kinect is switched off"
        ok, reason = kb.available()
        if not ok:
            return False, (reason or "the Kinect is unavailable")
    except Exception:
        return False, "the Kinect is unavailable"
    return True, ""


def _hand_in_view() -> Optional[bool]:
    """True/False if the Kinect can currently see a usable hand; None when the
    sensor is off/absent so the caller can phrase it honestly."""
    kb = _bridge()
    if kb is None:
        return None
    try:
        if not kb.get_enabled():
            return None
        ok, _reason = kb.available()
        if not ok:
            return None
        states = kb.get_hand_states()
        if not states.get("tracked"):
            return False
        return (states.get("right") in ("open", "closed")
                or states.get("left") in ("open", "closed"))
    except Exception:
        return None


# ─── actions ─────────────────────────────────────────────────────────────
def air_mouse_on(_: str = "") -> str:
    """Turn the air-mouse on (live + persisted)."""
    if _cfg_flag("KINECT_AIR_MOUSE_ENABLED"):
        already = "The air-mouse is already on, sir."
    else:
        already = None
    persisted = _set_enabled(True)
    ready, why = _sensor_ready()
    sensor_note = "" if ready else f" Note {why} — enable the Kinect so I can see your hand."
    if already:
        return already + sensor_note
    msg = ("Air-mouse on, sir — raise a hand above your shoulder to take the "
           "cursor, close your left hand to left-click or your right hand to "
           "right-click, hold a hand closed to drag, and lower your hand to "
           "release the cursor. Touch your real mouse and I'll yield instantly.")
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg + sensor_note


def air_mouse_off(_: str = "") -> str:
    """Turn the air-mouse off (live + persisted). Also clears the overlay so the
    reticle disappears immediately."""
    if not _cfg_flag("KINECT_AIR_MOUSE_ENABLED"):
        return "The air-mouse is already off, sir."
    persisted = _set_enabled(False)
    # Make sure nothing is left held (BOTH buttons) and the reticle is gone.
    try:
        _mouse_button("up", "left")
        _mouse_button("up", "right")
    except Exception:
        pass
    _shutdown_overlay()
    _clear_overlay_state()
    msg = "Air-mouse off, sir."
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


def air_mouse_status(_: str = "") -> str:
    """Report whether the air-mouse is on + whether a hand is in view.
    'is the air-mouse on' / 'air-mouse status'."""
    enabled = _cfg_flag("KINECT_AIR_MOUSE_ENABLED")
    how = ("raise a hand above your shoulder to take the cursor, close your left "
           "hand to left-click or your right hand to right-click, hold to drag, "
           "and lower your hand to release the cursor")
    if not enabled:
        return (f"The air-mouse is off, sir — say 'turn on the air-mouse' to "
                f"enable it. Once on, {how}.")
    in_view = _hand_in_view()
    if in_view is None:
        return ("The air-mouse is on, sir, but the Kinect is off or "
                "unavailable, so I can't see your hand right now.")
    if in_view:
        return f"The air-mouse is on and I can see your hand, sir — {how}."
    return ("The air-mouse is on, sir, but I don't see a hand in the Kinect's "
            f"view at the moment. Raise a hand above your shoulder and {how}.")


# ─── ARM / DISARM (the hybrid voice-arm mode) ────────────────────────────────
def air_mouse_arm_action(_: str = "") -> str:
    """ARM the air-mouse — the owner explicitly asked for the cursor, so relax the
    strict PASSIVE smart-pose gate to the RESPONSIVE armed gate (height-only + a
    short hold; no open-palm/facing/still/dwell required). Also ensures the feature
    is ON (an explicit "take the cursor" IS consent) so a raised hand actually
    drives. 'mouse control on' / 'take the cursor' / 'give me the cursor' /
    'hand mouse on'. NEVER raises out."""
    air_mouse_arm()
    # Arming implies wanting control now — make sure the master flag is on so a
    # raised hand actually moves the cursor (mirrors air_control_on's consent
    # model). If it was already on this is a harmless re-persist.
    was_on = _cfg_flag("KINECT_AIR_MOUSE_ENABLED")
    persisted = True
    if not was_on:
        persisted = _set_enabled(True)
    ready, why = _sensor_ready()
    sensor_note = ("" if ready
                   else f" Note {why} — enable the Kinect so I can see your hand.")
    msg = ("Mouse control armed, sir — raise a hand above your shoulder and I'll "
           "take the cursor right away; close your left hand to left-click, your "
           "right to right-click, and touch your real mouse to yield.")
    if not persisted:
        msg += " (I couldn't save the enable, so it'll revert on restart.)"
    return msg + sensor_note


def air_mouse_disarm_action(_: str = "") -> str:
    """DISARM the air-mouse — return to the PASSIVE strict smart-pose gate (open
    palm + facing + brief hold), so a casual raised/reaching hand no longer grabs
    the cursor. This does NOT turn the feature off (that's 'air-mouse off'); it
    just makes engaging deliberate again. Also releases any held button + clears
    the reticle so nothing is stranded. 'mouse control off' / 'release the cursor' /
    'hand mouse off'. NEVER raises out."""
    air_mouse_disarm()
    # Let go of anything currently held so disarming can't strand a button, and
    # hide the reticle promptly.
    try:
        _mouse_button("up", "left")
        _mouse_button("up", "right")
    except Exception:
        pass
    try:
        _clear_overlay_state()
    except Exception:
        pass
    return ("Mouse control released, sir — I'll only take the cursor now on a "
            "deliberate open-palm hold. Say 'air-mouse off' to disable it entirely.")


def calibrate_air_mouse(_: str = "") -> str:
    """The CALIBRATION walk-through ('calibrate air mouse' / 'calibrate reach').
    Repurposed for the HEIGHT gate: speaks the owner through it — RAISE a hand
    above the shoulder for ~3 s (capture the raised hand HEIGHT median), then LOWER
    it to the desk for ~3 s (capture the lowered median), fit the up/down engage
    margins ~60 %/40 % of the way lowered→raised, and persist them (KINECT_LIFT_* in
    user_settings.json) so the live gate auto-fits the owner's body. Honest on
    every failure — never claims to have calibrated something it didn't capture.

    Runs synchronously (like point_calibrate): it speaks each prompt, captures,
    then returns the spoken summary."""
    if _is_staging():
        return "Not while I'm in staging, sir."
    ready, why = _sensor_ready()
    if not ready:
        return (f"I can't calibrate the air-mouse, sir — {why}. Enable the "
                "Kinect and try again.")
    bridge = _bridge()
    if bridge is None:
        return "I can't calibrate the air-mouse, sir — the Kinect bridge isn't loaded."

    # 1) RAISED pose (hand up above the shoulder).
    _speak("Let's calibrate the air-mouse, sir. Raise one hand up above your "
           "shoulder, like you're pointing at the screen, and hold it there.")
    raised_lift, raised_straight, raised_n = _capture_reach(bridge)
    if raised_n == 0:
        return ("I couldn't see your hand while you raised it, sir — make sure "
                "you're in the Kinect's view and try calibrating again.")

    # 2) LOWERED pose (hand down at the desk).
    _speak("Got it. Now lower your hand down to the desk, by your side.")
    lowered_lift, lowered_straight, lowered_n = _capture_reach(bridge)
    if lowered_n == 0:
        return ("I lost track of you while you lowered your hand, sir — please "
                "try calibrating again.")

    # 3) Fit + persist the height margins (body-relative → position-independent).
    th = compute_reach_thresholds(lowered_lift, raised_lift,
                                  lowered_straight, raised_straight)
    persisted = _persist_reach_thresholds(th)
    # Did we actually fit the height margins from the capture (vs. default)?
    fitted_lift = (lowered_lift is not None and raised_lift is not None
                   and (raised_lift - lowered_lift) >= 0.15)
    if not fitted_lift:
        return ("I couldn't tell your raised hand apart from your lowered one, "
                "sir — raise it well above your shoulder then drop it to the "
                "desk, and calibrate again.")
    msg = ("Air-mouse calibrated, sir — raise a hand above your shoulder to take "
           "the cursor, and it releases when you lower your hand. It now works "
           "the same wherever you sit or stand.")
    if not persisted:
        msg += " (I couldn't save it, so it'll revert on restart.)"
    return msg


# ─── registration ────────────────────────────────────────────────────────
def register(actions):
    actions["air_mouse_on"] = air_mouse_on
    actions["air_mouse_off"] = air_mouse_off
    actions["air_mouse_status"] = air_mouse_status
    actions["calibrate_air_mouse"] = calibrate_air_mouse
    # ARM / DISARM (hybrid voice-arm + smart pose). Canonical names + natural-
    # language aliases sharing the same handler, so whichever token the LLM emits
    # from "mouse control on / take the cursor / give me the cursor / hand mouse on"
    # (and the off variants) resolves to arm/disarm.
    actions["air_mouse_arm"] = air_mouse_arm_action
    actions["air_mouse_disarm"] = air_mouse_disarm_action
    for alias in ("mouse_control_on", "take_the_cursor", "give_me_the_cursor",
                  "hand_mouse_on"):
        actions[alias] = air_mouse_arm_action
    for alias in ("mouse_control_off", "release_the_cursor", "hand_mouse_off"):
        actions[alias] = air_mouse_disarm_action

    # Guard against duplicate pollers on skill reload (same OS-thread-name check
    # kinect_gestures / face_tracker use). The loop self-gates on
    # KINECT_AIR_MOUSE_ENABLED + staging each tick, so it's cheap to leave
    # running even when disabled.
    if any(th.name == _THREAD_NAME and th.is_alive()
           for th in threading.enumerate()):
        print("  [air-mouse] poller already running — skipping duplicate (reload)")
    else:
        t = threading.Thread(target=_poll_loop, daemon=True, name=_THREAD_NAME)
        t.start()
        print(f"  [air-mouse] air-mouse poller active (~{AIR_MOUSE_POLL_HZ:.0f} Hz; "
              "opt-in via KINECT_AIR_MOUSE_ENABLED, off by default)")
