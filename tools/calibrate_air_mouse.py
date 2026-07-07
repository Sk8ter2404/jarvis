#!/usr/bin/env python3
"""Guided CALIBRATION WIZARD for the JARVIS Kinect air-mouse.

WHY THIS TOOL EXISTS
====================
The air-mouse (skills/kinect_air_mouse.py) engages when the controlling hand is
raised clearly ABOVE THE SHOULDER — the body-relative HEIGHT delta ``lift_m``
(hand_y minus a shoulder-line reference) clears an engage margin, with hysteresis
so a hand hovering at the line can't flap. The default margins (a raise of ~7 cm
to engage, a drop to ~10 cm below the shoulder to release) are chosen to fit a
"typical" seated user, but one person's relaxed-at-the-desk posture, chair height,
and comfortable pointing arc are not another's. When the defaults don't fit HIM
the result is the exact complaint that motivated this tool: engagement is
unreliable / false-triggers (a resting hand trips the gate, or a real raise fails
to take the cursor).

This wizard walks the owner through a short, guided capture of HIS OWN body at HIS
desk and fits the engage/disengage HEIGHT margins (and a comfortable reach-box) to
what it actually measured, then writes them to ``data/user_settings.json`` — which
the live skill reads every tick (via ``_reach_thresholds`` /
``_reach_box_for_virtual_desktop``) and which ``core.config`` layers over the code
defaults at startup. So the gate is tuned to the owner, not a stranger.

It is a SIBLING to skills/kinect_air_mouse.py, never a replacement: it does not
import or modify the skill. It only WRITES the same persisted keys the skill reads.

THE KEYS IT WRITES (must match what the skill reads)
====================================================
The height gate (the PRIMARY engage signal) reads these from user_settings.json
via skills.kinect_air_mouse._reach_thresholds():

    KINECT_LIFT_UP_MARGIN     lift_m (m above the shoulder) required to ENGAGE
    KINECT_LIFT_DOWN_MARGIN   lift_m to DISENGAGE  (< up, for hysteresis)

The reach-box (hand position -> desktop pixel) keys mirror the skill's module
constants REACH_HALF_W / REACH_HALF_H / REACH_CENTER_X / REACH_CENTER_Y so a
build that wires a per-user reach-box read picks them up; they are inert (but
preserved) until then:

    KINECT_REACH_HALF_W       +/- horizontal hand reach (m) mapped to screen width
    KINECT_REACH_HALF_H       +/- vertical hand reach (m) mapped to screen height
    KINECT_REACH_CENTER_X     horizontal centre of the reach-box (m)
    KINECT_REACH_CENTER_Y     vertical centre of the reach-box (m)

HEAVY / HEADLESS-SAFE
=====================
Importing this module is cheap and side-effect-free: no Kinect, no core.config,
no skill import happens at module load. The bridge + the skill's sampling helpers
are imported LAZILY inside the live-sampling entrypoint, so the CI import-guard
(and the unit test) exercise the PURE core — compute_calibration() and
merge_into_settings() — with synthetic samples and a temp file, touching no
hardware and no display.

Usage
-----
    # run the guided wizard live (owner, at the desk, Kinect enabled):
    python tools/calibrate_air_mouse.py

    # preview the computed values without writing anything:
    python tools/calibrate_air_mouse.py --dry-run

    # skip the confirmation prompt (write immediately after capture):
    python tools/calibrate_air_mouse.py --yes

    # print where the settings would be written and exit:
    python tools/calibrate_air_mouse.py --show-path
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_SETTINGS_PATH = os.path.join(_PROJECT_ROOT, "data", "user_settings.json")

# ── the persisted keys the live skill reads (keep in lockstep with
#    skills/kinect_air_mouse.py SETTING_UP_MARGIN / SETTING_DOWN_MARGIN and the
#    reach-box module constants). Written verbatim so a calibration takes effect
#    with no restart of the gate (it re-reads every tick). ────────────────────
KEY_UP_MARGIN = "KINECT_LIFT_UP_MARGIN"        # lift_m to ENGAGE (m above shoulder)
KEY_DOWN_MARGIN = "KINECT_LIFT_DOWN_MARGIN"    # lift_m to DISENGAGE (m; < up)
KEY_REACH_HALF_W = "KINECT_REACH_HALF_W"       # +/- horizontal hand reach (m)
KEY_REACH_HALF_H = "KINECT_REACH_HALF_H"       # +/- vertical hand reach (m)
KEY_REACH_CENTER_X = "KINECT_REACH_CENTER_X"   # reach-box horizontal centre (m)
KEY_REACH_CENTER_Y = "KINECT_REACH_CENTER_Y"   # reach-box vertical centre (m)

# ── capture cadence + clamps (the fit stays inside sane physical ranges so a
#    flubbed capture can never write an absurd, unusable gate). ───────────────
SAMPLES_PER_STEP = 12          # frames averaged per guided step
COUNTDOWN_SECONDS = 3          # "3.. 2.. 1.." before each capture begins
STEP_CAPTURE_SECONDS = 2.5     # how long to sample once capture begins
_MIN_TRACKED_FRAMES = 3        # need at least this many good frames to trust a step

# Where the engage / disengage margins are placed between the RESTING hand height
# and the deliberate RAISE height. The engage bar sits a good way up (so a resting
# hand clearly can't reach it) and the disengage bar a bit lower (hysteresis: a
# small sag while engaged doesn't drop the cursor).
ENGAGE_FRACTION = 0.55         # up-margin this far resting->raised
DISENGAGE_FRACTION = 0.35      # down-margin this far resting->raised (< engage)

# Absolute floors/ceilings on the fitted margins (metres, body-relative lift_m).
# A hand raised to point sits at/above the shoulder (lift ~ 0..+0.3); a resting
# hand sits well below (lift ~ -0.3..-0.5). These clamps keep the fit physical.
UP_MARGIN_MIN = 0.03           # never demand less than a 3 cm raise to engage
UP_MARGIN_MAX = 0.35           # never demand more than a 35 cm raise (unreachable)
DOWN_MARGIN_MIN = -0.30        # disengage no lower than 30 cm below the shoulder
DOWN_MARGIN_MAX = 0.20         # disengage bar must stay below the engage bar
MIN_HYSTERESIS = 0.04          # keep >= 4 cm between up and down so it can't flap

# Reach-box clamps (metres). A comfortable seated forearm arc is ~+/-0.15..0.40 m
# horizontal / ~+/-0.10..0.30 m vertical; clamp so a wild capture can't make the
# cursor hyper-sensitive (tiny box) or need whole-arm swings (huge box).
REACH_HALF_W_MIN, REACH_HALF_W_MAX = 0.12, 0.45
REACH_HALF_H_MIN, REACH_HALF_H_MAX = 0.08, 0.35
REACH_CENTER_X_MIN, REACH_CENTER_X_MAX = -0.40, 0.40
REACH_CENTER_Y_MIN, REACH_CENTER_Y_MAX = -0.20, 0.80

# Module defaults mirrored from skills/kinect_air_mouse.py so a step that fails to
# capture (or a degenerate span) falls back to a safe, usable value rather than
# emitting nonsense. Kept as plain literals so importing THIS module never drags
# in the skill / core.config.
_DEFAULT_UP_MARGIN = 0.07
_DEFAULT_DOWN_MARGIN = -0.10
_DEFAULT_REACH_HALF_W = 0.26
_DEFAULT_REACH_HALF_H = 0.16
_DEFAULT_REACH_CENTER_X = 0.0
_DEFAULT_REACH_CENTER_Y = 0.30


# ══════════════════════════════════════════════════════════════════════════
#  PURE CORE  (no sensor, no prompts, no display — unit-tested directly)
# ══════════════════════════════════════════════════════════════════════════

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp `value` into [lo, hi]. Pure."""
    return max(lo, min(hi, float(value)))


def _mean(values) -> "Optional[float]":
    """Arithmetic mean of a list of numbers, or None when empty. Pure; ignores
    None entries so a partially-tracked capture still averages the good frames."""
    xs = [float(v) for v in (values or []) if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def compute_calibration(samples: dict) -> dict:
    """Fit the air-mouse engage margins + reach-box from captured pose samples.

    PURE — no hardware, no prompts, no file I/O. The live wizard captures the
    frames and hands their averages here; the unit test feeds synthetic samples.

    `samples` is a dict with these (all optional; a missing/degenerate one falls
    back to the module default for that field):

        "rest_lift"   : float | None   # lift_m of the RESTING hand at the desk
        "raise_lift"  : float | None   # lift_m of the deliberately RAISED hand
        "reach_min_x" : float | None   # hand x at the TOP-LEFT reach extreme (m)
        "reach_max_x" : float | None   # hand x at the BOTTOM-RIGHT reach extreme
        "reach_min_y" : float | None   # hand y at the BOTTOM-RIGHT (low) extreme
        "reach_max_y" : float | None   # hand y at the TOP-LEFT (high) extreme

    Returns a flat dict keyed by the PERSISTED setting names the skill reads:
        {KINECT_LIFT_UP_MARGIN, KINECT_LIFT_DOWN_MARGIN,
         KINECT_REACH_HALF_W, KINECT_REACH_HALF_H,
         KINECT_REACH_CENTER_X, KINECT_REACH_CENTER_Y}

    GUARANTEES (asserted by the tests):
      * up_margin > down_margin ALWAYS (>= MIN_HYSTERESIS apart) — the gate can
        never be built without hysteresis, so it can't flap at the line.
      * both margins sit BETWEEN the resting height and the raise height when a
        usable resting<raise span was captured (a real raise clears the up bar; a
        resting hand sits below the down bar), so engagement fits the owner.
      * every output is CLAMPED to a physical range, so absurd input (a raise
        below the rest, a giant/tiny reach) yields a safe default-ish value, never
        a broken gate.
    """
    samples = samples or {}
    rest = samples.get("rest_lift")
    raise_ = samples.get("raise_lift")

    # ── HEIGHT margins ──────────────────────────────────────────────────────
    # Need BOTH ends AND a real upward span (raise clearly above rest) to fit;
    # otherwise keep the safe module defaults rather than emit an inverted / tiny
    # margin from a flubbed capture.
    if (rest is not None and raise_ is not None
            and (float(raise_) - float(rest)) >= 0.10):
        span = float(raise_) - float(rest)
        up_margin = float(rest) + ENGAGE_FRACTION * span
        down_margin = float(rest) + DISENGAGE_FRACTION * span
    else:
        up_margin = _DEFAULT_UP_MARGIN
        down_margin = _DEFAULT_DOWN_MARGIN

    # Clamp each into its physical range, THEN enforce the hysteresis ordering so
    # the clamps can never collapse up<=down. down is pinned strictly below up by
    # at least MIN_HYSTERESIS (and no lower than DOWN_MARGIN_MIN).
    up_margin = _clamp(up_margin, UP_MARGIN_MIN, UP_MARGIN_MAX)
    down_margin = _clamp(down_margin, DOWN_MARGIN_MIN, DOWN_MARGIN_MAX)
    if down_margin > up_margin - MIN_HYSTERESIS:
        down_margin = up_margin - MIN_HYSTERESIS
    down_margin = _clamp(down_margin, DOWN_MARGIN_MIN, DOWN_MARGIN_MAX)

    # ── REACH-BOX ───────────────────────────────────────────────────────────
    # The two reach extremes (top-left, bottom-right) give the hand's x and y
    # span; half-width/half-height are half those spans, centre is their midpoint.
    # A missing or degenerate (too-small) span falls back to the module default.
    min_x = samples.get("reach_min_x")
    max_x = samples.get("reach_max_x")
    min_y = samples.get("reach_min_y")
    max_y = samples.get("reach_max_y")

    half_w, center_x = _fit_axis(min_x, max_x,
                                 _DEFAULT_REACH_HALF_W, _DEFAULT_REACH_CENTER_X,
                                 min_span=0.06)
    half_h, center_y = _fit_axis(min_y, max_y,
                                 _DEFAULT_REACH_HALF_H, _DEFAULT_REACH_CENTER_Y,
                                 min_span=0.04)

    half_w = _clamp(half_w, REACH_HALF_W_MIN, REACH_HALF_W_MAX)
    half_h = _clamp(half_h, REACH_HALF_H_MIN, REACH_HALF_H_MAX)
    center_x = _clamp(center_x, REACH_CENTER_X_MIN, REACH_CENTER_X_MAX)
    center_y = _clamp(center_y, REACH_CENTER_Y_MIN, REACH_CENTER_Y_MAX)

    return {
        KEY_UP_MARGIN: round(up_margin, 4),
        KEY_DOWN_MARGIN: round(down_margin, 4),
        KEY_REACH_HALF_W: round(half_w, 4),
        KEY_REACH_HALF_H: round(half_h, 4),
        KEY_REACH_CENTER_X: round(center_x, 4),
        KEY_REACH_CENTER_Y: round(center_y, 4),
    }


def _fit_axis(lo, hi, default_half, default_center, *, min_span: float):
    """Fit (half_extent, center) for one reach axis from its two extremes, or
    fall back to the defaults when either extreme is missing or the |span| is too
    small to be a real reach. Robust to the two extremes arriving swapped (uses
    the absolute span). Pure."""
    if lo is None or hi is None:
        return default_half, default_center
    lo = float(lo)
    hi = float(hi)
    span = abs(hi - lo)
    if span < float(min_span):
        return default_half, default_center
    return span / 2.0, (lo + hi) / 2.0


def merge_into_settings(path: str, values: dict) -> dict:
    """Merge `values` into the JSON settings file at `path`, PRESERVING every
    existing key, and write it back atomically (temp file in the same directory,
    then os.replace). Returns the merged dict actually written.

    PURE-ISH: the only side effect is the atomic file write. No schema, no
    coercion, no hardware — it reads the existing JSON (treating a missing /
    unreadable / non-dict file as empty), overlays `values`, and writes. This is
    deliberately independent of tools.settings_window so the calibration merge
    preserves ALL existing keys verbatim (settings_window re-emits the full
    default schema; here we only touch the caller's keys) and stays trivially
    unit-testable against a temp file. NEVER partially writes a file: os.replace
    is atomic, so a reader sees either the old file or the whole new one."""
    import tempfile

    existing: dict = {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                decoded = json.loads(text)
                if isinstance(decoded, dict):
                    existing = decoded
    except (OSError, ValueError):
        existing = {}

    merged = dict(existing)
    merged.update(values or {})

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp", prefix="calib_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return merged


def format_summary(values: dict) -> str:
    """A human-readable block describing the computed calibration for the confirm
    prompt / --dry-run. Pure (string in, string out)."""
    up = values.get(KEY_UP_MARGIN)
    down = values.get(KEY_DOWN_MARGIN)
    hw = values.get(KEY_REACH_HALF_W)
    hh = values.get(KEY_REACH_HALF_H)
    cx = values.get(KEY_REACH_CENTER_X)
    cy = values.get(KEY_REACH_CENTER_Y)
    return (
        "  Computed calibration:\n"
        f"    {KEY_UP_MARGIN:<22} = {up:+.3f} m   (raise this far above the "
        "shoulder to ENGAGE)\n"
        f"    {KEY_DOWN_MARGIN:<22} = {down:+.3f} m   (drop below this to "
        "DISENGAGE)\n"
        f"    {KEY_REACH_HALF_W:<22} = {hw:.3f} m    (+/- horizontal reach -> "
        "screen width)\n"
        f"    {KEY_REACH_HALF_H:<22} = {hh:.3f} m    (+/- vertical reach -> "
        "screen height)\n"
        f"    {KEY_REACH_CENTER_X:<22} = {cx:+.3f} m\n"
        f"    {KEY_REACH_CENTER_Y:<22} = {cy:+.3f} m"
    )


# ══════════════════════════════════════════════════════════════════════════
#  LIVE WIRING  (Kinect sampling + prompts — lazy imports, live-run only)
# ══════════════════════════════════════════════════════════════════════════
# Everything below touches the sensor / stdin / stdout and is imported lazily so
# the module import (and CI / unit test) never pulls in the bridge or the skill.
# The unit test exercises the PURE core above; this path is run by the owner.

def _bridge():
    """Import + return the Kinect bridge, or None. Lazy so import stays cheap and
    headless-safe. NEVER raises."""
    try:
        from audio import kinect_bridge as kb
        return kb
    except Exception:
        return None


def _skill():
    """Import + return the air-mouse skill module (for its _hand_sample geometry),
    or None. Lazy; NEVER raises. We only READ from it (its sampling helpers) — we
    never modify it."""
    try:
        from skills import kinect_air_mouse as km
        return km
    except Exception:
        return None


def _bridge_ready(kb) -> "tuple[bool, str]":
    """(True, "") when the bridge is loaded, enabled, and a sensor is available;
    else (False, reason). NEVER raises."""
    if kb is None:
        return False, "the Kinect bridge could not be imported"
    try:
        if not kb.get_enabled():
            return False, ("the Kinect is disabled — enable it (Settings / "
                           "KINECT_ENABLED) and try again")
        ok, reason = kb.available()
        if not ok:
            return False, (reason or "the Kinect sensor is unavailable")
    except Exception as e:
        return False, f"the Kinect could not be reached ({type(e).__name__})"
    return True, ""


def _highest_hand_lift(km, kb) -> "Optional[float]":
    """The lift_m (height above the shoulder) of the highest-raised hand this
    frame, using the SAME _hand_sample geometry the live gate uses (so the mirror
    swap + tracking-state floor are identical), or None if no usable hand. NEVER
    raises."""
    try:
        left_ext, right_ext, _lg, _rg, tracked = km._hand_sample(kb)
        if not tracked:
            return None
        arms = [a for a in (left_ext, right_ext) if a is not None]
        arm = max(arms, key=lambda a: a.reach_score()) if arms else None
        if arm is None or arm.lift_m is None:
            return None
        return float(arm.lift_m)
    except Exception:
        return None


def _highest_hand_xy(km, kb) -> "Optional[tuple]":
    """The (x, y) camera-space metres of the highest-raised hand this frame (for
    the reach-box extremes), or None. Uses the controlling-hand joint from the
    same _hand_sample geometry. NEVER raises."""
    try:
        left_ext, right_ext, _lg, _rg, tracked = km._hand_sample(kb)
        if not tracked:
            return None
        arms = [a for a in (left_ext, right_ext)
                if a is not None and a.hand is not None]
        arm = max(arms, key=lambda a: a.reach_score()) if arms else None
        if arm is None or arm.hand is None or len(arm.hand) < 2:
            return None
        return float(arm.hand[0]), float(arm.hand[1])
    except Exception:
        return None


def _countdown(seconds: int, sleep_fn, out) -> None:
    """Print a simple 'N.. N-1.. ' countdown before a capture. Best-effort."""
    for n in range(int(seconds), 0, -1):
        out.write(f"    capturing in {n}...\n")
        out.flush()
        sleep_fn(1.0)


def _capture_step(km, kb, sampler, *, sleep_fn, now_fn, out,
                  seconds: float = STEP_CAPTURE_SECONDS,
                  target: int = SAMPLES_PER_STEP) -> list:
    """Sample `sampler(km, kb)` for up to `seconds`, collecting up to `target`
    non-None readings. Returns the list of readings (may be shorter than target if
    the sensor lost tracking). NEVER raises — a bad frame is simply skipped."""
    got: list = []
    deadline = now_fn() + float(seconds)
    interval = max(0.03, float(seconds) / max(1, target * 2))
    while now_fn() < deadline and len(got) < target:
        try:
            reading = sampler(km, kb)
        except Exception:
            reading = None
        if reading is not None:
            got.append(reading)
        sleep_fn(interval)
    return got


def run_wizard(argv=None, *, inp=None, out=None,
               sleep_fn=None, now_fn=None) -> int:
    """The guided CLI wizard. Returns a process exit code (0 ok, non-zero on a
    graceful failure). All I/O is injectable (inp/out/sleep/clock) so this could
    be driven in a harness, but it is NOT part of the unit-tested pure core — the
    live sensor path is import-guarded and exercised only by the owner.

    Steps:
      1. RESTING baseline — hand relaxed at the desk (the desk-level lift).
      2. ENGAGE height   — controlling hand raised to a comfortable 'take the
         cursor' height and held.
      3. Reach extents   — reach TOP-LEFT, then BOTTOM-RIGHT, of the comfortable
         pointing region (the reach-box corners).
    Then: compute, show, confirm (unless --yes), and MERGE into user_settings.json
    (unless --dry-run).
    """
    import time as _time

    inp = inp or sys.stdin
    out = out or sys.stdout
    sleep_fn = sleep_fn or _time.sleep
    now_fn = now_fn or _time.monotonic

    args = _build_parser().parse_args(argv)
    settings_path = args.path or _resolve_settings_path()

    if args.show_path:
        out.write(f"{settings_path}\n")
        return 0

    kb = _bridge()
    km = _skill()
    ready, why = _bridge_ready(kb)
    if not ready or km is None:
        if km is None and ready:
            why = "the air-mouse skill could not be imported"
        out.write("Air-mouse calibration\n")
        out.write(f"  Cannot start: {why}.\n")
        out.write("  (Connect + enable the Kinect, then run this again.)\n")
        return 1

    out.write("=" * 66 + "\n")
    out.write("  JARVIS air-mouse calibration wizard\n")
    out.write("  Tunes the raise-to-engage height + reach region to YOUR body\n")
    out.write("  and desk. Follow each prompt and hold still during capture.\n")
    out.write("=" * 66 + "\n\n")

    # ── Step 1: RESTING baseline ────────────────────────────────────────────
    out.write("STEP 1/3 - RESTING baseline\n")
    out.write("  Sit or stand where you normally use it, with your arms relaxed\n")
    out.write("  at the desk (hands NOT raised). Hold that pose.\n")
    _wait_ready(inp, out, args.yes, sleep_fn)
    _countdown(COUNTDOWN_SECONDS, sleep_fn, out)
    rest_readings = _capture_step(km, kb, _highest_hand_lift,
                                  sleep_fn=sleep_fn, now_fn=now_fn, out=out)
    if len(rest_readings) < _MIN_TRACKED_FRAMES:
        out.write("  I couldn't see your hand clearly during that step. Make sure\n")
        out.write("  you're in the Kinect's view and run the wizard again.\n")
        return 1
    rest_lift = _mean(rest_readings)
    out.write(f"  Captured resting height ({len(rest_readings)} frames).\n\n")

    # ── Step 2: ENGAGE height ───────────────────────────────────────────────
    out.write("STEP 2/3 - ENGAGE height\n")
    out.write("  Raise your controlling hand to a comfortable 'take the cursor'\n")
    out.write("  height - about where you'd point at the screen - and hold it.\n")
    _wait_ready(inp, out, args.yes, sleep_fn)
    _countdown(COUNTDOWN_SECONDS, sleep_fn, out)
    raise_readings = _capture_step(km, kb, _highest_hand_lift,
                                   sleep_fn=sleep_fn, now_fn=now_fn, out=out)
    if len(raise_readings) < _MIN_TRACKED_FRAMES:
        out.write("  I lost track of your raised hand. Please run the wizard again.\n")
        return 1
    raise_lift = _mean(raise_readings)
    out.write(f"  Captured raised height ({len(raise_readings)} frames).\n\n")

    # ── Step 3: reach extents (top-left, then bottom-right) ─────────────────
    out.write("STEP 3/3 - Comfortable reach region\n")
    out.write("  Reach to the TOP-LEFT of where you'd comfortably point, and hold.\n")
    _wait_ready(inp, out, args.yes, sleep_fn)
    _countdown(COUNTDOWN_SECONDS, sleep_fn, out)
    tl = _capture_step(km, kb, _highest_hand_xy,
                       sleep_fn=sleep_fn, now_fn=now_fn, out=out)
    out.write("  Now reach to the BOTTOM-RIGHT of that region, and hold.\n")
    _wait_ready(inp, out, args.yes, sleep_fn)
    _countdown(COUNTDOWN_SECONDS, sleep_fn, out)
    br = _capture_step(km, kb, _highest_hand_xy,
                       sleep_fn=sleep_fn, now_fn=now_fn, out=out)

    # Top-left = smaller x, larger y (camera y is UP); bottom-right = larger x,
    # smaller y. Average each corner's frames; a missing corner leaves those axes
    # to the default via compute_calibration's fallback.
    tl_x = _mean([p[0] for p in tl]) if tl else None
    tl_y = _mean([p[1] for p in tl]) if tl else None
    br_x = _mean([p[0] for p in br]) if br else None
    br_y = _mean([p[1] for p in br]) if br else None
    out.write("\n")

    samples = {
        "rest_lift": rest_lift,
        "raise_lift": raise_lift,
        "reach_min_x": tl_x, "reach_max_x": br_x,
        "reach_min_y": br_y, "reach_max_y": tl_y,
    }
    values = compute_calibration(samples)

    out.write(format_summary(values) + "\n\n")

    if args.dry_run:
        out.write("  --dry-run: nothing was written. Re-run without --dry-run to\n")
        out.write(f"  save these to {settings_path}\n")
        return 0

    if not args.yes:
        out.write(f"  Write these to {settings_path}? [y/N] ")
        out.flush()
        try:
            answer = inp.readline().strip().lower()
        except Exception:
            answer = ""
        if answer not in ("y", "yes"):
            out.write("  Cancelled - nothing was written.\n")
            return 0

    try:
        merge_into_settings(settings_path, values)
    except Exception as e:
        out.write(f"  Failed to write settings: {type(e).__name__}: {e}\n")
        return 1
    out.write("  Saved. The air-mouse reads these live - no restart needed.\n")
    out.write("  Raise a hand above your shoulder to take the cursor; lower it to\n")
    out.write("  release. Re-run this wizard any time it doesn't feel right.\n")
    return 0


def _wait_ready(inp, out, skip: bool, sleep_fn) -> None:
    """Pause until the owner presses Enter (or, with --yes, a short auto-delay so
    they can get into pose). Best-effort; a non-interactive stdin just proceeds."""
    if skip:
        out.write("    (--yes) getting ready...\n")
        out.flush()
        sleep_fn(1.5)
        return
    out.write("    Press Enter when you're in position...")
    out.flush()
    try:
        inp.readline()
    except Exception:
        pass


def _resolve_settings_path() -> str:
    """The settings file to write. Honours JARVIS_SETTINGS_PATH (the same override
    tests + the skill's settings reader use) so a test / redirect never touches the
    owner's real file; else the canonical data/user_settings.json."""
    override = (os.environ.get("JARVIS_SETTINGS_PATH") or "").strip()
    return override or _DEFAULT_SETTINGS_PATH


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Guided calibration wizard for the JARVIS Kinect air-mouse "
                    "(tunes the raise-to-engage height + reach region to you).")
    p.add_argument("--dry-run", action="store_true",
                   help="capture + compute + print, but write nothing.")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt (write after capture); also "
                        "auto-advances the pose prompts.")
    p.add_argument("--path", default=None,
                   help="settings file to write (default: data/user_settings.json, "
                        "or $JARVIS_SETTINGS_PATH).")
    p.add_argument("--show-path", action="store_true",
                   help="print the settings path that would be written, then exit.")
    return p


def main(argv=None) -> int:
    try:
        return run_wizard(argv)
    except KeyboardInterrupt:
        sys.stdout.write("\n  Cancelled.\n")
        return 130


if __name__ == "__main__":   # pragma: no cover - CLI entry, not CI-tested
    raise SystemExit(main())
