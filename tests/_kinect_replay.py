"""Offline REPLAY harness for the Kinect hand pipeline (2026-10-04).

Feeds numeric body traces - synthetic geometry, optionally carrying the
ANONYMISED jitter / tracking-state / grip-state streams recorded at the
owner's desk (tests/_kinect_replay_desk_noise.csv) - through the REAL code,
end to end, with simulated time:

    fake pykinect body  →  audio.kinect_bridge._parse_body_frame   (real parser)
                        →  audio.kinect_bridge.publish_body_frame   (real shared
                           stabiliser, exactly as the body pump runs it)
                        →  skills/kinect_two_hand._poll_once        (real poller)
                        →  skills/kinect_air_mouse._poll_once       (real poller)
                        →  skills/kinect_gestures._poll_once        (real poller)

Only the side effects are replaced (cursor moves / button presses / window
moves are RECORDED, overlay files and subprocesses are blocked, the real
desktop - foreground window, real-input yield, monitor layout - is pinned).
Pollers free-run on their own period against 30 Hz frames, like the live
threads, so a poll may see the same frame twice or skip one.

shared=False replays the same trace through the legacy per-poll path (a bridge
facade without get_tracked_frame): the air-mouse / two-hand behaviour
origin/main shipped (the gesture side-lock and release mute stay on). For the
complete before-picture run this harness on an origin/main tree, where the
bridge has no publish_body_frame and every consumer is the shipped code.

A replay leaves no state behind: run() resets what the skills publish to
each other (engaged flag, two-hand heartbeat), clears the bridge's body cache
(its frames carry SIMULATED stamps that a short-uptime machine would serve as
fresh) and unload_modules() puts back the sys.modules entries load_modules()
replaced. The two-hand heartbeat runs on the replay's simulated clock too.

Nothing here touches a sensor, the mouse, a window, the network or the
settings file. Stdlib + unittest.mock only.
"""
from __future__ import annotations

import contextlib
import csv
import math
import os
import types
from typing import Optional
from unittest import mock

FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_kinect_replay_desk_noise.csv")
FRAME_S = 1.0 / 30.0

# A GENERIC adult skeleton in camera space (metres; x sensor-right, y up, z away
# from the sensor), seated ~1.2 m out. NOT the owner's measurements.
SPINE_SHOULDER = (0.0, 0.30, 1.20)
SKELETON = {
    "spine_base": (0.0, -0.20, 1.25),
    "spine_mid": (0.0, 0.05, 1.22),
    "neck": (0.0, 0.37, 1.20),
    "head": (0.0, 0.52, 1.19),
    "spine_shoulder": SPINE_SHOULDER,
    "shoulder_left": (-0.18, 0.28, 1.20),
    "shoulder_right": (0.18, 0.28, 1.20),
    "hip_left": (-0.10, -0.25, 1.25),
    "hip_right": (0.10, -0.25, 1.25),
}
DESK_Y = 0.0                  # a hand resting on the desk: lift ≈ -0.30 m
RAISED_FORWARD_M = 0.30       # a raised, reaching hand sits this far in front
# The live virtual desktop (from the 2026-10-04 log): 7680 x 2880 px.
DESKTOP = (0, 0, 7680, 2880)


def lift_y(lift: float) -> float:
    """Hand y for a given lift above the shoulder line."""
    return SPINE_SHOULDER[1] + float(lift)


def raised_hand(side: str, lift: float, x: Optional[float] = None,
                forward: float = RAISED_FORWARD_M) -> tuple:
    sx = -0.15 if side == "left" else 0.15
    return (sx if x is None else x, lift_y(lift), SPINE_SHOULDER[2] - forward)


def desk_hand(side: str) -> tuple:
    return (-0.20 if side == "left" else 0.20, DESK_Y, 1.05)


# ─── the anonymised desk recording ───────────────────────────────────────────
def load_fixture(path: str = FIXTURE_PATH) -> list:
    """Rows of the anonymised recording as dicts of ints."""
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    rows = list(csv.DictReader(lines))
    return [{k: int(v) for k, v in r.items()} for r in rows]


# ─── one synthetic frame ─────────────────────────────────────────────────────
class Frame:
    """One body frame: dt since the previous frame, the hand targets, the SDK
    hand states/confidences and optional per-joint tracking states / noise."""

    __slots__ = ("dt", "left", "right", "hl", "hlc", "hr", "hrc", "states",
                 "noise", "body_id", "extra")

    def __init__(self, left, right, *, dt=FRAME_S, hl=2, hlc=1, hr=2, hrc=1,
                 states=None, noise=None, body_id=1, extra=None):
        self.dt = float(dt)
        self.left = left
        self.right = right
        self.hl, self.hlc, self.hr, self.hrc = hl, hlc, hr, hrc
        self.states = states or {}
        self.noise = noise or {}
        self.body_id = body_id
        self.extra = extra or []      # additional (body_id, distance_shift) bodies


def _arm(side: str, hand: tuple) -> dict:
    """hand / wrist / elbow / tip / thumb joints for one arm."""
    sh = SKELETON[f"shoulder_{side}"]
    hx, hy, hz = hand
    raised = hy > SKELETON[f"shoulder_{side}"][1] - 0.05
    if raised:
        wrist = (hx, hy - 0.06, hz + 0.02)
        elbow = ((sh[0] + hx) / 2.0, (sh[1] + hy) / 2.0 - 0.02, (sh[2] + hz) / 2.0)
    else:
        wrist = (hx, hy + 0.01, hz + 0.07)
        elbow = (sh[0], hy + 0.15, hz + 0.12)
    tip = (hx, hy + 0.05, hz - 0.01)
    thumb = (hx + (0.03 if side == "left" else -0.03), hy + 0.02, hz)
    return {f"hand_{side}": hand, f"wrist_{side}": wrist,
            f"elbow_{side}": elbow, f"hand_tip_{side}": tip,
            f"thumb_{side}": thumb}


def frame_joints(fr: Frame) -> dict:
    """Bridge-named joints {name: (x, y, z, state)} for a Frame."""
    pos = dict(SKELETON)
    pos.update(_arm("left", fr.left))
    pos.update(_arm("right", fr.right))
    out = {}
    for name, p in pos.items():
        n = fr.noise.get(name, (0.0, 0.0, 0.0))
        st = int(fr.states.get(name, 2))
        out[name] = (p[0] + n[0], p[1] + n[1], p[2] + n[2], st)
    for leg in ("knee_left", "ankle_left", "foot_left",
                "knee_right", "ankle_right", "foot_right"):
        out[leg] = (0.0, 0.0, 0.0, 0)          # under the desk: NotTracked
    return out


def _fake_body(joints: dict, fr: Frame, body_id: int, z_shift: float = 0.0):
    from audio import kinect_bridge as kb
    jl = []
    for name in kb._JOINT_NAMES:
        x, y, z, st = joints.get(name, (0.0, 0.0, 0.0, 0))
        if st and z_shift:
            z += z_shift
        jl.append(types.SimpleNamespace(
            Position=types.SimpleNamespace(x=x, y=y, z=z), TrackingState=st))
    return types.SimpleNamespace(
        is_tracked=True, tracking_id=body_id, joints=jl,
        hand_left_state=fr.hl, hand_right_state=fr.hr,
        hand_left_confidence=fr.hlc, hand_right_confidence=fr.hrc)


def parse(fr: Frame) -> list:
    """The REAL bridge parser over the fake frame (ghost gate, confidence,
    distance, facing - all live code). fr.body_id None = the owner is not in
    this frame. Each fr.extra entry is another body: (id, dz) copies the
    owner's pose `dz` metres deeper; (id, dz, left, right) gives it its own
    hand positions (open hands, everything Tracked)."""
    from audio import kinect_bridge as kb
    bodies = []
    joints = frame_joints(fr)
    if fr.body_id is not None:
        bodies.append(_fake_body(joints, fr, fr.body_id))
    for ex in fr.extra:
        if len(ex) == 2:
            bodies.append(_fake_body(joints, fr, ex[0], ex[1]))
        else:
            bid, dz, left, right = ex
            other = Frame(left, right)
            bodies.append(_fake_body(frame_joints(other), other, bid, dz))
    return kb._parse_body_frame(types.SimpleNamespace(bodies=bodies))


def recorded_noise(row: dict) -> dict:
    """Jitter residuals of one fixture row (0.1 mm ints) as metres per joint."""
    out = {}
    for joint, name in (("HandLeft", "hand_left"), ("HandRight", "hand_right"),
                        ("WristLeft", "wrist_left"), ("WristRight", "wrist_right")):
        out[name] = tuple(row[f"{joint}_{a}"] / 10000.0 for a in "xyz")
    return out


def recorded_states(row: dict, sides=("left", "right")) -> dict:
    """Tracking-state ints of one fixture row for the given arms."""
    m = {"left": ("HandLeft", "WristLeft", "ElbowLeft", "ShoulderLeft"),
         "right": ("HandRight", "WristRight", "ElbowRight", "ShoulderRight")}
    out = {"spine_shoulder": row["ts_SpineShoulder"]}
    for s in sides:
        h, w, e, sh = m[s]
        out[f"hand_{s}"] = row["ts_" + h]
        out[f"wrist_{s}"] = row["ts_" + w]
        out[f"elbow_{s}"] = row["ts_" + e]
        out[f"shoulder_{s}"] = row["ts_" + sh]
    return out


# ─── the pipeline driver ────────────────────────────────────────────────────
class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


class Result:
    """Per-poll timeline of what the real consumers did."""

    def __init__(self):
        self.polls: list = []      # dicts: t, engaged, hand, cursor, two, phase
        self.buttons: list = []    # (t, action, button)
        self.moves: list = []      # (t, x, y)
        self.windows: list = []    # (t, rect)
        self.gestures: list = []   # (t, name)
        self.snaps: list = []      # (t, snapshot) per frame (shared only)
        self.duration = 0.0

    # metrics -----------------------------------------------------------------
    def edges(self, key: str, value=True) -> list:
        """Times at which polls[key] became `value` (rising edges)."""
        out, prev = [], None
        for p in self.polls:
            cur = p[key] == value
            if cur and prev is False:
                out.append(p["t"])
            prev = cur
        return out

    def changes(self, key: str) -> list:
        """Times at which polls[key] changed value (any direction)."""
        out = []
        for a, b in zip(self.polls, self.polls[1:]):
            if a[key] != b[key]:
                out.append(b["t"])
        return out

    def per_minute(self, n: int) -> float:
        return 60.0 * n / max(1e-9, self.duration)

    def presses(self) -> list:
        return [(t, b) for t, a, b in self.buttons if a == "down"]


@contextlib.contextmanager
def _patched(am, th, gs, facade, clock):
    """Every side effect blocked / recorded; the real desktop pinned."""
    rec_moves, rec_buttons = [], []
    reach = am.ReachBox(DESKTOP[2], DESKTOP[3], origin_x=DESKTOP[0],
                        origin_y=DESKTOP[1])
    patches = [
        mock.patch.object(am, "_bridge", lambda: facade),
        mock.patch.object(am, "_set_cursor_pos",
                          lambda x, y: rec_moves.append((clock.t, x, y)) or True),
        mock.patch.object(am, "_mouse_button",
                          lambda a, b="left": rec_buttons.append((clock.t, a, b)) or True),
        mock.patch.object(am, "_publish_overlay_state", lambda *a, **k: None),
        mock.patch.object(am, "_clear_overlay_state", lambda *a, **k: None),
        mock.patch.object(am, "_spawn_overlay", lambda *a, **k: None),
        mock.patch.object(am, "_overlay_alive", lambda: True),
        mock.patch.object(am, "_atomic_write_state", lambda *a, **k: None),
        mock.patch.object(am, "_install_yield_watcher", lambda *a, **k: False),
        mock.patch.object(am, "real_input_recent", lambda *a, **k: False),
        mock.patch.object(am, "_per_app_disabled", lambda: False),
        mock.patch.object(am, "_air_mouse_enabled", lambda: True),
        mock.patch.object(am, "_is_staging", lambda: False),
        mock.patch.object(am, "_maybe_debug_log", lambda *a, **k: False),
        mock.patch.object(am, "_reach_box_for_virtual_desktop",
                          lambda *a, **k: reach),
        mock.patch.object(am, "_cached_virtual_bounds", lambda *a, **k: DESKTOP),
        mock.patch.object(am, "_reach_thresholds", lambda: {
            "up_margin": am.AIR_MOUSE_ENGAGE_UP_MARGIN_M,
            "down_margin": am.AIR_MOUSE_ENGAGE_DOWN_MARGIN_M,
            "ratio_engage": 0.0, "ratio_disengage": 0.0, "fwd_engage": 0.0,
            "fwd_disengage": 0.0, "straight_engage": 0.0,
            "straight_disengage": 0.0}),
        mock.patch.object(th, "_air_mouse_mod", lambda: am),
        mock.patch.object(th, "_is_staging", lambda: False),
        mock.patch.object(th, "_publish_two_hand_overlay", lambda *a, **k: None),
        mock.patch.object(th, "_ensure_overlay_alive", lambda *a, **k: None),
        mock.patch.object(gs, "_bridge", lambda: facade),
        mock.patch.object(gs, "_gestures_enabled", lambda: True),
        mock.patch.object(gs, "_is_staging", lambda: False),
        mock.patch.object(gs, "_set_last_gesture", lambda *a, **k: None),
        # The gesture skill's release mute runs on time.monotonic: simulate it.
        mock.patch.object(gs, "time", types.SimpleNamespace(
            monotonic=clock, time=clock, sleep=lambda s: None)),
    ]
    if hasattr(am, "_heartbeat_clock"):
        # The two-hand -> air-mouse heartbeat TTL on the SIMULATED clock, so a
        # replay neither mixes clocks nor flakes on a descheduled process.
        patches.append(mock.patch.object(am, "_heartbeat_clock", [clock]))
    for p in patches:
        p.start()
    try:
        yield rec_moves, rec_buttons, reach
    finally:
        for p in reversed(patches):
            p.stop()


_SKILL_KEYS = ("skill_kinect_air_mouse", "skill_kinect_two_hand",
               "skill_kinect_gestures")
_saved_skill_modules: list = []


def load_modules():
    """Fresh, isolated instances of the three skills (no poller threads). They
    register under their live sys.modules names (the pollers find each other
    there); call unload_modules() afterwards to put back whatever was there."""
    import sys
    from tests._skill_harness import load_skill_isolated
    _saved_skill_modules.append({k: sys.modules.get(k) for k in _SKILL_KEYS})
    am, _ = load_skill_isolated("kinect_air_mouse", register=False)
    th, _ = load_skill_isolated("kinect_two_hand", register=False)
    gs, _ = load_skill_isolated("kinect_gestures", register=False)
    return am, th, gs


def unload_modules() -> None:
    """Undo the newest load_modules(): restore the previous sys.modules entries
    so no later test reads a replay's skill module (or its state)."""
    import sys
    if not _saved_skill_modules:
        return
    for k, v in _saved_skill_modules.pop().items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _reset_published_state(am, th, gs) -> None:
    """Module state a replay leaves behind that OTHER readers consult (the
    gesture skill reads the air-mouse's engaged flag and two-hand heartbeat):
    back to idle, so a replay can never leak into a later test."""
    for fn in (lambda: am._set_air_mouse_state(False, "open", None),
               lambda: am.set_two_hand_active(False),
               lambda: th._grab_hwnd.__setitem__(0, 0),
               lambda: gs._muted_until.__setitem__(0, 0.0)):
        try:
            fn()
        except Exception:
            pass


def run(frames: list, *, shared: bool = True, poll_hz: float = 27.0,
        gesture_hz: float = 18.0, modules=None) -> Result:
    """Replay `frames` through the real pipeline. Pollers free-run at
    `poll_hz` (air-mouse + two-hand; the live sleep(1/30) plus work) and
    `gesture_hz`, each seeing the latest published frame at its poll time."""
    from audio import kinect_bridge as kb
    from audio import kinect_gestures as kgr
    own = modules is None
    am, th, gs = modules or load_modules()
    clock = Clock()
    t0 = clock.t
    state = {"bodies": []}
    facade = types.SimpleNamespace(
        get_enabled=lambda: True, available=lambda: (True, ""),
        get_bodies=lambda: list(state["bodies"]),
        arm_extension=kb.arm_extension)
    if shared:
        facade.get_tracked_frame = lambda: kb.get_tracked_frame(now=clock.t)
        facade.set_lift_margins = kb.set_lift_margins
        kb.reset_tracking()
    res = Result()
    win_rect = th.Rect(800, 400, 1600, 1000)
    try:
        _replay(frames, res, am, th, gs, kb, kgr, facade, state, clock, t0,
                win_rect, poll_hz, gesture_hz, shared)
    finally:
        _reset_published_state(am, th, gs)
        if shared:
            kb.reset_tracking()
        clear_body_cache()
        if own:
            unload_modules()
    return res


def clear_body_cache() -> None:
    """Empty the bridge's shared body cache. A replay's frames carry simulated
    stamps (t ~ 1000 s); on a machine up for less than that, get_bodies() would
    serve them as fresh to whatever runs next. NEVER raises."""
    try:
        from audio import kinect_bridge as kb
        with kb._body_cache_lock:
            kb._body_cache[0] = None
            kb._body_cache_at[0] = 0.0
    except Exception:
        pass


def _replay(frames, res, am, th, gs, kb, kgr, facade, state, clock, t0,
            win_rect, poll_hz, gesture_hz, shared) -> None:

    def fg():
        return (4242, win_rect)

    def swp(hwnd, rect):
        res.windows.append((clock.t, rect))
        return True

    with _patched(am, th, gs, facade, clock) as (moves, buttons, reach):
        nc = getattr(am, "_new_controller", None)
        ctrl = (nc(facade, reach=reach) if callable(nc)
                else am.AirMouseController(reach))
        ctrl._clock = clock
        th_ctrl = th.TwoHandController(clock=clock)
        rec = kgr.GestureRecognizer(now_fn=clock)
        # Event times.
        ft, frame_times = t0, []
        for fr in frames:
            ft += fr.dt
            frame_times.append(ft)
        end = frame_times[-1] if frame_times else t0
        poll_p, gest_p = 1.0 / poll_hz, 1.0 / gesture_hz
        next_poll, next_gest = t0 + 0.011, t0 + 0.023
        fi = 0
        while True:
            nxt = min(frame_times[fi] if fi < len(frames) else math.inf,
                      next_poll, next_gest)
            if nxt == math.inf or nxt > end + 0.2:
                break
            clock.t = nxt
            if fi < len(frames) and frame_times[fi] == nxt:
                bodies = parse(frames[fi])
                state["bodies"] = bodies
                if shared:
                    kb.publish_body_frame(bodies, now=nxt)
                    res.snaps.append((nxt, kb.get_tracked_frame(now=nxt)))
                fi += 1
                continue
            if next_poll == nxt:
                d2 = th._poll_once(th_ctrl, foreground_target=fg,
                                   set_window_pos=swp)
                d = am._poll_once(ctrl, facade)
                res.polls.append({
                    "t": nxt, "engaged": bool(ctrl.engaged), "hand": ctrl.hand,
                    "cursor": d.cursor if d is not None else None,
                    "two": bool(d2.active) if d2 is not None else False,
                    "phase": d2.phase if d2 is not None else "idle",
                    "why": getattr(ctrl, "last_release_reason", "")})
                next_poll += poll_p
                continue
            g = gs._poll_once(rec, None)
            if g:
                res.gestures.append((nxt, g))
            next_gest += gest_p
        res.moves = list(moves)
        res.buttons = list(buttons)
    res.duration = max(1e-9, end - t0)


# ─── scenario builders (synthetic geometry; recorded noise where noted) ─────
def rest_noise(rows: list, src: str = "left") -> list:
    """Per-frame hand + wrist jitter (metres) from the recording's AT-REST
    frames (still_<src> = 1: that hand and wrist moved < 15 mm over 1 s), in
    recorded order, WITH each joint's recorded TrackingState of that frame
    ("hand_state" / "wrist_state"). The hand residuals include the frames the
    SDK only Inferred the hand (wider excursions); a replay that injects them
    must label those frames Inferred too (scenario_raise_and_hold's
    noise_states), or it hands a Tracked-labelled guess to the pipeline - which
    flatters the wrist-first stabiliser (2026-10-04 review)."""
    J = "Left" if src == "left" else "Right"
    out = []
    for r in rows:
        if r.get("still_" + src):
            out.append({"hand": tuple(r[f"Hand{J}_{a}"] / 10000.0 for a in "xyz"),
                        "wrist": tuple(r[f"Wrist{J}_{a}"] / 10000.0 for a in "xyz"),
                        "hand_state": r["ts_Hand" + J],
                        "wrist_state": r["ts_Wrist" + J]})
    return out


def iid_noise(n: int, sd_m: float = 0.003, seed: int = 11) -> list:
    """`n` frames of independent Gaussian jitter (sd_m metres per axis), the
    SAME distribution on the hand and on the wrist - so a jitter comparison
    measures the filter, not which joint the pipeline reads."""
    import random
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        out.append({"hand": tuple(rnd.gauss(0.0, sd_m) for _ in range(3)),
                    "wrist": tuple(rnd.gauss(0.0, sd_m) for _ in range(3))})
    return out


SEATED = {"spine_base": 1, "hip_left": 1, "hip_right": 1}   # desk hides them


def scenario_desk(rows: list) -> list:
    """Seated, both hands resting at the desk, carrying the RECORDED noise,
    joint tracking states and SDK grip states/confidences, frame for frame."""
    out = []
    for r in rows:
        states = dict(SEATED)
        states.update(recorded_states(r))
        out.append(Frame(desk_hand("left"), desk_hand("right"),
                         dt=r["dtf"] * FRAME_S,
                         hl=r["hl_state"], hlc=r["hl_conf"],
                         hr=r["hr_state"], hrc=r["hr_conf"],
                         states=states, noise=recorded_noise(r)))
    return out


def scenario_raise_and_hold(rows: Optional[list] = None, *, seconds=8.0,
                            side="right", lift=0.25, grip=2, conf=1,
                            tracking_from: Optional[str] = None,
                            noise: Optional[list] = None,
                            noise_states: bool = False) -> list:
    """One hand raised (open palm, reaching) and held still; the other at the
    desk. With `rows`, the raised hand + wrist carry the recorded AT-REST
    jitter (rest_noise) - or pass `noise` (e.g. iid_noise) explicitly.
    noise_states=True labels each injected frame with the recorded hand/wrist
    TrackingState of that same rest frame. tracking_from="left"/"right" instead
    gives them the recorded TrackingState stream of that recorded hand (the
    right one toggled Tracked/Inferred 86 times in 30 s)."""
    out = []
    n = int(seconds * 30)
    other = "left" if side == "right" else "right"
    noise_seq = noise if noise is not None else (rest_noise(rows) if rows else [])
    for i in range(n):
        hand = raised_hand(side, lift)
        nzd, states = {}, {}
        if noise_seq:
            nz = noise_seq[i % len(noise_seq)]
            nzd = {f"hand_{side}": nz["hand"], f"wrist_{side}": nz["wrist"]}
            if noise_states and "hand_state" in nz:
                states = {f"hand_{side}": nz["hand_state"],
                          f"wrist_{side}": nz["wrist_state"]}
        if rows and tracking_from:
            r = rows[i % len(rows)]
            J = "Left" if tracking_from == "left" else "Right"
            states = {f"hand_{side}": r["ts_Hand" + J],
                      f"wrist_{side}": r["ts_Wrist" + J]}
        kw = {"hr": grip, "hrc": conf} if side == "right" else {"hl": grip, "hlc": conf}
        left = hand if side == "left" else desk_hand(other)
        right = hand if side == "right" else desk_hand(other)
        out.append(Frame(left, right, states=states, noise=nzd, **kw))
    return out


def scenario_second_hand_hover(*, seconds=20.0, hover_lift=0.07, amp=0.035,
                               hz=0.7, rows: Optional[list] = None) -> list:
    """The RIGHT hand drives (raised +0.25, open). The LEFT hand hovers AT the
    engage line (+0.07 ± amp, sinusoid) - near-threshold, the shape that flapped
    two-hand mode in the 2026-10-04 log. With `rows`, the LEFT hand and wrist
    carry the RECORDED flickery tracking states of the recording's right hand
    (Tracked/Inferred toggling, 86 Inferred runs in 30 s, 62 of them <= 3
    frames) - the per-frame "both hands fully Tracked" test's worst case."""
    out = []
    for i in range(int(seconds * 30)):
        t = i * FRAME_S
        ll = hover_lift + amp * math.sin(2 * math.pi * hz * t)
        states = {}
        if rows:
            r = rows[i % len(rows)]
            states = {"hand_left": r["ts_HandRight"],
                      "wrist_left": r["ts_WristRight"]}
        out.append(Frame(raised_hand("left", ll, forward=0.15),
                         raised_hand("right", 0.25), states=states))
    return out


def scenario_grip_flickers(*, seconds=12.0, flickers=((1, 1), (2, 1), (3, 1),
                                                      (2, 0), (4, 0), (6, 0)),
                           every_s=1.0, real_grip_at=None,
                           real_grip_s=0.6) -> list:
    """RIGHT hand raised + engaged with an open palm; every `every_s` a CLOSED
    flicker of (frames, confidence) - High (1) or Low (0). Optionally one real,
    High-confidence grip of `real_grip_s` starting at `real_grip_at`."""
    n = int(seconds * 30)
    hr = [2] * n
    hrc = [1] * n
    k = int(1.5 * 30)                       # let it engage first
    for frames, conf in flickers:
        for j in range(frames):
            if k + j < n:
                hr[k + j], hrc[k + j] = 3, conf
        k += int(every_s * 30)
    if real_grip_at is not None:
        a = int(real_grip_at * 30)
        for j in range(int(real_grip_s * 30)):
            if a + j < n:
                hr[a + j], hrc[a + j] = 3, 1
    return [Frame(desk_hand("left"), raised_hand("right", 0.25), hr=hr[i],
                  hrc=hrc[i]) for i in range(n)]


def scenario_two_hand_raise(*, before=1.0, rise=0.3, hold=3.0, fists_after=0.5,
                            after=1.5) -> list:
    """Both hands rise together from the desk to +0.25 over `rise`, are held
    (fists `fists_after` into the hold), then drop back."""
    out = []
    total = before + rise + hold + rise + after
    for i in range(int(total * 30)):
        t = i * FRAME_S
        if t < before:
            f = 0.0
        elif t < before + rise:
            f = (t - before) / rise
        elif t < before + rise + hold:
            f = 1.0
        elif t < before + rise + hold + rise:
            f = 1.0 - (t - before - rise - hold) / rise
        else:
            f = 0.0
        g = 3 if (before + rise + fists_after) <= t < (before + rise + hold) else 2

        def hand(side):
            d, r = desk_hand(side), raised_hand(side, 0.25)
            return tuple(d[k] + f * (r[k] - d[k]) for k in range(3))
        out.append(Frame(hand("left"), hand("right"), hl=g, hr=g))
    return out


def scenario_hand_switch(*, drive=3.0, switch=0.5, after=3.0) -> list:
    """RIGHT drives raised (+0.20); then the LEFT rises to +0.40 while the
    RIGHT lowers to the desk over `switch` seconds; the LEFT then drives."""
    out = []
    for i in range(int((drive + switch + after) * 30)):
        t = i * FRAME_S
        f = min(1.0, max(0.0, (t - drive) / switch))
        dl, ul = desk_hand("left"), raised_hand("left", 0.40)
        ur, dr = raised_hand("right", 0.20), desk_hand("right")
        left = tuple(dl[k] + f * (ul[k] - dl[k]) for k in range(3))
        right = tuple(ur[k] + f * (dr[k] - ur[k]) for k in range(3))
        out.append(Frame(left, right))
    return out


def scenario_second_hand_goes_high(*, drive=2.0, after=2.0) -> list:
    """RIGHT drives (raised +0.20). Then the LEFT rises (4 frames) to +0.50 -
    well past the right - and stays: two-hand mode will take over, and until
    it does the cursor must not be yanked across to the left hand."""
    out = []
    seq = [0.0] * int(drive * 30)
    seq += [k / 4.0 for k in range(1, 5)] + [1.0] * int(after * 30)
    for f in seq:
        d, u = desk_hand("left"), raised_hand("left", 0.50)
        left = tuple(d[k] + f * (u[k] - d[k]) for k in range(3))
        out.append(Frame(left, raised_hand("right", 0.20)))
    return out


def scenario_both_hands_alternating(*, seconds=6.0, period_frames=1) -> list:
    """Both hands held still at chest height (below the raise line, above the
    waist), 0.5 m apart, the HIGHER one alternating every `period_frames` frames
    by 3 cm (depth noise on two hands at the same height) - nobody waves or
    swipes."""
    out = []
    for i in range(int(seconds * 30)):
        hi, lo = -0.10, -0.13
        ll, rl = (hi, lo) if (i // period_frames) % 2 == 0 else (lo, hi)
        out.append(Frame(raised_hand("left", ll, x=-0.25, forward=0.20),
                         raised_hand("right", rl, x=0.25, forward=0.20)))
    return out


def scenario_passer_by(*, seconds=6.0, at=3.0, dur=0.5, dz=-0.4) -> list:
    """The owner drives with the RIGHT hand raised; a second person (same
    pose, `dz` metres NEARER the sensor) is in view from `at` for `dur`."""
    out = []
    for i in range(int(seconds * 30)):
        t = i * FRAME_S
        extra = [(2, dz)] if at <= t < at + dur else []
        out.append(Frame(desk_hand("left"), raised_hand("right", 0.25),
                         extra=extra))
    return out


def scenario_reach(*, hold=2.0, distance=0.30, move_s=0.6, side="right",
                   lift=0.20, repeats=3, rows: Optional[list] = None) -> list:
    """An engaged hand moving horizontally along a minimum-jerk path of
    `distance` metres in `move_s`, back and forth, with holds in between.
    Returns frames; the true hand x per frame is Frame.left/right[0]."""
    out = []
    x0 = 0.15 if side == "right" else -0.15
    path = []
    for _ in range(int(hold * 30)):
        path.append(x0)
    for rep in range(repeats):
        a, b = (x0, x0 - distance) if rep % 2 == 0 else (x0 - distance, x0)
        steps = max(2, int(move_s * 30))
        for i in range(1, steps + 1):
            s = i / steps
            m = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
            path.append(a + (b - a) * m)
        for _ in range(int(hold * 30)):
            path.append(b)
    noise_seq = rest_noise(rows) if rows else []
    for i, x in enumerate(path):
        hand = raised_hand(side, lift, x=x)
        noise = {}
        if noise_seq:
            nz = noise_seq[i % len(noise_seq)]
            noise = {f"hand_{side}": nz["hand"], f"wrist_{side}": nz["wrist"]}
        other = desk_hand("left" if side == "right" else "right")
        out.append(Frame(other if side == "right" else hand,
                         hand if side == "right" else other, noise=noise))
    return out


# ─── 2026-10-04 review scenarios (each one reproduced a finding) ────────────
def _timeline(seconds: float, fn) -> list:
    """Frames at 30 Hz: fn(t) -> Frame for each frame time t."""
    return [fn(i * FRAME_S) for i in range(int(seconds * 30))]


def scenario_offhand_fist_at_desk(*, seconds=4.0, at=2.0, dur=0.5, conf=1):
    """The RIGHT hand drives (raised, open); the LEFT hand rests on the desk,
    Tracked, and closes (High confidence) from `at` for `dur` - holding a cup,
    a phone, the real mouse."""
    def fn(t):
        return Frame(desk_hand("left"), raised_hand("right", 0.25),
                     hl=3 if at <= t < at + dur else 2, hlc=conf)
    return _timeline(seconds, fn)


def scenario_drag_then_open(*, seconds=6.0, after=2, after_conf=0,
                            move_m=0.10):
    """RIGHT hand engaged; a deliberate High-confidence fist 2.0-2.5 s, then
    the SDK reads `after` (2 = Open, 0 = Unknown) at confidence `after_conf`
    for 2 s while the hand moves `move_m` sideways (the owner thinks he let
    go), then Open/High."""
    def fn(t):
        if 2.0 <= t < 2.5:
            hr, hrc = 3, 1
        elif 2.5 <= t < 4.5:
            hr, hrc = after, after_conf
        else:
            hr, hrc = 2, 1
        x = 0.15 - move_m * min(1.0, max(0.0, (t - 2.6) / 1.0))
        return Frame(desk_hand("left"), raised_hand("right", 0.25, x=x),
                     hr=hr, hrc=hrc)
    return _timeline(seconds, fn)


def scenario_reach_with_hand_state(*, seconds=4.0, state=4, conf=1,
                                   lift=0.15):
    """The RIGHT hand rises to `lift` at 0.5 s and is held, reaching, with the
    SDK reading `state` (4 = Lasso / pointing, 3 = Closed, ...) at `conf`."""
    def fn(t):
        up = t >= 0.5
        return Frame(desk_hand("left"),
                     raised_hand("right", lift) if up else desk_hand("right"),
                     hr=state if up else 2, hrc=conf)
    return _timeline(seconds, fn)


def scenario_second_hand_crosses_then_rests(*, seconds=8.0, rest_lift=0.0,
                                            fists_at=None):
    """The RIGHT hand drives (+0.25). The LEFT crosses the line (+0.12) for
    0.5 s at 2.0 s (touching glasses), then rests at `rest_lift` (chin /
    chest height). Optionally both hands close from `fists_at`."""
    def fn(t):
        if t < 2.0:
            left = desk_hand("left")
        else:
            ll = 0.12 if t < 2.5 else rest_lift
            left = raised_hand("left", ll, forward=0.10)
        g = 3 if (fists_at is not None and t >= fists_at) else 2
        return Frame(left, raised_hand("right", 0.25), hl=g, hr=g)
    return _timeline(seconds, fn)


def scenario_hand_joint_inferred(*, seconds=7.0, lift=0.10, start=2.0,
                                 stop=5.0, grip_from=None):
    """A STILL raised RIGHT hand whose hand joint is only Inferred from
    `start` to `stop` (the wrist stays Tracked) - the desk recording had such
    stretches up to 11 s. Optionally a fist from `grip_from` (a held drag)."""
    def fn(t):
        st = {"hand_right": 1} if start <= t < stop else {}
        g = 3 if (grip_from is not None and t >= grip_from) else 2
        return Frame(desk_hand("left"), raised_hand("right", lift), states=st,
                     hr=g)
    return _timeline(seconds, fn)


def scenario_switch_mid_drag(*, seconds=8.0):
    """The RIGHT hand drags (fist from 1.5 s); at 3.0 s the LEFT rises while
    the RIGHT lowers to the desk over 0.5 s, where the SDK reads it the way
    the desk recording does (NotTracked / Unknown, Low) - no grip evidence.
    From 4.0 s the LEFT hand sweeps 0.3 m across, driving the cursor."""
    def fn(t):
        f = min(1.0, max(0.0, (t - 3.0) / 0.5))
        dl, ul = desk_hand("left"), raised_hand("left", 0.10, x=-0.20)
        ur, dr = raised_hand("right", 0.10, x=0.20), desk_hand("right")
        left = tuple(dl[k] + f * (ul[k] - dl[k]) for k in range(3))
        right = tuple(ur[k] + f * (dr[k] - ur[k]) for k in range(3))
        if t < 1.5:
            hr, hrc = 2, 1
        elif f < 1.0:
            hr, hrc = 3, 1
        else:
            hr, hrc = (1, 0) if int(t * 30) % 5 else (0, 0)
        if t >= 4.0:
            left = raised_hand("left", 0.10,
                               x=-0.20 + 0.3 * min(1.0, (t - 4.0) / 2.0))
        return Frame(left, right, hr=hr, hrc=hrc, hl=2, hlc=1)
    return _timeline(seconds, fn)


def scenario_two_hand_dip(*, seconds=8.0):
    """Both fists raised (+0.25) grab a window; at 3.0 s the LEFT dips to
    +0.03 (under the +0.07 engage line) for 0.6 s and comes back; fists held
    throughout - the owner wants to keep resizing."""
    def fn(t):
        up = t >= 0.5
        ll = 0.03 if 3.0 <= t < 3.6 else 0.25
        left = raised_hand("left", ll) if up else desk_hand("left")
        right = raised_hand("right", 0.25) if up else desk_hand("right")
        g = 3 if t >= 1.0 else 2
        return Frame(left, right, hl=g, hr=g)
    return _timeline(seconds, fn)


def scenario_two_hand_cycles(*, cycles=3):
    """`cycles` full two-hand resizes in a row (raise, fists, hold, open,
    lower), each a fresh grab."""
    out = []
    for _ in range(cycles):
        out += scenario_two_hand_raise(before=1.0, hold=2.5, after=1.5)
    return out


def scenario_owner_dropout(*, seconds=10.0, other_dz=0.20, new_id=False,
                           out_at=3.0, out_for=0.5):
    """The owner drives (RIGHT raised) with a passive person (hands at the
    desk) `other_dz` metres farther all along. The owner drops out of the
    frames for `out_for` s at `out_at` and returns (same id, or a NEW id as
    the SDK often assigns), hand still raised."""
    def fn(t):
        bid = 1
        if out_at <= t < out_at + out_for:
            bid = None
        elif t >= out_at + out_for and new_id:
            bid = 7
        fr = Frame(desk_hand("left"), raised_hand("right", 0.25), body_id=bid)
        fr.extra = [(2, other_dz, desk_hand("left"), desk_hand("right"))]
        return fr
    return _timeline(seconds, fn)


def scenario_passive_first(*, seconds=8.0, other_dz=0.20, owner_at=1.0):
    """A passive person (hands at the desk) is in view FIRST; the owner
    arrives at `owner_at`, `other_dz` metres nearer, with a hand raised."""
    def fn(t):
        fr = Frame(desk_hand("left"), raised_hand("right", 0.25),
                   body_id=(1 if t >= owner_at else None))
        fr.extra = [(2, other_dz, desk_hand("left"), desk_hand("right"))]
        return fr
    return _timeline(seconds, fn)


def scenario_drag_with_stall(*, seconds=8.0, stall_at=4.0, stall=0.45):
    """A held drag (RIGHT fist 2.0-6.0 s) while the body pump delivers NO
    frame for `stall` seconds at `stall_at` (the CPU was pinned on
    2026-10-04: 267 ms gaps, one over 4 s)."""
    out, t = [], 0.0
    while t < seconds:
        dt = stall if (stall > 0 and abs(t - stall_at) < FRAME_S / 2) else FRAME_S
        hr = 3 if 2.0 <= t < 6.0 else 2
        out.append(Frame(desk_hand("left"), raised_hand("right", 0.25), dt=dt,
                         hr=hr))
        t += dt
    return out


def scenario_two_hand_band_fists(*, seconds=6.0, band_lift=0.05):
    """Both hands rise OPEN to +0.25 (two-hand mode engages, nothing grabbed);
    at 2.0 s the LEFT settles at `band_lift` (under the +0.07 line, inside the
    mode's 4 cm band); both hands close at 2.5 s; at 4.0 s the left rises back
    to +0.25 with the fists held."""
    def fn(t):
        up = t >= 0.5
        ll = band_lift if 2.0 <= t < 4.0 else 0.25
        left = raised_hand("left", ll) if up else desk_hand("left")
        right = raised_hand("right", 0.25) if up else desk_hand("right")
        g = 3 if t >= 2.5 else 2
        return Frame(left, right, hl=g, hr=g)
    return _timeline(seconds, fn)
