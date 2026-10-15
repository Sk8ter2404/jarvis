// Holo XR (Quest 3 / WebXR) - the PURE gesture math. No DOM, no three.js, no
// timers, no network, so selftest.js can run it with a mock target.
//
// WHO DOES WHAT
//   The headset's own hand tracking replaces the webcam tracker: xr.js reads the
//   26 XRHand joints of each hand every frame and hands this module a snapshot,
//   {left: hand|null, right: hand|null}, hand = {joints: {name: [x, y, z]}} in
//   METRES in the XR reference space. This module recognises:
//     PINCH       thumb-tip to index-tip distance, with hysteresis
//     GRAB        one pinch near the object: it follows the hand in 3D
//     TWO-HAND    both pinching: scale (hand distance) + rotate (hand vector),
//                 the object is rigidly attached to the pair of pinch points
//     EXPLODE     both pinching hands pulled apart FAST -> explode into parts
//     CLAP        palms meet fast (neither hand pinching) -> assemble
//     PART + HOLD exploded: pinch a part to pull it out; hold ~1 s -> info label
//   and tells a duck-typed TARGET (xr.js's adapter; see MockTarget in selftest.js):
//     target.exploded                      (read)
//     target.sphere() -> {c:[x,y,z], r}    world metres, null when nothing is shown
//     target.position() -> [x,y,z]         where the object is (its steering target)
//     target.getScale() -> number
//     target.grabMove(pos) / grabEnd()
//     target.twoBegin() / twoMove({pos, scale, rot:[x,y,z,w]}) / twoEnd(revert)
//     target.explode() / assemble()
//     target.partAt(point) -> part|null    nearest part (exploded only)
//     target.pullPart(part, point) / movePart(part, point) / releasePart(part)
//     target.showInfo(part|null)
//
// All thresholds below are first guesses from published hand-tracking numbers;
// they need a session with the real headset to tune (the owner's hands, the
// Quest's latency). They are in ONE frozen table so that is a one-place edit.
//
// NEVER touches the camera or the OS; it only does arithmetic.

export const X = Object.freeze({
  // Pinch: thumb-tip to index-tip distance (the joint centres sit about 1 cm
  // apart when the finger pads touch). Hysteresis: on at <= 2.0 cm, off at >= 3.2 cm.
  PINCH_ON_M: 0.020,
  PINCH_OFF_M: 0.032,
  // A hand that vanishes for less than this keeps its pinch (tracking dropouts).
  PINCH_LOST_S: 0.15,
  // A pinch starts a grab within this far of the object's edge.
  GRAB_MARGIN_M: 0.07,
  // ...and picks a part within this far of the part's edge (exploded).
  PART_MARGIN_M: 0.04,
  // Two-hand scale limits (multiplier on the default size).
  SCALE_MIN: 0.25,
  SCALE_MAX: 4.0,
  // Explode: the pinch points of two pinching hands separate by >= 12 cm within
  // 0.3 s at >= 0.8 m/s (a slow scale gesture is ~0.2-0.5 m/s).
  APART_SPEED_MPS: 0.8,
  APART_TRAVEL_M: 0.12,
  APART_WINDOW_S: 0.30,
  APART_MIN_S: 0.10,
  // While the hands separate faster than this fraction of APART_SPEED_MPS the
  // two-hand scale holds still, so a yank that is about to explode never makes
  // the model swell first.
  APART_HOLD_FRAC: 0.5,
  // Clap: palm centres end <= 12 cm apart having closed >= 18 cm within 0.35 s
  // at >= 1.0 m/s. When the hands vanish at contact (they hide each other) a
  // closing that was already this fast and ended <= 18 cm apart counts too.
  CLAP_DIST_M: 0.12,
  CLAP_TRAVEL_M: 0.18,
  CLAP_SPEED_MPS: 1.0,
  CLAP_WINDOW_S: 0.35,
  CLAP_LOST_DIST_M: 0.18,
  CLAP_COOLDOWN_S: 0.8,
  // Exploded: a part held by a pinch that stays slower than 15 cm/s for 1 s
  // shows its info.
  INFO_HOLD_S: 1.0,
  HOLD_SPEED_MPS: 0.15,
  // After explode / assemble the opposite event is ignored this long (the end of
  // a clap must not read as an explode).
  SWITCH_LOCK_S: 1.2,
  // Poke (the model shelf): a fingertip within +1 cm / -3 cm of a panel's plane
  // and inside its edges is a press; it must pull back 7 cm before pressing again.
  POKE_FRONT_M: 0.01,
  POKE_DEPTH_M: 0.03,
  POKE_REARM_M: 0.07,
  // The hologram appears this far in front of the head and this far below it.
  PLACE_DIST_M: 0.5,
  PLACE_DROP_M: 0.30,
});

// ---------------------------------------------------------------------------
// small vector / quaternion helpers (plain arrays; quaternion = [x, y, z, w])
// ---------------------------------------------------------------------------
export function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
export const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const mul3 = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
export const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const len3 = a => Math.hypot(a[0], a[1], a[2]);
export const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
export const mid3 = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
export function unit3(a) {
  const l = len3(a);
  return l > 1e-9 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
}

export const QUAT_IDENTITY = Object.freeze([0, 0, 0, 1]);

export function quatMul(a, b) {                       // a * b (apply b, then a)
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

export function quatRotate(q, v) {                    // rotate vector v by unit quaternion q
  const u = [q[0], q[1], q[2]], w = q[3];
  const t = mul3(cross3(u, v), 2);
  return add3(add3(v, mul3(t, w)), cross3(u, t));
}

// The shortest rotation taking direction a onto direction b (any lengths).
export function quatFromVectors(a, b) {
  const ua = unit3(a), ub = unit3(b);
  if (len3(ua) === 0 || len3(ub) === 0) return [0, 0, 0, 1];
  const d = dot3(ua, ub);
  if (d > 0.999999) return [0, 0, 0, 1];
  if (d < -0.999999) {                                // opposite: 180 deg about any perpendicular
    let axis = cross3([1, 0, 0], ua);
    if (len3(axis) < 1e-6) axis = cross3([0, 1, 0], ua);
    axis = unit3(axis);
    return [axis[0], axis[1], axis[2], 0];
  }
  const c = cross3(ua, ub);
  const q = [c[0], c[1], c[2], 1 + d];
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

// ---------------------------------------------------------------------------
// pinch + hand features
// ---------------------------------------------------------------------------
// Pinch with hysteresis: on at <= PINCH_ON_M, stays on until >= PINCH_OFF_M.
export function pinchState(wasPinching, d) {
  return d <= (wasPinching ? X.PINCH_OFF_M : X.PINCH_ON_M);
}

export function pinchPoint(hand) {
  return mid3(hand.joints["thumb-tip"], hand.joints["index-finger-tip"]);
}

export function pinchDistance(hand) {
  return dist3(hand.joints["thumb-tip"], hand.joints["index-finger-tip"]);
}

// The middle of the palm: halfway between the wrist and the middle knuckle.
export function palmCenter(hand) {
  const j = hand.joints;
  const k = j["middle-finger-phalanx-proximal"] || j["middle-finger-metacarpal"] || j["wrist"];
  return mid3(j["wrist"], k);
}

export function scaleFromDistance(startScale, startDist, nowDist) {
  if (!(startDist > 1e-4)) return startScale;
  return clamp(startScale * (nowDist / startDist), X.SCALE_MIN, X.SCALE_MAX);
}

// Where the hologram first appears: PLACE_DIST_M ahead of the head along its
// HORIZONTAL facing, PLACE_DROP_M below it (about chest height). Looking straight
// down or up, the facing comes from the head's up vector instead.
export function placeInFront(headPos, headQuat, dist = X.PLACE_DIST_M, drop = X.PLACE_DROP_M) {
  const fwd = quatRotate(headQuat, [0, 0, -1]);
  let h = [fwd[0], 0, fwd[2]];
  if (len3(h) < 0.25) {
    const up = quatRotate(headQuat, [0, 1, 0]);
    h = fwd[1] < 0 ? [up[0], 0, up[2]] : [-up[0], 0, -up[2]];
  }
  h = unit3(h);
  if (len3(h) === 0) h = [0, 0, -1];
  return [headPos[0] + h[0] * dist, headPos[1] - drop, headPos[2] + h[2] * dist];
}

// ---------------------------------------------------------------------------
// poke (the in-headset model shelf)
// ---------------------------------------------------------------------------
// Basis of a panel at `c` that turns about the vertical axis to face the head.
// n points at the viewer; right is the viewer's right.
export function panelBasis(c, head) {
  let n = unit3([head[0] - c[0], 0, head[2] - c[2]]);
  if (len3(n) === 0) n = [0, 0, 1];
  return { n, right: [n[2], 0, -n[0]], up: [0, 1, 0] };
}

// Is a fingertip pressing a panel? panel = {c, hw, hh} (centre, half width, half height).
export function pokeHit(tip, panel, basis) {
  const r = sub3(tip, panel.c);
  const depth = dot3(r, basis.n);                     // + in front of the panel, - behind
  const u = dot3(r, basis.right), v = dot3(r, basis.up);
  const inside = Math.abs(u) <= panel.hw && Math.abs(v) <= panel.hh;
  const hit = inside && depth <= X.POKE_FRONT_M && depth >= -X.POKE_DEPTH_M;
  return { hit, depth, inside };
}

// One press per touch: fires once on contact, re-arms when the finger pulls back.
export class PokeGate {
  constructor() { this.armed = true; }
  update(hit, depth, inside) {
    if (this.armed && hit) { this.armed = false; return true; }
    if (!this.armed && (!inside || depth >= X.POKE_REARM_M)) this.armed = true;
    return false;
  }
}

// ---------------------------------------------------------------------------
// fast-apart (explode) and clap detectors
// ---------------------------------------------------------------------------
export class ApartDetector {
  constructor() { this.s = []; }
  reset() { this.s.length = 0; }
  // d = distance between the two pinch points now. True once, when they have
  // separated far and fast enough.
  update(t, d) {
    this.s.push([t, d]);
    while (this.s.length > 1 && t - this.s[0][0] > X.APART_WINDOW_S) this.s.shift();
    const [t0, d0] = this.s[0];
    const dt = t - t0;
    if (dt < X.APART_MIN_S) return false;
    const grow = d - d0;
    if (grow < X.APART_TRAVEL_M || grow / dt < X.APART_SPEED_MPS) return false;
    this.s.length = 0;
    return true;
  }
}

export class ClapDetector {
  constructor() { this.s = []; this.lastFire = -1e9; }
  reset() { this.s.length = 0; }
  // d = distance between the palm centres now, or null when a hand just
  // vanished (hands hide each other at contact). True once per clap.
  update(t, d) {
    const cooling = t - this.lastFire < X.CLAP_COOLDOWN_S;
    if (d !== null) this.s.push([t, d]);
    while (this.s.length && t - this.s[0][0] > X.CLAP_WINDOW_S) this.s.shift();
    if (cooling || !this.s.length) return false;
    const [tl, dl] = this.s[this.s.length - 1];       // the newest real sample
    if (d === null && t - tl > 0.12) return false;    // vanished long ago: not a clap
    const limit = d === null ? X.CLAP_LOST_DIST_M : X.CLAP_DIST_M;
    if (dl > limit) return false;
    let tm = tl, dm = dl;                             // the LATEST widest point of the closing
    for (const [ts, ds] of this.s) if (ds >= dm) { tm = ts; dm = ds; }
    const travel = dm - dl, span = tl - tm;
    if (travel < X.CLAP_TRAVEL_M || span <= 0 || travel / span < X.CLAP_SPEED_MPS) return false;
    this.lastFire = t;
    this.s.length = 0;
    return true;
  }
}

// ---------------------------------------------------------------------------
// the router
// ---------------------------------------------------------------------------
function newHand() {
  return { pinch: false, rose: false, ignored: false, tracked: false, lostS: 99,
    pp: null, palm: null, speed: 0 };
}

export class XRGestureRouter {
  constructor(target, opts = {}) {
    this.target = target;
    this.log = opts.log || (() => {});
    this.t = 0;
    this.h = { left: newHand(), right: newHand() };
    this.mode = null;        // {kind: grab|two|part|wait, ...}
    this.apart = new ApartDetector();
    this.clap = new ClapDetector();
    this.noExplodeUntil = 0;
    this.noAssembleUntil = 0;
  }

  get busy() { return this.mode !== null; }

  summary() {
    return {
      mode: this.mode ? this.mode.kind : "none",
      left: this.h.left.pinch ? "pinch" : (this.h.left.tracked ? "open" : "-"),
      right: this.h.right.pinch ? "pinch" : (this.h.right.tracked ? "open" : "-"),
    };
  }

  // Drop whatever is in progress (session ended, tracking gone).
  reset() {
    this._endAll();
    for (const s of ["left", "right"]) this.h[s] = newHand();
    this.apart.reset(); this.clap.reset();
  }

  _near(p, sphere) {
    return !!(sphere && p && dist3(p, sphere.c) <= sphere.r + X.GRAB_MARGIN_M);
  }

  _read(side, snap, dt) {
    const st = this.h[side];
    st.rose = false;
    const j = snap && snap.joints;
    if (!j || !j["thumb-tip"] || !j["index-finger-tip"] || !j["wrist"]) {
      st.tracked = false; st.lostS += dt;
      if (st.lostS > X.PINCH_LOST_S) { st.pinch = false; st.ignored = false; st.speed = 0; }
      return;
    }
    st.tracked = true; st.lostS = 0;
    const was = st.pinch;
    st.pinch = pinchState(was, pinchDistance(snap));
    st.rose = st.pinch && !was;
    if (!st.pinch) st.ignored = false;
    const pp = pinchPoint(snap);
    if (st.pp) st.speed += (dist3(pp, st.pp) / dt - st.speed) * 0.25;
    st.pp = pp;
    st.palm = palmCenter(snap);
  }

  // hands = {left: {joints}|null, right: {joints}|null}; dt in seconds.
  update(hands, dt) {
    dt = clamp(dt, 0.001, 0.1);
    this.t += dt;
    const T = this.target;
    const L = this.h.left, R = this.h.right;
    this._read("left", hands && hands.left, dt);
    this._read("right", hands && hands.right, dt);
    const sphere = T.sphere ? T.sphere() : null;

    this._clapStep(L, R);

    // The voice (or anything else) put it back together under a held part.
    if (this.mode && this.mode.kind === "part" && !T.exploded) this._endAll();

    if (!this.mode) return this._idle(L, R, sphere);
    switch (this.mode.kind) {
      case "wait": if (!L.pinch && !R.pinch) this.mode = null; break;
      case "grab": this._grab(L, R, sphere); break;
      case "two": this._two(L, R, sphere, dt); break;
      case "part": this._part(dt); break;
    }
  }

  // -- clap -> assemble --------------------------------------------------------
  _clapStep(L, R) {
    if (L.pinch || R.pinch) { this.clap.reset(); return; }
    let d;
    if (L.tracked && R.tracked) d = dist3(L.palm, R.palm);
    else if (L.palm && R.palm && L.lostS <= 0.2 && R.lostS <= 0.2) d = null;
    else { this.clap.reset(); return; }
    if (!this.clap.update(this.t, d)) return;
    const T = this.target;
    this.log("clap");
    if (T.exploded && this.t >= this.noAssembleUntil) {
      this._endAll();
      T.assemble();
      this.noExplodeUntil = this.t + X.SWITCH_LOCK_S;
    }
  }

  // -- nothing in progress: does a new pinch start something? ------------------
  _idle(L, R, sphere) {
    const T = this.target;
    if (!L.rose && !R.rose) return;
    if (L.pinch && R.pinch) {                         // two-hand pinch
      if (this._near(L.pp, sphere) || this._near(R.pp, sphere)) return this._beginTwo(L, R);
      L.ignored = R.ignored = true;
      return;
    }
    for (const side of ["left", "right"]) {
      const st = this.h[side];
      if (!st.rose) continue;
      if (!sphere) { st.ignored = true; continue; }
      if (T.exploded && T.partAt) {
        const part = T.partAt(st.pp);
        if (part) return this._beginPart(side, part);
      }
      if (this._near(st.pp, sphere)) return this._beginGrab(side);
      st.ignored = true;                              // pinched in empty air: not a grab later
    }
  }

  _beginGrab(side) {
    const T = this.target;
    this.mode = { kind: "grab", side, off: sub3(T.position(), this.h[side].pp) };
    this.log("grab " + side);
  }

  _grab(L, R, sphere) {
    const T = this.target;
    const m = this.mode, st = this.h[m.side];
    if (!st.pinch) { T.grabEnd(); this.mode = null; this.log("release"); return; }
    const other = this.h[m.side === "left" ? "right" : "left"];
    if (other.rose && other.pinch) {                  // the other hand joins: scale + rotate
      T.grabEnd();
      this._beginTwo(L, R);
      return;
    }
    T.grabMove(add3(st.pp, m.off));
  }

  _beginTwo(L, R) {
    const T = this.target;
    const d0 = Math.max(dist3(L.pp, R.pp), 0.05);
    this.mode = {
      kind: "two", d0, v0: unit3(sub3(R.pp, L.pp)), mid0: mid3(L.pp, R.pp),
      pos0: T.position(), scale0: T.getScale(), lastD: d0, rate: 0,
    };
    this.apart.reset();
    if (T.twoBegin) T.twoBegin();
    this.log("two-hand");
  }

  _two(L, R, sphere, dt) {
    const T = this.target, m = this.mode;
    if (!(L.pinch && R.pinch)) {                      // one let go: the other keeps holding
      T.twoEnd(false);
      this.mode = null;
      const rem = L.pinch ? "left" : (R.pinch ? "right" : null);
      if (rem && this._near(this.h[rem].pp, sphere)) this._beginGrab(rem);
      else if (rem) this.h[rem].ignored = true;
      return;
    }
    const d = dist3(L.pp, R.pp);
    if (!T.exploded && this.t >= this.noExplodeUntil && this.apart.update(this.t, d)) {
      T.twoEnd(true);                                 // the pull was not a scale: undo it
      this.log("explode");
      T.explode();
      this.noAssembleUntil = this.t + X.SWITCH_LOCK_S;
      this.mode = { kind: "wait" };                   // until both let go
      return;
    }
    m.rate = m.rate * 0.5 + ((d - m.lastD) / dt) * 0.5;   // how fast the hands are separating (m/s)
    m.lastD = d;
    if (!T.exploded && m.rate > X.APART_SPEED_MPS * X.APART_HOLD_FRAC) return;   // possibly a yank: hold still
    const scale = scaleFromDistance(m.scale0, m.d0, d);
    const rot = quatFromVectors(m.v0, sub3(R.pp, L.pp));
    const rel = mul3(sub3(m.pos0, m.mid0), scale / m.scale0);
    T.twoMove({ pos: add3(mid3(L.pp, R.pp), quatRotate(rot, rel)), scale, rot });
  }

  _beginPart(side, part) {
    const T = this.target;
    T.pullPart(part, this.h[side].pp);
    this.mode = { kind: "part", side, part, held: 0, shown: false };
    this.log("part " + part.name);
  }

  _part(dt) {
    const T = this.target, m = this.mode, st = this.h[m.side];
    if (!st.pinch) { this._endAll(); this.log("release part"); return; }
    T.movePart(m.part, st.pp);
    if (st.speed < X.HOLD_SPEED_MPS) m.held += dt;
    else { m.held = 0; if (m.shown) { m.shown = false; T.showInfo(null); } }
    if (!m.shown && m.held >= X.INFO_HOLD_S) { m.shown = true; T.showInfo(m.part); this.log("info " + m.part.name); }
  }

  _endAll() {
    const T = this.target, m = this.mode;
    if (!m) return;
    if (m.kind === "grab") T.grabEnd();
    else if (m.kind === "two") T.twoEnd(false);
    else if (m.kind === "part") { T.releasePart(m.part); T.showInfo(null); }
    this.mode = null;
    for (const s of ["left", "right"]) if (this.h[s].pinch) this.h[s].ignored = true;
  }
}
