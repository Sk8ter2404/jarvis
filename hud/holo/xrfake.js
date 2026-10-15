// Holo XR - a built-in FAKE pair of hands for /holo/xr?fake=1 (and the self
// tests): scripted joint snapshots in the same shape xr.js hands the gesture
// router, so the whole XR page - grab, two-hand scale + rotate, explode, pull a
// part and hold for its info, clap - can be exercised on a desktop with no
// headset. Pure: step(dt) returns {left, right} (each {joints}|null).
//
// probe (what the page knows about its hologram, world metres):
//   sphere() -> {c, r}                  where the object is
//   partCenter(name) -> [x, y, z]|null  where a named part is (exploded)
//   exploded() -> bool
// The script waits for the page where it must (the explode and the assemble
// animation) instead of trusting fixed times.

import { add3 } from "./xrgestures.js";

export const PART_FOR_INFO = "Core";

// One hand from where its PALM is and where its pinch point is.
export function makeXRHand(palm, pinchAt, pinching) {
  const gap = pinching ? 0.008 : 0.075;               // thumb tip <-> index tip
  const j = {};
  j["wrist"] = add3(palm, [0, 0.03, 0.04]);
  j["middle-finger-phalanx-proximal"] = add3(palm, [0, -0.03, -0.04]);
  j["thumb-tip"] = add3(pinchAt, [-gap / 2, 0, 0]);
  j["index-finger-tip"] = add3(pinchAt, [gap / 2, 0, 0]);
  return { joints: j };
}

const lerp3 = (a, b, u) => [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
const ease = u => u * u * (3 - 2 * u);

export class FakeXRHands {
  constructor(probe) {
    this.probe = probe;
    this.t = 0;
    this.i = 0;                  // current phase
    this.pt = 0;                 // time inside the phase
    this.ctx = null;             // captured when a phase starts
    this.PHASES = this._phases();
  }

  // Each phase: name, min seconds, optional wait() -> true when it may end, hands(u, ctx).
  _phases() {
    const P = this.probe;
    const at = (c, off) => add3(c, off);
    // palm a hand's length behind the finger tips (x.c is unused here: positions are absolute)
    const H = (_c, pp, pinching) => makeXRHand(add3(pp, [0, -0.02, 0.08]), pp, pinching);
    return [
      { name: "idle", s: 1.0, hands: (u, x) => ({
        left: H(x.c, at(x.c, [-0.30, -0.10, 0.1]), false), right: H(x.c, at(x.c, [0.30, -0.10, 0.1]), false) }) },
      // right hand reaches the object, pinches, drags it up-right, lets go
      { name: "reach", s: 0.8, hands: (u, x) => ({
        left: null, right: H(x.c, lerp3(at(x.c, [0.35, -0.1, 0.1]), at(x.c, [0.05, 0, 0]), ease(u)), false) }) },
      { name: "grab", s: 0.3, hands: (u, x) => ({ left: null, right: H(x.c, at(x.c, [0.05, 0, 0]), true) }) },
      { name: "drag", s: 1.2, hands: (u, x) => ({
        left: null, right: H(x.c, lerp3(at(x.c, [0.05, 0, 0]), at(x.c, [0.17, 0.06, -0.05]), ease(u)), true) }) },
      { name: "let go", s: 0.5, hands: (u, x) => ({ left: null, right: H(x.c, at(x.c, [0.17, 0.06, -0.05]), false) }) },
      // two hands pinch the sides and spread SLOWLY (scale), tilt (rotate), let go
      { name: "two-hand pinch", s: 0.5, hands: (u, x) => ({
        left: H(x.c, at(x.c, [-0.09, 0, 0]), true), right: H(x.c, at(x.c, [0.09, 0, 0]), true) }) },
      { name: "scale + rotate", s: 2.2, hands: (u, x) => {
        const w = 0.09 + 0.05 * ease(u), tilt = 0.10 * ease(u);
        return { left: H(x.c, at(x.c, [-w, 0, 0]), true), right: H(x.c, at(x.c, [w, tilt, 0]), true) };
      } },
      { name: "let go 2", s: 0.6, hands: (u, x) => ({
        left: H(x.c, at(x.c, [-0.14, 0, 0]), false), right: H(x.c, at(x.c, [0.14, 0.1, 0]), false) }) },
      // both pinch and pull apart FAST -> explode
      { name: "pinch to explode", s: 0.6, hands: (u, x) => ({
        left: H(x.c, at(x.c, [-0.07, 0, 0]), true), right: H(x.c, at(x.c, [0.07, 0, 0]), true) }) },
      { name: "yank apart", s: 0.4, hands: (u, x) => {
        const w = 0.07 + 0.30 * ease(u);
        return { left: H(x.c, at(x.c, [-w, 0, 0]), true), right: H(x.c, at(x.c, [w, 0, 0]), true) };
      } },
      { name: "wait: parts settle", s: 1.6, wait: () => P.exploded(), hands: (u, x) => ({
        left: H(x.c, at(x.c, [-0.37, 0, 0]), false), right: H(x.c, at(x.c, [0.37, 0, 0]), false) }) },
      // pinch the Core, hold for its info, let go
      { name: "reach for a part", s: 0.9, wait: () => !!P.partCenter(PART_FOR_INFO), hands: (u, x) => {
        const pc = P.partCenter(PART_FOR_INFO) || x.c;
        return { left: null, right: H(x.c, lerp3(at(x.c, [0.3, -0.1, 0.1]), pc, ease(u)), false) };
      } },
      { name: "pinch the part", s: 0.3, hands: (u, x) => {
        const pc = P.partCenter(PART_FOR_INFO) || x.c;
        return { left: null, right: H(x.c, pc, true) };
      } },
      { name: "hold for info", s: 1.7, hands: (u, x) => {
        const pc = P.partCenter(PART_FOR_INFO) || x.c;
        return { left: null, right: H(x.c, pc, true) };
      } },
      { name: "release the part", s: 0.8, hands: (u, x) => {
        const pc = P.partCenter(PART_FOR_INFO) || x.c;
        return { left: null, right: H(x.c, pc, false) };
      } },
      // clap -> assemble
      { name: "hands apart", s: 0.7, hands: (u, x) => ({
        left: H(x.c, at(x.c, [-0.3, -0.05, 0.2]), false), right: H(x.c, at(x.c, [0.3, -0.05, 0.2]), false) }) },
      { name: "clap", s: 0.25, hands: (u, x) => {
        const w = 0.3 - 0.27 * ease(u);
        return { left: H(x.c, at(x.c, [-w, -0.05, 0.2]), false), right: H(x.c, at(x.c, [w, -0.05, 0.2]), false) };
      } },
      { name: "after the clap", s: 2.0, wait: () => !P.exploded(), hands: (u, x) => ({
        left: H(x.c, at(x.c, [-0.03, -0.05, 0.2]), false), right: H(x.c, at(x.c, [0.03, -0.05, 0.2]), false) }) },
    ];
  }

  _enter() {
    const s = this.probe.sphere();
    this.ctx = { c: s ? s.c.slice() : [0, 1.2, -0.5] };
  }

  // Advance by dt seconds; returns {left, right}.
  step(dt) {
    if (!this.ctx) this._enter();
    this.t += dt; this.pt += dt;
    let ph = this.PHASES[this.i];
    const long = this.pt > ph.s + 4;                  // a wait that never ends: move on anyway
    if (this.pt >= ph.s && (!ph.wait || ph.wait() || long)) {
      this.i = (this.i + 1) % this.PHASES.length;
      this.pt = 0;
      if (this.i === 0) this._enter();                // a new loop re-reads where the object is
      ph = this.PHASES[this.i];
      if (ph.name === "pinch to explode" || ph.name === "two-hand pinch"
          || ph.name === "hands apart" || ph.name === "reach") this._enter();
    }
    const u = Math.min(1, this.pt / Math.max(1e-6, ph.s));
    const out = ph.hands(u, this.ctx);
    return { left: out.left || null, right: out.right || null, phase: ph.name };
  }
}
