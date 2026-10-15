// Built-in FAKE hand source for /holo?fake=1: scripted hand motion plus the
// same gesture events the tracker service sends, in the same message shapes
// ({"type":"hands",...} and {"type":"gesture",...}), so the whole page can be
// demonstrated and tested without a camera or the service. Pure: step(t)
// returns the messages due; holo.js drives it ~30 times a second.
//
// The script (one loop, LOOP_S seconds) walks the legend end to end:
//   palm_hold -> menu, point at "Arc reactor" + pinch -> spawn, fist grab and
//   move, point + flick (spin both ways), explode, pinch a part + hold for its
//   info, clap -> assemble.

export const LOOP_S = 17.0;

// Hand template in palm-size units, pointing up (negative y). Index = MediaPipe landmark.
const EXT = [
  [0, 0],
  [-0.35, -0.25], [-0.6, -0.5], [-0.8, -0.75], [-0.95, -1.0],      // thumb 1-4
  [-0.38, -1.0], [-0.4, -1.5], [-0.42, -1.85], [-0.43, -2.15],     // index 5-8
  [-0.1, -1.05], [-0.1, -1.6], [-0.1, -1.98], [-0.1, -2.3],        // middle 9-12
  [0.18, -1.0], [0.2, -1.5], [0.21, -1.85], [0.22, -2.12],         // ring 13-16
  [0.42, -0.9], [0.46, -1.3], [0.48, -1.55], [0.5, -1.78],         // pinky 17-20
];
// Curled finger: PIP/DIP/TIP folded back over the knuckle.
const CURL = {
  5: [[-0.38, -1.35], [-0.34, -1.2], [-0.3, -0.95]],
  9: [[-0.1, -1.4], [-0.08, -1.2], [-0.06, -0.92]],
  13: [[0.18, -1.35], [0.17, -1.15], [0.16, -0.9]],
  17: [[0.4, -1.15], [0.38, -1.0], [0.36, -0.8]],
};
const FINGER_BASE = { index: 5, middle: 9, ring: 13, pinky: 17 };

const lerp = (a, b, u) => a + (b - a) * u;
const ease = u => u * u * (3 - 2 * u);

// spec: {x,y: wrist position (normalized), size: palm size, rot: radians,
//        curl: {index,middle,ring,pinky: 0..1}, thumbIn: 0..1, pinch: bool}
export function makeLandmarks(spec) {
  const { x, y, size = 0.11, rot = 0 } = spec;
  const curl = spec.curl || {};
  const pts = EXT.map(p => [p[0], p[1]]);
  for (const [name, base] of Object.entries(FINGER_BASE)) {
    const c = curl[name] || 0;
    if (!c) continue;
    for (let k = 0; k < 3; k++) {
      const ext = EXT[base + 1 + k], cur = CURL[base][k];
      pts[base + 1 + k] = [lerp(ext[0], cur[0], c), lerp(ext[1], cur[1], c)];
    }
  }
  const ti = spec.thumbIn || 0;                 // thumb folded across the palm
  if (ti) {
    const tgt = [[-0.2, -0.5], [-0.05, -0.7], [0.05, -0.85]];
    for (let k = 0; k < 3; k++) {
      const e = EXT[2 + k];
      pts[2 + k] = [lerp(e[0], tgt[k][0], ti), lerp(e[1], tgt[k][1], ti)];
    }
    const e4 = EXT[4];
    pts[4] = [lerp(e4[0], 0.05, ti), lerp(e4[1], -1.0, ti)];
  }
  if (spec.pinch) {                              // thumb tip meets index tip
    const m = [(pts[4][0] + pts[8][0]) / 2 + 0.05, (pts[4][1] + pts[8][1]) / 2 + 0.25];
    pts[4] = [m[0] - 0.02, m[1]];
    pts[8] = [m[0] + 0.02, m[1]];
    pts[7] = [pts[7][0] + 0.02, pts[7][1] + 0.35];
    pts[6] = [pts[6][0] + 0.02, pts[6][1] + 0.12];
  }
  const cs = Math.cos(rot), sn = Math.sin(rot);
  return pts.map(([px, py], i) => {
    const rx = px * cs - py * sn, ry = px * sn + py * cs;
    return [x + rx * size * 0.75, y + ry * size, (i === 0 ? 0 : -0.02 * (i % 4))];   // 0.75: 4:3 aspect
  });
}

function palmOf(lm) {
  const ids = [0, 5, 9, 13, 17];
  return [ids.reduce((a, i) => a + lm[i][0], 0) / 5, ids.reduce((a, i) => a + lm[i][1], 0) / 5];
}

const POSES = {
  open: {},
  fist: { curl: { index: 1, middle: 1, ring: 1, pinky: 1 }, thumbIn: 1 },
  point: { curl: { middle: 1, ring: 1, pinky: 1 }, thumbIn: 1 },
  pinch: { curl: { middle: 0.6, ring: 0.8, pinky: 0.8 }, pinch: true },
};

function hand(side, state, x, y, extra = {}) {
  const spec = { x, y, size: extra.size || 0.11, rot: extra.rot || 0, ...POSES[state] };
  const lm = makeLandmarks(spec);
  const d = Math.hypot(lm[4][0] - lm[8][0], lm[4][1] - lm[8][1]);
  return {
    id: side === "Left" ? 1 : 0, side, score: 0.98, lm, world: [], state,
    pinch: state === "pinch" ? 1 : Math.max(0, Math.min(1, 1 - d / 0.25)),
    palm: palmOf(lm), palm_facing: state === "open",
  };
}

// Place a hand so a chosen landmark (default index tip 8) lands on (tx, ty).
function handAtTip(side, state, tx, ty, lmIndex = 8, rot = 0) {
  const h0 = hand(side, state, 0, 0, { rot });
  const dx = tx - h0.lm[lmIndex][0], dy = ty - h0.lm[lmIndex][1];
  return hand(side, state, dx, dy, { rot });
}

// Place a hand so its PALM CENTRE (what a fist drags the reactor by) lands on (px, py).
function handAtPalm(side, state, px, py) {
  const h0 = hand(side, state, 0, 0);
  return hand(side, state, px - h0.palm[0], py - h0.palm[1]);
}

export class FakeHands {
  // probe: {menuItemNorm(i)->[x,y]|null, partNorm(name)->[x,y]|null, reactorNorm()->[x,y]|null}
  // opts.depth: also fake the Kinect - every hand carries "z" (the fist phase
  // pushes toward the sensor then pulls back) and depthPlane(t) gives a 160x90
  // plane (wall 2.6 m, a torso band 1.4 m, a blob drifting past at 0.7 m).
  constructor(probe = {}, opts = {}) {
    this.probe = probe;
    this.depth = !!opts.depth;
    this._planeSeq = 0;
    this.t = 0;
    this.seq = 0;
    this._fired = new Set();
    this._lastLoop = 0;
  }

  _center() { return (this.probe.reactorNorm && this.probe.reactorNorm()) || [0.5, 0.5]; }

  // Hands for script time t (0..LOOP_S).
  _hands(t) {
    const R = "Right";
    if (t < 1.4) return [hand(R, "open", lerp(0.78, 0.46, ease(t / 1.4)), lerp(0.85, 0.62, ease(t / 1.4)))];
    if (t < 1.7) return [hand(R, "open", 0.46, 0.62)];
    if (t < 3.0) {                                    // point at the first menu item
      const tgt = (this.probe.menuItemNorm && this.probe.menuItemNorm(0)) || [0.5, 0.4];
      const u = ease((t - 1.7) / 1.3);
      const from = [0.52, 0.55];
      return [handAtTip(R, "point", lerp(from[0], tgt[0], u), lerp(from[1], tgt[1], u))];
    }
    if (t < 3.6) {                                    // pinch = click (state edge)
      const tgt = (this.probe.menuItemNorm && this.probe.menuItemNorm(0)) || [0.5, 0.4];
      return [handAtTip(R, "pinch", tgt[0], tgt[1])];
    }
    if (t < 4.2) return [hand(R, "open", 0.46, 0.7)];
    if (t < 6.8) {                                    // fist: grab at the reactor, drag right then back
      if (t < 4.25 || !this._grabC) this._grabC = this._center();   // fixed anchor: no feedback loop
      const c = this._grabC;
      const u = (t - 4.2) / 2.6;
      const dx = Math.sin(u * Math.PI * 2) * 0.12;
      return [handAtPalm(R, "fist", c[0] + dx, c[1])];
    }
    if (t < 8.9) {                                    // point, flick right at 7.6
      const c = this._center();
      return [handAtTip(R, "point", c[0] + 0.15, c[1] + 0.2)];
    }
    if (t < 10.2) {
      const c = this._center();
      return [handAtTip(R, "point", c[0] - 0.15, c[1] + 0.2)];
    }
    if (t < 11.6) return [hand("Left", "fist", 0.3, 0.55), hand(R, "fist", 0.7, 0.55)];
    if (t < 13.6) {                                   // pinch a part and hold it ~1.8 s
      // fixed anchor, captured before the pinch lands: the part follows the
      // hand, so re-reading the part's position every frame would chase itself
      if (t < 11.95 || !this._partC) this._partC = (this.probe.partNorm && this.probe.partNorm("Core")) || [0.5, 0.5];
      const tgt = this._partC;
      const u = Math.min(1, (t - 11.6) / 1.2);
      const wob = Math.sin(t * 3) * 0.004 * u;
      return [handAtTip(R, "pinch", tgt[0] + wob, tgt[1] + u * 0.12, 8)];
    }
    if (t < 14.4) return [hand(R, "open", 0.55, 0.7)];
    if (t < 15.0) {                                   // two palms close fast: clap
      const u = ease((t - 14.4) / 0.6);
      return [hand("Left", "open", lerp(0.3, 0.47, u), 0.62), hand(R, "open", lerp(0.7, 0.53, u), 0.62)];
    }
    if (t < 16.2) return [hand("Left", "open", 0.4, 0.7), hand(R, "open", 0.6, 0.7)];
    return [];
  }

  // Gesture events and the time each one fires at.
  _events(t) {
    const c = this._center();
    const part = this._partC || (this.probe.partNorm && this.probe.partNorm("Core")) || [0.5, 0.5];
    return [
      { at: 1.5, ev: { name: "palm_hold", side: "Right", pos: [0.46, 0.55] } },
      { at: 4.2, ev: { name: "fist_grab", side: "Right", pos: c } },
      { at: 6.8, ev: { name: "fist_release", side: "Right", pos: this._grabC || c } },
      { at: 7.6, ev: { name: "point_flick", side: "Right", pos: [c[0] + 0.15, c[1] + 0.2], value: 420 } },
      { at: 9.4, ev: { name: "point_flick", side: "Right", pos: [c[0] - 0.15, c[1] + 0.2], value: -380 } },
      { at: 10.8, ev: { name: "explode", side: "both", pos: [0.5, 0.55] } },
      { at: 11.9, ev: { name: "pinch_start", side: "Right", pos: part } },
      { at: 14.9, ev: { name: "clap", side: "both", pos: [0.5, 0.62] } },
    ];
  }

  // Fake hand depth in metres for script time t (the fist phase is 4.2-6.8 s).
  handZ(t) {
    if (t >= 4.2 && t < 6.8) return 0.85 - 0.25 * Math.sin(((t - 4.2) / 2.6) * Math.PI * 2);
    return 0.85;
  }

  // A fake depth plane {w, h, seq, mm} for absolute time `now`; null unless opts.depth.
  depthPlane(now) {
    if (!this.depth) return null;
    const w = 160, h = 90, mm = new Uint16Array(w * h);
    const bx = 0.5 + 0.28 * Math.sin(now * 0.5), by = 0.5;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let v = 2600;
        if (y > h * 0.72) v = 1400;                                   // torso band
        const dx = (x / w - bx) * (w / h), dy = y / h - by;
        if (dx * dx + dy * dy < 0.012) v = 700;                       // something passing in front
        mm[y * w + x] = v;
      }
    }
    this._planeSeq += 1;
    return { w, h, seq: this._planeSeq, mm };
  }

  // Advance to absolute time `now` (seconds). Returns the messages due.
  step(now) {
    const loop = Math.floor(now / LOOP_S);
    if (loop !== this._lastLoop) { this._fired.clear(); this._lastLoop = loop; }
    const t = now - loop * LOOP_S;
    const out = [];
    const hands = this._hands(t);
    if (this.depth) for (const hd of hands) hd.z = this.handZ(t);
    this.seq += 1;
    out.push({ type: "hands", t: Math.round(now * 1000), seq: this.seq, w: 640, h: 480, hands });
    for (const e of this._events(t)) {
      if (t >= e.at && t < e.at + 0.5 && !this._fired.has(e.at)) {
        this._fired.add(e.at);
        out.push({ type: "gesture", ...e.ev });
      }
    }
    return out;
  }
}
