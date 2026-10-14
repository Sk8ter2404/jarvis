// Self tests for the pure holo modules. Open /holo?selftest=1 in a browser, or
// import runSelfTests() anywhere (no DOM needed). Returns [{name, ok, err}].
import {
  T, GestureRouter, mapToScreen, mapFromScreen, flickToSpin, dampSpin,
  twoHandScale, pinchPoint, palmSize,
} from "./gestures.js";
import { FakeHands, makeLandmarks, LOOP_S } from "./fakehands.js";

class MockScene {
  constructor() {
    this.reactorVisible = true; this.exploded = false; this.menuOpen = false; this.scale = 1;
    this.calls = []; this.partList = [{ name: "Core", info: "i" }]; this.items = [{ model: "arc reactor", x: 500, y: 300 }];
  }
  reactorScreen() { return { x: 500, y: 400, r: 100 }; }
  toScreen(nx, ny) { return { x: nx * 1000, y: ny * 800 }; }
  openMenu(x, y) { this.menuOpen = true; this.calls.push(["openMenu", x, y]); }
  closeMenu() { this.menuOpen = false; }
  menuItemAt(x, y) { return Math.hypot(x - 500, y - 300) < 60 ? this.items[0] : null; }
  chooseMenuItem(it) { this.calls.push(["choose", it.model]); this.menuOpen = false; }
  spawn(m) { this.calls.push(["spawn", m]); }
  setReactorTarget(x, y) { this.calls.push(["target", Math.round(x), Math.round(y)]); }
  setScale(s) { this.scale = s; this.calls.push(["scale", s]); }
  addSpin(wy) { this.calls.push(["spin", wy]); }
  explode() { this.exploded = true; this.calls.push(["explode"]); }
  assemble() { this.exploded = false; this.calls.push(["assemble"]); }
  partAt(x, y) { return Math.hypot(x - 500, y - 400) < 80 ? this.partList[0] : null; }
  pullPart(p) { this.calls.push(["pull", p.name]); }
  movePart(p, x, y) { this.calls.push(["move", p.name, Math.round(x), Math.round(y)]); }
  releasePart(p) { this.calls.push(["release", p.name]); }
  showInfo(p) { this.calls.push(["info", p ? p.name : null]); }
  count(name) { return this.calls.filter(c => c[0] === name).length; }
}

function hand(side, state, extra = {}) {
  const lm = makeLandmarks({ x: 0.5, y: 0.5 });
  return { id: 0, side, lm, state, pinch: 0, palm: [0.5, 0.5], palm_facing: true, ...extra };
}

const eq = (a, b, msg) => { if (a !== b) throw new Error((msg || "") + " expected " + b + " got " + a); };
const near = (a, b, e, msg) => { if (Math.abs(a - b) > e) throw new Error((msg || "") + " expected ~" + b + " got " + a); };

const TESTS = [
  ["thresholds are sane", () => {
    if (!(T.INFO_HOLD_S >= 0.5 && T.INFO_HOLD_S <= 2)) throw new Error("INFO_HOLD_S");
    if (!(T.SPIN_MAX > 0 && T.SPIN_GAIN > 0)) throw new Error("spin consts");
  }],
  ["mapToScreen desktop stretches", () => {
    const p = mapToScreen(0.5, 0.25, 1000, 800, "desktop", 4 / 3);
    eq(p.x, 500); eq(p.y, 200);
  }],
  ["mapToScreen camera cover-fits and inverts", () => {
    const p = mapToScreen(0.5, 0.5, 1920, 1080, "camera", 4 / 3);
    near(p.x, 960, 1e-6); near(p.y, 540, 1e-6);
    const edge = mapToScreen(0, 0, 1920, 1080, "camera", 4 / 3);
    if (!(edge.y < 0)) throw new Error("16:9 window over 4:3 frame must crop top/bottom");
    const [nx, ny] = mapFromScreen(edge.x, edge.y, 1920, 1080, "camera", 4 / 3);
    near(nx, 0, 1e-6); near(ny, 0, 1e-6);
  }],
  ["flick -> spin: deadband, clamp, sign, damping", () => {
    eq(flickToSpin(0.1), 0);
    if (!(flickToSpin(2) > 0 && flickToSpin(-2) < 0)) throw new Error("sign");
    eq(flickToSpin(1000), T.SPIN_MAX);
    const w = dampSpin(10, 1);
    if (!(w < 10 && w > 0)) throw new Error("damp");
  }],
  ["two-hand scale is clamped", () => {
    near(twoHandScale(1, 0.2, 0.4), 2, 1e-9);
    eq(twoHandScale(1, 0.2, 100), T.SCALE_MAX);
    eq(twoHandScale(1, 0.2, 0.001), T.SCALE_MIN);
  }],
  ["landmark generator gives 21 points; pinch closes thumb and index", () => {
    const open = makeLandmarks({ x: 0.5, y: 0.5 });
    eq(open.length, 21);
    const pin = makeLandmarks({ x: 0.5, y: 0.5, pinch: true });
    const d = (a, i, j) => Math.hypot(a[i][0] - a[j][0], a[i][1] - a[j][1]);
    if (!(d(pin, 4, 8) < d(open, 4, 8) * 0.3)) throw new Error("pinch did not close");
    if (!(palmSize(open) > 0)) throw new Error("palm size");
    eq(pinchPoint(open).length, 2);
  }],
  ["palm_hold and pinch_hold open the menu once", () => {
    const s = new MockScene(), r = new GestureRouter(s);
    r.onGesture({ name: "palm_hold", pos: [0.2, 0.3], side: "Right" });
    r.onGesture({ name: "pinch_hold", pos: [0.2, 0.3], side: "Right" });
    eq(s.count("openMenu"), 1);
  }],
  ["pinch while hovering a menu item chooses it", () => {
    const s = new MockScene(), r = new GestureRouter(s);
    s.menuOpen = true;
    const h = hand("Right", "point");
    h.lm[8] = [0.5, 0.375, 0];                      // maps to (500, 300)
    r.update([h], 0.03);
    eq(s.count("choose"), 0);
    r.update([{ ...h, state: "pinch" }], 0.03);
    eq(s.count("choose"), 1);
  }],
  ["fist near the reactor grabs and follows; far fist does not; release lets go", () => {
    const s = new MockScene(), r = new GestureRouter(s);
    r.onGesture({ name: "fist_grab", side: "Right", pos: [0.95, 0.05] });
    eq(r.grab, null, "far fist");
    r.onGesture({ name: "fist_grab", side: "Right", pos: [0.5, 0.5] });
    eq(r.grab.side, "Right");
    r.update([hand("Right", "fist", { palm: [0.6, 0.5] })], 0.03);
    eq(s.calls.filter(c => c[0] === "target").pop()[1], 600);
    r.onGesture({ name: "fist_release", side: "Right" });
    eq(r.grab, null);
  }],
  ["point_flick spins both ways; a tiny flick is ignored", () => {
    const s = new MockScene(), r = new GestureRouter(s);
    r.onGesture({ name: "point_flick", value: 2.4 });
    r.onGesture({ name: "point_flick", value: -2.4 });
    r.onGesture({ name: "point_flick", value: 0.2 });
    eq(s.count("spin"), 2);
    if (!(s.calls[0][1] > 0 && s.calls[1][1] < 0)) throw new Error("direction");
  }],
  ["explode then clap assembles; the pair cannot flap inside the lock window", () => {
    const s = new MockScene(), r = new GestureRouter(s);
    r.onGesture({ name: "explode" });
    eq(s.exploded, true);
    r.onGesture({ name: "clap" });                  // inside the lock: ignored
    eq(s.exploded, true, "clap inside lock");
    r.update([], T.SWITCH_LOCK_S + 0.1);
    r.onGesture({ name: "clap" });
    eq(s.exploded, false);
    r.onGesture({ name: "explode" });               // inside the lock after assemble
    eq(s.exploded, false, "explode inside lock");
  }],
  ["pinch_part pulls a part, info appears after the hold, release lets go", () => {
    const s = new MockScene(), r = new GestureRouter(s);
    s.exploded = true;
    r.onGesture({ name: "pinch_part", side: "Right", pos: [0.5, 0.5] });
    eq(s.count("pull"), 1);
    const h = hand("Right", "pinch");
    for (let i = 0; i < 20; i++) r.update([h], 0.03);   // 0.6 s: no info yet
    eq(s.count("info"), 0);
    for (let i = 0; i < 20; i++) r.update([h], 0.03);   // 1.2 s
    eq(s.count("info"), 1);
    r.update([hand("Right", "open")], 0.2);
    eq(s.count("release"), 1);
    eq(s.calls.filter(c => c[0] === "info").pop()[1], null, "info cleared");
  }],
  ["pinch_part ignored when not exploded or nothing under the pinch", () => {
    const s = new MockScene(), r = new GestureRouter(s);
    r.onGesture({ name: "pinch_part", side: "Right", pos: [0.5, 0.5] });
    eq(s.count("pull"), 0);
    s.exploded = true;
    r.onGesture({ name: "pinch_part", side: "Right", pos: [0.05, 0.05] });
    eq(s.count("pull"), 0);
  }],
  ["two pinching hands scale the reactor", () => {
    const s = new MockScene(), r = new GestureRouter(s);
    const a = hand("Left", "pinch"), b = hand("Right", "pinch");
    a.lm = makeLandmarks({ x: 0.3, y: 0.5, pinch: true });
    b.lm = makeLandmarks({ x: 0.6, y: 0.5, pinch: true });
    r.update([a, b], 0.03);                         // sets the baseline
    b.lm = makeLandmarks({ x: 0.9, y: 0.5, pinch: true });
    r.update([a, b], 0.03);
    if (!(s.scale > 1.5)) throw new Error("scale did not grow: " + s.scale);
  }],
  ["fake hands loop emits the legend's gestures in shape", () => {
    const f = new FakeHands({});
    const names = new Set();
    for (let t = 0; t < LOOP_S; t += 0.033) {
      for (const m of f.step(t)) {
        if (m.type === "hands") {
          for (const h of m.hands) {
            eq(h.lm.length, 21); if (!h.palm || !h.state) throw new Error("hand fields");
          }
        } else if (m.type === "gesture") names.add(m.name);
      }
    }
    for (const n of ["palm_hold", "fist_grab", "fist_release", "point_flick", "explode", "pinch_part", "clap"]) {
      if (!names.has(n)) throw new Error("missing " + n);
    }
  }],
];

export function runSelfTests() {
  return TESTS.map(([name, fn]) => {
    try { fn(); return { name, ok: true }; }
    catch (e) { return { name, ok: false, err: String(e && e.message || e) }; }
  });
}
