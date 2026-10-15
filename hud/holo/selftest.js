// Self tests for the pure holo modules. Open /holo?selftest=1 in a browser, or
// import runSelfTests() anywhere (no DOM needed). Returns [{name, ok, err}].
import {
  T, GestureRouter, mapToScreen, mapFromScreen, flickToSpin, dampSpin,
  twoHandScale, pinchPoint, palmSize,
} from "./gestures.js";
import { FakeHands, makeLandmarks, LOOP_S } from "./fakehands.js";
import {
  D, parseDepthMessage, isDepthMessage, occlusionMask, maskCoverage, medianDepthM,
  defaultHoloZ, spawnZ, sizeAtDistance, PushPull,
} from "./depth.js";
import {
  formatSizeMm, uniqueNames, splitParts, fitModel, explodeDir, findModel, pageOf,
  TARGET_RADIUS, MODELS_PER_PAGE,
} from "./models.js";
import {
  X, XRGestureRouter, ApartDetector, ClapDetector, PokeGate, pinchState,
  pinchPoint as xrPinchPoint,
  palmCenter, scaleFromDistance, quatFromVectors, quatMul, quatRotate, placeInFront,
  panelBasis, pokeHit, dist3, add3,
} from "./xrgestures.js";
import { FakeXRHands, makeXRHand } from "./xrfake.js";

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
  beginPushPull(z) { this.calls.push(["ppBegin", z]); }
  pushPull(z, dt) { this.calls.push(["pp", z, dt]); }
  endPushPull() { this.calls.push(["ppEnd"]); }
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
    eq(flickToSpin(90), 0);
    if (!(flickToSpin(300) > 0 && flickToSpin(-300) < 0)) throw new Error("sign");
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
    r.onGesture({ name: "point_flick", value: 420 });
    r.onGesture({ name: "point_flick", value: -420 });
    r.onGesture({ name: "point_flick", value: 120 });
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
  ["pinch_start on a part pulls it, info appears after the hold, release lets go", () => {
    const s = new MockScene(), r = new GestureRouter(s);
    s.exploded = true;
    r.onGesture({ name: "pinch_start", side: "Right", pos: [0.5, 0.5] });
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
    for (const n of ["palm_hold", "fist_grab", "fist_release", "point_flick", "explode", "pinch_start", "clap"]) {
      if (!names.has(n)) throw new Error("missing " + n);
    }
  }],
  ["model sizes print in mm with sensible decimals", () => {
    eq(formatSizeMm(24, 18.52, 3.2), "24.0 x 18.5 x 3.2 mm");
    eq(formatSizeMm(120.4, 5, 0.04), "120 x 5.0 x 0.0 mm");
  }],
  ["part names: blanks get numbers, repeats get a suffix", () => {
    const n = uniqueNames(["Body", "", "Body", "Lid", "Body"]);
    eq(n.join("|"), "Body|Part 02|Body 2|Lid|Body 3");
  }],
  ["splitParts: 3MF objects / components become parts, one object stays one", () => {
    const mesh = () => ({ isMesh: true, children: [] });
    const grp = (...k) => ({ children: k });
    const a = grp(mesh()), b = grp(mesh()), c = grp(mesh(), mesh());
    eq(splitParts(grp(a, b, c)).length, 3, "three build items");
    eq(splitParts(grp(c))[0], c, "one object with two meshes is one part");
    const composite = grp(a, b);                       // one build item made of components
    eq(splitParts(grp(composite)).length, 2, "components");
    eq(splitParts(grp(grp(grp(a, b)))).length, 2, "wrapper groups are looked through");
    const m1 = mesh();
    eq(splitParts(m1)[0], m1, "bare mesh (STL)");
    const obj = grp(mesh(), mesh(), mesh());
    eq(splitParts(obj).length, 1, "several meshes stay one object by default");
    eq(splitParts(obj, { meshesAreParts: true }).length, 3, "OBJ objects are parts");
  }],
  ["fitModel centres the box and scales its half-diagonal to the reactor", () => {
    const f = fitModel([0, 0, 0], [100, 0, 0]);
    near(f.center[0], 50, 1e-9); near(f.radius, 50, 1e-9); near(f.scale, TARGET_RADIUS / 50, 1e-9);
    eq(fitModel([1, 1, 1], [1, 1, 1]).scale, 1, "degenerate box");
    const g = fitModel([-5, -5, -5], [5, 5, 5]);
    near(g.radius * g.scale, TARGET_RADIUS, 1e-9);
  }],
  ["explodeDir points away from the centre, centre parts go front/back", () => {
    const e = explodeDir([2, 0, 0], 0);
    if (!(e.dir[0] > 0.9 && e.dist > 0)) throw new Error("outward");
    near(Math.hypot(...e.dir), 1, 1e-9, "unit");
    eq(explodeDir([0, 0, 0], 0).dir[2], 1);
    eq(explodeDir([0, 0, 0], 1).dir[2], -1);
  }],
  ["findModel: id, exact name, unique prefix; ambiguity and junk give null", () => {
    const L = [{ id: "aaaaaaaaaaaa", name: "Tiger Pendant" }, { id: "bbbbbbbbbbbb", name: "Tiger Stand" },
      { id: "cccccccccccc", name: "Gear" }];
    eq(findModel(L, "id:bbbbbbbbbbbb").name, "Tiger Stand");
    eq(findModel(L, "tiger pendant").id, "aaaaaaaaaaaa");
    eq(findModel(L, "gea").name, "Gear");
    eq(findModel(L, "tiger"), null, "ambiguous");
    eq(findModel(L, "nope"), null);
    eq(findModel(L, ""), null);
    eq(findModel(null, "gear"), null);
  }],
  ["pageOf pages the model picker", () => {
    const L = Array.from({ length: MODELS_PER_PAGE + 2 }, (_, i) => ({ id: String(i) }));
    const p0 = pageOf(L, 0), p1 = pageOf(L, 5);
    eq(p0.items.length, MODELS_PER_PAGE); eq(p0.more, true); eq(p0.pages, 2);
    eq(p1.page, 1); eq(p1.items.length, 2); eq(p1.more, false);
    eq(pageOf([], 0).items.length, 0);
  }],
];

// ---- Kinect depth (Phase 2) -------------------------------------------------
// A wire message exactly as core/handtrack_depth.py pack_wire() lays it out.
function wireMessage(w, h, mm, { seq = 7, flags = 1, version = 1, magic = "JVDP" } = {}) {
  const buf = new ArrayBuffer(D.HEADER + w * h * 2);
  const dv = new DataView(buf);
  for (let i = 0; i < 4; i++) dv.setUint8(i, magic.charCodeAt(i));
  dv.setUint8(4, version); dv.setUint8(5, flags);
  dv.setUint16(6, w, true); dv.setUint16(8, h, true);
  dv.setBigUint64(12, BigInt(seq), true);
  dv.setUint16(20, 500, true); dv.setUint16(22, 4500, true);
  for (let i = 0; i < w * h; i++) dv.setUint16(D.HEADER + i * 2, mm[i], true);
  return buf;
}
function planeOf(w, h, fill, boxes = []) {
  const mm = new Uint16Array(w * h).fill(fill);
  for (const [x0, y0, x1, y1, v] of boxes) {
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) mm[y * w + x] = v;
  }
  return mm;
}

TESTS.push(
  ["depth wire message parses; wrong magic / version / length are refused", () => {
    const mm = planeOf(4, 3, 1234);
    const p = parseDepthMessage(wireMessage(4, 3, mm));
    eq(p.w, 4); eq(p.h, 3); eq(p.seq, 7); eq(p.mirrored, true);
    eq(p.nearMm, 500); eq(p.farMm, 4500); eq(p.mm[5], 1234); eq(p.mm.length, 12);
    eq(parseDepthMessage(wireMessage(4, 3, mm, { magic: "JVDX" })), null, "magic");
    eq(parseDepthMessage(wireMessage(4, 3, mm, { version: 2 })), null, "version");
    eq(parseDepthMessage(wireMessage(4, 3, mm).slice(0, 40)), null, "short");
    eq(parseDepthMessage(null), null);
    eq(parseDepthMessage(new ArrayBuffer(10)), null);
  }],
  ["isDepthMessage tells a depth plane from a JPEG", () => {
    eq(isDepthMessage(wireMessage(2, 2, planeOf(2, 2, 900))), true);
    const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 0]).buffer;
    eq(isDepthMessage(jpeg), false);
    eq(isDepthMessage(new ArrayBuffer(2)), false);
  }],
  ["occlusion: only what is nearer than the hologram hides it; unknown never does", () => {
    const w = 10, h = 8;
    const mm = planeOf(w, h, 3000, [
      [1, 1, 4, 4, 700],            // nearer than the hologram -> hides
      [6, 1, 9, 4, 1800],           // behind the hologram -> shows
      [1, 5, 4, 8, 0],              // unknown -> shows
    ]);
    const m = occlusionMask(mm, w, h, 1.0);
    eq(m[2 * w + 2], 255, "nearer pixel hides");
    eq(m[2 * w + 7], 0, "farther pixel shows");
    eq(m[6 * w + 2], 0, "unknown shows");
    eq(m[7 * w + 9], 0, "background shows");
    near(maskCoverage(m), 9 / (w * h), 1e-9, "3x3 block");
  }],
  ["occlusion: the margin and single-pixel noise are respected", () => {
    const w = 8, h = 6;
    const lim = Math.round((1.0 - D.OCCLUDE_MARGIN_M) * 1000);
    const at = v => occlusionMask(planeOf(w, h, 3000, [[2, 2, 4, 4, v]]), w, h, 1.0)[2 * w + 2];
    eq(at(lim - 1), 255, "just inside the margin hides");
    eq(at(lim), 0, "at the margin shows");
    eq(at(lim + 20), 0, "level with the hologram shows");
    const lone = occlusionMask(planeOf(w, h, 3000, [[3, 3, 4, 4, 600]]), w, h, 1.0);
    eq(maskCoverage(lone), 0, "an isolated pixel is noise");
    const pair = occlusionMask(planeOf(w, h, 3000, [[3, 3, 5, 4, 600]]), w, h, 1.0);
    eq(maskCoverage(pair) > 0, true, "two neighbours are an object");
    // moving the hologram nearer stops the same object hiding it
    eq(occlusionMask(planeOf(w, h, 3000, [[2, 2, 4, 4, 900]]), w, h, 0.8)[2 * w + 2], 0, "now behind");
  }],
  ["palm depth: median of a window, unknown ignored, too few samples = null", () => {
    const w = 20, h = 10;
    const mm = planeOf(w, h, 0, [[8, 4, 13, 7, 800]]);
    near(medianDepthM(mm, w, h, 0.5, 0.5), 0.8, 1e-9);
    eq(medianDepthM(mm, w, h, 0.05, 0.05), null, "all unknown");
    mm[5 * w + 10] = 900; mm[5 * w + 11] = 700;                       // outliers do not move a median
    near(medianDepthM(mm, w, h, 0.5, 0.5), 0.8, 1e-9);
    eq(medianDepthM(mm, w, h, 1.5, 0.5), null, "outside the picture");
  }],
  ["hologram distance: spawn in front of the hand, default in front of the nearest 10 %", () => {
    near(spawnZ(0.8), 0.8 - D.SPAWN_IN_FRONT_M, 1e-9);
    eq(spawnZ(null), D.HOLO_Z_DEFAULT);
    eq(spawnZ(0.1), D.HOLO_Z_MIN, "clamped near");
    eq(spawnZ(9), D.HOLO_Z_MAX, "clamped far");
    const mm = planeOf(20, 10, 2600, [[0, 6, 20, 10, 1200]]);          // 40 % of the view is a person at 1.2 m
    near(defaultHoloZ(mm), 1.1, 1e-9);
    eq(defaultHoloZ(new Uint16Array(200)), D.HOLO_Z_DEFAULT, "no depth: default");
  }],
  ["size follows perspective and is clamped", () => {
    eq(sizeAtDistance(1, 1), 1);
    near(sizeAtDistance(1, 0.5), 2, 1e-9);
    near(sizeAtDistance(1, 2), D.SIZE_MIN, 1e-9);
    eq(sizeAtDistance(1, 0.1), D.SIZE_MAX);
    eq(sizeAtDistance(1, 0), 1, "no distance, no change");
  }],
  ["push/pull: a fist moving toward the sensor brings the hologram nearer and bigger", () => {
    const p = new PushPull();
    eq(p.begin(null, 1.0), false, "no hand depth, no grab");
    eq(p.begin(0.9, 1.0), true);
    let z = 1.0;
    for (let i = 0; i < 90; i++) z = p.update(0.6, 0.033);            // hand 0.30 m nearer, held 3 s
    near(z, 1.0 - (0.30 - D.PUSH_DEADBAND_M), 0.01, "nearer by the travel minus the dead band");
    if (!(sizeAtDistance(1.0, z) > 1.3)) throw new Error("bigger when nearer");
    for (let i = 0; i < 150; i++) z = p.update(1.4, 0.033);           // pulled 0.5 m past the start
    near(z, 1.0 + (0.5 - D.PUSH_DEADBAND_M), 0.01, "farther");
    if (!(sizeAtDistance(1.0, z) < 0.8)) throw new Error("smaller when farther");
    eq(p.update(null, 0.033), z, "no depth this frame: hold");
    p.end();
    eq(p.update(0.1, 0.033), z, "inactive: no change");
  }],
  ["push/pull: jitter inside the dead band does nothing; limits hold", () => {
    const p = new PushPull();
    p.begin(0.9, 1.0);
    let z = 1.0;
    for (let i = 0; i < 60; i++) z = p.update(0.9 + 0.01 * Math.sin(i), 0.033);
    eq(z, 1.0, "centimetre jitter");
    for (let i = 0; i < 300; i++) z = p.update(0.3, 0.033);
    eq(z, D.HOLO_Z_MIN, "never nearer than the minimum");
    for (let i = 0; i < 600; i++) z = p.update(4.5, 0.033);
    eq(z, D.HOLO_Z_MAX, "never farther than the maximum");
  }],
  ["router: a fist with depth begins, drives and ends push/pull; without depth it never does", () => {
    const s = new MockScene(), r = new GestureRouter(s);
    r.onGesture({ name: "fist_grab", side: "Right", pos: [0.5, 0.5] });
    r.update([hand("Right", "fist", { z: 0.9 })], 0.03);
    r.update([hand("Right", "fist", { z: 0.7 })], 0.03);
    eq(s.count("ppBegin"), 1); eq(s.count("pp"), 1);
    eq(s.calls.filter(c => c[0] === "pp").pop()[1], 0.7);
    r.onGesture({ name: "fist_release", side: "Right" });
    eq(s.count("ppEnd"), 1);
    const s2 = new MockScene(), r2 = new GestureRouter(s2);
    r2.onGesture({ name: "fist_grab", side: "Right", pos: [0.5, 0.5] });
    r2.update([hand("Right", "fist")], 0.03);                        // no "z" key at all
    r2.update([hand("Right", "fist", { z: null })], 0.03);
    r2.onGesture({ name: "fist_release", side: "Right" });
    eq(s2.count("ppBegin") + s2.count("pp") + s2.count("ppEnd"), 0, "no depth, no push/pull");
  }],
  ["router: opening the hand ends push/pull; a scene without the hooks is untouched", () => {
    const s = new MockScene(), r = new GestureRouter(s);
    r.onGesture({ name: "fist_grab", side: "Right", pos: [0.5, 0.5] });
    r.update([hand("Right", "fist", { z: 0.9 })], 0.03);
    r.update([hand("Right", "open", { z: 0.9 })], 0.3);              // let go without fist_release
    eq(s.count("ppEnd"), 1);
    const bare = new MockScene();
    bare.beginPushPull = bare.pushPull = bare.endPushPull = undefined;
    const rb = new GestureRouter(bare);
    rb.onGesture({ name: "fist_grab", side: "Right", pos: [0.5, 0.5] });
    rb.update([hand("Right", "fist", { z: 0.9 })], 0.03);
    rb.onGesture({ name: "fist_release", side: "Right" });
    eq(rb.grab, null);
  }],
  ["fake Kinect: hands carry z, the plane has a near blob over a far wall", () => {
    const f = new FakeHands({}, { depth: true });
    const msgs = f.step(5.0).filter(m => m.type === "hands");
    eq(typeof msgs[0].hands[0].z, "number");
    const plane = f.depthPlane(5.0);
    eq(plane.w, 160); eq(plane.h, 90); eq(plane.mm.length, 160 * 90);
    const vals = new Set(plane.mm);
    for (const v of [700, 1400, 2600]) if (!vals.has(v)) throw new Error("missing " + v);
    eq(new FakeHands({}).depthPlane(1), null, "no depth unless asked");
    eq("z" in new FakeHands({}).step(5.0)[0].hands[0], false, "no z unless asked");
  }],
);

// ---------------------------------------------------------------------------
// Phase 3: the Quest 3 / WebXR page (xrgestures.js, xrfake.js) - pure math plus a
// mock target; xr.js itself needs a headset (or /holo/xr?fake=1).
// ---------------------------------------------------------------------------
class MockXRTarget {
  constructor() {
    this.pos = [0, 1.2, -0.5]; this.s = 1; this.q = [0, 0, 0, 1];
    this.exploded = false; this.calls = []; this.snap = null;
    this.part = { name: "Core", info: "i" };
    this.infoShown = null; this.pulled = null; this.pullAt = null; this.noSphere = false;
  }
  sphere() { return this.noSphere ? null : { c: this.pos.slice(), r: (this.exploded ? 0.28 : 0.14) * this.s }; }
  position() { return this.pos.slice(); }
  getScale() { return this.s; }
  grabMove(p) { this.pos = p.slice(); this.calls.push(["grabMove"]); }
  grabEnd() { this.calls.push(["grabEnd"]); }
  twoBegin() { this.snap = { pos: this.pos.slice(), s: this.s, q: this.q.slice() }; this.calls.push(["twoBegin"]); }
  twoMove(m) {
    this.pos = m.pos.slice(); this.s = m.scale; this.q = quatMul(m.rot, this.snap.q);
    this.maxS = Math.max(this.maxS || 0, this.s); this.calls.push(["twoMove"]);
  }
  twoEnd(revert) {
    if (revert && this.snap) { this.pos = this.snap.pos; this.s = this.snap.s; this.q = this.snap.q; }
    this.snap = null; this.calls.push(["twoEnd", revert]);
  }
  explode() { this.exploded = true; this.calls.push(["explode"]); }
  assemble() { this.exploded = false; this.calls.push(["assemble"]); }
  corePos() { return add3(this.pos, [0.2, 0.05, 0]); }
  partAt(p) { return dist3(p, this.corePos()) <= 0.03 * this.s + X.PART_MARGIN_M ? this.part : null; }
  pullPart(part) { this.pulled = part; this.calls.push(["pull", part.name]); }
  movePart(part, p) { this.pullAt = p; this.calls.push(["movePart"]); }
  releasePart(part) { this.pulled = null; this.calls.push(["release", part.name]); }
  showInfo(part) { this.infoShown = part ? part.name : null; this.calls.push(["info", this.infoShown]); }
  count(n) { return this.calls.filter(c => c[0] === n).length; }
}

const DT = 1 / 72;                                   // the Quest's hand-tracking rate
// A hand whose thumb and index tips are `gap` apart around `at`.
function xh(at, gap, palmOff = [0, -0.02, 0.08]) {
  const h = makeXRHand(add3(at, palmOff), at, false);
  h.joints["thumb-tip"] = add3(at, [-gap / 2, 0, 0]);
  h.joints["index-finger-tip"] = add3(at, [gap / 2, 0, 0]);
  return h;
}
const openH = at => xh(at, 0.075), pin = at => xh(at, 0.008);
// Run `n` frames; f(i, u) -> {left, right}; u = 0..1 through the run.
function drive(r, n, f) {
  for (let i = 0; i < n; i++) {
    const h = f(i, n <= 1 ? 1 : i / (n - 1));
    r.update({ left: h.left || null, right: h.right || null }, DT);
  }
}
const sec = s => Math.round(s / DT);
const lerp3 = (a, b, u) => [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
const C0 = [0, 1.2, -0.5];
const vclose = (a, b, e, msg) => { for (let i = 0; i < 3; i++) near(a[i], b[i], e, (msg || "") + "[" + i + "]"); };
const pair = (wl, wr, z = 0, y = 0) => ({ left: add3(C0, [-wl, y, z]), right: add3(C0, [wr, y, z]) });

TESTS.push(
  ["xr: thresholds are sane", () => {
    if (!(X.PINCH_ON_M < X.PINCH_OFF_M)) throw new Error("hysteresis band");
    if (!(X.SCALE_MIN < 1 && X.SCALE_MAX > 1)) throw new Error("scale limits");
    if (!(X.INFO_HOLD_S >= 0.5 && X.INFO_HOLD_S <= 2)) throw new Error("info hold");
    if (!(X.APART_SPEED_MPS > 0.5 && X.CLAP_SPEED_MPS >= X.APART_SPEED_MPS)) throw new Error("speeds");
  }],
  ["xr: quaternion helpers rotate vectors the way they claim", () => {
    const q = quatFromVectors([1, 0, 0], [0, 1, 0]);
    vclose(quatRotate(q, [1, 0, 0]), [0, 1, 0], 1e-9, "x->y");
    vclose(quatRotate(q, [0, 0, 1]), [0, 0, 1], 1e-9, "axis z fixed");
    const flip = quatFromVectors([0, 0, 1], [0, 0, -1]);
    vclose(quatRotate(flip, [0, 0, 1]), [0, 0, -1], 1e-9, "180 deg");
    vclose(quatRotate(quatFromVectors([0, 1, 0], [0, 1, 0]), [1, 2, 3]), [1, 2, 3], 1e-9, "identity");
    const a = quatFromVectors([1, 0, 0], [0, 1, 0]), b = quatFromVectors([0, 1, 0], [0, 0, 1]);
    vclose(quatRotate(quatMul(b, a), [1, 0, 0]), [0, 0, 1], 1e-9, "b after a");
    const yaw90 = [0, Math.sin(Math.PI / 4), 0, Math.cos(Math.PI / 4)];
    vclose(quatRotate(yaw90, [0, 0, -1]), [-1, 0, 0], 1e-9, "yaw +90 turns -z toward -x");
  }],
  ["xr: pinch has hysteresis (on <= 2.0 cm, off >= 3.2 cm)", () => {
    eq(pinchState(false, 0.019), true);
    eq(pinchState(false, 0.025), false, "between: stays open");
    eq(pinchState(true, 0.025), true, "between: stays pinched");
    eq(pinchState(true, 0.033), false);
    near(dist3(xrPinchPoint(openH(C0)), C0), 0, 1e-12);
    vclose(palmCenter(makeXRHand([0, 1, 0], [0, 1, -0.1], false)), [0, 1, 0], 1e-12, "palm");
  }],
  ["xr: scale follows hand distance and clamps", () => {
    near(scaleFromDistance(1, 0.2, 0.4), 2, 1e-9);
    eq(scaleFromDistance(1, 0.2, 99), X.SCALE_MAX);
    eq(scaleFromDistance(1, 0.2, 0.0001), X.SCALE_MIN);
    eq(scaleFromDistance(1.5, 0, 0.3), 1.5, "no start distance: unchanged");
  }],
  ["xr: the hologram appears 0.5 m ahead at chest height, from the horizontal facing", () => {
    vclose(placeInFront([0, 1.6, 0], [0, 0, 0, 1]), [0, 1.3, -0.5], 1e-9, "facing -z");
    const yaw90 = [0, Math.sin(Math.PI / 4), 0, Math.cos(Math.PI / 4)];       // faces -x
    vclose(placeInFront([1, 1.6, 2], yaw90), [0.5, 1.3, 2], 1e-9, "facing -x");
    const pitchDown = [Math.sin(-Math.PI / 4), 0, 0, Math.cos(-Math.PI / 4)]; // looking at the floor
    vclose(placeInFront([0, 1.6, 0], pitchDown), [0, 1.3, -0.5], 1e-9, "looking down: still ahead");
    const pitchUp = [Math.sin(Math.PI / 4), 0, 0, Math.cos(Math.PI / 4)];     // looking at the ceiling
    vclose(placeInFront([0, 1.6, 0], pitchUp), [0, 1.3, -0.5], 1e-9, "looking up: still ahead");
  }],
  ["xr: a pinch near the object grabs and follows without a snap; release lets go", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    const at = add3(C0, [0.05, 0, 0]), to = add3(at, [0.2, 0.1, -0.1]);
    drive(r, 5, () => ({ right: openH(at) }));
    eq(t.count("grabMove"), 0, "an open hand does nothing");
    drive(r, 1, () => ({ right: pin(at) }));
    eq(r.mode.kind, "grab");
    drive(r, sec(0.5), (i, u) => ({ right: pin(lerp3(at, to, u)) }));
    vclose(t.pos, add3(C0, [0.2, 0.1, -0.1]), 0.005, "the object kept its offset to the hand");
    drive(r, 3, () => ({ right: openH(to) }));
    eq(r.mode, null); eq(t.count("grabEnd"), 1);
  }],
  ["xr: a pinch far away, or one that drifts onto the object, never grabs", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    const far = add3(C0, [0.6, 0, 0]);
    drive(r, 4, () => ({ right: pin(far) }));
    drive(r, sec(0.4), (i, u) => ({ right: pin(lerp3(far, C0, u)) }));     // keeps pinching, moves onto it
    eq(r.mode, null); eq(t.count("grabMove"), 0);
    drive(r, 3, () => ({ right: openH(C0) }));                             // lets go, pinches again on it
    drive(r, 2, () => ({ right: pin(C0) }));
    eq(r.mode.kind, "grab");
  }],
  ["xr: pinch jitter inside the hysteresis band does not drop the grab", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    drive(r, 2, () => ({ right: pin(C0) }));
    eq(r.mode.kind, "grab");
    drive(r, 40, i => ({ right: xh(C0, i % 2 ? 0.028 : 0.012) }));       // wobbles between on and off
    eq(r.mode.kind, "grab"); eq(t.count("grabEnd"), 0);
  }],
  ["xr: a short tracking dropout keeps the grab; a long one drops it", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    drive(r, 2, () => ({ right: pin(C0) }));
    drive(r, sec(0.1), () => ({}));                                       // 0.1 s with no hand
    drive(r, 2, () => ({ right: pin(C0) }));
    eq(t.count("grabEnd"), 0, "kept");
    drive(r, sec(0.3), () => ({}));
    eq(t.count("grabEnd"), 1, "dropped");
  }],
  ["xr: nothing on screen means nothing to grab", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    t.noSphere = true;
    drive(r, 5, () => ({ right: pin(C0) }));
    eq(r.mode, null);
  }],
  ["xr: two hands scale with their distance, rotate with their vector, hold the object between them", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    const both = p => ({ left: pin(p.left), right: pin(p.right) });
    drive(r, 2, () => both(pair(0.09, 0.09)));
    eq(r.mode.kind, "two");
    drive(r, sec(2), (i, u) => both(pair(0.09 + 0.09 * u, 0.09 + 0.09 * u)));
    near(t.s, 2, 0.01, "twice the distance, twice the size");
    eq(t.count("explode"), 0, "a slow spread is a scale, not an explode");
    // now turn the pair 90 deg about the vertical axis (right hand toward the user)
    drive(r, sec(2), (i, u) => {
      const a = u * Math.PI / 2, w = 0.18;
      return { left: pin(add3(C0, [-w * Math.cos(a), 0, w * Math.sin(a)])), right: pin(add3(C0, [w * Math.cos(a), 0, -w * Math.sin(a)])) };
    });
    vclose(quatRotate(t.q, [1, 0, 0]), [0, 0, -1], 0.02, "the object turned with the hands");
    near(t.s, 2, 0.01, "turning does not change the size");
    drive(r, sec(0.3), () => ({}));                                       // both hands gone for good
    eq(t.count("twoEnd"), 1);
  }],
  ["xr: two-hand scale clamps", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    const both = p => ({ left: pin(p.left), right: pin(p.right) });
    drive(r, 2, () => both(pair(0.09, 0.09)));
    drive(r, sec(20), (i, u) => both(pair(0.09 + 0.9 * u, 0.09 + 0.9 * u)));
    eq(t.s, X.SCALE_MAX);
    drive(r, sec(20), (i, u) => both(pair(0.99 - 0.98 * u, 0.99 - 0.98 * u)));
    eq(t.s, X.SCALE_MIN);
  }],
  ["xr: one hand letting go of a two-hand hold hands the object to the other", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    drive(r, 2, () => ({ left: pin(pair(0.09, 0.09).left), right: pin(pair(0.09, 0.09).right) }));
    eq(r.mode.kind, "two");
    drive(r, sec(0.3), () => ({ right: pin(pair(0.09, 0.09).right) }));   // the left hand is gone
    eq(r.mode && r.mode.kind, "grab", "the other hand keeps holding it");
    eq(t.count("twoEnd"), 1);
  }],
  ["xr: pulling both pinching hands apart FAST explodes (and undoes the scale it looked like)", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    const both = p => ({ left: pin(p.left), right: pin(p.right) });
    drive(r, 3, () => both(pair(0.07, 0.07)));
    drive(r, sec(0.3), (i, u) => both(pair(0.07 + 0.3 * u, 0.07 + 0.3 * u)));
    eq(t.count("explode"), 1); eq(t.exploded, true);
    eq(t.s, 1, "the fast pull did not leave the size changed");
    if (!(t.maxS < 1.3)) throw new Error("the yank swelled the model before it exploded (max " + t.maxS + ")");
    eq(r.mode.kind, "wait");
    drive(r, sec(0.5), () => both(pair(0.4, 0.4)));
    eq(t.count("explode"), 1, "holding the pinch does not explode again");
  }],
  ["xr: once exploded, a fast pull is just a scale (no second explode)", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    const both = p => ({ left: pin(p.left), right: pin(p.right) });
    t.exploded = true;
    drive(r, 3, () => both(pair(0.09, 0.09)));
    drive(r, sec(0.3), (i, u) => both(pair(0.09 + 0.3 * u, 0.09 + 0.3 * u)));
    eq(t.count("explode"), 0);
    if (!(t.s > 2)) throw new Error("expected a bigger object, got " + t.s);
  }],
  ["xr: apart detector - slow spreads and short jumps do not fire", () => {
    const d = new ApartDetector();
    let fired = false;
    for (let i = 0; i < 200; i++) if (d.update(i * DT, 0.1 + 0.3 * i * DT)) fired = true;     // 0.3 m/s
    eq(fired, false, "slow");
    const d2 = new ApartDetector();
    for (let i = 0; i < 20; i++) if (d2.update(i * DT, 0.1 + (i > 10 ? 0.04 : 0))) fired = true;
    eq(fired, false, "4 cm");
    const d3 = new ApartDetector();
    for (let i = 0; i < 40 && !fired; i++) fired = d3.update(i * DT, 0.1 + 1.5 * i * DT);      // 1.5 m/s
    eq(fired, true, "fast");
  }],
  ["xr: a clap (palms meet fast) assembles an exploded model, only that", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    const hands = w => ({ left: openH(add3(C0, [-w, 0, 0.2])), right: openH(add3(C0, [w, 0, 0.2])) });
    t.exploded = true;
    drive(r, sec(0.5), () => hands(0.3));
    drive(r, sec(0.2), (i, u) => hands(0.3 - 0.27 * u));
    eq(t.count("assemble"), 1); eq(t.exploded, false);
    // not exploded: a clap changes nothing
    drive(r, sec(1.5), () => hands(0.3));
    drive(r, sec(0.2), (i, u) => hands(0.3 - 0.27 * u));
    eq(t.count("assemble"), 1);
  }],
  ["xr: bringing the hands together slowly, or while pinching, is not a clap", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    const hands = (w, f) => ({ left: f(add3(C0, [-w, 0, 0.2])), right: f(add3(C0, [w, 0, 0.2])) });
    t.exploded = true;
    drive(r, sec(2), (i, u) => hands(0.3 - 0.27 * u, openH));
    eq(t.count("assemble"), 0, "slow");
    const t2 = new MockXRTarget(), r2 = new XRGestureRouter(t2);
    t2.exploded = true;
    drive(r2, sec(0.5), () => hands(0.6, pin));
    drive(r2, sec(0.2), (i, u) => hands(0.6 - 0.57 * u, pin));
    eq(t2.count("assemble"), 0, "pinching hands meeting");
  }],
  ["xr: hands that vanish at contact still count as the clap", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    const hands = w => ({ left: openH(add3(C0, [-w, 0, 0.2])), right: openH(add3(C0, [w, 0, 0.2])) });
    t.exploded = true;
    drive(r, sec(0.5), () => hands(0.3));
    drive(r, sec(0.1), (i, u) => hands(0.3 - 0.22 * u));
    drive(r, 3, () => ({}));                                              // both lost at contact
    eq(t.count("assemble"), 1);
  }],
  ["xr: the clap detector has a cooldown", () => {
    const c = new ClapDetector();
    let n = 0;
    for (let k = 0; k < 2; k++) {
      for (let i = 0; i <= 14; i++) if (c.update(k * 0.4 + i * DT, 0.5 - 0.46 * i / 14)) n++;
    }
    eq(n, 1);
  }],
  ["xr: exploded, a pinch picks a part; holding still ~1 s shows its info; moving hides it", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    t.exploded = true;
    const pc = t.corePos();
    drive(r, 3, () => ({ right: pin(pc) }));
    eq(r.mode.kind, "part"); eq(t.pulled.name, "Core");
    drive(r, sec(0.8), () => ({ right: pin(pc) }));
    eq(t.infoShown, null, "not yet");
    drive(r, sec(0.4), () => ({ right: pin(pc) }));
    eq(t.infoShown, "Core", "after about a second");
    drive(r, sec(0.2), (i, u) => ({ right: pin(add3(pc, [0.3 * u, 0, 0])) }));       // yank it away
    eq(t.infoShown, null, "moving hides the card");
    vclose(t.pullAt, add3(pc, [0.3, 0, 0]), 0.01, "the part followed the hand");
    drive(r, 3, () => ({ right: openH(pc) }));
    eq(t.pulled, null); eq(t.count("release"), 1);
    eq(r.mode, null);
  }],
  ["xr: assembling by voice while a part is held lets it go", () => {
    const t = new MockXRTarget(), r = new XRGestureRouter(t);
    t.exploded = true;
    drive(r, 3, () => ({ right: pin(t.corePos()) }));
    eq(r.mode.kind, "part");
    t.exploded = false;
    drive(r, 2, () => ({ right: pin(t.corePos()) }));
    eq(r.mode, null); eq(t.count("release"), 1);
  }],
  ["xr: a poke presses a shelf panel once per touch and re-arms on pulling back", () => {
    const c = [0, 1.2, -0.5], head = [0, 1.5, 0.3];
    const basis = panelBasis(c, head), panel = { c, hw: 0.085, hh: 0.025 };
    vclose(basis.n, [0, 0, 1], 1e-9, "faces the viewer"); vclose(basis.right, [1, 0, 0], 1e-9, "viewer's right");
    const at = (x, y, z) => add3(c, [x, y, z]);
    eq(pokeHit(at(0.02, 0.01, 0.005), panel, basis).hit, true);
    eq(pokeHit(at(0.12, 0, 0.005), panel, basis).hit, false, "beside it");
    eq(pokeHit(at(0, 0.05, 0.005), panel, basis).hit, false, "above it");
    eq(pokeHit(at(0, 0, 0.05), panel, basis).hit, false, "5 cm in front: not touching");
    eq(pokeHit(at(0, 0, -0.06), panel, basis).hit, false, "far behind it");
    const g = new PokeGate(), fire = p => { const h = pokeHit(p, panel, basis); return g.update(h.hit, h.depth, h.inside); };
    eq(fire(at(0, 0, 0.005)), true, "contact");
    eq(fire(at(0, 0, -0.01)), false, "still in: no repeat");
    eq(fire(at(0, 0, 0.03)), false, "pulled back a little: still disarmed");
    eq(fire(at(0, 0, 0.08)), false, "fully back: re-armed");
    eq(fire(at(0, 0, 0.005)), true, "second touch");
  }],
  ["xr: the scripted fake hands drive grab, scale, explode, part info and clap end to end", () => {
    const t = new MockXRTarget(), seen = [];
    const r = new XRGestureRouter(t, { log: m => seen.push(m) });
    const f = new FakeXRHands({
      sphere: () => t.sphere(), exploded: () => t.exploded,
      partCenter: () => (t.exploded ? t.corePos() : null),
    });
    let maxS = 0, wrapped = false;
    for (let i = 0; i < sec(45) && !wrapped; i++) {
      const h = f.step(DT);
      r.update({ left: h.left, right: h.right }, DT);
      maxS = Math.max(maxS, t.s);
      if (f.i === 0 && i > 100) wrapped = true;
    }
    const order = ["grab right", "two-hand", "explode", "part Core", "info Core", "clap"];
    let from = 0;
    for (const want of order) {
      const k = seen.indexOf(want, from);
      if (k < 0) throw new Error("never saw '" + want + "' in order; saw " + seen.join(", "));
      from = k + 1;
    }
    if (!(maxS > 1.5)) throw new Error("the two-hand spread should have scaled it (max " + maxS + ")");
    eq(t.exploded, false, "the clap put it back together");
    eq(wrapped, true, "the script finished a loop");
  }],
);

export function runSelfTests() {
  return TESTS.map(([name, fn]) => {
    try { fn(); return { name, ok: true }; }
    catch (e) { return { name, ok: false, err: String(e && e.message || e) }; }
  });
}
