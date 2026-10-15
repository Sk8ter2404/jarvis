// JARVIS holo XR page (Phase 3): the SAME hologram as /holo - the procedural arc
// reactor and the owner's own models, with named parts - in a Quest 3's browser,
// in passthrough (immersive-ar) or, where AR is not offered, in VR. The headset's
// OWN hand tracking (26 joints per hand, WebXR) replaces the webcam tracker, so
// this page needs no tracker service, no camera and no WebSocket. JARVIS only
// serves the page and the models (/holo/xr, /api/holo/models, /holo/model/<id>).
// It never opens a camera and never moves the Windows cursor or sends OS input.
//
//   /holo/xr                    the entry page: "Enter AR" (or the honest reason it cannot)
//   /holo/xr?model=<name>       start with one of your models instead of the reactor
//   /holo/xr?selftest=1         run the pure-module self tests and show the result
//   /holo/xr?fake=1             DESKTOP preview with scripted fake hands (no headset)
//   /holo/xr?preview=1          desktop preview of the scene, no hands
//   /holo/xr?insecure=1         show the secure-address instructions (to read them)
//
// WebXR only exists on HTTPS or localhost. Over plain http the page does not show
// a dead button: it says what the headset needs (see showInsecure).
//
// Pinned libraries (exact versions; no runtime installs, no floating tags):
//   three.js 0.160.0   https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js
//   its WebXR helpers  .../three@0.160.0/examples/jsm/webxr/XRHandModelFactory.js
//                      .../three@0.160.0/examples/jsm/webxr/XRControllerModelFactory.js
//   controller models  @webxr-input-profiles/assets@1.0.20 (the factory's default is
//                      the floating @1.0; it is pinned below with setPath)
//   plus the STL / 3MF / OBJ loaders models.js imports (same three version).
// holo.html-style import map in xr.html maps "three" to the build above.
import {
  X, XRGestureRouter, placeInFront, panelBasis, pokeHit, PokeGate, dist3, mid3, sub3,
} from "/holo/xrgestures.js";
import { Reactor, REACTOR_NAME } from "/holo/reactor.js";
import { fetchModelList, findModel, loadOwnModel, pageOf, formatSizeMm } from "/holo/models.js";

const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
const HAND_FACTORY_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/webxr/XRHandModelFactory.js";
const CONTROLLER_FACTORY_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/webxr/XRControllerModelFactory.js";
const CONTROLLER_PROFILES_PATH = "https://cdn.jsdelivr.net/npm/@webxr-input-profiles/assets@1.0.20/dist/profiles";

// The session request: AR first (passthrough), hand tracking + a floor-relative
// space as OPTIONAL features so a headset without them still opens the session.
const AR_MODE = "immersive-ar";
const VR_MODE = "immersive-vr";
const SESSION_INIT = { optionalFeatures: ["hand-tracking", "local-floor"] };

const HOLO_UNIT = 0.11;               // metres per reactor unit: the housing is ~13 cm across the radius
const SHELF_PER_PAGE = 3;
const SHELF_W = 0.17, SHELF_H = 0.05, SHELF_GAP = 0.058;

const $ = id => document.getElementById(id);
const qs = new URLSearchParams(location.search);
const FAKE = qs.get("fake") === "1";
const PREVIEW = FAKE || qs.get("preview") === "1";
const TOKEN = (typeof window.JARVIS_TOKEN === "string") ? window.JARVIS_TOKEN : "";
const HINT_HOST = (typeof window.JARVIS_XR_HOST === "string") ? window.JARVIS_XR_HOST : "";
const authHeaders = () => (TOKEN ? { "X-Auth-Token": TOKEN } : {});

function chip(id, text, cls) {
  const c = $(id);
  c.textContent = text;
  c.className = "chip" + (cls ? " " + cls : "");
}
function say(text) { $("msg").textContent = text; }

// ---------------------------------------------------------------------------
// the secure-address instructions (plain http cannot run WebXR)
// ---------------------------------------------------------------------------
// Option A names the owner's LAN name from the HOLO_XR_HINT_HOST setting (baked
// into the page by the server); with no setting it says so generically.
function insecureTexts(host, port) {
  const p = port || "8766";
  const a = "Option A: a Cloudflare-issued certificate for a LAN name " +
    (host ? "(" + host + ")" : "you own") + " - your call.";
  const b = "Option B: Quest developer mode + adb reverse tcp:" + p + " tcp:" + p +
    ", then open http://localhost:" + p + "/holo/xr on the Quest.";
  return { a, b };
}

function showInsecure() {
  const t = insecureTexts(HINT_HOST, location.port);
  $("optA").textContent = t.a;
  $("optB").textContent = t.b;
  $("insecure").style.display = "";
  chip("chip-secure", "NOT A SECURE ADDRESS", "bad");
  chip("chip-xr", "XR: OFF UNTIL SECURE", "bad");
}

function showNoXr(text) {
  const b = $("noxr");
  b.textContent = text;
  b.style.display = "";
  chip("chip-xr", "XR: NOT AVAILABLE", "bad");
}

// ---------------------------------------------------------------------------
// self test page
// ---------------------------------------------------------------------------
async function runSelfTestPage() {
  const { runSelfTests } = await import("/holo/selftest.js");
  const res = runSelfTests();
  const bad = res.filter(r => !r.ok);
  document.title = bad.length ? "HOLO XR SELFTEST FAIL" : "HOLO XR SELFTEST PASS";
  const box = $("msg");
  box.style.whiteSpace = "pre";
  box.style.color = bad.length ? "#ffd2a1" : "#b8ffd5";
  box.textContent = (bad.length ? "FAIL " + bad.length + " of " : "PASS all ") + res.length + "\n" +
    res.map(r => (r.ok ? "ok   " : "FAIL ") + r.name + (r.ok ? "" : " - " + r.err)).join("\n");
  window.__selftest = res;
}

// ---------------------------------------------------------------------------
// the scene (built once three.js is loaded)
// ---------------------------------------------------------------------------
let THREE = null;
let renderer, scene, camera;
let holoRoot, cur, reactor, shownName = REACTOR_NAME, loadToken = 0;
let exploded = false;
let curS = 1;                                      // the hologram's size multiplier as drawn (smoothed toward tgt.s)
let placed = false, placedBasis = null, homePos = null;
let lastHead = null;
let router = null;
let handObjs = [], handFactory = null;
const tgt = { pos: null, quat: null, s: 1 };       // where the hands steer the hologram
let twoSnap = null;
let infoPanel = null, infoPart = null, hintPanel = null;
let halo = null;
const markers = {};
let shelf = null;                                   // {panels, titlePanel, page, list, items}
let modelCache = { t: 0, list: [] };
let flashText = "", flashUntil = 0;
let session = null, sessionStart = 0, noHandsWarned = false;
let fake = null, fakeViz = null;
let last = 0;

function flash(text, ms = 4000) { flashText = text; flashUntil = performance.now() + ms; }

async function modelList(force = false) {
  if (!force && performance.now() - modelCache.t < 4000) return modelCache.list;
  const list = await fetchModelList(authHeaders());
  modelCache = { t: performance.now(), list };
  return list;
}

// -- text panels: canvas -> texture on a plane (DOM is not visible in a headset) --
class TextPanel {
  constructor(w, h, px = 512) {
    this.w = w; this.h = h;
    this.canvas = document.createElement("canvas");
    this.canvas.width = px; this.canvas.height = Math.round(px * h / w);
    this.g = this.canvas.getContext("2d");
    this.tex = new THREE.CanvasTexture(this.canvas);
    this.tex.colorSpace = THREE.SRGBColorSpace;
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
      new THREE.MeshBasicMaterial({ map: this.tex, transparent: true, depthWrite: false }));
    this.mesh.renderOrder = 10;
    this.key = null;
  }
  // Redraw only when `key` changes.
  draw(key, fn) {
    if (key === this.key) return;
    this.key = key;
    const g = this.g, W = this.canvas.width, H = this.canvas.height;
    g.clearRect(0, 0, W, H);
    fn(g, W, H);
    this.tex.needsUpdate = true;
  }
  dispose() { this.tex.dispose(); this.mesh.geometry.dispose(); this.mesh.material.dispose(); }
}

function frameBox(g, W, H, hot) {
  g.fillStyle = hot ? "rgba(57,215,255,.55)" : "rgba(2,12,20,.82)";
  g.fillRect(0, 0, W, H);
  g.lineWidth = 4; g.strokeStyle = hot ? "#fff" : "rgba(57,215,255,.9)";
  g.strokeRect(2, 2, W - 4, H - 4);
}

function wrap(g, text, maxW) {
  const out = []; let line = "";
  for (const w of String(text).split(/\s+/)) {
    const t = line ? line + " " + w : w;
    if (g.measureText(t).width > maxW && line) { out.push(line); line = w; } else line = t;
  }
  if (line) out.push(line);
  return out;
}

function drawLines(g, W, lines, y0, size, color, lh, x = 18) {
  g.font = "600 " + size + "px 'Segoe UI', system-ui, sans-serif";
  g.fillStyle = color; g.textBaseline = "top";
  lines.forEach((l, i) => g.fillText(l, x, y0 + i * lh, W - 2 * x));
}

// -- the hologram --------------------------------------------------------------
// World-space bounding box of a part's MESHES (not its glow sprite or edge lines).
function meshBounds(root) {
  const box = new THREE.Box3();
  root.updateWorldMatrix(true, true);
  root.traverse(o => {
    if (!o.isMesh || !o.geometry) return;
    if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
    box.union(o.geometry.boundingBox.clone().applyMatrix4(o.matrixWorld));
  });
  return box;
}

function measureParts() {
  holoRoot.updateMatrixWorld(true);
  const k = holoRoot.scale.x || 1;
  const sz = new THREE.Vector3();
  for (const p of cur.parts) {
    meshBounds(p.mesh).getSize(sz);
    p._r = Math.max(0.08, 0.5 * sz.length() / k);      // radius in the hologram's own units
  }
}

function activate(obj, name) {
  if (cur) {
    holoRoot.remove(cur.group);
    if (cur !== reactor) cur.dispose();
  }
  cur = obj;
  holoRoot.add(cur.group);
  cur.group.rotation.set(0, 0, 0);
  shownName = name;
  exploded = false;
  cur.setExploded(false); cur.e = 0;
  for (const p of cur.parts) { p.pull = null; p.pullBack = false; p.mesh.position.copy(p.rest); }
  showInfo(null);
  tgt.s = 1; curS = 1; tgt.quat.identity();
  holoRoot.scale.setScalar(HOLO_UNIT);
  measureParts();
  if (shelf) drawShelf(shelf.hot);
}

async function showModel(entry) {
  const tok = ++loadToken;
  if (!entry) { activate(reactor, REACTOR_NAME); flash(REACTOR_NAME.toUpperCase()); return; }
  flash("LOADING " + entry.name.toUpperCase(), 20000);
  try {
    const model = await loadOwnModel(THREE, entry, authHeaders());
    if (tok !== loadToken) { model.dispose(); return; }
    activate(model, entry.name);
    const n = model.parts.length;
    flash(entry.name.toUpperCase() + "  " + formatSizeMm(...model.sizeMm) + "  " +
      (n === 1 ? "1 PART" : n + " PARTS"), 6000);
  } catch (e) {
    if (tok === loadToken) flash("COULD NOT LOAD " + entry.name.toUpperCase() + ": " + (e && e.message || e), 7000);
  }
}

function showInfo(part) {
  infoPart = part;
  if (!part) { infoPanel.mesh.visible = false; return; }
  const sz = meshBounds(part.mesh).getSize(new THREE.Vector3());
  const cm = v => (v * 100 >= 10 ? (v * 100).toFixed(0) : (v * 100).toFixed(1));
  const size = "SHOWN AT " + cm(sz.x) + " x " + cm(sz.y) + " x " + cm(sz.z) + " cm";
  infoPanel.draw(part.name + "|" + size + "|" + part.info, (g, W, H) => {
    frameBox(g, W, H, false);
    drawLines(g, W, [part.name.toUpperCase().slice(0, 28)], 14, 34, "#ffffff", 40);
    drawLines(g, W, [size], 58, 24, "#8fefff", 30);
    g.font = "500 24px 'Segoe UI', system-ui, sans-serif";
    drawLines(g, W, wrap(g, part.info, W - 36).slice(0, 4), 94, 24, "#bfeaf5", 30);
  });
  infoPanel.mesh.visible = true;
}

// What the gesture router steers (see the target contract in xrgestures.js).
const V3 = a => new THREE.Vector3(a[0], a[1], a[2]);
const A3 = v => [v.x, v.y, v.z];
let wpos = null;                                   // scratch vector (made once three.js is loaded)
const target = {
  get exploded() { return exploded; },
  sphere() {
    if (!cur || !placed) return null;
    let r = 1.25;
    for (const p of cur.parts) r = Math.max(r, p.mesh.position.length() + (p._r || 0.2));
    return { c: A3(holoRoot.position), r: r * holoRoot.scale.x };
  },
  position() { return A3(tgt.pos); },
  getScale() { return tgt.s; },
  grabMove(pos) { tgt.pos.set(pos[0], pos[1], pos[2]); },
  grabEnd() {},
  twoBegin() { twoSnap = { pos: tgt.pos.clone(), quat: tgt.quat.clone(), s: tgt.s }; },
  twoMove(m) {
    tgt.pos.set(m.pos[0], m.pos[1], m.pos[2]);
    tgt.s = m.scale;
    tgt.quat.set(m.rot[0], m.rot[1], m.rot[2], m.rot[3]).multiply(twoSnap.quat);
  },
  twoEnd(revert) {
    if (revert && twoSnap) { tgt.pos.copy(twoSnap.pos); tgt.quat.copy(twoSnap.quat); tgt.s = twoSnap.s; }
    twoSnap = null;
  },
  explode() {
    if (cur.parts.length < 2) { flash("ONE PART - NOTHING TO EXPLODE", 3500); return; }
    exploded = true; cur.setExploded(true);
  },
  assemble() {
    exploded = false; cur.setExploded(false);
    for (const p of cur.parts) { p.pull = null; p.pullBack = true; }
    showInfo(null);
  },
  partAt(pt) {
    if (!cur || !placed) return null;
    const v = V3(pt), k = holoRoot.scale.x;
    let best = null, bd = X.PART_MARGIN_M;
    for (const p of cur.parts) {
      p.mesh.getWorldPosition(wpos);
      const d = wpos.distanceTo(v) - (p._r || 0.2) * k;
      if (d < bd) { bd = d; best = p; }
    }
    return best;
  },
  pullPart(part, pt) {
    cur.group.updateWorldMatrix(true, false);
    const local = cur.group.worldToLocal(V3(pt));
    part.pullOff = part.mesh.position.clone().sub(local);
    part.pull = { local: part.mesh.position.clone() };
  },
  movePart(part, pt) {
    if (!part.pull) return;
    cur.group.updateWorldMatrix(true, false);
    part.pull.local.copy(cur.group.worldToLocal(V3(pt))).add(part.pullOff);
  },
  releasePart(part) { part.pull = null; part.pullBack = true; },
  showInfo,
};

// -- hands ---------------------------------------------------------------------
const JOINTS = ["wrist", "thumb-tip", "index-finger-tip", "middle-finger-phalanx-proximal"];

// The three.js hand group -> a plain {joints} snapshot in the reference space
// (the hand groups sit at the scene's origin, so a joint's position IS that).
function snapshot(h) {
  if (!h || !h.visible || !h.joints) return null;
  const joints = {};
  for (const n of JOINTS) {
    const j = h.joints[n];
    if (!j || !j.visible) { if (n === "middle-finger-phalanx-proximal") continue; return null; }
    joints[n] = [j.position.x, j.position.y, j.position.z];
  }
  return { joints };
}

function readHands() {
  if (fake) return fake.hands;
  const out = { left: null, right: null };
  for (const h of handObjs) {
    const side = h.userData.handedness;
    if (side === "left" || side === "right") out[side] = snapshot(h);
  }
  return out;
}

// -- the shelf of models, poked with an index finger ---------------------------
function shelfItems() {
  const items = [];
  const { page, list } = shelf;
  const pg = pageOf(list, page, SHELF_PER_PAGE);
  if (pg.page === 0) items.push({ label: "ARC REACTOR", sub: "procedural", entry: null });
  for (const m of pg.items) items.push({ label: m.name, sub: m.type.toUpperCase(), entry: m });
  if (list.length > SHELF_PER_PAGE) {
    items.push({ label: pg.more ? "MORE" : "FIRST PAGE", sub: "page " + (pg.page + 1) + "/" + pg.pages,
      page: pg.more ? pg.page + 1 : 0 });
  }
  return items;
}

function drawShelf(hotIndex = -1) {
  shelf.items = shelfItems();
  shelf.panels.forEach((p, i) => {
    const it = shelf.items[i];
    p.mesh.visible = !!it && placed;
    if (!it) return;
    const sel = !it.page && ((it.entry === null && shownName === REACTOR_NAME) ||
      (it.entry && it.entry.name === shownName));
    p.draw([it.label, it.sub, sel, i === hotIndex].join("|"), (g, W, H) => {
      frameBox(g, W, H, i === hotIndex);
      drawLines(g, W, [it.label.toUpperCase().slice(0, 22)], 10, 36, sel ? "#7dffb0" : "#ffffff", 40);
      drawLines(g, W, [it.sub], 52, 22, "#8fefff", 26);
    });
  });
}

async function refreshShelf() {
  try { shelf.list = await modelList(true); } catch (_) { shelf.list = []; }
  chip("chip-models", "MODELS: " + shelf.list.length, shelf.list.length ? "ok" : "");
  $("chip-models").style.display = "";
  drawShelf();
}

function placeShelf() {
  const right = [-placedBasis.fwd[2], 0, placedBasis.fwd[0]];
  const base = [holoRoot.position.x + right[0] * -0.38, holoRoot.position.y + 0.06,
    holoRoot.position.z + right[2] * -0.38];
  shelf.centers = shelf.panels.map((_, i) => [base[0], base[1] - i * SHELF_GAP, base[2]]);
  shelf.titleC = [base[0], base[1] + SHELF_GAP, base[2]];
}

function pokeShelf(head, tips) {
  if (!shelf.centers) return;
  let hot = -1;
  shelf.items.forEach((it, i) => {
    const c = shelf.centers[i];
    const basis = panelBasis(c, head);
    const panel = { c, hw: SHELF_W / 2, hh: SHELF_H / 2 };
    for (const tip of tips) {
      const r = pokeHit(tip.p, panel, basis);
      const key = tip.side + i;
      const gate = shelf.gates[key] || (shelf.gates[key] = new PokeGate());
      if (r.hit) hot = i;
      if (gate.update(r.hit && !tip.pinching, r.depth, r.inside)) { choose(it, i); }
    }
  });
  if (hot !== shelf.hot) { shelf.hot = hot; drawShelf(hot); }
}

function choose(it, i) {
  if (it.page !== undefined) { shelf.page = it.page; drawShelf(); return; }
  showModel(it.entry);
}

// -- placing -------------------------------------------------------------------
function place(head) {
  const p = placeInFront(head.pos, head.quat);
  holoRoot.position.set(p[0], p[1], p[2]);
  tgt.pos.copy(holoRoot.position);
  homePos = holoRoot.position.clone();
  const f = [p[0] - head.pos[0], 0, p[2] - head.pos[2]];
  const l = Math.hypot(f[0], f[2]) || 1;
  placedBasis = { fwd: [f[0] / l, 0, f[2] / l] };
  holoRoot.visible = true;
  placed = true;
  placeShelf();
  drawShelf();
}

// -- one frame -----------------------------------------------------------------
function faceHead(mesh, c, head) {
  mesh.position.set(c[0], c[1], c[2]);
  mesh.rotation.set(0, Math.atan2(head.pos[0] - c[0], head.pos[2] - c[2]), 0);
}

function frame(timeMs, xrFrame) {
  const dt = Math.min(0.1, Math.max(0.001, (timeMs - last) / 1000)); last = timeMs;
  const t = timeMs / 1000;

  let head = null;
  if (xrFrame && renderer.xr.isPresenting) {
    const rs = renderer.xr.getReferenceSpace();
    const pose = rs && xrFrame.getViewerPose(rs);
    if (pose) {
      const p = pose.transform.position, o = pose.transform.orientation;
      head = { pos: [p.x, p.y, p.z], quat: [o.x, o.y, o.z, o.w] };
    }
  } else if (PREVIEW) {
    head = { pos: A3(camera.position), quat: [camera.quaternion.x, camera.quaternion.y, camera.quaternion.z, camera.quaternion.w] };
  }
  if (head) lastHead = head;
  if (head && !placed) place(head);

  if (fake && placed) fake.step(dt);
  const hands = placed ? readHands() : { left: null, right: null };
  if (placed) router.update(hands, dt);
  if (placed && session && !noHandsWarned && timeMs - sessionStart > 3000 &&
      !handObjs.some(h => h.userData.hasHand)) {
    noHandsWarned = true;
    flash("NO HAND TRACKING - TURN IT ON IN THE QUEST SETTINGS (MOVEMENT TRACKING)", 12000);
  }

  if (placed) {
    // smooth the hologram toward what the hands steer; idle spin only when untouched
    const k = 1 - Math.exp(-dt * 24);
    holoRoot.position.lerp(tgt.pos, k);
    holoRoot.quaternion.slerp(tgt.quat, k);
    curS += (tgt.s - curS) * k;
    holoRoot.scale.setScalar(HOLO_UNIT * curS);
    if (!router.busy && !exploded) cur.group.rotation.y += 0.25 * dt;
    cur.step(dt, t);
  }

  // feedback: pinch markers, a halo while a hand is within reach
  const sph = placed ? target.sphere() : null;
  let near = false;
  for (const side of ["left", "right"]) {
    const st = router.h[side];
    markers[side].visible = !!(placed && st.pinch && st.pp);
    if (markers[side].visible) markers[side].position.set(st.pp[0], st.pp[1], st.pp[2]);
    if (sph && st.tracked && st.pp && dist3(st.pp, sph.c) <= sph.r + X.GRAB_MARGIN_M) near = true;
  }
  halo.visible = !!(sph && (near || router.busy));
  if (sph) {
    halo.position.set(sph.c[0], sph.c[1], sph.c[2]);
    halo.scale.setScalar(sph.r);
    halo.material.opacity = router.busy ? 0.14 : 0.07;
  }

  if (placed && head) {
    if (infoPart && infoPanel.mesh.visible) {
      infoPart.mesh.getWorldPosition(wpos);
      faceHead(infoPanel.mesh, [wpos.x, wpos.y + 0.10, wpos.z], head);
    }
    // hint panel under the hologram
    const now = performance.now();
    const line1 = now < flashUntil ? flashText : shownName.toUpperCase() + (exploded ? "  -  EXPLODED" : "");
    const line2 = exploded ? "PINCH A PART - HOLD FOR INFO - CLAP: ASSEMBLE"
      : "PINCH: GRAB - 2 HANDS: SCALE, ROTATE - PULL APART FAST: EXPLODE";
    hintPanel.draw(line1 + "|" + line2, (g, W, H) => {
      frameBox(g, W, H, false);
      g.font = "600 30px 'Segoe UI', system-ui, sans-serif";
      drawLines(g, W, wrap(g, line1, W - 36).slice(0, 2), 12, 30, "#ffffff", 34);
      g.font = "500 20px 'Segoe UI', system-ui, sans-serif";
      drawLines(g, W, wrap(g, line2, W - 36).slice(0, 2), 84, 20, "#8fefff", 26);
    });
    faceHead(hintPanel.mesh, [holoRoot.position.x, holoRoot.position.y - (sph ? sph.r : 0.14) - 0.09, holoRoot.position.z], head);

    // the shelf
    shelf.panels.forEach((p, i) => { if (p.mesh.visible) faceHead(p.mesh, shelf.centers[i], head); });
    faceHead(shelf.titlePanel.mesh, shelf.titleC, head);
    const tips = [];
    for (const side of ["left", "right"]) {
      const h = hands[side];
      if (h) tips.push({ side, p: h.joints["index-finger-tip"], pinching: router.h[side].pinch });
    }
    pokeShelf(head, tips);
  }

  if (fakeViz) updateFakeViz(hands);
  renderer.render(scene, camera);
}

// Fake hands have no three.js hand model: draw their tips and palm as dots.
function updateFakeViz(hands) {
  let i = 0;
  for (const side of ["left", "right"]) {
    const h = hands[side];
    for (const n of ["thumb-tip", "index-finger-tip", "wrist"]) {
      const m = fakeViz.children[i++];
      m.visible = !!h;
      if (h) m.position.set(h.joints[n][0], h.joints[n][1], h.joints[n][2]);
    }
  }
}

// ---------------------------------------------------------------------------
// building the scene
// ---------------------------------------------------------------------------
async function buildScene() {
  try {
    THREE = await import(THREE_URL);
  } catch (e) {
    throw new Error("three.js could not be loaded from the CDN (" + THREE_URL + "). The headset needs a connection for it. " + e);
  }
  renderer = new THREE.WebGLRenderer({ canvas: $("gl"), antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);              // transparent: the passthrough shows through in AR
  renderer.xr.enabled = true;
  wpos = new THREE.Vector3();
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, 1, 0.02, 50);
  scene.add(new THREE.AmbientLight(0x6fa8c0, 0.9));
  const key = new THREE.DirectionalLight(0xbfefff, 1.4);
  key.position.set(-0.3, 1.0, 0.9);
  scene.add(key);

  tgt.pos = new THREE.Vector3(); tgt.quat = new THREE.Quaternion();
  holoRoot = new THREE.Group();
  holoRoot.visible = false;
  holoRoot.scale.setScalar(HOLO_UNIT);
  scene.add(holoRoot);
  reactor = new Reactor(THREE);
  reactor.light.visible = false;                      // physical lights: its point light would blow out at this size
  cur = null;
  infoPanel = new TextPanel(0.26, 0.14); infoPanel.mesh.visible = false; scene.add(infoPanel.mesh);
  hintPanel = new TextPanel(0.36, 0.11); scene.add(hintPanel.mesh);
  activate(reactor, REACTOR_NAME);

  halo = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 1),
    new THREE.MeshBasicMaterial({ color: 0x39d7ff, wireframe: true, transparent: true, opacity: 0.12, depthWrite: false }));
  halo.visible = false; scene.add(halo);
  for (const side of ["left", "right"]) {
    markers[side] = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0x8fefff }));
    markers[side].visible = false; scene.add(markers[side]);
  }

  shelf = { panels: [], titlePanel: new TextPanel(SHELF_W, SHELF_H * 0.7), page: 0, list: [], items: [],
    centers: null, titleC: null, gates: {}, hot: -1 };
  shelf.titlePanel.draw("title", (g, W, H) => {
    g.fillStyle = "rgba(2,12,20,.6)"; g.fillRect(0, 0, W, H);
    drawLines(g, W, ["MODELS  -  POKE ONE"], 8, 28, "#8fefff", 30);
  });
  scene.add(shelf.titlePanel.mesh);
  for (let i = 0; i < SHELF_PER_PAGE + 2; i++) {
    const p = new TextPanel(SHELF_W, SHELF_H);
    p.mesh.visible = false; scene.add(p.mesh); shelf.panels.push(p);
  }

  router = new XRGestureRouter(target, { log: m => { void m; } });

  // hands (the groups are the headset's; the models are only for you to see them)
  for (let i = 0; i < 2; i++) {
    const h = renderer.xr.getHand(i);
    h.userData.handedness = ""; h.userData.hasHand = false;
    h.addEventListener("connected", e => {
      h.userData.handedness = e.data.handedness; h.userData.hasHand = !!e.data.hand;
    });
    h.addEventListener("disconnected", () => { h.userData.handedness = ""; h.userData.hasHand = false; });
    scene.add(h);
    handObjs.push(h);
  }
  try {
    const { XRHandModelFactory } = await import(HAND_FACTORY_URL);
    handFactory = new XRHandModelFactory();
    for (const h of handObjs) h.add(handFactory.createHandModel(h, "spheres"));
  } catch (e) { console.warn("hand models unavailable", e); }
  try {
    const { XRControllerModelFactory } = await import(CONTROLLER_FACTORY_URL);
    const cf = new XRControllerModelFactory();
    cf.path = CONTROLLER_PROFILES_PATH;               // pin the profile assets' version
    for (let i = 0; i < 2; i++) {
      const grip = renderer.xr.getControllerGrip(i);
      grip.add(cf.createControllerModel(grip));
      scene.add(grip);
    }
  } catch (e) { console.warn("controller models unavailable", e); }

  function onResize() {
    if (renderer.xr.isPresenting) return;
    renderer.setSize(innerWidth, innerHeight, false);
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  }
  addEventListener("resize", onResize);
  onResize();

  renderer.xr.addEventListener("sessionstart", () => {
    const s = renderer.xr.getSession();
    session = s; sessionStart = performance.now(); noHandsWarned = false;
    placed = false; holoRoot.visible = false;
    scene.background = s.environmentBlendMode === "opaque" ? new THREE.Color(0x02070d) : null;
    if (s.environmentBlendMode === "opaque" && !scene.getObjectByName("floor")) {
      const grid = new THREE.GridHelper(8, 32, 0x39d7ff, 0x0b3a4a);
      grid.name = "floor"; scene.add(grid);
    }
    s.addEventListener("end", onSessionEnd);
  });
  renderer.setAnimationLoop(frame);
}

function onSessionEnd() {
  session = null;
  placed = false; holoRoot.visible = false;
  router.reset();
  for (const p of [infoPanel, hintPanel, ...shelf.panels, shelf.titlePanel]) p.mesh.visible = false;
  const b = $("enter"); b.disabled = false;
  say("Session ended. Press the button to go back in.");
}

// ---------------------------------------------------------------------------
// the desktop preview (no headset): a plain camera, optionally scripted hands
// ---------------------------------------------------------------------------
async function startPreview() {
  document.body.classList.add("preview");
  camera.fov = 75;
  camera.position.set(0, 1.42, 0.55);
  camera.lookAt(-0.04, 1.1, -0.2);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  const grid = new THREE.GridHelper(4, 16, 0x39d7ff, 0x0b3a4a);
  grid.position.y = 0.9; scene.add(grid);
  if (FAKE) {
    document.body.classList.add("fake");
    const { FakeXRHands } = await import("/holo/xrfake.js");
    const f = new FakeXRHands({
      sphere: () => target.sphere(),
      exploded: () => exploded,
      partCenter: name => {
        const p = cur.partByName(name);
        if (!p || !exploded) return null;
        p.mesh.getWorldPosition(wpos);
        return A3(wpos);
      },
    });
    fake = { hands: { left: null, right: null }, phase: "",
      step(dt) {
        const o = f.step(dt);
        if (o.phase === "idle" && this.phase !== "idle" && homePos) {   // a new loop: back to the start so it does not compound
          tgt.pos.copy(homePos); tgt.quat.identity(); tgt.s = 1;
        }
        this.hands = { left: o.left, right: o.right }; this.phase = o.phase;
      } };
    fakeViz = new THREE.Group();
    for (let i = 0; i < 6; i++) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(i % 3 === 2 ? 0.02 : 0.01, 10, 10),
        new THREE.MeshBasicMaterial({ color: i % 3 === 2 ? 0x2a6a80 : 0xffffff }));
      fakeViz.add(m);
    }
    scene.add(fakeViz);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
async function enterSession(mode) {
  let s;
  try {
    s = await navigator.xr.requestSession(mode, SESSION_INIT);
  } catch (e) {
    say("The headset refused the session (" + (e && e.message || e) + ").");
    $("enter").disabled = false;
    return;
  }
  try {
    renderer.xr.setReferenceSpaceType("local-floor");
    await renderer.xr.setSession(s);
  } catch (e) {
    try {                                             // a headset without a floor-relative space
      renderer.xr.setReferenceSpaceType("local");
      await renderer.xr.setSession(s);
    } catch (e2) {
      say("The session could not start (" + (e2 && e2.message || e2) + ").");
      try { await s.end(); } catch (_) { /* already gone */ }
      $("enter").disabled = false;
    }
  }
}

async function main() {
  const secure = window.isSecureContext === true && qs.get("insecure") !== "1";
  if (secure) chip("chip-secure", "SECURE ADDRESS", "ok"); else showInsecure();
  if (PREVIEW) chip("chip-xr", "XR: DESKTOP PREVIEW", "");

  let mode = null;
  if (secure && !PREVIEW) {
    if (!navigator.xr) {
      showNoXr("This browser has no WebXR. On the Quest, open this page in the Meta Quest Browser.");
    } else {
      for (const m of [AR_MODE, VR_MODE]) {
        try { if (await navigator.xr.isSessionSupported(m)) { mode = m; break; } } catch (_) { /* try the next */ }
      }
      if (!mode) showNoXr("No headset session is available here (neither passthrough AR nor VR). " +
        "Open this page in the Meta Quest Browser with the headset on.");
    }
  }
  if (!mode && !PREVIEW) return;                      // the page already said why

  try {
    await buildScene();
  } catch (e) {
    say(String(e && e.message || e));
    return;
  }
  if (PREVIEW) await startPreview();
  refreshShelf();
  const want = qs.get("model");
  if (want) {
    try { const e = findModel(await modelList(true), want); if (e) showModel(e); else flash("NO MODEL CALLED " + want.toUpperCase()); }
    catch (_) { /* the shelf shows the same problem */ }
  }

  if (mode) {
    chip("chip-xr", mode === AR_MODE ? "XR: AR (PASSTHROUGH)" : "XR: VR ONLY", "ok");
    const b = $("enter");
    b.textContent = mode === AR_MODE ? "ENTER AR" : "ENTER VR";
    b.style.display = "inline-block";
    b.onclick = () => { b.disabled = true; say("Starting..."); enterSession(mode); };
    say(mode === AR_MODE ? "Put the headset on, then press ENTER AR." :
      "Passthrough AR isn't offered here, so this opens in VR. Press ENTER VR.");
  }
}

if (qs.get("selftest") === "1") {
  await runSelfTestPage();
} else {
  await main();
}

window.__holoxr = {                                   // debugging handle (page-local)
  get router() { return router; }, target, tgt, get cur() { return cur; },
  get placed() { return placed; }, get fake() { return fake; }, insecureTexts, get shelf() { return shelf; },
  get shownName() { return shownName; },
  get exploded() { return exploded; }, get THREE() { return THREE; }, mid3, sub3,
};
