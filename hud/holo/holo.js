// JARVIS holo mode page. Renders a procedural arc reactor with three.js and
// drives it from HAND DATA the tracker service sends over a WebSocket
// (ws://127.0.0.1:8793). It never opens the camera, runs no hand model, and
// cannot move the Windows cursor or send OS input: it only draws.
//
//   /holo?mode=desktop   solid dark HUD, no camera picture (default)
//   /holo?mode=camera    the tracker's mirrored camera frames behind the scene
//   /holo?fake=1         built-in scripted hands + gestures (no service needed)
//   /holo?selftest=1     run the pure-module self tests and show the result
//   other params: ws=<url> debug=1 mirror=1 (flip the frames again; the
//   service already sends them mirrored, matching the mirrored landmarks)
//
// Pinned libraries (exact versions; no runtime installs, no floating tags):
//   three.js 0.160.0  https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js
import {
  T, GestureRouter, mapToScreen, mapFromScreen, dampSpin, followStep, palmSize,
} from "/holo/gestures.js";
import { Reactor, REACTOR_NAME } from "/holo/reactor.js";

const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
const TRACKER_URL_DEFAULT = "ws://127.0.0.1:8793";

const $ = id => document.getElementById(id);
const qs = new URLSearchParams(location.search);
const FAKE = qs.get("fake") === "1";
const TOKEN = (typeof window.JARVIS_TOKEN === "string") ? window.JARVIS_TOKEN : "";

function showError(msg) {
  const e = $("err"); e.textContent = msg; e.style.display = "block";
}

// ---------------------------------------------------------------------------
// self test page
// ---------------------------------------------------------------------------
if (qs.get("selftest") === "1") {
  const { runSelfTests } = await import("/holo/selftest.js");
  const res = runSelfTests();
  const bad = res.filter(r => !r.ok);
  document.title = bad.length ? "HOLO SELFTEST FAIL" : "HOLO SELFTEST PASS";
  const box = $("err");
  box.style.display = "block";
  box.style.borderColor = bad.length ? "#c27b2e" : "#3fbf7a";
  box.style.color = bad.length ? "#ffd2a1" : "#b8ffd5";
  box.style.whiteSpace = "pre";
  box.textContent = (bad.length ? "FAIL " + bad.length + " of " : "PASS all ") + res.length + "\n" +
    res.map(r => (r.ok ? "ok   " : "FAIL ") + r.name + (r.ok ? "" : " - " + r.err)).join("\n");
  window.__selftest = res;
  throw new Error("selftest done");        // stop: do not boot the scene
}

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
let mode = (qs.get("mode") === "camera") ? "camera" : "desktop";
let mirrorFrame = qs.get("mirror") === "1";
let bw = false, showLabels = true, debug = qs.get("debug") === "1";
let W = innerWidth, H = innerHeight;
let aspect = 4 / 3;
let latestHands = [];
let lastGesture = "";
let trackerOk = false, lastHandsAt = 0;
let closed = false;
let ws = null, wsTimer = 0;
let frameBitmap = null, decoding = false;
let seenSeq = null;                    // first poll just syncs; never replays old commands

const URLS = { tracker: qs.get("ws") || TRACKER_URL_DEFAULT };

// ---------------------------------------------------------------------------
// three.js
// ---------------------------------------------------------------------------
let THREE;
try {
  THREE = await import(THREE_URL);
} catch (e) {
  showError("three.js could not be loaded from the CDN (" + THREE_URL + "). Holo mode needs a network connection for it. " + e);
  throw e;
}

const glCanvas = $("gl");
const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
renderer.setClearColor(0x000000, 0);
const sceneGL = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 10, 6000);
const CAM_Z = 1300;
sceneGL.add(new THREE.AmbientLight(0x6fa8c0, 0.9));
const key = new THREE.DirectionalLight(0xbfefff, 1.4);
key.position.set(-300, 500, 900);
sceneGL.add(key);

const reactor = new Reactor(THREE);
reactor.group.visible = false;
sceneGL.add(reactor.group);

function resize() {
  W = innerWidth; H = innerHeight;
  renderer.setSize(W, H, false);
  camera.aspect = W / H;
  camera.fov = 2 * Math.atan(H / 2 / CAM_Z) * 180 / Math.PI;   // z=0 plane: 1 unit = 1 px
  camera.position.set(0, 0, CAM_Z);
  camera.updateProjectionMatrix();
  for (const id of ["bg", "fx"]) { $(id).width = W; $(id).height = H; }
}
addEventListener("resize", resize);
resize();

// ---------------------------------------------------------------------------
// reactor transform + the "scene" the gesture router drives
// ---------------------------------------------------------------------------
const rs = {
  x: 0, y: 0, tx: 0, ty: 0,            // px (current, target)
  scale: 1,
  wy: 0, wx: 0,                         // angular velocity
  ry: 0.45, rx: -0.38,                  // orientation
};
const baseR = () => Math.min(W, H) * 0.17;
const toWorld = (px, py, z = 0) => new THREE.Vector3(px - W / 2, H / 2 - py, z);
const labelEls = new Map();
let infoEl = null;
let menuPos = { x: 0, y: 0 }, menuTimer = 0;

function toScreen(nx, ny) { return mapToScreen(nx, ny, W, H, mode, aspect); }

function projectPart(part) {
  reactor.group.updateMatrixWorld(true);
  const v = new THREE.Vector3();
  part.mesh.getWorldPosition(v);
  const wz = v.z;
  v.project(camera);
  return { x: (v.x * 0.5 + 0.5) * W, y: (-v.y * 0.5 + 0.5) * H, wz };
}

const scene = {
  reactorVisible: false,
  exploded: false,
  menuOpen: false,
  get scale() { return rs.scale; },
  reactorScreen() { return { x: rs.x, y: rs.y, r: baseR() * rs.scale }; },
  toScreen,
  setReactorTarget(x, y) { rs.tx = x; rs.ty = y; },
  setScale(s) { rs.scale = s; },
  addSpin(wy, wx) { rs.wy += wy; rs.wx += wx; },

  openMenu(x, y) {
    menuPos = { x: Math.max(160, Math.min(W - 160, x)), y: Math.max(160, Math.min(H - 160, y)) };
    const m = $("menu");
    m.style.left = menuPos.x + "px"; m.style.top = menuPos.y + "px";
    m.style.display = "block";
    scene.menuOpen = true;
    clearTimeout(menuTimer);
    menuTimer = setTimeout(() => scene.closeMenu(), 9000);
    setHint("POINT AT AN ITEM AND PINCH");
  },
  closeMenu() {
    $("menu").style.display = "none";
    scene.menuOpen = false;
    clearTimeout(menuTimer);
    setHint("");
  },
  menuItems() {
    return [...document.querySelectorAll("#menu button")].map(b => {
      const r = b.getBoundingClientRect();
      return { el: b, model: b.dataset.model, disabled: b.disabled, x: r.left + r.width / 2, y: r.top + r.height / 2 };
    });
  },
  menuItemAt(x, y) {
    let best = null, bd = 70;
    for (const it of scene.menuItems()) {
      const d = Math.hypot(it.x - x, it.y - y);
      if (d < bd) { bd = d; best = it; }
    }
    return best;
  },
  chooseMenuItem(item) {
    if (item.disabled) { setHint("COMING SOON"); return; }
    scene.closeMenu();
    scene.spawn(item.model, menuPos.x, menuPos.y);
  },
  spawn(model, x, y) {
    if (String(model).toLowerCase() !== REACTOR_NAME) { setHint("UNKNOWN MODEL"); return false; }
    const c = (x == null) ? { x: W / 2, y: H / 2 } : { x, y };
    rs.x = rs.tx = c.x; rs.y = rs.ty = c.y;
    rs.scale = 1; rs.wy = rs.wx = 0;
    scene.exploded = false; reactor.setExploded(false); reactor.e = 0;
    for (const p of reactor.parts) { p.pull = null; p.mesh.position.copy(p.rest); }
    scene.reactorVisible = true;
    reactor.group.visible = true;
    setHint("");
    return true;
  },
  explode() { scene.exploded = true; reactor.setExploded(true); },
  assemble() { scene.exploded = false; reactor.setExploded(false); scene.showInfo(null); },

  partAt(x, y) {
    if (!scene.reactorVisible) return null;
    let best = null, bd = T.PART_PICK_PX * Math.max(1, rs.scale * 0.8);
    for (const p of reactor.parts) {
      const s = projectPart(p);
      const d = Math.hypot(s.x - x, s.y - y);
      if (d < bd) { bd = d; best = p; }
    }
    return best;
  },
  pullPart(part) {
    const s = projectPart(part);
    part.pullZ = s.wz;
    part.pull = { local: part.mesh.position.clone() };
  },
  movePart(part, x, y) {
    if (!part.pull) return;
    reactor.group.updateMatrixWorld(true);
    part.pull.local = reactor.group.worldToLocal(toWorld(x, y, (part.pullZ || 0) + 120));
  },
  releasePart(part) { part.pull = null; part.pullBack = true; },
  showInfo(part) {
    if (infoEl) { infoEl.remove(); infoEl = null; }
    if (!part) return;
    infoEl = document.createElement("div");
    infoEl.className = "lbl info";
    infoEl.innerHTML = "<b></b><span></span>";
    infoEl.firstChild.textContent = part.name.toUpperCase();
    infoEl.lastChild.textContent = part.info;
    infoEl.dataset.part = part.name;
    $("labels").appendChild(infoEl);
  },
};

const router = new GestureRouter(scene, { log: m => { lastGesture = m; } });

// ---------------------------------------------------------------------------
// menu DOM (mouse works too)
// ---------------------------------------------------------------------------
(function buildMenu() {
  const m = $("menu");
  const items = [
    { label: "ARC REACTOR", sub: "procedural", model: REACTOR_NAME, ang: -90, off: false },
    { label: "MY MODELS", sub: "coming soon", model: "my models", ang: 30, off: true },
  ];
  for (const it of items) {
    const b = document.createElement("button");
    b.dataset.model = it.model;
    b.disabled = it.off;
    b.innerHTML = it.label + "<small>" + it.sub + "</small>";
    const a = it.ang * Math.PI / 180;
    b.style.left = Math.cos(a) * 150 + "px";
    b.style.top = Math.sin(a) * 150 + "px";
    b.addEventListener("click", () => scene.chooseMenuItem(scene.menuItems().find(x => x.el === b)));
    m.appendChild(b);
  }
})();

function setHint(t) { $("hint").textContent = t; }

// ---------------------------------------------------------------------------
// camera frames + background
// ---------------------------------------------------------------------------
function setMode(m) {
  if (m !== "desktop" && m !== "camera") return;
  mode = m;
  document.body.classList.toggle("desktop", m === "desktop");
  $("chip-mode").textContent = m === "desktop" ? "DESKTOP" : "CAMERA";
  subscribeFrames();
  updateLegend();
}

function subscribeFrames() {
  if (ws && ws.readyState === 1) {
    try { ws.send(JSON.stringify({ type: "subscribe", frames: mode === "camera" })); } catch (_) { /* socket closing */ }
  }
}

function drawBackground() {
  if (mode !== "camera") return;
  const g = $("bg").getContext("2d");
  g.fillStyle = "#02070d"; g.fillRect(0, 0, W, H);
  if (FAKE) {                                        // no frames in fake mode: a faint grid stands in
    g.strokeStyle = "rgba(57,215,255,.10)";
    for (let x = 0; x < W; x += 64) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
    for (let y = 0; y < H; y += 64) { g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }
    return;
  }
  if (!frameBitmap) return;
  const a = frameBitmap.width / frameBitmap.height;
  aspect = a;
  // same cover fit as mapToScreen, so landmarks line up with the picture
  const p0 = mapToScreen(0, 0, W, H, "camera", a), p1 = mapToScreen(1, 1, W, H, "camera", a);
  g.save();
  if (mirrorFrame) { g.translate(W, 0); g.scale(-1, 1); }
  const x0 = mirrorFrame ? W - p1.x : p0.x;
  g.drawImage(frameBitmap, x0, p0.y, p1.x - p0.x, p1.y - p0.y);
  g.restore();
}

// ---------------------------------------------------------------------------
// tracker connection (or the fake source)
// ---------------------------------------------------------------------------
function onTrackerMessage(msg) {
  if (msg.type === "hands") {
    latestHands = msg.hands || [];
    const now = performance.now();
    // The router runs on the message clock, not the animation clock: a window
    // that is covered or minimized stops requestAnimationFrame, and gestures
    // must not stall with it.
    const dt = lastHandsAt ? Math.min(0.1, (now - lastHandsAt) / 1000) : 0.033;
    lastHandsAt = now;
    if (msg.w && msg.h) aspect = msg.w / msg.h;
    router.update(latestHands, dt);
  } else if (msg.type === "gesture") {
    lastGesture = msg.name + (msg.value != null ? " " + Number(msg.value).toFixed(2) : "");
    router.onGesture(msg);
  }
}
function setTracker(ok) {
  trackerOk = ok;
  const c = $("chip-trk");
  c.textContent = ok ? "TRACKER: CONNECTED" : "TRACKER: NOT RUNNING";
  c.className = "chip " + (ok ? "ok" : "bad");
}

function connectTracker() {
  if (closed || FAKE) return;
  try {
    ws = new WebSocket(URLS.tracker);
  } catch (e) {
    setTracker(false); wsTimer = setTimeout(connectTracker, 2500); return;
  }
  ws.binaryType = "blob";
  ws.onopen = () => { setTracker(true); subscribeFrames(); };
  ws.onclose = () => { setTracker(false); latestHands = []; if (!closed) wsTimer = setTimeout(connectTracker, 2500); };
  ws.onerror = () => { try { ws.close(); } catch (_) { /* already closed */ } };
  ws.onmessage = ev => {
    if (typeof ev.data === "string") {
      let msg; try { msg = JSON.parse(ev.data); } catch (_) { return; }
      onTrackerMessage(msg);
    } else if (mode === "camera" && !decoding) {
      decoding = true;
      createImageBitmap(ev.data).then(bm => { frameBitmap = bm; }).catch(() => {}).finally(() => { decoding = false; });
    }
  };
}

let fake = null, fakeT0 = 0;
async function startFake() {
  const { FakeHands } = await import("/holo/fakehands.js");
  fake = new FakeHands({
    menuItemNorm: i => {
      const it = scene.menuOpen ? scene.menuItems()[i] : null;
      return it ? mapFromScreen(it.x, it.y, W, H, mode, aspect) : null;
    },
    partNorm: name => {
      const p = reactor.partByName(name);
      if (!p || !scene.reactorVisible) return null;
      const s = projectPart(p);
      return mapFromScreen(s.x, s.y, W, H, mode, aspect);
    },
    reactorNorm: () => scene.reactorVisible ? mapFromScreen(rs.x, rs.y, W, H, mode, aspect) : null,
  });
  fakeT0 = performance.now();
  $("chip-fake").style.display = "";
  setTracker(true);
  $("chip-trk").textContent = "TRACKER: FAKE";
  setInterval(() => {
    for (const m of fake.step((performance.now() - fakeT0) / 1000)) onTrackerMessage(m);
  }, 33);
}

// ---------------------------------------------------------------------------
// JARVIS commands (voice actions) - polled
// ---------------------------------------------------------------------------
function applyCommand(c) {
  switch (c.cmd) {
    case "show": if (!scene.spawn(c.arg || REACTOR_NAME)) { /* unknown model: the action already said so */ } break;
    case "explode": if (scene.reactorVisible) scene.explode(); break;
    case "assemble": if (scene.reactorVisible) scene.assemble(); break;
    case "mode": setMode(c.arg); break;
    case "close": closeHolo(); break;
  }
}

async function pollState() {
  if (closed) return;
  try {
    const q = new URLSearchParams({
      seen: String(seenSeq == null ? 0 : seenSeq),
      shown: scene.reactorVisible ? REACTOR_NAME : "",
      exploded: scene.exploded ? "1" : "0",
    });
    const headers = TOKEN ? { "X-Auth-Token": TOKEN } : {};
    const r = await fetch("/api/holo/state?" + q, { headers, cache: "no-store" });
    if (!r.ok) return;
    const d = await r.json();
    if (seenSeq == null) {
      // first contact: adopt, never replay. A model already requested spawns.
      seenSeq = d.seq;
      if (!qs.get("mode") && (d.mode === "camera" || d.mode === "desktop")) setMode(d.mode);
      if (d.model) scene.spawn(d.model);
      return;
    }
    for (const c of d.commands || []) {
      if (c.seq > seenSeq) { seenSeq = c.seq; applyCommand(c); }
    }
  } catch (_) { /* server restarting; try again next tick */ }
}

function closeHolo() {
  if (closed) return;
  closed = true;
  clearTimeout(wsTimer);
  try { if (ws) ws.close(); } catch (_) { /* already closed */ }
  $("closed").style.display = "flex";
  try { window.close(); } catch (_) { /* a plain tab cannot close itself; the overlay says closed */ }
}

// ---------------------------------------------------------------------------
// drawing: finger rings, hand bones, labels, debug
// ---------------------------------------------------------------------------
const TIPS = [4, 8, 12, 16, 20];
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

function ring(g, x, y, r, alpha = 1, fill = false) {
  g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2);
  if (fill) { g.fillStyle = "rgba(143,239,255," + (0.35 * alpha) + ")"; g.fill(); }
  g.strokeStyle = "rgba(90,230,255," + alpha + ")"; g.stroke();
}

function drawHands() {
  const g = $("fx").getContext("2d");
  g.clearRect(0, 0, W, H);
  g.lineWidth = 2.5; g.shadowColor = "#39d7ff"; g.shadowBlur = 16;
  for (const h of latestHands) {
    const P = h.lm.map(p => toScreen(p[0], p[1]));
    const sz = Math.max(24, palmSize(h.lm) * Math.hypot(W, H) * (mode === "camera" ? 0.55 : 0.5));
    g.lineWidth = 1.2;
    g.strokeStyle = "rgba(57,215,255,.35)";
    for (const [a, b] of BONES) { g.beginPath(); g.moveTo(P[a].x, P[a].y); g.lineTo(P[b].x, P[b].y); g.stroke(); }
    g.lineWidth = 2.5;
    for (const i of TIPS) ring(g, P[i].x, P[i].y, sz * 0.17, 0.95, h.state === "pinch" && (i === 4 || i === 8));
    const pc = toScreen(h.palm[0], h.palm[1]);
    ring(g, pc.x, pc.y, sz * 0.5, 0.9, h.state === "fist");
    ring(g, pc.x, pc.y, sz * 0.34, 0.55);
    if (h.state === "pinch") {
      const pp = { x: (P[4].x + P[8].x) / 2, y: (P[4].y + P[8].y) / 2 };
      ring(g, pp.x, pp.y, sz * 0.26, 1, true);
    }
    if (debug) {
      g.shadowBlur = 0; g.fillStyle = "#fff";
      P.forEach((p, i) => { g.fillRect(p.x - 2, p.y - 2, 4, 4); if (i % 4 === 0) g.fillText(String(i), p.x + 5, p.y - 5); });
      g.fillStyle = "#8fefff"; g.font = "12px Consolas, monospace";
      g.fillText(h.side + " " + h.state + " pinch=" + Number(h.pinch).toFixed(2), pc.x - 40, pc.y + sz * 0.6 + 12);
      g.shadowBlur = 16;
    }
  }
  g.shadowBlur = 0;
  // aim reticle on the index tip while the menu is open
  if (scene.menuOpen) {
    for (const h of latestHands) {
      const t = toScreen(h.lm[8][0], h.lm[8][1]);
      g.strokeStyle = "rgba(255,255,255,.9)"; g.lineWidth = 1.5;
      g.beginPath(); g.arc(t.x, t.y, 9, 0, Math.PI * 2); g.stroke();
      const hot = scene.menuItemAt(t.x, t.y);
      for (const it of scene.menuItems()) it.el.classList.toggle("hot", hot && hot.el === it.el);
    }
  } else {
    for (const b of document.querySelectorAll("#menu button.hot")) b.classList.remove("hot");
  }
}

function updateLabels() {
  const want = new Set();
  if (scene.reactorVisible && showLabels && reactor.e > 0.5) {
    for (const p of reactor.parts) {
      let el = labelEls.get(p.name);
      if (!el) {
        el = document.createElement("div"); el.className = "lbl";
        el.textContent = p.name; $("labels").appendChild(el); labelEls.set(p.name, el);
      }
      if (infoEl && infoEl.dataset.part === p.name) continue;   // its info card replaces the tag
      const s = projectPart(p);
      el.style.left = s.x + "px"; el.style.top = s.y + "px"; el.style.display = "";
      want.add(p.name);
    }
  }
  for (const [n, el] of labelEls) if (!want.has(n)) el.style.display = "none";
  if (infoEl) {
    const p = reactor.partByName(infoEl.dataset.part);
    if (p) { const s = projectPart(p); infoEl.style.left = s.x + 18 + "px"; infoEl.style.top = s.y + "px"; }
  }
}

function updateLegend() {
  $("legend").innerHTML = [
    "<b>PINCH &amp; HOLD</b> spawn / menu &nbsp; <b>FIST</b> move &nbsp; <b>POINT &amp; FLICK</b> spin",
    "<b>2 FISTS APART</b> / <b>PALMS TOGETHER THEN APART</b> / <b>TWIST A FIST</b> explode",
    "exploded: <b>PINCH</b> a part to pull it out, hold for info &nbsp; <b>CLAP</b> assemble &nbsp; <b>2 PINCHES</b> scale",
    "keys: <b>E</b> explode &nbsp;<b>C</b> camera/desktop &nbsp;<b>B</b> b&amp;w &nbsp;<b>F</b> mirror &nbsp;<b>L</b> labels &nbsp;<b>D</b> debug &nbsp;<b>R</b> reactor",
  ].join("<br>");
}

// ---------------------------------------------------------------------------
// keys
// ---------------------------------------------------------------------------
addEventListener("keydown", e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  switch (e.key.toLowerCase()) {
    case "e": if (scene.reactorVisible) (scene.exploded ? scene.assemble() : scene.explode()); break;
    case "c": setMode(mode === "camera" ? "desktop" : "camera"); break;
    case "b": bw = !bw; $("bg").classList.toggle("bw", bw); break;
    case "f": mirrorFrame = !mirrorFrame; break;
    case "l": showLabels = !showLabels; break;
    case "d": debug = !debug; break;
    case "r": scene.spawn(REACTOR_NAME); break;
    case "m": scene.openMenu(W / 2, H / 2); break;
    case "escape": scene.closeMenu(); break;
  }
});

// ---------------------------------------------------------------------------
// main loop
// ---------------------------------------------------------------------------
let last = performance.now();
function frame(now) {
  if (closed) return;
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  const t = now / 1000;

  // hands stale (tracker stopped): treat as none
  if (latestHands.length && now - lastHandsAt > 600) { latestHands = []; router.update([], dt); }

  if (scene.reactorVisible) {
    rs.x = followStep(rs.x, rs.tx, dt, T.FOLLOW_TAU_S);
    rs.y = followStep(rs.y, rs.ty, dt, T.FOLLOW_TAU_S);
    rs.wy = dampSpin(rs.wy, dt); rs.wx = dampSpin(rs.wx, dt);
    const idle = (Math.abs(rs.wy) < 0.4) ? 0.25 : 0;
    rs.ry += (rs.wy + idle) * dt;
    rs.rx += rs.wx * dt;
    const s = baseR() * rs.scale;
    const g = reactor.group;
    g.position.copy(toWorld(rs.x, rs.y, 0));
    g.scale.setScalar(s);
    g.rotation.set(rs.rx + Math.sin(t * 0.7) * 0.03, rs.ry, 0);
    reactor.step(dt, t);
  }
  drawBackground();
  renderer.render(sceneGL, camera);
  drawHands();
  updateLabels();
  if (debug) {
    const sm = router.summary();
    $("dbg").textContent = "mode " + mode + "  hands " + latestHands.length + "  last: " + lastGesture +
      "\ngrab " + sm.grab + "  pulled " + sm.pulled + "  twoHand " + sm.twoHand +
      "\n" + sm.hands.join(" ");
  } else $("dbg").textContent = "";
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------
// go
// ---------------------------------------------------------------------------
setMode(mode);
updateLegend();
setHint(mode === "desktop" ? "HOLD YOUR PALM OPEN, OR PINCH & HOLD, TO OPEN THE MENU" : "PINCH & HOLD TO SPAWN");
connectTracker();
if (FAKE) startFake();
setInterval(pollState, 300);
pollState();
requestAnimationFrame(frame);
window.__holo = { scene, router, reactor, rs };   // debugging handle (page-local)
