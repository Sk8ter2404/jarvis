// Holo mode - the owner's OWN 3D models (STL / 3MF / OBJ). The server lists them
// (GET /api/holo/models) and serves one at a time (GET /holo/model/<id>); this
// module fetches, parses (three.js loaders pinned to the same three@0.160.0 as
// the page), centres and scales each one to the reactor's size, and turns its
// objects into NAMED PARTS with the same shape the arc reactor's parts have, so
// explode / pinch-a-part / hold-for-info / clap-to-assemble work unchanged.
//
// The loaders import "three" (bare) and 3MFLoader also imports its own bundled
// fflate with a relative path; holo.html's import map points "three" at the one
// pinned build, so there is exactly one THREE in the page.
//
// The first block is PURE (no DOM, no three.js) so selftest.js can run it.
// NEVER touches the camera or the OS; it only fetches files from this server.
import { stepParts } from "/holo/reactor.js";

export const LOADER_URLS = Object.freeze({
  stl: "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/STLLoader.js",
  "3mf": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/3MFLoader.js",
  obj: "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/OBJLoader.js",
});

export const TARGET_RADIUS = 1.15;   // reactor units: the reactor's housing is ~1.16
export const MAX_PARTS = 40;         // more objects than this stay one piece
export const EDGE_TRI_LIMIT = 150000;
export const MODELS_PER_PAGE = 6;

// ---------------------------------------------------------------------------
// pure helpers
// ---------------------------------------------------------------------------

// "24.0 x 18.5 x 3.2 mm" from three extents in millimetres.
export function formatSizeMm(sx, sy, sz) {
  const f = v => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));
  return f(sx) + " x " + f(sy) + " x " + f(sz) + " mm";
}

// Names for parts: blank -> "Part 01", repeats -> "Name 2", "Name 3".
export function uniqueNames(names) {
  const seen = new Map();
  return names.map((n, i) => {
    let base = String(n || "").trim() || "Part " + String(i + 1).padStart(2, "0");
    const c = (seen.get(base) || 0) + 1;
    seen.set(base, c);
    return c === 1 ? base : base + " " + c;
  });
}

// Which scene-graph nodes are the model's parts. Duck-typed ({children, isMesh}).
//   3MF : each build item / component is a part; several meshes inside ONE
//         object (e.g. one per material) stay a single part.
//   OBJ : each mesh (o / g group) is a part (meshesAreParts).
//   STL : one mesh -> one part.
// Wrapper groups with a single child are looked through.
export function splitParts(root, opts = {}) {
  const asParts = !!opts.meshesAreParts;
  let node = root;
  for (let guard = 0; guard < 8; guard++) {
    const kids = node.children || [];
    if (kids.length === 0) return [node];
    if (kids.length === 1) {
      if (kids[0].isMesh) return [node];
      node = kids[0];
      continue;
    }
    if (asParts || kids.every(k => !k.isMesh)) return kids.slice();
    return [node];                       // several meshes of one object
  }
  return [node];
}

// Centre and scale that put a model's bounding box at the reactor's size.
// min / max are [x, y, z]. scale maps model units to reactor units.
export function fitModel(min, max, targetRadius = TARGET_RADIUS) {
  const c = [0, 1, 2].map(i => (min[i] + max[i]) / 2);
  const half = 0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  return { center: c, radius: half, scale: half > 1e-9 ? targetRadius / half : 1 };
}

// Explode direction and distance for a part whose centre is `offset` from the
// model's centre (any units). Parts at the centre alternate front / back.
export function explodeDir(offset, index) {
  const len = Math.hypot(offset[0], offset[1], offset[2]);
  if (len < 1e-9) return { dir: [0, 0, index % 2 ? -1 : 1], dist: 0.9 };
  const d = [offset[0] / len, offset[1] / len, offset[2] / len + 0.3];
  const n = Math.hypot(d[0], d[1], d[2]);
  return { dir: [d[0] / n, d[1] / n, d[2] / n], dist: 0.9 };
}

// Find a model in the server's list: "id:abc123def456", else the exact name
// (any case), else a unique name that starts with / contains the query.
export function findModel(list, query) {
  const q = String(query || "").trim();
  if (!q || !Array.isArray(list)) return null;
  if (q.toLowerCase().startsWith("id:")) {
    const id = q.slice(3).trim().toLowerCase();
    return list.find(m => m.id === id) || null;
  }
  const lq = q.toLowerCase();
  const exact = list.filter(m => String(m.name).toLowerCase() === lq);
  if (exact.length) return exact[0];
  for (const test of [m => String(m.name).toLowerCase().startsWith(lq),
                      m => String(m.name).toLowerCase().includes(lq)]) {
    const hits = list.filter(test);
    if (hits.length === 1) return hits[0];
  }
  return null;
}

// One page of the model picker: {items, page, pages, more}.
export function pageOf(list, page, per = MODELS_PER_PAGE) {
  const pages = Math.max(1, Math.ceil(list.length / per));
  const p = Math.min(Math.max(0, page | 0), pages - 1);
  return { items: list.slice(p * per, p * per + per), page: p, pages, more: p < pages - 1 };
}

// ---------------------------------------------------------------------------
// talking to the server
// ---------------------------------------------------------------------------
export async function fetchModelList(headers = {}) {
  const r = await fetch("/api/holo/models", { headers, cache: "no-store" });
  if (!r.ok) throw new Error("model list: HTTP " + r.status);
  const d = await r.json();
  return Array.isArray(d.models) ? d.models : [];
}

export async function fetchModelBytes(id, headers = {}) {
  const r = await fetch("/holo/model/" + encodeURIComponent(id), { headers, cache: "no-store" });
  if (r.status === 413) throw new Error("file is over the size cap");
  if (r.status === 404) throw new Error("file not found");
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.arrayBuffer();
}

// ---------------------------------------------------------------------------
// three.js side
// ---------------------------------------------------------------------------
function countTris(node) {
  let n = 0;
  node.traverse(o => {
    if (o.isMesh && o.geometry) {
      const g = o.geometry;
      n += g.index ? g.index.count / 3 : (g.attributes.position ? g.attributes.position.count / 3 : 0);
    }
  });
  return Math.round(n);
}

// Parse a file into a three.js object (Mesh or Group), in the file's own units
// and axes. kind: "stl" | "3mf" | "obj".
export async function parseModel(THREE, kind, buffer) {
  const url = LOADER_URLS[kind];
  if (!url) throw new Error("unsupported model type " + kind);
  const mod = await import(url);
  if (kind === "stl") {
    const geo = new mod.STLLoader().parse(buffer);
    if (!geo.attributes.normal) geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      color: 0xbfeaf5, roughness: 0.5, metalness: 0.25, vertexColors: !!geo.hasColors,
    });
    return new THREE.Mesh(geo, mat);
  }
  if (kind === "3mf") return new mod.ThreeMFLoader().parse(buffer);
  return new mod.OBJLoader().parse(new TextDecoder("utf-8").decode(buffer));
}

export class HoloModel {
  constructor(THREE, name) {
    this.THREE = THREE;
    this.name = name;
    this.group = new THREE.Group();       // moved / scaled / spun by holo.js
    this.parts = [];                      // same shape as Reactor.parts
    this.e = 0;
    this.exploded = false;
    this.sizeMm = [0, 0, 0];              // whole model, as authored
    this.triangles = 0;
  }
  setExploded(on) { this.exploded = !!on; }
  step(dt, t) { stepParts(this.THREE, this, dt, t); }
  partByName(n) { return this.parts.find(p => p.name === n) || null; }
  dispose() {
    this.group.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      const m = o.material;
      if (m) for (const x of (Array.isArray(m) ? m : [m])) x.dispose();
    });
  }
}

// Turn a parsed object into a HoloModel: Z-up files stand upright, every part
// gets a pivot at its own centre (so it spins and flies out from there), and
// the whole thing is centred and scaled to the reactor's size.
export function buildModel(THREE, root, name, kind) {
  root.updateMatrixWorld(true);
  let nodes = splitParts(root, { meshesAreParts: kind === "obj" });
  if (nodes.length > MAX_PARTS) nodes = [root];

  const rows = [];
  for (const node of nodes) {
    node.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(node);
    if (box.isEmpty()) continue;
    const size = box.getSize(new THREE.Vector3());
    rows.push({ node, size: [size.x, size.y, size.z], tris: countTris(node) });
  }
  if (!rows.length) throw new Error("the file has no geometry");

  const whole = new THREE.Box3().setFromObject(root);
  const ws = whole.getSize(new THREE.Vector3());
  const model = new HoloModel(THREE, name);
  model.sizeMm = [ws.x, ws.y, ws.z];
  model.triangles = rows.reduce((a, r) => a + r.tris, 0);

  const R = new THREE.Matrix4().makeRotationX(-Math.PI / 2);     // Z-up -> Y-up
  const all = new THREE.Box3();
  for (const r of rows) {
    r.box = new THREE.Box3().setFromObject(r.node).applyMatrix4(R);
    all.union(r.box);
  }
  const fit = fitModel([all.min.x, all.min.y, all.min.z], [all.max.x, all.max.y, all.max.z]);
  const C = new THREE.Vector3(...fit.center);

  const names = uniqueNames(rows.map(r => (rows.length === 1 ? name : r.node.name)));
  rows.forEach((r, i) => {
    const pc = r.box.getCenter(new THREE.Vector3());
    const off = pc.clone().sub(C);
    const pivot = new THREE.Group();
    pivot.name = names[i];
    pivot.scale.setScalar(fit.scale);
    pivot.position.copy(off).multiplyScalar(fit.scale);
    // node world matrix -> relative to the part's own centre, upright
    const M = new THREE.Matrix4().makeTranslation(-pc.x, -pc.y, -pc.z)
      .multiply(R).multiply(r.node.matrixWorld);
    M.decompose(r.node.position, r.node.quaternion, r.node.scale);
    r.node.updateMatrix();
    pivot.add(r.node);
    model.group.add(pivot);

    r.node.traverse(o => {
      if (!o.isMesh) return;
      for (const m of (Array.isArray(o.material) ? o.material : [o.material])) {
        if (m) { m.side = THREE.DoubleSide; m.needsUpdate = true; }
      }
      if (o.geometry && countTris(o) < EDGE_TRI_LIMIT) {     // hologram look: cyan edges
        o.add(new THREE.LineSegments(new THREE.EdgesGeometry(o.geometry, 28),
          new THREE.LineBasicMaterial({ color: 0x39d7ff, transparent: true, opacity: 0.35 })));
      }
    });

    const ex = explodeDir([off.x, off.y, off.z], i);
    const dir = new THREE.Vector3(...ex.dir);
    const info = "Size " + formatSizeMm(...r.size) + ". " + r.tris.toLocaleString("en-US") + " triangles.";
    const part = {
      name: names[i], info, mesh: pivot, rest: pivot.position.clone(),
      dir, dist: ex.dist, spin: 0.5 * (i % 2 ? 1 : -1), pull: null,
    };
    pivot.userData.part = part;
    model.parts.push(part);
  });
  return model;
}

// Fetch + parse + build, in one go.
export async function loadOwnModel(THREE, entry, headers = {}) {
  const buf = await fetchModelBytes(entry.id, headers);
  const root = await parseModel(THREE, entry.type, buf);
  return buildModel(THREE, root, entry.name, entry.type);
}
