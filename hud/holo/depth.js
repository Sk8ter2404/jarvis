// Holo mode - the PAGE-SIDE depth logic (Kinect, Phase 2). Pure ES module: no DOM,
// no three.js, no timers, no network, so selftest.js can run it.
//
// Where the numbers come from: the tracker service (tools/hand_tracker_service.py)
// relays a low-res DEPTH PLANE registered to the camera picture, only to a page
// that sent {"type":"subscribe","depth":true}. One binary WebSocket message per
// plane (a JPEG starts FF D8; this starts "JVDP"):
//
//    0  4  magic "JVDP"        4  1  version (1)        5  1  flags (bit0 = mirrored,
//    always set: the plane is already flipped to match the mirrored picture)
//    6  2  width u16 LE        8  2  height u16 LE      10 2  reserved
//   12  8  seq u64 LE (the colour frame / "hands" seq)  20 2  near_mm   22 2  far_mm
//   24  w*h*2  uint16 LE millimetres, row-major, 0 = unknown (outside 0.5-4.5 m)
//
// The plane covers exactly the picture, so plane pixel (i, j) sits at picture
// coordinates ((i+.5)/w, (j+.5)/h) - the same 0..1 space as the hand landmarks.
// Each hand in a "hands" message also carries "z": the palm's depth in metres
// (null when unknown); there are no "z" keys at all while no depth flows.
//
// WHAT THIS FILE DECIDES
//   * OCCLUSION: the hologram floats at ONE distance holoZ (metres). A plane
//     pixel that reads nearer than holoZ - OCCLUDE_MARGIN_M is "something real in
//     front of it": that pixel of the hologram is hidden. Unknown (0) never
//     hides anything (honest: no depth, no occlusion). Isolated single pixels are
//     dropped (depth noise / flying pixels at silhouettes).
//   * PUSH / PULL: while a fist holds the hologram, the HAND'S depth change since
//     the grab moves the hologram's distance by the same amount (a small dead
//     band keeps sensor jitter out); the size follows perspective,
//     size = zRef / holoZ, clamped. A hand with no depth just holds the distance.
//
// NEVER moves the OS cursor or sends OS input - nothing here can.

export const D = Object.freeze({
  MAGIC: [0x4a, 0x56, 0x44, 0x50],          // "JVDP"
  HEADER: 24,
  VERSION: 1,
  // The hologram's distance from the sensor (metres).
  HOLO_Z_DEFAULT: 1.0,
  HOLO_Z_MIN: 0.6,
  HOLO_Z_MAX: 3.5,
  // Something must be this much NEARER than the hologram to hide it.
  OCCLUDE_MARGIN_M: 0.04,
  // A spawned hologram floats this far in FRONT of the hand that spawned it.
  SPAWN_IN_FRONT_M: 0.05,
  // Push / pull: 1 m of hand travel = PUSH_GAIN m of hologram travel; the first
  // PUSH_DEADBAND_M of travel is ignored (sensor noise), the hand depth is
  // smoothed with time constant PUSH_TAU_S.
  PUSH_GAIN: 1.0,
  PUSH_DEADBAND_M: 0.015,
  PUSH_TAU_S: 0.12,
  // Perspective size limits for the hologram.
  SIZE_MIN: 0.5,
  SIZE_MAX: 2.5,
  // A plane older than this (ms) means depth stopped flowing.
  STALE_MS: 700,
});

export function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ArrayBuffer -> {w, h, seq, mirrored, nearMm, farMm, mm: Uint16Array} | null.
// Uses DataView (explicit little-endian), copies the plane (the socket buffer is
// not kept). Rejects anything that is not exactly a well-formed message.
export function parseDepthMessage(buf) {
  if (!buf || !(buf.byteLength >= D.HEADER)) return null;
  const dv = new DataView(buf);
  for (let i = 0; i < 4; i++) if (dv.getUint8(i) !== D.MAGIC[i]) return null;
  if (dv.getUint8(4) !== D.VERSION) return null;
  const flags = dv.getUint8(5);
  const w = dv.getUint16(6, true), h = dv.getUint16(8, true);
  if (!(w > 0 && h > 0) || buf.byteLength !== D.HEADER + w * h * 2) return null;
  const seq = Number(dv.getBigUint64(12, true));
  const mm = new Uint16Array(w * h);
  for (let i = 0; i < mm.length; i++) mm[i] = dv.getUint16(D.HEADER + i * 2, true);
  return { w, h, seq, mirrored: (flags & 1) === 1,
    nearMm: dv.getUint16(20, true), farMm: dv.getUint16(22, true), mm };
}

// True when the first four bytes of a binary message are the depth magic (the
// page tells depth from a JPEG without decoding anything).
export function isDepthMessage(buf) {
  if (!buf || !(buf.byteLength >= 4)) return false;
  const b = new Uint8Array(buf, 0, 4);
  return b[0] === D.MAGIC[0] && b[1] === D.MAGIC[1] && b[2] === D.MAGIC[2] && b[3] === D.MAGIC[3];
}

// Occlusion mask for a plane: 255 where the real surface is nearer than the
// hologram (hide the hologram), else 0. `out` (Uint8Array w*h) is reused if given.
//   - unknown (0) never hides
//   - a pixel counts only if at least one 4-neighbour is also an occluder
//     (an isolated pixel is noise, a real object is many pixels)
export function occlusionMask(mm, w, h, holoZ, out) {
  const n = w * h;
  const raw = new Uint8Array(n);
  const lim = Math.round((holoZ - D.OCCLUDE_MARGIN_M) * 1000);
  for (let i = 0; i < n; i++) { const v = mm[i]; raw[i] = (v > 0 && v < lim) ? 1 : 0; }
  const m = (out && out.length === n) ? out : new Uint8Array(n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!raw[i]) { m[i] = 0; continue; }
      const nb = (x > 0 && raw[i - 1]) || (x < w - 1 && raw[i + 1]) ||
        (y > 0 && raw[i - w]) || (y < h - 1 && raw[i + w]);
      m[i] = nb ? 255 : 0;
    }
  }
  return m;
}

// Fraction of plane pixels that are occluders (a cheap "is anything in front?").
export function maskCoverage(mask) {
  let c = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) c++;
  return mask.length ? c / mask.length : 0;
}

// Median depth (metres) in a (2*half+1)-square window around normalised (nx, ny);
// 0 / unknown samples are ignored; null with fewer than minValid valid ones.
// Mirrors the tracker's palm-z sampler (core/handtrack_depth.py sample_median_m).
export function medianDepthM(mm, w, h, nx, ny, half = 2, minValid = 5) {
  if (!(nx >= 0 && nx <= 1 && ny >= 0 && ny <= 1)) return null;
  const cx = Math.min(w - 1, Math.max(0, Math.floor(nx * w)));
  const cy = Math.min(h - 1, Math.max(0, Math.floor(ny * h)));
  const vals = [];
  for (let y = Math.max(0, cy - half); y <= Math.min(h - 1, cy + half); y++) {
    for (let x = Math.max(0, cx - half); x <= Math.min(w - 1, cx + half); x++) {
      const v = mm[y * w + x];
      if (v > 0) vals.push(v);
    }
  }
  if (vals.length < minValid) return null;
  vals.sort((a, b) => a - b);
  const mid = vals.length >> 1;
  const med = (vals.length & 1) ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
  return med / 1000;
}

// Where a hologram spawned with no hand to anchor it should float: a little in
// FRONT of the nearest ~10 % of what the Kinect sees (that is the owner), so he
// does not hide his own hologram. 1 m when the plane has nothing valid.
export function defaultHoloZ(mm) {
  const vals = [];
  for (let i = 0; i < mm.length; i++) if (mm[i] > 0) vals.push(mm[i]);
  if (vals.length < 20) return D.HOLO_Z_DEFAULT;
  vals.sort((a, b) => a - b);
  const p10 = vals[Math.floor(vals.length * 0.1)] / 1000;
  return clamp(p10 - 0.1, D.HOLO_Z_MIN, D.HOLO_Z_MAX);
}

// Distance for a hologram spawned by a hand at depth handZ (metres, may be null).
export function spawnZ(handZ) {
  if (typeof handZ !== "number" || !(handZ > 0)) return D.HOLO_Z_DEFAULT;
  return clamp(handZ - D.SPAWN_IN_FRONT_M, D.HOLO_Z_MIN, D.HOLO_Z_MAX);
}

// Perspective size factor: 1 at the reference distance, bigger when nearer.
export function sizeAtDistance(zRef, z) {
  if (!(z > 0) || !(zRef > 0)) return 1;
  return clamp(zRef / z, D.SIZE_MIN, D.SIZE_MAX);
}

// Push / pull. begin() when the fist grabs, update() every hands message while it
// holds, end() when it lets go. update() returns the hologram's distance.
export class PushPull {
  constructor() { this.active = false; this.holoZ0 = D.HOLO_Z_DEFAULT; this.hand0 = 0; this.handS = 0; this.z = D.HOLO_Z_DEFAULT; }

  begin(handZ, holoZ) {
    if (typeof handZ !== "number" || !(handZ > 0)) return false;
    this.active = true;
    this.hand0 = this.handS = handZ;
    this.holoZ0 = this.z = holoZ;
    return true;
  }

  update(handZ, dt) {
    if (!this.active) return this.z;
    if (typeof handZ === "number" && handZ > 0) {      // no depth this frame: hold
      const k = 1 - Math.exp(-Math.max(0, dt) / D.PUSH_TAU_S);
      this.handS += (handZ - this.handS) * k;
      const d = this.handS - this.hand0;
      const ad = Math.abs(d);
      const eff = ad <= D.PUSH_DEADBAND_M ? 0 : Math.sign(d) * (ad - D.PUSH_DEADBAND_M);
      this.z = clamp(this.holoZ0 + D.PUSH_GAIN * eff, D.HOLO_Z_MIN, D.HOLO_Z_MAX);
    }
    return this.z;
  }

  end() { this.active = false; return this.z; }
}
