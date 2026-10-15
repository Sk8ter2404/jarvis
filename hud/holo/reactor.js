// Procedural arc reactor for holo mode: no downloaded model. Every part is a
// separate named mesh with a one-line info text (the "hold for info" label),
// an explode direction and distance ("parts fly out along their own axes").
// THREE is passed in by holo.js so the library URL is pinned in exactly one place.

export const REACTOR_NAME = "arc reactor";

const CYAN = 0x39d7ff;

// Axial parts fly along +/- z (the reactor's own axis); coils fly radially.
const AXIAL = [
  // name, info, z at rest, z explode distance
  ["Back plate", "Rear mounting plate. Carries the whole assembly and bleeds heat into the chest housing.", -0.2, -1.9],
  ["Housing", "Outer case. Holds the coil ring in alignment and shields everything from the wearer.", -0.06, -1.0],
  ["Outer ring", "Thin containment ring. Keeps the field stable at the rim of the coil array.", 0.05, 0.8],
  ["Inner ring", "Focusing ring around the core. Shapes the plasma into a steady disc.", 0.07, 1.4],
  ["Core", "The reactor core. The glowing disc that does all the work.", 0.02, 2.0],
];

function glowTexture(THREE) {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(64, 64, 2, 64, 64, 62);
  grd.addColorStop(0, "rgba(220,255,255,1)");
  grd.addColorStop(0.25, "rgba(90,220,255,0.55)");
  grd.addColorStop(1, "rgba(0,160,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

// Shared by the reactor and the owner's own models (models.js): animate `host.e`
// toward exploded/assembled and move every part (host = {exploded, e, parts}).
// A part held by a pinch follows its hand; a released one eases home.
export function stepParts(THREE, host, dt, t) {
  const target = host.exploded ? 1 : 0;
  // critically-damped-ish ease; explode is a snap, assemble a pull-in
  const k = 1 - Math.exp(-dt * (host.exploded ? 5.5 : 7));
  host.e += (target - host.e) * k;
  if (Math.abs(host.e - target) < 1e-3) host.e = target;
  const tmp = new THREE.Vector3();
  for (const p of host.parts) {
    if (p.pull) {
      // being held: follow the pinch point (given in this group's local space)
      p.mesh.position.lerp(p.pull.local, 1 - Math.exp(-dt * 18));
      continue;
    }
    tmp.copy(p.dir).multiplyScalar(p.dist * host.e).add(p.rest);
    if (host.e > 0.02) {
      // gentle hover + own-axis tumble while exploded
      tmp.z += Math.sin(t * 1.3 + p.rest.x * 3 + p.rest.y * 2) * 0.04 * host.e;
      p.mesh.rotation.y += p.spin * dt * host.e;
    }
    p.mesh.position.lerp(tmp, p.pullBack ? 1 - Math.exp(-dt * 9) : 1);
    if (p.pullBack && p.mesh.position.distanceTo(tmp) < 0.02) p.pullBack = false;
  }
}

export class Reactor {
  constructor(THREE) {
    this.THREE = THREE;
    this.group = new THREE.Group();        // scaled/moved/spun by holo.js
    this.parts = [];                       // {name, info, mesh, rest, dir, dist, spin, pull}
    this.e = 0;                            // 0 assembled .. 1 exploded (animated)
    this.exploded = false;
    this._build();
  }

  _addPart(name, info, mesh, rest, dir, dist, spin = 0) {
    const THREE = this.THREE;
    mesh.name = name;
    mesh.position.copy(rest);
    // hologram look: cyan edge lines over the shaded body
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry, 28),
      new THREE.LineBasicMaterial({ color: CYAN, transparent: true, opacity: 0.55 }));
    mesh.add(edges);
    this.group.add(mesh);
    const part = { name, info, mesh, rest: rest.clone(), dir: dir.clone().normalize(), dist, spin, pull: null };
    mesh.userData.part = part;
    this.parts.push(part);
    return part;
  }

  _build() {
    const THREE = this.THREE;
    const V = (x, y, z) => new THREE.Vector3(x, y, z);
    const metal = (color, emissive, ei, rough = 0.45, met = 0.7) => new THREE.MeshStandardMaterial({
      color, emissive, emissiveIntensity: ei, roughness: rough, metalness: met,
      transparent: true, opacity: 0.93,
    });

    const [plate, housing, outer, inner, core] = AXIAL;

    // back plate - a flat disc facing the viewer's side of the axis
    const plateMesh = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.2, 0.07, 48),
      metal(0x2a3440, 0x06202a, 0.6));
    plateMesh.rotation.x = Math.PI / 2;
    this._addPart(plate[0], plate[1], plateMesh, V(0, 0, plate[2]), V(0, 0, 1), plate[3]);

    // housing - heavy gunmetal torus
    this._addPart(housing[0], housing[1],
      new THREE.Mesh(new THREE.TorusGeometry(1.02, 0.14, 20, 72), metal(0x4b5866, 0x082634, 0.7, 0.4, 0.85)),
      V(0, 0, housing[2]), V(0, 0, 1), housing[3]);

    // outer ring - thin, glowing
    this._addPart(outer[0], outer[1],
      new THREE.Mesh(new THREE.TorusGeometry(0.86, 0.035, 12, 96), metal(0x7fe9ff, CYAN, 1.4, 0.3, 0.2)),
      V(0, 0, outer[2]), V(0, 0, 1), outer[3]);

    // 10 copper coil segments around the ring, each flies out radially
    const COILS = 10;
    for (let i = 0; i < COILS; i++) {
      const a = (i / COILS) * Math.PI * 2;
      const box = new THREE.Mesh(new THREE.BoxGeometry(0.27, 0.2, 0.16),
        metal(0xb87333, 0x3a1a05, 0.5, 0.35, 0.85));
      box.rotation.z = a;
      const r = 0.62;
      const rest = V(Math.cos(a) * r, Math.sin(a) * r, 0);
      this._addPart("Copper coil " + String(i + 1).padStart(2, "0"),
        "Copper coil segment " + (i + 1) + " of " + COILS + ". Wound copper that carries the field around the ring.",
        box, rest, V(Math.cos(a), Math.sin(a), 0.35), 1.4, 0.6 * (i % 2 ? 1 : -1));
    }

    // inner ring
    this._addPart(inner[0], inner[1],
      new THREE.Mesh(new THREE.TorusGeometry(0.34, 0.045, 16, 64), metal(0xdfffff, CYAN, 1.8, 0.2, 0.3)),
      V(0, 0, inner[2]), V(0, 0, 1), inner[3]);

    // core - bright disc with a glow sprite
    const coreMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.1, 40),
      new THREE.MeshBasicMaterial({ color: 0xcfffff }));
    coreMesh.rotation.x = Math.PI / 2;
    const corePart = this._addPart(core[0], core[1], coreMesh, V(0, 0, core[2]), V(0, 0, 1), core[3]);
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTexture(THREE), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    glow.scale.set(1.7, 1.7, 1);
    glow.position.set(0, 1e-3, 0);        // child of the (rotated) disc; sprite ignores rotation
    coreMesh.add(glow);
    corePart.glow = glow;

    this.light = new THREE.PointLight(CYAN, 2.2, 8);
    this.light.position.set(0, 0, 0.8);
    this.group.add(this.light);
  }

  setExploded(on) { this.exploded = !!on; }

  step(dt, t) {
    stepParts(this.THREE, this, dt, t);
    const core = this.parts.find(p => p.name === "Core");
    if (core && core.glow) {
      const pulse = 0.92 + 0.08 * Math.sin(t * 3.1);
      core.glow.scale.setScalar(1.7 * pulse * (this.exploded ? 1.25 : 1));
    }
    this.light.intensity = 2.2 * (0.9 + 0.1 * Math.sin(t * 3.1));
  }

  partByName(n) { return this.parts.find(p => p.name === n) || null; }
}
