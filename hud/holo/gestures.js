// Holo mode - the PAGE-SIDE gesture logic. Pure ES module: no DOM, no three.js,
// no timers, no network, so selftest.js can run it with a mock scene.
//
// WHO DOES WHAT
//   Gesture RECOGNITION (is this a fist? a pinch held 0.6 s? two fists pulled
//   apart?) lives in the tracker service (tools/hand_tracker_service.py,
//   thresholds + hysteresis documented there). It sends:
//     {"type":"hands", "hands":[{id, side, lm[21][x,y,z], state, pinch, palm, palm_facing}]}
//     {"type":"gesture", name, side, pos:[x,y], value?}
//   names: palm_hold pinch_hold fist_grab fist_release point_flick explode
//          clap pinch_start pinch_end (pinch_part = older alias of pinch_start)
//   This module decides what the SCENE does with them (the "router"), plus the
//   small pieces of math the scene needs. Coordinates are normalized 0..1,
//   already mirrored for display by the service.
//
// WHAT THE SCENE (holo.js) MUST PROVIDE (duck-typed; see MockScene in selftest):
//   scene.reactorVisible, scene.exploded           (read)
//   scene.reactorScreen() -> {x, y, r}             px centre + radius
//   scene.toScreen(nx, ny) -> {x, y}               normalized -> px
//   scene.openMenu(x, y) / closeMenu() / menuOpen  (read for menuOpen)
//   scene.menuItemAt(x, y) -> item|null            what the menu has under a px point
//   scene.chooseMenuItem(item)
//   scene.spawn(model, x, y)
//   scene.setReactorTarget(x, y)                   the reactor follows (smoothed by the scene)
//   scene.setScale(s) / scene.scale                two-hand scale
//   scene.addSpin(wy, wx)                          angular velocity impulse, rad/s
//   scene.explode() / scene.assemble()
//   scene.partAt(x, y) -> part|null                nearest part to a px point (exploded only)
//   scene.pullPart(part) / movePart(part, x, y) / releasePart(part) / showInfo(part|null)
//   OPTIONAL (Kinect depth; a scene without them just has no push/pull):
//   scene.beginPushPull(handZ) / pushPull(handZ, dt) / endPushPull()
//                                                  a fist holds the hologram and the
//                                                  hand has a depth ("z", metres)
//
// NEVER moves the OS cursor or sends OS input - nothing here can.

export const T = Object.freeze({
  // Fist: grab only when the fist starts within this many reactor radii of the
  // reactor centre (plus a floor in px so a small reactor is still catchable).
  GRAB_RADII: 1.8,
  GRAB_MIN_PX: 200,
  // Point-and-flick: the service (tools/handtrack_gestures.py) sends the
  // wrist->index-tip direction's angular speed in deg/s (+ = clockwise on
  // screen, an upward finger swinging right) and only fires at >= 220 deg/s.
  // SPIN_GAIN turns deg/s into rad/s 1:1; clamped so one flick cannot make an
  // unreadable blur.
  FLICK_MIN: 200,
  SPIN_GAIN: Math.PI / 180,
  SPIN_MAX: 14,
  SPIN_DAMP_PER_S: 1.6,            // omega *= exp(-k t)
  // Exploded: pinch a part to pull it out, hold it this long for its info.
  INFO_HOLD_S: 1.0,
  PART_PICK_PX: 70,
  // A pinch that stopped for this long counts as released (jitter guard).
  PINCH_RELEASE_S: 0.15,
  // Two-hand scale: both hands pinching, distance ratio drives scale.
  SCALE_MIN: 0.4,
  SCALE_MAX: 3.0,
  // After explode/assemble ignore the opposite event for this long (the end of
  // a clap must not read as the start of an explode).
  SWITCH_LOCK_S: 1.2,
  // Follow smoothing for the fist (seconds; smaller = snappier).
  FOLLOW_TAU_S: 0.08,
});

export function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

export function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

// Normalized (mirrored) coordinates -> pixels.
//   desktop: the whole window is the hand space (stretch).
//   camera : the frame is drawn "cover"-fitted, so the same fit applies here
//            (aspect = frame w/h from the hands message, 4:3 if unknown).
export function mapToScreen(nx, ny, W, H, mode, aspect) {
  if (mode !== "camera") return { x: nx * W, y: ny * H };
  const a = aspect > 0 ? aspect : 4 / 3;
  let fw, fh;                       // drawn frame size in px (cover)
  if (W / H > a) { fw = W; fh = W / a; } else { fh = H; fw = H * a; }
  return { x: (nx - 0.5) * fw + W / 2, y: (ny - 0.5) * fh + H / 2 };
}

// Inverse of mapToScreen (px -> normalized), for placing fake hands on UI.
export function mapFromScreen(px, py, W, H, mode, aspect) {
  if (mode !== "camera") return [px / W, py / H];
  const a = aspect > 0 ? aspect : 4 / 3;
  let fw, fh;
  if (W / H > a) { fw = W; fh = W / a; } else { fh = H; fw = H * a; }
  return [(px - W / 2) / fw + 0.5, (py - H / 2) / fh + 0.5];
}

// Midpoint of thumb tip (4) and index tip (8), normalized.
export function pinchPoint(lm) {
  return [(lm[4][0] + lm[8][0]) / 2, (lm[4][1] + lm[8][1]) / 2];
}

// Palm size in normalized units: wrist (0) to middle-finger MCP (9).
export function palmSize(lm) { return dist(lm[0], lm[9]); }

export function flickToSpin(value) {
  if (!(Math.abs(value) >= T.FLICK_MIN)) return 0;
  return clamp(value * T.SPIN_GAIN, -T.SPIN_MAX, T.SPIN_MAX);
}

export function dampSpin(omega, dt) {
  return omega * Math.exp(-T.SPIN_DAMP_PER_S * dt);
}

// Exponential smoothing step toward a target.
export function followStep(cur, target, dt, tau) {
  const k = 1 - Math.exp(-dt / (tau || T.FOLLOW_TAU_S));
  return cur + (target - cur) * k;
}

export function twoHandScale(startScale, startDist, nowDist) {
  if (!(startDist > 1e-4)) return startScale;
  return clamp(startScale * (nowDist / startDist), T.SCALE_MIN, T.SCALE_MAX);
}

export class GestureRouter {
  constructor(scene, opts = {}) {
    this.scene = scene;
    this.hands = new Map();          // side -> latest hand
    this.grab = null;                // {side}
    this.pulled = null;              // {side, part, heldS, shown, lostS}
    this.lockUntil = 0;
    this.now = 0;                    // seconds, advanced by update()
    this.twoHand = null;             // {d0, s0}
    this._pinching = new Set();      // sides currently in state "pinch"
    this._menuPinchEdge = false;
    this.log = opts.log || (() => {});
  }

  // ---- continuous state: one "hands" message -------------------------------
  update(hands, dt) {
    const sc = this.scene;
    this.now += dt;
    const seen = new Set();
    for (const h of hands || []) {
      seen.add(h.side);
      const was = this.hands.get(h.side);
      this.hands.set(h.side, h);
      const nowPinch = h.state === "pinch";
      const wasPinch = this._pinching.has(h.side);
      if (nowPinch && !wasPinch) { this._pinching.add(h.side); this._onPinchStart(h); }
      if (!nowPinch && wasPinch) { this._pinching.delete(h.side); }
      void was;
    }
    for (const side of [...this.hands.keys()]) {
      if (!seen.has(side)) { this.hands.delete(side); this._pinching.delete(side); }
    }

    // Fist: the reactor follows the grabbing hand's palm.
    if (this.grab) {
      const h = this.hands.get(this.grab.side);
      // The hand that grabbed must still be a fist; opening it (or losing it)
      // for T.PINCH_RELEASE_S lets go even if fist_release never arrives.
      if (h && h.state === "fist") this.grab.lostS = 0;
      else this.grab.lostS = (this.grab.lostS || 0) + dt;
      if (!h || this.grab.lostS >= T.PINCH_RELEASE_S) { this._endGrab(); }
      else if (h.palm && h.state === "fist") {
        const p = sc.toScreen(h.palm[0], h.palm[1]);
        sc.setReactorTarget(p.x, p.y);
        // Push / pull: the fist's depth (the tracker's "z", metres) moves the
        // hologram toward / away from the sensor. The first depth the grab sees
        // is its baseline.
        if (typeof h.z === "number" && h.z > 0 && sc.beginPushPull && sc.pushPull) {
          if (!this.grab.pp) { this.grab.pp = true; sc.beginPushPull(h.z); }
          else sc.pushPull(h.z, dt);
        }
      }
    }

    // Pulled part follows the pinch point; hold = info; release on un-pinch.
    if (this.pulled) {
      const pl = this.pulled;
      const h = this.hands.get(pl.side);
      if (h && h.state === "pinch") {
        pl.lostS = 0;
        const pp = sc.toScreen(...pinchPoint(h.lm));
        sc.movePart(pl.part, pp.x, pp.y);
        pl.heldS += dt;
        if (!pl.shown && pl.heldS >= T.INFO_HOLD_S) { pl.shown = true; sc.showInfo(pl.part); }
      } else {
        pl.lostS += dt;
        if (pl.lostS >= T.PINCH_RELEASE_S) {
          sc.releasePart(pl.part);
          if (pl.shown) sc.showInfo(null);
          this.pulled = null;
        }
      }
    }

    // Two-hand scale: both hands pinching, nothing else holding a hand.
    const L = [...this.hands.values()].filter(h => h.state === "pinch");
    if (L.length === 2 && sc.reactorVisible && !this.pulled && !this.grab && !sc.menuOpen) {
      const d = dist(pinchPoint(L[0].lm), pinchPoint(L[1].lm));
      if (!this.twoHand) this.twoHand = { d0: d, s0: sc.scale || 1 };
      else sc.setScale(twoHandScale(this.twoHand.s0, this.twoHand.d0, d));
    } else {
      this.twoHand = null;
    }
  }

  // A pinch that STARTS while the menu is open and over an item is a click.
  _onPinchStart(h) {
    const sc = this.scene;
    if (!sc.menuOpen) return;
    const tip = sc.toScreen(h.lm[8][0], h.lm[8][1]);
    const item = sc.menuItemAt(tip.x, tip.y);
    if (item) sc.chooseMenuItem(item);
  }

  // ---- events: one "gesture" message ---------------------------------------
  onGesture(ev) {
    const sc = this.scene;
    const pos = ev.pos ? sc.toScreen(ev.pos[0], ev.pos[1]) : sc.reactorScreen();
    switch (ev.name) {
      case "palm_hold":
        if (!sc.menuOpen) sc.openMenu(pos.x, pos.y);
        break;

      case "pinch_hold": {
        if (sc.menuOpen) break;
        if (sc.exploded && sc.partAt(pos.x, pos.y)) break;   // pinch_part owns it
        sc.openMenu(pos.x, pos.y);
        break;
      }

      case "fist_grab": {
        if (!sc.reactorVisible) break;
        const r = sc.reactorScreen();
        const reach = Math.max(r.r * T.GRAB_RADII, T.GRAB_MIN_PX);
        if (Math.hypot(pos.x - r.x, pos.y - r.y) <= reach) {
          this.grab = { side: ev.side };
          sc.setReactorTarget(pos.x, pos.y);
        }
        break;
      }
      case "fist_release":
        this._endGrab();
        break;

      case "point_flick": {
        if (!sc.reactorVisible) break;
        const w = flickToSpin(Number(ev.value) || 0);
        if (w) sc.addSpin(w, 0);
        break;
      }

      case "explode":
        if (sc.reactorVisible && !sc.exploded && this.now >= this.lockUntil) {
          sc.explode();
          this.lockUntil = this.now + T.SWITCH_LOCK_S;
        }
        break;

      case "clap":
        if (sc.reactorVisible && sc.exploded && this.now >= this.lockUntil) {
          this._dropPulled();
          sc.assemble();
          this.lockUntil = this.now + T.SWITCH_LOCK_S;
        }
        break;

      // The service sends plain pinch_start / pinch_end; it does not know the
      // scene, so "is there a part under the pinch" is decided here. Release
      // of a pulled part comes from the continuous hand state (update()).
      case "pinch_end":
        break;
      case "pinch_start":
      case "pinch_part": {
        if (!sc.exploded || this.pulled) break;
        const part = sc.partAt(pos.x, pos.y);
        if (!part) break;
        // The hand that pinched: the event's side, else the nearest pinching one.
        const side = (ev.side === "Left" || ev.side === "Right") ? ev.side
          : ([...this._pinching][0] || "Right");
        sc.pullPart(part);
        this.pulled = { side, part, heldS: 0, shown: false, lostS: 0 };
        break;
      }
      default:
        this.log("unknown gesture " + ev.name);
    }
  }

  // Let go of a fist grab (and end its push / pull if one began).
  _endGrab() {
    const g = this.grab;
    this.grab = null;
    if (g && g.pp && this.scene.endPushPull) this.scene.endPushPull();
  }

  _dropPulled() {
    if (!this.pulled) return;
    this.scene.releasePart(this.pulled.part);
    if (this.pulled.shown) this.scene.showInfo(null);
    this.pulled = null;
  }

  // For the debug overlay.
  summary() {
    return {
      grab: this.grab ? this.grab.side : null,
      pulled: this.pulled ? this.pulled.part.name : null,
      twoHand: !!this.twoHand,
      hands: [...this.hands.values()].map(h => h.side + ":" + h.state),
    };
  }
}
