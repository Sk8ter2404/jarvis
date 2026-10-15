"""Kinect colour + DEPTH hand-off for the hologram (Phase 2, 2026-10-15).

WHAT IT DOES
------------
The hand tracker (tools/hand_tracker_service.py) reads frames JARVIS already
captured (core/handtrack_frames.py, segment ``jarvis_handtrack_frame``). Phase 1
fed it a webcam. With ``HAND_TRACK_CAMERA = "kinect"`` this module feeds it the
KINECT's colour stream instead and, with ``HOLO_DEPTH_ENABLED`` also on, a small
depth plane REGISTERED TO THAT COLOUR FRAME in a second segment, so the hologram
page can be hidden behind real objects (occlusion) and pushed / pulled in depth.

Both are OFF by default (a hologram opened by voice "with the Kinect" turns them on
for exactly as long as that hologram is up, see ``holo_session`` - no setting is
rewritten). Nothing here opens the Kinect: ``KinectFeed.step()`` is
called by the face-tracking loop (the thread that already reads Kinect colour
for the preview composite) and only uses the bridge's own accessors
(``get_stream_health`` for "is a runtime open", ``frame_clock`` to skip a frame
it already sent, ``get_color_bgr`` / ``get_depth`` for the pixels), so the
bridge's freshness rules apply unchanged and the pump is never touched. It
NEVER raises; at most ``MAX_FPS`` frames per second; with no tracker alive or
another camera selected it is one string compare.

THE DEPTH SEGMENT  (name ``jarvis_handtrack_depth``; little-endian)
-------------------------------------------------------------------
  offset  size  field
       0     8  magic            b"JVHD0001"
       8     8  seq              uint64, incremented AFTER the plane copy
      16     8  timestamp_ms     uint64, wall clock (time.time()*1000)
      24     4  width            uint32 (160)
      28     4  height           uint32 (90)
      32     4  bytes_per_sample uint32 (2: uint16 millimetres)
      36     4  method           uint32 registration method (0 nominal FOV, 1 SDK fit)
      40     8  seq_begin        uint64, set to seq+1 BEFORE the copy (seqlock)
      48     8  color_seq        uint64, the colour frame (jarvis_handtrack_frame
                                  seq) this plane is registered to
      56     8  (reserved, zero)
      64   w*h*2 uint16 millimetres, row-major, 0 = unknown / outside 0.5-4.5 m

The plane covers EXACTLY the published colour frame (same field of view, same
orientation, not mirrored): plane pixel (i, j) is the nearest valid depth that
falls inside the colour pixels (i/w, j/h) of the frame.

REGISTRATION  (depth 512x424 -> colour frame)
---------------------------------------------
Each valid depth pixel (u, v, z) is sent to colour pixel
    cx = kx*u + ox + bx/Z        cy = ky*v + oy + by/Z      (Z in metres)
(an affine map in depth pixels plus the parallax of the ~5.2 cm depth-to-colour
baseline). The six numbers come from one of two places:
  * method 1, SDK fit: the bridge already exposes the Kinect coordinate mapper
    one point at a time (get_color_space_mapper / get_depth_space_mapper, camera
    space -> colour / depth pixel). ``fit_calibration`` calls them ONCE on a
    small lattice (<= 60 calls, never per frame) and least-squares fits the six
    numbers. The fit's worst residual is kept; a fit worse than FIT_MAX_RESID_PX
    (colour pixels of 1920x1080) is rejected. Lens distortion is not linear, so
    expect ~10-15 px (about 1 % of the frame) at the edges; the number is in
    status().
  * method 0, nominal FOV: no mapper available -> the datasheet fields of view
    (depth ~70.6 x 60 deg on 512x424, colour ~84.1 x 53.8 deg on 1920x1080) with
    a pinhole model and NO baseline term. Honest error: the missing baseline
    shifts the depth ~56/Z pixels of the 1920 frame (about 70 px = 3.6 % of the
    width at 0.8 m, 28 px at 2 m); datasheet round numbers and ignored lens
    distortion add 2-3 % at the edges. Good enough to hide a hand-sized object,
    not to trace a fingertip.
Where depth does not overlap the colour frame (the colour is wider
horizontally, the depth taller vertically) the plane reads 0 (unknown). The
nearest sample wins inside a plane cell (conservative for occlusion). Depth
outside NEAR_MM..FAR_MM is 0. Depth and colour are two frames <= ~33 ms apart.

THE WIRE MESSAGE  (tracker -> page, binary WebSocket message; ``WIRE_*``)
------------------------------------------------------------------------
  0   4  magic "JVDP"            (a JPEG starts FF D8, so the two never clash)
  4   1  version = 1
  5   1  flags: bit0 = plane is MIRRORED for display (the tracker mirrors it so
                 it lines up with the mirrored picture and landmarks)
  6   2  width  uint16 LE
  8   2  height uint16 LE
 10   2  reserved = 0
 12   8  seq    uint64 LE: the colour frame seq it belongs to ("hands".seq)
 20   2  near_mm uint16 LE (500)
 22   2  far_mm  uint16 LE (4500)
 24  w*h*2 uint16 LE millimetres, row-major, 0 = unknown
Sent only to clients that subscribed with {"type":"subscribe","depth":true},
at most once per new depth plane (<= 15/s, ~29 KB each).

numpy is imported lazily; the tracker's venv imports this very file by path for
the reader half, so the layout lives in one place and it imports nothing from
JARVIS at module level.
"""
from __future__ import annotations

import math
import struct
import time
from typing import Optional

DEPTH_SHM_NAME = "jarvis_handtrack_depth"
MAGIC = b"JVHD0001"
OUT_W = 160
OUT_H = 90
MAX_W = 320
MAX_H = 180
HEADER_SIZE = 64
TOTAL_SIZE = HEADER_SIZE + MAX_W * MAX_H * 2
NEAR_MM = 500
FAR_MM = 4500
COLOR_W = 1920
COLOR_H = 1080
DEPTH_W = 512
DEPTH_H = 424
DEPTH_STRIDE = 2                  # every 2nd depth pixel each way: 256x212 samples

METHOD_NOMINAL = 0
METHOD_SDK_FIT = 1

# magic, seq, timestamp_ms, width, height, bytes_per_sample, method, seq_begin, color_seq
_HEAD_FMT = "<8sQQIIIIQQ"
OFF_SEQ = 8
OFF_WH = 24
OFF_SEQ_BEGIN = 40
OFF_COLOR_SEQ = 48

MAX_FPS = 15.0
MIN_GAP_S = 0.06                  # ~16 fps ceiling; the loop's own pace (~12-15) rules
IDLE_CLOSE_S = 60.0
ERROR_BACKOFF_S = 5.0
FIT_RETRY_S = 30.0
FIT_MAX_RESID_PX = 40.0           # colour pixels (of 1920): worse fit = rejected
FRESH_DEPTH_S = 0.5               # tracker side: an older plane counts as no depth

WIRE_MAGIC = b"JVDP"
WIRE_VERSION = 1
WIRE_HEADER = 24
_WIRE_FMT = "<4sBBHHHQHH"
FLAG_MIRRORED = 1

# Datasheet fields of view (degrees) used when no SDK mapper is available.
DEPTH_FOV = (70.6, 60.0)
COLOR_FOV = (84.1, 53.8)


KINECT_WORDS = ("kinect", "kinect2", "kinect v2", "kinect_v2")


def is_kinect_setting(setting) -> bool:
    return str(setting or "").strip().lower() in KINECT_WORDS


# ── calibration ──────────────────────────────────────────────────────────
class Calib:
    """colour_x = kx*u + ox + bx/Z ; colour_y = ky*v + oy + by/Z  (u, v = depth
    pixels on 512x424; Z in metres; colour pixels on 1920x1080)."""

    __slots__ = ("kx", "ox", "bx", "ky", "oy", "by", "method", "residual_px")

    def __init__(self, kx, ox, bx, ky, oy, by, method=METHOD_NOMINAL,
                 residual_px=None):
        self.kx, self.ox, self.bx = float(kx), float(ox), float(bx)
        self.ky, self.oy, self.by = float(ky), float(oy), float(by)
        self.method = int(method)
        self.residual_px = residual_px

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


def nominal_calibration() -> Calib:
    """The fixed approximate mapping from the datasheet fields of view. Pixel
    position is proportional to tan(angle) for a pinhole camera, so depth and
    colour pixels are related by an exact affine map for rays through the same
    optical centre; the real ~5.2 cm baseline is NOT modelled (bx = by = 0)."""
    t = lambda deg: math.tan(math.radians(deg) / 2.0)          # noqa: E731
    fdx = (DEPTH_W / 2.0) / t(DEPTH_FOV[0])
    fdy = (DEPTH_H / 2.0) / t(DEPTH_FOV[1])
    fcx = (COLOR_W / 2.0) / t(COLOR_FOV[0])
    fcy = (COLOR_H / 2.0) / t(COLOR_FOV[1])
    kx, ky = fcx / fdx, fcy / fdy
    return Calib(kx, COLOR_W / 2.0 - kx * DEPTH_W / 2.0, 0.0,
                 ky, COLOR_H / 2.0 - ky * DEPTH_H / 2.0, 0.0, METHOD_NOMINAL, None)


def fit_calibration(color_map, depth_map) -> Optional[Calib]:
    """Fit a Calib from the bridge's per-point SDK mappers.

    ``color_map(x, y, z)`` / ``depth_map(x, y, z)``: camera-space metres ->
    (px, py) on the 1920x1080 colour / 512x424 depth frame, or None. Called on
    a lattice of rays (x/Z in +-0.6, +-0.3, 0; y/Z in +-0.4, 0) at two distances
    (1 m, 2.5 m); returns None when too few points map or the worst residual is
    over FIT_MAX_RESID_PX. Five columns, not three, so the residual can actually
    SEE the lens's bowing (three points of an odd distortion are collinear). Pure
    apart from the two callables; at most 60 calls.
    """
    try:
        import numpy as np
        rows = []
        for z in (1.0, 2.5):
            for tx in (-0.6, -0.3, 0.0, 0.3, 0.6):
                for ty in (-0.4, 0.0, 0.4):
                    c = color_map(tx * z, ty * z, z)
                    d = depth_map(tx * z, ty * z, z)
                    if c is None or d is None:
                        continue
                    rows.append((d[0], d[1], 1.0 / z, c[0], c[1]))
        if len(rows) < 9:
            return None
        a = np.asarray(rows, dtype=np.float64)
        ax = np.stack([a[:, 0], np.ones(len(a)), a[:, 2]], axis=1)
        ay = np.stack([a[:, 1], np.ones(len(a)), a[:, 2]], axis=1)
        sx, *_ = np.linalg.lstsq(ax, a[:, 3], rcond=None)
        sy, *_ = np.linalg.lstsq(ay, a[:, 4], rcond=None)
        resid = max(float(np.abs(ax @ sx - a[:, 3]).max()),
                    float(np.abs(ay @ sy - a[:, 4]).max()))
        vals = [float(v) for v in (*sx, *sy)]
        if resid > FIT_MAX_RESID_PX or not all(math.isfinite(v) for v in vals):
            return None
        # sanity: depth u/v must still map "rightwards"/"downwards" across the
        # colour frame by roughly the FOV ratio (0.5..8x)
        if not (0.5 < abs(sx[0]) < 8.0 and 0.5 < abs(sy[0]) < 8.0):
            return None
        return Calib(sx[0], sx[1], sx[2], sy[0], sy[1], sy[2],
                     METHOD_SDK_FIT, round(resid, 1))
    except Exception:
        return None


_GRID_CACHE: dict = {}


def _grid(h: int, w: int, stride: int):
    key = (h, w, stride)
    g = _GRID_CACHE.get(key)
    if g is None:
        import numpy as np
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        g = (xs.astype(np.float32), ys.astype(np.float32))
        _GRID_CACHE[key] = g
    return g


def register_depth(depth_mm, calib: Calib, out_w: int = OUT_W, out_h: int = OUT_H,
                   color_w: int = COLOR_W, color_h: int = COLOR_H,
                   near_mm: int = NEAR_MM, far_mm: int = FAR_MM,
                   stride: int = DEPTH_STRIDE):
    """A (out_h, out_w) uint16 millimetre plane aligned with the colour frame
    from a (424, 512) depth frame; 0 = unknown. The nearest depth wins a cell."""
    import numpy as np
    dh, dw = depth_mm.shape[:2]
    xs, ys = _grid(dh, dw, stride)
    d = depth_mm[::stride, ::stride]
    ok = (d >= near_mm) & (d <= far_mm)
    out = np.zeros(out_h * out_w, dtype=np.uint16)
    if not ok.any():
        return out.reshape(out_h, out_w)
    z = d[ok].astype(np.float32)
    inv_z = 1000.0 / z                                   # 1/Z with Z in metres
    cx = calib.kx * xs[ok] + calib.ox + calib.bx * inv_z
    cy = calib.ky * ys[ok] + calib.oy + calib.by * inv_z
    ix = np.floor(cx * (out_w / float(color_w))).astype(np.int64)
    iy = np.floor(cy * (out_h / float(color_h))).astype(np.int64)
    inside = (ix >= 0) & (ix < out_w) & (iy >= 0) & (iy < out_h)
    if not inside.any():
        return out.reshape(out_h, out_w)
    cell = (iy[inside] * out_w + ix[inside]).astype(np.int64)
    zz = z[inside].astype(np.int64)
    # one winner per cell, the smallest z: sort by (cell, z), keep first per cell
    key = np.sort((cell << 16) | zz)
    cells = key >> 16
    first = np.ones(len(key), dtype=bool)
    first[1:] = cells[1:] != cells[:-1]
    out[cells[first]] = (key[first] & 0xFFFF).astype(np.uint16)
    return out.reshape(out_h, out_w)


def sample_median_m(plane, nx: float, ny: float, half: int = 2,
                    min_valid: int = 5) -> Optional[float]:
    """Median depth in METRES in a (2*half+1)-square window around normalised
    (nx, ny) of ``plane`` (0 = unknown ignored); None with fewer than
    ``min_valid`` valid samples or a point outside the plane."""
    try:
        import numpy as np
        h, w = plane.shape[:2]
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            return None
        cx = min(w - 1, max(0, int(nx * w)))
        cy = min(h - 1, max(0, int(ny * h)))
        win = plane[max(0, cy - half):cy + half + 1, max(0, cx - half):cx + half + 1]
        vals = win[win > 0]
        if vals.size < min_valid:
            return None
        return float(np.median(vals)) / 1000.0
    except Exception:
        return None


# ── the segment: header + writer + reader ────────────────────────────────
def pack_header(seq: int, timestamp_ms: int, width: int = OUT_W,
                height: int = OUT_H, method: int = METHOD_NOMINAL,
                color_seq: int = 0, seq_begin: Optional[int] = None) -> bytes:
    head = struct.pack(_HEAD_FMT, MAGIC, seq, timestamp_ms, width, height, 2,
                       method, seq if seq_begin is None else seq_begin, color_seq)
    return head.ljust(HEADER_SIZE, bytes(1))


def unpack_header(buf) -> Optional[dict]:
    try:
        if len(buf) < HEADER_SIZE:
            return None
        (magic, seq, ts, w, h, bps, method, begin,
         color_seq) = struct.unpack_from(_HEAD_FMT, buf, 0)
    except Exception:
        return None
    if magic != MAGIC:
        return None
    return {"seq": seq, "timestamp_ms": ts, "width": w, "height": h,
            "bytes_per_sample": bps, "method": method, "seq_begin": begin,
            "color_seq": color_seq}


def read_depth(buf, last_seq: int = 0):
    """(seq, color_seq, timestamp_ms, plane) for a NEW whole plane in ``buf``,
    else None. ``plane`` is an owned (h, w) uint16 copy; a torn read is None."""
    import numpy as np
    hdr = unpack_header(buf)
    if hdr is None or hdr["seq"] == last_seq or hdr["seq"] == 0:
        return None
    w, h = hdr["width"], hdr["height"]
    if (not (0 < w <= MAX_W and 0 < h <= MAX_H) or hdr["bytes_per_sample"] != 2
            or hdr["seq_begin"] != hdr["seq"]):
        return None
    view = np.ndarray((h, w), dtype=np.uint16, buffer=buf, offset=HEADER_SIZE)
    plane = view.copy()
    del view
    hdr2 = unpack_header(buf)
    if (hdr2 is None or hdr2["seq"] != hdr["seq"]
            or hdr2["seq_begin"] != hdr["seq"]):
        return None
    return hdr["seq"], hdr["color_seq"], hdr["timestamp_ms"], plane


class DepthWriter:
    """Writer half of the depth segment. Never raises out of publish()."""

    def __init__(self, shm_factory=None, shm_name=DEPTH_SHM_NAME, wall=time.time):
        self._shm_factory = shm_factory
        self._shm_name = shm_name
        self._wall = wall
        self._shm = None
        self._view = None
        self._view_wh = (0, 0)
        self._seq = 0
        self.frames = 0
        self.last_error = ""

    def _open(self) -> None:
        if self._shm is not None:
            return
        if self._shm_factory is not None:
            shm = self._shm_factory(self._shm_name, TOTAL_SIZE)
        else:
            from multiprocessing import shared_memory
            try:
                shm = shared_memory.SharedMemory(name=self._shm_name, create=True,
                                                 size=TOTAL_SIZE)
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=self._shm_name)
                if shm.size < TOTAL_SIZE:
                    shm.close()
                    raise
        self._shm = shm
        hdr = unpack_header(shm.buf)
        self._seq = int(hdr["seq"]) if hdr else 0
        shm.buf[:HEADER_SIZE] = pack_header(self._seq, 0)

    def publish(self, plane, color_seq: int, method: int) -> bool:
        try:
            import numpy as np
            h, w = plane.shape[:2]
            if not (0 < w <= MAX_W and 0 < h <= MAX_H) or plane.dtype != np.uint16:
                return False
            self._open()
            buf = self._shm.buf
            if self._view_wh != (w, h):
                self._view = np.ndarray((h, w), dtype=np.uint16, buffer=buf,
                                        offset=HEADER_SIZE)
                self._view_wh = (w, h)
            nxt = self._seq + 1
            struct.pack_into("<Q", buf, OFF_SEQ_BEGIN, nxt)       # seqlock: announce
            self._view[:, :] = plane
            struct.pack_into("<II", buf, OFF_WH, w, h)
            struct.pack_into("<Q", buf, OFF_COLOR_SEQ, int(color_seq))
            struct.pack_into("<I", buf, 36, int(method))
            struct.pack_into("<QQ", buf, OFF_SEQ, nxt, int(self._wall() * 1000))
            self._seq = nxt
            self.frames += 1
            return True
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self.close()
            return False

    def close(self) -> None:
        view, shm = self._view, self._shm
        self._view = None
        self._view_wh = (0, 0)
        self._shm = None
        del view
        if shm is not None:
            for fn in (shm.close, shm.unlink):
                try:
                    fn()
                except Exception:
                    pass

    @property
    def is_open(self) -> bool:
        return self._shm is not None

    @property
    def seq(self) -> int:
        return self._seq


# ── the wire message ─────────────────────────────────────────────────────
def pack_wire(plane, color_seq: int, mirrored: bool = True,
              near_mm: int = NEAR_MM, far_mm: int = FAR_MM) -> bytes:
    """The binary WebSocket message for one plane (see the module docstring)."""
    h, w = plane.shape[:2]
    head = struct.pack(_WIRE_FMT, WIRE_MAGIC, WIRE_VERSION,
                       FLAG_MIRRORED if mirrored else 0, w, h, 0,
                       int(color_seq) & 0xFFFFFFFFFFFFFFFF, near_mm, far_mm)
    body = plane.astype("<u2", copy=False).tobytes()
    return head + body


def unpack_wire(data) -> Optional[dict]:
    """Inverse of pack_wire (tests; Python clients). None when not ours."""
    try:
        import numpy as np
        if len(data) < WIRE_HEADER:
            return None
        (magic, ver, flags, w, h, _res, seq, near,
         far) = struct.unpack_from(_WIRE_FMT, data, 0)
        if magic != WIRE_MAGIC or ver != WIRE_VERSION:
            return None
        if len(data) != WIRE_HEADER + w * h * 2:
            return None
        plane = np.frombuffer(data, dtype="<u2", offset=WIRE_HEADER).reshape(h, w)
        return {"width": w, "height": h, "mirrored": bool(flags & FLAG_MIRRORED),
                "seq": seq, "near_mm": near, "far_mm": far, "plane": plane}
    except Exception:
        return None


# ── the JARVIS-side feed (Kinect colour + depth -> the two segments) ─────
class KinectFeed:
    """Publishes the Kinect's colour (and registered depth) for the tracker.

    Collaborators are injectable so tests use fakes and never a Kinect:
      bridge     object with get_stream_health / frame_clock / get_color_bgr /
                 get_depth / get_color_space_mapper / get_depth_space_mapper
                 (default: audio.kinect_bridge, imported lazily)
      frames     the colour-segment writer (default core.handtrack_frames.WRITER)
      setting    () -> HAND_TRACK_CAMERA string
      depth_on   () -> HOLO_DEPTH_ENABLED bool
      session    () -> True while a Kinect hologram is up (default: none)
    """

    def __init__(self, bridge=None, frames=None, depth_writer=None, setting=None,
                 depth_on=None, clock=time.monotonic, session=None):
        self._session = session            # () -> a Kinect hologram is up
        self._bridge = bridge
        self._frames = frames
        self._setting = setting
        self._depth_on = depth_on
        self._clock = clock
        self.depth_writer = depth_writer if depth_writer is not None else DepthWriter()
        self._last_pub = -1e9
        self._color_clock = None
        self._backoff_until = 0.0
        self._calib: Optional[Calib] = None
        self._fit_after = 0.0
        self._inactive_since = 0.0
        self.color_frames = 0
        self.depth_frames = 0
        self.last_depth_at = 0.0
        self.last_error = ""
        self.last_reason = ""

    # -- settings ------------------------------------------------------
    def _cfg(self, name, default):
        try:
            from core import config
            return getattr(config, name, default)
        except Exception:
            return default

    def kinect_selected(self) -> bool:
        """HAND_TRACK_CAMERA names the Kinect, or a Kinect hologram is up."""
        s = self._setting() if self._setting else self._cfg("HAND_TRACK_CAMERA", "primary")
        if is_kinect_setting(s):
            return True
        return self._session is not None and bool(self._session())

    def depth_wanted(self) -> bool:
        """HOLO_DEPTH_ENABLED, or a Kinect hologram (which asked for depth)."""
        v = self._depth_on() if self._depth_on else self._cfg("HOLO_DEPTH_ENABLED", False)
        return bool(v) or (self._session is not None and bool(self._session()))

    def _bridge_mod(self):
        if self._bridge is None:
            from audio import kinect_bridge
            self._bridge = kinect_bridge
        return self._bridge

    def _frames_mod(self):
        if self._frames is None:
            from core import handtrack_frames
            self._frames = handtrack_frames.WRITER
        return self._frames

    # -- calibration ---------------------------------------------------
    def calibration(self, kb, now: float) -> Calib:
        if self._calib is not None and (self._calib.method == METHOD_SDK_FIT
                                        or now < self._fit_after):
            return self._calib
        self._fit_after = now + FIT_RETRY_S
        fitted = None
        try:
            cm, dm = kb.get_color_space_mapper(), kb.get_depth_space_mapper()
            if callable(cm) and callable(dm):
                fitted = fit_calibration(cm, dm)
        except Exception:
            fitted = None
        self._calib = fitted or self._calib or nominal_calibration()
        return self._calib

    # -- the step ------------------------------------------------------
    def step(self) -> str:
        """One publish attempt. Returns "" when a colour frame was published,
        else the reason it was not. Never raises."""
        try:
            return self._step()
        except Exception as e:                      # nothing may reach the face loop
            self.last_error = f"{type(e).__name__}: {e}"
            self._backoff_until = self._clock() + ERROR_BACKOFF_S
            self.last_reason = "error"
            return "error"

    def _step(self) -> str:
        if not self.kinect_selected():
            return self._note("not selected")
        wr = self._frames_mod()
        if not wr.active():
            self._idle_close()
            return self._note("tracker away")
        self._inactive_since = 0.0
        now = self._clock()
        if now < self._backoff_until:
            return self._note("backing off")
        if now - self._last_pub < MIN_GAP_S:
            return self._note("rate cap")
        kb = self._bridge_mod()
        health = kb.get_stream_health() or {}
        if not health.get("open") or health.get("enabled") is False:
            return self._note("kinect not open")
        clk = kb.frame_clock("color") if hasattr(kb, "frame_clock") else None
        if clk is not None and clk == self._color_clock:
            return self._note("no new colour")
        frame = kb.get_color_bgr(require_new=False)
        if frame is None:
            return self._note("no colour frame")
        if not wr.publish(frame):
            return self._note("writer declined")
        self._last_pub = now
        self._color_clock = clk
        self.color_frames += 1
        color_seq = int(wr.status().get("seq", 0))
        if self.depth_wanted():
            self._depth_step(kb, now, color_seq)
        return self._note("")

    def _depth_step(self, kb, now: float, color_seq: int) -> None:
        depth = kb.get_depth(require_new=True)
        if depth is None:
            return
        calib = self.calibration(kb, now)
        plane = register_depth(depth, calib)
        if self.depth_writer.publish(plane, color_seq, calib.method):
            self.depth_frames += 1
            self.last_depth_at = now

    def _idle_close(self) -> None:
        now = self._clock()
        if self._inactive_since == 0.0:
            self._inactive_since = now
        elif now - self._inactive_since > IDLE_CLOSE_S and self.depth_writer.is_open:
            self.depth_writer.close()

    def _note(self, reason: str) -> str:
        self.last_reason = reason
        return reason

    # -- honest status (the voice) --------------------------------------
    def depth_active(self, within_s: float = 2.0) -> bool:
        return (self.depth_frames > 0
                and self._clock() - self.last_depth_at <= within_s)

    def status(self) -> dict:
        c = self._calib
        return {"selected": self.kinect_selected(), "depth_wanted": self.depth_wanted(),
                "color_frames": self.color_frames, "depth_frames": self.depth_frames,
                "depth_active": self.depth_active(), "reason": self.last_reason,
                "last_error": self.last_error,
                "registration": (None if c is None else
                                 {"method": "sdk-fit" if c.method == METHOD_SDK_FIT
                                  else "nominal-fov", "residual_px": c.residual_px})}


def effective_camera_setting(setting: str) -> str:
    """HAND_TRACK_CAMERA as the frame publishers must see it: "kinect" while a
    Kinect hologram is up (so no webcam frame is mixed into the segment), else
    the configured value. Never raises."""
    try:
        return "kinect" if holo_session() else setting
    except Exception:
        return setting


def holo_session() -> bool:
    """True while a hologram that asked for the Kinect + depth is up
    (core.holo_web.STATE.kinect_session). Never raises."""
    try:
        from core import holo_web
        return bool(holo_web.STATE.kinect_session())
    except Exception:
        return False


FEED = KinectFeed(session=holo_session)


def publish_kinect() -> str:
    """The ONE call the face-tracking loop makes per pass. Never raises."""
    try:
        return FEED.step()
    except Exception:
        return "error"


def feed_status() -> dict:
    try:
        return FEED.status()
    except Exception:
        return {"selected": False, "depth_wanted": False, "depth_active": False}


def kinect_selected() -> bool:
    try:
        return FEED.kinect_selected()
    except Exception:
        return False
