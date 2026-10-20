"""Frame hand-off from JARVIS's camera loop to the hand-tracker service.

WHY
---
Windows lets one process open a webcam, and JARVIS already holds it. The
tracker (tools/hand_tracker_service.py) runs in its OWN venv (MediaPipe pulls
OpenCV 5, which must never enter JARVIS's environment), so it cannot open the
camera too. Instead the face-tracking loop PUBLISHES its frames here, into a
named ``multiprocessing.shared_memory`` segment the tracker attaches to.

THE SEGMENT  (name ``jarvis_handtrack_frame``; all little-endian)
-----------------------------------------------------------------
  offset  size  field
       0     8  magic            b"JVHT0001"
       8     8  seq              uint64, incremented AFTER the pixel copy
      16     8  timestamp_ms     uint64, wall clock (time.time()*1000)
      24     4  width            uint32 (640)
      28     4  height           uint32 (480)
      32     4  channels         uint32 (3, BGR, uint8, row-major)
      36     4  (reserved)
      40     8  seq_begin        uint64, set to seq+1 BEFORE the copy
      48    16  (reserved, zero)
      64   w*h*c pixel bytes

``seq_begin`` makes the header a seqlock: a reader that sees
``seq_begin == seq`` both before and after its copy got a whole frame. (The
spec only requires "seq after the copy"; the extra field is invisible to a
reader that ignores it.)

WHEN IT RUNS
------------
Only while the tracker is alive. The service touches ``data/.handtrack_alive``
about once a second; ``HandTrackFrames.active()`` is true while that file's
mtime is under ALIVE_MAX_AGE_S (3 s), re-checked at most every 0.5 s, so with
no tracker the per-frame cost is one monotonic-clock compare. Nothing is
allocated until the first active frame, and the segment is closed again after
IDLE_CLOSE_S without a tracker.

``publish_frame()`` is the single entry point the camera loop calls. It NEVER
raises and never touches the camera: it only reads the numpy frame it is
given. While the tracker is alive the loop also asks ``PUMP.serve()`` for its
camera's frames (the steady-frames reader, see FramePump): that is the only
code here that reads a capture handle, and only the one the loop hands it.

Stdlib + numpy + cv2 (all already in JARVIS's environment; cv2 and numpy are
imported lazily so importing this module costs nothing). The tracker venv
imports this very file by path for the reader half, so the layout lives in
one place.
"""
from __future__ import annotations

import os
import struct
import threading
import time
from typing import Callable, Optional, Tuple

SHM_NAME = "jarvis_handtrack_frame"
MAGIC = b"JVHT0001"
# The segment holds up to WIDTH x HEIGHT; each frame is scaled to FIT inside
# it keeping its own shape (a 16:9 webcam -> 640x360), and the header carries
# the real width/height. Stretching 16:9 into 4:3 squeezed every hand ~25 %
# narrower, which skews the finger angles the gestures are measured with.
WIDTH = 640
HEIGHT = 480
CHANNELS = 3
HEADER_SIZE = 64
FRAME_BYTES = WIDTH * HEIGHT * CHANNELS
TOTAL_SIZE = HEADER_SIZE + FRAME_BYTES

# magic, seq, timestamp_ms, width, height, channels, reserved, seq_begin
_HEAD_FMT = "<8sQQIIIIQ"
_HEAD_BYTES = struct.calcsize(_HEAD_FMT)          # 48
OFF_SEQ = 8
OFF_WH = 24                    # width, height (u32 each)
OFF_SEQ_BEGIN = 40

ALIVE_MAX_AGE_S = 3.0          # tracker heartbeat older than this = inactive
ALIVE_CHECK_EVERY_S = 0.5      # how often the heartbeat file is stat()ed
MIN_PUBLISH_GAP_S = 1.0 / 32   # cap the hand-off at ~30 fps
IDLE_CLOSE_S = 60.0            # close the segment this long after the tracker left

ALIVE_FILE_NAME = ".handtrack_alive"            # touched by the tracker
JARVIS_FILE_NAME = ".handtrack_jarvis_alive"    # touched by JARVIS (launcher)


def data_dir() -> str:
    """JARVIS's data/ folder (next to core/)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")


def alive_path() -> str:
    return os.path.join(data_dir(), ALIVE_FILE_NAME)


def jarvis_alive_path() -> str:
    return os.path.join(data_dir(), JARVIS_FILE_NAME)


def file_age_s(path: str, now: Optional[float] = None) -> Optional[float]:
    """Seconds since ``path`` was last modified, or None when absent."""
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return None
    return max(0.0, (time.time() if now is None else now) - mtime)


def pack_header(seq: int, timestamp_ms: int, width: int = WIDTH,
                height: int = HEIGHT, channels: int = CHANNELS,
                seq_begin: Optional[int] = None) -> bytes:
    head = struct.pack(_HEAD_FMT, MAGIC, seq, timestamp_ms, width, height,
                       channels, 0, seq if seq_begin is None else seq_begin)
    return head.ljust(HEADER_SIZE, bytes(1))


def unpack_header(buf) -> Optional[dict]:
    """The header of a segment buffer as a dict, or None when it is not ours."""
    try:
        if len(buf) < HEADER_SIZE:
            return None
        magic, seq, ts, w, h, ch, _res, begin = struct.unpack_from(_HEAD_FMT, buf, 0)
    except Exception:
        return None
    if magic != MAGIC:
        return None
    return {"seq": seq, "timestamp_ms": ts, "width": w, "height": h,
            "channels": ch, "seq_begin": begin}


def fit_size(w: int, h: int) -> Tuple[int, int]:
    """(width, height) of a ``w`` x ``h`` frame scaled to fit WIDTH x HEIGHT
    with its aspect kept; never enlarged; even numbers (JPEG-friendly)."""
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        return WIDTH, HEIGHT
    scale = min(WIDTH / w, HEIGHT / h, 1.0)
    tw = max(2, min(WIDTH, int(round(w * scale)) // 2 * 2))
    th = max(2, min(HEIGHT, int(round(h * scale)) // 2 * 2))
    return tw, th


def _frame_matches(frame) -> bool:
    try:
        return (frame is not None and frame.ndim == 3 and frame.shape[2] == CHANNELS
                and frame.shape[0] > 0 and frame.shape[1] > 0)
    except Exception:
        return False


def camera_matches(setting: str, cam: dict, side: str = "") -> bool:
    """Does the CAMERAS entry ``cam`` match the HAND_TRACK_CAMERA setting?

    "" / "primary" -> the entry flagged primary; "left" / "right" -> the
    preview key of the entry (core.camera_tiles.percam_side); a digit string
    -> the camera index. Anything else (e.g. "kinect", not published by this
    loop) matches nothing, so the status action can say so honestly.
    "kinect" is published by core/handtrack_depth.py (Phase 2) from the Kinect
    bridge's own colour frames, never from a CAMERAS entry, so it still matches
    no webcam here - which is what keeps a webcam frame out of that segment.
    """
    key = str(setting or "primary").strip().lower()
    try:
        if key in ("", "primary", "default", "webcam"):
            return bool(cam.get("primary"))
        if key in ("left", "right"):
            return (side or "").lower() == key
        if key.isdigit():
            return int(cam.get("index", -1)) == int(key)
    except Exception:
        return False
    return False


class HandTrackFrames:
    """Writer half. One instance (``WRITER``) lives in the JARVIS process."""

    def __init__(self, clock=time.monotonic, wall=time.time, alive_file=None,
                 shm_factory=None, shm_name=SHM_NAME):
        self._clock = clock
        self._wall = wall
        self._alive_file = alive_file or alive_path()
        self._shm_factory = shm_factory          # tests: a fake segment type
        self._shm_name = shm_name
        self._shm = None
        self._view = None                        # numpy view of the pixels (current size)
        self._view_wh = (0, 0)
        self._seq = 0
        self._active = False
        self._next_check = 0.0
        self._last_pub = 0.0
        self._inactive_since = 0.0
        self._frames = 0
        self._last_error = ""
        # publish() can be called from the face loop AND from the frame pump's
        # reader thread (FramePump below): one writer at a time, so two copies
        # can never interleave inside the seqlock.
        self._pub_lock = threading.Lock()

    # ── is anyone listening? ────────────────────────────────────────────
    def active(self) -> bool:
        """True while the tracker's heartbeat file is fresh. Cheap: the file
        is stat()ed at most every ALIVE_CHECK_EVERY_S."""
        now = self._clock()
        if now < self._next_check:
            return self._active
        self._next_check = now + ALIVE_CHECK_EVERY_S
        age = file_age_s(self._alive_file, self._wall())
        self._active = age is not None and age < ALIVE_MAX_AGE_S
        if self._active:
            self._inactive_since = 0.0
        elif self._inactive_since == 0.0:
            self._inactive_since = now
        return self._active

    # ── the segment ─────────────────────────────────────────────────────
    def _open(self) -> bool:
        if self._view is not None:
            return True
        import numpy as np
        if self._shm_factory is not None:
            shm = self._shm_factory(self._shm_name, TOTAL_SIZE)
        else:
            from multiprocessing import shared_memory
            try:
                shm = shared_memory.SharedMemory(name=self._shm_name, create=True,
                                                 size=TOTAL_SIZE)
            except FileExistsError:
                # A segment a previous JARVIS left (or a test fixture): reuse
                # it when it is big enough, else refuse rather than guess.
                shm = shared_memory.SharedMemory(name=self._shm_name)
                if shm.size < TOTAL_SIZE:
                    shm.close()
                    raise
        self._shm = shm
        buf = shm.buf
        hdr = unpack_header(buf)
        self._seq = int(hdr["seq"]) if hdr else 0
        buf[:HEADER_SIZE] = pack_header(self._seq, 0)
        self._view = np.ndarray((HEIGHT, WIDTH, CHANNELS), dtype=np.uint8,
                                buffer=buf, offset=HEADER_SIZE)
        self._view_wh = (WIDTH, HEIGHT)
        return True

    def _view_for(self, tw: int, th: int):
        """A (th, tw, 3) view at the start of the pixel area (cached by size)."""
        if self._view_wh != (tw, th):
            import numpy as np
            self._view = np.ndarray((th, tw, CHANNELS), dtype=np.uint8,
                                    buffer=self._shm.buf, offset=HEADER_SIZE)
            self._view_wh = (tw, th)
        return self._view

    def close(self) -> None:
        """Release the segment (the OS frees it once the tracker lets go too)."""
        view, shm = self._view, self._shm
        self._view = None
        self._view_wh = (0, 0)
        self._shm = None
        del view
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass

    # ── the hot path ────────────────────────────────────────────────────
    def publish(self, frame) -> bool:
        """Publish ``frame`` (BGR uint8, any size) if the tracker is alive.
        Returns True when a frame was written. Never raises."""
        with self._pub_lock:
            return self._publish(frame)

    def _publish(self, frame) -> bool:
        try:
            if not self.active():
                if (self._shm is not None and self._inactive_since
                        and self._clock() - self._inactive_since > IDLE_CLOSE_S):
                    self.close()
                return False
            now = self._clock()
            if now - self._last_pub < MIN_PUBLISH_GAP_S:
                return False
            if not _frame_matches(frame):
                return False
            self._open()
            import cv2
            h, w = frame.shape[:2]
            tw, th = fit_size(w, h)
            if (w, h) != (tw, th):
                small = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            else:
                small = frame
            buf = self._shm.buf
            ts = int(self._wall() * 1000)
            nxt = self._seq + 1
            # seqlock: announce the write, copy pixels + size, then publish the
            # new seq (a reader that sees seq_begin != seq skips the frame).
            struct.pack_into("<Q", buf, OFF_SEQ_BEGIN, nxt)
            self._view_for(tw, th)[:, :, :] = small
            struct.pack_into("<II", buf, OFF_WH, tw, th)
            struct.pack_into("<QQ", buf, OFF_SEQ, nxt, ts)   # seq then timestamp
            self._seq = nxt
            self._last_pub = now
            self._frames += 1
            return True
        except Exception as e:                    # the camera path must never see this
            self._last_error = f"{type(e).__name__}: {e}"
            try:
                self.close()
            except Exception:
                pass
            self._next_check = self._clock() + 5.0   # back off after a failure
            return False

    def status(self) -> dict:
        return {"active": self._active, "open": self._shm is not None,
                "frames": self._frames, "seq": self._seq,
                "last_error": self._last_error}


WRITER = HandTrackFrames()


def _setting() -> str:
    try:
        from core import config
        cfg = str(getattr(config, "HAND_TRACK_CAMERA", "primary") or "primary")
    except Exception:
        return "primary"
    try:
        # A Kinect hologram (voice: "...with the Kinect") borrows the tracker's
        # camera for as long as it is up: no webcam frame may be mixed in.
        from core import handtrack_depth
        return handtrack_depth.effective_camera_setting(cfg)
    except Exception:
        return cfg


def is_hand_camera(cam: dict, setting: Optional[str] = None) -> bool:
    """True when ``cam`` is the camera HAND_TRACK_CAMERA names."""
    which = _setting() if setting is None else setting
    side = ""
    if str(which).strip().lower() in ("left", "right"):
        from core import camera_tiles
        side = camera_tiles.percam_side(cam)
    return camera_matches(which, cam, side)


def publish_frame(cam: dict, frame, setting: Optional[str] = None) -> None:
    """The ONE call the camera loop makes per good frame: if the tracker is
    alive and ``cam`` is the camera HAND_TRACK_CAMERA names, hand it the
    frame. Never raises; one monotonic compare when nothing is listening."""
    try:
        if not WRITER.active():
            WRITER.publish(None)          # lets an idle segment close itself
            return
        if is_hand_camera(cam, setting):
            WRITER.publish(frame)
    except Exception:
        pass


# ── burst reads (2026-10-14) ────────────────────────────────────────────
# Live on v2.0.194 the tracker got 5.8 fps: the face loop's own pace (one
# read, a Haar face pass and a 50 ms nap per pass) on a ~30 fps camera. A
# point-flick needs 3+ samples inside 0.18 s, so it could never fire. While
# the tracker is alive, the loop now keeps reading THE TRACKER'S camera for
# BURST_S right after its normal read - same thread, same capture handle, so
# never a second reader - and hands every frame over. Face detection then runs
# once on the newest frame, so face tracking keeps (about) its own cadence.
BURST_S = 0.15


def burst(cap, cam: dict, frame, beat=None, setting: Optional[str] = None,
          budget_s: float = BURST_S, clock=time.monotonic):
    """(newest_frame, extra_frames). While the tracker is alive and ``cam``
    is its camera, read ``cap`` for up to ``budget_s`` and publish each good
    frame. Stops at the first failed read (the loop's normal failure handling
    sees the camera on its next pass). With no tracker: one cheap check and
    ``(frame, 0)``. Never raises."""
    n = 0
    try:
        if cap is None or not WRITER.active() or not is_hand_camera(cam, setting):
            return frame, 0
        end = clock() + budget_s
        while clock() < end:
            if beat is not None:
                beat()
            ret, f = cap.read()
            if not ret or not _frame_matches(f):
                break
            WRITER.publish(f)
            frame = f
            n += 1
    except Exception:
        pass
    return frame, n


# ── steady frames: the frame pump (2026-10-20) ──────────────────────────
# Live on the Dell (v2.0.226) the burst above still left the hand cursor
# laggy: the tracker got ~18 frames/s, but in CLUMPS - ~0.15 s of reads, then
# a hole while the loop ran face detection, previews and the other cameras.
# The pump replaces the burst while the tracker is alive: ONE daemon thread
# reads the tracker's camera at the camera's own rate and publishes every
# frame, and the face loop takes the newest frame from it INSTEAD of calling
# cap.read() itself. Face detection keeps its own (slower) cadence.
#
# THE HANDLE DISCIPLINE (Windows MSMF: one handle per device, owned by the
# face loop) is unchanged:
#   * the pump never opens and never releases a capture. It only reads the
#     handle the face loop passes to serve(), and only from its one thread;
#   * while the reader runs on a handle, the face loop does NOT read it
#     (serve() returning None is the promise that no reader is on it);
#   * every release goes through detach() first (the monolith's
#     _release_capture_guarded calls it), which stops the reader and waits for
#     its in-flight read() to return - so a handle is never released while
#     another thread is inside read() on it;
#   * the reader takes no lock, exactly like the face loop's own reads (a
#     blocking read under _camera_io_lock is the documented whole-subsystem
#     hang).
# With no tracker, serve() is one attribute test plus WRITER.active() (one
# monotonic compare) and returns None: the loop reads exactly as before.
PUMP_WAIT_S = 1.0               # the face loop waits at most this for a NEW frame
PUMP_REARM_S = 1.0              # after the reader's failed read, the loop reads itself this long
PUMP_MIN_READ_GAP_S = 1.0 / 60  # backstop for a capture whose read() never blocks
PUMP_JOIN_WARN_S = 2.0          # detach() says so when a read keeps it waiting this long


class FramePump:
    """The paced reader for the tracker's camera. One instance (``PUMP``)."""

    def __init__(self, writer=None, clock=time.monotonic, log=None):
        self._writer = writer          # None = the module's WRITER, looked up per call
        self._clock = clock
        self._log = log or (lambda s: print(f"  [handtrack] {s}"))
        self._cond = threading.Condition(threading.Lock())
        self._cap = None               # the handle the reader is bound to
        self._thread: Optional[threading.Thread] = None
        self._stop: Optional[threading.Event] = None
        self._alive = False            # reader thread still looping
        self._failed = False           # it ended on a failed read
        self._frame = None
        self._seq = 0                  # frames read on the current binding
        self._served = 0               # newest seq handed to the face loop
        self._rearm_at = 0.0
        self.reads = 0                 # lifetime counters (status / tests)
        self.starts = 0
        # Optional hooks, set once by the monolith:
        #   accept(frame) -> bool   publish this frame to the tracker? (black check)
        #   on_frame(cam, frame)    runs on the reader thread for every good frame
        #                           (the HUD preview follows the steady frames)
        self.accept: Optional[Callable] = None
        self.on_frame: Optional[Callable] = None

    def _w(self):
        return self._writer if self._writer is not None else WRITER

    def bound(self) -> bool:
        return self._cap is not None

    # ── the face loop's side ────────────────────────────────────────────
    def serve(self, cap, cam: dict, setting: Optional[str] = None,
              pumpable: bool = True):
        """Called by the face loop INSTEAD of ``cap.read()``.

        None  -> the pump does not serve this read: the caller reads ``cap``
                 itself, exactly as before (and no reader is on ``cap``).
        (ret, frame) -> the newest frame the reader got since the last call,
                 or (False, None) when it failed / none came in PUMP_WAIT_S.
        Never raises."""
        try:
            if self._cap is None and (cap is None or not self._w().active()):
                return None                       # no tracker: the common path
            alive = self._w().active()
            want = (cap is not None and pumpable and alive
                    and is_hand_camera(cam, setting))
            if not want:
                if self._cap is not None and (self._cap is cap or not alive):
                    self.detach(self._cap)
                return None
            if self._cap is not None and self._cap is not cap:
                self.detach(self._cap)            # the loop reopened this camera
            if self._cap is None:
                if self._clock() < self._rearm_at:
                    return None                   # after a failure: the loop reads
                self._start(cap, cam)
            return self._next()
        except Exception:
            try:
                self.detach(cap)
            except Exception:
                pass
            return None

    def _start(self, cap, cam) -> None:
        stop = threading.Event()
        with self._cond:
            self._cap, self._stop = cap, stop
            self._alive, self._failed = True, False
            self._frame, self._seq, self._served = None, 0, 0
        t = threading.Thread(target=self._run, args=(cap, cam, stop),
                             name="handtrack-frame-pump", daemon=True)
        self._thread = t
        self.starts += 1
        t.start()

    def _next(self):
        # Real time on purpose (the waits are real): self._clock only decides
        # the re-arm back-off, which tests drive with a fake clock.
        deadline = time.monotonic() + PUMP_WAIT_S
        with self._cond:
            while self._seq <= self._served and self._alive:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False, None            # nothing new in PUMP_WAIT_S
                self._cond.wait(min(left, 0.1))
            if self._seq > self._served:
                self._served = self._seq
                return True, self._frame
            failed = self._failed
        # The reader has ended (a failed read): hand the loop that failure once,
        # then let it read the camera itself for PUMP_REARM_S.
        self.detach(self._cap)
        if failed:
            self._rearm_at = self._clock() + PUMP_REARM_S
        return False, None

    def detach(self, cap=None) -> bool:
        """Stop the reader (only if it is on ``cap``; None = whatever it is on)
        and wait for its in-flight read to return. True when it was bound.
        Must run before the handle is released. Never raises."""
        try:
            with self._cond:
                if self._cap is None or (cap is not None and self._cap is not cap):
                    return False
                t, stop = self._thread, self._stop
            if stop is not None:
                stop.set()
            if t is not None and t is not threading.current_thread():
                t0 = time.monotonic()
                warned = False
                while t.is_alive():
                    t.join(0.25)
                    if not warned and time.monotonic() - t0 > PUMP_JOIN_WARN_S:
                        warned = True
                        self._log("waiting for the frame pump's camera read to "
                                  "return before the camera is let go")
            with self._cond:
                self._cap = None
                self._thread = None
                self._stop = None
                self._alive = False
                self._frame = None
            return True
        except Exception:
            return False

    # ── the reader thread ───────────────────────────────────────────────
    def _run(self, cap, cam, stop) -> None:
        failed = False
        try:
            while not stop.is_set():
                t0 = time.monotonic()
                try:
                    ret, f = cap.read()
                except Exception:
                    ret, f = False, None
                self.reads += 1
                if stop.is_set():
                    break
                if not ret or not _frame_matches(f):
                    failed = True
                    break
                with self._cond:
                    self._frame = f
                    self._seq += 1
                    self._cond.notify_all()
                try:
                    if self.accept is None or self.accept(f):
                        self._w().publish(f)
                except Exception:
                    pass
                hook = self.on_frame
                if hook is not None:
                    try:
                        hook(cam, f)
                    except Exception:
                        pass
                gap = time.monotonic() - t0
                if gap < PUMP_MIN_READ_GAP_S:
                    stop.wait(PUMP_MIN_READ_GAP_S - gap)
        finally:
            with self._cond:
                self._alive = False
                self._failed = failed
                self._cond.notify_all()

    def status(self) -> dict:
        return {"bound": self._cap is not None, "alive": self._alive,
                "reads": self.reads, "starts": self.starts,
                "failed": self._failed}


PUMP = FramePump()


# ── reader half (the tracker service; also used by tests) ───────────────
def read_frame(buf, last_seq: int = 0) -> Optional[Tuple[int, int, "object"]]:
    """(seq, timestamp_ms, frame) for a NEW whole frame in ``buf``, else None.

    ``frame`` is an owned (H, W, 3) uint8 copy. A frame being written (or
    torn mid-copy) is skipped: the caller simply tries again.
    """
    import numpy as np
    hdr = unpack_header(buf)
    if hdr is None or hdr["seq"] == last_seq or hdr["seq"] == 0:
        return None
    w, h, c = hdr["width"], hdr["height"], hdr["channels"]
    if (not (0 < w <= WIDTH and 0 < h <= HEIGHT) or c != CHANNELS
            or hdr["seq_begin"] != hdr["seq"]):
        return None
    view = np.ndarray((h, w, c), dtype=np.uint8, buffer=buf, offset=HEADER_SIZE)
    frame = view.copy()
    del view
    hdr2 = unpack_header(buf)
    if (hdr2 is None or hdr2["seq"] != hdr["seq"]
            or hdr2["seq_begin"] != hdr["seq"]):
        return None                       # the writer moved on mid-copy: torn
    return hdr["seq"], hdr["timestamp_ms"], frame
