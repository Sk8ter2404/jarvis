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
given.

Stdlib + numpy + cv2 (all already in JARVIS's environment; cv2 and numpy are
imported lazily so importing this module costs nothing). The tracker venv
imports this very file by path for the reader half, so the layout lives in
one place.
"""
from __future__ import annotations

import os
import struct
import time
from typing import Optional, Tuple

SHM_NAME = "jarvis_handtrack_frame"
MAGIC = b"JVHT0001"
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
        self._view = None                        # numpy (H, W, 3) view of the pixels
        self._seq = 0
        self._active = False
        self._next_check = 0.0
        self._last_pub = 0.0
        self._inactive_since = 0.0
        self._frames = 0
        self._last_error = ""

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
        return True

    def close(self) -> None:
        """Release the segment (the OS frees it once the tracker lets go too)."""
        view, shm = self._view, self._shm
        self._view = None
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
            if (w, h) != (WIDTH, HEIGHT):
                small = cv2.resize(frame, (WIDTH, HEIGHT),
                                   interpolation=cv2.INTER_AREA)
            else:
                small = frame
            buf = self._shm.buf
            ts = int(self._wall() * 1000)
            nxt = self._seq + 1
            # seqlock: announce the write, copy, then publish the new seq.
            struct.pack_into("<Q", buf, OFF_SEQ_BEGIN, nxt)
            self._view[:, :, :] = small
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
        return str(getattr(config, "HAND_TRACK_CAMERA", "primary") or "primary")
    except Exception:
        return "primary"


def publish_frame(cam: dict, frame, setting: Optional[str] = None) -> None:
    """The ONE call the camera loop makes per good frame: if the tracker is
    alive and ``cam`` is the camera HAND_TRACK_CAMERA names, hand it the
    frame. Never raises; one monotonic compare when nothing is listening."""
    try:
        if not WRITER.active():
            WRITER.publish(None)          # lets an idle segment close itself
            return
        which = _setting() if setting is None else setting
        side = ""
        if str(which).strip().lower() in ("left", "right"):
            from core import camera_tiles
            side = camera_tiles.percam_side(cam)
        if camera_matches(which, cam, side):
            WRITER.publish(frame)
    except Exception:
        pass


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
    if (w, h, c) != (WIDTH, HEIGHT, CHANNELS) or hdr["seq_begin"] != hdr["seq"]:
        return None
    view = np.ndarray((h, w, c), dtype=np.uint8, buffer=buf, offset=HEADER_SIZE)
    frame = view.copy()
    del view
    hdr2 = unpack_header(buf)
    if (hdr2 is None or hdr2["seq"] != hdr["seq"]
            or hdr2["seq_begin"] != hdr["seq"]):
        return None                       # the writer moved on mid-copy: torn
    return hdr["seq"], hdr["timestamp_ms"], frame
