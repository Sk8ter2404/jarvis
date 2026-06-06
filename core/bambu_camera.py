"""
Bambu printer chamber-camera grabber for JARVIS.

Pulls frames from the printer's built-in camera over the LOCAL LAN and
writes the most recent JPEG to ``data/bambu_camera_frame.jpg`` (plus a
tiny status sidecar ``data/bambu_camera_state.json``) so the HUD widget
``hud/bambu_camera_hud.py`` can render it without ever touching the
network itself. The HUD is a pure *view*; this module is the only thing
that talks to the printer's camera — mirroring how skills/bambu_monitor.py
owns the MQTT side and the HUD reads bambu_overlay_state.json.

Two access paths, tried most-reliable-first for the configured printer:

  1. RTSPS (X1 / H2D / H2S / P2 class) — an authenticated, TLS-wrapped
     RTSP stream on port 322:

         rtsps://bblp:<access_code>@<ip>:322/streaming/live/1

     Decoded with OpenCV's FFmpeg backend. This is the H2D's path. It
     requires "LAN Only Liveview" enabled on the printer's screen
     (Settings -> LAN Only; the toggle that also reveals the Access
     Code). NOTE: on some firmware/region combinations Bambu has
     temporarily disabled H2D/H2S/P2S live streaming — when that's the
     case the connect simply fails and we fall through to path 2.

  2. JPEG-over-TLS stills (P1 / A1 class, and a graceful fallback for
     the X-class when RTSPS is unavailable) — a raw TLS socket to port
     6000. We send the 104-byte Bambu auth packet (username ``bblp`` +
     access code, both null-padded to 32 bytes) and then scan the
     byte-stream for JPEG frames (SOI ``ff d8 ff e0`` .. EOI ``ff d9``).
     The printer pushes a frame roughly once a second.

If neither path yields a frame the grabber records the failure in the
status sidecar; the HUD then shows a "camera offline" placeholder over
the existing print-status readout, so the feature degrades cleanly
instead of going blank.

Dependencies are all OPTIONAL and lazily probed so this module imports
on the CI runner (where ``cv2`` is intentionally not installed) and on a
box without the camera deps:
  • cv2 (opencv-python)  — only needed for the RTSPS path.
  • numpy + PIL          — only needed to re-encode an RTSPS frame to JPEG.
The JPEG-stills path needs only the stdlib (``ssl`` + ``socket``), so it
works even when cv2 is absent.

Public API:
  start_grabber()  -> bool   # idempotent; spins a daemon poll thread
  stop_grabber()   -> None
  is_running()     -> bool
  grab_once(...)   -> bytes|None   # one-shot fetch (used by tests/manual)
  camera_supported_for_model(model) -> str   # 'rtsps' | 'jpeg' | 'unknown'

Config (via bobert_companion, which re-exports core/config.py):
  HUD_BAMBU_CAMERA  — master enable (default True). When False the
                      grabber start() is a no-op.
  BAMBU_PRINTER_IP / BAMBU_ACCESS_CODE / BAMBU_SERIAL — same creds the
                      MQTT monitor uses (env/.env only — never committed).
"""
from __future__ import annotations

import json
import os
import socket
import ssl
import struct
import threading
import time

# ── optional deps (lazily flagged; never crash the import) ───────────────
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover - cv2 is intentionally absent on CI
    cv2 = None  # type: ignore
    _HAS_CV2 = False

try:
    import numpy as _np  # type: ignore
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - numpy present on dev + CI, defensive only
    _np = None  # type: ignore
    _HAS_NUMPY = False


_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_DIR, "data")
# The HUD reads these two. Both live under data/ which is gitignored, so a
# captured frame of the user's workshop is never committed.
FRAME_FILE = os.path.join(_DATA_DIR, "bambu_camera_frame.jpg")
STATE_FILE = os.path.join(_DATA_DIR, "bambu_camera_state.json")

# ── camera protocol constants ────────────────────────────────────────────
CAMERA_USER = "bblp"
RTSPS_PORT = 322
RTSPS_PATH = "streaming/live/1"
JPEG_STILL_PORT = 6000
# JPEG frame delimiters the printer emits on the port-6000 stream.
_JPEG_SOI = b"\xff\xd8\xff\xe0"
_JPEG_EOI = b"\xff\xd9"

# Poll cadence. The camera refreshes ~1 fps on the stills path; the RTSPS
# path can do far more but the HUD only repaints a few times a second, so
# grabbing one frame per second keeps CPU + LAN traffic modest while still
# feeling live. Tunable via JARVIS_BAMBU_CAMERA_FPS for power users.
try:
    _TARGET_FPS = float(os.getenv("JARVIS_BAMBU_CAMERA_FPS", "2.0"))
except (TypeError, ValueError):
    _TARGET_FPS = 2.0
if _TARGET_FPS <= 0:
    _TARGET_FPS = 2.0
POLL_INTERVAL_SECONDS = max(0.25, 1.0 / _TARGET_FPS)

# How long a single connection attempt may block before we treat the printer
# as unreachable and back off. Kept short so a powered-down printer doesn't
# stall the poll thread.
CONNECT_TIMEOUT_SECONDS = 5.0
# After a failed cycle we back off to this slower cadence so we're not
# hammering a sleeping printer every POLL_INTERVAL_SECONDS. A success drops
# us straight back to the fast cadence.
OFFLINE_POLL_INTERVAL_SECONDS = 15.0
# A frame older than this (seconds) is considered stale by readers.
FRAME_STALE_SECONDS = 20.0

# Models whose camera is reached over RTSPS (port 322) vs the JPEG stills
# socket (port 6000). Matched case-insensitively as a substring of whatever
# devmodel string we can see, so "C11" / "X1C" / "H2D" all resolve.
_RTSPS_MODEL_HINTS = ("x1", "h2d", "h2s", "p2", "x1c", "x1e", "x1-")
_JPEG_MODEL_HINTS = ("p1", "a1", "p1p", "p1s", "a1m", "a1mini")

# ── module state ─────────────────────────────────────────────────────────
_grab_thread: list = [None]
_stop_evt = threading.Event()
_state_lock = threading.Lock()
_last_status: dict = {
    "ok": False,
    "path": None,        # 'rtsps' | 'jpeg' | None
    "model_hint": None,
    "last_frame_at": 0.0,
    "last_error": None,
    "frame_count": 0,
}


# ── config bridge ────────────────────────────────────────────────────────
def _read_config() -> tuple[str, str, str]:
    """Pull printer IP / access code / serial from bobert_companion at call
    time (it re-exports core/config.py). Returns ('','','') on any failure
    so callers treat the camera as simply unconfigured."""
    try:
        import importlib
        bc = importlib.import_module("bobert_companion")
        ip = (getattr(bc, "BAMBU_PRINTER_IP", "") or "").strip()
        access = (getattr(bc, "BAMBU_ACCESS_CODE", "") or "").strip()
        serial = (getattr(bc, "BAMBU_SERIAL", "") or "").strip()
        return ip, access, serial
    except Exception:
        # Fall back to the raw env so the grabber still works in a thin
        # context (e.g. a manual `python -m core.bambu_camera` run) where
        # the monolith isn't importable.
        return (
            (os.getenv("BAMBU_PRINTER_IP", "") or "").strip(),
            (os.getenv("BAMBU_ACCESS_CODE", "") or "").strip(),
            (os.getenv("BAMBU_SERIAL", "") or "").strip(),
        )


def _camera_enabled() -> bool:
    """Master HUD_BAMBU_CAMERA flag (default True). Read lazily so the module
    imports cleanly before bobert_companion finishes initialising."""
    try:
        import importlib
        bc = importlib.import_module("bobert_companion")
        return bool(getattr(bc, "HUD_BAMBU_CAMERA", True))
    except Exception:
        return True


def _read_model_hint() -> str:
    """Best-effort printer model string, used only to pick the camera path.

    bambu_monitor doesn't surface the device model, so we read whatever the
    setup wizard may have sniffed into bambu_overlay_state.json, then fall
    back to the BAMBU_SERIAL prefix. Returns '' when nothing is known — the
    caller then just tries RTSPS first, JPEG second, which is correct for
    every model anyway."""
    # 1) Overlay state (bambu_monitor writes it; may carry a 'model' someday).
    try:
        ov = os.path.join(_PROJECT_DIR, "bambu_overlay_state.json")
        if os.path.exists(ov):
            with open(ov, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
            m = (d.get("model") or d.get("printer_model") or "").strip()
            if m:
                return m
    except Exception:
        pass
    # 2) Serial prefix. Bambu serials encode the model family in the first
    #    few chars (e.g. '094...' H2D, '00M...' X1C, '01S...' P1S). We don't
    #    decode it precisely — just return it so substring hints can match.
    _ip, _access, serial = _read_config()
    return serial or ""


def camera_supported_for_model(model: str) -> str:
    """Classify which local camera path a model uses.

    Returns 'rtsps' (X1/H2D/H2S/P2 — port 322), 'jpeg' (P1/A1 — port 6000),
    or 'unknown' when the hint doesn't match either family. 'unknown' is not
    a failure: the grabber tries RTSPS then JPEG regardless, so an unknown
    model still gets a feed if either path answers."""
    m = (model or "").lower()
    if not m:
        return "unknown"
    for h in _RTSPS_MODEL_HINTS:
        if h in m:
            return "rtsps"
    for h in _JPEG_MODEL_HINTS:
        if h in m:
            return "jpeg"
    return "unknown"


# ── path ordering ────────────────────────────────────────────────────────
def _ordered_paths(model_hint: str) -> list[str]:
    """Decide which camera path(s) to try, in order, for this printer.

    RTSPS is the richer feed so it leads for X-class. For an unknown model
    we still lead with RTSPS (the H2D is the configured printer) but keep
    JPEG as the fallback. The JPEG path is stdlib-only, so it's also the
    path that works when cv2 isn't installed."""
    kind = camera_supported_for_model(model_hint)
    if kind == "jpeg":
        order = ["jpeg", "rtsps"]
    else:  # 'rtsps' or 'unknown'
        order = ["rtsps", "jpeg"]
    # Drop RTSPS entirely when cv2 is missing — it can't decode the stream.
    if not _HAS_CV2:
        order = [p for p in order if p != "rtsps"]
        if not order:
            order = ["jpeg"]
    return order


def build_rtsps_url(ip: str, access: str) -> str:
    """Compose the authenticated RTSPS URL for the X1/H2D camera."""
    return f"rtsps://{CAMERA_USER}:{access}@{ip}:{RTSPS_PORT}/{RTSPS_PATH}"


# ── auth packet (port-6000 JPEG stills path) ─────────────────────────────
def build_auth_packet(username: str, access_code: str) -> bytes:
    """Build the 104-byte Bambu camera auth packet for the port-6000 stills
    protocol.

    Layout (little-endian): 0x40, 0x3000, 0, 0 as four uint32 (16 bytes),
    then the username ASCII null-padded to 32 bytes, then the access code
    ASCII null-padded to 32 bytes. Total 16 + 32 + 32 = 80 bytes.

    (The reference clients pad both credential fields to 32 bytes; the
    header is 16 bytes, so the packet is 80 bytes for the common 'bblp' +
    8-digit-code case. We build it generically from the field lengths so an
    unusual access-code length still produces a valid packet.)
    """
    u = username.encode("ascii", errors="ignore")[:32]
    c = access_code.encode("ascii", errors="ignore")[:32]
    pkt = bytearray()
    pkt += struct.pack("<I", 0x40)
    pkt += struct.pack("<I", 0x3000)
    pkt += struct.pack("<I", 0x00)
    pkt += struct.pack("<I", 0x00)
    pkt += u + (b"\x00" * (32 - len(u)))
    pkt += c + (b"\x00" * (32 - len(c)))
    return bytes(pkt)


def extract_jpeg(buffer: bytes) -> tuple[bytes | None, bytes]:
    """Pull one complete JPEG out of a streaming byte buffer.

    Returns (frame_or_None, remaining_buffer). When a full SOI..EOI frame is
    present the frame bytes (inclusive of both markers) are returned and the
    buffer is advanced past the frame. When no complete frame is present yet
    the frame is None and the buffer is returned trimmed of any leading junk
    before the next SOI (so it can't grow unbounded on a desync)."""
    soi = buffer.find(_JPEG_SOI)
    if soi < 0:
        # No start marker at all — keep only a small tail in case the marker
        # is split across the read boundary.
        return None, buffer[-3:] if len(buffer) > 3 else buffer
    if soi > 0:
        buffer = buffer[soi:]
    eoi = buffer.find(_JPEG_EOI, len(_JPEG_SOI))
    if eoi < 0:
        return None, buffer
    end = eoi + len(_JPEG_EOI)
    return buffer[:end], buffer[end:]


# ── status sidecar ───────────────────────────────────────────────────────
def _write_status(**updates) -> None:
    """Atomic-write the camera status sidecar the HUD polls. Best-effort —
    a write failure never takes down the poll loop."""
    with _state_lock:
        _last_status.update(updates)
        snapshot = dict(_last_status)
    snapshot["written_at"] = time.time()
    snapshot["has_cv2"] = _HAS_CV2
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)
        os.replace(tmp, STATE_FILE)
    except Exception:
        pass


def _write_frame(jpeg: bytes) -> bool:
    """Atomic-write the latest JPEG frame for the HUD. Returns True on success."""
    if not jpeg:
        return False
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        tmp = FRAME_FILE + ".tmp"
        with open(tmp, "wb") as f:
            f.write(jpeg)
        os.replace(tmp, FRAME_FILE)
        return True
    except Exception:
        return False


def get_status() -> dict:
    """Return a copy of the current grabber status (for tests / status action)."""
    with _state_lock:
        return dict(_last_status)


# ── RTSPS grab (cv2) ─────────────────────────────────────────────────────
def _grab_rtsps_once(ip: str, access: str) -> bytes | None:
    """Open the RTSPS stream, read one decoded frame, re-encode it to JPEG.
    Returns JPEG bytes or None on any failure. cv2-only."""
    if not _HAS_CV2:
        return None
    url = build_rtsps_url(ip, access)
    # FFmpeg options for OpenCV's capture backend:
    #   • rtsp_transport;tcp  — the bblp TLS proxy is TCP; UDP fails.
    #   • stimeout (microsec) — bound the connect/read so a dead printer
    #     doesn't hang the worker thread.
    # OpenCV reads these from this env var when it constructs the VideoCapture.
    prev_opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;"
        + str(int(CONNECT_TIMEOUT_SECONDS * 1_000_000))
    )
    cap = None
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        # Keep the internal buffer tiny so we read a *fresh* frame, not a
        # stale one queued during connect.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        ok2, buf = cv2.imencode(".jpg", frame)
        if not ok2:
            return None
        return bytes(buf.tobytes())
    except Exception:
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        # Restore the env so we don't leak capture options into any other
        # cv2 user in the process.
        if prev_opts is None:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev_opts


# ── JPEG-stills grab (stdlib ssl/socket) ─────────────────────────────────
def _grab_jpeg_once(ip: str, access: str,
                    timeout: float = CONNECT_TIMEOUT_SECONDS) -> bytes | None:
    """Connect to the port-6000 camera socket, authenticate, and return the
    first complete JPEG frame. Pure stdlib (works without cv2). Returns None
    on any failure."""
    ctx = ssl.create_default_context()
    # The printer presents a self-signed cert for its own LAN IP; we're
    # connecting to a known device on the local network by IP, so disable
    # hostname/CA verification (exactly what every Bambu client does here).
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    raw = None
    sconn = None
    try:
        raw = socket.create_connection((ip, JPEG_STILL_PORT), timeout=timeout)
        sconn = ctx.wrap_socket(raw, server_hostname=ip)
        sconn.settimeout(timeout)
        sconn.sendall(build_auth_packet(CAMERA_USER, access))
        buffer = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = sconn.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            buffer += chunk
            frame, buffer = extract_jpeg(buffer)
            if frame is not None:
                return frame
            # Guard against an unbounded buffer if the markers never align.
            if len(buffer) > 4_000_000:
                break
        return None
    except Exception:
        return None
    finally:
        for s in (sconn, raw):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass


# ── one-shot + poll loop ─────────────────────────────────────────────────
def grab_once(ip: str | None = None, access: str | None = None,
              model_hint: str | None = None) -> bytes | None:
    """Fetch a single JPEG frame using the best available path for this
    printer. Returns the JPEG bytes (also written to FRAME_FILE) or None.

    Used by the poll loop and exposed for manual / test use. Updates the
    status sidecar as a side effect."""
    if ip is None or access is None:
        cfg_ip, cfg_access, _serial = _read_config()
        ip = ip or cfg_ip
        access = access or cfg_access
    if model_hint is None:
        model_hint = _read_model_hint()
    if not ip or not access:
        _write_status(ok=False, path=None, model_hint=model_hint or None,
                      last_error="credentials not configured")
        return None

    last_err = "no camera path available"
    for path in _ordered_paths(model_hint):
        if path == "rtsps":
            frame = _grab_rtsps_once(ip, access)
        else:
            frame = _grab_jpeg_once(ip, access)
        if frame:
            _write_frame(frame)
            with _state_lock:
                _last_status["frame_count"] = _last_status.get("frame_count", 0) + 1
                count = _last_status["frame_count"]
            _write_status(ok=True, path=path, model_hint=model_hint or None,
                          last_frame_at=time.time(), last_error=None,
                          frame_count=count)
            return frame
        last_err = f"{path} path returned no frame"

    _write_status(ok=False, path=None, model_hint=model_hint or None,
                  last_error=last_err)
    return None


def _poll_loop(stop_evt: threading.Event) -> None:  # pragma: no cover - daemon loop exercised via grab_once() in tests; the loop body just times grab_once + sleeps on stop_evt.
    """Background daemon: grab a frame every POLL_INTERVAL_SECONDS while a
    camera is reachable, backing off to OFFLINE_POLL_INTERVAL_SECONDS after a
    miss so a powered-down printer isn't nudged hard."""
    while not stop_evt.is_set():
        ip, access, _serial = _read_config()
        model_hint = _read_model_hint()
        got = None
        if ip and access:
            try:
                got = grab_once(ip, access, model_hint)
            except Exception:
                got = None
        else:
            _write_status(ok=False, path=None, last_error="credentials not configured")
        interval = (POLL_INTERVAL_SECONDS if got
                    else OFFLINE_POLL_INTERVAL_SECONDS)
        if stop_evt.wait(interval):
            return


def is_running() -> bool:
    t = _grab_thread[0]
    return t is not None and t.is_alive()


def start_grabber() -> bool:
    """Start the camera poll thread. Idempotent and gated by HUD_BAMBU_CAMERA.

    Returns True when a poll thread is running afterwards, False when the
    camera is disabled or unconfigured. Safe to call repeatedly — a second
    call while running is a no-op."""
    if not _camera_enabled():
        _write_status(ok=False, path=None, last_error="HUD_BAMBU_CAMERA disabled")
        return False
    ip, access, _serial = _read_config()
    if not ip or not access:
        _write_status(ok=False, path=None, last_error="credentials not configured")
        return False
    if is_running():
        return True
    _stop_evt.clear()
    t = threading.Thread(target=_poll_loop, args=(_stop_evt,),
                         name="BambuCameraGrabber", daemon=True)
    t.start()
    _grab_thread[0] = t
    return True


def stop_grabber() -> None:
    """Stop the camera poll thread if running."""
    _stop_evt.set()
    t = _grab_thread[0]
    if t is not None and t.is_alive():
        try:
            t.join(timeout=2.0)
        except Exception:
            pass
    _grab_thread[0] = None


if __name__ == "__main__":  # pragma: no cover - manual smoke entry point
    import sys
    print(f"[bambu_camera] cv2={'yes' if _HAS_CV2 else 'no'} "
          f"target_fps={_TARGET_FPS} interval={POLL_INTERVAL_SECONDS:.2f}s")
    _ip, _access, _serial = _read_config()
    if not _ip or not _access:
        print("[bambu_camera] no credentials (BAMBU_PRINTER_IP / "
              "BAMBU_ACCESS_CODE) — nothing to do.")
        sys.exit(1)
    print(f"[bambu_camera] model hint: {_read_model_hint()!r} -> "
          f"paths {_ordered_paths(_read_model_hint())}")
    frame = grab_once(_ip, _access)
    if frame:
        print(f"[bambu_camera] grabbed {len(frame)} bytes -> {FRAME_FILE}")
        sys.exit(0)
    print(f"[bambu_camera] no frame: {get_status().get('last_error')}")
    sys.exit(2)
