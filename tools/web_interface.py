"""Live web interface for JARVIS — a local-LAN dashboard + text command channel.

WHAT THIS IS
============
A tiny, dependency-light HTTP server that lets the owner (1) SEE what JARVIS is
doing from a browser — a live tail of the session log, plus a status strip
(online dot / version / awake state / uptime / model routing / VRAM / air-mouse
armed) — and (2) TALK TO HIM BY TEXT: a typed command is fed through the *exact
same* file-based inject channel a spoken command uses, so a typed "what time is
it" behaves identically to the spoken one. The page also carries QUICK-ACTION
BUTTONS (preset commands POSTed to /api/say) and an AUTO-REFRESH toggle that
freezes the 1s status/log polling so the view can be read without it scrolling.

STATUS FIELDS (build_status)
============================
version / state / running / routing / vram (gpu_lines+gpu_bar) / now_playing /
last_spoken / last_transcript / model, plus ``uptime`` (seconds since the newest
log's first timestamp, or None when not derivable) and ``air_mouse`` (a
``{"armed": bool, "engaged": bool}`` dict — present ONLY when the air-mouse skill
is loaded in THIS process, omitted entirely otherwise so its presence is truthful).

WHY STDLIB http.server (not Flask)
==================================
Flask already lives in this environment (the AirTag tracker runs on it at :8443),
but hard-depending on it here would (a) risk clashing with that server's import
state and (b) break bare-CI / cloud-only installs that don't ship Flask. So this
module is PURE STDLIB (http.server + socketserver + threading + json). We *probe*
for Flask lazily only to note its availability in logs — we never import-fail on
its absence and never actually route through it. One code path, everywhere.

HOW THE INJECT CHANNEL WORKS (reused verbatim from the voice loop)
==================================================================
JARVIS's main loop calls ``_drain_injected_command()`` at the top of every
iteration (bobert_companion.py). That function atomically renames
``injected_commands.json`` to ``.consuming``, pops the FIRST list item, and
requeues the tail. Each item is either a bare string or ``{"text": "...", ...}``.
We APPEND to that same file with the same atomic write-temp-then-os.replace
pattern the run-jarvis driver uses, so a typed command enters the loop exactly
as a mic turn would. The REPLY is read back by tailing the session log
(``logs/session_*.log``) from the byte offset captured at inject time, watching
for the ``JARVIS:`` / ``[action]`` lines the loop prints for that turn — mirroring
driver.py's ``wait_for_reply``. If no reply lands inside the timeout we return
``accepted: true`` (the command still ran; we just didn't capture spoken text).

SECURITY MODEL
==============
The endpoint can INJECT COMMANDS JARVIS EXECUTES, so binding it off-box is a real
exposure. The server therefore:
  • binds 127.0.0.1 by default (loopback — unreachable off the machine),
  • REFUSES TO START on a non-local bind (0.0.0.0 / a LAN IP) when the token is
    empty — ``create_server`` raises ``InsecureBindError`` with a clear reason,
  • when a token IS set, requires it on EVERY request via an Authorization: Bearer
    header, an X-Auth-Token header, or a ?token=… query param; a mismatch is 401.
The GET / dashboard page is served WITHOUT a token (it's just static HTML/JS that
then supplies the token on its API calls) ONLY on a local bind; on a non-local
bind even the page requires the token, so a bare browser hit can't fingerprint us.

TESTABILITY / HEADLESS-CI CONTRACT
==================================
Everything here is stdlib and OS-neutral (no win32, no real JARVIS needed):
  • ``create_server`` takes explicit ``inject_path`` / ``log_dir`` / ``hud_state_path``
    so a test can point them at a temp dir and bind 127.0.0.1:0 (ephemeral port).
  • The status/log/gpu sources all DEGRADE GRACEFULLY when the file/JARVIS is
    absent (missing log → empty tail; missing hud_state → unknown state; gpu
    import failure → omitted). Nothing here raises into a request handler.
  • The reply-wait is injected as ``reply_reader`` so a test can stub it (no live
    log to tail) — the default reader tails the newest session log.
  • Everything read from the RUNNING monolith (camera roster, camera gate, live
    ACTIONS registry) goes through ``runtime`` (default ``LiveRuntime``), which
    answers only when this process IS the booted JARVIS; a test passes a fake.

CONTROL ROUTES ADDED 2026-09-30 (web audit)
===========================================
  GET  /api/camera-tiles   tiles from the LIVE CAMERAS + Kinect switch, each
                           with the camera gate's verdict (retrying in N …)
  POST /api/action         run ONE registered action BY NAME (never typed into
                           the command channel); side-effect / destructive
                           names need ``"confirm": true`` (_ACTION_CONFIRM_RULES)
  POST /api/control        the tray control plane (force_wake, enter_standby,
                           mute/mic/pause toggles, restart, wake_word_mode_on/
                           _off = the pinned switch) via tray_commands.json
  GET  /api/panels, GET /api/panel/<id>/state, POST /api/panel/<id>/action,
  GET  /api/panel/<id>/stream/<name>   skill-declared panels (core/web_panels.py)
"""
from __future__ import annotations

import glob
import hmac
import json
import os
import re
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
import zlib
from fnmatch import fnmatchcase
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from core import camera_tiles as _camera_tiles

# ── Optional Flask probe (informational only — we NEVER route through it) ───
# Documented in the module header: Flask is present in this env for the AirTag
# tracker, but hard-depending on it would break bare CI. We note availability so
# a boot log can say "(flask present, using stdlib anyway)" but the server is
# always the stdlib ThreadingHTTPServer below.
try:  # pragma: no cover - trivial import probe; result only affects a log line
    import flask as _flask  # noqa: F401
    FLASK_AVAILABLE = True
except Exception:  # pragma: no cover
    FLASK_AVAILABLE = False


# ── project paths (defaults; overridable per-instance for tests / blue-green) ─
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_THIS_DIR)          # tools/ -> project root
DEFAULT_INJECT_PATH = os.path.join(PROJECT_DIR, "injected_commands.json")
DEFAULT_LOG_DIR = os.path.join(PROJECT_DIR, "logs")
DEFAULT_HUD_STATE_PATH = os.path.join(PROJECT_DIR, "hud_state.json")
# The full-control-panel data sources: the live camera preview frame (written a
# few times a second by the main loop while the camera is on) and the machine-
# generated action inventory. Overridable per-instance so a test points them at a
# temp dir (mirroring inject_path/log_dir/hud_state_path).
DEFAULT_CAMERA_PREVIEW_PATH = os.path.join(PROJECT_DIR, "data", ".hud_camera_preview.jpg")
DEFAULT_ACTION_INDEX_PATH = os.path.join(PROJECT_DIR, "docs", "ACTION_INDEX.md")
# A preview frame older than this is treated as "camera off" (stale) and served
# as a 404 so the panel shows its placeholder rather than a frozen last frame.
_CAMERA_PREVIEW_STALE_S = 5.0

# The per-camera preview keys the dashboard may ask for (?cam=). One tuple, used
# by BOTH the still route and the stream route (see _preview_path_for) so the two
# can never drift into disagreeing about which names are valid - and since
# 2026-09-30 it is the SAME tuple the monolith's preview writer uses
# (core/camera_tiles.PREVIEW_KEYS), not a second copy. Which of these the Camera
# tab actually DRAWS comes from the live CAMERAS roster (_camera_tile_list).
_CAMERA_PREVIEW_CAMS = _camera_tiles.PREVIEW_KEYS

# ── /api/camera-stream (MJPEG) ───────────────────────────────────────────────
# WHY THIS EXISTS (measured live 2026-09-04, owner: "the preview of webcams is
# very slow still"): serving ONE still per request costs only ~1.6 ms end to end,
# so the request was never the problem — the dashboard's setInterval(…, 1000) WAS
# the frame rate. A still-per-frame design also cannot do better than the poll
# interval, and shrinking that interval multiplies connection churn (this server
# speaks HTTP/1.0, so every still is its own TCP connection).
# multipart/x-mixed-replace fixes both: ONE connection per tile, and the server
# pushes the next JPEG the instant the preview file changes, so latency is
# (producer write -> _CAMERA_STREAM_POLL_S) instead of (producer write -> up to a
# full poll interval). The still route is untouched — the HUD, old bookmarks and
# the tests still use it, and it is the client's fallback.
_CAMERA_STREAM_BOUNDARY = "jarvisframe"
_CAMERA_STREAM_POLL_S = 0.02      # how often a streaming thread re-stats the file
# Each stream holds a ThreadingHTTPServer worker thread for as long as the tab is
# open, so cap them: 3 tiles x a couple of dashboards, and no more. Beyond this a
# stream request is refused (503) and the client falls back to still polling — a
# refused stream must never be able to starve /api/status or /api/log of threads.
_CAMERA_STREAM_MAX_CLIENTS = 8
_camera_stream_clients = [0]
_camera_stream_clients_lock = threading.Lock()


def _camera_streams_saturated() -> bool:
    """True when every MJPEG slot is taken, so the NEXT /api/camera-stream is
    refused with 503 (see _stream_camera). One int read under the counter's own
    lock: no I/O, nothing to block on, safe to call from the reason path.

    This is the ONE thing the server actually knows about a refused stream, and
    _camera_off_reason exists to say only what is established."""
    with _camera_stream_clients_lock:
        return _camera_stream_clients[0] >= _CAMERA_STREAM_MAX_CLIENTS


class UnknownCamError(ValueError):
    """?cam= named something that is not one of _CAMERA_PREVIEW_CAMS."""


def _preview_path_for(cfg: dict, cam: str) -> str:
    """Resolve the preview JPEG path for ``cam`` ("" = the primary/composite file
    the HUD reads, otherwise one of _CAMERA_PREVIEW_CAMS). Raises UnknownCamError
    for any other name. Returns "" when no preview path is configured at all.

    ONE source of truth for /api/camera-preview AND /api/camera-stream: the two
    routes used to be free to disagree about where a tile lives, which is exactly
    how a stale duplicate starts."""
    p = cfg.get("camera_preview_path", "")
    cam = (cam or "").strip().lower()
    if not cam:
        return p
    if cam not in _CAMERA_PREVIEW_CAMS:
        raise UnknownCamError(cam)
    base = os.path.dirname(p) if p else os.path.join(PROJECT_DIR, "data")
    return os.path.join(base, f".hud_camera_preview_{cam}.jpg")


def _preview_stat(path: str):
    """(mtime_ns, size, age_seconds) for a preview file, or None when it is
    missing/unreadable. Cheap: a stat, never an open. Never raises."""
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size, time.time() - st.st_mtime


def _camera_live_map(cfg: dict) -> dict:
    """{cam: bool} — is each tile's preview file CURRENT right now.

    WHY THIS EXISTS (measured in Chrome 2026-09-05, see the MID-STREAM DEATH
    note in the page script): a multipart/x-mixed-replace <img> gets NO DOM
    event when the stream ends, so the page cannot notice a camera that dies
    mid-stream on its own. This is the ONE fact it was missing, and it is
    answered by three os.stat calls through the SAME (_preview_stat,
    _CAMERA_PREVIEW_STALE_S) pair /api/camera-preview, /api/camera-stream and
    _camera_off_reason use — so the supervisor can never disagree with the
    moment _stream_camera decides to close.

    Deliberately stat-ONLY: this sits on the dashboard's poll path, so it must
    never touch the reason ladder or the ~0.7 s device-enumeration probe. It
    reports freshness and nothing else; the WORDS still come from
    /api/camera-reason, on the down path only.
    """
    out = {}
    for cam in _CAMERA_PREVIEW_CAMS:
        try:
            info = _preview_stat(_preview_path_for(cfg, cam))
        except UnknownCamError:          # unreachable: cams come from the tuple
            info = None
        out[cam] = info is not None and info[2] <= _CAMERA_PREVIEW_STALE_S
    return out


# ── the RUNNING JARVIS, as seen from the in-process web server ──────────────
class NoRuntime:
    """Nothing is known about a running JARVIS: every reader returns None/''
    and the dashboard falls back to its defaults. What a bare web process, a
    test, or headless CI sees."""

    live = False

    def cameras(self):
        return None

    def kinect_enabled(self):
        return None

    def gate_key(self, cam) -> str:
        return ""

    def gate_snapshot(self):
        return None

    def actions(self):
        return None

    def speak_sets(self):
        return None


class LiveRuntime(NoRuntime):
    """Reads the RUNNING monolith - and ONLY when this process IS the booted
    JARVIS.

    WHY THE __main__ CHECK. The server runs in-process with the main loop
    (skills/web_interface calls create_server inside JARVIS), and the boot
    aliases ``sys.modules["bobert_companion"] = sys.modules["__main__"]``, so
    the two being the SAME object is exactly "this is the live assistant". A
    test that merely imported the monolith (tests/_monolith_harness) has a
    different ``__main__`` - the test runner - so it reads nothing here
    instead of a half-configured test copy. Every reader is a plain attribute
    read (no import, no device I/O) and never raises."""

    @staticmethod
    def _bc():
        try:
            bc = sys.modules.get("bobert_companion")
            if bc is not None and bc is sys.modules.get("__main__"):
                return bc
        except Exception:
            pass
        return None

    @property
    def live(self):  # type: ignore[override]
        return self._bc() is not None

    def cameras(self):
        bc = self._bc()
        cams = getattr(bc, "CAMERAS", None) if bc is not None else None
        return list(cams) if isinstance(cams, (list, tuple)) else None

    def kinect_enabled(self):
        bc = self._bc()
        if bc is None:
            return None
        v = getattr(bc, "KINECT_ENABLED", None)
        return None if v is None else bool(v)

    def gate_key(self, cam) -> str:
        bc = self._bc()
        fn = getattr(bc, "_camera_gate_key", None) if bc is not None else None
        try:
            return str(fn(cam)) if callable(fn) else ""
        except Exception:
            return ""

    def gate_snapshot(self):
        bc = self._bc()
        gate = getattr(bc, "_camera_gate", None) if bc is not None else None
        try:
            snap = gate.snapshot() if gate is not None else None
            return snap if isinstance(snap, dict) else None
        except Exception:
            return None

    def actions(self):
        bc = self._bc()
        acts = getattr(bc, "ACTIONS", None) if bc is not None else None
        return acts if isinstance(acts, dict) else None

    def speak_sets(self):
        bc = self._bc()
        if bc is None:
            return None
        return (set(getattr(bc, "SPEAK_RESULT_VERBATIM_ACTIONS", ()) or ()),
                set(getattr(bc, "INFORMATIVE_ACTIONS", ()) or ()),
                set(getattr(bc, "SELF_VOICED_ACTIONS", ()) or ()))


def _runtime(cfg: dict):
    rt = cfg.get("runtime") if isinstance(cfg, dict) else None
    return rt if rt is not None else NoRuntime()


def _camera_tile_list(cfg: dict) -> tuple:
    """``(tiles, source)``: the tiles the Camera tab draws.

    source "live" - built from the running JARVIS's CAMERAS roster plus its
    Kinect switch (core/camera_tiles.tiles_from_config), so a camera the owner
    removed from CAMERAS no longer gets a permanently dark tile. source
    "default" - no running JARVIS to ask (a bare web process, a test): every
    preview key, exactly the pre-2026-09-30 behaviour. Never raises."""
    rt = _runtime(cfg)
    try:
        cams = rt.cameras()
        if cams is not None:
            tiles = _camera_tiles.tiles_from_config(
                cams, rt.kinect_enabled(), gate_key=rt.gate_key)
            return tiles, "live"
    except Exception:
        pass
    labels = {"left": "Left webcam", "right": "Right webcam",
              "kinect": "Kinect (skeleton)"}
    return ([{"cam": k, "label": labels.get(k, k),
              "kind": "kinect" if k == _camera_tiles.KINECT_KEY else "webcam",
              "gate_key": _camera_tiles.KINECT_KEY
              if k == _camera_tiles.KINECT_KEY else ""}
             for k in _CAMERA_PREVIEW_CAMS], "default")


def _camera_gate_for(cfg: dict, gate_key: str, snapshot=None):
    """The camera gate's verdict on one device (core/camera_tiles.gate_summary)
    or None. ``snapshot`` lets a caller share one snapshot across tiles."""
    if not gate_key:
        return None
    if snapshot is None:
        snapshot = _runtime(cfg).gate_snapshot()
    return _camera_tiles.gate_summary(snapshot, gate_key)


def camera_tiles_payload(cfg: dict) -> dict:
    """GET /api/camera-tiles: the tiles to draw, each with the camera gate's
    verdict ({state, message, retry_in_s} or None). One gate snapshot per
    request - an in-memory read under the gate's own lock, no device I/O -
    so it is safe on the Camera tab's slow poll."""
    tiles, source = _camera_tile_list(cfg)
    snap = _runtime(cfg).gate_snapshot()
    out = []
    for t in tiles:
        row = {k: t[k] for k in ("cam", "label", "kind")}
        row["gate"] = _camera_gate_for(cfg, t.get("gate_key", ""), snap)
        out.append(row)
    return {"tiles": out, "source": source,
            "keys": list(_CAMERA_PREVIEW_CAMS)}


# -- WHY IS THIS TILE BLANK? (one source of truth) ---------------------------
# The owner asked for the words "Kinect not detected" on the tile. Measured on
# his machine 2026-09-04 23:09-23:28: the OS enumerated the sensor fine the whole
# time (Xbox NUI Sensor / WDF KinectSensor Interface 0 / Microphone Array, all
# Status OK) while the bridge streamed nothing for six minutes, and then it
# started working. So "not detected" would have been FALSE at every moment of the
# outage he was staring at, and it would have sent him to check a cable that was
# already fine. This module therefore treats "no picture" as a LADDER of states,
# each rung gated on a symbol that actually establishes it, and hands the owner
# his literal string ONLY on the rung where device enumeration came back empty.
#
# The rules the ladder obeys:
#   * "not detected" requires a COMPLETED enumeration that found nothing. A probe
#     that failed to run yields None (unknown), never False.
#   * "not powered" is never asserted. The OS cannot see mains power, so the
#     no-frames rung is worded as GUIDANCE ("check its power adapter") and keeps
#     the other candidate cause (another app holding the sensor) in the same
#     sentence, because those two are genuinely indistinguishable from here.
#   * when two states cannot be told apart, the message says the ambiguous thing.
#     There are two separate no-frames rungs for exactly that reason: one that
#     may say "plugged in" because enumeration proved it, and one that may not.
#   * enumeration is ASYMMETRIC, and the messages respect that. A device that
#     enumerates PRESENT is physically attached (-PresentOnly means attached
#     right now), so the present-rung may say "plugged in". A sweep that matched
#     NOTHING establishes only that Windows currently enumerates no Kinect. The
#     v2's USB3 adapter carries the data link AND the mains brick, so a brick
#     switched off or dead - and equally a removed, blocked or disabled driver
#     stack - stops the whole adapter enumerating while the cable is still fully
#     seated. "Kinect not detected" therefore names the OBSERVATION and never
#     asserts that nothing is plugged in: that claim would send the owner to
#     re-seat a cable that is fine while the power switch is the actual fault.
#   * left/right webcams never consult any Kinect signal, so a dead Kinect can
#     never change (or blank) their tiles.

# How long a device-enumeration result is reused. The probe is a powershell.exe
# spawn measured at 0.61-0.78 s on this box, so it must never sit on a hot path.
# 20 s is the smallest TTL at which a tile stuck in the dashboard's 4 Hz error
# retry still spawns at most 3 processes a minute, while a replug the owner does
# at the desk shows up before he has walked back to the keyboard. It is consulted
# ONLY on the error path (see /api/camera-reason), so while a camera is streaming
# normally this probe never runs at all.
_KINECT_ENUM_TTL_S = 20.0
_KINECT_ENUM_TIMEOUT_S = 8.0
_kinect_enum_cache = {"ts": 0.0, "present": None, "names": (), "how": "never run"}
_kinect_enum_lock = threading.Lock()

# Match on BOTH friendly names and raw Kinect USB hardware ids: a sensor whose
# driver failed to install has no recognisable FriendlyName but still enumerates
# its USB node, and calling that "not detected" would be the same lie in a new
# costume. -PresentOnly is the load-bearing flag - the PnP Enum registry keeps a
# key for every device the machine has EVER seen, so an unfiltered sweep would
# happily report a Kinect that was unplugged months ago.
_KINECT_PNP_PS = (
    "$ErrorActionPreference='SilentlyContinue';"
    "Get-PnpDevice -PresentOnly | Where-Object {"
    " $_.FriendlyName -like '*Kinect*'"
    " -or $_.FriendlyName -like '*NUI Sensor*'"
    " -or $_.InstanceId -like '*VID_045E&PID_02C4*'"
    " -or $_.InstanceId -like '*VID_045E&PID_02D8*'"
    " -or $_.InstanceId -like '*VID_045E&PID_02D9*'"
    " -or $_.InstanceId -like '*VID_045E&PID_02BB*'"
    " -or $_.InstanceId -like '*VID_045E&PID_02AE*' } |"
    # THE PROJECTION MUST EMIT SOMETHING NON-EMPTY FOR EVERY MATCH. Projecting
    # $_.FriendlyName alone silently undid the five InstanceId clauses above:
    # they exist precisely for a sensor whose driver failed to bind, and such a
    # node enumerates with a NULL FriendlyName. The pipeline then exited 0 with
    # no printable line, _probe_kinect_devices' `if ln.strip()` dropped it, and
    # a plugged-in, enumerated Kinect read as present=False - the one and only
    # reading that unlocks the "not_detected" rung. Measured on the owner's box
    # 2026-09-04: Get-PnpDevice -PresentOnly does return present nodes whose
    # FriendlyName is $null, so this was not hypothetical. Fall back
    # to the InstanceId, then to a literal, so a matched device can NEVER
    # vanish between the filter and stdout.
    " ForEach-Object {"
    " $n = $_.FriendlyName;"
    " if ([string]::IsNullOrWhiteSpace($n)) { $n = $_.InstanceId };"
    " if ([string]::IsNullOrWhiteSpace($n)) { $n = 'unnamed present device' };"
    " $n }"
)


def _probe_kinect_devices() -> tuple:
    """Run the OS device-enumeration probe ONCE. Returns (present, names, how):

      present True  - the sweep completed and matched at least one live device
      present False - the sweep completed and matched NOTHING (the only reading
                      that may ever produce the words "Kinect not detected")
      present None  - the sweep did not complete (not Windows, no powershell,
                      timed out, non-zero exit). UNKNOWN, never "no".

    Never raises. Kept separate from the cache so a test can patch just this."""
    if sys.platform != "win32":
        return None, (), "not windows"
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive",
             "-Command", _KINECT_PNP_PS],
            capture_output=True, text=True, timeout=_KINECT_ENUM_TIMEOUT_S,
            # No console flash over the owner's work - the same guard the
            # monolith puts on every subprocess it spawns from the main loop.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as e:
        return None, (), "probe failed: %s" % type(e).__name__
    if proc.returncode != 0:
        # A sweep that errored proves nothing about the hardware.
        return None, (), "probe exit %s" % proc.returncode
    names = tuple(ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip())
    return bool(names), names, "Get-PnpDevice -PresentOnly"


def _kinect_devices_present(now: float | None = None) -> tuple:
    """TTL-cached (present, names, how) from _probe_kinect_devices.

    Concurrency: three tiles can fire their error handlers in the same frame, so
    the probe sits behind a lock with a SHORT acquire timeout - a caller that
    loses the race returns whatever the cache already holds (unknown on a cold
    start) instead of queueing behind a second-long subprocess. Being briefly
    vague is fine; blocking three dashboard workers on powershell is not."""
    t = time.time() if now is None else now
    c = _kinect_enum_cache
    if c["how"] != "never run" and (t - c["ts"]) < _KINECT_ENUM_TTL_S:
        return c["present"], c["names"], c["how"]
    if not _kinect_enum_lock.acquire(timeout=0.05):
        return c["present"], c["names"], c["how"]
    try:
        if c["how"] != "never run" and (time.time() - c["ts"]) < _KINECT_ENUM_TTL_S:
            return c["present"], c["names"], c["how"]   # a winner just refreshed it
        present, names, how = _probe_kinect_devices()
        c.update({"ts": time.time(), "present": present, "names": names, "how": how})
        return present, names, how
    finally:
        _kinect_enum_lock.release()


def _kinect_health() -> dict | None:
    """audio.kinect_bridge.get_stream_health(), or None when not cheaply reachable.

    WHY sys.modules AND NOT AN IMPORT - the identical reasoning as
    _air_mouse_status() below, which is the precedent this follows: the web
    interface runs IN-PROCESS with the JARVIS main loop, so when the bridge is
    part of this build it is already in sys.modules under that key. Importing it
    here instead would drag pykinect2/comtypes into a bare-CI or cloud-only web
    process, and would make "the module is absent" indistinguishable from "the
    sensor is dead". None therefore means UNKNOWN and the ladder says so, rather
    than inventing a failure.

    WHY get_stream_health() AND NOT available()/get_runtime() - those two OPEN
    the sensor, and the winner of the bridge's open lock runs a retry gauntlet of
    up to ~16 s inline (kinect_bridge._open_runtime_locked); that gauntlet
    tripped the JARVIS main-loop watchdog on 2026-07-10. get_stream_health() is
    pure cell reads: no lock, no I/O, no open attempt."""
    try:
        kb = sys.modules.get("audio.kinect_bridge")
        getter = getattr(kb, "get_stream_health", None) if kb else None
        if callable(getter):
            h = getter()
            if isinstance(h, dict):
                return h
    except Exception:
        pass
    return None


# The rungs, kept as data so this module, the tests and the dashboard JS cannot
# drift into three different vocabularies for the same state.
_CAMERA_REASONS = {
    # THIS ANSWER IS ONLY EVER READ INSIDE AN EMPTY BOX. /api/camera-reason is
    # fetched from exactly one place - a tile's `error` handler - and its message
    # is written into .camoff, the dashed placeholder that is by definition
    # showing because there is no picture. "Live." there asserted the opposite of
    # what the owner was looking at, which is the same class of lie as "not
    # detected" on a sensor Windows can see.
    #
    # What a fresh preview file actually establishes is: the camera IS producing
    # frames, and this page is not showing them - so the fault is the page's
    # video connection, not the camera. WHICH fault (a refused stream, an aborted
    # connection, the browser's per-origin cap) is NOT established, so none is
    # named; the tile says the ambiguous thing and stops there.
    "live":            ("Camera is sending pictures, but this page is not "
                        "receiving them - reconnecting."),
    # The one sub-case the server CAN establish, because it is the thing the
    # server itself just did: every stream slot is taken, so /api/camera-stream
    # is refusing new clients (503) right now. Named separately because it is the
    # only rung with an action the owner can take from where he is sitting, and
    # because the dashboard uses it to stop asking (see explainTile).
    "stream_busy":     ("Camera is sending pictures, but every video connection "
                        "slot is in use - close other dashboard tabs."),
    "webcam_off":      "Webcam off - no picture is being sent right now.",
    "disabled":        "Kinect is switched off in JARVIS settings.",
    # THE OWNER'S LITERAL STRING. Reachable from exactly one place: a completed
    # enumeration that matched nothing. Never from a bridge signal. It stops at
    # what that sweep established - Windows enumerates no Kinect device - and
    # does NOT go on to say nothing is plugged in, which an empty sweep cannot
    # establish (the asymmetry note above; a Kinect whose mains brick is off
    # stops enumerating with its cable still seated).
    "not_detected":    "Kinect not detected - Windows sees no Kinect device.",
    "no_frames":       ("Kinect is plugged in but sending no pictures - check its "
                        "power adapter is on, or if another app is using it."),
    "no_frames_unverified": ("Kinect is sending no pictures - check it is plugged "
                             "in, its power adapter is on, and no other app has it."),
    # TWO rungs, not one, because the bridge genuinely tells them apart and they
    # send the owner to two different places. Both name the PYTHON PACKAGE, which
    # is what an import failure establishes; neither names the Kinect DRIVER,
    # which the bridge never reached (see the pykinect2 rung in the ladder).
    "pykinect2_missing":  ("JARVIS's Kinect Python package (pykinect2) is not installed, "
                           "so JARVIS never reached the Kinect driver - fix it with: "
                           "pip install pykinect2."),
    "pykinect2_unusable": ("JARVIS's Kinect Python package (pykinect2) would not load, "
                           "so JARVIS never reached the Kinect driver - hover this "
                           "message for the error."),
    # An open that is STILL RUNNING is not a failure, and dressing it as one is
    # the same class of lie as the rungs above: nothing has been established
    # except that no attempt has FINISHED. Present tense, no cause named, and it
    # stays true even if the cell is a few seconds stale - the worst it can be is
    # "not finished yet", never "it broke".
    "open_in_progress": ("Kinect is still being opened - JARVIS has not finished "
                         "trying yet."),
    "open_failed":     "JARVIS could not open the Kinect.",
    # NOT "nothing has probed it yet". That is only ONE of the two situations
    # that produce this exact reading, and no symbol the bridge publishes tells
    # them apart (see the rung in the ladder). Names both, asserts neither, and
    # blames no hardware - the reading is equally consistent with a perfectly
    # healthy sensor nobody has asked for yet.
    "not_open_no_error": ("Kinect is not open right now and JARVIS recorded no "
                          "error - either nothing has probed the sensor yet, or a "
                          "connection that was working has since dropped."),
    "worker_stopped":  "Kinect is connected, but the JARVIS worker reading it stopped.",
    # Reachable ONLY from color_pending True. Color is the single stream this
    # tile renders, so only color may claim the picture is being produced.
    "frames_not_shown": "Kinect is running, but its picture is not reaching this page.",
    # The sensor is alive on a stream this tile does NOT display (body/depth),
    # which establishes the Kinect is streaming and establishes NOTHING about the
    # color camera. Says the ambiguous thing on purpose: color_pending False is a
    # failure to confirm, not proof the camera died, so the message names both
    # candidates and picks neither. Never asserts the picture reaches the page.
    "color_unconfirmed": ("Kinect is running and sending data, but no new camera "
                          "picture has been confirmed - could be the color camera "
                          "or this page."),
    "open_quiet":      "Kinect is connected but no new pictures are arriving.",
    "unknown":         "Kinect state unknown.",
    "no_path":         "No preview is configured for this camera.",
    # 2026-09-30 (web audit). The running JARVIS's CAMERAS roster does not
    # list this camera (or the Kinect is switched off), so nothing will ever
    # write its preview: say THAT instead of a forever-dark "Webcam off".
    "not_configured":  "This camera is not in JARVIS's camera list.",
    # The camera gate (core/camera_gate.py) is deliberately NOT opening this
    # device right now. The WHY and the countdown come from the gate itself
    # (core/camera_tiles.gate_summary) and ride in `detail` + `retry_in_s`, so
    # the words cannot drift from the rule that produced them.
    "held_by_gate":    "JARVIS is deliberately not opening this camera right now.",
}

# The bridge's IN-FLIGHT marker, matched as a substring exactly the way the
# "no frames" / "pykinect2 " markers are. Named, rather than inlined at the rung,
# so the test that greps audio/kinect_bridge.py for it fails loudly the day that
# string is reworded there - a marker matched in one copy and reworded in the
# other is the stale-duplicate shape this ladder keeps getting bitten by.
# SOURCE: kinect_bridge._open_runtime_locked's lost-acquire return,
#   return None, (_open_error[0] or "Kinect open already in progress")
_KINECT_OPEN_IN_PROGRESS = "open already in progress"


def _reason(cam: str, state: str, detail: str | None = None) -> dict:
    """One rung as a payload. ``message`` always comes from _CAMERA_REASONS, so a
    state can never ship with hand-written prose that drifts from the table."""
    out = {"cam": cam, "state": state,
           "message": _CAMERA_REASONS.get(state, _CAMERA_REASONS["unknown"])}
    if detail:
        out["detail"] = detail
    return out


def _gate_reason(cam: str, gate: dict) -> dict:
    """The held_by_gate rung: the table sentence, the gate's own explanation
    as the (visible) detail, and the countdown the tile uses to stop asking
    until JARVIS itself will try again."""
    out = _reason(cam, "held_by_gate", detail=gate.get("message") or None)
    out["gate_state"] = gate.get("state", "")
    out["retry_in_s"] = gate.get("retry_in_s")
    return out


def _camera_off_reason(cfg: dict, cam: str) -> dict:
    """Why ``cam``'s tile has no picture: {"cam","state","message"[,"detail"]}.

    THE one place that answers this question, for every tile. Raises
    UnknownCamError for an unknown ?cam= (the route already handles that).

    Freshness is decided by _preview_stat + _CAMERA_PREVIEW_STALE_S - the SAME
    pair /api/camera-preview and /api/camera-stream use to decide their 404 - so
    the explanation can never disagree with the thing it is explaining."""
    path = _preview_path_for(cfg, cam)                 # may raise UnknownCamError
    cam = (cam or "").strip().lower()
    if not path:
        return _reason(cam, "no_path")
    info = _preview_stat(path)
    fresh = info is not None and info[2] <= _CAMERA_PREVIEW_STALE_S
    if fresh:
        # The file IS current, so the camera is not why this tile is blank - the
        # page's video connection is. Claiming the camera is off here would be a
        # lie; so is the bare word "Live." in a box that has no picture in it.
        # Split, because the cap is the one sub-case the server can prove: with
        # every slot taken, _stream_camera is answering 503 right now, and the
        # tile is told so (and stops re-arming a stream it cannot get).
        if _camera_streams_saturated():
            return _reason(cam, "stream_busy")
        return _reason(cam, "live")
    # NOT CONFIGURED (2026-09-30). Only when the running JARVIS was actually
    # asked (source "live"): its roster has no tile for this key, so no preview
    # will ever be written. A web process with no JARVIS to ask keeps the old
    # ladder rather than guessing.
    tiles, source = _camera_tile_list(cfg)
    by_key = {t["cam"]: t for t in tiles}
    if source == "live" and cam not in by_key:
        if cam == "kinect":
            return _reason(cam, "disabled",
                           detail="KINECT_ENABLED is off in the running JARVIS")
        return _reason(cam, "not_configured",
                       detail="no CAMERAS entry maps to the %s tile" % cam)
    gate = _camera_gate_for(cfg, (by_key.get(cam) or {}).get("gate_key", ""))
    if cam != "kinect":
        # Webcams expose no health surface at all, so "no recent frame" is
        # everything that is actually established - unless the camera GATE is
        # the reason, which it says itself. Deliberately routed through NO
        # Kinect signal: a dead Kinect must never alter a webcam's tile.
        if gate:
            return _gate_reason(cam, gate)
        return _reason(cam, "webcam_off")

    health = _kinect_health()
    # RUNG 1 - the switch. With KINECT_ENABLED off the bridge never opens the
    # sensor at all, so whether one is plugged in is not why this tile is blank.
    if health is not None and not health.get("enabled", True):
        return _reason(cam, "disabled")

    # RUNG 2 - the only rung that may say "not detected", and only on a COMPLETED
    # sweep that matched nothing. present None (the probe could not run) falls
    # through to the bridge signals rather than guessing.
    present, _names, how = _kinect_devices_present()
    if present is False:
        return _reason(cam, "not_detected", detail="device enumeration: %s" % how)

    # RUNG 2b (2026-09-30) - the camera GATE is holding the Kinect (its backoff,
    # the slow dies-on-open retry, a USB-storm cool-down, a quarantine). That is
    # an established fact about JARVIS's own behaviour, with a countdown, and it
    # explains the blank tile better than the bridge's latched open error does.
    if gate:
        return _gate_reason(cam, gate)

    if health is None:
        return _reason(cam, "unknown",
                       detail="audio.kinect_bridge is not loaded in this process")

    err = (health.get("open_error") or "").strip()
    detail = err or None
    if not health.get("open"):
        # RUNG 3 - opened but streamed nothing: the plugged-in-but-no-picture
        # case, and the one the owner actually hit tonight. Unpowered, unplugged
        # mid-session and held-by-another-process are indistinguishable from
        # here, so the message keeps every candidate and asserts none of them.
        if "no frames" in err:
            state = "no_frames" if present else "no_frames_unverified"
            return _reason(cam, state, detail=detail)
        # RUNG 4 - the bridge could not load the pykinect2 PYTHON PACKAGE. Its
        # two strings both start "pykinect2 " (kinect_bridge, the
        # import_pykinect2() call in _open_runtime_locked): "pykinect2 not
        # installed - pip install pykinect2" on ImportError, and "pykinect2
        # failed to load: {type}: {e}" on anything else - a broken comtypes, a
        # venv rebuilt against a new Python, a corrupted site-packages.
        #
        # WHY THIS MUST NOT SAY "DRIVER" (2026-09-05). Both are pip-level import
        # failures INSIDE JARVIS, raised before PyKinectRuntime is ever
        # constructed, so nothing about the Windows Kinect driver has been
        # established - and enumeration two rungs up may have just returned
        # present True, i.e. the driver is demonstrably fine. The old single
        # "Kinect driver software is missing" rung sent the owner off to
        # reinstall the Kinect SDK (an hour, a reboot) when the fix was one pip
        # command, and the true text survived only in the hover title. Name the
        # dependency that actually failed, and nothing else.
        if err.startswith("pykinect2 "):
            # "not installed" is the ImportError branch: the package is absent.
            if "not installed" in err:
                return _reason(cam, "pykinect2_missing", detail=detail)
            # Anything else got FURTHER than absent but still would not load.
            # WHY it died lives in the exception, which we do not interpret - and
            # this is also where an unrecognised future pykinect2 string lands,
            # so the fallback is the hedged message, never the specific one.
            return _reason(cam, "pykinect2_unusable", detail=detail)
        # RUNG 5 - an open that is STILL IN FLIGHT, which is not a completed
        # failure at all. kinect_bridge._open_runtime_locked hands
        # "Kinect open already in progress" to whoever LOSES the 0.5 s acquire on
        # its open-attempt lock while the winner is inside the ~16 s
        # verify/retry gauntlet, and get_runtime() then feeds that string to
        # _publish_open_failure, which latches it into the SAME open_error cell a
        # real failure uses (with the short 5 s cooldown, since it is not a
        # no-frames verdict). Nothing about the sensor has been established here:
        # the only fact is that another thread got there first.
        #
        # WHY IT MATTERS (2026-09-05). At boot this is the NORMAL reading - the
        # always-on body pump is in the gauntlet, the preview compositor's
        # get_color_bgr() -> get_runtime() loses the race, and the preview file is
        # still last session's, so the tile fires its error handler and asks
        # precisely then. Reported as open_failed it said "JARVIS could not open
        # the Kinect." during an open that went on to SUCCEED (measured
        # 2026-09-04 23:16: "[kinect] sensor live after 2 open attempts") - a
        # first-frame delay rendered as a hardware fault, which is exactly the
        # wrong place to send the owner. Checked AFTER the two markers above: the
        # bridge returns a previously latched error in preference to this string,
        # so an err carrying "no frames"/"pykinect2 " is a real completed
        # verdict and keeps its own rung.
        if _KINECT_OPEN_IN_PROGRESS in err:
            return _reason(cam, "open_in_progress", detail=detail)
        if err:
            return _reason(cam, "open_failed", detail=detail)
        # RUNG 6 - open False with NO recorded error. This is TWO situations
        # wearing one reading, and NOTHING in the health dict separates them
        # (reproduced against the real bridge 2026-09-05):
        #   * nothing has probed the sensor yet - the cold-start reading; and
        #   * a sensor that opened, streamed, and then DIED mid-session.
        #     kinect_bridge._publish_runtime clears _open_error[0] on every
        #     successful open, and reset_if_body_stale then drops _runtime[0] on
        #     the both-planes-stale path WITHOUT writing an error - so the
        #     instant the owner's "skeleton rendered for a while then stopped"
        #     intermittent fires, health reads open False / open_error None,
        #     exactly like a cold start, and keeps reading that way for the
        #     whole ~16 s reopen gauntlet.
        # The old message ("JARVIS has not tried the sensor yet") asserted the
        # first one, so the tile sent the owner hunting a boot problem while his
        # Kinect was mid-death. get_stream_health's own docstring codifies the
        # same inference ("(open False, open_error None) means NOT YET PROBED");
        # it is wrong there too, and neither copy can establish it.
        #
        # DO NOT try to re-derive the distinction from the other health keys:
        # set_enabled(True) calls start_body_pump(), which sets pump_alive AND
        # seeds BOTH frame clocks (so body_age_s/color_age_s go non-None) with
        # no sensor ever opened - a cold start and a torn-down runtime read the
        # same on all of them. Separating the two needs a has-ever-opened flag
        # the bridge does not publish; until it does, the tile says the
        # ambiguous thing rather than picking the likelier one.
        #
        # detail carries the one thing actually OBSERVED (it lands in the tile's
        # hover title): a Kinect preview frame was written at that mtime, or
        # there is no preview file at all. Evidence, not a verdict - the file
        # outlives the process, so it dates the last frame, not this session.
        age = None if info is None else info[2]
        return _reason(cam, "not_open_no_error", detail=(
            "no runtime open, no recorded open error; last %s preview frame: %s"
            % (cam, "none on disk" if age is None else "%.1fs ago" % age)))

    # open True from here: a runtime exists, so the sensor answered at least once.
    if not health.get("pump_alive", True):
        return _reason(cam, "worker_stopped")
    if health.get("color_pending"):
        # COLOR ONLY. color_pending True is poll-independent proof a NEW COLOR
        # frame is arriving, and COLOR is the one and only stream this tile
        # renders (bobert_companion._compose_kinect_preview builds the JPEG from
        # get_color_bgr), so the gap really is downstream of the sensor.
        #
        # WHY NOT any(color, body, depth) - 2026-09-05. That is what this rung
        # used to read, and it let a stream the tile does not display assert that
        # the picture was being produced:
        #   * DEPTH is pinned pending FOREVER. _depth_time_seen only advances
        #     inside get_depth(), and nothing polls depth on a schedule (the
        #     bridge says so itself at audio/kinect_bridge.py: "nothing polls
        #     depth on a schedule; body has the 30 Hz pump"). Its only callers
        #     are the on-demand kinect_status voice probe and the hand-depth
        #     heuristic. So while the sensor emits depth at all, `t > cell[0]`
        #     never stops being true.
        #   * BODY is the documented BODY-but-no-COLOR reopen - "the skeleton
        #     stayed dark while gestures worked" - the exact failure the bridge's
        #     require_color=True flag exists to reject.
        # Either one made health{open, pump_alive, color_pending=False,
        # depth_pending=True} render "Kinect is running, but its picture is not
        # reaching this page." on a DEAD COLOR CAMERA, sending the owner to debug
        # his browser. A rung that judges a COLOR tile reads color_pending.
        return _reason(cam, "frames_not_shown")

    others = [k for k in ("body_pending", "depth_pending") if health.get(k)]
    if others:
        # The sensor is demonstrably alive - a non-color stream is delivering new
        # frames - but COLOR, the only stream this tile shows, is unconfirmed.
        # That is a FAILURE TO ESTABLISH, not a dead camera: color_pending False
        # is also the normal reading in the instant after the preview composer
        # consumed the frame. Both candidates stay on the table, neither is
        # asserted, and the detail carries the evidence into the tooltip.
        return _reason(cam, "color_unconfirmed",
                       detail="new frames on %s; color_pending false"
                              % ", ".join(k[:-len("_pending")] for k in others))
    # Nothing pending is NOT proof of death - it is also the normal reading in the
    # instant after the 30 Hz body pump consumed the frame. Hence the hedged,
    # present-tense "no new pictures are arriving", with no cause named.
    return _reason(cam, "open_quiet")


# Reply-wait bounds. The main loop can take a while on a cloud LLM turn, so we
# allow a generous ceiling but poll cheaply. A caller (the POST handler) passes a
# per-request timeout; these just bound it so a hostile ?timeout can't hang a
# worker thread forever.
_REPLY_TIMEOUT_DEFAULT = 30.0
_REPLY_TIMEOUT_MAX = 120.0
_LOG_TAIL_MAX_LINES = 2000        # hard cap on ?lines= so a request can't slurp a huge log
_LOG_TAIL_WINDOW_BYTES = 256 * 1024  # bytes read from the log's END for a tail (2026-07-08)

# Serialises the read-merge-replace in _write_settings. ThreadingHTTPServer runs
# every POST on its own thread, so two concurrent /api/settings writes could each
# read the file, merge over their OWN stale copy, and replace — the second losing
# the first's update. This lock makes the whole read+merge+write atomic across
# threads (2026-07-08 finding).
_SETTINGS_WRITE_LOCK = threading.Lock()


class InsecureBindError(RuntimeError):
    """Raised by ``create_server`` when asked to bind a non-local address with an
    empty token. Refusing to start (rather than binding wide-open) is the whole
    point of the security contract, so this is a hard error the caller logs."""


# ── local-bind detection ────────────────────────────────────────────────────
_LOCAL_BINDS = {"127.0.0.1", "localhost", "::1", "127.0.0.1/32"}


def is_local_bind(bind: str) -> bool:
    """True when ``bind`` is a loopback address that nothing off-box can reach.
    Everything else (0.0.0.0, a LAN IP, a hostname) is treated as EXPOSED and so
    requires a token. Kept deliberately strict — an unknown value is 'exposed'."""
    return (bind or "").strip().lower() in _LOCAL_BINDS


# ── data sources (all graceful — a missing JARVIS/file degrades, never raises) ─

def _newest_log(log_dir: str) -> str | None:
    """Path to the most-recently-modified session_*.log, or None."""
    try:
        files = glob.glob(os.path.join(log_dir, "session_*.log"))
        if not files:
            return None
        return max(files, key=os.path.getmtime)
    except Exception:
        return None


# ANSI escape sequences leak into the session log from libraries that emit
# styled console output (ctranslate2/whisper notes print ESC[3m italics); the
# browser log panel rendered them as tofu ("⯑[3mNotes:"). Strip CSI sequences
# and any stray ESC before serving. (Audit finding 2026-07-10.)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _strip_ansi(line: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", line).replace("\x1b", "")


def tail_log(log_dir: str, lines: int, since: int | None = None,
             log_name: str | None = None) -> dict:
    """Return the last ``lines`` lines of the newest session log as a dict::

        {"log": "<basename or ''>", "lines": [...], "running": bool,
         "offset": <byte offset the next ?since= should use>,
         "append": bool}

    ``running`` is a best-effort liveness flag: the newest log was written to
    within the last 20 s (the loop logs whisper/vad activity constantly). Never
    raises — a missing logs dir yields an empty tail with running=False.

    INCREMENTAL MODE (2026-09-30 audit). With ``since`` (the ``offset`` a
    previous answer returned) and ``log_name`` still naming the newest log, only
    the COMPLETE lines written after that offset come back, with
    ``append: True``, so the page appends them instead of re-rendering the
    whole view every second (which wiped any text selection the owner was
    making). A rotated log, a shrunk file or a stale name falls back to a normal
    tail with ``append: False``."""
    lines = max(1, min(int(lines or 50), _LOG_TAIL_MAX_LINES))
    lg = _newest_log(log_dir)
    if not lg:
        return {"log": "", "lines": [], "running": False, "offset": 0,
                "append": False}
    if since is not None and log_name == os.path.basename(lg):
        inc = _tail_since(lg, int(since), lines)
        if inc is not None:
            return inc
    try:
        # Don't readlines() the ENTIRE (ever-growing) session log every 1s poll —
        # seek to a bounded tail window and split only that. 256KB comfortably
        # holds far more than _LOG_TAIL_MAX_LINES of normal log lines
        # (2026-07-08 finding). A line straddling the window boundary is dropped
        # via [-lines:] anyway, so a possibly-partial first line never surfaces.
        with open(lg, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - _LOG_TAIL_WINDOW_BYTES))
            chunk = f.read()
        # The offset handed back points just past the LAST COMPLETE line, so a
        # half-written final line is re-read whole by the next ?since= call.
        cut = chunk.rfind(b"\n") + 1
        offset = size - (len(chunk) - cut)
        text = chunk.decode("utf-8", errors="replace")
        tail = [_strip_ansi(l) for l in text.splitlines()[-lines:]]
    except Exception:
        return {"log": os.path.basename(lg), "lines": [], "running": False,
                "offset": 0, "append": False}
    return {
        "log": os.path.basename(lg),
        "lines": [ln.rstrip("\n") for ln in tail],
        "running": _log_running(lg),
        "offset": offset,
        "append": False,
    }


def _log_running(lg: str) -> bool:
    try:
        return (time.time() - os.path.getmtime(lg)) < 20.0
    except Exception:
        return False


def _tail_since(lg: str, since: int, lines: int) -> dict | None:
    """The complete lines written to ``lg`` after byte ``since``, or None when
    the offset no longer fits the file (rotated / truncated) so the caller
    falls back to a full tail. A burst larger than the tail window is capped to
    its last ``lines`` lines. Never raises."""
    try:
        size = os.path.getsize(lg)
        if since < 0 or since > size:
            return None
        start = max(since, size - _LOG_TAIL_WINDOW_BYTES)
        with open(lg, "rb") as f:
            f.seek(start)
            chunk = f.read(size - start)
        cut = chunk.rfind(b"\n") + 1          # complete lines only
        body = chunk[:cut]
        if start > since:                     # skipped bytes: drop the partial
            nl = body.find(b"\n")
            body = body[nl + 1:] if nl >= 0 else b""
        text = body.decode("utf-8", errors="replace")
        new = [_strip_ansi(l) for l in text.splitlines()][-lines:]
        return {"log": os.path.basename(lg), "lines": new,
                "running": _log_running(lg), "offset": start + cut,
                "append": True}
    except Exception:
        return None


def _read_hud_state(hud_state_path: str) -> dict:
    """Best-effort read of hud_state.json (empty dict on any error). Mirrors
    tray._read_hud_state — the same file the tray + HUD consume."""
    try:
        with open(hud_state_path, encoding="utf-8") as f:
            data = json.load(f)
        # Valid-but-non-object JSON (a list/number/string top level) would sail
        # past ``or {}`` and then .get() would AttributeError on every /api/status
        # poll — coerce anything that isn't a dict to {} (2026-07-08 finding).
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_version() -> str:
    """The shareable release string (VERSION file via core.version). Falls back
    to reading the VERSION file directly, then to 'unknown' — so a partial tree
    (or a test that imported us bare) still answers."""
    try:
        from core.version import version_string
        return version_string()
    except Exception:
        try:
            with open(os.path.join(PROJECT_DIR, "VERSION"), encoding="utf-8") as f:
                return f.read().strip() or "unknown"
        except Exception:
            return "unknown"


def _awake_state(hud: dict) -> str:
    """Map hud_state.json's sleep/standby flags into a single word for the strip.
    ``state`` is the canonical label the main loop writes ('Idle'/'Standby'/…);
    fall back to the boolean flags if it's absent."""
    state = hud.get("state")
    if isinstance(state, str) and state.strip():
        return state.strip()
    if hud.get("sleep_mode") or hud.get("standby_mode"):
        return "Asleep"
    return "Unknown"


def _gpu_summary() -> dict:
    """One-shot VRAM/model summary via core.gpu_usage, or a graceful stub.
    Returns ``{"lines": [...], "bar": "..."}``; on any failure (module missing on
    a cloud-only box, no GPU) returns empty lines so the strip shows 'GPU: n/a'."""
    try:
        from core import gpu_usage
        snap = gpu_usage.gpu_snapshot()
        return {"lines": gpu_usage.usage_lines(snap), "bar": gpu_usage.usage_bar(14, snap)}
    except Exception:
        return {"lines": [], "bar": ""}


def build_status(hud_state_path: str, log_dir: str) -> dict:
    """Assemble the /api/status payload from every (graceful) source."""
    hud = _read_hud_state(hud_state_path)
    lg = _newest_log(log_dir)
    running = False
    if lg:
        try:
            running = (time.time() - os.path.getmtime(lg)) < 20.0
        except Exception:
            running = False
    gpu = _gpu_summary()
    status = {
        "version": _read_version(),
        "state": _awake_state(hud),
        "running": running,
        # The REAL brain (2026-09-30 audit: this used to be last_intent_tag,
        # the model's last [intent:x] tag, shown under a "model" label). The
        # main loop publishes llm_backend ("anthropic" for Claude, else the
        # resolved local model tag) - the same field the tray's AI menu reads.
        "model": _model_label(hud),
        "llm_backend": str(hud.get("llm_backend") or ""),
        "last_intent_tag": str(hud.get("last_intent_tag") or ""),
        "routing": (_gpu_summary_routing(gpu)),
        "now_playing": hud.get("now_playing", ""),
        "last_spoken": hud.get("last_spoken", ""),
        "last_transcript": hud.get("last_transcript", ""),
        "gpu_lines": gpu["lines"],
        "gpu_bar": gpu["bar"],
        # Per-card VRAM (the gpu_lines TOTAL sums every card, which hid how full
        # the LLM card is behind a second, mostly idle one).
        "gpus": [{k: g.get(k) for k in ("index", "name", "mem_used_mb",
                                        "mem_total_mb", "util_pct")}
                 for g in _nvidia_smi_gpus()],
        # Uptime is None (→ omitted client-side) when no timestamped log exists; a
        # float of seconds otherwise. Kept as raw seconds so the client formats it.
        "uptime": _uptime_seconds(log_dir),
        "ts": time.time(),
    }
    status.update(_status_flags(hud))
    # air-mouse ARMED/ENGAGED is only present when the skill is loaded in THIS
    # process (see _air_mouse_status). Add the field ONLY when reachable so a bare
    # web process / headless CI simply omits it — the strip renders nothing for it
    # rather than a misleading "disarmed". This keeps the field's PRESENCE meaningful.
    am = _air_mouse_status()
    if am is not None:
        status["air_mouse"] = am
    return status


def _model_label(hud: dict) -> str:
    """Human label for the active brain from hud_state's ``llm_backend``:
    "Claude" for the anthropic backend, else the local model tag itself; ""
    when the loop has not published it yet."""
    b = str(hud.get("llm_backend") or "").strip()
    if not b:
        return ""
    return "Claude (cloud)" if b.lower() in ("anthropic", "claude") else b


def _status_flags(hud: dict) -> dict:
    """The control-plane flags the strip shows, straight from hud_state.json
    (the tray reads the same keys): awake / standby / sleep, mic + TTS mutes,
    paused daemons, and what the loop is doing right now. ``standby`` is True
    for BOTH sleep and wake-word standby - either way the main loop drops a
    typed command that does not start with the wake word."""
    sleep = bool(hud.get("sleep_mode"))
    standby = bool(hud.get("standby_mode"))
    state = str(hud.get("state") or "").strip().lower()
    asleep = sleep or standby or state in ("standby", "sleep", "sleeping")
    return {
        "awake": not asleep,
        "standby": asleep,
        "sleep_mode": sleep,
        "standby_mode": standby,
        "mic_muted": bool(hud.get("mic_muted")),
        "tts_muted": bool(hud.get("tts_muted")),
        "daemons_paused": bool(hud.get("daemons_paused")),
        "now_doing": str(hud.get("now_doing") or ""),
        "active_action": str(hud.get("active_action") or ""),
        # REQUIRE_WAKE_MODE as the RUNNING loop has it (published by
        # _act_wake_word_mode_set and at boot). None = not published (an
        # older JARVIS, or none running) - unknown, never reported as off.
        "require_wake_mode": (bool(hud["require_wake_mode"])
                              if "require_wake_mode" in hud else None),
    }


def _gpu_summary_routing(gpu: dict) -> str:
    """The routing line is the last entry of usage_lines() (chat→…  vision→…).
    Pulled out so the status payload carries a compact 'which brain' string."""
    lines = gpu.get("lines") or []
    for ln in reversed(lines):
        if "→" in ln:
            return ln
    return ""


def _air_mouse_status() -> dict | None:
    """The live air-mouse ARMED/ENGAGED flags, or None when not cheaply reachable.

    WHY sys.modules (no new import)
    ===============================
    The web interface runs IN-PROCESS with the JARVIS main loop (skills/web_interface
    imports tools.web_interface and calls create_server() inside the running process),
    so the air-mouse skill — when loaded — already lives in ``sys.modules`` under the
    key ``skill_kinect_air_mouse``. We therefore read its thread-safe getter the
    EXACT way bobert_companion._air_mouse_state_for_preview() does: fetch the module
    object from sys.modules (never import it — importing the skill standalone is
    heavy and would drag Kinect/pyautogui deps into a bare-CI web process) and call
    its ``get_air_mouse_state()`` if present. That getter returns a COPY of
    ``{'engaged': bool, 'armed': bool, 'hand': str|None, 'grip': str, ...}``.

    Returns a trimmed ``{"armed": bool, "engaged": bool}`` when the skill is loaded
    and readable, or ``None`` when it isn't — headless CI, a cloud-only box with no
    Kinect, or simply the air-mouse skill not being part of this build. ``None`` lets
    build_status OMIT the field entirely (the strip then shows nothing for it) rather
    than lying with a fabricated "disarmed", which would be indistinguishable from a
    genuinely-disarmed air-mouse. NEVER raises — every failure path degrades to None.
    """
    try:
        sk = sys.modules.get("skill_kinect_air_mouse")
        getter = getattr(sk, "get_air_mouse_state", None) if sk else None
        if callable(getter):
            st = getter()
            if isinstance(st, dict):
                return {"armed": bool(st.get("armed")),
                        "engaged": bool(st.get("engaged"))}
    except Exception:
        pass
    return None


# Boot time is recovered from the newest session log, preferring the FULL
# date+time encoded in its FILENAME (the loop names each log
# session_%Y-%m-%d_%H-%M-%S.log at boot — see bobert_companion's log setup), so
# the uptime chip stays correct across midnight. A hand-named/legacy log falls
# back to the "[HH:MM:SS]" clock of the first timestamped line (the loop
# timestamps every line): that heuristic is DATE-LESS, so it computes a same-day
# delta and clamps a negative (crossed-midnight / clock-skew) result to 0 rather
# than reporting a bogus ~24h uptime.
_LOG_TS_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]")
_LOG_NAME_TS_RE = re.compile(
    r"session_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})\.log$")


def _uptime_seconds(log_dir: str) -> float | None:
    """Best-effort session uptime in seconds, or None when not derivable.

    Preferred source: the newest log's FILENAME timestamp
    (session_%Y-%m-%d_%H-%M-%S.log) — a full date, so the epoch delta survives
    midnight. Fallback for an unparseable name: the "[HH:MM:SS]" prefix of the
    first line that HAS one (the loop's banner lines may precede the first
    timestamp) subtracted from the current wall clock's H:M:S — cheap but
    date-less, so it clamps a negative (crossed-midnight) delta to 0. This is a
    status nicety, not a billing meter. Returns None (→ field omitted) when
    there is no log, no parseable timestamp, or anything at all goes wrong.
    NEVER raises."""
    lg = _newest_log(log_dir)
    if not lg:
        return None
    try:
        m = _LOG_NAME_TS_RE.search(os.path.basename(lg))
        if m:
            y, mo, dy, hh, mm, ss = (int(g) for g in m.groups())
            boot = time.mktime((y, mo, dy, hh, mm, ss, 0, 0, -1))
            delta = time.time() - boot
            if delta >= 0:
                return float(delta)
            # A negative filename delta (clock skew / a future-dated hand-named
            # file) falls through to the head-scan heuristic below.
        first_hms = None
        with open(lg, encoding="utf-8", errors="replace") as f:
            # Only scan a bounded head of the file — the boot timestamp is in the
            # first handful of lines; we never want to read a multi-MB log to find it.
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                m = _LOG_TS_RE.search(line)
                if m:
                    first_hms = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    break
        if first_hms is None:
            return None
        now = time.localtime()
        boot_s = first_hms[0] * 3600 + first_hms[1] * 60 + first_hms[2]
        now_s = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
        delta = now_s - boot_s
        # Crossed midnight (or a clock skew) → negative; clamp to 0 rather than
        # reporting a spurious ~day of uptime. A same-day session reads correctly.
        return float(delta) if delta >= 0 else 0.0
    except Exception:
        return None


# ── inject channel (append with the SAME atomic pattern the loop drains) ─────

# Serialises the read-modify-write of every JSON command queue this server
# appends to (the inject queue AND tray_commands.json). ThreadingHTTPServer runs
# each POST on its own thread; without this two concurrent posts could both
# read the same list, each append their own item, and the second os.replace
# would silently DROP the first command (2026-09-30 audit, the same class the
# _SETTINGS_WRITE_LOCK below fixed for settings writes). The monolith's
# drainer claims the file by RENAME, which is atomic against our replace.
_QUEUE_WRITE_LOCK = threading.Lock()


def _append_json_queue(path: str, item: dict, *, prefix: str,
                       indent=None) -> None:
    """Append ``item`` to the JSON list at ``path`` - read, append, write a
    temp file in the same dir, os.replace - under _QUEUE_WRITE_LOCK. A
    missing / corrupt / non-list file starts a fresh list (the drainer may have
    just renamed it away). Raises on a failed write (the caller reports it)."""
    with _QUEUE_WRITE_LOCK:
        items: list = []
        try:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    raw = f.read().strip()
                if raw:
                    decoded, _end = json.JSONDecoder().raw_decode(raw)
                    if isinstance(decoded, list):
                        items = decoded
        except Exception:
            items = []
        items.append(item)
        _dir = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(dir=_dir, suffix=".tmp", prefix=prefix)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=indent)
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            raise


def inject_command(text: str, inject_path: str) -> None:
    """Append ``{"text": text, "ts": ...}`` to the inject queue atomically.

    Read-modify-write under a fresh temp + os.replace so a concurrent
    ``_drain_injected_command`` (which claims the file by renaming it) never sees
    a half-written array, and under _QUEUE_WRITE_LOCK so two concurrent posts
    can neither lose nor duplicate an item. If the queue was mid-consume
    (renamed away) we simply start a fresh list — the loop will drain ours next
    pass. Matches driver.py's ``inject`` and staging_instance's writer."""
    _append_json_queue(inject_path, {"text": text, "ts": time.time()},
                       prefix=".webinject_", indent=2)


# The tray control plane (tray.py -> tray_commands.json -> the monolith's 2 Hz
# _drain_tray_commands_once -> _dispatch_tray_command). It keeps draining in
# STANDBY and runs BEFORE any LLM call, which is why the dashboard's controls go
# here rather than through the inject channel. ONLY these commands are
# accepted from the web (checked against _dispatch_tray_command's own branches);
# the ones in _TRAY_CONFIRM need "confirm": true.
TRAY_WEB_COMMANDS = ("force_wake", "enter_standby", "mute_tts_toggle",
                     "mic_mute_toggle", "pause_daemons_toggle", "restart",
                     # The pinned wake-word switch (2026-10-01): the owner's
                     # "wake-word mode" is REQUIRE_WAKE_MODE, applied live by
                     # the same _act_wake_word_mode_set the voice command runs.
                     "wake_word_mode_on", "wake_word_mode_off")
_TRAY_CONFIRM = frozenset({"restart"})
DEFAULT_TRAY_COMMANDS_PATH = os.path.join(PROJECT_DIR, "tray_commands.json")


def send_tray_command(cmd: str, tray_path: str, **extra) -> None:
    """Append ``{"cmd": cmd, "ts": ..., "cid": ..., **extra}`` to
    tray_commands.json with the SAME read-append-temp-replace the tray uses
    (tray._send_command), under _QUEUE_WRITE_LOCK. Raises on a failed write.

    Every entry carries a unique ``cid``, exactly like the tray's own. The
    drainer claims the inbox with an os.replace that this lock cannot see, so
    a claim landing between our read and our replace makes us write the
    already-claimed commands BACK; _tray_cid_seen skips a cid it has already
    dispatched, but an entry WITHOUT one is never de-duplicated - a web
    mic_mute_toggle ran twice and flipped straight back (2026-10-01)."""
    item = {"cmd": cmd, "ts": time.time(), "source": "web",
            "cid": "w" + uuid.uuid4().hex}
    item.update(extra)
    _append_json_queue(tray_path, item, prefix="tray_web_")


# ── reply capture: what JARVIS actually said for ONE injected turn ───────────
# The loop's lines for a turn look like this (live session log 2026-09-29):
#
#   [22:07:44]   [inject] what time is it
#   [22:07:45]   JARVIS: [ACTION: get_time] One moment, sir.      <- lead-in
#   [22:07:45]   [action] get_time: current time is 10:07 PM ...  <- result
#   [22:07:51]   JARVIS: It is 10:07 PM, sir.                     <- the answer
#   [22:07:55]   [turn-timing] kind=inject outcome=ok ...         <- turn over
#   [22:07:55] Listening…
#
# The old capture returned ~1 s after the FIRST "JARVIS:" line, so the page
# showed "[ACTION: get_time] One moment, sir." and never the answer (2026-09-30
# audit, P0). A fallback that replaces the model's words prints
# "JARVIS (spoken): ..." after the model's line, and THAT is what was said.
_LOG_TS_PREFIX_RE = re.compile(r"^\s*\[\d{1,2}:\d{2}:\d{2}\]\s*")
_REPLY_TAG_RE = re.compile(r"\[\s*(?:ACTION|intent)\s*:[^\]]*\]", re.I)
_TURN_END_RE = re.compile(r"\[turn-timing\]\s+kind=inject\b", re.I)
_LOOP_IDLE_RE = re.compile(r"^(?:Listening|Standby|Sleeping)\b", re.I)
_STANDBY_DROP_RE = re.compile(r"^\[(?:standby|sleeping)\]\s+ignored\b", re.I)


def _log_body(line: str) -> str:
    """A log line without its "[HH:MM:SS]" stamp and indentation."""
    return _LOG_TS_PREFIX_RE.sub("", line or "").strip()


def clean_reply_text(text: str) -> str:
    """Display form of a reply: [ACTION: ...] / [intent: ...] tags removed and
    whitespace collapsed."""
    return re.sub(r"\s{2,}", " ", _REPLY_TAG_RE.sub("", text or "")).strip()


def parse_turn_lines(lines) -> dict:
    """Fold one turn's log lines (everything AFTER its [inject] anchor) into
    ``{"reply", "lines", "actions", "spoken", "ended", "standby"}``.

    reply: what JARVIS said, for display -
      * a "JARVIS (spoken): X" line REPLACES the model line it follows;
      * a model line that carried an [ACTION: ...] tag is a LEAD-IN: when the
        turn produced anything after it (a follow-up answer, or the action's
        result for a verbatim action) the lead-in is not the reply;
      * tags are stripped; a turn with no words at all gives ''.
    lines: every JARVIS / [action] line, cleaned, in order (the transcript).
    ended: the turn's own end marker was seen ([turn-timing] kind=inject, or
    the loop going back to Listening / Standby). standby: the loop ignored the
    command because JARVIS is asleep. Pure; never raises."""
    said = []            # [(text, is_lead_in)]
    actions = []
    shown = []
    spoken = False
    ended = False
    standby = False
    for raw in lines or ():
        body = _log_body(raw)
        if not body:
            continue
        low = body.lower()
        if _TURN_END_RE.search(body) or _LOOP_IDLE_RE.match(body):
            ended = True
            break
        if _STANDBY_DROP_RE.match(body):
            standby = True
            ended = True
            break
        if low.startswith("jarvis (spoken):"):
            txt = clean_reply_text(body.split(":", 1)[1])
            spoken = True
            if said:
                said[-1] = (txt, False)       # it REPLACES the model's line
            else:
                said.append((txt, False))
            shown.append("JARVIS (spoken): " + txt)
        elif low.startswith("jarvis:"):
            rest = body.split(":", 1)[1]
            lead = bool(re.search(r"\[\s*action\s*:", rest, re.I))
            txt = clean_reply_text(rest)
            said.append((txt, lead))
            if txt:
                shown.append("JARVIS: " + txt)
        elif low.startswith("[action]"):
            res = body[len("[action]"):].strip()
            name, _, out = res.partition(":")
            actions.append({"name": name.strip(), "result": out.strip()})
            shown.append("[action] " + res)
    answers = [t for t, lead in said if t and not lead]
    if answers:
        reply = "\n".join(answers)
    elif actions and any(a["result"] for a in actions):
        reply = "\n".join(a["result"] for a in actions if a["result"])
    else:
        reply = "\n".join(t for t, _lead in said if t)
    return {"reply": reply, "lines": shown, "actions": actions,
            "spoken": spoken, "ended": ended, "standby": standby}


def wait_for_reply(text: str, log_dir: str, timeout: float) -> dict:
    """Tail the newest session log from its current end and return what JARVIS
    said for THIS injected turn.

    Returns ``{"status", "lines", "reply", "actions", "spoken"}``:
      • status ok       — the turn ran and its end marker was seen (or reply
                          lines were captured before the timeout);
               standby  — JARVIS is asleep and IGNORED the command (the loop
                          logged "[standby] ignored"); nothing ran;
               accepted — injected, but nothing captured within the timeout
                          (the command may still run; e.g. a pure side effect);
               no_log   — no session log exists (JARVIS isn't running); the
                          command stays queued for the next boot.
      • reply — the ANSWER, not the lead-in (see parse_turn_lines).

    The capture STARTS at this command's "[inject] <text>" anchor, so another
    turn's output is never scraped, and ENDS at this turn's own
    "[turn-timing] kind=inject" line (or the loop's next "Listening…"), which
    the main loop prints once per injected turn after everything was said. A
    second "[inject]" line (the next queued command) also ends it."""
    lg = _newest_log(log_dir)
    if not lg:
        return {"status": "no_log", "lines": [], "reply": ""}
    try:
        pos = os.path.getsize(lg)
    except Exception:
        return {"status": "no_log", "lines": [], "reply": ""}
    snippet = (text or "")[:30].lower()
    saw_inject = False
    turn: list[str] = []
    buf = ""
    deadline = time.time() + max(1.0, min(float(timeout), _REPLY_TIMEOUT_MAX))
    while time.time() < deadline:
        time.sleep(0.25)
        try:
            with open(lg, encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except Exception:
            continue
        buf += chunk
        if "\n" not in buf:
            continue
        complete, buf = buf.rsplit("\n", 1)      # keep a half-written line
        for line in complete.split("\n"):
            low = line.lower()
            if "[inject]" in low:
                if not saw_inject and snippet and snippet in low:
                    saw_inject = True
                    continue
                if saw_inject:                   # the NEXT command started
                    turn.append("[turn-timing] kind=inject (next command)")
                    continue
            if saw_inject:
                turn.append(line)
        if saw_inject:
            res = parse_turn_lines(turn)
            if res["ended"]:
                status = "standby" if res["standby"] else "ok"
                return {"status": status, "lines": res["lines"],
                        "reply": res["reply"], "actions": res["actions"],
                        "spoken": res["spoken"]}
    res = parse_turn_lines(turn)
    got = bool(res["lines"])
    return {"status": "ok" if got else "accepted", "lines": res["lines"],
            "reply": res["reply"], "actions": res["actions"],
            "spoken": res["spoken"]}


# ── settings bridge (the FULL settings control panel) ───────────────────────
#
# WHY THIS EXISTS
# ===============
# The owner wants to "do it all from the web interface" — every user-facing
# config knob, including the wake-word mode toggle. Rather than re-declare those
# knobs here (they'd drift from the real config), we treat tools/settings_window's
# ``SCHEMA`` as the SINGLE SOURCE OF TRUTH: it already enumerates every
# user-facing knob keyed by name → {tab, label, type, default, help, choices}.
# We import it, render every persisted key, READ the CURRENT effective value from
# ``core.config`` (the constants _apply_user_settings() overrode at import), and
# WRITE changes back to data/user_settings.json (the same file the Settings GUI
# and _apply_user_settings() share).
#
# DEPENDENCY-LIGHT / GRACEFUL, like the rest of this module
# ========================================================
# Both imports are LAZY (inside the functions) and TOLERANT: a bare-CI import of
# web_interface never needs settings_window or core.config, and if either can't
# load we degrade (empty schema / schema-default value) instead of raising into a
# request handler — mirroring how _read_version / _gpu_summary already probe
# core.* lazily. settings_window's schema half is stdlib-only (the tkinter GUI is
# below its "# ── GUI ──" divider and imported lazily there), so importing it for
# SCHEMA/coerce_value costs nothing on a headless box.


def _load_settings_schema():
    """Import tools.settings_window and return ``(SCHEMA, coerce_value)``, or
    ``({}, None)`` if it can't load. Lazy + tolerant: importing web_interface must
    not require settings_window, and a broken/absent schema degrades to an empty
    panel rather than breaking every request. Never raises."""
    try:
        from tools import settings_window as sw
        return sw.SCHEMA, sw.coerce_value
    except Exception:
        return {}, None


def _config_value(key: str, default):
    """The CURRENT effective value of a config constant, read LIVE from
    ``core.config`` (the value _apply_user_settings() left after merging
    user_settings.json over the module defaults at import). Falls back to the
    schema ``default`` when core.config can't be imported (bare CI) or the
    constant is absent. Lazy import so a headless web process never drags the
    monolith's config in unless a settings request actually asks for it.

    NOTE: core.config is imported ONCE and then cached in sys.modules, so its
    constants reflect the values as of THAT import — which is boot time in a live
    JARVIS. A settings write does NOT mutate the running process's constants (see
    the restart caveat in _write_settings), so what we report here is the
    effective value the CURRENTLY-RUNNING loop is using. build_settings_schema
    overlays the saved user_settings.json on top of this for keys the owner has
    actually saved (with an honest ``pending_restart`` flag), so the panel shows
    the durable record instead of appearing to revert every save."""
    try:
        from core import config as _config
        return getattr(_config, key, default)
    except Exception:
        return default


# Sentinel distinguishing "core.config has NO such constant" (a GUI-only key
# like OBS_HOST_HINT, or bare CI where core.config can't import) from a
# constant whose value is legitimately falsy. build_settings_schema passes it
# as _config_value's default so those two cases route differently.
_NO_CONSTANT = object()


def _read_saved_settings(path: str | None) -> dict:
    """The raw on-disk user_settings.json as a dict — the durable record the
    settings write path maintains. ``path=None`` resolves the live path via
    settings_window.settings_path() (honouring the JARVIS_SETTINGS_PATH /
    staging redirects). Tolerant like _write_settings' reader: a missing,
    corrupt, or non-object file — or an unresolvable path on bare CI — degrades
    to ``{}`` (no overlay). Never raises."""
    if path is None:
        try:
            from tools import settings_window as sw
            path = sw.settings_path()
        except Exception:
            return {}
    try:
        # The Settings window's strict reader: utf-8-sig (a PowerShell BOM is
        # fine), JSON errors named. This overlay is read-only, so any problem
        # still degrades to {} here; the WRITE path refuses instead.
        from tools import settings_window as sw
        return sw.read_settings_file(path)
    except Exception:
        return {}


# Knobs whose VALUE is a secret. We render a row for them (so the owner can SET
# one from the panel) but we NEVER echo the current value back in the GET payload
# — the settings snapshot returns "" + secret/is_set flags instead of the live
# token. Reading /api/settings is already token-gated, so this isn't the only
# barrier, but echoing a live secret into a visible text field (shoulder-surfing,
# an accidental screen-share, a proxy log) is a footgun with no upside. The write
# path is unaffected: a POST can still set a new token.
_SECRET_SETTING_KEYS = frozenset({"WEB_INTERFACE_TOKEN"})


def build_settings_schema(settings_path: str | None = None) -> dict:
    """Assemble the /api/settings GET payload: every PERSISTED schema knob with
    its current value. Shape::

        {"settings": [ {name, tab, label, type, choices, help, value, default,
                        pending_restart?, secret?, is_set?}, ... ],
         "tabs": ["voice","ai","privacy","integrations","advanced"],
         "note": "…applies on restart…"}

    Read LIVE each call (no caching) so the panel always reflects the file/loop
    state. ``settings_path`` is the user_settings.json the write path targets —
    the GET handler passes cfg["user_settings_path"], so a GET reads the EXACT
    file the POST just wrote; None resolves the live path via settings_window.

    VALUE SOURCING (2026-07-21 fix — the panel used to echo only the boot-time
    core.config snapshot, so every save appeared to revert on the re-fetch, and
    GUI-only keys with no constant could never display at all):
      • key PRESENT in the saved file → the FILE value (what the owner saved),
        coerced through the same schema rules the write path applies, plus
        ``pending_restart: True`` when it differs from the live core.config
        constant — honest that the running loop lags the file until a restart;
      • key ABSENT from the file, live constant exists → that constant (the
        value the currently-running loop is using);
      • neither (a GUI-only key like OBS_HOST_HINT with no core.config
        constant, or bare CI where core.config can't import) → the schema
        default.
    We overlay ONLY file-present keys — NOT settings_window.load_settings(),
    which backfills every missing key with the SCHEMA default — so an unsaved
    knob keeps reporting the live constant even where that differs from the
    schema default.

    Status-only rows (keys starting with "_status_", type "status") are
    SKIPPED — they expose integration presence in the GUI but carry no persisted
    value and (deliberately) never surface a secret, so they have no place in a
    write-capable web panel. SECRET knobs (``_SECRET_SETTING_KEYS``) are rendered
    but their value is REDACTED to "" (with ``secret: True`` and ``is_set`` —
    True when the RUNNING loop has a value OR the file has one saved, so a
    panel-saved token registers immediately and a cleared-but-not-restarted one
    stays truthfully "set") so the live token never leaves the process. Never
    raises: an unloadable schema yields an empty list."""
    schema, _coerce = _load_settings_schema()
    saved = _read_saved_settings(settings_path)
    items: list[dict] = []
    tabs: list[str] = []
    for name, spec in schema.items():
        typ = spec.get("type")
        # Only real persisted knobs — skip the read-only integration status rows.
        if typ == "status" or name.startswith("_"):
            continue
        tab = spec.get("tab", "advanced")
        if tab not in tabs:
            tabs.append(tab)
        default = spec.get("default")
        live = _config_value(name, _NO_CONSTANT)
        if name in saved:
            # The durable record the owner saved. Coerce through the SAME rules
            # the write path applies so a hand-edited file still renders sanely.
            try:
                value = _coerce(spec, saved[name]) if _coerce else saved[name]
            except Exception:
                value = saved[name]
        elif live is not _NO_CONSTANT:
            value = live
        else:
            value = default
        row = {
            "name": name,
            "tab": tab,
            "label": spec.get("label", name),
            "type": typ,
            # choices only present for enum/combo/routing — omit when absent so
            # the client can rely on truthiness.
            "choices": spec.get("choices"),
            "help": spec.get("help", ""),
            "value": value,
            "default": default,
        }
        if name in saved:
            # Honest divergence flag: the file says one thing, the running
            # loop's constant another → the save applies on the next restart.
            row["pending_restart"] = bool(live is not _NO_CONSTANT
                                          and value != live)
        if name in _SECRET_SETTING_KEYS:
            # Redact: report only WHETHER a value is set, never the value itself.
            # The client renders a password field and (on an empty save) leaves
            # the existing secret untouched — see saveSetting. "Set" means the
            # running loop OR the saved file carries a non-empty value: a token
            # saved via the panel counts immediately, and one cleared in the
            # file stays set until the restart actually drops it from the loop.
            live_set = (live is not _NO_CONSTANT
                        and bool(live and str(live).strip()))
            file_set = name in saved and bool(str(saved[name] or "").strip())
            row["is_set"] = bool(live_set or file_set)
            row["value"] = ""
            row["default"] = ""
            row["secret"] = True
        items.append(row)
    return {
        "settings": items,
        "tabs": tabs,
        "note": SETTINGS_RESTART_NOTE,
    }


# The honest caveat we return on every write. _apply_user_settings() runs ONCE at
# core.config import (boot), so a saved value overrides the module constant only
# on the NEXT JARVIS start. We do NOT claim a live-apply we can't guarantee —
# some knobs are re-read live by their consumers, but many are import-time, so the
# safe, truthful blanket statement is "applies on restart".
SETTINGS_RESTART_NOTE = ("Saved. Most settings take effect the next time JARVIS "
                         "restarts.")


class SettingsWriteError(ValueError):
    """Raised by _coerce_setting/_write_settings on an unknown key or a value that
    can't be coerced to the schema type. The POST handler turns it into a 400 with
    this message — a clear, actionable error rather than a silent drop."""


def _coerce_setting(name: str, value, schema: dict, coerce_value) -> object:
    """Validate ``name`` against the schema and coerce ``value`` to its declared
    type, raising ``SettingsWriteError`` on an unknown key or a bad value.

    We reuse settings_window.coerce_value for the actual type conversion so the
    web panel and the GUI apply IDENTICAL coercion rules (bool truthiness, enum
    membership, int/float parsing, text→list, routing merge). But coerce_value is
    deliberately LENIENT — it falls back to the default rather than raising — so a
    web caller that fat-fingers an enum would silently write the default and think
    it succeeded. That's wrong for an API, so we add a STRICT pre-check for the two
    cases a user most wants an error on:
      • unknown key                → 400 (typo / stale client)
      • enum value not in choices  → 400 (invalid choice)
    int/float that won't parse also 400 (coerce_value would swallow it to the
    default).

    Then the value goes through settings_window.validate_value — the Settings
    window's OWN strict rule (range / min_exclusive / forbid / nonblank / NaN).
    This path used to stop at "does it parse", a stale copy of the rule the
    window tightened in v2.0.146: the panel said "saved" for WEB_INTERFACE_PORT
    8443 (the AirTag tracker's port) or 70000 — the dashboard then could not
    bind after the restart — VAD_THRESHOLD 5 (deaf) or 0 (never stops
    recording), and a blank LOCAL_LLM_MODEL (2026-10-01). One rule for both
    UIs now; validate_value hands bool / text / routing to coerce_value."""
    spec = schema.get(name)
    if spec is None or spec.get("type") == "status" or name.startswith("_"):
        raise SettingsWriteError(f"unknown setting: {name!r}")
    typ = spec.get("type")
    # Enum: must be one of the declared choices — reject rather than default.
    if typ == "enum":
        choices = spec.get("choices") or []
        if str(value) not in choices:
            raise SettingsWriteError(
                f"invalid value for {name!r}: {value!r} is not one of {choices}")
    # int/float: coerce_value swallows a bad parse to the default, so pre-validate
    # here to surface a real 400 instead of a silent wrong write.
    if typ in ("int", "float"):
        try:
            (int if typ == "int" else float)(value)
        except (TypeError, ValueError):
            raise SettingsWriteError(
                f"invalid {typ} value for {name!r}: {value!r}")
    # device (e.g. MICROPHONE_INDEX): coerce_value swallows a bad index to the
    # default (None) and the API would falsely report success. Pre-validate the
    # same way int/float do — but None / "" are LEGAL here (they mean 'auto'), so
    # only a present, non-empty value must parse as int (2026-07-08 finding).
    if typ == "device":
        empty = value is None or (isinstance(value, str) and value.strip() == "")
        if not empty:
            try:
                int(value)
            except (TypeError, ValueError):
                raise SettingsWriteError(
                    f"invalid device value for {name!r}: {value!r}")
    if coerce_value is None:                       # schema loaded but no coercer
        raise SettingsWriteError("settings coercion unavailable")
    # List/dict-valued settings ('routing', or a 'text'/list knob): the web panel
    # renders them as a single text input holding a JSON string. coerce_value's
    # 'text' branch would splitlines() that JSON into a bogus 1-element list, and
    # its 'routing' branch would reset a non-dict to the default — silently
    # corrupting the save (2026-07-08 finding). Parse the JSON back to the real
    # container FIRST so coerce_value gets the list/dict it accepts. Non-JSON
    # strings (a genuine newline-separated 'text' list) fall through untouched.
    if isinstance(value, str):
        default = spec.get("default")
        wants_list = typ == "text" or isinstance(default, list)
        wants_dict = typ == "routing" or isinstance(default, dict)
        s = value.strip()
        if (wants_list and s.startswith("[")) or (wants_dict and s.startswith("{")):
            try:
                parsed = json.loads(s)
                if (isinstance(parsed, list) and wants_list) or \
                   (isinstance(parsed, dict) and wants_dict):
                    value = parsed
            except (ValueError, TypeError):
                pass
    try:
        from tools import settings_window as sw
        validate_value = sw.validate_value
    except Exception:                              # pragma: no cover
        validate_value = None
    if validate_value is None:
        return coerce_value(spec, value)
    coerced, err = validate_value(spec, value)
    if err:
        raise SettingsWriteError(f"invalid value for {name!r}: {err}")
    return coerced


def _log_warn(msg: str) -> None:
    """One-line stderr warning for a settings write that DEGRADED.

    Per-request logging is deliberately silenced (see JarvisHandler.log_message
    — it would spam the session log we tail), but a settings write that quietly
    did less than it should is exactly what must never be silent. Settings
    writes are rare, so this cannot spam. Never raises."""
    try:
        print(f"[web_interface] {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _lockstep_vision(current: dict, new_chat) -> str | None:
    """The LOCAL_VISION_MODEL tag this settings write must ALSO set, or None.

    POST /api/settings can repoint LOCAL_LLM_MODEL, which makes the dashboard a
    model-switch entry point — but until 2026-08-20 it applied none of the
    chat↔vision lockstep the voice switch does, so a dashboard save forked the
    shipped one-multimodal-brain config onto a real second VLM (and neither UI
    could repair it afterwards: the voice path reads the mismatch as a pinned
    VLM). The RULE lives once in core.model_lockstep; this only resolves the
    inputs (the pre-merge document holds the OLD chat tag, and a key the
    document omits is still live at its core/config.py value).

    LOCAL_VISION_MODEL has no SCHEMA row, so _coerce_setting would 400 it —
    the caller merges the returned tag into the document directly, exactly the
    way settings_window's Save writes it as a passthrough key. Never raises."""
    try:
        from core.model_lockstep import (LOCKSTEP_TEXT_ONLY, config_default,
                                         vision_lockstep_decision)
    except Exception as exc:
        _log_warn(f"vision lockstep unavailable ({exc}) — a chat-model write "
                  f"will NOT move LOCAL_VISION_MODEL")
        return None
    try:
        old_chat = current.get("LOCAL_LLM_MODEL") or config_default(
            "LOCAL_LLM_MODEL")
        cur_vision = (current.get("LOCAL_VISION_MODEL")
                      or config_default("LOCAL_VISION_MODEL"))
        tag, reason = vision_lockstep_decision(old_chat, new_chat, cur_vision)
        if reason == LOCKSTEP_TEXT_ONLY:
            _log_warn(f"chat model set to {new_chat!r}, which is not vision-"
                      f"capable — local vision stays on {cur_vision!r}")
        return tag
    except Exception as exc:                       # pragma: no cover
        _log_warn(f"vision lockstep skipped: {exc}")
        return None


def _write_settings(updates: dict, path: str) -> dict:
    """MERGE ``updates`` (name→value) into the user_settings.json at ``path``,
    atomically, preserving every other key already in the file.

    Returns the ``{name: coerced_value, ...}`` actually applied. Raises
    ``SettingsWriteError`` if ANY update is invalid (unknown key / bad type) —
    validation happens for ALL updates BEFORE we touch disk, so a bad key in a
    batch never leaves a half-applied file.

    One key can be written that the caller did NOT send: repointing
    LOCAL_LLM_MODEL carries LOCAL_VISION_MODEL with it when the two share the
    one multimodal brain (see _lockstep_vision). It is included in the return
    value so the response names every key this write changed.

    ATOMICITY / PRESERVATION
    ========================
    We read the existing file, overlay ONLY the validated keys, and write via a
    fresh temp file + os.replace in the SAME directory — the identical crash-safe
    pattern as inject_command() above and settings_window.atomic_write_json /
    _apply_user_settings' reader. A concurrent reader (the Settings GUI, or JARVIS
    booting) therefore never observes a half-written document, and any key we
    didn't touch (including keys no schema knows about, e.g. a newer JARVIS's
    extra knobs) is preserved verbatim. We do NOT rewrite the full default
    template — a targeted merge is the whole contract ("preserve all other keys").

    RESTART CAVEAT: this writes the FILE only. core.config's live constants were
    set at import and are not mutated here, so the change reaches the running loop
    on its next restart (see SETTINGS_RESTART_NOTE)."""
    schema, coerce_value = _load_settings_schema()
    if not schema:
        raise SettingsWriteError("settings schema unavailable")
    # 1) Validate + coerce EVERYTHING first (fail closed before any disk write).
    applied: dict = {}
    for name, value in updates.items():
        applied[name] = _coerce_setting(name, value, schema, coerce_value)
    # 2+3) Read → merge → atomic-replace, serialised across threads so two
    #      concurrent writers can't each merge over a stale copy and drop the
    #      other's update (2026-07-08 finding). Validation above is pure and stays
    #      outside the lock to keep the critical section short.
    with _SETTINGS_WRITE_LOCK:
        # 2) Read the current file (tolerant: missing/corrupt → start from {}), so
        #    we MERGE over it and preserve keys we don't manage.
        # A file that can't be read must never be "merged" over as if empty:
        # that silently wiped every key this panel doesn't manage (CAMERAS,
        # KINECT_*, AUDIO_AUTOSWITCH_*, ...). Same rule as the Settings window
        # (2026-09-30): refuse, name the problem, leave the file alone.
        try:
            from tools import settings_window as sw
            current = sw.read_settings_file(path)
        except Exception as exc:
            raise SettingsWriteError(
                f"user_settings.json can't be read ({exc}); fix or restore "
                f"it first - nothing was saved") from exc
        # Vision LOCKSTEP — must happen HERE, inside the lock and BEFORE the
        # merge, because `current` is the only place the OLD chat tag still
        # exists (the rule's precondition is "vision currently denotes the old
        # chat tag"). Reported back in `applied` so the API answer names every
        # key this write actually changed. 2026-08-20 audit.
        if "LOCAL_LLM_MODEL" in applied:
            synced = _lockstep_vision(current, applied["LOCAL_LLM_MODEL"])
            if synced:
                applied["LOCAL_VISION_MODEL"] = synced
        current.update(applied)
        # 3) Atomic write (temp in the same dir + os.replace) — never a partial file.
        _dir = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=_dir, suffix=".tmp", prefix=".websettings_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            raise
    return applied


def _default_user_settings_path() -> str:
    """The live user_settings.json path — resolved from settings_window so the web
    panel writes the EXACT file the Settings GUI and core.config._apply_user_settings
    read (data/user_settings.json under the project root, honouring the
    JARVIS_SETTINGS_PATH redirect). Falls back to the known relative location if
    settings_window can't be imported, so create_server always has a concrete
    default. Never raises."""
    try:
        from tools import settings_window as sw
        return sw.settings_path()
    except Exception:
        return os.path.join(PROJECT_DIR, "data", "user_settings.json")


# ── control-panel data sources (System / Actions / Voice / Camera / Memory) ──
#
# Every function here is READ-ONLY and GRACEFUL — the exact contract the rest of
# this module honours: a missing tool / module / file degrades to an empty-but-
# valid payload, never an exception into a request handler. The GET endpoints
# that call them (do_GET) each add their own try/except belt on top. All heavy
# imports (psutil, core.voice_clone, core.long_term_memory) are LAZY so importing
# web_interface stays dependency-light and a bare-CI/cloud box just degrades.


def _smi_num(v):
    """Parse an nvidia-smi CSV cell to a number, or None. '[N/A]'/'' → None. A
    value containing a dot becomes a float (power.draw is fractional); otherwise
    an int. Never raises."""
    try:
        s = str(v).strip()
        if not s or s.lower().startswith("[n/a") or s.lower() == "n/a":
            return None
        return float(s) if "." in s else int(s)
    except Exception:
        return None


_NVIDIA_SMI_TTL = 2.0             # seconds a nvidia-smi snapshot is reused for
_nvidia_smi_cache: dict = {"ts": 0.0, "gpus": []}
_nvidia_smi_lock = threading.Lock()


def _nvidia_smi_gpus() -> list:
    """Per-GPU stats via a single nvidia-smi CSV query, or ``[]`` on ANY failure
    (no GPU, no driver, cloud-only box, command missing/timeout). Never raises.
    Uses CREATE_NO_WINDOW on Windows (read via getattr so the CI Linux-sim, which
    deletes that attribute, doesn't trip an AttributeError).

    TTL-CACHED (2026-07-08 finding): /api/system polls every ~2s and each call
    used to spawn a fresh nvidia-smi subprocess. We now reuse the last result for
    _NVIDIA_SMI_TTL seconds so a fast poller can't fork a subprocess per request."""
    now = time.time()
    with _nvidia_smi_lock:
        if (now - _nvidia_smi_cache["ts"]) < _NVIDIA_SMI_TTL:
            return _nvidia_smi_cache["gpus"]
    gpus = _nvidia_smi_gpus_uncached()
    with _nvidia_smi_lock:
        _nvidia_smi_cache["ts"] = time.time()
        _nvidia_smi_cache["gpus"] = gpus
    return gpus


def _nvidia_smi_gpus_uncached() -> list:
    """The actual nvidia-smi probe (see _nvidia_smi_gpus for the cached wrapper)."""
    import subprocess
    no_window = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                 if sys.platform == "win32" else 0)
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,"
             "utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4.0,
            creationflags=no_window,
        )
    except Exception:
        return []
    if r.returncode != 0:
        return []
    gpus: list = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        idx = _smi_num(parts[0])
        gpus.append({
            "index":        idx if idx is not None else 0,
            "name":         parts[1],
            "mem_used_mb":  _smi_num(parts[2]),
            "mem_total_mb": _smi_num(parts[3]),
            "util_pct":     _smi_num(parts[4]),
            "temp_c":       _smi_num(parts[5]),
            "power_w":      _smi_num(parts[6]),
        })
    return gpus


def _disks_info() -> list:
    """Free/total (GB) per mounted disk. Prefers psutil (every real partition);
    falls back to shutil.disk_usage on the project's own drive when psutil is
    absent (e.g. the Linux-CI sim, which BLOCKS psutil). Never raises."""
    disks: list = []
    try:
        import psutil
        for part in psutil.disk_partitions(all=False):
            try:
                u = psutil.disk_usage(part.mountpoint)
            except Exception:
                continue           # empty CD drive / permission → skip
            disks.append({
                "drive":    part.mountpoint,
                "free_gb":  round(u.free / 1e9, 1),
                "total_gb": round(u.total / 1e9, 1),
            })
        return disks
    except Exception:
        pass
    # Degraded fallback: just the drive JARVIS lives on.
    try:
        import shutil
        u = shutil.disk_usage(PROJECT_DIR)
        drive = os.path.splitdrive(PROJECT_DIR)[0] or os.path.abspath(os.sep)
        disks.append({"drive": drive or "/",
                      "free_gb": round(u.free / 1e9, 1),
                      "total_gb": round(u.total / 1e9, 1)})
    except Exception:
        pass
    return disks


# CPU% between two polls, from a MODULE-LEVEL cpu_times() baseline.
#
# WHY NOT psutil.cpu_percent(interval=None) (2026-09-30 audit, P0: the System
# tab read "CPU 0%" permanently). psutil keeps that call's baseline PER THREAD,
# and returns 0.0 on a thread's first call. ThreadingHTTPServer answers every
# request on a NEW thread, so every /api/system poll was some thread's first
# call - 0.0, forever. The baseline therefore lives here, shared by every
# request thread under a lock. Two polls closer than _CPU_MIN_SAMPLE_S reuse
# the last figure instead of measuring a few ms of noise.
_CPU_MIN_SAMPLE_S = 0.5
_cpu_lock = threading.Lock()
_cpu_state: dict = {"times": None, "at": 0.0, "pct": None}


def _cpu_busy_total(t) -> tuple:
    """(busy, total) seconds from a psutil cpu_times() tuple. Idle (and
    iowait, where the platform reports it) count as not busy; guest time is
    already inside user on Linux, so it is not added twice."""
    fields = getattr(t, "_fields", None) or ()
    vals = dict(zip(fields, t)) if fields else {}
    total = sum(v for k, v in vals.items() if k not in ("guest", "guest_nice"))
    idle = vals.get("idle", 0.0) + vals.get("iowait", 0.0)
    return total - idle, total


def _cpu_percent(psutil_mod, now: float | None = None):
    """System-wide CPU % since the previous call from ANY thread, or None when
    no interval has been measured yet (the very first call takes a 0.1 s
    sample so the tab never opens on a fake 0%). Never raises."""
    now = time.monotonic() if now is None else now
    try:
        with _cpu_lock:
            st = _cpu_state
            if st["times"] is not None and (now - st["at"]) < _CPU_MIN_SAMPLE_S:
                return st["pct"]
            prev = st["times"]
            if prev is None:
                prev = psutil_mod.cpu_times()
                time.sleep(0.1)
            cur = psutil_mod.cpu_times()
            b0, t0 = _cpu_busy_total(prev)
            b1, t1 = _cpu_busy_total(cur)
            dt = t1 - t0
            pct = None
            if dt > 0:
                pct = round(max(0.0, min(100.0, 100.0 * (b1 - b0) / dt)), 1)
            st.update({"times": cur, "at": now,
                       "pct": pct if pct is not None else st["pct"]})
            return st["pct"]
    except Exception:
        return None


def _system_info(hud_state_path: str, log_dir: str) -> dict:
    """The /api/system payload: GPUs (nvidia-smi), CPU/RAM (psutil), disks, plus
    version/uptime/routing reused from the status sources. EVERY field is always
    present with a safe default (None / []) so the client can rely on the shape
    even when a source is unavailable. Read-only; never raises."""
    cpu_pct = ram_used = ram_total = None
    try:
        import psutil
        cpu_pct = _cpu_percent(psutil)
        vm = psutil.virtual_memory()
        ram_used = round((vm.total - vm.available) / 1e9, 1)
        ram_total = round(vm.total / 1e9, 1)
    except Exception:
        pass
    return {
        "gpus":         _nvidia_smi_gpus(),
        "cpu_pct":      cpu_pct,
        "ram_used_gb":  ram_used,
        "ram_total_gb": ram_total,
        "disks":        _disks_info(),
        "version":      _read_version(),
        "uptime":       _uptime_seconds(log_dir),
        "routing":      _gpu_summary_routing(_gpu_summary()),
    }


def _norm_speak(cell: str) -> str:
    """Normalise a speak-class table cell ('**VERBATIM**' / '*INFORMATIVE*' /
    'neither') to a bare token: 'VERBATIM' | 'INFORMATIVE' | 'neither'."""
    up = (cell or "").replace("*", "").strip().upper()
    if up == "VERBATIM":
        return "VERBATIM"
    if up == "INFORMATIVE":
        return "INFORMATIVE"
    return "neither"


def _parse_action_index(path: str) -> dict:
    """Parse docs/ACTION_INDEX.md's 'Full index' table into
    ``{"actions": [{"name", "spoken"}], "count": N}``. Aliases sharing a handler
    (a comma-separated first cell) are EXPANDED so every dispatchable name is its
    own sendable row. Unreadable/absent file → ``{"actions": [], "count": 0}``.
    Never raises.

    Row shape:  ``| `name1`, `name2` | `handler:line` | **VERBATIM** | ex? | tests |``
    We keep only rows whose first cell wraps its name(s) in backticks — that skips
    the header ('action(s)'), the ``|---|`` separator, and the Summary table
    (prose first cell, no backticks)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return {"actions": [], "count": 0}
    actions: list = []
    seen: set = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        name_cell = cells[0]
        if "`" not in name_cell:
            continue
        spoken = _norm_speak(cells[2])
        for nm in name_cell.split(","):
            nm = nm.strip().strip("`").strip()
            if nm and nm not in seen:
                seen.add(nm)
                actions.append({"name": nm, "spoken": spoken})
    return {"actions": actions, "count": len(actions)}


# ── the Actions tab: LIVE registry + dispatch-by-name (2026-09-30 audit) ──────
#
# WHY LIVE. docs/ACTION_INDEX.md is generated from the TRACKED sources, goes
# stale between regenerations, and (by design, since the same day) never lists
# the owner's locally-installed private skills. The running JARVIS's ACTIONS
# dict is the truth, so the tab reads it when the server runs inside JARVIS and
# falls back to the index only for a bare web process.
#
# WHY A DEDICATED ENDPOINT. "Send" used to type the bare action NAME into the
# command channel, where the LLM re-interpreted it (or refused it). POST
# /api/action calls the registered handler directly - and because that skips
# the LLM's judgement entirely, side-effect and destructive names need an
# explicit confirmation first.
#
# _ACTION_CONFIRM_RULES is that DENYLIST: fnmatch patterns on the lower-cased
# name, each with the reason shown in the confirm prompt. A name matching none
# runs on one click. Deliberately broad - a spurious prompt costs a click, a
# missing one can message someone or wipe memory.
#
# A NAME rule alone missed ALIASES (2026-10-01): shutdown_jarvis asked first
# while shut_down / exit_jarvis / quit_jarvis / power_off_jarvis /
# turn_off_jarvis - the SAME handler - ran on one click, and so did
# smart_home_purge_cookie (forget_alexa_login's handler) and the code
# runner's run_python / python / eval_python / compute. So the live paths
# confirm by HANDLER too (_live_confirm_reason): a name inherits the reason of
# any other name bound to the same callable. The patterns below name the
# known aliases as well, for the index fallback, which has no handlers.
_ACTION_CONFIRM_RULES = (
    (("*shutdown*", "*shut_down*", "*restart*", "*reboot*", "*hibernate*",
      "sleep_pc", "*log_off*", "*logoff*", "*sign_out*", "lock_pc",
      "lock_screen", "*relaunch*", "exit_jarvis", "quit_jarvis",
      "*power_off*", "turn_off_jarvis"),
     "stops or restarts JARVIS or the PC"),
    (("send_*", "*_send", "reply_*", "*_reply", "text_*", "*_text_*",
      "email_*", "*_email", "sms_*", "call_*", "answer_call", "decline_call",
      "post_*", "publish_*", "share_*", "notify_*", "message_*", "*_message",
      "announce_*", "speak_*", "say_*"),
     "sends or says something to someone"),
    (("archive_*", "delete_*", "*_delete", "forget_*", "*_forget", "clear_*",
      "wipe_*", "reset_*", "*_reset", "*purge*", "remove_*", "*_remove",
      "erase_*", "empty_*", "drop_*", "scrap_*", "uninstall_*", "unenroll_*",
      "export_memory", "revoke_*"),
     "deletes, resets or exports data"),
    (("start_overnight_upgrade", "*upgrade*", "*self_update*", "apply_*",
      "install_*", "run_shell", "run_code", "run_python", "python",
      "eval_python", "compute", "execute_*", "*_execute", "*_script",
      "code_*", "pip_*", "git_*", "rollback*", "*_rollback"),
     "changes JARVIS's own code or runs code"),
    (("type", "type_*", "hotkey", "click", "*_click", "press_*", "kill_*",
      "close_*", "*_close", "stop_pipeline", "web_interface_off", "*_off_all",
      "force_*", "switch_llm", "switch_model", "set_model", "use_model"),
     "acts on the desktop or stops a running service"),
    (("buy_*", "order_*", "pay_*", "purchase_*", "checkout*", "transfer_*"),
     "spends money"),
)
# Handled by the tray control plane's hardened teardown instead of a request
# thread (a restart spawns a successor and exits this process mid-response).
# Keyed by the registry name whose HANDLER the tray command runs, and matched
# by handler (_action_via_tray), so every alias of restart / shutdown_jarvis
# goes the same way. The old name list held "shutdown", which is no ACTIONS
# key at all, so every shutdown alias tore JARVIS down on a request thread
# (2026-10-01).
_ACTION_VIA_TRAY = (("restart", "restart"), ("shutdown_jarvis", "shutdown"))
_ACTION_TIMEOUT_S = 20.0
_ACTION_MIN_GAP_S = 1.0          # per-name double-click guard
_action_last_call: dict = {}
_action_rate_lock = threading.Lock()


def action_confirm_reason(name: str) -> str:
    """The confirm-prompt reason for action ``name``, or '' when it may run
    on one click (see _ACTION_CONFIRM_RULES)."""
    n = str(name or "").strip().lower()
    for patterns, why in _ACTION_CONFIRM_RULES:
        if any(fnmatchcase(n, p) for p in patterns):
            return why
    return ""


def _live_confirm_reason(acts, name: str) -> str:
    """action_confirm_reason for a name in the LIVE registry ``acts``: its own
    reason, else the reason of any OTHER name bound to the same handler, so an
    alias can never run on one click while its twin asks first. Never
    raises."""
    why = action_confirm_reason(name)
    if why:
        return why
    try:
        fn = acts.get(name)
        if fn is None:
            return ""
        for other in _registry_names(acts):
            if other != name and acts.get(other) is fn:
                why = action_confirm_reason(other)
                if why:
                    return why
    except Exception:
        pass
    return ""


def _action_via_tray(acts, name: str) -> str:
    """The tray command that must run action ``name`` (see _ACTION_VIA_TRAY),
    matched by HANDLER, or ''. Never raises."""
    try:
        fn = acts.get(name)
        if fn is None:
            return ""
        for key, cmd in _ACTION_VIA_TRAY:
            if name == key or acts.get(key) is fn:
                return cmd
    except Exception:
        pass
    return ""


def _log_info(msg: str) -> None:
    """One stdout line (the live process tees stdout into the session log).
    For RARE control events only - never per request. Never raises."""
    try:
        print(f"  [web] {msg}", flush=True)
    except Exception:
        pass


def _registry_names(acts) -> list:
    """A SNAPSHOT of a live dict's keys. A skill reload may resize ACTIONS
    while we iterate, so retry on the RuntimeError that raises. Never
    raises."""
    for _ in range(3):
        try:
            return [str(k) for k in list(acts.keys())]
        except RuntimeError:
            time.sleep(0.01)
        except Exception:
            return []
    return []


def _voice_only_hint(cfg: dict, name):
    """The skill-declared hint when ``name`` is a VOICE-ONLY action (it acts
    only on a fresh spoken / typed request, so the generic Actions list can
    never run it - core/web_panels VOICE_ONLY_ACTIONS); None otherwise.
    Never raises."""
    try:
        fn = getattr(_panel_registry(cfg), "voice_only", None)
        return fn(name) if callable(fn) else None
    except Exception:
        return None


def actions_payload(cfg: dict) -> dict:
    """GET /api/actions: ``{"actions": [{name, spoken, confirm, why,
    voice_only, use_instead}], "count", "source": "live"|"index",
    "confirm_rules": [...]}``. Never raises."""
    rules = [{"patterns": list(p), "why": w} for p, w in _ACTION_CONFIRM_RULES]
    rt = _runtime(cfg)
    acts = None
    try:
        acts = rt.actions()
    except Exception:
        acts = None
    if acts is not None:
        sets = None
        try:
            sets = rt.speak_sets()
        except Exception:
            sets = None
        verbatim, informative, selfv = sets or (set(), set(), set())
        rows = []
        for n in sorted(set(_registry_names(acts))):
            spoken = ("VERBATIM" if n in verbatim else
                      "INFORMATIVE" if n in informative else
                      "SELF-VOICED" if n.lower() in selfv else "neither")
            why = _live_confirm_reason(acts, n)
            vo = _voice_only_hint(cfg, n)
            rows.append({"name": n, "spoken": spoken, "confirm": bool(why),
                         "why": why, "voice_only": vo is not None,
                         "use_instead": vo or ""})
        return {"actions": rows, "count": len(rows), "source": "live",
                "confirm_rules": rules}
    out = _parse_action_index(cfg.get("action_index_path", ""))
    for a in out["actions"]:
        why = action_confirm_reason(a["name"])
        a["confirm"], a["why"] = bool(why), why
    out["source"] = "index"
    out["confirm_rules"] = rules
    return out


def run_named_action(cfg: dict, name, arg="", *, confirm: bool = False) -> tuple:
    """``(http_code, payload)`` for POST /api/action. 400 bad input, 503 no
    live registry in this process, 404 unknown name, 409 confirmation needed,
    429 the same name again within _ACTION_MIN_GAP_S. Otherwise the handler
    runs on a daemon thread and we wait up to _ACTION_TIMEOUT_S: status
    "done" (with its result), "running" (still going - it keeps running), or
    "error" (it raised). Never raises."""
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        return 400, {"error": "name must be an action name"}
    name = name.strip()
    if arg is None:
        arg = ""
    if not isinstance(arg, str) or len(arg) > 2000:
        return 400, {"error": "arg must be a string of at most 2000 chars"}
    acts = _runtime(cfg).actions()
    if acts is None:
        return 503, {"error": "the live action registry is not reachable - "
                              "this server is not running inside JARVIS"}
    fn = acts.get(name)
    if not callable(fn):
        return 404, {"error": "unknown action", "name": name}
    # A voice-only action (2026-10-01) reads the owner's last spoken sentence
    # as proof he asked, so run from here it can only refuse ("binding",
    # "I need to hear you say ...") - and it marked that sentence as used.
    # Refused BEFORE the handler is ever called, confirmed or not.
    vo = _voice_only_hint(cfg, name)
    if vo is not None:
        return 409, {"error": "voice only - this action needs a fresh spoken "
                              "or typed request" + (": use " + vo if vo else ""),
                     "voice_only": True, "use_instead": vo, "name": name}
    why = _live_confirm_reason(acts, name)
    if why and not confirm:
        return 409, {"error": "confirmation required", "confirm_required": True,
                     "name": name, "why": why}
    now = time.monotonic()
    with _action_rate_lock:
        last = _action_last_call.get(name)
        if last is not None and now - last < _ACTION_MIN_GAP_S:
            return 429, {"error": "already sent - wait a moment",
                         "retry_after_s": round(_ACTION_MIN_GAP_S - (now - last), 2)}
        _action_last_call[name] = now
    via = _action_via_tray(acts, name)
    if via:
        try:
            send_tray_command(via, cfg["tray_commands_path"], arg=arg)
        except Exception as e:
            return 500, {"error": f"control write failed: {e}"}
        _log_info(f"action {name} queued on the tray control plane ({via})")
        return 200, {"ok": True, "status": "queued", "via": "tray",
                     "name": name}
    box: dict = {}

    def _run():
        try:
            box["result"] = fn(arg)
        except Exception as e:           # reported, never raised into HTTP
            box["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=_run, daemon=True, name="web-action-" + name)
    t.start()
    t.join(float(cfg.get("action_timeout_s", _ACTION_TIMEOUT_S)))
    if t.is_alive():
        _log_info(f"action {name} started from the dashboard (still running)")
        return 200, {"ok": True, "status": "running", "name": name,
                     "result": ""}
    if "error" in box:
        _log_info(f"action {name} raised: {box['error']}")
        return 200, {"ok": False, "status": "error", "name": name,
                     "error": box["error"]}
    res = box.get("result")
    text = res if isinstance(res, str) else ("" if res is None else str(res))
    _log_info(f"action {name} ran from the dashboard")
    return 200, {"ok": True, "status": "done", "name": name,
                 "result": text[:4000]}


def _panel_registry(cfg: dict):
    """The panel registry this server serves: the one create_server was given,
    else the process-wide core.web_panels.REGISTRY (what the skill loader
    fills). An import failure degrades to an empty registry. Never raises."""
    reg = cfg.get("panels") if isinstance(cfg, dict) else None
    if reg is not None:
        return reg
    try:
        from core import web_panels
        return web_panels.REGISTRY
    except Exception:
        class _Empty:
            def list_meta(self):
                return []

            def state(self, _pid):
                return 404, {"error": "unknown panel"}

            def call_action(self, *_a, **_k):
                return 404, {"error": "unknown panel"}

            def stream_source(self, *_a):
                return None
        return _Empty()


def _voices_info(config_mod=None) -> dict:
    """The /api/voices payload: enrolled voice-clone profiles (name/source and a
    ``usable`` flag straight from the consent gate) plus the active profile, the
    master switch, and the base TTS backend/voice — read live from core. Degrades
    to empty/defaults on any import failure (bare CI). READ-ONLY: it lists profile
    metadata only and never loads the cloning model. ``config_mod`` stands in
    for core.config (tests)."""
    profiles: list = []
    active = ""
    enabled = False
    tts_backend = ""
    tts_voice = ""
    try:
        from core import voice_clone
        for meta in voice_clone.list_profiles():
            try:
                usable = bool(voice_clone.profile_is_usable(meta))
            except Exception:
                usable = False
            profiles.append({
                "name":   meta.get("name", ""),
                "source": meta.get("source", ""),
                "usable": usable,
            })
    except Exception:
        profiles = []
    clone_model = clone_device = ai_backend = local_model = ""
    try:
        if config_mod is not None:
            _config = config_mod
        else:
            from core import config as _config
        active = getattr(_config, "VOICE_CLONE_PROFILE", "") or ""
        enabled = bool(getattr(_config, "VOICE_CLONE_ENABLED", False))
        tts_backend = getattr(_config, "TTS_BACKEND", "") or ""
        tts_voice = getattr(_config, "TTS_VOICE", "") or ""
        clone_model = str(getattr(_config, "VOICE_CLONE_MODEL", "") or "")
        clone_device = str(getattr(_config, "VOICE_CLONE_DEVICE", "") or "")
        ai_backend = str(getattr(_config, "AI_BACKEND", "") or "")
        local_model = str(getattr(_config, "LOCAL_LLM_MODEL", "") or "")
    except Exception:
        pass
    engine, voice = _base_voice(tts_backend, tts_voice)
    if enabled and active:
        summary = "clone '%s' via %s" % (active, clone_model or "chatterbox")
    else:
        summary = engine + (" · " + voice if voice else "")
    return {
        "profiles":    profiles,
        "active":      active,
        "enabled":     enabled,
        "tts_backend": tts_backend,
        # tts_voice is the EDGE voice knob only; `voice` is what the ACTIVE
        # engine actually uses (2026-09-30 audit: the tab said
        # "normal (en-GB-RyanNeural)" while Kokoro was speaking).
        "tts_voice":   tts_voice,
        "engine":      engine,
        "voice":       voice,
        "summary":     summary,
        # For the "use a cloned voice" warning: Chatterbox loads on the GPU
        # beside the local LLM, and with the 26B model resident that is a known
        # 24 GB VRAM overload on the owner's card.
        "clone_model":  clone_model,
        "clone_device": clone_device,
        "llm_local":    ai_backend.strip().lower() != "claude",
        "local_model":  local_model,
    }


def _base_voice(tts_backend: str, tts_voice: str) -> tuple:
    """(engine, voice) the NON-clone TTS path is really using. Kokoro's voice
    is read from the loaded module when present (core.kokoro_tts._VOICE, set
    from KOKORO_VOICE at import), never assumed from TTS_VOICE, which only the
    edge backend reads. Never raises."""
    b = (tts_backend or "").strip().lower()
    if b == "kokoro":
        v = ""
        try:
            kt = sys.modules.get("core.kokoro_tts")
            v = str(getattr(kt, "_VOICE", "") or "") if kt else ""
        except Exception:
            v = ""
        return "kokoro", v or os.environ.get("KOKORO_VOICE", "bm_george")
    if b == "edge":
        return "edge", tts_voice or ""
    if b == "xtts":
        return "xtts", "voice sample clone"
    if b == "pyttsx3":
        return "pyttsx3", "system voice"
    return (b or "unknown"), ""


def _read_json_list(path: str) -> list:
    """Read a JSON file expected to hold a list; ``[]`` on any error. Cheap; no
    deps; never raises."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _read_memory() -> dict:
    """The /api/memory payload: long-term semantic FACTS + recent EPISODES with
    counts. Deliberately READ-ONLY and CHEAP — we NEVER call the module's
    ensure_loaded() (which would rebuild the BM25 index / run first-boot
    migration) and never touch the embedder. Instead we read the already-loaded
    in-memory facts when a live JARVIS has them (instant, zero side effect) and
    otherwise read the JSON mirror + episode JSONL straight off disk. Degrades to
    empty on any failure. Shape::

        {"facts": [{text, source, tags, updated_at}],
         "episodes": [{text, role, iso}],  # newest-first, capped
         "counts": {"facts": N, "episodes": M}}
    """
    facts: list = []
    episodes: list = []
    try:
        from core import long_term_memory as ltm
    except Exception:
        return {"facts": [], "episodes": [],
                "counts": {"facts": 0, "episodes": 0}}
    # FACTS — prefer the loaded in-memory dict (a live JARVIS has it), else the
    # on-disk mirror. Both are pure reads; neither triggers a load/rebuild.
    try:
        raw_facts = None
        if getattr(ltm, "_loaded", False):
            lock = getattr(ltm, "_lock", None)
            if lock is not None:
                with lock:
                    raw_facts = list(getattr(ltm, "_facts", {}).values())
            else:
                raw_facts = list(getattr(ltm, "_facts", {}).values())
        if raw_facts is None:
            raw_facts = _read_json_list(getattr(ltm, "_FACTS_JSON", ""))
        for fentry in raw_facts:
            if isinstance(fentry, dict) and str(fentry.get("text", "")).strip():
                facts.append({
                    "text":       str(fentry.get("text", "")),
                    "source":     fentry.get("source", ""),
                    "tags":       fentry.get("tags", []),
                    "updated_at": fentry.get("updated_at"),
                })
    except Exception:
        facts = []
    # EPISODES — tail the JSONL log directly (read-only). Count all lines; keep
    # only the most recent 50 (newest-first) for display so a long history stays a
    # cheap payload.
    ep_count = 0
    try:
        ep_path = getattr(ltm, "_EPISODE_LOG", "")
        raw_lines: list = []
        if ep_path and os.path.exists(ep_path):
            with open(ep_path, encoding="utf-8", errors="replace") as f:
                raw_lines = [ln for ln in f.read().splitlines() if ln.strip()]
        ep_count = len(raw_lines)
        for ln in reversed(raw_lines[-50:]):
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if isinstance(e, dict) and str(e.get("text", "")).strip():
                episodes.append({
                    "text": str(e.get("text", "")),
                    "role": e.get("role", ""),
                    "iso":  e.get("iso", ""),
                })
    except Exception:
        episodes = []
    return {
        "facts":    facts,
        "episodes": episodes,
        "counts":   {"facts": len(facts), "episodes": ep_count},
    }


# ── the request handler ──────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    """Routes: GET / (dashboard), GET /api/status, GET /api/log/tail, GET
    /api/settings, the read-only control-panel GETs (system / actions / voices /
    memory / camera-*), POST /api/say, POST /api/settings, POST /api/action,
    POST /api/control, and the skill-panel routes (/api/panels, /api/panel/...).
    The owning server pins config onto the class instance via the ``config``
    attribute set in ``create_server`` (a small dict) so handlers are stateless
    beyond it."""

    # Per-request socket timeout (seconds). StreamRequestHandler.setup() applies
    # this to the connection, so an under-delivered Content-Length (a client that
    # promises N bytes then stalls) raises socket.timeout inside rfile.read()
    # instead of hanging this worker thread forever (2026-07-08 finding).
    timeout = 10

    # Silence the default per-request stderr logging — it would spam the session
    # log we're tailing. (The base class calls this for every request.)
    def log_message(self, fmt, *args):  # noqa: A003 - matches base signature
        return

    # ── auth ────────────────────────────────────────────────────────────────
    def _token(self) -> str:
        return self.server.config.get("token", "")  # type: ignore[attr-defined]

    def _request_token(self, query: dict) -> str:
        """Pull a caller-supplied token from (in priority) the Authorization
        Bearer header, an X-Auth-Token header, or a ?token= query param."""
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        xat = self.headers.get("X-Auth-Token")
        if xat:
            return xat.strip()
        q = query.get("token")
        if q:
            return q[0]
        return ""

    def _authorized(self, query: dict, *, is_page: bool) -> bool:
        """Auth gate. With no token configured (only reachable on a LOCAL bind —
        create_server enforces that) everything is allowed. With a token set, an
        API request must present it. The dashboard PAGE is allowed token-free on a
        local bind (convenience) but requires the token on an exposed bind."""
        token = self._token()
        if not token:
            return True
        if is_page and self.server.config.get("local_bind", True):  # type: ignore[attr-defined]
            return True
        # Constant-time compare (2026-09-30 audit): a plain == leaks how many
        # leading characters matched through its timing.
        return hmac.compare_digest(
            self._request_token(query).encode("utf-8", "replace"),
            token.encode("utf-8", "replace"))

    # ── anti-CSRF / anti-DNS-rebinding for state-changing POSTs ──────────────
    @staticmethod
    def _host_of(value: str) -> str:
        """Bare lowercase hostname from a Host/Origin/Referer header value —
        scheme, path and port stripped, IPv6 brackets kept (``[::1]``).
        ``"http://localhost:8766/x"`` → ``"localhost"``; ``"127.0.0.1:8766"`` →
        ``"127.0.0.1"``; ``"[::1]:8766"`` → ``"[::1]"``. Empty on junk."""
        if not value:
            return ""
        v = value.strip()
        if "://" in v:
            v = v.split("://", 1)[1]
        v = v.split("/", 1)[0]            # drop any path
        if v.startswith("["):            # IPv6 literal: keep the [..] intact
            return v.split("]", 1)[0].lower() + "]"
        if ":" in v:                     # strip :port on a plain host/IPv4
            v = v.rsplit(":", 1)[0]
        return v.lower()

    def _served_hosts(self) -> set:
        """Hostnames this server legitimately answers to: loopback plus the
        configured bind. A request whose Host/Origin is outside this set is either
        a DNS-rebinding attempt (foreign Host resolved to us) or a cross-site POST
        (foreign Origin)."""
        hosts = {"localhost", "127.0.0.1", "[::1]", "::1"}
        bind = str(self.server.config.get("bind", "127.0.0.1")).strip().lower()  # type: ignore[attr-defined]
        if bind and bind not in ("0.0.0.0", "::"):
            hosts.add(bind)
        return hosts

    def _state_change_allowed(self) -> tuple:
        """Anti-DNS-rebinding + anti-CSRF guard, applied to EVERY request (GET and
        POST). Blocks two browser-driven attacks:

          * DNS rebinding — a page on evil.com rebinds it to 127.0.0.1; caught
            because the Host header is ``evil.com``, not a host we answer to.
          * Cross-site request (CSRF) — a page on evil.com fetch()es our URL;
            caught because the Origin/Referer host is ``evil.com``.

        On a LOCAL (loopback) bind the ONLY host we legitimately answer to is
        loopback, so this is enforced regardless of any token. Crucially it does
        NOT short-circuit when a token is set: on a local bind the dashboard page
        is served token-free and bakes the token into its JS, so a rebinding page
        could read the token and then present it — the token cannot be the
        rebinding boundary here (2026-07-08 finding). It also now covers GET, so a
        rebinding page can't read the token-baked page, the session log, or the
        settings/system/memory snapshots.

        On a NON-local (exposed) bind a token is mandatory + unforgeable (enforced
        by _authorized) and the owner may legitimately reach the server by LAN IP /
        hostname, which the loopback allowlist would reject — so there the token is
        the boundary and we do not Host-restrict.

        Non-browser clients (curl, PowerShell, the driver) send no Origin and a
        loopback Host, so they pass untouched. Returns ``(ok, reason)``."""
        if not self.server.config.get("local_bind", True):  # type: ignore[attr-defined]
            return True, ""                       # exposed bind: token is the boundary
        served = self._served_hosts()
        host = self._host_of(self.headers.get("Host", ""))
        if host and host not in served:           # foreign Host → DNS rebinding
            return False, "host"
        origin = self.headers.get("Origin", "")
        if origin:
            if self._host_of(origin) not in served:
                return False, "origin"
        else:
            # Some browsers omit Origin on same-origin GET/POST; fall back to Referer.
            ref = self.headers.get("Referer", "")
            if ref and self._host_of(ref) not in served:
                return False, "referer"
        return True, ""

    def _forbidden(self, reason: str) -> None:
        self._send_json({"error": f"cross-origin request refused ({reason})"},
                        code=403)

    # ── security headers on EVERY response (2026-09-30 audit) ────────────────
    # The page can inject commands, so it must never render inside someone
    # else's frame (clickjacking: X-Frame-Options + CSP frame-ancestors), must
    # not be content-sniffed, and nothing it serves - page, JSON, frames - may
    # be cached. Added in end_headers so a route cannot forget them; a header
    # a route already sent (e.g. its own Cache-Control) is not duplicated.
    _SECURITY_HEADERS = (
        ("X-Frame-Options", "DENY"),
        ("Content-Security-Policy",
         "default-src 'self'; script-src 'self' 'unsafe-inline'; "
         "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
         "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
         "object-src 'none'; form-action 'self'"),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "same-origin"),
        ("Cache-Control", "no-store"),
    )

    def send_response(self, code, message=None):  # noqa: D401 - base API
        self._sent_header_names = set()
        super().send_response(code, message)

    def send_header(self, keyword, value):  # noqa: D401 - base API
        try:
            self._sent_header_names.add(str(keyword).lower())
        except AttributeError:
            self._sent_header_names = {str(keyword).lower()}
        super().send_header(keyword, value)

    def end_headers(self):  # noqa: D401 - base API
        sent = getattr(self, "_sent_header_names", set())
        for k, v in self._SECURITY_HEADERS:
            if k.lower() not in sent:
                super().send_header(k, v)
        self._sent_header_names = set()
        super().end_headers()

    # ── tiny response helpers ────────────────────────────────────────────────
    def _send_json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # No caching of live status/log/reply payloads.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _send_html(self, text: str, code: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _send_bytes(self, data: bytes, content_type: str, code: int = 200) -> None:
        """Send a raw binary body (e.g. the camera preview JPEG). Mirrors
        _send_html/_send_json but for arbitrary bytes, with no-store caching so a
        stale frame is never served from a browser cache."""
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except Exception:
            pass

    def _stream_camera(self, path: str) -> None:
        """Stream ``path`` as multipart/x-mixed-replace (MJPEG) until the camera
        goes away or the client disconnects. ONE connection, one frame pushed per
        producer write. Never raises out of the handler.

        Contract, deliberately identical to /api/camera-preview so the tile's
        on/off behaviour is unchanged:
          * missing or already-stale file  -> 404 before any body, so the <img>
            fires `error` and the tile shows its 'off' placeholder immediately;
          * the frame going stale mid-stream (camera switched off) -> CLOSE the
            stream. THE CLIENT IS NOT TOLD. This used to claim the close "fires
            the same `error` on the client"; measured against a real Chrome
            2026-09-05, that is false, and three different endings were tried:
                clean close after a complete part -> events [load @413ms] and
                    nothing after it; img.complete stayed true, naturalWidth 1
                truncated final part             -> the same single load, no
                    error, and naturalWidth silently fell to 0
                boundary + headers then close    -> the same single load
            `load` fires ONCE, just after the FIRST frame — not per frame, and
            not at the close — so there is no ending this route can produce that
            the DOM will report. The tile is therefore supervised on the
            client's own clock against /api/camera-live (_camera_live_map),
            which answers with this same staleness rule;
          * we only OPEN the file when its (mtime_ns, size) actually changed, so
            we read exactly as often as JARVIS writes, never at the poll rate.
            That matters on Windows: the producer publishes via os.replace, and a
            replace FAILS while any process holds the file open (measured
            2026-09-04: 200/200 replaces failed against a held handle), so every
            needless open is a chance to silently drop a frame.
        """
        info = _preview_stat(path)
        if info is None or info[2] > _CAMERA_PREVIEW_STALE_S:
            return self._send_json({"error": "no preview"}, code=404)

        def poll(last):
            info = _preview_stat(path)
            if info is None or info[2] > _CAMERA_PREVIEW_STALE_S:
                return "close", None, None   # camera off -> close -> client shows 'off'
            key = (info[0], info[1])
            if key == last:
                return "wait", None, None
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                # Lost the race with the producer's os.replace — try again on
                # the next tick rather than tearing the stream down.
                return "wait", None, None
            if not data:
                return "wait", None, None
            return "frame", key, data

        return self._mjpeg_loop(poll)

    def _mjpeg_loop(self, poll, poll_s: float = _CAMERA_STREAM_POLL_S) -> None:
        """THE multipart/x-mixed-replace writer, shared by the camera tiles and
        the skill panels' streams (2026-09-30), so both obey one slot cap and
        one liveness rule. ``poll(last_key)`` returns ``("frame", key, jpeg)``
        to push, ``("wait", _, _)`` for nothing new, ``("close", _, _)`` to end
        the stream. Never raises out of the handler."""
        with _camera_stream_clients_lock:
            if _camera_stream_clients[0] >= _CAMERA_STREAM_MAX_CLIENTS:
                # Refuse rather than hold yet another worker thread hostage; the
                # dashboard falls back to polling stills.
                return self._send_json({"error": "too many camera streams"},
                                       code=503)
            _camera_stream_clients[0] += 1
        try:
            self.send_response(200)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={_CAMERA_STREAM_BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            last = None
            while True:
                # LIVENESS: a client that closed shows up as a readable
                # socket yielding EOF. Without this, a stream whose frames
                # are momentarily frozen holds its worker thread AND one of
                # the _CAMERA_STREAM_MAX_CLIENTS slots until the frame goes
                # stale - long enough for a few tab reloads to lock every
                # slot out. MSG_PEEK so we never consume anything.
                try:
                    if select.select([self.connection], [], [], 0)[0]:
                        if not self.connection.recv(1, socket.MSG_PEEK):
                            return
                except Exception:
                    return
                verdict, key, data = poll(last)
                if verdict == "close":
                    return
                if verdict != "frame":
                    time.sleep(poll_s)
                    continue
                last = key
                head = (f"--{_CAMERA_STREAM_BOUNDARY}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(data)}\r\n\r\n").encode("ascii")
                # ONE write per frame (headers + body + trailing CRLF in a
                # single send) rather than three: fewer syscalls and no
                # tiny 2-byte tail segment for Nagle to sit on. Measured
                # 2026-09-04 this made no difference on loopback — it is
                # kept because it cannot be worse, not because it was the
                # win. The win is the stream itself (1.0 -> 15.2 fps).
                self.wfile.write(head + data + b"\r\n")
                self.wfile.flush()
        except Exception:
            return                      # client went away / broken pipe
        finally:
            with _camera_stream_clients_lock:
                _camera_stream_clients[0] -= 1

    def _stream_panel(self, source) -> None:
        """MJPEG for a skill panel's ``streams`` callable (latest JPEG or None).
        404 when it has no frame at all; the stream closes once it has returned
        None for longer than the camera staleness window. A source that raises
        is treated as "no frame". Same slot cap as the camera tiles."""
        def grab():
            try:
                data = source()
            except Exception:
                return None
            return data if isinstance(data, (bytes, bytearray)) and data else None

        if grab() is None:
            return self._send_json({"error": "no frame"}, code=404)
        state = {"none_since": None}

        def poll(last):
            data = grab()
            now = time.monotonic()
            if data is None:
                if state["none_since"] is None:
                    state["none_since"] = now
                if now - state["none_since"] > _CAMERA_PREVIEW_STALE_S:
                    return "close", None, None
                return "wait", None, None
            state["none_since"] = None
            key = (len(data), zlib.crc32(data))
            if key == last:
                return "wait", None, None
            return "frame", key, bytes(data)

        return self._mjpeg_loop(poll, poll_s=0.05)

    def _unauthorized(self) -> None:
        self._send_json({"error": "unauthorized"}, code=401)

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_GET(self):  # noqa: N802 - http.server API
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)
        cfg = self.server.config  # type: ignore[attr-defined]

        # Anti-rebinding/CSRF guard on GET too — a foreign Host on a local bind is
        # a DNS-rebinding page that would otherwise read the token-baked dashboard,
        # the session log, or the settings/system/memory snapshots. No-op for
        # non-browser clients and on an exposed (token-protected) bind.
        ok, why = self._state_change_allowed()
        if not ok:
            return self._forbidden(why)

        if path == "/":
            if not self._authorized(query, is_page=True):
                return self._unauthorized()
            return self._send_html(_dashboard_html(cfg.get("token", "")))

        if path == "/api/status":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            return self._send_json(build_status(cfg["hud_state_path"], cfg["log_dir"]))

        if path == "/api/log/tail":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            try:
                n = int(query.get("lines", ["50"])[0])
            except (TypeError, ValueError):
                n = 50
            since = None
            if "since" in query:
                try:
                    since = int(query.get("since", ["0"])[0])
                except (TypeError, ValueError):
                    since = None
            return self._send_json(tail_log(cfg["log_dir"], n, since=since,
                                            log_name=query.get("log", [""])[0]))

        if path == "/api/settings":
            # The FULL settings snapshot: every schema knob + its current value
            # (the saved file overlaid on the live constants — see
            # build_settings_schema), read fresh each call. Passing
            # user_settings_path makes this GET read the EXACT file POST
            # /api/settings writes, so a save round-trips instead of reverting.
            # Gated the same as every other API route (token when one is set) —
            # reading the config is less sensitive than writing it, but there's
            # no reason to leak it token-free on an exposed bind.
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            return self._send_json(
                build_settings_schema(cfg["user_settings_path"]))

        # ── control-panel endpoints (System / Actions / Voice / Memory + the
        #    camera preview image). Each is auth-gated exactly like /api/status,
        #    read-only, and wraps its data source in try/except so a source
        #    failure becomes a safe JSON error rather than a 500 traceback. ──
        if path == "/api/system":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            try:
                return self._send_json(
                    _system_info(cfg["hud_state_path"], cfg["log_dir"]))
            except Exception as e:
                return self._send_json({"error": f"system read failed: {e}",
                                        "gpus": [], "disks": []}, code=500)

        if path == "/api/actions":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            try:
                return self._send_json(actions_payload(cfg))
            except Exception as e:
                return self._send_json({"error": f"actions read failed: {e}",
                                        "actions": [], "count": 0}, code=500)

        if path == "/api/camera-tiles":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            try:
                return self._send_json(camera_tiles_payload(cfg))
            except Exception as e:
                return self._send_json({"error": f"camera tiles failed: {e}",
                                        "tiles": [], "source": "error"},
                                       code=500)

        if path == "/api/panels":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            return self._send_json({"panels": _panel_registry(cfg).list_meta()})

        if path.startswith("/api/panel/"):
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            return self._get_panel(cfg, path, query)

        if path == "/api/voices":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            try:
                return self._send_json(_voices_info())
            except Exception as e:
                return self._send_json({"error": f"voices read failed: {e}",
                                        "profiles": []}, code=500)

        if path == "/api/memory":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            try:
                return self._send_json(_read_memory())
            except Exception as e:
                return self._send_json({"error": f"memory read failed: {e}",
                                        "facts": [], "episodes": [],
                                        "counts": {"facts": 0, "episodes": 0}},
                                       code=500)

        if path == "/api/camera-preview":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            # Serve the live preview JPEG, or 404 when it's missing OR stale
            # (older than ~5 s = camera off). No side effects; never raises.
            # ?cam=left|right|kinect selects an INDIVIDUAL camera's tile (the
            # monolith publishes one file per camera, 2026-07-10); no param =
            # the historical primary/composite preview, so the HUD and old
            # bookmarks are unaffected.
            try:
                p = _preview_path_for(cfg, query.get("cam", [""])[0] or "")
            except UnknownCamError:
                return self._send_json({"error": "unknown cam"}, code=404)
            try:
                info = _preview_stat(p)
                if info is None or info[2] > _CAMERA_PREVIEW_STALE_S:
                    return self._send_json({"error": "no preview"}, code=404)
                with open(p, "rb") as f:
                    data = f.read()
                return self._send_bytes(data, "image/jpeg")
            except Exception:
                return self._send_json({"error": "no preview"}, code=404)

        if path == "/api/camera-reason":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            # WHY A SEPARATE ENDPOINT rather than putting the text in the 404
            # body of camera-preview/camera-stream: an <img> DISCARDS the body of
            # a failed load. The tile learns of the failure only through the DOM
            # `error` event, which carries nothing, so a second fetch is the only
            # mechanism that can put words in the placeholder at all. It also
            # keeps the cost strictly on the error path - the streaming path
            # never touches the ladder, and the device probe behind it never runs
            # while a camera is working.
            try:
                r = _camera_off_reason(cfg, query.get("cam", [""])[0] or "")
            except UnknownCamError:
                return self._send_json({"error": "unknown cam"}, code=404)
            except Exception as e:   # never let the explainer break the page
                return self._send_json({"cam": query.get("cam", [""])[0] or "",
                                        "state": "unknown",
                                        "message": _CAMERA_REASONS["unknown"],
                                        "detail": f"{type(e).__name__}: {e}"})
            return self._send_json(r)

        if path == "/api/camera-live":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            # THE TILE'S ONLY WAY TO NOTICE A MID-STREAM DEATH. An MJPEG <img>
            # is told nothing when the server closes its stream (measured, see
            # _stream_camera), so the dashboard asks here instead of waiting for
            # an event that never comes. Three stats, no ladder, no device
            # probe: it runs about once a second while the Camera tab is open,
            # which is a quarter of what the still-polling design it replaced
            # was already paying, and nothing at all on every other tab.
            return self._send_json({"stale_after": _CAMERA_PREVIEW_STALE_S,
                                    "cams": _camera_live_map(cfg)})

        if path == "/api/camera-stream":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            # LIVE MJPEG for the Camera tab. Same file, same staleness rule and
            # same ?cam= vocabulary as /api/camera-preview above (both go through
            # _preview_path_for) — the difference is that this one KEEPS PUSHING.
            try:
                p = _preview_path_for(cfg, query.get("cam", [""])[0] or "")
            except UnknownCamError:
                return self._send_json({"error": "unknown cam"}, code=404)
            return self._stream_camera(p)

        return self._send_json({"error": "not found"}, code=404)

    # ── POST ─────────────────────────────────────────────────────────────────
    def _read_body(self, cap: int = 64 * 1024) -> bytes:
        """Read the request body up to ``cap`` bytes (never raises). Shared by the
        /api/say and /api/settings handlers so the body-length parsing + size cap
        live in ONE place."""
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return b""
        try:
            return self.rfile.read(min(length, cap))
        except Exception:
            return b""

    def do_POST(self):  # noqa: N802 - http.server API
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)
        cfg = self.server.config  # type: ignore[attr-defined]

        # Both POST routes below change state (run a command / write config). On a
        # local bind, refuse a browser-driven cross-origin or DNS-rebound request
        # BEFORE doing anything. Applied unconditionally (any POST path).
        ok, why = self._state_change_allowed()
        if not ok:
            return self._forbidden(why)

        # POST /api/settings — WRITE settings. A settings write is POWERFUL (it can
        # flip WEB_INTERFACE_BIND/TOKEN, enable ambient listening, etc.), so it is
        # gated by the SAME auth as every other route: when a token is configured it
        # is REQUIRED here (a settings write is never allowed token-free on an
        # exposed bind; on a local bind with no token there's no token to require,
        # exactly like /api/say). See _handle_post_settings.
        if path == "/api/settings":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            return self._handle_post_settings(cfg)

        if path == "/api/action":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            return self._handle_post_action(cfg)

        if path == "/api/control":
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            return self._handle_post_control(cfg)

        if path.startswith("/api/panel/"):
            if not self._authorized(query, is_page=False):
                return self._unauthorized()
            return self._post_panel_action(cfg, path)

        if path != "/api/say":
            return self._send_json({"error": "not found"}, code=404)
        if not self._authorized(query, is_page=False):
            return self._unauthorized()

        # Parse the JSON body {"text": "...", "timeout": <optional seconds>}.
        raw = self._read_body()
        text = ""
        req_timeout = _REPLY_TIMEOUT_DEFAULT
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            data = {}
        if isinstance(data, dict):
            text = str(data.get("text", "")).strip()
            # Parse timeout SEPARATELY — a non-numeric timeout must fall back to the
            # default, NEVER discard an otherwise-valid command (2026-07-08 finding).
            if "timeout" in data:
                try:
                    req_timeout = float(data["timeout"])
                except (TypeError, ValueError):
                    req_timeout = _REPLY_TIMEOUT_DEFAULT
        if not text:
            return self._send_json({"error": "empty text"}, code=400)

        # Inject via the SAME channel the voice loop drains, then wait for a reply.
        try:
            inject_command(text, cfg["inject_path"])
        except Exception as e:
            return self._send_json({"error": f"inject failed: {e}"}, code=500)

        reader = cfg.get("reply_reader") or wait_for_reply
        try:
            res = reader(text, cfg["log_dir"], req_timeout)
        except Exception as e:
            # The command was injected and will run; we just couldn't tail a reply.
            return self._send_json({"accepted": True, "reply": "", "status": f"reply_error: {e}"})

        status = res.get("status", "accepted")
        lines = res.get("lines", []) or []
        # "reply" is the ANSWER (wait_for_reply folds lead-ins, action results
        # and "JARVIS (spoken):" overrides - see parse_turn_lines); a stub
        # reader that only returns lines keeps the old joined-lines reply.
        reply = res.get("reply")
        if not isinstance(reply, str) or not reply:
            reply = "\n".join(lines)
        return self._send_json({
            "accepted": True,           # queued; status "standby" = ignored
            "status": status,
            "reply": reply,
            "reply_lines": lines,
            "actions": res.get("actions", []) or [],
        })

    def _handle_post_settings(self, cfg: dict) -> None:
        """Handle POST /api/settings — validate + merge one or more settings into
        the user_settings.json file, atomically.

        BODY SHAPE (both accepted):
          • a single update:  {"name": "WAKE_WORD_AUTOSTART", "value": true}
          • a batch:          {"settings": {"WAKE_WORD_AUTOSTART": true,
                                            "TTS_BACKEND": "edge"}}
        RESPONSES:
          • 200 {"ok": true, "applied": {name: coerced_value, ...}, "note": "…"}
          • 400 on empty body / unknown key / bad type (clear message)
          • 500 if the atomic file write itself fails
        The ``note`` is the honest restart caveat (SETTINGS_RESTART_NOTE) — we do
        NOT claim a live-apply we can't guarantee."""
        raw = self._read_body()
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            return self._send_json({"error": "invalid JSON body"}, code=400)
        if not isinstance(data, dict):
            return self._send_json({"error": "body must be a JSON object"},
                                   code=400)
        # Normalise the two accepted shapes into one {name: value} dict.
        updates: dict = {}
        if isinstance(data.get("settings"), dict):
            updates = dict(data["settings"])
        elif "name" in data:
            updates = {str(data["name"]): data.get("value")}
        if not updates:
            return self._send_json(
                {"error": "no settings to apply — send {name, value} or "
                          "{settings: {name: value, ...}}"}, code=400)
        # Validate + merge (raises SettingsWriteError → 400 on a bad key/value).
        try:
            applied = _write_settings(updates, cfg["user_settings_path"])
        except SettingsWriteError as e:
            return self._send_json({"error": str(e)}, code=400)
        except Exception as e:                      # a disk write failure etc.
            return self._send_json({"error": f"settings write failed: {e}"},
                                   code=500)
        return self._send_json({
            "ok": True,
            "applied": applied,
            "note": SETTINGS_RESTART_NOTE,
        })

    # ── JSON body helper for the control routes ─────────────────────────────
    def _json_object_body(self, *, require_json_type: bool = False):
        """(dict, None) or (None, (code, error)). ``require_json_type`` makes
        a non-JSON Content-Type a 415 - a cross-site HTML form cannot send
        application/json without a CORS preflight this server never grants."""
        if require_json_type:
            ctype = (self.headers.get("Content-Type", "") or "").lower()
            if "application/json" not in ctype:
                return None, (415, "Content-Type must be application/json")
        raw = self._read_body()
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            return None, (400, "invalid JSON body")
        if not isinstance(data, dict):
            return None, (400, "body must be a JSON object")
        return data, None

    # ── POST /api/action — run ONE registered action BY NAME ────────────────
    def _handle_post_action(self, cfg: dict) -> None:
        """Body ``{"name": str, "arg"?: str, "confirm"?: true}``. The action is
        looked up in the LIVE registry and called directly (a daemon thread,
        bounded wait) - never typed into the command channel, where the bare
        name used to be re-interpreted by the LLM. Names matching
        _ACTION_CONFIRM_RULES answer 409 until the call carries confirm:true;
        restart/shutdown-shaped names go through the tray control plane (the
        hardened teardown), never a request thread."""
        data, err = self._json_object_body(require_json_type=True)
        if err:
            return self._send_json({"error": err[1]}, code=err[0])
        code, payload = run_named_action(
            cfg, data.get("name"), data.get("arg", ""),
            confirm=data.get("confirm") is True)
        return self._send_json(payload, code=code)

    # ── POST /api/control — the tray control plane ──────────────────────────
    def _handle_post_control(self, cfg: dict) -> None:
        """Body ``{"cmd": one of TRAY_WEB_COMMANDS, "confirm"?: true}``. Appends
        to tray_commands.json exactly as the tray does, so it works in standby
        and without the LLM. restart needs confirm:true."""
        data, err = self._json_object_body(require_json_type=True)
        if err:
            return self._send_json({"error": err[1]}, code=err[0])
        cmd = data.get("cmd")
        if cmd not in TRAY_WEB_COMMANDS:
            return self._send_json({"error": "unknown control",
                                    "allowed": list(TRAY_WEB_COMMANDS)},
                                   code=400)
        if cmd in _TRAY_CONFIRM and data.get("confirm") is not True:
            return self._send_json({"error": "confirmation required",
                                    "confirm_required": True, "cmd": cmd},
                                   code=409)
        try:
            send_tray_command(cmd, cfg["tray_commands_path"])
        except Exception as e:
            return self._send_json({"error": f"control write failed: {e}"},
                                   code=500)
        _log_info(f"control {cmd} queued from the web dashboard")
        return self._send_json({"ok": True, "queued": cmd})

    # ── skill panels (core/web_panels.py) ───────────────────────────────────
    _PANEL_PATH_RE = re.compile(
        r"^/api/panel/([a-z0-9][a-z0-9_-]{0,39})/"
        r"(state|action|stream/([A-Za-z0-9][A-Za-z0-9_.-]{0,39}))$")

    def _get_panel(self, cfg: dict, path: str, query: dict | None = None) -> None:
        m = self._PANEL_PATH_RE.match(path)
        if not m or m.group(2) == "action":
            return self._send_json({"error": "not found"}, code=404)
        reg = _panel_registry(cfg)
        pid = m.group(1)
        if m.group(2) == "state":
            code, payload = reg.state(pid)
            return self._send_json(payload, code=code)
        src = reg.stream_source(pid, m.group(3))
        if src is None:
            return self._send_json({"error": "unknown panel stream"}, code=404)
        if (query or {}).get("still", [""])[0] in ("1", "true"):
            # One JPEG (the "snapshot" image widget) instead of a stream.
            try:
                data = src()
            except Exception:
                data = None
            if not isinstance(data, (bytes, bytearray)) or not data:
                return self._send_json({"error": "no frame"}, code=404)
            return self._send_bytes(bytes(data), "image/jpeg")
        return self._stream_panel(src)

    def _post_panel_action(self, cfg: dict, path: str) -> None:
        m = self._PANEL_PATH_RE.match(path)
        if not m or m.group(2) != "action":
            return self._send_json({"error": "not found"}, code=404)
        data, err = self._json_object_body(require_json_type=True)
        if err:
            return self._send_json({"error": err[1]}, code=err[0])
        code, payload = _panel_registry(cfg).call_action(
            m.group(1), data.get("name"), data.get("args", {}),
            confirm=data.get("confirm") is True)
        return self._send_json(payload, code=code)


# ── the dashboard page (single inline dark, arc-reactor-cyan HTML/JS) ────────

# The page is a PLAIN (raw) string, not an f-string: every brace in the CSS/JS
# is literal, and the one dynamic value - the JSON-serialised token - is
# spliced into _TOKEN_SLOT by _dashboard_html. (Until 2026-09-30 this was an
# f-string with every JS/CSS brace doubled, which made each edit a trap.)
_TOKEN_SLOT = "__JARVIS_TOKEN_JSON__"
_DASHBOARD_PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>JARVIS — Live</title>
<style>
  /* --muted was #5b7a86 (4.3:1 on the page background - under the 4.5:1 WCAG
     AA floor for body text, 2026-09-30 audit); #86a3ae is ~7:1. */
  :root { --cyan:#22d3ee; --cyan-dim:#0e7490; --bg:#05080d; --panel:#0b1420;
           --edge:#1b3040; --text:#cfe9f2; --muted:#86a3ae; --warn:#f5b942;
           --bad:#ff6b6b; --good:#2ee6a6; color-scheme:dark; }
  * { box-sizing:border-box; }
  html { -webkit-text-size-adjust:100%; }
  body { margin:0; min-height:100vh;
          background:var(--bg) radial-gradient(1200px 600px at 50% -10%, #0a1a26 0%, var(--bg) 60%) no-repeat;
          color:var(--text); font:14px/1.5 ui-monospace,Menlo,Consolas,monospace; }
  /* THE PHONE LAYOUT (2026-09-30 audit). At 390 px the page was 583 px wide
     and the tabs sat off-screen: flex-wrap was on <header>, so the whole
     <nav> wrapped as ONE unbreakable 560 px item. The nav now scrolls
     horizontally inside its own row (min-width:0 lets it shrink below its
     content), and the sticky header stays two short rows on a phone. */
  header { display:flex; align-items:center; gap:10px 14px; padding:10px 18px;
            border-bottom:1px solid var(--edge); position:sticky; top:0; z-index:20;
            background:rgba(5,8,13,.94); backdrop-filter:blur(6px);
            flex-wrap:wrap; }
  .reactor { width:22px; height:22px; border-radius:50%; flex:0 0 auto;
              background:radial-gradient(circle at 50% 50%, #eafcff, var(--cyan) 45%, var(--cyan-dim) 70%, #04222b 100%);
              box-shadow:0 0 12px var(--cyan), 0 0 28px var(--cyan-dim); }
  h1 { font-size:15px; margin:0; letter-spacing:.28em; color:var(--cyan); white-space:nowrap; }
  .hdr-right { margin-left:auto; order:2; display:flex; align-items:center; gap:12px; }
  .wrap { max-width:1000px; margin:0 auto; padding:16px 18px 96px; }
  .strip { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:12px; }
  .chip { background:var(--panel); border:1px solid var(--edge); border-radius:8px;
           padding:8px 12px; min-width:120px; flex:0 1 auto; max-width:100%; }
  .chip .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.12em; }
  .chip .v { color:var(--cyan); font-size:15px; margin-top:2px; word-break:break-word; }
  .chip.warn { border-color:#7a5a12; } .chip.warn .v { color:var(--warn); }
  .chip.bad { border-color:#7a2222; } .chip.bad .v { color:var(--bad); }
  .chip.wide { flex:1 1 260px; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px;
          background:#666; vertical-align:middle; }
  .dot.on { background:var(--good); box-shadow:0 0 8px var(--good); }
  .banner { border:1px solid #7a5a12; background:#1a1405; color:var(--warn);
            border-radius:10px; padding:10px 14px; margin:0 0 12px; display:flex;
            flex-wrap:wrap; align-items:center; gap:10px; }
  .banner[hidden] { display:none; }
  #log { background:#04070c; border:1px solid var(--edge); border-radius:8px;
          height:52vh; overflow:auto; padding:10px 12px; white-space:pre-wrap;
          font-size:12.5px; color:#a9c7d1; }
  #log .ln { min-height:1.2em; }
  #log .a { color:var(--cyan); }
  #log .j { color:#eafcff; }
  #log.hide-noise .noise { display:none; }
  #log .ln[hidden] { display:none; }
  .logbar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:8px 0; }
  .logbar .search { flex:1 1 200px; margin:0; }
  form { display:flex; gap:8px; margin-top:14px; }
  input[type=text] { flex:1; min-width:0; background:#04070c; border:1px solid var(--edge);
           color:var(--text); border-radius:8px; padding:11px 12px; font:inherit; }
  input[type=text]:focus { outline:none; border-color:var(--cyan); box-shadow:0 0 0 1px var(--cyan-dim); }
  button { background:var(--cyan-dim); color:#eafcff; border:1px solid var(--cyan);
            border-radius:8px; padding:0 18px; min-height:36px; font:inherit; cursor:pointer; }
  button:hover { background:var(--cyan); color:#04222b; }
  button:disabled { opacity:.5; cursor:default; }
  button:focus-visible, input:focus-visible, select:focus-visible { outline:2px solid var(--cyan); outline-offset:2px; }
  button.danger { background:transparent; color:var(--bad); border-color:#a33; }
  button.danger:hover { background:#a33; color:#fff; }
  button.on { background:var(--cyan); color:#04222b; }
  /* Quick-action row: a horizontal, wrapping strip of preset-command buttons.
     They reuse the base button look but are smaller/pill-shaped and ghosted
     (transparent fill) so the primary Send button stays the visual anchor. */
  .actions, .controls { display:flex; flex-wrap:wrap; gap:8px; margin:4px 0 8px; }
  .actions button, .controls button { padding:7px 13px; min-height:34px; font-size:12.5px; border-radius:999px;
            background:transparent; color:var(--cyan); }
  .actions button:hover, .controls button:hover { background:var(--cyan); color:#04222b; }
  .controls button.on { background:#3a2a05; color:var(--warn); border-color:var(--warn); }
  .controls button.danger { color:var(--bad); border-color:#a33; }
  /* Auto-refresh toggle in the header — a small inline checkbox + label. */
  .toggle { display:inline-flex; align-items:center; gap:6px; color:var(--muted);
            font-size:12px; cursor:pointer; user-select:none; }
  .toggle input { accent-color:var(--cyan); cursor:pointer; }
  #reply { margin-top:10px; color:#eafcff; min-height:1.4em; white-space:pre-wrap; }
  .muted { color:var(--muted); }
  .sr-only { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }
  /* ── Views ───────────────────────────────────────────────────────────────
     The nav toggles which section is visible; only one shows at a time so the
     page stays a single self-contained screen. Same arc-reactor-cyan palette. */
  nav.views { order:1; display:flex; gap:8px; flex:1 1 auto; min-width:0; max-width:100%;
            flex-wrap:nowrap; overflow-x:auto; overscroll-behavior-x:contain;
            -webkit-overflow-scrolling:touch; scrollbar-width:thin; padding:2px 0; }
  nav.views button { flex:0 0 auto; padding:6px 14px; min-height:32px; font-size:12.5px; border-radius:999px;
            background:transparent; color:var(--cyan); white-space:nowrap; }
  nav.views button.active { background:var(--cyan); color:#04222b; }
  .view[hidden] { display:none; }
  /* The prominent wake-word switch sits at the top of Settings so it's the first
     thing the owner sees (the headline "do it all from the web" control). */
  .wakebanner { background:linear-gradient(90deg, #0b1a26, var(--panel));
            border:1px solid var(--cyan-dim); border-radius:10px;
            padding:14px 16px; margin-bottom:16px; display:flex;
            align-items:center; gap:14px; flex-wrap:wrap; }
  .wakebanner .lbl { color:var(--cyan); font-size:14px; letter-spacing:.06em; }
  .wakebanner .hint { color:var(--muted); font-size:12px; flex-basis:100%; }
  /* Each tab-group of settings is a titled card; rows stack inside it. */
  .sgroup { background:var(--panel); border:1px solid var(--edge);
            border-radius:10px; padding:6px 14px 12px; margin-bottom:14px; }
  .sgroup > h2 { font-size:12px; text-transform:uppercase; letter-spacing:.16em;
            color:var(--muted); margin:12px 2px 8px; }
  .srow { display:flex; align-items:flex-start; gap:12px; padding:9px 2px;
            border-top:1px solid #0c1a24; flex-wrap:wrap; }
  .srow:first-of-type { border-top:none; }
  .srow .meta { flex:1 1 260px; min-width:0; }
  .srow .meta .name { color:var(--text); word-break:break-word; }
  .srow .meta .help { color:var(--muted); font-size:12px; margin-top:2px; }
  .srow .ctl { flex:0 1 auto; display:flex; align-items:center; gap:8px; flex-wrap:wrap; max-width:100%; }
  .srow .ctl input[type=text], .srow .ctl input[type=number], .srow .ctl input[type=password], .srow .ctl select {
            background:#04070c; border:1px solid var(--edge); color:var(--text);
            border-radius:8px; padding:8px 10px; font:inherit; min-width:0; width:200px; max-width:100%; }
  .srow .ctl input:focus, .srow .ctl select:focus { outline:none;
            border-color:var(--cyan); box-shadow:0 0 0 1px var(--cyan-dim); }
  .srow .ctl input[type=checkbox] { width:18px; height:18px;
            accent-color:var(--cyan); cursor:pointer; }
  .srow .save { padding:7px 12px; font-size:12px; }
  .srow .saved { color:var(--good); font-size:12px; min-width:1em; }
  #settingsNote { color:var(--muted); font-size:12px; margin:2px 2px 14px; }
  /* ── Control-panel tabs (System / Actions / Voice / Camera / Memory) ───────
     All reuse the shared palette + the .view[hidden] show/hide mechanic. */
  /* System: a responsive grid of GPU cards + a stat row. */
  .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(min(240px,100%),1fr));
            gap:12px; margin-bottom:14px; }
  .card { background:var(--panel); border:1px solid var(--edge);
           border-radius:10px; padding:12px 14px; min-width:0; }
  .card h3 { font-size:13px; margin:0 0 8px; color:var(--cyan);
             word-break:break-word; }
  .card .kv { display:flex; justify-content:space-between; gap:10px;
              font-size:12.5px; padding:2px 0; color:#a9c7d1; }
  .card .kv b { color:var(--text); font-weight:normal; text-align:right; word-break:break-word; }
  /* A thin VRAM/usage bar: a filled inner track sized by percentage. */
  .bar { height:8px; border-radius:6px; background:#04070c;
          border:1px solid var(--edge); overflow:hidden; margin:6px 0 8px; }
  .bar > i { display:block; height:100%; background:linear-gradient(90deg,
          var(--cyan-dim), var(--cyan)); }
  /* A shared search box for the Actions + Memory lists. */
  .search { width:100%; background:#04070c; border:1px solid var(--edge);
          color:var(--text); border-radius:8px; padding:10px 12px; font:inherit;
          margin-bottom:10px; }
  .search:focus { outline:none; border-color:var(--cyan);
          box-shadow:0 0 0 1px var(--cyan-dim); }
  /* A scrollable list panel (Actions / Memory facts / episodes). */
  .listbox { background:#04070c; border:1px solid var(--edge); border-radius:8px;
          max-height:56vh; overflow:auto; }
  .lrow { display:flex; align-items:center; gap:10px; padding:8px 12px;
          border-top:1px solid #0c1a24; flex-wrap:wrap; }
  .lrow:first-child { border-top:none; }
  .lrow .nm { flex:1 1 180px; min-width:0; color:var(--text); word-break:break-word; cursor:pointer; }
  .lrow .nm:hover { color:var(--cyan); }
  .lrow .txt { flex:1; min-width:0; color:#a9c7d1; word-break:break-word; }
  .lrow .arg { flex:0 1 150px; min-width:0; padding:6px 8px; font-size:12px; }
  /* A small speak-class chip on each action row. */
  .schip { font-size:10.5px; letter-spacing:.06em; padding:2px 8px;
          border-radius:999px; border:1px solid var(--edge); color:var(--muted);
          white-space:nowrap; }
  .schip.verbatim { color:var(--good); border-color:#12604a; }
  .schip.informative { color:var(--cyan); border-color:var(--cyan-dim); }
  .schip.confirm { color:var(--warn); border-color:#7a5a12; }
  .lrow .send { padding:5px 12px; min-height:30px; font-size:12px; border-radius:999px;
          background:transparent; color:var(--cyan); }
  .lrow .send:hover { background:var(--cyan); color:#04222b; }
  .count { color:var(--muted); font-size:12px; margin:2px 2px 10px; }
  #actionResult { white-space:pre-wrap; color:#eafcff; margin:0 2px 10px; min-height:1.2em; }
  /* Voice: a wrapping row of profile buttons + an info strip. */
  .voicebtns { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0; }
  .voicebtns button { padding:8px 14px; font-size:12.5px; border-radius:999px;
          background:transparent; color:var(--cyan); }
  .voicebtns button:hover { background:var(--cyan); color:#04222b; }
  .voicebtns button.off { color:#ff7ad9; border-color:#a05; }
  /* Per-camera grid: each tile independent (one dead cam never blanks the
     row); wraps to a column on narrow windows. The tiles are BUILT from
     /api/camera-tiles (the live CAMERAS roster), never hard-coded. */
  .camgrid { display:flex; gap:12px; flex-wrap:wrap; }
  .camtile { flex:1 1 220px; min-width:0; margin:0;
          background:var(--panel); border:1px solid var(--edge);
          border-radius:10px; padding:8px; }
  .camtile img { width:100%; border-radius:6px; background:#04070c;
          display:none; }
  /* The placeholder is the tile's EXPLANATION, not the word "off" (see
     _camera_off_reason). Sized to be read at a glance from across the room:
     13px/1.5 in the main text colour. It wraps - a wrapped honest sentence
     beats a truncated one - and holds its height so a tile that goes dark
     doesn't jump the row. */
  .camtile .camoff { border:1px dashed var(--cyan-dim); border-radius:6px;
          padding:18px 10px; text-align:center; color:var(--text);
          font-size:13px; line-height:1.5; min-height:74px;
          display:flex; flex-direction:column; align-items:center; justify-content:center; gap:6px; }
  /* The reason's DETAIL is shown, not hidden in a hover title (a phone has no
     hover): the gate's countdown, the bridge's error text. */
  .camtile .camdetail { color:var(--muted); font-size:12px; line-height:1.4; word-break:break-word; }
  .camtile figcaption { color:var(--muted); font-size:12px; margin-top:6px;
          letter-spacing:.08em; text-transform:uppercase; }
  /* ── Skill panels (core/web_panels.py): a generic widget renderer ────── */
  .pgrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(min(220px,100%),1fr)); gap:12px; }
  .pw { background:var(--panel); border:1px solid var(--edge); border-radius:10px; padding:10px 12px; min-width:0; }
  .pw.span { grid-column:1 / -1; }
  .pw .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.12em; margin-bottom:4px; }
  .pw .v { color:var(--cyan); font-size:18px; word-break:break-word; }
  .pw .badge { display:inline-block; padding:2px 10px; border-radius:999px; border:1px solid var(--edge); }
  .pw .badge.ok { color:var(--good); border-color:#12604a; }
  .pw .badge.warn { color:var(--warn); border-color:#7a5a12; }
  .pw .badge.bad { color:var(--bad); border-color:#7a2222; }
  .pw .badge.info { color:var(--cyan); border-color:var(--cyan-dim); }
  .pw pre { margin:0; white-space:pre-wrap; word-break:break-word; color:var(--text); font:inherit; }
  .pw ul { margin:0; padding-left:18px; max-height:220px; overflow:auto; }
  .pw img { width:100%; border-radius:6px; background:#04070c; }
  .pw img[hidden] { display:none; }
  /* An image widget before its first frame: a neutral box, never a broken-image icon. */
  .pw .pimg-empty { display:flex; align-items:center; justify-content:center; min-height:120px;
          border:1px dashed var(--edge); border-radius:6px; background:#04070c;
          color:var(--muted); font-size:13px; }
  .pw .pimg-empty[hidden] { display:none; }
  .pw .row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  .pw input[type=range] { width:100%; accent-color:var(--cyan); }
  .pw button { padding:8px 14px; }
  .pw button.hold { min-height:64px; min-width:96px; font-size:15px; touch-action:none;
          user-select:none; -webkit-user-select:none; -webkit-touch-callout:none; }
  .pw button.hold.held { background:var(--cyan); color:#04222b; }
  .pstale { color:var(--warn); font-size:12px; margin:0 2px 8px; min-height:1.2em; }
  /* E-stops are PINNED: fixed to the viewport on every tab once a panel
     declares one, so a stop is never a scroll or a tab away. */
  #estopDock { position:fixed; right:14px; bottom:14px; z-index:50; display:flex;
          flex-direction:column; gap:8px; }
  #estopDock[hidden] { display:none; }
  #estopDock button { background:#b3121b; color:#fff; border:2px solid #ff9a9a; border-radius:14px;
          min-height:56px; min-width:120px; font-size:15px; font-weight:bold; letter-spacing:.08em;
          box-shadow:0 4px 18px rgba(0,0,0,.6); touch-action:manipulation; }
  #estopDock button:hover { background:#e01e28; color:#fff; }
  @media (max-width: 640px) {
    header { padding:8px 10px; gap:6px 10px; }
    h1 { font-size:13px; letter-spacing:.2em; }
    nav.views { order:3; flex:1 1 100%; }
    nav.views button { padding:5px 11px; }
    .toggle .lbltext { display:none; }
    .wrap { padding:10px 10px 96px; }
    .chip { min-width:0; flex:1 1 140px; padding:6px 10px; }
    .chip .v { font-size:13.5px; }
    #log { height:46vh; }
    form button { padding:0 12px; }
  }
  @media (prefers-reduced-motion: reduce) { * { scroll-behavior:auto !important; } }
</style></head><body>
<header><div class="reactor" aria-hidden="true"></div><h1>J.A.R.V.I.S.</h1>
  <div class="hdr-right">
    <label class="toggle" title="Pause/resume live status + log polling">
      <input id="autorefresh" type="checkbox" checked> <span class="lbltext">auto-refresh</span><span class="sr-only"> (auto-refresh)</span>
    </label>
    <span id="conn" class="muted" role="status">connecting…</span>
  </div>
  <!-- View switcher. Only one view is shown at a time so the page stays one
       self-contained screen. Skill panels (core/web_panels.py) append their own
       buttons here at load. On a phone this row scrolls sideways. -->
  <nav class="views" id="nav" aria-label="Views">
    <button id="navLive" class="active" type="button" aria-current="page">Live</button>
    <button id="navSystem" type="button">System</button>
    <button id="navActions" type="button">Actions</button>
    <button id="navVoice" type="button">Voice</button>
    <button id="navCamera" type="button">Camera</button>
    <button id="navMemory" type="button">Memory</button>
    <button id="navSettings" type="button">Settings</button>
  </nav>
</header>
<div class="wrap" id="wrap">
  <!-- ── LIVE VIEW: status strip / standby banner / controls / quick actions /
       log / command box ──────────────────────────────────────────────────── -->
  <section id="viewLive" class="view">
    <div class="strip" id="strip" aria-label="Status"></div>
    <div class="banner" id="standbyBanner" hidden role="status">
      <span id="standbyText">JARVIS is in standby: a typed command is ignored unless it starts with “JARVIS”.</span>
      <button type="button" id="standbyWake">Wake him</button>
    </div>
    <!-- The tray control plane (POST /api/control → tray_commands.json): works
         in standby and without the LLM. Restart asks first. -->
    <div class="controls" id="controls" aria-label="Controls"></div>
    <!-- Quick-action buttons are injected here from the QUICK_ACTIONS array below,
         so presets are edited in ONE data-driven place (no per-button markup). -->
    <div class="actions" id="actions" aria-label="Quick commands"></div>
    <div class="logbar">
      <label class="toggle" title="Hide [kinect-preview], [vad] timeout and [air-mouse] chatter">
        <input id="noiseToggle" type="checkbox" checked> hide noise
      </label>
      <input id="logSearch" class="search" type="text" autocomplete="off"
             placeholder="Search the log…" aria-label="Search the log">
    </div>
    <div id="log" class="muted hide-noise" role="log" aria-live="off" aria-label="Session log" tabindex="0">loading log…</div>
    <form id="say">
      <label for="text" class="sr-only">Command for JARVIS</label>
      <input id="text" type="text" autocomplete="off" placeholder="Type a command for JARVIS…" autofocus>
      <button id="send" type="submit">Send</button>
    </form>
    <div id="reply" aria-live="polite"></div>
  </section>

  <!-- ── SETTINGS VIEW (the full control panel) ──────────────────────────────
       The wake-word switch is pinned in the banner at the top; every other knob
       is rendered from /api/settings, grouped by tab, into #settingsGroups. The
       whole thing is built client-side from the schema so the panel never drifts
       from settings_window.SCHEMA (the single source of truth). The banner is
       the one LIVE switch: it drives REQUIRE_WAKE_MODE through the tray control
       plane, exactly like "wake word mode on/off" by voice. -->
  <section id="viewSettings" class="view" hidden>
    <div class="wakebanner">
      <label class="toggle" style="color:var(--cyan)">
        <input id="wakeToggle" type="checkbox"> <span class="lbl">Wake-word mode (answer only when addressed by name)</span>
      </label>
      <button id="wakeSave" class="save" type="button">Save</button>
      <span id="wakeSaved" class="saved" aria-live="polite"></span>
      <span class="hint">Only respond to commands that start with &ldquo;JARVIS&rdquo;
        (REQUIRE_WAKE_MODE) &mdash; applied live, the same as saying &ldquo;wake word mode
        on&rdquo;. Booting in standby and the neural wake detector are separate knobs below.</span>
    </div>
    <div id="settingsNote" class="muted">loading settings…</div>
    <div id="settingsGroups"></div>
  </section>

  <!-- ── SYSTEM VIEW (live hardware: GPUs / CPU / RAM / disks) ──────────────
       Auto-refreshes on a ~2s interval while visible (see showView/systemTimer). -->
  <section id="viewSystem" class="view" hidden>
    <div id="sysMeta" class="count">loading system…</div>
    <div class="cards" id="sysGpus"></div>
    <div class="cards" id="sysHost"></div>
  </section>

  <!-- ── ACTIONS VIEW (the "access everything" tab) ────────────────────────
       Built from the LIVE action registry (/api/actions). Send runs the action
       BY NAME through POST /api/action - side-effect / destructive names ask
       first. Clicking a name drops it into the Live command box instead. -->
  <section id="viewActions" class="view" hidden>
    <div id="actionsCount" class="count">loading actions…</div>
    <input id="actionsSearch" class="search" type="text" autocomplete="off"
           placeholder="Search actions… (name)" aria-label="Search actions">
    <div id="actionResult" aria-live="polite"></div>
    <div id="actionsList" class="listbox"></div>
  </section>

  <!-- ── VOICE VIEW (the REAL engine + voice, and one button per usable clone
       profile). Each button POSTs a spoken phrase to /api/say. -->
  <section id="viewVoice" class="view" hidden>
    <div id="voiceInfo" class="count">loading voices…</div>
    <div id="voiceBtns" class="voicebtns"></div>
  </section>

  <!-- ── CAMERA VIEW (live MJPEG streams, one connection per tile) ───────
       The tiles come from /api/camera-tiles: one per camera in the RUNNING
       JARVIS's CAMERAS roster, plus the Kinect when it is switched on (a bare
       web process falls back to left/right/kinect). There is deliberately NO
       unified /api/camera-preview tile — it serves the DEFAULT preview file,
       written from the PRIMARY camera, so it duplicated whichever named tile
       was primary (2026-09-04, owner-reported: "two of the cameras are the
       same"). Kinect INFRARED is not a tile: this pykinect2 build exposes
       has_new_infrared_frame but NOT get_last_infrared_frame. -->
  <section id="viewCamera" class="view" hidden>
    <!-- PLACEHOLDER TEXT (2026-09-04, owner: "add the kinect not detected
         message to the tile"). Each tile's placeholder carries a SENTENCE
         fetched from /api/camera-reason on the img `error` event - never on the
         happy path - and its detail line (the gate's countdown, the bridge's
         error) is shown under it, not only on hover. -->
    <div class="count" id="camNote">Live cameras from JARVIS's camera list — each tile updates while its camera is on.</div>
    <div class="camgrid" id="camgrid"></div>
  </section>

  <!-- ── MEMORY VIEW (long-term facts + recent episodes, searchable) ───────── -->
  <section id="viewMemory" class="view" hidden>
    <div id="memCount" class="count">loading memory…</div>
    <input id="memSearch" class="search" type="text" autocomplete="off"
           placeholder="Search facts…" aria-label="Search facts">
    <div id="memFacts" class="listbox"></div>
  </section>
  <!-- Skill panel views (core/web_panels.py) are appended here at load. -->
</div>
<div id="estopDock" hidden aria-label="Emergency stops"></div>
<script>
const TOKEN = __JARVIS_TOKEN_JSON__;
function hdr() { const h = {'Content-Type':'application/json'}; if (TOKEN) h['X-Auth-Token']=TOKEN; return h; }
function q(u) { return TOKEN ? (u + (u.includes('?')?'&':'?') + 'token=' + encodeURIComponent(TOKEN)) : u; }
// Per-viewer conveniences only (last tab, the noise filter). Storage can throw
// (private window, blocked site data), so every access is wrapped and the page
// works identically without it.
function lsGet(k, d) { try { const v = window.localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } }
function lsSet(k, v) { try { window.localStorage.setItem(k, v); } catch (e) {} }
async function postJSON(url, obj) {
  const r = await fetch(q(url), {method:'POST', headers:hdr(), body: JSON.stringify(obj||{})});
  let d = {};
  try { d = await r.json(); } catch (e) {}
  return {status: r.status, ok: r.ok, data: d};
}
const strip = document.getElementById('strip');
const logEl = document.getElementById('log');
const conn  = document.getElementById('conn');

function chip(k, v, cls) { const d=document.createElement('div'); d.className='chip' + (cls ? ' '+cls : '');
  d.innerHTML = '<div class="k"></div><div class="v"></div>';
  d.querySelector('.k').textContent = k;
  d.querySelector('.v').textContent = (v===''||v==null) ? '—' : v; return d; }

// Format a raw uptime-in-seconds float into a compact "2h13m" / "4m" / "45s".
function fmtUptime(secs) {
  if (secs==null || isNaN(secs)) return '';
  secs = Math.max(0, Math.floor(secs));
  const h = Math.floor(secs/3600), m = Math.floor((secs%3600)/60), s = secs%60;
  if (h) return h+'h'+String(m).padStart(2,'0')+'m';
  if (m) return m+'m'+String(s).padStart(2,'0')+'s';
  return s+'s';
}
function fmtGB(mb) { return (mb==null) ? '?' : (Math.round(mb/102.4)/10).toString(); }
function shortGpu(name) {
  return String(name||'').replace(/^NVIDIA\s+/i,'').replace(/^GeForce\s+/i,'');
}

// ── STATUS STRIP ─────────────────────────────────────────────────────────────
// Every chip is a field /api/status actually carries. "brain" is the REAL
// backend/model (hud llm_backend) - it used to show the model's last
// [intent:x] tag under a "model" label. Each GPU gets its own VRAM chip: the
// old single TOTAL summed the LLM card with the second card.
let LAST_STATUS = null;
// The wake-word switch was flipped and not saved yet: a status poll must not
// overwrite what is about to be saved (declared here, before refreshStatus).
let wakeDirty = false;
async function refreshStatus() {
  try {
    const r = await fetch(q('/api/status'), {headers:hdr()});
    if (r.status===401) { conn.textContent='unauthorized — token required'; return; }
    const s = await r.json();
    LAST_STATUS = s;
    // The pinned wake-word switch shows what the RUNNING loop has
    // (require_wake_mode), not just what the settings file says.
    const wakeSw = document.getElementById('wakeToggle');
    if (wakeSw && typeof s.require_wake_mode === 'boolean' && !wakeDirty)
      wakeSw.checked = s.require_wake_mode;
    conn.textContent = ''; conn.innerHTML =
      '<span class="dot '+(s.running?'on':'')+'"></span>'+(s.running?'live':'offline');
    strip.innerHTML='';
    const st = chip('online', s.running?'live':'offline');
    st.querySelector('.v').innerHTML =
      '<span class="dot '+(s.running?'on':'')+'"></span>'+(s.running?'live':'offline');
    strip.appendChild(st);
    strip.appendChild(chip('version', s.version));
    strip.appendChild(chip('state', s.standby ? ('STANDBY' + (s.state && !/standby/i.test(s.state) ? ' ('+s.state+')' : '')) : s.state,
                           s.standby ? 'warn' : ''));
    // Uptime — only when the server could derive it (field present + non-null).
    if (s.uptime!=null) strip.appendChild(chip('uptime', fmtUptime(s.uptime)));
    const brain = chip('brain', s.model || '—');
    if (s.routing) brain.title = s.routing;
    strip.appendChild(brain);
    if (s.routing) strip.appendChild(chip('routing', s.routing));
    const doing = s.active_action ? ('running: ' + s.active_action) : (s.now_doing || '');
    if (doing) strip.appendChild(chip('now doing', doing, s.active_action ? 'wide' : ''));
    if ((s.gpus||[]).length) {
      for (const g of s.gpus) {
        const pct = (g.mem_used_mb!=null && g.mem_total_mb) ? Math.round(100*g.mem_used_mb/g.mem_total_mb) : null;
        strip.appendChild(chip('GPU' + g.index + ' ' + shortGpu(g.name),
          fmtGB(g.mem_used_mb) + ' / ' + fmtGB(g.mem_total_mb) + ' GB' + (pct!=null ? ' ('+pct+'%)' : ''),
          (pct!=null && pct >= 92) ? 'bad' : ((pct!=null && pct >= 85) ? 'warn' : '')));
      }
    } else {
      const g = (s.gpu_lines&&s.gpu_lines.length) ? s.gpu_lines[s.gpu_lines.length- (s.routing?2:1)] : '';
      strip.appendChild(chip('vram', (s.gpu_bar||'') + (g? '  '+g : '')));
    }
    strip.appendChild(chip('mic', s.mic_muted ? 'MUTED' : 'on', s.mic_muted ? 'warn' : ''));
    strip.appendChild(chip('voice out', s.tts_muted ? 'MUTED' : 'on', s.tts_muted ? 'warn' : ''));
    if (s.daemons_paused) strip.appendChild(chip('daemons', 'paused', 'warn'));
    // Air-mouse chip — ONLY present when build_status could read the skill in-process
    // (s.air_mouse is omitted otherwise). Shows ARMED (+engaged) vs disarmed.
    if (s.air_mouse) {
      const am = s.air_mouse.armed
        ? ('armed' + (s.air_mouse.engaged ? ' · engaged' : ''))
        : 'disarmed';
      strip.appendChild(chip('air-mouse', am));
    }
    if (s.now_playing) strip.appendChild(chip('now playing', s.now_playing));
    if (s.last_transcript) strip.appendChild(chip('last heard', s.last_transcript, 'wide'));
    if (s.last_spoken) strip.appendChild(chip('last said', s.last_spoken, 'wide'));
    updateControls(s);
  } catch(e) { conn.textContent = 'connection lost'; }
}

// ── CONTROLS (the tray control plane) ─────────────────────────────────────────
// POST /api/control appends to tray_commands.json exactly as the tray menu
// does; the monolith drains it at 2 Hz even in standby and before any LLM call.
const controlsEl = document.getElementById('controls');
const standbyBanner = document.getElementById('standbyBanner');
const CONTROLS = [
  {cmd:'force_wake',           label:() => 'Wake',  show:(s) => !s || s.standby},
  {cmd:'enter_standby',        label:() => 'Standby', show:(s) => !s || !s.standby},
  {cmd:'mute_tts_toggle',      label:(s) => (s && s.tts_muted) ? 'Unmute voice' : 'Mute voice', on:(s) => s && s.tts_muted},
  {cmd:'mic_mute_toggle',      label:(s) => (s && s.mic_muted) ? 'Unmute mic' : 'Mute mic', on:(s) => s && s.mic_muted},
  {cmd:'pause_daemons_toggle', label:(s) => (s && s.daemons_paused) ? 'Resume daemons' : 'Pause daemons', on:(s) => s && s.daemons_paused},
  {cmd:'restart',              label:() => 'Restart JARVIS', danger:true,
   confirm:'Restart JARVIS now?\n\nHe goes offline for about a minute and comes back on his own.'},
];
async function sendControl(cmd, confirmText) {
  if (confirmText && !window.confirm(confirmText)) return false;
  const res = await postJSON('/api/control', {cmd, confirm: !!confirmText});
  if (res.status===401) { replyEl.textContent='unauthorized'; return false; }
  if (!res.ok) { replyEl.textContent = 'control failed: ' + (res.data.error || res.status); return false; }
  replyEl.innerHTML = '<span class="muted">sent: ' + cmd.replace(/_/g,' ') + '</span>';
  setTimeout(refreshStatus, 900);
  return true;
}
function updateControls(s) {
  controlsEl.innerHTML = '';
  for (const c of CONTROLS) {
    if (c.show && !c.show(s)) continue;
    const b = document.createElement('button'); b.type='button';
    b.textContent = c.label(s);
    if (c.on && c.on(s)) { b.classList.add('on'); b.setAttribute('aria-pressed','true'); }
    else if (c.on) b.setAttribute('aria-pressed','false');
    if (c.danger) b.classList.add('danger');
    b.addEventListener('click', async () => { b.disabled = true;
      try { await sendControl(c.cmd, c.confirm); } finally { b.disabled = false; } });
    controlsEl.appendChild(b);
  }
  standbyBanner.hidden = !(s && s.standby);
}
document.getElementById('standbyWake').addEventListener('click', () => sendControl('force_wake'));

// ── THE LOG ───────────────────────────────────────────────────────────────────
// APPEND-ONLY (2026-09-30 audit): the view used to be rebuilt from 200 lines
// every second, which destroyed any text selection. The server now hands back
// an offset; each poll asks only for what came after it (?since=) and appends
// those lines. A noise filter (on by default) hides the chatty lines, and the
// search box hides non-matching lines - both by toggling classes on existing
// nodes, never by re-rendering.
const LOG_MAX_LINES = 1500;
const NOISE_RE = /\[kinect-preview\]|\[vad\] timeout|\[air-mouse\]/i;
const noiseToggle = document.getElementById('noiseToggle');
const logSearch = document.getElementById('logSearch');
let logState = {name: '', offset: null, empty: true};
let pinned = true;
noiseToggle.checked = lsGet('jarvis.hideNoise', '1') === '1';
logEl.classList.toggle('hide-noise', noiseToggle.checked);
noiseToggle.addEventListener('change', () => {
  logEl.classList.toggle('hide-noise', noiseToggle.checked);
  lsSet('jarvis.hideNoise', noiseToggle.checked ? '1' : '0');
});
function logMatches(el) {
  const f = (logSearch.value || '').trim().toLowerCase();
  return !f || el.textContent.toLowerCase().indexOf(f) !== -1;
}
let logSearchTimer = null;
logSearch.addEventListener('input', () => {
  clearTimeout(logSearchTimer);
  logSearchTimer = setTimeout(() => {
    for (const el of logEl.children) el.hidden = !logMatches(el);
  }, 120);
});
logEl.addEventListener('scroll', () => {
  pinned = (logEl.scrollTop + logEl.clientHeight) >= (logEl.scrollHeight - 24);
});
function selectionInLog() {
  try {
    const sel = window.getSelection();
    return !!(sel && !sel.isCollapsed && sel.anchorNode && logEl.contains(sel.anchorNode));
  } catch (e) { return false; }
}
function logLine(text) {
  const d = document.createElement('div');
  d.className = 'ln' + (/\[action\]/i.test(text) ? ' a' : (/jarvis( \(spoken\))?:/i.test(text) ? ' j' : ''))
                     + (NOISE_RE.test(text) ? ' noise' : '');
  d.textContent = text;
  if (!logMatches(d)) d.hidden = true;
  return d;
}
async function refreshLog() {
  try {
    let url = '/api/log/tail?lines=200';
    if (logState.offset !== null && logState.name)
      url += '&since=' + logState.offset + '&log=' + encodeURIComponent(logState.name);
    const r = await fetch(q(url), {headers:hdr()});
    if (r.status===401) return;
    const d = await r.json();
    if (!d.append || logState.empty) { logEl.textContent = ''; logState.empty = false; }
    logEl.classList.remove('muted');
    const lines = d.lines || [];
    if (lines.length) {
      const frag = document.createDocumentFragment();
      for (const l of lines) frag.appendChild(logLine(l));
      logEl.appendChild(frag);
      // Trim from the top; a selection inside the kept lines survives.
      while (logEl.childElementCount > LOG_MAX_LINES) logEl.removeChild(logEl.firstElementChild);
    }
    if (!logEl.childElementCount) {
      logEl.innerHTML = '<span class="muted">(no log yet)</span>';
      logState.empty = true;
    }
    logState.name = d.log || '';
    logState.offset = (typeof d.offset === 'number') ? d.offset : null;
    if (pinned && lines.length && !selectionInLog()) logEl.scrollTop = logEl.scrollHeight;
  } catch(e) {}
}

const form = document.getElementById('say');
const textIn = document.getElementById('text');
const sendBtn = document.getElementById('send');
const replyEl = document.getElementById('reply');
const actionsEl = document.getElementById('actions');

// ── QUICK-ACTION PRESETS ──────────────────────────────────────────────────
// Data-driven so presets are trivial to edit HERE (one array) without touching
// markup or handlers. `label` is the button text; `cmd` is the exact phrase POSTed
// to /api/say — the SAME inject channel a spoken command uses, so "mouse control on"
// behaves identically typed, clicked, or spoken.
const QUICK_ACTIONS = [
  {label:'Arm mouse control', cmd:'mouse control on'},
  {label:'Release mouse',     cmd:'mouse control off'},
  {label:"What's my status",  cmd:'system status'},
  {label:'Go to sleep',       cmd:'go to sleep'},
  {label:'Wake up',           cmd:'wake up'},
];

// STANDBY GUARD (2026-09-30 audit). In standby the main loop DROPS a typed
// command that does not start with the wake word ("[standby] ignored"), and
// the page used to report that as "accepted". Ask first: wake him through the
// tray channel (which works in standby) and then send, or don't send.
// A command that DOES start with the wake word goes straight through: the
// standby handler runs "Jarvis, <command>" as the turn (2026-10-01), so there
// is nothing to wake first.
const WAKE_WORD_RE = /^\s*(hey\s+)?jarvis\b/i;
const WAKE_UP_RE = /^\s*wake(\s+up)?\s*[.!]?\s*$/i;
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// The ONE code path every command goes through — the typed form and every quick
// button both call this. Disables the sender, shows a pending marker, POSTs the
// phrase, renders the reply (or a queued/accepted note), and nudges the log.
async function sendCommand(text, opts) {
  text = (text||'').trim(); if (!text) return;
  opts = opts || {};
  const btn = opts.button || null;
  // "wake up" typed in standby would itself be ignored (only the wake WORD
  // wakes him), so it goes straight to the control plane instead.
  if (LAST_STATUS && LAST_STATUS.standby && WAKE_UP_RE.test(text)) {
    await sendControl('force_wake'); return;
  }
  if (LAST_STATUS && LAST_STATUS.standby && !WAKE_WORD_RE.test(text)) {
    const wake = window.confirm('JARVIS is in standby, so he would ignore "' + text + '".\n\n'
      + 'OK = wake him first, then send it.\nCancel = do not send.');
    if (!wake) { replyEl.innerHTML = '<span class="muted">not sent (standby).</span>'; return; }
    if (!(await sendControl('force_wake'))) return;
    await sleep(1500);
  }
  sendBtn.disabled = true; if (btn) btn.disabled = true;
  replyEl.textContent = '…';
  try {
    const r = await fetch(q('/api/say'), {method:'POST', headers:hdr(),
      body: JSON.stringify({text})});
    const d = await r.json();
    if (r.status===401) replyEl.textContent = 'unauthorized';
    else if (d.status==='standby') replyEl.innerHTML = '<span class="muted">JARVIS is in standby and ignored that — press Wake, then send it again.</span>';
    else if (d.reply) replyEl.textContent = d.reply;
    else if (d.status==='no_log') replyEl.innerHTML = '<span class="muted">queued — JARVIS is not running; it will run on next boot.</span>';
    else replyEl.innerHTML = '<span class="muted">accepted (no spoken reply captured).</span>';
  } catch(e) { replyEl.textContent = 'send failed'; }
  finally { sendBtn.disabled=false; if (btn) btn.disabled=false; refreshLog(); }
}

// Render the quick-action buttons from QUICK_ACTIONS. Each POSTs its preset phrase
// via the shared sendCommand(); the returned reply lands in the existing #reply area.
QUICK_ACTIONS.forEach(a => {
  const b = document.createElement('button');
  b.type = 'button'; b.textContent = a.label;
  b.addEventListener('click', () => sendCommand(a.cmd, {button:b}));
  actionsEl.appendChild(b);
});

form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const t = textIn.value;
  textIn.value='';
  await sendCommand(t);
  textIn.focus();
});

// ── AUTO-REFRESH TOGGLE ────────────────────────────────────────────────────
// The checkbox (default ON) gates the polls so the user can FREEZE the view.
// A HIDDEN tab (another app in front, a locked phone) polls nothing at all:
// pollsWanted() is false until it is visible again, then it catches up at once.
const autoEl = document.getElementById('autorefresh');
function autoOn() { return autoEl.checked; }
function pollsWanted() { return autoOn() && !document.hidden; }
autoEl.addEventListener('change', () => { if (autoOn()) { refreshStatus(); refreshLog(); } });

// ── SETTINGS CONTROL PANEL ─────────────────────────────────────────────────
// The full "do it all from the web" panel. It's built ENTIRELY from /api/settings
// (which serves settings_window.SCHEMA + live values), so it never drifts from the
// real config. Each control saves INDEPENDENTLY via POST /api/settings {name,value}
// and shows a per-row confirmation. A save writes the file; the effect lands on the
// next JARVIS restart (the note the server returns says so).
const navLive = document.getElementById('navLive');
const navSettings = document.getElementById('navSettings');
const viewLive = document.getElementById('viewLive');
const viewSettings = document.getElementById('viewSettings');
const settingsGroups = document.getElementById('settingsGroups');
const settingsNote = document.getElementById('settingsNote');
const wakeToggle = document.getElementById('wakeToggle');
const wakeSave = document.getElementById('wakeSave');
const wakeSaved = document.getElementById('wakeSaved');

// Friendly tab titles for the group headings (fallback to the raw key).
const TAB_TITLES = { voice:'Voice', hearing:'Hearing & Mic', ai:'AI & Models',
  cameras:'Cameras & Kinect', privacy:'Privacy', integrations:'Integrations',
  advanced:'Advanced' };
// The wake-word knob the banner switch drives — the headline control the owner
// asked for. His "wake-word mode" is REQUIRE_WAKE_MODE (respond only when
// addressed by name, 2026-10-01): the banner applies it LIVE through the tray
// control plane and shows the running loop's value (status require_wake_mode).
// START_IN_STANDBY (boot in standby) and WAKE_WORD_AUTOSTART (the neural
// detector) are ordinary rows below.
const WAKE_KEY = 'REQUIRE_WAKE_MODE';
let settingsLoaded = false;

// Element refs for the five control-panel tabs.
const navSystem  = document.getElementById('navSystem');
const navActions = document.getElementById('navActions');
const navVoice   = document.getElementById('navVoice');
const navCamera  = document.getElementById('navCamera');
const navMemory  = document.getElementById('navMemory');
const viewSystem  = document.getElementById('viewSystem');
const viewActions = document.getElementById('viewActions');
const viewVoice   = document.getElementById('viewVoice');
const viewCamera  = document.getElementById('viewCamera');
const viewMemory  = document.getElementById('viewMemory');
const navEl = document.getElementById('nav');
const wrapEl = document.getElementById('wrap');

// One registry drives show/hide + nav-active for EVERY view (skill panels add
// themselves as 'panel:<id>'). Settings keeps its own settingsLoaded flag (the
// wake-word save resets it to force a reload); the other lazy tabs use per-tab
// loaded flags. System, Camera and panels run a while-visible refresh timer
// (stopped on leave) so their live data updates without touching other tabs.
const VIEWS = {
  live:     {nav:navLive,     view:viewLive},
  system:   {nav:navSystem,   view:viewSystem},
  actions:  {nav:navActions,  view:viewActions},
  voice:    {nav:navVoice,    view:viewVoice},
  camera:   {nav:navCamera,   view:viewCamera},
  memory:   {nav:navMemory,   view:viewMemory},
  settings: {nav:navSettings, view:viewSettings},
};
let currentView = 'live';
let systemTimer = null, cameraTimer = null, panelTimer = null;
let actionsLoaded = false, voiceLoaded = false, memoryLoaded = false;

function stopViewTimers() {
  if (systemTimer) { clearInterval(systemTimer); systemTimer = null; }
  if (cameraTimer) {
    clearInterval(cameraTimer); cameraTimer = null;
    // An MJPEG stream holds a server worker thread AND one of the browser's ~6
    // connections-per-origin for as long as the <img> keeps its src. Leaving the
    // Camera tab must hand both back.
    stopCameraStreams();
  }
  if (panelTimer) { clearInterval(panelTimer); panelTimer = null; }
  stopPanelMedia();
}

function showView(which) {
  if (!VIEWS[which]) which = 'live';
  currentView = which;
  lsSet('jarvis.view', which);
  Object.keys(VIEWS).forEach(k => {
    const on = (k === which);
    VIEWS[k].view.hidden = !on;
    VIEWS[k].nav.classList.toggle('active', on);
    if (on) VIEWS[k].nav.setAttribute('aria-current', 'page');
    else VIEWS[k].nav.removeAttribute('aria-current');
  });
  stopViewTimers();
  if (document.hidden) return;              // resumed by the visibilitychange handler
  if (which === 'settings') { if (!settingsLoaded) loadSettings(); }
  else if (which === 'system') {
    loadSystem();
    systemTimer = setInterval(() => { if (pollsWanted()) loadSystem(); }, 2000);
  }
  else if (which === 'actions') { if (!actionsLoaded) { loadActions(); actionsLoaded = true; } }
  else if (which === 'voice')   { if (!voiceLoaded)   { loadVoices();  voiceLoaded  = true; } }
  else if (which === 'memory')  { if (!memoryLoaded)  { loadMemory();  memoryLoaded = true; } }
  else if (which === 'camera')  {
    // refreshCamera() is a SUPERVISOR tick (re-arms dropped MJPEG streams once
    // their per-tile backoff allows, or polls stills in fallback mode) - it
    // does not set the frame rate.
    loadCameraTiles(true);
    cameraTimer = setInterval(refreshCamera, CAM_TICK_MS);
  }
  else if (which.indexOf('panel:') === 0) startPanelView(VIEWS[which].panel);
}
navLive.addEventListener('click', () => showView('live'));
navSystem.addEventListener('click', () => showView('system'));
navActions.addEventListener('click', () => showView('actions'));
navVoice.addEventListener('click', () => showView('voice'));
navCamera.addEventListener('click', () => showView('camera'));
navMemory.addEventListener('click', () => showView('memory'));
navSettings.addEventListener('click', () => showView('settings'));

// A hidden tab stops every stream and timer; coming back restarts the current
// view and catches the live data up at once.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { stopViewTimers(); return; }
  showView(currentView);
  if (autoOn()) { refreshStatus(); refreshLog(); }
});

// Build ONE control for a schema item, returning {el, read} where read() yields
// the value to POST. bool→checkbox, enum→select, combo→text+datalist, int/float→
// number, everything else→text. Every control carries an aria-label.
function buildControl(it) {
  const t = it.type;
  const label = it.label || it.name;
  // Secret knobs (e.g. the web token) never receive the live value from the
  // server — it's redacted. Render a password field; an EMPTY save means "keep
  // the current secret" (the click handler skips the POST), so a blank field
  // can't wipe an existing token. Type something to replace it.
  if (it.secret) {
    const inp=document.createElement('input'); inp.type='password';
    inp.autocomplete='new-password'; inp.setAttribute('aria-label', label);
    inp.placeholder = it.is_set ? '•••••• (set — type to replace)' : '(not set)';
    return {el:inp, read:()=>inp.value, secret:true};
  }
  if (t === 'bool') {
    const cb = document.createElement('input'); cb.type='checkbox';
    cb.setAttribute('aria-label', label);
    cb.checked = !!it.value; return {el:cb, read:()=>cb.checked};
  }
  if (t === 'enum') {
    const sel = document.createElement('select'); sel.setAttribute('aria-label', label);
    (it.choices||[]).forEach(c => { const o=document.createElement('option');
      o.value=c; o.textContent=c; if (String(it.value)===String(c)) o.selected=true;
      sel.appendChild(o); });
    return {el:sel, read:()=>sel.value};
  }
  if (t === 'int' || t === 'float') {
    const inp=document.createElement('input'); inp.type='number';
    inp.setAttribute('aria-label', label);
    if (t==='float') inp.step='any';
    inp.value = (it.value==null?'':it.value);
    return {el:inp, read:()=> t==='int'?parseInt(inp.value,10):parseFloat(inp.value)};
  }
  // combo (free text + suggestions), str, device, text, routing → a text input.
  // combo gets a datalist of its suggested choices; the user can still type any.
  const inp=document.createElement('input'); inp.type='text';
  inp.setAttribute('aria-label', label);
  let val = it.value;
  if (val && typeof val === 'object') val = JSON.stringify(val);   // routing/list → shown as JSON
  inp.value = (val==null?'':val);
  if (t === 'combo' && (it.choices||[]).length) {
    const dl=document.createElement('datalist'); const id='dl_'+it.name;
    dl.id=id; (it.choices||[]).forEach(c=>{const o=document.createElement('option');
      o.value=c; dl.appendChild(o);}); inp.setAttribute('list', id);
    const frag=document.createDocumentFragment(); frag.appendChild(inp); frag.appendChild(dl);
    return {el:frag, read:()=>inp.value, focusEl:inp};
  }
  return {el:inp, read:()=>inp.value};
}

// POST a single {name,value} and reflect the outcome in `saved` (a small span).
async function saveSetting(name, value, saved) {
  saved.textContent='…'; saved.style.color='var(--muted)';
  try {
    const r = await fetch(q('/api/settings'), {method:'POST', headers:hdr(),
      body: JSON.stringify({name, value})});
    const d = await r.json();
    if (r.ok && d.ok) { saved.textContent='saved ✓'; saved.style.color='#2ee6a6';
      if (d.note) settingsNote.textContent = d.note; }
    else { saved.textContent = (d.error||'error'); saved.style.color='#f85149'; }
  } catch(e) { saved.textContent='failed'; saved.style.color='#f85149'; }
}

// Render the whole panel from an /api/settings payload: group by tab, one card per
// tab, one row per knob (meta + control + per-row Save button + confirmation).
function renderSettings(payload) {
  settingsGroups.innerHTML='';
  const items = payload.settings || [];
  settingsNote.textContent = payload.note || '';
  const tabs = payload.tabs && payload.tabs.length ? payload.tabs
    : Array.from(new Set(items.map(i=>i.tab)));
  tabs.forEach(tab => {
    const inTab = items.filter(i => i.tab === tab);
    if (!inTab.length) return;
    const group=document.createElement('div'); group.className='sgroup';
    const h=document.createElement('h2'); h.textContent = TAB_TITLES[tab]||tab;
    group.appendChild(h);
    inTab.forEach(it => {
      const row=document.createElement('div'); row.className='srow';
      const meta=document.createElement('div'); meta.className='meta';
      meta.innerHTML = '<div class="name"></div>'+(it.help?'<div class="help"></div>':'');
      meta.querySelector('.name').textContent = it.label + '  ('+it.name+')';
      if (it.help) meta.querySelector('.help').textContent = it.help;
      const ctl=document.createElement('div'); ctl.className='ctl';
      const c = buildControl(it);
      ctl.appendChild(c.el);
      const saveBtn=document.createElement('button'); saveBtn.className='save';
      saveBtn.type='button'; saveBtn.textContent='Save';
      saveBtn.setAttribute('aria-label', 'Save ' + (it.label || it.name));
      const saved=document.createElement('span'); saved.className='saved';
      saved.setAttribute('aria-live', 'polite');
      // Saved-but-not-yet-live: the file diverges from the running loop's
      // constant, so be honest that this value applies on the next restart.
      if (it.pending_restart) { saved.textContent='pending restart';
        saved.style.color='var(--muted)'; }
      saveBtn.addEventListener('click', () => {
        const v = c.read();
        // Empty save on a secret = "keep the current value" — never POST "" and
        // wipe an existing token by accident.
        if (c.secret && (v===''||v==null)) {
          saved.textContent='unchanged'; saved.style.color='var(--muted)'; return;
        }
        saveSetting(it.name, v, saved);
      });
      ctl.appendChild(saveBtn); ctl.appendChild(saved);
      row.appendChild(meta); row.appendChild(ctl);
      group.appendChild(row);
      // Mirror the wake-word row into the top banner switch — only while the
      // running loop has not published its live value (that one wins; see
      // refreshStatus).
      if (it.name === WAKE_KEY && !wakeDirty
          && !(LAST_STATUS && typeof LAST_STATUS.require_wake_mode === 'boolean'))
        wakeToggle.checked = !!it.value;
    });
    settingsGroups.appendChild(group);
  });
}

async function loadSettings() {
  try {
    const r = await fetch(q('/api/settings'), {headers:hdr()});
    if (r.status===401) { settingsNote.textContent='unauthorized — token required'; return; }
    const d = await r.json();
    renderSettings(d);
    settingsLoaded = true;
  } catch(e) { settingsNote.textContent='could not load settings'; }
}

// The prominent banner switch: applies REQUIRE_WAKE_MODE LIVE through the tray
// control plane (_act_wake_word_mode_set: the running loop, core.config and the
// settings file - the same path as the voice command), then refreshes the live
// status and reloads the panel so the mirrored row shows the saved value.
wakeToggle.addEventListener('change', () => { wakeDirty = true; });
wakeSave.addEventListener('click', async () => {
  const on = wakeToggle.checked;
  wakeSaved.textContent = '';
  const ok = await sendControl(on ? 'wake_word_mode_on' : 'wake_word_mode_off');
  wakeDirty = false;
  wakeSaved.textContent = ok ? (on ? 'on — live' : 'off — live') : 'not sent';
  wakeSaved.style.color = ok ? '' : 'var(--bad)';
  setTimeout(() => { refreshStatus(); settingsLoaded = false; loadSettings(); }, 1200);
});

// ── SYSTEM TAB ──────────────────────────────────────────────────────────────
// Live hardware view: a card per GPU (VRAM bar + temp/util/power), a CPU/RAM
// card, and a card per disk. Auto-refreshed ~2s while visible (showView).
const sysMeta = document.getElementById('sysMeta');
const sysGpus = document.getElementById('sysGpus');
const sysHost = document.getElementById('sysHost');
function pctOf(u, t) { return (u!=null && t) ? Math.max(0, Math.min(100, Math.round(100*u/t))) : 0; }
function kvRow(k, v) { const d=document.createElement('div'); d.className='kv';
  d.innerHTML='<span></span><b></b>';
  d.querySelector('span').textContent=k; d.querySelector('b').textContent=v; return d; }
function renderSystem(s) {
  sysMeta.textContent = 'version ' + (s.version||'?')
    + (s.uptime!=null ? '  ·  up ' + fmtUptime(s.uptime) : '')
    + (s.routing ? '  ·  ' + s.routing : '');
  sysGpus.innerHTML='';
  (s.gpus||[]).forEach(g => {
    const c=document.createElement('div'); c.className='card';
    const p=pctOf(g.mem_used_mb, g.mem_total_mb);
    const h=document.createElement('h3');
    h.textContent='GPU ' + g.index + ' · ' + (g.name||''); c.appendChild(h);
    const bar=document.createElement('div'); bar.className='bar';
    const fill=document.createElement('i'); fill.style.width=p+'%'; bar.appendChild(fill);
    c.appendChild(bar);
    c.appendChild(kvRow('VRAM', (g.mem_used_mb!=null?g.mem_used_mb:'?') + ' / '
      + (g.mem_total_mb!=null?g.mem_total_mb:'?') + ' MB (' + p + '%)'));
    c.appendChild(kvRow('Util', g.util_pct!=null ? g.util_pct + '%' : 'n/a'));
    c.appendChild(kvRow('Temp', g.temp_c!=null ? g.temp_c + '°C' : 'n/a'));
    c.appendChild(kvRow('Power', g.power_w!=null ? g.power_w + ' W' : 'n/a'));
    sysGpus.appendChild(c);
  });
  if (!(s.gpus||[]).length) {
    const c=document.createElement('div'); c.className='card';
    const h=document.createElement('h3'); h.textContent='GPU'; c.appendChild(h);
    c.appendChild(kvRow('status', 'no nvidia-smi / no GPU')); sysGpus.appendChild(c);
  }
  sysHost.innerHTML='';
  const hc=document.createElement('div'); hc.className='card';
  const hh=document.createElement('h3'); hh.textContent='CPU / Memory'; hc.appendChild(hh);
  hc.appendChild(kvRow('CPU', s.cpu_pct!=null ? s.cpu_pct + '%' : 'n/a'));
  hc.appendChild(kvRow('RAM', (s.ram_used_gb!=null && s.ram_total_gb!=null)
    ? s.ram_used_gb + ' / ' + s.ram_total_gb + ' GB' : 'n/a'));
  sysHost.appendChild(hc);
  (s.disks||[]).forEach(dk => {
    const c=document.createElement('div'); c.className='card';
    const h=document.createElement('h3'); h.textContent='Disk ' + (dk.drive||''); c.appendChild(h);
    const used=(dk.total_gb!=null && dk.free_gb!=null) ? (dk.total_gb - dk.free_gb) : null;
    const p=pctOf(used, dk.total_gb);
    const bar=document.createElement('div'); bar.className='bar';
    const fill=document.createElement('i'); fill.style.width=p+'%'; bar.appendChild(fill);
    c.appendChild(bar);
    c.appendChild(kvRow('free', (dk.free_gb!=null?dk.free_gb:'?') + ' / '
      + (dk.total_gb!=null?dk.total_gb:'?') + ' GB'));
    sysHost.appendChild(c);
  });
}
async function loadSystem() {
  try {
    const r = await fetch(q('/api/system'), {headers:hdr()});
    if (r.status===401) { sysMeta.textContent='unauthorized — token required'; return; }
    renderSystem(await r.json());
  } catch(e) { sysMeta.textContent='could not load system info'; }
}

// ── ACTIONS TAB ─────────────────────────────────────────────────────────────
// The "access everything" list, from the LIVE registry (/api/actions). Send
// runs the action BY NAME through POST /api/action - it no longer types the
// bare name into the command channel for the LLM to reinterpret. Names the
// server flags `confirm` (side effects / destructive - see
// _ACTION_CONFIRM_RULES) ask first. Clicking a name drops it into the Live
// command box instead (edit before send).
const actionsCount = document.getElementById('actionsCount');
const actionsSearch = document.getElementById('actionsSearch');
const actionsList = document.getElementById('actionsList');
const actionResult = document.getElementById('actionResult');
let ALL_ACTIONS = [];
let ACTIONS_SOURCE = '';
function speakChipClass(sp) {
  const s=(sp||'').toUpperCase();
  if (s==='VERBATIM') return 'schip verbatim';
  if (s==='INFORMATIVE') return 'schip informative';
  return 'schip';
}
async function runAction(a, arg, btn) {
  let confirmed = false;
  if (a.confirm) {
    if (!window.confirm('Run "' + a.name + '"?\n\nThis action ' + (a.why || 'has side effects')
        + '. It runs directly — JARVIS does not double-check it first.')) return;
    confirmed = true;
  }
  if (btn) btn.disabled = true;
  actionResult.textContent = a.name + ' …';
  try {
    let res = await postJSON('/api/action', {name:a.name, arg:arg||'', confirm:confirmed});
    if (res.status === 409 && res.data.confirm_required && !confirmed) {
      if (!window.confirm('Run "' + a.name + '"?\n\nThis action ' + (res.data.why || 'has side effects') + '.')) {
        actionResult.textContent = a.name + ': not run.'; return; }
      res = await postJSON('/api/action', {name:a.name, arg:arg||'', confirm:true});
    }
    const d = res.data || {};
    if (res.status === 401) actionResult.textContent = 'unauthorized';
    else if (res.status === 503) actionResult.textContent = a.name + ': ' + (d.error || 'unavailable')
        + ' — type it in the Live command box instead.';
    else if (!res.ok) actionResult.textContent = a.name + ': ' + (d.error || ('error ' + res.status));
    else if (d.status === 'running') actionResult.textContent = a.name + ': started (still running).';
    else if (d.status === 'queued') actionResult.textContent = a.name + ': queued on the control channel.';
    else if (d.status === 'error') actionResult.textContent = a.name + ' failed: ' + (d.error || '');
    else actionResult.textContent = a.name + ': ' + (d.result || 'done.');
  } catch (e) { actionResult.textContent = a.name + ': send failed'; }
  finally { if (btn) btn.disabled = false; refreshLog(); }
}
function renderActions(filter) {
  const f=(filter||'').trim().toLowerCase();
  actionsList.innerHTML=''; let shown=0, matched=0;
  const frag=document.createDocumentFragment();
  for (const a of ALL_ACTIONS) {
    if (f && a.name.toLowerCase().indexOf(f)===-1) continue;
    matched++; if (shown>=400) continue;   // cap the DOM; the overflow row below advertises the rest
    const row=document.createElement('div'); row.className='lrow';
    const nm=document.createElement('div'); nm.className='nm'; nm.textContent=a.name;
    nm.title='Click to edit in the Live command box';
    nm.addEventListener('click', () => { textIn.value=a.name; showView('live'); textIn.focus(); });
    const chip=document.createElement('span'); chip.className=speakChipClass(a.spoken);
    chip.textContent=a.spoken;
    row.appendChild(nm); row.appendChild(chip);
    if (a.voice_only) {
      // Acts only on a fresh spoken / typed request: Run could never work.
      const v=document.createElement('span'); v.className='schip';
      v.textContent='voice only';
      v.title='Say or type it in the Live command box' + (a.use_instead ? ', or use ' + a.use_instead : '');
      row.appendChild(v);
    } else if (a.confirm) {
      const c=document.createElement('span'); c.className='schip confirm';
      c.textContent='asks first'; c.title='Confirm needed: ' + (a.why||''); row.appendChild(c);
    }
    const arg=document.createElement('input'); arg.type='text'; arg.className='arg';
    arg.placeholder='arg (optional)'; arg.setAttribute('aria-label', 'Argument for ' + a.name);
    const send=document.createElement('button'); send.type='button';
    send.className='send' + (a.confirm ? ' danger' : ''); send.textContent='Run';
    send.setAttribute('aria-label', 'Run ' + a.name);
    if (a.voice_only) { send.disabled = true; arg.disabled = true;
      send.title = 'Voice only' + (a.use_instead ? ' - use ' + a.use_instead : ''); }
    else send.addEventListener('click', () => runAction(a, arg.value, send));
    row.appendChild(arg); row.appendChild(send);
    frag.appendChild(row); shown++;
  }
  if (matched>shown) {
    const more=document.createElement('div'); more.className='lrow';
    const m=document.createElement('span'); m.className='muted';
    m.textContent='…'+(matched-shown)+' more — refine your search';
    more.appendChild(m); frag.appendChild(more);
  }
  actionsList.appendChild(frag);
  // The shown qualifier is UNCONDITIONAL: with an empty search the cap still
  // applies, so the header must never claim the full total while rows are cut.
  actionsCount.textContent = ALL_ACTIONS.length + ' actions  ·  ' + shown + ' shown'
    + (ACTIONS_SOURCE === 'live' ? '  ·  live registry' : (ACTIONS_SOURCE ? '  ·  from docs index (JARVIS not reachable)' : ''));
}
actionsSearch.addEventListener('input', () => renderActions(actionsSearch.value));
async function loadActions() {
  try {
    const r = await fetch(q('/api/actions'), {headers:hdr()});
    if (r.status===401) { actionsCount.textContent='unauthorized — token required'; return; }
    const d = await r.json();
    ACTIONS_SOURCE = d.source || '';
    ALL_ACTIONS = (d.actions||[]).slice().sort((a,b)=>a.name.localeCompare(b.name));
    renderActions('');
  } catch(e) { actionsCount.textContent='could not load actions'; }
}

// ── VOICE TAB ───────────────────────────────────────────────────────────────
// Shows the REAL engine + voice (it said "normal (en-GB-RyanNeural)" while
// Kokoro was speaking) and a button per USABLE clone profile (POSTs "switch to
// the <name> voice") plus a "normal voice" button (POSTs "voice cloning off").
// Cloning loads Chatterbox on the GPU beside the local LLM - with the 26B model
// resident that is a known VRAM overload - so a clone button asks first.
const voiceInfo = document.getElementById('voiceInfo');
const voiceBtns = document.getElementById('voiceBtns');
function cloneWarning(d, name) {
  const dev = d.clone_device ? (' on ' + d.clone_device) : '';
  const llm = (d.llm_local && d.local_model) ? (' with the local model ' + d.local_model + ' already loaded') : '';
  return 'Use the cloned "' + name + '" voice?\n\n'
    + 'Voice cloning loads the ' + (d.clone_model || 'Chatterbox') + ' model on the GPU' + dev + llm + '. '
    + 'On this machine that is a known VRAM overload (24 GB card): JARVIS can stall, lose his voice, or crash.\n\n'
    + 'Continue?';
}
function renderVoices(d) {
  const usable=(d.profiles||[]).filter(p=>p.usable);
  const active = d.summary || ((d.enabled && d.active) ? d.active
    : ((d.engine || d.tts_backend || 'default') + (d.voice ? ' · ' + d.voice : '')));
  voiceInfo.textContent = 'Active voice: ' + active
    + '  ·  engine ' + (d.engine || d.tts_backend || '?')
    + '  ·  ' + usable.length + ' usable profile(s)';
  voiceBtns.innerHTML='';
  usable.forEach(p => {
    const b=document.createElement('button'); b.type='button';
    b.textContent='Use ' + p.name + (p.source? ' ('+p.source+')':'');
    b.addEventListener('click', () => {
      if (!window.confirm(cloneWarning(d, p.name))) return;
      sendCommand('switch to the ' + p.name + ' voice', {button:b});
    });
    voiceBtns.appendChild(b);
  });
  if (!usable.length) {
    const note=document.createElement('span'); note.className='muted';
    note.style.alignSelf='center'; note.textContent='No usable clone profiles enrolled.  ';
    voiceBtns.appendChild(note);
  }
  const off=document.createElement('button'); off.type='button'; off.className='off';
  off.textContent='Normal voice (cloning off)';
  off.addEventListener('click', () => sendCommand('voice cloning off', {button:off}));
  voiceBtns.appendChild(off);
}
async function loadVoices() {
  try {
    const r = await fetch(q('/api/voices'), {headers:hdr()});
    if (r.status===401) { voiceInfo.textContent='unauthorized — token required'; return; }
    renderVoices(await r.json());
  } catch(e) { voiceInfo.textContent='could not load voices'; }
}

// ── CAMERA TAB ──────────────────────────────────────────────────────────────
// Each tile is an MJPEG STREAM (/api/camera-stream?cam=...): ONE connection the
// server pushes a new JPEG down the instant JARVIS writes one.
//
// THE TILES are built from /api/camera-tiles - the RUNNING JARVIS's CAMERAS
// roster plus the Kinect when it is switched on - instead of a hard-coded
// left/right/kinect row (2026-09-30 audit: a camera the owner had removed from
// CAMERAS showed "Webcam off" forever). The same endpoint carries the camera
// gate's verdict per tile ("retrying in N min"), refreshed every few seconds.
//
// On load the tile shows; on error (404 = missing/stale, or the server closing a
// stream because that camera went off) its own placeholder shows, so one dead
// camera never blanks the others (2026-07-10 contract, unchanged).
//
// BACKOFF (2026-09-30): a dead tile used to be re-armed every CAM_TICK_MS
// (4x a second, forever). Each tile now waits CAM_TICK_MS * 2^errors before
// its next try (capped at CAM_BACKOFF_MAX_MS), and no sooner than the camera
// gate says JARVIS itself will retry. A working tile is unaffected.
//
// FALLBACK: if streaming never works in this browser (no tile ever loaded from a
// stream and every tile has errored repeatedly) we drop back to polling the STILL
// endpoint - identical behaviour, same backoff.
const CAM_TICK_MS = 250;            // supervisor tick == still-poll interval
const CAM_BACKOFF_MAX_MS = 60000;   // a dead tile re-arms at most once a minute
const CAM_TILES_REFRESH_MS = 5000;  // gate verdicts / roster refresh
let   camMode = 'stream';           // 'stream' | 'poll'
let   camEverStreamed = false;      // a stream has delivered at least one frame
let   camPairs = [];                // [[img, off], ...] for the tiles on screen
let   camTiles = [];
let   camTilesAt = 0, camTilesBusy = false;
const camGrid = document.getElementById('camgrid');
const camNote = document.getElementById('camNote');

function camBackoffMs(errs) {
  return Math.min(CAM_BACKOFF_MAX_MS, CAM_TICK_MS * Math.pow(2, Math.max(0, errs)));
}
function camDetail(off, text) {
  const det = off.querySelector('.camdetail');
  if (det) det.textContent = text || '';
}
function camSay(off, message, detail) {
  const msg = off.querySelector('.cammsg');
  if (msg) msg.textContent = message; else off.textContent = message;
  camDetail(off, detail);
  off.title = detail || '';
}

function buildCameraTiles(payload) {
  const tiles = (payload && payload.tiles) || [];
  const sig = tiles.map(t => t.cam).join(',');
  if (camGrid.dataset.sig !== sig) {
    stopCameraStreams();
    camGrid.innerHTML = '';
    camPairs = tiles.map(t => {
      const fig = document.createElement('figure'); fig.className = 'camtile';
      const img = document.createElement('img');
      img.id = 'cam_' + t.cam; img.dataset.cam = t.cam; img.alt = t.label || t.cam;
      const off = document.createElement('div'); off.className = 'camoff';
      off.id = 'cam_' + t.cam + 'Off';
      off.innerHTML = '<div class="cammsg">checking…</div><div class="camdetail"></div>';
      const cap = document.createElement('figcaption'); cap.textContent = t.label || t.cam;
      fig.appendChild(img); fig.appendChild(off); fig.appendChild(cap);
      camGrid.appendChild(fig);
      wireCameraTile(img, off);
      return [img, off];
    });
    camTiles = camPairs.map(([img]) => img);
    camGrid.dataset.sig = sig;
    camNote.textContent = tiles.length
      ? ('Live cameras from ' + (payload.source === 'live' ? "JARVIS's camera list" : 'the default set (JARVIS not reachable)')
         + ' — each tile updates while its camera is on.')
      : 'No cameras are configured in JARVIS (CAMERAS is empty and the Kinect is off).';
  }
  // The gate's verdict per tile: shown under a DOWN tile's sentence, and the
  // earliest moment that tile may re-arm (JARVIS will not open it sooner).
  const byCam = {};
  for (const t of tiles) byCam[t.cam] = t;
  for (const [img, off] of camPairs) {
    const g = (byCam[img.dataset.cam] || {}).gate;
    img.dataset.gate = g ? g.state : '';
    if (g && g.retry_in_s != null) {
      const until = Date.now() + Math.min(CAM_BACKOFF_MAX_MS, 1000 * g.retry_in_s);
      if (img.dataset.streaming !== '1') img.dataset.nextTry = String(Math.max(+img.dataset.nextTry || 0, until));
    }
    if (g && off.style.display !== 'none' && img.dataset.streaming !== '1') {
      camDetail(off, g.message);
      // A tile the gate is holding may never be armed before its countdown,
      // so it would never fire `error` and never get its sentence: ask now.
      explainTile(img, off);
    }
  }
}
function loadCameraTiles(force) {
  const now = Date.now();
  if (camTilesBusy || (!force && now - camTilesAt < CAM_TILES_REFRESH_MS)) return;
  camTilesAt = now; camTilesBusy = true;
  fetch(q('/api/camera-tiles'), {headers:hdr()})
    .then(r => r.ok ? r.json() : null)
    .then(j => { if (j) buildCameraTiles(j); if (force) refreshCamera(); })
    .catch(() => {})
    .finally(() => { camTilesBusy = false; });
}

function wireCameraTile(img, off) {
  img.dataset.errs = '0';
  img.dataset.loadAt = '0';
  img.dataset.nextTry = '0';
  img.addEventListener('load',  () => {
    img.style.display='block'; off.style.display='none';
    img.dataset.errs = '0';
    img.dataset.nextTry = '0';
    img.dataset.loadAt = String(Date.now());
    // Drop the old explanation and the throttle: the NEXT outage may have a
    // different cause, and a stale sentence sitting behind a live tile is
    // exactly the kind of leftover that gets read as current.
    camSay(off, 'checking…', '');
    img.dataset.reasonAt = '0';
    if (camMode === 'stream' && img.dataset.streaming === '1') camEverStreamed = true;
  });
  img.addEventListener('error', () => {
    const errs = (+img.dataset.errs || 0) + 1;
    img.dataset.errs = String(errs);
    img.dataset.nextTry = String(Math.max(+img.dataset.nextTry || 0, Date.now() + camBackoffMs(errs)));
    // Streaming is broken in this browser only if it NEVER worked anywhere.
    if (camMode === 'stream' && !camEverStreamed && camPairs.length &&
        camPairs.every(([t]) => (+t.dataset.errs || 0) >= 3)) camMode = 'poll';
    tileDown(img, off);
  });
}

// WHY THE TILE ASKS THE SERVER
// ---------------------------------
// An <img> that fails to load throws away the response body, so the 404 from
// /api/camera-stream can never reach the placeholder - the `error` event is all
// the DOM gives us, and it carries no text. /api/camera-reason is therefore a
// second, tiny fetch fired ONLY here, on the error path. The streaming path is
// untouched: while a camera works this never runs, and neither does the
// device-enumeration probe behind it.
//
// Throttled per tile (REASON_MIN_MS), and the supervisor itself only ticks
// while the Camera tab is open (showView starts cameraTimer, stopViewTimers
// kills it), so a dashboard sitting on any other tab asks nothing and the
// device probe never fires at all.
const REASON_MIN_MS = 4000;
function explainTile(img, off) {
  const now = Date.now();
  if (img.dataset.reasonBusy === '1') return;
  if (now - (+img.dataset.reasonAt || 0) < REASON_MIN_MS) return;
  img.dataset.reasonAt = String(now);
  img.dataset.reasonBusy = '1';
  fetch(q('/api/camera-reason?cam=' + img.dataset.cam), {headers:hdr()})
    .then(r => r.ok ? r.json() : null)
    .then(j => {
      // Only overwrite while the tile is still DOWN: a frame may have arrived
      // between the error and this reply, and stamping "no picture" over a
      // working tile would be its own little lie.
      if (j && j.message && off.style.display !== 'none') {
        camSay(off, j.message, j.detail || '');
      }
      // A camera JARVIS will never write (not in CAMERAS / Kinect off) is
      // never re-armed: nothing will ever answer.
      if (j && (j.state === 'not_configured' || j.state === 'disabled')) {
        img.dataset.mode = 'dead';
        img.dataset.streaming = '';
      }
      // The camera GATE says when JARVIS itself will try again: re-arming the
      // tile before that only re-asks a question with a known answer.
      if (j && j.retry_in_s != null) {
        img.dataset.nextTry = String(Math.max(+img.dataset.nextTry || 0,
          Date.now() + Math.min(CAM_BACKOFF_MAX_MS, 1000 * j.retry_in_s)));
      }
      // THE WAY OUT.  An `error` clears dataset.streaming, so the supervisor
      // re-arms this tile - forever, if the stream is refused every time (8
      // slots = 3 tiles x 3 tabs).  camMode's own fallback cannot rescue it:
      // that needs !camEverStreamed, which the other working tiles have
      // already falsified.  So when the SERVER has established the camera is
      // producing frames, the broken part is this page's stream: this ONE tile
      // stops asking for one and polls stills instead.  'stream_busy' is proof
      // (all slots taken -> the next request IS a 503), so it demotes at once;
      // 'live' waits for 3 consecutive failures so a single dropped connection
      // does not cost the session its pushed frames.
      if (j && img.dataset.mode !== 'poll' && img.dataset.mode !== 'dead' &&
          (j.state === 'stream_busy' ||
           (j.state === 'live' && (+img.dataset.errs || 0) >= 3))) {
        img.dataset.mode = 'poll';
        img.dataset.streaming = '';
        img.dataset.nextTry = '0';
      }
    })
    .catch(() => {})
    .finally(() => { img.dataset.reasonBusy = ''; });
}

// ── MID-STREAM DEATH ────────────────────────────────────────────────────────
// THE DEFECT THIS FIXES (2026-09-05): a camera that died WHILE STREAMING left
// its tile showing a frozen, minutes-old picture with no placeholder and no
// sentence — the dashboard silently asserting a live camera. dataset.streaming
// stayed '1', so startCameraStreams() skipped the tile forever.
//
// WHY it could not be fixed in the `error` handler: measured against a real
// Chrome on this box, a multipart/x-mixed-replace <img> is told NOTHING when
// the server closes its stream. Three endings were tried — clean close after a
// complete part, a truncated final part, and boundary+headers then close — and
// all three produced exactly one event, `load` at 413 ms (just after the FIRST
// frame), with nothing at all at the close ~1.6 s later. img.complete stayed
// true and naturalWidth kept the last frame's size throughout, so no DOM
// property betrays the ending either. `load` is not per-frame and is not an
// end-of-stream signal, so NO event-driven design can see this.
//
// The tile is therefore supervised on our own clock. Once a second the tick
// asks /api/camera-live — three os.stat calls, the same staleness rule
// _stream_camera uses to decide when to close, and deliberately nowhere near
// the reason ladder or the ~0.7 s device probe.
//
// It may only ever take a tile DOWN. A fresh FILE is not proof that THIS
// browser is receiving frames, so bringing a tile up stays the stream's job.
const CAM_LIVE_MIN_MS = 1000;
let camLiveAt = 0, camLiveBusy = false;

// The ONE way a tile goes dark, shared by the `error` event and the supervisor,
// so a tile can never be hidden without also being explained.
function tileDown(img, off) {
  img.style.display='none'; off.style.display='flex';
  img.dataset.streaming = '';                         // let the tick re-arm it
  // Drop the frozen frame and hand back the connection slot. Guarded on the
  // attribute being there so that if a browser ever does fire `error` for this,
  // the re-entry stops after one pass instead of looping.
  if (img.hasAttribute('src')) img.removeAttribute('src');
  explainTile(img, off);
}

function checkCameraLiveness() {
  if (camMode !== 'stream') return;    // poll mode 404s on its own every tick
  const now = Date.now();
  if (camLiveBusy || now - camLiveAt < CAM_LIVE_MIN_MS) return;
  camLiveAt = now; camLiveBusy = true;
  const asked = now;
  fetch(q('/api/camera-live'), {headers:hdr()})
    .then(r => r.ok ? r.json() : null)
    .then(j => {
      if (!j || !j.cams) return;
      for (const [img, off] of camPairs) {
        if (j.cams[img.dataset.cam] !== false) continue;   // fresh, or unknown
        if (img.dataset.streaming !== '1') continue;       // already down
        // A frame landed after we asked, so this answer is out of date — say
        // nothing rather than blanking a camera that just came back.
        if ((+img.dataset.loadAt || 0) > asked) continue;
        tileDown(img, off);
      }
    })
    .catch(() => {})
    .finally(() => { camLiveBusy = false; });
}

function camMayTry(img) {
  return img.dataset.mode !== 'dead' && Date.now() >= (+img.dataset.nextTry || 0);
}
function startCameraStreams() {
  for (const img of camTiles) {
    if (img.dataset.mode === 'poll') continue;        // demoted: it polls stills
    if (img.dataset.streaming === '1') continue;      // already connected
    if (!camMayTry(img)) continue;                    // backing off / never configured
    img.dataset.streaming = '1';
    img.src = q('/api/camera-stream?cam=' + img.dataset.cam);
  }
}
function stopCameraStreams() {
  for (const img of camTiles) {
    img.dataset.streaming = '';
    img.removeAttribute('src');                       // aborts the connection
  }
}
function pollTile(img) {
  if (!camMayTry(img)) return;
  img.dataset.streaming = '';
  img.src = q('/api/camera-preview?cam=' + img.dataset.cam + '&t=' + Date.now());
}
function refreshCamera() {
  // One tick drives both modes: re-arm any dropped stream, or poll stills.
  if (!autoOn() || document.hidden) { stopCameraStreams(); return; }
  loadCameraTiles(false);
  if (camMode === 'stream') {
    // Supervise BEFORE re-arming. The DOM never reports a stream that ENDED
    // (measured — see the MID-STREAM DEATH note above), so this is the only
    // thing on the page that can notice a camera which died while streaming.
    checkCameraLiveness();
    startCameraStreams();
    // A tile demoted by the reason check polls while the REST of the page
    // still streams: its stream is the thing that is broken, not the others'.
    for (const img of camTiles) if (img.dataset.mode === 'poll') pollTile(img);
    return;
  }
  for (const img of camTiles) pollTile(img);
}

// ── MEMORY TAB ──────────────────────────────────────────────────────────────
// Fact + episode counts and a searchable, scrollable list of long-term facts.
const memCount = document.getElementById('memCount');
const memSearch = document.getElementById('memSearch');
const memFacts = document.getElementById('memFacts');
let ALL_FACTS = [];
function renderFacts(filter) {
  const f=(filter||'').trim().toLowerCase();
  memFacts.innerHTML=''; let shown=0, matched=0;
  const frag=document.createDocumentFragment();
  for (const fact of ALL_FACTS) {
    if (f && (fact.text||'').toLowerCase().indexOf(f)===-1) continue;
    matched++; if (shown>=400) continue;   // same DOM cap as renderActions — and the same honest overflow row
    const row=document.createElement('div'); row.className='lrow';
    const txt=document.createElement('div'); txt.className='txt'; txt.textContent=fact.text;
    row.appendChild(txt);
    if (fact.source) { const chip=document.createElement('span'); chip.className='schip';
      chip.textContent=fact.source; row.appendChild(chip); }
    frag.appendChild(row); shown++;
  }
  if (matched>shown) {
    const more=document.createElement('div'); more.className='lrow';
    const m=document.createElement('span'); m.className='muted';
    m.textContent='…'+(matched-shown)+' more — refine your search';
    more.appendChild(m); frag.appendChild(more);
  }
  memFacts.appendChild(frag);
  if (!ALL_FACTS.length) memFacts.innerHTML=
    '<div class="lrow"><span class="muted">no facts stored</span></div>';
}
memSearch.addEventListener('input', () => renderFacts(memSearch.value));
async function loadMemory() {
  try {
    const r = await fetch(q('/api/memory'), {headers:hdr()});
    if (r.status===401) { memCount.textContent='unauthorized — token required'; return; }
    const d = await r.json();
    ALL_FACTS = d.facts||[];
    const c = d.counts||{};
    memCount.textContent = (c.facts!=null?c.facts:ALL_FACTS.length) + ' facts · '
      + (c.episodes!=null?c.episodes:0) + ' episodes';
    renderFacts('');
  } catch(e) { memCount.textContent='could not load memory'; }
}

// ── SKILL PANELS (core/web_panels.py) ────────────────────────────────────────
// Every panel a skill declares in WEB_PANELS becomes a nav tab + a view built by
// ONE generic widget renderer from its metadata (/api/panels), so most panels
// need no custom JS. State is polled ONLY while that panel is visible (and the
// page is), at the panel's poll_ms. Actions go to POST /api/panel/<id>/action,
// which calls the skill's callable directly (never the LLM).
//
// HOLD buttons (momentary control) resend their action at the action's rate_hz
// while pressed, and send the panel's stop_action the moment the press ends:
// pointerup / pointercancel / pointerleave, the window losing focus, the tab
// being hidden, or the view being left. E-STOP buttons are pinned in a fixed
// dock visible on every tab, stop every hold first, and never ask to confirm.
let PANELS = [];
const panelUpdaters = {};        // panel id -> [fn(state)]
const panelMedia = {};           // panel id -> [image widget] (see PANEL IMAGES)
const HOLD_STOPPERS = [];        // every hold button's release function
const estopDock = document.getElementById('estopDock');

function stopAllHolds() { for (const s of HOLD_STOPPERS) { try { s(); } catch (e) {} } }
async function panelAction(p, name, args, opts) {
  opts = opts || {};
  const meta = (p.actions || {})[name] || {};
  let confirmed = false;
  if (meta.confirm && !opts.noConfirm) {
    if (!window.confirm((meta.danger ? 'DANGER — ' : '') + (meta.label || name) + '?\n\nRun it on "' + p.title + '"?')) return null;
    confirmed = true;
  }
  const out = document.getElementById('pstale_' + p.id);
  try {
    const res = await postJSON('/api/panel/' + encodeURIComponent(p.id) + '/action',
                               {name: name, args: args || {}, confirm: confirmed});
    if (!opts.quiet && out) {
      if (res.ok) out.textContent = '';
      else out.textContent = (meta.label || name) + ': ' + ((res.data && res.data.error) || res.status);
    } else if (!res.ok && res.status !== 429 && out) {
      out.textContent = (meta.label || name) + ': ' + ((res.data && res.data.error) || res.status);
    }
    // A deliberate action (never a hold's resend) may have produced a frame:
    // arm the panel's image widgets and re-read the state now.
    if (res.ok && !opts.quiet) {
      panelImagesArm(p);
      if (p.has_state && currentView === 'panel:' + p.id) loadPanelState(p);
    }
    return res;
  } catch (e) { if (out) out.textContent = (meta.label || name) + ': send failed'; return null; }
}
function bindHold(btn, p, w) {
  const rate = ((p.actions || {})[w.action] || {}).rate_hz || 4;
  let timer = null, active = false;
  const send = () => panelAction(p, w.action, w.args || {}, {quiet: true, noConfirm: true});
  const start = (ev) => {
    if (ev) ev.preventDefault();
    if (active) return;
    active = true; btn.classList.add('held'); btn.setAttribute('aria-pressed', 'true');
    // NO pointer capture: a captured pointer never fires pointerleave, so
    // sliding off the button would keep driving (measured in headless Edge
    // 2026-09-30). Touch pointers are captured IMPLICITLY, so release that too.
    try {
      if (ev && ev.pointerId != null && btn.hasPointerCapture(ev.pointerId))
        btn.releasePointerCapture(ev.pointerId);
    } catch (e) {}
    send();
    timer = setInterval(send, Math.max(33, Math.round(1000 / rate)));
  };
  const stop = () => {
    if (!active) return;
    active = false; btn.classList.remove('held'); btn.setAttribute('aria-pressed', 'false');
    if (timer) { clearInterval(timer); timer = null; }
    if (p.stop_action) panelAction(p, p.stop_action, {}, {quiet: true, noConfirm: true});
  };
  btn.addEventListener('pointerdown', start);
  btn.addEventListener('pointerup', stop);
  btn.addEventListener('pointercancel', stop);
  btn.addEventListener('pointerleave', stop);
  btn.addEventListener('keydown', (ev) => { if ((ev.key === ' ' || ev.key === 'Enter') && !ev.repeat) start(ev); });
  btn.addEventListener('keyup', (ev) => { if (ev.key === ' ' || ev.key === 'Enter') stop(); });
  btn.addEventListener('contextmenu', (ev) => ev.preventDefault());
  window.addEventListener('blur', stop);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  HOLD_STOPPERS.push(stop);
}
function pwBox(w, span) {
  const box = document.createElement('div'); box.className = 'pw' + (span ? ' span' : '');
  if (w.label) { const k = document.createElement('div'); k.className = 'k'; k.textContent = w.label; box.appendChild(k); }
  return box;
}
function fmtVal(v, unit) {
  if (v === undefined || v === null || v === '') return '—';
  if (typeof v === 'object') v = JSON.stringify(v);
  return String(v) + (unit ? ' ' + unit : '');
}
function renderWidget(p, w) {
  const ups = panelUpdaters[p.id];
  const t = w.type;
  if (t === 'estop') {
    const b = document.createElement('button'); b.type = 'button';
    b.textContent = (w.label || 'STOP') + ' — ' + p.title;
    b.setAttribute('aria-label', 'Emergency stop: ' + p.title);
    b.addEventListener('click', () => { stopAllHolds(); panelAction(p, w.action, {}, {noConfirm: true}); });
    estopDock.appendChild(b); estopDock.hidden = false;
    return null;
  }
  if (t === 'stat' || t === 'text') {
    const box = pwBox(w, t === 'text');
    const v = document.createElement(t === 'text' ? 'pre' : 'div'); v.className = 'v';
    box.appendChild(v);
    ups.push((s) => { v.textContent = fmtVal(s[w.key], w.unit); });
    return box;
  }
  if (t === 'badge') {
    const box = pwBox(w);
    const v = document.createElement('span'); v.className = 'badge'; box.appendChild(v);
    ups.push((s) => { const val = s[w.key]; v.textContent = fmtVal(val);
      v.className = 'badge ' + (((w.map || {})[String(val)]) || 'info'); });
    return box;
  }
  if (t === 'gauge') {
    const box = pwBox(w);
    const v = document.createElement('div'); v.className = 'v';
    const bar = document.createElement('div'); bar.className = 'bar';
    const fill = document.createElement('i'); bar.appendChild(fill);
    bar.setAttribute('role', 'meter'); bar.setAttribute('aria-valuemin', w.min); bar.setAttribute('aria-valuemax', w.max);
    box.appendChild(v); box.appendChild(bar);
    ups.push((s) => { const val = Number(s[w.key]);
      const ok = !isNaN(val) && s[w.key] !== null && s[w.key] !== undefined;
      v.textContent = ok ? fmtVal(val, w.unit) : '—';
      const pct = ok ? Math.max(0, Math.min(100, 100 * (val - w.min) / (w.max - w.min))) : 0;
      fill.style.width = pct + '%'; if (ok) bar.setAttribute('aria-valuenow', val); });
    return box;
  }
  if (t === 'events') {
    const box = pwBox(w, true);
    const ul = document.createElement('ul'); box.appendChild(ul);
    ups.push((s) => {
      const list = Array.isArray(s[w.key]) ? s[w.key].slice(-w.max) : [];
      ul.innerHTML = '';
      for (const e of list.reverse()) { const li = document.createElement('li');
        li.textContent = (e && typeof e === 'object') ? ((e.ts ? e.ts + '  ' : '') + (e.text || JSON.stringify(e))) : String(e);
        ul.appendChild(li); }
      if (!list.length) { const li = document.createElement('li'); li.className = 'muted'; li.textContent = '(none)'; ul.appendChild(li); }
    });
    return box;
  }
  if (t === 'image') {
    // Starts as the neutral placeholder: nothing is requested until a frame
    // is known to exist (see PANEL IMAGES below).
    const box = pwBox(w, true);
    const empty = document.createElement('div'); empty.className = 'pimg-empty';
    empty.textContent = 'No picture yet';
    const img = document.createElement('img'); img.alt = w.label || w.stream; img.hidden = true;
    const m = {p: p, img: img, empty: empty, mode: w.mode, key: w.key || '',
               refresh: Math.max(100, +w.refresh_ms || 1000),
               src: '/api/panel/' + encodeURIComponent(p.id) + '/stream/' + encodeURIComponent(w.stream),
               active: false, has: false, armedUntil: 0, sig: null, timer: null, url: null, busy: false};
    img.addEventListener('load', () => { if (m.active && m.img.getAttribute('src')) panelImageShown(m); });
    img.addEventListener('error', () => { if (m.img.getAttribute('src')) panelImageEmpty(m); });
    box.appendChild(empty); box.appendChild(img);
    panelMedia[p.id].push(m);
    if (m.key) ups.push((s) => panelImageState(m, s[m.key]));
    return box;
  }
  if (t === 'buttons') {
    const box = pwBox(w, true);
    const row = document.createElement('div'); row.className = 'row';
    for (const b of w.buttons) {
      const btn = document.createElement('button'); btn.type = 'button'; btn.textContent = b.label;
      const meta = (p.actions || {})[b.action] || {};
      if (meta.danger) btn.classList.add('danger');
      btn.addEventListener('click', async () => { btn.disabled = true;
        try { await panelAction(p, b.action, b.args || {}); } finally { btn.disabled = false; } });
      row.appendChild(btn);
    }
    box.appendChild(row);
    return box;
  }
  if (t === 'input') {
    const box = pwBox(w, true);
    const row = document.createElement('form'); row.className = 'row'; row.style.margin = '0';
    const inp = document.createElement('input'); inp.type = 'text'; inp.placeholder = w.placeholder || '';
    inp.setAttribute('aria-label', w.label || w.action);
    const btn = document.createElement('button'); btn.type = 'submit'; btn.textContent = w.button || 'Send';
    row.appendChild(inp); row.appendChild(btn);
    row.addEventListener('submit', async (ev) => { ev.preventDefault();
      const args = {}; args[w.arg] = inp.value;
      const res = await panelAction(p, w.action, args);
      if (res && res.ok) inp.value = ''; });
    box.appendChild(row);
    return box;
  }
  if (t === 'toggle') {
    const box = pwBox(w);
    const lab = document.createElement('label'); lab.className = 'toggle';
    const cb = document.createElement('input'); cb.type = 'checkbox';
    cb.setAttribute('aria-label', w.label || w.key);
    lab.appendChild(cb); lab.appendChild(document.createTextNode(' ' + (w.label || w.key)));
    box.appendChild(lab);
    let busy = false;
    cb.addEventListener('change', async () => { busy = true;
      const args = {}; args[w.arg] = cb.checked;
      try { await panelAction(p, w.action, args); } finally { busy = false; } });
    ups.push((s) => { if (!busy && w.key in s) cb.checked = !!s[w.key]; });
    return box;
  }
  if (t === 'slider') {
    const box = pwBox(w);
    const v = document.createElement('div'); v.className = 'v';
    const r = document.createElement('input'); r.type = 'range';
    r.min = w.min; r.max = w.max; r.step = w.step; r.setAttribute('aria-label', w.label || w.key);
    box.appendChild(v); box.appendChild(r);
    let dragging = false;
    r.addEventListener('input', () => { dragging = true; v.textContent = r.value; });
    r.addEventListener('change', async () => {
      const args = {}; args[w.arg] = Number(r.value);
      try { await panelAction(p, w.action, args); } finally { dragging = false; } });
    ups.push((s) => { if (!dragging && w.key in s) { r.value = s[w.key]; v.textContent = fmtVal(s[w.key]); } });
    return box;
  }
  if (t === 'hold') {
    const box = pwBox(w);
    const btn = document.createElement('button'); btn.type = 'button'; btn.className = 'hold';
    btn.textContent = w.label || w.action; btn.setAttribute('aria-pressed', 'false');
    bindHold(btn, p, w);
    box.appendChild(btn);
    return box;
  }
  return null;
}
function buildPanelView(p) {
  const nav = document.createElement('button'); nav.type = 'button';
  nav.id = 'navPanel_' + p.id; nav.textContent = p.title;
  navEl.appendChild(nav);
  const sec = document.createElement('section'); sec.className = 'view'; sec.hidden = true;
  sec.id = 'viewPanel_' + p.id;
  const stale = document.createElement('div'); stale.className = 'pstale'; stale.id = 'pstale_' + p.id;
  stale.setAttribute('aria-live', 'polite');
  const grid = document.createElement('div'); grid.className = 'pgrid';
  sec.appendChild(stale); sec.appendChild(grid);
  wrapEl.appendChild(sec);
  panelUpdaters[p.id] = []; panelMedia[p.id] = [];
  for (const w of (p.layout || [])) {
    const el = renderWidget(p, w);
    if (el) grid.appendChild(el);
  }
  const key = 'panel:' + p.id;
  VIEWS[key] = {nav: nav, view: sec, panel: p};
  nav.addEventListener('click', () => showView(key));
}
async function loadPanelState(p) {
  const out = document.getElementById('pstale_' + p.id);
  try {
    const r = await fetch(q('/api/panel/' + encodeURIComponent(p.id) + '/state'), {headers: hdr()});
    if (!r.ok) { if (out) out.textContent = 'state unavailable (' + r.status + ')'; return; }
    const d = await r.json();
    const s = d.state || {};
    for (const fn of panelUpdaters[p.id] || []) { try { fn(s); } catch (e) {} }
    if (out) out.textContent = d.stale ? ('stale' + (d.age_s != null ? ' (' + Math.round(d.age_s) + ' s old)' : '')
      + (d.error ? ': ' + d.error : '')) : '';
  } catch (e) { if (out) out.textContent = 'state unavailable'; }
}
function startPanelView(p) {
  if (!p) return;
  if (p.has_state) {
    loadPanelState(p);
    panelTimer = setInterval(() => { if (pollsWanted()) loadPanelState(p); }, Math.max(250, p.poll_ms || 1000));
  }
  for (const m of panelMedia[p.id] || []) startPanelImage(m);
}
// ── PANEL IMAGES ─────────────────────────────────────────────────────────────
// Before a panel's first frame exists its image box shows "No picture yet" and
// the page requests NOTHING for it. (It used to point the <img> at the still
// URL at once and re-request it every refresh_ms: a broken-image icon and a 404
// every 2 s until something took a picture.) A frame is fetched only when:
//   * the panel's STATE says one exists - the widget's optional `key`: a
//     truthy value means "there is a frame", a new value means "a new one";
//     falsy puts the placeholder back and stops asking;
//   * the owner DID something on this panel - after an action (a Snapshot
//     button, say) a keyless widget looks every refresh_ms for up to
//     PANEL_IMAGE_ARM_MS, until a frame arrives;
//   * the view OPENS - one look (keyless widgets only), so a picture taken
//     earlier is shown when the owner comes back to the tab.
// A 404 ("no frame") puts the placeholder back and stops the polling again
// (once an action's window, if any, has run out). Once a frame exists,
// snapshot mode refreshes every refresh_ms exactly as before, and stream mode
// shows the live MJPEG.
const PANEL_IMAGE_ARM_MS = 30000;
function panelImageShown(m) {
  m.has = true; m.img.hidden = false; m.empty.hidden = true;
}
function panelImageEmpty(m) {
  // Keeps an action's arm (m.armedUntil): a Snapshot button's frame may land
  // a few seconds after the click, so its window keeps looking until it ends.
  m.has = false;
  m.img.hidden = true; m.empty.hidden = false;
  m.img.removeAttribute('src');             // frees a stream slot
  if (m.url) { URL.revokeObjectURL(m.url); m.url = null; }
}
async function panelImageFetch(m) {
  if (!m.active || m.busy) return;
  m.busy = true;
  try {
    const r = await fetch(q(m.src + '?still=1&t=' + Date.now()), {headers: hdr()});
    if (!m.active) return;
    if (r.ok) {
      const url = URL.createObjectURL(await r.blob());
      if (m.url) URL.revokeObjectURL(m.url);
      m.url = url; m.has = true; m.armedUntil = 0;
      m.img.src = url;                      // shown on its 'load' event
    } else if (r.status === 404) {
      panelImageEmpty(m);                   // no frame (any more): stop asking,
    }                                       // unless an action's arm still runs
  } catch (e) {} finally { m.busy = false; }
}
function panelImageLoad(m) {
  if (!m.active) return;
  if (m.mode === 'snapshot') panelImageFetch(m);
  else m.img.src = q(m.src);                // MJPEG; 'load' shows it, 'error' empties it
}
function panelImageTick(m) {
  if (!m.active || !pollsWanted() || m.mode !== 'snapshot') return;
  if (m.has || Date.now() < m.armedUntil) panelImageFetch(m);
}
function panelImageState(m, v) {
  const on = !(v === undefined || v === null || v === false || v === '' || v === 0);
  const sig = on ? JSON.stringify(v) : null;
  const changed = sig !== m.sig;
  m.sig = sig;
  if (!m.active) return;
  if (!on) { if (m.has || m.img.getAttribute('src')) panelImageEmpty(m); return; }
  if (changed || !m.has) panelImageLoad(m);
}
function panelImagesArm(p) {
  // A user action on this panel may have produced a frame.
  for (const m of panelMedia[p.id] || []) {
    if (!m.active || (m.key && p.has_state)) continue;   // keyed: the state decides
    m.armedUntil = Date.now() + PANEL_IMAGE_ARM_MS;
    if (m.mode === 'snapshot') panelImageFetch(m);
    else if (!m.has) panelImageLoad(m);
  }
}
function startPanelImage(m) {
  m.active = true;
  if (m.mode === 'snapshot' && !m.timer)
    m.timer = setInterval(() => panelImageTick(m), m.refresh);
  // Opening the view is one look for a keyless widget; a keyed one waits
  // for the state (startPanelView reads it straight away).
  if (!m.key || !m.p.has_state) panelImageLoad(m);
}
function stopPanelMedia() {
  stopAllHolds();
  for (const id of Object.keys(panelMedia)) {
    for (const m of panelMedia[id]) {
      m.active = false; m.armedUntil = 0;
      if (m.timer) { clearInterval(m.timer); m.timer = null; }
      // A stream holds a server slot for as long as the <img> keeps its src;
      // a snapshot is a local blob and keeps showing until the next look.
      if (m.mode !== 'snapshot') { m.has = false; m.img.removeAttribute('src');
        m.img.hidden = true; m.empty.hidden = false; }
    }
  }
}
async function loadPanels() {
  try {
    const r = await fetch(q('/api/panels'), {headers: hdr()});
    if (!r.ok) return;
    const d = await r.json();
    PANELS = d.panels || [];
    for (const p of PANELS) buildPanelView(p);
    if (SAVED_VIEW.indexOf('panel:') === 0 && VIEWS[SAVED_VIEW] && currentView === 'live') showView(SAVED_VIEW);
  } catch (e) {}
}

// Read the remembered tab ONCE: showView() rewrites it, and a panel tab only
// exists after loadPanels() has built it.
const SAVED_VIEW = lsGet('jarvis.view', 'live');
showView(SAVED_VIEW);
loadPanels();
refreshStatus(); refreshLog();
setInterval(() => { if (pollsWanted()) refreshStatus(); }, 1500);
setInterval(() => { if (pollsWanted()) refreshLog(); }, 1000);
</script></body></html>"""
assert _DASHBOARD_PAGE.count(_TOKEN_SLOT) == 1


def _dashboard_html(token: str) -> str:
    """Return the full dashboard page. Inline CSS/JS only (no external fetches),
    dark theme with arc-reactor cyan accents. Polls /api/status + /api/log/tail
    once a second and POSTs typed commands to /api/say. The token (if any) is
    baked into the JS so the page's own API calls carry it."""
    # Serialize the token as JSON for the JS context it lands in. <script> is an
    # HTML raw-text element — character references are NOT decoded there — so
    # html.escape was the WRONG escaper (a token containing &"'<> reached the JS
    # as &amp;/&quot;/… and every API call 401'd; a backslash produced an
    # unterminated JS string). json.dumps escapes quotes, backslashes and
    # control characters correctly for a JS string literal, and the "</" → "<\/"
    # replacement keeps a token containing "</script>" from terminating the
    # inline script early ("\/" is a valid JS string escape equal to "/").
    tok = json.dumps(token or "").replace("</", "<\\/")
    return _DASHBOARD_PAGE.replace(_TOKEN_SLOT, tok, 1)


# ── server factory + lifecycle ───────────────────────────────────────────────

def _port_actively_served(host: str, port: int, timeout: float = 0.35) -> bool:
    """True when something is ALREADY accepting connections on host:port — the
    signal that binding here would co-bind a live listener (see the SO_REUSEADDR
    note in create_server). A quick loopback connect: it succeeds only against an
    ACTIVE listener, so a free port or a TIME_WAIT socket from our own restart
    both read False (safe to bind). Probes the loopback even for a wildcard bind
    (0.0.0.0), since a wildcard listener still answers on 127.0.0.1. Never raises."""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    try:
        import socket as _socket
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((probe_host, int(port))) == 0
    except Exception:
        return False


def create_server(*, bind: str, port: int, token: str = "",
                  inject_path: str = DEFAULT_INJECT_PATH,
                  log_dir: str = DEFAULT_LOG_DIR,
                  hud_state_path: str = DEFAULT_HUD_STATE_PATH,
                  user_settings_path: str | None = None,
                  camera_preview_path: str = DEFAULT_CAMERA_PREVIEW_PATH,
                  action_index_path: str = DEFAULT_ACTION_INDEX_PATH,
                  reply_reader=None,
                  tray_commands_path: str = DEFAULT_TRAY_COMMANDS_PATH,
                  runtime=None,
                  panels=None,
                  action_timeout_s: float = _ACTION_TIMEOUT_S) -> ThreadingHTTPServer:
    """Build (but do not serve) a ThreadingHTTPServer for the web interface.

    SECURITY GATE: refuses to construct a server on a NON-LOCAL bind when the
    token is empty (raises InsecureBindError) — the caller must supply a token to
    expose it on the LAN. A local (loopback) bind needs no token.

    ``reply_reader`` lets a test stub the log-tail reply wait (default:
    ``wait_for_reply``). All paths default to the live project files but are
    injectable so a test can point them at a temp dir and bind 127.0.0.1:0.

    ``user_settings_path`` is where POST /api/settings MERGES its writes; it
    defaults (when None) to the live data/user_settings.json resolved from
    settings_window — the same file the Settings GUI and core.config read — but a
    test points it at a throwaway file so a settings write can never clobber the
    real one (mirroring inject_path/log_dir/hud_state_path)."""
    bind = (bind or "127.0.0.1").strip()
    local = is_local_bind(bind)
    if not local and not (token or "").strip():
        raise InsecureBindError(
            f"refusing to start the web interface: bind={bind!r} is not loopback "
            f"but WEB_INTERFACE_TOKEN is empty. Set a token to expose it on the "
            f"LAN, or bind 127.0.0.1 for localhost-only (no token needed)."
        )
    # PRE-BIND PROBE (Windows SO_REUSEADDR footgun): ThreadingHTTPServer sets
    # allow_reuse_address=True, and on Windows that lets a *second* socket bind
    # a port another process is ALREADY actively LISTENing on. The two sockets
    # then split incoming connections non-deterministically — so if a stale
    # instance (or a leaked test process) is squatting the port, JARVIS binds
    # "successfully" but half the requests land on the dead socket and hang
    # (observed live 2026-07-07: a leftover `-m unittest` process held 8766 and
    # the dashboard was unreachable). We can't tell that apart from a healthy
    # bind after the fact, so REFUSE up front when the port is already being
    # served: a real port>0 that answers a loopback connect is in use. We only
    # probe a concrete port (port 0 = ephemeral, always free) and only treat an
    # ACTIVE listener as a conflict — a TIME_WAIT socket from our own recent
    # restart doesn't answer connect(), so a normal reboot still rebinds. The
    # skill's _start() turns the resulting OSError into an honest spoken
    # "port in use" instead of a silent half-broken server.
    if int(port) > 0 and _port_actively_served(bind, int(port)):
        raise OSError(
            f"web interface port {port} on {bind} is already being served by "
            f"another process (a stale JARVIS or a leaked test server) — refusing "
            f"to co-bind it. Free the port (stop the other process) and retry."
        )
    httpd = ThreadingHTTPServer((bind, int(port)), _Handler)
    # Pin per-server config onto the instance so the stateless handler reads it.
    # user_settings_path resolves to the live data/user_settings.json when the
    # caller didn't override it — done HERE (not as a def-time default) so the
    # JARVIS_SETTINGS_PATH redirect is honoured at server-construction time.
    httpd.config = {  # type: ignore[attr-defined]
        "token": (token or "").strip(),
        "bind": bind,          # for the anti-CSRF/rebinding Host+Origin allowlist
        "local_bind": local,
        "inject_path": inject_path,
        "log_dir": log_dir,
        "hud_state_path": hud_state_path,
        "user_settings_path": user_settings_path or _default_user_settings_path(),
        # Full-control-panel sources: the live camera preview frame and the
        # machine-generated action inventory. Injectable so a test points them at
        # a temp path (mirroring inject_path/log_dir/hud_state_path).
        "camera_preview_path": camera_preview_path or DEFAULT_CAMERA_PREVIEW_PATH,
        "action_index_path": action_index_path or DEFAULT_ACTION_INDEX_PATH,
        "reply_reader": reply_reader,
        # The tray control plane's inbox (POST /api/control, and restart /
        # shutdown via POST /api/action). Injectable like every path above.
        "tray_commands_path": tray_commands_path or DEFAULT_TRAY_COMMANDS_PATH,
        # What the server may read from the running JARVIS (camera roster,
        # camera gate, live ACTIONS). LiveRuntime answers only inside the booted
        # assistant; a test passes a fake, and NoRuntime() pins the defaults.
        "runtime": runtime if runtime is not None else LiveRuntime(),
        # Skill-declared panels: None = the process-wide core.web_panels
        # REGISTRY the skill loader fills; a test passes its own PanelRegistry.
        "panels": panels,
        "action_timeout_s": float(action_timeout_s),
    }
    return httpd


def serve_in_thread(httpd: ThreadingHTTPServer) -> threading.Thread:
    """Run ``httpd.serve_forever()`` on a daemon thread and return it. The caller
    stops it with ``httpd.shutdown()`` (which unblocks serve_forever) then
    ``httpd.server_close()``. Daemon so a hung request never blocks JARVIS exit."""
    t = threading.Thread(target=httpd.serve_forever, name="web-interface",
                         daemon=True)
    t.start()
    return t
