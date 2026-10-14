"""Holo mode - the shared state and the static page for the hand-tracked
hologram (hud/holo/, served by tools/web_interface.py at GET /holo).

WHAT THIS IS
  A browser page (a three.js arc reactor) that the owner drives with his
  hands. The page does NOT open the camera and runs no hand model: hands and
  gestures arrive over a WebSocket from the separate tracker service
  (tools/hand_tracker_service.py, ws://127.0.0.1:8793), because Windows lets
  only one process hold the webcam and JARVIS already does. Two modes, one
  page: ``desktop`` (solid dark HUD, no camera picture) and ``camera`` (the
  tracker's mirrored frames behind the scene). JARVIS only does three things
  for the page:
    * serves the page and its four static files (``asset_for``),
    * keeps a tiny command queue the voice actions write to and the page polls
      every ~300 ms (``push_command`` / ``state_payload``),
    * notices whether a page is actually open (the poll is its heartbeat), so
      "explode it" with no page up can say so instead of lying.

WHAT IT MUST NEVER DO
  Move the Windows cursor or send any OS input. The page only draws; hand
  tracking for it lives in the browser and never reaches the air-mouse code.

PURE: stdlib only, no I/O beyond reading the page files, no threads except one
lock. The skill (skills/holo_mode.py) and the web handler share this module's
single ``STATE``; tests build their own ``HoloState``.
"""
from __future__ import annotations

import json
import os
import threading
import time

# Models the page can build. Only the procedural arc reactor exists so far;
# "my models" (the owner's own CAD files) is phase 2.
MODELS = ("arc reactor",)

MODES = ("desktop", "camera")
DEFAULT_MODE = "camera"

# A page polls every ~300 ms; this long without one and it is gone (tab
# closed, browser asleep).
ALIVE_S = 4.0

# Most commands kept for a page that polls late. Each carries a sequence
# number; the page asks for everything after the last one it applied.
MAX_COMMANDS = 16

COMMANDS = ("show", "explode", "assemble", "close", "mode")

_HOLO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hud", "holo")

# Published path -> (file in hud/holo, content type). The ONLY files served.
_ASSETS = {
    "/holo": ("holo.html", "text/html; charset=utf-8"),
    "/holo/holo.js": ("holo.js", "text/javascript; charset=utf-8"),
    "/holo/gestures.js": ("gestures.js", "text/javascript; charset=utf-8"),
    "/holo/reactor.js": ("reactor.js", "text/javascript; charset=utf-8"),
    "/holo/fakehands.js": ("fakehands.js", "text/javascript; charset=utf-8"),
    "/holo/selftest.js": ("selftest.js", "text/javascript; charset=utf-8"),
}

_TOKEN_SLOT = "__JARVIS_TOKEN__"

# The dashboard's CSP is 'self'-only; this page must also load three.js from
# jsdelivr and talk to the tracker service on loopback port 8793. Nothing else
# is opened up (no 'unsafe-eval', no wasm, no other host).
HOLO_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "media-src 'self' blob: mediastream:; "
    "connect-src 'self' https://cdn.jsdelivr.net "
    "ws://127.0.0.1:8793 ws://localhost:8793; "
    "worker-src 'self' blob:; "
    "frame-ancestors 'none'; base-uri 'none'; object-src 'none'; "
    "form-action 'self'")


class HoloState:
    """Thread-safe holo mode state. ``on`` is what the owner asked for;
    ``page_alive`` is whether a page has polled lately; ``shown`` / ``exploded``
    are what the page last reported about itself."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self.on = False
        self.mode = DEFAULT_MODE    # desktop | camera
        self.model = ""             # model to show when a page loads
        self.seq = 0
        self.commands: list = []    # [{"seq", "cmd", "arg"}]
        self._last_poll = None
        self.shown = ""             # what the page says is on screen
        self.exploded = False

    # -- writes (the voice actions) ------------------------------------------
    def set_on(self, on: bool, model: str = "", mode: str = "") -> None:
        with self._lock:
            self.on = bool(on)
            self.model = model if on else ""
            if on and mode in MODES:
                self.mode = mode
            if not on:
                self.shown = ""
                self.exploded = False

    def push_command(self, cmd: str, arg: str = "") -> int:
        """Queue one command for the page; returns its sequence number."""
        if cmd not in COMMANDS:
            raise ValueError(f"unknown holo command {cmd!r}")
        with self._lock:
            self.seq += 1
            self.commands.append({"seq": self.seq, "cmd": cmd, "arg": arg})
            del self.commands[:-MAX_COMMANDS]
            if cmd == "show":
                self.model = arg
            elif cmd == "close":
                self.on = False
                self.model = ""
            elif cmd == "mode" and arg in MODES:
                self.mode = arg
            return self.seq

    # -- the page's heartbeat -------------------------------------------------
    def heartbeat(self, shown=None, exploded=None) -> None:
        with self._lock:
            self._last_poll = self._clock()
            if shown is not None:
                self.shown = str(shown)[:40]
            if exploded is not None:
                self.exploded = bool(exploded)

    @property
    def page_alive(self) -> bool:
        with self._lock:
            return (self._last_poll is not None
                    and self._clock() - self._last_poll < ALIVE_S)

    # -- the poll payload -----------------------------------------------------
    def payload(self, seen: int = 0) -> dict:
        with self._lock:
            return {
                "on": self.on,
                "mode": self.mode,
                "model": self.model,
                "seq": self.seq,
                "commands": [c for c in self.commands if c["seq"] > seen],
                "shown": self.shown,
                "exploded": self.exploded,
            }


STATE = HoloState()


def _int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def state_payload(query: dict, state: HoloState = None) -> dict:
    """The body of GET /api/holo/state. ``query`` is parse_qs output:
    ``seen`` (last command seq the page applied), plus the page's own report
    ``shown`` (model name or empty) and ``exploded`` (0/1). Calling it counts
    as the page's heartbeat."""
    st = state or STATE
    def one(k):
        v = query.get(k)
        return v[0] if v else None
    exploded = one("exploded")
    st.heartbeat(shown=one("shown"),
                 exploded=None if exploded is None else exploded == "1")
    return st.payload(seen=_int(one("seen"), 0))


def asset_for(path: str, token: str = ""):
    """(body bytes, content type) for a published holo path, or None when the
    path is not one of the five known files or the file is missing. The page
    gets the web-interface token baked in, exactly like the dashboard."""
    spec = _ASSETS.get(path)
    if spec is None:
        return None
    fname, ctype = spec
    try:
        with open(os.path.join(_HOLO_DIR, fname), "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if path == "/holo":
        tok = json.dumps(token or "").replace("</", "<\\/")
        data = data.decode("utf-8").replace(_TOKEN_SLOT, tok, 1).encode("utf-8")
    return data, ctype
