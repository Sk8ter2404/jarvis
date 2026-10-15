"""Holo mode - the shared state and the static page for the hand-tracked
hologram (hud/holo/, served by tools/web_interface.py at GET /holo).

WHAT THIS IS
  A browser page (a three.js arc reactor) that the owner drives with his
  hands. The page does NOT open the camera and runs no hand model: hands and
  gestures arrive over a WebSocket from the separate tracker service
  (tools/hand_tracker_service.py, ws://127.0.0.1:8793), because Windows lets
  only one process hold the webcam and JARVIS already does. Two modes, one
  page: ``desktop`` (solid dark HUD, no camera picture) and ``camera`` (the
  tracker's mirrored frames behind the scene; with ``depth`` the Kinect's depth
  plane too, for occlusion and push/pull). JARVIS only does three things
  for the page:
    * serves the page and its static files (``asset_for``),
    * keeps a tiny command queue the voice actions write to and the page polls
      every ~300 ms (``push_command`` / ``state_payload``),
    * notices whether a page is actually open (the poll is its heartbeat), so
      "explode it" with no page up can say so instead of lying.

PHASE 3 (Quest 3): ``/holo/xr`` is the same hologram as a WebXR page for the
headset's browser (hud/holo/xr.js). The headset's own hand tracking replaces the
tracker service, so that page opens no WebSocket and gets a narrower CSP
(``HOLO_XR_CSP``). WebXR needs HTTPS (or localhost); the page says so itself
and names the owner's LAN name from ``HOLO_XR_HINT_HOST`` (``xr_hint_host``).
This module does not change what the web interface binds to.

WHAT IT MUST NEVER DO
  Move the Windows cursor or send any OS input. The page only draws; hand
  tracking for it lives in the browser and never reaches the air-mouse code.

The owner's OWN models: STL / 3MF / OBJ files from ``data/holo_models`` plus any
folders in ``HOLO_MODEL_DIRS`` are listed (``list_models``) and served one by one
at ``/holo/model/<id>`` (``model_response``). The id is a hash of the file's real
path and is only ever looked up in a fresh scan of the configured folders, so a
URL can never name an arbitrary file.

PURE: stdlib only, no I/O beyond reading the page files and scanning the model
folders, no threads except one lock. The skill (skills/holo_mode.py) and the web handler share this module's
single ``STATE``; tests build their own ``HoloState``.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import threading
import time

# The built-in procedural model. The owner's own files (see list_models) come
# on top of it.
MODELS = ("arc reactor",)

# Own models: extension -> kind. Nothing else is ever listed or served.
MODEL_EXTS = {".stl": "stl", ".3mf": "3mf", ".obj": "obj"}
MODEL_CONTENT_TYPES = {"stl": "model/stl", "3mf": "model/3mf", "obj": "model/obj"}
MODEL_MAX_DEPTH = 2          # folders below a model folder that are searched
MODEL_MAX_FILES = 400        # most models ever listed
DEFAULT_MODEL_MAX_MB = 64
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILTIN_MODEL_DIR = os.path.join(_REPO_ROOT, "data", "holo_models")
_MODEL_ID_RE = re.compile(r"^[0-9a-f]{12}$")

MODES = ("desktop", "camera")
DEFAULT_MODE = "camera"

# A page polls every ~300 ms; this long without one and it is gone (tab
# closed, browser asleep).
ALIVE_S = 4.0

# Most commands kept for a page that polls late. Each carries a sequence
# number; the page asks for everything after the last one it applied.
MAX_COMMANDS = 16

COMMANDS = ("show", "explode", "assemble", "close", "mode", "depth")

_HOLO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hud", "holo")

# Published path -> (file in hud/holo, content type). The ONLY files served.
_ASSETS = {
    "/holo": ("holo.html", "text/html; charset=utf-8"),
    "/holo/holo.js": ("holo.js", "text/javascript; charset=utf-8"),
    "/holo/gestures.js": ("gestures.js", "text/javascript; charset=utf-8"),
    "/holo/depth.js": ("depth.js", "text/javascript; charset=utf-8"),
    "/holo/reactor.js": ("reactor.js", "text/javascript; charset=utf-8"),
    "/holo/models.js": ("models.js", "text/javascript; charset=utf-8"),
    "/holo/fakehands.js": ("fakehands.js", "text/javascript; charset=utf-8"),
    "/holo/selftest.js": ("selftest.js", "text/javascript; charset=utf-8"),
    # Phase 3: the same hologram in the Quest 3's browser (WebXR). Its own page,
    # its own CSP (no tracker WebSocket: the headset's hand tracking replaces it).
    "/holo/xr": ("xr.html", "text/html; charset=utf-8"),
    "/holo/xr.js": ("xr.js", "text/javascript; charset=utf-8"),
    "/holo/xrgestures.js": ("xrgestures.js", "text/javascript; charset=utf-8"),
    "/holo/xrfake.js": ("xrfake.js", "text/javascript; charset=utf-8"),
}

_TOKEN_SLOT = "__JARVIS_TOKEN__"
_XR_HOST_SLOT = "__JARVIS_XR_HOST__"
# The pages that carry the token (and, for the XR page, the LAN-name hint).
_PAGES = ("/holo", "/holo/xr")

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

# The XR page opens NO WebSocket (the headset's own hands replace the tracker
# service) and talks to nothing but its own origin and the pinned CDN. ``blob:``
# is only for the glTF / 3MF loaders' embedded textures (controller models,
# coloured 3MFs); it is not a host.
HOLO_XR_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "connect-src 'self' https://cdn.jsdelivr.net blob:; "
    "worker-src 'self' blob:; "
    "frame-ancestors 'none'; base-uri 'none'; object-src 'none'; "
    "form-action 'self'")


def csp_for(path: str) -> str:
    """The Content-Security-Policy for a published /holo path."""
    return HOLO_XR_CSP if path == "/holo/xr" else HOLO_CSP


class HoloState:
    """Thread-safe holo mode state. ``on`` is what the owner asked for;
    ``page_alive`` is whether a page has polled lately; ``shown`` / ``exploded``
    are what the page last reported about itself."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self.on = False
        self.mode = DEFAULT_MODE    # desktop | camera
        self.depth = False          # camera mode with Kinect depth (occlusion, push/pull)
        self.opened_at = None       # when set_on(True) last ran (the opening grace)
        self.model = ""             # model to show when a page loads
        self.seq = 0
        self.commands: list = []    # [{"seq", "cmd", "arg"}]
        self._last_poll = None
        self.shown = ""             # what the page says is on screen
        self.exploded = False
        self.parts = 0              # parts of what is on screen (0 = unknown)

    # -- writes (the voice actions) ------------------------------------------
    def set_on(self, on: bool, model: str = "", mode: str = "",
               depth=None) -> None:
        with self._lock:
            self.on = bool(on)
            self.model = model if on else ""
            if on:
                self.opened_at = self._clock()
            if on and mode in MODES:
                self.mode = mode
            if on and depth is not None:
                self.depth = bool(depth)
            if not on:
                self.depth = False
                self.shown = ""
                self.exploded = False
                self.parts = 0

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
                self.depth = False
            elif cmd == "mode" and arg in MODES:
                self.mode = arg
            elif cmd == "depth":
                self.depth = arg in ("1", "on", "true")
            return self.seq

    # -- the page's heartbeat -------------------------------------------------
    def heartbeat(self, shown=None, exploded=None, parts=None) -> None:
        with self._lock:
            self._last_poll = self._clock()
            if shown is not None:
                self.shown = str(shown)[:60]
            if exploded is not None:
                self.exploded = bool(exploded)
            if parts is not None:
                self.parts = max(0, min(int(parts), 9999))

    @property
    def page_alive(self) -> bool:
        with self._lock:
            return (self._last_poll is not None
                    and self._clock() - self._last_poll < ALIVE_S)

    def kinect_session(self, grace_s: float = 10.0) -> bool:
        """True while a hologram that asked for the Kinect + depth is up: the
        page is polling, or it was opened under ``grace_s`` ago and has not
        polled yet. core/handtrack_depth.py uses this to pick the Kinect as the
        tracker's camera for exactly as long as that hologram lives."""
        with self._lock:
            if not (self.on and self.depth):
                return False
            if (self._last_poll is not None
                    and self._clock() - self._last_poll < ALIVE_S):
                return True
            return (self.opened_at is not None
                    and self._clock() - self.opened_at < grace_s)

    # -- the poll payload -----------------------------------------------------
    def payload(self, seen: int = 0) -> dict:
        with self._lock:
            return {
                "on": self.on,
                "mode": self.mode,
                "depth": self.depth,
                "model": self.model,
                "seq": self.seq,
                "commands": [c for c in self.commands if c["seq"] > seen],
                "shown": self.shown,
                "exploded": self.exploded,
                "parts": self.parts,
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
    parts = one("parts")
    st.heartbeat(shown=one("shown"),
                 exploded=None if exploded is None else exploded == "1",
                 parts=None if parts is None else _int(parts, 0))
    return st.payload(seen=_int(one("seen"), 0))


def asset_for(path: str, token: str = ""):
    """(body bytes, content type) for a published holo path, or None when the
    path is not one of the known files or the file is missing. The page
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
    if path in _PAGES:
        tok = json.dumps(token or "").replace("</", "<\\/")
        text = data.decode("utf-8").replace(_TOKEN_SLOT, tok, 1)
        if path == "/holo/xr":
            host = json.dumps(xr_hint_host()).replace("</", "<\\/")
            text = text.replace(_XR_HOST_SLOT, host, 1)
        data = text.encode("utf-8")
    return data, ctype


# A LAN name for the secure-address hint on the XR page (HOLO_XR_HINT_HOST). The
# page shows it in a sentence (text only, never a link); still, only a plain host
# name (letters, digits, dots, hyphens, an optional :port) is ever passed on.
_HOST_RE = re.compile(
    r"^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*(?::\d{1,5})?$")


def xr_hint_host(value=None) -> str:
    """HOLO_XR_HINT_HOST, or "" when it is unset or not a plain host name."""
    v = _cfg("HOLO_XR_HINT_HOST", "") if value is None else value
    v = str(v or "").strip()
    return v if _HOST_RE.match(v) else ""


# ---------------------------------------------------------------------------
# the owner's own models
# ---------------------------------------------------------------------------
def _cfg(name: str, default):
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def model_max_bytes(max_mb=None) -> int:
    """The size cap in bytes (HOLO_MODEL_MAX_MB, default 64)."""
    mb = _cfg("HOLO_MODEL_MAX_MB", DEFAULT_MODEL_MAX_MB) if max_mb is None else max_mb
    try:
        mb = float(mb)
    except (TypeError, ValueError):
        mb = DEFAULT_MODEL_MAX_MB
    if mb <= 0:
        mb = DEFAULT_MODEL_MAX_MB
    return int(mb * 1024 * 1024)


def model_dirs(extra=None, builtin=None, create: bool = True) -> list:
    """The folders models are searched in, as real (resolved) paths: the built-in
    ``data/holo_models`` (created lazily) first, then HOLO_MODEL_DIRS. Folders
    that do not exist are skipped; duplicates collapse."""
    if extra is None:
        extra = _cfg("HOLO_MODEL_DIRS", [])
    if isinstance(extra, (str, os.PathLike)):
        extra = [extra]
    bi = BUILTIN_MODEL_DIR if builtin is None else builtin
    if create and bi:
        try:
            os.makedirs(bi, exist_ok=True)
        except OSError:
            pass
    out, seen = [], set()
    for d in [bi] + list(extra or []):
        if not isinstance(d, (str, os.PathLike)) or not str(d).strip():
            continue
        try:
            real = os.path.realpath(os.path.expanduser(str(d)))
        except (OSError, ValueError):
            continue
        key = os.path.normcase(real)
        if key in seen or not os.path.isdir(real):
            continue
        seen.add(key)
        out.append(real)
    return out


def _inside(real_path: str, real_root: str) -> bool:
    try:
        rp, rr = os.path.normcase(real_path), os.path.normcase(real_root)
        return os.path.commonpath([rp, rr]) == rr and rp != rr
    except ValueError:                      # different drives
        return False


def model_id_for(real_path: str) -> str:
    """Stable id: the same file always gets the same id."""
    key = os.path.normcase(os.path.realpath(real_path))
    return hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:12]


def display_name(filename: str) -> str:
    """'Tiger_Pendant-FAST.3mf' -> 'Tiger Pendant-FAST'. Splits on BOTH path
    separators: os.path.basename on Linux keeps a Windows path whole (GitHub
    CI on v2.0.199 got 'C:\\x\\Tiger Pendant FAST')."""
    stem = os.path.splitext(re.split(r"[\\/]", str(filename))[-1])[0]
    stem = " ".join(stem.replace("_", " ").split())
    return stem[:60] or "model"


def _scan(dirs, max_bytes: int) -> list:
    """Every model file under ``dirs`` (at most MODEL_MAX_DEPTH folders deep),
    as dicts with the real path and a ``too_big`` flag. Order is stable."""
    found = []

    def walk(root, cur, depth):
        try:
            with os.scandir(cur) as it:
                entries = sorted(it, key=lambda e: e.name.lower())
        except OSError:
            return
        for e in entries:
            if len(found) >= MODEL_MAX_FILES:
                return
            try:
                if e.is_dir(follow_symlinks=False):
                    if depth < MODEL_MAX_DEPTH:
                        walk(root, e.path, depth + 1)
                    continue
                ext = os.path.splitext(e.name)[1].lower()
                kind = MODEL_EXTS.get(ext)
                if kind is None or not e.is_file():
                    continue
                real = os.path.realpath(e.path)
                if not _inside(real, root):         # a link pointing outside
                    continue
                size = os.path.getsize(real)
            except OSError:
                continue
            if size <= 0:
                continue
            found.append({"id": model_id_for(real), "name": display_name(e.name),
                          "type": kind, "size": size, "path": real,
                          "too_big": size > max_bytes})

    for root in dirs:
        walk(root, root, 0)
    seen, out = set(), []
    for m in found:                                  # one entry per real file
        if m["id"] not in seen:
            seen.add(m["id"])
            out.append(m)
    return out


def list_models(extra_dirs=None, builtin_dir=None, max_mb=None) -> list:
    """The models the page and the voice can show: [{id, name, type, size}],
    sorted by name (3MF before STL before OBJ for the same name). Files over
    the size cap are left out."""
    cap = model_max_bytes(max_mb)
    order = {"3mf": 0, "stl": 1, "obj": 2}
    rows = [m for m in _scan(model_dirs(extra_dirs, builtin_dir), cap)
            if not m["too_big"]]
    rows.sort(key=lambda m: (m["name"].lower(), order.get(m["type"], 9), m["id"]))
    return [{"id": m["id"], "name": m["name"], "type": m["type"], "size": m["size"]}
            for m in rows]


def models_payload(extra_dirs=None, builtin_dir=None, max_mb=None) -> dict:
    """The body of GET /api/holo/models."""
    return {"models": list_models(extra_dirs, builtin_dir, max_mb),
            "max_mb": round(model_max_bytes(max_mb) / (1024 * 1024), 1),
            "folder": "data/holo_models"}


def model_response(model_id: str, extra_dirs=None, builtin_dir=None, max_mb=None):
    """(status, body, content type) for GET /holo/model/<id>. ``body`` is the
    file's bytes on 200, else a small dict for the JSON error. The id is only
    ever looked up in a fresh scan of the configured folders; nothing from the
    URL becomes a path. 404 for an unknown id, 413 over the size cap."""
    if not isinstance(model_id, str) or not _MODEL_ID_RE.match(model_id):
        return 404, {"error": "not found"}, "application/json"
    cap = model_max_bytes(max_mb)
    dirs = model_dirs(extra_dirs, builtin_dir, create=False)
    hit = next((m for m in _scan(dirs, cap) if m["id"] == model_id), None)
    if hit is None:
        return 404, {"error": "not found"}, "application/json"
    real = os.path.realpath(hit["path"])
    if (not any(_inside(real, d) for d in dirs)
            or os.path.splitext(real)[1].lower() not in MODEL_EXTS):
        return 404, {"error": "not found"}, "application/json"
    if hit["too_big"]:
        return 413, {"error": "model is over the size cap",
                     "max_mb": round(cap / (1024 * 1024), 1)}, "application/json"
    try:
        with open(real, "rb") as fh:
            if os.fstat(fh.fileno()).st_size > cap:
                return 413, {"error": "model is over the size cap"}, "application/json"
            data = fh.read(cap + 1)
    except OSError:
        return 404, {"error": "not found"}, "application/json"
    if len(data) > cap:
        return 413, {"error": "model is over the size cap"}, "application/json"
    return 200, data, MODEL_CONTENT_TYPES[hit["type"]]


# -- fuzzy name matching (the voice: "show me the tiger pendant") ----------------
_FILLER = frozenset(("the", "a", "an", "my", "of", "model", "file"))


def _words(text: str) -> list:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).split()


def _spoken_words(text: str) -> list:
    """Like _words, minus filler a speaker adds ("the tiger model")."""
    w = [x for x in _words(text) if x not in _FILLER]
    return w or _words(text)


def _word_sim(q: str, n: str) -> float:
    if q == n:
        return 1.0
    if len(q) >= 3 and n.startswith(q):
        return 0.95
    r = difflib.SequenceMatcher(None, q, n).ratio()
    return r if r >= 0.75 else 0.0


def match_score(query: str, name: str) -> float:
    """0..1 - how well a spoken name fits a model name."""
    q, n = _spoken_words(query), _words(name)
    if not q or not n:
        return 0.0
    if q == n:
        return 1.0
    sims = [max(_word_sim(w, x) for x in n) for w in q]
    best = 0.0
    if all(s > 0 for s in sims):
        covered = {x for w in q for x in n if _word_sim(w, x) > 0}
        best = 0.6 * (sum(sims) / len(sims)) + 0.4 * (len(covered) / len(n))
    whole = difflib.SequenceMatcher(None, " ".join(q), " ".join(n)).ratio()
    if whole >= 0.8:
        best = max(best, whole)
    return best


def match_model(query: str, entries, threshold: float = 0.55):
    """The best entry (a dict with a ``name``) for a spoken name, or None.
    Ties go to the 3MF (it has parts), then the shorter name."""
    order = {"3mf": 0, "stl": 1, "obj": 2}
    best, best_key = None, None
    for e in entries or []:
        sc = match_score(query, e.get("name", ""))
        if sc < threshold:
            continue
        key = (-round(sc, 3), order.get(e.get("type"), 9), len(e.get("name", "")))
        if best_key is None or key < best_key:
            best, best_key = e, key
    return best
