"""Hand tracking control - the voice and boot surface over the MediaPipe tracker.

The tracker itself is a separate process (tools/hand_tracker_service.py, in its
own venv); core/hand_tracker_launcher.py starts, watches and stops it, and
core/handtrack_frames.py hands it the camera frames JARVIS already captures.
This skill adds:

Actions (each returns ONE finished sentence, spoken verbatim):
  hand_tracking_on      - start the tracker for this session (the HAND_TRACKING_ENABLED
                          setting starts it at every boot).
  hand_tracking_off     - stop it: nothing runs, no frames are copied.
  hand_tracking_status  - is it running, how many hands it sees, which camera.

The palm-hold trigger: while the tracker runs, a background listener reads its
websocket (ws://127.0.0.1:8793). A `palm_hold` gesture (open palm, fingers
spread, facing the camera, held ~1.5 s) opens the DESKTOP holo window - only
when HAND_MENU_ENABLED is on, at most once per 10 s, and never in game mode,
over a fullscreen app or in a call. It never moves the cursor or sends OS input.

The window is opened through the holo skill's `holo_mode_on` action with the
argument "desktop". That action belongs to the holo page work; until it exists
the trigger logs one line and does nothing - `set_open_hook()` is the clearly
named seam for another opener (tests use it too).
"""
from __future__ import annotations

import json
import sys
import threading
import time
from typing import Callable, Optional

SPEAK_VERBATIM_ACTIONS = (
    "hand_tracking_on", "hand_tracking_off", "hand_tracking_status",
)

OPEN_COOLDOWN_S = 10.0
HOLO_ACTION = "holo_mode_on"
HOLO_ARG = "desktop"
LOG = "  [hand-track]"

_state = {"last_open": -1e9, "opens": 0, "last_event": "", "last_event_at": 0.0,
          "last_block": "", "warned_missing": False, "listening": False}
_open_hook: list = [None]            # optional replacement for the holo action
_listener: list = [None]


# ── config ────────────────────────────────────────────────────────────────
def _cfg(name: str, default=None):
    try:
        from core import config
        return getattr(config, name, default)
    except Exception:
        return default


def _set_runtime(enabled: bool) -> None:
    try:
        from core import config
        config.HAND_TRACKING_ENABLED = bool(enabled)
    except Exception:
        pass


def _launcher():
    try:
        from core import hand_tracker_launcher
        return hand_tracker_launcher.LAUNCHER
    except Exception:
        return None


def _bobert():
    return sys.modules.get("bobert_companion") or sys.modules.get("__main__")


def _registry() -> dict:
    reg = getattr(_bobert(), "ACTIONS", None)
    return reg if isinstance(reg, dict) else {}


def set_open_hook(fn: Optional[Callable[[], object]]) -> None:
    """Replace the 'open the desktop holo window' call (None = use the
    holo_mode_on action)."""
    _open_hook[0] = fn


# ── what must stop the palm-hold trigger ──────────────────────────────────
def _game_mode_active() -> bool:
    try:
        sk = sys.modules.get("skill_game_mode")
        st = getattr(sk, "_st", None) if sk is not None else None
        return bool(getattr(st, "active", False))
    except Exception:
        return False


def _fullscreen_app() -> bool:
    try:
        sk = sys.modules.get("skill_game_mode")
        fn = getattr(sk, "_fullscreen_foreground", None) if sk is not None else None
        if not callable(fn):
            return False
        pid = fn()[0]
        return bool(pid)
    except Exception:
        return False


def _in_call() -> bool:
    for name in ("skill_banter", "skill_anticipation_engine",
                 "skill_weekly_digest_briefing"):
        try:
            fn = getattr(sys.modules.get(name), "_is_in_call", None)
            if callable(fn):
                return bool(fn())
        except Exception:
            continue
    return False


def blocked_reason() -> str:
    """Why a palm hold must not open the window right now ('' = it may)."""
    if _game_mode_active():
        return "game mode is on"
    if _fullscreen_app():
        return "a fullscreen app is in front"
    if _in_call():
        return "you are in a call"
    return ""


# ── the palm-hold trigger ─────────────────────────────────────────────────
def _open_desktop_window() -> str:
    hook = _open_hook[0]
    if callable(hook):
        hook()
        return "hook"
    fn = _registry().get(HOLO_ACTION)
    if callable(fn):
        fn(HOLO_ARG)
        return HOLO_ACTION
    return ""


def handle_gesture(ev: dict, now: Optional[float] = None) -> str:
    """React to one gesture event from the tracker. Returns what happened
    ('opened', 'ignored: <why>' ...) for logs and tests. Never raises."""
    try:
        name = str(ev.get("name", ""))
        now = time.monotonic() if now is None else now
        _state["last_event"] = name
        _state["last_event_at"] = time.time()
        if name != "palm_hold":
            return "ignored: not a palm hold"
        if not bool(_cfg("HAND_MENU_ENABLED", False)):
            return "ignored: hand menu is off"
        if now - _state["last_open"] < OPEN_COOLDOWN_S:
            return "ignored: opened a moment ago"
        why = blocked_reason()
        if why:
            _state["last_block"] = why
            print(f"{LOG} palm hold ignored: {why}")
            return f"ignored: {why}"
        try:
            how = _open_desktop_window()
        except Exception as e:
            print(f"{LOG} opening the desktop window failed: {type(e).__name__}: {e}")
            return "ignored: the window would not open"
        if not how:
            if not _state["warned_missing"]:
                _state["warned_missing"] = True
                print(f"{LOG} palm hold seen, but no '{HOLO_ACTION}' action is "
                      "registered yet and no open hook is set - nothing to open")
            return "ignored: no holo opener"
        _state["last_open"] = now
        _state["opens"] += 1
        print(f"{LOG} palm hold -> desktop holo window ({how})")
        return "opened"
    except Exception:
        return "ignored: error"


class GestureListener:
    """Reads the tracker's websocket and feeds gesture events to
    handle_gesture(). Reconnects with a short backoff while it is wanted."""

    def __init__(self, url: str, connect: Optional[Callable] = None,
                 on_event: Callable[[dict], object] = handle_gesture):
        self.url = url
        self._connect = connect
        self._on_event = on_event
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connected = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="hand-gesture-listener",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _open(self):
        if self._connect is not None:
            return self._connect(self.url)
        from websockets.sync.client import connect
        return connect(self.url, open_timeout=2, close_timeout=1, max_size=1 << 16)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ws = self._open()
            except Exception:
                self.connected = False
                if self._stop.wait(2.0):
                    return
                continue
            try:
                self.connected = True
                self.pump(ws)
            except Exception:
                pass
            finally:
                self.connected = False
                try:
                    ws.close()
                except Exception:
                    pass
            if self._stop.wait(1.0):
                return

    def pump(self, ws) -> None:
        """Read messages until stopped or the socket ends."""
        while not self._stop.is_set():
            try:
                msg = ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            self.feed(msg)

    def feed(self, msg) -> None:
        if not isinstance(msg, str) or not msg.startswith('{"type":"gesture"'):
            return            # hands messages (30/s) are not parsed here
        try:
            ev = json.loads(msg)
        except Exception:
            return
        if isinstance(ev, dict):
            self._on_event(ev)


def _ws_url() -> str:
    return f"ws://127.0.0.1:{int(_cfg('HAND_TRACK_PORT', 8793))}"


def _ensure_listener() -> None:
    if _listener[0] is None:
        _listener[0] = GestureListener(_ws_url())
    _listener[0].start()
    _state["listening"] = True


def _stop_listener() -> None:
    if _listener[0] is not None:
        _listener[0].stop()
    _state["listening"] = False


# ── actions ───────────────────────────────────────────────────────────────
def _staging() -> bool:
    try:
        from core import BLUE_GREEN_ROLE
        return BLUE_GREEN_ROLE == "staging"
    except Exception:
        return False


def start_tracking() -> tuple:
    """(ok, message): start the tracker + listener. Never raises."""
    la = _launcher()
    if la is None:
        return False, "the hand tracker module isn't available"
    if _staging():
        return False, "this is the staging copy, which never runs the tracker"
    ok, msg = la.start()
    if ok:
        _set_runtime(True)
        _ensure_listener()
    return ok, msg


def _hand_tracking_on(_: str = "") -> str:
    ok, msg = start_tracking()
    if not ok:
        return f"I couldn't start hand tracking, sir: {msg}."
    cam = str(_cfg("HAND_TRACK_CAMERA", "primary") or "primary")
    if msg == "already running":
        return "Hand tracking is already on, sir."
    menu = (" An open palm held up opens the hand menu." if bool(_cfg("HAND_MENU_ENABLED", False))
            else "")
    return (f"Hand tracking is on, sir. It's reading the {cam} camera and "
            f"never touches your cursor.{menu} Say 'hand tracking off' to stop it.")


def _hand_tracking_off(_: str = "") -> str:
    la = _launcher()
    was_on = bool(la is not None and (la.running() or la.wanted()))
    _stop_listener()
    if la is not None:
        la.stop()
    _set_runtime(False)
    if not was_on:
        return "Hand tracking was already off, sir."
    boot = (" It's still set to start at boot." if bool(_cfg("HAND_TRACKING_ENABLED", False))
            else "")
    return "Hand tracking is off, sir. Nothing is watching your hands." + boot


def status_text() -> str:
    la = _launcher()
    if la is None:
        return "The hand tracker module isn't available, sir."
    if not (la.running() or la.wanted()):
        if la.gave_up():
            return ("Hand tracking is off, sir - the tracker kept stopping, so "
                    "I gave up on it.")
        return "Hand tracking is off, sir."
    if la.gave_up():
        return ("Hand tracking is on but the tracker kept stopping, sir, so "
                "I gave up on it. Say 'hand tracking on' to try again.")
    hb = la.heartbeat()
    cam = str(_cfg("HAND_TRACK_CAMERA", "primary") or "primary")
    if not la.running():
        return "Hand tracking is starting up, sir."
    if hb is None or hb.get("age_s", 99) > 4:
        return "The hand tracker is loading, sir; it isn't answering yet."
    if not hb.get("attached"):
        return (f"The hand tracker is up, sir, but it isn't getting frames from "
                f"the {cam} camera yet.")
    hands = int(hb.get("hands", 0) or 0)
    fps = float(hb.get("fps", 0.0) or 0.0)
    seen = ("I can't see any hands" if hands == 0
            else "I can see one hand" if hands == 1 else "I can see both hands")
    menu = ("; the palm-hold menu is on" if bool(_cfg("HAND_MENU_ENABLED", False))
            else "")
    return (f"Hand tracking is on, sir: {seen}, tracking at {fps:.0f} frames a "
            f"second from the {cam} camera{menu}.")


def _hand_tracking_status(_: str = "") -> str:
    try:
        return status_text()
    except Exception as e:
        return f"I couldn't read the hand tracker's status, sir: {type(e).__name__}."


def start_at_boot() -> None:
    """Start the tracker when HAND_TRACKING_ENABLED is on. Runs on a daemon
    thread: it only spawns a process, but nothing here may delay boot."""
    def _go():
        try:
            ok, msg = start_tracking()
            print(f"{LOG} boot: " + ("tracker started" if ok else f"not started ({msg})"))
        except Exception as e:
            print(f"{LOG} boot start failed: {type(e).__name__}: {e}")
    threading.Thread(target=_go, name="hand-track-boot", daemon=True).start()


# ── registration ──────────────────────────────────────────────────────────
def register(actions: dict) -> None:
    actions["hand_tracking_on"] = _hand_tracking_on
    actions["hand_tracking_off"] = _hand_tracking_off
    actions["hand_tracking_status"] = _hand_tracking_status
    if bool(_cfg("HAND_TRACKING_ENABLED", False)):
        start_at_boot()
