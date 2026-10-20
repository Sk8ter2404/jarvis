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

The HAND MOUSE (2026-10-15, core/hand_mouse.py) lives here too:
  hand_mouse_on / hand_mouse_off / hand_mouse_status - drive the cursor with a
  pinch (hold still to engage, tap = click, second pinch held = drag, open palm
  = let go). OFF by default (HAND_MOUSE_ENABLED); "on" starts the tracker if it
  is not running; "hand tracking off" stops it too. Only while it is enabled does
  the listener parse the tracker's "hands" messages (30/s) - off, they are
  dropped unread, exactly as before. AIR_MOUSE_ENGINE = "hands" makes the
  existing air_mouse_on/off/status words route here (the routing lives in
  skills/kinect_air_mouse.py, which keeps the only registration of those names).

2026-10-20 additions, all through the same listener:
  RETICLE   while the hand mouse has the cursor, the Kinect air mouse's reticle
            overlay (hud/jarvis_air_cursor.py) shows it: its state file is
            written with skills/kinect_air_mouse's own writer and the overlay
            spawned with its own helper (only while the hand mouse is on, never
            in staging, never with a fake mouse output - i.e. never in tests;
            HAND_MOUSE_RETICLE). "hand mouse off" hides it and closes the
            overlay unless the Kinect air mouse is using it.
  TWO HANDS core/hand_two_hand.py: two fists grab the window between them,
            move it, spread / close to resize (HAND_TWO_HAND_ENABLED, default
            off, AIR_MOUSE_ENGINE = "hands" only). The hand mouse stands down
            while it is active.
  The hand mouse also stands down over an app on the air mouse's per-app
  disable list (AIR_MOUSE_PER_APP_DISABLE / AIR_MOUSE_DISABLED_APP_HINTS).

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
    "hand_mouse_on", "hand_mouse_off", "hand_mouse_status",
)

OPEN_COOLDOWN_S = 10.0
LOW_FPS = 15.0                       # status mentions the light below this
HOLO_ACTION = "holo_mode_on"
HOLO_ARG = "desktop"
LOG = "  [hand-track]"

_state = {"last_open": -1e9, "opens": 0, "last_event": "", "last_event_at": 0.0,
          "last_block": "", "warned_missing": False, "listening": False}
_open_hook: list = [None]            # optional replacement for the holo action
_listener: list = [None]
_mouse: list = [None]                # the core.hand_mouse.HandMouse, built on first use
_mouse_output_factory: list = [None]  # tests inject a fake; None = the real Windows one
_two_hand: list = [None]             # the core.hand_two_hand.TwoHandGrab, built on first use
_two_hand_backend_factory: list = [None]  # tests inject a fake; None = the real windows
_reticle_sink: list = [None]         # tests inject a fake; None = the real overlay (live only)
_reticle_state = {"last": None, "at": 0.0}
RETICLE_KEEPALIVE_S = 0.25           # rewrite this often while shown (the overlay hides
                                     # a state older than its STATE_STALE_HIDE_S, 0.6 s)


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


def _kinect_feed():
    """core.handtrack_depth's status (Kinect colour / depth hand-off), or {}."""
    try:
        from core import handtrack_depth
        return handtrack_depth.feed_status() or {}
    except Exception:
        return {}


def _camera_name() -> str:
    """The camera the tracker reads, as it should be spoken: "kinect" while the
    Kinect is selected (setting, or a Kinect hologram that is up), else the
    configured value."""
    try:
        from core import handtrack_frames
        cam = handtrack_frames._setting()
    except Exception:
        cam = str(_cfg("HAND_TRACK_CAMERA", "primary") or "primary")
    return "Kinect" if str(cam).strip().lower().startswith("kinect") else str(cam)


def _kinect_depth_phrase(hb: dict) -> str:
    """For the Kinect camera: whether DEPTH is really flowing (the tracker's own
    heartbeat says it sees a fresh plane), or why not. "" for any other camera."""
    fs = _kinect_feed()
    if not fs.get("selected"):
        return ""
    if not fs.get("depth_wanted"):
        return ", colour only (depth is off)"
    if hb.get("depth"):
        return ", with depth"
    return ", but depth isn't flowing yet"


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


# ── the hand mouse (core/hand_mouse.py) ───────────────────────────────────
def set_mouse_output_factory(fn: Optional[Callable[[], object]]) -> None:
    """Replace how the hand mouse reaches the OS (None = the real Windows
    layer in core/hand_mouse.py). Tests always set a fake; nothing else should."""
    _mouse_output_factory[0] = fn
    _mouse[0] = None


def _mouse_cue(kind: str, reason: str) -> None:
    """Short audible cue on engage / disengage (a log line is printed by the
    state machine). No popups. Off with HAND_MOUSE_CUE_SOUND = False."""
    if not bool(_cfg("HAND_MOUSE_CUE_SOUND", True)):
        return

    def _beep():
        try:
            import winsound
            winsound.Beep(1040 if kind == "engage" else 520, 70)
        except Exception:
            pass
    threading.Thread(target=_beep, name="hand-mouse-cue", daemon=True).start()


def _get_mouse():
    """The HandMouse, created on first use. None if it cannot be built."""
    if _mouse[0] is not None:
        return _mouse[0]
    try:
        from core import hand_mouse
        factory = _mouse_output_factory[0] or hand_mouse.WindowsMouse
        _mouse[0] = hand_mouse.HandMouse(
            factory(), blocker=mouse_blocked_reason, cue=_mouse_cue,
            params=hand_mouse.Params.from_config(_cfg),
            two_hand=_two_hand_active)
    except Exception as e:
        print(f"{LOG} hand mouse unavailable: {type(e).__name__}: {e}")
        _mouse[0] = None
    return _mouse[0]


def _mouse_enabled() -> bool:
    m = _mouse[0]
    return m is not None and bool(m.enabled)


def _wants_hands() -> bool:
    """Parse the tracker's "hands" messages at all? Only for the hand mouse or
    the two-hand grab; otherwise they are dropped unread, as before."""
    return _mouse_enabled() or _two_hand_enabled()


def _mouse_hands(msg: dict) -> None:
    """One "hands" message: the two-hand grab first (so the cursor knows to
    stand down on this very frame), then the hand mouse, then its reticle."""
    if _two_hand_enabled():
        g = _get_two_hand()
        if g is not None:
            g.on_hands(msg)
    elif _two_hand[0] is not None:
        _stop_two_hand()
    m = _mouse[0]
    if m is not None:
        m.on_hands(msg)
    _publish_reticle()


def _mouse_idle() -> None:
    m = _mouse[0]
    if m is not None and m.enabled:
        m.tick()
    _publish_reticle()


def _mouse_tracker_gone() -> None:
    m = _mouse[0]
    if m is not None:
        m.on_tracker_stopped()
    _stop_two_hand()
    _publish_reticle()


# ── the per-app stand-down (the air mouse's list) ─────────────────────────
def _per_app_blocked() -> bool:
    """The foreground app is on the air mouse's disable list
    (AIR_MOUSE_PER_APP_DISABLE / AIR_MOUSE_DISABLED_APP_HINTS, read live by
    skills/kinect_air_mouse._per_app_disabled). Never raises."""
    try:
        am = _air_mouse_mod()
        fn = getattr(am, "_per_app_disabled", None) if am is not None else None
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def mouse_blocked_reason() -> str:
    """Why the hand mouse (cursor, clicks, scroll) must not act ('' = it may):
    the palm-menu blockers plus the air mouse's per-app disable list."""
    why = blocked_reason()
    if why:
        return why
    if _per_app_blocked():
        return "an app on the air-mouse disable list is in front"
    return ""


def _air_mouse_mod():
    """skills/kinect_air_mouse.py (the loaded skill instance when there is one)."""
    mod = sys.modules.get("skill_kinect_air_mouse")
    if mod is not None:
        return mod
    try:
        import importlib
        return importlib.import_module("skills.kinect_air_mouse")
    except Exception:
        return None


# ── the reticle (the Kinect air mouse's overlay, 2026-10-20) ──────────────
class _OverlaySink:
    """Draws the hand mouse's cursor with the Kinect air mouse's reticle:
    its state-file writer, its overlay spawn, its shutdown - nothing copied."""

    def show(self, ret: dict) -> None:
        am = _air_mouse_mod()
        if am is None:
            return
        xy = (int(ret["x"]), int(ret["y"]))
        if ret["state"] == "prime":
            # engaged, the engaging pinch not yet opened: a full priming ring
            d = am.AirMouseDecision(None, None, None, "hidden", None, "closed", 1.0)
            am._publish_overlay_state(d, True, prime_xy=xy)
        else:
            d = am.AirMouseDecision(xy, None, None, ret["state"], None,
                                    "closed" if ret["state"] == "grab" else "open")
            am._publish_overlay_state(d, True)
        if not am._overlay_alive():
            am._spawn_overlay()

    def clear(self) -> None:
        am = _air_mouse_mod()
        if am is not None:
            am._clear_overlay_state()

    def shutdown(self) -> None:
        am = _air_mouse_mod()
        if am is None:
            return
        am._clear_overlay_state()
        if not bool(_cfg("KINECT_AIR_MOUSE_ENABLED", False)):
            am._shutdown_overlay()          # the Kinect air mouse is not using it


def set_reticle_sink(sink) -> None:
    """Replace the reticle output (tests). None = the real overlay, which is
    used only live: never in staging and never with a fake mouse output."""
    _reticle_sink[0] = sink
    _reticle_state["last"] = None


def _get_reticle_sink():
    if _reticle_sink[0] is not None:
        return _reticle_sink[0]
    if (_mouse_output_factory[0] is not None or _staging()
            or not bool(_cfg("HAND_MOUSE_RETICLE", True))):
        return None
    return _OverlaySink()


def _publish_reticle(now: Optional[float] = None) -> None:
    """Show / move / hide the reticle after a message. Writes only on a change
    (or every RETICLE_KEEPALIVE_S while shown), and hides it once. Stays quiet
    while the two-hand grab owns the overlay (dual reticles). Never raises."""
    try:
        if _two_hand_active():
            _reticle_state["last"] = None
            return
        sink = _get_reticle_sink()
        if sink is None:
            return
        now = time.monotonic() if now is None else now
        m = _mouse[0]
        ret = m.reticle() if m is not None else {"visible": False}
        last = _reticle_state["last"]
        if not ret.get("visible"):
            if last is not None and last[0]:
                sink.clear()
                _reticle_state["last"] = (False,)
            return
        key = (True, ret["x"], ret["y"], ret["state"])
        if key != last or now - _reticle_state["at"] >= RETICLE_KEEPALIVE_S:
            sink.show(ret)
            _reticle_state["last"] = key
            _reticle_state["at"] = now
    except Exception:
        pass


def _shutdown_reticle() -> None:
    try:
        sink = _get_reticle_sink()
        if sink is not None:
            sink.shutdown()
    except Exception:
        pass
    _reticle_state["last"] = None


# ── the two-hand window grab (core/hand_two_hand.py, 2026-10-20) ──────────
def _two_hand_enabled() -> bool:
    """HAND_TWO_HAND_ENABLED, with AIR_MOUSE_ENGINE = "hands", never in staging."""
    if not bool(_cfg("HAND_TWO_HAND_ENABLED", False)):
        return False
    if str(_cfg("AIR_MOUSE_ENGINE", "kinect") or "").strip().lower() != "hands":
        return False
    return not _staging()


def set_two_hand_backend_factory(fn: Optional[Callable[[], object]]) -> None:
    """Replace the two-hand grab's window / overlay layer (tests). None = the
    real one (skills/kinect_two_hand.py), used only with the real mouse output."""
    _two_hand_backend_factory[0] = fn
    _two_hand[0] = None


def _recent_input() -> bool:
    m = _mouse[0]
    try:
        idle = m.out.idle_s() if m is not None else None
        quiet = m.p.input_quiet_s if m is not None else 1.0
        return idle is not None and idle < quiet
    except Exception:
        return False


def _get_two_hand():
    if _two_hand[0] is not None:
        return _two_hand[0]
    try:
        from core import hand_mouse, hand_two_hand
        if _two_hand_backend_factory[0] is not None:
            win = _two_hand_backend_factory[0]()
        elif _mouse_output_factory[0] is not None:
            return None                     # a fake mouse = a test: no real windows
        else:
            win = hand_two_hand.WindowsBackend()
        _two_hand[0] = hand_two_hand.TwoHandGrab(
            win, params=hand_mouse.Params.from_config(_cfg),
            blocker=mouse_blocked_reason, recent_input=_recent_input)
    except Exception as e:
        print(f"{LOG} two-hand grab unavailable: {type(e).__name__}: {e}")
        _two_hand[0] = None
    return _two_hand[0]


def _two_hand_active() -> bool:
    g = _two_hand[0]
    try:
        return g is not None and g.active()
    except Exception:
        return False


def _stop_two_hand() -> None:
    g = _two_hand[0]
    if g is not None:
        try:
            g.stop()
        except Exception:
            pass


def _set_mouse_runtime(on: bool) -> None:
    try:
        from core import config
        config.HAND_MOUSE_ENABLED = bool(on)
    except Exception:
        pass


def _sync_mouse() -> None:
    """Make the hand mouse's enabled flag follow HAND_MOUSE_ENABLED once the
    tracker is up (boot setting). Builds it only if the flag is on."""
    if bool(_cfg("HAND_MOUSE_ENABLED", False)):
        m = _get_mouse()
        if m is not None:
            m.set_enabled(True)


def _stop_hand_mouse() -> bool:
    """Turn the hand mouse off and let go of everything. True if it was on."""
    was = bool(_cfg("HAND_MOUSE_ENABLED", False)) or _mouse_enabled()
    _set_mouse_runtime(False)
    m = _mouse[0]
    if m is not None:
        m.set_enabled(False)
    if was:
        _shutdown_reticle()
    return was


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
        if name == "pinch_hold" and _mouse_enabled():
            _mouse[0].on_gesture(ev, now)      # engage the hand mouse (if allowed)
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
                 on_event: Callable[[dict], object] = handle_gesture,
                 on_hands: Optional[Callable[[dict], object]] = None,
                 wants_hands: Optional[Callable[[], bool]] = None,
                 on_idle: Optional[Callable[[], object]] = None,
                 on_disconnect: Optional[Callable[[], object]] = None):
        self.url = url
        self._connect = connect
        self._on_event = on_event
        # The hand mouse's hooks. "hands" messages (30/s) are parsed ONLY when
        # on_hands is set AND wants_hands() is true - otherwise dropped unread.
        self._on_hands = on_hands
        self._wants_hands = wants_hands
        self._on_idle = on_idle
        self._on_disconnect = on_disconnect
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
                self._call(self._on_disconnect)
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
                self._call(self._on_idle)
                continue
            self.feed(msg)

    @staticmethod
    def _call(fn) -> None:
        if fn is not None:
            try:
                fn()
            except Exception:
                pass

    def feed(self, msg) -> None:
        if not isinstance(msg, str):
            return
        if msg.startswith('{"type":"hands"'):
            # 30/s: parsed only for the hand mouse, and only while it is on.
            if self._on_hands is None or self._wants_hands is None:
                return
            try:
                if not self._wants_hands():
                    return
                data = json.loads(msg)
                if isinstance(data, dict):
                    self._on_hands(data)
            except Exception:
                pass
            return
        if not msg.startswith('{"type":"gesture"'):
            return
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
        _listener[0] = GestureListener(
            _ws_url(), on_hands=_mouse_hands, wants_hands=_wants_hands,
            on_idle=_mouse_idle, on_disconnect=_mouse_tracker_gone)
    _listener[0].start()
    _state["listening"] = True


def _stop_listener() -> None:
    if _listener[0] is not None:
        _listener[0].stop()
    _state["listening"] = False
    _mouse_tracker_gone()              # the tracker is going: let go of the mouse


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
        _sync_mouse()
    return ok, msg


# ── the hologram's own use of the tracker ───────────────────────────────
# "Open up the hologram" needs the camera, so holo_mode_on (skills/holo_mode.py)
# starts the tracker when it isn't running. Only a tracker the HOLOGRAM started
# is stopped again when the hologram closes - by voice, or when its window has
# been gone for HOLO_IDLE_STOP_S. One the owner turned on ("hand tracking on" or
# the boot setting) stays his: the hologram never turns it off.
HOLO_IDLE_STOP_S = 45.0
HOLO_WATCH_EVERY_S = 5.0
_holo = {"owned": False, "idle_since": None, "watch": None}


def _holo_page_alive() -> bool:
    try:
        from core import holo_web
        return bool(holo_web.STATE.page_alive)
    except Exception:
        return False


def ensure_for_holo() -> tuple:
    """(ok, message): the hologram is opening - make sure the tracker runs.
    Never raises."""
    la = _launcher()
    if la is not None and la.running():
        return True, "already running"
    ok, msg = start_tracking()
    if ok and msg != "already running":
        _holo["owned"] = True
        _holo["idle_since"] = time.monotonic()   # grace while the window opens
        _start_holo_watch()
    return ok, msg


def release_for_holo() -> bool:
    """The hologram closed: stop the tracker only if the hologram started it.
    True when it was stopped."""
    if not _holo["owned"]:
        return False
    _holo["owned"] = False
    _holo["idle_since"] = None
    _stop_listener()
    la = _launcher()
    if la is not None:
        la.stop()
    _set_runtime(False)
    print(f"{LOG} hologram closed - stopped the tracker it started")
    return True


def _holo_watch_step(now: float, page_alive: bool) -> bool:
    """One check of a hologram-owned tracker: True when it was stopped because
    the hologram window has been gone for HOLO_IDLE_STOP_S."""
    if not _holo["owned"]:
        return False
    if page_alive:
        _holo["idle_since"] = None
        return False
    if _holo["idle_since"] is None:
        _holo["idle_since"] = now
        return False
    if now - _holo["idle_since"] >= HOLO_IDLE_STOP_S:
        return release_for_holo()
    return False


def _start_holo_watch() -> None:
    t = _holo.get("watch")
    if t is not None and t.is_alive():
        return

    def _loop():
        while _holo["owned"]:
            time.sleep(HOLO_WATCH_EVERY_S)
            try:
                _holo_watch_step(time.monotonic(), _holo_page_alive())
            except Exception:
                pass
    t = threading.Thread(target=_loop, name="holo-tracker-watch", daemon=True)
    _holo["watch"] = t
    t.start()


def _hand_tracking_on(_: str = "") -> str:
    _holo["owned"] = False                 # the owner asked: it is his now
    ok, msg = start_tracking()
    if not ok:
        return f"I couldn't start hand tracking, sir: {msg}."
    cam = _camera_name()
    if msg == "already running":
        return "Hand tracking is already on, sir."
    menu = (" An open palm held up opens the hand menu." if bool(_cfg("HAND_MENU_ENABLED", False))
            else "")
    return (f"Hand tracking is on, sir. It's reading the {cam} camera and "
            f"never touches your cursor.{menu} Say 'hand tracking off' to stop it.")


def _stop_air_mouse() -> bool:
    """'Hand tracking off' means everything that watches his hands (10-13: "turn
    off hand tracking ... I think that's what's stealing your mouse"), so a
    running Kinect air mouse is stopped too. True when it was on and is now
    off. Never raises."""
    if not bool(_cfg("KINECT_AIR_MOUSE_ENABLED", False)):
        return False
    fn = _registry().get("air_mouse_off")
    if not callable(fn):
        return False
    try:
        fn("")
    except Exception:
        return False
    return not bool(_cfg("KINECT_AIR_MOUSE_ENABLED", False))


def _still_on_tail() -> str:
    """Honest close: name any Kinect hand feature still on, else say nothing
    is watching his hands."""
    others = [name for key, name in (("KINECT_GESTURES_ENABLED", "Kinect gestures"),
                                     ("KINECT_POINT_CONTROL_ENABLED", "point control"))
              if bool(_cfg(key, False))]
    if not others:
        return " Nothing is watching your hands."
    return " " + " and ".join(others) + (" is" if len(others) == 1 else " are") + " still on."


def _hand_tracking_off(_: str = "") -> str:
    _holo["owned"] = False
    _holo["idle_since"] = None
    la = _launcher()
    was_on = bool(la is not None and (la.running() or la.wanted()))
    hand_mouse = _stop_hand_mouse()
    _stop_listener()
    if la is not None:
        la.stop()
    _set_runtime(False)
    mouse = _stop_air_mouse()
    if not was_on and not mouse and not hand_mouse:
        return "Hand tracking was already off, sir."
    boot = (" It's still set to start at boot." if bool(_cfg("HAND_TRACKING_ENABLED", False))
            else "")
    if not was_on:
        return ("The hand mouse is off, sir." if hand_mouse
                else "The air mouse is off, sir.") + _still_on_tail()
    return ("Hand tracking is off, sir."
            + (" The hand mouse is off too." if hand_mouse else "")
            + (" The air mouse is off too." if mouse else "")
            + _still_on_tail() + boot)


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
    cam = _camera_name()
    if not la.running():
        return "Hand tracking is starting up, sir."
    if hb is None or hb.get("age_s", 99) > 4:
        return "The hand tracker is loading, sir; it isn't answering yet."
    if not hb.get("attached"):
        why = ""
        if cam == "Kinect":
            fs = _kinect_feed()
            if fs.get("reason") == "kinect not open":
                why = " - the Kinect isn't open (it may be off in settings)"
            elif fs.get("last_error"):
                why = " - the Kinect hand-off hit an error"
        return (f"The hand tracker is up, sir, but it isn't getting frames from "
                f"the {cam} camera yet{why}.")
    hands = int(hb.get("hands", 0) or 0)
    fps = float(hb.get("fps", 0.0) or 0.0)
    seen = ("I can't see any hands" if hands == 0
            else "I can see one hand" if hands == 1 else "I can see both hands")
    menu = ("; the palm-hold menu is on" if bool(_cfg("HAND_MENU_ENABLED", False))
            else "")
    # Live 2026-10-14: the webcam (30 fps capable) gave ~12-15 in room light -
    # auto-exposure slows the sensor when the scene is dim. Say so when it
    # shows, since quick gestures (the flick, the clap) get harder below that.
    slow = ("; that's on the slow side - more light on your hands lets the "
            "webcam run faster" if 0 < fps < LOW_FPS else "")
    depth = _kinect_depth_phrase(hb) if cam == "Kinect" else ""
    return (f"Hand tracking is on, sir: {seen}, tracking at {fps:.0f} frames a "
            f"second from the {cam} camera{depth}{slow}{menu}.")


def _hand_tracking_status(_: str = "") -> str:
    try:
        return status_text()
    except Exception as e:
        return f"I couldn't read the hand tracker's status, sir: {type(e).__name__}."


def _hand_mouse_on(_: str = "") -> str:
    _holo["owned"] = False                 # the owner asked: the tracker is his now
    already = bool(_cfg("HAND_MOUSE_ENABLED", False)) and _mouse_enabled()
    la = _launcher()
    if la is None or not la.running():
        ok, msg = start_tracking()
        if not ok:
            return f"I couldn't start the hand mouse, sir: {msg}."
    _ensure_listener()
    _set_mouse_runtime(True)
    m = _get_mouse()
    if m is None:
        _set_mouse_runtime(False)
        return "I couldn't set up the hand mouse, sir."
    m.set_enabled(True)
    if already:
        return "The hand mouse is already on, sir."
    why = blocked_reason()
    tail = f" It won't take the cursor while {why}." if why else ""
    scroll = (" make a fist and move it up or down to scroll,"
              if bool(_cfg("HAND_MOUSE_SCROLL_ENABLED", True)) else "")
    return ("Hand mouse on, sir. Hold a pinch still for a moment to take the "
            "cursor, tap to click, pinch and hold to drag," + scroll + " and hold "
            "an open palm to let go. Say 'hand mouse off' to stop it." + tail)


def _hand_mouse_off(_: str = "") -> str:
    if not _stop_hand_mouse():
        return "The hand mouse is already off, sir."
    la = _launcher()
    left = (" Hand tracking is still on; say 'hand tracking off' to stop it."
            if la is not None and la.running() else "")
    return "Hand mouse off, sir. I've let go of the cursor." + left


def hand_mouse_status_text() -> str:
    m = _mouse[0]
    if not (bool(_cfg("HAND_MOUSE_ENABLED", False)) and m is not None and m.enabled):
        return "The hand mouse is off, sir."
    la = _launcher()
    if la is None or not la.running():
        return "The hand mouse is on, sir, but the hand tracker isn't running."
    st = m.status()
    if st["phase"] == "idle":
        why = blocked_reason()
        if why:
            return f"The hand mouse is on, sir, but it won't take the cursor while {why}."
        return "The hand mouse is on, sir. Hold a pinch still to take the cursor."
    if st["pressed"]:
        return "The hand mouse has the cursor, sir, and the button is held for a drag."
    if st["phase"] == "engaging":
        return "The hand mouse is engaged, sir. Let go of the pinch and move."
    return "The hand mouse has the cursor, sir. Open your palm to let go."


def _hand_mouse_status(_: str = "") -> str:
    try:
        return hand_mouse_status_text()
    except Exception as e:
        return f"I couldn't read the hand mouse status, sir: {type(e).__name__}."


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
    actions["hand_mouse_on"] = _hand_mouse_on
    actions["hand_mouse_off"] = _hand_mouse_off
    actions["hand_mouse_status"] = _hand_mouse_status
    if bool(_cfg("HAND_TRACKING_ENABLED", False)) or bool(_cfg("HAND_MOUSE_ENABLED", False)):
        start_at_boot()
