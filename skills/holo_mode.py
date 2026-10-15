"""holo_mode skill - the hand-tracked hologram window ("hand menu" / "the hologram").

THE FEATURE
===========
A browser page (hud/holo/, served by the web interface at /holo) with a
procedural three.js arc reactor the owner drives with his hands: pinch and hold
to spawn it, a fist to move it, a point-and-flick to spin it, two fists apart
(or palms together then apart, or a twisted fist) to explode it into its named
parts, a pinch to pull a part out and hold it for its info, a clap to put it
back together.

Hands are NOT read here and not by the page either: a separate tracker service
(tools/hand_tracker_service.py, ws://127.0.0.1:8793) owns the camera frames and
the hand model and sends landmarks + gesture events. Two modes, one page:
  desktop - a solid dark JARVIS window, no camera picture ("hand menu")
  camera  - the tracker's mirrored camera view behind the reactor ("hologram")

WHAT THIS MODULE DOES
  * opens the page as a Chrome app window (chrome.exe --app=URL), starting the
    web interface first when it is down (the owner just asked for it);
  * writes commands into core.holo_web.STATE, which the page polls every
    ~300 ms (GET /api/holo/state);
  * answers honestly: nothing up -> says so, unknown model -> says what exists.

NEVER moves the Windows cursor or sends OS input; the page only draws.
HOLO_MODE_ENABLED (core/config.py) False -> every action says it is off.

Voice actions (each returns ONE finished sentence, spoken verbatim):
  hand_menu, holo_mode_on [desktop|camera], holo_mode_off, holo_show [model],
  holo_explode, holo_assemble

holo_show also shows the owner's OWN models (.stl / .3mf / .obj in
data/holo_models or the HOLO_MODEL_DIRS folders, see core/holo_web.py): the
name is matched fuzzily against the file names; an unknown name is answered
with a few that exist, and with no models at all, with where to put them.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from core import holo_web

SPEAK_VERBATIM_ACTIONS = ("hand_menu", "holo_mode_on", "holo_mode_off",
                          "holo_show", "holo_explode", "holo_assemble")

# Injected by the skill loader before exec_module (pyflakes-visible definition).
skill_utils = globals().get("skill_utils", {})

DEFAULT_MODEL = holo_web.MODELS[0]
# A freshly opened window needs a moment before its first poll marks it alive;
# a second request inside this window must not open a second one.
OPENING_GRACE_S = 8.0
_last_open = [0.0]
_MODE_WORDS = {
    "desktop": "desktop", "menu": "desktop", "hand menu": "desktop",
    "camera": "camera", "hologram": "camera", "holo": "camera",
    "mirror": "camera",
}


# --- helpers ------------------------------------------------------------------
def _cfg(name: str, default):
    try:
        from core import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def _enabled() -> bool:
    return bool(_cfg("HOLO_MODE_ENABLED", True))


def _web():
    """The LOADED web_interface skill (it owns the one running server), or None.
    Looked up by its loader name; never imported a second time, which would
    create a second, empty copy of its server state."""
    return sys.modules.get("skill_web_interface")


def _server_up():
    """(running, host, port) of the web interface, read from its skill."""
    mod = _web()
    if mod is None:
        return False, "", 0
    if getattr(mod, "_httpd", None) is None:
        return False, "", 0
    bind, port = getattr(mod, "_bound", ("", 0))
    return True, str(bind), int(port)


def _ensure_server():
    """(ok, host, port, why). Starts the web interface when it is down: the
    owner asked for holo mode, which is consent (the same rule as saying
    'start the web interface'). Never raises."""
    mod = _web()
    if mod is None:
        return False, "", 0, ("The web interface isn't loaded, sir, so I can't "
                              "serve the hologram page.")
    up, bind, port = _server_up()
    if not up:
        try:
            ok, msg = mod._start()
        except Exception as e:                    # pragma: no cover - defensive
            return False, "", 0, f"I couldn't start the web interface, sir - {e}."
        if not ok:
            return False, "", 0, msg
        up, bind, port = _server_up()
        if not up:
            return False, "", 0, "The web interface didn't come up, sir."
    host = "127.0.0.1" if bind in ("", "0.0.0.0", "::", "localhost") else bind
    return True, host, port, ""


def _chrome_paths():
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    local = os.environ.get("LOCALAPPDATA", "")
    return [
        os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(pf86, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(local, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(pf86, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(pf, "Microsoft", "Edge", "Application", "msedge.exe"),
    ]


def _launch_window(url: str) -> bool:
    """Open ``url`` as a Chrome (else Edge) APP window - own window, no tabs, no
    URL bar - so it behaves like a JARVIS window. Falls back to the injected
    open_url helper (a normal tab) when neither browser is found. True when
    something was opened. Tests replace this function; the browser guard would
    block a real launch anyway."""
    for exe in _chrome_paths():
        if exe and os.path.isfile(exe):
            try:
                subprocess.Popen([exe, f"--app={url}", "--window-size=1280,800"],
                                 close_fds=True)
                return True
            except Exception as e:
                print(f"  [holo] app window failed ({type(e).__name__}); "
                      f"trying the next browser")
    fn = skill_utils.get("open_url") if isinstance(skill_utils, dict) else None
    if callable(fn):
        try:
            fn(url)
            return True
        except Exception as e:
            print(f"  [holo] open_url failed: {type(e).__name__}")
    return False


def _norm_model(arg: str) -> str:
    a = (arg or "").strip().strip("\"'").lower()
    for junk in ("the ", "a ", "an ", "my "):
        if a.startswith(junk):
            a = a[len(junk):]
    for tail in (" in the hologram", " in the holo", " in the hand menu",
                 " in holo mode", " please", " model", " file"):
        if a.endswith(tail):
            a = a[:-len(tail)]
    return a.strip()


def _norm_mode(arg: str, default: str) -> str:
    a = (arg or "").strip().lower()
    for word, mode in _MODE_WORDS.items():
        if word in a:
            return mode
    return default


def _open_page(mode: str, model: str = ""):
    """Start the server if needed and open the window. Returns (ok, why)."""
    ok, host, port, why = _ensure_server()
    if not ok:
        return False, why
    holo_web.STATE.set_on(True, model=model, mode=mode)
    url = f"http://{host}:{port}/holo?mode={mode}"
    if not _launch_window(url):
        holo_web.STATE.set_on(False)
        return False, "I couldn't open a browser window for it, sir."
    _last_open[0] = time.monotonic()
    return True, ""


def _opening() -> bool:
    """True while a window we just opened has not polled yet."""
    return (holo_web.STATE.on and not holo_web.STATE.page_alive
            and time.monotonic() - _last_open[0] < OPENING_GRACE_S)


def _off_line() -> str:
    return "Holo mode is switched off in my settings, sir."


# --- actions ------------------------------------------------------------------
def _tracker():
    """The LOADED hand_tracking skill (it owns the tracker service), or None."""
    return sys.modules.get("skill_hand_tracking")


def _ensure_tracker() -> str:
    """Start the hand tracker for the hologram when it isn't running (the
    owner asked for the hologram, which needs the camera). Returns "" or a
    short honest note to add to the reply. Never raises."""
    trk = _tracker()
    if trk is None or not hasattr(trk, "ensure_for_holo"):
        return " Hand tracking isn't loaded, so it won't see your hands."
    try:
        ok, msg = trk.ensure_for_holo()
    except Exception as e:                        # pragma: no cover - defensive
        ok, msg = False, type(e).__name__
    if ok:
        return ""
    return f" Hand tracking couldn't start ({msg}), so it won't see your hands."


def _release_tracker() -> None:
    trk = _tracker()
    if trk is not None and hasattr(trk, "release_for_holo"):
        try:
            trk.release_for_holo()
        except Exception:                         # pragma: no cover - defensive
            pass


def holo_mode_on(arg: str = "") -> str:
    if not _enabled():
        return _off_line()
    mode = _norm_mode(arg, "camera")
    st = holo_web.STATE
    if _opening():
        return "It's still coming up, sir."
    note = _ensure_tracker()
    if st.page_alive:
        if st.mode != mode:
            st.push_command("mode", mode)
            return "Switched the hologram view, sir." + note
        return "Holo mode is already up, sir." + note
    ok, why = _open_page(mode)
    if not ok:
        return why
    return ("Hand menu up, sir." if mode == "desktop" else "Holo mode up, sir.") + note


def hand_menu(arg: str = "") -> str:
    return holo_mode_on("desktop")


def holo_mode_off(arg: str = "") -> str:
    if not _enabled():
        return _off_line()
    st = holo_web.STATE
    if not (st.on or st.page_alive):
        return "Holo mode isn't up, sir."
    st.push_command("close")
    st.set_on(False)
    _release_tracker()
    return "Holo mode closed, sir."


def _own_models() -> list:
    """The owner's own models (core.holo_web.list_models); [] on any trouble."""
    try:
        return list(holo_web.list_models())
    except Exception as e:                        # pragma: no cover - defensive
        print(f"  [holo] model listing failed: {type(e).__name__}")
        return []


def _resolve_model(query: str, models: list):
    """(name to show, "" | spoken refusal). The built-in arc reactor is always
    a candidate; a model of his with the exact same name wins over it."""
    if not query:
        return DEFAULT_MODEL, ""
    cands = [{"name": DEFAULT_MODEL, "type": "builtin"}] + list(models)
    hit = holo_web.match_model(query, cands)
    if hit is None and query in ("reactor", "the reactor"):
        hit = cands[0]
    if hit is not None:
        return hit["name"], ""
    if not models:
        return "", ("I can't find any of your models, sir - put .stl, .3mf or .obj "
                    "files in data/holo_models or add a folder to HOLO_MODEL_DIRS.")
    names = []
    for m in models:
        if m["name"] not in names:
            names.append(m["name"])
    few = names[:3]
    listed = (", ".join(few[:-1]) + " and " + few[-1]) if len(few) > 1 else few[0]
    return "", f"I don't have a model called {query}, sir - I do have {listed}."


def holo_show(arg: str = "") -> str:
    if not _enabled():
        return _off_line()
    query = _norm_model(arg)
    model, refusal = _resolve_model(query, _own_models() if query else [])
    if refusal:
        return refusal
    st = holo_web.STATE
    if _opening():
        st.push_command("show", model)
        return f"The {model} will be up in a moment, sir."
    if st.page_alive:
        st.push_command("show", model)
        return f"The {model} is up, sir."
    ok, why = _open_page("desktop", model=model)
    if not ok:
        return why
    return f"The {model} is up in the hand menu, sir."


def _need_model() -> str:
    """'' when a model is on screen; else the honest reason there isn't."""
    st = holo_web.STATE
    if not st.page_alive:
        return "Holo mode isn't up, sir."
    if not st.shown:
        return "There's nothing on the holo display yet, sir."
    return ""


def holo_explode(arg: str = "") -> str:
    if not _enabled():
        return _off_line()
    why = _need_model()
    if why:
        return why
    if holo_web.STATE.parts == 1:
        return "It's a single piece, sir, so there's nothing to explode."
    if holo_web.STATE.exploded:
        return "It's already apart, sir."
    holo_web.STATE.push_command("explode")
    return "Exploding it, sir."


def holo_assemble(arg: str = "") -> str:
    if not _enabled():
        return _off_line()
    why = _need_model()
    if why:
        return why
    if not holo_web.STATE.exploded:
        return "It's already in one piece, sir."
    holo_web.STATE.push_command("assemble")
    return "Putting it back together, sir."


def register(actions: dict):
    actions["holo_mode_on"] = holo_mode_on
    actions["holo_mode_off"] = holo_mode_off
    actions["hand_menu"] = hand_menu
    actions["holo_show"] = holo_show
    actions["holo_explode"] = holo_explode
    actions["holo_assemble"] = holo_assemble
