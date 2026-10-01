"""
Kinect v2 voice actions for JARVIS.

Exposes the Xbox Kinect v2 sensor (skeleton presence + 1080p color + depth +
infrared) to the voice loop via four spoken-friendly actions:

  kinect_status            — is the Kinect connected, which streams are live,
                             how many people are in view.
  scan_room / who_is_here  — body count + nearest distance ("one person about
                             6 feet away, sir").
  kinect_look /
  what_do_you_see_kinect   — grab the Kinect color frame and route it through
                             ask_vision ("Looking through the Kinect, sir — …").

Everything degrades gracefully when the sensor is off (KINECT_ENABLED=False,
the privacy default) or absent — the actions say so honestly instead of
pretending. All sensor contact goes through audio/kinect_bridge.py, whose
accessors never raise and never touch pykinect2 at import time.
"""

import sys
import time


def _bridge():
    """The live kinect_bridge module, or None. Prefer the instance the monolith
    already imported (audio.kinect_bridge); fall back to a direct import."""
    mod = sys.modules.get("audio.kinect_bridge")
    if mod is not None:
        return mod
    try:
        from audio import kinect_bridge as _kb
        return _kb
    except Exception:
        return None


def _bc():
    """Live monolith module (main or by-name), or None."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _distance_phrase(metres) -> str:
    """'about 6 feet away' / '' when unknown. The owner uses US imperial for
    everyday distances (feet/yards) — metric is reserved for 3D printing — so
    convert the Kinect's native metres before speaking. 2026-07-08."""
    if not metres:
        return ""
    from core.units import meters_to_imperial_phrase
    return f" about {meters_to_imperial_phrase(metres)} away"


# How fresh a color/body frame must be to count as "streaming". The 30 Hz body
# pump consumes body and primes color, so both ages advance every ~33 ms on a
# live sensor.
_STREAM_LIVE_MAX_AGE_S = 1.0
# Depth has no poller, so its liveness is MEASURED: consume one frame, wait a
# few depth frames (30 Hz), and require a NEW one to have arrived.
_DEPTH_LIVE_PROBE_S = 0.15


def _live_streams(kb) -> list:
    """Which Kinect streams are delivering NEW frames right now. NEVER raises.

    STALE DUPLICATE FIXED 2026-10-01. This used to count a stream as live when
    get_color_bgr(require_new=False) / get_depth() returned non-None - but on
    the installed pykinect2 build both re-serve a frozen buffer FOREVER (the
    zero-filled buffers exist from construction), so the probe said "color,
    depth" about a sensor that had stopped streaming. The bridge grew
    get_stream_health() for exactly this (its docstring names this probe as the
    lie) and tools/web_interface.py moved to it; this voice action never did.
    It reads freshness now, never "did a getter return bytes"."""
    try:
        h = kb.get_stream_health() or {}
    except Exception:
        h = {}

    def _fresh(key):
        # The AGE, not *_pending: pending stays True for as long as nobody
        # consumes the last frame, so a dead pump would pin it (the depth trap).
        age = h.get(f"{key}_age_s")
        return age is not None and age < _STREAM_LIVE_MAX_AGE_S

    streams = []
    if _fresh("color"):
        streams.append("color")
    if _fresh("body"):
        streams.append("body")
    # Nothing polls depth, so depth_pending stays True after the sensor dies
    # until somebody consumes the last frame, and depth_age_s only means
    # "nobody asked recently" (see web_interface's color-only rung). Consume
    # one, then require a NEW frame within a few frame periods.
    try:
        kb.get_depth(require_new=True)
        time.sleep(_DEPTH_LIVE_PROBE_S)
        if (kb.get_stream_health() or {}).get("depth_pending"):
            streams.append("depth")
    except Exception:
        pass
    # Infrared is not probed: the bridge reports it "unsupported" on this
    # build, and a has_new flag on a fuller build is the same sticky-flag trap.
    return streams


# ─── actions ─────────────────────────────────────────────────────────────

def kinect_status(_: str = "") -> str:
    """Honest report of Kinect connectivity + which streams are live + people
    in view."""
    kb = _bridge()
    if kb is None:
        return ("I don't have the Kinect bridge loaded, sir — the pykinect2 "
                "module may not be installed.")
    if not kb.get_enabled():
        return ("The Kinect is switched off, sir — it's disabled by default for "
                "privacy. Enable KINECT_ENABLED to let me use it.")
    ok, reason = kb.available()
    if not ok:
        return f"The Kinect isn't available right now, sir — {reason}"

    streams = _live_streams(kb)
    if not streams:
        # A published runtime with no frames arriving is the R11 "dies seconds
        # after every open" Kinect (or a stalled sensor before the bridge's
        # 4 s stale reset): say so instead of "streaming (no streams yet)".
        return ("The Kinect is connected, sir, but no frames are arriving "
                "right now — it may have dropped off.")

    presence = {}
    try:
        presence = kb.get_presence()
    except Exception:
        presence = {}
    count = int(presence.get("count", 0) or 0)

    stream_str = ", ".join(streams)
    if count == 0:
        people = "no one in view at the moment"
    elif count == 1:
        people = "one person in view"
    else:
        people = f"{count} people in view"
    return (f"Kinect is connected and streaming ({stream_str}), sir — {people}.")


def who_is_here(_: str = "") -> str:
    """Body count + nearest distance from the Kinect skeleton tracker."""
    kb = _bridge()
    if kb is None:
        return "I can't reach the Kinect, sir — the bridge isn't loaded."
    if not kb.get_enabled():
        return ("The Kinect is off, sir — enable KINECT_ENABLED and I can count "
                "who's in the room.")
    ok, reason = kb.available()
    if not ok:
        return f"I can't see the room through the Kinect right now, sir — {reason}"
    try:
        presence = kb.get_presence()
    except Exception:
        presence = {}
    count = int(presence.get("count", 0) or 0)
    if count == 0:
        return "I don't see anyone in the room right now, sir."
    nearest = _distance_phrase(presence.get("nearest_m"))
    facing = presence.get("facing")
    if count == 1:
        face_note = ""
        if facing is True:
            face_note = ", facing me"
        elif facing is False:
            face_note = ", turned away"
        return f"I can see one person{nearest}{face_note}, sir."
    return f"I can see {count} people, sir — the nearest{nearest}."


# scan_room is an alias of who_is_here (the spec lists both phrasings).
def scan_room(_: str = "") -> str:
    return who_is_here(_)


def kinect_look(question: str = "") -> str:
    """Grab the Kinect color frame and route it through ask_vision."""
    kb = _bridge()
    if kb is None:
        return "I can't look through the Kinect, sir — its bridge isn't loaded."
    if not kb.get_enabled():
        return ("The Kinect is switched off, sir — enable KINECT_ENABLED and I "
                "can look through it.")
    ok, reason = kb.available()
    if not ok:
        return f"I can't see through the Kinect right now, sir — {reason}"
    try:
        png = kb.get_color_png()
    except Exception as e:
        return f"The Kinect didn't give me an image, sir — {type(e).__name__}."
    if png is None:
        return ("The Kinect is connected but didn't hand me a frame, sir — give "
                "it a moment and try again.")

    q = (question or "").strip() or ("Describe what you see in this room through "
                                     "the Kinect camera — who and what is present.")
    # Prefer the injected skill_utils seam; fall back to the monolith function.
    ask = None
    su = globals().get("skill_utils")
    if isinstance(su, dict):
        ask = su.get("ask_vision")
    if not callable(ask):
        bc = _bc()
        ask = getattr(bc, "ask_vision", None) if bc is not None else None
    if not callable(ask):
        return ("I have the Kinect image, sir, but vision isn't wired up to "
                "describe it.")
    try:
        answer = ask(q, png)
    except Exception as e:
        return f"Looking through the Kinect, sir — but vision failed: {e}"
    answer = (answer or "").strip()
    if not answer:
        return "Looking through the Kinect, sir — but I couldn't make it out."
    return f"Looking through the Kinect, sir — {answer}"


def what_do_you_see_kinect(question: str = "") -> str:
    return kinect_look(question)


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    actions["kinect_status"]          = kinect_status
    actions["who_is_here"]            = who_is_here
    actions["scan_room"]              = scan_room
    actions["kinect_look"]            = kinect_look
    actions["what_do_you_see_kinect"] = what_do_you_see_kinect
    print("  [kinect] voice actions registered "
          "(kinect_status, scan_room, who_is_here, kinect_look)")
