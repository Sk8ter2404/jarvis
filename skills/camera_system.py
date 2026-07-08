"""
Unified multi-camera awareness for JARVIS.

JARVIS has THREE cameras: two monitor-mounted USB webcams (tracked by the
monolith's _face_tracking_thread) and a room-facing Xbox Kinect v2 (the
audio/kinect_bridge.py sensor). Until now they were three separate signals —
the webcams drove gaze/which-monitor, the Kinect drove presence/skeleton, and
nothing tied them together. This skill FUSES them into one situational picture
and exposes three spoken-friendly actions:

  camera_status            — enumerate EVERY camera (both webcams + the Kinect)
                             and report each one's role + LIVE health. A webcam
                             is "live" if the face-track thread has a fresh
                             frame for its index; the Kinect via the bridge's
                             available()/presence. Honest about a dark/stale one.
                             'what cameras do you have', 'camera status'.

  situational_awareness /  — fuse webcam gaze (which monitor + dwell, from the
  where_am_i               face_tracker skill's state) with Kinect presence
                             (count, nearest distance, facing) into one coherent
                             line: "You're at the desk, about 0.7 metres back,
                             facing the right monitor, alone, sir." Degrades
                             gracefully to webcams-only / Kinect-only / nothing
                             and always SAYS which sources it used.
                             'where am I', 'what am I doing', "what's my status".

  look_around              — capture a frame from EACH available source (webcams
                             from the shared frame cache, the Kinect via the
                             bridge) and describe the whole scene in one short
                             paragraph that names each vantage. COST-CONSCIOUS:
                             prefers the free LOCAL VLM and says in the reply
                             when a vantage fell through to the cloud.
                             'look around', 'what do you see everywhere'.

Design notes
------------
* This skill OWNS no sensor state. It reads the monolith's shared frame caches
  (_camera_latest_frame / _camera_last_seen / _camera_last_frame_at under
  _camera_state_lock), the face_tracker skill's gaze snapshot, and the Kinect
  bridge — every one of which already exists. It never mutates any of them.
* Every external touch is wrapped so a missing monolith / absent Kinect / dead
  webcam degrades to an honest spoken line instead of raising into the voice
  loop. The single-camera see_user / kinect_look actions are left untouched;
  look_around is the new multi-cam sweep.
* The Kinect is opt-in (KINECT_ENABLED, off by default for privacy). When it's
  off or absent, every action simply reports the two webcams and says the
  Kinect is dark — it never pretends.
"""

import sys
import time


# ─── seams to the rest of JARVIS ─────────────────────────────────────────

def _bc():
    """The live monolith module (main or by-name), or None. Mirrors the lookup
    every other camera-aware skill uses."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _kinect_bridge():
    """The live kinect_bridge module, or None. Prefer the instance the monolith
    already imported (audio.kinect_bridge); fall back to a direct import so the
    skill works even when the monolith hasn't loaded it."""
    mod = sys.modules.get("audio.kinect_bridge")
    if mod is not None:
        return mod
    try:
        from audio import kinect_bridge as _kb
        return _kb
    except Exception:
        return None


def _face_tracker():
    """The live face_tracker skill module (registered as skill_face_tracker by
    the loader), or None. We read its gaze snapshot via _snapshot_state() rather
    than recompute gaze ourselves."""
    return sys.modules.get("skill_face_tracker")


def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence (early boot
    / standalone test). Read fresh each call so a Settings toggle takes effect
    without a restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


# ─── identity (SOFT hook into the face_id skill — never a hard dependency) ─

def _face_id_skill():
    """The live face_id skill module (registered as skill_face_id by the
    loader), or None. We reach it lazily and never hard-import the engine, so
    camera_system keeps working unchanged when face-ID is absent or off."""
    return sys.modules.get("skill_face_id")


def _identity_read() -> dict:
    """Best-effort identity for the person at the desk, used ONLY to enrich the
    spoken 'who's here' line. Returns:
        {"on": bool,            # face-ID enabled AND a recognition ran
         "owner": bool,         # the owner's face was recognised
         "others": list[str],   # other enrolled names recognised
         "unknown": int}        # count of faces matched to no one

    Fires ONLY when FACE_ID_ENABLED is on; otherwise (or on ANY failure /
    missing piece) returns {"on": False, ...} so the caller leaves its existing
    behaviour untouched. NEVER raises — identity is a bonus, never a blocker."""
    blank = {"on": False, "owner": False, "others": [], "unknown": 0}
    if not _cfg_flag("FACE_ID_ENABLED"):
        return blank
    # Prefer the skill's own recognizer (it owns the webcam frame-grab + the
    # owner-name mapping); fall back to the engine directly if the skill module
    # isn't loaded but the engine is.
    skill = _face_id_skill()
    eng = sys.modules.get("audio.face_id")
    if eng is None:
        try:
            from audio import face_id as eng  # type: ignore
        except Exception:
            eng = None
    if eng is None:
        return blank
    try:
        ok, _reason = eng.is_available()
    except Exception:
        ok = False
    if not ok:
        return blank
    # Grab the primary webcam frame via the skill's helper (so the index +
    # locking logic stay in one place); fall back to our own _collect_frames.
    frame = None
    if skill is not None:
        grab = getattr(skill, "_grab_frame", None)
        pidx = getattr(skill, "_primary_index", None)
        if callable(grab) and callable(pidx):
            try:
                frame = grab(pidx())
            except Exception:
                frame = None
    if frame is None:
        return blank
    try:
        results = eng.recognize(frame)
    except Exception:
        return blank
    if not results:
        return blank
    is_owner = getattr(skill, "_is_owner_label", None) if skill is not None else None
    owner = False
    others: list[str] = []
    unknown = 0
    for r in results:
        nm = r.get("name")
        if nm in (None, "unknown"):
            unknown += 1
            continue
        if callable(is_owner) and is_owner(nm):
            owner = True
        else:
            others.append(nm)
    return {"on": True, "owner": owner, "others": others, "unknown": unknown}


def _ask_vision():
    """Resolve the vision callable: prefer the injected skill_utils seam, fall
    back to the monolith's ask_vision. Returns a callable or None."""
    su = globals().get("skill_utils")
    if isinstance(su, dict):
        fn = su.get("ask_vision")
        if callable(fn):
            return fn
    bc = _bc()
    fn = getattr(bc, "ask_vision", None) if bc is not None else None
    return fn if callable(fn) else None


# How recently a webcam must have delivered a frame to count as "live". Matches
# the see_user staleness note (>5s old gets flagged) so the two agree.
_WEBCAM_LIVE_SECONDS = 5.0


# ─── webcam health (reads the monolith's shared frame caches) ────────────

def _webcam_health() -> list[dict]:
    """One entry per configured webcam (core.config.CAMERAS), each:
        {"index": int, "label": str, "primary": bool, "side": "left"|"right",
         "live": bool, "age": float|None, "seen_face": bool,
         "last_error": str|None}
    `live` is True when the face-track thread has a frame newer than
    _WEBCAM_LIVE_SECONDS for that index. Reads under _camera_state_lock so it
    never races the writer. Returns [] when the monolith / config isn't loaded.
    NEVER raises."""
    out: list[dict] = []
    try:
        from core.config import CAMERAS
    except Exception:
        CAMERAS = []
    bc = _bc()
    if bc is None or not CAMERAS:
        return out
    now = time.time()
    lock = getattr(bc, "_camera_state_lock", None)
    frame_at = getattr(bc, "_camera_last_frame_at", {}) or {}
    seen_at = getattr(bc, "_camera_last_seen", {}) or {}
    errors = getattr(bc, "_camera_last_read_error", {}) or {}

    def _read_snapshot():
        for cam in CAMERAS:
            idx = cam.get("index")
            last_at = frame_at.get(idx, 0.0) or 0.0
            age = (now - last_at) if last_at else None
            face_at = seen_at.get(idx, 0.0) or 0.0
            out.append({
                "index": idx,
                "label": cam.get("label", f"camera {idx}"),
                "primary": bool(cam.get("primary")),
                "side": ("left" if cam.get("look_x", 0.5) < 0.5 else "right"),
                "live": bool(last_at and age is not None
                             and age <= _WEBCAM_LIVE_SECONDS),
                "age": (round(age, 1) if age is not None else None),
                "seen_face": bool(face_at and (now - face_at) <= _WEBCAM_LIVE_SECONDS),
                "last_error": errors.get(idx),
            })

    try:
        if lock is not None:
            with lock:
                _read_snapshot()
        else:
            _read_snapshot()
    except Exception:
        return out
    return out


def _kinect_health() -> dict:
    """Kinect availability + presence as one dict:
        {"configured": bool,   # KINECT_ENABLED is on (user opted in)
         "available": bool,    # bridge.available() — sensor actually opened
         "reason": str,        # why it's not available (when not)
         "present": bool, "count": int, "nearest_m": float|None,
         "facing": bool|None}
    NEVER raises — every failure degrades to configured/available=False."""
    base = {"configured": False, "available": False, "reason": "",
            "present": False, "count": 0, "nearest_m": None, "facing": None}
    if not _cfg_flag("KINECT_ENABLED"):
        base["reason"] = "disabled (off by default for privacy)"
        return base
    base["configured"] = True
    kb = _kinect_bridge()
    if kb is None:
        base["reason"] = "bridge not loaded (pykinect2 may be missing)"
        return base
    try:
        ok, reason = kb.available()
    except Exception as e:   # pragma: no cover - bridge.available() already swallows
        base["reason"] = f"{type(e).__name__}"
        return base
    if not ok:
        base["reason"] = reason or "sensor unavailable"
        return base
    base["available"] = True
    try:
        presence = kb.get_presence() or {}
    except Exception:   # pragma: no cover - get_presence already swallows
        presence = {}
    base["present"] = bool(presence.get("present"))
    base["count"] = int(presence.get("count", 0) or 0)
    base["nearest_m"] = presence.get("nearest_m")
    base["facing"] = presence.get("facing")
    return base


# ─── gaze (reads the face_tracker skill's snapshot, else the raw timestamps) ─

def _gaze_snapshot() -> dict:
    """Best available gaze reading:
        {"monitor": str|None,   # "left"/"right"/"middle_or_top"/"away"/None
         "dwell_s": float|None, # seconds on the current monitor
         "face_visible": bool,
         "source": "face_tracker" | "timestamps" | "none"}
    Prefers the face_tracker skill's hysteresis-smoothed state; if that skill
    isn't loaded, derives a raw left/right/away read straight from the shared
    per-camera last-seen timestamps (the same signal face_tracker is built on).
    NEVER raises."""
    ft = _face_tracker()
    if ft is not None:
        getter = getattr(ft, "_snapshot_state", None)
        if callable(getter):
            try:
                snap = getter()
                monitor = snap.get("current_monitor")
                since = snap.get("monitor_since") or 0.0
                dwell = (time.time() - since) if (since and monitor
                                                  and monitor != "away") else None
                return {
                    "monitor": monitor,
                    "dwell_s": (round(dwell, 1) if dwell is not None else None),
                    "face_visible": bool(snap.get("face_visible")),
                    "source": "face_tracker",
                }
            except Exception:
                pass
    # Fallback: derive sides directly from the shared timestamps.
    health = _webcam_health()
    if not health:
        return {"monitor": None, "dwell_s": None, "face_visible": False,
                "source": "none"}
    sides = {c["side"] for c in health if c["seen_face"]}
    if not sides:
        monitor = "away"
    elif sides == {"left"}:
        monitor = "left"
    elif sides == {"right"}:
        monitor = "right"
    else:
        monitor = "middle_or_top"
    return {"monitor": monitor, "dwell_s": None,
            "face_visible": bool(sides), "source": "timestamps"}


# ─── the unified fusion ──────────────────────────────────────────────────

def situational_awareness() -> dict:
    """FUSE webcam gaze + Kinect presence into one situational dict:

        {"present": bool,            # is the user / anyone here at all
         "people": int,              # best body count (Kinect, else 1/0 from gaze)
         "distance_m": float|None,   # nearest-body distance (Kinect only)
         "facing_monitor": str|None, # which monitor the gaze is on
         "gaze": str,                # raw gaze label ("left"/"away"/…/"unknown")
         "sources": {                # which signals actually contributed
             "webcams": bool, "kinect": bool, "gaze": str}}

    Presence is the OR of the two: a Kinect body OR a webcam-visible face means
    present. People/distance come from the Kinect when it's live (the only
    sensor that counts bodies + measures range); the monitor + gaze label come
    from the webcams. Pure-ish: reads shared state, mutates nothing."""
    health = _webcam_health()
    kin = _kinect_health()
    gaze = _gaze_snapshot()

    webcams_up = bool(health)
    kinect_up = bool(kin.get("available"))
    face_visible = bool(gaze.get("face_visible"))
    monitor = gaze.get("monitor")
    gaze_label = monitor if monitor else ("unknown" if not webcams_up else None)

    # Presence: either sensor seeing a body/face.
    present = bool(kin.get("present")) or face_visible

    # People: trust the Kinect's body count when it's live; otherwise infer a
    # lone user from a webcam face (we can't count strangers without the Kinect).
    if kinect_up:
        people = int(kin.get("count", 0) or 0)
        if people == 0 and face_visible:
            # Webcam sees a face the skeleton tracker hasn't locked yet.
            people = 1
    else:
        people = 1 if face_visible else 0

    facing_monitor = monitor if (monitor and monitor != "away") else None

    return {
        "present": present,
        "people": people,
        "distance_m": (kin.get("nearest_m") if kinect_up else None),
        "facing_monitor": facing_monitor,
        "gaze": (gaze_label if gaze_label is not None else "unknown"),
        "sources": {
            "webcams": webcams_up,
            "kinect": kinect_up,
            "gaze": gaze.get("source", "none"),
        },
    }


# ─── phrasing helpers ────────────────────────────────────────────────────

def _monitor_word(name) -> str:
    if name == "middle_or_top":
        return "the middle or top monitor"
    if name in ("left", "right", "middle", "top"):
        return f"the {name} monitor"
    return "a monitor"


def _people_word(n: int) -> str:
    if n <= 1:
        return "one person"
    return f"{n} people"


# ─── action 1: camera_status ─────────────────────────────────────────────

def camera_status(_: str = "") -> str:
    """Enumerate EVERY camera (both webcams + the Kinect) and report each one's
    role + live health. Spoken-friendly, honest about anything dark/stale."""
    health = _webcam_health()
    kin = _kinect_health()

    if not health and not kin.get("configured"):
        return ("I can't reach the camera system right now, sir — the tracker "
                "may not have started yet.")

    # Webcam clauses.
    live = [c for c in health if c["live"]]
    dark = [c for c in health if not c["live"]]
    cam_total = len(health) + (1 if kin.get("configured") else 0)

    parts: list[str] = []
    if health:
        if not dark:
            # All webcams live.
            if len(live) == 1:
                parts.append("the " + live[0]["side"] + " monitor webcam is live")
            else:
                parts.append(
                    "the " + " and ".join(c["side"] for c in live)
                    + " monitor webcams are both live")
        elif not live:
            parts.append(
                "both monitor webcams are dark"
                if len(dark) > 1 else
                f"the {dark[0]['side']} monitor webcam is dark")
        else:
            parts.append(
                "the " + " and ".join(c["side"] for c in live)
                + (" webcam is live" if len(live) == 1 else " webcams are live"))
            parts.append(
                "the " + " and ".join(c["side"] for c in dark)
                + (" one is dark" if len(dark) == 1 else " ones are dark"))

    # Kinect clause.
    if kin.get("configured"):
        if kin.get("available"):
            count = kin.get("count", 0)
            if count <= 0:
                parts.append("the Kinect is up — no one in its view")
            else:
                near = kin.get("nearest_m")
                from core.units import meters_to_imperial_phrase
                near_txt = f" about {meters_to_imperial_phrase(near)} out" if near else ""
                parts.append(f"the Kinect sees {_people_word(count)}{near_txt}")
        else:
            parts.append(f"the Kinect is dark ({kin.get('reason')})")
    else:
        # Kinect off entirely — mention it so the count is honest.
        parts.append("the Kinect is off (disabled by default for privacy)")
        cam_total = len(health)   # don't count an off sensor in the headline

    headline = {0: "No cameras", 1: "One camera", 2: "Two cameras",
                3: "Three cameras"}.get(cam_total, f"{cam_total} cameras")
    body = "; ".join(parts) if parts else "none reporting"
    return f"{headline}, sir: {body}."


# ─── action 2: situational awareness ─────────────────────────────────────

def where_am_i(_: str = "") -> str:
    """Speak a coherent one-liner fusing gaze + Kinect presence, degrading
    gracefully and naming which sources it used."""
    s = situational_awareness()
    src = s["sources"]
    webcams, kinect = src["webcams"], src["kinect"]

    # Nothing to go on at all.
    if not webcams and not kinect:
        return ("I have no camera signal at the moment, sir — neither the "
                "webcams nor the Kinect are reporting.")

    # Build the descriptive clauses.
    clauses: list[str] = []

    if s["present"]:
        clauses.append("You're at the desk")
    else:
        # Present is false: say so plainly, then note what looked.
        looked = _sources_phrase(webcams, kinect, src["gaze"])
        return f"I don't see you right now, sir — {looked}."

    dist = s["distance_m"]
    if dist:
        from core.units import meters_to_imperial_phrase
        clauses.append(f"about {meters_to_imperial_phrase(dist)} back")

    if s["facing_monitor"]:
        clauses.append(f"facing {_monitor_word(s['facing_monitor'])}")
    elif s["gaze"] == "away" and kinect and s["present"]:
        # Kinect sees a body but the webcams don't see your face.
        clauses.append("turned away from the monitor webcams")

    # Company. When face-ID is on and recognised someone, NAME the identity
    # alongside the Kinect's body count ("you, alone" / "you, plus one person I
    # don't recognise"). This is a soft enrichment: if face-ID is off or unsure,
    # the existing Kinect-count clause is used unchanged.
    people = s["people"]
    ident = _identity_read()
    company = _company_clause(people, kinect, ident)
    if company:
        clauses.append(company)

    looked = _sources_phrase(webcams, kinect, src["gaze"])
    sentence = ", ".join(clauses)
    return f"{sentence}, sir. {looked}."


def _company_clause(people: int, kinect: bool, ident: dict) -> str:
    """The 'who's here' clause that gets comma-joined into the where_am_i line
    (the sentence already opens with 'You're at the desk').

    When identity is available (ident['on'] and it recognised someone) it names
    any OTHER enrolled people and cross-checks the Kinect body count to report
    unrecognised company: 'alone' / 'with Dana' / 'with one person I don't
    recognise' / 'with Dana and 2 people I don't recognise'. When face-ID is off
    or unsure, it returns the ORIGINAL Kinect-count phrasing — and '' when the
    Kinect isn't live — so the line is byte-for-byte unchanged without face-ID."""
    if ident.get("on") and (ident.get("owner") or ident.get("others")
                            or ident.get("unknown")):
        names = list(ident.get("others", []))
        # How many bodies are NOT accounted for by the faces we named. Prefer
        # the Kinect's body count when it's live (it sees bodies the face camera
        # can't); else fall back to face-ID's own unrecognised-face count.
        named = (1 if ident.get("owner") else 0) + len(names)
        if kinect and people > 0:
            extra = max(people - named, int(ident.get("unknown", 0)))
        else:
            extra = int(ident.get("unknown", 0))
        pieces: list[str] = []
        if names:
            pieces.append(_join_names(names))
        if extra > 0:
            pieces.append(_unknown_phrase(extra))
        if not pieces:
            # Just the owner (or just confirmed-no-company): alone.
            return "alone"
        return "with " + _join_names(pieces)
    # No identity — original Kinect-count behaviour, untouched.
    if kinect:
        if people <= 1:
            return "alone"
        return (f"with {people - 1} other "
                + ("person" if people - 1 == 1 else "people"))
    return ""


def _join_names(names: list) -> str:
    """'a' / 'a and b' / 'a, b and c'."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _unknown_phrase(n: int) -> str:
    """'one person I don't recognise' / 'N people I don't recognise'."""
    if n <= 1:
        return "one person I don't recognise"
    return f"{n} people I don't recognise"


def _sources_phrase(webcams: bool, kinect: bool, gaze_src: str) -> str:
    """A short honest clause naming which sensors produced the answer."""
    used = []
    if webcams:
        used.append("the monitor webcams")
    if kinect:
        used.append("the Kinect")
    if not used:
        return "no sensors are live"
    joined = " and ".join(used)
    return f"(from {joined})"


# ─── action 3: look_around — unified multi-camera vision sweep ───────────

def _encode_bgr_to_png(frame):
    """BGR ndarray → PNG bytes via cv2, or None. NEVER raises."""
    try:
        import cv2
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            return None
        return bytes(buf.tobytes())
    except Exception:
        return None


def _collect_frames() -> list[tuple[str, bytes]]:
    """Grab one PNG frame from EACH available source. Returns a list of
    (vantage_label, png_bytes). Webcams come from the shared _camera_latest_frame
    cache (encoded here); the Kinect via the bridge's get_color_png(). Missing /
    un-encodable frames are skipped. NEVER raises."""
    frames: list[tuple[str, bytes]] = []

    # Webcams: copy each cached frame under the lock, encode outside it.
    bc = _bc()
    raw: list[tuple[str, object]] = []
    if bc is not None:
        try:
            from core.config import CAMERAS
        except Exception:
            CAMERAS = []
        lock = getattr(bc, "_camera_state_lock", None)
        latest = getattr(bc, "_camera_latest_frame", {}) or {}
        try:
            if lock is not None:
                with lock:
                    for cam in CAMERAS:
                        idx = cam.get("index")
                        fr = latest.get(idx)
                        if fr is not None:
                            side = ("left" if cam.get("look_x", 0.5) < 0.5
                                    else "right")
                            raw.append((f"the {side} monitor webcam", fr.copy()))
            else:
                for cam in CAMERAS:
                    idx = cam.get("index")
                    fr = latest.get(idx)
                    if fr is not None:
                        side = ("left" if cam.get("look_x", 0.5) < 0.5 else "right")
                        raw.append((f"the {side} monitor webcam", fr))
        except Exception:
            raw = []
    for label, fr in raw:
        png = _encode_bgr_to_png(fr)
        if png is not None:
            frames.append((label, png))

    # Kinect: only when it's enabled AND streaming.
    kin = _kinect_health()
    if kin.get("available"):
        kb = _kinect_bridge()
        if kb is not None:
            try:
                png = kb.get_color_png()
            except Exception:
                png = None
            if png is not None:
                frames.append(("the Kinect (room view)", png))
    return frames


# The per-frame question. Kept short so the local VLM stays fast.
_LOOK_PROMPT = ("Briefly describe what is visible from this camera — the person "
                "if present (what they're doing) and anything notable in the "
                "scene. One or two sentences.")


def look_around(_: str = "") -> str:
    """Capture a frame from EVERY available camera and describe the whole scene
    in one short spoken paragraph that names each vantage.

    COST-CONSCIOUS: this sweep can fire several vision calls, so it prefers the
    free local VLM. When vision is routed 'local' (or the cloud backend is off)
    every frame goes to qwen2.5vl at $0; we detect that up front and tell the
    user. When the cloud route is active we still answer (ask_vision falls back
    to local on any cloud failure) but we note that the cloud eye was used so
    the cost is never silent. NEVER raises into the voice loop."""
    frames = _collect_frames()
    if not frames:
        # Distinguish "no cameras at all" from "cameras up but no frame yet".
        health = _webcam_health()
        kin = _kinect_health()
        if not health and not kin.get("configured"):
            return ("I have no cameras to look through right now, sir.")
        return ("My cameras are up but none handed me a frame just now, sir — "
                "give them a moment and ask again.")

    ask = _ask_vision()
    if not callable(ask):
        names = ", ".join(label for label, _ in frames)
        return (f"I have frames from {names}, sir, but vision isn't wired up "
                f"to describe them.")

    # Cost note: are we on the free local route?
    try:
        from core.config import model_route
        route = model_route("vision")
    except Exception:
        route = "auto"
    forced_local = (route == "local") or not _cfg_flag_cloud_backend()

    described: list[str] = []
    used_cloud = False
    answered_local = 0   # how many vantages the FREE local VLM actually answered
    for label, png in frames:
        try:
            answer = ask(_LOOK_PROMPT, png)
        except Exception:
            answer = ""
        answer = (answer or "").strip()
        if not answer:
            described.append(f"From {label}, I couldn't make anything out")
            continue
        # ask_vision prefixes local-VLM answers with "[local-vision] " — the
        # ground truth of which eye answered, regardless of the configured route.
        if answer.startswith("[local-vision]"):
            answer = answer[len("[local-vision]"):].strip()
            answered_local += 1
        elif not forced_local:
            used_cloud = True
        described.append(f"From {label}: {answer}")

    if not described:
        return "I looked through every camera, sir, but couldn't make any out."

    paragraph = " ".join(
        d if d.endswith((".", "!", "?")) else d + "."
        for d in described)
    n = len(frames)
    lead = (f"Looking around — {n} "
            + ("camera" if n == 1 else "cameras") + ", sir. ")
    tail = ""
    if used_cloud:
        tail = " (Used the cloud vision model for that sweep.)"
    elif forced_local or answered_local == len(described):
        # Either we forced local up front, or every vantage was answered by the
        # free local VLM — say so honestly so the cost (none) isn't a mystery.
        tail = " (Done locally, no cost.)"
    return f"{lead}{paragraph}{tail}"


def _cfg_flag_cloud_backend() -> bool:
    """True when the cloud (Claude) vision backend is actually usable. When it
    isn't, ask_vision routes every frame to the local VLM regardless of the
    per-function route, so look_around's cost note should say 'local'."""
    try:
        from core import config as _cfg
        return getattr(_cfg, "AI_BACKEND", "claude") == "claude"
    except Exception:
        return True


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    actions["camera_status"]          = camera_status
    actions["situational_awareness"]  = where_am_i
    actions["where_am_i"]             = where_am_i
    actions["look_around"]            = look_around
    print("  [camera-system] unified multi-camera actions registered "
          "(camera_status, situational_awareness/where_am_i, look_around)")
