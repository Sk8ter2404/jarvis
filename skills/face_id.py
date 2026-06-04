"""
Face-recognition voice skill for JARVIS — WHO is at the desk.

JARVIS's two monitor webcams sit right at the screens, closest to your face, so
they are the identity cameras. This skill pairs that identity with the Kinect's
body COUNT (via the existing situational_awareness fusion): not just "one person
in the room" but "it's you, sir". It is a thin voice wrapper over the engine in
audio/face_id.py — the engine owns the OpenCV detect→embed→match pipeline and
the (gitignored, biometric-PII) enrollment store; this skill owns the phrasing,
the webcam frame-grab, and the gates.

Voice actions
-------------
  enroll_face   ("learn my face" / "remember my face" / "this is me")
                — grab ~5 frames from the primary webcam over ~2 s and enroll
                  them as you (USER_NAME, or "owner" if that's blank).
  whoami / recognize_face
                ("who am I" / "do you recognize me" / "who's at the desk")
                — recognise faces in the current primary-webcam frame and say
                  whether it's you / someone unrecognised / no face.
  face_id_status ("is face recognition on" / "face id status")
                — enabled? models present? who's enrolled? camera available?
  forget_face <name>            — delete one person's face enrollment.
  list_enrolled_faces           — report who is enrolled.

Privacy + gates
---------------
* OFF by default. Every action refuses honestly unless core.config
  FACE_ID_ENABLED is True. Face biometrics are opt-in.
* No camera or model work in staging/test (mirrors the other camera skills'
  _is_staging gate) — the engine there would otherwise try to download models.
* Graceful everywhere: models missing, cv2 absent, camera dark, nobody enrolled
  → an honest spoken line, never an exception into the voice loop.
"""

from __future__ import annotations

import os
import sys
import time


# ─── seams to the rest of JARVIS ─────────────────────────────────────────

def _bc():
    """The live monolith module (main or by-name), or None."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _engine():
    """The audio.face_id engine module, or None. Prefer the instance the
    monolith already imported; fall back to a direct import so the skill works
    standalone (and in tests)."""
    mod = sys.modules.get("audio.face_id")
    if mod is not None:
        return mod
    try:
        from audio import face_id as _fi
        return _fi
    except Exception:
        return None


def _cfg_flag(name: str, default: bool = False) -> bool:
    """Read a live boolean from core.config, tolerating its absence. Read fresh
    each call so a Settings toggle takes effect without a restart."""
    try:
        from core import config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _is_staging() -> bool:
    """True on the staging/test instance — never touch a camera or download a
    model there. Matches the monolith's own gate plus the raw env var so the
    check holds even before the monolith is importable (mirrors the Kinect
    skills)."""
    if os.environ.get("JARVIS_STAGING", "").strip() == "1":
        return True
    bc = _bc()
    if bc is not None:
        fn = getattr(bc, "_is_staging", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
    return False


def _owner_name() -> str:
    """The configured user's name, or 'owner' when it's blank (USER_NAME may be
    unset). Used as the enrollment label and in spoken replies."""
    try:
        from core import config as _cfg
        name = getattr(_cfg, "USER_NAME", "") or ""
        name = name.strip()
        if name:
            return name
    except Exception:
        pass
    return "owner"


def _is_owner_label(name: str) -> bool:
    """True if a recognised name is the owner (their configured name OR the
    'owner' fallback label) — so whoami can say 'that's you'."""
    n = (name or "").strip().lower()
    return bool(n) and n in {_owner_name().strip().lower(), "owner"}


# ─── webcam frame-grab (reads the monolith's shared frame cache) ─────────

def _primary_index() -> int:
    """The configured primary webcam index (the one at the screen, closest to
    the face). Falls back to 0."""
    try:
        from core.config import CAMERAS
        for cam in CAMERAS:
            if cam.get("primary"):
                return int(cam.get("index", 0))
        if CAMERAS:
            return int(CAMERAS[0].get("index", 0))
    except Exception:
        pass
    return 0


def _grab_frame(index: int):
    """A copy of the most recent BGR frame for `index` from the monolith's
    shared _camera_latest_frame cache (copied under _camera_state_lock), or
    None. NEVER raises."""
    bc = _bc()
    if bc is None:
        return None
    lock = getattr(bc, "_camera_state_lock", None)
    latest = getattr(bc, "_camera_latest_frame", None)
    if latest is None:
        return None
    try:
        if lock is not None:
            with lock:
                fr = latest.get(index)
                return fr.copy() if fr is not None else None
        fr = latest.get(index)
        return fr.copy() if fr is not None else None
    except Exception:
        return None


def _grab_frames(index: int, n: int = 5, gap_s: float = 0.4) -> list:
    """Collect up to `n` webcam frames over ~n*gap_s seconds (small waits let a
    new frame arrive between grabs). De-dupes nothing — the engine picks the
    largest face per frame. NEVER raises."""
    out = []
    for i in range(max(1, n)):
        fr = _grab_frame(index)
        if fr is not None:
            out.append(fr)
        if i < n - 1:
            try:
                time.sleep(max(0.0, gap_s))
            except Exception:
                pass
    return out


def _camera_available(index: int) -> bool:
    """True if the primary webcam handed us a frame just now."""
    return _grab_frame(index) is not None


# ─── shared gate ──────────────────────────────────────────────────────────

def _refuse_if_off() -> str | None:
    """Return an honest refusal string if face-ID can't run (disabled / staging
    / models or cv2 absent), else None when it's safe to proceed. Centralises
    the gate so every action refuses identically."""
    if _is_staging():
        return ("Face recognition is parked on the staging instance, sir — "
                "I only run it on the live one.")
    if not _cfg_flag("FACE_ID_ENABLED"):
        return ("Face recognition is off, sir. It's opt-in for privacy — enable "
                "FACE_ID_ENABLED in settings and I'll start recognising faces.")
    eng = _engine()
    if eng is None:
        return ("I can't reach the face-recognition engine right now, sir.")
    ok, reason = eng.is_available()
    if not ok:
        return (f"Face recognition isn't ready, sir — {reason}.")
    return None


# ─── action: enroll_face ───────────────────────────────────────────────────

def enroll_face(arg: str = "") -> str:
    """Learn the owner's face from a short burst of primary-webcam frames."""
    gate = _refuse_if_off()
    if gate is not None:
        return gate
    idx = _primary_index()
    if not _camera_available(idx):
        return ("I can't see through the webcam right now, sir — make sure it's "
                "free and face me, then try again.")
    eng = _engine()
    name = _owner_name()
    frames = _grab_frames(idx, n=5, gap_s=0.4)
    if not frames:
        return ("I couldn't capture the webcam, sir — give it a moment and try "
                "again.")
    try:
        n = eng.enroll(name, frames)
    except Exception:   # pragma: no cover - engine swallows; belt-and-braces
        n = 0
    if n <= 0:
        return ("I couldn't get a clear look at your face, sir — face me square "
                "on and try again.")
    who = "you" if name == "owner" else name
    return (f"Got it, sir — I'll recognise {who} now. "
            f"(Captured {n} good {'view' if n == 1 else 'views'}.)")


# ─── action: whoami / recognize_face ───────────────────────────────────────

def whoami(arg: str = "") -> str:
    """Recognise whoever is in the current primary-webcam frame."""
    gate = _refuse_if_off()
    if gate is not None:
        return gate
    idx = _primary_index()
    frame = _grab_frame(idx)
    if frame is None:
        return ("I can't see through the webcam right now, sir.")
    eng = _engine()
    try:
        results = eng.recognize(frame)
    except Exception:   # pragma: no cover - engine swallows; belt-and-braces
        results = []
    if not results:
        return "I don't see a face right now, sir."

    named = [r for r in results if r.get("name") not in (None, "unknown")]
    owner_seen = any(_is_owner_label(r.get("name", "")) for r in named)
    others = [r.get("name") for r in named if not _is_owner_label(r.get("name", ""))]
    unknown_n = sum(1 for r in results if r.get("name") in (None, "unknown"))

    if owner_seen and not others and unknown_n == 0:
        return "That's you, sir."
    if not named and unknown_n:
        if unknown_n == 1:
            return "I see someone I don't recognise, sir."
        return f"I see {unknown_n} people I don't recognise, sir."

    # Mixed / other people enrolled.
    parts = []
    if owner_seen:
        parts.append("you")
    # De-dup other names, keep order.
    seen = set()
    for nm in others:
        key = (nm or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            parts.append(nm)
    if unknown_n:
        parts.append(f"{unknown_n} I don't recognise"
                     if unknown_n > 1 else "someone I don't recognise")
    if not parts:
        return "I don't see a face right now, sir."
    if len(parts) == 1:
        return f"I see {parts[0]}, sir."
    return f"I see {', '.join(parts[:-1])} and {parts[-1]}, sir."


# ─── action: face_id_status ────────────────────────────────────────────────

def face_id_status(arg: str = "") -> str:
    """Report whether face-ID is on, models present, who's enrolled, camera up."""
    enabled = _cfg_flag("FACE_ID_ENABLED")
    if _is_staging():
        return ("Face recognition is disabled on the staging instance, sir — "
                "it only runs on the live one.")
    eng = _engine()
    if eng is None:
        return ("Face recognition is "
                + ("on" if enabled else "off")
                + ", sir, but I can't reach its engine right now.")
    models_ok, reason = eng.is_available()
    try:
        enrolled = eng.list_enrolled()
    except Exception:   # pragma: no cover - engine swallows; belt-and-braces
        enrolled = []
    idx = _primary_index()
    cam_up = _camera_available(idx)

    bits = []
    bits.append("Face recognition is " + ("on" if enabled else "off (opt-in)"))
    if models_ok:
        bits.append("the models are loaded")
    else:
        bits.append(f"the models aren't ready ({reason})")
    if enrolled:
        who = ", ".join(f"{p['name']} ({p['count']} "
                        f"{'view' if p['count'] == 1 else 'views'})"
                        for p in enrolled)
        bits.append(f"enrolled: {who}")
    else:
        bits.append("no one is enrolled yet")
    bits.append("the webcam is live" if cam_up else "the webcam is dark")
    return "Sir — " + "; ".join(bits) + "."


# ─── action: forget_face / list_enrolled_faces ────────────────────────────

def forget_face(arg: str = "") -> str:
    """Delete one person's face enrollment (arg = their name)."""
    if _is_staging():
        return ("Face recognition is parked on the staging instance, sir.")
    eng = _engine()
    if eng is None:
        return "I can't reach the face-recognition engine right now, sir."
    name = (arg or "").strip()
    if not name:
        # Default to forgetting the owner if no name given.
        name = _owner_name()
    try:
        ok = eng.forget(name)
    except Exception:   # pragma: no cover - engine swallows
        ok = False
    who = "your" if _is_owner_label(name) else f"{name}'s"
    if ok:
        return f"Forgotten {who} face, sir."
    return f"I don't have {who} face enrolled, sir."


def list_enrolled_faces(arg: str = "") -> str:
    """Report who is enrolled for face recognition."""
    if _is_staging():
        return ("Face recognition is parked on the staging instance, sir.")
    eng = _engine()
    if eng is None:
        return "I can't reach the face-recognition engine right now, sir."
    try:
        enrolled = eng.list_enrolled()
    except Exception:   # pragma: no cover - engine swallows
        enrolled = []
    if not enrolled:
        return "No faces are enrolled yet, sir."
    who = ", ".join(f"{p['name']} ({p['count']} "
                    f"{'view' if p['count'] == 1 else 'views'})"
                    for p in enrolled)
    return f"Enrolled faces, sir: {who}."


# ─── registration ──────────────────────────────────────────────────────────

def register(actions):
    actions["enroll_face"]          = enroll_face
    actions["learn_my_face"]        = enroll_face
    actions["remember_my_face"]     = enroll_face
    actions["whoami"]               = whoami
    actions["who_am_i"]             = whoami
    actions["recognize_face"]       = whoami
    actions["do_you_recognize_me"]  = whoami
    actions["whos_at_the_desk"]     = whoami
    actions["face_id_status"]       = face_id_status
    actions["forget_face"]          = forget_face
    actions["list_enrolled_faces"]  = list_enrolled_faces
    print("  [face-id] face recognition actions registered "
          "(enroll_face, whoami/recognize_face, face_id_status, forget_face, "
          "list_enrolled_faces)")
