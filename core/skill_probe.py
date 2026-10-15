"""Shared probes for skills: face-tracker presence, Kinect bridge, DPAPI decrypt.

Canonical copies of helpers that were duplicated verbatim across skills.
Modules import them under the same local names so call sites and
`mock.patch.object(mod, "_name")` keep working.
"""
import base64 as _base64
import sys


def _user_at_desk() -> bool | None:
    """Return True if face_tracker has recently seen the user, False if it
    has seen 'away', or None if face_tracker isn't loaded / has no data."""
    mod = sys.modules.get("skill_face_tracker")
    if mod is None:
        return None
    snap_func = getattr(mod, "_snapshot_state", None)
    if snap_func is None:
        return None
    try:
        snap = snap_func()
    except Exception:
        return None
    if not snap.get("last_sample_at"):
        return None
    monitor = snap.get("current_monitor")
    if monitor in ("left", "right", "middle_or_top"):
        return True
    if monitor == "away":
        return False
    return None


def _bridge():
    """Live kinect_bridge module, or None. Prefer the instance the monolith
    already imported; fall back to a direct import (mirrors kinect_gestures)."""
    mod = sys.modules.get("audio.kinect_bridge")
    if mod is not None:
        return mod
    try:
        from audio import kinect_bridge as _kb
        return _kb
    except Exception:
        return None


def _dpapi_decrypt(b64: str) -> str | None:
    """Decrypt DPAPI base64 ciphertext back to plaintext, or None on failure."""
    if not b64:
        return None
    try:
        import win32crypt
        blob = _base64.b64decode(b64.encode("ascii"))
        _desc, data = win32crypt.CryptUnprotectData(blob, None, None, None, 0)
        return data.decode("utf-8")
    except Exception:
        return None
