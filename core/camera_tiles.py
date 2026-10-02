"""Which camera tiles the web dashboard shows, and what the gate says about each.

ONE home for two things that used to be duplicated or hard-coded:

  * the per-camera preview KEY vocabulary ("left" / "right" / "kinect") - the
    monolith's preview writer (bobert_companion._hud_percam_preview_write)
    and the dashboard (tools/web_interface) each carried their OWN copy of the
    same tuple until 2026-09-30, the stale-duplicate shape this codebase keeps
    paying for;
  * the side rule that maps a CAMERAS entry to its key (percam_side - the
    monolith's _percam_side now delegates here).

And two pure helpers the dashboard uses so its Camera tab follows the LIVE
configuration instead of always drawing left/right/kinect:

  * tiles_from_config(CAMERAS, KINECT_ENABLED) - one tile per configured
    webcam (by side) plus the Kinect when it is enabled. A camera the owner
    removed from CAMERAS (because it resets the USB hub) therefore stops
    getting a permanently dark "Webcam off" tile.
  * gate_summary(snapshot, key) - the camera gate's view of one device
    (core/camera_gate.CameraGate.snapshot) as a plain sentence plus an
    optional "retrying in N" countdown, so a tile says WHY JARVIS is not
    opening it instead of looking broken.

Stdlib only, no I/O, and nothing here raises: a malformed CAMERAS entry or an
odd snapshot degrades to "no tile" / "no gate verdict", never an exception
into a request handler.
"""
from __future__ import annotations

__all__ = ["PREVIEW_KEYS", "KINECT_KEY", "percam_side", "tiles_from_config",
           "gate_summary", "fmt_wait"]

KINECT_KEY = "kinect"
# Every per-camera preview file the monolith may write, and so every ?cam=
# value the dashboard accepts: data/.hud_camera_preview_<key>.jpg.
PREVIEW_KEYS = ("left", "right", KINECT_KEY)


def percam_side(cam: dict) -> str:
    """Stable per-camera preview key from a CAMERAS entry. Label first
    ("Left webcam (left monitor)"), look_x fallback with 0.5 counting as left
    (same rule as skills/camera_system — look_x<0.5 misclassified the live
    LEFT cam whose look_x is exactly 0.5)."""
    lbl = str(cam.get("label", "")).lower()
    if "left" in lbl:
        return "left"
    if "right" in lbl:
        return "right"
    try:
        return "left" if float(cam.get("look_x", 0.5)) <= 0.5 else "right"
    except (TypeError, ValueError):
        return "left"


def _is_kinect_entry(cam: dict) -> bool:
    return str(cam.get("type", "")).strip().lower() == KINECT_KEY


def tiles_from_config(cameras, kinect_enabled, gate_key=None) -> list:
    """The tiles to draw: ``[{"cam", "label", "kind", "gate_key"}]``.

    One tile per configured webcam, keyed by percam_side (the FIRST entry wins
    a side, matching the writer, which keys files by side), then the Kinect
    when ``kinect_enabled`` is truthy or CAMERAS itself carries a Kinect
    entry. ``gate_key(cam) -> str`` names each device for the camera gate
    (the monolith's _camera_gate_key); without it webcams get no gate key and
    the Kinect gets "kinect" (the key the bridge uses). Never raises."""
    tiles = []
    seen = set()
    kinect_in_cameras = False
    try:
        entries = list(cameras or [])
    except Exception:
        entries = []
    for cam in entries:
        if not isinstance(cam, dict):
            continue
        try:
            if _is_kinect_entry(cam):
                kinect_in_cameras = True
                continue
            side = percam_side(cam)
            if side in seen:
                continue
            seen.add(side)
            gk = ""
            if callable(gate_key):
                try:
                    gk = str(gate_key(cam) or "")
                except Exception:
                    gk = ""
            tiles.append({"cam": side,
                          "label": str(cam.get("label") or (side + " webcam")),
                          "kind": "webcam", "gate_key": gk})
        except Exception:
            continue
    if kinect_enabled or kinect_in_cameras:
        tiles.append({"cam": KINECT_KEY, "label": "Kinect (skeleton)",
                      "kind": "kinect", "gate_key": KINECT_KEY})
    return tiles


def fmt_wait(seconds) -> str:
    """'45 s' / '3 min' / '1 h 5 min' for a countdown shown on a tile."""
    try:
        s = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return ""
    if s < 90.0:
        return "%d s" % int(round(s))
    m = int(round(s / 60.0))
    if m < 60:
        return "%d min" % m
    h, m = divmod(m, 60)
    return ("%d h %d min" % (h, m)) if m else ("%d h" % h)


def gate_summary(snapshot, key) -> "dict | None":
    """What the camera gate says about device ``key``, or None when the gate
    is not holding it (or there is no gate / no snapshot).

    ``{"state", "message", "retry_in_s"}``: ``retry_in_s`` is a float when
    the gate will ask again on a timer, None when it waits on an EVENT (the
    device coming back, the other app letting go, the owner lifting a
    quarantine). Ordered from the most to the least decisive reason, the same
    order CameraGate.begin refuses in. Never raises."""
    try:
        if not isinstance(snapshot, dict) or not key:
            return None
        quarantined = snapshot.get("quarantined") or {}
        if key in quarantined:
            return {"state": "quarantined", "retry_in_s": None,
                    "message": ("Benched for this session: starting this "
                                "camera kept resetting the USB hub. Move it "
                                "to another port, then ask JARVIS to use it "
                                "again.")}
        dev = (snapshot.get("devices") or {}).get(key) or {}
        if snapshot.get("storm_active"):
            rem = float(snapshot.get("storm_remaining_s") or 0.0)
            return {"state": "usb_storm", "retry_in_s": rem,
                    "message": ("USB storm cool-down: JARVIS paused every "
                                "camera open - retrying in %s." % fmt_wait(rem))}
        hold = float(dev.get("hold_s") or 0.0)
        slow = float(dev.get("slow_retry_s") or 0.0)
        if slow > 0.0 and hold > 0.0:
            # Worded by what its last death SHOWED (core/camera_gate.py
            # dies_on_open_finding, 2026-10-02): "dropped off USB" and the
            # power check only when it was seen leaving the bus.
            off_bus = dev.get("slow_retry_off_bus")
            waits = (fmt_wait(slow), fmt_wait(hold))
            if off_bus is True:
                msg = ("It dropped off USB a few seconds after each start, "
                       "so JARVIS only retries it every %s - next try in %s. "
                       "Check its power supply." % waits)
            elif off_bus is False:
                msg = ("Its stream died a few seconds after each start "
                       "though it stayed connected, so JARVIS only retries "
                       "it every %s - next try in %s." % waits)
            else:
                msg = ("Its stream died within seconds of each start, so "
                       "JARVIS only retries it every %s - next try in %s. If "
                       "it is dropping off USB, check its power supply."
                       % waits)
            return {"state": "slow_retry", "retry_in_s": hold,
                    "message": msg}
        if dev.get("absent"):
            return {"state": "absent", "retry_in_s": None,
                    "message": ("It vanished from the device list (a USB "
                                "reset) - waiting for it to come back.")}
        lockers = [str(x) for x in (dev.get("locked_by") or []) if x]
        if lockers:
            return {"state": "locked", "retry_in_s": None,
                    "message": ("Another app is using a webcam (%s) - JARVIS "
                                "retries when it lets go."
                                % ", ".join(lockers[:3]))}
        if hold > 0.0:
            return {"state": "backoff", "retry_in_s": hold,
                    "message": ("Its last open failed - retrying in %s."
                                % fmt_wait(hold))}
        if dev.get("in_flight"):
            return {"state": "opening", "retry_in_s": None,
                    "message": "JARVIS is opening it right now."}
    except Exception:
        return None
    return None
