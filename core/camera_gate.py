"""ONE gate in front of every camera and Kinect open in this process.

────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS (measured live 2026-09-29)
────────────────────────────────────────────────────────────────────────────
The owner's webcams, the Kinect, a wireless-headset dongle, a desk mic, a
stream-deck controller and his keyboard all sit behind ONE chain of three
cascaded powered USB hubs on one PC port. Windows logged the top hub of that
chain "surprise removed ... Count of devices removed: 20" (Kernel-PnP event
1010) about once a minute for half an hour, and the resets lined up with
JARVIS and nothing else:

  * a burst just after every boot / restart;
  * ~once a minute while JARVIS ran with its cameras recovering;
  * ZERO resets for the 32 minutes the camera-preview producer sat wedged
    ("opening camera index 2") - i.e. while nothing in JARVIS opened a camera;
  * ZERO resets in the 11+ minutes after JARVIS was shut down.

So the camera OPENS were the load, and the thing that turned one bad moment
into a storm was a feedback loop: a hub reset drops every camera at once, every
camera's recovery path reopens it within seconds, the reopens hit a hub that is
still re-enumerating, and it resets again. Before this module there were at
least five openers (the face-track producer's initial open, its recovery
reopen and its soft wake; the side-tile compositor; the boot probes; the
self-diagnostic scan and wake; the Kinect bridge's stale-stream reopen), each
with its OWN spacing rule - 2 s here, 30 s there, per-frame in one place - and
none of them knew what the others were doing. That is this codebase's #1 bug
shape (the stale duplicate) applied to a physical bus.

────────────────────────────────────────────────────────────────────────────
WHAT THE GATE DOES
────────────────────────────────────────────────────────────────────────────
Every opener asks :meth:`CameraGate.begin` before it touches a device and
reports back through :meth:`CameraGate.end`. The gate refuses when, in order:

  usb-storm  the circuit breaker is open: >=2 cameras (or a camera and an
             audio endpoint) dropped within ~10 s, or >=3 open failures across
             >=2 devices within 60 s. ALL camera and Kinect opens stop for a
             cool-down (10 min, doubling on a repeat within the hour, capped at
             60 min). Streams that are already running are left alone.
  wedged     an open (of ANY device) is still stuck inside the camera driver.
             Nothing new starts until it returns; other devices are released
             after WEDGE_HOLD_OTHERS_MAX_S so a call that never returns cannot
             blind every camera for the session.
  absent     the device has VANISHED from the device list (a hub reset). Not
             retried on a timer: every ask polls presence, and
  settling   once it is listed again it must stay listed ABSENT_SETTLE_S
             before it is opened - a stream START into a hub that has just
             re-enumerated is what tripped the resets.
  held       another component is STREAMING this device right now (the Kinect
             bridge's runtime). A second in-process handle on a live device is
             what knocked the owner's cameras over for a day in 2026-09 - see
             core/camera_backend.py.
  in-flight  another component is in the middle of opening this device.
  locked     the last open failed the way a held device fails AND Windows'
             camera privacy log shows another app using a webcam right now.
             Not retried on a timer: every ask re-reads that log and the device
             is retried the moment the app stops, or after LOCKED_RETRY_S at
             the latest. ("A meeting app is RUNNING" is not evidence: that
             heuristic was wrong 23 of 23 times on 2026-09-29.)
  backoff    this device failed an open or needed a read-failure recovery:
             30 s -> 60 -> 120 -> 300 -> 600 s (cap = max_backoff_s). Reset
             only after HEALTHY_RESET_S of sustained healthy frames.
  min-gap    a DIFFERENT component opened this device less than min_gap_s ago.
             (A component's own retries are governed by its backoff, not this.)
  stagger    a DIFFERENT configured device was opened less than OPEN_STAGGER_S
             ago - so at boot, and after a cool-down, the cameras and the Kinect
             are never opened in the same second.

Only the last three are worth WAITING for (see TIMING_REASONS); the rest mean
"not now - ask again at wait_s".

WHAT IT DELIBERATELY DOES NOT DO: it never closes a stream, never reads a
frame, never touches a device. It is pure bookkeeping under one lock, with an
injectable clock, so every rule here is unit-testable on a runner with no
camera, no cv2 and no Windows (tests/test_camera_gate.py). The callers own the
device I/O and their existing wedge / retire / quarantine logic, unchanged.

NEVER RAISES from any public method: a bookkeeping fault must not be able to
stop a camera from opening or a producer from publishing. On an internal error
begin() ALLOWS (fail-open), because the alternative - a bug that silently
blinds every camera for the session - is the worse failure for the owner, and
the device-side bounds (timeouts, quarantine) still stand behind it.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, NamedTuple

__all__ = [
    "BACKOFF_STEPS_S", "Decision", "CameraGate", "TIMING_REASONS",
    "HOLD_REASONS", "KIND_CAMERA", "KIND_AUDIO", "minutes_phrase",
]

# ── the numbers (the three owner knobs are in core/config.py) ──────────────
# Per-device reopen backoff after a failed open or a read-failure recovery.
# The cap is CAMERA_REOPEN_MAX_BACKOFF_S; steps beyond the table keep doubling
# up to that cap, so raising the cap is honoured rather than silently clipped.
BACKOFF_STEPS_S = (30.0, 60.0, 120.0, 300.0, 600.0)
# A device must deliver frames CONTINUOUSLY this long before its backoff level
# resets. A camera that opens, streams for 5 s and dies again must not earn a
# fresh 30 s ladder every time - that would be the storm again, slower.
HEALTHY_RESET_S = 60.0
# A device whose open failed while a known webcam-locking app ran: retried
# when that app is gone (polled via the existing cached locker lookup), else at
# most this often.
LOCKED_RETRY_S = 600.0
# How often an ask re-polls the locker list for a locked device. The lookup is
# the monolith's _camera_lockers_cached (5 s TTL over a psutil walk), so this
# costs a dict read almost every time and never touches a device.
LOCKED_POLL_S = 5.0
# Spacing between opens of DIFFERENT configured devices ("don't open all the
# cameras and the Kinect in the same second").
OPEN_STAGGER_S = 3.0
# The circuit breaker.
STORM_DROP_WINDOW_S = 10.0
STORM_FAIL_WINDOW_S = 60.0
STORM_FAIL_COUNT = 3
STORM_COOLDOWN_MAX_S = 3600.0
# How long an in-flight reservation may stand before it is presumed abandoned
# (an opener that crashed between begin() and end()). Above the longest
# bounded open in the monolith (_CAMERA_LOOP_OPEN_TIMEOUT_S = 35 s).
IN_FLIGHT_STALE_S = 60.0
# ABSENT (2026-09-29, from the owner's USB diagnosis): a camera that has
# VANISHED from the device list is not retried on a timer at all. Every ask
# polls presence (the Media Foundation device list: measured free), and once
# the device is back it must stay listed this long before it is reopened -
# the hub it came back through has just re-enumerated, and a stream START is
# what tripped the resets (16 of 37 eMeet-class starts reset the hub within
# 6 s; the reset rate spiked in the first 2 s after a start).
ABSENT_POLL_S = 3.0
ABSENT_SETTLE_S = 15.0
# When presence cannot be read at all, an absent device is retried at most
# this often (the same ceiling as a locked one).
ABSENT_UNKNOWN_RETRY_S = 600.0
# WEDGED: while ANY open is stuck inside the camera driver, no new camera or
# Kinect open starts (measured: an open wedged for ~32 min and a hub reset
# landed during its teardown). The wedged device itself stays held until its
# stuck call returns; every OTHER device is released after this long, so a
# driver call that never returns cannot blind every camera for the session.
WEDGE_POLL_S = 5.0
WEDGE_HOLD_OTHERS_MAX_S = 3600.0
# Keys that are NOT staggered: unconfigured bare indices (index sweeps). Most
# of them are empty slots that touch no hardware, and staggering a 12-index
# sweep would turn a 3 s boot step into 36 s.
UNSTAGGERED_PREFIXES = ("dshow:",)

KIND_CAMERA = "camera"
KIND_AUDIO = "audio"

# Refusals a caller may reasonably WAIT out (seconds, not minutes).
TIMING_REASONS = frozenset({"min-gap", "stagger", "in-flight"})
# Refusals that mean "the device is being protected - do not open it now".
HOLD_REASONS = frozenset({"usb-storm", "wedged", "absent", "settling",
                          "held", "locked", "backoff"})


class Decision(NamedTuple):
    """The gate's answer. ``wait_s`` is when asking again makes sense (0.0
    when allowed); ``detail`` is a short human sentence for a log line."""
    allowed: bool
    reason: str
    wait_s: float
    detail: str


_ALLOW = Decision(True, "ok", 0.0, "")


def minutes_phrase(seconds: float) -> str:
    """'ten minutes', 'twenty minutes', 'an hour' ... for the spoken line.

    Spoken, not printed, so it avoids digits where a word reads naturally.
    NEVER raises."""
    try:
        m = int(round(float(seconds) / 60.0))
    except Exception:
        return "a while"
    if m >= 60 and m % 60 == 0:
        h = m // 60
        return "an hour" if h == 1 else f"{h} hours"
    words = {1: "a minute", 2: "two minutes", 3: "three minutes",
             5: "five minutes", 10: "ten minutes", 15: "fifteen minutes",
             20: "twenty minutes", 30: "thirty minutes", 40: "forty minutes",
             45: "forty-five minutes"}
    if m in words:
        return words[m]
    if m <= 0:
        return "under a minute"
    return f"{m} minutes"


def _fmt_s(seconds: float) -> str:
    s = max(0.0, float(seconds))
    if s >= 90.0:
        return f"{s / 60.0:.0f} min"
    return f"{s:.0f}s"


class CameraGate:
    """Process-wide open gate. One instance per process (the monolith owns it
    and hands it to the Kinect bridge and the self-diagnostic).

    ``clock``  - seconds, monotonic-or-wall; injectable so tests freeze time.
    ``log``    - one-argument callable for the [camera-gate]/[usb-storm] lines
                 (the monolith passes print: its stdout IS the session log).
    ``announce`` - one-argument callable that queues a SPOKEN line (the
                 monolith passes proactive_announce). Called at most once per
                 storm chain.
    ``lockers`` - zero-argument callable returning the apps that are USING a
                 webcam right now (the monolith reads Windows' camera privacy
                 log). Only called for a device in the locked state.
    ``presence`` - one-argument callable: is device ``key`` on the bus now?
                 True / False / None (cannot tell). Called after a failure or
                 a drop, and on every ask for a device in the absent state.
    """

    def __init__(self, *, min_gap_s: float = 10.0,
                 max_backoff_s: float = 600.0,
                 storm_cooldown_s: float = 600.0,
                 healthy_reset_s: float = HEALTHY_RESET_S,
                 locked_retry_s: float = LOCKED_RETRY_S,
                 locked_poll_s: float = LOCKED_POLL_S,
                 stagger_s: float = OPEN_STAGGER_S,
                 storm_cooldown_max_s: float = STORM_COOLDOWN_MAX_S,
                 storm_drop_window_s: float = STORM_DROP_WINDOW_S,
                 storm_fail_window_s: float = STORM_FAIL_WINDOW_S,
                 storm_fail_count: int = STORM_FAIL_COUNT,
                 clock: "Callable[[], float] | None" = None,
                 log: "Callable[[str], None] | None" = None,
                 announce: "Callable[[str], None] | None" = None,
                 lockers: "Callable[[], list] | None" = None,
                 presence: "Callable[[str], object] | None" = None) -> None:
        self.min_gap_s = self._num(min_gap_s, 10.0)
        self.max_backoff_s = self._num(max_backoff_s, 600.0)
        self.storm_cooldown_s = self._num(storm_cooldown_s, 600.0)
        self.healthy_reset_s = self._num(healthy_reset_s, HEALTHY_RESET_S)
        self.locked_retry_s = self._num(locked_retry_s, LOCKED_RETRY_S)
        self.locked_poll_s = max(0.5, self._num(locked_poll_s, LOCKED_POLL_S))
        self.stagger_s = self._num(stagger_s, OPEN_STAGGER_S)
        self.storm_cooldown_max_s = max(self.storm_cooldown_s,
                                        self._num(storm_cooldown_max_s,
                                                  STORM_COOLDOWN_MAX_S))
        self.storm_drop_window_s = self._num(storm_drop_window_s,
                                             STORM_DROP_WINDOW_S)
        self.storm_fail_window_s = self._num(storm_fail_window_s,
                                             STORM_FAIL_WINDOW_S)
        try:
            self.storm_fail_count = max(2, int(storm_fail_count))
        except Exception:
            self.storm_fail_count = STORM_FAIL_COUNT
        self._clock = clock or time.time
        self._log = log
        self._announce = announce
        self._lockers = lockers
        self._presence = presence
        self._lock = threading.RLock()
        self.reset()

    @staticmethod
    def _num(v, default: float) -> float:
        try:
            f = float(v)
        except Exception:
            return float(default)
        if f != f or f < 0.0:          # NaN / negative -> the shipped default
            return float(default)
        return f

    # ── state ─────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Forget everything (tests; a fresh process starts here anyway)."""
        with self._lock:
            self._dev: dict = {}
            self._last_open_at = 0.0          # last START of a staggered open
            self._last_open_key = ""
            self._drops: deque = deque()      # (t, key, kind)
            self._fails: deque = deque()      # (t, key)
            self._storm_until = 0.0
            self._storm_since = 0.0
            self._storm_reason = ""
            self._storm_trips = 0
            self._storm_last_cooldown = 0.0
            self._storm_last_end = 0.0
            self._storm_end_logged = True
            self._refusal_counts: dict = {}
            self._wedges: dict = {}           # token -> (key, component, since)
            self._wedge_others_released = False

    def _rec(self, key: str) -> dict:
        r = self._dev.get(key)
        if r is None:
            r = {"last_open_at": 0.0, "last_open_by": "", "last_ok_at": 0.0,
                 "last_ok_by": "", "in_flight": "", "in_flight_since": 0.0,
                 "level": 0, "hold_until": 0.0, "locked_by": (),
                 "locked_until": 0.0, "healthy_since": 0.0,
                 "recovering": False, "dropped": False, "held_by": "",
                 "opens": 0, "fails": 0,
                 "absent": False, "absent_since": 0.0, "arrived_at": 0.0}
            self._dev[key] = r
        return r

    def _step_s(self, level: int) -> float:
        if level <= 0 or self.max_backoff_s <= 0.0:
            return 0.0
        if level <= len(BACKOFF_STEPS_S):
            base = BACKOFF_STEPS_S[level - 1]
        else:
            base = BACKOFF_STEPS_S[-1] * (2.0 ** (level - len(BACKOFF_STEPS_S)))
        return min(base, self.max_backoff_s)

    def _staggered(self, key: str) -> bool:
        return not any(key.startswith(p) for p in UNSTAGGERED_PREFIXES)

    def _emit(self, lines: list, spoken: list) -> None:
        """Say what the locked section decided. Outside the lock, guarded."""
        for ln in lines:
            try:
                if self._log is not None:
                    self._log(ln)
            except Exception:
                pass
        for msg in spoken:
            try:
                if self._announce is not None:
                    self._announce(msg)
            except Exception:
                pass

    # ── the breaker ───────────────────────────────────────────────────────
    def _storm_tick_locked(self, now: float, lines: list) -> bool:
        """True while the cool-down runs. Logs its END exactly once."""
        if self._storm_until and now < self._storm_until:
            return True
        if self._storm_until and not self._storm_end_logged:
            self._storm_end_logged = True
            self._storm_last_end = self._storm_until
            lines.append(
                f"  [usb-storm] cool-down over after "
                f"{_fmt_s(self._storm_last_cooldown)} - camera and Kinect "
                f"opens are allowed again, one device at a time "
                f"({self.stagger_s:.0f}s apart).")
            self._storm_until = 0.0
            # Events from before the cool-down must not re-trip it the moment
            # it ends: they describe a bus that has since had minutes to settle.
            self._drops.clear()
            self._fails.clear()
        return False

    def _trip_locked(self, now: float, why: str, lines: list,
                     spoken: list) -> None:
        if self.storm_cooldown_s <= 0.0:
            return                              # breaker disabled by the owner
        if self._storm_tick_locked(now, lines):
            return                              # already cooling down
        repeat = bool(self._storm_last_end
                      and (now - self._storm_last_end) < self.storm_cooldown_max_s
                      and self._storm_last_cooldown > 0.0)
        cool = (min(self._storm_last_cooldown * 2.0, self.storm_cooldown_max_s)
                if repeat else self.storm_cooldown_s)
        self._storm_until = now + cool
        self._storm_since = now
        self._storm_reason = why
        self._storm_trips += 1
        self._storm_last_cooldown = cool
        self._storm_end_logged = False
        self._drops.clear()
        self._fails.clear()
        lines.append(
            f"  [usb-storm] {why} - treating it as a USB bus event. Holding "
            f"ALL camera and Kinect opens for {_fmt_s(cool)} (trip "
            f"#{self._storm_trips}"
            + (", doubled: the bus was unstable again within the hour"
               if repeat else "")
            + "). Streams that are already running are left alone; nothing "
              "is reopened until the cool-down ends.")
        if not repeat:
            # ONCE per chain. A repeat inside the hour is logged, not spoken:
            # the owner has already been told, and hearing it every hour is
            # noise, not information.
            spoken.append(f"Sir, the USB bus looks unstable, so I'm leaving "
                          f"the cameras alone for {minutes_phrase(cool)}.")

    def _evaluate_locked(self, now: float, lines: list, spoken: list) -> None:
        dw = self.storm_drop_window_s
        while self._drops and (now - self._drops[0][0]) > dw:
            self._drops.popleft()
        fw = self.storm_fail_window_s
        while self._fails and (now - self._fails[0][0]) > fw:
            self._fails.popleft()
        cams = sorted({k for _t, k, kind in self._drops if kind == KIND_CAMERA})
        audio = sorted({k for _t, k, kind in self._drops if kind == KIND_AUDIO})
        if len(cams) >= 2:
            self._trip_locked(now, f"{len(cams)} cameras dropped within "
                                   f"{dw:.0f}s ({', '.join(cams)})",
                              lines, spoken)
            return
        if cams and audio:
            self._trip_locked(now, f"a camera and an audio endpoint dropped "
                                   f"within {dw:.0f}s ({cams[0]}, {audio[0]})",
                              lines, spoken)
            return
        fkeys = sorted({k for _t, k in self._fails})
        if len(self._fails) >= self.storm_fail_count and len(fkeys) >= 2:
            self._trip_locked(now, f"{len(self._fails)} camera open failures "
                                   f"across {len(fkeys)} devices within "
                                   f"{fw:.0f}s ({', '.join(fkeys)})",
                              lines, spoken)

    # ── the decision ──────────────────────────────────────────────────────
    def _decide_locked(self, key: str, component: str, now: float,
                       lockers_now, present_now=None,
                       lines: "list | None" = None) -> Decision:
        r = self._rec(key)
        lines = [] if lines is None else lines
        if self._storm_until and now < self._storm_until:
            return Decision(False, "usb-storm", self._storm_until - now,
                            f"USB storm cool-down ({self._storm_reason})")
        if self._wedges:
            mine = [w for w in self._wedges.values() if w[0] == key]
            oldest = min(w[2] for w in self._wedges.values())
            if mine:
                return Decision(False, "wedged", WEDGE_POLL_S,
                                f"an open by {mine[0][1]} is still stuck inside "
                                f"the camera driver ({now - mine[0][2]:.0f}s)")
            if (now - oldest) < WEDGE_HOLD_OTHERS_MAX_S:
                w = min(self._wedges.values(), key=lambda v: v[2])
                return Decision(False, "wedged", WEDGE_POLL_S,
                                f"an open of {w[0]} is still stuck inside the "
                                f"camera driver ({now - w[2]:.0f}s)")
            if not self._wedge_others_released:
                self._wedge_others_released = True
                lines.append(
                    f"  [camera-gate] an open has been stuck inside the camera "
                    f"driver for {_fmt_s(now - oldest)} - releasing every "
                    f"OTHER device; the stuck one stays held until its call "
                    f"returns.")
        if r["absent"]:
            if present_now is True:
                if not r["arrived_at"]:
                    r["arrived_at"] = now
                    lines.append(
                        f"  [camera-gate] {key}: back on the device list after "
                        f"{_fmt_s(now - r['absent_since'])} - reopening it after "
                        f"a {ABSENT_SETTLE_S:.0f}s settle, not at once.")
                left = ABSENT_SETTLE_S - (now - r["arrived_at"])
                if left > 0.0:
                    return Decision(False, "settling", left,
                                    f"back on the bus {now - r['arrived_at']:.0f}s "
                                    f"ago; settling")
                r["absent"] = False
                r["arrived_at"] = 0.0
            elif present_now is False:
                r["arrived_at"] = 0.0
                return Decision(False, "absent", ABSENT_POLL_S,
                                f"gone from the device list "
                                f"{now - r['absent_since']:.0f}s ago")
            elif (now - r["absent_since"]) < ABSENT_UNKNOWN_RETRY_S:
                return Decision(False, "absent",
                                min(ABSENT_POLL_S * 10,
                                    ABSENT_UNKNOWN_RETRY_S
                                    - (now - r["absent_since"])),
                                "gone from the device list (presence "
                                "unreadable; retrying at the cap)")
            else:
                r["absent"] = False
        if r["held_by"] and r["held_by"] != component:
            return Decision(False, "held", self.min_gap_s or 1.0,
                            f"{r['held_by']} is streaming this device")
        if r["in_flight"] and r["in_flight"] != component:
            if (now - r["in_flight_since"]) < IN_FLIGHT_STALE_S:
                return Decision(False, "in-flight", 1.0,
                                f"{r['in_flight']} is opening it right now")
            r["in_flight"] = ""                 # presumed abandoned
        if r["locked_by"]:
            if now >= r["locked_until"]:
                pass                            # the at-most-every-N retry
            elif lockers_now is not None and not (
                    {str(x).lower() for x in lockers_now}
                    & {str(x).lower() for x in r["locked_by"]}):
                pass                            # the locker is gone: retry now
            else:
                return Decision(False, "locked",
                                min(self.locked_poll_s,
                                    max(0.5, r["locked_until"] - now)),
                                f"{', '.join(r['locked_by'])} appears to hold it")
        if r["hold_until"] and now < r["hold_until"]:
            return Decision(False, "backoff", r["hold_until"] - now,
                            f"reopen backoff level {r['level']}")
        if (self.min_gap_s > 0.0 and r["last_open_at"]
                and r["last_open_by"] != component
                and (now - r["last_open_at"]) < self.min_gap_s):
            return Decision(False, "min-gap",
                            self.min_gap_s - (now - r["last_open_at"]),
                            f"{r['last_open_by']} opened it "
                            f"{now - r['last_open_at']:.1f}s ago")
        if (self.stagger_s > 0.0 and self._staggered(key)
                and self._last_open_key and self._last_open_key != key
                and (now - self._last_open_at) < self.stagger_s):
            return Decision(False, "stagger",
                            self.stagger_s - (now - self._last_open_at),
                            f"{self._last_open_key} was opened "
                            f"{now - self._last_open_at:.1f}s ago")
        return _ALLOW

    def _poll_presence(self, key: str, only_if_absent: bool = True):
        """Presence of ``key`` (True/False/None), polled OUTSIDE the lock."""
        if self._presence is None:
            return None
        if only_if_absent:
            with self._lock:
                r = self._dev.get(key)
                if not (r and r["absent"]):
                    return None
        try:
            v = self._presence(key)
        except Exception:
            return None
        return v if v in (True, False) else None

    def _mark_absent_locked(self, key: str, now: float, lines: list) -> None:
        r = self._rec(key)
        if r["absent"]:
            return
        r["absent"] = True
        r["absent_since"] = now
        r["arrived_at"] = 0.0
        r["recovering"] = True
        lines.append(
            f"  [camera-gate] {key}: gone from the device list - it is NOT "
            f"reopened on a timer; JARVIS waits for it to come back, then "
            f"settles {ABSENT_SETTLE_S:.0f}s before opening it.")

    def _poll_lockers_if_needed(self, key: str):
        with self._lock:
            r = self._dev.get(key)
            need = bool(r and r["locked_by"])
        if not need or self._lockers is None:
            return None
        try:
            return list(self._lockers() or [])
        except Exception:
            return None

    def check(self, key: str, component: str,
              now: "float | None" = None) -> Decision:
        """What begin() would answer, WITHOUT reserving anything."""
        try:
            now = self._clock() if now is None else now
            lk = self._poll_lockers_if_needed(key)
            pres = self._poll_presence(key)
            lines: list = []
            with self._lock:
                self._storm_tick_locked(now, lines)
                d = self._decide_locked(key, component, now, lk, pres, lines)
            self._emit(lines, [])
            return d
        except Exception:
            return _ALLOW

    def begin(self, key: str, component: str,
              now: "float | None" = None) -> Decision:
        """Ask to open ``key`` on behalf of ``component``. When allowed, the
        open is RESERVED (in-flight, and stamped for min-gap/stagger) until the
        matching :meth:`end`. NEVER raises; fails OPEN on an internal error."""
        try:
            now = self._clock() if now is None else now
            lk = self._poll_lockers_if_needed(key)
            pres = self._poll_presence(key)
            lines: list = []
            with self._lock:
                self._storm_tick_locked(now, lines)
                d = self._decide_locked(key, component, now, lk, pres, lines)
                r = self._rec(key)
                if d.allowed:
                    if r["locked_by"]:
                        # Leaving the locked state is a transition the owner
                        # should see, exactly like entering it.
                        lines.append(
                            f"  [camera-gate] {key}: "
                            + (f"{', '.join(r['locked_by'])} is no longer "
                               f"using a webcam"
                               if now < r["locked_until"] else
                               f"still locked after {_fmt_s(self.locked_retry_s)}")
                            + " - retrying the open once.")
                        r["locked_by"] = ()
                        r["locked_until"] = 0.0
                    r["in_flight"] = component
                    r["in_flight_since"] = now
                    r["last_open_at"] = now
                    r["last_open_by"] = component
                    r["opens"] += 1
                    if self._staggered(key):
                        self._last_open_at = now
                        self._last_open_key = key
                else:
                    k = (key, d.reason)
                    self._refusal_counts[k] = self._refusal_counts.get(k, 0) + 1
            self._emit(lines, [])
            return d
        except Exception:
            return _ALLOW

    def end(self, key: str, component: str, ok: bool, *,
            now: "float | None" = None, lockers=None,
            escalate: bool = True, count: bool = True) -> None:
        """Report the outcome of an open begin() allowed.

        ``ok``       the open produced a usable device.
        ``lockers``  webcam-locking apps running when it failed (-> locked).
        ``escalate`` a failure arms the per-device backoff (False for one-shot
                     verdict gatherers like the boot probes, whose own bounded
                     retry policy must not be pre-empted by a 30 s ladder).
        ``count``    a failure counts toward the storm breaker.
        NEVER raises."""
        try:
            now = self._clock() if now is None else now
            gone = (not ok) and self._poll_presence(
                key, only_if_absent=False) is False
            lines: list = []
            spoken: list = []
            with self._lock:
                r = self._rec(key)
                if r["in_flight"] == component:
                    r["in_flight"] = ""
                if gone:
                    self._mark_absent_locked(key, now, lines)
                if ok:
                    r["last_ok_at"] = now
                    r["last_ok_by"] = component
                    r["locked_by"] = ()
                    r["locked_until"] = 0.0
                    if r["recovering"]:
                        # A read-failure RECOVERY that worked still spends a
                        # rung: if the device dies again soon, the next
                        # recovery waits. Only sustained frames reset it.
                        r["recovering"] = False
                        r["level"] += 1
                        r["hold_until"] = now + self._step_s(r["level"])
                    return
                r["fails"] += 1
                r["healthy_since"] = 0.0
                if lockers:
                    r["locked_by"] = tuple(str(x) for x in lockers)
                    r["locked_until"] = now + self.locked_retry_s
                if escalate or r["recovering"]:
                    r["recovering"] = False
                    r["level"] += 1
                    r["hold_until"] = now + self._step_s(r["level"])
                if count:
                    self._fails.append((now, key))
                    self._evaluate_locked(now, lines, spoken)
            self._emit(lines, spoken)
        except Exception:
            pass

    def cancel(self, key: str, component: str) -> None:
        """Drop a reservation that never reached the device (the opener was
        stuck behind a lock and gave up). Records nothing: no success, no
        failure. NEVER raises."""
        try:
            with self._lock:
                r = self._dev.get(key)
                if r is not None and r["in_flight"] == component:
                    r["in_flight"] = ""
        except Exception:
            pass

    # ── signals from the streams ──────────────────────────────────────────
    def note_frame(self, key: str, now: "float | None" = None) -> None:
        """A healthy frame arrived from ``key``. Call it as often as you like
        (it is a dict update); the backoff resets after healthy_reset_s of
        UNINTERRUPTED frames. NEVER raises."""
        try:
            now = self._clock() if now is None else now
            lines: list = []
            with self._lock:
                r = self._dev.get(key)
                if r is None:
                    return
                r["dropped"] = False
                if r["locked_by"]:
                    r["locked_by"] = ()
                    r["locked_until"] = 0.0
                if not r["healthy_since"]:
                    r["healthy_since"] = now
                if (r["level"] > 0
                        and (now - r["healthy_since"]) >= self.healthy_reset_s):
                    lines.append(
                        f"  [camera-gate] {key}: {_fmt_s(self.healthy_reset_s)}"
                        f" of healthy frames - reopen backoff reset (was level "
                        f"{r['level']}).")
                    r["level"] = 0
                    r["hold_until"] = 0.0
            self._emit(lines, [])
        except Exception:
            pass

    def note_drop(self, key: str, component: str = "",
                  kind: str = KIND_CAMERA,
                  now: "float | None" = None) -> bool:
        """A device that was delivering has stopped (a camera's read-failure
        escalation, the Kinect's stale-stream reset, an audio endpoint that
        vanished). Counted ONCE per episode per device - a camera failing for a
        minute is one drop, not 600. The next open of that device is a
        RECOVERY and spends a backoff rung. Returns True iff this call tripped
        the breaker. NEVER raises."""
        try:
            now = self._clock() if now is None else now
            gone = (kind == KIND_CAMERA and self._poll_presence(
                key, only_if_absent=False) is False)
            lines: list = []
            spoken: list = []
            tripped = False
            with self._lock:
                r = self._rec(key)
                already = False
                if kind == KIND_CAMERA:
                    if gone:
                        self._mark_absent_locked(key, now, lines)
                    if r["dropped"]:
                        already = True
                    else:
                        r["dropped"] = True
                        r["recovering"] = True
                        r["healthy_since"] = 0.0
                if not already:
                    before = self._storm_trips
                    self._drops.append((now, key, kind))
                    self._evaluate_locked(now, lines, spoken)
                    tripped = self._storm_trips != before
            self._emit(lines, spoken)
            return tripped
        except Exception:
            return False

    def note_wedged(self, key: str, component: str,
                    now: "float | None" = None):
        """An open of ``key`` by ``component`` is STUCK inside the camera
        driver (its bounded worker was abandoned while it held the camera I/O
        lock). Until :meth:`note_unwedged` is called with the returned token,
        no camera or Kinect open starts. NEVER raises; returns None on error."""
        try:
            now = self._clock() if now is None else now
            token = object()
            lines: list = []
            with self._lock:
                first = not self._wedges
                self._wedges[token] = (key, component, now)
                if first:
                    self._wedge_others_released = False
                    lines.append(
                        f"  [camera-gate] {key}: an open by {component} is stuck "
                        f"inside the camera driver - NO camera or Kinect open "
                        f"starts until it returns (a reset landed during such "
                        f"a teardown on 2026-09-29).")
            self._emit(lines, [])
            return token
        except Exception:
            return None

    def note_unwedged(self, token, now: "float | None" = None) -> None:
        """The stuck call behind ``token`` has returned. NEVER raises."""
        try:
            now = self._clock() if now is None else now
            lines: list = []
            with self._lock:
                w = self._wedges.pop(token, None)
                if w is not None and not self._wedges:
                    lines.append(
                        f"  [camera-gate] {w[0]}: the stuck open returned after "
                        f"{_fmt_s(now - w[2])} - camera opens are allowed again.")
            self._emit(lines, [])
        except Exception:
            pass

    def wedged(self) -> bool:
        try:
            with self._lock:
                return bool(self._wedges)
        except Exception:
            return False

    def hold(self, key: str, component: str) -> None:
        """``component`` is STREAMING ``key``: other components are refused
        until :meth:`unhold`. NEVER raises."""
        try:
            with self._lock:
                self._rec(key)["held_by"] = component
        except Exception:
            pass

    def unhold(self, key: str, component: str) -> None:
        try:
            with self._lock:
                r = self._dev.get(key)
                if r is not None and r["held_by"] == component:
                    r["held_by"] = ""
        except Exception:
            pass

    # ── read-outs ─────────────────────────────────────────────────────────
    def storm_active(self, now: "float | None" = None) -> bool:
        try:
            now = self._clock() if now is None else now
            lines: list = []
            with self._lock:
                active = self._storm_tick_locked(now, lines)
            self._emit(lines, [])
            return active
        except Exception:
            return False

    def retry_in(self, key: str, component: str,
                 now: "float | None" = None) -> "tuple[float, str]":
        """(seconds, reason) until ``component`` should next ASK for ``key``.
        0.0 means "ask now". For a locked device this is the cheap poll
        interval, not a promise of an open. NEVER raises."""
        d = self.check(key, component, now)
        if d.allowed:
            return 0.0, "ok"
        return max(0.0, float(d.wait_s)), d.reason

    def locked_by(self, key: str) -> list:
        """The locking apps recorded for ``key`` (empty when not locked).
        NEVER raises."""
        try:
            with self._lock:
                r = self._dev.get(key)
                return list(r["locked_by"]) if r else []
        except Exception:
            return []

    def recent_success(self, key: str, component: str, within_s: float,
                       now: "float | None" = None) -> "float | None":
        """Age of the last SUCCESSFUL open of ``key`` by ``component``, if it
        was within ``within_s``; else None. Lets a boot probe reuse a verdict
        it proved seconds ago instead of opening the device again."""
        try:
            now = self._clock() if now is None else now
            with self._lock:
                r = self._dev.get(key)
                if not r or not r["last_ok_at"] or r["last_ok_by"] != component:
                    return None
                age = now - r["last_ok_at"]
                return age if 0.0 <= age <= within_s else None
        except Exception:
            return None

    def snapshot(self, now: "float | None" = None) -> dict:
        """Plain-data view for status lines and tests. NEVER raises."""
        try:
            now = self._clock() if now is None else now
            with self._lock:
                devs = {}
                for k, r in self._dev.items():
                    devs[k] = {
                        "level": r["level"],
                        "hold_s": max(0.0, r["hold_until"] - now)
                        if r["hold_until"] else 0.0,
                        "locked_by": list(r["locked_by"]),
                        "held_by": r["held_by"],
                        "in_flight": r["in_flight"],
                        "opens": r["opens"], "fails": r["fails"],
                        "last_open_by": r["last_open_by"],
                        "recovering": r["recovering"],
                        "absent": r["absent"],
                    }
                active = bool(self._storm_until and now < self._storm_until)
                return {
                    "storm_active": active,
                    "storm_remaining_s": (self._storm_until - now) if active else 0.0,
                    "storm_reason": self._storm_reason if active else "",
                    "storm_trips": self._storm_trips,
                    "wedged": [w[0] for w in self._wedges.values()],
                    "storm_last_cooldown_s": self._storm_last_cooldown,
                    "devices": devs,
                    "refusals": {f"{k}|{r}": n for (k, r), n
                                 in self._refusal_counts.items()},
                }
        except Exception:
            return {"storm_active": False, "devices": {}, "refusals": {}}
