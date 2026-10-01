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

  quarantined  this device's own stream START has been followed by a USB bus
             event CULPRIT_THRESHOLD times within an hour (see CULPRIT
             QUARANTINE below). Never opened again automatically this
             session; only the owner lifts it (lift_quarantine).
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
             only after HEALTHY_RESET_S of sustained healthy frames. A device
             that DIES ON OPEN (below) waits dies_on_open_retry_s instead.
  min-gap    a DIFFERENT component opened this device less than min_gap_s ago.
             (A component's own retries are governed by its backoff, not this.)
  stagger    a DIFFERENT configured device was opened less than OPEN_STAGGER_S
             ago - so at boot, and after a cool-down, the cameras and the Kinect
             are never opened in the same second.

Only the last three are worth WAITING for (see TIMING_REASONS); the rest mean
"not now - ask again at wait_s".

────────────────────────────────────────────────────────────────────────────
PROBATION AND CULPRIT QUARANTINE (v2.0.132, from the 2026-09-29 live log)
────────────────────────────────────────────────────────────────────────────
The first cool-down worked. Then, 18 s after it ended, the face tracker
opened one webcam, the hub reset 0.8 s later and again 2.5 s after that, the
other webcam's open found its device gone from the list, the first webcam
went dead after 60 failed reads - and the breaker did NOT trip again, because
a failed open of a vanished device was not a "drop" and one drop is not two.
The webcam was reopened 21 s later and the hub reset 0.04 s after THAT. A
read-only USB test the same day (no JARVIS) showed that webcam's plain stream
starts reset the hub 8 of 10 times; its sibling on the same hub, 0 of 10.

  probation  for probation_s (default 180 s) after a cool-down ends, and for
             REOPEN_PROBATION_S after any successful reopen of a recovering
             device while that storm chain is live (a cool-down ended within
             the hour), ONE drop re-trips the breaker at once, with the
             doubled cool-down. A drop is: a camera gone from the device list
             (a drop report or a failed open whose device vanished), a camera
             read-failure burst (note_drop), or an audio endpoint vanishing.
             Open failures of a device that is still listed do not count.
  bursts     a read-failure burst is counted once per STREAM, not once per
             "episode since the last healthy frame": a device reopened after a
             drop that fails again is a new drop. So bursts on >=2 cameras
             within 10 s trip the breaker exactly like device-list drops.
  culprit    a USB bus event (a breaker trip, or a camera vanishing from the
             device list) whose ONSET falls inside a device's stream start -
             from the begin() of an open that SUCCEEDED to culprit_window_s
             (default 5 s) after it - is a strike against that device (the
             latest such starter; one strike per stream). culprit_threshold
             strikes (default 2) within an hour QUARANTINE it for the rest of
             the session: said once through ``announce`` with the device's
             friendly label, logged, and shown in snapshot(). A device whose
             own open FAILED never earns a strike - a victim that tried to
             start into a resetting hub is not the thing that reset it.
             The quarantine is not persisted: a restart (or
             lift_quarantine) forgets it. (The dies-on-open run below IS.)

────────────────────────────────────────────────────────────────────────────
DIES ON OPEN (R11, from the 2026-09-29 19:39-19:59 live log)
────────────────────────────────────────────────────────────────────────────
A depth sensor dropped off USB (Kernel-PnP "surprise removed", with its mic
array) within about a second of EVERY open, and each drop came exactly when
the backoff ladder let its bridge reopen it: 30, 60, 120, 300, 600 s, then
every 10 minutes for good. No other device was involved, so the breaker and
the culprit rule (both about the SHARED bus) never fired. Each reopen cost a
USB re-enumeration and an audio device-list change for nothing.

  dies-on-open  a full drop whose ONSET is within dies_on_open_window_s
             (DIES_ON_OPEN_WINDOW_S, 15 s) of that device's last SUCCESSFUL
             open. A stream that delivers a frame (note_frame) later than the
             window after its open, or whose drop begins after it, has
             streamed normally: it is not counted and it ends the run.
             dies_on_open_count (DIES_ON_OPEN_COUNT, 3) in a row raise the
             device's reopen hold to dies_on_open_retry_s (owner knob
             CAMERA_DIES_ON_OPEN_RETRY_S, 30 min; 0 = off), doubling on each
             further one up to DIES_ON_OPEN_RETRY_MAX_S (60 min), instead of
             the ladder's 10 min cap. One log line each time it is raised;
             said ONCE per device per session through ``announce`` (what was
             seen, what to check). NOT a quarantine: the device is still
             retried on that slow timer, a reopen that streams normally puts
             it back on the usual ladder, and lift_quarantine (the owner's
             "use it again") clears it and allows an open at once - and ONE
             more death after that puts it straight back on the slow retry,
             as the owner's reply promises (2026-10-01).

  survives a restart (2026-10-01)  with ``doo_state_path`` (the monolith
             passes data/camera_gate_doo.json) the run - count, slow retry,
             hold deadline, when it was said - is saved whenever it is armed,
             cleared or lifted, and restored at construction. Before this,
             every deploy / tray restart (14 on 2026-09-30) reopened the
             Kinect three more times - three USB re-enumerations and an audio
             device-list change - and SPOKE the warning again. Now a restart
             inside the hold does not open it at all, the first open after the
             hold is judged at once (one more death re-arms the doubled
             retry), and the warning is said at most once per
             DIES_ON_OPEN_SAY_AGAIN_S. Needs a wall clock (the default).

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

import json
import os
import threading
import time
from collections import deque
from typing import Callable, NamedTuple

__all__ = [
    "BACKOFF_STEPS_S", "Decision", "CameraGate", "TIMING_REASONS",
    "HOLD_REASONS", "KIND_CAMERA", "KIND_AUDIO", "minutes_phrase",
    "STORM_PROBATION_S", "REOPEN_PROBATION_S", "CULPRIT_WINDOW_S",
    "CULPRIT_THRESHOLD", "DIES_ON_OPEN_WINDOW_S", "DIES_ON_OPEN_COUNT",
    "DIES_ON_OPEN_RETRY_S", "DIES_ON_OPEN_RETRY_MAX_S",
    "DIES_ON_OPEN_SAY_AGAIN_S",
    "LIFT_QUARANTINE", "LIFT_SLOW_RETRY",
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
# PROBATION (v2.0.132): after a cool-down ends, a SINGLE drop re-trips the
# breaker for this long (owner knob CAMERA_STORM_PROBATION_S; 0 = off)...
STORM_PROBATION_S = 180.0
# ...and for this long after any successful reopen of a recovering device,
# while the storm chain is live (a cool-down ended within the hour).
REOPEN_PROBATION_S = 60.0
# CULPRIT QUARANTINE (v2.0.132): a USB bus event whose onset falls within this
# long of a device's stream start is a strike against that device (owner knob
# CAMERA_CULPRIT_WINDOW_S; 0 = off)...
CULPRIT_WINDOW_S = 5.0
# ...and this many strikes within CULPRIT_MEMORY_S quarantine it for the rest
# of the session (owner knob CAMERA_CULPRIT_THRESHOLD; 0 = off).
CULPRIT_THRESHOLD = 2
CULPRIT_MEMORY_S = 3600.0
# DIES ON OPEN (R11): a full drop that begins within this long of the device's
# last successful open means the stream died on arrival (the live case: gone
# within ~1 s, reported by the stale-stream check ~4-5 s after the open)...
DIES_ON_OPEN_WINDOW_S = 15.0
# ...this many of them in a row, with no normal stream in between...
DIES_ON_OPEN_COUNT = 3
# ...hold the device's next reopen this long (owner knob
# CAMERA_DIES_ON_OPEN_RETRY_S; 0 = off), doubling on each further one up to
# the max (or the knob, when the owner set it higher).
DIES_ON_OPEN_RETRY_S = 1800.0
DIES_ON_OPEN_RETRY_MAX_S = 3600.0
# A restored run (doo_state_path) that was already SPOKEN within this long is
# not spoken again on its next raise - logged only, as within one session.
DIES_ON_OPEN_SAY_AGAIN_S = 12 * 3600.0
# How often a caller refused as "quarantined" should ask again. Asking costs a
# dict lookup; the answer only changes when the owner lifts the quarantine.
QUARANTINE_POLL_S = 600.0
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
HOLD_REASONS = frozenset({"quarantined", "usb-storm", "wedged", "absent",
                          "settling", "held", "locked", "backoff"})
# What CameraGate.lift() can lift (the owner's "use the Kinect again").
LIFT_QUARANTINE = "quarantine"      # the culprit quarantine (hub knocked out)
LIFT_SLOW_RETRY = "dies-on-open"    # the slow dies-on-open retry (R11)


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
    ``labeler`` - one-argument callable: the SPOKEN name of device ``key``
                 ("the left webcam"). Only called when a device is
                 quarantined or put on the slow dies-on-open retry; a fault
                 falls back to a name built from the key.
    ``doo_state_path`` - JSON file the dies-on-open runs are saved to and
                 restored from, so they survive a restart. None (the default,
                 and every test that does not ask) = nothing on disk.
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
                 probation_s: float = STORM_PROBATION_S,
                 culprit_window_s: float = CULPRIT_WINDOW_S,
                 culprit_threshold: int = CULPRIT_THRESHOLD,
                 dies_on_open_retry_s: float = DIES_ON_OPEN_RETRY_S,
                 dies_on_open_window_s: float = DIES_ON_OPEN_WINDOW_S,
                 dies_on_open_count: int = DIES_ON_OPEN_COUNT,
                 clock: "Callable[[], float] | None" = None,
                 log: "Callable[[str], None] | None" = None,
                 announce: "Callable[[str], None] | None" = None,
                 lockers: "Callable[[], list] | None" = None,
                 presence: "Callable[[str], object] | None" = None,
                 labeler: "Callable[[str], str] | None" = None,
                 doo_state_path: "str | None" = None) -> None:
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
        self.probation_s = self._num(probation_s, STORM_PROBATION_S)
        self.culprit_window_s = self._num(culprit_window_s, CULPRIT_WINDOW_S)
        try:
            self.culprit_threshold = max(0, int(culprit_threshold))
        except Exception:
            self.culprit_threshold = CULPRIT_THRESHOLD
        self.dies_on_open_retry_s = self._num(dies_on_open_retry_s,
                                              DIES_ON_OPEN_RETRY_S)
        self.dies_on_open_retry_max_s = max(self.dies_on_open_retry_s,
                                            DIES_ON_OPEN_RETRY_MAX_S)
        self.dies_on_open_window_s = self._num(dies_on_open_window_s,
                                               DIES_ON_OPEN_WINDOW_S)
        try:
            self.dies_on_open_count = max(1, int(dies_on_open_count))
        except Exception:
            self.dies_on_open_count = DIES_ON_OPEN_COUNT
        self._clock = clock or time.time
        self._log = log
        self._announce = announce
        self._lockers = lockers
        self._presence = presence
        self._labeler = labeler
        self._lock = threading.RLock()
        self._doo_state_path = doo_state_path or None
        self._doo_io_lock = threading.Lock()
        self.reset()
        # NOT inside reset(): a test that resets a gate must stay hermetic.
        self._doo_load()

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
            self._reopen_at = 0.0             # last successful RE-open ...
            self._reopen_key = ""             # ... of a recovering device
            self._doo_said: set = set()       # keys told "dies on open" (R11)
            self._doo_said_at: dict = {}      # key -> clock time it was said
            self._doo_dirty = False           # runs changed since last save

    def _rec(self, key: str) -> dict:
        r = self._dev.get(key)
        if r is None:
            r = {"last_open_at": 0.0, "last_open_by": "", "last_ok_at": 0.0,
                 "last_ok_by": "", "in_flight": "", "in_flight_since": 0.0,
                 "level": 0, "hold_until": 0.0, "locked_by": (),
                 "locked_until": 0.0, "healthy_since": 0.0,
                 "recovering": False, "dropped": False, "held_by": "",
                 "opens": 0, "fails": 0,
                 "absent": False, "absent_since": 0.0, "arrived_at": 0.0,
                 # culprit bookkeeping (v2.0.132)
                 "stream_begin_at": 0.0, "stream_ok_at": 0.0, "opens_ok": 0,
                 "struck_open": -1, "strikes": [], "quarantined": False,
                 "quarantined_at": 0.0, "quarantine_why": "",
                 "quarantine_label": "",
                 # dies-on-open bookkeeping (R11): the run length, the
                 # opens_ok index of the last stream already judged (a drop
                 # counted, or proven by a late frame), and the slow retry
                 # interval armed (0.0 = on the normal ladder).
                 "doo_count": 0, "doo_judged": -1, "doo_retry_s": 0.0,
                 "doo_until": 0.0}
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
        self._doo_save()
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

    def _chain_live_locked(self, now: float) -> bool:
        """A cool-down ended within the hour: a trip now would be a REPEAT
        (doubled, not spoken)."""
        return bool(self._storm_last_end
                    and self._storm_last_cooldown > 0.0
                    and (now - self._storm_last_end) < self.storm_cooldown_max_s)

    def _probation_locked(self, now: float) -> str:
        """Why ONE drop would re-trip the breaker right now ('' when it would
        not). Call after _storm_tick_locked."""
        if self.probation_s <= 0.0 or self.storm_cooldown_s <= 0.0:
            return ""
        if self._storm_until and now < self._storm_until:
            return ""                           # cooling down: nothing to trip
        if self._storm_last_end:
            since = now - self._storm_last_end
            if 0.0 <= since < self.probation_s:
                return (f"{since:.0f}s after the last cool-down ended "
                        f"(probation {self.probation_s:.0f}s)")
        if self._reopen_at and self._chain_live_locked(now):
            since = now - self._reopen_at
            if 0.0 <= since < REOPEN_PROBATION_S:
                return (f"{since:.0f}s after {self._reopen_key} was reopened "
                        f"(probation {REOPEN_PROBATION_S:.0f}s)")
        return ""

    def _count_drop_locked(self, now: float, key: str, kind: str,
                           onset: float, cause: str, vanished: bool,
                           lines: list, spoken: list,
                           minor: bool = False) -> bool:
        """One NEW drop (already de-duplicated by the caller). Trips the
        breaker on probation or on the two-device rules, and treats a camera
        that VANISHED from the device list as a hub-level event for the
        culprit tally even when nothing trips. A MINOR drop (one failed read
        of a camera still on the device list) counts toward the two-device
        rules only: alone it is not evidence of a bus event, so it never
        re-trips the breaker on probation. Returns True iff it tripped."""
        before = self._storm_trips
        self._storm_tick_locked(now, lines)
        self._drops.append((now, key, kind, onset))
        why = "" if (minor and not vanished) else self._probation_locked(now)
        if why:
            self._trip_locked(now, f"{key} dropped ({cause}) {why}",
                              lines, spoken, onset=onset)
        else:
            self._evaluate_locked(now, lines, spoken)
        tripped = self._storm_trips != before
        if vanished and not tripped:
            self._attribute_locked(now, onset,
                                   f"{key} vanishing from the device list",
                                   lines, spoken)
        return tripped

    def _trip_locked(self, now: float, why: str, lines: list,
                     spoken: list, onset: "float | None" = None) -> None:
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
        self._attribute_locked(now, now if onset is None else onset,
                               f"USB storm trip #{self._storm_trips}",
                               lines, spoken)

    # ── the culprit ───────────────────────────────────────────────────────
    def _label_for(self, key: str) -> str:
        """The spoken name of ``key`` ("the left webcam")."""
        try:
            if self._labeler is not None:
                got = str(self._labeler(key) or "").strip()
                if got:
                    return got
        except Exception:
            pass
        if key == "kinect":
            return "the Kinect"
        name = key.split(":", 1)[1] if ":" in key else key
        return f"the {name} camera"

    def _attribute_locked(self, now: float, onset: float, why: str,
                          lines: list, spoken: list) -> None:
        """A USB bus event with this ONSET happened: strike the device whose
        stream START it followed, and quarantine a repeat offender."""
        if self.culprit_window_s <= 0.0 or self.culprit_threshold <= 0:
            return
        best, best_begin = None, 0.0
        for k, r in self._dev.items():
            if (not self._staggered(k) or r["quarantined"]
                    or not r["stream_ok_at"]
                    or r["struck_open"] == r["opens_ok"]):
                continue
            b = r["stream_begin_at"] or r["stream_ok_at"]
            if b <= onset <= r["stream_ok_at"] + self.culprit_window_s:
                if best is None or b > best_begin:
                    best, best_begin = k, b
        if best is None:
            return
        r = self._dev[best]
        r["struck_open"] = r["opens_ok"]
        r["strikes"] = [t for t in r["strikes"]
                        if (now - t) < CULPRIT_MEMORY_S] + [now]
        n = len(r["strikes"])
        lines.append(
            f"  [camera-culprit] {best}: {why} began "
            f"{onset - r['stream_ok_at']:+.1f}s after its stream started - "
            f"strike {n} of {self.culprit_threshold} within the hour.")
        if n < self.culprit_threshold:
            return
        label = self._label_for(best)
        r["quarantined"] = True
        r["quarantined_at"] = now
        r["quarantine_label"] = label
        r["quarantine_why"] = (f"its stream start was followed by a USB bus "
                               f"event {n} times within the hour")
        lines.append(
            f"  [camera-quarantine] {best} ({label}): {r['quarantine_why']} - "
            f"QUARANTINED for the rest of this session. JARVIS will not open "
            f"it again on its own; move it to a port on a different hub, then "
            f"say 'use {label} again'.")
        spoken.append(f"Sir, {label} keeps knocking the USB hub offline "
                      f"whenever it starts, so I've stopped using it until "
                      f"it's moved to another port.")

    # ── dies on open (R11) ────────────────────────────────────────────────
    def _doo_on_locked(self) -> bool:
        return self.dies_on_open_retry_s > 0.0 and self.dies_on_open_window_s > 0.0

    def _doo_save(self) -> None:
        """Write the armed dies-on-open runs to doo_state_path if they changed
        (2026-10-01). The snapshot is taken under the gate lock, the file is
        written outside it (atomic: temp file + os.replace). NEVER raises."""
        if not self._doo_dirty or not self._doo_state_path:
            return
        try:
            with self._doo_io_lock:
                with self._lock:
                    if not self._doo_dirty:
                        return
                    self._doo_dirty = False
                    devices = {
                        k: {"count": r["doo_count"],
                            "retry_s": r["doo_retry_s"],
                            "until": r["doo_until"],
                            "said_at": self._doo_said_at.get(k, 0.0)}
                        for k, r in self._dev.items() if r["doo_retry_s"] > 0.0}
                path = self._doo_state_path
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"version": 1, "devices": devices}, f)
                os.replace(tmp, path)
        except Exception:
            pass

    def _doo_load(self) -> None:
        """Restore the dies-on-open runs saved by a previous process. The hold
        is capped at now + the max retry (a clock jump cannot bench a device
        for good); doo_judged stays -1, so the first stream after the hold is
        judged normally - a death re-arms the doubled retry, a stream that
        runs past the window clears the run. NEVER raises."""
        path = self._doo_state_path
        if not path:
            return
        try:
            if not self._doo_on_locked() or not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                devices = (json.load(f) or {}).get("devices") or {}
            now = self._clock()
            lines: list = []
            with self._lock:
                for key, e in devices.items():
                    try:
                        key = str(key)
                        count = int(e.get("count") or 0)
                        retry = float(e.get("retry_s") or 0.0)
                        until = float(e.get("until") or 0.0)
                        said = float(e.get("said_at") or 0.0)
                    except Exception:
                        continue
                    if count <= 0 or not retry > 0.0:
                        continue
                    r = self._rec(key)
                    r["doo_count"] = count
                    r["doo_retry_s"] = min(retry, self.dies_on_open_retry_max_s)
                    until = min(until, now + self.dies_on_open_retry_max_s)
                    if until > now:
                        r["hold_until"] = r["doo_until"] = until
                    if said and 0.0 <= now - said < DIES_ON_OPEN_SAY_AGAIN_S:
                        self._doo_said.add(key)
                        self._doo_said_at[key] = said
                    held = (f"held for {_fmt_s(until - now)} more"
                            if until > now else "its hold has run out")
                    lines.append(
                        f"  [camera-gate] {key}: restored from the last run - "
                        f"its last {count} opens each died on open; {held}, "
                        f"then retried every {_fmt_s(r['doo_retry_s'])} "
                        f"until a reopen streams normally.")
            self._emit(lines, [])
        except Exception:
            pass

    def _doo_clear_locked(self, key: str, r: dict, why: str,
                          lines: list) -> None:
        """End a dies-on-open run: the device streamed normally."""
        was_slow = r["doo_retry_s"] > 0.0
        r["doo_count"] = 0
        r["doo_retry_s"] = 0.0
        if was_slow:
            self._doo_dirty = True
            lines.append(
                f"  [camera-gate] {key}: {why} - it no longer dies on open; "
                f"back on the normal reopen ladder.")

    def _dies_on_open_locked(self, key: str, r: dict, now: float,
                             onset: float, lines: list,
                             spoken: list) -> None:
        """A FULL drop of ``key`` beginning at ``onset`` was just counted.
        Judge its stream (once): died on arrival, or streamed normally."""
        if not self._doo_on_locked() or r["quarantined"]:
            return
        ok_at = r["stream_ok_at"]
        if not ok_at or r["doo_judged"] == r["opens_ok"]:
            return              # never streamed, or this stream is judged
        r["doo_judged"] = r["opens_ok"]
        win = self.dies_on_open_window_s
        lived = onset - ok_at
        if lived > win:
            self._doo_clear_locked(
                key, r, f"its last stream ran {_fmt_s(lived)} before it "
                        f"dropped", lines)
            return
        r["doo_count"] += 1
        n = r["doo_count"]
        if n < self.dies_on_open_count:
            return
        retry = min(self.dies_on_open_retry_s
                    * (2.0 ** (n - self.dies_on_open_count)),
                    self.dies_on_open_retry_max_s)
        r["doo_retry_s"] = retry
        r["hold_until"] = max(r["hold_until"], now + retry)
        r["doo_until"] = r["hold_until"]
        self._doo_dirty = True
        label = self._label_for(key)
        ladder = (f" instead of every {_fmt_s(self.max_backoff_s)}"
                  if self.max_backoff_s > 0.0 else "")
        lines.append(
            f"  [camera-gate] {key} ({label}): its stream died within "
            f"{max(0.0, lived):.1f}s of opening - {n} opens in a row "
            f"have died within {win:.0f}s, so it drops off as soon as it "
            f"starts streaming (check its power supply). Retrying it every "
            f"{_fmt_s(retry)}{ladder} until a reopen streams past "
            f"{win:.0f}s; 'use {label} again' retries it now.")
        if key not in self._doo_said:
            # ONCE per device per session: a repeat is logged, not spoken.
            # (With doo_state_path, once per DIES_ON_OPEN_SAY_AGAIN_S across
            # restarts: _doo_load re-seeds this set.)
            self._doo_said.add(key)
            self._doo_said_at[key] = now
            spoken.append(f"{label[:1].upper()}{label[1:]} drops off USB the "
                          f"moment it starts streaming, sir. That is usually "
                          f"its power supply. I'll only retry it every "
                          f"{minutes_phrase(retry)}.")

    def _evaluate_locked(self, now: float, lines: list, spoken: list) -> None:
        # drops: (t, key, kind, onset); fails: (t, key, onset)
        dw = self.storm_drop_window_s
        while self._drops and (now - self._drops[0][0]) > dw:
            self._drops.popleft()
        fw = self.storm_fail_window_s
        while self._fails and (now - self._fails[0][0]) > fw:
            self._fails.popleft()
        cams = sorted({d[1] for d in self._drops if d[2] == KIND_CAMERA})
        audio = sorted({d[1] for d in self._drops if d[2] == KIND_AUDIO})
        d_onset = min((d[3] for d in self._drops), default=now)
        if len(cams) >= 2:
            self._trip_locked(now, f"{len(cams)} cameras dropped within "
                                   f"{dw:.0f}s ({', '.join(cams)})",
                              lines, spoken, onset=d_onset)
            return
        if cams and audio:
            self._trip_locked(now, f"a camera and an audio endpoint dropped "
                                   f"within {dw:.0f}s ({cams[0]}, {audio[0]})",
                              lines, spoken, onset=d_onset)
            return
        fkeys = sorted({f[1] for f in self._fails})
        if len(self._fails) >= self.storm_fail_count and len(fkeys) >= 2:
            self._trip_locked(now, f"{len(self._fails)} camera open failures "
                                   f"across {len(fkeys)} devices within "
                                   f"{fw:.0f}s ({', '.join(fkeys)})",
                              lines, spoken,
                              onset=min(f[2] for f in self._fails))

    # ── the decision ──────────────────────────────────────────────────────
    def _decide_locked(self, key: str, component: str, now: float,
                       lockers_now, present_now=None,
                       lines: "list | None" = None) -> Decision:
        r = self._rec(key)
        lines = [] if lines is None else lines
        if r["quarantined"]:
            return Decision(False, "quarantined", QUARANTINE_POLL_S,
                            f"quarantined: {r['quarantine_why']}")
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
            if r["doo_retry_s"] > 0.0 and r["hold_until"] == r["doo_until"]:
                return Decision(False, "backoff", r["hold_until"] - now,
                                f"its last {r['doo_count']} opens each died "
                                f"within {self.dies_on_open_window_s:.0f}s - "
                                f"retrying every {_fmt_s(r['doo_retry_s'])}")
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
                # When THIS open started: the culprit window is measured from
                # it, and a failed open's device-gone drop dates from it.
                began = (r["in_flight_since"] if r["in_flight"] == component
                         and r["in_flight_since"] else
                         (r["last_open_at"] or now))
                if r["in_flight"] == component:
                    r["in_flight"] = ""
                if gone:
                    self._mark_absent_locked(key, now, lines)
                if ok:
                    r["last_ok_at"] = now
                    r["last_ok_by"] = component
                    r["locked_by"] = ()
                    r["locked_until"] = 0.0
                    # A NEW STREAM (v2.0.132): its failures are a new drop,
                    # not the tail of the last episode - a device reopened
                    # after a drop that fails again must count again.
                    r["dropped"] = False
                    r["stream_begin_at"] = min(began, now)
                    r["stream_ok_at"] = now
                    r["opens_ok"] += 1
                    if r["recovering"] and self._staggered(key):
                        self._reopen_at = now
                        self._reopen_key = key
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
                    self._fails.append((now, key, min(began, now)))
                # A device that has STREAMED this session and whose open now
                # fails with it gone from the device list has VANISHED - a
                # drop, exactly like one reported from a running stream (the
                # 2026-09-29 18:48:58 line), and counted once per open ATTEMPT
                # (an absent device is not re-attempted, so this cannot
                # repeat). One never seen streaming that is not listed is
                # simply not plugged in: not a bus event.
                if gone and r["opens_ok"] > 0:
                    r["dropped"] = True
                    r["recovering"] = True
                    self._count_drop_locked(
                        now, key, KIND_CAMERA, min(began, now),
                        "its open failed and it is gone from the device list",
                        True, lines, spoken)
                elif count:
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
                # DIES ON OPEN (R11): a frame later than the window after
                # this stream's open proves it streamed normally - once per
                # stream, and never for a stream whose drop was already
                # judged (a frame between a drop and the next successful
                # open belongs to no live stream).
                if (self._doo_on_locked() and r["stream_ok_at"]
                        and r["doo_judged"] != r["opens_ok"]
                        and (now - r["stream_ok_at"])
                        > self.dies_on_open_window_s):
                    r["doo_judged"] = r["opens_ok"]
                    self._doo_clear_locked(
                        key, r, f"streaming {_fmt_s(now - r['stream_ok_at'])}"
                                f" after its open", lines)
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
                  now: "float | None" = None, *,
                  onset: "float | None" = None, cause: str = "",
                  minor: bool = False) -> bool:
        """A device that was delivering has stopped (a camera's read-failure
        escalation, the Kinect's stale-stream reset, an audio endpoint that
        vanished). Counted ONCE per stream per device - a camera failing for a
        minute is one drop, not 600, but a camera REOPENED since (a successful
        end()) that fails again is a new drop. The next open of that device is
        a RECOVERY and spends a backoff rung. A full camera drop whose onset
        is within dies_on_open_window_s of the device's last successful open
        also counts toward DIES ON OPEN (the slow retry).

        ``onset``  when the trouble STARTED (the first failed read of the
                   burst), if the caller knows; the culprit and dies-on-open
                   windows are measured against it. Clamped to
                   [now - 60 s, now].
        ``cause``  a few words for the log line ("read-failure burst").
        ``minor``  ONE failed read, not a burst (a side tile's read that
                   failed once): it still counts toward the two-device storm
                   rules, but unless the camera is also gone from the device
                   list it never re-trips the breaker ALONE on probation. A
                   later full burst on the same stream still counts.

        Returns True iff this call tripped the breaker. NEVER raises."""
        try:
            now = self._clock() if now is None else now
            try:
                o = now if onset is None else float(onset)
                o = max(now - 60.0, min(o, now)) if o == o else now
            except Exception:
                o = now
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
                    # dropped: False / "minor" (one failed read counted) /
                    # True (a full drop counted). A full drop on a stream
                    # whose only report so far was minor still counts once.
                    if r["dropped"] is True or (r["dropped"] and minor
                                                and not gone):
                        already = True
                    else:
                        r["dropped"] = "minor" if (minor and not gone) else True
                        r["recovering"] = True
                        r["healthy_since"] = 0.0
                if not already:
                    tripped = self._count_drop_locked(
                        now, key, kind, o,
                        cause or ("read failures" if kind == KIND_CAMERA
                                  else "audio endpoint vanished"),
                        bool(gone), lines, spoken, minor=bool(minor))
                    # DIES ON OPEN (R11): a FULL drop of a device judges the
                    # stream it ended. One failed read of a camera still on
                    # the device list is not a dead stream.
                    if kind == KIND_CAMERA and (gone or not minor):
                        self._dies_on_open_locked(key, r, now, o,
                                                  lines, spoken)
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

    def quarantined(self, key: "str | None" = None):
        """With ``key``: is that device quarantined? Without: {key: spoken
        label} of every quarantined device. NEVER raises."""
        try:
            with self._lock:
                if key is not None:
                    r = self._dev.get(key)
                    return bool(r and r["quarantined"])
                return {k: (r["quarantine_label"] or self._label_for(k))
                        for k, r in self._dev.items() if r["quarantined"]}
        except Exception:
            return False if key is not None else {}

    def dies_on_open(self, key: "str | None" = None):
        """With ``key``: is that device on the slow DIES-ON-OPEN retry? Without:
        {key: spoken label} of every such device. NEVER raises."""
        try:
            with self._lock:
                if key is not None:
                    r = self._dev.get(key)
                    return bool(r and r["doo_retry_s"] > 0.0)
                return {k: self._label_for(k)
                        for k, r in self._dev.items() if r["doo_retry_s"] > 0.0}
        except Exception:
            return False if key is not None else {}

    def lift_quarantine(self, key: str, now: "float | None" = None) -> bool:
        """The OWNER says a device may be used again: a QUARANTINED one (it has
        been moved to another port) - its strikes are cleared too, so it
        starts from zero - and/or one on the slow DIES-ON-OPEN retry (its
        power has been seen to), whose run is cleared and whose long hold is
        dropped so it may be opened at once. True iff either was lifted.
        NEVER raises. :meth:`lift` says WHICH was lifted."""
        return bool(self.lift(key, now))

    def lift(self, key: str, now: "float | None" = None) -> tuple:
        """:meth:`lift_quarantine`, answering WHICH hold it lifted: a tuple of
        LIFT_QUARANTINE ("quarantine") and/or LIFT_SLOW_RETRY
        ("dies-on-open"), in that order; () when neither was set. The owner's
        reply depends on it - a quarantine is switched off again if the hub
        drops out, a slow retry goes back to half-hourly if the device still
        dies on open. NEVER raises."""
        try:
            now = self._clock() if now is None else now
            lines: list = []
            with self._lock:
                r = self._dev.get(key)
                if r is None:
                    return ()
                lifted: list = []
                if r["doo_retry_s"] > 0.0 or r["doo_count"]:
                    slow = r["doo_retry_s"] > 0.0
                    # ONE TRY, AS PROMISED (2026-10-01). The reply to "use
                    # the Kinect again" says "if it still drops off, I'll go
                    # back to retrying it every thirty minutes" - but a zeroed
                    # count took THREE more deaths (now, then 2 and 5 minutes
                    # later on the ladder) before the slow retry came back.
                    # Kept one short of the verdict, the next death re-arms it
                    # at the base retry. A partial run with no slow retry
                    # still starts over.
                    r["doo_count"] = (max(0, self.dies_on_open_count - 1)
                                      if slow else 0)
                    r["doo_retry_s"] = 0.0
                    if slow:
                        lifted.append(LIFT_SLOW_RETRY)
                        r["hold_until"] = 0.0
                        self._doo_dirty = True
                        lines.append(
                            f"  [camera-gate] {key}: the owner asked to use it "
                            f"again - the slow dies-on-open retry is cleared; "
                            f"it may be opened now (through the usual gate).")
                if r["quarantined"]:
                    lifted.insert(0, LIFT_QUARANTINE)
                    r["quarantined"] = False
                    r["quarantined_at"] = 0.0
                    r["quarantine_why"] = ""
                    r["strikes"] = []
                    r["struck_open"] = r["opens_ok"]
                    lines.append(
                        f"  [camera-quarantine] {key}: lifted by the owner - it "
                        f"may be opened again (through the usual gate, one "
                        f"device at a time).")
            self._emit(lines, [])
            return tuple(lifted)
        except Exception:
            return ()

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
            lines: list = []
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
                        "quarantined": r["quarantined"],
                        "culprit_strikes": len([
                            t for t in r["strikes"]
                            if (now - t) < CULPRIT_MEMORY_S]),
                        "dies_on_open": r["doo_count"],
                        "slow_retry_s": r["doo_retry_s"],
                    }
                q_keys = [k for k, r in self._dev.items() if r["quarantined"]]
                slow = {k: {"label": self._label_for(k),
                            "count": r["doo_count"],
                            "retry_s": r["doo_retry_s"]}
                        for k, r in self._dev.items() if r["doo_retry_s"] > 0.0}
                active = bool(self._storm_until and now < self._storm_until)
                self._storm_tick_locked(now, lines)
                probation = self._probation_locked(now)
                out = {
                    "storm_active": active,
                    "storm_remaining_s": (self._storm_until - now) if active else 0.0,
                    "storm_reason": self._storm_reason if active else "",
                    "storm_trips": self._storm_trips,
                    "wedged": [w[0] for w in self._wedges.values()],
                    "storm_last_cooldown_s": self._storm_last_cooldown,
                    "storm_probation": probation,
                    "devices": devs,
                    "refusals": {f"{k}|{r}": n for (k, r), n
                                 in self._refusal_counts.items()},
                    "quarantined": {
                        k: {"label": (self._dev[k]["quarantine_label"]
                                      or self._label_for(k)),
                            "why": self._dev[k]["quarantine_why"]}
                        for k in q_keys},
                    "dies_on_open": slow,
                }
            self._emit(lines, [])
            return out
        except Exception:
            return {"storm_active": False, "devices": {}, "refusals": {},
                    "quarantined": {}, "dies_on_open": {}}
