"""
Runtime self-diagnostic + auto-queue-for-repair.

Periodically probes every JARVIS subsystem and produces a structured health
report. When a probe fails it: (a) WARN-logs the failure, (b) on HIGH
severity, fires proactive_announce so the user hears about it in real time,
(c) appends a self-healing task to ``jarvis_todo.md`` so the next overnight
upgrade pass picks it up and tries to repair the component.

This is distinct from ``tools/audit_codebase.py`` — that one is *static*
pre-deploy analysis (linting, dead-code detection, import sanity). This
module monitors the *running* system: hardware that vanished, services that
went unreachable, files that got corrupted, models that fell out of memory.

Probes (15 subsystems)
----------------------
    1.  webcam            — open device, grab a non-black frame, verify the
                            haar face cascade loads.
    2.  microphone        — sounddevice enumerates input devices AND the mic
                            is proven live: by a private capture when nothing
                            owns the device, otherwise passively from the
                            audio the main loop is already reading (see
                            _passive_mic_liveness). A sweep that can do
                            neither reports UNVERIFIED — never a pass.
    3.  tts               — edge-tts CDN reachable AND pyttsx3 initialises.
    4.  stt               — Whisper model loaded; tiny synthesized audio
                            roundtrips through transcribe without raising.
    5.  claude_api        — 1-token ``messages.create`` returns 200.
    6.  internet          — DNS resolves anthropic.com AND 1.1.1.1 pings.
    7.  hud_subprocesses  — jarvis_hud / workshop_hud / jarvis_reticle /
                            tray PIDs still alive.
    8.  state_files       — every .json in the project root parses cleanly.
    9.  bambu             — MQTT connect with 5s timeout (skipped when
                            BAMBU_PRINTER_IP unset).
   10.  media_playback    — Chrome reachable on disk; Apple Music if found.
   11.  skill_imports     — every skill on disk (flat .py AND package dirs)
                            is in load_skills' success set; falls back to a
                            parse check when run outside JARVIS.
   12.  gpu               — torch.cuda.is_available() when WHISPER_DEVICE
                            wants CUDA OR a local LLM is configured.
   13.  disk              — > 1 GB free on the project drive.
   14.  ram               — < 90% utilised.
   15.  optional_skills   — placeholder probes for Alexa (research-4a) and
                            Deco router (research-4c); pass-through when the
                            owning skill hasn't loaded yet.

Severity policy
---------------
Each probe assigns one of LOW / MED / HIGH on failure.

    HIGH  — core capability gone: mic, STT, Claude API, internet, disk full,
            RAM saturated, state file corrupted, skill imports failing.
            Speaks aloud + auto-queues a repair task + pushes to phone if
            phone_bridge is configured.
    MED   — degraded but functional: webcam, TTS (when one backend still
            works), one HUD down, Bambu unreachable, media playback target
            missing. Auto-queues a repair task; doesn't speak by default
            (announces only via the next ``run_diagnostic`` summary).
    LOW   — cosmetic / intermittent. Logged only — never auto-queued.

    UNKNOWN — the THIRD outcome: the probe could not perform its check at
            all (the device is owned by the main loop, a lock is held, the
            host module isn't loaded). Carries ``ok=False`` + ``tested=False``,
            lands in ``run["unverified"]`` rather than ``run["failed"]``, is
            WARN-logged every sweep, and is never spoken or auto-queued.
            2026-08-20: this exists because every "couldn't check" path in
            this file used to return ``ok=True``. The microphone one mattered
            most — under START_IN_STANDBY the main loop holds the mic across
            essentially every sweep, so a HIGH-severity subsystem was reported
            healthy in 100 of 100 recorded runs without the device ever being
            opened, including the 90 minutes JARVIS was deaf. A check that did
            not run must never render as a pass.

A subsystem that is legitimately ABSENT or OFF *by configuration* (no
ANTHROPIC_API_KEY, no Bambu printer configured, SKILLS_ENABLED=False) still
passes: "there is nothing here to be broken" is a verified statement about the
config, not an unverified claim about hardware.

Persistence
-----------
Results land in ``data/self_diagnostic.json`` as a list of timestamped
runs, trimmed to the last ``MAX_HISTORY_RUNS`` (default 100). Each run is
a dict::

    {
        "ts": 1716937200.5,
        "iso": "2026-05-28T14:00:00",
        "duration_ms": 4123,
        "probes": {
            "webcam":    {"ok": True,  "latency_ms": 412, "error": None, ...},
            "microphone":{"ok": False, "latency_ms": 22,  "error": "...", ...},
            ...
        },
        "failed": ["microphone"],
        "severity_failed": {"microphone": "HIGH"},
        "unverified": ["webcam"],      # checks that could not run at all
    }

Schedule
--------
When ``core.scheduler`` is available, ``register()`` installs an interval
job that fires ``run_diagnostic`` every ``DEFAULT_INTERVAL_MINUTES`` (30 min
by default). The first sweep also runs ``ON_BOOT_DELAY_SECONDS`` after load
to surface cold-boot regressions.

Voice triggers (registered actions)
-----------------------------------
    run_diagnostic / system_check / are_you_ok    — fire immediate run.
    diagnostic_status                              — terse last-run summary.
    whats_broken                                   — read open self-diag tasks.
    diagnostic_history [N]                         — last N runs summary.
    last_diagnostic_run                            — raw JSON of last run.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import queue
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Optional

# Project root on sys.path so `core.atomic_io` and `bobert_companion`
# resolve whether we're loaded as `skills.self_diagnostic` or standalone.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:  # pragma: no cover - import-time sys.path guard; root already on path under the test harness
    sys.path.insert(0, _PROJECT_DIR)

try:
    from core.atomic_io import _atomic_write_json
except Exception:  # pragma: no cover — boot-order safety
    import tempfile

    def _atomic_write_json(path, data, *, indent=2):
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(dir_, exist_ok=True)
        fd: int = -1
        tmp: str | None = None
        try:
            fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = -1   # fdopen took ownership of the descriptor
                json.dump(data, f, indent=indent)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
            tmp = None
        except Exception:
            if fd >= 0:
                try:
                    os.close(fd)
                except Exception:
                    pass
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
            raise


try:
    from core.failure_markers import FAILURE_MARKERS
except Exception:  # pragma: no cover - boot-order safety; core/ is in-tree
    FAILURE_MARKERS = ("could not", "failed", "refused", "couldn't", "can't",
                       "didn't", "wouldn't", "unknown ", "format:")

_log = logging.getLogger("jarvis.self_diagnostic")

# ─── Config ──────────────────────────────────────────────────────────────
MAX_HISTORY_RUNS         = 100
DEFAULT_INTERVAL_MINUTES = 30
ON_BOOT_DELAY_SECONDS    = 60
PER_PROBE_TIMEOUT_S      = 15.0         # hard cap per probe — keeps a hung
                                        # probe from blocking the whole sweep.
                                        # Bumped 8→15 (2026-05-30) after a
                                        # transient ~8s claude_api blip
                                        # tripped the cap: typical probe
                                        # finishes in 1–2 s and p95 across
                                        # 99 successful runs is 3.2 s, so
                                        # 15 s gives headroom for the rare
                                        # HTTPS-handshake stall without
                                        # masking a genuinely-hung probe.
HIGH_SEVERITY_SPEAK      = True         # proactive_announce on HIGH failures
HIGH_SEVERITY_PHONE      = True         # also push_to_phone when configured
DISK_FREE_FLOOR_BYTES    = 1 * 1024 * 1024 * 1024   # 1 GB
RAM_PCT_CEILING          = 90.0
MIC_RMS_FLOOR            = 0.0005       # mic noise floor — quiet room is ~0.001

SEVERITY_LOW             = "LOW"
SEVERITY_MED             = "MED"
SEVERITY_HIGH            = "HIGH"
# The third outcome — NOT a failure severity. Attached to results whose probe
# could not perform its check; _run_all_probes routes these to
# run["unverified"] instead of run["failed"], so they are never announced and
# never queued for repair. See _unverified().
SEVERITY_UNKNOWN         = "UNKNOWN"

# Passive mic-liveness window. The capture loop publishes the raw RMS of every
# chunk it reads through core.audio_processor (note_vad_poll / note_raw_rms),
# so the probe can read a liveness signal WITHOUT opening a competing stream.
# A poll older than this means the loop isn't reading chunks right now, so its
# silence data says nothing. Deliberately the same 30 s window
# _collect_vad_stall_signal uses for the identical "is the input loop actually
# running" question — one constant, not two (stale-duplicate rule). The
# silence threshold itself is NOT redefined here: it is
# core.config.MIC_SILENT_WARN_SECONDS, the same value record_speech's
# silent-mic warning uses, judged against core.audio_processor's own
# _AUDIBLE_RMS_FLOOR.
_PASSIVE_POLL_FRESH_S    = 30.0

# ─── Spoken-surface wording rule (2026-08-20 review, HIGH) ───
# run_diagnostic / are_you_ok / system_check / self_diagnostic /
# diagnostic_status / diagnostic_history are ALL in
# bobert_companion.SPEAK_RESULT_VERBATIM_ACTIONS and in NEITHER
# INFORMATIVE_ACTIONS — so their return string is read back to the owner
# verbatim... unless it contains a core.failure_markers.FAILURE_MARKERS
# substring, in which case _speak_verbatim_results DROPS it (`continue`) and
# _is_failure re-routes it through the follow-up loop as a failed action: a
# whole extra local-LLM round-trip that re-words the sentence, and can drop the
# UNVERIFIED distinction entirely.
#
# The UNVERIFIED wording that shipped this morning read "N check(s) couldn't
# run" and "I can't call the system nominal" — TWO markers — so the flagship
# sentence of v2.0.95 was never spoken as written. Same trap
# core/smart_home_router.py:920-930 already documents by name ("can't read its
# live state" used to match "can't").
#
# RULE: no string this module emits on the HONEST (non-failure) surface may
# contain a FAILURE_MARKER. Say "did not run" instead of "couldn't run", and
# "I am not able to" instead of "I can't". Genuine failure text (whats_broken's
# "I couldn't scan jarvis_todo.md") KEEPS its marker — that is what puts it on
# the failure path. tests/skills/test_self_diagnostic.py enforces both halves.

# Short, speakable causes attached to an UNVERIFIED result so the summary can
# say WHY a check did not run instead of only naming it — "microphone (no audio
# frames to judge yet)" is actionable; a bare "microphone" is alarming. One
# table, not per-call-site literals, so the marker-free rule above can be
# enforced by a single test over the values.
_UNVERIFIED_SHORT_CAUSES: dict[str, str] = {
    "mic_no_frames":   "no audio frames to judge yet",
    "mic_disabled":    "the microphone is switched off in configuration",
    "mic_speaking":    "JARVIS was speaking at the time",
    "mic_disagree":    "the two microphone readings disagreed",
    "camera_busy":     "the face tracker had the camera",
    "stt_busy":        "a transcription was already in flight",
    "host_not_loaded": "the host module was not loaded",
}

# Per-subsystem default severity on failure. Overridable per-probe.
SUBSYSTEM_SEVERITY: dict[str, str] = {
    "webcam":           SEVERITY_MED,
    "microphone":       SEVERITY_HIGH,
    "tts":              SEVERITY_MED,
    "stt":              SEVERITY_HIGH,
    # Claude API is an OPTIONAL ENHANCEMENT, not a requirement — JARVIS runs
    # fully on the local Ollama model as its baseline. So a capped / absent
    # Claude API is LOW severity: it must NOT trigger the spoken "Sir, the
    # Claude API appears to be down — I'll queue a fix" alert (it fired ×120 in
    # the logs) and must NOT auto-queue a self-heal task (LOW failures are
    # never queued). 2026-05-30, per user: credits are a bonus, not a need.
    "claude_api":       SEVERITY_LOW,
    "internet":         SEVERITY_HIGH,
    "hud_subprocesses": SEVERITY_MED,
    "state_files":      SEVERITY_HIGH,
    "bambu":            SEVERITY_LOW,
    "media_playback":   SEVERITY_LOW,
    "skill_imports":    SEVERITY_HIGH,
    "gpu":              SEVERITY_MED,
    "disk":             SEVERITY_HIGH,
    "ram":              SEVERITY_HIGH,
    "optional_skills":  SEVERITY_LOW,
}

_HISTORY_PATH = os.path.join(_PROJECT_DIR, "data", "self_diagnostic.json")
_TODO_PATH    = os.path.join(_PROJECT_DIR, "jarvis_todo.md")

# ─── Auto-queue (self-healing pipeline → jarvis_todo.md) ─────────────────
# When the running system surfaces a repair-worthy condition that the
# probe-based reports don't already cover — repeated caught action
# failures, a VAD stall while JARVIS is supposed to be listening, or a
# face_tracker read-failure spike — we want a structured fix request
# appended to jarvis_todo.md so Claude Code (overnight_upgrade.py) can
# pick it up. This is what turns the self-healing pipeline from "log
# the problem and move on" into "actually feed Claude Code".
#
# Dedup is by signature with an 8-hour cooldown so we don't spam the
# same fix every sweep. State persists in data/self_diagnostic_autoqueue.json.
_AUTOQUEUE_PATH               = os.path.join(_PROJECT_DIR, "data",
                                             "self_diagnostic_autoqueue.json")
_AUTOQUEUE_COOLDOWN_S         = 8 * 3600          # don't requeue same sig <8h
_AUTOQUEUE_ERROR_GROUP_COUNT  = 3                 # ≥3 same-class errors in 1h
_AUTOQUEUE_ERROR_WINDOW_S     = 3600.0
_AUTOQUEUE_VAD_STALL_S        = 60.0              # poll fresh but no trip 60s+
_AUTOQUEUE_FACE_FAIL_THRESH   = 5                 # consecutive read failures
_AUTOQUEUE_LOG_TAIL_LINES     = 20                # session-log lines appended
_AUTOQUEUE_TRACEBACK_LINES    = 5                 # traceback excerpt size

# Threading: a single in-flight sweep at a time (probes do real I/O — a
# concurrent sweep just doubles the API cost without value).
_run_lock = threading.Lock()
_state: dict[str, Any] = {
    "last_run": None,            # the most-recent run dict
    "last_run_started_at": 0.0,
    "runs_completed": 0,
    "registered_at": time.time(),
}

# voice_mood layer hook — set by _announce_failures() when a HIGH-severity
# probe failure is announced. Read by get_recent_problem_flag() so other
# components (the holographic HUD, the voice_mood_selector) can tell that
# JARVIS *should* be sounding concerned right now. A simple float + lock so
# the announce path stays fast and the read path stays thread-safe.
_RECENT_PROBLEM_WINDOW_SEC = 600.0   # 10 min — long enough that a follow-up
                                     # reply ("yes, please queue the fix")
                                     # still lands in concerned_soft, short
                                     # enough that JARVIS doesn't keep
                                     # sounding worried after the issue
                                     # cleared.
_recent_problem_lock = threading.Lock()
_recent_problem_at: list[float] = [0.0]


def _mark_recent_problem(now: Optional[float] = None) -> None:
    with _recent_problem_lock:
        _recent_problem_at[0] = float(_now() if now is None else now)


def get_recent_problem_flag(now: Optional[float] = None) -> bool:
    """Return True iff a HIGH-severity probe failure was announced within
    the last _RECENT_PROBLEM_WINDOW_SEC. Wired into the voice_mood layer
    (core/voice_mood_selector) so the next utterance lands in
    `concerned_soft` while a real system problem is fresh.

    Thread-safe: reads the cached timestamp under _recent_problem_lock so
    a concurrent _announce_failures call can't tear the read.
    """
    with _recent_problem_lock:
        ts = _recent_problem_at[0]
    if ts <= 0.0:
        return False
    cur = float(_now() if now is None else now)
    return (cur - ts) <= _RECENT_PROBLEM_WINDOW_SEC


# ─── small helpers ───────────────────────────────────────────────────────
def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _today_iso_date() -> str:
    return datetime.now().date().isoformat()


def _result(ok: bool, latency_ms: float, *, error: str | None = None,
            details: dict | None = None, severity: str | None = None,
            tested: bool = True) -> dict:
    """Canonical probe-result shape. Probes return this so the aggregator
    doesn't have to special-case different keys.

    ``tested`` is the honesty bit: False means the probe could not perform its
    check, so ``ok=False`` there means "I don't know", not "it's broken".
    Build those through _unverified() rather than passing tested= by hand —
    it keeps the wording and the severity consistent.
    """
    return {
        "ok":         bool(ok),
        "tested":     bool(tested),    # False → UNVERIFIED, not a failure
        "latency_ms": round(float(latency_ms), 1),
        "error":      None if ok else (error or "unknown error"),
        "details":    details or {},
        "severity":   severity,        # may be None — aggregator fills default
    }


def _unverified(latency_ms: float, *, reason: str, details: dict | None = None,
                remedy: str | None = None, short_cause: str | None = None,
                transient: bool = False) -> dict:
    """Canonical UNVERIFIED result — the probe could NOT perform its check.

    Neither a pass nor a failure. ``ok`` is False so nothing downstream can
    mistake it for health; ``tested`` is False so _run_all_probes keeps it out
    of ``run["failed"]`` — which is what stops _announce_failures from speaking
    about it and _queue_repair_task from filing a repair for a check that never
    ran. Every one is WARN-logged by the aggregator and listed in
    ``run["unverified"]``, and _summarise / diagnostic_status refuse to say
    "all systems nominal" while one exists.

    2026-08-20 (H-flagship): before this existed, every "couldn't check" path
    in this file returned ``_result(True, …)``. The microphone probe's
    owned-device skip is the one that mattered — under START_IN_STANDBY the
    main loop holds the mic across essentially every sweep, so the probe
    reported a HIGH-severity subsystem healthy in 100 of 100 recorded runs
    without ever opening the device, including the 90 minutes JARVIS was deaf.
    The one safety net that should have caught that was reporting success it
    never verified.
    """
    msg = f"UNVERIFIED — {reason}"
    if remedy:
        if msg[-1] not in ".!?":
            msg += "."
        msg += f" {remedy}"
    d = dict(details or {})
    d["tested"] = False
    d["unverified_reason"] = reason
    # short_cause: a few marker-free words the SUMMARY may speak (see
    # _UNVERIFIED_SHORT_CAUSES). transient: this condition is expected to clear
    # on its own — the check had no data at that instant, not a standing
    # problem — so a reader may RE-MEASURE it live rather than repeat a
    # half-hour-old "UNVERIFIED" until the next sweep. Re-measure, never
    # carry-forward: a stale pass is the sin this release exists to fix.
    if short_cause:
        d["unverified_short_cause"] = short_cause
    if transient:
        d["unverified_transient"] = True
    return _result(False, latency_ms, error=msg, details=d,
                   severity=SEVERITY_UNKNOWN, tested=False)


def _bc():
    """Lazily resolve the bobert_companion module — returns None when
    JARVIS isn't fully imported yet (rare, but happens during pytest)."""
    return sys.modules.get("bobert_companion")


def _run_with_timeout(fn: Callable[[], dict], timeout_s: float, *, name: str) -> dict:
    """Run a probe with a hard timeout. The probe runs in a daemon thread
    because most failure modes here are I/O-bound (sockets, subprocesses,
    Whisper). Returns a timeout result if the probe doesn't finish in time."""
    box: dict[str, Any] = {"result": None, "exc": None}
    start = _now()

    def _runner():
        try:
            box["result"] = fn()
        except Exception as e:
            box["exc"] = e
            box["tb"]  = traceback.format_exc()

    t = threading.Thread(target=_runner, name=f"probe-{name}", daemon=True)
    t.start()
    t.join(timeout_s)
    elapsed_ms = (_now() - start) * 1000.0

    if t.is_alive():
        # Probe still running — we just abandon the thread (it's daemonized
        # so it can't block process exit; the next sweep starts a new one).
        return _result(False, elapsed_ms,
                       error=f"probe timed out after {timeout_s:.1f}s")

    if box["exc"] is not None:
        return _result(False, elapsed_ms,
                       error=f"{type(box['exc']).__name__}: {box['exc']}")

    if not isinstance(box["result"], dict):
        return _result(False, elapsed_ms,
                       error="probe returned non-dict result")
    return box["result"]


# ─── Probe 1: webcam ─────────────────────────────────────────────────────
_CAMERA_LOCK_PROCESSES_FALLBACK = {
    "teams.exe", "ms-teams.exe", "msteams.exe",
    "zoom.exe", "cpthost.exe",
    "obs64.exe", "obs32.exe", "obs.exe",
    "skype.exe", "skypeapp.exe",
    "discord.exe", "discordcanary.exe", "discordptb.exe",
    "webex.exe", "webexmta.exe", "atmgr.exe",
    "slack.exe",
    "googlemeet.exe", "meet.exe",
    "manycam.exe", "snapcamera.exe", "facerig.exe", "vmix.exe",
    "logi capture.exe", "logitune.exe", "logioptionsplus.exe",
    "windowscamera.exe", "cameraapp.exe",
    "nvbroadcast.exe", "nvidia broadcast.exe",
}


def _camera_lock_suspects() -> list[str]:
    """Best-effort list of running processes known to hold exclusive webcam
    locks (Teams, Zoom, OBS, Discord, ...). Prefers
    ``bobert_companion.find_camera_locking_processes`` when the parent
    module is already loaded so we share its CAMERA_LOCK_PROCESSES set;
    falls back to a local scan when self_diagnostic runs standalone (e.g.
    early-boot probe before bobert_companion has finished importing, or
    standalone unit tests). Returns [] when psutil is missing.
    """
    bc = _bc()
    finder = getattr(bc, "find_camera_locking_processes", None) if bc else None
    if callable(finder):
        try:
            res = finder()
            if isinstance(res, list):
                return res
        except Exception:
            pass

    try:
        import psutil  # type: ignore
    except Exception:
        return []
    suspects: list[str] = []
    try:
        for proc in psutil.process_iter(attrs=["name"]):
            try:
                raw = proc.info.get("name") or ""
                if raw.lower() in _CAMERA_LOCK_PROCESSES_FALLBACK and raw not in suspects:
                    suspects.append(raw)
            except Exception:
                continue
    except Exception:
        return []
    return suspects


def _windows_camera_hardware_count() -> int | None:
    """Ask Windows PnP whether any camera-class device exists at all.

    Returns the count of present camera-class devices, or None when the
    query can't run (non-Windows, missing PowerShell, timeout). Used to
    distinguish "webcam is broken" (hardware present, probe failed) from
    "no webcam plugged in" (hardware absent — expected on desktops without
    one). A genuinely-absent webcam shouldn't keep auto-queueing a repair
    task every 30 minutes, so we downgrade its severity to LOW.
    """
    if sys.platform != "win32":
        return None
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "(Get-PnpDevice -Class Camera,Image -PresentOnly -ErrorAction "
             "SilentlyContinue | Measure-Object).Count"],
            capture_output=True, text=True, timeout=4.0,
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if sys.platform == "win32" else 0),
        )
        if proc.returncode != 0:
            return None
        out = (proc.stdout or "").strip()
        if not out:
            return 0
        return int(out.splitlines()[-1].strip())
    except Exception:
        return None


def _windows_camera_pnp_devices() -> list[dict] | None:
    """Return per-device PnP info for every camera-class endpoint.

    Each entry has ``FriendlyName``, ``Status``, ``Class``, ``Problem``.
    ``Status`` is the PnP-level state (``OK``, ``Error``, ``Degraded``,
    ``Unknown``); ``Problem`` is the device-manager problem code (0 =
    none, 22 = disabled, 24 = device not present, 28 = drivers not
    installed, 43 = device reported a problem, etc.). This lets us tell
    the difference between:

      * device present + Status=OK + read() fails → stalled USB pipe or
        power-save (environmental, no code fix possible from here).
      * device present + Status=Error / Problem!=0 → driver/HW issue,
        still environmental.
      * device absent / Problem=24 → hardware unplugged, expected.

    Returns None when the query can't run (non-Windows, PowerShell
    missing, timeout, JSON parse failure). Returns [] when PnP confirms
    no camera devices are present.
    """
    if sys.platform != "win32":
        return None
    # PowerShell: emit one JSON object per camera device. ``ConvertTo-Json``
    # with a single element returns a bare object instead of an array, so
    # we force an array with ``@()``.
    ps_script = (
        "$d = @(Get-PnpDevice -Class Camera,Image -ErrorAction SilentlyContinue "
        "| Select-Object FriendlyName, Status, Class, Problem, Present); "
        "$d | ConvertTo-Json -Compress -Depth 3"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True, text=True, timeout=4.0,
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if sys.platform == "win32" else 0),
        )
        if proc.returncode != 0:
            return None
        raw = (proc.stdout or "").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except Exception:
            return None
        # ConvertTo-Json returns a bare object when there's exactly one
        # device (we forced @() above but defensively handle both shapes).
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return None
        cleaned: list[dict] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            cleaned.append({
                "name":     str(entry.get("FriendlyName") or "").strip(),
                "status":   str(entry.get("Status") or "").strip(),
                "class":    str(entry.get("Class") or "").strip(),
                "problem":  entry.get("Problem", 0),
                "present":  bool(entry.get("Present", False)),
            })
        return cleaned
    except Exception:
        return None


def _camera_pnp_diagnosis(devices: list[dict] | None) -> dict:
    """Summarise PnP device list into a flat diagnosis dict.

    Returns keys:
      * ``hardware_present`` (bool): True if any camera-class device is
        present (regardless of status).
      * ``healthy_devices`` (int): count of devices with Status=OK and
        Problem=0.
      * ``has_problem_device`` (bool): True when at least one camera-class
        device is present but in a non-OK Status / non-zero Problem.
      * ``failure_mode`` (str): one of ``"absent"``, ``"problem"``,
        ``"ok"``, ``"unknown"`` — easy single field for the diagnostic
        report.
      * ``summary`` (str): human-readable summary suitable for the task
        body.
    """
    if devices is None:
        return {"hardware_present": False, "healthy_devices": 0,
                "has_problem_device": False, "failure_mode": "unknown",
                "summary": "PnP query unavailable"}
    present = [d for d in devices if d.get("present")]
    healthy = [d for d in present
               if (d.get("status") or "").lower() == "ok"
               and not d.get("problem")]
    problem = [d for d in present
               if (d.get("status") or "").lower() != "ok"
               or d.get("problem")]
    if not present:
        return {"hardware_present": False, "healthy_devices": 0,
                "has_problem_device": False, "failure_mode": "absent",
                "summary": "no camera-class devices present in PnP"}
    if problem and not healthy:
        first = problem[0]
        return {
            "hardware_present": True, "healthy_devices": 0,
            "has_problem_device": True, "failure_mode": "problem",
            "summary": (f"camera device '{first.get('name') or '?'}' "
                        f"reports Status={first.get('status') or '?'} "
                        f"Problem={first.get('problem')}"),
        }
    return {
        "hardware_present": True, "healthy_devices": len(healthy),
        "has_problem_device": bool(problem), "failure_mode": "ok",
        "summary": (f"{len(healthy)} healthy camera device(s) per PnP; "
                    f"{len(problem)} problem"),
    }


# Seconds the index SCAN waits for a real frame from one index before moving
# on. Not a single read: a healthy camera can return False for its first reads
# after a cold open, and on MSMF a device another process holds reports
# isOpened() True and then never produces anything at all.
#
# SIZED AGAINST THE LOCK, not just the camera. The scan runs while holding
# bobert_companion._camera_io_lock, and the face-track producer waits on that
# lock, so the worst case that matters is "all three indices open and none
# delivers": 3 x (open + this budget). Measured on this rig 2026-09-05, an MSMF
# open with 1280x720 set on it costs ~0.5 s, and a frame arrived 0.30-0.42 s
# after the sets on BOTH webcams in every cycle of a 25-cycle bench. So:
#   1.5 s is ~3.5x the measured worst first-frame latency, and
#   3 x (0.5 + 1.5) = 6.0 s is the worst lock hold, comfortably under the 30 s
#   _FACE_TRACK_STALL_WARN_S that would otherwise blame a healthy producer.
# At 2.5 s that worst case was 9.0 s of held lock to buy margin nothing had
# asked for.
_CAMERA_PROBE_WARMUP_S = 1.5


def _camera_backend_name() -> str:
    """'msmf' or 'dshow' — the ONE backend this whole probe uses.

    Both halves of the webcam probe must agree, because the two backends do not
    index the same device list. See the block in _probe_webcam_locked."""
    try:
        from core.camera_backend import configured_backend
        return configured_backend()
    except Exception:
        # core unavailable (bare-skill import / CI). MSMF is what a
        # backend-less cv2.VideoCapture(idx) resolves to on Windows anyway, so
        # this default keeps the two halves consistent rather than silently
        # reintroducing the DirectShow half of the mismatch.
        return "msmf"


def _open_probe_capture(idx: int, backend: str,
                        require_frame: float = 0.0):
    """Open ``idx`` on ``backend``, or None. Never raises.

    ``require_frame`` > 0 makes the open PROVE the device delivers before
    handing the handle back. The SCAN needs that — under MSMF, index 0 on this
    rig is the Kinect's Media Foundation interface, which opens at 512x424 and
    then fails every read, so "first index that opens" picked a camera that can
    never work and the probe took the soft-wake branch every 30 minutes
    forever. The WAKE does not need it and must not use it: the wake reads the
    device itself, and that read IS its verdict — proving a frame at open too
    would just consume one and prove the same thing twice."""
    try:
        from core.camera_backend import open_camera
    except Exception:
        open_camera = None
    if open_camera is not None:
        return open_camera(idx, backend=backend, require_frame=require_frame)
    try:  # pragma: no cover - only when core/ is not importable
        import cv2  # type: ignore
        api = cv2.CAP_DSHOW if backend == "dshow" else cv2.CAP_MSMF
        c = cv2.VideoCapture(idx, api)
        if c is not None and c.isOpened():
            return c
        if c is not None:
            c.release()
    except Exception:
        pass
    return None


def _attempt_camera_wake(idx: int, timeout_s: float = 2.5,
                         backend: "str | None" = None) -> tuple[bool, str]:
    """Try a soft wake of camera ``idx``: release + brief sleep + reopen +
    read with warmup. Runs inside its own thread with a hard wall-clock
    timeout so a wedged open can't block the diagnostic sweep.

    ``backend`` MUST be the same backend the caller used to find ``idx``.
    It used to be hard-wired to cv2.CAP_DSHOW while its only caller found the
    index with a backend-less (MSMF) open, so the two disagreed about which
    physical camera the integer named — see _probe_webcam_locked. Defaults to
    the configured backend rather than to DirectShow, because defaulting to
    DirectShow is exactly the bug.

    Returns (success, note). Uses the same _camera_io_lock as the face-
    tracking thread when bobert_companion is loaded, so a wake here can't
    collide with the running tracker's release/reopen.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        return False, f"opencv unavailable: {e}"
    # AVAILABILITY CHECK ONLY. This function no longer touches cv2 itself — the
    # open goes through _open_probe_capture — but "opencv unavailable" is still
    # the honest verdict when it cannot be imported, and it has to be checked
    # before the worker starts or the failure would surface as a timeout.
    # Dropping the name keeps pyflakes honest about that (a bare `# noqa` would
    # not: pyflakes does not read noqa).
    del cv2
    backend = backend or _camera_backend_name()
    box: dict[str, Any] = {"ok": False, "note": ""}

    bc = _bc()
    io_lock = getattr(bc, "_camera_io_lock", None) if bc else None

    def _do_wake():
        cap = None
        acquired = False
        try:
            if io_lock is not None:
                # BOUNDED ACQUIRE (2026-07-14 audit). This was a plain
                # io_lock.acquire() with no timeout. Only the CALLER's join is
                # bounded (line ~601); the worker itself would block forever
                # waiting for the tracker to release the lock. Worse, when it
                # finally won the lock — long after the caller had given up and
                # moved on — it would open the camera anyway, stealing the
                # device from whatever the caller did next. Bound the worker's
                # own wait to the same budget; if it can't get the lock in time,
                # report contention and do NOT touch the device.
                acquired = io_lock.acquire(timeout=timeout_s)
                if not acquired:
                    box["note"] = "wake skipped — camera lock held by tracker"
                    return
            try:
                # First release any prior handle the face-tracker had open by
                # opening a fresh one. This used to double as a "did we fail to
                # claim the device?" probe on the strength of DirectShow
                # refusing a second open — that inference DOES NOT HOLD on
                # MSMF. Measured 2026-09-05 with one process holding the eMeet
                # and reading it: a DirectShow contender got isOpened() ==
                # False in 0.65 s, while an MSMF contender got isOpened() ==
                # True, set(W/H) == True, read 1280x720 back out of get(), and
                # then produced 0 frames out of 20. So the FRAME, never the
                # open, is what says the device is ours and alive — which is
                # why the verdict below is the read's and not isOpened()'s.
                # (require_frame is left at 0 here on purpose: this function
                # reads the device itself two lines down, and proving a frame
                # inside the open as well would consume one to learn the same
                # thing twice.)
                cap = _open_probe_capture(idx, backend)
                if cap is None:
                    box["note"] = f"wake reopen failed — device refused open ({backend})"
                    return
                # Brief warmup — generic UVC cameras commonly return False
                # on the very first read after a cold open.
                try:
                    cap.read()
                except Exception:
                    pass
                time.sleep(0.1)
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    box["ok"] = True
                    box["note"] = "wake succeeded — device produced a frame"
                else:
                    box["note"] = "wake reopened but read still returned no frame"
            finally:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:  # pragma: no cover - defensive: cv2 VideoCapture.release() failing during wake teardown (live-camera I/O, cv2 absent on CI)
                    pass
        finally:
            if io_lock is not None and acquired:
                try:
                    io_lock.release()
                except Exception:
                    pass

    t = threading.Thread(target=_do_wake, name=f"diag-wake-{idx}", daemon=True)
    t.start()
    # Join with a small margin OVER the worker's own lock budget (timeout_s) so
    # that on contention the worker's specific verdict ("lock held by tracker")
    # is observed, rather than this join firing at the same instant and
    # reporting the generic "timed out". 2026-07-14 audit.
    t.join(timeout=timeout_s + 0.5)
    if t.is_alive():
        return False, f"wake attempt timed out after {timeout_s:.1f}s"
    return bool(box["ok"]), str(box["note"] or "unknown wake outcome")


# Module-level state for one-time / cooldown announcements (so a 30-min
# sweep cadence doesn't spam the user with the same alert).
_announce_cooldown: dict[str, float] = {}
_ANNOUNCE_COOLDOWN_S = 6 * 3600   # don't re-announce the same condition
                                  # more than once per 6 hours

# Per-component last-announced failure signature, keyed by component name
# (e.g. "claude_api") → the error string we last spoke about. Used by
# _announce_failures() to suppress re-announcing a HIGH probe that keeps
# failing for the same reason every sweep (a known, dated outage shouldn't
# be spoken ×120 over a day). A component clears from re-announcement only
# when its error text changes; recovery is implicitly handled because a
# passing probe never reaches the announce path.
_announced_failure_state: dict[str, str] = {}


def _maybe_announce_once(key: str, message: str) -> None:
    """Proactively announce a condition (hardware unplugged, etc.) at most
    once per cooldown window. Safe to call from a probe — silently no-ops
    when bobert_companion isn't loaded yet."""
    last = _announce_cooldown.get(key, 0.0)
    if (_now() - last) < _ANNOUNCE_COOLDOWN_S:
        return
    _announce_cooldown[key] = _now()
    _proactive_announce(message)


class _CameraLockHold:
    """Bounded, idempotent hold on bobert_companion._camera_io_lock for the
    webcam probe. ``release()`` is safe to call more than once — the probe
    drops the lock EARLY before delegating to _attempt_camera_wake (whose
    worker acquires the same lock from a different thread, diag-wake-N;
    RLock reentrancy is per-thread, so holding on here would make every
    wake falsely time out with "lock held by tracker"), while the caller's
    ``finally`` still guarantees release on every other path. A None lock
    (bobert_companion not loaded — pytest, standalone runs) degrades to a
    no-op, exactly as in _attempt_camera_wake."""

    def __init__(self, lock) -> None:
        self._lock = lock
        self.held = False

    def acquire(self, timeout_s: float) -> bool:
        if self._lock is None:
            return True
        self.held = bool(self._lock.acquire(timeout=timeout_s))
        return self.held

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        try:
            self._lock.release()
        except Exception:  # pragma: no cover - defensive: foreign lock object refusing release
            pass


# ─── who owns the cameras ───────────────────────────────────────────────
#
# _camera_io_lock SERIALISES OPEN/RELEASE. IT DOES NOT MEAN THE DEVICE IS FREE.
# The face-tracking producer opens its cameras once and then reads from them
# for the life of the process, OUTSIDE that lock on purpose — holding it across
# a blocking cap.read() is what wedged the whole camera subsystem for two hours
# (v2.0.100). So a probe that takes the lock and then opens an index the
# producer is streaming gets a SECOND HANDLE ON A LIVE DEVICE.
#
# WHAT THAT COSTS, measured in-process on this rig 2026-09-06 with JARVIS
# stopped — one producer thread holding MSMF index 1 ("USB 2.0 Camera", the
# owner's right webcam) at 1280x720, 30 s windows either side of the event,
# CONTROL and PROBE arms identical but for the one extra open:
#
#   CONTROL (nothing else touches it)
#       before  828 reads / 0 fail / 30.0 fps   during 361 / 0 / 30.1 fps
#       after   539 reads / 0 fail / 30.0 fps
#   PROBE (one 3.1 s _probe_webcam_locked-shaped scan of indices 0,1,2)
#       before  826 reads / 0 fail / 30.0 fps
#       during  3,704,096 FAILED reads /  4.4 fps
#       after   5,411,405 FAILED reads /  0.0 fps  — and it never came back
#
# The probe won index 1: the producer's own camera. It did not merely slow the
# holder down, it ENDED the stream, permanently, for a device that was healthy.
#
# WHY THE MIGRATION'S EVIDENCE SAID THIS WAS SAFE. core/camera_backend.py's
# docstring records "Both backends leave the HOLDER undisturbed … so a partial
# migration is safe". That was measured ACROSS TWO PROCESSES, where MSMF does
# refuse the contender honestly — reproduced here, all three indices refused,
# zero disturbance to the holder. In-process it does not refuse, and in-process
# is the only configuration JARVIS runs in.
#
# WHY IT ONLY STARTED BITING WITH require_frame. Before the MSMF migration the
# scan accepted the first index that merely isOpened(), which under MSMF is
# index 0 — the Kinect — so it stopped one index SHORT of the producer's
# cameras every time. All 70+ recorded diagnostics from 2026-08-20 through
# 2026-09-05T20:25 carry details={'index': 0} with wake_attempted=True, and not
# one of them is followed by a face-track read failure inside the 8-10 s window
# that every post-migration sweep is (the pre-migration sessions DO contain read
# failures - they just fall nowhere near a sweep). Making the scan demand a real
# frame fixed the Kinect problem and walked it straight onto index 1. Every
# diagnostic from 2026-09-05T22:39 onward recorded {'index': 1, 'backend':
# 'msmf'}, and the session logs show a "[face-track] Right webcam (top of right
# monitor) (index 0) read failure #25" 8-17 s after EVERY one of them — NINE
# for nine, across three sessions:
#   session_2026-09-05_23-19-39  sweeps 23:21:05 23:50:05 00:20:05 00:50:05
#                                       01:20:05
#                                fails  23:21:14 23:50:14 00:20:15 00:50:13
#                                       01:20:13
#   session_2026-09-06_04-55-31  sweeps 04:57:04 05:26:04 05:56:04
#                                fails  04:57:13 05:26:12 05:56:13
#   session_2026-09-06_06-59-08  sweep  07:02:49   fail 07:03:06
# and at 04:32:56 the scan landed on index 2 instead, so that time it was the
# LEFT webcam that failed. No camera was safe.
#
# require_frame protects the CONTENDER from a useless handle. Nothing in it
# protects the HOLDER, and the damage is done by the open and the reads before
# the verdict is even reached. The only fix is not to open the device at all:
# ASK WHO OWNS IT FIRST.
#
# A camera the producer is reading from does not need a second handle to prove
# it works — the producer IS a continuous open-and-read test of that device.
# MEASURED LIVE 2026-09-06 by counting distinct writes of the shared HUD
# preview, which the producer emits once per loop iteration: 678 writes in
# 120 s = 6.4 fps median (5.7 mean), loop period median 0.156 s, p90 0.238 s,
# max 0.347 s. Several reads a second, forever, is strictly stronger evidence
# than one spot check every 30 minutes — and it is 30 fps ONLY in the
# measurement harness above, never here; do not copy that number across.
# So an owned-and-streaming camera is reported from the producer's own
# telemetry, and an owned-but-silent one is reported UNVERIFIED rather than
# opened. The genuinely-sick case is not lost: the producer's read-failure
# signal reaches the self-heal pipeline through _run_autoqueue_pass, which is
# where it already lived and where this file's own comments already point.

# A frame this dark is "the sensor is streaming but sees nothing" — lens cap,
# privacy shutter, unpowered sensor. ONE definition, because there are now two
# places that judge it (an opened device, and the producer's own cached frame)
# and a threshold fixed in one copy while the other rots is this codebase's
# most expensive bug shape.
_BLACK_FRAME_MEAN_MIN = 1.0
_BLACK_FRAME_RETRIES  = 3

# How recently the producer must have pulled a frame for that camera to count
# as PROVEN WORKING. 10 s is 2x the 5 s at which the dashboard calls a preview
# tile dead and a third of the 30 s at which the producer's own watchdog calls
# the loop stalled, so a single slow iteration cannot trip it: the producer's
# period measured 0.156 s median / 0.347 s max over 120 s on 2026-09-06, and
# the worst this codebase has ever recorded is the ~2.1 s of 2026-09-04. 10 s
# clears both by more than 4x.
_PRODUCER_FRAME_FRESH_S = 10.0

# How much of a camera's quarantine window must remain before the probe will
# treat the bench as "this device is free". 30 s is 5x the scan's worst-case
# hold, so the bench cannot expire out from under an open in flight. See the
# benched-camera branch in _producer_camera_ownership.
_QUARANTINE_SAFE_MARGIN_S = 30.0


def _translate_camera_index(cfg_idx, name: str, backend: str):
    """The index ``backend`` uses for the camera CAMERAS calls ``cfg_idx`` /
    ``name``, or None when we cannot say.

    None is a REAL ANSWER and callers must read it as "I do not know which scan
    index this device is", never as "not owned" — see the fail-closed branch in
    _probe_webcam_locked. Translation is by NAME, never by arithmetic on the
    configured integer: the two backends enumerate different device lists
    (DirectShow 0=USB 2.0 Camera 1=Kinect 2=eMeet 3=OBS Virtual Camera; Media
    Foundation 0=Kinect 1=USB 2.0 Camera 2=eMeet, with no OBS entry at all),
    which is the whole reason core/camera_backend.py exists."""
    try:
        if backend == "dshow":
            # CAMERAS holds DirectShow indices; that IS this index space.
            return None if cfg_idx is None else int(cfg_idx)
        if backend != "msmf":
            return None                      # unknown index space — fail closed
        from core.camera_backend import msmf_index_for_name
    except Exception:
        return None
    if not name:
        return None
    try:
        return msmf_index_for_name(name)
    except Exception:
        return None


def _producer_camera_ownership(backend: str) -> dict:
    """Which capture targets the face-tracking producer OWNS right now, in
    ``backend``'s index space, plus what its own telemetry says about each.

    Keys:
      ``owns``       — the producer is between "starting" and "stopped", so its
                       configured cameras are OFF LIMITS to this probe.
      ``indices``    — set of indices IN ``backend``'s space not to open.
      ``unresolved`` — labels of owned cameras we could not place in that index
                       space. Non-empty while ``owns`` means we do not know
                       which scan index is the producer's camera, so the caller
                       must FAIL CLOSED rather than guess — guessing wrong is
                       the 0 fps outcome documented above.
      ``fresh``      — owned cameras that delivered a frame within
                       _PRODUCER_FRAME_FRESH_S. Non-empty means the webcam is
                       proven working, with no device I/O at all.
      ``cameras``    — per-camera telemetry, for the result details.

    Reads ONLY the monolith's public accessors plus its published capture list,
    and NEVER raises: every failure degrades to "we know nothing", which leaves
    the pre-existing scan behaviour exactly as it was. CI, bare-skill imports
    and a JARVIS whose producer never started all land there."""
    out: dict[str, Any] = {"owns": False, "indices": set(), "unresolved": [],
                           "fresh": [], "cameras": [], "producer": {}}
    bc = _bc()
    if bc is None:
        return out

    # ── is the producer holding devices? ──
    # Two independent signals, UNIONED, because either one alone has a hole:
    #   * the heartbeat STAGE. The supervisor beats "starting" before any open
    #     and "stopped" in its finally, AFTER _face_track_release_all. Anything
    #     between the two means it holds — or is about to open — a camera. This
    #     is the signal that covers the race between an open returning and the
    #     handle being recorded, and it stays true for a WEDGED producer, which
    #     still owns its devices however dead it looks.
    #   * the published capture list. A handle recorded there is owned even if
    #     the heartbeat says something unexpected.
    try:
        live = bc.get_face_track_liveness()
        if isinstance(live, dict):
            out["producer"] = {"stage":   live.get("stage"),
                               "age_s":   live.get("age_s"),
                               "stalled": live.get("stalled"),
                               "iters":   live.get("iters")}
            if live.get("at") and live.get("stage") not in ("stopped",
                                                            "not started"):
                out["owns"] = True
    except Exception:
        pass
    held_indices: set = set()
    try:
        caps = getattr(bc, "_face_track_caps", None)
        entries = caps[0] if caps else None
        for entry in (entries or []):
            if not isinstance(entry, dict) or entry.get("cap") is None:
                continue
            out["owns"] = True
            cam = entry.get("cam") or {}
            if isinstance(cam, dict) and cam.get("index") is not None:
                held_indices.add(cam["index"])
    except Exception:
        pass
    if not out["owns"]:
        return out

    try:
        health = bc.get_camera_health() or {}
    except Exception:
        health = {}
    try:
        cameras = list(getattr(bc, "CAMERAS", None) or [])
    except Exception:
        cameras = []
    now = _now()
    for cam in cameras:
        if not isinstance(cam, dict):
            continue
        # A Kinect slot is not a cv2 webcam — the producer reaches it through
        # pykinect2, never through VideoCapture, so it owns no index here.
        if str(cam.get("type") or "").lower() == "kinect":
            continue
        cfg_idx = cam.get("index")
        name    = (cam.get("name") or "").strip()
        label   = cam.get("label") or f"index {cfg_idx}"
        h       = health.get(cfg_idx) or {}
        last_at = float(h.get("last_frame_at") or 0.0)
        age     = (now - last_at) if last_at else None
        # A BENCHED CAMERA IS NOT OWNED. The producer's quarantine gate is an
        # unconditional `continue` that releases the handle and never opens the
        # device again until the bench expires, so for that window the camera
        # genuinely belongs to nobody — and it is exactly the camera the
        # diagnostic most wants to look at. Fencing it off would trade the
        # collision for a permanently blind probe on a single-camera rig whose
        # one camera is sick.
        #
        # ONLY WITH ROOM TO SPARE. The bench expiring mid-probe would put the
        # producer's reopen and this open back in the same race, so we need
        # more of the window left than the probe's worst-case hold: 3 indices x
        # (open + _CAMERA_PROBE_WARMUP_S) is ~6 s, and this margin is 5x that.
        q_until = float(h.get("quarantine_until") or 0.0)
        benched = (bool(h.get("quarantined"))
                   and (q_until - now) > _QUARANTINE_SAFE_MARGIN_S)
        rec: dict[str, Any] = {
            "config_index":     cfg_idx,
            "label":            label,
            "held":             cfg_idx in held_indices,
            "last_frame_age_s": None if age is None else round(age, 1),
            "quarantined":      bool(h.get("quarantined")),
            "benched_free":     benched,
            "last_read_error":  h.get("last_read_error"),
        }
        probe_idx = _translate_camera_index(cfg_idx, name, backend)
        rec["probe_index"] = probe_idx
        if benched:
            pass                       # free: neither fenced nor unresolvable
        elif probe_idx is None:
            out["unresolved"].append(label)
        else:
            out["indices"].add(probe_idx)
        if age is not None and age <= _PRODUCER_FRAME_FRESH_S:
            out["fresh"].append(rec)
        out["cameras"].append(rec)
    return out


def _producer_latest_frame(cfg_idx):
    """The face-tracking producer's OWN newest frame for camera ``cfg_idx``,
    or None.

    The producer caches every frame it successfully reads, under the same
    _camera_state_lock and in the same critical section that stamps
    ``last_frame_at`` — so this is exactly the image the freshness figure is
    about, obtained without opening anything. That is what lets the
    no-device-I/O verdict keep the black-frame check instead of quietly
    dropping it. NEVER raises."""
    bc = _bc()
    if bc is None:
        return None
    try:
        store = getattr(bc, "_camera_latest_frame", None)
        if store is None:
            return None
        lock = getattr(bc, "_camera_state_lock", None)
        if lock is None:
            return store.get(cfg_idx)
        with lock:
            return store.get(cfg_idx)
    except Exception:
        return None


def _frame_mean(frame) -> float:
    try:
        return float(frame.mean())
    except Exception:  # pragma: no cover - defensive: mean() on a malformed frame
        return 0.0


def _face_cascade_status(cv2_mod) -> "tuple[bool, str]":
    """(ok, note) for the Haar cascade face_tracker depends on.

    Device-free, so every webcam verdict — including the ones that open nothing
    because the producer owns the cameras — still runs it."""
    try:
        path = cv2_mod.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2_mod.CascadeClassifier(path)
        if cascade.empty():
            return False, f"face cascade failed to load from {path}"
        return True, "loaded"
    except Exception as e:
        return False, f"face cascade failed: {type(e).__name__}: {e}"


def _probe_webcam() -> dict:
    start = _now()
    try:
        import cv2  # type: ignore  # noqa: F401 — availability check before taking the lock
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"opencv not importable: {e}")

    # 2026-07-21 audit: every in-process cv2.VideoCapture open/release must
    # hold bobert_companion._camera_io_lock — overlapping open/release in
    # DirectShow's plumbing heap-corrupts the process (0xc0000374), and an
    # unlocked probe steals the device from the face tracker mid-cycle.
    # Same lock + bounded acquire as _attempt_camera_wake; contention means
    # the tracker is actively using the camera, so skip as a NON-failure —
    # the upgrade pipeline must not queue repair tasks for a busy device.
    bc = _bc()
    hold = _CameraLockHold(getattr(bc, "_camera_io_lock", None) if bc else None)
    if not hold.acquire(2.5):
        # 2026-08-20: contention is not a failure — but it is not a PASS
        # either. We never opened the device, so we know nothing about it; the
        # tracker holding the lock could be failing every read (that case is
        # covered separately by face_tracker's read-failure signal, which
        # _run_autoqueue_pass consumes). UNVERIFIED keeps the "never queue a
        # repair for a busy device" behaviour — unverified results never reach
        # _queue_repair_task — without claiming the camera works.
        return _unverified((_now() - start) * 1000.0,
                           reason=("the face tracker holds _camera_io_lock, so "
                                   "the probe never opened the device"),
                           details={"skipped": "camera busy — probe skipped "
                                               "(lock held by tracker)"},
                           short_cause=_UNVERIFIED_SHORT_CAUSES["camera_busy"],
                           transient=True)
    try:
        return _probe_webcam_locked(start, hold)
    finally:
        hold.release()


def _probe_webcam_locked(start: float, hold: _CameraLockHold) -> dict:
    """Body of the webcam probe. Runs with ``hold`` taken on
    _camera_io_lock (when bobert_companion is loaded); drops it early
    before delegating to _attempt_camera_wake — see _CameraLockHold.

    HOLDING THAT LOCK IS NOT PERMISSION TO OPEN A CAMERA. It only means no
    other open/release is in flight; the face-tracking producer holds its
    handles across it and reads outside it. So this function asks
    _producer_camera_ownership() who owns what BEFORE it opens anything, and
    every device the producer owns is off limits — see the block above that
    function for the measurement that made this rule, and for the 30 minutes
    of 0 fps it cost the owner each time it was broken."""
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover — the caller already imported cv2
        return _result(False, (_now() - start) * 1000.0,
                       error=f"opencv not importable: {e}")

    details: dict[str, Any] = {}
    # Try indices 0..2 — most laptops expose the integrated camera at 0,
    # USB cams take 1/2 depending on enumeration order.
    #
    # TWO BUGS LIVED IN THESE SIX LINES, and they compounded (measured
    # 2026-09-05 on the owner's rig, JARVIS stopped, cameras idle):
    #
    # 1. MIXED INDEX SPACES. This scan called cv2.VideoCapture(idx) with no
    #    backend, which resolves to MSMF here — verified live,
    #    getBackendName() == 'MSMF'. The winning index was then handed to
    #    _attempt_camera_wake(), which opened it with cv2.CAP_DSHOW. The two
    #    backends enumerate DIFFERENT device lists (Media Foundation
    #    0=Kinect 1=USB 2.0 Camera 2=eMeet; DirectShow 0=USB 2.0 Camera
    #    1=Kinect 2=eMeet), so "index 0" meant the Kinect to the scan and the
    #    USB webcam to the wake. The probe diagnosed one device and then
    #    tried to revive a different one.
    # 2. OPENED IS NOT WORKING. The scan accepted the first index that merely
    #    isOpened(). Under MSMF that is index 0, the Kinect's Media
    #    Foundation interface, which opens at 512x424 and then fails every
    #    read (OnReadSample error -2147024809) — inherently, not because
    #    something else holds it. So the scan ALWAYS picked a device that can
    #    never produce a frame, always fell into the soft-wake branch, and
    #    the DirectShow open inside that branch is the +3 OS threads and
    #    ~+309 handles this probe leaked every 30 minutes.
    #
    # Both are fixed by asking the shared resolver for a camera that actually
    # DELIVERS AN IMAGE, on one stated backend, and telling the wake helper
    # which backend that was. _CAMERA_PROBE_WARMUP_S is a budget rather than a
    # single read because a healthy camera can legitimately return False for
    # its first couple of seconds while it warms up.
    cap = None
    cam_index = None
    cam_backend = _camera_backend_name()

    # A THIRD BUG LIVED HERE, and fixing #2 is what armed it (2026-09-06).
    # Demanding a real frame stopped the scan settling on the Kinect at MSMF
    # index 0 — and walked it onto MSMF index 1, which is the face-tracking
    # producer's own right webcam. Opening that collapsed the producer's stream
    # to 0 fps, permanently, every 30 minutes. The full measurement, the
    # nine-for-nine live confirmation and why the migration's cross-process
    # evidence said it was safe are in the block above
    # _producer_camera_ownership.
    #
    # So: ASK WHO OWNS THE DEVICE BEFORE OPENING IT.
    owned = _producer_camera_ownership(cam_backend)
    if owned["owns"]:
        details["producer"] = owned["producer"]
        details["producer_cameras"] = owned["cameras"]
        if owned["unresolved"]:
            # FAIL CLOSED. The producer owns a camera we could not place in
            # this backend's index space, so every index in the sweep might be
            # it. Opening the wrong one costs the owner his primary vision
            # until JARVIS restarts; skipping one sweep costs a data point.
            return _unverified(
                (_now() - start) * 1000.0,
                reason=("the face tracker owns "
                        + ", ".join(owned["unresolved"])
                        + " and the probe was not able to work out which "
                          "capture index that is, so it opened nothing"),
                details=dict(details,
                             skipped="camera owned by the face tracker; "
                                     "index not resolvable on "
                                     + cam_backend),
                short_cause=_UNVERIFIED_SHORT_CAUSES["camera_busy"],
                transient=True)
        if owned["fresh"]:
            # PROVEN WORKING, WITHOUT TOUCHING THE DEVICE. The producer pulled
            # a frame off this camera within the last _PRODUCER_FRAME_FRESH_S —
            # a live open-and-read test of exactly the thing this probe exists
            # to check, running at a measured 6.4 fps median (2026-09-06, 678
            # preview writes in 120 s). Strictly better evidence than one spot
            # check every half hour, and it costs no handle, no thread and no
            # risk to the stream. Everything below the device I/O still runs.
            best = min(owned["fresh"],
                       key=lambda r: r["last_frame_age_s"])
            details["verified_via"]   = "face-track producer telemetry"
            # ``index`` KEEPS ITS OLD MEANING: the index in ``backend``'s space,
            # because that pairing is what every reader of this record assumes
            # and the two spaces disagree (the owner's right webcam is
            # DirectShow 0 and Media Foundation 1). The CAMERAS index goes
            # alongside under its own name rather than silently overloading
            # this one — on this rig the eMeet happens to be 2 in BOTH lists,
            # which is exactly the coincidence that would hide the mix-up.
            details["index"]          = best["probe_index"]
            details["config_index"]   = best["config_index"]
            details["backend"]        = cam_backend
            details["camera"]         = best["label"]
            details["frame_age_s"]    = best["last_frame_age_s"]
            details["device_opened"]  = False
            # THE BLACK-FRAME CHECK IS NOT DROPPED WITH THE OPEN. A camera that
            # streams perfectly and sees nothing — lens cap, privacy shutter,
            # unpowered sensor — is a real finding this probe has always made,
            # and losing it would be paying for the fix with a blind spot. The
            # producer hands us the image for free.
            #
            # The retries here re-SAMPLE the producer's cache rather than
            # pulling new reads off the device, and the producer's period can
            # be up to ~2.1 s, so two samples may be the same frame; the
            # timestamps recorded alongside make that visible instead of
            # implied. This is why the gap is 0.25 s and not the scan path's
            # 0.05 s. It matters less than it looks: the transient blackness
            # those retries exist for is a driver INITIALISATION artifact, and
            # a frame in this cache is by definition one the producer already
            # read successfully from a running stream.
            frame = _producer_latest_frame(best["config_index"])
            if frame is not None:
                mean_val = _frame_mean(frame)
                details["frame_mean"]  = round(mean_val, 2)
                details["frame_shape"] = list(getattr(frame, "shape", ()))
                if mean_val < _BLACK_FRAME_MEAN_MIN:
                    means = [mean_val]
                    for _ in range(_BLACK_FRAME_RETRIES):
                        time.sleep(0.25)
                        f2 = _producer_latest_frame(best["config_index"])
                        m2 = _frame_mean(f2) if f2 is not None else 0.0
                        means.append(m2)
                        if m2 >= _BLACK_FRAME_MEAN_MIN:
                            mean_val = m2
                            break
                    details["frame_mean"]        = round(mean_val, 2)
                    details["frame_retry_means"] = [round(m, 2) for m in means]
                if mean_val < _BLACK_FRAME_MEAN_MIN:
                    details["auto_repairable"] = False
                    details["failure_mode"]    = "persistent_black_frame"
                    _maybe_announce_once(
                        "webcam_black_frame",
                        "Sir, the webcam is producing only black frames — "
                        "check the lens cover, privacy shutter, or USB power.",
                    )
                    return _result(
                        False, (_now() - start) * 1000.0,
                        error=(f"{best['label']} is streaming to the face "
                               f"tracker but every frame is black (mean "
                               f"{details['frame_mean']}). Sensor is running "
                               f"and sees nothing. Check (in order): (1) lens "
                               f"cover or privacy shutter, (2) USB cable / hub "
                               f"power, (3) Device Manager → camera driver. "
                               f"This is environmental and cannot be "
                               f"auto-fixed from code."),
                        details=details,
                        severity=SEVERITY_LOW,
                    )
            ok_c, note = _face_cascade_status(cv2)
            details["cascade"] = note
            if not ok_c:
                return _result(False, (_now() - start) * 1000.0,
                               error=note, details=details)
            return _result(True, (_now() - start) * 1000.0, details=details)

    skip = set(owned["indices"])
    if skip:
        details["skipped_indices"] = sorted(skip)
    for idx in (0, 1, 2):
        if idx in skip:
            continue                 # the producer is streaming this device
        try:
            c = _open_probe_capture(idx, cam_backend,
                                    require_frame=_CAMERA_PROBE_WARMUP_S)
            if c is not None:
                cap = c
                cam_index = idx
                break
        except Exception:
            continue
    if cap is None and skip:
        # Nothing left to open, because the only cameras that could have
        # answered are the ones the producer holds and none of them is
        # currently delivering. That is NOT "no webcam found" — we never
        # looked at the devices in question. The sick-camera case is already
        # carried by the producer's own read-failure signal, which reaches the
        # self-heal pipeline through _run_autoqueue_pass; reporting a failure
        # here as well would file a repair task for a device we did not test.
        return _unverified(
            (_now() - start) * 1000.0,
            reason=("the face tracker owns every camera and none of them has "
                    "produced a frame recently, so the probe opened nothing "
                    "rather than taking a second handle on a live device"),
            details=dict(details,
                         skipped="camera owned by the face tracker"),
            short_cause=_UNVERIFIED_SHORT_CAUSES["camera_busy"],
            transient=True)
    if cap is None:
        pnp_devices = _windows_camera_pnp_devices()
        diag = _camera_pnp_diagnosis(pnp_devices)
        hw_count = (_windows_camera_hardware_count()
                    if pnp_devices is None else len(pnp_devices or []))
        if diag["failure_mode"] == "absent":
            # Genuinely no camera hardware on this box — don't pester the
            # self-heal pipeline with repair tasks for absent hardware.
            _maybe_announce_once(
                "webcam_absent",
                "Sir, the webcam appears to be unplugged or absent — "
                "face tracking won't be available until it's reconnected.",
            )
            return _result(False, (_now() - start) * 1000.0,
                           error="no webcam hardware detected (Windows PnP "
                                 "reports 0 camera-class devices) — unplugged "
                                 "or none installed. Hardware-absent failures "
                                 "cannot be auto-fixed and the upgrade pipeline "
                                 "should skip them.",
                           details={"pnp_camera_count": 0,
                                    "pnp_diagnosis": diag,
                                    "auto_repairable": False,
                                    "failure_mode": "hardware_absent"},
                           severity=SEVERITY_LOW)
        # Hardware present (or PnP unavailable) but VideoCapture refused
        # to open any index. Most common cause: another app holds the
        # camera in exclusive mode. Surface the suspect process so the
        # repair task tells the user exactly what to close.
        suspects = _camera_lock_suspects()
        details_open: dict[str, Any] = {"pnp_diagnosis": diag}
        if hw_count is not None:
            details_open["pnp_camera_count"] = hw_count
        if suspects:
            details_open["camera_lock_suspects"] = suspects
            details_open["auto_repairable"] = False
            details_open["failure_mode"] = "locked_by_other_app"
            return _result(
                False, (_now() - start) * 1000.0,
                error=(f"no usable webcam found at indices 0..2 — "
                       f"{', '.join(suspects)} appears to be holding the "
                       f"camera lock. Close it and the next sweep should "
                       f"pass; this is environmental, not a code bug."),
                details=details_open,
                severity=SEVERITY_LOW,
            )
        if diag["failure_mode"] == "problem":
            # Device is present in PnP but reporting an error / non-zero
            # problem code. Driver wedge, disabled in Device Manager, or a
            # USB enumeration glitch — none of which we can fix from code.
            details_open["auto_repairable"] = False
            details_open["failure_mode"] = "pnp_device_problem"
            _maybe_announce_once(
                "webcam_pnp_problem",
                f"Sir, the webcam reports a device error ({diag['summary']}) — "
                f"try Device Manager → disable + re-enable, or reinstall the driver.",
            )
            return _result(
                False, (_now() - start) * 1000.0,
                error=(f"webcam present but PnP reports a device problem: "
                       f"{diag['summary']}. Try Device Manager → disable + "
                       f"re-enable, or update/reinstall the camera driver. "
                       f"This is environmental and cannot be auto-fixed."),
                details=details_open,
                severity=SEVERITY_LOW,
            )
        extra = "" if hw_count is None else f" (PnP sees {hw_count} camera device(s))"
        details_open["auto_repairable"] = False
        details_open["failure_mode"] = "open_failed"
        return _result(False, (_now() - start) * 1000.0,
                       error=(f"no usable webcam found at indices 0..2{extra} — "
                              f"no known webcam-locking app is running, so this "
                              f"likely indicates a driver issue (check Device "
                              f"Manager, update / reinstall the camera driver)."),
                       details=details_open)
    details["index"] = cam_index
    details["backend"] = cam_backend

    try:
        # Some cameras need a warmup frame — read twice and use the second.
        cap.read()
        ok, frame = cap.read()
        if not ok or frame is None:
            # cap.isOpened() succeeded but read() returned nothing. Before
            # we declare the device dead and queue a (probably futile)
            # repair task, try a soft wake: release this handle and
            # reopen. This recovers the common power-save / stalled-USB
            # pattern where the device is fine but its pipe needs a poke.
            try:
                cap.release()
            except Exception:  # pragma: no cover - defensive: cv2 release() before wake retry (live-camera I/O, cv2 absent on CI)
                pass
            cap = None
            # Release the camera lock BEFORE delegating to the wake helper:
            # its worker acquires the same lock from a different thread
            # (diag-wake-N), and RLock reentrancy is per-thread — holding on
            # here would make every wake falsely time out with "lock held by
            # tracker". cap is already released and None, and every
            # post-wake path returns without touching the device, so no
            # re-acquire is needed.
            hold.release()
            wake_ok, wake_note = _attempt_camera_wake(
                cam_index, backend=cam_backend)
            details["wake_attempted"] = True
            details["wake_recovered"] = bool(wake_ok)
            details["wake_note"]      = wake_note
            if wake_ok:
                # Soft recovery succeeded — the next face-tracker read
                # should also succeed. Return OK so the upgrade pipeline
                # doesn't queue a repair task for a self-healed glitch.
                return _result(True, (_now() - start) * 1000.0,
                               details=details)

            # Wake failed. Try PnP to tell the user *why* — was the device
            # actually unplugged in the meantime, or is it just stuck?
            pnp_devices = _windows_camera_pnp_devices()
            diag = _camera_pnp_diagnosis(pnp_devices)
            details["pnp_diagnosis"] = diag

            suspects = _camera_lock_suspects()
            if suspects:
                details["camera_lock_suspects"] = suspects
                details["auto_repairable"] = False
                details["failure_mode"] = "locked_by_other_app"
                return _result(
                    False, (_now() - start) * 1000.0,
                    error=(f"webcam.read returned no frame at index "
                           f"{cam_index} — {', '.join(suspects)} is "
                           f"currently using the camera. Close it and "
                           f"the next sweep should pass."),
                    details=details,
                    severity=SEVERITY_LOW,
                )

            if diag["failure_mode"] == "absent":
                # Device disappeared between the open() and the failed
                # read — most likely physically unplugged or USB hub power
                # cycled. Auto-repair cannot bring back hardware that
                # isn't physically present.
                details["auto_repairable"] = False
                details["failure_mode"] = "hardware_unplugged"
                _maybe_announce_once(
                    "webcam_unplugged_midstream",
                    "Sir, the webcam appears to have been unplugged — "
                    "Windows no longer sees the device. Face tracking is offline.",
                )
                return _result(
                    False, (_now() - start) * 1000.0,
                    error=(f"webcam at index {cam_index} disappeared from "
                           f"PnP between open and read — hardware appears "
                           f"to have been physically disconnected. This "
                           f"cannot be auto-fixed; manual intervention "
                           f"required (re-plug the cable)."),
                    details=details,
                    severity=SEVERITY_LOW,
                )

            if diag["failure_mode"] == "problem":
                # Device present but PnP flags a problem — driver crashed,
                # power management put it to sleep, etc. Still environmental.
                details["auto_repairable"] = False
                details["failure_mode"] = "pnp_device_problem"
                _maybe_announce_once(
                    "webcam_pnp_problem",
                    f"Sir, the webcam reports a device error ({diag['summary']}) "
                    f"and the soft wake didn't recover it.",
                )
                return _result(
                    False, (_now() - start) * 1000.0,
                    error=(f"webcam at index {cam_index} read returned no "
                           f"frame and PnP reports a device problem: "
                           f"{diag['summary']}. Wake attempt: {wake_note}. "
                           f"Environmental — cannot auto-fix."),
                    details=details,
                    severity=SEVERITY_LOW,
                )

            # PnP says the device is OK but the read still fails after a
            # wake. That's a stalled USB pipe / driver hang — we can't
            # repair it from code, but it often clears on its own within a
            # minute or two. Downgrade severity to LOW so we don't pile up
            # repair tasks for a transient condition.
            details["auto_repairable"] = False
            details["failure_mode"] = "unresponsive_after_wake"
            return _result(
                False, (_now() - start) * 1000.0,
                error=(f"webcam.read returned no frame at index {cam_index} "
                       f"and a release+reopen wake did not recover it "
                       f"({wake_note}). PnP reports the device as healthy "
                       f"— likely a stalled USB pipe or power-save state. "
                       f"Try unplug + replug, or Device Manager → disable "
                       f"+ re-enable. No code change can repair this; the "
                       f"upgrade pipeline should not queue further "
                       f"webcam repair tasks for this mode."),
                details=details,
                severity=SEVERITY_LOW,
            )
        # Verify the frame isn't a uniform black image. A single black
        # frame after warmup can be a driver initialization artifact —
        # some UVC drivers stream a few zeroed buffers before the sensor
        # gain settles. Retry a few times with a brief delay to separate
        # transient init blackness from a persistent condition (lens cap
        # on, sensor unpowered, privacy shutter closed). We cap retries
        # tightly to keep probe latency bounded.
        try:
            mean_val = float(frame.mean())
        except Exception:  # pragma: no cover - defensive: numpy frame.mean() on a malformed cv2 frame (live-camera I/O, cv2 absent on CI)
            mean_val = 0.0
        details["frame_mean"] = round(mean_val, 2)
        details["frame_shape"] = list(getattr(frame, "shape", ()))
        if mean_val < _BLACK_FRAME_MEAN_MIN:
            retry_means: list[float] = [mean_val]
            for _ in range(_BLACK_FRAME_RETRIES):
                time.sleep(0.05)
                try:
                    ok_r, frame_r = cap.read()
                except Exception:  # pragma: no cover - defensive: cv2 read() raising mid black-frame retry (live-camera I/O, cv2 absent on CI)
                    ok_r, frame_r = False, None
                if not ok_r or frame_r is None:
                    retry_means.append(0.0)
                    continue
                try:
                    rmean = float(frame_r.mean())
                except Exception:  # pragma: no cover - defensive: numpy mean() on a malformed retry frame (live-camera I/O, cv2 absent on CI)
                    rmean = 0.0
                retry_means.append(rmean)
                if rmean >= _BLACK_FRAME_MEAN_MIN:
                    # Sensor warmed up — accept this frame and move on.
                    mean_val = rmean
                    frame = frame_r
                    break
            details["frame_mean"]       = round(mean_val, 2)
            details["frame_retry_means"] = [round(m, 2) for m in retry_means]
            if mean_val < _BLACK_FRAME_MEAN_MIN:
                # Every retry yielded a black frame. The device is opening
                # and streaming buffers but the sensor sees nothing.
                # Auto-repair cannot distinguish a deliberately-covered
                # lens / closed privacy shutter from a failed sensor —
                # both look identical from software. Mark as LOW + not
                # auto_repairable so the upgrade pipeline doesn't queue
                # the same repair task every sweep.
                details["auto_repairable"] = False
                details["failure_mode"]   = "persistent_black_frame"
                _maybe_announce_once(
                    "webcam_black_frame",
                    "Sir, the webcam is producing only black frames — "
                    "check the lens cover, privacy shutter, or USB power.",
                )
                return _result(
                    False, (_now() - start) * 1000.0,
                    error=(f"webcam at index {cam_index} returned only black "
                           f"frames across {len(retry_means)} reads (means: "
                           f"{[round(m,1) for m in retry_means]}). Sensor is "
                           f"streaming but sees nothing. Check (in order): "
                           f"(1) lens cover or privacy shutter, (2) USB "
                           f"cable / hub power, (3) Device Manager → camera "
                           f"driver. This is environmental and cannot be "
                           f"auto-fixed from code."),
                    details=details,
                    severity=SEVERITY_LOW,
                )
    finally:
        # `cap` may already be None (we released it earlier to attempt a
        # wake); guard so the finally never raises AttributeError.
        if cap is not None:
            try:
                cap.release()
            except Exception:  # pragma: no cover - defensive: cv2 release() in the probe's finally (live-camera I/O, cv2 absent on CI)
                pass

    # Verify the face cascade loads — face_tracker depends on it. Shared with
    # the producer-telemetry verdict above, which opens no device but must
    # still make this check.
    ok_c, note = _face_cascade_status(cv2)
    details["cascade"] = note
    if not ok_c:
        return _result(False, (_now() - start) * 1000.0,
                       error=note, details=details)

    return _result(True, (_now() - start) * 1000.0, details=details)


# ─── Probe 2: microphone ─────────────────────────────────────────────────
# Names that look like mic inputs in Windows PnP FriendlyName values.
# Audio "input" endpoints include line-in jacks; the wider regex avoids
# missing Realtek line-in or unusual third-party USB caps.
_MIC_PNP_NAME_REGEX = "microphone|line in|\\bmic\\b|input"

# Virtual / loopback inputs that won't produce ambient noise even when the
# physical hardware is fine. Skipped when scanning alternates so we don't
# waste a probe slot on a guaranteed-silent device. Matches as a substring
# (case-insensitive) against the sounddevice device name.
_VIRTUAL_INPUT_RE = re.compile(
    r"sound mapper|steam streaming|stereo mix|loopback|"
    r"virtual cable|vb-?audio|cable output|wave\b",
    re.IGNORECASE,
)


def _windows_microphone_hardware_count() -> int | None:
    """Ask Windows PnP whether any mic-class audio endpoint exists at all.

    Returns the count of present audio endpoints whose FriendlyName looks
    like a microphone (or line-in / input jack), or None when the query
    can't run (non-Windows, missing PowerShell, timeout). Used to
    distinguish "audio stack is broken" (HIGH — driver or PortAudio
    failed) from "user has muted everything / wireless headset is off"
    (LOW — environmental, can't be auto-fixed from code). The latter
    shouldn't keep auto-queueing the same repair task every 30 minutes.
    """
    if sys.platform != "win32":
        return None
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "(Get-PnpDevice -Class AudioEndpoint -PresentOnly "
             "-ErrorAction SilentlyContinue | "
             "Where-Object { $_.FriendlyName -match "
             f"'{_MIC_PNP_NAME_REGEX}' " "} | "
             "Measure-Object).Count"],
            capture_output=True, text=True, timeout=4.0,
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if sys.platform == "win32" else 0),
        )
        if proc.returncode != 0:
            return None
        out = (proc.stdout or "").strip()
        if not out:
            return 0
        return int(out.splitlines()[-1].strip())
    except Exception:
        return None


def _jarvis_active_mic_index(sd) -> int | None:
    """Return the input-device index JARVIS would use right now.

    Prefers ``bobert_companion.get_input_device()`` so the probe reflects
    the device the wake-word listener actually opens (which may differ
    from the system default — bobert has a PREFERRED_INPUT_DEVICES list).
    Falls back to ``sd.default.device[0]`` when bobert isn't loaded yet
    (early-boot window, pytest, standalone execution).
    """
    bc = _bc()
    if bc is not None:
        try:
            idx = bc.get_input_device()
            if isinstance(idx, int) and idx >= 0:
                return idx
        except Exception:
            pass
    try:
        idx = sd.default.device[0]
        if isinstance(idx, int) and idx >= 0:
            return idx
    except Exception:  # pragma: no cover - defensive: sounddevice default-device lookup failing (audio I/O, sounddevice absent on CI)
        pass
    return None


# ─── Passive mic liveness (no device access) ───────────────────────────
# Stable text — _announce_failures dedups per component on the ERROR STRING, so
# this must not carry a changing number or JARVIS would re-speak it every
# sweep. The measurements live in details["passive_mic"] instead.
_MIC_PASSIVE_SILENT_ERROR = (
    "JARVIS is not hearing anything: the capture loop is reading chunks, but "
    "every one of them has been below the audible floor for longer than the "
    "silent-mic window — the mic is enumerated and open yet delivering null "
    "frames. Judged from the audio the main loop itself captured (see "
    "details.passive_mic). Check Windows Privacy -> Microphone, the mixer mute, "
    "and whether the wireless mic is powered on."
)


def _passive_mic_liveness() -> dict:
    """Judge the mic from audio THE MAIN LOOP ALREADY READ — no device access.

    record_speech measures a raw RMS for every chunk it inspects and publishes
    it through core.audio_processor (``note_vad_poll`` + ``note_raw_rms``, at
    bobert_companion.py's capture loop). Reading those counters needs no
    stream, so it cannot contend with an owner, cannot trip the crash-fix-3
    WASAPI double-open, and — unlike ``_collect_vad_stall_signal``, which bails
    while sleeping — works in awake AND standby, which is where the mic probe
    spends essentially all of its life.

    Returns ``{"verdict": "alive" | "silent" | "nodata", "reason": str, …}``:

        alive   — POSITIVE evidence: the loop read a chunk above
                  core.audio_processor._AUDIBLE_RMS_FLOOR within
                  core.config.MIC_SILENT_WARN_SECONDS. (A dead mic measured
                  ~1e-5 on 2026-08-20 while a genuinely quiet room measured
                  ~1e-3 — two orders apart, so "quiet user" does not read as
                  dead.)
        silent  — the loop IS polling, yet nothing has crossed that floor for
                  longer than the window: null frames, i.e. JARVIS is deaf.
        nodata  — no basis for either statement (loop not polling, counters
                  cold, audio_processor unavailable). The caller must report
                  UNVERIFIED, never a pass.

    Deliberately STATELESS: every call re-derives its verdict from the live
    counters, and nothing here caches, latches or blacklists a device. The
    moment the Corsair powers back on and delivers one audible chunk,
    ``last_audible_chunk_ts`` moves and the very next sweep says "alive" again.

    Thresholds are reused, never re-invented (stale-duplicate rule):
    MIC_SILENT_WARN_SECONDS is core.config's, the audible floor is
    audio_processor's own (applied inside note_raw_rms), and the poll-freshness
    window is _PASSIVE_POLL_FRESH_S, shared with _collect_vad_stall_signal.
    """
    out: dict[str, Any] = {"verdict": "nodata", "reason": "",
                           "source": "core.audio_processor"}
    try:
        from core import audio_processor as _ap
    except Exception as e:
        out["reason"] = (f"core.audio_processor is unavailable "
                         f"({type(e).__name__}) — no passive signal to read")
        return out
    try:
        st = _ap.get_vad_state() or {}
    except Exception as e:
        out["reason"] = f"get_vad_state raised {type(e).__name__}"
        return out
    try:
        from core.config import MIC_SILENT_WARN_SECONDS as _silent_window_s
    except Exception:
        # Config unreachable: fall back to the poll window rather than
        # inventing a third silence threshold.
        _silent_window_s = _PASSIVE_POLL_FRESH_S
    out["silent_window_s"] = float(_silent_window_s)

    now = _now()
    last_poll = float(st.get("last_vad_poll_ts") or 0.0)
    poll_age = (now - last_poll) if last_poll > 0.0 else float("inf")
    out["poll_age_s"] = None if poll_age == float("inf") else round(poll_age, 1)
    if poll_age > _PASSIVE_POLL_FRESH_S:
        out["reason"] = ("the capture loop has not inspected a chunk recently, "
                         "so it has no frames to judge")
        return out

    try:
        silence_age = float(_ap.seconds_since_audible_chunk())
    except Exception as e:
        out["reason"] = f"seconds_since_audible_chunk raised {type(e).__name__}"
        return out
    # last_audible_chunk_ts > 0 is what separates POSITIVE evidence from
    # seconds_since_audible_chunk's cold-start fallback (it returns time since
    # the session began when the mic has never yet produced an audible chunk).
    has_audible = float(st.get("last_audible_chunk_ts") or 0.0) > 0.0
    out["silence_age_s"] = None if silence_age == float("inf") else round(silence_age, 1)
    out["has_audible_chunk"] = has_audible
    if silence_age == float("inf"):
        out["reason"] = "the capture loop has not started polling this session"
        return out
    if silence_age > float(_silent_window_s):
        out["verdict"] = "silent"
        out["reason"] = ("the capture loop is polling but no chunk has crossed "
                         "the audible floor for longer than the silent-mic "
                         "window")
        return out
    if has_audible:
        out["verdict"] = "alive"
        out["reason"] = ("the capture loop read audible audio within the "
                         "silent-mic window")
        return out
    out["reason"] = ("polling started only moments ago and has not delivered an "
                     "audible chunk yet — too early to call it either way")
    return out


def _mic_live_recheck() -> bool:
    """Re-measure the microphone RIGHT NOW, device-free. True only on POSITIVE
    evidence that the capture loop is hearing audible audio.

    _passive_mic_liveness is stateless and opens nothing, so this is cheap
    enough to run on a read (`diagnostic status`) rather than waiting for the
    next 30-minute sweep. Anything that is not a live "alive" verdict returns
    False and the entry stays UNVERIFIED — this may only CLEAR an unverified
    entry, never create a pass out of missing data.
    """
    try:
        return _passive_mic_liveness().get("verdict") == "alive"
    except Exception:
        return False


# component -> "is it healthy right now?" probe that is safe to run on a READ:
# device-free, side-effect-free, and returning True only on positive evidence.
# Consulted for TRANSIENT unverified entries only (see _unverified).
_LIVE_RECHECKS: dict[str, Callable[[], bool]] = {
    "microphone": _mic_live_recheck,
}


def _live_recheck_unverified(run: dict,
                             unverified: list[str]) -> tuple[list[str], list[str]]:
    """Split ``unverified`` into (still-unverified, cleared-by-live-evidence).

    WHY (2026-08-20 review, MED): an UNVERIFIED that cannot clear until the
    next sweep is noise, and noise is how a real alarm gets ignored. The
    microphone probe's passive signal exists only while record_speech's chunk
    loop is running, so a sweep that lands during a long awake turn latches
    "microphone: UNVERIFIED" into last_run — and "diagnostic status" then
    refuses "all systems nominal" for up to half an hour with nothing wrong.

    This does NOT carry a stale pass forward. It takes a NEW measurement
    through the same device-free path the probe uses and clears the entry only
    on positive evidence. The recorded run and the history file are left
    untouched — the sweep genuinely had no data at the time, and that record
    stays true.
    """
    still: list[str] = []
    cleared: list[str] = []
    probes = (run or {}).get("probes") or {}
    for comp in unverified:
        det = ((probes.get(comp) or {}).get("details") or {})
        checker = (_LIVE_RECHECKS.get(comp)
                   if det.get("unverified_transient") else None)
        if checker is None:
            still.append(comp)
            continue
        try:
            ok = bool(checker())
        except Exception:
            ok = False
        (cleared if ok else still).append(comp)
    return still, cleared


def _mic_result_without_capture(start: float, details: dict, *,
                                mic_off: bool = False,
                                tts_live: bool = False,
                                partial_silence: bool = False) -> dict:
    """Verdict for a sweep whose own live capture could not run.

    THE flagship fix (2026-08-20). This path used to ``return _result(True, …)``
    — a HIGH-severity probe reporting a pass for a check it never performed, on
    the path that runs in virtually every sweep under START_IN_STANDBY. It now
    asks _passive_mic_liveness what the MAIN LOOP is hearing, which needs no
    device and so cannot fight the ownership gate that forced the skip:

        alive  → a real pass, with the evidence in details["passive_mic"]
        silent → a real failure at the microphone's default severity (HIGH):
                 this is not the environmental "every device I opened was
                 quiet" verdict in _probe_microphone's step 3, which stays
                 LOW — it means the loop
                 that is listening RIGHT NOW is getting null frames, i.e.
                 JARVIS is deaf, which is exactly the condition nothing
                 surfaced for 90 minutes. Announcement is deduped per
                 component on the (constant) error string, so it is spoken
                 once, not every sweep.
        nodata → UNVERIFIED — visibly distinct from both, kept out of
                 run["failed"], never announced, never queued.

    ``mic_off``: hard-disabled by configuration; silence is the CONFIGURED
    state so it can never be a failure — but it is not a pass either.
    ``tts_live``: JARVIS is speaking; its own playback can legitimately make
    the input look null, so a silence verdict is downgraded to UNVERIFIED
    rather than blamed on the mic. A positive "alive" reading is still trusted.
    ``partial_silence``: the probe already measured the active device silent
    before an owner cut the scan short, so a passive "alive" may not be
    upgraded to a clean pass — the two readings disagree and that is exactly
    the state we must not paper over.
    """
    latency = (_now() - start) * 1000.0
    if mic_off:
        return _unverified(latency,
                           reason=("the microphone is hard-disabled in "
                                   "configuration (staging / MICROPHONE_INDEX "
                                   "< 0), so there is nothing listening to "
                                   "check"),
                           details=details,
                           short_cause=_UNVERIFIED_SHORT_CAUSES["mic_disabled"])
    passive = _passive_mic_liveness()
    details["passive_mic"] = passive
    verdict = passive.get("verdict")

    if verdict == "silent" and tts_live:
        return _unverified(latency,
                           reason=("the capture loop is reading null frames, "
                                   "but JARVIS is speaking and its own "
                                   "playback can mask the input — not "
                                   "attributable to the mic"),
                           details=details,
                           remedy="Re-checked on the next sweep.",
                           short_cause=_UNVERIFIED_SHORT_CAUSES["mic_speaking"],
                           transient=True)
    if verdict == "silent":
        if partial_silence:
            # Two independent readings agree: our own capture of the active
            # device was below the floor AND the main loop is getting null
            # frames. Recorded in details, never in the error text — that
            # string is the announce-dedup key and must stay constant.
            details["silence_corroborated_by_probe_capture"] = True
        return _result(False, latency, error=_MIC_PASSIVE_SILENT_ERROR,
                       details=details)      # severity → subsystem default (HIGH)
    if verdict == "alive" and not partial_silence:
        return _result(True, latency, details=details)
    if verdict == "alive":
        return _unverified(latency,
                           reason=("the probe's own capture of the active "
                                   "device measured silent and the alternate "
                                   "scan was cut short by an owner, while the "
                                   "main loop reports audible audio — the two "
                                   "readings disagree"),
                           details=details,
                           remedy="Re-checked on the next sweep.",
                           short_cause=_UNVERIFIED_SHORT_CAUSES["mic_disagree"],
                           transient=True)
    # nodata. TRANSIENT by construction: the passive signal exists only while
    # record_speech's chunk loop is running, so ANY awake stretch longer than
    # _PASSIVE_POLL_FRESH_S with no capture — a long local-LLM turn, a long TTS
    # reply, a _HEAVY_ACTIONS skill, the tray "Mute Mic" toggle — lands here
    # with a perfectly healthy microphone. Latching that into last_run for the
    # whole 30-minute interval turns an honest signal into noise, and noise is
    # how a real alarm gets ignored, so readers may re-measure it live
    # (_live_recheck_unverified).
    return _unverified(latency,
                       reason=("the live capture was skipped and there is no "
                               "passive liveness signal either: "
                               + str(passive.get("reason") or "no data")),
                       details=details,
                       remedy=("Nothing here says the mic works — it says the "
                               "check did not run."),
                       short_cause=_UNVERIFIED_SHORT_CAUSES["mic_no_frames"],
                       transient=True)


def _probe_microphone() -> dict:
    start = _now()
    try:
        import sounddevice as sd  # type: ignore
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"sounddevice not importable: {e}")

    try:
        devices = sd.query_devices()
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"sounddevice.query_devices failed: {e}")

    inputs = [(i, d) for i, d in enumerate(devices)
              if d.get("max_input_channels", 0) > 0]
    if not inputs:
        # No input device enumerated at all. This is genuinely a broken
        # audio stack (driver / PortAudio failure) — HIGH severity.
        # Distinct from "hardware present but muted", which we treat as
        # an environmental condition below.
        return _result(False, (_now() - start) * 1000.0,
                       error="no input devices enumerated")

    details: dict[str, Any] = {"input_count": len(inputs)}
    active_idx = _jarvis_active_mic_index(sd)
    if active_idx is not None:
        details["active_input"] = active_idx
        try:
            details["active_input_name"] = devices[active_idx]["name"]
        except Exception:  # pragma: no cover - defensive: device-name subscript on a sparse sounddevice list (audio I/O, sounddevice absent on CI)
            pass

    # crash-fix-3 (2026-05-28), extended 2026-07-21 audit: opening an
    # probe capture stream from this probe's daemon thread while any
    # other stream is live on the mic causes the documented WASAPI
    # double-open contention (record_speech records garbage for ~70s until
    # the watchdog resets the loop), and when the probe hits
    # PER_PROBE_TIMEOUT_S the thread is abandoned mid-capture, PortAudio is
    # left holding the buffer, and the next sweep triggers heap corruption.
    # Skip the live capture step entirely when:
    #   • any mic/audio OWNERSHIP FLAG is set — the monolith's canonical
    #     mic-ownership rule (mirrors bobert_companion._pa_streams_live, the
    #     owner list _refresh_devices' teardown gate checks). This is the
    #     load-bearing check: in sleep/
    #     standby the main loop runs record_speech(timeout=20) continuously,
    #     so the old awake-only skip gated the capture exactly backwards —
    #     _sleep_mode is a conversation-mode latch, not a mic-ownership
    #     flag;
    #   • JARVIS is awake — kept as a belt-and-braces skip because the
    #     wake-word detector's persistent InputStream sets NONE of the
    #     ownership flags, so awake gaps between turns are not flag-covered;
    #   • the mic is hard-disabled.
    # Enumeration plus PnP hardware count is enough to confirm the audio
    # stack is alive.
    bc = _bc()

    def _owner_flag(name: str) -> bool:
        """Read a bobert_companion mic-ownership flag: a 1-element list
        ([False]/[True], or _ambient_stream_active's refcount [0]). An
        absent attribute / odd shape (SimpleNamespace test doubles, early
        boot) degrades to False."""
        try:
            v = getattr(bc, name, None)
            return bool(v and v[0])
        except Exception:
            return False

    def _mic_owned() -> bool:
        # Mirrors bobert_companion._pa_streams_live (the canonical owner
        # list) MINUS _diag_capture_active — that cell is THIS probe's own
        # claim, held across its captures below; consulting it here
        # would make the probe defer against itself. Update BOTH lists
        # together (stale-duplicate rule).
        #
        # H-4 (2026-08-20): mirroring the owner list is NECESSARY BUT NOT
        # SUFFICIENT here. Every call of this is a check-then-act instant,
        # while the hazard is SPAN-long: a background _speak can begin inside
        # a 0.25 s capture. That is why _capture_rms opens a PRIVATE
        # InputStream (never published into sounddevice’s global
        # _last_callback, so a concurrent sd.play() cannot close it) AND
        # re-checks this predicate inside its own drain loop.
        return bc is not None and (
            _owner_flag("_record_speech_active")
            or _owner_flag("_pathb_mic_active")
            or _owner_flag("_ambient_stream_active")   # refcount — truthy when > 0
            or _owner_flag("_enroll_capture_active")
            or _owner_flag("_tts_playback_active")
            # H-6 (2026-08-20): a native Pa_CloseStream handed to a daemon
            # whose caller already gave up waiting. The owner flag is down but
            # PortAudio is still executing the close, so opening another
            # capture stream here is the same class of hazard the flags above
            # cover. COUNT cell — truthy when > 0.
            or _owner_flag("_pa_close_pending"))

    awake = bool(bc is not None and not getattr(bc, "_sleep_mode", [True])[0])
    mic_off = bool(getattr(bc, "_mic_input_disabled", lambda: False)())
    mic_owned = _mic_owned()
    if awake or mic_off or mic_owned:
        details["live_capture_skipped"] = (
            ("mic hard-disabled (staging / MICROPHONE_INDEX < 0)" if mic_off
             else ("mic owned by record_speech/Path B/ambient/TTS, or an "
                   "abandoned native close is still in flight") if mic_owned
             else "JARVIS awake — mic owned by main loop")
            + "; skipping the live capture (crash-fix-3 / no-mic guard)"
        )
        # The skip itself is CORRECT and must stay (crash-fix-3). What was
        # wrong until 2026-08-20 was returning a PASS from it: under
        # START_IN_STANDBY the main loop holds the mic on essentially every
        # sweep, so this line reported "microphone: ok" in 100 of 100 recorded
        # runs without ever opening the device. Fall through to the passive
        # judgement instead — same information the main loop already has,
        # no competing stream.
        return _mic_result_without_capture(
            start, details, mic_off=mic_off,
            tts_live=_owner_flag("_tts_playback_active"))

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"numpy not importable: {e}",
                       details=details)

    def _close_probe_stream(stream) -> None:
        """Tear down one probe InputStream through the host’s guarded closer.

        bobert_companion._safe_close_stream stops synchronously, runs the
        native close on a daemon, registers it with the teardown gate
        (_pa_close_pending) and ABANDONS it on a bounded timeout. Prefer it;
        the inline fallback below exists only for stand-alone runs and test
        doubles with no monolith loaded.

        H-3 (2026-08-20): neither path may call the process-global sd.stop().
        It cannot free an explicitly constructed InputStream (sounddevice
        0.5.5 stops+closes ``_last_callback``, published only by
        sd.play()/sd.rec()/sd.playrec()) and its one reachable effect is
        closing a live TTS playback stream out from under
        play_with_lipsync’s reaper — two threads inside Pa_CloseStream on one
        WASAPI stream, i.e. 0xc0000374. Copies of that rule live in
        bobert_companion._safe_close_stream, core/wake_word,
        skills/ambient_listen and skills/enroll_voice — change all of them
        together (stale-duplicate rule)."""
        if stream is None:
            return
        bc_close = getattr(bc, "_safe_close_stream", None) if bc is not None else None
        if callable(bc_close):
            try:
                bc_close(stream)
                return
            except Exception:
                pass   # fall through to the inline teardown below
        try:
            stream.stop()
        except Exception:
            pass
        _closed = threading.Event()

        def _close_in_daemon():
            try:
                stream.close()
            except Exception:
                pass
            finally:
                _closed.set()
        threading.Thread(target=_close_in_daemon, daemon=True).start()
        if not _closed.wait(timeout=2.0):
            print("  [self-diag] probe stream.close hung >2.0s — abandoning "
                  "the close daemon (it dies with the process); sd.stop() is "
                  "deliberately NOT called (H-3)")

    def _capture_rms(device_idx: int | None,
                     duration_s: float = 0.25,
                     rate: int = 16000) -> tuple[float | None, str | None]:
        """Record ``duration_s`` of mono float32 audio from ``device_idx``
        (None → system default) and return ``(rms, err)``. ``rms`` is
        None when capture itself raised.

        H-4 (2026-08-20): this used to be ``sd.rec(...)`` + ``sd.wait()``, the
        PROCESS-GLOBAL convenience pair. Both defects that removed were real:

        (a) CROSS-CLOSE. sounddevice 0.5.5 publishes exactly one global
            ``_last_callback``, and ``_CallbackContext.start_stream`` opens
            with ``stop()`` — which stops AND closes whatever the previous
            play/rec context left there. So this probe’s ``sd.rec`` closed the
            monolith’s live TTS playback stream, and a concurrent ``sd.play``
            closed this probe’s recording stream mid-capture, waking our
            ``sd.wait()`` whose ``finally`` closes the same handle from a
            second thread. ``_StreamBase.close`` is
            ``Pa_CloseStream(self._ptr)`` and only THEN nulls ``_ptr``, with no
            lock: two concurrent Pa_CloseStream on one WASAPI stream is the
            0xc0000374 heap corruption the monolith’s ``_reap_playback``
            contract ("sd.stop()/sd.wait() are NEVER called on this stream")
            was written to eliminate. The probe holds no _SPEAK_LOCK and runs
            on an abandonable daemon, so neither of that contract’s stated
            preconditions held here.

        (b) UNBOUNDED WAIT. ``sd.wait()`` has no timeout, and the probe thread
            holds the ``_diag_capture_active`` refcount across it; the release
            lives in a ``finally`` a never-returning wait never reaches, and
            ``_run_with_timeout`` only ABANDONS the thread. The identical
            idiom in skills/enroll_voice was fixed on 2026-07-14 for exactly
            this reason — this copy rotted (stale-duplicate rule).

        The replacement is the pattern already proven in-tree at
        bobert_companion.get_mic_buffer Path B: a PRIVATE ``sd.InputStream``
        drained through a local queue under a wall-clock deadline. A private
        stream is never published into ``_last_callback``, so no ``sd.play()``
        can stop or close it and no ``sd.wait()`` of ours can ever bind to a
        play context — exactly ONE thread makes native calls on this stream.
        The deadline is inherently bounded, so the refcount can no longer be
        pinned. The loop also re-checks ``_mic_owned()`` DURING the capture,
        not merely between captures: TTS can start inside the 0.25 s span."""
        need = int(duration_s * rate)
        if need <= 0:
            return 0.0, None
        frames: "queue.Queue" = queue.Queue()

        def _cb(indata, n_frames, time_info, status):  # noqa: ARG001
            # Runs on the PortAudio callback thread: cheap, exception-proof,
            # and it MUST copy — sounddevice reuses the input buffer.
            try:
                mono = indata[:, 0] if getattr(indata, "ndim", 1) > 1 else indata
                frames.put(mono.copy())
            except Exception:
                pass

        try:
            stream = sd.InputStream(samplerate=rate, channels=1,
                                    dtype="float32", blocksize=1024,
                                    device=device_idx, callback=_cb)
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        chunks: list = []
        captured = 0
        try:
            stream.start()
            # Wall-clock bound, not _now(): _now is monkeypatched to a frozen
            # clock in tests, and a frozen deadline would never expire.
            deadline = time.monotonic() + duration_s + 1.0
            while captured < need and time.monotonic() < deadline:
                if _mic_owned():
                    # An owner appeared mid-capture (the standby loop’s next
                    # record_speech turn, or a background _speak). Yield the
                    # device immediately rather than sitting as a second open
                    # stream on it; the finally below closes ours.
                    break
                try:
                    frame = frames.get(timeout=0.2)
                except queue.Empty:
                    continue
                chunks.append(frame)
                captured += int(getattr(frame, "size", 0) or len(frame))
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        finally:
            _close_probe_stream(stream)
        if not chunks:
            return None, (f"TimeoutError: mic delivered no frames within "
                          f"{duration_s + 1.0:.2f}s")
        a = np.concatenate(chunks)
        if len(a) > need:
            a = a[:need]
        rms = float(np.sqrt(np.mean(np.square(a)))) if len(a) else 0.0
        return rms, None

    # Step 1: try the device JARVIS would actually use. This is the only
    # device that matters for "can JARVIS hear me right now".
    # Re-check ownership immediately before EACH capture (here and per
    # alternate below): the standby loop's flag drops briefly between
    # record_speech calls and this probe spans ~1s, so an owner can appear
    # mid-probe. The monolith's _refresh_devices closed its own snapshot race
    # with the _pa_gate check-and-latch (2026-08-14); these per-capture
    # re-checks remain because they police a DIFFERENT hazard the gate does
    # not — opening a COMPETING stream while an owner is live (the WASAPI
    # double-open contention of crash-fix-3).
    if _mic_owned():
        details["live_capture_skipped"] = (
            "mic/audio stream went live before capture; skipping the live "
            "capture (crash-fix-3 / no-mic guard)")
        return _mic_result_without_capture(
            start, details, tts_live=_owner_flag("_tts_playback_active"))

    # TEARDOWN-GATE CLAIM (2026-08-14): hold bobert's _diag_capture_active
    # refcount across the WHOLE capture span (this capture and every
    # alternate below) via bc._pa_claim_owner, so _refresh_devices cannot run its
    # destructive sd._terminate()/_initialize() under one of these ~0.25s
    # captures — they set none of the classic owner flags, which is exactly
    # how the probe used to get PortAudio torn down beneath a live capture
    # callback (0xc0000374). Degrades to the old unguarded behaviour when the
    # host predates the gate API (SimpleNamespace doubles, older monoliths);
    # a False claim (reinit in flight past the bounded wait) skips the
    # captures with the same benign detail as a live owner. Released in the
    # finally below, which brackets every capture INCLUDING the mid-scan
    # early returns.
    _diag_release = None
    if bc is not None:
        _claim_fn = getattr(bc, "_pa_claim_owner", None)
        _release_fn = getattr(bc, "_pa_release_owner", None)
        _diag_cell = getattr(bc, "_diag_capture_active", None)
        if (callable(_claim_fn) and callable(_release_fn)
                and isinstance(_diag_cell, list) and _diag_cell):
            try:
                if not _claim_fn(_diag_cell, refcount=True, timeout=0.5):
                    details["live_capture_skipped"] = (
                        "PortAudio reinit in flight; skipping the live "
                        "capture (teardown gate / no-mic guard)")
                    return _mic_result_without_capture(
                        start, details,
                        tts_live=_owner_flag("_tts_playback_active"))
                _diag_release = (_release_fn, _diag_cell)
            except Exception:
                _diag_release = None   # gate bookkeeping failed — old behaviour
    try:
        active_rms, active_err = _capture_rms(active_idx)
        details["active_rms"] = round(active_rms, 5) if active_rms is not None else None
        if active_err:
            details["active_capture_error"] = active_err
        if active_rms is not None and active_rms >= MIC_RMS_FLOOR:
            details["rms"] = round(active_rms, 5)  # back-compat field
            return _result(True, (_now() - start) * 1000.0, details=details)

        # Step 2: active mic is silent (or capture raised). Scan a small set
        # of physical alternates so we can distinguish "audio stack broken
        # entirely" from "user's preferred mic is muted/off but the box has
        # other working inputs". Skip virtual devices (Steam Streaming, Sound
        # Mapper, etc.) and dedupe by name root so we don't try the same
        # physical mic four times via different hostapis.
        seen_names: set[str] = set()
        alternates: list[tuple[int, str]] = []
        for idx, dev in inputs:
            if idx == active_idx:
                continue
            name = (dev.get("name") or "").strip()
            if not name or _VIRTUAL_INPUT_RE.search(name):
                continue
            key = name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            alternates.append((idx, name))

        MAX_ALTERNATES = 4
        alternates_tried: list[dict] = []
        best_rms = active_rms if active_rms is not None else 0.0
        best_idx = active_idx if active_rms is not None else None
        best_name: str | None = None
        if active_idx is not None:
            try:
                best_name = devices[active_idx]["name"]
            except Exception:  # pragma: no cover - defensive: device-name subscript when scanning mic alternates (audio I/O, sounddevice absent on CI)
                best_name = None
        for idx, name in alternates[:MAX_ALTERNATES]:
            if _mic_owned():
                # An owner grabbed the mic mid-scan (e.g. standby's next
                # record_speech turn) — stop opening devices immediately and
                # report a benign skip rather than a false "silent mic"
                # verdict built from contended captures.
                details["live_capture_skipped"] = (
                    "mic/audio stream went live mid-scan; skipping remaining "
                    "captures (crash-fix-3 / no-mic guard)")
                # partial_silence: the ACTIVE device already measured below the
                # floor, so this is not a clean skip — a passive "alive" may
                # not launder it into a pass.
                return _mic_result_without_capture(
                    start, details, partial_silence=True,
                    tts_live=_owner_flag("_tts_playback_active"))
            rms, err = _capture_rms(idx, duration_s=0.15)
            alternates_tried.append({
                "index": idx, "name": name,
                "rms": round(rms, 5) if rms is not None else None,
                "error": err,
            })
            if rms is not None and rms > best_rms:
                best_rms = rms
                best_idx = idx
                best_name = name
    finally:
        # Release the diag refcount AFTER the last capture’s stream has been
        # torn down — close-then-release, so the flag covers the stream’s
        # whole native lifetime. _capture_rms closes its private InputStream in
        # its own finally before returning, so by here every native handle this
        # probe opened is closed (or registered as an abandoned in-flight close
        # with the host’s _pa_close_pending gate). Runs on every exit: normal
        # fall-through, the loud-active early return, and the mid-scan
        # owner-appeared returns above.
        #
        # H-4 (2026-08-20): the note that used to sit here — "sd.rec+sd.wait
        # are synchronous" — was the disproven assertion that licensed this
        # release. sd.wait() is unbounded, and the enroll_voice twin was fixed
        # for that on 2026-07-14; the captures no longer use either call.
        if _diag_release is not None:
            try:
                _diag_release[0](_diag_release[1], refcount=True)
            except Exception:
                pass
    details["rms"] = round(best_rms, 5)
    if alternates_tried:
        details["alternates_tried"] = alternates_tried

    # Step 3: all probed inputs were silent. This is almost always an
    # environmental condition (Windows mixer muted, wireless headset off
    # / out of battery, no signal source plugged into the active jack)
    # that can't be auto-fixed from code. Downgrade to LOW so the
    # self-heal pipeline stops re-queueing the same repair task every
    # half hour — mirrors the webcam-absent path above.
    hw_count = _windows_microphone_hardware_count()
    if hw_count is not None:
        details["pnp_mic_count"] = hw_count

    if best_rms >= MIC_RMS_FLOOR and best_idx != active_idx:
        # An alternate mic IS producing signal — JARVIS's preferred
        # device just isn't. Actionable: power on the headset, or update
        # bobert_companion.PREFERRED_INPUT_DEVICES. Still LOW because
        # it's user state, not a code bug.
        active_name = details.get("active_input_name") or f"index {active_idx}"
        return _result(False, (_now() - start) * 1000.0,
                       error=f"JARVIS's active mic ({active_name}) is silent "
                             f"but alternate {best_name!r} (index {best_idx}) "
                             f"has signal (RMS {best_rms:.5f}). "
                             f"Likely the preferred device is muted, the "
                             f"wireless headset is off, or you need to update "
                             f"PREFERRED_INPUT_DEVICES.",
                       details=details,
                       severity=SEVERITY_LOW)

    if hw_count == 0:
        # PnP confirms there's nothing mic-shaped on this box. Rare
        # (sounddevice enumerated something), but if it ever happens,
        # don't pester the upgrade pipeline.
        return _result(False, (_now() - start) * 1000.0,
                       error="no microphone hardware detected (Windows PnP "
                             "reports 0 mic-class endpoints) — none "
                             "installed or all disabled",
                       details=details,
                       severity=SEVERITY_LOW)

    # Hardware present, every input silent. User has muted things /
    # turned off the headset / unplugged the jack. Environmental.
    pnp_hint = "" if hw_count is None else f" ({hw_count} mic device(s) present per PnP)"
    return _result(False, (_now() - start) * 1000.0,
                   error=f"all {len(inputs)} input devices silent — best RMS "
                         f"{best_rms:.5f} < floor {MIC_RMS_FLOOR:.5f}. Mic "
                         f"muted, wireless headset off, or no signal source "
                         f"plugged into the active input{pnp_hint}.",
                   details=details,
                   severity=SEVERITY_LOW)


# ─── Probe 3: TTS ────────────────────────────────────────────────────────
def _probe_tts() -> dict:
    start = _now()
    details: dict[str, Any] = {}

    # edge-tts is HTTP-only; we don't need to actually synthesise audio,
    # just verify the CDN responds.
    edge_ok = False
    edge_err: str | None = None
    try:
        import requests  # type: ignore
        # edge-tts uses speech.platform.bing.com for the WebSocket; a simple
        # GET against the public token endpoint will tell us if Microsoft
        # is reachable at all.
        r = requests.get("https://speech.platform.bing.com/", timeout=4)
        # Any response (even 400/404) means we reached Microsoft's CDN.
        details["edge_status"] = r.status_code
        edge_ok = True
    except Exception as e:
        edge_err = f"{type(e).__name__}: {e}"
        details["edge_status"] = edge_err

    # pyttsx3 — offline fallback. We initialise the engine but don't
    # actually speak (the test machine may have no audio output).
    pyttsx_ok = False
    pyttsx_err: str | None = None
    try:
        import pyttsx3  # type: ignore
        eng = pyttsx3.init()
        # Probe one voice property to confirm the SAPI/NSSpeechSynthesizer
        # bridge actually came up.
        _ = eng.getProperty("voices")
        try:
            eng.stop()
        except Exception:
            pass
        pyttsx_ok = True
    except Exception as e:
        pyttsx_err = f"{type(e).__name__}: {e}"

    details["edge_ok"]   = edge_ok
    details["pyttsx_ok"] = pyttsx_ok
    if edge_err:
        details["edge_error"] = edge_err
    if pyttsx_err:
        details["pyttsx_error"] = pyttsx_err

    if edge_ok or pyttsx_ok:
        # At least one TTS backend works — call the probe a success even
        # if the other is degraded (we'll annotate which in details).
        sev = None if (edge_ok and pyttsx_ok) else SEVERITY_LOW
        return _result(True, (_now() - start) * 1000.0,
                       details=details, severity=sev)

    return _result(False, (_now() - start) * 1000.0,
                   error=f"both TTS backends failed (edge: {edge_err}; pyttsx3: {pyttsx_err})",
                   details=details)


# ─── Probe 4: STT ────────────────────────────────────────────────────────
# Substrings (lowercased) in a Whisper exception that indicate a missing /
# unloadable CUDA runtime DLL rather than a code bug. When we see one we
# downgrade severity to LOW (auto-fix can't ship a DLL) and emit the pip
# remediation hint so the user knows what to actually run.
_STT_CUDA_DLL_PATTERNS = (
    "cublas64", "cudnn64", "cudart64", "nvcuda.dll",
    "is not found or cannot be loaded",
    "could not load library", "library not found",
)


def _is_stt_cuda_dll_error(exc: BaseException) -> bool:
    s = f"{type(exc).__name__}: {exc}".lower()
    return any(p in s for p in _STT_CUDA_DLL_PATTERNS)


def _stt_cuda_remediation_note() -> str:
    """Single-line hint surfaced when a CUDA DLL load failure is detected.
    Mirrors the note bobert_companion._cuda_dll_remediation_note() emits
    so the diagnostic report and the boot log read the same."""
    bc = _bc()
    fn = getattr(bc, "_cuda_dll_remediation_note", None) if bc else None
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass
    return ("CUDA runtime DLLs (cublas64_12.dll / cudnn64_9.dll) are not "
            "loadable. Fix: pip install --upgrade nvidia-cublas-cu12 "
            "nvidia-cudnn-cu12  (or set WHISPER_DEVICE='cpu' to skip GPU).")


def _probe_stt() -> dict:
    start = _now()
    details: dict[str, Any] = {}

    bc = _bc()
    # If JARVIS has already loaded a Whisper model on the main thread,
    # reuse it — no point loading a second copy just for this probe.
    cached_model = getattr(bc, "_stt", None) if bc else None
    cached_name  = getattr(bc, "_stt_model_name", None) if bc else None
    cached_dev   = getattr(bc, "_stt_device", None) if bc else None

    if cached_model is None:
        # Try to load a tiny model just to verify the lib + weights are
        # available. We deliberately use the tiny model so this probe is
        # cheap even when the main loop hasn't loaded Whisper yet.
        try:
            import whisper as _wlib  # type: ignore
        except Exception as e:
            return _result(False, (_now() - start) * 1000.0,
                           error=f"whisper not importable: {e}")
        try:
            model = _wlib.load_model("tiny")
            details["model_loaded"] = "tiny (probe-local)"
        except Exception as e:
            return _result(False, (_now() - start) * 1000.0,
                           error=f"whisper.load_model('tiny') failed: {type(e).__name__}: {e}")
    else:
        model = cached_model
        details["model_loaded"] = f"{cached_name} ({cached_dev}) [cached from main loop]"

    # Synthesize a 1-second 440Hz sine, hand it to Whisper. We don't
    # assert on the transcription content — Whisper happily returns "" on
    # pure tones — we just verify .transcribe() runs without raising.
    #
    # Adapt to whichever engine the main loop is using: faster-whisper
    # (WhisperModel from faster_whisper) returns (segments_gen, info);
    # openai-whisper returns a dict. Detect by class name to avoid the
    # import dance.
    def _do_transcribe(m):
        import numpy as np  # type: ignore
        sr = 16000
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        if type(m).__name__ == "WhisperModel":
            # faster-whisper path
            segs_gen, _info = m.transcribe(audio, language="en")
            segs = list(segs_gen)
            return " ".join((s.text or "").strip() for s in segs).strip()
        else:
            # openai-whisper path. Modern openai-whisper (v20250115+) removed
            # the fp16= kwarg — precision is now set at load time, so passing
            # it raises TypeError on current installs.
            result = m.transcribe(audio, language="en")
            return (result.get("text") or "").strip()

    # 2026-07-21 audit: when reusing the main loop's cached model we MUST
    # hold bc._stt_lock across the ENTIRE call — the .transcribe() AND the
    # generator drain inside _do_transcribe — because two threads driving
    # one CTranslate2 instance corrupt native state and terminate the
    # process with an uncatchable 0xc0000409 (see bobert_companion
    # .transcribe(), the lock's contract). Bounded acquire: contention is
    # not a failure — but it is not a pass either (2026-08-20). A held lock
    # proves a transcription is IN FLIGHT, not that it will succeed, and this
    # probe did not run one: report UNVERIFIED. The
    # probe-local models (tiny above, CPU fallback below) stay unlocked —
    # they are private to this probe thread. Deliberately NOT routed
    # through bc.transcribe(): _transcribe_impl swallows every exception
    # and returns ("", ...), which would bypass the _is_stt_cuda_dll_error
    # classification below and turn real STT failures (including the
    # CUDA-DLL environmental case) into false-passing probes.
    using_cached = cached_model is not None and model is cached_model
    lock = getattr(bc, "_stt_lock", None) if using_cached else None
    if lock is not None and not lock.acquire(timeout=5.0):
        return _unverified((_now() - start) * 1000.0,
                           reason=("the main loop holds _stt_lock (a "
                                   "transcription is in flight), so the probe "
                                   "never ran one of its own"),
                           details={**details,
                                    "skipped": "main loop holds _stt_lock "
                                               "(STT busy) — probe deferred"},
                           short_cause=_UNVERIFIED_SHORT_CAUSES["stt_busy"],
                           transient=True)
    try:
        if lock is not None:
            try:
                text = _do_transcribe(model)
            finally:
                lock.release()
        else:
            text = _do_transcribe(model)
        details["transcribed_text"] = text[:60]
    except Exception as e:
        # CUDA DLL load failures are environmental: the GPU runtime
        # libraries (cublas64_12.dll / cudnn64_9.dll) couldn't be loaded
        # by ctranslate2. Auto-repair can't fix this — the user must
        # pip-reinstall the nvidia-cublas-cu12 / nvidia-cudnn-cu12
        # wheels or set WHISPER_DEVICE='cpu'. Downgrade to LOW so the
        # self-heal pipeline stops queueing the same repair task every
        # half hour, and try a CPU fallback so STT keeps working in the
        # meantime.
        if _is_stt_cuda_dll_error(e):
            details["failure_mode"]   = "cuda_dll_missing"
            details["auto_repairable"] = False
            details["remediation"]    = _stt_cuda_remediation_note()
            details["original_error"] = f"{type(e).__name__}: {e}"

            # Best-effort CPU fallback so we can at least confirm the
            # STT pipeline works on CPU and report degraded-but-functional.
            cpu_ok = False
            cpu_note = ""
            try:
                from faster_whisper import WhisperModel as _FWM  # type: ignore
                cpu_model = _FWM("tiny", device="cpu", compute_type="int8")
                _do_transcribe(cpu_model)
                cpu_ok = True
                cpu_note = "tiny model on CPU transcribed cleanly"
            except Exception as cpu_e:
                cpu_note = f"CPU fallback also failed: {type(cpu_e).__name__}: {cpu_e}"
            details["cpu_fallback_ok"]   = cpu_ok
            details["cpu_fallback_note"] = cpu_note

            if cpu_ok:
                _maybe_announce_once(
                    "stt_cuda_dll_missing",
                    "Sir, the GPU speech recogniser is offline — its CUDA "
                    "libraries can't be loaded. I'll fall back to CPU until "
                    "you reinstall nvidia-cublas-cu12 and nvidia-cudnn-cu12, "
                    "or set WHISPER_DEVICE to 'cpu'.",
                )
            else:
                _maybe_announce_once(
                    "stt_cuda_dll_missing_and_cpu_broken",
                    "Sir, speech recognition is down — the CUDA libraries "
                    "can't be loaded and the CPU fallback also failed.",
                )

            return _result(False, (_now() - start) * 1000.0,
                           error=(f"whisper.transcribe failed: "
                                  f"{type(e).__name__}: {e}. "
                                  f"{_stt_cuda_remediation_note()}"),
                           details=details,
                           severity=SEVERITY_LOW)
        return _result(False, (_now() - start) * 1000.0,
                       error=f"whisper.transcribe failed: {type(e).__name__}: {e}",
                       details=details)

    return _result(True, (_now() - start) * 1000.0, details=details)


# ─── Probe 5: Claude API ─────────────────────────────────────────────────
# SDK-level timeout for the 1-token ping. Must stay strictly below
# PER_PROBE_TIMEOUT_S so the SDK raises (with a real error string) before
# the outer thread.join() abandons the probe. See PER_PROBE_TIMEOUT_S
# for the history behind the 12 s value.
_CLAUDE_API_PROBE_TIMEOUT_S = 12.0


def _probe_claude_api() -> dict:
    start = _now()
    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        # No key configured = no point probing. Reported as a benign skip,
        # not a failure, because some users intentionally run JARVIS local-only.
        return _result(True, 0.0, details={"skipped": "ANTHROPIC_API_KEY not set"})

    try:
        import anthropic  # type: ignore
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"anthropic SDK not importable: {e}")

    bc = _bc()
    model = (getattr(bc, "CLAUDE_MODEL", None) or
             os.environ.get("CLAUDE_MODEL") or
             "claude-haiku-4-5")

    try:
        # SDK-level timeout (httpx) covers the request once it's actually
        # issued. TLS handshake / DNS happen before the timer starts, so
        # in practice the outer PER_PROBE_TIMEOUT_S is what catches a
        # full network stall. Bumped 6→12 (2026-05-30) after a transient
        # stall hit the previous 8 s outer cap; observed p95 across 99
        # successful runs is 3.2 s, so 12 s leaves comfortable headroom
        # without masking a genuinely-dead endpoint.
        client = anthropic.Anthropic(timeout=_CLAUDE_API_PROBE_TIMEOUT_S)
        client.messages.create(
            model=model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
            timeout=_CLAUDE_API_PROBE_TIMEOUT_S,
        )
    except Exception as e:
        # Distinguish timeout / network failures (environmental — no code
        # fix possible) from API-side failures (auth, rate-limit, bad
        # model name — actionable). Class-name match keeps the probe
        # working across anthropic SDK revs even when the specific
        # exception subclass moves.
        ename = type(e).__name__
        emsg  = str(e)
        is_timeout = ename in ("APITimeoutError", "ReadTimeout",
                               "ConnectTimeout", "WriteTimeout",
                               "Timeout", "TimeoutError")
        is_network = ename in ("APIConnectionError", "ConnectionError",
                               "ConnectError", "RemoteProtocolError")
        if is_timeout:
            return _result(False, (_now() - start) * 1000.0,
                           error=(f"Claude API ping timed out: {ename}: {emsg}. "
                                  f"SDK call did not return within "
                                  f"{_CLAUDE_API_PROBE_TIMEOUT_S:.0f}s — likely "
                                  f"a slow link, captive portal, corporate "
                                  f"proxy, or transient anthropic.com latency. "
                                  f"Environmental; no auto-fix from code."),
                           details={"model": model,
                                    "failure_mode": "network_timeout",
                                    "auto_repairable": False,
                                    "sdk_timeout_s": _CLAUDE_API_PROBE_TIMEOUT_S})
        if is_network:
            return _result(False, (_now() - start) * 1000.0,
                           error=(f"Claude API ping unreachable: {ename}: {emsg}. "
                                  f"DNS or TCP/TLS to api.anthropic.com is "
                                  f"failing; see the internet probe for the "
                                  f"underlying connectivity state."),
                           details={"model": model,
                                    "failure_mode": "network_unreachable",
                                    "auto_repairable": False})
        return _result(False, (_now() - start) * 1000.0,
                       error=f"Claude API ping failed: {ename}: {emsg}",
                       details={"model": model,
                                "failure_mode": "api_error"})

    return _result(True, (_now() - start) * 1000.0,
                   details={"model": model})


# ─── Probe 6: internet ───────────────────────────────────────────────────
def _probe_internet() -> dict:
    start = _now()
    details: dict[str, Any] = {}

    # DNS
    dns_ok = False
    dns_err: str | None = None
    try:
        ip = socket.gethostbyname("api.anthropic.com")
        details["api_anthropic_com"] = ip
        dns_ok = True
    except Exception as e:
        dns_err = f"{type(e).__name__}: {e}"

    # Ping 1.1.1.1 — uses subprocess so we don't need raw-socket privileges.
    ping_ok = False
    ping_err: str | None = None
    try:
        if sys.platform == "win32":
            cmd = ["ping", "-n", "1", "-w", "2000", "1.1.1.1"]
        else:
            cmd = ["ping", "-c", "1", "-W", "2", "1.1.1.1"]
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=4.0,
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        )
        ping_ok = (proc.returncode == 0)
        details["ping_rc"] = proc.returncode
        if not ping_ok:
            ping_err = (proc.stdout or proc.stderr or "").strip()[:160] or "non-zero exit"
    except Exception as e:
        ping_err = f"{type(e).__name__}: {e}"

    details["dns_ok"]  = dns_ok
    details["ping_ok"] = ping_ok
    if dns_err:
        details["dns_error"] = dns_err
    if ping_err:
        details["ping_error"] = ping_err

    if dns_ok and ping_ok:
        return _result(True, (_now() - start) * 1000.0, details=details)
    if dns_ok or ping_ok:
        # Half-internet: DNS reachable but ICMP filtered (common on
        # corporate networks) is still functional for our purposes.
        return _result(True, (_now() - start) * 1000.0,
                       details=details, severity=SEVERITY_LOW)
    return _result(False, (_now() - start) * 1000.0,
                   error=f"DNS and ICMP both failed (dns: {dns_err}; ping: {ping_err})",
                   details=details)


# ─── Probe 7: HUD subprocesses ───────────────────────────────────────────
def _probe_hud_subprocesses() -> dict:
    start = _now()
    bc = _bc()
    if bc is None:
        # No host module → no process handles to inspect. That is "I cannot
        # see the HUDs", not "the HUDs are fine" (2026-08-20).
        return _unverified(0.0,
                           reason=("bobert_companion is not loaded, so the HUD "
                                   "process handles cannot be inspected"),
                           details={"skipped": "bobert_companion not loaded"},
                           short_cause=_UNVERIFIED_SHORT_CAUSES["host_not_loaded"])

    details: dict[str, Any] = {}
    alive: list[str] = []
    dead:  list[str] = []

    # ── jarvis_hud, jarvis_reticle, tray ── managed by bobert_companion
    for varname, hud_name in (("_hud_process",      "jarvis_hud"),
                              ("_reticle_process",  "jarvis_reticle"),
                              ("_tray_process",     "tray")):
        proc = getattr(bc, varname, None)
        if proc is None:
            # Variable exists but no subprocess spawned — could be that
            # the feature is disabled. Not a failure.
            details[hud_name] = "not-spawned"
            continue
        try:
            rc = proc.poll()
            if rc is None:
                alive.append(hud_name)
                details[hud_name] = f"alive (pid {proc.pid})"
            else:
                dead.append(hud_name)
                details[hud_name] = f"DEAD (exit {rc})"
        except Exception as e:
            dead.append(hud_name)
            details[hud_name] = f"poll-failed: {e}"

    # ── workshop_hud ── managed by skills/holographic_overlay.py
    overlay = sys.modules.get("skill_holographic_overlay")
    if overlay is not None:
        is_alive = getattr(overlay, "_workshop_hud_is_alive", None)
        if callable(is_alive):
            try:
                if is_alive():
                    alive.append("workshop_hud")
                    details["workshop_hud"] = "alive"
                else:
                    # Not alive isn't necessarily a failure here — the
                    # workshop HUD is opt-in. Only flag if a state file
                    # says it was spawned. Treat the not-spawned case as
                    # benign.
                    details["workshop_hud"] = "not-spawned"
            except Exception as e:
                dead.append("workshop_hud")
                details["workshop_hud"] = f"poll-failed: {e}"

    if dead:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"HUD subprocess(es) down: {', '.join(dead)}",
                       details=details)
    return _result(True, (_now() - start) * 1000.0, details=details)


# ─── Probe 8: state file integrity ───────────────────────────────────────
def _probe_state_files() -> dict:
    start = _now()
    bad: list[dict] = []
    parsed = 0
    # Walk the project root for top-level .json files. We deliberately
    # skip nested directories (data/, logs/, backups/) because those hold
    # rolling histories where a partial write is acceptable mid-flight;
    # the canonical state lives at the project root.
    try:
        entries = os.listdir(_PROJECT_DIR)
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"could not list project root: {e}")

    # Skip files modified in the last 30 s. The pipeline writes state via
    # `core.atomic_io._atomic_write_json` (os.replace), so a torn JSON is
    # impossible — but a writer can still have the rename in flight when
    # the boot sweep fires 60 s after launch, surfacing as a transient
    # PermissionError on Windows. Cooling off recently-touched files keeps
    # the probe honest without raising false alarms.
    now_ts = _now()
    skipped_recent: list[str] = []
    for name in entries:
        if not name.endswith(".json"):
            continue
        path = os.path.join(_PROJECT_DIR, name)
        if not os.path.isfile(path):
            continue
        # Skip files we know are touched mid-write by other processes —
        # the atomic_io path makes these safe, but very brief windows can
        # still happen if something raced.
        if name in {"pending_speech.json"}:
            continue
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = 0.0
        if now_ts - mtime < 30.0:
            skipped_recent.append(name)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
            parsed += 1
        except Exception as e:
            bad.append({"file": name, "error": f"{type(e).__name__}: {e}"})

    details = {"parsed": parsed, "bad_files": bad,
               "skipped_recent": skipped_recent}
    if bad:
        files = ", ".join(b["file"] for b in bad)
        return _result(False, (_now() - start) * 1000.0,
                       error=f"{len(bad)} state file(s) failed to parse: {files}",
                       details=details)
    return _result(True, (_now() - start) * 1000.0, details=details)


# ─── Probe 9: Bambu MQTT ─────────────────────────────────────────────────
def _probe_bambu() -> dict:
    start = _now()
    bc = _bc()
    if bc is None:
        # Unlike the two skips below — which read real config and conclude
        # there is nothing to talk to — this one cannot even read the config,
        # so it must not claim the printer is reachable (2026-08-20).
        return _unverified(0.0,
                           reason=("bobert_companion is not loaded, so the "
                                   "printer settings cannot even be read"),
                           details={"skipped": "bobert_companion not loaded"},
                           short_cause=_UNVERIFIED_SHORT_CAUSES["host_not_loaded"])

    ip      = (getattr(bc, "BAMBU_PRINTER_IP", "")   or "").strip()
    access  = (getattr(bc, "BAMBU_ACCESS_CODE", "")  or "").strip()
    serial  = (getattr(bc, "BAMBU_SERIAL", "")       or "").strip()
    if not (ip and access and serial):
        return _result(True, 0.0,
                       details={"skipped": "Bambu printer not configured"})

    # If bambu_monitor has already decided the printer is offline/asleep,
    # skip the 5s MQTT connect entirely. Otherwise this probe fires on
    # every boot sweep and spams "Bambu MQTT connect timed out (5s)" as a
    # LOW-severity FAIL even when the printer is just powered down.
    try:
        from skills import bambu_monitor as _bm  # type: ignore
        if getattr(_bm, "is_printer_offline", None) and _bm.is_printer_offline():
            return _result(True, 0.0,
                           details={"skipped": "printer offline (monitor backed off)",
                                    "ip": ip})
    except Exception:
        # If we can't import or query, just fall through to the real probe.
        pass

    try:
        import paho.mqtt.client as mqtt  # type: ignore
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"paho-mqtt not installed: {e}",
                       severity=SEVERITY_MED)

    connect_event = threading.Event()
    box: dict[str, Any] = {"rc": None}

    def _on_connect(client, userdata, flags, rc, properties=None):
        box["rc"] = rc
        connect_event.set()

    try:
        client = mqtt.Client(client_id=f"jarvis-diag-{os.getpid()}", protocol=mqtt.MQTTv311)
        client.username_pw_set("bblp", access)
        client.tls_set_context(__import__("ssl").create_default_context())
        # Bambu's self-signed certs — accept them.
        client.tls_insecure_set(True)
        client.on_connect = _on_connect
        client.connect_async(ip, 8883, keepalive=10)
        client.loop_start()
        connect_event.wait(timeout=5.0)
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass
        if box["rc"] is None:
            return _result(False, (_now() - start) * 1000.0,
                           error="Bambu MQTT connect timed out (5s)",
                           details={"ip": ip})
        if box["rc"] != 0:
            return _result(False, (_now() - start) * 1000.0,
                           error=f"Bambu MQTT rc={box['rc']} (1=bad protocol, 4=bad creds, 5=not authorised)",
                           details={"ip": ip, "rc": box["rc"]})
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"Bambu MQTT connect raised: {type(e).__name__}: {e}",
                       details={"ip": ip})

    return _result(True, (_now() - start) * 1000.0, details={"ip": ip})


# ─── Probe 10: media playback target ─────────────────────────────────────
def _probe_media_playback() -> dict:
    start = _now()
    details: dict[str, Any] = {}

    # Chrome — check the install path is on disk. This is what
    # bobert_companion uses for spotify/web playback fallback.
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    chrome_ok = any(os.path.exists(p) for p in chrome_paths)
    details["chrome"] = "found" if chrome_ok else "missing"

    # Apple Music for Windows ships as a UWP app; the executable name to
    # look for is AppleMusic.exe. We check Program Files first and then
    # any running process by name as a backup.
    am_ok = False
    am_paths = [
        os.path.expandvars(r"%LOCALAPPDATA%\Apple\AppleMusic\AppleMusic.exe"),
        r"C:\Program Files\WindowsApps\AppleInc.AppleMusicWin_*",
    ]
    am_ok = any(os.path.exists(p) for p in am_paths if "*" not in p)
    if not am_ok:
        # Glob over the WindowsApps wildcard
        try:
            import glob as _glob
            for p in am_paths:
                if "*" in p and _glob.glob(p):
                    am_ok = True
                    break
        except Exception:
            pass
    if not am_ok:
        # Check running processes as a last resort — Apple Music may be
        # installed in a path we don't recognise.
        try:
            import psutil  # type: ignore
            for proc in psutil.process_iter(["name"]):
                try:
                    if (proc.info.get("name") or "").lower() in (
                            "applemusic.exe", "music.exe"):
                        am_ok = True
                        break
                except Exception:
                    continue
        except Exception:
            pass
    details["apple_music"] = "found" if am_ok else "not-detected"

    # We pass the probe as long as *some* playback target exists. Both
    # missing is reported as MED — neither lo-fi nor podcasts work then.
    if chrome_ok or am_ok:
        return _result(True, (_now() - start) * 1000.0, details=details)
    return _result(False, (_now() - start) * 1000.0,
                   error="no playback target detected (Chrome + Apple Music both missing)",
                   details=details,
                   severity=SEVERITY_MED)


# ─── Probe 11: skill imports ─────────────────────────────────────────────
def _compile_skill_source(path: str) -> str | None:
    """Compile (never exec — exec has side effects: daemon threads, network
    calls) a skill's source. Returns an error string, or None when it parses."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        compile(source, path, "exec")
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _probe_skill_imports() -> dict:
    start = _now()
    skills_dir = os.path.join(_PROJECT_DIR, "skills")
    if not os.path.isdir(skills_dir):
        return _result(False, (_now() - start) * 1000.0,
                       error=f"skills directory missing at {skills_dir}")

    bc = _bc()
    if bc is not None and not getattr(bc, "SKILLS_ENABLED", True):
        # Skills are switched off entirely — nothing SHOULD be loaded, so
        # flagging every on-disk stem would be pure noise.
        return _result(True, (_now() - start) * 1000.0,
                       details={"skipped": "skills disabled"})
    loaded = getattr(bc, "_loaded_skill_names", None) if bc is not None else None
    if not isinstance(loaded, set):
        loaded = None

    failures: list[dict] = []
    checked = 0

    try:
        entries = sorted(os.listdir(skills_dir))
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"could not list skills dir: {e}")

    # Enumerate the SAME entries load_skills() enumerates: package dirs
    # with an __init__.py (which take import precedence) plus flat *.py
    # stems not shadowed by a package. The probe historically only saw
    # flat modules, so package skills were never checked at all.
    to_check: list[tuple[str, str]] = []   # (stem, source path)
    pkg_stems: set[str] = set()
    for name in entries:
        sub = os.path.join(skills_dir, name)
        if not os.path.isdir(sub) or name.startswith("_") or name == "__pycache__":
            continue
        init_path = os.path.join(sub, "__init__.py")
        if os.path.isfile(init_path):
            to_check.append((name, init_path))
            pkg_stems.add(name)
    for name in entries:
        if not name.endswith(".py") or name.startswith("_"):
            continue
        stem = name[:-3]
        if stem in pkg_stems:
            continue   # loader uses the package, not the flat file
        to_check.append((stem, os.path.join(skills_dir, name)))

    for stem, path in to_check:
        checked += 1

        if loaded is not None:
            # JARVIS live: cross-check the loader's own success set.
            # A sys.modules["skill_<stem>"] entry is NOT proof of health —
            # load_skills registers the module BEFORE exec (so package
            # sub-imports resolve), so a skill whose module body or
            # register() raised used to leave a half-initialized module
            # there and this probe called it green (2026-07-21 audit).
            # _loaded_skill_names is only added to after successful
            # exec + register, so it is the authoritative record.
            if stem in loaded:
                continue
            err = _compile_skill_source(path)
            if err is None:
                err = ("on disk but not loaded at boot — import/register "
                       "raised, see boot log; run reload_skills if newly added")
            failures.append({"skill": stem, "error": err})
            continue

        # Loader set unavailable (probe running outside JARVIS, e.g.
        # standalone pytest): fall back to trusting a live sys.modules
        # entry, and verifying that unloaded files at least parse.
        if sys.modules.get(f"skill_{stem}") is not None:
            continue
        err = _compile_skill_source(path)
        if err is not None:
            failures.append({"skill": stem, "error": err})

    details = {"checked": checked, "failures": failures,
               "loaded_modules": sum(1 for k in sys.modules if k.startswith("skill_")),
               # Honest-failure contract: say which check actually ran. The
               # fallback is genuinely WEAKER — it proves the files parse, not
               # that they imported and registered — so a pass from it must
               # not read like a pass from the authoritative loader set.
               "verification": ("loader success set (_loaded_skill_names)"
                                if loaded is not None
                                else "parse-only fallback — loader success set "
                                     "unavailable (probe running outside "
                                     "JARVIS); import/register was NOT proven")}
    if loaded is None:
        _log.info("self-diagnostic: skill_imports ran its parse-only fallback "
                  "— _loaded_skill_names unavailable, so import/register "
                  "success is unproven for %d skill(s).", checked)

    if failures:
        names = ", ".join(f["skill"] for f in failures)
        return _result(False, (_now() - start) * 1000.0,
                       error=f"{len(failures)} skill(s) failed to load: {names}",
                       details=details)
    return _result(True, (_now() - start) * 1000.0, details=details)


# ─── Probe 12: GPU ───────────────────────────────────────────────────────
def _probe_gpu() -> dict:
    start = _now()
    bc = _bc()
    whisper_device = (getattr(bc, "WHISPER_DEVICE", "auto") or "auto").lower() if bc else "auto"
    needs_cuda = whisper_device in ("cuda", "auto")
    details: dict[str, Any] = {"whisper_device": whisper_device}

    try:
        import torch  # type: ignore
    except Exception as e:
        if needs_cuda and whisper_device == "cuda":
            return _result(False, (_now() - start) * 1000.0,
                           error=f"torch not importable: {e}",
                           details=details)
        # auto-mode + no torch = CPU fallback, which is fine.
        return _result(True, (_now() - start) * 1000.0,
                       details={**details, "skipped": "torch not installed"})

    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception as e:
        cuda_ok = False
        details["cuda_error"] = f"{type(e).__name__}: {e}"

    details["cuda_available"] = cuda_ok
    if cuda_ok:
        try:
            details["device_name"] = torch.cuda.get_device_name(0)
            details["vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**2))
        except Exception:
            pass

    if whisper_device == "cuda" and not cuda_ok:
        return _result(False, (_now() - start) * 1000.0,
                       error="WHISPER_DEVICE=cuda but torch.cuda.is_available() is False",
                       details=details)
    return _result(True, (_now() - start) * 1000.0, details=details)


# ─── Probe 13: disk ──────────────────────────────────────────────────────
def _probe_disk() -> dict:
    start = _now()
    try:
        import shutil
        total, used, free = shutil.disk_usage(_PROJECT_DIR)
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"shutil.disk_usage failed: {e}")
    details = {
        "total_gb": round(total / (1024**3), 1),
        "free_gb":  round(free  / (1024**3), 1),
        "free_pct": round(free * 100.0 / total, 1) if total else 0.0,
    }
    if free < DISK_FREE_FLOOR_BYTES:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"only {details['free_gb']} GB free on project drive",
                       details=details)
    return _result(True, (_now() - start) * 1000.0, details=details)


# ─── Probe 14: RAM ───────────────────────────────────────────────────────
def _probe_ram() -> dict:
    start = _now()
    try:
        import psutil  # type: ignore
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"psutil not importable: {e}",
                       severity=SEVERITY_MED)
    try:
        vm = psutil.virtual_memory()
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"psutil.virtual_memory failed: {e}")
    details = {
        "percent":  vm.percent,
        "used_gb":  round(vm.used / (1024**3), 1),
        "total_gb": round(vm.total / (1024**3), 1),
    }
    if vm.percent >= RAM_PCT_CEILING:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"RAM at {vm.percent:.0f}% (ceiling {RAM_PCT_CEILING:.0f}%)",
                       details=details)
    return _result(True, (_now() - start) * 1000.0, details=details)


# ─── Probe 15: optional skills (Alexa / Deco placeholders) ───────────────
def _probe_optional_skills() -> dict:
    """Pass-through probe for skills that haven't landed yet (research-4a
    Alexa, research-4c Deco router). When those skills exist, we'll call
    their own ``self_diagnostic`` hooks; until then this just reports
    "not-loaded" without flagging it as a failure."""
    start = _now()
    details: dict[str, Any] = {}

    # Alexa (research-4a) — check for the skill module + cookie file
    alexa_mod = sys.modules.get("skill_alexa") or sys.modules.get("skill_alexa_voice")
    if alexa_mod is None:
        details["alexa"] = "skill not loaded"
    else:
        hook = getattr(alexa_mod, "diagnostic_probe", None)
        if callable(hook):
            try:
                details["alexa"] = hook()
            except Exception as e:
                details["alexa"] = f"probe-raised: {type(e).__name__}: {e}"
        else:
            details["alexa"] = "loaded, no probe hook"

    # Deco router (research-4c)
    deco_mod = sys.modules.get("skill_network_deco")
    if deco_mod is None:
        details["deco"] = "skill not loaded"
    else:
        hook = getattr(deco_mod, "diagnostic_probe", None)
        if callable(hook):
            try:
                details["deco"] = hook()
            except Exception as e:
                details["deco"] = f"probe-raised: {type(e).__name__}: {e}"
        else:
            details["deco"] = "loaded, no probe hook"

    return _result(True, (_now() - start) * 1000.0, details=details)


# ─── Probe registry ──────────────────────────────────────────────────────
PROBES: dict[str, Callable[[], dict]] = {
    "webcam":           _probe_webcam,
    "microphone":       _probe_microphone,
    "tts":              _probe_tts,
    "stt":              _probe_stt,
    "claude_api":       _probe_claude_api,
    "internet":         _probe_internet,
    "hud_subprocesses": _probe_hud_subprocesses,
    "state_files":      _probe_state_files,
    "bambu":            _probe_bambu,
    "media_playback":   _probe_media_playback,
    "skill_imports":    _probe_skill_imports,
    "gpu":              _probe_gpu,
    "disk":             _probe_disk,
    "ram":              _probe_ram,
    "optional_skills":  _probe_optional_skills,
}


# ─── Sweep + persistence ─────────────────────────────────────────────────
def _run_all_probes() -> dict:
    """Run every probe sequentially and return the aggregated run dict.

    Three buckets, not two. A probe that could not perform its check
    (``tested`` False — see _unverified) goes to ``unverified``, NOT to
    ``failed``: it must never be announced or auto-queued as a broken
    subsystem, and it must never be averaged into "all systems nominal"
    either. Every one is WARN-logged here, once per sweep, so the degraded
    path leaves a trail even when nobody reads the summary.
    """
    sweep_start = _now()
    probes_out: dict[str, dict] = {}
    failed: list[str] = []
    unverified: list[str] = []
    sev_failed: dict[str, str] = {}

    for name, fn in PROBES.items():
        r = _run_with_timeout(fn, PER_PROBE_TIMEOUT_S, name=name)
        # Resolve severity: per-result override > subsystem default.
        if r.get("severity") is None:
            r["severity"] = SUBSYSTEM_SEVERITY.get(name, SEVERITY_MED)
        probes_out[name] = r
        if r.get("ok", False):
            continue
        # .get(..., True) so any result built outside _result() (older shape,
        # a test double) is treated as a real, tested failure — fail loud.
        if not r.get("tested", True):
            unverified.append(name)
            _log.warning("self-diagnostic: %s UNVERIFIED — %s (this is NOT a "
                         "pass: the check did not run)", name, r.get("error"))
            continue
        failed.append(name)
        sev_failed[name] = r["severity"]

    run = {
        "ts":           sweep_start,
        "iso":          _iso(sweep_start),
        "duration_ms":  round((_now() - sweep_start) * 1000.0, 1),
        "probes":       probes_out,
        "failed":       failed,
        "severity_failed": sev_failed,
        "unverified":   unverified,
    }
    return run


def _load_history() -> list[dict]:
    if not os.path.exists(_HISTORY_PATH):
        return []
    try:
        with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("runs"), list):
            return data["runs"]
    except Exception as e:
        _log.warning("failed to read %s: %s", _HISTORY_PATH, e)
    return []


def _save_history(history: list[dict]) -> None:
    # Trim from the front so newest stay
    if len(history) > MAX_HISTORY_RUNS:
        history = history[-MAX_HISTORY_RUNS:]
    try:
        os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
        _atomic_write_json(_HISTORY_PATH, history)
    except Exception as e:
        _log.warning("failed to write %s: %s", _HISTORY_PATH, e)


# ─── Auto-queue repair tasks ─────────────────────────────────────────────
_SELF_DIAG_LINE_RE = re.compile(
    r"^\-\s+\[\s+\]\s+\*\*([\d\-]+)\*\*\s+\[self-diag\]\s+\-\s+Fix:\s+(\S+)",
)
# whats_broken reads BOTH the [self-diag] sweep tasks AND the [self-heal]
# pipeline tasks (repeated action failures, VAD stalls, camera errors — written
# at ~2524/2548/2574). The dedup reader above stays self-diag-only (the
# self-heal writers dedup themselves), but the user-facing "what's broken"
# readout must surface every open repair task — before this it silently missed
# every self-heal item (2026-07-06 audit tail).
# 2026-07-07 bug-hunt (MED): capture ENOUGH after "Fix: " to tell distinct
# self-heal tasks apart. The self-heal writers emit "Fix: action 'foo' …",
# "Fix: camera 0 …", "Fix: VAD …" — a bare (\S+) grabbed only the generic word
# ("action"/"camera"), so whats_broken deduped two different failing actions to
# ONE and named it with a bare word. We now also pull the trailing quoted name
# or numeric index when present, so "action 'foo'" and "action 'bar'" (or camera
# 0 and camera 1) stay distinct; self-diag's "Fix: <component> reports …" keeps
# capturing just the component (no quote/number follows).
_ANY_REPAIR_LINE_RE = re.compile(
    r"^\-\s+\[\s+\]\s+\*\*([\d\-]+)\*\*\s+\[(?:self-diag|self-heal)\]\s+\-\s+"
    r"Fix:\s+(\S+(?:\s+'[^']+'|\s+\"[^\"]+\"|\s+\d+)?)",
)


def _open_selfdiag_components() -> set[str]:
    """Return the set of components that already have an OPEN (unchecked)
    self-diag fix task in jarvis_todo.md. Dedupe so we don't pile up the
    same task every 30 minutes."""
    if not os.path.exists(_TODO_PATH):
        return set()
    try:
        open_components: set[str] = set()
        with open(_TODO_PATH, "r", encoding="utf-8") as f:
            for line in f:
                m = _SELF_DIAG_LINE_RE.match(line)
                if m:
                    open_components.add(m.group(2))
        return open_components
    except Exception as e:
        _log.warning("failed to scan jarvis_todo.md for open self-diag tasks: %s", e)
        return set()


def _last_successful_ts(history: list[dict], component: str) -> str | None:
    """Walk the history backwards to find the most recent run where
    ``component`` was OK. Returns the ISO timestamp, or None if it's never
    been seen healthy in our history window."""
    for run in reversed(history):
        probe = (run.get("probes") or {}).get(component, {})
        if probe.get("ok"):
            return run.get("iso") or _iso(run.get("ts", 0.0))
    return None


def _suggested_files_for(component: str) -> str:
    """Hint the auto-repair task with the source files most relevant to
    fixing each subsystem. The upgrade pipeline reads this and gives the
    target files to whatever LLM agent it spawns."""
    suggestions = {
        "webcam":           "skills/face_tracker.py, hud/jarvis_hud.py",
        "microphone":       "bobert_companion.py (audio capture loop), skills/wake_listener.py",
        "tts":              "core/tts.py, bobert_companion.py (TTS path)",
        "stt":              "bobert_companion.py (_ensure_whisper / Whisper config)",
        "claude_api":       "bobert_companion.py (CLAUDE_MODEL, _call_llm), .env",
        "internet":         "(network — likely not a code fix; check connection)",
        "hud_subprocesses": "hud/jarvis_hud.py, hud/jarvis_reticle.py, hud/workshop_hud.py, tray.py",
        "state_files":      "(check which file failed; restore from backups/)",
        "bambu":            "skills/bambu_monitor.py, skills/bambu_setup.py, .env (BAMBU_*)",
        "media_playback":   "skills/apple_music_intel.py, bobert_companion.py (play_music)",
        "skill_imports":    "(check which skill failed; syntax error in skills/<name>.py)",
        "gpu":              "bobert_companion.py (WHISPER_DEVICE, _resolve_whisper_device)",
        "disk":             "(not a code fix; clean up data/ or backups/)",
        "ram":              "(not a code fix; identify the runaway process)",
        "optional_skills":  "(when research-4a/4c land, point here)",
    }
    return suggestions.get(component, "(no suggestion)")


def _queue_repair_task(component: str, run: dict, history: list[dict]) -> bool:
    """Append a self-healing task to jarvis_todo.md for ``component``.
    Returns True if appended, False if already-queued (dedupe) or write
    failed. Only MED+ severity gets queued — LOW failures stay in history
    only."""
    probe = (run.get("probes") or {}).get(component, {})
    # Belt-and-braces (2026-08-20): run_diagnostic only walks run["failed"],
    # which _run_all_probes already keeps unverified probes out of — but a
    # future caller must not be able to file a repair task for a check that
    # never ran. "I could not look" is not a defect report.
    if not probe.get("tested", True):
        return False
    sev = probe.get("severity") or SUBSYSTEM_SEVERITY.get(component, SEVERITY_MED)
    if sev == SEVERITY_LOW:
        return False

    open_components = _open_selfdiag_components()
    if component in open_components:
        return False

    err          = probe.get("error") or "(no error message)"
    last_ok      = _last_successful_ts(history, component) or "never (within history window)"
    suggestions  = _suggested_files_for(component)
    today        = _today_iso_date()
    latency      = probe.get("latency_ms", 0)
    details      = probe.get("details") or {}
    details_blob = ""
    if details:
        try:
            details_blob = json.dumps(details, default=str)[:240]
        except Exception:
            details_blob = "(details unavailable)"

    line = (
        f"- [ ] **{today}** [self-diag] - Fix: {component} reports {err}. "
        f"Last successful: {last_ok}. Severity: {sev}. "
        f"Probe latency: {latency} ms. "
        f"Diagnostic data in data/self_diagnostic.json. "
        f"Investigate {suggestions} and either repair the component or "
        f"document why it can't be auto-fixed (e.g. hardware unplugged). "
        f"Probe details: {details_blob}"
    )

    try:
        with open(_TODO_PATH, "a", encoding="utf-8") as f:
            # Make sure we land on a fresh line — the file may or may not
            # end with a newline depending on the previous editor.
            f.write("\n" + line + "\n")
        return True
    except Exception as e:
        _log.warning("failed to append self-diag task for %s: %s", component, e)
        return False


# ─── Voice / announcement ────────────────────────────────────────────────
def _proactive_announce(message: str, *, mood: Optional[str] = None) -> None:
    """Route a HIGH-severity alert through bobert_companion's proactive
    announcer. Silent if the parent module isn't loaded yet (early boot).

    `mood` (optional) opts into the voice_mood layer. _announce_failures
    passes mood='concerned_soft' for HIGH-severity probe failures so the
    spoken alert lands softer + slower rather than alarmed."""
    bc = _bc()
    if bc is None:
        return
    announcer = getattr(bc, "proactive_announce", None)
    if callable(announcer):
        try:
            if mood:
                announcer(message, source="self_diagnostic", mood=mood)
            else:
                announcer(message, source="self_diagnostic")
        except TypeError:
            # Older bobert_companion build without the mood= kwarg — fall
            # back to the signature it does support so the alert still fires.
            try:
                announcer(message, source="self_diagnostic")
            except Exception as e:
                _log.warning("proactive_announce failed: %s", e)
        except Exception as e:
            _log.warning("proactive_announce failed: %s", e)


def _push_phone(message: str, priority: str = "high") -> None:
    """Best-effort push to phone via phone_bridge. Silent if the skill
    isn't loaded or no backend is configured."""
    mod = sys.modules.get("skill_phone_bridge")
    if mod is None:
        return
    fn = getattr(mod, "push_to_phone", None)
    if not callable(fn):
        return
    try:
        # Diagnostic pages are urgent system alerts, not user-composed
        # drafts — bypass the pre-send confirmation gate so a critical
        # warning still reaches the phone even when the user isn't at
        # the microphone to confirm.
        fn(message, priority=priority, source="self_diagnostic", confirm=False)
    except Exception as e:
        _log.warning("push_to_phone failed: %s", e)


def _announce_failures(run: dict) -> None:
    """Speak about HIGH-severity failures; push to phone if configured.

    Re-announce dedup: a HIGH probe that keeps failing for the *same*
    reason (e.g. claude_api down for a known, dated outage) is announced
    once, not every 30-min sweep. We track a per-component state signature
    (component + its error string) in _announced_failure_state and only
    speak about components whose signature changed since we last announced
    them. A component that recovers and later fails again with a different
    error re-announces, as does the first occurrence of any failure.
    """
    high = [c for c, s in (run.get("severity_failed") or {}).items()
            if s == SEVERITY_HIGH]
    if not high:
        return

    # Per-component dedup: announce only components whose failure signature
    # changed since the last time we spoke about them. The signature folds
    # in the error text so a *different* failure on the same component still
    # surfaces, while a persistent identical failure stays quiet.
    probes = run.get("probes") or {}
    changed: list[str] = []
    for c in high:
        sig = str((probes.get(c) or {}).get("error") or "failed")
        if _announced_failure_state.get(c) != sig:
            changed.append(c)
        _announced_failure_state[c] = sig

    if not changed:
        # Every HIGH failure this sweep is a known, already-announced
        # condition with an unchanged cause — don't re-speak / re-push.
        return

    # Phrase the announcement naturally. Each entry carries its own article
    # ("the Claude API", "a state file", "" for bare nouns) so we never
    # double up on "the" — the template no longer prepends one.
    pretty = {
        "microphone":       "the microphone",
        "stt":              "speech recognition",
        "claude_api":       "the Claude API",
        "internet":         "internet connectivity",
        "state_files":      "a state file",
        "skill_imports":    "one or more skills",
        "disk":             "the disk",
        "ram":              "system memory",
    }
    names = [pretty.get(c, c.replace("_", " ")) for c in changed]
    if len(names) == 1:
        msg = f"Sir, {names[0]} appears to be down. I'll queue a fix."
    else:
        first = ", ".join(names[:-1])
        msg = (f"Sir, multiple core systems are reporting failures: "
               f"{first}, and {names[-1]}. I'll queue fixes.")

    # Mark before speaking so the voice_mood layer sees the flag on the
    # very first utterance (the announcement itself lands in concerned_soft).
    _mark_recent_problem()
    if HIGH_SEVERITY_SPEAK:
        _proactive_announce(msg, mood="concerned_soft")
    if HIGH_SEVERITY_PHONE:
        _push_phone(msg, priority="urgent")


# ─── Self-healing auto-queue ─────────────────────────────────────────────
def _load_autoqueue_state() -> dict:
    """Per-signature last-queued timestamps used to dedup the auto-queue."""
    if not os.path.exists(_AUTOQUEUE_PATH):
        return {}
    try:
        with open(_AUTOQUEUE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        _log.warning("autoqueue state read failed: %s", e)
    return {}


def _save_autoqueue_state(state: dict) -> None:
    try:
        _atomic_write_json(_AUTOQUEUE_PATH, state)
    except Exception as e:
        _log.warning("autoqueue state write failed: %s", e)


def _session_log_tail(n_lines: int = _AUTOQUEUE_LOG_TAIL_LINES) -> list[str]:
    """Return the last ``n_lines`` lines of the live session log, or [] when
    logging is off / the file isn't readable. Reads atomically — open + read
    + close — so a concurrent stdout writer can't tear our read."""
    bc = _bc()
    path_fn = getattr(bc, "get_session_log_path", None) if bc else None
    log_path = None
    if callable(path_fn):
        try:
            log_path = path_fn()
        except Exception:
            log_path = None
    if not (log_path and isinstance(log_path, str) and os.path.exists(log_path)):
        return []
    # Read the tail with a bounded byte budget so we never slurp a multi-MB
    # log into memory. n_lines * ~400 bytes/line is a safe upper bound.
    try:
        size = os.path.getsize(log_path)
        budget = max(8192, n_lines * 400)
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            if size > budget:
                f.seek(size - budget)
                # Drop the first (probably partial) line.
                f.readline()
            tail = f.readlines()[-n_lines:]
        # Strip trailing newlines; preserve indentation.
        return [ln.rstrip("\n") for ln in tail]
    except Exception as e:
        _log.warning("session log tail read failed: %s", e)
        return []


def _traceback_excerpt(tb_text: str,
                       max_lines: int = _AUTOQUEUE_TRACEBACK_LINES) -> str:
    """Return the last ``max_lines`` non-blank lines of a traceback so the
    fix request shows where the exception fired without dumping the full
    stack. The bottom of the traceback is the most actionable part."""
    if not tb_text:
        return ""
    lines = [ln for ln in tb_text.splitlines() if ln.strip()]
    return "\n".join(lines[-max_lines:])


def _suggested_files_for_action(action_name: str) -> str:
    """Best-effort hint at which source files own a given action so the
    autoqueue task points Claude Code at a useful starting point. Walks the
    live ACTIONS dict via bobert_companion to find the owning skill module."""
    bc = _bc()
    if bc is None:
        return "bobert_companion.py (action dispatcher)"
    actions = getattr(bc, "ACTIONS", None)
    if not isinstance(actions, dict):
        return "bobert_companion.py (action dispatcher)"
    fn = actions.get(action_name)
    mod_name = getattr(fn, "__module__", None) if fn is not None else None
    if not mod_name:
        return "bobert_companion.py (action dispatcher)"
    # Translate the runtime module name back to its file path. Skill modules
    # are registered as ``skill_<stem>`` (see load_skills); the source file
    # lives under skills/<stem>.py.
    if mod_name == "bobert_companion":
        return "bobert_companion.py"
    if mod_name.startswith("skill_"):
        return f"skills/{mod_name[len('skill_'):]}.py"
    if mod_name.startswith("core."):
        return f"{mod_name.replace('.', '/')}.py"
    return f"{mod_name} (module)"


def _collect_action_error_groups() -> list[dict]:
    """Group recent action errors by (action, exc_class) and return one
    entry per group whose count crosses the auto-queue threshold.

    Each entry: {signature, count, action, exc_class, exc_msg, traceback,
                 first_ts, last_ts}. Signature is stable across sweeps so
    the dedup cooldown holds."""
    bc = _bc()
    getter = getattr(bc, "get_recent_action_errors", None) if bc else None
    if not callable(getter):
        return []
    try:
        errors = getter(_AUTOQUEUE_ERROR_WINDOW_S) or []
    except Exception as e:
        _log.warning("get_recent_action_errors raised: %s", e)
        return []
    groups: dict[str, dict] = {}
    for e in errors:
        try:
            action = str(e.get("action") or "")
            klass  = str(e.get("exc_class") or "Exception")
            sig    = f"action_error::{action}::{klass}"
            g = groups.get(sig)
            if g is None:
                groups[sig] = {
                    "signature":  sig,
                    "count":      1,
                    "action":     action,
                    "exc_class":  klass,
                    "exc_msg":    e.get("exc_msg") or "",
                    "traceback":  e.get("traceback") or "",
                    "first_ts":   float(e.get("ts") or 0.0),
                    "last_ts":    float(e.get("ts") or 0.0),
                }
            else:
                g["count"] += 1
                ts = float(e.get("ts") or 0.0)
                if ts > g["last_ts"]:
                    g["last_ts"] = ts
                    g["exc_msg"]   = e.get("exc_msg") or g["exc_msg"]
                    g["traceback"] = e.get("traceback") or g["traceback"]
                if ts < g["first_ts"] or g["first_ts"] == 0.0:
                    g["first_ts"] = ts
        except Exception:
            continue
    return [g for g in groups.values()
            if g["count"] >= _AUTOQUEUE_ERROR_GROUP_COUNT]


def _collect_vad_stall_signal() -> dict | None:
    """Return a VAD-stall signal dict, or None when no stall is detected.

    A stall is: the input capture loop is actively polling (last_poll_ts
    fresh, within VAD_STALL window) but no VAD trip has fired for more
    than _AUTOQUEUE_VAD_STALL_S. Only fires while JARVIS is awake — when
    sleeping there's no expectation of VAD activity."""
    try:
        from core import audio_processor as _ap
    except Exception:
        return None
    bc = _bc()
    if bc is None:
        return None
    # Only consider stalls while JARVIS is awake — sleep_mode = True means
    # we explicitly don't want to capture, so silence is fine.
    try:
        sleep_flag = getattr(bc, "_sleep_mode", [True])
        sleeping = bool(sleep_flag[0]) if sleep_flag else True
    except Exception:
        sleeping = True
    if sleeping:
        return None
    try:
        st = _ap.get_vad_state()
    except Exception:
        return None
    now = _now()
    last_poll  = float(st.get("last_vad_poll_ts")   or 0.0)
    last_trip  = float(st.get("last_vad_active_ts") or 0.0)
    session_start = float(st.get("vad_session_start") or 0.0)
    # Need a real session worth of polling. If poll has never happened OR is
    # itself stale, that's a separate problem covered by the microphone
    # probe — don't double-queue.
    poll_age = (now - last_poll) if last_poll else float("inf")
    if poll_age > _PASSIVE_POLL_FRESH_S:   # shared with _passive_mic_liveness
        return None
    # Also need enough time to have elapsed since first poll — otherwise
    # we'd false-positive on the very first capture session.
    if session_start > 0.0 and (now - session_start) < _AUTOQUEUE_VAD_STALL_S:
        return None
    trip_age = (now - last_trip) if last_trip else float("inf")
    if trip_age < _AUTOQUEUE_VAD_STALL_S:
        return None
    return {
        "signature":            "vad_stall",
        "seconds_since_active": round(trip_age, 1) if trip_age != float("inf") else None,
        "seconds_since_poll":   round(poll_age, 1),
        "total_vad_trips":      int(st.get("total_vad_trips") or 0),
    }


def _collect_face_failure_signals() -> list[dict]:
    """One entry per camera whose face_tracker read-failure spike crosses the
    auto-queue threshold. Delegates to skills/face_tracker.get_read_failure_
    spike_signals so the threshold + spike heuristic stays in one place."""
    mod = sys.modules.get("skill_face_tracker")
    if mod is None:
        return []
    fn = getattr(mod, "get_read_failure_spike_signals", None)
    if not callable(fn):
        return []
    try:
        raw = fn(threshold=_AUTOQUEUE_FACE_FAIL_THRESH) or []
    except Exception as e:
        _log.warning("face_tracker read-failure probe raised: %s", e)
        return []
    out: list[dict] = []
    for sig in raw:
        try:
            out.append({
                "signature":              f"face_read_fail::cam{sig['cam_index']}",
                "cam_index":              sig.get("cam_index"),
                "consecutive_fails":      int(sig.get("consecutive_fails") or 0),
                "max_consecutive_fails":  int(sig.get("max_consecutive_fails") or 0),
                "last_error":             sig.get("last_error"),
                "seconds_since_last_ok":  sig.get("seconds_since_last_ok"),
            })
        except Exception:
            continue
    return out


def _format_action_error_task(group: dict, log_tail: list[str]) -> str:
    """Render a single action-error group as a structured jarvis_todo.md
    task line. Keeps the existing format (- [ ] **YYYY-MM-DD** [tag] - ...)
    so the watcher and existing scanners don't break, then folds the rich
    fix-request payload into a fenced details block via embedded newlines.
    """
    today        = _today_iso_date()
    action       = group["action"]
    klass        = group["exc_class"]
    count        = group["count"]
    msg          = (group.get("exc_msg") or "").strip().replace("\n", " ")[:160]
    files_hint   = _suggested_files_for_action(action)
    tb_excerpt   = _traceback_excerpt(group.get("traceback") or "")
    log_block    = "\n".join(log_tail) if log_tail else "(session log unavailable)"
    repro        = (f"call action {action!r} from the dispatcher and observe "
                    f"{klass}; the failure has repeated {count}x in the last "
                    f"{_AUTOQUEUE_ERROR_WINDOW_S/60:.0f} min")
    # All structured payload is embedded inside the single task line via
    # literal '\n' so the existing one-line-per-task watcher still sees it
    # as one entry, but Claude Code's reader gets the full context.
    payload = (
        f"\n  - file: {files_hint}"
        f"\n  - traceback (last {_AUTOQUEUE_TRACEBACK_LINES} lines):"
        f"\n    ```\n{tb_excerpt}\n    ```"
        f"\n  - last {len(log_tail)} session log lines:"
        f"\n    ```\n{log_block}\n    ```"
        f"\n  - one-line repro: {repro}"
    )
    return (
        f"- [ ] **{today}** [self-heal] - Fix: action {action!r} keeps raising "
        f"{klass} ({count}x in {_AUTOQUEUE_ERROR_WINDOW_S/60:.0f} min). "
        f"Last error: {msg}.{payload}"
    )


def _format_vad_stall_task(signal: dict, log_tail: list[str]) -> str:
    today = _today_iso_date()
    secs = signal.get("seconds_since_active")
    secs_str = f"{secs:.0f}s" if isinstance(secs, (int, float)) else "unknown"
    log_block = "\n".join(log_tail) if log_tail else "(session log unavailable)"
    repro = ("with JARVIS awake, wait for record_speech to call note_vad_poll "
             "for a full session without ever calling note_vad_active — confirm "
             "core.audio_processor.get_vad_state()['last_vad_active_ts'] stays "
             "stale while last_vad_poll_ts updates")
    payload = (
        f"\n  - file: bobert_companion.py (record_speech VAD loop), "
        f"core/audio_processor.py (VAD instrumentation)"
        f"\n  - traceback: (no Python exception — this is a behavioral stall)"
        f"\n  - last {len(log_tail)} session log lines:"
        f"\n    ```\n{log_block}\n    ```"
        f"\n  - one-line repro: {repro}"
    )
    return (
        f"- [ ] **{today}** [self-heal] - Fix: VAD has not tripped in {secs_str} "
        f"while JARVIS is awake and the capture loop is still polling — likely a "
        f"silent mic, AEC over-ducking, or a noise gate threshold drift.{payload}"
    )


def _format_face_fail_task(signal: dict, log_tail: list[str]) -> str:
    today = _today_iso_date()
    idx = signal.get("cam_index")
    consec = signal.get("consecutive_fails")
    peak   = signal.get("max_consecutive_fails")
    err    = (signal.get("last_error") or "(no detail)").strip()[:160]
    log_block = "\n".join(log_tail) if log_tail else "(session log unavailable)"
    repro = (f"with camera index {idx} attached, watch "
             f"bobert_companion.get_camera_failure_summary()[{idx}]"
             f"['consecutive_fails'] climb past "
             f"{_AUTOQUEUE_FACE_FAIL_THRESH} between cap.read() returns of False")
    payload = (
        f"\n  - file: bobert_companion.py (_face_tracking_thread "
        f"cap.read() loop), skills/face_tracker.py"
        f"\n  - traceback: (no Python exception — cv2.VideoCapture.read returned False)"
        f"\n  - last {len(log_tail)} session log lines:"
        f"\n    ```\n{log_block}\n    ```"
        f"\n  - one-line repro: {repro}"
    )
    return (
        f"- [ ] **{today}** [self-heal] - Fix: camera {idx} hit a face_tracker "
        f"read-failure spike (consecutive={consec}, peak={peak}). "
        f"Last error: {err}.{payload}"
    )


def _append_autoqueue_line(line: str) -> bool:
    """Append a single self-heal task line to jarvis_todo.md. Preserves the
    existing trailing-newline contract so the watcher's line-by-line scan
    keeps working."""
    try:
        with open(_TODO_PATH, "a", encoding="utf-8") as f:
            f.write("\n" + line + "\n")
        return True
    except Exception as e:
        _log.warning("autoqueue append failed: %s", e)
        return False


def _run_autoqueue_pass() -> list[str]:
    """Collect every signal, dedup against the persisted cooldown state, and
    append a structured fix request to jarvis_todo.md for each one that
    survives the dedup. Returns the list of signatures appended this pass.

    Called from run_diagnostic AFTER the probe sweep so failures detected by
    the regular probes (which already auto-queue via _queue_repair_task) get
    surfaced first and don't double-fire here."""
    appended: list[str] = []
    try:
        now = _now()
        state = _load_autoqueue_state()
        log_tail = _session_log_tail()
        cutoff = now - _AUTOQUEUE_COOLDOWN_S

        # 1. Caught action failures (≥3 in 1h, same action+exc_class).
        for group in _collect_action_error_groups():
            sig = group["signature"]
            last = float(state.get(sig, {}).get("last_queued_ts", 0.0))
            if last >= cutoff:
                continue
            line = _format_action_error_task(group, log_tail)
            if _append_autoqueue_line(line):
                state[sig] = {"last_queued_ts": now, "kind": "action_error",
                              "action": group["action"],
                              "exc_class": group["exc_class"],
                              "count_at_queue": group["count"]}
                appended.append(sig)

        # 2. VAD stall (>60s without trip while awake).
        vad = _collect_vad_stall_signal()
        if vad is not None:
            sig = vad["signature"]
            last = float(state.get(sig, {}).get("last_queued_ts", 0.0))
            if last < cutoff:
                line = _format_vad_stall_task(vad, log_tail)
                if _append_autoqueue_line(line):
                    state[sig] = {"last_queued_ts": now, "kind": "vad_stall",
                                  "seconds_since_active": vad.get("seconds_since_active")}
                    appended.append(sig)

        # 3. face_tracker read-failure spikes (consecutive >=5).
        for fsig in _collect_face_failure_signals():
            sig = fsig["signature"]
            last = float(state.get(sig, {}).get("last_queued_ts", 0.0))
            if last >= cutoff:
                continue
            line = _format_face_fail_task(fsig, log_tail)
            if _append_autoqueue_line(line):
                state[sig] = {"last_queued_ts": now, "kind": "face_read_fail",
                              "cam_index": fsig.get("cam_index"),
                              "consecutive_fails": fsig.get("consecutive_fails")}
                appended.append(sig)

        if appended:
            _save_autoqueue_state(state)
            _log.warning("self-heal autoqueue appended %d signature(s): %s",
                         len(appended), ", ".join(appended))
    except Exception as e:
        _log.exception("self-heal autoqueue raised: %s", e)
    return appended


# ─── Run + summary ───────────────────────────────────────────────────────
def run_diagnostic(_: str = "") -> str:
    """Fire a full sweep, persist results, queue repairs, return a
    one-line summary in JARVIS voice."""
    # Single-flight: a sweep in progress means the next caller waits for
    # the result rather than starting a second concurrent sweep.
    if not _run_lock.acquire(blocking=False):
        return "A diagnostic sweep is already in flight, sir — give me a moment."

    try:
        _state["last_run_started_at"] = _now()
        run = _run_all_probes()
        _state["last_run"] = run
        _state["runs_completed"] += 1

        history = _load_history()
        history.append(run)
        _save_history(history)

        # Queue repair tasks for MED+ failures
        queued: list[str] = []
        for comp in run["failed"]:
            sev = run["severity_failed"].get(comp, SEVERITY_MED)
            if sev != SEVERITY_LOW:
                if _queue_repair_task(comp, run, history):
                    queued.append(comp)

        # Self-healing auto-queue: caught action failures, VAD stalls, and
        # face_tracker read-failure spikes that the probe sweep doesn't
        # already cover. Runs AFTER the probe-based queue so probe-level
        # failures get the canonical task first and the auto-queue dedup
        # doesn't double-queue the same component.
        try:
            autoqueued = _run_autoqueue_pass()
        except Exception:
            _log.exception("autoqueue pass raised")
            autoqueued = []
        if autoqueued:
            queued.extend(autoqueued)

        # Speak about HIGH-severity failures
        _announce_failures(run)

        # Log all failures at WARN level — EXCEPT a deliberately LOW-severity
        # Claude API outage, which is the NORMAL credits-optional baseline
        # (JARVIS runs fully on the local model). Log that calmly at INFO so it
        # never reads as a problem or trips the exception-burst anomaly logic.
        for comp in run["failed"]:
            probe = run["probes"][comp]
            sev = probe.get("severity")
            if comp == "claude_api" and sev == SEVERITY_LOW:
                _log.info("self-diagnostic: Claude API enhancement unavailable "
                          "(%s) — running on the local model; this is fine.",
                          str(probe.get("error"))[:80])
                continue
            _log.warning("self-diagnostic: %s FAILED (severity=%s) — %s",
                         comp, sev, probe.get("error"))

        return _summarise(run, queued)
    finally:
        _run_lock.release()


def _short_cause_for(run: dict | None, comp: str) -> str:
    """The few marker-free words explaining why ``comp`` was not measured, or
    "" when the run carries none (older history entries, foreign results)."""
    try:
        det = (((run or {}).get("probes") or {}).get(comp) or {}).get("details") or {}
        cause = str(det.get("unverified_short_cause") or "").strip()
    except Exception:
        return ""
    if not cause:
        return ""
    low = cause.lower()
    if any(m.lower() in low for m in FAILURE_MARKERS):
        # Belt-and-braces for the wording rule at the top of this module: a
        # cause carrying a marker would take the WHOLE honest sentence off the
        # verbatim speak path. Drop the four words rather than lose the
        # sentence; the full reason is still in the probe result and the log.
        _log.warning("self-diagnostic: dropping short cause for %s — it "
                     "contains a failure marker: %r", comp, cause)
        return ""
    return cause


def _unverified_phrase(unverified: list[str], run: dict | None = None) -> str:
    """Shared wording for the third outcome so _summarise, diagnostic_status
    and diagnostic_history can never drift apart (stale-duplicate rule).

    MARKER-FREE BY CONTRACT — see the spoken-surface wording rule at the top of
    this module. "couldn't run" (the wording that shipped this morning) matches
    core.failure_markers.FAILURE_MARKERS, which drops the whole sentence off
    bobert_companion's verbatim speak path and burns an extra LLM round-trip
    re-wording it. "did not run" says exactly the same thing and is spoken.

    ``run`` is optional so history entries (which carry no probe details) still
    render; when given, each component also names WHY it was not measured.
    """
    if not unverified:
        return ""
    parts = []
    for c in unverified:
        pretty = c.replace("_", " ")
        cause = _short_cause_for(run, c)
        parts.append(f"{pretty} ({cause})" if cause else pretty)
    names = ", ".join(parts)
    n = len(unverified)
    return (f"{n} check{'s' if n != 1 else ''} did not run, so "
            f"{'they are' if n != 1 else 'it is'} UNVERIFIED — {names}")


def _summarise(run: dict, queued: list[str]) -> str:
    failed = run.get("failed") or []
    unverified = run.get("unverified") or []
    unv = _unverified_phrase(unverified, run)
    duration_s = (run.get("duration_ms") or 0) / 1000.0
    if not failed:
        if not unverified:
            return f"All systems nominal, sir. ({duration_s:.1f}s sweep, {len(PROBES)} probes.)"
        # Never "nominal" while something was not checked — that is the exact
        # sentence that covered for a deaf JARVIS on 2026-08-20.
        #
        # "I am not able to", NOT "I can't": this is THE sentence the release
        # was built around, and "can't" is a FAILURE_MARKER, so the version
        # that shipped this morning was dropped by _speak_verbatim_results and
        # re-worded by an LLM follow-up instead of being read to him. See the
        # spoken-surface wording rule at the top of this module.
        return (f"Sir, nothing is reporting a failure, but I am not able to "
                f"call the system nominal: {unv}. ({duration_s:.1f}s sweep, "
                f"{len(PROBES)} probes.)")

    counts: dict[str, int] = {}
    for c in failed:
        sev = run["severity_failed"].get(c, SEVERITY_MED)
        counts[sev] = counts.get(sev, 0) + 1

    pretty_names = [c.replace("_", " ") for c in failed]
    if len(pretty_names) == 1:
        body = f"one issue — {pretty_names[0]}"
    elif len(pretty_names) <= 4:
        body = f"{len(pretty_names)} issues: {', '.join(pretty_names)}"
    else:
        # Avoid reading off a paragraph of subsystem names.
        body = (f"{len(pretty_names)} issues: {', '.join(pretty_names[:3])}"
                f", and {len(pretty_names) - 3} more")

    qstr = ""
    if queued:
        qstr = (f" {len(queued)} repair task{'s' if len(queued) != 1 else ''} "
                f"queued in jarvis_todo.md.")

    sev_breakdown = " / ".join(f"{n} {sev.lower()}" for sev, n in counts.items())
    ustr = f" Separately, {unv}." if unv else ""
    # NB the failure branch is marker-free too ("Sir, one issue — microphone.
    # Severity breakdown: 1 high.") and is meant to be: it is SPOKEN VERBATIM,
    # which is how the owner hears the subsystem names and the severity counts
    # rather than an LLM's paraphrase of them.
    return (f"Sir, {body}. Severity breakdown: {sev_breakdown}.{qstr}{ustr} "
            f"({duration_s:.1f}s sweep.)")


def diagnostic_status(_: str = "") -> str:
    """Terse summary of the most recent sweep."""
    run = _state.get("last_run")
    if not run:
        return ("No diagnostic has run yet, sir — say 'run diagnostic' or "
                "give me a moment to do the boot sweep.")
    age_s = _now() - run["ts"]
    if age_s < 90:
        age = f"{age_s:.0f} seconds ago"
    elif age_s < 3600:
        age = f"{age_s / 60.0:.0f} minutes ago"
    else:
        age = f"{age_s / 3600.0:.1f} hours ago"

    failed = run.get("failed") or []
    # TRANSIENT unverified entries are re-measured HERE, at read time, through
    # the same device-free path the probe uses — so a sweep that happened to
    # land mid-turn does not make "diagnostic status" cry UNVERIFIED for the
    # next half hour with nothing wrong. Positive evidence only; anything else
    # stays unverified, and the recorded run is not rewritten.
    unverified, cleared = _live_recheck_unverified(run, run.get("unverified") or [])
    cleared_note = ""
    if cleared:
        cnames = ", ".join(c.replace("_", " ") for c in cleared)
        label = "check" if len(cleared) == 1 else "checks"
        cleared_note = (f" The {cnames} {label} had no data at the time — I "
                        f"re-measured just now and the signal is live.")
    unv = _unverified_phrase(unverified, run)
    if not failed:
        if not unverified:
            return (f"All systems nominal as of {age}, sir. "
                    f"{len(PROBES)} probes, "
                    f"{(run['duration_ms'] / 1000):.1f}s sweep.{cleared_note}")
        return (f"As of {age}, sir, nothing is reporting a failure — but {unv}. "
                f"{len(PROBES)} probes, "
                f"{(run['duration_ms'] / 1000):.1f}s sweep.{cleared_note}")
    names = ", ".join(c.replace("_", " ") for c in failed)
    ustr = f" Separately, {unv}." if unv else ""
    return (f"Last sweep was {age}, sir — {len(failed)} subsystem(s) "
            f"reporting issues: {names}.{ustr}{cleared_note}")


def whats_broken(_: str = "") -> str:
    """Read back any OPEN repair tasks from jarvis_todo.md — BOTH the
    [self-diag] sweep tasks and the [self-heal] pipeline tasks."""
    if not os.path.exists(_TODO_PATH):
        return "I can't find jarvis_todo.md, sir."
    try:
        components: list[tuple[str, str]] = []
        with open(_TODO_PATH, "r", encoding="utf-8") as f:
            for line in f:
                m = _ANY_REPAIR_LINE_RE.match(line)
                if m:
                    components.append((m.group(1), m.group(2)))
    except Exception as e:
        return f"I couldn't scan jarvis_todo.md, sir: {type(e).__name__}: {e}"

    if not components:
        return "Nothing flagged for repair, sir. The queue is clean."

    # Deduplicate (multiple opens for the same component shouldn't happen,
    # but if they do, just list each component once).
    seen: set[str] = set()
    uniq = []
    for date, comp in components:
        if comp in seen:
            continue
        seen.add(comp)
        uniq.append((date, comp))

    if len(uniq) == 1:
        date, comp = uniq[0]
        return f"One open repair task, sir — {comp.replace('_', ' ')} flagged on {date}."
    names = ", ".join(c.replace("_", " ") for _, c in uniq)
    return f"{len(uniq)} open repair tasks, sir: {names}."


def diagnostic_history(arg: str = "") -> str:
    """List the last N runs (default 5, max 25)."""
    n = 5
    if arg.strip():
        try:
            n = max(1, min(25, int(arg.strip().split()[0])))
        except Exception:
            pass

    history = _load_history()
    if not history:
        return "No diagnostic history yet, sir."

    recent = history[-n:]
    lines = []
    for run in recent:
        failed = run.get("failed") or []
        unverified = run.get("unverified") or []
        iso = run.get("iso") or _iso(run.get("ts", 0))
        if failed:
            line = f"{iso}: {len(failed)} issue(s) — {', '.join(failed)}"
            if unverified:
                line += f" (+{len(unverified)} unverified)"
            lines.append(line)
        elif unverified:
            lines.append(f"{iso}: no failures, {len(unverified)} unverified "
                         f"— {', '.join(unverified)}")
        else:
            lines.append(f"{iso}: all nominal")
    return f"Last {len(recent)} sweeps, sir: " + " | ".join(lines)


def last_diagnostic_run(_: str = "") -> str:
    """Raw JSON of the last run (for console / HUD debugging)."""
    run = _state.get("last_run")
    if not run:
        return "{}"
    try:
        return json.dumps(run, default=str, indent=2)
    except Exception as e:
        return f"(couldn't serialise last run: {e})"


# ─── Scheduling ──────────────────────────────────────────────────────────
def _schedule_recurring_sweep() -> bool:
    """Install an APScheduler interval job that fires every
    DEFAULT_INTERVAL_MINUTES. Falls back to a thread-based timer if
    APScheduler isn't installed."""
    try:
        from core import scheduler as sched  # type: ignore
        if not sched.is_available():
            return False
        try:
            sched.schedule_interval(
                action="run_diagnostic",
                arg="",
                minutes=DEFAULT_INTERVAL_MINUTES,
                job_id="self_diagnostic_interval",
            )
            print(f"  [self-diag] interval sweep scheduled "
                  f"(every {DEFAULT_INTERVAL_MINUTES} min via APScheduler)")
            return True
        except Exception as e:
            # bootstrap may not have run yet — when the scheduler comes
            # up later it will not include this job. We'll fall back to
            # the timer instead.
            _log.info("scheduler.schedule_interval skipped (%s) — falling back to timer", e)
            return False
    except Exception:
        return False


def _timer_based_sweep_loop():
    """Fallback when APScheduler isn't available: a daemon thread that
    sleeps DEFAULT_INTERVAL_MINUTES between sweeps. We avoid APScheduler's
    persistence here on purpose — if it WAS available, the function above
    would have wired the persistent path."""
    # Boot sweep first, after a short delay so the rest of JARVIS finishes
    # loading.
    time.sleep(ON_BOOT_DELAY_SECONDS)
    try:
        run_diagnostic("")
    except Exception:
        _log.exception("boot self-diagnostic raised")

    interval_s = DEFAULT_INTERVAL_MINUTES * 60
    while True:
        time.sleep(interval_s)
        try:
            run_diagnostic("")
        except Exception:
            _log.exception("interval self-diagnostic raised")


def _spawn_timer_thread() -> None:
    t = threading.Thread(target=_timer_based_sweep_loop,
                         name="self-diagnostic", daemon=True)
    t.start()
    print(f"  [self-diag] interval sweep scheduled "
          f"(every {DEFAULT_INTERVAL_MINUTES} min via thread timer; "
          f"first sweep in {ON_BOOT_DELAY_SECONDS}s)")


# ─── register() ──────────────────────────────────────────────────────────
def register(actions: dict) -> None:
    actions["run_diagnostic"]      = run_diagnostic   # INTENTIONAL_WRAP: this skill IS the full diagnostic; intentionally overrides the core bridge handler
    actions["system_check"]        = run_diagnostic
    actions["are_you_ok"]          = run_diagnostic
    actions["self_diagnostic"]     = run_diagnostic
    actions["diagnostic_status"]   = diagnostic_status   # INTENTIONAL_WRAP: bobert_companion re-asserts the diagnostic_daemons version after skills load (~13435), by design
    actions["whats_broken"]        = whats_broken
    actions["what_is_broken"]      = whats_broken
    actions["diagnostic_history"]  = diagnostic_history
    actions["last_diagnostic_run"] = last_diagnostic_run

    # Schedule recurring sweeps — APScheduler first, thread-timer
    # fallback. Either way the boot sweep runs after ON_BOOT_DELAY_SECONDS.
    scheduled = _schedule_recurring_sweep()
    if not scheduled:
        _spawn_timer_thread()
    else:
        # APScheduler took the recurring slot but won't fire the boot
        # sweep — that's our job. Run it in a one-shot daemon thread so
        # we don't block register().
        def _boot_sweep():
            time.sleep(ON_BOOT_DELAY_SECONDS)
            try:
                run_diagnostic("")
            except Exception:
                _log.exception("boot self-diagnostic raised")
        threading.Thread(target=_boot_sweep, name="self-diag-boot", daemon=True).start()
        print(f"  [self-diag] boot sweep queued (fires in {ON_BOOT_DELAY_SECONDS}s)")
