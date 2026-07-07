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
    2.  microphone        — sounddevice can enumerate input devices and the
                            preferred input registers RMS > floor.
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
   11.  skill_imports     — every .py in skills/ imports without raising.
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
            details: dict | None = None, severity: str | None = None) -> dict:
    """Canonical probe-result shape. Probes return this so the aggregator
    doesn't have to special-case different keys."""
    return {
        "ok":         bool(ok),
        "latency_ms": round(float(latency_ms), 1),
        "error":      None if ok else (error or "unknown error"),
        "details":    details or {},
        "severity":   severity,        # may be None — aggregator fills default
    }


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


def _attempt_camera_wake(idx: int, timeout_s: float = 2.5) -> tuple[bool, str]:
    """Try a soft wake of camera ``idx``: release + brief sleep + reopen +
    read with warmup. Runs inside its own thread with a hard wall-clock
    timeout so a wedged DirectShow open can't block the diagnostic sweep.

    Returns (success, note). Uses the same _camera_io_lock as the face-
    tracking thread when bobert_companion is loaded, so a wake here can't
    collide with the running tracker's release/reopen.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:
        return False, f"opencv unavailable: {e}"
    box: dict[str, Any] = {"ok": False, "note": ""}

    bc = _bc()
    io_lock = getattr(bc, "_camera_io_lock", None) if bc else None

    def _do_wake():
        cap = None
        try:
            if io_lock is not None:
                io_lock.acquire()
            try:
                # First release any prior handle the face-tracker had open
                # by opening a fresh one — DirectShow refuses to hand out
                # the device to two open()s in parallel, so this also acts
                # as a "did we fail to claim the device?" probe.
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    box["note"] = "wake reopen failed — device refused open"
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
            if io_lock is not None:
                try:
                    io_lock.release()
                except Exception:
                    pass

    t = threading.Thread(target=_do_wake, name=f"diag-wake-{idx}", daemon=True)
    t.start()
    t.join(timeout=timeout_s)
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


def _probe_webcam() -> dict:
    start = _now()
    try:
        import cv2  # type: ignore
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"opencv not importable: {e}")

    details: dict[str, Any] = {}
    # Try indices 0..2 — most laptops expose the integrated camera at 0,
    # USB cams take 1/2 depending on enumeration order.
    cap = None
    cam_index = None
    for idx in (0, 1, 2):
        try:
            c = cv2.VideoCapture(idx)
            if c is not None and c.isOpened():
                cap = c
                cam_index = idx
                break
            else:
                try:
                    c.release()
                except Exception:  # pragma: no cover - defensive: cv2 release() on a non-opened capture (live-camera I/O, cv2 absent on CI)
                    pass
        except Exception:
            continue
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
            wake_ok, wake_note = _attempt_camera_wake(cam_index)
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
        if mean_val < 1.0:
            BLACK_FRAME_RETRIES = 3
            retry_means: list[float] = [mean_val]
            for _ in range(BLACK_FRAME_RETRIES):
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
                if rmean >= 1.0:
                    # Sensor warmed up — accept this frame and move on.
                    mean_val = rmean
                    frame = frame_r
                    break
            details["frame_mean"]       = round(mean_val, 2)
            details["frame_retry_means"] = [round(m, 2) for m in retry_means]
            if mean_val < 1.0:
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

    # Verify the face cascade loads — face_tracker depends on it.
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return _result(False, (_now() - start) * 1000.0,
                           error=f"face cascade failed to load from {cascade_path}",
                           details=details)
        details["cascade"] = "loaded"
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"face cascade failed: {type(e).__name__}: {e}",
                       details=details)

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

    # crash-fix-3 (2026-05-28): opening an `sd.rec()` capture stream from
    # this probe's daemon thread races the main loop's record_speech and
    # the wake-word InputStream. When the probe hits PER_PROBE_TIMEOUT_S
    # the thread is abandoned mid-capture, PortAudio is left holding the
    # buffer, and the next sweep triggers heap corruption. Skip the live
    # capture step entirely when JARVIS is awake (mic in active use by
    # the main loop). Enumeration plus PnP hardware count is enough to
    # confirm the audio stack is alive.
    bc = _bc()
    awake = bool(bc is not None and not getattr(bc, "_sleep_mode", [True])[0])
    mic_off = bool(getattr(bc, "_mic_input_disabled", lambda: False)())
    if awake or mic_off:
        details["live_capture_skipped"] = (
            ("mic hard-disabled (staging / MICROPHONE_INDEX < 0)" if mic_off
             else "JARVIS awake — mic owned by main loop")
            + "; skipping sd.rec() (crash-fix-3 / no-mic guard)"
        )
        return _result(True, (_now() - start) * 1000.0, details=details)

    try:
        import numpy as np  # type: ignore
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"numpy not importable: {e}",
                       details=details)

    def _capture_rms(device_idx: int | None,
                     duration_s: float = 0.25,
                     rate: int = 16000) -> tuple[float | None, str | None]:
        """Record ``duration_s`` of mono float32 audio from ``device_idx``
        (None → system default) and return ``(rms, err)``. ``rms`` is
        None when capture itself raised."""
        try:
            audio = sd.rec(int(duration_s * rate), samplerate=rate,
                           channels=1, dtype="float32",
                           device=device_idx)
            sd.wait()
            a = audio.squeeze() if hasattr(audio, "squeeze") else audio
            rms = float(np.sqrt(np.mean(np.square(a)))) if len(a) else 0.0
            return rms, None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    # Step 1: try the device JARVIS would actually use. This is the only
    # device that matters for "can JARVIS hear me right now".
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

    try:
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
        return _result(True, 0.0,
                       details={"skipped": "bobert_companion not loaded"})

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
        return _result(True, 0.0,
                       details={"skipped": "bobert_companion not loaded"})

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
def _probe_skill_imports() -> dict:
    start = _now()
    skills_dir = os.path.join(_PROJECT_DIR, "skills")
    if not os.path.isdir(skills_dir):
        return _result(False, (_now() - start) * 1000.0,
                       error=f"skills directory missing at {skills_dir}")

    # Most skills already imported successfully (load_skills ran at boot).
    # The probe checks the cache first — if the live module exists, we
    # don't re-import (re-importing has side effects: re-spawning daemon
    # threads, re-binding actions). For skills NOT in sys.modules we do
    # the cheap spec-only resolution to verify the file still parses.
    failures: list[dict] = []
    checked = 0

    try:
        entries = sorted(os.listdir(skills_dir))
    except Exception as e:
        return _result(False, (_now() - start) * 1000.0,
                       error=f"could not list skills dir: {e}")

    for name in entries:
        if not name.endswith(".py") or name.startswith("_"):
            continue
        stem = name[:-3]
        modname = f"skill_{stem}"
        checked += 1

        # If the live module is already loaded we trust it — it
        # registered actions at boot so a failed import would have
        # already surfaced.
        if sys.modules.get(modname) is not None:
            continue

        # Otherwise verify the file parses. We compile rather than
        # exec because exec can have side effects (network calls,
        # daemon threads) we don't want to repeat per sweep.
        path = os.path.join(skills_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            compile(source, path, "exec")
        except SyntaxError as e:
            failures.append({"skill": stem, "error": f"SyntaxError: {e.msg} (line {e.lineno})"})
        except Exception as e:
            failures.append({"skill": stem, "error": f"{type(e).__name__}: {e}"})

    details = {"checked": checked, "failures": failures,
               "loaded_modules": sum(1 for k in sys.modules if k.startswith("skill_"))}

    if failures:
        names = ", ".join(f["skill"] for f in failures)
        return _result(False, (_now() - start) * 1000.0,
                       error=f"{len(failures)} skill(s) failed to compile: {names}",
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
    """Run every probe sequentially and return the aggregated run dict."""
    sweep_start = _now()
    probes_out: dict[str, dict] = {}
    failed: list[str] = []
    sev_failed: dict[str, str] = {}

    for name, fn in PROBES.items():
        r = _run_with_timeout(fn, PER_PROBE_TIMEOUT_S, name=name)
        # Resolve severity: per-result override > subsystem default.
        if r.get("severity") is None:
            r["severity"] = SUBSYSTEM_SEVERITY.get(name, SEVERITY_MED)
        probes_out[name] = r
        if not r.get("ok", False):
            failed.append(name)
            sev_failed[name] = r["severity"]

    run = {
        "ts":           sweep_start,
        "iso":          _iso(sweep_start),
        "duration_ms":  round((_now() - sweep_start) * 1000.0, 1),
        "probes":       probes_out,
        "failed":       failed,
        "severity_failed": sev_failed,
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
    if poll_age > 30.0:
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


def _summarise(run: dict, queued: list[str]) -> str:
    failed = run.get("failed") or []
    duration_s = (run.get("duration_ms") or 0) / 1000.0
    if not failed:
        return f"All systems nominal, sir. ({duration_s:.1f}s sweep, {len(PROBES)} probes.)"

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
    return (f"Sir, {body}. Severity breakdown: {sev_breakdown}.{qstr} "
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
    if not failed:
        return (f"All systems nominal as of {age}, sir. "
                f"{len(PROBES)} probes, {(run['duration_ms'] / 1000):.1f}s sweep.")
    names = ", ".join(c.replace("_", " ") for c in failed)
    return (f"Last sweep was {age}, sir — {len(failed)} subsystem(s) reporting issues: {names}.")


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
        iso = run.get("iso") or _iso(run.get("ts", 0))
        if failed:
            lines.append(f"{iso}: {len(failed)} issue(s) — {', '.join(failed)}")
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
