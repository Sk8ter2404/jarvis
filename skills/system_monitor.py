"""
System health monitor skill for JARVIS.

Actions:
  check_system   — report current CPU%, RAM, top 3 CPU-hogging processes,
                   C: disk space, and network up/down rates in JARVIS style.

Background monitor:
  Polls CPU + RAM every 5 seconds. If CPU stays above CPU_ALERT_PCT for
  CPU_ALERT_SUSTAIN_SECONDS, or RAM goes above RAM_ALERT_PCT at any sample,
  queues a spoken alert. Cooldown prevents repeats.

  The CPU culprit (2026-10-02) is the process that used the most CPU over the
  WHOLE window (per-process CPU time at the window's first high sample vs at
  the alert), named the way he would say it ("Windows Defender", "Chrome",
  never "MsMpEng.exe"). Routine Windows maintenance (a Defender scan, Search
  indexing, Windows Update) is logged, not announced; JARVIS's own process is
  "my own speech recognition". Live 2026-10-01: "MsMpEng.exe appears to be the
  culprit. You may want to investigate" over a show at 18:14:56 (a routine
  Defender scan), and "python.exe ..." at 22:06:12 (JARVIS itself) - each named
  from one 0.5 s sample taken after the window. The spoken alert waits for the
  owner like every queued line (the presence hold in _speak_pending).
"""
import json
import logging
import os
import sys
import threading
import time
from collections import deque

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - psutil is a guaranteed dep (dev + CI); import never fails
    _HAS_PSUTIL = False

# ─── thresholds ──────────────────────────────────────────────────────────
CPU_ALERT_PCT             = 90.0
CPU_ALERT_SUSTAIN_SECONDS = 60.0
CPU_HIGH_SAMPLE_RATIO     = 0.8    # ≥80% of samples in window must be high
RAM_ALERT_PCT             = 90.0
POLL_INTERVAL_SECONDS     = 5.0
ALERT_COOLDOWN_SECONDS    = 600.0   # 10 min between repeat alerts
INITIAL_DELAY_SECONDS     = 120     # let JARVIS finish booting first
# ─────────────────────────────────────────────────────────────────────────

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")

if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

_speech_lock = threading.Lock()
_alert_lock = threading.Lock()
_last_cpu_alert_at = [0.0]
_last_ram_alert_at = [0.0]

# ─── naming the CPU culprit (2026-10-02) ─────────────────────────────────
# The key _top_processes / _window_culprit use for JARVIS's own process.
_SELF_KEY = "__jarvis__"
# Image name (lower-case) -> how he would say it.
_FRIENDLY_PROCESS_NAMES = {
    "msmpeng.exe": "Windows Defender",
    "mpdefendercoreservice.exe": "Windows Defender",
    "nissrv.exe": "Windows Defender",
    "mssense.exe": "Windows Defender",
    "searchindexer.exe": "Windows Search indexing",
    "searchprotocolhost.exe": "Windows Search indexing",
    "searchfilterhost.exe": "Windows Search indexing",
    "tiworker.exe": "Windows Update",
    "trustedinstaller.exe": "Windows Update",
    "mousocoreworker.exe": "Windows Update",
    "usoclient.exe": "Windows Update",
    "wuauclt.exe": "Windows Update",
    "sihclient.exe": "Windows Update",
    "dismhost.exe": "Windows Update",
    "compattelrunner.exe": "Windows telemetry",
    "defrag.exe": "the drive optimiser",
    "chrome.exe": "Chrome",
    "msedge.exe": "Edge",
    "msedgewebview2.exe": "an Edge web view",
    "firefox.exe": "Firefox",
    "ms-teams.exe": "Teams",
    "teams.exe": "Teams",
    "explorer.exe": "File Explorer",
    "dwm.exe": "the desktop compositor",
    "code.exe": "VS Code",
    "claude.exe": "the Claude app",
    "obs64.exe": "OBS",
    "bambu-studio.exe": "Bambu Studio",
    "bambustudio.exe": "Bambu Studio",
    "nextcloud.exe": "Nextcloud",
    "onedrive.exe": "OneDrive",
    "llama-server.exe": "the local model server",
    "ollama.exe": "the local model server",
    "fortniteclient-win64-shipping.exe": "Fortnite",
}
# Routine Windows maintenance: a sustained CPU window it causes is LOGGED, not
# spoken - there is nothing for him to investigate, and it runs whenever the
# machine is idle (exactly when he is watching a show or away).
_MAINTENANCE_PROCESSES = frozenset({
    "msmpeng.exe", "mpdefendercoreservice.exe", "nissrv.exe", "mssense.exe",
    "searchindexer.exe", "searchprotocolhost.exe", "searchfilterhost.exe",
    "tiworker.exe", "trustedinstaller.exe", "mousocoreworker.exe",
    "usoclient.exe", "wuauclt.exe", "sihclient.exe", "dismhost.exe",
    "compattelrunner.exe", "defrag.exe",
})
_IDLE_NAMES = frozenset({"system idle process", "idle"})


def _spoken_process_name(name: str) -> str:
    """How to say a process image name: the friendly map, JARVIS's own process
    as "my own speech recognition", else the image without ".exe"."""
    key = (name or "").strip().lower()
    if key == _SELF_KEY:
        return "my own speech recognition"
    if key in _FRIENDLY_PROCESS_NAMES:
        return _FRIENDLY_PROCESS_NAMES[key]
    stem = (name or "").strip()
    if stem.lower().endswith(".exe"):
        stem = stem[:-4]
    if stem.islower():
        stem = stem[:1].upper() + stem[1:]
    return stem or "an unnamed process"


def _is_maintenance(name: str) -> bool:
    return (name or "").strip().lower() in _MAINTENANCE_PROCESSES


def _proc_cpu_snapshot() -> dict:
    """{pid: (image_name, cpu_seconds)} for every process right now (user +
    system CPU time). {} when unavailable. Never raises."""
    if not _HAS_PSUTIL:
        return {}
    snap: dict = {}
    try:
        for p in psutil.process_iter(["name", "cpu_times"]):
            try:
                info = p.info or {}
                ct = info.get("cpu_times")
                if ct is None:
                    continue
                name = info.get("name") or f"pid {p.pid}"
                snap[int(p.pid)] = (str(name),
                                    float(ct.user) + float(ct.system))
            except Exception:
                continue
    except Exception:
        return {}
    return snap


def _window_culprit(base: dict, now: dict, elapsed_s: float, *,
                    own_pid: int | None = None,
                    ncpu: int | None = None) -> tuple | None:
    """(name_key, avg_pct) for the process image that used the most CPU
    between two _proc_cpu_snapshot()s taken elapsed_s apart - averaged over
    the window, not one instant. avg_pct is a share of the whole machine
    (cpu_percent's scale). JARVIS's own pid is keyed _SELF_KEY; a process that
    started inside the window counts all its CPU time. None when there is
    nothing to compare. Never raises."""
    try:
        if not base or not now or not elapsed_s or elapsed_s <= 0:
            return None
        own = os.getpid() if own_pid is None else int(own_pid)
        if ncpu is None:
            try:
                ncpu = int(psutil.cpu_count() or 1)
            except Exception:
                ncpu = 1
        totals: dict = {}
        for pid, (name, cpu_now) in now.items():
            prev = base.get(pid)
            start = prev[1] if (prev is not None and prev[0] == name) else 0.0
            used = cpu_now - start
            if used <= 0:
                continue
            key = _SELF_KEY if pid == own else name.lower()
            if key in _IDLE_NAMES:
                continue
            totals[key] = totals.get(key, 0.0) + used
        if not totals:
            return None
        key = max(totals, key=totals.get)
        return key, totals[key] / float(elapsed_s) / max(1, int(ncpu)) * 100.0
    except Exception:
        return None


def _cpu_alert_line(culprit_key: str | None) -> str | None:
    """The spoken sustained-CPU alert, or None when the culprit is routine
    Windows maintenance (log-only)."""
    head = ("Sir, CPU usage has been pinned above 90 percent for most of the "
            "past minute")
    if not culprit_key:
        return head + ". You may want to investigate."
    if _is_maintenance(culprit_key):
        return None
    if culprit_key == _SELF_KEY:
        return head + " — mostly my own speech recognition."
    return (f"{head} — {_spoken_process_name(culprit_key)} appears to be the "
            f"culprit. You may want to investigate.")


def _claim_shared_ram_alert(now: float) -> bool:
    """2026-10-01: one spoken high-RAM alert per hour across BOTH alerters.
    skills/system_pulse.py speaks at 88 % with an hourly per-key cooldown and
    this monitor at 90 % with its own 10-min one; neither saw the other, so on
    2026-09-30 the owner heard both (plus self_diagnostic) within 3.5 minutes.
    Check-and-stamp the pulse's own "ram" stamp under its lock, so whichever
    fires first holds the slot for PROACTIVE_COOLDOWN_SECONDS. Pulse not
    loaded -> True (this monitor's own cooldown alone, as before)."""
    pulse = sys.modules.get("skill_system_pulse")
    lock = getattr(pulse, "_alert_lock", None)
    stamps = getattr(pulse, "_last_abnormal_alert", None)
    cooldown = getattr(pulse, "PROACTIVE_COOLDOWN_SECONDS", None)
    if lock is None or not isinstance(stamps, dict) or cooldown is None:
        return True
    try:
        with lock:
            if (now - stamps.get("ram", 0.0)) <= cooldown:
                return False
            stamps["ram"] = now
    except Exception:
        return True
    return True


def _enqueue_speech(message: str) -> None:
    """Route a proactive announcement through bobert_companion's public
    proactive_announce() API when available, falling back to a direct atomic
    write against pending_speech.json if the parent module hasn't loaded yet
    (e.g. unit test, import-time skill registration before bobert_companion
    finishes initialising)."""
    try:
        import importlib
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="system_monitor")
            return
    except Exception:
        # Fall through to local write — never let a broken parent import
        # silence a system-monitor alert.
        pass

    with _speech_lock:
        data = []
        if os.path.exists(_SPEECH_QUEUE):
            try:
                with open(_SPEECH_QUEUE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append({"ts": time.time(), "message": message})
        try:
            _atomic_write_json(_SPEECH_QUEUE, data)
        except Exception as e:
            # Atomic write failed (e.g. read-only network share, full disk,
            # permission denied). Fall back to console so the alert isn't
            # silently lost — at minimum the user sees it in the log stream.
            print(f"  [sysmon] speech-queue write failed ({e}); alert: {message}")


def _top_processes(n: int = 3) -> list[tuple[str, float]]:
    """Return [(name, cpu_pct), ...] for the n top CPU-hogging processes.
    First snapshot is throwaway because psutil.cpu_percent() needs two reads."""
    if not _HAS_PSUTIL:
        return []
    procs = []
    for p in psutil.process_iter(["name"]):
        try:
            p.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        procs.append(p)
    time.sleep(0.5)
    rated = []
    own = os.getpid()
    for p in procs:
        try:
            cpu = p.cpu_percent(None)
            name = p.info.get("name") or f"pid {p.pid}"
            if getattr(p, "pid", None) == own:
                name = _SELF_KEY      # JARVIS itself, not "python.exe"
            rated.append((name, cpu))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # Sort by CPU% desc, dedupe names by summing (Chrome has many child procs)
    aggregated: dict[str, float] = {}
    for name, cpu in rated:
        aggregated[name] = aggregated.get(name, 0.0) + cpu
    top = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    # Drop the System Idle process which always dominates
    top = [(n, c) for (n, c) in top
           if n.lower() not in ("system idle process", "idle")]
    return top[:n]


def _network_rates(window_seconds: float = 1.0) -> tuple[float, float]:
    """Return (down_kbps, up_kbps) measured over a short window."""
    if not _HAS_PSUTIL:
        return 0.0, 0.0
    a = psutil.net_io_counters()
    time.sleep(window_seconds)
    b = psutil.net_io_counters()
    down = (b.bytes_recv - a.bytes_recv) / window_seconds / 1024.0
    up   = (b.bytes_sent - a.bytes_sent) / window_seconds / 1024.0
    return down, up


def _build_report() -> str:
    if not _HAS_PSUTIL:
        return ("System monitor requires the psutil package — run "
                "pip install psutil and restart me.")

    cpu_pct = psutil.cpu_percent(interval=0.5)
    vm      = psutil.virtual_memory()
    ram_used_gb  = vm.used  / (1024**3)
    ram_total_gb = vm.total / (1024**3)

    top = _top_processes(3)
    down_kbps, up_kbps = _network_rates()

    try:
        disk = psutil.disk_usage("C:\\")
        c_free_gb  = disk.free / (1024**3)
        c_total_gb = disk.total / (1024**3)
    except Exception:
        c_free_gb = c_total_gb = 0.0

    # Sentence 1 — overall posture
    if cpu_pct < 50 and vm.percent < 75:
        opener = "Systems nominal, sir."
    elif cpu_pct < 80 and vm.percent < 90:
        opener = "Systems holding up, sir."
    else:
        opener = "Systems are working rather hard at the moment, sir."

    # Sentence 2 — CPU + RAM
    cpu_ram = (
        f"CPU at {cpu_pct:.0f} percent, {ram_used_gb:.0f} of {ram_total_gb:.0f} "
        f"gigs committed"
    )
    if top:
        primary = top[0][0]
        # No closing period: the return below ends this sentence (it used to
        # read "the primary offender.. C drive has ...").
        if "chrome" in primary.lower():
            cpu_ram += ". Chrome is, as usual, the primary offender"
        elif primary == _SELF_KEY:
            cpu_ram += ". Most of that is my own speech recognition"
        else:
            cpu_ram += f". {_spoken_process_name(primary)} is the primary offender"

    # Sentence 3 — disk + network
    extras = []
    if c_total_gb:
        extras.append(f"C drive has {c_free_gb:.0f} gigs free of {c_total_gb:.0f}")
    if down_kbps > 5 or up_kbps > 5:
        extras.append(f"network at {down_kbps:.0f} down, {up_kbps:.0f} up kilobytes per second")
    extras_str = ". ".join(extras)

    if extras_str:
        return f"{opener} {cpu_ram}. {extras_str}."
    return f"{opener} {cpu_ram}."


def _monitor_loop():
    """Sample CPU + RAM at POLL_INTERVAL. Sliding window over the last
    CPU_ALERT_SUSTAIN_SECONDS — alert when ≥ CPU_HIGH_SAMPLE_RATIO of
    samples in that window are above CPU_ALERT_PCT. This way a borderline-
    pegged process that briefly dips below 90% doesn't reset the counter."""
    if not _HAS_PSUTIL:
        return
    time.sleep(INITIAL_DELAY_SECONDS)

    # (timestamp, was_high) — pruned to last CPU_ALERT_SUSTAIN_SECONDS each tick.
    cpu_samples: deque[tuple[float, bool]] = deque()
    # (timestamp, _proc_cpu_snapshot()) taken at each HIGH sample, pruned the
    # same way: the oldest one is the culprit baseline for the window.
    proc_snaps: deque[tuple[float, dict]] = deque()
    while True:
        try:
            cpu_pct = psutil.cpu_percent(interval=POLL_INTERVAL_SECONDS)
            ram_pct = psutil.virtual_memory().percent
            now = time.time()

            cpu_samples.append((now, cpu_pct >= CPU_ALERT_PCT))
            if cpu_pct >= CPU_ALERT_PCT:
                snap = _proc_cpu_snapshot()
                if snap:
                    proc_snaps.append((now, snap))
            cutoff = now - CPU_ALERT_SUSTAIN_SECONDS
            while cpu_samples and cpu_samples[0][0] < cutoff:
                cpu_samples.popleft()
            while proc_snaps and proc_snaps[0][0] < cutoff:
                proc_snaps.popleft()

            # Only evaluate once the window is mostly filled, so we don't alert
            # off a single sample at startup.
            window_span = (cpu_samples[-1][0] - cpu_samples[0][0]
                           if len(cpu_samples) >= 2 else 0.0)
            if window_span >= CPU_ALERT_SUSTAIN_SECONDS * 0.9:
                high_count = sum(1 for _, h in cpu_samples if h)
                if high_count / len(cpu_samples) >= CPU_HIGH_SAMPLE_RATIO:
                    if (now - _last_cpu_alert_at[0]) > ALERT_COOLDOWN_SECONDS:
                        # The culprit is averaged over the window; one
                        # instant's top process only when no baseline exists.
                        culprit = None
                        if proc_snaps:
                            base_ts, base_snap = proc_snaps[0]
                            culprit = _window_culprit(base_snap,
                                                      _proc_cpu_snapshot(),
                                                      now - base_ts)
                        if culprit is None:
                            top = _top_processes(1)
                            culprit = ((top[0][0].lower(), top[0][1])
                                       if top else None)
                        key = culprit[0] if culprit else None
                        line = _cpu_alert_line(key)
                        if line:
                            _enqueue_speech(line)
                        else:
                            print(f"  [sysmon] CPU pinned for the past minute by "
                                  f"{_spoken_process_name(key)} ({key}, "
                                  f"{culprit[1]:.0f}% avg) — routine maintenance, "
                                  f"not announced")
                        with _alert_lock:
                            _last_cpu_alert_at[0] = now
                        # Clear so the next alert needs a fresh window of evidence.
                        cpu_samples.clear()
                        proc_snaps.clear()

            # RAM single-sample check
            if ram_pct >= RAM_ALERT_PCT:
                if ((now - _last_ram_alert_at[0]) > ALERT_COOLDOWN_SECONDS
                        and _claim_shared_ram_alert(now)):
                    _enqueue_speech(
                        f"Sir, memory usage is at {ram_pct:.0f} percent. "
                        f"Things may start swapping shortly."
                    )
                    with _alert_lock:
                        _last_ram_alert_at[0] = now

        except Exception:
            logging.exception("[sysmon] monitor loop iteration failed")
            time.sleep(POLL_INTERVAL_SECONDS)

        # Hard floor on iteration cadence. psutil.cpu_percent(interval=N) is
        # supposed to block for N seconds, but if it ever returns early
        # (psutil bug, interval misconfigured to 0, etc.) this prevents the
        # loop from pegging a CPU core.
        time.sleep(0.1)


def register(actions):
    def check_system(_: str = "") -> str:
        try:
            return _build_report()
        except Exception as e:
            return f"system check failed: {e}"

    actions["check_system"] = check_system

    if _HAS_PSUTIL:
        t = threading.Thread(target=_monitor_loop, daemon=True)
        t.start()
        print(
            f"  [sysmon] background monitor active — CPU>{CPU_ALERT_PCT:.0f}% "
            f"for {CPU_ALERT_SUSTAIN_SECONDS:.0f}s or RAM>{RAM_ALERT_PCT:.0f}% triggers an alert"
        )
    else:
        print("  [sysmon] psutil not installed — actions registered but "
              "background monitor disabled. pip install psutil to enable.")
