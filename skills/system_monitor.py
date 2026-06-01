"""
System health monitor skill for JARVIS.

Actions:
  check_system   — report current CPU%, RAM, top 3 CPU-hogging processes,
                   C: disk space, and network up/down rates in JARVIS style.

Background monitor:
  Polls CPU + RAM every 5 seconds. If CPU stays above CPU_ALERT_PCT for
  CPU_ALERT_SUSTAIN_SECONDS, or RAM goes above RAM_ALERT_PCT at any sample,
  queues a spoken alert. Cooldown prevents repeats.
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
except ImportError:
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
    for p in procs:
        try:
            cpu = p.cpu_percent(None)
            name = p.info.get("name") or f"pid {p.pid}"
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
        if "chrome" in primary.lower():
            cpu_ram += f". Chrome is, as usual, the primary offender."
        else:
            cpu_ram += f". {primary} is the primary offender."

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
    while True:
        try:
            cpu_pct = psutil.cpu_percent(interval=POLL_INTERVAL_SECONDS)
            ram_pct = psutil.virtual_memory().percent
            now = time.time()

            cpu_samples.append((now, cpu_pct >= CPU_ALERT_PCT))
            cutoff = now - CPU_ALERT_SUSTAIN_SECONDS
            while cpu_samples and cpu_samples[0][0] < cutoff:
                cpu_samples.popleft()

            # Only evaluate once the window is mostly filled, so we don't alert
            # off a single sample at startup.
            window_span = (cpu_samples[-1][0] - cpu_samples[0][0]
                           if len(cpu_samples) >= 2 else 0.0)
            if window_span >= CPU_ALERT_SUSTAIN_SECONDS * 0.9:
                high_count = sum(1 for _, h in cpu_samples if h)
                if high_count / len(cpu_samples) >= CPU_HIGH_SAMPLE_RATIO:
                    if (now - _last_cpu_alert_at[0]) > ALERT_COOLDOWN_SECONDS:
                        top = _top_processes(1)
                        culprit = (f" — {top[0][0]} appears to be the culprit"
                                   if top else "")
                        _enqueue_speech(
                            f"Sir, CPU usage has been pinned above 90 percent "
                            f"for most of the past minute{culprit}. You may "
                            f"want to investigate."
                        )
                        with _alert_lock:
                            _last_cpu_alert_at[0] = now
                        # Clear so the next alert needs a fresh window of evidence.
                        cpu_samples.clear()

            # RAM single-sample check
            if ram_pct >= RAM_ALERT_PCT:
                if (now - _last_ram_alert_at[0]) > ALERT_COOLDOWN_SECONDS:
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
