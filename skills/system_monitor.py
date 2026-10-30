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
# The RAM alert's escalation state (v3.0.10, RAM_ALERT_ESCALATE_ONLY): the
# percent last spoken (None = armed: the next crossing alerts), and the stamp
# this monitor last put in the pulse's shared "ram" slot to hold the pulse's
# own hourly line while it stays latched (0.0 = none).
_ram_last_spoken_pct: list = [None]
_ram_hold_stamp = [0.0]
# v3.0.10 review: when the reading first went under RAM_REARM_BELOW_PCT while
# latched (0.0 = not under it; RAM_REARM_HOLD_S), and the pulse's own "ram"
# stamp from before the boot quiet began stamping it (None = not saved), put
# back when the quiet ends (_end_boot_ram_hold).
_ram_below_since = [0.0]
_ram_boot_saved: list = [None]
# The pulse's own RAM-line stamp when this monitor latched behind it (0.0 =
# none), so a re-arm hands the slot back at the right time.
_ram_pulse_said_at = [0.0]

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


def _cfg_value(name: str, default):
    """core.config.<name>, read live; ``default`` when unreadable (v3.0.10).
    Never raises."""
    try:
        import core.config as _cfg
        return getattr(_cfg, name, default)
    except Exception:
        return default


def _ram_alert_line(ram_pct: float) -> str:
    """The spoken high-RAM alert. With RAM_ALERT_NAME_USERS on (v3.0.10) it
    names the biggest memory users through the LOADED system_pulse skill
    (_memory_users / _memory_users_sentence - the helper an asked-for pulse
    uses) instead of "Things may start swapping shortly". The 3.0.9 line when
    the setting is off, the pulse is not loaded or nobody stands out.
    Never raises."""
    head = f"Sir, memory usage is at {ram_pct:.0f} percent."
    old = head + " Things may start swapping shortly."
    try:
        if not bool(_cfg_value("RAM_ALERT_NAME_USERS", False)):
            return old
        pulse = sys.modules.get("skill_system_pulse")
        if pulse is None:
            return old
        users = pulse._memory_users_sentence(pulse._memory_users()) or ""
        return f"{head} {users}" if users else old
    except Exception:
        return old


def _hold_shared_ram_slot(now: float) -> None:
    """While the RAM alert is latched (RAM_ALERT_ESCALATE_ONLY) or the boot
    quiet runs (RAM_ALERT_BOOT_QUIET_S), keep the pulse's shared "ram" slot
    stamped so its own hourly RAM line stays quiet too; the stamp is
    remembered so this monitor's own claim treats it as free. Pulse not
    loaded -> nothing. Never raises."""
    pulse = sys.modules.get("skill_system_pulse")
    lock = getattr(pulse, "_alert_lock", None)
    stamps = getattr(pulse, "_last_abnormal_alert", None)
    if lock is None or not isinstance(stamps, dict):
        return
    try:
        with lock:
            stamps["ram"] = now
            _ram_hold_stamp[0] = now
    except Exception:
        pass


def _boot_hold_ram_slot(ram_pct: float, now: float) -> None:
    """The boot quiet's hold on the pulse's "ram" slot (v3.0.10 review): the
    pulse's own stamp is saved before the first hold, and the slot is
    stamped only while the reading is at the pulse's RAM_ABNORMAL_PCT or
    over (under it the pulse would not speak anyway). Never raises."""
    pulse = sys.modules.get("skill_system_pulse")
    lock = getattr(pulse, "_alert_lock", None)
    stamps = getattr(pulse, "_last_abnormal_alert", None)
    if lock is None or not isinstance(stamps, dict):
        return
    try:
        if ram_pct < float(getattr(pulse, "RAM_ABNORMAL_PCT", 88.0)):
            return
        with lock:
            if _ram_boot_saved[0] is None:
                _ram_boot_saved[0] = float(stamps.get("ram", 0.0) or 0.0)
        _hold_shared_ram_slot(now)
    except Exception:
        pass


def _end_boot_ram_hold() -> None:
    """The boot quiet is over: put the pulse's own "ram" stamp back when the
    slot still holds the quiet's last stamp, so the pulse's hourly cooldown
    does not run from the quiet (it kept the pulse silent ~70 min, not 10).
    Never raises."""
    pulse = sys.modules.get("skill_system_pulse")
    lock = getattr(pulse, "_alert_lock", None)
    stamps = getattr(pulse, "_last_abnormal_alert", None)
    try:
        saved = _ram_boot_saved[0]
        _ram_boot_saved[0] = None
        if saved is None or lock is None or not isinstance(stamps, dict):
            return
        with lock:
            if _ram_hold_stamp[0] and stamps.get("ram") == _ram_hold_stamp[0]:
                stamps["ram"] = saved
                _ram_hold_stamp[0] = 0.0
    except Exception:
        pass


def _pulse_spoke_ram_last() -> float:
    """The pulse's shared "ram" stamp when it is the pulse's own RAM line:
    newer than this monitor's last alert and not this monitor's hold stamp
    (v3.0.10 review); else 0.0. Never raises (0.0)."""
    pulse = sys.modules.get("skill_system_pulse")
    stamps = getattr(pulse, "_last_abnormal_alert", None)
    try:
        if not isinstance(stamps, dict):
            return 0.0
        prev = float(stamps.get("ram", 0.0) or 0.0)
        if (prev > float(_last_ram_alert_at[0] or 0.0)
                and prev != _ram_hold_stamp[0]):
            return prev
        return 0.0
    except Exception:
        return 0.0


def _release_ram_hold() -> None:
    """The latch re-armed (v3.0.10 review): a pulse "ram" stamp that is only
    this monitor's hold goes back to the time of the last alert actually
    spoken, and the hold stamp is forgotten, so the next alert (from either
    alerter) waits out the shared hourly slot from that alert - never more
    often than 3.0.9's one an hour. Never raises."""
    pulse = sys.modules.get("skill_system_pulse")
    lock = getattr(pulse, "_alert_lock", None)
    stamps = getattr(pulse, "_last_abnormal_alert", None)
    try:
        if lock is not None and isinstance(stamps, dict):
            with lock:
                if (_ram_hold_stamp[0]
                        and stamps.get("ram") == _ram_hold_stamp[0]):
                    stamps["ram"] = max(float(_last_ram_alert_at[0] or 0.0),
                                        float(_ram_pulse_said_at[0] or 0.0))
        _ram_hold_stamp[0] = 0.0
        _ram_pulse_said_at[0] = 0.0
    except Exception:
        pass


def _ram_should_alert(ram_pct: float, now: float) -> bool:
    """One poll's RAM decision with RAM_ALERT_ESCALATE_ONLY on (v3.0.10).
    Armed (nothing spoken since the last re-arm): alert at RAM_ALERT_PCT,
    subject to the 10-min cooldown and the shared hourly slot. Latched: alert
    again only on a rise of RAM_REALERT_RISE_PTS over the last spoken
    percent (the 10-min cooldown still applies), or once on reaching
    RAM_REALERT_CRITICAL_PCT; after RAM_REARM_HOLD_S under
    RAM_REARM_BELOW_PCT the alert re-arms. While latched the pulse's hourly
    RAM line is held; a pulse RAM line spoken first latches it too. Updates
    the latch when it says True. Never raises (False)."""
    try:
        rearm = float(_cfg_value("RAM_REARM_BELOW_PCT", 85.0))
        rise = float(_cfg_value("RAM_REALERT_RISE_PTS", 3.0))
        last = _ram_last_spoken_pct[0]
        if ram_pct < rearm:
            # v3.0.10 review (RAM_REARM_HOLD_S): re-arm only after a sustained
            # stay under the re-arm point, and forget the hold stamp so a
            # re-armed alert waits out the shared hourly slot again.
            hold_s = float(_cfg_value("RAM_REARM_HOLD_S", 0) or 0)
            if _ram_below_since[0] <= 0.0:
                _ram_below_since[0] = now
            if last is not None and now - _ram_below_since[0] >= hold_s:
                _ram_last_spoken_pct[0] = None
                _release_ram_hold()
            return False
        _ram_below_since[0] = 0.0
        if last is not None:
            _hold_shared_ram_slot(now)
        if ram_pct < RAM_ALERT_PCT:
            return False
        crit = float(_cfg_value("RAM_REALERT_CRITICAL_PCT", 0) or 0)
        if (last is not None and crit > 0 and ram_pct >= crit
                and float(last) < crit):
            # v3.0.10 review: heading for out-of-memory speaks once per
            # latch, whatever the rise and the 10-min cooldown.
            _ram_last_spoken_pct[0] = float(ram_pct)
            return True
        if (now - _last_ram_alert_at[0]) <= ALERT_COOLDOWN_SECONDS:
            return False
        if last is None:
            if not _claim_shared_ram_alert(now):
                said_at = _pulse_spoke_ram_last()
                if said_at:
                    # v3.0.10 review: the pulse spoke first (at its 88 %):
                    # latch at this reading, so its hourly line is held from
                    # here on too.
                    _ram_pulse_said_at[0] = said_at
                    _ram_last_spoken_pct[0] = float(ram_pct)
                    _hold_shared_ram_slot(now)
                return False
        elif ram_pct < float(last) + rise:
            return False
        else:
            _hold_shared_ram_slot(now)
        _ram_last_spoken_pct[0] = float(ram_pct)
        return True
    except Exception:
        return False


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
            prev = stamps.get("ram", 0.0)
            # v3.0.10: a stamp this monitor made only to hold the pulse while
            # its alert was latched (_hold_shared_ram_slot) is no claim.
            ours = bool(prev) and prev == _ram_hold_stamp[0]
            if not ours and (now - prev) <= cooldown:
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


def _ram_users_sentence(ram_pct: float, arg: str = "") -> str:
    """Live 2026-10-28 15:59:28: "Jarvis, why is the ram so high?" at 92 %.
    The SYSTEM HEALTH turn block teaches check_system, whose report named
    only the CPU offender. At RAM_ALERT_PCT or over, or when his words (or
    the action argument) ask about ram/memory, name the biggest memory users
    via the LOADED system_pulse skill (its _memory_users /
    _memory_users_sentence - not a copy, so there is one implementation).
    Pulse not loaded -> "". Never raises."""
    try:
        pulse = sys.modules.get("skill_system_pulse")
        if pulse is None:
            return ""
        said = f"{pulse._turn_words()} {arg if isinstance(arg, str) else ''}"
        if not (ram_pct >= RAM_ALERT_PCT or pulse._asks_about_memory(said)):
            return ""
        return pulse._memory_users_sentence(pulse._memory_users()) or ""
    except Exception:
        return ""


def _memory_question_lead(ram_pct: float, arg: str = "") -> str:
    """v3.0.7: when his words (or the action argument) ask about ram/memory
    and SYSTEM_PULSE_MEMORY_ANSWER is on, the opener is the memory reading
    itself - the LOADED system_pulse skill's _memory_lead ("Memory at 82
    percent, sir.") - instead of "Systems nominal" / "holding up" (live
    2026-10-28 18:53 the pulse answered "why is the RAM so high?" at 82 %
    with its all-nominal line; this sibling opens "nominal" under 75 %).
    "" otherwise (pulse not loaded, setting off, not asked).
    Never raises."""
    try:
        pulse = sys.modules.get("skill_system_pulse")
        if pulse is None or not pulse._memory_answer_enabled():
            return ""
        said = f"{pulse._turn_words()} {arg if isinstance(arg, str) else ''}"
        if not pulse._asks_about_memory(said):
            return ""
        return pulse._memory_lead(ram_pct) or ""
    except Exception:
        return ""


def _build_report(arg: str = "") -> str:
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
    users = _ram_users_sentence(vm.percent, arg)
    mem_lead = _memory_question_lead(vm.percent, arg)
    if mem_lead:
        # Asked about memory: the number, then who holds it, then the rest.
        mem_part = f"{mem_lead} {users}" if users else mem_lead
        if cpu_pct >= 80:
            # Review: "why is the CPU so high, is it a memory leak?" - the CPU
            # is what is working hard, so that opener stays and the memory
            # sentence follows it instead of replacing it.
            opener = f"{opener} {mem_part}"
        else:
            opener = mem_part
    elif users:
        cpu_ram += ". " + users.rstrip(".")

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
    # v3.0.10 (RAM_ALERT_BOOT_QUIET_S): the brain model loading fills memory
    # in the first minutes; no RAM alert until this long after the start.
    started_at = time.time()
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
            boot_quiet = float(_cfg_value("RAM_ALERT_BOOT_QUIET_S", 0) or 0)
            in_boot_quiet = boot_quiet > 0 and (now - started_at) < boot_quiet
            if not in_boot_quiet and _ram_boot_saved[0] is not None:
                # v3.0.10 review: the quiet is over; the pulse's own stamp
                # comes back so its hourly cooldown does not run from it.
                _end_boot_ram_hold()
            if in_boot_quiet:
                # v3.0.10: still booting (the brain model is loading); the
                # pulse's 88 % line is held with this monitor's.
                _boot_hold_ram_slot(ram_pct, now)
            elif bool(_cfg_value("RAM_ALERT_ESCALATE_ONLY", False)):
                # v3.0.10: once per rise, not once an hour while it stays high.
                if _ram_should_alert(ram_pct, now):
                    _enqueue_speech(_ram_alert_line(ram_pct))
                    with _alert_lock:
                        _last_ram_alert_at[0] = now
            elif ram_pct >= RAM_ALERT_PCT:
                if ((now - _last_ram_alert_at[0]) > ALERT_COOLDOWN_SECONDS
                        and _claim_shared_ram_alert(now)):
                    _enqueue_speech(_ram_alert_line(ram_pct))
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
    def check_system(arg: str = "") -> str:
        try:
            return _build_report(arg)
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
