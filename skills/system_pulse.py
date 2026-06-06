"""
System pulse skill for JARVIS — the wider sibling of system_monitor.

Actions:
  system_pulse  — manually requested "status report" that gathers CPU, RAM,
                  GPU temperature, disk free, network throughput, battery,
                  system uptime, active-window count, Bambu print status, and
                  Anthropic credit balance, then renders a JARVIS-cadence
                  sentence ending with "Anything further?".

Proactive thread:
  Every PULSE_PROACTIVE_INTERVAL_SECONDS (default 15 min) the same readings
  are gathered silently. If anything looks abnormal — high CPU, high RAM,
  low disk, hot GPU, low battery, failed Bambu print, low Anthropic credits
  — JARVIS speaks the pulse unprompted, leading with the abnormal item.

HUD widget:
  Every PULSE_HUD_REFRESH_SECONDS (default 15 s) a compact strip of the
  non-redundant metrics (GPU temp, battery, uptime, app count, network) is
  written into hud_state.json under the key `pulse_strip`. The HUD overlay
  picks it up and renders it as an extra line below the existing ticker
  zone, so the data is visible without speaking.

Why not just reuse skills/system_monitor?
  system_monitor is a narrow CPU/RAM alerter — single threshold, single
  cadence. system_pulse is a fleet-wide snapshot that pulls cross-skill
  data (Bambu, credits) and adds GPU/battery/uptime/app-count metrics
  the existing alerter doesn't touch.
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time

from core.atomic_io import _atomic_write_json

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - psutil is a guaranteed dep (dev + CI); import never fails
    _HAS_PSUTIL = False

try:
    import pygetwindow as gw   # type: ignore
    _HAS_GW = True
except Exception:
    _HAS_GW = False

# ─── thresholds (what counts as "abnormal" for the proactive trigger) ────
CPU_ABNORMAL_PCT          = 85.0
RAM_ABNORMAL_PCT          = 88.0
DISK_FREE_ABNORMAL_GB     = 20.0
GPU_TEMP_ABNORMAL_C       = 80.0
BATTERY_LOW_PCT           = 20.0
CREDITS_LOW_DOLLARS       = 5.0
NETWORK_HOT_KBPS          = 50_000   # ≥50 MB/s sustained is unusual on a desktop

# ─── cadences ────────────────────────────────────────────────────────────
PULSE_PROACTIVE_INTERVAL_SECONDS = 15 * 60
PULSE_HUD_REFRESH_SECONDS        = 15
PROACTIVE_COOLDOWN_SECONDS       = 60 * 60   # don't repeat the same abnormal item more than hourly
INITIAL_PROACTIVE_DELAY_SECONDS  = 180       # let JARVIS settle before the first pulse

_PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE   = os.path.join(_PROJECT_DIR, "pending_speech.json")
_HUD_STATE_FILE = os.path.join(_PROJECT_DIR, "hud_state.json")
_CREDITS_STATE  = os.path.join(_PROJECT_DIR, "credits_state.json")

_speech_lock         = threading.Lock()
_alert_lock          = threading.Lock()
_last_abnormal_alert = {}   # reason_key -> ts (guarded by _alert_lock)

# Latest gathered pulse cached by the HUD loop so the voice action can format
# a reply instantly instead of re-running nvidia-smi + a 0.6 s net sample on
# the main voice thread. {"pulse": dict, "ts": float}; None until first run.
_pulse_cache_lock = threading.Lock()
_pulse_cache: dict | None = None
# A status readout is fine up to this age; the HUD loop refreshes every
# PULSE_HUD_REFRESH_SECONDS (15 s), so bound staleness explicitly at 20 s.
PULSE_CACHE_MAX_AGE_SECONDS = 20.0


# ─── speech queue (mirrors the pattern used by sibling skills) ───────────

def _enqueue_speech(message: str) -> None:
    """Route a proactive pulse announcement through bobert_companion's public
    proactive_announce() API when available, falling back to a direct atomic
    write against pending_speech.json via the shared `_atomic_write_json`
    helper so this writer can't race with sibling skills that touch the same
    queue file."""
    try:
        import importlib
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="pulse")
            return
    except Exception:
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
            print(f"  [pulse] speech-queue write failed ({e}); pulse: {message}")


# ─── HUD pulse strip publishing ──────────────────────────────────────────

def _publish_hud_strip(strip: str) -> None:
    """Merge `pulse_strip` into HUD state via the canonical writer so the
    sibling `status_panel` skill can't clobber our field in a r-m-w race."""
    writer = None
    try:
        writer = skill_utils.get("write_hud_state")  # type: ignore[name-defined]
    except NameError:
        writer = None
    if writer is None:
        return
    try:
        writer(pulse_strip=strip, pulse_updated_at=time.time())
    except Exception:
        pass


# ─── metric collectors ───────────────────────────────────────────────────

def _read_cpu_ram() -> tuple[float, float, float]:
    """Return (cpu_pct, ram_pct, ram_used_gb)."""
    if not _HAS_PSUTIL:
        return 0.0, 0.0, 0.0
    cpu = psutil.cpu_percent(interval=0.4)
    vm  = psutil.virtual_memory()
    return cpu, vm.percent, vm.used / (1024 ** 3)


def _read_disk_free_gb() -> float:
    if not _HAS_PSUTIL:
        return 0.0
    try:
        return psutil.disk_usage("C:\\").free / (1024 ** 3)
    except Exception:
        return 0.0


def _read_network_rates() -> tuple[float, float]:
    """Return (down_kbps, up_kbps) over a 0.6s window."""
    if not _HAS_PSUTIL:
        return 0.0, 0.0
    try:
        a = psutil.net_io_counters()
        time.sleep(0.6)
        b = psutil.net_io_counters()
        down = (b.bytes_recv - a.bytes_recv) / 0.6 / 1024.0
        up   = (b.bytes_sent - a.bytes_sent) / 0.6 / 1024.0
        return down, up
    except Exception:
        return 0.0, 0.0


def _read_battery() -> tuple[float, bool] | None:
    """Return (percent, plugged_in) or None on a desktop with no battery."""
    if not _HAS_PSUTIL:
        return None
    try:
        b = psutil.sensors_battery()
        if b is None:
            return None
        return float(b.percent), bool(b.power_plugged)
    except Exception:
        return None


def _read_uptime_seconds() -> float:
    if not _HAS_PSUTIL:
        return 0.0
    try:
        return max(0.0, time.time() - psutil.boot_time())
    except Exception:
        return 0.0


def _read_gpu_temp_c(hwinfo_gpu_temp_c: float | None = None) -> float | None:
    """Best-effort GPU temperature in Celsius.
    1) nvidia-smi if available (NVIDIA, most reliable)
    2) HWiNFO shared memory (reliable on Windows when Shared Memory Support is on)
    3) psutil.sensors_temperatures() — Windows rarely supports this but try
    Returns None if no GPU temp could be read.

    `hwinfo_gpu_temp_c` is the pre-fetched HWiNFO `gpu_temp_c` (from a single
    hwinfo.summary() in _gather_pulse). When the caller supplies it we skip the
    in-function HWiNFO read so the shared-memory block is parsed once per pulse;
    the nvidia-smi-first ordering is preserved either way."""
    try:
        exe = shutil.which("nvidia-smi")
        if exe:
            out = subprocess.run(
                [exe, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0,
                creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
            )
            line = (out.stdout or "").strip().splitlines()
            if line:
                # Multiple GPUs → take the hottest
                temps = []
                for v in line:
                    v = v.strip()
                    if v.isdigit():
                        temps.append(int(v))
                if temps:
                    return float(max(temps))
    except Exception:
        pass

    # 2) HWiNFO shared memory — reliable on Windows when Shared Memory Support is
    #    on (psutil.sensors_temperatures rarely returns a GPU temp there). Use a
    #    pre-fetched reading when given (single-read path); otherwise read here.
    try:
        gt = hwinfo_gpu_temp_c
        if gt is None:
            from audio import hwinfo
            gt = hwinfo.summary().get("gpu_temp_c")
        if gt is not None:
            return float(gt)
    except Exception:
        pass

    if _HAS_PSUTIL:
        try:
            sensors = psutil.sensors_temperatures()  # type: ignore[attr-defined]
            for label, entries in (sensors or {}).items():
                if any(k in label.lower() for k in ("gpu", "nvidia", "amd", "radeon")):
                    temps = [e.current for e in entries if getattr(e, "current", None)]
                    if temps:
                        return float(max(temps))
        except Exception:
            pass
    return None


def _read_cpu_temp_c(hwinfo_cpu_temp_c: float | None = None) -> float | None:
    """Best-effort CPU package temperature in Celsius.
    1) HWiNFO shared memory — effectively the only reliable CPU-temp source on
       Windows (psutil has no coretemp there).
    2) psutil.sensors_temperatures() coretemp/k10temp (Linux mostly).
    Returns None if no CPU temp could be read.

    `hwinfo_cpu_temp_c` is the pre-fetched HWiNFO `cpu_temp_c` (from a single
    hwinfo.summary() in _gather_pulse). When supplied we skip the in-function
    HWiNFO read so the shared-memory block is parsed once per pulse."""
    try:
        ct = hwinfo_cpu_temp_c
        if ct is None:
            from audio import hwinfo
            ct = hwinfo.summary().get("cpu_temp_c")
        if ct is not None:
            return float(ct)
    except Exception:
        pass
    if _HAS_PSUTIL:
        try:
            sensors = psutil.sensors_temperatures()  # type: ignore[attr-defined]
            for label, entries in (sensors or {}).items():
                if any(k in label.lower()
                       for k in ("cpu", "core", "coretemp", "k10", "package", "tctl")):
                    temps = [e.current for e in entries if getattr(e, "current", None)]
                    if temps:
                        return float(max(temps))
        except Exception:
            pass
    return None


def _read_active_app_count() -> int:
    """Count distinct visible top-level windows with non-empty titles."""
    if not _HAS_GW:
        return 0
    try:
        wins = gw.getAllWindows()
        seen = set()
        for w in wins:
            try:
                if not getattr(w, "visible", True):
                    continue
                title = (getattr(w, "title", "") or "").strip()
                if not title:
                    continue
                # Some Windows utility windows have a fixed system title; skip a few obvious ones.
                if title in ("Default IME", "MSCTFIME UI", "Program Manager"):
                    continue
                seen.add(title)
            except Exception:
                continue
        return len(seen)
    except Exception:
        return 0


def _read_bambu_status() -> dict:
    """Pull bambu_monitor's state across sys.modules. Returns dict with
    gcode_state / percent / hours_into / minutes_remaining / failed flag."""
    mod = sys.modules.get("skill_bambu_monitor")
    if mod is None:
        return {}
    try:
        with getattr(mod, "_state_lock"):
            state = dict(getattr(mod, "_state"))
    except Exception:
        return {}
    if not state or state.get("last_update", 0.0) == 0.0:
        return {}
    gcode = (state.get("gcode_state") or "").upper()
    out = {"gcode_state": gcode}
    try:
        pct = state.get("mc_percent")
        if pct is not None:
            out["percent"] = int(pct)
    except Exception:
        pass
    try:
        rem = state.get("mc_remaining")
        if rem is not None:
            out["minutes_remaining"] = int(rem)
    except Exception:
        pass
    if gcode == "RUNNING":
        try:
            pct = int(state.get("mc_percent") or 0)
            rem = int(state.get("mc_remaining") or 0)
            if pct > 0 and rem >= 0:
                # estimate hours_elapsed from percent + remaining
                total_min = (rem * 100.0 / max(pct, 1)) if pct else 0
                elapsed_min = max(0, total_min - rem)
                out["hours_into"] = elapsed_min / 60.0
        except Exception:
            pass
    return out


def _read_credit_balance() -> float | None:
    """Read the last-known credit balance from credits_state.json. Stale
    entries (>24h) are ignored — the proactive pulse doesn't trigger the
    expensive vision-based recheck."""
    if not os.path.exists(_CREDITS_STATE):
        return None
    try:
        with open(_CREDITS_STATE, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return None
    bal = data.get("balance")
    ts  = data.get("checked_at", 0.0)
    if bal is None:
        return None
    if (time.time() - ts) > 24 * 3600:
        return None
    try:
        return float(bal)
    except (TypeError, ValueError):
        return None


# ─── pulse aggregation + formatting ──────────────────────────────────────

def _read_hwinfo_summary() -> dict:
    """Read the HWiNFO shared-memory block once, returning its summary dict
    (or {} if HWiNFO/its shared memory is unavailable). Calling this once per
    pulse lets _read_gpu_temp_c / _read_cpu_temp_c reuse the same parse instead
    of each opening + parsing the shared-memory block independently."""
    try:
        from audio import hwinfo
        return hwinfo.summary() or {}
    except Exception:
        return {}


def _gather_pulse() -> dict:
    cpu, ram_pct, ram_used = _read_cpu_ram()
    # Parse the HWiNFO shared-memory block ONCE and feed the CPU/GPU temps into
    # the readers below — they fall back to nvidia-smi / psutil as before but no
    # longer each re-open + re-parse the shared memory.
    hw = _read_hwinfo_summary()
    pulse = {
        "cpu_pct":         cpu,
        "ram_pct":         ram_pct,
        "ram_used_gb":     ram_used,
        "disk_free_gb":    _read_disk_free_gb(),
        "gpu_temp_c":      _read_gpu_temp_c(hwinfo_gpu_temp_c=hw.get("gpu_temp_c")),
        "cpu_temp_c":      _read_cpu_temp_c(hwinfo_cpu_temp_c=hw.get("cpu_temp_c")),
        "uptime_seconds":  _read_uptime_seconds(),
        "active_apps":     _read_active_app_count(),
        "bambu":           _read_bambu_status(),
        "credits_dollars": _read_credit_balance(),
    }
    down, up = _read_network_rates()
    pulse["net_down_kbps"] = down
    pulse["net_up_kbps"]   = up
    bat = _read_battery()
    if bat is not None:
        pulse["battery_pct"]    = bat[0]
        pulse["battery_plugged"] = bat[1]
    return pulse


def _fmt_uptime(seconds: float) -> str:
    if seconds <= 0:
        return "unknown"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    if days > 0:
        return f"{days} day{'s' if days != 1 else ''} {hours} hour{'s' if hours != 1 else ''}"
    if hours > 0:
        return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


def _fmt_bambu(b: dict) -> str:
    if not b:
        return ""
    gcode = b.get("gcode_state", "")
    if gcode == "RUNNING":
        hours = b.get("hours_into")
        pct = b.get("percent")
        if hours is not None and hours >= 1:
            if pct is not None:
                return f"Bambu printer is {hours:.0f} hours into its print, {pct} percent complete"
            return f"Bambu printer is {hours:.0f} hours into its print"
        if pct is not None:
            return f"Bambu printer is {pct} percent into a print"
        return "Bambu printer is running"
    if gcode == "FAILED":
        return "I'm afraid the Bambu print has failed"
    if gcode == "FINISH":
        return "Bambu printer finished its last print"
    return ""


def _format_report(pulse: dict, lead: str = "") -> str:
    """Render the JARVIS-cadence pulse sentence. If `lead` is non-empty, it
    replaces the default 'All systems nominal' opener (used when an
    abnormality forced the proactive announcement)."""
    cpu = pulse.get("cpu_pct", 0.0)
    ram = pulse.get("ram_pct", 0.0)
    gpu = pulse.get("gpu_temp_c")
    disk = pulse.get("disk_free_gb", 0.0)
    apps = pulse.get("active_apps", 0)
    bambu_line = _fmt_bambu(pulse.get("bambu") or {})
    credits = pulse.get("credits_dollars")
    bat = pulse.get("battery_pct")

    opener = lead or "All systems nominal, sir."

    cpu_temp = pulse.get("cpu_temp_c")
    parts: list[str] = []
    if cpu_temp is not None:
        parts.append(f"CPU {cpu:.0f} percent at {cpu_temp:.0f} degrees, "
                     f"memory {ram:.0f} percent")
    else:
        parts.append(f"CPU {cpu:.0f} percent, memory {ram:.0f} percent")
    if gpu is not None:
        verb = "running hot at" if gpu >= GPU_TEMP_ABNORMAL_C else "idling at"
        parts.append(f"GPU {verb} {gpu:.0f} degrees")
    if bat is not None and pulse.get("battery_plugged") is False:
        parts.append(f"battery at {bat:.0f} percent on battery power")
    if disk and disk < DISK_FREE_ABNORMAL_GB * 2:
        # only worth mentioning if it's getting tight
        parts.append(f"C drive has {disk:.0f} gigs free")
    if apps:
        parts.append(f"{apps} windows open")

    middle = ", ".join(parts) + "."
    tail_pieces = []
    if bambu_line:
        tail_pieces.append(bambu_line + ".")
    if credits is not None:
        tail_pieces.append(f"Anthropic credit balance: ${credits:.2f}.")
    tail = " " + " ".join(tail_pieces) if tail_pieces else ""

    return f"{opener} {middle}{tail} Anything further?".strip()


def _format_hud_strip(pulse: dict) -> str:
    """Compact one-line view of the non-redundant metrics — CPU/RAM are
    already on the rings, so we surface GPU/battery/uptime/apps/net here."""
    bits: list[str] = []
    cpu_temp = pulse.get("cpu_temp_c")
    if cpu_temp is not None:
        bits.append(f"CPU {cpu_temp:.0f}C")
    gpu = pulse.get("gpu_temp_c")
    if gpu is not None:
        bits.append(f"GPU {gpu:.0f}C")
    bat = pulse.get("battery_pct")
    if bat is not None:
        suffix = "+" if pulse.get("battery_plugged") else ""
        bits.append(f"BAT {bat:.0f}%{suffix}")
    up = pulse.get("uptime_seconds", 0.0)
    if up > 0:
        days = int(up // 86400)
        hours = int((up % 86400) // 3600)
        if days > 0:
            bits.append(f"UP {days}d{hours:02d}h")
        else:
            mins = int((up % 3600) // 60)
            bits.append(f"UP {hours}h{mins:02d}m")
    apps = pulse.get("active_apps")
    if apps:
        bits.append(f"APPS {apps}")
    down = pulse.get("net_down_kbps", 0.0)
    up_kbps = pulse.get("net_up_kbps", 0.0)
    if down >= 50 or up_kbps >= 50:
        if down >= 1024 or up_kbps >= 1024:
            bits.append(f"NET {down/1024:.1f}/{up_kbps/1024:.1f} MB/s")
        else:
            bits.append(f"NET {down:.0f}/{up_kbps:.0f} kB/s")
    return "  ·  ".join(bits)


# ─── abnormality detection ───────────────────────────────────────────────

def _abnormal_reasons(pulse: dict) -> list[tuple[str, str]]:
    """Return [(key, lead_sentence), ...] for each abnormal reading.
    The key is used for per-reason cooldown so a sustained high-CPU doesn't
    nag every 15 min."""
    reasons: list[tuple[str, str]] = []
    if pulse.get("cpu_pct", 0.0) >= CPU_ABNORMAL_PCT:
        reasons.append(("cpu", f"Slight problem, sir — CPU at {pulse['cpu_pct']:.0f} percent."))
    if pulse.get("ram_pct", 0.0) >= RAM_ABNORMAL_PCT:
        reasons.append(("ram", f"Memory at {pulse['ram_pct']:.0f} percent, sir — getting tight."))
    if 0 < pulse.get("disk_free_gb", 999.0) < DISK_FREE_ABNORMAL_GB:
        reasons.append(("disk", f"I'm afraid the C drive is down to {pulse['disk_free_gb']:.0f} gigs free."))
    gpu = pulse.get("gpu_temp_c")
    if gpu is not None and gpu >= GPU_TEMP_ABNORMAL_C:
        reasons.append(("gpu", f"GPU running rather hot, sir — {gpu:.0f} degrees."))
    bat = pulse.get("battery_pct")
    if (bat is not None
            and bat < BATTERY_LOW_PCT
            and pulse.get("battery_plugged") is False):
        reasons.append(("battery", f"Battery at {bat:.0f} percent, sir — you may want to plug in."))
    bambu = pulse.get("bambu") or {}
    if (bambu.get("gcode_state") or "").upper() == "FAILED":
        reasons.append(("bambu_fail", "I'm afraid the Bambu print has failed, sir."))
    credits = pulse.get("credits_dollars")
    if credits is not None and credits < CREDITS_LOW_DOLLARS:
        reasons.append(("credits", f"Heads up, sir — Anthropic credit balance is at only ${credits:.2f}."))
    down = pulse.get("net_down_kbps", 0.0)
    up   = pulse.get("net_up_kbps", 0.0)
    if down >= NETWORK_HOT_KBPS or up >= NETWORK_HOT_KBPS:
        peak = max(down, up) / 1024
        reasons.append(("network", f"Notable network activity, sir — sustained {peak:.0f} megabytes per second."))
    return reasons


# ─── background threads ──────────────────────────────────────────────────

def _hud_publish_loop() -> None:
    while True:
        try:
            pulse = _gather_pulse()
            # Cache the gathered pulse so the voice action can format from it
            # instead of re-running the blocking collect on the main thread.
            global _pulse_cache
            with _pulse_cache_lock:
                _pulse_cache = {"pulse": pulse, "ts": time.time()}
            strip = _format_hud_strip(pulse)
            if strip:
                _publish_hud_strip(strip)
        except Exception as e:
            print(f"  [pulse] hud publish error: {e}")
        time.sleep(PULSE_HUD_REFRESH_SECONDS)


def _proactive_loop() -> None:
    time.sleep(INITIAL_PROACTIVE_DELAY_SECONDS)
    while True:
        try:
            pulse = _gather_pulse()
            reasons = _abnormal_reasons(pulse)
            now = time.time()
            with _alert_lock:
                fresh_reasons = [
                    (k, lead) for (k, lead) in reasons
                    if (now - _last_abnormal_alert.get(k, 0.0)) > PROACTIVE_COOLDOWN_SECONDS
                ]
            if fresh_reasons:
                # Lead the report with the first abnormal item, then drop the
                # full pulse sentence behind it so the user gets context.
                key, lead = fresh_reasons[0]
                message = _format_report(pulse, lead=lead)
                _enqueue_speech(message)
                with _alert_lock:
                    for k, _ in fresh_reasons:
                        _last_abnormal_alert[k] = now
                print(f"  [pulse] proactive fired (reasons: {[k for k,_ in fresh_reasons]})")
        except Exception:
            logging.exception("  [pulse] proactive loop error")
        time.sleep(PULSE_PROACTIVE_INTERVAL_SECONDS)


# ─── action registration ─────────────────────────────────────────────────

def register(actions):
    def system_pulse(_: str = "") -> str:
        try:
            pulse = None
            with _pulse_cache_lock:
                cached = _pulse_cache
            if (cached
                    and (time.time() - cached.get("ts", 0.0))
                    <= PULSE_CACHE_MAX_AGE_SECONDS):
                # Serve the loop's recent pulse instead of re-running
                # nvidia-smi + a 0.6 s net sample on the main voice thread.
                pulse = cached.get("pulse")
            if not pulse:
                # Cache empty/stale (first call before the loop has run) —
                # fall back to a live gather.
                pulse = _gather_pulse()
            return _format_report(pulse)
        except Exception as e:
            return f"system pulse failed: {e}"

    actions["system_pulse"] = system_pulse
    # Common aliases the LLM may emit for "JARVIS, status report"
    actions["status_report"] = system_pulse

    if not _HAS_PSUTIL:
        print("  [pulse] psutil missing — actions registered but threads disabled. "
              "pip install psutil to enable proactive + HUD widget.")
        return

    # Guard against duplicate loops on skill reload (load_skills re-execs the
    # module → fresh globals, so only an OS-thread name check survives).
    if not any(t.name == "pulse-hud" and t.is_alive()
               for t in threading.enumerate()):
        threading.Thread(target=_hud_publish_loop, daemon=True,
                         name="pulse-hud").start()
    if not any(t.name == "pulse-proactive" and t.is_alive()
               for t in threading.enumerate()):
        threading.Thread(target=_proactive_loop, daemon=True,
                         name="pulse-proactive").start()
    print(
        f"  [pulse] proactive every {PULSE_PROACTIVE_INTERVAL_SECONDS // 60} min; "
        f"HUD strip refresh every {PULSE_HUD_REFRESH_SECONDS}s"
    )
