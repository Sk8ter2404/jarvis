"""
Disk + budget watchdog skill for JARVIS.

Long-running guardrail for autonomous overnight runs that could silently bleed
two finite resources: free space on C: and Anthropic API credits. Polls every
POLL_INTERVAL_SECONDS (default 5 min) and queues a spoken alert when either
metric drops below threshold. Per-metric cooldowns prevent spam.

Actions:
  check_budget   — manual audit. Returns current C: free GB and the last-known
                   credit balance (with staleness flag), JARVIS-style.

Background monitor:
  - C: free space via psutil.disk_usage(). Alert when below DISK_ALERT_GB.
  - Credit balance via credits_state.json (written by skills/credits_monitor).
    Alert when below CREDITS_ALERT_DOLLARS. Snapshot >24h old is treated as
    unknown rather than triggering an alert — credits_monitor will refresh it
    on its own hourly cadence.

Reads credits_state.json instead of calling the vision-driven check itself, so
the watchdog stays cheap (no Chrome spawn per poll) and there's a single
source of truth for the balance.
"""
import json
import logging
import os
import sys
import threading
import time

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ─── thresholds ──────────────────────────────────────────────────────────
DISK_ALERT_GB             = 10.0
CREDITS_ALERT_DOLLARS     = 10.0
POLL_INTERVAL_SECONDS     = 300        # 5 minutes
INITIAL_DELAY_SECONDS     = 180        # let JARVIS finish booting first
ALERT_COOLDOWN_SECONDS    = 4 * 3600   # 4 hours between repeat alerts per metric
CREDITS_STATE_STALE_SECONDS = 24 * 3600  # ignore credits snapshot older than this
DISK_PATH                 = "C:\\"
# ─────────────────────────────────────────────────────────────────────────

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")
_CREDITS_STATE = os.path.join(_PROJECT_DIR, "credits_state.json")

if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

_speech_lock = threading.Lock()
_last_disk_alert_at    = [0.0]
_last_credits_alert_at = [0.0]


def _enqueue_speech(message: str) -> None:
    """Route a proactive announcement through bobert_companion's public
    proactive_announce() API when available, falling back to a direct atomic
    write against pending_speech.json. Mirrors the pattern in
    skills/credits_monitor.py and skills/system_monitor.py."""
    try:
        import importlib
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="disk_budget_watchdog")
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
            print(f"  [watchdog] speech-queue write failed ({e}); alert: {message}")


def _disk_free_gb() -> float | None:
    if not _HAS_PSUTIL:
        return None
    try:
        return psutil.disk_usage(DISK_PATH).free / (1024 ** 3)
    except Exception:
        return None


def _read_credits_snapshot() -> tuple[float | None, float | None]:
    """Return (balance_dollars, age_seconds). Either may be None if the
    state file is missing, malformed, or has no usable balance."""
    if not os.path.exists(_CREDITS_STATE):
        return None, None
    try:
        with open(_CREDITS_STATE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None
    balance = data.get("balance")
    checked_at = data.get("checked_at")
    age = (time.time() - checked_at) if isinstance(checked_at, (int, float)) else None
    if not isinstance(balance, (int, float)):
        return None, age
    return float(balance), age


def _check_disk(now: float) -> None:
    free_gb = _disk_free_gb()
    if free_gb is None:
        return
    if free_gb >= DISK_ALERT_GB:
        return
    if (now - _last_disk_alert_at[0]) <= ALERT_COOLDOWN_SECONDS:
        return
    _enqueue_speech(
        f"Sir, C drive is down to {free_gb:.1f} gigabytes free. "
        f"You may want to clear some room before the overnight run continues."
    )
    _last_disk_alert_at[0] = now


def _check_credits(now: float) -> None:
    balance, age = _read_credits_snapshot()
    if balance is None:
        return
    if age is not None and age > CREDITS_STATE_STALE_SECONDS:
        # Snapshot too old to trust — credits_monitor will refresh it.
        return
    if balance >= CREDITS_ALERT_DOLLARS:
        return
    if (now - _last_credits_alert_at[0]) <= ALERT_COOLDOWN_SECONDS:
        return
    _enqueue_speech(
        f"Heads up, sir — Claude credit balance is down to "
        f"{balance:.2f} dollars. Top up before the autonomous run drains it."
    )
    _last_credits_alert_at[0] = now


def _monitor_loop() -> None:
    time.sleep(INITIAL_DELAY_SECONDS)
    while True:
        try:
            now = time.time()
            _check_disk(now)
            _check_credits(now)
        except Exception:
            logging.exception("[watchdog] monitor loop iteration failed")
        try:
            time.sleep(POLL_INTERVAL_SECONDS)
        except Exception:
            logging.exception("[watchdog] monitor loop sleep failed")


def register(actions):
    def check_budget(_: str = "") -> str:
        parts = []
        free_gb = _disk_free_gb()
        if free_gb is None:
            parts.append("disk reading unavailable (psutil missing)")
        else:
            parts.append(f"C drive has {free_gb:.1f} gigabytes free")

        balance, age = _read_credits_snapshot()
        if balance is None:
            parts.append("no Claude credit balance recorded yet")
        else:
            stale = age is not None and age > CREDITS_STATE_STALE_SECONDS
            stale_note = " (snapshot is stale)" if stale else ""
            parts.append(
                f"Claude credit balance ${balance:.2f}{stale_note}"
            )
        return "Budget watchdog: " + "; ".join(parts) + "."

    actions["check_budget"] = check_budget

    t = threading.Thread(target=_monitor_loop, daemon=True)
    t.start()
    print(
        f"  [watchdog] disk+budget monitor active — polls every "
        f"{POLL_INTERVAL_SECONDS}s, alert below {DISK_ALERT_GB:.0f} GB / "
        f"${CREDITS_ALERT_DOLLARS:.2f}"
    )
    if not _HAS_PSUTIL:
        print("  [watchdog] psutil missing — disk check disabled, "
              "credit-balance check still active")
