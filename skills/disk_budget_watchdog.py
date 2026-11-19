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

The alert stamps are PERSISTED (2026-11-17, data/budget_watchdog_state.json).
Live 11-16/11-17 "Heads up, sir - Claude credit balance is down to 7.10
dollars" was said or recapped 8 times in 38 hours with the balance unchanged:
the 4 h cooldown lived in memory, so every restart (6 in that window)
announced it again 3 minutes after boot, and while he was away a copy queued
every 4 h. Now an unchanged low balance is news once: it is said again only
when it falls below the next step (CREDITS_ALERT_STEPS) or, as a reminder,
CREDITS_REMIND_SECONDS after the last time; a top-up above the threshold
re-arms it. Queued copies carry one supersede / dedupe family
(CREDITS_LOW_FAMILY, shared with skills/credits_monitor), so a held copy is
replaced, never stacked.

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
except ImportError:  # pragma: no cover - psutil is a CI dep; degradation branch only taken on a bare host
    _HAS_PSUTIL = False

# ─── thresholds ──────────────────────────────────────────────────────────
DISK_ALERT_GB             = 10.0
CREDITS_ALERT_DOLLARS     = 10.0
POLL_INTERVAL_SECONDS     = 300        # 5 minutes
INITIAL_DELAY_SECONDS     = 180        # let JARVIS finish booting first
ALERT_COOLDOWN_SECONDS    = 4 * 3600   # 4 hours between repeat disk alerts
# The same low credit balance is said again at most this often (a reminder);
# a drop below the next step is news at once.
CREDITS_REMIND_SECONDS    = 24 * 3600
CREDITS_ALERT_STEPS       = (5.0, 2.0, 0.0)
# The supersede / dedupe family of every queued "credits are low" line (this
# skill and skills/credits_monitor): one held copy, never a stack.
CREDITS_LOW_FAMILY        = "credits_low"
CREDITS_STATE_STALE_SECONDS = 24 * 3600  # ignore credits snapshot older than this
DISK_PATH                 = "C:\\"
# ─────────────────────────────────────────────────────────────────────────

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")
_CREDITS_STATE = os.path.join(_PROJECT_DIR, "credits_state.json")

if _PROJECT_DIR not in sys.path:  # pragma: no cover - import-time sys.path guard; root already on path under the test harness
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

_speech_lock = threading.Lock()
_state_lock = threading.Lock()
_last_disk_alert_at    = [0.0]
_last_credits_alert_at = [0.0]
# The balance the last credits alert named (None = no alert since the last
# top-up): an unchanged low balance is not news.
_last_credits_alert_balance: list = [None]
_STATE_NAME = "budget_watchdog_state.json"


def _state_path() -> str:
    """data/budget_watchdog_state.json, resolved at call time (staging- and
    JARVIS_DATA_DIR-aware, core/paths)."""
    from core.paths import data_file
    return data_file(_STATE_NAME)


def _load_alert_state() -> None:
    """Restore the persisted alert stamps (keeping whichever is newer) and the
    last announced balance. A missing / corrupt file, or a stamp from the
    future, leaves the cells alone. Never raises."""
    try:
        path = _state_path()
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        now = time.time()
        with _state_lock:
            for key, cell in (("last_disk_alert_at", _last_disk_alert_at),
                              ("last_credits_alert_at", _last_credits_alert_at)):
                v = data.get(key)
                if (isinstance(v, (int, float)) and not isinstance(v, bool)
                        and 0 < v <= now + 60):
                    cell[0] = max(cell[0], float(v))
            bal = data.get("last_credits_alert_balance")
            if (isinstance(bal, (int, float)) and not isinstance(bal, bool)
                    and _last_credits_alert_balance[0] is None):
                _last_credits_alert_balance[0] = float(bal)
    except Exception:
        pass


def _save_alert_state() -> None:
    """Persist the alert stamps and the last announced balance. Never raises."""
    try:
        with _state_lock:
            payload = {
                "last_disk_alert_at": _last_disk_alert_at[0],
                "last_credits_alert_at": _last_credits_alert_at[0],
                "last_credits_alert_balance": _last_credits_alert_balance[0],
            }
        _atomic_write_json(_state_path(), payload)
    except Exception as e:
        print(f"  [watchdog] could not save the alert state: {e}")


def _credits_step(balance: float) -> int:
    """How many alert steps ``balance`` is below: 1 under
    CREDITS_ALERT_DOLLARS, one more under each of CREDITS_ALERT_STEPS."""
    marks = (CREDITS_ALERT_DOLLARS,) + tuple(CREDITS_ALERT_STEPS)
    return sum(1 for m in marks if balance < m)


def _enqueue_speech(message: str, family: "str | None" = None) -> None:
    """Route a proactive announcement through bobert_companion's public
    proactive_announce() API when available, falling back to a direct atomic
    write against pending_speech.json. Mirrors the pattern in
    skills/credits_monitor.py and skills/system_monitor.py. ``family``: the
    supersede / dedupe tag, so a newer copy replaces a held one."""
    try:
        import importlib
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            if family:
                announcer(message, source="disk_budget_watchdog",
                          supersede=family, dedupe_key=family)
            else:
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
        f"You may want to clear some room before the overnight run continues.",
        family="disk_low",
    )
    _last_disk_alert_at[0] = now
    _save_alert_state()


def _check_credits(now: float) -> None:
    balance, age = _read_credits_snapshot()
    if balance is None:
        return
    if age is not None and age > CREDITS_STATE_STALE_SECONDS:
        # Snapshot too old to trust — credits_monitor will refresh it.
        return
    if balance >= CREDITS_ALERT_DOLLARS:
        if _last_credits_alert_balance[0] is not None:
            # Topped up: the next drop below the threshold is news again.
            _last_credits_alert_balance[0] = None
            _save_alert_state()
        return
    last_bal = _last_credits_alert_balance[0]
    lower_step = (last_bal is None
                  or _credits_step(balance) > _credits_step(float(last_bal)))
    since = now - _last_credits_alert_at[0]
    if not lower_step and since < CREDITS_REMIND_SECONDS:
        return
    _enqueue_speech(
        f"Heads up, sir — Claude credit balance is down to "
        f"{balance:.2f} dollars. Top up before the autonomous run drains it.",
        family=CREDITS_LOW_FAMILY,
    )
    print(f"  [watchdog] credits alert queued (${balance:.2f}; "
          + ("a new low step" if lower_step else
             f"a reminder, the last one {since / 3600:.0f} h ago") + ")")
    _last_credits_alert_at[0] = now
    # Only a LOWER reading moves the remembered balance: one that bobs back
    # over a step (4.99 / 5.01) is not news twice.
    if last_bal is None or balance < float(last_bal):
        _last_credits_alert_balance[0] = balance
    _save_alert_state()


def _monitor_loop() -> None:  # pragma: no cover - non-terminating background daemon (sleeps INITIAL_DELAY then polls forever); its body delegates to _check_disk/_check_credits, which are unit-tested directly
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

    # Restore the cooldowns BEFORE the monitor can alert: a restart must not
    # announce an unchanged balance again.
    _load_alert_state()
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
