"""
Morning chain controller for JARVIS.

Owns the SINGLE daemon thread that watches bobert_companion._last_wake_date
for the day's first wake event inside [6 AM, 12 PM) local, then picks
exactly ONE of the three morning skills (morning_arrival, morning_handoff,
morning_briefing) to fire — instead of letting all three fire on the same
wake event and queue overlapping speech.

Why this exists:
  morning_arrival, morning_handoff, and morning_briefing each used to spawn
  their own _watch_for_first_wake daemon. All three watched the same
  bobert_companion._last_wake_date and would fire independently on the same
  wake, producing three back-to-back briefings (the cold-open + the chained
  handoff + the lighter recap). This controller consolidates wake-watching
  into one thread and selects exactly one morning skill per wake event.

Selection precedence (first match wins):
  1. JSON config at <project>/morning_chain_config.json — schema:
       {
         "by_weekday": {"monday": "handoff", "saturday": "arrival"},
         "default":   "handoff"
       }
     by_weekday keys are lowercased English day names. Values are one of
     "arrival" / "handoff" / "briefing" (with or without the "morning_"
     prefix).
  2. Environment variable DEFAULT_MORNING_SKILL — same value set.
  3. Time-of-day fallback inside the morning window:
       06–08 → arrival   (lightest cold-open)
       08–10 → handoff   (full chained briefing + workspace setup)
       10–12 → briefing  (lighter recap)

Coordination contract with the three skills:
  Each skill exposes a _fire_from_chain(reason) entry point. The chain calls
  it synchronously; the skill is responsible for its own pre-fire delay and
  same-day suppression check (the original three watchers' TOCTOU pattern is
  preserved verbatim inside each skill). The skills also keep their public
  manual-trigger actions (e.g. "morning arrival"), which always run via
  force=True regardless of the chain.

Failure modes:
  - bobert_companion unavailable → the chain silently disables; the three
    skills remain manually callable.
  - chosen skill module missing or _fire_from_chain attr missing → the chain
    logs and skips for the day. Manual triggers still work.
  - Per-tick exceptions are caught so transient errors can't kill the
    thread.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_FILE = os.path.join(_PROJECT_DIR, "morning_chain_config.json")

# Morning window — first wake whose local hour falls in [start, end) drives
# the chain. Matches the union of the three skills' original windows so the
# chain covers every previous trigger condition.
CHAIN_START_HOUR = 6
CHAIN_END_HOUR   = 12

# Background poll interval for the wake-event watcher. Matches the value the
# three original watchers used so the felt latency is unchanged.
WATCH_POLL_SECONDS = 5.0

# Time-of-day fallback boundaries when no config / env preference is set.
TOD_ARRIVAL_UNTIL_HOUR = 8
TOD_HANDOFF_UNTIL_HOUR = 10

SKILL_NAMES = ("arrival", "handoff", "briefing")

_SKILL_MODULE_MAP = {
    "arrival":  "morning_arrival",
    "handoff":  "morning_handoff",
    "briefing": "morning_briefing",
}

# Each skill's on-disk same-day flag. The chain reads these directly rather
# than importing the skill at decision time so a broken skill module can't
# block the chain from picking a different skill.
_SKILL_STATE_FILES = {
    "arrival":  ("morning_arrival_state.json", "json"),
    "handoff":  ("morning_handoff_state.json", "json"),
    "briefing": (".morning_briefing_last",     "text"),
}

_SKILL_CHAIN_ENTRY = "_fire_from_chain"


# ─── configuration ───────────────────────────────────────────────────────

def _load_chain_config() -> dict:
    if not os.path.exists(_CONFIG_FILE):
        return {}
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"  [morning-chain] config read failed: {e}")
        return {}


def _normalize_skill(name) -> str | None:
    """Map a user-supplied skill name to one of SKILL_NAMES or None."""
    if not isinstance(name, str):
        return None
    n = name.strip().lower()
    if n.startswith("morning_"):
        n = n[len("morning_"):]
    return n if n in SKILL_NAMES else None


def _choose_skill_for_today(now_hour: int) -> str:
    """Decide which of the three morning skills should fire today.

    Precedence: config by_weekday → config default → DEFAULT_MORNING_SKILL
    env var → time-of-day fallback inside the morning window."""
    cfg = _load_chain_config()

    by_weekday = cfg.get("by_weekday")
    if isinstance(by_weekday, dict):
        weekday = time.strftime("%A").lower()
        choice = _normalize_skill(by_weekday.get(weekday))
        if choice:
            return choice

    cfg_default = _normalize_skill(cfg.get("default"))
    if cfg_default:
        return cfg_default

    env_default = _normalize_skill(os.environ.get("DEFAULT_MORNING_SKILL"))
    if env_default:
        return env_default

    if now_hour < TOD_ARRIVAL_UNTIL_HOUR:
        return "arrival"
    if now_hour < TOD_HANDOFF_UNTIL_HOUR:
        return "handoff"
    return "briefing"


# ─── state-file reads ────────────────────────────────────────────────────

def _skill_already_fired_today(short_name: str) -> bool:
    """Check a skill's on-disk state file for today's date. Read directly so
    a broken skill module can't block a chain decision."""
    info = _SKILL_STATE_FILES.get(short_name)
    if not info:
        return False
    fname, fmt = info
    path = os.path.join(_PROJECT_DIR, fname)
    if not os.path.exists(path):
        return False
    today = time.strftime("%Y-%m-%d")
    try:
        if fmt == "text":
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip() == today
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return isinstance(data, dict) and data.get("last_fired_date") == today
    except Exception:
        return False


def _arrival_v2_fired_today() -> bool:
    """True if skills/morning_arrival_v2's presence-watcher already briefed
    today. v2 is NOT one of the chain's SKILL_NAMES (it runs its own watcher),
    so without this the chain can't see a v2 fire and the two double-brief. This
    is the mirror of v2's _chain_morning_briefing_fired_today(): v2 stands down
    when the chain fires; the chain stands down when v2 fired. Resolved the same
    way v2 resolves the chain (the loader registers skills as 'skill_<name>'),
    and defensive -- returns False on any failure so it can never block a chain
    decision."""
    try:
        import sys
        import importlib as _il
        v2 = sys.modules.get("skill_morning_arrival_v2")
        if v2 is None:
            try:
                v2 = _il.import_module("skills.morning_arrival_v2")
            except Exception:
                v2 = _il.import_module("morning_arrival_v2")
        return bool(v2._already_fired_today())
    except Exception:
        return False


# ─── skill dispatch ──────────────────────────────────────────────────────

def _import_skill(short_name: str):
    mod_name = _SKILL_MODULE_MAP[short_name]
    try:
        return importlib.import_module(f"skills.{mod_name}")
    except Exception:
        pass
    try:
        skills_dir = os.path.dirname(os.path.abspath(__file__))
        if skills_dir not in sys.path:
            sys.path.insert(0, skills_dir)
        return importlib.import_module(mod_name)
    except Exception as e:
        print(f"  [morning-chain] can't import {mod_name}: {e}")
        return None


def _invoke_skill(short_name: str, reason: str) -> bool:
    mod = _import_skill(short_name)
    if mod is None:
        return False
    entry = getattr(mod, _SKILL_CHAIN_ENTRY, None)
    if not callable(entry):
        print(f"  [morning-chain] {_SKILL_MODULE_MAP[short_name]} has no "
              f"{_SKILL_CHAIN_ENTRY} — skipping (manual trigger still works)")
        return False
    try:
        entry(reason)
        return True
    except Exception as e:
        print(f"  [morning-chain] {_SKILL_MODULE_MAP[short_name]} fire failed: {e}")
        return False


# ─── wake-event watcher ──────────────────────────────────────────────────

def _watch_for_first_wake() -> None:
    """Poll bobert_companion._last_wake_date and, on the day's first wake
    inside the morning window, dispatch the chosen skill exactly once."""
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception as e:
        print(f"  [morning-chain] disabled — can't import bobert_companion: {e}")
        return

    print(f"  [morning-chain] active (window {CHAIN_START_HOUR}–{CHAIN_END_HOUR}, "
          f"poll {WATCH_POLL_SECONDS:.0f}s)")

    # Track the date for which we've already dispatched so a slow skill (its
    # _fire_from_chain sleeps DELAY seconds inline) doesn't get re-entered on
    # the next tick before it finishes.
    dispatched_for_date: str | None = None

    while True:
        try:
            wake_date = None
            try:
                # READ SIDE of the mutable-reference-sharing contract defined
                # at bobert_companion.py:_last_wake_date. The list wrapper is
                # the entire reason this cross-thread read is safe without a
                # lock: writers do `_last_wake_date[0] = today` in-place, the
                # list identity never changes, and CPython's GIL makes the
                # index read/write atomic. We read [0] every poll tick to pick
                # up wake events fired by either context_aware_greeting() (the
                # wake-word path) or the tray force_wake handler.
                #
                # If a future maintainer "simplifies" bc._last_wake_date to a
                # plain string, this access (`[0]`) will raise TypeError on
                # None or IndexError on a str — but only at runtime in this
                # background thread, which would be silently swallowed by the
                # except below and the morning chain would just never fire.
                # If you're touching bc._last_wake_date, read the WHY comment
                # there first.
                wake_date = bc._last_wake_date[0]
            except Exception:
                pass

            today = time.strftime("%Y-%m-%d")
            hour  = time.localtime().tm_hour

            if (wake_date == today
                    and CHAIN_START_HOUR <= hour < CHAIN_END_HOUR
                    and dispatched_for_date != today):
                chosen = _choose_skill_for_today(hour)
                if _skill_already_fired_today(chosen) or _arrival_v2_fired_today():
                    # Manual trigger, a prior chain dispatch (this JARVIS process
                    # can't remember across a restart), OR morning_arrival_v2's
                    # presence-watcher already covered today -- treat the day as
                    # handled and stop racing. The v2 check mirrors v2's own
                    # chain-suppression so the two can never double-brief.
                    print(f"  [morning-chain] today already briefed "
                          f"(pick was '{chosen}') -- chain idle for {today}")
                    dispatched_for_date = today
                else:
                    print(f"  [morning-chain] wake @ {hour:02d}:xx, "
                          f"weekday={time.strftime('%A')} — dispatching '{chosen}'")
                    if _invoke_skill(
                            chosen,
                            f"morning chain pick (weekday={time.strftime('%A')}, hour={hour})"):
                        dispatched_for_date = today
        except Exception as e:
            print(f"  [morning-chain] tick error: {e}")
        time.sleep(WATCH_POLL_SECONDS)


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    def morning_chain_pick(_: str = "") -> str:
        """Debug helper — report which morning skill the chain would pick
        right now and whether it has already fired today."""
        hour = time.localtime().tm_hour
        chosen = _choose_skill_for_today(hour)
        weekday = time.strftime("%A")
        fired_today = {s: _skill_already_fired_today(s) for s in SKILL_NAMES}
        return (f"morning chain pick for {weekday} {hour:02d}:xx → {chosen} "
                f"(fired today: {fired_today})")

    actions["morning_chain_pick"] = morning_chain_pick

    t = threading.Thread(target=_watch_for_first_wake, daemon=True)
    t.start()
