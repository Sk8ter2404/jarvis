"""
Banter engine — occasional dry zingers when JARVIS notices a tell.

Watches the command history (pattern_memory's voice_commands.jsonl) plus
ambient signals (window count, JARVIS's own music-action timestamp) for
behavioural tells, and once every 30+ minutes injects ONE in-character
zinger via the pending_speech queue.

Tells detected:
  • repeat_question     — same (or near-same) utterance ≥ 2× in the last 10 min
  • repeat_open         — same target opened ≥ 5× today
  • tab_clutter         — > 40 chrome.exe processes (~one per tab) OR > 40
                          total visible top-level windows
  • music_while_music   — user asks to play music while JARVIS already
                          started something in the last 30 min

Gating:
  • Cooldown:     ≥ BANTER_COOLDOWN_MINUTES between any two zingers
                  (default 30, persisted across restarts)
  • Probability:  even with a match, fire with FIRE_PROBABILITY (0.5)
                  so the engine feels rare, not punctual
  • Sleep/standby: silent
  • Calls:         silent (Teams / Zoom / Meet / Webex / Discord call)
  • Same tell:    per-tell cooldown so the same observation doesn't recur
                  every 30 min — uses BANTER_PER_TELL_COOLDOWN_MINUTES
                  (default 180 = 3 h)

Swearing is allowed per user preference but used sparingly — the bank
prioritises dry observation over profanity.

Action registered:
  banter_status — short status report on the engine (last fire, recent tells)

Config knobs (bobert_companion.py, optional):
  BANTER_ENABLED                       (bool, default True)
  BANTER_COOLDOWN_MINUTES              (int,  default 30)
  BANTER_PER_TELL_COOLDOWN_MINUTES     (int,  default 180)
"""

from __future__ import annotations

import datetime
import importlib
import json
import logging
import os
import random
import sys
import tempfile
import threading
import time

from core.atomic_io import _atomic_write_json

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")
_STATE_FILE   = os.path.join(_PROJECT_DIR, "banter_state.json")

POLL_INTERVAL_SECONDS = 90.0     # check tells every 90 s
INITIAL_DELAY_SECONDS = 180      # let JARVIS settle before the first scan
FIRE_PROBABILITY      = 0.5      # post-match dice roll

REPEAT_QUESTION_WINDOW = 10 * 60     # "same question twice in 10 min"
REPEAT_OPEN_THRESHOLD  = 5           # "same URL opened 5+ times today"
TAB_CLUTTER_THRESHOLD  = 40          # chrome procs OR total windows
MUSIC_REPEAT_WINDOW    = 30 * 60     # "music while music already playing"

# Window-title fragments indicating the user is currently on a call. Mirror
# anticipation_engine's list so behaviour is consistent across proactive skills.
CALL_WINDOW_HINTS = (
    "meeting now",
    "meeting in ",
    " | microsoft teams meeting",
    "microsoft teams meeting |",
    "zoom meeting",
    "zoom - meeting",
    "webex meetings",
    "google meet -",
    "meet -",
    "discord call",
)

# Zinger banks. Each tell has 3-5 variants so repeated firings stay fresh.
_ZINGER_BANK: dict[str, list[str]] = {
    "repeat_question": [
        "That's the {n}th time you've asked me that today, sir.",
        # Use {minutes} not {n}: n is the count, minutes is the gap.
        "I have an answer ready, sir — it hasn't changed since you last asked, {minutes} minutes ago.",
        "Asking again won't alter the result, sir.",
        "Sir, my answer is unlikely to improve through repetition.",
    ],
    "repeat_open": [
        "Opening {target} again, sir. That's {n} times today. Productive, I'm sure.",
        "{target}, for the {n}th time today. I assume there's a strategy at play.",
        "{target} again, sir? I'm beginning to feel like a doorman.",
    ],
    "tab_clutter": [
        "{n} tabs, sir. I'm sure each of them is essential.",
        "{n} tabs open, sir. The browser is weeping quietly.",
        "Sir, {n} tabs. Bold of you to call this organised.",
        "{n} tabs. I'd offer to close some, but I doubt you'd notice.",
    ],
    "tab_clutter_windows": [
        "{n} open windows, sir. Either you're orchestrating something brilliant, or the desktop has rather got away from you.",
        "{n} windows on the go, sir. I admire the ambition.",
    ],
    "music_while_music": [
        "Music is already playing, sir. Shall I just leave it on a loop?",
        "Sir, the music hasn't stopped. Are we layering things on purpose?",
        "I am, in fact, already playing music for you, sir.",
        "Another track on top of the current one, sir? Bold choice.",
    ],
}

_state_lock  = threading.Lock()
_speech_lock = threading.Lock()


# ─── speech queue ────────────────────────────────────────────────────────

def _enqueue_speech(message: str) -> None:
    """Append a spoken alert to pending_speech.json for the main loop.

    Routes through bobert_companion.proactive_announce() so this skill shares
    one write path with every other pending_speech.json co-writer (timer,
    wellness, night_owl_mode, dnd_focus_mode, …) and they don't race each
    other. Falls back to a local atomic write only when the parent module
    isn't loaded yet (import-time registration / unit tests) or the announcer
    call fails — so a broken parent import can't silence a zinger."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="banter")
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
            print(f"  [banter] speech-queue write failed ({e}); line: {message}")


# ─── config + persistent state ───────────────────────────────────────────

def _read_config() -> dict:
    try:
        bc = importlib.import_module("bobert_companion")
    except Exception:
        bc = None
    return {
        "enabled":      bool(getattr(bc, "BANTER_ENABLED",                   True)) if bc else True,
        "cooldown":     int (getattr(bc, "BANTER_COOLDOWN_MINUTES",          30))   if bc else 30,
        "per_tell_cd":  int (getattr(bc, "BANTER_PER_TELL_COOLDOWN_MINUTES", 180))  if bc else 180,
    }


def _load_state() -> dict:
    if not os.path.exists(_STATE_FILE):
        return {}
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    with _state_lock:
        try:
            fd, tmp = tempfile.mkstemp(dir=_PROJECT_DIR, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                os.replace(tmp, _STATE_FILE)
            except Exception:
                try: os.unlink(tmp)
                except Exception: pass
                raise
        except Exception as e:
            print(f"  [banter] could not persist state: {e}")


# ─── environment inspection ──────────────────────────────────────────────

def _all_window_titles() -> list[str]:
    try:
        import pygetwindow as gw   # type: ignore
    except Exception:
        return []
    out: list[str] = []
    try:
        for w in gw.getAllWindows():
            t = getattr(w, "title", "") or ""
            if t.strip():
                out.append(t)
    except Exception:
        pass
    return out


def _is_in_call() -> bool:
    titles = _all_window_titles()
    if not titles:
        return False
    lowered = [t.lower() for t in titles]
    for hint in CALL_WINDOW_HINTS:
        for t in lowered:
            if hint in t:
                return True
    return False


def _is_sleep_or_standby() -> bool:
    bc = sys.modules.get("bobert_companion")
    if bc is None:
        return False
    try:
        if getattr(bc, "_sleep_mode")[0]:
            return True
        if getattr(bc, "_standby_mode")[0]:
            return True
    except Exception:
        return False
    return False


def _chrome_process_count() -> int:
    """Approximate Chrome tab count via chrome.exe process count. Chrome
    runs each tab/extension/renderer in its own process, so the count is
    typically tabs + a small overhead. Returns 0 if psutil unavailable."""
    try:
        import psutil  # type: ignore
    except Exception:
        return 0
    total = 0
    try:
        for p in psutil.process_iter(["name"]):
            try:
                n = (p.info.get("name") or "").lower()
            except Exception:
                continue
            if n in ("chrome.exe", "chrome"):
                total += 1
    except Exception:
        return 0
    return total


def _visible_window_count() -> int:
    """Count of distinct visible top-level windows. Used as a 'desktop
    clutter' fallback when Chrome isn't the offender."""
    titles = _all_window_titles()
    # Filter the usual system/HUD entries so we don't false-positive on
    # JARVIS's own overlays.
    ignored = (
        "program manager", "default ime", "msctfime ui",
        "jarvis_hud", "jarvis_reticle", "settings",
    )
    filtered = [t for t in titles
                if not any(ig in t.lower() for ig in ignored)]
    return len(filtered)


# ─── pattern_memory access ───────────────────────────────────────────────

def _load_voice_commands() -> list[dict]:
    """Return the parsed voice_commands.jsonl file. Empty list on any error."""
    try:
        pm = importlib.import_module("pattern_memory")
    except Exception:
        try:
            pm = importlib.import_module("memory")
        except Exception:
            return []
    loader = getattr(pm, "_load_entries", None)
    if not callable(loader):
        return []
    try:
        return loader() or []
    except Exception:
        return []


def _extract_target_safe(text: str) -> tuple[str, str] | None:
    """Wrapper around pattern_memory._extract_target. Returns None on failure."""
    try:
        pm = importlib.import_module("pattern_memory")
    except Exception:
        try:
            pm = importlib.import_module("memory")
        except Exception:
            return None
    fn = getattr(pm, "_extract_target", None)
    if not callable(fn):
        return None
    try:
        return fn(text)
    except Exception:
        return None


def _normalize_text(s: str) -> str:
    """Cheap canonicalisation for near-duplicate detection."""
    s = (s or "").lower().strip()
    # Strip trailing punctuation + collapse whitespace
    s = "".join(c for c in s if c.isalnum() or c == " ")
    s = " ".join(s.split())
    return s


# ─── tell detection ──────────────────────────────────────────────────────

def _detect_repeat_question(entries: list[dict]) -> dict | None:
    """Same canonical utterance ≥ 2× within the last 10 min."""
    now = time.time()
    cutoff = now - REPEAT_QUESTION_WINDOW
    recent = [e for e in entries
              if isinstance(e.get("ts"), (int, float)) and e["ts"] >= cutoff]
    if len(recent) < 2:
        return None
    # Bucket by normalised text. Only consider utterances ≥ 3 words so
    # bare commands like "stop" or "next" don't trigger.
    from collections import defaultdict
    buckets: dict[str, list[float]] = defaultdict(list)
    for e in recent:
        norm = _normalize_text(e.get("text", ""))
        if len(norm.split()) < 3:
            continue
        buckets[norm].append(float(e["ts"]))
    best_norm = ""
    best_ts: list[float] = []
    for norm, ts_list in buckets.items():
        if len(ts_list) >= 2 and len(ts_list) > len(best_ts):
            best_norm = norm
            best_ts   = ts_list
    if not best_ts:
        return None
    minutes_since_first = max(1, int((now - min(best_ts)) / 60))
    return {
        "tell":  "repeat_question",
        "key":   f"repeat_question:{best_norm}",
        "n":     len(best_ts),
        "minutes": minutes_since_first,
        "text":  best_norm,
    }


def _detect_repeat_open(entries: list[dict]) -> dict | None:
    """Same open target ≥ REPEAT_OPEN_THRESHOLD times today."""
    today = datetime.date.today().isoformat()
    from collections import Counter
    counter: Counter[str] = Counter()
    for e in entries:
        iso = e.get("iso", "")
        if not isinstance(iso, str) or not iso.startswith(today):
            continue
        extracted = _extract_target_safe(e.get("text", ""))
        if extracted is None:
            continue
        cat, target = extracted
        if cat != "open" or not target:
            continue
        counter[target] += 1
    if not counter:
        return None
    target, count = counter.most_common(1)[0]
    if count < REPEAT_OPEN_THRESHOLD:
        return None
    return {
        "tell":   "repeat_open",
        "key":    f"repeat_open:{target}:{today}",
        "target": target,
        "n":      count,
    }


def _detect_tab_clutter() -> dict | None:
    """Chrome process count > threshold, else fall back to total visible windows."""
    chrome = _chrome_process_count()
    if chrome > TAB_CLUTTER_THRESHOLD:
        return {
            "tell": "tab_clutter",
            "key":  "tab_clutter:chrome",
            "n":    chrome,
        }
    win_count = _visible_window_count()
    if win_count > TAB_CLUTTER_THRESHOLD:
        return {
            "tell": "tab_clutter_windows",
            "key":  "tab_clutter:windows",
            "n":    win_count,
        }
    return None


def _detect_music_while_music(entries: list[dict]) -> dict | None:
    """User asked to play music in the last N minutes, AND JARVIS already
    started playback in the recent grace window."""
    bc = sys.modules.get("bobert_companion")
    if bc is None:
        return None
    last_jarvis_music = 0.0
    try:
        ts_slot = getattr(bc, "_jarvis_played_music_at", None)
        if isinstance(ts_slot, list) and ts_slot:
            last_jarvis_music = float(ts_slot[0])
    except Exception:
        return None
    if last_jarvis_music <= 0:
        return None
    if (time.time() - last_jarvis_music) > MUSIC_REPEAT_WINDOW:
        return None
    # Now find a 'play X' user request AFTER the JARVIS playback timestamp.
    candidates = [
        e for e in entries
        if isinstance(e.get("ts"), (int, float)) and e["ts"] > last_jarvis_music
    ]
    for e in candidates:
        extracted = _extract_target_safe(e.get("text", ""))
        if extracted is None:
            continue
        cat, target = extracted
        if cat != "play":
            continue
        # Found a play request that the user issued while music was already
        # going. Key on the utterance ts so we don't re-fire on the same one.
        return {
            "tell":   "music_while_music",
            "key":    f"music_while_music:{int(e['ts'])}",
            "target": target,
        }
    return None


def _pick_zinger(tell: dict) -> str:
    bank = _ZINGER_BANK.get(tell["tell"], [])
    if not bank:
        return ""
    line = random.choice(bank)
    try:
        return line.format(**tell)
    except Exception:
        return line


# ─── scheduler ───────────────────────────────────────────────────────────

def _scheduler_loop() -> None:
    time.sleep(INITIAL_DELAY_SECONDS)
    while True:
        try:
            cfg = _read_config()
            if not cfg["enabled"]:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Hard gates
            if _is_sleep_or_standby():
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            if _is_in_call():
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            state = _load_state()
            cooldown_s = max(60, int(cfg["cooldown"]) * 60)
            last_fire = float(state.get("last_fire_at", 0.0) or 0.0)
            if last_fire and (time.time() - last_fire) < cooldown_s:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Pull the command history once per tick.
            entries = _load_voice_commands()

            # Detect in priority order — recent question repeats are the most
            # immediately funny; clutter & music-while-music are slower-burn.
            detectors = (
                lambda: _detect_repeat_question(entries),
                lambda: _detect_music_while_music(entries),
                lambda: _detect_repeat_open(entries),
                lambda: _detect_tab_clutter(),
            )
            tell: dict | None = None
            for det in detectors:
                try:
                    tell = det()
                except Exception as e:
                    print(f"  [banter] detector error: {e}")
                    tell = None
                if tell:
                    # Per-tell cooldown — don't repeat the same observation
                    # over and over.
                    per_tell_cd_s = max(60, int(cfg["per_tell_cd"]) * 60)
                    per_tell_map = state.get("last_per_tell_at") or {}
                    last_for_tell = float(per_tell_map.get(tell["key"], 0.0) or 0.0)
                    if last_for_tell and (time.time() - last_for_tell) < per_tell_cd_s:
                        tell = None
                        continue
                    break

            if not tell:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            if random.random() > FIRE_PROBABILITY:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            line = _pick_zinger(tell)
            if not line:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            print(f"  [banter] firing ({tell['tell']}): {line}")
            _enqueue_speech(line)

            # Persist
            state["last_fire_at"] = time.time()
            state["last_tell"]    = tell["tell"]
            state["last_line"]    = line
            per_tell_map = dict(state.get("last_per_tell_at") or {})
            per_tell_map[tell["key"]] = time.time()
            # Prune ancient per-tell entries (older than 7 days) so the dict
            # doesn't grow without bound.
            cutoff = time.time() - 7 * 86400
            per_tell_map = {k: v for k, v in per_tell_map.items()
                            if isinstance(v, (int, float)) and v >= cutoff}
            state["last_per_tell_at"] = per_tell_map
            _save_state(state)

        except Exception:
            logging.exception("[banter] scheduler error")
        time.sleep(POLL_INTERVAL_SECONDS)


# ─── action: banter_status ───────────────────────────────────────────────

def _format_status() -> str:
    cfg = _read_config()
    state = _load_state()
    parts: list[str] = []
    if not cfg["enabled"]:
        parts.append("engine disabled in config")
    last_fire = float(state.get("last_fire_at", 0.0) or 0.0)
    if last_fire:
        age = int(time.time() - last_fire)
        if age >= 3600:
            ago = f"{age // 3600} hour{'s' if age // 3600 != 1 else ''} ago"
        elif age >= 60:
            ago = f"{age // 60} minute{'s' if age // 60 != 1 else ''} ago"
        else:
            ago = f"{age} seconds ago"
        parts.append(f"last zinger {ago} ({state.get('last_tell','?')})")
        cooldown_s = max(60, int(cfg["cooldown"]) * 60)
        remaining = cooldown_s - age
        if remaining > 0:
            parts.append(f"{remaining // 60} minute{'s' if remaining // 60 != 1 else ''} until next eligible")
    else:
        parts.append("no zingers yet this session")
    if _is_in_call():
        parts.append("currently in a call (suppressed)")
    if _is_sleep_or_standby():
        parts.append("sleep/standby active (suppressed)")
    return "Banter engine — " + "; ".join(parts) + "."


# ─── registration ────────────────────────────────────────────────────────

def register(actions):
    def banter_status(_: str = "") -> str:
        try:
            return _format_status()
        except Exception as e:
            return f"banter status failed: {e}"

    actions["banter_status"] = banter_status

    cfg = _read_config()
    if not cfg["enabled"]:
        print("  [banter] BANTER_ENABLED is False — engine disabled")
        return

    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()
    print(
        f"  [banter] background loop running "
        f"(poll {int(POLL_INTERVAL_SECONDS)}s, cooldown {cfg['cooldown']}m, "
        f"per-tell {cfg['per_tell_cd']}m, p(fire)={FIRE_PROBABILITY:.2f})"
    )


# ─── offline smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("--- banter engine smoke test ---")
    fake_entries = [
        {"ts": time.time() - 60,  "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "text": "What's the weather today"},
        {"ts": time.time() - 30,  "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "text": "Whats the weather today!"},
        {"ts": time.time() - 200, "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "text": "open chrome"},
        {"ts": time.time() - 190, "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "text": "open chrome"},
        {"ts": time.time() - 180, "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "text": "open chrome"},
        {"ts": time.time() - 170, "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "text": "open chrome"},
        {"ts": time.time() - 160, "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "text": "open chrome"},
    ]
    print("repeat_question:", _detect_repeat_question(fake_entries))
    print("repeat_open    :", _detect_repeat_open(fake_entries))
    print("tab_clutter    :", _detect_tab_clutter())
    print("chrome procs   :", _chrome_process_count())
    print("visible windows:", _visible_window_count())
    print("zinger sample  :", _pick_zinger({"tell": "tab_clutter", "n": 47}))
    print("status         :", _format_status())
