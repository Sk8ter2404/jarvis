"""
Timer / reminder skill for JARVIS.

Actions added:
  set_timer, <duration> [| <message>] e.g. "5 minutes", "5 minutes for tea",
                                      "an hour and a half", or the legacy
                                      "5 minutes | check the oven"
  list_timers                         show what's pending
  cancel_timer[, <id|label|all>]      no arg → cancel the running/soonest one;
                                      a number → that id; "all" → clear all;
                                      text → match a timer by its message

When a timer fires it writes the reminder to pending_speech.json in the project
root. The JARVIS main loop polls that file between listen cycles and speaks any
pending messages aloud.
"""
import importlib
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta

_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEECH_QUEUE = os.path.join(_PROJECT_DIR, "pending_speech.json")

# Ensure the project root is importable so `core.atomic_io` resolves whether
# this module is loaded as `skills.timer` or run directly.
if _PROJECT_DIR not in sys.path:  # pragma: no cover - import-time sys.path guard; already satisfied under the test harness (root on path)
    sys.path.insert(0, _PROJECT_DIR)

from core.atomic_io import _atomic_write_json  # noqa: E402

# File-level lock so concurrent timer firings don't interleave their
# read-modify-write on pending_speech.json and corrupt it.
_speech_lock = threading.Lock()

_timers: dict[int, tuple[threading.Timer, str, float]] = {}
_next_id = [1]
_lock    = threading.Lock()


# Spelled-out small numbers the LLM (and users) emit in voice transcripts:
# "set a timer for five minutes", "ten min". Kept short — anything bigger is
# almost always dictated as digits.
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "forty five": 45,
    "sixty": 60, "ninety": 90,
}

# A number followed by a unit word, as two shapes unified into one finditer
# via an alternation:
#   digits + (optionally bare-letter) unit:  "10m", "5 min", "30 seconds"
#   spelled word + a SPELLED-OUT unit only:  "five minutes", "ten min"
# Bare single-letter units (s/m/h/d) are allowed ONLY after digits — nobody
# writes "tenm", and allowing a lone letter after a word makes the 'd' in
# "and" parse as a day. group(1)/group(2) = digit form; group(3)/group(4) = word form.
_DUR_UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b"
    r"|"
    r"\b([a-z]+)\s+(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b"
)

# "half an hour" / "half a minute" — the whole span, so we can both add 0.5
# of the unit AND remove it from the text before the digit+unit scan (else
# the 'an hour' inside would also score a full hour).
_HALF_UNIT_RE = re.compile(
    r"\bhalf\s+(?:an?\s+)?(hour|hr|minute|min|day|second|sec)s?\b")
# "<unit> and a half" → add 0.5 of that trailing unit.
_AND_HALF_RE = re.compile(
    r"\b(hour|hr|minute|min|day|second|sec)s?\s+and\s+a\s+half\b")


def _unit_seconds(unit: str) -> int:
    if unit in ("s",) or unit.startswith("sec"):
        return 1
    if unit in ("m",) or unit.startswith("min"):
        return 60
    if unit in ("h",) or unit.startswith(("hr", "hour")):
        return 3600
    if unit in ("d",) or unit.startswith("day"):
        return 86400
    return 0


def _word_to_num(tok: str) -> float | None:
    tok = tok.strip()
    try:
        return float(tok)
    except ValueError:
        return _WORD_NUMBERS.get(tok)


def _parse_clock_time(text: str) -> int | None:
    """If `text` names an absolute CLOCK time ('8 pm', '7:30 am', '20:00'),
    return seconds from now until the next occurrence of it (today, or tomorrow
    if it has already passed). Else None.

    Without this, 'remind me at 8 pm' fell through to the bare-number rule and
    set an 8-MINUTE timer (the 'pm' was ignored) — a silent wrong result."""
    t = (text or "").strip().lower()
    hour = minute = None
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?\b", t)
    if m:
        hour = int(m.group(1)) % 12
        if m.group(3) == "p":
            hour += 12
        minute = int(m.group(2) or 0)
    else:
        # 24-hour "20:00" / "07:30" — require the colon so a bare "8" isn't read
        # as a time (that stays a duration).
        m2 = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t)
        if m2:
            hour, minute = int(m2.group(1)), int(m2.group(2))
    if hour is None or not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    now = datetime.fromtimestamp(time.time())
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return int((target - now).total_seconds())


def _parse_duration(text: str) -> int | None:
    """Parse a free-text duration → total seconds, or None if unparseable.

    Handles the shapes the local LLM and users actually emit:
      '5 minutes', '30 seconds', '2 hours', '1 day'   (digit + unit)
      '5 min', '10m', '90 secs', '1h'                 (abbrev / bare-letter unit)
      'five minutes', 'an hour', 'ten min'            (spelled-out number)
      '1 hour 30 minutes', '2 hours 15 minutes'       (combined units, summed)
      'an hour and a half', 'half an hour'            ('and a half' / 'half a')
      '5'                                             (bare number → minutes)
    """
    text = (text or "").strip().lower()
    if not text:
        return None

    # An absolute clock time ('8 pm', '7:30 am', '20:00') wins over the
    # duration rules below — otherwise '8 pm' loses its 'pm' and becomes an
    # 8-MINUTE timer.
    clock = _parse_clock_time(text)
    if clock is not None:
        return clock

    total = 0.0
    found = False

    # '<unit> and a half' first (covers 'an hour and a half') — add 0.5 of the
    # trailing unit. The leading 'an hour' is still scored as a full unit below.
    tail = _AND_HALF_RE.search(text)
    if tail:
        total += 0.5 * _unit_seconds(tail.group(1))
        found = True

    # 'half an hour' / 'half a minute' → 0.5 of the named unit, then REMOVE the
    # span so the 'an hour' inside it isn't double-counted as a full hour.
    def _half_repl(m):
        nonlocal total, found
        total += 0.5 * _unit_seconds(m.group(1))
        found = True
        return " "
    text = _HALF_UNIT_RE.sub(_half_repl, text)

    for m in _DUR_UNIT_RE.finditer(text):
        num_tok = m.group(1) if m.group(1) is not None else m.group(3)
        unit_tok = m.group(2) if m.group(2) is not None else m.group(4)
        num = _word_to_num(num_tok)
        if num is None:
            continue
        per = _unit_seconds(unit_tok)
        if per:
            total += num * per
            found = True

    if not found:
        # Bare number with no unit ("5", "for 5", "set a timer for 5") — JARVIS
        # convention is minutes, matching how people speak. Accept a lone digit
        # token anywhere, or a single spelled-out number word.
        digits = re.findall(r"\d+(?:\.\d+)?", text)
        if len(digits) == 1:
            num = float(digits[0])
            if num > 0:
                return int(round(num * 60))
        if not digits:
            # No digits → try a single spelled-out number word ("five").
            # Exclude the articles 'a'/'an' (they're filler here, not a real
            # count — otherwise "set a timer" with no number → a phantom 1-min
            # timer). A real count requires an explicit number word.
            words = [w for w in re.findall(r"[a-z]+", text)
                     if w in _WORD_NUMBERS and w not in ("a", "an")]
            if len(words) == 1:
                num = _WORD_NUMBERS[words[0]]
                if num > 0:
                    return int(round(num * 60))
        return None

    secs = int(round(total))
    return secs if secs > 0 else None


def _enqueue_speech(message: str):
    """Append a reminder to pending_speech.json for the main loop to speak.

    Routes through bobert_companion.proactive_announce() so this skill shares
    one write path with every other pending_speech.json co-writer
    (bambu_monitor, status_panel, night_owl_mode, screen_watch, teams_nudge,
    …) and they don't race each other. Falls back to a local atomic write
    only when the parent module isn't loaded yet (import-time registration /
    unit tests) or the announcer call fails — so a broken parent import
    can't silence a timer."""
    try:
        bc = importlib.import_module("bobert_companion")
        announcer = getattr(bc, "proactive_announce", None)
        if callable(announcer):
            announcer(message, source="timer")
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
            # Atomic write failed (e.g. read-only network share, full disk,
            # permission denied). Fall back to console so the reminder isn't
            # silently lost — at minimum the user sees it in the log stream.
            print(f"  [timer] speech-queue write failed ({e}); reminder: {message}")


def enumerate_timers() -> list[dict]:
    """Snapshot every active timer as {id, message, fire_at}.

    Used by the blue/green producer (bobert_companion handoff writer) so the
    incoming process can rebuild outstanding reminders instead of dropping
    them. Snapshot is taken under _lock so a concurrent fire/cancel can't
    expose a half-mutated entry. fire_at is absolute wall-clock seconds so
    the consumer can compute a fresh remaining duration."""
    with _lock:
        return [
            {"id": tid, "message": msg, "fire_at": float(fire_at)}
            for tid, (_, msg, fire_at) in sorted(_timers.items())
        ]


def restore_timers(payload: list) -> int:
    """Reinstate timers from a handoff payload. Returns the count restored.

    Each entry must look like {id, message, fire_at}. Entries whose fire_at
    has already passed fire immediately via _enqueue_speech (so the user
    isn't silently robbed of a reminder by handoff latency). _next_id is
    bumped past every restored id so post-handoff set_timer calls can't
    collide with a reinstated one."""
    if not isinstance(payload, list):
        return 0
    restored = 0
    now = time.time()
    with _lock:
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            try:
                tid = int(entry.get("id"))
                msg = str(entry.get("message") or "")
                fire_at = float(entry.get("fire_at") or 0.0)
            except (TypeError, ValueError):
                continue
            if not msg or tid <= 0:
                continue
            if tid in _timers:
                continue
            remaining = fire_at - now
            if remaining <= 0:
                # Fire-on-restore — releases _lock briefly so _enqueue_speech
                # (which may route through proactive_announce) doesn't deadlock
                # if the announcer ever calls back into this module.
                _next_id[0] = max(_next_id[0], tid + 1)
                try:
                    _enqueue_speech(f"Reminder, sir — {msg}")
                except Exception as e:
                    print(f"  [timer] restore-fire failed for #{tid}: {e}")
                restored += 1
                continue

            def _fire(_tid=tid, _msg=msg):
                print(f"  [timer] 🔔 fired #{_tid}: {_msg}")
                _enqueue_speech(f"Reminder, sir — {_msg}")
                with _lock:
                    _timers.pop(_tid, None)

            t = threading.Timer(remaining, _fire)
            t.daemon = True
            t.start()
            _timers[tid] = (t, msg, fire_at)
            _next_id[0] = max(_next_id[0], tid + 1)
            restored += 1
    return restored


# Words that introduce a label after a duration in free text:
# "5 minutes FOR tea", "10 min TO check the oven", "30 seconds, then stretch".
_LABEL_INTRO_RE = re.compile(r"\b(?:for|to|about|that|then|and)\b", re.IGNORECASE)


def _split_timer_args(args: str) -> tuple[int | None, str]:
    """Pull a (duration_seconds, label) pair out of whatever the LLM emitted.

    Accepts, in priority order:
      1. legacy 'duration | message'      → '5 minutes | check the oven'
      2. duration + 'for/to' label        → '5 minutes for tea', '10 min to stretch'
      3. label-then-duration              → 'tea timer 5 minutes', 'oven 10 min'
      4. bare duration, no label          → '5 minutes', 'an hour', '5'
    Returns (None, '') when no duration can be found so the caller can emit an
    HONEST parse-failure instead of silently inventing one.
    """
    raw = (args or "").strip()
    if not raw:
        return None, ""

    # 1. Legacy pipe form — keep supporting it verbatim.
    if "|" in raw:
        dur_str, label = (s.strip() for s in raw.split("|", 1))
        return _parse_duration(dur_str), label

    # 2/3. There's a duration token somewhere; split the string around the
    # FIRST number+unit match so text on either side becomes the label.
    m = _DUR_UNIT_RE.search(raw)
    num_tok = (m.group(1) or m.group(3)) if m else None
    if m and num_tok is not None and _word_to_num(num_tok) is not None:
        secs = _parse_duration(raw)  # parse the whole thing (handles compounds)
        before = raw[:m.start()].strip(" ,.-")
        # Label is whatever trails the duration ("for tea"), else whatever
        # preceded it ("tea timer"). Strip a leading "for/to" connector and a
        # trailing/leading "timer"/"reminder" noise word.
        after = raw[m.end():].strip(" ,.-")
        # For compound durations ("1 hour 30 minutes for tea") keep trimming
        # trailing duration fragments out of `after`, and drop a dangling
        # "and a half" / "a half" so it doesn't leak into the label.
        after = _DUR_UNIT_RE.sub("", after)
        after = re.sub(r"^\s*(?:and\s+)?a\s+half\b", "", after).strip(" ,.-")
        label = after or before
        label = _LABEL_INTRO_RE.sub("", label, count=1).strip(" ,.-") if label else ""
        label = re.sub(r"\b(?:timer|reminder)\b", "", label, flags=re.IGNORECASE).strip(" ,.-")
        return secs, label

    # 4. Bare number / spelled-out number with no unit word ("5", "five").
    return _parse_duration(raw), ""


def register(actions):
    def set_timer(args: str) -> str:
        secs, msg = _split_timer_args(args)
        if secs is None or secs <= 0:
            return ("I couldn't tell how long, sir — try 'set a timer for "
                    "5 minutes' (or '5 minutes for tea').")
        # A label is optional; default to a generic reminder so a bare
        # "set a timer for 5 minutes" still fires a real, speakable timer.
        labelled = bool(msg.strip())
        msg = msg.strip() or "your timer is up"

        with _lock:
            tid = _next_id[0]
            _next_id[0] += 1

        def _fire():
            print(f"  [timer] 🔔 fired #{tid}: {msg}")
            _enqueue_speech(f"Reminder, sir — {msg}")
            with _lock:
                _timers.pop(tid, None)

        t = threading.Timer(secs, _fire)
        t.daemon = True
        t.start()
        with _lock:
            _timers[tid] = (t, msg, time.time() + secs)

        # Human-readable summary of when it'll fire
        if secs < 60:
            when = f"{secs}s"
        elif secs < 3600:
            when = f"{secs // 60}m {secs % 60}s" if secs % 60 else f"{secs // 60}m"
        else:
            when = f"{secs // 3600}h {(secs % 3600) // 60}m"
        if labelled:
            return f"timer #{tid} set — will remind you in {when}: '{msg}'"
        return f"timer #{tid} set — will remind you in {when}"

    def list_timers(_: str = "") -> str:
        with _lock:
            if not _timers:
                return "no active timers"
            now = time.time()
            lines = []
            for tid, (_, msg, fire_at) in sorted(_timers.items()):
                remaining = max(0, int(fire_at - now))
                if remaining < 60:
                    rem_str = f"{remaining}s"
                elif remaining < 3600:
                    rem_str = f"{remaining // 60}m {remaining % 60}s"
                else:
                    rem_str = f"{remaining // 3600}h {(remaining % 3600) // 60}m"
                lines.append(f"  #{tid}: '{msg}' in {rem_str}")
            return f"{len(lines)} active timer(s):\n" + "\n".join(lines)

    def _cancel_one(tid: int) -> str:
        with _lock:
            entry = _timers.pop(tid, None)
        if entry is None:
            return f"no timer #{tid}"
        timer, msg, _ = entry
        timer.cancel()
        return f"cancelled timer #{tid} ('{msg}')"

    def cancel_timer(args: str = "") -> str:
        """Cancel a timer from whatever the LLM emitted.

        '' / 'my timer' / 'the timer' → cancel the only running one (or, if
        several, the one firing SOONEST, and say which). 'all' → clear them
        all. A bare number → that id. Any other text → match a timer whose
        message contains it. Honest 'no timers' message when none exist."""
        args = (args or "").strip()
        low = args.lower()

        # Cancel-everything.
        if low in ("all", "everything", "them all", "all timers"):
            with _lock:
                count = len(_timers)
                for _tid, (timer, _, _) in list(_timers.items()):
                    timer.cancel()
                _timers.clear()
            if not count:
                return "there are no timers running, sir."
            return f"cancelled {count} timer(s)"

        # Explicit id ("cancel timer 3").
        m = re.search(r"\d+", args)
        if m and m.group(0) == args.strip("#").strip():
            return _cancel_one(int(m.group(0)))

        # No specific target — pick the single running timer, or the soonest.
        loose = (not args) or low in (
            "my timer", "the timer", "timer", "my", "the", "it",
            "this", "that", "my reminder", "the reminder", "reminder",
            "current", "active",
        )
        if loose:
            with _lock:
                if not _timers:
                    return "there are no timers running, sir."
                # soonest to fire = smallest fire_at
                tid = min(_timers, key=lambda k: _timers[k][2])
                only = len(_timers) == 1
            res = _cancel_one(tid)
            if only or res.startswith("no timer"):
                return res
            return res + " (the next one due; say 'cancel all' to clear the rest)"

        # Fall back to matching the timer message by substring
        # ("cancel the tea timer" → the timer whose message mentions tea).
        with _lock:
            if not _timers:
                return "there are no timers running, sir."
            needle = re.sub(r"\b(?:timer|reminder|for|the|my)\b", "", low).strip()
            matches = [tid for tid, (_, msg, _) in _timers.items()
                       if needle and needle in msg.lower()]
        if len(matches) == 1:
            return _cancel_one(matches[0])
        if len(matches) > 1:
            return (f"several timers match '{needle}', sir — "
                    f"say the number: {', '.join(f'#{t}' for t in sorted(matches))}.")
        # Bare number with stray chars, or unmatched label.
        if m:
            return _cancel_one(int(m.group(0)))
        return (f"I don't see a timer matching '{args}', sir — "
                "say 'list timers' to hear them, or 'cancel all'.")

    actions["set_timer"]    = set_timer
    actions["list_timers"]  = list_timers
    actions["cancel_timer"] = cancel_timer
