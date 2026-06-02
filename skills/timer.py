"""
Timer / reminder skill for JARVIS.

Actions added:
  set_timer, <duration> | <message>   e.g. "5 minutes | check the oven"
  list_timers                         show what's pending
  cancel_timer, <id>                  cancel by id

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


def _parse_duration(text: str) -> int | None:
    """Parse '5 minutes', '30 seconds', '2 hours', '1 day' → total seconds."""
    text = text.strip().lower()
    total = 0
    found = False
    pattern = r"(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)"
    for n, unit in re.findall(pattern, text):
        n = int(n)
        if unit.startswith(("sec",)):
            total += n
        elif unit.startswith(("min",)):
            total += n * 60
        elif unit.startswith(("hr", "hour")):
            total += n * 3600
        elif unit.startswith("day"):
            total += n * 86400
        found = True
    return total if found else None


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


def register(actions):
    def set_timer(args: str) -> str:
        if "|" not in args:
            return "format: set_timer, <duration> | <message>  (e.g. '5 minutes | check the oven')"
        dur_str, msg = (s.strip() for s in args.split("|", 1))
        secs = _parse_duration(dur_str)
        if secs is None or secs <= 0:
            return f"could not parse duration '{dur_str}'. Try '5 minutes', '30 seconds', '2 hours'."
        if not msg:
            return "timer needs a message describing what to remind about"

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
        return f"timer #{tid} set — will remind you in {when}: '{msg}'"

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

    def cancel_timer(args: str) -> str:
        args = args.strip()
        if args.lower() == "all":
            with _lock:
                count = len(_timers)
                for tid, (timer, _, _) in list(_timers.items()):
                    timer.cancel()
                _timers.clear()
            return f"cancelled {count} timer(s)"
        try:
            tid = int(args)
        except ValueError:
            return "format: cancel_timer, <id>  (or 'all')"
        with _lock:
            entry = _timers.pop(tid, None)
        if entry is None:
            return f"no timer #{tid}"
        timer, msg, _ = entry
        timer.cancel()
        return f"cancelled timer #{tid} ('{msg}')"

    actions["set_timer"]    = set_timer
    actions["list_timers"]  = list_timers
    actions["cancel_timer"] = cancel_timer
