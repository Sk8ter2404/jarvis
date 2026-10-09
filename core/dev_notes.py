"""core/dev_notes.py - "tell Claude ..." becomes a note Claude really reads
(2026-10-05).

WHY THIS EXISTS
===============
Live 00:30:42 "also tell Claude to ... do some research on the screen
visioning": the brain ran a web_search itself and summarised unrelated
results. Live 00:32:48 "tell Claude that I want it to be able to watch
everything ...": JARVIS said "I'll relay your instructions to him
immediately" - and relayed nothing; there is no channel to Claude.

So: the owner's words, verbatim, plus what JARVIS had just been doing (the
last action results, the last vision-trace ids, the session log and its
byte offset), are appended to ``data/notes_for_claude.jsonl`` - a local,
gitignored file the developer reads at the start of the next session - and
JARVIS says exactly that, never that he "relayed" anything. This is not
report_bug (that one opens a PUBLIC GitHub issue).

Optional mirror: NOTES_FOR_CLAUDE_MIRROR (a second path, e.g. a synced
folder the owner chooses; empty = none). Never raises.
"""
from __future__ import annotations

import json
import os
import threading
import time

__all__ = ["SPOKEN_LINE", "notes_path", "add_note", "read_notes",
           "set_context_provider"]

# "can’t" with a typographic apostrophe: the ASCII "can't" is a failure
# marker (core.failure_markers), which would keep this finished sentence from
# being spoken word for word and start a follow-up round instead.
SPOKEN_LINE = ("Noted for Claude, sir — it's in the developer notes. I "
               "can’t reach him directly; he'll see it next time he works "
               "on me.")

_lock = threading.Lock()
_ctx_provider = [None]


def set_context_provider(fn) -> None:
    """``fn() -> {"session_log", "log_offset", "last_results",
    "trace_ids"}`` (the monolith registers one). Never raises."""
    _ctx_provider[0] = fn if callable(fn) else None


def notes_path() -> str:
    from core.paths import data_file
    return data_file("notes_for_claude.jsonl")


def _mirror_path() -> str:
    try:
        from core import config as _c
        p = str(getattr(_c, "NOTES_FOR_CLAUDE_MIRROR", "") or "").strip()
        return os.path.expandvars(os.path.expanduser(p)) if p else ""
    except Exception:
        return ""


def add_note(note, utterance="", *, now=None) -> "dict | None":
    """Append one note. Returns the record, or None on failure. Never
    raises. No network."""
    try:
        rec = {"ts": round(float(time.time() if now is None else now), 3),
               "when": time.strftime("%Y-%m-%d %H:%M:%S",
                                     time.localtime(now or time.time())),
               "utterance": str(utterance or ""),
               "note": " ".join(str(note or "").split())}
        try:
            from core.version import version_string
            rec["version"] = version_string()
        except Exception:
            pass
        try:
            fn = _ctx_provider[0]
            ctx = fn() if fn is not None else {}
            if isinstance(ctx, dict):
                rec["session_log"] = ctx.get("session_log")
                rec["log_offset"] = ctx.get("log_offset")
                rec["last_5_action_results"] = [
                    str(r)[:300] for r in list(ctx.get("last_results") or [])[-5:]]
                rec["last_5_trace_ids"] = [
                    str(t) for t in list(ctx.get("trace_ids") or [])[-5:]]
        except Exception:
            pass
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        paths = [notes_path()]
        mirror = _mirror_path()
        if mirror:
            paths.append(mirror)
        wrote = False
        with _lock:
            for i, p in enumerate(paths):
                try:
                    d = os.path.dirname(p)
                    if d:
                        os.makedirs(d, exist_ok=True)
                    with open(p, "a", encoding="utf-8") as f:
                        f.write(line)
                    wrote = wrote or i == 0
                except Exception as e:
                    print(f"  [dev-notes] could not write "
                          f"{'mirror' if i else 'note'}: {type(e).__name__}",
                          flush=True)
        if not wrote:
            return None
        print(f"  [dev-notes] note for Claude saved ({len(rec['note'])} chars)",
              flush=True)
        return rec
    except Exception:
        return None


def read_notes(limit=None) -> list:
    try:
        out = []
        with open(notes_path(), encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out[-int(limit):] if limit else out
    except Exception:
        return []
