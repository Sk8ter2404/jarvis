"""
Multimodal ambient fact-extractor.

Runs as an independent background daemon (started by register()) that wakes
every AMBIENT_EXTRACT_INTERVAL_S, reads the last AMBIENT_EXTRACT_BATCH
entries from all three ambient logs (mic, system audio, screen) plus the
in-memory rolling buffer maintained by skills/ambient_listen, and calls a
small LLM prompt to:

  1. Attribute mentions to a likely SPEAKER (mic), APP/WINDOW (system_audio),
     or VISIBLE-ON-SCREEN context (screen entry).
  2. Extract a short list of "new_facts" and "new_projects" the user said
     or saw during the window — same shape that bobert_companion.learn_from_turn
     already merges into bobert_memory.json.

New facts/projects are appended to bobert_memory.json via the same MAX_*
ceilings + dedupe pass the live learner uses. The extractor also writes a
per-run summary into data/ambient_extracts.jsonl so the user can audit what
was learned.

Actions registered:
    ambient_extract_start    — kick the loop manually (idempotent)
    ambient_extract_stop     — stop the loop
    ambient_extract_status   — last run + counters
    ambient_extract_now      — fire one extraction pass synchronously
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import deque
from typing import Optional


_PROJECT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR        = os.path.join(_PROJECT_DIR, "data")
_AUDIO_JSONL     = os.path.join(_DATA_DIR, "ambient_transcripts.jsonl")
_SCREEN_JSONL    = os.path.join(_DATA_DIR, "ambient_screen_log.jsonl")
_EXTRACT_JSONL   = os.path.join(_DATA_DIR, "ambient_extracts.jsonl")

# Hard cap so this file can't grow unbounded between rotations.
_EXTRACT_HARD_CAP = 2000


_lock = threading.RLock()
_thread: Optional[threading.Thread] = None
_stop_evt = threading.Event()
_started_at: Optional[float] = None
_last_run_at: float = 0.0
_runs_total: int = 0
_facts_added_total: int = 0
_projects_added_total: int = 0
_last_error: Optional[str] = None


def _ensure_project_on_path() -> None:
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)


def _get_bobert():
    return (sys.modules.get("bobert_companion")
            or sys.modules.get("__main__"))


def _get_config(name: str, default):
    b = _get_bobert()
    if b is None:
        return default
    return getattr(b, name, default)


def _tail_jsonl(path: str, n: int) -> list[dict]:
    if not os.path.exists(path) or n <= 0:
        return []
    try:
        # Stream from the tail to avoid slurping huge logs.
        with open(path, "r", encoding="utf-8") as f:
            dq: deque[str] = deque(f, maxlen=n)
    except Exception as e:
        print(f"  [ambient-extract] read failed ({path}): {e}")
        return []
    out: list[dict] = []
    for line in dq:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _format_window(entries: list[dict]) -> str:
    """Render the merged stream in a compact form the LLM can attribute."""
    lines: list[str] = []
    for e in entries:
        ts = e.get("ts", 0.0)
        t  = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "?"
        src = e.get("source") or ("mic" if "text" in e and "summary" not in e else "screen")
        if src == "screen":
            win = (e.get("window") or "")[:60]
            summary = e.get("summary") or ""
            ents = ", ".join((e.get("entities") or [])[:6])
            if e.get("sensitive"):
                continue   # never feed redacted lines to the extractor
            lines.append(f"[{t}] screen ({win}) :: {summary}"
                         + (f" {{entities: {ents}}}" if ents else ""))
        elif src == "system_audio":
            win = (e.get("window") or "")[:60]
            txt = (e.get("text") or "").strip()[:300]
            lines.append(f"[{t}] system_audio ({win}) :: {txt}")
        else:  # mic / unknown
            win = (e.get("window") or "")[:60]
            txt = (e.get("text") or "").strip()[:300]
            tag = "mic"
            tail = f" ({win})" if win else ""
            lines.append(f"[{t}] {tag}{tail} :: {txt}")
    return "\n".join(lines)


def _llm_extract(window_text: str) -> dict:
    """Call the small LLM helper that learn_from_turn already uses so we
    share the AI_BACKEND / CLAUDE_MODEL config. Returns the parsed JSON
    object on success, or an empty dict on any failure."""
    b = _get_bobert()
    if b is None:
        return {}
    quick = getattr(b, "_llm_quick", None)
    if not callable(quick):
        return {}

    system = (
        "You extract long-term memory items from a multimodal observation "
        "window. The user has been working at a PC; the input below is a "
        "mixed stream of microphone speech (mic), PC audio output "
        "(system_audio: podcasts/calls/videos), and brief VLM summaries of "
        "the screen (screen). Output ONLY valid JSON in this exact shape, "
        "nothing else:\n"
        '{"new_facts": ["..."], "new_projects": ["..."], "mentions": '
        '[{"text": "...", "source": "mic|system_audio|screen", '
        '"attribution": "speaker|app|visible"}]}\n\n'
        "STRICT RULES:\n"
        "- new_facts: ONLY include facts the *user* clearly stated through "
        "the MIC. Do NOT extract facts that came only from system_audio or "
        "screen — those are content they are consuming, not personal facts.\n"
        "- new_projects: same — only if the user said it through the mic.\n"
        "- mentions: short list (max 8) of named things you noticed across "
        "all three streams. attribution='speaker' for mic, 'app' for "
        "system_audio, 'visible' for screen.\n"
        "- If nothing meaningful was observed, return empty lists.\n"
    )

    try:
        raw = quick(system, window_text, max_tokens=500)
    except Exception as e:
        print(f"  [ambient-extract] LLM call failed: {e}")
        return {}

    if not raw:
        return {}
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _merge_into_memory(extracted: dict) -> tuple[int, int]:
    """Append new_facts / new_projects to bobert_memory.json via
    bobert_companion.merge_memory, which holds _memory_lock across the
    full load-modify-save cycle so this daemon cannot race the live
    learn_from_turn worker."""
    b = _get_bobert()
    if b is None:
        return 0, 0

    new_facts    = [str(x).strip() for x in (extracted.get("new_facts") or []) if str(x).strip()]
    new_projects = [str(x).strip() for x in (extracted.get("new_projects") or []) if str(x).strip()]

    if not new_facts and not new_projects:
        return 0, 0

    merge = getattr(b, "merge_memory", None)
    if not callable(merge):
        return 0, 0

    try:
        added_facts_list, added_projs_list = merge(
            new_facts=new_facts,
            new_projects=new_projects,
        )
    except Exception as e:
        print(f"  [ambient-extract] memory merge failed: {e}")
        return 0, 0

    return len(added_facts_list), len(added_projs_list)


def _append_extract_log(entry: dict) -> None:
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_EXTRACT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [ambient-extract] log append failed: {e}")
    # Cheap rotation: keep only the last _EXTRACT_HARD_CAP lines.
    try:
        if not os.path.exists(_EXTRACT_JSONL):
            return
        with open(_EXTRACT_JSONL, "rb") as f:
            n = sum(1 for _ in f)
        if n < int(_EXTRACT_HARD_CAP * 1.5):
            return
        with open(_EXTRACT_JSONL, "r", encoding="utf-8") as f:
            tail = f.readlines()[-_EXTRACT_HARD_CAP:]
        fd, tmp = tempfile.mkstemp(dir=_DATA_DIR, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(tail)
        os.replace(tmp, _EXTRACT_JSONL)
    except Exception as e:
        print(f"  [ambient-extract] log rotate failed: {e}")


def _run_once() -> dict:
    """One full extraction pass. Returns a summary dict for logging."""
    global _last_run_at, _runs_total
    global _facts_added_total, _projects_added_total, _last_error

    batch = int(_get_config("AMBIENT_EXTRACT_BATCH", 50))
    batch = max(5, min(500, batch))

    audio_entries  = _tail_jsonl(_AUDIO_JSONL, batch)
    screen_entries = _tail_jsonl(_SCREEN_JSONL, batch)

    merged = audio_entries + screen_entries
    merged.sort(key=lambda e: float(e.get("ts", 0.0)))
    # Window: only consider the last interval × 2 worth of seconds so we
    # don't reprocess hours of history if the log was idle.
    interval = float(_get_config("AMBIENT_EXTRACT_INTERVAL_S", 300.0))
    cutoff = time.time() - max(60.0, interval * 2.5)
    merged = [e for e in merged if float(e.get("ts", 0.0)) >= cutoff]

    summary = {
        "ts": time.time(),
        "audio_entries": sum(1 for e in merged if e.get("source") == "system_audio"),
        "mic_entries":   sum(1 for e in merged if e.get("source") == "mic"),
        "screen_entries": sum(1 for e in merged if e.get("source") == "screen"),
        "facts_added": 0,
        "projects_added": 0,
        "mentions": [],
    }

    if not merged:
        with _lock:
            _last_run_at = time.time()
            _runs_total += 1
        _append_extract_log(summary)
        return summary

    window_text = _format_window(merged)
    if not window_text.strip():
        with _lock:
            _last_run_at = time.time()
            _runs_total += 1
        _append_extract_log(summary)
        return summary

    extracted = _llm_extract(window_text)
    facts, projs = _merge_into_memory(extracted)

    summary["facts_added"]     = facts
    summary["projects_added"]  = projs
    summary["mentions"]        = (extracted.get("mentions") or [])[:8]

    with _lock:
        _last_run_at = time.time()
        _runs_total += 1
        _facts_added_total    += facts
        _projects_added_total += projs
        _last_error = None

    _append_extract_log(summary)
    return summary


def _loop() -> None:
    global _last_error
    interval = float(_get_config("AMBIENT_EXTRACT_INTERVAL_S", 300.0))
    interval = max(60.0, min(3600.0, interval))
    print(f"  [ambient-extract] daemon online, interval={int(interval)}s")
    while not _stop_evt.is_set():
        try:
            _run_once()
        except Exception as e:
            _last_error = f"extraction failed: {e}"
            print(f"  [ambient-extract] {_last_error}")
        if _stop_evt.wait(interval):
            break
    print("  [ambient-extract] daemon exiting")


def ambient_extract_start(_: str = "") -> str:
    global _thread, _stop_evt, _started_at, _last_error
    with _lock:
        if _thread is not None and _thread.is_alive():
            return "Ambient extractor is already running, sir."
        _stop_evt = threading.Event()
        _started_at = time.time()
        _last_error = None
        _thread = threading.Thread(
            target=_loop, name="ambient-extract", daemon=True)
        _thread.start()
    interval = int(_get_config("AMBIENT_EXTRACT_INTERVAL_S", 300.0))
    return (f"Ambient extractor engaged, sir. I'll fold mic + system audio + "
            f"screen observations into long-term memory every {interval}s.")


def ambient_extract_stop(_: str = "") -> str:
    global _thread, _started_at
    with _lock:
        if _thread is None or not _thread.is_alive():
            return "Ambient extractor is not running, sir."
        _stop_evt.set()
        t = _thread
    t.join(timeout=5.0)
    with _lock:
        if t.is_alive():
            return "Ambient extractor did not stop cleanly within 5 s, sir."
        _thread = None
        elapsed = (time.time() - _started_at) if _started_at else 0.0
        _started_at = None
        runs = _runs_total
        facts = _facts_added_total
        projs = _projects_added_total
    return (f"Ambient extractor disengaged, sir. {runs} passes over "
            f"{int(elapsed)}s — {facts} facts and {projs} projects merged.")


def ambient_extract_status(_: str = "") -> str:
    with _lock:
        running = _thread is not None and _thread.is_alive()
        runs = _runs_total
        facts = _facts_added_total
        projs = _projects_added_total
        last_run = _last_run_at
        err = _last_error
    last = (time.strftime("%H:%M:%S", time.localtime(last_run))
            if last_run else "never")
    state = "ON" if running else "OFF"
    suffix = f" (last error: {err})" if err and not running else ""
    return (f"Ambient extractor {state}, sir — {runs} passes, "
            f"{facts} facts, {projs} projects, last run {last}{suffix}.")


def ambient_extract_now(_: str = "") -> str:
    """Synchronous one-shot run — useful for testing and on-demand summaries."""
    try:
        summary = _run_once()
    except Exception as e:
        return f"Ambient extraction failed, sir: {e}."
    return (f"Ambient extraction complete, sir — "
            f"{summary['mic_entries']} mic + "
            f"{summary['audio_entries']} system audio + "
            f"{summary['screen_entries']} screen entries processed; "
            f"{summary['facts_added']} new facts, "
            f"{summary['projects_added']} new projects.")


def register(actions: dict) -> None:
    actions["ambient_extract_start"]  = ambient_extract_start
    actions["ambient_extract_stop"]   = ambient_extract_stop
    actions["ambient_extract_status"] = ambient_extract_status
    actions["ambient_extract_now"]    = ambient_extract_now

    if _get_config("AMBIENT_EXTRACT_ENABLED", False):
        def _bg():
            try:
                time.sleep(4.0)
                ambient_extract_start("")
            except Exception as e:
                print(f"  [ambient-extract] autostart failed: {e}")
        threading.Thread(target=_bg,
                         name="ambient-extract-autostart",
                         daemon=True).start()
