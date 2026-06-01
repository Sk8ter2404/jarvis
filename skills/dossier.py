"""
JARVIS 'dossier' skill — the Stark-workshop "pull up the file on X" moment.

When the user says any of:
  • "pull up the file on a contact"
  • "what do you have on Apple Music"
  • "dossier on Bambu"
  • "the file on X" / "everything on X" / "show me what you have on X"

JARVIS aggregates everything it knows about the subject and slides a card
onto the top monitor titled  DOSSIER — <subject>  while reading a short
two-sentence summary aloud.

Sources combined for the subject X (all case-insensitive substring match):
  1. memory facts + projects from bobert_memory.json
  2. recent log lines from logs/session_*.log (last 8 files)
  3. open + closed entries in jarvis_todo.md mentioning X
  4. a fast DuckDuckGo Instant Answer abstract (no API key, short timeout)

Architecture:
  • The skill action returns a structured report string. Because the action
    is registered into INFORMATIVE_ACTIONS (patched in bobert_companion.py),
    the main loop fires a follow-up LLM call to phrase the JARVIS-voice
    reply from that report.
  • The slide-in HUD card is rendered by a subprocess of this same file
    invoked with --render, so the tkinter event loop doesn't fight the main
    JARVIS thread (same pattern hud_card.py + jarvis_hud.py use).

Public action names (all map to the same handler):
  dossier, pull_up_file, pull_up_dossier, file_on, dossier_on,
  what_do_you_have_on, whats_on_file
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from typing import Optional


# ─── paths ────────────────────────────────────────────────────────────────

_SKILL_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR  = os.path.dirname(_SKILL_DIR)
_MEMORY_FILE  = os.path.join(_PROJECT_DIR, "bobert_memory.json")
_TODO_FILE    = os.path.join(_PROJECT_DIR, "jarvis_todo.md")
_LOGS_DIR     = os.path.join(_PROJECT_DIR, "logs")
_STATE_FILE   = os.path.join(_PROJECT_DIR, "dossier_state.json")
_PID_FILE     = os.path.join(_PROJECT_DIR, "dossier_card.pid")


# ─── tunables ─────────────────────────────────────────────────────────────

CARD_DURATION_SECONDS = 25.0
MAX_FACTS_SHOWN       = 4
MAX_TASKS_SHOWN       = 4
MAX_LOG_LINES_SHOWN   = 4
LOG_SCAN_FILES        = 8       # newest N session logs to grep
LOG_LINE_MAX_LEN      = 140
# Cap bytes read per session log so a long-running day's log can't make the
# dossier compile slow (latency was growing unbounded with log size). We only
# need recent mentions, so reading the tail is sufficient.
LOG_TAIL_BYTES        = 262144  # 256 KB
WEB_TIMEOUT_SECONDS   = 4.0
SLIDE_DURATION_MS     = 380     # slide-in animation length

DDG_INSTANT_ANSWER_URL = (
    "https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&no_html=1"
)


# ─── shared synchronization ───────────────────────────────────────────────

_lock = threading.Lock()


# ─── aggregation helpers ──────────────────────────────────────────────────

def _normalize_topic(raw: str) -> str:
    """Strip leading filler words ('the', 'on', 'about') and surrounding
    punctuation from whatever the LLM passed as the dossier argument."""
    s = (raw or "").strip().strip("\"'.,;:!?")
    # Drop common conversational lead-ins.
    s = re.sub(
        r"^(?:the\s+file\s+on\s+|file\s+on\s+|dossier\s+on\s+|"
        r"on\s+|about\s+|the\s+)",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def _match(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    return needle.lower() in haystack.lower()


def _gather_memory(topic: str) -> list[str]:
    """Return memory facts + projects that mention the topic. Newest-first
    if order is preserved (the codebase appends new facts to the end)."""
    if not os.path.exists(_MEMORY_FILE):
        return []
    try:
        with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
            mem = json.load(f)
    except Exception:
        return []
    out: list[str] = []
    for item in reversed(mem.get("facts") or []):
        if isinstance(item, str) and _match(topic, item):
            out.append(item)
    for item in reversed(mem.get("projects") or []):
        if isinstance(item, str) and _match(topic, item):
            out.append(f"(project) {item}")
    return out


def _gather_tasks(topic: str) -> list[str]:
    """Return task lines mentioning the topic (both open and completed),
    open ones first so the card highlights what's still pending."""
    if not os.path.exists(_TODO_FILE):
        return []
    try:
        with open(_TODO_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    open_tasks: list[str] = []
    done_tasks: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not _match(topic, line):
            continue
        stripped = line.lstrip()
        if stripped.startswith("- [ ]"):
            open_tasks.append(_compact_task_line(stripped))
        elif stripped.startswith("- [x]") or stripped.startswith("- [X]"):
            done_tasks.append(_compact_task_line(stripped))
    return open_tasks + done_tasks


def _compact_task_line(line: str) -> str:
    """Trim the bullet/checkbox prefix and any DONE-tail so the card row stays
    readable; preserve the leading date stamp if present."""
    line = re.sub(r"^- \[[ xX]\]\s*", "", line)
    # Drop the trailing '✓ DONE — ...' summary so we just see the original task.
    line = re.sub(r"\s+✓\s*DONE\b.*$", "", line)
    return line.strip()


def _gather_logs(topic: str) -> list[str]:
    """Grep the newest LOG_SCAN_FILES session logs for the topic. Returns a
    list of '[HH:MM] ...trimmed line...' strings, newest log first."""
    if not os.path.isdir(_LOGS_DIR):
        return []
    try:
        files = sorted(
            (f for f in os.listdir(_LOGS_DIR) if f.endswith(".log")),
            reverse=True,
        )[:LOG_SCAN_FILES]
    except Exception:
        return []
    out: list[str] = []
    needle = topic.lower()
    for fname in files:
        path = os.path.join(_LOGS_DIR, fname)
        try:
            # Read only the last LOG_TAIL_BYTES of the file. Opening in binary
            # lets us seek by byte offset (text-mode seeks aren't reliable);
            # we decode the tail ourselves. This bounds per-call latency so a
            # large session log can't slow the whole dossier compile.
            with open(path, "rb") as fb:
                fb.seek(0, os.SEEK_END)
                size = fb.tell()
                seeked = size > LOG_TAIL_BYTES
                fb.seek(max(0, size - LOG_TAIL_BYTES))
                chunk = fb.read()
            text = chunk.decode("utf-8", errors="replace")
            lines = text.splitlines()
            # When we seeked into the middle of the file the first line is a
            # partial fragment — drop it so we don't emit a garbled mention.
            if seeked and lines:
                lines = lines[1:]
            for raw in lines:
                if needle not in raw.lower():
                    continue
                line = raw.rstrip("\n").strip()
                if len(line) > LOG_LINE_MAX_LEN:
                    line = line[: LOG_LINE_MAX_LEN - 1] + "…"
                out.append(line)
                if len(out) >= MAX_LOG_LINES_SHOWN * 2:
                    break
        except Exception:
            continue
        if len(out) >= MAX_LOG_LINES_SHOWN * 2:
            break
    return out[:MAX_LOG_LINES_SHOWN]


def _gather_web(topic: str) -> str:
    """Fast DuckDuckGo Instant Answer fetch. Returns '' on any failure so
    the dossier still works fully offline."""
    if not topic:
        return ""
    try:
        url = DDG_INSTANT_ANSWER_URL.format(q=urllib.parse.quote(topic))
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, timeout=WEB_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"  [dossier] web fetch failed: {e}")
        return ""
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        return _shorten_sentence(abstract, limit=320)
    # Fallback to the first RelatedTopics entry's text.
    related = data.get("RelatedTopics") or []
    if related and isinstance(related, list):
        first = related[0] or {}
        if isinstance(first, dict):
            t = (first.get("Text") or "").strip()
            if t:
                return _shorten_sentence(t, limit=320)
    return ""


def _shorten_sentence(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


# ─── two-sentence summary (what JARVIS reads) ─────────────────────────────

def _build_spoken_summary(topic: str, facts: list[str], tasks: list[str],
                          logs: list[str], web: str) -> str:
    """Compose a brief JARVIS-style two-sentence summary. Kept deterministic
    so the spoken delivery is the same shape every time."""
    n_facts = len(facts)
    n_tasks = len(tasks)
    n_logs  = len(logs)
    parts: list[str] = []
    if n_facts:
        parts.append(f"{n_facts} memory fact{'s' if n_facts != 1 else ''}")
    if n_tasks:
        parts.append(f"{n_tasks} task entr{'ies' if n_tasks != 1 else 'y'}")
    if n_logs:
        parts.append(f"{n_logs} log mention{'s' if n_logs != 1 else ''}")

    if parts:
        if len(parts) == 1:
            blob = parts[0]
        elif len(parts) == 2:
            blob = " and ".join(parts)
        else:
            blob = ", ".join(parts[:-1]) + ", and " + parts[-1]
        first = f"Pulling up the file on {topic}, sir — I've got {blob}."
    else:
        first = (
            f"I'm afraid I have very little on {topic}, sir — "
            f"nothing in memory, tasks, or recent logs."
        )

    # Second sentence: prefer a web abstract, then most-recent fact, then a
    # task highlight, then a dry fallback.
    if web:
        second = web if web.endswith(".") else web + "."
        second = _shorten_sentence(second, limit=220)
    elif facts:
        second = "Most recent note: " + _shorten_sentence(facts[0], limit=180)
        if not second.endswith("."):
            second += "."
    elif tasks:
        second = "Top of the queue: " + _shorten_sentence(tasks[0], limit=180)
        if not second.endswith("."):
            second += "."
    else:
        second = "Card displayed on the top monitor — that's the lot."

    return f"{first} {second}"


# ─── card state I/O ───────────────────────────────────────────────────────

def _top_monitor_geometry() -> tuple[int, int, int, int]:
    """Read MONITORS['top'] from the loaded bobert_companion module. Falls
    back to a sane default when running this file standalone for demo."""
    bc = sys.modules.get("bobert_companion") or sys.modules.get("__main__")
    try:
        m = getattr(bc, "MONITORS", None) if bc else None
        if isinstance(m, dict):
            if "top" in m:
                return tuple(int(v) for v in m["top"])  # type: ignore[return-value]
            if m:
                return tuple(int(v) for v in next(iter(m.values())))  # type: ignore[return-value]
    except Exception:
        pass
    return (0, 0, 1920, 1080)


def _write_state(state: dict) -> None:
    with _lock:
        dir_ = os.path.dirname(_STATE_FILE)
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f)
            os.replace(tmp, _STATE_FILE)
        except Exception:
            try: os.unlink(tmp)
            except Exception: pass
            raise


def _load_state_safe() -> Optional[dict]:
    if not os.path.exists(_STATE_FILE):
        return None
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ─── renderer subprocess management ───────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil   # type: ignore
        return psutil.pid_exists(pid)
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False


def _renderer_alive() -> bool:
    if not os.path.exists(_PID_FILE):
        return False
    try:
        with open(_PID_FILE, "r", encoding="utf-8") as f:
            pid = int((f.read() or "0").strip() or "0")
    except Exception:
        return False
    return _pid_alive(pid)


def _ensure_renderer_running() -> None:
    if _renderer_alive():
        # Already up — the existing renderer will pick up the new state on
        # its next 250 ms poll.
        return
    try:
        parent_pid = os.getpid()
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__),
             "--render", "--parent-pid", str(parent_pid)],
            creationflags=creationflags,
            close_fds=True,
        )
    except Exception as e:
        print(f"  [dossier] failed to spawn card renderer: {e}")


# ─── tkinter renderer ─────────────────────────────────────────────────────

def _renderer_main(parent_pid: int) -> int:
    import tkinter as tk

    try:
        with open(_PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    state = _load_state_safe()
    if not state:
        return 0

    geom = state.get("geometry") or [0, 0, 1920, 1080]
    try:
        mon_x, mon_y, mon_w, mon_h = (int(v) for v in geom[:4])
    except (TypeError, ValueError):
        mon_x, mon_y, mon_w, mon_h = 0, 0, 1920, 1080

    CARD_W = max(640, int(mon_w * 0.55))
    CARD_H = max(420, int(mon_h * 0.70))
    FINAL_X = mon_x + (mon_w - CARD_W) // 2
    CARD_Y  = mon_y + int(mon_h * 0.10)
    START_X = mon_x + mon_w                # off-screen right
    BG       = "#04080d"
    BORDER   = "#4cc9ff"
    TITLE_FG = "#9ee7ff"
    SECT_FG  = "#5d8aa3"
    TEXT_FG  = "#cfeefb"
    DIM_FG   = "#6f8aa0"
    GOLD     = "#ffd166"

    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry(f"{CARD_W}x{CARD_H}+{START_X}+{CARD_Y}")
    root.attributes("-topmost", True)
    try:
        root.attributes("-alpha", 0.95)
    except Exception:
        pass
    root.configure(bg=BG)

    border_frame = tk.Frame(root, bg=BORDER, padx=2, pady=2)
    border_frame.pack(fill="both", expand=True)
    inner = tk.Frame(border_frame, bg=BG)
    inner.pack(fill="both", expand=True, padx=2, pady=2)

    topic = state.get("topic") or "Subject"
    title = f"DOSSIER  —  {topic.upper()}"
    tk.Label(
        inner, text=title,
        font=("Segoe UI", 22, "bold"), fg=TITLE_FG, bg=BG,
        anchor="w",
    ).pack(fill="x", padx=24, pady=(18, 4))
    tk.Label(
        inner, text=time.strftime("%A, %B %d  •  %H:%M"),
        font=("Segoe UI", 11), fg=DIM_FG, bg=BG, anchor="w",
    ).pack(fill="x", padx=24, pady=(0, 14))

    body = tk.Frame(inner, bg=BG)
    body.pack(fill="both", expand=True, padx=24)

    def _add_section(title_text: str, lines: list[str], empty_text: str,
                     accent: str = TEXT_FG) -> None:
        tk.Label(
            body, text=title_text,
            font=("Segoe UI", 10, "bold"), fg=SECT_FG, bg=BG, anchor="w",
        ).pack(fill="x", pady=(12, 4))
        if lines:
            for ln in lines:
                tk.Label(
                    body, text="•  " + ln,
                    font=("Segoe UI", 12), fg=accent, bg=BG,
                    anchor="w", justify="left", wraplength=CARD_W - 80,
                ).pack(fill="x", padx=(4, 0), pady=1)
        else:
            tk.Label(
                body, text=empty_text,
                font=("Segoe UI", 12, "italic"), fg=DIM_FG, bg=BG,
                anchor="w",
            ).pack(fill="x", padx=(4, 0))

    _add_section("MEMORY",
                 (state.get("facts") or [])[:MAX_FACTS_SHOWN],
                 "Nothing in memory.")
    _add_section("TASK QUEUE",
                 (state.get("tasks") or [])[:MAX_TASKS_SHOWN],
                 "No queued tasks reference this subject.",
                 accent=GOLD)
    _add_section("RECENT LOGS",
                 (state.get("logs") or [])[:MAX_LOG_LINES_SHOWN],
                 "No log mentions in the recent sessions.")

    web_blob = (state.get("web") or "").strip()
    tk.Label(
        body, text="WEB",
        font=("Segoe UI", 10, "bold"), fg=SECT_FG, bg=BG, anchor="w",
    ).pack(fill="x", pady=(12, 4))
    tk.Label(
        body, text=web_blob or "No web abstract available.",
        font=("Segoe UI", 12), fg=TEXT_FG if web_blob else DIM_FG, bg=BG,
        anchor="w", justify="left", wraplength=CARD_W - 80,
    ).pack(fill="x", padx=(4, 0))

    hint = tk.Label(
        inner, text="", font=("Segoe UI", 10), fg=DIM_FG, bg=BG,
    )
    hint.pack(side="bottom", pady=10)

    # ─── slide-in animation ───
    slide_steps = max(8, int(SLIDE_DURATION_MS / 20))
    state_anim = {"step": 0}

    def _ease(t: float) -> float:
        # ease-out cubic
        return 1 - (1 - t) ** 3

    def _slide():
        if state_anim["step"] >= slide_steps:
            return
        state_anim["step"] += 1
        t = state_anim["step"] / slide_steps
        x = int(START_X + (FINAL_X - START_X) * _ease(t))
        root.geometry(f"{CARD_W}x{CARD_H}+{x}+{CARD_Y}")
        root.after(20, _slide)

    root.after(30, _slide)

    # ─── auto-dismiss + parent-watchdog tick ───
    def _tick():
        try:
            cur = _load_state_safe()
            now = time.time()
            if cur is None or cur.get("dismissed"):
                root.destroy()
                return
            expiry = float(cur.get("expiry_ts", 0.0))
            if now >= expiry:
                root.destroy()
                return
            if parent_pid > 0 and not _pid_alive(parent_pid):
                root.destroy()
                return
            rem = max(0, int(expiry - now))
            hint.config(text=f"Auto-dismiss in {rem}s")
            root.after(250, _tick)
        except Exception:
            try: root.destroy()
            except Exception: pass

    root.after(50, _tick)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        try: os.remove(_PID_FILE)
        except Exception: pass
    return 0


# ─── compile_dossier (callable from other skills) ─────────────────────────

def compile_dossier(topic_raw: str) -> dict:
    """Aggregate everything JARVIS knows about a subject. Returned dict has
    keys: topic, facts[], tasks[], logs[], web, summary."""
    topic = _normalize_topic(topic_raw)
    if not topic:
        return {
            "topic": "",
            "facts": [],
            "tasks": [],
            "logs":  [],
            "web":   "",
            "summary": "I need a subject for the dossier, sir.",
        }
    facts = _gather_memory(topic)
    tasks = _gather_tasks(topic)
    logs  = _gather_logs(topic)
    web   = _gather_web(topic)
    summary = _build_spoken_summary(topic, facts, tasks, logs, web)
    return {
        "topic": topic,
        "facts": facts,
        "tasks": tasks,
        "logs":  logs,
        "web":   web,
        "summary": summary,
    }


# ─── action handler ───────────────────────────────────────────────────────

def _act_dossier(arg: str) -> str:
    data = compile_dossier(arg)
    topic = data["topic"]
    if not topic:
        return data["summary"]

    # Show the card.
    try:
        geom = _top_monitor_geometry()
        state = {
            "topic":     topic,
            "facts":     data["facts"],
            "tasks":     data["tasks"],
            "logs":      data["logs"],
            "web":       data["web"],
            "summary":   data["summary"],
            "geometry":  list(geom),
            "shown_at":  time.time(),
            "expiry_ts": time.time() + CARD_DURATION_SECONDS,
            "dismissed": False,
        }
        _write_state(state)
        _ensure_renderer_running()
    except Exception as e:
        print(f"  [dossier] card render failed: {e}")

    # Build the result string the follow-up LLM will phrase aloud.
    lines: list[str] = [f"DOSSIER on '{topic}':"]
    if data["facts"]:
        lines.append("  facts (newest first):")
        for f in data["facts"][:MAX_FACTS_SHOWN]:
            lines.append(f"    - {_shorten_sentence(f, 220)}")
    else:
        lines.append("  facts: (none)")
    if data["tasks"]:
        lines.append("  tasks (open first):")
        for t in data["tasks"][:MAX_TASKS_SHOWN]:
            lines.append(f"    - {_shorten_sentence(t, 220)}")
    else:
        lines.append("  tasks: (none)")
    if data["logs"]:
        lines.append("  recent log mentions:")
        for ln in data["logs"][:MAX_LOG_LINES_SHOWN]:
            lines.append(f"    - {_shorten_sentence(ln, 220)}")
    else:
        lines.append("  recent log mentions: (none)")
    if data["web"]:
        lines.append("  web abstract: " + _shorten_sentence(data["web"], 280))
    else:
        lines.append("  web abstract: (unavailable)")
    lines.append(f"  HUD card displayed on the top monitor.")
    lines.append(f"  Spoken summary: {data['summary']}")
    return "\n".join(lines)


# ─── registration ─────────────────────────────────────────────────────────

def register(actions: dict) -> None:
    actions["dossier"]              = _act_dossier
    actions["pull_up_file"]         = _act_dossier
    actions["pull_up_dossier"]      = _act_dossier
    actions["file_on"]              = _act_dossier
    actions["dossier_on"]           = _act_dossier
    actions["what_do_you_have_on"]  = _act_dossier
    actions["whats_on_file"]        = _act_dossier

    # Patch the host module's INFORMATIVE_ACTIONS set so the main loop fires
    # a follow-up LLM call (turning the structured report into a JARVIS-voice
    # reply) instead of going silent after the action runs.
    _dossier_names = {
        "dossier", "pull_up_file", "pull_up_dossier", "file_on",
        "dossier_on", "what_do_you_have_on", "whats_on_file",
    }
    try:
        import bobert_companion as _bc  # type: ignore
        info = getattr(_bc, "INFORMATIVE_ACTIONS", None)
        if isinstance(info, set):
            info.update(_dossier_names)
        # Dossier compile is one of the four spec-named long-running flows
        # (auto-play streaming / upgrade / overnight / dossier compile) —
        # register every alias as long-running so the dispatcher's
        # mid_task_status timer bridges the 15-30 s LLM + web-fetch + card
        # render pipeline with a dry "Still compiling, sir." line.
        long_running = getattr(_bc, "LONG_RUNNING_ACTIONS", None)
        if isinstance(long_running, set):
            long_running.update(_dossier_names)
        bucket_map = getattr(_bc, "_MID_TASK_STATUS_BUCKET", None)
        if isinstance(bucket_map, dict):
            for _alias in _dossier_names:
                bucket_map[_alias] = "dossier"
    except Exception as e:
        print(f"  [dossier] couldn't extend INFORMATIVE_ACTIONS: {e}")


# ─── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JARVIS dossier card")
    parser.add_argument("--render", action="store_true",
                        help="Renderer subprocess mode (read state, draw card)")
    parser.add_argument("--parent-pid", type=int, default=0,
                        help="Exit when this PID dies")
    parser.add_argument("--demo", type=str, default="",
                        help="Compile + show a dossier for the given topic")
    args = parser.parse_args()

    if args.demo:
        result = _act_dossier(args.demo)
        print(result)
        # Keep the parent alive long enough for the subprocess renderer
        # to read the state file before this CLI exits.
        end = time.time() + CARD_DURATION_SECONDS + 1.0
        while time.time() < end and _renderer_alive():
            time.sleep(0.5)
        sys.exit(0)
    elif args.render:
        sys.exit(_renderer_main(args.parent_pid))
    else:
        parser.print_help()
