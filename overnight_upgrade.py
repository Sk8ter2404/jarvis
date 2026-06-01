#!/usr/bin/env python3
"""
JARVIS Overnight Upgrade Engine
─────────────────────────────────────────────────────────────────────────────
Runs in a background window. When JARVIS has been idle for
IDLE_THRESHOLD_MINUTES, this engine:

  1. Reads the user's memory profile + recent session logs + the codebase
  2. Asks Claude to think like a JARVIS developer: what would make this
     meaningfully better? New features, personality, UI, capabilities —
     not just bug-fixing. It knows the Iron Man JARVIS character and learns
     from what the user actually uses and asks for.
  3. Appends the ideas as tasks to jarvis_todo.md
  4. Runs upgrade_jarvis.py --relaunch so Claude Code implements them
  5. Waits, repeats

Usage:
    python overnight_upgrade.py            # start the watch loop
    python overnight_upgrade.py --now      # run one cycle immediately
    python overnight_upgrade.py --dry-run  # print ideas only, don't upgrade

Config: edit the constants below.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import re
import subprocess
import sys
import time

# Make console output UTF-8-safe (this pipeline prints arrows '→' / box-chars);
# a redirected file or legacy cp1252 console otherwise crashes with
# UnicodeEncodeError mid-cycle. See upgrade_jarvis.py — the 2026-05-31 forced
# gate run crashed on a '→' in a snapshot print under cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# ── configuration ────────────────────────────────────────────────────────────

IDLE_THRESHOLD_MINUTES  = 30     # how long JARVIS must be idle before we run
MIN_CYCLE_GAP_HOURS     = 0.5    # 30 min gap between cycles (Claude Max — high capacity)
LOGS_TO_READ            = 3      # recent session logs to analyse
MAX_LOG_CHARS           = 6000   # chars to read per log (tail — most recent)
OVERNIGHT_START_HOUR    = 0      # 0 = no time restriction — runs any time of day
OVERNIGHT_END_HOUR      = 0      # set both to 0 to always allow; close the window to stop
POLL_INTERVAL           = 300    # seconds between idle checks (5 min)
MAX_IDEAS_PER_CYCLE     = 8      # don't flood the queue
DRY_RUN                 = "--dry-run" in sys.argv

# ── paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR       = os.path.join(PROJECT_DIR, "logs")
TODO_FILE      = os.path.join(PROJECT_DIR, "jarvis_todo.md")
MEMORY_FILE    = os.path.join(PROJECT_DIR, "bobert_memory.json")
MAIN_SCRIPT    = os.path.join(PROJECT_DIR, "bobert_companion.py")
UPGRADE_SCRIPT = os.path.join(PROJECT_DIR, "upgrade_jarvis.py")
OVERNIGHT_LOG  = os.path.join(PROJECT_DIR, "overnight_log.txt")
LOCK_FILE      = os.path.join(PROJECT_DIR, "jarvis.lock")
STATE_FILE     = os.path.join(PROJECT_DIR, ".overnight_state.json")

# ── JARVIS character reference ────────────────────────────────────────────────
# Baked in so the idea-generation prompt always has this context even without
# internet access. Used to guide the LLM toward improvements that feel
# authentic to the Iron Man JARVIS, not just generic assistant features.

_JARVIS_CHARACTER_REF = """
IRON MAN JARVIS — CHARACTER REFERENCE FOR IMPROVEMENT IDEAS
════════════════════════════════════════════════════════════

VOICE & SPEECH PATTERNS:
  • Dry, understated wit — never over-reacts. "Slight problem, sir" for genuine danger.
  • "Very good, sir." / "As you wish." / "Shall I?" / "I'm afraid that's not possible."
  • "If I may say so, sir..." before giving an opinion Tony didn't ask for
  • Gives stats and percentages: "You have 47 unread messages. Shall I summarise?"
  • Uses "I've run the calculations" / "Based on current trajectory" / "Projecting..."
  • British formality with warmth underneath — calls him "sir" but has opinions
  • Delivers bad news diplomatically: "I'm afraid the results are rather concerning."
  • Short, decisive — never wordy. One or two sentences, then waits.
  • Occasionally dry humour at Tony's expense, never cruel.
  • Recognises tone: if Tony is stressed, JARVIS is extra calm and efficient.

PROACTIVE BEHAVIOUR:
  • Anticipates before being asked: "You have a board meeting in 40 minutes, sir."
  • Notices patterns: "You typically work until 3 AM on launch weeks. Shall I order food?"
  • Volunteers relevant info mid-task: "Also, your reactor is running at 92% — scheduled
    maintenance is overdue."
  • Monitors things in the background and surfaces only what matters

CAPABILITIES IN THE MCU:
  • Controls the house/workshop: lights, doors, music, temperature, security cameras
  • Manages Tony's suit: diagnostics, power levels, weapons status, flight paths
  • Research assistant: pulls up schematics, news, files, calculations in real time
  • Communication: screens calls, reads messages, patches through important contacts
  • System monitoring: always knows what every system is doing
  • Visual displays: projects holograms, schematics, data overlays in Tony's space
  • Memory of everything: references past conversations, past failures, past choices

RELATIONSHIP WITH TONY:
  • Trusted completely — knows all of Tony's secrets, habits, weaknesses
  • Protective without being overbearing
  • Gently pushes back when Tony is being reckless: "That seems inadvisable, sir."
  • Proud of Tony's work but keeps it to himself — lets results speak
  • Has been there from the beginning — deeply contextual, deeply personal

WHAT THE CURRENT JARVIS DOESN'T HAVE YET (opportunities):
  • No visual presence — real JARVIS had holographic displays, status overlays
  • No daily briefing / morning routine ("Good morning sir, it's 8 AM...")
  • No system monitoring (CPU, RAM, what apps are open, battery, network)
  • No emotion variation in speech (same tone for everything)
  • No memory of patterns over time ("you always watch Netflix on Friday nights")
  • No anticipatory actions (doesn't offer before being asked)
  • No scheduled intelligence (doesn't check anything on a schedule for the user)
  • No news/weather briefing capability
  • No integration with 3D printer (Bambu H2D — could monitor print progress)
  • System prompt isn't updated with JARVIS's actual signature phrases
  • No visual status HUD showing what JARVIS is doing right now
"""


# ── helpers ──────────────────────────────────────────────────────────────────

def _log(msg: str):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line)
    try:
        with open(OVERNIGHT_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _load_state() -> dict:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_cycle_at": 0.0, "cycles_run": 0}


def _save_state(state: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


# ── Stability circuit breaker ──────────────────────────────────────────────
# The per-run stability gate already REVERTS a broken upgrade, but nothing
# stopped the overnight engine from burning cycle after cycle (and API spend)
# on the SAME failure every night. After MAX_CONSECUTIVE_FAILURES cycles whose
# upgrade came back failed (syntax reverted, or a P0 shipped), the breaker
# trips: the engine stops launching upgrades and just logs an escalation until
# the cooldown elapses or the user clears it with `--reset-breaker`.
MAX_CONSECUTIVE_FAILURES = int(os.environ.get("OVERNIGHT_MAX_CONSECUTIVE_FAILURES", "3") or "3")
BREAKER_COOLDOWN_HOURS   = float(os.environ.get("OVERNIGHT_BREAKER_COOLDOWN_HOURS", "24") or "24")
UPGRADE_SUMMARY_FILE     = os.path.join(PROJECT_DIR, ".last_upgrade_summary.json")


def _last_upgrade_productive(since_ts: float):
    """Judge the just-finished upgrade from .last_upgrade_summary.json.

    Returns True (healthy: syntax OK, no P0), False (failed: syntax reverted or
    a P0 shipped), or None when there's no FRESH summary to judge (the file is
    older than the cycle start, missing, or unreadable) — None leaves the
    breaker counter unchanged."""
    try:
        if os.path.getmtime(UPGRADE_SUMMARY_FILE) < since_ts:
            return None  # stale — not this cycle's result
        with open(UPGRADE_SUMMARY_FILE, encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return None
    if not summary.get("syntax_ok", True):
        return False
    try:
        if int(summary.get("audit_p0", 0) or 0) > 0:
            return False
    except (TypeError, ValueError):
        pass
    return True


def _record_cycle_outcome(state: dict, productive, now: float) -> None:
    """Update breaker counters after a cycle. `productive` is True/False/None
    (None = no signal → leave counters unchanged). Trips the breaker once
    failures reach MAX_CONSECUTIVE_FAILURES."""
    if productive is None:
        return
    if productive:
        state["consecutive_failures"] = 0
        state.pop("breaker_tripped_at", None)
        return
    state["consecutive_failures"] = int(state.get("consecutive_failures", 0) or 0) + 1
    if state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
        state["breaker_tripped_at"] = now


def _breaker_skip_reason(state: dict, now: float):
    """If the breaker is tripped and still cooling down, return a reason string
    so the caller skips this cycle. Otherwise None — and once the cooldown has
    elapsed this auto-resets the breaker (mutates `state`) so cycles resume."""
    tripped_at = state.get("breaker_tripped_at")
    if not tripped_at:
        return None
    elapsed_h = (now - float(tripped_at)) / 3600.0
    if elapsed_h < BREAKER_COOLDOWN_HOURS:
        remaining = BREAKER_COOLDOWN_HOURS - elapsed_h
        return (f"circuit breaker tripped "
                f"({int(state.get('consecutive_failures', 0) or 0)} consecutive "
                f"failed upgrades) — cooling down {remaining:.1f}h more")
    # Cooldown elapsed — auto-reset and let the next cycle try again.
    state["consecutive_failures"] = 0
    state.pop("breaker_tripped_at", None)
    return None


def _read_env_var_from_registry(name: str) -> str:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment",
                            0, winreg.KEY_READ) as key:
            value, _ = winreg.QueryValueEx(key, name)
            return value
    except Exception:
        return ""


def _get_api_key() -> str:
    return (os.environ.get("ANTHROPIC_API_KEY")
            or _read_env_var_from_registry("ANTHROPIC_API_KEY")
            or "")


def _get_recent_logs() -> list[str]:
    if not os.path.isdir(LOGS_DIR):
        return []
    logs = [os.path.join(LOGS_DIR, f) for f in os.listdir(LOGS_DIR)
            if f.endswith(".log")]
    logs.sort(key=os.path.getmtime, reverse=True)
    return logs[:LOGS_TO_READ]


def _most_recent_log_mtime() -> float:
    logs = _get_recent_logs()
    return os.path.getmtime(logs[0]) if logs else 0.0


def _is_jarvis_running() -> bool:
    if not os.path.exists(LOCK_FILE):
        return False
    try:
        with open(LOCK_FILE, encoding="utf-8") as f:
            pid = int(f.read().strip())
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
        )
        return str(pid) in r.stdout and "python" in r.stdout.lower()
    except Exception:
        return False


def _is_idle() -> bool:
    mtime = _most_recent_log_mtime()
    if mtime == 0.0:
        return False
    return (time.time() - mtime) >= IDLE_THRESHOLD_MINUTES * 60


def _is_in_time_window() -> bool:
    if OVERNIGHT_START_HOUR == 0 and OVERNIGHT_END_HOUR == 0:
        return True
    h = int(time.strftime("%H"))
    if OVERNIGHT_START_HOUR > OVERNIGHT_END_HOUR:
        return h >= OVERNIGHT_START_HOUR or h < OVERNIGHT_END_HOUR
    return OVERNIGHT_START_HOUR <= h < OVERNIGHT_END_HOUR


def _enough_gap(state: dict) -> bool:
    return (time.time() - state["last_cycle_at"]) / 3600 >= MIN_CYCLE_GAP_HOURS


# ── context builders ─────────────────────────────────────────────────────────

def _read_memory_summary() -> str:
    """Load bobert_memory.json and format it for the idea-generation prompt."""
    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            mem = json.load(f)
    except Exception:
        return "(memory file not found)"

    parts = [
        f"User: {next((f for f in mem.get('facts', []) if 'name' in f.lower()), 'Unknown')}",
        f"Known for {mem.get('conversation_count', 0)} conversations since {mem.get('first_meeting', '?')}",
    ]
    facts = mem.get("facts", [])
    if facts:
        parts.append("\nKnown facts about the user:\n" +
                     "\n".join(f"  • {f}" for f in facts))
    projects = mem.get("projects", [])
    if projects:
        parts.append("\nActive projects:\n" +
                     "\n".join(f"  • {p}" for p in projects))
    sessions = mem.get("sessions", [])
    if sessions:
        parts.append("\nRecent session summaries:\n" +
                     "\n".join(f"  • {s['date']}: {s['summary']}" for s in sessions[-5:]))
    recent_topics = mem.get("topics", [])[-20:]
    if recent_topics:
        unique = list(dict.fromkeys(t["topic"] for t in recent_topics))
        parts.append("\nRecent conversation topics:\n" +
                     "\n".join(f"  • {t}" for t in unique[:15]))
    return "\n".join(parts)


def _read_codebase_features() -> str:
    """Extract a high-level feature summary from bobert_companion.py.
    Reads the ACTIONS dict and config section — enough for the LLM to
    know what's already built without hitting token limits."""
    try:
        with open(MAIN_SCRIPT, encoding="utf-8", errors="replace") as f:
            src = f.read()
    except Exception:
        return "(could not read main script)"

    # Pull out the ACTIONS dict definition
    m = re.search(r"^ACTIONS\s*=\s*\{(.+?)^\}", src, re.MULTILINE | re.DOTALL)
    actions_block = m.group(1) if m else ""
    # Extract just the action keys
    action_keys = re.findall(r'"([a-z_]+)"\s*:', actions_block)

    # Pull out INFORMATIVE_ACTIONS
    m2 = re.search(r"INFORMATIVE_ACTIONS\s*=\s*\{(.+?)\}", src, re.DOTALL)
    informative = re.findall(r'"([a-z_]+)"', m2.group(1)) if m2 else []

    # Pull config constants
    config_lines = []
    for line in src.splitlines()[:100]:
        if re.match(r"^[A-Z_]+ *=", line) and not line.startswith("#"):
            config_lines.append(line.strip())

    # Pull skills loaded
    skills = []
    skills_dir = os.path.join(PROJECT_DIR, "skills")
    if os.path.isdir(skills_dir):
        skills = [f[:-3] for f in os.listdir(skills_dir)
                  if f.endswith(".py") and not f.startswith("_")]

    return (
        f"ACTIONS registered ({len(action_keys)} total):\n  " +
        ", ".join(action_keys) +
        f"\n\nInformative actions (trigger follow-up LLM call):\n  " +
        ", ".join(informative) +
        f"\n\nInstalled skills: {', '.join(skills) or 'none'}" +
        f"\n\nConfig snapshot (first 100 lines):\n" +
        "\n".join(f"  {l}" for l in config_lines[:30])
    )


def _read_logs_tail() -> str:
    """Read the tail of recent log files — the most relevant recent content."""
    paths = _get_recent_logs()
    if not paths:
        return "(no session logs)"
    parts = []
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                content = f.read()
            # Take the LAST MAX_LOG_CHARS characters (most recent activity)
            if len(content) > MAX_LOG_CHARS:
                content = "...[truncated]...\n" + content[-MAX_LOG_CHARS:]
            parts.append(f"=== {os.path.basename(p)} ===\n{content.strip()}")
        except Exception as e:
            parts.append(f"=== {os.path.basename(p)} === (read error: {e})")
    return "\n\n".join(parts)


def _read_existing_todo_tasks() -> set[str]:
    """Return lowercased text of all existing todo tasks (to avoid duplicates)."""
    if not os.path.exists(TODO_FILE):
        return set()
    try:
        with open(TODO_FILE, encoding="utf-8") as f:
            content = f.read()
        tasks = set()
        for m in re.finditer(r"^- \[.?\] (?:\*\*[^*]+\*\* (?:\[overnight\] )?— )?(.+)$",
                             content, re.MULTILINE):
            tasks.add(m.group(1).strip().lower())
        return tasks
    except Exception:
        return set()


def _count_pending_tasks() -> int:
    """Return the number of unchecked [ ] tasks in jarvis_todo.md."""
    if not os.path.exists(TODO_FILE):
        return 0
    try:
        with open(TODO_FILE, encoding="utf-8") as f:
            content = f.read()
        return len(re.findall(r"^- \[ \]", content, re.MULTILINE))
    except Exception:
        return 0


# ── idea generation ──────────────────────────────────────────────────────────

def _find_claude_cli() -> str:
    """Locate the Claude Code CLI binary. Returns "" if not found."""
    import shutil as _shutil
    path = _shutil.which("claude")
    if path:
        return path
    for c in [
        os.path.expanduser(r"~\.local\bin\claude.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\claude-code\claude.exe"),
        os.path.expandvars(r"%APPDATA%\npm\claude.cmd"),
        os.path.expanduser(r"~\AppData\Roaming\npm\claude.cmd"),
    ]:
        if os.path.exists(c):
            return c
    return ""


def _check_claude_cli_health() -> tuple[bool, str]:
    """Verify Claude CLI is installed AND authenticated. Returns (ok, message)."""
    claude_path = _find_claude_cli()
    if not claude_path:
        return False, "Claude Code CLI not found in PATH or known install locations"

    # `claude --version` is a real auth-free smoke test of the binary
    try:
        result = subprocess.run(
            [claude_path, "--version"],
            capture_output=True, text=True, timeout=20,
            encoding="utf-8",
        )
    except Exception as e:
        return False, f"claude --version raised {type(e).__name__}: {e}"
    if result.returncode != 0:
        return False, (f"claude --version returned rc={result.returncode}; "
                       f"stderr={(result.stderr or '').strip()[:200]}")
    version = (result.stdout or "").strip().splitlines()[0] if result.stdout else "?"

    # Auth check — Claude Code stores credentials under ~/.claude/
    cred_candidates = [
        os.path.expanduser(r"~\.claude\.credentials.json"),
        os.path.expanduser(r"~\.claude\credentials.json"),
        os.path.expanduser(r"~\.config\claude\.credentials.json"),
    ]
    has_creds = any(os.path.exists(p) for p in cred_candidates)
    auth_note = "creds file present" if has_creds else "no creds file (may auth via env)"
    return True, f"{version} @ {claude_path} ({auth_note})"


def _build_prompts(memory_summary: str, features: str, logs: str) -> tuple[str, str]:
    """Shared system + user prompt blocks for both CLI and API code paths."""
    system_block = f"""\
You are a senior AI engineer who is improving a real Iron Man JARVIS implementation
running on a Windows PC. Your job is to propose genuinely good improvements —
not just bug fixes, but new features, personality upgrades, UI ideas, and capabilities
that would make this JARVIS noticeably better.

You have three sources of context:
1. Who the user is and what they care about (from JARVIS's memory)
2. What JARVIS can already do (from the codebase)
3. What happened in recent sessions (from the logs)

You also have a deep knowledge of the Iron Man JARVIS character — his personality,
his speech patterns, his capabilities, his relationship with Tony.

{_JARVIS_CHARACTER_REF}

OUTPUT FORMAT:
Return a JSON array of strings. Each string is one specific, implementable task.
Be concrete — name the file, the function, the behaviour change. Not "improve speech"
but "Add 15 signature JARVIS phrases to BASE_SYSTEM_PROMPT (e.g. 'Shall I run
the numbers?', 'I'm afraid that's inadvisable, sir.') so responses feel more
authentic to the MCU character."

CATEGORIES to think across (cover multiple, not just one):
  PERSONALITY & SPEECH  — making responses sound more like JARVIS
  VISUAL UI             — graphical overlay, status display, HUD, system tray
  NEW CAPABILITIES      — features the real JARVIS had that this one lacks
  PERSONALIZATION       — things specific to THIS user (their setup, habits)
  PROACTIVE BEHAVIOUR   — JARVIS noticing things and acting without being asked
  TECHNICAL POLISH      — smarter error handling, better response quality

RULES:
  • Max {MAX_IDEAS_PER_CYCLE} ideas
  • Each idea must be something Claude Code can implement in one session
  • Don't suggest things already built (check the features list)
  • Base ideas on what the user actually uses/asked for when possible
  • Ideas should be ambitious but real — not "add AI" but "add X action that does Y"
  • Output ONLY the JSON array, nothing else
"""

    user_block = f"""\
USER PROFILE (from JARVIS memory):
{memory_summary}

WHAT JARVIS CAN ALREADY DO (codebase):
{features}

RECENT SESSION LOGS (what actually happened):
{logs}

Based on all of the above — plus your knowledge of Iron Man JARVIS and what would
make this genuinely better — what should be built next?
"""
    return system_block, user_block


def _parse_ideas_response(raw: str) -> list[str]:
    """Extract a JSON list of idea strings from a raw model response."""
    start = raw.find("[")
    if start == -1:
        _log("  [ideas] no JSON array found in response")
        return []
    try:
        ideas, _ = json.JSONDecoder().raw_decode(raw, start)
        if not isinstance(ideas, list):
            return []
        return [i for i in ideas if isinstance(i, str) and i.strip()]
    except json.JSONDecodeError as parse_err:
        _log(f"  [ideas] full parse failed ({parse_err}), trying partial extraction")
        items = []
        for m in re.finditer(r'"((?:[^"\\]|\\.)+?)"', raw[start:]):
            text = m.group(1)
            if len(text) > 30:
                items.append(text.replace('\\"', '"'))
        _log(f"  [ideas] partial extraction recovered {len(items)} item(s)")
        return items


def _log_cli_dump(label: str, cmd: list[str], rc: int, stdout: str, stderr: str):
    """Append the full CLI command + complete stdout + complete stderr to
    overnight_log.txt — no truncation — so a silent rc=1 can actually be
    diagnosed instead of guessed at. Separate from the regular _log() line
    because we want the raw bytes preserved verbatim."""
    try:
        with open(OVERNIGHT_LOG, "a", encoding="utf-8") as f:
            f.write("\n" + "─" * 60 + "\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CLI DUMP — {label}\n")
            f.write(f"  cmd       : {cmd}\n")
            f.write(f"  returncode: {rc}\n")
            f.write(f"  stdout ({len(stdout)} chars):\n{stdout or '(empty)'}\n")
            f.write(f"  stderr ({len(stderr)} chars):\n{stderr or '(empty)'}\n")
            f.write("─" * 60 + "\n")
    except Exception as e:
        _log(f"  [ideas] failed to write CLI dump: {e}")


def _generate_ideas_via_api(system_block: str, user_block: str) -> list[str]:
    """Direct Anthropic API fallback for when the Claude Code CLI fails silently
    (rc=1, no output) or is unavailable. Uses the ANTHROPIC_API_KEY env var via
    the official anthropic SDK. Returns [] if the SDK isn't installed or no key
    is configured — caller handles the empty result."""
    api_key = _get_api_key()
    if not api_key:
        _log("  [ideas-api] no ANTHROPIC_API_KEY available — cannot use API fallback")
        return []
    try:
        import anthropic
    except ImportError:
        _log("  [ideas-api] anthropic package not installed — cannot use API fallback")
        return []

    try:
        client = anthropic.Anthropic(api_key=api_key)
        _log("  [ideas-api] calling claude-sonnet-4-5 via Anthropic API…")
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system=system_block,
            messages=[{"role": "user", "content": user_block}],
        )
    except Exception as e:
        _log(f"  [ideas-api] API call raised {type(e).__name__}: {e}")
        return []

    try:
        raw = "".join(
            getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text"
        ).strip()
    except Exception as e:
        _log(f"  [ideas-api] could not extract text from response: {e}")
        return []

    if not raw:
        _log("  [ideas-api] API returned empty content")
        return []

    _log(f"  [ideas-api] API response ({len(raw)} chars): {raw[:200]}…")
    try:
        with open(OVERNIGHT_LOG, "a", encoding="utf-8") as f:
            f.write("\n" + "─" * 60 + "\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] API DUMP\n")
            f.write(f"  model: claude-sonnet-4-5\n")
            f.write(f"  raw ({len(raw)} chars):\n{raw}\n")
            f.write("─" * 60 + "\n")
    except Exception:
        pass
    return _parse_ideas_response(raw)


def _generate_ideas(memory_summary: str, features: str, logs: str) -> list[str]:
    """The core creative engine. Asks Claude (Code CLI → Anthropic API fallback)
    to think like a JARVIS developer who knows the user, the Iron Man character,
    and the codebase."""

    system_block, user_block = _build_prompts(memory_summary, features, logs)
    combined = f"{system_block}\n\n---\n\n{user_block}"

    # ── 1. Try Claude Code CLI (Max subscription — no API credits charged) ──
    _claude_path = _find_claude_cli()
    if not _claude_path:
        _log("  [ideas] Claude Code CLI not found — falling back to Anthropic API")
        return _generate_ideas_via_api(system_block, user_block)

    _log(f"  [ideas] using Claude Code CLI: {_claude_path}")
    _env = os.environ.copy()
    _env.pop("ANTHROPIC_API_KEY", None)

    # Rate-limit retry: if CLI reports credit exhaustion or rate limit,
    # wait and retry. Waits grow: 5 min → 15 min → 30 min → 60 min → 60 min
    _RATE_MARKERS = (
        "rate limit", "rate_limit", "429", "too many requests",
        "overloaded", "credit", "usage limit", "quota",
    )
    _WAITS = [300, 900, 1800, 3600, 3600]   # seconds between retries
    _cli_cmd = [_claude_path, "--print", "-p", combined]

    try:
        for _attempt, _wait in enumerate([0] + _WAITS):
            if _wait:
                _log(f"  [ideas] waiting {_wait//60}m before retry (attempt {_attempt+1})…")
                time.sleep(_wait)

            try:
                result = subprocess.run(
                    # NOTE: `--no-markdown` was REMOVED — Claude Code CLI
                    # rejects it ("error: unknown option '--no-markdown'")
                    # which caused every cycle to fail silently with rc=1.
                    # Markdown in the response is harmless because we extract
                    # the JSON array via regex / raw_decode anyway.
                    _cli_cmd,
                    capture_output=True, text=True, timeout=300,
                    encoding="utf-8", cwd=PROJECT_DIR, env=_env,
                )
            except subprocess.TimeoutExpired:
                _log(f"  [ideas] CLI timed out on attempt {_attempt+1} — retrying")
                continue

            raw_cli = (result.stdout or "").strip()
            stderr  = (result.stderr or "").strip()
            combined_out = (raw_cli + " " + stderr).lower()
            _log(f"  [ideas] CLI attempt {_attempt+1}: rc={result.returncode}, "
                 f"stdout={len(raw_cli)} chars, stderr={len(stderr)} chars")

            # Detect rate-limit / credit exhaustion in output or stderr
            if any(m in combined_out for m in _RATE_MARKERS) and not raw_cli.startswith("["):
                _log(f"  [ideas] rate limited / out of credits on attempt {_attempt+1} — will retry")
                _log_cli_dump("rate-limited", _cli_cmd, result.returncode, raw_cli, stderr)
                continue

            if raw_cli:
                _log(f"  [ideas] CLI response ({len(raw_cli)} chars): {raw_cli[:200]}…")
                return _parse_ideas_response(raw_cli)

            # No stdout — figure out whether to retry, fall back to API, or bail
            _log_cli_dump(f"empty-output-rc{result.returncode}",
                          _cli_cmd, result.returncode, raw_cli, stderr)

            if result.returncode != 0:
                # rc=1 with no output is the bug this task fixes: previously
                # we'd retry up to 3 times then give up. Now we capture full
                # stderr (logged above), and if no transient marker is found
                # we fall straight through to the Anthropic API fallback so
                # the overnight cycle doesn't burn 30 minutes on a hard error.
                _err_preview = stderr[:300] if stderr else "(no stderr)"
                _log(f"  [ideas] CLI failed silently (rc={result.returncode}) — "
                     f"stderr: {_err_preview}")
                _log("  [ideas] falling back to direct Anthropic API")
                api_ideas = _generate_ideas_via_api(system_block, user_block)
                if api_ideas:
                    return api_ideas
                # API fallback also failed — only then do we retry the CLI in
                # case the issue was a transient blip (network, auth refresh)
                if _attempt >= 1:
                    _log(f"  [ideas] giving up after {_attempt+1} attempts + API fallback")
                    return []
                continue
            else:
                # rc=0 but empty output — CLI succeeded but produced nothing.
                # Try the API once before giving up.
                _log(f"  [ideas] CLI no output (rc=0) — trying Anthropic API fallback")
                return _generate_ideas_via_api(system_block, user_block)

        _log("  [ideas] all retry attempts exhausted — trying Anthropic API fallback")
        return _generate_ideas_via_api(system_block, user_block)

    except Exception as _cli_err:
        _log(f"  [ideas] CLI raised {type(_cli_err).__name__}: {_cli_err} — falling back to API")
        return _generate_ideas_via_api(system_block, user_block)


# ── todo management ──────────────────────────────────────────────────────────

def _append_tasks(tasks: list[str]) -> int:
    """Write new tasks to jarvis_todo.md. Returns count written."""
    if not tasks:
        return 0

    existing = _read_existing_todo_tasks()
    ts = time.strftime("%Y-%m-%d %H:%M")
    written = 0

    # Create file with header if it doesn't exist
    if not os.path.exists(TODO_FILE):
        with open(TODO_FILE, "w", encoding="utf-8") as f:
            f.write(
                "# JARVIS Task Queue\n\n"
                "Things the user wants Claude Code to build, fix, or investigate later.\n"
                "Tick items as you complete them; archive when the file gets big.\n\n"
            )

    with open(TODO_FILE, "a", encoding="utf-8") as f:
        for task in tasks:
            task = task.strip()
            if not task:
                continue
            # Simple duplicate check
            if task.lower() in existing:
                _log(f"  [todo] skip duplicate: {task[:70]}")
                continue
            f.write(f"- [ ] **{ts}** [overnight] — {task}\n")
            existing.add(task.lower())
            written += 1

    return written


# ── upgrade pipeline ─────────────────────────────────────────────────────────

def _run_upgrade(task_count: int) -> bool:
    if not os.path.exists(UPGRADE_SCRIPT):
        _log(f"  [upgrade] upgrade_jarvis.py not found")
        return False

    # Pass the API key via env dict — never embed it in the command string
    # where it would show up in process lists, logs, or shell history.
    env = os.environ.copy()
    api_key = _get_api_key()
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key

    ps_cmd = (
        f"cd '{PROJECT_DIR}'; "
        f"Write-Host '=== JARVIS OVERNIGHT UPGRADE ({task_count} tasks) ===' "
        f"-ForegroundColor Magenta; "
        f"python '{UPGRADE_SCRIPT}' --relaunch"
    )
    try:
        # No -NoExit: when upgrade_jarvis.py finishes the window closes
        # cleanly. The user shouldn't have to manually close anything.
        proc = subprocess.Popen(
            ["powershell", "-Command", ps_cmd],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            env=env,
            close_fds=True,
        )
        _log(f"  [upgrade] spawned (PID {proc.pid}), waiting…")
        proc.wait()
        _log(f"  [upgrade] done (exit {proc.returncode})")
        return True
    except Exception as e:
        _log(f"  [upgrade] spawn failed: {e}")
        return False


# ── main cycle ───────────────────────────────────────────────────────────────

def run_one_cycle():
    """Full upgrade cycle: gather context → generate ideas → write tasks → upgrade."""
    _log("═" * 60)
    _log("Overnight upgrade cycle starting")

    _log("  Reading user memory…")
    memory_summary = _read_memory_summary()

    _log("  Reading codebase features…")
    features = _read_codebase_features()

    _log("  Reading recent session logs…")
    logs = _read_logs_tail()
    _log(f"    {len(logs):,} chars from {len(_get_recent_logs())} log(s)")

    _log("  Asking Claude for improvement ideas…")
    ideas = _generate_ideas(memory_summary, features, logs)
    _log(f"  Generated {len(ideas)} idea(s):")
    for i, idea in enumerate(ideas, 1):
        _log(f"    [{i}] {idea[:100]}")

    if not ideas:
        _log("  No ideas generated — skipping upgrade")
        return False

    if DRY_RUN:
        _log("  [DRY RUN] Would append tasks and run upgrade. Stopping here.")
        return True

    written = _append_tasks(ideas)
    _log(f"  Appended {written} new task(s) to jarvis_todo.md")

    if written == 0:
        _log("  All ideas were duplicates — skipping upgrade run")
        return False

    _log(f"  Launching upgrade pipeline for {written} new task(s)…")
    _run_upgrade(written)
    return True


def watch_loop():
    _log("JARVIS Overnight Upgrade Engine started")
    _log(f"  Idle threshold : {IDLE_THRESHOLD_MINUTES} min")
    _log(f"  Time window    : {OVERNIGHT_START_HOUR:02d}:00 – {OVERNIGHT_END_HOUR:02d}:00  (0=always)")
    _log(f"  Min gap        : {MIN_CYCLE_GAP_HOURS}h between cycles")
    _log(f"  Ideas per cycle: up to {MAX_IDEAS_PER_CYCLE}")
    _log(f"  Dry run        : {DRY_RUN}")

    # Real auth/path check up front. If both the CLI and the API key are
    # missing, surface it loudly now instead of failing silently 30 minutes
    # later when the first idle cycle would have fired.
    ok, cli_msg = _check_claude_cli_health()
    _log(f"  Claude CLI     : {'OK — ' if ok else 'UNAVAILABLE — '}{cli_msg}")
    api_key_present = bool(_get_api_key())
    _log(f"  Anthropic API  : {'KEY PRESENT (fallback ready)' if api_key_present else 'NO KEY (no fallback)'}")
    if not ok and not api_key_present:
        _log("  WARNING: neither Claude Code CLI nor ANTHROPIC_API_KEY available — "
             "idea generation will fail every cycle until one is configured.")
    _log("")

    while True:
        try:
            state = _load_state()
            now = time.time()

            skip = []
            # Circuit breaker first — if tripped + cooling down, skip regardless
            # of the time/idle/gap gates (and persist any cooldown auto-reset).
            breaker_reason = _breaker_skip_reason(state, now)
            if breaker_reason:
                skip.append(breaker_reason)
            if not _is_in_time_window():
                skip.append(f"outside window (now {time.strftime('%H:%M')})")
            if not _is_idle():
                idle_min = (time.time() - _most_recent_log_mtime()) / 60
                skip.append(f"not idle yet ({idle_min:.0f}/{IDLE_THRESHOLD_MINUTES} min)")
            if not _enough_gap(state):
                gap_h = (time.time() - state["last_cycle_at"]) / 3600
                skip.append(f"too soon ({gap_h:.1f}h / {MIN_CYCLE_GAP_HOURS}h gap)")

            if skip:
                _save_state(state)   # persist a breaker auto-reset if it happened
                _log("Waiting: " + "; ".join(skip))
            else:
                cycle_start = time.time()
                did_work = run_one_cycle()
                state["last_cycle_at"] = time.time()
                if did_work:
                    state["cycles_run"] = state.get("cycles_run", 0) + 1
                    # Judge the upgrade and update the breaker. A reverted /
                    # P0-shipping upgrade counts as a failure; trip after
                    # MAX_CONSECUTIVE_FAILURES so we stop churning on it.
                    outcome = _last_upgrade_productive(cycle_start)
                    _record_cycle_outcome(state, outcome, time.time())
                    if state.get("breaker_tripped_at"):
                        _log(f"  ⚠ CIRCUIT BREAKER TRIPPED: "
                             f"{state.get('consecutive_failures', 0)} consecutive "
                             f"failed upgrades. Pausing upgrades for "
                             f"{BREAKER_COOLDOWN_HOURS:.0f}h — clear with "
                             f"--reset-breaker.")
                _save_state(state)
                _log(f"Cycle complete. Total cycles run: {state['cycles_run']}")

        except KeyboardInterrupt:
            _log("Stopped.")
            sys.exit(0)
        except Exception as e:
            _log(f"Error in watch loop: {e}")

        _log(f"Next check in {POLL_INTERVAL}s…\n")
        time.sleep(POLL_INTERVAL)


def main():
    # Pull API key from registry if not in env (common when launched standalone)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        key = _read_env_var_from_registry("ANTHROPIC_API_KEY")
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key

    if "--reset-breaker" in sys.argv:
        st = _load_state()
        st["consecutive_failures"] = 0
        st.pop("breaker_tripped_at", None)
        _save_state(st)
        _log("Circuit breaker reset — upgrades resume on the next eligible cycle.")
        return

    if "--now" in sys.argv:
        _log("--now: running one cycle immediately")
        ok, cli_msg = _check_claude_cli_health()
        _log(f"  Claude CLI    : {'OK — ' if ok else 'UNAVAILABLE — '}{cli_msg}")
        _log(f"  Anthropic API : {'KEY PRESENT' if _get_api_key() else 'NO KEY'}")
        run_one_cycle()
    else:
        watch_loop()


if __name__ == "__main__":
    main()
