"""Which Claude subscription a JARVIS Claude Code job runs on (2026-10-14).

THE OWNER'S ASK
===============
He has two subscriptions: his Max account (the DEFAULT Claude Code profile,
~/.claude) and a second Pro account he has already cancelled, which stays paid
until it lapses. Until CLAUDE_PRO_FIRST_UNTIL (inclusive) every Claude Code job
JARVIS spawns runs on the Pro profile FIRST, so its paid usage gets used; when
Pro hits its 5-hour (session) or weekly limit the job runs on Max instead, and
Pro is tried again once that limit resets. From the day after the date, Pro is
never used. An empty date (the default) turns the whole thing off: Max only,
exactly as before.

HOW A PROFILE IS PICKED
=======================
The Pro profile is its own Claude Code config folder (CLAUDE_PRO_CONFIG_DIR,
default ~/.claude-pro) that HE signs in to once, with CLAUDE_CONFIG_DIR set.
Claude Code keeps that sign-in inside the folder. This module only checks that
the sign-in file EXISTS - it never opens it - and hands a job the profile via
CLAUDE_CONFIG_DIR in the job's environment. No token passes through JARVIS, its
settings or its logs: log lines carry the profile name, the limit kind and the
reset time, nothing else.

LIMITS
======
Claude Code reports a refused request two ways; both are parsed here.
  * stream-json: {"type": "rate_limit_event", "rate_limit_info": {"status":
    "rejected", "rateLimitType": "five_hour" | "seven_day" | ..., "resetsAt":
    <epoch>}} - the exact reset time.
  * text: "You've hit your session limit · resets 9pm" / "... weekly limit ..."
    (or the older "Claude AI usage limit reached|<epoch>"). Text without an
    epoch gets a re-probe time instead (session 30 min, weekly 3 h): the next
    job tries Pro again, and a still-limited Pro refuses within seconds.
A Pro run that fails because the profile is signed out (or the subscription has
lapsed) parks Pro for 6 hours the same way. State, with no secrets, lives in
data/claude_account_state.json.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

PRO = "pro"
MAX = "max"

SIGN_IN_FILE = ".credentials.json"     # Claude Code's own file; never opened here
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(_ROOT, "data", "claude_account_state.json")

# Re-probe delays when a limit arrives as text with no reset time.
PROBE_S = {
    "five_hour": 30 * 60,
    "seven_day": 3 * 3600,
    "seven_day_opus": 3 * 3600,
    "seven_day_sonnet": 3 * 3600,
    "signed_out": 6 * 3600,
}
DEFAULT_PROBE_S = 3600
RESET_MARGIN_S = 60                    # a reset time is honoured a minute late

# Variables that would override the profile's own sign-in (an API key bills
# API credits; an OAuth/auth token pins one account regardless of the folder).
_AUTH_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
# A JARVIS started from inside a Claude Code / desktop-app session inherits
# that SESSION's identity: its API proxy and host-managed sign-in. A job is its
# own session on its own profile, so those are dropped too (measured
# 2026-10-14: with them set, `claude auth status` reports the wrong state).
_HOST_SESSION_ENV = (
    "ANTHROPIC_BASE_URL", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_HOST_SESSION_ID", "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH",
    "CLAUDE_CODE_ACCOUNT_UUID", "CLAUDE_CODE_ORGANIZATION_UUID",
    "CLAUDE_CODE_USER_EMAIL", "CLAUDE_CODE_OAUTH_SCOPES",
)

LABELS = {
    "five_hour": "5-hour limit",
    "seven_day": "weekly limit",
    "seven_day_opus": "weekly Opus limit",
    "seven_day_sonnet": "weekly Sonnet limit",
    "overage": "extra-usage limit",
    "usage": "usage limit",
    "signed_out": "sign-in",
}

_TEXT_KINDS = {
    "session": "five_hour",
    "weekly": "seven_day",
    "opus": "seven_day_opus",
    "sonnet": "seven_day_sonnet",
}
# "You've hit your session limit · resets 9pm" (Claude Code 2.1.x) and the
# older "Claude AI usage limit reached|1760000000".
_HIT_RE = re.compile(r"You(?:'|’)ve hit your (?:(session|weekly|opus|sonnet) )?(?:usage )?limit",
                     re.IGNORECASE)
_OLD_RE = re.compile(r"Claude AI usage limit reached\|(\d{9,13})")
# Signed out / no subscription on the profile.
_SIGNED_OUT_RE = re.compile(
    r"Not logged in|Run claude auth login|Please run /login|Invalid API key"
    r"|OAuth token (?:has )?expired|does not have access to Claude Code"
    r"|requires a Claude (?:Pro|Max)", re.IGNORECASE)
# A limit message is one short line; a longer reply that merely QUOTES the
# words (a task about rate limits) must not park Pro.
_TEXT_MAX_CHARS = 400


@dataclass
class Limit:
    kind: str                       # five_hour | seven_day | ... | usage
    resets_at: Optional[float]      # epoch seconds, when known


# ── settings ───────────────────────────────────────────────────────────────
def _cfg(name: str, default):
    try:
        from core import config
        return getattr(config, name, default)
    except Exception:
        return default


def pro_dir() -> str:
    d = str(_cfg("CLAUDE_PRO_CONFIG_DIR", "") or "").strip()
    return d or os.path.join(os.path.expanduser("~"), ".claude-pro")


def pro_until() -> Optional[_dt.date]:
    """The Pro account's last day (inclusive), or None when pro-first is off."""
    raw = str(_cfg("CLAUDE_PRO_FIRST_UNTIL", "") or "").strip()
    if not raw:
        return None
    try:
        return _dt.date.fromisoformat(raw)
    except ValueError:
        return None


def pro_signed_in(folder: Optional[str] = None) -> bool:
    """True when the Pro profile has a sign-in file. Existence only."""
    return os.path.isfile(os.path.join(folder or pro_dir(), SIGN_IN_FILE))


# ── state (no secrets) ─────────────────────────────────────────────────────
def load_state(path: Optional[str] = None) -> dict:
    try:
        with open(path or STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(state: dict, path: Optional[str] = None) -> bool:
    """Atomic write. Never raises (a test run's live-data guard may refuse)."""
    path = path or STATE_PATH
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(state, f, indent=1)
            f.write("\n")
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def _hhmm(epoch: float) -> str:
    try:
        t = _dt.datetime.fromtimestamp(epoch)
        today = _dt.date.today()
        return t.strftime("%H:%M") if t.date() == today else t.strftime("%b %d %H:%M")
    except Exception:
        return "?"


# ── the choice ─────────────────────────────────────────────────────────────
def choose(now: Optional[float] = None, today: Optional[_dt.date] = None,
           state: Optional[dict] = None,
           signed_in: Optional[bool] = None) -> Tuple[str, str]:
    """(profile, why) for the next job."""
    now = time.time() if now is None else now
    today = _dt.date.today() if today is None else today
    until = pro_until()
    if until is None:
        return MAX, "pro-first is off"
    if today > until:
        return MAX, f"the Pro account's last day was {until.isoformat()}"
    if not (pro_signed_in() if signed_in is None else signed_in):
        return MAX, "the Pro profile isn't signed in"
    st = load_state() if state is None else state
    blocked = float(st.get("pro_blocked_until") or 0)
    if blocked > now:
        kind = LABELS.get(st.get("pro_blocked_kind", ""), "limit")
        return MAX, f"Pro hit its {kind}; back on Pro at {_hhmm(blocked)}"
    return PRO, f"Pro first until {until.isoformat()}"


def env_for(profile: str, base: Optional[dict] = None) -> dict:
    """The job's environment for ``profile``: auth overrides removed, and the
    Pro profile's folder (Pro) or the default folder (Max) selected."""
    env = dict(os.environ if base is None else base)
    for k in _AUTH_ENV + _HOST_SESSION_ENV:
        env.pop(k, None)
    if profile == PRO:
        env["CLAUDE_CONFIG_DIR"] = pro_dir()
    else:
        env.pop("CLAUDE_CONFIG_DIR", None)
    return env


# ── reading Claude Code's answers ──────────────────────────────────────────
def _epoch(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x <= 0:
        return None
    return x / 1000.0 if x > 1e12 else x


def limit_from_event(obj) -> Optional[Limit]:
    """A refused-request event from --output-format stream-json, else None."""
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "rate_limit_event":
        info = obj.get("rate_limit_info") or {}
        if isinstance(info, dict) and info.get("status") == "rejected":
            return Limit(str(info.get("rateLimitType") or "usage"),
                         _epoch(info.get("resetsAt")))
        return None
    if obj.get("type") == "result" and obj.get("is_error"):
        return limit_from_text(str(obj.get("result") or ""))
    return None


def limit_from_text(text: str) -> Optional[Limit]:
    """A limit refusal printed as text (a short line), else None."""
    if not text:
        return None
    s = text.strip()
    m = _OLD_RE.search(s)
    if m:
        return Limit("usage", _epoch(m.group(1)))
    if len(s) > _TEXT_MAX_CHARS:
        return None
    m = _HIT_RE.search(s)
    if m:
        return Limit(_TEXT_KINDS.get((m.group(1) or "").lower(), "usage"), None)
    return None


def scan_output(text: str) -> Optional[Limit]:
    """The limit that REFUSED a job, from its output: stream-json lines or a
    text reply. A limit event on a job whose final result succeeded (Claude
    Code moved to another model and finished) is not a refusal."""
    if not text:
        return None
    found: Optional[Limit] = None
    result_ok = False
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result" and not obj.get("is_error"):
            result_ok = True
        found = found or limit_from_event(obj)
    if result_ok:
        return None
    return found or limit_from_text(text)


def signed_out_text(text: str) -> bool:
    return bool(text) and len(text.strip()) <= 2000 and bool(_SIGNED_OUT_RE.search(text))


# ── recording what happened ────────────────────────────────────────────────
def note_pro_refusal(limit: Optional[Limit] = None, signed_out: bool = False,
                     now: Optional[float] = None, path: Optional[str] = None) -> Tuple[float, str]:
    """Park Pro until its limit resets (or a re-probe time). Returns
    (blocked_until, message-for-the-log)."""
    now = time.time() if now is None else now
    kind = "signed_out" if signed_out else (limit.kind if limit else "usage")
    if limit and limit.resets_at:
        until = max(now + 60, limit.resets_at + RESET_MARGIN_S)
    else:
        until = now + PROBE_S.get(kind, DEFAULT_PROBE_S)
    st = load_state(path)
    st.update({"pro_blocked_until": round(until), "pro_blocked_kind": kind,
               "pro_blocked_at": round(now)})
    save_state(st, path)
    what = ("isn't signed in or has lapsed" if signed_out
            else f"hit its {LABELS.get(kind, 'limit')}")
    return until, f"Pro {what}; this job and the next ones run on Max until {_hhmm(until)}"


def pro_refused(text: str, returncode: int) -> Tuple[Optional[Limit], bool]:
    """(limit, signed_out) for a finished Pro job's combined output."""
    lim = scan_output(text)
    if lim:
        return lim, False
    return None, (returncode != 0 and signed_out_text(text))


# ── one call, Pro first ────────────────────────────────────────────────────
def run(cmd, *, env: Optional[dict] = None, runner: Callable = subprocess.run,
        log: Optional[Callable[[str], None]] = None, **kwargs):
    """subprocess.run-style call of a CAPTURED Claude Code job on the chosen
    profile. When Pro refuses (a limit, or signed out) the same command runs
    again on Max. The result carries ``.claude_profile``."""
    profile, why = choose()
    res = runner(cmd, env=env_for(profile, env), **kwargs)
    if profile == PRO:
        out = f"{getattr(res, 'stdout', '') or ''}\n{getattr(res, 'stderr', '') or ''}"
        lim, signed_out = pro_refused(out, int(getattr(res, "returncode", 0) or 0))
        if lim or signed_out:
            _until, msg = note_pro_refusal(lim, signed_out)
            if log:
                log(f"[claude-account] {msg}")
            profile = MAX
            res = runner(cmd, env=env_for(MAX, env), **kwargs)
    try:
        res.claude_profile = profile
    except Exception:
        pass
    return res


def status_line() -> str:
    """One line for logs and the status tool."""
    profile, why = choose()
    return f"next Claude Code job runs on {'Pro' if profile == PRO else 'Max'} ({why})"
