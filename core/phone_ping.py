"""core/phone_ping.py — JARVIS texts the owner's phone ONLY when something
needs him.

skills/phone_bridge.py already sends (Telegram / ntfy / Pushover). This module
is the POLICY in front of it: which events may ping, when, how often, and what
the text may contain. The bridge attaches itself at register() time
(attach_bridge); until it does, or while no backend can send an unsolicited
message, every ping is a no-op and one line says so.

Categories (each a Settings switch, PHONE_PING_<NAME>):
  print     a print FINISHED, FAILED, or PAUSED with an error (a hook on
            skills/bambu_monitor's state-change fan-out; first observation after
            a restart never pings, so a finished print is not re-sent per boot)
  confirm   a confirmation he left unanswered while away: the monolith's
            _pending_confirmation queue or the overnight shutdown prompt, older
            than PHONE_PING_CONFIRM_AFTER_MIN, once per prompt. Action NAMES
            only - an argument (a shell command, a message body) is never sent.
  security  a guard-mode alert (skills/guard_mode routes its push through here)
  robot     anything a skill reports with skill_utils["ping_phone"]("robot", …)
  summary   an optional once-a-day digest at PHONE_PING_SUMMARY_TIME

Gates, in order, for an ordinary (non-critical) ping:
  master switch + bridge configured + category switch + never on staging/tests
  -> dedupe (same key within the hour) -> focus mode (dropped) -> he is HERE
  (an owner turn within PHONE_PING_AWAY_MIN: JARVIS already said it out loud)
  -> quiet hours (held, then sent as ONE message when they end, or folded into
  the summary when that is on) -> at most PHONE_PING_MAX_PER_HOUR an hour.
A CRITICAL ping (security) skips focus / presence / quiet hours and the hourly
cap, but has its own ceiling (CRITICAL_MAX_PER_HOUR) so a runaway loop can
never flood the phone.

Every text is scrubbed before it leaves (scrub()): the value of any secret-
looking environment variable, credential-shaped strings (bot tokens, sk-…
keys, long random tokens, "password is …") are redacted, and a text that still
reads like a credential is replaced by a generic line.

The send itself runs on a sender thread, so a ping from the Bambu MQTT
callback or the guard monitor never waits on the network. Nothing here touches
audio or the main loop; the watcher ticks every TICK_S on its own daemon
thread.

Stdlib-only (CI light tier): the monolith and the skills are reached through
sys.modules, never imported.
"""
from __future__ import annotations

import collections
import datetime as _dt
import os
import queue
import re
import sys
import threading
import time
from typing import Any, Callable, Optional

__all__ = [
    "CATEGORIES", "CATEGORY_FLAGS", "DEFAULTS", "PhonePinger", "scrub",
    "attach_bridge", "get_pinger", "ping", "status", "start_watcher",
    "stop_watcher", "confirm_text", "parse_hhmm",
]

CATEGORIES = ("print", "confirm", "security", "robot", "summary")

# Spelled out literally: tests/test_settings_schema_wiring.py greps the tree
# for each persisted key, so a name built with an f-string would read as dead.
CATEGORY_FLAGS = {
    "print": "PHONE_PING_PRINT",
    "confirm": "PHONE_PING_CONFIRM",
    "security": "PHONE_PING_SECURITY",
    "robot": "PHONE_PING_ROBOT",
    "summary": "PHONE_PING_SUMMARY",
}

# Mirrors core/config.py (the fallback when core.config is not importable).
DEFAULTS: dict[str, Any] = {
    "PHONE_PING_ENABLED": True,
    "PHONE_PING_PRINT": True,
    "PHONE_PING_CONFIRM": True,
    "PHONE_PING_SECURITY": True,
    "PHONE_PING_ROBOT": True,
    "PHONE_PING_SUMMARY": False,
    "PHONE_PING_SUMMARY_TIME": "07:30",
    "PHONE_PING_MAX_PER_HOUR": 6,
    "PHONE_PING_QUIET_START": "23:00",
    "PHONE_PING_QUIET_END": "07:00",
    "PHONE_PING_AWAY_MIN": 10.0,
    "PHONE_PING_CONFIRM_AFTER_MIN": 2.0,
}

CRITICAL_MAX_PER_HOUR = 20     # a ceiling even for security alerts
DEDUPE_S = 3600.0              # same dedupe key inside this window = one ping
TICK_S = 30.0                  # watcher cadence
HELD_MAX = 12                  # quiet-hours backlog kept for the morning
JOURNAL_MAX = 60               # events remembered for the summary
SUMMARY_LINES = 8              # lines in one summary / digest
SUMMARY_WINDOW_H = 3.0         # a summary missed by more than this is skipped
FOLD_INTO_SUMMARY_MIN = 120    # held pings wait for a summary due this soon
MESSAGE_MAX = 400              # one ping's text, after scrubbing
SEND_QUEUE_MAX = 16

# Outcomes ping() returns (and the journal records).
QUEUED = "queued"              # handed to the sender thread
SENT = "sent"                  # the bridge reported success (journal only)
FAILED = "failed"              # the bridge reported failure (journal only)
EMPTY = "empty"
UNKNOWN = "unknown_category"
DISABLED = "disabled"          # PHONE_PING_ENABLED is off
UNCONFIGURED = "unconfigured"  # no backend can send an unsolicited message
CATEGORY_OFF = "category_off"
BLOCKED = "blocked"            # staging / test instance
DEDUPED = "dedupe"
FOCUS = "focus"                # focus mode: dropped (JARVIS recaps it himself)
PRESENT = "present"            # he is here: JARVIS already said it
QUIET = "quiet"                # held for the end of quiet hours
RATE_LIMITED = "rate_limited"

_RETRYABLE = frozenset({FOCUS, PRESENT})
_JOURNALLED = frozenset({QUEUED, SENT, FAILED, FOCUS, PRESENT, QUIET,
                         RATE_LIMITED})

_CATEGORY_LABEL = {
    "print": "print", "confirm": "confirmation", "security": "security",
    "robot": "robot", "summary": "summary",
}

_BOOT_MONO = time.monotonic()


# ─── scrubbing ────────────────────────────────────────────────────────────

# Env vars whose VALUE is a secret even though the name has no hint word: the
# ntfy topic IS the password of an ntfy channel, the Pushover user key
# addresses the account.
_SECRET_ENV_NAMES = frozenset({"NTFY_TOPIC", "PUSHOVER_USER"})
_SECRET_ENV_HINTS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY",
                     "APIKEY", "ACCESS_CODE", "PRIVATE_KEY", "CREDENTIAL")
_TOKEN_SHAPES = (
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}"),                 # Telegram bot
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"),              # sk-ant-…
    re.compile(r"\b(?:ghp|gho|ghs|ghu|github_pat|xox[abpr])[_-][A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                            # AWS key id
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"),
    # A long run of letters AND digits with no spaces: an API key, a session
    # cookie, a base64 blob. Real ping text (printer file names have their
    # _ and - turned into spaces) never contains one.
    re.compile(r"(?<![A-Za-z0-9])(?=[A-Za-z0-9+/_=-]*\d)"
               r"(?=[A-Za-z0-9+/_=-]*[A-Za-z])[A-Za-z0-9+/_=-]{32,}"),
)
_KV_SECRET = re.compile(
    r"(?i)\b(password|passwd|pwd|passcode|pin|token|api[ _-]?key|secret|"
    r"access[ _-]?code)\b(\s*(?:is|=|:)\s*)(\S+)")
_REDACTED = "[redacted]"

_WITHHELD = {
    "print": "The printer needs you, sir. I've left the details off the "
             "phone because they looked sensitive; ask me at the desk.",
    "confirm": "Something is waiting on your yes, sir. I've left the details "
               "off the phone because they looked sensitive.",
    "security": "Security alert, sir. I've left the details off the phone "
                "because they looked sensitive; check the cameras.",
    "robot": "The robot needs attention, sir. I've left the details off the "
             "phone because they looked sensitive.",
    "summary": "Your summary had details I wouldn't put on a phone, sir; ask "
               "me for it at the desk.",
}


def _secret_env_values(env) -> list[str]:
    out = []
    try:
        items = list(env.items())
    except Exception:
        return out
    for name, value in items:
        n = str(name or "").upper()
        if not (n in _SECRET_ENV_NAMES or n.endswith("_KEY")
                or any(h in n for h in _SECRET_ENV_HINTS)):
            continue
        v = str(value or "").strip()
        if len(v) >= 6:
            out.append(v)
    # Longest first, so a value that contains another is removed whole.
    return sorted(set(out), key=len, reverse=True)


def _looks_secret(text: str) -> bool:
    """core.memory_guards' credential test (the same one that keeps secrets
    out of cloud memory), with a local copy of its keywords as the fallback."""
    try:
        from core.memory_guards import _is_secret_fact
        return bool(_is_secret_fact(text))
    except Exception:
        return bool(re.search(r"(?i)\b(password|passwd|passphrase|api[\s_-]?key|"
                              r"secret|token|access[\s_-]?code|credential|ssn|"
                              r"credit[\s_-]?card|cvv)\b", text or ""))


def scrub(text: str, env=None) -> tuple[str, bool]:
    """``(clean_text, withheld)``. Redacts secret env values and credential-
    shaped strings; ``withheld`` is True when what is left still reads like a
    credential, and the caller must send a generic line instead. Never
    raises."""
    try:
        s = str(text or "")
        for value in _secret_env_values(os.environ if env is None else env):
            if value in s:
                s = s.replace(value, _REDACTED)
        s = _KV_SECRET.sub(lambda m: m.group(1) + m.group(2) + _REDACTED, s)
        for rx in _TOKEN_SHAPES:
            s = rx.sub(_REDACTED, s)
        return s, _looks_secret(s)
    except Exception:
        return "", True


def _clip(text: str, limit: int = MESSAGE_MAX) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit - 1)
    if cut < limit // 2:
        cut = limit - 1
    return text[:cut].rstrip(" ,;:-") + "…"


# ─── small parsers / text builders ────────────────────────────────────────

_HHMM = re.compile(r"^\s*(\d{1,2})(?:[:.h](\d{2}))?\s*$")


def parse_hhmm(value, default: str) -> int:
    """'23:00' / '7' / '07.30' / 23 -> minutes after midnight. A bad value
    falls back to ``default`` (itself 'HH:MM')."""
    for cand in (value, default):
        if isinstance(cand, bool):
            continue
        if isinstance(cand, (int, float)):
            h, m = int(cand), 0
        else:
            mt = _HHMM.match(str(cand or ""))
            if not mt:
                continue
            h, m = int(mt.group(1)), int(mt.group(2) or 0)
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h * 60 + m
    return 0


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


_ACTION_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _humanize_action(name) -> str:
    n = str(name or "").strip().lower()
    if not _ACTION_NAME.match(n):
        return "an action"
    return n.replace("_", " ")


def confirm_text(names, *, lapsed: bool) -> str:
    """The confirm-category ping. Action NAMES only — never an argument."""
    labels = [_humanize_action(n) for n in list(names or [])]
    if not labels:
        labels = ["an action"]
    shown = labels[:2]
    what = "'" + "' and '".join(shown) + "'"
    if len(labels) > 2:
        what += f" and {len(labels) - 2} more"
    if lapsed:
        return (f"You left {what} waiting on a yes, sir. Nothing ran: it "
                f"lapsed unanswered, so ask me again when you're back.")
    return (f"{what} is waiting on your yes, sir. Nothing has run; I'll drop "
            f"it if it isn't answered.")


SHUTDOWN_PROMPT_TEXT = ("You asked me to shut down and stepped away before "
                        "answering the overnight question, sir. Nothing shut "
                        "down: I'm still running.")


def _print_name(snapshot: dict) -> str:
    raw = ""
    for key in ("subtask_name", "filename", "gcode_file"):
        val = snapshot.get(key) if isinstance(snapshot, dict) else None
        if val:
            raw = str(val)
            break
    base = os.path.basename(raw.replace("\\", "/"))
    base = re.sub(r"(?i)(\.gcode)?\.(3mf|gcode|gco)$", "", base)
    base = " ".join(re.sub(r"[_\-]+", " ", base).split())
    return base[:60]


def _layer_phrase(snapshot: dict) -> str:
    layer = snapshot.get("layer_num") if isinstance(snapshot, dict) else None
    total = snapshot.get("total_layer") if isinstance(snapshot, dict) else None
    try:
        layer_i = int(layer)
    except (TypeError, ValueError):
        return ""
    if layer_i <= 0:
        return ""
    try:
        total_i = int(total)
    except (TypeError, ValueError):
        total_i = 0
    if total_i > 0:
        return f" at layer {layer_i} of {total_i}"
    return f" at layer {layer_i}"


_PRINT_ACTIVE = frozenset({"RUNNING", "PAUSE", "PREPARE", "SLICING"})


# ─── live seams (each replaceable on a PhonePinger instance) ─────────────

def _bc():
    """The running monolith (the boot aliases bobert_companion to __main__)."""
    return sys.modules.get("bobert_companion")


def _live_cfg(name: str):
    try:
        from core import config as _c
        return getattr(_c, name, DEFAULTS.get(name))
    except Exception:
        return DEFAULTS.get(name)


def _live_focus_active() -> bool:
    """Either focus mode: the monolith's announcement gate (honoured only while
    FOCUS_MODE_ENABLED, like proactive_announce) or skills/dnd_focus_mode."""
    try:
        enabled = _live_cfg("FOCUS_MODE_ENABLED")
        if enabled is None or bool(enabled):
            bc = _bc()
            fn = getattr(bc, "focus_mode_active", None) if bc is not None else None
            if callable(fn) and fn():
                return True
    except Exception:
        pass
    try:
        mod = sys.modules.get("skill_dnd_focus_mode")
        fn = getattr(mod, "is_focus_mode_active", None) if mod is not None else None
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def _live_owner_idle_s() -> Optional[float]:
    """Seconds since the owner's last accepted turn (voice or typed; a phone
    message is not one). Before his first turn this process, counted from when
    this module loaded — a fresh boot is not proof he left."""
    bc = _bc()
    if bc is None:
        return None
    try:
        cell = getattr(bc, "_last_owner_turn_at", None)
        at = float(cell[0]) if cell else 0.0
    except Exception:
        at = 0.0
    if at <= 0.0:
        at = _BOOT_MONO
    return max(0.0, time.monotonic() - at)


def _live_blocked() -> str:
    """Never ping from a staging / blue-green / test process."""
    if os.environ.get("JARVIS_STAGING", "").strip() == "1" \
            or "--staging" in sys.argv:
        return "staging"
    if os.environ.get("JARVIS_TEST_MODE", "").strip() == "1":
        return "test mode"
    return ""


def _live_pending() -> list[dict]:
    """Confirmation prompts waiting on the owner: [{key, age_s, text}]. Reads
    the monolith's state cells; copies the list first (GIL-atomic), and takes
    each queued action's NAME only."""
    bc = _bc()
    if bc is None:
        return []
    out: list[dict] = []
    try:
        queue_ = list(getattr(bc, "_pending_confirmation", None) or [])
        at_cell = getattr(bc, "_pending_confirmation_at", None)
        at = float(at_cell[0]) if at_cell else 0.0
        if queue_ and at > 0.0:
            names = [entry[0] for entry in queue_
                     if isinstance(entry, (tuple, list)) and entry]
            age = max(0.0, time.monotonic() - at)
            ttl = float(getattr(bc, "CONFIRMATION_TTL_S", 45.0) or 45.0)
            out.append({"key": f"queue:{at:.3f}", "age_s": age,
                        "text": confirm_text(names, lapsed=age > ttl)})
    except Exception:
        pass
    try:
        sp = getattr(bc, "_shutdown_prompt_pending", None)
        if isinstance(sp, dict) and sp.get("armed"):
            expires = float(sp.get("expires_at") or 0.0)
            timeout = float(getattr(bc, "SHUTDOWN_PROMPT_TIMEOUT_S", 30.0) or 30.0)
            if expires > 0.0:
                age = max(0.0, time.time() - (expires - timeout))
                out.append({"key": f"shutdown:{expires:.3f}", "age_s": age,
                            "text": SHUTDOWN_PROMPT_TEXT})
    except Exception:
        pass
    return out


def _live_printer_now() -> str:
    """One line on a print that is running / paused right now, for the
    summary. Empty when there is none or the monitor is not loaded."""
    mod = sys.modules.get("skill_bambu_monitor")
    if mod is None:
        return ""
    try:
        lock = getattr(mod, "_state_lock", None)
        state = getattr(mod, "_state", None)
        if not isinstance(state, dict):
            return ""
        if lock is not None:
            with lock:
                snap = dict(state)
        else:
            snap = dict(state)
    except Exception:
        return ""
    gstate = str(snap.get("gcode_state") or "").upper()
    if gstate not in ("RUNNING", "PAUSE"):
        return ""
    name = _print_name(snap)
    try:
        pct = int(float(snap.get("mc_percent")))
        pct_s = f" {pct}%"
    except (TypeError, ValueError):
        pct_s = ""
    verb = "printing" if gstate == "RUNNING" else "paused"
    return f"Printer: {verb}{pct_s}" + (f" ({name})" if name else "") + "."


def _live_env():
    return os.environ


def _default_log(line: str) -> None:
    try:
        print(f"  [phone-ping] {line}")
    except Exception:
        pass


def _default_state_path() -> Optional[str]:
    try:
        from core.paths import data_file
        return data_file("phone_ping_state.json")
    except Exception:
        return None


# ─── the pinger ──────────────────────────────────────────────────────────

class PhonePinger:
    """All policy state. Every seam is an attribute so a test can pin it:
    send, configured, cfg, wall_now, mono_now, focus_active, owner_idle_s,
    blocked, pending, printer_now, env, state_path, spawn, log."""

    def __init__(self, *, send: Optional[Callable] = None,
                 configured: Optional[Callable[[], bool]] = None,
                 cfg: Optional[Callable[[str], Any]] = None,
                 wall_now: Optional[Callable[[], _dt.datetime]] = None,
                 mono_now: Optional[Callable[[], float]] = None,
                 focus_active: Optional[Callable[[], bool]] = None,
                 owner_idle_s: Optional[Callable[[], Optional[float]]] = None,
                 blocked: Optional[Callable[[], str]] = None,
                 pending: Optional[Callable[[], list]] = None,
                 printer_now: Optional[Callable[[], str]] = None,
                 env: Optional[Callable[[], Any]] = None,
                 state_path: Any = "default",
                 spawn: Optional[Callable[[Callable[[], None]], Any]] = None,
                 log: Optional[Callable[[str], None]] = None) -> None:
        self.send = send
        self.configured = configured
        self.cfg = cfg or _live_cfg
        self.wall_now = wall_now or _dt.datetime.now
        self.mono_now = mono_now or time.monotonic
        self.focus_active = focus_active or _live_focus_active
        self.owner_idle_s = owner_idle_s or _live_owner_idle_s
        self.blocked = blocked or _live_blocked
        self.pending = pending or _live_pending
        self.printer_now = printer_now or _live_printer_now
        self.env = env or _live_env
        self._state_path = state_path
        self.spawn = spawn or self._thread_spawn
        self.log = log or _default_log

        self._lock = threading.RLock()
        self._loaded = False
        self._sent_at: collections.deque = collections.deque(maxlen=256)
        self._dedupe: dict[str, float] = {}
        self._held: list[dict] = []
        self._journal: list[dict] = []
        self._summary_sent_on = ""
        self._summary_skipped_on = ""
        self._last_summary_t = 0.0
        self._pending_done: collections.OrderedDict = collections.OrderedDict()
        self._soft_seen: collections.OrderedDict = collections.OrderedDict()
        self._logged_unconfigured = False
        self._bambu_hooked: Any = None
        self._q: Optional[queue.Queue] = None
        self._sender: Optional[threading.Thread] = None
        self.last_outcome: dict[str, str] = {}

    # ── config ────────────────────────────────────────────────────────────
    def _flag(self, name: str) -> bool:
        try:
            val = self.cfg(name)
        except Exception:
            val = None
        if val is None:
            val = DEFAULTS.get(name, False)
        return bool(val)

    def _num(self, name: str) -> float:
        try:
            val = float(self.cfg(name))
        except Exception:
            val = float(DEFAULTS.get(name, 0) or 0)
        if val != val or val < 0:          # NaN / negative -> the default
            val = float(DEFAULTS.get(name, 0) or 0)
        return val

    def _minutes(self, name: str) -> int:
        try:
            raw = self.cfg(name)
        except Exception:
            raw = None
        return parse_hhmm(raw, str(DEFAULTS[name]))

    def bridge_configured(self) -> bool:
        if self.send is None or self.configured is None:
            return False
        try:
            return bool(self.configured())
        except Exception:
            return False

    def log_unconfigured_once(self) -> None:
        """The ONE line that says pings are a no-op without a bridge."""
        if self._logged_unconfigured:
            return
        self._logged_unconfigured = True
        self.log("phone bridge not configured — phone pings are a no-op "
                 "(say 'how do I connect my phone')")

    def _gate(self, category: str) -> str:
        """'' when this category may ping at all right now, else the reason.
        Logs the not-configured case ONCE per process."""
        if not self._flag("PHONE_PING_ENABLED"):
            return DISABLED
        if not self.bridge_configured():
            self.log_unconfigured_once()
            return UNCONFIGURED
        flag = CATEGORY_FLAGS.get(category)
        if flag and not self._flag(flag):
            return CATEGORY_OFF
        try:
            if self.blocked():
                return BLOCKED
        except Exception:
            return BLOCKED
        return ""

    def active(self) -> bool:
        """Master on + bridge configured + not a staging/test process."""
        if not self._flag("PHONE_PING_ENABLED") or not self.bridge_configured():
            return False
        try:
            return not self.blocked()
        except Exception:
            return False

    # ── clocks / presence ────────────────────────────────────────────────
    def in_quiet_hours(self, now_w: Optional[_dt.datetime] = None) -> bool:
        now_w = now_w or self.wall_now()
        start = self._minutes("PHONE_PING_QUIET_START")
        end = self._minutes("PHONE_PING_QUIET_END")
        if start == end:
            return False
        m = now_w.hour * 60 + now_w.minute
        if start < end:
            return start <= m < end
        return m >= start or m < end

    def _owner_present(self) -> bool:
        away_s = self._num("PHONE_PING_AWAY_MIN") * 60.0
        if away_s <= 0:
            return False
        try:
            idle = self.owner_idle_s()
        except Exception:
            idle = None
        return idle is not None and idle < away_s

    def _focus(self) -> bool:
        try:
            return bool(self.focus_active())
        except Exception:
            return False

    def _count_last_hour(self, now_m: float, *, critical: bool) -> int:
        return sum(1 for t, crit in self._sent_at
                   if crit == critical and now_m - t < 3600.0)

    # ── persistence ──────────────────────────────────────────────────────
    def _path(self) -> Optional[str]:
        if self._state_path == "default":
            return _default_state_path()
        return self._state_path

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        path = self._path()
        if not path or not os.path.exists(path):
            return
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        if not isinstance(data, dict):
            return

        def _rows(key):
            rows = data.get(key)
            if not isinstance(rows, list):
                return []
            ok = []
            for r in rows:
                if (isinstance(r, dict) and isinstance(r.get("t"), (int, float))
                        and isinstance(r.get("text"), str)
                        and r.get("cat") in CATEGORIES):
                    ok.append({"t": float(r["t"]), "cat": r["cat"],
                               "text": _clip(r["text"]),
                               "out": str(r.get("out") or "")})
            return ok

        self._held = _rows("held")[-HELD_MAX:]
        self._journal = _rows("journal")[-JOURNAL_MAX:]
        for key in ("summary_sent_on", "summary_skipped_on"):
            if isinstance(data.get(key), str):
                setattr(self, "_" + key, data[key][:10])
        if isinstance(data.get("last_summary_t"), (int, float)):
            self._last_summary_t = float(data["last_summary_t"])

    def _save(self) -> None:
        path = self._path()
        if not path:
            return
        data = {"held": self._held, "journal": self._journal,
                "summary_sent_on": self._summary_sent_on,
                "summary_skipped_on": self._summary_skipped_on,
                "last_summary_t": self._last_summary_t}
        try:
            from core.atomic_io import _atomic_write_json
            _atomic_write_json(path, data)
        except Exception as e:
            self.log(f"state save failed: {type(e).__name__}")

    # ── journal ──────────────────────────────────────────────────────────
    def _journal_add(self, t: float, cat: str, text: str, out: str) -> dict:
        entry = {"t": t, "cat": cat, "text": text, "out": out}
        if out in _JOURNALLED and cat != "summary":
            self._journal.append(entry)
            del self._journal[:-JOURNAL_MAX]
        return entry

    # ── the public entry point ───────────────────────────────────────────
    def ping(self, category: str, message: str, *, critical: bool = False,
             priority: Optional[str] = None, title: str = "",
             dedupe_key: Optional[str] = None, dedupe_s: Optional[float] = None,
             away_gate: bool = True) -> str:
        """Decide and (maybe) send one ping. Returns an outcome string
        (QUEUED when it went to the sender). Never raises."""
        try:
            return self._ping(category, message, critical=critical,
                              priority=priority, title=title,
                              dedupe_key=dedupe_key, dedupe_s=dedupe_s,
                              away_gate=away_gate)
        except Exception as e:      # a bug here must never break the caller
            self.log(f"ping failed: {type(e).__name__}: {e}")
            return FAILED

    def _ping(self, category, message, *, critical, priority, title,
              dedupe_key, dedupe_s, away_gate) -> str:
        cat = str(category or "").strip().lower()
        msg = " ".join(str(message or "").split())
        if not msg:
            return EMPTY
        if cat not in CATEGORIES or cat == "summary":
            return UNKNOWN
        gate = self._gate(cat)
        if gate:
            self.last_outcome[cat] = gate
            return gate
        now_m = float(self.mono_now())
        if dedupe_key:
            # Before the scrub: a hook that re-reports the same event on every
            # MQTT push costs one dict lookup, not a regex pass.
            with self._lock:
                last = self._dedupe.get(dedupe_key)
            window = DEDUPE_S if dedupe_s is None else float(dedupe_s)
            if last is not None and now_m - last < window:
                self.last_outcome[cat] = DEDUPED
                return DEDUPED
        text, withheld = scrub(msg, env=self._env())
        if withheld or not text.strip():
            text = _WITHHELD[cat]
        text = _clip(text)
        now_w = self.wall_now()
        with self._lock:
            self._load()
            if critical:
                out = (RATE_LIMITED if self._count_last_hour(now_m, critical=True)
                       >= CRITICAL_MAX_PER_HOUR else QUEUED)
            elif self._focus():
                out = FOCUS
            elif away_gate and self._owner_present():
                out = PRESENT
            elif self.in_quiet_hours(now_w):
                out = QUIET
            elif self._count_last_hour(now_m, critical=False) >= int(
                    self._num("PHONE_PING_MAX_PER_HOUR")):
                out = RATE_LIMITED
            else:
                out = QUEUED
            if dedupe_key and out not in _RETRYABLE:
                self._dedupe[dedupe_key] = now_m
                self._soft_seen.pop(dedupe_key, None)
                if len(self._dedupe) > 256:
                    for k in sorted(self._dedupe, key=self._dedupe.get)[:64]:
                        self._dedupe.pop(k, None)
            elif dedupe_key:
                # A retry (he is here / focus is on) of an event already noted
                # as such: no second journal line, no disk write, no log.
                if self._soft_seen.get(dedupe_key) == out:
                    self.last_outcome[cat] = out
                    return out
                self._soft_seen[dedupe_key] = out
                while len(self._soft_seen) > 64:
                    self._soft_seen.popitem(last=False)
            entry = self._journal_add(now_w.timestamp(), cat, text, out)
            if out == QUIET:
                self._held.append({"t": now_w.timestamp(), "cat": cat,
                                   "text": text, "out": QUIET})
                del self._held[:-HELD_MAX]
            if out == QUEUED:
                self._sent_at.append((now_m, bool(critical)))
            if out in _JOURNALLED:
                self._save()
        self.last_outcome[cat] = out
        if out == QUEUED:
            prio = priority or ("high" if critical else "normal")
            self._dispatch(text, cat, prio, title, entry)
        self.log(f"{cat}: {out}")
        return out

    def _env(self):
        try:
            return self.env()
        except Exception:
            return os.environ

    # ── delivery ─────────────────────────────────────────────────────────
    def _dispatch(self, text: str, cat: str, priority: str, title: str,
                  entry: Optional[dict]) -> None:
        send = self.send

        def job() -> None:
            ok = False
            try:
                res = send(text, priority=priority, title=title or "JARVIS",
                           category=cat) if send is not None else None
                if isinstance(res, dict):
                    ok = any(bool(v) for v in res.values())
                else:
                    ok = bool(res)
            except Exception as e:
                self.log(f"{cat}: send raised {type(e).__name__}")
            if entry is not None:
                with self._lock:
                    entry["out"] = SENT if ok else FAILED
                    if entry in self._journal:
                        self._save()
            if not ok:
                self.log(f"{cat}: send failed")

        try:
            accepted = self.spawn(job)
        except Exception:
            accepted = False
        if accepted is False and entry is not None:
            with self._lock:
                entry["out"] = FAILED
            self.log(f"{cat}: sender queue full - dropped")

    def _thread_spawn(self, job: Callable[[], None]) -> bool:
        with self._lock:
            if self._q is None:
                self._q = queue.Queue(maxsize=SEND_QUEUE_MAX)
            if self._sender is None or not self._sender.is_alive():
                t = threading.Thread(target=self._sender_loop,
                                     name="phone-ping-sender", daemon=True)
                t.start()
                self._sender = t
            q = self._q
        try:
            q.put_nowait(job)
            return True
        except queue.Full:
            return False

    def _sender_loop(self) -> None:  # pragma: no cover - daemon; jobs are unit-tested via an inline spawn
        while True:
            q = self._q
            if q is None:
                return
            job = q.get()
            try:
                job()
            except Exception:
                pass

    def _deliver_internal(self, text: str, *, cat: str, now_w) -> str:
        """The digest / summary path: master + bridge + staging + focus only
        (each item already passed its own category's gates)."""
        if not self.active():
            return DISABLED
        if self._focus():
            return FOCUS
        text, withheld = scrub(text, env=self._env())
        if withheld:
            text = _WITHHELD["summary"]
        text = _clip(text, 1200)
        self._dispatch(text, cat, "normal", "JARVIS", None)
        self.last_outcome[cat] = QUEUED
        self.log(f"{cat}: queued")
        return QUEUED

    # ── the watcher's work ───────────────────────────────────────────────
    def tick(self) -> None:
        """One watcher pass. Cheap when nothing is due. Never raises."""
        try:
            if not self.active():
                return
            self._hook_bambu()
            self._check_pending()
            now_w = self.wall_now()
            self._maybe_release_held(now_w)
            self._maybe_summary(now_w)
        except Exception as e:
            self.log(f"tick failed: {type(e).__name__}: {e}")

    def _hook_bambu(self) -> None:
        mod = sys.modules.get("skill_bambu_monitor")
        if mod is None or mod is self._bambu_hooked:
            return
        reg = getattr(mod, "register_state_change_hook", None)
        if not callable(reg):
            return
        reg(self.on_bambu_state)
        self._bambu_hooked = mod

    def _check_pending(self) -> None:
        after_s = self._num("PHONE_PING_CONFIRM_AFTER_MIN") * 60.0
        try:
            items = list(self.pending() or [])
        except Exception:
            items = []
        for item in items:
            key = str(item.get("key") or "")
            if not key or key in self._pending_done:
                continue
            try:
                age = float(item.get("age_s") or 0.0)
            except (TypeError, ValueError):
                continue
            if age < after_s:
                continue
            out = self.ping("confirm", str(item.get("text") or ""),
                            dedupe_key="confirm:" + key)
            if out not in _RETRYABLE:
                self._pending_done[key] = True
                while len(self._pending_done) > 64:
                    self._pending_done.popitem(last=False)

    def _summary_enabled(self) -> bool:
        return self._flag("PHONE_PING_SUMMARY")

    def _summary_carries_held(self, now_w) -> bool:
        """True when the daily summary is on, has not gone out today and is
        due within FOLD_INTO_SUMMARY_MIN: it will carry the held pings, so a
        separate digest would only say the same thing twice."""
        if not self._summary_enabled():
            return False
        today = now_w.strftime("%Y-%m-%d")
        if today in (self._summary_sent_on, self._summary_skipped_on):
            return False
        due = self._minutes("PHONE_PING_SUMMARY_TIME")
        wait = due - (now_w.hour * 60 + now_w.minute)
        return 0 <= wait <= FOLD_INTO_SUMMARY_MIN

    def _maybe_release_held(self, now_w) -> None:
        """Quiet hours are over: send what was held as ONE message (unless the
        summary is about to carry it)."""
        with self._lock:
            self._load()
            if not self._held or self.in_quiet_hours(now_w):
                return
            if self._summary_carries_held(now_w):
                return
            held = list(self._held)
        lines = [f"{_dt.datetime.fromtimestamp(h['t']).strftime('%H:%M')} "
                 f"{h['text']}" for h in held[-SUMMARY_LINES:]]
        extra = len(held) - len(lines)
        text = ("While you were away overnight, sir:\n• " + "\n• ".join(lines))
        if extra > 0:
            text += f"\n…and {extra} more."
        if self._deliver_internal(text, cat="digest", now_w=now_w) == QUEUED:
            with self._lock:
                self._held = []
                self._save()

    def _maybe_summary(self, now_w) -> None:
        if not self._summary_enabled():
            return
        today = now_w.strftime("%Y-%m-%d")
        with self._lock:
            self._load()
            if today in (self._summary_sent_on, self._summary_skipped_on):
                return
        due = self._minutes("PHONE_PING_SUMMARY_TIME")
        now_min = now_w.hour * 60 + now_w.minute
        if now_min < due:
            return
        if now_min - due > SUMMARY_WINDOW_H * 60:
            # JARVIS was down at summary time: a "morning summary" at 3 pm is
            # stale, so skip today rather than send it late.
            with self._lock:
                self._summary_skipped_on = today
                self._save()
            self.log("summary: missed today's slot - skipped")
            return
        text = self.build_summary(now_w)
        if self._deliver_internal(text, cat="summary", now_w=now_w) == QUEUED:
            with self._lock:
                self._summary_sent_on = today
                self._last_summary_t = now_w.timestamp()
                self._held = []
                self._save()

    def build_summary(self, now_w) -> str:
        with self._lock:
            self._load()
            since = self._last_summary_t or (now_w.timestamp() - 86400.0)
            events = [e for e in self._journal if e["t"] > since]
        since_dt = _dt.datetime.fromtimestamp(since)
        same_day = since_dt.date() == now_w.date()
        since_label = (since_dt.strftime("%H:%M") if same_day
                       else "yesterday " + since_dt.strftime("%H:%M")
                       if (now_w.date() - since_dt.date()).days == 1
                       else since_dt.strftime("%b %d %H:%M"))
        head = "Morning summary" if now_w.hour < 12 else (
            "Afternoon summary" if now_w.hour < 17 else "Evening summary")
        parts = []
        if events:
            lines = [f"{_dt.datetime.fromtimestamp(e['t']).strftime('%H:%M')} "
                     f"{e['text']}" for e in events[-SUMMARY_LINES:]]
            extra = len(events) - len(lines)
            noun = "thing" if len(events) == 1 else "things"
            parts.append(f"{head}, sir. {len(events)} {noun} since "
                         f"{since_label}:\n• " + "\n• ".join(lines))
            if extra > 0:
                parts.append(f"…and {extra} earlier.")
        else:
            parts.append(f"{head}, sir: all quiet since {since_label}. Nothing "
                         f"needed you.")
        try:
            now_line = self.printer_now() or ""
        except Exception:
            now_line = ""
        if now_line:
            parts.append(now_line)
        return "\n".join(parts)

    # ── event sources ────────────────────────────────────────────────────
    def on_bambu_state(self, snapshot, prev_gcode, gcode_state) -> None:
        """skills/bambu_monitor state-change hook (runs on the MQTT thread:
        decides and queues, never sends inline). Pings on a TRANSITION into
        FINISH / FAILED from an active state, and on a pause with an error;
        the first observation after a restart (prev None) never pings."""
        try:
            snap = snapshot if isinstance(snapshot, dict) else {}
            cur = str(gcode_state or "").upper()
            prev = str(prev_gcode or "").upper()
            name = _print_name(snap)
            quoted = f"'{name}'" if name else "your print"
            key_name = name or "_anon_"
            if cur == "FINISH" and prev in _PRINT_ACTIVE:
                self.ping("print", f"Print finished, sir: {quoted} is done.",
                          dedupe_key=f"print:finish:{key_name}")
            elif cur == "FAILED" and prev in _PRINT_ACTIVE:
                self.ping("print",
                          f"Print failed{_layer_phrase(snap)}, sir: {quoted}. "
                          f"You'll want to check the printer.",
                          priority="high",
                          dedupe_key=f"print:failed:{key_name}")
            elif cur == "PAUSE":
                err = snap.get("print_error")
                if err not in (None, 0, "0", "", "00000000"):
                    self.ping("print",
                              f"The print is paused with error {err}"
                              f"{_layer_phrase(snap)}, sir: {quoted} needs "
                              f"you.", priority="high",
                              dedupe_key=f"print:pause:{key_name}:{err}")
        except Exception as e:
            self.log(f"bambu hook failed: {type(e).__name__}")

    # ── read-outs ────────────────────────────────────────────────────────
    def status(self) -> dict:
        now_m = float(self.mono_now())
        with self._lock:
            self._load()
            last_sent = [e for e in self._journal if e["out"] in (QUEUED, SENT)]
            snap = {
                "enabled": self._flag("PHONE_PING_ENABLED"),
                "configured": self.bridge_configured(),
                "categories": {c: self._flag(f) for c, f in CATEGORY_FLAGS.items()},
                "sent_last_hour": self._count_last_hour(now_m, critical=False),
                "critical_last_hour": self._count_last_hour(now_m, critical=True),
                "max_per_hour": int(self._num("PHONE_PING_MAX_PER_HOUR")),
                "quiet_start": _hhmm(self._minutes("PHONE_PING_QUIET_START")),
                "quiet_end": _hhmm(self._minutes("PHONE_PING_QUIET_END")),
                "summary_time": _hhmm(self._minutes("PHONE_PING_SUMMARY_TIME")),
                "held": len(self._held),
                "last": dict(last_sent[-1]) if last_sent else None,
            }
        try:
            snap["blocked"] = self.blocked() or ""
        except Exception:
            snap["blocked"] = "error"
        return snap


# ─── module singleton ────────────────────────────────────────────────────

_PINGER: list = [None]
_PINGER_LOCK = threading.Lock()
_watcher: list = [None]
_watcher_stop: list = [threading.Event()]


def get_pinger() -> PhonePinger:
    with _PINGER_LOCK:
        if _PINGER[0] is None:
            _PINGER[0] = PhonePinger()
        return _PINGER[0]


def attach_bridge(send: Callable, configured: Callable[[], bool]) -> PhonePinger:
    """skills/phone_bridge.register(): the sender (text, *, priority, title,
    category) -> {backend: ok} and the "can it send unsolicited?" probe."""
    p = get_pinger()
    p.send = send
    p.configured = configured
    return p


def ping(category: str, message: str, **kwargs) -> str:
    """skill_utils["ping_phone"] lands here. See PhonePinger.ping."""
    return get_pinger().ping(category, message, **kwargs)


def status() -> dict:
    return get_pinger().status()


def _watch_loop(stop: threading.Event) -> None:  # pragma: no cover - daemon loop; tick() is unit-tested directly
    while not stop.wait(TICK_S):
        get_pinger().tick()


def start_watcher() -> bool:
    """Start the TICK_S watcher once (idempotent). False when pings cannot run
    (master off / bridge not configured / staging) - nothing is started."""
    p = get_pinger()
    if not p.active():
        return False
    t = _watcher[0]
    if t is not None and t.is_alive():
        return True
    stop = threading.Event()
    _watcher_stop[0] = stop
    t = threading.Thread(target=_watch_loop, args=(stop,),
                         name="phone-ping-watcher", daemon=True)
    t.start()
    _watcher[0] = t
    return True


def stop_watcher() -> None:
    _watcher_stop[0].set()
    _watcher[0] = None


def _reset_for_tests() -> None:
    stop_watcher()
    with _PINGER_LOCK:
        _PINGER[0] = None
