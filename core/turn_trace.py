"""One structured JSON line per owner turn (v3 W20, TURN_TRACE).

Why: every "flip it on when it looks right" decision (W02 head-hold, W09 learn
pre-filter, W11 action-fit, W21 loop guard) used to be measured by regex over
session logs. This keeps ONE record per turn instead, in
``data/turns/turns-YYYY-MM-DD.jsonl``, and ``tools/turn_trace_report.py``
summarises a day from it. The idea (a trace collector + store) is OpenJarvis's
``traces/`` package (Apache-2.0); no code is copied from it.

One record (keys with no value are left out; every string is scrubbed)::

    {"v": 1, "ts": 1790000000.0, "day": "2026-10-27",
     "channel": "voice",              # voice | typed | web (how the owner spoke)
     "kind": "voice",                 # the [turn-timing] kind
     "outcome": "ok",                 # ok | error | shortcut
     "you_chars": 11,                 # LENGTH of the owner's words (never the text)
     "reply_chars": 20,               # LENGTH of the reply (never the text)
     "sections": ["Browser", "Media"],  # routed prompt sections
     "actions": [{"n": "open_app", "ok": true}, ...],
     "cut": "no new progress",        # why the follow-up chain stopped
     "rounds": 1,                     # follow-up rounds
     "validators": {"validation": 1},  # lines the speech/claim validators printed
     "shadow": {"head-hold": 1},      # W02/W09/W11/W21 "would ..." lines
     "model": "gemma4:26b-a4b-it-qat",
     "pe_state": "warm",              # warm | partial | full (prefill estimate)
     "llm_calls": 1, "filler": 0,
     "stages": {"stt": 1578, "llm": 2760, "actions": 8,
                "first_play": 5120, "audible": 5207, "end": 5890}}   # ms

``stages`` are read from the turn's own ``[turn-timing]`` line (core.turn_timing
is the one source of the numbers): stt = stt_end - stt_start, llm = llm_done -
llm_post, actions = actions_done - llm_done; first_play / audible / end are ms
from the turn's t0 (the VAD break for a spoken turn).

PRIVACY. The file is local, never shared with the peer machine (only settings
are) and not part of any Discord / phone path. It holds NO free text: neither
the owner's words nor the reply, only their lengths (security review DAR-1,
2026-10-27: the scrubber missed spoken PINs and door / alarm / verification
codes, and tools/turn_trace_report.py never read either field). Action names,
section headers, validator / shadow tags and timings are all it keeps, so
outside text (a web page, a caption, the Agent Office API) never lands here
and nothing read back from this file can start an action: the file is only
ever read by tools/turn_trace_report.py.

Writes are debounced and run on a daemon timer, never on the voice thread:
``commit`` appends the finished record to a bounded queue and schedules ONE
timer-driven flush (the pattern of core/llm_usage.py). The flush also runs at
atexit. A day file is append-only; when it passes MAX_FILE_BYTES it is renamed
``turns-YYYY-MM-DD.N.jsonl`` and a fresh one starts. Files older than
RETENTION_DAYS are deleted (at most one sweep a day). Nothing here raises into
the turn: a full or failing disk costs the record, never the reply.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
from collections import deque
from typing import Optional

SCHEMA_VERSION = 1
RETENTION_DAYS = 30
MAX_FILE_BYTES = 8 * 1024 * 1024
FLUSH_DELAY_S = 2.0
QUEUE_MAX = 500
TEXT_MAX = 200          # owner words and reply: first N characters only
FILE_PREFIX = "turns-"
FILE_SUFFIX = ".jsonl"
DIR_NAME = "turns"

# The shadow-mode "would ..." lines of the v3 items, by log tag. A line is
# counted only when it carries the word "would" (shadow's wording), so the
# 'on' mode's "skipped" / "not speaking" lines are not mistaken for shadow.
SHADOW_TAGS = ("head-hold", "learn-prefilter", "action-fit", "loop-guard")
# Lines the speech / claim validators and guards print when they act.
VALIDATOR_TAGS = ("validation", "speech-hygiene", "ack-hold", "result-hold",
                  "claim", "timers", "send-guard", "auth-guard",
                  "autocorrect", "turn-check")

_LINE_RE = re.compile(r"^\s*(?:\[\d\d:\d\d:\d\d\]\s*)?\[([a-z0-9_-]+)\]\s?(.*)",
                      re.DOTALL)
_DAY_RE = re.compile(r"^turns-(\d{4}-\d{2}-\d{2})(?:\.(\d+))?\.jsonl$")
_MAX_ACTIONS = 40

_lock = threading.Lock()          # guards the collector and the queue
_flush_lock = threading.Lock()    # one writer of the files at a time
_cur: dict = {}                   # the open turn's collector
_active = [False]                 # between begin() and commit()
_carry: dict = {}                 # shadow lines seen while no turn was open
_queue: deque = deque()
_timer = [None]
_atexit_armed = [False]
_last_prune_day = [None]
_dropped = [0]                    # records lost to a full queue (diagnostic)


# ── paths ─────────────────────────────────────────────────────────────────

def trace_dir() -> str:
    from core import paths
    return os.path.join(paths.data_dir(), DIR_NAME)


def day_file(day: str, directory: Optional[str] = None) -> str:
    return os.path.join(directory or trace_dir(),
                        f"{FILE_PREFIX}{day}{FILE_SUFFIX}")


def _today(now: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d",
                         time.localtime(time.time() if now is None else now))


# ── scrubbing ─────────────────────────────────────────────────────────────

def scrub_text(text, limit: int = TEXT_MAX) -> str:
    """``text`` with secrets redacted, whitespace collapsed and cut to
    ``limit`` characters. Text that still reads like a credential after the
    redaction comes back as '' (it is not stored at all). Never raises."""
    try:
        s = " ".join(str(text or "").split())
        if not s:
            return ""
        from core import phone_ping
        clean, withheld = phone_ping.scrub(s)
        if withheld:
            return ""
        try:
            from core import bug_reporter
            clean = bug_reporter.scrub(clean)
        except Exception:
            pass
        clean = " ".join(clean.split())
        return clean[:limit]
    except Exception:
        return ""


def _raw(text) -> str:
    """Whitespace-collapsed and bounded, NOT yet scrubbed: scrub_text can
    read a file on its first call, so it runs on the writer's timer thread
    (see _finalize), never on the voice thread."""
    try:
        return " ".join(str(text or "").split())[:TEXT_MAX * 2]
    except Exception:
        return ""


def _finalize(rec: dict) -> dict:
    """Last line of defence on the writer thread: a queued record never
    carries the owner's words or the reply (DAR-1), only their lengths."""
    rec.pop("you", None)
    rec.pop("reply", None)
    return rec


# ── the open turn's collector ─────────────────────────────────────────────

def begin() -> None:
    """A new turn started: clear the collector. Shadow lines counted while no
    turn was open ride with this turn (see observe). Never raises."""
    try:
        with _lock:
            _cur.clear()
            _cur["shadow"] = dict(_carry)
            _carry.clear()
            _active[0] = True
    except Exception:
        pass


def note_text(text, channel: str = "") -> None:
    """The owner's words and how they arrived. The first call of a turn wins
    (the same text is offered by several paths). Never raises."""
    try:
        with _lock:
            if not _active[0] or "you" in _cur:
                return
            _cur["you"] = _raw(text)
            if channel:
                _cur["channel"] = str(channel)[:12]
    except Exception:
        pass


def note_sections(headers) -> None:
    """The prompt sections routed for this turn (their header names)."""
    try:
        names = []
        for h in headers or ():
            h = " ".join(str(h or "").split())[:40]
            if h and h not in names:
                names.append(h)
        if names:
            with _lock:
                if _active[0]:
                    _cur["sections"] = names[:12]
    except Exception:
        pass


def note_chain(reply, results, cut: str = "", rounds: int = 0,
               failed=None) -> None:
    """The follow-up chain is over. ``results`` is the chain's
    ``(name, result, informative)`` tuples; ``failed(result)`` says whether
    one failed. Only action NAMES and ok/fail are kept, never a result
    text. Never raises."""
    try:
        actions = []
        for item in list(results or ())[:_MAX_ACTIONS]:
            try:
                name = str(item[0])
                result = item[1]
            except Exception:
                continue
            ok = True
            if failed is not None:
                try:
                    ok = not failed(result)
                except Exception:
                    ok = True
            actions.append({"n": name[:48], "ok": bool(ok)})
        with _lock:
            if not _active[0]:
                return
            _cur["reply"] = _raw(reply)
            _cur["actions"] = actions
            if cut:
                _cur["cut"] = str(cut)[:40]
            if rounds:
                _cur["rounds"] = int(rounds)
    except Exception:
        pass


def note_model(model) -> None:
    try:
        m = str(model or "").strip()[:80]
        if m:
            with _lock:
                if _active[0]:
                    _cur["model"] = m
    except Exception:
        pass


def observe(msg) -> None:
    """Look at one chunk the log tee is writing (cheap: most chunks return at
    the first test). A ``[tag] ... would ...`` line of a W02/W09/W11/W21 tag
    is counted under ``shadow``; a validator tag's line under ``validators``.
    Counted into the open turn; a shadow line with no turn open (the learn
    pre-filter runs after the dispatch) rides with the next turn, so a day's
    totals stay whole. Only the TAG is kept, never the line. Never raises."""
    try:
        if not msg or "[" not in msg:
            return
        m = _LINE_RE.match(msg)
        if not m:
            return
        tag, rest = m.group(1), m.group(2)
        if tag in SHADOW_TAGS:
            if "would" not in rest:
                return
            with _lock:
                bucket = (_cur.setdefault("shadow", {}) if _active[0]
                          else _carry)
                bucket[tag] = bucket.get(tag, 0) + 1
        elif tag in VALIDATOR_TAGS or tag.startswith("claim"):
            with _lock:
                if _active[0]:
                    v = _cur.setdefault("validators", {})
                    v[tag] = v.get(tag, 0) + 1
    except Exception:
        pass


# ── building a record ─────────────────────────────────────────────────────

def _ms(kv: dict, key: str) -> Optional[int]:
    v = kv.get(key)
    try:
        if v is None or v == "-":
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def stages_from_line(kv: dict) -> dict:
    """Stage latencies (ms) from a parsed [turn-timing] line."""
    out = {}

    def span(name, a, b):
        x, y = _ms(kv, a), _ms(kv, b)
        if x is not None and y is not None and y >= x:
            out[name] = y - x

    span("stt", "stt_start", "stt_end")
    span("llm", "llm_post", "llm_done")
    span("actions", "llm_done", "actions_done")
    for name, key in (("first_play", "first_play"), ("audible", "audible_ms"),
                      ("end", "end")):
        v = _ms(kv, key)
        if v is not None and v >= 0:
            out[name] = v
    return out


def build_record(line: str, outcome: str, collected: dict,
                 now: Optional[float] = None) -> dict:
    """The record for one finished turn from its [turn-timing] ``line``."""
    from core import turn_timing
    kv = turn_timing.parse_line(line or "") or {}
    now = time.time() if now is None else now
    rec = {"v": SCHEMA_VERSION, "ts": round(now, 1), "day": _today(now)}
    kind = kv.get("kind") or ""
    rec["kind"] = kind
    rec["channel"] = collected.get("channel") or (
        "voice" if kind == "voice" else "typed")
    rec["outcome"] = outcome or kv.get("outcome") or ""
    # Lengths only, never the text (DAR-1).
    for key in ("you", "reply"):
        val = collected.get(key)
        if val:
            rec[key + "_chars"] = len(val)
    for key in ("sections", "actions", "cut", "rounds",
                "validators", "shadow", "model"):
        val = collected.get(key)
        if val:
            rec[key] = val
    pe = kv.get("pe_state")
    if pe and pe != "-":
        rec["pe_state"] = pe
    for key in ("llm_calls", "filler"):
        v = _ms(kv, key)
        if v is not None:
            rec[key] = v
    rec["stages"] = stages_from_line(kv)
    return rec


def commit(line, outcome: str = "ok", enabled: bool = True) -> Optional[dict]:
    """The turn's [turn-timing] line was emitted: close the collector, build
    the record and hand it to the debounced writer. Returns the record (or
    None when disabled or there was no line). Never raises."""
    try:
        with _lock:
            collected = dict(_cur)
            was_active = _active[0]
            _cur.clear()
            _active[0] = False
        if not enabled or not line or not was_active:
            return None
        rec = build_record(line, outcome, collected)
        submit(rec)
        return rec
    except Exception:
        return None


# ── the debounced writer ──────────────────────────────────────────────────

def submit(rec: dict) -> None:
    """Queue one record and make sure a flush is scheduled. Returns at once."""
    try:
        with _lock:
            if len(_queue) >= QUEUE_MAX:
                _queue.popleft()
                _dropped[0] += 1
            _queue.append(rec)
        _schedule()
    except Exception:
        pass


def _start_timer(delay: float, fn) -> threading.Timer:
    t = threading.Timer(delay, fn)
    t.daemon = True
    t.start()
    return t


def _schedule() -> None:
    if not _atexit_armed[0]:
        _atexit_armed[0] = True
        atexit.register(flush)
    with _lock:
        if _timer[0] is not None:
            return
        _timer[0] = _start_timer(FLUSH_DELAY_S, _timer_flush)


def _timer_flush() -> None:
    try:
        flush()
    finally:
        with _lock:
            _timer[0] = None
            again = bool(_queue)
        if again:
            _schedule()


def _rotate_if_big(path: str, add: int, directory: str, day: str) -> None:
    try:
        if os.path.getsize(path) + add <= MAX_FILE_BYTES:
            return
    except OSError:
        return
    n = 1
    while os.path.exists(os.path.join(
            directory, f"{FILE_PREFIX}{day}.{n}{FILE_SUFFIX}")):
        n += 1
    os.replace(path, os.path.join(
        directory, f"{FILE_PREFIX}{day}.{n}{FILE_SUFFIX}"))


def flush() -> bool:
    """Append every queued record to its day file. True when the queue is
    empty afterwards. Never raises."""
    if not _flush_lock.acquire(timeout=2.0):
        return False
    try:
        with _lock:
            batch = list(_queue)
            _queue.clear()
        if not batch:
            return True
        directory = trace_dir()
        by_day: dict = {}
        for rec in batch:
            by_day.setdefault(str(rec.get("day") or _today()), []).append(rec)
        try:
            os.makedirs(directory, exist_ok=True)
            for day, recs in by_day.items():
                text = "".join(json.dumps(_finalize(r), ensure_ascii=False,
                                          separators=(",", ":")) + "\n"
                               for r in recs)
                path = day_file(day, directory)
                _rotate_if_big(path, len(text.encode("utf-8")), directory, day)
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(text)
            prune(directory)
            return True
        except Exception as e:
            print(f"  [turn-trace] write failed: {type(e).__name__}")
            return False
    except Exception:
        return False
    finally:
        _flush_lock.release()


def prune(directory: Optional[str] = None, now: Optional[float] = None,
          force: bool = False) -> int:
    """Delete trace files older than RETENTION_DAYS (by the date in the
    name). At most one sweep per calendar day unless ``force``. Returns how
    many files went. Never raises."""
    try:
        today = _today(now)
        if not force and _last_prune_day[0] == today:
            return 0
        _last_prune_day[0] = today
        directory = directory or trace_dir()
        cutoff = _today((time.time() if now is None else now)
                        - RETENTION_DAYS * 86400)
        gone = 0
        for name in os.listdir(directory):
            m = _DAY_RE.match(name)
            if m and m.group(1) < cutoff:
                try:
                    os.remove(os.path.join(directory, name))
                    gone += 1
                except OSError:
                    pass
        return gone
    except Exception:
        return 0


def reset_for_tests() -> None:
    """Clear all module state (tests only)."""
    with _lock:
        _cur.clear()
        _carry.clear()
        _queue.clear()
        _active[0] = False
        t = _timer[0]
        _timer[0] = None
    if t is not None:
        try:
            t.cancel()
        except Exception:
            pass
    _last_prune_day[0] = None
    _dropped[0] = 0
