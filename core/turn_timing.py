"""Per-turn millisecond timing line and Ollama call stats (2026-09-29).

Speed-plan rank 1: every later latency claim has to be provable live, and the
console log only stamps whole seconds. This module turns the stage marks the
monolith already passes through (VAD break, Whisper, the local-LLM POST, the
first audio out) into ONE line per turn::

    [turn-timing] kind=voice outcome=ok vad_break=0 stt_start=3 stt_end=1581 \
you=1584 llm_post=1650 llm_done=4410 actions_done=4418 synth_start=4420 \
first_play=5120 end=5890 prompt_eval_count=11873 prompt_eval_ms=2702 \
eval_count=84 eval_ms=790 llm_calls=1 turn_ctx_chars=1342 sys_chars=31012 \
followup_rounds=0 filler=0 filler_ms=- lead_dropped=0

Offsets are integer milliseconds from the turn's t0: the record_speech VAD
break for a spoken turn, the inject-queue drain for a typed/injected turn. A
mark that never happened prints as ``-`` so the schema is fixed and greppable.

An injected turn's offsets EXCLUDE its queue wait: the main loop drains the
inject queue only at the top of an iteration. Since 2026-10-01 an idle
record_speech yields to a newly queued command within ~0.25 s, so the wait is
long only while an utterance is being captured (up to MAX_RECORDING_SECS in a
noisy room) or a turn is running. Measure that wait from the queue file's own
timestamp, not from this line.

llm_done and the prompt/eval counters describe the first local-LLM response
that actually ANSWERED (non-empty text) on the turn's thread; an HTTP error or
an empty reply before a failover still counts toward llm_calls only.

synth_start / first_play count only audio from the turn's own thread or a
helper it adopted (the streaming-TTS flush threads), and only after "you": a
reminder, tray command or mid-task status line spoken by another thread is not
the answer's first audio. The processing filler is tracked separately
(filler / filler_ms).

lead_dropped is 1 when the answer-first rule (ANSWER_FIRST_ENABLED) skipped the
model's short lead-in ("One moment, sir.") on this turn, so first_play then
measures the real answer rather than the lead-in.

Contracts (the monolith relies on every one):
  * print-only: nothing here changes behaviour, and no method ever raises;
  * thread-safe: the filler thread and the sentence-flush threads call _speak
    concurrently with the turn thread, so every state change is under one lock
    and the print happens outside it;
  * cheap: no I/O besides the single print at the end of the turn, no extra
    Ollama calls (the stats come from the response the turn already has).

Stdlib only, so the CI-light tier covers it (tests/test_turn_timing.py).
"""
from __future__ import annotations

import sys
import threading
import time

# Stage marks, in the order a plain local-LLM voice turn passes them. The line
# prints them in this order; offsets are monotonic along it for such a turn.
MARKS = ("vad_break", "stt_start", "stt_end", "you", "llm_post", "llm_done",
         "actions_done", "synth_start", "first_play")

# The answer's audio marks. They only count once the transcript has been
# accepted ("you") — a reminder drained between the inject drain and the
# transcript is not the answer's first audio — and only from the turn's own
# thread or a helper thread it adopted (see TurnTiming.adopt): a reminder,
# tray command or mid-task status line spoken concurrently by another thread
# must not be recorded as the answer's first audio.
_AFTER_YOU = frozenset(("synth_start", "first_play"))

# Stats fields, printed after the marks in this order.
STAT_FIELDS = ("prompt_eval_count", "prompt_eval_ms", "eval_count", "eval_ms",
               "llm_calls", "turn_ctx_chars", "sys_chars", "followup_rounds",
               "filler", "filler_ms", "lead_dropped")

# Local-LLM wrappers: when one of these is the direct caller of
# _call_local_llm, the caller tag also names the function that called IT, so
# a background _llm_quick is distinguishable from the voice turn's call.
LLM_WRAPPERS = frozenset(("_llm_quick", "_local_then_cloud_or_honest",
                          "_local_fallback_or"))


def _ms(ns) -> "int | None":
    """Ollama durations are nanoseconds; None when absent or unusable."""
    try:
        if ns is None or isinstance(ns, bool):
            return None
        return int(round(float(ns) / 1e6))
    except Exception:
        return None


def _count(v) -> "int | None":
    try:
        if v is None or isinstance(v, bool):
            return None
        return int(v)
    except Exception:
        return None


def ollama_stats(j) -> dict:
    """Pull Ollama's own per-request counters out of an /api/chat JSON body.

    Returns a dict with prompt_eval_count, prompt_eval_ms, eval_count, eval_ms,
    load_ms and total_ms; a field the body does not carry is None (Ollama omits
    prompt_eval_count entirely on a full KV-cache hit on some versions, and a
    non-JSON / error body carries none of them). Never raises."""
    out = {"prompt_eval_count": None, "prompt_eval_ms": None,
           "eval_count": None, "eval_ms": None,
           "load_ms": None, "total_ms": None}
    try:
        if not isinstance(j, dict):
            return out
        out["prompt_eval_count"] = _count(j.get("prompt_eval_count"))
        out["prompt_eval_ms"] = _ms(j.get("prompt_eval_duration"))
        out["eval_count"] = _count(j.get("eval_count"))
        out["eval_ms"] = _ms(j.get("eval_duration"))
        out["load_ms"] = _ms(j.get("load_duration"))
        out["total_ms"] = _ms(j.get("total_duration"))
    except Exception:
        pass
    return out


def response_stats(r) -> dict:
    """ollama_stats() of a requests-style response (anything with .json()).
    A body that is not JSON gives all-None stats. Never raises."""
    try:
        return ollama_stats(r.json())
    except Exception:
        return ollama_stats(None)


def _q(v) -> str:
    return "?" if v is None else str(v)


def served_via_suffix(stats, caller: str = "") -> str:
    """The text appended to a ``[local-llm] served via <model>`` line:
    ``pe=<count>/<ms> ev=<count> caller=<tag>``. Unknown values print as
    ``?``. A small pe count on a long prompt is a KV-cache hit; a count near
    the full prompt is a cold re-evaluation. Never raises."""
    try:
        s = stats if isinstance(stats, dict) else {}
        out = (f"pe={_q(s.get('prompt_eval_count'))}/"
               f"{_q(s.get('prompt_eval_ms'))} "
               f"ev={_q(s.get('eval_count'))}")
        if caller:
            out += f" caller={caller}"
        return out
    except Exception:
        return "pe=?/? ev=?"


def caller_tag(frame=None, wrappers=LLM_WRAPPERS) -> str:
    """Name the code that asked for a local-LLM call.

    `frame` is the CALLER's frame (pass ``sys._getframe(1)`` from inside the
    called function). Gives ``<func>`` or, when that function is one of the
    thin `wrappers`, ``<wrapper><<its caller>``; then ``@<thread name>`` so a
    background worker is obvious even when the function name is generic.
    Returns ``?`` on any failure. Never raises."""
    try:
        f = frame if frame is not None else sys._getframe(2)
        names = []
        while f is not None and len(names) < 3:
            names.append(f.f_code.co_name)
            if names[-1] not in wrappers:
                break
            f = f.f_back
        tag = "<".join(names) or "?"
        # Python 3.10+ names an unnamed Thread "Thread-N (target)": collapse
        # the whitespace so the tag stays one key=value token.
        thread = "_".join(str(threading.current_thread().name).split()) or "?"
        return f"{tag}@{thread}"
    except Exception:
        return "?"


def _new_turn(kind: str, t0: float, owner: int) -> dict:
    return {"kind": kind, "t0": t0, "owner": owner, "marks": {},
            "stats": {}, "llm_calls": 0, "followup_rounds": 0,
            "filler": 0, "filler_at": None, "lead_dropped": 0,
            "helpers": []}


class TurnTiming:
    """One active turn at a time; see the module docstring.

    `print_fn` receives the finished line (default: print). `clock` is the
    monotonic seconds source (default: time.perf_counter; tests pass a fake).
    """

    def __init__(self, print_fn=None, clock=None):
        self._lock = threading.Lock()
        self._print = print_fn if print_fn is not None else print
        self._clock = clock if clock is not None else time.perf_counter
        self._turn = None
        self._last_vad = None

    # ── turn lifecycle ────────────────────────────────────────────────────
    def now(self) -> "float | None":
        try:
            return self._clock()
        except Exception:
            return None

    def note_vad_break(self) -> None:
        """record_speech's end-of-utterance break. Only remembered; the turn
        that consumes it begins in the capture path (begin_voice)."""
        try:
            self._last_vad = self._clock()
        except Exception:
            pass

    def begin(self, kind: str, t0=None) -> None:
        """Start a new turn on the calling thread, replacing (silently) any
        turn that never reached an emit — a filtered / gated utterance."""
        try:
            now = self._clock()
            start = now if t0 is None else t0
            turn = _new_turn(kind, start, threading.get_ident())
            with self._lock:
                self._turn = turn
        except Exception:
            pass

    def begin_voice(self, since=None) -> None:
        """Start a spoken turn at the last VAD break, provided that break
        happened at or after `since` (the clock value taken just before the
        record_speech call); otherwise at now. vad_break is offset 0."""
        try:
            vad = self._last_vad
            use_vad = vad is not None and (since is None or vad >= since)
            self.begin("voice", t0=vad if use_vad else None)
            if use_vad:
                with self._lock:
                    if self._turn is not None:
                        self._turn["marks"]["vad_break"] = vad
        except Exception:
            pass

    def discard(self) -> None:
        try:
            with self._lock:
                self._turn = None
        except Exception:
            pass

    reset = discard

    def active(self) -> bool:
        try:
            with self._lock:
                return self._turn is not None
        except Exception:
            return False

    # ── marks ─────────────────────────────────────────────────────────────
    def adopt(self, thread) -> None:
        """The turn's own thread hands part of the answer's speech to
        `thread` (a threading.Thread, adopted BEFORE it starts so its first
        mark cannot race the adoption). Its synth_start / first_play then
        count as the turn's. Calls from any other thread are ignored."""
        try:
            ident = threading.get_ident()
            with self._lock:
                t = self._turn
                if t is not None and ident == t["owner"] and thread is not None:
                    t["helpers"].append(thread)
        except Exception:
            pass

    def mark(self, name: str, owner_only: bool = False) -> None:
        """Record the FIRST occurrence of `name` in the active turn.
        owner_only: ignore calls from any thread but the turn's own (the
        background LLM callers). first_play/synth_start only count after
        'you', and only from the turn's thread or an adopted helper."""
        try:
            now = self._clock()
            ident = threading.get_ident()
            cur = threading.current_thread()
            with self._lock:
                t = self._turn
                if t is None or name in t["marks"]:
                    return
                if owner_only and ident != t["owner"]:
                    return
                if name in _AFTER_YOU:
                    if "you" not in t["marks"]:
                        return
                    if ident != t["owner"] and not any(
                            h is cur for h in t["helpers"]):
                        return
                t["marks"][name] = now
        except Exception:
            pass

    def llm_response(self, stats, served: bool = True) -> None:
        """The turn thread got a local-LLM response. Every one counts toward
        llm_calls; the first one that SERVED the reply (served=True: non-empty
        text) marks llm_done and supplies the prompt/eval stats, so an HTTP
        error or an empty reply before a failover cannot stand in for the
        answer. Calls from other threads are ignored."""
        try:
            now = self._clock()
            ident = threading.get_ident()
            with self._lock:
                t = self._turn
                if t is None or ident != t["owner"]:
                    return
                t["llm_calls"] += 1
                if served and "llm_done" not in t["marks"]:
                    t["marks"]["llm_done"] = now
                    s = stats if isinstance(stats, dict) else {}
                    for k in ("prompt_eval_count", "prompt_eval_ms",
                              "eval_count", "eval_ms"):
                        t["stats"][k] = s.get(k)
        except Exception:
            pass

    def set_first(self, key: str, value) -> None:
        """Turn-thread-only stat, first value wins (the main call's prompt,
        not a follow-up round's)."""
        try:
            ident = threading.get_ident()
            with self._lock:
                t = self._turn
                if t is None or ident != t["owner"] or key in t["stats"]:
                    return
                t["stats"][key] = value
        except Exception:
            pass

    def followup_round(self) -> None:
        try:
            ident = threading.get_ident()
            with self._lock:
                t = self._turn
                if t is not None and ident == t["owner"]:
                    t["followup_rounds"] += 1
        except Exception:
            pass

    def note_filler(self) -> None:
        """The processing filler played a clip (any thread)."""
        try:
            now = self._clock()
            with self._lock:
                t = self._turn
                if t is None:
                    return
                t["filler"] += 1
                if t["filler_at"] is None:
                    t["filler_at"] = now
        except Exception:
            pass

    def note_lead_dropped(self) -> None:
        """The answer-first rule skipped this turn's lead-in. Turn-thread
        only, like followup_round."""
        try:
            ident = threading.get_ident()
            with self._lock:
                t = self._turn
                if t is not None and ident == t["owner"]:
                    t["lead_dropped"] = 1
        except Exception:
            pass

    # ── output ────────────────────────────────────────────────────────────
    def emit(self, outcome: str = "ok") -> "str | None":
        """Finish the active turn and print its line — once. Only the turn's
        own thread can finish it (a dispatch on another thread must not cut a
        voice turn short). Returns the line, or None when nothing was
        printed. Never raises."""
        try:
            now = self._clock()
            ident = threading.get_ident()
            with self._lock:
                t = self._turn
                if t is None or ident != t["owner"]:
                    return None
                self._turn = None
            line = format_line(t, now, outcome)
        except Exception:
            return None
        try:
            self._print(line)
        except Exception:
            pass
        return line


def _off(t0, v) -> str:
    try:
        if v is None:
            return "-"
        return str(int(round((v - t0) * 1000.0)))
    except Exception:
        return "-"


def format_line(turn: dict, end, outcome: str = "ok") -> str:
    """Render one finished turn (see the module docstring for the shape)."""
    t0 = turn["t0"]
    marks = turn["marks"]
    stats = turn["stats"]
    parts = [f"kind={turn['kind']}", f"outcome={outcome}"]
    for m in MARKS:
        parts.append(f"{m}={_off(t0, marks.get(m))}")
    parts.append(f"end={_off(t0, end)}")
    vals = {
        "prompt_eval_count": stats.get("prompt_eval_count"),
        "prompt_eval_ms": stats.get("prompt_eval_ms"),
        "eval_count": stats.get("eval_count"),
        "eval_ms": stats.get("eval_ms"),
        "llm_calls": turn["llm_calls"],
        "turn_ctx_chars": stats.get("turn_ctx_chars"),
        "sys_chars": stats.get("sys_chars"),
        "followup_rounds": turn["followup_rounds"],
        "filler": turn["filler"],
        "lead_dropped": turn.get("lead_dropped", 0),
    }
    for k in STAT_FIELDS:
        if k == "filler_ms":
            parts.append(f"filler_ms={_off(t0, turn['filler_at'])}")
        else:
            v = vals.get(k)
            parts.append(f"{k}={'-' if v is None else v}")
    return "  [turn-timing] " + " ".join(parts)


def parse_line(line: str) -> dict:
    """Inverse of format_line for tests and log tools: {key: str}. Lines
    without the tag give {}. Never raises."""
    try:
        tag = "[turn-timing]"
        i = line.find(tag)
        if i < 0:
            return {}
        out = {}
        for tok in line[i + len(tag):].split():
            k, sep, v = tok.partition("=")
            if sep:
                out[k] = v
        return out
    except Exception:
        return {}
