"""Per-turn millisecond timing line and Ollama call stats (2026-09-29).

Speed-plan rank 1: every later latency claim has to be provable live, and the
console log only stamps whole seconds. This module turns the stage marks the
monolith already passes through (VAD break, Whisper, the local-LLM POST, the
first audio out) into ONE line per turn::

    [turn-timing] kind=voice outcome=ok vad_break=0 stt_start=3 stt_end=1581 \
you=1584 llm_post=1650 llm_done=4410 actions_done=4418 synth_start=4420 \
first_play=5120 end=5890 prompt_eval_count=11873 prompt_eval_ms=2702 \
eval_count=84 eval_ms=790 llm_calls=1 turn_ctx_chars=1342 sys_chars=31012 \
followup_rounds=0 filler=0 filler_ms=- tail_ms=1410 cap_lag_ms=128 \
clip_ms=3904 stt_wait_ms=0 stt_engine=- load_ms=13 total_ms=3512 \
play_open_ms=41 out_lat_ms=46 filler_clip_ms=- eot=- st_p=- st_n=- pre=- \
cut=- amb_deferred=- cache=- clone=- clone_ms=- t3_ms_tok=- clone_cache=- \
audible_ms=5207 keeper=- opens=1 opens_ms=41 lead_dropped=0

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

Speed-plan R1 fields (2026-10-01), printed after filler_ms and before
lead_dropped (NOTE_FIELDS; ``-`` = not measured on this turn):

  tail_ms        measured end of speech -> the captured clip's last sample,
                 in AUDIO time: a speech detector (core/endpointing.py,
                 Silero) run over the clip on a daemon. Only a capture that
                 ended on its VAD break (vad_break=0) ends at an end of
                 speech; one cut at MAX_RECORDING_SECS prints vad_break=-.
  cap_lag_ms     how far the VAD break (t0, a wall-clock instant) trails the
                 clip's last sample: the input stream's reported latency plus
                 the chunks still queued behind the capture loop when it
                 broke (noise suppression running behind real time). A lower
                 bound (the elapsed part of the newest chunk is not counted).
                 Set by note_vad_break with the break itself, so it is only
                 ever on the turn whose t0 IS that break. EOS -> first answer
                 audio is ``tail_ms + cap_lag_ms + first_play``.
  clip_ms        length of the clip handed to STT (pre-roll included).
  stt_wait_ms    how long transcribe() waited for _stt_lock (an ambient
                 decode holding Whisper), the turn's own capture only.
  stt_engine     R6 (STT_ENGINE='parakeet' only): which engine's transcript
                 the turn used — parakeet, parakeet-rescued (Parakeet's text
                 was empty or lost the wake word; Whisper decoded again),
                 parakeet-gated (that rescue was skipped over music:
                 MUSIC_GATE_MODE 'on', core/music_gate.py),
                 whisper-fallback (Parakeet failed and latched off on this
                 capture), whisper-loading (the model was still loading; no
                 latch) or whisper (latched off earlier). ``-`` on the
                 default Whisper path, shadow mode included.
  load_ms        Ollama load_duration of the answering local-LLM response.
  total_ms       Ollama total_duration of the same response.
  play_open_ms   the answer's first playback, from entering the playback body
                 (before the PortAudio claim, the output-device refresh, the
                 barge-in listener and the music duck) to sd.play() returning
                 with the stream started.
  out_lat_ms     that stream's output latency as PortAudio reported it at
                 open — the rest of the way to the first audible sample. EOS
                 -> audible is EOS -> first_play + play_open_ms + out_lat_ms.
  filler_clip_ms length of the first processing-filler clip played.
  eot st_p st_n  reserved (R7): end-of-turn verdict, Smart Turn p, checks.
  pre            reserved (R3): 1 when the answer was pre-rendered.
  cut            reserved (R8): ms of filler played before a soft cut.
  amb_deferred   reserved (R2): ambient decodes deferred during the turn.
  cache          reserved (R4): Kokoro render-cache verdict.
  clone          the clone voice server (VOICE_CLONE_MODEL
                 'chatterbox_turbo_server') on the answer's first render: 1 =
                 it voiced it, 0 = it was tried and that line fell back to
                 Kokoro. ``-`` = the clone was not in use.
  clone_ms       that first clone line's wall time, request to finished
                 audio; for a cached line, the time to confirm the server
                 is up and fetch the take (0-2 ms; clone_cache says it was
                 cached).
  t3_ms_tok      C8 (2026-10-05): that first clone line's T3 decode ms per
                 speech token, from the server's X-T3-Ms / X-Speech-Tokens
                 (~4.3 on the fast cuda-graph decoder, ~24 on the slow
                 loop). ``-`` when it came from the cache, missed, or the
                 server sent no headers -- never a later line's speed.
  clone_cache    where that first clone line came from: mem / disk (the
                 render cache), miss (the server rendered it, or failed),
                 refused (cached, but the server was not answering ready,
                 so Kokoro voiced it), shadow-hit (VOICE_CLONE_CACHE
                 'shadow': rendered, but the disk would have served it).
  audible_ms     computed when the line is printed, never noted: the
                 answer's first audible sample, ms from t0 = first_play +
                 play_open_ms + out_lat_ms (``-`` unless all three are
                 known). From the owner's last word: tail_ms + cap_lag_ms +
                 audible_ms.
  keeper         PLAYBACK_KEEPER (2026-10-05): 1 = the playback keeper's
                 silent stream was holding the speaker when the answer's
                 first playback opened, 0 = it was not (yet). ``-`` = the
                 keeper is off.
  opens          how many playbacks the answer opened (one per sentence /
                 clip), counted like play_open_ms (after "you", the turn's
                 thread or a helper it adopted).
  opens_ms       their play_open_ms values added up: the mean open of the
                 LATER sentences is (opens_ms - play_open_ms) / (opens - 1),
                 the part of each sentence gap the keeper is meant to cut.

They are set through TurnTiming.note_stat (load_ms / total_ms come with the
answering response through llm_response, cap_lag_ms with the VAD break through
note_vad_break), so later batches only fill a slot.

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

# Speed-plan R1 fields (see the module docstring). Inserted BEFORE
# lead_dropped, which stays last: tests/test_turn_timing.py pins that, and the
# log tools read every field by key (parse_line), never by position.
NOTE_FIELDS = ("tail_ms", "cap_lag_ms", "clip_ms", "stt_wait_ms",
               "stt_engine", "load_ms", "total_ms", "play_open_ms",
               "out_lat_ms", "filler_clip_ms", "eot", "st_p", "st_n", "pre",
               "cut", "amb_deferred", "cache", "clone", "clone_ms",
               "t3_ms_tok", "clone_cache", "audible_ms",
               "keeper", "opens", "opens_ms")
# Computed by format_line from the marks and notes, never noted.
_COMPUTED_NAMES = frozenset(("audible_ms",))

# Stats fields, printed after the marks in this order.
# turn_ctx_chars is the per-turn context actually SENT; budget_trimmed=1 when
# the local prompt budget trimmed the turn (2026-10-02: a trimmed turn's
# prompt_eval_count must not enter the chars-per-token calibration as if it
# were whole), '-' when no budget ran (a cloud turn).
# Brain-prefix fields (2026-10-04), printed LAST, after budget_trimmed - what
# makes the local brain's prompt cache measurable on every turn:
#   pe_new    estimated prompt tokens the answering local call really
#             evaluated (the rest of prompt_eval_count came from Ollama's
#             cache): see prefill_estimate;
#   pe_state  warm (under PE_WARM_FRACTION of the prompt was evaluated),
#             full (at least PE_FULL_FRACTION: a whole re-read), or partial;
#   reprime   the idle re-prime's verdict for this turn's prefix (hit /
#             evicted / stale), '-' when no prime was stamped.
BRAIN_FIELDS = ("pe_new", "pe_state", "reprime")
STAT_FIELDS = ("prompt_eval_count", "prompt_eval_ms", "eval_count", "eval_ms",
               "llm_calls", "turn_ctx_chars", "sys_chars", "followup_rounds",
               "filler", "filler_ms", *NOTE_FIELDS, "lead_dropped",
               "budget_trimmed", *BRAIN_FIELDS)

# Prefill cost of the brain (gemma4:26b-a4b on the 3090), fitted to Ollama's
# server.log 2026-10-01..10-04 (1,216 brain prompts of 8k+ tokens, median
# ms by evaluated tokens: 15 -> 160, 176 -> 222, 998 -> 426, 1,831 -> 623,
# 13,604 -> 3,449): ms ~ PREFILL_FIXED_MS + PREFILL_MS_PER_TOKEN * tokens.
PREFILL_FIXED_MS = 150.0
PREFILL_MS_PER_TOKEN = 0.24
PE_WARM_FRACTION = 0.25
PE_FULL_FRACTION = 0.80

# The R1 fields kept in a turn's notes (printed from there): all but
# load_ms / total_ms, which only the answering response supplies
# (llm_response).
_NOTE_PRINTED = frozenset(NOTE_FIELDS) - {"load_ms", "total_ms"}
# The names note_stat accepts: those, but cap_lag_ms, which only the VAD
# break supplies (note_vad_break) — one writer each — and the computed ones.
_NOTE_NAMES = _NOTE_PRINTED - {"cap_lag_ms"} - _COMPUTED_NAMES

# Thread rules for note_stat (everything else follows mark(owner_only=True)):
#   * any thread — the filler clip, a soft cut on the playback reaper, the
#     ambient worker's deferral count all happen off the turn's thread;
#   * like first_play — only after "you", from the turn's thread or a helper
#     it adopted, so a reminder or tray line played first is not the answer.
_ANY_THREAD_NAMES = frozenset(("filler_clip_ms", "cut", "amb_deferred"))
_AFTER_YOU_NAMES = frozenset(("play_open_ms", "out_lat_ms", "cache", "clone",
                              "clone_ms", "t3_ms_tok", "clone_cache",
                              "keeper", "opens", "opens_ms"))
# Counts that add up over the turn instead of keeping the first value.
_ADDITIVE_NAMES = frozenset(("amb_deferred", "opens", "opens_ms"))
# Any-thread names that may arrive before their turn begins and be adopted
# by it: an ambient decode deferred while the owner was still talking. A
# filler clip or a soft cut only ever happens inside a turn; with no turn
# they are dropped, never handed to the NEXT one.
_PRE_TURN_ANY_NAMES = frozenset(("amb_deferred",))

# Bounds on the pre-turn stash (see TurnTiming.note_stat): entries per key,
# and keys (threads + the shared any-thread key).
_STASH_MAX_ENTRIES = 32
_STASH_MAX_KEYS = 8
_ANY = "*"

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


def prefill_estimate(prompt_eval_count, prompt_eval_ms) -> tuple:
    """(pe_new, pe_state) for one local response: the prompt tokens it most
    likely evaluated (Ollama's prompt_eval_count is the WHOLE prompt even
    when the prefix came from the cache; the time says how much was really
    read), clamped to [0, count], and warm / partial / full (see
    BRAIN_FIELDS). (None, None) when either number is missing. An estimate:
    a fast prompt that was mostly re-read reads warm, never the reverse by
    more than the fit's scatter. Never raises."""
    try:
        if (prompt_eval_count is None or prompt_eval_ms is None
                or isinstance(prompt_eval_count, bool)
                or isinstance(prompt_eval_ms, bool)):
            return (None, None)
        count = int(prompt_eval_count)
        ms = float(prompt_eval_ms)
        if count <= 0 or ms < 0:
            return (None, None)
        new = int(round((ms - PREFILL_FIXED_MS) / PREFILL_MS_PER_TOKEN))
        new = max(0, min(count, new))
        frac = new / count
        if frac < PE_WARM_FRACTION:
            state = "warm"
        elif frac >= PE_FULL_FRACTION:
            state = "full"
        else:
            state = "partial"
        return (new, state)
    except Exception:
        return (None, None)


def _new_turn(kind: str, t0: float, owner: int) -> dict:
    return {"kind": kind, "t0": t0, "owner": owner, "marks": {},
            "stats": {}, "llm_calls": 0, "followup_rounds": 0,
            "filler": 0, "filler_at": None, "lead_dropped": 0,
            "helpers": [], "notes": {}}


def _put_note(notes: dict, name: str, value) -> None:
    """Apply one note_stat value: the first value wins, except the additive
    counts, which add up (a non-numeric addend is ignored)."""
    if name in _ADDITIVE_NAMES:
        try:
            add = int(value)
        except Exception:
            return
        cur = notes.get(name)
        notes[name] = add if not isinstance(cur, int) else cur + add
        return
    if name not in notes:
        notes[name] = value


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
        # (clock, cap_lag_ms or None) of the last VAD break: ONE assignment,
        # so a reader never pairs one break's time with another's lag.
        self._last_vad = None
        # note_stat values that arrived while NO turn was active, keyed by the
        # writer's thread ident (or _ANY): [(clock, name, value), ...]. A
        # standby wake transcribes BEFORE its turn begins; begin_voice adopts
        # what its own capture recorded. Bounded; see _stash_put.
        self._stash = {}

    # ── turn lifecycle ────────────────────────────────────────────────────
    def now(self) -> "float | None":
        try:
            return self._clock()
        except Exception:
            return None

    def note_vad_break(self, lag_ms=None) -> None:
        """record_speech's end-of-utterance break. Only remembered; the turn
        that consumes it begins in the capture path (begin_voice).

        `lag_ms`: how far this break trails the clip's last sample (the
        cap_lag_ms field; record_speech's _capture_lag_ms). Kept WITH the
        break, so only the turn that starts at this break prints it; anything
        that is not a finite whole number of ms is dropped."""
        try:
            lag = None
            if lag_ms is not None and not isinstance(lag_ms, bool):
                try:
                    lag = int(lag_ms)
                except Exception:
                    lag = None
            self._last_vad = (self._clock(), lag)
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
        record_speech call); otherwise at now. vad_break is offset 0, and
        the break's cap_lag_ms comes with it."""
        try:
            last = self._last_vad
            vad, lag = last if last is not None else (None, None)
            use_vad = vad is not None and (since is None or vad >= since)
            self.begin("voice", t0=vad if use_vad else None)
            with self._lock:
                t = self._turn
                if t is not None and use_vad:
                    t["marks"]["vad_break"] = vad
                    if lag is not None:
                        t["notes"]["cap_lag_ms"] = lag
                # Adopt the stats this capture recorded before the turn
                # existed: the caller's own pre-turn notes plus the any-thread
                # ones, stamped at or after `since` (older ones belong to an
                # earlier capture). No `since`, nothing is adopted.
                if t is not None:
                    ident = threading.get_ident()
                    for key in (ident, _ANY):
                        for ts, name, value in self._stash.pop(key, ()):
                            if since is not None and ts >= since:
                                _put_note(t["notes"], name, value)
        except Exception:
            pass

    def discard(self) -> None:
        try:
            with self._lock:
                self._turn = None
        except Exception:
            pass

    def reset(self) -> None:
        """discard() plus the pre-turn stash and the remembered VAD break:
        nothing left from before (the test harness's per-test reset)."""
        try:
            with self._lock:
                self._turn = None
                self._stash = {}
                self._last_vad = None
        except Exception:
            pass

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
                              "eval_count", "eval_ms", "load_ms", "total_ms"):
                        t["stats"][k] = s.get(k)
        except Exception:
            pass

    def note_stat(self, name: str, value) -> None:
        """Record one speed-plan field (NOTE_FIELDS) for the active turn.

        Refuses any other name (and cap_lag_ms: note_vad_break's; and
        audible_ms, which format_line computes). The first value wins
        (amb_deferred, opens and opens_ms add up). Thread rules: the turn's
        own thread only, like mark(owner_only=True) — except
        filler_clip_ms / cut / amb_deferred (any thread) and play_open_ms /
        out_lat_ms / cache / clone / clone_ms / t3_ms_tok / clone_cache /
        keeper / opens / opens_ms (like first_play: after "you", from the
        turn's thread or an adopted helper).

        With NO active turn, a value is kept for the turn that is about to
        begin: a standby wake transcribes its capture before begin_voice, and
        record_speech runs before every voice turn. begin_voice(since) adopts
        the caller's own values (amb_deferred: anyone's) stamped at or after
        `since`; everything else ages out of a small bounded stash.
        play_open_ms / out_lat_ms / cache / clone / clone_ms / filler_clip_ms
        / cut need a live turn.

        `value` may be a zero-argument callable (a result still being worked
        out on a daemon, e.g. tail_ms): it is called once when the line is
        printed, on the emitting thread, and prints ``-`` if it returns None
        or raises — never late, never blocking. Never raises."""
        try:
            if name not in _NOTE_NAMES:
                return
            ident = threading.get_ident()
            cur = threading.current_thread()
            with self._lock:
                t = self._turn
                if t is None:
                    if name in _AFTER_YOU_NAMES:
                        return   # needs "you"; there is no turn yet
                    if (name in _ANY_THREAD_NAMES
                            and name not in _PRE_TURN_ANY_NAMES):
                        return   # only ever inside a turn
                    # Only the stash needs a time (begin_voice's `since`):
                    # the clock is read on this path alone, so a value noted
                    # into a live turn costs no clock read at all.
                    self._stash_put(_ANY if name in _ANY_THREAD_NAMES
                                    else ident, self._clock(), name, value)
                    return
                if name in _AFTER_YOU_NAMES:
                    if "you" not in t["marks"]:
                        return
                    if ident != t["owner"] and not any(
                            h is cur for h in t["helpers"]):
                        return
                elif name not in _ANY_THREAD_NAMES and ident != t["owner"]:
                    return
                _put_note(t["notes"], name, value)
        except Exception:
            pass

    def _stash_put(self, key, ts, name, value) -> None:
        """Caller holds self._lock. Newest kept; oldest entry / key dropped."""
        entries = self._stash.get(key)
        if entries is None:
            if len(self._stash) >= _STASH_MAX_KEYS:
                oldest = min(self._stash,
                             key=lambda k: (self._stash[k][-1][0]
                                            if self._stash[k] else 0.0))
                self._stash.pop(oldest, None)
            entries = self._stash[key] = []
        entries.append((ts, name, value))
        if len(entries) > _STASH_MAX_ENTRIES:
            del entries[:len(entries) - _STASH_MAX_ENTRIES]

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


def _note_value(v) -> str:
    """One note_stat value as a single key=value token. A callable is
    resolved here (once, never raising); None prints ``-``; whitespace can
    never split the token."""
    try:
        if callable(v):
            v = v()
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, float):
            v = round(v, 3)
        s = "_".join(str(v).split())
        return s or "-"
    except Exception:
        return "-"


def _audible_ms(t0, marks: dict, notes: dict) -> str:
    """first_play + play_open_ms + out_lat_ms as ms from t0 (the answer's
    first audible sample), or '-' unless all three are known whole numbers.
    Never raises."""
    try:
        fp = marks.get("first_play")
        po = notes.get("play_open_ms")
        ol = notes.get("out_lat_ms")
        if fp is None or callable(po) or callable(ol):
            return "-"
        if any(v is None or isinstance(v, bool) for v in (po, ol)):
            return "-"
        total = (fp - t0) * 1000.0 + float(po) + float(ol)
        return str(int(round(total)))
    except Exception:
        return "-"


def format_line(turn: dict, end, outcome: str = "ok") -> str:
    """Render one finished turn (see the module docstring for the shape)."""
    t0 = turn["t0"]
    marks = turn["marks"]
    stats = turn["stats"]
    notes = turn.get("notes") or {}
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
        "load_ms": stats.get("load_ms"),
        "total_ms": stats.get("total_ms"),
        "lead_dropped": turn.get("lead_dropped", 0),
        "budget_trimmed": stats.get("budget_trimmed"),
        "reprime": stats.get("reprime"),
    }
    vals["pe_new"], vals["pe_state"] = prefill_estimate(
        stats.get("prompt_eval_count"), stats.get("prompt_eval_ms"))
    for k in STAT_FIELDS:
        if k == "filler_ms":
            parts.append(f"filler_ms={_off(t0, turn['filler_at'])}")
        elif k == "audible_ms":
            parts.append(f"audible_ms={_audible_ms(t0, marks, notes)}")
        elif k in _NOTE_PRINTED:
            parts.append(f"{k}={_note_value(notes.get(k))}")
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
