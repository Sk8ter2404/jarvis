"""core/dialogue.py — a short scripted back-and-forth between JARVIS and a
talking device in the room, run turn by turn.

WHY THIS MODULE EXISTS
======================
A skill that owns a talking device can stage a little exchange: JARVIS says a
line, the device answers, JARVIS replies, and so on, then the main loop takes
over again. Everything that makes that safe is generic and lives here, so a
private skill supplies only its own words and its device transport:

  * ``validate_script`` turns a model's JSON script into checked ``Line``s:
    tags stripped, rule-breaking lines cut (with everything after them), the
    device's lines chunked to its limit, and the result ALWAYS ends on a
    JARVIS line flagged ``final`` (so the device is never the last voice the
    main loop hears);
  * ``Runner`` plays the lines strictly one at a time. After every device
    line it runs the caller's synchronous stop-listen capture (the only time
    a microphone is open), waits until the device has FINISHED talking, and
    only then lets JARVIS speak. It never returns while the device is still
    talking, and it checks the session's stop state before every line;
  * ``is_stop_utterance`` / ``conflicts`` / ``STOP_PHRASES`` are the shared
    vocabulary checks for "the owner wants this to end" and for lines that
    must never be spoken.

The monolith provides the session, the speech and the capture (skill_utils
"dialogue_session", "speak_line", "listen_for_stop"); this module never
touches audio, the network or disk. It uses the device speech filter only for
its normaliser and stop-word list.

Vocabulary is deliberately generic: "self" is JARVIS, "device" is the other
speaker. Pure stdlib (dataclasses, json, re, threading, time).
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional

from core import device_speech_filter as _dsf

# Owner phrases that end a dialogue on top of the filter's stop words (stop,
# halt, freeze, abort, emergency, estop, cancel). Matched as whole word runs
# of the normalised transcript.
STOP_PHRASES: tuple = (
    "that's enough", "enough", "quiet", "be quiet", "shut up",
    "knock it off", "never mind", "okay okay", "that'll do", "that will do")

# Why a dialogue ended (Outcome.reason). Single words, none of them a
# FAILURE_MARKERS substring, so a skill can put one in its result string.
# "sleep": JARVIS was sent to sleep mid-dialogue (the session's stopped()).
REASONS: tuple = (
    "done", "owner_stop", "wake", "interrupted", "device_lost", "device_busy",
    "device_muted", "device_heap", "device_no_answer", "tts_muted",
    "mic_muted", "sleep", "expired", "error")

# device_say() result -> the reason the dialogue ends with.
_DEVICE_REASON = {
    "busy": "device_busy", "muted": "device_muted",
    "no_answer": "device_no_answer", "heap": "device_heap",
    "refused": "error", "unsupported": "error",
}
# device_say() results that end the dialogue as "error" (Outcome.reason keeps
# the contract's REASONS) but look up a FINER closing first, in this order,
# before the "error" one: without them a refused say ended silently after the
# opener. The caller supplies the wording under these keys in ``closings``;
# none supplied = silence, as before.
DEVICE_CLOSING_KEYS = {
    "refused": ("device_refused",),
    "unsupported": ("device_unsupported", "device_refused"),
}
# speak_self() result -> the reason (when the session has none of its own).
_SPEAK_REASON = {
    "interrupted": "interrupted", "muted": "tts_muted", "failed": "error",
    "staging": "error",
}

_LEAD_TAG_RE = re.compile(
    r"^\s*\[(?:wry|mood\s*:\s*[\w-]+|intent\s*:\s*[\w-]+)\]\s*", re.I)
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_FENCE_RE = re.compile(r"```(?:json)?", re.I)


class DialogueUnavailable(RuntimeError):
    """A device dialogue cannot start: raised on entering a dialogue session.
    ``.reason`` is one word - the monolith's _dialogue_ready() reasons
    (disabled, staging, boot_grace, tts_muted, mic_muted, sleep,
    realtime_voice, active, error). ONE class for every raiser: the
    monolith's ``_dialogue_session`` (``bobert_companion.DialogueUnavailable``
    is this class) and an unwired ``JarvisServices.dialogue_session``
    ("disabled"), so a skill can catch it, or read ``getattr(exc, "reason")``,
    the same way whichever path it took. A RuntimeError, so older
    ``except RuntimeError`` callers keep working."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason)


@dataclass(frozen=True)
class Line:
    """One line of a script. ``who`` is "self" (JARVIS) or "device".
    ``chunks`` are the device's say() pieces (one, unless a long line was
    salvaged into two). ``final`` marks JARVIS's last line (the punchline)."""
    who: str
    text: str
    chunks: tuple = ()
    final: bool = False


@dataclass(frozen=True)
class Outcome:
    """How a run ended: lines actually voiced (the opener and every script
    line; not a stall or closing line) and one of REASONS."""
    lines_spoken: int
    reason: str


class ListenCapture:
    """The result of one synchronous stop-listen capture.

    ``available`` is False when no capture ran (mic busy, muted, disabled).
    ``beat_voiced`` is True when the short beat after the device finished
    held voice energy (the owner may be talking). ``result(timeout)`` returns
    ("pending"|"stop"|"wake"|"speech"|"", text) — text only for "speech";
    "pending" while the transcription is still running."""

    __slots__ = ("available", "beat_voiced", "_evt", "_res")

    def __init__(self, available: bool = True, beat_voiced: bool = False):
        self.available = bool(available)
        self.beat_voiced = bool(beat_voiced)
        self._evt = threading.Event()
        self._res = ("", "")
        if not self.available:
            self._evt.set()

    @classmethod
    def unavailable(cls) -> "ListenCapture":
        return cls(available=False)

    def set_result(self, kind: str, text: str = "") -> None:
        """Publish the verdict (the capture's worker thread calls this)."""
        kind = kind if kind in ("stop", "wake", "speech") else ""
        self._res = (kind, text if kind == "speech" else "")
        self._evt.set()

    def result(self, timeout: float = 0.0) -> tuple:
        try:
            t = max(0.0, float(timeout))
        except Exception:
            t = 0.0
        if not self._evt.wait(t):
            return ("pending", "")
        return self._res


# ── vocabulary checks ─────────────────────────────────────────────────────
def _words(text) -> list:
    return _dsf.normalise(text).split()


def _run_at(words: list, seq: list, max_gap: int) -> bool:
    """True when the word sequence ``seq`` occurs in ``words`` in order with at
    most ``max_gap`` other words between neighbours. A pattern word ending
    in "*" matches by prefix."""
    if not seq:
        return False

    def eq(w, p):
        return w.startswith(p[:-1]) if p.endswith("*") else w == p

    n = len(words)
    for i in range(n):
        if not eq(words[i], seq[0]):
            continue
        pos, ok = i, True
        for p in seq[1:]:
            found = -1
            for j in range(pos + 1, min(n, pos + 2 + max_gap)):
                if eq(words[j], p):
                    found = j
                    break
            if found < 0:
                ok = False
                break
            pos = found
        if ok:
            return True
    return False


def _norm_phrases(phrases) -> list:
    """Normalised word lists of ``phrases``; a trailing "*" survives on the
    last word (prefix match)."""
    out = []
    for p in phrases or ():
        p = str(p)
        star = p.endswith("*")
        seq = _dsf.normalise(p[:-1] if star else p).split()
        if star and seq:
            seq[-1] = seq[-1] + "*"
        if seq:
            out.append(seq)
    return out


def is_stop_utterance(text, wake_phrases=()) -> str:
    """"stop" when ``text`` holds a stop word or one of STOP_PHRASES, "wake"
    when it holds a wake phrase, else "". Stop wins over wake. Never
    raises."""
    try:
        words = _words(text)
        if not words:
            return ""
        if any(w in _dsf.STOP_WORDS for w in words):
            return "stop"
        for seq in _norm_phrases(STOP_PHRASES):
            if _run_at(words, seq, 0):
                return "stop"
        for seq in _norm_phrases(wake_phrases):
            if _run_at(words, seq, 0):
                return "wake"
        return ""
    except Exception:
        return ""


def conflicts(text, *, words: Iterable = (), phrases: Iterable = (),
              near: Iterable = (), max_gap: int = 1) -> Optional[str]:
    """The first rule ``text`` breaks, or None.

    * ``words``: single words ("dance*" matches by prefix: dance, dancing);
    * ``phrases``: word runs, allowing up to ``max_gap`` extra words between
      neighbours ("turn left" also catches "turn hard left");
    * ``near``: (first_words, second_words, distance) triples: a word of the
      first set within ``distance`` words of one of the second, either order.
    Returns "word:<w>", "phrase:<p>" or "near:<a>~<b>". Never raises (an
    internal error reports "error", so a broken check fails CLOSED)."""
    try:
        ws = _words(text)
        if not ws:
            return None
        for w in words or ():
            seq = _norm_phrases([w])
            if seq and _run_at(ws, seq[0], 0):
                return f"word:{w}"
        for p in phrases or ():
            seq = _norm_phrases([p])
            if seq and _run_at(ws, seq[0], max(0, int(max_gap))):
                return f"phrase:{p}"
        for item in near or ():
            a_set, b_set, dist = item
            a_seqs = [s[0] for s in _norm_phrases(a_set)]
            b_seqs = [s[0] for s in _norm_phrases(b_set)]
            for i, w in enumerate(ws):
                if not any(_run_at([w], [a], 0) for a in a_seqs):
                    continue
                lo, hi = max(0, i - int(dist)), min(len(ws), i + int(dist) + 1)
                for j in range(lo, hi):
                    if j != i and any(_run_at([ws[j]], [b], 0) for b in b_seqs):
                        return f"near:{w}~{ws[j]}"
        return None
    except Exception:
        return "error"


def strip_speech_tags(text) -> str:
    """``text`` without its leading [wry] / [mood:x] / [intent:x] tags (any
    order, any count). Never raises."""
    if not isinstance(text, str):
        return ""
    s = text
    while True:
        m = _LEAD_TAG_RE.match(s)
        if not m:
            break
        s = s[m.end():]
    return s.strip()


def strip_lead_words(text, lead_words: Iterable, min_words: int = 3) -> str:
    """Drop ONE leading filler word (e.g. "Indeed,") and re-capitalise, when
    at least ``min_words`` words remain; otherwise ``text`` unchanged.
    A helper for a skill's clean_self. Never raises."""
    try:
        s = (text or "").strip()
        m = re.match(r"^([A-Za-z']+)[\s,.!;:-]+(.*)$", s)
        if not m:
            return s
        lead = m.group(1).lower()
        if lead not in {w.lower() for w in lead_words}:
            return s
        rest = m.group(2).strip()
        if len(rest.split()) < max(1, int(min_words)):
            return s
        return rest[:1].upper() + rest[1:]
    except Exception:
        return text if isinstance(text, str) else ""


def chunk_device_text(text, limit: int, max_chunks: int = 2) -> list:
    """Split ``text`` into at most ``max_chunks`` pieces of at most ``limit``
    characters, at sentence ends first, then at word boundaries. [] when it
    cannot be done (the caller drops the line). Never raises."""
    try:
        s = _SPACE_RE.sub(" ", str(text or "")).strip()
        limit = int(limit)
        if not s or limit <= 0:
            return []
        if len(s) <= limit:
            return [s]
        pieces: list = []
        for sent in _SENTENCE_SPLIT_RE.split(s):
            if len(sent) <= limit:
                pieces.append(sent)
                continue
            cur = ""
            for w in sent.split(" "):
                if len(w) > limit:
                    return []
                cand = f"{cur} {w}".strip()
                if len(cand) <= limit:
                    cur = cand
                else:
                    pieces.append(cur)
                    cur = w
            if cur:
                pieces.append(cur)
        chunks: list = []
        for p in pieces:
            if chunks and len(chunks[-1]) + 1 + len(p) <= limit:
                chunks[-1] = f"{chunks[-1]} {p}"
            else:
                chunks.append(p)
        return chunks if len(chunks) <= max(1, int(max_chunks)) else []
    except Exception:
        return []


# ── script validation ─────────────────────────────────────────────────────
def _parse_script(raw) -> Optional[list]:
    """The list of raw line dicts in ``raw`` (a JSON string, possibly fenced
    or wrapped in prose, or an already-parsed dict/list), else None."""
    data = raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        s = _FENCE_RE.sub("", raw)
        starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
        if not starts:
            return None
        try:
            data, _ = json.JSONDecoder().raw_decode(s[min(starts):])
        except Exception:
            return None
    if isinstance(data, dict):
        data = data.get("lines")
    return data if isinstance(data, list) else None


def _is_ascii_printable(s: str) -> bool:
    return all(32 <= ord(c) < 127 for c in s)


def validate_script(raw, *, who_map: dict, min_lines: int, max_lines: int,
                    clean_self: Callable, clean_device: Callable,
                    check_self: Callable, check_device: Callable,
                    check_script: Optional[Callable] = None,
                    closer: Optional[Callable] = None,
                    device_limit: int = 90,
                    device_salvage_limit: int = 120) -> Optional[list]:
    """Checked ``Line``s from a model's script, or None when fewer than
    ``min_lines`` survive.

    ``raw``: {"lines": [{"who": ..., "text": ...}, ...]} or a bare list, as a
    JSON string (code fences / surrounding prose tolerated) or parsed.
    ``who_map``: model speaker name -> "self" | "device" (case-insensitive).
    ``clean_self`` / ``clean_device``: text -> cleaned text, or None to reject
    (the skill's caps and wording rules live there). ``check_self`` /
    ``check_device``: text -> the rule it breaks, or None.
    ``check_script``: the whole list -> a (possibly shorter) list.
    ``closer``: () -> a JARVIS line appended when the script would otherwise
    end on the device; without one, trailing device lines are dropped.

    Rules, in order: an unknown speaker, a non-string text, a rejected clean
    or a failed check drops THAT line and everything after it (the rest of a
    conversation no longer follows). JARVIS lines lose leading speech tags;
    "[action" / "[intent" anywhere rejects the line. Device lines must be
    printable ASCII of at most ``device_limit`` chars, or up to
    ``device_salvage_limit`` chars split into two chunks. Two neighbours from
    the same speaker are merged when the merge still cleans, else the second
    is dropped. The result is cut to ``max_lines``, always ends on a "self"
    line, and that line has ``final=True``. Never raises (None on error)."""
    try:
        items = _parse_script(raw)
        if not items:
            return None
        wm = {str(k).strip().lower(): v for k, v in (who_map or {}).items()}
        lines: list = []
        for item in items:
            if not isinstance(item, dict):
                break
            who = wm.get(str(item.get("who", "")).strip().lower())
            text = item.get("text")
            if who not in ("self", "device") or not isinstance(text, str):
                break
            text = _SPACE_RE.sub(" ", text).strip()
            line = _clean_line(who, text, clean_self, clean_device,
                               device_limit, device_salvage_limit)
            if line is None:
                break
            rule = (check_self if who == "self" else check_device)(line.text)
            if rule:
                break
            if lines and lines[-1].who == who:
                merged = _clean_line(who, f"{lines[-1].text} {line.text}",
                                     clean_self, clean_device, device_limit,
                                     device_salvage_limit)
                if merged is not None and not (
                        check_self if who == "self" else check_device
                )(merged.text):
                    lines[-1] = merged
                continue
            lines.append(line)
        if check_script is not None:
            lines = list(check_script(list(lines)) or [])
        lines = lines[:max(1, int(max_lines))]
        if len(lines) < max(1, int(min_lines)):
            return None
        if lines[-1].who != "self":
            text = closer() if closer is not None else None
            if isinstance(text, str) and text.strip():
                lines.append(Line("self", text.strip()))
            else:
                while lines and lines[-1].who != "self":
                    lines.pop()
        if not lines or lines[-1].who != "self":
            return None
        lines[-1] = replace(lines[-1], final=True)
        return lines
    except Exception:
        return None


def _clean_line(who, text, clean_self, clean_device, device_limit,
                device_salvage_limit) -> Optional[Line]:
    if who == "self":
        t = strip_speech_tags(text)
        low = t.lower()
        if "[action" in low or "[intent" in low or not t:
            return None
        t = clean_self(t)
        if not isinstance(t, str) or not t.strip():
            return None
        return Line("self", t.strip())
    t = clean_device(text)
    if not isinstance(t, str):
        return None
    t = _SPACE_RE.sub(" ", t).strip()
    if not t or not _is_ascii_printable(t):
        return None
    if len(t) <= device_limit:
        return Line("device", t, (t,))
    if len(t) > device_salvage_limit:
        return None
    chunks = chunk_device_text(t, device_limit, 2)
    if not chunks:
        return None
    return Line("device", t, tuple(chunks))


# ── the runner ────────────────────────────────────────────────────────────
class Runner:
    """Plays one dialogue: the opener, then the script, strictly in turn.

    Callbacks (all synchronous, on the caller's thread):
      speak_self(text, final) -> "spoken"|"interrupted"|"muted"|"failed"|
                                 "staging"   (JARVIS speaks; may block; a
                                 session.stop() from ANY thread cuts it)
      device_say(chunk) -> "ok"|"busy"|"muted"|"no_answer"|"heap"|"refused"|
                           "unsupported"     (queue one device line)
      device_done() -> bool   non-blocking: has the device finished the
                              chunk it was last told to say? (Answer for its
                              speaker as a whole when it can: a "busy" say
                              usually means it is talking.)
      listen(until, *, beat_s, max_s) -> ListenCapture   the synchronous
                              stop-listen: capture until until() is true,
                              then beat_s more (or max_s), mic closed on
                              return (skill_utils["listen_for_stop"] fits)
      session: .stopped() -> reason | None ; .stop(reason) -> bool

    Guarantees: listen() runs after every successful device_say and before
    the next speak_self; the device has finished (device_done() true, or
    ``device_wait_s`` elapsed) before JARVIS speaks again and before run()
    returns - a closing after a failed say (a "busy" give-up, a stop during
    the retries) included; the session's stop state is checked before every
    line; a device chunk that directly follows another device chunk waits
    (at most ``verdict_wait_s``) for the previous stop-listen verdict first.
    """

    def __init__(self, *, speak_self, device_say, device_done, listen,
                 session, beat_s: float = 0.6, busy_retry_s: float = 2.0,
                 busy_step_s: float = 0.3, clock=time.monotonic,
                 sleep=time.sleep, voiced_wait_s: float = 1.5,
                 device_wait_s: float = 12.0, done_poll_s: float = 0.05,
                 verdict_wait_s: float = 3.0):
        self.speak_self = speak_self
        self.device_say = device_say
        self.device_done = device_done
        self.listen = listen
        self.session = session
        self.beat_s = max(0.0, float(beat_s))
        self.busy_retry_s = max(0.0, float(busy_retry_s))
        self.busy_step_s = max(0.01, float(busy_step_s))
        self.clock = clock
        self.sleep = sleep
        self.voiced_wait_s = max(0.0, float(voiced_wait_s))
        self.device_wait_s = max(0.0, float(device_wait_s))
        self.done_poll_s = max(0.005, float(done_poll_s))
        # How long a device line waits for the PREVIOUS line's stop-listen
        # verdict (see _device_line). Longer than voiced_wait_s: the whole
        # capture, not just the beat, may still be transcribing.
        self.verdict_wait_s = max(0.0, float(verdict_wait_s))
        # The finer closing keys the last failed device say named (see
        # DEVICE_CLOSING_KEYS); reset by every run().
        self._closing_keys: tuple = ()
        # The last device line's stop-listen capture; reset by every run().
        self._last_cap = None

    # -- helpers -------------------------------------------------------
    def _stopped(self) -> Optional[str]:
        try:
            r = self.session.stopped()
        except Exception:
            return "error"
        return str(r) if r else None

    def _speak(self, text: str, final: bool) -> str:
        """"" when spoken, else the reason to end with."""
        try:
            res = self.speak_self(text, final)
        except Exception:
            res = "failed"
        if res == "spoken":
            return ""
        return self._stopped() or _SPEAK_REASON.get(res, "error")

    def _done(self) -> bool:
        try:
            return bool(self.device_done())
        except Exception:
            return True

    def _wait_device_done(self) -> None:
        """Bounded wait for the device to finish its current chunk. Runs even
        after a stop: the device cannot be cut, so JARVIS must not talk over
        it."""
        deadline = self.clock() + self.device_wait_s
        while not self._done() and self.clock() < deadline:
            self.sleep(self.done_poll_s)

    def _say_failed(self, reason: str, closing_keys: tuple = ()) -> str:
        """A chunk was not said: remember the finer closing keys, then hold
        (bounded) until the device has finished. A "busy" answer means it
        was talking a moment ago, and the closing that follows must not
        talk over it. Returns ``reason`` - or the session's own stop reason
        when one landed during that wait (a wake / tray stop then ends the
        run with ITS closing, never the failed say's)."""
        self._closing_keys = tuple(closing_keys)
        self._wait_device_done()
        r = self._stopped()
        if r and r != reason:
            self._closing_keys = ()
            return r
        return reason

    def _device_line(self, line: Line) -> str:
        """Say every chunk of ``line``; "" when all were said, else the
        reason to end with. Always returns with the device finished (or
        ``device_wait_s`` spent waiting for it) - a failed say included."""
        for chunk in (line.chunks or (line.text,)):
            # A device line cannot be cut once said (2026-10-01): a "stop"
            # the owner said over the PREVIOUS device line is transcribed on
            # a worker after its listen returned, so with no JARVIS line in
            # between (chunk 2 of a long line, the first scripted line after
            # the stall line) its verdict was still pending here and the
            # device said a whole line after he said stop. Let that verdict
            # land (bounded) before saying anything more. A verdict that has
            # landed returns at once; a JARVIS line in between clears it
            # (run()).
            prev, self._last_cap = self._last_cap, None
            if prev is not None and getattr(prev, "available", False):
                try:
                    prev.result(self.verdict_wait_s)
                except Exception:
                    pass
            r = self._stopped()
            if r:
                return r
            give_up = self.clock() + self.busy_retry_s
            while True:
                try:
                    res = self.device_say(chunk)
                except Exception:
                    res = None          # a skill bug: plain "error"
                if res == "ok":
                    break
                if res == "busy" and self.clock() + self.busy_step_s <= give_up:
                    self.sleep(self.busy_step_s)
                    r = self._stopped()
                    if r:
                        return self._say_failed(r)
                    continue
                return self._say_failed(_DEVICE_REASON.get(res, "error"),
                                        DEVICE_CLOSING_KEYS.get(res, ()))
            cap = None
            try:
                # Keywords, so skill_utils["listen_for_stop"] (keyword-only
                # beat_s / max_s) can be passed in directly.
                cap = self.listen(self._done, beat_s=self.beat_s,
                                  max_s=self.device_wait_s)
            except Exception:
                cap = None
            self._last_cap = cap
            available = bool(getattr(cap, "available", False))
            self._wait_device_done()
            if not available:
                self.sleep(self.beat_s)
            elif getattr(cap, "beat_voiced", False):
                # The owner may be talking over the beat: wait (bounded) for
                # the verdict so JARVIS never speaks over him.
                try:
                    cap.result(self.voiced_wait_s)
                except Exception:
                    pass
            r = self._stopped()
            if r:
                return r
        return ""

    def _closing(self, closings, reason) -> None:
        """Speak the caller's closing for ``reason``: a finer key a failed
        say named (DEVICE_CLOSING_KEYS) first, then the reason's own; none
        (missing / None / blank) = silence."""
        keys = tuple(getattr(self, "_closing_keys", ())) + (reason,)
        for key in keys:
            text = (closings or {}).get(key)
            if isinstance(text, str) and text.strip():
                self._speak(text.strip(), True)
                return

    # -- the run -------------------------------------------------------
    def run(self, opener: str, script, fallback, *, script_deadline: float,
            stall=None, closings: Optional[dict] = None,
            preflight=None) -> Outcome:
        """Play ``opener`` and then the lines of ``script`` (a Future of a
        list of Line, or of None), or ``fallback()``'s lines when the script
        is not ready by ``script_deadline`` (a clock() value) or is empty.
        ``stall()`` -> a device line said while a late script is awaited.
        ``preflight``: a Future of "" (the device is ready) or a reason; a
        reason ends the run after the opener with that reason's closing.
        ``closings``: reason -> a JARVIS line spoken when the run ends that
        way (missing / None = silence); a refused / unsupported device say
        looks up its DEVICE_CLOSING_KEYS first (e.g. "device_refused"), then
        "error". Never raises."""
        spoken = 0
        self._closing_keys = ()
        self._last_cap = None
        try:
            r = self._stopped()
            if r:
                return Outcome(0, r)
            r = self._speak(opener, False)
            if r:
                return Outcome(0, r)
            spoken += 1
            if preflight is not None:
                try:
                    pre = preflight.result(
                        timeout=max(0.0, script_deadline - self.clock()))
                except Exception:
                    pre = "device_no_answer"
                if pre:
                    pre = str(pre)
                    r = self._stopped()
                    if r:
                        return Outcome(spoken, r)
                    self._closing(closings, pre)
                    return Outcome(spoken, pre)
            lines = None
            try:
                late = not script.done()
            except Exception:
                late = False
            if late and stall is not None:
                try:
                    st = stall()
                except Exception:
                    st = None
                if isinstance(st, str) and st.strip():
                    r = self._device_line(Line("device", st.strip(),
                                               (st.strip(),)))
                    if r:
                        self._closing(closings, r)
                        return Outcome(spoken, r)
            try:
                lines = script.result(
                    timeout=max(0.0, script_deadline - self.clock()))
            except Exception:
                lines = None
            if not lines:
                try:
                    lines = fallback()
                except Exception:
                    lines = None
            if not lines:
                return Outcome(spoken, "error")
            reason = "done"
            for line in lines:
                r = self._stopped()
                if r:
                    reason = r
                    break
                if line.who == "self":
                    # JARVIS's own line comes between: a late verdict cuts
                    # HIM (the caller's speak_self), so the next device line
                    # need not wait on it (see _device_line).
                    self._last_cap = None
                    r = self._speak(line.text, bool(line.final))
                else:
                    r = self._device_line(line)
                if r:
                    reason = r
                    break
                spoken += 1
            if reason != "done":
                self._closing(closings, reason)
            return Outcome(spoken, reason)
        except Exception:
            try:
                self._wait_device_done()
            except Exception:
                pass
            return Outcome(spoken, "error")
