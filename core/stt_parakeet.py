"""core/stt_parakeet.py — Parakeet TDT on the CPU for the owner's captures
(speed plan R6, 2026-10-02).

WHY THIS EXISTS
---------------
Whisper (large-v3-turbo) decodes an owner capture in ~1.66 s p50 on this box.
NVIDIA's Parakeet TDT 0.6B v2, int8 ONNX on the CPU through onnx-asr, decoded
the same kind of clip in ~0.16 s against Whisper's 1.60 s in the model
research, with a lower word error rate (2.6 % vs 3.4 % on clean speech, 4.1 %
vs 4.9 % with fan noise). Two flags in core/config.py, both OFF by default:

  STT_ENGINE  'whisper' | 'parakeet'  (env JARVIS_STT_ENGINE wins when it
              names one of them). Which engine decodes the owner's captures —
              _transcribe_capture ONLY. Ambient and in-turn captures stay on
              Whisper, which stays loaded as the hot fallback.
  STT_SHADOW  '' | 'parakeet'. Whisper stays primary; Parakeet re-decodes
              each owner capture later, once nothing is happening, and both
              transcripts are judged by the real gates into
              data/stt_ab.jsonl. The words go there only — never the log.

WHAT PARAKEET LOSES, AND THE RESCUE
-----------------------------------
  * Whisper's hotwords: STT_HOTWORDS is ignored (logged once).
    STT_REPLACEMENTS applies as for Whisper, then STT_REPLACEMENTS_PARAKEET
    maps Parakeet's own mishearings.
  * Wake-word mode refuses any mic turn whose transcript is not led by
    "JARVIS" (and standby / sleep wake on the wake word alone), so ONE
    misheard wake word loses the whole turn. The rescue bounds that: when
    Parakeet's text is empty — or, while only a wake word gets through, is
    not led by the wake word while the speech detector hears speech in the
    clip's first 0.8 s — Whisper decodes the clip once and its transcript is
    used. Every rescue is counted.
  * Any Parakeet failure latches it off for the session (one log line) and
    Whisper decodes that capture and every later one.

CONFIDENCE (map_conf)
---------------------
is_valid_speech / hallucination_verdict judge Whisper's confidence dict, so
Parakeet returns the same shape: ``no_speech_prob`` is 0.0 with text and 1.0
without (live Whisper reported 0.0 on every one of 10,674 logged entries);
``avg_logprob`` is the mean token log-probability mapped through
PARAKEET_CONF_ANCHORS (piecewise linear, monotone, never above 0) onto
Whisper's scale, and -10.0 with no tokens. Extra keys (engine, tok_lp_mean,
tok_lp_min, n_tok, speech_s, stt_ms) are for the A/B rows; every consumer of
the dict reads its keys with .get or by those two names only.

Calibration of the default anchors (2026-10-02, this CPU, int8): the token
mean was -0.0001 .. -0.048 on clean synthetic and LibriSpeech speech and
-0.046 on speech under noise at -6 dB SNR with one word wrong; silence, white
noise, a tone chord and reversed speech gave no tokens at all. So a mean of
-0.05 maps to Whisper's ordinary -0.3, and -0.6 (an average token
probability near 0.55) sits on the gate's -1.5.

CI SAFETY: stdlib only at import. numpy, onnxruntime and onnx_asr are
imported inside the functions that need them (onnx_asr behind find_spec), so
a box without them, and a process with both flags off, never imports them
(tests/test_stt_parakeet.py pins that in a fresh interpreter).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import queue
import threading
import time

ENGINES = ("whisper", "parakeet")
SHADOWS = ("", "parakeet")
ENV_ENGINE = "JARVIS_STT_ENGINE"
SAMPLE_RATE = 16000
MODEL_TYPE = "nemo-conformer-tdt"
QUANTIZATION = "int8"
DEFAULT_THREADS = 8
# (token-logprob mean, Whisper-scale avg_logprob), see CONFIDENCE above.
DEFAULT_CONF_ANCHORS = ((-1.5, -3.0), (-0.6, -1.5), (-0.3, -1.0),
                        (-0.05, -0.3), (0.0, -0.1))
EMPTY_AVG_LOGPROB = -10.0
# Shortest clip worth a decode; the capture callers never pass one this
# short (they drop anything under 0.4 s), so this only guards odd input.
MIN_SAMPLES = 160
RESCUE_HEAD_S = 0.8        # "speech in the clip's first 0.8 s"

# The shadow worker (STT_SHADOW). The queue holds at most SHADOW_QUEUE_MAX
# captures; a capture waits at most SHADOW_WAIT_S for the turn to finish.
SHADOW_QUEUE_MAX = 4
SHADOW_WAIT_S = 30.0
SHADOW_POLL_S = 0.25
AB_MAX_BYTES = 64 * 1024 * 1024


# ── settings ──────────────────────────────────────────────────────────────
def engine_setting(value=None) -> str:
    """'whisper' or 'parakeet': JARVIS_STT_ENGINE when it names one of them,
    else `value` (core.config.STT_ENGINE) when it does, else 'whisper'.
    Never raises."""
    try:
        for cand in (os.environ.get(ENV_ENGINE, ""), value):
            if isinstance(cand, str) and cand.strip().lower() in ENGINES:
                return cand.strip().lower()
    except Exception:
        pass
    return "whisper"


def shadow_setting(value=None) -> str:
    """'' or 'parakeet' (core.config.STT_SHADOW); anything else is ''."""
    try:
        v = value.strip().lower() if isinstance(value, str) else ""
    except Exception:
        return ""
    return v if v in SHADOWS else ""


def package_available() -> bool:
    """onnx_asr is importable (find_spec only: nothing is imported)."""
    try:
        return importlib.util.find_spec("onnx_asr") is not None
    except Exception:
        return False


# ── confidence ────────────────────────────────────────────────────────────
def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(float(v))


def anchors_or_default(anchors) -> tuple:
    """`anchors` as sorted ((x, y), ...) when they describe a usable map — two
    or more points, every coordinate finite and <= 0, x strictly increasing,
    y never decreasing — else DEFAULT_CONF_ANCHORS. Never raises."""
    try:
        pts = sorted((float(x), float(y)) for x, y in anchors)
        if len(pts) < 2:
            return DEFAULT_CONF_ANCHORS
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if not x1 > x0 or y1 < y0:
                return DEFAULT_CONF_ANCHORS
        if not all(_finite(c) and c <= 0.0 for p in pts for c in p):
            return DEFAULT_CONF_ANCHORS
        return tuple(pts)
    except Exception:
        return DEFAULT_CONF_ANCHORS


def _interp(x: float, pts) -> float:
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def map_conf(logprobs, anchors=None) -> dict:
    """Parakeet's per-token log-probabilities -> a Whisper-shaped confidence
    dict (see CONFIDENCE above). Pure; never raises.

    With no usable token: no_speech_prob 1.0, avg_logprob -10.0, n_tok 0.
    Otherwise no_speech_prob 0.0 and avg_logprob = the anchors' map of the
    token mean: monotone non-decreasing in the mean and never above 0.
    Non-finite entries are skipped; a positive one counts as 0."""
    lps = []
    try:
        for v in (logprobs or ()):
            if _finite(v):
                lps.append(min(0.0, float(v)))
    except Exception:
        lps = []
    if not lps:
        return {"no_speech_prob": 1.0, "avg_logprob": EMPTY_AVG_LOGPROB,
                "tok_lp_mean": None, "tok_lp_min": None, "n_tok": 0}
    mean = sum(lps) / len(lps)
    pts = anchors_or_default(anchors if anchors is not None
                             else DEFAULT_CONF_ANCHORS)
    return {"no_speech_prob": 0.0,
            "avg_logprob": min(0.0, float(_interp(mean, pts))),
            "tok_lp_mean": mean, "tok_lp_min": min(lps), "n_tok": len(lps)}


# ── the engine ────────────────────────────────────────────────────────────
def session_options(threads=DEFAULT_THREADS, rt=None):
    """onnxruntime SessionOptions exactly as core/kokoro_tts._tuned_session
    builds them: a bounded intra-op pool (`threads`, clamped to 1 ..
    cpu_count), one inter-op thread, sequential execution, and spin-wait OFF
    on both pools (a worker that finishes its slice sleeps instead of
    burning a core). `rt` is the onnxruntime module (tests pass a fake)."""
    if rt is None:
        import onnxruntime as rt
    try:
        n = int(threads)
    except Exception:
        n = DEFAULT_THREADS
    if n <= 0:
        n = DEFAULT_THREADS
    n = max(1, min(n, os.cpu_count() or n))
    so = rt.SessionOptions()
    so.intra_op_num_threads = n
    so.inter_op_num_threads = 1
    so.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return so


def load(model_dir, threads=DEFAULT_THREADS):
    """The Parakeet engine: onnx-asr's nemo-conformer-tdt model from
    `model_dir`, int8, CPU ONLY (CPUExecutionProvider; nothing is placed on
    a GPU), the tuned session options, with timestamps (so each result
    carries per-token log-probabilities). Raises on any failure — the
    package or the model directory missing, or the load itself; the caller
    latches the engine off."""
    if not package_available():
        raise RuntimeError("onnx_asr is not installed")
    if not model_dir or not os.path.isdir(str(model_dir)):
        raise FileNotFoundError(f"Parakeet model directory not found: "
                                f"{model_dir}")
    import onnx_asr
    model = onnx_asr.load_model(MODEL_TYPE, str(model_dir),
                                quantization=QUANTIZATION,
                                sess_options=session_options(threads),
                                providers=["CPUExecutionProvider"])
    return model.with_timestamps()


def transcribe(engine, audio, anchors=None, clock=time.perf_counter):
    """Decode one 16 kHz mono clip -> (text, conf). The audio is cast to
    contiguous float32 first. `text` is the transcript with its whitespace
    collapsed ('' when none); `conf` is map_conf() plus engine, speech_s and
    stt_ms (the decode's wall time). A clip under MIN_SAMPLES is not
    decoded (empty result). Raises what the engine raises."""
    import numpy as np
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2 and 1 in a.shape:
        a = a.reshape(-1)
    if a.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {a.shape}")
    a = np.ascontiguousarray(a, dtype=np.float32)
    speech_s = round(a.shape[0] / float(SAMPLE_RATE), 3)
    if a.shape[0] < MIN_SAMPLES:
        conf = map_conf((), anchors)
        conf.update(engine="parakeet", speech_s=speech_s, stt_ms=0)
        return "", conf
    t0 = clock()
    res = engine.recognize(a, sample_rate=SAMPLE_RATE)
    stt_ms = int(round((clock() - t0) * 1000.0))
    text = " ".join(str(getattr(res, "text", "") or "").split())
    conf = map_conf(getattr(res, "logprobs", None) if text else (), anchors)
    conf.update(engine="parakeet", speech_s=speech_s, stt_ms=stt_ms)
    return text, conf


# ── the rescue ────────────────────────────────────────────────────────────
def rescue_reason(text, *, wake_mode, has_wake_prefix, head_speech) -> str:
    """Why Whisper must decode this capture after all ('' = keep Parakeet's
    text). Only names a reason, never the text:

      'empty'   — Parakeet heard nothing;
      'no-wake' — wake_mode() is on (wake-word mode, or standby / sleep:
                  the monolith's _parakeet_wake_mode), has_wake_prefix(text)
                  is False, and
                  head_speech() says the clip's first RESCUE_HEAD_S hold
                  speech — or cannot tell (None): a misheard wake word
                  must not lose the turn.

    The callables are read in that order and only when needed. Any error in
    them rescues ('check-failed'): a slower turn, never a lost one."""
    try:
        if not (text or "").strip():
            return "empty"
        if not wake_mode():
            return ""
        if has_wake_prefix(text):
            return ""
        speech = head_speech()
        if speech is None or speech:
            return "no-wake"
        return ""
    except Exception:
        return "check-failed"


class Latch:
    """Parakeet's off switch for the session, shared by the primary path, the
    shadow worker and the boot warmer. Thread-safe; logs once."""

    def __init__(self, log=print):
        self._log = log
        self._mu = threading.Lock()
        self.failed = ""

    def trip(self, exc, quiet: bool = False) -> bool:
        """Latch off on `exc`. True the first time (and then logs, unless
        `quiet`: the boot warmer's own failure line is the log line)."""
        why = f"{type(exc).__name__}: {exc}"[:200] or "failed"
        with self._mu:
            if self.failed:
                return False
            self.failed = why
        if not quiet:
            try:
                self._log(f"  [stt] parakeet off for this session ({why}) "
                          f"— Whisper decodes the owner's captures")
            except Exception:
                pass
        return True


class Primary:
    """STT_ENGINE='parakeet' for one owner capture (_transcribe_capture).
    Every dependency is injected:

      decode(audio)  -> (text, conf)   Parakeet, under the monolith's
                                       _parakeet_lock (never _stt_lock)
      whisper(audio) -> (text, conf)   the monolith's transcribe()
      post_text(text) -> text          STT_REPLACEMENTS, then
                                       STT_REPLACEMENTS_PARAKEET
      rescue(text, audio) -> str       rescue_reason() with the live wake
                                       mode and the Silero head check
      note(value)                      [turn-timing] stt_engine
      hotwords() -> bool               STT_HOTWORDS is set (then logged once)

    run() returns what the turn uses. Never raises past whisper(): a
    Parakeet failure latches off (one log line) and Whisper decodes."""

    def __init__(self, decode, whisper, *, latch, post_text=None,
                 rescue=None, note=None, hotwords=None, log=print):
        self._decode = decode
        self._whisper = whisper
        self.latch = latch
        self._post = post_text or (lambda t: t)
        self._rescue = rescue or (lambda t, a: "" if (t or "").strip()
                                  else "empty")
        self._note = note or (lambda v: None)
        self._hotwords = hotwords or (lambda: False)
        self._log = log
        self._mu = threading.Lock()
        self._hot_logged = False
        self.decodes = 0
        self.rescues = 0

    def _say(self, line: str) -> None:
        try:
            self._log(line)
        except Exception:
            pass

    def _note_engine(self, value: str) -> None:
        try:
            self._note(value)
        except Exception:
            pass

    def _hotwords_once(self) -> None:
        try:
            if self._hot_logged or not self._hotwords():
                return
            self._hot_logged = True
            self._say("  [stt] parakeet ignores STT_HOTWORDS (Whisper-only); "
                      "map its mishearings in STT_REPLACEMENTS_PARAKEET")
        except Exception:
            pass

    def run(self, audio):
        if self.latch.failed:
            self._note_engine("whisper")
            return self._whisper(audio)
        self._hotwords_once()
        try:
            text, conf = self._decode(audio)
            text = self._post(text)
        except Exception as e:
            self.latch.trip(e)
            self._note_engine("whisper-fallback")
            return self._whisper(audio)
        with self._mu:
            self.decodes += 1
        why = self._rescue(text, audio)
        if why:
            with self._mu:
                self.rescues += 1
                n, of = self.rescues, self.decodes
            self._say(f"  [stt] parakeet -> whisper rescue ({why}; {n} of "
                      f"{of} parakeet decodes)")
            self._note_engine("parakeet-rescued")
            return self._whisper(audio)
        self._note_engine("parakeet")
        return text, conf


# ── the shadow A/B (STT_SHADOW) ───────────────────────────────────────────
def _norm_words(text) -> str:
    out = []
    for w in str(text or "").lower().split():
        w = "".join(ch for ch in w if ch.isalnum() or ch == "'")
        if w:
            out.append(w)
    return " ".join(out)


def _num(v, nd=4):
    return round(float(v), nd) if _finite(v) else None


def _side(text, conf, judged, ms) -> dict:
    c = conf if isinstance(conf, dict) else {}
    d = {"text": text or "", "stt_ms": ms,
         "no_speech_prob": _num(c.get("no_speech_prob")),
         "avg_logprob": _num(c.get("avg_logprob"))}
    for k in ("tok_lp_mean", "tok_lp_min", "n_tok"):
        if k in c:
            d[k] = _num(c.get(k)) if k != "n_tok" else c.get(k)
    d.update(judged if isinstance(judged, dict) else {"judge_error": True})
    return d


def append_jsonl(path, row, max_bytes=AB_MAX_BYTES) -> bool:
    """Append one JSON line to `path` unless the file already holds
    `max_bytes`. True when written. Never raises."""
    try:
        try:
            if os.path.getsize(path) >= max_bytes:
                return False
        except OSError:
            pass
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
    except Exception:
        return False


class Shadow:
    """STT_SHADOW='parakeet': Whisper's transcript stands; Parakeet decodes
    the same capture LATER and both are judged into the A/B file.

      decode(audio) -> (text, conf)   Parakeet (under _parakeet_lock)
      busy() -> bool                  a turn or an utterance is in progress
      judge(text, conf, peak, ctx)    the real gates' verdicts (a dict)
      write_row(row) -> bool          append to data/stt_ab.jsonl

    offer() never blocks the voice thread: it copies the audio into a
    bounded queue (a full queue drops the capture and counts it). ONE daemon
    takes captures in order; each waits until busy() is False, at most
    SHADOW_WAIT_S from when it was offered (else it is dropped and counted),
    so the decode never competes with a turn. The log gets numbers only —
    never either transcript (the words go to the A/B file alone). Clips are
    never saved. A Parakeet failure trips the shared latch: no more offers.
    `clock` / `wait` are injectable (tests run without sleeping)."""

    def __init__(self, decode, busy, judge, write_row, *, latch,
                 maxsize=SHADOW_QUEUE_MAX, wait_s=SHADOW_WAIT_S,
                 poll_s=SHADOW_POLL_S, log=print, clock=time.monotonic,
                 wait=None, wall=time.time):
        self._decode = decode
        self._busy = busy
        self._judge = judge
        self._write = write_row
        self.latch = latch
        self._q = queue.Queue(maxsize=max(1, int(maxsize)))
        self._wait_s = float(wait_s)
        self._poll_s = float(poll_s)
        self._log = log
        self._clock = clock
        self._stop = threading.Event()
        self._wait = wait if wait is not None else self._stop.wait
        self._wall = wall
        self._mu = threading.Lock()
        self._thread = None
        self.offered = 0
        self.dropped_full = 0
        self.dropped_busy = 0
        self.rows = 0

    # -- voice thread --------------------------------------------------------
    def offer(self, audio, text, conf, stt_ms, peak, ctx=None,
              start: bool = True) -> bool:
        """Queue one capture: a COPY of `audio`, Whisper's (text, conf), its
        decode wall time `stt_ms`, the capture's peak RMS and the gate
        context `ctx` read now. False (and nothing queued) when the latch is
        off, the queue is full, or anything fails. Never raises."""
        try:
            if self.latch.failed:
                return False
            import numpy as np
            item = {"t": self._clock(), "wall": self._wall(),
                    "audio": np.array(audio, dtype=np.float32,
                                      copy=True).reshape(-1),
                    "text": text or "",
                    "conf": dict(conf) if isinstance(conf, dict) else {},
                    "stt_ms": stt_ms, "peak": float(peak or 0.0),
                    "ctx": dict(ctx) if isinstance(ctx, dict) else {}}
            try:
                self._q.put_nowait(item)
            except queue.Full:
                with self._mu:
                    self.dropped_full += 1
                return False
            with self._mu:
                self.offered += 1
            if start:
                self._ensure_thread()
            return True
        except Exception:
            return False

    def pending(self) -> int:
        return self._q.qsize()

    def _ensure_thread(self) -> None:
        with self._mu:
            th = self._thread
            if th is not None and th.is_alive():
                return
            th = threading.Thread(target=self._run, daemon=True,
                                  name="stt-shadow")
            self._thread = th
        th.start()

    def stop(self) -> None:
        self._stop.set()

    # -- the daemon ------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self.process(item)
            except Exception:
                pass

    def wait_idle(self, deadline: float) -> bool:
        """True once busy() is False; False when `deadline` (clock time)
        passes first or the worker is stopped. A busy() error counts as
        busy."""
        while True:
            try:
                busy = bool(self._busy())
            except Exception:
                busy = True
            if not busy:
                return True
            if self._stop.is_set() or self._clock() >= deadline:
                return False
            self._wait(self._poll_s)

    def process(self, item) -> "dict | None":
        """One queued capture -> its A/B row (also written), or None when it
        was dropped (busy past the deadline, latch off, decode failed)."""
        if not self.wait_idle(item["t"] + self._wait_s):
            with self._mu:
                self.dropped_busy += 1
                n = self.dropped_busy
            self._say(f"  [stt-shadow] capture dropped: a turn ran past "
                      f"{self._wait_s:.0f} s ({n} dropped)")
            return None
        if self.latch.failed:
            return None
        waited_ms = int(round((self._clock() - item["t"]) * 1000.0))
        try:
            p_text, p_conf = self._decode(item["audio"])
        except Exception as e:
            self.latch.trip(e)
            return None
        ctx = item["ctx"]
        w_text, w_conf = item["text"], item["conf"]
        try:
            w_judged = self._judge(w_text, w_conf, item["peak"], ctx)
        except Exception:
            w_judged = None
        try:
            p_judged = self._judge(p_text, p_conf, item["peak"], ctx)
        except Exception:
            p_judged = None
        p_ms = p_conf.get("stt_ms") if isinstance(p_conf, dict) else None
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S",
                                time.localtime(item["wall"])),
            "speech_s": round(len(item["audio"]) / float(SAMPLE_RATE), 3),
            "peak": _num(item["peak"]),
            "wake_mode": bool(ctx.get("wake_mode")),
            "standby": bool(ctx.get("standby")),
            "shadow_wait_ms": waited_ms,
            "same_words": _norm_words(w_text) == _norm_words(p_text),
            "whisper": _side(w_text, w_conf, w_judged, item["stt_ms"]),
            "parakeet": _side(p_text, p_conf, p_judged, p_ms),
        }
        wrote = False
        try:
            wrote = bool(self._write(row))
        except Exception:
            wrote = False
        with self._mu:
            self.rows += 1 if wrote else 0
            n = self.rows

        def _acc(j):
            return "?" if not isinstance(j, dict) else \
                ("1" if j.get("accepted") else "0")
        self._say(f"  [stt-shadow] parakeet {p_ms if p_ms is not None else '?'}"
                  f" ms vs whisper {item['stt_ms']} ms, same words="
                  f"{int(row['same_words'])}, accepted whisper={_acc(w_judged)}"
                  f" parakeet={_acc(p_judged)}"
                  + (f" (row {n})" if wrote else " (row not written)"))
        return row

    def _say(self, line: str) -> None:
        try:
            self._log(line)
        except Exception:
            pass
