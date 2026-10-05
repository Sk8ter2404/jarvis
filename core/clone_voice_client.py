"""Client for the local clone voice server (VOICE_CLONE_MODEL =
"chatterbox_turbo_server", 2026-10-03).

WHY A SEPARATE PROCESS
----------------------
The clone engine needs CUDA torch and dependency pins (an older transformers,
numpy<2) that do not belong in JARVIS's own interpreter, whose torch is
CPU-only. The older in-process path (core/voice_clone.py) also put a CUDA
context inside JARVIS, which is what turns a terminated process into a
kernel-stuck corpse (see core.actions._release_native_resources). So the
model runs in its OWN process and venv, and JARVIS only speaks HTTP to it on
the loopback: no torch, no CUDA and no model in this process, ever.

THE SERVER CONTRACT (any server that follows it will do)
  GET  /health    200 + JSON {"ok": true, "ref_sha256": "<hex>", "pid": N}
                  once the model is loaded; 503 while it is still loading.
  POST /tts       body {"text": "..."} (Content-Type: application/json)
                  -> 200 audio/wav, 16-bit PCM; header X-Render-Ms.
  POST /shutdown  body {} -> stops the server and frees its GPU memory.

WHAT THIS MODULE DOES
  * start() reuses a server that is already up, else spawns
    VOICE_CLONE_SERVER_CMD as a detached, windowless process and polls
    /health for at most BOOT_WAIT_S. It prints ONE line for the outcome and
    runs on a daemon (start_async), never on the boot path.
  * The consent gate still holds. The clone is used only while the server's
    voice prompt (its ref_sha256) is the reference.wav of the ACTIVE consented
    profile (core.voice_clone.resolve_active_profile). A server that was
    started with any other voice is never used.
  * render(text) is one POST /tts bounded by a deadline (line_deadline_s).
    A line the listener is waiting for (a reply's first line) gets its
    LATENCY BUDGET: timeout_s, plus a per-character allowance for long
    lines. A line rendered AHEAD while earlier audio of the same reply still
    plays (needed_by, from core.sentence_tts.needed_by) may also take until
    it is needed, less NEEDED_BY_MARGIN_S (time for a Kokoro render). And
    once the reply already speaks in the clone voice (``hold``), such a line
    may run up to its own budget PAST the time it is needed: the listener
    then hears a pause no longer than a first line may make him wait, never
    a change of voice. The connect alone is capped at CONNECT_TIMEOUT_S, so
    a server that is gone costs half a second, not the ~2 s a refused
    loopback connect takes on Windows. ``cancel`` (the reply's stop Event,
    core.sentence_tts.reply_stopped) ends the wait at once when the reply
    is stopped: nothing will play that line any more. Never raises.
  * MAX_FAILURES LATENCY-CRITICAL misses IN A ROW (a first line that timed
    out, a line that missed the time it was needed, or a hard error: refused,
    HTTP error, silent render) put the clone into a COOL-DOWN: Kokoro speaks
    for COOLDOWN_BASE_S (5 min), then the clone is tried again ON PROBATION:
    its first counted miss starts the next cool-down at once (a server that
    is still slow or gone costs one line per cool-down, not three), its
    first success ends the probation. Each further cool-down doubles,
    capped at COOLDOWN_MAX_S (30 min); the doubling and the probation both
    lapse once the clone has run a whole COOLDOWN_MAX_S since the last
    cool-down ended. One log line per state change, nothing spoken. A
    look-ahead line given up on BEFORE it was needed (its wait capped at
    LOOKAHEAD_MAX_S), a line abandoned because its reply was stopped, and a
    background render nobody waits for (``count=False``: the filler clips)
    do not count. A render the server returns resets the count; a cache hit
    does not (no request was made, so it says nothing about the server), nor
    does a background render.
    (Until 2026-10-04 three timeouts of ANY kind latched the clone off for
    the whole session: one slow minute on a busy GPU -- a briefing line and
    two one-line replies, each a few tenths of a second over its budget --
    cost the owner his voice until the next restart.)
  * The consent gate holds across a cool-down: the first render after one
    asks /health again, and a server that now speaks another voice (or
    hides it) is never used -- whatever answers at the address may have
    changed while the clone rested.
  * Every render is trimmed (the model leaves ~0.3 s of near-silence at the
    end) and loudness-matched to Kokoro's level, so a reply that mixes the
    two engines (one fallback line) does not jump in volume.
  * Finished renders are kept in a small in-memory LRU (CACHE_MAX_BYTES),
    keyed by a hash of the voice, the model facts /health reports and the
    text (the text itself is never stored), so a stock line ("Right away,
    sir.") is instant the second time. With VOICE_CLONE_CACHE 'shadow' / 'on'
    and a disk folder attached (attach_cache: the monolith does it at boot)
    the takes are also kept on disk across restarts -- see
    core/clone_render_cache.py for the key, the take gate and the files, and
    core/clone_seed.py for the line ledger and the quiet-time seeding. A
    cached take is served only while the client is 'ready' (never while the
    server is down, starting or cooling down: no voice change mid-reply) and,
    from disk, only while the server's voice is the active consented
    profile's. A cached take that would OPEN a reply is served only while
    the clone looks healthy (no miss pending, the fast decoder on) and the
    server answers /health ready in the same process, voice and model
    (_cache_live: a recent check stands for LIVE_MEMO_S) -- else the line
    is rendered or missed like any other, so a reply never opens in the
    clone only to fall to Kokoro. A take is kept on disk only once a
    /health read AFTER it arrived shows the same server process, voice and
    model that the key names (_verify_take, on the writer thread): /tts
    does not say which voice spoke, and a server restarted with another
    reference would otherwise have its takes filed under the consented
    voice's key. Takes of a voice that is no longer a consented profile's
    are purged (purge_unconsented).
  * The fast decoder (C3, 2026-10-05): when /health says t3_decode is not
    'cuda-graph', or a line comes back with X-T3-Engine 'eager', ONE log line
    says the server is on its slow decoder (and one when it is back);
    decode_note() feeds voice_clone_status. Nothing is spoken and nothing is
    restarted (the server re-captures its graphs itself).
  * forget_last_reply() drops the takes of the reply the owner just heard
    (the "forget that line" action): the next time, it is rendered afresh.

Nothing here is sound: it never plays audio. Stdlib + numpy (imported
defensively); no monolith import. Tests: tests/test_clone_voice_client.py
(light tier, against a fake loopback server) and
tests/test_clone_render_cache.py.
"""
from __future__ import annotations

import collections
import hashlib
import http.client
import io
import json
import os
import re
import select
import shlex
import socket
import subprocess
import threading
import time
import wave
from typing import Callable, Optional
from urllib.parse import urlsplit

from core import clone_render_cache as _crc
from core import clone_seed as _cseed

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy is present wherever audio is
    np = None  # type: ignore

__all__ = ["MODEL_ID", "DEFAULT_URL", "MAX_FAILURES", "is_server_model",
           "parse_url", "render_budget_s", "line_deadline_s", "build_command",
           "decode_wav", "decode_wav_pcm16", "finish_audio", "Outcome",
           "CloneVoiceClient", "CLIENT"]

# The VOICE_CLONE_MODEL value that selects this engine.
MODEL_ID = "chatterbox_turbo_server"
DEFAULT_URL = "http://127.0.0.1:8767"

# Latency-critical misses in a row before the clone rests (the cool-down).
MAX_FAILURES = 3
# The cool-down: Kokoro speaks this long, then the clone is tried again. Each
# further cool-down doubles, capped at COOLDOWN_MAX_S; the doubling starts
# over once the clone has run COOLDOWN_MAX_S without one.
COOLDOWN_BASE_S = 300.0
COOLDOWN_MAX_S = 1800.0
# A line rendered ahead must be back this long before it is needed: the
# hand-off to the player, and a Kokoro render if the clone still misses.
# (Not with ``hold``: the reply already speaks in the clone, so a short pause
# is preferred to a change of voice -- see line_deadline_s.) An estimate, not
# a measurement: the one live sample of a Kokoro reply took 0.84 s from the
# render's start to the first play for a 2.9 s clip, play setup included, so
# a long line Kokoro voices after a miss can still leave a short gap.
NEEDED_BY_MARGIN_S = 0.5
# The longest a line rendered ahead waits, however much audio is queued
# before it (bounds a wedged server; a give-up before the line was needed is
# not counted). Never shortens a line's own latency budget.
LOOKAHEAD_MAX_S = 30.0
# While a render waits on a reply that may be stopped (``cancel``), the stop
# is checked this often.
CANCEL_POLL_S = 0.05
# Boot: how long start() waits for /health to say ready (a cold load from
# disk measured 18 s; 10-11 s once the weights are in the file cache).
BOOT_WAIT_S = 90.0
BOOT_POLL_S = 0.5
# A live loopback server accepts in well under a millisecond, even mid-render
# (its HTTP thread accepts while the model works). A refused loopback connect
# takes ~2 s on Windows (measured 2,039-2,046 ms), so cap the connect alone.
CONNECT_TIMEOUT_S = 0.5
HEALTH_TIMEOUT_S = 2.0
# The server refuses longer text; such a line is voiced by Kokoro without
# counting as a failure.
MAX_CHARS = 1000
# Deadline for a render: timeout_s covers a line of up to BASE_CHARS
# characters; each character beyond that adds PER_CHAR_S (the measured render
# rate is ~0.015 s per character on the 3090, so this is about 2x headroom).
BASE_CHARS = 80
PER_CHAR_S = 0.03
# The consent gate (profile meta + reference hash) is re-read at most this
# often, so a profile switch or a revoked consent is seen within seconds.
PROFILE_TTL_S = 5.0

# Loudness match: Kokoro's speech sits at an active RMS of ~0.10 (measured on
# its renders); the clone's short lines came out ~10 dB quieter.
TARGET_RMS = 0.10
PEAK_CAP = 0.95
MAX_GAIN = 8.0
# Trim: a sample counts as sound above TRIM_REL of the peak (and TRIM_ABS);
# keep this much before the first and after the last such sample (Kokoro's own
# renders keep ~0.04 s / ~0.10 s).
TRIM_REL = 0.01
TRIM_ABS = 5e-4
LEAD_KEEP_S = 0.04
TAIL_KEEP_S = 0.10

# In-memory render cache (float32 audio; 16 MB is ~170 s of speech).
CACHE_MAX_BYTES = 16 * 1024 * 1024
# The server decodes a line on its CUDA graphs up to ~600 characters (its
# static KV cache); a longer line uses the slow loop BY DESIGN, so only a
# line up to this long on 'eager' raises the fast-decode alert (C3).
GRAPH_MAX_CHARS = 500
# "Forget that line": the lines voiced for a listener in the last burst --
# consecutive lines no more than FORGET_GAP_S apart -- and only if that
# burst ended within FORGET_WINDOW_S.
FORGET_GAP_S = 10.0
FORGET_WINDOW_S = 600.0
RECENT_MAX = 32
# The same line voiced again within this long (an R3 pre-render dropped in
# the lock and rendered again) is one saying, not two, for the ledger.
LEDGER_DEDUPE_S = 15.0
# Before a cached take OPENS a reply, GET /health must answer ready in the
# same server process, voice and model, within this long (_cache_live). A
# bare connect is not enough: the server binds its port before it loads
# (~13 s of 503s) and a hung one still accepts, so a reply could open in the
# clone and go on in Kokoro -- the owner's "no cached lines while the server
# is down" (B6). Measured on this box: ~1 ms p99 with _fast_connect.
LIVENESS_TIMEOUT_S = 0.25
# A /health that confirmed the server (same pid, voice and model) this
# recently stands for it: no new probe (a reply's later cached lines, and
# the writer's checks of its rendered ones, keep it fresh).
LIVE_MEMO_S = 2.0
# The writer's /health read before a take is kept on disk (off the voice
# path; a server that does not answer in time keeps nothing).
VERIFY_TIMEOUT_S = 1.0

_LOOPBACK = ("127.0.0.1", "::1", "localhost")
# Interpreter variables of JARVIS's own Python that must not leak into the
# server's (a different Python in its own venv).
_STRIP_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
              "__PYVENV_LAUNCHER__")


# ═══════════════════════════════════════════════════════════════════════════
# Pure helpers
# ═══════════════════════════════════════════════════════════════════════════

def is_server_model(model) -> bool:
    """True when a VOICE_CLONE_MODEL value selects the clone voice server."""
    try:
        return str(model or "").strip().lower() == MODEL_ID
    except Exception:
        return False


def parse_url(url) -> Optional[tuple]:
    """(host, port) for a loopback ``http://host:port`` URL, else None.

    Only the loopback is accepted: reply text is never sent off the machine.
    'localhost' is mapped to 127.0.0.1 -- on this box it resolves ::1 first
    while the server binds IPv4, which cost ~2 s per request (the localhost
    tax, see the hard-won lessons)."""
    try:
        parts = urlsplit(str(url or "").strip())
        if parts.scheme != "http":
            return None
        host = (parts.hostname or "").lower()
        if host not in _LOOPBACK:
            return None
        if host == "localhost":
            host = "127.0.0.1"
        port = parts.port
        if not port:
            return None
        return host, int(port)
    except Exception:
        return None


def render_budget_s(text_len: int, timeout_s: float) -> float:
    """The whole-request deadline for a line of ``text_len`` characters."""
    try:
        base = float(timeout_s)
    except Exception:
        base = 2.5
    return base + PER_CHAR_S * max(0, int(text_len) - BASE_CHARS)


def line_deadline_s(budget_s: float, needed_by, now: float,
                    hold: bool = False) -> tuple:
    """(seconds to wait, waited for need) for one line.

    No ``needed_by`` (a reply's first line: the listener waits now): the
    latency budget. Otherwise the line plays only once the audio queued ahead
    of it runs out, so it may take until then, less NEEDED_BY_MARGIN_S (time
    for a Kokoro render if it still misses). With ``hold`` (the reply already
    speaks in the clone voice) it may run its whole budget PAST the time it
    is needed instead: a pause no longer than a first line may make the
    listener wait, rather than a change of voice mid-reply. Either way never
    less than the budget, and never more than LOOKAHEAD_MAX_S unless the
    budget itself is longer. The flag is True when the time it is needed
    (not the budget alone) set the wait."""
    budget = float(budget_s)
    if needed_by is None:
        return budget, False
    try:
        need_in = float(needed_by) - float(now)
    except Exception:
        return budget, False
    if not need_in == need_in:          # NaN: unusable, a first line
        return budget, False
    slack = (need_in + budget) if hold else (need_in - NEEDED_BY_MARGIN_S)
    wait = min(slack, LOOKAHEAD_MAX_S)
    if wait > budget:
        return wait, True
    return budget, False


def build_command(cmd, ref: str, port: int):
    """The Popen argument for VOICE_CLONE_SERVER_CMD with ``{ref}`` (the active
    profile's reference.wav) and ``{port}`` filled in. None when ``cmd`` is
    blank or cannot be parsed.

    * a JSON array (text starting with '[') -> a list, placeholders filled
      per element (no quoting needed);
    * otherwise, on Windows, a command line string for CreateProcess, with
      ``{ref}`` quoted when it contains spaces (a ``"{ref}"`` the owner
      already quoted is not quoted twice);
    * elsewhere, shlex-split, then filled per element."""
    try:
        text = str(cmd or "").strip()
        if not text:
            return None
        ref = str(ref or "")
        port_s = str(int(port))
        if text.startswith("["):
            items = json.loads(text)
            if not isinstance(items, list) or not items:
                return None
            return [str(x).replace("{ref}", ref).replace("{port}", port_s)
                    for x in items]
        if os.name == "nt":
            quoted = subprocess.list2cmdline([ref]) if ref else '""'
            text = text.replace('"{ref}"', quoted).replace("{ref}", quoted)
            return text.replace("{port}", port_s)
        return [x.replace("{ref}", ref).replace("{port}", port_s)
                for x in shlex.split(text)]
    except Exception:
        return None


def decode_wav_pcm16(data: bytes):
    """(int16 mono array, sample rate) from 16-bit PCM WAV bytes: the
    server's own samples (several channels are averaged and rounded). Raises
    on anything else."""
    if np is None:
        raise RuntimeError("numpy unavailable")
    with wave.open(io.BytesIO(data), "rb") as w:
        ch = w.getnchannels()
        width = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got {8 * width}-bit")
    if sr <= 0 or ch <= 0:
        raise ValueError("bad WAV header")
    a = np.frombuffer(raw, dtype="<i2")
    if ch > 1:
        a = np.round(a[: (a.size // ch) * ch].reshape(-1, ch)
                     .astype(np.float32).mean(axis=1))
    return np.ascontiguousarray(a, dtype=np.int16), int(sr)


def decode_wav(data: bytes):
    """(float32 mono array, sample rate) from 16-bit PCM WAV bytes. Raises on
    anything else."""
    pcm, sr = decode_wav_pcm16(data)
    return (np.ascontiguousarray(pcm.astype(np.float32) / 32768.0,
                                 dtype=np.float32), sr)


def _header_float(headers: dict, name: str):
    """A numeric X- header as a float, None when absent or not a number."""
    try:
        v = float(str(headers.get(name)).strip())
        return v if v == v else None
    except Exception:
        return None


def _active_rms(a, sr: int) -> float:
    """RMS over the 20 ms frames that carry speech (>= 10 % of the loudest
    frame), so pauses do not drag the level down."""
    fr = max(1, int(0.02 * sr))
    n = a.size // fr
    if n == 0:
        return float(np.sqrt(np.mean(a * a))) if a.size else 0.0
    frames = np.sqrt(np.mean(a[: n * fr].reshape(n, fr) ** 2, axis=1))
    top = float(frames.max())
    if top <= 0.0:
        return 0.0
    act = frames[frames >= 0.1 * top]
    return float(np.sqrt(np.mean(act ** 2))) if act.size else 0.0


def finish_audio(audio, sr: int):
    """Trim the ends and match Kokoro's loudness. None for an empty or silent
    render (the caller treats that as a failed line)."""
    if np is None:
        return None
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0 or sr <= 0:
        return None
    mag = np.abs(a)
    peak = float(mag.max())
    if not peak >= 1e-3:          # silent (NaN fails too)
        return None
    loud = np.flatnonzero(mag > max(TRIM_REL * peak, TRIM_ABS))
    start = max(0, int(loud[0]) - int(LEAD_KEEP_S * sr))
    end = min(a.size, int(loud[-1]) + 1 + int(TAIL_KEEP_S * sr))
    a = a[start:end]
    rms = _active_rms(a, sr)
    if rms > 0.0:
        gain = min(TARGET_RMS / rms, MAX_GAIN, PEAK_CAP / peak)
        a = a * np.float32(gain)
    return np.ascontiguousarray(a, dtype=np.float32)


def _normalize_text(text: str) -> str:
    """Times / decimals / versions / percents spelled out: the clone model has
    no text normaliser ("20.07" was heard as "two thousand seven"). The same
    pre-pass the in-process clone and Kokoro use."""
    try:
        from core.voice_clone import _normalize_numbers_for_speech as _norm
        return _norm(text)
    except Exception:
        return text


class Outcome:
    """One render attempt. ``audio`` is None when it did not produce audio;
    ``reason`` then says why: 'empty' / 'not-ready' / 'too-long' (nothing was
    sent, not a failure), 'cancelled' (its reply was stopped while it waited:
    nothing will play it, not a failure), or 'timed out' / 'error (...)' /
    'http NNN' (a failure; ``counted`` says whether it counted toward the
    cool-down).

    ``lookahead``  -- rendered ahead with a needed-by time (not a first line)
    ``deadline_s`` -- the wait this render was given
    ``by_need``    -- that wait was set by the needed-by time, not the
                      latency budget alone
    ``held``       -- ``hold`` applied: the reply already spoke in the clone,
                      so this line could run past the time it was needed
    ``late_s``     -- a look-ahead line that came back AFTER the time it was
                      needed: by how much (an estimate; the listener heard
                      about that much pause before it)
    ``cache``      -- where a cached take came from: 'mem' / 'disk' ('' when
                      the server rendered it)
    ``shadow``     -- VOICE_CLONE_CACHE 'shadow' only: 'would-hit' /
                      'would-miss' (would the disk have served this line)
    ``t3_ms`` ``tokens`` ``engine`` ``audio_ms`` -- the server's X-T3-Ms,
                      X-Speech-Tokens, X-T3-Engine and X-Audio-Ms for a
                      rendered line (None / '' when absent or cached)"""

    __slots__ = ("audio", "sr", "ms", "reason", "cached", "server_ms",
                 "lookahead", "deadline_s", "by_need", "counted", "held",
                 "late_s", "cache", "shadow", "t3_ms", "tokens", "engine",
                 "audio_ms")

    def __init__(self, audio=None, sr: int = 0, ms: int = 0, reason: str = "",
                 cached: bool = False, server_ms=None, lookahead: bool = False,
                 deadline_s: float = 0.0, by_need: bool = False,
                 counted: bool = False, held: bool = False,
                 late_s: float = 0.0, cache: str = "", shadow=None,
                 t3_ms=None, tokens=None, engine: str = "", audio_ms=None):
        self.audio = audio
        self.sr = sr
        self.ms = ms
        self.reason = reason
        self.cached = cached
        self.server_ms = server_ms
        self.lookahead = lookahead
        self.deadline_s = deadline_s
        self.by_need = by_need
        self.counted = counted
        self.held = held
        self.late_s = late_s
        self.cache = cache
        self.shadow = shadow
        self.t3_ms = t3_ms
        self.tokens = tokens
        self.engine = engine
        self.audio_ms = audio_ms

    @property
    def t3_ms_per_token(self):
        """T3 decode ms per speech token for a rendered line, else None."""
        try:
            if self.t3_ms is None or not self.tokens:
                return None
            return round(float(self.t3_ms) / float(self.tokens), 2)
        except Exception:
            return None

    @property
    def ok(self) -> bool:
        return self.audio is not None

    @property
    def cancelled(self) -> bool:
        """Its reply was stopped while it waited (not a failure)."""
        return self.reason == "cancelled"

    @property
    def attempted(self) -> bool:
        """A request was made (or the cache answered for one)."""
        return self.ok or self.reason not in ("empty", "not-ready", "too-long")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Outcome(ok={self.ok}, sr={self.sr}, ms={self.ms}, "
                f"reason={self.reason!r}, cached={self.cached}, "
                f"lookahead={self.lookahead}, deadline_s={self.deadline_s}, "
                f"held={self.held}, late_s={self.late_s}, "
                f"counted={self.counted})")


# ═══════════════════════════════════════════════════════════════════════════
# The client
# ═══════════════════════════════════════════════════════════════════════════

class CloneVoiceClient:
    """State for one JARVIS process. Status is one of:

      idle      nothing tried yet (the caller may start it)
      starting  start() is running on its daemon
      ready     the server answers and speaks the active profile's voice
      down      it never came up, or is not usable (terminal for the session;
                rearm() resets it)
      cooldown  MAX_FAILURES latency-critical misses in a row: Kokoro speaks
                until the cool-down ends, then the clone is 'ready' again,
                on probation (checked on every status read); rearm() ends
                it at once

    Thread-safe: renders run on the voice thread, the per-sentence worker,
    the pre-render and the filler warm, so every state change is under one
    lock and no lock is held across a request."""

    def __init__(self, *, log: Optional[Callable[[str], None]] = None,
                 clock: Callable[[], float] = time.monotonic,
                 popen=None, sleep: Optional[Callable[[float], None]] = None,
                 boot_wait_s: float = BOOT_WAIT_S):
        self._mu = threading.Lock()
        self._log_fn = log
        self._clock = clock
        self._popen = popen
        self._sleep = sleep or time.sleep
        self._boot_wait_s = float(boot_wait_s)
        self._status = "idle"
        self._reason = ""
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._server_sha = ""
        self._server_pid = None
        self._fails = 0
        # The cool-down: when it ends, how many have run back to back (the
        # doubling), when the last one ended (the doubling's reset), and the
        # probation after it (one counted miss rests the clone again).
        self._cool_until = 0.0
        self._cool_level = 0
        self._cool_ended_at = float("-inf")
        self._probation = False
        # Set when a cool-down ends: the next render re-checks /health (the
        # consent gate) before it sends anything.
        self._recheck = False
        self._proc = None
        self._profile_memo: dict = {}   # name -> (expires_at, sha or "")
        self._sha_memo: dict = {}       # path -> ((size, mtime_ns), sha)
        self._logged: set = set()
        # The profile start() was asked for (the disk cache serves only
        # while the server speaks ITS consented voice).
        self._profile = ""
        # What /health reported about the model (the cache key's facts) and
        # the decoder in use: 'cuda-graph' / 'eager' / '' (not known yet).
        self._server_info: dict = {}
        self._decode = ""
        self._health_at = float("-inf")
        # The last /health that confirmed the server: (clock when it was
        # SENT, _identity). It vouches for takes that arrived before it was
        # sent (_verify_take) and, for LIVE_MEMO_S, for the server being up
        # (_cache_live).
        self._verified: Optional[tuple] = None
        self._consented_memo = (float("-inf"), None)
        # The render cache: memory tier always; disk tier once attached.
        self.store = _crc.CloneRenderCache(
            mem_cap_fn=lambda: CACHE_MAX_BYTES)
        # Lines voiced for a listener: (clock, key, prefix, text), newest
        # last (forget_last_reply), and how often each was said (seeding).
        self._recent: "collections.deque" = collections.deque(
            maxlen=RECENT_MAX)
        self.ledger = _cseed.LineLedger(None)
        self.budget = _cseed.SeedBudget(None)
        self.keeper = None

    # ── small helpers ────────────────────────────────────────────────────
    def _log(self, msg: str) -> None:
        try:
            (self._log_fn or print)(msg)
        except Exception:
            pass

    def _log_once(self, key, msg: str) -> None:
        with self._mu:
            if key in self._logged:
                return
            self._logged.add(key)
        self._log(msg)

    def status(self) -> tuple:
        """(status, reason)."""
        self._cooldown_tick()
        with self._mu:
            return self._status, self._reason

    def server_pid(self):
        with self._mu:
            return self._server_pid

    def failures(self) -> int:
        with self._mu:
            return self._fails

    def cooldown_left_s(self) -> float:
        """Seconds of cool-down left (0.0 when not cooling down)."""
        self._cooldown_tick()
        with self._mu:
            if self._status != "cooldown":
                return 0.0
            return max(0.0, self._cool_until - self._clock())

    def _cooldown_tick(self) -> None:
        """A cool-down whose time is up ends here: back to 'ready' with a
        fresh count, on probation (the first counted miss rests the clone
        again; the first success ends it), with the server's voice to be
        re-checked before the next render (_recheck_voice), and ONE log line.
        Called from every status read."""
        msg = None
        with self._mu:
            if self._status == "cooldown":
                now = self._clock()
                if now >= self._cool_until:
                    self._status = "ready"
                    self._reason = ""
                    self._fails = 0
                    self._probation = True
                    self._recheck = True
                    self._cool_ended_at = now
                    msg = ("  [clone-voice] cool-down over; trying the clone "
                           "voice again (one more miss rests it again)")
        if msg:
            self._log(msg)

    # ── the consent gate ─────────────────────────────────────────────────
    def _file_sha(self, path: str) -> str:
        """SHA-256 of a file, memoised by (size, mtime). '' if unreadable."""
        try:
            st = os.stat(path)
            sig = (st.st_size, getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
            with self._mu:
                memo = self._sha_memo.get(path)
            if memo is not None and memo[0] == sig:
                return memo[1]
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            sha = h.hexdigest()
            with self._mu:
                self._sha_memo[path] = (sig, sha)
            return sha
        except Exception:
            return ""

    def _profile_ref(self, profile_name: str):
        """(reference.wav path, sha) for a usable (consented) profile, else
        None. Uncached: used once per start."""
        try:
            from core import voice_clone as _vc
            meta = _vc.resolve_active_profile(True, str(profile_name or ""))
            if meta is None:
                return None
            path = str(meta.get("reference_wav") or "")
            sha = self._file_sha(path) if path else ""
            return (path, sha) if sha else None
        except Exception:
            return None

    def profile_sha(self, profile_name: str) -> str:
        """The active profile's reference hash ('' = no usable profile),
        re-checked at most every PROFILE_TTL_S."""
        name = str(profile_name or "")
        now = self._clock()
        with self._mu:
            memo = self._profile_memo.get(name)
        if memo is not None and now < memo[0]:
            return memo[1]
        ref = self._profile_ref(name)
        sha = ref[1] if ref else ""
        with self._mu:
            self._profile_memo[name] = (now + PROFILE_TTL_S, sha)
        return sha

    def usable_for(self, profile_name: str) -> bool:
        """Ready AND speaking the voice of ``profile_name`` (consented). Never
        raises; a mismatch is logged once per (profile, voice)."""
        try:
            self._cooldown_tick()
            with self._mu:
                if self._status != "ready":
                    return False
                server_sha = self._server_sha
            want = self.profile_sha(profile_name)
            if want and want == server_sha:
                return True
            self._log_once(
                ("mismatch", str(profile_name or ""), want, server_sha),
                f"  [clone-voice] the server's voice is not the "
                f"'{profile_name}' profile's (or that profile is not "
                f"consented); Kokoro speaks until they match")
            return False
        except Exception:
            return False

    # ── HTTP ─────────────────────────────────────────────────────────────
    def _left(self, deadline: float) -> float:
        left = deadline - self._clock()
        if left <= 0:
            raise TimeoutError("deadline passed")
        return left

    def _request(self, method: str, path: str, body: Optional[bytes],
                 budget_s: float, cancel=None):
        """(status, headers dict, body bytes). Raises on any transport
        error; TimeoutError when the deadline passes; _Cancelled when
        ``cancel`` (an Event) is set while it waits for the reply."""
        with self._mu:
            host, port = self._host, self._port
        if not host or not port:
            raise ConnectionError("no server address")
        deadline = self._clock() + float(budget_s)
        conn = http.client.HTTPConnection(
            host, port, timeout=min(CONNECT_TIMEOUT_S, self._left(deadline)))
        try:
            # Refused, or no answer within CONNECT_TIMEOUT_S: the server is
            # not there (ConnectionError). Not a render timeout. Connected
            # here, not by http.client: see _fast_connect.
            sock = _fast_connect(host, port,
                                 min(CONNECT_TIMEOUT_S, self._left(deadline)))
            conn.sock = sock
            sock.settimeout(self._left(deadline))
            headers = {"Connection": "close"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            conn.request(method, path, body=body, headers=headers)
            if cancel is not None:
                # Wait for the render in short slices so a stopped reply
                # ends the wait at once (an HTTP wait, unlike a native
                # render, can be abandoned).
                _await_reply(sock, self._left(deadline), cancel)
                sock.settimeout(self._left(deadline))
            resp = conn.getresponse()     # waits for the render
            try:
                # The body follows the headers at once; bound it by what is
                # left of the same deadline.
                sock.settimeout(self._left(deadline))
            except TimeoutError:
                raise
            except Exception:
                pass
            data = resp.read()
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _health(self, budget_s: float = HEALTH_TIMEOUT_S):
        """(status code, JSON dict) or (None, {}) when nothing answers
        within ``budget_s``."""
        try:
            code, _h, data = self._request("GET", "/health", None,
                                           float(budget_s))
        except Exception:
            return None, {}
        try:
            obj = json.loads(data.decode("utf-8")) if data else {}
            if not isinstance(obj, dict):
                obj = {}
        except Exception:
            obj = {}
        return code, obj

    def stop_server(self) -> bool:
        """POST /shutdown (best effort). True if the server accepted it."""
        try:
            code, _h, _d = self._request("POST", "/shutdown", b"{}",
                                         HEALTH_TIMEOUT_S)
            return code == 200
        except Exception:
            return False

    # ── start ────────────────────────────────────────────────────────────
    def start_async(self, *, url: str, cmd: str, profile: str,
                    log_path: Optional[str] = None,
                    on_ready: Optional[Callable[[], None]] = None,
                    thread_factory=threading.Thread) -> bool:
        """Start the server on a 'clone-voice-start' daemon, once: only from
        'idle'. Returns True when this call started it. Never raises."""
        try:
            with self._mu:
                if self._status != "idle":
                    return False
                self._status = "starting"
                self._reason = ""
            th = thread_factory(target=self.start,
                                kwargs={"url": url, "cmd": cmd,
                                        "profile": profile,
                                        "log_path": log_path,
                                        "on_ready": on_ready},
                                name="clone-voice-start", daemon=True)
            th.start()
            return True
        except Exception as e:
            self._down(f"could not start the starter ({type(e).__name__}: {e})")
            return False

    def start(self, *, url: str, cmd: str, profile: str,
              log_path: Optional[str] = None,
              on_ready: Optional[Callable[[], None]] = None) -> str:
        """The blocking body of start_async: reuse a running server or spawn
        one, then wait (bounded) for it. Returns the final status. Prints ONE
        line for the outcome. Never raises."""
        t0 = self._clock()
        with self._mu:
            self._status = "starting"
            self._profile = str(profile or "")
        spawned = False
        try:
            hp = parse_url(url)
            if hp is None:
                return self._down(f"VOICE_CLONE_SERVER_URL {str(url)!r} is not "
                                  f"a loopback http://host:port address")
            with self._mu:
                self._host, self._port = hp
            ref = self._profile_ref(profile)
            if ref is None:
                return self._down(f"there is no consented voice profile "
                                  f"'{profile}' to speak with")
            ref_path, want = ref
            code, h = self._health()
            if code == 200 and h.get("ok"):
                return self._ready_from(h, want, "already running", t0,
                                        on_ready, spawned=False)
            if code is None:
                argv = build_command(cmd, ref_path, hp[1])
                if argv is None:
                    return self._down("no server is running and "
                                      "VOICE_CLONE_SERVER_CMD is empty")
                self._spawn(argv, log_path)
                spawned = True
            elif code not in (200, 503):
                return self._down(f"something else answers at {url} "
                                  f"(HTTP {code} on /health)")
            # 503 (another process is loading it) or we just spawned it.
            deadline = t0 + self._boot_wait_s
            while self._clock() < deadline:
                if spawned:
                    rc = self._proc.poll() if self._proc is not None else None
                    if rc is not None:
                        # The server binds its port BEFORE it loads, so a
                        # second instance exits at once (measured: code 4
                        # after 121 ms, no GPU work) when another one holds
                        # the port -- e.g. a restart racing the instance the
                        # previous boot just spawned. Wait for that one
                        # instead of writing the clone off for the session.
                        code, h = self._health()
                        if code not in (200, 503):
                            where = f" (log: {log_path})" if log_path else ""
                            return self._down(f"the server exited with code "
                                              f"{rc} while loading{where}")
                        spawned = False
                        if code == 200 and h.get("ok"):
                            return self._ready_from(h, want, "loaded", t0,
                                                    on_ready, spawned=False)
                self._sleep(BOOT_POLL_S)
                code, h = self._health()
                if code == 200 and h.get("ok"):
                    return self._ready_from(
                        h, want, "started" if spawned else "loaded", t0,
                        on_ready, spawned=spawned)
                if code is not None and code not in (200, 503):
                    return self._down(f"something else answers at {url} "
                                      f"(HTTP {code} on /health)")
            stopped = ""
            if spawned and self.stop_server():
                stopped = "; asked it to stop"
            return self._down(f"the server did not come up within "
                              f"{self._boot_wait_s:.0f} s{stopped}")
        except Exception as e:
            return self._down(f"start failed ({type(e).__name__}: {e})")

    def _spawn(self, argv, log_path: Optional[str]) -> None:
        env = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV}
        kw = {"stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT,
              "close_fds": True, "env": env}
        if os.name == "nt":
            # A HIDDEN console, inherited by the venv launcher's child
            # interpreter: zero windows. Not DETACHED_PROCESS, which leaves
            # the child to allocate a VISIBLE console of its own (the Ollama
            # ghost-window lesson). The process outlives JARVIS either way.
            kw["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        else:
            kw["start_new_session"] = True
        logf = None
        try:
            if log_path:
                os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
                logf = open(log_path, "wb")
                kw["stdout"] = logf
            else:
                kw["stdout"] = subprocess.DEVNULL
            popen = self._popen or subprocess.Popen
            self._proc = popen(argv, **kw)
        finally:
            if logf is not None:
                try:
                    logf.close()
                except Exception:
                    pass

    def _ready_from(self, h: dict, want: str, how: str, t0: float,
                    on_ready, spawned: bool) -> str:
        sha = str(h.get("ref_sha256") or "")
        if not sha or sha != want:
            # Never speak in a voice the consent gate has not passed. A server
            # this process started is useless then: ask it to stop. One that
            # was already running belongs to someone else: leave it alone.
            stopped = "; asked it to stop" if spawned and self.stop_server() else ""
            if not sha:
                return self._down("the server does not report its voice "
                                  "prompt (ref_sha256), so it cannot be "
                                  f"checked{stopped}")
            return self._down("the server was started with a different voice "
                              f"prompt than the active profile's{stopped}")
        with self._mu:
            self._status = "ready"
            self._reason = ""
            self._server_sha = sha
            self._server_pid = h.get("pid")
            self._server_info = _server_facts(h)
            self._health_at = self._clock()
            # Nothing renders before 'ready', so this /health vouches for
            # the server from now.
            self._verified = (self._health_at, _health_identity(h))
            self._fails = 0
            self._probation = False
            self._recheck = False
        self._log(f"  [clone-voice] ready ({how}, {self._clock() - t0:.1f} s, "
                  f"server pid {h.get('pid')}): replies use the clone voice; "
                  f"Kokoro covers any line it misses")
        self._note_decode(h.get("t3_decode"), "its /health")
        self.purge_unconsented()
        if on_ready is not None:
            try:
                on_ready()
            except Exception:
                pass
        return "ready"

    def _down(self, reason: str) -> str:
        with self._mu:
            self._status = "down"
            self._reason = reason
        self._log(f"  [clone-voice] not used this session: {reason}. Kokoro "
                  f"keeps speaking.")
        return "down"

    def rearm(self) -> bool:
        """Back to 'idle' from 'down' / 'cooldown' (an explicit 'use the
        clone voice' from the owner), so the next use starts or re-checks the
        server, and the cool-down doubling starts over. True if it changed
        anything."""
        with self._mu:
            if self._status not in ("down", "cooldown"):
                return False
            self._status = "idle"
            self._reason = ""
            self._fails = 0
            self._cool_until = 0.0
            self._cool_level = 0
            self._cool_ended_at = float("-inf")
            self._probation = False
            self._recheck = False      # start() checks the voice anyway
            self._logged.clear()
            return True

    def _recheck_voice(self, budget_s: float):
        """After a cool-down, before the next render: does the server at the
        address still speak the voice start() checked? None when it does (and
        the re-check is done), "voice" when it now speaks another voice or
        hides it (the client is then down for the session: the consent gate
        never passes a voice it has not checked), else the failure reason of
        an unreachable / not-ready server (counted like a failed render; the
        re-check stays pending). Bounded by ``budget_s`` (and the connect cap);
        never raises."""
        sent = self._clock()
        try:
            code, _h, data = self._request("GET", "/health", None,
                                           min(HEALTH_TIMEOUT_S, budget_s))
        except TimeoutError:
            return "timed out (voice re-check)"
        except Exception as e:
            return f"error ({type(e).__name__}: {e})"
        try:
            obj = json.loads(data.decode("utf-8")) if data else {}
            if not isinstance(obj, dict):
                obj = {}
        except Exception:
            obj = {}
        if code != 200 or not obj.get("ok"):
            return f"http {code} on the voice re-check"
        sha = str(obj.get("ref_sha256") or "")
        with self._mu:
            same = bool(sha) and sha == self._server_sha
            if same:
                self._recheck = False
                self._server_pid = obj.get("pid")
                self._server_info = _server_facts(obj)
                self._health_at = self._clock()
        if same:
            self._mark_verified(sent, obj)
            self._note_decode(obj.get("t3_decode"), "its /health")
            return None
        self._down("after a cool-down the server answering at its address "
                   "speaks a different voice prompt (or does not say which), "
                   "so it is no longer used")
        return "voice"

    # ── render ───────────────────────────────────────────────────────────
    def render(self, text: str, timeout_s: float, *, needed_by=None,
               budget_chars=None, hold: bool = False,
               cancel=None, count: bool = True) -> Outcome:
        """One line through the server, trimmed and loudness-matched. Never
        raises; see Outcome for the reasons.

        ``timeout_s`` is the latency budget (VOICE_CLONE_TIMEOUT_S), grown
        per character past BASE_CHARS -- of ``budget_chars`` when that is
        longer than the line (the rest of a split first line keeps the whole
        line's budget). ``needed_by`` (time.monotonic()) marks a line rendered
        AHEAD of playback: it may wait until then, or with ``hold`` (the
        reply already speaks in the clone) up to its budget past then
        (line_deadline_s); a miss counts toward the cool-down only if it ran
        past that time. ``cancel`` (an Event: the reply was stopped) ends the
        wait at once with reason 'cancelled', which never counts.
        ``count=False`` marks a background render nobody is waiting for (the
        filler clips): its outcome never touches the miss count, the
        probation or the cool-down."""
        t = _crc.normalise_text(text)
        try:
            needed_by = None if needed_by is None else float(needed_by)
            if needed_by is not None and not needed_by == needed_by:
                needed_by = None      # NaN
        except Exception:
            needed_by = None          # unusable: treated as a first line
        lookahead = needed_by is not None
        held = bool(hold) and lookahead
        if not t:
            return Outcome(reason="empty", lookahead=lookahead)
        if _is_set(cancel):
            return Outcome(reason="cancelled", lookahead=lookahead)
        self._cooldown_tick()
        with self._mu:
            if self._status != "ready":
                # Never a cached take either: the clone is not speaking now,
                # and a cached line amid Kokoro ones would change the voice
                # mid-reply.
                return Outcome(reason="not-ready", lookahead=lookahead)
            sha = self._server_sha
            pid = self._server_pid
            facts = dict(self._server_info)
        spoken = _crc.normalise_text(_normalize_text(t))
        if len(spoken) > MAX_CHARS:
            return Outcome(reason="too-long", lookahead=lookahead)
        key = self._key(sha, facts, spoken)
        prefix = _crc.voice_prefix(sha)
        mode = _crc.mode()
        # A take that would OPEN a reply for a listener: served only while
        # the clone looks healthy and the server is confirmed up in the same
        # process, voice and model (_cache_live) -- else this line renders
        # or misses like any other, so the reply never opens in the clone
        # only to go on in Kokoro (B6, one voice per reply). A LOOK-AHEAD
        # line belongs to a reply already speaking in the clone: its cached
        # take can only keep more of that reply in the one voice, so it is
        # served while 'ready' without a probe (as the in-memory cache
        # always was). Background renders (count=False) likewise.
        opener = bool(count) and not lookahead
        source = ""
        take = self.store.mem_get(key) if key else None
        if take is not None:
            source = "mem"
        elif key and mode == "on" and self._serve_ok(sha):
            take = self._disk_take(key, prefix, facts, len(spoken))
            if take is not None:
                source = "disk"
        if take is not None and opener:
            if not self._healthy_for_cache():
                take = None             # rendered (or missed) instead
            else:
                live = self._cache_live()
                if live == "down":
                    return Outcome(reason="not-ready", lookahead=lookahead)
                if live in ("recheck", "render"):
                    take = None         # the render path (re-checks first)
                elif live != "ok":
                    self.store._count("refused")
                    return self._failed(
                        "error (the server is not answering ready; its "
                        "cached take is not used)", self._clock(), True,
                        {"lookahead": lookahead, "cache": "refused"})
        if take is not None:
            # No request was made, so this says nothing about the server: it
            # must NOT reset the failure streak. Otherwise a hung server never
            # cools down while cached acks come between the answers, and
            # every new line waits out its whole deadline.
            self.store._count("mem_hits" if source == "mem" else "disk_hits")
            if count:
                self._remember_line(key, prefix, t)
            return Outcome(audio=take[0], sr=take[1], ms=0, cached=True,
                           cache=source, lookahead=lookahead)
        shadow = None
        if key and mode == "shadow" and self.store.disk_dir is not None:
            would = self.store.disk_has(key, prefix)
            shadow = "would-hit" if would else "would-miss"
            self.store._count("shadow_would_hit" if would
                              else "shadow_would_miss")
            if count:
                st = self.store.stats()
                h = st["shadow_would_hit"]
                n = h + st["shadow_would_miss"]
                self._log(f"  [clone-cache] shadow {shadow} ({h}/{n} lines "
                          f"the disk would have served)")
        self.store._count("misses")
        try:
            n_budget = max(len(spoken), int(budget_chars or 0))
        except Exception:
            n_budget = len(spoken)
        budget = render_budget_s(n_budget, timeout_s)
        with self._mu:
            recheck = self._recheck
        if recheck:
            # The first request after a cool-down: is it still the voice the
            # consent gate passed? (Not timed into the line's deadline: a
            # live server answers /health in milliseconds.)
            t_r = self._clock()
            why = self._recheck_voice(budget)
            if why == "voice":
                return Outcome(reason="not-ready", lookahead=lookahead)
            if why:
                return self._failed(why, t_r, bool(count),
                                    {"lookahead": lookahead})
        t0 = self._clock()
        wait_s, by_need = line_deadline_s(budget, needed_by, t0, hold=held)
        info = {"lookahead": lookahead, "deadline_s": wait_s,
                "by_need": by_need, "held": held}
        try:
            body = json.dumps({"text": spoken}).encode("utf-8")
            code, headers, data = self._request("POST", "/tts", body, wait_s,
                                                cancel=cancel)
            if code != 200:
                raise _HttpStatus(code)
            pcm16, sr = decode_wav_pcm16(data)
            audio = finish_audio(pcm16.astype(np.float32) / 32768.0, sr)
            if audio is None:
                raise ValueError("silent render")
        except _Cancelled:
            # The reply was stopped: nothing will play this line, so it says
            # nothing about the server's speed -- no count, no streak reset.
            ms = int(round((self._clock() - t0) * 1000.0))
            return Outcome(reason="cancelled", ms=ms, **info)
        except TimeoutError:
            # Latency-critical unless this was a look-ahead line given up on
            # BEFORE it was needed (its wait capped at LOOKAHEAD_MAX_S): a
            # first line keeps the listener waiting, and a look-ahead line
            # that ran out its wait ran past the time it was needed.
            critical = (needed_by is None
                        or t0 + wait_s >= needed_by
                        - NEEDED_BY_MARGIN_S - 1e-6)
            return self._failed("timed out", t0, critical and bool(count),
                                info)
        except _HttpStatus as e:
            return self._failed(f"http {e.code}", t0, bool(count), info)
        except Exception as e:
            return self._failed(f"error ({type(e).__name__}: {e})", t0,
                                bool(count), info)
        now = self._clock()
        ms = int(round((now - t0) * 1000.0))
        late_s = (max(0.0, now - needed_by) if needed_by is not None
                  else 0.0)
        if count:
            self._succeeded()
        server_ms = headers.get("x-render-ms")
        t3_ms = _header_float(headers, "x-t3-ms")
        tokens = _header_float(headers, "x-speech-tokens")
        engine = str(headers.get("x-t3-engine") or "").strip().lower()
        audio_ms = 1000.0 * pcm16.size / float(sr)
        self._note_engine(engine, len(spoken))
        if key:
            self.store.mem_put(key, audio, sr, prefix)
            if mode in ("shadow", "on"):
                self._persist_take(key, prefix, pcm16, sr, audio_ms,
                                   len(spoken), facts,
                                   _identity(pid, sha, facts), now)
        if count:
            self._remember_line(key, prefix, t)
        return Outcome(audio=audio, sr=sr, ms=ms, server_ms=server_ms,
                       late_s=late_s, shadow=shadow, t3_ms=t3_ms,
                       tokens=(int(tokens) if tokens is not None else None),
                       engine=engine, audio_ms=round(audio_ms, 1), **info)

    def _succeeded(self) -> None:
        with self._mu:
            self._fails = 0
            self._probation = False

    def _failed(self, reason: str, t0: float, critical: bool = True,
                info: Optional[dict] = None) -> Outcome:
        """A failed render. Only a latency-critical one counts toward the
        cool-down; MAX_FAILURES of them in a row start it -- or ONE on
        probation, right after a cool-down -- with one log line. A failure
        landing while a cool-down already runs (a render that was in flight
        when it began) changes nothing."""
        now = self._clock()
        ms = int(round((now - t0) * 1000.0))
        rest_s = 0.0
        what = ""
        with self._mu:
            if critical:
                self._fails += 1
                # The doubling and the probation both lapse once the clone
                # has run a whole COOLDOWN_MAX_S since the last cool-down.
                lapsed = now - self._cool_ended_at >= COOLDOWN_MAX_S
                if lapsed:
                    self._probation = False
                if self._status == "ready" and (
                        self._fails >= MAX_FAILURES or self._probation):
                    what = (f"{MAX_FAILURES} lines missed in a row"
                            if self._fails >= MAX_FAILURES else
                            "the first line after a cool-down missed too")
                    if lapsed:
                        self._cool_level = 0
                    rest_s = min(COOLDOWN_BASE_S * (2 ** self._cool_level),
                                 COOLDOWN_MAX_S)
                    self._cool_level = min(self._cool_level + 1, 16)
                    self._cool_until = now + rest_s
                    self._status = "cooldown"
                    self._reason = reason
                    self._probation = False
        if rest_s:
            self._log(f"  [clone-voice] {what} (last: {reason}); the clone "
                      f"voice rests for {rest_s / 60.0:.0f} min and Kokoro "
                      f"speaks until then")
        return Outcome(reason=reason, ms=ms, counted=bool(critical),
                       **(info or {}))

    # ── cache ────────────────────────────────────────────────────────────
    @staticmethod
    def _key(sha: str, facts: dict, spoken: str):
        """The render cache key for `spoken` in this server's voice and
        model (core.clone_render_cache.make_key), or None."""
        return _crc.make_key(sha, facts.get("model"), facts.get("t3_dtype"),
                             facts.get("sample_rate"), spoken)

    def _serve_ok(self, sha: str) -> bool:
        """A take from DISK may be served: the server's voice is still the
        consented active profile's (re-read at most every PROFILE_TTL_S).
        Never raises."""
        try:
            with self._mu:
                profile = self._profile
            want = self.profile_sha(profile)
            return bool(want) and want == sha
        except Exception:
            return False

    def _disk_take(self, key: str, prefix: str, facts: dict, n_chars: int):
        """(finished float32 take, sr) from the disk tier, kept in memory
        too, or None. The take's length is checked against its text again
        (the fixed band of the take gate): a file that does not fit -- kept
        under an older rule, or damaged -- is deleted, never served. Never
        raises."""
        try:
            sr = int(facts.get("sample_rate") or 0)
            if sr <= 0:
                return None
            raw = self.store.disk_get(key, prefix, sr)
            if raw is None:
                return None
            r = _crc.take_ratio(1000.0 * raw.size / float(sr), n_chars)
            lo, hi = _crc.GATE_DEFAULT_BAND
            if r is None or not lo <= r <= hi:
                self.store.forget([key], count=False)
                self.store._count("rejected_on_read")
                return None
            audio = finish_audio(raw.astype(np.float32) / 32768.0, sr)
            if audio is None:
                return None
            self.store.mem_put(key, audio, sr, prefix)
            return audio, sr
        except Exception:
            return None

    def _persist_take(self, key: str, prefix: str, pcm16, sr: int,
                      audio_ms: float, n_chars: int, facts: dict,
                      ident: tuple, got_at: float) -> None:
        """Queue a fresh take for the disk. Only a take at the sample rate
        the server reports (the key carries it) is queued; on the writer it
        is kept only if the server it came from is still the one the key
        names (_verify_take) and it passes the take gate. Never raises."""
        try:
            if self.store.disk_dir is None:
                return
            if int(facts.get("sample_rate") or 0) != int(sr):
                return

            def keep() -> bool:
                if not self._verify_take(key, ident, got_at):
                    return False
                ok, ratio, band = self.store.gate.admit(prefix, audio_ms,
                                                        n_chars)
                if not ok:
                    self.store._count("rejected")
                    r = "?" if ratio is None else f"{ratio:.2f}"
                    self._log(f"  [clone-cache] take not kept on disk: its "
                              f"length is {r}x the usual for {n_chars} "
                              f"characters (kept range {band[0]:.2f}-"
                              f"{band[1]:.2f}); it plays this time and is "
                              f"rendered again next time")
                return ok

            self.store.disk_put(key, prefix, pcm16, sr=sr, keep=keep)
        except Exception:
            pass

    def _mark_verified(self, sent_at: float, h: dict) -> None:
        """A /health sent at `sent_at` confirmed the server `h` describes
        (the newest confirmation is kept)."""
        with self._mu:
            v = self._verified
            if v is None or sent_at >= v[0]:
                self._verified = (sent_at, _health_identity(h))

    def _verify_take(self, key: str, ident: tuple, got_at: float) -> bool:
        """On the writer, before a take is kept on disk: the take came from
        the server and voice its key names. The /tts reply does not say which
        process or voice prompt made it, so a /health sent AFTER the take
        arrived must show the same process (pid), voice prompt hash and
        model facts -- a pid is never reused within seconds, and a
        restarted server needs ~13 s to load, so a server that answers
        /health as that process now made the take. One confirmation covers
        every take that arrived before it was sent. Also: the voice is still
        the active consented profile's.

        A different server answering (a restart, another reference, another
        model): the take is dropped from memory too, and the client follows
        the server or stops using it (_adopt). No answer: not kept (it is
        rendered again next time), nothing else changes. Never raises."""
        try:
            if not self._serve_ok(ident[1]):
                return False
            with self._mu:
                v = self._verified
            if v is not None and v[0] >= got_at and v[1] == ident:
                return True
            sent = self._clock()
            code, h = self._health(VERIFY_TIMEOUT_S)
            if code != 200 or not h.get("ok"):
                return False
            if _health_identity(h) == ident:
                self._mark_verified(sent, h)
                return True
            self.store.forget([key], count=False)
            with self._mu:
                ready = self._status == "ready"
                cur = _identity(self._server_pid, self._server_sha,
                                self._server_info)
            if ready and cur == ident:
                self._adopt(h, sent)
            return False
        except Exception:
            return False

    def _healthy_for_cache(self) -> bool:
        """The clone looks able to voice the REST of a reply: no counted miss
        since its last success, not on probation, the fast decoder not known
        to be off. A cached take opens a reply only then (else a reply could
        open in the clone and miss to Kokoro on its next line)."""
        with self._mu:
            return (not self._fails and not self._probation
                    and self._decode != "eager")

    def _cache_live(self) -> str:
        """May a cached take open a reply now? 'ok' -- the server answered
        /health ready in the same process, voice and model (or did within
        LIVE_MEMO_S); 'recheck' -- the voice re-check after a cool-down is
        pending (the render path does it); 'down' -- it now speaks another
        voice (the client stopped using it); 'render' -- it restarted, or now
        speaks the active profile's new reference (the client follows it;
        this line is rendered, not served from the old process's cache);
        'no' -- not answering ready within LIVENESS_TIMEOUT_S (dead,
        loading, stuck). Never raises."""
        try:
            with self._mu:
                if self._recheck:
                    return "recheck"
                ident = _identity(self._server_pid, self._server_sha,
                                  self._server_info)
                v = self._verified
            now = self._clock()
            if v is not None and v[1] == ident and now - v[0] <= LIVE_MEMO_S:
                return "ok"
            code, h = self._health(LIVENESS_TIMEOUT_S)
            if code != 200 or not h.get("ok"):
                return "no"
            if _health_identity(h) == ident:
                self._mark_verified(now, h)
                self._note_decode(h.get("t3_decode"), "its /health")
                # The probe may have just found the slow decoder: then the
                # rest of the reply would crawl -- render this line too.
                return "ok" if self._healthy_for_cache() else "render"
            return "render" if self._adopt(h, now) else "down"
        except Exception:
            return "no"

    def _adopt(self, h: dict, sent_at: float) -> bool:
        """A ready /health from the server at the client's address that may
        describe a different server than the one recorded: the same voice
        prompt (a restart, new model facts) -- followed; the active consented
        profile's NEW reference -- followed, the cache keys follow the hash;
        anything else -- the client is down (the consent gate never passes a
        voice it has not checked). True while the client still uses the
        server. Never raises."""
        try:
            sha = str(h.get("ref_sha256") or "")
            with self._mu:
                same = bool(sha) and sha == self._server_sha
                profile = self._profile
                old_pid = self._server_pid
            if not same:
                if sha and sha == self.profile_sha(profile):
                    with self._mu:
                        self._server_sha = sha
                    self._log("  [clone-voice] the voice server now speaks "
                              "the active profile's new reference; the "
                              "render cache follows it")
                else:
                    self._down("the server answering at its address now "
                               "speaks a different voice prompt (or does "
                               "not say which), so it is no longer used")
                    return False
            elif h.get("pid") != old_pid:
                self._log(f"  [clone-voice] the voice server restarted "
                          f"(pid {old_pid} -> {h.get('pid')}) with the same "
                          f"voice; following it")
            with self._mu:
                self._server_pid = h.get("pid")
                self._server_info = _server_facts(h)
                self._health_at = self._clock()
            self._mark_verified(sent_at, h)
            self._note_decode(h.get("t3_decode"), "its /health")
            return True
        except Exception:
            return False

    def _remember_line(self, key, prefix: str, text: str) -> None:
        """A line voiced for a listener: for "forget that line" and for the
        seeding ledger. Never raises."""
        try:
            now = self._clock()
            with self._mu:
                again = any(k == key and now - ts < LEDGER_DEDUPE_S
                            for ts, k, _p, _t in self._recent)
                if not again:
                    self._recent.append((now, key, prefix, text))
            if not again:
                self.ledger.record(text)
        except Exception:
            pass

    def server_gpu_index(self):
        """The physical (PCI-order, = NVML) index of the GPU the server
        renders S3Gen on, from /health's 'cuda:N (physical, PCI order)';
        None when not known."""
        try:
            with self._mu:
                dev = str(self._server_info.get("device") or "")
            m = re.match(r"^cuda:(\d+)", dev.strip())
            return int(m.group(1)) if m else None
        except Exception:
            return None

    def cache_len(self) -> int:
        """Takes in the memory tier."""
        return self.store.mem_len()

    def cache_stats(self) -> dict:
        try:
            out = self.store.stats()
            out["mode"] = _crc.mode()
            out["ledger_lines"] = len(self.ledger)
            return out
        except Exception:
            return {}

    def voice_prefix(self) -> str:
        """The cache prefix of the voice the server speaks ('' = none)."""
        with self._mu:
            return _crc.voice_prefix(self._server_sha)

    def attach_cache(self, disk_dir: str) -> bool:
        """Give the render cache its disk tier (and the ledger and seed
        budget their files) under `disk_dir`. Takes of voices that are not a
        consented profile's are purged at once. True when usable. Never
        raises."""
        try:
            if not self.store.attach(disk_dir):
                return False
            self.ledger = _cseed.LineLedger(
                os.path.join(disk_dir, _cseed.LEDGER_FILE))
            self.budget = _cseed.SeedBudget(
                os.path.join(disk_dir, _cseed.STATE_FILE))
            self.purge_unconsented()
            return True
        except Exception:
            return False

    def start_keeper(self, **kw) -> bool:
        """Start the cache keeper daemon once (core.clone_seed.CacheKeeper,
        keyword arguments as its own). True when this call started it."""
        try:
            with self._mu:
                if self.keeper is not None:
                    return False
                self.keeper = _cseed.CacheKeeper(self, **kw)
                keeper = self.keeper
            return keeper.start()
        except Exception:
            return False

    def is_cached(self, text) -> bool:
        """VOICE_CLONE_CACHE 'on', the clone ready and healthy enough for a
        cached take to open a reply (_healthy_for_cache, no voice re-check
        pending), and `text` would be served from the cache right now
        (memory, or disk while the voice is the consented one). For the
        chunk planner: no I/O at all (render() still checks the server is
        up before a cached take opens the reply). Never raises."""
        try:
            if _crc.mode() != "on":
                return False
            t = _crc.normalise_text(text)
            if not t:
                return False
            self._cooldown_tick()
            if not self._healthy_for_cache():
                return False
            with self._mu:
                if self._status != "ready" or self._recheck:
                    return False
                sha = self._server_sha
                facts = dict(self._server_info)
            spoken = _crc.normalise_text(_normalize_text(t))
            key = self._key(sha, facts, spoken)
            if not key:
                return False
            if self.store.mem_has(key):
                return True
            if int(facts.get("sample_rate") or 0) <= 0:
                return False
            return (self.store.disk_has(key, _crc.voice_prefix(sha))
                    and self._serve_ok(sha))
        except Exception:
            return False

    def consented_prefixes(self):
        """The cache prefixes of every consented profile's reference.wav --
        an EMPTY set when the profiles folder is gone (deleting it is the
        most complete way to withdraw consent: everything is purged) -- or
        None when that cannot be told (the folder is there but cannot be
        read): nothing is purged on a None. Memoised for PROFILE_TTL_S.
        Never raises."""
        try:
            now = self._clock()
            with self._mu:
                until, memo = self._consented_memo
            if now < until:
                return memo
            from core import voice_clone as _vc
            out = set()
            if os.path.isdir(_vc.PROFILES_DIR):
                # list_profiles() reads an unlistable folder as empty; that
                # must not purge everything: the listing error -> None.
                os.listdir(_vc.PROFILES_DIR)
                for meta in _vc.list_profiles():
                    if not _vc.profile_is_usable(meta):
                        continue
                    p = _crc.voice_prefix(
                        self._file_sha(str(meta.get("reference_wav") or "")))
                    if p:
                        out.add(p)
            with self._mu:
                self._consented_memo = (now + PROFILE_TTL_S, out)
            return out
        except Exception:
            return None

    def purge_unconsented(self) -> int:
        """Drop every cached take whose voice is not a consented profile's
        reference any more (consent revoked, reference replaced, profile
        removed), in memory and on disk. One log line when anything went.
        Runs whatever VOICE_CLONE_CACHE says: those files are the cloned
        voice. Never raises."""
        try:
            held = self.store.prefixes()
            if not held:
                return 0
            keep = self.consented_prefixes()
            if keep is None or held <= keep:
                return 0
            n = self.store.purge_except(keep)
            # Consent changed: re-read the active profile at the next
            # _serve_ok (a write still queued is checked against it).
            with self._mu:
                self._profile_memo.clear()
            if n:
                self._log(f"  [clone-cache] removed {n} cached take"
                          f"{'s' if n != 1 else ''} of a voice that is no "
                          f"longer a consented profile's (consent withdrawn "
                          f"or its reference replaced)")
            return n
        except Exception:
            return 0

    def forget_last_reply(self, before=None) -> list:
        """'Forget that line': drop the takes of the last burst of lines
        voiced for a listener (lines no more than FORGET_GAP_S apart, the
        burst ended within FORGET_WINDOW_S) from memory and disk, and never
        seed them again. ``before`` (this client's clock: when the owner's
        request was accepted) leaves out every line voiced at or after it --
        the acknowledgement of the request itself ("Certainly, sir.") can be
        voiced before the action runs, and is not "that line". Returns the
        texts, in the order they were said ([] when there was nothing
        recent). The next time one is said it is rendered afresh. A write
        of one still queued never lands. Never raises."""
        try:
            now = self._clock()
            try:
                cut = None if before is None else float(before)
                if cut is not None and not cut > 0.0:
                    cut = None
            except Exception:
                cut = None
            with self._mu:
                items = [it for it in self._recent
                         if cut is None or it[0] < cut]
            if not items or now - items[-1][0] > FORGET_WINDOW_S:
                return []
            group = [items[-1]]
            for it in reversed(items[:-1]):
                if group[-1][0] - it[0] <= FORGET_GAP_S:
                    group.append(it)
                else:
                    break
            group.reverse()
            self.store.forget({it[1] for it in group if it[1]})
            texts = []
            for it in group:
                self.ledger.forget(it[3])
                if it[3] not in texts:
                    texts.append(it[3])
            drop = {id(it) for it in group}
            with self._mu:
                kept = [it for it in self._recent if id(it) not in drop]
                self._recent.clear()
                self._recent.extend(kept)
            self._log(f"  [clone-cache] forgot {len(texts)} line"
                      f"{'s' if len(texts) != 1 else ''} of the last reply; "
                      f"they are rendered afresh next time")
            return texts
        except Exception:
            return []

    # ── health: the fast decoder (C3) and a server that changed ───────────
    def _note_decode(self, value, source: str) -> None:
        """Track the server's decoder ('cuda-graph' / 'eager'); ONE log line
        when it falls back to the slow loop, one when it is back. Never
        raises."""
        try:
            v = {"cuda-graph": "cuda-graph", "graph": "cuda-graph",
                 "eager": "eager"}.get(str(value or "").strip().lower())
            if not v:
                return
            with self._mu:
                prev = self._decode
                self._decode = v
            if prev == v:
                return
            if v == "eager":
                self._log(f"  [clone-voice] fast decode is OFF on the voice "
                          f"server ({source} says the slow decoder): new "
                          f"lines render several times slower until it "
                          f"re-captures its CUDA graphs by itself; nothing "
                          f"is restarted")
            elif prev == "eager":
                self._log("  [clone-voice] fast decode is back on the voice "
                          "server (cuda-graph)")
        except Exception:
            pass

    def _note_engine(self, engine: str, n_chars: int) -> None:
        """A rendered line's X-T3-Engine. 'eager' on a line short enough for
        the graphs means the fast decoder is off (C3)."""
        if engine == "graph":
            self._note_decode("cuda-graph", "a rendered line")
        elif engine == "eager" and int(n_chars) <= GRAPH_MAX_CHARS:
            self._note_decode("eager", "a rendered line")

    def decode_state(self) -> str:
        """'cuda-graph' / 'eager' / '' (not known yet)."""
        with self._mu:
            return self._decode

    def decode_note(self) -> str:
        """A clause for voice_clone_status while the fast decoder is off,
        else ''."""
        if self.decode_state() == "eager":
            return ("the voice server's fast decoder is off just now, so new "
                    "lines come a little slower than usual")
        return ""

    def refresh_health(self, max_age_s: float = 0.0) -> bool:
        """Read /health again (if the last read is older than `max_age_s`)
        while the client is ready: the decoder (C3) and the server's voice.
        A server that now speaks the active consented profile's (new)
        reference is followed -- the cache keys follow the hash; one that
        speaks anything else is no longer used (down). True when the server
        answered ready. Bounded (HEALTH_TIMEOUT_S); never raises."""
        try:
            self._cooldown_tick()
            with self._mu:
                if self._status != "ready":
                    return False
                if self._clock() - self._health_at < float(max_age_s):
                    return True
            sent = self._clock()
            code, h = self._health()
            with self._mu:
                self._health_at = self._clock()
            if code != 200 or not h.get("ok"):
                return False
            return self._adopt(h, sent)
        except Exception:
            return False

    def seed_ready(self, max_health_age_s: float = 60.0):
        """None when a seed render may be sent now, else why not: the
        clone ready and not missing lines, its voice the consented one, the
        fast decoder on (a seed on the slow loop costs ~5x the GPU), a disk
        tier to keep it in. Never raises."""
        try:
            st = self.status()[0]
            if st != "ready":
                return f"clone {st}"
            with self._mu:
                if self._fails or self._probation or self._recheck:
                    return "clone missing lines"
            if self.store.disk_dir is None:
                return "no disk cache"
            if not self.refresh_health(max_health_age_s):
                return "server not answering ready"
            with self._mu:
                decode = self._decode
                sha = self._server_sha
            if decode != "cuda-graph":
                return "fast decode off"
            if not self._serve_ok(sha):
                return "voice not the consented profile's"
            return None
        except Exception as e:
            return f"error ({type(e).__name__})"


def _server_facts(h: dict) -> dict:
    """The model facts from a /health body that the cache key carries, plus
    its decoder. Missing ones are ''/0."""
    try:
        sr = int(h.get("sample_rate") or 0)
    except Exception:
        sr = 0
    return {"model": str(h.get("model") or ""),
            "t3_dtype": str(h.get("t3_dtype") or ""),
            "sample_rate": sr,
            "t3_decode": str(h.get("t3_decode") or ""),
            "device": str(h.get("s3gen_device") or h.get("device") or "")}


def _identity(pid, sha, facts: dict) -> tuple:
    """Which server made / would make a take: its process, voice prompt hash
    and the model facts the cache key carries."""
    try:
        sr = int(facts.get("sample_rate") or 0)
    except Exception:
        sr = 0
    return (pid, str(sha or ""), str(facts.get("model") or ""),
            str(facts.get("t3_dtype") or ""), sr)


def _health_identity(h: dict) -> tuple:
    """_identity of the server a /health body describes."""
    return _identity(h.get("pid"), h.get("ref_sha256"), _server_facts(h))


def _fast_connect(host: str, port: int, timeout_s: float):
    """A connected TCP socket to ``host:port``, or ConnectionError. Each
    address is tried for at most ``timeout_s`` (as socket.create_connection
    does) with a NON-BLOCKING connect and select(): CPython's own timed
    connect costs ~20 ms on about a quarter of loopback connects on Windows
    (measured 2026-10-05, 300 connects 10 ms apart: p90 20.3 ms, p99 25 ms;
    this way p90 0.5 ms, p99 0.7-1.0 ms). The socket is returned blocking;
    the caller sets its timeout."""
    last = None
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise ConnectionError(f"cannot resolve {host} ({e})") from None
    for fam, typ, proto, _cn, addr in infos:
        sock = None
        try:
            sock = socket.socket(fam, typ, proto)
            sock.setblocking(False)
            try:
                sock.connect(addr)
            except (BlockingIOError, InterruptedError):
                pass
            _r, w, x = select.select([], [sock], [sock],
                                     max(0.0, float(timeout_s)))
            if not w and not x:
                raise TimeoutError("connect timed out")
            # Windows reports a failed connect in the exception set.
            err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if err or x:
                raise OSError(err, os.strerror(err) if err else "refused")
            sock.setblocking(True)
            return sock
        except OSError as e:
            last = e
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
    raise ConnectionError(f"cannot connect to {host}:{port} "
                          f"({type(last).__name__ if last else 'no address'})")


class _HttpStatus(Exception):
    def __init__(self, code):
        super().__init__(f"http {code}")
        self.code = code


class _Cancelled(Exception):
    """The reply a render belonged to was stopped while it waited."""


def _is_set(ev) -> bool:
    """True when ``ev`` (an Event-like stop flag, or None) is set. Never
    raises."""
    if ev is None:
        return False
    try:
        return bool(ev.is_set())
    except Exception:
        return False


def _await_reply(sock, wait_s: float, cancel) -> None:
    """Wait at most ``wait_s`` (real seconds, like a socket timeout) for the
    reply to start arriving on ``sock``, checking ``cancel`` every
    CANCEL_POLL_S. Raises _Cancelled / TimeoutError; returns once the socket
    is readable (data, or the server closing it -- the read that follows
    reports that)."""
    end = time.monotonic() + max(0.0, float(wait_s))
    while True:
        if _is_set(cancel):
            raise _Cancelled()
        left = end - time.monotonic()
        if left <= 0:
            raise TimeoutError("deadline passed")
        readable, _w, _x = select.select([sock], [], [],
                                         min(CANCEL_POLL_S, left))
        if readable:
            return


# The process-wide client (the monolith, core.voice_clone and the voice-clone
# skill all read this one).
CLIENT = CloneVoiceClient()
