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
  * render(text) is one POST /tts bounded by a deadline. A line the listener
    is waiting for (a reply's first line) gets its LATENCY BUDGET: timeout_s,
    plus a per-character allowance for long lines. A line rendered AHEAD
    while earlier audio of the same reply still plays (needed_by, from
    core.sentence_tts.needed_by) gets max(that budget, the time until it is
    needed - NEEDED_BY_MARGIN_S): once a reply speaks in the clone voice,
    waiting for its next line beats switching that line to Kokoro. The
    connect alone is capped at CONNECT_TIMEOUT_S, so a server that is gone
    costs half a second, not the ~2 s a refused loopback connect takes on
    Windows. Never raises.
  * MAX_FAILURES LATENCY-CRITICAL misses IN A ROW (a first line that timed
    out, a line that missed the time it was needed, or a hard error: refused,
    HTTP error, silent render) put the clone into a COOL-DOWN: Kokoro speaks
    for COOLDOWN_BASE_S (5 min), then the clone is tried again; each further
    cool-down doubles, capped at COOLDOWN_MAX_S (30 min), and the doubling
    starts over once the clone has run a whole COOLDOWN_MAX_S without one.
    One log line per state change, nothing spoken. A look-ahead line that
    was given up on BEFORE it was needed (its wait capped at LOOKAHEAD_MAX_S)
    does not count. A render the server returns resets the count; a cache
    hit does not (no request was made, so it says nothing about the server).
    (Until 2026-10-04 three timeouts of ANY kind latched the clone off for
    the whole session: one slow minute on a busy GPU -- three long lines of
    one briefing rendered ahead with seconds of audio still queued -- cost
    the owner his voice until the next restart.)
  * Every render is trimmed (the model leaves ~0.3 s of near-silence at the
    end) and loudness-matched to Kokoro's level, so a reply that mixes the
    two engines (one fallback line) does not jump in volume.
  * Finished renders are kept in a small in-memory LRU (CACHE_MAX_BYTES),
    keyed by a hash of the voice and the text (the text itself is never
    stored), so a stock line ("Right away, sir.") is instant the second time.

Nothing here is sound: it never plays audio. Stdlib + numpy (imported
defensively); no monolith import. Tests: tests/test_clone_voice_client.py
(light tier, against a fake loopback server).
"""
from __future__ import annotations

import collections
import hashlib
import http.client
import io
import json
import os
import shlex
import subprocess
import threading
import time
import wave
from typing import Callable, Optional
from urllib.parse import urlsplit

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy is present wherever audio is
    np = None  # type: ignore

__all__ = ["MODEL_ID", "DEFAULT_URL", "MAX_FAILURES", "is_server_model",
           "parse_url", "render_budget_s", "line_deadline_s", "build_command",
           "decode_wav", "finish_audio", "Outcome", "CloneVoiceClient",
           "CLIENT"]

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
NEEDED_BY_MARGIN_S = 0.5
# The longest a line rendered ahead waits, however much audio is queued
# before it (bounds a wedged server; such a give-up is not counted).
LOOKAHEAD_MAX_S = 30.0
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


def line_deadline_s(budget_s: float, needed_by, now: float) -> tuple:
    """(seconds to wait, waited for need) for one line.

    No ``needed_by`` (a reply's first line: the listener waits now): the
    latency budget. Otherwise the line plays only once the audio queued ahead
    of it runs out, so it may take until then, less NEEDED_BY_MARGIN_S --
    never less than the budget, never more than LOOKAHEAD_MAX_S. The flag is
    True when the time it is needed (not the budget) set the wait."""
    budget = float(budget_s)
    if needed_by is None:
        return budget, False
    try:
        slack = float(needed_by) - NEEDED_BY_MARGIN_S - float(now)
    except Exception:
        return budget, False
    if slack > budget:
        return min(slack, LOOKAHEAD_MAX_S), True
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


def decode_wav(data: bytes):
    """(float32 mono array, sample rate) from 16-bit PCM WAV bytes. Raises on
    anything else."""
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
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a[: (a.size // ch) * ch].reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(a, dtype=np.float32), int(sr)


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
    sent, not a failure), or 'timed out' / 'error (...)' / 'http NNN' (a
    failure; ``counted`` says whether it counted toward the cool-down).

    ``lookahead``  -- rendered ahead with a needed-by time (not a first line)
    ``deadline_s`` -- the wait this render was given
    ``by_need``    -- that wait was set by the needed-by time, not the
                      latency budget"""

    __slots__ = ("audio", "sr", "ms", "reason", "cached", "server_ms",
                 "lookahead", "deadline_s", "by_need", "counted")

    def __init__(self, audio=None, sr: int = 0, ms: int = 0, reason: str = "",
                 cached: bool = False, server_ms=None, lookahead: bool = False,
                 deadline_s: float = 0.0, by_need: bool = False,
                 counted: bool = False):
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

    @property
    def ok(self) -> bool:
        return self.audio is not None

    @property
    def attempted(self) -> bool:
        """A request was made (or the cache answered for one)."""
        return self.ok or self.reason not in ("empty", "not-ready", "too-long")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Outcome(ok={self.ok}, sr={self.sr}, ms={self.ms}, "
                f"reason={self.reason!r}, cached={self.cached}, "
                f"lookahead={self.lookahead}, deadline_s={self.deadline_s}, "
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
                until the cool-down ends, then the clone is 'ready' again
                (checked on every status read); rearm() ends it at once

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
        # doubling), and when the last one ended (the doubling's reset).
        self._cool_until = 0.0
        self._cool_level = 0
        self._cool_ended_at = float("-inf")
        self._proc = None
        self._profile_memo: dict = {}   # name -> (expires_at, sha or "")
        self._sha_memo: dict = {}       # path -> ((size, mtime_ns), sha)
        self._cache: "collections.OrderedDict" = collections.OrderedDict()
        self._cache_bytes = 0
        self._logged: set = set()

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
        fresh count, and ONE log line. Called from every status read."""
        msg = None
        with self._mu:
            if self._status == "cooldown":
                now = self._clock()
                if now >= self._cool_until:
                    self._status = "ready"
                    self._reason = ""
                    self._fails = 0
                    self._cool_ended_at = now
                    msg = ("  [clone-voice] cool-down over; trying the clone "
                           "voice again")
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

    def is_ready(self) -> bool:
        self._cooldown_tick()
        with self._mu:
            return self._status == "ready"

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
                 budget_s: float):
        """(status, headers dict, body bytes). Raises on any transport
        error; TimeoutError when the deadline passes."""
        with self._mu:
            host, port = self._host, self._port
        if not host or not port:
            raise ConnectionError("no server address")
        deadline = self._clock() + float(budget_s)
        conn = http.client.HTTPConnection(
            host, port, timeout=min(CONNECT_TIMEOUT_S, self._left(deadline)))
        try:
            try:
                conn.connect()
            except OSError as e:
                # Refused, or no answer within CONNECT_TIMEOUT_S: the server
                # is not there. Not a render timeout.
                raise ConnectionError(f"cannot connect to {host}:{port} "
                                      f"({type(e).__name__})") from None
            sock = conn.sock
            sock.settimeout(self._left(deadline))
            headers = {"Connection": "close"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            conn.request(method, path, body=body, headers=headers)
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

    def _health(self):
        """(status code, JSON dict) or (None, {}) when nothing answers."""
        try:
            code, _h, data = self._request("GET", "/health", None,
                                           HEALTH_TIMEOUT_S)
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
            self._fails = 0
        self._log(f"  [clone-voice] ready ({how}, {self._clock() - t0:.1f} s, "
                  f"server pid {h.get('pid')}): replies use the clone voice; "
                  f"Kokoro covers any line it misses")
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
            self._logged.clear()
            return True

    # ── render ───────────────────────────────────────────────────────────
    def render(self, text: str, timeout_s: float, *, needed_by=None,
               budget_chars=None) -> Outcome:
        """One line through the server, trimmed and loudness-matched. Never
        raises; see Outcome for the reasons.

        ``timeout_s`` is the latency budget (VOICE_CLONE_TIMEOUT_S), grown
        per character past BASE_CHARS -- of ``budget_chars`` when that is
        longer than the line (the rest of a split first line keeps the whole
        line's budget). ``needed_by`` (time.monotonic()) marks a line rendered
        AHEAD of playback: it may wait until then (line_deadline_s), and a
        miss counts toward the cool-down only if it ran past that time."""
        t = str(text or "").strip()
        try:
            needed_by = None if needed_by is None else float(needed_by)
        except Exception:
            needed_by = None          # unusable: treated as a first line
        lookahead = needed_by is not None
        if not t:
            return Outcome(reason="empty", lookahead=lookahead)
        self._cooldown_tick()
        with self._mu:
            if self._status != "ready":
                return Outcome(reason="not-ready", lookahead=lookahead)
            sha = self._server_sha
        spoken = _normalize_text(t)
        if len(spoken) > MAX_CHARS:
            return Outcome(reason="too-long", lookahead=lookahead)
        key = hashlib.sha256((sha + "\0" + spoken).encode("utf-8")).hexdigest()
        hit = self._cache_get(key)
        if hit is not None:
            # No request was made, so this says nothing about the server: it
            # must NOT reset the failure streak. Otherwise a hung server never
            # cools down while cached acks come between the answers, and
            # every new line waits out its whole deadline.
            return Outcome(audio=hit[0].copy(), sr=hit[1], ms=0, cached=True,
                           lookahead=lookahead)
        try:
            n_budget = max(len(spoken), int(budget_chars or 0))
        except Exception:
            n_budget = len(spoken)
        budget = render_budget_s(n_budget, timeout_s)
        t0 = self._clock()
        wait_s, by_need = line_deadline_s(budget, needed_by, t0)
        info = {"lookahead": lookahead, "deadline_s": wait_s,
                "by_need": by_need}
        try:
            body = json.dumps({"text": spoken}).encode("utf-8")
            code, headers, data = self._request("POST", "/tts", body, wait_s)
            if code != 200:
                raise _HttpStatus(code)
            audio, sr = decode_wav(data)
            audio = finish_audio(audio, sr)
            if audio is None:
                raise ValueError("silent render")
        except TimeoutError:
            # Latency-critical unless this was a look-ahead line given up on
            # BEFORE it was needed (its wait capped at LOOKAHEAD_MAX_S): a
            # first line keeps the listener waiting, and a look-ahead line
            # that ran out its wait ran past the time it was needed.
            critical = (needed_by is None
                        or t0 + wait_s >= needed_by
                        - NEEDED_BY_MARGIN_S - 1e-6)
            return self._failed("timed out", t0, critical, info)
        except _HttpStatus as e:
            return self._failed(f"http {e.code}", t0, True, info)
        except Exception as e:
            return self._failed(f"error ({type(e).__name__}: {e})", t0, True,
                                info)
        ms = int(round((self._clock() - t0) * 1000.0))
        self._succeeded()
        self._cache_put(key, audio, sr)
        server_ms = headers.get("x-render-ms")
        return Outcome(audio=audio, sr=sr, ms=ms, server_ms=server_ms, **info)

    def _succeeded(self) -> None:
        with self._mu:
            self._fails = 0

    def _failed(self, reason: str, t0: float, critical: bool = True,
                info: Optional[dict] = None) -> Outcome:
        """A failed render. Only a latency-critical one counts toward the
        cool-down; MAX_FAILURES of them in a row start it (one log line)."""
        now = self._clock()
        ms = int(round((now - t0) * 1000.0))
        rest_s = 0.0
        with self._mu:
            if critical:
                self._fails += 1
                if self._fails >= MAX_FAILURES and self._status == "ready":
                    # The doubling starts over once the clone has run a
                    # whole COOLDOWN_MAX_S since the last cool-down ended.
                    if now - self._cool_ended_at >= COOLDOWN_MAX_S:
                        self._cool_level = 0
                    rest_s = min(COOLDOWN_BASE_S * (2 ** self._cool_level),
                                 COOLDOWN_MAX_S)
                    self._cool_level = min(self._cool_level + 1, 16)
                    self._cool_until = now + rest_s
                    self._status = "cooldown"
                    self._reason = reason
        if rest_s:
            self._log(f"  [clone-voice] {MAX_FAILURES} lines missed in a row "
                      f"(last: {reason}); the clone voice rests for "
                      f"{rest_s / 60.0:.0f} min and Kokoro speaks until then")
        return Outcome(reason=reason, ms=ms, counted=bool(critical),
                       **(info or {}))

    # ── cache ────────────────────────────────────────────────────────────
    def _cache_get(self, key: str):
        with self._mu:
            v = self._cache.get(key)
            if v is None:
                return None
            self._cache.move_to_end(key)
            return v

    def _cache_put(self, key: str, audio, sr: int) -> None:
        try:
            nbytes = int(audio.nbytes)
            if nbytes > CACHE_MAX_BYTES // 4:
                return
            with self._mu:
                old = self._cache.pop(key, None)
                if old is not None:
                    self._cache_bytes -= int(old[0].nbytes)
                self._cache[key] = (audio.copy(), int(sr))
                self._cache_bytes += nbytes
                while self._cache_bytes > CACHE_MAX_BYTES and self._cache:
                    _k, (a, _s) = self._cache.popitem(last=False)
                    self._cache_bytes -= int(a.nbytes)
        except Exception:
            pass

    def cache_len(self) -> int:
        with self._mu:
            return len(self._cache)


class _HttpStatus(Exception):
    def __init__(self, code):
        super().__init__(f"http {code}")
        self.code = code


# The process-wide client (the monolith, core.voice_clone and the voice-clone
# skill all read this one).
CLIENT = CloneVoiceClient()
