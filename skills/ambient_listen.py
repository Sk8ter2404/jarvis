"""
Ambient listening daemon — passive whisper transcription with wake-word
nudges, plus optional system-audio and periodic screen-snapshot daemons
(ambient-mode-2 multimodal extension).

Mic mode (the original ambient-mode-1):
    Opens a dedicated sd.InputStream on a background thread, continuously
    captures audio in short chunks, batches ~2-3 s of audio at a time, and
    runs each batch through bobert_companion.transcribe() without blocking
    the main listen loop. Every accepted transcript is appended to a rolling
    in-memory buffer (last AMBIENT_LISTEN_BUFFER_MINUTES) and mirrored to
    ambient_listen_state.json so the user can read what was overheard.

    Each new transcript is scanned for any bobert_companion.WAKE_PHRASES
    match using word-boundary regex (so 'jar visit' doesn't trip 'jarvis').
    On a match the skill calls bobert_companion.proactive_announce(...) so
    the next turn boundary speaks an acknowledgement.

System-audio mode (ambient-mode-2):
    Opens a WASAPI loopback InputStream on the default speaker device so
    Whisper sees whatever the PC itself is playing (YouTube, Teams call,
    podcast). Buffers 30 s chunks, RMS-gated, results appended to
    data/ambient_transcripts.jsonl tagged source='system_audio' plus the
    focused-window title for attribution. Independent of mic mode — both
    can run together.

Screen mode (ambient-mode-2):
    Captures all monitors every AMBIENT_SCREEN_INTERVAL_S, dedupes against
    the prior screenshot via a pHash-style similarity check, and on novel
    frames fires the local VLM (qwen2.5vl preferred via
    bobert_companion._call_local_vision) with a prompt asking for content
    summary + entity mentions + sensitive-data flags. Results land in
    data/ambient_screen_log.jsonl. A daily AMBIENT_VISION_BUDGET_USD cap
    and an AMBIENT_SCREEN_BLOCKLIST (regex over focused-window title /
    URL) prevent runaway spend or screenshotting 1Password / banking.

Actions registered with the dispatcher:
    ambient_listen_start         — open the MIC stream (idempotent)
    ambient_listen_stop          — close the mic stream
    ambient_listen_status        — running state + buffer + multimodal info
    ambient_audio_start          — open the WASAPI loopback stream
    ambient_audio_stop           — close the loopback stream
    ambient_screen_start         — begin periodic screen-snapshot loop
    ambient_screen_stop          — stop the screen-snapshot loop
    ambient_full_start           — mic + system audio + screen all on
    ambient_full_stop            — all three off
    ambient_mic_only             — keep mic on, turn audio + screen off

All daemons share the existing _lock RLock pattern; state cleanup uses the
same hard-cap + time-window logic as ambient-mode-1.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import deque
from typing import Optional

import numpy as np


_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# STAGING ISOLATION (2026-07-21): resolve through core.paths so a
# JARVIS_STAGING process writes data_staging/ instead of the live data/.
# A private join here is how a staging-isolated action sweep overwrote the
# LIVE smart-home catalog while the settings md5 tripwire stayed green.
try:
    from core.paths import data_dir as _jarvis_data_dir
    _DATA_DIR = _jarvis_data_dir()
except Exception:   # pragma: no cover - core.paths is in-tree
    _DATA_DIR = os.path.join(_PROJECT_DIR, "data")
_STATE_PATH   = os.path.join(_PROJECT_DIR, "ambient_listen_state.json")
_AUDIO_JSONL  = os.path.join(_DATA_DIR, "ambient_transcripts.jsonl")
_SCREEN_JSONL = os.path.join(_DATA_DIR, "ambient_screen_log.jsonl")
_BUDGET_PATH  = os.path.join(_DATA_DIR, "ambient_vision_budget.json")

# Hard ceiling on buffered mic entries regardless of time-window cleanup, so
# a wedged cleanup pass can't let the JSON file grow without bound.
_HARD_ENTRY_CAP = 5000
# How long the worker can go without a heartbeat tick before status reports
# the thread as unhealthy (covers slow Whisper passes + GIL hiccups).
_HEARTBEAT_STALE_SECONDS = 60.0
# Per-JSONL hard cap; rotation triggers when exceeded.
_JSONL_HARD_CAP = 10_000
# Rough $-per-call estimate for the daily budget tracker. Local-VLM calls
# are effectively free but we charge a token to give the user visible
# rate-limiting; cloud falls would be much more expensive.
_VISION_COST_LOCAL_USD = 0.0001
_VISION_COST_CLOUD_USD = 0.02


# ── runtime state ────────────────────────────────────────────────────────
_lock = threading.RLock()

# Mic daemon (ambient-mode-1)
_thread:      Optional[threading.Thread] = None
_stop_evt = threading.Event()
_started_at:  Optional[float] = None
_heartbeat:   float = 0.0
_buffer: "deque[dict]" = deque()
_last_wake_at: float = 0.0
_last_error:  Optional[str] = None
_wake_pattern: Optional[re.Pattern] = None
# Wall-clock of the last tick on which JARVIS's own TTS playback was live.
# The mic worker samples _tts_playback_active every loop iteration; the wake
# nudge refuses to fire while playback is live or within _TTS_ECHO_COOLDOWN_S
# of it ending, so JARVIS saying his own name ("JARVIS online", a quoted
# reply) can't echo back through the mic and trip "I heard my name".
_tts_last_active: list[float] = [0.0]
_TTS_ECHO_COOLDOWN_S = 3.0

# System-audio daemon
_audio_thread: Optional[threading.Thread] = None
_audio_stop_evt = threading.Event()
_audio_started_at: Optional[float] = None
_audio_heartbeat: float = 0.0
_audio_entries_total: int = 0
_audio_last_error: Optional[str] = None

# Screen-snapshot daemon
_screen_thread: Optional[threading.Thread] = None
_screen_stop_evt = threading.Event()
_screen_started_at: Optional[float] = None
_screen_heartbeat: float = 0.0
_screen_entries_total: int = 0
_screen_skipped_total: int = 0
_screen_blocked_total: int = 0
_screen_last_phash: Optional[int] = None
_screen_last_error: Optional[str] = None

# Tray "Pause Daemons" toggle in bobert_companion sets this via set_paused()
# so the three workers idle (no Whisper, no VLM, no log writes) until the
# user flips the toggle off. Each loop checks it near the top of every tick.
# Streams stay open so we don't churn the audio device on every toggle.
_paused: list[bool] = [False]
# ─────────────────────────────────────────────────────────────────────────


def set_paused(p: bool) -> None:
    """Pause / resume all three ambient daemons in place. Called from
    bobert_companion._dispatch_tray_command on the pause_daemons toggle."""
    _paused[0] = bool(p)


def _ensure_project_on_path() -> None:
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)


def _voice_id_module():
    """Lazy-import core.voice_id and cache it on the function object so we
    don't re-attempt the import on every transcript. Returns the module or
    None if it can't be loaded (e.g. resemblyzer missing — gracefully
    falls back to single-user mode by leaving speaker_id null)."""
    cached = getattr(_voice_id_module, "_mod", "unset")
    if cached != "unset":
        return cached
    mod = None
    try:
        _ensure_project_on_path()
        from core import voice_id as _vid  # type: ignore
        mod = _vid
    except Exception as e:
        print(f"  [ambient-listen] voice_id unavailable: {e}")
        mod = None
    _voice_id_module._mod = mod
    return mod


def _identify_speaker_safe(audio: np.ndarray, sample_rate: int) -> tuple[Optional[str], float]:
    """Best-effort speaker ID for a transcript batch. Returns (None, 0.0)
    on any failure path — voice ID never blocks transcript persistence.
    Per-call latency is ~0.5-1 s on CPU; only invoked after the transcript
    has already cleared Whisper + validity gates so cost lands on real
    speech, not silence."""
    vid = _voice_id_module()
    if vid is None:
        return None, 0.0
    try:
        # Skip identify entirely when no one is enrolled — keeps single-user
        # mode latency identical to pre-voice-ID behaviour.
        if not vid.list_enrolled():
            return None, 0.0
        return vid.identify_speaker(audio, sample_rate)
    except Exception as e:
        print(f"  [ambient-listen] identify_speaker failed: {e}")
        return None, 0.0


def _apply_audio_processing(audio: np.ndarray,
                            sample_rate: int,
                            *,
                            enable_aec: bool = True,
                            enable_ns: bool = True,
                            enable_agc: bool = True,
                            record_mic_stats: bool = True) -> np.ndarray:
    """Best-effort: route a batch through core/audio_processor before it
    reaches Whisper. Falls back to the raw batch if the module isn't
    importable, the parent has disabled processing, or any layer fails.
    Loopback callers pass enable_aec=False — the loopback signal IS the
    speaker output, so echo cancellation would zero useful audio — and
    record_mic_stats=False so system-audio loudness doesn't pollute the
    mic-only silent-mic / stress stats (2026-07-14 bug-hunt #17)."""
    b = _get_bobert()
    if b is not None and not bool(getattr(b, "_audio_master_enabled", [True])[0]):
        return audio
    try:
        _ensure_project_on_path()
        from core import audio_processor as ap  # type: ignore
        proc = ap.get_processor(int(sample_rate))
        return proc.process(
            audio,
            enable_aec=enable_aec,
            enable_ns=enable_ns,
            enable_agc=enable_agc,
            record_mic_stats=record_mic_stats,
        )
    except Exception as e:
        print(f"  [ambient-listen] audio-proc fallthrough: {e}")
        return audio


def _ensure_data_dir() -> None:
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
    except Exception:
        pass


def _get_bobert():
    """Return the running bobert_companion module, or None if it hasn't
    been imported yet (e.g. unit-test context)."""
    return (sys.modules.get("bobert_companion")
            or sys.modules.get("__main__"))


def _set_ambient_stream_active(claim: bool) -> None:
    """Refcount bobert_companion._ambient_stream_active: +1 when this worker's
    dedicated ambient InputStream opens, -1 when it closes. _refresh_devices
    defers its destructive PortAudio sd._terminate()/_initialize() while the
    count is > 0, so a USB plug/unplug mid-capture can't tear PortAudio down
    under a live callback thread → 0xc0000374 heap corruption (HIGH, 2026-07-08).

    It MUST be a refcount, not a boolean: the mic worker and the WASAPI loopback
    worker are independent skills that can run CONCURRENTLY (room mic + system
    audio). With a shared boolean, whichever exits first cleared the guard while
    the other's stream was still live — re-opening the exact crash window this
    guards (bug-hunt 2026-07-08). Callers MUST pair every True with exactly one
    False (only on the code path that actually opened a stream). Guarded by
    bobert's _mic_lock so the ++/-- is atomic across workers. Best-effort: a
    missing module/flag/lock (unit tests) still adjusts the count when present."""
    bc = _get_bobert()
    if bc is None:
        return
    try:
        flag = getattr(bc, "_ambient_stream_active", None)
        if not isinstance(flag, list) or not flag:
            return
        lock = getattr(bc, "_mic_lock", None)
        delta = 1 if claim else -1
        if lock is not None:
            with lock:
                flag[0] = max(0, int(flag[0]) + delta)
        else:
            flag[0] = max(0, int(flag[0]) + delta)
    except Exception:
        pass


def _safe_close_stream(stream) -> None:
    """Tear down a sounddevice InputStream via bobert_companion._safe_close_stream
    when available, falling back to a daemon-thread close() pattern when bobert
    isn't loaded. Mirrors skills/enroll_voice.py.

    2026-05-29 silent-crash fix: `with sd.InputStream(...)` invokes close()
    unguarded on context exit, which has SIGSEGV'd at sounddevice.py:1167
    during exceptional unwinding. Route teardown through this guarded path so
    close() always runs on a daemon thread with a timeout."""
    if stream is None:
        return
    bc_close = None
    bc = _get_bobert()
    if bc is not None:
        bc_close = getattr(bc, "_safe_close_stream", None)
    if callable(bc_close):
        try:
            bc_close(stream)
        except Exception:
            pass
        return
    try:
        stream.stop()
    except Exception:
        pass
    done = threading.Event()
    def _close_in_daemon():
        try:
            stream.close()
        except Exception:
            pass
        finally:
            done.set()
    threading.Thread(target=_close_in_daemon, daemon=True).start()
    if not done.wait(timeout=2.0):
        # Escape hatch — mirrors bobert_companion._safe_close_stream. If the
        # daemon's close() hangs (PortAudio occasionally wedges on Windows),
        # force-stop every stream globally so the next open() doesn't inherit
        # a half-torn-down handle and SIGSEGV at sounddevice.py:1167.
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass


def _get_config(name: str, default):
    b = _get_bobert()
    if b is None:
        return default
    return getattr(b, name, default)


def _compile_wake_pattern() -> re.Pattern:
    """Build a case-insensitive regex from bobert_companion.WAKE_PHRASES
    with word-boundary guards on each side so 'jar visit' doesn't match
    'jarvis' and 'awakened' doesn't match 'wake'."""
    phrases = _get_config("WAKE_PHRASES", {"jarvis", "hey jarvis"})
    parts = []
    for p in phrases:
        p = (p or "").strip().lower()
        if not p:
            continue
        parts.append(re.escape(p))
    if not parts:
        parts.append(re.escape("jarvis"))
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def _wake_listener_active() -> bool:
    """True when skills/wake_listener.py has a running detector that owns
    the mic. We don't want to even attempt sd.InputStream open in that
    case — on Windows it'll raise PortAudioError."""
    wl = sys.modules.get("skill_wake_listener")
    if wl is None:
        return False
    det = getattr(wl, "_detector", None)
    if det is None:
        return False
    try:
        return bool(det.is_running())
    except Exception:
        return False


def _persist_state() -> None:
    """Atomically write the rolling mic buffer + multimodal status to
    ambient_listen_state.json. Called from inside _lock."""
    payload = {
        "started_at": _started_at,
        "last_persist_at": time.time(),
        "heartbeat": _heartbeat,
        "buffer_minutes": _get_config("AMBIENT_LISTEN_BUFFER_MINUTES", 10),
        "entries": list(_buffer),
        "audio_daemon": {
            "running": _audio_thread is not None and _audio_thread.is_alive(),
            "started_at": _audio_started_at,
            "heartbeat": _audio_heartbeat,
            "entries_total": _audio_entries_total,
            "last_error": _audio_last_error,
        },
        "screen_daemon": {
            "running": _screen_thread is not None and _screen_thread.is_alive(),
            "started_at": _screen_started_at,
            "heartbeat": _screen_heartbeat,
            "entries_total": _screen_entries_total,
            "skipped_total": _screen_skipped_total,
            "blocked_total": _screen_blocked_total,
            "last_error": _screen_last_error,
        },
    }
    try:
        fd, tmp = tempfile.mkstemp(dir=_PROJECT_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, _STATE_PATH)
        except Exception:
            try: os.unlink(tmp)
            except Exception: pass
            raise
    except Exception as e:
        print(f"  [ambient-listen] state persist failed: {e}")


def _trim_buffer(now: float) -> None:
    """Drop entries older than the configured window, then enforce the
    hard cap. Called from inside _lock."""
    window_s = max(60.0, float(_get_config("AMBIENT_LISTEN_BUFFER_MINUTES", 10)) * 60.0)
    cutoff = now - window_s
    while _buffer and _buffer[0].get("ts", 0.0) < cutoff:
        _buffer.popleft()
    while len(_buffer) > _HARD_ENTRY_CAP:
        _buffer.popleft()


def _tts_recently_active(b) -> bool:
    """True while JARVIS's own TTS playback is live, or within
    _TTS_ECHO_COOLDOWN_S of the last tick it was seen live. Echo
    cancellation is best-effort, so a transcript batch of JARVIS's own
    voice can land shortly AFTER playback ends — the cooldown covers
    the batching latency."""
    try:
        if bool(getattr(b, "_tts_playback_active", [False])[0]):
            _tts_last_active[0] = time.time()
            return True
    except Exception:
        pass
    return (time.time() - _tts_last_active[0]) < _TTS_ECHO_COOLDOWN_S


def _maybe_nudge_wake(text: str) -> None:
    """If text contains a wake phrase AND JARVIS is asleep / in standby,
    fire proactive_announce so the main loop greets the user on its next
    turn boundary. We debounce on _last_wake_at so a long monologue with
    multiple 'jarvis' mentions doesn't queue ten greetings.

    Two deliberate suppressions (2026-07-06 — the "he keeps saying I heard
    my name" bug):
      * AWAKE: when JARVIS is neither asleep nor in standby, the main
        listen loop hears the SAME audio this daemon tapped and is already
        handling the command — announcing "I heard my name" on top of that
        is pure noise, once per utterance that contains 'jarvis' (i.e.
        nearly every command). Log the match, say nothing.
      * OWN VOICE: skip while TTS playback is live or just ended, so
        JARVIS speaking his own name can't echo-trip the nudge.
    """
    global _last_wake_at
    if not _wake_pattern or not text:
        return
    if not _wake_pattern.search(text):
        return
    now = time.time()
    if now - _last_wake_at < 4.0:
        return

    b = _get_bobert()
    if b is None:
        return
    if _tts_recently_active(b):
        print(f"  [ambient-listen] wake match ignored (own TTS echo): {text!r}")
        return
    sleep_mode_list = getattr(b, "_sleep_mode", None)
    in_sleep = bool(sleep_mode_list and sleep_mode_list[0])
    standby_list = getattr(b, "_standby_mode", None)
    in_standby = bool(standby_list and standby_list[0])
    if not (in_sleep or in_standby):
        # Awake — the main loop already owns this utterance.
        print(f"  [ambient-listen] wake match (awake, no nudge): {text!r}")
        return
    _last_wake_at = now

    announce = getattr(b, "proactive_announce", None)
    msg = ("Yes, listening, sir." if in_sleep
           else "I heard my name, sir — go ahead.")
    if callable(announce):
        try:
            announce(msg, source="ambient-listen")
        except Exception:
            pass
    print(f"  [ambient-listen] wake match in: {text!r}")


# ── shared jsonl helper ──────────────────────────────────────────────────

def _append_jsonl(path: str, entry: dict) -> None:
    """Append one JSON object as a line. Caller is responsible for any
    rotation logic — this is a thin wrapper to keep the daemons readable."""
    _ensure_data_dir()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [ambient-listen] jsonl append failed ({path}): {e}")


def _rotate_jsonl_if_needed(path: str, cap: int = _JSONL_HARD_CAP) -> None:
    """Keep the last `cap` lines if the file grows past 1.5×cap. Cheap +
    Windows-safe (single open, single rename)."""
    try:
        if not os.path.exists(path):
            return
        # Cheap line count without slurping the whole file
        with open(path, "rb") as f:
            n_lines = sum(1 for _ in f)
        if n_lines < int(cap * 1.5):
            return
        with open(path, "r", encoding="utf-8") as f:
            tail = f.readlines()[-cap:]
        fd, tmp = tempfile.mkstemp(dir=_DATA_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.writelines(tail)
            os.replace(tmp, path)
        except Exception:
            try: os.unlink(tmp)
            except Exception: pass
            raise
        print(f"  [ambient-listen] rotated {path} -> {len(tail)} lines")
    except Exception as e:
        print(f"  [ambient-listen] rotation failed for {path}: {e}")


# ── focused-window helpers (Windows; fall back to '' elsewhere) ──────────

def _focused_window_title() -> str:
    """Return the current foreground-window title, or '' if it can't be
    queried. Used both for transcript attribution and for the screen
    blocklist check."""
    try:
        if sys.platform != "win32":
            return ""
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value or ""
    except Exception:
        return ""


def _focused_proc_name() -> str:
    """Return foreground-window process name (e.g. 'chrome.exe'), or ''."""
    try:
        if sys.platform != "win32":
            return ""
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(260)
            psapi.GetModuleBaseNameW(h, None, buf, 260)
            return (buf.value or "").lower()
        finally:
            kernel32.CloseHandle(h)
    except Exception:
        return ""


# ── default blocklist + matcher ──────────────────────────────────────────

_DEFAULT_SCREEN_BLOCKLIST = (
    r"(?i)\b1password\b",
    r"(?i)\bbitwarden\b",
    r"(?i)\bkeepass\b",
    r"(?i)\blastpass\b",
    r"(?i)\bdashlane\b",
    r"(?i)\bbanking\b",
    r"(?i)\bchase\.com\b",
    r"(?i)\bcapitalone\b",
    r"(?i)\bbankofamerica\b",
    r"(?i)\bwellsfargo\b",
    r"(?i)\b(visa|mastercard)\.com\b",
    r"(?i)\bpaypal\.com\b",
    r"(?i)\bvenmo\.com\b",
    r"(?i)\bcoinbase\b",
    r"(?i)\bcredit\s*card\b",
    # Generic auth screens
    r"(?i)\b(sign\s*in|log\s*in|login).*(password|2fa|otp)\b",
    r"(?i)\bauthenticator\b",
    r"(?i)\bone\s*time\s*passcode\b",
    r"(?i)\bsocial\s*security\b",
    r"(?i)\bssn\b",
)


def _compile_blocklist() -> list[re.Pattern]:
    extra = _get_config("AMBIENT_SCREEN_BLOCKLIST", ())
    patterns: list[re.Pattern] = []
    for src in tuple(_DEFAULT_SCREEN_BLOCKLIST) + tuple(extra or ()):
        try:
            patterns.append(re.compile(src))
        except re.error as e:
            print(f"  [ambient-listen] bad blocklist regex {src!r}: {e}")
    return patterns


def _is_sensitive_window(title: str, proc: str, blocklist: list[re.Pattern]) -> bool:
    blob = f"{title} {proc}"
    return any(p.search(blob) for p in blocklist)


# ── vision budget tracker ────────────────────────────────────────────────

def _budget_today_key() -> str:
    return time.strftime("%Y-%m-%d")


def _load_budget() -> dict:
    _ensure_data_dir()
    try:
        with open(_BUDGET_PATH, encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return {}
        return d
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save_budget(d: dict) -> None:
    _ensure_data_dir()
    try:
        fd, tmp = tempfile.mkstemp(dir=_DATA_DIR, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, _BUDGET_PATH)
    except Exception as e:
        print(f"  [ambient-listen] budget save failed: {e}")


def _vision_budget_remaining() -> float:
    cap = float(_get_config("AMBIENT_VISION_BUDGET_USD", 1.0))
    d = _load_budget()
    spent = float(d.get(_budget_today_key(), 0.0))
    return max(0.0, cap - spent)


def _vision_budget_charge(cost: float) -> None:
    d = _load_budget()
    k = _budget_today_key()
    d[k] = float(d.get(k, 0.0)) + float(cost)
    # Keep only the last 14 days
    keep = sorted(d.keys())[-14:]
    d = {k: d[k] for k in keep}
    _save_budget(d)


# ── perceptual hash for screen dedupe ────────────────────────────────────

def _phash64(png_bytes: bytes) -> Optional[int]:
    """Cheap 64-bit average-hash. Lets us skip the VLM when the screen
    barely changed between sampling ticks (e.g. you're reading a doc)."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(png_bytes)).convert("L").resize((8, 8))
        pixels = list(im.getdata())
        avg = sum(pixels) / 64.0
        bits = 0
        for i, p in enumerate(pixels):
            if p >= avg:
                bits |= (1 << i)
        return bits
    except Exception:
        return None


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ── mic worker (ambient-mode-1, unchanged behaviour) ─────────────────────

def _worker_loop() -> None:
    """Background mic thread: open stream, batch chunks, transcribe."""
    global _heartbeat, _last_error

    _ensure_project_on_path()
    try:
        import sounddevice as sd
    except Exception as e:
        _last_error = f"sounddevice import failed: {e}"
        print(f"  [ambient-listen] {_last_error}")
        return

    b = _get_bobert()
    if b is None:
        _last_error = "bobert_companion not loaded"
        print(f"  [ambient-listen] {_last_error}")
        return

    sample_rate = int(getattr(b, "SAMPLE_RATE", 16000))
    chunk_secs  = float(_get_config("AMBIENT_LISTEN_CHUNK_DURATION_SECONDS", 0.5))
    chunk_secs  = max(0.1, min(2.0, chunk_secs))
    batch_secs  = 2.5
    blocksize   = max(256, int(sample_rate * chunk_secs))

    get_input_device = getattr(b, "get_input_device", lambda: None)
    transcribe       = getattr(b, "transcribe", None)
    is_valid_speech  = getattr(b, "is_valid_speech", None)
    is_ambient_music = getattr(b, "is_ambient_music", lambda _t: False)
    if not callable(transcribe):
        _last_error = "bobert_companion.transcribe is not callable"
        print(f"  [ambient-listen] {_last_error}")
        return

    try:
        device = get_input_device()
    except Exception:
        device = None

    audio_q: "deque[np.ndarray]" = deque()
    q_lock = threading.Lock()

    def _audio_cb(indata, frames, time_info, status):
        if _stop_evt.is_set():
            return
        try:
            chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        except Exception:
            return
        with q_lock:
            audio_q.append(chunk)

    # Never open a capture stream when the mic is hard-disabled (staging green
    # candidate / MICROPHONE_INDEX < 0): device would be None, and InputStream
    # treats None as the SYSTEM DEFAULT mic — i.e. it would listen anyway.
    if getattr(b, "_mic_input_disabled", lambda: False)():
        _last_error = ("mic disabled (staging / MICROPHONE_INDEX < 0) — "
                       "ambient capture not started")
        print(f"  [ambient-listen] {_last_error}")
        return

    # ── PREFERRED PATH: share the main loop's mic via the record_speech tap ──
    # bobert_companion.record_speech is the SOLE owner of the input device; it
    # fans every captured frame out to registered taps (add_record_tap /
    # _fanout_record_frame). Opening a SECOND sd.InputStream on the same device
    # is exactly the WASAPI contention the host warns about — on Windows it
    # either fails outright or starves record_speech so the main loop goes deaf
    # to the wake word "JARVIS" (the user-reported "go ambient → won't wake"
    # bug). So we register a TAP and consume the host's frames instead, leaving
    # the wake/standby path fully functional. We only fall back to our own
    # stream when the host doesn't expose the tap API (older monolith) AND no
    # wake-word listener owns the device.
    #
    # The tap is a queue.Queue; a tiny shim drains it into the same audio_q the
    # batching loop below reads, so the rest of the worker is unchanged whether
    # we're tapping or running our own callback stream.
    tap_q = None
    tap_drain_stop = None
    tap_drain_thread = None
    add_tap = getattr(b, "add_record_tap", None)
    remove_tap = getattr(b, "remove_record_tap", None)
    stream = None
    claimed_ambient = False   # True only once THIS worker refcounts the ambient guard
    if callable(add_tap) and callable(remove_tap):
        import queue as _queue
        tap_q = _queue.Queue()
        try:
            add_tap(tap_q)   # returns whether record_speech is live right now;
        except Exception as e:  # we register regardless so we catch it the moment it next opens.
            _last_error = f"add_record_tap failed: {e}"
            print(f"  [ambient-listen] {_last_error}")
            return

        def _drain_tap():
            # Pull frames the host captured and hand them to the batch loop.
            while tap_drain_stop is not None and not tap_drain_stop.is_set():
                try:
                    frame = tap_q.get(timeout=0.2)
                except Exception:
                    continue
                if frame is None:
                    continue
                try:
                    chunk = (frame[:, 0].copy() if getattr(frame, "ndim", 1) > 1
                             else frame.copy())
                except Exception:
                    continue
                with q_lock:
                    audio_q.append(chunk)

        tap_drain_stop = threading.Event()
        tap_drain_thread = threading.Thread(
            target=_drain_tap, name="ambient-listen-mic-tap", daemon=True)
        tap_drain_thread.start()
        print(f"  [ambient-listen] sharing the main-loop mic via record_speech "
              f"tap (no competing stream), sample_rate={sample_rate}, "
              f"chunk={chunk_secs:.2f}s")
    else:
        # ── FALLBACK PATH: dedicated InputStream (host has no tap API) ──────
        # TOCTOU FIX (2026-07-14 audit, 0xc0000374): publish mic ownership
        # BEFORE the stream is opened+started, never after. The old order
        # (start() → _set_ambient_stream_active(True)) left a window in which
        # the callback thread was ALREADY LIVE while the guard still read 0 —
        # and _refresh_devices only defers on that flag. A concurrent
        # get_input_device() in that gap runs sd._terminate()/sd._initialize(),
        # freeing PortAudio out from under the live callback and heap-corrupting
        # the process (a silent, traceless kill). This is the exact ordering
        # bug already fixed for record_speech in bobert_companion; both ambient
        # workers still had the pre-fix order. Claim first, and release on
        # EVERY failure path so a failed open can't strand the refcount.
        _set_ambient_stream_active(True)
        claimed_ambient = True
        try:
            stream = sd.InputStream(
                samplerate=sample_rate, channels=1, dtype="float32",
                blocksize=blocksize, device=device, callback=_audio_cb)
        except Exception as e:
            _set_ambient_stream_active(False)
            claimed_ambient = False
            if _wake_listener_active():
                _last_error = ("mic locked by wake-word listener — stop "
                               "the wake-word listener first")
            else:
                _last_error = f"InputStream open failed: {e}"
            print(f"  [ambient-listen] {_last_error}")
            return

        # 2026-05-29 silent-crash fix: don't use `with stream:` — the implicit
        # __exit__ calls sd close() unguarded, which has SIGSEGV'd at
        # sounddevice.py:1167. Start explicitly and route teardown through
        # _safe_close_stream so close() runs on a daemon thread with a timeout.
        try:
            stream.start()
        except Exception as e:
            _last_error = f"InputStream.start failed: {e}"
            print(f"  [ambient-listen] {_last_error}")
            _safe_close_stream(stream)
            _set_ambient_stream_active(False)
            claimed_ambient = False
            return
        # (ownership was claimed BEFORE the open — see the TOCTOU note above;
        # the finally still decrements it, but only when claimed_ambient is
        # True, so the tap path never under-counts.)

        print(f"  [ambient-listen] stream open on device={device}, "
              f"sample_rate={sample_rate}, chunk={chunk_secs:.2f}s")

    samples_per_batch = int(sample_rate * batch_secs)
    pending: list[np.ndarray] = []
    pending_samples = 0

    try:
        while not _stop_evt.is_set():
            # Sample own-voice playback every tick (~0.1 s) so the wake
            # nudge's echo cooldown window starts when playback ENDS, not
            # when the (2.5 s-batched) echo transcript finally lands.
            _tts_recently_active(b)
            # Daemon-pause guard: drop accumulated chunks so we don't
            # process a wall-of-audio surge when the user un-pauses.
            # Stream stays open to keep the audio device warm.
            if _paused[0]:
                with q_lock:
                    audio_q.clear()
                pending.clear()
                pending_samples = 0
                _heartbeat = time.time()
                if _stop_evt.wait(0.5):
                    break
                continue

            with q_lock:
                while audio_q:
                    chunk = audio_q.popleft()
                    pending.append(chunk)
                    pending_samples += chunk.shape[0]

            if pending_samples < samples_per_batch:
                _heartbeat = time.time()
                if _stop_evt.wait(0.1):
                    break
                continue

            audio = np.concatenate(pending).astype(np.float32, copy=False)
            pending.clear()
            pending_samples = 0

            # Gate on the RAW RMS — the noise-cancel-1 processor's
            # AGC could otherwise inflate near-silent batches above
            # the 0.003 floor and overwhelm Whisper with hum.
            rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
            _heartbeat = time.time()
            if rms < 0.003:
                continue

            # Apply the three-layer cleanup (AEC → NS → AGC) so
            # whatever Whisper sees has JARVIS's own playback,
            # stationary background noise, and gain drift removed.
            audio = _apply_audio_processing(audio, sample_rate)

            try:
                text, conf = transcribe(audio)
            except Exception as e:
                _last_error = f"transcribe failed: {e}"
                print(f"  [ambient-listen] {_last_error}")
                continue

            if not text:
                continue
            if is_ambient_music(text):
                continue
            if callable(is_valid_speech):
                ok, _reason = is_valid_speech(text, conf, peak_rms=rms)
                if not ok:
                    continue

            # Identify the speaker for this batch BEFORE persisting so
            # downstream consumers (anticipation, banter, per-speaker
            # routing) see a stable speaker_id on every transcript. The
            # call is best-effort: None when resemblyzer is missing, no
            # one is enrolled, or the embedding doesn't clear the
            # confidence threshold — single-user mode stays untouched.
            speaker_id, speaker_score = _identify_speaker_safe(audio, sample_rate)

            entry = {
                "ts": time.time(),
                "text": text,
                "no_speech_prob": float(conf.get("no_speech_prob", 1.0)),
                "avg_logprob":    float(conf.get("avg_logprob", -10.0)),
                "rms":            rms,
                "speaker_id":     speaker_id,
                "speaker_score":  float(speaker_score),
            }
            with _lock:
                _buffer.append(entry)
                _trim_buffer(entry["ts"])
                _persist_state()
            # Mirror the mic transcript into the multimodal log too so
            # the extractor sees all three streams in one place.
            mirror = dict(entry)
            mirror["source"] = "mic"
            mirror["window"] = _focused_window_title()
            _append_jsonl(_AUDIO_JSONL, mirror)
            _rotate_jsonl_if_needed(_AUDIO_JSONL)
            _maybe_nudge_wake(text)
    except Exception as e:
        _last_error = f"worker crashed: {e}"
        print(f"  [ambient-listen] {_last_error}")
    finally:
        # Tap path: detach from record_speech's fan-out and stop the drain
        # thread. Stream path: close our dedicated InputStream. Exactly one of
        # these is live depending on which acquisition branch ran above.
        if tap_drain_stop is not None:
            tap_drain_stop.set()
        if tap_q is not None and callable(remove_tap):
            try:
                remove_tap(tap_q)
            except Exception:
                pass
        if stream is not None:
            _safe_close_stream(stream)
        # Release device ownership ONLY if we claimed it (the tap path never did).
        # Close the stream FIRST so the refcount never drops while our own stream
        # is still tearing down.
        if claimed_ambient:
            _set_ambient_stream_active(False)
        print("  [ambient-listen] mic worker exiting")


# ── system-audio (WASAPI loopback) worker ────────────────────────────────

def _find_loopback_device(sd) -> Optional[int]:
    """Pick a WASAPI loopback device matching the default output.
    sounddevice exposes the loopback flag on Windows via WASAPI; we look
    for a host-API named 'Windows WASAPI' and an input device whose name
    matches the default output (Windows reports loopback inputs there)."""
    try:
        hostapis = sd.query_hostapis()
        wasapi_idx = None
        for i, h in enumerate(hostapis):
            if "wasapi" in (h.get("name") or "").lower():
                wasapi_idx = i
                break
        if wasapi_idx is None:
            return None
        default_out_idx = hostapis[wasapi_idx].get("default_output_device", -1)
        default_out_name = ""
        if default_out_idx is not None and default_out_idx >= 0:
            default_out_name = (sd.query_devices(default_out_idx).get("name") or "").lower()
        # Prefer a device whose name matches the default output AND can do input.
        devs = sd.query_devices()
        # Some drivers expose explicit "(loopback)" entries.
        for idx, d in enumerate(devs):
            if d.get("hostapi") != wasapi_idx:
                continue
            name = (d.get("name") or "").lower()
            if "loopback" in name and d.get("max_input_channels", 0) > 0:
                return idx
        if default_out_name:
            for idx, d in enumerate(devs):
                if d.get("hostapi") != wasapi_idx:
                    continue
                if d.get("max_input_channels", 0) <= 0:
                    continue
                name = (d.get("name") or "").lower()
                if name == default_out_name or default_out_name in name:
                    return idx
        return None
    except Exception:
        return None


def _audio_worker_loop() -> None:
    """WASAPI loopback worker: capture system audio output, Whisper it,
    append to data/ambient_transcripts.jsonl tagged source='system_audio'."""
    global _audio_heartbeat, _audio_last_error, _audio_entries_total

    _ensure_project_on_path()
    try:
        import sounddevice as sd
    except Exception as e:
        _audio_last_error = f"sounddevice import failed: {e}"
        print(f"  [ambient-audio] {_audio_last_error}")
        return

    b = _get_bobert()
    if b is None:
        _audio_last_error = "bobert_companion not loaded"
        return

    transcribe       = getattr(b, "transcribe", None)
    is_valid_speech  = getattr(b, "is_valid_speech", None)
    is_ambient_music = getattr(b, "is_ambient_music", lambda _t: False)
    if not callable(transcribe):
        _audio_last_error = "bobert_companion.transcribe is not callable"
        return

    sample_rate = int(getattr(b, "SAMPLE_RATE", 16000))
    batch_secs  = float(_get_config("AMBIENT_AUDIO_CHUNK_DURATION_SECONDS", 30.0))
    batch_secs  = max(5.0, min(120.0, batch_secs))
    blocksize   = max(1024, sample_rate // 4)

    device_idx = _find_loopback_device(sd)
    if device_idx is None:
        _audio_last_error = ("no WASAPI loopback device found — system audio "
                             "capture is Windows-only")
        print(f"  [ambient-audio] {_audio_last_error}")
        return

    # Read native device rate so resampling math is correct. Default dev_info
    # to {} first so the open-banner below stays safe when query_devices()
    # raises — otherwise dev_info is unbound and the UnboundLocalError escapes
    # before the main try/except, silently killing the audio daemon thread.
    dev_info: dict = {}
    try:
        dev_info = sd.query_devices(device_idx)
        dev_sr = int(dev_info.get("default_samplerate") or sample_rate)
        in_channels = max(1, int(dev_info.get("max_input_channels") or 1))
    except Exception:
        dev_sr = sample_rate
        in_channels = 1

    # Build WASAPI loopback extra settings if available.
    extra_settings = None
    try:
        WasapiSettings = getattr(sd, "WasapiSettings", None)
        if WasapiSettings is not None:
            extra_settings = WasapiSettings(loopback=True)
    except Exception:
        extra_settings = None

    audio_q: "deque[np.ndarray]" = deque()
    q_lock = threading.Lock()

    def _cb(indata, frames, time_info, status):
        if _audio_stop_evt.is_set():
            return
        try:
            if indata.ndim > 1:
                chunk = indata.mean(axis=1).astype(np.float32, copy=False)
            else:
                chunk = indata.astype(np.float32, copy=False).copy()
        except Exception:
            return
        with q_lock:
            audio_q.append(chunk)

    # TOCTOU FIX (2026-07-14 audit, 0xc0000374): claim device ownership BEFORE
    # the stream is opened+started. The old order set the flag AFTER start(),
    # leaving a window where the WASAPI loopback callback was already live while
    # _refresh_devices' guard still read 0 — a concurrent device refresh in that
    # gap runs sd._terminate()/sd._initialize() under the live callback and
    # heap-corrupts the process. Same bug record_speech was fixed for; both
    # ambient workers still had the pre-fix order. Released on every failure path.
    _set_ambient_stream_active(True)
    try:
        stream = sd.InputStream(
            samplerate=dev_sr,
            channels=in_channels,
            dtype="float32",
            blocksize=blocksize,
            device=device_idx,
            callback=_cb,
            extra_settings=extra_settings,
        )
    except Exception as e:
        _set_ambient_stream_active(False)
        _audio_last_error = f"loopback open failed: {e}"
        print(f"  [ambient-audio] {_audio_last_error}")
        return

    # 2026-05-29 silent-crash fix: don't use `with stream:` — the implicit
    # __exit__ calls sd close() unguarded, which has SIGSEGV'd at
    # sounddevice.py:1167. Start explicitly and route teardown through
    # _safe_close_stream so the WASAPI loopback handle is relinquished on a
    # daemon thread with a timeout instead of the caller thread.
    try:
        stream.start()
    except Exception as e:
        _audio_last_error = f"loopback start failed: {e}"
        print(f"  [ambient-audio] {_audio_last_error}")
        _safe_close_stream(stream)
        _set_ambient_stream_active(False)
        return
    # (ownership was claimed BEFORE the open — see the TOCTOU note above; the
    # finally below still clears it.)

    print(f"  [ambient-audio] loopback open on device #{device_idx} "
          f"({dev_info.get('name', '?')}), native_sr={dev_sr}, "
          f"channels={in_channels}, batch={batch_secs:.0f}s")

    samples_per_batch = int(dev_sr * batch_secs)
    pending: list[np.ndarray] = []
    pending_samples = 0

    try:
        while not _audio_stop_evt.is_set():
            # Daemon-pause guard mirrors the mic worker.
            if _paused[0]:
                with q_lock:
                    audio_q.clear()
                pending.clear()
                pending_samples = 0
                _audio_heartbeat = time.time()
                if _audio_stop_evt.wait(1.0):
                    break
                continue

            with q_lock:
                while audio_q:
                    chunk = audio_q.popleft()
                    pending.append(chunk)
                    pending_samples += chunk.shape[0]

            if pending_samples < samples_per_batch:
                _audio_heartbeat = time.time()
                if _audio_stop_evt.wait(0.5):
                    break
                continue

            audio = np.concatenate(pending).astype(np.float32, copy=False)
            pending.clear()
            pending_samples = 0

            rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
            _audio_heartbeat = time.time()
            # System audio gate is tighter than mic — we don't want to
            # Whisper near-silent menu hum or a paused video.
            if rms < 0.005:
                continue

            # Resample to Whisper's 16 kHz if the loopback ran natively
            # at 44.1/48 kHz. Cheap linear resample keeps deps light.
            if dev_sr != sample_rate:
                n_out = int(round(audio.shape[0] * sample_rate / dev_sr))
                if n_out <= 0:  # pragma: no cover - unreachable: the batch gate guarantees audio.shape[0] >= dev_sr*batch_secs, so n_out ~= sample_rate*batch_secs >= 5 (see test NOTE)
                    continue
                x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
                audio = np.interp(x_new, x_old, audio).astype(np.float32, copy=False)

            # Skip AEC, AGC, and NS here — the shared AudioProcessor
            # singleton is also used by the mic capture path, and loud/
            # quiet loopback chunks would poison its AGC running_rms and
            # NS noise spectrum, degrading mic transcription. Whisper
            # handles its own normalization for loopback audio.
            audio = _apply_audio_processing(audio, sample_rate,
                                            enable_aec=False,
                                            enable_agc=False,
                                            enable_ns=False,
                                            record_mic_stats=False)

            try:
                text, conf = transcribe(audio)
            except Exception as e:
                _audio_last_error = f"transcribe failed: {e}"
                print(f"  [ambient-audio] {_audio_last_error}")
                continue

            if not text:
                continue
            if is_ambient_music(text):
                continue
            # Re-use the mic's hallucination gate but skip the wake-word
            # check — system audio is allowed to be free-form.
            if callable(is_valid_speech):
                ok, _reason = is_valid_speech(text, conf, peak_rms=rms)
                if not ok:
                    continue

            # Loopback audio sometimes carries a recognisable speaker
            # too (a Teams call, a YouTube clip of a known voice) — try
            # to ID them so downstream behaviour (e.g. "your mother is
            # on a call") has the same speaker_id schema as mic entries.
            speaker_id, speaker_score = _identify_speaker_safe(audio, sample_rate)

            entry = {
                "ts": time.time(),
                "source": "system_audio",
                "text": text,
                "no_speech_prob": float(conf.get("no_speech_prob", 1.0)),
                "avg_logprob":    float(conf.get("avg_logprob", -10.0)),
                "rms": rms,
                "window": _focused_window_title(),
                "proc":   _focused_proc_name(),
                "speaker_id":    speaker_id,
                "speaker_score": float(speaker_score),
            }
            _append_jsonl(_AUDIO_JSONL, entry)
            _rotate_jsonl_if_needed(_AUDIO_JSONL)
            with _lock:
                _audio_entries_total += 1
                _persist_state()
    except Exception as e:
        _audio_last_error = f"audio worker crashed: {e}"
        print(f"  [ambient-audio] {_audio_last_error}")
    finally:
        # Close the loopback stream BEFORE releasing the refcount, so the guard
        # never drops (allowing a PortAudio reinit) while our own stream is still
        # tearing down — matches the mic worker's close-then-release order. The
        # loopback worker always claims (unconditional set above), so it always
        # releases. 2026-07-08.
        _safe_close_stream(stream)
        _set_ambient_stream_active(False)
        print("  [ambient-audio] worker exiting")


# ── screen-snapshot worker ───────────────────────────────────────────────

def _summarize_screen_via_vlm(png_bytes: bytes) -> Optional[dict]:
    """Ask the local VLM to summarise the screen, extract entities, and
    flag sensitive content. Returns a structured dict, or None if the VLM
    isn't available. Falls back to cloud only when the budget allows."""
    b = _get_bobert()
    if b is None:
        return None

    prompt = (
        "You are a privacy-aware screen observer. Look at the screenshot "
        "and reply with ONLY a compact JSON object in this exact shape:\n"
        '{"summary": "...", "entities": ["..."], "sensitive": false, '
        '"sensitive_reason": ""}\n\n'
        "Rules:\n"
        "- summary: ONE sentence (max 40 words) describing what is visible "
        "(app, page topic, what the user appears to be doing).\n"
        "- entities: short list of named things mentioned/visible (people, "
        "projects, products, code symbols, URLs). Empty list if none.\n"
        "- sensitive: true if you see passwords, credit-card / SSN / bank "
        "details, authenticator codes, or any private credentials.\n"
        "- sensitive_reason: short phrase if sensitive=true; '' otherwise.\n"
        "If you cannot tell what is on screen, return summary='' and "
        "sensitive=false. Output ONLY the JSON object, no prose."
    )

    text = None
    try:
        call_local = getattr(b, "_call_local_vision", None)
        if callable(call_local):
            text = call_local(prompt, [png_bytes], max_tokens=400)
            if text:
                _vision_budget_charge(_VISION_COST_LOCAL_USD)
    except Exception as e:
        print(f"  [ambient-screen] local VLM call failed: {e}")
        text = None

    if not text:
        # Cloud fall-through only if budget remains. Refuse if budget cap hit.
        if _vision_budget_remaining() < _VISION_COST_CLOUD_USD:
            print("  [ambient-screen] vision budget exhausted, skipping cloud fallback")
            return None
        try:
            ask = getattr(b, "ask_vision", None)
            if callable(ask):
                text = ask(prompt, png_bytes)
                if text:
                    _vision_budget_charge(_VISION_COST_CLOUD_USD)
        except Exception as e:
            print(f"  [ambient-screen] cloud vision call failed: {e}")
            text = None

    if not text:
        return None

    # Parse the JSON object out of the reply (the model occasionally wraps
    # it in code fences or prefixes a `[local-vision]` tag).
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"summary": text.strip()[:500], "entities": [],
                "sensitive": False, "sensitive_reason": "",
                "raw": True}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {"summary": text.strip()[:500], "entities": [],
                "sensitive": False, "sensitive_reason": "",
                "raw": True}

    return {
        "summary": str(obj.get("summary", ""))[:600],
        "entities": [str(x)[:80] for x in (obj.get("entities") or [])][:20],
        "sensitive": bool(obj.get("sensitive", False)),
        "sensitive_reason": str(obj.get("sensitive_reason", ""))[:200],
    }


def _screen_worker_loop() -> None:
    """Periodic screen-snapshot daemon. Honours blocklist, dedupes via
    pHash, and respects the daily vision budget."""
    global _screen_heartbeat, _screen_last_error
    global _screen_entries_total, _screen_skipped_total, _screen_blocked_total
    global _screen_last_phash

    _ensure_project_on_path()
    b = _get_bobert()
    if b is None:
        _screen_last_error = "bobert_companion not loaded"
        return

    take_all = getattr(b, "take_all_monitor_screenshots", None)
    take_one = getattr(b, "take_screenshot", None)
    if not (callable(take_all) or callable(take_one)):
        _screen_last_error = "no screenshot helper available"
        return

    blocklist = _compile_blocklist()
    interval = float(_get_config("AMBIENT_SCREEN_INTERVAL_S", 60.0))
    interval = max(15.0, min(600.0, interval))

    print(f"  [ambient-screen] daemon online, interval={interval:.0f}s, "
          f"blocklist={len(blocklist)} patterns, "
          f"budget=${_vision_budget_remaining():.4f}")

    while not _screen_stop_evt.is_set():
        _screen_heartbeat = time.time()
        # Daemon-pause guard — skip the capture + VLM call entirely so the
        # daily vision budget isn't consumed while paused.
        if _paused[0]:
            if _screen_stop_evt.wait(interval):
                break
            continue
        loop_started = time.time()

        title = _focused_window_title()
        proc  = _focused_proc_name()

        if _is_sensitive_window(title, proc, blocklist):
            with _lock:
                _screen_blocked_total += 1
                _persist_state()
            print(f"  [ambient-screen] BLOCKED (focused window matches "
                  f"blocklist: {title!r} / {proc!r})")
            if _screen_stop_evt.wait(interval):
                break
            continue

        # Budget guard before we even capture — keeps cost predictable.
        if _vision_budget_remaining() <= 0.0:
            with _lock:
                _screen_skipped_total += 1
                _persist_state()
            print("  [ambient-screen] daily vision budget exhausted, sleeping")
            if _screen_stop_evt.wait(interval):
                break
            continue

        # Prefer all-monitor capture so a multi-monitor user gets the full
        # context; fall back to single-monitor if the helper is missing.
        png_bytes: Optional[bytes] = None
        try:
            if callable(take_all):
                shots = take_all()
                if shots:
                    # Pick the largest of the captures as the "primary" frame
                    # for hashing + summarisation. Sending all monitors costs
                    # 4× tokens and we're optimising for budget.
                    name = max(shots, key=lambda n: len(shots[n]))
                    png_bytes = shots[name]
            elif callable(take_one):
                png_bytes = take_one(None)
        except Exception as e:
            _screen_last_error = f"screenshot capture failed: {e}"
            print(f"  [ambient-screen] {_screen_last_error}")
            if _screen_stop_evt.wait(interval):
                break
            continue

        if not png_bytes:
            with _lock:
                _screen_skipped_total += 1
                _persist_state()
            if _screen_stop_evt.wait(interval):
                break
            continue

        # Dedupe against the previous tick via 64-bit average hash.
        h = _phash64(png_bytes)
        if h is not None and _screen_last_phash is not None:
            if _hamming(h, _screen_last_phash) <= 4:
                with _lock:
                    _screen_skipped_total += 1
                    _persist_state()
                _screen_last_phash = h
                # Compensate for time spent capturing + hashing.
                wait = max(1.0, interval - (time.time() - loop_started))
                if _screen_stop_evt.wait(wait):
                    break
                continue
        _screen_last_phash = h

        result = _summarize_screen_via_vlm(png_bytes)
        if not result:
            with _lock:
                _screen_skipped_total += 1
                _persist_state()
            wait = max(1.0, interval - (time.time() - loop_started))
            if _screen_stop_evt.wait(wait):
                break
            continue

        # Hard redaction: if the VLM itself flags sensitive data, scrub the
        # text fields before persisting. We still keep a one-line audit
        # entry so the user can see we filtered something.
        if result.get("sensitive"):
            redacted_reason = result.get("sensitive_reason") or "model-flagged sensitive"
            result["summary"] = "[redacted: sensitive on-screen content]"
            result["entities"] = []
            result["sensitive_reason"] = redacted_reason

        entry = {
            "ts": time.time(),
            "source": "screen",
            "window": title,
            "proc": proc,
            **result,
            "budget_remaining_usd": round(_vision_budget_remaining(), 6),
        }
        _append_jsonl(_SCREEN_JSONL, entry)
        _rotate_jsonl_if_needed(_SCREEN_JSONL)
        with _lock:
            _screen_entries_total += 1
            _persist_state()

        wait = max(1.0, interval - (time.time() - loop_started))
        if _screen_stop_evt.wait(wait):
            break

    print("  [ambient-screen] daemon exiting")


# ── action handlers ──────────────────────────────────────────────────────

def ambient_listen_start(_: str = "") -> str:
    """Open the dedicated mic stream and begin passive transcription."""
    global _thread, _started_at, _stop_evt, _wake_pattern, _last_error, _heartbeat
    with _lock:
        if _thread is not None and _thread.is_alive():
            return "Ambient listening is already active, sir."
        if _wake_listener_active():
            return ("Ambient mode requires an exclusive mic connection — "
                    "stop the wake-word listener first, sir.")
        _stop_evt = threading.Event()
        _wake_pattern = _compile_wake_pattern()
        _started_at = time.time()
        _heartbeat = _started_at
        _last_error = None
        _thread = threading.Thread(
            target=_worker_loop, name="ambient-listen", daemon=True)
        _thread.start()
    time.sleep(0.4)
    with _lock:
        if _last_error and (_thread is None or not _thread.is_alive()):
            err = _last_error
            _started_at = None
            return f"Ambient mode failed to start, sir: {err}."
        mins = _get_config("AMBIENT_LISTEN_BUFFER_MINUTES", 10)
        return (f"Ambient listening engaged, sir. I'll keep a "
                f"{mins}-minute rolling transcript and stay silent unless "
                "I hear my name.")


def ambient_listen_stop(_: str = "") -> str:
    """Close the stream and stop transcribing. Buffer is preserved."""
    global _thread, _started_at
    with _lock:
        if _thread is None or not _thread.is_alive():
            return "Ambient listening is not running, sir."
        _stop_evt.set()
        t = _thread
    t.join(timeout=3.0)
    with _lock:
        if t.is_alive():
            return "Ambient listener did not stop cleanly within 3 s, sir."
        _thread = None
        elapsed = (time.time() - _started_at) if _started_at else 0.0
        _started_at = None
        entries = len(_buffer)
        _persist_state()
    return (f"Ambient listening disengaged, sir. {entries} entries "
            f"captured over {int(elapsed)}s.")


def ambient_audio_start(_: str = "") -> str:
    """Open WASAPI loopback and begin transcribing system audio output."""
    global _audio_thread, _audio_stop_evt, _audio_started_at
    global _audio_heartbeat, _audio_last_error
    if sys.platform != "win32":
        return "System-audio capture is Windows-only, sir — WASAPI loopback required."
    with _lock:
        if _audio_thread is not None and _audio_thread.is_alive():
            return "System-audio capture is already active, sir."
        _audio_stop_evt = threading.Event()
        _audio_started_at = time.time()
        _audio_heartbeat = _audio_started_at
        _audio_last_error = None
        _audio_thread = threading.Thread(
            target=_audio_worker_loop, name="ambient-audio", daemon=True)
        _audio_thread.start()
    time.sleep(0.6)
    with _lock:
        if _audio_last_error and (_audio_thread is None or not _audio_thread.is_alive()):
            err = _audio_last_error
            _audio_started_at = None
            return f"System-audio capture failed to start, sir: {err}."
    return ("System-audio capture engaged, sir. I'll passively transcribe "
            "whatever the PC is playing.")


def ambient_audio_stop(_: str = "") -> str:
    """Stop the WASAPI loopback capture."""
    global _audio_thread, _audio_started_at
    with _lock:
        if _audio_thread is None or not _audio_thread.is_alive():
            return "System-audio capture is not running, sir."
        _audio_stop_evt.set()
        t = _audio_thread
    t.join(timeout=3.0)
    with _lock:
        if t.is_alive():
            return "System-audio capture did not stop cleanly within 3 s, sir."
        _audio_thread = None
        elapsed = (time.time() - _audio_started_at) if _audio_started_at else 0.0
        _audio_started_at = None
        total = _audio_entries_total
        _persist_state()
    return (f"System-audio capture disengaged, sir. {total} entries over "
            f"{int(elapsed)}s.")


def ambient_screen_start(_: str = "") -> str:
    """Begin the periodic screen-snapshot loop."""
    global _screen_thread, _screen_stop_evt, _screen_started_at
    global _screen_heartbeat, _screen_last_error, _screen_last_phash
    with _lock:
        if _screen_thread is not None and _screen_thread.is_alive():
            return "Screen watcher is already active, sir."
        _screen_stop_evt = threading.Event()
        _screen_started_at = time.time()
        _screen_heartbeat = _screen_started_at
        _screen_last_error = None
        _screen_last_phash = None
        _screen_thread = threading.Thread(
            target=_screen_worker_loop, name="ambient-screen", daemon=True)
        _screen_thread.start()
    interval = float(_get_config("AMBIENT_SCREEN_INTERVAL_S", 60.0))
    budget = _vision_budget_remaining()
    return (f"Screen watcher engaged, sir. Sampling every {int(interval)}s "
            f"with ${budget:.4f} of today's vision budget remaining.")


def ambient_screen_stop(_: str = "") -> str:
    """Stop the screen-snapshot loop."""
    global _screen_thread, _screen_started_at
    with _lock:
        if _screen_thread is None or not _screen_thread.is_alive():
            return "Screen watcher is not running, sir."
        _screen_stop_evt.set()
        t = _screen_thread
    t.join(timeout=5.0)
    with _lock:
        if t.is_alive():
            return "Screen watcher did not stop cleanly within 5 s, sir."
        _screen_thread = None
        elapsed = (time.time() - _screen_started_at) if _screen_started_at else 0.0
        _screen_started_at = None
        total   = _screen_entries_total
        skipped = _screen_skipped_total
        blocked = _screen_blocked_total
        _persist_state()
    return (f"Screen watcher disengaged, sir. {total} snapshots logged, "
            f"{skipped} skipped, {blocked} blocked over {int(elapsed)}s.")


def ambient_full_start(_: str = "") -> str:
    """Turn on mic + system audio + screen at once."""
    r1 = ambient_listen_start("")
    r2 = ambient_audio_start("")
    r3 = ambient_screen_start("")
    return "Full ambient mode engaged, sir. " + " | ".join(
        x.split(",", 1)[0] if "," in x else x for x in (r1, r2, r3))


def ambient_full_stop(_: str = "") -> str:
    """Turn off mic + system audio + screen at once."""
    r1 = ambient_listen_stop("")
    r2 = ambient_audio_stop("")
    r3 = ambient_screen_stop("")
    return "Full ambient mode disengaged, sir. " + " | ".join((r1, r2, r3))


def ambient_mic_only(_: str = "") -> str:
    """Keep the mic on; turn off system audio + screen daemons."""
    msgs = []
    with _lock:
        mic_running = _thread is not None and _thread.is_alive()
    if not mic_running:
        msgs.append(ambient_listen_start(""))
    msgs.append(ambient_audio_stop(""))
    msgs.append(ambient_screen_stop(""))
    return "Mic-only ambient mode, sir. " + " | ".join(msgs)


def ambient_listen_status(_: str = "") -> str:
    """One-line summary across mic + audio + screen daemons."""
    with _lock:
        mic_running = _thread is not None and _thread.is_alive()
        mic_entries = len(_buffer)
        last_text = _buffer[-1]["text"] if _buffer else ""
        last_ts   = _buffer[-1]["ts"] if _buffer else 0.0
        hb_age = (time.time() - _heartbeat) if _heartbeat else 0.0
        err = _last_error
        started = _started_at

        audio_running = _audio_thread is not None and _audio_thread.is_alive()
        audio_total = _audio_entries_total
        audio_err = _audio_last_error

        screen_running = _screen_thread is not None and _screen_thread.is_alive()
        screen_total = _screen_entries_total
        screen_skipped = _screen_skipped_total
        screen_blocked = _screen_blocked_total
        screen_err = _screen_last_error

    budget = _vision_budget_remaining()

    # Mic line
    if mic_running:
        uptime = int(time.time() - started) if started else 0
        last_str = (time.strftime("%H:%M:%S", time.localtime(last_ts))
                    if last_ts else "never")
        snippet = (last_text[:60] + "…") if len(last_text) > 60 else last_text
        health = ""
        if hb_age > _HEARTBEAT_STALE_SECONDS:
            health = f" (heartbeat stale {int(hb_age)}s)"
        mic_line = (f"mic ON {uptime}s, {mic_entries} entries, last "
                    f"{last_str}: {snippet!r}{health}")
    else:
        mic_line = f"mic OFF" + (f" (last error: {err})" if err else "")

    audio_line = (f"audio {'ON' if audio_running else 'OFF'} "
                  f"({audio_total} entries"
                  + (f", err={audio_err}" if audio_err and not audio_running else "")
                  + ")")
    screen_line = (f"screen {'ON' if screen_running else 'OFF'} "
                   f"({screen_total} logged, {screen_skipped} skipped, "
                   f"{screen_blocked} blocked"
                   + (f", err={screen_err}" if screen_err and not screen_running else "")
                   + f", budget ${budget:.4f})")

    return f"Ambient status, sir — {mic_line}; {audio_line}; {screen_line}."


def register(actions: dict) -> None:
    """Skill-loader entry point — register every action and optionally
    autostart per AMBIENT_LISTEN_ENABLED + the new multimodal toggles."""
    actions["ambient_listen_start"]  = ambient_listen_start
    actions["ambient_listen_stop"]   = ambient_listen_stop
    actions["ambient_listen_status"] = ambient_listen_status

    actions["ambient_audio_start"]   = ambient_audio_start
    actions["ambient_audio_stop"]    = ambient_audio_stop
    actions["ambient_screen_start"]  = ambient_screen_start
    actions["ambient_screen_stop"]   = ambient_screen_stop

    actions["ambient_full_start"]    = ambient_full_start
    actions["ambient_full_stop"]     = ambient_full_stop
    actions["ambient_mic_only"]      = ambient_mic_only

    def _bg_autostart():
        try:
            time.sleep(2.5)
            if _get_config("AMBIENT_LISTEN_ENABLED", False):
                ambient_listen_start("")
            if _get_config("AMBIENT_AUDIO_ENABLED", False):
                ambient_audio_start("")
            if _get_config("AMBIENT_SCREEN_ENABLED", False):
                ambient_screen_start("")
        except Exception as e:
            print(f"  [ambient-listen] autostart failed: {e}")

    if (_get_config("AMBIENT_LISTEN_ENABLED", False)
            or _get_config("AMBIENT_AUDIO_ENABLED", False)
            or _get_config("AMBIENT_SCREEN_ENABLED", False)):
        threading.Thread(target=_bg_autostart,
                         name="ambient-listen-autostart",
                         daemon=True).start()
