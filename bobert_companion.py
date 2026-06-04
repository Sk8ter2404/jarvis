#!/usr/bin/env python3
"""
Bobert PC Companion
─────────────────────────────────────────────────────────────────────────────
Mic → Whisper STT → LLM → edge-tts → speakers (+ optional robot lip-sync)
Cameras → face detection → robot eye tracking (or attention awareness)
Memory: persistent across conversations, updated in real time
PC control: LLM can launch apps, open URLs, search, screenshot, etc.

Multi-instance setup (one on desk PC, one on bedroom laptop, etc.)
──────────────────────────────────────────────────────────────────
Run this script on each machine. Each instance has its own LOCATION
identifier and (optionally) its own physical robot. They share a single
MEMORY_FILE — point it at a folder synced via OneDrive / Dropbox / network
share so all instances see the same facts/projects/sessions.

Voice-only mode (no robot needed)
─────────────────────────────────
Set ROBOT_ENABLED = False. Audio plays through speakers, PC control still
works, memory still persists. Good for a laptop without a robot attached.

Quick setup
───────────
1.  pip install -r requirements.txt
2.  Set ROBOT_IP (or ROBOT_ENABLED = False for voice-only)
3.  python bobert_companion.py --list-cameras  → set CAMERAS indexes
4.  Claude: set ANTHROPIC_API_KEY env var.
    Ollama: install Ollama, `ollama pull llama3`, AI_BACKEND = "ollama".
─────────────────────────────────────────────────────────────────────────────
"""

# ── Early-boot singleton lock ────────────────────────────────────────────
# Run BEFORE the heavy imports (numpy / sounddevice / cv2 / requests) so
# the boot watchdog — which polls for jarvis.lock with a 30s timeout —
# sees us alive within milliseconds, not 5-10s later once PortAudio /
# OpenCV finish initialising. stdlib-only on purpose.
import os, sys, subprocess, time

# Make stdout/stderr UTF-8 so JARVIS's non-ASCII output (─, ≥, →, em-dashes,
# etc.) never raises UnicodeEncodeError on a legacy cp1252 Windows console — a
# tester may run `python bobert_companion.py` in a raw console rather than the
# utf-8 log redirect the desktop launcher uses. Best-effort; a no-op if the
# stream is already utf-8 or doesn't support reconfigure.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - import-time console-encoding fallback; only fires on a stream without reconfigure()
        pass

def _read_lock_pid(path, max_retries=10, retry_delay=0.05):
    """Read a PID from a singleton lock file with retries for the write-race.

    Returns:
      -1  → file does not exist (caller should retry O_EXCL, not delete)
       0  → file is empty or unparseable after all retries (truly stale)
      >0  → the recorded PID

    Why retry: when one process wins os.open(..., O_CREAT | O_EXCL) but
    hasn't yet flushed its PID to disk, a second process arriving in the
    same millisecond gets FileExistsError and reads an empty file. The
    pre-fix code treated empty as 'stale' and deleted the lock — the
    winner's subsequent os.write(fd, pid) then wrote into an unlinked
    inode while the loser claimed a fresh O_EXCL. Both processes thought
    they owned the singleton (observed 2026-05-29 19:14: PIDs 45196 +
    39540 spawned in the same millisecond, watchdog caught the duplicate).

    Retry budget (10 × 50ms = 500ms) easily covers the microsecond gap
    between O_EXCL succeeding and os.write completing, while still
    bounding the cost when a crashed instance left a genuinely empty
    lock behind."""
    for _ in range(max_retries):
        try:
            with open(path, "r", encoding="utf-8") as _f:
                raw = _f.read(64).strip()
        except FileNotFoundError:
            return -1
        except (OSError, UnicodeDecodeError):
            time.sleep(retry_delay)
            continue
        if raw:
            try:
                return int(raw)
            except ValueError:
                time.sleep(retry_delay)
                continue
        time.sleep(retry_delay)
    return 0


# ── Single-instance guard: OS-held byte-range lock ────────────────────────
# The fd that holds the kernel-level exclusive lock for this process's entire
# lifetime. Stored in a module global so the garbage collector never closes it
# (closing it would release the lock and let a second instance in). The kernel
# releases the lock ONLY when the handle table is torn down — i.e. real process
# death — so a force-killed-but-hung process keeps holding it. That is what
# makes this immune to the duplicate-instance race the old PID-file approach
# had (a half-killed process ran its atexit handler, deleted jarvis.lock, then
# hung; a second instance found no lock and booted — observed 2026-05-30 10:58).
#
# The OS lock lives on a DEDICATED file (jarvis.singleton.lock) that NOTHING
# ever reads — on Windows byte-range locks are mandatory, so any reader of a
# locked file region gets a sharing violation. Keeping the mutex on its own
# file lets the plain PID file (jarvis.lock) stay freely readable by the smoke
# test, watchdog, and upgrade pipeline. The PID file is informational only; the
# OS lock is the sole authority on "is another instance alive".
_SINGLETON_HELD_FD = None


def _acquire_os_singleton_lock(fd) -> bool:
    """Non-blocking exclusive OS lock on byte 0 of the dedicated mutex file.

    Returns True if we got it (we are the sole instance), False if another live
    process already holds it (refuse to boot). On any platform/availability
    problem we fail OPEN (return True) — a possible duplicate is less bad than
    refusing to start at all when the locking machinery is simply missing.
    """
    try:
        if sys.platform == "win32":
            import msvcrt
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # non-blocking exclusive
                return True
            except OSError:
                return False
        else:  # pragma: no cover - POSIX fcntl lock branch (unreachable on the win32 test/runtime host)
            import fcntl
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, 0)
                return True
            except OSError:
                return False
    except Exception:
        return True


def _early_boot_singleton_lock():  # pragma: no cover - boot entry: real OS byte-range lock acquire + PID-file write + duplicate-refusal sys.exit; short-circuited by the test sentinel
    """Acquire the authoritative OS-held single-instance lock at the very top
    of boot, before the heavy imports. See _SINGLETON_HELD_FD / the
    _acquire_os_singleton_lock docstring for why a held byte-range lock beats
    the old PID-file + liveness-poll scheme (which let a hung-but-alive process
    free a second instance).

    Two files are involved:
      • jarvis.singleton.lock  — the DEDICATED mutex. We hold a kernel
        byte-range lock on byte 0 for our whole lifetime. Nothing reads it.
        This is the authoritative guard.
      • jarvis.lock            — the plain PID file the smoke test / watchdog /
        upgrade pipeline read. Informational only; freely readable.
    """
    global _SINGLETON_HELD_FD
    # ── Re-entrancy / re-import guard (CRITICAL) ──────────────────────────
    # When bobert_companion.py runs as __main__ and is THEN imported as the
    # module `bobert_companion` (which dozens of core/ + skills/ modules do),
    # Python re-executes this top-level call under a SECOND module identity in
    # the SAME process. Each module object has its own _SINGLETON_HELD_FD, so a
    # naive re-run opens a fresh fd and tries to lock the byte our __main__ fd
    # already holds — Windows byte-range locks are per-handle, so that lock
    # FAILS even for the same process, the code thinks 'another JARVIS holds
    # it', and sys.exit(0)s mid-import, killing the whole process (observed
    # 2026-05-30 11:10: PID 38888 self-terminated this way). The env sentinel
    # is process-wide (shared across all module objects), so the re-import
    # short-circuits to a no-op. The old O_EXCL code survived this via
    # `if old_pid == os.getpid(): return True`; this is the equivalent.
    if os.environ.get("_JARVIS_SINGLETON_PID") == str(os.getpid()):
        return True
    is_staging = (os.environ.get("JARVIS_STAGING", "").strip() == "1"
                  or "--staging" in sys.argv)
    lock_name = "jarvis_staging.lock" if is_staging else "jarvis.lock"
    mutex_name = ("jarvis_staging.singleton.lock" if is_staging
                  else "jarvis.singleton.lock")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    lock_path = os.path.join(root_dir, lock_name)
    mutex_path = os.path.join(root_dir, mutex_name)
    # Side-channel marker: detached pythonw.exe has no parent console, so
    # stderr alone won't reach the boot script. Drop a small file the
    # stability smoke test can read when the lock never materialises, so
    # 'jarvis.lock never appeared' becomes an actionable cause.
    boot_err_name = ("jarvis_staging_boot_error.txt" if is_staging
                     else "jarvis_boot_error.txt")
    boot_err_path = os.path.join(root_dir, boot_err_name)
    try:
        if os.path.exists(boot_err_path):
            os.remove(boot_err_path)
    except OSError:
        pass

    # ── Acquire the OS-held exclusive lock on the dedicated mutex file ────
    # Open-or-create jarvis.singleton.lock with O_RDWR and grab a kernel
    # byte-range lock on byte 0. The mutex file persists across runs; its
    # existence carries no meaning — the held lock is the sole arbiter of "is
    # another instance alive". A force-killed-but-hung prior process still holds
    # the kernel lock, so we fail to acquire it and refuse, instead of finding a
    # deleted PID file and booting alongside the zombie (the 2026-05-30 race).
    last_err = None
    fd = None
    try:
        fd = os.open(mutex_path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        # Couldn't even open the mutex (transient OneDrive / AV hold). Fall
        # through to the failure path, which fast-fails so the smoke test sees
        # a clear cause instead of a 30 s lock-wait.
        last_err = exc

    if fd is not None:
        if not _acquire_os_singleton_lock(fd):
            # Couldn't lock. Before refusing, rule out the re-import-of-self
            # case: if the PID file already names US, this is just a second
            # module-identity of our own process — we already hold the mutex on
            # the __main__ fd. Treat as success, not a duplicate.
            if _read_lock_pid(lock_path) == os.getpid():
                try:
                    os.close(fd)
                except OSError:
                    pass
                os.environ["_JARVIS_SINGLETON_PID"] = str(os.getpid())
                return True
            # Another LIVE JARVIS holds the mutex — including a hung-but-alive
            # one, which is precisely the case the old PID-file scheme missed.
            old_pid = _read_lock_pid(lock_path)
            try:
                os.close(fd)
            except OSError:
                pass
            print("\n[singleton] Another JARVIS already holds the lock"
                  + (f" (PID {old_pid})." if old_pid and old_pid > 0 else "."))
            print("[singleton] Refusing to start a duplicate. "
                  "Quit the existing one first.")
            print(f"[singleton] If you're certain none is running, delete: "
                  f"{mutex_path}\n")
            try:
                sys.stdout.flush()
            except Exception:
                pass
            sys.exit(0)

        # We own the mutex. KEEP THE FD OPEN for the whole process lifetime —
        # the module global stops GC from closing it (which would drop the
        # lock). The mutex file itself stays empty; we write the PID to the
        # SEPARATE plain jarvis.lock that tooling reads.
        _SINGLETON_HELD_FD = fd
        # Process-wide sentinel so a later re-import of this module short-
        # circuits the whole function instead of fighting our own lock.
        os.environ["_JARVIS_SINGLETON_PID"] = str(os.getpid())
        try:
            with open(lock_path, "w", encoding="utf-8") as _pf:
                _pf.write(str(os.getpid()))
            last_err = None
        except OSError as exc:
            last_err = exc

    if last_err is not None:
        errno_val = getattr(last_err, "errno", None)
        winerr = getattr(last_err, "winerror", None)
        msg = (f"[early-singleton] could not claim lock: "
               f"{last_err!r} (errno={errno_val}, winerror={winerr}, "
               f"path={lock_path})")
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        try:
            with open(boot_err_path, "w", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass
        # Append a JSONL record so the BootFailureDaemon in a later/parallel
        # JARVIS can detect & queue an [anomaly] task. boot_error.txt is the
        # current-failure marker (cleared at next boot); the jsonl is the
        # durable history that survives across launches.
        try:
            data_dir = os.path.join(root_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            with open(os.path.join(data_dir, "boot_failures.jsonl"),
                      "a", encoding="utf-8") as f:
                import json as _json
                f.write(_json.dumps({
                    "ts": time.time(),
                    "iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                         time.localtime()),
                    "kind": "lock_write_failed",
                    "lock_path": lock_path,
                    "errno": errno_val,
                    "winerror": winerr,
                    "error_repr": repr(last_err),
                    "is_staging": is_staging,
                }) + "\n")
        except Exception:
            pass
        # Fast-fail: stability smoke test was waiting 30s for a lock that
        # would never appear. Exit now so the test sees process death + the
        # boot_error.txt marker within <1s instead of hitting the timeout.
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.exit(1)
    try:
        sys.stdout.flush()
    except Exception:
        pass
    return True

try:  # pragma: no cover - boot-entry call wrapper (lock fast-fail / lockless-fallback paths run only at real boot)
    _early_boot_singleton_lock()
except SystemExit:  # pragma: no cover - boot lock fast-fail propagation (real boot only)
    # Lock-write fast-fail above — propagate so we exit immediately.
    raise
except Exception as _ebse:  # pragma: no cover - boot lockless-fallback on unexpected early-lock error (real boot only)
    # Never let the early-boot check raise on UNEXPECTED errors — better to
    # proceed lock-less than to crash silently before the session log is open.
    print(f"  [early-singleton] check failed, proceeding without lock: {_ebse}")

import asyncio, hashlib, io, json, logging, math, queue, random, re, tempfile, threading, time, traceback
from collections import deque
import numpy as np
import sounddevice as sd
import soundfile as sf
import cv2
import requests

# Command-level pattern memory — separate from bobert_memory.json (long-term
# facts) and memory/patterns.json (session aggregates). Aliased because
# `memory` is a frequently-used local var name throughout this module.
import memory as pattern_memory

# JARVIS-voice failure library — keyed by failure class (network, permission,
# parse, app-not-found, timeout, com, ui_automation, io, unknown). Imported
# eagerly so the dispatcher can lean on it without a hot-path try/except.
import jarvis_failure_lines as _jfl

# Canonical MCU phrasebook (acknowledgements / pushback / status / initiative /
# observation / dry humour / concern / minimal). Injected into the system prompt
# by build_system_prompt() so the LLM draws from JARVIS's signature lines
# instead of generic acknowledgements, with rotation tracked via memory.
import mcu_phrases as _mcu_phrases

# Fuzzy-match middleware for action names the LLM emits but ACTIONS doesn't
# contain (e.g. 'enable_ambient_learning_mode' → 'ambient_listening'). Best-
# effort: if the import fails we just fall through to the existing "unknown
# action" error so a broken file can't take the dispatcher offline.
try:
    import command_autocorrect as _cmd_autocorrect  # type: ignore
except Exception:  # pragma: no cover - defensive fallback when an optional module is absent (present in this env)
    _cmd_autocorrect = None  # type: ignore
_AUTOCORRECT_THRESHOLD = 0.75
# When two candidates both clear the threshold and sit within this gap of
# each other, the dispatcher asks 'did you mean X or Y' instead of silently
# routing to the top hit. Tuned alongside the embedding-blended scoring so
# the prompt only fires on genuinely close calls.
_AUTOCORRECT_AMBIG_GAP = 0.10
# Disambiguation queue: at most one pending choice at a time. Populated by
# parse_and_run_actions when the autocorrect layer returns "ambiguous" and
# drained by handle_autocorrect_disambig_response on the next user turn.
# Shape: [{"primary": (name, arg), "secondary": (name, arg), "original": str}]
_pending_autocorrect_choice: list[dict] = []

# Five-label emotional state classifier (stressed/frustrated/excited/focused/tired).
# Best-effort: if the package import fails for any reason the LLM call falls
# back to the existing detect_tone() output without raising.
try:
    from core import emotion_tracker as _emotion_tracker  # type: ignore
except Exception:  # pragma: no cover - defensive fallback when an optional module is absent (present in this env)
    _emotion_tracker = None  # type: ignore

# Tone-aware TTS layer — owns the [wry] tag parser, the per-spec 'concerned'
# / 'wry' preset values, and the punchline-pause split helper. Falls through
# to inline logic if the import fails so a partial install never silences TTS.
try:
    from core import tts as _tts_layer  # type: ignore
    # Emotion/intent/user-tone preset data + the text-emotion classifier were
    # consolidated into core/tts.py. Re-export them so _parse_intent_tag, the
    # thin _resolve_tts_preset shim below, the night_owl_mode monkeypatch, and
    # tools/audit_codebase.py keep resolving them off this namespace.
    from core.tts import (  # noqa: F401  (re-exported for in-file + skill callers)
        _TTS_EMOTION_PRESETS,
        _INTENT_PRESETS,
        _USER_TONE_TTS,
        _TTS_EMOTION_KEYWORDS,
        detect_tts_emotion,
    )
except Exception:  # pragma: no cover - defensive fallback when an optional module is absent (present in this env)
    _tts_layer = None  # type: ignore
    # Neutral-only inline fallbacks so a broken core.tts never NameErrors the
    # monolith — JARVIS keeps talking, just without prosody variety.
    _TTS_EMOTION_PRESETS = {"neutral": {"rate": "+0%", "pitch": "+0Hz", "gain": 1.0}}  # type: ignore
    _INTENT_PRESETS = {}  # type: ignore
    _USER_TONE_TTS = {}  # type: ignore
    _TTS_EMOTION_KEYWORDS = ()  # type: ignore

    def detect_tts_emotion(_text=""):  # type: ignore
        return "neutral"

# Shared Anthropic client mechanics (model/timeout/streaming in one place).
# Falls through to None so the inline create() fallbacks below still run if the
# module is missing — behaviour is identical either way.
try:
    from core import llm_client as _llm_client  # type: ignore
except Exception:  # pragma: no cover - defensive fallback when an optional module is absent (present in this env)
    _llm_client = None  # type: ignore

# Pre-LLM tone classifier (pure heuristics + data) lives in core/tone_detector.py.
# Hard import — it's stdlib-only so it can't fail on optional deps. The thin
# detect_tone() wrapper defined below supplies the previous utterance for
# cross-turn detection; the voice-emotion router and _call_llm reference the
# re-exported names (_EXCITEMENT_PHRASES / _STRESS_SWEAR_WORDS /
# _is_late_night_hour / _tone_system_addendum) directly.
from core import tone_detector as _tone_detector
from core.tone_detector import (  # noqa: F401  (re-exported for in-file callers)
    _STRESS_SWEAR_WORDS,
    _EXCITEMENT_PHRASES,
    _is_late_night_hour,
    _tone_system_addendum,
)

# Whisper transcription gates (is_valid_speech / is_ambient_music) + their
# tuning constants live in core/speech_filter.py. Re-export the two functions
# and WHISPER_TRUST_RMS (the one threshold the main loop also reads); the other
# constants are used only inside is_valid_speech and stay in that module.
from core.speech_filter import (  # noqa: F401
    is_valid_speech, is_ambient_music, WHISPER_TRUST_RMS,
)

# Voice-emotion mood router lives in core/voice_emotion.py; the monolith keeps a
# thin route_voice_emotion(user_text, now=None) wrapper (defined below) that
# supplies the previous utterance from conversation_history.
from core import voice_emotion as _voice_emotion

# Legacy flat memory store (load_memory / save_memory / _empty_memory) lives in
# core/legacy_memory.py. configure() is called below — once MEMORY_FILE is final
# — to point it at this role's file and share _memory_lock. merge_memory stays
# in this file (it orchestrates load→dedupe→save).
from core import legacy_memory as _legacy_memory
from core.legacy_memory import load_memory, save_memory, _empty_memory  # noqa: F401

# Voice-mood response adapter — turns a 'stressed' label from the voice
# emotion router into concrete side effects (proactive-nudge suppression,
# warm-dim lights, deferential prompt addendum). Best-effort import so a
# broken adapter never blocks the LLM path.
try:
    from adapters import voice_mood_response as _voice_mood_response  # type: ignore
except Exception:  # pragma: no cover - defensive fallback when an optional module is absent (present in this env)
    _voice_mood_response = None  # type: ignore

# Contextual callbacks / pending promises — when JARVIS says "I'll let you
# know when X finishes", make_promise() stores the deferred message and the
# watcher thread fires proactive_announce() when X's condition is met.
# Inspectable at memory/pending_promises.json.
try:
    from core import memory as _promises  # type: ignore
except Exception:  # pragma: no cover - defensive fallback when an optional module is absent (present in this env)
    _promises = None  # type: ignore

# Real-time mic cleanup pipeline (noise-cancel-1). Three layers — echo
# cancellation, noise suppression, gain normalisation — applied in order
# to every captured chunk before it reaches Whisper. Import is best-
# effort so a missing numpy/dep never silences the legacy path.
try:
    from core import audio_processor as _audio_processor  # type: ignore
except Exception:  # pragma: no cover - defensive fallback when an optional module is absent (present in this env)
    _audio_processor = None  # type: ignore

# Draft preview / confirmation gate — middleware that intercepts every
# send_* action, reads the pending draft body aloud, and waits 8 s for a
# voice 'send' / 'confirm' before actually firing the underlying action.
# Best-effort import so the dispatcher keeps working if the gate file is
# broken or absent — degrades to legacy "send fires immediately" behaviour.
try:
    from core import draft_preview_gate as _draft_preview_gate  # type: ignore
except Exception:  # pragma: no cover - defensive fallback when an optional module is absent (present in this env)
    _draft_preview_gate = None  # type: ignore

# iTunes COM is now isolated in audio/itunes_bridge.py so importing it
# (here or from a skill) never touches win32com / pythoncom at module-load.
# get_client() is the lazy entry point — see the bridge module's docstring.
from audio import itunes_bridge as _itunes_bridge  # type: ignore

# ──────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────

# Phase-1 refactor (2026-05-29): top-level config constants now live in
# core/config.py. Importing * here keeps every existing reference working
# while letting the pipeline reviewer load just the small file when it
# needs to reason about a config knob. Add new top-level constants
# THERE, not here. See core/config.py docstring for the contract.
from core.config import *  # noqa: F401,F403

# (LOCAL_LLM_*, ORCHESTRATOR_*, LOCAL_VISION_*, IMAGE_GEN_*, TTS_*, XTTS_*,
# WHISPER_MODEL — all moved to core/config.py in Phase-1 refactor.
# Override per-machine by editing core/config.py or by setting the
# matching JARVIS_* environment variable when the runtime helper picks
# the value, never here.)

# (VOICE_MODE moved to core/config.py in Phase 1B.)
# (RAG_INDEX_PATHS, RAG_EMBED_MODEL, RAG_OLLAMA_ENDPOINT, RAG_RERANKER_MODEL
# moved to core/config.py in Phase 1D. RAG_INDEX_PATHS uses
# os.path.expanduser("~") at import time — see core/config.py docstring
# for the single-exception note on import-time side effects.)

# Phase-1B (2026-05-29): WHISPER_DEVICE/CUDA/CPU, AUDIO_DUCKING_-
# ENABLED/LEVEL/FADE_MS, MISSION_NARRATION_*, MID_TASK_STATUS_* moved to
# core/config.py. AUDIO_DUCKING_TARGETS stays here (lives next to the
# duck_session() consumer that case-insensitive-substring-matches it).
AUDIO_DUCKING_TARGETS = (
    "chrome", "spotify", "itunes", "apple music", "applemusic",
    "msedge", "firefox", "vlc", "wmplayer", "brave",
)

# (CAMERAS, CAMERA_PROBE_ENABLED/MAX/TIMEOUT_SEC, CAMERA_LOCK_PROCESSES
# moved to core/config.py in Phase 1C. Run `python bobert_companion.py
# --list-cameras` to find the right indexes if you need to adjust.)

# (MONITORS + CONSOLE_MONITOR moved to core/config.py in Phase 1C. Run
# `python bobert_companion.py --list-monitors` if you need to re-map.)

# (HUD_ENABLED, HUD_MONITOR, RETICLE_OVERLAY_ENABLED, TRAY_ENABLED,
# HOLOGRAPHIC_OVERLAY_AUTO_LAUNCH, HOLO_WORKSHOP_AUTO_ON_THINK moved to
# core/config.py in Phase 1C.)

# (BAMBU_PRINTER_IP, BAMBU_ACCESS_CODE, BAMBU_SERIAL moved to
# core/config.py in Phase 1C. The bambu_monitor skill imports them via
# the wildcard re-export.)

# (ITUNES_AUTO_LAUNCH moved to core/config.py in Phase 1C. The
# set_auto_launch() call MUST stay here so the bridge picks the live
# value at boot.)
_itunes_bridge.set_auto_launch(ITUNES_AUTO_LAUNCH)

# Bambu H2D corner overlay — RETIRED 2026-05-30. The unified HUD
# (hud/jarvis_unified_hud.py) now shows live print progress + ETA inline, so
# the standalone corner widget is redundant and was COLLIDING with the unified
# HUD on the top monitor. Forced False here because this local assignment runs
# AFTER `from core.config import *` and would otherwise override the config
# flag of the same name. Flip back to True only if you want the separate
# corner widget again. Still available on demand via `bambu overlay on`.
BAMBU_OVERLAY_AUTO_WHILE_PRINTING = False

# Daily briefing — skills/daily_briefing.py fires once per day at this local
# clock time and delivers a JARVIS-style summary (time, weather, first Teams
# meeting if Outlook is available, Bambu print status). If the user is at the
# desk (via face_tracker), the briefing fires immediately; otherwise it waits
# up to DAILY_BRIEFING_WAIT_MINUTES for them to appear before speaking anyway.
DAILY_BRIEFING_ENABLED       = True
DAILY_BRIEFING_HOUR          = 8     # 24-hour clock, local time
DAILY_BRIEFING_MINUTE        = 0
DAILY_BRIEFING_WAIT_MINUTES  = 30    # how long to wait for user presence

# Briefing fallback sources — skills/briefing_sources.py runs a fallback chain
# (wttr → Open-Meteo → cached) for weather and (Outlook COM → Microsoft Graph
# → Google Calendar ICS) for events so a single dead service can't silently
# degrade the briefings. Leave the next two None to auto-geolocate by IP via
# ipapi.co (cached for 30 days); pin them to skip the network lookup entirely.
OPEN_METEO_LAT = None    # e.g. 40.7128
OPEN_METEO_LON = None    # e.g. -74.0060
# Optional public "secret address" ICS URL from Google Calendar settings.
# Leave blank to skip the Google Calendar layer.
GOOGLE_CALENDAR_ICS_URL = ""

# Microsoft Graph (calendar + unread mail count for hud_card + morning_briefing).
# Replaces the Outlook COM lookup that failed whenever Outlook desktop wasn't
# running. Two ways to authenticate:
#   1. MSAL device-code flow (recommended). Set MS_GRAPH_CLIENT_ID to your
#      Azure AD public-client app id, install msal (pip install msal), and run:
#         python -m skills.ms_graph --auth
#      Token caches in ms_graph_msal_cache.json beside this file.
#   2. Manual: drop microsoft_graph_token.json with
#         {"access_token": "...", "expires_at": <epoch>}
#      next to this file. Add "refresh_token" + MS_GRAPH_CLIENT_ID for silent
#      refresh.
# Leave MS_GRAPH_CLIENT_ID blank to disable both — Graph is a no-op fallback.
MS_GRAPH_CLIENT_ID = ""           # Azure AD app (public client) id
MS_GRAPH_TENANT_ID = "common"     # 'common' supports personal + work accounts
MS_GRAPH_SCOPES    = ["Calendars.Read", "Mail.Read"]

# Amazon order tracker — skills/amazon_order_tracker.py polls whichever email
# backend (Microsoft Graph / Gmail via skills/email_triage.py) is already
# configured every 15 minutes, parses Amazon shipment notifications, and
# proactively announces order status transitions (shipped → out for delivery
# → delivered, plus delay warnings). Reuses the existing email backend
# credentials — no separate Amazon login needed. Leave False to keep the
# voice actions (check_orders / recent_delivery / amazon_tracking_status)
# available without starting the background poller.
AMAZON_TRACKING_ENABLED = False

# Evening briefing — skills/evening_briefing.py fires once per day at this
# local clock time (default 22:00) with an end-of-day summary: voice
# interaction count, tasks completed today, Bambu print status, tomorrow's
# weather/calendar preview, plus one dry observation drawn from today's
# session logs. Same presence-wait behaviour as the morning daily briefing.
EVENING_BRIEFING_ENABLED      = True
EVENING_BRIEFING_HOUR         = 22    # 24-hour clock, local time
EVENING_BRIEFING_MINUTE       = 0
EVENING_BRIEFING_WAIT_MINUTES = 30    # how long to wait for user presence

# Weather briefing — skills/weather_briefing.py adds forward-looking
# precipitation alerts ("I'd suggest the umbrella today, sir — 80% chance of
# rain at 3 PM") to the morning/evening briefings and runs a background
# watcher that proactively warns about significant weather transitions
# within the next 2 hours. Uses Open-Meteo's hourly forecast (no API key).
WEATHER_BRIEFING_ENABLED          = True
WEATHER_BRIEFING_PROACTIVE        = True   # background watcher on/off
WEATHER_POLL_MINUTES              = 30
WEATHER_UMBRELLA_PROB_THRESHOLD   = 50     # % precipitation probability
WEATHER_LOOKAHEAD_HOURS           = 2      # proactive alert window
WEATHER_SIGNIFICANT_TEMP_DROP_C   = 5      # °C drop within window
WEATHER_ALERT_COOLDOWN_HOURS      = 4      # per-alert-class cooldown

# Daily recap -- skills/daily_recap.py fires once per day at this local clock
# time (default 22:30) with a JARVIS-style end-of-day summary synthesised from
# the pattern_learning JSONL, today's session logs, Teams events, and
# bambu_monitor state ("Sir, today you spent 2 hours 40 minutes in Bambu
# Studio, completed one print, took 4 Teams calls including one from a
# colleague, and played 11 tracks. Shall I queue the same morning
# briefing for tomorrow?"). Also runs on demand via the daily_recap action
# ("JARVIS, recap my day").
DAILY_RECAP_ENABLED           = True
DAILY_RECAP_HOUR              = 22    # 24-hour clock, local time
DAILY_RECAP_MINUTE            = 30

# News briefing — skills/news_briefing.py fetches headlines from a configurable
# RSS feed list, optionally summarises each in one sentence via the Claude
# backend, and is pulled into the morning + evening briefings to turn the
# one-line weather greeting into an actual intelligence briefing.
# Feeds can be plain URL strings or {"name": "...", "url": "..."} dicts;
# the optional name shows up in the spoken intro.
NEWS_BRIEFING_ENABLED         = True
NEWS_BRIEFING_FEEDS           = [
    {"name": "technology", "url": "https://feeds.bbci.co.uk/news/technology/rss.xml"},
    {"name": "world",      "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "weather",    "url": "https://alerts.weather.gov/cap/us.atom"},
]
NEWS_BRIEFING_HEADLINE_COUNT  = 3     # total headlines across all feeds
NEWS_BRIEFING_TIMEOUT         = 6.0   # per-feed HTTP timeout (seconds)
NEWS_BRIEFING_SUMMARIZE       = True  # rewrite each headline as one JARVIS sentence
NEWS_BRIEFING_CACHE_MINUTES   = 30    # in-process feed cache TTL

# Anticipation engine — skills/anticipation_engine.py runs a 60-second poll and
# occasionally volunteers ONE in-character line through TTS based on pattern_memory
# habits, multi-hour dwell on productivity apps, or late-hour active sessions.
# Suppressed while in a call (Teams/Zoom/Meet/Webex), asleep, in standby, or away.
ANTICIPATION_ENABLED          = True
ANTICIPATION_COOLDOWN_MINUTES = 20    # minimum gap between proactive lines

# Anticipation BRIEFING — separate from the engine above. Reads the
# pattern_learning aggregated snapshot and proactively voices predictions at
# relevant times (precise-clock predictions surface with lead time, broad
# windows fire when the current hour falls inside them). Throttled to
# once-per-day per prediction key. Suppressed when asleep / in standby /
# in a call / user reported "away" by face_tracker.
ANTICIPATION_BRIEFING_ENABLED        = True
ANTICIPATION_BRIEFING_POLL_MINUTES   = 5      # how often the scheduler thread checks (1–60)
ANTICIPATION_BRIEFING_LEAD_MINUTES   = 15     # lead time for precise-clock predictions (1–60)
ANTICIPATION_BRIEFING_CONFIDENCE_MIN = 0.5    # minimum prediction ratio to surface (0–1)

# Weekly digest — skills/weekly_digest_briefing.py reads the pattern_learning
# weekly_summaries table (cached habit clusters keyed by day-of-week × hour
# window — e.g. "Friday 8–10 PM: Netflix 4/4 weeks") and surfaces up to
# WEEKLY_DIGEST_MAX_CARDS proactive offers per day at relevant times
# ("Sir, it's Friday — and around 8 PM you usually queue Netflix. Shall I
# bring it up?"). Throttled once-per-week per cluster key so the same
# Friday-evening line doesn't surface twice in a week. Hard gates mirror
# anticipation_briefing: sleep/standby/in-call/away all suppress.
WEEKLY_DIGEST_ENABLED                = True
WEEKLY_DIGEST_POLL_MINUTES           = 15     # scheduler poll cadence (1–60)
WEEKLY_DIGEST_CONFIDENCE_MIN         = 0.5    # cluster confidence floor (0–1)
WEEKLY_DIGEST_MAX_CARDS              = 3      # max cluster cards to surface per day
WEEKLY_DIGEST_LEAD_MINUTES           = 30     # surface cluster this many min before the hour band

# (OVERNIGHT_UPGRADE_ENABLED, OVERNIGHT_IDLE_MINUTES, OVERNIGHT_CYCLE_-
# GAP_HOURS, OVERNIGHT_MODE_HOURS moved to core/config.py in Phase 1D.
# OVERNIGHT_FLAG_FILE stays here — it depends on __file__ which only
# resolves correctly inside this module.)
OVERNIGHT_FLAG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".overnight_active"
)

# (MIRROR_EYES_X/Y, MOUTH_SCALE, AUDIO_PROCESSING_ENABLED, VAD_DEBUG
# moved to core/config.py in Phase 1E. _audio_master/aec/ns/agc_enabled,
# _tts_muted, _ambient_mode_active, _daemons_paused, _debug_mode moved
# to core/state.py in Phase 2. Wildcard re-export below makes
# `bobert_companion._sleep_mode[0]` etc. resolve for skills/* that
# access these via the bc.* attribute path — see core/state.py
# docstring for the cross-module mutation contract.)
from core.state import *  # noqa: F401,F403

# Audio device auto-switching
# ─────────────────────────────
# Lists of preferred device-name substrings, in order. Bobert picks the
# first one that's currently connected. When you plug/unplug your headset,
# he switches automatically within a few seconds. Set MICROPHONE_INDEX /
# SPEAKER_INDEX to a number to override and disable auto-switching.
# Populated from the JARVIS_PREFERRED_INPUT_DEVICES / _OUTPUT_DEVICES env
# vars (comma-separated substrings); default empty so a fresh clone simply
# uses whatever the OS reports as default until the owner sets the hints.
PREFERRED_INPUT_DEVICES  = [s.strip() for s in os.getenv("JARVIS_PREFERRED_INPUT_DEVICES", "").split(",") if s.strip()]
PREFERRED_OUTPUT_DEVICES = [s.strip() for s in os.getenv("JARVIS_PREFERRED_OUTPUT_DEVICES", "").split(",") if s.strip()]

# Manual overrides — set to an integer index to force a specific device and
# disable auto-switching. None = use PREFERRED_*_DEVICES lookup.
MICROPHONE_INDEX = None
SPEAKER_INDEX    = None

# How often to re-check what devices are connected (seconds)
DEVICE_CHECK_INTERVAL = 4.0

# Barge-in: let user interrupt JARVIS while he's speaking. Only active when
# wearing a headset (otherwise speaker output would feedback through the mic).
BARGE_IN_ENABLED        = True
BARGE_IN_THRESHOLD      = 0.015   # mic RMS to trigger interruption
BARGE_IN_SUSTAIN_CHUNKS = 2       # consecutive loud chunks before aborting (~130ms)
# Substrings used to detect headset output device (case-insensitive).
# Generic defaults cover the common device-name words; the owner can add
# brand-specific hints via JARVIS_HEADSET_HINTS (comma-separated), which are
# appended to the generic list.
HEADSET_NAME_HINTS      = ["headset", "headphone", "earphone"] + [s.strip() for s in os.getenv("JARVIS_HEADSET_HINTS", "").split(",") if s.strip()]

# Whisper transcription gates (is_valid_speech / is_ambient_music) + their
# tuning constants (WHISPER_*, WAKE_WORD) moved to core/speech_filter.py,
# re-exported at the top of this file.

# Voice biometric gating for wake_listener — when the user has enrolled
# their voiceprint via skills/enroll_voice.py, the wake-word event is
# checked against the enrolled embedding before triggering the recording
# pipeline. This rejects ambient TV/music wake-ups that would otherwise
# pass through openwakeword's detector. Read by skills/wake_listener.py
# at register-time (see _apply_bobert_overrides there) so flipping the
# constant here changes the boot default without code edits in the skill.
#
#   VOICE_BIOMETRIC_ENABLED   — master switch. Default False so a fresh
#                               install with no enrolled voiceprint keeps
#                               the legacy permissive wake behavior; flip
#                               to True once enrollment is done.
#   VOICE_BIOMETRIC_THRESHOLD — cosine-similarity floor (0..1). 0.72 matches
#                               core.voice_id's CONFIDENCE_THRESHOLD; may
#                               need 0.65-0.68 for a single wake utterance.
#   GUEST_MODE_ENABLED        — temporary bypass for visitors. Resets to
#                               False on every wake_listener restart, so
#                               guests have to re-enable per boot.
VOICE_BIOMETRIC_ENABLED   = False
VOICE_BIOMETRIC_THRESHOLD = 0.72
GUEST_MODE_ENABLED        = False

# Sleep / wake mode — say a sleep phrase to mute JARVIS until the wake phrase
# is heard. While sleeping JARVIS only transcribes to check for the wake word;
# it ignores everything else completely.
SLEEP_PHRASES = {
    "stop listening", "go to sleep", "sleep mode", "stand by",
    "go on standby", "be quiet", "mute yourself", "take a break",
    "go idle", "pause listening",
}
WAKE_PHRASES = {
    "jarvis", "hey jarvis", "wake up", "start listening",
    "i need you", "come back", "resume listening", "wake",
}
# Ambient listening mode (skills/ambient_listen.py) — passive transcription
# daemon that keeps the mic open and records everything to a rolling buffer
# without responding. Wake phrases inside the buffer fire proactive_announce.
# Disabled by default because it competes with record_speech for the input
# device (Windows WASAPI rejects two opens on the same mic).
# AMBIENT_LISTEN_ENABLED now lives in core/config.py (Settings-GUI knob) and
# arrives via `from core.config import *`; do NOT re-declare it here or the
# literal would shadow the user's override.
AMBIENT_LISTEN_BUFFER_MINUTES         = 10
AMBIENT_LISTEN_CHUNK_DURATION_SECONDS = 0.5

# ── Ambient-learning mode (2026-05-31) ───────────────────────────────────────
# After an upgrade/overnight pipeline relaunches JARVIS (it sets
# JARVIS_AMBIENT_LEARNING=1 on the child's env), JARVIS comes up SILENT in
# standby: it listens + keeps the learning subsystems fed from what it hears,
# but won't speak until you say the wake phrase 'JARVIS'. This reuses the
# existing sleep/standby loop (already silent-until-wake). Two wake-resume
# sub-modes control what happens once you DO wake it:
#   'answer_then_quiet' — respond to that one turn, then drop straight back to
#                         silent ambient-learning (you say 'JARVIS' each time).
#   'stay_talkative'    — wake into full interactive mode (proactive back on)
#                         until a sleep phrase ('go to sleep' / 'standby').
AMBIENT_LEARNING_BOOT = os.environ.get("JARVIS_AMBIENT_LEARNING", "").strip() == "1"
WAKE_RESUME_MODE = (os.environ.get("JARVIS_WAKE_RESUME", "").strip().lower()
                    or "answer_then_quiet")
if WAKE_RESUME_MODE not in ("answer_then_quiet", "stay_talkative"):
    WAKE_RESUME_MODE = "answer_then_quiet"  # pragma: no cover - import-time env sanitiser; only runs when JARVIS_WAKE_RESUME holds an invalid value

# Ambient mode 2 — multimodal passive learning (skills/ambient_listen.py +
# skills/ambient_multimodal_extract.py). Three independent daemons that can
# be flipped via the voice commands 'also listen to system audio' / 'also
# watch the screen' / 'full ambient' / 'mic only'.
#
#   AMBIENT_AUDIO_ENABLED               — autostart WASAPI loopback capture
#                                         of system audio output (Windows).
#   AMBIENT_AUDIO_CHUNK_DURATION_SECONDS — 30 s default; Whisper-batch size.
#   AMBIENT_SCREEN_ENABLED              — autostart periodic screen-snapshot
#                                         analysis via the local VLM.
#   AMBIENT_SCREEN_INTERVAL_S           — seconds between screen samples.
#   AMBIENT_SCREEN_BLOCKLIST            — extra regex patterns (over focused
#                                         window title + process name) that
#                                         skip the screen capture entirely.
#                                         Defaults inside the skill already
#                                         cover 1Password, banking sites,
#                                         authenticator codes, SSNs, etc.
#   AMBIENT_VISION_BUDGET_USD           — daily ceiling on vision spend; the
#                                         daemon sleeps when the day's tally
#                                         meets/exceeds it. Local VLM calls
#                                         are charged at ~$0.0001 to keep
#                                         the tracker active even when only
#                                         the offline eye is being used.
#   AMBIENT_EXTRACT_ENABLED             — autostart the fact-extraction loop
#                                         that fuses mic + audio + screen
#                                         streams into bobert_memory.json.
#   AMBIENT_EXTRACT_INTERVAL_S          — seconds between extraction passes.
#   AMBIENT_EXTRACT_BATCH               — max log lines per extraction pass.
AMBIENT_AUDIO_ENABLED                  = False
AMBIENT_AUDIO_CHUNK_DURATION_SECONDS   = 30.0
# AMBIENT_SCREEN_ENABLED now lives in core/config.py (Settings-GUI knob) and
# arrives via `from core.config import *`; do NOT re-declare it here or the
# literal would shadow the user's override.
AMBIENT_SCREEN_INTERVAL_S              = 60.0
AMBIENT_SCREEN_BLOCKLIST               = ()      # extra regex strings
AMBIENT_VISION_BUDGET_USD              = 1.00    # daily cap
AMBIENT_EXTRACT_ENABLED                = False
AMBIENT_EXTRACT_INTERVAL_S             = 300.0
AMBIENT_EXTRACT_BATCH                  = 50
# Wake-work standby mode — distinct from sleep. Triggered either by an explicit
# phrase or by detecting ambient music bleeding into the mic. HUD shows
# 'Standby' instead of 'Idle' and resume message is 'Listening, sir.'
STANDBY_TRIGGER_PHRASES = {
    "wake work mode", "wake-work mode", "work mode",
    "go to standby", "standby mode", "enter standby",
}
# Voice-triggered shutdown — these are AMBIGUOUS phrases that should NOT
# fire a full power-off without first asking "would you like overnight first?".
# Bedtime phrases ('goodnight' / 'going to bed' / 'time to sleep') stay routed
# to start_overnight_upgrade directly because the user's intent is unambiguous;
# the prompt is ONLY for these shutdown phrases.
SHUTDOWN_TRIGGER_PHRASES = (
    "shut down jarvis", "shutdown jarvis",
    "exit jarvis", "quit jarvis",
    "turn yourself off", "turn off jarvis",
    "go offline",
    "power off jarvis",
    # Bare forms — checked last so longer forms above can match first.
    "shut down", "shutdown", "power off",
)
# Yes / no replies recognised while the shutdown prompt is armed. Matched
# as whole utterances (case-insensitive, trimmed) rather than substrings so
# a stray "yes" inside a sentence doesn't accidentally confirm. Match in
# this order so "no overnight" hits NO before YES's "overnight".
SHUTDOWN_PROMPT_YES_PHRASES = (
    "yes", "yep", "yeah", "yup", "sure",
    "go ahead", "do it",
    "overnight", "overnight protocol",
    "start overnight", "start the overnight protocol",
    "yes please", "yes go ahead",
)
SHUTDOWN_PROMPT_NO_PHRASES = (
    "no", "nope", "nah",
    "no overnight",
    "just shut down", "just shutdown",
    "shut down completely", "shutdown completely",
    "full shutdown", "full shut down",
    "no thanks", "no thank you",
)
# Goodbye lines spoken before _act_shutdown_jarvis terminates the process.
# Picked randomly so repeated shutdowns don't feel scripted.
SHUTDOWN_GOODBYE_LINES = (
    "Going dark, sir. Until next time.",
    "Powering down, sir.",
    "Off until you need me again, sir.",
    "Going offline, sir. Take care.",
)
# How long (seconds) the shutdown prompt waits for a yes/no reply before
# silently cancelling and falling back to normal LLM routing.
SHUTDOWN_PROMPT_TIMEOUT_S = 30.0
# Music-emitting actions — when any of these fire, we treat the next chunk of
# audio captured by the mic as JARVIS-initiated playback and don't count
# 'music' transcriptions toward an ambient-music standby trigger.
MUSIC_ACTION_NAMES = {
    "play_music", "resume_music", "next_song", "previous_song",
    "apple_music", "spotify", "youtube_play",
    "youtube_search_direct", "youtube_direct", "yt_direct",
    "netflix", "prime_video", "disney_plus", "hulu", "max",
    "play_streaming", "youtube",
    "media_next", "media_prev", "media_playpause",
    # apple_music_intel actions also produce playback
    "play_unheard", "play_vibe", "skip_track",
    # itunes_library actions also start playback
    "play_playlist", "shuffle_library",
}
# Window-title fragments (lowercase) likely to host an active music session
# that swallows global media keys unless focused. Ordered by priority — first
# match wins. Chrome doesn't route VK_MEDIA_* to a background tab; focusing
# the music window before the keypress makes media controls actually work.
MUSIC_WINDOW_HINTS = (
    "apple music",
    "spotify",
    "youtube music",
    "itunes",
    "tidal",
    "soundcloud",
    "amazon music",
    "pandora",
    "deezer",
    "youtube",
)
# How long (seconds) a recent JARVIS music action suppresses ambient-music
# standby. 30 min is long enough that an ordinary playlist won't trip it; any
# music transcribed after this window is treated as 'external' (radio, phone
# speakers, someone else's music in the room).
MUSIC_GRACE_PERIOD     = 1800
# Number of music-marker transcriptions inside MUSIC_HITS_WINDOW seconds
# before auto-entering standby. >1 keeps a single stray Whisper hallucination
# from putting JARVIS on standby unexpectedly.
MUSIC_HITS_TO_STANDBY  = 2
MUSIC_HITS_WINDOW      = 60

# Learning: extract new facts from every exchange in the background.
# Adds one extra LLM call per turn but makes Bobert remember things immediately.
LEARN_EVERY_TURN = True

# Proactive idle behavior — JARVIS-style spontaneous comments
PROACTIVE_ENABLED       = True
PROACTIVE_MIN_SILENCE   = 180    # seconds of silence before he MIGHT speak up
PROACTIVE_MAX_SILENCE   = 900    # by this many seconds, very likely to comment
PROACTIVE_REQUIRE_FACE  = True   # only comment if a webcam can see you

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bobert_memory.json")

# Logging — every session is written to a timestamped file in logs/ for
# later debugging. Console output is unchanged.
LOGGING_ENABLED   = True
LOGS_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_KEEP_COUNT    = 50    # delete oldest beyond this

# ──────────────────────────────────────────────────────────────────────────
#  BLUE / GREEN deployment — staging mode isolation
# ──────────────────────────────────────────────────────────────────────────
# When --staging is on argv (or JARVIS_STAGING=1 in the env), this process
# is the GREEN candidate. We must NOT touch the same lock files, audio
# devices, HUD overlay, or tray icon that the live PROD JARVIS is using.
# blue_green_manager.resource_paths(role) is the single source of truth
# for which files this role should read/write; every module-level path
# below is overridden in place so downstream code never has to branch
# on role beyond the flag globals here.
try:
    import blue_green_manager as _bgm  # type: ignore
    BLUE_GREEN_ROLE = _bgm.resolve_role()
    _BLUE_GREEN_PATHS = _bgm.resource_paths(BLUE_GREEN_ROLE)
    if BLUE_GREEN_ROLE == "staging":
        # Pre-create per-role dirs so the first staging boot doesn't trip
        # on a missing data_staging/ tree.
        _bgm.ensure_role_dirs("staging")
        # Disable every subsystem that would collide with the live prod
        # instance. Mic / tray / cameras / Bambu / overnight upgrade
        # all touch shared singletons (audio device, tray icon, MQTT
        # client) that prod owns until handoff completes.
        TRAY_ENABLED             = False
        RETICLE_OVERLAY_ENABLED  = False
        CAMERA_PROBE_ENABLED     = False
        OVERNIGHT_UPGRADE_ENABLED = False
        PROACTIVE_ENABLED        = False
        ROBOT_ENABLED            = False
        # No mic in staging — input comes via injected_commands_staging.json.
        MICROPHONE_INDEX         = -1
        # Redirect logs so we don't tail-corrupt the live session log.
        LOGS_DIR                 = _BLUE_GREEN_PATHS["logs_dir"]
        # Redirect persistent memory so tests can't pollute the real one.
        MEMORY_FILE              = os.path.join(
            _BLUE_GREEN_PATHS["data_dir"], "bobert_memory.json"
        )
        # blue-green-2: the staging HUD is VISIBLE on the LEFT monitor so
        # the user can watch the ceremony. It reads from a separate state
        # file (data_staging/hud_state.json) and is pinned to the
        # manager-declared monitor so it never blinks the prod HUD.
        HUD_ENABLED              = bool(_BLUE_GREEN_PATHS.get(
            "hud_enabled_in_staging", False))
        HUD_MONITOR              = _BLUE_GREEN_PATHS.get("monitor_name",
                                                         "left")
        # Mute the speakers in staging without killing the synthesise()
        # pipeline — core/tts.is_muted() reads this. The blue/green
        # smoke test still exercises edge-tts so a broken voice render
        # fails the gate.
        os.environ.setdefault("MUTE_TTS", "1")
        # Belt-and-suspenders for "no mic in staging": neutralise sounddevice
        # INPUT capture globally so NO probe or skill (self-diag mic/STT probes,
        # the standby lyric loop, ambient_listen, enroll_voice, the wake-word
        # detector) can open the default microphone by opening its own stream.
        # Every capture site is wrapped in try/except and degrades gracefully on
        # a PortAudioError. Output (sd.play for the muted TTS path) is untouched.
        # This is what finally made staging provably deaf. 2026-05-31.
        try:
            import sounddevice as _sd_guard
            def _staging_mic_off(*_a, **_k):  # pragma: no cover - only fires if a staging mic capture is attempted
                raise _sd_guard.PortAudioError("microphone disabled in staging")
            _sd_guard.rec = _staging_mic_off
            class _StagingNoInputStream:   # raises like a failed device open
                def __init__(self, *_a, **_k):  # pragma: no cover - only fires if a staging InputStream is opened
                    raise _sd_guard.PortAudioError("microphone disabled in staging")
            _sd_guard.InputStream = _StagingNoInputStream
        except Exception as _smic:  # pragma: no cover - defensive: sounddevice is present in this env
            print(f"  [staging] could not neutralise mic capture: {_smic}")
except Exception as _bgerr:  # pragma: no cover - blue/green manager fail-open-to-prod fallback (manager imports cleanly in this env)
    # If the manager is unavailable for any reason, fail open as PROD —
    # better to keep the existing behaviour than to crash the boot.
    print(f"  [blue-green] init failed, defaulting to prod: {_bgerr}")
    _bgm = None  # type: ignore
    BLUE_GREEN_ROLE = "prod"
    _BLUE_GREEN_PATHS = {
        "role":           "prod",
        "lock_file":      os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis.lock"),
        "data_dir":       os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        "logs_dir":       LOGS_DIR,
        "memory_dir":     os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory"),
        "hud_state_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "hud_state.json"),
        "inject_file":    os.path.join(os.path.dirname(os.path.abspath(__file__)), "injected_commands.json"),
        "tray_enabled":   True,
        "hud_enabled":    True,
        "mic_enabled":    True,
        "tts_audio_out":  True,
        "bambu_enabled":  True,
        "camera_enabled": True,
        "replies_file":   os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "replies.jsonl"),
    }


def _is_staging() -> bool:
    """Cheap helper used at hot-path call sites (e.g. _speak) to decide
    whether to skip audio/TTS and route to replies.jsonl instead."""
    return BLUE_GREEN_ROLE == "staging"

# ──────────────────────────────────────────────────────────────────────────

ROBOT_URL = f"http://{ROBOT_IP}:{ROBOT_PORT}/command"

# Phase 3 refactor (2026-05-29): the ~300-line BASE_SYSTEM_PROMPT
# string literal moved verbatim into core/prompts.py. Imported here
# so existing references (`_system_prompt = BASE_SYSTEM_PROMPT`)
# resolve unchanged. Edit the prompt in core/prompts.py — the
# pipeline reviewer loads that small file when reasoning about the
# personality block, vs. the full bobert_companion.py.
from core.prompts import BASE_SYSTEM_PROMPT  # noqa: F401

conversation_history: list[dict] = []
_system_prompt = BASE_SYSTEM_PROMPT   # extended with memory at startup

# Cap the rolling history (10 user+assistant turns). The trim loop was copy-
# pasted at two call sites with a local MAX_HISTORY each — centralised here so
# the "trim in pairs from the front" invariant (the Claude API requires the
# first message to be 'user') can't drift between them.
MAX_CONVERSATION_HISTORY = 20


def _trim_conversation_history(max_history: int = MAX_CONVERSATION_HISTORY) -> None:
    """Trim conversation_history to at most `max_history` messages, removing the
    oldest user+assistant PAIR from the front so role alternation is preserved."""
    while len(conversation_history) > max_history:
        conversation_history.pop(0)
        if conversation_history and conversation_history[0]["role"] == "assistant":
            conversation_history.pop(0)

# (_sleep_mode, _shutdown_prompt_pending, _standby_mode,
# _jarvis_played_music_at, _ambient_music_hits, _ambient_music_last_hit,
# _overnight_run_now, _wake_history, _last_wake_date moved to
# core/state.py in Phase 2 and re-exported via `from core.state import *`
# higher up in this file. The cross-module mutation contract — the
# list-identity invariant that the morning_chain watcher thread relies
# on — is preserved verbatim in core/state.py's docstring + the
# _last_wake_date comment block. DO NOT recreate inline state lists
# here; that would shadow the imported names and break skills/* readers
# that access them via the `bobert_companion.*` attribute path.)

# Pre-wake silence snapshot in seconds — captured by context_aware_greeting()
# at the moment of wake-event detection, BEFORE the greeting bumps
# last_speech_time. Consumers (e.g. skills/morning_arrival's 6-hour silence
# gate) read [0] to measure the gap from the user's last interaction without
# being clobbered by JARVIS's own greeting reply. Same list-of-one wrapper
# convention as _last_wake_date so the cross-thread read stays atomic
# without a lock — see the WHY comment above _last_wake_date if you're
# tempted to flatten this to a plain float.
_pre_wake_silence_seconds: list = [0.0]

# Phrase bank for the wake_response_variety system. Each (phrase, tags) entry
# is selectable when context_aware_greeting() reaches its default branch (i.e.
# none of the high-priority hooks fired: still-up, morning, mid-print,
# looking-away). Tags drive selection — see _pick_wake_variety().
#   general  — appropriate any time (always in the candidate pool)
#   formal   — full-greeting register; first wake / standby exit / morning
#   terse    — clipped reply; used when user is stressed/rushed or rapidly re-waking
#   soft     — gentler register; late-night or tired user
#   playful  — only chosen when tone == 'playful'
_WAKE_PHRASE_BANK: list[tuple[str, set[str]]] = [
    ("Yes, sir?",                   {"general"}),
    ("At your service.",            {"general", "formal"}),
    ("How may I be of assistance?", {"general", "formal"}),
    ("Sir.",                        {"terse", "soft"}),
    ("Hm?",                         {"playful", "terse"}),
    ("Listening.",                  {"terse"}),
    ("Standing by, sir.",           {"general"}),
    ("Sir?",                        {"terse"}),
    ("Online and ready, sir.",      {"general", "formal"}),
    ("Ready when you are, sir.",    {"general"}),
    ("I'm here, sir.",              {"general", "soft"}),
    ("Yes?",                        {"terse", "playful"}),
]

# Track the last picked phrase so we avoid immediate repeats when there are
# multiple equally-good candidates.
_last_wake_phrase: list[str | None] = [None]
# Ambient-learning runtime state (see AMBIENT_LEARNING_BOOT). _ambient_learning
# marks the mode active; _resume_to_ambient is the one-shot "a woken turn just
# finished — drop back to silent standby on the next loop pass" flag used by the
# 'answer_then_quiet' wake-resume sub-mode. Single-element lists per the
# core/state.py lock-free convention.
_ambient_learning: list[bool] = [False]
_resume_to_ambient: list[bool] = [False]


# ──────────────────────────────────────────────────────────────────────────
#  MEMORY  — facts, projects, topics, sessions persist across conversations.
#  Updated in real time after every exchange, not just on exit.
# ──────────────────────────────────────────────────────────────────────────

MAX_FACTS    = 120
MAX_PROJECTS = 20
MAX_TOPICS   = 60
MAX_SESSIONS = 20

_memory_lock = threading.RLock()


# _empty_memory / load_memory / save_memory moved to core/legacy_memory.py
# (re-exported at the top of this file). Point the store at THIS role's memory
# file and share _memory_lock so writes there serialise with the other holders
# of this lock (learn_from_turn, the ambient extractor, …). merge_memory below
# still orchestrates load → dedupe → save via the re-exports.
_legacy_memory.configure(MEMORY_FILE, _memory_lock)


# Write-time memory guards (credential redaction + internal-noise filter) moved
# to core/memory_guards.py so the security-critical logic is unit-tested in
# isolation. merge_memory (below) uses both via this re-export.
from core.memory_guards import _is_secret_fact, _is_internal_noise_fact  # noqa: E402,F401

# Canonical action-result failure markers, shared with core/dispatcher.py so the
# follow-up-loop _is_failure() check below can't drift from the dispatcher's.
from core.failure_markers import FAILURE_MARKERS  # noqa: E402


def merge_memory(new_facts=None, new_projects=None, new_topic=""):
    """Atomically merge new facts/projects/topic into bobert_memory.json.

    Holds _memory_lock across load → dedupe → trim → save so concurrent
    writers (learn_from_turn worker + ambient_multimodal_extract daemon)
    cannot lose each other's additions. Dedupe is case-insensitive on
    facts and projects. Returns (added_facts, added_projects) — the
    items actually written, after dedupe.

    Credential redaction: any candidate fact matching _SECRET_FACT_PATTERNS
    is dropped before write (security audit 2026-05-30) so secrets can't
    leak into the system prompt that's sent to the cloud every turn.
    """
    _raw_facts = [f.strip() for f in (new_facts or [])
                  if isinstance(f, str) and f.strip()]
    # Drop anything that looks like a credential before it can be stored.
    new_facts = []
    for _f in _raw_facts:
        if _is_secret_fact(_f):
            print(f"  [memory] redacted candidate fact (looks like a secret): "
                  f"{_f[:40]}…")
            continue
        if _is_internal_noise_fact(_f):
            print(f"  [memory] dropped internal-noise candidate fact: {_f[:40]}…")
            continue
        new_facts.append(_f)
    new_projects = [p.strip() for p in (new_projects or [])
                    if isinstance(p, str) and p.strip()
                    and not _is_internal_noise_fact(p.strip())]
    new_topic = new_topic.strip() if isinstance(new_topic, str) else ""

    added_facts: list[str] = []
    added_projects: list[str] = []

    if not new_facts and not new_projects and not new_topic:
        return added_facts, added_projects

    with _memory_lock:
        memory = load_memory()
        memory.setdefault("facts", [])
        memory.setdefault("projects", [])
        memory.setdefault("topics", [])

        existing_facts = {f.lower() for f in memory["facts"] if isinstance(f, str)}
        for f in new_facts:
            if f.lower() in existing_facts:
                continue
            memory["facts"].append(f)
            existing_facts.add(f.lower())
            added_facts.append(f)

        existing_projs = {p.lower() for p in memory["projects"] if isinstance(p, str)}
        for p in new_projects:
            if p.lower() in existing_projs:
                continue
            memory["projects"].append(p)
            existing_projs.add(p.lower())
            added_projects.append(p)

        if new_topic:
            memory["topics"].append({
                "date":     time.strftime("%Y-%m-%d"),
                "location": LOCATION,
                "topic":    new_topic,
            })

        memory["facts"]    = memory["facts"][-MAX_FACTS:]
        memory["projects"] = memory["projects"][-MAX_PROJECTS:]
        memory["topics"]   = memory["topics"][-MAX_TOPICS:]

        save_memory(memory)

    return added_facts, added_projects


# Phase 3 refactor (2026-05-29): the ~1160-line PC_CONTROL_PROMPT
# string literal moved verbatim into core/prompts.py. Edit the
# action catalogue there — that file is the canonical source.
from core.prompts import PC_CONTROL_PROMPT  # noqa: F401

# Phase 4A refactor (2026-05-29): 11 simple _act_* handlers (open_url,
# web_search, youtube, get_time, screenshot, media_next/prev/playpause,
# volume_up/down/mute) live in core/actions.py and are re-exported by
# wildcard so the ACTIONS dispatch dict at the bottom of this file still
# resolves them by name. core/actions.py uses a deferred `_bc()` helper
# to call back into this module for helpers like _get_pyautogui() and
# _media_key_with_focus() — no circular-import problem because the
# late-bind doesn't fire until an action is actually dispatched.
from core.actions import *  # noqa: F401,F403


_CHAPPIE_STANDING_RULES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "chappie_standing_rules.json"
)


def _load_chappie_standing_rules() -> str:
    """Read data/chappie_standing_rules.json and format its rules as a prompt
    block. Returns '' if the file is missing, unreadable, or malformed so a
    bad rules file never blocks an LLM turn. Re-read on every call so edits
    take effect immediately (the 'read-before-send' rule must always reflect
    the latest on-disk state)."""
    try:
        with open(_CHAPPIE_STANDING_RULES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""
    rules = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(rules, list) or not rules:
        return ""
    lines = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        text = r.get("rule")
        sev = r.get("severity") or "rule"
        if not rid or not text:
            continue
        lines.append(f"- {rid} ({sev}): {text}")
    if not lines:
        return ""
    return (
        "STANDING RULES (from data/chappie_standing_rules.json — honour every "
        "turn, no exceptions; hard-rule entries override other instructions):\n"
        + "\n".join(lines)
    )


def build_system_prompt(memory: dict) -> str:
    prompt = BASE_SYSTEM_PROMPT

    # Standing rules from data/chappie_standing_rules.json. Injected before
    # the phrasebook so 'read-before-send', 'no impersonation without explicit
    # confirmation', and 'no upgrade on vague approval' take priority on every
    # turn — not just after the next daily Chappie reflection cycle.
    rules_block = _load_chappie_standing_rules()
    if rules_block:
        prompt += "\n\n" + rules_block

    # Inject the canonical JARVIS phrasebook with per-intent rotation hints.
    # The phrases themselves live in mcu_phrases.py so the prompt stays the
    # single source of voice instruction while the lines stay editable in a
    # dedicated module. last_used_by_intent may be absent on legacy memory
    # files — load_memory() backfills via _empty_memory(), so .get() suffices.
    last_used = memory.get("last_used_phrase_by_intent") or {}
    prompt += "\n\n" + _mcu_phrases.render_phrasebook_block(last_used)

    if PC_CONTROL_ENABLED:
        prompt += PC_CONTROL_PROMPT

    days_known = 0
    try:
        from datetime import date
        y, m, d = map(int, memory["first_meeting"].split("-"))
        days_known = (date.today() - date(y, m, d)).days
    except Exception:
        pass

    prompt += (
        f"\n\nContext: you've known your owner for {days_known} day(s) across "
        f"{memory['conversation_count']} conversations. "
        f"You are currently the {LOCATION} instance — there may be other "
        f"versions of you running in other rooms, sharing the same memory."
    )

    if memory["facts"]:
        prompt += "\n\nWhat you know about your owner:\n"
        prompt += "\n".join(f"- {f}" for f in memory["facts"])

    if memory["projects"]:
        prompt += "\n\nProjects they've mentioned:\n"
        prompt += "\n".join(f"- {p}" for p in memory["projects"])

    if memory["topics"]:
        recent_topics = memory["topics"][-15:]
        prompt += "\n\nRecent topics you've discussed:\n"
        prompt += "\n".join(
            f"- {t['date']} ({t.get('location', '?')}): {t['topic']}"
            for t in recent_topics
        )

    if memory["sessions"]:
        recent = memory["sessions"][-5:]
        prompt += "\n\nRecent conversation summaries:\n"
        prompt += "\n".join(
            f"- {s['date']} ({s.get('location', '?')}): {s['summary']}"
            for s in recent
        )

    return prompt


# Hard per-request timeout for EVERY Claude SDK call. Without it the SDK
# default is ~600s × 2 retries, and these calls run synchronously on the main
# voice thread (conversation, vision, follow-up, _llm_quick) — a hung TLS
# socket would freeze JARVIS for minutes. On timeout the call raises and the
# surrounding code falls back to the local model. 2026-05-30 deep audit.
_ANTHROPIC_TIMEOUT_S = 30.0


def _llm_quick(system: str, user: str, max_tokens: int = 200) -> str:
    """One-shot LLM call for memory extraction / proactive comments.

    Falls back to the local Ollama model when the Claude API is
    unavailable (monthly usage cap / quota / network) so background
    learning, fact-extraction and proactive comments keep working instead
    of silently failing every turn — the 2026-05-30 cap exposed this:
    convos fell back to local but `[learn]` kept dying on the 400.

    When AMBIENT_LEARNING_FORCE_LOCAL is set, this one-shot ALWAYS uses the local
    model and never touches Claude, so ambient/background learning is free."""
    from core.config import AMBIENT_LEARNING_FORCE_LOCAL, model_route
    if AMBIENT_LEARNING_FORCE_LOCAL or model_route("ambient") == "local":
        local = _call_local_llm(
            system, [{"role": "user", "content": user}], max_tokens=max_tokens)
        if local:
            return local
        print("  [llm_quick] ambient forced local; local model unavailable")
        return ""
    if AI_BACKEND == "claude":
        import anthropic
        try:
            msg = anthropic.Anthropic(timeout=_ANTHROPIC_TIMEOUT_S).messages.create(
                model=CLAUDE_MODEL, max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
        except Exception as e:
            # Cloud unavailable — route this one-shot through the local
            # model so learning doesn't stall while the cap is active.
            local = _call_local_llm(
                system, [{"role": "user", "content": user}],
                max_tokens=max_tokens,
            )
            if local:
                return local
            print(f"  [llm_quick] cloud failed and no local fallback "
                  f"({type(e).__name__}: {e})")
            return ""
    elif AI_BACKEND == "ollama":
        import ollama
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp["message"]["content"]
    return ""


def _parse_json_array(text: str) -> list:
    """Extract first JSON array from a string, return [] if none."""
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []


def learn_from_turn(user_msg: str, ai_reply: str, memory: dict):
    """Background: extract new facts/projects/topic from this exchange."""
    if not LEARN_EVERY_TURN:
        return

    # Snapshot existing facts/projects so the extractor can avoid duplicates
    existing_facts_str    = "\n".join(f"- {f}" for f in memory.get("facts", []))
    existing_projects_str = "\n".join(f"- {p}" for p in memory.get("projects", []))

    def _worker():
        try:
            system = (
                "You extract long-term memory items from a conversation turn. "
                "Output ONLY valid JSON in this exact shape, nothing else:\n"
                '{"new_facts": ["..."], "new_projects": ["..."], "topic": "..."}\n\n'
                "STRICT RULES — follow exactly:\n\n"
                "new_facts:\n"
                "  - ONLY add facts that are CLEARLY and EXPLICITLY stated by the user.\n"
                "  - For names: ONLY add a name fact if user says 'my name is X' or 'I'm X'\n"
                "    or similar EXPLICIT self-introduction. Never infer a name from a "
                "    garbled phrase. If unsure, DO NOT add it.\n"
                "  - Skip facts that are already in memory (listed below). Don't add\n"
                "    near-duplicates either (e.g. if 'User uses Apple Music' exists,\n"
                "    don't add 'User listens to music on Apple Music').\n"
                "  - Skip facts about THIS conversation/session (e.g. 'user asked X'\n"
                "    or 'user wanted Y to happen' — those are transient, not durable).\n"
                "  - Empty list [] if no genuinely-new durable facts.\n\n"
                "new_projects:\n"
                "  - Only ongoing real-world projects the user is actually working on\n"
                "    (e.g. 'Building a robot'). Not session-level requests.\n"
                "  - Skip if similar already in memory.\n\n"
                "topic: 2–5 word label for what THIS turn was about.\n\n"
                "Existing facts (do NOT duplicate these or paraphrase them):\n"
                f"{existing_facts_str or '(none yet)'}\n\n"
                "Existing projects (do NOT duplicate):\n"
                f"{existing_projects_str or '(none yet)'}"
            )
            user = f"User said: {user_msg}\nAssistant said: {ai_reply}"
            text = _llm_quick(system, user, max_tokens=250)

            # Extract the FIRST complete JSON object from the response.
            # Using raw_decode instead of a regex so we never accidentally
            # capture two objects (which produces JSONDecodeError: Extra data).
            start = text.find("{")
            if start == -1:
                return
            try:
                data, _ = json.JSONDecoder().raw_decode(text, start)
            except json.JSONDecodeError:
                return

            raw_topic = data.get("topic", "")
            topic = raw_topic.strip() if isinstance(raw_topic, str) else ""

            # Atomic load → dedupe → trim → save under _memory_lock so we
            # cannot lose writes that race with the ambient extractor.
            added_facts, added_projects = merge_memory(
                new_facts=data.get("new_facts"),
                new_projects=data.get("new_projects"),
                new_topic=topic,
            )

            added = [f"fact: {f}" for f in added_facts] \
                  + [f"project: {p}" for p in added_projects]
            if added:
                print(f"  [learned] {'; '.join(added)}")

        except Exception as e:
            # Log to console (which logs to file too) but don't break the chat
            print(f"  [learn] failed: {type(e).__name__}: {e}")

    threading.Thread(target=_worker, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────
#  SESSION PATTERN MEMORY  — records when/how the user typically uses JARVIS
#  so it can surface patterns proactively at startup ('it's Friday night,
#  shall I queue something on Netflix?').
# ──────────────────────────────────────────────────────────────────────────

PATTERNS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")
PATTERNS_FILE = os.path.join(PATTERNS_DIR, "patterns.json")
MAX_PATTERN_ENTRIES = 200
MIN_SESSIONS_FOR_PATTERN = 5    # need this many sessions before surfacing

_session_start_time = time.time()
_session_action_counts: dict[str, int] = {}
_session_app_names: set[str] = set()
_session_pattern_lock = threading.Lock()

# Replay-last-action history. parse_and_run_actions appends every executed
# (action, arg, result) tuple here; the voice trigger 'do that again' /
# 'replay that' re-fires the most recent. Bounded at 5 so the deque self-trims;
# lock protects against concurrent appends from background skill threads.
_action_history: deque = deque(maxlen=5)
_action_history_lock = threading.Lock()

# Actions we refuse to replay silently — they mutate or destroy state, or
# exit the agent. The voice trigger returns a refusal phrase instead and
# requires the user to re-issue the command explicitly so the destructive
# step gets the normal confirmation/pushback path.
_DESTRUCTIVE_REPLAY_ACTIONS = frozenset({
    "close_window",
    "kill_process",
    "restart",
    "upgrade",
    "start_overnight_upgrade",
    "run_shell",
})


# Sliding 1-hour window of caught action failures. parse_and_run_actions'
# `except Exception` arm calls record_action_error() so skills/self_diagnostic
# can group repeated failures by (action, exception class), exceed the
# threshold (>=3 in 1h), and append a structured fix request to jarvis_todo.md
# — turning the self-healing pipeline from "log and forget" into "queue a real
# fix request". maxlen caps memory even if errors spike past the time prune.
_ACTION_ERROR_LOG_MAXLEN = 512
_ACTION_ERROR_LOG_WINDOW_S = 3600.0
_action_error_log: deque = deque(maxlen=_ACTION_ERROR_LOG_MAXLEN)
_action_error_log_lock = threading.Lock()


def record_action_error(action_name: str, exc: BaseException,
                        traceback_text: str | None = None) -> None:
    """Record a caught action failure for the self-diagnostic auto-queue.

    Bounded (maxlen + 1h time window pruned on every insert) so a runaway
    failure can't grow the deque without limit. Safe to call from any thread.
    """
    try:
        now = time.time()
        if traceback_text is None:
            try:
                traceback_text = traceback.format_exc()
            except Exception:
                traceback_text = ""
        entry = {
            "ts":         now,
            "action":     str(action_name)[:80],
            "exc_class":  type(exc).__name__,
            "exc_msg":    str(exc)[:240],
            "traceback":  (traceback_text or "")[-4096:],
        }
        cutoff = now - _ACTION_ERROR_LOG_WINDOW_S
        with _action_error_log_lock:
            _action_error_log.append(entry)
            while _action_error_log and _action_error_log[0]["ts"] < cutoff:
                _action_error_log.popleft()
    except Exception:
        # Recording must never raise into the action dispatcher.
        pass


def get_recent_action_errors(window_s: float | None = None) -> list[dict]:
    """Snapshot of recorded action errors inside the given window (default 1h).
    Returns a shallow copy so the caller can iterate without holding the lock."""
    if window_s is None:
        window_s = _ACTION_ERROR_LOG_WINDOW_S
    cutoff = time.time() - float(window_s)
    with _action_error_log_lock:
        # Lazy prune on every snapshot so the deque stays bounded by age too.
        while _action_error_log and _action_error_log[0]["ts"] < cutoff:
            _action_error_log.popleft()
        return [dict(e) for e in _action_error_log if e["ts"] >= cutoff]


def get_session_log_path() -> str | None:
    """Return the live session log path, or None if logging is off / not
    initialised yet. self_diagnostic uses this to tail recent log lines into
    auto-queued fix requests."""
    p = globals().get("_log_file_path")
    return p if isinstance(p, str) and p else None


# Per-camera read-failure summary populated by _face_tracking_thread on every
# read attempt. Self-diagnostic auto-queue tails this to spot read-failure
# spikes (consecutive >5) without needing to walk the running face thread's
# private state. Lock-protected because the tracking thread updates it from a
# different thread than the diagnostic probe that reads it.
_camera_failure_summary: dict[int, dict] = {}
_camera_failure_summary_lock = threading.Lock()


def _note_camera_read_attempt(cam_index: int, *, ok: bool,
                              fails: int = 0,
                              error: str | None = None) -> None:
    """Called by _face_tracking_thread after every cap.read() so the self-
    diagnostic probe can detect read-failure spikes. Records consecutive
    failure count and, separately, the all-time-max for this session so a
    transient spike that has since recovered is still visible in the report."""
    try:
        now = time.time()
        with _camera_failure_summary_lock:
            entry = _camera_failure_summary.get(cam_index)
            if entry is None:
                entry = {
                    "consecutive_fails": 0,
                    "max_consecutive_fails": 0,
                    "last_error": None,
                    "last_error_at": 0.0,
                    "last_ok_at": 0.0,
                    "total_fails": 0,
                }
                _camera_failure_summary[cam_index] = entry
            if ok:
                entry["consecutive_fails"] = 0
                entry["last_ok_at"] = now
            else:
                entry["consecutive_fails"] = int(fails)
                entry["total_fails"] += 1
                if entry["consecutive_fails"] > entry["max_consecutive_fails"]:
                    entry["max_consecutive_fails"] = entry["consecutive_fails"]
                if error:
                    entry["last_error"] = str(error)[:240]
                entry["last_error_at"] = now
    except Exception:
        pass


def get_camera_failure_summary() -> dict[int, dict]:
    """Snapshot copy of the per-camera failure summary. Self-diagnostic uses
    this to detect read-failure spikes worth queueing a fix for."""
    with _camera_failure_summary_lock:
        return {idx: dict(v) for idx, v in _camera_failure_summary.items()}


def _load_patterns() -> list[dict]:
    if not os.path.exists(PATTERNS_FILE):
        return []
    try:
        with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_patterns(entries: list[dict]) -> None:
    try:
        os.makedirs(PATTERNS_DIR, exist_ok=True)
        tmp = PATTERNS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries[-MAX_PATTERN_ENTRIES:], f, indent=2)
        os.replace(tmp, PATTERNS_FILE)
    except Exception as e:
        print(f"  [patterns] save failed: {e}")


def record_action_history(action_name: str, arg: str, result: str) -> None:
    """Append an executed action to the bounded replay deque.

    Called from parse_and_run_actions after a non-confirmation action runs
    (success or in-action failure both count — the user can replay either).
    Skipped when the replay handler itself fires the action so 'do that
    again, do that again' doesn't recurse into itself.
    """
    with _action_history_lock:
        _action_history.append({
            "action": action_name,
            "arg": arg,
            "result": result if isinstance(result, str) else str(result),
            "at": time.time(),
        })


def record_session_action(action_name: str, arg: str = "") -> None:
    """Called from parse_and_run_actions for every executed action so we can
    track which actions dominate this session."""
    with _session_pattern_lock:
        _session_action_counts[action_name] = _session_action_counts.get(action_name, 0) + 1
        # Track app/url names from launch_app + open_url
        if action_name == "launch_app" and arg:
            _session_app_names.add(arg.strip().lower()[:40])
        elif action_name == "open_url" and arg:
            host = re.sub(r"^https?://", "", arg.strip()).split("/")[0].lower()
            if host:
                _session_app_names.add(host[:40])
    # Refresh the JARVIS-initiated music timestamp so ambient-music standby
    # detection doesn't fire on audio that's coming out of our own speakers.
    if action_name in MUSIC_ACTION_NAMES:
        _jarvis_played_music_at[0] = time.time()
    # Forward to the pattern_learning skill so it can mine action-level
    # habits for proactive offers. Skill is optional; absent module → no-op.
    _pl = sys.modules.get("skill_pattern_learning")
    if _pl is not None:
        try:
            _pl.log_event(action_name, arg)
        except Exception as _e:
            print(f"  [pattern_learning] log_event failed: {_e}")


def save_session_pattern() -> None:
    """At shutdown, append this session's stats to patterns.json."""
    with _session_pattern_lock:
        if not _session_action_counts and not _session_app_names:
            return
        # Sort actions by count desc, take top 5
        top_actions = sorted(_session_action_counts.items(),
                             key=lambda x: x[1], reverse=True)[:5]
        start_lt = time.localtime(_session_start_time)
        end_lt   = time.localtime()
        entry = {
            "date":         time.strftime("%Y-%m-%d", start_lt),
            "day":          time.strftime("%A", start_lt),
            "hour_started": start_lt.tm_hour,
            "hour_ended":   end_lt.tm_hour,
            "top_actions":  [name for name, _ in top_actions],
            "action_counts": dict(top_actions),
            "apps":         sorted(_session_app_names),
        }
    entries = _load_patterns()
    entries.append(entry)
    _save_patterns(entries)
    print(f"  [patterns] saved session: {entry['day']} "
          f"{entry['hour_started']:02d}:00–{entry['hour_ended']:02d}:00, "
          f"top: {entry['top_actions']}")


def detect_startup_pattern() -> str:
    """Look for a pattern matching current day/hour and return a JARVIS-style
    proactive remark, or '' if no strong pattern is found.

    Strategy:
      • Need at least MIN_SESSIONS_FOR_PATTERN total sessions.
      • Among entries within ±1 hour of current hour AND same day-of-week,
        count what dominates. If 3+ entries match, surface their dominant
        signal.
      • Distinguish 'winding down' (late-evening, streaming actions) from
        'building' (any time, dev actions / Claude Code launches).
    """
    entries = _load_patterns()
    if len(entries) < MIN_SESSIONS_FOR_PATTERN:
        return ""

    now = time.localtime()
    cur_day  = time.strftime("%A", now)
    cur_hour = now.tm_hour

    matching = [
        e for e in entries
        if e.get("day") == cur_day and abs(e.get("hour_started", 0) - cur_hour) <= 1
    ]
    if len(matching) < 3:
        return ""

    # Tally top actions across matching sessions
    tally: dict[str, int] = {}
    for e in matching:
        for a in e.get("top_actions", [])[:3]:
            tally[a] = tally.get(a, 0) + 1
    if not tally:
        return ""

    STREAMING = {"netflix", "prime_video", "disney_plus", "hulu", "max",
                 "apple_music", "spotify", "youtube_play", "play_streaming",
                 "play_music"}
    BUILDING  = {"upgrade", "queue_task", "see_screen", "create_skill",
                 "start_overnight_upgrade"}

    streaming_count = sum(tally.get(a, 0) for a in STREAMING)
    building_count  = sum(tally.get(a, 0) for a in BUILDING)

    is_evening = cur_hour >= 19 or cur_hour < 4

    if streaming_count >= 3 and is_evening:
        return (f"It's {cur_day} evening, sir. Based on your recent pattern, "
                f"you usually wind down around now — shall I queue something "
                f"on Netflix?")
    if building_count >= 3:
        return (f"You typically start a build session around this time, sir. "
                f"Claude Code is available if you'd like to kick one off.")
    return ""


def save_session_to_memory(memory: dict):
    """On shutdown: write a one-sentence summary of this session."""
    if len(conversation_history) < 4:
        return
    print("\nSummarising session…")
    # Snapshot first — list() is an atomic copy under the GIL, so a background
    # append (pending-speech / proactive-alert thread) during shutdown can't
    # change the list size mid-iteration and raise RuntimeError.
    transcript = "\n".join(
        f"{m['role'].title()}: {m['content']}" for m in list(conversation_history)
    )
    try:
        text = _llm_quick(
            system="Summarise this conversation in ONE sentence. Just the sentence, nothing else.",
            user=transcript, max_tokens=80,
        )
        summary = text.strip().split("\n")[0]
        if summary:
            session_entry = {
                "date":     time.strftime("%Y-%m-%d"),
                "location": LOCATION,
                "summary":  summary,
            }
            # CRITICAL: persist via a FRESH locked read-modify-write — do NOT
            # save the long-lived `memory` dict. That dict is the startup
            # snapshot; every fact/project/topic the background extractor and
            # learn_from_turn merged this session was written to disk by
            # merge_memory() under _memory_lock but is ABSENT from this stale
            # copy. Saving it here overwrote bobert_memory.json with the old
            # snapshot, discarding a whole session of learning on every clean
            # shutdown (force-kills skipped this path, which masked it).
            # Mutate ONLY `sessions` on the current on-disk state. 2026-05-30 audit.
            with _memory_lock:
                _fresh = load_memory()
                _fresh.setdefault("sessions", []).append(session_entry)
                _fresh["sessions"] = _fresh["sessions"][-MAX_SESSIONS:]
                save_memory(_fresh)
            memory["sessions"] = _fresh["sessions"]   # keep local copy consistent
            print(f"Saved session: {summary}")
            # Also append to the queryable session-summary index so the
            # session_memory_recall action can find it later.
            try:
                pattern_memory.record_session_summary(
                    summary,
                    start_ts=_session_start_time,
                    end_ts=time.time(),
                    location=LOCATION,
                )
            except Exception as _e:
                print(f"  [pattern_memory] session-summary append failed: {_e}")
    except Exception as e:
        print(f"Session save failed: {e}")


# Interval between in-session summary checkpoints. The shutdown-only writer
# (save_session_to_memory) almost never ran because JARVIS is usually
# force-killed / crashes / killed by the upgrade pipeline — all of which bypass
# the clean-exit hook, leaving session_summaries.json empty (2026-05-30 audit).
# This periodic checkpoint persists a summary mid-session so a crash loses at
# most ~10 minutes of recall context. record_session_summary() is idempotent
# per session (keyed on iso_start), so each checkpoint UPDATES one entry rather
# than appending duplicates.
_SESSION_CHECKPOINT_INTERVAL_S = 600          # 10 minutes
_session_checkpoint_last_len   = [0]          # conversation_history len at last checkpoint


def _session_summary_checkpoint_thread():
    """Background daemon: periodically write a one-sentence session summary so
    the recall index survives an unclean exit. Skips work when the conversation
    hasn't grown since the last checkpoint (no new turns → no new LLM cost)."""
    # Give the session a chance to accumulate a few turns before first write.
    time.sleep(_SESSION_CHECKPOINT_INTERVAL_S)
    while True:
        try:
            # Snapshot under the GIL (list() copy is atomic) so the main thread
            # appending/popping conversation_history mid-iteration can't raise
            # "list changed size during iteration" in this background thread.
            _hist = list(conversation_history)
            hist_len = len(_hist)
            # Only checkpoint when there's something new worth summarising.
            if hist_len >= 4 and hist_len != _session_checkpoint_last_len[0]:
                transcript = "\n".join(
                    f"{m['role'].title()}: {m['content']}"
                    for m in _hist
                )
                try:
                    text = _llm_quick(
                        system=("Summarise this conversation in ONE sentence. "
                                "Just the sentence, nothing else."),
                        user=transcript, max_tokens=80,
                    )
                    summary = (text or "").strip().split("\n")[0]
                    if summary:
                        pattern_memory.record_session_summary(
                            summary,
                            start_ts=_session_start_time,
                            end_ts=time.time(),
                            location=LOCATION,
                        )
                        _session_checkpoint_last_len[0] = hist_len
                        print(f"  [session-checkpoint] saved: {summary[:80]}")
                except Exception as _e:
                    print(f"  [session-checkpoint] summary failed: {_e}")
        except Exception as _e:
            print(f"  [session-checkpoint] loop error: {_e}")
        time.sleep(_SESSION_CHECKPOINT_INTERVAL_S)


# ──────────────────────────────────────────────────────────────────────────
#  LOGGING  — write everything stdout/stderr to a timestamped file
# ──────────────────────────────────────────────────────────────────────────

class _TimestampedTee:
    """Mirror writes to console AND a log file. Log file gets timestamps;
    console output is left untouched so the live terminal stays clean."""
    def __init__(self, console, log_file):
        self.console     = console
        self.log_file    = log_file
        self._line_start = True

    def write(self, msg):
        # Console: as-is
        try:
            self.console.write(msg)
            self.console.flush()
        except Exception:
            pass
        # File: prepend timestamp at the start of every line
        try:
            for ch in msg:
                if self._line_start and ch != "\n":
                    self.log_file.write(time.strftime("[%H:%M:%S] "))
                    self._line_start = False
                self.log_file.write(ch)
                if ch == "\n":
                    self._line_start = True
            self.log_file.flush()
        except Exception:
            pass

    def flush(self):
        try: self.console.flush()
        except Exception: pass
        try: self.log_file.flush()
        except Exception: pass


_log_file_handle = None
_log_file_path   = None


def _cleanup_old_logs():
    if not os.path.isdir(LOGS_DIR):
        return
    try:
        logs = sorted(
            (os.path.join(LOGS_DIR, f) for f in os.listdir(LOGS_DIR) if f.endswith(".log")),
            key=os.path.getmtime,
        )
        for old in logs[:-LOG_KEEP_COUNT]:
            try: os.unlink(old)
            except Exception: pass
    except Exception:
        pass


def setup_logging():
    """Redirect stdout/stderr to a Tee that also writes to a timestamped log
    file. Install a global exception hook so crashes are captured too."""
    global _log_file_handle, _log_file_path
    if not LOGGING_ENABLED:
        return
    os.makedirs(LOGS_DIR, exist_ok=True)
    _cleanup_old_logs()

    fname = f"session_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    _log_file_path   = os.path.join(LOGS_DIR, fname)
    _log_file_handle = open(_log_file_path, "w", encoding="utf-8", buffering=1)

    # Header at the top of the log
    _log_file_handle.write(
        f"=== J.A.R.V.I.S. session log ===\n"
        f"Started:  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Location: {LOCATION}\n"
        f"Python:   {sys.version.splitlines()[0]}\n"
        f"Platform: {sys.platform}\n"
        f"==========================\n\n"
    )

    sys.stdout = _TimestampedTee(sys.__stdout__, _log_file_handle)
    sys.stderr = _TimestampedTee(sys.__stderr__, _log_file_handle)

    # Catch anything that escapes a try/except so crashes end up in the log
    def _excepthook(exc_type, exc_value, exc_tb):
        import traceback
        print("\n[FATAL] Uncaught exception:", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
        sys.stderr.flush()
    sys.excepthook = _excepthook

    # task-68: silent crashes (PIDs 21424, 37600, 43380, 45708) all died
    # without a Python traceback because the crash was native — OpenCV
    # release/reopen, PortAudio race, comtypes-on-Bambu-MQTT timeout
    # cascade etc. The standard excepthook above only catches Python
    # exceptions; SIGSEGV in a C extension bypasses it entirely.
    # faulthandler.enable() dumps a Python+native stack on SIGSEGV/SIGABRT
    # so the next silent crash actually leaves a trace.
    try:
        import faulthandler
        # _TimestampedTee doesn't expose fileno(), so route faulthandler
        # directly to a dedicated crash file. Open on a real OS fd so
        # SIGSEGV in a C extension can write to it after Python state
        # is unsafe to touch.
        _crash_log_path = os.path.join(LOGS_DIR, "crash_traces.log")
        _crash_fd = open(_crash_log_path, "ab", buffering=0)
        faulthandler.enable(file=_crash_fd, all_threads=True)
        print(f"  [faulthandler] enabled — native crashes -> {_crash_log_path}")
    except Exception as _fh_e:
        print(f"  [faulthandler] not available: {_fh_e}")


def close_log():
    """Called at clean shutdown to write a footer and close the file."""
    if _log_file_handle is None:
        return
    try:
        _log_file_handle.write(f"\n=== Session ended {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        _log_file_handle.flush()
        _log_file_handle.close()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────
#  ROBOT CONTROL
# ──────────────────────────────────────────────────────────────────────────

def send(**kwargs):
    if not ROBOT_ENABLED:
        return
    try:
        requests.get(ROBOT_URL, params=kwargs, timeout=0.4)
    except Exception:
        pass


# Tracks the most recently set high-level state so action-dispatch code can
# revert now_doing back to the right label after an EXECUTING: ... burst.
_current_state_label = ["Idle"]


def _now_doing_label(state: str) -> str:
    """Build the now_doing string for the holographic now_doing ring.
    Thinking includes the active LLM model name so the user can tell at a
    glance whether Claude or local LLM is on the wire (jarvis_todo.md
    2026-05-29 18:05 — 'are you working' confusion fix)."""
    s = (state or "idle").lower()
    if s == "thinking":
        model = CLAUDE_MODEL if AI_BACKEND == "claude" else OLLAMA_MODEL
        return f"THINKING ({model})"
    if s == "listening":
        return "LISTENING"
    if s == "speaking":
        return "SPEAKING"
    if s == "standby":
        return "STANDBY"
    if s == "sleep":
        return "SLEEP"
    return "IDLE"


def set_state(state: str):
    if state == "idle":
        # If standby is active, surface that in the HUD instead of plain Idle —
        # the robot pose is the same idle neutral, but the label reads 'Standby'.
        if _standby_mode[0]:
            send(eyes_x=0.5, eyes_y=0.5, leds="off")
            _current_state_label[0] = "Standby"
            _write_hud_state(state="Standby",
                             now_doing=_now_doing_label("standby"))
            return
        send(eyes_x=0.5, eyes_y=0.5, leds="white")
    elif state == "listening":
        send(eyes_x=0.5, eyes_y=0.25, leds="white")   # eyes up = attentive
    elif state == "thinking":
        send(eyes_x=0.5, eyes_y=0.65, leds="off")      # eyes down = processing
    elif state == "speaking":
        send(eyes_x=0.5, eyes_y=0.35, leds="pink")
    elif state == "sleep":
        send(eyes_x=0.5, eyes_y=0.5, left_arm=0.0, right_arm=0.0, leds="off", mouth=0.0)
    # Mirror state into the HUD file so jarvis_hud.py can render it
    _current_state_label[0] = state.capitalize()
    _write_hud_state(state=state.capitalize(),
                     now_doing=_now_doing_label(state))


# ──────────────────────────────────────────────────────────────────────────
#  HUD STATE  — written every set_state() so the overlay can read it
# ──────────────────────────────────────────────────────────────────────────

HUD_STATE_FILE = _BLUE_GREEN_PATHS["hud_state_file"]
_hud_state_lock = threading.Lock()
_hud_state_cache: dict = {
    "state": "Idle", "now_playing": "", "timers": [],
    "mic_level": 0.0, "active_action": "",
    # now_doing — single-string realtime status for the holographic HUD's
    # now_doing ring (jarvis_todo.md 2026-05-29 18:05). Combines current
    # high-level state with EXECUTING: <action> while a handler is on the
    # wire. Written by set_state() and parse_and_run_actions().
    "now_doing": "IDLE",
    # Ticker fields (jarvis_todo.md 2026-05-27 06:15 — ticker line):
    # recent_action remains set after an action completes so the HUD can
    # show 'last: X (Ns ago)' even when no action is currently running.
    "recent_action": "", "recent_action_at": 0.0,
    # Epoch seconds when .overnight_active expires. 0 when overnight isn't
    # active. The HUD reads this to render the countdown timer.
    "overnight_expiry": 0.0,
    # Last heard user transcript — published from the main loop right after
    # whisper validates an utterance. Surfaces as the S radial readout.
    "last_transcript": "", "last_transcript_at": 0.0,
    # Rolling 5-entry transcript history — most-recent-last. Drives the
    # holographic HUD v2 scrolling transcript panel (jarvis_todo.md
    # 2026-05-29 09:18 holographic_hud_v2). Appended to in the main loop
    # alongside last_transcript; capped at 5 entries on write.
    "transcript_history": [],
    # Last [intent:xxx] tag parsed from the most recent LLM reply (see
    # _parse_intent_tag). Surfaces in the holographic HUD v2 INTENT row.
    "last_intent_tag": "",
    # Last spoken JARVIS reply — published from _speak() right before TTS.
    # Surfaces in the holographic overlay's center text panel (the "what
    # JARVIS just said" half of the spec for jarvis_todo.md 2026-05-27 09:18).
    "last_spoken": "", "last_spoken_at": 0.0,
    # TTS playback amplitude (0.0–1.0) — published from play_with_lipsync()
    # at ~30 Hz during speech. Drives the speaking-state ring brightness in
    # the arc-reactor HUD (jarvis_todo.md 2026-05-27 07:31, arc_reactor_hud).
    "tts_amplitude": 0.0,
    # System-tray applet (tray.py) status flags. Republished by a tiny
    # background thread that polls the system_monitor + bambu_monitor
    # skill modules — they live in separate skill_* modules and don't
    # import bobert_companion directly, so this publisher is the bridge.
    "alert_active": False,    # sustained CPU/RAM alert from skills/system_monitor
    "bambu_active": False,    # Bambu H2D mid-print (gcode_state == RUNNING)
}
_last_mic_hud_write = [0.0]  # throttle audio→HUD writes to ~10Hz


def _write_hud_state(**updates):
    """Merge updates into the HUD state cache and atomically write it.
    Silent on any failure — HUD is a nice-to-have, not load-bearing."""
    if not HUD_ENABLED:
        return
    try:
        # Whole read-modify-write under the lock. Previously only the cache
        # merge was locked; the file write used a FIXED shared temp
        # (HUD_STATE_FILE + ".tmp") OUTSIDE the lock, so two of the ~30
        # concurrent callers (amp pump @30Hz, tray publisher, _speak, main
        # loop) could truncate the same temp mid-dump and os.replace a half-
        # written file — a corrupt hud_state.json. A per-call mkstemp temp plus
        # the serialized replace removes both the collision and the lost update.
        with _hud_state_lock:
            _hud_state_cache.update(updates)
            data = dict(_hud_state_cache)
            data["updated_at"] = time.time()
            _dir = os.path.dirname(os.path.abspath(HUD_STATE_FILE)) or "."
            _fd, _tmp = tempfile.mkstemp(dir=_dir, prefix=".hud_", suffix=".tmp")
            try:
                with os.fdopen(_fd, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(_tmp, HUD_STATE_FILE)
            except Exception:
                try:
                    if os.path.exists(_tmp):
                        os.remove(_tmp)
                except Exception:
                    pass
                raise
    except Exception:
        pass


_hud_process = None


def _launch_hud():
    """Spawn the unified HUD (hud/jarvis_unified_hud.py) pinned to HUD_MONITOR.
    The HUD auto-exits when its parent (us) dies, so we don't strictly need to
    manage it on shutdown, but we still call terminate() in main() for clean
    closes.

    2026-05-30: repointed from the old slim corner ring (hud/jarvis_hud.py) to
    the unified, draggable/resizable, feature-packed HUD. The unified HUD
    ignores extra launcher args (--role/--state-file) via parse_known_args and
    restores its own saved geometry, so the blue/green per-role plumbing below
    is harmless — the saved position simply wins on a normal boot."""
    global _hud_process
    if not HUD_ENABLED:
        return
    hud_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "hud", "jarvis_unified_hud.py")
    if not os.path.exists(hud_path):
        print(f"  [hud] script missing at {hud_path} — skipping")
        return
    # Honour the manager's per-role monitor pick when blue/green is active.
    monitor_name = HUD_MONITOR
    try:
        if _bgm is not None:
            monitor_name = (_BLUE_GREEN_PATHS.get("monitor_name")
                            or HUD_MONITOR)
    except Exception:
        pass
    mon = MONITORS.get(monitor_name)
    if not mon:
        # Defensive fallback — staging on a single-monitor laptop falls
        # back to the primary monitor rather than crashing _launch_hud.
        mon = MONITORS.get("top") or MONITORS.get("middle") or (0, 0, 1920, 1080)
    mx, my, mw, mh = (list(mon) + [0, 0, 1920, 1080])[:4]

    # First-launch geometry for the unified HUD: a portrait panel tucked into
    # the top-right corner of the chosen monitor, out of the central work area.
    # (The old corner-ring wanted the full monitor width; the unified HUD is a
    # portrait card, so we pass a fixed size instead of mw.) After the user
    # drags/resizes it once, the HUD's own saved geometry
    # (unified_hud_geometry.json) overrides these — this only decides where it
    # first appears.
    HUD_W, HUD_H = 430, 580
    hud_x = int(mx) + int(mw) - HUD_W - 40
    hud_y = int(my) + 40

    # Pass the blue/green role + per-role state file so the HUD can render
    # its STAGING badge and test-progress overlay without inspecting argv
    # globally. Both args default to safe values inside the HUD script so
    # an older HUD binary keeps working.
    role = BLUE_GREEN_ROLE
    state_file = _BLUE_GREEN_PATHS.get("hud_state_file", "")

    try:
        argv = [sys.executable, hud_path,
                "--x", str(hud_x), "--y", str(hud_y),
                "--width", str(HUD_W), "--height", str(HUD_H),
                "--parent-pid", str(os.getpid()),
                "--role", role]
        if state_file:
            argv.extend(["--state-file", state_file])
        env = os.environ.copy()
        env["BLUE_GREEN_ROLE"] = role
        _hud_process = subprocess.Popen(
            argv,
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
            close_fds=True,
            env=env,
        )
        print(f"  [hud] launched on {monitor_name} monitor "
              f"(role={role}, pid {_hud_process.pid})")
    except Exception as e:
        print(f"  [hud] launch failed: {e}")
        _hud_process = None


# ──────────────────────────────────────────────────────────────────────────
#  RETICLE OVERLAY  — full-virtual-screen target reticle for UI automation
# ──────────────────────────────────────────────────────────────────────────

RETICLE_STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hud_reticles.json"
)
_reticle_lock = threading.Lock()
_reticle_process = None
# Spec: 2-second reticle. Keep entries on disk a touch longer than the TTL so
# the overlay's read-loop never races a just-pruned write.
_RETICLE_DISK_TTL = 3.0


def _publish_reticle(x, y, label: str = ""):
    """Record a UI-automation event for the reticle overlay to draw.

    Coordinates are virtual-desktop pixels (may be negative on monitors
    placed left/above the primary). Best-effort: failures are silent so a
    write-locked or missing state file can never break a UI action."""
    if not RETICLE_OVERLAY_ENABLED:
        return
    try:
        x_i = int(x)
        y_i = int(y)
    except (TypeError, ValueError):
        return
    try:
        now = time.time()
        with _reticle_lock:
            entries = []
            if os.path.exists(RETICLE_STATE_FILE):
                try:
                    with open(RETICLE_STATE_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    entries = data.get("reticles", []) or []
                    if not isinstance(entries, list):
                        entries = []
                except Exception:
                    entries = []
            # Prune anything older than the disk TTL so the file doesn't grow.
            entries = [
                r for r in entries
                if (now - float(r.get("created_at", 0) or 0)) < _RETICLE_DISK_TTL
            ]
            entries.append({
                "x": x_i, "y": y_i,
                "label": str(label)[:24],
                "created_at": now,
            })
            tmp = RETICLE_STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"reticles": entries}, f)
            os.replace(tmp, RETICLE_STATE_FILE)
    except Exception:
        pass


def _active_window_center():
    """Return (cx, cy) virtual-desktop coords of the currently focused
    window, or None if it can't be determined. Used for reticles on
    actions that don't have explicit click coords (type/press/hotkey)."""
    try:
        import pygetwindow as gw
    except Exception:
        return None
    try:
        w = gw.getActiveWindow()
        if w is None:
            return None
        try:
            cx = int(w.left + w.width / 2)
            cy = int(w.top + w.height / 2)
        except Exception:
            return None
        # Clamp degenerate sizes (Win32 sometimes returns 0 width for
        # minimized windows).
        if cx == 0 and cy == 0:
            return None
        return cx, cy
    except Exception:
        return None


def _virtual_screen_bounds():
    """Compute (x, y, w, h) covering every monitor in MONITORS. Returns a
    safe default if MONITORS is empty/misconfigured."""
    xs = []
    ys = []
    for m in MONITORS.values():
        try:
            mx, my, mw, mh = m
            xs.extend([mx, mx + mw])
            ys.extend([my, my + mh])
        except Exception:
            continue
    if not xs or not ys:
        return 0, 0, 2560, 1440
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _launch_reticle_overlay():
    """Spawn hud/jarvis_reticle.py as a subprocess sized to the virtual
    desktop. Silent on failure so a missing tkinter / unusual geometry
    can never break the main JARVIS startup path."""
    global _reticle_process
    if not RETICLE_OVERLAY_ENABLED:
        return
    overlay_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "hud", "jarvis_reticle.py",
    )
    if not os.path.exists(overlay_path):
        print(f"  [reticle] script missing at {overlay_path} — skipping")
        return
    vx, vy, vw, vh = _virtual_screen_bounds()
    # Reset the state file so old reticles from a previous session don't
    # render briefly on startup.
    try:
        with open(RETICLE_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"reticles": []}, f)
    except Exception:
        pass
    try:
        _reticle_process = subprocess.Popen(
            [sys.executable, overlay_path,
             "--x", str(vx), "--y", str(vy),
             "--width", str(vw), "--height", str(vh),
             "--parent-pid", str(os.getpid())],
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
            close_fds=True,
        )
        print(f"  [reticle] launched ({vw}x{vh} @ {vx},{vy}, pid {_reticle_process.pid})")
    except Exception as e:
        print(f"  [reticle] launch failed: {e}")
        _reticle_process = None


def _shutdown_reticle_overlay():
    """Terminate the reticle overlay subprocess on clean JARVIS exit."""
    global _reticle_process
    if _reticle_process is None:
        return
    try:
        _reticle_process.terminate()
    except Exception:
        pass
    _reticle_process = None


def _shutdown_hud():
    """Terminate the HUD subprocess on clean JARVIS exit."""
    global _hud_process
    if _hud_process is None:
        return
    try:
        _hud_process.terminate()
        _hud_process = None
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────
#  SYSTEM TRAY APPLET  (tray.py — animated arc-reactor icon + menu)
# ──────────────────────────────────────────────────────────────────────────

TRAY_COMMANDS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tray_commands.json"
)
_tray_process = None
_tray_drain_stop = threading.Event()
_tray_publisher_stop = threading.Event()
# Last time the HUD calendar/unread-mail were published to hud_state.json.
# The unified HUD subprocess can't import ms_graph itself (it would pull in
# this module + trip the singleton), so the main process fetches + publishes
# them here, throttled. 2026-05-30.
_hud_cal_last = [0.0]


def _launch_tray():
    """Spawn tray.py as a subprocess. Silent on failure so a missing pystray
    install can't break the rest of startup — check_dependencies() already
    warns the user with a spoken alert if pystray is unimportable."""
    global _tray_process
    if not TRAY_ENABLED:
        return
    tray_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "tray.py")
    if not os.path.exists(tray_path):
        print(f"  [tray] script missing at {tray_path} — skipping")
        return
    # Reset the command inbox so commands from a previous session don't
    # fire on startup of the new one. Also clear any .inflight file left
    # behind by an unclean shutdown mid-drain.
    for _stale in (TRAY_COMMANDS_FILE, TRAY_COMMANDS_FILE + ".inflight"):
        try:
            if os.path.exists(_stale):
                os.remove(_stale)
        except Exception:
            pass
    try:
        _tray_process = subprocess.Popen(
            [sys.executable, tray_path,
             "--parent-pid", str(os.getpid())],
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
            close_fds=True,
        )
        print(f"  [tray] launched (pid {_tray_process.pid})")
        # Seed hud_state.json with the current audio-processing flags so the
        # tray's Audio Controls checkmarks reflect reality on first open.
        _publish_audio_state()
    except Exception as e:
        print(f"  [tray] launch failed: {e}")
        _tray_process = None


def _shutdown_tray():
    """Terminate the tray subprocess on clean JARVIS exit and signal any
    background drainer/publisher threads to stop."""
    global _tray_process
    _tray_drain_stop.set()
    _tray_publisher_stop.set()
    if _tray_process is None:
        return
    try:
        _tray_process.terminate()
    except Exception:
        pass
    _tray_process = None


def _process_inflight(inflight_path: str) -> int:
    """Read, parse, and dispatch the command list stored at inflight_path.
    Deletes the file before dispatch so relaunch-style commands cannot
    re-fire on next boot. Errors are logged but never propagate."""
    try:
        with open(inflight_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            try: os.remove(inflight_path)
            except Exception: pass
            return 0
        try:
            cmds, _ = json.JSONDecoder().raw_decode(raw)
        except Exception:
            print("  [tray] corrupt command file — deleting")
            try: os.remove(inflight_path)
            except Exception: pass
            return 0
        if not isinstance(cmds, list):
            try: os.remove(inflight_path)
            except Exception: pass
            return 0
    except Exception as e:
        print(f"  [tray] read failed: {e}")
        try: os.remove(inflight_path)
        except Exception: pass
        return 0

    # Remove the claim file BEFORE dispatch so commands that trigger
    # relaunch (restart, upgrade) don't re-fire after the new process
    # boots and sees a stale inbox.
    try: os.remove(inflight_path)
    except Exception: pass

    n = 0
    for entry in cmds:
        if not isinstance(entry, dict):
            continue
        cmd = entry.get("cmd", "")
        try:
            _dispatch_tray_command(cmd, entry)
            n += 1
        except Exception as e:
            print(f"  [tray] dispatch failed for {cmd!r}: {e}")
    return n


def _drain_tray_commands_once() -> int:
    """Atomically claim tray_commands.json, dispatch its commands, then
    delete the claim file. Returns the number of commands processed.
    Errors are logged but never propagate — the tray inbox is a
    best-effort channel.

    Atomic claim avoids a race where a tray click landing between read
    and delete would be lost: os.replace() renames the inbox to
    .inflight in one step, so any new write from tray.py's _send_command
    lands in a fresh tray_commands.json instead of being clobbered.

    Orphaned-inflight recovery: if a previous drain crashed mid-dispatch
    an .inflight file may still exist. Windows os.replace() overwrites
    its destination, so we must finish the orphaned batch BEFORE
    claiming any newly-arrived tray_commands.json — otherwise the
    claim would silently discard the orphaned commands. Any new
    commands that land during orphan processing will be picked up on
    the next drain tick."""
    inflight = TRAY_COMMANDS_FILE + ".inflight"

    # Process an orphaned inflight from a previous crash first. Doing
    # this before the os.replace() claim below prevents the new batch
    # from silently overwriting the orphan on Windows.
    n = 0
    if os.path.exists(inflight):
        n += _process_inflight(inflight)

    if not os.path.exists(TRAY_COMMANDS_FILE):
        return n

    try:
        os.replace(TRAY_COMMANDS_FILE, inflight)
    except Exception as e:
        print(f"  [tray] claim failed: {e}")
        return n

    n += _process_inflight(inflight)
    return n


def _dispatch_tray_command(cmd: str, entry: dict) -> None:
    """Route a single tray command to the matching JARVIS action handler."""
    if cmd == "enter_standby":
        _sleep_mode[0]   = True
        _standby_mode[0] = True
        # task-70: persist so a bounce comes back in standby instead of
        # "Listening…" and answering the next utterance out loud.
        _write_hud_state(state="Standby",
                         sleep_mode=True, standby_mode=True)
        print("  [tray] enter_standby — sleep + standby flags set")
    elif cmd == "force_wake":
        # Serialize with the background standby auto-engage daemon (same lock
        # the wake-word path uses) so an auto-engage firing the instant after
        # we clear these flags can't re-assert standby and leave JARVIS stuck
        # asleep with no spoken-wake recourse. 2026-05-30 deep audit.
        with _standby_auto_engage_lock:
            _sleep_mode[0]   = False
            _standby_mode[0] = False
        _write_hud_state(sleep_mode=False, standby_mode=False)
        # Mirror context_aware_greeting's bookkeeping so morning_handoff and
        # any other consumer that watches _last_wake_date sees the tray wake
        # as the day's first wake event.
        try:
            from datetime import datetime as _dt
            _last_wake_date[0] = _dt.now().date().isoformat()
        except Exception:
            pass
        # Same overnight-flag cleanup the wake-word path uses, so a wake from
        # the tray during overnight mode stops the autonomous engine cleanly.
        try:
            if os.path.exists(OVERNIGHT_FLAG_FILE):
                os.remove(OVERNIGHT_FLAG_FILE)
                _write_hud_state(overnight_expiry=0.0)
        except Exception:
            pass
        _write_hud_state(state="Idle")
        try: _speak("At your service, sir.")
        except Exception: pass
        print("  [tray] force_wake — cleared sleep/standby")
    elif cmd == "open_hud":
        # Terminate the existing HUD (if any) then re-launch so a hidden /
        # crashed window comes back. _launch_hud is a no-op if HUD_ENABLED
        # is False — flip the flag on for this invocation if the user is
        # explicitly asking to see it.
        global HUD_ENABLED
        _shutdown_hud()
        if not HUD_ENABLED:
            HUD_ENABLED = True
        _launch_hud()
        print("  [tray] open_hud — re-launched HUD subprocess")
    elif cmd == "restart":
        print("  [tray] restart — relaunching JARVIS")
        try: _act_restart()
        except Exception as e:
            print(f"  [tray] restart action failed: {e}")
    elif cmd == "trigger_overnight":
        print("  [tray] trigger_overnight — starting overnight engine")
        try: _act_start_overnight_upgrade()
        except Exception as e:
            print(f"  [tray] overnight action failed: {e}")
    elif cmd == "audio_processing_toggle":
        _audio_master_enabled[0] = not _audio_master_enabled[0]
        _publish_audio_state()
        print(f"  [tray] audio_processing_toggle -> {_audio_master_enabled[0]}")
    elif cmd == "audio_echo_cancel_toggle":
        _audio_aec_enabled[0] = not _audio_aec_enabled[0]
        _publish_audio_state()
        print(f"  [tray] audio_echo_cancel_toggle -> {_audio_aec_enabled[0]}")
    elif cmd == "audio_noise_suppress_toggle":
        _audio_ns_enabled[0] = not _audio_ns_enabled[0]
        _publish_audio_state()
        print(f"  [tray] audio_noise_suppress_toggle -> {_audio_ns_enabled[0]}")
    elif cmd == "audio_agc_toggle":
        _audio_agc_enabled[0] = not _audio_agc_enabled[0]
        _publish_audio_state()
        print(f"  [tray] audio_agc_toggle -> {_audio_agc_enabled[0]}")
    elif cmd == "mute_tts_toggle":
        _tts_muted[0] = not _tts_muted[0]
        _write_hud_state(tts_muted=bool(_tts_muted[0]))
        print(f"  [tray] mute_tts_toggle -> {_tts_muted[0]}")
    elif cmd == "ambient_mode_toggle":
        _ambient_mode_active[0] = not _ambient_mode_active[0]
        _write_hud_state(ambient_mode_active=bool(_ambient_mode_active[0]))
        # Actually start / stop the ambient_listen daemon so the toggle
        # has runtime effect. ACTIONS may not contain ambient_listen_*
        # if the skill failed to load — fall back to a print so the user
        # still sees something happened.
        action = "ambient_listen_start" if _ambient_mode_active[0] else "ambient_listen_stop"
        fn = ACTIONS.get(action)
        if fn is not None:
            try:
                fn("")
            except Exception as e:
                print(f"  [tray] ambient_mode_toggle action {action!r} raised: {e}")
        else:
            print(f"  [tray] ambient_mode_toggle: {action!r} not registered")
        print(f"  [tray] ambient_mode_toggle -> {_ambient_mode_active[0]}")
    elif cmd == "pause_daemons_toggle":
        _daemons_paused[0] = not _daemons_paused[0]
        _write_hud_state(daemons_paused=bool(_daemons_paused[0]))
        # Mirror the toggle into the diagnostic_daemons paused-state JSON
        # (its four worker loops already check state["paused"]) AND into
        # ambient_listen via skill_ambient_listen.set_paused so its mic /
        # audio / screen loops idle instead of consuming CPU + budget.
        try:
            from core import diagnostic_daemons as _diag_daemons
            if _daemons_paused[0]:
                _diag_daemons.pause_diagnostics()
            else:
                _diag_daemons.resume_diagnostics()
        except Exception as e:
            print(f"  [tray] pause_daemons_toggle diag-daemons failed: {e}")
        try:
            _al = sys.modules.get("skill_ambient_listen")
            if _al is not None and hasattr(_al, "set_paused"):
                _al.set_paused(bool(_daemons_paused[0]))
        except Exception as e:
            print(f"  [tray] pause_daemons_toggle ambient_listen failed: {e}")
        print(f"  [tray] pause_daemons_toggle -> {_daemons_paused[0]}")
    elif cmd == "debug_mode_toggle":
        _debug_mode[0] = not _debug_mode[0]
        _write_hud_state(debug_mode=bool(_debug_mode[0]))
        print(f"  [tray] debug_mode_toggle -> {_debug_mode[0]}")
    elif cmd == "mic_mute_toggle":
        # Mute Mic: drop captured mic input before dispatch (see
        # _capture_utterance) so JARVIS hears nothing and stays idle — distinct
        # from standby (wake-word-only) and Mute TTS (still acts, just silent).
        # Drives the tray's red/muted listen indicator.
        _mic_muted[0] = not _mic_muted[0]
        _write_hud_state(mic_muted=bool(_mic_muted[0]))
        print(f"  [tray] mic_mute_toggle -> {_mic_muted[0]}")
    else:
        # Generic fallthrough — route to a registered ACTIONS handler so the
        # 20+ commands tray.py sends (shutdown_jarvis, run_diagnostic, test_mic,
        # switch_llm, …) actually dispatch instead of dropping with "unknown".
        # switch_llm carries its target in `backend`; everything else passes
        # whatever sits under `arg` (empty string when the tray sent no payload,
        # which matches the bare-string signature every _act_* helper uses).
        fn = ACTIONS.get(cmd)
        if fn is None:
            print(f"  [tray] unknown command: {cmd!r}")
            return
        if cmd == "switch_llm":
            arg = entry.get("backend", "") or ""
        else:
            arg = entry.get("arg", "") or ""
        # Heavy actions (run_diagnostic, test_each_skill, latency_benchmark)
        # may have been replaced by a synchronous skill implementation after
        # load_skills() ran. Spawn into a daemon so a 30–60s sweep can't
        # stall the 2 Hz drainer and back up other tray clicks.
        if cmd in _HEAVY_ACTIONS:
            _tray_async(cmd, lambda f=fn, a=arg: f(a))
            print(f"  [tray] {cmd} dispatched async")
            return
        try:
            result = fn(arg)
        except Exception as e:
            print(f"  [tray] action {cmd!r} raised: {e}")
            return
        head = result if isinstance(result, str) else ""
        if head:
            head = head.split("\n", 1)[0]
            if len(head) > 120:
                head = head[:117].rstrip() + "..."
            print(f"  [tray] {cmd} -> {head}")
        else:
            print(f"  [tray] {cmd} dispatched")


def _publish_audio_state() -> None:
    """Mirror the audio-processing flags to hud_state.json so the tray's
    Audio Controls submenu can render accurate checkmarks. Called on every
    audio_* toggle and once at startup."""
    try:
        _write_hud_state(
            audio_processing_enabled = bool(_audio_master_enabled[0]),
            echo_cancel_enabled      = bool(_audio_aec_enabled[0]),
            noise_suppress_enabled   = bool(_audio_ns_enabled[0]),
            agc_enabled              = bool(_audio_agc_enabled[0]),
        )
    except Exception:
        pass


def _restore_tray_toggle_state() -> None:
    """Read hud_state.json and restore the four tray-toggle cells
    (_tts_muted, _ambient_mode_active, _daemons_paused, _debug_mode) so
    user preference survives a restart. Then re-sync the runtime side:
    if ambient_mode was on, kick the ambient_listen daemon back on; if
    daemons were paused, push that into diagnostic_daemons + skill state.
    Silent on any failure — a missing or corrupt state file just means
    we boot with the in-file defaults."""
    try:
        if not os.path.exists(HUD_STATE_FILE):
            return
        with open(HUD_STATE_FILE, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        if not isinstance(persisted, dict):
            return
    except Exception as e:
        print(f"  [tray-restore] read failed: {e}")
        return

    if "tts_muted" in persisted:
        _tts_muted[0] = bool(persisted.get("tts_muted"))
    if "mic_muted" in persisted:
        _mic_muted[0] = bool(persisted.get("mic_muted"))
    if "ambient_mode_active" in persisted:
        _ambient_mode_active[0] = bool(persisted.get("ambient_mode_active"))
    if "daemons_paused" in persisted:
        _daemons_paused[0] = bool(persisted.get("daemons_paused"))
    if "debug_mode" in persisted:
        _debug_mode[0] = bool(persisted.get("debug_mode"))
    # task-70: sleep/standby survive bounces. A crash + auto-respawn would
    # otherwise come up "Listening…" and answer the next phrase out loud.
    if "sleep_mode" in persisted:
        _sleep_mode[0] = bool(persisted.get("sleep_mode"))
    if "standby_mode" in persisted:
        _standby_mode[0] = bool(persisted.get("standby_mode"))
    # Alexa-style boot: START_IN_STANDBY comes up SILENT in wake-word standby
    # (say "JARVIS" to wake → it answers → back to standby) instead of always-
    # listening — UNLESS a persisted crash-survival sleep state already decided.
    # Env override: JARVIS_START_IN_STANDBY.
    if "sleep_mode" not in persisted:
        from core.config import START_IN_STANDBY as _sis_default
        _sis = (os.environ.get("JARVIS_START_IN_STANDBY", "").strip()
                or str(_sis_default)).strip().lower()
        if _sis in {"1", "true", "yes", "on"}:
            _sleep_mode[0] = True
            _standby_mode[0] = True
    if "audio_processing_enabled" in persisted:
        _audio_master_enabled[0] = bool(persisted.get("audio_processing_enabled"))
    if "echo_cancel_enabled" in persisted:
        _audio_aec_enabled[0] = bool(persisted.get("echo_cancel_enabled"))
    if "noise_suppress_enabled" in persisted:
        _audio_ns_enabled[0] = bool(persisted.get("noise_suppress_enabled"))
    if "agc_enabled" in persisted:
        _audio_agc_enabled[0] = bool(persisted.get("agc_enabled"))

    # Mirror back so the cache holds the restored values and the next
    # _write_hud_state from any other code path doesn't clobber them.
    _write_hud_state(
        tts_muted                = bool(_tts_muted[0]),
        mic_muted                = bool(_mic_muted[0]),
        ambient_mode_active      = bool(_ambient_mode_active[0]),
        daemons_paused           = bool(_daemons_paused[0]),
        debug_mode               = bool(_debug_mode[0]),
        sleep_mode               = bool(_sleep_mode[0]),
        standby_mode             = bool(_standby_mode[0]),
        audio_processing_enabled = bool(_audio_master_enabled[0]),
        echo_cancel_enabled      = bool(_audio_aec_enabled[0]),
        noise_suppress_enabled   = bool(_audio_ns_enabled[0]),
        agc_enabled              = bool(_audio_agc_enabled[0]),
    )
    # Publish the active LLM backend so the tray's AI submenu shows the right
    # checkmark on first open (tray reads `llm_backend`: "anthropic" for Claude,
    # else the ollama model tag it matches via .startswith()).
    try:
        _write_hud_state(
            llm_backend=("anthropic" if str(AI_BACKEND).lower() == "claude"
                         else str(OLLAMA_MODEL)))
    except Exception:
        pass

    # Bring ambient_listen up if the user had it on last time. ACTIONS
    # was populated by load_skills() already.
    if _ambient_mode_active[0]:
        fn = ACTIONS.get("ambient_listen_start")
        if fn is not None:
            try:
                fn("")
                print("  [tray-restore] ambient_listen resumed from prior session")
            except Exception as e:
                print(f"  [tray-restore] ambient_listen_start failed: {e}")

    # Push the paused state into both daemon owners.
    if _daemons_paused[0]:
        try:
            from core import diagnostic_daemons as _diag_daemons
            _diag_daemons.pause_diagnostics()
        except Exception as e:
            print(f"  [tray-restore] pause_diagnostics failed: {e}")
        try:
            _al = sys.modules.get("skill_ambient_listen")
            if _al is not None and hasattr(_al, "set_paused"):
                _al.set_paused(True)
        except Exception as e:
            print(f"  [tray-restore] ambient_listen.set_paused failed: {e}")
        print("  [tray-restore] daemons restored to paused state")

    if any((_tts_muted[0], _ambient_mode_active[0],
            _daemons_paused[0], _debug_mode[0])):
        print(f"  [tray-restore] tts_muted={_tts_muted[0]} "
              f"ambient={_ambient_mode_active[0]} "
              f"paused={_daemons_paused[0]} debug={_debug_mode[0]}")


def _tray_command_drainer():
    """Background poll of tray_commands.json. Runs at 2 Hz so menu clicks
    feel responsive even while the main loop is blocked in record_speech."""
    while not _tray_drain_stop.is_set():
        try:
            _drain_tray_commands_once()
            _tray_drain_stop.wait(0.5)
        except Exception:
            logging.exception("[tray] drainer iteration failed")
            # back off on persistent failures so we don't spin
            _tray_drain_stop.wait(0.5)


def _tray_state_publisher():
    """Bridge the system_monitor + bambu_monitor skill modules to the HUD
    state file so the tray (a separate process) can read their state.

    Both skills load lazily into sys.modules under skill_<name> and don't
    import bobert_companion, so this publisher reads their module-level
    state under their own locks. Runs at 1 Hz — these flags don't change
    fast and an extra poll per second is cheap."""
    while not _tray_publisher_stop.is_set():
        try:
            alert = False
            bambu = False
            # ── system_monitor: alert if a CPU OR RAM alert fired in the last
            #    cooldown window (10 min default). We don't have a direct
            #    "alert_active" flag; piggy-back on the last-alert timestamps.
            try:
                sm = sys.modules.get("skill_system_monitor")
                if sm is not None:
                    last_cpu = (getattr(sm, "_last_cpu_alert_at", [0])[0]
                                if hasattr(sm, "_last_cpu_alert_at") else 0)
                    last_ram = (getattr(sm, "_last_ram_alert_at", [0])[0]
                                if hasattr(sm, "_last_ram_alert_at") else 0)
                    most_recent = max(last_cpu or 0, last_ram or 0)
                    # Show red for 30s after an alert fires, then clear.
                    if most_recent and (time.time() - most_recent) < 30:
                        alert = True
            except Exception:
                pass
            # ── bambu_monitor: print active when gcode_state == RUNNING
            try:
                bm = sys.modules.get("skill_bambu_monitor")
                if bm is not None and hasattr(bm, "_state"):
                    state_dict = getattr(bm, "_state", None)
                    lock = getattr(bm, "_state_lock", None)
                    if isinstance(state_dict, dict):
                        if lock is not None:
                            with lock:
                                gcode = str(state_dict.get("gcode_state") or "")
                        else:
                            gcode = str(state_dict.get("gcode_state") or "")
                        if gcode.upper() == "RUNNING":
                            bambu = True
            except Exception:
                pass
            # ── calendar + unread mail for the unified HUD ──
            # The HUD subprocess can't import ms_graph (it would pull in this
            # 14k-line module and trip the singleton lock), so the main process
            # fetches them via hud_card — safe here — and publishes to
            # hud_state.json. Throttled to every 5 min (ms_graph is networked).
            try:
                _now = time.time()
                if _now - _hud_cal_last[0] > 300.0:
                    _hud_cal_last[0] = _now
                    import hud_card as _hc
                    _events = _hc._gather_calendar() or []
                    _write_hud_state(
                        next_event=(_events[0] if _events else None),
                        unread_mail=_hc._gather_unread_mail(),
                    )
            except Exception:
                pass
            # Only write when something actually changed so we don't churn the
            # state file 1× per second.
            with _hud_state_lock:
                stale_a = _hud_state_cache.get("alert_active")
                stale_b = _hud_state_cache.get("bambu_active")
            if stale_a != alert or stale_b != bambu:
                _write_hud_state(alert_active=alert, bambu_active=bambu)
        except Exception:
            logging.exception("[tray] state publisher iteration failed")
        _tray_publisher_stop.wait(1.0)


# ──────────────────────────────────────────────────────────────────────────
#  FACE TRACKING  (Kinect RGB camera + OpenCV face detection)
# ──────────────────────────────────────────────────────────────────────────

# CUDA face-detection investigation (jarvis_todo 2026-05-29 13:34, closed
# 2026-05-29). Verdict: stay on CPU. cv2.cuda.CascadeClassifier does exist in
# opencv-contrib's CUDA build and the 3090 would happily run it, but:
#   1. The pip-installable `opencv-python` / `opencv-contrib-python` wheels
#      are NOT built with CUDA. Getting cv2.cuda.* requires either a manual
#      CMake/MSVC compile of opencv-contrib against the CUDA toolkit (a
#      multi-hour, fragile install on Windows) or a third-party prebuilt
#      wheel (cuda4opencv, asmaloney/opencv-cuda) that lags upstream and
#      conflicts with the stock wheel. Big regression-surface for a single
#      hot path.
#   2. Measured on this box (3090, OpenCV 4.13.0, opencv-python, no real
#      face in frame so the worst-case 4-pass fallthrough runs):
#         640x480  single strict-frontal pass    ~3.5 ms
#         1280x720 single strict-frontal pass    ~7.2 ms
#         1280x720 frontal+frontal-loose+profile+profile-mirrored ~13.2 ms
#      The skill polls at 2 Hz (face_tracker.GAZE_POLL_INTERVAL=0.5s), so
#      even worst-case CPU work is ~26 ms/sec/camera — invisible next to
#      the rest of the JARVIS loop.
#   3. CUDA Haar at this resolution is dominated by HtoD/DtoH transfer
#      (~270 KB grayscale per frame, ~50-100 µs each way on PCIe 4) and
#      kernel-launch overhead (~10-50 µs per launch × 4 cascade passes).
#      Published benchmarks (OpenCV CUDA module docs, NVIDIA samples) show
#      ~2-4x speedup on 1080p+, dropping below 1x on smaller frames where
#      transfer cost dominates compute — i.e. exactly our regime.
#   4. The plan's gating criterion was ">1 ms improvement justifies the
#      change". Realistic best case is ~5-8 ms saved on the 1280x720 hot
#      pass, which DOES clear the bar — but only after eating the build
#      complexity in (1) and the runtime fallback complexity in (2).
# Decision: keep CPU. A self-contained benchmark harness lives at
# tools/face_detect_bench.py for any future re-evaluation (e.g. if someone
# is already running a CUDA OpenCV build for another reason).
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# Profile (side-view) cascade — the frontal cascade misses faces turned more
# than ~30° off-axis, which is the common case when the user is glancing at a
# side monitor. Used as a fallback when the frontal pass turns up nothing.
_profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
if _face_cascade.empty():  # pragma: no cover - import-time guard; only true if the bundled haarcascade XML is missing/corrupt
    print("  [WARN] Could not load haarcascade_frontalface_default.xml — "
          "face detection disabled. Re-installing opencv-python should fix this.")
    _face_cascade = None
if _profile_cascade is not None and _profile_cascade.empty():  # pragma: no cover - import-time guard; only true if the profile haarcascade XML is missing/corrupt
    _profile_cascade = None

# Shared state written by face-tracking thread, read by main thread control
_face_track_pause = threading.Event()   # set = paused
_face_track_stop  = threading.Event()   # set = thread should exit

_smooth_x = 0.5
_smooth_y = 0.5
_last_sent_x = -1.0
_last_sent_y = -1.0

# Used by proactive idle behavior — last time any camera saw a face
last_face_seen   = 0.0
last_speech_time = time.time()

# Per-camera awareness — used by where_is_user / see_user actions so Bobert
# can tell which monitor you're facing. Updated by the face-tracking thread.
_camera_last_seen: dict[int, float]    = {}            # index → timestamp
_camera_latest_frame: dict[int, "np.ndarray"] = {}     # index → most recent frame
_camera_state_lock = threading.Lock()

# Webcam I/O health bookkeeping (self-diag 2026-05-28). When cv2's read()
# returns no frame the face-tracking thread can't always tell the user what
# went wrong — was the device just unplugged? Did Teams steal it mid-stream?
# Did the USB pipe stall? We track the last raw failure reason + when the
# device last yielded a frame so see_user / self-diagnostic can surface
# actionable detail instead of just "no frame".
_camera_last_read_error: dict[int, str]      = {}      # index → last error
_camera_last_read_error_at: dict[int, float] = {}      # index → ts
_camera_last_frame_at: dict[int, float]      = {}      # index → ts of last good frame
_camera_wake_attempts: dict[int, int]        = {}      # index → wake attempts since last good frame
_camera_recoveries: dict[int, int]           = {}      # index → cumulative successful wakes

# Serializes every cv2.VideoCapture open / release across threads (probe sweep,
# face-tracking, list-cameras, snapshot). DirectShow heap-corrupts when an
# abandoned probe worker's release() overlaps another thread's release(), so
# every cv2 capture handle is brought up and torn down inside this lock.
# RLock so a single thread can do open + release in one critical section.
_camera_io_lock = threading.RLock()


def _detect_face(frame_bgr: np.ndarray) -> tuple[float, float] | None:
    """Returns (fx, fy) 0.0–1.0 face-centre coords, or None."""
    if _face_cascade is None:
        return None
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Equalize histogram so backlit / dim-room frames don't fall below the
    # cascade's contrast threshold. Cheap (~1 ms at 1280x720).
    gray = cv2.equalizeHist(gray)
    # scaleFactor 1.1 → 1.05 gives finer pyramid steps (more positions checked
    # per image scale) — material recall bump for heads that fall between the
    # cascade's stride boundaries on the previous setting.
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40)
    )
    # Escalation pass: when the strict frontal cascade returns nothing, retry
    # once at minNeighbors=3 but with a larger minSize to keep false positives
    # from tiny shadow artifacts in check. Only runs on frames the strict pass
    # already rejected, so per-frame FP rate stays bounded.
    if len(faces) == 0:
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60)
        )
    # Frontal cascade misses heads turned past ~30°. Fall through to the
    # profile cascade — and a mirrored second pass since profileface is
    # trained on left-facing heads only — before giving up.
    if len(faces) == 0 and _profile_cascade is not None:
        prof = _profile_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40)
        )
        if len(prof) > 0:
            faces = prof
        else:
            mirror = cv2.flip(gray, 1)
            prof_m = _profile_cascade.detectMultiScale(
                mirror, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40)
            )
            if len(prof_m) > 0:
                faces = [(w - x - fw, y, fw, fh) for (x, y, fw, fh) in prof_m]
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])   # largest face
    fx = (x + fw / 2) / w
    fy = (y + fh / 2) / h
    if MIRROR_EYES_X: fx = 1.0 - fx
    if MIRROR_EYES_Y: fy = 1.0 - fy
    return fx, fy


def _probe_camera_index(idx: int, timeout_sec: float = CAMERA_PROBE_TIMEOUT_SEC) -> bool:
    """Open a camera index with a hard wall-clock timeout.

    cv2.VideoCapture(..., CAP_DSHOW) can block 20-30 s in its internal retry
    loop when an index has no device behind it. Running the open in a worker
    thread lets the main flow give up after `timeout_sec` and move on — the
    worker may keep running, but it's daemon and will finish (or never).
    Returns True only if the camera opened AND yielded a real frame in time.
    """
    result = {"ok": False}
    def _open():
        # Hold the camera I/O lock across the whole open+release so an
        # abandoned worker (one whose main-thread joiner timed out) can't
        # race a release with another camera operation later on. If this
        # worker is wedged inside cv2.VideoCapture, the main thread bails
        # via t.join(timeout=...) without trying to acquire the lock —
        # subsequent camera ops will simply wait for the wedge to clear
        # before re-using DirectShow plumbing.
        with _camera_io_lock:
            cap = None
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    result["ok"] = bool(ret)
            except Exception:
                pass
            finally:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass

    t = threading.Thread(target=_open, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        # Wedged in the C-level open call. Abandon the worker and report
        # failure — the underlying handle will be released when the worker
        # eventually returns (or when the process exits).
        return False
    return result["ok"]


def find_camera_locking_processes() -> list[str]:
    """Return display names of currently-running processes known to grab
    exclusive webcam locks (Teams, Zoom, OBS, etc.). Empty list if psutil
    is missing or no suspects are running. Used to give the user an
    actionable hint when the probe finds zero cameras.
    """
    try:
        import psutil
    except ImportError:
        return []
    suspects: list[str] = []
    try:
        for proc in psutil.process_iter(attrs=["name"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if name in CAMERA_LOCK_PROCESSES and proc.info["name"] not in suspects:
                    suspects.append(proc.info["name"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        return []
    return suspects


def probe_cameras_and_update_config() -> tuple[list[int], list[int]]:
    """At startup, try the configured CAMERA indices. If both fail, fall back
    to probing indices 0..CAMERA_PROBE_MAX-1 to find any working webcam, and
    rewrite CAMERAS in-place so face tracking can actually run.

    Each probe is capped at CAMERA_PROBE_TIMEOUT_SEC so a missing index can
    no longer freeze startup for ~26 s of DSHOW retries.

    Returns (working_indices, failed_configured_indices).
    """
    if not CAMERA_PROBE_ENABLED:
        return [], []

    # Helper: probe a list of indices IN PARALLEL — total time becomes the
    # longest single probe (~3 s) instead of the sum (~30 s for 10 indices).
    def _probe_many(indices: list[int]) -> dict[int, bool]:
        results: dict[int, bool] = {}
        threads = []
        def _runner(i):
            try:
                results[i] = _probe_camera_index(i)
            except Exception:
                logging.exception("[cam-probe] _runner failed for index %s", i)
                results[i] = False
        for i in indices:
            t = threading.Thread(target=_runner, args=(i,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=CAMERA_PROBE_TIMEOUT_SEC + 0.5)
        return results

    # Step 1: probe configured indices in parallel
    configured = [cam["index"] for cam in CAMERAS]
    cfg_results = _probe_many(configured)
    working_configured: list[int] = []
    for idx in configured:
        if cfg_results.get(idx, False):
            working_configured.append(idx)
            print(f"  [cam-probe] index {idx}: working ✓")
        else:
            print(f"  [cam-probe] index {idx}: failed to open")

    # Step 2: if any configured cameras worked, keep the original config
    if working_configured:
        good = [cam for cam in CAMERAS if cam["index"] in working_configured]
        CAMERAS[:] = good
        return working_configured, [i for i in configured if i not in working_configured]

    # Step 3: short-circuit — if a known webcam-locking app (Teams / Zoom /
    # OBS / Snap Camera) is running, don't bother sweeping. The cameras
    # aren't going to appear no matter how many indices we try.
    suspects = find_camera_locking_processes()
    if suspects:
        print(f"  [cam-probe] no configured cameras worked — and "
              f"{', '.join(suspects)} is holding the webcam lock. "
              f"Skipping fallback sweep.")
        print(f"  [cam-probe] close {suspects[0]} and restart, or run "
              f"`python bobert_companion.py --list-cameras`")
        return [], configured

    # Step 4: no obvious culprit — probe indices 0..MAX-1 in parallel
    print(f"  [cam-probe] no configured cameras worked — sweeping 0..{CAMERA_PROBE_MAX - 1} in parallel")
    sweep_indices = [i for i in range(CAMERA_PROBE_MAX) if i not in configured]
    sweep_results = _probe_many(sweep_indices)
    found: list[int] = []
    for idx in sorted(sweep_results.keys()):
        if sweep_results[idx]:
            found.append(idx)
            print(f"  [cam-probe] index {idx}: working ✓ (fallback)")

    if not found:
        # Surface the most likely culprit before returning empty so the user
        # knows what to close instead of just seeing 'cameras dead' again.
        suspects = find_camera_locking_processes()
        if suspects:
            print(f"  [cam-probe] no cameras found — these running apps commonly "
                  f"hold webcam locks: {', '.join(suspects)}")
            print(f"  [cam-probe] close them and re-run, or `python bobert_companion.py --list-cameras`")
        else:
            print(f"  [cam-probe] no cameras found and no known webcam-locking apps "
                  f"are running — webcams may be unplugged or disabled in Device Manager")
        return [], configured

    # Rewrite CAMERAS with the found indices. First found = "left", second =
    # "right" (heuristic — user can re-run --list-cameras to confirm).
    new_cams = []
    for i, idx in enumerate(found[:2]):
        if i == 0:
            new_cams.append({
                "index": idx, "label": f"Probed webcam (index {idx})",
                "primary": True, "look_x": 0.15, "look_y": 0.5,
            })
        else:
            new_cams.append({
                "index": idx, "label": f"Probed webcam (index {idx})",
                "primary": False, "look_x": 0.85, "look_y": 0.5,
            })
    CAMERAS[:] = new_cams
    return found, [i for i in configured if i not in found]


def _face_tracking_thread():
    """
    Multi-camera attention tracking:
      • Primary camera sees you → track face position precisely
      • Only a side camera sees you → look toward that camera's preset
      • No one sees you → eyes drift back to centre
    """
    global _smooth_x, _smooth_y, _last_sent_x, _last_sent_y
    SMOOTH   = 0.12   # primary-cam tracking smoothness
    SNAP     = 0.18   # side-cam snap-to-direction smoothness
    DRIFT    = 0.015  # drift back to centre when no face
    MIN_MOVE = 0.02
    # If c.read() returns False this many times in a row, the capture is
    # presumed wedged (Teams stole the cam, USB hub flickered, DirectShow
    # driver returned half-init Mat headers). Release and reopen instead of
    # continuing to call .read() on a dead handle (heap-corrupting on Logitech).
    MAX_READ_FAILURES = 60
    REOPEN_BACKOFF_SEC = 2.0   # don't hammer reopen attempts
    # Self-diag 2026-05-29: bumped WAKE_AFTER from 10→25 and added a wall-
    # clock gate (WAKE_AFTER_SEC) after observing the right webcam log a
    # "read failure #1 → woke via release+reopen" burst roughly every 30 s.
    # The pattern is a brief USB power-save blip — the device returns no
    # frame for ~1 s, then resumes on its own. Kicking it via release+reopen
    # at fail-count 10 (~0.5-1 s of downtime) churns the driver pointlessly
    # and is the leading suspect for the silent crash cascade. We now require
    # both ~25 consecutive failed reads AND at least 2 s elapsed since the
    # last good frame before triggering the soft wake, so transient blips
    # heal themselves silently.
    WAKE_AFTER          = 25
    WAKE_AFTER_SEC      = 2.0   # AND require >= this many seconds of silence
    WAKE_BACKOFF_SEC    = 3.0   # don't wake more than once per ~3 s
    # Only log once we're past WAKE_AFTER (i.e. about to act). A USB power-
    # save blip resolves in <1 s and shouldn't leave any trace in the log.
    LOG_FIRST_AT_FAILS  = 25
    LOG_EVERY_N_FAILS   = 25    # then every Nth fail beyond that

    def _open_capture(idx: int):
        # Serialize against probe-sweep / list-cameras / snapshot opens &
        # releases. Without this, an abandoned probe worker's eventual
        # release() can collide with this open or its failure-path
        # release() and trash DirectShow's heap.
        with _camera_io_lock:
            c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                # Buffersize=1 keeps the driver from queueing stale frames.
                # Some DirectShow drivers silently ignore this; harmless if so.
                try:
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:  # pragma: no cover - defensive: some DirectShow drivers reject CAP_PROP_BUFFERSIZE
                    pass
                return c
            try:
                c.release()
            except Exception:  # pragma: no cover - defensive: release of a half-opened capture handle rarely raises
                pass
            return None

    # Open all configured cameras at HD. Each entry is a mutable dict so
    # individual captures can be released & reopened independently mid-loop.
    caps: list[dict] = []
    for cam in CAMERAS:
        c = _open_capture(cam["index"])
        if c is not None:
            caps.append({"cam": cam, "cap": c, "fails": 0,
                         "next_reopen_at": 0.0, "next_wake_at": 0.0})
            w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [face-track] Opened {cam['label']} (index {cam['index']}) at {w}x{h}")
        else:
            print(f"  [face-track] Could not open {cam['label']} (index {cam['index']}) — skipping")

    if not caps:
        print(f"  [face-track] No cameras available. Try: --list-cameras")
        return

    primary_seen_recently = 0.0   # timestamp of last primary detection

    while not _face_track_stop.is_set():  # pragma: no cover - live-capture daemon loop (per-frame camera read/detect/smooth, runs until stop)
        try:
            # Always read from cameras so the latest frames stay fresh for see_user,
            # even when the tracker is paused (e.g. during speaking/listening).
            paused = _face_track_pause.is_set()

            # Read all cameras, run face detection on each
            primary_face = None
            side_hits: list[tuple[float, float]] = []

            now_loop = time.time()
            for entry in caps:
                cam = entry["cam"]
                c = entry["cap"]
                # Recovery path: capture was released after too many failures.
                # Try to reopen if backoff window has elapsed.
                if c is None:
                    if now_loop < entry["next_reopen_at"]:
                        continue
                    new_c = _open_capture(cam["index"])
                    entry["next_reopen_at"] = now_loop + REOPEN_BACKOFF_SEC
                    if new_c is None:
                        continue
                    entry["cap"] = new_c
                    entry["fails"] = 0
                    print(f"  [face-track] Reopened {cam['label']} (index {cam['index']}) after recovery")
                    c = new_c

                ret, frame = c.read()
                if not ret:
                    entry["fails"] += 1
                    # Record the failure so see_user / self-diagnostic can
                    # surface "camera went silent N seconds ago" instead of
                    # just guessing. Single-line, throttled log so a stalled
                    # USB pipe can't drown the session log.
                    err_msg = "read returned no frame (cap.isOpened still True)"
                    with _camera_state_lock:
                        _camera_last_read_error[cam["index"]] = err_msg
                        _camera_last_read_error_at[cam["index"]] = now_loop
                    _note_camera_read_attempt(cam["index"], ok=False,
                                              fails=entry["fails"],
                                              error=err_msg)
                    last_good = _camera_last_frame_at.get(cam["index"], 0.0)
                    time_since_good = (now_loop - last_good) if last_good else -1.0
                    # Don't log USB power-save blips that resolve in <1 s.
                    # Only emit once we've crossed LOG_FIRST_AT_FAILS, then
                    # every Nth fail beyond that.
                    if (entry["fails"] >= LOG_FIRST_AT_FAILS
                            and entry["fails"] % LOG_EVERY_N_FAILS
                                == LOG_FIRST_AT_FAILS % LOG_EVERY_N_FAILS):
                        gap_str = (f"{time_since_good:.1f}s since last frame"
                                   if time_since_good >= 0 else "no good frame yet")
                        print(f"  [face-track] {cam['label']} (index {cam['index']}) "
                              f"read failure #{entry['fails']} — {gap_str}")
                    # Try a soft wake before escalating to the full reopen.
                    # Gate on BOTH consecutive-fail count AND wall-clock silence
                    # so a transient USB power-save blip (resolves in ~1 s) is
                    # tolerated without churning the driver.
                    if (entry["fails"] >= WAKE_AFTER
                            and time_since_good >= WAKE_AFTER_SEC
                            and now_loop >= entry["next_wake_at"]
                            and entry["fails"] < MAX_READ_FAILURES):
                        with _camera_state_lock:
                            _camera_wake_attempts[cam["index"]] = (
                                _camera_wake_attempts.get(cam["index"], 0) + 1)
                        with _camera_io_lock:
                            try:
                                c.release()
                            except Exception:
                                logging.exception("[face-track] wake release raised")
                            # The old handle is now released — NEVER touch it
                            # again. Drop it from entry so that if the reopen
                            # below fails, the next loop iteration takes the
                            # recovery path (c is None → _open_capture) and
                            # re-opens cleanly instead of calling .read() on a
                            # RELEASED handle, which is heap corruption /
                            # 0xc0000374. 2026-05-30 deep audit.
                            entry["cap"] = None
                            new_c = None
                            woke_ret, woke_frame = False, None
                            time.sleep(0.15)
                            try:
                                new_c = cv2.VideoCapture(cam["index"], cv2.CAP_DSHOW)
                                if new_c.isOpened():
                                    try: new_c.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                                    except Exception: pass
                                    try: new_c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                                    except Exception: pass
                                    try:
                                        new_c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                    except Exception:
                                        pass
                                    # Warmup frame — first read often comes back
                                    # empty even on a healthy reopen.
                                    try:
                                        new_c.read()
                                    except Exception:
                                        pass
                                    woke_ret, woke_frame = new_c.read()
                                else:
                                    try: new_c.release()
                                    except Exception: pass
                                    new_c = None
                            except Exception:
                                # A DirectShow open/set/read on a half-init
                                # handle can raise — release the new handle so
                                # it can't leak; entry["cap"] stays None so the
                                # recovery path re-opens next iteration.
                                logging.exception("[face-track] wake reopen raised")
                                if new_c is not None:
                                    try: new_c.release()
                                    except Exception: pass
                                new_c = None
                                woke_ret, woke_frame = False, None
                        if new_c is not None and woke_ret and woke_frame is not None:
                            entry["cap"] = new_c
                            entry["fails"] = 0
                            with _camera_state_lock:
                                _camera_recoveries[cam["index"]] = (
                                    _camera_recoveries.get(cam["index"], 0) + 1)
                                _camera_last_read_error.pop(cam["index"], None)
                                _camera_last_read_error_at.pop(cam["index"], None)
                            print(f"  [face-track] {cam['label']} (index {cam['index']}) "
                                  f"woke via release+reopen")
                            # Use the wake's frame for this iteration so we
                            # don't waste a tick.
                            frame = woke_frame
                            ret   = True
                            c     = new_c
                        else:
                            entry["next_wake_at"] = now_loop + WAKE_BACKOFF_SEC
                            if new_c is not None:
                                # We managed to reopen but still no frame —
                                # the device is genuinely unresponsive. Keep
                                # the new handle so the MAX_READ_FAILURES
                                # branch below has something to release.
                                entry["cap"] = new_c
                                c = new_c
                            else:
                                # Both the wake's release AND the reopen
                                # failed. The original handle is already
                                # released; null out the entry so the
                                # recovery path (top of next iteration)
                                # takes over instead of touching a stale
                                # handle.
                                entry["cap"] = None
                                entry["next_reopen_at"] = now_loop + REOPEN_BACKOFF_SEC
                                c = None
                    if not ret:
                        if entry["fails"] >= MAX_READ_FAILURES:
                            # Capture is wedged — release before .read() returns
                            # a half-initialized Mat and trashes the heap. Mark
                            # for reopen on the next loop after backoff.
                            if c is not None:
                                with _camera_io_lock:
                                    try:
                                        c.release()
                                    except Exception:
                                        logging.exception("[face-track] release after %d failed reads", entry["fails"])
                            entry["cap"] = None
                            entry["next_reopen_at"] = now_loop + REOPEN_BACKOFF_SEC
                            with _camera_state_lock:
                                _camera_last_read_error[cam["index"]] = (
                                    f"capture wedged after {entry['fails']} read failures"
                                    f" + {_camera_wake_attempts.get(cam['index'], 0)} wake attempts")
                                _camera_last_read_error_at[cam["index"]] = now_loop
                            print(f"  [face-track] {cam['label']} (index {cam['index']}) "
                                  f"dead after {entry['fails']} failed reads; will reopen in {REOPEN_BACKOFF_SEC:.1f}s")
                        continue
                entry["fails"] = 0
                _note_camera_read_attempt(cam["index"], ok=True)
                # Cache frame for see_user action regardless of face detection
                with _camera_state_lock:
                    _camera_latest_frame[cam["index"]] = frame.copy()
                    _camera_last_frame_at[cam["index"]] = now_loop
                    # A real frame means whatever transient error we recorded
                    # has resolved — clear it so see_user reports clean state.
                    if cam["index"] in _camera_last_read_error:
                        _camera_last_read_error.pop(cam["index"], None)
                        _camera_last_read_error_at.pop(cam["index"], None)
                if paused:
                    continue   # skip detection/eye-control while paused
                face = _detect_face(frame)
                if not face:
                    continue
                # Record that this specific camera just saw a face
                with _camera_state_lock:
                    _camera_last_seen[cam["index"]] = now_loop
                if cam["primary"]:
                    primary_face = face
                else:
                    side_hits.append((cam["look_x"], cam["look_y"]))

            # Average all side cameras that see the face → handles "both cameras
            # see you" (= looking forward) gracefully.
            side_hit = None
            if side_hits:
                side_hit = (
                    sum(h[0] for h in side_hits) / len(side_hits),
                    sum(h[1] for h in side_hits) / len(side_hits),
                )

            # If we're paused, frames have been refreshed but skip the tracking math
            if paused:
                time.sleep(0.05)
                continue

            # Decide target eye position by priority
            global last_face_seen
            now = time.time()
            if primary_face:
                tx, ty = primary_face
                primary_seen_recently = now
                last_face_seen = now
                _smooth_x += SMOOTH * (tx - _smooth_x)
                _smooth_y += SMOOTH * (ty - _smooth_y)
            elif side_hit and (now - primary_seen_recently) > 0.4:
                # Primary has lost the face — follow the side camera
                tx, ty = side_hit
                last_face_seen = now
                _smooth_x += SNAP * (tx - _smooth_x)
                _smooth_y += SNAP * (ty - _smooth_y)
            else:
                _smooth_x += DRIFT * (0.5 - _smooth_x)
                _smooth_y += DRIFT * (0.5 - _smooth_y)

            if (abs(_smooth_x - _last_sent_x) > MIN_MOVE or
                    abs(_smooth_y - _last_sent_y) > MIN_MOVE):
                send(eyes_x=round(_smooth_x, 3), eyes_y=round(_smooth_y, 3))
                _last_sent_x = _smooth_x
                _last_sent_y = _smooth_y

            time.sleep(0.05)
        except Exception:
            logging.exception("[face-track] error in tracking loop iteration")
            time.sleep(0.1)

    for entry in caps:
        c = entry["cap"]
        if c is None:
            continue  # pragma: no cover - defensive: a None cap entry only occurs if an earlier open failed
        with _camera_io_lock:
            try:
                c.release()
            except Exception:  # pragma: no cover - defensive: camera release during shutdown rarely raises
                logging.exception("[face-track] release on shutdown")
    print("  [face-track] Stopped")


def get_monitors() -> list[tuple[int, int, int, int]]:
    """Detect all physical monitors via Win32. Returns [(x, y, w, h), ...]."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes
    monitors = []
    MonitorEnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
        ctypes.POINTER(wintypes.RECT), wintypes.LPARAM,
    )
    def _cb(hmonitor, hdc, lprect, lparam):
        r = lprect.contents
        monitors.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
        return 1
    cb_proc = MonitorEnumProc(_cb)
    ctypes.windll.user32.EnumDisplayMonitors(0, 0, cb_proc, 0)
    return monitors


def list_monitors_cli():
    mons = get_monitors()
    if not mons:
        print("No monitors detected (or running on non-Windows).")
        return
    print(f"Detected {len(mons)} monitor(s):\n")
    for i, (x, y, w, h) in enumerate(mons):
        # Heuristic guess at which physical position it occupies
        if y < 0:
            guess = "ABOVE the others"
        elif y > 0:
            guess = "BELOW the others"
        elif x == 0:
            guess = "leftmost (likely primary)"
        elif x > 0:
            guess = f"to the right of x={x}"
        else:
            guess = f"to the LEFT of primary (x={x})"
        print(f"  [{i}] x={x:>5}, y={y:>5}, {w}x{h}   — {guess}")
    print("\nCopy these into the MONITORS dict at the top of the script.")
    print("Example:")
    print("  MONITORS = {")
    for i, (x, y, w, h) in enumerate(mons):
        name = ["middle", "right", "left", "top", "extra"][i] if i < 5 else f"mon{i}"
        print(f'      "{name}": ({x}, {y}, {w}, {h}),')
    print("  }")


def list_microphones():
    """Print all available audio input devices with their indexes."""
    print("Available microphones (input devices):\n")
    devices = sd.query_devices()
    default_in = sd.default.device[0] if sd.default.device else None
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = "  ← system default" if i == default_in else ""
            print(f"  [{i:>2}] {d['name']}  ({d['max_input_channels']} ch, "
                  f"{int(d['default_samplerate'])} Hz){marker}")
    print("\nTo use a specific mic, set MICROPHONE_INDEX at the top of the script.")


def list_speakers():
    """Print all available audio output devices with their indexes."""
    print("Available speakers (output devices):\n")
    devices = sd.query_devices()
    default_out = sd.default.device[1] if sd.default.device else None
    for i, d in enumerate(devices):
        if d["max_output_channels"] > 0:
            marker = "  ← system default" if i == default_out else ""
            print(f"  [{i:>2}] {d['name']}  ({d['max_output_channels']} ch, "
                  f"{int(d['default_samplerate'])} Hz){marker}")


# Device auto-switching state
_device_cache = {"in": None, "out": None, "checked_at": 0.0,
                 "last_in_name": None, "last_out_name": None,
                 "last_devices_signature": None}
# Serializes _refresh_devices so two callers (e.g. listen loop + a skill)
# can't both be tearing PortAudio down at the same time, and so the wake-word
# pause/resume bracket around sd._terminate() isn't racing with concurrent
# sd.query_devices() calls coming from a peer refresh.
_device_refresh_lock = threading.Lock()


def _devices_signature():
    """Hashable snapshot of the current PortAudio device list.

    Used by _refresh_devices() to skip the destructive
    sd._terminate()/sd._initialize() cycle when nothing has actually
    changed. The reinit is what gives us USB hotplug detection but it is
    also what causes 0xc0000374 heap corruption against live streams
    (crash-fix-1 in jarvis_todo.md), so we only pay that cost when the
    device list legitimately differs from the last enumeration.

    Captures (index, name, in-channels, out-channels) per device so a
    plug/unplug that shifts indices, a renamed endpoint, or a capability
    change all force a fresh enumeration."""
    try:
        devices = sd.query_devices()
    except Exception:
        return None
    try:
        return tuple(
            (i, d.get("name", ""),
             d.get("max_input_channels", 0),
             d.get("max_output_channels", 0))
            for i, d in enumerate(devices)
        )
    except Exception:
        return None


def _friendly_device_name(raw: str) -> str:
    """Speakable short name for a sounddevice description, e.g.
    'Microphone (USB Mic), MME' → 'USB Mic';
    'Headset Microphone (Gaming Headset), Windows DirectSound' → 'Gaming Headset';
    'Speakers (Realtek)' → 'Realtek'."""
    if not raw:
        return ""
    name = raw.split(",")[0].strip()
    m = re.search(r"\(([^)]+)\)", name)
    if m:
        return m.group(1).strip()
    for prefix in ("Headset Microphone ", "External Microphone ",
                   "Microphone ", "Headphones ", "Speakers "):
        if name.startswith(prefix):
            return name[len(prefix):].strip() or name
    return name


# Speech-queue dedup state — protects _speak_pending against the looping
# notification bug where the same toast text lands in pending_speech.json
# repeatedly (because the notification listener loses its _seen_ids state
# on bounce). In-memory only: spans a single JARVIS lifetime, which is the
# window in which a runaway loop actually does damage. Window is wide
# enough to suppress a runaway flood but well under notification_triage's
# 300s SNOOZE_SECONDS so legitimate re-announcements still get through.
_RECENT_SPEECH_DEDUPE_WINDOW = 60.0
_recent_spoken_messages: dict[str, float] = {}
_recent_spoken_lock = threading.Lock()


def _speech_was_recently_spoken(message: str) -> bool:
    """Return True if `message` was spoken inside the dedupe window. Also
    prunes expired entries so the dict stays small."""
    now = time.time()
    cutoff = now - _RECENT_SPEECH_DEDUPE_WINDOW
    with _recent_spoken_lock:
        # Prune as we go — cheap and bounded by how many distinct messages
        # we speak per minute.
        for k in list(_recent_spoken_messages.keys()):
            if _recent_spoken_messages[k] < cutoff:
                del _recent_spoken_messages[k]
        last = _recent_spoken_messages.get(message, 0.0)
        return last >= cutoff


def _mark_speech_spoken(message: str) -> None:
    with _recent_spoken_lock:
        _recent_spoken_messages[message] = time.time()


# Serialises concurrent writers of pending_speech.json (see proactive_announce).
_pending_speech_lock = threading.Lock()


def proactive_announce(message: str, source: str = "skill",
                       *, mood: str | None = None,
                       volume_scale: float = 1.0) -> bool:
    """Public proactive-speech API for skills.

    Skills that want JARVIS to speak something unprompted (print milestones,
    Teams nudges, timer reminders, etc.) should call this — it appends the
    message to pending_speech.json and the main listen loop will drain the
    queue at the next turn boundary. Returns True on successful enqueue.

    `source` is just a tag for the console fallback log so it's obvious which
    skill produced an announcement that couldn't be written to disk.

    `mood` (optional) is the voice_mood layer opt-in. When set, the drainer
    forwards it as mood= to _speak() so the queued utterance lands with the
    matching preset (urgent_clipped for VIP alerts, concerned_soft for
    self-diagnostic problems, etc.).
    """
    # Serialise the whole read-modify-write. proactive_announce is called from
    # many daemon threads (timers, promises, Teams/weather/device-flap nudges);
    # without this two concurrent enqueues both read state N, both append, and
    # the later os.replace clobbers the earlier writer's entry — a silently
    # dropped announcement. The unique mkstemp temp already prevents tmp
    # collisions; this prevents the lost update.
    _pending_speech_lock.acquire()
    try:
        proj_dir   = os.path.dirname(os.path.abspath(__file__))
        queue_path = os.path.join(proj_dir, "pending_speech.json")
        data: list = []
        if os.path.exists(queue_path):
            try:
                with open(queue_path, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                if raw:
                    try:
                        decoded, _ = json.JSONDecoder().raw_decode(raw)
                        if isinstance(decoded, list):
                            data = decoded
                    except Exception:
                        data = []
            except Exception:
                data = []
        entry: dict = {"ts": time.time(), "message": message}
        if mood:
            entry["mood"] = mood
        if volume_scale != 1.0:
            # Per-utterance TTS volume (boss-mode / night-owl whisper). Lets the
            # volume_scale co-writers route through this single locked path
            # instead of doing their own unsynchronised pending_speech.json writes.
            entry["volume_scale"] = float(volume_scale)
        data.append(entry)
        # Cap the pending-speech queue. Every enqueue rewrites the whole file
        # (O(n²)); a lagging drainer or a chatty skill (timer reminders,
        # device-flap alerts, Teams nudges) must not grow it unbounded. Keep
        # the most recent 50 announcements. 2026-05-30 audit.
        if len(data) > 50:
            data = data[-50:]
        fd: int = -1
        tmp: str | None = None
        try:
            fd, tmp = tempfile.mkstemp(dir=proj_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = -1   # fdopen took ownership of the descriptor
                json.dump(data, f, indent=2)
            os.replace(tmp, queue_path)
            tmp = None
        except Exception:
            if fd >= 0:
                try: os.close(fd)
                except Exception: pass  # pragma: no cover - defensive: os.close of an owned descriptor effectively never raises
            if tmp is not None:
                try: os.unlink(tmp)
                except Exception: pass
            raise
        return True
    except Exception as e:
        # Read-only share, full disk, perms — surface the message to console so
        # the announcement isn't silently lost.
        print(f"  [{source}] speech-queue write failed ({e}); announcement: {message}")
        return False
    finally:
        _pending_speech_lock.release()


def _update_check_thread():
    """Background, throttled once-per-day update check. Sleeps briefly so it
    doesn't compete with the boot bring-up, then asks core.update_checker to
    compare this build to the latest GitHub release and, if a newer one exists,
    queue ONE spoken nudge via proactive_announce. Best-effort + total — any
    failure is swallowed so it can never affect the main loop. Started from
    main() (skipped in staging)."""
    try:
        time.sleep(45)  # let whisper load + the boot greeting finish first
        from core import update_checker as _uc
        _uc.boot_nudge(
            lambda m: proactive_announce(m, source="update_check"),
            enabled=UPDATE_CHECK_ENABLED,
        )
    except Exception as e:  # pragma: no cover - defensive; boot_nudge is total
        print(f"  [update_check] skipped: {e}")


def _enqueue_device_announcement(message: str) -> None:
    """Audio-device change notifier. Thin wrapper around proactive_announce()
    so the device-change call site keeps its dedicated `[audio]` log tag for
    the console-fallback path."""
    proactive_announce(message, source="audio")


def _input_openable(idx: int) -> bool:
    """True when an InputStream can actually be opened on `idx` at JARVIS's
    capture format (SAMPLE_RATE / mono / float32).

    _pick_device() gates input selection on this so a device that merely
    *enumerates* but cannot be opened is skipped in favour of the next
    preferred mic — instead of being selected and then failing the real
    InputStream open with -9996/-9994 and silently falling back to the
    system-default mic. Two cases this catches in practice:
      • a phantom endpoint left behind by a just-unplugged headset (some
        headsets linger in the device list for a beat after they drop);
      • a WDM-KS-only device (e.g. the Realtek USB adapter) whose exclusive
        endpoint won't honour 16 kHz, so the check raises 'format not
        supported'.
    check_input_settings only queries Pa_IsFormatSupported — it does not open
    a stream — so this is cheap and safe to call during a device refresh."""
    try:
        sd.check_input_settings(device=idx, samplerate=SAMPLE_RATE,
                                channels=1, dtype="float32")
        return True
    except Exception:
        return False


def _pick_device(preferred_names: list[str], want_input: bool) -> tuple[int | None, str]:
    """Find the first preferred device that's connected — and, for inputs,
    that can actually be opened at the capture format. Returns (index, name)
    or (None, '') if no preference matched a usable device.

    Scanning continues past an unopenable match (e.g. the WDM-KS variant of a
    device, or a phantom headset endpoint) so a lower-index dud never masks a
    higher-index, openable variant of the same or a lower-priority device."""
    try:
        devices = sd.query_devices()
    except Exception:
        return None, ""
    for pref in preferred_names:
        for i, d in enumerate(devices):
            name = d["name"]
            if pref.lower() not in name.lower():
                continue
            if want_input:
                if d["max_input_channels"] > 0 and _input_openable(i):
                    return i, name
            elif d["max_output_channels"] > 0:
                return i, name
    return None, ""


def _refresh_devices(force: bool = False):
    """Re-check preferred devices if enough time has passed.
    Logs a message when the active device changes.

    PortAudio caches its device list at sd._initialize() time; it never
    notices USB hotplug on its own. To pick up a freshly-plugged/unplugged
    mic mid-session we have to call sd._terminate() + sd._initialize().
    That call is hostile to any sd.InputStream that's already open: the
    teardown frees backing state out from under live callbacks (0xc0000374
    heap corruption — see crash-fix-1 in jarvis_todo.md).

    Two mitigations together make it safe:
      • _device_refresh_lock serializes concurrent refresh attempts and
        protects the critical section from peer sd.query_devices() callers.
      • The persistent wake-word InputStream is paused before re-init and
        resumed after. The short-lived barge-in stream is scoped to TTS
        playback and we don't refresh during that window in practice."""
    now = time.time()
    if not force and (now - _device_cache["checked_at"] < DEVICE_CHECK_INTERVAL):
        return

    # Snapshot the current device list *before* the lock so we can decide
    # whether the destructive sd._terminate()/sd._initialize() cycle below
    # is actually warranted. PortAudio's query is read-only and cheap; the
    # reinit is what tears live stream state down (0xc0000374 — see the
    # _devices_signature() docstring).
    current_sig = _devices_signature()

    with _device_refresh_lock:
        # Re-check inside the lock — a peer thread may have just refreshed
        # while we were waiting, in which case there's nothing to do.
        now = time.time()
        if not force and (now - _device_cache["checked_at"] < DEVICE_CHECK_INTERVAL):
            return

        # If the device list hasn't drifted since the last enumeration we
        # skip the reinit entirely: every avoided sd._terminate() is one
        # less window in which an active wake-word or barge-in InputStream
        # can land on freed PortAudio state. We still bump checked_at so
        # the time gate above re-arms for another DEVICE_CHECK_INTERVAL.
        # force=True (e.g. get_current_mic_name()) always re-enumerates.
        last_sig = _device_cache["last_devices_signature"]
        if (not force and last_sig is not None
                and current_sig is not None
                and current_sig == last_sig):
            _device_cache["checked_at"] = now
            return

        # Pause the wake-word detector's persistent InputStream so PortAudio
        # can be torn down without dangling stream state. resume() reopens
        # it after we've re-enumerated.
        wl = sys.modules.get("skill_wake_listener")
        det = getattr(wl, "_detector", None) if wl is not None else None
        paused_det = None
        if det is not None and hasattr(det, "pause") and hasattr(det, "is_running"):
            try:
                if det.is_running():
                    det.pause()
                    paused_det = det
            except Exception as e:
                print(f"  [audio] wake-word pause failed: {e}")

        try:
            # Force PortAudio to re-enumerate so USB plug/unplug events
            # show up in sd.query_devices(). This sd._terminate()/_initialize()
            # cycle frees backing state out from under EVERY open stream, not
            # just the wake-word one we pause below — so if the main loop is
            # live inside record_speech (its InputStream open), tearing
            # PortAudio down here corrupts the heap (0xc0000374). The
            # wake-word pause does not cover record_speech, barge-in, or
            # ambient streams. Guard on the record_speech ownership flag:
            # when it holds the mic we SKIP the destructive reinit and just
            # re-pick from the existing enumeration. The drift (mic plugged/
            # unplugged) gets picked up on the next refresh once record_speech
            # is briefly idle — which happens every utterance / 20s timeout.
            if _record_speech_active[0]:
                print("  [audio] device drift detected but record_speech is "
                      "live — deferring PortAudio reinit to avoid a mid-"
                      "capture teardown (0xc0000374).")
            elif _tts_playback_active[0]:
                print("  [audio] device drift detected but TTS playback is "
                      "live — deferring PortAudio reinit to avoid tearing down "
                      "the barge-in stream (0xc0000374).")
            else:
                try:
                    sd._terminate()
                    sd._initialize()
                except Exception as e:
                    print(f"  [audio] PortAudio re-init failed: {e}")

            # Input
            if MICROPHONE_INDEX is not None:
                in_idx = MICROPHONE_INDEX
                try: in_name = sd.query_devices(in_idx)["name"]
                except Exception: in_name = ""
            else:
                in_idx, in_name = _pick_device(PREFERRED_INPUT_DEVICES, want_input=True)

            # Output
            if SPEAKER_INDEX is not None:
                out_idx = SPEAKER_INDEX
                try: out_name = sd.query_devices(out_idx)["name"]
                except Exception: out_name = ""
            else:
                out_idx, out_name = _pick_device(PREFERRED_OUTPUT_DEVICES, want_input=False)

            # Log changes — and on a genuine mid-session mic switch (i.e. the name
            # changed away from a previous non-None value), enqueue a spoken alert so
            # JARVIS doesn't silently fall back to a worse mic. Initial detection at
            # startup (None → first name) is suppressed so we don't announce on boot.
            if in_name and in_name != _device_cache["last_in_name"]:
                print(f"  [audio] mic → [{in_idx}] {in_name}")
                prev_in = _device_cache["last_in_name"]
                if prev_in:
                    new_friendly = _friendly_device_name(in_name) or "the fallback mic"
                    prev_lower   = prev_in.lower()
                    if "headset" in prev_lower or "headphone" in prev_lower:
                        msg = (f"Switched to {new_friendly}, sir — "
                               f"your headset appears to have dropped off.")
                    else:
                        old_friendly = _friendly_device_name(prev_in) or "the previous mic"
                        msg = (f"Switched to {new_friendly}, sir — "
                               f"{old_friendly} appears to have disconnected.")
                    _enqueue_device_announcement(msg)
                _device_cache["last_in_name"] = in_name
            if out_name and out_name != _device_cache["last_out_name"]:
                print(f"  [audio] speakers → [{out_idx}] {out_name}")
                _device_cache["last_out_name"] = out_name

            _device_cache["in"]         = in_idx
            _device_cache["out"]        = out_idx
            _device_cache["checked_at"] = now
            # Recompute the signature *after* the reinit so the next pass
            # compares against the freshly-enumerated list, not the stale
            # snapshot taken before the lock. Indices can shift across a
            # reinit even when the underlying hardware is unchanged.
            _device_cache["last_devices_signature"] = _devices_signature()
        finally:
            if paused_det is not None:
                try:
                    paused_det.resume()
                except Exception as e:
                    print(f"  [audio] wake-word resume failed: {e}")


def get_input_device() -> int | None:
    # Hard-disabled mic (staging green candidate / MICROPHONE_INDEX < 0): no
    # device, no capture, and crucially DON'T fall through to the "using system
    # default" path below — that message (and the None it returns, which every
    # InputStream open treats as the default mic) is exactly how staging ended
    # up listening. _mic_input_disabled is defined just below; resolved at call
    # time so the forward reference is fine.
    if _mic_input_disabled():
        return None
    _refresh_devices()
    idx = _device_cache["in"]
    if idx is None:
        return None
    # Validate that the cached index is still queryable. PortAudio's
    # device list can go stale between _refresh_devices() calls — e.g.
    # a USB mic dropping off, or sd._terminate()/_initialize() shifting
    # indices. Opening an InputStream with a stale index raises
    # PortAudioError("Error querying device N") mid-stack and crashed
    # the main loop (see session_2026-05-28_20-39-03.log). Clearing the
    # cache and returning None lets sounddevice fall back to the system
    # default; the next _refresh_devices() pass will re-pick.
    try:
        sd.query_devices(idx)
        return idx
    except Exception as e:
        print(f"  [audio] cached mic index {idx} no longer queryable ({e}); using system default")
        _device_cache["in"] = None
        _device_cache["checked_at"] = 0.0
        return None


def get_output_device() -> int | None:
    _refresh_devices()
    idx = _device_cache["out"]
    if idx is None:
        return None
    try:
        sd.query_devices(idx)
        return idx
    except Exception as e:
        print(f"  [audio] cached speaker index {idx} no longer queryable ({e}); using system default")
        _device_cache["out"] = None
        _device_cache["checked_at"] = 0.0
        return None


def _mic_input_disabled() -> bool:
    """True when audio INPUT must not be opened AT ALL: the staging/blue-green
    GREEN candidate (the mic is owned by the live prod instance) or an explicit
    MICROPHONE_INDEX < 0. A negative index otherwise fell through
    get_input_device() → None → the SYSTEM DEFAULT mic, so every capture path
    silently opened the default microphone — defeating 'no mic' AND competing
    with prod for the device (the staging block at the top of this file says it
    must NOT touch the audio devices prod owns). 2026-05-31."""
    if _is_staging():
        return True
    return MICROPHONE_INDEX is not None and MICROPHONE_INDEX < 0


def get_current_mic_name() -> str:
    """Friendly name of whatever mic is currently selected."""
    _refresh_devices(force=True)
    idx = _device_cache["in"]
    if idx is None:
        return "(system default)"
    try:
        return f"[{idx}] {sd.query_devices(idx)['name']}"
    except Exception:
        return f"[{idx}] (unknown)"


def get_current_speaker_name() -> str:
    idx = _device_cache["out"]
    if idx is None:
        return "(system default)"
    try:
        return f"[{idx}] {sd.query_devices(idx)['name']}"
    except Exception:
        return f"[{idx}] (unknown)"


def list_cameras(max_check: int = CAMERA_PROBE_MAX):
    """Probe camera indexes at their max resolution, save a preview JPEG
    for each working one so you can tell which physical camera is which.

    Each probe is wrapped in a thread+timeout so missing indices return
    quickly (cv2.CAP_DSHOW retries internally for ~26 s otherwise).
    """
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_previews")
    os.makedirs(out_dir, exist_ok=True)

    # Check for lock-holders up front — saves the user staring at "not
    # available" for every index when the real cause is Teams in the tray.
    suspects = find_camera_locking_processes()
    if suspects:
        print(f"⚠  These running apps commonly hold webcam locks: {', '.join(suspects)}")
        print(f"   If cameras show 'not available' below, close them and re-run.\n")

    print(f"Scanning indices 0..{max_check - 1} (timeout {CAMERA_PROBE_TIMEOUT_SEC}s each)…")

    def _open_with_warmup(idx: int) -> tuple[bool, bool, int, int, float, "np.ndarray | None"]:
        """Returns (opened, got_frame, w, h, mean_brightness, frame_or_None)."""
        result = {"opened": False, "ret": False, "w": 0, "h": 0,
                  "bright": 0.0, "frame": None}
        def _do():
            # Same serialization rationale as _probe_camera_index._open:
            # if this worker wedges past the join() timeout, the lock keeps
            # its eventual release() from colliding with later camera ops.
            with _camera_io_lock:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                try:
                    if not cap.isOpened():
                        return
                    result["opened"] = True
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    # Warm up — many cameras need several frames before auto-exposure works
                    for _ in range(15):
                        cap.read()
                        time.sleep(0.05)
                    ret, frame = cap.read()
                    if ret:
                        result["ret"] = True
                        result["h"], result["w"] = frame.shape[:2]
                        result["bright"] = float(frame.mean())
                        result["frame"] = frame
                except Exception:
                    pass
                finally:
                    try: cap.release()
                    except Exception: pass  # pragma: no cover - defensive: list-cameras probe release rarely raises

        t = threading.Thread(target=_do, daemon=True)
        t.start()
        # Generous timeout for list-cameras (warm-up frames take ~0.75 s alone)
        t.join(timeout=max(CAMERA_PROBE_TIMEOUT_SEC, 5.0))
        return (result["opened"], result["ret"], result["w"], result["h"],
                result["bright"], result["frame"])

    for i in range(max_check):
        opened, ret, w, h, mean_brightness, frame = _open_with_warmup(i)
        if ret and frame is not None:
            path = os.path.join(out_dir, f"camera_{i}.jpg")
            cv2.imwrite(path, frame)
            quality = "OK" if mean_brightness > 10 else "⚠ BLACK / blocked"
            status = f"working  ({w}x{h})  [{quality}, brightness={mean_brightness:.1f}]  → preview saved"
        elif opened:
            status = "opened but no frame"
        else:
            status = "not available"
        print(f"  Camera {i}: {status}")
    print(f"\nPreviews saved to: {out_dir}")
    print("Open the JPEGs to identify which index is the Kinect, left webcam, and right webcam.")
    print("Then edit the CAMERAS list at the top of the script.")


def pause_face_tracking():
    _face_track_pause.set()

def resume_face_tracking():
    _face_track_pause.clear()


# ──────────────────────────────────────────────────────────────────────────
#  PROACTIVE / IDLE BEHAVIOR  — JARVIS-style spontaneous observations
# ──────────────────────────────────────────────────────────────────────────

def should_be_proactive() -> bool:
    """Decide whether to fire a proactive comment right now."""
    if not PROACTIVE_ENABLED:
        return False

    silence = time.time() - last_speech_time
    if silence < PROACTIVE_MIN_SILENCE:
        return False

    # Voice-mood suppression — when the user was just detected as stressed,
    # the adapter writes 'voice_mood_stressed_until' into memory; honour
    # that 15-minute quiet window before checking face/probability.
    if _voice_mood_response is not None:
        try:
            with _memory_lock:
                if _voice_mood_response.is_stress_suppression_active(load_memory()):
                    return False
        except Exception:
            pass

    # Must be able to see the user (so we're not talking to an empty room)
    if PROACTIVE_REQUIRE_FACE:
        if last_face_seen == 0.0 or (time.time() - last_face_seen) > 60:
            return False

    # Probability scales from 0 at MIN_SILENCE to 1 at MAX_SILENCE
    span = max(1, PROACTIVE_MAX_SILENCE - PROACTIVE_MIN_SILENCE)
    progress = (silence - PROACTIVE_MIN_SILENCE) / span
    progress = max(0.0, min(1.0, progress))
    # Each check has at most a 35% chance even at max silence
    return random.random() < (progress * 0.35)


def generate_proactive_comment() -> str:
    """Use the LLM + memory to write a brief JARVIS-style observation."""
    system = _system_prompt + (
        "\n\nYou are now generating a PROACTIVE comment. Your owner has been "
        "quiet for a while but you can see them at their desk. "
        "Pick ONE of: a brief observation about something you remember they're "
        "working on, a thoughtful question about a recent topic, a short interesting "
        "fact related to their interests, or a check-in. "
        "Keep it to ONE sentence. Do not start with 'Hey' or any greeting. "
        "Sound natural, like you just thought of it."
    )
    user = "(There has been silence. Generate one short proactive comment.)"
    try:
        text = _llm_quick(system=system, user=user, max_tokens=120)
        return text.strip().split("\n")[0]
    except Exception as e:
        print(f"  [proactive] generation failed: {e}")
        return ""


# ──────────────────────────────────────────────────────────────────────────
#  LATE-NIGHT CHECK  (1:00 AM – 5:00 AM dry remark before complying)
# ──────────────────────────────────────────────────────────────────────────
# When the user gives a command during the wee hours, prepend a brief,
# in-character JARVIS remark before the actual response. Cycles through a
# small phrase bank so it doesn't get repetitive, and throttles to one
# remark per LATE_NIGHT_COOLDOWN seconds so rapid-fire commands during a
# late-night session don't get a remark stacked on every single one.
#
# Suppressed for the rest of the night when the user explicitly says
# "no comments tonight" (or similar). Suppression is stored in memory
# under "late_night_no_comments_until" — an ISO date string — and is
# cleared automatically at the next 5 AM rollover.

LATE_NIGHT_START_HOUR = 1        # inclusive
LATE_NIGHT_END_HOUR   = 5        # exclusive  → window is 01:00:00 – 04:59:59
LATE_NIGHT_COOLDOWN   = 600      # seconds between remarks (10 min)

LATE_NIGHT_PHRASES = (
    "It's nearly {hr} AM, sir. The suit can wait until morning, but very well.",
    "Past {hr} in the morning, sir — I'll assume you know what you're doing.",
    "{hr} AM, sir. Sleep is, statistically, also an option.",
    "Working through the night again, sir? Right away.",
    "It is well past midnight, sir. As you wish.",
    "I'd remark on the hour, sir, but you've clearly already noticed.",
    "{hr} AM. I do hope this is important, sir.",
    "Burning the proverbial candle at both ends, sir. On it.",
)

LATE_NIGHT_SUPPRESS_PHRASES = (
    "no comments tonight",
    "no remarks tonight",
    "spare me the commentary",
    "no commentary tonight",
    "skip the remarks",
    "skip the commentary",
)

# Module-level cursor for round-robin phrase rotation
_late_night_phrase_idx: list[int]  = [0]
_late_night_last_remark: list[float] = [0.0]


def _in_late_night_window(now: float | None = None) -> bool:
    """True if the local hour is in the late-night remark window."""
    lt = time.localtime(now if now is not None else time.time())
    return LATE_NIGHT_START_HOUR <= lt.tm_hour < LATE_NIGHT_END_HOUR


def _late_night_session_key(now: float | None = None) -> str:
    """A YYYY-MM-DD key identifying the current late-night session.
    Treats the whole 01:00–04:59 block as belonging to that calendar date,
    so a suppression set at 02:30 covers the rest of that same night.
    """
    return time.strftime("%Y-%m-%d", time.localtime(now if now is not None else time.time()))


def _is_late_night_suppressed(memory: dict, now: float | None = None) -> bool:
    """True if the user has muted late-night remarks for tonight."""
    until = (memory or {}).get("late_night_no_comments_until", "")
    if not until:
        return False
    return until == _late_night_session_key(now)


def _set_late_night_suppression(memory: dict) -> None:
    """Mute late-night remarks for the rest of this calendar night."""
    memory["late_night_no_comments_until"] = _late_night_session_key()
    try:
        save_memory(memory)
    except Exception as e:
        print(f"  [late-night] failed to persist suppression: {e}")


def _matches_suppress_phrase(text: str) -> bool:
    """Word-boundary match against LATE_NIGHT_SUPPRESS_PHRASES, short utterances only."""
    tl = (text or "").strip().lower()
    if not tl or len(tl.split()) > 8:
        return False
    for p in LATE_NIGHT_SUPPRESS_PHRASES:
        if re.search(r'\b' + re.escape(p) + r'\b', tl):
            return True
    return False


def _late_night_hour_word(now: float | None = None) -> str:
    """Format the current late-night hour for phrase interpolation.
    e.g. 03:14 → '3', 01:02 → '1'. (Always within 1..4 inside the window.)
    """
    lt = time.localtime(now if now is not None else time.time())
    return str(lt.tm_hour)


def maybe_late_night_remark(user_text: str, memory: dict) -> str:
    """Return a one-line dry remark to speak before complying, or '' if
    no remark is appropriate right now.

    Side effects:
      - If the user said a suppression phrase, sets the memory flag and
        returns a brief acknowledgement.
      - Updates the cooldown timestamp + phrase-rotation cursor when a
        remark is returned.
    """
    if not _in_late_night_window():
        return ""

    if _matches_suppress_phrase(user_text):
        _set_late_night_suppression(memory)
        return "As you wish, sir. Silent until morning."

    if _is_late_night_suppressed(memory):
        return ""

    now = time.time()
    if now - _late_night_last_remark[0] < LATE_NIGHT_COOLDOWN:
        return ""

    idx = _late_night_phrase_idx[0] % len(LATE_NIGHT_PHRASES)
    phrase = LATE_NIGHT_PHRASES[idx].format(hr=_late_night_hour_word(now))
    _late_night_phrase_idx[0] = (idx + 1) % len(LATE_NIGHT_PHRASES)
    _late_night_last_remark[0] = now
    return phrase


# ──────────────────────────────────────────────────────────────────────────
#  THINKING ANIMATION  (slow eye sweep while LLM generates a response)
# ──────────────────────────────────────────────────────────────────────────

def _thinking_loop(stop_evt: threading.Event):
    t = 0.0
    _i = 0
    while not stop_evt.is_set():
        try:
            x = 0.5 + 0.38 * math.sin(t * 0.8)          # slow left-right sweep
            y = 0.62 + 0.06 * math.sin(t * 2.5)          # subtle up-down bob
            send(eyes_x=round(x, 3), eyes_y=round(y, 3))
            t += 0.10
            # Keep the main-loop watchdog satisfied while the LLM thinks.
            # On the LOCAL model a single reply can take 20–40 s under VRAM
            # pressure; without a tick here the 60 s heartbeat goes stale
            # mid-think and the watchdog false-trips, aborting the *next*
            # record_speech (the 2026-05-30 12:18 stall). This animation
            # only runs while JARVIS is actively thinking, so ticking here
            # means exactly "the main loop is alive and working".
            _i += 1
            if _i % 20 == 0:          # ≈ every 1 s of wall-clock (20 × 0.05s)
                _heartbeat()
            time.sleep(0.05)
        except Exception:
            logging.exception("_thinking_loop iteration failed")
            time.sleep(0.05)


def get_response_with_animation(user_text: str) -> str:
    """Run the thinking eye animation concurrently while the LLM thinks."""
    pause_face_tracking()
    set_state("thinking")

    stop_evt  = threading.Event()
    anim      = threading.Thread(target=_thinking_loop, args=(stop_evt,), daemon=True)
    anim.start()

    reply = _call_llm(user_text)

    stop_evt.set()
    anim.join()
    return reply


# ──────────────────────────────────────────────────────────────────────────
#  AUDIO RECORDING  (energy-based VAD)
# ──────────────────────────────────────────────────────────────────────────

_last_recording_peak = 0.0   # set by record_speech, read by callers

# 2026-05-30 [self-heal]: one-shot guard for the silent-mic warning emitted
# by record_speech when raw mic RMS stays at zero past MIC_SILENT_WARN_SECONDS
# while JARVIS is awake. Resets back to False once an audible chunk is seen,
# so a transient driver glitch + recovery can re-warn next time.
_silent_mic_warned = [False]

# ── Main-loop watchdog (bug-5) ────────────────────────────────────────────
# Guards against record_speech() (or anything else on the main thread)
# hanging indefinitely on a flaky mic. The main loop ticks the heartbeat
# by calling _heartbeat() around each of its 'Listening' / 'Recording' /
# 'Sleeping' / '[inject]' status-line prints; a daemon thread polls every
# 10s and, if it's been >60s since the last tick, sets the reset signal
# so the open InputStream's audio_q.get(timeout=0.1) wakes, the
# `with _record_stream:` block exits (closing the stream), and the main
# loop snaps back to its top.
_MAIN_LOOP_HEARTBEAT_TIMEOUT = 60.0   # seconds of silence before recovery
_MAIN_LOOP_WATCHDOG_INTERVAL = 10.0   # how often the watchdog checks
_main_loop_heartbeat = [time.time()]
_watchdog_reset_signal = threading.Event()
_watchdog_stop_event   = threading.Event()   # only for test cleanup


def _heartbeat():
    """Mark the main loop as alive. Called by the main loop just before
    each of its four status-line prints (Listening / Recording / Sleeping
    / [inject]). Also clears any stale reset signal — if we got here, the
    last recovery succeeded and the loop is healthy again."""
    _main_loop_heartbeat[0] = time.time()
    if _watchdog_reset_signal.is_set():
        _watchdog_reset_signal.clear()


def _main_loop_watchdog_check(now: float | None = None,
                              threshold: float | None = None) -> bool:
    """One tick of the watchdog. Returns True if a stall was detected and
    the reset signal was just raised. Separated from the thread loop so a
    unit test can drive a single tick without sleeping."""
    if now is None:
        now = time.time()
    limit = threshold if threshold is not None else _MAIN_LOOP_HEARTBEAT_TIMEOUT
    age = now - _main_loop_heartbeat[0]
    if age > limit and not _watchdog_reset_signal.is_set():
        print(f"[watchdog] main loop stalled — recovering "
              f"(heartbeat {age:.1f}s old)")
        _watchdog_reset_signal.set()
        return True
    return False


def _main_loop_watchdog_thread():
    """Daemon: wake every _MAIN_LOOP_WATCHDOG_INTERVAL seconds and check
    whether the main loop's heartbeat is stale. Best-effort — never let
    a check exception take the thread down."""
    while not _watchdog_stop_event.is_set():
        try:
            _main_loop_watchdog_check()
        except Exception as e:
            print(f"[watchdog] check failed: {e}")
        # Use Event.wait so tests can stop the thread promptly.
        if _watchdog_stop_event.wait(_MAIN_LOOP_WATCHDOG_INTERVAL):
            return


def _process_capture_chunk(chunk: np.ndarray,
                           sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Run a single mic chunk through the noise-cancel-1 pipeline (AEC →
    noise suppression → AGC). Falls through to the raw chunk when the
    processor module isn't available or any stage errors — VAD is run on
    the RAW signal by the caller so processing failures can never affect
    speech endpointing."""
    if not _audio_master_enabled[0] or _audio_processor is None:
        return chunk
    try:
        proc = _audio_processor.get_processor(sample_rate)
        return proc.process(
            chunk,
            enable_aec=bool(_audio_aec_enabled[0]),
            enable_ns=bool(_audio_ns_enabled[0]),
            enable_agc=bool(_audio_agc_enabled[0]),
        )
    except Exception as e:
        if _debug_mode[0]:
            print(f"  [audio-proc] process failed, passing through: {e}")
        return chunk


def _feed_playback_reference(audio: np.ndarray, sr: int) -> None:
    """Hand outgoing TTS audio to the noise-cancel-1 processor so its AEC
    layer has a far-end reference signal. Silent no-op if the module
    isn't loaded."""
    if not _audio_master_enabled[0] or _audio_processor is None:
        return
    try:
        _audio_processor.feed_playback(audio, sample_rate=sr)
    except Exception:
        pass


def _safe_close_stream(stream, timeout_sec: float = 2.0) -> None:
    """Stop+close a sounddevice stream without ever blocking the caller.

    sounddevice.close() at sounddevice.py:1167 has SIGSEGV'd in production
    (faulthandler caught it across multiple PIDs on 2026-05-29). The pattern
    here mirrors the previous barge_stream fix: stop synchronously, then
    close on a daemon thread guarded by `timeout_sec`. If native close hangs
    we force `sd.stop()` and let the daemon die with the process. Use this
    in place of `with sd.InputStream(...)` so context-manager exits — which
    were the unprotected path — never crash the interpreter."""
    if stream is None:
        return
    try:
        stream.stop()
    except Exception:
        logging.exception("[audio] stream.stop raised — proceeding to close")
    done = threading.Event()
    def _do_close():
        try:
            stream.close()
        except Exception:
            logging.exception("[audio] stream.close raised — swallowing")
        finally:
            done.set()
    t = threading.Thread(target=_do_close, daemon=True)
    t.start()
    if not done.wait(timeout=timeout_sec):
        logging.warning("[audio] stream.close hung >%.1fs — forcing sd.stop()",
                        timeout_sec)
        try:
            sd.stop()
        except Exception:
            pass


# ── Live taps on record_speech's mic stream ───────────────────────────────
# get_mic_buffer() (voice enrollment, speaker ID, the standby-audio loop)
# registers a queue here instead of opening a SECOND InputStream on the same
# device. Windows WASAPI starves/garbles concurrent opens on one mic — that
# was the root cause of the 2026-05-30 "records noise for ~70s → watchdog
# reset" stall: the standby-audio loop opened a competing stream every 5s,
# pushing record_speech's frames above the VAD floor so it recorded forever.
# When record_speech holds the mic, skills tap its frames; only when it's
# idle does get_mic_buffer fall back to opening its own stream.
_record_speech_active = [False]          # True while record_speech holds the mic
_tts_playback_active  = [False]          # True while play_with_lipsync owns the speakers + barge-in stream
_record_speech_sr     = [SAMPLE_RATE]    # sample rate of the live stream
_record_speech_taps: "list[queue.Queue]" = []
_record_speech_taps_lock = threading.Lock()

# Hard ceiling on a single utterance. Once recording starts, the only other
# way out is SILENCE_SECS of sustained quiet — but if the mic feeds
# continuous above-threshold audio (music, a noisy room, a stuck driver
# buffer, or contention garbage) silence never lands and we'd record until
# the 60s watchdog kills the whole main loop. No real voice command runs
# this long; cap it and return what we have for transcription.
MAX_RECORDING_SECS = 30.0


def _fanout_record_frame(mono: "np.ndarray") -> None:
    """Push a captured mono frame to any registered taps. Runs on the
    PortAudio callback thread, so it must be cheap and exception-proof — a
    slow or dead tap consumer must never stall mic capture."""
    if not _record_speech_taps:
        return
    with _record_speech_taps_lock:
        taps = list(_record_speech_taps)
    for q in taps:
        try:
            q.put_nowait(mono)
        except Exception:
            pass


def add_record_tap(q: "queue.Queue") -> bool:
    """Register a queue to receive copies of record_speech's mic frames.
    Returns True if record_speech is currently live (so the caller knows the
    tap will actually deliver). Always pair with remove_record_tap in a
    finally."""
    with _record_speech_taps_lock:
        if q not in _record_speech_taps:
            _record_speech_taps.append(q)
    return bool(_record_speech_active[0])


def remove_record_tap(q: "queue.Queue") -> None:
    with _record_speech_taps_lock:
        try:
            _record_speech_taps.remove(q)
        except ValueError:
            pass


def record_speech(timeout: float | None = None) -> np.ndarray | None:
    """
    Blocks until the user speaks and finishes, then returns the audio.
    Returns None if `timeout` seconds pass without any speech starting.

    Uses a callback-based stream because some drivers (notably certain
    gaming headsets) don't honor blocking InputStream.read() — they return
    zero bytes immediately. Callback delivery works on all drivers.
    """
    # Blue/green: staging has no mic. Sleep briefly so the main loop's
    # busy-wait stays cheap, then yield None — the loop's `if audio is
    # None: continue` short-circuits cleanly and the inject drainer at
    # the top of the loop is what actually feeds work into staging.
    if _mic_input_disabled():
        time.sleep(0.5 if timeout is None else min(0.5, max(0.1, timeout)))
        return None
    CHUNK       = 1024
    PRE_BUFFER  = 12
    silence_lim = int(SILENCE_SECS * SAMPLE_RATE / CHUNK)
    pre_ring: list[np.ndarray] = []
    chunks:   list[np.ndarray] = []
    recording   = False
    silence_n   = 0
    start_time  = time.time()
    peak_rms    = 0.0
    silent_peak = 0.0   # peak RMS while NOT recording (= ambient floor)

    audio_q: queue.Queue = queue.Queue()

    def _audio_cb(indata, frames, time_info, status):  # pragma: no cover - live mic stream callback
        # indata is (frames, channels); flatten to 1-D mono
        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        audio_q.put(mono)
        # Fan out to any skill taps so they never open a competing stream
        # on the same mic (WASAPI contention → the ~70s stall). Cheap +
        # exception-proof so it can't stall this callback.
        _fanout_record_frame(mono)

    # Defense-in-depth against a stale cached mic index: even after
    # get_input_device() validates, the device can disappear between
    # that query and InputStream open. Catch PortAudioError and retry
    # once with device=None so we don't crash main().
    try:  # pragma: no cover - opens a live PortAudio input stream (needs real mic)
        _record_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=CHUNK, device=get_input_device(),
            callback=_audio_cb)
    except sd.PortAudioError as e:  # pragma: no cover - live mic open-retry path (needs real device)
        print(f"  [record_speech] InputStream open failed on cached mic ({e}); retrying with system default")
        _device_cache["in"] = None
        _device_cache["checked_at"] = 0.0
        _record_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=CHUNK, device=None,
            callback=_audio_cb)
    # 2026-05-29 silent-crash fix: don't use `with _record_stream:` — the
    # implicit __exit__ calls sd close() unguarded, which SIGSEGV'd during
    # watchdog-driven exits and early returns. Start the stream and route
    # teardown through _safe_close_stream so close runs on a daemon thread.
    try:  # pragma: no cover - starts the live mic stream (needs real mic)
        _record_stream.start()
    except Exception:
        logging.exception("[record_speech] InputStream.start failed")
        _safe_close_stream(_record_stream)
        return None
    # Publish that we now own the mic so get_mic_buffer taps this stream
    # instead of opening a competing one.
    _record_speech_sr[0] = SAMPLE_RATE
    _record_speech_active[0] = True
    record_start_ts = 0.0   # set when recording actually begins (VAD trip)
    try:  # pragma: no cover - live mic capture loop (blocks on real audio frames until utterance ends)
        while True:
            # Watchdog-driven recovery: if the main-loop watchdog has
            # flagged a stall (typically because we were stuck waiting on
            # audio_q.get and nothing was ever delivered), bail out. The
            # finally below closes the InputStream on the way out, which
            # the watchdog comment promises as part of recovery.
            if _watchdog_reset_signal.is_set():
                print("  [record_speech] watchdog reset signalled — "
                      "closing InputStream and returning")
                return None
            try:
                data = audio_q.get(timeout=0.1)
            except queue.Empty:
                # No audio arrived in the last 100ms — check watchdog + timeout
                if _watchdog_reset_signal.is_set():
                    print("  [record_speech] watchdog reset signalled — "
                          "closing InputStream and returning")
                    return None
                if (not recording and timeout is not None and
                        (time.time() - start_time) >= timeout):
                    if _debug_mode[0]:
                        print(f"  [vad] timeout — silent peak RMS={silent_peak:.4f} "
                              f"(threshold {VAD_THRESHOLD})")
                    return None
                continue

            # A frame arrived — the mic callback is alive and the loop is
            # cycling, so the main loop is healthy. Tick the watchdog here
            # (not only at record-start) so a stale heartbeat means "the mic
            # stopped delivering frames", never "the user is mid-utterance".
            # Without this, a long/continuous capture starved the heartbeat
            # and the 60s watchdog reset the loop mid-stream (the 2026-05-30
            # ~70s stall).
            _heartbeat()
            # Hard utterance ceiling: if continuous above-threshold audio
            # kept us recording past the cap, finalize now and let whisper
            # sort it out, rather than recording until the watchdog fires.
            if recording and (time.time() - record_start_ts) > MAX_RECORDING_SECS:
                print(f"  [record_speech] max utterance "
                      f"{MAX_RECORDING_SECS:.0f}s reached — finalizing")
                break

            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms > peak_rms:
                peak_rms = rms

            # Publish a smoothed mic level to the HUD ~10 Hz. RMS is small
            # (≪1) so scale relative to the VAD threshold to fill the gauge.
            now_ts = time.time()
            if HUD_ENABLED and (now_ts - _last_mic_hud_write[0]) > 0.1:
                _last_mic_hud_write[0] = now_ts
                level = min(1.0, rms / max(VAD_THRESHOLD * 4.0, 1e-4))
                _write_hud_state(mic_level=level)

            # Surface VAD activity to skills/self_diagnostic. note_vad_poll
            # fires every chunk so the probe can tell the input loop is
            # actually running; note_vad_active fires only when we cross
            # the VAD floor so the probe can spot a "polling but never
            # tripping" stall. note_raw_rms feeds the silent-mic detector
            # (audible-chunk timestamp updates when rms crosses the hard-
            # ware floor, separate from VAD_THRESHOLD which gates speech).
            # Wrapped in try so an import-time issue with the audio_-
            # processor module can't kill the capture loop.
            try:
                from core import audio_processor as _ap_for_vad
                _ap_for_vad.note_vad_poll(now_ts)
                _ap_for_vad.note_raw_rms(rms, now_ts)
                if rms > VAD_THRESHOLD:
                    _ap_for_vad.note_vad_active(now_ts)
                # Silent-mic health check: if raw RMS has been ≈0 for
                # MIC_SILENT_WARN_SECONDS straight while JARVIS is awake,
                # the mic driver is almost certainly handing us null
                # frames (Windows mic-privacy block, USB unplug, dead
                # capture driver). Emit one warning per silent stretch
                # so the user sees a concrete pointer instead of just
                # "JARVIS isn't hearing me".
                silent_age = _ap_for_vad.seconds_since_audible_chunk()
                if (not _silent_mic_warned[0]
                        and silent_age != float("inf")
                        and silent_age > float(MIC_SILENT_WARN_SECONDS)):
                    print(f"  [vad] WARNING: raw mic RMS has been ≈0 for "
                          f"{silent_age:.0f}s while JARVIS is awake — "
                          f"likely silent-mic hardware/driver fault. "
                          f"Check Windows Privacy → Microphone, verify "
                          f"the active input device, and try unplug/"
                          f"replug if USB. (threshold "
                          f"MIC_SILENT_WARN_SECONDS={MIC_SILENT_WARN_SECONDS})")
                    _silent_mic_warned[0] = True
                elif rms > 1e-5 and _silent_mic_warned[0]:
                    # Mic recovered — allow the warning to re-fire next
                    # time we drop silent for that long.
                    _silent_mic_warned[0] = False
            except Exception:
                pass

            # VAD decision uses the RAW RMS so existing VAD_THRESHOLD
            # tuning is unaffected by the processor's gain stage. We
            # only swap in the processed chunk when we *keep* it.
            processed = _process_capture_chunk(data, SAMPLE_RATE)
            if rms > VAD_THRESHOLD:
                if not recording:
                    recording = True
                    record_start_ts = time.time()
                    chunks.extend(pre_ring)
                    _heartbeat()
                    print("  🎙  Recording…")
                    pause_face_tracking()
                    set_state("listening")
                chunks.append(processed.copy())
                silence_n = 0
            else:
                if not recording and rms > silent_peak:
                    silent_peak = rms
                pre_ring.append(processed.copy())
                if len(pre_ring) > PRE_BUFFER:
                    pre_ring.pop(0)
                if recording:
                    chunks.append(processed.copy())
                    silence_n += 1
                    if silence_n >= silence_lim:
                        break
                elif timeout is not None and (time.time() - start_time) >= timeout:
                    if _debug_mode[0]:
                        print(f"  [vad] timeout — silent peak RMS={silent_peak:.4f} "
                              f"(threshold {VAD_THRESHOLD})")
                    return None
    finally:
        # Release mic ownership BEFORE closing the stream so any tap loop
        # sees _record_speech_active flip false and stops waiting on us.
        _record_speech_active[0] = False
        _safe_close_stream(_record_stream)

    if _debug_mode[0]:
        print(f"  [vad] peak RMS={peak_rms:.4f}  threshold={VAD_THRESHOLD}  "
              f"ambient={silent_peak:.4f}")
    global _last_recording_peak
    _last_recording_peak = peak_rms
    if not chunks:
        return None  # pragma: no cover - defensive: the loop only breaks after recording began, so chunks is never empty here
    return np.concatenate(chunks).flatten()


def apply_capture_auto_gain(audio_f32, peak_rms):
    """CONSERVATIVE input auto-gain (normalization) for the captured mic buffer,
    applied right BEFORE faster-whisper sees it on both the normal-turn and the
    standby/wake path.

    A quiet mic records clear speech at a low peak RMS (~0.01–0.06) where Whisper
    returns an EMPTY string and the wake word "JARVIS" is never detected. This
    boosts such audio toward a usable level WITHOUT degrading already-good audio.

    Returns ``(audio_f32, applied_gain)``:

      * auto-gain disabled, ``peak_rms`` already ≥ the loud-enough target, OR
        ``peak_rms`` at/below a tiny noise floor (so pure silence / room hiss is
        never amplified into Whisper hallucinations) → the audio is returned
        UNCHANGED with ``applied_gain == 1.0``.
      * otherwise the gain is ``min(MAX_GAIN, TARGET_PEAK / max(peak_rms, eps))``;
        the buffer is multiplied and HARD-CLIPPED to ``[-1.0, 1.0]`` to prevent
        overflow distortion, then returned with that gain.

    Never raises: any failure returns the ORIGINAL audio at gain 1.0 so a bad
    sample or a missing config knob can never break the capture path. The
    thresholds are read live from ``core.config`` so the Settings-GUI /
    ``user_settings.json`` override path reaches them.
    """
    try:
        if audio_f32 is None:
            return audio_f32, 1.0
        from core import config as _cfg
        if not getattr(_cfg, "CAPTURE_AUTO_GAIN_ENABLED", True):
            return audio_f32, 1.0
        target = float(getattr(_cfg, "CAPTURE_AUTO_GAIN_TARGET_PEAK", 0.25))
        max_gain = float(getattr(_cfg, "CAPTURE_AUTO_GAIN_MAX", 10.0))
        noise_floor = float(getattr(_cfg, "CAPTURE_AUTO_GAIN_NOISE_FLOOR", 0.005))
        peak = float(peak_rms)
        # Already loud enough, or so quiet it's silence/room hiss → leave it.
        # (peak <= noise_floor guards against amplifying pure noise into
        # hallucinated transcripts; peak >= target means normal audio, untouched.)
        if not (noise_floor < peak < target):
            return audio_f32, 1.0
        gain = min(max_gain, target / max(peak, 1e-9))
        if gain <= 1.0:
            return audio_f32, 1.0
        boosted = np.asarray(audio_f32, dtype=np.float32) * np.float32(gain)
        # Hard-clip so the boost can never overflow [-1, 1] into harsh distortion.
        np.clip(boosted, -1.0, 1.0, out=boosted)
        return boosted, float(gain)
    except Exception:
        # A normalization failure must never break capture — return the input.
        return audio_f32, 1.0


def get_mic_buffer(seconds: float,
                   sample_rate: int | None = None) -> np.ndarray | None:
    """Capture `seconds` of float32 mono audio for skills that need a fixed
    chunk of mic input (voice enrollment, speaker ID, etc.).

    Prefers to tap the wake-word listener's persistent InputStream when one
    is running — Windows WASAPI rejects a second open on the same input
    device, so a naive `sd.rec` from a skill while the listener is up will
    silently fail. Falls back to a fresh InputStream when no persistent
    stream exists, or when its sample rate doesn't match what was asked
    for (resampling would distort voiceprints).

    Returns None on capture failure or if sounddevice isn't available.
    """
    if _mic_input_disabled():
        # Mic hard-disabled (staging / MICROPHONE_INDEX<0): never open a capture
        # stream — get_input_device() would resolve it to the SYSTEM DEFAULT mic
        # and silently listen (this is what made staging transcribe in standby).
        return None
    target_sr = int(sample_rate or SAMPLE_RATE)
    need = max(1, int(target_sr * seconds))

    # Path A: tap the wake-word listener if available and rate-compatible.
    wl = sys.modules.get("skill_wake_listener")
    det = getattr(wl, "_detector", None) if wl is not None else None
    if (det is not None
            and getattr(det, "is_running", lambda: False)()
            and int(getattr(det, "sample_rate", 0)) == target_sr
            and hasattr(det, "add_tap")):
        tap_q: queue.Queue = queue.Queue()
        det.add_tap(tap_q)
        chunks: list[np.ndarray] = []
        captured = 0
        deadline = time.time() + seconds + 1.0
        try:
            while captured < need and time.time() < deadline:
                try:
                    frame = tap_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                chunks.append(frame)
                captured += int(frame.size)
        finally:
            det.remove_tap(tap_q)
        if not chunks:
            return None
        out = np.concatenate(chunks).astype(np.float32, copy=False)
        return out[:need] if out.size > need else out

    # Path A2: tap record_speech's live stream if it owns the mic at a
    # compatible rate. This is the common case (the main loop is almost
    # always inside record_speech) and the whole point of the tap: never
    # open a SECOND InputStream on a mic record_speech is holding — WASAPI
    # contention there caused the 2026-05-30 ~70s capture stall. If
    # record_speech is live we either return its tapped audio or None
    # (skip this cycle); we deliberately do NOT fall through to Path B,
    # because Path B would re-introduce the competing open.
    if _record_speech_active[0] and int(_record_speech_sr[0]) == target_sr:
        tap_q2: queue.Queue = queue.Queue()
        add_record_tap(tap_q2)
        tapped: list[np.ndarray] = []
        got = 0
        deadline = time.time() + seconds + 1.0
        try:
            while got < need and time.time() < deadline:
                if not _record_speech_active[0]:
                    break   # record_speech closed the stream mid-tap
                try:
                    frame = tap_q2.get(timeout=0.2)
                except queue.Empty:
                    continue
                tapped.append(frame)
                got += int(frame.size)
        finally:
            remove_record_tap(tap_q2)
        if not tapped:
            return None
        out_t = np.concatenate(tapped).astype(np.float32, copy=False)
        return out_t[:need] if out_t.size > need else out_t

    # Path B: fall back to a temporary InputStream.
    chunks2: list[np.ndarray] = []
    q_local: queue.Queue = queue.Queue()

    def _cb(indata, frames, time_info, status):  # noqa: ARG001
        mono = indata[:, 0] if indata.ndim > 1 else indata
        q_local.put(mono.astype(np.float32, copy=False).copy())

    # 2026-05-29 silent-crash fix: avoid `with sd.InputStream(...)`. Open the
    # stream explicitly and tear it down via _safe_close_stream so the close
    # path can't SIGSEGV the interpreter during exceptional exits.
    try:
        stream = sd.InputStream(samplerate=target_sr, channels=1,
                                dtype="float32", blocksize=1024,
                                device=get_input_device(), callback=_cb)
    except Exception as e:
        print(f"  [get_mic_buffer] InputStream open failed: {e}")
        return None
    try:
        stream.start()
        captured = 0
        deadline = time.time() + seconds + 1.0
        while captured < need and time.time() < deadline:
            try:
                frame = q_local.get(timeout=0.2)
            except queue.Empty:
                continue
            chunks2.append(frame)
            captured += int(frame.size)
    except Exception as e:
        print(f"  [get_mic_buffer] InputStream failed: {e}")
        return None
    finally:
        _safe_close_stream(stream)
    if not chunks2:
        return None
    out2 = np.concatenate(chunks2).astype(np.float32, copy=False)
    return out2[:need] if out2.size > need else out2


# ──────────────────────────────────────────────────────────────────────────
#  SPEECH-TO-TEXT  (Whisper — fully local)
# ──────────────────────────────────────────────────────────────────────────

_stt = None  # lazy-loaded on first transcribe() call
_stt_device = None      # resolved at load time: "cuda" or "cpu"
_stt_model_name = None  # resolved at load time: actual model loaded
_stt_engine = None      # "faster_whisper" or "openai_whisper"


# Patterns in the stringified exception that indicate the CUDA runtime DLLs
# (cublas64_12.dll / cudnn64_9.dll / cudart64_*.dll) couldn't be loaded.
# When we see one of these we surface remediation steps instead of just
# the raw error, and treat the failure as environmental (auto-fall back to
# CPU; don't keep re-queuing a code-fix task).
_CUDA_DLL_ERROR_PATTERNS = (
    "cublas64", "cudnn64", "cudart64", "nvcuda.dll",
    "is not found or cannot be loaded",
    "could not load library", "library not found",
)


def _is_cuda_dll_error(exc: BaseException) -> bool:
    """True when `exc` looks like a CUDA runtime DLL load failure rather
    than e.g. an out-of-memory or wrong-model-name error. Used to pick
    between 'retry on CPU silently' and 'surface remediation steps'."""
    s = f"{type(exc).__name__}: {exc}".lower()
    return any(p in s for p in _CUDA_DLL_ERROR_PATTERNS)


def _cuda_dll_remediation_note() -> str:
    """Single-line remediation hint for CUDA DLL load failures. Includes
    what _register_cuda_dll_dirs actually saw so the user knows whether
    the pip package is missing vs. just unreachable."""
    reg  = getattr(_register_cuda_dll_dirs, "_registered", []) or []
    miss = getattr(_register_cuda_dll_dirs, "_missing", [])    or []
    reason = getattr(_register_cuda_dll_dirs, "_reason", None)
    parts = [
        "CUDA runtime DLLs (cublas64_12.dll / cudnn64_9.dll) are not loadable.",
        "Fix: pip install --upgrade nvidia-cublas-cu12 nvidia-cudnn-cu12  "
        "(or set WHISPER_DEVICE='cpu' to skip GPU).",
    ]
    if reason:
        parts.append(f"DLL-dir scan: {reason}.")
    elif reg or miss:
        parts.append(f"DLL-dir scan: registered {len(reg)}, missing {len(miss)}.")
    return " ".join(parts)


def _resolve_whisper_device() -> str:
    """Honour WHISPER_DEVICE, falling back to CPU when CUDA isn't actually
    available. Returns 'cuda' or 'cpu'. Prefers ctranslate2's CUDA check
    (faster-whisper backend) over torch.cuda because Py 3.14 + the cu124
    torch wheel doesn't ship yet — but faster-whisper / ctranslate2 has
    its own CUDA runtime and can see the 3090 just fine.

    'cuda' is honoured verbatim even if the runtime check fails; the
    actual load attempt in _ensure_whisper() will then fall back to CPU
    with a clearer error message. 'auto' silently falls back to CPU when
    no GPU backend reports a device."""
    pref = (WHISPER_DEVICE or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda"
    # auto — try ctranslate2 first (covers faster-whisper), then torch
    try:
        import ctranslate2 as _ct2
        if _ct2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception as e:
        print(f"  [whisper] ctranslate2 CUDA check failed: {type(e).__name__}: {e}")
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _register_cuda_dll_dirs() -> None:
    """ctranslate2 needs cublas64_12.dll + cudnn64_9.dll at runtime. The
    nvidia-cublas-cu12 + nvidia-cudnn-cu12 pip packages drop them under
    site-packages\\nvidia\\{cublas,cudnn}\\bin\\ but Python on Windows
    doesn't auto-search those directories. Register them once so the
    DLL loader sees them on the first faster_whisper import.

    Records its outcome on the function object so the STT probe and the
    Whisper loader can surface 'why did CUDA fail' to the user:
      ._done       -> True once registration has been attempted
      ._registered -> list of dirs that were successfully add_dll_directory'd
      ._missing    -> list of dirs that should exist but don't
      ._reason     -> human-readable failure note (None on success)
    """
    if getattr(_register_cuda_dll_dirs, "_done", False):
        return
    _register_cuda_dll_dirs._registered = []
    _register_cuda_dll_dirs._missing    = []
    _register_cuda_dll_dirs._reason     = None
    try:
        import nvidia
    except ImportError as e:
        _register_cuda_dll_dirs._reason = (
            f"nvidia pip namespace not importable ({e}); ctranslate2 will "
            f"fail to find cublas/cudnn — install with: "
            f"pip install --upgrade nvidia-cublas-cu12 nvidia-cudnn-cu12"
        )
        print(f"  [cuda-dll] {_register_cuda_dll_dirs._reason}")
        _register_cuda_dll_dirs._done = True
        return
    except Exception as e:
        _register_cuda_dll_dirs._reason = f"nvidia import raised: {type(e).__name__}: {e}"
        print(f"  [cuda-dll] {_register_cuda_dll_dirs._reason}")
        _register_cuda_dll_dirs._done = True
        return

    try:
        base = list(nvidia.__path__)[0]
    except Exception as e:
        _register_cuda_dll_dirs._reason = f"could not resolve nvidia.__path__: {e}"
        print(f"  [cuda-dll] {_register_cuda_dll_dirs._reason}")
        _register_cuda_dll_dirs._done = True
        return

    # Both PATH prepend AND add_dll_directory: ctranslate2's LoadLibraryEx
    # call chain reaches the PATH-based search before the secure-search
    # paths registered via add_dll_directory in some Python builds, so
    # belt-and-braces is the only reliable cure.
    #
    # History: 2026-05-29 first attempt used add_dll_directory only and
    # silently failed inside ctranslate2 on Python 3.14 — first symptom was
    # `[transcribe] failed: Library cublas64_12.dll is not found`. Adding
    # the PATH prepend fixed it. Re-confirmed 2026-05-30 10:29 after a
    # fresh boot reproduced the same `cublas64_12.dll is not found` error
    # because a code refactor reverted the PATH prepend at some point.
    _existing_path = os.environ.get("PATH", "")
    for sub in ("cublas", "cudnn"):
        bin_dir = os.path.join(base, sub, "bin")
        if not os.path.isdir(bin_dir):
            _register_cuda_dll_dirs._missing.append(bin_dir)
            continue
        # PATH prepend — primary fix for ctranslate2 / faster-whisper.
        if bin_dir not in _existing_path:
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            _existing_path = os.environ["PATH"]
        # add_dll_directory — secondary, for other DLL loaders that DO
        # honour the secure-search list (some pip C extensions do).
        try:
            os.add_dll_directory(bin_dir)
            _register_cuda_dll_dirs._registered.append(bin_dir)
        except (FileNotFoundError, OSError) as e:
            # add_dll_directory failure doesn't undo the PATH prepend,
            # so faster-whisper still works — record the partial success.
            _register_cuda_dll_dirs._missing.append(f"{bin_dir} (add_dll_directory: {e})")
            # Still record as registered since the PATH prepend works.
            _register_cuda_dll_dirs._registered.append(bin_dir)

    if _register_cuda_dll_dirs._registered:
        print(f"  [cuda-dll] registered {len(_register_cuda_dll_dirs._registered)} dir(s): "
              f"{', '.join(os.path.basename(os.path.dirname(d)) for d in _register_cuda_dll_dirs._registered)}")
    if _register_cuda_dll_dirs._missing:
        print(f"  [cuda-dll] WARNING: missing {len(_register_cuda_dll_dirs._missing)} expected dir(s) — "
              f"reinstall with: pip install --upgrade nvidia-cublas-cu12 nvidia-cudnn-cu12")
        if not _register_cuda_dll_dirs._reason:
            _register_cuda_dll_dirs._reason = (
                f"missing CUDA DLL dirs: {_register_cuda_dll_dirs._missing}"
            )
    _register_cuda_dll_dirs._done = True


# Set by the startup preflight (_startup_preflight) when cublas64_12.dll
# cannot be located in any standard install path. _ensure_whisper() reads
# this and forces device='cpu' + compute_type='int8' so we skip the slow
# CUDA load → DLL-error → CPU-fallback dance that bloats boot time.
_force_whisper_cpu_int8 = False


def _ensure_whisper():
    """Load the Whisper model the first time it's actually needed.
    Prefers `faster-whisper` (CTranslate2-based, 2-4x faster on CPU AND
    GPU than openai-whisper, and ships CUDA runtime for Py 3.14 where
    torch's cu124 wheel doesn't exist yet). Falls back to openai-whisper
    if faster-whisper isn't installed.
    Picks `WHISPER_MODEL_CUDA` on GPU, `WHISPER_MODEL_CPU` on CPU."""
    global _stt, _stt_device, _stt_model_name, _stt_engine
    if _stt is not None:
        return
    _register_cuda_dll_dirs()
    device = _resolve_whisper_device()
    # Preflight already verified cublas64_12.dll is missing — short-circuit
    # straight to CPU so we don't waste 5-10s on the CUDA load attempt that
    # the existing inner try/except would have caught anyway.
    if _force_whisper_cpu_int8 and device == "cuda":
        print(f"  [whisper] preflight flagged cublas64_12.dll missing — "
              f"forcing CPU + int8 (skipping CUDA load attempt)")
        device = "cpu"
    model  = WHISPER_MODEL_CUDA if device == "cuda" else WHISPER_MODEL_CPU

    # ── Engine 1: faster-whisper (preferred for GPU speed + Py 3.14 CUDA) ──
    try:
        from faster_whisper import WhisperModel as _FWM
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Loading faster-whisper '{model}' on {device} "
              f"(compute_type={compute_type})…")
        try:
            _stt = _FWM(model, device=device, compute_type=compute_type)
            _stt_engine = "faster_whisper"
            _stt_device = device
            _stt_model_name = model
            print(f"faster-whisper '{model}' ready on {device}.\n")
            return
        except Exception as e:
            if device == "cuda":
                if _is_cuda_dll_error(e):
                    note = _cuda_dll_remediation_note()
                    print(f"  [whisper] faster-whisper CUDA load failed (CUDA DLL "
                          f"missing): {e}")
                    print(f"  [whisper] {note}")
                else:
                    print(f"  [whisper] faster-whisper CUDA load failed: {e}")
                print(f"  [whisper] retrying on CPU with int8…")
                _stt = _FWM(model, device="cpu", compute_type="int8")
                _stt_engine = "faster_whisper"
                _stt_device = "cpu"
                _stt_model_name = model
                print(f"faster-whisper '{model}' ready on cpu (fallback).\n")
                return
            raise
    except ImportError:
        pass  # faster-whisper not installed — fall through to openai-whisper

    # ── Engine 2: openai-whisper (legacy fallback) ─────────────────────────
    print(f"Loading openai-whisper '{model}' on {device}… "
          f"(faster-whisper unavailable — install with: pip install faster-whisper)")
    import whisper as _wlib
    try:
        _stt = _wlib.load_model(model, device=device)
    except Exception as e:
        if device == "cuda":
            if _is_cuda_dll_error(e):
                print(f"  [whisper] CUDA load of '{model}' failed (CUDA DLL "
                      f"missing): {e}")
                print(f"  [whisper] {_cuda_dll_remediation_note()}")
            else:
                print(f"  [whisper] CUDA load of '{model}' failed: {e}")
            print(f"  [whisper] falling back to CPU model '{WHISPER_MODEL_CPU}'.")
            device = "cpu"
            model = WHISPER_MODEL_CPU
            _stt = _wlib.load_model(model, device=device)
        else:
            raise
    _stt_engine = "openai_whisper"
    _stt_device = device
    _stt_model_name = model
    print(f"openai-whisper '{model}' ready on {device}.\n")


# Map pip distribution name → (import module name, JARVIS-style feature note for
# the spoken alert). If the import module differs from the pip name, the lookup
# falls back to the pip name itself. Features listed here generate a one-line
# spoken alert; packages without a feature note get a console warning only.
_DEP_IMPORT_NAME = {
    "openai-whisper": "whisper",
    "opencv-python":  "cv2",
    "edge-tts":       "edge_tts",
    "pillow":         "PIL",
    "pywin32":        "win32com",
    "paho-mqtt":      "paho.mqtt.client",
}
_DEP_FEATURE_NOTE = {
    "psutil":         "the system monitor is offline",
    "paho-mqtt":      "the Bambu printer monitor is offline",
    "opencv-python":  "face tracking is offline",
    "edge-tts":       "text-to-speech is offline",
    "pyttsx3":        "the offline TTS fallback is unavailable — a network blip will silence me",
    "anthropic":      "the Claude backend is offline",
    "mss":            "screen vision is offline",
    "pyautogui":      "UI automation is offline",
    "pywin32":        "Windows automation is offline",
    "pystray":        "the system tray applet is offline",
}


def _parse_requirements(path: str) -> list[str]:
    """Pull pip-distribution names out of requirements.txt. Skips comments,
    blank lines, and trailing inline comments. Strips version specifiers and
    extras so 'paho-mqtt>=2.0  # comment' → 'paho-mqtt'."""
    pkgs: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                # Skip pip option lines (--extra-index-url, --index-url, -r,
                # -e, -c, …) — they're directives, not package names, and
                # were being mis-parsed into a bogus '--extra-index-url'
                # "missing package" warning at boot.
                if line.startswith("-"):
                    continue
                # Strip version specifiers / extras / env markers
                name = re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0].strip()
                if name:
                    pkgs.append(name)
    except Exception as e:
        print(f"  [dep-check] could not read {path}: {e}")
    return pkgs


def check_dependencies() -> list[str]:
    """Walk requirements.txt, report any packages whose import fails, and
    speak a one-line alert naming the silently-disabled features. Returns
    the list of missing pip-distribution names (empty list = all good)."""
    import importlib
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    pkgs = _parse_requirements(req_path)
    missing: list[str] = []
    for pkg in pkgs:
        mod = _DEP_IMPORT_NAME.get(pkg, pkg.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)

    if not missing:
        print("  [dep-check] all requirements.txt packages importable.")
        return missing

    print(f"  [dep-check] WARNING: {len(missing)} package(s) from requirements.txt "
          f"are missing from the active environment:")
    for pkg in missing:
        feat = _DEP_FEATURE_NOTE.get(pkg, "")
        suffix = f"  ({feat})" if feat else ""
        print(f"    - {pkg}{suffix}")
    print(f"  [dep-check] install with:  pip install {' '.join(missing)}")

    # Build the spoken alert: one line, JARVIS-style understatement. Only
    # mention features we have a human-readable note for; if multiple, list
    # the first two so the alert stays short.
    feature_msgs = [(_DEP_FEATURE_NOTE[p], p) for p in missing if p in _DEP_FEATURE_NOTE]
    if feature_msgs:
        if len(feature_msgs) == 1:
            feat, pkg = feature_msgs[0]
            alert = f"Sir, {feat} — {pkg} is missing."
        else:
            parts = ", ".join(f"{feat} ({pkg} missing)" for feat, pkg in feature_msgs[:2])
            extra = "" if len(feature_msgs) <= 2 else f", and {len(feature_msgs) - 2} more"
            alert = f"Sir, a few things are offline: {parts}{extra}."
        try:
            _speak(alert)
            conversation_history.append({"role": "assistant", "content": alert})
        except Exception as e:
            print(f"  [dep-check] spoken alert failed: {e}")

    return missing


def transcribe(audio: np.ndarray) -> tuple[str, dict]:
    """Returns (text, confidence) where confidence has no_speech_prob and avg_logprob.
    Abstracts over faster-whisper (preferred, GPU-accelerated on 3090) and
    openai-whisper (legacy fallback). Both produce the same return shape."""
    global _stt
    try:
        _ensure_whisper()
        if _stt_engine == "faster_whisper":
            # faster-whisper returns (segments_generator, info). Drain to a
            # list so we can compute averages. vad_filter=True skips
            # 'inaudible' chunks; same model dims as openai-whisper.
            # vad_filter=True runs faster-whisper's built-in Silero VAD before
            # transcribing; on quiet/desk mics it frequently drops LEGITIMATE
            # speech (a real "JARVIS" scores below the gate) and returns zero
            # segments -> "". Use a permissive threshold, and if it STILL finds
            # nothing, retry ONCE WITHOUT the VAD filter so a genuine utterance
            # is never silently lost — the difference between "heard you" and a
            # wall of [standby] ignored: ''. The caller's audio already cleared
            # the mic VAD gate, so it is not pure silence.
            segments_gen, info = _stt.transcribe(
                audio, language="en",
                vad_filter=True,
                vad_parameters=dict(threshold=0.3, min_speech_duration_ms=80),
                beam_size=5,
            )
            segments = list(segments_gen)
            if not segments:
                segments_gen, info = _stt.transcribe(
                    audio, language="en",
                    vad_filter=False,
                    beam_size=5,
                )
                segments = list(segments_gen)
            text = " ".join((s.text or "").strip() for s in segments).strip()
            if not segments:
                # info still carries some signal even on empty transcription
                nsp = float(getattr(info, "no_speech_prob", 1.0) or 1.0)
                return text, {"no_speech_prob": nsp, "avg_logprob": -10.0}
            n = len(segments)
            no_speech = sum(float(getattr(s, "no_speech_prob", 0.0) or 0.0)
                            for s in segments) / n
            logprob   = sum(float(getattr(s, "avg_logprob", 0.0) or 0.0)
                            for s in segments) / n
            return text, {"no_speech_prob": no_speech, "avg_logprob": logprob}

        # openai-whisper path (legacy). Modern openai-whisper (v20250115+)
        # removed the fp16= kwarg — precision is now derived from the dtype
        # the model was loaded with, so passing it raises TypeError.
        result = _stt.transcribe(audio, language="en")
        text = result["text"].strip()
        segments = result.get("segments", [])
        if not segments:
            return text, {"no_speech_prob": 1.0, "avg_logprob": -10.0}
        n = len(segments)
        no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / n
        logprob   = sum(s.get("avg_logprob",    0.0) for s in segments) / n
        return text, {"no_speech_prob": no_speech, "avg_logprob": logprob}
    except Exception as e:
        _err = f"{type(e).__name__}: {e}"
        print(f"  [transcribe] failed: {_err}")
        # On a CUDA/VRAM error (the standby whisper-tiny + the main large
        # model + Ollama/vision can crowd the 3090's 24 GB), free fragmented
        # VRAM and DROP the model so the next utterance reloads from a clean
        # state instead of repeatedly failing from a half-broken GPU context.
        # The current utterance is lost (the user re-speaks) but recovery is
        # automatic. 2026-05-30 deep audit.
        if any(k in _err.lower() for k in
               ("out of memory", "cuda", "cublas", "cudnn", "cudart")):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            _stt = None
            print("  [transcribe] dropped GPU model after CUDA error — "
                  "reloading on next utterance")
        return "", {"no_speech_prob": 1.0, "avg_logprob": -10.0}


# is_ambient_music / is_valid_speech + _MUSIC_MARKERS moved to
# core/speech_filter.py (re-exported at the top of this file).


# ──────────────────────────────────────────────────────────────────────────
#  TONE DETECTOR  (cheap pre-LLM pass over the user's transcript)
# ──────────────────────────────────────────────────────────────────────────
#
# MCU JARVIS reads Tony's mood and adjusts his register on the fly. We
# mimic that with a tiny pure-Python classifier that runs on every
# transcribed utterance BEFORE the LLM call. Detected stress markers —
# swearing, urgency words, clipped imperatives, exclamation, or the user
# repeating themselves across turns — produce a per-turn addendum to the
# system prompt instructing JARVIS to skip pleasantries and be terse.

# Tone-classifier DATA + heuristics moved to core/tone_detector.py (imported at
# the top of this file). The per-utterance caches below stay here — they're
# runtime state, not classification logic.

# Cache the most recent user utterance's tone so get_followup_response
# can reuse it without re-running the detector (and to keep the register
# stable across a multi-step action chain).
_last_user_tone: list[str | None] = [None]

# Cache the most recent user utterance text so the context-aware TTS layer
# (core.tts.detect_emergency_keywords) can score 'fuck'/'shit'/'help' on
# whatever sir just said. Set in _call_llm before LLM dispatch, read by
# _resolve_tts_preset(). Mirrors _last_user_tone's lifecycle.
_last_user_text: list[str | None] = [None]

# Five-label emotion classification from core/emotion_tracker — cached per
# utterance so the follow-up loop reuses the same register without re-running
# the heuristics. Stored as the EmotionResult so synthesise() can fetch the
# preset name without re-classifying.
_last_emotion: list[object | None] = [None]


def detect_tone(user_text: str) -> str | None:
    """Thin wrapper over core.tone_detector.detect_tone: supply the user's
    PREVIOUS utterance from conversation_history so the pure classifier can
    spot cross-turn repetition, then delegate. Kept here (not re-exported)
    so every existing call site detect_tone(text) stays unchanged."""
    prev = next(
        (m["content"] for m in reversed(conversation_history[:-1])
         if m.get("role") == "user"),
        None,
    )
    return _tone_detector.detect_tone(user_text, prev_user_text=prev)


# Voice-emotion mood router (route_voice_emotion + _VOICE_MOOD_HINTS +
# _detect_excited + VOICE_EMOTION_ROUTER_ENABLED) moved to
# core/voice_emotion.py. The _last_voice_route cache below stays here
# (runtime state); route_voice_emotion is re-exported via the thin wrapper.

# Module-level cache of the route picked for the most recent user
# utterance. Mirrors `_last_user_tone[0]` so synthesise() and
# get_followup_response() can read the route without re-running the
# classifier (and so register stays stable across action chains).
_last_voice_route: list[dict | None] = [None]


def route_voice_emotion(user_text: str, now: float | None = None) -> dict:
    """Thin wrapper over core.voice_emotion.route_voice_emotion: supply the
    previous user utterance from conversation_history so the router's tone
    classifier can spot cross-turn repetition, then delegate. Call sites
    are unchanged."""
    prev = next(
        (m["content"] for m in reversed(conversation_history[:-1])
         if m.get("role") == "user"),
        None,
    )
    return _voice_emotion.route_voice_emotion(
        user_text, now=now, prev_user_text=prev)


# ──────────────────────────────────────────────────────────────────────────
#  LLM  (Claude or Ollama)
# ──────────────────────────────────────────────────────────────────────────

# Local-LLM fallback state — guarded so the winget install + ollama pull
# fire at most once per process. Set true on first attempt so a failed
# install doesn't hammer winget on every subsequent rate-limit.
_OLLAMA_INSTALL_TRIGGERED = [False]
_OLLAMA_PULL_TRIGGERED    = [False]


def _log_gpu_state(model_name: str) -> None:
    """One-shot nvidia-smi snapshot helper. Confirms Ollama is actually
    holding the model in VRAM on first use. Never raises — silently
    no-ops on non-NVIDIA hosts. Defers to core.gpu_state for the
    shared dedup set so embeddings + LLM + VLM share one log entry per
    model across the process."""
    try:
        from core import gpu_state as _gpu_state
        _gpu_state.log_gpu_state(model_name)
    except Exception:
        # Diagnostics must never block the Ollama call path.
        pass

# Resolved local-LLM model tag — filled on the first call to
# _get_local_llm_model() that actually finds an installed model (or honours
# the env override). Until then the resolver returns LOCAL_LLM_MODEL without
# caching, so a finished background pull promotes the user to qwen2.5:14b on
# the next call instead of being locked to a stale 'no models installed'
# choice. Single-element list so reassignment doesn't need a global stmt.
_RESOLVED_LOCAL_LLM_MODEL: list[str | None] = [None]

# Fallback chain for the local-LLM selector. First entry is the preferred
# default (smarter on ambiguous voice intents, ~10 GB VRAM on the 3090);
# second entry is the lower-VRAM legacy option. JARVIS_LOCAL_LLM_MODEL
# bypasses this list entirely.
_LOCAL_LLM_PREFERENCE = (
    "qwen2.5:14b-instruct-q5_K_M",
    "llama3.1:8b-instruct-q5_K_M",
)


def _ollama_alive() -> bool:
    try:
        return requests.get(f"{LOCAL_LLM_BASE_URL}/api/tags", timeout=2).ok
    except Exception:
        return False


def _get_local_llm_model() -> str:
    """Resolve which Ollama tag the local-LLM fallback should target.

    Priority order:
      1. JARVIS_LOCAL_LLM_MODEL env var (non-empty after stripping).
      2. First entry in _LOCAL_LLM_PREFERENCE that's installed locally
         (matched exactly, then by bare base name so a `:latest` variant
         counts).
      3. First tag returned by Ollama's /api/tags — any installed model
         beats none, even off-list.
      4. LOCAL_LLM_MODEL — the preferred default. Returned (without
         caching) when Ollama is unreachable or has zero models, so the
         background pull triggered by _call_local_llm() targets the
         smart model and a finished pull is picked up on the next call.

    The selection is cached in _RESOLVED_LOCAL_LLM_MODEL once a real
    installed model (or the env override) is found, so the 'model
    selected' log line fires exactly once per process.
    """
    cached = _RESOLVED_LOCAL_LLM_MODEL[0]
    if cached:
        return cached

    override = (os.environ.get("JARVIS_LOCAL_LLM_MODEL") or "").strip()
    if override:
        _RESOLVED_LOCAL_LLM_MODEL[0] = override
        print(f"  [local-llm] model selected via JARVIS_LOCAL_LLM_MODEL: `{override}`")
        _log_gpu_state(override)
        return override

    try:
        r = requests.get(f"{LOCAL_LLM_BASE_URL}/api/tags", timeout=2)
        installed = [m.get("name", "") for m in r.json().get("models", [])] if r.ok else []
    except Exception:
        installed = []
    installed = [n for n in installed if n]

    if installed:
        installed_set = set(installed)
        for pref in _LOCAL_LLM_PREFERENCE:
            if pref in installed_set:
                pick = pref
            else:
                base = pref.split(":", 1)[0]
                pick = next((n for n in installed if n.split(":", 1)[0] == base), None)
            if pick:
                _RESOLVED_LOCAL_LLM_MODEL[0] = pick
                print(f"  [local-llm] model selected from preference chain: `{pick}`")
                _log_gpu_state(pick)
                return pick
        pick = installed[0]
        _RESOLVED_LOCAL_LLM_MODEL[0] = pick
        print(f"  [local-llm] no preferred model installed — using first available: `{pick}`")
        _log_gpu_state(pick)
        return pick

    return LOCAL_LLM_MODEL


def _ollama_has_model(model: str) -> bool:
    try:
        r = requests.get(f"{LOCAL_LLM_BASE_URL}/api/tags", timeout=2)
        if not r.ok:
            return False
        # Tags can be exact ("llama3.1:8b-instruct-q5_K_M") or carry the
        # implicit ":latest" suffix when the user pulled by bare name.
        names = {m.get("name", "") for m in r.json().get("models", [])}
        return model in names or any(n.split(":", 1)[0] == model.split(":", 1)[0] for n in names)
    except Exception:
        return False


def _ollama_install_async() -> None:
    if _OLLAMA_INSTALL_TRIGGERED[0]:
        return
    _OLLAMA_INSTALL_TRIGGERED[0] = True

    def _do() -> None:
        print("  [local-llm] Ollama not detected — running `winget install Ollama.Ollama` in background…")
        try:
            import subprocess as _sp
            _sp.run(
                ["winget", "install", "--id", "Ollama.Ollama",
                 "--silent", "--accept-package-agreements", "--accept-source-agreements"],
                capture_output=True, text=True, timeout=600,
            )
            print(f"  [local-llm] winget install finished. Once Ollama is up, model `{LOCAL_LLM_MODEL}` will pull automatically.")
        except Exception as _e:
            print(f"  [local-llm] winget install failed: {_e}")

    threading.Thread(target=_do, daemon=True).start()


def _ollama_pull_async(model: str) -> None:
    if _OLLAMA_PULL_TRIGGERED[0]:
        return
    _OLLAMA_PULL_TRIGGERED[0] = True

    def _do() -> None:
        print(f"  [local-llm] pulling model `{model}` in background…")
        try:
            with requests.post(
                f"{LOCAL_LLM_BASE_URL}/api/pull",
                json={"name": model}, timeout=1800, stream=True,
            ) as r:
                for _ in r.iter_lines():
                    pass
            print(f"  [local-llm] model `{model}` ready.")
        except Exception as _e:
            print(f"  [local-llm] model pull failed: {_e}")

    threading.Thread(target=_do, daemon=True).start()


# Reinforces the essentials for the LOCAL baseline model (qwen2.5). It follows
# the action-token grammar + JARVIS persona less reliably than Claude, which in
# the runtime logs produced "claims it did X without doing it", verbose
# rambling, and unsolicited moralising. Appended at the FINAL (most-salient)
# position of the system prompt on every local call. 2026-05-30.
_LOCAL_MODE_DIRECTIVE = (
    "\n\n----------\n"
    "YOU ARE RUNNING ON THE LOCAL MODEL. Obey these rules EXACTLY:\n"
    "1. To DO anything on the computer (play music, set a timer, open an "
    "app or website, click, control devices, search the web, take a "
    "screenshot), you MUST output the literal token [ACTION: name, argument] "
    "— for example [ACTION: set_timer, 5 minutes] or [ACTION: play_music, "
    "Michael Jackson]. Do NOT describe doing it without the token.\n"
    "2. Use ONLY action names from the list above. NEVER invent an action "
    "(no [ACTION: calculate], [ACTION: answer], [ACTION: think], etc.). For "
    "anything you can answer from your own knowledge — arithmetic, facts, "
    "definitions, opinions, banter — just SAY the answer in one short "
    "sentence with NO token (e.g. 'Three hundred ninety-one, sir.').\n"
    "3. NEVER claim you have completed, queued, started, paused, resumed, or "
    "'taken the liberty of' something unless you emitted its [ACTION: ...] "
    "token in THIS reply. If you cannot do it, say so plainly in one sentence.\n"
    "4. Be CONCISE — one or two short sentences. Your words are spoken aloud.\n"
    "5. Do NOT lecture, moralise, or add unsolicited safety/health advice. "
    "Dry wit is welcome; the user is fine with profanity.\n"
    "6. NEVER invent live numbers you were not explicitly given — print "
    "percentage, current/total layer count, ETA, temperatures, prices, "
    "counts. If you lack an exact figure, say it plainly ('the print's still "
    "going, sir') or use the matching action (e.g. [ACTION: print_status]). "
    "Do NOT guess a total like 'layer 105 of 300' when the total is unknown, "
    "and do NOT claim you paused/started/stopped a print you didn't act on.\n"
    "7. Stay in character as JARVIS: composed, capable, faintly sardonic; "
    "address the user as 'sir'.\n"
    "----------\n"
)


_LOCAL_CHEATSHEET_CACHE: list[str | None] = [None]


def _local_cheatsheet() -> str:
    """A COMPACT action reference that REPLACES the ~18k-token PC_CONTROL_PROMPT
    for local-model calls.

    The full Claude-tuned PC_CONTROL_PROMPT is ~18,400 tokens. Feeding that to a
    14B local model every turn is the core reason local mode felt bad: (1) the
    prompt-eval of ~23k total tokens dominates latency, (2) instruction-
    following in a 14B model collapses on 20k+ token prompts ("lost in the
    middle") → hallucinated action execution + rambling, and (3) it forces a
    32k context whose KV cache (~9 GB) spills the model partly onto the CPU.

    This keeps the EXACT [ACTION: name, arg] grammar plus the most common
    actions spelled out, then appends EVERY registered action name straight
    from the live ACTIONS registry so nothing is unreachable and the list never
    drifts out of sync. ~2k tokens instead of ~18k. Cached after first build.
    """
    cached = _LOCAL_CHEATSHEET_CACHE[0]
    if cached is not None:
        return cached
    common = (
        "\n\n=== CONTROLLING THE PC ===\n"
        "To DO something, output a token EXACTLY like:  [ACTION: name, argument]\n"
        "Put it on its own line; the system runs it and hands you the result to\n"
        "comment on. Emit an action ONLY when sir wants something DONE — for\n"
        "ordinary conversation, just reply normally with no token.\n\n"
        "*** ALWAYS USE THE ACTION — NEVER GUESS THESE ***\n"
        "You do NOT know the current time, date, your version, the weather, or\n"
        "any system stat without running its action. NEVER state them from\n"
        "memory — emit the token and let the system fill in the real value:\n"
        "  [ACTION: get_time]            \"what time is it\" / \"what's the date\"\n"
        "  [ACTION: version_info]        \"what version are you on\" / \"when were you updated\"\n"
        "  [ACTION: weather_briefing]    \"what's the weather\" / \"is it going to rain\"\n"
        "  [ACTION: system_pulse]        \"system status\" / \"how are you running\" / CPU/RAM\n"
        "  [ACTION: whats_broken]        \"what's broken\" / \"anything wrong\"\n"
        "  [ACTION: list_timers]         \"list my timers\" / \"what timers are running\"\n"
        "If sir asks any of the above, your reply must contain ONLY the action\n"
        "token (plus at most a short lead-in like \"One moment, sir.\"). Do not\n"
        "invent a time, version number, temperature, or status — you will be\n"
        "wrong.\n\n"
        "Most-used actions:\n"
        "  [ACTION: play_music, <artist/song/playlist>]   play music in the browser\n"
        "  [ACTION: play_playlist, <name>]   play ANY named playlist — prefers sir's local iTunes library, auto-falls back to Apple Music streaming if not owned ('shuffle ' prefix shuffles). Use this for every 'play my/the <name> playlist', NOT apple_music.\n"
        "  [ACTION: list_playlists]   list sir's iTunes playlists   [ACTION: shuffle_library]   shuffle all music\n"
        "  [ACTION: pause_music]  [ACTION: resume_music]  [ACTION: next_song]\n"
        "  [ACTION: media_playpause]  [ACTION: media_next]  [ACTION: media_prev]\n"
        "  [ACTION: volume_up]  [ACTION: volume_down]  [ACTION: volume_mute]\n"
        "  [ACTION: netflix, <title>]  [ACTION: youtube, <search>]  [ACTION: spotify, <query>]\n"
        "  [ACTION: apple_music, <query>]  [ACTION: disney_plus, <title>]  [ACTION: hulu, <title>]\n"
        "  [ACTION: set_timer, 5 minutes]   (or '5 minutes for tea')   [ACTION: list_timers]   [ACTION: cancel_timer]  (no arg cancels the running one)\n"
        "  [ACTION: see_screen]  or  [ACTION: see_screen, middle]   look at the screen & describe it\n"
        "  [ACTION: find_on_screen, <thing>]   [ACTION: recall_screen]\n"
        "  [ACTION: web_search, <query>]   open a web search in the browser\n"
        "  [ACTION: open_url, <url>]   [ACTION: launch_app, <app name>]\n"
        "  [ACTION: open_on_monitor, <app or url> | <left|middle|right|top>]\n"
        "  [ACTION: type, <text to type>]   [ACTION: click, <what to click>]\n"
        "  [ACTION: move_window_to_monitor, <left|middle|right|top>]\n"
        "  [ACTION: minimize_window]   [ACTION: close_window]\n"
        "  [ACTION: check_system]   [ACTION: system_pulse]   [ACTION: check_print]\n"
        "  [ACTION: weather_briefing]   [ACTION: news_briefing]   [ACTION: morning_briefing]\n"
        "  [ACTION: queue_task, <task for Claude Code>]   [ACTION: show_tasks]\n"
        "  [ACTION: smart_home_control, <plain request>]   [ACTION: make_picture, <prompt>]\n"
    )
    # Append EVERY registered action name (late-bound from the live registry —
    # ACTIONS is fully populated by the time this runs at request time).
    allnames = ""
    try:
        names = sorted({k for k in ACTIONS.keys() if isinstance(k, str)})
        if names:
            allnames = (
                "\nEvery available action (same [ACTION: name, arg] grammar — "
                "use these exact names):\n  " + ", ".join(names) + "\n"
            )
    except Exception:
        pass
    tail = (
        "\nACTION RULES:\n"
        "- NEVER say you did / started / queued / opened / set something unless\n"
        "  you emitted its [ACTION: ...] token in THIS reply. If you cannot do\n"
        "  it, say so plainly in one sentence — do not pretend.\n"
        "- One action per reply unless the task plainly needs more.\n"
        "- The argument is plain text after the comma — no JSON, no quotes.\n"
        "=== END PC CONTROL ===\n"
    )
    out = common + allnames + tail
    _LOCAL_CHEATSHEET_CACHE[0] = out
    return out


def _call_local_llm(system: str, messages: list, max_tokens: int = 500) -> str | None:
    """POST to Ollama /api/chat. Returns the assistant text or None on any
    failure (no Ollama running, no model pulled, HTTP error, timeout)."""
    if not LOCAL_LLM_FALLBACK:
        return None
    if not _ollama_alive():
        _ollama_install_async()
        return None
    model = _get_local_llm_model()
    if not _ollama_has_model(model):
        _ollama_pull_async(model)
        return None
    # Anti-hallucination guard: if the recent conversation shows a web_search
    # was fired but no subsequent see_screen read the results, the local LLM
    # tends to fabricate source attributions ("from census data…"). Scan the
    # last few assistant messages narrowly to avoid false positives from
    # trimmed history.
    # Swap the ~18k-token PC_CONTROL_PROMPT for the compact local cheatsheet —
    # the single biggest lever on local quality + speed (see _local_cheatsheet).
    # Everything else (persona, memory, per-turn tone hint) is preserved.
    sys_prompt = system
    try:
        if PC_CONTROL_PROMPT and PC_CONTROL_PROMPT in sys_prompt:
            sys_prompt = sys_prompt.replace(PC_CONTROL_PROMPT, _local_cheatsheet())
    except Exception:
        pass
    try:
        recent = messages[-6:] if isinstance(messages, list) else []
        last_search_idx = -1
        last_see_idx = -1
        for i, m in enumerate(recent):
            if not isinstance(m, dict) or m.get("role") != "assistant":
                continue
            content = m.get("content") or ""
            if not isinstance(content, str):
                continue
            if "[ACTION: web_search" in content:
                last_search_idx = i
            if "[ACTION: see_screen" in content:
                last_see_idx = i
        if last_search_idx >= 0 and last_see_idx <= last_search_idx:
            sys_prompt = (
                "IMPORTANT: a web search was just fired but the results have "
                "not been read. Do NOT fabricate or claim source attributions. "
                "Acknowledge the search was opened in the browser and offer to "
                "read the results via see_screen.\n\n"
            ) + sys_prompt
    except Exception:
        pass
    # Reinforce the local-mode behaviour rules at the most-salient (final)
    # position — addresses the hallucinated-execution / verbose / moralising
    # replies the giant Claude prompt produced on a 14B model.
    sys_prompt = sys_prompt + _LOCAL_MODE_DIRECTIVE
    try:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": sys_prompt}] + messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                # Cap context at 16k (was the model-default 32k). With the
                # compact prompt the real prompt is ~8k, so 16k leaves ample
                # room for history + generation while HALVING the KV cache so
                # the model fits 100% on the GPU (no CPU spill → much faster).
                "num_ctx": 16384,
                # Lower than Ollama's 0.8 default → more focused, less rambly,
                # better at emitting the exact action-token grammar.
                "temperature": 0.4,
                "top_p": 0.9,
            },
            # Keep the model resident between turns so a voice burst doesn't pay
            # the ~3-5s reload each time (it competes with whisper on reload).
            "keep_alive": "20m",
        }
        r = requests.post(f"{LOCAL_LLM_BASE_URL}/api/chat", json=payload, timeout=120)
        if not r.ok:
            print(f"  [local-llm] HTTP {r.status_code}: {r.text[:200]}")
            return None
        text = ((r.json().get("message") or {}).get("content") or "").strip()
        if not text:
            return None
        print(f"  [local-llm] served via {model}")
        return text
    except Exception as _e:
        print(f"  [local-llm] call failed: {_e}")
        return None


def _local_fallback_or(sys_prompt: str, default_reply: str) -> str:
    """Try the local LLM; on success prepend `[local] `, else return the
    cloud-side error message untouched."""
    text = _call_local_llm(sys_prompt, conversation_history)
    if not text:
        return default_reply
    # Deliberately NO "[local]" prefix on the reply — it was being spoken
    # aloud ("local, sir…") every turn during the API cap and the user asked
    # for it to stop (2026-05-30). The per-call `[local-llm] served via …`
    # console line already records that the local model answered, so the
    # debug signal is preserved without polluting the voiced/displayed text.
    # Strip a stale leading tag too, in case one is already present.
    t = text.lstrip()
    if t.startswith("[local]"):
        t = t[len("[local]"):].lstrip()
    return t


# ── Local VLM (vision) over the same Ollama instance ─────────────────────
_LOCAL_VISION_PULL_TRIGGERED = [False]


def _ollama_pull_vision_async(model: str) -> None:
    """Background-pull a vision model. Separate trigger flag so the text
    fallback's pull and the vision fallback's pull don't block each other."""
    if _LOCAL_VISION_PULL_TRIGGERED[0]:
        return
    _LOCAL_VISION_PULL_TRIGGERED[0] = True

    def _do() -> None:
        print(f"  [local-vision] pulling VLM `{model}` in background…")
        try:
            with requests.post(
                f"{LOCAL_LLM_BASE_URL}/api/pull",
                json={"name": model}, timeout=3600, stream=True,
            ) as r:
                for _ in r.iter_lines():
                    pass
            print(f"  [local-vision] VLM `{model}` ready.")
        except requests.RequestException as _e:
            # Reset the trigger so a transient network failure doesn't
            # permanently block local vision until the process restarts.
            _LOCAL_VISION_PULL_TRIGGERED[0] = False
            print(f"  [local-vision] VLM pull failed: {_e}")

    threading.Thread(target=_do, daemon=True).start()


def _call_local_vision(question: str, png_images: list[bytes],
                       max_tokens: int = 600) -> str | None:
    """POST a vision request to Ollama's /api/chat with one or more PNGs.

    Returns the assistant text on success, or None if local vision is
    disabled, Ollama isn't reachable, the VLM isn't pulled (a background
    pull is kicked off in that case), or the HTTP call fails. The caller
    is responsible for prepending the `[local-vision]` tag on success."""
    if not LOCAL_VISION_FALLBACK or not LOCAL_VISION_MODEL:
        return None
    if not png_images:
        return None
    if not _ollama_alive():
        _ollama_install_async()
        return None
    if not _ollama_has_model(LOCAL_VISION_MODEL):
        _ollama_pull_vision_async(LOCAL_VISION_MODEL)
        return None
    _log_gpu_state(LOCAL_VISION_MODEL)
    try:
        b64_images = [base64.standard_b64encode(p).decode("utf-8") for p in png_images]
        payload = {
            "model": LOCAL_VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": question,
                "images": b64_images,
            }],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        # Vision calls on a 7B VLM take ~2-6 s on the 3090; allow generous
        # headroom for the first call after a model swap (Ollama lazily
        # loads weights into VRAM on first hit).
        r = requests.post(f"{LOCAL_LLM_BASE_URL}/api/chat", json=payload, timeout=180)
        if not r.ok:
            print(f"  [local-vision] HTTP {r.status_code}: {r.text[:200]}")
            return None
        try:
            body = r.json()
        except ValueError as _je:
            print(f"  [local-vision] non-JSON response: {_je}; body={r.text[:200]!r}")
            return None
        msg = body.get("message")
        if not isinstance(msg, dict):
            print(f"  [local-vision] unexpected response shape (no message dict): keys={list(body.keys())}")
            return None
        text = (msg.get("content") or "").strip()
        if not text:
            print(f"  [local-vision] empty content; done_reason={body.get('done_reason')!r}")
            return None
        print(f"  [local-vision] served via {LOCAL_VISION_MODEL}")
        return text
    except requests.RequestException as _e:
        print(f"  [local-vision] HTTP call failed: {_e}")
        return None


def _call_llm(user_text: str) -> str:
    conversation_history.append({"role": "user", "content": user_text})

    # Classify mood from this utterance and (if non-default) extend the
    # system prompt for this single turn. Cache the label so the follow-up
    # call stays in the same register without re-running the detector.
    tone = detect_tone(user_text)
    _last_user_tone[0] = tone
    _last_user_text[0] = user_text
    if tone:
        print(f"  [tone] {tone}")

    # Voice emotion router: fuses text-tone with time-of-day into a single
    # mood label and emits a per-turn system-prompt addendum (reply length
    # + register). TTS prosody for the same mood is picked downstream by
    # synthesise() via _USER_TONE_TTS. Cached so the follow-up call stays
    # in the same register without re-classifying.
    route = route_voice_emotion(user_text)
    _last_voice_route[0] = route
    if route["mood"] != "casual":
        print(f"  [voice-mood] {route['mood']}")

    # Voice-mood response: closes the loop on a 'stressed' label by
    # suppressing proactive nudges for 15 min, dimming Hue/Govee lights to
    # warm 2700K on a daemon thread, and returning an extra-deferential
    # addendum to stack onto sys_prompt_now. Returns '' for every other
    # mood so non-stressed turns are unchanged.
    voice_mood_addendum = ""
    if _voice_mood_response is not None:
        try:
            voice_mood_addendum = _voice_mood_response.apply_voice_mood_response(
                route,
                user_text,
                memory_lock=_memory_lock,
                load_memory=load_memory,
                save_memory=save_memory,
            )
        except Exception as _vm_err:
            print(f"  [voice-mood] response adapter failed: {_vm_err}")

    # Five-label emotion tracker (core/emotion_tracker). Adds a 'focused'
    # bucket the legacy detector lacks and maps each label to a TTS preset
    # (stressed→calm, excited→amused, focused→briefing, tired→concerned).
    # Its addendum stacks on top of detect_tone()'s — they overlap on the
    # shared labels but emphasise different facets of the response strategy.
    emotion_addendum = ""
    if _emotion_tracker is not None:
        try:
            er = _emotion_tracker.classify_emotion(user_text)
            _last_emotion[0] = er
            if er and er.label:
                print(f"  [emotion] {er.label} ({er.reason})")
                # Only stack the new addendum when its label differs from
                # what detect_tone already covered — avoids duplicating the
                # same directive twice in the system prompt.
                if er.label != tone:
                    emotion_addendum = er.addendum
        except Exception as _err:
            print(f"  [emotion-tracker] failed: {_err}")
            _last_emotion[0] = None
    else:
        _last_emotion[0] = None

    # Agent-mode addendum: when the user is in agent mode, extend the
    # system prompt with a PLAN→EXECUTE→CRITIQUE→REPORT directive so the
    # LLM iterates autonomously rather than stopping after one step.
    # Empty string in smart/controlled mode, so this is a no-op for the
    # default path.
    mode_addendum = ""
    try:
        from core.mode_router import system_prompt_addendum as _mode_addendum
        mode_addendum = _mode_addendum()
    except Exception:
        pass

    sys_prompt_now = (
        _system_prompt
        + _tone_system_addendum(tone)
        + route["addendum"]
        + emotion_addendum
        + mode_addendum
        + voice_mood_addendum
    )

    from core.config import model_route
    if model_route("chat") == "local":
        reply = _local_fallback_or(sys_prompt_now, "(the local model is unavailable, sir)")
    elif AI_BACKEND == "claude":
        import anthropic
        try:
            if _llm_client is not None:
                reply = _llm_client.complete(
                    model=CLAUDE_MODEL, max_tokens=500,
                    system=sys_prompt_now, messages=conversation_history,
                    timeout=_ANTHROPIC_TIMEOUT_S,
                )
            else:
                msg = anthropic.Anthropic(timeout=_ANTHROPIC_TIMEOUT_S).messages.create(
                    model=CLAUDE_MODEL, max_tokens=500,
                    system=sys_prompt_now,
                    messages=conversation_history,
                )
                reply = msg.content[0].text
        except anthropic.BadRequestError as _e:
            _s = str(_e).lower()
            if "credit balance" in _s or "too low" in _s or "upgrade or purchase" in _s:
                reply = _local_fallback_or(
                    sys_prompt_now,
                    "I'm afraid my API credits have run out, sir. "
                    "Please top up the balance at console.anthropic.com "
                    "to bring me back online.",
                )
            elif ("usage limit" in _s or "regain access" in _s
                  or "monthly limit" in _s or "spend limit" in _s
                  or "quota" in _s or "exceeded" in _s
                  or "reached your" in _s or "rate limit" in _s):
                reply = _local_fallback_or(
                    sys_prompt_now,
                    "I've reached the monthly API usage limit, sir. "
                    "Access will be restored automatically once the limit resets.",
                )
            else:
                # Unrecognised 4xx — the cloud API is unavailable for this
                # request regardless of the reason, so fall back to the local
                # LLM and only surface the raw error if the fallback fails.
                reply = _local_fallback_or(
                    sys_prompt_now,
                    f"API request rejected (400): {_e}",
                )
        except anthropic.RateLimitError:
            reply = _local_fallback_or(
                sys_prompt_now,
                "I'm being rate-limited by the API, sir — "
                "please give it a moment and try again.",
            )
        except anthropic.APIStatusError as _e:
            reply = _local_fallback_or(
                sys_prompt_now,
                f"API error {_e.status_code}: {_e.message}",
            )
        except Exception as _e:
            reply = _local_fallback_or(
                sys_prompt_now,
                f"Unexpected LLM error: {_e}",
            )

    elif AI_BACKEND == "ollama":
        # Mirror the claude branch's guard: a dead/unreachable Ollama server
        # (or a missing model) must NOT propagate out of _call_llm. The outer
        # main loop only catches KeyboardInterrupt, so an unguarded raise here
        # would crash the whole conversation loop. 2026-05-30 audit.
        try:
            import ollama
            resp = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "system", "content": sys_prompt_now}] + conversation_history,
            )
            reply = resp["message"]["content"]
        except Exception as _e:
            print(f"  [ollama] chat failed: {_e}")
            reply = ("I'm afraid my local model isn't responding, sir — "
                     "Ollama may be down. Please check it and try again.")

    else:
        reply = "AI backend not configured. Check AI_BACKEND in the script."

    conversation_history.append({"role": "assistant", "content": reply})

    # Update phrasebook rotation state — scan the reply for any canonical MCU
    # lines and record them in memory so the next turn's system prompt asks
    # the LLM to rotate to a different phrase in each matched intent bucket.
    try:
        hits = _mcu_phrases.detect_phrases_in_reply(reply)
        if hits:
            with _memory_lock:
                _mem = load_memory()
                _last = _mem.setdefault("last_used_phrase_by_intent", {})
                _last.update(hits)
                save_memory(_mem)
    except Exception as _phr_err:
        print(f"  [phrase-rotation] update failed: {_phr_err}")

    # Keep history bounded — see _trim_conversation_history (trims in pairs so
    # the first message stays 'user', as the Claude API requires).
    _trim_conversation_history()

    return reply


# ──────────────────────────────────────────────────────────────────────────
#  TEXT-TO-SPEECH  (edge-tts, falls back to pyttsx3 if offline)
# ──────────────────────────────────────────────────────────────────────────

# Emotion-aware TTS prosody presets (_TTS_EMOTION_PRESETS), the [intent:xxx]→
# preset map (_INTENT_PRESETS), the user-tone→preset map (_USER_TONE_TTS), the
# text-keyword classifier (detect_tts_emotion + _TTS_EMOTION_KEYWORDS), and the
# priority resolver now live in core/tts.py (consolidated so the edge-tts
# (rate, pitch, gain) triples sit in one module). They're re-exported at the
# top-of-file `from core.tts import (...)` so the references below still resolve
# off this namespace; _resolve_tts_preset() further down is a thin shim that
# feeds the live _last_* state into core.tts.resolve_tts_preset().

# _INTENT_PRESETS (the [intent:xxx]→preset-name map) moved to core/tts.py and
# is re-exported at the top of this file; _parse_intent_tag() below reads it.

# Slot for the most-recent parsed [intent:xxx] tag. Set by _parse_intent_tag()
# inside _speak() right before synthesise() is called, read by
# _resolve_tts_preset(), then cleared. Mirrors the _last_user_tone /
# _last_voice_route pattern so synthesise() can pick it up without changing
# the call signature.
_last_intent_override: list[str | None] = [None]

# Slot for the [wry] marker. When True, _resolve_tts_preset() returns the
# 'wry' preset (overrides intent/emotion/tone for this line) and synthesise()
# inserts a short beat before the punchline. Set in _speak() via
# core.tts.parse_wry_tag(), cleared in the same finally block.
_last_wry: list[bool] = [False]

# Slot for the voice_mood layer (core/voice_mood_selector). Carries a mood
# name (calm_efficient / urgent_clipped / dry_amused / concerned_soft) when
# the caller passed mood= to _speak(), or when [mood:xxx] was parsed off the
# leading tag. Read by _resolve_tts_preset() between the [intent:xxx]
# override and the emotion-tracker fallback so explicit intents still win.
_last_mood: list[str | None] = [None]

# Match a leading [mood:xxx] tag, same shape as [intent:xxx]. Skills that
# return text via the dispatcher (chappie recall, banter quips) can opt
# into a mood without holding a reference to _speak(); the tag is parsed in
# _speak() and forwarded into _last_mood for the synthesise() turn.
_MOOD_TAG_RE = re.compile(r"^\s*\[\s*mood\s*:\s*([a-z_][a-z0-9_]*)\s*\]\s*", re.IGNORECASE)

_VOICE_MOOD_NAMES: frozenset[str] = frozenset({
    "calm_efficient", "urgent_clipped", "dry_amused", "concerned_soft",
})


def _parse_mood_tag(text: str) -> tuple[str | None, str]:
    """If `text` starts with [mood:xxx], return (mood_name_if_known, stripped).
    Unknown mood names are still stripped from the spoken text but resolve
    to None so they fall through to the rest of the priority chain."""
    if not text:
        return None, text
    m = _MOOD_TAG_RE.match(text)
    if not m:
        return None, text
    raw = m.group(1).lower().strip()
    stripped = text[m.end():]
    if raw in _VOICE_MOOD_NAMES:
        return raw, stripped
    return None, stripped

# Match a leading [intent:xxx] tag (case-insensitive, allowing surrounding
# whitespace). Anchored to the start because the LLM is instructed to lead
# replies with it. The pattern is permissive on the intent name so unknown
# values still get stripped from the spoken text rather than read aloud.
_INTENT_TAG_RE = re.compile(r"^\s*\[\s*intent\s*:\s*([a-z_][a-z0-9_]*)\s*\]\s*", re.IGNORECASE)


def _parse_intent_tag(text: str) -> tuple[str | None, str]:
    """If `text` starts with [intent:xxx], return (intent_name, text_with_tag_stripped).
    Otherwise return (None, text). Unknown intents are still stripped from
    the text — we never want a literal '[intent:foo]' to be spoken aloud —
    but the returned intent will be None, falling back to the existing
    tone/keyword routing."""
    if not text:
        return None, text
    m = _INTENT_TAG_RE.match(text)
    if not m:
        return None, text
    raw = m.group(1).lower().strip()
    stripped = text[m.end():]
    if raw in _INTENT_PRESETS:
        return raw, stripped
    return None, stripped

# _USER_TONE_TTS (user-tone→preset map), the text-keyword classifier
# detect_tts_emotion() and its _TTS_EMOTION_KEYWORDS table all moved to
# core/tts.py and are re-exported at the top of this file.


def _resolve_tts_preset(text: str, user_tone: str | None) -> tuple[str, dict[str, object]]:
    """Thin state-supplying shim around core.tts.resolve_tts_preset().

    The pure preset-priority logic lives in core/tts.py; this reads the live
    runtime singletons it needs (_last_wry / _last_intent_override / _last_mood
    / _last_user_text / _last_emotion) plus the recent vocal RMS and forwards
    them as arguments. Kept at this 2-arg (text, user_tone) signature so the
    synthesise() call site AND the night_owl_mode monkeypatch (which wraps
    bobert_companion._resolve_tts_preset) keep working unchanged.
    """
    if _tts_layer is None:
        # core.tts failed to import — flat neutral prosody keeps JARVIS talking.
        return "neutral", {"rate": "+0%", "pitch": "+0Hz", "gain": 1.0}

    try:
        from core.audio_processor import recent_peak_rms as _recent_peak_rms
        peak_rms = float(_recent_peak_rms())
    except Exception:
        peak_rms = 0.0

    er = _last_emotion[0]
    emotion_preset = getattr(er, "tts_preset", None) if er is not None else None

    return _tts_layer.resolve_tts_preset(
        text,
        user_tone,
        wry=_last_wry[0],
        intent_override=_last_intent_override[0],
        mood=_last_mood[0],
        user_text=_last_user_text[0],
        peak_rms=peak_rms,
        emotion_preset=emotion_preset,
    )


async def _tts_bytes(text: str, rate: str = "+0%", pitch: str = "+0Hz") -> bytes:
    import edge_tts
    buf = io.BytesIO()
    async for chunk in edge_tts.Communicate(text, TTS_VOICE, rate=rate, pitch=pitch).stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()


# Persistent asyncio loop in a background thread so each TTS call doesn't
# spin up + tear down its own loop.
_tts_loop: asyncio.AbstractEventLoop | None = None
_tts_loop_thread: "threading.Thread | None" = None
_TTS_LOOP_LOCK = threading.Lock()


def _ensure_tts_loop():
    global _tts_loop, _tts_loop_thread
    # Lock the entire check-and-assign so two threads racing through here
    # cannot both observe None and both create a new loop (which would
    # leak the first loop's selector + aiohttp connectors).
    with _TTS_LOOP_LOCK:
        # Recreate unless the loop is running AND its background thread is
        # still alive. is_running() ALONE misses the case where the loop
        # thread died while the loop object still reports running — then every
        # run_coroutine_threadsafe() hangs at its 30s timeout forever and TTS
        # goes mute with no recovery. Track + check the thread. 2026-05-30 audit.
        if (_tts_loop is not None and _tts_loop.is_running()
                and _tts_loop_thread is not None and _tts_loop_thread.is_alive()):
            return
        # Best-effort: stop a half-dead old loop before replacing it so we
        # don't leak its selector/connectors.
        if _tts_loop is not None:
            try:
                _tts_loop.call_soon_threadsafe(_tts_loop.stop)
            except Exception:
                pass
        _tts_loop = asyncio.new_event_loop()
        _tts_loop_thread = threading.Thread(target=_tts_loop.run_forever, daemon=True)
        _tts_loop_thread.start()


def _render_edge_tts(text: str, rate: str, pitch: str) -> tuple[np.ndarray, int]:
    """Render `text` through edge-tts and decode to a mono float32 buffer.
    Separated from synthesise() so the wry path can call it twice (setup,
    punchline) with a silence gap spliced between.

    Transient Bing WebSocket errors (503, handshake, network blips) get up
    to 3 attempts with 0.5/1/2-second backoff before propagating. Non-
    transient failures (ImportError, bad text) raise immediately.

    Repeated short phrases are served from the core.tts render cache, skipping
    the network round-trip entirely. The cache key is the PRE-gain (voice,
    rate, pitch, text) tuple; synthesise() re-applies per-mood gain after."""
    if _tts_layer is not None:
        try:
            _cached = _tts_layer.tts_cache_get(text, TTS_VOICE, rate, pitch)
        except Exception:
            _cached = None
        if _cached is not None:
            return _cached
    _ensure_tts_loop()
    for attempt in range(3):
        try:
            fut = asyncio.run_coroutine_threadsafe(_tts_bytes(text, rate=rate, pitch=pitch), _tts_loop)
            raw = fut.result(timeout=30)
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if _tts_layer is not None:
                try:
                    _tts_layer.tts_cache_put(text, TTS_VOICE, rate, pitch, audio, sr)
                except Exception:
                    pass
            return audio, sr
        except Exception as e:
            err_str = str(e).lower()
            transient = any(s in err_str for s in (
                "503", "handshake", "timeout", "timed out", "connection",
                "websocket", "wsserver", "temporarily", "reset",
            ))
            if isinstance(e, ImportError) or not transient or attempt == 2:
                raise
            backoff = 0.5 * (2 ** attempt)  # 0.5s, 1.0s, 2.0s
            print(f"  [tts] edge-tts attempt {attempt + 1}/3 failed ({type(e).__name__}: {e}); retrying in {backoff:.1f}s")
            time.sleep(backoff)
    raise RuntimeError("unreachable")  # pragma: no cover  (loop always returns or raises)


def _silent_clip(sr: int = 24000, ms: int = 80) -> tuple[np.ndarray, int]:
    """Return a tiny silent buffer so callers that expect (audio, sr) can
    keep going when every TTS backend has failed. 24 kHz mirrors edge-tts'
    native output so the downstream sample-rate path doesn't change."""
    samples = max(1, int(sr * ms / 1000))
    return np.zeros(samples, dtype=np.float32), sr


def _render_xtts_or_raise(text: str, rate: str, pitch: str) -> tuple[np.ndarray, int]:
    """Render `text` via skills/custom_voice.py (Coqui XTTS-v2). Raises on
    any failure so the caller can fall back to edge-tts. Looked up via
    sys.modules so the skill is optional — if it isn't loaded yet we raise
    the same way as a normal render failure and the fallback path engages."""
    mod = sys.modules.get("skill_custom_voice")
    if mod is None:
        raise RuntimeError("custom_voice skill not loaded")
    return mod.render(text, rate, pitch)


def synthesise(text: str) -> tuple[np.ndarray, int]:
    # Voice-mood router output wins over raw detect_tone() output here:
    # the router fuses tone with time-of-day, so its label is the one
    # that should drive prosody. Fall back to the fine-grained text tone
    # (rushed/tired/playful/frustrated) when the router said 'casual'.
    try:
        route = _last_voice_route[0]
        mood = route["mood"] if route else "casual"
        user_tone = mood if mood != "casual" else _last_user_tone[0]
        chosen, preset = _resolve_tts_preset(text, user_tone)
        rate  = str(preset.get("rate",  "+0%"))
        pitch = str(preset.get("pitch", "+0Hz"))
        gain  = float(preset.get("gain", 1.0))  # type: ignore[arg-type]
        if chosen != "neutral" or gain != 1.0:
            tone_tag = f" tone={user_tone}" if user_tone else ""
            mood_tag = f" mood={_last_mood[0]}" if _last_mood[0] else ""
            print(f"  [tts] preset={chosen}{tone_tag}{mood_tag} rate={rate} pitch={pitch} gain={gain:.2f}")

        # Voice-clone backend takes precedence when the user has opted in
        # via TTS_BACKEND='xtts'. Any failure (deps missing, sample WAV
        # unreadable, render crash) falls straight through to edge-tts so
        # an over-eager toggle never silences JARVIS.
        backend = (globals().get("TTS_BACKEND", "edge") or "edge").lower()
        if backend == "xtts":
            try:
                audio, sr = _render_xtts_or_raise(text, rate, pitch)
                if gain != 1.0:
                    audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
                return audio, sr
            except Exception as e:
                print(f"  [tts] XTTS render failed ({e}); falling back to edge-tts")

        if backend == "pyttsx3":
            try:
                audio, sr = _pyttsx3_tts(text)
                if gain != 1.0:
                    audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
                return audio, sr
            except Exception as e:
                print(f"  [tts] pyttsx3 render failed ({e}); falling back to edge-tts")

        try:
            # Wry deliveries get a brief beat spliced in before the final clause
            # so the punchline actually lands. core.tts.split_for_wry_pause()
            # returns (text, None) when no good split exists — we fall through
            # to the single-pass render in that case.
            wry_split = None
            if chosen == "wry" and _tts_layer is not None:
                try:
                    head, tail = _tts_layer.split_for_wry_pause(text)
                    if tail:
                        wry_split = (head, tail, int(_tts_layer.WRY_PAUSE_MS))
                except Exception:
                    wry_split = None

            if wry_split is not None:
                head, tail, pause_ms = wry_split
                head_audio, sr = _render_edge_tts(head, rate, pitch)
                tail_audio, _  = _render_edge_tts(tail, rate, pitch)
                pause_samples  = max(0, int(sr * pause_ms / 1000))
                silence        = np.zeros(pause_samples, dtype=np.float32)
                audio          = np.concatenate([head_audio, silence, tail_audio]).astype(np.float32)
            else:
                audio, sr = _render_edge_tts(text, rate, pitch)

            if gain != 1.0:
                audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
            return audio, sr
        except Exception as e:
            print(f"  edge-tts failed ({e}), using pyttsx3…")
            audio, sr = _pyttsx3_tts(text)
            if gain != 1.0:
                audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
            return audio, sr
    except Exception as e:
        # Both backends are dead (e.g. edge-tts 503 + pyttsx3 not installed).
        # Returning silence keeps main() alive — losing one line of speech
        # is strictly better than crashing the whole assistant.
        print(f"  [tts] total failure ({type(e).__name__}: {e}); returning silence so JARVIS stays online")
        return _silent_clip()


def _try_sapi5_then_silence(text: str, rate: str = "+0%", pitch: str = "+0Hz") -> tuple[np.ndarray, int]:
    """Final fallback chain: SAPI5 via PowerShell, then silence. Called from
    _pyttsx3_tts when pyttsx3 itself fails so the edge-tts → pyttsx3 →
    SAPI5 → silence ladder remains transparent to synthesise()."""
    try:
        from tts.render import render_sapi5
    except ImportError as e:
        print(f"  [tts] SAPI5 module unavailable ({e}); returning silence")
        return _silent_clip()
    try:
        audio, sr = render_sapi5(text, rate=rate, pitch=pitch)
        print(f"  [tts] SAPI5 fallback succeeded ({len(audio)} samples @ {sr} Hz)")
        return audio, sr
    except Exception as e:
        print(f"  [tts] SAPI5 fallback failed ({type(e).__name__}: {e}); returning silence")
        return _silent_clip()


def _pyttsx3_tts(text: str) -> tuple[np.ndarray, int]:
    try:
        import pyttsx3
    except ImportError as e:
        # Fallback engine not installed; surface to console but do NOT crash
        # the caller. Try SAPI5 next so a missing pyttsx3 doesn't silence us.
        print(f"  [tts] pyttsx3 unavailable ({e}); trying SAPI5")
        return _try_sapi5_then_silence(text)
    try:
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            engine.save_to_file(text, path)
            engine.runAndWait()
            engine.stop()   # release the file handle before we try to delete it
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio, sr
        finally:
            # Always remove the temp .wav — if save_to_file/runAndWait/sf.read
            # raised before we got here the file would otherwise leak.
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass   # Windows can hold the file briefly; not worth crashing over
    except Exception as e:
        # pyttsx3 itself blew up (init failed, save_to_file failed, etc.).
        # Try SAPI5 before giving up — this is the cascade the user kept
        # seeing as 'edge-tts failed → pyttsx3 render failed → silence'.
        print(f"  [tts] pyttsx3 render failed ({type(e).__name__}: {e}); trying SAPI5")
        return _try_sapi5_then_silence(text)


# ──────────────────────────────────────────────────────────────────────────
#  PLAYBACK + LIP-SYNC
# ──────────────────────────────────────────────────────────────────────────

def is_using_headset() -> bool:
    """True if the current output device name looks like a headset.
    Used to decide whether barge-in is safe (no mic feedback)."""
    try:
        idx = get_output_device()
        if idx is None:
            return False
        name = sd.query_devices(idx)["name"].lower()
        return any(kw in name for kw in HEADSET_NAME_HINTS)
    except Exception:
        return False


# Barge-in shared state
_barge_in_interrupted = False


def _start_barge_in_listener():
    """Open a mic InputStream that flags sustained speech during playback.
    The callback ONLY sets `_barge_in_interrupted = True`; calling sd.stop()
    from inside a PortAudio callback frees the stream while the callback
    that triggered the free is still on the stack (use-after-free → 0xc0000374).
    The actual sd.stop() lives in the watch thread inside play_with_lipsync.
    Returns the stream so we can close it when playback finishes, or None
    on failure."""
    global _barge_in_interrupted
    _barge_in_interrupted = False
    sustained = [0]

    def cb(indata, frames, time_info, status):
        global _barge_in_interrupted
        if _barge_in_interrupted:
            return
        data = indata[:, 0] if indata.ndim > 1 else indata
        rms = float(np.sqrt(np.mean(data ** 2)))
        if rms > BARGE_IN_THRESHOLD:
            sustained[0] += 1
            if sustained[0] >= BARGE_IN_SUSTAIN_CHUNKS:
                _barge_in_interrupted = True
        else:
            sustained[0] = 0

    if _mic_input_disabled():
        return None
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=1024, device=get_input_device(), callback=cb,
        )
        stream.start()
        return stream
    except Exception as e:
        print(f"  [barge-in] could not start listener: {e}")
        return None


# ── Audio ducking (pycaw) ──────────────────────────────────────────────────
# Lazy WASAPI-session ducker. duck() runs the fade-down on a background
# thread so TTS playback can start immediately; restore() cancels any
# in-flight fade-down and fades back up synchronously so the next utterance
# sees a fully-restored mixer. No-op when pycaw isn't installed, when not on
# Windows, or when no matching session is currently playing.
class _AudioDucker:
    _AVAILABLE: bool | None = None
    _SELF_PID = os.getpid()

    def __init__(self):
        self._lock = threading.Lock()
        self._saved: list[tuple[object, float]] = []
        self._fade_cancel = threading.Event()
        self._fade_done = threading.Event()
        self._fade_done.set()
        self._worker_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._work_queue: queue.Queue = queue.Queue()

    @classmethod
    def _check_available(cls) -> bool:
        if cls._AVAILABLE is not None:
            return cls._AVAILABLE
        if sys.platform != "win32":
            cls._AVAILABLE = False
            return False
        try:
            import pycaw.pycaw  # noqa: F401
            import comtypes     # noqa: F401
            cls._AVAILABLE = True
        except Exception:
            cls._AVAILABLE = False
        return cls._AVAILABLE

    def _ensure_worker(self) -> None:
        # A single long-lived worker thread initializes COM once at startup
        # and reuses the same MTA apartment for every fade. Spawning a fresh
        # thread per TTS playback used to leak an apartment each call (the
        # CoInitialize had no paired CoUninitialize).
        with self._worker_lock:
            t = self._worker_thread
            if t is not None and t.is_alive():
                return
            new_t = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="AudioDuckerFade",
            )
            new_t.start()
            self._worker_thread = new_t

    def _worker_loop(self) -> None:
        com_inited = False
        try:
            import comtypes
            comtypes.CoInitialize()
            com_inited = True
        except Exception:
            pass
        try:
            while True:
                job = self._work_queue.get()
                if job is None:
                    return
                plans, target_level, cancellable, done_event = job
                try:
                    self._fade_run(plans, target_level, cancellable)
                except Exception as e:
                    print(f"  [audio-duck] fade failed: {e}")
                finally:
                    if done_event is not None:
                        done_event.set()
        finally:
            if com_inited:
                try:
                    import comtypes
                    comtypes.CoUninitialize()
                except Exception:
                    pass

    def _enumerate_targets(self) -> list[tuple[object, float]]:
        from pycaw.pycaw import AudioUtilities
        com_inited = False
        try:
            import comtypes
            comtypes.CoInitialize()
            com_inited = True
        except Exception:
            pass
        try:
            try:
                sessions = AudioUtilities.GetAllSessions()
            except Exception as e:
                print(f"  [audio-duck] enumerate failed: {e}")
                return []
            targets = [t.lower() for t in AUDIO_DUCKING_TARGETS]
            matched: list[tuple[object, float]] = []
            for sess in sessions:
                try:
                    proc = sess.Process
                    if proc is None or proc.pid == self._SELF_PID:
                        continue
                    name = (proc.name() or "").lower()
                    if not any(t in name for t in targets):
                        continue
                    vol_iface = sess.SimpleAudioVolume
                    if vol_iface is None:
                        continue
                    cur = float(vol_iface.GetMasterVolume())
                    # Skip sessions already at/below the duck level so we don't
                    # later 'restore' them to a louder level than the user set.
                    if cur <= AUDIO_DUCKING_LEVEL + 0.01:
                        continue
                    matched.append((vol_iface, cur))
                except Exception:
                    continue
            return matched
        finally:
            if com_inited:
                try:
                    import comtypes
                    comtypes.CoUninitialize()
                except Exception:
                    pass

    def _fade_run(self, plans, target_level, cancellable: bool) -> None:
        # Called only from the worker thread, which has COM already initialized
        # for its lifetime. Do not CoInitialize here — that's what was leaking.
        if not plans:
            return
        steps = 10
        step_dur = (AUDIO_DUCKING_FADE_MS / 1000.0) / steps
        started = []
        for iface, original in plans:
            try:
                start = float(iface.GetMasterVolume())
            except Exception:
                continue
            end = original if target_level is None else target_level
            started.append((iface, start, end))
        for i in range(1, steps + 1):
            if cancellable and self._fade_cancel.is_set():
                return
            frac = i / steps
            for iface, start, end in started:
                lvl = max(0.0, min(1.0, start + (end - start) * frac))
                try:
                    iface.SetMasterVolume(lvl, None)
                except Exception:
                    pass
            time.sleep(step_dur)

    def duck(self) -> None:
        if not AUDIO_DUCKING_ENABLED or not self._check_available():
            return
        with self._lock:
            if self._saved:
                return
            try:
                matched = self._enumerate_targets()
            except Exception as e:
                print(f"  [audio-duck] duck failed: {e}")
                return
            if not matched:
                return
            self._saved = matched
            self._fade_cancel.clear()
            self._fade_done.clear()
            self._ensure_worker()
            self._work_queue.put(
                (list(matched), AUDIO_DUCKING_LEVEL, True, self._fade_done)
            )

    def restore(self) -> None:
        with self._lock:
            if not self._saved:
                return
            saved = list(self._saved)
            self._fade_cancel.set()
            self._saved = []
            fade_done = self._fade_done
        # Wait for any in-flight duck fade to observe the cancel and exit.
        try:
            fade_done.wait(timeout=AUDIO_DUCKING_FADE_MS / 1000.0 + 0.3)
        except Exception:
            pass
        # Enqueue the fade-up on the worker so it runs in the COM-initialized
        # apartment, then wait for completion synchronously.
        self._ensure_worker()
        restore_done = threading.Event()
        self._work_queue.put((saved, None, False, restore_done))
        try:
            restore_done.wait(timeout=AUDIO_DUCKING_FADE_MS / 1000.0 + 1.0)
        except Exception:
            pass


_audio_ducker = _AudioDucker()


def play_with_lipsync(audio: np.ndarray, sr: int):
    """Play TTS audio through speakers. If a robot is connected, also stream
    mouth-open values at ~30 fps so it lip-syncs. If barge-in is enabled and
    the user is on a headset, listen for interruptions during playback.

    Always publishes a per-chunk TTS amplitude to the HUD state (separate
    field from mic_level) so the arc-reactor HUD can brighten the ring
    during speech — see hud/jarvis_hud.py (arc_reactor_hud upgrade)."""
    # Mark TTS playback live for the whole function so _refresh_devices skips
    # its destructive sd._terminate()/_initialize() reinit while the barge-in
    # InputStream / sd.stop() are in play. Mirrors the _record_speech_active
    # guard — a device refresh firing here otherwise frees PortAudio state
    # under the live barge-in stream (0xc0000374). Reset in the finally below.
    _tts_playback_active[0] = True
    out_dev = get_output_device()

    # Start barge-in listener if conditions met
    barge_stream = None
    if BARGE_IN_ENABLED and is_using_headset():
        barge_stream = _start_barge_in_listener()

    # Barge-in watch: the InputStream callback flags `_barge_in_interrupted`
    # but cannot call sd.stop() itself (that triggers a PortAudio use-after-
    # free). A daemon thread polls the flag and stops playback from a safe
    # context. Started only when the listener is actually live.
    barge_watch_stop = threading.Event()
    barge_watch_thread: threading.Thread | None = None
    if barge_stream is not None:
        def _barge_watch():
            while not barge_watch_stop.wait(0.02):
                if _barge_in_interrupted:
                    print("  [barge-in] user interrupted")
                    try:
                        sd.stop()
                    except Exception:
                        pass
                    return
        barge_watch_thread = threading.Thread(target=_barge_watch, daemon=True)
        barge_watch_thread.start()

    # Duck Chrome / Spotify / Apple Music / Edge so JARVIS sits cleanly
    # over whatever's already playing. Fades down in the background so
    # playback starts immediately; restored in the finally block.
    _audio_ducker.duck()

    CHUNK_SECS = 0.033
    chunk_n    = int(sr * CHUNK_SECS)
    amp_stop   = threading.Event()

    def _amp_pump():
        """Stream RMS amplitude to the HUD state file at ~30 fps. Decoupled
        from the robot lip-sync loop so it runs even when ROBOT_ENABLED=False."""
        pos = 0
        try:
            while pos < len(audio) and not amp_stop.is_set():
                chunk = audio[pos : pos + chunk_n]
                if len(chunk):
                    amp = float(np.clip(
                        np.sqrt(np.mean(chunk.astype(np.float32) ** 2)),
                        0.0, 1.0,
                    ))
                    _write_hud_state(tts_amplitude=amp)
                pos += chunk_n
                time.sleep(CHUNK_SECS)
        finally:
            _write_hud_state(tts_amplitude=0.0)

    amp_thread = threading.Thread(target=_amp_pump, daemon=True)
    amp_thread.start()

    # Hand the outgoing waveform to the noise-cancel-1 processor before
    # playback starts so its AEC layer has a far-end reference for the
    # duration of speech.
    _feed_playback_reference(audio, sr)

    # MUTE_TTS support — synthesise() already ran, so the prosody pipeline
    # has been exercised. Skip the actual sd.play() call and just sleep for
    # the audio duration so the HUD amplitude pump + lip-sync helpers stay
    # in sync with what would have been audible. core/tts.is_muted() is
    # checked fresh per call so an operator can toggle MUTE_TTS mid-session.
    _muted = False
    try:
        if _tts_layer is not None and _tts_layer.is_muted():
            _muted = True
    except Exception:
        _muted = False
    audio_secs = (len(audio) / float(sr)) if sr else 0.0

    try:
        if _muted:
            # Replace the device write with a plain sleep — keeps the
            # surrounding amp_thread + barge-in watchdog logic identical.
            time.sleep(min(audio_secs, 30.0))
        elif not ROBOT_ENABLED:
            # 2026-05-29 silent-crash fix: sd.wait() blocks until the stream
            # closes natively, and sounddevice's close() at C-level SIGSEGVs
            # under bursty boot speech (faulthandler trace caught it across
            # 7+ PIDs today). Wrap wait+close in a thread with a hard timeout
            # so a hung native close can't take down the whole process.
            sd.play(audio, sr, device=out_dev)
            _done_evt = threading.Event()
            def _safe_wait():
                try:
                    sd.wait()
                except Exception:
                    logging.exception("[audio] sd.wait raised — proceeding")
                finally:
                    _done_evt.set()
            _t = threading.Thread(target=_safe_wait, daemon=True)
            _t.start()
            # Bound the wait at audio length + 2s grace; if it stalls past
            # that, force-stop the stream and move on. Process stays alive.
            _done_evt.wait(timeout=max(audio_secs + 2.0, 5.0))
            if not _done_evt.is_set():
                try: sd.stop()  # pragma: no cover - timeout-recovery: only runs if a live sd.wait() hangs past the grace window
                except Exception: pass  # pragma: no cover - timeout-recovery: only runs if a live sd.wait() hangs past the grace window
        else:
            pos = 0

            def _sync():
                nonlocal pos
                while pos < len(audio):
                    try:
                        chunk = audio[pos : pos + chunk_n]
                        if len(chunk):
                            mouth = float(np.clip(np.sqrt(np.mean(chunk ** 2)) * MOUTH_SCALE, 0.0, 1.0))
                            send(mouth=mouth)
                        pos += chunk_n
                        time.sleep(CHUNK_SECS)
                    except Exception:
                        logging.exception("[_sync] lip-sync iteration failed")
                        pos += chunk_n
                        time.sleep(CHUNK_SECS)
                try:
                    send(mouth=0.0)
                except Exception:
                    logging.exception("[_sync] failed to send mouth=0.0 on exit")

            t = threading.Thread(target=_sync, daemon=True)
            t.start()
            sd.play(audio, sr, device=out_dev)
            # 2026-05-29 silent-crash fix: same hardening as the no-robot branch
            # above — sd.wait()/native close has SIGSEGV'd during boot speech.
            # Bound the wait so a hung close can't kill the process.
            _done_evt = threading.Event()
            def _safe_wait_robot():
                try:
                    sd.wait()
                except Exception:
                    logging.exception("[audio] sd.wait raised (robot) — proceeding")
                finally:
                    _done_evt.set()
            _t = threading.Thread(target=_safe_wait_robot, daemon=True)
            _t.start()
            _done_evt.wait(timeout=max(audio_secs + 2.0, 5.0))
            if not _done_evt.is_set():
                try: sd.stop()  # pragma: no cover - timeout-recovery: only runs if a live sd.wait() hangs past the grace window (robot arm)
                except Exception: pass  # pragma: no cover - timeout-recovery: only runs if a live sd.wait() hangs past the grace window (robot arm)
            t.join(timeout=0.5)
    finally:
        # Clear the TTS-playback guard first so a queued device refresh can
        # resume its normal reinit path once the barge-in stream is closed
        # below. Set unconditionally to mirror the start-of-function flag.
        _tts_playback_active[0] = False
        amp_stop.set()
        try:
            amp_thread.join(timeout=0.2)
        except Exception:
            pass
        # Stop the barge-in watch thread before tearing down the InputStream
        # so it can't observe the flag after `barge_stream` is closed.
        barge_watch_stop.set()
        if barge_watch_thread is not None:
            try:
                barge_watch_thread.join(timeout=0.2)
            except Exception:
                pass
        # Ensure HUD doesn't show stale amplitude after playback
        _write_hud_state(tts_amplitude=0.0)
        if barge_stream is not None:
            _safe_close_stream(barge_stream)
        # Fade Chrome/Spotify/Apple Music/Edge back to their original volumes
        try:
            _audio_ducker.restore()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
#  SCREEN VISION  — Bobert sees what's on screen via Claude's vision API
# ──────────────────────────────────────────────────────────────────────────

import base64

# Spoken refusal returned by the vision/screenshot actions when the focused
# window matches SCREENSHOT_PRIVACY_BLOCKLIST — kept as a single constant so
# every entry point says the same thing.
SCREENSHOT_PRIVACY_REFUSAL = (
    "I'm not capturing the screen — a private window is in focus, sir."
)


def screenshot_privacy_block_reason() -> str | None:
    """Return the SCREENSHOT_PRIVACY_BLOCKLIST entry matching the focused
    window's title (case-insensitive substring), or None when capture is
    allowed.

    Empty/missing blocklist → always None (no behaviour change). Any failure
    reading the focused window is treated as 'allowed' so a transient Win32
    hiccup never silently blinds vision — the blocklist is an opt-in privacy
    guard, not a hard gate."""
    try:
        from core import config as _cfg
        blocklist = getattr(_cfg, "SCREENSHOT_PRIVACY_BLOCKLIST", ()) or ()
    except Exception:
        return None
    if not blocklist:
        return None
    try:
        _, title, _ = _read_focused_window()
    except Exception:
        return None
    if not title:
        return None
    low = title.lower()
    for entry in blocklist:
        try:
            needle = str(entry).strip().lower()
        except Exception:
            continue
        if needle and needle in low:
            return str(entry)
    return None


def take_screenshot(monitor: str | None = None, max_dim: int = 1568) -> bytes | None:
    """Return PNG bytes of the requested monitor (defaults to primary),
    downscaled so the longest edge is at most max_dim. Claude vision
    requires the image to be under ~5MB and ≤ 1568px on the long side.

    Uses mss (preferred — DPI-safe, handles all multi-monitor configs on
    Windows without crashing) with a PIL.ImageGrab fallback.

    Hard privacy gate: if the focused window matches
    SCREENSHOT_PRIVACY_BLOCKLIST, returns None without capturing — so every
    caller (vision, saved screenshots, the loopback grabbers) is covered even
    if a higher-level entry point forgot to check."""
    blocked = screenshot_privacy_block_reason()
    if blocked:
        print(f"  [vision] screenshot refused — private window in focus "
              f"(matched {blocked!r})")
        return None
    try:
        from PIL import Image
    except ImportError:
        print("  [vision] Pillow not installed — pip install pillow")
        return None

    # ── mss path (preferred) ──────────────────────────────────────────────
    try:
        import mss
        _MSSClass = getattr(mss, "MSS", mss.mss)   # MSS in ≥9, mss in older
        with _MSSClass() as sct:
            if monitor and monitor in MONITORS:
                x, y, w, h = MONITORS[monitor]
                region = {"left": x, "top": y, "width": w, "height": h}
            else:
                # sct.monitors[0] is the virtual screen spanning all displays
                region = sct.monitors[0]
            raw = sct.grab(region)
            # mss returns BGRA; convert to PIL RGB
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    except Exception as e_mss:
        # ── PIL fallback ──────────────────────────────────────────────────
        try:
            from PIL import ImageGrab
            if monitor and monitor in MONITORS:
                x, y, w, h = MONITORS[monitor]
                img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
            else:
                # monitor=None must capture the WHOLE virtual desktop to match
                # the mss path (and what find_click_target's translate assumes).
                # Bare ImageGrab.grab() returns ONLY the primary monitor, so a
                # target on a secondary/negative-origin display would be cropped
                # out and the coords mistranslated. Grab the full virtual bounds.
                vx, vy, vw, vh = _virtual_screen_bounds()
                img = ImageGrab.grab(bbox=(vx, vy, vx + vw, vy + vh),
                                     all_screens=True)
        except Exception as e_pil:
            print(f"  [vision] screenshot failed (mss: {e_mss}, pil: {e_pil})")
            return None

    try:
        # Downscale so the longest edge ≤ max_dim — Claude rejects larger images
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception as e:
        print(f"  [vision] screenshot encode failed: {e}")
        return None


def ask_vision(question: str, png_bytes: bytes | None = None) -> str:
    """Send a screenshot to Claude with a question, return its answer.

    If the Claude vision call fails (rate-limit, credit exhausted, network,
    5xx) or the Claude backend is disabled outright, falls through to a
    local VLM via Ollama (controlled by LOCAL_VISION_FALLBACK /
    LOCAL_VISION_MODEL). Local-vision answers are prefixed with
    `[local-vision] ` so the user knows the offline eye answered."""
    if not SCREEN_VISION_ENABLED:
        return "(screen vision is disabled — set SCREEN_VISION_ENABLED = True)"
    # Privacy gate first — when a blocklisted window is focused, refuse with a
    # spoken line instead of the generic capture-failure text. Skip the check
    # when the caller already supplied png_bytes (it captured before whatever
    # is now focused; the gate ran at that capture).
    if png_bytes is None:
        if screenshot_privacy_block_reason():
            return SCREENSHOT_PRIVACY_REFUSAL
        png_bytes = take_screenshot()
    if png_bytes is None:
        return "(could not capture screen)"

    # Per-function routing: force the local VLM when vision is routed "local".
    from core.config import model_route
    if model_route("vision") == "local":
        local = _call_local_vision(question, [png_bytes])
        return f"[local-vision] {local}" if local else "(local vision unavailable — Ollama not reachable)"

    # Cloud-disabled path: jump straight to the local VLM.
    if AI_BACKEND != "claude":
        local = _call_local_vision(question, [png_bytes])
        if local:
            return f"[local-vision] {local}"
        return "(screen vision requires Claude backend or a local VLM via Ollama)"

    try:
        import anthropic
        b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
        msg = anthropic.Anthropic(timeout=_ANTHROPIC_TIMEOUT_S).messages.create(
            model=SCREEN_VISION_MODEL, max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": b64
                    }},
                    {"type": "text", "text": question},
                ],
            }],
        )
        return msg.content[0].text.strip()
    except anthropic.APIStatusError as e:
        # 4xx/5xx from Claude (rate limit, credit exhausted, server error,
        # auth) — exactly the case the local VLM fallback was built for.
        print(f"  [vision] Claude API {e.status_code} → local VLM fallback")
        local = _call_local_vision(question, [png_bytes])
        if local:
            return f"[local-vision] {local}"
        return f"(vision failed: HTTP {e.status_code})"
    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
        print(f"  [vision] Claude API unreachable ({type(e).__name__}) → local VLM fallback")
        local = _call_local_vision(question, [png_bytes])
        if local:
            return f"[local-vision] {local}"
        return f"(vision failed: {e})"
    except Exception as e:
        # Catch-all: an empty content list → IndexError on content[0], or
        # `import anthropic` failing → ImportError. Neither is an anthropic.*
        # subclass, so without this they'd escape uncaught and crash the
        # vision action — bypassing the local-VLM fallback this whole
        # function is built around. Route to local like the API-error
        # paths do. 2026-05-30 audit.
        print(f"  [vision] Claude vision failed "
              f"({type(e).__name__}: {e}) → local VLM fallback")
        local = _call_local_vision(question, [png_bytes])
        if local:
            return f"[local-vision] {local}"
        return f"(vision failed: {type(e).__name__})"


def take_all_monitor_screenshots(max_dim: int = 1024) -> dict[str, bytes]:
    """Capture every monitor in MONITORS and return {name: png_bytes}.
    Monitors that fail to capture are skipped silently. Each image is
    downscaled to max_dim on its longest edge — smaller than the
    single-monitor default (1568) because we're going to send 4 of them at
    once and Claude has total-payload limits to respect."""
    out: dict[str, bytes] = {}
    for name in MONITORS:
        png = take_screenshot(monitor=name, max_dim=max_dim)
        if png is not None:
            out[name] = png
    return out


def ask_vision_multi(question: str, images: dict[str, bytes]) -> str:
    """Send several labelled screenshots to Claude in a single message and
    return its answer. Each image is preceded by a text block naming the
    monitor it came from, so the model can refer to them in its answer.
    Used by see_screen when no specific monitor is requested — gives JARVIS
    full situational awareness across all 4 displays in one call.

    Falls back to a local VLM via Ollama on Claude failure (or when the
    Claude backend is disabled). The local prompt embeds each monitor's
    name inline since Ollama's chat API doesn't support per-image labels
    the way Claude's interleaved content blocks do."""
    if not SCREEN_VISION_ENABLED:
        return "(screen vision is disabled — set SCREEN_VISION_ENABLED = True)"
    if not images:
        return "(no screens to look at)"

    def _local_multi_fallback() -> str | None:
        # Build an interleaved-style prompt that names each monitor in the
        # text, then pass all PNGs in order. VLMs like qwen2.5vl handle
        # multi-image with positional ordering preserved.
        names = list(images.keys())
        pngs  = [images[n] for n in names]
        labels = "\n".join(
            f"Image #{i+1} = {n.upper()} monitor" for i, n in enumerate(names)
        )
        prompt = (
            f"You are looking at {len(images)} monitors at once. They are "
            f"provided in this order:\n{labels}\n\n"
            f"When answering, name which monitor(s) the relevant content is on. "
            f"If the question doesn't apply to a given monitor, you can skip it.\n\n"
            f"Question: {question}"
        )
        text = _call_local_vision(prompt, pngs, max_tokens=900)
        return f"[local-vision] {text}" if text else None

    from core.config import model_route
    if model_route("vision") == "local":
        local = _local_multi_fallback()
        return local if local else "(local vision unavailable — Ollama not reachable)"

    if AI_BACKEND != "claude":
        local = _local_multi_fallback()
        if local:
            return local
        return "(screen vision requires Claude backend or a local VLM via Ollama)"

    try:
        import anthropic

        intro = (
            f"You are looking at {len(images)} monitors at once. Each image "
            f"below is labelled with which monitor it is. When answering, "
            f"mention which monitor(s) the relevant content is on. If "
            f"something is on only one monitor, name that monitor. If the "
            f"question doesn't apply to a given monitor, you can skip it."
        )

        content: list = [{"type": "text", "text": intro}]
        for name, png in images.items():
            b64 = base64.standard_b64encode(png).decode("utf-8")
            content.append({"type": "text", "text": f"--- {name.upper()} monitor ---"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })
        content.append({"type": "text", "text": question})

        msg = anthropic.Anthropic(timeout=_ANTHROPIC_TIMEOUT_S).messages.create(
            model=SCREEN_VISION_MODEL, max_tokens=800,
            messages=[{"role": "user", "content": content}],
        )
        return msg.content[0].text.strip()
    except anthropic.APIStatusError as e:
        # 4xx/5xx from Claude (rate limit, credit exhausted, server error,
        # auth) — exactly the case the local VLM fallback was built for.
        print(f"  [vision] Claude API {e.status_code} → local VLM fallback")
        local = _local_multi_fallback()
        if local:
            return local
        return f"(vision failed: HTTP {e.status_code})"
    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
        print(f"  [vision] Claude API unreachable ({type(e).__name__}) → local VLM fallback")
        local = _local_multi_fallback()
        if local:
            return local
        return f"(vision failed: {e})"
    except Exception as e:
        # Catch-all (empty content → IndexError, import failure, unexpected
        # SDK error). Without this they'd crash the multi-monitor vision
        # action instead of degrading to the local VLM. 2026-05-30 audit.
        print(f"  [vision] Claude multi-vision failed "
              f"({type(e).__name__}: {e}) → local VLM fallback")
        local = _local_multi_fallback()
        if local:
            return local
        return f"(vision failed: {type(e).__name__})"


def _query_vision_for_coords(description: str, png_bytes: bytes, w: int, h: int) -> tuple[int, int] | None:
    """One-shot ask: find a UI element in a screenshot. Returns pixel coords
    in the SAME coordinate space as the input image, or None."""
    prompt = (
        f"You are helping a UI automation agent click PRECISELY on a target.\n"
        f"The image is {w}x{h} pixels. Origin (0,0) is the TOP-LEFT.\n"
        f"Target: {description}\n\n"
        f"Reply with ONLY the pixel coordinates of the EXACT VISUAL CENTRE of the "
        f"clickable element (not the centre of its label, the centre of the "
        f"clickable area itself).\n"
        f"Format: X,Y    (e.g. 432,718)\n"
        f"If the element isn't visible, reply: NOT_FOUND"
    )
    answer = ask_vision(prompt, png_bytes).strip()
    if "NOT_FOUND" in answer.upper():
        return None
    ans = answer.strip()
    # The model is told to reply with ONLY the coordinates, so do NOT mine an
    # arbitrary "\d+,\d+" out of prose — a reply like "I see 2 buttons, 3 tabs"
    # was yielding a confident click at (2,3) (esp. on the chattier local VLM).
    # Accept a clean coordinate reply, or a coordinate at the very END of a
    # SHORT answer; otherwise refuse — safer not to click than to click
    # garbage on the user's live screen. 2026-05-30 deep audit.
    m = re.fullmatch(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\.?", ans)
    if not m and len(ans) <= 32:
        m = re.search(r"(\d+)\s*,\s*(\d+)\s*\)?\.?\s*$", ans)
    if not m:
        return None
    x, y = int(m.group(1)), int(m.group(2))
    if 0 <= x <= w and 0 <= y <= h:
        return x, y
    return None


def _captured_region(monitor: str | None) -> dict | None:
    """Geometry of the region take_screenshot(monitor) actually grabs, as the
    SAME mss dict take_screenshot uses: {left, top, width, height} in the
    capture's NATIVE pixel space.

    This is the single source of truth that ties the capture origin to the
    click translation in find_click_target. For monitor=None it returns the
    LIVE mss virtual-screen rect (mss.monitors[0]) — whose left/top is the true
    bounding-box origin of all displays and is NEGATIVE on this owner's rig
    (a monitor sits left of/above the primary). Reading it live (instead of
    recomputing from the static MONITORS config) means the absolute-click origin
    can never silently disagree with what was photographed — the failure mode
    that lands clicks on the wrong monitor when the config and the real desktop
    diverge (e.g. a display was rearranged, or per-monitor DPI makes the
    DPI-aware mss origin differ from the logical config). For a named monitor it
    returns that monitor's configured rect. Returns None if mss is unavailable
    and no usable config entry exists."""
    if monitor and monitor in MONITORS:
        m = MONITORS[monitor]
        if len(m) >= 4 and m[2] and m[3]:
            return {"left": int(m[0]), "top": int(m[1]),
                    "width": int(m[2]), "height": int(m[3])}
    try:
        import mss
        _MSSClass = getattr(mss, "MSS", mss.mss)
        with _MSSClass() as sct:
            vs = sct.monitors[0]   # virtual screen — the monitor=None default
            return {"left": int(vs["left"]), "top": int(vs["top"]),
                    "width": int(vs["width"]), "height": int(vs["height"])}
    except Exception:
        return None


def _native_capture_size(monitor: str | None) -> tuple[int, int]:
    """Native pixel (width, height) of the region take_screenshot(monitor)
    captures. Used to rescale Pass-1 coords back to native when the full-res
    Pass-2 capture fails. Returns (0, 0) if it can't be determined.

    Prefers the LIVE mss size (true native pixels) for both the virtual screen
    and a named monitor, falling back to the configured MONITORS size only when
    mss can't be queried. The config holds LOGICAL dims, so on a DPI-scaled
    display the live mss size is the correct 'native' value here."""
    reg = _captured_region(monitor)
    if reg and reg["width"] and reg["height"]:
        return int(reg["width"]), int(reg["height"])
    if monitor and monitor in MONITORS:
        m = MONITORS[monitor]
        if len(m) >= 4 and m[2] and m[3]:
            return int(m[2]), int(m[3])
    return 0, 0


def find_click_target(description: str, monitor: str | None = None) -> tuple[int, int] | None:
    """Two-pass vision: locate a UI element with sub-pixel-ish precision.

    Pass 1: downscaled full screenshot → rough estimate.
    Pass 2: crop a small native-resolution region around the estimate and
            ask vision to refine the coords inside that crop.

    Returns absolute screen coordinates the click action can use directly,
    or None if the element isn't found.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    # Pass 1: low-res
    png = take_screenshot(monitor=monitor, max_dim=1568)
    if png is None:
        return None
    img1 = Image.open(io.BytesIO(png))
    w1, h1 = img1.size

    pass1 = _query_vision_for_coords(description, png, w1, h1)
    if pass1 is None:
        return None
    rx1, ry1 = pass1

    # Capture the geometry of the region we actually photographed ONCE, from the
    # same source take_screenshot used (live mss for monitor=None, the config
    # rect for a named monitor). This single rect drives BOTH the native size we
    # un-downscale to AND the absolute origin we add in the translate below — so
    # the click origin can never silently disagree with what was captured.
    cap = _captured_region(monitor)

    # Pass 2: grab the full-res monitor and crop a small region around the estimate
    full_png = take_screenshot(monitor=monitor, max_dim=10000)   # 10k = effectively no downscale
    if full_png is None:
        # Pass-2 capture failed — we have no full-res image to refine on.
        # CRITICAL: img1 is the DOWNSCALED Pass-1 image (max_dim=1568), so its
        # coords are NOT native screen pixels. Use the TRUE native size of the
        # captured region so the Pass-1 coords scale to native below. The old
        # code set full_w,full_h = w1,h1, treating downscaled coords as
        # full-res → an off-by-hundreds / wrong-monitor click on any >1568px
        # display (or the virtual desktop, where it could land a NEGATIVE X on
        # the wrong monitor). 2026-05-30 deep audit.
        full_img = None
        nw, nh = _native_capture_size(monitor)
        full_w = nw or w1
        full_h = nh or h1
    else:
        full_img = Image.open(io.BytesIO(full_png))
        full_w, full_h = full_img.size

    # Scale pass-1 coords from low-res image space → full-res image space
    scale_to_full_x = full_w / w1
    scale_to_full_y = full_h / h1
    cx_full = int(rx1 * scale_to_full_x)
    cy_full = int(ry1 * scale_to_full_y)

    refined_x = cx_full
    refined_y = cy_full

    # Only do pass 2 if we actually have a higher-res image to refine on
    # (full_img is None when the Pass-2 capture failed — see fallback above).
    if full_img is not None and (full_w > w1 or full_h > h1):
        CROP = 500   # 500x500 region around the estimate
        left   = max(0, cx_full - CROP // 2)
        top    = max(0, cy_full - CROP // 2)
        right  = min(full_w, cx_full + CROP // 2)
        bottom = min(full_h, cy_full + CROP // 2)
        crop = full_img.crop((left, top, right, bottom))
        cw, ch = crop.size
        crop_buf = io.BytesIO()
        crop.save(crop_buf, format="PNG")
        crop_png = crop_buf.getvalue()

        print(f"  [vision] 🔍 Refining position in {cw}x{ch} crop…", flush=True)
        pass2 = _query_vision_for_coords(description, crop_png, cw, ch)
        if pass2 is not None:
            refined_x = left + pass2[0]
            refined_y = top  + pass2[1]
            print(f"  [vision] ✓ Refined offset: {pass2[0]-cw//2:+},{pass2[1]-ch//2:+} px", flush=True)
        # If pass 2 fails, keep pass-1 estimate

    # Translate full-res image coords → absolute screen coordinates.
    #
    # refined_x/refined_y are an offset (in the captured image's NATIVE pixels)
    # from the TOP-LEFT of the region we photographed. To produce a coordinate
    # pyautogui can click we map that offset proportionally onto the region's
    # LOGICAL (DIP) extent and add the region's LOGICAL origin:
    #
    #     abs = logical_origin + refined * (logical_extent / native_extent)
    #
    # Two independent things this gets right on the owner's negative-origin,
    # multi-monitor rig (virtual desktop 7680x2880 @ -2560,-1440):
    #   • ORIGIN. We add the region's real top-left, which is NEGATIVE here. For
    #     monitor=None that origin is taken from the SAME capture (live mss
    #     monitors[0]) rather than recomputed from a possibly-stale config, so
    #     the add can't disagree with the photo. A target on the top-left
    #     monitor therefore lands at a negative absolute coordinate instead of
    #     being clamped onto the primary.
    #   • DPI. native_extent comes from the actual captured image (or the live
    #     mss size when Pass-2 failed), logical_extent from the config, so a
    #     display scaled >100% — where the native grab is LARGER than its logical
    #     size — is scaled back down before the offset is applied. At 100% the
    #     ratio is exactly 1.0, so a single-monitor / un-scaled rig is unchanged.
    if monitor and monitor in MONITORS:
        mx, my, lw, lh = MONITORS[monitor]
        sx = (lw / full_w) if full_w else 1.0
        sy = (lh / full_h) if full_h else 1.0
        if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
            print(f"  [vision] DPI scale: monitor={monitor} "
                  f"native={full_w}x{full_h} logical={lw}x{lh} "
                  f"→ click x({sx:.3f},{sy:.3f})", flush=True)
        return int(mx + refined_x * sx), int(my + refined_y * sy)
    # No monitor specified — Pass-2 captured the whole virtual screen (mss
    # monitors[0]). Use the LOGICAL origin (config-derived virtual bounds, which
    # is what pyautogui clicks in) but scale by the LIVE captured native size so
    # the origin and the scale come from one coherent picture of the desktop.
    vx, vy, vw, vh = _virtual_screen_bounds()
    # Prefer the live captured origin when it is available AND matches the
    # logical virtual origin (uniform-DPI / correctly-configured case). When the
    # config is stale the two diverge; we keep the logical origin because that is
    # the space pyautogui clicks in, but emit a one-line warning so a genuinely
    # wrong layout is visible in the log rather than silently mis-clicking.
    if cap is not None and (cap["left"] != vx or cap["top"] != vy):
        print(f"  [vision] ⚠ captured virtual origin ({cap['left']},{cap['top']}) "
              f"≠ configured ({vx},{vy}); using configured (pyautogui-logical) "
              f"origin. If clicks miss, re-run --list-monitors.", flush=True)
    sx = (vw / full_w) if full_w else 1.0
    sy = (vh / full_h) if full_h else 1.0
    if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
        print(f"  [vision] DPI scale: virtual native={full_w}x{full_h} "
              f"logical={vw}x{vh} → click x({sx:.3f},{sy:.3f})", flush=True)
    return int(vx + refined_x * sx), int(vy + refined_y * sy)


# ──────────────────────────────────────────────────────────────────────────
#  UI AUTOMATION  — click, type, key-press via pyautogui
# ──────────────────────────────────────────────────────────────────────────

_pyautogui = None


class UIFailsafeError(Exception):
    """Raised when pyautogui's fail-safe trips and a mouse-nudge retry
    also failed. Callers should catch this and surface the friendly
    message instead of the raw pyautogui traceback."""


_FAILSAFE_MSG = (
    "Your mouse was in a corner, sir — try again with the cursor elsewhere"
)


def _get_pyautogui():
    global _pyautogui
    if _pyautogui is None:
        try:
            import pyautogui as pag
            pag.FAILSAFE = True   # slam mouse into top-left corner to abort
            pag.PAUSE    = 0.15   # small pause between actions
            _pyautogui = pag
        except ImportError:
            print("  [ui] pyautogui not installed — run: pip install pyautogui")
    return _pyautogui


def _nudge_from_corner(pag) -> bool:
    """If the cursor is sitting on (or within 5 px of) a pyautogui FAILSAFE
    point, move it ~50 px toward screen center so the next call can run.
    Returns True if a nudge was performed."""
    try:
        cx, cy = pag.position()
        sw, sh = pag.size()
        corners = getattr(pag, "FAILSAFE_POINTS", [(0, 0)])
        for fx, fy in corners:
            if abs(cx - fx) <= 5 and abs(cy - fy) <= 5:
                new_x = 50 if fx < sw / 2 else sw - 50
                new_y = 50 if fy < sh / 2 else sh - 50
                # Disable FAILSAFE for the nudge itself so the move doesn't trip it.
                prev = pag.FAILSAFE
                pag.FAILSAFE = False
                try:
                    pag.moveTo(new_x, new_y, duration=0.05)
                finally:
                    pag.FAILSAFE = prev
                return True
    except Exception:
        pass
    return False


def _ui_safe(pag, op, *args, **kwargs):
    """Run a pyautogui operation. On FailSafeException, nudge the mouse off
    the corner and retry once; if it still trips, raise UIFailsafeError."""
    try:
        return op(*args, **kwargs)
    except Exception as e:
        # Match by class name so we don't have to import the exception type
        # at module load (pyautogui may be unavailable).
        if type(e).__name__ != "FailSafeException":
            raise
        if _nudge_from_corner(pag):
            try:
                return op(*args, **kwargs)
            except Exception as e2:
                if type(e2).__name__ == "FailSafeException":
                    raise UIFailsafeError(_FAILSAFE_MSG) from e2
                raise
        raise UIFailsafeError(_FAILSAFE_MSG) from e


def ui_click(x: int, y: int, button: str = "left"):
    pag = _get_pyautogui()
    if not pag:
        return
    # Clamp into the virtual-desktop bounds so a bad coordinate — a negative X
    # from a failed two-pass refine, or an out-of-range LLM-emitted
    # [ACTION: click, x,y] — can't fire pyautogui at the wrong monitor or off
    # screen. pyautogui's FAILSAFE only trips at the (0,0) corner, not on
    # arbitrary bad coords. 2026-05-30 deep audit.
    try:
        vx, vy, vw, vh = _virtual_screen_bounds()
        x = max(vx, min(int(x), vx + vw - 1))
        y = max(vy, min(int(y), vy + vh - 1))
    except Exception:
        pass
    # Flash the reticle BEFORE the click so the user sees where JARVIS aimed
    # even if the click itself raises (FailSafe, etc.). The overlay auto-fades
    # in ~2s — failed actions still leave a visible reticle for debugging.
    label = "click" if button == "left" else f"{button}-click"
    _publish_reticle(x, y, label)
    _ui_safe(pag, pag.click, x, y, button=button)


def ui_double_click(x: int, y: int):
    pag = _get_pyautogui()
    if not pag:
        return
    _publish_reticle(x, y, "dblclick")
    _ui_safe(pag, pag.doubleClick, x, y)

def ui_type(text: str):
    pag = _get_pyautogui()
    if not pag:
        return
    center = _active_window_center()
    if center is not None:
        _publish_reticle(center[0], center[1], "type")
    _ui_safe(pag, pag.write, text, interval=0.04)

def ui_press(key: str):
    pag = _get_pyautogui()
    if not pag:
        return
    center = _active_window_center()
    if center is not None:
        _publish_reticle(center[0], center[1], f"key:{key}"[:24])
    _ui_safe(pag, pag.press, key)

def ui_hotkey(*keys: str):
    pag = _get_pyautogui()
    if not pag:
        return
    center = _active_window_center()
    if center is not None:
        _publish_reticle(center[0], center[1], "+".join(keys)[:24])
    _ui_safe(pag, pag.hotkey, *keys)

def ui_scroll(amount: int):
    pag = _get_pyautogui()
    if not pag:
        return
    try:
        pos = pag.position()
        _publish_reticle(int(pos[0]), int(pos[1]), f"scroll {amount}")
    except Exception:
        pass
    _ui_safe(pag, pag.scroll, amount)


# ──────────────────────────────────────────────────────────────────────────
#  PC CONTROL  — Bobert can do things on the computer.
#
#  The LLM emits action tokens in its reply, e.g.:
#     [ACTION: open_url, https://news.ycombinator.com]
#     [ACTION: launch_app, notepad]
#     [ACTION: web_search, claude code documentation]
#     [ACTION: youtube, lofi hip hop]
#     [ACTION: screenshot]
#     [ACTION: get_time]
#
#  The token is stripped from what gets spoken aloud, then executed.
# ──────────────────────────────────────────────────────────────────────────

import shutil
import subprocess
import urllib.parse
import webbrowser

# Action-name class includes digits: `[a-z_]+` truncated names like
# `switch_to_gpt4` at the first digit, so any digit-bearing action name (or a
# future MCP-catalog name) silently failed to dispatch. 2026-05-30 deep audit.
_ACTION_RE = re.compile(r"\[ACTION:\s*([a-z0-9_]+)\s*(?:,\s*(.+?))?\s*\]", re.IGNORECASE)


# Phase 4A refactor (2026-05-29): _act_open_url moved to core/actions.py.
# Imported via `from core.actions import *` higher up in this file.
# The ACTIONS dispatch dict still resolves it by name.


# Common Chrome install locations. First existing path wins. Used by
# _open_url_new_window when we need a SEPARATE Chrome window rather than
# a tab in the user's current window.
_CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]


def _find_chrome() -> str | None:
    for p in _CHROME_PATHS:
        if p and os.path.exists(p):
            return p
    return None


def _open_url_new_window(url: str) -> bool:
    """Spawn a NEW Chrome window for `url`, leaving any existing Chrome
    windows untouched. Returns True if a new window was spawned, False if
    Chrome wasn't found (caller should fall back to webbrowser.open)."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    chrome = _find_chrome()
    if not chrome:
        return False
    try:
        # --new-window forces a separate top-level window even when Chrome
        # is already running. Without it, Chrome reuses the most recent
        # window and just adds a tab — which would steal focus from
        # whatever the user was watching there.
        subprocess.Popen([chrome, "--new-window", url], close_fds=True)
        return True
    except Exception:
        return False


# Coordinates well outside the virtual screen bounding box for this rig.
# MONITORS spans (-2560,-1440)..(5120,1440); placing a window at (10000,-30000)
# guarantees no part of it lands on any real monitor.
_OFFSCREEN_X, _OFFSCREEN_Y = 10000, -30000
_OFFSCREEN_W, _OFFSCREEN_H = 1600, 1200


def _open_url_offscreen_capture(url: str, page_load_wait: float = 6.0) -> tuple[bytes | None, str]:
    """Open `url` in a NEW Chrome window that is immediately parked off the
    virtual screen, wait for the page to render, capture the window's pixel
    content directly with win32 PrintWindow (PW_RENDERFULLCONTENT), then
    close the window. Used for headless-style screenshotting that reuses the
    user's logged-in Chrome session without ever showing the window on any
    monitor.

    Returns (png_bytes, "ok") on success or (None, error_reason) on failure.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        import win32gui
        import win32ui
        import win32con
        from ctypes import windll
        from PIL import Image
    except ImportError as e:
        return None, f"missing dep for offscreen capture: {e}"

    chrome = _find_chrome()
    if not chrome:
        return None, "chrome_not_found"

    chrome_class = "Chrome_WidgetWin_1"

    def _enum_chrome_hwnds() -> set:
        found: set = set()
        def _cb(hwnd, _):
            try:
                if (win32gui.GetClassName(hwnd) == chrome_class
                        and win32gui.IsWindowVisible(hwnd)):
                    rect = win32gui.GetWindowRect(hwnd)
                    w, h = rect[2] - rect[0], rect[3] - rect[1]
                    if w > 200 and h > 200:
                        found.add(hwnd)
            except Exception:
                pass
            return True
        try:
            win32gui.EnumWindows(_cb, None)
        except Exception:
            pass
        return found

    pre = _enum_chrome_hwnds()

    try:
        # Pass the offscreen position as Chrome launch flags so the window
        # starts at (10000, -30000) from frame 0 — no visible flash, no race
        # between spawn and SetWindowPos. --new-window keeps it separate from
        # any existing Chrome windows. --no-first-run / --no-default-browser-check
        # suppress startup dialogs that would create extra visible windows.
        subprocess.Popen([
            chrome,
            "--new-window",
            f"--window-position={_OFFSCREEN_X},{_OFFSCREEN_Y}",
            f"--window-size={_OFFSCREEN_W},{_OFFSCREEN_H}",
            "--no-first-run",
            "--no-default-browser-check",
            url,
        ], close_fds=True)
    except Exception as e:
        return None, f"chrome_spawn_failed: {e}"

    # Tight poll for the new top-level Chrome window so we can move it off
    # screen with minimal on-screen flash (~50ms in practice).
    target_hwnd = None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        time.sleep(0.02)
        delta = _enum_chrome_hwnds() - pre
        if delta:
            target_hwnd = next(iter(delta))
            break

    if not target_hwnd:
        return None, "chrome_window_not_found"

    # Park the window off the virtual screen so it isn't visible anywhere.
    # SWP_NOZORDER (0x0004) | SWP_NOACTIVATE (0x0010) — don't steal focus.
    try:
        win32gui.SetWindowPos(
            target_hwnd, 0,
            _OFFSCREEN_X, _OFFSCREEN_Y, _OFFSCREEN_W, _OFFSCREEN_H,
            0x0004 | 0x0010,
        )
    except Exception:
        pass

    time.sleep(page_load_wait)

    png_bytes: bytes | None = None
    try:
        left, top, right, bot = win32gui.GetWindowRect(target_hwnd)
        w, h = right - left, bot - top
        if w > 0 and h > 0:  # pragma: no cover - live Win32 GDI PrintWindow capture of a real offscreen window handle
            hwndDC = None
            mfcDC = None
            saveDC = None
            saveBitMap = None
            try:
                hwndDC = win32gui.GetWindowDC(target_hwnd)
                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()
                saveBitMap = win32ui.CreateBitmap()
                saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
                saveDC.SelectObject(saveBitMap)
                # PW_RENDERFULLCONTENT (0x00000002) forces a full re-render via
                # the GDI DC — required for hardware-accelerated apps like
                # Chrome where the front buffer isn't otherwise GDI-readable.
                ok = windll.user32.PrintWindow(
                    target_hwnd, saveDC.GetSafeHdc(), 0x00000002
                )
                if ok:
                    bmpinfo = saveBitMap.GetInfo()
                    bmpstr  = saveBitMap.GetBitmapBits(True)
                    img = Image.frombuffer(
                        "RGB",
                        (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
                        bmpstr, "raw", "BGRX", 0, 1,
                    )
                    if max(img.size) > 1568:
                        ratio = 1568 / max(img.size)
                        img = img.resize(
                            (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                            Image.LANCZOS,
                        )
                    buf = io.BytesIO()
                    img.save(buf, format="PNG", optimize=True)
                    png_bytes = buf.getvalue()
            finally:
                # Release every GDI handle we actually acquired, in reverse
                # order of acquisition. Each cleanup is independently guarded
                # so one failure doesn't strand the rest.
                if saveBitMap is not None:
                    try:
                        win32gui.DeleteObject(saveBitMap.GetHandle())
                    except Exception:
                        pass
                if saveDC is not None:
                    try:
                        saveDC.DeleteDC()
                    except Exception:
                        pass
                if mfcDC is not None:
                    try:
                        mfcDC.DeleteDC()
                    except Exception:
                        pass
                if hwndDC is not None:
                    try:
                        win32gui.ReleaseDC(target_hwnd, hwndDC)
                    except Exception:
                        pass
    except Exception as e:
        # Continue to close even if capture failed
        print(f"  [offscreen] PrintWindow capture error: {e}")

    # Close the window by posting WM_CLOSE (0x0010) directly to the HWND.
    # Safer than matching by title — guaranteed to hit the right window.
    try:
        win32gui.PostMessage(target_hwnd, 0x0010, 0, 0)
    except Exception:  # pragma: no cover - defensive: WM_CLOSE PostMessage to a live HWND rarely raises
        pass

    if png_bytes is None:
        return None, "printwindow_failed"
    return png_bytes, "ok"  # pragma: no cover - live Win32 GDI PrintWindow grab succeeded; needs a real on-screen window


# Known apps that aren't on PATH and don't have a Start-menu shortcut name
# that os.startfile() can resolve. Add new entries here when JARVIS fails to
# launch something the user asks for. Keys are normalised lower-case;
# aliases all point at the same canonical entry.
#
# Each value is a list of candidate absolute paths, tried in order — the
# first one that exists wins. Use lists so a single alias works across
# install locations (Program Files vs Program Files (x86) vs LocalAppData).
_KNOWN_APP_PATHS: dict[str, list[str]] = {
    "bambu studio": [
        r"C:\Program Files\Bambu Studio\bambu-studio.exe",
        r"C:\Program Files (x86)\Bambu Studio\bambu-studio.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Bambu Studio\bambu-studio.exe"),
    ],
}

# Aliases → canonical key in _KNOWN_APP_PATHS. Whisper transcripts can be
# inconsistent ("bambustudio", "bamboo studio") so we cast a wide net.
_KNOWN_APP_ALIASES: dict[str, str] = {
    "bambu":         "bambu studio",
    "bambustudio":   "bambu studio",
    "bambu_studio":  "bambu studio",
    "bambu-studio":  "bambu studio",
    "bamboo":        "bambu studio",
    "bamboo studio": "bambu studio",
    "bambu slicer":  "bambu studio",
}


def _resolve_known_app(name: str) -> str | None:
    """Return an existing executable path for `name` if it matches a known
    alias, otherwise None. Case-insensitive, tolerant of extra whitespace."""
    key = re.sub(r"\s+", " ", name.strip().lower())
    key = _KNOWN_APP_ALIASES.get(key, key)
    for candidate in _KNOWN_APP_PATHS.get(key, []):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


# Phase 4D refactor: _act_launch_app moved to core/actions.py.


# Match a YouTube video ID in any of the forms Google's SERP HTML actually
# emits: bare URL, www./m. subdomain, youtu.be short link, AND the percent-
# encoded `/url?q=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D…` wrapper.
_YT_VIDEO_RE = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?youtube\.com/watch\?[^\s\"'<>]*v=([A-Za-z0-9_-]{11})"
    r"|(?:https?://)?youtu\.be/([A-Za-z0-9_-]{11})"
    r"|youtube\.com(?:%2F|/)watch(?:%3F|\?)[^\s\"'<>]*v(?:%3D|=)([A-Za-z0-9_-]{11})"
)
_VIDEO_QUERY_HINTS = (
    "youtube", "video", "watch", "music video", "trailer",
    "yt ", " yt", "song ", "play "
)


def _extract_youtube_url_from_search(query: str) -> str | None:
    """Resolve *query* to a direct youtube.com/watch?v=… URL, or None.

    Two-stage lookup:
      1. If the youtube_search skill is loaded and yt-dlp is installed,
         use it — that bypasses Google rate-limiting and gives us the
         canonical watch URL without HTML parsing.
      2. Otherwise (or if yt-dlp returned nothing) fall back to the
         original Google SERP regex extractor with a 5 s budget.

    Failure is silent at every layer; the caller falls back to opening
    Google in the browser as before."""
    # Stage 1 — yt-dlp via the youtube_search skill (no Google involved).
    yt_skill = sys.modules.get("skill_youtube_search")
    if yt_skill is not None:
        try:
            direct = yt_skill.find_direct_url(query)
            if direct:
                return direct
        except Exception:
            pass

    # Stage 2 — Google SERP regex fallback. Kept verbatim so the video-intent
    # shortcut still has a chance even when yt-dlp is missing or rate-limited.
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200 or not r.text:
            return None
        m = _YT_VIDEO_RE.search(r.text)
        if not m:
            return None
        vid = m.group(1) or m.group(2) or m.group(3)
        if not vid:
            return None  # pragma: no cover - defensive: every _YT_VIDEO_RE alternative captures a fixed 11-char id, so a match never yields an empty group
        return f"https://www.youtube.com/watch?v={vid}"
    except Exception:
        return None


# Phase 4A refactor (2026-05-29): _act_web_search moved to core/actions.py.
# Imported via `from core.actions import *` higher up in this file.
# The ACTIONS dispatch dict still resolves it by name.


# Phase 4A refactor (2026-05-29): _act_youtube moved to core/actions.py.
# Imported via `from core.actions import *` higher up in this file.
# The ACTIONS dispatch dict still resolves it by name.


# ──────────────────────────────────────────────────────────────────────────
#  STREAMING SERVICES AUTO-PLAY
#
#  When the user says "play [title]" — or "play [title] on Netflix",
#  "play [song] on Spotify", etc. — open the service in the browser, click
#  the first search result, then click play. Falls back gracefully if vision
#  can't locate the elements.
# ──────────────────────────────────────────────────────────────────────────

# Each service knows where its search page lives and what vision hints to use
# for the two click steps (open the first result, then start playback).
#  fullscreen_key:  key (or hotkey tuple) to press once playback has actually
#                   started, to put the player into full-screen. None = skip.
#                   For music services (Apple Music, Spotify) 'f' opens the
#                   full-screen Now Playing view in the web player.
#  fullscreen_wait: seconds to wait after the play click before pressing the
#                   key — gives the player time to initialise and grab focus.
_STREAMING_SERVICES = {
    "netflix": {
        "name":             "Netflix",
        "home":             "https://www.netflix.com",
        "search_url":       "https://www.netflix.com/search?q={q}",
        "load_wait":        5.0,
        "post_click":       3.5,
        "result_hint":      "the first show or movie thumbnail in the search results grid (not the search box, not a header link)",
        "play_hint":        "the large red play button on the title's detail page",
        "fullscreen_key":   "f",
        "fullscreen_wait":  3.0,
    },
    "prime_video": {
        "name":             "Prime Video",
        "home":             "https://www.primevideo.com",
        "search_url":       "https://www.primevideo.com/search/ref=atv_nb_sr?phrase={q}&ie=UTF8",
        "load_wait":        5.0,
        "post_click":       3.5,
        "result_hint":      "the first movie or TV show poster in the Prime Video search results",
        "play_hint":        "the Play button or Watch Now button on the title's detail page",
        "fullscreen_key":   "f",
        "fullscreen_wait":  4.0,   # Prime can be slow to start the stream
    },
    "disney_plus": {
        "name":             "Disney+",
        "home":             "https://www.disneyplus.com",
        "search_url":       "https://www.disneyplus.com/search?q={q}",
        "load_wait":        5.0,
        "post_click":       3.5,
        "result_hint":      "the first show or movie tile in the Disney+ search results",
        "play_hint":        "the Play button on the title's detail page",
        "fullscreen_key":   "f",
        "fullscreen_wait":  3.5,
    },
    "hulu": {
        "name":             "Hulu",
        "home":             "https://www.hulu.com",
        "search_url":       "https://www.hulu.com/search?q={q}",
        "load_wait":        5.0,
        "post_click":       3.5,
        "result_hint":      "the first show or movie tile in the Hulu search results",
        "play_hint":        "the Play button on the title's detail page",
        "fullscreen_key":   "f",
        "fullscreen_wait":  3.5,
    },
    "max": {
        "name":             "Max",
        "home":             "https://play.max.com",
        "search_url":       "https://play.max.com/search?q={q}",
        "load_wait":        5.0,
        "post_click":       3.5,
        "result_hint":      "the first show or movie tile in the Max search results",
        "play_hint":        "the Play button on the title's detail page",
        "fullscreen_key":   "f",
        "fullscreen_wait":  3.5,
    },
    "apple_music": {
        "name":             "Apple Music",
        "home":             "https://music.apple.com",
        "search_url":       "https://music.apple.com/us/search?term={q}",
        "load_wait":        5.0,
        "post_click":       3.0,
        # Result selection: keyboard navigation (Tab + Enter) instead of
        # vision+click. find_on_screen coordinates were unreliable on Apple
        # Music's dense web layout (Top Result + tile grid + song rows),
        # and a single-click on a song row only selects it. Tabbing past
        # the page chrome and pressing Enter on the first focusable result
        # works regardless of which result type appears first.
        "select_method":    "keyboard",
        "keyboard_pre_wait":     1.0,
        "keyboard_tab_count":    6,
        "keyboard_tab_interval": 0.18,
        "keyboard_post_wait":    0.6,
        "result_hint":      (
            "the 'Top Result' card OR the first album / artist tile in the "
            "Apple Music search results — strongly prefer the Top Result "
            "card or an album/artist tile over a plain song row, because "
            "single-clicking a song row only selects it (it does NOT start "
            "playback or navigate to a detail page). If only song rows are "
            "visible, pick the first song row, but otherwise prefer a tile "
            "that opens an album/artist/playlist detail page. Do NOT pick "
            "the search box, the sidebar, header links, or any small "
            "inline play icon."
        ),
        "play_hint":        (
            "ONLY the LARGE round Play button (▶) or the Shuffle button at "
            "the TOP of an Apple Music album / artist / playlist detail "
            "page (positioned next to the large artwork). Do NOT match: a "
            "song row in a track list, a small inline play arrow inside a "
            "track row, the search results page, the now-playing footer "
            "controls at the bottom of the screen, or a generic 'play' "
            "text label. If no such large detail-page Play/Shuffle button "
            "is clearly visible, return that no target was found."
        ),
        "fullscreen_key":   "f",
        "fullscreen_wait":  2.5,
        # Strict mode: after pressing Enter (or clicking play), verify
        # with vision that playback actually started and retry if it
        # didn't. Apple Music's detail pages occasionally swallow the
        # first action (especially on a fresh page-load) and JARVIS used
        # to silently leave the user staring at a paused album page.
        "verify_play":      True,
        "verify_first":     True,   # check before any play attempt — Enter on a song row may already be playing
        "verify_attempts":  3,
        "verify_wait":      2.5,
        # Retry with a *different* strategy each attempt rather than
        # blindly re-clicking the same play button. Order matters:
        #   1. play_button — works for album/artist/playlist tiles that
        #                    navigated to a detail page after Enter.
        #   2. space       — works for a song row that received focus
        #                    but Enter didn't start playback.
        #   3. play_button — final fallback in case the detail page just
        #                    finished loading.
        "play_strategies":  ["play_button", "space", "play_button"],
        "verify_question": (
            "Is music ACTIVELY playing right now on this Apple Music page? "
            "Answer YES only if at least one of these is clearly visible: "
            "(a) the now-playing footer at the bottom of the screen "
            "showing a track title with a visible progress scrubber, "
            "(b) a pause button (two vertical bars) where a large round "
            "Play button used to be on a detail page, "
            "(c) an animated equaliser / 'currently playing' indicator "
            "next to a song row in the track list. "
            "Answer NO if you only see: a paused detail page with a "
            "triangular Play button (▶), a static search-results page, "
            "or an empty now-playing footer. "
            "Reply with YES or NO on the first line, then a one-line reason."
        ),
    },
    "spotify": {
        "name":             "Spotify",
        "home":             "https://open.spotify.com",
        "search_url":       "https://open.spotify.com/search/{q}",
        "load_wait":        4.0,
        "post_click":       3.0,
        "result_hint":      "the Top Result card or the first song / album / artist in the Spotify search results",
        "play_hint":        "the big green play button on the album / artist / playlist page",
        "fullscreen_key":   "f",
        "fullscreen_wait":  2.5,
    },
    "youtube": {
        "name":             "YouTube",
        "home":             "https://www.youtube.com",
        "search_url":       "https://www.youtube.com/results?search_query={q}",
        "load_wait":        3.5,
        "post_click":       0.0,
        "result_hint":      "the first video thumbnail in the YouTube search results (skip ads and 'Shorts' rows — pick a normal video result)",
        "play_hint":        None,   # YouTube auto-plays once a video is opened
        "fullscreen_key":   "f",
        "fullscreen_wait":  4.0,    # let the watch page + player finish loading
    },
}

# Aliases the LLM (or user, via the LLM) might emit. Normalised by
# _normalize_service before dict lookup.
_STREAMING_ALIASES = {
    "prime":              "prime_video",
    "amazon":             "prime_video",
    "amazon_prime":       "prime_video",
    "amazon_video":       "prime_video",
    "primevideo":         "prime_video",
    "disney":             "disney_plus",
    "disneyplus":         "disney_plus",
    "disney+":            "disney_plus",
    "hbo":                "max",
    "hbo_max":            "max",
    "hbomax":             "max",
    "apple":              "apple_music",
    "applemusic":         "apple_music",
    "itunes_store":       "apple_music",   # for streaming requests — local lib uses play_music
    "yt":                 "youtube",
    "you_tube":           "youtube",
}


def _normalize_service(s: str) -> str:
    key = re.sub(r"[\s\-]+", "_", s.strip().lower())
    key = key.replace("+", "_plus")
    return _STREAMING_ALIASES.get(key, key)


def _vision_answer_is_yes(answer: str) -> bool:
    """Strict YES/NO parser for the verify-playback prompt. Returns True only
    when the first word of the answer is YES (case-insensitive). Anything
    ambiguous — including vision-failure stubs like '(vision failed: …)' —
    counts as NO so the caller retries instead of declaring victory."""
    if not answer:
        return False
    first = answer.strip().split(maxsplit=1)[0] if answer.strip() else ""
    return first.rstrip(".,:;!?").upper() == "YES"


def _streaming_find_with_retry(
    hint: str, attempts: int = 3, wait_between: float = 2.0
) -> tuple[int, int] | None:
    """Wrap find_click_target with retries — Apple Music's detail pages can
    finish painting after our initial load_wait elapsed, so a single miss
    shouldn't mean we bail to 'click it yourself'. Returns None only if
    every attempt fails."""
    for i in range(attempts):
        coords = find_click_target(hint)
        if coords is not None:
            return coords
        if i < attempts - 1:
            print(
                f"  [auto-play] vision didn't see target, retrying in "
                f"{wait_between}s (attempt {i + 2}/{attempts})…",
                flush=True,
            )
            time.sleep(wait_between)
    return None


def _streaming_verify_playback(verify_question: str) -> tuple[bool, str]:
    """Take a fresh screenshot and ask vision whether playback is actually
    happening. Returns (is_playing, raw_answer). Vision-failures collapse
    to (False, error_text) so the caller retries the play click."""
    png = take_screenshot()
    if png is None:
        return False, "could not capture screen"
    answer = ask_vision(verify_question, png)
    return _vision_answer_is_yes(answer), answer


def _streaming_apply_play_strategy(
    strategy: str,
    cfg: dict,
    result_coords: tuple[int, int] | None,
) -> tuple[bool, str]:
    """Apply one play strategy. Returns (was_attempted, description).
    was_attempted=False means the strategy was a no-op for this context
    (e.g. double_click_result with no remembered result coords); the
    caller should advance to the next strategy without verifying.
    UIFailsafeError propagates — caller surfaces it to the user."""
    if strategy == "play_button":
        play_coords = _streaming_find_with_retry(
            cfg["play_hint"], attempts=2, wait_between=1.5
        )
        if play_coords is None:
            return False, "vision couldn't locate a detail-page play button"
        ui_click(play_coords[0], play_coords[1])
        return True, f"clicked detail-page play button at {play_coords}"
    if strategy == "double_click_result":
        if result_coords is None:
            return False, "no remembered result coords to double-click"
        ui_double_click(result_coords[0], result_coords[1])
        return True, f"double-clicked first result at {result_coords}"
    if strategy == "space":
        ui_press("space")
        return True, "pressed space (play/pause shortcut)"
    if strategy == "playpause":
        ui_press("playpause")
        return True, "pressed media play/pause key"
    return False, f"unknown play strategy '{strategy}'"


def _streaming_play_and_verify(
    cfg: dict,
    service_label: str,
    q: str,
    result_coords: tuple[int, int] | None = None,
) -> str:
    """Click play, then verify with vision that playback actually started.
    Retries up to `verify_attempts` times, using a different strategy each
    attempt (from cfg['play_strategies'], default ['play_button']) so we
    don't blindly repeat a click that already failed. Used by services
    with `verify_play: True` (Apple Music) so JARVIS doesn't silently
    leave the user staring at a paused/unstarted page.

    `result_coords` is the screen position of the search result we
    originally clicked — needed for the `double_click_result` fallback
    that fixes Apple Music song rows (single-click only selects them)."""
    attempts = max(1, int(cfg.get("verify_attempts", 3)))
    verify_wait = float(cfg.get("verify_wait", 2.5))
    verify_q = cfg.get("verify_question") or (
        f"Is media currently playing on this {service_label} page? Answer "
        "YES or NO on the first line, then a brief reason."
    )
    strategies = list(cfg.get("play_strategies") or ["play_button"])

    # Optional pre-check: if the result-selection step (e.g. keyboard
    # Enter on a focused song row) already started playback, skip the
    # play strategies entirely. Without this, we'd needlessly press the
    # play button on an already-playing track and toggle it off.
    if cfg.get("verify_first"):
        print(
            f"  [auto-play] checking whether playback already started on "
            f"{service_label}…",
            flush=True,
        )
        is_playing, vision_answer = _streaming_verify_playback(verify_q)
        if is_playing:
            snippet = (vision_answer or "").strip().splitlines()
            snippet = snippet[0][:120] if snippet else "no answer"
            print(
                f"  [auto-play] ✓ already playing — skipping play step "
                f"({snippet})",
                flush=True,
            )
            _streaming_go_fullscreen(cfg, service_label)
            return f"playing '{q}' on {service_label}"

    last_msg = ""
    for attempt in range(1, attempts + 1):
        strategy = strategies[min(attempt - 1, len(strategies) - 1)]
        print(
            f"  [auto-play] play/verify attempt {attempt}/{attempts} on "
            f"{service_label} (strategy: {strategy})…",
            flush=True,
        )
        try:
            attempted, desc = _streaming_apply_play_strategy(
                strategy, cfg, result_coords
            )
        except UIFailsafeError as e:
            return f"play attempt on {service_label} aborted: {e}"

        if not attempted:
            last_msg = desc
            print(
                f"  [auto-play] ✗ skipped strategy '{strategy}': {desc}",
                flush=True,
            )
            time.sleep(1.0)
            continue

        print(
            f"  [auto-play] {desc}; verifying playback in {verify_wait}s…",
            flush=True,
        )
        time.sleep(verify_wait)

        is_playing, vision_answer = _streaming_verify_playback(verify_q)
        snippet = (vision_answer or "").strip().splitlines()
        snippet = snippet[0][:120] if snippet else "no answer"
        if is_playing:
            print(f"  [auto-play] ✓ vision confirms playback ({snippet})", flush=True)
            _streaming_go_fullscreen(cfg, service_label)
            return f"playing '{q}' on {service_label}"
        last_msg = snippet
        if attempt < attempts:
            print(
                f"  [auto-play] ✗ vision says not playing yet: {snippet} — retrying",
                flush=True,
            )
        else:
            print(
                f"  [auto-play] ✗ vision still not confirming after "
                f"{attempts} attempts: {snippet}",
                flush=True,
            )

    return (
        f"opened '{q}' on {service_label} and tried to start playback, but "
        f"vision couldn't confirm it after {attempts} attempts — you may "
        f"need to click play yourself ({last_msg})"
    )


def _streaming_keyboard_select_first_result(
    cfg: dict, service_label: str
) -> bool:
    """Focus and activate the first search result via keyboard navigation
    (Tab past the page chrome, then Enter). Avoids vision+click for the
    result step on services where find_on_screen coords were unreliable
    (Apple Music's dense web layout in particular).

    Returns True if the key sequence was sent. The caller verifies the
    outcome via the same vision-playback check used by mouse-click flows,
    so a 'wrong number of tabs' miss still gets caught by the existing
    retry-with-different-strategy loop."""
    if not UI_AUTOMATION_ENABLED:
        return False
    pre_wait     = float(cfg.get("keyboard_pre_wait", 0.8))
    tab_count    = int(cfg.get("keyboard_tab_count", 5))
    tab_interval = float(cfg.get("keyboard_tab_interval", 0.18))
    post_wait    = float(cfg.get("keyboard_post_wait", 0.5))

    # Pre-wait covers the gap between webbrowser.open returning and the
    # search page becoming interactive — Apple Music in particular keeps
    # focus on the search input for a moment after results paint.
    time.sleep(pre_wait)
    try:
        for _ in range(tab_count):
            ui_press("tab")
            time.sleep(tab_interval)
        time.sleep(post_wait)
        ui_press("enter")
    except UIFailsafeError:
        raise
    except Exception as e:
        print(
            f"  [auto-play] keyboard select on {service_label} failed: {e}",
            flush=True,
        )
        return False
    print(
        f"  [auto-play] keyboard-selected first result on {service_label} "
        f"({tab_count} tabs + enter)",
        flush=True,
    )
    return True


def _streaming_auto_play(service_key: str, query: str) -> str:
    """Open a streaming service's search page for `query`, click (or
    keyboard-activate) the first result, then click play. Services with
    `verify_play: True` additionally confirm playback via vision and
    retry the play step on failure (so JARVIS doesn't silently end at a
    paused detail page)."""
    cfg = _STREAMING_SERVICES.get(service_key)
    if cfg is None:
        return (
            f"unknown streaming service '{service_key}'. Known: "
            + ", ".join(sorted(_STREAMING_SERVICES.keys()))
        )

    q = query.strip()
    service_label = cfg["name"]

    # Empty query: just open the homepage and stop.
    if not q:
        webbrowser.open(cfg["home"])
        return f"opened {service_label}"

    # Step 1: open the search URL. Browser focus is needed for the click
    # steps to land on the right window.
    url = cfg["search_url"].format(q=urllib.parse.quote(q))
    webbrowser.open(url)
    print(f"  [auto-play] opened {service_label} search for '{q}'", flush=True)
    time.sleep(cfg["load_wait"])

    # Vision-based click requires Claude vision + UI automation
    if not (SCREEN_VISION_ENABLED and UI_AUTOMATION_ENABLED and AI_BACKEND == "claude"):
        return (
            f"opened {service_label} search for '{q}' — auto-click needs "
            f"SCREEN_VISION_ENABLED + UI_AUTOMATION_ENABLED + Claude backend"
        )

    strict = bool(cfg.get("verify_play"))
    select_method = cfg.get("select_method", "vision")

    # Step 2: activate the first result. Two paths:
    #   keyboard — Tab past the page chrome and press Enter. No coords are
    #              produced (so double_click_result fallback is unavailable),
    #              but the existing verify-and-retry loop catches misses by
    #              trying play_button / space strategies next.
    #   vision   — original behaviour: find_click_target → ui_click.
    coords: tuple[int, int] | None = None
    if select_method == "keyboard":
        try:
            sent = _streaming_keyboard_select_first_result(cfg, service_label)
        except UIFailsafeError as e:
            return f"opened {service_label} search for '{q}' but {e}"
        if not sent:
            return (
                f"opened {service_label} search for '{q}' but UI "
                f"automation is unavailable — you may need to click the "
                f"first result yourself"
            )
    else:
        print(
            f"  [auto-play] looking for first result on {service_label}…",
            flush=True,
        )
        if strict:
            coords = _streaming_find_with_retry(
                cfg["result_hint"], attempts=3, wait_between=2.0
            )
        else:
            coords = find_click_target(cfg["result_hint"])
        if coords is None:
            return (
                f"opened {service_label} search for '{q}', but couldn't see the "
                f"first result — you may need to click it yourself"
            )
        try:
            ui_click(coords[0], coords[1])
        except UIFailsafeError as e:
            return f"opened {service_label} search for '{q}' but {e}"
        print(f"  [auto-play] clicked first result at {coords}", flush=True)

    # Step 3: click the play button (services that have a separate detail
    # page after the result click).
    if not cfg.get("play_hint"):
        # No explicit play step (e.g., YouTube — clicking the thumbnail
        # navigates straight into autoplay). Still try to full-screen.
        _streaming_go_fullscreen(cfg, service_label)
        return f"playing '{q}' on {service_label}"

    time.sleep(cfg["post_click"])

    # Strict mode: click play, verify with vision, retry on failure.
    # Pass the result coords so fallback strategies (e.g. double-click on a
    # song row that single-click only selected) can re-target them.
    if strict:
        return _streaming_play_and_verify(cfg, service_label, q, result_coords=coords)

    # Default path: single play click, no verification.
    print(f"  [auto-play] looking for play button on {service_label}…", flush=True)
    play_coords = find_click_target(cfg["play_hint"])
    if play_coords is None:
        return (
            f"opened '{q}' on {service_label}, but couldn't locate the play "
            f"button — try saying 'click the play button'"
        )
    try:
        ui_click(play_coords[0], play_coords[1])
    except UIFailsafeError as e:
        return f"found play button on {service_label} but {e}"

    # Step 4: press the service's full-screen shortcut so playback fills the
    # screen automatically.
    _streaming_go_fullscreen(cfg, service_label)
    return f"playing '{q}' on {service_label}"


def _streaming_go_fullscreen(cfg: dict, service_label: str) -> None:
    """Best-effort full-screen step after a streaming service starts playing.
    Waits for the player to initialise, then sends the configured key (or
    hotkey tuple). Silent no-op if the service has no shortcut or if
    UI automation is unavailable."""
    fs_key = cfg.get("fullscreen_key")
    if not fs_key or not UI_AUTOMATION_ENABLED:
        return
    wait = float(cfg.get("fullscreen_wait", 2.5))
    try:
        time.sleep(wait)
        if isinstance(fs_key, (list, tuple)):
            ui_hotkey(*fs_key)
            pressed = "+".join(fs_key)
        else:
            ui_press(fs_key)
            pressed = fs_key
        print(
            f"  [auto-play] pressed '{pressed}' for full-screen on {service_label}",
            flush=True,
        )
    except Exception as e:
        print(f"  [auto-play] full-screen step failed: {e}", flush=True)


# Playlist intent detectors. Match either an explicit "playlist" keyword
# (prefix, "playlist:X" / "playlist X" / "playlist called X") or the natural
# suffix form ("X playlist" / "my X playlist"). Used to route playlist
# requests through the Library > Playlists flow instead of search.
_PLAYLIST_PREFIX_RE = re.compile(
    r"^(?:play\s+)?(?:my\s+|the\s+)?playlist\s*[:\-]?\s*"
    r"(?:called\s+|named\s+|titled\s+)?(.+)$",
    re.IGNORECASE,
)
_PLAYLIST_SUFFIX_RE = re.compile(
    r"^(?:play\s+)?(?:my\s+|the\s+)?(.+?)\s+playlist$",
    re.IGNORECASE,
)
_PLAYLIST_GENERIC_NAMES = {"music", "song", "songs", "track", "tracks", "stuff"}


def _looks_like_playlist_request(q: str) -> tuple[bool, str]:
    """Detect whether the user is asking for a named playlist (rather than a
    song/album/artist). Returns (is_playlist, cleaned_name)."""
    if not q:
        return False, q
    s = q.strip()
    m = _PLAYLIST_PREFIX_RE.match(s)
    if m:
        name = m.group(1).strip().strip("\"'")
        if name:
            return True, name
    m = _PLAYLIST_SUFFIX_RE.match(s)
    if m:
        name = m.group(1).strip().strip("\"'")
        if name and name.lower() not in _PLAYLIST_GENERIC_NAMES:
            return True, name
    return False, s


def _apple_music_play_playlist(name: str) -> str:
    """Navigate Apple Music's Library > Playlists directly, locate the named
    playlist via vision, open it, click Play/Shuffle, and verify playback.
    Unlike the search-based flow, this jumps straight to the saved-playlists
    view so JARVIS never scrolls aimlessly through search results."""
    cfg = _STREAMING_SERVICES["apple_music"]
    service_label = "Apple Music"

    # Step 1: open Library > Playlists directly. Skips the sidebar clicks
    # when possible; sidebar fallback covers the case where Apple Music
    # redirects (not signed in, first-load behaviour).
    library_url = "https://music.apple.com/library/playlists"
    webbrowser.open(library_url)
    print(
        f"  [auto-play] opened Apple Music Library > Playlists for '{name}'",
        flush=True,
    )
    time.sleep(cfg["load_wait"])

    if not (SCREEN_VISION_ENABLED and UI_AUTOMATION_ENABLED and AI_BACKEND == "claude"):
        return (
            f"opened Apple Music Library > Playlists — auto-click needs "
            f"SCREEN_VISION_ENABLED + UI_AUTOMATION_ENABLED + Claude backend"
        )

    # Step 2: locate the named playlist tile. If the direct URL didn't land
    # us on the playlists view, fall back to clicking 'Library' then
    # 'Playlists' in the sidebar before retrying.
    playlist_hint = (
        f"the playlist tile, row, or card labelled '{name}' (or a close "
        f"textual match) inside the Apple Music playlists list — pick the "
        f"playlist artwork/title itself, NOT the search box, NOT the "
        f"sidebar 'Playlists' link, and NOT a 'Made For You' header"
    )
    playlist_coords = _streaming_find_with_retry(
        playlist_hint, attempts=2, wait_between=2.0
    )

    if playlist_coords is None:
        print(
            "  [auto-play] direct URL didn't surface the playlist — clicking "
            "Library in the sidebar",
            flush=True,
        )
        lib_coords = find_click_target(
            "the 'Library' link in the Apple Music left sidebar"
        )
        if lib_coords is not None:
            try:
                ui_click(lib_coords[0], lib_coords[1])
            except UIFailsafeError as e:
                return f"couldn't navigate to Apple Music Library — {e}"
            time.sleep(cfg["post_click"])
            pl_coords = find_click_target(
                "the 'Playlists' sub-link in the Apple Music left sidebar "
                "(under the Library section)"
            )
            if pl_coords is not None:
                try:
                    ui_click(pl_coords[0], pl_coords[1])
                except UIFailsafeError as e:
                    return f"couldn't open Apple Music Playlists — {e}"
                time.sleep(cfg["post_click"])
        playlist_coords = _streaming_find_with_retry(
            playlist_hint, attempts=2, wait_between=2.0
        )

    if playlist_coords is None:
        return (
            f"opened Apple Music Library > Playlists, but couldn't find a "
            f"playlist named '{name}' — you may need to scroll to it or "
            f"click it yourself"
        )

    # Step 3: click into the playlist.
    try:
        ui_click(playlist_coords[0], playlist_coords[1])
    except UIFailsafeError as e:
        return f"found playlist '{name}' on Apple Music but {e}"
    print(
        f"  [auto-play] clicked playlist '{name}' at {playlist_coords}",
        flush=True,
    )
    time.sleep(cfg["post_click"])

    # Step 4-5: click Play/Shuffle and verify playback via vision. Reuses
    # the strict verify-and-retry path the album/artist search flow uses,
    # so playlists inherit the same reliability guarantees (Step 6).
    return _streaming_play_and_verify(cfg, service_label, name)


# Phase 4C refactor: _act_apple_music moved to core/actions.py.


# Phase 4B refactor: _act_netflix moved to core/actions.py.


# Phase 4B refactor: _act_prime_video moved to core/actions.py.


# Phase 4B refactor: _act_disney_plus moved to core/actions.py.


# Phase 4B refactor: _act_hulu moved to core/actions.py.


# Phase 4B refactor: _act_max moved to core/actions.py.


# Phase 4B refactor: _act_spotify moved to core/actions.py.


# Phase 4B refactor: _act_youtube_play moved to core/actions.py.


# Phase 4F refactor: _act_play_streaming moved to core/actions.py.


# Phase 4A refactor (2026-05-29): _act_screenshot moved to core/actions.py.
# Imported via `from core.actions import *` higher up in this file.
# The ACTIONS dispatch dict still resolves it by name.


# Phase 4A refactor (2026-05-29): _act_get_time moved to core/actions.py.
# Imported via `from core.actions import *` higher up in this file.
# The ACTIONS dispatch dict still resolves it by name.


def get_camera_health() -> dict:
    """Snapshot of webcam I/O health for self-diagnostic / see_user.

    Returns a dict keyed by camera index with: ``last_frame_at`` (float
    epoch seconds; 0.0 if the device has never produced a frame),
    ``last_read_error`` (str or None), ``last_read_error_at`` (epoch or
    0.0), ``wake_attempts`` (int), ``recoveries`` (int). Safe to call from
    any thread — uses the same lock as the face-tracking writer.
    """
    out: dict[int, dict] = {}
    with _camera_state_lock:
        indices: set[int] = set()
        indices.update(_camera_last_frame_at.keys())
        indices.update(_camera_last_read_error.keys())
        indices.update(_camera_wake_attempts.keys())
        indices.update(_camera_recoveries.keys())
        for cam in CAMERAS:
            indices.add(cam["index"])
        for idx in indices:
            out[idx] = {
                "last_frame_at":       _camera_last_frame_at.get(idx, 0.0),
                "last_read_error":     _camera_last_read_error.get(idx),
                "last_read_error_at":  _camera_last_read_error_at.get(idx, 0.0),
                "wake_attempts":       _camera_wake_attempts.get(idx, 0),
                "recoveries":          _camera_recoveries.get(idx, 0),
            }
    return out


# Webcam awareness actions — let Bobert know where you are and look at you
# Phase 4H refactor: _act_where_is_user moved to core/actions.py.


# Phase 4I refactor: _act_see_user moved to core/actions.py.


# Phase 4I refactor: _act_which_monitor moved to core/actions.py.


# ──────────────────────────────────────────────────────────────────────────
#  iTUNES MUSIC PLAYBACK  — via Windows COM Dispatch
# ──────────────────────────────────────────────────────────────────────────

# iTunes COM search field enum
_ITUNES_SEARCH_ALL     = 0
_ITUNES_SEARCH_ARTISTS = 2
_ITUNES_SEARCH_ALBUMS  = 3
_ITUNES_SEARCH_SONGS   = 5


# iTunes COM helpers moved to audio/itunes_bridge.py. The bridge lazy-loads
# win32com / pythoncom only inside get_client(), so importing it (above) has
# no side effects at startup. _get_itunes() stays as a name in this module
# so the action handlers below (and any cached AST references) keep working.
def _itunes_is_running() -> bool:
    return _itunes_bridge.is_running()


def _get_itunes(wait_for_ready: bool = True, timeout: float = 12.0,
                force: bool = False):
    """Thin wrapper around audio.itunes_bridge.get_client(). All gating
    (Apple-Music-active short-circuit, is-running check, auto-launch
    permission) lives in the bridge — see its docstring."""
    return _itunes_bridge.get_client(
        wait_for_ready=wait_for_ready, timeout=timeout, force=force,
    )


# Apple Music browser-routing cache — Chrome/Edge only expose the foreground
# tab's title through Win32 GetWindowText, so an Apple Music tab that has
# been pushed to the background of a focused browser window would otherwise
# read as absent and the music actions would fall through to iTunes COM.
# When ANY live or background detection sees Apple Music, we stamp this
# cache; subsequent calls within _APPLE_MUSIC_SEEN_TTL_SECS treat Apple
# Music as still active and keep routing to browser media keys.
_APPLE_MUSIC_SEEN_TTL_SECS = 300
_apple_music_last_seen = [0.0]


def _note_apple_music_seen() -> None:
    """Mark Apple Music as having just been observed in a browser window.
    Called by the live title scan in _apple_music_chrome_active() and by
    apple_music_intel's listen loop whenever it parses an 'Apple Music'
    window title, keeping the browser-routing cache warm so play/pause/
    next/previous never accidentally hit iTunes COM."""
    _apple_music_last_seen[0] = time.time()


def _apple_music_chrome_active() -> bool:
    """Return True iff Apple Music is — or was recently — visible in a
    browser tab or Chrome PWA. iTunes desktop's window title is 'iTunes',
    not 'Apple Music', so this won't false-positive on the COM app.

    Used by the music actions below to short-circuit iTunes COM calls in
    favor of browser media keys / the streaming auto-play pipeline whenever
    the user's actual audio source is the Apple Music web app — otherwise
    a voice command like 'next track' would yank iTunes open and steal the
    audio session.

    Chrome/Edge only expose the *foreground* tab's title via Win32
    GetWindowText, so a live scan misses an Apple Music tab that's been
    pushed to the background of a focused browser window. To paper over
    that, a positive sighting from any source (this function's live scan
    or apple_music_intel's background poller) is cached for
    _APPLE_MUSIC_SEEN_TTL_SECS — long enough that switching tabs for a
    minute doesn't cause "next song" to suddenly start launching iTunes.
    """
    try:
        import pygetwindow as gw
        for w in gw.getAllWindows():
            title = (w.title or "").lower()
            if title and "apple music" in title:
                _note_apple_music_seen()
                return True
    except ImportError:
        pass
    except Exception:
        pass
    return (time.time() - _apple_music_last_seen[0]) < _APPLE_MUSIC_SEEN_TTL_SECS


# Now that the predicate exists, install it into the bridge so get_client()
# can short-circuit when the user's actual player is the Apple Music web app.
# force=True calls (explicit `library:` / play_unheard) still bypass this.
_itunes_bridge.set_apple_music_active_check(_apple_music_chrome_active)


# iTunes COM calls (LibraryPlaylist.Search / Play / NextTrack / CurrentTrack)
# have no timeout and run on the voice thread. A wedged / Not-Responding
# iTunes blocks the entire voice loop indefinitely. _run_itunes_com_timeout
# runs the COM work on a short-lived daemon worker with a join-timeout so a
# stuck iTunes degrades to a clean spoken line instead of freezing JARVIS.
#
# The worker MUST acquire its OWN iTunes handle inside `work` (call
# _get_itunes there) — a COM Application proxy obtained on the voice thread's
# STA apartment cannot be safely used from this worker thread. The worker
# does its own CoInitialize/CoUninitialize per thread, mirroring
# audio/itunes_bridge.py and get_mic_buffer's per-thread COM bookkeeping.
ITUNES_COM_TIMEOUT_SECONDS = 10.0


def _run_itunes_com_timeout(work, *, timeout: float = ITUNES_COM_TIMEOUT_SECONDS,
                            timeout_msg: str = "iTunes isn't responding, sir."):
    """Run `work` (a no-arg callable that does the iTunes COM calls and
    returns its result) on a daemon thread, returning its result. If the
    worker hasn't finished within `timeout` seconds (iTunes wedged), return
    `timeout_msg` and let the orphaned worker die with the process.

    `work` must build its own iTunes handle internally (via _get_itunes) so
    no STA proxy crosses the thread boundary.
    """
    holder: dict = {}

    def _runner() -> None:
        com_inited = False
        try:
            import pythoncom  # type: ignore
            pythoncom.CoInitialize()
            com_inited = True
        except Exception:
            pass
        try:
            holder["result"] = work()
        except Exception as e:  # noqa: BLE001 — surfaced to caller below
            holder["error"] = e
        finally:
            if com_inited:
                try:
                    import pythoncom  # type: ignore
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    t = threading.Thread(target=_runner, daemon=True, name="iTunesCOM")
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f"  [itunes] COM call exceeded {timeout:.0f}s — iTunes not responding")
        return timeout_msg
    if "error" in holder:
        raise holder["error"]
    return holder.get("result")


def _play_music_core(args: str, *, force: bool = False) -> tuple[bool, str]:
    """Implementation of play_music that returns (success, message).
    success=True means a track was actually started; False means the request
    couldn't be fulfilled (bad args, iTunes unreachable, no matches, COM
    error). Used directly by skills (e.g. apple_music_intel._act_play_vibe)
    that need to branch on success without parsing the message string.

    `force=True` bypasses the Apple-Music-in-browser short-circuit inside
    _get_itunes() so the explicit `library:` voice prefix still reaches
    iTunes COM even while Apple Music is open in Chrome.
    """
    if not args.strip():
        return False, "format: play_music, <song/artist/album name>"

    query = args.strip()
    field = _ITUNES_SEARCH_ALL
    m = re.match(r"^(artist|song|album|track):\s*(.+)$", query, re.IGNORECASE)
    if m:
        kind = m.group(1).lower()
        query = m.group(2).strip()
        field = {
            "artist": _ITUNES_SEARCH_ARTISTS,
            "song":   _ITUNES_SEARCH_SONGS,
            "track":  _ITUNES_SEARCH_SONGS,
            "album":  _ITUNES_SEARCH_ALBUMS,
        }[kind]

    # Acquire the iTunes handle AND run the COM calls inside the worker so a
    # wedged iTunes can't block the voice thread past the join-timeout. The
    # worker owns its own handle (no STA proxy crosses threads).
    def _work() -> tuple[bool, str]:
        app, err = _get_itunes(force=force)
        if app is None:
            return False, err
        try:
            library = app.LibraryPlaylist
            tracks = library.Search(query, field)
            count = tracks.Count if tracks else 0
            if count == 0:
                return False, f"no tracks found matching '{query}' in iTunes library"

            first = tracks.Item(1)   # iTunes COM uses 1-based indexing
            first.Play()

            name   = getattr(first, "Name",   "Unknown")
            artist = getattr(first, "Artist", "Unknown")
            more   = f" ({count} matches — queued)" if count > 1 else ""
            return True, f"playing '{name}' by {artist}{more}"
        except Exception as e:
            return False, f"iTunes playback failed: {e}"

    return _run_itunes_com_timeout(
        _work, timeout_msg=(False, "iTunes isn't responding, sir."))


# Phase 4H refactor: _act_play_music moved to core/actions.py.


# Phase 4D refactor: _act_pause_music moved to core/actions.py.


# Phase 4D refactor: _act_resume_music moved to core/actions.py.


# Phase 4E refactor: _act_next_song moved to core/actions.py.


# Phase 4E refactor: _act_previous_song moved to core/actions.py.


# ──────────────────────────────────────────────────────────────────────────
#  TASK QUEUE  — voice-captured ideas → markdown file for Claude Code
# ──────────────────────────────────────────────────────────────────────────

TODO_FILE             = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_todo.md")
UPGRADE_SUMMARY_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_upgrade_summary.json")
# Sidecar holding the signature (sha256 of the sorted TASK LIST only, excluding
# the timestamp) of the most recently SPOKEN upgrade announcement. Boot dedup
# reads this so a summary whose timestamp was bumped but whose task list is
# unchanged (the old overnight engine re-wrote the same tasks with a fresh
# "upgraded_at") is never re-announced as if it just ran.
UPGRADE_ANNOUNCED_SIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_upgrade_announced")


# Phase 4D refactor: _act_queue_task moved to core/actions.py.


# Phase 4E refactor: _act_show_tasks moved to core/actions.py.


# ──────────────────────────────────────────────────────────────────────────
#  SESSION RESUME  — "Welcome back, sir — when we left off you were working
#  on X" greeting fired automatically on a warm restart (previous session
#  ended within WARM_RESTART_WINDOW_SECONDS) AND exposed as an action so the
#  user can ask verbally "JARVIS, where did we leave off?"
#
#  Sources of "X":
#    1. Most recent unchecked task in jarvis_todo.md       (strongest signal)
#    2. Most recent user voice commands                    (last 3, fallback)
#    3. Last session summary (bobert_memory or session_summaries.json)
#
#  Tone — the spec asks for the bad_news / confirmation TTS presets to vary
#  with context, so the chosen phrasing intentionally embeds keywords that
#  detect_tts_emotion() picks up:
#    • Resume offer with a clear task → "At your service" → confirmation
#    • Nothing concrete to resume     → "I'm afraid…"     → bad_news
# ──────────────────────────────────────────────────────────────────────────

WARM_RESTART_WINDOW_SECONDS = 18 * 3600   # 18 hours
_session_resume_done: list[bool] = [False]   # only auto-greet once per process


def _last_session_end_ts() -> float:
    """Best-effort epoch seconds of the most-recent previous session's end.
    Returns 0.0 when no prior session is on record."""
    try:
        recent = pattern_memory.get_session_summaries("", limit=1)
        if recent:
            ts = recent[0].get("ts")
            if ts:
                try:
                    return float(ts)
                except Exception:
                    pass
            iso_end = recent[0].get("iso_end") or recent[0].get("iso_start") or ""
            if iso_end:
                try:
                    return time.mktime(time.strptime(iso_end, "%Y-%m-%dT%H:%M:%S"))
                except Exception:
                    pass
            date = recent[0].get("date") or ""
            if date:
                try:
                    # Legacy entries only have a date — treat as 20:00 local
                    # so the warm-restart window is computed sensibly.
                    return time.mktime(time.strptime(date + "T20:00:00",
                                                     "%Y-%m-%dT%H:%M:%S"))
                except Exception:
                    pass
    except Exception:
        pass
    return 0.0


def _last_n_user_commands(n: int = 3) -> list[str]:
    """Tail memory/voice_commands.jsonl for the last N accepted utterances.
    Returns newest-first; empty list if the log doesn't exist yet."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "memory", "voice_commands.jsonl")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    out: list[str] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        t = (obj.get("text") or "").strip()
        if t:
            out.append(t)
        if len(out) >= n:
            break
    return out


def _last_queued_task_line() -> str:
    """Return the raw line text of the most recently appended USER-facing
    `- [ ]` entry in jarvis_todo.md, or '' if none.

    Skips JARVIS's OWN auto-generated maintenance tasks (self-heal, anomaly,
    deep-audit, regression, self-diag, overnight). Those dominate the queue
    and were being surfaced in the 'Welcome back, sir — you were working on X'
    greeting as if the USER had been working on an internal exception-burst
    anomaly — confusing and wrong (caught live 2026-05-30 watching the
    session-resume greeting say 'working on … [anomaly] 24 unhandled exception
    traces')."""
    if not os.path.exists(TODO_FILE):
        return ""
    try:
        with open(TODO_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return ""
    matches = re.findall(r'^- \[ \].*$', content, flags=re.M)
    _INTERNAL_TASK = re.compile(
        r'\[\s*(anomaly|regression|self-?heal|self-?diag|deep-?audit|'
        r'overnight|auto|pipeline|diag)\b', re.IGNORECASE)
    for line in reversed(matches):
        if not _INTERNAL_TASK.search(line):
            return line
    return ""


def _summarise_task_line(line: str) -> str:
    """Strip the `- [ ] **YYYY-MM-DD HH:MM** [tag] — ` prefix and return a
    short JARVIS-readable phrase for the description portion. Splits on
    sentence-ending punctuation (period/colon followed by whitespace) so
    inline file paths like 'tray.py' don't truncate the summary."""
    s = line.strip()
    s = re.sub(r'^- \[ \]\s*', '', s)
    s = re.sub(r'^\*\*[^*]+\*\*\s*(?:\[[^\]]+\]\s*)?[—–-]+\s*', '', s)
    m = re.search(r'[.!?:]\s', s)
    first = (s[:m.start()] if m else s).strip().rstrip(",;:")
    if len(first) > 90:
        first = first[:87].rstrip() + "…"
    return first


def _build_session_resume(force: bool = False) -> tuple[str, dict]:
    """Compose the welcome-back greeting. Returns (text, details).

    `force=False` enforces the 18-hour warm-restart window — used by the
    auto-greeting hook at startup. `force=True` skips the window check —
    used by the `session_resume` action so the user can ask verbally
    even days later (the action will then explain the staleness)."""
    last_ts = _last_session_end_ts()
    age = (time.time() - last_ts) if last_ts > 0 else float("inf")

    details = {
        "last_session_ts": last_ts,
        "age_seconds":     age,
        "last_commands":   _last_n_user_commands(3),
        "next_task_line":  _last_queued_task_line(),
        "in_window":       0 < age <= WARM_RESTART_WINDOW_SECONDS,
    }

    if not force and not details["in_window"]:
        return ("", details)

    # Pick the strongest available "X" for "you were working on X".
    work = ""
    if details["next_task_line"]:
        work = _summarise_task_line(details["next_task_line"])
    if not work and details["last_commands"]:
        first_cmd = details["last_commands"][0].rstrip(".!?")
        if len(first_cmd) <= 90 and first_cmd:
            work = first_cmd.lower()

    last_summary = ""
    try:
        recent = pattern_memory.get_session_summaries("", limit=1)
        if recent:
            last_summary = (recent[0].get("summary") or "").strip()
    except Exception:
        pass
    if not work and last_summary:
        work = last_summary.split(".", 1)[0].strip()
        if len(work) > 90:
            work = work[:87].rstrip() + "…"
    details["work"]         = work
    details["last_summary"] = last_summary

    # When age is outside the warm-restart window but we're answering the
    # action verbally, lead with a candid "rather a while ago" note so the
    # context-staleness is acknowledged (and the bad_news preset fires).
    if force and not details["in_window"]:
        if last_ts <= 0:
            return ("I'm afraid I have no prior session on record, sir — "
                    "this appears to be a fresh start.", details)
        hrs = int(round(age / 3600.0))
        when = f"{hrs} hours" if hrs < 48 else f"{int(round(age / 86400.0))} days"
        if work:
            return (f"I'm afraid our last session was rather a while ago — "
                    f"about {when}, sir. When we left off you were working "
                    f"on {work}.", details)
        return (f"I'm afraid our last session was rather a while ago — "
                f"about {when}, sir. The thread has likely gone cold.",
                details)

    # Warm-restart greeting. Embeds "At your service" so the confirmation
    # preset fires when we have something concrete to resume.
    if work:
        return (
            f"Welcome back, sir. When we left off you were working on "
            f"{work} — shall I resume, or is there something else? "
            f"At your service.",
            details,
        )
    # Warm restart but no concrete X — embed "I'm afraid" so the bad_news
    # preset fires (slower + lower delivery) to acknowledge the gap.
    return (
        "Welcome back, sir. I'm afraid I have no clear recollection of "
        "where we left off — shall we pick something fresh?",
        details,
    )


def maybe_session_resume_greeting() -> str:
    """Startup hook: return the welcome-back string for a warm restart,
    or '' if outside the 18-hour window. Idempotent within a process."""
    if _session_resume_done[0]:
        return ""
    text, details = _build_session_resume(force=False)
    if text:
        _session_resume_done[0] = True
        try:
            age_h = details.get("age_seconds", 0.0) / 3600.0
            print(f"  [session_resume] warm-restart greeting "
                  f"(last session {age_h:.1f}h ago)")
        except Exception:
            pass
    return text


# Phase 4C refactor: _act_session_resume moved to core/actions.py.


# Phase 4I refactor: _act_session_memory_recall moved to core/actions.py.


# Phase 4C refactor: _act_clear_tasks moved to core/actions.py.


# Phase 4K refactor: _act_upgrade moved to core/actions.py.


# Phase 4C refactor: _act_restart moved to core/actions.py.


# ─── HUD visibility controls ───────────────────────────────────────────────
# Cheap show/hide via a `visible` field in hud_state.json — the HUD subprocess
# reads it once per tick and withdraws/deiconifies the window accordingly,
# so we never kill or relaunch the HUD process. Toggling is near-instant.

# Phase 4B refactor: _act_hide_hud moved to core/actions.py.


# Phase 4B refactor: _act_show_hud moved to core/actions.py.


# Phase 4B refactor: _act_toggle_hud moved to core/actions.py.


# Phase 4J refactor: _act_start_overnight_upgrade moved to core/actions.py.


# Phase 4K refactor: _act_shutdown_jarvis moved to core/actions.py.


def _check_and_arm_shutdown_prompt(text: str) -> bool:
    """If `text` matches a SHUTDOWN_TRIGGER_PHRASE, speak the overnight-protocol
    prompt and arm _shutdown_prompt_pending for SHUTDOWN_PROMPT_TIMEOUT_S seconds.
    Returns True when consumed (caller should skip normal routing for this turn).

    Match logic: substring match against the trimmed lowercase utterance —
    "JARVIS, shut down" / "shut down please" / "go offline now" all hit. Only
    fires when the utterance is short (≤ 6 words) to avoid catching casual
    mentions inside longer prose ("if I say shut down it should…").
    """
    if not text:
        return False
    tl = text.strip().lower()
    if not tl or len(tl.split()) > 6:
        return False
    if not any(p in tl for p in SHUTDOWN_TRIGGER_PHRASES):
        return False
    _shutdown_prompt_pending["armed"] = True
    _shutdown_prompt_pending["expires_at"] = time.time() + SHUTDOWN_PROMPT_TIMEOUT_S
    print(f"  [shutdown] prompt armed for {SHUTDOWN_PROMPT_TIMEOUT_S:.0f}s "
          f"(trigger: '{tl}')")
    _speak("Would you like to start the overnight protocol first, sir? Yes or no.")
    return True


def _handle_shutdown_prompt(text: str) -> bool:
    """If the shutdown prompt is armed AND not expired, classify `text` as
    yes / no / shutdown-reinforcement / unrelated and dispatch accordingly.
    Returns True when the message was consumed.

    Branches:
      - YES phrase → fire start_overnight_upgrade (overnight mode)
      - NO phrase  → fire _act_shutdown_jarvis (full shutdown)
      - Another SHUTDOWN_TRIGGER_PHRASE → user is insisting — full shutdown
      - Unrelated speech → speak "Shutdown cancelled." and clear the flag
      - Expired flag → silently clear and return False (normal LLM routing)
    """
    if not _shutdown_prompt_pending.get("armed"):
        return False
    # Expired — clear silently and let normal routing handle the utterance.
    if time.time() > _shutdown_prompt_pending.get("expires_at", 0.0):
        _shutdown_prompt_pending["armed"] = False
        print("  [shutdown] prompt expired — falling through to normal routing")
        return False
    tl = (text or "").strip().lower()
    # Clear the flag eagerly so a second-arming or a re-entrant call can't
    # re-trigger this branch. Each dispatch path below is terminal.
    _shutdown_prompt_pending["armed"] = False
    # Edge case: user repeats a shutdown phrase ('shut down... shut down').
    # Interpret as "yes, full shutdown — I'm insisting" rather than re-arming
    # the prompt and looping.
    if tl and any(p in tl for p in SHUTDOWN_TRIGGER_PHRASES):
        print(f"  [shutdown] reinforced shutdown via '{tl}' — firing full shutdown")
        try: _act_shutdown_jarvis()
        except Exception as _e:
            print(f"  [shutdown] reinforced shutdown failed: {_e}")
        return True
    # Check NO first so "no overnight" doesn't match YES's "overnight" substring.
    if tl in SHUTDOWN_PROMPT_NO_PHRASES or any(tl.startswith(p + " ") for p in SHUTDOWN_PROMPT_NO_PHRASES):
        print(f"  [shutdown] user declined overnight ('{tl}') — full shutdown")
        try: _act_shutdown_jarvis()
        except Exception as _e:
            print(f"  [shutdown] _act_shutdown_jarvis failed: {_e}")
        return True
    if tl in SHUTDOWN_PROMPT_YES_PHRASES or any(tl.startswith(p + " ") for p in SHUTDOWN_PROMPT_YES_PHRASES):
        print(f"  [shutdown] user confirmed overnight ('{tl}') — start_overnight_upgrade")
        try: _act_start_overnight_upgrade()
        except Exception as _e:
            print(f"  [shutdown] _act_start_overnight_upgrade failed: {_e}")
        return True
    # Unrelated speech — cancel the prompt and let the original utterance
    # fall through to normal LLM routing so the user isn't stuck waiting.
    print(f"  [shutdown] unrelated reply ('{tl}') — cancelling prompt, "
          f"falling through to normal routing")
    try: _speak("Shutdown cancelled.")
    except Exception: pass
    return False


# Phase 4D refactor: _act_now_playing moved to core/actions.py.


def _find_windows_by_title(query: str) -> list:
    """Return all open windows whose title contains query (case-insensitive)."""
    try:
        import pygetwindow as gw
    except ImportError:
        return []
    q = query.lower().strip()
    out = []
    for w in gw.getAllWindows():
        if w.title and q in w.title.lower():
            out.append(w)
    return out


# Phase 4D refactor: _act_list_windows moved to core/actions.py.


def _find_music_window():
    """Return the highest-priority open window likely to host a music session.

    Scans every visible window title for one of MUSIC_WINDOW_HINTS and returns
    the first match in priority order, or None. Used by the media-key actions
    so the keypress lands on the right window (Chrome won't honor VK_MEDIA_*
    on a background tab).
    """
    try:
        import pygetwindow as gw
    except ImportError:
        return None
    titles = [(w, (w.title or "").lower()) for w in gw.getAllWindows() if w.title]
    for hint in MUSIC_WINDOW_HINTS:
        for w, t in titles:
            if hint in t:
                return w
    return None


def _focus_music_window() -> str | None:
    """Bring a likely music-host window to the foreground.

    Returns the focused window title on success, or None if no candidate was
    found or focusing failed. Callers should sleep briefly (~120ms) after a
    successful focus so the foreground change settles before sending input.
    """
    target = _find_music_window()
    if target is None:
        return None
    try:
        target.activate()
        return target.title
    except Exception as e:
        msg = str(e).lower()
        if "operation completed successfully" in msg or "error code from windows: 0" in msg:
            return target.title
        try:
            target.minimize()
            target.restore()
            return target.title
        except Exception:
            return None


def _flash_window_reticle(win, label: str = "focus"):
    """Flash a reticle at the geometric center of a pygetwindow object.
    Best-effort: silently skips if the window doesn't expose its geometry
    (some Win32 windows raise on attribute access immediately after activate)."""
    try:
        cx = int(win.left + win.width / 2)
        cy = int(win.top + win.height / 2)
        title_snippet = (win.title or "").split(" — ")[0].split(" - ")[0][:20]
        _publish_reticle(cx, cy, f"{label}: {title_snippet}" if title_snippet else label)
    except Exception:
        pass


# Phase 4D refactor: _act_focus_window moved to core/actions.py.


# Phase 4D refactor: _act_minimize_window moved to core/actions.py.


# Phase 4D refactor: _act_close_window moved to core/actions.py.


# Phase 4J refactor: _act_open_on_monitor moved to core/actions.py.


# Phase 4J refactor: _act_move_window_to_monitor moved to core/actions.py.


# Screen vision actions
def _parse_monitor_prefix(args: str) -> tuple[str | None, str]:
    """If args starts with 'monitor:<name>|', strip it and return (monitor, rest).
    Otherwise return (None, args)."""
    m = re.match(r"^\s*monitor:([\w-]+)\s*\|\s*(.*)$", args, re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2)
    return None, args


# ── Screen context cache ─────────────────────────────────────────────────
# Short-term visual memory: the last few see_screen captures with their
# answers, so JARVIS can reference what he saw a moment ago without
# re-capturing. recall_screen reads from this; see_screen writes to it.
SCREEN_CACHE_MAX_ENTRIES = 5
SCREEN_CACHE_TTL_SECONDS = 300   # 5 minutes per spec
_screen_cache: list[dict] = []
_screen_cache_lock = threading.Lock()

# Per-intent see_screen budget. Caps how many vision captures a single
# parse_and_run_actions dispatch may burn before _act_see_screen starts
# refusing and pointing the LLM at recall_screen / cached context. Stops
# the "sledgehammer" loop where JARVIS fires see_screen → scroll →
# find_on_screen → see_screen … for minutes without finding the target.
# Counter is per-thread (threading.local) so concurrent dispatches don't
# collide, and is reset at the top of parse_and_run_actions so it never
# leaks across unrelated user requests.
SEE_SCREEN_BUDGET_PER_INTENT = 3
_see_screen_budget_state = threading.local()


def _push_screen_context(
    monitor: str | None,
    question: str,
    answer: str,
    images: dict[str, bytes],
) -> None:
    """Append a see_screen result to the cache (capped at SCREEN_CACHE_MAX_ENTRIES).
    Stores the PNG bytes too so recall_screen can re-ask vision against the
    same visual state without a fresh capture."""
    with _screen_cache_lock:
        _screen_cache.append({
            "ts": time.time(),
            "monitor": monitor,            # None = all monitors
            "question": question,
            "answer": answer,
            "images": dict(images),         # shallow copy — bytes are immutable
        })
        while len(_screen_cache) > SCREEN_CACHE_MAX_ENTRIES:
            _screen_cache.pop(0)


def _recent_screen_contexts(max_age: float = SCREEN_CACHE_TTL_SECONDS) -> list[dict]:
    """Return cached entries newer than max_age, newest first."""
    cutoff = time.time() - max_age
    with _screen_cache_lock:
        return [e for e in reversed(_screen_cache) if e["ts"] >= cutoff]


def _format_screen_age(seconds: float) -> str:
    if seconds < 5:
        return "just now"
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    if seconds < 3600:
        m = int(seconds // 60)
        return f"{m} minute{'s' if m != 1 else ''} ago"
    h = int(seconds // 3600)
    return f"{h} hour{'s' if h != 1 else ''} ago"


# ── Glance-response mode ─────────────────────────────────────────────────
# When the focused window has just changed AND the user asks an ambiguous
# question ("what is this?", "should I worry?", "wait, what?", "explain"),
# auto-attach a screenshot of just the focused window to vision and reply
# in ONE sentence — cuts the "JARVIS, look at my screen and tell me…"
# friction down to just "JARVIS, what?".

GLANCE_WINDOW_CHANGE_TTL_SECONDS = 5.0

# Whole-utterance triggers. Matched after lowercasing and stripping common
# end-punctuation, and only when the utterance is short (≤ 5 words) so a
# mid-sentence "explain" inside a longer prompt doesn't accidentally fire.
GLANCE_AMBIGUOUS_PATTERNS = (
    "what",
    "what?",
    "huh",
    "huh?",
    "wait",
    "wait what",
    "wait, what",
    "what is this",
    "what's this",
    "whats this",
    "what is that",
    "what's that",
    "whats that",
    "what was that",
    "what now",
    "what's going on",
    "whats going on",
    "what's happening",
    "whats happening",
    "should i worry",
    "should i be worried",
    "should i be concerned",
    "is that bad",
    "is this bad",
    "explain",
    "explain this",
    "explain that",
)

_focused_window_state: dict = {
    "hwnd": None,
    "title": "",
    "rect": None,             # (left, top, width, height) virtual-desktop coords
    "changed_at": float("-inf"),
}
_focus_tracker_stop = threading.Event()


def _read_focused_window() -> tuple[int | None, str, tuple[int, int, int, int] | None]:
    """Return (hwnd, title, rect) for the currently focused top-level window,
    where rect is (left, top, width, height) in virtual-desktop coords, or
    None if the window is minimised / unobtainable."""
    try:
        import win32gui
    except Exception:
        return None, "", None
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None, "", None
        title = win32gui.GetWindowText(hwnd) or ""
        try:
            left, top, right, bot = win32gui.GetWindowRect(hwnd)
            w = right - left
            h = bot - top
            rect = (left, top, w, h) if (w > 1 and h > 1) else None
        except Exception:
            rect = None
        return hwnd, title, rect
    except Exception:
        return None, "", None


def _focus_tracker_loop() -> None:
    while not _focus_tracker_stop.is_set():
        try:
            hwnd, title, rect = _read_focused_window()
            if hwnd is not None:
                prev = _focused_window_state.get("hwnd")
                if hwnd != prev:
                    _focused_window_state["changed_at"] = time.monotonic()
                _focused_window_state["hwnd"] = hwnd
                _focused_window_state["title"] = title
                if rect is not None:
                    _focused_window_state["rect"] = rect
        except Exception:
            pass
        if _focus_tracker_stop.wait(1.0):
            break


def _start_focus_tracker() -> None:
    """Launch the background focus tracker. Safe to call once at startup."""
    t = threading.Thread(target=_focus_tracker_loop, daemon=True,
                         name="focus-tracker")
    t.start()


def _focus_changed_recently() -> bool:
    """True if the foreground window changed within the last
    GLANCE_WINDOW_CHANGE_TTL_SECONDS. Also performs a live read so a focus
    switch that happened in the < 1 s gap before the tracker's next tick
    still registers as 'just changed'."""
    try:
        hwnd, title, rect = _read_focused_window()
    except Exception:
        hwnd, title, rect = None, "", None
    prev = _focused_window_state.get("hwnd")
    if hwnd is not None and prev is not None and hwnd != prev:
        # Live read picked up a change the tracker hasn't seen yet
        _focused_window_state["hwnd"] = hwnd
        _focused_window_state["title"] = title
        if rect is not None:
            _focused_window_state["rect"] = rect
        _focused_window_state["changed_at"] = time.monotonic()
        return True
    changed_at = _focused_window_state.get("changed_at", float("-inf"))
    age = time.monotonic() - changed_at
    return 0 <= age <= GLANCE_WINDOW_CHANGE_TTL_SECONDS


def _is_glance_ambiguous_question(text: str) -> bool:
    """True if `text` is one of the short, ambiguous utterances that almost
    always refers to whatever just appeared on screen rather than something
    in the conversation history."""
    s = (text or "").strip().lower()
    # Strip trailing punctuation we don't care about
    s = s.rstrip(".!,;:")
    if not s:
        return False
    if len(s.split()) > 5:
        return False
    if s in GLANCE_AMBIGUOUS_PATTERNS:
        return True
    # Also accept patterns followed by a trailing '?'
    if s.endswith("?") and s[:-1].strip() in GLANCE_AMBIGUOUS_PATTERNS:
        return True
    return False


def _capture_focused_window_png() -> bytes | None:
    """PNG bytes of just the currently-focused window's region, downscaled
    to vision-friendly dimensions. Returns None if the rect can't be
    obtained or the capture fails."""
    rect = _focused_window_state.get("rect")
    if not rect:
        _, _, rect = _read_focused_window()
    if not rect:
        return None
    left, top, w, h = rect
    if w < 50 or h < 50:
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        import mss
        _MSSClass = getattr(mss, "MSS", mss.mss)
        with _MSSClass() as sct:
            region = {"left": int(left), "top": int(top),
                      "width": int(w), "height": int(h)}
            raw = sct.grab(region)
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    except Exception as e_mss:
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab(
                bbox=(int(left), int(top), int(left + w), int(top + h)),
                all_screens=True,
            )
        except Exception as e_pil:
            print(f"  [glance] capture failed (mss: {e_mss}, pil: {e_pil})")
            return None
    try:
        max_dim = 1568
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize(
                (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception as e:
        print(f"  [glance] encode failed: {e}")
        return None


def maybe_glance_response(user_text: str) -> str | None:
    """Fast-path dispatcher entry. If the user just said something ambiguous
    AND the focused window changed in the last few seconds, capture just
    that window, ask vision for a single-sentence answer, and return it.
    Otherwise return None and let the caller fall through to the normal
    LLM path.

    On success this also:
      • appends the user/assistant turn to conversation_history (mirroring
        what _call_llm would have done) so future turns see the exchange;
      • pushes the capture into the see_screen cache so a follow-up
        'recall_screen' question still works.
    """
    if not _is_glance_ambiguous_question(user_text):
        return None
    if not _focus_changed_recently():
        return None
    if AI_BACKEND != "claude" or not SCREEN_VISION_ENABLED:
        return None
    png = _capture_focused_window_png()
    if png is None:
        return None

    title = _focused_window_state.get("title") or "the focused window"
    print(f"  [glance] focus recently changed → '{title}' — one-shot vision")

    # Run the thinking-eye animation while vision works.
    pause_face_tracking()
    set_state("thinking")
    stop_evt = threading.Event()
    anim = threading.Thread(target=_thinking_loop, args=(stop_evt,), daemon=True)
    anim.start()
    q = (
        f"The user just glanced at this window (title: '{title}') and asked: "
        f"\"{user_text.strip()}\".\n\n"
        f"Reply in ONE concise sentence as JARVIS — dry, helpful, in "
        f"character. If something on screen looks wrong, lead with "
        f"'I'm afraid'. No bullet points, no over-explaining."
    )
    # Run ask_vision on a worker thread with a bounded join so a slow vision
    # backend can't stall the main voice loop. On Claude failure ask_vision
    # falls back to _call_local_vision, whose Ollama POST uses timeout=180 —
    # blocking the loop that long would trip the watchdog. Cap at 12s; on
    # timeout return None so the normal LLM path proceeds. The worker is
    # daemon so a still-running call can't keep the process alive.
    _vis_result: list = [None]
    def _vis_worker():
        try:
            _vis_result[0] = ask_vision(q, png)
        except Exception:
            logging.exception("[glance] ask_vision worker raised")
    _vis_thread = threading.Thread(target=_vis_worker, daemon=True)
    try:
        _vis_thread.start()
        _vis_thread.join(timeout=12.0)
    finally:
        stop_evt.set()
        anim.join()

    if _vis_thread.is_alive():
        print("  [glance] vision timed out (>12s) — falling through to LLM")
        return None
    answer = _vis_result[0]

    if not answer or answer.startswith("("):
        return None

    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": answer})
    _trim_conversation_history()
    try:
        _push_screen_context(None, q, answer, {"focused": png})
    except Exception:
        pass
    return answer


# Phase 4H refactor: _act_see_screen moved to core/actions.py.


# Phase 4I refactor: _act_recall_screen moved to core/actions.py.


# Phase 4C refactor: _act_find_on_screen moved to core/actions.py.


# Self-preservation: titles/keywords Bobert must NEVER close or click on.
# Closing his own host PowerShell kills the Python process he's running in.
FORBIDDEN_TARGETS = [
    "powershell", "windows powershell", "pwsh", "cmd", "command prompt",
    "python", "bobert", "terminal", "windows terminal",
]
_CLOSE_VERBS = ("close", "exit", "quit", "kill", "terminate", "x button", "shut down")


def _is_self_close_attempt(description: str) -> bool:
    """Return True if the click description is targeting Bobert's own host
    process (e.g. 'close button on Windows PowerShell')."""
    d = description.lower()
    if not any(verb in d for verb in _CLOSE_VERBS):
        return False
    return any(target in d for target in FORBIDDEN_TARGETS)


# UI automation actions
# Phase 4F refactor: _act_click moved to core/actions.py.


# Aliases for keys Bobert might say but pyautogui doesn't recognize on Windows
_KEY_ALIASES = {
    "super": "win", "cmd": "win", "meta": "win", "command": "win",
    "windows": "win", "winkey": "win",
    "control": "ctrl",
    "option": "alt", "opt": "alt",
    "return": "enter",
    "escape": "esc",
}

def _normalize_key(k: str) -> str:
    return _KEY_ALIASES.get(k.strip().lower(), k.strip().lower())


# Phase 4F refactor: _act_hotkey moved to core/actions.py.


# Recognisable shell verbs at the start of a line — used to detect when the LLM
# is trying to "run" a command by typing it into whatever happens to have focus.
# PowerShell uses Verb-Noun cmdlets (Get-*, Set-*, New-*, etc.), unix-y tools
# all start with their name, and common dev tools (python/pip/git/npm/...) are
# called by name. If text begins with one of these AND no terminal is focused,
# we refuse the type and suggest run_shell instead.
_SHELL_CMDLET_VERBS = (
    "Get-", "Set-", "New-", "Remove-", "Invoke-", "Start-", "Stop-", "Restart-",
    "Test-", "Add-", "Out-", "Where-", "Select-", "ForEach-", "Import-", "Export-",
    "Read-", "Write-", "Clear-", "Copy-", "Move-", "Rename-", "Format-", "Measure-",
    "Convert-", "ConvertFrom-", "ConvertTo-", "Resolve-", "Push-", "Pop-",
    "Enable-", "Disable-", "Install-", "Uninstall-", "Update-", "Find-",
)
_SHELL_CMD_NAMES = {
    # PowerShell-ish
    "ls", "dir", "cd", "pwd", "echo", "type", "cls", "del", "ren", "md", "rd",
    # Unix-style commonly aliased on Windows
    "rm", "cp", "mv", "cat", "grep", "awk", "sed", "head", "tail", "less",
    "more", "touch", "mkdir", "rmdir", "chmod", "chown", "which",
    # Dev tooling
    "python", "python3", "py", "pip", "pip3", "pipx", "poetry", "uv",
    "node", "npm", "npx", "yarn", "pnpm", "bun", "deno",
    "git", "gh", "hg", "svn",
    "docker", "kubectl", "helm", "minikube",
    "make", "cmake", "ninja", "cargo", "go", "rustc", "rustup",
    "javac", "java", "gradle", "mvn", "dotnet", "msbuild",
    "ruby", "gem", "bundle", "rails", "rake",
    "ssh", "scp", "rsync", "curl", "wget",
    "pytest", "tox", "pyenv", "conda", "mamba",
    # Windows shell helpers
    "start", "powershell", "pwsh", "cmd", "wt", "robocopy", "xcopy", "tasklist",
    "taskkill", "ipconfig", "netstat", "ping", "tracert",
}

# Window-title fragments that indicate a terminal is currently focused. If one
# of these is in the foreground window's title, typing a shell command is the
# user's own keystrokes going to the terminal — fine to pass through.
_TERMINAL_TITLE_FRAGMENTS = (
    "powershell", "windows powershell", "pwsh",
    "command prompt", "cmd.exe",
    "windows terminal",
    "git bash", "mingw", "msys",
    "wsl", "ubuntu", "debian", "kali", "alpine",
    "hyper", "tabby", "conemu", "cmder", "alacritty", "wezterm", "kitty",
    "terminal",
)


def _active_window_is_terminal() -> bool:
    """Return True if the currently-focused window appears to be a terminal."""
    try:
        import pygetwindow as gw
        active = gw.getActiveWindow()
        if not active or not active.title:
            return False
        t = active.title.lower()
        return any(frag in t for frag in _TERMINAL_TITLE_FRAGMENTS)
    except Exception:
        return False


def _looks_like_shell_command(text: str) -> bool:
    """Heuristic: does this text look like a shell command (vs. prose)?

    True if the first non-whitespace token matches a known shell command name
    or starts with a PowerShell Verb-Noun prefix (Get-, Set-, etc.).
    """
    stripped = text.lstrip()
    if not stripped:
        return False
    # PowerShell env-var assignment / call — e.g. "$env:PATH" or "& 'C:\..'"
    if stripped.startswith(("$env:", "& '", '& "', "& \\", "& C:", "& D:")):
        return True
    first = stripped.split(None, 1)[0]
    if first.startswith(_SHELL_CMDLET_VERBS):
        return True
    if first.lower() in _SHELL_CMD_NAMES:
        return True
    return False


# Phase 4D refactor: _act_type moved to core/actions.py.


# Commands that must never run via run_shell even if the LLM tries. These would
# either destroy data, take down the machine, or kill JARVIS itself.
_SHELL_FORBIDDEN_PATTERNS = (
    "format ",                     # format C: etc.
    "shutdown",                    # shutdown /s
    # rm -rf variants — substring match, so "rm -rf /" also catches
    # "rm -rf /home", "rm -rf /usr", etc.
    "rm -rf /",                    # nuke from orbit (and /home, /usr, …)
    "rm -rf ~",                    # home dir via tilde (incl. ~/, ~/x)
    "rm -rf $",                    # env-var expansion: $HOME, $USER, $env:…
    "rm -rf %",                    # cmd-style env vars: %USERPROFILE%, %HOME%
    "rm -rf *",                    # wildcard everything in cwd
    "rm -rf .",                    # current dir (catches "rm -rf ." and "..")
    "rm -rf c:",
    "rm -rf d:",
    "sudo rm ",                    # any privilege-escalated rm
    "sudo dd ",                    # dd as root
    "remove-item -recurse -force c:",
    "remove-item -recurse -force d:",
    "remove-item -recurse -force ~",
    "remove-item -recurse -force $home",
    "remove-item -recurse -force $env:",
    "del /f /s /q c:",
    "del /f /s /q d:",
    "del /f /s /q %",              # %USERPROFILE% etc.
    "diskpart",
    "cipher /w",                   # secure-wipe a drive
    "mkfs",                        # format a filesystem
    "dd if=",                      # disk destroyer
    ":(){ :|:& };:",               # fork bomb
    "taskkill /f /im python",      # kills our own host
    "taskkill /f /im pwsh",
    "taskkill /f /im powershell",
    "stop-process -name python",
    "stop-process -name pwsh",
    "stop-process -name powershell",
)

# Cap how long a single run_shell call can run before we kill it.
RUN_SHELL_TIMEOUT_SEC = 30
# Cap the size of stdout/stderr we return to the LLM (very large dumps would
# blow the context budget).
RUN_SHELL_OUTPUT_MAX_CHARS = 4000


# Phase 4H refactor: _act_run_shell moved to core/actions.py.


# Phase 4C refactor: _act_press moved to core/actions.py.


# Phase 4C refactor: _act_scroll moved to core/actions.py.


# Media keys — work for Apple Music in browser, Spotify, YouTube, any media app.
# Use these for streaming services that don't have a COM interface.
#
# Chrome (where Apple Music lives) does NOT honor VK_MEDIA_* on a background
# tab — the key has to land while a music window is foregrounded. Native apps
# like Spotify/iTunes register a system-wide media-key hook and respond either
# way, but focusing them costs nothing. So: focus a music window first, send
# the key, and if no music window exists at all, fall back to clicking a
# visible Next/Prev/Play button via vision.
def _media_key_with_focus(vk: str, vision_hint: str, label: str) -> str:
    pag = _get_pyautogui()
    if not pag:
        return "pyautogui unavailable"
    focused = _focus_music_window()
    if focused:
        time.sleep(0.12)
        try:
            _ui_safe(pag, pag.press, vk)
        except UIFailsafeError as _e:
            return str(_e)
        return f"{label} (focused {focused!r})"
    # No music host window — try the global keypress anyway (Spotify desktop,
    # iTunes desktop, and any app with a system-wide hook will pick it up).
    try:
        _ui_safe(pag, pag.press, vk)
    except UIFailsafeError as _e:
        return str(_e)
    # Vision fallback: if nothing seems to be hosting music, look for an
    # on-screen control. Cheap because we only do this when there's no
    # focusable music window.
    try:
        coords = find_click_target(vision_hint)
    except Exception:
        coords = None
    if coords:
        try:
            ui_click(coords[0], coords[1])
            return f"{label} (no music window found — clicked on-screen button at {coords})"
        except UIFailsafeError as e:
            return f"{label} (no music window found; on-screen button click blocked: {e})"
        except Exception as e:
            return f"{label} (no music window found; on-screen button click failed: {e})"
    return f"{label} (no music window found and no on-screen control visible — key sent globally)"


# Phase 4A refactor (2026-05-29): _act_media_next moved to core/actions.py.
# Imported via `from core.actions import *` higher up in this file.
# The ACTIONS dispatch dict still resolves it by name.

# Phase 4A refactor (2026-05-29): _act_media_prev moved to core/actions.py.
# Imported via `from core.actions import *` higher up in this file.
# The ACTIONS dispatch dict still resolves it by name.

# Phase 4A refactor (2026-05-29): _act_media_playpause moved to core/actions.py.
# Imported via `from core.actions import *` higher up in this file.
# The ACTIONS dispatch dict still resolves it by name.

# Phase 4A refactor (2026-05-29): _act_volume_up moved to core/actions.py.
# Imported via `from core.actions import *` higher up in this file.
# The ACTIONS dispatch dict still resolves it by name.

# Phase 4A refactor (2026-05-29): _act_volume_down moved to core/actions.py.
# Imported via `from core.actions import *` higher up in this file.
# The ACTIONS dispatch dict still resolves it by name.

# Phase 4A refactor (2026-05-29): _act_volume_mute moved to core/actions.py.
# Imported via `from core.actions import *` higher up in this file.
# The ACTIONS dispatch dict still resolves it by name.


def _substitute_monitor_in_arg(action_name: str, arg: str, monitor: str) -> str:
    """Return `arg` rewritten so the named action targets `monitor` instead.

    Two known monitor-arg shapes:
      open_on_monitor:        '<monitor> | <url-or-app>'   (monitor first)
      move_window_to_monitor: '<title>   | <monitor>'      (monitor last)
    For any other action we leave `arg` alone — the replay handler will
    still fire, just on its original monitor.
    """
    mon = (monitor or "").strip().lower()
    if not mon or not arg or "|" not in arg:
        return arg
    if action_name == "open_on_monitor":
        _, rest = arg.split("|", 1)
        return f"{mon} | {rest.strip()}"
    if action_name == "move_window_to_monitor":
        head, _ = arg.split("|", 1)
        return f"{head.strip()} | {mon}"
    return arg


# Phase 4H refactor: _act_replay_last_action moved to core/actions.py.


# Voice phrases that route directly to replay_last_action without an LLM
# round-trip. The optional monitor capture matches names ('left', 'right',
# 'top', 'middle') and bare digits ('2', '3') so users can say either.
_REPLAY_VOICE_RE = re.compile(
    r"^(?:please\s+)?"
    r"(?:do|run|fire|execute|repeat|replay)\s+"
    r"(?:that|it|the\s+(?:last|previous)(?:\s+(?:thing|action|one|step))?)"
    r"(?:\s+(?:one\s+more\s+time|again|once\s+more))?"
    r"(?:\s+on\s+(?:the\s+)?(?:monitor\s+)?(\w+)(?:\s+monitor)?)?"
    r"[.!?]*\s*$",
    re.IGNORECASE,
)


def maybe_replay_last_action(utterance: str) -> str | None:
    """Detect a replay voice trigger and dispatch it.

    Returns the spoken confirmation if the utterance matched and the replay
    fired (or refused), else None so the caller falls through to its normal
    dispatch path.
    """
    if not utterance or not utterance.strip():
        return None
    m = _REPLAY_VOICE_RE.match(utterance.strip())
    if not m:
        return None
    monitor = (m.group(1) or "").strip()
    return _act_replay_last_action(monitor)


# Phase 4I refactor: _act_read_changelog moved to core/actions.py.


# Phase 4G refactor: _act_version_info moved to core/actions.py.


# ──────────────────────────────────────────────────────────────────────────
#  Tray-only action handlers — register at the bottom of ACTIONS.
#
#  These back the Power Tools / AI / Memory / Diagnostics submenus in
#  tray.py. The tray dispatcher (_dispatch_tray_command) runs on the
#  2 Hz drainer thread, so anything that could block for more than ~100ms
#  is spawned into a daemon and prints its result asynchronously instead
#  of returning it. Each handler still returns a short status string so
#  the dispatcher's one-line log entry is informative.
# ──────────────────────────────────────────────────────────────────────────

# Tray commands whose ACTIONS handler may run synchronously for tens of
# seconds. Skills can overwrite ACTIONS entries at load time with sync
# implementations (e.g. skills/self_diagnostic.py replaces our async
# `_act_run_diagnostic_tray` with a blocking `run_diagnostic`), so the
# generic dispatcher in _dispatch_tray_command routes any cmd in this
# set through _tray_async to keep the 2 Hz drainer from stalling.
_HEAVY_ACTIONS = {"run_diagnostic", "test_each_skill", "latency_benchmark"}


def _tray_async(name: str, fn) -> None:
    """Run fn() in a daemon thread and print a single-line tray log entry
    when it returns. Used by every tray handler whose probe/wait could
    stall the drainer."""
    def _wrap():
        t0 = time.time()
        try:
            out = fn() or ""
            head = str(out).split("\n", 1)[0] if out else "done"
            if len(head) > 120:
                head = head[:117].rstrip() + "..."
            print(f"  [tray-async] {name} -> {head}  ({(time.time()-t0)*1000:.0f}ms)")
        except Exception as exc:
            print(f"  [tray-async] {name} raised: {exc}")
    threading.Thread(target=_wrap, name=f"tray-{name}", daemon=True).start()


# Phase 4F refactor: _act_stop_pipeline moved to core/actions.py.


# Phase 4E refactor: _act_ambient_mode_set moved to core/actions.py.


# Phase 4B refactor: _act_ambient_mode_toggle moved to core/actions.py.


# Phase 4F refactor: _act_force_backup moved to core/actions.py.


# Phase 4E refactor: _act_reload_skills moved to core/actions.py.


# Phase 4G refactor: _act_run_smoke_test moved to core/actions.py.


# ── LLM control ──────────────────────────────────────────────────────────
#
# AI_BACKEND is a module-level string ("claude" | "ollama"). Switching
# rebinds the global so subsequent _llm_quick() calls see the new value.
# We don't persist across restarts — that requires editing the source,
# which the user can do from "Switch LLM > Other" in the tray once the
# picker dialog is implemented. For now we accept the runtime-only flip.

_KNOWN_OLLAMA_MODELS = {
    "llama3", "llama3.1:8b", "llama3.1:70b", "llama3.2",
    "qwen2.5:14b", "qwen2.5:7b", "qwen2.5:32b",
    "mistral", "mistral:7b", "mixtral", "phi3", "gemma2",
    "codellama", "deepseek-r1:8b", "deepseek-r1:14b",
}


# Phase 4K refactor: _act_switch_llm moved to core/actions.py.


# Phase 4C refactor: _act_switch_llm_picker moved to core/actions.py.


# Phase 4C refactor: _act_show_llm_stats moved to core/actions.py.


# Phase 4B refactor: _act_clear_llm_cache moved to core/actions.py.


# reset_llm_cache is a tray alias for clear_llm_cache — registered below.


# ── Memory ───────────────────────────────────────────────────────────────

# Phase 4E refactor: _act_show_recent_facts moved to core/actions.py.


# Phase 4F refactor: _act_reset_memory moved to core/actions.py.


# Phase 4E refactor: _act_export_memory moved to core/actions.py.


# Phase 4G refactor: _act_forget_last_hour moved to core/actions.py.


# ── Diagnostics ──────────────────────────────────────────────────────────
#
# self_diagnostic.py registers its own action names ('run_diagnostic',
# 'last_diagnostic_run'). The tray uses different verbs
# ('run_diagnostic' matches, 'show_last_diagnostic' does NOT) and adds
# four single-component probes ('test_mic' / 'test_tts' / 'test_vision'
# / 'test_each_skill') plus a latency benchmark. We wrap the self_diag
# helpers when available and fall back to a clear status string when
# the skill failed to load.

def _selfdiag_module():
    """Return the loaded self_diagnostic skill module, or None."""
    return sys.modules.get("skill_self_diagnostic")


# Phase 4E refactor: _act_run_diagnostic_tray moved to core/actions.py.


# Phase 4E refactor: _act_show_last_diagnostic moved to core/actions.py.


def _probe_via_selfdiag(name: str, probe_attr: str) -> str:
    """Shared body for test_mic / test_tts / test_vision. Spawns the
    component probe from skills/self_diagnostic.py in a daemon, using
    the skill's own _run_with_timeout helper so a hung mic/camera can't
    leak the worker thread past the per-probe budget."""
    sd = _selfdiag_module()
    if sd is None or not hasattr(sd, probe_attr):
        return f"{name} probe unavailable (self_diagnostic not loaded)"
    probe = getattr(sd, probe_attr)
    runner = getattr(sd, "_run_with_timeout", None)
    budget = getattr(sd, "PER_PROBE_TIMEOUT_S", 15.0)
    def _do():
        try:
            if runner is not None:
                r = runner(probe, float(budget), name=name)
            else:
                r = probe()
            ok = bool(r.get("ok"))
            err = r.get("error") or ""
            latency = r.get("latency_ms")
            tag = "OK" if ok else f"FAIL ({err})"
            if latency:
                tag += f" — {latency:.0f}ms"
            return f"{name}: {tag}"
        except Exception as e:
            return f"{name}: probe raised {e}"
    _tray_async(f"test_{name}", _do)
    return f"{name} probe running"


# Phase 4B refactor: _act_test_mic moved to core/actions.py.


# Phase 4B refactor: _act_test_tts moved to core/actions.py.


# Phase 4B refactor: _act_test_vision moved to core/actions.py.


# Phase 4G refactor: _act_test_each_skill moved to core/actions.py.


# Phase 4G refactor: _act_latency_benchmark moved to core/actions.py.


# Whitelist of actions Bobert is allowed to perform
ACTIONS = {
    "open_url":        _act_open_url,
    "launch_app":      _act_launch_app,
    "web_search":      _act_web_search,
    "search":          _act_web_search,
    "youtube":         _act_youtube,
    "screenshot":      _act_screenshot,
    "get_time":        _act_get_time,
    # Screen vision
    "see_screen":      _act_see_screen,
    "recall_screen":   _act_recall_screen,
    "last_screen":     _act_recall_screen,
    "previous_screen": _act_recall_screen,
    "screen_history":  _act_recall_screen,
    "find_on_screen":  _act_find_on_screen,
    # Webcam awareness
    "where_is_user":   _act_where_is_user,
    "see_user":        _act_see_user,
    "which_monitor":   _act_which_monitor,
    # Multi-monitor app launching
    "open_on_monitor": _act_open_on_monitor,
    "move_window_to_monitor": _act_move_window_to_monitor,
    # Window management
    "list_windows":    _act_list_windows,
    "focus_window":    _act_focus_window,
    "minimize_window": _act_minimize_window,
    "close_window":    _act_close_window,
    # iTunes music playback
    "play_music":      _act_play_music,
    "pause_music":     _act_pause_music,
    "resume_music":    _act_resume_music,
    "next_song":       _act_next_song,
    "previous_song":   _act_previous_song,
    "now_playing":     _act_now_playing,
    # Streaming services — open + auto-click first result + click play
    "apple_music":     _act_apple_music,
    "netflix":         _act_netflix,
    "prime_video":     _act_prime_video,
    "disney_plus":     _act_disney_plus,
    "hulu":            _act_hulu,
    "max":             _act_max,
    "spotify":         _act_spotify,
    "youtube_play":    _act_youtube_play,
    "play_streaming":  _act_play_streaming,
    # Task queue (for Claude Code handoff)
    "queue_task":            _act_queue_task,
    "show_tasks":            _act_show_tasks,
    "clear_tasks":           _act_clear_tasks,
    "session_memory_recall": _act_session_memory_recall,
    "session_resume":        _act_session_resume,
    "upgrade":               _act_upgrade,
    "restart":               _act_restart,
    "start_overnight_upgrade": _act_start_overnight_upgrade,
    # task-66: voice routing for ambient mode. Single function, multiple
    # phrasings so "ambient mode" / "silent learning" / "start eavesdropping"
    # all route to the same toggle the tray menu uses.
    "ambient_mode":          _act_ambient_mode_toggle,
    "ambient_mode_on":       lambda _="": _act_ambient_mode_set(True),
    "ambient_mode_off":      lambda _="": _act_ambient_mode_set(False),
    "silent_learning":       _act_ambient_mode_toggle,
    "silent_learning_on":    lambda _="": _act_ambient_mode_set(True),
    "silent_learning_off":   lambda _="": _act_ambient_mode_set(False),
    "ambient_listening":     _act_ambient_mode_toggle,
    "start_eavesdropping":   lambda _="": _act_ambient_mode_set(True),
    "stop_eavesdropping":    lambda _="": _act_ambient_mode_set(False),
    "chappie_mode":          _act_ambient_mode_toggle,
    # Voice-triggered full shutdown (graceful — distinct from upgrade/restart).
    # Triggered by the SHUTDOWN_TRIGGER_PHRASES pre-router after the user
    # answers 'no' to the "overnight first?" prompt, or directly by the LLM
    # if it ever decides to route a shutdown utterance through the normal
    # action channel. All aliases dispatch to _act_shutdown_jarvis.
    "shutdown_jarvis":       _act_shutdown_jarvis,
    "shut_down":             _act_shutdown_jarvis,
    "exit_jarvis":           _act_shutdown_jarvis,
    "quit_jarvis":           _act_shutdown_jarvis,
    "power_off_jarvis":      _act_shutdown_jarvis,
    "turn_off_jarvis":       _act_shutdown_jarvis,
    # HUD visibility — JARVIS can hide or restore the on-screen overlay on request
    "hide_hud":              _act_hide_hud,
    "show_hud":              _act_show_hud,
    "toggle_hud":            _act_toggle_hud,
    # UI automation
    "click":           _act_click,
    "type":            _act_type,
    "press":           _act_press,
    "hotkey":          _act_hotkey,
    "scroll":          _act_scroll,
    "run_shell":       _act_run_shell,
    # Media + system keys (work for browser music, system volume, etc.)
    "media_next":      _act_media_next,
    "media_prev":      _act_media_prev,
    "media_playpause": _act_media_playpause,
    "volume_up":       _act_volume_up,
    "volume_down":     _act_volume_down,
    "volume_mute":     _act_volume_mute,
    # Replay the most recent non-destructive action ('do that again', etc.)
    "replay_last_action": _act_replay_last_action,
    # Changelog + version info ('what's new', 'show changelog', 'what version are you on')
    "read_changelog":  _act_read_changelog,
    "whats_new":       _act_read_changelog,
    "what_changed":    _act_read_changelog,
    "show_changelog":  _act_read_changelog,
    "recent_changes":  _act_read_changelog,
    "version_info":    _act_version_info,
    "what_version":    _act_version_info,
    "when_updated":    _act_version_info,
    # Update awareness — compare the running build to the latest GitHub release
    "check_for_updates":  _act_check_for_updates,
    "check_updates":      _act_check_for_updates,
    "is_there_an_update": _act_check_for_updates,
    # Bug reporting — a USER-reported bug, scrubbed locally then offered as a
    # pre-filled GitHub issue (consent-gated; no auto-send).
    "report_bug":         _act_report_bug,
    "report_a_bug":       _act_report_bug,
    "log_a_bug":          _act_report_bug,
    "file_a_bug":         _act_report_bug,
    # Tray Power Tools submenu — these were silently dropping with
    # "unknown command" until round5-H-2 wired them in. Every handler is
    # defined just above this dict; the destructive ones (force_backup,
    # reset_memory) snapshot before mutating.
    "stop_pipeline":      _act_stop_pipeline,
    "force_backup":       _act_force_backup,
    "reload_skills":      _act_reload_skills,
    "run_smoke_test":     _act_run_smoke_test,
    "reset_llm_cache":    _act_clear_llm_cache,   # tray alias for clear_llm_cache
    # AI submenu
    "switch_llm":         _act_switch_llm,
    "switch_llm_picker":  _act_switch_llm_picker,
    "show_llm_stats":     _act_show_llm_stats,
    # Model cost transparency — list models + estimated cost per conversation
    "model_costs":        _act_model_costs,
    "llm_costs":          _act_model_costs,
    "model_prices":       _act_model_costs,
    "compare_models":     _act_model_costs,
    "clear_llm_cache":    _act_clear_llm_cache,
    # Memory submenu
    "show_recent_facts":  _act_show_recent_facts,
    "reset_memory":       _act_reset_memory,
    "export_memory":      _act_export_memory,
    "forget_last_hour":   _act_forget_last_hour,
    # Diagnostics submenu — run_diagnostic is ALSO registered by
    # skills/self_diagnostic.py at boot. The skill's register() runs
    # after this dict is built, so the skill version wins — that's
    # intentional (it's the canonical implementation). The tray-only
    # wrapper here is the safety net for when the skill failed to load.
    "run_diagnostic":         _act_run_diagnostic_tray,
    "show_last_diagnostic":   _act_show_last_diagnostic,
    "test_mic":               _act_test_mic,
    "test_tts":               _act_test_tts,
    "test_vision":            _act_test_vision,
    "test_each_skill":        _act_test_each_skill,
    "latency_benchmark":      _act_latency_benchmark,
}


# ──────────────────────────────────────────────────────────────────────────
#  SKILLS  — Bobert can write new Python modules to teach himself new tasks
# ──────────────────────────────────────────────────────────────────────────

SKILLS_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
PENDING_SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pending_skills")

# Utilities skills can import. We assign these here so skill modules can
# import them via:  from bobert_companion import skill_utils
skill_utils = {
    "ask_vision":       lambda *a, **kw: ask_vision(*a, **kw),
    "take_screenshot":  lambda: take_screenshot(),
    "find_click_target": lambda d: find_click_target(d),
    "click":            lambda x, y, b="left": ui_click(x, y, b),
    "type_text":        lambda t: ui_type(t),
    "press_key":        lambda k: ui_press(k),
    "hotkey":           lambda *k: ui_hotkey(*k),
    "scroll":           lambda n: ui_scroll(n),
    "sleep":            lambda s: time.sleep(s),
    "launch_app":       lambda n: _act_launch_app(n),
    "open_url":         lambda u: _act_open_url(u),
    # Canonical hud_state.json publisher — uses the module-level
    # _hud_state_lock + cached state so concurrent skill writers can't
    # clobber each other's fields via independent read-modify-write races.
    # Skills should call: skill_utils["write_hud_state"](key=value, ...)
    "write_hud_state":  lambda **kw: _write_hud_state(**kw),
    # Contextual callbacks: skills that want a deferred announcement
    # ('print finishes', 'bed cools') should call:
    #   skill_utils["make_promise"](message, condition, params=..., source=...)
    # Returns the promise id, or None if the core/memory module isn't loaded.
    "make_promise":     (lambda *a, **kw: _promises.make_promise(*a, **kw)) if _promises else (lambda *a, **kw: None),
    "register_promise_condition": (lambda *a, **kw: _promises.register_condition(*a, **kw)) if _promises else (lambda *a, **kw: None),
    "fulfil_promise":   (lambda pid: _promises.fulfil_promise(pid)) if _promises else (lambda pid: False),
}

# M2 Phase 1 (2026-06-02): typed capability seam. JarvisServices wraps the
# untyped skill_utils dict above in a typed facade (one method per capability,
# accurate signatures, graceful degradation) so skills can depend on an
# *interface* instead of a dict of lambdas closing over monolith globals — the
# prerequisite for later moving a capability across a process bus without
# touching skills. See docs/design/M2-process-isolation.md §"the two seams".
#
# This is purely ADDITIVE: the skill_utils dict and its injection are unchanged;
# load_skills() injects this object as `mod.services` *alongside* `mod.skill_utils`
# so existing `skill_utils["write_hud_state"](...)` calls keep working untouched
# and skills migrate to `services.write_hud_state(...)` one at a time. core.services
# is stdlib-only, so importing it here can't introduce an import cycle; the
# try/except is belt-and-braces — a hypothetical broken core/services.py must never
# stop skills from loading (the loader just falls back to dict-only injection).
try:
    from core.services import JarvisServices as _JarvisServices  # noqa: E402
    _jarvis_services = _JarvisServices.from_skill_utils(skill_utils)
except Exception as _svc_exc:  # pragma: no cover - defensive: core.services is stdlib-only and present in this tree
    print(f"  [skill] JarvisServices unavailable ({_svc_exc}); "
          f"skills fall back to skill_utils dict only")
    _jarvis_services = None


_loaded_skill_names: set[str] = set()   # skills load_skills() has loaded


def load_skills():
    """Import every .py file and every package directory in ./skills/ and
    let it register actions.

    Skills can be either a flat module (`skills/foo.py`) or a package
    (`skills/foo/__init__.py`). Both load via spec_from_file_location and
    end up registered in sys.modules as `skill_<name>` so dependants that
    do `sys.modules.get("skill_foo")` find them either way. Packages are
    handed `submodule_search_locations` so relative imports inside
    `__init__.py` resolve to sibling files in the same directory.
    """
    if not SKILLS_ENABLED:
        return
    os.makedirs(SKILLS_DIR, exist_ok=True)
    import glob, importlib.util

    # Build a (name, path, submodule_search_locations) list. Packages
    # take priority over a same-named flat .py — matches Python's own
    # import precedence — but we still warn if both exist so the user
    # knows to delete one.
    entries: list[tuple[str, str, list[str] | None]] = []
    seen: set[str] = set()
    for entry in sorted(os.listdir(SKILLS_DIR)):
        sub = os.path.join(SKILLS_DIR, entry)
        if not os.path.isdir(sub):
            continue
        if entry.startswith("_") or entry == "__pycache__":
            continue
        init_path = os.path.join(sub, "__init__.py")
        if os.path.isfile(init_path):
            entries.append((entry, init_path, [sub]))
            seen.add(entry)
    for path in sorted(glob.glob(os.path.join(SKILLS_DIR, "*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.startswith("_"):
            continue
        if name in seen:
            print(f"  [skill] {name}: both .py and package found — using package")
            continue
        entries.append((name, path, None))

    for name, path, search_locs in entries:
        # Idempotent reload: skip skills load_skills() has already loaded.
        # reload_skills exists to pick up NEWLY-added skills; re-exec'ing an
        # already-loaded module resets its globals AND re-runs register(),
        # which duplicates the background daemon threads ~20 skills start
        # without an is_alive() guard (a reload-only thread leak found in the
        # 2026-05-30 exhaustive audit). Edits to an existing skill are picked
        # up on the next full restart (the upgrade pipeline restarts JARVIS).
        # Tracked in a dedicated set, not `in sys.modules`, so a cross-skill
        # `import skill_x` can't trick us into skipping x before we load it.
        if name in _loaded_skill_names:
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"skill_{name}", path,
                submodule_search_locations=search_locs,
            )
            if not spec or not spec.loader:
                continue  # pragma: no cover - defensive: spec_from_file_location returns a loaded spec for any real .py path
            mod = importlib.util.module_from_spec(spec)
            mod.skill_utils = skill_utils   # inject utilities
            # M2 Phase 1: additively inject the typed JarvisServices facade
            # alongside the dict. Skills can use `services.write_hud_state(...)`
            # instead of `skill_utils["write_hud_state"](...)`; both reach the
            # same lambdas. Only injected when construction succeeded above —
            # otherwise skills keep using the dict (still present, unchanged).
            if _jarvis_services is not None:
                mod.services = _jarvis_services
            # Make sub-imports resolve for packages — register before
            # exec so any `from .submodule import x` inside __init__ can
            # find the parent module.
            sys.modules[f"skill_{name}"] = mod
            spec.loader.exec_module(mod)
            if hasattr(mod, "register"):
                before = set(ACTIONS.keys())
                mod.register(ACTIONS)
                added = set(ACTIONS.keys()) - before
                if added:
                    print(f"  [skill] {name}: added actions: {', '.join(sorted(added))}")
                else:
                    print(f"  [skill] {name}: loaded (no new actions)")
            _loaded_skill_names.add(name)   # mark loaded only after success
        except Exception as e:
            print(f"  [skill] {name}: failed to load — {e}")


# ── bridges into optional skill modules (looked up lazily so JARVIS still
#    boots cleanly if a skill failed to load) ─────────────────────────────
def _audio_music_feed(audio, sample_rate: int) -> None:
    """Forward a freshly-recorded audio chunk to the spectral music
    detector if the standby_audio_detect skill is loaded. No-op otherwise."""
    mod = sys.modules.get("skill_standby_audio_detect")
    if mod is None:
        return
    try:
        mod.feed_audio(audio, sample_rate)
    except Exception:
        pass


def _ambient_learning_feed(text: str) -> None:
    """Ambient-learning standby: persist a non-wake overheard utterance to
    data/ambient_transcripts.jsonl (source='mic', tagged 'ambient_standby')
    so the multimodal fact-extractor distills it into long-term memory on its
    next interval pass. Best-effort and SILENT — never raises into the listen
    loop, never speaks, and does NO LLM work here (the extractor's own loop
    owns distillation). Empty / 1-2 word transcripts are dropped: Whisper
    emits stray single tokens ('you', 'thanks') on near-silence and they carry
    no durable signal. In staging the write is redirected to a *.staging.jsonl
    sibling so test injects never pollute real memory."""
    try:
        t = (text or "").strip()
        if len(t) < 8 or len(t.split()) < 3:
            return
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        fname = ("ambient_transcripts.staging.jsonl" if _is_staging()
                 else "ambient_transcripts.jsonl")
        rec = {"ts": time.time(), "source": "mic", "text": t[:500],
               "tag": "ambient_standby"}
        with open(os.path.join(data_dir, fname), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _audio_music_should_refuse_wake(text: str) -> bool:
    """Ask the spectral music detector whether a wake-word match looks like
    a lyric near-miss. False if the skill isn't loaded."""
    mod = sys.modules.get("skill_standby_audio_detect")
    if mod is None:
        return False
    try:
        return bool(mod.should_refuse_wake(text))
    except Exception:
        return False


# Standby auto-engage bridge. The background lyric-detection thread in
# skills/standby_audio_detect calls this when it's seen sustained vocal
# music while the headset is the active output — flipping standby state
# from a worker thread can race with the main loop's reads of the same
# slot, so the bridge serialises the mutation under a lock and never
# fires a second time while standby is already active.
_standby_auto_engage_lock = threading.Lock()


def _standby_auto_engage(reason: str = "music") -> bool:
    """Engage standby/wake-word-only mode programmatically. Returns True
    if state was changed, False if it was already in standby or the call
    was suppressed. Safe to call from a background thread."""
    with _standby_auto_engage_lock:
        if _standby_mode[0] or _sleep_mode[0]:
            return False
        _sleep_mode[0]   = True
        _standby_mode[0] = True
        try:
            _write_hud_state(state="Standby",
                             sleep_mode=True, standby_mode=True)
        except Exception:
            pass
    # Lock RELEASED before _speak/set_state. Do NOT hold
    # _standby_auto_engage_lock across the 3-5s blocking TTS line — the wake
    # path (main loop) takes the SAME lock to clear these flags, so holding it
    # across playback froze "say JARVIS to wake" for the entire spoken line (a
    # regression from adding the wake-clear lock). The flags are already set
    # above under the lock; speaking is just the notification. 2026-05-30 audit.
    print(f"  [standby] auto-engage ({reason}) — entering standby")
    try:
        _speak("I'll wait until you call, sir.")
    except Exception as e:
        print(f"  [standby] auto-engage TTS failed: {e}")
    try:
        set_state("idle")
    except Exception:
        pass
    return True


def _act_ambient_learning_set(on: bool) -> str:
    """Voice-controllable twin of the JARVIS_AMBIENT_LEARNING boot flag.
    ON  → JARVIS drops to SILENT standby, keeps feeding the fact-extractor
          from what it overhears, and won't speak until you say 'JARVIS'.
    OFF → exit back to normal interactive listening."""
    if on:
        with _standby_auto_engage_lock:
            _sleep_mode[0]   = True
            _standby_mode[0] = True
        _ambient_learning[0]  = True
        _resume_to_ambient[0] = False
        if not _is_staging():
            _ext = sys.modules.get("skill_ambient_multimodal_extract")
            if _ext is not None:
                try:
                    _ext.ambient_extract_start("")
                except Exception:
                    pass
        try:
            _write_hud_state(sleep_mode=True, standby_mode=True)
        except Exception:
            pass
        return ("Entering ambient-learning mode, sir — I'll listen and learn "
                "quietly. Say 'JARVIS' when you want me.")
    _ambient_learning[0]  = False
    _resume_to_ambient[0] = False
    with _standby_auto_engage_lock:
        _sleep_mode[0]   = False
        _standby_mode[0] = False
    try:
        _write_hud_state(sleep_mode=False, standby_mode=False)
    except Exception:
        pass
    return "Ambient-learning mode off, sir. Back to normal."


def _act_wake_resume_set(mode: str) -> str:
    """Switch what JARVIS does AFTER the wake word in ambient-learning:
    'answer_then_quiet' (reply once, then back to silent standby) or
    'stay_talkative' (stay fully interactive until a sleep phrase)."""
    global WAKE_RESUME_MODE
    m = (mode or "").strip().lower().replace(" ", "_").replace("-", "_")
    if m not in ("answer_then_quiet", "stay_talkative"):
        return (f"I don't recognise wake-resume mode '{mode}', sir. Use "
                "'answer then quiet' or 'stay talkative'.")
    WAKE_RESUME_MODE = m
    if m == "answer_then_quiet":
        return ("Wake-resume set to answer-then-quiet, sir — after I answer "
                "I'll slip back into silent learning.")
    return ("Wake-resume set to stay-talkative, sir — once you wake me I'll "
            "stay chatty until you send me to standby.")


# Wire the ambient-learning controls into the dispatch table. Registered here
# (after the ACTIONS literal, like the daemon actions) since these helpers are
# defined below it.
ACTIONS.update({
    "ambient_learning_mode":     lambda _="": _act_ambient_learning_set(
                                     not _ambient_learning[0]),
    "ambient_learning_mode_on":  lambda _="": _act_ambient_learning_set(True),
    "ambient_learning_mode_off": lambda _="": _act_ambient_learning_set(False),
    "enter_ambient_learning":    lambda _="": _act_ambient_learning_set(True),
    "exit_ambient_learning":     lambda _="": _act_ambient_learning_set(False),
    "wake_resume_answer_then_quiet":
        lambda _="": _act_wake_resume_set("answer_then_quiet"),
    "wake_resume_stay_talkative":
        lambda _="": _act_wake_resume_set("stay_talkative"),
})


def _user_looking_away() -> bool:
    """True if the face tracker says the user is not in view. Conservative:
    returns False if the tracker isn't loaded or hasn't established gaze yet,
    so a missing camera never triggers the 'quieter' wake greeting."""
    mod = sys.modules.get("skill_face_tracker")
    if mod is None:
        return False
    try:
        snap = mod._snapshot_state()
        if not snap.get("last_sample_at"):
            return False
        return snap.get("current_monitor") == "away"
    except Exception:
        return False


def _bambu_print_progress() -> int | None:
    """Return current Bambu print percentage if a print is actively running,
    else None. Looked up lazily so JARVIS still boots cleanly if the skill
    isn't loaded or the printer credentials aren't configured."""
    mod = sys.modules.get("skill_bambu_monitor")
    if mod is None:
        return None
    try:
        with mod._state_lock:
            gcode_state = (mod._state.get("gcode_state") or "").upper()
            pct = mod._state.get("mc_percent")
        if gcode_state == "RUNNING" and isinstance(pct, (int, float)) and 1 <= pct <= 99:
            return int(pct)
    except Exception:
        pass
    return None


def _pick_wake_variety(from_standby: bool, wake_text: str = "") -> tuple[str, float]:
    """Pick one of 12 wake acknowledgements (see _WAKE_PHRASE_BANK), weighted
    by time-of-day, recent wake activity, and stress markers in the wake
    utterance itself. Returns (text, volume_scale).

    Selection: build the set of preferred tags from context, pick all phrases
    in the bank that match at least one preferred tag, then drop the last
    used phrase (when ≥2 candidates remain) so we don't repeat back-to-back.
    """
    from datetime import datetime as _dt
    hour = _dt.now().hour
    tone = detect_tone(wake_text) if wake_text else None

    # Repeat-wake heuristic: if the user has woken JARVIS twice or more in
    # the last 5 minutes, prefer terser replies — they're working through
    # something, not making conversation.
    now = time.time()
    recent_wakes = sum(1 for t in _wake_history if (now - t) <= 300)

    preferred: set[str] = set()
    if tone in {"frustrated", "stressed", "rushed"} or recent_wakes >= 2:
        preferred.add("terse")
    if tone == "playful":
        preferred.add("playful")
    if tone == "tired" or hour >= 22 or hour < 5:
        preferred.add("soft")
    # Formal register fits the start of the day and a wake out of standby.
    if from_standby or (5 <= hour < 11):
        preferred.add("formal")
    # General is always in the pool so we have a non-empty candidate set.
    preferred.add("general")

    candidates = [
        (text, tags) for (text, tags) in _WAKE_PHRASE_BANK
        if tags & preferred
    ]
    if not candidates:
        candidates = list(_WAKE_PHRASE_BANK)

    last = _last_wake_phrase[0]
    if last and len(candidates) > 1:
        filtered = [(t, tg) for (t, tg) in candidates if t != last]
        if filtered:
            candidates = filtered

    chosen_text, chosen_tags = random.choice(candidates)
    _last_wake_phrase[0] = chosen_text

    # Soft register and looking-away aren't mutually exclusive — but the
    # looking-away path is handled higher up. Here a 'soft' pick (late-night
    # or tired) gets a slightly gentler volume so JARVIS isn't booming at 2am.
    volume = 0.85 if "soft" in (chosen_tags & preferred) else 1.0
    return (chosen_text, volume)


def context_aware_greeting(from_standby: bool, wake_text: str = "") -> tuple[str, float]:
    """Pick a wake-word greeting that varies with time-of-day, recent wake
    frequency, gaze, printer state, and the tone of the wake utterance.
    Returns (text, volume_scale).

    Priority (first match wins):
      1. 01–05 local AND ≥3 wakes in the last 10 min → "Still up, sir?"
      2. 05–11 local AND first wake of the day       → "Good morning, sir."
      3. Bambu H2D actively printing                 → "At your service…"
      4. User out of view on every camera            → quieter "Yes, sir?"
      5. Variety bank — one of 12 acknowledgements selected by tone /
         time-of-day / recent-wake count (see _pick_wake_variety).
    """
    from datetime import datetime as _dt
    now = time.time()
    # Snapshot pre-wake silence BEFORE this wake's greeting bumps
    # last_speech_time downstream. skills/morning_arrival reads
    # _pre_wake_silence_seconds[0] for its 6-hour overnight gate; reading
    # last_speech_time directly would always be ~0 by the time the morning
    # chain dispatch loop fires (the wake greeting is already speaking).
    _pre_wake_silence_seconds[0] = max(0.0, now - last_speech_time)

    # Slide the wake-history window — keep only the last 10 minutes.
    _wake_history[:] = [t for t in _wake_history if (now - t) <= 600]
    _wake_history.append(now)

    local = _dt.now()
    hour  = local.hour
    today = local.date().isoformat()
    first_of_day = (_last_wake_date[0] != today)
    _last_wake_date[0] = today

    if 1 <= hour < 5 and len(_wake_history) >= 3:
        return ("Still up, sir?", 1.0)

    if 5 <= hour < 12 and first_of_day:
        return ("Good morning, sir.", 1.0)

    pct = _bambu_print_progress()
    if pct is not None:
        return (f"At your service — the print is at {pct}%, by the way.", 1.0)

    if _user_looking_away():
        return ("Yes, sir?", 0.55)

    return _pick_wake_variety(from_standby=from_standby, wake_text=wake_text)


# Phase 4C refactor: _act_list_skills moved to core/actions.py.


# Phase 4J refactor: _act_create_skill moved to core/actions.py.


ACTIONS["list_skills"]  = _act_list_skills
ACTIONS["create_skill"] = _act_create_skill


# Pending confirmation slot: when set, the user's next utterance must
# start with "yes" / "confirm" / "do it" to execute, anything else cancels.
# (_sleep_mode and _overnight_run_now moved up to the top global state block.)
_pending_confirmation: list[tuple[str, str]] = []   # list of (action_name, arg)

# Actions whose RESULT contains information the user actually asked for
# (vs. side-effect-only actions like launching apps). When these run, we do
# a follow-up LLM call so Bobert can actually report what he found.
INFORMATIVE_ACTIONS = {
    "see_screen", "recall_screen", "last_screen", "previous_screen",
    "screen_history",
    "find_on_screen", "get_time", "list_skills", "screenshot",
    "where_is_user", "see_user", "which_monitor", "list_windows", "now_playing",
    # Music actions: results contain "playing X by Y" or failure info that JARVIS
    # should report back to the user.
    "play_music", "pause_music", "resume_music", "next_song", "previous_song",
    # Task queue: show_tasks returns the list contents to read back
    "show_tasks", "queue_task",
    # Credits skill: balance reading is something the user explicitly asked for
    "check_credits",
    # Bambu printer skill: check_print / how_is_the_print / print_details return
    # the print status the user explicitly asked for, so JARVIS must take a
    # follow-up turn to SPEAK it — without this they were logged but never voiced
    # (the result is neither a _is_failure match nor informative). See
    # skills/bambu_monitor.py.
    "check_print", "how_is_the_print", "print_details",
    # Web actions: opening a URL or searching is preparatory — JARVIS should
    # follow up with see_screen to read what actually loaded rather than stopping.
    "open_url", "web_search",
}

# Actions whose RESULT is already a finished, user-facing sentence that should
# be SPOKEN VERBATIM — no follow-up LLM round-trip to restate it. These are NOT
# in INFORMATIVE_ACTIONS on purpose: routing "I'm on version 1.20.4, last
# updated this morning at 7:03 AM, sir." back through get_followup_response just
# burns an LLM call to re-phrase a perfect string (and risks the LLM dropping or
# garbling the number). Instead, the main turn loop speaks the result directly,
# right after the inline reply, via the SAME _speak() path.
#
# THE BUG THIS FIXES (live session 2026-06-03): "what version are you on?" ran
# version_info, which returned the answer to the LOG, but the user only heard
# the "One moment, sir." preamble. version_info wasn't in INFORMATIVE_ACTIONS
# (so no follow-up fired) and its result isn't a failure (so the failure
# follow-up didn't fire either) — the answer was dropped on the floor. Same
# class as the earlier check_print fix, but these results don't NEED an LLM to
# read them back, so they get spoken verbatim rather than re-summarised.
#
# GUARDRAIL — only put actions here whose result is a complete spoken sentence
# the user explicitly asked for AND that is NOT already covered by a spoken
# confirmation. Side-effect actions (play_music / volume_up / set_timer / …)
# must NEVER go here: their effect is the point and the inline reply already
# confirms them, so verbatim-speaking their result would double-speak. The
# speaker also skips any result that's already a substring of what was just
# spoken, and skips failure results (those still route through the failure
# follow-up). Skills register status-report aliases into ACTIONS after boot, so
# they can extend this set the same way they extend INFORMATIVE_ACTIONS.
SPEAK_RESULT_VERBATIM_ACTIONS: set[str] = {
    # Version / last-upgrade readout (core.actions._act_version_info + aliases).
    "version_info", "what_version", "when_updated",
    # Single-sentence health/status aggregator (skills/system_pulse.py) and its
    # natural-phrasing aliases. Each returns one finished status sentence.
    "system_pulse", "check_system", "status_report",
}

# Actions whose runtime can plausibly exceed MID_TASK_STATUS_DELAY (~8 s).
# A timer wrapped around fn(arg) in parse_and_run_actions speaks ONE dry
# status line if the action is still running by then — bridging the silent
# 15-30 s gap during streaming auto-play, the dossier compile pipeline, the
# upgrade hand-off, and the overnight engine spin-up. Skills can extend
# this set the same way they extend INFORMATIVE_ACTIONS (see
# skills/dossier.py for the registration pattern).
LONG_RUNNING_ACTIONS: set[str] = {
    # Auto-play streaming sequence — open service tab, search, click first
    # result, hit play. Each step routinely takes 5-10 s; the chain can run
    # 20+ s end to end.
    "play_streaming",
    "apple_music", "netflix", "prime_video",
    "disney_plus", "hulu", "max",
    "spotify", "youtube_play",
    # Upgrade pipeline / overnight ideas — spawn a console subprocess; the
    # _act_ wrappers themselves return fast, but they're listed for spec
    # completeness so a future synchronous variant inherits the wrapper.
    "upgrade", "start_overnight_upgrade",
}

# Actions that schedule os._exit(0) on a short background timer. The
# mid-task status bridge fires at MID_TASK_STATUS_DELAY (8 s) — well after
# the action's own self-exit timer (1.5–3 s) — so a cosmetic bridge line
# can land mid-shutdown, producing odd half-spoken utterances after the
# user already heard the confirmation. Skip the wrapper entirely for them.
_FIRE_AND_EXIT_ACTIONS: set[str] = {
    "start_overnight_upgrade",
    "upgrade",
    "restart",
    # All shutdown aliases — their handler schedules os._exit(0) on a 2s
    # timer, so the quip layer's tail can't land before the process dies.
    "shutdown_jarvis",
    "shut_down",
    "exit_jarvis",
    "quit_jarvis",
    "power_off_jarvis",
    "turn_off_jarvis",
}

# Status lines keyed by action class — picked in this order:
#   1. service-specific override (Prime Video / Netflix / etc.)
#   2. class bucket (streaming / upgrade / overnight / dossier)
#   3. generic fallback
# Lines stay short (≤ 12 words) and dry, per spec. The "Apologies for the
# delay — X is being uncooperative." pattern lets the spec-quoted Prime
# Video example fall out of the streaming bucket naturally.
_MID_TASK_STATUS_LINES: dict[str, tuple[str, ...]] = {
    "_generic": (
        "Still working, sir, give me a moment.",
        "Almost there, sir.",
        "Working on it.",
        "Bear with me, sir.",
        "Just a moment longer, sir.",
    ),
    "streaming": (
        "Still working, sir — streaming services do test one's patience.",
        "Almost there, sir.",
        "Apologies for the delay — {service} is being uncooperative.",
        "Wrangling {service} now, sir.",
    ),
    "upgrade": (
        "The upgrade pipeline is grinding away, sir — give it a moment.",
        "Still handing things off to Claude Code, sir.",
    ),
    "overnight": (
        "Spinning up the overnight engine, sir.",
        "Almost there, sir — engine's coming online.",
    ),
    "dossier": (
        "Still compiling, sir — fetching the relevant background.",
        "Almost there, sir — finishing the dossier.",
        "Drawing the threads together now, sir.",
    ),
}

# Map action names to a phrase bucket. Skills can extend this dict and
# LONG_RUNNING_ACTIONS together when they register (see skills/dossier.py).
_MID_TASK_STATUS_BUCKET: dict[str, str] = {
    "play_streaming":  "streaming",
    "apple_music":     "streaming",
    "netflix":         "streaming",
    "prime_video":     "streaming",
    "disney_plus":     "streaming",
    "hulu":            "streaming",
    "max":             "streaming",
    "spotify":         "streaming",
    "youtube_play":    "streaming",
    "upgrade":         "upgrade",
    "start_overnight_upgrade": "overnight",
}

# Friendly service names for the streaming bucket's {service} placeholder.
_STREAMING_SERVICE_LABEL: dict[str, str] = {
    "apple_music":  "Apple Music",
    "netflix":      "Netflix",
    "prime_video":  "Prime Video",
    "disney_plus":  "Disney Plus",
    "hulu":         "Hulu",
    "max":          "Max",
    "spotify":      "Spotify",
    "youtube_play": "YouTube",
    "play_streaming": "the service",
}


def _pick_mid_task_status_line(name: str, arg: str = "") -> str:
    """Choose a single dry status line for a long-running action."""
    bucket = _MID_TASK_STATUS_BUCKET.get(name, "_generic")
    lines = _MID_TASK_STATUS_LINES.get(bucket) or _MID_TASK_STATUS_LINES["_generic"]
    line = random.choice(lines)
    if "{service}" in line:
        service = _STREAMING_SERVICE_LABEL.get(name, "the service")
        line = line.replace("{service}", service)
    return line


def _emit_mid_task_status(name: str, arg: str, fired_flag: list[bool]) -> None:
    """Timer callback: speaks the mid-task status line if it hasn't already.
    Runs on a background thread. Failures are swallowed — a missed bridge
    line is strictly cosmetic and must never crash the action dispatch."""
    if fired_flag[0]:
        return
    fired_flag[0] = True
    try:
        line = _pick_mid_task_status_line(name, arg)
        print(f"  [mid_task_status] {name}: {line}")
        _speak(line)
    except Exception as _e:
        print(f"  [mid_task_status] speak failed for {name}: {_e}")


# Phrases that suggest JARVIS just *claimed* to perform an action. If the
# reply contains one of these but no [ACTION: ...] token actually ran, the
# LLM is hallucinating execution (e.g. "Restarting now, sir." with no
# restart action emitted). Detector below appends a synthetic
# _unverified_claim result so the follow-up loop can self-correct.
_ACTION_CLAIM_PHRASES = (
    "restarting", "rebooting",
    "opening ", "launching ", "starting up",
    "playing ", "queueing ", "queuing ", "queued ",
    "closing ", "shutting down", "killing ",
    "pausing ", "resuming ",
    "skipping ", "switching to ", "moving ",
    "minimising", "minimizing", "maximising", "maximizing",
    "typing ", "clicking ", "pressing ",
    "searching for ", "looking up ",
    "setting a timer", "starting a timer",
    "logging in", "logging out",
    "taking a screenshot", "taking a look",
    "on it, sir", "right away, sir",
)


# Preemptive hallucination patterns. The reactive _ACTION_CLAIM_PHRASES check
# below only fires AFTER parse_and_run_actions has finished — meaning TTS has
# already spoken the LLM's incorrect "switching to ambient mode" prose by the
# time the synthetic warning reaches the follow-up loop. These patterns run
# BEFORE _ACTION_RE.sub so we can either auto-inject the missing [ACTION:]
# token (when the claim phrase maps unambiguously to a real action) or refuse
# the reply entirely (when no clear action exists) and force an immediate
# re-prompt before any speech goes out.
#
# Each entry is (regex, action_name_or_None, short_description).
#   action_name in ACTIONS → 'inject': append [ACTION: name] and continue.
#   action_name is None    → 'refuse': drop spoken text, force re-prompt.
_PREEMPTIVE_HALLUCINATION_PATTERNS: list[tuple["re.Pattern", str | None, str]] = [
    # Ambient-LEARNING mode (silent standby + fact-extraction, wake on
    # 'JARVIS') is a DIFFERENT feature from the multimodal ambient/eavesdrop
    # toggle below. Its prose ("entering ambient learning mode") contains the
    # word "ambient", so the generic ambient pattern further down would
    # otherwise hijack it to ambient_mode_on. These MORE-SPECIFIC patterns
    # must precede it — first match wins (see the loop above).
    (re.compile(
        r"\b(?:entering|going\s+into|switching\s+(?:to|into)|enabling|"
        r"starting(?:\s+up)?|activating|turning\s+on|dropping\s+(?:in)?to)\s+"
        r"(?:silent\s+)?ambient[-\s]*learning(?:\s+(?:mode|standby))?\b",
        re.IGNORECASE),
     "ambient_learning_mode_on", "enter ambient-learning standby"),
    (re.compile(
        r"\b(?:exiting|leaving|stopping|disabling|ending|turning\s+off|"
        r"coming\s+out\s+of)\s+"
        r"(?:silent\s+)?ambient[-\s]*learning(?:\s+(?:mode|standby))?\b",
        re.IGNORECASE),
     "ambient_learning_mode_off", "exit ambient-learning standby"),
    (re.compile(r"\banswer[-\s]*then[-\s]*quiet\b", re.IGNORECASE),
     "wake_resume_answer_then_quiet", "wake-resume: answer then quiet"),
    (re.compile(r"\bstay[-\s]*(?:talkative|chatty)(?:\s+mode)?\b", re.IGNORECASE),
     "wake_resume_stay_talkative", "wake-resume: stay talkative"),

    # Ambient mode — covers the toggle action and its many aliases. Direction
    # verbs ("switching to", "enabling", "going into") imply ON; their
    # opposites ("stopping", "disabling", "exiting") imply OFF. Routing to
    # the explicit on/off setter is safer than the toggle because it is
    # idempotent if the current state already matches.
    (re.compile(
        r"\b(?:switching\s+(?:to|into|over\s+to)|entering|going\s+into|"
        r"enabling|starting(?:\s+up)?|activating|turning\s+on)\s+"
        r"(?:ambient|silent\s+learning|chappie|eavesdropping)\s*"
        r"(?:mode|listening)?\b",
        re.IGNORECASE),
     "ambient_mode_on", "enable ambient mode"),
    (re.compile(
        r"\b(?:switching\s+off|stopping|disabling|ending|exiting|"
        r"turning\s+off|leaving)\s+"
        r"(?:ambient|silent\s+learning|chappie|eavesdropping)\s*"
        r"(?:mode|listening)?\b",
        re.IGNORECASE),
     "ambient_mode_off", "disable ambient mode"),
    (re.compile(
        r"\btoggling\s+(?:ambient|silent\s+learning|chappie)\s*"
        r"(?:mode|listening)?\b",
        re.IGNORECASE),
     "ambient_mode", "toggle ambient mode"),

    # Camera movement — no concrete action exists for physically panning the
    # webcams (they're fixed). When the LLM claims to move them, refuse and
    # re-prompt so it can admit the limitation instead of pretending. Kept
    # narrow (camera/webcam/lens only — no "view"/"head"/"gaze") to avoid
    # false positives on metaphorical prose like "adjusting my view".
    (re.compile(
        r"\b(?:moving|panning|tilting|rotating|aiming|pointing|"
        r"repositioning|swinging|swiveling|swivelling|angling)\s+"
        r"(?:the\s+|my\s+|your\s+|one\s+of\s+(?:the\s+|my\s+|your\s+)?)?"
        r"(?:camera|cameras|webcam|webcams|lens)\b",
        re.IGNORECASE),
     None, "physically move the camera"),

    # ── Fabricated INFORMATIONAL answers (local-qwen safety net) ──────────
    # The local 14B model often answers "what time/version/weather/status" from
    # its head instead of routing to the action (it made up "1:47 AM",
    # "version 12.4", "64 degrees", invented CPU/RAM). When the drafted reply
    # ASSERTS one of these facts but emitted no [ACTION:] token, inject the real
    # action so JARVIS speaks the true value instead of the hallucination.
    # Kept deliberately narrow — each pattern needs a concrete factual claim
    # (an actual clock time, a version number, a temperature, a CPU/RAM stat),
    # not a mere mention of the topic — so ordinary conversation never trips it.

    # Clock time: "it's 1:47", "it is 1:47 AM", "the time is 10:52 PM",
    # "the current time is 9 AM". Requires a real clock shape (H:MM or N AM/PM)
    # so "it's 5 minutes left" / "it's time to go" don't match.
    (re.compile(
        r"\b(?:it'?s|it\s+is|the\s+(?:current\s+)?time\s+is|"
        r"right\s+now\s+it'?s)\s+(?:about\s+|approximately\s+|around\s+)?"
        r"\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?|o'?clock)\b",
        re.IGNORECASE),
     "get_time", "state the current time from memory"),
    (re.compile(
        r"\b(?:the\s+(?:current\s+)?time\s+is|it'?s\s+currently)\s+"
        r"\d{1,2}:\d{2}\b",
        re.IGNORECASE),
     "get_time", "state the current time from memory"),

    # Version claim: "version 1.20", "I'm on version 12.4", "running version
    # 3", "I'm running 1.20.6". Requires the word 'version'/'running' next to a
    # number so "version control" / "running late" don't match.
    (re.compile(
        r"\b(?:(?:on|running|at|i'?m\s+on|currently\s+on)\s+)?version\s+v?\d+(?:\.\d+)*\b",
        re.IGNORECASE),
     "version_info", "state the version from memory"),
    (re.compile(
        r"\bi'?m\s+running\s+v?\d+\.\d+(?:\.\d+)*\b",
        re.IGNORECASE),
     "version_info", "state the version from memory"),

    # Weather: "64 degrees Fahrenheit", "it's 72 and sunny", "currently 58
    # degrees and cloudy", "72°F outside". Requires a temperature TIED to a
    # weather signal (°F/°C, fahrenheit/celsius, a weather condition, or an
    # outdoor/now cue) so a bare "turn it 90 degrees" (rotation) doesn't match.
    (re.compile(
        r"\b\d{1,3}\s*(?:°\s*[fc]?\b|degrees?\s+(?:fahrenheit|celsius|[fc])\b)",
        re.IGNORECASE),
     "weather_briefing", "state the weather from memory"),
    (re.compile(
        r"\b\d{1,3}\s*(?:°|degrees?)\b(?:[^.]*?\b"
        r"(?:outside|out\s+there|today|right\s+now|currently|"
        r"sunny|cloudy|clear|overcast|rain(?:y|ing)?|snow(?:y|ing)?|"
        r"windy|foggy|humid|chilly|breezy|forecast|high|low|feels\s+like))",
        re.IGNORECASE),
     "weather_briefing", "state the weather from memory"),
    (re.compile(
        r"\bit'?s\s+(?:currently\s+|about\s+)?\d{1,3}\s+and\s+"
        r"(?:sunny|cloudy|clear|overcast|rain(?:y|ing)?|snow(?:y|ing)?|"
        r"windy|foggy|humid|warm|cold|hot|mild|partly)\b",
        re.IGNORECASE),
     "weather_briefing", "state the weather from memory"),

    # System stats: "CPU is at 40%", "CPU at 12", "40% CPU", "memory is at 80%",
    # "RAM usage is 60%". Requires a percentage/number tied to cpu/ram/memory.
    (re.compile(
        r"\b(?:cpu|ram|memory)\s+(?:usage\s+)?(?:is\s+)?(?:at\s+|sitting\s+at\s+|around\s+)?\d{1,3}\s*%",
        re.IGNORECASE),
     "system_pulse", "state system stats from memory"),
    (re.compile(
        r"\b\d{1,3}\s*%\s+(?:cpu|ram|memory|memory\s+usage)\b",
        re.IGNORECASE),
     "system_pulse", "state system stats from memory"),
    (re.compile(
        r"\bcpu\s+(?:is\s+)?(?:at\s+|sitting\s+at\s+|running\s+at\s+)\d{1,3}\b",
        re.IGNORECASE),
     "system_pulse", "state system stats from memory"),
]


def _detect_preemptive_hallucination(
    reply: str,
) -> tuple[str, str | None, str] | None:
    """Pre-flight scan for hallucinated execution claims.

    Returns one of:
      ('inject', action_name, description) — auto-append the missing
            [ACTION: name] token so the action actually runs.
      ('refuse', None, description) — drop spoken text and force re-prompt.
      None — no hallucination detected.

    Skips when the reply already contains any [ACTION:] token (the LLM did
    emit something — the reactive detector handles partial-emission cases).
    """
    if _ACTION_RE.search(reply):
        return None
    for regex, action_name, desc in _PREEMPTIVE_HALLUCINATION_PATTERNS:
        if not regex.search(reply):
            continue
        if action_name and action_name in ACTIONS:
            return ("inject", action_name, desc)
        return ("refuse", None, desc)
    return None


def _needs_confirmation(name: str, arg: str) -> bool:
    if not CONFIRM_KEYWORDS:
        return False
    haystack = f"{name} {arg}".lower()
    return any(kw in haystack for kw in CONFIRM_KEYWORDS)


# ── JARVIS pushback ──────────────────────────────────────────────────────
# Softer-than-CONFIRM_KEYWORDS safety layer. The hard list above always
# blocks (delete/format/buy etc.). Pushback handles the gray-zone cases the
# user wanted JARVIS to actually push back on conversationally rather than
# refuse outright: force-closing many windows, wiping a long task queue,
# destructive-looking shell commands that aren't on the hard blocklist, and
# sketchy URLs. Returns an in-character objection line + a reason; the
# caller queues the action onto _pending_confirmation and speaks the line.

# Title fragments that strongly suggest a window holds unsaved work. Used to
# embellish the close_window pushback line so it can name what's at stake.
_UNSAVED_WINDOW_HINTS = (
    "untitled",
    "unsaved",
    "(modified)",
    "[modified]",
    " — draft",
    " - draft",
)

# Apps that don't reliably surface dirty/unsaved state in the WINDOW title —
# VS Code shows the `●` indicator only in the TAB strip, Google Docs has no
# marker at all (it auto-saves, but a force-close mid-keystroke can still
# lose the last few seconds of typing). For these we treat the window as
# "may have live edits" so the user at least gets a heads-up before a bulk
# close. Substring is matched case-insensitively against the title.
_LIVE_EDIT_APP_MARKERS = (
    (" - visual studio code",       "your open VS Code window"),
    (" — visual studio code",       "your open VS Code window"),
    (" - code - oss",               "your open VS Code window"),
    (" - cursor",                   "your open Cursor window"),
    (" - google docs",              "your open Google Doc"),
    (" - google sheets",            "your open Google Sheet"),
    (" - google slides",            "your open Google Slides deck"),
    (" - google forms",             "your open Google Form"),
)


def _unsaved_window_blurb(titles: list[str]) -> str | None:
    """If any of `titles` looks like unsaved work, return a short blurb like
    'your unsaved Bambu Studio project'. Falls back to a softer 'your open
    VS Code window' / 'your open Google Doc' style blurb for apps that don't
    surface dirty state in the window title (VS Code tabs, Google Docs
    auto-save). None if nothing matches."""
    live_edit_fallback: str | None = None
    for raw in titles or []:
        t = (raw or "").strip()
        if not t:
            continue
        low = t.lower()
        starts_with_unsaved_mark = t.startswith(("*", "●", "•"))
        # JetBrains and some VS Code variants shift the bullet mid-title
        # ("filename ● - App"); treat that the same as a leading bullet.
        has_embedded_unsaved_mark = (" ● " in t) or (" • " in t)
        if not (starts_with_unsaved_mark
                or has_embedded_unsaved_mark
                or any(p in low for p in _UNSAVED_WINDOW_HINTS)):
            # No strong signal on this title — record the first live-edit
            # app match as a fallback, but keep looking for a stronger
            # signal on later titles.
            if live_edit_fallback is None:
                for marker, blurb in _LIVE_EDIT_APP_MARKERS:
                    if marker in low:
                        live_edit_fallback = blurb
                        break
            continue
        # Try to extract a recognizable app name from the title
        # ("Document — App" / "Document - App" / "App | Document").
        for sep in (" — ", " - ", " | ", " : "):
            if sep in t:
                parts = [p.strip() for p in t.split(sep) if p.strip()]
                if parts:
                    # Heuristic: app name is usually the LAST segment for
                    # "Doc — App" / "Doc - App" patterns. Trim leading
                    # unsaved-mark from the chosen segment.
                    app = parts[-1].lstrip("*●• ").strip()
                    if app:
                        return f"your unsaved {app} project"
        return f'your unsaved "{t.lstrip("*●• ")[:40].strip()}"'
    return live_edit_fallback


# Shell patterns that aren't on the hard _SHELL_FORBIDDEN_PATTERNS blocklist
# but still warrant a "shall I proceed regardless?" before running. The hard
# list takes priority and refuses outright; this softer list just defers
# until the user confirms.
_DESTRUCTIVE_SHELL_PATTERNS = (
    "rm -rf ",
    "rm -fr ",
    "rm --recursive",
    "remove-item -recurse",
    "remove-item -r ",
    "ri -recurse",
    "rmdir /s",
    "rd /s",
    "del /s",
    "del /q",
    "del /f",
    "git reset --hard",
    "git clean -fd",
    "git clean -fx",
    "git push --force",
    "git push -f ",
    "drop table",
    "drop database",
    "truncate table",
)


def _looks_like_destructive_shell(low_cmd: str) -> bool:
    return any(pat in low_cmd for pat in _DESTRUCTIVE_SHELL_PATTERNS)


# Sketchy-URL patterns. Conservative on purpose — false positives just
# prompt a "yes" from the user. Covers: bare IPs over plain HTTP, common
# free-phishing TLDs, tunnel hosts (ngrok / trycloudflare / loca.lt),
# .onion, and link shorteners (which hide the real destination).
_SKETCHY_URL_PATTERNS = (
    re.compile(r"^https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/|$|\?)"),
    re.compile(r"\.(?:tk|ml|cf|gq|top|xyz|click|country|zip|mov)(?:/|$|\?|:)",
               re.IGNORECASE),
    re.compile(r"\.onion(?:/|$|\?|:)", re.IGNORECASE),
    re.compile(r"://[^/\s]*\.(?:ngrok-free\.app|ngrok\.io|trycloudflare\.com|loca\.lt|serveo\.net)",
               re.IGNORECASE),
    re.compile(r"://(?:bit\.ly|tinyurl\.com|goo\.gl|t\.co|is\.gd|buff\.ly|ow\.ly|rebrand\.ly)/",
               re.IGNORECASE),
)


def _is_local_or_lan_host(host: str) -> bool:
    """True if `host` is localhost / loopback / private LAN. Used to whitelist
    dev/LAN URLs out of the sketchy-URL check so http://192.168.1.5 or
    http://localhost:8080 don't trip a pushback."""
    h = (host or "").lower()
    if not h:
        return False
    if h == "localhost" or h.endswith(".local"):
        return True
    if h.startswith("127.") or h.startswith("192.168.") or h.startswith("10."):
        return True
    if h.startswith("169.254."):
        return True
    if h.startswith("172."):
        try:
            octet = int(h.split(".")[1])
        except (IndexError, ValueError):
            return False
        return 16 <= octet <= 31
    return False


def _looks_like_sketchy_url(url: str) -> bool:
    s = (url or "").strip()
    if not s:
        return False
    # Match the action's own normalization so user-typed "example.tk/foo"
    # doesn't slip past us just because it's missing a scheme.
    if not s.startswith(("http://", "https://")):
        s = "https://" + s
    # Extract host first; if it's a localhost / LAN address, short-circuit as
    # safe so neither the bare-IP regex nor the http-without-tls check trips.
    after_scheme = s.split("://", 1)[1]
    host = after_scheme.split("/", 1)[0].split(":", 1)[0].lower()
    if _is_local_or_lan_host(host):
        return False
    for r in _SKETCHY_URL_PATTERNS:
        if r.search(s):
            return True
    # Plain HTTP (non-https) over a public host → flag.
    if s.startswith("http://"):
        return True
    return False


def _jarvis_pushback(name: str, arg: str) -> tuple[str, str] | None:
    """Return (objection_line, reason) if this action deserves an in-character
    objection before running. None means proceed without confirmation. The
    objection is meant to be SPOKEN to the user; the caller queues the
    action on _pending_confirmation and ordinary 'yes' / 'no' handling
    decides whether it actually fires."""
    if not PUSHBACK_ENABLED:
        return None
    nm  = (name or "").lower()
    raw = (arg or "").strip()
    low = raw.lower()

    # close_window: matched too many windows. Embellish the phrase with an
    # unsaved-work hint if any of the matches look like a modified editor.
    if nm == "close_window" and low:
        try:
            matches = _find_windows_by_title(low)
        except Exception:
            matches = []
        if len(matches) > PUSHBACK_MAX_CLOSE_WINDOWS:
            titles = [(m.title or "") for m in matches]
            blurb = _unsaved_window_blurb(titles)
            if blurb:
                phrase = (f"If I may, sir — that will close {len(matches)} "
                          f"windows including {blurb}. Are you certain?")
            else:
                phrase = (f"If I may, sir — that will close {len(matches)} "
                          "windows. Are you certain?")
            return (phrase, f"close_window matched {len(matches)} windows")

    # queue_task: the LLM occasionally bulk-files a newline-separated list.
    # >N items at once is unusual enough to ask.
    if nm == "queue_task" and raw:
        items = [p.strip() for p in re.split(r"[\n;]+", raw) if p.strip()]
        if len(items) > PUSHBACK_MAX_QUEUE_TASKS_BULK:
            phrase = (f"That's {len(items)} tasks at once, sir — well above "
                      "the usual pace. Shall I really file them all?")
            return (phrase, f"bulk queue {len(items)} tasks")

    # clear_tasks: wiping a long backlog should always come with a confirm.
    if nm == "clear_tasks":
        pending = 0
        try:
            with open(TODO_FILE, "r", encoding="utf-8") as f:
                pending = sum(1 for ln in f if ln.lstrip().startswith("- [ ]"))
        except Exception:
            pending = 0
        if pending > PUSHBACK_MAX_CLEAR_PENDING:
            phrase = (f"If I may, sir — that will wipe {pending} pending "
                      "tasks from the queue. Are you certain?")
            return (phrase, f"clear_tasks would delete {pending} pending")

    # reset_memory / forget_last_hour: irreversible memory wipes. Unlike the
    # threshold cases above these are ALWAYS destructive, so push back every
    # time and defer onto _pending_confirmation. (reset_memory auto-backs-up;
    # this confirmation gate is defense-in-depth so a stray LLM-emitted
    # [ACTION: reset_memory] can't erase memory without a spoken 'yes'.)
    if nm == "reset_memory":
        phrase = ("If I may, sir — that will erase my entire memory. "
                  "I'll snapshot it to backups first, but are you certain?")
        return (phrase, "reset_memory wipes all memory")
    if nm == "forget_last_hour":
        phrase = ("If I may, sir — that will forget everything from the "
                  "last hour. Are you certain?")
        return (phrase, "forget_last_hour wipes recent memory")

    # run_shell: destructive-looking command that slipped past the hard
    # blocklist (rm -rf node_modules, Remove-Item -Recurse, git reset --hard…).
    if nm == "run_shell" and low:
        if _looks_like_destructive_shell(low):
            phrase = "That seems inadvisable, sir. Shall I proceed regardless?"
            return (phrase, f"destructive shell pattern: {low[:60]}")

    # open_url: bare-IP / known shortener / sketchy TLD / tunnel host.
    if nm == "open_url" and low:
        if _looks_like_sketchy_url(low):
            phrase = ("That URL looks rather unsavoury, sir. Shall I open "
                      "it regardless?")
            return (phrase, f"sketchy URL: {low[:80]}")

    return None


# ── Mission narration ─────────────────────────────────────────────────────
# Templates for the one-line cue spoken before each chained action. {arg}
# is substituted with a trimmed copy of the action argument. Unmapped
# actions fall back to a generic "Step N: <name>." cue so an unfamiliar
# skill-provided action still narrates cleanly.
_MISSION_NARRATION_CUES = {
    # Web / vision
    "open_url":         "Opening the page",
    "web_search":       "Searching for {arg}",
    "search":           "Searching for {arg}",
    "youtube":          "Pulling up {arg} on YouTube",
    "see_screen":       "Reading what's on screen",
    "recall_screen":    "Recalling what I saw a moment ago",
    "last_screen":      "Recalling what I saw a moment ago",
    "previous_screen":  "Recalling what I saw a moment ago",
    "screen_history":   "Recalling what I saw a moment ago",
    "find_on_screen":   "Looking for {arg}",
    "screenshot":       "Taking a screenshot",
    # Awareness
    "where_is_user":    "Locating you",
    "see_user":         "Taking a look at you",
    "which_monitor":    "Checking which monitor you're on",
    "get_time":         "Checking the time",
    "list_windows":     "Surveying the windows",
    "list_skills":      "Cataloguing my skills",
    # Multi-monitor launching
    "open_on_monitor":  "Opening on the {arg} monitor",
    "move_window_to_monitor": "Moving the window to {arg}",
    # Window management
    "focus_window":     "Bringing {arg} forward",
    "minimize_window":  "Minimising {arg}",
    "close_window":     "Closing {arg}",
    "launch_app":       "Launching {arg}",
    # Music / streaming
    "play_music":       "Playing {arg}",
    "pause_music":      "Pausing the music",
    "resume_music":     "Resuming",
    "next_song":        "Skipping ahead",
    "previous_song":    "Going back a track",
    "now_playing":      "Checking what's playing",
    "apple_music":      "Queueing {arg} on Apple Music",
    "netflix":          "Pulling up {arg} on Netflix",
    "prime_video":      "Pulling up {arg} on Prime Video",
    "disney_plus":      "Pulling up {arg} on Disney Plus",
    "hulu":             "Pulling up {arg} on Hulu",
    "max":              "Pulling up {arg} on Max",
    "spotify":          "Playing {arg} on Spotify",
    "youtube_play":     "Playing {arg} on YouTube",
    "youtube_search_direct": "Opening {arg} on YouTube",
    "youtube_direct":   "Opening {arg} on YouTube",
    "yt_direct":        "Opening {arg} on YouTube",
    "play_streaming":   "Starting {arg}",
    # Task queue / system
    "queue_task":       "Filing that on the task list",
    "show_tasks":       "Pulling up the task list",
    "clear_tasks":      "Clearing the task list",
    "session_memory_recall": "Recalling from memory",
    "session_resume":   "Resuming the session",
    "upgrade":          "Running the upgrade",
    "restart":          "Restarting",
    "start_overnight_upgrade": "Starting the overnight engine",
    "check_credits":    "Checking your credits",
    # UI automation
    "click":            "Clicking",
    "type":             "Typing",
    "press":            "Pressing {arg}",
    "hotkey":           "Sending the hotkey",
    "scroll":           "Scrolling",
    "run_shell":        "Running the command",
    # Media / system keys
    "media_next":       "Next track",
    "media_prev":       "Previous track",
    "media_playpause":  "Toggling playback",
    "volume_up":        "Volume up",
    "volume_down":      "Volume down",
    "volume_mute":      "Toggling mute",
}

# Spelled-out small numbers for the opening line. >9 falls through to digits.
_MISSION_NARRATION_NUMBERS = {
    2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine",
}


def _mission_narration_intro(count: int) -> str:
    """Opening line for a chained-action turn. Example: 'Three steps, sir.'"""
    word = _MISSION_NARRATION_NUMBERS.get(count, str(count))
    return f"{word} steps, sir."


def _mission_narration_cue(name: str, arg: str, step: int, total: int) -> str:
    """One-line cue spoken before a single action in a chained turn."""
    short_arg = (arg or "").strip()
    if len(short_arg) > 40:
        short_arg = short_arg[:37].rstrip() + "…"
    tpl = _MISSION_NARRATION_CUES.get(name)
    if tpl:
        try:
            body = tpl.format(arg=short_arg).rstrip()
        except Exception:
            body = tpl.rstrip()
        # Strip a dangling "{arg}" placeholder when arg was empty
        body = body.replace("  ", " ").rstrip(" ,")
    else:
        body = f"Step {step} of {total}: {name.replace('_', ' ')}"
    return body + "…"


# Continuation enforcer: explicit future-tense markers the LLM uses when
# promising another chained step. We require a real future marker (not just
# a bare "and") so past-tense narration ("I opened the page and read it")
# is not mistaken for a dropped step.
_FUTURE_MARKER_RE = re.compile(
    r"\b(?:i'?ll|i\s+will|i'?m\s+(?:going\s+to|gonna|about\s+to)|"
    r"let\s+me|let's\b|then\s+i'?ll|going\s+to|gonna\b|"
    r"after\s+that(?:\s+i'?ll)?|and\s+then(?:\s+i'?ll)?)\b",
    re.IGNORECASE,
)

# Intent targets: (regex matching the *action description* on its own,
# expected [ACTION:] name, short human-readable description). The future
# marker is matched separately by _FUTURE_MARKER_RE; a target only counts
# as a dropped step if it appears within ~120 characters AFTER a marker.
_CONTINUATION_INTENTS: list[tuple[re.Pattern, str, str]] = [
    # "read it to you" / "tell you what's there" / "check the page" / "see what's loaded"
    (re.compile(r"\b(?:read\s+(?:it|that|them|the\s+\w+)|"
                r"tell\s+you\s+(?:what|the|how|whether)|let\s+you\s+know|"
                r"report\s+back|"
                r"see\s+what(?:'s|s|\s+is|\s+has)\s+(?:on|loaded|there|up)|"
                r"check\s+(?:the\s+)?(?:page|screen|results?|tab|window))",
                re.IGNORECASE),
     "see_screen", "read what's on screen"),
    # "take a screenshot" / "grab a screen capture"
    (re.compile(r"\b(?:take|grab|capture|snap)\s+(?:a\s+|another\s+|fresh\s+)*"
                r"(?:screen\s?shot|screen\s+capture)\b",
                re.IGNORECASE),
     "screenshot", "take a screenshot"),
    # "search for X" / "google it" / "look that up"
    (re.compile(r"\b(?:search\s+(?:for\s+|the\s+web\s+(?:for\s+)?)?\S+|"
                r"google\s+\S+|look\s+(?:that|it|this)\s+up)\b",
                re.IGNORECASE),
     "web_search", "search the web"),
    # "set a timer" / "start a timer"
    (re.compile(r"\b(?:set|start)\s+(?:a\s+|the\s+)?timer\b",
                re.IGNORECASE),
     "set_timer", "set a timer"),
    # "play some music" / "queue up your jazz playlist"
    (re.compile(r"\b(?:play|queue(?:\s+up)?)\s+(?:some\s+|the\s+|your\s+|that\s+|a\s+)?"
                r"(?:music|song|track|playlist|album)\b",
                re.IGNORECASE),
     "play_music", "play music"),
    # "check where you are" / "see which monitor you're on"
    (re.compile(r"\b(?:check|see|find\s+out)\s+(?:where|which\s+monitor)\s+"
                r"you\s+(?:are|'?re)\b",
                re.IGNORECASE),
     "where_is_user", "find where you are"),
]

# How far after a future marker an intent verb still counts as part of the
# same promise. ~120 chars covers e.g. "I'll open the page and then read
# the article and tell you what's there" without crossing into unrelated
# sentences further down the reply.
_CONTINUATION_WINDOW = 120


def _detect_dropped_steps(reply_text: str,
                          emitted_actions: set[str]) -> list[tuple[str, str]]:
    """Look for chained-intent phrases in the LLM's prose that weren't backed
    by a corresponding [ACTION: ...] token. Returns list of
    (expected_action, description) for steps that were promised but dropped.

    Only flags actions that (a) are currently registered in ACTIONS — so
    skill-provided actions don't fire false positives when the skill failed
    to load — and (b) were not already emitted in this reply, and
    (c) appear within _CONTINUATION_WINDOW characters after an explicit
    future-tense marker like "I'll" / "let me" / "then I'll"."""
    prose = _ACTION_RE.sub(" ", reply_text)
    marker_ends = [m.end() for m in _FUTURE_MARKER_RE.finditer(prose)]
    if not marker_ends:
        return []
    dropped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for target_re, action_name, desc in _CONTINUATION_INTENTS:
        if action_name in emitted_actions or action_name in seen:
            continue
        if action_name not in ACTIONS:
            continue
        for tm in target_re.finditer(prose):
            target_start = tm.start()
            if any(0 < target_start - me <= _CONTINUATION_WINDOW
                   for me in marker_ends):
                dropped.append((action_name, desc))
                seen.add(action_name)
                break
    return dropped


def parse_and_run_actions(reply: str) -> tuple[str, list[tuple[str, str, bool]]]:
    """
    Find all [ACTION: ...] tokens, execute whitelisted ones, defer risky ones
    until the user confirms verbally.
    Returns: (cleaned_reply, [(action_name, result_string, is_informative), ...])
    """
    if not PC_CONTROL_ENABLED:
        return reply, []

    # Preemptive hallucination check. Runs BEFORE _ACTION_RE.sub so we can
    # either auto-inject a missing token (when a claim phrase maps cleanly
    # to a real action) or refuse outright and force an immediate re-prompt
    # — the reactive detector at the bottom of this function only catches
    # the slip AFTER TTS has already spoken the wrong claim.
    _preempt = _detect_preemptive_hallucination(reply)
    if _preempt is not None:
        _kind, _action_name, _desc = _preempt
        if _kind == "inject":
            print(
                f"  [preemptive_hallucination] LLM claimed '{_desc}' without "
                f"[ACTION:] — injecting [ACTION: {_action_name}]"
            )
            reply = f"{reply.rstrip()} [ACTION: {_action_name}]"
        else:
            warn = (
                f"reply claimed to '{_desc}' but no [ACTION: ...] token was "
                "emitted and no clear action mapping exists — refused before "
                "TTS; re-emit with a real action token or explicitly admit "
                "you cannot perform that"
            )
            print(f"  [preemptive_hallucination] refused — {_desc}")
            return "", [("_preemptive_hallucinated_claim", warn, True)]

    # Reset the per-intent see_screen budget. Bounded per dispatch (not per
    # session) so it never carries over between unrelated user requests.
    _see_screen_budget_state.used = 0

    results: list[tuple[str, str, bool]] = []
    # JARVIS-style objection lines accumulated during this dispatch. Any
    # pushback that fired replaces the LLM's spoken prose so the user hears
    # the objection, not the original "I'll close them all, sir." prelude.
    _pushback_objections: list[str] = []

    # Mission narration — pre-scan to count [ACTION:] tokens. When the LLM
    # has chained 3+ actions in one reply, speak an opening line and emit a
    # cue before each step so the chain feels intentional rather than silent.
    _planned_actions = [m for m in _ACTION_RE.finditer(reply)
                        if not _needs_confirmation(m.group(1).strip().lower(),
                                                   (m.group(2) or "").strip())]
    _narrate = (MISSION_NARRATION_ENABLED
                and len(_planned_actions) >= MISSION_NARRATION_THRESHOLD)
    _narration_step = [0]
    if _narrate:
        try:
            _speak(_mission_narration_intro(len(_planned_actions)))
        except Exception as _e:
            print(f"  [mission_narration] intro failed: {_e}")

    def _runner(match):
        name = match.group(1).strip().lower()
        arg  = (match.group(2) or "").strip()
        fn   = ACTIONS.get(name)
        if fn is None and _cmd_autocorrect is not None:
            # Fuzzy-correction layer. ACTIONS may have grown via
            # load_skills() since boot, so we always pass the live keys.
            # Three outcomes:
            #   silent     → top match clearly beats runner-up; route now
            #                and tell the user we interpreted it
            #   ambiguous  → two candidates within _AUTOCORRECT_AMBIG_GAP
            #                of each other; ask 'did you mean X or Y' and
            #                defer execution onto _pending_autocorrect_choice
            #                so the next utterance resolves the pick
            #   none       → no candidate cleared the floor; original
            #                'unknown action' path
            try:
                choice = _cmd_autocorrect.autocorrect_command_choice(
                    name, ACTIONS.keys(),
                    threshold=_AUTOCORRECT_THRESHOLD,
                    ambiguity_gap=_AUTOCORRECT_AMBIG_GAP,
                )
            except Exception as _e:
                print(f"  [autocorrect] scoring failed for {name!r}: {_e}")
                choice = {"status": "none", "primary": None, "secondary": None}
            status = choice.get("status", "none")
            primary = choice.get("primary")
            secondary = choice.get("secondary")
            if status == "silent" and primary and primary[0] in ACTIONS:
                best, conf = primary
                print(f"  [autocorrect] {name!r} -> {best!r} (conf={conf:.2f})")
                try:
                    _speak(f"Interpreting that as `{best}`, sir.")
                except Exception as _e:
                    print(f"  [autocorrect] confirmation TTS failed: {_e}")
                name = best
                fn = ACTIONS.get(name)
            elif (status == "ambiguous"
                    and primary and secondary
                    and primary[0] in ACTIONS and secondary[0] in ACTIONS):
                a, a_conf = primary
                b, b_conf = secondary
                print(f"  [autocorrect] ambiguous {name!r} -> "
                      f"{a!r} ({a_conf:.2f}) vs {b!r} ({b_conf:.2f})")
                _pending_autocorrect_choice.clear()
                _pending_autocorrect_choice.append({
                    "primary": (a, arg),
                    "secondary": (b, arg),
                    "original": name,
                })
                try:
                    _speak(f"Did you mean `{a}` or `{b}`, sir?")
                except Exception as _e:
                    print(f"  [autocorrect] disambig TTS failed: {_e}")
                results.append(
                    (name,
                     f"⚠  AMBIGUOUS: {a} vs {b} — awaiting clarification",
                     False),
                )
                return ""
            else:
                best_seen = primary[1] if primary else 0.0
                print(f"  [autocorrect] no match for {name!r} "
                      f"(best conf={best_seen:.2f})")
        if fn is None:
            results.append((name, f"unknown action: {name}", False))
            return ""

        if _needs_confirmation(name, arg):
            _pending_confirmation.append((name, arg))
            msg = f"⚠  REQUIRES CONFIRMATION: {name}({arg}) — say 'yes' to proceed"
            print(f"  [action] {msg}")
            results.append((name, msg, False))
            return ""

        # JARVIS-style pushback for gray-zone actions. Same deferred-execution
        # pipeline as CONFIRM_KEYWORDS, but the user hears an in-character
        # objection rather than a generic "say yes" warning, and the LLM's
        # original prose is dropped so JARVIS isn't both warning AND claiming
        # to do it.
        _pb = _jarvis_pushback(name, arg)
        if _pb:
            objection, reason = _pb
            _pending_confirmation.append((name, arg))
            _pushback_objections.append(objection)
            print(f"  [pushback] {reason} → '{objection}'")
            results.append((name, f"⚠  PUSHBACK: {objection}", False))
            return ""

        if _narrate:
            _narration_step[0] += 1
            try:
                _speak(_mission_narration_cue(name, arg,
                                              _narration_step[0],
                                              len(_planned_actions)))
            except Exception as _e:
                print(f"  [mission_narration] cue failed: {_e}")

        _write_hud_state(active_action=name,
                         now_doing=f"EXECUTING: {name}")
        # Mid-task status bridge: for long-running actions, start a timer
        # that speaks one dry status line at the 8 s mark so JARVIS doesn't
        # feel frozen during 15-30 s sequences. Canceled the moment fn(arg)
        # returns; joined briefly so a mid-speech status line finishes
        # before the main loop's follow-up TTS starts.
        _mid_task_timer = None
        _mid_task_fired = [False]
        if (MID_TASK_STATUS_ENABLED
                and name in LONG_RUNNING_ACTIONS
                and name not in _FIRE_AND_EXIT_ACTIONS):
            try:
                _mid_task_timer = threading.Timer(
                    MID_TASK_STATUS_DELAY,
                    _emit_mid_task_status,
                    args=(name, arg, _mid_task_fired),
                )
                _mid_task_timer.daemon = True
                _mid_task_timer.start()
            except Exception as _e:
                print(f"  [mid_task_status] timer start failed for {name}: {_e}")
                _mid_task_timer = None
        try:
            # Draft preview / confirmation gate. Every send_* action routes
            # through the middleware so the user always hears the draft body
            # read back and gets an 8-second 'shall I send it, sir?' window
            # before the underlying send fires. Pass-through when no draft
            # is pending or the gate module isn't loaded.
            if (_draft_preview_gate is not None
                    and _draft_preview_gate.should_gate(name)):
                res = _draft_preview_gate.run_with_gate(name, arg, fn)
            else:
                res = fn(arg)
            results.append((name, res, name in INFORMATIVE_ACTIONS))
            print(f"  [action] {name}: {res[:120]}{'…' if len(res) > 120 else ''}")
            record_session_action(name, arg)
            # Replay-last-action history. Skip when the replay handler itself
            # is the action so the deque continues to point at the real target.
            if name != "replay_last_action":
                record_action_history(name, arg, res)
        except UIFailsafeError as e:
            # Translate the pyautogui fail-safe trip into a clean, user-facing
            # message so the LLM doesn't see "failed: ..." and retry forever.
            results.append((name, str(e), False))
            print(f"  [action] {name}: {e}")
        except Exception as e:
            # JARVIS-voice failure: classify the exception, draw a random
            # in-character line for that class, and lead the result string
            # with it. The "(action failed; ...)" suffix keeps the raw
            # exception visible to the follow-up LLM call AND keeps the
            # word "failed" in the result so the main loop's _is_failure
            # heuristic still routes it through the follow-up chain.
            klass, line, technical = _jfl.failure_message(e, name)
            result_str = f"{line} (action failed; class={klass}; {technical})"
            results.append((name, result_str, False))
            print(f"  [action] {name} failed [{klass}]: {e}")
            # Feed the self-diagnostic auto-queue. Capture the live traceback
            # before any other code runs (traceback.format_exc reflects the
            # CURRENT sys.exc_info, so we grab it right here in the except).
            try:
                _tb = traceback.format_exc()
            except Exception:
                _tb = ""
            record_action_error(name, e, _tb)
            # Self-detect bug report: a SCRUBBED, rate-limited record of the
            # exception written to the LOCAL outbox (core.bug_reporter). Local
            # only here — submission stays a separate opt-in. Never raises;
            # disable with JARVIS_BUG_AUTO_CAPTURE=0.
            if os.environ.get("JARVIS_BUG_AUTO_CAPTURE", "1") != "0":
                try:
                    import core.bug_reporter as _br
                    _rep = _br.auto_capture(e, where=name,
                                            context={"arg": arg[:200]})
                    # Autonomous upstream submission only when opted in.
                    if _rep is not None and _br.auto_submit_enabled():
                        _br.api_submit_issue(_rep)
                except Exception:
                    pass
        finally:
            # Stop the mid-task status timer. If it hasn't fired yet,
            # cancel() prevents it from firing. If it's already mid-speech,
            # join() briefly so the bridge line finishes before the main
            # loop's reply TTS starts (caps at the configured delay so a
            # stuck speech thread can never hang the dispatcher).
            if _mid_task_timer is not None:
                try:
                    _mid_task_timer.cancel()
                    if _mid_task_fired[0]:
                        _mid_task_timer.join(timeout=MID_TASK_STATUS_DELAY)
                except Exception:
                    pass
            # Clear active_action but pin recent_action so the HUD ticker
            # keeps showing the last-executed action name (with relative time).
            # now_doing reverts to the current high-level state label so the
            # holographic now_doing ring doesn't get stuck on EXECUTING: ...
            _write_hud_state(
                active_action="",
                recent_action=name,
                recent_action_at=time.time(),
                now_doing=_now_doing_label(_current_state_label[0]),
            )
        return ""

    cleaned = _ACTION_RE.sub(_runner, reply).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    # Hallucinated-action detector: if the reply *claims* to have done
    # something but no [ACTION: ...] token actually ran, the LLM is faking
    # execution. Log it and append a synthetic informative result so the
    # follow-up loop surfaces the slip back to the LLM and gives it a chance
    # to either emit the real token or admit the failure.
    if not results:
        low = cleaned.lower()
        matched = next((p for p in _ACTION_CLAIM_PHRASES if p in low), None)
        if matched:
            warn = (
                f"reply claims '{matched.strip()}' but no [ACTION: ...] token "
                "was emitted — JARVIS appears to have hallucinated execution"
            )
            print(f"  [validation] {warn}")
            results.append(("_unverified_claim", warn, True))

    # Continuation enforcer: when at least one action *did* run, scan the
    # prose for chained-intent phrases ("and read it to you", "then I'll
    # take a screenshot") whose corresponding [ACTION:] token was *not*
    # emitted. Each dropped step becomes a synthetic informative result so
    # the follow-up loop re-prompts the LLM and the chain actually
    # completes instead of stalling halfway through.
    if results:
        emitted_names = {n for (n, _, _) in results}
        dropped = _detect_dropped_steps(reply, emitted_names)
        if dropped:
            promised_total = len(emitted_names) + len(dropped)
            print(f"  [continuation_enforcer] declared ~{promised_total} step(s), "
                  f"emitted {len(emitted_names)} "
                  f"({', '.join(sorted(emitted_names))}); missing: "
                  f"{', '.join(f'{a} ({d})' for a, d in dropped)}")
            for action_name, desc in dropped:
                warn = (
                    f"reply promised to {desc} (expected [ACTION: {action_name}, ...]) "
                    "but no such token was emitted — chain step dropped"
                )
                results.append(("_dropped_step", warn, True))

    # When mission narration fired, the per-step cues replaced the LLM's
    # prose. Drop the cleaned text so the main loop doesn't immediately
    # re-speak whatever narration the LLM also tried to inline.
    if _narrate:
        cleaned = ""

    # If any JARVIS-pushback objection fired, the user should hear the
    # objection — not the LLM's "I'll close them all, sir." prelude. Replace
    # the spoken reply with the accumulated objections (one per deferred
    # action, joined with a space). Confirmation handling stays on the same
    # _pending_confirmation queue; "yes" will run the deferred actions.
    if _pushback_objections:
        cleaned = " ".join(_pushback_objections)

    return cleaned, results


def get_followup_response(action_results: list[tuple[str, str]]) -> str:
    """After informational actions ran, ask the LLM to actually answer the
    user's original question using the action results."""
    summary = "\n".join(f"- [{name}] returned: {result}" for name, result in action_results)
    extra = (
        "(System: the actions in your previous response just ran. Here is what "
        "they returned:\n\n"
        f"{summary}\n\n"
        "Continue working toward completing the original task. Important rules:\n"
        "- If a result is from [_unverified_claim], it means your previous "
        "reply *said* you did something (e.g. 'Restarting now, sir.') but "
        "you did NOT emit an [ACTION: ...] token, so nothing actually "
        "happened. Either emit the correct [ACTION: ...] token now to "
        "actually do it, or briefly admit to the user that you cannot "
        "perform that action — do not claim success again.\n"
        "- If a result is from [_dropped_step], it means your previous reply "
        "*promised* to do another step in the chain (e.g. 'and read it to "
        "you', 'then I'll take a screenshot') but you did NOT emit the "
        "[ACTION: ...] token for that step. Emit the missing token now to "
        "finish the chain — don't stop mid-task and don't re-narrate the "
        "promise without backing it with a real action token.\n"
        "- If you just opened a URL or ran a web search, use [ACTION: see_screen, ...] "
        "to read what loaded before responding — don't stop after just opening the page.\n"
        "- If the result shows an error or dead end, try an alternative approach.\n"
        "- If a result begins with a JARVIS-voice failure line (e.g. "
        "\"Windows is being precious about that one, sir.\" / \"It seems "
        "the internet has opinions today, sir.\") followed by "
        "\"(action failed; class=...; ...)\", the leading sentence is "
        "already in-character — speak it as-is or paraphrase lightly. The "
        "\"class=\" hint tells you what went wrong (network / permission / "
        "parse / app_not_found / timeout / com / ui_automation / io / "
        "unknown) — use it to decide whether to retry, try an alternative, "
        "or just relay the failure to the user.\n"
        "- Only stop and reply to the user when you have actually found the answer "
        "or completed the task. Don't give up mid-chain.\n"
        "- When you do have the answer, reply conversationally. Don't paste raw output.)"
    )

    # Carry the tone of the original user turn into the follow-up so the
    # register doesn't snap back to default mid-chain (e.g. user barks
    # "stop!" → action runs → JARVIS suddenly returns to flowery prose).
    # Also carry the voice-mood route so stressed/late-night/excited
    # register persists across action chains.
    _route = _last_voice_route[0] or {"addendum": ""}
    # Mirror the agent-mode addendum injection from _call_llm so the
    # follow-up turn stays in the same register as the primary turn
    # (otherwise self-critique would silently drop on the second pass).
    _mode_add = ""
    try:
        from core.mode_router import system_prompt_addendum as _mode_addendum
        _mode_add = _mode_addendum()
    except Exception:
        pass
    sys_prompt_now = (
        _system_prompt
        + _tone_system_addendum(_last_user_tone[0])
        + _route.get("addendum", "")
        + _mode_add
    )

    try:
        if AI_BACKEND == "claude":
            import anthropic
            msgs = list(conversation_history) + [{"role": "user", "content": extra}]
            if _llm_client is not None:
                return _llm_client.complete(
                    model=CLAUDE_MODEL, max_tokens=400,
                    system=sys_prompt_now, messages=msgs,
                    timeout=_ANTHROPIC_TIMEOUT_S,
                )
            msg = anthropic.Anthropic(timeout=_ANTHROPIC_TIMEOUT_S).messages.create(
                model=CLAUDE_MODEL, max_tokens=400,
                system=sys_prompt_now, messages=msgs,
            )
            return msg.content[0].text
        elif AI_BACKEND == "ollama":
            import ollama
            msgs = ([{"role": "system", "content": sys_prompt_now}]
                    + list(conversation_history)
                    + [{"role": "user", "content": extra}])
            resp = ollama.chat(model=OLLAMA_MODEL, messages=msgs)
            return resp["message"]["content"]
    except Exception as e:
        # Cloud unavailable (e.g. the monthly usage cap) — keep the action
        # chain alive on the local model instead of dropping it with a
        # spoken "(follow-up failed …)". Mirrors _call_llm / _llm_quick.
        local = _call_local_llm(
            sys_prompt_now,
            list(conversation_history) + [{"role": "user", "content": extra}],
            max_tokens=400,
        )
        if local:
            return local
        print(f"  [follow-up] cloud failed and no local fallback "
              f"({type(e).__name__}: {e})")
        return ""
    return ""


def handle_autocorrect_disambig_response(user_text: str) -> bool:
    """
    If the autocorrect layer asked 'did you mean X or Y' on the previous
    turn, interpret this utterance as the user's pick and run the chosen
    action. Returns True if we consumed the message (main loop should skip
    normal LLM dispatch).

    Resolution rules, in order:
      * literal mention of one of the two candidate names → run that one
      * 'yes' / 'first' / 'the first' / 'one' / candidate-1 keyword → primary
      * 'second' / 'the other' / 'b' / candidate-2 keyword       → secondary
      * 'no' / 'neither' / 'cancel' / 'nevermind'                → cancel
      * anything else                                            → cancel
        and pass through to normal dispatch (caller checks return value)
    """
    if not _pending_autocorrect_choice:
        return False
    choice = _pending_autocorrect_choice[0]
    primary_name, primary_arg = choice["primary"]
    secondary_name, secondary_arg = choice["secondary"]
    t = user_text.strip().lower()

    # Strip punctuation off the edges so 'yes.' / 'second!' still parse.
    t = t.strip(".!?,;: ")

    # Heuristic: if either action name (or its main token) is mentioned
    # outright, that's the strongest signal. Check the longer name first
    # so e.g. 'ambient_listen_start' beats 'ambient_listen_stop' on the
    # 'start' / 'stop' tail rather than matching the shared prefix.
    def _name_mentioned(action_name: str) -> bool:
        n = action_name.lower().replace("_", " ")
        return n in t or action_name.lower() in t

    picked: tuple[str, str] | None = None
    reason = ""

    if _name_mentioned(primary_name) and not _name_mentioned(secondary_name):
        picked = (primary_name, primary_arg)
        reason = "named primary"
    elif _name_mentioned(secondary_name) and not _name_mentioned(primary_name):
        picked = (secondary_name, secondary_arg)
        reason = "named secondary"
    elif t in ("yes", "yeah", "yep", "yup", "sure", "ok", "okay",
               "first", "the first", "first one", "former", "one",
               "do it", "go ahead", "proceed", "confirm"):
        picked = (primary_name, primary_arg)
        reason = "affirmative -> primary"
    elif t in ("second", "the second", "second one", "latter", "two",
               "the other", "the other one", "other", "b"):
        picked = (secondary_name, secondary_arg)
        reason = "explicit secondary"
    elif t in ("no", "nope", "nah", "neither", "cancel", "nevermind",
               "never mind", "skip", "forget it", "stop"):
        picked = None
        reason = "negative -> cancel"
    else:
        # Unmatched response — treat as cancellation but don't consume the
        # turn; the user probably issued a fresh command rather than
        # picking. Return False so normal dispatch handles it.
        print(f"  [autocorrect-disambig] unrelated reply "
              f"{user_text!r}; cancelling pending choice")
        _pending_autocorrect_choice.clear()
        return False

    _pending_autocorrect_choice.clear()
    if picked is None:
        print(f"  [autocorrect-disambig] {reason}")
        try:
            _speak("Cancelled, sir.")
        except Exception as _e:
            print(f"  [autocorrect-disambig] cancel TTS failed: {_e}")
        return True

    name, arg = picked
    fn = ACTIONS.get(name)
    if fn is None:
        # Action vanished between turns (skill unloaded?). Bail out cleanly.
        print(f"  [autocorrect-disambig] picked {name!r} but ACTIONS lost it")
        try:
            _speak(f"I can't run `{name}` anymore, sir.")
        except Exception:
            pass
        return True
    # Safety gate: this path calls fn(arg) directly, bypassing the
    # _needs_confirmation / _jarvis_pushback layers that parse_and_run_actions
    # applies. A destructive action surfaced via a fuzzy typo must not fire
    # without confirmation — refuse and make the user re-issue it explicitly
    # so it goes through the normal confirmation/pushback path (mirrors the
    # replay_last_action destructive-action refusal).
    if name in _DESTRUCTIVE_REPLAY_ACTIONS:
        print(f"  [autocorrect-disambig] refusing destructive pick {name!r} "
              "without confirmation")
        msg = (f"I won't run `{name}` from a guessed match, sir — "
               "please issue that command explicitly.")
        try:
            _speak(msg)
        except Exception as _e:
            print(f"  [autocorrect-disambig] refusal TTS failed: {_e}")
        return True
    print(f"  [autocorrect-disambig] {reason} -> running {name}({arg!r})")
    try:
        _speak(f"Running `{name}`, sir.")
    except Exception as _e:
        print(f"  [autocorrect-disambig] confirm TTS failed: {_e}")
    try:
        res = fn(arg)
        print(f"  [action] {name}: {res}")
        record_session_action(name, arg)
        if name != "replay_last_action":
            record_action_history(name, arg, res)
    except Exception as e:
        print(f"  [action] {name} failed: {e}")
    return True


def handle_confirmation_response(user_text: str) -> bool:
    """
    If we're waiting on a confirmation, interpret this user message as either
    yes/no. Returns True if we consumed the message (so main loop should
    skip the normal LLM call). Bobert speaks brief feedback either way.
    """
    if not _pending_confirmation:
        return False
    t = user_text.strip().lower()
    affirmative = any(t.startswith(w) for w in ("yes", "confirm", "do it", "go ahead", "proceed"))
    if affirmative:
        count = len(_pending_confirmation)
        print(f"  [confirm] User confirmed — executing {count} pending action(s)")
        executed = []
        while _pending_confirmation:
            name, arg = _pending_confirmation.pop(0)
            fn = ACTIONS.get(name)
            if not fn:
                continue
            try:
                res = fn(arg)
                executed.append(name)
                print(f"  [action] {name}: {res}")
            except Exception as e:
                print(f"  [action] {name} failed: {e}")
        # Audible feedback so the user knows the action ran
        feedback = "Done." if len(executed) == 1 else f"Done — ran {len(executed)} actions."
        _speak(feedback)
    else:
        count = len(_pending_confirmation)
        print(f"  [confirm] User declined — cancelling {count} pending action(s)")
        _pending_confirmation.clear()
        _speak("Cancelled.")
    return True


# ──────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────

def _apply_quip_layer(spoken_text: str,
                      action_results: list[tuple[str, str, bool]]) -> str:
    """Run a confirmation reply through core.tts.jarvis_quip_layer using the
    most-recently executed real action as category context. No-op if the TTS
    layer isn't loaded, if there were no actions, or if `spoken_text` doesn't
    meet the quip layer's short-confirmation criteria."""
    if not spoken_text or _tts_layer is None:
        return spoken_text
    if not hasattr(_tts_layer, "jarvis_quip_layer"):
        return spoken_text
    primary: str | None = None
    for n, _r, _i in reversed(action_results or []):
        if not n:
            continue
        ln = n.lower()
        if ln.startswith("_") or "unknown" in ln:
            continue
        primary = n
        break
    # Skip the quip layer for fire-and-exit actions (restart / upgrade /
    # start_overnight_upgrade). Their handlers schedule os._exit(0) on a
    # ~1.5 s timer, so any quip the layer appends to "Restarting now, sir."
    # would be cut off mid-word when the process dies. The confirmation is
    # the last thing the user will hear — let it land cleanly.
    if primary and primary.lower() in _FIRE_AND_EXIT_ACTIONS:
        return spoken_text
    try:
        return _tts_layer.jarvis_quip_layer(spoken_text, primary)
    except Exception as _e:
        print(f"  [quip] layer skipped: {_e}")
        return spoken_text


def _speak_verbatim_results(
    action_results: list[tuple[str, str, bool]],
    already_spoken: str = "",
) -> set[str]:
    """Speak the result of any SPEAK_RESULT_VERBATIM_ACTIONS action directly,
    so informational answers that are already finished sentences (version_info,
    system_pulse, …) are voiced instead of merely logged.

    These actions are deliberately NOT in INFORMATIVE_ACTIONS — their result is
    a complete, user-facing sentence that needs no LLM restatement — so without
    this the main loop's follow-up never fires and the answer is dropped (the
    2026-06-03 "you didn't speak it" bug). Speaks through the same _speak() path
    the inline reply uses, right after it.

    Guardrails against double-speak / noise:
      * Only acts on actions in the explicit verbatim allow-list, so side-effect
        actions (play_music / volume_up / set_timer) are never touched.
      * Skips a result already contained in ``already_spoken`` (e.g. the LLM
        inlined the answer in its prose, or the quip layer already carried it),
        comparing case-insensitively so a fresh re-speak is suppressed.
      * Skips failure results — those still route through the failure follow-up
        loop so JARVIS reports the problem rather than reading a raw error.
      * Skips empty / sentinel results.

    Returns the set of action names whose result was spoken, so the caller can
    mark them handled and avoid any later re-speak.
    """
    spoken_names: set[str] = set()
    if not action_results:
        return spoken_names
    prior = (already_spoken or "").lower()
    fail_markers = tuple(m.lower() for m in FAILURE_MARKERS)
    for name, result, _is_info in action_results:
        if not name or name.lower() not in SPEAK_RESULT_VERBATIM_ACTIONS:
            continue
        text = (result or "").strip()
        if not text:
            continue
        low = text.lower()
        # A failure ("could not read version info: …") belongs to the failure
        # follow-up path, not a verbatim read-back.
        if any(m in low for m in fail_markers):
            continue
        # Already voiced as part of the inline reply / quip — don't repeat it.
        if low in prior:
            continue
        try:
            _speak(text)
            spoken_names.add(name.lower())
            # Fold into the running spoken text so a duplicate result from a
            # second action in the same reply isn't spoken twice.
            prior = f"{prior} {low}".strip()
        except Exception as _e:
            print(f"  [verbatim_result] speak failed for {name}: {_e}")
    return spoken_names


# _speak() can be called from multiple threads: the main turn loop, the
# timer-skill, the credits monitor, the bambu announcer, the mid-task status
# timer in parse_and_run_actions, and the tray-applet drainer. Without a
# lock, two callers can race on the global edge-tts loop, the sd.play() call,
# and the _last_intent_override / _last_wry mutable singletons, producing
# overlapping audio and corrupted prosody state. This lock serialises the
# entire speak operation (synthesise + lipsync playback).
_SPEAK_LOCK = threading.Lock()

_MARKDOWN_FOR_SPEECH_RE_BOLD   = re.compile(r"\*\*([^*]+?)\*\*")
_MARKDOWN_FOR_SPEECH_RE_ITALIC = re.compile(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])")
_MARKDOWN_FOR_SPEECH_RE_CODE   = re.compile(r"`([^`\n]+?)`")
_MARKDOWN_FOR_SPEECH_RE_LINK   = re.compile(r"\[([^\]]+?)\]\([^)]+\)")
_MARKDOWN_FOR_SPEECH_RE_HEAD   = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MARKDOWN_FOR_SPEECH_RE_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MARKDOWN_FOR_SPEECH_RE_HRULE  = re.compile(r"^[-=_]{3,}\s*$", re.MULTILINE)


def _strip_markdown_for_speech(text: str) -> str:
    """Convert markdown into bare prose so edge-tts doesn't read 'asterisk
    asterisk' or 'underscore' aloud. Targets the formatting JARVIS commonly
    emits when reading back task descriptions, upgrade summaries, or other
    file content — NOT the [intent:xxx] / [wry] markers, which are handled
    separately before this runs."""
    if not text:
        return text
    # **bold**  →  bold
    text = _MARKDOWN_FOR_SPEECH_RE_BOLD.sub(r"\1", text)
    # *italic*  →  italic   (only when surrounded by non-word — protects
    # snake_case * literals and ordinary asterisks inside words)
    text = _MARKDOWN_FOR_SPEECH_RE_ITALIC.sub(r"\1", text)
    # `code`    →  code
    text = _MARKDOWN_FOR_SPEECH_RE_CODE.sub(r"\1", text)
    # [text](url) →  text
    text = _MARKDOWN_FOR_SPEECH_RE_LINK.sub(r"\1", text)
    # # heading  →  heading
    text = _MARKDOWN_FOR_SPEECH_RE_HEAD.sub("", text)
    # - bullet   →  bullet (also covers leading * and +)
    text = _MARKDOWN_FOR_SPEECH_RE_BULLET.sub("", text)
    # --- horizontal rule → drop the line
    text = _MARKDOWN_FOR_SPEECH_RE_HRULE.sub("", text)
    # Snake_case identifiers — replace underscores with spaces so
    # `BAMBU_PRINTER_IP` reads "bambu printer i p" instead of
    # "bambu underscore printer underscore i p".
    text = re.sub(r"(?<=\w)_(?=\w)", " ", text)
    # Stray backticks left behind by malformed code blocks
    text = text.replace("`", "")
    # File-extension dots and version numbers stay (they read OK), but kill
    # the ugly "✓ DONE — " sentinel and surrounding decorative bullets that
    # show up in upgrade summaries
    text = text.replace("✓ DONE —", "")
    text = text.replace("✓ DONE -", "")
    # Collapse runs of whitespace + any leftover stray asterisks
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _speak(text: str, volume_scale: float = 1.0, mood: str | None = None):
    """Run a string through TTS + lip-sync + state changes. Used by both
    reactive responses and proactive idle comments.

    `volume_scale` < 1.0 attenuates the rendered audio in-place before
    playback (used by context_aware_greeting for the 'looking away' case).

    `mood` (optional) opts the caller into the voice_mood layer
    (core/voice_mood_selector). One of: calm_efficient / urgent_clipped /
    dry_amused / concerned_soft. When set, _resolve_tts_preset uses the
    matching preset between the [intent:xxx] override and the
    emotion-tracker / user-tone fallbacks. None preserves legacy behaviour
    (no mood override) so untouched skills are unaffected.

    Three LLM-emitted markers are honoured at the start of `text`:
        [intent:xxx]  → routes prosody to a named preset (see _INTENT_PRESETS)
        [wry]         → forces the 'wry' preset and inserts a beat before the
                        punchline so the joke lands
        [mood:xxx]    → equivalent to passing mood= explicitly; lets skills
                        that return text via the dispatcher opt into a mood
                        without needing a reference to _speak()
    All three tags are stripped before synthesis and never spoken aloud.
    They may appear in any order at the start; the others (if present) are
    parsed on the stripped remainder. The mood= kwarg takes precedence over
    a [mood:xxx] tag.

    Markdown formatting (asterisks, underscores, backticks, headings, bullet
    markers) is stripped via _strip_markdown_for_speech AFTER tag parsing
    so the LLM's tags survive and the spoken text doesn't read raw punctuation."""
    global last_speech_time
    # Boot-window speech throttle. The post-upgrade announcement, changelog
    # announcement, and greeting all fire in quick succession during the first
    # few seconds, raising the chance of a sounddevice race when synthesis
    # threads collide. Enforce a 1-second minimum gap between _speak calls for
    # the first 30 seconds after boot so the burst spreads out. See
    # jarvis_todo.md 2026-05-29 19:20.
    _now = time.time()
    if _now - _session_start_time < 30.0:
        _gap = _now - last_speech_time
        if _gap < 1.0:
            time.sleep(1.0 - _gap)
    # NOTE: the parsed intent / wry / mood are kept in LOCALS through this
    # pre-lock window and only published to the shared _last_* cells INSIDE
    # _SPEAK_LOCK right before synthesise() consumes them. Writing them here
    # would let a concurrent background _speak (tray/proactive) overwrite the
    # cells before this thread reaches the lock, so the other thread would
    # synthesise with our prosody (and vice-versa). 2026-05-30 audit.
    intent, spoken_text = _parse_intent_tag(text)
    # Parse a leading [mood:xxx] tag off the stripped remainder. The mood=
    # kwarg wins when the caller provided one — that's the explicit opt-in
    # path (vip_intercept, self_diagnostic) and shouldn't be overridden
    # by a leftover tag in the queued message.
    tag_mood, spoken_text = _parse_mood_tag(spoken_text)
    chosen_mood = mood if mood in _VOICE_MOOD_NAMES else (tag_mood if tag_mood in _VOICE_MOOD_NAMES else None)
    # Publish the parsed intent tag for the holographic HUD v2 INTENT row.
    # Spec: jarvis_todo.md 2026-05-29 09:18 (holographic_hud_v2). Write
    # an empty string when no tag was emitted so the panel decays back
    # to "—" rather than showing a stale value from the previous reply.
    try:
        _write_hud_state(last_intent_tag=(intent or ""))
    except Exception:
        pass
    # Tray "Mute TTS" toggle — short-circuit before synthesis so the audio
    # device isn't even opened. We still record the speech timestamp so
    # idle-proactive cadence stays paced as if JARVIS had spoken. We do NOT
    # touch the shared _last_* cells here: this path never set them (they're
    # only published under _SPEAK_LOCK), so clearing them would stomp on a
    # concurrent speaker that legitimately owns them. Lip-sync state isn't
    # touched because the play step never runs.
    if _tts_muted[0]:
        last_speech_time = time.time()
        return
    if _tts_layer is not None:
        try:
            wry_flag, spoken_text = _tts_layer.parse_wry_tag(spoken_text)
            # Allow the LLM to put [wry] first too — re-run the intent + mood
            # parsers against the stripped remainder so [wry][intent:xxx] and
            # [wry][mood:xxx] still route.
            if wry_flag and not intent:
                intent, spoken_text = _parse_intent_tag(spoken_text)
            if wry_flag and chosen_mood is None and mood not in _VOICE_MOOD_NAMES:
                late_mood, spoken_text = _parse_mood_tag(spoken_text)
                if late_mood:
                    chosen_mood = late_mood
        except Exception:
            wry_flag = False
    else:
        wry_flag = False
    # Strip markdown punctuation AFTER intent/wry tag parsing so the LLM's
    # tags survive (they're already gone by this point) but the visible
    # asterisks / underscores / backticks / bullets the LLM may have left
    # in a file readback don't get spoken as "asterisk asterisk".
    spoken_text = _strip_markdown_for_speech(spoken_text)
    # Nothing audible left after stripping tags + markdown (the reply was only
    # an [ACTION:]/markup token, or whitespace). Skip the entire TTS path — an
    # empty string sent to synthesise() fails edge-tts → pyttsx3
    # (AssertionError) → SAPI5 ("empty text for SAPI5 render") → silence,
    # spamming the logs and spinning up the duck/barge-in machinery for
    # nothing (seen in the 2026-05-30 log audit). Mirror the mute-path
    # bookkeeping and return.
    if not (spoken_text or "").strip():
        last_speech_time = time.time()
        _last_intent_override[0] = None
        _last_wry[0] = False
        _last_mood[0] = None
        return
    # Publish the stripped text so the holographic overlay's center text
    # panel can render "what JARVIS just said" alongside last_transcript.
    if spoken_text:
        _write_hud_state(last_spoken=spoken_text, last_spoken_at=time.time())
    # Blue/green: a staging instance must not open the audio device — that
    # device belongs to prod for the duration of the smoke test. Route the
    # spoken text to data_staging/replies.jsonl instead so the pipeline
    # can verify the dispatcher produced a sensible response.
    if _is_staging():
        try:
            import staging_instance as _stg
            _stg.record_reply(spoken_text or text,
                              kind="tts",
                              extra={"intent": intent, "wry": wry_flag, "mood": chosen_mood})
        except Exception as _re:
            print(f"  [staging] record_reply failed: {_re}")
        last_speech_time = time.time()
        # No shared-cell clear here either — staging never published them.
        return
    # Serialize the synthesise+play cycle so concurrent callers (timer
    # skill, mid-task-status timer, tray drainer, main loop) don't garble
    # each other's audio or corrupt the shared intent/wry globals. Publish
    # the parsed prosody from our locals into the shared _last_* cells INSIDE
    # the lock, immediately before synthesise()/_resolve_tts_preset read them,
    # so set + consume is atomic w.r.t. any concurrent _speak. The finally
    # clears them again so the next utterance starts clean.
    with _SPEAK_LOCK:
        try:
            _last_intent_override[0] = intent
            _last_wry[0] = wry_flag
            _last_mood[0] = chosen_mood
            set_state("speaking")
            audio_out, sr = synthesise(spoken_text)
            if volume_scale != 1.0:
                try:
                    audio_out = (audio_out.astype(np.float32) * float(volume_scale)).astype(audio_out.dtype)
                except Exception:
                    pass
            play_with_lipsync(audio_out, sr)
            last_speech_time = time.time()
            set_state("idle")
        except Exception as _spk_err:
            # A PortAudio/device hiccup (e.g. a stale output-device index
            # after a hot-swap, or a synthesis error) must NOT propagate out
            # of _speak into the main loop / tray drainer / proactive turn.
            # Lose the line, log it, recover to idle. 2026-05-30 audit.
            print(f"  [speak] playback failed: "
                  f"{type(_spk_err).__name__}: {_spk_err}")
            try:
                set_state("idle")
            except Exception:
                pass
        finally:
            _last_intent_override[0] = None
            _last_wry[0] = False
            _last_mood[0] = None
            # play_with_lipsync sets _tts_playback_active True before its own
            # try/finally that resets it. If it raised during setup (e.g. a
            # thread-create RuntimeError under load) BEFORE reaching that
            # finally, the flag would be left stuck True — permanently disabling
            # _refresh_devices' PortAudio hotplug reinit for the rest of the
            # session. _speak is the sole caller and nothing is playing once it
            # returns, so force the guard back to False here.
            _tts_playback_active[0] = False


def _do_proactive_turn(memory: dict):
    """Generate and speak a JARVIS-style spontaneous comment."""
    pause_face_tracking()
    set_state("thinking")

    stop_evt = threading.Event()
    anim     = threading.Thread(target=_thinking_loop, args=(stop_evt,), daemon=True)
    anim.start()
    text = generate_proactive_comment()
    stop_evt.set()
    anim.join()

    if not text:
        set_state("idle")
        resume_face_tracking()
        return

    print(f"\n  [proactive]")
    print(f"  JARVIS: {text}")
    conversation_history.append({"role": "assistant", "content": text})
    spoken, _proactive_results = parse_and_run_actions(text)
    spoken = _apply_quip_layer(spoken, _proactive_results)
    _speak(spoken)
    resume_face_tracking()


_LOCK_FILE = _BLUE_GREEN_PATHS["lock_file"]


def _enforce_singleton():
    """If another JARVIS is already running, refuse to start a second copy.
    Writes our PID to jarvis.lock; on next launch, checks if the recorded PID
    is still alive and exits cleanly with a message if so.

    The top-of-module early-boot block (_early_boot_singleton_lock) already
    wrote our PID into the lock before the heavy imports ran — this call is
    the belt-and-suspenders re-check from main(). When the lock already
    names us, just return."""
    # If we hold the authoritative OS byte-range lock, we ARE the singleton —
    # nothing more to check. Re-opening the file for write here would also be
    # unsafe: a "w" truncate on the locked file from a second handle can fight
    # the lock we hold at offset 4096. Defer entirely to the held lock.
    if _SINGLETON_HELD_FD is not None:
        return
    if os.path.exists(_LOCK_FILE):
        # Mirror the early-boot lock's retry-aware read so we don't
        # mistake an in-flight write by a sibling boot for a stale lock.
        # _read_lock_pid returns -1 if the file vanished, 0 if it's
        # truly empty/unparseable after retries, or the recorded PID.
        old_pid = _read_lock_pid(_LOCK_FILE)
        if old_pid == os.getpid():
            return
        if old_pid > 0:
            # Check if the process is still alive (Windows: tasklist, Unix: signal 0)
            # Fail-safe: on any ambiguity, assume the old process is alive
            # rather than risk spawning a duplicate. Matches the early-boot
            # lock's posture (see _pid_alive in _early_boot_singleton_lock).
            still_running = True
            if sys.platform == "win32":
                try:
                    r = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {old_pid}", "/FO", "CSV", "/NH"],
                        capture_output=True, text=True, timeout=5,
                    )
                    still_running = (str(old_pid) in r.stdout and "python" in r.stdout.lower())
                except subprocess.TimeoutExpired:
                    still_running = True
                except Exception:
                    still_running = True
            else:
                try:
                    os.kill(old_pid, 0)
                    still_running = True
                except ProcessLookupError:
                    still_running = False
                except PermissionError:
                    still_running = True
            if still_running:
                print(f"\n[singleton] Another JARVIS is already running (PID {old_pid}).")
                print("[singleton] Refusing to start a duplicate. Quit the existing one first.")
                print(f"[singleton] To force-clear the lock, delete: {_LOCK_FILE}\n")
                sys.exit(0)
        # old_pid is -1 (file gone) or 0 (truly empty) or a dead PID —
        # fall through and overwrite with our own.

    # Write our own PID
    try:
        with open(_LOCK_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except OSError:
        pass   # if we can't write, just proceed — singleton is best-effort


def _release_singleton():
    """Release the held OS byte-range lock on clean shutdown.

    The lock FILE is intentionally LEFT IN PLACE — the kernel byte-range lock
    (not file existence) is the single-instance guard, and the kernel drops it
    automatically the instant our fd closes or the process dies. Deleting the
    file on shutdown was the OLD behaviour that opened the duplicate-instance
    race: a force-killed process ran this handler, deleted jarvis.lock, then
    hung; a second instance found no file and booted alongside the zombie
    (observed 2026-05-30 10:58). Now we only drop the lock; a genuinely dead
    process's lock is released by the kernel, and a hung one keeps holding it.
    """
    global _SINGLETON_HELD_FD
    fd = _SINGLETON_HELD_FD
    _SINGLETON_HELD_FD = None
    if fd is None:
        return
    try:
        if sys.platform == "win32":
            import msvcrt
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl
            try:
                fcntl.lockf(fd, fcntl.LOCK_UN, 1, 0)
            except OSError:
                pass
    finally:
        try:
            os.close(fd)   # closing also drops the lock as a backstop
        except OSError:
            pass


def _overnight_upgrade_thread():  # pragma: no cover - background daemon (while-True self-improvement engine; never invoked outside a Thread)
    """Background thread: runs the self-improvement engine inside JARVIS.
    Every OVERNIGHT_CYCLE_GAP_HOURS (minimum), if JARVIS has been idle for
    OVERNIGHT_IDLE_MINUTES, asks Claude to invent improvements, writes them
    to jarvis_todo.md, and fires the upgrade pipeline."""
    import importlib
    _log_prefix = "  [overnight]"
    last_cycle_at = 0.0

    def _is_idle():
        # Use last_speech_time — only updated when the user actually speaks.
        # Do NOT use log file mtime — the overnight engine's own print statements
        # update the log and would falsely mark the user as active mid-cycle.
        return (time.time() - last_speech_time) >= OVERNIGHT_IDLE_MINUTES * 60

    def _enough_gap():
        return (time.time() - last_cycle_at) / 3600 >= OVERNIGHT_CYCLE_GAP_HOURS

    print(f"{_log_prefix} self-improvement engine active "
          f"(idle={OVERNIGHT_IDLE_MINUTES}min, gap={OVERNIGHT_CYCLE_GAP_HOURS}h)")

    # Persistence resume: if the user said "goodnight" in a previous session
    # and the flag hasn't expired, re-arm the engine immediately so overnight
    # mode survives the kill+relaunch cycles that upgrades cause.
    try:
        if os.path.exists(OVERNIGHT_FLAG_FILE):
            _expiry = None
            try:
                with open(OVERNIGHT_FLAG_FILE, encoding="utf-8") as _f:
                    _expiry = float(_f.read().strip())
            except (ValueError, OSError, UnicodeDecodeError) as _parse_err:
                # Corrupt flag content — nuke it so we don't trip every startup.
                try:
                    os.remove(OVERNIGHT_FLAG_FILE)
                except OSError:
                    pass
                _write_hud_state(overnight_expiry=0.0)
                print(f"{_log_prefix} overnight flag corrupt ({_parse_err}) — cleared")
            if _expiry is not None:
                if time.time() < _expiry:
                    _sleep_mode[0] = True
                    _overnight_run_now.set()
                    _write_hud_state(overnight_expiry=_expiry)
                    print(f"{_log_prefix} resuming overnight mode "
                          f"(active until {time.strftime('%H:%M', time.localtime(_expiry))})")
                else:
                    try:
                        os.remove(OVERNIGHT_FLAG_FILE)
                    except OSError:
                        pass
                    _write_hud_state(overnight_expiry=0.0)
                    print(f"{_log_prefix} overnight flag expired — cleared")
    except Exception as _e:
        print(f"{_log_prefix} flag check failed: {_e}")

    time.sleep(60)  # give JARVIS a minute to fully boot before first check

    while True:
        try:
            time.sleep(60)  # check every minute
            force = _overnight_run_now.is_set()
            if force:
                _overnight_run_now.clear()
                print(f"{_log_prefix} triggered immediately (user requested)")
            else:
                if not _enough_gap():
                    continue
                if not _is_idle():
                    continue

            print(f"{_log_prefix} {'immediate' if force else 'idle threshold reached'} — starting improvement cycle")
            # Note: last_cycle_at is NOT updated here. It's only updated when
            # the cycle actually does work (fires an upgrade). Failed cycles
            # (Max throttled, no ideas) fall through to retry on the next 60s
            # poll instead of waiting OVERNIGHT_CYCLE_GAP_HOURS.

            # Import overnight_upgrade for idea generation (reuse its helpers)
            try:
                spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "overnight_upgrade.py")
                import importlib.util as _ilu
                spec = _ilu.spec_from_file_location("overnight_upgrade", spec_path)
                ou = _ilu.module_from_spec(spec)
                spec.loader.exec_module(ou)
            except Exception as e:
                print(f"{_log_prefix} failed to load overnight_upgrade: {e}")
                continue

            # ── Phase 1: implement any existing pending tasks first ──────────
            pending = ou._count_pending_tasks()
            if pending > 0:
                print(f"{_log_prefix} {pending} pending task(s) in jarvis_todo.md — running upgrade first")

                if not force and not _is_idle():
                    print(f"{_log_prefix} user became active — deferring upgrade")
                    continue

                # Capture task descriptions before upgrade_jarvis.py runs so
                # JARVIS can announce them on the next startup.
                try:
                    try:
                        with open(TODO_FILE, encoding="utf-8") as _tf:
                            _pending_tasks = re.findall(
                                r"^- \[ \] (.+)$", _tf.read(), re.MULTILINE)
                    except Exception:
                        _pending_tasks = []
                    with open(UPGRADE_SUMMARY_FILE, "w", encoding="utf-8") as _sf:
                        json.dump({
                            "upgraded_at": time.strftime("%H:%M"),
                            "tasks": _pending_tasks,
                            "syntax_ok": True,   # upgrade_jarvis.py updates this
                        }, _sf, indent=2)
                except Exception as _e:
                    print(f"{_log_prefix} couldn't write summary: {_e}")

                # ── Pipeline singleton guard ─────────────────────────────────
                # Permanent fix for the duplicate-pipeline race observed on
                # 2026-05-29 (Pipeline A: 41992-44316-49124, Pipeline B:
                # 13176-48228-44452 — both writing to the same code files).
                # The race happens when upgrade_jarvis.py --relaunch spawns a
                # fresh JARVIS mid-cycle and that JARVIS's own overnight thread
                # sees the .overnight_active flag and fires a second pipeline.
                # Scan for any running upgrade_jarvis.py; if found, skip.
                try:
                    _ps_existing = []
                    try:
                        import psutil as _psutil
                        for _proc in _psutil.process_iter(
                                attrs=["pid", "name", "cmdline"]):
                            try:
                                _cl = " ".join(_proc.info.get("cmdline") or [])
                                if "upgrade_jarvis.py" in _cl and _proc.info["pid"] != os.getpid():
                                    _ps_existing.append(_proc.info["pid"])
                            except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                                continue
                    except ImportError:
                        # psutil unavailable — fall through and risk the race
                        # rather than blocking the engine forever.
                        pass
                    if _ps_existing:
                        print(f"{_log_prefix} pipeline already running "
                              f"(PIDs {_ps_existing}) — skipping this cycle "
                              f"to prevent duplicate-pipeline race")
                        continue
                except Exception as _sg_e:
                    print(f"{_log_prefix} singleton guard error: {_sg_e}")
                    # Don't block on the guard failing — just log + proceed

                print(f"{_log_prefix} triggering upgrade pipeline for {pending} existing task(s)…")
                last_cycle_at = time.time()   # success — start the gap timer
                _act_upgrade()
                continue   # JARVIS shuts down here; new session handles Phase 2

            # ── Phase 2: no pending tasks — generate fresh ideas ────────────
            print(f"{_log_prefix} no pending tasks — generating new improvement ideas…")
            memory_summary = ou._read_memory_summary()
            features       = ou._read_codebase_features()
            logs           = ou._read_logs_tail()

            ideas = ou._generate_ideas(memory_summary, features, logs)
            print(f"{_log_prefix} {len(ideas)} idea(s) generated")
            for i, idea in enumerate(ideas, 1):
                print(f"{_log_prefix}   [{i}] {idea[:100]}")

            if not ideas:
                # Idea generation failed (Max throttled, CLI silent, etc.).
                # Don't update last_cycle_at — fall through to retry on the
                # next 60s poll. If we were force-triggered, re-arm the event
                # so retries persist instead of dying after one failure.
                print(f"{_log_prefix} no ideas (likely throttled) — will retry on next poll")
                if force:
                    _overnight_run_now.set()
                continue

            written = ou._append_tasks(ideas)
            print(f"{_log_prefix} {written} new task(s) written to jarvis_todo.md")

            if written == 0:
                print(f"{_log_prefix} all duplicates — skipping upgrade")
                if force:
                    _overnight_run_now.set()
                continue

            # Final activity check before firing upgrade
            if not force and not _is_idle():
                print(f"{_log_prefix} user became active mid-cycle — deferring upgrade")
                continue

            try:
                with open(UPGRADE_SUMMARY_FILE, "w", encoding="utf-8") as _sf:
                    json.dump({
                        # Consistent with Phase 1 + upgrade_jarvis.py — short time-only
                        # form so the spoken announcement reads naturally ("ran at 03:45").
                        "upgraded_at": time.strftime("%H:%M"),
                        "tasks":       ideas[:8],
                        "syntax_ok":   True,   # upgrade_jarvis.py overwrites with the real result
                    }, _sf, indent=2)
            except Exception as _e:
                print(f"{_log_prefix} couldn't write summary: {_e}")

            print(f"{_log_prefix} triggering upgrade pipeline for {written} new task(s)…")
            last_cycle_at = time.time()   # success — start the gap timer
            _act_upgrade()

        except Exception as e:
            print(f"{_log_prefix} error in cycle: {e}")


def _move_console_to_monitor(monitor_name: str):
    """Move the JARVIS console window to the named monitor at startup.
    Uses GetConsoleWindow → SetWindowPos so the window lands on the right
    screen immediately, before any output appears. Silent on any failure."""
    if not monitor_name or monitor_name not in MONITORS:
        return
    mx, my, mw, mh = MONITORS[monitor_name]
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if not hwnd:
            return
        # SWP_NOSIZE (0x0001) — keep existing window size, just move it
        SWP_NOSIZE = 0x0001
        ctypes.windll.user32.SetWindowPos(hwnd, 0, mx, my, 0, 0, SWP_NOSIZE)
    except Exception:
        pass


# Built-in Windows High Performance plan GUID — stable across Win10/11.
_HIGH_PERF_GUID = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
# Captured at boot so shutdown can restore whatever the user had before.
_prior_power_plan_guid: str | None = None


def _get_active_power_plan_guid() -> str | None:
    try:
        out = subprocess.check_output(
            ["powercfg", "/getactivescheme"],
            stderr=subprocess.STDOUT, text=True, timeout=5,
        )
    except Exception:
        return None
    # Format: "Power Scheme GUID: <guid>  (<friendly name>)"
    m = re.search(r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                  r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})", out)
    return m.group(1) if m else None


def _set_power_plan(guid: str) -> bool:
    try:
        subprocess.run(
            ["powercfg", "/setactive", guid],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return True
    except Exception:
        return False


def _activate_high_performance_plan() -> None:
    """Switch Windows to High Performance at boot; remember prior plan for shutdown."""
    global _prior_power_plan_guid
    try:
        prior = _get_active_power_plan_guid()
        if not prior:
            print("  [power] could not read active plan — skipping switch")
            return
        if prior.lower() == _HIGH_PERF_GUID.lower():
            print("  [power] already on High Performance")
            _prior_power_plan_guid = prior
            return
        if _set_power_plan(_HIGH_PERF_GUID):
            _prior_power_plan_guid = prior
            print(f"  [power] switched to High Performance "
                  f"(was {prior[:8]}…) — will restore on shutdown")
        else:
            print("  [power] failed to activate High Performance plan")
    except Exception as e:
        print(f"  [power] activate failed: {e}")


def _restore_prior_power_plan() -> None:
    """Restore the power plan that was active before JARVIS started."""
    try:
        if not _prior_power_plan_guid:
            return
        if _prior_power_plan_guid.lower() == _HIGH_PERF_GUID.lower():
            return  # nothing to restore — we never changed it
        if _set_power_plan(_prior_power_plan_guid):
            print(f"  [power] restored prior plan ({_prior_power_plan_guid[:8]}…)")
        else:
            print("  [power] failed to restore prior power plan")
    except Exception as e:
        print(f"  [power] restore failed: {e}")


# Standard locations where cublas64_12.dll lives on a healthy CUDA 12 box.
# Checked at boot (`_preflight_cublas_check`) so we can short-circuit the
# slow CUDA-load → DLL-error → CPU-fallback dance in _ensure_whisper().
# We expand env vars so the per-version Toolkit path resolves on any CUDA
# v12.x install; the pip-installed `nvidia-cublas-cu12` wheel landing under
# site-packages is the cheaper case to hit first.
_CUBLAS_DLL_NAME = "cublas64_12.dll"
_CUBLAS_SEARCH_GLOBS = (
    # pip-installed nvidia-cublas-cu12 (under each site-packages on PATH).
    os.path.join("nvidia", "cublas", "bin", _CUBLAS_DLL_NAME),
    # CUDA Toolkit 12.x system install — versions vary, glob handles it.
    os.path.join("CUDA", "v12*", "bin", _CUBLAS_DLL_NAME),
)


def _find_cublas_dll() -> str | None:
    """Walk the standard CUDA 12 install locations looking for
    cublas64_12.dll. Returns the first hit or None.

    Checked: site-packages/nvidia/cublas/bin (pip-installed wheel),
    every directory on $PATH, and `%CUDA_PATH%` + the per-version
    NVIDIA GPU Computing Toolkit dirs under Program Files."""
    import glob

    # 1. Each site-packages dir (covers both pip-install --user and venv).
    try:
        import site
        site_dirs: list[str] = []
        try:
            site_dirs.extend(site.getsitepackages() or [])
        except Exception:
            pass
        try:
            usp = site.getusersitepackages()
            if usp:
                site_dirs.append(usp)
        except Exception:
            pass
        for sp in site_dirs:
            cand = os.path.join(sp, "nvidia", "cublas", "bin", _CUBLAS_DLL_NAME)
            if os.path.isfile(cand):
                return cand
    except Exception:
        pass

    # 2. Anywhere on PATH (CUDA Toolkit install adds its bin dir there).
    for p in (os.environ.get("PATH") or "").split(os.pathsep):
        if not p:
            continue
        cand = os.path.join(p, _CUBLAS_DLL_NAME)
        if os.path.isfile(cand):
            return cand

    # 3. CUDA_PATH env var (set by the Toolkit installer).
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cand = os.path.join(cuda_path, "bin", _CUBLAS_DLL_NAME)
        if os.path.isfile(cand):
            return cand

    # 4. Program Files glob — covers users who never put CUDA on PATH.
    for base in (
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ):
        if not base:
            continue
        pattern = os.path.join(base, "NVIDIA GPU Computing Toolkit",
                               "CUDA", "v12*", "bin", _CUBLAS_DLL_NAME)
        hits = glob.glob(pattern)
        if hits:
            return hits[0]

    return None


def _ctranslate2_sees_cuda() -> bool:
    """True iff ctranslate2 thinks at least one CUDA device is present.
    Used to decide whether cublas64_12.dll's absence is a real problem
    (we'd have tried CUDA) vs. a non-event (no GPU at all, will fall
    back to CPU regardless)."""
    try:
        import ctranslate2 as _ct2
        return _ct2.get_cuda_device_count() > 0
    except Exception:
        return False


def _preflight_api_key(timeout_sec: float = 10.0) -> tuple[bool, str]:
    """Verify ANTHROPIC_API_KEY exists AND the Claude API answers a
    1-token ping inside `timeout_sec`. Returns (ok, reason). `reason`
    is a short human-readable string when ok=False, else ''.

    Skipped (returns ok=True) when AI_BACKEND isn't 'claude' — Ollama
    users don't need the key. The ping uses the cheapest possible
    completion (1 token) so we don't burn cache / quota on boot."""
    if AI_BACKEND != "claude":
        return True, ""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY environment variable is not set"
    try:
        import anthropic
    except ImportError as e:
        return False, f"anthropic package not importable: {e}"

    # Quick 1-token ping with a hard timeout. Use a worker thread so a
    # hung HTTPS handshake can't freeze boot — the SDK's per-call timeout
    # only kicks in once the request is actually issued.
    result: dict = {"ok": False, "err": "timed out"}

    def _ping():
        try:
            client = anthropic.Anthropic(timeout=timeout_sec)
            client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "."}],
            )
            result["ok"] = True
        except Exception as e:
            result["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=_ping, daemon=True)
    t.start()
    t.join(timeout=timeout_sec + 0.5)
    if t.is_alive():
        return False, f"Claude API ping timed out after {timeout_sec:.1f}s"
    if not result["ok"]:
        return False, f"Claude API ping failed: {result['err']}"
    return True, ""


def _preflight_cublas_check() -> bool:
    """Probe for cublas64_12.dll in the standard CUDA 12 install paths.
    If missing AND ctranslate2 can see a CUDA device (i.e. we'd actually
    try GPU), queue 'install CUDA 12 cuBLAS' to jarvis_todo.md and set
    _force_whisper_cpu_int8 so _ensure_whisper() skips the doomed CUDA
    load attempt. Returns True when the DLL was found (no fallback)."""
    global _force_whisper_cpu_int8
    found = _find_cublas_dll()
    if found:
        print(f"  [preflight] cublas64_12.dll: ok ({found})")
        return True

    sees_cuda = _ctranslate2_sees_cuda()
    if not sees_cuda:
        # No GPU in play anyway — the missing DLL is irrelevant. Don't
        # spam the todo queue and don't force the int8 flag (CPU is
        # already what _resolve_whisper_device would pick).
        print(f"  [preflight] cublas64_12.dll not found, but ctranslate2 "
              f"sees no CUDA device — CPU mode is fine")
        return False

    print(f"  [preflight] cublas64_12.dll MISSING — forcing whisper to CPU "
          f"int8 mode. Queueing CUDA install task.")
    _force_whisper_cpu_int8 = True

    # Append the install task to jarvis_todo.md via overnight_upgrade's
    # atomic helper so we don't race the overnight thread mid-write.
    try:
        import overnight_upgrade as _ou
        try:
            n = _ou._append_tasks(["install CUDA 12 cuBLAS (cublas64_12.dll "
                                   "missing — faster-whisper falls back to "
                                   "CPU int8 without it)"])
            if n:
                print(f"  [preflight] queued {n} CUDA install task to jarvis_todo.md")
            else:
                print(f"  [preflight] CUDA install task already queued — skipping duplicate")
        except Exception as e:
            print(f"  [preflight] could not append CUDA task: {e}")
    except Exception as e:
        print(f"  [preflight] overnight_upgrade not importable: {e}")
    return False


def _preflight_cameras(timeout_sec: float = 2.0) -> None:
    """Probe each configured camera index with a short timeout BEFORE
    probe_cameras_and_update_config() runs, and drop any indices that
    fail to open from CAMERAS. Stops the face-tracking thread from
    entering the read-failure / release / reopen loop on a phantom
    device that simply isn't there.

    The full probe (probe_cameras_and_update_config) still runs after
    this — it handles the all-failed-sweep case and rewrites CAMERAS
    with whatever's actually plugged in. This pre-check just keeps a
    half-broken configured index from poisoning the rest of boot."""
    if not CAMERAS:
        return
    if not CAMERA_PROBE_ENABLED:
        return
    # Reuse the existing _probe_camera_index helper — it already runs
    # the open in a worker thread with a hard timeout and serialises
    # against other camera I/O via _camera_io_lock. Fan out across
    # indices so total wall time stays ~timeout_sec instead of N×.
    results: dict[int, bool] = {}

    def _check(i: int):
        try:
            results[i] = _probe_camera_index(i, timeout_sec=timeout_sec)
        except Exception as e:
            print(f"  [preflight] camera index {i}: probe raised "
                  f"{type(e).__name__}: {e} — treating as bad")
            results[i] = False

    threads: list[threading.Thread] = []
    for cam in list(CAMERAS):
        idx = cam.get("index")
        if idx is None:
            continue
        t = threading.Thread(target=_check, args=(idx,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=timeout_sec + 0.5)

    bad: list[int] = []
    for cam in list(CAMERAS):
        idx = cam.get("index")
        if idx is None:
            continue
        if results.get(idx, False):
            print(f"  [preflight] camera index {idx}: opens cleanly ✓")
        else:
            print(f"  [preflight] camera index {idx}: failed to open in "
                  f"{timeout_sec:.1f}s — marking bad")
            bad.append(idx)
    if bad and len(bad) < len(CAMERAS):
        # Only drop when at least one camera survived; if ALL configured
        # indices fail, leave CAMERAS untouched so probe_cameras_and_
        # update_config can do its sweep-and-rewrite step.
        CAMERAS[:] = [c for c in CAMERAS if c.get("index") not in bad]
        print(f"  [preflight] dropped {len(bad)} bad camera index/es; "
              f"{len(CAMERAS)} remain")


def _startup_preflight() -> None:
    """Three self-heal checks the orchestrator runs right after logging
    starts and before any of the heavy boot work (whisper, daemons,
    cameras). Each check is wrapped so a single failure can't fully
    abort boot — except the API-key check, which IS fatal because the
    rest of the session is useless without a working Claude backend.

    See jarvis_todo.md 2026-05-29 23:27 (startup self-heal preflight)
    for the original incident: a missing ANTHROPIC_API_KEY caused a
    silent FATAL exit with only a print statement; cublas DLL absence
    caused repeated transcription failures; camera index 1 failing to
    open kicked off a read-failure / reopen loop in the face tracker."""
    print("─" * 60)
    print("Startup preflight…")

    # (1) ANTHROPIC_API_KEY + 1-token Claude ping.
    try:
        ok, reason = _preflight_api_key(timeout_sec=10.0)
    except Exception as e:
        ok, reason = False, f"preflight raised {type(e).__name__}: {e}"
    if not ok:
        # Claude API is an OPTIONAL ENHANCEMENT, not a requirement. The local
        # Ollama model is JARVIS's always-on baseline brain; Claude just makes
        # replies sharper WHEN credits are available. So Claude being
        # unavailable for ANY reason — no key at all, the monthly usage cap,
        # quota, rate-limit, or a network blip — is NOT an error as long as the
        # local model is reachable: we simply run local, no fuss. Only a TRUE
        # fatal (no Claude AND no local brain) aborts boot. 2026-05-30, per the
        # user: "I don't want to NEED API credits — it's a bonus that runs
        # better when present."
        _local_ok = False
        try:
            _local_ok = bool(_ollama_alive())
        except Exception:
            _local_ok = False
        if _local_ok:
            _why = reason.split(" - ", 1)[0] if " - " in reason else reason
            print(f"  [preflight] Claude enhancement not active ({_why[:80]}).")
            print("  [preflight] Running on the LOCAL model — fully "
                  "operational, sir. Claude engages automatically when API "
                  "credits are available; it's a bonus, not a requirement.")
            # Deliberately NO spoken alarm and NO self-heal task queued: local
            # is the normal baseline, so there's nothing 'wrong' to announce.
            # fall through — boot continues on the local model.
        else:
            print(f"\n[FATAL] No thinking backend available: {reason}")
            print("  [preflight] Neither a local Ollama model NOR the Claude "
                  "API is reachable. Start Ollama (the baseline brain) — the "
                  "Claude API key is optional, only for the cloud bonus.")
            try:
                _speak("Sir, I have no thinking backend at all — the local "
                       "model isn't reachable. Please make sure Ollama is "
                       "running.")
            except Exception as _spk_err:
                print(f"  [preflight] could not speak the error: {_spk_err}")
            close_log()
            sys.exit(1)
    if ok:
        print(f"  [preflight] Claude API: reachable — cloud enhancement active, sir.")

    # (2) cublas64_12.dll — gate whisper's CUDA load attempt.
    try:
        _preflight_cublas_check()
    except Exception as e:
        print(f"  [preflight] cublas check raised "
              f"{type(e).__name__}: {e} — continuing without forcing CPU")

    # (3) Per-camera open-and-read smoke test.
    try:
        _preflight_cameras(timeout_sec=2.0)
    except Exception as e:
        print(f"  [preflight] camera check raised "
              f"{type(e).__name__}: {e} — leaving CAMERAS untouched")

    print("Preflight complete.")
    print("─" * 60)


# Pending-speech + injected-command queue paths. Module-level constants so
# the queue helpers below (_drain_injected_command / _speak_pending) and the
# main-loop capture front can share them without nesting inside main().
# Deterministic paths + a launch-time env flag — computed once at import.
PENDING_SPEECH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pending_speech.json")
INJECTED_COMMANDS_PATH = _BLUE_GREEN_PATHS["inject_file"]
_INJECT_TEST_MODE = os.environ.get("JARVIS_TEST_MODE") == "1"


def _drain_injected_command():
    """Pop and return the next injected command text, or None.

    Race-safe consume-and-rename pattern (mirrors _speak_pending):
    atomically rename the queue file to `.consuming` to claim the
    snapshot, parse the JSON array, take the first entry, then
    rewrite any remaining items back to a fresh queue file so we
    don't strand the tail. Any JSON / encoding error discards the
    whole snapshot rather than crashing the main loop.
    """
    if not os.path.exists(INJECTED_COMMANDS_PATH):
        return None
    consume_path = INJECTED_COMMANDS_PATH + ".consuming"
    try:
        os.replace(INJECTED_COMMANDS_PATH, consume_path)
    except FileNotFoundError:
        return None
    except Exception as _e:
        print(f"  [inject] claim failed: {_e}")
        return None
    try:
        with open(consume_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except Exception as _e:
        print(f"  [inject] read failed: {_e}")
        try: os.remove(consume_path)
        except Exception: pass
        return None
    if not raw:
        try: os.remove(consume_path)
        except Exception: pass
        return None
    try:
        items, _ = json.JSONDecoder().raw_decode(raw)
    except Exception as _e:
        # External tool wrote malformed JSON — drop the snapshot so
        # the same garbage doesn't re-trip us next pass.
        print(f"  [inject] corrupt JSON discarded: {_e}")
        try: os.remove(consume_path)
        except Exception: pass
        return None
    if not isinstance(items, list) or not items:
        try: os.remove(consume_path)
        except Exception: pass
        return None
    first = items[0]
    remaining = items[1:]
    if remaining:
        # Put the tail back atomically so it survives this iteration
        # and the next loop pass picks it up. If this write fails the
        # tail is lost — but that's better than re-firing the head.
        try:
            _proj = os.path.dirname(INJECTED_COMMANDS_PATH)
            fd: int = -1
            tmp: str | None = None
            try:
                fd, tmp = tempfile.mkstemp(dir=_proj, suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    fd = -1   # fdopen took ownership of the descriptor
                    json.dump(remaining, f, indent=2)
                os.replace(tmp, INJECTED_COMMANDS_PATH)
                tmp = None
            except Exception:
                if fd >= 0:
                    try: os.close(fd)
                    except Exception: pass  # pragma: no cover - defensive: os.close of the owned requeue descriptor effectively never raises
                if tmp is not None:
                    try: os.unlink(tmp)
                    except Exception: pass  # pragma: no cover - defensive: os.unlink of the just-created requeue temp effectively never raises
                raise
        except Exception as _e:
            print(f"  [inject] failed to requeue {len(remaining)} item(s): {_e}")
    try: os.remove(consume_path)
    except Exception: pass
    if isinstance(first, str):
        text = first.strip()
    elif isinstance(first, dict):
        text = str(first.get("text", "")).strip()
    else:
        text = ""
    if not text:
        return None
    return text

def _speak_pending():
    """If skills (like the timer) have queued reminders, speak them now.

    Race-safe consume-and-rename pattern: we rename the queue file to a
    sibling `.consuming` BEFORE iterating, so any skill that writes a new
    reminder during the speak loop creates a fresh `pending_speech.json`
    on disk rather than appending to a file we're about to delete. Items
    that get written DURING the speak loop fire on the next idle pass
    instead of being silently dropped."""
    if not os.path.exists(PENDING_SPEECH_PATH):
        return False
    consume_path = PENDING_SPEECH_PATH + ".consuming"
    try:
        # Atomic rename — if anything errors, we leave the original file
        # alone for the next call to retry.
        try:
            os.replace(PENDING_SPEECH_PATH, consume_path)
        except FileNotFoundError:
            return False
        with open(consume_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            try: os.remove(consume_path)
            except Exception: pass
            return False
        try:
            items, _ = json.JSONDecoder().raw_decode(raw)
        except Exception:
            # Unrecoverable JSON — drop the snapshot and move on.
            print(f"  [pending] corrupt speech queue — deleting")
            try: os.remove(consume_path)
            except Exception: pass
            return False
    except Exception:
        return False
    if not items:
        try: os.remove(consume_path)
        except Exception: pass
        return False
    # Speak each reminder under its own try/except so a single edge-tts
    # failure doesn't crash the main loop or strand the rest of the queue.
    # Second-layer dedup: skip messages we've already spoken inside the
    # _RECENT_SPEECH_DEDUPE_WINDOW. Catches the looping-notification bug
    # where the same toast lands in the queue many times in one snapshot
    # (because the upstream listener lost its dedupe state across a
    # bounce). Timers/promises that intentionally repeat use distinct
    # message strings, so this won't suppress them.
    spoke_any = False
    seen_in_batch: set[str] = set()
    for item in items:
        msg = item.get("message", "")
        if not msg:
            continue
        if msg in seen_in_batch or _speech_was_recently_spoken(msg):
            print(f"  [pending] suppressed duplicate: {msg[:80]}")
            continue
        seen_in_batch.add(msg)
        print(f"  🔔 [reminder] {msg}")
        try:
            vol = float(item.get("volume_scale", 1.0))
        except (TypeError, ValueError):
            vol = 1.0
        item_mood = item.get("mood")
        if not isinstance(item_mood, str) or item_mood not in _VOICE_MOOD_NAMES:
            item_mood = None
        try:
            _speak(msg, volume_scale=vol, mood=item_mood)
            _mark_speech_spoken(msg)
            spoke_any = True
        except Exception as _spe:
            # Don't let one bad TTS attempt nuke the main loop. Print and
            # continue to the next reminder. The snapshot is gone (we
            # already renamed), so this one is lost — but JARVIS stays up.
            print(f"  [pending] speak failed for reminder: {_spe}")
    # Snapshot fully consumed — delete it. Any new items written DURING
    # this loop live in a fresh pending_speech.json that the next call
    # will pick up.
    try:
        os.remove(consume_path)
    except Exception:
        pass
    return spoke_any

# Sub-agent orchestrator (core/orchestrator) — explicit multi-agent briefing
# requests fan out to parallel READ-ONLY sub-agents (email / calendar / news /
# system) and merge into one spoken brief. Gated by ENABLE_ORCHESTRATOR
# (core/config, default False) OR a JARVIS_ENABLE_ORCHESTRATOR=1 env override,
# AND a narrow trigger — so the default turn pipeline is completely untouched
# unless the operator explicitly opts in.
# Tightened to ONLY the standing-briefing intents the orchestrator can actually
# fulfil (a fixed email/news/weather/system fan-out). Deliberately NOT a bare
# "brief me …" — "brief me on the Johnson account" is an arbitrary topic the
# orchestrator can't honour, so it must fall through to the normal LLM turn.
_ORCHESTRATE_RE = re.compile(
    r"\b("
    r"orchestrate"
    r"|(?:morning|daily|evening|day'?s?)\s+(?:brief|briefing|rundown|summary)"
    r"|(?:summari[sz]e|sum\s+up)\s+(?:my\s+)?(?:morning|day)"
    r"|brief\s+me\s+on\s+(?:my\s+|the\s+)?(?:morning|day)"
    r"|(?:system|status)\s+brief(?:ing)?"
    r")\b",
    re.IGNORECASE,
)


def _orchestrator_enabled() -> bool:
    """True when the orchestrator is wired in — via the config flag or the
    JARVIS_ENABLE_ORCHESTRATOR=1 env override (used for staging validation)."""
    try:
        if ENABLE_ORCHESTRATOR:
            return True
    except NameError:
        pass
    return os.environ.get("JARVIS_ENABLE_ORCHESTRATOR", "").strip() == "1"


def _is_orchestration_request(text: str) -> bool:
    """True only for the narrow set of explicit multi-agent briefing phrasings,
    kept high-precision (short utterance + specific phrase) so a normal request
    is never hijacked into a multi-second orchestrated fan-out."""
    if not text:
        return False
    t = text.strip()
    if len(t.split()) > 9:
        return False
    return _ORCHESTRATE_RE.search(t) is not None


def _maybe_orchestrate(text: str) -> bool:
    """If the orchestrator is enabled AND `text` is an explicit briefing
    request, fan out to the sub-agent orchestrator and speak the merged brief.
    Returns True when it handled the turn (caller ``continue``s). Best-effort:
    any failure or an empty result returns False so the utterance falls through
    to the normal LLM turn — the orchestrator never blocks a normal request."""
    if not _orchestrator_enabled() or not _is_orchestration_request(text):
        return False
    try:
        from core.orchestrator import orchestrate as _orchestrate
    except Exception as _e:
        print(f"  [orchestrator] unavailable: {_e}")
        return False
    print("  [orchestrator] briefing request — fanning out to sub-agents…")
    set_state("thinking")
    try:
        merged = _orchestrate(
            text, ACTIONS,
            planner_model=ORCHESTRATOR_PLANNER_MODEL,
            worker_model=ORCHESTRATOR_WORKER_MODEL,
            merger_model=ORCHESTRATOR_MERGER_MODEL,
            max_parallel=ORCHESTRATOR_MAX_PARALLEL,
            worker_timeout_s=ORCHESTRATOR_WORKER_TIMEOUT_S,
            planner_timeout_s=ORCHESTRATOR_PLANNER_TIMEOUT_S,
            merger_timeout_s=ORCHESTRATOR_MERGER_TIMEOUT_S,
        )
    except Exception as _e:
        print(f"  [orchestrator] failed ({_e}) — falling through to normal turn")
        return False
    # Speak the merged brief directly — it's final synthesised prose. Do NOT
    # route it through parse_and_run_actions: the sub-agents ALREADY performed
    # the reads, so the hallucination guard would misfire on the brief's
    # legitimate "I checked your …" phrasing. Defensively strip any stray
    # [ACTION:…] token so it can never be read aloud.
    clean = re.sub(r"\[\s*ACTION\s*:[^\]]*\]", "", merged or "").strip()
    if not clean:
        print("  [orchestrator] empty result — falling through to normal turn")
        return False
    print(f"  [orchestrator] brief: {clean[:300]}")
    conversation_history.append({"role": "user", "content": text})
    conversation_history.append({"role": "assistant", "content": clean})
    _trim_conversation_history()
    _speak(clean)
    set_state("idle")
    return True


def _run_voice_shortcuts(text: str) -> bool:
    """Normal-mode voice shortcuts that bypass the LLM round-trip: replay-
    last-action, the TTS-backend toggle, conversation-mode toggle /
    controlled-mode dispatch, and the multi-step chain resolver. Each
    speaks its own reply and appends the turn to conversation_history.
    Returns True when a shortcut handled the utterance (caller should
    ``continue``); False to fall through to the full LLM dispatch.
    """
    # Replay-last-action voice trigger: 'do that again' / 'replay that'
    # / 'do it again' / 'do that on monitor right' — re-fires the most
    # recent non-destructive action straight out of the in-memory deque
    # without a Claude round-trip. Returns None when the utterance
    # doesn't match, so normal dispatch continues.
    try:
        _replay_reply = maybe_replay_last_action(text)
    except Exception as _e:
        print(f"  [replay] handler failed: {_e}")
        _replay_reply = None
    if _replay_reply is not None:
        print(f"  [replay] {_replay_reply}")
        conversation_history.append({"role": "user", "content": text})
        conversation_history.append(
            {"role": "assistant", "content": _replay_reply}
        )
        _speak(_replay_reply)
        set_state("idle")
        return True

    # TTS-backend voice toggle: 'use my voice' / 'switch to edge voice'
    # / 'switch to the pyttsx3 voice' — pre-router that flips the
    # TTS_BACKEND knob without an LLM round-trip. The confirmation
    # line is spoken in the NEW backend so the user immediately hears
    # whether the switch took.
    try:
        _voice_mod = sys.modules.get("skill_custom_voice")
        _backend_reply = _voice_mod.maybe_switch_backend(text) if _voice_mod else None
    except Exception as _e:
        print(f"  [tts-toggle] handler failed: {_e}")
        _backend_reply = None
    if _backend_reply is not None:
        print(f"  [tts-toggle] {_backend_reply}")
        conversation_history.append({"role": "user", "content": text})
        conversation_history.append(
            {"role": "assistant", "content": _backend_reply}
        )
        _speak(_backend_reply)
        set_state("idle")
        return True

    # Conversation mode (controlled / smart / agent — research-14).
    # Three layered checks share the same pre-router slot:
    #
    #   1. Mode toggle. 'JARVIS, controlled mode' / 'agent mode' /
    #      'smart mode' / 'what mode are you in' — handled before
    #      any other routing so the user can always flip modes,
    #      even when the previous mode would have suppressed
    #      conversational replies.
    #   2. Controlled-mode dispatch. When in controlled mode, route
    #      the utterance through the deterministic intent matcher;
    #      anything that doesn't match a registered skill gets a
    #      refusal line rather than being passed to the LLM.
    #      controlled_dispatch() handles both single-intent and
    #      multi-step chains, so the dedicated chain resolver
    #      block below is skipped in this mode.
    #   3. Multi-step intent (chain resolver). Active in smart and
    #      agent modes — dispatches utterances like "play X and
    #      start a Y timer and …" without an LLM round-trip.
    try:
        from core.mode_router import (
            maybe_handle_mode_toggle as _mode_toggle,
            controlled_dispatch as _controlled_dispatch,
            is_in_controlled_mode as _is_controlled,
        )
    except Exception as _e:
        print(f"  [mode] router unavailable: {_e}")
        _mode_toggle = _controlled_dispatch = None
        _is_controlled = lambda: False

    if _mode_toggle is not None:
        try:
            _toggle_reply = _mode_toggle(text)
        except Exception as _e:
            print(f"  [mode] toggle handler failed: {_e}")
            _toggle_reply = None
        if _toggle_reply is not None:
            print(f"  [mode] {_toggle_reply}")
            conversation_history.append({"role": "user", "content": text})
            conversation_history.append(
                {"role": "assistant", "content": _toggle_reply}
            )
            _speak(_toggle_reply)
            set_state("idle")
            return True

    if _is_controlled():
        if _controlled_dispatch is not None:
            try:
                _ctrl_reply = _controlled_dispatch(text, ACTIONS)
            except Exception as _e:
                print(f"  [mode] controlled dispatch failed: {_e}")
                _ctrl_reply = (
                    "Controlled mode is on, sir, but the dispatcher "
                    "errored — say 'smart mode' to recover."
                )
            if _ctrl_reply is not None:
                print(f"  [mode-controlled] {_ctrl_reply}")
                conversation_history.append({"role": "user", "content": text})
                conversation_history.append(
                    {"role": "assistant", "content": _ctrl_reply}
                )
                _speak(_ctrl_reply)
                set_state("idle")
                return True

    # Multi-step intent: if the user chained ≥2 recognized commands
    # in one utterance ("play X and start a Y timer and …"), dispatch
    # them directly with a single consolidated confirmation, skipping
    # the LLM round-trip where it would otherwise muddle the request.
    # Returns None for everything else, so single-intent utterances
    # fall through to the normal LLM path below.
    try:
        from core.dispatcher import resolve_and_dispatch as _cc_resolve
        _chain_reply = _cc_resolve(text, ACTIONS)
    except Exception as _e:
        print(f"  [chain] resolver failed: {_e}")
        _chain_reply = None
    if _chain_reply is not None:
        print(f"  [chain] {_chain_reply}")
        conversation_history.append({"role": "user", "content": text})
        conversation_history.append(
            {"role": "assistant", "content": _chain_reply}
        )
        _speak(_chain_reply)
        set_state("idle")
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────
#  Experimental low-latency voice wiring (opt-in, default-off, fail-safe)
# ──────────────────────────────────────────────────────────────────────────
#
# Two fully-built subsystems (core/realtime_voice.py + core/wake_word.py) are
# wired in behind config flags that DEFAULT OFF. The SELECTION logic lives in
# core/voice_pipeline.py (CI-unit-tested with deps absent); the monolith just
# calls those selectors and branches, ALWAYS inside try/except so any failure
# (missing optional dep, init error, runtime raise) transparently falls back to
# the historical turn-based / Whisper-substring path for the rest of the
# session. When the flags are off these helpers short-circuit to None/False on
# their first line, so the default hot path is byte-for-byte unchanged.
#
# Single-element-list GIL-atomic latches (the same idiom as _sleep_mode etc.):
#   [0] holds the live object (or None); a second slot records that we already
#   tried + failed so we don't re-probe every loop iteration.
_realtime_session = [None]            # RealtimeVoicePipeline | None
_realtime_disabled_for_session = [False]
_realtime_utterances: "queue.Queue[str]" = queue.Queue()

_standby_wake_detector = [None]       # WakeWordDetector (idle) | None
_standby_wake_disabled_for_session = [False]


def _get_realtime_session():
    """Return the live realtime streaming session, lazily creating it the first
    time VOICE_MODE=='realtime' AND its deps are present. Returns None to mean
    'stay on the turn-based loop' — which is the default and every failure mode.

    Never raises. A construction failure latches _realtime_disabled_for_session
    so we attempt it at most once per process and the loop silently stays
    turn-based thereafter."""
    if _realtime_session[0] is not None:
        return _realtime_session[0]
    if _realtime_disabled_for_session[0]:
        return None
    try:
        from core import voice_pipeline as _vp
        if not _vp.realtime_enabled():
            # Flag off → permanently turn-based for this run; don't re-probe.
            _realtime_disabled_for_session[0] = True
            return None

        def _on_utterance(_text: str) -> None:
            # The session's STT thread calls this for each finalised turn; hand
            # it to the main loop via the same blocking-queue contract the
            # turn-based path uses (record_speech timeout semantics).
            try:
                _realtime_utterances.put_nowait(_text)
            except Exception:
                pass

        sess = _vp.make_realtime_session(
            on_user_utterance=_on_utterance,
            stt_language="en",
            tts_engine="system",
            tts_voice=TTS_VOICE,
        )
    except Exception as _e:
        print(f"  [voice] realtime init failed ({_e}); staying turn_based")
        _realtime_disabled_for_session[0] = True
        return None
    if sess is None:
        _realtime_disabled_for_session[0] = True
        return None
    _realtime_session[0] = sess
    return sess


def _realtime_capture(timeout: float = 20.0):
    """Realtime-mode capture: block up to `timeout` s for the next finalised
    utterance the streaming STT pipeline pushed onto _realtime_utterances.

    Returns (text, conf) shaped exactly like transcribe() so the caller is
    branch-symmetric with the Whisper path, or None on timeout (caller
    ``continue``s the loop, same as record_speech() returning None). conf is
    synthesised trustworthy metadata because RealtimeSTT already endpointed +
    finalised the turn (no per-segment logprobs to forward)."""
    try:
        text = _realtime_utterances.get(timeout=timeout)
    except queue.Empty:
        return None
    if not text or not text.strip():
        return None
    return text, {"no_speech_prob": 0.0, "avg_logprob": -0.1}


def _get_standby_wake_detector():
    """Return an IDLE WakeWordDetector for the standby fast-path, lazily built
    the first time WAKE_WORD_AUTOSTART is on AND the engine's dep is present.
    Returns None to mean 'use the Whisper-substring standby path' — the default
    and every failure mode.

    autostart=False: we do NOT let it open its own mic stream (Windows WASAPI
    rejects a 2nd open on the device record_speech() already uses). Instead the
    standby loop feeds it the audio buffer it just captured. Latches a
    one-shot disable on failure so we don't re-probe each standby tick."""
    if _standby_wake_detector[0] is not None:
        return _standby_wake_detector[0]
    if _standby_wake_disabled_for_session[0]:
        return None
    try:
        from core import voice_pipeline as _vp
        if not _vp.wake_word_autostart_enabled():
            _standby_wake_disabled_for_session[0] = True
            return None
        det = _vp.make_wake_detector(
            wake_words=sorted(WAKE_PHRASES),
            sample_rate=SAMPLE_RATE,
            autostart=False,
        )
    except Exception as _e:
        print(f"  [standby] neural wake init failed ({_e}); using Whisper path")
        _standby_wake_disabled_for_session[0] = True
        return None
    if det is None:
        _standby_wake_disabled_for_session[0] = True
        return None
    _standby_wake_detector[0] = det
    return det


def _standby_wake_detected(audio) -> bool | None:
    """Feed an already-captured float32 mono `audio` buffer to the neural wake
    detector and report whether a wake phrase fired.

    Returns:
        True  — a wake word was detected in the buffer.
        False — detector ran cleanly, no wake word (caller stays asleep).
        None  — the neural path is unavailable/disabled OR errored; the caller
                must fall back to the existing Whisper-substring check.

    Never raises. Any detector error returns None so the standby loop degrades
    to the historical path with no behaviour change."""
    det = _get_standby_wake_detector()
    if det is None:
        return None
    try:
        # Drain any stale events from a prior buffer so we only judge this one.
        while True:
            try:
                det.events.get_nowait()
            except queue.Empty:
                break
        # Push the captured audio through the detector's frame pipeline in the
        # 80ms frames it expects, exactly as its own InputStream callback would.
        frame_size = max(1, int(SAMPLE_RATE * core_wake_frame_ms() / 1000))
        buf = np.asarray(audio, dtype=np.float32).reshape(-1)
        for i in range(0, buf.size - frame_size + 1, frame_size):
            det._on_frame(buf[i:i + frame_size])
        # Any event in the queue → a wake fired.
        try:
            det.events.get_nowait()
            return True
        except queue.Empty:
            return False
    except Exception as _e:
        print(f"  [standby] neural wake check errored ({_e}); using Whisper path")
        _standby_wake_disabled_for_session[0] = True
        return None


def core_wake_frame_ms() -> int:
    """The frame size (ms) core.wake_word expects, read defensively so a tree
    without that constant still works (falls back to 80 ms)."""
    try:
        from core import wake_word as _ww
        return int(getattr(_ww, "DEFAULT_FRAME_MS", 80))
    except Exception:
        return 80


def _capture_utterance(injected_text, memory):
    """Normal-mode capture phase: drain queued speech, then get the next
    utterance — an injected command (mic + Whisper bypassed, safe pass-through
    metadata) or a fresh mic recording transcribed by Whisper.

    Returns (text, conf) for the turn, or None when the caller should
    ``continue`` the loop: a queued reminder was just spoken, the silence
    timeout fired (and a proactive turn may have run), or the clip was too
    short to be speech.
    """
    global _last_recording_peak
    # Drain any speech queued between turns (timer reminders, device auto-switch
    # alerts, etc.) so they fire promptly instead of only after a 20s timeout.
    _speak_pending()

    if injected_text is not None:
        # Inject path: bypass mic + Whisper but route through every downstream
        # handler (wake-word/sleep triggers, confirmation router, intent
        # dispatch, LLM, action parser) exactly as a spoken utterance would.
        # Synthesise safe pass-through metadata: peak_rms above WHISPER_TRUST_RMS
        # so is_valid_speech skips word-count + confidence gates, and neutral
        # confidence values so any downstream caller reading conf gets sane data.
        text = injected_text
        conf = {"no_speech_prob": 0.0, "avg_logprob": -0.1}
        _last_recording_peak = max(
            WHISPER_TRUST_RMS * 2.0, _last_recording_peak
        )
        _heartbeat()
        if _INJECT_TEST_MODE:
            print(f"  [inject] (test-mode) text={text!r} "
                  f"conf={conf} peak_rms={_last_recording_peak:.4f}")
        else:
            print(f"  [inject] {text}")
        set_state("listening")
        return text, conf

    # Mute Mic (tray "Mute Mic"): ignore the live mic entirely while muted —
    # queued reminders still drain above and injected commands still pass, but no
    # fresh recording is taken or dispatched. Brief sleep so the loop idles
    # rather than busy-spinning on the immediate `continue`. Distinct from
    # standby (which still wakes on the wake word) and Mute TTS (still acts, just
    # silent). Drives the tray's red/muted listen indicator.
    if _mic_muted[0]:
        _heartbeat()
        time.sleep(0.3)
        return None

    # ── EXPERIMENTAL: realtime streaming capture (VOICE_MODE='realtime') ──────
    # Off by default: _get_realtime_session() returns None unless the flag is on
    # AND RealtimeSTT/RealtimeTTS import, so this whole block is skipped and the
    # historical record_speech()+transcribe() path below runs byte-for-byte. The
    # streaming STT pipeline endpoints + finalises turns on its own thread and
    # pushes them onto a queue; here we just block on that queue with the same
    # 20 s budget record_speech() uses, returning a transcribe()-shaped tuple.
    # Wrapped so any error degrades to the turn-based path for the rest of the
    # session (the selector latched it; the next call returns None instantly).
    try:
        _rt_sess = _get_realtime_session()
    except Exception:
        _rt_sess = None
    if _rt_sess is not None:
        _heartbeat()
        print("Listening… (realtime)")
        resume_face_tracking()
        try:
            _rt_cap = _realtime_capture(timeout=20.0)
        except Exception as _e:
            print(f"  [voice] realtime capture errored ({_e}); "
                  "falling back to turn_based")
            _realtime_disabled_for_session[0] = True
            _rt_cap = None
            _rt_sess = None
        if _rt_sess is not None:
            if _rt_cap is None:
                # No utterance within the window — mirror the record_speech()
                # timeout branch exactly (reminders, then maybe a proactive turn).
                if _speak_pending():
                    set_state("idle")
                    return None
                if should_be_proactive():
                    _do_proactive_turn(memory)
                set_state("idle")
                return None
            return _rt_cap

    _heartbeat()
    print("Listening…")
    resume_face_tracking()

    # Wait for speech, but only briefly — so we can check proactive
    audio = record_speech(timeout=20)

    if audio is None:
        # First check if any timers/reminders fired and need speaking
        if _speak_pending():
            set_state("idle")
            return None
        # No speech within timeout. Should we volunteer something?
        if should_be_proactive():
            _do_proactive_turn(memory)
        set_state("idle")
        return None

    if len(audio) < SAMPLE_RATE * 0.4:
        set_state("idle")
        return None

    # Keep the spectral music detector primed in normal mode too, so its
    # sustained-music state survives any sleep→standby transition triggered by
    # a music marker below. Feed it the RAW audio (pre auto-gain) so its
    # tuning is unaffected by the normalization stage below.
    _audio_music_feed(audio, SAMPLE_RATE)

    # CONSERVATIVE auto-gain: a quiet mic records speech too softly for Whisper
    # (empty transcript). Boost a sub-target buffer toward a usable peak just
    # before transcription; a no-op for already-loud/normal audio.
    audio, _ag = apply_capture_auto_gain(audio, _last_recording_peak)
    if _ag > 1.0:
        print(f"  [auto-gain] quiet input rms={_last_recording_peak:.4f} "
              f"-> x{_ag:.1f}")

    print("  Transcribing…")
    text, conf = transcribe(audio)
    return text, conf


# Cross-iteration blue/green loop-tick timers (single-element-list GIL-atomic
# idiom): last-heartbeat wall-clock and when a handoff signal was first seen.
_bg_last_heartbeat = [0.0]
_bg_handoff_seen_at = [0.0]


def _blue_green_loop_tick() -> bool:
    """Per-iteration blue/green housekeeping at the top of the main loop:
    refresh the instances.json heartbeat (every 5s), watch for a handoff
    signal (prod only — announce takeover + snapshot conversation tail /
    timers / last_speech_time to handoff.json), and react to the upgrade
    ceremony's abort / handoff-failure signals. Returns True when prod has
    observed a handoff and its grace window elapsed — the caller then returns
    from main() so the green instance can take over. All best-effort; a
    filesystem hiccup must never break the loop.
    """
    _now_bg = time.time()
    if _bgm is not None and (_now_bg - _bg_last_heartbeat[0]) > 5.0:
        try:
            _bgm.heartbeat(role=BLUE_GREEN_ROLE,
                           version=_bgm.read_version())
        except Exception:
            pass
        _bg_last_heartbeat[0] = _now_bg
    if _bgm is not None and BLUE_GREEN_ROLE == "prod" and _bg_handoff_seen_at[0] == 0.0:
        try:
            _signal = _bgm.consume_handoff_signal()
        except Exception:
            _signal = None
        if _signal:
            _bg_handoff_seen_at[0] = _now_bg
            _target = _signal.get("target_version") or "the new build"
            # blue-green-2: cinematic single-line takeover. Short
            # enough that the 3-second post-announce gap in the
            # pipeline lets the clip drain before promote_staging.
            try:
                _speak(f"Switching to the new version, sir — "
                       f"{_target}.")
            except Exception as _spe:
                print(f"  [blue-green] handoff announce failed: {_spe}")
            # Snapshot live timers so the new instance can re-arm
            # them (round5-M-5). Wrapped: a disabled or unloaded
            # timer skill must not block the handoff write.
            _active_timers_snapshot = []
            _skill_timer_mod = sys.modules.get("skill_timer")
            if _skill_timer_mod is not None:
                try:
                    _enum = getattr(_skill_timer_mod, "enumerate_timers", None)
                    if callable(_enum):
                        _active_timers_snapshot = _enum()
                except Exception as _ete:
                    print(f"  [blue-green] timer enumerate failed: {_ete}")
            try:
                _bgm.write_handoff_state({
                    "active_timers":   _active_timers_snapshot,
                    "conversation_tail": conversation_history[-6:]
                        if isinstance(conversation_history, list) else [],
                    "last_speech_time": last_speech_time,
                    "version_at_handoff": _bgm.read_version(),
                    "signaled_at": _signal.get("signaled_at", _now_bg),
                })
            except Exception as _hse:
                print(f"  [blue-green] handoff state save failed: {_hse}")
    # blue-green-2: prod-side reactions to the upgrade ceremony's
    # negative signals. Both are best-effort one-shots — the
    # consume_*() helpers delete the signal file on read.
    if _bgm is not None and BLUE_GREEN_ROLE == "prod":
        try:
            _abort = _bgm.consume_upgrade_aborted_signal()
        except Exception:
            _abort = None
        if _abort:
            try:
                _speak("I'm afraid the upgrade was aborted, sir. "
                       "Staying on the current version.")
            except Exception as _ae:
                print(f"  [blue-green] abort announce failed: {_ae}")
        try:
            _fail = _bgm.consume_handoff_failure_signal()
        except Exception:
            _fail = None
        if _fail:
            # Cancel any pending exit — the takeover isn't happening.
            _bg_handoff_seen_at[0] = 0.0
            try:
                _speak("Handoff failure, sir — I'll stay on the "
                       "current build.")
            except Exception as _fe:
                print(f"  [blue-green] handoff-failure announce "
                      f"failed: {_fe}")
    # If we observed a handoff signal, exit `grace_seconds` later
    # so any in-flight TTS completes before the green takeover.
    if _bg_handoff_seen_at[0] and (_now_bg - _bg_handoff_seen_at[0]) >= 10.0:
        print("  [blue-green] handoff window elapsed — prod exiting cleanly")
        return True
    return False


def _consume_blue_green_handoff() -> tuple[float | None, list]:
    """Blue-green-2: when relaunched as the new prod after a successful
    upgrade, pull the previous prod's in-flight conversation tail (+ last
    speech time and active timers) from data/handoff.json so the user doesn't
    lose the thread mid-sentence across the swap.

    Extends conversation_history in place and returns
    (pending_last_speech, pending_timers) for main() to apply AFTER the boot
    sequence's set_state('idle') reset. Restricted to the prod role — staging
    must never consume prod's handoff payload. No-op (returns (None, [])) when
    blue-green is disabled, we're not prod, or the resume flag is absent.
    """
    pending_last_speech: float | None = None
    pending_timers: list = []
    if not (
        _bgm is not None
        and BLUE_GREEN_ROLE == "prod"
        and _bgm.RESUME_HANDOFF_FLAG in sys.argv
    ):
        return pending_last_speech, pending_timers
    try:
        _handoff = _bgm.consume_handoff_state()
    except Exception as _hce:
        print(f"  [blue-green] handoff consume failed: {_hce}")
        _handoff = None
    if isinstance(_handoff, dict):
        _tail = _handoff.get("conversation_tail")
        if isinstance(_tail, list) and _tail:
            try:
                conversation_history.extend(
                    m for m in _tail
                    if isinstance(m, dict) and "role" in m and "content" in m
                )
                print(f"  [blue-green] resumed {len(_tail)} message(s) from handoff")
            except Exception as _hxe:
                print(f"  [blue-green] handoff replay failed: {_hxe}")
        # round5-M-5: pull the four other payload fields the previous consumer
        # silently dropped. last_speech_time + active_timers are applied later
        # (after the unconditional set_state('idle') resets them at the bottom
        # of the boot sequence). version_at_handoff + signaled_at are advisory —
        # log them so blue/green audits can correlate the handoff in upgrade.log.
        _v = _handoff.get("version_at_handoff")
        _sat = _handoff.get("signaled_at")
        if _v or _sat:
            print(f"  [blue-green] handoff from version={_v} "
                  f"signaled_at={_sat}")
        _lst = _handoff.get("last_speech_time")
        if isinstance(_lst, (int, float)) and _lst > 0:
            pending_last_speech = float(_lst)
        _ats = _handoff.get("active_timers")
        if isinstance(_ats, list) and _ats:
            pending_timers = _ats
    return pending_last_speech, pending_timers


def _handle_sleep_standby(injected_text: str | None) -> None:
    """SLEEP / STANDBY phase of the main loop: listen only for the wake
    phrase. Either an injected command (which must contain a wake phrase to
    act, per the safety rule) or a fresh mic capture is transcribed; on a wake
    phrase JARVIS clears sleep/standby and greets, otherwise the line is fed to
    the ambient learner (when active) and ignored. Returns to the caller, which
    immediately ``continue``s the loop — this phase never falls through to a
    normal turn, so its early-outs are plain returns.
    """
    _label = "Standby" if _standby_mode[0] else "Sleeping"
    if injected_text is not None:
        # Honour the spec safety rule: injects only act on a sleeping JARVIS
        # if the text contains a wake phrase. We log + fall through to the
        # existing wake-phrase check below so injects can wake the assistant
        # in exactly the same way a spoken 'JARVIS' would.
        _heartbeat()
        print(f"  [inject] (standby) {injected_text}")
        text = injected_text
    else:
        _heartbeat()
        print(f"{_label}… (say 'JARVIS' to wake)")
        set_state("idle")
        audio = record_speech(timeout=20)
        if audio is None or len(audio) < SAMPLE_RATE * 0.4:
            return
        # Feed the chunk to the spectral music detector so the standby loop
        # can spot sustained song audio and refuse lyric near-misses on the
        # wake-word check below. Fed the RAW audio (pre auto-gain) so the
        # music-refuse guard's tuning is unaffected by the normalization below.
        _audio_music_feed(audio, SAMPLE_RATE)
        # CONSERVATIVE auto-gain: on a QUIET mic the captured buffer is too soft
        # for the wake check — Whisper returns '' and the neural detector never
        # fires, so "JARVIS" is missed. Boost a sub-target buffer toward a usable
        # peak BEFORE both wake paths; a no-op for already-loud/normal audio.
        audio, _ag = apply_capture_auto_gain(audio, _last_recording_peak)
        if _ag > 1.0:
            print(f"  [auto-gain] quiet input rms={_last_recording_peak:.4f} "
                  f"-> x{_ag:.1f}")
        # ── EXPERIMENTAL: neural wake detector (WAKE_WORD_AUTOSTART) ──────────
        # Off by default: _standby_wake_detected() returns None unless the flag
        # is on AND openWakeWord/Porcupine import, in which case the line below
        # runs the FULL Whisper transcription exactly as before. When the neural
        # path IS engaged it judges the captured buffer directly — no per-
        # utterance Whisper — and we synthesise a minimal `text` so the existing
        # wake / music-refuse / greeting logic downstream is unchanged:
        #   True  → 'jarvis' (a wake phrase; the spectral music-refuse guard,
        #           fed REAL audio above, still vetoes lyric near-misses).
        #   False → '' (no wake phrase → the asleep/ignore branch). We skip the
        #           ambient-learning feed in this case because there is no
        #           transcript to learn from — the deliberate latency/accuracy
        #           trade of the neural fast-path. On ANY detector error it
        #           returns None and we fall back to Whisper for this + every
        #           later standby tick (the selector latched the failure).
        _wake_hit = _standby_wake_detected(audio)
        if _wake_hit is None:
            text, _ = transcribe(audio)
        else:
            text = "jarvis" if _wake_hit else ""
    tl = text.strip().lower()
    # Check for any wake phrase
    if any(wp in tl for wp in WAKE_PHRASES):
        if _audio_music_should_refuse_wake(text):
            print(f"  [{_label.lower()}] wake-word ignored "
                  f"(music playing, lyric near-miss): '{text[:60]}'")
            return
        _was_standby = _standby_mode[0]
        # Serialize the wake-clear with the background standby auto-engage
        # daemon, which sets these same flags True under
        # _standby_auto_engage_lock. Without the lock, an auto-engage firing in
        # the instant after we clear them would re-assert standby and leave
        # JARVIS stuck asleep right after the user said "JARVIS". (auto-engage
        # only calls _speak/set_state while holding the lock, so no deadlock.)
        with _standby_auto_engage_lock:
            _sleep_mode[0]    = False
            _standby_mode[0]  = False
        _ambient_music_hits[0] = 0
        # Ambient-learning 'answer_then_quiet': arm a one-shot so the NEXT
        # fully-processed command turn drops back to silent standby (the
        # re-sleep fires at the bottom of the loop, after the reply — NOT here,
        # so the user gets their one answer first). 'stay_talkative' leaves the
        # flag False so JARVIS stays awake until a sleep phrase.
        if _ambient_learning[0] and WAKE_RESUME_MODE == "answer_then_quiet":
            _resume_to_ambient[0] = True
        # Clear the overnight persistence flag — user is awake again and
        # reclaiming JARVIS. Otherwise the next 60s poll would re-trigger an
        # upgrade cycle right in the middle of their use.
        try:
            if os.path.exists(OVERNIGHT_FLAG_FILE):
                os.remove(OVERNIGHT_FLAG_FILE)
                _write_hud_state(overnight_expiry=0.0)
                print("  [overnight] persistence flag cleared — user awake")
        except Exception:
            pass
        print("  [wake] Waking up")
        # Context-aware greeting — see context_aware_greeting() for the
        # priority order (late-night repeat > morning-first > mid-print >
        # looking-away > variety bank). The wake text is passed so the variety
        # picker can read stress markers (rushed / frustrated → terse ack).
        _greeting, _vol = context_aware_greeting(
            from_standby=_was_standby, wake_text=text)
        print(f"  [wake] greeting='{_greeting}' vol={_vol}")
        _speak(_greeting, volume_scale=_vol)
        set_state("idle")
    else:
        # Ambient-learning: the overheard line wasn't a wake word. Persist it
        # (silently) so the fact-extractor can learn from it, then ignore it
        # for speech purposes as usual.
        if _ambient_learning[0]:
            _ambient_learning_feed(text)
        print(f"  [{_label.lower()}] ignored: '{text[:60]}'")


def _handle_ambient_music(text: str) -> bool:
    """Normal-mode ambient-music gate. Whisper emits markers like [Music] / ♪
    when the mic picks up song audio it can't transcribe as words. If JARVIS
    itself didn't recently kick off playback, count it toward an auto-standby
    trigger so we stop pestering the user with hallucinated transcriptions of
    overheard music. Returns True when the utterance was overheard music
    (caller should ``continue`` the loop); False to fall through to normal
    handling.
    """
    if not is_ambient_music(text):
        return False
    _from_jarvis = (time.time() - _jarvis_played_music_at[0]
                    < MUSIC_GRACE_PERIOD)
    if not _from_jarvis:
        # Reset rolling window if it's been too long since last hit
        if time.time() - _ambient_music_last_hit[0] > MUSIC_HITS_WINDOW:
            _ambient_music_hits[0] = 0
        _ambient_music_hits[0] += 1
        _ambient_music_last_hit[0] = time.time()
        print(f"  [ambient-music] hit {_ambient_music_hits[0]}/"
              f"{MUSIC_HITS_TO_STANDBY}: '{text[:40]}'")
        if _ambient_music_hits[0] >= MUSIC_HITS_TO_STANDBY:
            _sleep_mode[0]   = True
            _standby_mode[0] = True
            _ambient_music_hits[0] = 0
            print("  [standby] ambient music detected — entering standby")
            _speak("Music in the room, sir. I'll be in standby — "
                   "say 'JARVIS' when you need me.")
            set_state("idle")
            return True
    # Either way, the music transcription itself isn't speech to respond to.
    set_state("idle")
    return True


def _handle_sleep_triggers(text: str) -> bool:
    """Normal-mode sleep / standby trigger gate, checked before sending to the
    LLM. Two-gate: the utterance must be short (≤ 6 words) AND a phrase must
    match with word boundaries, so casual mentions ("if I say stop listening
    it should…") don't trigger sleep mid-explanation. Returns True when sleep
    or standby was entered (caller should ``continue``); False otherwise.
    """
    tl = text.strip().lower()
    _triggered_sleep   = False
    _triggered_standby = False
    if len(tl.split()) <= 6:
        for _sp in STANDBY_TRIGGER_PHRASES:
            if re.search(r'\b' + re.escape(_sp) + r'\b', tl):
                _triggered_standby = True
                break
        if not _triggered_standby:
            for _sp in SLEEP_PHRASES:
                if re.search(r'\b' + re.escape(_sp) + r'\b', tl):
                    _triggered_sleep = True
                    break
    if _triggered_standby:
        _sleep_mode[0]   = True
        _standby_mode[0] = True
        print("  [standby] Entering standby (work mode)")
        _speak("In standby, sir. Say 'JARVIS' when you need me.")
        set_state("idle")
        return True
    if _triggered_sleep:
        _sleep_mode[0] = True
        print("  [sleep] Entering sleep mode")
        _speak("Standing by, sir. Say 'JARVIS' when you need me.")
        set_state("idle")
        return True
    return False


def _run_llm_dispatch(text: str) -> str:
    """Single-utterance LLM turn: the glance fast-path or a full LLM
    response, execution of any [ACTION:] tokens, then the informative /
    failure follow-up loop. Speaks each reply and appends follow-ups to
    conversation_history. Returns the original `reply` so the caller can
    feed it to learn_from_turn(). The follow-up loop's breaks are its own
    (inner `for depth` loop) — this never touches the main loop's flow.
    """
    print("  Thinking…")
    # Glance-response fast path: if the focused window changed in the
    # last few seconds AND the utterance is ambiguous ("what is
    # this?" / "should I worry?" / "wait, what?" / "explain"), grab
    # just that window and reply in one sentence — no [ACTION:
    # see_screen, …] round-trip needed.
    _glance_reply = maybe_glance_response(text)
    if _glance_reply is not None:
        print("  [glance] one-shot reply (focused window auto-attached)")
        reply = _glance_reply
    else:
        reply = get_response_with_animation(text)
    print(f"  JARVIS: {reply}")

    # Execute any [ACTION: ...] tokens, get the cleaned text for TTS
    spoken_text, action_results = parse_and_run_actions(reply)
    spoken_text = _apply_quip_layer(spoken_text, action_results)
    if spoken_text:
        _speak(spoken_text)

    # Speak verbatim-result actions (version_info, system_pulse, …) directly.
    # Their result is a finished sentence the user asked for, but they're not
    # in INFORMATIVE_ACTIONS, so the follow-up loop below never voices them —
    # without this the answer is logged but never spoken (the 2026-06-03
    # "you didn't speak it" bug). Deduped against the inline reply just spoken.
    _spoke_verbatim = _speak_verbatim_results(action_results, spoken_text)

    # If any informational actions ran (see_screen, etc.), feed the
    # results back so Bobert can actually report what he found.
    # Loop in case the follow-up itself emits another informative
    # action (e.g. screenshot → see_screen → final answer).
    # Also auto-trigger follow-up for any FAILED action so failures
    # don't go silently unreported to the user.
    FAIL_MARKERS = FAILURE_MARKERS  # canonical list — see core/failure_markers.py
    def _is_failure(result: str) -> bool:
        lower = result.lower()
        return any(m.lower() in lower for m in FAIL_MARKERS)

    current_results = action_results
    _chain_seen: set[str] = set()   # loop-break: actions already fired this chain
    _failed_seen: set[str] = set()  # actions that already returned a failure once
    # Agent mode gets a deeper follow-up loop so autonomous tasks
    # have more room to plan → execute → critique → repeat. Smart
    # and controlled modes keep the historical depth of 5.
    try:
        from core.mode_router import followup_loop_depth as _followup_depth
        _max_followup = _followup_depth(default=5)
    except Exception:
        _max_followup = 5
    for depth in range(_max_followup):
        informative = [
            (n, r) for (n, r, is_info) in current_results
            if is_info or _is_failure(r)
        ]
        if not informative:
            break
        # Break the re-prompt loop when a FAILING action is repeating.
        # Re-prompting the LLM on the same already-failed action
        # (click / focus_window / launch_app, …) just burns 1–3 s LLM
        # round-trips with no progress — and only _loop_actions were
        # guarded below, so a stuck real action could re-loop the full
        # depth (the 2026-05-30 audit's multi-second voice stall).
        # Give each failing action exactly one retry, then stop; info
        # actions still chain freely through the logic below.
        _failing_now = {n for (n, r) in informative if _is_failure(r)}
        _failing_repeat = _failing_now & _failed_seen
        if _failing_repeat:
            print(f"  [follow-up] action(s) failing repeatedly "
                  f"({', '.join(sorted(_failing_repeat))}) — stopping")
            break
        _failed_seen |= _failing_now
        # Loop detection: actions that should only fire ONCE per chain.
        # check_credits opens an offscreen browser, reads the balance,
        # and closes it — emitting it twice means JARVIS is going in
        # circles re-opening the billing page. Same logic for open_url
        # and web_search.
        _loop_actions = {"open_url", "web_search", "check_credits",
                         "_unverified_claim", "_dropped_step",
                         "_preemptive_hallucinated_claim"}
        # check_credits is also terminal — once it returns a balance,
        # there's nothing left to do but report it. Stop after the
        # first follow-up that reports the balance.
        _terminal_actions = {"check_credits"}
        _repeating = {n for n, _ in informative} & _chain_seen & _loop_actions
        if _repeating:
            print(f"  [follow-up] loop detected ({', '.join(_repeating)}) — stopping")
            break
        # If a terminal action has already run once, don't loop again —
        # the follow-up that reports the balance is the final word.
        if _terminal_actions & _chain_seen:  # pragma: no cover - dead branch: the sole terminal action (check_credits) is also in _loop_actions, so the loop-detect break above always fires first
            print(f"  [follow-up] terminal action already ran — stopping")
            break
        _chain_seen |= {n for n, _ in informative}
        print(f"  Reading results (depth {depth+1})…")
        _heartbeat()   # follow-up LLM call can be slow on local — keep watchdog fresh
        set_state("thinking")
        followup = get_followup_response(informative)
        if not followup:
            break
        print(f"  JARVIS: {followup}")
        f_spoken, current_results = parse_and_run_actions(followup)
        f_spoken = _apply_quip_layer(f_spoken, current_results)
        # Append follow-up to history so context carries forward
        conversation_history.append({"role": "assistant", "content": followup})
        if f_spoken:
            _speak(f_spoken)
        # A follow-up reply can itself emit a verbatim-result action (e.g. the
        # LLM chains system_pulse). Voice its result here too, deduped against
        # the follow-up prose just spoken.
        _spoke_verbatim |= _speak_verbatim_results(current_results, f_spoken)
    return reply


def _upgrade_task_signature(clean_tasks) -> str:
    """Stable signature of an upgrade summary's TASK LIST ONLY (timestamp
    excluded), so a re-write of the same tasks with a bumped ``upgraded_at`` is
    recognised as already-announced. sha256 over the sorted, cleaned task
    strings — order-independent so a reshuffled-but-identical list still
    matches."""
    joined = "\n".join(sorted(str(t) for t in clean_tasks))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _announce_upgrade_summary() -> None:
    """Boot-time spoken announcement of the last upgrade cycle's work, with
    DEDUP so the SAME task-set is never announced twice even if its timestamp
    was bumped.

    The (now-disabled) overnight engine used to re-write ``.last_upgrade_-
    summary.json`` with a FRESH ``upgraded_at`` but the SAME old task list, so
    every boot falsely re-announced days-old work as if it had just run. We now
    compute a signature of the TASK LIST only (see _upgrade_task_signature) and
    compare it to the last-announced signature persisted in UPGRADE_ANNOUNCED_-
    SIG_FILE:
      * signature matches the sidecar  → SKIP the spoken announcement (stale
        replay), but STILL delete the summary file in the finally.
      * new signature (genuine new upgrade) → announce as before, then write the
        new signature to the sidecar, then delete the summary.

    try/finally on the unlink so a corrupt summary can't get stuck re-announcing
    on every startup. Never raises — boot must continue regardless.
    """
    if not os.path.exists(UPGRADE_SUMMARY_FILE):
        return
    try:
        try:
            with open(UPGRADE_SUMMARY_FILE, "r", encoding="utf-8") as _sf:
                _summary = json.load(_sf)
            _tasks      = _summary.get("tasks", [])
            _when       = _summary.get("upgraded_at", "recently")
            _syntax_ok  = _summary.get("syntax_ok", True)
            _syn_errors = _summary.get("syntax_errors", [])

            # Strip the **date** — prefix from each task so it reads naturally
            _clean_tasks = []
            for _t in _tasks:
                _c = re.sub(r'^\*\*[^*]+\*\*\s*[—–-]+\s*', '', _t).strip()
                _clean_tasks.append(_c if _c else _t)

            # DEDUP: signature of the cleaned TASK LIST only (no timestamp). If it
            # matches what we last announced, this is a stale timestamp-bumped
            # replay — skip the speech but still delete the summary (finally).
            _sig = _upgrade_task_signature(_clean_tasks)
            _prev_sig = ""
            try:
                if os.path.exists(UPGRADE_ANNOUNCED_SIG_FILE):
                    with open(UPGRADE_ANNOUNCED_SIG_FILE, "r", encoding="utf-8") as _gf:
                        _prev_sig = _gf.read().strip()
            except Exception:
                _prev_sig = ""
            if _clean_tasks and _sig == _prev_sig:
                print("  [upgrade] tasks already announced — skipping stale replay")
                return

            _count = len(_clean_tasks)
            if _count > 0:
                _snippets = []
                for _t in _clean_tasks[:3]:
                    _short = _t.split(".")[0].strip()   # first sentence only
                    if len(_short) > 70:
                        _short = _short[:67] + "…"
                    _snippets.append(_short)
                _extra = f" and {_count - 3} more" if _count > 3 else ""
                _task_line = "; ".join(_snippets) + _extra
            else:
                _task_line = ""

            if _syntax_ok:
                _status_line = "All syntax checks passed."
            else:
                _err_names = ", ".join(_syn_errors[:2]) if _syn_errors else "unknown files"
                _status_line = f"Warning — syntax errors detected in {_err_names}. A manual review may be advisable, sir."

            if _count > 0:
                _announcement = (
                    f"Upgrade complete, sir — ran at {_when}. "
                    f"{_count} task{'s' if _count != 1 else ''} implemented: "
                    f"{_task_line}. {_status_line}"
                )
            else:
                # Claude Code ran but task list was empty (e.g. overnight idea gen)
                _announcement = (
                    f"I completed an upgrade at {_when}, sir. {_status_line}"
                )

            print(f"  [upgrade] announcing: {_announcement}")
            try:
                _speak(_announcement)
            except Exception as _se:
                print(f"  [upgrade] speak failed: {_se}")
            # Persist the just-announced task signature so a later boot that sees
            # the same tasks (timestamp bumped) stays silent. Best-effort: a
            # write failure only risks one duplicate announcement, never a crash.
            if _clean_tasks:
                try:
                    with open(UPGRADE_ANNOUNCED_SIG_FILE, "w", encoding="utf-8") as _gf:
                        _gf.write(_sig)
                except Exception as _we:
                    print(f"  [upgrade] couldn't persist announced signature: {_we}")
        except Exception as _e:
            print(f"  [startup] couldn't read upgrade summary: {_e}")
    finally:
        try:
            os.remove(UPGRADE_SUMMARY_FILE)
        except Exception:
            pass


def main():  # pragma: no cover - boot entrypoint + infinite main event loop (singleton, device/boot bring-up, while-True turn loop)
    global _system_prompt, last_speech_time, _last_recording_peak

    # Singleton lock FIRST — before any other startup work — so the stability
    # smoke test (and _boot_jarvis.ps1's lock-watcher) can confirm we're alive
    # within milliseconds. Anything heavier (console move, ctypes calls, blue/
    # green registration, logging) happens AFTER the lock is on disk.
    import atexit
    try:
        _enforce_singleton()
    except Exception as _se:
        # Never let singleton-check raise — better to run without a lock than
        # to crash silently in detached pythonw before writing one.
        print(f"  [singleton] check failed, proceeding without lock: {_se}")
    atexit.register(_release_singleton)

    # Post-lock verification: belt-and-suspenders. The early-boot lock and
    # _enforce_singleton both swallow some failure modes, so confirm the
    # lock file is actually on disk. If it isn't, the stability smoke test
    # (and any external lock-watcher) will hang for 30s — write a
    # boot_error marker and exit immediately so the test fails fast.
    if not os.path.exists(_LOCK_FILE):
        _root_dir = os.path.dirname(os.path.abspath(__file__))
        _err_name = ("jarvis_staging_boot_error.txt" if _is_staging()
                     else "jarvis_boot_error.txt")
        _err_path = os.path.join(_root_dir, _err_name)
        _msg = (f"[singleton] lock file missing after _enforce_singleton "
                f"(path={_LOCK_FILE}) — exiting to avoid lock-watcher hang")
        print(_msg)
        try:
            with open(_err_path, "w", encoding="utf-8") as _f:
                _f.write(_msg + "\n")
        except OSError:
            pass
        try:
            _data_dir = os.path.join(_root_dir, "data")
            os.makedirs(_data_dir, exist_ok=True)
            with open(os.path.join(_data_dir, "boot_failures.jsonl"),
                      "a", encoding="utf-8") as _f:
                _f.write(json.dumps({
                    "ts": time.time(),
                    "iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                         time.localtime()),
                    "kind": "post_lock_missing",
                    "lock_path": _LOCK_FILE,
                    "is_staging": _is_staging(),
                }) + "\n")
        except Exception:
            pass
        sys.exit(1)

    # Don't move the console in staging — it would yank the operator's
    # window for the smoke test. Also, the prod console is what the user
    # expects on their working monitor; staging boots silently in place.
    if not _is_staging():
        _move_console_to_monitor(CONSOLE_MONITOR)
    atexit.register(_shutdown_hud)
    atexit.register(_shutdown_tray)

    # Blue/green: register this PID into data/instances.json so the
    # upgrade pipeline and the running prod instance can both see us.
    # Heartbeat refresh happens later in the main loop's tick.
    if _bgm is not None:
        try:
            _version = _bgm.read_version()
            _bgm.register_instance(role=BLUE_GREEN_ROLE, version=_version)
            atexit.register(_bgm.unregister_instance)
            # Reflect our presence in deployment_state.json so observers
            # (tray, voice "are you the new one") know the picture.
            if BLUE_GREEN_ROLE == "staging":
                _bgm.write_state({
                    "staging_pid":     os.getpid(),
                    "staging_version": _version,
                })
            else:
                _bgm.write_state({
                    "prod_pid":     os.getpid(),
                    "prod_version": _version,
                })
        except Exception as _bge:
            print(f"  [blue-green] instance registration failed: {_bge}")

    # blue-green-2: pull any in-flight conversation tail + timers the previous
    # prod left in data/handoff.json (see _consume_blue_green_handoff). The two
    # pending values are applied to last_speech_time / active timers further
    # below, after the unconditional set_state('idle') reset.
    _pending_handoff_last_speech, _pending_handoff_timers = _consume_blue_green_handoff()

    setup_logging()
    if _log_file_path:
        print(f"Logging session to: {_log_file_path}")

    # Snap Windows into High Performance for the duration of the session.
    # Wrapped + best-effort: any failure (locked-down machine, missing
    # powercfg, AC-only restriction) must not block boot. Restoration
    # happens in the Ctrl-C shutdown path and via atexit as a safety net.
    _activate_high_performance_plan()
    atexit.register(_restore_prior_power_plan)

    # Three startup self-heal checks (API key + ping, cublas64_12.dll,
    # per-camera open smoke test). Replaces the bare ANTHROPIC_API_KEY
    # check this block used to perform — see _startup_preflight for the
    # incident history. Camera + cublas results gate downstream behaviour
    # (CAMERAS may shrink; _force_whisper_cpu_int8 may flip).
    _startup_preflight()

    _ensure_whisper()   # load now so first user utterance isn't delayed

    # Walk requirements.txt and warn loudly about any missing packages so
    # silent feature-disabling (psutil → no system monitor / no HUD CPU-RAM,
    # paho-mqtt → no Bambu monitor, etc.) never goes unnoticed again.
    # Runs AFTER whisper is loaded (so the model-load output isn't tangled
    # with the dep warnings) and BEFORE the spoken greeting (so the alert
    # is the first thing the user hears if something's broken).
    try:
        _missing_deps = check_dependencies()
    except Exception as _e:
        print(f"  [dep-check] check failed: {_e}")
        _missing_deps = []

    memory = load_memory()
    memory["conversation_count"] += 1
    save_memory(memory)
    _system_prompt = build_system_prompt(memory)

    if memory["facts"] or memory["sessions"]:
        print(
            f"Memory loaded: {len(memory['facts'])} facts, "
            f"{len(memory['projects'])} projects, "
            f"{len(memory['topics'])} topics, "
            f"{len(memory['sessions'])} past sessions"
        )

    robot_str = f"robot @ {ROBOT_IP}:{ROBOT_PORT}" if ROBOT_ENABLED else "VOICE-ONLY (no robot)"
    print(f"\nJ.A.R.V.I.S.  —  [{LOCATION}]  —  {robot_str}")
    _stt_label = f"{_stt_model_name or WHISPER_MODEL} ({_stt_device or 'cpu'})"
    print(f"AI: {AI_BACKEND}  |  TTS: {TTS_VOICE}  |  STT: whisper-{_stt_label}")
    print(f"Mic:      {get_current_mic_name()}")
    print(f"Speakers: {get_current_speaker_name()}")
    print(f"Cameras: {', '.join(c['label'] for c in CAMERAS)}")
    print(f"Learning: {'on' if LEARN_EVERY_TURN else 'off'}  |  "
          f"Proactive: {'on' if PROACTIVE_ENABLED else 'off'}  |  "
          f"PC control: {'on' if PC_CONTROL_ENABLED else 'off'}")
    print(f"Vision: {'on' if SCREEN_VISION_ENABLED else 'off'}  |  "
          f"UI automation: {'on' if UI_AUTOMATION_ENABLED else 'off'}  |  "
          f"Skills: {'on' if SKILLS_ENABLED else 'off'}")

    # Load any installed skills now so their actions are available
    load_skills()

    # Spin up the three always-on diagnostic daemons (self-diag, crash
    # watcher, deep auditor). Wired in after load_skills so the self-diag
    # daemon can lazy-import skills.self_diagnostic without races. Voice
    # control happens via the pause/resume/status actions registered
    # below in the ACTIONS dict.
    try:
        from core import diagnostic_daemons as _diag_daemons
        _diag_daemons.start_diagnostic_daemons()
        ACTIONS["pause_diagnostics"]    = _diag_daemons.act_pause_diagnostics
        ACTIONS["resume_diagnostics"]   = _diag_daemons.act_resume_diagnostics
        ACTIONS["diagnostic_daemon_status"] = _diag_daemons.act_diagnostic_status
        # Override the self_diagnostic.diagnostic_status entry so the same
        # voice phrase ("diagnostic status") now reports daemon last-run +
        # budget + findings instead of only the self-diag sweep summary.
        # Self-diag's terse summary is still reachable via "last diagnostic
        # run" / "are you ok".
        ACTIONS["diagnostic_status"] = _diag_daemons.act_diagnostic_status
    except Exception as _e:
        print(f"  [diag-daemons] startup failed: {_e}")

    # Restore the four tray-toggle preferences from the last shutdown so
    # the user's mute / ambient / pause / debug choices survive a restart.
    # Each handler that flips one of these cells already writes hud_state
    # via _write_hud_state, so reading the file here is the symmetric
    # restore step. Wrapped in try/except — the file may be absent on a
    # fresh install or corrupt after a hard crash.
    _restore_tray_toggle_state()

    # ── Ambient-learning boot (see AMBIENT_LEARNING_BOOT) ─────────────────────
    # Set by the upgrade/overnight pipeline when it relaunches JARVIS (or
    # JARVIS_AMBIENT_LEARNING=1). JARVIS comes up SILENT in standby: it listens
    # and keeps the learning subsystems fed from what it hears (see the sleep
    # loop), but won't speak until you say 'JARVIS'. We engage the existing
    # silent-until-wake standby loop rather than open a competing ambient stream.
    if AMBIENT_LEARNING_BOOT:
        with _standby_auto_engage_lock:
            _sleep_mode[0]   = True
            _standby_mode[0] = True
        _ambient_learning[0] = True
        print("  [ambient-learning] booted SILENT in standby — listening + "
              f"learning, say 'JARVIS' to wake (wake-resume: {WAKE_RESUME_MODE})")
        # Kick the multimodal fact-extractor so what JARVIS overhears in
        # standby is distilled into long-term memory on its interval loop.
        # Idempotent + best-effort; _run_once() no-ops (no LLM call) when the
        # window is empty, so it costs nothing during silence. Skipped in
        # staging (no mic → nothing to learn, and no LLM spend in a test box).
        if not _is_staging():
            _ext_mod = sys.modules.get("skill_ambient_multimodal_extract")
            if _ext_mod is not None:
                try:
                    _ext_msg = _ext_mod.ambient_extract_start("")
                    print(f"  [ambient-learning] fact-extractor: {_ext_msg}")
                except Exception as _ext_exc:
                    print(f"  [ambient-learning] fact-extractor start "
                          f"failed: {_ext_exc!r}")

    # Contextual-callback watcher: surfaces deferred announcements ('I'll
    # let you know when the print finishes') once their condition is met.
    # Started AFTER load_skills so skills that register custom conditions
    # have a chance to do so before the first tick. Actions are wired into
    # ACTIONS so the user can ask "list promises" verbally.
    if _promises is not None:
        try:
            _promises.register_actions(ACTIONS)
            _promises.start_watcher(announce_callable=proactive_announce)
        except Exception as e:
            print(f"  [promises] startup failed: {e}")

    # Start overnight self-improvement engine as a background thread
    if OVERNIGHT_UPGRADE_ENABLED:
        ot = threading.Thread(target=_overnight_upgrade_thread, daemon=True)
        ot.start()

    # Main-loop watchdog (bug-5): catches record_speech() hangs on flaky mics.
    # Seed the heartbeat NOW so the watchdog doesn't false-fire during the
    # slow tail of boot (whisper load, camera probe, iron_man_boot greeting).
    _main_loop_heartbeat[0] = time.time()
    threading.Thread(target=_main_loop_watchdog_thread, daemon=True).start()

    # Periodic session-summary checkpoint so the recall index survives an
    # unclean exit (the shutdown-only writer rarely runs — see the thread's
    # docstring + the 2026-05-30 audit finding).
    threading.Thread(target=_session_summary_checkpoint_thread,
                     daemon=True).start()

    # Background, once-per-day check against the latest GitHub release; queues a
    # single spoken nudge if a newer version is published. Skipped in staging.
    if UPDATE_CHECK_ENABLED and not _is_staging():
        threading.Thread(target=_update_check_thread, daemon=True).start()

    print("─" * 60)
    print("Press Ctrl-C to quit (session will be summarised to memory).\n")

    # Probe cameras BEFORE the tracking thread starts so we can rewrite
    # CAMERAS with whatever's actually plugged in, and report a single
    # spoken status line instead of the per-frame failure spam from the
    # tracking thread's own log lines. Staging skips the probe — its
    # CAMERA_PROBE_ENABLED override already false but belt-and-braces.
    _cam_status_msg = ""
    if CAMERA_PROBE_ENABLED and not _is_staging():
        working, failed = probe_cameras_and_update_config()
        if working:
            _cam_status_msg = "Face tracking online, sir."
        else:
            _cam_status_msg = "I'm afraid neither camera is cooperating, sir."
        print(f"  [cam-probe] result: working={working}, failed={failed}")

    # Start face tracking thread — skipped in staging since the camera
    # device is owned by prod for the duration of the smoke test.
    if not _is_staging():
        ft = threading.Thread(target=_face_tracking_thread, daemon=True)
        ft.start()

        # Start the focused-window tracker so maybe_glance_response can tell
        # when the user just switched windows and an ambiguous question
        # ("what is this?") should attach a fresh screenshot.
        _start_focus_tracker()

    # Launch the on-screen HUD overlay (separate process so its tkinter loop
    # doesn't fight the main thread).
    if HUD_ENABLED:
        _launch_hud()

    # Launch the full-virtual-screen reticle overlay so UI-automation actions
    # (click, type, focus_window, …) flash a 2-second target where they fired.
    if RETICLE_OVERLAY_ENABLED:
        _launch_reticle_overlay()

    # Launch the system-tray applet (animated arc-reactor icon + menu) and
    # start its companion threads: a 2 Hz drainer that processes commands
    # from the tray's right-click menu, and a 1 Hz publisher that bridges
    # the system_monitor + bambu_monitor skills into hud_state.json so the
    # tray subprocess can colour-code its icon (red on alert, orange on print).
    if TRAY_ENABLED:
        _launch_tray()
        threading.Thread(target=_tray_command_drainer, daemon=True).start()
        threading.Thread(target=_tray_state_publisher, daemon=True).start()

    set_state("idle")
    last_speech_time = time.time()

    # round5-M-5: if we resumed from a blue/green handoff, restore the
    # previous instance's last_speech_time so idle-proactive gating doesn't
    # treat the swap as a fresh user utterance. Then re-arm any timers that
    # were live at handoff via the timer-skill restore entry point (called
    # AFTER load_skills() so skill_timer is importable).
    if _pending_handoff_last_speech is not None:
        last_speech_time = _pending_handoff_last_speech
        print(f"  [blue-green] restored last_speech_time from handoff "
              f"({time.time() - last_speech_time:.1f}s ago)")
    if _pending_handoff_timers:
        _skill_timer_mod = sys.modules.get("skill_timer")
        if _skill_timer_mod is not None:
            try:
                _restore = getattr(_skill_timer_mod, "restore_timers", None)
                if callable(_restore):
                    _n_restored = _restore(_pending_handoff_timers)
                    print(f"  [blue-green] re-armed {_n_restored} "
                          f"timer(s) from handoff")
            except Exception as _rte:
                print(f"  [blue-green] timer restore failed: {_rte}")
        else:
            print("  [blue-green] timer restore skipped — "
                  "skill_timer not loaded")

    # Vocal startup — JARVIS "coming online" moment from the films.
    # iron_man_boot.py plays a ~1.5s suit power-on sting, drives a 4.5s
    # arc-reactor power-up animation on the HUD that scrolls INITIALISING →
    # DIAGNOSTICS → ONLINE, then speaks a single line:
    #   "JARVIS online. All systems nominal. Good <time-of-day>, sir."
    # See jarvis_todo.md 2026-05-27 10:04, iron_man_boot.
    #
    # Falls back to the older boot_sequence.py (inventory greeting) if
    # iron_man_boot fails to import or run, and finally to a plain greeting.
    _boot_used_sequence = False
    # Staging skips the cinematic boot — _speak is already routed to
    # replies.jsonl, but the sound-sting in iron_man_boot still opens
    # the audio device and would collide with prod's playback.
    if _is_staging():
        _boot_used_sequence = True
        print("  [staging] skipping iron_man_boot (audio device owned by prod)")
    try:
        if not _boot_used_sequence:
            import iron_man_boot as _iron_boot
            _boot_line = _iron_boot.play_iron_man_boot(
                speak_fn=_speak,
                write_hud_state=_write_hud_state,
                output_device=get_output_device(),
                tts_muted=bool(_tts_muted[0]),
            )
            conversation_history.append({"role": "assistant", "content": _boot_line})
            _boot_used_sequence = True
    except Exception as _ibe:
        print(f"  [iron_man_boot] failed, falling back to boot_sequence: {_ibe}")

    if not _boot_used_sequence:
        try:
            import boot_sequence as _boot_seq

            def _friendly_dev(raw: str) -> str:
                if not raw:
                    return ""
                # get_current_*_name() returns "[N] Microphone (Name), API"
                if raw.startswith("[") and "] " in raw:
                    raw = raw.split("] ", 1)[1]
                return _friendly_device_name(raw)

            _last_session_ts = 0.0
            try:
                _recent = pattern_memory.get_session_summaries("", limit=1)
                if _recent:
                    _last_session_ts = float(_recent[0].get("ts") or 0)
            except Exception as _e:
                print(f"  [boot_sequence] session lookup failed: {_e}")

            _n_skills = sum(1 for m in sys.modules if m.startswith("skill_"))

            _boot_line, _inv_line = _boot_seq.play_boot_sequence(
                speak_fn=_speak,
                write_hud_state=_write_hud_state,
                n_actions=len(ACTIONS),
                n_skills=_n_skills,
                mic_name=_friendly_dev(get_current_mic_name()),
                speaker_name=_friendly_dev(get_current_speaker_name()),
                last_session_ts=_last_session_ts,
            )
            conversation_history.append({"role": "assistant", "content": _boot_line})
            if _inv_line:
                conversation_history.append({"role": "assistant", "content": _inv_line})
            _boot_used_sequence = True
        except Exception as _be:
            print(f"  [boot_sequence] failed, using plain greeting: {_be}")

    if not _boot_used_sequence:
        # Fallback to the historical greeting if boot_sequence couldn't run
        if memory["conversation_count"] <= 1:
            greeting = "J.A.R.V.I.S. online and ready."
        elif memory["facts"]:
            greeting = "All systems online. Standing by, sir."
        else:
            greeting = "Online and ready."
        try:
            _speak(greeting)
            conversation_history.append({"role": "assistant", "content": greeting})
        except Exception as e:
            print(f"  [startup] greeting failed: {e}")

    # Camera probe result — spoken AFTER the greeting so the user hears the
    # cheerful 'online' message first, then the diagnostic.
    if _cam_status_msg:
        try:
            _speak(_cam_status_msg)
            conversation_history.append({"role": "assistant", "content": _cam_status_msg})
        except Exception as e:
            print(f"  [startup] cam-status speak failed: {e}")

    # Suit-up cinematic: on the day's first warm-restart, play the
    # holographic boot sequence (arc-reactor spin-up + diagnostics readout +
    # 'Welcome back, sir. Systems are yours.') instead of the plain
    # warm-restart greeting below. See jarvis_todo.md 2026-05-27 11:26,
    # suit_up_sequence. Once-per-day gated inside the skill.
    _suit_up_fired = ""
    try:
        import skills.suit_up as _suit_up
        _spk_name = ""
        try:
            _raw_spk = get_current_speaker_name()
            if _raw_spk.startswith("[") and "] " in _raw_spk:
                _raw_spk = _raw_spk.split("] ", 1)[1]
            _spk_name = _friendly_device_name(_raw_spk)
        except Exception:
            pass
        _suit_up_fired = _suit_up.maybe_play_morning_suit_up(
            speak_fn=_speak,
            write_hud_state=_write_hud_state,
            speaker_name=_spk_name,
        )
        if _suit_up_fired:
            conversation_history.append(
                {"role": "assistant", "content": _suit_up_fired}
            )
    except Exception as _se:
        print(f"  [suit_up] morning check failed: {_se}")

    # Warm-restart greeting: if the previous session ended within 18 hours
    # AND we have something concrete to resume (latest queued task / recent
    # commands / last session summary), offer to pick it back up. The
    # returned text intentionally embeds "At your service" / "I'm afraid"
    # so detect_tts_emotion() picks the matching confirmation / bad_news
    # preset — the spec's "vary tone with context" hook.
    # Skipped when the suit-up cinematic already played — that sequence
    # ends with "Welcome back, sir. Systems are yours." and stacking the
    # resume line on top would feel redundant.
    _resume_line = ""
    if not _suit_up_fired:
        try:
            _resume_line = maybe_session_resume_greeting()
        except Exception as _e:
            _resume_line = ""
            print(f"  [session_resume] check failed: {_e}")
    else:
        # Mark the resume hook as 'done' for this process so a later code
        # path can't double-fire the plain greeting on top of the cinematic.
        _session_resume_done[0] = True
    if _resume_line:
        try:
            print(f"  [session_resume] surfacing: {_resume_line}")
            _speak(_resume_line)
            conversation_history.append(
                {"role": "assistant", "content": _resume_line}
            )
        except Exception as e:
            print(f"  [session_resume] speak failed: {e}")

    # Surface a session-pattern observation if we've recorded enough sessions
    # to spot one ('it's Friday night, you usually wind down around now…')
    try:
        _pattern_remark = detect_startup_pattern()
    except Exception as _e:
        _pattern_remark = ""
        print(f"  [patterns] detect failed: {_e}")
    if _pattern_remark:
        try:
            print(f"  [patterns] surfacing: {_pattern_remark}")
            _speak(_pattern_remark)
        except Exception as e:
            print(f"  [patterns] speak failed: {e}")

    # Command-level pattern offers (separate from the session-level remark
    # above): "It's Friday evening, sir — shall I queue the Michael Jackson
    # essentials?" — fires at most once per pattern key per day, throttled
    # via memory/pattern_offers_state.json so multiple JARVIS restarts in a
    # window don't repeat the same offer.
    try:
        _cmd_offer = pattern_memory.maybe_pattern_offer()
    except Exception as _e:
        _cmd_offer = ""
        print(f"  [pattern_memory] offer check failed: {_e}")
    if _cmd_offer:
        try:
            print(f"  [pattern_memory] surfacing: {_cmd_offer}")
            _speak(_cmd_offer)
            conversation_history.append(
                {"role": "assistant", "content": _cmd_offer}
            )
        except Exception as e:
            print(f"  [pattern_memory] speak failed: {e}")

    # Announce changes from the last upgrade cycle, if any (deduped so a
    # timestamp-bumped replay of already-announced work stays silent).
    _announce_upgrade_summary()

    # Changelog announcement — when version has bumped since the last
    # user-facing announcement, summarise the latest CHANGELOG.md entry via
    # the existing LLM helper and speak it. See jarvis_todo.md
    # 2026-05-28 changelog-1. Wrapped end-to-end so a missing or malformed
    # changelog/version file never crashes boot.
    try:
        _ver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "data", "version.json")
        _changelog_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "CHANGELOG.md")
        if os.path.exists(_ver_path):
            with open(_ver_path, "r", encoding="utf-8") as _vf:
                _vdata = json.load(_vf)
            _cur_ver = _vdata.get("version")
            _last_ver = _vdata.get("last_announced_version")

            def _stamp_version_announced(_data: dict, _ver: str, _path: str) -> None:
                from datetime import datetime as _dt2
                _data["last_announced_version"] = _ver
                _data["last_announced_to_user_at"] = _dt2.now().isoformat(timespec="seconds")
                try:
                    from core.atomic_io import _atomic_write_json as _aw
                    _aw(_path, _data)
                except Exception:
                    with open(_path, "w", encoding="utf-8") as _wf:
                        json.dump(_data, _wf, indent=2)

            if _cur_ver and _last_ver is None:
                # First boot after the changelog system landed — silently
                # stamp; the spec explicitly says say nothing.
                _stamp_version_announced(_vdata, _cur_ver, _ver_path)
                print(f"  [changelog] first boot — silently set "
                      f"last_announced_version to {_cur_ver}")
            elif (_cur_ver and _last_ver and _cur_ver != _last_ver
                    and os.path.exists(_changelog_path)):
                with open(_changelog_path, "r", encoding="utf-8") as _cf:
                    _changelog_text = _cf.read()
                # Latest entry: from the first "## v" header to the next
                # "---" separator (entries are newest-first).
                _hdr = re.search(r"^## v.+$", _changelog_text, re.MULTILINE)
                _latest_entry = ""
                if _hdr:
                    _after_hdr = _changelog_text[_hdr.end():]
                    _sep = re.search(r"^---\s*$", _after_hdr, re.MULTILINE)
                    _latest_entry = (
                        _after_hdr[:_sep.start()] if _sep else _after_hdr
                    ).strip()
                _summary = ""
                if _latest_entry:
                    _system = (
                        "You are J.A.R.V.I.S. Summarise the following CHANGELOG.md "
                        "entry into ONE to THREE concise sentences spoken in your "
                        "own voice for the user (he/sir). Lead with phrasing like "
                        "'Sir, since we last spoke...'. Call out new capabilities "
                        "by name and the rough number of bug fixes if visible. "
                        "End with a brief pointer to the full changelog at "
                        "C:\\JARVIS\\CHANGELOG.md. Plain prose only — no markdown, "
                        "no bullets, no headers."
                    )
                    _user = (
                        f"Previous announced version: {_last_ver}\n"
                        f"Current version: {_cur_ver}\n\n"
                        f"Latest changelog entry:\n\n{_latest_entry[:8000]}"
                    )
                    try:
                        _summary = (_llm_quick(_system, _user, max_tokens=300) or "").strip()
                    except Exception as _le:
                        print(f"  [changelog] LLM summarise failed: {_le}")
                if _summary:
                    print(f"  [changelog] announcing v{_last_ver}→v{_cur_ver}: {_summary}")
                    try:
                        _speak(_summary)
                        conversation_history.append(
                            {"role": "assistant", "content": _summary}
                        )
                    except Exception as _spe:
                        print(f"  [changelog] speak failed: {_spe}")
                else:
                    print(f"  [changelog] no summary produced — skipping speech "
                          f"(latest_entry_len={len(_latest_entry)})")
                # Stamp regardless of whether the summary spoke so a missing
                # LLM backend or empty entry doesn't loop on every boot.
                _stamp_version_announced(_vdata, _cur_ver, _ver_path)
    except Exception as _ce:
        print(f"  [changelog] announcement block failed: {_ce}")

    # _drain_injected_command() / _speak_pending() and their queue-path
    # constants now live at module scope (above) so they no longer nest
    # inside main().

    try:
        # Re-seed the main-loop watchdog heartbeat right before the loop
        # starts iterating — the boot tail (TTS greeting, camera probe)
        # between launching the watchdog and this point can run several
        # seconds, and we don't want a slow boot to look like a stall.
        _main_loop_heartbeat[0] = time.time()
        while True:
            if _blue_green_loop_tick():
                return
            # Check for an injected command BEFORE we block on the mic. An
            # external tester (Claude Code, tools/say_to_jarvis.py, smoke
            # harness) can drop one in injected_commands.json to feed a
            # voice-equivalent user turn into the loop. The handlers below
            # branch on `_injected_text is not None` to skip record_speech
            # + transcribe and synthesise safe pass-through metadata.
            _injected_text = _drain_injected_command()

            # ── SLEEP / STANDBY MODE — only listen for the wake phrase ────────
            if _sleep_mode[0]:
                _handle_sleep_standby(_injected_text)
                continue
            # ── NORMAL MODE ───────────────────────────────────────────────────

            # Capture the next utterance (inject or mic->Whisper) and drain
            # any queued speech; short-circuit the loop on a reminder /
            # silence-timeout / too-short clip. See _capture_utterance.
            _cap = _capture_utterance(_injected_text, memory)
            if _cap is None:
                continue
            text, conf = _cap

            # ── AMBIENT MUSIC DETECTION → auto-standby ────────────────────────
            # Whisper emits markers like [Music] / ♪ when the mic picks up
            # song audio it can't transcribe as words. If we see those AND
            # JARVIS itself didn't recently kick off playback, count it
            # toward an auto-standby trigger so we stop pestering the user
            # with hallucinated transcriptions of overheard music.
            if _handle_ambient_music(text):
                continue

            # ── Ambient-music gate (2026-05-30 runtime-log audit) ──────────
            # The marker check above only catches Whisper's [Music]/♪ tags;
            # CLEAN song lyrics transcribe as ordinary words and slipped past
            # it, so JARVIS was replying to the user's music/TV/another person
            # (~60% of "commands" in the logs — the user repeatedly told it to
            # "stop listening"). Reuse the standby detector's TESTED state:
            # `should_refuse_wake` returns False unless SUSTAINED (>15s) room
            # music is active, and even then lets a clear leading "JARVIS"
            # through — so quiet-room turns are unaffected and the user can
            # still command over music by prefixing the wake word. Anything
            # else while music plays is treated as overheard audio and dropped.
            from core.config import AMBIENT_MUSIC_REFUSE_WAKE as _refuse_over_music
            if _refuse_over_music and _audio_music_should_refuse_wake(text):
                print(f"  [ambient-music] sustained room music active — "
                      f"ignoring non-wake utterance: '{text[:40]}'")
                set_state("idle")
                continue

            valid, reason = is_valid_speech(text, conf, peak_rms=_last_recording_peak)
            if not valid:
                # Show what got dropped so user can tune thresholds if needed
                snippet = (text[:60] + "…") if len(text) > 60 else text
                print(f"  [filter] dropped: '{snippet}' — {reason}")
                set_state("idle")
                continue

            print(f"  You:    {text}")
            # Rolling 5-line history feeds the holographic HUD v2
            # scrolling transcript panel. Cap at 5 entries here so the
            # JSON file stays small.
            try:
                with _hud_state_lock:
                    _hist = list(_hud_state_cache.get("transcript_history") or [])
            except Exception:
                _hist = []
            _hist.append(text)
            _hist = _hist[-5:]
            _write_hud_state(last_transcript=text,
                             last_transcript_at=time.time(),
                             transcript_history=_hist)

            # ── BRIEFING CARD DISMISS — "thank you, JARVIS" closes a card ─────
            # Doesn't consume the turn; the LLM still gets to respond to the
            # thanks naturally. Fail closed if hud_card isn't importable.
            try:
                import hud_card as _hud_card  # noqa: WPS433 (lazy local import)
                if _hud_card.is_card_active() and _hud_card.matches_dismiss_phrase(text):
                    _hud_card.dismiss_card()
                    print("  [hud_card] dismissed by phrase")
            except Exception:
                pass

            # ── SLEEP TRIGGER — check before sending to LLM ───────────────────
            # Two-gate check: (a) utterance must be short (≤ 6 words) AND
            # (b) the phrase must match with word boundaries — so casual
            # mentions like "if I say stop listening it should…" don't
            # trigger sleep mid-explanation.
            if _handle_sleep_triggers(text):
                continue

            # ── SHUTDOWN PROMPT — yes/no router for the overnight-protocol
            #    prompt armed by a previous SHUTDOWN_TRIGGER_PHRASE. Runs FIRST
            #    so a 'yes' answer doesn't get swallowed by any other handler.
            #    Returns True only when the message was consumed (YES / NO /
            #    reinforced-shutdown branches); an unrelated reply clears the
            #    flag and returns False so the original utterance still routes
            #    through normal LLM dispatch below.
            if _handle_shutdown_prompt(text):
                set_state("idle")
                continue

            # ── SHUTDOWN TRIGGER — arms the overnight-first prompt for any
            #    of the ambiguous shutdown phrases ('shut down' / 'power off'
            #    / 'go offline' / etc.). The NEXT utterance is then handled
            #    by _handle_shutdown_prompt above. Unambiguous bedtime phrases
            #    ('goodnight' / 'going to bed') are NOT in this list — they
            #    still route directly to start_overnight_upgrade via the LLM.
            if _check_and_arm_shutdown_prompt(text):
                set_state("idle")
                continue

            # If the autocorrect layer asked 'did you mean X or Y' on the
            # previous turn, interpret this utterance as the user's pick.
            # Runs BEFORE handle_confirmation_response because a disambig
            # 'yes' is meaningfully different from a confirmation 'yes' —
            # the disambig handler short-circuits to the picked action and
            # clears its own queue; an unrelated reply returns False so the
            # original utterance still falls through to normal dispatch.
            if handle_autocorrect_disambig_response(text):
                set_state("idle")
                continue

            # If a high-risk action is pending, treat this utterance as the
            # confirmation/cancellation rather than a new prompt.
            if handle_confirmation_response(text):
                set_state("idle")
                continue

            # Command-level pattern memory: log this utterance with day-of-week
            # so get_patterns() can mine repeated targets (artist names, app
            # names) at the same time-of-week. Logged AFTER the sleep / standby
            # / confirmation guards so trigger phrases don't pollute the log.
            try:
                pattern_memory.record_voice_command(text)
            except Exception as _e:
                print(f"  [pattern_memory] record failed: {_e}")

            # Late-night commentary (01:00–04:59). Either an in-character
            # remark before complying, or an acknowledgement that the user
            # has muted further remarks for the night.
            _ln_remark = maybe_late_night_remark(text, memory)
            if _ln_remark:
                print(f"  [late-night] {_ln_remark}")
                _speak(_ln_remark)
                # Deliberately NOT appended to conversation_history. This
                # remark is a spoken aside that PRECEDES the user's pending
                # turn; appending it as an assistant message here — before
                # _call_llm() appends the user message just below — produced
                # two consecutive assistant turns and a Claude 400 "roles
                # must alternate", swallowed into a silent downgrade-to-local
                # on every 01:00–04:59 turn. 2026-05-30 audit.

            # Explicit multi-agent briefing? Fan out to the sub-agent
            # orchestrator (opt-in; default path untouched). See _maybe_orchestrate.
            if _maybe_orchestrate(text):
                continue

            if _run_voice_shortcuts(text):
                continue

            reply = _run_llm_dispatch(text)

            # Real-time learning: extract facts in background (non-blocking)
            learn_from_turn(text, reply, memory)

            # Ambient-learning 'answer_then_quiet': this normal-mode turn was the
            # ONE reply granted after the wake word — now drop straight back to
            # silent standby so the next interaction needs 'JARVIS' again. Only a
            # fully-processed command turn reaches here (music / invalid-speech /
            # silence-timeout paths all 'continue' earlier), so the user always
            # gets their answer BEFORE JARVIS goes quiet. 'stay_talkative' never
            # arms the flag, so it stays interactive until a sleep/standby phrase.
            if _resume_to_ambient[0]:
                _resume_to_ambient[0] = False
                with _standby_auto_engage_lock:
                    _sleep_mode[0]   = True
                    _standby_mode[0] = True
                try:
                    _write_hud_state(sleep_mode=True, standby_mode=True)
                except Exception:
                    pass
                print("  [ambient-learning] reply delivered — back to silent "
                      "standby (say 'JARVIS' for more)")

            # Rebuild prompt with any newly-learned facts after a short delay
            # so the very next turn already has them. Daemon so Ctrl-C exits
            # cleanly even if a Timer is pending. Reload memory FRESH from disk
            # — learn_from_turn's background worker writes new facts via
            # merge_memory() under _memory_lock, NOT into the main loop's
            # stale `memory` local; building from that local meant this rebuild
            # never actually saw the new facts (the whole point of the delay).
            # 2026-05-30 deep audit.
            _t = threading.Timer(2.0, lambda: globals().__setitem__(
                "_system_prompt", build_system_prompt(load_memory())
            ))
            _t.daemon = True
            _t.start()

            print()

    except KeyboardInterrupt:
        print("\nShutting down…")
        # Stop any audio that might be mid-play/wait so sd.wait() doesn't block
        try: sd.stop()
        except Exception: pass
        # Signal background threads
        _face_track_stop.set()
        _focus_tracker_stop.set()
        set_state("sleep")
        _shutdown_hud()
        _shutdown_tray()
        # Save the lightweight session-pattern entry first (no LLM call,
        # always quick) so the pattern history survives even if the
        # LLM-driven summary times out below.
        try:
            save_session_pattern()
        except Exception as _e:
            print(f"  [patterns] save failed at shutdown: {_e}")
        # Session save calls the LLM — give it a short window then force exit
        saver = threading.Thread(
            target=save_session_to_memory, args=(memory,), daemon=True
        )
        saver.start()
        saver.join(timeout=8)
        if saver.is_alive():
            print("  (session save timed out — exiting anyway)")
        # Restore the user's prior power plan before the forced _exit (which
        # bypasses atexit handlers entirely).
        try:
            _restore_prior_power_plan()
        except Exception:
            pass
        close_log()
        # Force exit even if any thread is wedged in a C-level blocking call
        os._exit(0)


if __name__ == "__main__":  # pragma: no cover - boot entrypoint; only runs when executed as a script, never under the import-based test harness
    # Make `import bobert_companion` and sys.modules.get("bobert_companion")
    # from skills + core/* resolve to THIS running module. Run as a script,
    # JARVIS lives in sys.modules["__main__"]; without this alias a later
    # `import bobert_companion` (e.g. core/actions._bc()) builds a SEPARATE
    # module object with its OWN globals — which silently defeated the
    # record_speech mic tap (the standby loop tapped the duplicate's empty
    # _record_speech_taps / always-False _record_speech_active and fell back
    # to opening a competing mic stream every 5s) and is the root of the
    # re-import self-kill class. Aliasing here means there is only ever ONE
    # bobert_companion module object. 2026-05-30 deep audit.
    sys.modules["bobert_companion"] = sys.modules["__main__"]

    if "--list-cameras" in sys.argv:
        list_cameras()
        sys.exit(0)
    if "--list-mics" in sys.argv or "--list-microphones" in sys.argv:
        list_microphones()
        sys.exit(0)
    if "--list-speakers" in sys.argv or "--list-outputs" in sys.argv:
        list_speakers()
        sys.exit(0)
    if "--list-monitors" in sys.argv:
        list_monitors_cli()
        sys.exit(0)
    main()
