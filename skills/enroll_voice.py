"""
Voice-enrollment skill for JARVIS multi-user mode.

Voice trigger
-------------
  "JARVIS, learn my voice"        — record ~15 s of speech and save under
                                    your name (the configured user, or whatever
                                    VOICE_ID_DEFAULT_USER is set to) — see also
                                    `enroll_voice <name>` if you'd like to
                                    enroll someone else.
  "JARVIS, who's talking"         — say which enrolled speaker is currently
                                    active (live identification on a 2 s
                                    sample).
  "JARVIS, list enrolled voices"  — report every enrolled speaker.
  "JARVIS, forget <name>'s voice" — delete that speaker's voiceprint.
  "JARVIS, set active speaker <name>"
                                  — manually pin the active speaker.

Recording
---------
The skill opens a sounddevice InputStream on the same input device the
main companion is using (via `bobert_companion.get_input_device()`).
That keeps it in sync if the user has overridden the default mic at
runtime, e.g. with the existing `/mic ...` admin command. Falls back to
the system default if bobert_companion isn't importable (running the
skill stand-alone).

Audio is float32 mono at 16 kHz which is exactly what Resemblyzer
expects.

Permissions
-----------
The very first speaker enrolled is auto-promoted to "owner" — gets
sudo / shell / memory_write granted. Every subsequent enrollment lands
in a conservative default set (no sudo, no shell, no memory_write).
Callers can re-grant any flag via `voice_id.grant(name, capability)`.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional

import numpy as np


# Resemblyzer expects 16 kHz mono.
ENROLL_SAMPLE_RATE = 16000
# 15 s gives Resemblyzer enough variation to build a robust embedding while
# staying short enough that family members don't need to
# monologue. Per-task spec: 10-15 seconds of audio per speaker.
ENROLL_SECONDS     = 15.0
IDENTIFY_SECONDS   = 2.0


def _voice_id():
    try:
        from core import voice_id  # type: ignore
        return voice_id
    except Exception as e:
        print(f"  [enroll_voice] could not import core.voice_id: {e}")
        return None


def _bobert():
    return sys.modules.get("bobert_companion") or sys.modules.get("__main__")


def _input_device() -> Optional[int]:
    bc = _bobert()
    if bc is not None:
        getter = getattr(bc, "get_input_device", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                pass
    return None


def _default_user() -> str:
    bc = _bobert()
    if bc is not None:
        name = getattr(bc, "VOICE_ID_DEFAULT_USER", None) or getattr(bc, "USER_NAME", None)
        if isinstance(name, str) and name.strip():
            return name.strip()
    env = os.environ.get("VOICE_ID_DEFAULT_USER") or os.environ.get("JARVIS_USER")
    if env and env.strip():
        return env.strip()
    return "user"


def _record_seconds(seconds: float) -> Optional[np.ndarray]:
    """Block-record `seconds` of float32 mono audio at 16 kHz.

    Returns None if sounddevice isn't installed or the stream errors out.
    """
    # Prefer the companion's shared mic buffer when available — it taps the
    # wake-word listener's persistent InputStream so we don't trigger a
    # WASAPI "device in use" rejection by opening a second exclusive stream
    # on the same input device (the sd.rec path this used to fall back to
    # was the bug; it is gone — see the H-4 note in _record_seconds).
    bc = _bobert()
    if bc is not None:
        getter = getattr(bc, "get_mic_buffer", None)
        if callable(getter):
            try:
                buf = getter(seconds, ENROLL_SAMPLE_RATE)
            except Exception as e:
                print(f"  [enroll_voice] get_mic_buffer raised ({e}); "
                      f"falling back to local capture")
                buf = None
            if buf is not None and getattr(buf, "size", 0) > 0:
                return np.asarray(buf, dtype=np.float32)

    try:
        import sounddevice as sd
    except Exception as e:
        print(f"  [enroll_voice] sounddevice unavailable: {e}")
        return None

    # TEARDOWN-GATE CLAIM (2026-08-14): this local fallback capture (the
    # private InputStream below) sets none of the monolith's
    # classic mic-ownership flags, so bobert's _refresh_devices could run its
    # destructive sd._terminate()/_initialize() under a live callback
    # (0xc0000374). Claim bobert_companion._enroll_capture_active for the
    # whole capture span via bc._pa_claim_owner when the host exposes the
    # gate; degrade to the old unguarded behaviour when it doesn't
    # (stand-alone runs, older monoliths, unit-test doubles). A False claim
    # means a reinit stayed in flight past the bounded wait — bail like a
    # capture failure. Released in the finally at the bottom, AFTER the
    # stream teardown (close-then-release).
    _gate_release = None
    _gate_bc = _bobert()
    if _gate_bc is not None:
        _claim_fn = getattr(_gate_bc, "_pa_claim_owner", None)
        _release_fn = getattr(_gate_bc, "_pa_release_owner", None)
        _enroll_cell = getattr(_gate_bc, "_enroll_capture_active", None)
        if (callable(_claim_fn) and callable(_release_fn)
                and isinstance(_enroll_cell, list) and _enroll_cell):
            try:
                if not _claim_fn(_enroll_cell, timeout=1.0):
                    print("  [enroll_voice] PortAudio reinit in flight — "
                          "cannot capture right now, try again in a moment")
                    return None
                _gate_release = (_release_fn, _enroll_cell)
            except Exception:
                _gate_release = None   # gate bookkeeping failed — old behaviour
    try:
        device = _input_device()
        # H-4, SECOND COPY (2026-08-20). This used to open with
        # ``sd.rec(...)`` + a daemon in ``sd.wait()`` + ``sd.stop()`` on
        # overrun, falling back to the private InputStream below only when
        # that raised. All three are PROCESS-GLOBAL calls and the same trio
        # skills/self_diagnostic's mic probe was rebuilt to remove:
        #
        #   * sounddevice publishes exactly ONE module-global
        #     ``_last_callback``, and ``_CallbackContext.start_stream`` opens
        #     with ``stop()`` — which stops AND CLOSES whatever the previous
        #     play/rec context left there. So this capture closed the
        #     monolith's live TTS playback stream mid-sentence, and a
        #     concurrent ``sd.play()`` (the timer skill, the mid-task-status
        #     timer, the tray drainer, a proactive announcement — all of which
        #     can speak while enrolment runs on the main voice thread) closed
        #     THIS recording stream while our daemon sat in ``sd.wait()``,
        #     whose ``finally`` then closed the same handle from a second
        #     thread. ``_StreamBase.close`` runs Pa_CloseStream and only THEN
        #     nulls ``_ptr``, with no lock: two concurrent Pa_CloseStream on
        #     one WASAPI stream is the 0xc0000374 heap corruption.
        #   * The ``sd.stop()`` on the overrun path fired against whatever
        #     ``_last_callback`` held AT THAT MOMENT — possibly the live TTS
        #     stream owned by play_with_lipsync's single-toucher reaper. That
        #     is verbatim the cross-close H-3 deleted from four other copies.
        #   * The abandoned ``_await_rec`` daemon's eventual close was never
        #     registered with ``_pa_close_pending``, so _refresh_devices could
        #     run ``sd._terminate()`` on top of it.
        #
        # A comment shipped earlier today blessed that ``sd.stop()`` as "a
        # DIFFERENT and legitimate call — that path owns ``_last_callback``
        # because it called sd.rec itself". The premise is false:
        # ``_last_callback`` is ONE process-global slot with no ownership at
        # all, and this capture holds no _SPEAK_LOCK.
        #
        # The private InputStream below has none of those properties: it is
        # never published into ``_last_callback``, so no ``sd.play()`` can
        # stop or close it and no ``sd.wait()`` of ours can bind to a play
        # context — exactly ONE thread makes native calls on it, and its
        # teardown goes through bobert's _safe_close_stream (which registers
        # the abandonable close with the teardown gate). It was already the
        # known-good fallback here; now it is the only path.
        import queue
        chunks: list[np.ndarray] = []
        q: queue.Queue = queue.Queue()

        def _cb(indata, frames, time_info, status):  # noqa: ARG001
            mono = indata[:, 0] if indata.ndim > 1 else indata
            q.put(mono.astype(np.float32, copy=False).copy())

        # 2026-05-29 silent-crash fix: avoid `with sd.InputStream(...)`. The
        # context manager's __exit__ invokes sd close() unguarded, which has
        # SIGSEGV'd in production. Route teardown through bobert_companion's
        # _safe_close_stream when available so close runs on a daemon thread.
        bc_close = None
        bc = _bobert()
        if bc is not None:
            bc_close = getattr(bc, "_safe_close_stream", None)
        try:
            stream = sd.InputStream(
                samplerate=ENROLL_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=1024,
                device=device,
                callback=_cb,
            )
        except Exception as e:
            print(f"  [enroll_voice] InputStream open failed: {e}")
            return None
        try:
            stream.start()
            start = time.time()
            while (time.time() - start) < seconds:
                try:
                    chunks.append(q.get(timeout=0.2))
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"  [enroll_voice] InputStream failed: {e}")
            return None
        finally:
            if callable(bc_close):
                try:
                    bc_close(stream)
                except Exception:
                    pass
            else:
                # Fallback when running stand-alone (no bobert_companion loaded):
                # mirror the helper inline so we never call close() on the caller
                # thread without a timeout guard.
                import threading as _threading
                try:
                    stream.stop()
                except Exception:
                    pass
                _done = _threading.Event()
                def _close_in_daemon():
                    try:
                        stream.close()
                    except Exception:
                        pass
                    finally:
                        _done.set()
                _threading.Thread(target=_close_in_daemon, daemon=True).start()
                if not _done.wait(timeout=2.0):
                    # H-3 (2026-08-20) — STALE-DUPLICATE HALF, deleted in all
                    # FIVE copies (bobert_companion._safe_close_stream,
                    # core/wake_word, skills/ambient_listen, this one, and
                    # core/actions._release_native_resources — that fifth one
                    # was MISSED on the first pass and found by the same-day
                    # review, which is exactly why the count is written down
                    # here). The old
                    # comment claimed sd.stop() forces "every PortAudio stream"
                    # to stop; sounddevice 0.5.5 (sounddevice.py:406-418) stops
                    # and closes ONLY `_last_callback`, published solely by
                    # sd.play()/sd.rec()/sd.playrec(). `stream` here is an
                    # explicitly constructed InputStream, so the call could not
                    # free this handle — while it COULD close the host's live
                    # TTS playback stream (double Pa_CloseStream on one WASAPI
                    # stream -> 0xc0000374). Abandon the daemon; it dies with
                    # the process. (The sd.rec/sd.wait/sd.stop rung that used
                    # to sit above this one — and the note that blessed its
                    # sd.stop() as "different and legitimate" — is gone: see
                    # the H-4 comment at the top of this function.)
                    print("  [enroll_voice] stream.close hung >2.0s — "
                          "abandoning the close daemon (it dies with the "
                          "process). sd.stop() is deliberately NOT called: it "
                          "cannot free this InputStream and would instead "
                          "close a live TTS playback stream.")

        if not chunks:
            return None
        return np.concatenate(chunks).astype(np.float32, copy=False)
    finally:
        # Release AFTER every capture/teardown path above (incl. the inner
        # finally's stream close) — close-then-release, so the flag covers the
        # streams' whole native lifetime.
        if _gate_release is not None:
            try:
                _gate_release[0](_gate_release[1])
            except Exception:
                pass


def _say(text: str) -> None:
    """Speak via the main companion if it's loaded; otherwise just print."""
    bc = _bobert()
    if bc is not None:
        say = getattr(bc, "say", None) or getattr(bc, "synthesise", None)
        if callable(say):
            try:
                say(text)
                return
            except Exception:
                pass
    print(f"  [enroll_voice] {text}")


# ── action handlers ─────────────────────────────────────────────────────────

def enroll_voice(arg: str = "") -> str:
    """Record ENROLL_SECONDS of speech and save it as the user's voiceprint."""
    vid = _voice_id()
    if vid is None:
        return "Voice ID core is missing, sir. I could not load core.voice_id."
    if not vid.is_available():
        return ("Resemblyzer isn't installed, sir. "
                "Run `pip install resemblyzer` and try again.")

    name = (arg or "").strip() or _default_user()
    _say(f"Listening. Please speak naturally for {int(ENROLL_SECONDS)} seconds, sir.")
    audio = _record_seconds(ENROLL_SECONDS)
    if audio is None or audio.size == 0:
        return "I couldn't capture the mic, sir. Voice enrollment aborted."

    result = vid.enroll_from_audio(name, audio, ENROLL_SAMPLE_RATE, append=True)
    if not result.get("ok"):
        return f"Enrollment failed, sir: {result.get('error', 'unknown error')}."

    samples = result.get("sample_count", 1)
    suffix = "" if samples == 1 else f" That's sample {samples} for you."
    return (f"Voiceprint saved for {result.get('name', name)}, sir.{suffix} "
            f"I will recognise that voice from now on.")


def whos_talking(arg: str = "") -> str:
    """Sample IDENTIFY_SECONDS of audio and report the matched speaker."""
    vid = _voice_id()
    if vid is None:
        return "Voice ID core is missing, sir."
    enrolled = vid.list_enrolled()
    if not enrolled:
        return "No voiceprints are enrolled, sir. We're still in single-user mode."
    if not vid.is_available():
        return "Resemblyzer is unavailable, sir; I can't identify speakers right now."

    audio = _record_seconds(IDENTIFY_SECONDS)
    if audio is None or audio.size == 0:
        return "I couldn't capture the mic, sir."
    name, score = vid.identify_speaker(audio, ENROLL_SAMPLE_RATE)
    if name is None:
        return (f"That voice doesn't match anyone enrolled, sir. "
                f"(best similarity {score:.2f}, threshold "
                f"{vid.CONFIDENCE_THRESHOLD:.2f})")
    return f"That sounds like {name}, sir. (confidence {score:.2f})"


def list_enrolled_voices(arg: str = "") -> str:
    vid = _voice_id()
    if vid is None:
        return "Voice ID core is missing, sir."
    names = vid.list_enrolled()
    if not names:
        return "No voiceprints enrolled yet, sir. Single-user mode is active."
    active = vid.get_active_speaker()
    pretty = ", ".join(names)
    if active:
        return f"Enrolled voices, sir: {pretty}. Active speaker: {active}."
    return f"Enrolled voices, sir: {pretty}."


def forget_voice(arg: str = "") -> str:
    vid = _voice_id()
    if vid is None:
        return "Voice ID core is missing, sir."
    name = (arg or "").strip()
    if not name:
        return "Tell me whose voiceprint to forget, sir."
    if vid.forget_speaker(name):
        return f"Forgotten {name}'s voiceprint, sir."
    return f"I don't have a voiceprint enrolled for {name}, sir."


def set_active_voice(arg: str = "") -> str:
    vid = _voice_id()
    if vid is None:
        return "Voice ID core is missing, sir."
    name = (arg or "").strip()
    if not name:
        vid.set_active_speaker(None)
        return "Active speaker cleared, sir."
    if vid.set_active_speaker(name):
        return f"Active speaker set to {name}, sir."
    return f"I don't have a voiceprint enrolled for {name}, sir."


def voice_id_status(arg: str = "") -> str:
    vid = _voice_id()
    if vid is None:
        return "Voice ID core is missing, sir."
    s = vid.encoder_status()
    if not s["encoder_loaded"]:
        return (f"Voice ID is offline, sir: "
                f"{s.get('encoder_error') or 'encoder not loaded'}.")
    enrolled = ", ".join(s["enrolled"]) or "(none)"
    return (f"Voice ID is online, sir. Enrolled: {enrolled}. "
            f"Active speaker: {s.get('active_speaker') or 'none'}. "
            f"Confidence threshold {s['threshold']:.2f}.")


# ── skill registration ─────────────────────────────────────────────────────

def register(actions: dict) -> None:
    actions["enroll_voice"]          = enroll_voice
    actions["learn_my_voice"]        = enroll_voice
    actions["whos_talking"]          = whos_talking
    actions["who_is_talking"]        = whos_talking
    actions["identify_speaker"]      = whos_talking
    actions["list_enrolled_voices"]  = list_enrolled_voices
    actions["enrolled_voices"]       = list_enrolled_voices
    actions["forget_voice"]          = forget_voice
    actions["set_active_speaker"]    = set_active_voice
    actions["voice_id_status"]       = voice_id_status
