"""
Voice-clone control skill for JARVIS — the voice-facing surface over
``core.voice_clone`` (Resemble AI Chatterbox).

Actions registered:
  list_voice_profiles     — report the enrolled clone profiles (consented ones
                            usable; flags any that lack consent).
  set_voice_profile <name>— select an active clone profile + enable the clone.
                            Refuses an unknown profile or one without consent
                            (never silently arms an un-consented voice).
  voice_clone_status      — say whether cloning is on, which profile, and
                            whether the engine (chatterbox + CUDA) is available.
  disable_voice_clone     — turn cloning back off (revert to the normal ladder).

Each returns ONE finished, user-facing sentence and does NOT self-speak, so
these names go in bobert_companion.SPEAK_RESULT_VERBATIM_ACTIONS (near the
gpu_usage entries) — the main loop voices the returned string verbatim.

ETHICS: this skill only ever SELECTS among profiles that were enrolled with an
explicit consent flag (the owner's own voice, or a JARVIS in-character
non-celebrity voice). It cannot create a profile — enrollment is out-of-band
via ``tools/enroll_voice.py``. The consent gate lives in
``core.voice_clone.profile_is_usable`` and is re-checked here before arming.

Everything heavy (chatterbox, torch, CUDA) stays inside core.voice_clone and is
lazily imported there, so this skill imports cleanly on a headless / no-GPU CI
box and every action degrades to an honest one-liner.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional


def _bobert():
    """Resolve the running monolith module (whichever name it loaded under) so
    a runtime toggle updates the live global synthesise() reads. None when the
    skill is imported stand-alone (tests) — every caller tolerates that."""
    return sys.modules.get("__main__") or sys.modules.get("bobert_companion")


def _voice_clone():
    """Lazily resolve core.voice_clone. Kept defensive so a broken import can
    never take down the loader — the actions then report the engine is off."""
    try:
        from core import voice_clone  # type: ignore
        return voice_clone
    except Exception:
        return None


def _config():
    try:
        from core import config  # type: ignore
        return config
    except Exception:
        return None


# ─── config read/write (runtime globals + best-effort persistence) ──────────

def _get_enabled() -> bool:
    cfg = _config()
    return bool(getattr(cfg, "VOICE_CLONE_ENABLED", False)) if cfg else False


def _get_profile() -> str:
    cfg = _config()
    return str(getattr(cfg, "VOICE_CLONE_PROFILE", "") or "") if cfg else ""


def _apply_runtime(enabled: Optional[bool] = None, profile: Optional[str] = None) -> None:
    """Write the two knobs onto BOTH the live monolith module (whose module
    globals synthesise() reads via ``globals()``) AND core.config (which
    core.voice_clone.is_available() reads) so a toggle takes effect immediately
    on every consumer. Best-effort: a missing module is simply skipped.
    """
    cfg = _config()
    bc = _bobert()
    for target in (cfg, bc):
        if target is None:
            continue
        try:
            if enabled is not None:
                setattr(target, "VOICE_CLONE_ENABLED", bool(enabled))
            if profile is not None:
                setattr(target, "VOICE_CLONE_PROFILE", str(profile))
        except Exception:
            continue


def _persist(enabled: Optional[bool] = None, profile: Optional[str] = None) -> None:
    """Best-effort write-through to data/user_settings.json so the selection
    survives a restart (mirrors how the Settings GUI persists these knobs).
    Never raises — a read-only / missing data dir just means the choice is
    runtime-only, exactly like the existing TTS-backend voice toggle.
    """
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root, "data")
        path = os.path.join(data_dir, "user_settings.json")
        current: dict = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    current = loaded
            except Exception:
                current = {}
        if enabled is not None:
            current["VOICE_CLONE_ENABLED"] = bool(enabled)
        if profile is not None:
            current["VOICE_CLONE_PROFILE"] = str(profile)
        os.makedirs(data_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
    except Exception:
        # Persistence is a nicety, not a requirement — the runtime globals are
        # already set. Swallow so a locked/absent file never breaks the action.
        pass


# ─── action bodies ──────────────────────────────────────────────────────────

def _list_voice_profiles(_: str = "") -> str:
    vc = _voice_clone()
    if vc is None:
        return "The voice-clone engine isn't loaded, sir, so I can't list any profiles."
    try:
        profiles = vc.list_profiles()
    except Exception as e:
        return f"I couldn't read the voice profiles, sir: {e}"
    if not profiles:
        return ("No voice-clone profiles are enrolled yet, sir. "
                "Run tools/enroll_voice.py to add one.")
    parts: list[str] = []
    for meta in profiles:
        name = meta.get("name", "?")
        if vc.profile_is_usable(meta):
            source = meta.get("source", "unknown")
            parts.append(f"{name} ({source})")
        else:
            # Surface — but never arm — a profile missing its consent flag.
            parts.append(f"{name} (not consented — unusable)")
    active = _get_profile()
    active_tag = f" Active profile: {active}." if active else " No profile is active."
    return f"Enrolled voice profiles, sir: {', '.join(parts)}.{active_tag}"


def _set_voice_profile(name: str = "") -> str:
    vc = _voice_clone()
    n = (name or "").strip()
    if not n:
        return "Which voice profile should I use, sir? Say 'list voice profiles' to hear the options."
    if vc is None:
        return "The voice-clone engine isn't loaded, sir, so I can't switch profiles."
    try:
        meta = vc.load_profile(n)
    except Exception:
        meta = None
    if meta is None:
        return (f"I don't have a voice profile called '{n}', sir. "
                f"Say 'list voice profiles' to hear the enrolled ones.")
    if not vc.profile_is_usable(meta):
        # Consent gate refusal — the ethics chokepoint on the voice side.
        return (f"I won't use the '{n}' profile, sir — it isn't marked as "
                f"consented, so it's off-limits.")
    _apply_runtime(enabled=True, profile=n)
    _persist(enabled=True, profile=n)
    engine_ready = False
    try:
        engine_ready = vc.is_available()
    except Exception:
        engine_ready = False
    if engine_ready:
        return f"Switched to the '{n}' voice, sir."
    # Selected + enabled, but the engine can't render yet (no chatterbox / no
    # GPU) — be honest that I'll fall back until it's installed.
    return (f"Selected the '{n}' voice, sir, but the clone engine isn't ready "
            f"(needs chatterbox-tts and a CUDA GPU), so I'll use my normal "
            f"voice until it is.")


def _voice_clone_status(_: str = "") -> str:
    vc = _voice_clone()
    enabled = _get_enabled()
    profile = _get_profile()
    if not enabled:
        return "Voice cloning is off, sir — I'm using my normal voice."
    if not profile:
        return "Voice cloning is on, sir, but no profile is selected, so I'm on my normal voice."
    ready = False
    if vc is not None:
        try:
            ready = vc.is_available()
        except Exception:
            ready = False
    if ready:
        return f"Voice cloning is on, sir, speaking as the '{profile}' profile."
    return (f"Voice cloning is on with the '{profile}' profile, sir, but the "
            f"engine isn't available (needs chatterbox-tts and CUDA), so I'm "
            f"falling back to my normal voice.")


def _disable_voice_clone(_: str = "") -> str:
    _apply_runtime(enabled=False)
    _persist(enabled=False)
    return "Voice cloning off, sir — back to my normal voice."


# ─── registration ────────────────────────────────────────────────────────────

def register(actions: dict) -> None:
    actions["list_voice_profiles"] = _list_voice_profiles
    actions["set_voice_profile"] = _set_voice_profile
    actions["voice_clone_status"] = _voice_clone_status
    actions["disable_voice_clone"] = _disable_voice_clone
    # Natural-language aliases the LLM tends to emit — same handlers.
    actions["use_voice_profile"] = _set_voice_profile
    actions["switch_voice_profile"] = _set_voice_profile
    actions["stop_voice_clone"] = _disable_voice_clone
    actions["voice_clone_off"] = _disable_voice_clone
