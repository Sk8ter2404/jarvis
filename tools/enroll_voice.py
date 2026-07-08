#!/usr/bin/env python3
"""
Out-of-band voice-clone ENROLLMENT for JARVIS (Chatterbox backend).

Writes a voice-clone profile the owner can then select with 'switch to my
voice'. A profile is a directory under the GITIGNORED
``data/voice_profiles/<name>/`` holding:

    reference.wav   — the consented reference clip (~5-10 s of clean speech)
    meta.json       — {name, created_at, consent: true, source, model}

This script is the ONLY writer of a profile. It is deliberately run LIVE by the
owner, never by CI: recording / copying real audio and (optionally) probing the
GPU are side-effecting and machine-specific. Heavy libs (soundfile,
sounddevice) are imported LAZILY inside the functions that need them, so the
module itself imports on a bare box — the CI check is import-only.

═══════════════════════════════════════════════════════════════════════════
ETHICS / CONSENT  (enforced, not advisory)
═══════════════════════════════════════════════════════════════════════════
Only two kinds of profile may be enrolled:

  --source owner       the OWNER's OWN voice, from his OWN recording.
  --source character   a JARVIS in-character British-butler voice, from a clip
                       the owner provides OR a bundled non-celebrity voice.

Enrollment REQUIRES the explicit ``--consent`` flag. Without it the script
refuses to write anything — there is no path here that clones a named real
person / celebrity / "the real voice actor". The written meta.json always
carries ``consent: true`` (the file only exists because consent was given at
enroll time); ``core.voice_clone.profile_is_usable`` refuses any profile whose
meta lacks that flag.

Usage
-----
    # enroll from an existing wav (owner's own consented recording):
    python tools/enroll_voice.py --name me --source owner \\
        --consent --from-wav C:/path/to/my_reference.wav

    # enroll a JARVIS character voice from a provided clip:
    python tools/enroll_voice.py --name jarvis --source character \\
        --consent --from-wav C:/path/to/butler_clip.wav

    # record ~8 s live from the default mic instead of copying a file:
    python tools/enroll_voice.py --name me --source owner --consent --record

    # list / remove:
    python tools/enroll_voice.py --list
    python tools/enroll_voice.py --remove me
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIR = os.path.join(_PROJECT_ROOT, "data", "voice_profiles")

REFERENCE_WAV_NAME = "reference.wav"
META_NAME = "meta.json"
_ALLOWED_SOURCES = ("owner", "character")
DEFAULT_RECORD_SECONDS = 8.0
RECORD_SAMPLE_RATE = 24000   # 24 kHz mono matches the clone engine's native rate


def _profile_dir(name: str) -> str:
    return os.path.join(PROFILES_DIR, name)


def _valid_name(name: str) -> bool:
    """A profile name must be a safe single path segment (no traversal, no
    separators) so it can only ever land inside PROFILES_DIR."""
    if not name or not isinstance(name, str):
        return False
    if name.startswith(".") or name.startswith("_"):
        return False
    return all(c.isalnum() or c in "-_" for c in name)


def _write_meta(name: str, source: str, model: str) -> str:
    """Write meta.json with consent:true. Returns the path. Called only after
    the consent flag + a valid reference.wav are confirmed."""
    meta = {
        "name": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "consent": True,          # only ever written when --consent was given
        "source": source,         # "owner" | "character"
        "model": model,
    }
    path = os.path.join(_profile_dir(name), META_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def _copy_reference_wav(name: str, src_wav: str) -> str:
    """Copy the owner-provided wav into the profile dir as reference.wav.
    Validates it decodes as audio (lazy soundfile import) so a bad file is
    caught at enroll time, not on the first synth."""
    if not os.path.isfile(src_wav):
        raise FileNotFoundError(f"reference wav not found: {src_wav}")
    # Lazy import — this is the heavy dep the CI import-guard must not require.
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(src_wav)
        if info.frames <= 0:
            raise ValueError("reference wav is empty")
    except ImportError:
        # soundfile absent (won't happen on the owner's box): skip validation
        # and copy anyway rather than block enrollment.
        print("  [enroll] soundfile not installed; skipping wav validation")
    dst = os.path.join(_profile_dir(name), REFERENCE_WAV_NAME)
    shutil.copyfile(src_wav, dst)
    return dst


def _record_reference_wav(name: str, seconds: float) -> str:
    """Record `seconds` of mono audio from the default mic to reference.wav.
    Heavy (sounddevice + soundfile) and side-effecting → lazy imports, live-run
    only. Never reached by CI."""
    import sounddevice as sd     # type: ignore  (lazy: mic capture)
    import soundfile as sf       # type: ignore  (lazy: wav write)
    print(f"  [enroll] recording {seconds:.0f}s from the default mic — speak now…")
    frames = int(seconds * RECORD_SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=RECORD_SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    dst = os.path.join(_profile_dir(name), REFERENCE_WAV_NAME)
    sf.write(dst, audio, RECORD_SAMPLE_RATE)
    print(f"  [enroll] saved {dst}")
    return dst


def enroll(
    name: str,
    source: str,
    consent: bool,
    from_wav: Optional[str] = None,
    record: bool = False,
    record_seconds: float = DEFAULT_RECORD_SECONDS,
    model: str = "chatterbox",
) -> str:
    """Create a voice-clone profile. Returns the profile dir path.

    Raises ValueError on any policy violation (no consent, bad source, bad
    name, no audio source) BEFORE writing anything — so a refused enrollment
    leaves no half-written profile on disk.
    """
    if not consent:
        raise ValueError(
            "enrollment requires explicit --consent (the reference speaker must "
            "have consented to being cloned). Refusing to write a profile.")
    if source not in _ALLOWED_SOURCES:
        raise ValueError(
            f"--source must be one of {_ALLOWED_SOURCES} "
            f"(owner = your own voice; character = a JARVIS/non-celebrity voice). "
            f"Cloning a named real person is out of scope.")
    if not _valid_name(name):
        raise ValueError(
            f"invalid profile name {name!r} — use letters, digits, '-' or '_'.")
    if not from_wav and not record:
        raise ValueError("provide either --from-wav <path> or --record.")

    _dir = _profile_dir(name)
    _preexisting = os.path.isdir(_dir)
    os.makedirs(_dir, exist_ok=True)
    try:
        if from_wav:
            _copy_reference_wav(name, from_wav)
        else:
            _record_reference_wav(name, record_seconds)
        _write_meta(name, source, model)
    except Exception:
        # A failed reference-wav copy/record (missing / empty / undecodable wav,
        # mic error, …) must NOT leave a half-written profile dir behind — an
        # empty dir with no meta.json/reference.wav would still show up in --list
        # as an enrolled profile, contradicting the docstring's promise. Remove
        # the dir ONLY if we just created it (never wipe a pre-existing profile on
        # a failed RE-enroll), then re-raise so the caller still reports the error.
        # 2026-07-08.
        if not _preexisting:
            shutil.rmtree(_dir, ignore_errors=True)
        raise
    return _dir


def list_profiles() -> list[str]:
    try:
        return sorted(
            e for e in os.listdir(PROFILES_DIR)
            if os.path.isdir(_profile_dir(e)) and not e.startswith((".", "_"))
        )
    except Exception:
        return []


def remove_profile(name: str) -> bool:
    d = _profile_dir(name)
    if not os.path.isdir(d):
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Enroll a JARVIS voice-clone profile (consent required).")
    p.add_argument("--name", help="profile name (letters/digits/-/_).")
    p.add_argument("--source", choices=_ALLOWED_SOURCES,
                   help="owner = your own voice; character = JARVIS/non-celebrity.")
    p.add_argument("--consent", action="store_true",
                   help="REQUIRED: affirm the speaker consented to cloning.")
    p.add_argument("--from-wav", dest="from_wav",
                   help="path to an existing reference wav to copy in.")
    p.add_argument("--record", action="store_true",
                   help="record a fresh reference clip from the default mic.")
    p.add_argument("--seconds", type=float, default=DEFAULT_RECORD_SECONDS,
                   help=f"record length in seconds (default {DEFAULT_RECORD_SECONDS}).")
    p.add_argument("--model", default="chatterbox", help="engine id.")
    p.add_argument("--list", action="store_true", help="list enrolled profiles.")
    p.add_argument("--remove", help="delete the named profile.")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.list:
        names = list_profiles()
        print("Enrolled voice profiles:" if names else "No voice profiles enrolled.")
        for n in names:
            print(f"  - {n}")
        return 0
    if args.remove:
        ok = remove_profile(args.remove)
        print(f"Removed profile {args.remove!r}." if ok
              else f"No profile named {args.remove!r}.")
        return 0 if ok else 1
    if not args.name or not args.source:
        print("error: --name and --source are required to enroll "
              "(or use --list / --remove).", file=sys.stderr)
        return 2
    try:
        path = enroll(
            name=args.name, source=args.source, consent=args.consent,
            from_wav=args.from_wav, record=args.record,
            record_seconds=args.seconds, model=args.model,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Enrolled voice profile at {path}")
    print("Enable it: set VOICE_CLONE_ENABLED=true and "
          f"VOICE_CLONE_PROFILE={args.name!r} (Settings GUI or user_settings.json), "
          f"or say 'switch to the {args.name} voice'.")
    return 0


if __name__ == "__main__":   # pragma: no cover - CLI entry, not CI-tested
    raise SystemExit(main())
