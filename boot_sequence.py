#!/usr/bin/env python3
"""
JARVIS boot sequence — the "coming online" moment.

Spec (jarvis_todo.md 2026-05-27 08:19 boot_sequence): invoked from
bobert_companion.py on startup. Plays a 4–5 second arc-reactor power-up
animation on the HUD (concentric rings filling) while JARVIS speaks a
randomised MCU-style boot line, then quickly reports the inventory it
actually loaded (n actions, n skills, mic/speaker device, last session
timestamp).

The module is intentionally decoupled from bobert_companion's globals —
the caller passes in the speak function, the HUD-state writer, and the
inventory data, so this file has no imports from the main module and
can be unit-tested or reused from a different host process.

Animation contract with the HUD (hud/jarvis_hud.py):
    write_hud_state(boot_phase="powering",
                    boot_started_at=<epoch>,
                    boot_duration=<seconds>)
    ... speak ...
    write_hud_state(boot_phase="", boot_started_at=0.0)

The HUD also self-clears when (now - boot_started_at) > duration + 0.5s
so a crashed parent can't strand the overlay.
"""
import random
import time

# ── Boot lines (the "JARVIS coming online" moment from the films) ──────────
# Per spec: at least three variations so the same line doesn't fire every
# startup. The first three are the exact spec phrasings; the rest extend
# the bank so the same line doesn't repeat on consecutive boots.
BOOT_LINES = [
    "All systems nominal, sir. Welcome back.",
    "Online and at your service, sir.",
    "Good to have you back, sir — diagnostics complete, all clear.",
    "Systems initialised, sir. Standing by.",
    "Powering up. Welcome back, sir.",
    "Reactor at full output, sir. Ready when you are.",
]

# Animation length. The HUD reads this and self-clears when elapsed exceeds
# duration + 0.5s so a crashed parent can't strand the overlay.
BOOT_ANIMATION_SECONDS = 4.5

# After speech finishes, ensure the animation has been visible at least this
# long so the rings get a chance to fill all the way out even on a fast
# TTS backend.
MIN_VISIBLE_SECONDS = 4.0


def _humanise_ago(then_ts: float, now: float | None = None) -> str:
    """Render an epoch timestamp as a JARVIS-readable interval.
    Returns 'never' for a missing/zero timestamp so the caller can decide
    whether to skip the line entirely."""
    if not then_ts or then_ts <= 0:
        return "never"
    now = now if now is not None else time.time()
    delta = now - then_ts
    if delta < 90:
        return "moments ago"
    if delta < 3600:
        return f"{int(delta // 60)} minutes ago"
    if delta < 86400:
        hrs = int(round(delta / 3600))
        return "an hour ago" if hrs == 1 else f"{hrs} hours ago"
    if delta < 86400 * 2:
        return "yesterday"
    days = int(delta // 86400)
    return f"{days} days ago"


def build_inventory_line(n_actions: int, n_skills: int,
                         mic_name: str, speaker_name: str,
                         last_session_ts: float) -> str:
    """Compose the "what came online" line. Kept terse — at most three
    short clauses. Exposed so callers (and tests) can preview the line."""
    if n_actions and n_skills:
        head = f"{n_actions} actions and {n_skills} skills standing by, sir."
    elif n_actions:
        head = f"{n_actions} actions standing by, sir."
    elif n_skills:
        head = f"{n_skills} skills standing by, sir."
    else:
        head = "Inventory loaded, sir."

    # Device line — only if we actually have at least one device name.
    extras = []
    mic_short = (mic_name or "").strip()
    if mic_short:
        extras.append(f"Microphone on the {mic_short}")
    spk_short = (speaker_name or "").strip()
    if spk_short and spk_short.lower() != mic_short.lower():
        extras.append(f"speakers on the {spk_short}")
    device_line = "; ".join(extras) + "." if extras else ""

    when = _humanise_ago(last_session_ts)
    session_line = f"Last session was {when}." if when != "never" else ""

    pieces = [head]
    if device_line:
        pieces.append(device_line)
    if session_line:
        pieces.append(session_line)
    return " ".join(pieces)


def pick_boot_line(rng=None) -> str:
    """Pick one MCU-style boot line. Exposed so tests can pin a phrase."""
    chooser = rng or random
    return chooser.choice(BOOT_LINES)


def play_boot_sequence(speak_fn,
                       write_hud_state=None,
                       n_actions: int = 0,
                       n_skills: int = 0,
                       mic_name: str = "",
                       speaker_name: str = "",
                       last_session_ts: float = 0.0,
                       rng=None,
                       staging: bool = False) -> tuple[str, str]:
    """Run the boot sequence. Returns (boot_line, inventory_line) so the
    caller can append both to conversation_history for the LLM's context.

    Args:
        speak_fn: callable(str) → None. Synchronous (blocks until TTS done).
        write_hud_state: optional callable(**fields) → None. Used to publish
            boot_phase / boot_started_at / boot_duration to the HUD state
            file so the HUD can render the power-up animation. None disables
            the visual but the spoken sequence still fires.
        n_actions, n_skills: integer counts for the inventory line.
        mic_name, speaker_name: friendly device names ("USB Mic", etc.).
        last_session_ts: epoch seconds of the most recent prior session.
        rng: optional random.Random for deterministic phrase selection.
        staging: True when called from a blue/green staging instance — skips
            the HUD animation and the post-speech delay so the smoke test
            isn't held up by a cosmetic 4.5-second wait.
    """
    boot_line = pick_boot_line(rng)
    inventory_line = build_inventory_line(
        n_actions=n_actions,
        n_skills=n_skills,
        mic_name=mic_name,
        speaker_name=speaker_name,
        last_session_ts=last_session_ts,
    )

    # Start the HUD animation just before speaking so the rings power up
    # while JARVIS is talking. Failures are non-fatal — the spoken boot
    # still fires. Staging skips the HUD publish entirely since green has
    # no HUD subprocess and the smoke test doesn't need the visual.
    started_at = time.time()
    if write_hud_state is not None and not staging:
        try:
            write_hud_state(
                boot_phase="powering",
                boot_started_at=started_at,
                boot_duration=BOOT_ANIMATION_SECONDS,
                state="Initialising",
            )
        except Exception as e:
            print(f"  [boot_sequence] HUD start publish failed: {e}")

    try:
        speak_fn(boot_line)
    except Exception as e:
        print(f"  [boot_sequence] boot line speak failed: {e}")

    try:
        speak_fn(inventory_line)
    except Exception as e:
        print(f"  [boot_sequence] inventory speak failed: {e}")

    # Ensure the HUD's boot overlay stays up for at least MIN_VISIBLE_SECONDS
    # — on a fast TTS backend the two lines might finish in under 3 seconds
    # and snapping the animation off mid-fill looks jarring. Skip the wait
    # in staging so the smoke loop reaches its first inject promptly.
    if write_hud_state is not None and not staging:
        try:
            elapsed = time.time() - started_at
            if elapsed < MIN_VISIBLE_SECONDS:
                time.sleep(MIN_VISIBLE_SECONDS - elapsed)
            write_hud_state(
                boot_phase="",
                boot_started_at=0.0,
                state="Idle",
            )
        except Exception as e:
            print(f"  [boot_sequence] HUD clear failed: {e}")

    return boot_line, inventory_line
