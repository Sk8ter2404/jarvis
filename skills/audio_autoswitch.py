"""Voice + daemon wiring for audio/audio_switch.py.

Auto-switches the Windows DEFAULT audio device when a wireless headset powers
on/off (the dongle stays plugged in, so plug/unplug detection misses it — see
audio/audio_switch.py). Opt-in via AUDIO_AUTOSWITCH_ENABLED + a headset name
fragment. Registers status / on / off / manual-switch voice actions and starts
the background watcher at boot when enabled.

EVERY power question here goes through `audio_switch.headset_powered()`, which
is THREE-VALUED (True / False / None). It does NOT use `find_active()` for
that: measured 2026-09-04, a CORSAIR VOID ELITE that is POWERED OFF still
reports both endpoints as Active, so the endpoint check answers "on" forever.
`find_active()` is still used to RESOLVE a device id once a decision is made —
that part of it was never wrong.

A None answer is spoken as "I can't tell", never as "off".
"""
import importlib
import os

from audio import audio_switch

# Every action below returns a finished, user-facing sentence, so each must be
# declared here or the answer is computed, logged and never voiced. The OUTPUT
# actions (use_headset / use_speakers / ...) are declared the other way, in
# bobert_companion.SPEAK_RESULT_VERBATIM_ACTIONS; that list predates this
# mechanism. Declaring the new names HERE keeps the monolith untouched.
SPEAK_VERBATIM_ACTIONS = (
    "use_headset_mic", "switch_to_headset_mic",
    "use_desk_mic", "switch_to_desk_mic",
    "which_mic_is_active",
)

# Registering an action is only HALF of shipping it. A name the model has never
# seen is not a name the model can emit: it emits the nearest wrong-but-
# plausible token instead and the owner hears a confident answer to a different
# question (proven 2026-09-04 - "what microphone are you using" routed to
# system_pulse). tests/test_audit_action_reachability.py fails the build on a
# name reachable by neither route, and this block is route 2.
#
# TODO(migrate): core/prompts.py is the long-term home for a TRACKED skill's
# examples, next to the existing "AUDIO OUTPUT DEVICE" section - these are the
# INPUT half of that same section. They live here for now because core/prompts.py
# is a shared file that other work is editing tonight; move this block under
# that section (and add the mic trigger words to the section's keyword list in
# core/prompt_router.py::_SECTION_KEYWORDS) once that settles.
PROMPT_EXAMPLES = (
    "AUDIO INPUT DEVICE (which MICROPHONE JARVIS listens on; follows the\n"
    "  wireless headset's power when the input half is enabled):\n"
    "  use_headset_mic / switch_to_headset_mic\n"
    "                       - listen on the HEADSET's microphone. Fire on\n"
    "                         'use my headset mic', 'listen on my headset',\n"
    "                         'switch the mic to my headset', 'hear me through\n"
    "                         the headset'.\n"
    "  use_desk_mic / switch_to_desk_mic\n"
    "                       - listen on the DESK microphone instead. Fire on\n"
    "                         'use my desk mic', 'switch to the desk mic',\n"
    "                         'stop using the headset mic', 'use the Snowball'.\n"
    "  which_mic_is_active  - report WHICH recording device is selected and\n"
    "                         whether anything is overriding the choice. Fire\n"
    "                         on 'which mic is active', 'which microphone are\n"
    "                         you set to', 'are you on the headset mic'.\n"
    "    Example: 'use my desk mic' -> [ACTION: use_desk_mic]\n"
    "    Example: 'listen on my headset' -> [ACTION: use_headset_mic]\n"
    "    Example: 'which mic is active' -> [ACTION: which_mic_is_active]\n"
    "  These change or report the INPUT device only. They are not volume, and\n"
    "  they are not use_headset / use_speakers, which move the OUTPUT."
)

_DAEMON = [None]          # the live AudioAutoSwitch (module-level so on/off reach it)


def _cfg(name, default=None):
    try:
        bc = importlib.import_module("bobert_companion")
        return getattr(bc, name, default)
    except Exception:
        return default


def _cfg_str(name) -> str:
    v = _cfg(name, "")
    return v.strip() if isinstance(v, str) else ""


def _cfg_float(name, default: float) -> float:
    try:
        return float(_cfg(name, default))
    except Exception:
        return default


def _announce(message: str) -> None:
    """Route a spoken alert through the canonical pending_speech writer."""
    try:
        bc = importlib.import_module("bobert_companion")
        ann = getattr(bc, "proactive_announce", None)
        if callable(ann) and ann(message, source="audio"):
            return
    except Exception as e:
        print(f"  [audio-switch] speech-queue write failed ({e}); msg: {message}")
        return
    print(f"  [audio-switch] {message}")


def _capture_sentence(name: str) -> str:
    """"Listening on X now, sir." - but ONLY when that is actually true.

    THE DEFECT THIS CLOSES (2026-09-05). The daemon's two spoken transitions
    were gated by audio_switch.capture_claim() the same night; these two VOICE
    actions were not, and they make the identical claim on the identical
    evidence. set_default_capture() moves the WINDOWS default recording
    endpoint, and on this machine that is NOT what bobert_companion records
    from: PREFERRED_INPUT_DEVICES is live in data/user_settings.json
    (['Blue Snowball', 'eMeet C960', 'CORSAIR VOID']), _pick_device matches the
    Snowball on every pass, and the one branch that reads the default endpoint
    therefore never runs. Replayed read-only against the live PortAudio list
    2026-09-05: _pick_device(..., want_input=True) -> (6, 'Microphone (Blue
    Snowball )'). So "Listening on the headset, sir" was a spoken claim about a
    state nobody had checked - said in direct answer to an order, which is the
    worst possible place to say it.

    The DECISION is not re-implemented here. It comes from the same
    audio_switch.capture_override() the daemon's claim uses, because two copies
    of one rule is exactly how one copy ends up rotting.

    What the hedged sentence deliberately does NOT say is "I am not recording
    from it". That would be a fresh unverified claim in the other direction:
    when the preferred list happens to name the very device just selected (his
    desk mic IS the Blue Snowball, the first entry), he really is being heard
    on it. All capture_override() establishes is WHICH RULE DECIDES, so that is
    the only thing asserted. Proving a microphone is actually producing audio
    needs signal - the VAD's peak RMS, where a dead device reads exactly 0.0000
    - and signal needs the input stream, which only the monolith may open.
    """
    why = audio_switch.capture_override()
    if why is None:
        return f"Listening on {name} now, sir."
    print(f"  [audio-switch] the Windows default recording device is now "
          f"{name}, but that is not what selects JARVIS's microphone: {why}. "
          f"Clear PREFERRED_INPUT_DEVICES (and leave MICROPHONE_INDEX unset) "
          f"in data/user_settings.json, then restart, so the input direction "
          f"follows the Windows default again.")
    return (f"Windows' microphone is set to {name} now, sir - but that isn't "
            f"what decides which microphone I record from: {why}. Ask me which "
            f"mic is active for the full picture.")


def _make_daemon():
    return audio_switch.AudioAutoSwitch(
        headset=_cfg_str("AUDIO_AUTOSWITCH_HEADSET"),
        fallback=_cfg_str("AUDIO_AUTOSWITCH_FALLBACK"),
        poll_s=_cfg_float("AUDIO_AUTOSWITCH_POLL_S", 3.0),
        announce=_announce,
        mic_fallback=_cfg_str("AUDIO_AUTOSWITCH_MIC_FALLBACK"),
        follow_mic=bool(_cfg("AUDIO_AUTOSWITCH_MIC", False)),
        # The steady-state ON signal watchdog. Read through _cfg_float so a
        # Settings-GUI edit reaches it the same way every other knob here does;
        # the default matches core/config.py so a missing key still protects
        # him rather than silently restoring fire-once-and-never-look-again.
        mic_silent_s=_cfg_float("AUDIO_AUTOSWITCH_MIC_SILENT_S", 60.0),
    )


def _start_daemon() -> bool:
    if _DAEMON[0] is None:
        _DAEMON[0] = _make_daemon()
    return _DAEMON[0].start()


def register(actions):
    def audio_autoswitch_status(_: str = "") -> str:
        if _DAEMON[0] is not None:
            return _DAEMON[0].status()
        headset = _cfg_str("AUDIO_AUTOSWITCH_HEADSET")
        if not headset:
            return ("Audio auto-switch isn't set up, sir — give me the headset's "
                    "device name (AUDIO_AUTOSWITCH_HEADSET) and turn it on.")
        powered = audio_switch.headset_powered(headset)
        if powered is None:
            return (f"Audio auto-switch is off, sir — and I can't tell whether the "
                    f"'{headset}' headset is powered right now.")
        return (f"Audio auto-switch is off, sir. The '{headset}' headset is "
                f"{'on' if powered else 'off'} right now.")

    def audio_autoswitch_on(_: str = "") -> str:
        if not _cfg_str("AUDIO_AUTOSWITCH_HEADSET"):
            return "Tell me the headset's device name first, sir."
        _start_daemon()
        return "Audio auto-switch is on, sir — I'll follow the headset's power."

    def audio_autoswitch_off(_: str = "") -> str:
        if _DAEMON[0] is not None:
            _DAEMON[0].stop()
        return "Audio auto-switch is off, sir."

    def use_headset(_: str = "") -> str:
        # An explicit "switch to the headset" is obeyed unless the headset is
        # MEASURED off — sending audio to a powered-down headset is silence.
        # An UNKNOWN power state still switches (the owner asked), but says so.
        headset = _cfg_str("AUDIO_AUTOSWITCH_HEADSET")
        if not headset:
            return "I don't have a headset device configured, sir."
        powered = audio_switch.headset_powered(headset)
        if powered is False:
            return "The headset is powered off, sir — I'd be sending audio nowhere."
        hs = audio_switch.find_active(headset)
        if not hs:
            return (f"I can't find a playback device matching '{headset}', sir — "
                    f"the name may be wrong.")
        if not audio_switch.set_default_render(hs[0]):
            return "I couldn't switch to the headset, sir."
        if powered is None:
            return ("Audio's on the headset now, sir — though I couldn't confirm "
                    "it's actually powered on.")
        return "Audio's on the headset now, sir."

    def use_speakers(_: str = "") -> str:
        # resolve_fallback() logs exactly why an unusable fallback failed; this
        # says it out loud too, rather than the old silent "not configured".
        fb = _cfg_str("AUDIO_AUTOSWITCH_FALLBACK")
        if not fb:
            return "I don't have a speakers device configured, sir."
        spk = audio_switch.resolve_fallback(fb)
        if not spk:
            problem = audio_switch.fallback_problem(fb) or "it could not be resolved"
            return (f"I can't switch to the speakers, sir: {problem}.")
        return ("Audio's on the speakers now, sir." if audio_switch.set_default_render(spk[0])
                else "I couldn't switch to the speakers, sir.")

    # ── the INPUT half ──────────────────────────────────────────────────
    # These exist so the owner is never STUCK on the wrong microphone waiting
    # for a poll, a restart, or a settings edit. Deaf-safety rule for both
    # switchers: resolve and verify the target FIRST, and if it cannot be
    # verified say why and change nothing - he keeps the microphone that is
    # working.
    def use_headset_mic(_: str = "") -> str:
        headset = _cfg_str("AUDIO_AUTOSWITCH_HEADSET")
        if not headset:
            return "I don't have a headset configured, sir."
        powered = audio_switch.headset_powered(headset)
        if powered is False:
            # MEASURED off. Its microphone is a hole in the air - this is
            # exactly the state that made me deaf at ten to one this morning.
            return ("The headset is powered off, sir - its microphone would "
                    "hear nothing at all, so I've left the input where it is.")
        mic = audio_switch.find_active_capture(headset)
        if not mic:
            return (f"I can't find a recording device matching '{headset}', sir "
                    f"- the name may be wrong.")
        if not audio_switch.set_default_capture(mic[0]):
            # Hedged since 2026-09-05: set_default_capture reads the default
            # back, so False can also mean "the write was accepted but I could
            # not prove the device moved". Naming the old device as certain
            # would be a fresh unverified claim.
            return ("I couldn't confirm the microphone moved to the headset, "
                    "sir - check `which mic is active` before you rely on it.")
        # Gated, not asserted - see _capture_sentence. Moving the Windows
        # default is what actually happened; being heard on it is a separate
        # fact with a separate owner (bobert_companion's _pick_device).
        said = _capture_sentence(mic[1])
        if powered is None:
            # Obeyed rather than refused on purpose: UNKNOWN is what the dongle
            # reports for the ~105 seconds it spends pairing, which is exactly
            # when he'd ask. And it is self-healing - if the headset really is
            # off, the watcher's next measured-off sample moves him straight
            # back off it.
            return (f"{said} Though I couldn't confirm the headset is actually "
                    f"powered on - if it isn't, I'll move back to your desk mic "
                    f"on my own.")
        return said

    def use_desk_mic(_: str = "") -> str:
        fb = _cfg_str("AUDIO_AUTOSWITCH_MIC_FALLBACK")
        if not fb:
            return ("I don't have a desk microphone configured, sir - set "
                    "AUDIO_AUTOSWITCH_MIC_FALLBACK to its name.")
        mic = audio_switch.resolve_mic_fallback(fb)     # logs exactly why on failure
        if not mic:
            problem = audio_switch.mic_fallback_problem(fb) or "it could not be resolved"
            return f"I can't switch to the desk microphone, sir: {problem}."
        if not audio_switch.set_default_capture(mic[0]):
            return ("I couldn't confirm the microphone moved to the desk mic, "
                    "sir - check `which mic is active` before you rely on it.")
        return _capture_sentence(mic[1])

    def which_mic_is_active(_: str = "") -> str:
        """Which recording device is selected - and what is overriding it.

        Deliberately reports the SELECTION and never claims the device is
        producing audio: "selected" is not "working", and asserting the latter
        from the former is the whole bug class this feature came out of. It
        also names the override, because with PREFERRED_INPUT_DEVICES set the
        Windows default is not what JARVIS is listening on, and an answer that
        omitted that would be true and useless."""
        rows = audio_switch.list_render()
        cur = audio_switch.default_capture_id()
        name = audio_switch.endpoint_name(cur, rows=rows) if cur else None
        where = f"Windows' default microphone is {name}" if name \
            else "I couldn't read Windows' default microphone"

        idx = _cfg("MICROPHONE_INDEX", None)
        if isinstance(idx, int) and not isinstance(idx, bool):
            if idx < 0:
                return ("My microphone is switched off entirely, sir - "
                        f"MICROPHONE_INDEX is {idx}, the hard-off setting. "
                        f"({where}.)")
            return (f"I'm pinned to microphone index {idx}, sir, so I'm not "
                    f"following the system default at all. {where}.")

        pref = _cfg("PREFERRED_INPUT_DEVICES", None) or []
        if pref:
            listed = ", ".join(str(x) for x in pref)
            return (f"{where}, sir - but I'm not following it. I'm taking the "
                    f"first connected match from my preferred list ({listed}), "
                    f"which means the headset microphone can't be chosen "
                    f"automatically even when the headset is on.")

        headset = _cfg_str("AUDIO_AUTOSWITCH_HEADSET")
        follow = bool(_cfg("AUDIO_AUTOSWITCH_MIC", False))
        tail = ""
        if headset:
            powered = audio_switch.headset_powered(headset)
            word = {True: "is on", False: "is off",
                    None: "I can't tell whether it's on"}[powered]
            tail = (f" The headset {word}, and I "
                    f"{'am' if follow else 'am not'} following its power for "
                    f"the microphone.")
        return f"{where}, sir.{tail}"

    actions["use_headset_mic"] = use_headset_mic
    actions["switch_to_headset_mic"] = use_headset_mic
    actions["use_desk_mic"] = use_desk_mic
    actions["switch_to_desk_mic"] = use_desk_mic
    actions["which_mic_is_active"] = which_mic_is_active

    actions["audio_autoswitch_status"] = audio_autoswitch_status
    actions["audio_autoswitch_on"] = audio_autoswitch_on
    actions["audio_autoswitch_off"] = audio_autoswitch_off
    actions["use_headset"] = use_headset
    actions["switch_to_headset"] = use_headset
    actions["use_speakers"] = use_speakers
    actions["switch_to_speakers"] = use_speakers

    # Start the watcher at boot when enabled. Gated hard on a REAL headset name
    # (a mocked config in unit tests yields ""), and never under staging — it
    # mutates the real default audio device and spawns a thread.
    headset = _cfg_str("AUDIO_AUTOSWITCH_HEADSET")
    if bool(_cfg("AUDIO_AUTOSWITCH_ENABLED", False)) and headset and not os.getenv("JARVIS_STAGING"):
        try:
            if _start_daemon():
                print(f"  [audio-switch] watching '{headset}' — default audio follows its power")
        except Exception as e:
            print(f"  [audio-switch] daemon start failed: {e}")
