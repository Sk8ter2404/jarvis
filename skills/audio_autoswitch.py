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


def _make_daemon():
    return audio_switch.AudioAutoSwitch(
        headset=_cfg_str("AUDIO_AUTOSWITCH_HEADSET"),
        fallback=_cfg_str("AUDIO_AUTOSWITCH_FALLBACK"),
        poll_s=_cfg_float("AUDIO_AUTOSWITCH_POLL_S", 3.0),
        announce=_announce,
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
