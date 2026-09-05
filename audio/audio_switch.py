"""Auto-switch the Windows DEFAULT audio devices when a wireless headset powers
on/off — without plugging/unplugging the dongle.

    PLAYBACK (always, whenever the watcher runs):
      headset ON  -> default = the headset   (remember the prior default)
      headset OFF -> default = the prior default, else a configured fallback

    RECORDING (only when follow_mic / AUDIO_AUTOSWITCH_MIC is on):
      headset ON  -> default mic = the headset's own microphone
      headset OFF -> default mic = the remembered mic, else
                     AUDIO_AUTOSWITCH_MIC_FALLBACK, else ANY other ACTIVE
                     recording endpoint — but ONLY when the current default mic
                     IS the powered-off headset, and ONLY to a target verified
                     ACTIVE on that same pass.
      headset ON and STAYING on, but its microphone has produced literal
                     digital silence for AUDIO_AUTOSWITCH_MIC_SILENT_S while
                     the monolith's capture loop was demonstrably running
                     -> the same rescue ladder as the OFF row.

    "SELECTED" IS NOT "PRODUCING AUDIO", so the ON side checks. It used to fire
    once on the power-up transition and never look again, which left the whole
    steady-state ON case unguarded: a headset that is powered ON but hearing
    nothing — boom mic flipped up (its own hardware mute), endpoint muted in
    Windows, owner out of range — kept the capture default for as long as it
    stayed switched on, and the OFF rescue could not help because the headset
    MEASURES powered on. The watchdog does not open a stream to find that out;
    it reads the peak RMS the monolith's capture loop already measures and
    publishes (core.audio_processor). See capture_signal_state() and
    AudioAutoSwitch._mic_silence_watchdog.

    THE LAST RUNG IS NOT OPTIONAL. "Nothing configured resolved, so hold" is
    the right answer everywhere in this module EXCEPT here, because here the
    device being held is one this code has just measured powered OFF. Holding
    it is not caution, it is the outage. So when neither the remembered mic nor
    the configured fallback resolves, the rescue moves to any other ACTIVE
    recording endpoint rather than staying — and says, in the log and out loud,
    that the new device is NOT verified to be producing audio. An unproven
    microphone beats a proven-dead one. Only a machine with no other Active
    recording endpoint at all leaves him where he is, and that says so too.

WHY THE RECORDING HALF EXISTS — measured 2026-09-05 00:48-00:52 on this
machine. The owner powered the headset off. The playback half worked: the
output moved to his speakers. The INPUT did not move at all, because nothing in
this repo had ever written the default CAPTURE endpoint — so Windows went on
pointing the microphone at a headset that was switched off, and JARVIS logged
peak RMS = 0.0000 twice, thirty seconds apart. Exactly zero, i.e. completely
deaf. The state machine and its deaf-safety rules are in AudioAutoSwitch, under
"THE INPUT HALF".

HOW THE POWER STATE IS DETECTED  (corrected 2026-09-04)
======================================================
This module used to assume that Windows flips the headset's audio ENDPOINT
between ``Active`` (on) and ``NotPresent``/``Unplugged`` (off). MEASURED
2026-09-04 on this machine, that assumption is FALSE for the CORSAIR VOID
ELITE Wireless: with the headset POWERED OFF, Windows still reported

    OK / Active   Headset Earphone   (CORSAIR VOID ELITE Wireless Gaming Headset)
    OK / Active   Headset Microphone (CORSAIR VOID ELITE Wireless Gaming Headset)

because what Windows sees is the DONGLE, and the dongle is plugged in whether
or not the headset is powered. The endpoint state therefore carries NO
information about that headset's power, `find_active()` concluded "headset on"
forever, and the auto-switch could never fire. Nobody re-checked the sentence
that said otherwise for months — which is why the corrected wording now names
the measurement and its date.

So the power question goes through `headset_powered()` instead:

  * a configured name that looks like a Corsair VOID -> audio.void_link, which
    asks the dongle's Corsair vendor HID interface directly (measured protocol,
    three-valued answer).
  * anything else -> the endpoint check, which is all that exists for other
    hardware. NOTE it has not been DISPROVEN for other headsets, but neither
    has it been verified for any specific one.

`headset_powered()` is THREE-VALUED: True / False / None, where None means
"genuinely unknown". Unknown HOLDS the current state and is never treated as
off: the VOID's dongle goes completely silent for ~105 seconds while the
headset pairs (measured 2026-09-04, 20:19:41 -> 20:21:26), and mapping that
silence to "off" would fire a spurious device switch every single time the
owner powers his headset on.

Setting the default is done via the ``IPolicyConfigVista`` COM interface
(comtypes) — no external .exe (Smart App Control blocks those), no new pip deps
(comtypes + pycaw already ship with JARVIS).

Standalone:
    python -m audio.audio_switch --list       # render devices + states
    python -m audio.audio_switch --list-mics  # RECORDING devices + states
    python -m audio.audio_switch --state      # what the detector actually says
    python -m audio.audio_switch --test       # switch to headset + restore (proves it)
    python -m audio.audio_switch --test-mic "Blue Snowball"
                                              # switch the default MIC + restore.
                                              # The only thing here that proves a
                                              # capture default really MOVES; run
                                              # it deliberately, on a desk mic.
    python -m audio.audio_switch --daemon [--follow-mic --mic-fallback "Blue Snowball"]
"""
from __future__ import annotations

import re
import sys
import threading
import time
import warnings as _warnings

# pycaw's GetAllDevices()/GetAllSessions() raise a COMError for each audio
# endpoint property (PKEY 62-69) they can't read on devices that don't expose
# them; pycaw surfaces it as a UserWarning. Non-fatal (the device list is still
# returned), but it floods the headset-state poll loop. Silence just that
# COMError-property warning so the poller stays quiet — this also covers running
# `python -m audio.audio_switch` standalone.
_warnings.filterwarnings("ignore", message=r".*COMError attempting to get property.*")

# ── IPolicyConfigVista — the proven default-endpoint setter ──────────────────
try:
    import comtypes
    from comtypes import GUID, COMMETHOD, HRESULT, IUnknown, CoCreateInstance, CLSCTX_ALL
    from ctypes import POINTER, c_int, c_longlong
    from ctypes.wintypes import LPCWSTR, DWORD
    _HAS_COM = True
except Exception:  # pragma: no cover - non-Windows / no comtypes
    _HAS_COM = False

if _HAS_COM:
    class _IPolicyConfigVista(IUnknown):
        # Only SetDefaultEndpoint is called; the earlier methods just occupy
        # their vtable slots in order (the exact format-pointer types are
        # irrelevant because we never call them). SetDefaultEndpoint is slot 9.
        _iid_ = GUID("{568b9108-44bf-40b4-9006-86afe5b5a620}")
        _methods_ = (
            COMMETHOD([], HRESULT, "GetMixFormat",
                      (["in"], LPCWSTR, "n"), (["out"], POINTER(POINTER(c_int)), "f")),
            COMMETHOD([], HRESULT, "GetDeviceFormat",
                      (["in"], LPCWSTR, "n"), (["in"], c_int, "d"),
                      (["out"], POINTER(POINTER(c_int)), "f")),
            COMMETHOD([], HRESULT, "SetDeviceFormat",
                      (["in"], LPCWSTR, "n"), (["in"], POINTER(c_int), "a"),
                      (["in"], POINTER(c_int), "b")),
            COMMETHOD([], HRESULT, "GetProcessingPeriod",
                      (["in"], LPCWSTR, "n"), (["in"], c_int, "d"),
                      (["out"], POINTER(c_longlong), "a"), (["out"], POINTER(c_longlong), "b")),
            COMMETHOD([], HRESULT, "SetProcessingPeriod",
                      (["in"], LPCWSTR, "n"), (["in"], POINTER(c_longlong), "a")),
            COMMETHOD([], HRESULT, "GetShareMode",
                      (["in"], LPCWSTR, "n"), (["out"], POINTER(c_int), "a")),
            COMMETHOD([], HRESULT, "SetShareMode",
                      (["in"], LPCWSTR, "n"), (["in"], POINTER(c_int), "a")),
            COMMETHOD([], HRESULT, "GetPropertyValue",
                      (["in"], LPCWSTR, "n"), (["in"], c_int, "s"),
                      (["in"], POINTER(c_int), "k"), (["out"], POINTER(c_int), "v")),
            COMMETHOD([], HRESULT, "SetPropertyValue",
                      (["in"], LPCWSTR, "n"), (["in"], c_int, "s"),
                      (["in"], POINTER(c_int), "k"), (["in"], POINTER(c_int), "v")),
            COMMETHOD([], HRESULT, "SetDefaultEndpoint",
                      (["in"], LPCWSTR, "n"), (["in"], DWORD, "role")),
            COMMETHOD([], HRESULT, "SetEndpointVisibility",
                      (["in"], LPCWSTR, "n"), (["in"], c_int, "v")),
        )

    _CLSID_PolicyConfigVistaClient = GUID("{294935CE-F637-4E7C-A41B-AB255460B862}")


def _log(msg: str) -> None:
    """Every diagnostic in this module goes through here.

    Silence is the defect this module was built on — a fallback that matches
    nothing, an enumeration that fails, a switch that doesn't take, all used to
    return None and say nothing. Anything that fails must SAY SO, on one
    greppable ``[audio-switch]`` line."""
    print(f"  [audio-switch] {msg}", flush=True)


# MMDevice id prefixes. These are the ONLY structural difference between a
# playback and a recording endpoint id, and every direction decision in this
# module is made on them rather than on a device NAME — the CORSAIR's earphone
# and its microphone share one friendly name, so a name can never tell them
# apart. Measured on this machine 2026-09-05: render default
# "{0.0.0.00000000}.{b068b686-...}", capture default "{0.0.1.00000000}.{da8ac56b-...}".
RENDER_PREFIX = "{0.0.0."
CAPTURE_PREFIX = "{0.0.1."


def set_default_endpoint(device_id: str) -> bool:
    """Make `device_id` (an MMDevice id string) the default endpoint for all
    three roles. Returns True on success.

    DIRECTION-NEUTRAL. Windows takes the direction from the ID ITSELF; there is
    no separate capture method on IPolicyConfig, and nothing in this body was
    ever render-specific except the old function's NAME.

    WHAT WAS ACTUALLY MEASURED, and what was not — 2026-09-05 00:5x, live:
      PROVEN: called with the CAPTURE id that was ALREADY the default for all
        three capture roles ({0.0.1.00000000}.{da8ac56b-...} — the CORSAIR
        headset microphone), it returned S_OK and afterwards all three capture
        roles and the render default read exactly as before. Writing a value
        that equals the value already there cannot move a device, so this is a
        proven no-op — and it proves the one thing it was run to prove: the
        interface ACCEPTS a {0.0.1. id instead of rejecting it as E_INVALIDARG.
      NOT PROVEN: that a capture default actually MOVES to a DIFFERENT device.
        That needs a real switch on a live box, and the owner was mid-session
        with JARVIS listening, so it was deliberately not done. Prove it with
        `python -m audio.audio_switch --test-mic "Blue Snowball"` (switches,
        reads back, restores, and prints every step) before trusting the
        microphone follow. Until then treat the input half as UNVERIFIED —
        `AudioAutoSwitch` is written so that an unproven or failing capture
        write leaves the previous device in place rather than half-switching.

    THE RETURN VALUE OF THIS FUNCTION IS AN HRESULT, NOT A DEVICE STATE.
    True here means "the COM call did not raise", and on the capture direction
    that is exactly the thing never proven to imply a move. Nothing may treat
    it as "the microphone is now X": `set_default_capture` reads the default
    back and only IT gets to say a capture default moved. Added 2026-09-05
    because every deaf-safety rule in `AudioAutoSwitch` reasoned about the
    WRITE and none about the RESULT, so an accepted-but-ineffective write would
    have been announced as a completed rescue.
    """
    if not _HAS_COM or not device_id:
        return False
    try:
        pc = CoCreateInstance(_CLSID_PolicyConfigVistaClient, _IPolicyConfigVista, CLSCTX_ALL)
        for role in (0, 1, 2):          # console / multimedia / communications
            pc.SetDefaultEndpoint(device_id, role)
        return True
    except Exception as e:
        _log(f"set_default_endpoint failed: {e}")
        return False


def set_default_render(device_id: str) -> bool:
    """Set the default PLAYBACK endpoint. Refuses a recording id.

    The guard is deliberately LOOSE — it rejects only the known-wrong prefix
    ({0.0.1.) and lets anything else through untouched. This half of the
    feature has been working in production since v2.0.97 and a strict
    allowlist could regress it on some id shape nobody here has seen; the
    only failure this needs to make impossible is the mix-up, and the mix-up
    has exactly one shape. Contrast set_default_capture, which is strict
    because every id it is ever handed comes from a prefix-filtered source."""
    if device_id and device_id.startswith(CAPTURE_PREFIX):
        _log(f"REFUSED: {device_id} is a RECORDING endpoint and was passed to the "
             f"PLAYBACK setter — that mix-up would silence the speakers. Nothing "
             f"was changed.")
        return False
    return set_default_endpoint(device_id)


# Read-back budget for a capture write. THREE reads, 50 ms apart, so the worst
# case a caller can ever block for is ~125 ms (measured 2026-09-05 on this
# machine: default_capture_id() costs 7.8 / 8.2 / 26.3 ms for min / median /
# max of nine reads). The retries exist only so a propagation lag cannot be
# mistaken for a refusal; a device that has genuinely moved is confirmed on the
# FIRST read and nothing sleeps at all.
_CAPTURE_READBACK_TRIES = 3
_CAPTURE_READBACK_WAIT_S = 0.05


def _capture_default_confirmed(device_id: str) -> bool:
    """Did the default RECORDING endpoint actually BECOME `device_id`?

    THE POINT OF THIS FUNCTION: S_OK IS NOT "SELECTED". `set_default_endpoint`
    reports whether the COM call returned success, and
    `set_default_endpoint`'s own honest-limits note records that a capture
    default was never once observed MOVING — the only live call was a write of
    the id that was already default for all three roles, a proven no-op. So
    every "the microphone moved" claim in this module rested on an HRESULT.
    If IPolicyConfigVista accepts a {0.0.1. id and does nothing (a real
    possibility on builds where the capture direction wants the non-Vista
    IPolicyConfig GUID), the old code announced "listening on <desk mic>" while
    Windows was still pointing at the powered-off headset and the VAD read
    0.0000. Nothing in the four deaf-safety rules caught it, because all four
    reason about the WRITE and none about the RESULT.

    Three outcomes, and they are logged as three DIFFERENT things because they
    are three different states of the world:
      * read-back == device_id  -> True. Verified, by the same reader
        `--test-mic` uses.
      * read-back == some other id -> False, and the log NAMES what Windows
        actually reports. This is the S_OK-without-a-move case.
      * read-back unreadable (None) -> False, and the log says UNVERIFIED
        rather than claiming the device did not move. "I could not prove it"
        is not "it failed", and this module does not get to pretend otherwise.

    Unverified counts as NOT moved on purpose. It is the safe direction: the
    caller then logs loudly and does not announce a switch it cannot back up,
    and the OFF-side rescue re-runs on the very next measured-off poll.

    HONEST LIMIT: the reader (`pycaw.AudioUtilities.GetMicrophone`, read
    2026-09-05) asks for eCapture/eMultimedia — ROLE 1 of the three roles the
    write sets. So this proves role 1 moved; roles 0 and 2 were written in the
    same call and are NOT separately read back. It also proves nothing about
    whether that microphone produces AUDIO — only the VAD's peak RMS can say
    that, and a dead device reads exactly 0.0000."""
    seen = None
    for attempt in range(1, _CAPTURE_READBACK_TRIES + 1):
        seen = default_capture_id()
        if seen and seen == device_id:
            if attempt > 1:
                _log(f"capture default confirmed on read-back attempt {attempt}")
            return True
        if attempt < _CAPTURE_READBACK_TRIES:
            time.sleep(_CAPTURE_READBACK_WAIT_S)
    if seen is None:
        _log(f"UNVERIFIED: the write of {device_id} returned success but Windows "
             f"would not name the default recording device on read-back, so I "
             f"cannot prove the microphone moved. Treating it as NOT moved — an "
             f"unverified switch is the one claim this module exists to refuse.")
    else:
        _log(f"NOT MOVED: SetDefaultEndpoint returned SUCCESS for {device_id}, but "
             f"the default recording endpoint still reads {seen} after "
             f"{_CAPTURE_READBACK_TRIES} read-backs. The write was ACCEPTED and the "
             f"device did NOT move — JARVIS is listening on {seen}, not on what was "
             f"asked for.")
    return False


def set_default_capture(device_id: str) -> bool:
    """Set the default RECORDING endpoint AND PROVE IT MOVED. Returns True only
    when a read-back says `device_id` is the default. Requires a recording id.

    STRICT on purpose: every id that reaches this comes from
    `find_active_capture()` (prefix-filtered) or `default_capture_id()` (the
    live capture default), so requiring {0.0.1. costs nothing and makes
    "JARVIS pointed its microphone at a pair of speakers" structurally
    impossible rather than merely unlikely. A refusal leaves the previous
    input device exactly where it was, which is the safe direction: a
    wrong-but-working microphone beats a right-but-silent one.

    WHY THE READ-BACK IS INSIDE THIS FUNCTION rather than at the call sites:
    this is the ONLY door to a capture write in the tree (audio_switch's two
    follow paths, skills/audio_autoswitch's two voice actions, and the
    --test-mic CLI all come through here), so putting the proof here means
    there is no second copy to rot. The repo's number-one bug class is the
    stale duplicate — a rule fixed in one copy while the others go on being
    wrong — and four call sites each remembering to verify is that bug waiting
    to happen. See `_capture_default_confirmed` for what the proof does and
    does not cover. Cost: ~8 ms on the rare write path (measured), against a
    286 ms full enumeration; nothing is added to the common no-write poll.

    NOT CHANGED HERE, deliberately: `set_default_render`. The render direction
    was OBSERVED moving on this machine 2026-09-05 00:48-00:52 (the output went
    to the speakers while the input stayed put), which is exactly the evidence
    the capture direction lacks, and a one-defect change should not quietly
    alter the half that is working. It is structurally the same gap and
    deserves the same treatment as its own change."""
    if not device_id:
        return False
    if not device_id.startswith(CAPTURE_PREFIX):
        _log(f"REFUSED: {device_id} is not a RECORDING endpoint id (expected a "
             f"{CAPTURE_PREFIX} prefix) — the microphone was NOT moved.")
        return False
    if not set_default_endpoint(device_id):
        return False            # the write itself failed; it already said why
    return _capture_default_confirmed(device_id)


def _au():
    from pycaw.pycaw import AudioUtilities
    return AudioUtilities


def list_render() -> list[tuple[str, str, str]]:
    """[(id, friendly_name, state)] for every render/all device, best effort."""
    out = []
    try:
        for d in _au().GetAllDevices():
            out.append((d.id, d.FriendlyName or "", str(getattr(d, "state", "?")).split(".")[-1]))
    except Exception as e:
        _log(f"enumerate failed: {e}")
    return out


def default_render_id() -> str | None:
    try:
        s = _au().GetSpeakers()
        return s.id if hasattr(s, "id") else s.GetId()
    except Exception:
        return None


def default_capture() -> tuple[str, str] | None:
    """(id, friendly name) of the live default RECORDING endpoint, or None.

    Deliberately a SINGLE-DEVICE reader rather than a search through
    `list_render()`. MEASURED 2026-09-05 on this machine: 8.5 ms here against
    286 ms for a full enumeration of its 51 endpoints — a 33x gap, and it is
    load-bearing. The OFF-side rescue in `AudioAutoSwitch._mic_off_headset`
    runs on EVERY measured-off poll (that is exactly what makes it
    self-healing), so answering "no, the microphone is fine" with a full
    enumeration would burn ~10% of a core continuously, forever.

    That is only HALF the loop, and the other half was missed for a day: the
    cheap answer here short-circuits when the default is SOMEONE ELSE'S
    microphone, which is precisely the case the rescue does not exist for. When
    the default IS the dead headset the name test cannot fire and the pass falls
    through to the enumeration anyway — so `_mic_off_headset` additionally holds
    a proven "nothing to move to" verdict for MIC_RESCUE_RETRY_S (deaf-safety
    rule 7). Both mechanisms are needed; neither replaces the other.

    Note `GetMicrophone()` alone is NOT enough: it hands back a raw IMMDevice
    with no FriendlyName and no `.id` (verified 2026-09-05 — reading `.id` on
    it raises AttributeError). `CreateDevice` is what opens the property store
    and yields both, and it is where the 8.5 ms goes.

    None means "Windows would not tell me", NOT "there isn't one" — every
    caller treats it as a reason to HOLD rather than to switch, the same rule
    the power detector uses for UNKNOWN."""
    try:
        au = _au()
        dev = au.CreateDevice(au.GetMicrophone())
        did = getattr(dev, "id", None)
        if not did:
            return None
        return did, (getattr(dev, "FriendlyName", "") or "")
    except Exception as e:
        _log(f"could not read the default recording device: {e}")
        return None


def default_capture_id() -> str | None:
    """Just the id of the live default RECORDING endpoint, or None."""
    cur = default_capture()
    return cur[0] if cur else None


def find_active(fragment: str, render_only: bool = True,
                rows: list[tuple[str, str, str]] | None = None) -> tuple[str, str] | None:
    """Return (id, name) of the first ACTIVE RENDER device whose friendly name
    contains `fragment` (case-insensitive), else None.

    WHAT "Active" ACTUALLY MEANS — corrected 2026-09-04, after the old wording
    here ("'Active' == powered on and present; a wireless headset that's OFF
    reads NotPresent/Unplugged") was finally measured and found FALSE:

        Active means the ENDPOINT is present and enabled. For a USB-dongle
        wireless headset the endpoint belongs to the DONGLE, so it stays
        Active while the headset itself is powered off. Measured 2026-09-04 on
        this machine with the CORSAIR VOID ELITE Wireless POWERED OFF, Windows
        reported both of its endpoints as Status=OK / Active:
            OK  Headset Earphone   (CORSAIR VOID ELITE Wireless Gaming Headset)
            OK  Headset Microphone (CORSAIR VOID ELITE Wireless Gaming Headset)

    So this function answers "is there a usable endpoint with this name?" and
    NOT "is the headset switched on?". For the power question call
    `headset_powered()`. NotPresent/Unplugged/Disabled do still mean unusable
    (a wired headphone jack really does read Unplugged), so this remains the
    right way to RESOLVE A DEVICE ID — it is only the power inference that was
    wrong. Whether some OTHER wireless headset flips its endpoint on power was
    not tested here; no claim is made either way.

    render_only filters to playback endpoints by MMDevice id prefix — render
    ids start with ``{0.0.0.`` and capture (mic) ids with ``{0.0.1.``. Without
    it a headset's MICROPHONE could be matched for an OUTPUT switch, which is
    wrong (the earphone and the mic share the device name).

    `rows` lets a caller pass an already-enumerated list_render() result so one
    poll doesn't enumerate the whole endpoint list twice; None = enumerate."""
    if not fragment:
        return None
    frag = fragment.lower()
    for did, name, state in (list_render() if rows is None else rows):
        if render_only and not did.startswith(RENDER_PREFIX):
            continue
        if state.lower() == "active" and frag in name.lower():
            return did, name
    return None


def find_active_capture(fragment: str,
                        rows: list[tuple[str, str, str]] | None = None
                        ) -> tuple[str, str] | None:
    """Return (id, name) of the first ACTIVE RECORDING device whose friendly
    name contains `fragment` (case-insensitive), else None.

    DO NOT use `find_active(fragment, render_only=False)` for this. That flag
    does not INVERT the filter, it DISABLES it — so the function then returns
    whichever endpoint comes first in the enumeration, and for a headset that
    is the earphone. MEASURED live 2026-09-05 on this machine:

        find_active("CORSAIR VOID ELITE", render_only=False)
          -> ("{0.0.0.…30c225e7}", "Headset Earphone (CORSAIR VOID ELITE …)")

    i.e. a PLAYBACK id handed back from what a reader would assume is the
    microphone lookup. Shipping that as "the mic resolver" would be the exact
    mirror image of the fallback defect this module was rewritten to fix (a
    microphone sitting in the speaker slot), and it would end with
    SetDefaultEndpoint being called on a render id for the input direction.
    Hence a separate function with its own explicit prefix test, and
    `set_default_capture()` refusing anything that is not {0.0.1. as a second
    line of defence.

    Same "Active" caveat as find_active: Active means the ENDPOINT exists and
    is enabled, NOT that the headset is powered. Power comes from
    `headset_powered()` and from nowhere else."""
    if not fragment:
        return None
    frag = fragment.lower()
    for did, name, state in (list_render() if rows is None else rows):
        if not did.startswith(CAPTURE_PREFIX):
            continue
        if state.lower() == "active" and frag in name.lower():
            return did, name
    return None


def endpoint_name(device_id: str,
                  rows: list[tuple[str, str, str]] | None = None) -> str | None:
    """Friendly name for an MMDevice id, or None when the id is not in `rows`.

    Used to say WHICH device something is in a log line. An id that resolves to
    no name is reported as unknown rather than guessed at."""
    if not device_id:
        return None
    for did, name, _state in (list_render() if rows is None else rows):
        if did == device_id:
            return name or None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Is the headset actually POWERED?  (three-valued — see the module docstring)
# ═══════════════════════════════════════════════════════════════════════════
_VOID_NAME_RE = re.compile(r"\bvoid\b", re.IGNORECASE)
_void_import_warned = False


def looks_like_corsair_void(fragment: str) -> bool:
    """True when this configured NAME FRAGMENT names a Corsair VOID headset.

    Matched on the fragment the owner configured, not against a hardcoded
    device string, so "CORSAIR VOID ELITE", "void elite", "Corsair VOID RGB"
    and a bare "VOID" all route to the HID detector. Word-boundary, so
    "avoid" and "Devoid" do not.

    HONEST LIMIT: only the VOID ELITE Wireless dongle (VID 1B1C / PID 0A51)
    was ever probed. audio.void_link is PID-gated, so a DIFFERENT VOID model
    would never be discovered and `headset_powered()` would report UNKNOWN
    forever — the feature goes inert rather than acting on the endpoint signal
    that is known to be wrong for this family. That is a deliberate choice of
    safe failure, NOT a measurement: no other VOID model was available to test.
    """
    return bool(fragment) and bool(_VOID_NAME_RE.search(fragment))


def _void_link():
    """audio.void_link, or None if it cannot be imported.

    Lazy so audio_switch keeps importing where void_link's HID plumbing does
    not exist (ubuntu CI). void_link itself never raises and degrades to
    'unknown' on its own, so this returning a module is not a promise that a
    dongle is present."""
    global _void_import_warned
    try:
        from audio import void_link
        return void_link
    except Exception as e:   # pragma: no cover - void_link guards its own imports
        if not _void_import_warned:
            _void_import_warned = True
            _log(f"audio.void_link unavailable ({e}) — VOID power state reads UNKNOWN")
        return None


def headset_powered(fragment: str) -> bool | None:
    """Is the headset named by `fragment` actually POWERED ON?

    THREE-VALUED and that is the entire point:
        True  — measured on
        False — measured off
        None  — genuinely unknown; the caller must HOLD, not assume off.

    Collapsing None into False is the defect that makes this feature misfire:
    the VOID's dongle answered nothing at all for 105 consecutive seconds
    while the headset was pairing (measured 2026-09-04), and "no answer" read
    as "off" would yank the default audio device mid-power-on.

    Routing:
      * a VOID-looking name -> audio.void_link (the dongle's own vendor HID
        report). It NEVER falls back to the endpoint check for these, because
        the endpoint check is *known* to answer "on" while this headset is
        off — falling back would reintroduce exactly the bug being fixed.
      * anything else -> the endpoint check, which is the only signal that
        exists for other hardware. An empty enumeration is reported as UNKNOWN
        rather than "off", since a machine with zero audio endpoints means COM
        or pycaw failed, not that a headset powered down.
    """
    if not fragment:
        return None
    if looks_like_corsair_void(fragment):
        vl = _void_link()
        if vl is None:
            return None
        try:
            return vl.is_headset_on()      # True / False / None; never raises
        except Exception as e:             # pragma: no cover - defensive only
            _log(f"void_link probe failed ({e}) — reporting UNKNOWN, not off")
            return None
    rows = list_render()
    if not rows:
        # list_render() already logged why. Zero endpoints is a broken
        # enumeration, not a powered-down headset.
        return None
    try:
        return find_active(fragment, rows=rows) is not None
    except Exception as e:                 # pragma: no cover - defensive only
        _log(f"endpoint check for '{fragment}' failed ({e}) — UNKNOWN, not off")
        return None


def void_battery_pct(fragment: str) -> int | None:
    """void_link's last believed battery-ish reading for a VOID, or None.

    Does not start a new HID exchange — it reads the value the last
    `headset_powered()` poll already fetched, so calling it every tick is free.
    None whenever the link is not believed up: a powered-down headset reads
    0x00 in that byte and "0%" would be a fabricated number.

    WHY THIS EXISTS: measured 2026-09-04, HWiNFO is running here with Shared
    Memory Support ON (815 readings visible) and contains NO Corsair / VOID /
    headset sensor at all — `hwinfo.battery("VOID")` and `hwinfo.find("VOID")`
    both returned None while void_link reported 90. So the HWiNFO path has
    never had a number for this headset and the low-battery warning could
    never have fired.

    DO NOT TRUST THIS AS A CALIBRATED PERCENTAGE. void_link decodes it as a
    battery percent, but measured here 2026-09-04 across ~25 minutes with the
    headset sitting idle and ON, the value read 90, then 85, then 82, then 87,
    then 85 — and dropped from 87 to 85 in FOUR SECONDS (20:48:06 -> 20:48:08).
    It is non-monotonic and jitters by several points on a seconds timescale,
    which no real fuel gauge does; it looks more like an instantaneous voltage
    reading. What the number physically means was NOT determined. It is fine
    for "roughly how full", and `low_pct` warnings from it may flap near the
    threshold (bounded: `_check_low_battery` warns once and re-arms only after
    a 5-point recovery)."""
    if not looks_like_corsair_void(fragment):
        return None
    vl = _void_link()
    if vl is None:
        return None
    try:
        return vl.battery_percent()
    except Exception:                      # pragma: no cover - defensive only
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Fallback resolution — loud on failure (a silent no-op here was defect #2)
# ═══════════════════════════════════════════════════════════════════════════
def fallback_problem(fragment: str,
                     rows: list[tuple[str, str, str]] | None = None) -> str | None:
    """Why `fragment` cannot serve as the fallback PLAYBACK device, or None.

    Written because the saved setting really was
    ``"AUDIO_AUTOSWITCH_FALLBACK": "Blue Snowball"`` — a MICROPHONE. Resolved
    through `find_active(..., render_only=True)` a mic can never match, so the
    switch-away silently did nothing, forever, with no message anywhere.
    Verified live 2026-09-04: find_active("Blue Snowball") -> None while
    find_active("Blue Snowball", render_only=False) -> the capture endpoint
    "Microphone (Blue Snowball )"."""
    if not fragment:
        return None                        # nothing configured is not a fault
    rows = list_render() if rows is None else rows
    if find_active(fragment, rows=rows) is not None:
        return None
    if not rows:
        return (f"the audio device list came back EMPTY, so the fallback "
                f"'{fragment}' could not be resolved (enumeration failed)")
    frag = fragment.lower()
    matches = [(did, name, state) for did, name, state in rows if frag in name.lower()]
    if not matches:
        return (f"'{fragment}' matches NO audio device on this machine — check the "
                f"name against `python -m audio.audio_switch --list`")
    render = [m for m in matches if m[0].startswith("{0.0.0.")]
    if render:
        states = ", ".join(sorted({m[2] for m in render}))
        return (f"'{fragment}' matches the playback device '{render[0][1]}' but its "
                f"state is {states} — not Active — so it cannot be selected")
    return (f"'{fragment}' matches only a RECORDING device ('{matches[0][1]}') — a "
            f"microphone can never be a playback fallback. Point "
            f"AUDIO_AUTOSWITCH_FALLBACK at a playback device (your speakers) instead")


def resolve_fallback(fragment: str) -> tuple[str, str] | None:
    """(id, name) of the fallback RENDER device, or None — and it SAYS WHY.

    Never silently returns None for a configured-but-unusable fallback; that
    silence is what hid a microphone sitting in the fallback slot."""
    if not fragment:
        return None
    rows = list_render()
    hit = find_active(fragment, rows=rows)
    if hit is not None:
        return hit
    problem = fallback_problem(fragment, rows=rows)
    if problem:
        _log(f"FALLBACK UNUSABLE: {problem}")
    return None


def mic_fallback_problem(fragment: str,
                         rows: list[tuple[str, str, str]] | None = None) -> str | None:
    """Why `fragment` cannot serve as the fallback MICROPHONE, or None.

    The capture-side twin of `fallback_problem`, and it exists because the
    same trap is pre-loaded on this side. MEASURED live 2026-09-05: the value
    already sitting in AUDIO_AUTOSWITCH_FALLBACK, "Realtek USB2.0 Audio",
    resolves to ZERO usable recording endpoints on this machine —

        Unplugged  Line In (Realtek USB2.0 Audio)
        Unplugged  Microphone (Realtek USB2.0 Audio)
        Disabled   Internal AUX Jack (Realtek USB2.0 Audio)

    — so reusing the playback fallback for the input direction would resolve
    to nothing and no-op in silence, which is defect #2 all over again with
    the two directions swapped. That is why the mic side has its OWN setting
    (AUDIO_AUTOSWITCH_MIC_FALLBACK) and its own diagnosis."""
    if not fragment:
        return None                        # nothing configured is not a fault
    rows = list_render() if rows is None else rows
    if find_active_capture(fragment, rows=rows) is not None:
        return None
    if not rows:
        return (f"the audio device list came back EMPTY, so the microphone "
                f"fallback '{fragment}' could not be resolved (enumeration failed)")
    frag = fragment.lower()
    matches = [(did, name, state) for did, name, state in rows if frag in name.lower()]
    if not matches:
        return (f"'{fragment}' matches NO audio device on this machine — check the "
                f"name against `python -m audio.audio_switch --list-mics`")
    capture = [m for m in matches if m[0].startswith(CAPTURE_PREFIX)]
    if capture:
        states = ", ".join(sorted({m[2] for m in capture}))
        listing = "; ".join(f"{n} [{st}]" for _d, n, st in capture)
        return (f"'{fragment}' matches {len(capture)} recording device(s) but none "
                f"is Active — {listing} (state{'s' if len(capture) > 1 else ''}: "
                f"{states}) — so none can be selected as a microphone")
    return (f"'{fragment}' matches only a PLAYBACK device ('{matches[0][1]}') — a "
            f"speaker can never be a microphone. Point "
            f"AUDIO_AUTOSWITCH_MIC_FALLBACK at a recording device (your desk mic) "
            f"instead")


def resolve_mic_fallback(fragment: str,
                         rows: list[tuple[str, str, str]] | None = None,
                         quiet: bool = False
                         ) -> tuple[str, str] | None:
    """(id, name) of the fallback RECORDING device, or None — and it SAYS WHY.

    Never silently returns None for a configured-but-unusable fallback. On the
    input direction that silence is not merely untidy: it is the difference
    between "moved him to the desk mic" and "left him deaf on a powered-down
    headset", so the caller can also tell the two apart by the log line.

    `quiet=True` suppresses ONLY the repeat. It exists because the OFF rescue
    re-runs this diagnosis on a timer while the fault persists, and the same
    sentence 1,200 times an hour is not louder than saying it once — it is
    quieter, because it buries everything else in JARVIS's stdout. The caller
    that passes it (`AudioAutoSwitch._mic_off_headset`) is required to print
    the full diagnosis on first sight of a fault and again every
    MIC_RESCUE_RELOG_S while it lasts; the default stays False so no other
    caller can lose a message by accident."""
    if not fragment:
        return None
    rows = list_render() if rows is None else rows
    hit = find_active_capture(fragment, rows=rows)
    if hit is not None:
        return hit
    problem = mic_fallback_problem(fragment, rows=rows)
    if problem and not quiet:
        _log(f"MIC FALLBACK UNUSABLE: {problem}")
    return None


# ── does JARVIS'S OWN capture actually follow the Windows default? ───────────
# THE BUG THIS EXISTS TO KILL (measured 2026-09-05, live config):
#   data/user_settings.json carries PREFERRED_INPUT_DEVICES =
#   ['Blue Snowball', 'eMeet C960', 'CORSAIR VOID']. bobert_companion's capture
#   loop asks _pick_device(PREFERRED_INPUT_DEVICES, want_input=True) FIRST and
#   only falls through to the Windows default when that returns None. The
#   Snowball is connected and openable, so it wins on every single pass and
#   `in_idx` is never None - which means the one branch that resolves the live
#   default endpoint (_endpoint_device_identity(..., want_input=True)) NEVER
#   RUNS. set_default_capture() below therefore moves the WINDOWS default while
#   JARVIS goes on recording through the Snowball, and _mic_to_headset used to
#   say "listening on Headset Microphone (CORSAIR VOID ...)" anyway. The owner
#   puts the headset on, is told the boom mic is live, talks into it, and is
#   heard through a desk mic across the room.
#
# That precedence is NOT a mistake and must not be "fixed" here: the monolith's
# own comment records that a name-pinned list beating the default is what saved
# it from a powered-off pinned headset - 90 minutes deaf, 2026-08-20. So the
# honest move is not to change who wins, it is to STOP CLAIMING otherwise.
#
# WHAT THIS CAN AND CANNOT PROVE. It reads the two settings that decide the
# precedence, from the SAME module namespace the capture loop reads them from
# (bobert_companion's globals, populated by `from core.config import *`). That
# makes it a statement about WHICH SELECTION RULE WINS - which is exactly the
# claim being made - and it is deliberately NOT a claim that any microphone is
# producing audio. Proving THAT needs signal (the VAD's peak RMS; a dead device
# reads exactly 0.0000) and signal needs the input stream, which this process
# must never open from a daemon thread: sd.stop()/rec()/wait() act on a module-
# global _last_callback rather than the stream handed to them, and
# _StreamBase.close() calls Pa_CloseStream then nulls _ptr with no lock, so a
# concurrent close from here is heap corruption. The monolith owns that stream.
def capture_override() -> str | None:
    """None when JARVIS'S recording really does follow the Windows default
    capture device; otherwise a short spoken phrase naming what overrides it.

    Never raises and never imports the monolith - it only looks in sys.modules,
    so `python -m audio.audio_switch` and the unit tests stay standalone. When
    the monolith is not loaded there is no capture loop to override, so the
    answer is None."""
    try:
        bc = sys.modules.get("bobert_companion")
        if bc is None:
            return None

        idx = getattr(bc, "MICROPHONE_INDEX", None)
        if isinstance(idx, int) and not isinstance(idx, bool):
            if idx < 0:
                return f"my microphone is switched off in settings (index {idx})"
            return f"I am pinned to microphone index {idx}"

        pref = getattr(bc, "PREFERRED_INPUT_DEVICES", None) or []
        try:
            listed = ", ".join(str(x) for x in pref)
        except Exception:
            listed = ""
        if listed:
            return (f"I take the first connected match from my preferred "
                    f"microphone list instead ({listed})")
        return None
    except Exception as e:
        # Cannot read the precedence => cannot confirm the claim. Hedging is
        # the safe direction here: the caller only uses a non-None answer to
        # SOFTEN what it says, never to skip a switch, so an error here can
        # make JARVIS vaguer but can never make him deaf.
        return f"I could not check what my microphone selection is following ({e})"


def capture_claim(name: str, moved: str) -> str:
    """The one sentence the daemon is allowed to say about the microphone.

    `moved` is the transition ("headset on" / "headset off"). Says "listening
    on X" ONLY when capture_override() confirms nothing outranks the Windows
    default; otherwise it reports what actually happened - the Windows default
    moved - and names what JARVIS is really following instead."""
    why = capture_override()
    if why is None:
        return f"{moved} - listening on {name}"
    _log(f"the Windows default recording device is now {name}, but JARVIS is NOT "
         f"recording from it: {why}. The capture auto-switch cannot reach "
         f"JARVIS's own microphone until that override is cleared - clear "
         f"PREFERRED_INPUT_DEVICES (and leave MICROPHONE_INDEX unset) in "
         f"data/user_settings.json, then restart, so the input direction "
         f"follows the Windows default again.")
    return (f"{moved} - Windows' microphone is now {name}, but I am not "
            f"recording from it: {why}")


# ═══════════════════════════════════════════════════════════════════════════
# Is the selected microphone actually PRODUCING AUDIO?  (the one signal test)
# ═══════════════════════════════════════════════════════════════════════════
# Everything else in this module is careful to say that "selected" is not
# "producing audio". That honesty left a hole, and the ON side fell straight
# into it: `_follow_mic_state` had `if was_on is True: return None` as its
# ENTIRE steady-state ON branch, so once the capture default had been moved
# onto the headset's microphone nothing re-evaluated it for as long as the
# headset stayed powered on. A headset that is ON but hearing nothing - boom
# mic flipped up (its own hardware mute), endpoint muted in Windows, owner in
# the next room - therefore reads exactly 0.0000 RMS for hours, and the OFF
# rescue cannot save him because the headset MEASURES powered on.
#
# THIS MODULE STILL MUST NOT OPEN A STREAM. sd.stop()/sd.rec()/sd.wait() act on
# a module-global _last_callback rather than on the stream handed to them, and
# _StreamBase.close() calls Pa_CloseStream then nulls _ptr with no lock, so a
# second stream opened from this daemon thread is 0xc0000374 heap corruption
# with no traceback. The monolith owns the input stream.
#
# IT DOES NOT HAVE TO. The monolith's capture loop already measures exactly
# this and already publishes it. core.audio_processor keeps three numbers,
# written by record_speech on EVERY chunk it inspects:
#
#   last_vad_poll_ts       note_vad_poll() - proves the capture loop is running
#                          RIGHT NOW. Without this, "no audio" cannot be told
#                          apart from "nobody is listening", and reading the
#                          second as the first is how a false alarm would move
#                          a working microphone.
#   last_audible_chunk_ts  note_raw_rms() - updated only when a chunk's RAW rms
#                          crosses core.audio_processor._AUDIBLE_RMS_FLOOR
#                          (1e-5). A device handing back null samples never
#                          moves it.
#   vad_session_start      the first poll, so a device that has NEVER produced
#                          audio this session is measurable from cold start -
#                          which is the boot case (headset already on, boom up).
#
# Reading them costs a dict copy and opens no device. The 1e-5 floor and the
# claim that a working capture device's noise floor sits above it are
# core.audio_processor's, made there in 2026-05-30 and reused here rather than
# re-asserted.

# How stale the capture loop's last inspected chunk may be and still count as
# "listening right now". record_speech polls continuously while it holds the
# mic but does NOT hold it while JARVIS is speaking or thinking, so a gap
# between turns is normal and must never be read as evidence of silence.
CAPTURE_POLL_FRESH_S = 15.0


def _secs(n: float) -> str:
    """"1 second" / "2 seconds". `why` below is SPOKEN as well as logged, and
    a TTS voice reading "1 seconds" is the kind of small wrongness that makes
    the rest of the sentence easier to disbelieve."""
    whole = int(round(n))
    return f"{whole} second" + ("" if whole == 1 else "s")


def capture_signal_state(min_silent_s: float,
                         poll_fresh_s: float = CAPTURE_POLL_FRESH_S
                         ) -> tuple[str, str]:
    """Has JARVIS's capture actually HEARD anything lately?

    Returns (state, why) where state is one of:
        "silent"  - MEASURED: the capture loop is running and every chunk it
                    inspected in the last `min_silent_s` seconds was below the
                    audible floor. This is the only answer that may move a
                    device.
        "audible" - MEASURED: something crossed the floor recently.
        "unknown" - no evidence. The loop is not running, has never run, the
                    watchdog is switched off, or the numbers could not be read.

    THREE-VALUED for the same reason headset_powered() is, and the UNKNOWN row
    is load-bearing in the same way: JARVIS is not listening between turns, and
    reading that gap as "the microphone is dead" would yank a working device
    every time he asked a long question. Callers HOLD on unknown.

    `why` is written first-person and spoken-safe, because it is used verbatim
    both in the log and in the sentence _say_deaf() puts in his ear - one
    string, so the two can never drift into two different truths.

    Never imports the monolith's audio stack: it only looks in sys.modules, the
    same discipline capture_override() uses, so `python -m audio.audio_switch`
    and the unit tests stay standalone and no daemon thread can drag numpy or
    PortAudio in behind the owner's back. Not loaded = no capture loop = no
    evidence = unknown."""
    try:
        if float(min_silent_s) <= 0:
            return ("unknown", "the silent-microphone watchdog is switched off "
                               "(AUDIO_AUTOSWITCH_MIC_SILENT_S is not positive)")
        ap = sys.modules.get("core.audio_processor")
        if ap is None:
            return ("unknown", "nothing in this process has measured a microphone "
                               "yet (the capture pipeline is not loaded)")
        state = ap.get_vad_state() or {}
        now = time.time()
        poll_ts = float(state.get("last_vad_poll_ts") or 0.0)
        if poll_ts <= 0.0:
            return ("unknown", "I have not listened to a single chunk of audio "
                               "yet this session")
        poll_age = now - poll_ts
        if poll_age > float(poll_fresh_s):
            return ("unknown", f"I was not listening just now (my last chunk of "
                               f"audio was {_secs(poll_age)} ago), so silence "
                               f"would prove nothing")
        audible_ts = float(state.get("last_audible_chunk_ts") or 0.0)
        ref = audible_ts if audible_ts > 0.0 else float(
            state.get("vad_session_start") or 0.0)
        if ref <= 0.0:
            return ("unknown", "my own listening timestamps are unset, so I cannot "
                               "tell how long it has been")
        silent_age = now - ref
        if silent_age >= float(min_silent_s):
            ever = ("" if audible_ts > 0.0 else
                    " - I have not heard anything at all since I started")
            return ("silent",
                    f"I have heard nothing at all through my microphone for "
                    f"{_secs(silent_age)}, though I was listening the whole "
                    f"time (my last chunk of audio was {_secs(poll_age)} "
                    f"ago){ever}")
        return ("audible", f"I heard sound {_secs(silent_age)} ago")
    except Exception as e:
        # Cannot read the signal => no evidence. Never "silent": an error here
        # must be able to make the watchdog inert, never make it act.
        return ("unknown", f"I could not read my own microphone signal ({e})")


# ── LAST-RESORT microphone ranking (AudioAutoSwitch._last_resort_captures) ──
# Both regexes express a PREFERENCE and never an exclusion: every ACTIVE
# recording endpoint that is not the powered-off headset stays a candidate,
# because the only device here that has been MEASURED dead is the one he is
# already on.
#
# THIS IS NOT A SIGNAL TEST and nothing here pretends it is. Whether a
# microphone is actually producing audio can only be answered by CAPTURING from
# it, and this module must not open a stream to do so: the monolith owns the
# input stream (sounddevice/PortAudio), sd.stop()/sd.rec() act on a
# module-global rather than on the stream handed to them, and
# _StreamBase.close() nulls its pointer with no lock — a second stream opened
# from this daemon thread is a heap-corruption crash with no traceback. So the
# ranking is an ORDERING GUESS over candidates that are all equally unproven,
# and every log line it produces says so.
#
# NARROWED 2026-09-05, because the sweeping version of that sentence was itself
# an unchecked claim. There IS one microphone whose signal this module can read
# without capturing anything: the one the monolith is recording from RIGHT NOW,
# whose peak RMS its capture loop already measures and publishes through
# core.audio_processor — see capture_signal_state(). That is what the ON side's
# silence watchdog runs on. It does not help HERE, because a CANDIDATE is by
# definition a device nothing is recording from yet, so these really are
# unproven. The distinction matters: "no signal is available for this device"
# is true, "no signal is available at all" was not.
#
# What the guess is built from, both learned tonight:
_SOFTWARE_MIC_RE = re.compile(
    # A virtual mic is a ROUTE, not a microphone: it produces audio only while
    # the app behind it is running, and a loopback/monitor device hears the
    # SPEAKERS rather than the owner. Measured on this machine 2026-09-05, one
    # of the six Active recording endpoints is "Microphone (Voicemod)".
    r"voicemod|voicemeeter|vb-?audio|virtual|stereo mix|what u hear|"
    r"cable output|loopback|wave ?link|sound mapper|steam streaming|"
    r"nvidia broadcast",
    re.IGNORECASE)
_SELF_POWERED_MIC_RE = re.compile(
    # Anything carrying its own battery can be off in exactly the way tonight's
    # headset was off — a sleeping DualSense enumerates Active and hears
    # nothing, which is the same lie the CORSAIR told. Ranked below USB/mains
    # devices, never excluded, because "probably asleep" still beats "measured
    # dead".
    r"wireless|bluetooth|headset|dualsense|airpods",
    re.IGNORECASE)


def mic_liveness_rank(name: str) -> int:
    """0 (best) / 1 / 2 — how good a LAST-RESORT bet this device name looks.

    An ORDERING over unproven candidates, not a verdict on any of them. See the
    comment block above: no signal exists for a device nothing is recording
    from, which is what every candidate here is."""
    n = name or ""
    if _SOFTWARE_MIC_RE.search(n):
        return 2
    if _SELF_POWERED_MIC_RE.search(n):
        return 1
    return 0


class AudioAutoSwitch:
    """Background watcher: on the headset's power transitions, move the Windows
    default render device. Idempotent + terminable. Mirrors JARVIS's other
    opt-in daemons (start/stop, a _STOP event, never raises into the loop).

    THE STATE MACHINE IS THREE-VALUED. `_believed_on` is True / False / None
    and an UNKNOWN sample changes nothing at all:

        believed | measured | action
        ---------+----------+----------------------------------------------
        anything | UNKNOWN  | HOLD — no switch, belief untouched
        None     | ON       | switch to the headset (startup sync; a no-op
                 |          | when it is already the default)
        None     | OFF      | record only — never yank the default at boot
                 |          | just because the headset happens to be off
        False    | ON       | switch to the headset
        True     | OFF      | switch away (prior default, else fallback)
        same     | same     | nothing

    The UNKNOWN row is load-bearing: the VOID dongle answered nothing for 105
    consecutive seconds while the headset paired (measured 2026-09-04). At the
    default 3 s poll that is ~35 unknown samples in a row, every single time
    the owner powers his headset on.
    """

    # ── the OFF rescue's cost ceiling ────────────────────────────────────────
    # MEASURED 2026-09-05 on this machine: list_render() = 334.9 / 307.4 /
    # 305.3 / 290.6 ms over 51 endpoints, against 8.0 ms for default_capture().
    # The OFF rescue is evaluated on EVERY measured-off poll (that is what makes
    # it self-healing), so anything it does unconditionally is multiplied by
    # 1200 an hour at the default 3 s poll.
    #
    # Once a pass has ENUMERATED and PROVEN there is nothing verified to move
    # to, repeating that same enumeration 3 seconds later can only produce the
    # same answer unless something changed — and everything that could change
    # the answer cheaply is already in the hold key, which cancels the hold the
    # instant it moves. So the enumeration is spaced out instead:
    #
    #   MIC_RESCUE_RETRY_S  how long a PROVEN-stuck verdict is trusted before
    #                       paying for another full enumeration. 30 s at a 3 s
    #                       poll is 1 enumeration per 10 polls: ~1% of a core
    #                       instead of ~10%, and the worst case it can cost is
    #                       a rescue that fires up to 30 s late — never one that
    #                       does not fire.
    #   MIC_RESCUE_RELOG_S  how often the full diagnosis is allowed to repeat
    #                       itself in the log while the SAME fault persists.
    #                       The first sighting always prints in full, and the
    #                       repeat carries the count of what was suppressed, so
    #                       nothing is hidden — it is just not said 1,200 times
    #                       an hour into the stdout the owner has to read.
    MIC_RESCUE_RETRY_S = 30.0
    MIC_RESCUE_RELOG_S = 300.0

    # SPEECH is throttled SEPARATELY from the log, because the two limits are
    # answering different questions. MIC_RESCUE_RELOG_S keeps a log file
    # readable; these keep a VOICE bearable, and the right number is not the
    # same. A spoken deaf-risk alert lands immediately, repeats after
    # DEAF_SPEAK_FIRST_S, then backs off by doubling up to DEAF_SPEAK_MAX_S.
    #
    # It backs off forever rather than stopping after N. A JARVIS that cannot
    # hear cannot be TOLD to stop talking about it, so "stop after three" would
    # hand the owner exactly the silence this whole feature exists to remove —
    # and a fault that makes the assistant useless is not noise. Half an hour
    # is the floor on how long he can be deaf without being told again.
    DEAF_SPEAK_FIRST_S = 300.0
    DEAF_SPEAK_MAX_S = 1800.0

    def __init__(self, headset: str, fallback: str = "", poll_s: float = 3.0,
                 announce=None, mic_fallback: str = "", follow_mic: bool = False,
                 mic_silent_s: float = 60.0):
        self.headset = headset
        self.fallback = fallback           # name fragment, or "" = remember prior
        self.poll_s = max(1.0, float(poll_s))
        self.announce = announce or (lambda msg: _log(msg))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prior_default: str | None = None
        self._low_warned = False
        self.low_pct = 15            # warn once when battery drops below this
        # None = nothing believed yet (startup). NOT False — see tick().
        self._believed_on: bool | None = None
        self._unknown_streak = 0
        # ── the INPUT half (opt-in, OFF unless the caller asks for it) ───────
        self.mic_fallback = mic_fallback   # desk-mic name fragment
        self.follow_mic = bool(follow_mic)
        self._prior_capture: str | None = None
        self.last_mic_label: str | None = None
        # ── the STEADY-STATE ON signal watchdog (see _mic_silence_watchdog) ──
        # Seconds of MEASURED digital silence, while the capture loop is
        # demonstrably running, before the headset's own microphone is treated
        # as dead. <= 0 switches the watchdog off and restores the pre-
        # 2026-09-05 behaviour, where the ON side fired once and never looked
        # again. The default matches core/config.AUDIO_AUTOSWITCH_MIC_SILENT_S
        # so a caller that forgets to wire it up still gets the protection.
        try:
            self.mic_silent_s = float(mic_silent_s)
        except Exception:
            self.mic_silent_s = 60.0
        # ── the OFF rescue's bounded-retry state (see _mic_off_headset) ─────
        # Set once a pass has PROVEN, against a real enumeration, that this
        # exact cheap state has nothing verified to move to. None = no such
        # verdict is being held, so the next measured-off poll pays in full.
        self._mic_hold_key: tuple | None = None
        self._mic_hold_next: float = 0.0     # monotonic: next enumeration allowed
        self._mic_hold_relog: float = 0.0    # monotonic: next full diagnosis allowed
        self._mic_hold_skipped: int = 0      # polls short-circuited since the last one
        # ── the SPOKEN deaf-risk alert's own state (see _say_deaf) ───────────
        self._deaf_key: str | None = None    # which fault is currently being said
        self._deaf_seen: int = 0             # consecutive samples showing it
        self._deaf_said_at: float = 0.0      # monotonic of the last time it was said
        self._deaf_said_n: int = 0           # how many times it has been said

    def battery_pct(self) -> float | None:
        """Headset battery %, or None when nothing can supply one.

        Prefers void_link's reading for a VOID (measured 2026-09-04: HWiNFO
        has no sensor for this headset at all, so the HWiNFO path returned
        None while the dongle itself answered 90), and falls back to HWiNFO
        shared memory for any other headset. See `void_battery_pct` — that
        number is jittery and its physical meaning is not established, so
        treat it as approximate."""
        v = void_battery_pct(self.headset)
        if v is not None:
            return float(v)
        try:
            from audio import hwinfo
            return hwinfo.battery(self.headset)
        except Exception:
            return None

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="audio-autoswitch", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> str:
        """Spoken status — and it admits when it does not know.

        'off' and 'I can't tell' are different answers and are reported as
        different answers; saying "off" for an unknown would be the same
        unverified confidence that broke this feature."""
        powered = headset_powered(self.headset)
        running = bool(self._thread and self._thread.is_alive())
        head = f"Audio auto-switch is {'running' if running else 'stopped'}, sir."
        if powered is None:
            return (f"{head} I can't tell whether the '{self.headset}' headset is "
                    f"powered — the dongle isn't answering, so I'm holding the "
                    f"current device.")
        batt = self.battery_pct()
        suffix = f" at {round(batt)}% battery" if (powered and batt and batt > 0) else ""
        return (f"{head} The '{self.headset}' headset is "
                f"{'ON' + suffix if powered else 'off'}.")

    def _check_low_battery(self) -> None:
        """Announce once when the headset battery drops below low_pct; re-arm
        when it recovers (recharged) so the next drain warns again."""
        batt = self.battery_pct()
        if batt is None or batt <= 0:
            return
        if batt < self.low_pct and not self._low_warned:
            self._low_warned = True
            self.announce(f"headset battery is low — {round(batt)} percent, sir")
        elif batt >= self.low_pct + 5:
            self._low_warned = False

    def _run(self) -> None:  # pragma: no cover - daemon loop, logic unit-tested via tick()
        if _HAS_COM:
            try:
                comtypes.CoInitialize()
            except Exception:
                pass
        # Initial sync happens inside tick(): believed=None + measured ON grabs
        # the headset, believed=None + measured OFF only records. There is no
        # separate pre-loop probe any more — one sample per pass, one source of
        # truth, so the loop can never disagree with the state machine.
        try:
            self.tick()
        except Exception as e:
            _log(f"initial tick error: {e}")
        while not self._stop.wait(self.poll_s):
            try:
                self.tick()
                self._check_low_battery()
            except Exception as e:
                _log(f"tick error: {e}")
        if _HAS_COM:
            try:
                comtypes.CoUninitialize()
            except Exception:
                pass

    def tick(self, was_on: bool | None = None) -> str | None:
        """One poll step. Returns a short action label or None.

        `was_on` defaults to the watcher's own believed state; pass it
        explicitly (as the existing tests do) to drive one transition in
        isolation. None means "nothing believed yet".

        UNKNOWN HOLDS: a sample that cannot answer returns None and leaves both
        the belief and the audio device exactly where they were."""
        if was_on is None:
            was_on = self._believed_on
        now = headset_powered(self.headset)

        if now is None:
            self._unknown_streak += 1
            if self._unknown_streak == 1:
                self._log_unknown_start(was_on)
            return None

        if self._unknown_streak:
            _log(f"headset state readable again after {self._unknown_streak} "
                 f"unknown sample(s) — measured {'ON' if now else 'OFF'}")
            self._unknown_streak = 0

        self._believed_on = now
        label = None
        try:
            if now and not was_on:
                # Covers False->True and the startup None->True sync. Idempotent:
                # _switch_to_headset() no-ops when the headset is already default.
                label = self._switch_to_headset()
            elif was_on is True and not now:
                label = self._switch_away()
            # Deliberately no branch for None->False on the OUTPUT side: at
            # startup with the headset already off, the owner's chosen speakers
            # are left exactly alone. The INPUT side does NOT get that pass —
            # see _follow_mic_state; a default microphone that IS the
            # powered-off headset is not a preference, it is the deafness.
        finally:
            # `finally`, not a plain following statement, and it is load-bearing.
            # DEAF-SAFETY OUTRANKS THE SPEAKERS: if the output half raises —
            # a COM failure, an enumeration that blows up mid-pass — the INPUT
            # rescue must still run, because the state it repairs is JARVIS
            # being unable to hear at all. Caught 2026-09-05 by
            # MicDeafSafetyTests::test_an_exception_in_the_mic_half_...: with a
            # plain statement here, one raise in _switch_away() silently
            # skipped the microphone entirely. The exception still propagates
            # to _run(), which logs "tick error" exactly as it did before, so
            # nothing about the output half's reporting changed.
            if self.follow_mic:
                self.last_mic_label = self._follow_mic_state(now, was_on)
            else:
                self.last_mic_label = None
        return label

    def _log_unknown_start(self, was_on: bool | None) -> None:
        """Say — once per silent run — that the state is unreadable and why the
        watcher is doing nothing about it. This is the 105-second pairing
        window; it is expected, and it must not look like a hang."""
        held = {True: "ON", False: "off", None: "nothing believed yet"}[was_on]
        detail = ""
        if looks_like_corsair_void(self.headset):
            vl = _void_link()
            try:
                found = vl.discover_device() is not None if vl else False
            except Exception:
                found = False
            detail = (" (dongle interface found, but it is not answering — normal "
                      "for ~2 minutes while the headset pairs)" if found else
                      " (no VOID dongle interface found at all — unplugged, or held "
                      "by another process)")
        _log(f"can't tell whether '{self.headset}' is powered{detail}; HOLDING at "
             f"{held} — unknown is never treated as off")

    def _switch_to_headset(self) -> str | None:
        hs = find_active(self.headset)
        if not hs:
            _log(f"'{self.headset}' measures POWERED ON but no ACTIVE playback "
                 f"endpoint matches that name — nothing to switch to. Check the "
                 f"name against `python -m audio.audio_switch --list`.")
            return None
        cur = default_render_id()
        if cur == hs[0]:
            return None
        self._prior_default = cur
        if set_default_render(hs[0]):
            self.announce(f"headset on — audio moved to {hs[1]}")
            return "to_headset"
        _log(f"headset on but the switch to {hs[1]} FAILED — default left alone")
        return None

    def _switch_away(self) -> str | None:
        # Headset just powered off. Prefer the default we had before we grabbed
        # the headset; else a configured fallback fragment; else leave Windows'
        # own pick — but SAY so either way. The old version returned None in
        # silence when the fallback matched nothing, which is how a microphone
        # sat in the fallback slot unnoticed.
        target = self._prior_default
        tname = "the previous device"
        if not target and self.fallback:
            f = resolve_fallback(self.fallback)     # logs loudly when unusable
            if f:
                target, tname = f[0], f[1]
        if not target:
            why = (f"fallback '{self.fallback}' did not resolve" if self.fallback
                   else "no fallback configured and no prior default remembered")
            _log(f"headset off but there is nothing to switch back to ({why}) — "
                 f"leaving the default where Windows has it")
            return None
        if set_default_render(target):
            self.announce(f"headset off — audio back to {tname}")
            self._prior_default = None
            return "away"
        _log(f"headset off but the switch back to {tname} FAILED — default unchanged")
        return None


    # ══════════════════════════════════════════════════════════════════════
    # THE INPUT HALF — make the microphone follow the headset's power
    # ══════════════════════════════════════════════════════════════════════
    # WHAT WENT WRONG (measured 2026-09-05 00:48-00:52). The owner powered the
    # headset off. The OUTPUT moved correctly to his speakers. The INPUT did
    # not move at all, because nothing in this repo had ever written the
    # default CAPTURE endpoint - so Windows kept pointing the microphone at the
    # dead headset and JARVIS logged peak RMS = 0.0000 twice, thirty seconds
    # apart. Exactly zero. That is what "deaf" reads like in the log.
    #
    # THE ASYMMETRY, and it is deliberate - do not "tidy" it into symmetry:
    #
    #   ON  (headset powered up) is a PREFERENCE. It fires only on a TRANSITION
    #       into ON, so that after it has moved once the owner can override the
    #       microphone by hand and the watcher will not yank it back three
    #       seconds later.
    #       BUT A PREFERENCE IS NOT A LICENCE TO STOP LOOKING (fixed
    #       2026-09-05). Firing once and never re-checking meant that once the
    #       capture default had been moved onto the headset's microphone,
    #       nothing re-evaluated it for as long as the headset stayed powered
    #       on - and a headset can be powered on while hearing nothing at all
    #       (boom mic flipped up, endpoint muted in Windows, owner in another
    #       room). The OFF rescue cannot cover that: the headset MEASURES on.
    #       So steady-state ON now runs _mic_silence_watchdog(), which acts
    #       ONLY on MEASURED silence from the monolith's own capture loop and
    #       ONLY to move OFF the headset's own microphone. It still never
    #       takes away a microphone the owner chose.
    #
    #   OFF (headset powered down) is a SAFETY CORRECTION, so it is evaluated
    #       on EVERY measured-off sample, not just on the transition - and it
    #       is narrow: the only device it will ever move away from is one that
    #       matches the configured headset name while that headset MEASURES
    #       powered off. It cannot take away a microphone the owner chose, and
    #       it self-heals if a transition is ever missed (a raised exception, a
    #       restart, Windows re-electing the endpoint on a dongle replug).
    #
    # DEAF-SAFETY, stated as rules the code below actually implements:
    #   1. Never act on an UNKNOWN power sample. tick() returns before this is
    #      reached, so `now` here is always a measured bool.
    #   2. Resolve and verify the TARGET before touching anything. Every target
    #      must be an ACTIVE {0.0.1. endpoint present in the enumeration read
    #      on THIS pass - not a remembered id, not a name, not a guess.
    #   3. If the target cannot be verified, DO NOT SWITCH. Say so loudly and
    #      leave him on whatever is working - EXCEPT when what he is on has
    #      itself been measured dead, which is rule 8. "Leave him on what is
    #      working" was written as the safe direction and then applied, once,
    #      to a device this same method had just measured POWERED OFF. That is
    #      not a safe direction; it is the outage with a reassuring log line.
    #   4. If the write fails, the previous device is still the default (the
    #      COM call either takes or it does not; there is no half state), and
    #      the failure is logged naming the device he is still on.
    #   5. A remembered prior microphone is re-verified as ACTIVE before it is
    #      restored, and is never remembered at all if it is the headset's own
    #      endpoint - otherwise "restore what he had" would restore him to the
    #      very device that just powered off.
    #   6. NEVER SAY "listening on X" WITHOUT CHECKING THAT JARVIS FOLLOWS THE
    #      WINDOWS DEFAULT. Moving the default endpoint is all this class can
    #      do; it is not the same event as JARVIS changing what it records
    #      from, and in the owner's live config (PREFERRED_INPUT_DEVICES set)
    #      it is not even correlated with it. Every microphone sentence goes
    #      through capture_claim(), which downgrades the claim to what was
    #      actually verified. Speaking the stronger version is worse than
    #      silence: he trusts it, talks into a boom mic that JARVIS is not
    #      reading, and blames the recogniser.
    #   6. The CONFIGURED FALLBACK gets that same test at the point of use. It
    #      is a name fragment the owner types, the headset's endpoints stay
    #      ACTIVE while it is powered off, and a fallback that resolves to the
    #      headset's own microphone would "switch" to the device already
    #      selected: S_OK, nothing moved, and a spoken rescue every poll.
    #      Refusing it and shouting is the only honest outcome.
    #   8. WHEN THE DEVICE HE IS ON IS THE PROVEN-DEAD ONE, MOVE ANYWAY. Once
    #      the default recording device is confirmed to be the microphone of a
    #      headset measured powered off, "nothing configured resolved" stops
    #      being a reason to hold: every other ACTIVE recording endpoint is
    #      unproven, and unproven beats measured-dead. _last_resort_captures()
    #      supplies them, ranked; the switch is announced as a last resort and
    #      logged as NOT verified to hear anything, because it is not. Only a
    #      machine with no other Active recording endpoint at all stays put.
    #   7. The rescue may be SPACED OUT but never SWITCHED OFF. Some
    #      measured-off polls end in "nothing verified to move to", and reaching
    #      that answer costs a ~300 ms enumeration (measured 2026-09-05: 334.9 /
    #      307.4 / 305.3 / 290.6 ms over 51 endpoints, against 8.0 ms for
    #      default_capture()). Nothing about that state changes on a pass that
    #      does not switch, so re-deciding it every 3 s runs forever: ~10% of a
    #      core and ~1,200 log lines an hour. A verdict is therefore held for
    #      MIC_RESCUE_RETRY_S.
    #      HOW OFTEN that happens shrank when rule 6's last-resort ladder
    #      landed - with any other ACTIVE recording endpoint present the rescue
    #      now switches and the loop ends by itself. What is left is not
    #      hypothetical: a machine whose only Active mic IS the headset's, and
    #      the enumeration-confirmed "not the headset" exit that a nameless
    #      default has to reach the slow way. Both still repeat forever without
    #      this. Three properties make holding safe, and any change here has to
    #      keep all three:
    #        (a) a hold is only ever ENTERED by a pass that already enumerated
    #            and already decided not to switch. It can therefore delay a
    #            rescue by at most MIC_RESCUE_RETRY_S; it can never suppress a
    #            switch the same pass would have made.
    #        (b) every cheap fact that could change the answer - the current
    #            default capture id, the headset fragment, the mic fallback
    #            fragment, the remembered prior mic - is in the key, and a key
    #            that differs cancels the hold on the very next poll, with no
    #            waiting. So does any measured-ON sample.
    #        (c) it expires. A hold is a timer, not a latch: the one thing the
    #            key cannot see is a new endpoint appearing in the enumeration,
    #            which is exactly why it re-checks rather than staying quiet.
    #      A FAILED write is explicitly NOT held - see the comment at that
    #      branch. That path is actively trying to restore his hearing.

    def _follow_mic_state(self, now: bool, was_on: bool | None) -> str | None:
        """The input half of one poll. `now` is always a MEASURED bool."""
        try:
            if now:
                # The headset is ON, so any held "nothing to move to" verdict is
                # about a world that no longer exists. Drop it here rather than
                # in _mic_to_headset, so the steady-state ON path re-arms too:
                # after a power cycle the very next measured-off poll must pay
                # for a full check immediately, not up to MIC_RESCUE_RETRY_S
                # later. Costs nothing - four assignments, no COM.
                self._mic_hold_clear()
                if was_on is True:
                    # NOT `return None`. Until 2026-09-05 that WAS the whole
                    # steady-state ON branch, and it is the defect: the ON side
                    # fired once on the transition and never re-evaluated, so a
                    # headset that is powered on and hearing nothing kept the
                    # capture default for as long as it stayed powered on. The
                    # watchdog below still does not fight a manual choice - it
                    # will only ever move OFF the headset's own microphone, and
                    # only on MEASURED silence. See _mic_silence_watchdog.
                    return self._mic_silence_watchdog()
                return self._mic_to_headset()
            return self._mic_off_headset()
        except Exception as e:
            # An exception here must never leave a half-switch or take the
            # output half down with it. Nothing was written unless
            # set_default_capture() returned, so the device is wherever it was.
            _log(f"microphone follow error ({e}) - the input device was left "
                 f"exactly as it was, which is only GOOD NEWS when that device "
                 f"is not the powered-off headset")
            if now is False:
                # The old wording here - "JARVIS is still listening on whatever
                # it was using" - read as reassurance in precisely the state
                # where it is not. Headset measured off, rescue aborted
                # mid-flight: "whatever it was using" may be the dead headset.
                # Unknown, and now said as unknown, out loud.
                self._say_deaf(
                    "follow-error",
                    f"the '{self.headset}' headset measures powered off and "
                    f"something went wrong while I was moving the microphone "
                    f"off it.",
                    "I have changed nothing, so I cannot tell whether I am "
                    "still set to the headset.")
            return None

    def _is_headset_capture(self, device_id: str | None,
                            rows: list[tuple[str, str, str]]) -> bool:
        """Is `device_id` the CONFIGURED HEADSET's own recording endpoint?

        Two ways in, because one of them can go missing. Normally the id
        matches what find_active_capture() resolves; but if the endpoint has
        stopped being Active (dongle unplugged mid-flight) that lookup returns
        None while the STALE DEFAULT still points at it - so the name recorded
        in the enumeration is checked too. Getting this wrong in the False
        direction is what would leave him deaf, so it errs toward True."""
        if not device_id or not self.headset:
            return False
        hs = find_active_capture(self.headset, rows=rows)
        if hs and device_id == hs[0]:
            return True
        name = endpoint_name(device_id, rows=rows)
        return bool(name and self.headset.lower() in name.lower())

    @staticmethod
    def _active_capture_by_id(device_id: str | None,
                              rows: list[tuple[str, str, str]]
                              ) -> tuple[str, str] | None:
        """(id, name) if `device_id` is STILL an ACTIVE recording endpoint.

        The re-verification behind deaf-safety rule 5. A remembered id is a
        claim about the past; this turns it back into evidence about now."""
        if not device_id:
            return None
        for did, name, state in rows:
            if (did == device_id and did.startswith(CAPTURE_PREFIX)
                    and state.lower() == "active"):
                return did, name
        return None

    # Bounds the COM writes one poll may attempt. The rescue re-runs on
    # every measured-off sample, so an unbounded ladder on a box where every
    # write fails would multiply that cost by the endpoint count. Four is the
    # same cap skills/self_diagnostic.py uses when it probes alternate mics.
    LAST_RESORT_MAX = 4

    def _last_resort_captures(self, rows: list[tuple[str, str, str]]
                              ) -> list[tuple[str, str]]:
        """Every ACTIVE recording endpoint that is NOT the powered-off headset,
        best bet first - the rung below the remembered mic and the configured
        fallback (deaf-safety rule 8).

        WHY THIS EXISTS. Without it the OFF rescue had exactly two candidate
        sources, the remembered id and one configured name fragment, and when
        neither resolved it logged DEAF RISK and left the default on the
        headset microphone it had just MEASURED powered off. Reproduced with
        mocks 2026-09-05 before this method existed - Snowball unplugged,
        fallback "Blue Snowball" therefore unresolvable - set_default_capture
        was never called, while the SAME enumeration held "Microphone (2- HD
        Webcam eMeet C960)" and "Headset Microphone (DualSense Wireless
        Controller)", both Active.

        WHAT THESE CANDIDATES ARE AND ARE NOT. Each is verified to be an ACTIVE
        {0.0.1. endpoint in the enumeration read on THIS pass, and verified not
        to be the headset. NONE is verified to be producing audio, and nothing
        can verify that about a device nothing is recording from - see the
        ranking comment above `mic_liveness_rank`, and capture_signal_state()
        for the one microphone whose signal IS readable (the one the monolith
        is recording from now, which is never a candidate here by definition).
        So the order is a guess, the caller says so out
        loud, and what makes moving anyway correct is not confidence in the
        target: it is that the alternative has already been measured dead."""
        ranked = []
        for idx, (did, name, state) in enumerate(rows):
            if not did.startswith(CAPTURE_PREFIX) or state.lower() != "active":
                continue
            if self._is_headset_capture(did, rows):
                continue               # never "rescue" him onto the dead thing
            ranked.append((mic_liveness_rank(name), idx, did, name))
        ranked.sort()
        return [(d, n) for _rank, _idx, d, n in ranked[:self.LAST_RESORT_MAX]]

    # ── the OFF rescue's bounded retry (deaf-safety rule 7) ─────────────────
    @staticmethod
    def _now() -> float:
        """Monotonic seconds. A method purely so a test can pin the clock
        without patching `time` for the whole process — the retry spacing is
        the thing under test and it must be testable without sleeping."""
        return time.monotonic()

    def _mic_hold_state(self, cur_id: str | None) -> tuple:
        """Everything CHEAP that, if it changed, could change the verdict.

        Deliberately not "everything that could change the verdict" — the
        endpoint enumeration could change too, and that is exactly what this
        key cannot see and why the hold EXPIRES rather than being permanent.
        What it does guarantee is that a change JARVIS can observe for 8 ms
        cancels the hold immediately, with no waiting:

          cur_id            Windows re-elected the default recording device.
          headset           the configured name changed (a live settings edit).
          mic_fallback      the owner just set the thing that was missing.
          _prior_capture    a microphone worth restoring was remembered.
        """
        return (cur_id, self.headset, self.mic_fallback, self._prior_capture)

    def _mic_hold_clear(self) -> None:
        """Forget any held verdict — the next measured-off poll pays in full.

        Called whenever the world has demonstrably moved: a successful capture
        switch, a default that no longer looks like the headset, or ANY
        measured-ON sample (a power cycle must re-arm the rescue instantly, not
        30 seconds later)."""
        self._mic_hold_key = None
        self._mic_hold_next = 0.0
        self._mic_hold_relog = 0.0
        self._mic_hold_skipped = 0

    def _mic_hold_due(self, key: tuple) -> bool:
        """May this poll pay for a full list_render()?

        True whenever the cheap state differs from the held verdict (so a
        changed world is never made to wait) or the retry window has expired.
        Counts the skip otherwise, so the next diagnosis can say how much it
        suppressed."""
        if key != self._mic_hold_key:
            return True
        if self._now() >= self._mic_hold_next:
            return True
        self._mic_hold_skipped += 1
        return False

    def _mic_hold_verbose(self, key: tuple) -> bool:
        """Should THIS pass print its full diagnosis?

        A DIFFERENT fault than the one being held always speaks up — otherwise
        a fault that changed shape 10 seconds into a 300-second quiet window
        would go unreported, which is the silence this module exists to remove.
        The SAME fault repeats itself once every MIC_RESCUE_RELOG_S. Must be
        read BEFORE _mic_hold_keep writes the timers, which is why it is a
        separate call rather than a return value."""
        if key != self._mic_hold_key:
            return True
        return self._now() >= self._mic_hold_relog

    def _mic_hold_note(self) -> str:
        """The rate limit's own confession, appended to a repeat diagnosis.

        The suppressed polls are STATED rather than simply not mentioned. A
        rate limit that hides its own existence is the same silence this module
        was rewritten to remove — the owner has to be able to tell "said once,
        still true" from "said once, then stopped happening"."""
        if not self._mic_hold_skipped:
            return ""
        since = round(self._now() - self._mic_hold_relog + self.MIC_RESCUE_RELOG_S)
        return (f" [still true {since}s later; {self._mic_hold_skipped} poll(s) "
                f"short-circuited since the last message, full re-check every "
                f"{round(self.MIC_RESCUE_RETRY_S)}s]")

    def _mic_hold_keep(self, key: tuple, logged: bool) -> None:
        """Record that `key` has been PROVEN to have nothing to move to.

        Only ever called on a path that already enumerated and already decided
        not to switch, so it can delay a future rescue by at most
        MIC_RESCUE_RETRY_S — it can never suppress a switch this pass would
        have made."""
        now = self._now()
        if key != self._mic_hold_key:
            self._mic_hold_key = key
            self._mic_hold_relog = 0.0     # a new fault always speaks up
            self._mic_hold_skipped = 0
        self._mic_hold_next = now + self.MIC_RESCUE_RETRY_S
        if logged:
            self._mic_hold_relog = now + self.MIC_RESCUE_RELOG_S
            self._mic_hold_skipped = 0

    # ══════════════════════════════════════════════════════════════════════
    # SAYING IT OUT LOUD — the difference between a diagnosis and an alert
    # ══════════════════════════════════════════════════════════════════════
    # _log() is a print(). Up to 2026-09-05 EVERY exit below that leaves the
    # owner on a powered-off microphone used only that: the console. announce()
    # — which reaches proactive_announce -> pending_speech.json and is provably
    # working, since the OUTPUT half's success line is what moved him to
    # speakers he could hear — was wired to SUCCESSES ONLY.
    #
    # Reproduced with mocks 2026-09-05, all six deaf-making exits, five ticks
    # each, headset already ON at start so no ON transition had ever run and
    # _prior_capture was None. Spoken sentences about the microphone: ZERO, in
    # every case. And in five of the six the owner was not merely told nothing
    # — he was told "headset off - audio back to Speakers (Realtek USB2.0
    # Audio)" by the output half, which reads as "the switch worked". The one
    # thing JARVIS said out loud during its own deafness was reassurance.
    #
    # That is the whole outage of 00:48-00:52 wearing a fix's clothes: he asks
    # something, gets silence, and still does not know why.
    #
    # WHAT IS AND IS NOT VERIFIED WHEN THIS SPEAKS. Two things are measured:
    # the headset reports POWERED OFF (void_link, over the dongle's own HID
    # interface) and the default capture endpoint's id is that headset's, in
    # the enumeration read on THIS pass. Deafness itself is NOT measured on
    # THIS path. Proving it needs peak RMS, this is a daemon thread that must
    # never open an input stream to get it — the monolith owns that stream, and
    # PortAudio's close path here is a documented heap-corruption crash — and
    # the headset is powered OFF here, so nothing is recording from it to
    # measure. So the wording is "I may not be able to hear you", never "the
    # microphone is dead", and _deaf_sentence() weakens it further when it
    # cannot even claim that much. Nothing below ever says a microphone WORKS.
    #
    # The ON side is the exception, and only because it does not have to open
    # anything: _mic_silence_watchdog() reads the peak RMS the monolith's OWN
    # capture loop already measured (core.audio_processor, via
    # capture_signal_state), for the one device that loop is recording from.
    # That is why it is allowed to say "I have heard nothing at all" as fact
    # while everything here stays hedged.

    def _speak(self, message: str) -> None:
        """announce(), where a failure to speak can never break the rescue.

        Called only AFTER every device decision is made, so a raise here cannot
        leave a half-switch — but it must not vanish either, or the alert about
        silence would fail silently."""
        try:
            self.announce(message)
        except Exception as e:
            _log(f"could not SPEAK a deaf-risk alert ({e}) — it exists only as "
                 f"the printed line above, which is the failure mode this alert "
                 f"was added to remove")

    @staticmethod
    def _deaf_sentence(fact: str, action: str) -> str:
        """Compose the alert, letting capture_override() cap how strong it gets.

        With nothing outranking the Windows default, that default IS what
        JARVIS records from, so "I may not be able to hear you" is a fair
        reading of the two measured facts. With an override in force it is NOT:
        measured in the owner's LIVE settings 2026-09-05,
        PREFERRED_INPUT_DEVICES = ['Blue Snowball', 'eMeet C960', 'CORSAIR
        VOID'] and _refresh_devices consults that list BEFORE the Windows
        default — so a stuck default says nothing about what he is heard on,
        and asserting deafness from it would be the FOURTH unverified claim of
        the night rather than a fix for the first three.

        Note the third entry in that live list is the headset itself, so an
        override is not automatically good news either. The wording therefore
        claims NEITHER way: it reports the stuck default as fact and says
        plainly that it cannot tell, from here, what that means for hearing."""
        why = capture_override()
        # `fact` is written lower-case so it can be used mid-sentence in the
        # override branch; only the leading letter is lifted here.
        opener = fact[:1].upper() + fact[1:]
        if why is None:
            return f"Sir, I may not be able to hear you. {opener} {action}"
        return (f"Sir, {fact} I am not recording from Windows' default "
                f"microphone anyway — {why} — so I cannot tell from here "
                f"whether I can still hear you. {action}")

    def _say_deaf(self, key: str, fact: str, action: str, confirm: int = 1) -> None:
        """Speak a state that may leave JARVIS unable to HEAR.

        `key` identifies the FAULT, not the sample: the same key arriving again
        is one ongoing problem, not a new one. The OFF branch is evaluated on
        every measured-off poll by design, so without this an unresolvable
        fault would be spoken every poll_s seconds forever.

        `confirm` is how many CONSECUTIVE polls must show the same key before
        it is said. 1 for the faults CONFIRMED against a real enumeration; 2
        for the merely UNVERIFIABLE ones (Windows would not name the default,
        the enumeration came back empty), where one transient COM failure must
        not put a false alarm in his ear — at poll_s=3.0 that costs three
        seconds of delay on a real fault and removes a whole class of
        spurious ones."""
        now = self._now()
        if key != self._deaf_key:
            self._deaf_key = key
            self._deaf_seen = 0
            self._deaf_said_n = 0
            self._deaf_said_at = 0.0
        self._deaf_seen += 1
        if self._deaf_seen < confirm:
            return
        spoken = self._deaf_sentence(fact, action)
        if self._deaf_said_n:
            due = min(self.DEAF_SPEAK_FIRST_S * (2 ** (self._deaf_said_n - 1)),
                      self.DEAF_SPEAK_MAX_S)
            if now - self._deaf_said_at < due:
                return
            spoken = f"Still no change, sir. {spoken}"
        self._deaf_said_at = now
        self._deaf_said_n += 1
        self._speak(spoken)

    def _deaf_clear(self, say: bool = True) -> None:
        """Re-arm the alert, and tell him it is over if he was ever told it began.

        Called ONLY from places that actually CHECKED the microphone situation
        and found it healthy — never merely because a poll went by, which would
        silence a standing fault on the next tick. `say=False` where the caller
        already announces the recovery itself, so the owner never hears the
        same good news twice."""
        if self._deaf_key is None:
            return
        said = self._deaf_said_n
        self._deaf_key = None
        self._deaf_seen = 0
        self._deaf_said_n = 0
        self._deaf_said_at = 0.0
        if said and say:
            # Deliberately not "I can hear you again": nothing here has
            # measured signal, and claiming recovery would be the same
            # unverified confidence as claiming the failure.
            self._speak("Windows' default microphone is off the powered-off "
                        "headset now, sir. I still cannot prove it is picking "
                        "up sound until I hear you.")

    def _mic_to_headset(self) -> str | None:
        """Headset just powered ON - move the WINDOWS DEFAULT recording device
        to its microphone.

        Says "move the Windows default", not "listen on it", because those are
        two different facts and only the first one is in this method's power.
        Whether JARVIS's own capture loop follows that default is decided by
        MICROPHONE_INDEX / PREFERRED_INPUT_DEVICES in the monolith, and in the
        owner's CURRENT configuration it does NOT - see capture_override().
        The write below still happens (it is correct for the machine, and it is
        what makes the feature work the moment the override is cleared); only
        the SPOKEN claim is gated, by capture_claim()."""
        rows = list_render()
        if not rows:
            _log("headset measures ON but the audio device list came back EMPTY "
                 "- the microphone was NOT moved (enumeration failed)")
            return None
        hs = find_active_capture(self.headset, rows=rows)
        if not hs:
            _log(f"'{self.headset}' measures POWERED ON but no ACTIVE RECORDING "
                 f"endpoint matches that name - the microphone was NOT moved. "
                 f"Check the name against `python -m audio.audio_switch --list-mics`.")
            return None
        cur = default_capture_id()
        if cur == hs[0]:
            # Nothing is announced on this path - so if a deaf-risk alert is
            # standing, THIS is the poll that resolves it (he powered the
            # headset back on while its microphone was already the default) and
            # he would otherwise never be told it ended. say=True on purpose.
            self._deaf_clear()
            return None                    # already listening there
        # Remember where to come back to - but NEVER remember the headset's own
        # microphone. If we did, the next power-down would "restore" him to the
        # device that had just powered off, which is precisely tonight's outage
        # wearing the costume of a fix.
        if cur and not self._is_headset_capture(cur, rows):
            self._prior_capture = cur
        if not set_default_capture(hs[0]):
            still = endpoint_name(cur, rows=rows) or "the device Windows had"
            _log(f"headset on, but moving the microphone to {hs[1]} FAILED or could "
                 f"not be VERIFIED - as far as I can prove, the default recording "
                 f"device is still {still}")
            return None
        # NOT `f"listening on {hs[1]}"`. set_default_capture moved the WINDOWS
        # default; whether JARVIS follows it is a separate, checkable fact and
        # capture_claim is the only thing allowed to assert it. See
        # capture_override() for why the answer is currently "no".
        self._deaf_clear(say=False)    # capture_claim below IS the recovery line
        self.announce(capture_claim(hs[1], "headset on"))
        return "mic_to_headset"

    # ══════════════════════════════════════════════════════════════════════
    # STEADY-STATE ON - the only place in this module that measures HEARING
    # ══════════════════════════════════════════════════════════════════════
    # The _say_deaf keys this watchdog owns. Kept apart from the OFF rescue's
    # keys so it can recognise ITS OWN standing alert and speak the right
    # recovery for it, instead of clearing a fault it never diagnosed.
    _SILENT_KEYS = ("silent-capture-overridden", "silent-default-unreadable",
                    "silent-enumeration-empty", "silent-nowhere-to-go",
                    "silent-every-candidate-refused")

    def _mic_silence_watchdog(self) -> str | None:
        """Headset measures ON and we are already past the transition. Check
        that its microphone is actually PRODUCING AUDIO, and move off it if it
        provably is not.

        THE DEFECT THIS REPLACES. `if was_on is True: return None` was the
        entire steady-state ON branch. Power the headset on with the boom mic
        flipped up - its own hardware mute, and its resting position - and
        void_link reports the link ON, the transition fires, the capture
        default moves to a microphone that reads exactly 0.0000 RMS, and then
        nothing looks again. The OFF rescue cannot fire (the headset MEASURES
        on). He is deaf until he physically powers the headset down. Same shape
        if the endpoint is muted in Windows, or he walks out of range.

        WHAT IS MEASURED BEFORE ANYTHING MOVES, in this order, cheapest first:
          1. SIGNAL. capture_signal_state() must say "silent" - the monolith's
             capture loop is demonstrably running AND has seen nothing above
             core.audio_processor's audible floor for mic_silent_s. Anything
             else, including every kind of "I cannot tell", HOLDS.
          2. THAT THE SIGNAL IS ABOUT THIS DEVICE. The default recording
             endpoint must BE the headset's own microphone (read cheaply by
             name, then confirmed against this pass's enumeration), and
             capture_override() must say nothing outranks the Windows default.
             With an override in force the silence is about some other device,
             so the headset's microphone is UNKNOWN and rule 3 forbids acting.
          3. A TARGET, verified ACTIVE in this pass's own enumeration and
             verified not to be the headset - the same ladder the OFF rescue
             uses, via the same helpers.

        WHAT IT WILL NOT DO. It will not move him off a microphone he chose:
        the only device it ever moves away from is the configured headset's
        own. When the silent device is something else, this returns None
        without a word - record_speech's own silent-mic alert already SPEAKS
        about that at MIC_SILENT_WARN_SECONDS (30 s), i.e. before this
        watchdog's 60 s, and two voices on one fault is worse than one.

        COST. Steps 2 and 3 (8 ms, then ~300 ms) are reached only while step 1
        says he is deaf RIGHT NOW, so there is no hold/backoff here on purpose
        - that is the same trade _mic_off_headset makes on its failed-write
        path, for the same reason: spending a core's 10% to keep trying to
        restore his hearing is the right trade."""
        state, why = capture_signal_state(self.mic_silent_s)
        if state != "silent":
            if state == "audible" and self._deaf_key in self._SILENT_KEYS:
                # MEASURED recovery, and this is the one place in the module
                # entitled to say so out loud: a chunk crossed the audible
                # floor, which is signal, not selection. _deaf_clear's own
                # sentence is written for the OFF rescue ("off the powered-off
                # headset now") and would be wrong here, so it is suppressed
                # and this says the true thing instead.
                said = self._deaf_said_n
                self._deaf_clear(say=False)
                if said:
                    self._speak(f"My microphone is picking up sound again, sir "
                                f"- {why}.")
            return None

        # MEASURED SILENT. Everything below is about whether that fact is about
        # a device this watcher is allowed to move.
        cur = default_capture()            # ~8 ms, and only ever while deaf
        if cur is None:
            self._say_deaf(
                "silent-default-unreadable",
                f"{why} - and Windows will not tell me which microphone is "
                f"selected, so I cannot check whether it is the "
                f"'{self.headset}' headset's.",
                "I have changed nothing.",
                confirm=2)
            return None
        cur_id, cur_name = cur
        if not (cur_name and self.headset
                and self.headset.lower() in cur_name.lower()):
            # Deaf, but not on the headset's microphone. Not this watchdog's
            # device to move, and not its alert to speak - see the docstring.
            return None

        override = capture_override()
        if override is not None:
            # UNKNOWN, so HOLD. The signal measures what the MONOLITH records
            # from; with an override in force that is not the Windows default,
            # so a silent capture says nothing about the headset's microphone
            # and moving the default would not change what JARVIS hears. Note
            # _deaf_sentence() reads capture_override() itself and downgrades
            # the wording accordingly, so this fact is never spoken as a claim
            # of deafness.
            self._say_deaf(
                "silent-capture-overridden",
                f"{why}, and Windows' microphone is the '{self.headset}' "
                f"headset's own.",
                "Clear PREFERRED_INPUT_DEVICES in your settings (and leave "
                "MICROPHONE_INDEX unset) if you want me to follow Windows' "
                "microphone, and I will be able to move it for you.",
                confirm=2)
            return None

        rows = list_render()               # ~300 ms, only while proven deaf
        if not rows:
            self._say_deaf(
                "silent-enumeration-empty",
                f"{why}, and the audio device list came back empty, so I could "
                f"not verify anything.",
                "I have changed nothing.",
                confirm=2)
            return None
        if not self._is_headset_capture(cur_id, rows):
            return None    # confirmed against the enumeration: not the headset

        # PROVEN, all three: JARVIS follows the Windows default, that default
        # IS this headset's microphone, and that microphone has produced
        # literal digital silence while the capture loop was running. This is
        # the one branch in the module with SIGNAL behind it, so it is the one
        # branch allowed to say a device is not hearing.
        ladder = self._silence_ladder(rows)
        if not ladder:
            _log(f"DEAF: the '{self.headset}' headset is powered ON, Windows' "
                 f"default recording device is its microphone, and {why} - but "
                 f"this machine has NO other ACTIVE recording endpoint to move "
                 f"to. Plug a microphone in; `python -m audio.audio_switch "
                 f"--list-mics` shows everything Windows can see.")
            self._say_deaf(
                "silent-nowhere-to-go",
                f"{why}. Windows' microphone is the '{self.headset}' headset's "
                f"own and the headset is powered on, so it is on but not "
                f"hearing - check the boom mic is not flipped up or muted.",
                "This machine has no other recording device switched on, so "
                "there is nowhere for me to move. Plug a microphone in.")
            return None

        for cand in ladder:
            _log(f"SILENT MICROPHONE: {why} VERIFIED: Windows' default recording "
                 f"device is the '{self.headset}' headset's own microphone in "
                 f"this pass's own enumeration, nothing overrides the Windows "
                 f"default, and '{cand[1]}' is an ACTIVE recording endpoint that "
                 f"is not the headset. NOT VERIFIED: that '{cand[1]}' hears "
                 f"anything - what is measured is that the device it is moving "
                 f"OFF does not.")
            if set_default_capture(cand[0]):
                self._deaf_clear(say=False)   # the announce below IS the news
                self.announce(
                    capture_claim(
                        cand[1],
                        f"the '{self.headset}' headset is powered on but I have "
                        f"heard nothing through its microphone")
                    + " - I can't confirm the new one hears you either, sir")
                self._prior_capture = None
                self._mic_hold_clear()
                return "mic_silent_rescue"
            _log(f"the move off the silent headset microphone to {cand[1]} FAILED "
                 f"or could not be VERIFIED - trying the next candidate")
        _log(f"DEAF: all {len(ladder)} recording device(s) I could have moved to "
             f"refused the switch or could not be verified to have taken it - as "
             f"far as I can prove the default recording device is still the "
             f"silent '{self.headset}' microphone.")
        self._say_deaf(
            "silent-every-candidate-refused",
            f"{why}, and every one of the {len(ladder)} microphones I could "
            f"have moved to refused the switch.",
            "As far as I can prove I am still set to the headset. Please "
            "change the microphone by hand.")
        return None

    def _silence_ladder(self, rows: list[tuple[str, str, str]]
                        ) -> list[tuple[str, str]]:
        """Where to move a MEASURED-silent headset microphone, best bet first.

        The SAME policy _mic_off_headset applies - remembered mic, then the
        configured fallback, then any other ACTIVE recording endpoint ranked by
        mic_liveness_rank - expressed once, over the same helpers, because the
        two rescues are asking the identical question about the identical
        device and only the trigger differs.

        THIS REPO'S MOST EXPENSIVE RECURRING BUG IS A RULE FIXED IN ONE COPY
        WHILE ANOTHER ROTS, so the agreement is pinned by a test rather than by
        this comment: MicSilentOnHeadsetTests::
        test_the_two_rescues_choose_the_same_target drives both paths over one
        set of rows and fails the build the moment their first choice
        diverges."""
        out: list[tuple[str, str]] = []
        prior = self._active_capture_by_id(self._prior_capture, rows)
        if prior and not self._is_headset_capture(prior[0], rows):
            out.append(prior)             # rule 5: re-verified, not the headset
        if self.mic_fallback:
            # quiet=True: the OFF rescue owns the loud fallback diagnosis on its
            # own timer, and this path can run every poll while he is deaf.
            fb = resolve_mic_fallback(self.mic_fallback, rows=rows, quiet=True)
            if fb and not self._is_headset_capture(fb[0], rows):
                out.append(fb)            # rule 6: never the headset itself
        for cand in self._last_resort_captures(rows):   # rule 8, already ranked
            if cand not in out:
                out.append(cand)
        return out

    def _mic_off_headset(self) -> str | None:
        """Headset measures OFF. Move the microphone off it - and ONLY off it."""
        # FAST PATH FIRST, and it is not premature optimisation. This method is
        # evaluated on EVERY measured-off sample - that is what makes the
        # rescue self-healing - and measured 2026-09-05 a full list_render()
        # costs 286 ms on this machine against 8.5 ms to ask about the one
        # default device. Enumerating unconditionally would spend ~10% of a
        # core, forever, to keep answering "no, the microphone is fine".
        cur = default_capture()
        if cur is None:
            _log("headset measures OFF but Windows would not name the default "
                 "recording device - leaving the microphone alone (unknown is "
                 "never a reason to switch)")
            # confirm=2: one transient COM failure must not put a false alarm in
            # his ear. At poll_s=3.0 a real fault is still spoken 3 s later.
            self._say_deaf(
                "default-unreadable",
                f"the '{self.headset}' headset measures powered off and Windows "
                f"would not tell me which microphone is selected.",
                "I have changed nothing. If I stop answering you, that is where "
                "to look.",
                confirm=2)
            return None
        cur_id, cur_name = cur
        if cur_name and self.headset and self.headset.lower() not in cur_name.lower():
            self._mic_hold_clear()
            self._deaf_clear()      # healthy, and CHECKED - safe to re-arm
            return None    # the default is not the dead headset - nothing to fix

        # ── SECOND FAST PATH, and it is the one the first cannot cover ───────
        # The test above answers "is the default someone else's microphone?"
        # and short-circuits when it is. In the state this whole rescue EXISTS
        # for it cannot fire: the default IS the dead headset, so the name DOES
        # contain the headset fragment, and control falls through to the 300 ms
        # enumeration below on every single poll - forever, because a pass with
        # nothing verified to move to changes nothing about the state it keyed
        # off of. MEASURED with mocks 2026-09-05 against the working tree AS IT
        # THEN STOOD (before rule 6's last-resort ladder), 20 measured-off polls
        # in the SHIPPED default configuration - AUDIO_AUTOSWITCH_MIC just
        # switched on, AUDIO_AUTOSWITCH_MIC_FALLBACK still "": 20 list_render()
        # calls and 20 log lines, i.e. ~10% of a core and 1,200 lines an hour
        # for as long as the headset stayed off.
        #
        # BE PRECISE ABOUT WHAT THE LADDER CHANGED. It made that loop RARER, not
        # gone: with some other ACTIVE recording endpoint present the rescue now
        # switches to it and the state moves on by itself. It still runs forever
        # on a machine whose only Active recording endpoint IS the headset's own
        # microphone, and on the enumeration-confirmed "not the headset" exit
        # below, which a nameless default can only reach the slow way.
        #
        # So a PROVEN-stuck verdict is trusted for MIC_RESCUE_RETRY_S. Read the
        # deaf-safety argument in rule 7 below before loosening this: the hold
        # is only ever entered by a pass that already looked and already
        # decided not to switch, and every cheap fact that could change the
        # answer is in the key, which cancels the hold the instant it moves.
        key = self._mic_hold_state(cur_id)
        if not self._mic_hold_due(key):
            return None
        verbose = self._mic_hold_verbose(key)

        # Either the name matches the headset, or Windows returned no name at
        # all. Both need the full picture, so pay for the enumeration now. A
        # nameless device falls through on purpose: skipping the rescue because
        # we could not read a name would be assuming the safe case.
        rows = list_render()
        if not rows:
            if verbose:
                _log("headset measures OFF but the audio device list came back EMPTY "
                     "- nothing could be verified, so the microphone is left alone"
                     + self._mic_hold_note())
            self._mic_hold_keep(key, verbose)
            self._say_deaf(
                "enumeration-empty",
                f"the '{self.headset}' headset measures powered off and I could "
                f"not list the audio devices at all.",
                "I have changed nothing, so I cannot tell whether I am still "
                "set to its microphone.",
                confirm=2)
            return None
        if not self._is_headset_capture(cur_id, rows):
            # Confirmed against the enumeration: not the headset. Held too, so a
            # nameless default (which cannot use the first fast path) does not
            # re-enumerate every 3 s just to reach this same silent answer.
            self._mic_hold_keep(key, logged=False)
            self._deaf_clear()      # healthy, and CONFIRMED against `rows`
            return None

        target = None
        prior = self._active_capture_by_id(self._prior_capture, rows)
        if prior and not self._is_headset_capture(prior[0], rows):
            target = prior
        elif self._prior_capture and verbose:
            _log(f"the microphone I remembered ({self._prior_capture}) is no longer "
                 f"an active recording device - falling back to "
                 f"'{self.mic_fallback or '(nothing configured)'}'")
        why_fb = ""                    # set when the fallback resolved but is unusable
        if target is None and self.mic_fallback:
            # quiet on a repeat only - `verbose` is True on the first sighting
            # of this fault and once every MIC_RESCUE_RELOG_S thereafter.
            target = resolve_mic_fallback(self.mic_fallback, rows=rows,
                                          quiet=not verbose)  # logs why
            # DEAF-SAFETY RULE 6. The fallback must not BE the dead headset.
            #
            # Everything above this line is about finding a target; this is the
            # only thing that checks the target is not the very device we are
            # running away from. `_active_capture_by_id` gave the REMEMBERED
            # path that check (rule 5) and `_mic_to_headset` refuses to
            # remember the headset at all, but the FALLBACK arrived here
            # unchecked, and it is a name fragment the owner types by hand.
            #
            # It resolves. That is the problem. MEASURED 2026-09-05: with the
            # VOID ELITE powered OFF, Windows still lists
            # "Headset Microphone (CORSAIR VOID ELITE Wireless Gaming Headset)"
            # as Active (see the module docstring — endpoint state carries no
            # power information for this dongle), so a fallback fragment of
            # "CORSAIR", "VOID ELITE" or "Headset Microphone" resolves to the
            # headset's own microphone. It is a {0.0.1. id, so
            # set_default_capture accepts it; SetDefaultEndpoint on the device
            # that is ALREADY the default returns S_OK; so the write
            # "succeeds", `self._prior_capture` is cleared, and he is TOLD
            # "headset off - listening on Headset Microphone (CORSAIR ...)".
            # The default never moved. He is deaf and has just been informed he
            # was rescued.
            #
            # And it does not happen once. The OFF branch is evaluated on every
            # measured-off sample by design, and the state it keys off of is
            # never changed by the "successful" write — so it repeats every
            # poll_s seconds, forever. Reproduced with mocks 2026-09-05: three
            # ticks, three identical spoken rescues, zero movement.
            #
            # A misconfigured fallback is not a hypothetical here. This module's
            # entire history is a wrong device sitting in a fallback slot: the
            # playback fallback really was set to "Blue Snowball", a MICROPHONE
            # (fixed 2026-09-04), and the mic fallback slot sits directly under
            # a setting whose value is the headset's own name.
            #
            # Refuse and fall through to the DEAF RISK message below. Doing
            # nothing while SAYING so is strictly better than doing nothing
            # while claiming to have fixed it: the log then names a problem he
            # can act on, and `use_desk_mic` / a corrected setting still work.
            if target is not None and self._is_headset_capture(target[0], rows):
                why_fb = (f"the configured microphone fallback "
                          f"'{self.mic_fallback}' resolves to '{target[1]}', which "
                          f"IS the powered-off headset's own microphone")
                if verbose:
                    _log(f"MIC FALLBACK REFUSED: {why_fb} - switching to it would "
                         f"return success without moving anything and leave JARVIS "
                         f"deaf while announcing a rescue. Point "
                         f"AUDIO_AUTOSWITCH_MIC_FALLBACK at a recording device that "
                         f"is NOT the headset (`python -m audio.audio_switch "
                         f"--list-mics`).")
                target = None

        # ── DEAF-SAFETY RULE 8 ───────────────────────────────────
        # "NOTHING CONFIGURED RESOLVED" IS NOT A REASON TO STAY HERE.
        #
        # Everywhere else in this module an unverifiable target means HOLD, and
        # that is right, because holding leaves him on a device that is merely
        # unproven. It is not right on this line. Control only gets here when
        # the default recording device has been CONFIRMED, against this pass's
        # own enumeration, to be the microphone of a headset this code has just
        # MEASURED powered off. That device is not unproven - it is the one
        # thing in the room that has been proven dead: peak RMS 0.0000, twice,
        # thirty seconds apart, 2026-09-05 00:50:40. Holding it is not caution,
        # it is the outage this whole feature was written to end.
        #
        # So the ladder gets one more rung: any other ACTIVE recording
        # endpoint. It is a WEAKER claim than the rungs above and is treated as
        # one - the remembered mic is one he was demonstrably using, the
        # configured fallback is one he named, a last resort is neither. It is
        # announced as a last resort, logged as UNVERIFIED, and never described
        # as working. "Selected" is not "producing audio", and the log line
        # says which of the two this is.
        why = ""
        if why_fb:
            why = why_fb
        elif self.mic_fallback:
            why = (f"the configured microphone fallback "
                   f"'{self.mic_fallback}' did not resolve")
        else:
            why = ("no AUDIO_AUTOSWITCH_MIC_FALLBACK is configured and no "
                   "earlier microphone was remembered")

        # A target the OWNER chose - the mic he was on, or the fragment he
        # typed - keeps the single attempt it always had. Cascading past a
        # FAILED write of a chosen target would be a second change to a second
        # behaviour, and this one is scoped to the state where no chosen target
        # exists at all. When that write does not take, the pass logs and
        # returns unheld, so the next measured-off poll tries again from the
        # top - and reaches this rung the moment the chosen target stops
        # resolving.
        if target is not None:
            if not set_default_capture(target[0]):
                # DELIBERATELY NOT HELD, and this is the one asymmetry that
                # matters. Every other exit above is a pass that looked and
                # decided not to switch; this is a pass that TRIED and could
                # not show it worked. He may be deaf RIGHT NOW and a retry is
                # the thing that un-deafens him, so this path keeps paying the
                # full 300 ms every poll and keeps saying so.
                #
                # Deliberately hedged. set_default_capture returns False for
                # THREE different worlds - the COM write failed, the write
                # returned success and the device did not move, or the
                # read-back was unreadable - and it logs which one on its own
                # line. Asserting "the default is STILL the headset" would be
                # the same unverified confidence in the opposite direction.
                _log(f"headset off, but moving the microphone to {target[1]} FAILED "
                     f"or could not be VERIFIED - I cannot show that the default "
                     f"recording device moved off the powered-off headset, so "
                     f"JARVIS may be deaf until this is fixed")
                # The log line above is not an alert. Only _say_deaf reaches a
                # channel he is actually using - and this pass is not held, so
                # without the voice's own backoff this would be the one place
                # that could talk every poll.
                self._say_deaf(
                    "write-failed",
                    f"the '{self.headset}' headset measures powered off and I "
                    f"could not verify that I moved the microphone off it.",
                    f"I tried to switch to {target[1]} and cannot show that it "
                    f"took. Please change the microphone by hand, or turn the "
                    f"headset back on.")
                return None
            # Same rule on the way back. The RESCUE ran unconditionally - the
            # dead headset is off the Windows default either way - only the
            # spoken claim is conditioned on JARVIS actually following it.
            # say=False: capture_claim() is the recovery message and it is the
            # honest one - it reports what actually moved, and nothing more.
            self._deaf_clear(say=False)
            self.announce(capture_claim(target[1], "headset off"))
            self._prior_capture = None
            self._mic_hold_clear()   # the world moved - never hold a stale verdict
            return "mic_away"

        # Nothing he chose resolved. THE LAST RESORT, ranked, and cascading
        # within itself: these are all guesses of equal standing, so a guess
        # that will not take is a reason to try the next guess rather than to
        # stay on the device already measured dead.
        ladder = self._last_resort_captures(rows)

        if not ladder:
            # Genuinely nothing: not one ACTIVE recording endpoint on this
            # machine that is not the powered-off headset. This is the ONLY
            # state in which staying put is the honest answer, and it is a
            # hardware statement rather than a configuration one.
            if verbose:
                _log(f"DEAF RISK: the '{self.headset}' headset measures POWERED OFF "
                     f"and the default recording device is still its microphone; "
                     f"{why}, and this machine has NO other ACTIVE recording "
                     f"endpoint at all - there is nothing anywhere to move to. Plug "
                     f"a microphone in; `python -m audio.audio_switch --list-mics` "
                     f"shows everything Windows can see." + self._mic_hold_note())
            self._mic_hold_keep(key, verbose)
            # CONFIRMED deaf, so confirm=1 - this is not a transient read
            # failure, it is a fully enumerated machine with nowhere to go.
            self._say_deaf(
                "no-recording-device",
                f"the '{self.headset}' headset measures powered off, Windows' "
                f"default microphone is still its own microphone, and this "
                f"machine has no other recording device switched on at all.",
                "There is nowhere for me to move it, so I have changed nothing. "
                "Plug a microphone in, or turn the headset back on.")
            return None

        for cand in ladder:
            if verbose:
                _log(f"LAST RESORT: {why}, so I am moving the microphone to "
                     f"'{cand[1]}' rather than leaving it on the powered-off "
                     f"headset. VERIFIED: that device is an ACTIVE recording "
                     f"endpoint in this pass's own enumeration and is not the "
                     f"headset. NOT VERIFIED: that it hears anything - nothing "
                     f"is recording from it yet, and this daemon must never open "
                     f"an input stream of its own to find out (the monolith owns "
                     f"that stream; a second one from this thread crashes the "
                     f"process). If it turns out to be silent too, the thing "
                     f"that will say so is record_speech's own silent-mic alert "
                     f"at MIC_SILENT_WARN_SECONDS, not this daemon - the ON-side "
                     f"signal watchdog only ever moves off the HEADSET's own "
                     f"microphone. What IS measured is that the device it "
                     f"is moving OFF is dead. Set AUDIO_AUTOSWITCH_MIC_FALLBACK "
                     f"from `python -m audio.audio_switch --list-mics` to make this "
                     f"a deliberate choice instead of my guess.")
            if set_default_capture(cand[0]):
                # Hedged TWICE, and both hedges are load-bearing. capture_claim
                # covers whether JARVIS follows the Windows default at all; the
                # suffix covers this device, which is verified present and
                # verified not-the-headset and is verified nothing else. He is
                # told he was moved and told what that does and does not mean.
                self.announce(capture_claim(cand[1], "headset off")
                              + " - a last resort, sir; I can't confirm it "
                                "hears you")
                self._deaf_clear(say=False)   # the announce above IS the recovery
                self._prior_capture = None
                self._mic_hold_clear()   # the world moved - never hold a stale verdict
                return "mic_last_resort"
            if verbose:
                _log(f"the last-resort move to {cand[1]} FAILED or could not be "
                     f"VERIFIED - trying the next candidate")
        # Every rung refused. NOT held, for the reason the chosen-target failure
        # above is not held either: this pass TRIED and could not show it
        # worked, so the next poll pays in full and tries again rather than
        # sitting on a proven-dead microphone for MIC_RESCUE_RETRY_S.
        _log(f"DEAF RISK: all {len(ladder)} recording device(s) I could have moved "
             f"to refused the switch or could not be verified to have taken it - as "
             f"far as I can prove, the default is still the powered-off "
             f"'{self.headset}' microphone and JARVIS may be deaf.")
        self._say_deaf(
            "every-candidate-refused",
            f"the '{self.headset}' headset measures powered off and every one of "
            f"the {len(ladder)} microphones I could have moved to refused the "
            f"switch.",
            "As far as I can prove I am still set to the headset. Please change "
            "the microphone by hand, or turn the headset back on.")
        return None


# ── standalone CLI ───────────────────────────────────────────────────────────
def _main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Auto-switch default audio on headset power.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--list-mics", action="store_true",
                    help="show RECORDING endpoints + states (the input half)")
    ap.add_argument("--test-mic", metavar="FRAGMENT", default=None,
                    help="REALLY switch the default RECORDING device to the "
                         "first ACTIVE mic matching FRAGMENT, read it back, then "
                         "restore. This is the only thing in the tree that "
                         "proves a capture default actually MOVES \u2014 see "
                         "set_default_endpoint's honest-limits note. It changes "
                         "a live device, so it is a deliberate human command and "
                         "is never called by the watcher.")
    ap.add_argument("--state", action="store_true",
                    help="what the power detector actually says (three-valued)")
    ap.add_argument("--test", action="store_true", help="switch to headset then restore")
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--headset", default="CORSAIR VOID ELITE")
    # NOTE: this default is the owner's real speakers. data/user_settings.json
    # has carried "Blue Snowball" — a MICROPHONE — in AUDIO_AUTOSWITCH_FALLBACK,
    # which can never match a render endpoint. `--state` prints that verdict.
    ap.add_argument("--fallback", default="Realtek USB2.0 Audio")
    # No default here on purpose. The PLAYBACK fallback above cannot serve the
    # input direction: measured 2026-09-05, "Realtek USB2.0 Audio" has ZERO
    # Active recording endpoints on this machine (Line In and Microphone both
    # Unplugged, Internal AUX Jack Disabled), so borrowing it would resolve to
    # nothing and no-op in silence.
    ap.add_argument("--mic-fallback", default="",
                    help="desk-microphone name fragment for the input half")
    ap.add_argument("--follow-mic", action="store_true",
                    help="with --daemon, also move the default RECORDING device")
    ap.add_argument("--mic-silent-s", type=float, default=60.0,
                    help="seconds of MEASURED digital silence from a powered-ON "
                         "headset's microphone before the watcher moves off it "
                         "(0 disables the check and restores the pre-2026-09-05 "
                         "fire-once-on-transition behaviour)")
    args = ap.parse_args(argv)

    if _HAS_COM:
        try:
            comtypes.CoInitialize()
        except Exception:
            pass

    if args.list:
        for did, name, state in list_render():
            mark = " <== DEFAULT" if did == default_render_id() else ""
            print(f"  {state:11} {name}{mark}")
        return 0

    if args.list_mics:
        cur = default_capture_id()
        rows = [r for r in list_render() if r[0].startswith(CAPTURE_PREFIX)]
        if not rows:
            print("  no recording endpoints enumerated at all \u2014 COM/pycaw failed")
            return 1
        for did, name, state in rows:
            mark = " <== DEFAULT" if did == cur else ""
            print(f"  {state:11} {name or '(unnamed)'}{mark}")
        return 0

    if args.test_mic is not None:
        # Deliberately loud and deliberately reversible. Run this with the
        # owner watching, on the DESK mic (never the headset), to close the one
        # gap the automated tests cannot: that SetDefaultEndpoint really does
        # MOVE a capture default rather than merely accepting the id.
        rows = list_render()
        hit = find_active_capture(args.test_mic, rows=rows)
        if not hit:
            prob = mic_fallback_problem(args.test_mic, rows=rows)
            print(f"  cannot test: {prob or 'no active recording endpoint matched'}")
            return 1
        orig = default_capture_id()
        print(f"  current default mic : {endpoint_name(orig, rows=rows)}  ({orig})")
        print(f"  switching to        : {hit[1]}  ({hit[0]})")
        ok = set_default_capture(hit[0])
        moved = default_capture_id()
        print(f"  read back           : {endpoint_name(moved, rows=rows)}  ({moved})")
        print(f"  set returned ok={ok}, MOVED={moved == hit[0]}")
        if orig:
            print(f"  restoring           : {endpoint_name(orig, rows=rows)}")
            set_default_capture(orig)
            back = default_capture_id()
            print(f"  restored            : {endpoint_name(back, rows=rows)}  "
                  f"(exact={back == orig})")
            return 0 if (ok and moved == hit[0] and back == orig) else 1
        print("  NOT RESTORED \u2014 there was no readable original to restore to.")
        return 1

    if args.state:
        powered = headset_powered(args.headset)
        word = {True: "POWERED ON", False: "OFF", None: "UNKNOWN (watcher holds)"}[powered]
        via = "audio.void_link (HID)" if looks_like_corsair_void(args.headset) \
            else "endpoint state (NOT verified for this hardware)"
        print(f"  headset   '{args.headset}': {word}")
        print(f"  detector  : {via}")
        batt = void_battery_pct(args.headset)
        print(f"  battery   : {batt if batt is not None else 'unknown'}")
        ep = find_active(args.headset)
        print(f"  endpoint  : {ep[1] if ep else 'no ACTIVE playback endpoint'}")
        print(f"              (endpoint state does NOT imply power — see find_active)")
        prob = fallback_problem(args.fallback)
        if prob:
            print(f"  fallback  : UNUSABLE — {prob}")
        else:
            fb = find_active(args.fallback)
            print(f"  fallback  : {fb[1] if fb else '(none configured)'}")
        # ---- the input half -------------------------------------------------
        rows = list_render()
        mep = find_active_capture(args.headset, rows=rows)
        curmic = default_capture_id()
        print(f"  mic (hs)  : {mep[1] if mep else 'no ACTIVE recording endpoint'}")
        print(f"  mic (now) : {endpoint_name(curmic, rows=rows) or 'unreadable'}")
        if args.mic_fallback:
            mprob = mic_fallback_problem(args.mic_fallback, rows=rows)
            print(f"  mic fb    : {'UNUSABLE — ' + mprob if mprob else find_active_capture(args.mic_fallback, rows=rows)[1]}")
        else:
            print(f"  mic fb    : (none configured — the input half has nothing "
                  f"to fall back to)")
        return 0

    if args.test:
        powered = headset_powered(args.headset)
        if powered is False:
            print(f"headset '{args.headset}' measures OFF — power it on first.")
            return 1
        if powered is None:
            print(f"headset '{args.headset}' power state is UNKNOWN; testing the "
                  f"endpoint switch anyway.")
        hs = find_active(args.headset)
        if not hs:
            print(f"headset '{args.headset}' has no ACTIVE playback endpoint.")
            return 1
        orig = default_render_id()
        print(f"current default: {orig}")
        print(f"switching to headset: {hs[1]} ({hs[0]})")
        ok = set_default_render(hs[0])
        print(f"  set -> {default_render_id()}  (ok={ok})")
        time.sleep(1.0)
        print(f"restoring original: {orig}")
        set_default_render(orig)
        print(f"  restored -> {default_render_id()}")
        return 0 if ok and default_render_id() == orig else 1

    if args.daemon:
        sw = AudioAutoSwitch(args.headset, args.fallback,
                             mic_fallback=args.mic_fallback,
                             follow_mic=args.follow_mic,
                             mic_silent_s=args.mic_silent_s)
        sw.start()
        print("[audio-switch] daemon running — Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sw.stop()
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
