"""Auto-switch the Windows DEFAULT audio device when a wireless headset powers
on/off — without plugging/unplugging the dongle.

    headset ON  -> default = the headset   (remember the prior default)
    headset OFF -> default = the prior default, else a configured fallback

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
    python -m audio.audio_switch --list      # show render devices + states
    python -m audio.audio_switch --state     # what the detector actually says
    python -m audio.audio_switch --test      # switch to headset + restore (proves it)
    python -m audio.audio_switch --daemon     # run the watcher in the foreground
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


def set_default_render(device_id: str) -> bool:
    """Make `device_id` (an MMDevice id string) the default render endpoint for
    all three roles. Returns True on success."""
    if not _HAS_COM or not device_id:
        return False
    try:
        pc = CoCreateInstance(_CLSID_PolicyConfigVistaClient, _IPolicyConfigVista, CLSCTX_ALL)
        for role in (0, 1, 2):          # console / multimedia / communications
            pc.SetDefaultEndpoint(device_id, role)
        return True
    except Exception as e:
        _log(f"set_default_render failed: {e}")
        return False


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
        if render_only and not did.startswith("{0.0.0."):
            continue
        if state.lower() == "active" and frag in name.lower():
            return did, name
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

    def __init__(self, headset: str, fallback: str = "", poll_s: float = 3.0,
                 announce=None):
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
        if now and not was_on:
            # Covers False->True and the startup None->True sync. Idempotent:
            # _switch_to_headset() no-ops when the headset is already default.
            return self._switch_to_headset()
        if was_on is True and not now:
            return self._switch_away()
        # Deliberately no branch for None->False: at startup with the headset
        # already off, the owner's chosen default is left exactly alone.
        return None

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


# ── standalone CLI ───────────────────────────────────────────────────────────
def _main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Auto-switch default audio on headset power.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--state", action="store_true",
                    help="what the power detector actually says (three-valued)")
    ap.add_argument("--test", action="store_true", help="switch to headset then restore")
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--headset", default="CORSAIR VOID ELITE")
    # NOTE: this default is the owner's real speakers. data/user_settings.json
    # has carried "Blue Snowball" — a MICROPHONE — in AUDIO_AUTOSWITCH_FALLBACK,
    # which can never match a render endpoint. `--state` prints that verdict.
    ap.add_argument("--fallback", default="Realtek USB2.0 Audio")
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
        sw = AudioAutoSwitch(args.headset, args.fallback)
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
