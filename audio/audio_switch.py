"""Auto-switch the Windows DEFAULT audio device when a wireless headset powers
on/off — without plugging/unplugging the dongle.

A USB-dongle headset (e.g. the CORSAIR VOID ELITE Wireless) keeps its dongle
plugged in whether the headset is on or off, so plug/unplug detection misses
the power state. But Windows flips the headset's audio ENDPOINT between
``Active`` (on) and ``NotPresent``/``Unplugged`` (off). This module polls that
state and sets the system default render device accordingly:

    headset ON  -> default = the headset   (remember the prior default)
    headset OFF -> default = the prior default, else a configured fallback

Setting the default is done via the ``IPolicyConfigVista`` COM interface
(comtypes) — no external .exe (Smart App Control blocks those), no new pip deps
(comtypes + pycaw already ship with JARVIS).

Standalone:
    python -m audio.audio_switch --list      # show render devices + states
    python -m audio.audio_switch --test      # switch to headset + restore (proves it)
    python -m audio.audio_switch --daemon     # run the watcher in the foreground
"""
from __future__ import annotations

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
        print(f"  [audio-switch] set_default_render failed: {e}", flush=True)
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
        print(f"  [audio-switch] enumerate failed: {e}", flush=True)
    return out


def default_render_id() -> str | None:
    try:
        s = _au().GetSpeakers()
        return s.id if hasattr(s, "id") else s.GetId()
    except Exception:
        return None


def find_active(fragment: str, render_only: bool = True) -> tuple[str, str] | None:
    """Return (id, name) of the first ACTIVE RENDER device whose friendly name
    contains `fragment` (case-insensitive), else None. 'Active' == powered on
    and present (a wireless headset that's OFF reads NotPresent/Unplugged).

    render_only filters to playback endpoints by MMDevice id prefix — render
    ids start with ``{0.0.0.`` and capture (mic) ids with ``{0.0.1.``. Without
    it a headset's MICROPHONE could be matched for an OUTPUT switch, which is
    wrong (the earphone and the mic share the device name)."""
    if not fragment:
        return None
    frag = fragment.lower()
    for did, name, state in list_render():
        if render_only and not did.startswith("{0.0.0."):
            continue
        if state.lower() == "active" and frag in name.lower():
            return did, name
    return None


class AudioAutoSwitch:
    """Background watcher: on the headset's power transitions, move the Windows
    default render device. Idempotent + terminable. Mirrors JARVIS's other
    opt-in daemons (start/stop, a _STOP event, never raises into the loop)."""

    def __init__(self, headset: str, fallback: str = "", poll_s: float = 3.0,
                 announce=None):
        self.headset = headset
        self.fallback = fallback           # name fragment, or "" = remember prior
        self.poll_s = max(1.0, float(poll_s))
        self.announce = announce or (lambda msg: print(f"  [audio-switch] {msg}", flush=True))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prior_default: str | None = None
        self._low_warned = False
        self.low_pct = 15            # warn once when battery drops below this

    def battery_pct(self) -> float | None:
        """Headset battery % from HWiNFO shared memory, or None if HWiNFO /
        Shared Memory Support isn't available."""
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
        on = find_active(self.headset) is not None
        running = bool(self._thread and self._thread.is_alive())
        batt = self.battery_pct()
        suffix = f" at {round(batt)}% battery" if (on and batt and batt > 0) else ""
        return (f"Audio auto-switch is {'running' if running else 'stopped'}, sir. "
                f"The '{self.headset}' headset is {'ON' + suffix if on else 'off'}.")

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
        was_on = find_active(self.headset) is not None
        # Initial sync: if the headset is already on and isn't the default, grab it.
        if was_on:
            self._switch_to_headset()
        while not self._stop.wait(self.poll_s):
            try:
                self.tick(was_on)
                was_on = find_active(self.headset) is not None
                self._check_low_battery()
            except Exception as e:
                print(f"  [audio-switch] tick error: {e}", flush=True)
        if _HAS_COM:
            try:
                comtypes.CoUninitialize()
            except Exception:
                pass

    def tick(self, was_on: bool) -> str | None:
        """One poll step. Returns a short action label or None. Split out so the
        transition logic is unit-testable without the thread."""
        now = find_active(self.headset)
        on = now is not None
        if on and not was_on:
            return self._switch_to_headset()
        if was_on and not on:
            return self._switch_away()
        return None

    def _switch_to_headset(self) -> str | None:
        hs = find_active(self.headset)
        if not hs:
            return None
        cur = default_render_id()
        if cur == hs[0]:
            return None
        self._prior_default = cur
        if set_default_render(hs[0]):
            self.announce(f"headset on — audio moved to {hs[1]}")
            return "to_headset"
        return None

    def _switch_away(self) -> str | None:
        # Headset just powered off. Prefer the default we had before we grabbed
        # the headset; else a configured fallback fragment; else leave Windows'
        # own pick.
        target = self._prior_default
        tname = "the previous device"
        if not target and self.fallback:
            f = find_active(self.fallback)
            if f:
                target, tname = f[0], f[1]
        if target and set_default_render(target):
            self.announce(f"headset off — audio back to {tname}")
            self._prior_default = None
            return "away"
        return None


# ── standalone CLI ───────────────────────────────────────────────────────────
def _main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Auto-switch default audio on headset power.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--test", action="store_true", help="switch to headset then restore")
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--headset", default="CORSAIR VOID ELITE")
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

    if args.test:
        hs = find_active(args.headset)
        if not hs:
            print(f"headset '{args.headset}' is not ACTIVE (power it on first).")
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
