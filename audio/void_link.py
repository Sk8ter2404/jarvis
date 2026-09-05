"""
void_link — is the CORSAIR VOID ELITE WIRELESS headset actually POWERED ON?

WHY THIS MODULE EXISTS
======================
audio/audio_switch.py needs to know when the wireless headset powers on/off so
it can move the Windows default render device. Its `find_active()` docstring
states, as fact:

    'Active' == powered on and present (a wireless headset that's OFF reads
    NotPresent/Unplugged).

THAT CLAIM IS FALSE FOR THIS HEADSET, and nobody had ever checked it. Measured
2026-09-04 on this machine with the headset POWERED OFF: BOTH VOID ELITE
endpoints ("Headset Earphone (CORSAIR VOID ELITE Wireless Gaming Dongle)" and
its microphone) report Status=OK / state Active. The dongle is what Windows
sees, and the dongle is plugged in whether or not the headset is on. So the
MMDevice endpoint state can NEVER be the detector, and the auto-switch feature
has never worked because it was built on one unverified sentence.

The real signal is on the dongle's Corsair vendor-defined HID interface
(VID_1B1C, PID 0x0A51, usage page 0xFFC5): write the two-byte request
``C9 64`` and read back a 5-byte report.

WHAT WAS ACTUALLY MEASURED (2026-09-04, one sample/second, changes only)
=======================================================================
Raw capture, verbatim (hex is the 5-byte reply; `None` = the read timed out
with no report at all):

    20:15:27  6400003300   headset OFF
    20:15:29  6400003400   headset OFF   (byte[3] flips 33<->34 constantly)
    ... 14 such byte[3] flips over 4 minutes, byte[4] always 00 ...
    20:19:41  None         NO REPLY AT ALL - sustained 105 s (pairing handshake)
    20:21:26  6400002600   byte[4]=00, byte[2]=00
    20:21:31  6400003300   byte[4]=00
    20:21:34  6400e0b100   byte[2]=0xe0 (battery appears!) but byte[4] STILL 00
    20:21:37  6400dea101   byte[4]=01  <-- LINK UP
    20:21:38  6400deb101   byte[4]=01  (byte[3] a1<->b1 = noise)
    20:21:40  6400dcb101   byte[4]=01
    20:21:55  6400dca101   byte[4]=01
    20:21:56  6400dcb101   byte[4]=01  (stable ON ever since)

THE ONLY INTERPRETATIONS THE DATA SUPPORTS
==========================================
  reply[0] == 0x64  -> the reply is valid (it echoes the 0x64 in the request).
                       Anything else is garbage -> UNKNOWN, never a parse.
  reply[4] == 0x01  -> link UP   (appeared exactly at link-up, 20:21:37)
  reply[4] == 0x00  -> link DOWN
  reply[4] == other -> UNKNOWN. Not observed; not guessed.
  battery_pct = reply[2] & 0x7F, and ONLY meaningful while the link is up.
                0xde -> 94, 0xdc -> 92, 0xe0 -> 96: a plausible draining pack.
                While off the byte reads 0x00 -> report None, NEVER "0%".
  reply[1], reply[3] carry NO usable state. byte[3] oscillates in BOTH the on
                and the off state (33<->34 off, a1<->b1 on). Any detector that
                reads them is wrong.

Everything else about this protocol is UNKNOWN and is deliberately not claimed
here. In particular the meaning of bit 0x80 in byte[2] (masked off above), the
0x26 seen mid-handshake, and whether byte[4] has values other than 00/01 were
all NOT determined. Unknown is reported as unknown.

LIVE VERIFICATION OF *THIS* MODULE (2026-09-04, after it was written)
=====================================================================
Run against the real dongle with the headset ON:

    discover_device() -> ('\\\\?\\hid#vid_1b1c&pid_0a51&mi_03&col02#...',
                          out_len=20, in_len=5)   in 0.002 s
    _raw_sample()     -> 6400dab101  -> ('on', 90)  in 0.002-0.003 s  (x4)
    VoidLink().state()-> 'unknown' (1st sample), then ('on', 90)
    is_headset_on()   -> True                      in 0.004 s

Battery 90 continues the capture's drain (96 -> 94 -> 92 -> 90), which is an
independent confirmation of the `& 0x7F` reading.

NOT VERIFIED LIVE: the headset was powered ON for that run, so the OFF and the
no-reply branches of THIS module were exercised only against the recorded bytes
in the tests, not against the live device. They are believed correct because
the bytes are real, but that is a re-run of a recording, not a fresh live
measurement — say so rather than claiming a clean end-to-end proof.

THREE HARD-WON BEHAVIOURS, each forced by the capture above
===========================================================
1. NO REPLY IS NOT "OFF". The dongle went completely silent for 105 SECONDS
   during the pairing handshake (20:19:41 -> 20:21:26). A detector that mapped
   no-reply to OFF would have fired a spurious device switch in the middle of
   the owner turning his headset ON. No-reply is UNKNOWN, and UNKNOWN HOLDS the
   last known state.

2. A SINGLE SAMPLE CAN LIE. At 20:21:34 the battery byte was already populated
   (0xe0 = 96%) while byte[4] was still 00 — a transient mid-connect state 3
   seconds before the link actually came up. So state changes are debounced
   over N consecutive AGREEING samples (default 2) before they are believed.

3. THE DEVICE PATH IS NOT STABLE. This dongle has been unplugged and replugged
   for months; the machine currently carries ~19 present (and many more ghost)
   VID_1B1C HID interfaces across different ports. The path is cached for
   speed but INVALIDATED the moment a sample fails, so it is re-discovered
   rather than trusted forever.

IMPLEMENTATION NOTES
====================
* stdlib ctypes only (hid.dll / setupapi.dll / kernel32.dll). Adding hidapi to
  JARVIS is not acceptable and is not needed — the ctypes path is already
  proven on this exact hardware.

* Every read is OVERLAPPED with a timeout + CancelIo. A plain synchronous
  ReadFile on a HID handle BLOCKS FOREVER when no report arrives, which is
  precisely the headset-off case this module exists to detect. The read/write
  timeouts below (400 ms / 300 ms) are the values the live capture ran with;
  the true device latency was never measured, so they are NOT tightened here.

* NOTHING in this module raises to the caller and nothing blocks for long: a
  missing dongle, a denied handle, an iCUE conflict, a non-Windows host (CI is
  ubuntu-latest) all degrade to LINK_UNKNOWN.

* Handles are opened per sample and closed immediately, mirroring the poller
  that produced the capture — a transient conflict with iCUE then costs one
  sample instead of the whole session.
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time

# ── public state vocabulary (the contract other components are written to) ──
LINK_ON = "on"
LINK_OFF = "off"
LINK_UNKNOWN = "unknown"

# ── protocol constants, all taken from the 2026-09-04 capture ───────────────
VOID_VID = 0x1B1C           # Corsair
VOID_PID = 0x0A51           # VOID ELITE Wireless dongle
VOID_USAGE_PAGE = 0xFFC5    # Corsair vendor-defined page
REQUEST = (0xC9, 0x64)      # the two bytes written to ask for link+battery
REPLY_MAGIC = 0x64          # reply[0] echoes this when the reply is real
REPLY_MIN_LEN = 5           # every observed reply was exactly 5 bytes

LINK_BYTE = 4               # reply[4]: 0x01 up, 0x00 down
BATTERY_BYTE = 2            # reply[2] & 0x7F, only while the link is up
BATTERY_MASK = 0x7F

# Timeouts. These are the values the proven capture run used. A live round trip
# with the headset ON measures ~2-3 ms (2026-09-04), so 400 ms is enormous
# headroom for the answering case — but the case that MATTERS is the silent
# one, where the timeout IS the answer, and how long a barely-responsive
# dongle might take was never measured. Keeping the proven, generous numbers.
READ_TIMEOUT_MS = 400
WRITE_TIMEOUT_MS = 300
# A whole state() call must stay "about a second". One exchange can burn
# WRITE+READ = 0.7 s, so the stale-path retry is only attempted when the first
# attempt failed FAST (which is what a stale/vanished path actually does —
# CreateFileW returns immediately). A slow timeout means the dongle is there
# and silent: that is the headset-off / handshake case, and retrying it would
# only double the wait for the same answer.
#
# THE BUDGET MUST THEREFORE SIT *BELOW* THE CHEAPEST TIMEOUT, NOT ABOVE IT.
# It was 0.5 s until 2026-09-04, which sat above BOTH timeouts, so the guard
# could never fire and the "don't ask a silent device twice" rule above was
# documentation for behaviour the code did not have. Every single-failure path
# costs less than 0.5 s:
#     stale/denied path  — CreateFileW fails immediately        ~0 ms
#     write timed out    — WRITE_TIMEOUT_MS, read skipped        300 ms
#     device silent      — write fast + READ_TIMEOUT_MS         ~400 ms
# Measured on this box with _exchange stubbed at the module's own silent-device
# cost and discover_device at its real 2.9 ms: _raw_sample() took 0.804 s doing
# TWO exchanges + a re-enumeration, and is_headset_on() took 2.415 s / 6
# exchanges / 3 discoveries against a docstring promising ~1 s. Re-measured on
# the same rig after this change: 0.400 s / 1 exchange / 0 re-discoveries, and
# 1.209 s / 3 exchanges. The silent case now costs exactly one timeout, which
# is the floor — the timeout IS the answer and cannot be avoided, only
# not-paid-twice.
#
# Derived from the timeouts rather than hand-picked, so it stays correct if
# they are ever retuned. Half the cheapest timeout = 150 ms: ~50x the measured
# fast-fail cost and half the cheapest slow-fail cost. HONEST LIMIT: the exact
# midpoint is a choice, not a measurement. What IS measured is only that a
# fast failure is single-digit ms and the cheapest slow failure is 300 ms, so
# any threshold well inside that gap behaves identically; if a real machine is
# ever found where opening a stale HID path takes >150 ms, this number is
# wrong and needs re-measuring rather than nudging.
RETRY_BUDGET_S = min(READ_TIMEOUT_MS, WRITE_TIMEOUT_MS) / 1000.0 / 2.0

# The debounce EVERY production caller runs with: shared_link(), is_headset_on()
# and battery_percent() all construct VoidLink() with no argument. Hard-won
# behaviour 2 (a single sample can lie) only protects anybody at 2 or more, so
# this must not drop to 1 for snappiness — a one-sample transient would then
# flip the Windows default render device and the next sample would flip it
# back. Floored by tests/test_void_link.py::DebounceTests::
# test_shipped_default_debounce_is_at_least_two.
DEFAULT_DEBOUNCE = 2


# ═══════════════════════════════════════════════════════════════════════════
# Win32 / HID plumbing.  Guarded so this module imports cleanly on the ubuntu
# CI runner (and anywhere else without hid.dll) — it just reports UNKNOWN.
# ═══════════════════════════════════════════════════════════════════════════
_HID_READY = False
try:  # pragma: no cover - exercised implicitly; the tests mock the raw layer
    if sys.platform != "win32":
        raise OSError("HID link probe is Windows-only")

    from ctypes import wintypes

    _hid = ctypes.WinDLL("hid")
    _setupapi = ctypes.WinDLL("setupapi", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_EXISTING = 3
    FILE_FLAG_OVERLAPPED = 0x40000000
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    DIGCF_PRESENT = 0x02
    DIGCF_DEVICEINTERFACE = 0x10
    WAIT_OBJECT_0 = 0x0         # the ONLY return that means "the I/O finished"
    WAIT_TIMEOUT = 0x102
    ERROR_IO_PENDING = 997
    HIDP_STATUS_SUCCESS = 0x00110000

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    class SP_DEVICE_INTERFACE_DATA(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("InterfaceClassGuid", GUID),
                    ("Flags", wintypes.DWORD),
                    ("Reserved", ctypes.POINTER(ctypes.c_ulonglong))]

    class SP_DEVICE_INTERFACE_DETAIL_DATA_W(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD),
                    ("DevicePath", ctypes.c_wchar * 1024)]

    class HIDD_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Size", wintypes.ULONG), ("VendorID", wintypes.USHORT),
                    ("ProductID", wintypes.USHORT),
                    ("VersionNumber", wintypes.USHORT)]

    class HIDP_CAPS(ctypes.Structure):
        _fields_ = [("Usage", wintypes.USHORT), ("UsagePage", wintypes.USHORT),
                    ("InputReportByteLength", wintypes.USHORT),
                    ("OutputReportByteLength", wintypes.USHORT),
                    ("FeatureReportByteLength", wintypes.USHORT),
                    ("Reserved", wintypes.USHORT * 17),
                    ("NumberLinkCollectionNodes", wintypes.USHORT),
                    ("NumberInputButtonCaps", wintypes.USHORT),
                    ("NumberInputValueCaps", wintypes.USHORT),
                    ("NumberInputDataIndices", wintypes.USHORT),
                    ("NumberOutputButtonCaps", wintypes.USHORT),
                    ("NumberOutputValueCaps", wintypes.USHORT),
                    ("NumberOutputDataIndices", wintypes.USHORT),
                    ("NumberFeatureButtonCaps", wintypes.USHORT),
                    ("NumberFeatureValueCaps", wintypes.USHORT),
                    ("NumberFeatureDataIndices", wintypes.USHORT)]

    class OVERLAPPED(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_void_p),
                    ("InternalHigh", ctypes.c_void_p),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                    ("hEvent", wintypes.HANDLE)]

    _setupapi.SetupDiGetClassDevsW.restype = ctypes.c_void_p
    _setupapi.SetupDiGetClassDevsW.argtypes = [
        ctypes.POINTER(GUID), wintypes.LPCWSTR, wintypes.HWND, wintypes.DWORD]
    _setupapi.SetupDiEnumDeviceInterfaces.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(GUID), wintypes.DWORD,
        ctypes.POINTER(SP_DEVICE_INTERFACE_DATA)]
    _setupapi.SetupDiGetDeviceInterfaceDetailW.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(SP_DEVICE_INTERFACE_DATA),
        ctypes.POINTER(SP_DEVICE_INTERFACE_DETAIL_DATA_W), wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    _setupapi.SetupDiDestroyDeviceInfoList.argtypes = [ctypes.c_void_p]

    _kernel32.CreateFileW.restype = ctypes.c_void_p
    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.CreateEventW.restype = wintypes.HANDLE
    _kernel32.CancelIo.argtypes = [ctypes.c_void_p]

    _HID_READY = True
except Exception:  # pragma: no cover - non-Windows / no hid.dll
    _HID_READY = False


# ── low-level helpers (each returns a sentinel; none raises) ────────────────
def _hid_interface_paths() -> list[str]:
    """Every PRESENT HID interface path belonging to VID_1B1C.

    Note DIGCF_PRESENT: the machine also carries a large number of GHOST
    VID_1B1C interfaces from months of replugging, and those must not be
    opened."""
    if not _HID_READY:
        return []
    guid = GUID()
    _hid.HidD_GetHidGuid(ctypes.byref(guid))
    hdev = _setupapi.SetupDiGetClassDevsW(
        ctypes.byref(guid), None, None, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE)
    if not hdev or hdev == INVALID_HANDLE_VALUE:
        return []
    out: list[str] = []
    i = 0
    try:
        while True:
            did = SP_DEVICE_INTERFACE_DATA()
            did.cbSize = ctypes.sizeof(SP_DEVICE_INTERFACE_DATA)
            if not _setupapi.SetupDiEnumDeviceInterfaces(
                    hdev, None, ctypes.byref(guid), i, ctypes.byref(did)):
                break
            i += 1
            detail = SP_DEVICE_INTERFACE_DETAIL_DATA_W()
            # Documented Win32 quirk: cbSize is the size of the FIXED part
            # only — 8 on x64 (DWORD + one WCHAR, padded), NOT sizeof(struct).
            # Passing sizeof(struct) fails with ERROR_INVALID_USER_BUFFER.
            detail.cbSize = 8 if ctypes.sizeof(ctypes.c_void_p) == 8 else 6
            need = wintypes.DWORD(0)
            _setupapi.SetupDiGetDeviceInterfaceDetailW(
                hdev, ctypes.byref(did), ctypes.byref(detail),
                ctypes.sizeof(detail), ctypes.byref(need), None)
            path = detail.DevicePath
            if path and "vid_1b1c" in path.lower():
                out.append(path)
    finally:
        _setupapi.SetupDiDestroyDeviceInfoList(hdev)
    return out


def _open(path: str):
    """Open a HID path for overlapped read+write, or None."""
    if not _HID_READY:
        return None
    h = _kernel32.CreateFileW(
        path, GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING,
        FILE_FLAG_OVERLAPPED, None)
    return None if (not h or h == INVALID_HANDLE_VALUE) else h


def _close(h) -> None:
    try:
        _kernel32.CloseHandle(ctypes.c_void_p(h))
    except Exception:
        pass


def _caps(h):
    pre = ctypes.c_void_p()
    if not _hid.HidD_GetPreparsedData(ctypes.c_void_p(h), ctypes.byref(pre)):
        return None
    try:
        c = HIDP_CAPS()
        if _hid.HidP_GetCaps(pre, ctypes.byref(c)) != HIDP_STATUS_SUCCESS:
            return None
        return c
    finally:
        _hid.HidD_FreePreparsedData(pre)


def _attrs(h):
    a = HIDD_ATTRIBUTES()
    a.Size = ctypes.sizeof(a)
    if not _hid.HidD_GetAttributes(ctypes.c_void_p(h), ctypes.byref(a)):
        return None
    return a


def _write(h, payload, outlen: int) -> bool:
    """Overlapped write with a timeout + CancelIo."""
    buf = (ctypes.c_ubyte * outlen)()
    for i, b in enumerate(payload[:outlen]):
        buf[i] = b
    ov = OVERLAPPED()
    ov.hEvent = _kernel32.CreateEventW(None, True, False, None)
    try:
        wrote = wintypes.DWORD(0)
        ok = _kernel32.WriteFile(ctypes.c_void_p(h), buf, outlen,
                                 ctypes.byref(wrote), ctypes.byref(ov))
        if not ok:
            if ctypes.get_last_error() != ERROR_IO_PENDING:
                return False
            if _kernel32.WaitForSingleObject(
                    ov.hEvent, WRITE_TIMEOUT_MS) == WAIT_TIMEOUT:
                _kernel32.CancelIo(ctypes.c_void_p(h))
                return False
            return True
        return True
    finally:
        _kernel32.CloseHandle(ov.hEvent)


def _read_timeout(h, nbytes: int, ms: int) -> bytes | None:
    """Overlapped read with a HARD timeout. Returns bytes, or None on timeout.

    This is the single most important primitive in the module. A synchronous
    ReadFile on a HID handle never returns when the device sends nothing —
    which is exactly the headset-off and the 105-second-handshake case. The
    timeout is what turns "silent" into an answerable UNKNOWN instead of a
    hung JARVIS thread.

    So the timeout has to hold on EVERY exit path, not just the tidy one. Two
    ways it did not, both fixed here and both pinned by
    tests/test_void_link.py::ReadTimeoutIsHonouredTests:

      * CreateEventW's result was never checked. It CAN return NULL in a
        long-lived pythonw process (handle exhaustion / low resources), and a
        Win32 overlapped read with hEvent == NULL signals the FILE HANDLE
        rather than an event — leaving nothing to wait on and no way to bound
        the wait. The read is now refused outright in that case.

      * Only WAIT_TIMEOUT counted as "no answer". Every other return fell
        through to GetOverlappedResult(..., bWait=TRUE), an UNBOUNDED wait for
        a report that, with the headset off, never arrives — and
        WaitForSingleObject(NULL, ms) returns exactly such a value
        (WAIT_FAILED). That hung the caller's thread for the life of the
        process while holding the HID handle open, i.e. precisely the failure
        this function exists to prevent. Anything that is not WAIT_OBJECT_0
        now cancels the I/O and reports "no answer".

    NOT MEASURED: CreateEventW has never been seen to fail on this machine.
    The hang was reproduced with a stand-in kernel32, not with a genuinely
    exhausted handle table, so the guard is reasoning from the documented
    Win32 contract rather than from an observed local failure. Both branches
    are exercised only against that stand-in."""
    buf = (ctypes.c_ubyte * nbytes)()
    event = _kernel32.CreateEventW(None, True, False, None)
    if not event:
        # No event -> no bounded wait is possible (see the docstring). Do not
        # start an I/O that cannot be abandoned; report "no answer", which
        # every caller already treats as UNKNOWN and never as OFF.
        return None
    ov = OVERLAPPED()
    ov.hEvent = event
    try:
        got = wintypes.DWORD(0)
        ok = _kernel32.ReadFile(ctypes.c_void_p(h), buf, nbytes,
                                ctypes.byref(got), ctypes.byref(ov))
        if not ok:
            if ctypes.get_last_error() != ERROR_IO_PENDING:
                return None
            # WAIT_OBJECT_0 is the only return that means the read completed,
            # and the only one that makes the bWait=TRUE call below bounded.
            # Test for that one good value rather than against WAIT_FAILED:
            # WaitForSingleObject has no restype set, so ctypes hands 0xFFFFFFFF
            # back as the signed int -1 and `== WAIT_FAILED` would never fire.
            if _kernel32.WaitForSingleObject(event, ms) != WAIT_OBJECT_0:
                _kernel32.CancelIo(ctypes.c_void_p(h))
                return None
            if not _kernel32.GetOverlappedResult(
                    ctypes.c_void_p(h), ctypes.byref(ov),
                    ctypes.byref(got), True):
                return None
        return bytes(buf[:got.value])
    finally:
        _kernel32.CloseHandle(event)


# ── device discovery + cache (hard-won behaviour 3) ─────────────────────────
_device_lock = threading.Lock()
_device: tuple[str, int, int] | None = None   # (path, out_len, in_len)


def discover_device() -> tuple[str, int, int] | None:
    """Find the dongle's Corsair vendor interface: (path, out_len, in_len).

    Selected by PID 0x0A51 AND usage page 0xFFC5 — the dongle exposes several
    HID interfaces (keyboard/consumer/vendor) and only the vendor one answers
    C9 64. Returns None when the dongle is absent or every handle is denied."""
    if not _HID_READY:
        return None
    try:
        for path in _hid_interface_paths():
            h = _open(path)
            if h is None:
                continue                     # in use by iCUE, or denied
            try:
                a = _attrs(h)
                if a is None or a.ProductID != VOID_PID or a.VendorID != VOID_VID:
                    continue
                c = _caps(h)
                if (c and c.UsagePage == VOID_USAGE_PAGE
                        and c.OutputReportByteLength >= len(REQUEST)
                        and c.InputReportByteLength >= REPLY_MIN_LEN):
                    return (path, c.OutputReportByteLength,
                            c.InputReportByteLength)
            finally:
                _close(h)
    except Exception:
        return None
    return None


def _exchange(device: tuple[str, int, int]) -> bytes | None:
    """One C9 64 request/response on an already-known device. None on any
    failure — a vanished path, a denied handle, or (the important one) the
    device simply not answering within READ_TIMEOUT_MS."""
    path, outlen, inlen = device
    h = _open(path)
    if h is None:
        return None
    try:
        if not _write(h, REQUEST, outlen):
            return None
        return _read_timeout(h, inlen, READ_TIMEOUT_MS)
    except Exception:
        return None
    finally:
        _close(h)


def invalidate_device() -> None:
    """Drop the cached device path so the next sample re-discovers it."""
    global _device
    with _device_lock:
        _device = None


def _raw_sample() -> bytes | None:
    """One raw C9 64 exchange. Returns the reply bytes, or None.

    None genuinely means "no answer" — it is NOT "off" (see hard-won
    behaviour 1). The cached path is invalidated on every failure so a
    replugged dongle is picked up on the next call rather than being wrong
    forever (hard-won behaviour 3)."""
    global _device
    if not _HID_READY:
        return None
    try:
        started = time.monotonic()
        with _device_lock:
            device, from_cache = _device, _device is not None
        if device is None:
            device = discover_device()
            with _device_lock:
                _device = device
        if device is None:
            return None

        data = _exchange(device)
        if data is not None:
            return data

        # Failed. The path may be stale (this dongle moves ports constantly),
        # so forget it. Only pay for a re-discovery + second exchange when the
        # first attempt failed FAST; a slow failure is the device being
        # silent, and asking a silent device twice just doubles the wait.
        invalidate_device()
        if not from_cache:
            return None
        if time.monotonic() - started > RETRY_BUDGET_S:
            return None
        device = discover_device()
        with _device_lock:
            _device = device
        if device is None:
            return None
        return _exchange(device)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Parsing — pure, and the only place the capture's byte meanings are encoded.
# ═══════════════════════════════════════════════════════════════════════════
def parse_reply(reply: bytes | None) -> tuple[str, int | None]:
    """Interpret one raw 5-byte reply. Pure; never raises.

    Returns (state, battery_pct). battery_pct is None unless the link is UP
    and the battery byte is non-zero — a powered-down headset reads 0x00 there
    and must NEVER be reported as "0%"."""
    try:
        if not reply or len(reply) < REPLY_MIN_LEN:
            # No reply at all, or a truncated one. UNKNOWN — see the 105-second
            # silence at 20:19:41. This is emphatically not OFF.
            return LINK_UNKNOWN, None
        if reply[0] != REPLY_MAGIC:
            # reply[0] echoes the 0x64 we asked with. Anything else means we
            # are looking at some other interface's report or a stale buffer;
            # parsing it would be inventing state.
            return LINK_UNKNOWN, None

        link = reply[LINK_BYTE]
        if link == 0x01:
            raw = reply[BATTERY_BYTE] & BATTERY_MASK
            return LINK_ON, (raw if raw else None)
        if link == 0x00:
            # Note 20:21:34 (6400e0b100): the battery byte can already be
            # populated here while the link is still down. Battery is reported
            # ONLY with the link up, so that transient reads as plain OFF.
            return LINK_OFF, None
        # Never observed. Not guessed.
        return LINK_UNKNOWN, None
    except Exception:
        return LINK_UNKNOWN, None


def probe_once() -> tuple[str, int | None]:
    """One raw sample. Returns (state, battery_pct).

    No debounce and no memory — this is the un-smoothed truth of a single
    exchange, including its known ability to lie mid-handshake. NEVER raises.
    """
    try:
        return parse_reply(_raw_sample())
    except Exception:
        return LINK_UNKNOWN, None


# ═══════════════════════════════════════════════════════════════════════════
# Debounced view (hard-won behaviours 1 and 2)
# ═══════════════════════════════════════════════════════════════════════════
class VoidLink:
    """Debounced headset link state with memory.

    Two rules, both forced by the capture:

      * UNKNOWN HOLDS. A sample that returns no answer contributes nothing —
        it neither changes the state nor disturbs a pending change. The dongle
        was silent for 105 consecutive seconds while the owner was turning the
        headset ON; treating that as OFF would fire a spurious device switch.

      * A CHANGE NEEDS `debounce` CONSECUTIVE AGREEING SAMPLES. At 20:21:34 a
        single sample showed a live battery byte with the link still down,
        3 seconds before the link actually came up.

    JUDGEMENT CALL, not a measurement: an UNKNOWN sample in the middle of a
    pending change is treated as "no information" and leaves the pending run
    intact rather than resetting it. The capture does not contain such a
    sequence, so neither choice is proven; holding is the conservative one
    because it cannot manufacture a state change.

    That call is now PINNED BY TEST — see, in tests/test_void_link.py,
    UnknownHoldsTests.test_silence_does_not_erase_a_pending_change and
    test_off_alternating_with_silence_still_reaches_off. It had to be: the
    reversed version (clear _pending on UNKNOWN) previously passed the whole
    suite, while at runtime it makes an OFF, silence, OFF, silence, ... run —
    a real polling pattern around a link transition — never reach the
    debounce, so the detector reports ON forever after the headset is off.
    Reverse this branch deliberately or not at all.
    """

    def __init__(self, debounce: int = DEFAULT_DEBOUNCE):
        try:
            n = int(debounce)
        except Exception:
            n = DEFAULT_DEBOUNCE
        self._debounce = n if n >= 1 else 1
        self._state = LINK_UNKNOWN
        self._battery: int | None = None
        self._pending: str | None = None
        self._pending_count = 0
        self._lock = threading.Lock()

    @property
    def debounce(self) -> int:
        return self._debounce

    @property
    def last_battery(self) -> int | None:
        """Last believed battery percentage, or None. None whenever the link is
        not up — a powered-down headset has no reading, and "0%" would be a
        fabricated one."""
        return self._battery

    def state(self) -> tuple[str, int | None]:
        """Debounced (state, battery_pct), holding the last known state across
        UNKNOWN samples. NEVER raises.

        Wall clock, measured 2026-09-04 rather than asserted: the dominant
        cost is one exchange, so the worst realistic case is a silent device
        at ~READ_TIMEOUT_MS -> 0.404 s. The stale-path retry adds a second
        exchange only after a FAST failure (see RETRY_BUDGET_S), which is the
        near-free CreateFileW case, so it cannot stack two timeouts."""
        try:
            sampled, battery = probe_once()
        except Exception:
            sampled, battery = LINK_UNKNOWN, None

        with self._lock:
            if sampled == LINK_UNKNOWN:
                # Hold. Deliberately does not touch _pending: see the class
                # docstring's judgement call.
                return self._state, self._battery

            if sampled == self._state:
                # Confirms what we already believe; cancel any pending flip and
                # refresh the battery reading while the link is up.
                self._pending = None
                self._pending_count = 0
                if sampled == LINK_ON and battery is not None:
                    self._battery = battery
                return self._state, self._battery

            # Disagrees with the believed state -> candidate change.
            if sampled == self._pending:
                self._pending_count += 1
            else:
                self._pending = sampled
                self._pending_count = 1

            if self._pending_count >= self._debounce:
                self._state = sampled
                self._battery = battery if sampled == LINK_ON else None
                self._pending = None
                self._pending_count = 0

            return self._state, self._battery

    def reset(self) -> None:
        """Forget everything (used when the dongle is known to have changed)."""
        with self._lock:
            self._state = LINK_UNKNOWN
            self._battery = None
            self._pending = None
            self._pending_count = 0


# ── module-level convenience ───────────────────────────────────────────────
_shared_lock = threading.Lock()
_shared: VoidLink | None = None


def shared_link() -> VoidLink:
    """The process-wide debounced link, so repeated one-shot callers converge
    instead of each starting from scratch."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = VoidLink()
        return _shared


def is_headset_on() -> bool | None:
    """True / False, or None when the state is GENUINELY unknown.

    The three-valued return is the whole point. Collapsing unknown into False
    is the exact defect that makes this feature misfire: it would report the
    headset OFF during the 105-second pairing silence, mid-handshake, or when
    the dongle is simply unplugged.

    A brand-new process starts UNKNOWN by construction (the debounce has not
    yet seen `debounce` agreeing samples), so this takes up to `debounce`
    extra samples to settle. The 1.0 s deadline below starts AFTER the first
    sample and is only checked BETWEEN samples, so the honest worst case is
    "first sample + 1 s + one in-flight sample", not a flat 1 s. Measured
    2026-09-04 against a silent device at the module's own READ_TIMEOUT_MS
    (the slowest case there is): 1.209 s for 3 samples with debounce=2. It was
    2.415 s until RETRY_BUDGET_S was corrected — see the note on that
    constant. Bounded, and short enough for a 1 s poller, but state it as the
    ~1.2 s it measures rather than the ~1 s it used to claim."""
    link = shared_link()
    state, _ = link.state()
    deadline = time.monotonic() + 1.0
    attempts = 0
    while (state == LINK_UNKNOWN and attempts < link.debounce
           and time.monotonic() < deadline):
        attempts += 1
        state, _ = link.state()
    if state == LINK_ON:
        return True
    if state == LINK_OFF:
        return False
    return None


def battery_percent() -> int | None:
    """Last believed battery percentage, or None when the link is not up."""
    return shared_link().last_battery


if __name__ == "__main__":  # pragma: no cover - manual smoke on real hardware
    dev = discover_device()
    print(f"device: {dev}")
    link = VoidLink()
    for _ in range(10):
        st, batt = link.state()
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp}  debounced={st:<8} battery={batt}")
        time.sleep(1.0)
