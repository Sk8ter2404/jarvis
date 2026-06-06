"""Read HWiNFO's shared-memory sensor block (e.g. the Corsair VOID headset's
battery %). HWiNFO Pro publishes every sensor it polls into a named shared
memory block (``Global\\HWiNFO_SENS_SM2``) when **Settings -> Shared Memory
Support** is enabled (a toggle separate from the Pro licence; it activates live,
no restart). This module maps that block read-only and parses the readings.

Used by audio/audio_switch.py to add battery info to the headset status. Pure
read-only + fully graceful: if HWiNFO isn't running or Shared Memory Support is
off, every function returns "no data" rather than raising.

Probe once the toggle is on:
    python -m audio.hwinfo --find VOID
    python -m audio.hwinfo --battery "VOID"
"""
from __future__ import annotations

import struct
import sys

# HWiNFO_SENSORS_SHARED_MEM2 header: sig,ver,rev, poll(i64), then 6 section DWORDs.
_HDR = struct.Struct("<IIIqIIIIII")          # 44 bytes
_HDR_SIZE = _HDR.size
# Each reading: tReading, sensorIndex, readingID (3 DWORDs) then
# szLabelOrig[128], szLabelUser[128], szUnit[16], then 4 doubles (Value/Min/Max/Avg).
_LBL = 128
_UNIT = 16
_VALUE_OFFSET_IN_READING = 12 + _LBL + _LBL + _UNIT   # = 284 -> first double (Value)

_SM_NAMES = ("Global\\HWiNFO_SENS_SM2", "HWiNFO_SENS_SM2", "Local\\HWiNFO_SENS_SM2")


def _read_raw() -> bytes | None:
    """Map the whole HWiNFO SM block read-only and copy it out, or None."""
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenFileMappingW.restype = wintypes.HANDLE
        k32.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        k32.MapViewOfFile.restype = ctypes.c_void_p
        k32.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.DWORD, ctypes.c_size_t]
        k32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
    except Exception:
        return None
    FILE_MAP_READ = 0x0004
    for name in _SM_NAMES:
        h = k32.OpenFileMappingW(FILE_MAP_READ, False, name)
        if not h:
            continue
        ptr = k32.MapViewOfFile(h, FILE_MAP_READ, 0, 0, 0)   # 0 = whole block
        if not ptr:
            k32.CloseHandle(h)
            continue
        try:
            head = ctypes.string_at(ptr, _HDR_SIZE)
            # HWiNFO dwSignature == 0x53695748; in little-endian memory those
            # bytes read as b"HWiS" (NOT b"SiWH", the big-endian / C multi-char
            # constant rendering, which never appears in the actual mapping).
            if struct.unpack_from("<I", head, 0)[0] != 0x53695748:
                continue
            _, _, _, _, _s_off, _s_sz, _s_n, r_off, r_sz, r_n = _HDR.unpack(head)
            total = r_off + r_sz * r_n
            return ctypes.string_at(ptr, total)
        except Exception:
            return None
        finally:
            try:
                k32.UnmapViewOfFile(ptr)
                k32.CloseHandle(h)
            except Exception:
                pass
    return None


def parse_readings(raw: bytes) -> list[tuple[str, float, str]]:
    """[(label, value, unit)] for each reading in a HWiNFO SM2 block. Pure —
    unit-testable against a synthesised block."""
    if not raw or len(raw) < _HDR_SIZE or struct.unpack_from("<I", raw, 0)[0] != 0x53695748:
        return []
    _, _, _, _, _s_off, _s_sz, _s_n, r_off, r_sz, r_n = _HDR.unpack(raw[:_HDR_SIZE])
    out = []
    for i in range(r_n):
        base = r_off + i * r_sz
        if base + _VALUE_OFFSET_IN_READING + 8 > len(raw):
            break
        lo = raw[base + 12:base + 12 + _LBL].split(b"\x00", 1)[0].decode("latin-1", "replace")
        lu = raw[base + 12 + _LBL:base + 12 + 2 * _LBL].split(b"\x00", 1)[0].decode("latin-1", "replace")
        unit = raw[base + 12 + 2 * _LBL:base + 12 + 2 * _LBL + _UNIT].split(b"\x00", 1)[0].decode("latin-1", "replace")
        value = struct.unpack_from("<d", raw, base + _VALUE_OFFSET_IN_READING)[0]
        out.append((lu or lo, value, unit))
    return out


def available() -> bool:
    """True if HWiNFO's shared memory is readable right now."""
    return _read_raw() is not None


def readings() -> list[tuple[str, float, str]]:
    return parse_readings(_read_raw() or b"")


def find(*fragments: str) -> tuple[str, float, str] | None:
    """First reading whose label contains ALL `fragments` (case-insensitive)."""
    frags = [f.lower() for f in fragments if f]
    for label, value, unit in readings():
        low = label.lower()
        if all(f in low for f in frags):
            return label, value, unit
    return None


def battery(name_fragment: str) -> float | None:
    """Battery % for the device whose label contains `name_fragment`. Prefers a
    reading whose label says 'battery'/'charge' (the headset also exposes a
    volume reading in '%', which must NOT be mistaken for the battery), then
    falls back to any '%' reading. None if HWiNFO/the sensor isn't available."""
    if not name_fragment:
        return None
    nf = name_fragment.lower()
    rs = readings()
    for label, value, unit in rs:          # 1) explicit battery/charge reading
        low = label.lower()
        if nf in low and ("batt" in low or "charge" in low):
            return value
    for label, value, unit in rs:          # 2) fall back to a percentage reading
        if nf in label.lower() and unit == "%":
            return value
    return None


def summary() -> dict:
    """Grouped snapshot of the most useful HWiNFO sensors for a spoken/HUD
    read-out. Selects purely by unit + label fragment so it's robust to ANY
    HWiNFO config (never a hard-coded sensor name). Returns ``available=False``
    with empty lists when shared memory is off — never raises."""
    rs = readings()
    out: dict = {
        "available": bool(rs),
        "count": len(rs),
        "cpu_temp_c": None, "gpu_temp_c": None,
        "cpu_load_pct": None, "gpu_load_pct": None,
        "temps_c": [], "fans_rpm": [], "power_w": [],
        "clocks_mhz": [], "voltages_v": [],
    }

    def _temp_unit(u: str) -> str | None:
        # HWiNFO temps come through as "C"/"°C" or, if HWiNFO is set to display
        # Fahrenheit globally, "F"/"°F"/"FAH" depending on encoding. Returns
        # "C", "F", or None (not a temperature unit).
        bare = u.replace("°", "").strip().upper()
        if bare in ("C", "CEL"):
            return "C"
        if bare in ("F", "FAH"):
            return "F"
        return None

    for label, value, unit in rs:
        low = label.lower()
        u = (unit or "").strip()
        uu = u.upper()
        tu = _temp_unit(u)
        if tu is not None:
            # Keep cpu_temp_c/gpu_temp_c Celsius-canonical: convert °F on ingest.
            if tu == "F":
                value = (value - 32) / 1.8
            out["temps_c"].append((label, value))
        elif uu == "RPM":
            out["fans_rpm"].append((label, value))
        elif uu == "W":
            out["power_w"].append((label, value))
        elif uu in ("MHZ", "GHZ"):
            out["clocks_mhz"].append((label, value * (1000.0 if uu == "GHZ" else 1.0)))
        elif uu == "V":
            out["voltages_v"].append((label, value))
        elif u == "%" and any(k in low for k in ("usage", "load", "total", "util")):
            if "cpu" in low and out["cpu_load_pct"] is None:
                out["cpu_load_pct"] = value
            elif "gpu" in low and out["gpu_load_pct"] is None:
                out["gpu_load_pct"] = value

    def _pick_temp(frag: str, prefer: tuple[str, ...]):
        cands = [(l, v) for l, v in out["temps_c"] if frag in l.lower()]
        for l, v in cands:                      # prefer the canonical package/edge sensor
            if any(p in l.lower() for p in prefer):
                return v
        return max((v for _, v in cands), default=None)   # else the hottest labelled one

    out["cpu_temp_c"] = _pick_temp("cpu", ("package", "tctl", "tdie", "ccd"))
    out["gpu_temp_c"] = _pick_temp("gpu", ("edge", "hot spot", "temperature", "core"))
    return out


def _main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Read HWiNFO shared-memory sensors.")
    ap.add_argument("--find", default="", help="print readings whose label contains this")
    ap.add_argument("--battery", default="", help="print battery %% for a device fragment")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    if not available():
        print("HWiNFO shared memory NOT available — enable HWiNFO -> Settings -> "
              "Shared Memory Support (it activates live).")
        return 1
    rs = readings()
    print(f"HWiNFO shared memory OK — {len(rs)} readings.")
    if args.battery:
        print(f"battery({args.battery!r}) = {battery(args.battery)}")
    if args.find or args.all:
        f = args.find.lower()
        for label, value, unit in rs:
            if args.all or f in label.lower():
                print(f"  {label!r:48} = {value:g} {unit}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
