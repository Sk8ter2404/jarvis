"""'How's the Backblaze backup doing?' answered from Backblaze's own files
(2026-10-19; master sheet idea #361 - on 10-18 the numbers were read by hand).

READ-ONLY. The Backblaze client keeps its status in
C:\\ProgramData\\Backblaze\\bzdata\\bzreports\\:
  bzstat_totalbackup.xml      <totals totnumfilesforbackup= totnumbytesforbackup=>
  bzstat_remainingbackup.xml  <remaining remainingnumfilesforbackup= remainingnumbytesforbackup=>
  bzstat_lastbackupcompleted.xml  <lastbackupcompleted gmt_millis=>
This skill reads those three and says only what they hold. Nothing is changed
(no pause, no "Backup Now"): those stay the owner's.
"""
from __future__ import annotations

import os
import re
import time

SPEAK_VERBATIM_ACTIONS = ("backblaze_status", "backup_status", "how_is_the_backup")

REPORTS = r"C:\ProgramData\Backblaze\bzdata\bzreports"

PROMPT_EXAMPLES = (
    "BACKBLAZE (skills/backblaze_status.py - READ-ONLY):\n"
    "  backblaze_status - how much of the PC's Backblaze backup is done, what is\n"
    "                     left and when the last pass finished. Fire on 'how's\n"
    "                     the backup', 'how's Backblaze doing', 'is my backup\n"
    "                     done'. -> [ACTION: backblaze_status]\n"
)


def _attr(text: str, tag: str, attr: str):
    m = re.search(r"<%s\b[^>]*\b%s=\"([^\"]*)\"" % (tag, attr), text or "")
    return m.group(1) if m else None


def _read(name: str, folder: str = REPORTS) -> str:
    try:
        with open(os.path.join(folder, name), "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def read_status(folder: str = REPORTS) -> dict:
    """Numbers from the three files; a field that could not be read is None."""
    def num(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    tot = _read("bzstat_totalbackup.xml", folder)
    rem = _read("bzstat_remainingbackup.xml", folder)
    last = _read("bzstat_lastbackupcompleted.xml", folder)
    ms = num(_attr(last, "lastbackupcompleted", "gmt_millis"))
    return {
        "total_files": num(_attr(tot, "totals", "totnumfilesforbackup")),
        "total_bytes": num(_attr(tot, "totals", "totnumbytesforbackup")),
        "left_files": num(_attr(rem, "remaining", "remainingnumfilesforbackup")),
        "left_bytes": num(_attr(rem, "remaining", "remainingnumbytesforbackup")),
        "last_done": (ms / 1000.0) if ms else None,
    }


def _size(n: int) -> str:
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6)):
        if n >= div:
            v = n / div
            return f"{v:.2f} {unit}" if v < 10 else f"{v:.1f} {unit}" if v < 100 else f"{v:.0f} {unit}"
    return f"{n} bytes"


def _when(ts, now=None) -> str:
    now = time.time() if now is None else now
    lt, nt = time.localtime(ts), time.localtime(now)
    clock = time.strftime("%I:%M %p", lt).lstrip("0")
    if (lt.tm_year, lt.tm_yday) == (nt.tm_year, nt.tm_yday):
        return f"{clock} today"
    if now - ts < 6 * 86400:
        return f"{time.strftime('%A', lt)} at {clock}"
    return f"{time.strftime('%B', lt)} {lt.tm_mday} at {clock}"


def describe(st: dict, now=None) -> str:
    tb, lb, lf = st.get("total_bytes"), st.get("left_bytes"), st.get("left_files")
    if not tb:
        return "I couldn't read Backblaze's status files just now, sir."
    parts = []
    if lb is not None:
        done = max(0.0, min(100.0, 100.0 * (tb - lb) / tb))
        pct = f"{done:.1f}" if done >= 99 else f"{done:.0f}"
        parts.append(f"Backblaze has {pct} percent of your {_size(tb)} backed up, sir")
        if lb > 0:
            files = f"{lf:,} file{'' if lf == 1 else 's'}, " if lf is not None else ""
            parts.append(f"with {files}{_size(lb)} still to go")
        else:
            parts.append("with nothing waiting")
    else:
        parts.append(f"Backblaze is backing up {_size(tb)}, sir, but I couldn't read what's left")
    line = ", ".join(parts) + "."
    if st.get("last_done"):
        line += f" The last full pass finished {_when(st['last_done'], now)}."
    return line


def backblaze_status(_: str = "") -> str:
    if not os.path.isdir(REPORTS):
        return "Backblaze doesn't seem to be installed on this PC, sir."
    return describe(read_status())


def register(actions):
    actions["backblaze_status"] = backblaze_status
    actions["backup_status"] = backblaze_status
    actions["how_is_the_backup"] = backblaze_status
