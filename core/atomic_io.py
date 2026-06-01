"""Shared atomic-write helper for JSON state files.

Several skills race on the same state files (most notably `pending_speech.json`,
which the bobert_companion main loop drains and any number of skills append to).
Each skill rolling its own mkstemp/os.replace block created subtle differences
in error handling and indentation, so audits kept flagging the duplicates as
potential write-races. This module centralises the write path so every caller
emits byte-identical output and a fault-tolerant ENOENT/permission fallback.
"""
import json
import os
import tempfile
import time
from typing import Any

# Windows holds an exclusive lock on a file while any process has it open
# without FILE_SHARE_DELETE, which Python's stdlib `open()` does not set.
# When a reader (e.g. hud/bambu_h2d_overlay.py polls the state file) momentarily
# overlaps with our writer's os.replace, MoveFileEx returns ERROR_ACCESS_DENIED
# (WinError 5) and Python raises PermissionError. The window is microseconds
# but at >1Hz write cadences it surfaces several times an hour. Retrying with
# a short backoff lets the reader's `with open():` block exit and the replace
# succeed on the next attempt.
_REPLACE_RETRY_DELAYS_S = (0.010, 0.020, 0.050, 0.100, 0.200)


def _atomic_write_json(path: str, data: Any, *, indent: int | None = 2) -> None:
    """Write `data` as JSON to `path` atomically.

    Strategy: serialise to a sibling tempfile in the same directory, fsync the
    handle (so the bytes are on disk before the rename), then os.replace into
    place. os.replace is atomic on POSIX and Windows for same-volume renames,
    which is what we need — readers either see the previous version or the new
    one, never a half-written file.

    On Windows the final os.replace can transiently fail with PermissionError
    (WinError 5) when a concurrent reader holds the destination open; we retry
    a handful of times with short backoffs before giving up.

    Raises on failure so the caller can decide whether to log or re-queue.
    """
    dir_ = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync isn't available on every filesystem (e.g. some network
                # shares); the os.replace below is still atomic, just without
                # the durability guarantee. Don't fail the write over it.
                pass
        _replace_with_retry(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def _replace_with_retry(src: str, dst: str) -> None:
    """os.replace with Windows-specific retry on PermissionError.

    On POSIX or on the first attempt anywhere, this is just `os.replace`.
    The retries only kick in when Windows returns ERROR_ACCESS_DENIED because
    a reader has the destination momentarily open.
    """
    if os.name != "nt":
        os.replace(src, dst)
        return
    for delay in _REPLACE_RETRY_DELAYS_S:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(delay)
    # Final attempt — let any exception propagate so callers can log it.
    os.replace(src, dst)
