"""Cross-process transport for the message bus (M2 — de-monolith roadmap).

``core.message_bus`` is the IN-process pub/sub; this adds the WIRE FORMAT + a
transport so SEPARATE processes (the tray, the HUD) can exchange bus messages,
replacing the ~50 ``*_state.json`` polling files (M2 design §4). This is the
prerequisite the in-process bus was missing — the tray/HUD wiring follows once
this transport is proven (file IPC retained as a fallback for one release).

Wire format: length-prefixed JSON frames — ``[4-byte big-endian length][utf-8
json]``, where the JSON is ``{"kind": "event"|"request"|..., "name":
<topic/method>, "payload": <json value>}`` (mirrors the native audio service's
binary framing, but JSON for the control plane).

Default transport is a localhost TCP socket (127.0.0.1, loopback-only); a Windows
named-pipe transport can swap in behind the same send/recv helpers. Stdlib only,
every function total (never raises across a socket boundary).
"""
from __future__ import annotations

import json
import socket
import struct
from typing import Any, List, Optional, Tuple

_HEADER = struct.Struct(">I")  # 4-byte big-endian frame length


def encode_frame(kind: str, name: str, payload: Any = None) -> bytes:
    """Encode one bus message to a length-prefixed JSON frame."""
    body = json.dumps({"kind": kind, "name": name, "payload": payload}).encode("utf-8")
    return _HEADER.pack(len(body)) + body


def decode_frames(buffer: bytes) -> Tuple[List[dict], bytes]:
    """Parse every COMPLETE frame at the front of ``buffer``. Returns
    ``(messages, leftover_bytes)``. A frame with corrupt or non-object JSON is
    skipped (its bytes consumed) so one bad frame can't wedge the stream. Never
    raises."""
    msgs: List[dict] = []
    pos = 0
    n = len(buffer)
    while n - pos >= _HEADER.size:
        (length,) = _HEADER.unpack_from(buffer, pos)
        end = pos + _HEADER.size + length
        if end > n:
            break  # incomplete frame — wait for more bytes
        body = buffer[pos + _HEADER.size:end]
        try:
            obj = json.loads(body.decode("utf-8"))
            if isinstance(obj, dict):
                msgs.append(obj)
        except Exception:
            pass  # skip a corrupt frame
        pos = end
    return msgs, buffer[pos:]


class FrameReader:
    """Accumulates bytes from successive socket reads and yields complete bus
    messages — a stateful wrapper around ``decode_frames`` for streaming use."""

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, data: bytes) -> List[dict]:
        self._buf += data
        msgs, self._buf = decode_frames(self._buf)
        return msgs

    @property
    def pending(self) -> int:
        return len(self._buf)


def send_frame(sock: socket.socket, kind: str, name: str, payload: Any = None) -> bool:
    """Send one framed bus message on a connected socket. Returns False on any
    socket error instead of raising."""
    try:
        sock.sendall(encode_frame(kind, name, payload))
        return True
    except OSError:
        return False


def recv_into(sock: socket.socket, reader: FrameReader,
              bufsize: int = 4096) -> Optional[List[dict]]:
    """Read once from ``sock`` and return any complete messages, or None if the
    peer closed (empty read) or the socket errored."""
    try:
        data = sock.recv(bufsize)
    except OSError:
        return None
    if not data:
        return None
    return reader.feed(data)
