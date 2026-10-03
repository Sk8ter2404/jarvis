"""A fake clone voice server for tests (core/clone_voice_client.py).

A real ThreadingHTTPServer bound to 127.0.0.1 on an ephemeral port, serving
the server contract the client relies on:

  GET  /health    -> health_code + {"ok": ..., "ref_sha256": ..., "pid": ...}
  POST /tts       -> tts_status + a 16-bit mono WAV (after tts_delay seconds)
  POST /shutdown  -> 200

It records every request. Nothing leaves the loopback, nothing is played,
and stop() releases a handler still sleeping in tts_delay at once.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import struct
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_wav(sr: int = 24000, lead_s: float = 0.05, speech_s: float = 0.5,
             tail_s: float = 0.3, amp: float = 0.03) -> bytes:
    """Silence, a 220 Hz tone at `amp`, silence -- as 16-bit mono WAV."""
    n_lead = int(sr * lead_s)
    n_speech = int(sr * speech_s)
    n_tail = int(sr * tail_s)
    frames = bytearray()
    for _ in range(n_lead):
        frames += struct.pack("<h", 0)
    for i in range(n_speech):
        v = amp * math.sin(2.0 * math.pi * 220.0 * i / sr)
        frames += struct.pack("<h", int(round(v * 32767)))
    for _ in range(n_tail):
        frames += struct.pack("<h", 0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
    return buf.getvalue()


class FakeCloneServer:
    def __init__(self, *, ref_sha: str = "", health_code: int = 200,
                 ok: bool = True, tts_delay: float = 0.0,
                 tts_status: int = 200, wav: bytes | None = None):
        self.ref_sha = ref_sha
        self.health_code = health_code
        self.ok = ok
        self.tts_delay = tts_delay
        self.tts_status = tts_status
        self.wav = wav if wav is not None else make_wav()
        # Texts answered with HTTP 500 (one line of a reply fails).
        self.fail_texts: set = set()
        self.requests: list = []
        self._release = threading.Event()
        self._srv = None
        self._th = None
        self._lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self, port: int = 0) -> "FakeCloneServer":
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                return

            def _send(self, code, body, ctype):
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("X-Render-Ms", "12.5")
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass   # the client gave up (timeout); fine

            def do_GET(self):
                fake._record("GET", self.path, None)
                if self.path == "/health":
                    body = json.dumps({"ok": fake.ok, "ref_sha256": fake.ref_sha,
                                       "pid": 4242}).encode()
                    return self._send(fake.health_code, body,
                                      "application/json")
                return self._send(404, b"{}", "application/json")

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    obj = json.loads(raw.decode()) if raw else None
                except Exception:
                    obj = None
                fake._record("POST", self.path, obj,
                             self.headers.get("Content-Type"))
                if self.path == "/shutdown":
                    return self._send(200, b'{"ok": true}', "application/json")
                if self.path != "/tts":
                    return self._send(404, b"{}", "application/json")
                if fake.tts_delay > 0:
                    fake._release.wait(fake.tts_delay)
                text = obj.get("text") if isinstance(obj, dict) else None
                if fake.tts_status != 200 or text in fake.fail_texts:
                    code = fake.tts_status if fake.tts_status != 200 else 500
                    return self._send(code, b'{"error": "x"}',
                                      "application/json")
                return self._send(200, fake.wav, "audio/wav")

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                return    # client-side timeouts close sockets mid-response

        self._srv = Server(("127.0.0.1", int(port)), Handler)
        self._th = threading.Thread(target=self._srv.serve_forever,
                                    kwargs={"poll_interval": 0.02},
                                    name="fake-clone-server", daemon=True)
        self._th.start()
        return self

    def stop(self) -> None:
        self._release.set()
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None

    @property
    def port(self) -> int:
        return self._srv.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ── records ─────────────────────────────────────────────────────────
    def _record(self, method, path, body, ctype=None):
        with self._lock:
            self.requests.append((method, path, body, ctype))

    def tts_texts(self) -> list:
        with self._lock:
            return [b.get("text") for (m, p, b, _c) in self.requests
                    if m == "POST" and p == "/tts" and isinstance(b, dict)]

    def count(self, method: str, path: str) -> int:
        with self._lock:
            return sum(1 for (m, p, _b, _c) in self.requests
                       if m == method and p == path)


def free_port() -> int:
    """A loopback port nothing listens on (bound then released)."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ProfileDir:
    """A temporary data/voice_profiles/ with one consented profile."""

    def __init__(self, name: str = "butler", *, consent=True,
                 source: str = "character", wav: bytes = b"RIFF fake ref"):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.name = name
        pdir = os.path.join(self.root, name)
        os.makedirs(pdir)
        self.ref = os.path.join(pdir, "reference.wav")
        with open(self.ref, "wb") as f:
            f.write(wav)
        meta = {"name": name, "source": source}
        if consent is not None:
            meta["consent"] = consent
        with open(os.path.join(pdir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        self.sha = hashlib.sha256(wav).hexdigest()

    def cleanup(self) -> None:
        self._tmp.cleanup()
