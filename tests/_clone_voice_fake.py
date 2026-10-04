"""A fake clone voice server for tests (core/clone_voice_client.py).

A real ThreadingHTTPServer bound to 127.0.0.1 on an ephemeral port, serving
the server contract the client relies on:

  GET  /health    -> health_code + {"ok": ..., "ref_sha256": ..., "pid": ...}
  POST /tts       -> tts_status + a 16-bit mono WAV (after tts_delay seconds)
  POST /shutdown  -> 200

Per-request control (2026-10-04, the live-budget replay):
  latency_for  dict text -> seconds, or a callable(text) -> seconds: how long
               THIS line "renders" (falls back to tts_delay)
  wav_for      dict text -> WAV bytes: what THIS line returns (its audio
               length), falling back to `wav`
  serial       True: one render at a time, like the real server (a request
               waits for the one before it, even one the client gave up on)
  timings      (text, start, end) in time.monotonic() per finished render

It records every request. Nothing leaves the loopback, nothing is played,
and stop() releases a handler still sleeping in a delay at once.
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
import time
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
                 tts_status: int = 200, wav: bytes | None = None,
                 latency_for=None, wav_for: dict | None = None,
                 serial: bool = False):
        self.ref_sha = ref_sha
        self.health_code = health_code
        self.ok = ok
        self.tts_delay = tts_delay
        self.tts_status = tts_status
        self.wav = wav if wav is not None else make_wav()
        self.latency_for = latency_for
        self.wav_for = dict(wav_for or {})
        self.serial = serial
        self._render_mu = threading.Lock()
        self.timings: list = []
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
                text = obj.get("text") if isinstance(obj, dict) else None
                if fake.serial:
                    with fake._render_mu:
                        fake._render(text)
                else:
                    fake._render(text)
                if fake.tts_status != 200 or text in fake.fail_texts:
                    code = fake.tts_status if fake.tts_status != 200 else 500
                    return self._send(code, b'{"error": "x"}',
                                      "application/json")
                return self._send(200, fake.wav_for.get(text, fake.wav),
                                  "audio/wav")

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

    def latency(self, text) -> float:
        """How long `text` renders: latency_for, else tts_delay."""
        lf = self.latency_for
        if callable(lf):
            return float(lf(text))
        if isinstance(lf, dict) and text in lf:
            return float(lf[text])
        return float(self.tts_delay)

    def _render(self, text) -> None:
        t0 = time.monotonic()
        delay = self.latency(text)
        if delay > 0:
            self._release.wait(delay)
        with self._lock:
            self.timings.append((text, t0, time.monotonic()))

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


# ── The live pattern of 2026-10-04 10:36 (a 7-sentence briefing) ──────────
# Line lengths, render times and audio lengths as the clone server logged
# them that morning (rounded; the text is made up, the shape is not). Lines
# 5-7 took longer than the fixed per-line budget (2.5 s + 0.03 s per char past
# 80) although seconds of earlier audio were still queued: Kokoro voiced them
# mid-reply and three misses in a row latched the clone off for the session.
LIVE_1036_LINES = (
    "Good evening, sir.",
    "A quiet day on the voice channel.",
    "Tomorrow looks mild, with a light breeze and grey skies.",
    "Today's headlines, sir.",
    "The city council has approved a new plan for the riverside park, and "
    "work should begin early next spring, sir.",
    "Local schools will open an hour late on Monday while crews finish the "
    "repairs to the heating.",
    "And finally, a team of university students is heading south to compete "
    "in a national robotics challenge, sir.",
)
LIVE_1036_RENDER_S = (1.0, 1.4, 1.6, 1.1, 3.8, 3.5, 3.8)
LIVE_1036_AUDIO_S = (1.24, 2.24, 4.64, 1.6, 6.16, 4.64, 6.44)


def live_1036_server(ref_sha: str, scale: float) -> "FakeCloneServer":
    """A serial fake server that renders the 10:36 lines with their measured
    render times and audio lengths, both multiplied by `scale` (a test runs
    the pattern faster; every ratio is kept). Not started."""
    lat = {t: r * scale for t, r in zip(LIVE_1036_LINES, LIVE_1036_RENDER_S)}
    wavs = {t: make_wav(lead_s=0.0, speech_s=a * scale, tail_s=0.0, amp=0.3)
            for t, a in zip(LIVE_1036_LINES, LIVE_1036_AUDIO_S)}
    return FakeCloneServer(ref_sha=ref_sha, latency_for=lat, wav_for=wavs,
                           serial=True)
