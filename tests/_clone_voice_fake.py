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

The model facts the client keys its render cache on (2026-10-05): /health
also reports pid (``pid``, 4242) / model / t3_dtype / sample_rate /
t3_decode (``t3_decode``), and
every /tts reply carries X-Audio-Ms, X-Sample-Rate, X-Speech-Tokens (the
clip's 40 ms tokens), X-T3-Ms (``t3_ms_per_token`` each) and X-T3-Engine
(``engine``) like the real server.

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
                 serial: bool = False, t3_decode: str = "cuda-graph",
                 engine: str = "graph", model: str = "chatterbox-turbo",
                 t3_dtype: str = "fp16", sample_rate: int = 24000,
                 t3_ms_per_token: float = 4.25):
        self.ref_sha = ref_sha
        self.health_code = health_code
        self.ok = ok
        self.tts_delay = tts_delay
        self.tts_status = tts_status
        self.wav = wav if wav is not None else make_wav()
        self.latency_for = latency_for
        self.wav_for = dict(wav_for or {})
        self.serial = serial
        self.t3_decode = t3_decode
        self.engine = engine
        self.model = model
        self.t3_dtype = t3_dtype
        self.sample_rate = sample_rate
        self.t3_ms_per_token = t3_ms_per_token
        self.device = "cuda:0 (physical, PCI order)"
        # The process id /health reports: a test "restarts" the server by
        # changing it (with ref_sha / ok / health_code for what came back).
        self.pid = 4242
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

            def _send(self, code, body, ctype, extra=None):
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("X-Render-Ms", "12.5")
                    for k, v in (extra or {}).items():
                        self.send_header(k, str(v))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass   # the client gave up (timeout); fine

            def do_GET(self):
                fake._record("GET", self.path, None)
                if self.path == "/health":
                    body = json.dumps({"ok": fake.ok, "ref_sha256": fake.ref_sha,
                                       "pid": fake.pid, "model": fake.model,
                                       "t3_dtype": fake.t3_dtype,
                                       "sample_rate": fake.sample_rate,
                                       "t3_decode": fake.t3_decode,
                                       "s3gen_device": fake.device}).encode()
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
                wav = fake.wav_for.get(text, fake.wav)
                return self._send(200, wav, "audio/wav", fake.headers_for(wav))

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

    def headers_for(self, wav: bytes) -> dict:
        """The real server's X- headers for a reply carrying `wav`."""
        try:
            with wave.open(io.BytesIO(wav), "rb") as w:
                n, sr = w.getnframes(), w.getframerate()
        except Exception:
            n, sr = 0, self.sample_rate
        audio_ms = 1000.0 * n / max(1, sr)
        tokens = max(1, int(audio_ms // 40))
        return {"X-Audio-Ms": f"{audio_ms:.1f}", "X-Sample-Rate": sr,
                "X-Speech-Tokens": tokens,
                "X-T3-Ms": f"{tokens * self.t3_ms_per_token:.1f}",
                "X-T3-Engine": self.engine}

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


# ── The live pattern of 2026-10-04 10:36-10:37 ────────────────────────────
# Taken from the clone server's own log of that morning (one render at a
# time; "render" = its time on the server, "audio" = the clip's length) and
# the JARVIS session log. The text is made up; every length is the live one
# (normalised characters -- these lines contain nothing the normaliser
# rewrites). The per-line budget then was 2.5 s + 0.03 s per char past 80.
#
#  * 10:36:25, a 7-sentence briefing. Lines 1-6 came back inside their
#    budget; line 7 (114 chars) took 3.76 s against 3.52 s: Kokoro voiced it
#    mid-reply -- miss 1.
#  * 10:36:58, one sentence of 95 chars: 3.33 s against 2.95 s -- miss 2.
#  * 10:37:09, two short sentences, 100 chars (under MIN_CHARS: ONE render):
#    3.32 s against 3.10 s -- miss 3, and the clone latched off for the
#    session ("hes still speaking with old voice").
LIVE_1036_LINES = (
    "Good evening, sir.",
    "A quiet day on the voice channel.",
    "Tomorrow looks mild, with a gentler breeze and grey skies.",
    "Today's headlines, sir.",
    "The city council has approved a new plan for the riverside park, and "
    "the work should begin late next spring, sir.",
    "Local schools will open an hour late on Monday while crews finish the "
    "repairs to the heating.",
    "And finally, a team of local university students is heading west to "
    "compete in a national robotics challenge, sir.",
)
LIVE_1036_RENDER_S = (1.016, 1.368, 2.218, 1.147, 2.962, 2.274, 3.761)
LIVE_1036_AUDIO_S = (1.24, 2.24, 4.64, 1.60, 6.16, 4.64, 6.44)

# The two one-line replies that followed (the misses that latched it).
LIVE_1037_AWAY = ("While you were away, sir: the parcel you ordered was "
                  "delivered and was left by the garage door.")
LIVE_1037_AWAY_RENDER_S = 3.334
LIVE_1037_AWAY_AUDIO_S = 5.40
LIVE_1037_MORNING = ("Good morning, sir. It is just after eight, the air "
                     "outside is cool and the sky is grey and overcast.")
LIVE_1037_MORNING_RENDER_S = 3.318
LIVE_1037_MORNING_AUDIO_S = 5.96

# Since 2026-10-04 the clone voices those two in pieces: a clause head and
# the rest, and sentence by sentence. Those pieces were never rendered live,
# so their times are ESTIMATES: each line's measured render split as a fixed
# cost per render (RENDER_FIXED_S, the intercept of a straight-line fit over
# that morning's nine renders: 0.59 s + 0.025 s per char, +-0.5 s) plus the
# rest of it shared by characters; its audio shared by characters.
LIVE_1037_AWAY_PIECES = ("While you were away, sir:",
                         "the parcel you ordered was delivered and was left "
                         "by the garage door.")
LIVE_1037_MORNING_PIECES = ("Good morning, sir.",
                            "It is just after eight, the air outside is cool "
                            "and the sky is grey and overcast.")
RENDER_FIXED_S = 0.59


def piece_estimate(whole: str, render_s: float, audio_s: float,
                   piece: str) -> tuple:
    """(render s, audio s) estimated for `piece` of the live line `whole`."""
    share = len(piece) / float(len(whole))
    return (RENDER_FIXED_S + (render_s - RENDER_FIXED_S) * share,
            audio_s * share)


def live_1037_timings() -> dict:
    """text -> (render s, audio s) for both later replies: the whole lines
    as measured (what origin/main sent) and their pieces as estimated."""
    out = {}
    for whole, r, a, pieces in (
            (LIVE_1037_AWAY, LIVE_1037_AWAY_RENDER_S, LIVE_1037_AWAY_AUDIO_S,
             LIVE_1037_AWAY_PIECES),
            (LIVE_1037_MORNING, LIVE_1037_MORNING_RENDER_S,
             LIVE_1037_MORNING_AUDIO_S, LIVE_1037_MORNING_PIECES)):
        out[whole] = (r, a)
        for p in pieces:
            out[p] = piece_estimate(whole, r, a, p)
    return out


def live_1036_server(ref_sha: str, scale: float,
                     later: bool = False) -> "FakeCloneServer":
    """A serial fake server that renders the 10:36 lines with their measured
    render times and audio lengths, both multiplied by `scale` (a test runs
    the pattern faster; every ratio is kept). `later` adds the two replies
    of 10:36:58 and 10:37:09 (whole and in pieces). Not started."""
    timings = {t: (r, a) for t, r, a in zip(
        LIVE_1036_LINES, LIVE_1036_RENDER_S, LIVE_1036_AUDIO_S)}
    if later:
        timings.update(live_1037_timings())
    lat = {t: r * scale for t, (r, _a) in timings.items()}
    wavs = {t: make_wav(lead_s=0.0, speech_s=a * scale, tail_s=0.0, amp=0.3)
            for t, (_r, a) in timings.items()}
    return FakeCloneServer(ref_sha=ref_sha, latency_for=lat, wav_for=wavs,
                           serial=True)
