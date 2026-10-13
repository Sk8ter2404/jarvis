"""core/discord_bridge.py - JARVIS <-> the owner over Discord direct messages.

One bot, one human. The bot DMs the owner (alerts through the phone bridge,
replies to his commands) and the owner DMs the bot to talk to JARVIS - each
DM is handled like a typed command and the answer comes back in the same DM.

Configuration is environment-only and never stored in the repo, the settings
file or a log line:
  DISCORD_BOT_TOKEN   the bot token from the Discord developer portal
  DISCORD_OWNER_ID    the owner's numeric Discord user id (digits only)
Both must be set, otherwise this module is inert (configured() is False).
Setup walk-through: docs/DISCORD_SETUP.md.

Why no discord.py: it is a large dependency and its client wants to own the
event loop. The Discord gateway is a small JSON-over-websocket protocol and
the `websockets` package (already a JARVIS dependency) plus `requests` cover
it in a few hundred lines:
  * OUTBOUND  send_dm() is plain REST (open the DM channel once, POST
    messages, honour 429 retry_after). It needs no gateway connection, so
    alerts still go out while the listener is reconnecting.
  * INBOUND   GatewayClient runs its own thread with its own asyncio loop:
    HELLO -> IDENTIFY (or RESUME), heartbeats, MESSAGE_CREATE. Reconnects
    with exponential backoff + jitter and resumes the session when it can.
    A message is handed to a worker thread, never run on the gateway loop,
    so a slow command can never starve the heartbeat or the voice loop.

Security: ONLY a DM from DISCORD_OWNER_ID is acted on. Anyone else - another
user DMing the bot, a message in a server channel, another bot, the bot's own
echo - is dropped silently (no reply, so the bot is not a probe for who owns
it); a counter and a content-free log line record that it happened.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import random
import re
import threading
import time
from typing import Any, Callable, Optional

_log = logging.getLogger(__name__)

API_BASE = "https://discord.com/api/v10"
GATEWAY_URL = "wss://gateway.discord.gg"
GATEWAY_QUERY = "?v=10&encoding=json"

# Gateway intents: DIRECT_MESSAGES (1 << 12) + MESSAGE_CONTENT (1 << 15).
INTENT_DIRECT_MESSAGES = 1 << 12
INTENT_MESSAGE_CONTENT = 1 << 15
INTENTS = INTENT_DIRECT_MESSAGES | INTENT_MESSAGE_CONTENT

HTTP_TIMEOUT = 10.0
MAX_MESSAGE_LEN = 1900            # Discord's hard cap is 2000
MAX_RATE_LIMIT_WAIT = 15.0        # never sleep longer than this on a 429
BACKOFF_START = 2.0
BACKOFF_MAX = 120.0
STABLE_AFTER_S = 60.0             # a connection this old resets the backoff
WORK_QUEUE_MAX = 8                # owner DMs waiting for a worker
INIT_DELAY_S = 8.0                # let the rest of JARVIS boot first

# Close codes that mean "retrying cannot help": bad token, bad intents, etc.
FATAL_CLOSE_CODES = {
    4004: "the bot token was rejected (DISCORD_BOT_TOKEN is wrong or was reset)",
    4010: "invalid shard (internal error)",
    4011: "sharding required (internal error)",
    4012: "invalid gateway API version (internal error)",
    4013: "invalid intents (internal error)",
    4014: "a privileged intent is not enabled: turn on 'Message Content "
          "Intent' for the bot in the Discord developer portal",
}

_SECRET_RE = re.compile(r"[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{20,}")


# --- configuration ------------------------------------------------------

def token() -> str:
    return (os.environ.get("DISCORD_BOT_TOKEN") or "").strip()


def owner_id() -> str:
    """The owner's user id as a digit string, or '' when unset or malformed
    (a malformed id must mean 'nobody', never 'anybody')."""
    raw = (os.environ.get("DISCORD_OWNER_ID") or "").strip()
    return raw if raw.isdigit() else ""


def configured() -> bool:
    """Both env vars present and the owner id well-formed."""
    return bool(token() and owner_id())


def _scrub(text: object) -> str:
    """Text safe to log: the bot token (and anything shaped like one) is
    replaced. Defence in depth - the token is never put in a URL or message
    this module builds, but library exception strings are not ours."""
    s = str(text)
    tok = token()
    if tok:
        s = s.replace(tok, "[token]")
    return _SECRET_RE.sub("[token]", s)


def split_message(text: str, limit: int = MAX_MESSAGE_LEN) -> list[str]:
    """Cut `text` into <= limit pieces, preferring paragraph / line / word
    boundaries."""
    text = (text or "").strip()
    if not text:
        return []
    out: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = max(rest.rfind("\n\n", 0, limit), rest.rfind("\n", 0, limit),
                  rest.rfind(" ", 0, limit))
        if cut < limit // 2:
            cut = limit
        out.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        out.append(rest)
    return out


# --- shared status ------------------------------------------------------

_lock = threading.RLock()
_status: dict[str, Any] = {
    "connected": False,
    "started_at": None,
    "last_connect_at": None,
    "last_inbound_at": None,
    "last_outbound_at": None,
    "last_error": None,
    "messages_in": 0,
    "messages_out": 0,
    "ignored": 0,
    "reconnects": 0,
    "fatal": None,
}
_dm_channel: list[Optional[str]] = [None]   # cached DM channel id


def status() -> dict:
    with _lock:
        return dict(_status)


def _set_error(msg: str) -> None:
    with _lock:
        _status["last_error"] = _scrub(msg)[:200]


# --- outbound REST ------------------------------------------------------

def _headers() -> dict:
    return {"Authorization": f"Bot {token()}",
            "Content-Type": "application/json",
            "User-Agent": "DiscordBot (JARVIS, 1.0)"}


def _http():
    import requests  # type: ignore
    return requests


def _api(method: str, path: str, *, payload: Optional[dict] = None,
         http=None, sleep: Optional[Callable[[float], None]] = None):
    """One REST call with a single 429 retry. Returns the response or None."""
    rq = http or _http()
    url = API_BASE + path
    for attempt in (0, 1):
        try:
            r = rq.request(method, url, headers=_headers(), json=payload,
                           timeout=HTTP_TIMEOUT)
        except Exception as e:
            _log.warning("[discord] %s %s failed: %s", method, path, _scrub(e))
            _set_error(f"http: {e}")
            return None
        if r.status_code == 429 and attempt == 0:
            try:
                wait = float((r.json() or {}).get("retry_after", 1.0))
            except Exception:
                wait = 1.0
            (sleep or time.sleep)(min(max(wait, 0.0), MAX_RATE_LIMIT_WAIT))
            continue
        return r
    return None


def _open_dm_channel(http=None) -> Optional[str]:
    """The id of the DM channel with the owner (created on first use,
    then cached)."""
    with _lock:
        cached = _dm_channel[0]
    if cached:
        return cached
    oid = owner_id()
    if not (token() and oid):
        return None
    r = _api("POST", "/users/@me/channels", payload={"recipient_id": oid},
             http=http)
    if r is None:
        return None
    if r.status_code >= 400:
        _log.warning("[discord] could not open the DM channel: HTTP %s",
                     r.status_code)
        _set_error(f"open DM: HTTP {r.status_code}")
        return None
    try:
        cid = str((r.json() or {}).get("id") or "")
    except Exception:
        cid = ""
    if not cid:
        return None
    with _lock:
        _dm_channel[0] = cid
    return cid


def send_dm(text: str, *, http=None) -> bool:
    """DM `text` to the owner. True only when EVERY chunk was accepted.
    Blocking (a couple of HTTPS calls) - call from a worker thread, as the
    phone-ping sender and the inbound worker both do. Never raises."""
    try:
        if not configured():
            return False
        chunks = split_message(text)
        if not chunks:
            return False
        cid = _open_dm_channel(http=http)
        if not cid:
            return False
        for chunk in chunks:
            r = _api("POST", f"/channels/{cid}/messages",
                     payload={"content": chunk}, http=http)
            if r is None:
                return False
            if r.status_code >= 400:
                _log.warning("[discord] send failed: HTTP %s", r.status_code)
                _set_error(f"send: HTTP {r.status_code}")
                if r.status_code in (403, 404):
                    # The cached channel may be stale (e.g. the user closed
                    # the DM): forget it so the next send re-opens one.
                    with _lock:
                        _dm_channel[0] = None
                return False
        with _lock:
            _status["last_outbound_at"] = time.time()
            _status["messages_out"] += 1
        return True
    except Exception as e:
        _log.warning("[discord] send_dm crashed: %s", _scrub(e))
        _set_error(f"send: {e}")
        return False


# --- inbound gateway ----------------------------------------------------

class _Reconnect(Exception):
    """Drop this connection; `resume` says whether the session survives."""

    def __init__(self, resume: bool, delay: float = 0.0):
        super().__init__("reconnect")
        self.resume = resume
        self.delay = delay


class GatewayClient:
    """Owner-DM listener. `handler(text) -> str` turns one command into a
    reply; it runs on a worker thread. Construct with injected `connect` /
    `sender` / `sleep` in tests; production uses the defaults."""

    def __init__(self, handler: Callable[[str], str], *,
                 connect: Optional[Callable] = None,
                 sender: Optional[Callable[[str], bool]] = None,
                 token_fn: Callable[[], str] = token,
                 owner_fn: Callable[[], str] = owner_id):
        self._handler = handler
        self._connect = connect
        self._send = sender or send_dm
        self._token_fn = token_fn
        self._owner_fn = owner_fn
        self._thread: Optional[threading.Thread] = None
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._work: "queue.Queue[str]" = queue.Queue(maxsize=WORK_QUEUE_MAX)
        self._seq: Optional[int] = None
        self._session_id: Optional[str] = None
        self._resume_url: Optional[str] = None
        self._bot_id: Optional[str] = None
        self._seen_ids: list[str] = []
        self._acked = True

    # -- lifecycle --
    def start(self, *, initial_delay: float = INIT_DELAY_S) -> bool:
        """Spawn the listener thread (idempotent)."""
        if not (self._token_fn() and self._owner_fn()):
            return False
        t = self._thread
        if t is not None and t.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._thread_main, args=(initial_delay,),
            name="discord-gateway", daemon=True)
        self._thread.start()
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._worker_main, name="discord-worker", daemon=True)
            self._worker.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def is_running(self) -> bool:
        t = self._thread
        return bool(t is not None and t.is_alive() and not self._stop.is_set())

    # -- thread bodies --
    def _thread_main(self, initial_delay: float) -> None:  # pragma: no cover - thin shell
        try:
            if initial_delay and self._stop.wait(initial_delay):
                return
            asyncio.run(self.run_forever())
        except Exception as e:
            _log.warning("[discord] gateway thread ended: %s", _scrub(e))
            _set_error(f"thread: {e}")
        finally:
            with _lock:
                _status["connected"] = False

    def _worker_main(self) -> None:  # pragma: no cover - thin shell
        while not self._stop.is_set():
            try:
                text = self._work.get(timeout=1.0)
            except queue.Empty:
                continue
            self.process_owner_text(text)

    # -- the reconnect loop --
    async def run_forever(self, *, sleep=asyncio.sleep) -> None:
        backoff = BACKOFF_START
        with _lock:
            _status["started_at"] = time.time()
        while not self._stop.is_set():
            began = time.monotonic()
            resume = False
            extra_delay = 0.0
            try:
                await self._connect_once()
            except _Reconnect as r:
                resume, extra_delay = r.resume, r.delay
            except _Fatal as f:
                with _lock:
                    _status["fatal"] = str(f)
                _log.error("[discord] giving up: %s", f)
                print(f"  [discord] stopped: {f}")
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _log.warning("[discord] connection lost: %s", _scrub(e))
                _set_error(f"gateway: {e}")
                resume = True
            with _lock:
                _status["connected"] = False
                _status["reconnects"] += 1
            if not resume:
                self._session_id = None
                self._seq = None
                self._resume_url = None
            if time.monotonic() - began >= STABLE_AFTER_S:
                backoff = BACKOFF_START
            delay = max(extra_delay, backoff * (0.75 + random.random() * 0.5))
            backoff = min(backoff * 2, BACKOFF_MAX)
            if self._stop.is_set():
                return
            await sleep(delay)

    async def _connect_once(self) -> None:
        connect = self._connect
        if connect is None:
            import websockets  # type: ignore
            connect = websockets.connect
        base = self._resume_url if (self._session_id and self._resume_url) else GATEWAY_URL
        async with connect(base.rstrip("/") + "/" + GATEWAY_QUERY,
                           max_size=None, open_timeout=20) as ws:
            try:
                await self.session(ws)
            finally:
                code = getattr(ws, "close_code", None)
                if code in FATAL_CLOSE_CODES:
                    raise _Fatal(FATAL_CLOSE_CODES[code])

    # -- one connection --
    async def session(self, ws) -> None:
        """Drive one open websocket until it ends (raises _Reconnect /
        _Fatal / a connection error)."""
        hello = json.loads(await ws.recv())
        if hello.get("op") != 10:
            raise _Reconnect(resume=False)
        interval = float((hello.get("d") or {}).get("heartbeat_interval", 41250)) / 1000.0
        self._acked = True
        hb = asyncio.ensure_future(self._heartbeat_loop(ws, interval))
        try:
            if self._session_id and self._seq is not None:
                await ws.send(json.dumps({"op": 6, "d": {
                    "token": self._token_fn(), "session_id": self._session_id,
                    "seq": self._seq}}))
            else:
                await ws.send(json.dumps({"op": 2, "d": {
                    "token": self._token_fn(), "intents": INTENTS,
                    "properties": {"os": "windows", "browser": "jarvis",
                                   "device": "jarvis"}}}))
            while not self._stop.is_set():
                raw = await ws.recv()
                await self.handle_payload(json.loads(raw), ws)
        finally:
            hb.cancel()

    async def _heartbeat_loop(self, ws, interval: float) -> None:
        try:
            await asyncio.sleep(interval * random.random())
            while True:
                if not self._acked:
                    # The last heartbeat was never acknowledged: the link is
                    # a zombie. 4000 keeps the session resumable.
                    await ws.close(code=4000)
                    return
                self._acked = False
                await ws.send(json.dumps({"op": 1, "d": self._seq}))
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def handle_payload(self, p: dict, ws=None) -> None:
        op = p.get("op")
        if p.get("s") is not None:
            self._seq = p["s"]
        if op == 0:
            self._on_dispatch(p.get("t"), p.get("d") or {})
        elif op == 1 and ws is not None:
            await ws.send(json.dumps({"op": 1, "d": self._seq}))
        elif op == 7:
            raise _Reconnect(resume=True)
        elif op == 9:
            raise _Reconnect(resume=bool(p.get("d")), delay=random.uniform(1.0, 5.0))
        elif op == 11:
            self._acked = True

    def _on_dispatch(self, t: Optional[str], d: dict) -> None:
        if t == "READY":
            self._session_id = d.get("session_id")
            self._resume_url = d.get("resume_gateway_url")
            self._bot_id = str((d.get("user") or {}).get("id") or "") or None
            with _lock:
                _status["connected"] = True
                _status["last_connect_at"] = time.time()
                _status["last_error"] = None
            print("  [discord] gateway connected - listening for owner DMs")
        elif t == "RESUMED":
            with _lock:
                _status["connected"] = True
                _status["last_connect_at"] = time.time()
        elif t == "MESSAGE_CREATE":
            self.on_message_create(d)

    # -- the security gate --
    def on_message_create(self, d: dict) -> bool:
        """Accept a MESSAGE_CREATE only if it is a plain DM from the owner.
        Returns True when it was queued for a worker."""
        author = d.get("author") or {}
        uid = str(author.get("id") or "")
        owner = self._owner_fn()
        mid = str(d.get("id") or "")
        is_owner_dm = bool(
            owner and uid == owner
            and not author.get("bot")
            and not d.get("guild_id")            # DMs only, never a server channel
            and uid != (self._bot_id or ""))
        if not is_owner_dm:
            with _lock:
                _status["ignored"] += 1
            _log.info("[discord] ignored a message that is not an owner DM")
            return False
        if mid:
            if mid in self._seen_ids:            # a RESUME can replay events
                return False
            self._seen_ids.append(mid)
            del self._seen_ids[:-50]
        text = str(d.get("content") or "").strip()
        if not text:                             # attachment-only / embed
            return False
        with _lock:
            _status["last_inbound_at"] = time.time()
            _status["messages_in"] += 1
        try:
            self._work.put_nowait(text)
        except queue.Full:
            self._send("I'm still working through your earlier messages, "
                       "sir. Try again in a moment.")
            return False
        return True

    def process_owner_text(self, text: str) -> None:
        """Run one command and DM the reply. Never raises."""
        try:
            reply = self._handler(text)
        except Exception as e:
            _log.warning("[discord] handler crashed: %s", _scrub(e))
            reply = f"That crashed on my side, sir ({type(e).__name__})."
        if reply and str(reply).strip():
            self._send(str(reply))


class _Fatal(Exception):
    """Retrying cannot help (bad token, intent not enabled)."""
