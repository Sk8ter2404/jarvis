"""Hermetic tests for core/discord_bridge.py (the Discord owner-DM bridge).

No network, no sockets, no real websocket: REST goes through a fake
requests-shaped object, the gateway through a scripted fake websocket, and the
reconnect loop's sleeps are injected. stdlib unittest + mock only.
"""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest import mock

from core import discord_bridge as db

# A bot-token-SHAPED value built at run time (no secret-looking literal in the
# source for tools/check_no_pii.py to flag).
_TOKEN = ".".join(("A" * 24, "b" * 6, "C" * 27))
_OWNER = "1000000000000000001"
_STRANGER = "1000000000000000002"
_ENV = {"DISCORD_BOT_TOKEN": _TOKEN, "DISCORD_OWNER_ID": _OWNER}


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body if body is not None else {}

    def json(self):
        return self._body


class FakeHttp:
    """requests-shaped: .request(method, url, headers=, json=, timeout=)."""

    def __init__(self, *scripted):
        self.calls = []
        self._scripted = list(scripted)

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if self._scripted:
            item = self._scripted.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if url.endswith("/users/@me/channels"):
            return _Resp(200, {"id": "555"})
        return _Resp(200, {"id": "999"})


class Base(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(os.environ, _ENV, clear=True)
        p.start()
        self.addCleanup(p.stop)
        db._dm_channel[0] = None
        with db._lock:
            for k, v in (("messages_in", 0), ("messages_out", 0),
                         ("ignored", 0), ("reconnects", 0)):
                db._status[k] = v
            db._status["fatal"] = None
            db._status["connected"] = False
            db._status["last_error"] = None


class ConfigTests(Base):
    def test_configured_needs_both(self):
        self.assertTrue(db.configured())
        with mock.patch.dict(os.environ, {"DISCORD_OWNER_ID": ""}):
            self.assertFalse(db.configured())
        with mock.patch.dict(os.environ, {"DISCORD_BOT_TOKEN": ""}):
            self.assertFalse(db.configured())

    def test_malformed_owner_id_means_nobody(self):
        with mock.patch.dict(os.environ, {"DISCORD_OWNER_ID": "someone#1234"}):
            self.assertEqual(db.owner_id(), "")
            self.assertFalse(db.configured())

    def test_scrub_removes_the_token_and_token_shapes(self):
        self.assertNotIn(_TOKEN, db._scrub(f"boom {_TOKEN} boom"))
        other = ".".join(("Z" * 25, "y" * 7, "X" * 30))
        self.assertNotIn(other, db._scrub(other))

    def test_split_message(self):
        self.assertEqual(db.split_message("  "), [])
        self.assertEqual(db.split_message("hi"), ["hi"])
        long = ("word " * 1000).strip()
        parts = db.split_message(long)
        self.assertTrue(all(len(p) <= db.MAX_MESSAGE_LEN for p in parts))
        self.assertEqual(" ".join(parts), long)
        blob = "x" * 5000
        self.assertEqual("".join(db.split_message(blob)), blob)


class SendTests(Base):
    def test_not_configured_makes_no_call(self):
        http = FakeHttp()
        with mock.patch.dict(os.environ, {"DISCORD_BOT_TOKEN": ""}):
            self.assertFalse(db.send_dm("hello", http=http))
        self.assertEqual(http.calls, [])

    def test_opens_dm_channel_once_then_posts(self):
        http = FakeHttp()
        self.assertTrue(db.send_dm("first", http=http))
        self.assertTrue(db.send_dm("second", http=http))
        methods = [(m, u.rsplit("/v10", 1)[1]) for m, u, _ in http.calls]
        self.assertEqual(methods, [("POST", "/users/@me/channels"),
                                   ("POST", "/channels/555/messages"),
                                   ("POST", "/channels/555/messages")])
        self.assertEqual(http.calls[0][2]["json"], {"recipient_id": _OWNER})
        self.assertEqual(http.calls[1][2]["json"], {"content": "first"})
        self.assertEqual(db.status()["messages_out"], 2)

    def test_token_only_in_auth_header_never_url_or_body(self):
        http = FakeHttp()
        db.send_dm("hello", http=http)
        for _m, url, kw in http.calls:
            self.assertNotIn(_TOKEN, url)
            self.assertNotIn(_TOKEN, json.dumps(kw["json"]))
            self.assertEqual(kw["headers"]["Authorization"], f"Bot {_TOKEN}")

    def test_long_text_is_chunked(self):
        http = FakeHttp()
        self.assertTrue(db.send_dm("word " * 1000, http=http))
        posts = [c for c in http.calls if "/messages" in c[1]]
        self.assertGreater(len(posts), 1)
        self.assertTrue(all(len(c[2]["json"]["content"]) <= 2000 for c in posts))

    def test_rate_limit_is_retried_once_after_waiting(self):
        http = FakeHttp(_Resp(200, {"id": "555"}),
                        _Resp(429, {"retry_after": 0.5}), _Resp(200, {"id": "1"}))
        waits = []
        with mock.patch.object(db.time, "sleep", waits.append):
            self.assertTrue(db.send_dm("hi", http=http))
        self.assertEqual(waits, [0.5])

    def test_rate_limit_wait_is_capped(self):
        http = FakeHttp(_Resp(200, {"id": "555"}),
                        _Resp(429, {"retry_after": 9999}), _Resp(200))
        waits = []
        with mock.patch.object(db.time, "sleep", waits.append):
            db.send_dm("hi", http=http)
        self.assertEqual(waits, [db.MAX_RATE_LIMIT_WAIT])

    def test_forbidden_clears_the_cached_channel_and_fails(self):
        http = FakeHttp(_Resp(200, {"id": "555"}), _Resp(403))
        self.assertFalse(db.send_dm("hi", http=http))
        self.assertIsNone(db._dm_channel[0])
        self.assertIn("403", db.status()["last_error"])

    def test_channel_open_failure_and_network_error_return_false(self):
        self.assertFalse(db.send_dm("hi", http=FakeHttp(_Resp(401))))
        err = RuntimeError(f"connection died for {_TOKEN}")
        self.assertFalse(db.send_dm("hi", http=FakeHttp(err)))
        self.assertNotIn(_TOKEN, db.status()["last_error"])

    def test_never_raises_on_a_broken_http_object(self):
        self.assertFalse(db.send_dm("hi", http=object()))


class FakeWS:
    """Scripted websocket. `incoming` items are dicts (sent as JSON) or
    Exceptions (raised); an exhausted script raises ConnectionError."""

    def __init__(self, incoming, close_code=None):
        self._in = list(incoming)
        self.sent = []
        self.closed_with = None
        self.close_code = close_code

    async def recv(self):
        if not self._in:
            raise ConnectionError("script over")
        item = self._in.pop(0)
        if isinstance(item, Exception):
            raise item
        return json.dumps(item)

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def close(self, code=1000):
        self.closed_with = code


def _hello(interval_ms=10_000_000):
    return {"op": 10, "d": {"heartbeat_interval": interval_ms}}


def _ready(sid="sess1", user="777"):
    return {"op": 0, "s": 1, "t": "READY",
            "d": {"session_id": sid, "resume_gateway_url": "wss://resume.example",
                  "user": {"id": user}}}


def _msg(author=_OWNER, content="play music", guild=None, bot=False, mid="m1", seq=2):
    d = {"id": mid, "content": content, "channel_id": "555",
         "author": {"id": author, "bot": bot}}
    if guild:
        d["guild_id"] = guild
    return {"op": 0, "s": seq, "t": "MESSAGE_CREATE", "d": d}


def _client(handler=None, sent=None):
    sent = sent if sent is not None else []
    c = db.GatewayClient(handler or (lambda t: "ok: " + t),
                         sender=lambda t: sent.append(t) or True)
    return c, sent


def run(coro):
    return asyncio.run(coro)


class GateTests(Base):
    def test_owner_dm_is_queued(self):
        c, _ = _client()
        self.assertTrue(c.on_message_create(_msg()["d"]))
        self.assertEqual(c._work.get_nowait(), "play music")
        self.assertEqual(db.status()["messages_in"], 1)

    def test_everyone_else_is_ignored_silently(self):
        c, sent = _client()
        c._bot_id = "777"
        cases = [
            _msg(author=_STRANGER)["d"],                 # another user
            _msg(guild="42")["d"],                       # owner, but a server channel
            _msg(bot=True)["d"],                         # a bot
            _msg(author="777")["d"],                     # the bot's own echo
            {"id": "x", "content": "hi", "author": {}},  # no author id
        ]
        for d in cases:
            self.assertFalse(c.on_message_create(d), d)
        self.assertTrue(c._work.empty())
        self.assertEqual(sent, [])                       # no reply at all
        self.assertEqual(db.status()["ignored"], len(cases))
        self.assertEqual(db.status()["messages_in"], 0)

    def test_unset_owner_id_admits_nobody(self):
        c = db.GatewayClient(lambda t: "x", sender=lambda t: True,
                             owner_fn=lambda: "")
        self.assertFalse(c.on_message_create(_msg()["d"]))

    def test_blank_and_replayed_messages_are_dropped(self):
        c, _ = _client()
        self.assertFalse(c.on_message_create(_msg(content="   ")["d"]))
        self.assertTrue(c.on_message_create(_msg(mid="dup")["d"]))
        self.assertFalse(c.on_message_create(_msg(mid="dup")["d"]))
        self.assertEqual(c._work.qsize(), 1)

    def test_full_queue_tells_the_owner_instead_of_dropping_silently(self):
        c, sent = _client()
        for i in range(db.WORK_QUEUE_MAX):
            self.assertTrue(c.on_message_create(_msg(mid=f"q{i}")["d"]))
        self.assertFalse(c.on_message_create(_msg(mid="over")["d"]))
        self.assertEqual(len(sent), 1)
        self.assertIn("sir", sent[0])


class ProcessTests(Base):
    def test_reply_is_sent_back(self):
        c, sent = _client(lambda t: f"done {t}")
        c.process_owner_text("lights on")
        self.assertEqual(sent, ["done lights on"])

    def test_handler_crash_is_contained_and_reported(self):
        def boom(_t):
            raise ValueError("secret detail")
        c, sent = _client(boom)
        c.process_owner_text("x")
        self.assertEqual(len(sent), 1)
        self.assertIn("ValueError", sent[0])
        self.assertNotIn("secret detail", sent[0])

    def test_empty_reply_sends_nothing(self):
        c, sent = _client(lambda t: "  ")
        c.process_owner_text("x")
        self.assertEqual(sent, [])


class SessionTests(Base):
    def test_identify_then_ready_then_owner_dm(self):
        c, _ = _client()
        ws = FakeWS([_hello(), _ready(), _msg(), ConnectionError("closed")])
        with mock.patch("builtins.print"), self.assertRaises(ConnectionError):
            run(c.session(ws))
        ident = ws.sent[0]
        self.assertEqual(ident["op"], 2)
        self.assertEqual(ident["d"]["intents"],
                         db.INTENT_DIRECT_MESSAGES | db.INTENT_MESSAGE_CONTENT)
        self.assertEqual(ident["d"]["token"], _TOKEN)
        self.assertEqual(c._session_id, "sess1")
        self.assertEqual(c._resume_url, "wss://resume.example")
        self.assertEqual(c._work.get_nowait(), "play music")
        self.assertTrue(db.status()["connected"])

    def test_second_connection_resumes_the_session(self):
        c, _ = _client()
        with mock.patch("builtins.print"), self.assertRaises(ConnectionError):
            run(c.session(FakeWS([_hello(), _ready(), _msg(), ConnectionError()])))
        ws2 = FakeWS([_hello(), ConnectionError()])
        with self.assertRaises(ConnectionError):
            run(c.session(ws2))
        self.assertEqual(ws2.sent[0]["op"], 6)
        self.assertEqual(ws2.sent[0]["d"]["session_id"], "sess1")
        self.assertEqual(ws2.sent[0]["d"]["seq"], 2)

    def test_reconnect_op_and_invalid_session(self):
        c, _ = _client()
        with self.assertRaises(db._Reconnect) as cm:
            run(c.session(FakeWS([_hello(), {"op": 7}])))
        self.assertTrue(cm.exception.resume)
        with self.assertRaises(db._Reconnect) as cm:
            run(c.session(FakeWS([_hello(), {"op": 9, "d": False}])))
        self.assertFalse(cm.exception.resume)
        self.assertGreaterEqual(cm.exception.delay, 1.0)

    def test_missing_hello_forces_a_fresh_session(self):
        c, _ = _client()
        with self.assertRaises(db._Reconnect) as cm:
            run(c.session(FakeWS([{"op": 0, "t": "X"}])))
        self.assertFalse(cm.exception.resume)

    def test_server_heartbeat_request_is_answered(self):
        c, _ = _client()
        ws = FakeWS([_hello(), {"op": 1}, ConnectionError()])
        with self.assertRaises(ConnectionError):
            run(c.session(ws))
        self.assertTrue(any(m["op"] == 1 for m in ws.sent))

    def test_unacknowledged_heartbeat_closes_with_a_resumable_code(self):
        c, _ = _client()
        ws = FakeWS([])
        run(c._heartbeat_loop(ws, 0.01))
        self.assertEqual(ws.closed_with, 4000)
        self.assertEqual([m["op"] for m in ws.sent], [1])

    def test_ack_keeps_the_link_alive(self):
        c, _ = _client()

        class AckingWS(FakeWS):
            async def send(self, raw):
                await super().send(raw)
                c._acked = True          # the gateway answers every heartbeat

        ws = AckingWS([])

        async def go():
            task = asyncio.ensure_future(c._heartbeat_loop(ws, 0.01))
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        run(go())
        self.assertIsNone(ws.closed_with)
        self.assertGreaterEqual(len(ws.sent), 2)


class _CM:
    def __init__(self, ws):
        self.ws = ws

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, *a):
        return False


class LoopTests(Base):
    def test_reconnects_with_growing_backoff_then_stops(self):
        c, _ = _client()
        urls = []
        sleeps = []

        def connect(url, **kw):
            urls.append(url)
            return _CM(FakeWS([_hello(), ConnectionError("down")]))

        c._connect = connect

        async def fake_sleep(d):
            sleeps.append(d)
            if len(sleeps) == 4:
                c.stop()

        run(c.run_forever(sleep=fake_sleep))
        self.assertEqual(len(urls), 4)
        self.assertTrue(all(u.startswith("wss://gateway.discord.gg/") for u in urls))
        self.assertTrue(all(u.endswith("v=10&encoding=json") for u in urls))
        # exponential: each base doubles, jitter is +-25%
        for i in range(1, len(sleeps)):
            self.assertGreater(sleeps[i], sleeps[i - 1] * 1.2)
        self.assertLessEqual(max(sleeps), db.BACKOFF_MAX * 1.25)
        self.assertEqual(db.status()["reconnects"], 4)
        self.assertFalse(db.status()["connected"])

    def test_resume_uses_the_resume_url(self):
        c, _ = _client()
        urls = []
        scripts = [[_hello(), _ready(), ConnectionError()],
                   [_hello(), ConnectionError()]]

        def connect(url, **kw):
            urls.append(url)
            return _CM(FakeWS(scripts.pop(0)))
        c._connect = connect
        n = []

        async def fake_sleep(d):
            n.append(d)
            if len(n) == 2:
                c.stop()
        with mock.patch("builtins.print"):
            run(c.run_forever(sleep=fake_sleep))
        self.assertTrue(urls[0].startswith("wss://gateway.discord.gg/"))
        self.assertTrue(urls[1].startswith("wss://resume.example/"))

    def test_bad_token_close_code_is_fatal_and_not_retried(self):
        c, _ = _client()
        calls = []

        def connect(url, **kw):
            calls.append(url)
            return _CM(FakeWS([_hello(), ConnectionError("closed")], close_code=4004))
        c._connect = connect

        async def never(_d):
            raise AssertionError("must not back off and retry")
        with mock.patch("builtins.print"):
            run(c.run_forever(sleep=never))
        self.assertEqual(len(calls), 1)
        self.assertIn("token", db.status()["fatal"])

    def test_missing_intent_close_code_names_the_fix(self):
        c, _ = _client()
        c._connect = lambda url, **kw: _CM(
            FakeWS([_hello(), ConnectionError()], close_code=4014))

        async def never(_d):
            raise AssertionError("must not retry")
        with mock.patch("builtins.print"):
            run(c.run_forever(sleep=never))
        self.assertIn("Message Content", db.status()["fatal"])

    def test_invalid_session_discards_the_session_before_reconnecting(self):
        c, _ = _client()
        c._session_id, c._seq, c._resume_url = "old", 5, "wss://old.example"
        scripts = [[_hello(), {"op": 9, "d": False}]]

        def connect(url, **kw):
            return _CM(FakeWS(scripts.pop(0)))
        c._connect = connect

        async def fake_sleep(_d):
            c.stop()
        run(c.run_forever(sleep=fake_sleep))
        self.assertIsNone(c._session_id)
        self.assertIsNone(c._resume_url)


class StartTests(Base):
    def test_start_without_config_does_nothing(self):
        c = db.GatewayClient(lambda t: "", token_fn=lambda: "", owner_fn=lambda: "")
        with mock.patch.object(db.threading, "Thread") as th:
            self.assertFalse(c.start())
        th.assert_not_called()

    def test_start_is_idempotent_and_runs_off_the_calling_thread(self):
        c, _ = _client()
        with mock.patch.object(db.threading, "Thread") as th:
            th.return_value.is_alive.return_value = True
            self.assertTrue(c.start(initial_delay=0))
            self.assertTrue(c.start(initial_delay=0))
        names = [k.kwargs.get("name") for k in th.call_args_list]
        self.assertEqual(names.count("discord-gateway"), 1)
        self.assertTrue(all(k.kwargs.get("daemon") for k in th.call_args_list))


if __name__ == "__main__":
    unittest.main()
