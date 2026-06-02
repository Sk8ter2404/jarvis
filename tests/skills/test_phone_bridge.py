"""Logic tests for skills/phone_bridge.py.

Push to / pull from the phone over Telegram / ntfy / Pushover. Every backend is
env-configured and absent in the test env, so the no-backend degradation paths
are easy and central. We test:
  * whitelist parsing, long-message chunking, priority shorthand parsing,
  * the priority maps,
  * each _send_* turning a (mocked) requests response into a bool, plus the
    requests-missing / exception / pushover-urgent / title branches,
  * push_to_phone's confirm-gate fail-closed behaviour + per-backend fan-out,
    the telegram leg, the no-backend console-log leg, and the success
    bookkeeping (messages_out / last_outbound_at),
  * the stateless _llm_fallback (anthropic happy path, no-key, import-fail,
    Ollama fallback, total failure),
  * _dispatch_remote's full resolution ladder (mode toggle, chain, single
    intent — success / empty-return / raising — and the LLM fallback),
  * _handle_slash routing (help/status/pause/resume/mode + fall-through),
  * _process_update edge paths (non-int chat_id, unauthorised, slash/dispatch
    crash isolation),
  * the Telegram long-poll worker (_telegram_polling_loop) drained for one
    batch with a faked getUpdates, plus its no-token / no-whitelist / HTTP-error
    / not-ok / requests-missing guards,
  * _start_polling_thread idempotence,
  * _load_state / _save_state round-trip against a temp file,
  * the registered voice actions (notify/status/list/pause/resume) including
    the configured-backend rendering branches,
  * register() wiring.

requests / anthropic / bobert_companion / core.dispatcher / core.mode_router are
faked per-test and removed in tearDown; no sockets, no Telegram, no real disk
writes (state save mocked or redirected to a temp dir), no threads, no sleeps.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_NO_ENV = {}  # used with clear=True to guarantee "no backend configured"
_SENTINEL = object()


# ─── module-injection helpers (mirror tests/skills/test_self_diagnostic.py) ──
@contextlib.contextmanager
def inject_modules(**mods):
    """Temporarily install fake modules into sys.modules (requests, anthropic,
    bobert_companion, core.dispatcher, ...). For dotted names the leaf is ALSO
    set as an attribute on its already-imported parent package, because
    ``from core import dispatcher`` resolves the leaf via getattr(parent, leaf)
    when the parent is a real package. Restores prior state — including absence
    — on exit so each test is isolated. Keys may be passed via ``**{"a.b": o}``.
    """
    saved_mod: dict[str, object] = {}
    missing: set[str] = set()
    saved_attr: list = []
    for name, obj in mods.items():
        saved_mod[name] = sys.modules.get(name, _SENTINEL)
        if saved_mod[name] is _SENTINEL:
            missing.add(name)
        if obj is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = obj
            if "." in name:
                parent_name, _, leaf = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    saved_attr.append(
                        (parent, leaf, getattr(parent, leaf, _SENTINEL)))
                    setattr(parent, leaf, obj)
    try:
        yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            if prev is _SENTINEL:
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
            else:
                setattr(parent, leaf, prev)
        for name in mods:
            prev = saved_mod.get(name, _SENTINEL)
            if name in missing:
                sys.modules.pop(name, None)
            elif prev is not _SENTINEL:
                sys.modules[name] = prev


@contextlib.contextmanager
def block_import(*names):
    """Force ``import <name>`` to raise ImportError inside the with-block so a
    skill's missing-dependency branch is exercised even when the real dep is
    installed on the dev box. Also detaches an already-imported target from
    sys.modules (and its parent-package attr) for the duration so the import
    machinery can't satisfy it from cache, then restores both."""
    real_import = __import__
    blocked = set(names)

    def _fake_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if name in blocked or top in blocked:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    saved_mod: dict[str, object] = {}
    saved_attr: list = []
    for name in blocked:
        if name in sys.modules:
            saved_mod[name] = sys.modules.pop(name)
        if "." in name:
            parent_name, _, leaf = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and hasattr(parent, leaf):
                saved_attr.append((parent, leaf, getattr(parent, leaf)))
                try:
                    delattr(parent, leaf)
                except AttributeError:
                    pass
    try:
        with mock.patch("builtins.__import__", side_effect=_fake_import):
            yield
    finally:
        for parent, leaf, prev in reversed(saved_attr):
            setattr(parent, leaf, prev)
        for name, mod in saved_mod.items():
            sys.modules[name] = mod


class _Resp:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


def _fake_requests(resp=None, raise_exc=None):
    """A requests stand-in. Uses a real ModuleType (not MagicMock) so that
    ``import requests`` inside the skill binds it cleanly and .post/.get are
    plain callables we can assert on."""
    mod = types.ModuleType("requests")
    if raise_exc is not None:
        mod.post = mock.MagicMock(side_effect=raise_exc)
        mod.get = mock.MagicMock(side_effect=raise_exc)
    else:
        mod.post = mock.MagicMock(return_value=resp or _Resp(200))
        mod.get = mock.MagicMock(return_value=resp or _Resp(200))
    return mod


class WhitelistTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def test_empty_is_empty_set(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            self.assertEqual(self.mod._telegram_whitelist(), set())

    def test_comma_and_semicolon_separated(self):
        with mock.patch.dict(os.environ, {"TELEGRAM_USER_ID": "111, 222; 333"}, clear=True):
            self.assertEqual(self.mod._telegram_whitelist(), {111, 222, 333})

    def test_skips_non_integer_entries(self):
        with mock.patch.dict(os.environ, {"TELEGRAM_USER_ID": "111, bad, 222"}, clear=True):
            self.assertEqual(self.mod._telegram_whitelist(), {111, 222})

    def test_skips_empty_pieces_from_trailing_separators(self):
        # Trailing / doubled separators yield empty pieces that are skipped.
        with mock.patch.dict(os.environ, {"TELEGRAM_USER_ID": "111,,222,"}, clear=True):
            self.assertEqual(self.mod._telegram_whitelist(), {111, 222})


class SplitLongTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def test_short_message_single_chunk(self):
        self.assertEqual(self.mod._split_long("hello", 100), ["hello"])

    def test_long_message_splits_and_reassembles(self):
        msg = "word " * 500  # ~2500 chars
        chunks = self.mod._split_long(msg, 1000)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 1000)
        # No content lost (modulo whitespace at split points).
        self.assertEqual("".join(chunks).replace(" ", ""), msg.replace(" ", ""))

    def test_prefers_newline_boundary(self):
        msg = "a" * 40 + "\n" + "b" * 40
        chunks = self.mod._split_long(msg, 50)
        self.assertEqual(chunks[0], "a" * 40)   # cut at the newline

    def test_hard_cut_when_no_whitespace(self):
        # One unbroken token longer than the cap: no newline, no space in the
        # first half → hard-cut exactly at the limit.
        msg = "x" * 250
        chunks = self.mod._split_long(msg, 100)
        self.assertEqual(chunks[0], "x" * 100)
        self.assertEqual("".join(chunks), msg)
        for c in chunks:
            self.assertLessEqual(len(c), 100)


class PriorityArgTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def test_plain_message_normal_priority(self):
        self.assertEqual(self.mod._format_priority_arg("build finished"),
                         ("build finished", "normal"))

    def test_bang_urgent_shorthand(self):
        self.assertEqual(self.mod._format_priority_arg("!urgent build is red"),
                         ("build is red", "urgent"))

    def test_bang_high_shorthand(self):
        msg, prio = self.mod._format_priority_arg("!high disk almost full")
        self.assertEqual((msg, prio), ("disk almost full", "high"))

    def test_unknown_bang_token_left_intact(self):
        # '!banana' isn't a known priority → not stripped, stays normal.
        msg, prio = self.mod._format_priority_arg("!banana hello")
        self.assertEqual(prio, "normal")
        self.assertIn("banana", msg)


class PriorityMapTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def test_ntfy_map(self):
        self.assertEqual(self.mod._NTFY_PRIORITY_MAP["urgent"], "urgent")
        self.assertEqual(self.mod._NTFY_PRIORITY_MAP["normal"], "default")

    def test_pushover_map_levels(self):
        self.assertEqual(self.mod._PUSHOVER_PRIORITY_MAP["low"], -1)
        self.assertEqual(self.mod._PUSHOVER_PRIORITY_MAP["urgent"], 2)


class SendBackendTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def test_send_telegram_success(self):
        with mock.patch.dict(os.environ,
                             {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_USER_ID": "42"},
                             clear=True), \
             mock.patch.dict(sys.modules, {"requests": _fake_requests(_Resp(200))}):
            self.assertTrue(self.mod._send_telegram("hi"))

    def test_send_telegram_http_error_returns_false(self):
        with mock.patch.dict(os.environ,
                             {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_USER_ID": "42"},
                             clear=True), \
             mock.patch.dict(sys.modules,
                             {"requests": _fake_requests(_Resp(429, "rate limited"))}):
            self.assertFalse(self.mod._send_telegram("hi"))

    def test_send_telegram_no_token(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            self.assertFalse(self.mod._send_telegram("hi"))

    def test_send_telegram_no_whitelist_cannot_send(self):
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t"}, clear=True):
            self.assertFalse(self.mod._send_telegram("hi"))   # chat_id unresolved

    def test_send_ntfy_success(self):
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "secret-topic"}, clear=True), \
             mock.patch.dict(sys.modules, {"requests": _fake_requests(_Resp(200))}):
            self.assertTrue(self.mod._send_ntfy("hi", title="JARVIS", priority="high"))

    def test_send_ntfy_4xx_returns_false(self):
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.dict(sys.modules, {"requests": _fake_requests(_Resp(403))}):
            self.assertFalse(self.mod._send_ntfy("hi"))

    def test_send_pushover_success(self):
        with mock.patch.dict(os.environ,
                             {"PUSHOVER_TOKEN": "tok", "PUSHOVER_USER": "usr"},
                             clear=True), \
             mock.patch.dict(sys.modules, {"requests": _fake_requests(_Resp(200))}):
            self.assertTrue(self.mod._send_pushover("hi"))

    def test_send_pushover_not_configured(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            self.assertFalse(self.mod._send_pushover("hi"))


class PushToPhoneTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def test_empty_message_returns_empty(self):
        self.assertEqual(self.mod.push_to_phone("   ", confirm=False), {})

    def test_confirm_gate_unavailable_fails_closed(self):
        # draft_confirm is None → with confirm=True the push is dropped.
        with mock.patch.object(self.mod, "draft_confirm", None):
            out = self.mod.push_to_phone("secret message", confirm=True)
        self.assertEqual(out, {})

    def test_confirm_denied_drops(self):
        with mock.patch.object(self.mod, "draft_confirm", return_value=False):
            out = self.mod.push_to_phone("msg", confirm=True)
        self.assertEqual(out, {})

    def test_fans_out_to_configured_backends(self):
        with mock.patch.dict(os.environ,
                             {"NTFY_TOPIC": "t", "PUSHOVER_TOKEN": "k",
                              "PUSHOVER_USER": "u"}, clear=True), \
             mock.patch.object(self.mod, "_send_ntfy", return_value=True) as ntfy, \
             mock.patch.object(self.mod, "_send_pushover", return_value=True) as push, \
             mock.patch.object(self.mod, "_save_state"):
            out = self.mod.push_to_phone("hi", confirm=False)
        self.assertEqual(out, {"ntfy": True, "pushover": True})
        ntfy.assert_called_once()
        push.assert_called_once()

    def test_backends_filter_restricts(self):
        with mock.patch.dict(os.environ,
                             {"NTFY_TOPIC": "t", "PUSHOVER_TOKEN": "k",
                              "PUSHOVER_USER": "u"}, clear=True), \
             mock.patch.object(self.mod, "_send_ntfy", return_value=True), \
             mock.patch.object(self.mod, "_send_pushover", return_value=True) as push, \
             mock.patch.object(self.mod, "_save_state"):
            out = self.mod.push_to_phone("hi", confirm=False, backends=["ntfy"])
        self.assertIn("ntfy", out)
        self.assertNotIn("pushover", out)
        push.assert_not_called()


class ProcessUpdateTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def _update(self, user_id, chat_id, text):
        return {"update_id": 1, "message": {
            "text": text, "chat": {"id": chat_id}, "from": {"id": user_id}}}

    def test_unauthorised_user_rejected(self):
        sent = []
        with mock.patch.object(self.mod, "_send_telegram",
                               side_effect=lambda msg, **kw: sent.append((msg, kw)) or True), \
             mock.patch.object(self.mod, "_dispatch_remote") as disp:
            self.mod._process_update(self._update(999, 999, "hello"), whitelist={42})
        # The dispatcher is never reached for an unauthorised sender.
        disp.assert_not_called()
        self.assertTrue(any("Unauthorised" in m for m, _ in sent))

    def test_authorised_user_dispatched_and_replied(self):
        with mock.patch.object(self.mod, "_dispatch_remote",
                               return_value="Playing lo-fi, sir.") as disp, \
             mock.patch.object(self.mod, "_send_telegram", return_value=True) as send:
            self.mod._process_update(self._update(42, 42, "play lo-fi"), whitelist={42})
        disp.assert_called_once_with("play lo-fi")
        # Reply routed back to the originating chat.
        self.assertEqual(send.call_args.kwargs.get("chat_id"), 42)
        self.assertIn("lo-fi", send.call_args[0][0])

    def test_slash_command_routed_to_handler(self):
        with mock.patch.object(self.mod, "_handle_slash",
                               return_value="help text") as slash, \
             mock.patch.object(self.mod, "_send_telegram", return_value=True):
            self.mod._process_update(self._update(42, 42, "/help"), whitelist={42})
        slash.assert_called_once_with("/help")

    def test_empty_text_ignored(self):
        with mock.patch.object(self.mod, "_dispatch_remote") as disp:
            self.mod._process_update(self._update(42, 42, ""), whitelist={42})
        disp.assert_not_called()


class HandleSlashTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def test_help(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            out = self.mod._handle_slash("/help")
        self.assertIn("phone bridge", out.lower())
        self.assertIn("/status", out)

    def test_status_routes(self):
        with mock.patch.object(self.mod, "phone_bridge_status",
                               return_value="status-string") as st:
            self.assertEqual(self.mod._handle_slash("/status"), "status-string")
        st.assert_called_once()

    def test_strips_botname_suffix(self):
        with mock.patch.object(self.mod, "pause_phone_bridge",
                               return_value="paused") as pause:
            self.mod._handle_slash("/pause@JarvisBot")
        pause.assert_called_once()

    def test_unknown_slash_falls_through_to_dispatch(self):
        with mock.patch.object(self.mod, "_dispatch_remote",
                               return_value="dispatched") as disp:
            out = self.mod._handle_slash("/wibble some args")
        self.assertEqual(out, "dispatched")
        disp.assert_called_once_with("wibble some args")


class DispatchRemoteTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def test_empty_message(self):
        self.assertIn("Empty message", self.mod._dispatch_remote(""))

    def test_boot_order_guard_when_actions_unset(self):
        self.mod._actions_ref[0] = None
        out = self.mod._dispatch_remote("play lo-fi")
        self.assertIn("not fully booted", out)


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("phone_bridge")

    def test_notify_phone_no_backends(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            out = self.actions["notify_phone"]("print finished")
        self.assertIn("No phone backends configured", out)
        self.assertIn("TELEGRAM_BOT_TOKEN", out)

    def test_notify_phone_empty_message(self):
        out = self.actions["notify_phone"]("")
        self.assertIn("Pass the message", out)

    def test_notify_phone_success_summary(self):
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.object(self.mod, "push_to_phone", return_value={"ntfy": True}):
            out = self.actions["notify_phone"]("build is green")
        self.assertIn("sent to ntfy", out.lower())

    def test_notify_phone_partial_failure(self):
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.object(self.mod, "push_to_phone",
                               return_value={"ntfy": True, "pushover": False}):
            out = self.actions["notify_phone"]("msg")
        self.assertIn("ntfy", out)
        self.assertIn("pushover", out)
        self.assertIn("failed", out)

    def test_notify_phone_all_backends_fail(self):
        # Every configured backend returns False (or the gate dropped it →
        # empty dict): neither sent-only nor partial branch applies.
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.object(self.mod, "push_to_phone",
                               return_value={"ntfy": False}):
            out = self.actions["notify_phone"]("msg")
        self.assertIn("Send failed on every configured backend", out)

    def test_notify_phone_strips_priority_shorthand(self):
        # The '!urgent ' prefix is consumed; the remaining body is forwarded.
        captured = {}
        def _fake_push(message, **kw):
            captured["message"] = message
            captured["priority"] = kw.get("priority")
            return {"ntfy": True}
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.object(self.mod, "push_to_phone", side_effect=_fake_push):
            self.actions["notify_phone"]("!urgent build is red")
        self.assertEqual(captured["message"], "build is red")
        self.assertEqual(captured["priority"], "urgent")

    def test_list_backends_all_unconfigured(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            out = self.actions["list_phone_backends"]("")
        self.assertIn("telegram: NOT configured", out)
        self.assertIn("ntfy: NOT configured", out)
        self.assertIn("pushover: NOT configured", out)

    def test_pause_then_resume_toggles_flag(self):
        out_p = self.actions["pause_phone_bridge"]("")
        self.assertIn("paused", out_p)
        self.assertTrue(self.mod._pause_flag[0])
        # resume tries to (re)start the poll thread; harness neutered start,
        # and with no token _start_polling_thread is a no-op anyway.
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            out_r = self.actions["resume_phone_bridge"]("")
        self.assertIn("resumed", out_r)
        self.assertFalse(self.mod._pause_flag[0])

    def test_status_string_shape(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            out = self.actions["phone_bridge_status"]("")
        self.assertIn("Phone bridge", out)
        self.assertTrue(out.rstrip().endswith("sir."))
        self.assertIn("telegram not configured", out)


class SendBackendEdgeTests(unittest.TestCase):
    """The requests-missing, exception, and metadata branches of each sender."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    # ── _send_telegram ────────────────────────────────────────────────────
    def test_send_telegram_requests_missing(self):
        with mock.patch.dict(os.environ,
                             {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_USER_ID": "42"},
                             clear=True), \
             block_import("requests"):
            self.assertFalse(self.mod._send_telegram("hi"))

    def test_send_telegram_explicit_chat_id_skips_whitelist(self):
        # chat_id supplied → no whitelist needed even with token only.
        req = _fake_requests(_Resp(200))
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t"}, clear=True), \
             inject_modules(requests=req):
            self.assertTrue(self.mod._send_telegram("hi", chat_id=7))
        body = req.post.call_args.kwargs["json"]
        self.assertEqual(body["chat_id"], 7)

    def test_send_telegram_network_exception_sets_last_error(self):
        with mock.patch.dict(os.environ,
                             {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_USER_ID": "42"},
                             clear=True), \
             inject_modules(requests=_fake_requests(raise_exc=RuntimeError("conn reset"))):
            self.assertFalse(self.mod._send_telegram("hi"))
        self.assertIn("telegram", self.mod._status["last_error"])

    def test_send_telegram_chunks_long_message(self):
        # A >MAX_MSG_LEN body is sent as 2 sequential posts; both must be 200.
        req = _fake_requests(_Resp(200))
        long_msg = "word " * 2000  # ~10k chars > TELEGRAM_MAX_MSG_LEN (4000)
        with mock.patch.dict(os.environ,
                             {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_USER_ID": "42"},
                             clear=True), \
             inject_modules(requests=req):
            self.assertTrue(self.mod._send_telegram(long_msg))
        self.assertGreaterEqual(req.post.call_count, 2)

    # ── _send_ntfy ────────────────────────────────────────────────────────
    def test_send_ntfy_requests_missing(self):
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             block_import("requests"):
            self.assertFalse(self.mod._send_ntfy("hi"))

    def test_send_ntfy_not_configured(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            self.assertFalse(self.mod._send_ntfy("hi"))

    def test_send_ntfy_title_transliterated_and_headers_set(self):
        req = _fake_requests(_Resp(200))
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "topic"}, clear=True), \
             inject_modules(requests=req):
            self.assertTrue(self.mod._send_ntfy("body", title="café", priority="urgent"))
        headers = req.post.call_args.kwargs["headers"]
        # Non-ASCII title transliterated to ASCII (the é becomes '?').
        self.assertEqual(headers["Title"], "caf?")
        self.assertEqual(headers["Priority"], "urgent")
        self.assertEqual(headers["Tags"], "robot")

    def test_send_ntfy_custom_host_used(self):
        req = _fake_requests(_Resp(200))
        with mock.patch.dict(os.environ,
                             {"NTFY_TOPIC": "topic", "NTFY_HOST": "https://n.example.com/"},
                             clear=True), \
             inject_modules(requests=req):
            self.assertTrue(self.mod._send_ntfy("body"))
        url = req.post.call_args[0][0]
        self.assertEqual(url, "https://n.example.com/topic")  # trailing / stripped

    def test_send_ntfy_network_exception_sets_last_error(self):
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             inject_modules(requests=_fake_requests(raise_exc=RuntimeError("dns fail"))):
            self.assertFalse(self.mod._send_ntfy("hi"))
        self.assertIn("ntfy", self.mod._status["last_error"])

    # ── _send_pushover ────────────────────────────────────────────────────
    def test_send_pushover_requests_missing(self):
        with mock.patch.dict(os.environ,
                             {"PUSHOVER_TOKEN": "k", "PUSHOVER_USER": "u"}, clear=True), \
             block_import("requests"):
            self.assertFalse(self.mod._send_pushover("hi"))

    def test_send_pushover_urgent_adds_retry_expire(self):
        req = _fake_requests(_Resp(200))
        with mock.patch.dict(os.environ,
                             {"PUSHOVER_TOKEN": "k", "PUSHOVER_USER": "u"}, clear=True), \
             inject_modules(requests=req):
            self.assertTrue(self.mod._send_pushover("hi", title="T", priority="urgent"))
        data = req.post.call_args.kwargs["data"]
        self.assertEqual(data["priority"], 2)
        self.assertEqual(data["retry"], 30)
        self.assertEqual(data["expire"], 3600)
        self.assertEqual(data["title"], "T")

    def test_send_pushover_http_error_sets_last_error(self):
        with mock.patch.dict(os.environ,
                             {"PUSHOVER_TOKEN": "k", "PUSHOVER_USER": "u"}, clear=True), \
             inject_modules(requests=_fake_requests(_Resp(500, "server error"))):
            self.assertFalse(self.mod._send_pushover("hi"))
        self.assertIn("pushover", self.mod._status["last_error"])

    def test_send_pushover_network_exception_sets_last_error(self):
        with mock.patch.dict(os.environ,
                             {"PUSHOVER_TOKEN": "k", "PUSHOVER_USER": "u"}, clear=True), \
             inject_modules(requests=_fake_requests(raise_exc=RuntimeError("timeout"))):
            self.assertFalse(self.mod._send_pushover("hi"))
        self.assertIn("pushover", self.mod._status["last_error"])


class PushToPhoneFanoutTests(unittest.TestCase):
    """The telegram leg, the bookkeeping leg, and the no-backend console log."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def test_telegram_leg_fires_and_updates_counters(self):
        with mock.patch.dict(os.environ,
                             {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_USER_ID": "42"},
                             clear=True), \
             mock.patch.object(self.mod, "_send_telegram", return_value=True) as tg, \
             mock.patch.object(self.mod, "_save_state") as save:
            out = self.mod.push_to_phone("hi", confirm=False)
        self.assertEqual(out, {"telegram": True})
        tg.assert_called_once()
        # Successful send bumps the out counters and stamps last_outbound_at.
        self.assertEqual(self.mod._status["messages_out"], 1)
        self.assertEqual(self.mod._persisted["messages_out"], 1)
        self.assertIsNotNone(self.mod._status["last_outbound_at"])
        save.assert_called_once()

    def test_no_backend_logs_to_console_and_skips_save(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True), \
             mock.patch.object(self.mod, "_save_state") as save, \
             mock.patch("builtins.print") as pr:
            out = self.mod.push_to_phone("orphan message", confirm=False, source="bambu")
        self.assertEqual(out, {})
        save.assert_not_called()
        printed = " ".join(str(c.args[0]) for c in pr.call_args_list if c.args)
        self.assertIn("no backend configured", printed)
        self.assertIn("bambu", printed)

    def test_confirm_true_with_gate_yes_proceeds(self):
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.object(self.mod, "draft_confirm", return_value=True) as gate, \
             mock.patch.object(self.mod, "_send_ntfy", return_value=True), \
             mock.patch.object(self.mod, "_save_state"):
            out = self.mod.push_to_phone("approved body", confirm=True, recipient="mum")
        self.assertEqual(out, {"ntfy": True})
        gate.assert_called_once()
        # recipient is forwarded to the gate's spoken prompt.
        self.assertEqual(gate.call_args.kwargs.get("recipient"), "mum")

    def test_failed_send_does_not_bump_counters(self):
        # Backend configured but the send returns False → results non-empty but
        # sent_count == 0, so messages_out stays 0 (state still saved).
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.object(self.mod, "_send_ntfy", return_value=False), \
             mock.patch.object(self.mod, "_save_state"):
            out = self.mod.push_to_phone("hi", confirm=False)
        self.assertEqual(out, {"ntfy": False})
        self.assertEqual(self.mod._status["messages_out"], 0)


class LlmFallbackTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def _fake_anthropic(self, text="Indeed, sir.", raise_exc=None):
        mod = types.ModuleType("anthropic")
        block = types.SimpleNamespace(text=text)
        message = types.SimpleNamespace(content=[block])
        client = mock.MagicMock()
        if raise_exc is not None:
            client.messages.create.side_effect = raise_exc
        else:
            client.messages.create.return_value = message
        mod.Anthropic = mock.MagicMock(return_value=client)
        return mod, client

    def test_no_api_key_and_no_local_returns_none(self):
        # No ANTHROPIC_API_KEY → Claude path skipped; the local Ollama bridge
        # reports nothing → None. A fake bobert_companion is injected (never the
        # real monolith) whose _call_local_llm yields None.
        bc = types.ModuleType("bobert_companion")
        bc._call_local_llm = mock.MagicMock(return_value=None)
        with mock.patch.dict(os.environ, _NO_ENV, clear=True), \
             inject_modules(bobert_companion=bc):
            self.assertIsNone(self.mod._llm_fallback("hello"))

    def test_anthropic_happy_path(self):
        anth, client = self._fake_anthropic("Quite so, sir.")
        with mock.patch.dict(os.environ,
                             {"ANTHROPIC_API_KEY": "sk-x", "PHONE_BRIDGE_MODEL": "claude-test"},
                             clear=True), \
             inject_modules(anthropic=anth):
            out = self.mod._llm_fallback("what time is it")
        self.assertEqual(out, "Quite so, sir.")
        # PHONE_BRIDGE_MODEL is honoured (no bobert_companion lookup needed).
        self.assertEqual(client.messages.create.call_args.kwargs["model"], "claude-test")

    def test_model_falls_back_to_companion_claude_model(self):
        anth, client = self._fake_anthropic("ok")
        bc = types.ModuleType("bobert_companion")
        bc.CLAUDE_MODEL = "companion-model-x"
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-x"}, clear=True), \
             inject_modules(anthropic=anth, bobert_companion=bc):
            self.mod._llm_fallback("hi")
        self.assertEqual(client.messages.create.call_args.kwargs["model"],
                         "companion-model-x")

    def test_model_defaults_when_companion_import_fails(self):
        # When the companion can't be imported for the model lookup, the model
        # defaults to _DEFAULT_LLM_MODEL. The skill uses importlib.import_module
        # (NOT builtins.__import__), so we force that to raise the
        # isolation-safe way: a None entry in sys.modules makes import_module
        # raise ImportError WITHOUT ever executing the real monolith (which
        # would sys.exit on its boot singleton lock).
        anth, client = self._fake_anthropic("ok")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-x"}, clear=True), \
             inject_modules(anthropic=anth), \
             mock.patch.dict(sys.modules, {"bobert_companion": None}):
            self.mod._llm_fallback("hi")
        self.assertEqual(client.messages.create.call_args.kwargs["model"],
                         self.mod._DEFAULT_LLM_MODEL)

    def test_anthropic_import_fails_then_local_llm_used(self):
        # API key present but anthropic import blocked → Claude path disabled,
        # Ollama fallback via bobert_companion._call_local_llm answers.
        bc = types.ModuleType("bobert_companion")
        bc.CLAUDE_MODEL = "m"
        bc._call_local_llm = mock.MagicMock(return_value="  Local reply, sir.  ")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-x"}, clear=True), \
             block_import("anthropic"), inject_modules(bobert_companion=bc):
            out = self.mod._llm_fallback("hello")
        self.assertEqual(out, "Local reply, sir.")
        bc._call_local_llm.assert_called_once()

    def test_anthropic_call_raises_then_local_fallback(self):
        anth, _client = self._fake_anthropic(raise_exc=RuntimeError("429 overloaded"))
        bc = types.ModuleType("bobert_companion")
        bc.CLAUDE_MODEL = "m"
        bc._call_local_llm = mock.MagicMock(return_value="Fallback, sir.")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-x"}, clear=True), \
             inject_modules(anthropic=anth, bobert_companion=bc):
            out = self.mod._llm_fallback("hello")
        self.assertEqual(out, "Fallback, sir.")

    def test_local_llm_returns_blank_yields_none(self):
        anth, _client = self._fake_anthropic(raise_exc=RuntimeError("down"))
        bc = types.ModuleType("bobert_companion")
        bc.CLAUDE_MODEL = "m"
        bc._call_local_llm = mock.MagicMock(return_value="   ")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-x"}, clear=True), \
             inject_modules(anthropic=anth, bobert_companion=bc):
            self.assertIsNone(self.mod._llm_fallback("hello"))

    def test_local_llm_raises_yields_none(self):
        bc = types.ModuleType("bobert_companion")
        bc.CLAUDE_MODEL = "m"
        bc._call_local_llm = mock.MagicMock(side_effect=RuntimeError("ollama down"))
        with mock.patch.dict(os.environ, _NO_ENV, clear=True), \
             inject_modules(bobert_companion=bc):
            self.assertIsNone(self.mod._llm_fallback("hello"))

    def test_anthropic_empty_content_then_local(self):
        # Claude returns no text blocks → out stays "" → falls to local LLM.
        anth = types.ModuleType("anthropic")
        client = mock.MagicMock()
        client.messages.create.return_value = types.SimpleNamespace(content=[])
        anth.Anthropic = mock.MagicMock(return_value=client)
        bc = types.ModuleType("bobert_companion")
        bc.CLAUDE_MODEL = "m"
        bc._call_local_llm = mock.MagicMock(return_value="Local, sir.")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-x"}, clear=True), \
             inject_modules(anthropic=anth, bobert_companion=bc):
            self.assertEqual(self.mod._llm_fallback("hi"), "Local, sir.")


class DispatchRemoteLadderTests(unittest.TestCase):
    """Exercise the full resolution ladder in _dispatch_remote. core.dispatcher
    and core.mode_router are faked per-test."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("phone_bridge")
        # _dispatch_remote early-returns unless the action registry is wired.
        self.mod._actions_ref[0] = self.actions

    def _mode_router(self, toggle=None, current="smart"):
        m = types.ModuleType("core.mode_router")
        m.maybe_handle_mode_toggle = mock.MagicMock(return_value=toggle)
        m.current_mode = mock.MagicMock(return_value=current)
        return m

    def _dispatcher(self, chain=None, step=_SENTINEL):
        d = types.ModuleType("core.dispatcher")
        d.resolve_and_dispatch = mock.MagicMock(return_value=chain)
        if step is _SENTINEL:
            step = None
        d.match_single_intent = mock.MagicMock(return_value=step)
        return d

    def test_mode_toggle_short_circuits(self):
        with inject_modules(**{"core.mode_router":
                               self._mode_router(toggle="Smart mode, sir.")}):
            out = self.mod._dispatch_remote("smart mode")
        self.assertEqual(out, "Smart mode, sir.")

    def test_chain_dispatch_wins(self):
        with inject_modules(**{"core.mode_router": self._mode_router(),
                               "core.dispatcher": self._dispatcher(chain="Chain done, sir.")}):
            out = self.mod._dispatch_remote("play lo-fi and dim lights")
        self.assertEqual(out, "Chain done, sir.")

    def test_chain_raises_then_single_intent_runs_action(self):
        d = self._dispatcher(chain=None,
                             step=types.SimpleNamespace(action="play_music", arg="lofi",
                                                        confirmation="playing"))
        d.resolve_and_dispatch.side_effect = RuntimeError("chain boom")
        self.actions["play_music"] = lambda arg: f"Now playing {arg}, sir."
        with inject_modules(**{"core.mode_router": self._mode_router(),
                               "core.dispatcher": d}):
            out = self.mod._dispatch_remote("play lofi")
        self.assertEqual(out, "Now playing lofi, sir.")

    def test_single_intent_action_returns_blank_uses_confirmation(self):
        step = types.SimpleNamespace(action="dim", arg="", confirmation="dimming")
        self.actions["dim"] = lambda arg: ""   # falsy → fall back to confirmation
        with inject_modules(**{"core.mode_router": self._mode_router(),
                               "core.dispatcher": self._dispatcher(step=step)}):
            out = self.mod._dispatch_remote("dim the lights")
        self.assertEqual(out, "Dimming, sir.")

    def test_single_intent_action_raises_is_caught(self):
        step = types.SimpleNamespace(action="boom", arg="x", confirmation="c")
        def _boom(_):
            raise ValueError("kaboom")
        self.actions["boom"] = _boom
        with inject_modules(**{"core.mode_router": self._mode_router(),
                               "core.dispatcher": self._dispatcher(step=step)}):
            out = self.mod._dispatch_remote("boom now")
        self.assertIn("That action failed", out)
        self.assertIn("ValueError", out)

    def test_match_single_intent_raises_then_llm_fallback(self):
        d = self._dispatcher(chain=None)
        d.match_single_intent.side_effect = RuntimeError("matcher boom")
        with inject_modules(**{"core.mode_router": self._mode_router(),
                               "core.dispatcher": d}), \
             mock.patch.object(self.mod, "_llm_fallback", return_value="LLM says hi, sir."):
            out = self.mod._dispatch_remote("ramble")
        self.assertEqual(out, "LLM says hi, sir.")

    def test_no_match_no_llm_returns_help_hint(self):
        with inject_modules(**{"core.mode_router": self._mode_router(),
                               "core.dispatcher": self._dispatcher()}), \
             mock.patch.object(self.mod, "_llm_fallback", return_value=None):
            out = self.mod._dispatch_remote("gibberish xyzzy")
        self.assertIn("couldn't match that", out)
        self.assertIn("/help", out)

    def test_mode_router_import_failure_is_swallowed(self):
        # mode_router import raises → pipeline still falls through to LLM.
        with block_import("core.mode_router"), \
             inject_modules(**{"core.dispatcher": self._dispatcher()}), \
             mock.patch.object(self.mod, "_llm_fallback", return_value="ok sir"):
            out = self.mod._dispatch_remote("hello there")
        self.assertEqual(out, "ok sir")


class HandleSlashExtraTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def test_pause_routes(self):
        with mock.patch.object(self.mod, "pause_phone_bridge",
                               return_value="paused-string") as p:
            self.assertEqual(self.mod._handle_slash("/pause"), "paused-string")
        p.assert_called_once()

    def test_resume_routes(self):
        with mock.patch.object(self.mod, "resume_phone_bridge",
                               return_value="resumed-string") as r:
            self.assertEqual(self.mod._handle_slash("/resume"), "resumed-string")
        r.assert_called_once()

    def test_mode_reports_current_mode(self):
        m = types.ModuleType("core.mode_router")
        m.current_mode = mock.MagicMock(return_value="agent")
        with inject_modules(**{"core.mode_router": m}):
            out = self.mod._handle_slash("/mode")
        self.assertIn("agent", out)
        self.assertTrue(out.rstrip().endswith("sir."))

    def test_mode_router_unavailable(self):
        with block_import("core.mode_router"):
            out = self.mod._handle_slash("/mode")
        self.assertIn("Mode router not loaded", out)

    def test_start_alias_returns_help(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            self.assertIn("phone bridge", self.mod._handle_slash("/start").lower())


class HelpTextTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def test_lists_configured_backends(self):
        with mock.patch.dict(os.environ,
                             {"TELEGRAM_BOT_TOKEN": "t", "NTFY_TOPIC": "x",
                              "PUSHOVER_TOKEN": "k", "PUSHOVER_USER": "u"},
                             clear=True):
            out = self.mod._help_text()
        self.assertIn("telegram", out)
        self.assertIn("ntfy", out)
        self.assertIn("pushover", out)

    def test_none_configured_says_none(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            out = self.mod._help_text()
        self.assertIn("(none)", out)


class ProcessUpdateEdgeTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def _update(self, user_id, chat_id, text, update_id=1):
        return {"update_id": update_id, "message": {
            "text": text, "chat": {"id": chat_id}, "from": {"id": user_id}}}

    def test_non_integer_chat_id_ignored(self):
        upd = {"update_id": 1, "message": {
            "text": "hi", "chat": {"id": "not-an-int"}, "from": {"id": 42}}}
        with mock.patch.object(self.mod, "_dispatch_remote") as disp:
            self.mod._process_update(upd, whitelist={42})
        disp.assert_not_called()

    def test_empty_whitelist_allows_through(self):
        # whitelist falsy → the `whitelist and ...` guard is skipped, message
        # is processed (the loop warns separately but does not block here).
        with mock.patch.object(self.mod, "_dispatch_remote",
                               return_value="done, sir.") as disp, \
             mock.patch.object(self.mod, "_send_telegram", return_value=True):
            self.mod._process_update(self._update(5, 5, "hello"), whitelist=set())
        disp.assert_called_once_with("hello")

    def test_chat_id_in_whitelist_even_if_user_not(self):
        # Group chats: user_id may differ but chat_id is whitelisted.
        with mock.patch.object(self.mod, "_dispatch_remote",
                               return_value="ok, sir.") as disp, \
             mock.patch.object(self.mod, "_send_telegram", return_value=True):
            self.mod._process_update(self._update(999, 42, "hi"), whitelist={42})
        disp.assert_called_once()

    def test_slash_handler_crash_isolated(self):
        with mock.patch.object(self.mod, "_handle_slash",
                               side_effect=RuntimeError("slash boom")), \
             mock.patch.object(self.mod, "_send_telegram", return_value=True) as send:
            self.mod._process_update(self._update(42, 42, "/help"), whitelist={42})
        reply = send.call_args[0][0]
        self.assertIn("Slash handler crashed", reply)

    def test_dispatch_crash_isolated(self):
        with mock.patch.object(self.mod, "_dispatch_remote",
                               side_effect=RuntimeError("dispatch boom")), \
             mock.patch.object(self.mod, "_send_telegram", return_value=True) as send:
            self.mod._process_update(self._update(42, 42, "play x"), whitelist={42})
        reply = send.call_args[0][0]
        self.assertIn("Dispatch crashed", reply)

    def test_unauthorised_send_exception_swallowed(self):
        # The courtesy "Unauthorised" reply itself failing must not propagate.
        with mock.patch.object(self.mod, "_send_telegram",
                               side_effect=RuntimeError("send down")), \
             mock.patch.object(self.mod, "_dispatch_remote") as disp:
            self.mod._process_update(self._update(7, 7, "hi"), whitelist={42})
        disp.assert_not_called()

    def test_inbound_counters_bumped_for_authorised(self):
        with mock.patch.object(self.mod, "_dispatch_remote", return_value="r"), \
             mock.patch.object(self.mod, "_send_telegram", return_value=True):
            self.mod._process_update(self._update(42, 42, "hello"), whitelist={42})
        self.assertEqual(self.mod._status["messages_in"], 1)
        self.assertEqual(self.mod._persisted["messages_in"], 1)


class PollingLoopTests(unittest.TestCase):
    """Drive _telegram_polling_loop deterministically: time.sleep is a no-op and
    the faked getUpdates flips the pause flag so the while-loop runs exactly one
    batch then exits. No real sockets, no threads, no waiting."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")
        # Neuter every sleep in the module (init delay + retry back-off).
        self._sleep = mock.patch.object(self.mod.time, "sleep", lambda *_a, **_k: None)
        self._sleep.start()
        self.addCleanup(self._sleep.stop)

    def _env(self, **extra):
        base = {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_USER_ID": "42"}
        base.update(extra)
        return base

    def test_no_token_returns_immediately(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            # Should fall through the init sleep and return without touching net.
            self.mod._telegram_polling_loop()
        self.assertFalse(self.mod._status["polling"])

    def test_requests_missing_sets_error_and_returns(self):
        with mock.patch.dict(os.environ, self._env(), clear=True), \
             block_import("requests"):
            self.mod._telegram_polling_loop()
        self.assertIn("requests not installed", self.mod._status["last_error"])

    def test_no_whitelist_warns_but_still_polls_once(self):
        # Token set, whitelist empty → records the warning, then enters the loop.
        req = _fake_requests()
        ok_payload = _Resp(200)
        ok_payload.json = lambda: {"ok": True, "result": []}
        req.get = mock.MagicMock(side_effect=lambda *a, **k: self._stop_after(ok_payload))
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t"}, clear=True), \
             inject_modules(requests=req):
            self.mod._telegram_polling_loop()
        self.assertEqual(self.mod._status["last_error"], "no TELEGRAM_USER_ID whitelist")

    def _stop_after(self, resp):
        """Helper: flip the pause flag so the loop exits after this iteration."""
        self.mod._pause_flag[0] = True
        return resp

    def test_one_batch_processes_update_and_advances_offset(self):
        update = {"update_id": 555, "message": {
            "text": "hi", "chat": {"id": 42}, "from": {"id": 42}}}
        resp = _Resp(200)
        resp.json = lambda: {"ok": True, "result": [update]}
        req = _fake_requests()
        req.get = mock.MagicMock(side_effect=lambda *a, **k: self._stop_after(resp))
        with mock.patch.dict(os.environ, self._env(), clear=True), \
             inject_modules(requests=req), \
             mock.patch.object(self.mod, "_process_update") as proc, \
             mock.patch.object(self.mod, "_save_state"):
            self.mod._telegram_polling_loop()
        proc.assert_called_once()
        # Offset advanced past the processed update so it isn't re-fetched.
        self.assertEqual(self.mod._persisted["last_update_id"], 555)
        self.assertFalse(self.mod._status["polling"])   # cleared on exit

    def test_http_error_sets_error_then_retries_to_exit(self):
        # First call: HTTP 500 (logs error, sleeps-noop, continues). The flip
        # happens in the side_effect so the second loop check exits.
        resp = _Resp(500, "telegram 500")
        req = _fake_requests()
        req.get = mock.MagicMock(side_effect=lambda *a, **k: self._stop_after(resp))
        with mock.patch.dict(os.environ, self._env(), clear=True), \
             inject_modules(requests=req):
            self.mod._telegram_polling_loop()
        self.assertIn("getUpdates 500", self.mod._status["last_error"])

    def test_getupdates_not_ok_sets_error(self):
        resp = _Resp(200)
        resp.json = lambda: {"ok": False, "description": "bot was blocked"}
        req = _fake_requests()
        req.get = mock.MagicMock(side_effect=lambda *a, **k: self._stop_after(resp))
        with mock.patch.dict(os.environ, self._env(), clear=True), \
             inject_modules(requests=req):
            self.mod._telegram_polling_loop()
        self.assertEqual(self.mod._status["last_error"], "bot was blocked")

    def test_getupdates_exception_sets_error(self):
        def _raise_then_stop(*a, **k):
            self.mod._pause_flag[0] = True
            raise RuntimeError("socket reset")
        req = _fake_requests()
        req.get = mock.MagicMock(side_effect=_raise_then_stop)
        with mock.patch.dict(os.environ, self._env(), clear=True), \
             inject_modules(requests=req):
            self.mod._telegram_polling_loop()
        self.assertIn("getUpdates:", self.mod._status["last_error"])

    def test_update_handler_crash_does_not_abort_batch(self):
        update = {"update_id": 9, "message": {"text": "hi", "chat": {"id": 42},
                                              "from": {"id": 42}}}
        resp = _Resp(200)
        resp.json = lambda: {"ok": True, "result": [update]}
        req = _fake_requests()
        req.get = mock.MagicMock(side_effect=lambda *a, **k: self._stop_after(resp))
        with mock.patch.dict(os.environ, self._env(), clear=True), \
             inject_modules(requests=req), \
             mock.patch.object(self.mod, "_process_update",
                               side_effect=RuntimeError("handler boom")), \
             mock.patch.object(self.mod, "_save_state"):
            self.mod._telegram_polling_loop()   # must not raise
        # Offset still advances despite the handler crash (crash-safe drain).
        self.assertEqual(self.mod._persisted["last_update_id"], 9)


class StartPollingThreadTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")

    def test_no_token_is_noop(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True), \
             mock.patch("threading.Thread") as T:
            self.mod._start_polling_thread()
        T.assert_not_called()

    def test_starts_thread_when_token_present(self):
        fake_thread = mock.MagicMock()
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t"}, clear=True), \
             mock.patch.object(self.mod.threading, "Thread", return_value=fake_thread) as T:
            self.mod._start_polling_thread()
        T.assert_called_once()
        fake_thread.start.assert_called_once()
        self.assertIs(self.mod._polling_thread[0], fake_thread)

    def test_idempotent_when_thread_alive(self):
        alive = mock.MagicMock()
        alive.is_alive.return_value = True
        self.mod._polling_thread[0] = alive
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t"}, clear=True), \
             mock.patch.object(self.mod.threading, "Thread") as T:
            self.mod._start_polling_thread()
        T.assert_not_called()   # already running → no second thread


class PersistenceTests(unittest.TestCase):
    """_load_state / _save_state round-trip against a real temp file (the only
    place we let the skill touch disk — never the project's data/ dir)."""
    def setUp(self):
        self.mod, _ = load_skill_isolated("phone_bridge")
        self.tmp = tempfile.mkdtemp(prefix="phonebridge_test_")
        self.addCleanup(self._cleanup)
        self.state_path = os.path.join(self.tmp, "data", "phone_bridge_state.json")
        self.mod._STATE_FILE = self.state_path

    def _cleanup(self):
        for root, _dirs, files in os.walk(self.tmp, topdown=False):
            for fn in files:
                try:
                    os.unlink(os.path.join(root, fn))
                except OSError:
                    pass
            try:
                os.rmdir(root)
            except OSError:
                pass

    def test_load_state_no_file_keeps_defaults(self):
        # File does not exist → no-op, defaults intact.
        self.mod._load_state()
        self.assertEqual(self.mod._persisted["last_update_id"], 0)

    def test_save_then_load_round_trip(self):
        self.mod._persisted["last_update_id"] = 1234
        self.mod._persisted["messages_in"] = 7
        self.mod._persisted["messages_out"] = 9
        self.mod._save_state()
        self.assertTrue(os.path.exists(self.state_path))
        # Mutate in memory, then reload from disk to prove the read path.
        self.mod._persisted["last_update_id"] = 0
        self.mod._persisted["messages_in"] = 0
        self.mod._load_state()
        self.assertEqual(self.mod._persisted["last_update_id"], 1234)
        self.assertEqual(self.mod._persisted["messages_in"], 7)
        self.assertEqual(self.mod._persisted["messages_out"], 9)

    def test_load_state_ignores_non_dict_payload(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(["not", "a", "dict"], f)
        self.mod._load_state()   # tolerated, defaults kept
        self.assertEqual(self.mod._persisted["last_update_id"], 0)

    def test_load_state_ignores_wrong_typed_values(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({"last_update_id": "high", "messages_in": 3}, f)
        self.mod._load_state()
        # Non-int last_update_id rejected; valid int messages_in accepted.
        self.assertEqual(self.mod._persisted["last_update_id"], 0)
        self.assertEqual(self.mod._persisted["messages_in"], 3)

    def test_load_state_corrupt_json_tolerated(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            f.write("{ this is not json")
        self.mod._load_state()   # logs a warning, does not raise
        self.assertEqual(self.mod._persisted["last_update_id"], 0)

    def test_save_state_failure_is_swallowed(self):
        # Point the state file at an impossible location so makedirs/write fail;
        # _save_state must log and swallow, never raise.
        self.mod._STATE_FILE = os.path.join(self.tmp, "f", "\x00bad", "s.json")
        try:
            self.mod._save_state()
        except Exception as e:  # pragma: no cover - asserting it does NOT raise
            self.fail(f"_save_state raised: {e!r}")


class StatusRenderTests(unittest.TestCase):
    """The detail branches of phone_bridge_status / list_phone_backends that the
    all-unconfigured tests never reach."""
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("phone_bridge")

    def test_status_with_telegram_whitelist_and_activity(self):
        now = self.mod.time.time()
        with mock.patch.dict(os.environ,
                             {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_USER_ID": "1,2",
                              "NTFY_TOPIC": "x", "PUSHOVER_TOKEN": "k",
                              "PUSHOVER_USER": "u"}, clear=True):
            self.mod._status["polling"] = True
            self.mod._status["messages_in"] = 3
            self.mod._status["messages_out"] = 4
            self.mod._status["last_inbound_at"] = now - 10
            self.mod._status["last_outbound_at"] = now - 20
            self.mod._status["last_error"] = "telegram 429"
            self.mod._pause_flag[0] = True
            out = self.actions["phone_bridge_status"]("")
        self.assertIn("telegram (whitelist: 2)", out)
        self.assertIn("ntfy: yes", out)
        self.assertIn("pushover: yes", out)
        self.assertIn("polling", out)
        self.assertIn("last inbound", out)
        self.assertIn("last outbound", out)
        self.assertIn("paused", out)
        self.assertIn("last error", out)

    def test_status_telegram_no_whitelist_branch(self):
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t"}, clear=True):
            out = self.actions["phone_bridge_status"]("")
        self.assertIn("telegram (no whitelist)", out)

    def test_list_backends_all_configured(self):
        with mock.patch.dict(os.environ,
                             {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_USER_ID": "55",
                              "NTFY_TOPIC": "secret", "NTFY_HOST": "https://n.io",
                              "PUSHOVER_TOKEN": "k", "PUSHOVER_USER": "u"},
                             clear=True):
            out = self.actions["list_phone_backends"]("")
        self.assertIn("telegram: configured, whitelist=[55]", out)
        self.assertIn("ntfy: configured, topic=secret", out)
        self.assertIn("host=https://n.io", out)
        self.assertIn("pushover: configured", out)

    def test_list_backends_telegram_configured_no_whitelist(self):
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t"}, clear=True):
            out = self.actions["list_phone_backends"]("")
        self.assertIn("(none — no inbound)", out)

    def test_any_backend_configured_true_false(self):
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "x"}, clear=True):
            self.assertTrue(self.mod.any_backend_configured())
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            self.assertFalse(self.mod.any_backend_configured())


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.mod, self.actions = load_skill_isolated("phone_bridge")

    def test_register_wires_all_actions(self):
        # load_skill_isolated already called register(); assert the surface.
        for name in ("notify_phone", "push_to_phone", "text_my_phone",
                     "phone_bridge_status", "phone_status", "list_phone_backends",
                     "pause_phone_bridge", "resume_phone_bridge"):
            self.assertIn(name, self.actions)
        # The three send aliases point at the same callable.
        self.assertIs(self.actions["push_to_phone"], self.actions["notify_phone"])
        self.assertIs(self.actions["text_my_phone"], self.actions["notify_phone"])
        # _actions_ref captured for the inbound dispatch path.
        self.assertIs(self.mod._actions_ref[0], self.actions)

    def test_register_prints_configured_backends_branch(self):
        # All three configured so every `configured.append(...)` line runs.
        fresh = {}
        with mock.patch.dict(os.environ,
                             {"TELEGRAM_BOT_TOKEN": "t", "NTFY_TOPIC": "x",
                              "PUSHOVER_TOKEN": "k", "PUSHOVER_USER": "u"},
                             clear=True), \
             mock.patch.object(self.mod, "_start_polling_thread"), \
             mock.patch.object(self.mod, "_load_state"), \
             mock.patch("builtins.print") as pr:
            self.mod.register(fresh)
        printed = " ".join(str(c.args[0]) for c in pr.call_args_list if c.args)
        self.assertIn("phone_bridge ready", printed)
        self.assertIn("telegram", printed)
        self.assertIn("ntfy", printed)
        self.assertIn("pushover", printed)

    def test_register_kicks_polling_thread(self):
        fresh = {}
        with mock.patch.dict(os.environ, _NO_ENV, clear=True), \
             mock.patch.object(self.mod, "_start_polling_thread") as start, \
             mock.patch.object(self.mod, "_load_state"):
            self.mod.register(fresh)
        start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
