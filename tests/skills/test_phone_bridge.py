"""Logic tests for skills/phone_bridge.py.

Push to / pull from the phone over Telegram / ntfy / Pushover. Every backend is
env-configured and absent in the test env, so the no-backend degradation paths
are easy and central. We test:
  * whitelist parsing, long-message chunking, priority shorthand parsing,
  * the priority maps,
  * each _send_* turning a (mocked) requests response into a bool,
  * push_to_phone's confirm-gate fail-closed behaviour + per-backend fan-out,
  * inbound _process_update whitelist enforcement (mocked send),
  * _handle_slash routing + _dispatch_remote's boot-order guard,
  * the registered voice actions (notify/status/list/pause/resume).

requests is faked, no sockets, no Telegram, no disk writes (state save mocked).
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

from tests._skill_harness import load_skill_isolated

_NO_ENV = {}  # used with clear=True to guarantee "no backend configured"


class _Resp:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


def _fake_requests(resp=None, raise_exc=None):
    mod = mock.MagicMock(name="requests")
    if raise_exc is not None:
        mod.post.side_effect = raise_exc
        mod.get.side_effect = raise_exc
    else:
        mod.post.return_value = resp or _Resp(200)
        mod.get.return_value = resp or _Resp(200)
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


if __name__ == "__main__":
    unittest.main()
