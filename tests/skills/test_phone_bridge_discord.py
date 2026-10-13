"""Tests for the Discord leg of skills/phone_bridge.py: Discord first / ntfy as
the fallback in push_to_phone, the configured-backend probes, the inbound DM
handler and the guarded listener start.

core.discord_bridge.send_dm and GatewayClient are mocked and requests is faked,
so nothing touches the network, a socket or a thread. stdlib unittest + mock.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from core import config as _cfg
from core import phone_ping as pp
from tests._skill_harness import load_skill_isolated

# Token-shaped value built at run time (nothing secret-looking in the source).
_TOKEN = ".".join(("A" * 24, "b" * 6, "C" * 27))
_DISCORD = {"DISCORD_BOT_TOKEN": _TOKEN, "DISCORD_OWNER_ID": "1000000000000000001"}
_NTFY = {"NTFY_TOPIC": "t0p1c-xyz"}


class _Resp:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


def _fake_requests(status=200):
    mod = types.ModuleType("requests")
    mod.post = mock.MagicMock(return_value=_Resp(status))
    mod.get = mock.MagicMock(return_value=_Resp(status))
    return mod


class Base(unittest.TestCase):
    def setUp(self):
        pp._reset_for_tests()
        self.addCleanup(pp._reset_for_tests)
        for name in ("DISCORD_ENABLED", "DISCORD_INBOUND", "DISCORD_NTFY_FALLBACK_ONLY"):
            self.addCleanup(setattr, _cfg, name, getattr(_cfg, name))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.mod, self.actions = load_skill_isolated("phone_bridge")
        self.addCleanup(sys.modules.pop, "skill_phone_bridge", None)
        pp.get_pinger()._state_path = None
        mock.patch.object(self.mod, "_save_state").start()
        self.addCleanup(mock.patch.stopall)
        self.send_dm = mock.patch.object(self.mod._discord, "send_dm",
                                         return_value=True).start()


class DefaultsTests(unittest.TestCase):
    def test_settings_default_on_and_in_the_schema(self):
        from tools import settings_window as sw
        for name in ("DISCORD_ENABLED", "DISCORD_INBOUND",
                     "DISCORD_NTFY_FALLBACK_ONLY"):
            self.assertIs(getattr(_cfg, name), True)
            self.assertEqual(sw.SCHEMA[name]["type"], "bool")
        self.assertIn("DISCORD_BOT_TOKEN",
                      sw.SCHEMA["_status_phone"]["secret_env"])


class ProbeTests(Base):
    def test_needs_both_env_vars(self):
        with mock.patch.dict(os.environ, {"DISCORD_BOT_TOKEN": _TOKEN}, clear=True):
            self.assertFalse(self.mod._discord_configured())
            self.assertFalse(self.mod.can_push_unsolicited())
        with mock.patch.dict(os.environ, _DISCORD, clear=True):
            self.assertTrue(self.mod._discord_configured())
            self.assertTrue(self.mod.can_push_unsolicited())
            self.assertTrue(self.mod.any_backend_configured())

    def test_master_switch_turns_it_off(self):
        _cfg.DISCORD_ENABLED = False
        with mock.patch.dict(os.environ, _DISCORD, clear=True):
            self.assertFalse(self.mod._discord_configured())

    def test_discord_is_listed_first(self):
        with mock.patch.dict(os.environ, {**_DISCORD, **_NTFY}, clear=True):
            self.assertEqual(self.mod.configured_backend_names(),
                             ["discord", "ntfy"])


class PushTests(Base):
    def _push(self, env, **kw):
        req = _fake_requests()
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.dict(sys.modules, {"requests": req}):
            res = self.mod.push_to_phone("Print finished, sir.", confirm=False, **kw)
        return res, req

    def test_discord_first_and_ntfy_skipped_when_it_works(self):
        res, req = self._push({**_DISCORD, **_NTFY})
        self.assertEqual(res, {"discord": True})
        self.send_dm.assert_called_once_with("Print finished, sir.")
        req.post.assert_not_called()

    def test_ntfy_is_the_fallback_when_discord_fails(self):
        self.send_dm.return_value = False
        res, req = self._push({**_DISCORD, **_NTFY})
        self.assertEqual(res, {"discord": False, "ntfy": True})
        req.post.assert_called_once()

    def test_both_when_fallback_only_is_off(self):
        _cfg.DISCORD_NTFY_FALLBACK_ONLY = False
        res, req = self._push({**_DISCORD, **_NTFY})
        self.assertEqual(res, {"discord": True, "ntfy": True})

    def test_an_explicit_backend_list_is_honoured(self):
        res, req = self._push({**_DISCORD, **_NTFY}, backends=["ntfy"])
        self.assertEqual(res, {"ntfy": True})
        self.send_dm.assert_not_called()

    def test_ntfy_only_install_is_unchanged(self):
        res, req = self._push(_NTFY)
        self.assertEqual(res, {"ntfy": True})
        self.send_dm.assert_not_called()

    def test_title_and_priority_are_folded_into_the_dm(self):
        self._push(_DISCORD, priority="urgent", title="Guard mode")
        body = self.send_dm.call_args[0][0]
        self.assertTrue(body.startswith("**[URGENT]** **Guard mode**\n"))
        self.assertIn("Print finished, sir.", body)

    def test_failure_is_recorded_for_the_status_line(self):
        self.send_dm.return_value = False
        with mock.patch.object(self.mod._discord, "status",
                               return_value={"last_error": "send: HTTP 403"}):
            res, _req = self._push(_DISCORD)
        self.assertEqual(res, {"discord": False})
        self.assertIn("403", self.mod._status["last_error"])

    def test_ping_sender_goes_through_discord(self):
        req = _fake_requests()
        with mock.patch.dict(os.environ, {**_DISCORD, **_NTFY}, clear=True), \
             mock.patch.dict(sys.modules, {"requests": req}):
            res = self.mod._bridge_send("Guard alert", priority="high",
                                        title="JARVIS", category="security")
        self.assertEqual(res, {"discord": True})


class InboundTests(Base):
    def test_plain_text_goes_through_the_remote_dispatcher(self):
        with mock.patch.object(self.mod, "_dispatch_remote",
                               return_value="Lights on, sir.") as d:
            self.assertEqual(self.mod._discord_handle("turn on the lights"),
                             "Lights on, sir.")
        d.assert_called_once_with("turn on the lights")

    def test_slash_commands_use_the_slash_handler(self):
        with mock.patch.object(self.mod, "_handle_slash", return_value="help") as h:
            self.assertEqual(self.mod._discord_handle("/help"), "help")
        h.assert_called_once_with("/help")

    def test_pause_and_resume_do_not_reach_the_telegram_flag(self):
        for cmd in ("/pause", "/resume"):
            with mock.patch.object(self.mod, "_handle_slash") as h:
                out = self.mod._discord_handle(cmd)
            h.assert_not_called()
            self.assertIn("desk", out)
        self.assertFalse(self.mod._pause_flag[0])


class ListenerStartTests(Base):
    def _start(self, env, **cfg):
        for k, v in cfg.items():
            setattr(_cfg, k, v)
        self.mod._discord_client[0] = None
        client = mock.MagicMock()
        client.start.return_value = True
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(self.mod._discord, "GatewayClient",
                               return_value=client) as ctor, \
             mock.patch("builtins.print"):
            return self.mod._start_discord_listener(), ctor, client

    def test_not_configured_starts_nothing(self):
        ok, ctor, _ = self._start({})
        self.assertFalse(ok)
        ctor.assert_not_called()

    def test_configured_starts_the_listener_with_the_handler(self):
        ok, ctor, client = self._start(_DISCORD)
        self.assertTrue(ok)
        ctor.assert_called_once_with(self.mod._discord_handle)
        client.start.assert_called_once()

    def test_inbound_switch_off_means_send_only(self):
        ok, ctor, _ = self._start(_DISCORD, DISCORD_INBOUND=False)
        self.assertFalse(ok)
        ctor.assert_not_called()

    def test_a_staging_instance_never_listens(self):
        ok, ctor, _ = self._start({**_DISCORD, "JARVIS_STAGING": "1"})
        self.assertFalse(ok)
        ctor.assert_not_called()

    def test_a_test_mode_instance_never_listens(self):
        ok, ctor, _ = self._start({**_DISCORD, "JARVIS_TEST_MODE": "1"})
        self.assertFalse(ok)
        ctor.assert_not_called()

    def test_register_starts_the_listener(self):
        with mock.patch.object(self.mod, "_start_discord_listener") as s, \
             mock.patch.object(self.mod, "_start_polling_thread"), \
             mock.patch.object(self.mod, "_attach_pings"), \
             mock.patch("builtins.print"):
            self.mod.register({})
        s.assert_called_once()


class TextTests(Base):
    def test_status_and_list_name_discord(self):
        with mock.patch.dict(os.environ, _DISCORD, clear=True), \
             mock.patch.object(self.mod._discord, "status",
                               return_value={"connected": True}):
            self.assertIn("discord (listening)", self.mod.phone_bridge_status(""))
            self.assertIn("discord: configured", self.mod.list_phone_backends(""))
            self.assertIn("discord", self.mod._help_text())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIn("DISCORD_BOT_TOKEN", self.mod.list_phone_backends(""))
            self.assertIn("DISCORD_BOT_TOKEN", self.mod.notify_phone("hi"))

    def test_the_token_never_appears_in_any_status_text(self):
        with mock.patch.dict(os.environ, _DISCORD, clear=True):
            for fn in (self.mod.phone_bridge_status, self.mod.list_phone_backends):
                self.assertNotIn(_TOKEN, fn(""))
            self.assertNotIn(_TOKEN, self.mod._help_text())


if __name__ == "__main__":
    unittest.main()
