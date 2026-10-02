"""Tests for the proactive-ping half of skills/phone_bridge.py (2026-10-02):
the bridge attaching itself to core/phone_ping, the voice actions
(phone_setup_help / phone_ping_status / phone_pings_on / phone_pings_off /
phone_ping_test), their utterance route, and the end-to-end path from
ping_phone() through the policy to a FAKE requests module.

Network: requests is faked per test; nothing leaves the box. Settings: the
voice toggle's save is mocked (and the runner redirects JARVIS_SETTINGS_PATH
anyway); core.config.PHONE_PING_ENABLED is restored after each test. The
core.phone_ping singleton is reset around every test.

stdlib unittest + mock only.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import types
import unittest
from unittest import mock

from core import config as _cfg
from core import phone_ping as pp
from tests._skill_harness import load_skill_isolated

_NO_ENV: dict = {}
# A bot-token-SHAPED value built at run time (no secret-looking literal in the
# source for tools/check_no_pii.py to flag).
_FAKE_BOT = "123456789:" + ("q7Xk2Lm9Pz4Rt8Vw3Yb6Nc1Hd5Jf0Gs" * 2)[:35]
_TELEGRAM_FULL = {"TELEGRAM_BOT_TOKEN": _FAKE_BOT, "TELEGRAM_USER_ID": "4242"}


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
        saved = _cfg.PHONE_PING_ENABLED
        self.addCleanup(setattr, _cfg, "PHONE_PING_ENABLED", saved)
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            self.mod, self.actions = load_skill_isolated("phone_bridge")
        self.addCleanup(sys.modules.pop, "skill_phone_bridge", None)
        # Never write the real state file or the live settings.
        pp.get_pinger()._state_path = None
        self.persist = mock.patch.object(self.mod, "_persist_setting",
                                         return_value=True).start()
        self.addCleanup(mock.patch.stopall)


class RegisterTests(Base):
    def test_new_actions_are_registered_and_spoken_verbatim(self):
        for name in ("phone_setup_help", "phone_ping_status", "phone_pings_on",
                     "phone_pings_off", "phone_ping_test"):
            self.assertIn(name, self.actions)
            self.assertIn(name, self.mod.SPEAK_VERBATIM_ACTIONS)

    def test_utterance_route_is_registered(self):
        utils = self.mod.skill_utils
        utils["register_utterance_route"].assert_any_call(
            self.mod._phone_route, "phone pings")

    def test_register_attaches_the_bridge_to_the_policy(self):
        p = pp.get_pinger()
        self.assertIs(p.send, self.mod._bridge_send)
        self.assertIs(p.configured, self.mod.can_push_unsolicited)

    def test_unconfigured_register_logs_one_line_and_starts_nothing(self):
        lines = []
        pp._reset_for_tests()
        pp.get_pinger().log = lines.append
        with mock.patch.dict(os.environ, _NO_ENV, clear=True), \
             mock.patch.object(pp, "start_watcher") as start:
            self.assertFalse(self.mod._attach_pings())
            self.assertFalse(self.mod._attach_pings())
            self.assertEqual(pp.ping("print", "Print finished."), pp.UNCONFIGURED)
        start.assert_not_called()
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("not configured", lines[0])

    def test_configured_register_starts_the_watcher(self):
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t0p1c-xyz"}, clear=True), \
             mock.patch.object(pp, "start_watcher", return_value=True) as start, \
             mock.patch("builtins.print"):
            self.assertTrue(self.mod._attach_pings())
        start.assert_called_once()

    def test_master_switch_off_starts_nothing(self):
        _cfg.PHONE_PING_ENABLED = False
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t0p1c-xyz"}, clear=True), \
             mock.patch.object(pp, "start_watcher") as start, \
             mock.patch("builtins.print"):
            self.assertFalse(self.mod._attach_pings())
        start.assert_not_called()


class UnsolicitedTests(Base):
    def test_a_telegram_token_alone_cannot_message_first(self):
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "x"}, clear=True):
            self.assertFalse(self.mod.can_push_unsolicited())
            self.assertEqual(self.mod.configured_backend_names(), [])
        with mock.patch.dict(os.environ, _TELEGRAM_FULL, clear=True):
            self.assertTrue(self.mod.can_push_unsolicited())
            self.assertEqual(self.mod.configured_backend_names(), ["telegram"])
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True):
            self.assertTrue(self.mod.can_push_unsolicited())
        with mock.patch.dict(os.environ, {"PUSHOVER_TOKEN": "a",
                                          "PUSHOVER_USER": "b"}, clear=True):
            self.assertEqual(self.mod.configured_backend_names(), ["pushover"])

    def test_bridge_send_is_fire_and_forget(self):
        with mock.patch.object(self.mod, "push_to_phone",
                               return_value={"ntfy": True}) as push:
            res = self.mod._bridge_send("Print finished.", priority="high",
                                        title="JARVIS", category="print")
        self.assertEqual(res, {"ntfy": True})
        push.assert_called_once_with("Print finished.", priority="high",
                                     source="ping:print", title="JARVIS",
                                     confirm=False)


class SetupHelpTests(Base):
    def test_unconfigured_gives_the_botfather_steps(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            text = self.actions["phone_setup_help"]("")
        for needle in ("@BotFather", "/newbot", "TELEGRAM_BOT_TOKEN",
                       "@userinfobot", "TELEGRAM_USER_ID", "press Start",
                       ".env", "send a test ping", "NTFY_TOPIC"):
            self.assertIn(needle, text)

    def test_token_without_user_id_names_the_missing_step(self):
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "x"}, clear=True):
            text = self.actions["phone_setup_help"]("")
        self.assertIn("nearly there", text)
        self.assertIn("TELEGRAM_USER_ID", text)
        self.assertNotIn("/newbot", text)

    def test_already_connected(self):
        with mock.patch.dict(os.environ, _TELEGRAM_FULL, clear=True):
            text = self.actions["phone_setup_help"]("")
        self.assertIn("already connected through telegram", text)
        # The token never appears in anything spoken.
        self.assertNotIn(_FAKE_BOT[10:], text)


class RouteTests(Base):
    CASES = {
        "how do I connect my phone": "[ACTION: phone_setup_help]",
        "Jarvis, how do I connect my phone to you?": "[ACTION: phone_setup_help]",
        "how can I link my iPhone to Jarvis": "[ACTION: phone_setup_help]",
        "walk me through how to set up my phone": "[ACTION: phone_setup_help]",
        "connect my phone to you": "[ACTION: phone_setup_help]",
        "how do I set up the phone bridge": "[ACTION: phone_setup_help]",
        "set up telegram": "[ACTION: phone_setup_help]",
        "turn off phone pings": "[ACTION: phone_pings_off]",
        "stop pinging my phone": "[ACTION: phone_pings_off]",
        "phone pings off please": "[ACTION: phone_pings_off]",
        "turn on phone pings": "[ACTION: phone_pings_on]",
        "enable phone pings": "[ACTION: phone_pings_on]",
        "phone ping status": "[ACTION: phone_ping_status]",
        "what's the phone ping status": "[ACTION: phone_ping_status]",
        "are phone pings on": "[ACTION: phone_ping_status]",
        "send a test ping": "[ACTION: phone_ping_test]",
        "send me a test ping to my phone": "[ACTION: phone_ping_test]",
        "test my phone pings": "[ACTION: phone_ping_test]",
    }
    NOT_OURS = (
        "connect my phone to the tv",
        "how do I connect my phone to the speaker",
        "connect my phone",
        "what's my phone number",
        "text my phone the print is done",
        "play telephone by lady gaga",
        "ping google dot com",
        "",
        None,
    )

    def test_exact_phrasings_route(self):
        for text, token in self.CASES.items():
            with self.subTest(text=text):
                self.assertEqual(self.mod._phone_route(text), token)

    def test_other_phone_requests_are_left_alone(self):
        for text in self.NOT_OURS:
            with self.subTest(text=text):
                self.assertIsNone(self.mod._phone_route(text))


class StatusTests(Base):
    def test_off(self):
        _cfg.PHONE_PING_ENABLED = False
        self.assertIn("Phone pings are off", self.actions["phone_ping_status"](""))

    def test_on_but_unconfigured(self):
        _cfg.PHONE_PING_ENABLED = True
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            text = self.actions["phone_ping_status"]("")
        self.assertIn("isn't connected yet", text)
        self.assertIn("how do I connect my phone", text)

    def test_on_and_configured_reads_the_switches(self):
        _cfg.PHONE_PING_ENABLED = True
        p = pp.get_pinger()
        p.blocked = lambda: ""
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.object(_cfg, "PHONE_PING_ROBOT", False), \
             mock.patch.object(_cfg, "PHONE_PING_CONFIRM", True):
            text = self.actions["phone_ping_status"]("")
        self.assertIn("through ntfy: prints, unanswered confirmations and "
                      "guard alerts.", text)
        self.assertIn("The daily summary is off.", text)
        self.assertIn("23:00 to 07:00", text)
        self.assertIn("at most 6 an hour", text)
        self.assertIn("0 sent in the last hour.", text)

    def test_off_but_connected_says_guard_alerts_still_go(self):
        _cfg.PHONE_PING_ENABLED = False
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True):
            text = self.actions["phone_ping_status"]("")
        self.assertIn("Phone pings are off", text)
        self.assertIn("Guard-mode alerts still reach your phone", text)


class ToggleTests(Base):
    def test_off_then_on_flip_and_persist(self):
        _cfg.PHONE_PING_ENABLED = True
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.object(pp, "start_watcher", return_value=True) as start, \
             mock.patch.object(pp, "stop_watcher") as stop:
            off = self.actions["phone_pings_off"]("")
            self.assertFalse(_cfg.PHONE_PING_ENABLED)
            on = self.actions["phone_pings_on"]("")
            self.assertTrue(_cfg.PHONE_PING_ENABLED)
            again = self.actions["phone_pings_on"]("")
        self.assertIn("Phone pings off", off)
        self.assertIn("'text my phone' still works", off)
        # 2026-10-02 review: it says the guard is NOT silenced by this.
        self.assertIn("Guard-mode alerts still reach your phone", off)
        self.assertIn("Phone pings on", on)
        self.assertIn("already on", again)
        stop.assert_called_once()
        self.assertEqual(start.call_count, 2)
        self.assertEqual(self.persist.call_args_list,
                         [mock.call("PHONE_PING_ENABLED", False),
                          mock.call("PHONE_PING_ENABLED", True),
                          mock.call("PHONE_PING_ENABLED", True)])

    def test_on_without_a_phone_says_so(self):
        _cfg.PHONE_PING_ENABLED = False
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            text = self.actions["phone_pings_on"]("")
        self.assertIn("isn't connected yet", text)
        self.assertTrue(_cfg.PHONE_PING_ENABLED)

    def test_off_with_guard_alerts_switched_off_says_so(self):
        with mock.patch.object(_cfg, "PHONE_PING_SECURITY", False):
            text = self.actions["phone_pings_off"]("")
        self.assertIn("Guard-mode alerts are switched off in Settings too", text)

    def test_dont_ping_me_leaves_the_guard_armed_end_to_end(self):
        """'don't ping me' -> phone_pings_off saves the master switch off;
        a guard alert through the bridge's ping_phone still reaches ntfy."""
        _cfg.PHONE_PING_ENABLED = True
        self.assertEqual(self.mod._phone_route("don't ping me"),
                         "[ACTION: phone_pings_off]")
        fake = _fake_requests()
        p = pp.get_pinger()
        p.blocked = lambda: ""
        p.spawn = lambda job: (job(), True)[1]
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.dict(sys.modules, {"requests": fake}), \
             mock.patch.object(pp, "stop_watcher"):
            self.actions["phone_pings_off"]("")
            self.assertFalse(_cfg.PHONE_PING_ENABLED)
            self.assertEqual(self.mod.ping_phone("print", "Print finished."),
                             pp.DISABLED)
            out = self.mod.ping_phone("security", "Someone is at the desk.",
                                      critical=True, priority="urgent")
        self.assertEqual(out, pp.QUEUED)
        self.assertEqual(fake.post.call_count, 1)

    def test_a_failed_save_is_reported(self):
        self.persist.return_value = False
        text = self.actions["phone_pings_off"]("")
        self.assertIn("couldn't save it", text)

    def test_the_persist_path_writes_the_settings_file(self):
        """The real writer (no mock): settings_window.save_settings with the
        key, under the runner's throwaway JARVIS_SETTINGS_PATH."""
        mock.patch.stopall()
        fake_sw = types.ModuleType("tools.settings_window")
        fake_sw.load_settings = lambda: {"OTHER": 1}
        fake_sw.save_settings = mock.MagicMock()
        import tools as tools_pkg
        with mock.patch.dict(sys.modules, {"tools.settings_window": fake_sw}), \
             mock.patch.object(tools_pkg, "settings_window", fake_sw, create=True):
            self.assertTrue(self.mod._persist_setting("PHONE_PING_ENABLED", False))
        fake_sw.save_settings.assert_called_once_with(
            {"OTHER": 1, "PHONE_PING_ENABLED": False},
            changed=("PHONE_PING_ENABLED",))


class TestPingTests(Base):
    def test_unconfigured(self):
        with mock.patch.dict(os.environ, _NO_ENV, clear=True):
            self.assertIn("isn't connected yet",
                          self.actions["phone_ping_test"](""))

    def test_sends_one_fixed_text_without_a_readback(self):
        pp.get_pinger().blocked = lambda: ""
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.object(self.mod, "push_to_phone",
                               return_value={"ntfy": True}) as push:
            text = self.actions["phone_ping_test"]("")
        self.assertEqual(text, "Test ping sent through ntfy, sir.")
        args, kwargs = push.call_args
        self.assertEqual(args[0], self.mod._TEST_TEXT)
        self.assertIs(kwargs["confirm"], False)

    def test_partial_and_total_failure(self):
        pp.get_pinger().blocked = lambda: ""
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.object(self.mod, "push_to_phone",
                               return_value={"ntfy": True, "pushover": False}):
            self.assertIn("but pushover failed",
                          self.actions["phone_ping_test"](""))
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.object(self.mod, "push_to_phone",
                               return_value={"ntfy": False}):
            self.assertIn("failed on every backend",
                          self.actions["phone_ping_test"](""))

    def test_never_from_staging(self):
        pp.get_pinger().blocked = lambda: "staging"
        with mock.patch.dict(os.environ, {"NTFY_TOPIC": "t"}, clear=True), \
             mock.patch.object(self.mod, "push_to_phone") as push:
            self.assertIn("staging", self.actions["phone_ping_test"](""))
        push.assert_not_called()


class EndToEndTests(Base):
    """ping_phone -> core.phone_ping policy -> _bridge_send -> push_to_phone
    -> a fake requests.post to Telegram. The real bridge, a fake network."""

    def _arm(self):
        # push_to_phone's own counter file lives in the live data/ dir.
        mock.patch.object(self.mod, "_save_state").start()
        p = pp.get_pinger()
        p.blocked = lambda: ""
        p.focus_active = lambda: False
        p.owner_idle_s = lambda: 3600.0
        p.wall_now = lambda: dt.datetime(2026, 10, 2, 14, 0)
        p.spawn = lambda job: (job(), True)[1]
        p.log = lambda s: None
        return p

    def test_a_print_ping_reaches_telegram_scrubbed(self):
        self._arm()
        req = _fake_requests()
        with mock.patch.dict(os.environ, _TELEGRAM_FULL, clear=True), \
             mock.patch.dict(sys.modules, {"requests": req}):
            out = self.mod.ping_phone(
                "print", "Print finished, sir: 'benchy' is done. ref "
                + _TELEGRAM_FULL["TELEGRAM_BOT_TOKEN"])
        self.assertEqual(out, pp.QUEUED)
        req.post.assert_called_once()
        url = req.post.call_args.args[0]
        body = req.post.call_args.kwargs["json"]
        self.assertIn("/sendMessage", url)
        self.assertEqual(body["chat_id"], 4242)
        self.assertTrue(body["text"].startswith("Print finished, sir: 'benchy'"))
        self.assertNotIn(_FAKE_BOT[10:], body["text"])
        self.assertIn("[redacted]", body["text"])

    def test_without_a_backend_nothing_is_attempted(self):
        self._arm()
        req = _fake_requests()
        with mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "x"}, clear=True), \
             mock.patch.dict(sys.modules, {"requests": req}):
            out = self.mod.ping_phone("print", "Print finished.")
        self.assertEqual(out, pp.UNCONFIGURED)
        req.post.assert_not_called()

    def test_ping_phone_without_core_reports_unavailable(self):
        with mock.patch.object(self.mod, "_phone_ping", None):
            self.assertEqual(self.mod.ping_phone("print", "x"), "unavailable")
            self.assertIn("aren't available",
                          self.actions["phone_ping_status"](""))


if __name__ == "__main__":
    unittest.main()
