"""Light-tier regressions for the 2026-10-01 web / tray / Settings fix batch.

One class per finding (the monolith halves live in
tests/monolith/test_monolith_web_tray_settings.py):

  * B012  a typed "JARVIS, <command>" in standby only woke him - the command
          was dropped and the page said "accepted" (fixed in the main loop by
          the voice-loop batch; the page half is checked here).
  * B014  the dashboard's headline wake-word switch named the wrong knob.
  * B015  POST /api/settings saved values the Settings window refuses.
  * B074  the Actions tab ran shutdown / code-runner / cookie-wipe ALIASES of
          confirm-gated actions on one click, and shutdown aliases bypassed
          the tray control plane.
  * B076  switch_llm refused installed models the tray picker offers.
  * B077  web / Settings-window tray commands carried no cid, so the drain
          race could run one twice.

Pure modules only (tools.web_interface, tools.settings_window, core.actions
with a stand-in bobert_companion): no server, no monolith, no live file.

    python -B -m unittest tests.test_web_tray_settings_fixes
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import core.actions as A
from tools import settings_window as sw
from tools import web_interface as wi


def _js_fn(html, name):
    start = html.index("function %s(" % name)
    return html[start:html.index("\n}", start)]


# ── B012 ───────────────────────────────────────────────────────────────────
# Merged with the voice-loop batch's B010, which fixed the same drop in the
# main loop: _handle_sleep_standby now hands "Jarvis, <command>" back as the
# turn, for a typed inject as well as for speech. So the page keeps posting a
# wake-prefixed command straight to /api/say (a force_wake round-trip first
# would only add a 1.5 s wait), and only a command WITHOUT the wake word -
# which standby still ignores - asks and wakes him first.
class StandbyCommandTests(unittest.TestCase):
    PAGE = wi._DASHBOARD_PAGE

    def test_only_a_command_without_the_wake_word_wakes_first(self):
        send = _js_fn(self.PAGE, "sendCommand")
        guard = send[send.index("!WAKE_WORD_RE.test(text)"):]
        guard = guard[:guard.index("sendBtn.disabled = true")]
        self.assertIn("window.confirm(", guard)
        self.assertIn("sendControl('force_wake')", guard)
        self.assertLess(send.index("!WAKE_WORD_RE.test(text)"),
                        send.index("fetch(q('/api/say')"))
        # "wake up" (control plane) and the guard above: no third wake path
        # that would also stall a wake-prefixed command.
        self.assertEqual(send.count("sendControl('force_wake')"), 2)

    def test_the_standby_handler_runs_a_wake_prefixed_inject(self):
        # The main-loop half the page relies on (source check: the light tier
        # cannot import the monolith; tests/monolith cover the behaviour).
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, os.pardir, "bobert_companion.py"),
                  encoding="utf-8") as f:
            src = f.read()
        body = src[src.index("def _handle_sleep_standby("):]
        body = body[:body.index("\ndef ", 10)]
        self.assertIn("_standby_wake_carries_command(text)", body)
        self.assertIn("return (text, _wake_conf)", body)

    def test_the_banner_matches_what_standby_does(self):
        self.assertIn("a typed command is ignored unless it starts with",
                      self.PAGE)


# ── B014 (web half) ────────────────────────────────────────────────────────
class WakeBannerHintTests(unittest.TestCase):
    def test_the_hint_names_the_knob_the_switch_drives(self):
        page = wi._DASHBOARD_PAGE
        self.assertIn("const WAKE_KEY = 'START_IN_STANDBY';", page)
        banner = page[page.index('<div class="wakebanner">'):]
        banner = banner[:banner.index("</div>")]
        self.assertIn("(START_IN_STANDBY)", banner)
        self.assertNotIn("WAKE_WORD_AUTOSTART", banner)


# ── B015 ───────────────────────────────────────────────────────────────────
class WebSettingsUseTheWindowsRuleTests(unittest.TestCase):
    BAD = (("WEB_INTERFACE_PORT", 8443), ("WEB_INTERFACE_PORT", 70000),
           ("WEB_INTERFACE_PORT", 0), ("VAD_THRESHOLD", 0),
           ("VAD_THRESHOLD", 5), ("VAD_THRESHOLD", -1),
           ("VAD_THRESHOLD", "nan"), ("LOCAL_LLM_MODEL", ""),
           ("WHISPER_MODEL_CUDA", "   "), ("TTS_VOICE", ""))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "user_settings.json")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"KEEP": 1}, f)
        with open(self.path, encoding="utf-8") as f:
            self.before = f.read()

    def test_values_the_settings_window_refuses_are_refused(self):
        for name, value in self.BAD:
            with self.subTest(name=name, value=value):
                # The Settings window's own verdict on the same value...
                _v, err = sw.validate_value(sw.SCHEMA[name], value)
                self.assertTrue(err, f"fixture: the window accepts {name}={value!r}")
                # ...is the web panel's verdict too, and nothing is written.
                with self.assertRaises(wi.SettingsWriteError):
                    wi._write_settings({name: value}, self.path)
                with open(self.path, encoding="utf-8") as f:
                    self.assertEqual(f.read(), self.before)

    def test_a_bad_value_in_a_batch_writes_nothing(self):
        with self.assertRaises(wi.SettingsWriteError):
            wi._write_settings({"VAD_THRESHOLD": 0.02,
                                "WEB_INTERFACE_PORT": 8443}, self.path)
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(f.read(), self.before)

    def test_good_values_still_save(self):
        applied = wi._write_settings({"VAD_THRESHOLD": "0.02",
                                      "WEB_INTERFACE_PORT": 8767,
                                      "LOCAL_LLM_MODEL": " some-model:7b "},
                                     self.path)
        self.assertEqual(applied["VAD_THRESHOLD"], 0.02)
        self.assertEqual(applied["WEB_INTERFACE_PORT"], 8767)
        self.assertEqual(applied["LOCAL_LLM_MODEL"], "some-model:7b")
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["KEEP"], 1)

    def test_the_reason_reaches_the_api_error(self):
        with self.assertRaises(wi.SettingsWriteError) as cm:
            wi._write_settings({"WEB_INTERFACE_PORT": 8443}, self.path)
        self.assertIn("AirTag", str(cm.exception))


# ── B074 ───────────────────────────────────────────────────────────────────
class _Runtime(wi.NoRuntime):
    live = True

    def __init__(self, acts):
        self._acts = acts

    def actions(self):
        return self._acts

    def speak_sets(self):
        return (set(), set(), set())


class ActionAliasConfirmTests(unittest.TestCase):
    SHUTDOWN_ALIASES = ("shutdown_jarvis", "shut_down", "exit_jarvis",
                        "quit_jarvis", "power_off_jarvis", "turn_off_jarvis")

    def setUp(self):
        wi._action_last_call.clear()
        self.addCleanup(wi._action_last_call.clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tray = os.path.join(self.tmp.name, "tray_commands.json")
        self.shutdown = mock.Mock(return_value="bye")
        self.restart = mock.Mock(return_value="restarting")
        self.purge = mock.Mock(return_value="purged")
        self.run_py = mock.Mock(return_value="42")
        acts = {name: self.shutdown for name in self.SHUTDOWN_ALIASES}
        acts.update({"restart": self.restart,
                     "forget_alexa_login": self.purge,
                     "smart_home_purge_cookie": self.purge,
                     "run_python": self.run_py, "python": self.run_py,
                     "eval_python": self.run_py, "compute": self.run_py,
                     "get_time": mock.Mock(return_value="noon")})
        self.cfg = {"runtime": _Runtime(acts), "tray_commands_path": self.tray,
                    "action_timeout_s": 2.0}

    def _queued(self):
        with open(self.tray, encoding="utf-8") as f:
            return [e["cmd"] for e in json.load(f)]

    def test_every_alias_of_a_gated_handler_asks_first(self):
        rows = {r["name"]: r for r in wi.actions_payload(self.cfg)["actions"]}
        for name in self.SHUTDOWN_ALIASES + ("smart_home_purge_cookie",
                                             "run_python", "python",
                                             "eval_python", "compute"):
            with self.subTest(name=name):
                self.assertTrue(rows[name]["confirm"], name)
                code, d = wi.run_named_action(self.cfg, name)
                self.assertEqual(code, 409, name)
                self.assertTrue(d["confirm_required"])
        self.shutdown.assert_not_called()
        self.purge.assert_not_called()
        self.run_py.assert_not_called()
        self.assertFalse(os.path.exists(self.tray))

    def test_a_read_still_runs_on_one_click(self):
        code, d = wi.run_named_action(self.cfg, "get_time")
        self.assertEqual((code, d["status"], d["result"]), (200, "done", "noon"))

    def test_an_alias_inherits_its_twins_reason(self):
        # A name no pattern knows, bound to the shutdown handler.
        self.cfg["runtime"]._acts["bye_bye"] = self.shutdown
        self.assertEqual(wi._live_confirm_reason(self.cfg["runtime"]._acts,
                                                 "bye_bye"),
                         "stops or restarts JARVIS or the PC")

    def test_every_shutdown_alias_goes_through_the_tray_control_plane(self):
        for name in self.SHUTDOWN_ALIASES:
            wi._action_last_call.clear()
            code, d = wi.run_named_action(self.cfg, name, confirm=True)
            self.assertEqual((code, d.get("via")), (200, "tray"), name)
        self.shutdown.assert_not_called()   # never on a request thread
        self.assertEqual(self._queued(), ["shutdown"] * len(self.SHUTDOWN_ALIASES))

    def test_restart_still_goes_through_the_tray(self):
        code, d = wi.run_named_action(self.cfg, "restart", confirm=True)
        self.assertEqual(d["via"], "tray")
        self.assertEqual(self._queued(), ["restart"])
        self.restart.assert_not_called()


# ── B076 ───────────────────────────────────────────────────────────────────
class SwitchLlmInstalledTagTests(unittest.TestCase):
    INSTALLED = {"gpt-oss:20b", "laguna-xs-2.1:latest", "gemma4:12b",
                 "nomic-embed-text:latest"}

    def _bc(self):
        bc = mock.Mock()
        bc.AI_BACKEND = "claude"
        bc.OLLAMA_MODEL = "gemma4:12b"
        bc._KNOWN_OLLAMA_MODELS = {"gemma4:12b"}
        bc._get_local_llm_model.return_value = "gemma4:12b"
        bc._ollama_resolve_model.side_effect = \
            lambda t: t if t in self.INSTALLED else None
        bc._RESOLVED_LOCAL_LLM_MODEL = ["gemma4:12b"]
        bc.LOCAL_VISION_MODEL = "some-vlm:7b"     # never moved by these tests
        return bc

    def _switch(self, bc, tag):
        with mock.patch.object(A, "_bc", return_value=bc), \
                mock.patch("core.config.CLAUDE_MODEL", "claude-x"):
            return A._act_switch_llm(tag)

    def test_an_installed_tag_outside_the_family_list_switches(self):
        for tag in ("gpt-oss:20b", "laguna-xs-2.1:latest"):
            with self.subTest(tag=tag):
                bc = self._bc()
                out = self._switch(bc, tag)
                self.assertNotIn("unknown backend tag", out)
                self.assertIn(f"switched to ollama / {tag}", out)
                self.assertEqual(bc.AI_BACKEND, "ollama")
                self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], tag)

    def test_an_embedding_model_is_never_the_chat_brain(self):
        bc = self._bc()
        out = self._switch(bc, "nomic-embed-text:latest")
        self.assertIn("unknown backend tag", out)
        self.assertEqual(bc.AI_BACKEND, "claude")
        self.assertEqual(bc._RESOLVED_LOCAL_LLM_MODEL[0], "gemma4:12b")

    def test_an_uninstalled_unknown_tag_is_still_refused(self):
        bc = self._bc()
        out = self._switch(bc, "turbotron9000")
        self.assertIn("unknown backend tag", out)
        bc._ollama_pull_async.assert_not_called()


# ── B077 ───────────────────────────────────────────────────────────────────
class TrayCommandCidTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "tray_commands.json")

    def _entries(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def test_web_commands_carry_a_unique_cid(self):
        wi.send_tray_command("mic_mute_toggle", self.path)
        wi.send_tray_command("mic_mute_toggle", self.path)
        cids = [e.get("cid") for e in self._entries()]
        self.assertTrue(all(isinstance(c, str) and c for c in cids), cids)
        self.assertEqual(len(set(cids)), 2)

    def test_settings_window_commands_carry_a_unique_cid(self):
        self.assertTrue(sw.send_tray_command("restart", self.path))
        self.assertTrue(sw.send_tray_command("restart", self.path))
        cids = [e.get("cid") for e in self._entries()]
        self.assertTrue(all(isinstance(c, str) and c for c in cids), cids)
        self.assertEqual(len(set(cids)), 2)

    def test_the_writers_cids_cannot_collide(self):
        wi.send_tray_command("force_wake", self.path)
        sw.send_tray_command("restart", self.path)
        a, b = (e["cid"] for e in self._entries())
        self.assertNotEqual(a[0], b[0], "web and Settings ids share a prefix")


if __name__ == "__main__":
    unittest.main()
