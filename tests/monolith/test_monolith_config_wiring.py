"""Monolith side of the 2026-10-01 config-wiring fixes.

Each class drives the REAL bobert_companion code for one bug in the
config-wiring audit batch: a setting the owner chose (local-only backend,
privacy blocklist, a Settings audio toggle, ambient off, "switch to Claude")
that one stale copy of a code path ignored.

No device, window, network or live settings file is touched: captures are
faked, the orchestrator is a stub module, hud_state.json and
user_settings.json live in temp dirs.

    python -B -m unittest tests.monolith.test_monolith_config_wiring
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        import core.config as cfg
        self.cfg = cfg

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _quiet(self, fn, *a, **k):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)

    def _settings_file(self, doc):
        d = tempfile.mkdtemp(prefix="cfg_wiring_mono_")
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "user_settings.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        env = mock.patch.dict(os.environ, {"JARVIS_SETTINGS_PATH": path})
        env.start()
        self.addCleanup(env.stop)
        return path


# ──────────────────────────────────────────────────────────────────────────
#  B004 — the briefing orchestrator follows the chat path's cloud gate
# ──────────────────────────────────────────────────────────────────────────

class OrchestratorCloudGateTests(_Base):
    def _run(self, backend, route, key="sk-test-not-real"):
        bc = self.bc
        captured = {}
        fake = types.ModuleType("core.orchestrator")

        def _orchestrate(text, actions, **kw):
            captured.update(kw)
            return "Your brief, sir."
        fake.orchestrate = _orchestrate
        self._p(bc, "_orchestrator_enabled", return_value=True)
        self._p(bc, "_is_orchestration_request", return_value=True)
        self._p(bc, "AI_BACKEND", backend)
        self._p(bc, "set_state")
        self._p(bc, "_speak")
        self._p(bc, "_append_turn")
        self.cfg.MODEL_ROUTING["chat"] = route    # harness-owned dict
        with mock.patch.dict(sys.modules, {"core.orchestrator": fake}), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": key}):
            handled = self._quiet(bc._maybe_orchestrate, "morning briefing")
        self.assertTrue(handled)
        return captured

    def test_local_only_backend_keeps_every_stage_off_the_cloud(self):
        # The owner's settings: AI_BACKEND=ollama, chat routed local, a key
        # still in the environment.
        self.assertIs(self._run("ollama", "local")["cloud_allowed"], False)

    def test_claude_backend_with_chat_routed_local_stays_local(self):
        self.assertIs(self._run("claude", "local")["cloud_allowed"], False)

    def test_no_key_means_no_cloud(self):
        self.assertIs(self._run("claude", "auto", key="")["cloud_allowed"],
                      False)

    def test_claude_backend_still_uses_the_cloud(self):
        self.assertIs(self._run("claude", "auto")["cloud_allowed"], True)


# ──────────────────────────────────────────────────────────────────────────
#  B049 — the glance capture honours SCREENSHOT_PRIVACY_BLOCKLIST
# ──────────────────────────────────────────────────────────────────────────

class _GrabSpy:
    """A fake mss whose grab() records that a capture happened."""
    def __init__(self):
        self.grabs = []

    def module(self):
        spy = self

        class _Sct:
            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False

            def grab(self_, region):
                spy.grabs.append(region)
                return types.SimpleNamespace(size=(800, 600),
                                             bgra=b"\x00" * 16)
        mod = types.ModuleType("mss")
        mod.mss = lambda: _Sct()
        return mod


class GlancePrivacyTests(_Base):
    def setUp(self):
        super().setUp()
        self._p(self.cfg, "SCREENSHOT_PRIVACY_BLOCKLIST", ["passvault"])
        self.spy = _GrabSpy()

    def _capture(self):
        with mock.patch.dict(self.bc.sys.modules, {"mss": self.spy.module()}):
            return self._quiet(self.bc._capture_focused_window_png)

    def test_capture_refused_when_the_focused_window_is_blocklisted(self):
        bc = self.bc
        self._p(bc, "_focused_window_state",
                {"rect": (0, 0, 800, 600), "title": "PassVault - Vault"})
        self._p(bc, "_read_focused_window",
                return_value=(1, "PassVault - Vault", (0, 0, 800, 600)))
        self.assertIsNone(self._capture())
        self.assertEqual(self.spy.grabs, [])        # nothing was captured

    def test_capture_refused_on_the_trackers_cached_title_too(self):
        # The rect comes from the tracker's cache; its title is what was
        # focused when that rect was recorded.
        bc = self.bc
        self._p(bc, "_focused_window_state",
                {"rect": (0, 0, 800, 600), "title": "passvault login"})
        self._p(bc, "_read_focused_window",
                return_value=(1, "Editor", (0, 0, 800, 600)))
        self.assertIsNone(self._capture())
        self.assertEqual(self.spy.grabs, [])

    def test_glance_on_a_private_window_never_reaches_vision(self):
        bc = self.bc
        self._p(bc, "_is_glance_ambiguous_question", return_value=True)
        self._p(bc, "_focus_changed_recently", return_value=True)
        self._p(bc, "SCREEN_VISION_ENABLED", True)
        self._p(bc, "_vision_click_backend_available", return_value=True)
        vision = self._p(bc, "ask_vision", return_value="It's your vault.")
        self._p(bc, "_focused_window_state",
                {"rect": (0, 0, 800, 600), "title": "PassVault - Vault"})
        self._p(bc, "_read_focused_window",
                return_value=(1, "PassVault - Vault", (0, 0, 800, 600)))
        with mock.patch.dict(bc.sys.modules, {"mss": self.spy.module()}):
            out = self._quiet(bc.maybe_glance_response, "what's this?")
        self.assertIsNone(out)
        vision.assert_not_called()
        self.assertEqual(self.spy.grabs, [])


# ──────────────────────────────────────────────────────────────────────────
#  B051 — a Settings change to an audio-cleanup / VAD-debug row wins
# ──────────────────────────────────────────────────────────────────────────

class AudioToggleRestoreTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        d = tempfile.mkdtemp(prefix="cfg_wiring_hud_")
        self.addCleanup(shutil.rmtree, d, True)
        self.hud = os.path.join(d, "hud_state.json")
        self._p(bc, "HUD_STATE_FILE", self.hud)
        self._p(bc, "HUD_ENABLED", True)
        self._p(bc, "ACTIONS", {})
        self.ns = self._p(bc, "_audio_ns_enabled", [False])
        self.dbg = self._p(bc, "_debug_mode", [False])

    def _restore(self, doc):
        with open(self.hud, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        self._quiet(self.bc._restore_tray_toggle_state)
        with open(self.hud, encoding="utf-8") as f:
            return json.load(f)

    def test_settings_change_since_the_tray_value_was_saved_wins(self):
        # The tray saved NS on while Settings also said on; the owner then
        # unticked NS in Settings (config now False) and restarted.
        self._p(self.cfg, "AUDIO_NOISE_SUPPRESS", False)
        self.ns[0] = False                       # core.state seeds from config
        out = self._restore({"noise_suppress_enabled": True,
                             "toggle_cfg_seed": {"noise_suppress_enabled": True}})
        self.assertIs(self.ns[0], False)
        self.assertIs(out["noise_suppress_enabled"], False)
        self.assertIs(out["toggle_cfg_seed"]["noise_suppress_enabled"], False)

    def test_vad_debug_setting_change_wins_too(self):
        self._p(self.cfg, "VAD_DEBUG", False)
        self._restore({"debug_mode": True,
                       "toggle_cfg_seed": {"debug_mode": True}})
        self.assertIs(self.dbg[0], False)

    def test_tray_flip_survives_while_the_setting_is_unchanged(self):
        self._p(self.cfg, "AUDIO_NOISE_SUPPRESS", True)
        self.ns[0] = True
        self._restore({"noise_suppress_enabled": False,
                       "toggle_cfg_seed": {"noise_suppress_enabled": True}})
        self.assertIs(self.ns[0], False)

    def test_file_from_before_the_seed_keeps_the_tray_value_once(self):
        self._p(self.cfg, "AUDIO_NOISE_SUPPRESS", False)
        self.ns[0] = False
        out = self._restore({"noise_suppress_enabled": True})
        self.assertIs(self.ns[0], True)
        self.assertIn("toggle_cfg_seed", out)    # recorded for next boot


# ──────────────────────────────────────────────────────────────────────────
#  B052 / B053 — ambient off really stops, and stays off after a restart
# ──────────────────────────────────────────────────────────────────────────

class AmbientToggleTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        thread = mock.Mock()
        thread.is_alive.return_value = True
        # AMBIENT_LISTEN_ENABLED auto-started the daemon; the cell is False.
        self.amb = types.SimpleNamespace(_thread=thread)
        self.start = mock.Mock(return_value="on")
        self.stop = mock.Mock(return_value="off")
        self._p(bc, "_ambient_mode_active", [False])
        self._p(bc, "_write_hud_state")
        mods = mock.patch.dict(sys.modules, {
            "skill_ambient_listen": self.amb,
            "skill_ambient_multimodal_extract": None})
        mods.start()
        self.addCleanup(mods.stop)
        acts = mock.patch.dict(bc.ACTIONS, {"ambient_listen_start": self.start,
                                            "ambient_listen_stop": self.stop})
        acts.start()
        self.addCleanup(acts.stop)

    def test_voice_toggle_stops_the_running_daemon(self):
        out = self._quiet(self.bc.ACTIONS["ambient_mode"], "")
        self.stop.assert_called_once_with("")
        self.start.assert_not_called()
        self.assertIn("off", out)

    def test_voice_off_is_saved_for_the_next_boot(self):
        bc = self.bc
        path = self._settings_file({"AMBIENT_LISTEN_ENABLED": True})
        self._p(bc, "_is_staging", return_value=False)
        self._quiet(bc.ACTIONS["stop_eavesdropping"], "")
        with open(path, encoding="utf-8") as f:
            self.assertIs(json.load(f)["AMBIENT_LISTEN_ENABLED"], False)
        self.assertIs(bc.AMBIENT_LISTEN_ENABLED, False)   # live learner gate

    def test_tray_toggle_off_is_saved_for_the_next_boot(self):
        bc = self.bc
        path = self._settings_file({"AMBIENT_LISTEN_ENABLED": True})
        self._p(bc, "_is_staging", return_value=False)
        self._quiet(bc._dispatch_tray_command, "ambient_mode_toggle", {})
        self.stop.assert_called_once_with("")
        with open(path, encoding="utf-8") as f:
            self.assertIs(json.load(f)["AMBIENT_LISTEN_ENABLED"], False)


# ──────────────────────────────────────────────────────────────────────────
#  B054 — "switch to Claude" really takes the next turn off the local branch
# ──────────────────────────────────────────────────────────────────────────

class SwitchToClaudeTests(_Base):
    def test_switch_llm_claude_leaves_the_local_branch(self):
        bc = self.bc
        self._p(bc, "AI_BACKEND", "ollama")
        self._p(bc, "_write_hud_state")
        self.cfg.MODEL_ROUTING["chat"] = "local"     # the owner's routing
        self.assertTrue(bc._chat_takes_local_branch())
        with mock.patch.dict(os.environ,
                             {"ANTHROPIC_API_KEY": "sk-test-not-real"}):
            out = bc.ACTIONS["switch_llm"]("claude")
            self.assertTrue(bc._claude_reachable())
        self.assertIn("switched to claude", out)
        self.assertFalse(bc._chat_takes_local_branch())

    def test_switch_llm_ollama_puts_the_turn_back_on_the_local_branch(self):
        bc = self.bc
        self._p(bc, "AI_BACKEND", "claude")
        self._p(bc, "_write_hud_state")
        self._p(bc, "_get_local_llm_model", return_value="local-model:1b")
        self.cfg.MODEL_ROUTING["chat"] = "cloud"
        bc.ACTIONS["switch_llm"]("ollama")
        self.assertTrue(bc._chat_takes_local_branch())
        self.assertEqual(bc.AI_BACKEND, "ollama")


if __name__ == "__main__":
    unittest.main()
