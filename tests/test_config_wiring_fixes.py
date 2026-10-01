"""Light-tier regression tests for the 2026-10-01 config-wiring fixes.

Each class pins one bug from the config-wiring audit batch: a setting or a
toggle that the code claimed to honour but that a stale duplicate path
ignored. All tests are offline and hermetic: no network, no real Ollama or
Anthropic client, and every settings write goes to a temp file.

    python -B -m unittest tests.test_config_wiring_fixes
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import core.actions as A
import core.config as cfg
import core.model_catalog as mc
import core.orchestrator as orch
from tests._skill_harness import load_skill_isolated
from tools import settings_window as sw


# ──────────────────────────────────────────────────────────────────────────
#  B004 — a local-only backend must keep the orchestrator off the cloud
# ──────────────────────────────────────────────────────────────────────────

class OrchestratorCloudGateTests(unittest.TestCase):
    """"Morning briefing" fans out to a planner, Haiku workers and a merger.
    Each stage called Claude FIRST and used Ollama only after a Claude
    exception, so on an AI_BACKEND=ollama install inbox / news / system data
    still went to the cloud. With cloud_allowed=False no stage may call
    Claude; the existing local-Ollama and raw-data fallbacks take over."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="orch_cloud_gate_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        saved = (orch._claude_call, orch._ollama_call, orch._ollama_reachable,
                 orch._resolve_local_model, orch._default_orchestrator)

        def _restore():
            (orch._claude_call, orch._ollama_call, orch._ollama_reachable,
             orch._resolve_local_model, orch._default_orchestrator) = saved
        self.addCleanup(_restore)
        orch._default_orchestrator = None
        self.claude_calls = []

        def _no_cloud(*a, **k):
            self.claude_calls.append(a)
            return "CLOUD"
        orch._claude_call = _no_cloud
        with open(os.path.join(self.dir, "inbox.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"name": "inbox", "description": "reads the inbox",
                       "allowed_actions": ["email_briefing"],
                       "model_preference": "haiku"}, f)
        self.actions = {"email_briefing": lambda a: "2 urgent: Alice, Bob"}

    def test_pipeline_runs_fully_local_when_cloud_disallowed(self):
        local = []
        replies = iter([
            json.dumps({"sub_tasks": [{"sub_agent": "inbox",
                                       "task": "summarise"}]}),
            "two urgent mails",
            "LOCAL BRIEF",
        ])

        def _ollama(model, system, user, **k):
            local.append(model)
            return next(replies)
        orch._ollama_call = _ollama
        orch._ollama_reachable = lambda *a, **k: True
        orch._resolve_local_model = lambda configured: "local-model"
        out = orch.orchestrate("morning briefing", self.actions,
                               cloud_allowed=False, specs_dir=self.dir)
        self.assertEqual(out, "LOCAL BRIEF")
        self.assertEqual(self.claude_calls, [])        # nothing left the PC
        self.assertEqual(local, ["local-model"] * 3)   # planner+worker+merger

    def test_no_local_model_degrades_to_raw_data_not_cloud(self):
        orch._ollama_reachable = lambda *a, **k: False
        orch._resolve_local_model = lambda configured: None
        o = orch.Orchestrator(specs_dir=self.dir)
        out = o.orchestrate("morning briefing", self.actions,
                            cloud_allowed=False)
        self.assertEqual(self.claude_calls, [])
        self.assertIn("Alice", out)                    # raw, real tool data

    def test_cloud_allowed_still_uses_claude(self):
        replies = iter([
            json.dumps({"sub_tasks": [{"sub_agent": "inbox",
                                       "task": "summarise"}]}),
            "W", "CLOUD BRIEF",
        ])
        orch._claude_call = lambda *a, **k: next(replies)
        o = orch.Orchestrator(specs_dir=self.dir)
        self.assertEqual(o.orchestrate("morning briefing", self.actions),
                         "CLOUD BRIEF")


# ──────────────────────────────────────────────────────────────────────────
#  B050 — browser agent: no sampling params, honest failure, no "400" = cap
# ──────────────────────────────────────────────────────────────────────────

def _run_coro(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _ChatAnthropicLike:
    """Shape of browser-use's ChatAnthropic: temperature is an optional field
    (default None) that, when set, is sent on every messages.create."""
    def __init__(self, model=None, temperature=None, max_tokens=None):
        self.model = model
        self.temperature = temperature


class _FailedHistory:
    """An AgentHistoryList whose every step failed: no final result, one
    error string per step (browser-use's errors() contract)."""
    def final_result(self):
        return None

    def is_done(self):
        return False

    def errors(self):
        return [None, "Error code: 400 - temperature is not supported",
                "Error code: 400 - temperature is not supported"]

    def __str__(self):
        return "AgentHistoryList(all_results=[ActionResult(error=...)])"


class _FailingAgent:
    def __init__(self, **kwargs):
        pass

    async def run(self, **kwargs):
        return _FailedHistory()


class BrowserAgentConfigWiringTests(unittest.TestCase):
    def setUp(self):
        self.mod, _ = load_skill_isolated("browser_agent")
        saved = dict(self.mod._state)

        def _restore():
            self.mod._state.clear()
            self.mod._state.update(saved)
        self.addCleanup(_restore)

    def test_llm_is_built_without_a_temperature(self):
        # Sonnet 5 rejects a non-default temperature with HTTP 400, so the
        # agent must leave it unset (None = the API default).
        with mock.patch.object(self.mod, "_model_name",
                               return_value="claude-sonnet-5"):
            llm = self.mod._make_llm({"ChatAnthropic": _ChatAnthropicLike})
        self.assertEqual(llm.model, "claude-sonnet-5")
        self.assertIsNone(llm.temperature)

    def test_all_steps_failed_reports_the_error_not_a_history_dump(self):
        imports = {"Agent": _FailingAgent, "Browser": object,
                   "BrowserConfig": None, "ChatAnthropic": _ChatAnthropicLike}
        with mock.patch.object(self.mod, "_bu_imports", return_value=imports), \
                mock.patch.object(self.mod, "_make_browser",
                                  new=mock.AsyncMock(return_value=object())):
            out = _run_coro(self.mod._run_task_inner("find a thing", 5, True))
        self.assertNotIn("AgentHistoryList", out)
        self.assertIn("couldn't finish", out)
        self.assertIn("temperature is not supported", out)

    def test_plain_bad_request_is_not_blamed_on_the_cap(self):
        err = RuntimeError("Error code: 400 - invalid_request_error: "
                           "messages.0.content is empty")
        with mock.patch.object(self.mod, "_run_task_inner",
                               new=mock.AsyncMock(side_effect=err)), \
                mock.patch.object(self.mod, "_close_browser_async",
                                  new=mock.AsyncMock()):
            out = _run_coro(self.mod._orchestrate("task", 5, True))
        self.assertNotIn("capped", out)
        self.assertIn("Browser agent failed", out)

    def test_real_usage_cap_still_reported_as_the_cap(self):
        err = RuntimeError("Error code: 400 - You have reached your "
                           "specified API usage limits.")
        with mock.patch.object(self.mod, "_run_task_inner",
                               new=mock.AsyncMock(side_effect=err)), \
                mock.patch.object(self.mod, "_close_browser_async",
                                  new=mock.AsyncMock()):
            out = _run_coro(self.mod._orchestrate("task", 5, True))
        self.assertIn("capped until it resets", out)


# ──────────────────────────────────────────────────────────────────────────
#  Shared: a throwaway settings file + core.config restore
# ──────────────────────────────────────────────────────────────────────────

class _SettingsFileCase(unittest.TestCase):
    """Points every settings read/write at a temp file (settings_path()
    honours JARVIS_SETTINGS_PATH at call time) and restores the core.config
    globals the setters under test mutate."""

    _CFG_KEYS = ("AMBIENT_LISTEN_ENABLED", "GREET_NEW_PEOPLE_ENABLED")

    def setUp(self):
        d = tempfile.mkdtemp(prefix="cfg_wiring_")
        self.addCleanup(shutil.rmtree, d, True)
        self.path = os.path.join(d, "user_settings.json")
        env = mock.patch.dict(os.environ, {"JARVIS_SETTINGS_PATH": self.path})
        env.start()
        self.addCleanup(env.stop)
        for k in self._CFG_KEYS:
            self.addCleanup(setattr, cfg, k, getattr(cfg, k))
        routing = cfg.MODEL_ROUTING
        saved = dict(routing)
        self.addCleanup(lambda: (routing.clear(), routing.update(saved)))

    def write(self, doc):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(doc, f)

    def read(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)


# ──────────────────────────────────────────────────────────────────────────
#  B053 — "ambient mode off" must survive a restart
# ──────────────────────────────────────────────────────────────────────────

class AmbientOffPersistsTests(_SettingsFileCase):
    """The boot autostart reads AMBIENT_LISTEN_ENABLED only, so the voice
    "off" has to land there (live and on disk), not just in hud_state."""

    def _bc(self):
        bc = mock.Mock()
        bc._ambient_mode_active = [True]
        bc.AMBIENT_LISTEN_ENABLED = True
        bc.ACTIONS = {"ambient_listen_stop": mock.Mock(return_value=""),
                      "ambient_listen_start": mock.Mock(return_value="")}
        bc._is_staging = lambda: False
        return bc

    def test_off_is_saved_and_applied_live(self):
        self.write({"AMBIENT_LISTEN_ENABLED": True, "OTHER_KEY": "keep"})
        cfg.AMBIENT_LISTEN_ENABLED = True
        bc = self._bc()
        with mock.patch.object(A, "_bc", return_value=bc), \
                mock.patch.dict(sys.modules,
                                {"skill_ambient_multimodal_extract": None}):
            out = A._act_ambient_mode_set(False)
        doc = self.read()
        self.assertIs(doc["AMBIENT_LISTEN_ENABLED"], False)
        self.assertEqual(doc["OTHER_KEY"], "keep")
        self.assertIs(bc.AMBIENT_LISTEN_ENABLED, False)   # gates live learning
        self.assertIs(cfg.AMBIENT_LISTEN_ENABLED, False)
        bc.ACTIONS["ambient_listen_stop"].assert_called_once_with("")
        self.assertNotIn("couldn't save", out)

    def test_reply_says_so_when_the_save_fails(self):
        bc = self._bc()
        with mock.patch.object(A, "_bc", return_value=bc), \
                mock.patch.object(sw, "update_settings",
                                  side_effect=OSError("read-only")), \
                mock.patch.dict(sys.modules,
                                {"skill_ambient_multimodal_extract": None}):
            out = A._act_ambient_mode_set(False)
        self.assertIn("couldn't save that for next boot", out)


# ──────────────────────────────────────────────────────────────────────────
#  B052 — the voice toggle flips from what is really running
# ──────────────────────────────────────────────────────────────────────────

class AmbientVoiceToggleTests(unittest.TestCase):
    def test_toggle_stops_a_running_daemon_even_with_the_cell_false(self):
        # AMBIENT_LISTEN_ENABLED auto-started the daemon; the cell is False.
        bc = mock.Mock()
        bc._ambient_mode_active = [False]
        bc._ambient_effective_on.return_value = True
        with mock.patch.object(A, "_bc", return_value=bc), \
                mock.patch.object(A, "_act_ambient_mode_set",
                                  return_value="off") as mset:
            A._act_ambient_mode_toggle("")
        mset.assert_called_once_with(False)


# ──────────────────────────────────────────────────────────────────────────
#  B100 — "stop greeting people" must survive a restart
# ──────────────────────────────────────────────────────────────────────────

class GreetNewPeoplePersistsTests(_SettingsFileCase):
    def test_off_is_saved_over_an_owner_file_that_has_it_on(self):
        self.write({"GREET_NEW_PEOPLE_ENABLED": True, "OTHER_KEY": "keep"})
        out = A._act_greet_new_people_set(False)
        doc = self.read()
        self.assertIs(doc["GREET_NEW_PEOPLE_ENABLED"], False)
        self.assertEqual(doc["OTHER_KEY"], "keep")
        self.assertIs(cfg.GREET_NEW_PEOPLE_ENABLED, False)
        self.assertNotIn("couldn't save", out)

    def test_on_is_saved_too(self):
        self.write({"GREET_NEW_PEOPLE_ENABLED": False})
        A._act_greet_new_people_set(True)
        self.assertIs(self.read()["GREET_NEW_PEOPLE_ENABLED"], True)

    def test_reply_says_so_when_the_save_fails(self):
        with mock.patch.object(sw, "update_settings",
                               side_effect=OSError("read-only")):
            out = A._act_greet_new_people_set(False)
        self.assertIn("couldn't save that for next boot", out)
        self.assertIs(cfg.GREET_NEW_PEOPLE_ENABLED, False)   # still live


# ──────────────────────────────────────────────────────────────────────────
#  B054 — "switch to Claude" must move the chat route, and vice versa
# ──────────────────────────────────────────────────────────────────────────

class ChatBrainSwitchTests(_SettingsFileCase):
    """_call_llm takes the local branch whenever model_route('chat') is
    'local', whatever AI_BACKEND says; set_brain only moved the route and
    switch_llm only moved AI_BACKEND, so neither alone switched brains."""

    def _bc(self, backend):
        bc = mock.Mock()
        bc.AI_BACKEND = backend
        bc.MODEL_ROUTING = cfg.MODEL_ROUTING      # the star-import alias
        bc._KNOWN_OLLAMA_MODELS = {"qwen2.5:14b"}
        bc._get_local_llm_model.return_value = "qwen2.5:14b"
        bc._ollama_resolve_model.side_effect = lambda t: t
        bc._RESOLVED_LOCAL_LLM_MODEL = ["qwen2.5:14b"]
        return bc

    def test_switch_to_claude_takes_chat_off_the_local_route(self):
        cfg.MODEL_ROUTING.update({"chat": "local", "vision": "local"})
        bc = self._bc("ollama")
        with mock.patch.object(A, "_bc", return_value=bc):
            out = A._act_switch_llm("anthropic")
        self.assertEqual(bc.AI_BACKEND, "claude")
        self.assertEqual(cfg.model_route("chat"), "cloud")
        self.assertEqual(cfg.model_route("vision"), "local")   # untouched
        self.assertIn("switched to claude", out)

    def test_switch_to_ollama_puts_chat_on_the_local_route(self):
        cfg.MODEL_ROUTING.update({"chat": "cloud"})
        bc = self._bc("claude")
        with mock.patch.object(A, "_bc", return_value=bc):
            A._act_switch_llm("ollama")
        self.assertEqual(bc.AI_BACKEND, "ollama")
        self.assertEqual(cfg.model_route("chat"), "local")

    def test_set_brain_cloud_also_moves_and_saves_ai_backend(self):
        from skills import model_picker as M
        self.write({"AI_BACKEND": "ollama",
                    "MODEL_ROUTING": {"chat": "local", "vision": "local",
                                      "ambient": "local"}})
        cfg.MODEL_ROUTING.update({"chat": "local"})
        fake = types.ModuleType("fake_monolith")
        fake.AI_BACKEND = "ollama"
        fake.MODEL_ROUTING = cfg.MODEL_ROUTING
        with mock.patch.object(M, "_monolith", return_value=fake):
            M.set_brain("cloud")
        self.assertEqual(fake.AI_BACKEND, "claude")
        self.assertEqual(cfg.model_route("chat"), "cloud")
        doc = self.read()
        self.assertEqual(doc["AI_BACKEND"], "claude")
        self.assertEqual(doc["MODEL_ROUTING"]["chat"], "cloud")
        self.assertEqual(doc["MODEL_ROUTING"]["vision"], "local")

    def test_set_brain_local_moves_ai_backend_to_ollama(self):
        from skills import model_picker as M
        fake = types.ModuleType("fake_monolith")
        fake.AI_BACKEND = "claude"
        fake.MODEL_ROUTING = cfg.MODEL_ROUTING
        with mock.patch.object(M, "_monolith", return_value=fake):
            M.set_brain("local")
        self.assertEqual(fake.AI_BACKEND, "ollama")
        self.assertEqual(self.read()["AI_BACKEND"], "ollama")


# ──────────────────────────────────────────────────────────────────────────
#  B101 — a voice toggle must not rewrite keys it never touched
# ──────────────────────────────────────────────────────────────────────────

class ToggleSaveKeepsHandSetValuesTests(_SettingsFileCase):
    def test_hand_set_model_outside_the_choices_survives_a_toggle(self):
        # A hand-edited value the enum's choices don't list (a newer model id
        # than the schema knows) loads as the default...
        self.write({"CLAUDE_MODEL": "claude-hypothetical-9",
                    "REQUIRE_WAKE_MODE": True})
        cur = sw.load_settings()
        self.assertNotEqual(cur["CLAUDE_MODEL"], "claude-hypothetical-9")
        # ...and the voice toggles' load -> change one key -> save pattern
        # used to write that default back over it.
        cur["REQUIRE_WAKE_MODE"] = False
        sw.save_settings(cur)
        doc = self.read()
        self.assertEqual(doc["CLAUDE_MODEL"], "claude-hypothetical-9")
        self.assertIs(doc["REQUIRE_WAKE_MODE"], False)

    def test_a_real_change_to_that_key_is_still_written(self):
        self.write({"CLAUDE_MODEL": "claude-hypothetical-9"})
        cur = sw.load_settings()
        cur["CLAUDE_MODEL"] = "claude-haiku-4-5"
        sw.save_settings(cur)
        self.assertEqual(self.read()["CLAUDE_MODEL"], "claude-haiku-4-5")


# ──────────────────────────────────────────────────────────────────────────
#  B102 — the spoken cost readout prices Sonnet 5 at its list price
# ──────────────────────────────────────────────────────────────────────────

class SonnetFivePriceTests(unittest.TestCase):
    def test_sonnet_5_is_two_and_ten_per_mtok(self):
        m = mc.by_id("claude-sonnet-5")
        self.assertEqual((m.in_price, m.out_price), (2.0, 10.0))
        # 12000/1e6*2 + 1500/1e6*10 = 0.024 + 0.015
        self.assertAlmostEqual(
            m.cost_per_conversation(in_tokens=12000, out_tokens=1500),
            0.039, places=4)

    def test_cloud_rows_still_read_cheapest_first(self):
        cloud = [m for m in mc.catalog() if m.backend == "claude"]
        costs = [m.cost_per_conversation(in_tokens=12000, out_tokens=1500)
                 for m in cloud]
        self.assertEqual(costs, sorted(costs))


if __name__ == "__main__":
    unittest.main()
