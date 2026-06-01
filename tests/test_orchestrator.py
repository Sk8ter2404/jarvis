"""Tests for core.orchestrator's worker execution — specifically the
no-fabrication contract added 2026-05-31: a worker must run a REAL registered
action and summarise only that, and must return EMPTY (never invent) when no
allowed_action is registered. These pin the safety guarantee that lets the
orchestrator be enabled on the live assistant."""
import unittest

import core.orchestrator as orch


def _spec(allowed):
    return orch.SubAgentSpec(
        name="t_agent",
        description="test agent",
        allowed_actions=list(allowed),
        model_preference="haiku",
        system_prompt="Summarise tersely.",
    )


def _run(spec, actions, args=None):
    task = orch.SubTask(sub_agent="t_agent", task="do the thing", args=args or {})
    return orch._run_worker_sync(
        spec, task, actions,
        worker_model="claude-test",
        local_model="x",            # truthy → never imports bobert_companion
        local_base_url="http://localhost:0",
        timeout_s=5.0,
    )


class WorkerNoFabricationTests(unittest.TestCase):
    def setUp(self):
        # Save the LLM/ollama seams we monkeypatch so each test is isolated.
        self._claude = orch._claude_call
        self._ollama = orch._ollama_call
        self._reach = orch._ollama_reachable
        self._resolve = orch._resolve_local_model

    def tearDown(self):
        orch._claude_call = self._claude
        orch._ollama_call = self._ollama
        orch._ollama_reachable = self._reach
        orch._resolve_local_model = self._resolve

    def test_no_registered_action_returns_empty(self):
        # Spec's only action is NOT in the actions dict → must return empty
        # (so the merger omits it) and never call the LLM (no fabrication).
        calls = {"claude": 0}
        orch._claude_call = lambda *a, **k: calls.__setitem__("claude", calls["claude"] + 1) or "FABRICATED"
        res = _run(_spec(["unregistered_action"]), actions={})
        self.assertEqual(res.output, "")
        self.assertIn("no registered tool", (res.error or "").lower())
        self.assertEqual(calls["claude"], 0)     # LLM never invoked

    def test_runs_real_action_and_summarises_only_it(self):
        called = {"n": 0}

        def fake_read(arg):
            called["n"] += 1
            return "REAL_SENSOR_VALUE_42"

        # LLM echoes its user message so we can confirm the worker fed it the
        # REAL action output (not the bare task description).
        orch._claude_call = lambda model, system, user, **k: user
        res = _run(_spec(["fake_read"]), actions={"fake_read": fake_read})
        self.assertEqual(called["n"], 1)                       # real action ran
        self.assertIn("REAL_SENSOR_VALUE_42", res.output)      # fed to the LLM
        self.assertEqual(res.error, None)

    def test_llm_down_falls_back_to_raw_real_data(self):
        def boom(*a, **k):
            raise RuntimeError("claude down")

        orch._claude_call = boom
        orch._resolve_local_model = lambda *a, **k: None        # no ollama fallback
        res = _run(_spec(["fake_read"]),
                   actions={"fake_read": lambda a: "RAW_REAL_DATA"})
        # LLM unavailable → return the RAW real data, still real, never invented.
        self.assertEqual(res.output, "RAW_REAL_DATA")
        self.assertEqual(res.error, None)

    def test_explicit_direct_action_executes_without_llm(self):
        calls = {"claude": 0}
        orch._claude_call = lambda *a, **k: calls.__setitem__("claude", calls["claude"] + 1) or "x"
        res = _run(
            _spec(["fake_read"]),
            actions={"fake_read": lambda a: f"GOT:{a}"},
            args={"direct_action": "fake_read", "arg": "hello"},
        )
        self.assertEqual(res.output, "GOT:hello")
        self.assertEqual(res.model_used, "direct")
        self.assertEqual(calls["claude"], 0)


if __name__ == "__main__":
    unittest.main()
