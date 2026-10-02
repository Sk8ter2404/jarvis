"""Monolith side of the 2026-10-02 local-first / retired-model work.

* ``_maybe_orchestrate`` honours ORCHESTRATOR_BACKEND ("local" sends every
  stage down its existing local path) and resolves a blank
  ORCHESTRATOR_WORKER_MODEL to CLAUDE_FAST_MODEL.
* ``ask_vision`` / ``ask_vision_multi``: a SCREEN_VISION_MODEL Anthropic
  answered not_found for goes to the local VLM — the first time through the
  SDK's own 404 handler, after that through core.llm_client's
  RetiredModelError, with no repeat network call and ONE guard log line.

No device, network or live settings file is touched: the orchestrator is a
stub module, the Anthropic client a fake, and the local VLM a mock.

    python -B -m unittest tests.monolith.test_monolith_local_features
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        import core.config as cfg
        import core.claude_model_guard as mg
        self.cfg = cfg
        self.mg = mg
        mg.GUARD.reset()
        self.addCleanup(mg.GUARD.reset)

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _mark_gone(self, model):
        with contextlib.redirect_stdout(io.StringIO()):
            self.mg.GUARD.note_not_found(model)


class OrchestratorBackendTests(_Base):
    def _run(self, backend="claude", worker=""):
        bc = self.bc
        fake = types.ModuleType("core.orchestrator")
        fake.orchestrate = mock.Mock(return_value="Your brief, sir.")
        self._p(bc, "_orchestrator_enabled", return_value=True)
        self._p(bc, "_is_orchestration_request", return_value=True)
        self._p(bc, "_chat_cloud_allowed", return_value=True)
        self._p(bc, "ORCHESTRATOR_BACKEND", backend)
        self._p(bc, "ORCHESTRATOR_WORKER_MODEL", worker)
        self._p(bc, "set_state")
        self._p(bc, "_speak")
        self._p(bc, "_append_turn")
        out = io.StringIO()
        with mock.patch.dict(sys.modules, {"core.orchestrator": fake}), \
                contextlib.redirect_stdout(out):
            handled = bc._maybe_orchestrate("morning briefing")
        return handled, fake.orchestrate, out.getvalue()

    def test_shipped_defaults(self):
        self.assertEqual(self.cfg.ORCHESTRATOR_BACKEND, "claude")
        self.assertEqual(self.cfg.ORCHESTRATOR_WORKER_MODEL, "")
        self.assertEqual(self.bc.CLAUDE_FAST_MODEL, self.cfg.CLAUDE_FAST_MODEL)

    def test_claude_backend_keeps_cloud_stages(self):
        handled, orch, _ = self._run("claude")
        self.assertTrue(handled)
        self.assertIs(orch.call_args.kwargs["cloud_allowed"], True)

    def test_local_backend_sends_every_stage_local(self):
        handled, orch, log = self._run("local")
        self.assertTrue(handled)
        self.assertIs(orch.call_args.kwargs["cloud_allowed"], False)
        self.assertIn("ORCHESTRATOR_BACKEND=local", log)

    def test_blank_worker_model_is_claude_fast_model(self):
        _, orch, _ = self._run(worker="")
        self.assertEqual(orch.call_args.kwargs["worker_model"],
                         self.bc.CLAUDE_FAST_MODEL)
        self.assertEqual(orch.call_args.kwargs["worker_model"], "claude-haiku-4-5")

    def test_explicit_worker_model_wins(self):
        _, orch, _ = self._run(worker="claude-pinned-worker")
        self.assertEqual(orch.call_args.kwargs["worker_model"],
                         "claude-pinned-worker")


class _NotFoundClient:
    """A fake Anthropic client whose model is gone: every messages.create
    raises a genuine anthropic.NotFoundError (built offline)."""

    def __init__(self, model):
        import anthropic
        import httpx
        body = {"type": "error", "error": {"type": "not_found_error",
                                           "message": f"model: {model}"}}
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        self._err = anthropic.NotFoundError(
            f"Error code: 404 - {body}",
            response=httpx.Response(404, request=req, json=body), body=body)
        self.messages = mock.Mock()
        self.messages.create.side_effect = self._err


class VisionRetiredModelTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._p(bc, "SCREEN_VISION_ENABLED", True)
        self._p(bc, "AI_BACKEND", "claude")
        self._p(bc, "SCREEN_VISION_MODEL", "claude-vision-gone")
        self.local = self._p(bc, "_call_local_vision", return_value="a code editor")

    def _ask(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ans = self.bc.ask_vision("what is on screen?", b"PNGBYTES")
        return ans, out.getvalue()

    def test_first_404_then_no_repeat_call_and_one_guard_line(self):
        client = _NotFoundClient("claude-vision-gone")
        self._p(self.bc, "_anthropic_client", return_value=client)
        first, log1 = self._ask()
        self.assertEqual(first, "[local-vision] a code editor")
        self.assertIn("Claude API 404", log1)          # the SDK error's handler
        second, log2 = self._ask()
        third, log3 = self._ask()
        self.assertEqual(second, "[local-vision] a code editor")
        self.assertEqual(third, "[local-vision] a code editor")
        self.assertEqual(client.messages.create.call_count, 1)
        guard_lines = [ln for ln in (log1 + log2 + log3).splitlines()
                       if "[model-guard]" in ln]
        self.assertEqual(len(guard_lines), 1)
        self.assertIn("claude-vision-gone", guard_lines[0])
        # The skipped calls are quiet: no "Claude vision failed" per question.
        self.assertNotIn("Claude vision failed", log2 + log3)
        self.assertEqual(self.local.call_count, 3)

    def test_retired_and_no_local_vision_gives_the_honest_line(self):
        self._mark_gone("claude-vision-gone")
        self.local.return_value = None
        with mock.patch.object(self.bc, "_anthropic_client") as factory:
            ans, _ = self._ask()
        self.assertEqual(ans, self.bc._VISION_MODEL_RETIRED_REPLY)
        self.assertTrue(ans.startswith("(vision failed"))
        factory.return_value.messages.create.assert_not_called()

    def test_configured_successor_answers_vision(self):
        self._mark_gone("claude-vision-gone")
        self._p(self.cfg, "CLAUDE_MODEL_SUCCESSORS",
                {"claude-vision-gone": "claude-sonnet-5-5"})
        blk = types.SimpleNamespace(type="text", text="a terminal window")
        client = mock.Mock()
        client.messages.create.return_value = types.SimpleNamespace(
            content=[blk], stop_reason="end_turn", usage=None)
        self._p(self.bc, "_anthropic_client", return_value=client)
        ans, _ = self._ask()
        self.assertEqual(ans, "a terminal window")
        self.assertEqual(client.messages.create.call_args.kwargs["model"],
                         "claude-sonnet-5-5")
        self.local.assert_not_called()

    def test_multi_monitor_vision_goes_local_quietly(self):
        self._mark_gone("claude-vision-gone")
        out = io.StringIO()
        with mock.patch.object(self.bc, "_anthropic_client") as factory, \
                contextlib.redirect_stdout(out):
            ans = self.bc.ask_vision_multi("anything new?",
                                           {"left": b"A", "right": b"B"})
        self.assertEqual(ans, "[local-vision] a code editor")
        factory.return_value.messages.create.assert_not_called()
        self.assertNotIn("multi-vision failed", out.getvalue())
        pngs = self.local.call_args.args[1]
        self.assertEqual(pngs, [b"A", b"B"])


if __name__ == "__main__":
    unittest.main()
