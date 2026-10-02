"""The monolith's direct Claude paths on the 5.5 generation (2026-10-01).

CLAUDE_MODEL is now claude-sonnet-5-5: it thinks by default (a reply can OPEN
with a thinking block, and thinking counts toward max_tokens), takes effort via
output_config, and declines with HTTP 200 + stop_reason "refusal". The old
``msg.content[0].text`` read crashed on a thinking block (AttributeError →
silent local fallback) and spoke nothing on a refusal.

These pin, with a fake client (no network), that every direct path —
_llm_quick, _claude_oneshot, the follow-up round, screen vision and the boot
ping — sends the per-model shaping (effort low, the max_tokens floor) and
reads the reply by block TYPE, and that a refusal lands in the path's existing
fallback instead of an empty string.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests._monolith_harness import (  # noqa: E402
    MonolithGlobalsTestCase, requires_monolith,
)


def _blk(btype, **kw):
    return types.SimpleNamespace(type=btype, **kw)


def _msg(*blocks, stop_reason="end_turn"):
    return types.SimpleNamespace(content=list(blocks), stop_reason=stop_reason,
                                 stop_details=None, usage=None)


_THINKING_FIRST = _msg(_blk("thinking", thinking="", signature="sig"),
                       _blk("text", text="Right away, sir."))
_REFUSED = _msg(stop_reason="refusal")


class _FakeClient:
    def __init__(self, reply):
        self.calls = []
        outer = self

        class _M:
            def create(self, **kw):
                outer.calls.append(kw)
                return reply
        self.messages = _M()


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        self.client = None

    def _use(self, reply, model="claude-sonnet-5-5"):
        self.client = _FakeClient(reply)
        for target, value in (
                ((self.bc, "_anthropic_client"), lambda *a, **k: self.client),
                ((self.bc, "AI_BACKEND"), "claude"),
                ((self.bc, "CLAUDE_MODEL"), model),
                ((self.bc, "SCREEN_VISION_MODEL"), model)):
            p = mock.patch.object(*target, value)
            p.start()
            self.addCleanup(p.stop)
        return self.client

    def sent(self):
        self.assertTrue(self.client.calls, "no Claude request was sent")
        return self.client.calls[-1]


class LlmQuickTests(_Base):
    def _cloud_route(self):
        import core.config as cfg
        for target, value in (
                ((cfg, "AMBIENT_LEARNING_FORCE_LOCAL"), False),
                ((cfg, "model_route"), lambda _fn: "cloud"),
                ((self.bc, "_llm_quick_goes_local"), lambda: False)):
            p = mock.patch.object(*target, value)
            p.start()
            self.addCleanup(p.stop)

    def test_thinking_first_reply_returns_the_text_and_request_is_shaped(self):
        self._cloud_route()
        self._use(_THINKING_FIRST)
        out = self.bc._llm_quick("sys", "user", max_tokens=60)
        self.assertEqual(out, "Right away, sir.")
        sent = self.sent()
        self.assertEqual(sent["extra_body"], {"output_config": {"effort": "low"}})
        self.assertEqual(sent["max_tokens"], 2048)
        self.assertNotIn("temperature", sent)

    def test_refusal_falls_back_to_local(self):
        self._cloud_route()
        self._use(_REFUSED)
        with mock.patch.object(self.bc, "_call_local_llm",
                               lambda *a, **k: "local answer"):
            self.assertEqual(self.bc._llm_quick("sys", "user"), "local answer")

    def test_older_model_request_is_unchanged(self):
        self._cloud_route()
        self._use(_msg(_blk("text", text="ok")), model="claude-haiku-4-5")
        self.bc._llm_quick("sys", "user", max_tokens=60)
        sent = self.sent()
        self.assertEqual(sent["max_tokens"], 60)
        self.assertNotIn("extra_body", sent)


class OneshotAndFollowupTests(_Base):
    def test_claude_oneshot_direct_path_reads_past_thinking(self):
        self._use(_THINKING_FIRST)
        with mock.patch.object(self.bc, "_claude_reachable", lambda: True), \
                mock.patch.object(self.bc, "_llm_client", None):
            out = self.bc._claude_oneshot("sys", [{"role": "user", "content": "x"}])
        self.assertEqual(out, "Right away, sir.")
        self.assertEqual(self.sent()["extra_body"]["output_config"]["effort"], "low")

    def test_claude_oneshot_refusal_is_none_so_the_honest_line_runs(self):
        self._use(_REFUSED)
        with mock.patch.object(self.bc, "_claude_reachable", lambda: True), \
                mock.patch.object(self.bc, "_llm_client", None):
            self.assertIsNone(
                self.bc._claude_oneshot("sys", [{"role": "user", "content": "x"}]))

    def test_oneshot_via_llm_client_names_the_voice_purpose(self):
        seen = {}

        class _Wrapped:
            def complete(self, **kw):
                seen.update(kw)
                return "ok"
        self._use(_THINKING_FIRST)
        with mock.patch.object(self.bc, "_claude_reachable", lambda: True), \
                mock.patch.object(self.bc, "_llm_client", _Wrapped()):
            self.bc._claude_oneshot("sys", [{"role": "user", "content": "x"}])
        self.assertEqual(seen["purpose"], "voice")


class VisionTests(_Base):
    def _cloud_vision(self):
        import core.config as cfg
        p = mock.patch.object(cfg, "model_route", lambda _fn: "cloud")
        p.start()
        self.addCleanup(p.stop)

    def test_vision_reads_past_thinking_with_vision_shaping(self):
        self._cloud_vision()
        self._use(_THINKING_FIRST)
        out = self.bc.ask_vision("what is on screen?", b"\x89PNGfake")
        self.assertEqual(out, "Right away, sir.")
        sent = self.sent()
        self.assertEqual(sent["model"], "claude-sonnet-5-5")
        self.assertEqual(sent["extra_body"], {"output_config": {"effort": "low"}})
        self.assertEqual(sent["max_tokens"], 2048)

    def test_vision_refusal_falls_back_to_the_local_vlm(self):
        self._cloud_vision()
        self._use(_REFUSED)
        with mock.patch.object(self.bc, "_call_local_vision",
                               lambda *a, **k: "a code editor"):
            out = self.bc.ask_vision("what is on screen?", b"\x89PNGfake")
        self.assertEqual(out, "[local-vision] a code editor")


class PreflightPingTests(_Base):
    def test_ping_keeps_one_token_and_sends_effort_low(self):
        self._use(_msg(_blk("text", text=".")))
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            ok, reason = self.bc._preflight_api_key(timeout_sec=2.0)
        self.assertTrue(ok, reason)
        sent = self.sent()
        self.assertEqual(sent["max_tokens"], 1)
        self.assertEqual(sent["extra_body"], {"output_config": {"effort": "low"}})


if __name__ == "__main__":
    unittest.main()
