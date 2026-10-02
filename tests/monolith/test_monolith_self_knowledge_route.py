"""SELF-KNOWLEDGE on the cloud fallback (review 2026-10-02).

A LOCAL-routed turn carries the live SELF-KNOWLEDGE render in its user
message: "you are answering on your LOCAL brain, <tag>". When the local call
fails, _local_then_cloud_or_honest sends the SAME messages to the cloud
model, which would then describe itself as the local model. The fallback
re-renders the section for the cloud route; the local path is untouched.
"""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith
from core import prompts


@requires_monolith
class CloudFallbackSelfKnowledgeTests(MonolithGlobalsTestCase):
    def _msgs(self):
        render = prompts.render_self_knowledge_section(route="local")
        self.assertIn("you are answering on your LOCAL brain", render)
        return [{"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "Indeed, sir."},
                {"role": "user",
                 "content": "<ctx>" + render + "</ctx>how smart are you?"}]

    def test_a_failed_over_turn_tells_the_cloud_it_is_the_cloud(self):
        bc = self.bc
        msgs = self._msgs()
        before = [dict(m) for m in msgs]
        seen = {}

        def _cloud(system, messages, max_tokens=500):
            seen["msgs"] = messages
            return "I am JARVIS, sir."
        with mock.patch.object(bc, "_call_local_llm", return_value=None), \
                mock.patch.object(bc, "_claude_oneshot", side_effect=_cloud), \
                mock.patch.object(bc, "_sac_blocked_local_recently",
                                  return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            out = bc._local_then_cloud_or_honest("SYS", msgs)
        self.assertEqual(out, "I am JARVIS, sir.")
        sent = seen["msgs"][-1]["content"]
        self.assertNotIn("you are answering on your LOCAL brain", sent)
        self.assertIn("answered by the cloud model", sent.lower())
        self.assertTrue(sent.startswith("<ctx>"))
        self.assertTrue(sent.endswith("</ctx>how smart are you?"))
        self.assertEqual(seen["msgs"][:2], msgs[:2])
        self.assertEqual(msgs, before, "the caller's messages were mutated")

    def test_the_local_path_sends_the_local_render(self):
        bc = self.bc
        msgs = self._msgs()
        seen = {}

        def _local(system, messages, max_tokens=500):
            seen["msgs"] = messages
            return "Local answer, sir."
        with mock.patch.object(bc, "_call_local_llm", side_effect=_local), \
                mock.patch.object(bc, "_claude_oneshot") as cloud:
            out = bc._local_then_cloud_or_honest("SYS", msgs)
        self.assertEqual(out, "Local answer, sir.")
        cloud.assert_not_called()
        self.assertIs(seen["msgs"], msgs)


if __name__ == "__main__":
    unittest.main()
