"""A preferred local brain that finishes downloading AFTER the first resolve is
adopted at the next turn boundary (todo P2-7). Before this, the first installed
tag was cached for the whole process."""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests._monolith_harness import (                   # noqa: E402
    MonolithGlobalsTestCase, load_monolith, requires_monolith)
from tests.test_audit_2026_07_21 import (               # noqa: E402
    GEMMA_12B, GEMMA_26B, QWEN_14B, _fake_tags)


@requires_monolith
class RefreshLocalLlmModelTests(MonolithGlobalsTestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def setUp(self):
        bc = self.bc
        self._saved = (list(bc._RESOLVED_LOCAL_LLM_MODEL),
                       list(bc._LOCAL_LLM_AUTO_PICK),
                       list(bc._local_llm_recheck_at))
        bc._RESOLVED_LOCAL_LLM_MODEL[0] = None
        bc._LOCAL_LLM_AUTO_PICK[0] = None
        bc._local_llm_recheck_at[0] = 0.0
        self._env = os.environ.pop("JARVIS_LOCAL_LLM_MODEL", None)
        import core.config as cfg
        p = [mock.patch.object(bc, "LOCAL_LLM_MODEL", cfg._SHIPPED_LOCAL_LLM_MODEL),
             mock.patch.object(bc, "_log_gpu_state"),
             mock.patch.object(bc, "_sync_vision_to_chat", create=True)]
        for x in p:
            x.start()
            self.addCleanup(x.stop)

    def tearDown(self):
        bc = self.bc
        bc._RESOLVED_LOCAL_LLM_MODEL[:] = self._saved[0]
        bc._LOCAL_LLM_AUTO_PICK[:] = self._saved[1]
        bc._local_llm_recheck_at[:] = self._saved[2]
        if self._env is not None:
            os.environ["JARVIS_LOCAL_LLM_MODEL"] = self._env

    def _resolve(self, *tags):
        with mock.patch.object(self.bc, "requests", _fake_tags(*tags)):
            return self.bc._get_local_llm_model()

    def _refresh(self, *tags, now=None):
        if now is None:        # a re-check window after the first resolve
            now = self.bc._local_llm_recheck_at[0] + self.bc._LOCAL_LLM_RECHECK_S + 1
        with mock.patch.object(self.bc, "requests", _fake_tags(*tags)), \
                mock.patch("skills.model_picker._sync_vision_to_chat") as sync:
            out = self.bc._refresh_local_llm_model(now=now)
        return out, sync

    def test_finished_pull_of_preferred_model_is_adopted(self):
        self.assertEqual(self._resolve(GEMMA_12B), GEMMA_12B)
        out, sync = self._refresh(GEMMA_12B, GEMMA_26B)
        self.assertEqual(out, GEMMA_26B)
        self.assertEqual(self.bc._get_local_llm_model(), GEMMA_26B)
        sync.assert_called_once()          # vision lockstep runs, live only
        self.assertEqual(sync.call_args.args[:2], (GEMMA_12B, GEMMA_26B))
        self.assertFalse(sync.call_args.kwargs.get("persist", True))

    def test_off_list_first_available_is_upgraded_to_a_chain_model(self):
        self.assertEqual(self._resolve("mystery:7b"), "mystery:7b")
        out, _ = self._refresh("mystery:7b", QWEN_14B)
        self.assertEqual(out, QWEN_14B)

    def test_worse_or_same_model_is_not_adopted(self):
        self._resolve(GEMMA_12B, GEMMA_26B)        # already the best (26B)
        out, _ = self._refresh(GEMMA_12B, GEMMA_26B)
        self.assertIsNone(out)
        self.bc._RESOLVED_LOCAL_LLM_MODEL[0] = None
        self._resolve(GEMMA_12B)
        out, _ = self._refresh(GEMMA_12B, QWEN_14B)   # QWEN ranks below GEMMA_12B
        self.assertIsNone(out)

    def test_top_preference_never_probes(self):
        self._resolve(GEMMA_26B)
        with mock.patch.object(self.bc, "requests") as req:
            self.assertIsNone(self.bc._refresh_local_llm_model(now=1e12))
        req.get.assert_not_called()

    def test_throttled_to_one_probe_per_window(self):
        self._resolve(GEMMA_12B)
        t0 = self.bc._local_llm_recheck_at[0]
        with mock.patch.object(self.bc, "requests") as req:
            self.assertIsNone(self.bc._refresh_local_llm_model(now=t0 + 5))
        req.get.assert_not_called()
        out, _ = self._refresh(GEMMA_12B, GEMMA_26B,
                               now=t0 + self.bc._LOCAL_LLM_RECHECK_S + 1)
        self.assertEqual(out, GEMMA_26B)

    def test_pinned_choices_are_never_overridden(self):
        # a voice/tray switch, game-mode swap or failover writes the cache
        self._resolve(GEMMA_12B)
        self.bc._RESOLVED_LOCAL_LLM_MODEL[0] = QWEN_14B
        out, _ = self._refresh(GEMMA_12B, GEMMA_26B, QWEN_14B)
        self.assertIsNone(out)
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], QWEN_14B)
        # the env override pins too
        os.environ["JARVIS_LOCAL_LLM_MODEL"] = "my:custom"
        try:
            self.bc._RESOLVED_LOCAL_LLM_MODEL[0] = None
            self.assertEqual(self.bc._get_local_llm_model(), "my:custom")
            out, _ = self._refresh(GEMMA_12B, GEMMA_26B)
            self.assertIsNone(out)
        finally:
            os.environ.pop("JARVIS_LOCAL_LLM_MODEL", None)

    def test_saved_pick_that_finishes_downloading_wins(self):
        with mock.patch.object(self.bc, "LOCAL_LLM_MODEL", QWEN_14B):
            self.assertEqual(self._resolve(GEMMA_26B), GEMMA_26B)  # not installed yet
            out, _ = self._refresh(GEMMA_26B, QWEN_14B)
            self.assertEqual(out, QWEN_14B)
            # a saved pick is final: nothing is auto-upgraded afterwards
            self.assertIsNone(self.bc._LOCAL_LLM_AUTO_PICK[0])

    def test_unreachable_ollama_changes_nothing(self):
        self._resolve(GEMMA_12B)
        out, _ = self._refresh()
        self.assertIsNone(out)
        self.assertEqual(self.bc._RESOLVED_LOCAL_LLM_MODEL[0], GEMMA_12B)

    def test_called_at_the_turn_boundary_only(self):
        import inspect
        src = inspect.getsource(self.bc._run_llm_dispatch)
        self.assertIn("_refresh_local_llm_model()", src)
        self.assertLess(src.index("_refresh_local_llm_model()"),
                        src.index("_run_llm_dispatch_body(text)"))
        # the resolver itself never swaps a cached model
        self.assertNotIn("_refresh_local_llm_model(",
                         inspect.getsource(self.bc._get_local_llm_model))


if __name__ == "__main__":
    unittest.main()
