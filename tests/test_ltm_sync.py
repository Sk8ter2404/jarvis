"""Tests for the merge_memory → semantic-LTM sync hook (_ltm_learn_facts).

2026-07-15: merge_memory (structured store A, bobert_memory.json) and the
semantic store (chroma + BM25) were never wired together, so facts learned after
the 2026-05-28 migration never became fuzzy-searchable. _ltm_learn_facts closes
that gap — fire-and-forget, gated, exception-isolated. These pin: gated-off is a
no-op, learned facts/projects reach long_term_memory.add_fact, and bad/empty
input is safe.
"""
from __future__ import annotations

import threading
import unittest
from unittest import mock

from tests._monolith_harness import load_monolith, requires_monolith


def _join_learn_worker(timeout: float = 3.0) -> None:
    for t in threading.enumerate():
        if t.name == "ltm-learn":
            t.join(timeout=timeout)


@requires_monolith
class LtmSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def test_gated_off_is_noop(self):
        bc = self.bc
        with mock.patch.object(bc, "_ltm_enabled", return_value=False), \
             mock.patch.object(bc, "_ltm_module") as m:
            bc._ltm_learn_facts(["a fact"], ["a project"])
        m.assert_not_called()

    def test_learned_facts_and_projects_reach_add_fact(self):
        bc = self.bc
        fake = mock.Mock()
        with mock.patch.object(bc, "_ltm_enabled", return_value=True), \
             mock.patch.object(bc, "_ltm_module", return_value=fake):
            bc._ltm_learn_facts(["User likes tea"], ["Building a robot"])
            _join_learn_worker()
        texts = [c.args[0] for c in fake.add_fact.call_args_list]
        self.assertIn("User likes tea", texts)
        self.assertIn("Building a robot", texts)
        # the project is tagged so recall can distinguish it
        tags = [c.kwargs.get("tags") for c in fake.add_fact.call_args_list]
        self.assertTrue(any(t and "project" in t for t in tags))

    def test_empty_input_never_touches_the_store(self):
        bc = self.bc
        with mock.patch.object(bc, "_ltm_enabled", return_value=True), \
             mock.patch.object(bc, "_ltm_module") as m:
            bc._ltm_learn_facts([], [])
            bc._ltm_learn_facts(None, None)
            bc._ltm_learn_facts(["  ", 42, None], None)   # nothing valid
        m.assert_not_called()

    def test_add_fact_exception_is_isolated(self):
        bc = self.bc
        boom = mock.Mock()
        boom.add_fact.side_effect = RuntimeError("chroma boom")
        with mock.patch.object(bc, "_ltm_enabled", return_value=True), \
             mock.patch.object(bc, "_ltm_module", return_value=boom):
            bc._ltm_learn_facts(["a durable fact"])   # must not raise
            _join_learn_worker()
        boom.add_fact.assert_called()   # it tried, and swallowed the error


if __name__ == "__main__":
    unittest.main()
