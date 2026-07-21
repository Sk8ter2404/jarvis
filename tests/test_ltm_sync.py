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


class _InlineThread:
    """Drop-in for ``threading.Thread(target=...)`` that runs the target
    synchronously on ``.start()`` so worker bodies execute deterministically."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, **_):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, *a, **k):
        return None


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


@requires_monolith
class ReflectorWiringTests(unittest.TestCase):
    """_ltm_boot_warm must inject the local-LLM adjudicator into the LTM
    reflector via ltm.set_reflector_llm — the contradiction pass was DEAD in
    production because record_turn's trigger had no llm_call to pass
    (2026-07-21 audit #39). This is the invariant that keeps the injection
    from silently un-wiring again: the same 'built but zero production
    callers' failure mode the 2026-07-06 audit found for LTM as a whole."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _run_warm(self, fake_ltm):
        bc = self.bc
        with mock.patch.object(bc, "_ltm_enabled", return_value=True), \
             mock.patch.object(bc, "_ltm_module", return_value=fake_ltm), \
             mock.patch.object(bc.threading, "Thread", _InlineThread):
            bc._ltm_boot_warm()

    def test_boot_warm_installs_reflector_adjudicator(self):
        fake = mock.Mock()
        fake.list_facts.return_value = []
        self._run_warm(fake)
        fake.ensure_loaded.assert_called_once()
        fake.set_reflector_llm.assert_called_once()
        (adapter,) = fake.set_reflector_llm.call_args[0]
        self.assertTrue(callable(adapter))

    def test_adapter_feeds_llm_quick_prompt_and_both_fact_texts(self):
        fake = mock.Mock()
        fake.list_facts.return_value = []
        self._run_warm(fake)
        adapter = fake.set_reflector_llm.call_args[0][0]
        with mock.patch.object(self.bc, "_llm_quick",
                               return_value="A") as mq:
            out = adapter("prompt", [{"role": "fact_a", "text": "x"},
                                     {"role": "fact_b", "text": "y"}])
        self.assertEqual(out, "A")
        _, kwargs = mq.call_args
        self.assertEqual(kwargs.get("system"), "prompt")
        self.assertIn("fact_a: x", kwargs.get("user", ""))
        self.assertIn("fact_b: y", kwargs.get("user", ""))

    def test_adapter_tolerates_empty_context(self):
        fake = mock.Mock()
        fake.list_facts.return_value = []
        self._run_warm(fake)
        adapter = fake.set_reflector_llm.call_args[0][0]
        with mock.patch.object(self.bc, "_llm_quick", return_value=""):
            self.assertEqual(adapter("prompt", None), "")

    def test_failed_warm_up_does_not_wire(self):
        fake = mock.Mock()
        fake.ensure_loaded.side_effect = RuntimeError("store locked")
        self._run_warm(fake)                      # must not raise
        fake.set_reflector_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
