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

    def test_boot_warm_installs_the_reflector_sink(self):
        # B078 (2026-10-01): the reflector's settled contradictions now reach
        # the prompt's own fact list through this sink.
        fake = mock.Mock()
        fake.list_facts.return_value = []
        self._run_warm(fake)
        fake.set_reflector_sink.assert_called_once_with(
            self.bc._ltm_reflector_sink)

    def test_adapter_leaves_room_for_a_whole_merged_fact(self):
        # A "MERGE: <fact>" reply was cut mid-word at the old 60-token cap.
        fake = mock.Mock()
        fake.list_facts.return_value = []
        self._run_warm(fake)
        adapter = fake.set_reflector_llm.call_args[0][0]
        with mock.patch.object(self.bc, "_llm_quick", return_value="") as mq:
            adapter("prompt", [])
        self.assertGreaterEqual(mq.call_args.kwargs.get("max_tokens"), 100)

    def test_failed_warm_up_does_not_wire(self):
        fake = mock.Mock()
        fake.ensure_loaded.side_effect = RuntimeError("store locked")
        self._run_warm(fake)                      # must not raise
        fake.set_reflector_llm.assert_not_called()

    def test_boot_warm_loads_the_embedder(self):
        # v2.0.139 (2026-09-29, live): ensure_loaded() does not load the
        # embedder, so it loaded inside the owner's FIRST turn after every
        # start. The boot warm-up now loads it off the voice thread.
        fake = mock.Mock()
        fake.list_facts.return_value = []
        self._run_warm(fake)
        fake._try_import_embedder.assert_called_once_with()

    def test_embedder_failure_is_swallowed(self):
        fake = mock.Mock()
        fake.list_facts.return_value = []
        fake._try_import_embedder.side_effect = RuntimeError("no torch")
        self._run_warm(fake)                      # must not raise
        fake.set_reflector_llm.assert_called_once()


@requires_monolith
class ReflectorSinkTests(unittest.TestCase):
    """B078 (2026-10-01): the LTM reflector settled contradictions in the
    semantic store only, while bobert_memory.json -- whose facts go into
    every system prompt -- kept both sides. _ltm_reflector_sink applies each
    settled decision there. Generic fixtures; the real store is never read."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()

    def _sink(self, store, changes):
        import copy
        bc = self.bc
        saved = []

        def _load():
            return copy.deepcopy(store)

        with mock.patch.object(bc, "load_memory", _load), \
             mock.patch.object(bc, "save_memory", saved.append), \
             mock.patch.object(bc, "_request_prompt_rebuild") as rebuild, \
             mock.patch("builtins.print") as p:
            bc._ltm_reflector_sink(changes)
        logged = " ".join(str(c.args[0]) for c in p.call_args_list if c.args)
        return saved, rebuild, logged

    def test_the_settled_loser_leaves_the_prompt_facts(self):
        store = {"facts": ["User lives in Boulder", "User likes tea",
                           "User lives in Denver"], "projects": []}
        saved, rebuild, logged = self._sink(
            store, [("contradiction", ("user lives in boulder",),
                     "User lives in Denver")])
        self.assertEqual(saved[-1]["facts"],
                         ["User likes tea", "User lives in Denver"])
        rebuild.assert_called_once_with()
        self.assertNotIn("Boulder", logged)        # counts only, never text

    def test_a_survivor_store_a_does_not_hold_changes_nothing(self):
        # 2026-10-01 (adversarial review): the sink wrote the LTM survivor
        # into store A in the condemned fact's place -- and LTM can still
        # hold a fact the owner removed from bobert_memory.json (the guest-
        # learned facts awaiting quarantine). One wrong A/B verdict swapped
        # the true fact for the removed one, past merge_memory's owner gate.
        store = {"facts": ["User lives in Boulder", "User likes tea"],
                 "projects": []}
        saved, rebuild, _l = self._sink(
            store, [("contradiction", ("User lives in Boulder",),
                     "User lives in Denver"),
                    ("contradiction", ("User likes tea",), None)])
        self.assertEqual(saved, [])
        rebuild.assert_not_called()

    def test_a_merge_replaces_in_place_then_drops_the_other(self):
        store = {"facts": ["User has a dog", "User likes tea",
                           "User's dog is named Rex"], "projects": []}
        saved, _r, _l = self._sink(store, [
            ("merge", ("User's dog is named Rex", "User has a dog"),
             "User has a dog named Rex")])
        self.assertEqual(saved[-1]["facts"],
                         ["User likes tea", "User has a dog named Rex"])

    def test_a_merge_with_a_text_store_a_does_not_hold_changes_nothing(self):
        store = {"facts": ["User has a dog", "User likes tea"],
                 "projects": []}
        saved, rebuild, _l = self._sink(store, [
            ("merge", ("User has a dog", "User's dog is named Rex"),
             "User has a dog named Rex")])
        self.assertEqual(saved, [])
        rebuild.assert_not_called()

    def test_a_merge_chain_applies_in_order(self):
        store = {"facts": ["a dog", "a dog named Rex", "Rex is brown"],
                 "projects": []}
        saved, _r, _l = self._sink(store, [
            ("merge", ("a dog", "a dog named Rex"), "a dog named Rex"),
            ("merge", ("a dog named Rex", "Rex is brown"),
             "a brown dog named Rex")])
        self.assertEqual(saved[-1]["facts"], ["a brown dog named Rex"])

    def test_a_secret_shaped_merge_is_never_written(self):
        store = {"facts": ["User has a router", "User's router is new"],
                 "projects": []}
        saved, _r, _l = self._sink(store, [
            ("merge", ("User has a router", "User's router is new"),
             "User's router password is hunter2")])
        self.assertEqual(saved, [])

    def test_texts_not_in_the_prompt_memory_change_nothing(self):
        store = {"facts": ["User likes tea"], "projects": []}
        saved, rebuild, _l = self._sink(
            store, [("contradiction", ("User likes coffee",),
                     "User likes tea")])
        self.assertEqual(saved, [])
        rebuild.assert_not_called()

    def test_the_real_reflector_never_writes_a_removed_fact_back(self):
        # End to end through the REAL reflector and the REAL sink: LTM still
        # holds "Denver" (removed from store A by the owner), the model picks
        # it, and store A must keep "Boulder" and never gain "Denver".
        import copy
        from core import long_term_memory as ltm
        bc = self.bc
        store = {"facts": ["User lives in Boulder", "User likes tea"],
                 "projects": []}
        saved = []
        facts = {
            fid: {"id": fid, "text": text, "source": "merge_memory",
                  "tags": [], "created_at": ca, "updated_at": ca}
            for fid, text, ca in (("a", "User lives in Boulder", 1.0),
                                  ("b", "User lives in Denver", 2.0))}
        with mock.patch.object(ltm, "_facts", facts),              mock.patch.object(ltm, "ensure_loaded"),              mock.patch.object(ltm, "_embed", lambda texts: [1] * len(texts)),              mock.patch.object(ltm, "_cosine_sim", lambda x, y: 0.7),              mock.patch.object(ltm, "_chroma_delete"),              mock.patch.object(ltm, "_chroma_upsert"),              mock.patch.object(ltm, "_save_facts_locked"),              mock.patch.object(ltm, "_rebuild_bm25_locked"),              mock.patch.object(ltm, "_reflector_sink", bc._ltm_reflector_sink),              mock.patch.object(bc, "load_memory",
                               lambda: copy.deepcopy(store)),              mock.patch.object(bc, "save_memory", saved.append),              mock.patch.object(bc, "_request_prompt_rebuild"),              mock.patch("builtins.print"):
            # Presented newest first: "Denver" is A. "A" keeps Denver.
            summary = ltm.reflect_and_consolidate(llm_call=lambda p, c: "A")
        self.assertEqual(summary["contradictions_resolved"], 1)
        self.assertEqual(sorted(facts), ["b"])          # LTM settled it
        self.assertEqual(saved, [])                     # store A untouched


if __name__ == "__main__":
    unittest.main()
