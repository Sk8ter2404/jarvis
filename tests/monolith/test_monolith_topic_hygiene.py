"""Monolith wiring for the learned-topic hygiene gates (core/topic_hygiene.py).

THE LIVE FAILURE (2026-09-29): "what am I working on lately" -> JARVIS named a
"mystery" and a "project" that were Whisper mis-hearings of TV audio. Each
had been learned from ONE overheard line (learn_from_turn -> merge_memory),
stored verbatim, and rendered into every system prompt as "Projects they've
mentioned" / "Recent topics you've discussed", which the persona then told the
model to volunteer.

Pinned here:
  * merge_memory's write gate: ambient speech never stores a topic/project; a
    low-confidence transcript is refused; one owner turn only HOLDS an item;
    two separate owner turns surface it; a mostly-non-word label never
    surfaces unless the owner uses the words himself.
  * the learners pass the provenance the gate needs (the answered turn is
    owner-directed with its Whisper metadata; the ambient path is not).
  * read time: the prompt frames learned items as unverified hints, the
    proactive-comment prompt no longer orders the model to raise them, LTM
    recall marks learned projects, and the local never-guess guard names
    project_status.

GENERIC fixtures only; load/save are patched to an in-memory dict, so no real
memory file is read or written.

    python -m unittest tests.monolith.test_monolith_topic_hygiene
"""
from __future__ import annotations

import copy
import inspect
import unittest
from unittest import mock

from tests._monolith_harness import MonolithGlobalsTestCase, requires_monolith

_CLEAR = {"no_speech_prob": 0.05, "avg_logprob": -0.2}


def _owner(text, conf=_CLEAR):
    return {"owner_directed": True, "turn_text": text, "conf": conf,
            "source": "test turn"}


class _InlineThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, **_):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


@requires_monolith
class _StoreBase(MonolithGlobalsTestCase):
    """merge_memory against an in-memory store; owner vocabulary empty unless
    a test sets self.vocab."""

    def setUp(self):
        self.store = {"facts": [], "projects": [], "topics": [], "sessions": []}
        self.vocab = frozenset()

        def _load():
            return copy.deepcopy(self.store)

        def _save(m):
            self.store = m

        for name, value in (("load_memory", _load), ("save_memory", _save),
                            ("_owner_vocab", lambda: self.vocab),
                            ("_ltm_learn_facts", mock.MagicMock())):
            p = mock.patch.object(self.bc, name, value)
            p.start()
            self.addCleanup(p.stop)

    def merge(self, text, *, topic="", projects=None, facts=None,
              conf=_CLEAR, owner=True):
        prov = _owner(text, conf) if owner else {
            "owner_directed": False, "turn_text": text, "source": "ambient"}
        return self.bc.merge_memory(new_facts=facts, new_projects=projects,
                                    new_topic=topic, provenance=prov)

    def topics(self):
        return [t["topic"] for t in self.store["topics"]]


class MergeMemoryGateTests(_StoreBase):
    def test_ambient_speech_never_stores_a_topic_or_project(self):
        added_f, added_p = self.merge(
            "the garden shed is on tonight", topic="garden shed",
            projects=["Building a garden shed"], facts=["User likes tea"],
            owner=False)
        self.assertEqual(added_p, [])
        self.assertEqual(self.store["projects"], [])
        self.assertEqual(self.store["topics"], [])
        self.assertNotIn("topic_candidates", self.store)
        # Facts keep their existing ambient path.
        self.assertEqual(added_f, ["User likes tea"])

    def test_one_owner_turn_only_holds_the_item(self):
        _f, added_p = self.merge("I'm building a garden shed",
                                 topic="garden shed",
                                 projects=["Building a garden shed"])
        self.assertEqual(added_p, [])
        self.assertEqual(self.store["projects"], [])
        self.assertEqual(self.store["topics"], [])
        kinds = sorted(c["kind"] for c in self.store["topic_candidates"])
        self.assertEqual(kinds, ["project", "topic"])
        self.bc._ltm_learn_facts.assert_called_with([], [])

    def test_second_separate_owner_turn_surfaces_it(self):
        self.merge("I'm building a garden shed", topic="garden shed",
                   projects=["Building a garden shed"])
        _f, added_p = self.merge("what wood should the garden shed use",
                                 topic="garden shed build",
                                 projects=["Garden shed build"])
        self.assertEqual(added_p, ["Garden shed build"])
        self.assertEqual(self.store["projects"], ["Garden shed build"])
        self.assertEqual(self.topics(), ["garden shed build"])
        # A later paraphrase of a listed project is not added beside it.
        self.merge("the garden shed roof is done", topic="garden shed",
                   projects=["Building a garden shed"])
        self.assertEqual(self.store["projects"], ["Garden shed build"])

    def test_the_same_utterance_twice_stays_held(self):
        for _ in range(2):
            self.merge("tell me about the garden shed", topic="garden shed")
        self.assertEqual(self.store["topics"], [])

    def test_low_confidence_transcript_teaches_no_topic(self):
        shaky = {"no_speech_prob": 0.2, "avg_logprob": -1.3}
        for text in ("the mystery deepens tonight", "a mystery for the ages"):
            added_f, _p = self.merge(text, topic="the mystery",
                                     projects=["Mystery research"],
                                     facts=["User likes tea"], conf=shaky)
        self.assertEqual(self.store["topics"], [])
        self.assertEqual(self.store["projects"], [])
        self.assertNotIn("topic_candidates", self.store)
        self.assertIn("User likes tea", self.store["facts"])

    def test_non_word_label_never_surfaces(self):
        self.merge("something about the zorblat case", topic="Zorblat mystery")
        self.merge("more on that zorblat thing", topic="Zorblat mystery")
        self.assertEqual(self.store["topics"], [])

    def test_owner_vocabulary_lets_his_proper_noun_through(self):
        self.vocab = frozenset({"zorblat"})
        self.merge("how is the zorblat build going", topic="Zorblat build")
        self.merge("order more zorblat parts", topic="Zorblat build")
        self.assertEqual(self.topics(), ["Zorblat build"])

    def test_stored_facts_make_a_client_name_known(self):
        self.store["facts"] = ["User does contract work for Zorblat Ltd",
                               "Zorblat is the user's biggest client"]
        self.merge("how is the zorblat rollout", topic="Zorblat rollout")
        self.merge("zorblat called about it", topic="Zorblat rollout")
        self.assertEqual(self.topics(), ["Zorblat rollout"])

    def test_legacy_caller_without_provenance_is_unchanged(self):
        _f, added_p = self.bc.merge_memory(new_projects=["Building a treehouse"],
                                           new_topic="weekend plans")
        self.assertEqual(added_p, ["Building a treehouse"])
        self.assertEqual(self.topics(), ["weekend plans"])

    def test_a_bare_string_is_one_item_not_characters(self):
        self.bc.merge_memory(new_projects="Building a treehouse",
                             new_facts="User likes tea")
        self.assertEqual(self.store["projects"], ["Building a treehouse"])
        self.assertEqual(self.store["facts"], ["User likes tea"])


@requires_monolith
class LearnerProvenanceTests(MonolithGlobalsTestCase):
    _PAYLOAD = ('{"new_facts": [], "new_projects": ["Building a garden shed"],'
                ' "topic": "garden shed"}')

    def _learn(self, **kw):
        with mock.patch.object(self.bc, "LEARN_EVERY_TURN", True), \
             mock.patch.object(self.bc.threading, "Thread", _InlineThread), \
             mock.patch.object(self.bc, "load_memory",
                               return_value={"facts": [], "projects": []}), \
             mock.patch.object(self.bc, "_llm_quick",
                               return_value=self._PAYLOAD), \
             mock.patch.object(self.bc, "merge_memory",
                               return_value=([], [])) as mmerge:
            self.bc.learn_from_turn("I'm building a garden shed", "Noted.",
                                    {}, **kw)
        return mmerge.call_args.kwargs["provenance"]

    def test_answered_turn_is_owner_directed_with_its_whisper_metadata(self):
        prov = self._learn(conf=_CLEAR)
        self.assertTrue(prov["owner_directed"])
        self.assertEqual(prov["turn_text"], "I'm building a garden shed")
        self.assertEqual(prov["conf"], _CLEAR)

    def test_ambient_learning_is_not_owner_directed(self):
        self.assertFalse(self._learn(owner_directed=False)["owner_directed"])

    def _gated(self, owner_voice):
        bc = self.bc
        vid = ((True, "owner", 0.9) if owner_voice
               else (False, "unavailable", 0.0))
        with mock.patch.object(bc, "AMBIENT_LISTEN_ENABLED", True), \
             mock.patch.object(bc, "_ambient_media_is_playing",
                               return_value=False), \
             mock.patch.object(bc, "_ambient_owner_voice", return_value=vid), \
             mock.patch.object(bc, "_call_local_llm", return_value="PERSON"), \
             mock.patch.object(bc, "learn_from_turn") as lft:
            bc._ambient_learn_from_gated("we should repaint the kitchen soon",
                                         {}, conf=_CLEAR)
        lft.assert_called_once()
        return lft.call_args.kwargs

    def test_both_ambient_ingest_paths_mark_speech_as_overheard(self):
        for owner_voice in (True, False):
            with self.subTest(owner_voice=owner_voice):
                kw = self._gated(owner_voice)
                self.assertIs(kw["owner_directed"], False)
                self.assertEqual(kw["conf"], _CLEAR)

    def test_main_loop_passes_the_turn_confidence(self):
        src = inspect.getsource(self.bc.main)
        # 2026-09-30: the same call now also passes typed / wake / the raw
        # capture for owner-only learning (test_monolith_learn_gate).
        self.assertIn("learn_from_turn(text, reply, memory, conf=conf,", src)


@requires_monolith
class ReadTimeGroundingTests(MonolithGlobalsTestCase):
    def _prompt(self, mem):
        with mock.patch.object(self.bc, "_load_chappie_standing_rules",
                               return_value=""), \
             mock.patch.object(self.bc._mcu_phrases, "render_phrasebook_block",
                               return_value="PB"):
            return self.bc.build_system_prompt(mem)

    def test_learned_projects_and_topics_are_framed_as_unverified_hints(self):
        mem = self.bc._empty_memory()
        mem["projects"] = ["Garden shed"]
        mem["topics"] = [{"date": "2026-09-01", "location": "desk",
                          "topic": "carpentry"}]
        prompt = self._prompt(mem)
        self.assertNotIn("Projects they've mentioned:", prompt)
        self.assertNotIn("Recent topics you've discussed:", prompt)
        proj = prompt.index("Garden shed")
        head = prompt.rindex("\n\n", 0, proj)
        self.assertIn("unverified", prompt[head:proj])
        self.assertIn("never volunteer", prompt[head:proj])
        self.assertIn("project_status", prompt[head:proj])
        top = prompt.index("carpentry")
        head = prompt.rindex("\n\n", 0, top)
        self.assertIn("never volunteer", prompt[head:top])
        self.assertIn("mis-heard", prompt[head:top])

    def test_proactive_comment_no_longer_raises_learned_topics(self):
        seen = {}

        def fake_quick(system, user, max_tokens=120):
            seen["system"] = system
            return "A check-in, sir."
        with mock.patch.object(self.bc, "_llm_quick", side_effect=fake_quick):
            self.bc.generate_proactive_comment()
        tail = seen["system"][len(self.bc._system_prompt):]
        self.assertNotIn("something you remember they're working on", tail)
        self.assertNotIn("question about a recent topic", tail)
        self.assertIn("Never raise an auto-learned topic or project", tail)

    def test_ltm_recall_marks_learned_projects(self):
        fake = mock.MagicMock()
        fake.retrieve_facts.return_value = [
            {"text": "Zorblat research", "tags": ["project"]},
            {"text": "User likes tea", "tags": ["learned"]}]
        with mock.patch.object(self.bc, "_ltm_enabled", return_value=True), \
             mock.patch.object(self.bc, "_ltm_module", return_value=fake), \
             mock.patch.object(self.bc, "_LTM_RETRIEVE_BUDGET_S", 5.0):
            out = self.bc._ltm_context("what am I working on")
        self.assertIn("- (auto-learned project mention, unverified) "
                      "Zorblat research", out)
        self.assertIn("- User likes tea", out)

    def test_local_never_guess_guard_names_project_status(self):
        self.assertIn("[ACTION: project_status]",
                      self.bc._LOCAL_NEVER_GUESS_GUARD)


if __name__ == "__main__":
    unittest.main()
