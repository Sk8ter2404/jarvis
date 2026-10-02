"""SELF-KNOWLEDGE: honest answers to 'how smart are you' (2026-10-01).

Live 2026-10-01 20:56-20:58 the owner asked three times how smart JARVIS is
next to Claude Opus 5.5 and other JARVIS-like assistants, and got a deflecting
joke every time. Nothing in the prompt told the local brain what it runs on or
what it is good and bad at, and the router made it worse: "how smart are you"
loaded the three SMART HOME sections (light-control grammar) on the bare header
word "smart", and "how do you compare to other AIs" loaded nothing at all.

These pin:
  * the router loads SELF-KNOWLEDGE for the comparison / capability phrasings,
    and no longer hands "how smart are you" the smart-home grammar;
  * the section the LOCAL route ships is RENDERED per turn from runtime values
    (the live local model tag, the cloud models, the latency constant), never
    a tag frozen into the prompt text;
  * the instructions: honest, concrete, in character, offer the cloud for hard
    questions, never claim to be Opus.

Stdlib unittest, CI-safe: no Ollama, no LLM, no monolith import (the monolith
is faked through sys.modules, exactly where the renderer looks for it).

    python tools/run_tests.py self_knowledge
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from core import config as cfg                    # noqa: E402
from core import prompt_router as pr              # noqa: E402
from core import prompts                          # noqa: E402

FULL = prompts.PC_CONTROL_PROMPT
SECTION = "SELF-KNOWLEDGE"

# Phrasings the section must load for. The first three are the live turns'
# shape; the rest cover each keyword family the item names.
_SELF_KNOWLEDGE_TURNS = (
    "how smart are you",
    "how smart are you compared to Claude Opus 5.5",
    "seriously, how do you stack up against opus",
    "what model are you running on",
    "how do you compare to other AIs",
    "how do you compare to other JARVIS assistants",
    "are you better than GPT",
    "is there a better model than you",
    "how good are you really",
    "how smart are you compared to claude",
)

# Fake values: nothing here may be a real tag, so a hit can only have come
# from the runtime value the test planted.
_FAKE_LOCAL = "unit-test-brain:3b-q4"
_FAKE_CACHED = "unit-test-cached:9b"
_FAKE_CLOUD = "unit-test-cloud-voice"


_MISSING = object()


class _Monolith:
    """Context: sys.modules["bobert_companion"] is ``module`` (None = absent)
    and JARVIS_LOCAL_LLM_MODEL is unset. Touches ONLY that one sys.modules key
    — a wholesale patch.dict(sys.modules) would also drop every module first
    imported inside the block, leaving duplicates for later tests."""

    def __init__(self, module=None):
        self._module = module

    def __enter__(self):
        self._saved = sys.modules.get("bobert_companion", _MISSING)
        if self._module is None:
            sys.modules.pop("bobert_companion", None)
        else:
            sys.modules["bobert_companion"] = self._module
        self._env = mock.patch.dict(os.environ)
        self._env.start()
        os.environ.pop("JARVIS_LOCAL_LLM_MODEL", None)
        return self

    def __exit__(self, *exc):
        self._env.stop()
        if self._saved is _MISSING:
            sys.modules.pop("bobert_companion", None)
        else:
            sys.modules["bobert_companion"] = self._saved
        return False


def _NoMonolith():
    """No running monolith, no env override — the config fallback."""
    return _Monolith(None)


def _fake_monolith(cached=None, local=None, cloud=None, actions=None):
    bc = types.ModuleType("bobert_companion")
    bc._RESOLVED_LOCAL_LLM_MODEL = [cached]
    if local is not None:
        bc.LOCAL_LLM_MODEL = local
    if cloud is not None:
        bc.CLAUDE_MODEL = cloud
    if actions is not None:
        bc.ACTIONS = actions
    return bc


class SelfKnowledgeRoutingTests(unittest.TestCase):
    def setUp(self):
        _core, self.sections = pr.split_pc_control(FULL)
        self.names = [h.strip() for h, _b in self.sections]

    def test_section_is_parsed(self):
        self.assertIn(SECTION, self.names)

    def test_capability_and_comparison_turns_load_the_section(self):
        for q in _SELF_KNOWLEDGE_TURNS:
            with self.subTest(q=q):
                inc, _ = pr.select_sections(q, self.sections)
                self.assertIn(SECTION, inc)

    def test_the_turn_block_carries_the_section_body(self):
        # Selection alone is not delivery: the BODY has to reach the volatile
        # tail the local route actually sends.
        for q in _SELF_KNOWLEDGE_TURNS:
            with self.subTest(q=q):
                self.assertIn("SELF-KNOWLEDGE (", pr.turn_pc_block(q, FULL))

    def test_unrelated_turns_do_not_load_it(self):
        for q in ("turn on the living room lights", "play some jazz",
                  "what time is it", "set a timer for ten minutes",
                  "run a self test", "take a selfie"):
            with self.subTest(q=q):
                inc, _ = pr.select_sections(q, self.sections)
                self.assertNotIn(SECTION, inc)

    def test_how_smart_are_you_no_longer_ships_smart_home_grammar(self):
        # The live turn loaded SMART HOME + SMART HOME DISCOVERY + PER-BRAND
        # LIST on the bare header word "smart": lights grammar for a question
        # about intelligence.
        inc, _ = pr.select_sections("how smart are you", self.sections)
        self.assertEqual([n for n in inc if n.startswith("SMART HOME")], [])

    def test_real_smart_home_turns_keep_their_sections(self):
        for q, want in (("turn on the smart plug", "SMART HOME"),
                        ("discover smart home devices", "SMART HOME DISCOVERY"),
                        ("list smart devices", "SMART HOME — PER-BRAND LIST")):
            with self.subTest(q=q):
                inc, _ = pr.select_sections(q, self.sections)
                self.assertIn(want, inc)


class SelfKnowledgeRuntimeValueTests(unittest.TestCase):
    """The section names the LIVE engines, read at render time."""

    def test_local_model_tag_comes_from_config_at_render_time(self):
        with _NoMonolith(), mock.patch.object(cfg, "LOCAL_LLM_MODEL", _FAKE_LOCAL):
            tail = pr.turn_pc_block("how smart are you", FULL)
            slim = pr.slim_pc_control("how smart are you", FULL)
        self.assertIn(_FAKE_LOCAL, tail)
        self.assertIn(_FAKE_LOCAL, slim)
        # ...and the real shipped tag is NOT baked into the section text.
        self.assertNotIn(cfg._SHIPPED_LOCAL_LLM_MODEL, tail)

    def test_value_change_changes_the_section(self):
        with _NoMonolith():
            with mock.patch.object(cfg, "LOCAL_LLM_MODEL", "unit-test-a:1b"):
                a = prompts.render_self_knowledge_section()
            with mock.patch.object(cfg, "LOCAL_LLM_MODEL", "unit-test-b:2b"):
                b = prompts.render_self_knowledge_section()
        self.assertIn("unit-test-a:1b", a)
        self.assertIn("unit-test-b:2b", b)
        self.assertNotIn("unit-test-a:1b", b)

    def test_monolith_resolver_cache_wins_over_config(self):
        # set_model repoints bobert_companion._RESOLVED_LOCAL_LLM_MODEL; that
        # is the model the next turn really runs on.
        bc = _fake_monolith(cached=_FAKE_CACHED, local=_FAKE_LOCAL)
        with _Monolith(bc):
            tail = pr.turn_pc_block("how smart are you", FULL)
        self.assertIn(_FAKE_CACHED, tail)
        self.assertNotIn(_FAKE_LOCAL, tail)

    def test_env_override_beats_a_cold_cache(self):
        bc = _fake_monolith(cached=None, local=_FAKE_LOCAL)
        with _Monolith(bc), \
                mock.patch.dict(os.environ, {"JARVIS_LOCAL_LLM_MODEL": "unit-test-env:5b"}):
            facts = prompts.self_knowledge_facts()
        self.assertEqual(facts["local_model"], "unit-test-env:5b")

    def test_cloud_model_and_action_count_are_live(self):
        bc = _fake_monolith(cached=_FAKE_CACHED, cloud=_FAKE_CLOUD,
                            actions={f"a{i}": None for i in range(417)})
        with _Monolith(bc):
            body = prompts.render_self_knowledge_section()
        self.assertIn(_FAKE_CLOUD, body)
        self.assertIn("417", body)

    def test_cloud_model_falls_back_to_config(self):
        with _NoMonolith(), mock.patch.object(cfg, "CLAUDE_MODEL", _FAKE_CLOUD):
            self.assertIn(_FAKE_CLOUD, prompts.render_self_knowledge_section())

    def test_deep_model_is_named(self):
        with _NoMonolith():
            facts = prompts.self_knowledge_facts()
            body = prompts.render_self_knowledge_section()
        self.assertTrue(facts["deep_model"])
        self.assertIn(facts["deep_model"], body)

    def test_latency_is_a_dated_constant(self):
        self.assertIsInstance(prompts.SELF_KNOWLEDGE_TURN_LATENCY_S, float)
        self.assertRegex(prompts.SELF_KNOWLEDGE_LATENCY_MEASURED_ON,
                         r"^\d{4}-\d{2}-\d{2}$")
        with _NoMonolith(), \
                mock.patch.object(prompts, "SELF_KNOWLEDGE_TURN_LATENCY_S", 3.3), \
                mock.patch.object(prompts, "SELF_KNOWLEDGE_LATENCY_MEASURED_ON",
                                  "2031-01-02"):
            body = prompts.render_self_knowledge_section()
        self.assertIn("3.3", body)
        self.assertIn("2031-01-02", body)

    def test_render_never_raises_on_a_broken_monolith(self):
        bc = types.ModuleType("bobert_companion")
        bc._RESOLVED_LOCAL_LLM_MODEL = "not-a-list"
        bc.ACTIONS = 42
        with _Monolith(bc):
            body = prompts.render_self_knowledge_section()
        self.assertIn("SELF-KNOWLEDGE (", body)

    def test_stable_block_does_not_carry_the_live_values(self):
        # The live tag rides only in the volatile tail: the cached prefix must
        # stay byte-identical whatever model is loaded.
        with _NoMonolith():
            with mock.patch.object(cfg, "LOCAL_LLM_MODEL", "unit-test-a:1b"):
                pr.turn_pc_block("how smart are you", FULL)
                a = pr.stable_pc_block(FULL)
            with mock.patch.object(cfg, "LOCAL_LLM_MODEL", "unit-test-b:2b"):
                pr.turn_pc_block("how smart are you", FULL)
                b = pr.stable_pc_block(FULL)
        self.assertEqual(a, b)
        self.assertNotIn("unit-test-a:1b", a)


class SelfKnowledgeInstructionTests(unittest.TestCase):
    def setUp(self):
        with _NoMonolith():
            self.body = prompts.render_self_knowledge_section()
        self.low = self.body.lower()

    def test_demands_an_honest_concrete_answer_not_a_deflection(self):
        self.assertIn("honest", self.low)
        self.assertIn("joke", self.low)          # the deflection is named

    def test_never_claims_to_be_opus(self):
        self.assertIn("never claim to be", self.low)
        self.assertIn("opus", self.low)

    def test_offers_the_cloud_for_hard_questions(self):
        self.assertIn("[ACTION: set_brain, cloud]", self.body)

    def test_names_strengths_and_weaknesses(self):
        for word in ("kinect", "robot", "privacy", "reasoning", "slow"):
            with self.subTest(word=word):
                self.assertIn(word, self.low)

    def test_names_the_hardware_split(self):
        self.assertIn("whisper", self.low)
        self.assertIn("kokoro", self.low)

    def test_static_copy_for_the_cloud_route_points_at_current_model(self):
        # The full PC_CONTROL_PROMPT (cloud route) carries the static copy;
        # with no live values attached it must say how to get them.
        bodies = dict(pr.split_pc_control(FULL)[1])
        static = bodies[SECTION]
        self.assertIn("[ACTION: current_model]", static)
        self.assertNotIn(cfg._SHIPPED_LOCAL_LLM_MODEL, static)

    def test_identity_rule_allows_naming_the_engine(self):
        # BASE's "never introduce yourself as Gemma / Claude" rule is about
        # identity; it must not read as "never say what runs you".
        self.assertIn("SELF-KNOWLEDGE", prompts.BASE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
