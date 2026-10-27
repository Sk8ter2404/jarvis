"""PROMPT_SLIM_MODE on the real monolith (2026-10-12).

What slim mode promises, checked on a real build_system_prompt() and real
_call_llm() owner turns (every /api/chat POST is recorded, never sent):

  * PREFIX: two different consecutive owner turns send a byte-identical
    system prompt, so they share a prefix of at least _MIN_SHARED_TOKENS
    (the static head is ~10.5k estimated tokens; nothing learned or per-turn
    sits in it), and a learned fact changes neither the local prompt nor
    asks for a re-prime;
  * BUDGET: the slim local system prompt stays under _SLIM_SYS_BUDGET
    estimated tokens however many facts / topics are learned, and below the
    "off" layout's;
  * NAMEABLE: every action name the full PC_CONTROL_PROMPT + skill examples
    document (self-terminating ones excepted - they route on the owner's own
    words only) is in the slim system prompt, itself or a same-handler alias;
  * SAFETY: the never-guess guard, the standing rules, the PC core preamble
    (action grammar, SAFETY, sign-in pages) and SELF-PRESERVATION are in the
    static head;
  * PER TURN: facts / topics / skill examples ride the turn only when
    relevant, and "what do you know about me" gets every fact;
  * OFF: PROMPT_SLIM_MODE "off" sends the previous layout unchanged.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests._monolith_harness import (  # noqa: E402
    MonolithGlobalsTestCase, requires_monolith,
)

_MIN_SHARED_TOKENS = 9000
_SLIM_SYS_BUDGET = 11500


class _Resp:
    ok = True
    status_code = 200
    text = "body"

    def json(self):
        return {"model": "gemma-test",
                "message": {"role": "assistant", "content": "Very good, sir."},
                "done": True, "eval_count": 3}


def _memory(bc, n_facts=120, n_topics=15):
    mem = bc._empty_memory()
    mem["first_meeting"] = "2026-05-26"
    mem["conversation_count"] = 900
    mem["facts"] = [f"Owner fact {i}: likes widget model {i} in blue"
                    for i in range(n_facts)]
    mem["facts"].append("Owner keeps a pet tortoise named Sheldon")
    mem["projects"] = ["the rover chassis", "home lab rack"]
    mem["sessions"] = [{"date": "2026-10-1%d" % i, "location": "office",
                        "summary": f"Session summary {i}"} for i in range(5)]
    mem["topics"] = [{"date": "2026-10-12", "location": "office",
                      "topic": f"cooking show episode {i}"}
                     for i in range(n_topics)]
    return mem


def _word_in(name, text):
    return re.search(r"(?<![a-z0-9_])" + re.escape(name) + r"(?![a-z0-9_])",
                     text) is not None


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def setUp(self):
        super().setUp()
        bc = self.bc
        import core.config as cfg
        self._p(cfg, "model_route", return_value="local")
        self._p(bc, "_DYNAMIC_LOCAL_PROMPT", True)
        self._p(bc, "_STABLE_LOCAL_PREFIX", True)
        self._p(bc, "PROMPT_SLIM_MODE", "on", create=True)
        self._p(bc, "_RESOLVED_LOCAL_LLM_MODEL", ["gemma-test"])
        self._p(bc, "_ltm_context", return_value="")
        self._p(bc, "_ltm_enqueue")
        self._p(bc, "load_memory", return_value=bc._empty_memory())
        self._p(bc, "save_memory")
        self._p(bc, "_voice_mood_response", None)
        bc._phrase_rotation_last[0] = {}
        self.posted = []
        fake = mock.Mock()

        def _post(url, json=None, timeout=None, **_k):
            self.posted.append(json)
            return _Resp()
        fake.post.side_effect = _post
        self._p(bc, "LOCAL_LLM_FALLBACK", True)
        self._p(bc, "_ollama_alive", return_value=True)
        self._p(bc, "_ollama_has_model", return_value=True)
        self._p(bc, "_get_local_llm_model", return_value="gemma-test")
        self._p(bc, "_next_local_llm_fallback", return_value=None)
        self._p(bc, "requests", fake)
        self._p(bc, "_system_prompt", bc._system_prompt)
        self._p(bc, "_system_prompt_head", [""])
        self._p(bc, "_system_prompt_blocks", [{}])
        self._p(bc, "_SLIM_PC_CACHE", {})
        self._live(_memory(bc))

    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _build(self, mem):
        bc = self.bc
        with mock.patch.object(bc, "_load_chappie_standing_rules",
                               return_value="STANDING RULES BLOCK"), \
                mock.patch.object(bc, "PC_CONTROL_ENABLED", True):
            return bc.build_system_prompt(mem)

    def _live(self, mem):
        self.bc._system_prompt = self._build(mem)
        return self.bc._system_prompt

    def _turn(self, text):
        bc = self.bc
        self.posted.clear()
        bc.conversation_history[:] = []
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc._call_llm(text)
        self.assertEqual(len(self.posted), 1, out.getvalue()[-800:])
        return self.posted[0]

    @staticmethod
    def _sys(payload):
        return payload["messages"][0]["content"]

    @staticmethod
    def _user(payload):
        return payload["messages"][-1]["content"]

    def _tok(self, text):
        return self.bc._prompt_budget.estimate_tokens(text)


class PrefixStabilityTests(_Base):
    def test_two_different_owner_turns_share_the_static_prefix(self):
        a = self._turn("play some music")
        b = self._turn("what's the weather tomorrow")
        self.assertEqual(self._sys(a), self._sys(b))
        self.assertNotEqual(self._user(a), self._user(b))
        shared = os.path.commonprefix([self._sys(a) + self._user(a),
                                       self._sys(b) + self._user(b)])
        self.assertGreaterEqual(self._tok(shared), _MIN_SHARED_TOKENS)
        head = self.bc._local_prompt_head(self._sys(a))
        self.assertTrue(head and self._sys(a).startswith(head))
        # v3 W12 (PROMPT_INDEX_CANON) cuts ~2.3k chars of persona phrase
        # lists out of the head on purpose: the floor drops with it.
        floor = _MIN_SHARED_TOKENS - (
            1000 if getattr(self.bc, "PROMPT_INDEX_CANON", False) else 0)
        self.assertGreaterEqual(self._tok(head), floor)

    def test_learning_a_fact_or_topic_leaves_the_local_prompt_alone(self):
        bc = self.bc
        before = self._sys(self._turn("play some music"))
        mem = _memory(bc)
        mem["facts"].append("Owner just bought a new kayak")
        mem["topics"].append({"date": "2026-10-12", "location": "office",
                              "topic": "a brand new topic"})
        mem["projects"].append("a brand new project")
        self._p(bc, "load_memory", return_value=mem)
        self._p(bc, "_chat_takes_local_branch", return_value=True)
        self._p(bc, "_load_chappie_standing_rules",
                return_value="STANDING RULES BLOCK")
        self._p(bc, "PC_CONTROL_ENABLED", True)
        self.assertTrue(bc._apply_prompt_rebuild())
        self.assertFalse(bc._rebuild_local_changed[0])
        self.assertEqual(self._sys(self._turn("play some music")), before)

    def test_the_reprime_warms_the_same_bytes(self):
        # The sent system prompt is the stable prompt + the local-mode
        # directive; the re-prime builds both with the same code.
        sys_now = self._sys(self._turn("play some music"))
        self.assertTrue(sys_now.startswith(
            self.bc._local_stable_system_prompt()))
        payload = self.bc._build_reprime_payload()
        self.assertIsNotNone(payload)
        self.assertEqual(payload["messages"][0]["content"], sys_now)


class BudgetTests(_Base):
    def test_slim_system_prompt_is_small_and_does_not_grow_with_memory(self):
        bc = self.bc
        slim = bc._local_stable_system_prompt()
        self.assertLessEqual(self._tok(slim), _SLIM_SYS_BUDGET)
        self._live(_memory(bc, n_facts=10, n_topics=2))
        self.assertEqual(bc._local_stable_system_prompt(), slim)
        self._live(_memory(bc))
        self._p(bc, "PROMPT_SLIM_MODE", "off")
        off = bc._local_stable_system_prompt()
        self.assertLess(self._tok(slim), self._tok(off) - 1000)

    def test_a_many_section_turn_fits_well_under_the_window(self):
        p = self._turn("kinect depth, the 3d printer, netflix and hulu, "
                       "browser agent, queue a task")
        msgs = p["messages"]
        est = self.bc._prompt_budget.estimate_chat_tokens(msgs[0]["content"],
                                                          msgs[1:])
        # PROMPT_INDEX_CANON (default ON since 3.0.0, fair prompt gate in
        # PLAN.md) takes the STREAMING SERVICES rows out of the stable head;
        # this turn names netflix and hulu, so they ride it (~1.5k tokens)
        # while every other turn's head is ~600 tokens smaller. Off: < 15000.
        limit = 16000 if getattr(self.bc, "PROMPT_INDEX_CANON", False) else 15000
        self.assertLess(est, limit)
        self.assertLess(est, self.bc._prompt_budget.budget_for(
            self.bc._local_num_ctx("gemma-test"), 500))


class NameableTests(_Base):
    def test_every_documented_action_is_in_the_static_prompt(self):
        bc = self.bc
        from core import prompt_router as pr
        from core.action_risk import SELF_TERMINATING_ACTIONS
        skills = bc._system_prompt_blocks[0].get("skills") or ""
        documented = pr.documented_actions(bc.PC_CONTROL_PROMPT + skills,
                                           set(bc.ACTIONS))
        self.assertGreater(len(documented), 150)
        slim = bc._local_stable_system_prompt()

        def handler(n):
            f = bc.ACTIONS.get(n)
            return id(getattr(f, "__func__", f)) if f is not None else n
        present = {handler(n) for n in bc.ACTIONS if _word_in(n, slim)}
        missing = [a for a in documented
                   if a not in SELF_TERMINATING_ACTIONS
                   and not _word_in(a, slim) and handler(a) not in present]
        if getattr(bc, "PROMPT_INDEX_CANON", False):
            # v3 W12: a hidden duplicate stays registered but is not listed,
            # and a collapsed capability is named by its header (its actions
            # arrive with the routed section). tests/test_prompt_index_canon
            # proves both; here they are simply not "missing".
            from core import action_aliases as _aa
            _core, _secs = pr.split_pc_control(bc.PC_CONTROL_PROMPT)
            _fold = {h.upper() for _l, hs in pr.COLLAPSED_ROWS for h in hs}
            _routed = set(pr.documented_actions(
                " ".join(b for h, b in _secs if h.strip().upper() in _fold),
                set(bc.ACTIONS)))
            missing = [a for a in missing
                       if a not in _aa.HIDDEN_DUPLICATES and a not in _routed]
        self.assertEqual(missing, [])
        for a in SELF_TERMINATING_ACTIONS:
            idx = slim[slim.find("ACTION INDEX"):]
            self.assertFalse(_word_in(a, idx), a)


class SafetyHeadTests(_Base):
    def test_safety_rules_are_in_the_static_head(self):
        bc = self.bc
        from core import prompt_router as pr
        head = bc._local_prompt_head(bc._local_stable_system_prompt())
        self.assertTrue(head)
        core, sections = pr.split_pc_control(bc.PC_CONTROL_PROMPT)
        bodies = dict((h.strip(), b) for h, b in sections)
        for must in (bc._LOCAL_NEVER_GUESS_GUARD, "STANDING RULES BLOCK",
                     core, bodies["SELF-PRESERVATION"],
                     bodies["MULTI-MONITOR APP LAUNCHING"],
                     bc._base_system_prompt()):
            self.assertIn(must, head)


class PerTurnTests(_Base):
    def test_facts_topics_and_skills_ride_only_relevant_turns(self):
        bc = self.bc
        sys_now = bc._local_stable_system_prompt()
        for gone in ("Owner fact 3:", "cooking show episode",
                     "the rover chassis"):
            self.assertNotIn(gone, sys_now)
        self.assertIn("Session summary 4", sys_now)
        self.assertIn("across 900 conversations", sys_now)
        joke = self._user(self._turn("tell me a joke"))
        self.assertNotIn("Owner fact", joke)
        self.assertNotIn("cooking show", joke)
        pet = self._user(self._turn("what is my tortoise called"))
        self.assertIn("Sheldon", pet)
        self.assertNotIn("Owner fact 3:", pet)
        show = self._user(self._turn("did I mention a cooking show"))
        self.assertIn("cooking show episode 14", show)
        rover = self._user(self._turn("how is the rover chassis going"))
        self.assertIn("the rover chassis", rover)

    def test_what_do_you_know_about_me_gets_every_fact(self):
        u = self._user(self._turn("what do you know about me"))
        for i in (0, 60, 119):
            self.assertIn(f"Owner fact {i}:", u)

    def test_skill_examples_ride_the_turn_that_names_them(self):
        bc = self.bc
        self._p(bc, "_SKILL_PROMPT_EXAMPLES",
                {"zz_widget": "WIDGET POLISHER:\n  polish_widget - fire on "
                              "'polish the widget'.\n    Example: 'polish "
                              "the widget' -> [ACTION: polish_widget]"})
        self._live(_memory(bc))
        sys_now = bc._local_stable_system_prompt()
        self.assertIn("WIDGET POLISHER: polish_widget", sys_now)
        self.assertNotIn("fire on 'polish the widget'", sys_now)
        self.assertIn("fire on 'polish the widget'",
                      self._user(self._turn("polish the widget")))
        self.assertNotIn("fire on 'polish the widget'",
                         self._user(self._turn("tell me a joke")))


class ReviewFindingTests(_Base):
    """Review of e9e3e93 (2026-10-12): each test failed before its fix."""

    def test_a_mode_addendum_keeps_the_learned_hints(self):
        # skills/dnd_focus_mode (and night-owl / workshop) append an addendum
        # to the live _system_prompt; the blocks' build is still its prefix.
        bc = self.bc
        bc._system_prompt = bc._system_prompt + "\n\n[Focus mode]\nBe brief."
        self.assertIn("Sheldon",
                      self._user(self._turn("what is my tortoise called")))
        u = self._user(self._turn("what do you know about me"))
        self.assertIn("Owner fact 119:", u)
        # A different build (not a prefix) still sends no stale hints.
        bc._system_prompt = "something else entirely"
        self.assertEqual(bc._slim_turn_hints("my tortoise", ""), "")

    def test_a_huge_fact_list_is_capped_not_dropped(self):
        bc = self.bc
        self._live(_memory(bc, n_facts=300))
        out = io.StringIO()
        self.posted.clear()
        bc.conversation_history[:] = []
        with contextlib.redirect_stdout(out):
            bc._call_llm("what do you know about me")
        self.assertNotIn("learned hints", out.getvalue())
        u = self._user(self.posted[0])
        # Newest facts ride, oldest are cut, and the model is told so.
        self.assertIn("Sheldon", u)
        self.assertIn("Owner fact 299:", u)
        self.assertNotIn("Owner fact 0:", u)
        self.assertIn("older facts not shown", u)
        self.assertLessEqual(
            self._tok(bc._slim_turn_hints("what do you know about me", "")),
            bc._SLIM_FACTS_MAX_TOKENS + 40)
        # 120 facts still all ride.
        self._live(_memory(bc))
        u = self._user(self._turn("what do you know about me"))
        self.assertIn("Owner fact 0:", u)
        self.assertNotIn("older facts not shown", u)

    def test_follow_up_round_keeps_the_learned_hints(self):
        bc = self.bc
        self._turn("what is my tortoise called")
        self.posted.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            bc.get_followup_response([("get_time", "It is noon.")])
        self.assertEqual(len(self.posted), 1)
        self.assertIn("Sheldon", self._user(self.posted[0]))
        self.assertNotIn("Sheldon", self._sys(self.posted[0]))

    def test_skill_with_a_column0_note_routes_whole(self):
        bc = self.bc
        plex = ("PLEX CONTROL:\n  plex_play - fire on 'play the movie'.\n"
                "NOTE: plex_pause needs a title. Fire on 'pause plex'.")
        self._p(bc, "_SKILL_PROMPT_EXAMPLES",
                {"aa_plex": plex, "bb_lamp": "  lamp_on - fire on 'lamp on'."})
        self._live(_memory(bc))
        u = self._user(self._turn("play the movie"))
        self.assertIn("NOTE: plex_pause needs a title", u)
        self.assertNotIn("lamp_on - fire", u)


class OffModeTests(_Base):
    def test_off_sends_the_previous_layout(self):
        bc = self.bc
        self._p(bc, "PROMPT_SLIM_MODE", "off")
        from core import prompt_router as pr
        expected = bc._system_prompt.replace(
            bc.PC_CONTROL_PROMPT,
            bc._LOCAL_NEVER_GUESS_GUARD + "\n"
            + pr.stable_pc_block(bc.PC_CONTROL_PROMPT), 1)
        self.assertTrue(self._sys(self._turn("tell me a joke")).startswith(
            expected))
        self.assertIn("Owner fact 3:", expected)
        self.assertNotIn("ACTION INDEX", expected)


if __name__ == "__main__":
    unittest.main()
