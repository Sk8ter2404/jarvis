"""The follow-up round must still be able to SEE the action reference.

WHAT THIS PROTECTS (2026-09-06 review of the unverified latency work)
--------------------------------------------------------------------
The cache-stable split (``_STABLE_LOCAL_PREFIX``) replaces PC_CONTROL_PROMPT in
the system prompt with ``stable_pc_block()`` — the core grammar plus an INDEX of
bare capability names — and moves the section BODIES onto the user message via
``_with_turn_context``. Three things then conspire against an action chain:

  1. ``get_followup_response`` reuses ``_last_stable_sys_prompt[0]``, so the
     follow-up's system prompt has no PC_CONTROL_PROMPT in it either.
  2. ``_call_local_llm``'s cheatsheet swap is gated on
     ``PC_CONTROL_PROMPT in sys_prompt`` — now False, so the 11.7k-char
     ``_local_cheatsheet()`` never lands.
  3. ``_with_turn_context`` returns a COPY, so the primary turn's bodies are
     not in ``conversation_history`` for the follow-up to inherit.

Result before the fix, MEASURED: registered action names reachable by a
follow-up round dropped from 135/135 (legacy: the full cheatsheet) to 11/135.
"…and then mute the volume" left ``volume_mute`` nowhere in the follow-up's
context; the model emitted ``[ACTION: mute_volume]`` — not a registered action —
while saying it had muted the volume. The token dispatched to nothing and
JARVIS claimed success. Same shape for "…then check for updates", which came
back as the registered-but-wrong-subject ``version_info``.

The fix carries the primary turn's ``turn_pc_block()`` bodies into the
follow-up's turn context (``_last_turn_pc_block``). These tests pin that the
vocabulary a chain's remaining steps need is actually in front of the model.
"""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from tests._monolith_harness import load_monolith, requires_monolith  # noqa: E402


# (utterance, action the SECOND step of the chain needs). Every one of these
# was a live failure before the fix.
_CHAINS = [
    ("check the print status and then mute the volume", "volume_mute"),
    ("tell me the time and then mute the volume", "volume_mute"),
    ("take a screenshot and then check for updates", "check_for_updates"),
    ("what's the weather and then play some music", "play_music"),
]


@requires_monolith
class FollowupSeesActionReferenceTests(unittest.TestCase):
    """Drive the REAL get_followup_response and inspect what it hands the LLM.

    Intercepting ``_local_then_cloud_or_honest`` is the whole point: it is the
    last hop before the model, so whatever reaches it is exactly what the model
    sees. A test that only re-derived the strings would have passed happily
    while the wiring stayed broken.
    """

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()
        from core import prompt_router as pr
        cls.pr = pr
        cls.PC = cls.bc.PC_CONTROL_PROMPT

    def _capture_followup(self, user_text):
        """Simulate one stable-split primary turn for `user_text`, then run the
        real get_followup_response and return (sys_prompt, messages)."""
        bc = self.bc
        seen = {}

        def _fake_llm(sys_prompt, messages, **kw):
            seen["sys"] = sys_prompt
            seen["messages"] = messages
            return "ok"

        stable = (bc._LOCAL_NEVER_GUESS_GUARD + "\n"
                  + self.pr.stable_pc_block(self.PC))
        saved = (bc._last_stable_sys_prompt[0], bc._last_turn_pc_block[0],
                 list(bc.conversation_history),
                 bc._local_then_cloud_or_honest)
        try:
            # What _call_llm stamps on a stable-split local turn.
            bc._last_stable_sys_prompt[0] = stable
            bc._last_turn_pc_block[0] = self.pr.turn_pc_block(user_text, self.PC)
            # conversation_history holds the CLEAN user text — the turn context
            # is deliberately never written back into it.
            bc.conversation_history.clear()
            bc.conversation_history.append({"role": "user", "content": user_text})
            bc.conversation_history.append(
                {"role": "assistant", "content": "[ACTION: get_time]"})
            bc._local_then_cloud_or_honest = _fake_llm
            bc.get_followup_response([
                ("get_time", "3:14 PM"),
                ("_dropped_step", "you promised a second step but emitted no token"),
            ])
        finally:
            (bc._last_stable_sys_prompt[0], bc._last_turn_pc_block[0],
             _hist, bc._local_then_cloud_or_honest) = saved
            bc.conversation_history.clear()
            bc.conversation_history.extend(_hist)

        self.assertIn("messages", seen,
                      "get_followup_response did not reach the local LLM hop — "
                      "model_route('chat') is probably not 'local' in this env")
        return seen["sys"], seen["messages"]

    @staticmethod
    def _everything_the_model_sees(sys_prompt, messages):
        parts = [sys_prompt or ""]
        for m in messages:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                parts.append(m["content"])
        return "\n".join(parts)

    def test_second_step_action_name_is_present(self):
        for user_text, needed in _CHAINS:
            with self.subTest(user_text):
                sys_prompt, messages = self._capture_followup(user_text)
                blob = self._everything_the_model_sees(sys_prompt, messages)
                self.assertIn(needed, blob,
                              f"'{needed}' is not anywhere in the follow-up's "
                              f"context for {user_text!r}; the model has to "
                              f"invent a token for the dropped step")

    def test_bodies_ride_the_user_message_not_the_system_prompt(self):
        """The whole point of the split: the system prompt must stay
        byte-identical across turns, or the KV prefix is thrown away."""
        sys_a, _ = self._capture_followup(_CHAINS[0][0])
        sys_b, _ = self._capture_followup(_CHAINS[2][0])
        self.assertEqual(sys_a, sys_b,
                         "the follow-up's system prompt varies with the user "
                         "text — the cached prefix is dead")

    def test_turn_context_is_framed_as_reference_not_speech(self):
        _sys, messages = self._capture_followup(_CHAINS[0][0])
        last_user = [m for m in messages
                     if isinstance(m, dict) and m.get("role") == "user"][-1]
        self.assertIn(self.bc._TURN_CTX_OPEN, last_user["content"])
        self.assertIn(self.bc._TURN_CTX_CLOSE, last_user["content"])

    def test_history_is_never_mutated_by_the_follow_up(self):
        """_with_turn_context copies; if it ever stopped, the reference block
        would be replayed as history next turn AND land in the cached prefix."""
        bc = self.bc
        before = [dict(m) for m in bc.conversation_history]
        self._capture_followup(_CHAINS[0][0])
        self.assertEqual([dict(m) for m in bc.conversation_history], before)


@requires_monolith
class FollowupActionCoverageTests(unittest.TestCase):
    """The coverage number itself. 11/135 was the regression; the stable block
    alone carries only the always-on section plus bare names."""

    @classmethod
    def setUpClass(cls):
        cls.bc = load_monolith()
        from core import prompt_router as pr
        cls.pr = pr

    def test_carrying_the_bodies_beats_the_index_alone(self):
        bc, pr = self.bc, self.pr
        PC = bc.PC_CONTROL_PROMPT
        actions = sorted(bc.ACTIONS.keys())
        stable = bc._LOCAL_NEVER_GUESS_GUARD + "\n" + pr.stable_pc_block(PC)
        for user_text, _needed in _CHAINS:
            with self.subTest(user_text):
                bodies = pr.turn_pc_block(user_text, PC)
                index_only = sum(1 for a in actions if a in stable)
                with_bodies = sum(1 for a in actions
                                  if a in (stable + "\n" + bodies))
                self.assertGreater(
                    with_bodies, index_only,
                    "carrying turn_pc_block into the follow-up added no action "
                    "vocabulary at all — the split is no longer a superset")

    def test_glance_turn_clears_the_carried_bodies(self):
        """A glance turn bypasses _call_llm entirely. If it clears the system
        prompt slot but not the bodies, the NEXT chain reasons about the
        PREVIOUS turn's sections."""
        import re
        with open(os.path.join(_PROJECT, "bobert_companion.py"),
                  encoding="utf-8") as f:
            src = f.read()
        # Every site that clears the sys-prompt slot must clear the bodies too.
        for m in re.finditer(r"_last_stable_sys_prompt\[0\] = \"\"", src):
            window = src[m.end():m.end() + 200]
            self.assertIn('_last_turn_pc_block[0] = ""', window,
                          "a site clears _last_stable_sys_prompt without "
                          "clearing _last_turn_pc_block — stale bodies leak "
                          "into the next chain")


if __name__ == "__main__":
    unittest.main()
