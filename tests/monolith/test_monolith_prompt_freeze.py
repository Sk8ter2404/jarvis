"""Local prompt-prefix stability (2026-09-29).

WHY
===
A live probe measured a cold local main call at 3.94 s, 3.2 s of it prompt
evaluation (12,393 tokens), against ~1.25 s for a turn on a warm prefix. The
cold turns lined up with the SYSTEM PROMPT CHANGING between turns (the
post-turn rebuild Timer, the phrasebook "last used" hint) and with the history
trim popping one pair off the front every turn once full. Ollama runs one
slot, so any change near the front re-evaluates everything after it.

WHAT THESE TESTS PIN
====================
1. The post-turn rebuild is DEFERRED during an active conversation on the
   local route and applied ONCE after PROMPT_FREEZE_QUIET_S of quiet; outside
   the window (or on the cloud route) it applies immediately.
2. The phrase-rotation hint is absent from the system prompt and present in
   the per-turn tail.
3. The history trims in chunks (cap + 2 -> cap - 6 in one step), and every
   trim site goes through the one helper.
4. The idle re-prime's payload is the next real call's payload minus its last
   user message, its gates hold (and do NOT include _record_speech_active),
   it never triggers itself, and it is single-flight.

Every class here fails on the pre-2026-09-29 code: the helpers do not exist,
the system prompt carries "(last used: ...)", and the trim pops one pair.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import os
import re
import sys
import threading
import time
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


class _Resp:
    def __init__(self, body, ok=True, status_code=200):
        self._body = body
        self.ok = ok
        self.status_code = status_code
        self.text = "body"

    def json(self):
        return self._body


_CHAT_BODY = {"model": "gemma-test",
              "message": {"role": "assistant", "content": "ok"},
              "done": True, "prompt_eval_count": 1234, "eval_count": 1}


def _pairs(n, tag="t"):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"{tag}-u{i}"})
        out.append({"role": "assistant", "content": f"{tag}-a{i}"})
    return out


@requires_monolith
class _Base(MonolithGlobalsTestCase):
    def _p(self, *args, **kwargs):
        patcher = mock.patch.object(*args, **kwargs)
        m = patcher.start()
        self.addCleanup(patcher.stop)
        return m

    def _route(self, route):
        import core.config as cfg
        self._p(cfg, "model_route", return_value=route)

    def _stable_layout(self):
        """A system prompt the cache-stable local split applies to."""
        bc = self.bc
        self._p(bc, "_DYNAMIC_LOCAL_PROMPT", True)
        self._p(bc, "_STABLE_LOCAL_PREFIX", True)
        self.assertTrue(bc.PC_CONTROL_PROMPT)
        self._p(bc, "_system_prompt",
                "BASE IDENTITY\n" + bc.PC_CONTROL_PROMPT
                + "\n\nWhat you know about your owner:\n- likes tea")

    def _local_llm(self):
        """Route every /api/chat POST to a recorder; returns the list of
        payloads."""
        bc = self.bc
        posted = []
        fake_req = mock.Mock()

        def _post(url, json=None, timeout=None, **_k):
            posted.append(json)
            return _Resp(_CHAT_BODY)
        fake_req.post.side_effect = _post
        self._p(bc, "LOCAL_LLM_FALLBACK", True)
        self._p(bc, "_ollama_alive", return_value=True)
        self._p(bc, "_ollama_has_model", return_value=True)
        self._p(bc, "_get_local_llm_model", return_value="gemma-test")
        self._p(bc, "_next_local_llm_fallback", return_value=None)
        self._p(bc, "requests", fake_req)
        return posted

    def _quiet_turn_helpers(self):
        """Keep _call_llm off disk / LTM so only the prompt shaping runs."""
        bc = self.bc
        self._p(bc, "_ltm_context", return_value="")
        self._p(bc, "_ltm_enqueue")
        self._p(bc, "load_memory", return_value=bc._empty_memory())
        self._p(bc, "save_memory")
        self._p(bc, "_voice_mood_response", None)
        bc._phrase_rotation_last[0] = {}


# ════════════════════════════════════════════════════════════════════════════
#  1. Deferred rebuild (mocked clock)
# ════════════════════════════════════════════════════════════════════════════
class DeferredRebuildTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._route("local")
        self._p(bc, "PROMPT_FREEZE_QUIET_S", 30.0)
        self._p(bc, "_system_prompt", "OLD PROMPT")
        self.build = self._p(bc, "build_system_prompt",
                             return_value="NEW PROMPT")
        self._p(bc, "load_memory", return_value={})
        self.waiter = self._p(bc, "_ensure_prompt_rebuild_waiter",
                              return_value=True)
        self.reprime = self._p(bc, "_schedule_local_reprime",
                               return_value=True)
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = False
        bc._prompt_rebuild_pending[0] = False

    def test_deferred_during_conversation_then_applied_once_after_quiet(self):
        bc = self.bc
        bc._note_conversation_activity(now=1000.0)      # a reply
        self.assertEqual(bc._request_prompt_rebuild(now=1002.0), "deferred")
        self.assertEqual(bc._system_prompt, "OLD PROMPT")
        # A second turn's rebuild request coalesces into the same pending one.
        bc._note_conversation_activity(now=1010.0)
        self.assertEqual(bc._request_prompt_rebuild(now=1012.0), "deferred")
        self.build.assert_not_called()
        self.assertTrue(bc._prompt_rebuild_pending[0])
        # Still inside the window of the LATEST activity: nothing applies.
        self.assertFalse(bc._poll_deferred_prompt_rebuild(now=1035.0))
        self.assertEqual(bc._system_prompt, "OLD PROMPT")
        # Window elapsed: applied exactly once.
        self.assertTrue(bc._poll_deferred_prompt_rebuild(now=1040.5))
        self.assertEqual(bc._system_prompt, "NEW PROMPT")
        self.assertEqual(self.build.call_count, 1)
        self.assertFalse(bc._prompt_rebuild_pending[0])
        self.assertFalse(bc._poll_deferred_prompt_rebuild(now=1100.0))
        self.assertEqual(self.build.call_count, 1)
        # The prompt changed and the owner is quiet: one re-prime trigger.
        self.reprime.assert_called_once_with()

    def test_outside_the_window_applies_immediately(self):
        bc = self.bc
        bc._note_conversation_activity(now=1000.0)
        self.assertEqual(bc._request_prompt_rebuild(now=1030.5), "applied")
        self.assertEqual(bc._system_prompt, "NEW PROMPT")
        self.assertFalse(bc._prompt_rebuild_pending[0])
        self.waiter.assert_not_called()
        # An immediate rebuild is not a deferred one: no re-prime.
        self.reprime.assert_not_called()

    def test_no_activity_yet_applies_immediately(self):
        bc = self.bc
        bc._last_convo_activity[0] = 0.0
        self.assertEqual(bc._request_prompt_rebuild(now=5.0), "applied")
        self.assertEqual(bc._system_prompt, "NEW PROMPT")

    def test_turn_or_capture_in_progress_counts_as_active(self):
        bc = self.bc
        bc._last_convo_activity[0] = 0.0
        for flag in (bc._turn_in_progress, bc._utterance_in_progress):
            with self.subTest(flag=flag):
                flag[0] = True
                try:
                    self.assertEqual(bc._request_prompt_rebuild(now=9e9),
                                     "deferred")
                    self.assertFalse(
                        bc._poll_deferred_prompt_rebuild(now=9e9))
                finally:
                    flag[0] = False
        self.assertTrue(bc._poll_deferred_prompt_rebuild(now=9e9))
        self.assertEqual(self.build.call_count, 1)

    def test_cloud_route_is_untouched(self):
        bc = self.bc
        self._route("cloud")
        bc._note_conversation_activity(now=1000.0)
        self.assertEqual(bc._request_prompt_rebuild(now=1001.0), "applied")
        self.assertEqual(bc._system_prompt, "NEW PROMPT")

    def test_unchanged_rebuild_does_not_reprime(self):
        bc = self.bc
        self.build.return_value = "OLD PROMPT"
        bc._note_conversation_activity(now=1000.0)
        bc._request_prompt_rebuild(now=1001.0)
        self.assertTrue(bc._poll_deferred_prompt_rebuild(now=1031.0))
        self.reprime.assert_not_called()

    def test_zero_quiet_window_restores_immediate_rebuilds(self):
        bc = self.bc
        self._p(bc, "PROMPT_FREEZE_QUIET_S", 0.0)
        bc._note_conversation_activity(now=1000.0)
        self.assertEqual(bc._request_prompt_rebuild(now=1000.0), "applied")

    def test_zero_quiet_window_ignores_turn_and_capture_flags(self):
        # 0.0 is documented as "the old behaviour": every rebuild applies at
        # once. The 2 s Timer routinely fires after the owner has started the
        # NEXT capture; that must not defer it (nor ever schedule a re-prime).
        bc = self.bc
        self._p(bc, "PROMPT_FREEZE_QUIET_S", 0.0)
        bc._note_conversation_activity(now=1000.0)
        for flag in (bc._utterance_in_progress, bc._turn_in_progress):
            with self.subTest(flag=flag):
                flag[0] = True
                try:
                    self._p(bc, "_system_prompt", "OLD PROMPT")
                    self.assertEqual(bc._request_prompt_rebuild(now=1000.0),
                                     "applied")
                    self.assertEqual(bc._system_prompt, "NEW PROMPT")
                finally:
                    flag[0] = False
        self.assertFalse(bc._prompt_rebuild_pending[0])
        self.waiter.assert_not_called()
        self.reprime.assert_not_called()


class RealWaiterThreadTests(_Base):
    def test_real_waiter_thread_applies_once(self):
        bc = self.bc
        self._route("local")
        self._p(bc, "_system_prompt", "OLD PROMPT")
        build = self._p(bc, "build_system_prompt", return_value="NEW PROMPT")
        self._p(bc, "load_memory", return_value={})
        reprime = self._p(bc, "_schedule_local_reprime", return_value=True)
        self._p(bc, "_PROMPT_REBUILD_POLL_S", 0.02)
        # A 1 s window against a 0.02 s poll: the two requests and the
        # "still OLD" check below have a full second of slack on a loaded
        # machine (0.15 s could flake).
        self._p(bc, "PROMPT_FREEZE_QUIET_S", 1.0)
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = False
        bc._prompt_rebuild_pending[0] = False
        bc._prompt_rebuild_waiter[0] = None
        bc._note_conversation_activity()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bc._request_prompt_rebuild(), "deferred")
            first = bc._prompt_rebuild_waiter[0]
            self.assertEqual(bc._request_prompt_rebuild(), "deferred")
            self.assertIs(bc._prompt_rebuild_waiter[0], first,
                          "a second request started a second waiter")
            self.assertEqual(bc._system_prompt, "OLD PROMPT")
            deadline = time.monotonic() + 5.0
            while (bc._prompt_rebuild_waiter[0] is not None
                   and time.monotonic() < deadline):
                time.sleep(0.02)
        self.assertIsNone(bc._prompt_rebuild_waiter[0])
        self.assertEqual(bc._system_prompt, "NEW PROMPT")
        self.assertEqual(build.call_count, 1)
        reprime.assert_called_once_with()


class FreezeWiringTests(_Base):
    """main() cannot run in a test; pin the wiring at source level."""

    def test_timer_routes_through_the_deferring_entry_point(self):
        src = inspect.getsource(self.bc.main)
        self.assertIn("threading.Timer(2.0, _request_prompt_rebuild)", src)
        self.assertNotIn('"_system_prompt", build_system_prompt', src)

    def test_owner_turn_and_boundary_are_marked(self):
        src = inspect.getsource(self.bc.main)
        top = src.index("_tt_loop_top(_injected_text)")
        self.assertLess(top, src.index("_note_turn_boundary()", top))
        self.assertLess(src.index("_note_turn_boundary()", top),
                        src.index("if _sleep_mode[0]:", top))
        you = src.index('_tt("mark", "you")')
        self.assertLess(you, src.index("_note_owner_turn()", you))
        self.assertLess(src.index("_note_owner_turn()", you),
                        src.index("reply = _run_llm_dispatch(text"))

    def test_reply_is_activity(self):
        src = inspect.getsource(self.bc._call_llm)
        app = src.index('conversation_history.append({"role": "assistant", '
                        '"content": reply})')
        self.assertLess(app, src.index("_note_conversation_activity()", app))

    def test_owner_turn_never_passes_through_idle(self):
        # The deferred-rebuild waiter polls from another thread. If
        # _note_owner_turn released the capture flag BEFORE marking the turn,
        # the waiter could see "no capture, no turn, activity > window ago"
        # and apply the pending rebuild at the very start of the turn.
        bc = self.bc
        self._p(bc, "PROMPT_FREEZE_QUIET_S", 30.0)
        seen = []

        class _Watched(list):
            def __setitem__(self, i, v):
                super().__setitem__(i, v)
                seen.append(bc._conversation_active())

        self._p(bc, "_utterance_in_progress", _Watched([True]))
        self._p(bc, "_turn_in_progress", _Watched([False]))
        self._p(bc, "_last_convo_activity", _Watched([1.0]))
        bc._note_owner_turn()
        self.assertTrue(seen)
        self.assertTrue(all(seen), f"idle window during the hand-over: {seen}")
        self.assertTrue(bc._turn_in_progress[0])
        self.assertFalse(bc._utterance_in_progress[0])

    def test_owner_turn_and_boundary_semantics(self):
        bc = self.bc
        bc._utterance_in_progress[0] = True
        bc._last_convo_activity[0] = 0.0
        bc._note_owner_turn()
        self.assertTrue(bc._turn_in_progress[0])
        self.assertFalse(bc._utterance_in_progress[0])
        self.assertGreater(bc._last_convo_activity[0], 0.0)
        bc._last_convo_activity[0] = 1.0
        bc._utterance_in_progress[0] = True
        bc._note_turn_boundary()
        self.assertFalse(bc._turn_in_progress[0])
        self.assertFalse(bc._utterance_in_progress[0])
        self.assertGreater(bc._last_convo_activity[0], 1.0,
                           "the end of a turn restarts the quiet window")


# ════════════════════════════════════════════════════════════════════════════
#  2. Phrase rotation lives in the turn tail, never the system prompt
# ════════════════════════════════════════════════════════════════════════════
class PhraseRotationTests(_Base):
    LAST = {"acknowledgements": "Very good, sir.", "minimal": "Working."}

    def test_system_prompt_has_the_phrasebook_but_no_rotation_hint(self):
        bc = self.bc
        mem = bc._empty_memory()
        mem["last_used_phrase_by_intent"] = dict(self.LAST)
        self._p(bc, "_load_chappie_standing_rules", return_value="")
        prompt = bc.build_system_prompt(mem)
        self.assertIn("Canonical JARVIS phrasebook", prompt)
        self.assertNotIn("(last used:", prompt)
        self.assertNotIn("Phrasebook rotation", prompt)

    def _turn(self, text, last):
        bc = self.bc
        self._route("local")
        self._stable_layout()
        posted = self._local_llm()
        self._quiet_turn_helpers()
        bc._phrase_rotation_last[0] = dict(last)
        bc.conversation_history[:] = []
        bc._call_llm(text)
        self.assertEqual(len(posted), 1)
        return posted[0]

    def test_hint_rides_the_turn_tail_on_the_stable_local_layout(self):
        payload = self._turn("what time is it", self.LAST)
        system = payload["messages"][0]["content"]
        last_user = payload["messages"][-1]
        self.assertEqual(last_user["role"], "user")
        self.assertNotIn("Phrasebook rotation", system)
        self.assertNotIn("Very good, sir.' —", system)
        self.assertIn("Phrasebook rotation", last_user["content"])
        self.assertIn("acknowledgements: 'Very good, sir.'",
                      last_user["content"])
        # ...inside the reference block, BEFORE the owner's own words.
        self.assertTrue(last_user["content"].endswith("what time is it"))
        # conversation_history itself stays clean.
        self.assertEqual(self.bc.conversation_history[0]["content"],
                         "what time is it")

    def test_rotation_never_changes_the_system_prompt(self):
        a = self._turn("hello", {"acknowledgements": "Very good, sir."})
        b = self._turn("hello", {"acknowledgements": "Quite.",
                                 "status": "Running the numbers now."})
        self.assertEqual(a["messages"][0]["content"],
                         b["messages"][0]["content"])
        self.assertNotEqual(a["messages"][-1]["content"],
                            b["messages"][-1]["content"])

    def test_legacy_layout_puts_the_hint_in_the_uncached_tail(self):
        bc = self.bc
        self._route("local")
        self._stable_layout()
        self._p(bc, "_STABLE_LOCAL_PREFIX", False)
        posted = self._local_llm()
        self._quiet_turn_helpers()
        bc._phrase_rotation_last[0] = dict(self.LAST)
        bc.conversation_history[:] = []
        bc._call_llm("hello")
        system = posted[0]["messages"][0]["content"]
        self.assertIn("Phrasebook rotation", system)
        # After everything the stable layout would cache (the memory section).
        self.assertGreater(system.index("Phrasebook rotation"),
                           system.index("likes tea"))

    def test_reply_updates_the_in_process_rotation_cache(self):
        bc = self.bc
        self._route("local")
        self._stable_layout()
        posted = self._local_llm()
        self._quiet_turn_helpers()
        body = dict(_CHAT_BODY, message={"role": "assistant",
                                         "content": "Very good, sir."})
        bc.requests.post.side_effect = \
            lambda url, json=None, timeout=None, **k: (posted.append(json),
                                                       _Resp(body))[1]
        bc._call_llm("thanks")
        self.assertEqual(bc._phrase_rotation_last[0].get("acknowledgements"),
                         "Very good, sir.")
        self.assertIn("Very good, sir.", bc._phrase_rotation_hint())


    # ── the follow-up round keeps the hint (it used to ride _system_prompt) ──
    def _followup_setup(self):
        bc = self.bc
        self._quiet_turn_helpers()
        bc._phrase_rotation_last[0] = dict(self.LAST)
        bc.conversation_history[:] = [
            {"role": "user", "content": "what time is it"},
            {"role": "assistant", "content": "[ACTION: get_time] One moment."}]

    def test_followup_round_on_the_stable_local_layout_gets_the_hint(self):
        bc = self.bc
        self._route("local")
        self._stable_layout()
        posted = self._local_llm()
        self._followup_setup()
        stable = bc._local_stable_system_prompt()
        self._p(bc, "_last_stable_sys_prompt", [stable])
        self._p(bc, "_last_turn_pc_block", ["PC-BODIES"])
        bc.get_followup_response([("get_time", "noon")])
        self.assertEqual(len(posted), 1)
        system = posted[0]["messages"][0]["content"]
        tail = posted[0]["messages"][-1]["content"]
        self.assertNotIn("Phrasebook rotation", system)
        self.assertIn("Phrasebook rotation", tail)
        self.assertIn("acknowledgements: 'Very good, sir.'", tail)

    @staticmethod
    def _flatten(system):
        if isinstance(system, str):
            return [(system, False)]
        return [(b.get("text", ""), bool(b.get("cache_control")))
                for b in system]

    def _claude_prompt(self):
        bc = self.bc
        body = "CORE " * 1200
        self._p(bc, "_system_prompt", body + "MEMORY likes tea")
        self._p(bc, "_system_prompt_stable_len", [len(body)])

    def _assert_hint_only_in_the_uncached_tail(self, system):
        self.assertIsInstance(system, list, "cache split lost")
        blocks = self._flatten(system)
        cached = [t for t, c in blocks if c]
        uncached = [t for t, c in blocks if not c]
        self.assertTrue(cached)
        for t in cached:
            self.assertNotIn("Phrasebook rotation", t)
        self.assertEqual(len(uncached), 1)
        self.assertIs(blocks[-1][1], False, "the tail must be the last block")
        self.assertIn("Phrasebook rotation", uncached[0])

    def test_followup_round_on_claude_carries_the_hint_uncached(self):
        bc = self.bc
        self._route("cloud")
        self._claude_prompt()
        self._followup_setup()
        seen = {}

        class FakeClient:
            def complete(self, **kwargs):
                seen.update(kwargs)
                return "It is noon, sir."
        self._p(bc, "AI_BACKEND", "claude")
        self._p(bc, "_llm_client", FakeClient())
        self.assertEqual(bc.get_followup_response([("get_time", "noon")]),
                         "It is noon, sir.")
        self._assert_hint_only_in_the_uncached_tail(seen["system"])

    def test_primary_claude_turn_carries_the_hint_uncached(self):
        bc = self.bc
        self._route("cloud")
        self._claude_prompt()
        self._quiet_turn_helpers()
        bc._phrase_rotation_last[0] = dict(self.LAST)
        bc.conversation_history[:] = []
        seen = {}

        class FakeClient:
            def complete(self, **kwargs):
                seen.update(kwargs)
                return "Good evening, sir."

            def stream_text(self, **kwargs):
                seen.update(kwargs)
                return "Good evening, sir."
        self._p(bc, "AI_BACKEND", "claude")
        self._p(bc, "_llm_client", FakeClient())
        self._p(bc, "_streaming_tts_enabled", return_value=False)
        with mock.patch.dict(sys.modules, {"anthropic": mock.MagicMock()}):
            bc._call_llm("good evening")
        self.assertIn("system", seen, "the Claude branch was not reached")
        self._assert_hint_only_in_the_uncached_tail(seen["system"])


# ════════════════════════════════════════════════════════════════════════════
#  3. Chunked trim
# ════════════════════════════════════════════════════════════════════════════
class ChunkedTrimTests(_Base):
    def test_cap_plus_two_trims_to_cap_minus_six_in_one_step(self):
        bc = self.bc
        cap = bc.MAX_CONVERSATION_HISTORY
        self.assertEqual(cap, 20)
        bc.conversation_history[:] = _pairs((cap + 2) // 2)
        bc._trim_conversation_history()
        self.assertEqual(len(bc.conversation_history), cap - 6)
        self.assertEqual(bc.conversation_history[0]["role"], "user")
        self.assertEqual(bc.conversation_history[0]["content"], "t-u4")
        self.assertEqual(bc.conversation_history[-1]["content"], "t-a10")

    def test_at_or_below_cap_nothing_changes(self):
        bc = self.bc
        cap = bc.MAX_CONVERSATION_HISTORY
        for n in (0, 2, cap - 2, cap):
            with self.subTest(n=n):
                bc.conversation_history[:] = _pairs(n // 2)
                before = list(bc.conversation_history)
                bc._trim_conversation_history()
                self.assertEqual(bc.conversation_history, before)

    def test_front_is_stable_for_the_next_turns(self):
        # The point of chunking: after one trim, the next three turns append
        # without touching the front, so the cached prefix survives them.
        bc = self.bc
        bc.conversation_history[:] = _pairs(11)
        bc._trim_conversation_history()
        front = list(bc.conversation_history)
        for i in range(3):
            bc._append_turn(f"later u{i}", f"later a{i}")
            self.assertEqual(bc.conversation_history[:len(front)], front)

    def test_list_identity_and_pure_helper(self):
        bc = self.bc
        hist = bc.conversation_history
        hist[:] = _pairs(11)
        snapshot = list(hist)
        predicted = bc._trimmed_history(hist)
        self.assertEqual(hist, snapshot, "_trimmed_history mutated its input")
        bc._trim_conversation_history()
        self.assertIs(bc.conversation_history, hist)
        self.assertEqual(hist, predicted)

    def test_leading_assistant_run_still_normalised(self):
        bc = self.bc
        bc.conversation_history[:] = (
            [{"role": "assistant", "content": "orphan"}] * 3 + _pairs(10))
        bc._trim_conversation_history()
        self.assertEqual(bc.conversation_history[0]["role"], "user")
        self.assertLessEqual(len(bc.conversation_history), 14)

    def test_every_trim_site_uses_the_helper(self):
        bc = self.bc
        src = inspect.getsource(bc)
        self.assertNotIn("conversation_history.pop(", src,
                         "a hand-rolled trim loop bypasses the chunked helper")
        self.assertEqual(src.count("del conversation_history["), 1)
        for pat in (r"conversation_history\[\s*:\s*\]\s*=(?!=)",
                    r"conversation_history\s*=\s*conversation_history\s*\["):
            self.assertIsNone(re.search(pat, src),
                              "a hand-rolled slice trim bypasses the helper")
        self.assertIn("del conversation_history[",
                      inspect.getsource(bc._trim_conversation_history))
        for fn in (bc._call_llm, bc._run_llm_dispatch_body, bc._append_turn):
            with self.subTest(fn=fn.__name__):
                self.assertIn("_trim_conversation_history()",
                              inspect.getsource(fn))
        # _call_llm trims both before the request and at the end of the turn.
        self.assertGreaterEqual(
            inspect.getsource(bc._call_llm).count(
                "_trim_conversation_history()"), 2)


# ════════════════════════════════════════════════════════════════════════════
#  4a. Re-prime payload == the next real call's payload minus the last user msg
# ════════════════════════════════════════════════════════════════════════════
class ReprimePayloadTests(_Base):
    def setUp(self):
        super().setUp()
        self._route("local")
        self._stable_layout()
        self.posted = self._local_llm()
        self._quiet_turn_helpers()

    def _assert_prime_matches_next_turn(self, history, text):
        bc = self.bc
        bc.conversation_history[:] = [dict(m) for m in history]
        prime = bc._build_reprime_payload()
        self.assertIsNotNone(prime)
        bc._call_llm(text)
        self.assertEqual(len(self.posted), 1)
        real = self.posted[0]
        self.assertEqual(prime["model"], real["model"])
        self.assertEqual(prime["keep_alive"], real["keep_alive"])
        self.assertEqual(prime.get("think"), real.get("think"))
        self.assertEqual(prime["stream"], real["stream"])
        self.assertEqual(prime["messages"][0]["content"],
                         real["messages"][0]["content"],
                         "system string differs: the prime warms nothing")
        self.assertEqual(prime["messages"], real["messages"][:-1])
        self.assertEqual(real["messages"][-1]["role"], "user")
        self.assertTrue(real["messages"][-1]["content"].endswith(text))
        p_opts = dict(prime["options"])
        r_opts = dict(real["options"])
        self.assertEqual(p_opts.pop("num_predict"), 1)
        r_opts.pop("num_predict")
        self.assertEqual(p_opts, r_opts)
        self.assertIn("num_ctx", p_opts)
        self.assertEqual(bc._local_prefix_hash(prime, drop_last=False),
                         bc._local_prefix_hash(real, drop_last=True))
        return prime, real

    def test_plain_turn(self):
        self._assert_prime_matches_next_turn([], "what time is it")

    def test_follow_up_turn(self):
        hist = _pairs(3) + [{"role": "assistant",
                             "content": "[ACTION: get_time] It is noon."},
                            {"role": "assistant",
                             "content": "Anything else, sir?"}]
        self._assert_prime_matches_next_turn(hist, "and the weather")

    def test_turn_that_triggers_the_chunked_trim(self):
        cap = self.bc.MAX_CONVERSATION_HISTORY
        hist = _pairs(cap // 2)                       # at the cap
        prime, real = self._assert_prime_matches_next_turn(hist, "one more")
        # The real call really did trim (cap + 1 -> <= cap - 6) and the
        # prime predicted it.
        self.assertLessEqual(len(real["messages"]) - 1, cap - 6)
        self.assertNotIn("t-u0", [m.get("content") for m in prime["messages"]])

    def test_web_search_guard_does_not_break_the_match(self):
        hist = [{"role": "user", "content": "search for otters"},
                {"role": "assistant",
                 "content": "[ACTION: web_search, otters] Searching, sir."}]
        self._assert_prime_matches_next_turn(hist, "what did it find")

    def test_no_payload_without_the_stable_layout(self):
        self._p(self.bc, "_STABLE_LOCAL_PREFIX", False)
        self.assertIsNone(self.bc._build_reprime_payload())

    def test_shared_builder_is_used_by_the_real_call(self):
        src = inspect.getsource(self.bc._call_local_llm)
        self.assertIn("_local_chat_prompt(system, messages)", src)
        self.assertIn("_local_chat_payload(model_tag, sys_prompt, messages",
                      src)
        opts_src = inspect.getsource(self.bc._local_chat_payload)
        self.assertIn("chat_options", opts_src)
        self.assertIn("_local_num_ctx(model_tag)", opts_src)
        self.assertIn('"keep_alive": "20m"', opts_src)


# ════════════════════════════════════════════════════════════════════════════
#  4b. Re-prime gates
# ════════════════════════════════════════════════════════════════════════════
class ReprimeGateTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._route("local")
        self._stable_layout()
        self.posted = self._local_llm()
        self._quiet_turn_helpers()
        self._p(bc, "LOCAL_PREFIX_REPRIME", True)
        import core.ollama_opts as oo
        self.resident = self._p(oo, "model_resident", return_value=True)
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = False
        bc.conversation_history[:] = _pairs(2)

    def _once(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            res = self.bc._reprime_once()
        return res, out.getvalue()

    def test_primes_when_every_gate_is_open(self):
        res, log = self._once()
        self.assertEqual(res, "primed")
        self.assertEqual(len(self.posted), 1)
        self.assertEqual(self.posted[0]["options"]["num_predict"], 1)
        self.assertRegex(log, r"\[reprime\] \d+ pe=1234\b")
        self.resident.assert_called_once()
        self.assertEqual(self.resident.call_args.args[0], "gemma-test")
        self.assertTrue(self.bc._reprime_prefix_hash[0])

    def test_skipped_when_model_not_resident(self):
        self.resident.return_value = False
        self.assertEqual(self._once()[0], "not-resident")
        self.assertEqual(self.posted, [])

    def test_skipped_on_the_cloud_route(self):
        self._route("cloud")
        self.assertEqual(self._once()[0], "route")
        self.assertEqual(self.posted, [])
        self.resident.assert_not_called()

    def test_skipped_while_an_utterance_is_in_progress(self):
        self.bc._utterance_in_progress[0] = True
        self.assertEqual(self._once()[0], "utterance")
        self.assertEqual(self.posted, [])

    def test_skipped_while_a_turn_is_in_progress(self):
        self.bc._turn_in_progress[0] = True
        self.assertEqual(self._once()[0], "turn")
        self.assertEqual(self.posted, [])

    def test_skipped_in_game_mode(self):
        fake = types.SimpleNamespace(_st=types.SimpleNamespace(active=True))
        with mock.patch.dict(sys.modules, {"skill_game_mode": fake}):
            self.assertEqual(self._once()[0], "game")
        self.assertEqual(self.posted, [])

    def test_skipped_when_disabled(self):
        self._p(self.bc, "LOCAL_PREFIX_REPRIME", False)
        self.assertEqual(self._once()[0], "disabled")
        self.assertFalse(self.bc._schedule_local_reprime())
        self.assertEqual(self.posted, [])

    def test_not_skipped_merely_because_record_speech_is_listening(self):
        # JARVIS idles inside record_speech(timeout=20) with this flag True;
        # a gate on it would never open.
        bc = self.bc
        saved = bc._record_speech_active[0]
        bc._record_speech_active[0] = True
        try:
            self.assertEqual(self._once()[0], "primed")
        finally:
            bc._record_speech_active[0] = saved
        self.assertEqual(len(self.posted), 1)
        self.assertNotIn("_record_speech_active",
                         inspect.getsource(bc._reprime_skip_reason)
                         .split('"""')[-1])

    def test_skipped_while_a_realtime_voice_session_is_live(self):
        # VOICE_MODE='realtime' never goes through record_speech, so
        # _utterance_in_progress cannot see the owner mid-sentence there.
        bc = self.bc
        self._p(bc, "_realtime_session", [object()])
        self._p(bc, "_realtime_disabled_for_session", [False])
        self.assertEqual(self._once()[0], "utterance")
        self.assertEqual(self.posted, [])

    def test_record_speech_sets_the_utterance_flag_on_trip(self):
        src = inspect.getsource(self.bc.record_speech)
        trip = src.index("recording = True")
        self.assertLess(trip, src.index("_utterance_in_progress[0] = True",
                                        trip))
        self.assertLess(src.index("_utterance_in_progress[0] = True", trip),
                        src.index("record_start_ts = time.time()", trip))


class ReprimeExactResidencyTests(_Base):
    """The re-prime POST names the configured model; with
    OLLAMA_MAX_LOADED_MODELS=1 a POST for a model that is not loaded evicts
    whatever is and cold-loads it. The shipped chat brain and the game-mode
    brain are two tags of ONE family, so a family-level residency match
    would let the prime evict the game brain for a ~15 GB load."""

    BIG = "fam4:26b-a4b-it-qat"

    def setUp(self):
        super().setUp()
        bc = self.bc
        self._route("local")
        self._stable_layout()
        self.posted = self._local_llm()
        self._quiet_turn_helpers()
        self._p(bc, "LOCAL_PREFIX_REPRIME", True)
        self._p(bc, "_get_local_llm_model", return_value=self.BIG)
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = False
        bc.conversation_history[:] = _pairs(2)

    def _ps(self, *names):
        import json as _json
        body = _json.dumps({"models": [{"name": n} for n in names]}).encode()

        class _R:
            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        import urllib.request
        self._p(urllib.request, "urlopen", lambda req, timeout=None: _R())

    def _once(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.bc._reprime_once()

    def test_same_family_other_tag_is_not_resident(self):
        self._ps("fam4:12b")
        self.assertEqual(self._once(), "not-resident")
        self.assertEqual(self.posted, [])

    def test_exact_tag_resident_primes(self):
        self._ps("fam4:12b", self.BIG)
        self.assertEqual(self._once(), "primed")
        self.assertEqual(len(self.posted), 1)
        self.assertEqual(self.posted[0]["model"], self.BIG)


# ════════════════════════════════════════════════════════════════════════════
#  4c. No self-trigger, single-flight, stale detection
# ════════════════════════════════════════════════════════════════════════════
class ReprimeSingleFlightTests(_Base):
    def setUp(self):
        super().setUp()
        bc = self.bc
        self._route("local")
        self._stable_layout()
        self.posted = self._local_llm()
        self._quiet_turn_helpers()
        self._p(bc, "LOCAL_PREFIX_REPRIME", True)
        import core.ollama_opts as oo
        self._p(oo, "model_resident", return_value=True)
        self._p(bc, "_REPRIME_DEBOUNCE_S", 0.05)
        bc._turn_in_progress[0] = False
        bc._utterance_in_progress[0] = False
        bc._reprime_running[0] = False
        bc._reprime_again[0] = False
        bc._prompt_rebuild_pending[0] = False
        bc.conversation_history[:] = _pairs(2)

    def _wait_idle(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while self.bc._reprime_running[0] and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.bc._reprime_running[0], "worker never finished")

    def test_the_reprime_post_never_triggers_anything(self):
        bc = self.bc
        sched = self._p(bc, "_schedule_local_reprime")
        req = self._p(bc, "_request_prompt_rebuild")
        tt = self._p(bc, "_tt")
        bc._last_convo_activity[0] = 123.0
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bc._reprime_once(), "primed")
        sched.assert_not_called()
        req.assert_not_called()
        tt.assert_not_called()
        self.assertEqual(bc._last_convo_activity[0], 123.0)
        self.assertFalse(bc._prompt_rebuild_pending[0])

    def test_two_triggers_one_post(self):
        bc = self.bc
        # The second trigger must land inside the debounce; 0.5 s leaves
        # ample slack on a loaded machine.
        self._p(bc, "_REPRIME_DEBOUNCE_S", 0.5)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(bc._schedule_local_reprime())
            self.assertFalse(bc._schedule_local_reprime())
            self._wait_idle()
        self.assertEqual(len(self.posted), 1)

    def test_trigger_during_the_post_runs_once_more(self):
        bc = self.bc
        calls = []

        def _fake_once():
            calls.append(1)
            if len(calls) == 1:
                # A deferred rebuild landing while the POST is in flight.
                self.assertFalse(bc._schedule_local_reprime())
            return "primed"
        self._p(bc, "_reprime_once", side_effect=_fake_once)
        self.assertTrue(bc._schedule_local_reprime())
        self._wait_idle()
        self.assertEqual(len(calls), 2)
        # ...and a fresh trigger afterwards starts a new worker.
        self.assertTrue(bc._schedule_local_reprime())
        self._wait_idle()
        self.assertEqual(len(calls), 3)

    def test_worker_failure_releases_single_flight(self):
        bc = self.bc
        self._p(bc, "_reprime_once", side_effect=RuntimeError("boom"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(bc._schedule_local_reprime())
            self._wait_idle()

    def test_matching_turn_is_not_stale_changed_prefix_is(self):
        bc = self.bc
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bc._reprime_once(), "primed")
        primed_hash = bc._reprime_prefix_hash[0]
        self.posted.clear()
        # The next owner turn on the same history: warm, not stale.
        bc._turn_in_progress[0] = True
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc._call_llm("next question")
        self.assertNotIn("[reprime] stale", out.getvalue())
        self.assertEqual(bc._reprime_prefix_hash[0], "", "hash not consumed")
        # A prefix that moved after the prime is reported.
        bc._reprime_prefix_hash[0] = primed_hash
        bc.conversation_history.insert(0, {"role": "user", "content": "x"})
        bc.conversation_history.insert(1, {"role": "assistant",
                                           "content": "y"})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc._call_llm("another")
        self.assertIn("[reprime] stale", out.getvalue())

    # ── a turn that starts while the prime is IN FLIGHT ────────────────────
    def _prime_in_flight(self, during):
        """Run _reprime_once on a thread whose POST blocks until `during()`
        (the owner turn) has run. Returns the captured stdout."""
        bc = self.bc
        entered, release = threading.Event(), threading.Event()
        posted = self.posted

        def _post(url, json=None, timeout=None, **_k):
            if (json.get("options") or {}).get("num_predict") == 1:
                entered.set()
                release.wait(10)
            posted.append(json)
            return _Resp(_CHAT_BODY)
        bc.requests.post.side_effect = _post
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            th = threading.Thread(target=bc._reprime_once, daemon=True)
            th.start()
            self.assertTrue(entered.wait(10), "prime never posted")
            try:
                bc._turn_in_progress[0] = True
                during()
            finally:
                release.set()
                th.join(10)
        self.assertFalse(th.is_alive())
        return out.getvalue()

    def test_turn_during_an_in_flight_prime_is_compared_not_stale(self):
        bc = self.bc
        log = self._prime_in_flight(lambda: bc._call_llm("next question"))
        self.assertNotIn("[reprime] stale", log)
        self.assertRegex(log, r"\[reprime\] \d+ pe=")
        # Consumed by THIS turn: the prime finishing afterwards must not
        # re-arm it for the next (different) turn.
        self.assertEqual(bc._reprime_prefix_hash[0], "")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc._call_llm("and another")
        self.assertNotIn("[reprime] stale", out.getvalue())

    def test_turn_during_an_in_flight_prime_with_a_moved_prefix_is_stale(self):
        bc = self.bc

        def _moved_turn():
            bc.conversation_history.insert(0, {"role": "user", "content": "x"})
            bc.conversation_history.insert(1, {"role": "assistant",
                                               "content": "y"})
            bc._call_llm("next question")
        self.assertIn("[reprime] stale", self._prime_in_flight(_moved_turn))
        self.assertEqual(bc._reprime_prefix_hash[0], "")

    def test_failed_prime_leaves_no_hash(self):
        bc = self.bc
        bc.requests.post.side_effect = \
            lambda *a, **k: _Resp({}, ok=False, status_code=500)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bc._reprime_once(), "failed")
        self.assertEqual(bc._reprime_prefix_hash[0], "")

    def test_non_owner_local_calls_during_a_turn_do_not_consume(self):
        # Background _llm_quick / learn calls and the follow-up round all go
        # through _call_local_llm while _turn_in_progress is True; only the
        # owner turn's primary chat call may take the verdict.
        bc = self.bc
        bc._reprime_prefix_hash[0] = "abc"
        bc._turn_in_progress[0] = True
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bc._call_local_llm("background system", [{"role": "user",
                                                     "content": "extract"}])
            done = threading.Event()
            threading.Thread(target=lambda: (bc._call_local_llm(
                "bg", [{"role": "user", "content": "x"}]), done.set()),
                daemon=True).start()
            self.assertTrue(done.wait(10))
        self.assertEqual(bc._reprime_prefix_hash[0], "abc")
        self.assertNotIn("[reprime] stale", out.getvalue())

    def test_background_calls_do_not_consume_the_prime(self):
        bc = self.bc
        bc._reprime_prefix_hash[0] = "abc"
        bc._turn_in_progress[0] = False
        self.assertIsNone(bc._reprime_check_stale({"messages": []}))
        self.assertEqual(bc._reprime_prefix_hash[0], "abc")


class ConfigDefaultsTests(_Base):
    def test_new_settings_reach_the_monolith_with_their_types(self):
        bc = self.bc
        self.assertIsInstance(bc.PROMPT_FREEZE_QUIET_S, float)
        self.assertEqual(bc.PROMPT_FREEZE_QUIET_S, 30.0)
        self.assertIs(bc.LOCAL_PREFIX_REPRIME, True)


if __name__ == "__main__":
    unittest.main()
